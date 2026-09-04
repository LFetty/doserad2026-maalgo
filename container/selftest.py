"""End-to-end self-test for container/inference.py (Proton dose on CT).

Builds a synthetic Grand Challenge `/input` from a real training case, runs the
container inference, then verifies the output contract AND cross-checks dose
placement: the per-beamlet path (reassembled here) must equal the engine's
independent *summed* dose path for the same beams.

    uv run python container/selftest.py --case <DoseRAD2026>/proton/training/1ABB006 --beams 2
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "container"))

from doserad_proton_utils import _resolve_case_files  # noqa: E402
import inference as cinf  # noqa: E402


def build_gc_input(case_dir: Path, input_dir: Path, n_beams: int, cutoff: float) -> dict:
    """Remap a training plan into the GC input contract. Returns expected info."""
    import SimpleITK as sitk

    pid, plan_path, ct_path, _ = _resolve_case_files(case_dir)
    plan = json.loads(plan_path.read_text())
    beams_src = plan["beams"][:n_beams]

    # CT into slot 1 (image_file_idx 0)
    slot_dir = input_dir / "images" / "radiation-dose-calculation-source-ct-image-1"
    slot_dir.mkdir(parents=True, exist_ok=True)
    ct = sitk.ReadImage(str(ct_path))
    sitk.WriteImage(ct, str(slot_dir / "ct.mha"), useCompression=True)

    # Remap beams; assign output_info with a global running idx (all -> slot 0).
    idx = 0
    beams_out = []
    for b in beams_src:
        rays_out = []
        for r in b["rays"]:
            bls = []
            for bl in r["beamlets"]:
                bls.append({
                    "beamlet_uuid": f"{pid}-{idx:06d}",
                    "energy": float(bl["energy"]),
                    "output_info": {
                        "output_file_idx": 0,
                        "idx_in_output": idx,
                        "minimum_cutoff": cutoff,
                    },
                })
                idx += 1
            rays_out.append({"ray_source": r["ray_source"], "ray_target": r["ray_target"], "beamlets": bls})
        beams_out.append({
            "iso_center": plan.get("iso_center", [0.0, 0.0, 0.0]),
            "gantry_angle": float(b["gantry_angle"]),
            "rays": rays_out,
        })

    metadata = [{"image_file_idx": 0, "anatomical_region": "thoracic", "beams": beams_out}]
    (input_dir / "stacked-proton-beam-level-metadata.json").write_text(json.dumps(metadata))
    return {"patient_id": pid, "n_beamlets": idx, "ct_path": ct_path, "beams": beams_out}


def verify(output_dir: Path, expected: dict, model: cinf.DoseModel) -> None:
    import SimpleITK as sitk

    # --- structural checks -------------------------------------------------
    images_dir = output_dir / "images"
    slot_dirs = sorted(images_dir.glob("stacked-radiation-dose-map-*"))
    assert len(slot_dirs) == cinf.N_OUTPUT_SLOTS, f"expected 10 slots, got {len(slot_dirs)}"
    for sd in slot_dirs:
        mhas = list(sd.glob("*.mha"))
        assert len(mhas) == 1, f"{sd.name}: expected 1 .mha, got {len(mhas)}"

    slot1 = sitk.ReadImage(str(next((images_dir / "stacked-radiation-dose-map-1").glob("*.mha"))))
    assert slot1.GetDimension() == 4, f"slot1 must be 4D, got {slot1.GetDimension()}D"
    n_frames = slot1.GetSize()[3]
    assert n_frames == expected["n_beamlets"], f"frame count {n_frames} != beamlets {expected['n_beamlets']}"

    # grid match against the input CT (first 3 dims)
    ct = sitk.ReadImage(str(expected["ct_path"]))
    assert tuple(slot1.GetSize()[:3]) == tuple(ct.GetSize()), "slot1 xy z size != CT size"
    assert np.allclose(slot1.GetSpacing()[:3], ct.GetSpacing(), atol=1e-4), "spacing mismatch"
    assert np.allclose(slot1.GetOrigin()[:3], ct.GetOrigin(), atol=1e-3), "origin mismatch"

    # compression
    import SimpleITK as _s
    r = _s.ImageFileReader(); r.SetFileName(str(next((images_dir / "stacked-radiation-dose-map-1").glob("*.mha"))))
    r.ReadImageInformation()
    # (metadata flag not always exposed; rely on file being written with useCompression)

    frames = sitk.GetArrayFromImage(slot1)  # (T, Z, Y, X)
    reassembled_sum = frames.sum(axis=0)
    print(f"  structural OK: 10 slots, slot1 4D [{n_frames} frames], grid matches CT, "
          f"total dose sum={reassembled_sum.sum():.4g}")

    # --- cross-check: per-beamlet reassembled == engine summed path --------
    ct_hu = sitk.GetArrayFromImage(ct).astype(np.float32)
    from doserad_proton_utils import _origin_zyx, _resolution_zyx, _make_beamlet_batch_sequence
    from pydose_rt.physics.spr import patient_dose_mask, spr_and_mass_density
    from pydose_rt.engine.ion_dose_engine import IonDoseEngine

    origin_zyx = _origin_zyx(ct); resolution_zyx = _resolution_zyx(ct)
    ct_hu_t = torch.from_numpy(ct_hu).to(device=model.device, dtype=model.dtype)
    model.correction_hook.set_hu_volume(ct_hu_t)
    # Must mirror _process_patient: per-axis checkpoint crops, NOT the bev_crop_hw
    # fallback. Deriving from bev_crop_hw here is what let the crop bug stay invisible —
    # the selftest reproduced it, so the two sides agreed while both were wrong.
    crop_h = int(model.correction_hook.bev_crop_h)
    crop_w = int(model.correction_hook.bev_crop_w)
    field = (crop_h * 2, crop_w * 2)
    hu_to_density = model.beam_parameters.get("hu_to_density", {}).get("entries", None)

    plan = {"beams": expected["beams"]}
    engine_sum = np.zeros(ct_hu.shape, dtype=np.float32)
    with torch.inference_mode():
        for bi, bjson in enumerate(plan["beams"]):
            e_ref = float(np.mean([float(bl["energy"]) for ray in bjson["rays"] for bl in ray["beamlets"]]))
            spr, mass = spr_and_mass_density(ct_hu_t, e_ref, hu_to_density)
            seq, ssd = _make_beamlet_batch_sequence(
                plan=plan, beam_parameters=model.beam_parameters, ct_hu=ct_hu, origin_zyx=origin_zyx,
                resolution_zyx=resolution_zyx, beam_index=bi, particles_per_beamlet=cinf.PARTICLES_PER_BEAMLET,
                gantry_offset_deg=0.0, skin_hu_threshold=cinf.SKIN_HU_THRESHOLD, sigma_mode=cinf.SIGMA_MODE,
                bams_to_iso_dist_mm=cinf.BAMS_TO_ISO_DIST_MM, lut=model.lut, device=model.device, dtype=model.dtype)
            engine = IonDoseEngine(
                machine_config=model.machine_config, lut=model.lut, dose_grid_spacing=resolution_zyx,
                dose_grid_shape=ct_hu.shape, beam_template=seq, device=model.device, dtype=model.dtype,
                lateral_model=cinf.LATERAL_MODEL, transport_step_mm=None, sparse_hooks=model.sparse_hooks,
                field_size=field, heterogeneous_mcs=cinf.HETEROGENEOUS_MCS, material_radiation_length=False)
            # Mirror _process_patient: score dose everywhere but air open to the outside.
            # Leaving this on the plain density threshold makes the two sides disagree by
            # the whole internal-cavity dose (71% of peak on a thorax case).
            engine.set_patient_dose_mask(patient_dose_mask(mass))
            beam_sum = engine.compute_dose_bev_lattice_sparse_batch(
                seq, spr.unsqueeze(0), mass_density_image=mass.unsqueeze(0), overwrite=False,
                ssd_mm=ssd, finalize_chunk_size=cinf.DENSE_HOOK_BATCH_ITEMS)[0].detach().cpu().numpy()
            engine_sum += beam_sum.astype(np.float32)
            del engine, seq

    # NOTE: cutoff=0 in the harness so summed paths are directly comparable.
    diff = np.abs(reassembled_sum - engine_sum)
    denom = float(engine_sum.max()) or 1.0
    print(f"  cross-check: max|Δ|={diff.max():.4g} ({100*diff.max()/denom:.3f}% of peak), "
          f"engine_peak={engine_sum.max():.4g}")
    assert diff.max() <= 1e-4 * denom, "per-beamlet reassembly disagrees with summed engine path!"
    print("  ✓ placement verified: per-beamlet reassembly == engine summed path")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", type=Path, required=True, help="DoseRAD training case directory to build the request from")
    ap.add_argument("--beams", type=int, default=2)
    ap.add_argument("--cutoff", type=float, default=0.0)
    ap.add_argument("--workdir", type=Path, default=_REPO / "out" / "container_selftest")
    ap.add_argument("--checkpoint", type=Path, default=cinf.DEFAULT_CHECKPOINT)
    ap.add_argument("--machine-mat", type=Path, default=cinf.DEFAULT_MACHINE_MAT)
    ap.add_argument("--beam-params", type=Path, default=cinf.DEFAULT_BEAM_PARAMS)
    args = ap.parse_args()

    work = args.workdir
    if work.exists():
        shutil.rmtree(work)
    in_dir = work / "input"; out_dir = work / "output"
    in_dir.mkdir(parents=True); out_dir.mkdir(parents=True)

    print(f"[1/3] building GC input from {args.case.name} ({args.beams} beams)")
    expected = build_gc_input(args.case, in_dir, args.beams, args.cutoff)
    print(f"      patient={expected['patient_id']} beamlets={expected['n_beamlets']}")

    print("[2/3] loading model + running container inference")
    model = cinf.DoseModel(checkpoint=args.checkpoint, machine_mat=args.machine_mat, beam_params=args.beam_params).load()
    model.run(in_dir, out_dir)

    print("[3/3] verifying output")
    verify(out_dir, expected, model)
    print("SELFTEST PASSED")


if __name__ == "__main__":
    main()
