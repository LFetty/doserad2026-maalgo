"""Score the CONTAINER inference path against MC ground truth with the official
challenge Level-1 metrics, so it can be compared directly with the eval path.

Why this exists: `container/selftest.py` only cross-checks the container against
*itself* (per-beamlet reassembly vs a summed engine call). It never scores the
container's dose against ground truth, so a systematic container-only error would
pass the selftest while showing up on the leaderboard.

It also forces the chunked path. Training beams hold 30 beamlets and
MAX_BEAMLETS_PER_CHUNK defaults to 32, so local runs never chunk at all -- while a
real plan (2000-3000 beamlets/beam) chunks ~80x per beam. `--chunk` sets the cap.

    uv run python scripts/compare_container_vs_eval.py --beams 2 --chunk 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "container"))

import inference as cinf  # noqa: E402
from selftest import build_gc_input  # noqa: E402


def official_mae(pred: np.ndarray, ref: np.ndarray) -> float:
    """doserad2026_evaluator.metrics_beam.masked_beam_mae."""
    rm = float(ref.max())
    if rm <= 0.0:
        return float("nan")
    m = ref >= 0.1 * rm
    return float(np.abs(pred[m] - ref[m]).mean() / rm)


def official_idd_z(pred: np.ndarray, ref: np.ndarray) -> float:
    """doserad2026_evaluator.metrics_beam.idd_curve_distance, beam_axis=0."""
    ip = pred.sum(axis=(1, 2), dtype=np.float64)
    ir = ref.sum(axis=(1, 2), dtype=np.float64)
    mx = float(ir.max())
    if mx <= 0.0:
        return float("nan")
    return float(np.sqrt(np.mean(((ip - ir) / mx) ** 2)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case-dir", type=Path, required=True, help="DoseRAD training case directory")
    p.add_argument("--beams", type=int, default=2)
    p.add_argument("--chunk", type=int, default=8, help="MAX_BEAMLETS_PER_CHUNK; < beamlets/beam forces chunking")
    p.add_argument("--cutoff", type=float, default=0.0)
    p.add_argument("--work", type=Path, default=Path("out/container_vs_eval"))
    args = p.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    input_dir = args.work / "input"
    if input_dir.exists():
        import shutil
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True)

    expected = build_gc_input(args.case_dir, input_dir, args.beams, args.cutoff)
    pid = expected["patient_id"]
    dose_dir = args.case_dir / "dose"

    model = cinf.init_model()
    # AFTER init_model: DoseModel.load() calls _apply_runtime_config(), which rewrites
    # these module globals from model/config.json. Setting them earlier is silently undone.
    cinf.MAX_BEAMLETS_PER_CHUNK = int(args.chunk)
    print(f"[cfg] MAX_BEAMLETS_PER_CHUNK={cinf.MAX_BEAMLETS_PER_CHUNK} "
          f"DENSE_HOOK_BATCH_ITEMS={cinf.DENSE_HOOK_BATCH_ITEMS} case={pid} beams={args.beams}")
    metadata = json.loads((input_dir / "stacked-proton-beam-level-metadata.json").read_text())
    ct_image = sitk.ReadImage(str(expected["ct_path"]))

    slots: dict = {}
    slot_refs: dict = {}
    model._process_patient(ct_image, metadata[0], slots, slot_refs)

    # GT filenames in the SAME rays->beamlets order the container emits.
    plan = json.loads((args.case_dir / f"{pid}.json").read_text())
    gt_paths = []
    for b in plan["beams"][: args.beams]:
        for r in b["rays"]:
            for bl in r["beamlets"]:
                gt_paths.append(dose_dir / f"Dose_B{b['beam_idx']}_R{r['ray_idx']}_L{bl['beamlet_idx']}.mha")

    payloads = slots.get(0, {})
    full_shape = sitk.GetArrayFromImage(ct_image).shape
    maes, idds, ratios = [], [], []
    for i, gp in enumerate(gt_paths):
        pay = payloads.get(i)
        if pay is None or not gp.exists():
            continue
        ref = sitk.GetArrayFromImage(sitk.ReadImage(str(gp))).astype(np.float64)
        pred = cinf._densify(pay, full_shape).astype(np.float64)
        maes.append(official_mae(pred, ref))
        idds.append(official_idd_z(pred, ref))
        rs = float(ref.sum())
        ratios.append(float(pred.sum()) / rs if rs > 0 else float("nan"))

    print()
    print(f"CONTAINER path, {len(maes)} beamlets, chunk={args.chunk}")
    print(f"  Beam MAE  mean={np.nanmean(maes):.4f}  median={np.nanmedian(maes):.4f}  p95={np.nanpercentile(maes,95):.4f}")
    print(f"  IDD(z)    mean={np.nanmean(idds):.5f}")
    print(f"  int_ratio median={np.nanmedian(ratios):.4f}")
    out = args.work / f"container_{pid}_chunk{args.chunk}.json"
    out.write_text(json.dumps({"mae": maes, "idd": idds, "ratio": ratios}, indent=2))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
