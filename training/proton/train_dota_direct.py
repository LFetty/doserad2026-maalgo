from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from doserad_proton_utils import (  # noqa: E402
    RefBbox,
    _b2nd_path,
    _load_json,
    _make_ray_sequence,
    _read_reference_dose,
    _read_reference_dose_b2nd,
    _reference_paths_for_selection,
    _selected_ray_indices,
    _xyz_to_zyx,
)
from pydose_rt.data.ion_beam import IonSpotBeamSequence  # noqa: E402
from pydose_rt.data.machine_config import MachineConfig  # noqa: E402
from pydose_rt.physics.constants import MEV_CM2_PER_G_TO_GY_MM2  # noqa: E402
from pydose_rt.physics.kernels.ion_lut import PyRadPlanIonLUT  # noqa: E402
from pydose_rt.physics.spr import spr_and_mass_density  # noqa: E402
from training.common.dota_corrector import DoseTransformerDirect  # noqa: E402
from training.proton.train_dense_correction import (  # noqa: E402
    _beam_mean_energy,
    _build_engine,
    _compute_lattice_edep_bev_batch,
    _engine_sample_bev_multi_crop,
    _load_case,
    _material_id_from_hu,
    _reference_patient_tensor,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a low-resolution DoTA-style direct proton beamlet dose model.")
    p.add_argument("--case-dir", type=Path, action="append", default=None)
    p.add_argument("--case-list", type=Path, default=None)
    p.add_argument("--val-case-dir", type=Path, action="append", default=None)
    p.add_argument("--val-case-list", type=Path, default=None)
    p.add_argument("--beam-params-path", type=Path, required=True)
    p.add_argument("--machine-mat", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--val-seed", type=int, default=None)

    p.add_argument("--mode", choices=("gauss_double", "gauss"), default="gauss_double")
    p.add_argument("--transport-step-mm", type=float, default=None)
    p.add_argument("--heterogeneous-mcs", action="store_true")
    p.add_argument("--sigma-mode", choices=("focus", "beam_params", "point_source"), default="beam_params")
    p.add_argument("--bams-to-iso-dist-mm", type=float, default=1000.0)
    p.add_argument("--skin-hu-threshold", type=float, default=-500.0)
    p.add_argument("--particles-per-beamlet", type=float, default=1_000_000.0)
    p.add_argument("--field-size", type=int, nargs=2, default=(96, 96), help="High-resolution BEV crop size before low-res resampling.")

    p.add_argument("--feature-set", choices=("ct", "rich"), default="rich")
    p.add_argument("--prediction-mode", choices=("direct", "correction"), default="direct")
    p.add_argument("--additive-scale-frac", type=float, default=0.25)
    p.add_argument("--lowres-depth", type=int, default=150)
    p.add_argument("--lowres-h", type=int, default=24)
    p.add_argument("--lowres-w", type=int, default=24)
    p.add_argument("--latent-h", type=int, default=6)
    p.add_argument("--latent-w", type=int, default=6)
    p.add_argument("--latent-channels", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--compile-model", action="store_true")
    p.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="reduce-overhead")

    p.add_argument("--epochs", type=int, default=56)
    p.add_argument("--steps-per-epoch", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--beam-sampling", choices=("random", "full"), default="random")
    p.add_argument("--beamlet-chunk-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--poly-power", type=float, default=0.9)
    p.add_argument("--val-steps-per-epoch", type=int, default=8)
    p.add_argument("--validate-every-epochs", type=int, default=1)
    p.add_argument("--validate-before-training", action="store_true")
    p.add_argument("--reference-io-workers", type=int, default=8)
    p.add_argument("--checkpoint-every-steps", type=int, default=0)
    p.add_argument("--wandb-project", type=str, default="pydose-dota-direct")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def _read_case_list(path: Path) -> list[Path]:
    return [Path(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _model_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": {
            "depth": int(args.lowres_depth),
            "height": int(args.lowres_h),
            "width": int(args.lowres_w),
            "latent_hw": [int(args.latent_h), int(args.latent_w)],
            "latent_channels": int(args.latent_channels),
            "num_heads": int(args.num_heads),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
        },
        "feature_set": str(args.feature_set),
        "prediction_mode": str(args.prediction_mode),
    }


def _read_reference(ref_path: Path) -> tuple[np.ndarray, RefBbox | None]:
    if _b2nd_path(ref_path).exists():
        return _read_reference_dose_b2nd(ref_path)
    return _read_reference_dose(ref_path), None


def _full_units(cases: list[Any], chunk_size: int) -> list[tuple[int, int, list[tuple[int, int]]]]:
    units = []
    chunk_size = max(1, int(chunk_size))
    for case_i, (_pid, plan, *_rest) in enumerate(cases):
        for bi, beam_json in enumerate(plan["beams"]):
            ray_idxs = _selected_ray_indices(beam_json, None, bi)
            items = [(ri, li) for ri in ray_idxs for li in range(len(beam_json["rays"][ri]["beamlets"]))]
            for start in range(0, len(items), chunk_size):
                units.append((case_i, bi, items[start:start + chunk_size]))
    return units


def _random_unit(cases: list[Any], rng: random.Random, chunk_size: int) -> tuple[int, int, list[tuple[int, int]]]:
    case_i = rng.randrange(len(cases))
    _pid, plan, *_rest = cases[case_i]
    bi = rng.randrange(len(plan["beams"]))
    beam_json = plan["beams"][bi]
    ray_idxs = _selected_ray_indices(beam_json, None, bi)
    items = [(ri, li) for ri in ray_idxs for li in range(len(beam_json["rays"][ri]["beamlets"]))]
    if not items:
        return _random_unit(cases, rng, chunk_size)
    chunk_size = min(max(1, int(chunk_size)), len(items))
    selected = rng.sample(items, k=chunk_size)
    return case_i, bi, selected


def _resize_volume(volume: torch.Tensor, shape: tuple[int, int, int], mode: str = "trilinear") -> torch.Tensor:
    kwargs = {} if mode == "nearest" else {"align_corners": False}
    return F.interpolate(volume.unsqueeze(1), size=shape, mode=mode, **kwargs).squeeze(1)


def _lowres_features(
    hu_bev: torch.Tensor,
    spr_bev: torch.Tensor,
    weq_bev: torch.Tensor,
    material_bev: torch.Tensor,
    dose_pb_bev: torch.Tensor | None,
    args: argparse.Namespace,
) -> torch.Tensor:
    shape = (int(args.lowres_depth), int(args.lowres_h), int(args.lowres_w))
    hu_lr = _resize_volume((hu_bev / 1000.0).clamp(-1.0, 3.0), shape)
    if args.feature_set == "ct":
        channels = [hu_lr]
        if args.prediction_mode == "correction":
            if dose_pb_bev is None:
                raise ValueError("correction mode requires dose_pb_bev")
            pb_lr = _resize_volume(dose_pb_bev, shape)
            pb_scale = pb_lr.detach().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
            channels.append(pb_lr / pb_scale)
        return torch.stack(channels, dim=1)
    spr_lr = _resize_volume(spr_bev - 1.0, shape)
    weq_lr = _resize_volume(weq_bev / max(float(shape[0]), 1.0), shape)
    mat_lr = _resize_volume(material_bev.float() / 85.0, shape, mode="nearest")
    b, d, h, w = hu_lr.shape
    depth = torch.linspace(0.0, 1.0, d, device=hu_lr.device, dtype=hu_lr.dtype).view(1, d, 1, 1).expand(b, d, h, w)
    hh = torch.linspace(-1.0, 1.0, h, device=hu_lr.device, dtype=hu_lr.dtype).view(1, 1, h, 1).expand(b, d, h, w)
    ww = torch.linspace(-1.0, 1.0, w, device=hu_lr.device, dtype=hu_lr.dtype).view(1, 1, 1, w).expand(b, d, h, w)
    channels = [hu_lr, spr_lr, weq_lr, depth, hh, ww, mat_lr]
    if args.prediction_mode == "correction":
        if dose_pb_bev is None:
            raise ValueError("correction mode requires dose_pb_bev")
        pb_lr = _resize_volume(dose_pb_bev, shape)
        pb_scale = pb_lr.detach().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
        channels.append(pb_lr / pb_scale)
    return torch.stack(channels, dim=1)


def _lowres_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    peak = target.detach().amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-8)
    mask = (target.detach() > 0.001 * peak).to(dtype=pred.dtype)
    l1 = ((pred - target).abs() / peak * mask).sum() / mask.sum().clamp_min(1.0)
    high10 = (target.detach() > 0.10 * peak).to(dtype=pred.dtype)
    high10_l1 = ((pred - target).abs() / peak * high10).sum() / high10.sum().clamp_min(1.0)
    return l1, {"mae_pct": float((100.0 * l1).detach()), "mae_high10_pct": float((100.0 * high10_l1).detach())}


def main() -> None:
    args = _parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    amp_enabled = bool(args.amp and device.type == "cuda" and dtype == torch.float32)
    print(f"device={device} dtype={dtype} amp={amp_enabled}")

    beam_parameters = _load_json(args.beam_params_path.resolve())
    hu_to_density = beam_parameters["hu_to_density"]["entries"]
    lut = PyRadPlanIonLUT(args.machine_mat)
    machine_config = MachineConfig(
        tpr_20_10=0.7,
        number_of_leaf_pairs=40,
        fit_air_offset_mm=0.0,
        bams_to_iso_dist_mm=float(args.bams_to_iso_dist_mm),
    )

    input_channels = 1 if args.feature_set == "ct" else 7
    if args.prediction_mode == "correction":
        input_channels += 1
    cfg = _model_config(args)
    model = DoseTransformerDirect.from_config(input_channels, cfg, available_energies=lut.available_energies)
    model = model.to(device=device, dtype=dtype)
    if args.compile_model:
        compile_mode = None if args.compile_mode == "default" else str(args.compile_mode)
        print(f"compiling model with torch.compile(mode={args.compile_mode})")
        model = torch.compile(model, mode=compile_mode)
    print(f"model params: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    full_units: list[tuple[int, int, list[tuple[int, int]]]] = []
    all_case_dirs = list(args.case_dir or [])
    if args.case_list is not None:
        all_case_dirs.extend(_read_case_list(args.case_list))
    val_dirs = list(args.val_case_dir or [])
    if args.val_case_list is not None:
        val_dirs.extend(_read_case_list(args.val_case_list))
    explicit_val = {path.resolve() for path in val_dirs}
    train_cases = [_load_case(path.resolve()) for path in all_case_dirs if path.resolve() not in explicit_val]
    val_cases = [_load_case(path.resolve()) for path in sorted(explicit_val)]
    print(f"cases train={len(train_cases)} val={len(val_cases)}")
    if not train_cases:
        raise RuntimeError("No training cases")
    if args.beam_sampling == "full":
        full_units = _full_units(train_cases, int(args.beamlet_chunk_size))
        steps_per_epoch = len(full_units)
    else:
        steps_per_epoch = max(1, int(args.steps_per_epoch))
    print(f"beam sampling={args.beam_sampling} effective_steps_per_epoch={steps_per_epoch}")
    total_iters = int(args.max_steps) if args.max_steps is not None else int(args.epochs) * steps_per_epoch
    scheduler = (
        torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_iters, power=float(args.poly_power))
        if float(args.poly_power) > 0.0
        else None
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    use_wandb = not bool(args.no_wandb)
    wandb = None
    if use_wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or args.output_dir.name,
            dir=str(args.output_dir),
            config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()} | {"model_cfg": cfg},
        )

    ref_pool = ThreadPoolExecutor(max_workers=max(1, int(args.reference_io_workers)))
    ref_cache: collections.OrderedDict[Path, tuple[np.ndarray, RefBbox | None]] = collections.OrderedDict()
    hu_cache: dict[str, torch.Tensor] = {}
    mat_cache: dict[str, torch.Tensor] = {}
    history: list[dict[str, Any]] = []
    global_step = 0

    def get_ref(ref_path: Path) -> tuple[np.ndarray, RefBbox | None]:
        if ref_path in ref_cache:
            ref_cache.move_to_end(ref_path)
            return ref_cache[ref_path]
        value = _read_reference(ref_path)
        ref_cache[ref_path] = value
        if len(ref_cache) > 256:
            ref_cache.popitem(last=False)
        return value

    def run_unit(case: Any, case_i: int, bi: int, items: list[tuple[int, int]], epoch: int, training: bool) -> dict[str, Any]:
        patient_id, plan, dose_dir, ct_hu, origin_zyx, res_zyx = case
        beam_json = plan["beams"][bi]
        e_ref = _beam_mean_energy(plan, bi)
        if patient_id not in hu_cache:
            hu_cache[patient_id] = torch.from_numpy(ct_hu).to(device=device, dtype=dtype)
            mat_cache[patient_id] = _material_id_from_hu(hu_cache[patient_id])
        hu_t = hu_cache[patient_id]
        spr_vol, _mass_vol = spr_and_mass_density(hu_t, e_ref, hu_to_density)
        mat_patient = mat_cache[patient_id]

        seqs = []
        ssd_values = []
        ref_paths = []
        crop_centers = []
        energies = []
        for ri, li in items:
            ray_json = beam_json["rays"][ri]
            seq, ssd_mm = _make_ray_sequence(
                plan=plan,
                beam_parameters=beam_parameters,
                ct_hu=ct_hu,
                origin_zyx=origin_zyx,
                resolution_zyx=res_zyx,
                beam_index=bi,
                ray_index=ri,
                beamlet_index=li,
                particles_per_beamlet=float(args.particles_per_beamlet),
                gantry_offset_deg=0.0,
                skin_hu_threshold=float(args.skin_hu_threshold),
                sigma_mode=args.sigma_mode,
                bams_to_iso_dist_mm=float(args.bams_to_iso_dist_mm),
                lut=lut,
                device=device,
                dtype=dtype,
            )
            paths = _reference_paths_for_selection(dose_dir, beam_json, bi, [ri], li)
            if len(paths) != 1:
                raise RuntimeError(f"Expected one MC reference for {patient_id} beam={bi} ray={ri} beamlet={li}")
            seqs.append(seq)
            ssd_values.append(ssd_mm)
            ref_paths.append(paths[0])
            crop_centers.append(tuple((_xyz_to_zyx(ray_json["ray_target"]) - origin_zyx).tolist()))
            energies.append(float(ray_json["beamlets"][li]["energy"]))

        combined_seq = IonSpotBeamSequence.from_beams([seq[0] for seq in seqs])
        engine = _build_engine(args, lut, machine_config, res_zyx, ct_hu.shape, combined_seq, device, dtype)
        ssd_batch = [float(v) for v in ssd_values if v is not None]
        with torch.no_grad():
            resolved_offset = engine._resolve_rad_depth_offset(
                None,
                ssd_batch if len(ssd_batch) == len(ssd_values) else None,
                engine.number_of_beams,
            )
        crop_centers_hw = [(float(center[0]) / float(res_zyx[0]), float(center[2]) / float(res_zyx[2])) for center in crop_centers]
        crops = [
            engine._build_dense_bev_crop(center_h=h, center_w=w, size_h=int(args.field_size[0]), size_w=int(args.field_size[1]))
            for h, w in crop_centers_hw
        ]
        spr_bev, weq_bev_pb = engine._dense_forward_bev_multi_crop(spr_vol.unsqueeze(0), crops)
        weq_bev = weq_bev_pb
        if resolved_offset is not None:
            weq_bev = weq_bev + resolved_offset.view(-1, 1, 1, 1)
        hu_bev = _engine_sample_bev_multi_crop(engine, hu_t.unsqueeze(0), crops, mode="bilinear")
        mat_bev = _engine_sample_bev_multi_crop(engine, mat_patient.unsqueeze(0).float(), crops, mode="nearest")
        dose_pb_bev = None
        if args.prediction_mode == "correction":
            feature_centers_hw = [
                (center_h - float(crop["target_h_start"]), center_w - float(crop["target_w_start"]))
                for (center_h, center_w), crop in zip(crop_centers_hw, crops, strict=True)
            ]
            with torch.no_grad():
                edep_bev = _compute_lattice_edep_bev_batch(
                    engine,
                    combined_seq,
                    weq_bev_pb,
                    feature_centers_hw,
                    resolved_offset,
                )
            voxel_volume_mm3 = float(res_zyx[0]) * float(res_zyx[1]) * float(res_zyx[2])
            transport_step = float(engine.transport_step_mm or res_zyx[1])
            lateral_area_mm2 = voxel_volume_mm3 / transport_step
            dose_pb_bev = edep_bev * (MEV_CM2_PER_G_TO_GY_MM2 / lateral_area_mm2)
        features = _lowres_features(hu_bev, spr_bev, weq_bev, mat_bev, dose_pb_bev, args)

        targets = []
        for ref_path, crop, idx in zip(ref_paths, crops, range(len(ref_paths)), strict=True):
            ref_arr, ref_bbox = get_ref(ref_path)
            ref_patient = _reference_patient_tensor(ref_arr, ref_bbox, ct_hu.shape, device=device, dtype=dtype)
            ref_bev = _engine_sample_bev_multi_crop(engine, ref_patient.unsqueeze(0), [crop], beam_indices=[idx])
            ref_lr = _resize_volume(ref_bev, (int(args.lowres_depth), int(args.lowres_h), int(args.lowres_w)))
            targets.append(ref_lr.unsqueeze(1))
        target = torch.cat(targets, dim=0)
        dose_pb_lr = (
            _resize_volume(dose_pb_bev, (int(args.lowres_depth), int(args.lowres_h), int(args.lowres_w))).unsqueeze(1)
            if dose_pb_bev is not None
            else None
        )
        energy_t = torch.tensor(energies, device=device, dtype=dtype)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(features, energy=energy_t)
            if args.prediction_mode == "correction":
                if dose_pb_lr is None:
                    raise RuntimeError("correction mode requires a PB dose baseline")
                scale = dose_pb_lr.detach().amax(dim=(2, 3, 4), keepdim=True).clamp_min(torch.finfo(dose_pb_lr.dtype).tiny)
                pred = (dose_pb_lr + outputs["residual"] * scale * float(args.additive_scale_frac)).clamp_min(0.0)
            else:
                pred = outputs["dose_hat"]
            loss, terms = _lowres_loss(pred, target)
        if training:
            scaler.scale(loss).backward()
        return {
            "epoch": int(epoch),
            "patient": patient_id,
            "beam": int(bi),
            "n_beamlets": len(items),
            "loss": float(loss.detach()),
            **terms,
        }

    def save_checkpoint(epoch: int) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state": scaler.state_dict(),
                "config": cfg,
                "epoch": int(epoch),
                "step": int(global_step),
                "args": vars(args) | {"output_dir": str(args.output_dir), "machine_mat": str(args.machine_mat)},
            },
            args.output_dir / "latest.pt",
        )
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    def validate(epoch: int) -> None:
        if not val_cases or int(args.val_steps_per_epoch) <= 0:
            return
        model.eval()
        rng = random.Random(int(args.val_seed if args.val_seed is not None else args.seed) + 100_000 + int(epoch))
        records = []
        with torch.no_grad():
            for _ in range(int(args.val_steps_per_epoch)):
                case_i, bi, items = _random_unit(val_cases, rng, int(args.beamlet_chunk_size))
                records.append(run_unit(val_cases[case_i], case_i, bi, items, epoch, training=False))
        rec = {
            "epoch": int(epoch),
            "loss": float(np.mean([r["loss"] for r in records])),
            "mae_pct": float(np.mean([r["mae_pct"] for r in records])),
            "mae_high10_pct": float(np.mean([r["mae_high10_pct"] for r in records])),
        }
        print(
            f"[val e{epoch} s{global_step}] loss={rec['loss']:.6g} "
            f"mae%={rec['mae_pct']:.3g} mae_ref>10%={rec['mae_high10_pct']:.3g}"
        )
        if use_wandb:
            wandb.log({"val/loss": rec["loss"], "val/mae_pct": rec["mae_pct"], "val/mae_high10_pct": rec["mae_high10_pct"]}, step=global_step)

    if args.validate_before_training:
        validate(0)
        save_checkpoint(0)

    train_rng = random.Random(int(args.seed))
    for epoch in range(1, int(args.epochs) + 1):
        if args.beam_sampling == "full":
            epoch_units = list(full_units)
            random.Random(int(args.seed) + epoch).shuffle(epoch_units)
        else:
            epoch_units = [_random_unit(train_cases, train_rng, int(args.beamlet_chunk_size)) for _ in range(steps_per_epoch)]
        for case_i, bi, items in epoch_units:
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            model.train()
            rec = run_unit(train_cases[case_i], case_i, bi, items, epoch, training=True)
            if float(args.grad_clip) > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            global_step += 1
            rec["step"] = int(global_step)
            rec["compute_s"] = round(time.perf_counter() - t0, 3)
            history.append(rec)
            print(
                f"[e{epoch} s{global_step}] {rec['patient']} beam{rec['beam']} "
                f"loss={rec['loss']:.6g} mae%={rec['mae_pct']:.3g} "
                f"mae_ref>10%={rec['mae_high10_pct']:.3g} "
                f"({rec['n_beamlets']} bl, {rec['compute_s']}s)"
            )
            if use_wandb:
                wandb.log({"train/loss": rec["loss"], "train/mae_pct": rec["mae_pct"], "train/mae_high10_pct": rec["mae_high10_pct"], "train/compute_s": rec["compute_s"]}, step=global_step)
            if args.checkpoint_every_steps and global_step % int(args.checkpoint_every_steps) == 0:
                save_checkpoint(epoch)
            if args.max_steps is not None and global_step >= int(args.max_steps):
                save_checkpoint(epoch)
                ref_pool.shutdown(wait=True)
                if use_wandb:
                    wandb.finish()
                return
        if int(args.validate_every_epochs) > 0 and epoch % int(args.validate_every_epochs) == 0:
            validate(epoch)
        save_checkpoint(epoch)

    ref_pool.shutdown(wait=True)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
