"""Train a dense BEV correction model for the proton pencil-beam engine.

For each selected DoseRAD beamlet the pipeline is:
  1. Rotate patient SPR and material-ID volumes into BEV using the dense engine grids.
  2. Compute base pencil-beam energy deposition in BEV using the dense
     ``IonDoseEngine`` (frozen).
  3. Build a feature tensor in BEV and apply ``FanGridConvCorrector``.
  4. Compare against the MC reference in patient space, or optionally sample the
     reference into BEV and compute the training loss there.

Lateral BEV crops around the beam axis keep memory bounded.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from doserad_proton_utils import (  # noqa: E402
    DEFAULT_MC_FIT_MAT,
    RefBbox,
    _b2nd_path,
    _load_json,
    _make_ray_sequence,
    _origin_zyx,
    _plot_total_comparison,
    _read_reference_dose,
    _read_reference_dose_b2nd,
    _reference_paths_for_selection,
    _resolution_zyx,
    _resolve_case_files,
    _selected_ray_indices,
    _xyz_to_zyx,
)
from pydose_rt.data.machine_config import MachineConfig  # noqa: E402
from pydose_rt.engine.ion_dose_engine import IonDoseEngine  # noqa: E402
from pydose_rt.physics.constants import MEV_CM2_PER_G_TO_GY_MM2  # noqa: E402
from pydose_rt.physics.kernels.ion_lut import PyRadPlanIonLUT  # noqa: E402
from pydose_rt.physics.materials import GEANT4_HU_BOUNDS, GEANT4_NUM_MATERIALS  # noqa: E402
from pydose_rt.physics.spr import patient_dose_mask, spr_and_mass_density  # noqa: E402
from training.common.checkpoints import load_model_state_dict, portable_model_state_dict  # noqa: E402
from training.proton.hooks import (  # noqa: E402
    BEV_FEATURE_CHANNELS,
    _FEATURE_H_OFFSET_CH,  # noqa: F401  -- re-exported for tests/unittests
    _FEATURE_W_OFFSET_CH,  # noqa: F401  -- re-exported for tests/unittests
    _d4_apply,
    _d4_apply_features,
    _d4_inverse,
    _lateral_weq_gradient,
)
from training.common.ray_sequence_corrector import FanGridConvCorrector  # noqa: E402
from training.common.repvgg_unet_corrector import RepVGGUNetCorrector  # noqa: E402
from training.common.separable_fan_grid_corrector import SeparableFanGridConvCorrector  # noqa: E402


LOSS_KEYS = (
    "loss",
    "dose",
    "dose_raw",
    "energy",
    "mae_pct",
    "mae_high10_pct",
    "idd_z",
    "integral_ratio",
)

# D4 augmentation for BEV spatial dims (H, W) — dims 3, 4 of [B, C, D, H, W]
_D4_SYMMETRIES: tuple[tuple[bool, int], ...] = (
    (False, 0),
    (False, 1),
    (False, 2),
    (False, 3),
    (True, 0),
    (True, 1),
    (True, 2),
    (True, 3),
)


def _material_id_from_hu(hu: torch.Tensor) -> torch.Tensor:
    bounds = torch.as_tensor(GEANT4_HU_BOUNDS, device=hu.device, dtype=hu.dtype)
    clipped = hu.clamp(float(GEANT4_HU_BOUNDS[0]), float(GEANT4_HU_BOUNDS[-1]))
    ids = torch.bucketize(clipped.contiguous(), bounds[1:].contiguous(), right=True)
    return torch.clamp(ids, 0, GEANT4_NUM_MATERIALS - 1).long()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-dir", type=Path, action="append", default=None)
    p.add_argument("--case-list", type=Path, default=None)
    p.add_argument("--beam-params-path", type=Path, default=ROOT / "example_data" / "beam_parameters.json")
    p.add_argument("--machine-mat", type=Path, default=DEFAULT_MC_FIT_MAT)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")

    # Engine settings
    p.add_argument("--mode", choices=("gauss_double", "gauss"), default="gauss_double")
    p.add_argument("--transport-step-mm", type=float, default=None)
    p.add_argument(
        "--heterogeneous-mcs",
        action="store_true",
        help="Enable Fermi-Eyges/Kanematsu heterogeneity-aware MCS lateral scattering.",
    )
    p.add_argument("--sigma-mode", choices=("focus", "beam_params", "point_source"), default="beam_params")
    p.add_argument("--bams-to-iso-dist-mm", type=float, default=1000.0)
    p.add_argument("--skin-hu-threshold", type=float, default=-500.0)
    p.add_argument("--particles-per-beamlet", type=float, default=1_000_000.0)
    p.add_argument(
        "--field-size",
        type=int,
        nargs=2,
        default=None,
        help="Dense PB BEV field size. Defaults to 2 * --bev-crop-hw in each lateral dimension.",
    )

    # BEV crop
    p.add_argument("--bev-crop-hw", type=int, default=64, help="Lateral crop half-width in voxels around beam axis in BEV")
    p.add_argument("--bev-crop-h", type=int, default=None, help="Optional BEV crop half-width in H voxels")
    p.add_argument("--bev-crop-w", type=int, default=None, help="Optional BEV crop half-width in W voxels")
    p.add_argument(
        "--bev-feature-set",
        choices=tuple(sorted(BEV_FEATURE_CHANNELS)),
        default="v1",
        help="BEV input channel stack. 'v1' (8ch) is the historical layout every existing "
        "checkpoint was trained on. 'v2' appends (weq - R_peak(E))/depth_scale, water-equivalent "
        "depth relative to this beamlet's own Bragg peak, which puts all energies on a common "
        "frame. v2 changes the input dimension, so a v2 run cannot warm-start from a v1 "
        "checkpoint's input projection and its checkpoints are not loadable as v1.",
    )

    # Model
    p.add_argument("--model-kind", choices=("fan_conv", "separable_fan_conv", "repvgg_unet"), default="fan_conv")
    p.add_argument("--no-augmentation", action="store_true")
    p.add_argument("--no-grad-checkpoint", action="store_true")
    p.add_argument("--compile-model", action="store_true", help="Compile the dense BEV correction model with torch.compile")
    p.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    p.add_argument("--compile-cudagraphs", action="store_true", help="Allow Inductor CUDA Graph capture for compiled model")
    p.add_argument("--model-depth", type=int, default=0, help="Pad model inputs to this fixed BEV depth; 0 keeps native depth.")
    p.add_argument("--hidden-dim", type=int, default=16)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--depth-kernel-size", type=int, default=7)
    p.add_argument("--mix-ratio", type=float, default=0.5)
    p.add_argument("--use-repvgg", action="store_true")
    p.add_argument("--unet-native-dim", type=int, default=8)
    p.add_argument("--unet-stage-dims", type=int, nargs=3, default=[16, 24, 32])
    p.add_argument("--unet-stage-blocks", type=int, nargs=5, default=[1, 1, 1, 2, 1])
    p.add_argument("--unet-equalize-axis", choices=("h", "w"), default="w")
    p.add_argument("--unet-equalize-factor", type=int, default=3)
    p.add_argument("--unet-extra-stage-dim", type=int, default=0)
    p.add_argument("--unet-extra-stage-blocks", type=int, default=2)
    p.add_argument("--unet-latent-depth-mixer", action="store_true")
    p.add_argument("--unet-latent-depth-mixer-kernel-size", type=int, default=11)
    p.add_argument("--unet-latent-depth-mixer-dilations", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--unet-energy-conditioning", choices=("none", "embedding", "scalar", "fourier"), default="embedding")
    p.add_argument("--unet-energy-fourier-bands", type=int, default=4)
    p.add_argument("--unet-norm", choices=("group", "instance"), default="group")
    p.add_argument(
        "--unet-conditioning-injection",
        choices=("entrance", "adagn", "entrance_adagn"),
        default="entrance",
    )
    p.add_argument("--no-unet-deep-supervision", action="store_true")
    p.add_argument("--use-depth-attention", action="store_true")
    p.add_argument("--use-lateral-attention", action="store_true")
    p.add_argument("--attention-heads", type=int, default=1)
    p.add_argument("--attention-dim", type=int, default=None)
    p.add_argument("--attention-layers", choices=("all", "last"), default="all")
    p.add_argument("--use-se-attention", action="store_true")
    p.add_argument("--se-ratio", type=float, default=0.25)
    p.add_argument("--use-scsam", action="store_true",
                   help="Add SCSAM (sequential channel+spatial attention) to each fan block")
    p.add_argument("--scsam-reduction", type=int, default=4,
                   help="Channel reduction ratio for SCSAM MLP (default: 4)")
    p.add_argument("--scsam-dilation", type=int, nargs=3, default=[2, 1, 1],
                   metavar=("D", "H", "W"),
                   help="Dilation (depth, height, width) for SCSAM spatial conv (default: 2 1 1)")
    p.add_argument("--attn-loss-weight", type=float, default=0.0,
                   help="Weight of auxiliary SCSAM spatial-attention supervision loss "
                        "(target = normalised |predicted residual|). 0 disables. "
                        "Recommended starting value: 0.05")
    p.add_argument("--identity-loss-weight", type=float, default=0.0,
                   help="CycleGAN-style identity regularizer: feed the MC ground truth as the "
                        "'pencil beam' input and penalise any correction (dose_hat should reproduce "
                        "the GT, i.e. residual ~ 0). 0 disables. Suggested 0.1-0.25.")
    p.add_argument("--identity-check", action="store_true",
                   help="Diagnostic only: run the identity pass (MC ground truth in as the "
                        "dose) during validation and log `identity_loss`, without adding it "
                        "to the objective. Answers whether the net is already a no-op on MC.")
    p.add_argument("--identity-loss-every", type=int, default=1,
                   help="Run the identity pass only every N training steps to amortise its "
                        "extra forward/backward cost. 1 = every step.")
    p.add_argument("--feature-probe", action="store_true",
                   help="Diagnostic only: does the encoder SEE dose correctness? Runs the "
                        "net on four inputs identical except the dose channel (PB, the net's "
                        "own corrected dose, MC ground truth, and MC*1.017) and reports "
                        "pairwise L1 between stage activations. Prerequisite for a feature "
                        "loss: if d(corrected,ref) < d(pb,ref) the features carry the signal "
                        "and such a loss is meaningful; if all distances are equal the "
                        "encoder is blind to the dose channel and the loss would be a no-op.")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--material-embedding-dim", type=int, default=4)
    p.add_argument("--use-sigma-conditioning", action="store_true")
    p.add_argument("--residual-mode", choices=("additive", "multiplicative"), default="additive")
    p.add_argument("--additive-scale-frac", type=float, default=0.25)
    p.add_argument("--dose-eps", type=float, default=1e-3)

    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-accum-steps", type=int, default=1, help="Micro-batches per optimizer update.")
    p.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--poly-power", type=float, default=0.0)
    p.add_argument("--ema-decay", type=float, default=0.0, help="EMA decay for evaluation/checkpoints; 0 disables EMA.")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16",
                   help="Autocast dtype when --amp is set. bfloat16 has fp32 exponent range (no overflow, no GradScaler); float16 is the legacy behavior.")
    p.add_argument("--seed", type=int, default=12345)

    # Loss
    p.add_argument("--w-dose", type=float, default=1.0)
    p.add_argument(
        "--loss-mask",
        choices=("nonzero", "high10", "blend", "additive"),
        default="nonzero",
        help="Support of the dose MAE term. 'nonzero' (legacy) averages over every nonzero "
        "reference voxel; 'high10' uses ref >= 10%% of the beamlet max, which is exactly the "
        "region the challenge's masked_beam_mae scores (30-87x fewer voxels, ~80%% of the "
        "dose); 'blend' mixes them by --loss-high10-frac; 'additive' keeps full support at "
        "FULL weight and adds high10 on top, scaled by --loss-high10-weight.",
    )
    p.add_argument(
        "--loss-high10-frac",
        type=float,
        default=0.7,
        help="With --loss-mask blend: weight on the high10 term, remainder on full support.",
    )
    # The two terms are NOT on the same scale: dose_high10 divides by the high10 voxel
    # count and dose by the full nonzero count, and the halo outnumbers the scored core
    # 30-87x, so dose_high10 runs ~17x larger. That makes `blend` at frac=0.7 an effective
    # 40:1 split, not 70:30 -- the full-support term carries ~2.4% of the dose loss, which
    # is why switching to blend cost ~27% of mae_pct. 'additive' exists to decouple them:
    # raising this weight adds high10 pressure without suppressing full support, whereas
    # raising --loss-high10-frac does both at once.
    #   ~0.06 (= 1/17) makes the two terms contribute equally
    #   ~0.15 gives high10 roughly 2.5x the full-support term
    #   ~40   reproduces the current blend frac=0.7
    p.add_argument(
        "--loss-high10-weight",
        type=float,
        default=0.15,
        help="With --loss-mask additive: weight on the high10 term added to full-support "
        "MAE at weight 1.0. See the note above on the ~17x scale gap between the terms.",
    )
    p.add_argument("--w-energy", type=float, default=0.0)
    p.add_argument("--w-profile", type=float, default=0.0)
    p.add_argument("--w-idd", type=float, default=0.0,
                   help="Depth-dose (Bragg curve) profile term, binned along the radiological "
                        "BEAM axis. This is NOT the challenge IDD metric -- use --w-idd-z for that.")
    p.add_argument("--w-idd-z", type=float, default=0.0,
                   help="Challenge Level-1.2 IDD term: profiles along world z (the BEV h axis), "
                        "summing over the two transverse axes, normalized by the reference peak. "
                        "Differentiable form of the reported idd_z metric.")
    p.add_argument("--w-halo", type=float, default=0.0,
                   help="Halo term: masked L1 over the halo (nonzero, below 10%% of the "
                        "beamlet peak), normalized by the MEAN HALO reference rather than the "
                        "peak. The halo is ~98%% of voxels but ~1/100th the per-voxel gradient "
                        "under `dose`, which is why it stays at -5%% (-13%% above 180 MeV). "
                        "Targets IDD, which integrates the whole plane; beam MAE masks to "
                        "ref>10%% so this cannot move it either way. Oracle ceiling: IDD "
                        "-35.7%% overall, -56%% above 180 MeV. Needs no energy gate -- the "
                        "residual is ~0 where the halo is already correct. "
                        "SCALE: dose_halo reads out as the halo relative error directly "
                        "(~0.05 for -5%%), while the whole rest of the objective is ~0.0013, "
                        "so weights are SMALL: 0.005 puts it at ~20%% of the loss, 0.02 at "
                        "~75%%. Try 0.005-0.02; 0.15 would be 20x everything else.")
    p.add_argument("--w-halo-int", type=float, default=0.0,
                   help="Signed HALO INTEGRAL term: ((sum_pred - sum_ref)/sum_ref)^2 over the "
                        "halo (ref<10%% of beamlet peak). This is the statistic IDD responds "
                        "to; --w-halo's per-voxel L1 is ~60%% irreducible scatter and a 64k "
                        "screen moved it only -2%% with iddz flat. Same form as --w-energy but "
                        "halo-masked, which is what makes it safe: w_energy/w_idd targeted the "
                        "TOTAL integral and were satisfied by inflating the core, collapsing "
                        "plan MAE/gamma/DVH. SCALE: at a -5%% halo bias this is 2.5e-3 against "
                        "a ~1.3e-3 objective, so 0.05 is ~10%% of the loss and 0.5 ~100%%. "
                        "Try 0.05-0.5. Read `halo_signed` for the signed bias directly.")
    p.add_argument("--w-peak", type=float, default=0.0)
    p.add_argument("--huber-delta", type=float, default=0.05)
    p.add_argument("--depth-bin-mm", type=float, default=1.0)
    p.add_argument("--peak-scale-mm", type=float, default=2.0)
    p.add_argument("--peak-tau-frac", type=float, default=0.05)
    p.add_argument(
        "--loss-space",
        choices=("patient", "bev"),
        default="patient",
        help="Compute training loss in patient space or directly in BEV. Validation always uses patient space.",
    )
    p.add_argument(
        "--bev-deep-supervision-weight",
        type=float,
        default=0.05,
        help="Auxiliary BEV decoder loss weight for repvgg_unet. The primary loss remains controlled by --loss-space.",
    )

    # Sampling
    p.add_argument("--beam-index", type=int, default=None)
    p.add_argument("--max-beamlets-per-beam", type=int, default=4)
    p.add_argument(
        "--beam-sampling",
        choices=("random", "full"),
        default="random",
        help="'random' samples --steps-per-epoch beams with replacement; 'full' shuffles and visits every training beam once per epoch.",
    )
    p.add_argument("--beamlet-sampling", choices=("random", "stride"), default="random")
    p.add_argument("--steps-per-epoch", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=None)

    # Validation
    p.add_argument("--val-case-dir", type=Path, action="append", default=None)
    p.add_argument("--val-case-list", type=Path, default=None)
    p.add_argument("--val-seed", type=int, default=None, help="Validation sampling seed. Defaults to --seed.")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--val-steps-per-epoch", type=int, default=5)
    p.add_argument("--validate-every-epochs", type=int, default=1)
    p.add_argument("--validate-before-training", action="store_true")
    p.add_argument("--reference-io-workers", type=int, default=8)
    p.add_argument(
        "--patient-cache-size",
        type=int,
        default=1,
        help="Max patients whose CT/material-id volumes are kept resident on the GPU (LRU evicted). "
        "Lower this if GPU memory grows until OOM; 0 disables the cap (unbounded).",
    )
    p.add_argument("--checkpoint-every-steps", type=int, default=25)
    # Metric that decides best.pt. mae_high10_pct is the local proxy for the challenge's
    # masked_beam_mae (mask = ref >= 10% of the beamlet max). All choices are
    # lower-is-better; integral_ratio is deliberately not offered, since "best" for it
    # means closest to 1, not smallest.
    p.add_argument("--best-metric", default="mae_high10_pct",
                   choices=["mae_high10_pct", "mae_pct", "idd_z", "loss", "dose"])
    p.add_argument("--profile-timing", action="store_true", help="Synchronize CUDA phase timings. Slower; use only for profiling.")
    p.add_argument("--debug-divergence", action="store_true", help="Register per-layer activation hooks and dump a full report on the first non-finite loss/grad. Use eager (no --compile-model) so hooks fire.")
    p.add_argument("--plot-every-epochs", type=int, default=1)
    p.add_argument("--plot-worst-beamlets", type=int, default=4)
    p.add_argument("--plot-worst-metric", choices=("loss", "idd", "peak_abs_shift_mm", "mae_pct"), default="loss")

    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="Warm start: load model weights from this checkpoint but keep a fresh optimizer, "
        "LR schedule and step counter. Use to continue converged weights under a new objective.",
    )
    p.add_argument(
        "--init-from-ema",
        action="store_true",
        help="With --init-from, take 'ema_model_state' instead of 'model_state'.",
    )
    p.add_argument("--plot-only", action="store_true", help="Resume checkpoint, run validation + plots, then exit (no training)")

    # Wandb
    p.add_argument("--wandb-project", type=str, default="pydose-dense-correction")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def _model_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": {
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
            "material_embedding_dim": int(args.material_embedding_dim),
            "num_materials": GEANT4_NUM_MATERIALS,
            "use_sigma_conditioning": bool(args.use_sigma_conditioning),
            "residual_mode": str(args.residual_mode),
            "additive_scale_frac": float(args.additive_scale_frac),
            "kind": str(args.model_kind),
            "grad_checkpoint": not bool(args.no_grad_checkpoint),
            "depth_kernel_size": int(args.depth_kernel_size),
            "mix_ratio": float(args.mix_ratio),
            "use_repvgg": bool(args.use_repvgg),
            "unet_native_dim": int(args.unet_native_dim),
            "unet_stage_dims": list(args.unet_stage_dims),
            "unet_stage_blocks": list(args.unet_stage_blocks),
            "unet_equalize_axis": str(args.unet_equalize_axis),
            "unet_equalize_factor": int(args.unet_equalize_factor),
            "unet_extra_stage_dim": int(args.unet_extra_stage_dim),
            "unet_extra_stage_blocks": int(args.unet_extra_stage_blocks),
            "unet_latent_depth_mixer": bool(args.unet_latent_depth_mixer),
            "unet_latent_depth_mixer_kernel_size": int(args.unet_latent_depth_mixer_kernel_size),
            "unet_latent_depth_mixer_dilations": list(args.unet_latent_depth_mixer_dilations),
            "unet_energy_conditioning": str(args.unet_energy_conditioning),
            "unet_energy_fourier_bands": int(args.unet_energy_fourier_bands),
            "unet_norm": str(args.unet_norm),
            "unet_conditioning_injection": str(args.unet_conditioning_injection),
            "unet_deep_supervision": not bool(args.no_unet_deep_supervision),
            "use_depth_attention": bool(args.use_depth_attention),
            "use_lateral_attention": bool(args.use_lateral_attention),
            "attention_heads": int(args.attention_heads),
            "attention_dim": None if args.attention_dim is None else int(args.attention_dim),
            "attention_layers": str(args.attention_layers),
            "use_se_attention": bool(args.use_se_attention),
            "se_ratio": float(args.se_ratio),
            "use_scsam": bool(args.use_scsam),
            "scsam_reduction": int(args.scsam_reduction),
            "scsam_dilation": list(args.scsam_dilation),
        },
        "attn_loss_weight": float(args.attn_loss_weight),
        "bev_deep_supervision_weight": float(args.bev_deep_supervision_weight),
        "identity_loss_weight": float(args.identity_loss_weight),
        "identity_loss_every": int(args.identity_loss_every),
        "dose": {
            "eps": float(args.dose_eps),
        },
    }


def _model_class(model_kind: str) -> type[nn.Module]:
    model_classes: dict[str, type[nn.Module]] = {
        "fan_conv": FanGridConvCorrector,
        "separable_fan_conv": SeparableFanGridConvCorrector,
        "repvgg_unet": RepVGGUNetCorrector,
    }
    try:
        return model_classes[str(model_kind)]
    except KeyError as exc:
        raise ValueError(f"Unsupported dense correction model kind: {model_kind!r}") from exc


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------


def _load_case(case_dir: Path):
    import SimpleITK as sitk

    patient_id, plan_path, ct_path, dose_dir = _resolve_case_files(case_dir)
    plan = _load_json(plan_path)
    ct_image = sitk.ReadImage(str(ct_path))
    ct_hu = sitk.GetArrayFromImage(ct_image).astype(np.float32, copy=False)
    return patient_id, plan, dose_dir, ct_hu, _origin_zyx(ct_image), _resolution_zyx(ct_image)


def _beam_mean_energy(plan: dict, beam_index: int) -> float:
    energies = [
        float(bl["energy"])
        for ray in plan["beams"][beam_index]["rays"]
        for bl in ray["beamlets"]
    ]
    return float(np.mean(energies)) if energies else 150.0


def _ray_gantry_angle_deg(beam_json: dict, ray_json: dict) -> float:
    if "ray_source" not in ray_json or "ray_target" not in ray_json:
        return float(beam_json["gantry_angle"])
    source_zyx = _xyz_to_zyx(ray_json["ray_source"]).astype(np.float64)
    target_zyx = _xyz_to_zyx(ray_json["ray_target"]).astype(np.float64)
    axis = target_zyx - source_zyx
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        return float(beam_json["gantry_angle"])
    axis = axis / norm
    return float(math.degrees(math.atan2(-axis[2], axis[1])))


# ---------------------------------------------------------------------------
# BEV feature construction
# ---------------------------------------------------------------------------

# Input channel counts live in hooks.BEV_FEATURE_CHANNELS so training and inference
# cannot drift apart; resolve per-run from args.bev_feature_set, never a module constant.


#: Bragg-peak depth per energy. The trainer builds exactly one LUT per run, so a
#: module-level cache is safe; ``get_edep_curve`` is itself cached inside the LUT,
#: which makes this a dict hit after the first beamlet at each energy.
_PEAK_DEPTH_CACHE: dict[float, float] = {}


def _peak_depth_mm(lut: PyRadPlanIonLUT, energy_mev: float) -> float:
    """Water-equivalent depth of the Bragg peak at ``energy_mev``, in mm.

    Mirrors ``ProtonDenseBevCorrectionHook._peak_depth_mm``.
    """
    key = round(float(energy_mev), 4)
    cached = _PEAK_DEPTH_CACHE.get(key)
    if cached is None:
        depth, idd = lut.get_edep_curve(key, energy_value_hint=key)
        cached = float(depth[torch.argmax(idd)])
        _PEAK_DEPTH_CACHE[key] = cached
    return cached


def _adapt_state_for_feature_set(
    state: dict[str, torch.Tensor],
    src_channels: int,
    dst_channels: int,
) -> dict[str, torch.Tensor]:
    """Let a v2 run warm-start from a v1 checkpoint without discarding it.

    ``in_proj`` consumes ``cat((bev_features, material_embedding), dim=1)``, so going
    v1 (8) -> v2 (9) inserts a channel at index 8 and shifts the material-embedding
    block right by one. Zero-initialising the inserted column makes the v2 model
    *exactly* functionally identical to the v1 checkpoint at step 0: the new feature
    contributes nothing until training gives it a weight. Without this the input
    projection would have to be re-learned from scratch and a v1/v2 comparison would
    confound the new channel with a partially reinitialised model.
    """
    if src_channels == dst_channels:
        return state
    if dst_channels < src_channels:
        raise ValueError(
            f"cannot warm-start a {dst_channels}-channel model from a "
            f"{src_channels}-channel checkpoint; channels are only ever added"
        )
    added = dst_channels - src_channels
    key = "in_proj.weight"
    weight = state.get(key)
    if weight is None:
        return state
    material_dim = int(weight.shape[1]) - src_channels
    if material_dim < 0:
        raise ValueError(
            f"{key} has {weight.shape[1]} input channels, fewer than the "
            f"{src_channels} the checkpoint declares"
        )
    adapted = weight.new_zeros((weight.shape[0], weight.shape[1] + added, *weight.shape[2:]))
    adapted[:, :src_channels] = weight[:, :src_channels]
    if material_dim:
        adapted[:, dst_channels:] = weight[:, src_channels:]
    out = dict(state)
    out[key] = adapted
    print(
        f"warm start: widened {key} {src_channels}->{dst_channels} BEV channels "
        f"(+{material_dim} material), new columns zero-init -> identical at step 0"
    )
    return out


def _build_bev_features(
    spr_bev: torch.Tensor,
    weq_bev: torch.Tensor,
    dose_pb_bev: torch.Tensor,
    material_id_bev: torch.Tensor,
    bev_crop_hw: int,
    crop_center_hw: tuple[float, float],
    depth_scale: float = 100.0,
    bev_crop_h: int | None = None,
    bev_crop_w: int | None = None,
    feature_set: str = "v1",
    peak_depth_mm: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the FanGridConvCorrector input tensor from BEV quantities.

    All inputs are ``[N, D, H, W]``.  A lateral crop of ``±bev_crop_hw`` voxels
    around the centre (beam axis) is applied.

    Returns ``(features, dose_pb, valid_mask, material_id, fan_mask)``
    each ``[1, *, D, crop_H, crop_W]``.
    """
    N, D, H, W = spr_bev.shape
    crop_h = int(bev_crop_hw if bev_crop_h is None else bev_crop_h)
    crop_w = int(bev_crop_hw if bev_crop_w is None else bev_crop_w)

    h_src, h_dst, h_target = _crop_slices(crop_center_hw[0], H, crop_h)
    w_src, w_dst, w_target = _crop_slices(crop_center_hw[1], W, crop_w)

    cH, cW = h_target.stop - h_target.start, w_target.stop - w_target.start
    spr_c = spr_bev.new_zeros((N, D, cH, cW))
    weq_c = weq_bev.new_zeros((N, D, cH, cW))
    dose_c = dose_pb_bev.new_zeros((N, D, cH, cW))
    mat_c = material_id_bev.new_zeros((N, D, cH, cW))
    spr_c[:, :, h_dst, w_dst] = spr_bev[:, :, h_src, w_src]
    weq_c[:, :, h_dst, w_dst] = weq_bev[:, :, h_src, w_src]
    dose_c[:, :, h_dst, w_dst] = dose_pb_bev[:, :, h_src, w_src]
    mat_c[:, :, h_dst, w_dst] = material_id_bev[:, :, h_src, w_src]

    # Lateral offsets from beam axis in voxels
    h_offsets = torch.arange(h_target.start, h_target.stop, device=spr_c.device, dtype=spr_c.dtype) - float(crop_center_hw[0])
    w_offsets = torch.arange(w_target.start, w_target.stop, device=spr_c.device, dtype=spr_c.dtype) - float(crop_center_hw[1])
    h_grid = h_offsets.view(1, 1, cH, 1).expand(N, D, cH, cW) / max(depth_scale, 1e-8)
    w_grid = w_offsets.view(1, 1, 1, cW).expand(N, D, cH, cW) / max(depth_scale, 1e-8)

    # Depth index as t_samples analogue
    t_depth = (torch.arange(D, device=spr_c.device, dtype=spr_c.dtype) + 0.5).view(1, D, 1, 1).expand(N, D, cH, cW)
    dose_scale = dose_c.clamp_min(1e-8).amax(dim=(1, 2, 3), keepdim=True)

    # Feature channels: [N, C, D, cH, cW], C per BEV_FEATURE_CHANNELS[feature_set].
    # Must stay byte-identical to ProtonDenseBevCorrectionHook._build_bev_features --
    # training and inference build this stack through two separate code paths, which is
    # exactly how the BEV-crop bug survived to a scored submission.
    channels = [
        spr_c - 1.0,                           # density - 1
        torch.sqrt(h_grid.square() + w_grid.square()),  # radius
        t_depth / max(depth_scale, 1e-8),       # normalised depth index
        weq_c / max(depth_scale, 1e-8),         # water-equivalent depth
        h_grid,                                 # lateral H offset
        w_grid,                                 # lateral W offset
        spr_c.clamp_min(0.0),                   # sigma proxy channel 1 (SPR as surrogate)
        dose_c.clamp_min(0.0) / dose_scale,     # normalised dose
    ]
    if feature_set in ("v2", "v3"):
        if peak_depth_mm is None:
            raise ValueError(f"feature_set={feature_set!r} requires peak_depth_mm")
        # Residual range: WEQ from this beamlet's own Bragg peak. Negative proximal,
        # positive distal. Puts every energy on a common frame.
        channels.append((weq_c - float(peak_depth_mm)) / max(depth_scale, 1e-8))
    if feature_set == "v3":
        # Range mixing: lateral WEQ structure the pencil-beam model cannot express.
        channels.append(_lateral_weq_gradient(weq_c) / max(depth_scale, 1e-8))
    features = torch.stack(channels, dim=1)

    dose_pb_5d = dose_c.unsqueeze(1)  # [1, 1, D, cH, cW]
    valid_mask = (dose_c > 0.0).unsqueeze(1)  # [1, 1, D, cH, cW]
    material_id_5d = mat_c.unsqueeze(1)  # [1, 1, D, cH, cW]
    fan_mask = torch.ones(N, 1, 1, cH, cW, device=spr_c.device, dtype=torch.bool)

    return features, dose_pb_5d, valid_mask, material_id_5d, fan_mask


def _compute_lattice_edep_bev(
    engine: IonDoseEngine,
    seq,
    weq_bev: torch.Tensor,
    crop_center_hw: tuple[float, float],
    resolved_offset: torch.Tensor | None,
) -> torch.Tensor:
    """Compute one-beamlet BEV energy deposition through the engine's split kernel."""
    return _compute_lattice_edep_bev_batch(
        engine,
        seq,
        weq_bev,
        [crop_center_hw],
        resolved_offset,
    )


def _compute_lattice_edep_bev_batch(
    engine: IonDoseEngine,
    seq,
    weq_bev: torch.Tensor,
    crop_centers_hw: list[tuple[float, float]],
    resolved_offset: torch.Tensor | None,
) -> torch.Tensor:
    """Compute a batch of BEV lattice pencil beams through the split kernel."""
    from pydose_rt.data.ion_beam import IonSpotBeamSequence

    (
        _spot_positions_mm,
        spot_weights,
        _spot_layer_index,
        spot_mask,
        layer_energies_mev,
        layer_sigmas_mm,
        layer_mask,
    ) = IonSpotBeamSequence.stack([seq])
    device = weq_bev.device
    dtype = weq_bev.dtype
    spot_weights = spot_weights.to(device=device, dtype=dtype)
    spot_mask = spot_mask.to(device=device, dtype=torch.bool)
    layer_energies_mev = layer_energies_mev.to(device=device, dtype=dtype)
    layer_sigmas_mm = layer_sigmas_mm.to(device=device, dtype=dtype)
    layer_mask = layer_mask.to(device=device, dtype=torch.bool)
    if resolved_offset is None:
        resolved_offset = torch.zeros(len(seq), device=device, dtype=dtype)
    else:
        resolved_offset = resolved_offset.to(device=device, dtype=dtype)

    G, _D, H, W = weq_bev.shape
    if G != len(seq) or len(crop_centers_hw) != G:
        raise ValueError("weq_bev, seq, and crop_centers_hw must have matching beam counts")
    res_h, _res_d, res_w = (float(v) for v in engine.dose_grid_spacing)
    h_coords = torch.arange(H, device=device, dtype=dtype)
    w_coords = torch.arange(W, device=device, dtype=dtype)
    crop_centers_t = torch.tensor(crop_centers_hw, device=device, dtype=dtype)
    valid_lateral = torch.ones((G, H, W), device=device, dtype=torch.bool)
    edep = []
    for g_idx in range(G):
        energy = layer_energies_mev[0, g_idx, 0]
        edep.append(
            engine.compute_layer_edep(
                g_idx,
                energy,
                float(energy.detach()),
                layer_sigmas_mm[0, g_idx, 0, 0],
                layer_sigmas_mm[0, g_idx, 0, 1],
                spot_weights[0, g_idx, 0],
                weq_bev.unsqueeze(0),
                resolved_offset,
                crop_centers_t,
                res_w,
                res_h,
                h_coords,
                w_coords,
                H,
                W,
                valid_lateral,
                spot_mask,
                layer_mask,
                splitting_mode="split",
                n_per_dim=9,
            )
        )
    return torch.stack(edep, dim=0)


# ---------------------------------------------------------------------------
# Loss (mirrors train_lut_calibration)
# ---------------------------------------------------------------------------


def _depth_dose_profile(vol: torch.Tensor, depth_mm: torch.Tensor, bin_mm: float) -> tuple[torch.Tensor, torch.Tensor]:
    values = vol.reshape(-1).clamp_min(0.0)
    depth = depth_mm.reshape(-1)
    width = max(float(bin_mm), 1e-6)
    lo = torch.floor(depth.detach().amin() / width) * width
    hi = torch.ceil(depth.detach().amax() / width) * width
    n_bins = max(int(math.ceil(float((hi - lo).detach()) / width)) + 1, 1)
    idx = torch.floor((depth.detach() - lo) / width).long().clamp(0, n_bins - 1)
    profile = values.new_zeros(n_bins)
    profile.scatter_add_(0, idx, values)
    centers = lo + (torch.arange(n_bins, device=values.device, dtype=values.dtype) + 0.5) * width
    return centers, profile


def _idd_curve_loss(
    pred_profile: torch.Tensor,
    ref_profile: torch.Tensor,
    args: argparse.Namespace,
    eps: float = 1e-8,
) -> torch.Tensor:
    ref_peak = ref_profile.detach().amax().clamp_min(eps)
    ref_norm = ref_profile.detach() / ref_peak
    pred_norm = pred_profile / ref_peak
    return torch.sqrt(torch.mean((pred_norm - ref_norm).square()).clamp_min(eps))


def _soft_peak_depth(profile: torch.Tensor, depth_mm: torch.Tensor, tau_frac: float) -> torch.Tensor:
    scale = profile.detach().amax().clamp_min(1e-8)
    weights = torch.softmax(profile / (scale * max(float(tau_frac), 1e-6)), dim=0)
    return (weights * depth_mm).sum()


def _depth_volume(shape: tuple[int, int, int], resolution: tuple[float, float, float], gantry_deg: float, device, dtype):
    z, y, x = shape
    theta = math.radians(float(gantry_deg))
    axis = (0.0, math.cos(theta), -math.sin(theta))
    zz = torch.arange(z, device=device, dtype=dtype) * float(resolution[0])
    yy = torch.arange(y, device=device, dtype=dtype) * float(resolution[1])
    xx = torch.arange(x, device=device, dtype=dtype) * float(resolution[2])
    return zz[:, None, None] * axis[0] + yy[None, :, None] * axis[1] + xx[None, None, :] * axis[2]


def _dose_bbox(pred: torch.Tensor, ref: torch.Tensor, pad: int = 2):
    mask = (pred.detach() != 0.0) | (ref.detach() != 0.0)
    if not bool(mask.any()):
        return None
    zs = torch.where(mask.any(dim=2).any(dim=1))[0]
    ys = torch.where(mask.any(dim=2).any(dim=0))[0]
    xs = torch.where(mask.any(dim=1).any(dim=0))[0]
    return (
        max(int(zs[0]) - pad, 0),
        min(int(zs[-1]) + 1 + pad, pred.shape[0]),
        max(int(ys[0]) - pad, 0),
        min(int(ys[-1]) + 1 + pad, pred.shape[1]),
        max(int(xs[0]) - pad, 0),
        min(int(xs[-1]) + 1 + pad, pred.shape[2]),
    )


_OPTIONAL_KEYS = ("profile", "idd", "idd_z_loss", "peak", "peak_shift_mm",
                  "peak_abs_shift_mm", "identity_loss", "attn_loss")


def _optional_terms_str(rec: dict) -> str:
    """Render only the loss terms this run actually enabled (see LOSS_KEYS note)."""
    out = ""
    for key, fmt in (("profile", ".4g"), ("idd", ".6g"), ("peak", ".4g"),
                     ("peak_abs_shift_mm", ".2f"), ("identity_loss", ".3g"),
                     ("attn_loss", ".3g")):
        if key in rec:
            out += f" {key}={float(rec[key]):{fmt}}"
    return out


def _is_new_best(best_state: dict[str, Any], value: float | None) -> bool:
    """Does `value` beat the incumbent in `best_state`? All best metrics are lower-better.

    A missing or non-finite value never wins: a NaN comparison is False in either
    direction, so an unguarded `<` would silently freeze best.pt at the first NaN epoch.
    """
    if value is None or not math.isfinite(float(value)):
        return False
    incumbent = best_state.get("value")
    return incumbent is None or float(value) < float(incumbent)


def _loss(pred: torch.Tensor, ref: torch.Tensor, depth_mm: torch.Tensor, args: argparse.Namespace):
    eps = 1e-8
    ref_max = ref.detach().amax().clamp_min(eps)
    ref_sum = ref.detach().sum().clamp_min(eps)
    mask = (ref.detach() != 0.0).to(dtype=pred.dtype)
    if not mask.any():
        mask = (pred.detach() != 0.0).to(dtype=pred.dtype)
    if not mask.any():
        mask = torch.ones_like(ref)

    dose_vox = (pred - ref).abs()
    dose_raw = (dose_vox * mask).sum() / mask.sum().clamp_min(1.0)
    dose = dose_raw / ref_max
    high10_mask = (ref.detach() > 0.10 * ref_max).to(dtype=pred.dtype)
    dose_high10 = (
        (dose_vox * high10_mask).sum() / high10_mask.sum().clamp_min(1.0) / ref_max
    )
    # The scored quantity is dose_high10 (challenge masked_beam_mae); `dose` averages over
    # the whole nonzero support, where the halo outnumbers the scored core 30-87x.
    loss_mask = getattr(args, "loss_mask", "nonzero")
    if loss_mask == "high10":
        dose_term = dose_high10
    elif loss_mask == "blend":
        f = float(getattr(args, "loss_high10_frac", 0.7))
        dose_term = f * dose_high10 + (1.0 - f) * dose
    elif loss_mask == "additive":
        # Full support keeps weight 1.0; high10 is added on top rather than traded against
        # it. Unlike `blend`, turning this up does not turn full support down.
        dose_term = dose + float(getattr(args, "loss_high10_weight", 0.15)) * dose_high10
    else:
        dose_term = dose

    # Halo term (--w-halo). The halo (nonzero, below 10% of peak) is ~98% of the voxels but
    # is effectively invisible to `dose`, which divides by ref_max: a voxel at 1% of peak
    # contributes ~1/100th the gradient of a core voxel. Measured consequence: the shipped
    # model sits at core -0.2% / halo -5.25% (-13% above 180 MeV). Normalising by the MEAN
    # HALO REFERENCE instead puts halo error on a comparable footing, so the term actually
    # pushes. An oracle that fixes the halo perfectly is worth IDD -35.7% (-56% above
    # 180 MeV) at ZERO beam-MAE cost -- MAE masks to ref>10%, so it cannot move either way.
    # No energy gate: the term is scaled by the halo residual, which is ~0 where the halo is
    # already right, so it self-suppresses at low energy instead of needing a threshold.
    dose_halo = pred.new_zeros(())
    halo_int = pred.new_zeros(())
    halo_signed = 0.0
    if getattr(args, "w_halo", 0.0) > 0.0 or getattr(args, "w_halo_int", 0.0) > 0.0:
        halo_mask = ((ref.detach() > 0.0) & (ref.detach() < 0.10 * ref_max)).to(dtype=pred.dtype)
        halo_n = halo_mask.sum().clamp_min(1.0)
        ref_halo_mean = ((ref.detach() * halo_mask).sum() / halo_n).clamp_min(eps)
        # Per-voxel L1. MEASURED 2026-08-09 TO BE THE WRONG TARGET: dose_halo sits at 0.13
        # while the systematic bias is only 0.05, so ~60% of it is irreducible per-voxel
        # scatter (reference MC noise + shape mismatch) that is not predictable from CT+PB.
        # A 4-arm 64k screen moved it just -2.0% and plateaued, with iddz unmoved.
        dose_halo = (dose_vox * halo_mask).sum() / halo_n / ref_halo_mean
        # Signed halo INTEGRAL -- the statistic IDD actually responds to, and the one the
        # oracle fixes for IDD -35.7% (-56% above 180 MeV). Same form as `energy` but
        # restricted to the halo, which is what makes it safe: w_energy/w_idd targeted the
        # TOTAL integral and the net satisfied them by inflating the core, where plan dose
        # accumulates -- that is what collapsed plan MAE/gamma/DVH on the platform. Masked
        # to ref<10% of peak, the core cannot be touched at all.
        p_halo = (pred * halo_mask).sum()
        r_halo = (ref.detach() * halo_mask).sum().clamp_min(eps)
        halo_ratio = (p_halo - r_halo) / r_halo
        halo_int = halo_ratio.pow(2)
        halo_signed = float(halo_ratio.detach())

    energy = ((pred.sum() - ref.sum()) / ref_sum).pow(2)

    profile = pred.new_zeros(())
    idd = pred.new_zeros(())
    peak = pred.new_zeros(())
    peak_shift = 0.0
    if args.w_profile > 0.0 or args.w_idd > 0.0 or args.w_peak > 0.0:
        pd, pp = _depth_dose_profile(pred, depth_mm, args.depth_bin_mm)
        _, rp = _depth_dose_profile(ref.detach(), depth_mm, args.depth_bin_mm)
        pmax = rp.detach().amax().clamp_min(eps)
        pmask = (rp.detach() != 0.0).float()
        if not pmask.any():
            pmask = torch.ones_like(rp)
        prof_vox = F.huber_loss(pp / pmax, rp / pmax, delta=float(args.huber_delta), reduction="none")
        profile = (prof_vox * pmask).sum() / pmask.sum().clamp_min(1.0)
        idd = _idd_curve_loss(pp, rp, args, eps=eps)
        if args.w_peak > 0.0:
            shift = _soft_peak_depth(pp, pd, args.peak_tau_frac) - _soft_peak_depth(rp.detach(), pd, args.peak_tau_frac)
            peak = (shift / float(args.peak_scale_mm)).pow(2)
            peak_shift = float(shift.detach())

    # Differentiable form of the idd_z metric below, i.e. the axis the challenge actually
    # scores. NOT the same thing as `--w-idd`: that one profiles along depth_mm (the
    # radiological beam axis) and is a Bragg-curve term. Level-1.2 profiles along numpy
    # axis 0 = world z, which is the BEV `h` axis here -- perpendicular to the beam.
    idd_z_loss = pred.new_zeros(())
    if getattr(args, "w_idd_z", 0.0) > 0.0:
        zp_l = pred.sum(dim=(-3, -1)).reshape(-1, pred.shape[-2])
        zr_l = ref.detach().sum(dim=(-3, -1)).reshape(-1, ref.shape[-2])
        zmax_l = zr_l.amax(dim=1, keepdim=True).clamp_min(eps)
        # eps inside the sqrt: the gradient of sqrt is unbounded at 0, which a perfectly
        # matched profile would otherwise hit.
        idd_z_loss = torch.sqrt(
            torch.mean(((zp_l - zr_l) / zmax_l).square(), dim=1) + eps
        ).mean()

    total = (
        args.w_dose * dose_term
        + args.w_energy * energy
        + args.w_profile * profile
        + args.w_idd * idd
        + args.w_peak * peak
        + getattr(args, "w_idd_z", 0.0) * idd_z_loss
        + getattr(args, "w_halo", 0.0) * dose_halo
        + getattr(args, "w_halo_int", 0.0) * halo_int
    )
    # Proxy for the challenge Level-1.2 IDD distance, which profiles along world z --
    # the BEV `h` lateral axis here (the eval builds its BEV with u = [1,0,0] in zyx and
    # the gantry rotates about z, so h is z-invariant). Reported per beamlet then meaned,
    # matching the challenge's nanmean. Proxy because it lives on the BEV crop rather than
    # the full 164-slice CT grid, so only relative values are meaningful.
    with torch.no_grad():
        zp = pred.detach().sum(dim=(-3, -1)).reshape(-1, pred.shape[-2])
        zr = ref.detach().sum(dim=(-3, -1)).reshape(-1, ref.shape[-2])
        zmax = zr.amax(dim=1, keepdim=True).clamp_min(eps)
        idd_z = torch.sqrt(torch.mean(((zp - zr) / zmax).square(), dim=1)).mean()

    mae_pct = 100.0 * dose
    mae_high10_pct = 100.0 * dose_high10
    terms = {
        "dose": float(dose.detach()),
        "dose_raw": float(dose_raw.detach()),
        "energy": float(energy.detach()),
        "mae_pct": float(mae_pct.detach()),
        "mae_high10_pct": float(mae_high10_pct.detach()),
        "idd_z": float(idd_z.detach()),
        "integral_ratio": float((pred.detach().sum() / ref_sum).detach()),
    }
    # profile/idd/peak are only computed when their weight is nonzero (see above); drop
    # the placeholder zeros rather than logging dead columns for every run.
    if args.w_profile > 0.0:
        terms["profile"] = float(profile.detach())
    if getattr(args, "w_idd_z", 0.0) > 0.0:
        terms["idd_z_loss"] = float(idd_z_loss.detach())
    if args.w_idd > 0.0:
        terms["idd"] = float(idd.detach())
    if getattr(args, "w_halo", 0.0) > 0.0:
        terms["dose_halo"] = float(dose_halo.detach())
    if getattr(args, "w_halo_int", 0.0) > 0.0:
        terms["halo_int"] = float(halo_int.detach())
        terms["halo_signed"] = halo_signed
    if args.w_peak > 0.0:
        terms["peak"] = float(peak.detach())
        terms["peak_shift_mm"] = peak_shift
        terms["peak_abs_shift_mm"] = abs(peak_shift)
    return total, terms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_items(
    items: list[tuple[int, int]],
    args: argparse.Namespace,
    epoch: int,
    case_i: int,
    beam_i: int,
    *,
    seed: int | None = None,
):
    n = int(args.max_beamlets_per_beam or 0)
    if n <= 0 or len(items) <= n:
        return items
    if args.beamlet_sampling == "stride":
        idx = np.linspace(0, len(items) - 1, n, dtype=int)
        offset = ((epoch - 1) * n + case_i * 997 + beam_i * 37) % len(items)
        return [items[(int(i) + offset) % len(items)] for i in idx]
    base_seed = int(args.seed) if seed is None else int(seed)
    rng = np.random.default_rng(base_seed + epoch * 1_000_003 + case_i * 1009 + beam_i)
    idx = np.sort(rng.choice(len(items), size=n, replace=False))
    return [items[int(i)] for i in idx]


def _crop_slices(center: float, size: int, half_width: int) -> tuple[slice, slice, slice]:
    center_i = int(round(float(center)))
    target_lo = center_i - int(half_width)
    target_hi = center_i + int(half_width)
    src_lo = max(target_lo, 0)
    src_hi = min(target_hi, int(size))
    dst_lo = src_lo - target_lo
    dst_hi = dst_lo + max(src_hi - src_lo, 0)
    return slice(src_lo, src_hi), slice(dst_lo, dst_hi), slice(target_lo, target_hi)


def _crop_bev_volume(
    bev: torch.Tensor,
    crop_center_hw: tuple[float, float],
    crop_h: int,
    crop_w: int,
) -> torch.Tensor:
    """Crop ``[B,D,H,W]`` BEV data with zero padding outside the source volume."""
    B, D, H, W = bev.shape
    h_src, h_dst, h_target = _crop_slices(crop_center_hw[0], H, crop_h)
    w_src, w_dst, w_target = _crop_slices(crop_center_hw[1], W, crop_w)
    out = bev.new_zeros((B, D, h_target.stop - h_target.start, w_target.stop - w_target.start))
    out[:, :, h_dst, w_dst] = bev[:, :, h_src, w_src]
    return out


def _retry_io(fn, *args, attempts: int = 4, base_delay: float = 0.5):
    """Retry a (possibly NFS-flaky) filesystem read with exponential backoff.

    Transient mount/visibility glitches on networked storage make a file that is
    physically present look momentarily absent (e.g. SimpleITK "file does not
    exist" on an existing .mha). A short backoff lets the mount recover instead of
    crashing a multi-hour training run on a single blip. Re-raises the last error
    if every attempt fails.
    """
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001 - any read failure is worth retrying
            last_exc = exc
            if i == attempts - 1:
                break
            delay = base_delay * (2 ** i)
            print(
                f"[io-retry] read failed ({type(exc).__name__}: {exc}); "
                f"retry {i + 1}/{attempts - 1} in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _reference_patient_tensor(
    ref_arr: np.ndarray,
    ref_bbox: RefBbox | None,
    shape: tuple[int, int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Expand a bbox-compressed MC reference into its patient-space grid."""
    ref = torch.from_numpy(ref_arr).to(device=device, dtype=dtype)
    if ref_bbox is None:
        if tuple(ref.shape) != tuple(shape):
            raise ValueError(f"Expected full reference shape {shape}, got {tuple(ref.shape)}")
        return ref
    z0, z1, y0, y1, x0, x1 = ref_bbox
    if tuple(ref.shape) != (z1 - z0, y1 - y0, x1 - x0):
        raise ValueError(f"Reference bbox {ref_bbox} does not match payload shape {tuple(ref.shape)}")
    full = torch.zeros(shape, device=device, dtype=dtype)
    full[z0:z1, y0:y1, x0:x1] = ref
    return full


def _engine_sample_bev(engine: IonDoseEngine, volume: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Sample ``[B,H,D,W]`` into BEV using the dense engine rotation grid."""
    B, H, D, W = volume.shape
    G = engine.rad_depth_layer._inv_rot_grid.shape[1]
    crop = getattr(engine, "_dense_bev_crop", None)
    if crop is not None and G == 1:
        h_src = crop["h_src"]
        h_dst = crop["h_dst"]
        w_src = crop["w_src"]
        w_dst = crop["w_dst"]
        out_h, out_w = crop["shape_hw"]
        out = volume.new_zeros((B, D, out_h, out_w))
        if h_src.stop <= h_src.start or w_src.stop <= w_src.start:
            return out
        src_h = h_src.stop - h_src.start
        vol_flat = volume[:, h_src, :, :].reshape(B * src_h, 1, D, W)
        grid_g = engine.rad_depth_layer._inv_rot_grid[0, 0, 0, :, w_src]
        grid_g = grid_g.unsqueeze(0).expand(B * src_h, -1, -1, -1)
        sampled = F.grid_sample(
            vol_flat,
            grid_g,
            mode=mode,
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.reshape(B, src_h, D, w_src.stop - w_src.start)
        sampled = sampled.permute(0, 2, 1, 3).contiguous()
        out[:, :, h_dst, w_dst] = sampled
        return out

    vol_flat = volume.reshape(B * H, 1, D, W)
    bev_parts = []
    for g in range(G):
        grid_g = engine.rad_depth_layer._inv_rot_grid[0, g, 0]
        grid_g = grid_g.unsqueeze(0).expand(B * H, -1, -1, -1)
        sampled = F.grid_sample(
            vol_flat,
            grid_g,
            mode=mode,
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.reshape(B, H, D, W)
        bev_parts.append(sampled)
    bev = torch.stack(bev_parts, dim=1)
    return bev.permute(0, 1, 3, 2, 4).reshape(B * G, D, H, W).contiguous()


def _engine_sample_bev_multi_crop(
    engine: IonDoseEngine,
    volume: torch.Tensor,
    crops: list[dict[str, Any]],
    mode: str = "bilinear",
    beam_indices: list[int] | None = None,
) -> torch.Tensor:
    """Sample ``[B,H,D,W]`` into matching cropped BEV grids for each beam."""
    B, _full_h, D, full_w = volume.shape
    G = len(crops)
    out_h, out_w = crops[0]["shape_hw"]
    out = volume.new_zeros((B, G, D, out_h, out_w))
    if beam_indices is None:
        beam_indices = list(range(G))
    if len(beam_indices) != G:
        raise ValueError("beam_indices must match crops")
    for out_idx, (beam_idx, crop) in enumerate(zip(beam_indices, crops, strict=True)):
        h_src = crop["h_src"]
        h_dst = crop["h_dst"]
        w_src = crop["w_src"]
        w_dst = crop["w_dst"]
        if h_src.stop <= h_src.start or w_src.stop <= w_src.start:
            continue
        src_h = h_src.stop - h_src.start
        vol_flat = volume[:, h_src, :, :].reshape(B * src_h, 1, D, full_w)
        grid_g = engine.rad_depth_layer._inv_rot_grid[0, beam_idx, 0, :, w_src]
        grid_g = grid_g.unsqueeze(0).expand(B * src_h, -1, -1, -1)
        sampled = F.grid_sample(
            vol_flat,
            grid_g,
            mode=mode,
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.reshape(B, src_h, D, w_src.stop - w_src.start)
        out[:, out_idx, :, h_dst, w_dst] = sampled.permute(0, 2, 1, 3)
    return out.reshape(B * G, D, out_h, out_w).contiguous()


def _bev_deep_supervision_loss(
    predictions: tuple[torch.Tensor, ...] | list[torch.Tensor],
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Masked multi-scale BEV L1, normalized to the native target peak.

    ``eps`` floors that peak: raised 1e-8 -> 1e-3 so low-dose beamlets can't blow
    up the normalized error (the aux-eps value from the working recipe).
    """
    if not predictions:
        return target.new_zeros(())
    peak = target.detach().amax(dim=(2, 3, 4), keepdim=True).clamp_min(eps)
    losses = []
    for scale_idx, prediction in enumerate(predictions):
        pooled_target = F.adaptive_avg_pool3d(target, prediction.shape[2:])
        pooled_mask = F.adaptive_max_pool3d(valid_mask.to(dtype=target.dtype), prediction.shape[2:]) > 0.0
        error = (prediction - pooled_target).abs() / peak
        losses.append((error * pooled_mask).sum() / pooled_mask.sum().clamp_min(1))
    weights = target.new_tensor([0.5 ** scale_idx for scale_idx in range(len(losses))])
    return sum(weight * loss for weight, loss in zip(weights, losses, strict=True)) / weights.sum()


def _pad_model_depth(x: torch.Tensor, target_depth: int) -> tuple[torch.Tensor, int]:
    """Pad a ``[B,C,D,H,W]`` tensor distally to a fixed model depth."""
    original_depth = int(x.shape[2])
    target_depth = int(target_depth)
    if target_depth <= 0 or target_depth == original_depth:
        return x, original_depth
    if original_depth > target_depth:
        raise ValueError(f"model input depth {original_depth} exceeds --model-depth {target_depth}")
    return F.pad(x, (0, 0, 0, 0, 0, target_depth - original_depth)), original_depth


def _trim_model_outputs(outputs: dict[str, Any], original_depth: int) -> dict[str, Any]:
    """Remove model-only distal padding from native and auxiliary outputs."""
    trimmed = dict(outputs)
    for key in ("dose_hat", "residual"):
        value = trimmed.get(key)
        if isinstance(value, torch.Tensor):
            trimmed[key] = value[:, :, :original_depth]
    deep_supervision = trimmed.get("deep_supervision")
    if deep_supervision:
        trimmed["deep_supervision"] = tuple(value[:, :, :original_depth] for value in deep_supervision)
    attn_maps = trimmed.get("attn_maps")
    if attn_maps:
        trimmed["attn_maps"] = [value[:, :, :original_depth] for value in attn_maps]
    return trimmed


def _engine_density_weq_bev(engine: IonDoseEngine, density: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    crop = getattr(engine, "_dense_bev_crop", None)
    if crop is not None:
        return engine._dense_forward_bev_multi_crop(density, [crop])
    return engine.rad_depth_layer.forward_bev(density)


def _engine_local_crop_center_hw(engine: IonDoseEngine, center_hw: tuple[float, float]) -> tuple[float, float]:
    crop = getattr(engine, "_dense_bev_crop", None)
    if crop is None:
        return center_hw
    return (
        float(center_hw[0]) - float(crop["target_h_start"]),
        float(center_hw[1]) - float(crop["target_w_start"]),
    )


def _engine_expand_bev_crop(engine: IonDoseEngine, bev: torch.Tensor) -> torch.Tensor:
    crop = getattr(engine, "_dense_bev_crop", None)
    if crop is None or bev.shape[-2:] != crop["shape_hw"]:
        return bev
    B, D, _H, _W = bev.shape
    full_h, full_w = crop["full_shape_hw"]
    full = bev.new_zeros((B, D, full_h, full_w))
    full[:, :, crop["h_src"], crop["w_src"]] = bev[:, :, crop["h_dst"], crop["w_dst"]]
    return full


def _rotate_cropped_bev_to_patient_slab(
    engine: IonDoseEngine,
    dose_bev: torch.Tensor,
    crop: dict[str, Any],
    beam_idx: int,
) -> tuple[torch.Tensor, slice]:
    """Rotate one cropped BEV dose directly into its patient-space height slab."""
    B, D, _cH, cW = dose_bev.shape
    h_src = crop["h_src"]
    h_dst = crop["h_dst"]
    src_h = h_src.stop - h_src.start
    dose_g = dose_bev[:, :, h_dst, :]
    dose_g = dose_g.permute(0, 2, 1, 3).contiguous().reshape(B * src_h, 1, D, cW)
    full_w = int(crop["full_shape_hw"][1])
    grid_full = engine.rotation_layer.rot_grid[0, beam_idx, 0].to(device=dose_bev.device, dtype=dose_bev.dtype)
    full_x = ((grid_full[..., 0] + 1.0) * float(full_w) - 1.0) * 0.5
    crop_x = full_x - float(crop["target_w_start"])
    crop_grid_x = (2.0 * (crop_x + 0.5) / float(cW)) - 1.0
    grid_g = torch.stack((crop_grid_x, grid_full[..., 1]), dim=-1)
    grid_g = grid_g.unsqueeze(0).expand(B * src_h, -1, -1, -1)
    rotated = F.grid_sample(dose_g, grid_g, mode="bilinear", padding_mode="zeros", align_corners=False)
    return rotated.reshape(B, src_h, D, full_w), h_src


def _best_axial_slice(volume: np.ndarray) -> int:
    return int(np.argmax(volume.max(axis=(1, 2))))


def _plot_axial_prediction_check(
    out_path: Path,
    ref: np.ndarray,
    pred: np.ndarray,
    ct: np.ndarray,
    title: str,
    display_percentile: float = 99.5,
) -> None:
    import matplotlib.pyplot as plt

    axial_idx = _best_axial_slice(ref)
    ref_slice = ref[axial_idx]
    pred_slice = pred[axial_idx]
    ct_slice = ct[axial_idx]
    robust_vmax = float(max(np.percentile(ref, display_percentile), ref.max(), 1e-8))
    diff = pred_slice - ref_slice
    diff_vmax = float(np.max(np.abs(diff))) or 1e-8

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    panels = (
        (ct_slice, "CT", "gray", None, None),
        (ref_slice, "Reference", "inferno", 0.0, robust_vmax),
        (pred_slice, "Prediction", "inferno", 0.0, robust_vmax),
        (diff, "Prediction - reference", "bwr", -diff_vmax, diff_vmax),
    )
    for ax, (image, label, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"{title} | axial z={axial_idx}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _record_key(record: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(record["patient"]),
        int(record["beam"]),
        int(record["ray"]),
        int(record["beamlet"]),
    )


def _mean(acc: dict[str, float], count: int):
    """Average the always-present LOSS_KEYS plus whatever optional terms this run
    actually accumulated (see the LOSS_KEYS note: profile/idd/peak only appear when
    their weight is nonzero)."""
    denom = max(int(count), 1)
    keys = set(LOSS_KEYS) | set(acc)
    return {k: float(acc.get(k, 0.0)) / denom for k in keys}


def _add(acc: dict[str, float], loss_value: float, terms: dict[str, float]):
    acc["loss"] = acc.get("loss", 0.0) + float(loss_value)
    for key, value in terms.items():
        acc[key] = acc.get(key, 0.0) + float(value)


def _center_of_mass_mm(volume: np.ndarray, resolution: tuple[float, float, float]) -> np.ndarray:
    weights = np.clip(volume.astype(np.float64, copy=False), 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return np.zeros(3, dtype=np.float64)
    grids = np.indices(weights.shape, dtype=np.float64)
    return np.asarray([(weights * grids[i]).sum() / total * float(resolution[i]) for i in range(3)], dtype=np.float64)


def _ray_axis_zyx(ray_json: dict) -> np.ndarray:
    source = np.asarray([ray_json["ray_source"][2], ray_json["ray_source"][1], ray_json["ray_source"][0]], dtype=np.float64)
    target = np.asarray([ray_json["ray_target"][2], ray_json["ray_target"][1], ray_json["ray_target"][0]], dtype=np.float64)
    axis = target - source
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        return np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    return axis / norm


# ---------------------------------------------------------------------------
# Dense engine builder
# ---------------------------------------------------------------------------


def _build_engine(args, lut, machine_config, resolution_zyx, ct_shape, sequence, device, dtype):
    return IonDoseEngine(
        machine_config=machine_config,
        lut=lut,
        dose_grid_spacing=resolution_zyx,
        dose_grid_shape=ct_shape,
        beam_template=sequence,
        device=device,
        dtype=dtype,
        lateral_model=args.mode,
        transport_step_mm=args.transport_step_mm,
        field_size=tuple(args.field_size),
        heterogeneous_mcs=bool(args.heterogeneous_mcs),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    if args.bev_crop_h is None:
        args.bev_crop_h = int(args.bev_crop_hw)
    if args.bev_crop_w is None:
        args.bev_crop_w = int(args.bev_crop_hw)
    if args.field_size is None:
        args.field_size = (int(args.bev_crop_h) * 2, int(args.bev_crop_w) * 2)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    amp_enabled = bool(args.amp and device.type == "cuda" and dtype == torch.float32)
    amp_dtype = torch.bfloat16 if str(args.amp_dtype) == "bfloat16" else torch.float16
    print(f"device={device} dtype={dtype} amp={amp_enabled} amp_dtype={amp_dtype if amp_enabled else 'n/a'}")
    if int(args.model_depth) > 0:
        print(f"fixed model depth={int(args.model_depth)}")

    beam_parameters = _load_json(args.beam_params_path.resolve())
    hu_to_density = beam_parameters["hu_to_density"]["entries"]
    lut = PyRadPlanIonLUT(args.machine_mat)
    machine_config = MachineConfig(
        tpr_20_10=0.7,
        number_of_leaf_pairs=40,
        fit_air_offset_mm=0.0,
        bams_to_iso_dist_mm=float(args.bams_to_iso_dist_mm),
    )

    cfg = _model_config(args)
    model_cls = _model_class(args.model_kind)
    model = model_cls.from_config(
        BEV_FEATURE_CHANNELS[args.bev_feature_set], cfg,
        available_energies=lut.available_energies,
    ).to(device=device, dtype=dtype)
    # --- divergence diagnostics: per-layer activation tracking ---
    _dbg_act_max: dict[str, float] = {}
    _dbg_act_nonfinite: dict[str, int] = {}
    if bool(args.debug_divergence):
        def _make_act_hook(layer_name: str):
            def _hook(_module, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                if isinstance(t, torch.Tensor) and t.is_floating_point():
                    d = t.detach()
                    finite = torch.isfinite(d)
                    _dbg_act_nonfinite[layer_name] = int((~finite).sum().item())
                    _dbg_act_max[layer_name] = float(d[finite].abs().max().item()) if bool(finite.any()) else float("inf")
            return _hook
        n_hooked = 0
        for _name, _mod in model.named_modules():
            if len(list(_mod.children())) == 0 and _name:
                _mod.register_forward_hook(_make_act_hook(_name))
                n_hooked += 1
        print(f"[debug-divergence] registered {n_hooked} activation hooks (run eager so they fire)")

    ema_model = None
    if float(args.ema_decay) > 0.0:
        if not 0.0 < float(args.ema_decay) < 1.0:
            raise ValueError("--ema-decay must be in (0, 1), or 0 to disable")
        ema_model = deepcopy(model).requires_grad_(False).eval()
        print(f"EMA enabled decay={float(args.ema_decay):.6g}")
    if bool(args.compile_model):
        import torch._inductor.config as inductor_config
        inductor_config.triton.cudagraphs = bool(args.compile_cudagraphs)
        inductor_config.triton.cudagraph_trees = bool(args.compile_cudagraphs)
        inductor_config.triton.cudagraph_skip_dynamic_graphs = not bool(args.compile_cudagraphs)
        compile_mode = None if args.compile_mode == "default" else str(args.compile_mode)
        print(
            f"compiling model with torch.compile(mode={args.compile_mode}, "
            f"cudagraphs={bool(args.compile_cudagraphs)})"
        )
        model = torch.compile(model, mode=compile_mode)
    print(f"model params: {sum(p.numel() for p in model.parameters())}")

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=float(args.lr),
            momentum=float(args.momentum), weight_decay=float(args.weight_decay),
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = None
    # GradScaler is only needed/valid for fp16; bf16 has fp32 range so no loss scaling.
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    # Resume
    resume_epoch = 0
    resume_step = 0
    resume_history: list[dict[str, Any]] = []
    resume_val_history: list[dict[str, Any]] = []
    resume_best: dict[str, Any] | None = None
    if args.init_from is not None:
        # Warm start: weights only. Optimizer, schedule, step counter and history all stay
        # fresh, so a run under a NEW objective starts from converged weights instead of
        # from scratch. Distinct from --resume, which continues the same run verbatim.
        init_path = Path(args.init_from)
        if not init_path.exists():
            raise FileNotFoundError(f"--init-from but {init_path} not found")
        init_ckpt = torch.load(init_path, map_location=device, weights_only=False)
        key = "ema_model_state" if args.init_from_ema else "model_state"
        state = init_ckpt.get(key)
        if state is None:
            raise KeyError(f"{init_path} has no '{key}' (keys: {sorted(init_ckpt)})")
        state = _adapt_state_for_feature_set(
            state,
            src_channels=int(init_ckpt.get("fan_input_dim", BEV_FEATURE_CHANNELS["v1"])),
            dst_channels=BEV_FEATURE_CHANNELS[args.bev_feature_set],
        )
        load_model_state_dict(model, state)
        if ema_model is not None:
            load_model_state_dict(ema_model, state)
        print(f"warm start from {init_path} [{key}] (fresh optimizer/schedule/step)")
        del init_ckpt, state
    if args.resume:
        ckpt_path = args.output_dir / "latest.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume but {ckpt_path} not found")
        print(f"resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        load_model_state_dict(model, ckpt["model_state"])
        if ema_model is not None:
            load_model_state_dict(ema_model, ckpt.get("ema_model_state", ckpt["model_state"]))
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        resume_epoch = ckpt.get("epoch", 0)
        resume_step = ckpt.get("step", 0)
        hist_path = args.output_dir / "history.json"
        if hist_path.exists():
            resume_history = json.loads(hist_path.read_text(encoding="utf-8"))
        val_hist_path = args.output_dir / "val_history.json"
        if val_hist_path.exists():
            resume_val_history = json.loads(val_hist_path.read_text(encoding="utf-8"))
        # Carry the incumbent forward, else the first epoch after a resume always looks
        # like a new best and best.pt regresses to whatever the restart happened to reach.
        resume_best = ckpt.get("best_state")
        print(f"resumed at epoch={resume_epoch} step={resume_step}")
        del ckpt

    # Load cases
    def _read_case_list(path: Path) -> list[Path]:
        return [Path(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    all_case_dirs = list(args.case_dir or [])
    if args.case_list is not None:
        all_case_dirs.extend(_read_case_list(args.case_list))
    if not all_case_dirs:
        raise RuntimeError("No cases specified (use --case-dir or --case-list)")
    all_val_dirs = list(args.val_case_dir or [])
    if args.val_case_list is not None:
        all_val_dirs.extend(_read_case_list(args.val_case_list))

    explicit_val_dirs = {path.resolve() for path in all_val_dirs}
    train_dirs = [path.resolve() for path in all_case_dirs if path.resolve() not in explicit_val_dirs]
    if not train_dirs:
        raise RuntimeError("No training cases remain after excluding --val-case-dir")
    train_cases = [_load_case(path) for path in train_dirs]
    val_cases = [_load_case(path) for path in sorted(explicit_val_dirs)]
    if not explicit_val_dirs:
        all_cases = train_cases
        indices = list(range(len(all_cases)))
        random.Random(int(args.seed)).shuffle(indices)
        n_val = 0
        if len(all_cases) > 1 and args.val_fraction > 0.0:
            n_val = max(1, int(round(len(all_cases) * min(max(float(args.val_fraction), 0.0), 0.9))))
        val_idx = set(indices[:n_val])
        val_cases = [case for i, case in enumerate(all_cases) if i in val_idx]
        train_cases = [case for i, case in enumerate(all_cases) if i not in val_idx]
    print(f"cases train={len(train_cases)} val={len(val_cases)}")
    if val_cases:
        print("validation patients: " + ", ".join(case[0] for case in val_cases))

    # Build inventory (case_i, beam_i)
    inventory: list[tuple[int, int]] = []
    for case_i, (_pid, plan, *_rest) in enumerate(train_cases):
        if args.beam_index is None:
            inventory.extend((case_i, bi) for bi in range(len(plan["beams"])))
        else:
            inventory.append((case_i, int(args.beam_index)))
    if not inventory:
        raise RuntimeError("No training beams selected")

    full_epoch_units: list[tuple[int, int, list[tuple[int, int]] | None]] = []
    if args.beam_sampling == "full":
        chunk_size = max(1, int(args.max_beamlets_per_beam or 1))
        for case_i, bi in inventory:
            _pid, plan, *_rest = train_cases[case_i]
            beam_json = plan["beams"][bi]
            ray_idxs = _selected_ray_indices(beam_json, None, bi)
            items = [(ri, li) for ri in ray_idxs for li in range(len(beam_json["rays"][ri]["beamlets"]))]
            if not items:
                continue
            for start in range(0, len(items), chunk_size):
                full_epoch_units.append((case_i, bi, items[start:start + chunk_size]))
        if not full_epoch_units:
            raise RuntimeError("No training beamlets selected for full beam sampling")

    micro_batches_per_epoch = (
        len(full_epoch_units)
        if args.beam_sampling == "full"
        else max(int(args.steps_per_epoch), 1)
    )
    grad_accum_steps = max(1, int(args.grad_accum_steps))
    effective_steps_per_epoch = math.ceil(micro_batches_per_epoch / grad_accum_steps)
    print(
        f"beam sampling={args.beam_sampling} micro_batches_per_epoch={micro_batches_per_epoch} "
        f"grad_accum_steps={grad_accum_steps} effective_steps_per_epoch={effective_steps_per_epoch}"
    )

    if args.poly_power > 0.0:
        total_iters = int(args.max_steps) if args.max_steps is not None else int(args.epochs) * effective_steps_per_epoch
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=total_iters, power=float(args.poly_power),
        )

    # Fixed validation beamlets
    val_inventory: list[tuple[int, int]] = []
    for case_i, (_pid, plan, *_rest) in enumerate(val_cases):
        val_inventory.extend((case_i, bi) for bi in range(len(plan["beams"])))
    val_seed = int(args.seed if args.val_seed is None else args.val_seed)
    val_rng = random.Random(val_seed + 100_000)
    fixed_val: list[tuple[int, int, list[tuple[int, int]]]] = []
    if val_inventory and int(args.val_steps_per_epoch) > 0:
        target = min(int(args.val_steps_per_epoch), len(val_inventory))
        per_case: list[tuple[int, int]] = []
        for case_i, (_pid, plan, *_rest) in enumerate(val_cases):
            beam_indices = list(range(len(plan["beams"])))
            if beam_indices:
                per_case.append((case_i, val_rng.choice(beam_indices)))
        fixed_beams = per_case[:target]
        remaining = [item for item in val_inventory if item not in set(fixed_beams)]
        while len(fixed_beams) < target and remaining:
            idx = val_rng.randrange(len(remaining))
            fixed_beams.append(remaining.pop(idx))
        for case_i, bi in fixed_beams:
            _pid, plan, *_rest = val_cases[case_i]
            beam_json = plan["beams"][bi]
            ray_idxs = _selected_ray_indices(beam_json, None, bi)
            items = [(ri, li) for ri in ray_idxs for li in range(len(beam_json["rays"][ri]["beamlets"]))]
            fixed_items = _select_items(items, args, 0, case_i, bi, seed=val_seed)
            if fixed_items:
                fixed_val.append((case_i, bi, fixed_items))
    fixed_val_records = [
        {"patient": val_cases[case_i][0], "case_i": int(case_i), "beam": int(bi), "ray": int(ri), "beamlet": int(li)}
        for case_i, bi, selected_items in fixed_val
        for ri, li in selected_items
    ]
    (args.output_dir / "fixed_validation_beamlets.json").write_text(
        json.dumps(fixed_val_records, indent=2), encoding="utf-8",
    )

    # Wandb
    use_wandb = not bool(args.no_wandb)
    wandb = None
    if use_wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb_kwargs: dict[str, Any] = dict(
            project=args.wandb_project,
            name=args.wandb_run_name or args.output_dir.name,
            dir=str(args.output_dir),
            config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()} | {"model_cfg": cfg},
        )
        if args.resume:
            wandb_kwargs["resume"] = "allow"
        wandb.init(**wandb_kwargs)

    ref_pool = ThreadPoolExecutor(max_workers=max(int(args.reference_io_workers), 1))
    history: list[dict[str, Any]] = list(resume_history)
    val_history: list[dict[str, Any]] = list(resume_val_history)
    fixed_plot_keys: list[tuple[str, int, int, int]] | None = None
    global_step = resume_step
    micro_step = resume_step * grad_accum_steps
    ema_parameters = list(ema_model.parameters()) if ema_model is not None else []
    model_parameters = list(model.parameters()) if ema_model is not None else []
    if ema_model is not None and len(ema_parameters) != len(model_parameters):
        raise RuntimeError("EMA and training model parameter structures do not match")

    @torch.no_grad()
    def update_ema() -> None:
        if ema_model is None:
            return
        torch._foreach_lerp_(ema_parameters, model_parameters, 1.0 - float(args.ema_decay))

    def mark_cudagraph_step() -> None:
        if bool(args.compile_model) and bool(args.compile_cudagraphs):
            torch.compiler.cudagraph_mark_step_begin()

    @contextlib.contextmanager
    def ema_weights_in_training_model():
        """Temporarily use EMA values in the compiled model to reuse its CUDA graph pool."""
        if ema_model is None:
            yield
            return
        raw_parameters = [parameter.detach().clone() for parameter in model_parameters]
        with torch.no_grad():
            torch._foreach_copy_(model_parameters, ema_parameters)
        try:
            yield
        finally:
            with torch.no_grad():
                torch._foreach_copy_(model_parameters, raw_parameters)

    # Best-checkpoint selection. `latest.pt` is whichever epoch the run happened to stop
    # on, which stops meaning "best" as soon as the curve flattens and starts wobbling --
    # the long continuations move ~0.5% between adjacent epochs. Selection reads the EMA
    # validation when EMA is on (that is the checkpoint we ship) and the raw one otherwise.
    # Every eligible metric is lower-is-better; integral_ratio is not, and is rejected in
    # argument parsing rather than silently selecting the worst conservation.
    best_source = "val_ema" if ema_model is not None else "val"
    best_state: dict[str, Any] = dict(
        resume_best or {"value": None, "epoch": -1, "step": -1, "metric": args.best_metric,
                        "source": best_source}
    )
    best_state["pending"] = False

    def save_checkpoint(epoch: int, step_checkpoint: bool = False):
        is_best = bool(best_state.pop("pending", False))
        ckpt = {
            "model_state": portable_model_state_dict(model),
            "ema_model_state": portable_model_state_dict(ema_model) if ema_model is not None else None,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict(),
            "config": cfg,
            "fan_input_dim": BEV_FEATURE_CHANNELS[args.bev_feature_set],
            "epoch": epoch,
            "step": global_step,
            "best_state": {k: v for k, v in best_state.items() if k != "pending"},
            "args": vars(args) | {
                "case_dir": [str(p) for p in (args.case_dir or [])],
                "val_case_dir": [str(p) for p in (args.val_case_dir or [])],
                "output_dir": str(args.output_dir),
                "machine_mat": str(args.machine_mat),
                "beam_params_path": str(args.beam_params_path),
            },
        }
        torch.save(ckpt, args.output_dir / "latest.pt")
        if is_best:
            torch.save(ckpt, args.output_dir / "best.pt")
        if step_checkpoint:
            torch.save(ckpt, args.output_dir / f"step_{global_step:06d}.pt")
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (args.output_dir / "val_history.json").write_text(json.dumps(val_history, indent=2), encoding="utf-8")

    _PATIENT_CACHE_MAX = max(int(args.patient_cache_size), 0)
    _hu_cache: collections.OrderedDict[str, torch.Tensor] = collections.OrderedDict()
    _material_id_cache: collections.OrderedDict[str, torch.Tensor] = collections.OrderedDict()
    # Dose-scoring region per patient. Beam-independent, and the connected-component
    # pass is far too slow to redo per beamlet, so it caches alongside the HU volume.
    _dose_mask_cache: collections.OrderedDict[str, torch.Tensor] = collections.OrderedDict()
    _REF_CACHE_MAX = 200
    _ref_cache: collections.OrderedDict[Path, tuple[np.ndarray, RefBbox | None]] = collections.OrderedDict()

    # -----------------------------------------------------------------------
    # Per-beam step (core training loop body)
    # -----------------------------------------------------------------------

    def run_beam(
        case,
        case_i: int,
        bi: int,
        epoch: int,
        training: bool,
        selected_items: list[tuple[int, int]] | None = None,
        inference_model: nn.Module | None = None,
        backward_scale: float = 1.0,
    ) -> dict[str, Any]:
        from pydose_rt.data.ion_beam import IonSpotBeamSequence

        def _sync_timing() -> None:
            if bool(args.profile_timing) and device.type == "cuda":
                torch.cuda.synchronize()

        def _read_reference(ref_path: Path) -> tuple[np.ndarray, RefBbox | None]:
            # The b2nd-existence check is inside the retried closure on purpose: a
            # transient mount glitch can make the .b2nd look absent, fall through to
            # the .mha, and fail there too -- re-checking on retry recovers cleanly.
            def _do() -> tuple[np.ndarray, RefBbox | None]:
                if _b2nd_path(ref_path).exists():
                    return _read_reference_dose_b2nd(ref_path)
                return _read_reference_dose(ref_path), None
            return _retry_io(_do)

        def _get_reference(
            ref_path: Path,
            ref_futures: dict[Path, Any],
            pinned_refs: dict[Path, tuple[np.ndarray, RefBbox | None]],
        ) -> tuple[np.ndarray, RefBbox | None]:
            if ref_path in pinned_refs:
                return pinned_refs[ref_path]
            if ref_path in _ref_cache:
                _ref_cache.move_to_end(ref_path)
                return _ref_cache[ref_path]
            future = ref_futures.get(ref_path)
            arr, bbox = _read_reference(ref_path) if future is None else future.result()
            _ref_cache[ref_path] = (arr, bbox)
            if len(_ref_cache) > _REF_CACHE_MAX:
                _ref_cache.popitem(last=False)
            return arr, bbox

        patient_id, plan, dose_dir, ct_hu, origin_zyx, res_zyx = case
        beam_json = plan["beams"][bi]
        e_ref = _beam_mean_energy(plan, bi)
        _t_setup = time.perf_counter()

        if patient_id in _hu_cache:
            _hu_cache.move_to_end(patient_id)
        else:
            _hu_cache[patient_id] = torch.from_numpy(ct_hu).to(device=device, dtype=dtype)
        hu_t = _hu_cache[patient_id]
        spr_vol, mass_vol = spr_and_mass_density(hu_t, e_ref, hu_to_density)
        depth_vol = _depth_volume(ct_hu.shape, res_zyx, beam_json["gantry_angle"], device, dtype)

        if patient_id in _material_id_cache:
            _material_id_cache.move_to_end(patient_id)
        else:
            _material_id_cache[patient_id] = _material_id_from_hu(hu_t)
        material_id_patient = _material_id_cache[patient_id]

        if patient_id in _dose_mask_cache:
            _dose_mask_cache.move_to_end(patient_id)
        else:
            _dose_mask_cache[patient_id] = patient_dose_mask(mass_vol)
        dose_mask_patient = _dose_mask_cache[patient_id]

        # Evict least-recently-used patients so the resident GPU volumes stay
        # bounded; unbounded growth here maxes out the GPU over an epoch.
        if _PATIENT_CACHE_MAX > 0:
            while len(_hu_cache) > _PATIENT_CACHE_MAX:
                _hu_cache.popitem(last=False)
            while len(_material_id_cache) > _PATIENT_CACHE_MAX:
                _material_id_cache.popitem(last=False)
            while len(_dose_mask_cache) > _PATIENT_CACHE_MAX:
                _dose_mask_cache.popitem(last=False)

        _sync_timing()
        t_setup_ms = round((time.perf_counter() - _t_setup) * 1e3, 1)

        ray_idxs = _selected_ray_indices(beam_json, None, bi)
        items = [(ri, li) for ri in ray_idxs for li in range(len(beam_json["rays"][ri]["beamlets"]))]
        selected = selected_items if selected_items is not None else _select_items(items, args, epoch, case_i, bi)

        acc: dict[str, float] = {}
        beamlet_records: list[dict[str, Any]] = []
        done = 0
        losses: list[torch.Tensor] = []
        t0 = time.perf_counter()
        t_engine_total = 0.0
        t_bev_total = 0.0
        t_model_total = 0.0
        t_loss_total = 0.0
        t_backward_total = 0.0

        ref_futures: dict[Path, Any] = {}
        pinned_refs: dict[Path, tuple[np.ndarray, RefBbox | None]] = {}
        item_specs: list[dict[str, Any]] = []
        sequences = []
        ssd_values: list[float | None] = []
        for ri, li in selected:
            key = (patient_id, bi, ri, li)
            ray_json = beam_json["rays"][ri]
            ref_paths = _reference_paths_for_selection(dose_dir, beam_json, bi, [ri], li)
            if len(ref_paths) != 1:
                raise RuntimeError(f"Expected one MC dose for {key}, got {len(ref_paths)}")
            energy_mev = float(ray_json["beamlets"][li]["energy"])
            iso_mm = tuple((_xyz_to_zyx(ray_json["ray_target"]) - origin_zyx).tolist())
            seq, ssd_mm = _make_ray_sequence(
                plan=plan, beam_parameters=beam_parameters, ct_hu=ct_hu,
                origin_zyx=origin_zyx, resolution_zyx=res_zyx,
                beam_index=bi, ray_index=ri, beamlet_index=li,
                particles_per_beamlet=float(args.particles_per_beamlet),
                gantry_offset_deg=0.0,
                skin_hu_threshold=float(args.skin_hu_threshold),
                sigma_mode=args.sigma_mode,
                bams_to_iso_dist_mm=float(args.bams_to_iso_dist_mm),
                lut=lut, device=device, dtype=dtype,
            )
            ref_path = ref_paths[0]
            if ref_path in _ref_cache:
                _ref_cache.move_to_end(ref_path)
                pinned_refs[ref_path] = _ref_cache[ref_path]
            elif ref_path not in ref_futures:
                ref_futures[ref_path] = ref_pool.submit(_read_reference, ref_path)
            sequences.append(seq)
            ssd_values.append(ssd_mm)
            item_specs.append(
                {
                    "ri": int(ri),
                    "li": int(li),
                    "energy_mev": float(energy_mev),
                    "iso_mm": iso_mm,
                    "ref_path": ref_path,
                }
            )

        if not item_specs:
            raise RuntimeError("No beamlets selected")

        # --- Steps 1-2: one multi-beam engine, one BEV sample, one PB batch ---
        _t_eng = time.perf_counter()
        combined_seq = IonSpotBeamSequence.from_beams([seq[0] for seq in sequences])
        engine = _build_engine(args, lut, machine_config, res_zyx, ct_hu.shape, combined_seq, device, dtype)
        ssd_batch = [float(v) for v in ssd_values if v is not None]
        with torch.no_grad():
            resolved_offset = engine._resolve_rad_depth_offset(
                None,
                ssd_batch if len(ssd_batch) == len(ssd_values) else None,
                engine.number_of_beams,
            )
        _sync_timing()
        t_engine_total += time.perf_counter() - _t_eng

        _t_bev = time.perf_counter()
        crop_centers_hw = [
            (float(spec["iso_mm"][0]) / float(res_zyx[0]), float(spec["iso_mm"][2]) / float(res_zyx[2]))
            for spec in item_specs
        ]
        crops = [
            engine._build_dense_bev_crop(
                center_h=center_h,
                center_w=center_w,
                size_h=int(args.field_size[0]),
                size_w=int(args.field_size[1]),
            )
            for center_h, center_w in crop_centers_hw
        ]
        feature_centers_hw = [
            (center_h - float(crop["target_h_start"]), center_w - float(crop["target_w_start"]))
            for (center_h, center_w), crop in zip(crop_centers_hw, crops, strict=True)
        ]
        spr_bev, weq_bev_pb = engine._dense_forward_bev_multi_crop(spr_vol.unsqueeze(0), crops)
        weq_bev = weq_bev_pb
        if resolved_offset is not None:
            weq_bev = weq_bev + resolved_offset.view(-1, 1, 1, 1)

        with torch.no_grad():
            edep_bev = _compute_lattice_edep_bev_batch(
                engine,
                combined_seq,
                weq_bev_pb,
                feature_centers_hw,
                resolved_offset,
            )

        mat_id_bev = _engine_sample_bev_multi_crop(
            engine,
            material_id_patient.unsqueeze(0).float(),
            crops,
            mode="nearest",
        ).long()

        voxel_volume_mm3 = float(res_zyx[0]) * float(res_zyx[1]) * float(res_zyx[2])
        transport_step = float(engine.transport_step_mm or res_zyx[1])
        lateral_area_mm2 = voxel_volume_mm3 / transport_step
        dose_pb_bev = edep_bev * (MEV_CM2_PER_G_TO_GY_MM2 / lateral_area_mm2)
        feature_items = [
            _build_bev_features(
                spr_bev[idx:idx + 1], weq_bev[idx:idx + 1], dose_pb_bev[idx:idx + 1], mat_id_bev[idx:idx + 1],
                bev_crop_hw=int(args.bev_crop_hw),
                crop_center_hw=feature_centers_hw[idx],
                bev_crop_h=int(args.bev_crop_h),
                bev_crop_w=int(args.bev_crop_w),
                feature_set=args.bev_feature_set,
                peak_depth_mm=_peak_depth_mm(lut, item_specs[idx]["energy_mev"]),
            )
            for idx in range(len(item_specs))
        ]
        features = torch.cat([item[0] for item in feature_items], dim=0)
        dose_pb_5d = torch.cat([item[1] for item in feature_items], dim=0)
        valid_mask = torch.cat([item[2] for item in feature_items], dim=0)
        mat_id_5d = torch.cat([item[3] for item in feature_items], dim=0)
        fan_mask = torch.cat([item[4] for item in feature_items], dim=0)
        _sync_timing()
        t_bev_total += time.perf_counter() - _t_bev

        # --- Step 3: one correction-model call for the selected beamlet batch ---
        _t_model = time.perf_counter()
        aug_flip, aug_k = False, 0
        if not bool(args.no_augmentation) and training:
            symmetries = _D4_SYMMETRIES
            if args.model_kind == "repvgg_unet":
                symmetries = ((False, 0), (False, 2), (True, 0), (True, 2))
            aug_flip, aug_k = symmetries[int(torch.randint(0, len(symmetries), (1,)).item())]
            if aug_flip or aug_k:
                if args.model_kind == "repvgg_unet":
                    features = _d4_apply_features(features, aug_flip, aug_k)
                else:
                    # Legacy full-D4 path (90-degree rotations) would also need h/w
                    # offset-channel and sigma swap; not validated, kept as-is.
                    features = _d4_apply(features, aug_flip, aug_k)
                dose_pb_5d = _d4_apply(dose_pb_5d, aug_flip, aug_k)
                mat_id_5d = _d4_apply(mat_id_5d, aug_flip, aug_k)
        _, _, _, _, energy_t, sigma_t, _ = IonSpotBeamSequence.stack([combined_seq])
        energy_t = energy_t.to(device=device, dtype=dtype)[0, :, 0]
        sigma_t = sigma_t.to(device=device, dtype=dtype)[0, :, 0]
        model_features, original_model_depth = _pad_model_depth(features, int(args.model_depth))
        model_dose_pb, _ = _pad_model_depth(dose_pb_5d, int(args.model_depth))
        model_mat_id, _ = _pad_model_depth(mat_id_5d, int(args.model_depth))
        active_model = model if inference_model is None else inference_model
        mark_cudagraph_step()
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = active_model(
                model_features, model_dose_pb, None, None,
                material_id=model_mat_id, energy=energy_t, sigma_mm=sigma_t,
            )
        outputs = _trim_model_outputs(outputs, original_model_depth)
        corrected_bev_cropped = outputs["dose_hat"]
        if aug_flip or aug_k:
            corrected_bev_cropped = _d4_inverse(corrected_bev_cropped, aug_flip, aug_k)
        _sync_timing()
        t_model_total += time.perf_counter() - _t_model

        # --- Step 4: patient-space primary loss; optional BEV auxiliary loss ---
        _t_loss = time.perf_counter()
        use_bev_loss = bool(training and args.loss_space == "bev")
        need_bev_target = bool(
            use_bev_loss
            or (training and float(args.bev_deep_supervision_weight) > 0.0 and outputs.get("deep_supervision"))
        )
        bev_targets: list[torch.Tensor] = []
        for idx, spec in enumerate(item_specs):
            ref_arr, ref_bbox = _get_reference(spec["ref_path"], ref_futures, pinned_refs)
            bev_target = None
            if need_bev_target:
                ref_patient = _reference_patient_tensor(
                    ref_arr, ref_bbox, ct_hu.shape,
                    device=device, dtype=corrected_bev_cropped.dtype,
                )
                with torch.no_grad():
                    ref_bev = _engine_sample_bev_multi_crop(
                        engine,
                        ref_patient.unsqueeze(0),
                        [crops[idx]],
                        beam_indices=[idx],
                    )
                    bev_target = _crop_bev_volume(
                        ref_bev,
                        feature_centers_hw[idx],
                        int(args.bev_crop_h),
                        int(args.bev_crop_w),
                    ).unsqueeze(1)
                bev_targets.append(bev_target)

            if use_bev_loss:
                bev_mask = (valid_mask[idx, 0] & fan_mask[idx, 0])
                pred_c = torch.where(bev_mask, corrected_bev_cropped[idx, 0], 0.0)
                ref_c = torch.where(bev_mask, bev_target[0, 0], 0.0)
                depth_c = (
                    (torch.arange(pred_c.shape[0], device=device, dtype=pred_c.dtype) + 0.5)
                    * float(res_zyx[1])
                ).view(-1, 1, 1).expand_as(pred_c)
            else:
                _, D_bev, H_bev, W_bev = spr_bev[idx:idx + 1].shape
                h_src, h_dst, _h_target = _crop_slices(feature_centers_hw[idx][0], H_bev, int(args.bev_crop_h))
                w_src, w_dst, _w_target = _crop_slices(feature_centers_hw[idx][1], W_bev, int(args.bev_crop_w))
                corrected_bev_full = dose_pb_bev[idx:idx + 1].clone()
                corrected_bev_full[:, :, h_src, w_src] = corrected_bev_cropped[idx:idx + 1, 0, :, h_dst, w_dst]
                pred_slab, slab_h = _rotate_cropped_bev_to_patient_slab(engine, corrected_bev_full, crops[idx], idx)
                pred_slab = torch.where(
                    dose_mask_patient[slab_h, :, :],
                    pred_slab[0],
                    torch.zeros_like(pred_slab[0]),
                )
                if ref_bbox is not None:
                    z0, z1, y0, y1, x0, x1 = ref_bbox
                    if z0 >= z1:
                        continue
                    pred_c = pred_slab.new_zeros((z1 - z0, y1 - y0, x1 - x0))
                    overlap_start = max(z0, int(slab_h.start))
                    overlap_stop = min(z1, int(slab_h.stop))
                    if overlap_start < overlap_stop:
                        pred_c[overlap_start - z0:overlap_stop - z0] = pred_slab[
                            overlap_start - int(slab_h.start):overlap_stop - int(slab_h.start),
                            y0:y1,
                            x0:x1,
                        ]
                    ref_c = torch.from_numpy(ref_arr).to(device=device, dtype=pred_slab.dtype)
                    depth_c = depth_vol[z0:z1, y0:y1, x0:x1]
                else:
                    pred = torch.zeros_like(mass_vol)
                    pred[slab_h, :, :] = pred_slab
                    ref = torch.from_numpy(ref_arr).to(device=device, dtype=pred.dtype)
                    box = _dose_bbox(pred, ref)
                    if box is None:
                        continue
                    z0, z1, y0, y1, x0, x1 = box
                    pred_c = pred[z0:z1, y0:y1, x0:x1]
                    ref_c = ref[z0:z1, y0:y1, x0:x1]
                    depth_c = depth_vol[z0:z1, y0:y1, x0:x1]

            loss, terms = _loss(pred_c, ref_c, depth_c, args)
            loss_value = float(loss.detach())
            _add(acc, loss_value, terms)
            beamlet_records.append({
                "epoch": epoch, "patient": patient_id, "case_i": int(case_i),
                "beam": int(bi), "ray": spec["ri"], "beamlet": spec["li"],
                "energy_mev": spec["energy_mev"], "loss": loss_value, **terms,
            })
            done += 1
            if training:
                losses.append(loss / max(len(selected), 1))

        aux_loss = corrected_bev_cropped.new_zeros(())
        if need_bev_target and bev_targets:
            native_bev_target = torch.cat(bev_targets, dim=0)
            if aug_flip or aug_k:
                native_bev_target = _d4_apply(native_bev_target, aug_flip, aug_k)
                aux_valid_mask = _d4_apply(valid_mask & fan_mask, aug_flip, aug_k)
            else:
                aux_valid_mask = valid_mask & fan_mask
            aux_predictions = outputs.get("deep_supervision", ())
            aux_loss = _bev_deep_supervision_loss(
                aux_predictions,
                native_bev_target,
                aux_valid_mask,
            )
            if training and float(args.bev_deep_supervision_weight) > 0.0:
                losses.append(float(args.bev_deep_supervision_weight) * aux_loss)

        _attn_loss = corrected_bev_cropped.new_zeros(())
        _attn_loss_w = float(args.attn_loss_weight)
        if training and _attn_loss_w > 0.0:
            _attn_maps: list[torch.Tensor] = outputs.get("attn_maps", [])
            if _attn_maps:
                _res_mag = outputs["residual"].detach().abs()
                _attn_target = _res_mag / _res_mag.amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
                _attn_loss = sum(F.l1_loss(m, _attn_target) for m in _attn_maps) / len(_attn_maps)
                losses.append(_attn_loss_w * _attn_loss)

        # --- Identity (CycleGAN-style) regularizer ---------------------------------
        # Feed the MC ground truth as the "pencil beam" input. Every channel except the
        # dose (ch 7 of features + the standalone dose_pb arg) is CT/geometry-derived and
        # therefore identical, so we only swap the dose. The corrected output should then
        # reproduce the GT unchanged (residual ~ 0): penalise any deviation. This anchors
        # the corrector to be a no-op on already-correct dose, discouraging over-correction.
        _identity_loss = corrected_bev_cropped.new_zeros(())
        _id_weight = float(args.identity_loss_weight)
        _id_every = max(int(args.identity_loss_every), 1)
        # --identity-check runs the pass as a pure DIAGNOSTIC (validation included, no
        # gradient contribution): feed the MC ground truth in as the dose and see whether
        # the network returns it unchanged. A residual near zero means the correction is
        # already a no-op on real MC and an identity regulariser has nothing to teach.
        _id_check = bool(getattr(args, "identity_check", False))
        _id_active = (_id_weight > 0.0 and training) or _id_check

        def _run_identity_pass():
            """Second forward through the net with MC ground truth as the dose.

            Deferred until AFTER the primary backward on purpose: run inline, the primary
            graph is still alive while this one is being built, so peak memory is
            primary+identity and a 40 GB A100 OOMs at chunk 10 (job 540904). Called after
            the primary graph is freed, peak is max(primary, identity) instead, and this
            pass does its own backward so the gradients simply accumulate into .grad.
            """
            nonlocal _identity_loss
            id_feature_items = []
            for idx, spec in enumerate(item_specs):
                ref_arr, ref_bbox = _get_reference(spec["ref_path"], ref_futures, pinned_refs)
                ref_patient = _reference_patient_tensor(
                    ref_arr, ref_bbox, ct_hu.shape,
                    device=device, dtype=corrected_bev_cropped.dtype,
                )
                with torch.no_grad():
                    id_ref_bev = _engine_sample_bev_multi_crop(
                        engine, ref_patient.unsqueeze(0), [crops[idx]], beam_indices=[idx],
                    )
                id_feature_items.append(_build_bev_features(
                    spr_bev[idx:idx + 1], weq_bev[idx:idx + 1], id_ref_bev, mat_id_bev[idx:idx + 1],
                    bev_crop_hw=int(args.bev_crop_hw),
                    crop_center_hw=feature_centers_hw[idx],
                    bev_crop_h=int(args.bev_crop_h),
                    bev_crop_w=int(args.bev_crop_w),
                    feature_set=args.bev_feature_set,
                    peak_depth_mm=_peak_depth_mm(lut, item_specs[idx]["energy_mev"]),
                ))
            id_features = torch.cat([it[0] for it in id_feature_items], dim=0)
            id_dose = torch.cat([it[1] for it in id_feature_items], dim=0)
            id_mask = torch.cat([it[2] for it in id_feature_items], dim=0)
            id_mat = torch.cat([it[3] for it in id_feature_items], dim=0)
            if aug_flip or aug_k:
                id_features = _d4_apply_features(id_features, aug_flip, aug_k)
                id_dose = _d4_apply(id_dose, aug_flip, aug_k)
                id_mask = _d4_apply(id_mask, aug_flip, aug_k)
                id_mat = _d4_apply(id_mat, aug_flip, aug_k)
            id_features_p, _ = _pad_model_depth(id_features, int(args.model_depth))
            id_dose_p, _ = _pad_model_depth(id_dose, int(args.model_depth))
            id_mat_p, _ = _pad_model_depth(id_mat, int(args.model_depth))
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                id_outputs = active_model(
                    id_features_p, id_dose_p, None, None,
                    material_id=id_mat_p, energy=energy_t, sigma_mm=sigma_t,
                )
            id_outputs = _trim_model_outputs(id_outputs, original_model_depth)
            id_dose_hat = id_outputs["dose_hat"]
            # Peak-normalised masked L1 (same convention as the primary/aux losses),
            # computed in the (possibly augmented) BEV frame; flips/rotations are L1-invariant.
            id_peak = id_dose.detach().amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-3)
            id_err = (id_dose_hat - id_dose).abs() / id_peak
            _identity_loss = (id_err * id_mask).sum() / id_mask.sum().clamp_min(1)
            if training and _id_weight > 0.0:
                scaler.scale(_id_weight * _identity_loss * float(backward_scale)).backward()

        def _feature_probe():
            """Measure how far encoder activations move when ONLY the dose channel changes.

            Four inputs, item 0 of the batch, every non-dose channel identical:
              pb      the analytic pencil beam (what the net actually receives)
              corr    the net's own corrected output
              ref     the MC ground truth
              ref1017 the MC ground truth scaled by 1.017

            Reported as mean|a-b| / mean|act(ref)| per stage, in percent.

            Reading it:
              * d(corr,ref) < d(pb,ref)  -> activations track dose correctness, so a feature
                loss carries gradient about the thing we care about. Necessary condition.
              * all distances ~equal     -> the encoder is effectively blind to the dose
                channel and a feature loss would be a no-op.
              * d(ref1017,ref) ~ 0       -> EXPECTED, and the point of that arm: BEV channel 7
                is dose/dose.amax(), so a uniform scale cancels exactly. A feature loss is
                therefore structurally incapable of seeing an integral/scale bias -- which is
                precisely the halo deficit. It can only teach SHAPE.
            """
            idx = 0
            spec = item_specs[idx]
            ref_arr, ref_bbox = _get_reference(spec["ref_path"], ref_futures, pinned_refs)
            ref_patient = _reference_patient_tensor(
                ref_arr, ref_bbox, ct_hu.shape, device=device, dtype=corrected_bev_cropped.dtype,
            )
            ref_bev = _engine_sample_bev_multi_crop(
                engine, ref_patient.unsqueeze(0), [crops[idx]], beam_indices=[idx],
            )
            # Scatter the cropped correction back into a full-BEV volume so every variant
            # goes through the identical _build_bev_features path.
            _, _D_bev, H_bev, W_bev = spr_bev[idx:idx + 1].shape
            h_src, h_dst, _ = _crop_slices(feature_centers_hw[idx][0], H_bev, int(args.bev_crop_h))
            w_src, w_dst, _ = _crop_slices(feature_centers_hw[idx][1], W_bev, int(args.bev_crop_w))
            corr_bev = dose_pb_bev[idx:idx + 1].clone()
            corr_bev[:, :, h_src, w_src] = corrected_bev_cropped[idx:idx + 1, 0, :, h_dst, w_dst]

            # Halo-only perturbations REPRODUCE THE ACTUAL ERROR: measured on the shipped
            # model the core is ~-0.2% but the halo is -5% (-13% above 180 MeV). That is a
            # differential change, which peak-normalisation does NOT cancel -- unlike the
            # uniform `ref1017` arm, which is kept only as a negative control (channel 7 is
            # dose/dose.amax(), so a global scale is identically invisible).
            _rmax = ref_bev.amax().clamp_min(1e-12)
            _halo = ref_bev < 0.10 * _rmax
            def _scale_halo(vol, f):
                out = vol.clone()
                out[_halo] = out[_halo] * f
                return out

            variants = {
                "pb": dose_pb_bev[idx:idx + 1],
                "corr": corr_bev,
                "ref": ref_bev,
                "ref1017": ref_bev * 1.017,      # uniform: negative control, must be ~0
                "halo95": _scale_halo(ref_bev, 0.95),   # core intact, halo -5%  (shipped mean)
                "halo90": _scale_halo(ref_bev, 0.90),   # core intact, halo -10% (>180 MeV)
            }

            net = getattr(active_model, "_orig_mod", active_model)
            stage_names = ("native_stage", "equal_stage", "encoder_stage", "bottleneck_stage")
            captured: dict[str, torch.Tensor] = {}
            handles = []
            for sname in stage_names:
                mod = getattr(net, sname, None)
                if mod is None:
                    continue
                handles.append(mod.register_forward_hook(
                    lambda _m, _i, out, _n=sname: captured.__setitem__(_n, out.detach().float())
                ))
            acts: dict[str, dict[str, torch.Tensor]] = {}
            try:
                for vname, dose_bev in variants.items():
                    f_i, d_i, _v_i, m_i, _fan_i = _build_bev_features(
                        spr_bev[idx:idx + 1], weq_bev[idx:idx + 1], dose_bev, mat_id_bev[idx:idx + 1],
                        bev_crop_hw=int(args.bev_crop_hw),
                        crop_center_hw=feature_centers_hw[idx],
                        bev_crop_h=int(args.bev_crop_h), bev_crop_w=int(args.bev_crop_w),
                        feature_set=args.bev_feature_set,
                        peak_depth_mm=_peak_depth_mm(lut, spec["energy_mev"]),
                    )[:5]
                    f_p, _ = _pad_model_depth(f_i, int(args.model_depth))
                    d_p, _ = _pad_model_depth(d_i, int(args.model_depth))
                    m_p, _ = _pad_model_depth(m_i, int(args.model_depth))
                    captured.clear()
                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                        # sigma is (..,2) per item however it was flattened upstream; go
                        # through reshape(-1,2) so one item is selected, not one scalar.
                        active_model(f_p, d_p, None, None, material_id=m_p,
                                     energy=energy_t.reshape(-1)[idx:idx + 1],
                                     sigma_mm=sigma_t.reshape(-1, 2)[idx:idx + 1])
                    acts[vname] = dict(captured)
            finally:
                for h in handles:
                    h.remove()

            pairs = (("pb", "ref"), ("corr", "ref"), ("halo95", "ref"),
                     ("halo90", "ref"), ("ref1017", "ref"))
            cols = "  ".join(f"{a}->{b}".rjust(13) for a, b in pairs)
            print(f"[featprobe e{epoch} {patient_id} b{bi} {spec['energy_mev']:.1f}MeV] "
                  f"stage{'':>10s}{cols}", flush=True)
            for sname in stage_names:
                if sname not in acts.get("ref", {}):
                    continue
                den = acts["ref"][sname].abs().mean().clamp_min(1e-12)
                vals = []
                for a, b in pairs:
                    vals.append(float(((acts[a][sname] - acts[b][sname]).abs().mean() / den) * 100.0))
                print(f"[featprobe]   {sname:<18s}" + "  ".join(f"{v:12.4f}%" for v in vals), flush=True)

        if _id_active and (global_step % _id_every == 0) and not (training and _id_weight > 0.0):
            # Diagnostic-only (--identity-check): no gradient, so ordering is irrelevant.
            with torch.no_grad():
                _run_identity_pass()

        # --- Feature probe (--feature-probe) -----------------------------------------
        # Is a feature/perceptual loss meaningful here? Such a loss only teaches anything if
        # the encoder's activations actually respond to whether the dose is CORRECT. Feed
        # four inputs that differ ONLY in the dose (every other channel is CT/geometry
        # derived and identical) and measure how far the stage activations move.
        if bool(getattr(args, "feature_probe", False)) and not training:
            with torch.no_grad():
                _feature_probe()

        _sync_timing()
        t_loss_total += time.perf_counter() - _t_loss

        # Backward
        if training and losses:
            _t_bwd = time.perf_counter()
            total_loss = torch.stack(losses).sum() * float(backward_scale)
            loss_finite = bool(torch.isfinite(total_loss).detach().item())
            if bool(args.debug_divergence):
                _top = sorted(_dbg_act_max.items(), key=lambda kv: kv[1], reverse=True)[:1]
                _topname, _topval = (_top[0] if _top else ("-", 0.0))
                _nf_layers = [k for k, v in _dbg_act_nonfinite.items() if v > 0]
                _dh = outputs.get("dose_hat")
                _dhmax = float(_dh.detach().abs().max().item()) if isinstance(_dh, torch.Tensor) else float("nan")
                print(
                    f"[dbg e{epoch} {patient_id} b{bi}] loss_finite={int(loss_finite)} "
                    f"loss={float(total_loss.detach()) if loss_finite else 'nan'} "
                    f"max_act={_topval:.4g}@{_topname} dose_hat_max={_dhmax:.4g} "
                    f"nonfinite_act_layers={len(_nf_layers)}",
                    file=sys.stderr, flush=True,
                )
            if not loss_finite:
                if bool(args.debug_divergence):
                    print(f"\n[DIVERGENCE] non-finite loss at e{epoch} {patient_id} beam{bi}", file=sys.stderr, flush=True)
                    for i, lp in enumerate(losses):
                        print(f"[DIVERGENCE] loss_part[{i}]={lp.detach()}", file=sys.stderr, flush=True)
                    _dh = outputs.get("dose_hat"); _rs = outputs.get("residual")
                    for nm, tn in (("dose_hat", _dh), ("residual", _rs), ("corrected_bev_cropped", corrected_bev_cropped)):
                        if isinstance(tn, torch.Tensor):
                            d = tn.detach(); fin = torch.isfinite(d)
                            print(f"[DIVERGENCE] {nm}: nonfinite={int((~fin).sum())}/{d.numel()} "
                                  f"absmax={float(d[fin].abs().max()) if bool(fin.any()) else 'inf'}", file=sys.stderr, flush=True)
                    print("[DIVERGENCE] per-layer activations (sorted by |max|, top 25):", file=sys.stderr, flush=True)
                    for nm, v in sorted(_dbg_act_max.items(), key=lambda kv: kv[1], reverse=True)[:25]:
                        nf = _dbg_act_nonfinite.get(nm, 0)
                        print(f"[DIVERGENCE]   {v:12.4g}  nonfinite={nf:<6}  {nm}", file=sys.stderr, flush=True)
                raise FloatingPointError(
                    f"non-finite training loss for patient={patient_id} beam={bi} epoch={epoch}"
                )
            scaler.scale(total_loss).backward()
            # Primary graph is freed here; only now build the identity graph so the two
            # never coexist (see _run_identity_pass).
            if _id_weight > 0.0 and (global_step % _id_every == 0):
                _run_identity_pass()
            _sync_timing()
            t_backward_total += time.perf_counter() - _t_bwd

        rec = {
            "epoch": epoch, "patient": patient_id, "beam": bi,
            "n_beamlets": done, "n_beamlets_total": len(items),
            "compute_s": round(time.perf_counter() - t0, 1),
            "t_setup_ms": t_setup_ms,
            "t_engine_ms": round(t_engine_total * 1e3, 1),
            "t_bev_ms": round(t_bev_total * 1e3, 1),
            "t_model_ms": round(t_model_total * 1e3, 1),
            "t_loss_ms": round(t_loss_total * 1e3, 1),
            "t_backward_ms": round(t_backward_total * 1e3, 1),
            "bev_deep_supervision": float(aux_loss.detach()),
            "beamlet_records": beamlet_records,
            **_mean(acc, done),
        }
        # Only record terms that are actually switched on. Logging a column that is
        # identically zero for every step of every run is noise in the history and in
        # wandb; these come back automatically the moment their weight is nonzero.
        if float(args.attn_loss_weight) > 0.0:
            rec["attn_loss"] = float(_attn_loss.detach())
            rec["loss"] += float(args.attn_loss_weight) * rec["attn_loss"]
        if _id_active:
            rec["identity_loss"] = float(_identity_loss.detach())
            if float(args.identity_loss_weight) > 0.0:
                rec["loss"] += float(args.identity_loss_weight) * rec["identity_loss"]
        rec["loss"] += float(args.bev_deep_supervision_weight) * rec["bev_deep_supervision"]
        return rec

    # -------------------------------------------------------------------
    # Validation plotting
    # -------------------------------------------------------------------

    @torch.no_grad()
    def plot_validation_worst(epoch: int, records: list[dict[str, Any]]):
        nonlocal fixed_plot_keys
        n_worst = max(int(args.plot_worst_beamlets), 0)
        if n_worst <= 0 or not records:
            return None
        import matplotlib
        matplotlib.use("Agg")

        if fixed_plot_keys is None:
            fixed_plot_keys = [
                _record_key(record)
                for record in sorted(records, key=lambda r: float(r[args.plot_worst_metric]), reverse=True)[:n_worst]
            ]
            (args.output_dir / "fixed_validation_plot_panels.json").write_text(
                json.dumps([
                    {"patient": p, "beam": b, "ray": r, "beamlet": bl}
                    for p, b, r, bl in fixed_plot_keys
                ], indent=2), encoding="utf-8",
            )

        record_by_key = {_record_key(r): r for r in records}
        worst = [record_by_key[key] for key in fixed_plot_keys if key in record_by_key]
        if len(worst) < n_worst:
            existing = {_record_key(r) for r in worst}
            worst.extend(
                r for r in sorted(records, key=lambda r: float(r[args.plot_worst_metric]), reverse=True)
                if _record_key(r) not in existing
            )
            worst = worst[:n_worst]

        out_dir = args.output_dir / f"val_epoch{epoch:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        panel_index: list[dict[str, Any]] = []
        wandb_images: list[Any] = []

        for rec_i, record in enumerate(worst):
            case = val_cases[int(record["case_i"])]
            patient_id, plan, dose_dir, ct_hu, origin_zyx, res_zyx = case
            bi = int(record["beam"])
            ri = int(record["ray"])
            li = int(record["beamlet"])
            beam_json = plan["beams"][bi]
            ray_json = beam_json["rays"][ri]
            e_ref = _beam_mean_energy(plan, bi)
            hu_t = torch.from_numpy(ct_hu).to(device=device, dtype=dtype)
            spr_vol, mass_vol = spr_and_mass_density(hu_t, e_ref, hu_to_density)
            ref_paths = _reference_paths_for_selection(dose_dir, beam_json, bi, [ri], li)
            if len(ref_paths) != 1:
                continue
            energy_mev = float(ray_json["beamlets"][li]["energy"])
            seq, ssd_mm = _make_ray_sequence(
                plan=plan, beam_parameters=beam_parameters, ct_hu=ct_hu,
                origin_zyx=origin_zyx, resolution_zyx=res_zyx,
                beam_index=bi, ray_index=ri, beamlet_index=li,
                particles_per_beamlet=float(args.particles_per_beamlet),
                gantry_offset_deg=0.0,
                skin_hu_threshold=float(args.skin_hu_threshold),
                sigma_mode=args.sigma_mode,
                bams_to_iso_dist_mm=float(args.bams_to_iso_dist_mm),
                lut=lut, device=device, dtype=dtype,
            )
            # Run full pipeline for plot
            iso_mm = tuple((_xyz_to_zyx(ray_json["ray_target"]) - origin_zyx).tolist())
            crop_center_hw = (float(iso_mm[0]) / float(res_zyx[0]), float(iso_mm[2]) / float(res_zyx[2]))
            material_id_patient = _material_id_from_hu(hu_t)

            engine = _build_engine(args, lut, machine_config, res_zyx, ct_hu.shape, seq, device, dtype)
            resolved_offset = engine._resolve_rad_depth_offset(None, ssd_mm, engine.number_of_beams)
            from pydose_rt.data.ion_beam import IonSpotBeamSequence
            (sp, sw, sli, sm, le, ls, lm) = IonSpotBeamSequence.stack([seq])

            engine._dense_bev_crop = engine._build_dense_bev_crop(
                center_h=crop_center_hw[0],
                center_w=crop_center_hw[1],
                size_h=int(args.field_size[0]),
                size_w=int(args.field_size[1]),
            )
            feature_center_hw = _engine_local_crop_center_hw(engine, crop_center_hw)
            spr_bev, weq_bev = _engine_density_weq_bev(engine, spr_vol.unsqueeze(0))
            weq_bev_pb = weq_bev

            B_weq = 1
            G_weq = engine.number_of_beams
            D_weq = weq_bev.shape[1]
            missing_weq = torch.zeros((B_weq, G_weq), device=device, dtype=dtype)
            weq_bev = weq_bev.view(B_weq, G_weq, D_weq, weq_bev.shape[2], weq_bev.shape[3])
            weq_bev = weq_bev + missing_weq[:, :, None, None, None]
            if resolved_offset is not None:
                weq_bev = weq_bev + resolved_offset.view(1, G_weq, 1, 1, 1)
            weq_bev = weq_bev.view(B_weq * G_weq, D_weq, weq_bev.shape[3], weq_bev.shape[4])

            edep_bev = _compute_lattice_edep_bev(engine, seq, weq_bev_pb, feature_center_hw, resolved_offset)

            mat_id_bev = _engine_sample_bev(
                engine,
                material_id_patient.unsqueeze(0).float(),
                mode="nearest",
            ).long()
            voxel_volume_mm3 = float(res_zyx[0]) * float(res_zyx[1]) * float(res_zyx[2])
            transport_step = float(engine.transport_step_mm or res_zyx[1])
            lateral_area_mm2 = voxel_volume_mm3 / transport_step
            dose_pb_bev = edep_bev * (MEV_CM2_PER_G_TO_GY_MM2 / lateral_area_mm2)

            features, dose_pb_5d, valid_mask, mat_id_5d, fan_mask = _build_bev_features(
                spr_bev, weq_bev, dose_pb_bev, mat_id_bev,
                bev_crop_hw=int(args.bev_crop_hw),
                crop_center_hw=feature_center_hw,
                bev_crop_h=int(args.bev_crop_h),
                bev_crop_w=int(args.bev_crop_w),
                feature_set=args.bev_feature_set,
                peak_depth_mm=_peak_depth_mm(lut, energy_mev),
            )
            energy_t = torch.tensor([energy_mev], device=device, dtype=dtype)
            sigma_t = ls.to(device=device, dtype=dtype)[0, 0, 0].view(1, 2)
            model_features, original_model_depth = _pad_model_depth(features, int(args.model_depth))
            model_dose_pb, _ = _pad_model_depth(dose_pb_5d, int(args.model_depth))
            model_mat_id, _ = _pad_model_depth(mat_id_5d, int(args.model_depth))
            mark_cudagraph_step()
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(
                    model_features, model_dose_pb, None, None,
                    material_id=model_mat_id, energy=energy_t, sigma_mm=sigma_t,
                )
            outputs = _trim_model_outputs(outputs, original_model_depth)

            H_bev, W_bev = spr_bev.shape[2], spr_bev.shape[3]
            h_src, h_dst, _h_target = _crop_slices(feature_center_hw[0], H_bev, int(args.bev_crop_h))
            w_src, w_dst, _w_target = _crop_slices(feature_center_hw[1], W_bev, int(args.bev_crop_w))
            corrected_bev_full = dose_pb_bev.clone()
            corrected_bev_full[:, :, h_src, w_src] = outputs["dose_hat"].squeeze(1)[:, :, h_dst, w_dst]
            corrected_bev_full = _engine_expand_bev_crop(engine, corrected_bev_full)
            pred_t = engine.rotation_layer(corrected_bev_full.unsqueeze(1)).sum(dim=1)[0]
            pred = torch.where(
                patient_dose_mask(mass_vol), pred_t, torch.zeros_like(pred_t)
            ).detach().cpu().numpy()

            ref_path = ref_paths[0]
            ref = _read_reference_dose(ref_path).astype(np.float32, copy=False)
            com_delta = _center_of_mass_mm(pred, res_zyx) - _center_of_mass_mm(ref, res_zyx)
            ray_axis = _ray_axis_zyx(ray_json)
            com_parallel = float(np.dot(com_delta, ray_axis))
            com_lateral = float(np.linalg.norm(com_delta - com_parallel * ray_axis))
            scale = 1.0
            scaled_pred = pred * scale
            panel_name = f"{patient_id}_B{bi}_R{ri}_L{li}"
            out_path = out_dir / f"panel{rec_i:02d}_{panel_name}.png"
            axial_check_path = out_dir / f"panel{rec_i:02d}_{panel_name}_axial_check.png"
            _plot_total_comparison(
                patient_id=(
                    f"{patient_id} B{bi} R{ri} L{li} {float(energy_mev):.1f}MeV "
                    f"corrected e{epoch} loss_space={args.loss_space} "
                    f"loss={float(record['loss']):.4f} "
                    f"|dPk|={float(record.get('peak_abs_shift_mm', float('nan'))):.2f}mm "
                    f"dCOM_parallel={com_parallel:+.1f}mm dCOM_lateral={com_lateral:.1f}mm"
                ),
                ct=ct_hu, ref_total=ref, pred_total=scaled_pred, scale=scale,
                mask_fraction=0.0, display_percentile=99.5, out_path=out_path,
            )
            _plot_axial_prediction_check(
                out_path=axial_check_path,
                ref=ref,
                pred=scaled_pred,
                ct=ct_hu,
                title=(
                    f"{patient_id} B{bi} R{ri} L{li} {float(energy_mev):.1f}MeV "
                    f"scale={scale:.6g}"
                ),
            )
            panel_index.append({
                "panel": int(rec_i), "patient": patient_id,
                "beam": bi, "ray": ri, "beamlet": li,
                "energy_mev": float(energy_mev), "path": str(out_path),
                "axial_check_path": str(axial_check_path),
                "scale": scale, "dcom_parallel_mm": com_parallel,
                "dcom_lateral_mm": com_lateral,
                "loss": float(record["loss"]),
                "mae_pct": float(record["mae_pct"]),
                "mae_high10_pct": float(record["mae_high10_pct"]),
                "idd_z": float(record.get("idd_z", float("nan"))),
                # optional: only present when w_peak > 0 (see LOSS_KEYS note)
                **({"peak_abs_shift_mm": float(record["peak_abs_shift_mm"])}
                   if "peak_abs_shift_mm" in record else {}),
            })
            if use_wandb:
                wandb_images.append(wandb.Image(str(out_path), caption=panel_name))
            del engine, spr_vol, mass_vol

        index_path = out_dir / "index.json"
        index_path.write_text(json.dumps(panel_index, indent=2), encoding="utf-8")
        if use_wandb:
            wandb.log({"val/beamlet_panels": wandb_images}, step=global_step)
        print(f"    validation panels -> {out_dir}")
        return out_dir

    # -------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------

    @torch.no_grad()
    def validate(
        epoch: int,
        inference_model: nn.Module | None = None,
        metric_prefix: str = "val",
        make_plots: bool = True,
    ):
        if not fixed_val:
            return
        eval_model = model if inference_model is None else inference_model
        eval_model.eval()
        acc: dict[str, float] = {}
        beamlet_records: list[dict[str, Any]] = []
        n = 0
        for case_i, bi, selected_items in fixed_val:
            rec = run_beam(
                val_cases[case_i], case_i, bi, epoch, training=False,
                selected_items=selected_items, inference_model=eval_model,
            )
            for key in (*LOSS_KEYS, *_OPTIONAL_KEYS):
                if key in rec:
                    acc[key] = acc.get(key, 0.0) + float(rec[key])
            beamlet_records.extend(rec["beamlet_records"])
            n += 1
        val_rec = {
            "epoch": epoch, "step": global_step, "n_beams": n,
            "weights": "ema" if metric_prefix == "val_ema" else "raw",
            **_mean(acc, n),
        }
        val_history.append(val_rec)
        safe_prefix = metric_prefix.replace("/", "_")
        (args.output_dir / f"{safe_prefix}_beamlets_epoch{epoch:03d}.json").write_text(
            json.dumps(beamlet_records, indent=2), encoding="utf-8",
        )
        print(
            f"[{metric_prefix} e{epoch} s{global_step}] loss={val_rec['loss']:.6g} "
            f"dose={val_rec['dose']:.6g} raw={val_rec['dose_raw']:.6g} "
            f"mae%={val_rec['mae_pct']:.3g} "
            f"mae_ref>10%={val_rec['mae_high10_pct']:.3g} "
            f"iddz={val_rec.get('idd_z', float('nan')):.4g} "
            f"int_ratio={val_rec.get('integral_ratio', float('nan')):.4f}"
            + _optional_terms_str(val_rec)
        )
        if use_wandb:
            wandb.log({f"{metric_prefix}/{k}": val_rec[k]
                       for k in (*LOSS_KEYS, *_OPTIONAL_KEYS) if k in val_rec}, step=global_step)
        if metric_prefix == best_source:
            cur = val_rec.get(args.best_metric)
            if _is_new_best(best_state, cur):
                best_state.update(value=float(cur), epoch=epoch, step=global_step,
                                  metric=args.best_metric, source=best_source, pending=True)
                print(f"[best] {best_source}/{args.best_metric}={cur:.6g} "
                      f"@ e{epoch} s{global_step} -> best.pt")
        if make_plots and max(int(args.plot_every_epochs), 0) > 0 and epoch % int(args.plot_every_epochs) == 0:
            plot_validation_worst(epoch, beamlet_records)
        model.train()

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------

    if args.validate_before_training and resume_epoch == 0:
        validate(0)
        if ema_model is not None:
            with ema_weights_in_training_model():
                validate(0, metric_prefix="val_ema", make_plots=False)
        save_checkpoint(0)

    if args.plot_only:
        validate(resume_epoch)
        if ema_model is not None:
            with ema_weights_in_training_model():
                validate(resume_epoch, metric_prefix="val_ema", make_plots=False)
        ref_pool.shutdown(wait=True)
        if use_wandb:
            wandb.finish()
        return

    for epoch in range(1, int(args.epochs) + 1):
        if epoch <= resume_epoch:
            continue
        plan_rng = random.Random(int(args.seed) + epoch)
        if args.beam_sampling == "full":
            epoch_plan = list(full_epoch_units)
            plan_rng.shuffle(epoch_plan)
        else:
            epoch_plan = [
                (*plan_rng.choice(inventory), None)
                for _ in range(max(int(args.steps_per_epoch), 1))
            ]
        accumulated_records: list[dict[str, Any]] = []
        for micro_index, (case_i, bi, selected_items) in enumerate(epoch_plan):
            group_start = (micro_index // grad_accum_steps) * grad_accum_steps
            group_end = min(group_start + grad_accum_steps, len(epoch_plan))
            group_size = group_end - group_start
            if micro_index == group_start:
                optimizer.zero_grad(set_to_none=True)
                accumulated_records = []
            model.train()
            rec = run_beam(
                train_cases[case_i], case_i, bi, epoch, training=True,
                selected_items=selected_items, backward_scale=1.0 / float(group_size),
            )
            rec.pop("beamlet_records", None)
            micro_step += 1
            rec["micro_step"] = micro_step
            accumulated_records.append(rec)
            if micro_index + 1 != group_end:
                continue
            grad_norm = float("nan")
            grad_clip_active = 0.0
            if float(args.grad_clip) > 0.0:
                scaler.unscale_(optimizer)
                grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                grad_norm = float(grad_norm_t.detach())
                grad_clip_active = float(math.isfinite(grad_norm) and grad_norm > float(args.grad_clip))
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after_step = scaler.get_scale()
            if ema_model is not None and scaler.get_scale() >= scale_before_step:
                update_ema()
            if scheduler is not None:
                scheduler.step()
            global_step += 1
            averaged_keys = (*LOSS_KEYS, "bev_deep_supervision", "attn_loss", "identity_loss")
            log_rec = dict(accumulated_records[-1])
            for key in averaged_keys:
                if all(key in item for item in accumulated_records):
                    log_rec[key] = sum(float(item[key]) for item in accumulated_records) / len(accumulated_records)
            for key in ("compute_s", "t_engine_ms", "t_bev_ms", "t_model_ms", "t_loss_ms", "t_backward_ms", "t_cleanup_ms"):
                if all(key in item for item in accumulated_records):
                    log_rec[key] = sum(float(item[key]) for item in accumulated_records)
            log_rec["n_beamlets"] = sum(int(item["n_beamlets"]) for item in accumulated_records)
            log_rec["n_beamlets_total"] = sum(int(item["n_beamlets_total"]) for item in accumulated_records)
            log_rec["step"] = global_step
            log_rec["grad_accum_steps"] = len(accumulated_records)
            log_rec["grad_norm"] = grad_norm
            log_rec["grad_clip_active"] = grad_clip_active
            log_rec["amp_scale"] = float(scale_after_step)
            log_rec["amp_step_skipped"] = float(scale_after_step < scale_before_step)
            history.append(log_rec)
            print(
                f"[e{epoch} s{global_step}] {log_rec['patient']} beam{log_rec['beam']} "
                f"loss={log_rec['loss']:.6g} dose={log_rec['dose']:.6g} raw={log_rec['dose_raw']:.6g} "
                f"mae%={log_rec['mae_pct']:.3g} "
                f"mae_ref>10%={log_rec['mae_high10_pct']:.3g} "
                f"iddz={log_rec.get('idd_z', float('nan')):.4g} "
                f"bev_aux={log_rec['bev_deep_supervision']:.4g} "
                f"grad={log_rec['grad_norm']:.3g} clip={int(log_rec['grad_clip_active'])} "
                f"amp_skip={int(log_rec['amp_step_skipped'])} accum={len(accumulated_records)} "
                f"({log_rec['n_beamlets']}/{log_rec['n_beamlets_total']} bl, {log_rec['compute_s']}s "
                f"eng={log_rec['t_engine_ms']}ms bev={log_rec['t_bev_ms']}ms "
                f"model={log_rec['t_model_ms']}ms loss={log_rec['t_loss_ms']}ms "
                f"bwd={log_rec['t_backward_ms']}ms)"
                + (
                    f" peakGB={torch.cuda.max_memory_allocated() / 1e9:.2f}"
                    if device.type == "cuda"
                    else ""
                )
            )
            if use_wandb:
                wandb.log(
                    {f"train/{k}": log_rec[k] for k in (*LOSS_KEYS, *_OPTIONAL_KEYS) if k in log_rec}
                    | {
                        "train/compute_s": log_rec["compute_s"],
                        "train/bev_deep_supervision": log_rec["bev_deep_supervision"],
                        "train/t_engine_ms": log_rec["t_engine_ms"],
                        "train/t_bev_ms": log_rec["t_bev_ms"],
                        "train/t_model_ms": log_rec["t_model_ms"],
                        "train/t_loss_ms": log_rec["t_loss_ms"],
                        "train/t_backward_ms": log_rec["t_backward_ms"],
                        "train/grad_accum_steps": len(accumulated_records),
                        "train/grad_norm": log_rec["grad_norm"],
                        "train/grad_clip_active": log_rec["grad_clip_active"],
                        "train/amp_scale": log_rec["amp_scale"],
                        "train/amp_step_skipped": log_rec["amp_step_skipped"],
                    },
                    step=global_step,
                )
            if args.checkpoint_every_steps and global_step % int(args.checkpoint_every_steps) == 0:
                save_checkpoint(epoch, step_checkpoint=True)
            if args.max_steps is not None and global_step >= int(args.max_steps):
                save_checkpoint(epoch)
                ref_pool.shutdown(wait=True)
                if use_wandb:
                    wandb.finish()
                return

        if int(args.validate_every_epochs) > 0 and epoch % int(args.validate_every_epochs) == 0:
            validate(epoch)
            if ema_model is not None:
                with ema_weights_in_training_model():
                    validate(epoch, metric_prefix="val_ema", make_plots=False)
        save_checkpoint(epoch)

    ref_pool.shutdown(wait=True)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
