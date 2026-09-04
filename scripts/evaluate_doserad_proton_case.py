"""Evaluate a DoseRAD proton case against the DoseRAD 2026 competition metrics.

Single-case mode (existing behaviour):
    python evaluate_doserad_proton_case.py --case-dir <case_dir>

Batch mode (new):
    python evaluate_doserad_proton_case.py --cases-dir /path/to/all/cases [--skip-existing]
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import math
import os
import time
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pydosert-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))          # scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))   # project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from doserad_proton_utils import (
    ROOT,
    DEFAULT_MC_FIT_3D_OPT_MAT,
    _assert_case_is_complete,
    _doserad_density_from_hu,
    _expected_dose_filename,
    _load_json,
    _load_reference_paths_sum,
    _make_beamlet_batch_sequence,
    _origin_zyx,
    _plot_total_comparison,
    _read_reference_dose,
    _reference_paths_for_selection,
    _resolution_zyx,
    _resolve_case_files,
    _robust_positive_max,
    _selected_beamlets,
    _selected_ray_indices,
    _xyz_to_zyx,
)

from pydose_rt.data.machine_config import MachineConfig
from pydose_rt.engine.ion_dose_engine import IonDoseEngine
from pydose_rt.physics.kernels.ion_lut import PyRadPlanIonLUT
from pydose_rt.physics.spr import patient_dose_mask, spr_and_mass_density
from pydose_rt.sparse.ions import IonSparseHooks
from pydose_rt.utils.gamma import local_gamma_pass_rate as _torch_gamma_pass_rate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # --- case / paths ---
    parser.add_argument("--case-dir", type=Path, default=None, help="Single DoseRAD case directory (mutually exclusive with --cases-dir)")
    parser.add_argument("--cases-dir", type=Path, default=None, help="Directory of case subdirectories; evaluates all of them (batch mode)")
    parser.add_argument("--beam-params-path", type=Path, default=Path("./example_data/beam_parameters.json"), help="Path to DoseRAD beam_parameters.json")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for metrics JSONs and (optionally) figures")
    parser.add_argument("--pyradplan-machine-mat", type=Path, default=DEFAULT_MC_FIT_3D_OPT_MAT, help="pyRadPlan machine base data .mat")
    # --- engine ---
    parser.add_argument("--device", type=str, default=None, help="Torch device, default: cuda if available else cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=("float16", "float32", "float64"))
    parser.add_argument("--particles-per-beamlet", type=float, default=1_000_000.0)
    parser.add_argument("--sigma-mode", type=str, default="beam_params", choices=("focus", "beam_params", "point_source"))
    parser.add_argument("--lateral-model", type=str, default="gauss_double", choices=("gauss", "gauss_double"))
    parser.add_argument("--split-mode", type=str, default="split", choices=("single", "split"),
                        help="Override ion PB kernel mode. Defaults to the engine setting.")
    parser.add_argument("--heterogeneous-mcs", action="store_true", default=True,
                        help="Enable Fermi-Eyges/Kanematsu heterogeneity-aware MCS lateral scattering (Fuchs dH)")
    parser.add_argument("--material-radiation-length", action="store_true",
                        help="Use per-material radiation length (DoseRAD compositions) for MCS; needs --heterogeneous-mcs")
    parser.add_argument("--transport-step-mm", type=float, default=None)
    parser.add_argument("--air-offset-correction", type=lambda s: s.lower() not in ("0", "false", "no", "off"), default=True)
    parser.add_argument("--fit-air-offset-mm", type=float, default=0.0)
    parser.add_argument("--bams-to-iso-dist-mm", type=float, default=1000.0)
    parser.add_argument("--skin-hu-threshold", type=float, default=-500.0)
    parser.add_argument("--gantry-offset-deg", type=float, default=0.0)
    parser.add_argument(
        "--spr",
        type=lambda s: s.lower() not in ("0", "false", "no", "off"),
        default=True,
        help=(
            "Use deterministic SPR as the PB stopping volume while keeping physical "
            "density for MeV->Gy conversion. --spr=0 restores the legacy raw-density path."
        ),
    )
    parser.add_argument(
        "--dense-field-size",
        type=int,
        nargs=2,
        default=(25, 74),
        help=(
            "Dense BEV PB field size in lateral voxels. Defaults to the dense "
            "correction crop size."
        ),
    )
    # --- selection (single-case mode only) ---
    parser.add_argument("--beam-index", type=int, default=None, help="Process a single 0-based beam index instead of all beams")
    parser.add_argument("--beam-stride", type=int, default=1,
                        help="Score every Nth beam instead of all of them. Level-1 is a nanmean over "
                        "beamlets, so a strided subset estimates it unbiasedly at 1/N the runtime; "
                        "the same beams are picked every run, so comparisons stay paired. Ignored "
                        "with --beam-index.")
    parser.add_argument("--ray-index", type=int, default=None)
    parser.add_argument("--beamlet-index", type=int, default=None)
    # --- evaluation ---
    parser.add_argument("--prescription-dose-gy", type=float, default=None, help="Prescription dose in Gy; auto-estimated (95th %%ile of ref) if omitted")
    parser.add_argument("--mask-threshold-fraction", type=float, default=0.0, help="Reference-dose threshold for global metrics mask; 0 uses union nonzero support")
    parser.add_argument("--gamma-dose-threshold", type=float, default=1.0, help="Gamma dose-difference criterion in %% (local)")
    parser.add_argument("--gamma-distance-threshold", type=float, default=1.0, help="Gamma distance-to-agreement criterion in mm")
    parser.add_argument("--gamma-interp-fraction", type=int, default=5, help="Repo torch gamma interpolation fraction (5 = 0.2mm steps at 1mm voxels)")
    parser.add_argument("--gamma-random-subset", type=int, default=None, help="Evaluate gamma on a random subset of reference points (approximate, much faster)")
    parser.add_argument("--skip-gamma", action="store_true", help="Skip local gamma computation (can be slow on large volumes)")
    parser.add_argument("--range-map", action="store_true", help="Compute per-BEV-column distal-R80 range-difference map (pred-ref) per beam and save it.")
    parser.add_argument("--per-beamlet-range", action="store_true", help="Per-beamlet distal-R80 range diff (single-energy, unambiguous) vs per-beamlet MC reference, over the lattice. Requires --beam-index.")
    parser.add_argument("--write-beamlets", action="store_true", help="With --per-beamlet-range: persist each per-beamlet pred volume bbox-cropped (.npz) under <out_dir>/beamlets/.")
    parser.add_argument(
        "--minimum-cutoff",
        type=float,
        default=0.0,
        help="Emulate the challenge output contract: zero predicted voxels <= this ABSOLUTE "
        "dose before scoring (container/inference.py does this from output_info.minimum_cutoff, "
        "the local eval otherwise does not). The masked MAE is blind to it (its mask starts at "
        "10%% of the beamlet peak) but the IDD integrates the whole volume, so a cutoff at ~1%% "
        "of peak alone costs ~0.003 IDD -- the size of the entire leaderboard spread.",
    )
    parser.add_argument(
        "--speed-only",
        action="store_true",
        help="Benchmark only synchronized PB/correction compute. Skips reference MHA loading, metrics, gamma, and figures.",
    )
    # --- output ---
    parser.add_argument("--reference-io-workers", type=int, default=min(16, max(1, os.cpu_count() or 1)))
    parser.add_argument("--display-percentile", type=float, default=99.5)
    parser.add_argument("--skip-figures", action="store_true", help="Skip all PNG generation")
    parser.add_argument("--skip-beam-plots", action="store_true", default=True, help="Skip per-beam comparison plots (still saves total comparison)")
    # --- batch mode ---
    parser.add_argument("--skip-existing", action="store_true", help="Batch mode: skip cases whose metrics JSON already exists")
    # --- correction model (optional dense BEV hook) ---
    parser.add_argument(
        "--correction-checkpoint",
        type=Path,
        default=None,
        help="Dense BEV correction checkpoint to load, e.g. model_ct/latest_ema.pt. "
        "If omitted, the uncorrected analytic pencil beam is scored.",
    )
    parser.add_argument(
        "--dense-tta",
        action="store_true",
        help="Average the correction model over the four lateral reflections at inference. "
        "Every model here was trained with --no-augmentation, so the net is not "
        "reflection-equivariant and this can hurt as easily as help -- measure, do not assume. "
        "Costs 4x the dense forward (which is not the runtime bottleneck; BEV sampling is).",
    )
    parser.add_argument(
        "--no-correction",
        action="store_true",
        help="Ignore --correction-checkpoint and score the plain analytic pencil beam. "
        "Use this to attribute error between the baseline and the correction net.",
    )
    parser.add_argument(
        "--n-per-dim",
        type=int,
        default=None,
        help="Override ion_dose_engine.N_PER_DIM (sub-beams per lateral dimension; "
        "total per beamlet is the square). Engine default is 9 (81 sub-beams). Raising "
        "it costs roughly quadratic engine time and is the main accuracy-for-speed lever.",
    )
    parser.add_argument(
        "--dense-bev-crop-hw",
        type=int,
        default=None,
        help="Override dense correction BEV crop half-width. Defaults to checkpoint args['bev_crop_hw'], then 64.",
    )
    parser.add_argument(
        "--dense-hook-batch-items",
        type=int,
        default=6,
        help="Max number of dense correction BEV samples to run through the checkpoint model at once.",
    )
    parser.add_argument(
        "--dense-hook-amp",
        action="store_true",
        help="Run dense BEV correction model inference under CUDA float16 autocast.",
    )
    parser.add_argument(
        "--dense-isotropic-crop",
        action="store_true",
        help="Keep the dense BEV correction crop square in voxel space instead of matching physical lateral width.",
    )
    parser.add_argument(
        "--compile-correction-model",
        action="store_true",
        help="Compile the correction model with torch.compile. Usually slower for one-shot case evaluation.",
    )
    parser.add_argument(
        "--compile-dynamic-shapes",
        action="store_true",
        help="Use dynamic-shape torch.compile for the correction model. Default is static-shape compile.",
    )
    parser.add_argument(
        "--compile-cache-path",
        type=Path,
        default=None,
        help="Load/save torch.compile cache artifacts for the correction model at this path.",
    )
    parser.add_argument(
        "--save-compile-cache",
        action="store_true",
        help="Save torch.compile cache artifacts after evaluation. Requires --compile-cache-path.",
    )
    parser.add_argument(
        "--profile-dense-timing",
        action="store_true",
        help="Print synchronized dense engine and correction-hook phase timings.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def _compute_idd(
    dose: np.ndarray,
    resolution_zyx: tuple[float, float, float],
    gantry_angle_deg: float,
    origin_offset_vox: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """Project dose onto the beam axis and return (bin_centres_mm, idd_values).

    ``origin_offset_vox`` shifts the voxel grid origin (z,y,x) so a bbox-cropped sub-volume
    keeps its absolute depth coordinate — required to compare R80 across pred/ref crops
    that have different bboxes."""
    theta = math.radians(gantry_angle_deg)
    axis_zyx = np.array([0.0, math.cos(theta), -math.sin(theta)], dtype=np.float64)

    Z, Y, X = dose.shape
    oz, oy, ox = origin_offset_vox
    z_coords = (np.arange(Z, dtype=np.float64) + oz) * float(resolution_zyx[0])
    y_coords = (np.arange(Y, dtype=np.float64) + oy) * float(resolution_zyx[1])
    x_coords = (np.arange(X, dtype=np.float64) + ox) * float(resolution_zyx[2])

    depths = (
        z_coords[:, None, None] * axis_zyx[0]
        + y_coords[None, :, None] * axis_zyx[1]
        + x_coords[None, None, :] * axis_zyx[2]
    )

    d_min = float(depths.min())
    d_max = float(depths.max())
    bin_edges = np.arange(d_min, d_max + 1.0, 1.0)
    n_bins = len(bin_edges) - 1
    if n_bins < 1:
        return np.array([0.5 * (d_min + d_max)]), np.array([float(dose.sum())])

    bin_indices = np.clip(np.digitize(depths.ravel(), bin_edges) - 1, 0, n_bins - 1)
    idd = np.zeros(n_bins, dtype=np.float64)
    np.add.at(idd, bin_indices, dose.ravel().astype(np.float64))

    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centres, idd


def _idd_distance(
    pred_dose: np.ndarray,
    ref_dose: np.ndarray,
    resolution_zyx: tuple[float, float, float],
    gantry_angle_deg: float,
) -> float:
    """Level 1.2: RMS IDD curve distance normalised by peak reference IDD."""
    bins_ref, idd_ref = _compute_idd(ref_dose, resolution_zyx, gantry_angle_deg)
    bins_pred, idd_pred = _compute_idd(pred_dose, resolution_zyx, gantry_angle_deg)

    idd_pred_interp = np.interp(bins_ref, bins_pred, idd_pred)
    ref_peak = float(idd_ref.max())
    if ref_peak <= 0.0:
        return 0.0
    return float(np.sqrt(np.mean((idd_pred_interp - idd_ref) ** 2))) / ref_peak


def _distal_r80_depth(bin_centres: np.ndarray, profile: np.ndarray) -> float:
    """Distal R80 of the *main* peak: walk distally from the global max to the first
    sub-80% crossing. Walking from the peak (rather than taking the last over-80% bin
    in the whole column) ignores far secondary bumps from other-energy beamlets that
    clip the column, which otherwise cause spurious tens-of-mm range jumps."""
    peak = float(profile.max())
    if peak <= 0.0:
        return float("nan")
    thr = 0.8 * peak
    pk = int(np.argmax(profile))
    n = len(profile)
    # walk distal from the peak until the profile drops below threshold
    i = pk
    while i + 1 < n and profile[i + 1] >= thr:
        i += 1
    if i >= n - 1:
        return float(bin_centres[i])
    d0, d1 = bin_centres[i], bin_centres[i + 1]
    v0, v1 = profile[i], profile[i + 1]
    if v0 == v1:
        return float(d0)
    return float(d0 + (thr - v0) / (v1 - v0) * (d1 - d0))


def _per_column_range_diff(
    pred_dose: np.ndarray,
    ref_dose: np.ndarray,
    resolution_zyx: tuple[float, float, float],
    gantry_angle_deg: float,
    *,
    lateral_bin_mm: float = 2.0,
    depth_bin_mm: float = 1.0,
    column_dose_frac: float = 0.10,
) -> dict[str, object]:
    """Per-BEV-column distal-R80 range difference map (pred - ref).

    Projects both doses into a common BEV grid (depth along the beam axis, two
    perpendicular lateral axes), then per lateral column computes distal R80 and the
    pred-ref difference. Unlike the laterally-integrated IDD, this resolves *where*
    range disagrees across the lattice. Columns below ``column_dose_frac`` of the
    reference column-peak max are masked (range ill-defined in the penumbra).
    """
    theta = math.radians(gantry_angle_deg)
    axis = np.array([0.0, math.cos(theta), -math.sin(theta)], dtype=np.float64)  # depth dir (zyx)
    u = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # couch / superior-inferior, perp to axis
    v = np.cross(axis, u)  # in-plane lateral
    v /= np.linalg.norm(v)

    Z, Y, X = pred_dose.shape
    zc = np.arange(Z, dtype=np.float64) * float(resolution_zyx[0])
    yc = np.arange(Y, dtype=np.float64) * float(resolution_zyx[1])
    xc = np.arange(X, dtype=np.float64) * float(resolution_zyx[2])
    pos = np.stack(np.meshgrid(zc, yc, xc, indexing="ij"), axis=-1).reshape(-1, 3)
    depth = pos @ axis
    cu = pos @ u
    cv = pos @ v

    def _edges(a, step):
        lo, hi = float(a.min()), float(a.max())
        return np.arange(lo, hi + step, step)

    d_edges = _edges(depth, depth_bin_mm)
    u_edges = _edges(cu, lateral_bin_mm)
    v_edges = _edges(cv, lateral_bin_mm)
    nd, nu, nv = len(d_edges) - 1, len(u_edges) - 1, len(v_edges) - 1
    if min(nd, nu, nv) < 1:
        return {"valid_columns": 0}

    di = np.clip(np.digitize(depth, d_edges) - 1, 0, nd - 1)
    ui = np.clip(np.digitize(cu, u_edges) - 1, 0, nu - 1)
    vi = np.clip(np.digitize(cv, v_edges) - 1, 0, nv - 1)
    flat = (ui * nv + vi) * nd + di

    def _bev(dose):
        b = np.zeros(nu * nv * nd, dtype=np.float64)
        np.add.at(b, flat, dose.ravel().astype(np.float64))
        return b.reshape(nu, nv, nd)

    bev_pred = _bev(pred_dose)
    bev_ref = _bev(ref_dose)
    d_centres = 0.5 * (d_edges[:-1] + d_edges[1:])

    ref_col_peak = bev_ref.max(axis=2)
    global_max = float(ref_col_peak.max())
    valid = ref_col_peak > column_dose_frac * global_max if global_max > 0 else np.zeros_like(ref_col_peak, dtype=bool)

    dr80 = np.full((nu, nv), np.nan, dtype=np.float64)
    for iu in range(nu):
        for iv in range(nv):
            if not valid[iu, iv]:
                continue
            rp = _distal_r80_depth(d_centres, bev_pred[iu, iv])
            rr = _distal_r80_depth(d_centres, bev_ref[iu, iv])
            if np.isfinite(rp) and np.isfinite(rr):
                dr80[iu, iv] = rp - rr

    finite = dr80[np.isfinite(dr80)]
    if finite.size == 0:
        return {"valid_columns": 0}
    return {
        "valid_columns": int(finite.size),
        "range_diff_median_mm": float(np.median(finite)),
        "range_diff_iqr_mm": float(np.percentile(finite, 75) - np.percentile(finite, 25)),
        "range_diff_p95_abs_mm": float(np.percentile(np.abs(finite), 95)),
        "range_diff_frac_gt1mm": float(np.mean(np.abs(finite) > 1.0)),
        "range_diff_frac_gt3mm": float(np.mean(np.abs(finite) > 3.0)),
        "range_diff_map": dr80,
        "u_centres_mm": 0.5 * (u_edges[:-1] + u_edges[1:]),
        "v_centres_mm": 0.5 * (v_edges[:-1] + v_edges[1:]),
    }


def _dose_centroid_zyx(dose: np.ndarray, resolution_zyx: tuple[float, float, float], origin_offset_vox: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Dose-weighted centroid in mm (z,y,x), with optional voxel-grid origin offset."""
    tot = float(dose.sum())
    if tot <= 0.0:
        return np.array([np.nan, np.nan, np.nan])
    idx = [np.average(np.arange(dose.shape[a]), weights=dose.sum(axis=tuple(j for j in range(3) if j != a))) for a in range(3)]
    return np.array([(idx[a] + origin_offset_vox[a]) * float(resolution_zyx[a]) for a in range(3)])


def _write_beamlet_volume_bbox(path: Path, vol: np.ndarray) -> dict:
    """Persist a beamlet dose volume bbox-cropped to its nonzero support (+meta), like
    the reference .b2nd convention. Returns the bbox dict."""
    nz = np.argwhere(vol > 0.0)
    if nz.size == 0:
        bbox = (0, 1, 0, 1, 0, 1)
        crop = vol[0:1, 0:1, 0:1]
    else:
        z0, y0, x0 = nz.min(axis=0)
        z1, y1, x1 = nz.max(axis=0) + 1
        bbox = (int(z0), int(z1), int(y0), int(y1), int(x0), int(x1))
        crop = vol[z0:z1, y0:y1, x0:x1]
    np.savez_compressed(path, dose=crop.astype(np.float32, copy=False), bbox=np.array(bbox, dtype=np.int32), full_shape=np.array(vol.shape, dtype=np.int32))
    return {"bbox": list(bbox), "full_shape": list(vol.shape)}


def _bbox_crop(vol: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]] | None:
    nz = np.argwhere(vol > 0.0)
    if nz.size == 0:
        return None
    z0, y0, x0 = nz.min(axis=0)
    z1, y1, x1 = nz.max(axis=0) + 1
    return vol[z0:z1, y0:y1, x0:x1], (int(z0), int(y0), int(x0))


def _union_mae_normalized(pcrop, poff, rcrop, roff) -> float:
    """Masked MAE (ref>10% of ref-max) over the union bbox of two offset crops."""
    lo = [min(poff[a], roff[a]) for a in range(3)]
    hi = [max(poff[a] + pcrop.shape[a], roff[a] + rcrop.shape[a]) for a in range(3)]
    shp = tuple(hi[a] - lo[a] for a in range(3))
    pa = np.zeros(shp, dtype=np.float32)
    ra = np.zeros(shp, dtype=np.float32)
    pz, py, px = (poff[a] - lo[a] for a in range(3))
    rz, ry, rx = (roff[a] - lo[a] for a in range(3))
    pa[pz:pz + pcrop.shape[0], py:py + pcrop.shape[1], px:px + pcrop.shape[2]] = pcrop
    ra[rz:rz + rcrop.shape[0], ry:ry + rcrop.shape[1], rx:rx + rcrop.shape[2]] = rcrop
    return _beam_masked_mae_normalized(pa, ra)


def _oracle_halo_metrics(pcrop, poff, rcrop, roff, full_shape, resolution_zyx,
                         direction_xyz) -> dict[str, float]:
    """CEILING for a halo-targeted loss: what would the metrics be if the halo were perfect?

    Rescales the prediction's halo (ref < 10% of beamlet max) by a single factor so its
    integral matches the reference exactly, leaves the core untouched, and recomputes.
    This is an ORACLE -- it uses the reference to pick the factor -- so it bounds what any
    halo-targeted training term could achieve, and is not itself a method.

    Note the scored beam MAE masks to ref > 10% (the core), so a halo fix cannot move it at
    all by construction. Only IDD, which integrates the whole plane, can respond. That
    asymmetry is the point of measuring this separately.
    """
    lo = [min(poff[a], roff[a]) for a in range(3)]
    hi = [max(poff[a] + pcrop.shape[a], roff[a] + rcrop.shape[a]) for a in range(3)]
    shp = tuple(hi[a] - lo[a] for a in range(3))
    pa = np.zeros(shp, dtype=np.float64)
    ra = np.zeros(shp, dtype=np.float64)
    pz, py, px = (poff[a] - lo[a] for a in range(3))
    rz, ry, rx = (roff[a] - lo[a] for a in range(3))
    pa[pz:pz + pcrop.shape[0], py:py + pcrop.shape[1], px:px + pcrop.shape[2]] = pcrop
    ra[rz:rz + rcrop.shape[0], ry:ry + rcrop.shape[1], rx:rx + rcrop.shape[2]] = rcrop

    out: dict[str, float] = {}
    rmax = float(ra.max())
    if rmax <= 0.0:
        return out
    halo = (ra > 0.0) & (ra < 0.10 * rmax)
    p_halo = float(pa[halo].sum())
    r_halo = float(ra[halo].sum())
    if p_halo <= 0.0 or r_halo <= 0.0:
        return out
    lo_t = (int(lo[0]), int(lo[1]), int(lo[2]))
    pa_o = pa.copy()
    pa_o[halo] *= r_halo / p_halo
    out["idd_oracle_halo"] = float(_official_idd_distance_beam(
        pa_o, lo_t, ra, lo_t, full_shape, resolution_zyx, direction_xyz))
    out["halo_scale_needed"] = r_halo / p_halo

    # Depth-resolved variant: one scale PER SLICE instead of one per beamlet. A trained
    # loss has spatial freedom, so this is the fairer ceiling -- the single-scalar version
    # above understates it wherever the halo error varies along depth (which is why the
    # scalar oracle makes low energies *worse*: their mean error is ~0 but the structure
    # is not, so a uniform rescale adds error without removing any).
    pa_d = pa.copy()
    ph = np.where(halo, pa, 0.0).sum(axis=(1, 2))
    rh = np.where(halo, ra, 0.0).sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(ph > 0, rh / np.maximum(ph, 1e-30), 1.0)
    s = np.clip(s, 0.0, 10.0)
    pa_d = np.where(halo, pa * s[:, None, None], pa)
    out["idd_oracle_halo_depth"] = float(_official_idd_distance_beam(
        pa_d, lo_t, ra, lo_t, full_shape, resolution_zyx, direction_xyz))
    return out


def _signed_residual_core_halo(pcrop, poff, rcrop, roff) -> dict[str, float]:
    """Split the signed beamlet residual at 10% of the beamlet max: core (>=10%, 77.7% of the
    reference energy) and halo (the rest).

    The split is complete -- ``1 + core*f + halo*(1-f)`` reproduces ``integral_ratio`` to
    3e-3 over all 8640 evaluation beamlets -- so it fully accounts for where our dose goes.

    Measured 2026-08-07/08 over 8640 beamlets, the deficit is almost entirely halo:
    shipped core -0.21% / halo -5.25%; additive -0.22% / -4.61%. The core is flat in energy
    (+0.0009 %/MeV); the halo runs -1.80% below 60 MeV to -11.71% above 180 (-0.0702 %/MeV),
    the signature of an underweighted nuclear-scatter component.

    ROOT CAUSE is the LUT kernel integration window, NOT the lateral model. ``Z``
    (``ray_edep_s``) is built by ``build_windowed_integral_image`` as the MC dose over
    +/-37 mm only (``--kernel-width-mm 74.0``, deliberate: full-range integration
    over-counts large-angle scatter the kernel cannot represent). The engine then
    normalises both the narrow core and the broad halo to sum 1 over the crop and
    multiplies by ``Z`` (ion_dose_engine.py:629-656), so it deposits *exactly* ``Z``.
    Measured reference energy beyond 37 mm is 1.15% of total (0.46% <60 MeV -> 2.36%
    >180 MeV) against a halo deficit of 1.17% in the same units (0.33% -> 2.55%): 98%
    agreement overall, 93% in the top band, 88% of the whole 1.30% integral deficit.
    Only ~0.15% is genuine lateral-model error.

    This is also why the water calibration fits to ~100% while the halo is -5%: its target
    is the *windowed* integral, with the out-of-window energy already removed.

    BUT WIDENING THE WINDOW DOES NOT FIX IT -- measured 2026-08-08, `--no-correction` on
    1ABB006 (1080 beamlets). Widening to the MC's full +/-50 mm (`--kernel-width-mm 100`)
    gives IDD -4.2% but MAE +3.5%; widening the engine crop to 34x100 to match barely
    differs, because sigma2 only grew 0.78% and the engine has no shape to place dose at
    37-50 mm. Refitting the halo to MC (`--double-fit-mode direct`) is worse still --
    IDD +1.5%, MAE +8.4% -- overshooting halo to +3.39% while starving the core to -1.86%.
    MC statistics are not the limit either (~0.05% standard error on the fit target at
    r=38 mm vs a 13.5% systematic). This is a limitation of the double-Gaussian FORM.
    Keep `lut_fast_3d_1e8_opt.mat`; do not re-run these experiments.

    Widening the correction net's BEV crop 26x74 -> 52x148 moved the halo residual 0.04pp.
    That is a DIFFERENT truncation from the LUT window, so that null result does not rule
    the window out -- an earlier "definitively not truncation" call was wrong on that basis.
    Uncorrected, the baseline is +1.73% hot in the core and -7.84% cold in the halo, which
    nearly cancel (int_ratio 0.9957) -- largely a redistribution on top of the window loss.

    The network corrects the core (+1.73% -> -0.08%) and barely touches the halo
    (-7.84% -> -4.39%), so it makes the *total* worse than the engine it corrects
    (0.9957 -> 0.9898). That is why gamma and beam MAE rank 1st/6th while IDD ranks 10th.

    A bias-correcting loss term still must not target the total -- ``w_idd`` drove
    int_ratio to 1.0006 and halo to +0.09% yet lost plan MAE, gamma and DVH, because IDD
    constrains how much dose sits at each depth and never where it sits laterally. Fixing
    the halo requires constraining its radial *shape*, in the engine or in the objective.
    """
    lo = [min(poff[a], roff[a]) for a in range(3)]
    hi = [max(poff[a] + pcrop.shape[a], roff[a] + rcrop.shape[a]) for a in range(3)]
    shp = tuple(hi[a] - lo[a] for a in range(3))
    pa = np.zeros(shp, dtype=np.float64)
    ra = np.zeros(shp, dtype=np.float64)
    pz, py, px = (poff[a] - lo[a] for a in range(3))
    rz, ry, rx = (roff[a] - lo[a] for a in range(3))
    pa[pz:pz + pcrop.shape[0], py:py + pcrop.shape[1], px:px + pcrop.shape[2]] = pcrop
    ra[rz:rz + rcrop.shape[0], ry:ry + rcrop.shape[1], rx:rx + rcrop.shape[2]] = rcrop

    out: dict[str, float] = {}
    rmax = float(ra.max())
    if rmax <= 0.0:
        return out
    core = ra >= 0.10 * rmax
    halo = (ra > 0.0) & ~core
    for tag, mask in (("core", core), ("halo", halo)):
        rs = float(ra[mask].sum())
        out[f"signed_{tag}"] = ((float(pa[mask].sum()) - rs) / rs) if rs > 0.0 else float("nan")
    out["core_frac_of_ref"] = float(ra[core].sum() / ra.sum()) if ra.sum() > 0 else float("nan")
    return out


def _official_idd_distance_z(pcrop, poff, rcrop, roff, n_z: int) -> float:
    """Challenge Level-1.2 IDD distance, ``doserad2026_evaluator.metrics_beam`` semantics.

    The official metric sums each volume over the two *transverse* array axes and
    profiles along ``beam_axis`` (config default 0 = the numpy z axis), i.e. the
    slice direction -- NOT the gantry-rotated beam axis our ``_idd_distance`` uses.
    Computed on the crops, which is exact: everything outside a crop is zero.
    """
    idd_p = np.zeros(n_z, dtype=np.float64)
    idd_r = np.zeros(n_z, dtype=np.float64)
    idd_p[poff[0]:poff[0] + pcrop.shape[0]] = pcrop.sum(axis=(1, 2), dtype=np.float64)
    idd_r[roff[0]:roff[0] + rcrop.shape[0]] = rcrop.sum(axis=(1, 2), dtype=np.float64)
    idd_max = float(idd_r.max())
    if idd_max <= 0.0:
        return float("nan")
    return float(np.sqrt(np.mean((idd_p / idd_max - idd_r / idd_max) ** 2)))


def _ray_direction_xyz(beam_json, ray_index: int) -> np.ndarray:
    """Beam propagation direction, as ``directions_of()`` derives it upstream: the proton
    branch uses ``ray_target - ray_source`` per ray, normalised, in world xyz."""
    ray = beam_json["rays"][int(ray_index)]
    v = np.asarray(ray["ray_target"], dtype=np.float64) - np.asarray(ray["ray_source"], dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm <= 0.0:
        raise ValueError(f"degenerate ray direction on ray {ray_index}")
    return v / norm


def _official_idd_curve_beam(plane_yx, resolution_zyx, direction_xyz) -> np.ndarray:
    """Port of ``doserad2026_evaluator.metrics_beam.compute_idd_curve`` (commit fcb42a5).

    The upstream metric was fixed on 2026-08-04: it used to sum the two transverse axes and
    profile along world z, which for a proton beam travelling in the transverse plane runs
    *perpendicular* to the beam and is not a depth dose at all. It now sums z out first, then
    rotates the remaining plane so the beam lies along the output's first axis and integrates
    laterally.

    ``plane_yx`` is the volume already summed over z, so callers can place a bbox crop into a
    full-size plane instead of a full-size volume -- exact, since everything outside the crop
    is zero, and far cheaper.

    Deliberately mirrors upstream's grid arithmetic exactly (sample step ``max(sx, sy)``, a
    square output wide enough for the in-plane diagonal, origin placing the middle sample at
    0) rather than reusing this file's own ``_compute_idd``, which bins along the beam axis by
    histogram. The two are not interchangeable and only the upstream one is scored.
    """
    import SimpleITK as sitk

    if abs(float(direction_xyz[2])) > 1e-9:
        raise ValueError("beam leaves the transverse plane; z cannot be summed")

    ny, nx = plane_yx.shape
    sx, sy = float(resolution_zyx[2]), float(resolution_zyx[1])
    plane_centre = ((nx - 1) * sx / 2.0, (ny - 1) * sy / 2.0)

    step = max(sx, sy)
    n = int(math.ceil(math.hypot(nx * sx, ny * sy) / step)) + 1
    out_origin = (-(n - 1) * step / 2.0,) * 2

    source = sitk.GetImageFromArray(np.ascontiguousarray(plane_yx, dtype=np.float64))
    source.SetSpacing((sx, sy))

    to_beam = sitk.Euler2DTransform()
    to_beam.SetCenter((0.0, 0.0))
    to_beam.SetAngle(math.atan2(float(direction_xyz[1]), float(direction_xyz[0])))
    to_beam.SetTranslation(plane_centre)
    aligned = sitk.Resample(source, (n, n), to_beam, sitk.sitkLinear,
                            out_origin, (step, step), (1.0, 0.0, 0.0, 1.0),
                            0.0, sitk.sitkFloat64)
    return sitk.GetArrayFromImage(aligned).sum(axis=0)


#: Rotation-resample grids for the beam-axis IDD, keyed by (ny, nx, sx, sy, angle). The grid
#: is identity-valued in dose, depending only on geometry, so it is reused across every
#: beamlet that shares a ray direction -- which within a beam is all of them.
_IDD_GRID_CACHE: dict = {}
_IDD_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _idd_beam_grid(ny: int, nx: int, sx: float, sy: float, angle: float):
    """grid_sample grid replicating the SimpleITK Euler2D resample in _official_idd_curve_beam.

    For output pixel (row oy, col ox) at physical point p_out, SimpleITK samples the source at
    p_in = R(angle)*p_out + plane_centre (Euler2D, centre 0), continuous index p_in/(sx, sy),
    linear interp, zero outside. That is exactly grid_sample(mode='bilinear',
    padding_mode='zeros', align_corners=True) with the normalised version of that index.
    """
    key = (ny, nx, round(sx, 6), round(sy, 6), round(float(angle), 9))
    grid = _IDD_GRID_CACHE.get(key)
    if grid is not None:
        return grid
    step = max(sx, sy)
    n = int(math.ceil(math.hypot(nx * sx, ny * sy) / step)) + 1
    out0 = -(n - 1) * step / 2.0
    cx, cy = (nx - 1) * sx / 2.0, (ny - 1) * sy / 2.0
    c, s = math.cos(angle), math.sin(angle)
    dev = _IDD_DEVICE
    idx = torch.arange(n, device=dev, dtype=torch.float32)
    X = (out0 + idx * step).view(1, n).expand(n, n)   # p_out_x, varies along columns (ox)
    Y = (out0 + idx * step).view(n, 1).expand(n, n)   # p_out_y, varies along rows (oy)
    pix = c * X - s * Y + cx
    piy = s * X + c * Y + cy
    gx = 2.0 * (pix / sx) / (nx - 1) - 1.0
    gy = 2.0 * (piy / sy) / (ny - 1) - 1.0
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)  # (1, n, n, 2)
    _IDD_GRID_CACHE[key] = grid
    return grid


def _official_idd_distance_beam(pcrop, poff, rcrop, roff, full_shape,
                                resolution_zyx, direction_xyz) -> float:
    """Challenge Level-1.2 IDD distance under the 2026-08-04 beam-direction definition.

    GPU port of the (pred, ref) resample-and-project: one batched grid_sample instead of two
    per-beamlet SimpleITK CPU Resample calls, which dominated eval time (~2160 full-plane CPU
    resamples/case). Numerically matches _official_idd_curve_beam to ~1e-6 relative.
    """
    if abs(float(direction_xyz[2])) > 1e-9:
        raise ValueError("beam leaves the transverse plane; z cannot be summed")
    _nz, ny, nx = (int(s) for s in full_shape)
    plane_p = np.zeros((ny, nx), dtype=np.float32)
    plane_r = np.zeros((ny, nx), dtype=np.float32)
    plane_p[poff[1]:poff[1] + pcrop.shape[1], poff[2]:poff[2] + pcrop.shape[2]] = \
        pcrop.sum(axis=0, dtype=np.float32)
    plane_r[roff[1]:roff[1] + rcrop.shape[1], roff[2]:roff[2] + rcrop.shape[2]] = \
        rcrop.sum(axis=0, dtype=np.float32)

    sx, sy = float(resolution_zyx[2]), float(resolution_zyx[1])
    angle = math.atan2(float(direction_xyz[1]), float(direction_xyz[0]))
    grid = _idd_beam_grid(ny, nx, sx, sy, angle).expand(2, -1, -1, -1)
    inp = torch.from_numpy(np.stack((plane_p, plane_r))).unsqueeze(1).to(_IDD_DEVICE)  # (2,1,ny,nx)
    aligned = torch.nn.functional.grid_sample(
        inp, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    curves = aligned[:, 0].sum(dim=1)  # sum over output-y (dim=1) -> (2, n) along beam axis
    idd_p, idd_r = curves[0], curves[1]
    idd_max = idd_r.max()
    if float(idd_max) <= 0.0:
        return float("nan")
    return float(torch.sqrt(torch.mean(((idd_p - idd_r) / idd_max) ** 2)))


def _run_per_beamlet_range(
    *,
    plan, beam_parameters, ct_hu, origin_zyx, resolution_zyx, beam_index, args, lut,
    device, dtype, out_dir, output_stem, dose_dir, beam_input, beam_mass_input,
    make_engine,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Per-beamlet distal-R80 range diff vs per-beamlet MC reference, from ONE batched
    engine pass (``return_per_beamlet=True``). Correspondence is exact: the engine emits
    beamlets in the sequence order (rays->beamlets), identical to the reference-path order.
    Returns (stats, sum_pred, sum_ref); metrics run on bbox crops so they're O(beamlet),
    not O(volume). A dose-centroid guard drops pairs whose spots disagree."""
    beam_json = plan["beams"][beam_index]
    gantry_deg = float(beam_json["gantry_angle"]) + args.gantry_offset_deg
    theta = math.radians(gantry_deg)
    axis = np.array([0.0, math.cos(theta), -math.sin(theta)])
    u = np.array([1.0, 0.0, 0.0]); v = np.cross(axis, u); v /= np.linalg.norm(v)

    write_vols = bool(getattr(args, "write_beamlets", False))
    bl_dir = out_dir / "beamlets"
    if write_vols:
        bl_dir.mkdir(parents=True, exist_ok=True)

    ray_indices = list(range(len(beam_json["rays"])))
    sequence, ssd_values_mm = _make_beamlet_batch_sequence(
        plan=plan, beam_parameters=beam_parameters, ct_hu=ct_hu, origin_zyx=origin_zyx,
        resolution_zyx=resolution_zyx, beam_index=beam_index,
        particles_per_beamlet=args.particles_per_beamlet, gantry_offset_deg=args.gantry_offset_deg,
        skin_hu_threshold=float(args.skin_hu_threshold), sigma_mode=args.sigma_mode,
        bams_to_iso_dist_mm=float(args.bams_to_iso_dist_mm), lut=lut, device=device, dtype=dtype,
    )
    # metadata + reference paths in the SAME rays->beamlets order as the sequence
    meta: list[dict] = []
    for ri in ray_indices:
        ray_json = beam_json["rays"][ri]
        iso = np.asarray((_xyz_to_zyx(ray_json["ray_target"]) - origin_zyx), dtype=np.float64)
        for li, bl in enumerate(ray_json["beamlets"]):
            meta.append({"ray": ri, "beamlet": li, "energy_mev": float(bl["energy"]),
                         "spot_u_mm": float(iso @ u), "spot_v_mm": float(iso @ v),
                         "ref_path": dose_dir / _expected_dose_filename(beam_json, ray_json, bl)})

    engine = make_engine(sequence)
    with torch.no_grad():
        per_beamlet = engine.compute_dose_bev_lattice_sparse_batch(
            sequence, beam_input, mass_density_image=beam_mass_input, overwrite=False,
            ssd_mm=ssd_values_mm if args.air_offset_correction else None,
            finalize_chunk_size=max(1, int(args.dense_hook_batch_items)),
            return_per_beamlet=True,
        )
    del engine, sequence
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if len(per_beamlet) != len(meta):
        raise RuntimeError(f"per-beamlet count {len(per_beamlet)} != metadata {len(meta)} (correspondence broken)")

    rows: list[dict] = []
    official: list[dict] = []
    n_flagged = 0
    sum_pred = np.zeros(ct_hu.shape, dtype=np.float32)
    sum_ref = np.zeros(ct_hu.shape, dtype=np.float32)
    for pb, m in zip(per_beamlet, meta):
        if pb is None or not m["ref_path"].exists():
            continue
        pcrop = pb["dose"][0].detach().cpu().numpy().astype(np.float32, copy=False)
        cutoff = float(getattr(args, "minimum_cutoff", 0.0))
        if cutoff > 0.0:
            pcrop = pcrop.copy()
            pcrop[pcrop <= cutoff] = 0.0
        poff = tuple(int(o) for o in pb["offset"])
        pz, py, px = poff
        sum_pred[pz:pz + pcrop.shape[0], py:py + pcrop.shape[1], px:px + pcrop.shape[2]] += pcrop

        ref = _read_reference_dose(m["ref_path"])
        sum_ref += ref
        rc = _bbox_crop(ref)
        if rc is None:
            continue
        rcrop, roff = rc

        # Challenge Level-1 metrics, scored per BEAMLET on every beamlet: the official
        # evaluator has no centroid guard, so gross mismatches must count here too.
        ref_int = float(rcrop.sum(dtype=np.float64))
        official.append({
            "ray": m["ray"], "beamlet": m["beamlet"], "energy_mev": m["energy_mev"],
            "mae": float(_union_mae_normalized(pcrop, poff, rcrop, roff)),
            # `idd` is the SCORED metric (beam direction, post-2026-08-04). `idd_z_legacy` is
            # the old world-z definition, kept only so historical numbers stay comparable.
            "idd": float(_official_idd_distance_beam(
                pcrop, poff, rcrop, roff, sum_ref.shape, resolution_zyx, _ray_direction_xyz(beam_json, m["ray"]))),
            "idd_z_legacy": float(_official_idd_distance_z(pcrop, poff, rcrop, roff, int(sum_ref.shape[0]))),
            "integral_ratio": (float(pcrop.sum(dtype=np.float64)) / ref_int) if ref_int > 0.0 else float("nan"),
            **_signed_residual_core_halo(pcrop, poff, rcrop, roff),
            # Opt-in: two extra beam-axis IDD resamples per beamlet, ~45% on runtime.
            **(_oracle_halo_metrics(
                pcrop, poff, rcrop, roff, sum_ref.shape, resolution_zyx,
                _ray_direction_xyz(beam_json, m["ray"]))
               if getattr(args, "oracle_halo", False) else {}),
        })

        cp = _dose_centroid_zyx(pcrop, resolution_zyx, poff)
        cr = _dose_centroid_zyx(rcrop, resolution_zyx, roff)
        if not (np.all(np.isfinite(cp)) and np.all(np.isfinite(cr))):
            continue
        if float(np.linalg.norm(cp - cr)) > 8.0:  # mm; gross spot mismatch
            n_flagged += 1
            continue

        bp, ip = _compute_idd(pcrop, resolution_zyx, gantry_deg, poff)
        br, ir = _compute_idd(rcrop, resolution_zyx, gantry_deg, roff)
        r80p = _distal_r80_depth(bp, ip)
        r80r = _distal_r80_depth(br, ir)
        if not (np.isfinite(r80p) and np.isfinite(r80r)):
            continue
        row = {
            "ray": m["ray"], "beamlet": m["beamlet"], "energy_mev": m["energy_mev"],
            "spot_u_mm": m["spot_u_mm"], "spot_v_mm": m["spot_v_mm"],
            "r80_pred_mm": float(r80p), "r80_ref_mm": float(r80r), "dr80_mm": float(r80p - r80r),
            "mae_normalized": float(_union_mae_normalized(pcrop, poff, rcrop, roff)),
        }
        if write_vols:
            vol_path = bl_dir / f"{output_stem}_b{int(beam_json['beam_idx']):02d}_r{m['ray']:03d}_l{m['beamlet']:02d}.npz"
            np.savez_compressed(vol_path, dose=pcrop, offset=np.array(poff, np.int32), full_shape=np.array(pb["full_shape"], np.int32))
            row["volume_file"] = vol_path.name
        rows.append(row)

    official_stats: dict = {"official_n": len(official)}
    if official:
        omae = np.array([r["mae"] for r in official], dtype=np.float64)
        oidd = np.array([r["idd"] for r in official], dtype=np.float64)
        orat = np.array([r["integral_ratio"] for r in official], dtype=np.float64)
        official_stats.update({
            # nanmean over beamlets == the challenge's per-plan `beam_mae_mean`
            "official_mae_mean": float(np.nanmean(omae)),
            "official_mae_median": float(np.nanmedian(omae)),
            "official_mae_p95": float(np.nanpercentile(omae, 95)),
            "official_idd_mean": float(np.nanmean(oidd)),
            "official_idd_median": float(np.nanmedian(oidd)),
            "official_idd_p95": float(np.nanpercentile(oidd, 95)),
            "official_idd_z_legacy_mean": float(np.nanmean(
                np.array([r["idd_z_legacy"] for r in official], dtype=np.float64))),
            "integral_ratio_mean": float(np.nanmean(orat)),
            "integral_ratio_median": float(np.nanmedian(orat)),
        })
        with (out_dir / f"{output_stem}_beam{int(beam_json['beam_idx']):02d}_official.json").open("w") as fh:
            json.dump({"stats": official_stats, "rows": official}, fh, indent=2)

    if not rows:
        return {"n_beamlets": 0, "n_flagged_centroid": n_flagged, **official_stats}, sum_pred, sum_ref
    d = np.array([r["dr80_mm"] for r in rows])
    mae = np.array([r["mae_normalized"] for r in rows])
    stats = {
        "n_beamlets": len(rows),
        "n_flagged_centroid": n_flagged,
        "dr80_median_mm": float(np.median(d)),
        "dr80_iqr_mm": float(np.percentile(d, 75) - np.percentile(d, 25)),
        "dr80_p95_abs_mm": float(np.percentile(np.abs(d), 95)),
        "dr80_frac_gt1mm": float(np.mean(np.abs(d) > 1.0)),
        "dr80_frac_gt3mm": float(np.mean(np.abs(d) > 3.0)),
        "mae_normalized_median": float(np.median(mae)),
        "mae_normalized_p95": float(np.percentile(mae, 95)),
        **official_stats,
    }
    # lattice scatter
    uu = np.array([r["spot_u_mm"] for r in rows]); vv = np.array([r["spot_v_mm"] for r in rows])
    vmax = max(float(np.percentile(np.abs(d), 98)), 0.5)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(uu, vv, c=d, cmap="RdBu_r", vmin=-vmax, vmax=vmax, s=60, edgecolors="k", linewidths=0.3)
    fig.colorbar(sc, ax=ax, label="distal R80 pred-ref [mm]")
    ax.set_title(f"B{beam_json['beam_idx']} {gantry_deg:.0f}° per-beamlet ΔR80 | "
                 f"med={stats['dr80_median_mm']:.2f} p95|Δ|={stats['dr80_p95_abs_mm']:.2f}mm (n={len(rows)})")
    ax.set_xlabel("spot u [mm]"); ax.set_ylabel("spot v [mm]"); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / f"{output_stem}_beam{int(beam_json['beam_idx']):02d}_perbeamlet_dr80.png", dpi=160)
    plt.close(fig)
    with (out_dir / f"{output_stem}_beam{int(beam_json['beam_idx']):02d}_perbeamlet_dr80.json").open("w") as fh:
        json.dump({"stats": stats, "rows": rows}, fh, indent=2)
    return stats, sum_pred, sum_ref


def _beam_masked_mae_normalized(pred: np.ndarray, ref: np.ndarray) -> float:
    """Level 1.1: MAE in high-dose region (≥10% of beam max), normalised by beam max."""
    ref_max = float(ref.max())
    if ref_max <= 0.0:
        return 0.0
    mask = ref > 0.10 * ref_max
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(pred[mask] - ref[mask]))) / ref_max


def _stratified_plan_mae(
    pred: np.ndarray,
    ref: np.ndarray,
    prescription_gy: float,
) -> dict[str, float]:
    """Level 2.1: unweighted mean of MAE/Rx across three dose strata."""
    if prescription_gy <= 0.0:
        return {"high": 0.0, "mid": 0.0, "low": 0.0, "combined": 0.0}

    strata: dict[str, np.ndarray] = {
        "high": ref >= 0.80 * prescription_gy,
        "mid": (ref >= 0.30 * prescription_gy) & (ref < 0.80 * prescription_gy),
        "low": (ref >= 0.10 * prescription_gy) & (ref < 0.30 * prescription_gy),
    }

    result: dict[str, float] = {}
    for name, mask in strata.items():
        if np.any(mask):
            result[name] = float(np.mean(np.abs(pred[mask] - ref[mask]))) / prescription_gy
        else:
            result[name] = 0.0
    result["combined"] = (result["high"] + result["mid"] + result["low"]) / 3.0
    return result


def _dvh_surrogate(
    pred: np.ndarray,
    ref: np.ndarray,
    prescription_gy: float,
) -> dict[str, float]:
    """Contour-free stand-in for the challenge's DVH clinical score.

    The scored DVH metric needs organ/target contours we do not have. But the property that
    makes DVH hypersensitive is reproducible without them: DVH points are *quantiles* of the
    dose distribution inside a high-dose region, so a systematic shift moves them at gain 1,
    while MAE averages it away at gain << 1. That is exactly the failure mode that cost the
    2026-08-07 submission -- Beam MAE improved 4.5% while the DVH score worsened 78%.

    Evaluate on the target-like region ``ref >= 0.8 * Rx`` and report the near-min (D98/D95)
    and near-max (D2) percentiles of both dose distributions, as a fraction of Rx. The signed
    deltas are the diagnostic: a coherent hot bias shows up as all three moving together and
    positive, which no magnitude metric will show.
    """
    out: dict[str, float] = {}
    if prescription_gy <= 0.0:
        return out
    mask = ref >= 0.80 * prescription_gy
    n = int(np.count_nonzero(mask))
    out["dvh_n_voxels"] = float(n)
    if n < 100:  # too small to be a meaningful quantile
        return out
    p, r = pred[mask], ref[mask]
    for tag, q in (("d98", 2.0), ("d95", 5.0), ("d2", 98.0)):
        pv = float(np.percentile(p, q)) / prescription_gy
        rv = float(np.percentile(r, q)) / prescription_gy
        out[f"{tag}_pred"] = pv
        out[f"{tag}_ref"] = rv
        out[f"{tag}_delta"] = pv - rv          # signed: bias shows here, magnitude does not
    # Mean signed error over the target region -- the coherent component that accumulates
    # across beamlets and that plan-level metrics punish.
    out["target_mean_signed"] = float(np.mean(p - r)) / prescription_gy
    out["target_mae"] = float(np.mean(np.abs(p - r))) / prescription_gy
    return out


def _local_gamma_pass_rate(
    pred: np.ndarray,
    ref: np.ndarray,
    resolution_zyx: tuple[float, float, float],
    prescription_gy: float,
    dose_pct: float,
    dist_mm: float,
    interp: int,
    random_subset: int | None = None,
    device: torch.device | None = None,
) -> tuple[float, dict[str, object]]:
    """Level 2.2: local gamma pass rate (%) evaluated in region ≥10% of prescription dose."""
    crop_info: dict[str, object] = {
        "original_shape": list(ref.shape),
        "reference_shape": list(ref.shape),
        "evaluation_shape": list(pred.shape),
        "reference_voxels_above_cutoff": (
            int(np.count_nonzero(ref > 0.1 * prescription_gy)) if prescription_gy > 0.0 else 0
        ),
        "cropped": False,
        "random_subset": random_subset,
        "backend": "pydose_rt.utils.gamma.local_gamma_pass_rate",
        "device": str(device) if device is not None else None,
    }
    pass_rate = _torch_gamma_pass_rate(
        ref=ref,
        pred=pred,
        voxel_size_mm=resolution_zyx,
        dose_threshold_pct=dose_pct,
        dist_threshold_mm=dist_mm,
        prescription_gy=prescription_gy,
        lower_cutoff_pct=10.0,
        max_gamma=2.0,
        interp_fraction=interp,
        device=device,
    )
    return pass_rate, crop_info


# ---------------------------------------------------------------------------
# Statistics aggregation
# ---------------------------------------------------------------------------

def _metric_statistics(values: list[float]) -> dict:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p1": float(np.percentile(arr, 1)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(values),
    }


def _aggregate_all_metrics(all_case_metrics: list[dict]) -> dict:
    def collect(key_path: list[str]) -> list[float]:
        values = []
        for m in all_case_metrics:
            v: object = m
            try:
                for k in key_path:
                    v = v[k]  # type: ignore[index]
                if v is not None:
                    values.append(float(v))  # type: ignore[arg-type]
            except (KeyError, TypeError):
                pass
        return values

    def stats(key_path: list[str]) -> dict | None:
        vs = collect(key_path)
        return _metric_statistics(vs) if vs else None

    return {
        "n_cases": len(all_case_metrics),
        "level1_per_beam_mean": {
            "masked_mae_normalized": stats(["level1_per_beam_mean", "masked_mae_normalized"]),
            "idd_distance": stats(["level1_per_beam_mean", "idd_distance"]),
        },
        "level2_plan": {
            "stratified_mae_high": stats(["level2_plan", "stratified_mae", "high"]),
            "stratified_mae_mid": stats(["level2_plan", "stratified_mae", "mid"]),
            "stratified_mae_low": stats(["level2_plan", "stratified_mae", "low"]),
            "stratified_mae_combined": stats(["level2_plan", "stratified_mae", "combined"]),
            "gamma_pass_rate_pct": stats(["level2_plan", "gamma_pass_rate_pct"]),
        },
        "timing": {
            "runtime_per_beam_s": stats(["timing", "runtime_per_beam_s"]),
        },
    }


def _print_aggregate_stats(agg: dict) -> None:
    hdr = f"  {'metric':<40s}  {'mean':>9}  {'std':>9}  {'median':>9}  {'p1':>9}  {'p99':>9}  {'min':>9}  {'max':>9}  {'n':>4}"
    sep = "  " + "-" * (len(hdr) - 2)
    print()
    print(f"╔══ Aggregate Statistics — {agg['n_cases']} cases ══╗")
    print(hdr)
    print(sep)
    sections: list[tuple[str, str]] = [
        ("Level 1 per-beam mean", "level1_per_beam_mean"),
        ("Level 2 plan", "level2_plan"),
        ("Timing", "timing"),
    ]
    for section_name, section_key in sections:
        print(f"  {section_name}:")
        for metric_name, s in agg[section_key].items():
            if s is None:
                print(f"    {metric_name:<38s}  skipped")
                continue
            print(
                f"    {metric_name:<38s}"
                f"  {s['mean']:9.5f}  {s['std']:9.5f}  {s['median']:9.5f}"
                f"  {s['p1']:9.5f}  {s['p99']:9.5f}  {s['min']:9.5f}  {s['max']:9.5f}"
                f"  {s['n']:4d}"
            )
    print()


# ---------------------------------------------------------------------------
# Per-beam comparison plot
# ---------------------------------------------------------------------------

def _plot_beam_comparison(
    patient_id: str,
    beam_id: int,
    gantry_angle_deg: float,
    ct: np.ndarray,
    beam_ref: np.ndarray,
    beam_pred: np.ndarray,
    metrics: dict[str, float],
    display_percentile: float,
    out_path: Path,
) -> None:
    beam_pred_display = beam_pred
    diff = beam_pred_display - beam_ref

    max_pos = np.unravel_index(int(np.argmax(beam_ref)), beam_ref.shape)
    z_mid, y_mid, x_mid = int(max_pos[0]), int(max_pos[1]), int(max_pos[2])

    dose_vmax = _robust_positive_max(
        np.concatenate([beam_ref[beam_ref > 0], beam_pred_display[beam_pred_display > 0]])
        if np.any(beam_ref > 0)
        else np.array([1.0]),
        display_percentile,
    )

    views = [
        ("Transversal (y,x)", beam_ref[z_mid], beam_pred_display[z_mid], diff[z_mid], ct[z_mid]),
        ("Coronal (z,x)", beam_ref[:, y_mid], beam_pred_display[:, y_mid], diff[:, y_mid], ct[:, y_mid]),
        ("Sagittal (z,y)", beam_ref[..., x_mid], beam_pred_display[..., x_mid], diff[..., x_mid], ct[..., x_mid]),
    ]

    col_titles = ["Reference MC", "Predicted", "Difference (pred−ref)"]
    fig, axes_grid = plt.subplots(3, 3, figsize=(13.5, 12.0))

    for row, (view_label, ref_view, pred_view, diff_view, ct_view) in enumerate(views):
        diff_vmax = max(float(np.abs(diff_view).max()), 1e-8)

        panels = [
            (ref_view, "inferno", 0.0, dose_vmax),
            (pred_view, "inferno", 0.0, dose_vmax),
            (diff_view, "bwr", -diff_vmax, diff_vmax),
        ]

        for col, (img, cmap, vmin, vmax) in enumerate(panels):
            ax = axes_grid[row, col]
            ax.imshow(ct_view, origin="lower", cmap="gray", vmin=ct.min(), vmax=ct.max())
            im = ax.imshow(
                img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                interpolation="nearest", aspect="auto", alpha=0.45,
            )
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(view_label, fontsize=9)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    mae_str = f"{metrics.get('masked_mae_normalized', float('nan')):.4f}"
    idd_str = f"{metrics.get('idd_distance', float('nan')):.4f}"
    fig.suptitle(
        f"DoseRAD {patient_id}  |  Beam {beam_id:02d}  |  Gantry {gantry_angle_deg:.1f}°\n"
        f"Masked MAE (norm.) = {mae_str}  |  IDD distance = {idd_str}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------

def _run_case(
    case_dir: Path,
    args: argparse.Namespace,
    beam_parameters: dict,
    lut: PyRadPlanIonLUT,
    machine_config: MachineConfig,
    device: torch.device,
    dtype: torch.dtype,
    out_dir: Path,
    correction_hook=None,
    sparse_hooks: IonSparseHooks | None = None,
) -> dict | None:
    """Evaluate a single case. Returns the metrics dict (also saved as JSON), or None if skipped."""
    import SimpleITK as sitk

    try:
        patient_id, plan_path, ct_path, dose_dir = _resolve_case_files(case_dir)
    except FileNotFoundError as exc:
        print(f"  [skip] {case_dir.name}: {exc}")
        return None

    output_stem = patient_id
    if args.beam_index is not None:
        output_stem = f"{output_stem}_beam{args.beam_index}"
    if args.ray_index is not None:
        output_stem = f"{output_stem}_ray{args.ray_index}"
    if args.beamlet_index is not None:
        output_stem = f"{output_stem}_beamlet{args.beamlet_index}"
    if args.gantry_offset_deg:
        output_stem = f"{output_stem}_goff{args.gantry_offset_deg:g}"

    metrics_path = out_dir / f"{output_stem}_metrics.json"
    if args.skip_existing and metrics_path.exists():
        print(f"  [skip] {patient_id}: metrics already exist")
        return None

    plan = _load_json(plan_path)
    if not args.speed_only:
        _assert_case_is_complete(plan, dose_dir)

    ct_image = sitk.ReadImage(str(ct_path))
    ct_hu = sitk.GetArrayFromImage(ct_image).astype(np.float32, copy=False)
    origin_zyx = _origin_zyx(ct_image)
    resolution_zyx = _resolution_zyx(ct_image)

    has_correction = sparse_hooks is not None
    density_np = _doserad_density_from_hu(ct_hu, beam_parameters)
    hu_to_density_entries = beam_parameters.get("hu_to_density", {}).get("entries", None)
    ct_hu_t = torch.from_numpy(ct_hu).to(device=device, dtype=dtype) if args.spr or has_correction else None
    if has_correction and correction_hook is not None:
        correction_hook.set_hu_volume(ct_hu_t)

    dense_field_size: tuple[int, int] | None = (int(args.dense_field_size[0]), int(args.dense_field_size[1])) if args.dense_field_size is not None else None
    if has_correction and correction_hook is not None:
        if dense_field_size is not None:
            crop_h = max(1, int(math.ceil(float(dense_field_size[0]) * 0.5)))
            crop_w = max(1, int(math.ceil(float(dense_field_size[1]) * 0.5)))
        elif args.dense_isotropic_crop:
            crop_h = int(correction_hook.bev_crop_hw)
            crop_w = int(correction_hook.bev_crop_hw)
            dense_field_size = (crop_h * 2, crop_w * 2)
        else:
            # Per-axis crops as read from the checkpoint by from_checkpoint(). NOT
            # bev_crop_hw, which is a legacy scalar fallback (64 on this model, vs the
            # trained (13, 37)) and yields a BEV window the correction net never saw.
            crop_h = int(correction_hook.bev_crop_h)
            crop_w = int(correction_hook.bev_crop_w)
            dense_field_size = (crop_h * 2, crop_w * 2)
        correction_hook.set_bev_crop_half_widths(crop_h, crop_w)
    if dense_field_size is None:
        dense_hw = int(args.dense_bev_crop_hw or 37)
        if not args.dense_isotropic_crop:
            res_h, _, res_w = resolution_zyx[0], resolution_zyx[1], resolution_zyx[2]
            physical_half_width_mm = float(dense_hw) * res_w
            crop_h = max(1, int(math.ceil(physical_half_width_mm / max(res_h, 1e-8))))
            dense_field_size = (crop_h * 2, dense_hw * 2)
        else:
            dense_field_size = (dense_hw * 2, dense_hw * 2)

    def _new_engine(sequence):
        return IonDoseEngine(
            machine_config=machine_config,
            lut=lut,
            dose_grid_spacing=resolution_zyx,
            dose_grid_shape=ct_hu.shape,
            beam_template=sequence,
            device=device,
            dtype=dtype,
            lateral_model=args.lateral_model,
            transport_step_mm=args.transport_step_mm,
            sparse_hooks=sparse_hooks,
            field_size=dense_field_size,
            heterogeneous_mcs=args.heterogeneous_mcs,
            material_radiation_length=args.material_radiation_length,
        )

    def _new_engine_masked(sequence):
        engine = _new_engine(sequence)
        engine.set_patient_dose_mask(dose_mask)
        return engine

    num_plan_beams = len(plan["beams"])
    if args.beam_index is not None and (args.beam_index < 0 or args.beam_index >= num_plan_beams):
        raise ValueError(f"--beam-index must be in [0, {num_plan_beams - 1}], got {args.beam_index}")
    beam_indices = [args.beam_index] if args.beam_index is not None else list(range(num_plan_beams))
    if args.beam_index is None and int(args.beam_stride) > 1:
        # Level-1 is a nanmean over beamlets, so a strided beam subset is an unbiased
        # estimate of it. Stride rather than a random draw so two runs score the IDENTICAL
        # beamlets and the checkpoint comparison stays paired. Gantry angles are ordered,
        # so a stride also samples the full angular range instead of one sector.
        beam_indices = beam_indices[:: int(args.beam_stride)]

    selected_ray_indices_by_beam: dict[int, list[int]] = {}
    total_beamlets = 0
    total_rays = 0
    for beam_index in beam_indices:
        beam_json = plan["beams"][beam_index]
        ray_indices = _selected_ray_indices(beam_json, args.ray_index, beam_index)
        selected_ray_indices_by_beam[beam_index] = ray_indices
        total_rays += len(ray_indices)
        total_beamlets += sum(
            len(_selected_beamlets(beam_json["rays"][ri], args.beamlet_index, beam_index, ri))
            for ri in ray_indices
        )

    print(
        f"  {patient_id}: {len(beam_indices)}/{num_plan_beams} beams, "
        f"{total_rays} rays, {total_beamlets} beamlets"
        + (" [+correction]" if has_correction else "")
    )
    pb_config_label = (
        f"PB lut={args.pyradplan_machine_mat.name} lateral={args.lateral_model} "
        f"split={args.split_mode} mcs={bool(args.heterogeneous_mcs)} "
        f"mat_x0={bool(args.material_radiation_length)} spr={bool(args.spr)}"
    )
    print(f"  {pb_config_label}")

    n_ref_workers = max(1, int(args.reference_io_workers))

    def _load_beam_ref(bi: int) -> np.ndarray:
        paths = _reference_paths_for_selection(
            dose_dir=dose_dir,
            beam_json=plan["beams"][bi],
            beam_index=bi,
            ray_indices=selected_ray_indices_by_beam[bi],
            beamlet_index=args.beamlet_index,
        )
        return _load_reference_paths_sum(paths, io_workers=n_ref_workers)

    total_pred = None if args.speed_only else np.zeros(ct_hu.shape, dtype=np.float32)
    total_ref = None if args.speed_only else np.zeros(ct_hu.shape, dtype=np.float32)
    per_beam_metrics: dict[str, dict[str, float]] = {}
    pydosert_compute_s = 0.0
    processed_beamlets = 0
    input_image_tensor = torch.from_numpy(density_np).unsqueeze(0).to(device=device, dtype=dtype)
    mass_density_tensor = input_image_tensor
    # Dose-scoring region. Density-only would zero internal air (trachea, bowel gas),
    # where the MC reference carries real dose. Beam-independent -> computed once.
    dose_mask = patient_dose_mask(mass_density_tensor[0])
    print(f"  dose mask: {int((~dose_mask).sum()):,} voxels zeroed as external air "
          f"({int((mass_density_tensor[0] <= 0.03).sum()) - int((~dose_mask).sum()):,} "
          f"sub-threshold voxels kept as internal cavities)")

    with torch.inference_mode():
        for selected_idx, beam_index in enumerate(beam_indices, start=1):
            beam_json = plan["beams"][beam_index]
            ray_indices = selected_ray_indices_by_beam[beam_index]

            # Keep the PB stopping volume independent of whether a correction
            # checkpoint is attached. MeV->Gy conversion always uses physical density.
            if args.spr or has_correction:
                e_ref = float(np.mean([
                    float(bl["energy"])
                    for ray in beam_json["rays"]
                    for bl in ray["beamlets"]
                ]) or 150.0)
                beam_spr, beam_mass = spr_and_mass_density(ct_hu_t, e_ref, hu_to_density_entries)
                beam_input = beam_spr.unsqueeze(0)
                beam_mass_input = beam_mass.unsqueeze(0)
            else:
                beam_input = input_image_tensor
                beam_mass_input = mass_density_tensor

            gantry_deg = float(beam_json["gantry_angle"]) + args.gantry_offset_deg
            beam_pred = None
            beam_ref = None

            if args.per_beamlet_range:
                # Primary path: one batched engine pass returning per-beamlet volumes;
                # metrics per beamlet, pred/ref summed here for the plan-level gamma.
                _t0 = time.perf_counter()
                pbr, beam_pred, beam_ref = _run_per_beamlet_range(
                    plan=plan, beam_parameters=beam_parameters, ct_hu=ct_hu, origin_zyx=origin_zyx,
                    resolution_zyx=resolution_zyx, beam_index=beam_index, args=args, lut=lut,
                    device=device, dtype=dtype, out_dir=out_dir, output_stem=output_stem, dose_dir=dose_dir,
                    beam_input=beam_input, beam_mass_input=beam_mass_input, make_engine=_new_engine_masked,
                )
                pydosert_compute_s += time.perf_counter() - _t0
                total_pred += beam_pred
                total_ref += beam_ref
                beam_metrics = {f"pbr_{k}": v for k, v in pbr.items()}
                # Report the CHALLENGE definitions here: per-beamlet masked MAE and
                # z-axis IDD, aggregated with nanmean (not median) exactly as
                # doserad2026_evaluator.metrics_beam.evaluate_beam_level does.
                beam_metrics["masked_mae_normalized"] = float(pbr.get("official_mae_mean", float("nan")))
                beam_metrics["idd_distance"] = float(pbr.get("official_idd_mean", float("nan")))
                print(
                    f"    OFFICIAL per-beamlet: MAE_mean={pbr.get('official_mae_mean', float('nan')):.4f} "
                    f"(med={pbr.get('official_mae_median', float('nan')):.4f}, "
                    f"p95={pbr.get('official_mae_p95', float('nan')):.4f})  "
                    f"IDDz_mean={pbr.get('official_idd_mean', float('nan')):.4f} "
                    f"(p95={pbr.get('official_idd_p95', float('nan')):.4f})  "
                    f"int_ratio_med={pbr.get('integral_ratio_median', float('nan')):.4f}  "
                    f"n={pbr.get('official_n', 0)}"
                )
                print(
                    f"    per-beamlet ΔR80: median={pbr.get('dr80_median_mm', float('nan')):.3f}mm  "
                    f"p95|Δ|={pbr.get('dr80_p95_abs_mm', float('nan')):.3f}mm  "
                    f"frac>1mm={pbr.get('dr80_frac_gt1mm', float('nan')):.3f}  "
                    f"frac>3mm={pbr.get('dr80_frac_gt3mm', float('nan')):.3f}  "
                    f"MAE_med={pbr.get('mae_normalized_median', float('nan')):.4f}  "
                    f"(n={pbr.get('n_beamlets', 0)}, flagged={pbr.get('n_flagged_centroid', 0)})"
                )
            else:
                sequence, ssd_values_mm = _make_beamlet_batch_sequence(
                    plan=plan,
                    beam_parameters=beam_parameters,
                    ct_hu=ct_hu,
                    origin_zyx=origin_zyx,
                    resolution_zyx=resolution_zyx,
                    beam_index=beam_index,
                    particles_per_beamlet=args.particles_per_beamlet,
                    gantry_offset_deg=args.gantry_offset_deg,
                    skin_hu_threshold=float(args.skin_hu_threshold),
                    sigma_mode=args.sigma_mode,
                    bams_to_iso_dist_mm=float(args.bams_to_iso_dist_mm),
                    lut=lut,
                    device=device,
                    dtype=dtype,
                    ray_indices=ray_indices,
                    beamlet_index=args.beamlet_index,
                )
                engine = _new_engine_masked(sequence)

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                _t0 = time.perf_counter()
                beam_pred_gpu = engine.compute_dose_bev_lattice_sparse_batch(
                    sequence,
                    beam_input,
                    mass_density_image=beam_mass_input,
                    overwrite=False,
                    ssd_mm=ssd_values_mm if args.air_offset_correction else None,
                    finalize_chunk_size=max(1, int(args.dense_hook_batch_items)),
                )[0].detach()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                beam_compute_s = time.perf_counter() - _t0
                pydosert_compute_s += beam_compute_s

                if args.speed_only:
                    del beam_pred_gpu
                    del sequence, engine
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    beam_beamlet_count = sum(
                        len(_selected_beamlets(beam_json["rays"][ri], args.beamlet_index, beam_index, ri))
                        for ri in ray_indices
                    )
                    processed_beamlets += beam_beamlet_count
                    print(
                        f"  Beam {selected_idx}/{len(beam_indices)} (B{beam_json['beam_idx']}, "
                        f"{gantry_deg:.1f}°): compute={beam_compute_s:.4g}s, "
                        f"{beam_compute_s / max(beam_beamlet_count, 1):.4g}s/beamlet "
                        f"(n={beam_beamlet_count})"
                    )
                    continue

                del sequence, engine
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                beam_pred = beam_pred_gpu.cpu().numpy().astype(np.float32, copy=False)
                del beam_pred_gpu

                beam_ref = _load_beam_ref(beam_index)

                assert total_pred is not None and total_ref is not None
                total_pred += beam_pred
                total_ref += beam_ref

                beam_metrics: dict[str, float] = {
                    "masked_mae_normalized": _beam_masked_mae_normalized(beam_pred, beam_ref),
                    "idd_distance": _idd_distance(beam_pred, beam_ref, resolution_zyx, gantry_deg),
                }
                if args.range_map:
                    rdiff = _per_column_range_diff(beam_pred, beam_ref, resolution_zyx, gantry_deg)
                    for k in ("valid_columns", "range_diff_median_mm", "range_diff_iqr_mm", "range_diff_p95_abs_mm", "range_diff_frac_gt1mm", "range_diff_frac_gt3mm"):
                        if k in rdiff:
                            beam_metrics[k] = rdiff[k]
                    if isinstance(rdiff.get("range_diff_map"), np.ndarray):
                        rmap = rdiff["range_diff_map"]
                        fig, ax = plt.subplots(figsize=(6, 5))
                        vmax = float(np.nanpercentile(np.abs(rmap), 98)) if np.isfinite(rmap).any() else 1.0
                        vmax = max(vmax, 0.5)
                        im = ax.imshow(
                            rmap.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                            extent=[float(rdiff["u_centres_mm"][0]), float(rdiff["u_centres_mm"][-1]),
                                    float(rdiff["v_centres_mm"][0]), float(rdiff["v_centres_mm"][-1])],
                        )
                        fig.colorbar(im, ax=ax, label="distal R80 pred-ref [mm]")
                        ax.set_title(
                            f"B{beam_json['beam_idx']} {gantry_deg:.0f}° distal-R80 diff | "
                            f"med={rdiff.get('range_diff_median_mm', float('nan')):.2f} "
                            f"p95|Δ|={rdiff.get('range_diff_p95_abs_mm', float('nan')):.2f}mm"
                        )
                        ax.set_xlabel("couch lateral u [mm]")
                        ax.set_ylabel("in-plane lateral v [mm]")
                        fig.tight_layout()
                        fig.savefig(out_dir / f"{output_stem}_beam{int(beam_json['beam_idx']):02d}_rangemap.png", dpi=160)
                        plt.close(fig)
                    print(
                        f"    range diff: median={beam_metrics.get('range_diff_median_mm', float('nan')):.3f}mm  "
                        f"p95|Δ|={beam_metrics.get('range_diff_p95_abs_mm', float('nan')):.3f}mm  "
                        f"frac>1mm={beam_metrics.get('range_diff_frac_gt1mm', float('nan')):.3f}  "
                        f"frac>3mm={beam_metrics.get('range_diff_frac_gt3mm', float('nan')):.3f}  "
                        f"(n={beam_metrics.get('valid_columns', 0)})"
                    )

            per_beam_metrics[str(int(beam_json["beam_idx"]))] = beam_metrics

            beam_beamlet_count = sum(
                len(_selected_beamlets(beam_json["rays"][ri], args.beamlet_index, beam_index, ri))
                for ri in ray_indices
            )
            processed_beamlets += beam_beamlet_count

            print(
                f"  Beam {selected_idx}/{len(beam_indices)} (B{beam_json['beam_idx']}, "
                f"{gantry_deg:.1f}°): "
                f"MAE_norm={beam_metrics['masked_mae_normalized']:.4f}  "
                f"IDD_dist={beam_metrics['idd_distance']:.4f}"
            )

            if not args.skip_figures and not args.skip_beam_plots:
                beam_plot_path = (
                    out_dir / f"{output_stem}_beam{int(beam_json['beam_idx']):02d}_comparison.png"
                )
                _plot_beam_comparison(
                    patient_id=patient_id,
                    beam_id=int(beam_json["beam_idx"]),
                    gantry_angle_deg=gantry_deg,
                    ct=ct_hu,
                    beam_ref=beam_ref,
                    beam_pred=beam_pred,
                    metrics=beam_metrics,
                    display_percentile=args.display_percentile,
                    out_path=beam_plot_path,
                )
                print(f"    Saved beam plot: {beam_plot_path}")

    del input_image_tensor, mass_density_tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    n_beams_processed = len(beam_indices)
    runtime_per_beam = pydosert_compute_s / max(n_beams_processed, 1)

    if args.speed_only:
        metrics_out: dict = {
            "patient_id": patient_id,
            "mode": "speed_only",
            "timing": {
                "engine_total_s": pydosert_compute_s,
                "runtime_per_beam_s": runtime_per_beam,
                "runtime_per_beamlet_s": pydosert_compute_s / max(processed_beamlets, 1),
                "n_beams_processed": n_beams_processed,
                "n_beamlets_processed": processed_beamlets,
            },
            "correction": {
                "checkpoint": str(args.correction_checkpoint) if args.correction_checkpoint is not None else None,
                "dense_field_size": list(dense_field_size) if dense_field_size is not None else None,
            },
            "pencil_beam": {
                "machine_mat": str(args.pyradplan_machine_mat),
                "lateral_model": str(args.lateral_model),
                "split_mode": str(args.split_mode),
                "heterogeneous_mcs": bool(args.heterogeneous_mcs),
                "material_radiation_length": bool(args.material_radiation_length),
                "transport_step_mm": (
                    float(args.transport_step_mm) if args.transport_step_mm is not None else None
                ),
                "air_offset_correction": bool(args.air_offset_correction),
                "fit_air_offset_mm": float(args.fit_air_offset_mm),
                "spr_stopping_volume": bool(args.spr),
            },
        }
        with metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics_out, fh, indent=2)
        print(
            f"  Speed-only compute: {pydosert_compute_s:.4g}s total, "
            f"{pydosert_compute_s / max(processed_beamlets, 1):.4g}s/beamlet, "
            f"{runtime_per_beam:.3f}s/beam"
        )
        print(f"  Timing → {metrics_path}")
        return metrics_out

    # -----------------------------------------------------------------------
    # Level 2 — plan-level metrics
    # -----------------------------------------------------------------------

    assert total_pred is not None and total_ref is not None
    nonzero_ref = total_ref[total_ref > 0]
    if args.prescription_dose_gy is not None:
        prescription_gy = float(args.prescription_dose_gy)
    elif nonzero_ref.size > 0:
        prescription_gy = float(np.percentile(nonzero_ref, 95))
    else:
        prescription_gy = 0.0
        print("  Warning: reference dose is all zeros; prescription_gy set to 0")

    stratified = _stratified_plan_mae(total_pred, total_ref, prescription_gy)
    dvh_surrogate = _dvh_surrogate(total_pred, total_ref, prescription_gy)
    for _k, _v in dvh_surrogate.items():
        print(f"  {_k:35s} = {_v:.5f}")

    gamma_pass_rate: float | None = None
    gamma_info: dict[str, object] = {}
    gamma_compute_s = 0.0
    if not args.skip_gamma:
        print(
            f"  Computing local gamma {args.gamma_dose_threshold}%/"
            f"{args.gamma_distance_threshold}mm with repo torch backend on {device}…"
        )
        try:
            gamma_t0 = time.perf_counter()
            gamma_pass_rate, gamma_info = _local_gamma_pass_rate(
                total_pred,
                total_ref,
                resolution_zyx,
                prescription_gy=prescription_gy,
                dose_pct=args.gamma_dose_threshold,
                dist_mm=args.gamma_distance_threshold,
                interp=args.gamma_interp_fraction,
                random_subset=args.gamma_random_subset,
                device=device,
            )
            gamma_compute_s = time.perf_counter() - gamma_t0
            if gamma_info.get("cropped"):
                print(
                    "  Gamma crop: "
                    f"{gamma_info['original_shape']} -> ref {gamma_info['reference_shape']}, "
                    f"eval {gamma_info['evaluation_shape']}"
                )
            if args.gamma_random_subset is not None:
                print(f"  Gamma random subset: {args.gamma_random_subset} reference points")
            print(f"  Gamma compute: {gamma_compute_s:.3f}s")
        except Exception as exc:
            print(f"  Gamma computation failed: {exc}")

    # -----------------------------------------------------------------------
    # Total comparison plot
    # -----------------------------------------------------------------------

    if not args.skip_figures:
        total_cmp_path = out_dir / f"{output_stem}_total_comparison.png"
        _plot_total_comparison(
            patient_id=f"{patient_id} [{pb_config_label}]",
            ct=ct_hu,
            ref_total=total_ref,
            pred_total=total_pred,
            scale=1.0,
            mask_fraction=args.mask_threshold_fraction,
            display_percentile=args.display_percentile,
            out_path=total_cmp_path,
        )
        print(f"  Saved total comparison: {total_cmp_path}")

    # -----------------------------------------------------------------------
    # Metrics dict
    # -----------------------------------------------------------------------

    per_beam_means: dict[str, float] = {}
    if per_beam_metrics:
        for k in next(iter(per_beam_metrics.values())):
            per_beam_means[k] = float(np.mean([v[k] for v in per_beam_metrics.values()]))

    metrics_out: dict = {
        "patient_id": patient_id,
        "prescription_gy": prescription_gy,
        "level1_per_beam": per_beam_metrics,
        "level1_per_beam_mean": per_beam_means,
        "level2_plan": {
            "stratified_mae": stratified,
            "gamma_pass_rate_pct": gamma_pass_rate,
            "gamma_criteria": (
                f"{args.gamma_dose_threshold}% / {args.gamma_distance_threshold}mm local"
            ),
            "gamma_info": gamma_info,
        },
        "timing": {
            "engine_total_s": pydosert_compute_s,
            "gamma_s": gamma_compute_s,
            "runtime_per_beam_s": runtime_per_beam,
            "n_beams_processed": n_beams_processed,
            "n_beamlets_processed": processed_beamlets,
        },
        "correction": {
            "checkpoint": str(args.correction_checkpoint) if args.correction_checkpoint is not None else None,
            "dense_field_size": list(dense_field_size) if dense_field_size is not None else None,
        },
        "pencil_beam": {
            "machine_mat": str(args.pyradplan_machine_mat),
            "lateral_model": str(args.lateral_model),
            "split_mode": str(args.split_mode),
            "heterogeneous_mcs": bool(args.heterogeneous_mcs),
            "material_radiation_length": bool(args.material_radiation_length),
            "transport_step_mm": (
                float(args.transport_step_mm) if args.transport_step_mm is not None else None
            ),
            "air_offset_correction": bool(args.air_offset_correction),
            "fit_air_offset_mm": float(args.fit_air_offset_mm),
            "spr_stopping_volume": bool(args.spr),
        },
    }

    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics_out, fh, indent=2)

    print(f"  Prescription dose: {prescription_gy:.4g} Gy")
    for k, v in per_beam_means.items():
        print(f"  Level1 {k:<30s} = {v:.5f}")
    for k, v in stratified.items():
        print(f"  stratified_mae_{k:<20s} = {v:.5f}")
    if gamma_pass_rate is not None:
        print(f"  gamma_pass_rate ({args.gamma_dose_threshold}%/{args.gamma_distance_threshold}mm) = {gamma_pass_rate:.2f}%")
    print(
        f"  Compute: {pydosert_compute_s:.4g}s total, "
        f"{pydosert_compute_s / max(processed_beamlets, 1):.4g}s/beamlet, "
        f"{runtime_per_beam:.3f}s/beam"
    )
    print(f"  Metrics → {metrics_path}")

    return metrics_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    torch.set_float32_matmul_precision("high")

    if args.speed_only:
        args.skip_gamma = True
        args.skip_figures = True
        args.range_map = False
        if args.per_beamlet_range:
            raise SystemExit("--speed-only is not compatible with --per-beamlet-range")
    if args.save_compile_cache and args.compile_cache_path is None:
        raise SystemExit("--save-compile-cache requires --compile-cache-path")

    if args.profile_dense_timing:
        os.environ["PYDOSERT_DENSE_ENGINE_TIMING"] = "1"
        os.environ["PYDOSERT_DENSE_HOOK_TIMING"] = "1"

    if args.split_mode is not None:
        from pydose_rt.engine import ion_dose_engine

        ion_dose_engine.SPLITTING_MODE = args.split_mode

    if args.n_per_dim is not None:
        from pydose_rt.engine import ion_dose_engine

        ion_dose_engine.N_PER_DIM = int(args.n_per_dim)
        print(f"[engine] N_PER_DIM={ion_dose_engine.N_PER_DIM} "
              f"({int(args.n_per_dim) ** 2} sub-beams/beamlet)")

    if args.no_correction:
        # Plain pencil beam. Needed to separate "how good is the analytic baseline"
        # from "how much does the correction net add" -- the two are not separable
        # from the corrected number alone.
        args.correction_checkpoint = None

    if args.case_dir is not None and args.cases_dir is not None:
        raise SystemExit("Error: --case-dir and --cases-dir are mutually exclusive")
    if args.case_dir is None and args.cases_dir is None:
        raise SystemExit("Error: pass either --case-dir <case> or --cases-dir <root>")

    if args.ray_index is not None and args.beam_index is None:
        raise SystemExit("Error: --ray-index requires --beam-index")
    if args.beamlet_index is not None and args.ray_index is None:
        raise SystemExit("Error: --beamlet-index requires --ray-index")

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = {"float16": torch.float16, "float32": torch.float32, "float64": torch.float64}[args.dtype]
    print(
        f"Torch device: {device} "
        f"(cuda_available={torch.cuda.is_available()}, cuda_devices={torch.cuda.device_count()})"
    )

    lut = PyRadPlanIonLUT(args.pyradplan_machine_mat)
    if args.sigma_mode == "focus" and not lut.has_initial_focus:
        raise RuntimeError(f"LUT does not provide initFocus data: {args.pyradplan_machine_mat}")

    machine_config = MachineConfig(
        tpr_20_10=0.7,
        number_of_leaf_pairs=40,
        fit_air_offset_mm=float(args.fit_air_offset_mm),
        bams_to_iso_dist_mm=float(args.bams_to_iso_dist_mm),
    )

    # Resolve beam_parameters path; in batch mode fall back to {cases_dir}/beam_parameters.json
    beam_params_path = args.beam_params_path.resolve()
    if not beam_params_path.exists() and args.cases_dir is not None:
        fallback = args.cases_dir.resolve() / "beam_parameters.json"
        if fallback.exists():
            print(f"Using beam_parameters.json from cases dir: {fallback}")
            beam_params_path = fallback
    beam_parameters = _load_json(beam_params_path)

    # -----------------------------------------------------------------------
    # Optional correction hook (dense BEV only)
    # -----------------------------------------------------------------------
    correction_hook = None
    eval_sparse_hooks = None
    compile_cache_loaded = False
    if args.correction_checkpoint is not None:
        from training.proton.hooks import ProtonDenseBevCorrectionHook

        correction_hook = ProtonDenseBevCorrectionHook.from_checkpoint(
            args.correction_checkpoint,
            device=device,
            dtype=dtype,
            available_energies=lut.available_energies,
            bev_crop_hw=args.dense_bev_crop_hw,
            max_inference_batch_items=args.dense_hook_batch_items,
            inference_amp=args.dense_hook_amp,
            tta=bool(getattr(args, "dense_tta", False)),
        )
        repvgg_blocks = sum(1 for module in correction_hook.model.modules() if hasattr(module, "reparam_deployed"))
        if hasattr(correction_hook.model, "fuse_repvgg"):
            correction_hook.model.fuse_repvgg()
        repvgg_deployed = sum(
            1
            for module in correction_hook.model.modules()
            if bool(getattr(module, "reparam_deployed", False))
        )
        if device.type == "cuda" and args.compile_correction_model:
            if args.compile_cache_path is not None and args.compile_cache_path.exists():
                if not hasattr(torch.compiler, "load_cache_artifacts"):
                    raise RuntimeError("This PyTorch build does not provide torch.compiler.load_cache_artifacts")
                torch.compiler.load_cache_artifacts(args.compile_cache_path.read_bytes())
                compile_cache_loaded = True
                print(f"Loaded torch.compile cache artifacts from {args.compile_cache_path}")
            correction_hook.model = torch.compile(
                correction_hook.model,
                dynamic=bool(args.compile_dynamic_shapes),
            )
        eval_sparse_hooks = IonSparseHooks(dense_bev=correction_hook)
        ckpt = torch.load(args.correction_checkpoint, map_location="cpu", weights_only=False)
        epoch = ckpt.get("epoch", "?")
        step = ckpt.get("step", "?")
        print(
            f"Loaded dense BEV correction hook from {args.correction_checkpoint} "
            f"(epoch={epoch} step={step}, bev_crop_hw={correction_hook.bev_crop_hw}, "
            f"repvgg_fused={repvgg_deployed}/{repvgg_blocks})"
        )
        del ckpt

    def _save_compile_cache_if_requested() -> None:
        if not (
            args.save_compile_cache
            and args.compile_cache_path is not None
            and device.type == "cuda"
            and args.compile_correction_model
        ):
            return
        if not hasattr(torch.compiler, "save_cache_artifacts"):
            raise RuntimeError("This PyTorch build does not provide torch.compiler.save_cache_artifacts")
        artifacts = torch.compiler.save_cache_artifacts()
        if artifacts is None:
            print("No torch.compile cache artifacts available to save")
            return
        artifact_bytes, cache_info = artifacts
        args.compile_cache_path.parent.mkdir(parents=True, exist_ok=True)
        args.compile_cache_path.write_bytes(artifact_bytes)
        print(
            f"Saved torch.compile cache artifacts to {args.compile_cache_path} "
            f"({len(artifact_bytes)} bytes, info={cache_info})"
        )

    def _call_run_case(case_dir: Path, out_dir: Path) -> dict | None:
        return _run_case(
            case_dir=case_dir,
            args=args,
            beam_parameters=beam_parameters,
            lut=lut,
            machine_config=machine_config,
            device=device,
            dtype=dtype,
            out_dir=out_dir,
            correction_hook=correction_hook,
            sparse_hooks=eval_sparse_hooks,
        )

    # -----------------------------------------------------------------------
    # Batch mode
    # -----------------------------------------------------------------------
    if args.cases_dir is not None:
        cases_dir = args.cases_dir.resolve()
        out_dir = (
            args.out_dir.resolve()
            if args.out_dir is not None
            else ROOT / "out" / f"eval_{cases_dir.name}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        case_dirs = sorted(p for p in cases_dir.iterdir() if p.is_dir())
        print(f"Batch mode: {len(case_dirs)} candidate directories in {cases_dir}")
        print(f"Output dir: {out_dir}")
        print()

        all_metrics: list[dict] = []
        failed_cases: list[dict] = []

        for i, case_dir in enumerate(case_dirs, start=1):
            print(f"[{i:3d}/{len(case_dirs)}] {case_dir.name}")
            try:
                metrics = _call_run_case(case_dir, out_dir)
            except Exception as exc:
                import traceback
                print(f"  [FAIL] {case_dir.name}: {exc}")
                traceback.print_exc()
                failed_cases.append({"case": case_dir.name, "error": str(exc)})
                continue
            if metrics is not None:
                all_metrics.append(metrics)

        print()
        print(f"{'─'*60}")
        print(f"Batch done: {len(all_metrics)} evaluated, {len(failed_cases)} failed")
        if failed_cases:
            print("  Failed:", [fc["case"] for fc in failed_cases])

        if all_metrics:
            agg = _aggregate_all_metrics(all_metrics)
            agg["failed_cases"] = failed_cases
            agg["compile_cache_loaded"] = compile_cache_loaded
            _print_aggregate_stats(agg)
            agg_path = out_dir / "aggregate_statistics.json"
            with agg_path.open("w", encoding="utf-8") as fh:
                json.dump(agg, fh, indent=2)
            print(f"Aggregate statistics → {agg_path}")

        _save_compile_cache_if_requested()

        return

    # -----------------------------------------------------------------------
    # Single-case mode
    # -----------------------------------------------------------------------
    case_dir = args.case_dir.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else ROOT / "out" / f"eval_{case_dir.name}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating case: {case_dir}")
    _call_run_case(case_dir, out_dir)
    _save_compile_cache_if_requested()


if __name__ == "__main__":
    main()
