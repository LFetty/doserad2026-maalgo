from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import SimpleITK as sitk

import fit_proton_lut as base
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEAM_PARAMS_PATH = ROOT / "example_data" / "beam_parameters.json"
DEFAULT_MACHINE_MAT_PATH = ROOT / "example_data" / "pyradplan" / "protons_Generic.mat"
DEFAULT_OUTPUT_MAT_PATH = ROOT / "example_data" / "mc_fit_smooth" / "lut_fast_3d_1e8.mat"
DEFAULT_SUMMARY_JSON_PATH = ROOT / "out" / "mc_fit_smooth" / "summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edep-dir",
        type=Path,
        required=True,
        help="Directory containing MC edep files named like '<particles>_<energy>MeV__edep.mhd'.",
    )
    parser.add_argument(
        "--beam-params-path",
        type=Path,
        default=DEFAULT_BEAM_PARAMS_PATH,
        help="DoseRAD beam_parameters.json",
    )
    parser.add_argument(
        "--machine-mat-path",
        type=Path,
        default=DEFAULT_MACHINE_MAT_PATH,
        help="Reference pyRadPlan proton machine .mat file",
    )
    parser.add_argument(
        "--output-mat-path",
        type=Path,
        default=DEFAULT_OUTPUT_MAT_PATH,
        help="Output pyRadPlan-style LUT .mat path.",
    )
    parser.add_argument(
        "--summary-json-path",
        type=Path,
        default=DEFAULT_SUMMARY_JSON_PATH,
        help="Per-energy export summary JSON.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Parallel worker count for per-energy fitting.",
    )
    parser.add_argument(
        "--match",
        type=str,
        default=None,
        help="Comma-separated filename substrings; only matching edep files are fit (others "
        "in the existing output LUT are kept untouched, since the export merges into it).",
    )
    parser.add_argument(
        "--kernel-width-mm",
        type=float,
        default=base.KERNEL_WIDTH_MM,
        help="Full lateral width (mm) over which the MC dose is integrated for the depth dose, matching the pencil-beam kernel support.",
    )
    parser.add_argument(
        "--longitudinal-smoothing-mm",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma in mm applied along depth to exported dense sigma/sigma1/sigma2 curves. Set to 0 to disable.",
    )
    parser.add_argument(
        "--post-peak-margin-mm",
        type=float,
        default=5.0,
        help="Use MC-fitted lateral parameters only up to (peak depth + margin) in mm; farther depths keep reference lateral parameters.",
    )
    parser.add_argument(
        "--sigma-fit-min-z-rel",
        type=float,
        default=5e-3,
        help="Minimum relative integrated depth-dose value required for fitting MC lateral sigma. Lower-dose tails keep reference lateral parameters.",
    )
    parser.add_argument(
        "--sigma2-high-energy-start-mev",
        type=float,
        default=None,
        help="Optional energy threshold where exported sigma2 is multiplied by --sigma2-high-energy-scale.",
    )
    parser.add_argument(
        "--sigma2-high-energy-scale",
        type=float,
        default=1.0,
        help="Optional scale factor for sigma2 at energies >= --sigma2-high-energy-start-mev.",
    )
    parser.add_argument(
        "--double-fit-mode",
        type=str,
        default="scaled_reference",
        choices=("scaled_reference", "direct"),
        help="How to export sigma1/sigma2/weight. 'direct' fits the normalized double-Gaussian lateral shape to MC profiles.",
    )
    parser.add_argument(
        "--direct-double-fit-step-mm",
        type=float,
        default=1.0,
        help="Depth spacing for direct double-Gaussian fits. Intermediate dense depths are interpolated and smoothed.",
    )
    parser.add_argument(
        "--direct-double-fit-min-energy-mev",
        type=float,
        default=None,
        help="Optional minimum energy for direct double-Gaussian fitting; lower energies use scaled-reference halo export.",
    )
    parser.add_argument(
        "--halo-tuning-preset",
        type=str,
        default="none",
        choices=("none", "mc_lateral_v1"),
        help="Optional energy-banded sigma2/weight tuning calibrated against MC lateral profiles.",
    )
    return parser.parse_args()


def _collect_edep_paths(edep_dir: Path) -> list[Path]:
    paths = sorted(edep_dir.glob("*__edep.mhd"))
    if not paths:
        raise FileNotFoundError(f"No '*__edep.mhd' files found in {edep_dir}")

    seen_energies: dict[float, Path] = {}
    selected: list[Path] = []
    for path in paths:
        energy_mev, source_particles = base.parse_edep_filename(path)
        if energy_mev is None or source_particles is None:
            continue
        key = round(float(energy_mev), 6)
        if key in seen_energies:
            raise ValueError(
                f"Duplicate MC edep files for energy {energy_mev:.6f} MeV: "
                f"{seen_energies[key]} and {path}"
            )
        seen_energies[key] = path
        selected.append(path)

    if not selected:
        raise FileNotFoundError(
            f"Found edep files in {edep_dir}, but none matched the expected '<particles>_<energy>MeV__edep.mhd' pattern"
        )
    return selected


def _fit_dense_single_sigma_curve(
    x_fit: np.ndarray,
    profiles_by_depth: np.ndarray,
    depth_indices: np.ndarray,
    depths_mm: np.ndarray,
    sigma_guesses_total: np.ndarray,
    sigma0_mm: float,
    fallback_sigma_lut: np.ndarray,
    low_dose_threshold: float,
    fit_depth_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_depths = len(depth_indices)
    single_sigma_total_raw = np.full(n_depths, np.nan, dtype=np.float64)
    fit_ok_single = np.zeros(n_depths, dtype=bool)
    used_fallback = np.zeros(n_depths, dtype=bool)
    if fit_depth_mask is None:
        fit_depth_mask = np.ones(n_depths, dtype=bool)
    else:
        fit_depth_mask = np.asarray(fit_depth_mask, dtype=bool)

    prev_single_sigma_total = float("nan")
    for j, depth_idx in enumerate(depth_indices):
        if not fit_depth_mask[j]:
            continue
        y_fit = profiles_by_depth[:, int(depth_idx)]
        peak = float(np.nanmax(y_fit))
        if not np.isfinite(peak) or peak <= low_dose_threshold:
            continue

        sigma_single_guess = (
            prev_single_sigma_total if np.isfinite(prev_single_sigma_total) else sigma_guesses_total[j]
        )
        try:
            _, sigma_single_total = base.fit_single_gaussian(
                x_fit=x_fit,
                y_fit=y_fit,
                sigma_guess=sigma_single_guess,
            )
            single_sigma_total_raw[j] = sigma_single_total
            fit_ok_single[j] = True
            prev_single_sigma_total = sigma_single_total
        except Exception:
            single_sigma_total_raw[j] = sigma_single_guess
            prev_single_sigma_total = sigma_single_guess
            used_fallback[j] = True

    single_sigma_raw = np.sqrt(np.maximum(single_sigma_total_raw**2 - sigma0_mm**2, 0.0))
    single_sigma_raw[~fit_depth_mask] = fallback_sigma_lut[~fit_depth_mask]
    single_sigma_lut = np.clip(
        base.fill_curve_from_valid(depths_mm, single_sigma_raw, fallback_sigma_lut),
        0.0,
        None,
    )
    return single_sigma_total_raw, single_sigma_lut, fit_ok_single | used_fallback


def _fit_dense_double_curve(
    x_fit: np.ndarray,
    profiles_by_depth: np.ndarray,
    depth_indices: np.ndarray,
    depths_mm: np.ndarray,
    sigma_single_total_raw: np.ndarray,
    sigma0_mm: float,
    sigma1_guess_total: np.ndarray,
    sigma2_prior_total: np.ndarray,
    weight_guess_total: np.ndarray,
    ref_sigma_total: np.ndarray,
    ref_sigma1_total: np.ndarray,
    ref_sigma2_total: np.ndarray,
    fallback_sigma1_lut: np.ndarray,
    fallback_sigma2_lut: np.ndarray,
    fallback_weight: np.ndarray,
    low_dose_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_depths = len(depth_indices)
    sigma1_total_raw = np.full(n_depths, np.nan, dtype=np.float64)
    sigma2_total_raw = np.full(n_depths, np.nan, dtype=np.float64)
    weight_raw = np.full(n_depths, np.nan, dtype=np.float64)
    fit_ok_double = np.zeros(n_depths, dtype=bool)
    repr_ok_double = np.zeros(n_depths, dtype=bool)
    fallback_used = np.zeros(n_depths, dtype=bool)

    prev_double_sigma1_total = float("nan")
    prev_double_weight = float("nan")

    for j, depth_idx in enumerate(depth_indices):
        y_fit = profiles_by_depth[:, int(depth_idx)]
        peak = float(np.nanmax(y_fit))
        if not np.isfinite(peak) or peak <= low_dose_threshold:
            continue

        sigma_single_total = sigma_single_total_raw[j]
        if not np.isfinite(sigma_single_total):
            sigma_single_total = ref_sigma_total[j]

        sigma1_guess = (
            prev_double_sigma1_total if np.isfinite(prev_double_sigma1_total) else sigma1_guess_total[j]
        )
        weight_guess = prev_double_weight if np.isfinite(prev_double_weight) else weight_guess_total[j]

        result = None
        try:
            result = base.fit_double_gaussian_safe(
                x_fit=x_fit,
                y_fit=y_fit,
                sigma_single_total=sigma_single_total,
                sigma1_guess=sigma1_guess,
                sigma2_prior=sigma2_prior_total[j],
                weight_guess=weight_guess,
            )
        except Exception:
            result = None

        if result is None:
            result = base.fallback_double_gaussian_from_single(
                x_fit=x_fit,
                y_fit=y_fit,
                sigma_single_total=sigma_single_total,
                sigma0_mm=sigma0_mm,
                ref_sigma_total=ref_sigma_total[j],
                ref_sigma1_total=ref_sigma1_total[j],
                ref_sigma2_total=ref_sigma2_total[j],
                ref_weight=weight_guess_total[j],
            )
            fallback_used[j] = result is not None
        else:
            fit_ok_double[j] = True

        if result is None:
            continue

        sigma1_total_raw[j] = result["sigma1_total_mm"]
        sigma2_total_raw[j] = result["sigma2_total_mm"]
        weight_raw[j] = result["weight"]
        repr_ok_double[j] = True
        prev_double_sigma1_total = result["sigma1_total_mm"]
        prev_double_weight = result["weight"]

    sigma1_raw = np.sqrt(np.maximum(sigma1_total_raw**2 - sigma0_mm**2, 0.0))
    sigma2_raw = np.sqrt(np.maximum(sigma2_total_raw**2 - sigma0_mm**2, 0.0))
    sigma1_lut = np.clip(base.fill_curve_from_valid(depths_mm, sigma1_raw, fallback_sigma1_lut), 0.0, None)
    sigma2_lut = np.clip(base.fill_curve_from_valid(depths_mm, sigma2_raw, fallback_sigma2_lut), 0.0, None)
    weight_lut = np.clip(base.fill_curve_from_valid(depths_mm, weight_raw, fallback_weight), 0.0, base.WEIGHT_MAX)
    sigma2_lut = np.maximum(sigma2_lut, sigma1_lut + base.SIGMA_MIN_MM)
    return sigma1_lut, sigma2_lut, weight_lut, fit_ok_double, repr_ok_double, fallback_used


def _normalized_double_gaussian(
    x_mm: np.ndarray,
    sigma1_total_mm: float,
    sigma2_total_mm: float,
    weight: float,
) -> np.ndarray:
    profile = base.double_gaussian_profile_1d_amplitude_weight(
        x_mm,
        1.0,
        sigma1_total_mm,
        sigma2_total_mm,
        weight,
    )
    peak = float(np.nanmax(profile))
    if not np.isfinite(peak) or peak <= 0.0:
        return np.zeros_like(x_mm, dtype=np.float64)
    return profile / peak


def _fit_direct_double_gaussian_shape(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    sigma0_mm: float,
    init_sigma1_lut: float,
    init_sigma2_lut: float,
    init_weight: float,
) -> tuple[float, float, float] | None:
    peak = float(np.nanmax(y_fit))
    if not np.isfinite(peak) or peak <= 0.0:
        return None

    keep = np.isfinite(x_fit) & np.isfinite(y_fit)
    keep &= np.abs(x_fit) <= base.FIT_RADIUS_MM
    keep &= y_fit >= 0.0
    keep &= y_fit > peak * 1e-5
    if keep.sum() < base.MIN_DOUBLE_FIT_POINTS:
        return None

    x = np.asarray(x_fit[keep], dtype=np.float64)
    y_norm = np.asarray(y_fit[keep], dtype=np.float64) / peak
    residual_weights = np.sqrt(np.clip(y_norm, 0.02, 1.0))

    init_sigma1_lut = max(float(init_sigma1_lut), 0.0)
    init_sigma2_lut = max(float(init_sigma2_lut), init_sigma1_lut + base.SIGMA_MIN_MM)
    init_weight = float(np.clip(init_weight, 0.0, base.WEIGHT_MAX))
    init_sigma1_total = float(np.sqrt(float(sigma0_mm) ** 2 + init_sigma1_lut**2))
    init_sigma2_total = float(np.sqrt(float(sigma0_mm) ** 2 + init_sigma2_lut**2))
    init_sigma2_total = max(init_sigma2_total, init_sigma1_total + base.SIGMA_MIN_MM)
    p0 = np.array([init_sigma1_total, init_sigma2_total, np.clip(init_weight, 1e-5, 0.8)], dtype=np.float64)

    def residual(params: np.ndarray) -> np.ndarray:
        sigma1_total = float(params[0])
        sigma2_total = max(float(params[1]), sigma1_total + base.SIGMA_MIN_MM)
        weight = float(np.clip(params[2], 0.0, 0.8))
        pred = _normalized_double_gaussian(x, sigma1_total, sigma2_total, weight)
        regularization = np.array(
            [
                0.02 * (sigma1_total - p0[0]) / max(p0[0], 1.0),
                0.01 * (sigma2_total - p0[1]) / max(p0[1], 1.0),
                0.01 * (weight - p0[2]) / max(p0[2], 0.05),
            ],
            dtype=np.float64,
        )
        return np.concatenate([(pred - y_norm) * residual_weights, regularization])

    try:
        result = least_squares(
            residual,
            p0,
            bounds=(
                np.array([max(0.1, 0.5 * sigma0_mm), max(0.2, 0.5 * sigma0_mm), 0.0], dtype=np.float64),
                np.array([base.SIGMA1_MAX_MM, base.SIGMA2_MAX_MM, 0.8], dtype=np.float64),
            ),
            max_nfev=300,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
    except Exception:
        return None

    sigma1_total = float(result.x[0])
    sigma2_total = max(float(result.x[1]), sigma1_total + base.SIGMA_MIN_MM)
    weight = float(np.clip(result.x[2], 0.0, base.WEIGHT_MAX))
    sigma1_lut = float(np.sqrt(max(sigma1_total**2 - float(sigma0_mm) ** 2, 0.0)))
    sigma2_lut = float(np.sqrt(max(sigma2_total**2 - float(sigma0_mm) ** 2, 0.0)))
    sigma2_lut = max(sigma2_lut, sigma1_lut + base.SIGMA_MIN_MM)
    return sigma1_lut, sigma2_lut, weight


def _fit_dense_direct_double_curve(
    x_fit: np.ndarray,
    profiles_by_depth: np.ndarray,
    depth_indices: np.ndarray,
    depths_mm: np.ndarray,
    sigma0_mm: float,
    init_sigma1_lut: np.ndarray,
    init_sigma2_lut: np.ndarray,
    init_weight: np.ndarray,
    fit_depth_mask: np.ndarray,
    fit_step_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_depths = len(depth_indices)
    sigma1_raw = np.full(n_depths, np.nan, dtype=np.float64)
    sigma2_raw = np.full(n_depths, np.nan, dtype=np.float64)
    weight_raw = np.full(n_depths, np.nan, dtype=np.float64)
    fit_ok = np.zeros(n_depths, dtype=bool)

    fit_depth_mask = np.asarray(fit_depth_mask, dtype=bool)
    valid_positions = np.flatnonzero(fit_depth_mask)
    if valid_positions.size == 0:
        return init_sigma1_lut.copy(), init_sigma2_lut.copy(), init_weight.copy(), fit_ok

    positive_step = max(float(fit_step_mm), 0.0)
    if positive_step > 0.0 and valid_positions.size > 1:
        depth_step = float(np.median(np.diff(depths_mm)))
        stride = max(1, int(round(positive_step / max(depth_step, 1e-12))))
    else:
        stride = 1
    fit_positions = valid_positions[::stride]
    if fit_positions[-1] != valid_positions[-1]:
        fit_positions = np.append(fit_positions, valid_positions[-1])

    for j in fit_positions:
        depth_idx = int(depth_indices[j])
        result = _fit_direct_double_gaussian_shape(
            x_fit=x_fit,
            y_fit=profiles_by_depth[:, depth_idx],
            sigma0_mm=sigma0_mm,
            init_sigma1_lut=init_sigma1_lut[j],
            init_sigma2_lut=init_sigma2_lut[j],
            init_weight=init_weight[j],
        )
        if result is None:
            continue
        sigma1_raw[j], sigma2_raw[j], weight_raw[j] = result
        fit_ok[j] = True

    sigma1 = base.fill_curve_from_valid(depths_mm, sigma1_raw, init_sigma1_lut)
    sigma2 = base.fill_curve_from_valid(depths_mm, sigma2_raw, init_sigma2_lut)
    weight = base.fill_curve_from_valid(depths_mm, weight_raw, init_weight)
    sigma1[~fit_depth_mask] = init_sigma1_lut[~fit_depth_mask]
    sigma2[~fit_depth_mask] = init_sigma2_lut[~fit_depth_mask]
    weight[~fit_depth_mask] = init_weight[~fit_depth_mask]
    sigma2 = np.maximum(sigma2, sigma1 + base.SIGMA_MIN_MM)
    return sigma1, sigma2, np.clip(weight, 0.0, base.WEIGHT_MAX), fit_ok


def _halo_tuning_scales(energy_mev: float, preset: str) -> tuple[float, float]:
    if preset == "none":
        return 1.0, 1.0
    if preset == "mc_lateral_v1":
        # Grid-search optimum on the MC lateral centerline objective:
        # low energies were already close to pyRadPlan; mid/high energies need
        # a narrower broad component with more halo weight.
        if energy_mev < 100.0:
            return 0.70, 1.20
        if energy_mev < 160.0:
            return 0.55, 2.00
        return 0.65, 2.00
    raise ValueError(f"Unknown halo tuning preset: {preset}")


def _process_single_edep(job: tuple[str, str, str, float, float, float, float, float | None, float, str, float, float | None, str]) -> dict[str, Any]:
    (
        edep_path_str,
        beam_params_path_str,
        machine_mat_path_str,
        kernel_width_mm,
        longitudinal_smoothing_mm,
        post_peak_margin_mm,
        sigma_fit_min_z_rel,
        sigma2_high_energy_start_mev,
        sigma2_high_energy_scale,
        double_fit_mode,
        direct_double_fit_step_mm,
        direct_double_fit_min_energy_mev,
        halo_tuning_preset,
    ) = job
    edep_path = Path(edep_path_str)
    beam_params_path = Path(beam_params_path_str)
    machine_mat_path = Path(machine_mat_path_str)

    beam_parameters = json.loads(beam_params_path.read_text())
    inferred_energy_mev, inferred_source_particles = base.parse_edep_filename(edep_path)
    if inferred_energy_mev is None or inferred_source_particles is None:
        raise ValueError(f"Could not infer energy/source_particles from {edep_path.name}")

    energy_mev = float(inferred_energy_mev)
    source_particles = float(inferred_source_particles)
    ref = base.load_reference_entry(machine_mat_path, energy_mev)
    sigma0_mm = base.dose_rad_sigma_spot_mm(beam_parameters, energy_mev)

    edep_img = sitk.ReadImage(str(edep_path))
    z_from_edep = base.build_windowed_integral_image(edep_img, source_particles, kernel_width_mm)
    integral_img = np.asarray(z_from_edep["integral_img"], dtype=np.float64)  # (lateral_y, depth)
    ref_depths_mm = np.asarray(ref["depths_mm"], dtype=np.float64)
    ref_sigma = np.asarray(ref["sigma_mm"], dtype=np.float64)
    ref_sigma1 = np.asarray(ref["sigma1_mm"], dtype=np.float64)
    ref_sigma2 = np.asarray(ref["sigma2_mm"], dtype=np.float64)
    ref_weight = np.asarray(ref["weight"], dtype=np.float64)
    ref_let = np.asarray(ref["LET"], dtype=np.float64)

    mc_depth_mm_all = np.asarray(z_from_edep["depth_mm"], dtype=np.float64)
    mc_z_all = np.asarray(z_from_edep["Z_est"], dtype=np.float64)
    mc_peak_idx = int(np.nanargmax(mc_z_all)) if np.isfinite(mc_z_all).any() else 0
    mc_peak_depth_mm = float(mc_depth_mm_all[mc_peak_idx]) if mc_depth_mm_all.size else 0.0
    max_export_depth_mm = min(float(mc_depth_mm_all[-1]), float(ref_depths_mm[-1]))
    dense_depth_keep = mc_depth_mm_all <= max_export_depth_mm
    dense_depth_indices = np.flatnonzero(dense_depth_keep)
    dense_depths_mm = mc_depth_mm_all[dense_depth_keep]
    dense_z_curve = np.clip(mc_z_all[dense_depth_keep], 0.0, None)
    dense_z_peak = float(np.nanmax(dense_z_curve)) if dense_z_curve.size else 0.0
    dense_lateral_fit_mask = np.ones_like(dense_z_curve, dtype=bool)
    dense_lateral_fit_mask &= dense_depths_mm <= (mc_peak_depth_mm + float(post_peak_margin_mm))
    if np.isfinite(dense_z_peak) and dense_z_peak > 0.0:
        dense_lateral_fit_mask &= dense_z_curve >= (float(sigma_fit_min_z_rel) * dense_z_peak)

    dense_ref_sigma = base.interp_reference_curve(ref_depths_mm, ref_sigma, dense_depths_mm)
    dense_ref_sigma1 = base.interp_reference_curve(ref_depths_mm, ref_sigma1, dense_depths_mm)
    dense_ref_sigma2 = base.interp_reference_curve(ref_depths_mm, ref_sigma2, dense_depths_mm)
    dense_ref_weight = base.interp_reference_curve(ref_depths_mm, ref_weight, dense_depths_mm)
    dense_ref_let = base.interp_reference_curve(ref_depths_mm, ref_let, dense_depths_mm)
    dense_ref_sigma_total = np.sqrt(sigma0_mm**2 + dense_ref_sigma**2)

    y_mm = np.asarray(z_from_edep["y_mm"], dtype=np.float64)
    fit_mask = np.abs(y_mm) <= base.FIT_RADIUS_MM
    x_fit = y_mm[fit_mask]
    profiles_by_depth = integral_img[fit_mask, :].astype(np.float64, copy=False)

    global_peak = float(np.nanmax(integral_img))
    low_dose_threshold = max(global_peak * base.LOW_DOSE_REL_THRESHOLD, base.LOW_DOSE_ABS_THRESHOLD)

    dense_single_sigma_total_raw, dense_single_sigma_lut_unsmoothed, dense_fit_ok_single = _fit_dense_single_sigma_curve(
        x_fit=x_fit,
        profiles_by_depth=profiles_by_depth,
        depth_indices=dense_depth_indices,
        depths_mm=dense_depths_mm,
        sigma_guesses_total=dense_ref_sigma_total,
        sigma0_mm=sigma0_mm,
        fallback_sigma_lut=dense_ref_sigma,
        low_dose_threshold=low_dose_threshold,
        fit_depth_mask=dense_lateral_fit_mask,
    )

    # Match scripts/fit_proton_lut.py export semantics exactly: export the MC-fitted
    # single sigma, but keep the halo in the pyRadPlan parameter space by scaling
    # reference sigma1/sigma2 with the MC-derived single-sigma ratio and keeping
    # the reference halo weight.
    halo_scale = np.ones_like(dense_single_sigma_lut_unsmoothed)
    scale_mask = dense_ref_sigma > base.SIGMA_MIN_MM
    halo_scale[scale_mask] = dense_single_sigma_lut_unsmoothed[scale_mask] / dense_ref_sigma[scale_mask]
    halo_scale = np.clip(halo_scale, 0.5, 2.0)
    dense_sigma1_lut_unsmoothed = np.clip(dense_ref_sigma1 * halo_scale, 0.0, None)
    dense_sigma2_lut_unsmoothed = np.clip(dense_ref_sigma2 * halo_scale, 0.0, None)
    dense_weight_lut = np.clip(dense_ref_weight.copy(), 0.0, base.WEIGHT_MAX)
    dense_sigma1_lut_unsmoothed[~dense_lateral_fit_mask] = dense_ref_sigma1[~dense_lateral_fit_mask]
    dense_sigma2_lut_unsmoothed[~dense_lateral_fit_mask] = dense_ref_sigma2[~dense_lateral_fit_mask]
    dense_weight_lut[~dense_lateral_fit_mask] = dense_ref_weight[~dense_lateral_fit_mask]
    dense_sigma2_lut_unsmoothed = np.maximum(
        dense_sigma2_lut_unsmoothed,
        dense_sigma1_lut_unsmoothed + base.SIGMA_MIN_MM,
    )
    direct_double_fit_valid = 0
    direct_fit_allowed = double_fit_mode == "direct" and (
        direct_double_fit_min_energy_mev is None or energy_mev >= float(direct_double_fit_min_energy_mev)
    )
    if direct_fit_allowed:
        (
            dense_sigma1_lut_unsmoothed,
            dense_sigma2_lut_unsmoothed,
            dense_weight_lut,
            dense_direct_fit_ok,
        ) = _fit_dense_direct_double_curve(
            x_fit=x_fit,
            profiles_by_depth=profiles_by_depth,
            depth_indices=dense_depth_indices,
            depths_mm=dense_depths_mm,
            sigma0_mm=sigma0_mm,
            init_sigma1_lut=dense_sigma1_lut_unsmoothed,
            init_sigma2_lut=dense_sigma2_lut_unsmoothed,
            init_weight=dense_weight_lut,
            fit_depth_mask=dense_lateral_fit_mask,
            fit_step_mm=direct_double_fit_step_mm,
        )
        direct_double_fit_valid = int(np.sum(dense_direct_fit_ok))

    dense_single_sigma_lut = np.clip(
        base.smooth_curve_along_depth(dense_depths_mm, dense_single_sigma_lut_unsmoothed, longitudinal_smoothing_mm),
        0.0,
        None,
    )
    dense_sigma1_lut = np.clip(
        base.smooth_curve_along_depth(dense_depths_mm, dense_sigma1_lut_unsmoothed, longitudinal_smoothing_mm),
        0.0,
        None,
    )
    dense_sigma2_lut = np.clip(
        base.smooth_curve_along_depth(dense_depths_mm, dense_sigma2_lut_unsmoothed, longitudinal_smoothing_mm),
        0.0,
        None,
    )
    if double_fit_mode == "direct":
        dense_weight_lut = np.clip(
            base.smooth_curve_along_depth(dense_depths_mm, dense_weight_lut, longitudinal_smoothing_mm),
            0.0,
            base.WEIGHT_MAX,
        )
    halo_sigma2_scale, halo_weight_scale = _halo_tuning_scales(energy_mev, halo_tuning_preset)
    if halo_sigma2_scale != 1.0 or halo_weight_scale != 1.0:
        dense_sigma2_lut = dense_sigma2_lut * halo_sigma2_scale
        dense_weight_lut = np.clip(dense_weight_lut * halo_weight_scale, 0.0, base.WEIGHT_MAX)
    sigma2_scale_applied = 1.0
    if sigma2_high_energy_start_mev is not None and energy_mev >= float(sigma2_high_energy_start_mev):
        sigma2_scale_applied = float(sigma2_high_energy_scale)
        dense_sigma2_lut = dense_sigma2_lut * sigma2_scale_applied
    dense_sigma2_lut = np.maximum(dense_sigma2_lut, dense_sigma1_lut + base.SIGMA_MIN_MM)

    valid_dense_depth_mask = dense_z_curve > 0.05 * np.nanmax(dense_z_curve)
    mae_dense_sigma = float(
        np.nanmean(np.abs(dense_single_sigma_lut[valid_dense_depth_mask] - dense_ref_sigma[valid_dense_depth_mask]))
    ) if np.any(valid_dense_depth_mask) else float("nan")
    mae_dense_sigma1 = float(
        np.nanmean(np.abs(dense_sigma1_lut[valid_dense_depth_mask] - dense_ref_sigma1[valid_dense_depth_mask]))
    ) if np.any(valid_dense_depth_mask) else float("nan")
    mae_dense_sigma2 = float(
        np.nanmean(np.abs(dense_sigma2_lut[valid_dense_depth_mask] - dense_ref_sigma2[valid_dense_depth_mask]))
    ) if np.any(valid_dense_depth_mask) else float("nan")

    return {
        "energy_mev": energy_mev,
        "edep_path": str(edep_path),
        "source_particles": source_particles,
        "sigma0_mm": sigma0_mm,
        "depths_mm": dense_depths_mm,
        "z_curve": dense_z_curve,
        "sigma_mm": dense_single_sigma_lut,
        "sigma1_mm": dense_sigma1_lut,
        "sigma2_mm": dense_sigma2_lut,
        "weight": dense_weight_lut,
        "let_curve": dense_ref_let,
        "summary": {
            "energy_mev": energy_mev,
            "edep_path": str(edep_path),
            "source_particles": source_particles,
            "sigma0_mm": sigma0_mm,
            "dense_depth_bins": int(len(dense_depths_mm)),
            "dense_single_valid": int(np.sum(dense_fit_ok_single)),
            "dense_lateral_fit_bins": int(np.sum(dense_lateral_fit_mask)),
            "dense_lateral_reference_fallback_bins": int(len(dense_lateral_fit_mask) - np.sum(dense_lateral_fit_mask)),
            "halo_export_mode": (
                "direct_normalized_double_fit"
                if direct_fit_allowed
                else "reference_scaled_by_single_sigma"
            ),
            "direct_double_fit_step_mm": float(direct_double_fit_step_mm),
            "direct_double_fit_min_energy_mev": (
                None if direct_double_fit_min_energy_mev is None else float(direct_double_fit_min_energy_mev)
            ),
            "direct_double_fit_valid": int(direct_double_fit_valid),
            "longitudinal_smoothing_mm": float(longitudinal_smoothing_mm),
            "post_peak_margin_mm": float(post_peak_margin_mm),
            "sigma_fit_min_z_rel": float(sigma_fit_min_z_rel),
            "sigma2_high_energy_start_mev": (
                None if sigma2_high_energy_start_mev is None else float(sigma2_high_energy_start_mev)
            ),
            "sigma2_high_energy_scale": float(sigma2_high_energy_scale),
            "sigma2_scale_applied": float(sigma2_scale_applied),
            "halo_tuning_preset": str(halo_tuning_preset),
            "halo_tuning_sigma2_scale": float(halo_sigma2_scale),
            "halo_tuning_weight_scale": float(halo_weight_scale),
            "mc_peak_depth_mm": float(mc_peak_depth_mm),
            "max_export_depth_mm": float(max_export_depth_mm),
            "dense_single_sigma_mae_mm": mae_dense_sigma,
            "dense_sigma1_mae_mm": mae_dense_sigma1,
            "dense_sigma2_mae_mm": mae_dense_sigma2,
        },
    }


def _update_machine_entry(entries: np.ndarray, result: dict[str, Any]) -> None:
    entry_idx = base.find_machine_entry_index_from_struct(entries, float(result["energy_mev"]))
    depths_mm = np.asarray(result["depths_mm"], dtype=np.float64)
    z_curve = np.asarray(result["z_curve"], dtype=np.float64)

    entries["depths"][0, entry_idx] = base.as_mat_column(depths_mm)
    entries["Z"][0, entry_idx] = base.as_mat_column(result["z_curve"])
    entries["sigma"][0, entry_idx] = base.as_mat_column(result["sigma_mm"])
    entries["sigma1"][0, entry_idx] = base.as_mat_column(result["sigma1_mm"])
    entries["sigma2"][0, entry_idx] = base.as_mat_column(result["sigma2_mm"])
    entries["weight"][0, entry_idx] = base.as_mat_column(result["weight"])
    entries["LET"][0, entry_idx] = base.as_mat_column(result["let_curve"])

    peak_idx = int(np.nanargmax(z_curve)) if np.isfinite(z_curve).any() else 0
    entries["peakPos"][0, entry_idx] = np.array([[float(depths_mm[peak_idx])]], dtype=np.float64)


def main() -> None:
    args = parse_args()
    edep_paths = _collect_edep_paths(args.edep_dir)
    if args.match:
        tokens = [t.strip() for t in args.match.split(",") if t.strip()]
        edep_paths = [p for p in edep_paths if any(t in p.name for t in tokens)]
        if not edep_paths:
            raise SystemExit(f"--match {args.match!r} matched no edep files in {args.edep_dir}")

    jobs = [
        (
            str(path),
            str(args.beam_params_path),
            str(args.machine_mat_path),
            float(args.kernel_width_mm),
            float(args.longitudinal_smoothing_mm),
            float(args.post_peak_margin_mm),
            float(args.sigma_fit_min_z_rel),
            None if args.sigma2_high_energy_start_mev is None else float(args.sigma2_high_energy_start_mev),
            float(args.sigma2_high_energy_scale),
            str(args.double_fit_mode),
            float(args.direct_double_fit_step_mm),
            None if args.direct_double_fit_min_energy_mev is None else float(args.direct_double_fit_min_energy_mev),
            str(args.halo_tuning_preset),
        )
        for path in edep_paths
    ]

    print(f"Found {len(jobs)} proton MC edep files in {args.edep_dir}")
    print(f"Using {args.jobs} worker(s)")
    print(f"Kernel integration width: {args.kernel_width_mm:.3f} mm full width")
    print(f"Longitudinal smoothing sigma: {args.longitudinal_smoothing_mm:.3f} mm")
    print(f"Post-peak export margin: {args.post_peak_margin_mm:.3f} mm")
    print(f"Minimum relative Z for lateral fitting: {args.sigma_fit_min_z_rel:.6g}")
    if args.sigma2_high_energy_start_mev is not None:
        print(
            "High-energy sigma2 scale: "
            f"E >= {args.sigma2_high_energy_start_mev:.3f} MeV -> x{args.sigma2_high_energy_scale:.6g}"
        )
    print(f"Double-Gaussian export mode: {args.double_fit_mode}")
    print(f"Halo tuning preset: {args.halo_tuning_preset}")
    if args.double_fit_mode == "direct":
        print(f"Direct double-Gaussian fit step: {args.direct_double_fit_step_mm:.3f} mm")
        if args.direct_double_fit_min_energy_mev is not None:
            print(f"Direct double-Gaussian min energy: {args.direct_double_fit_min_energy_mev:.3f} MeV")

    import time as _time

    results: list[dict[str, Any]] = []
    total = len(jobs)
    t_start = _time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        future_map = {executor.submit(_process_single_edep, job): job[0] for job in jobs}
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            summary = result["summary"]
            n_done = len(results)
            elapsed = _time.perf_counter() - t_start
            rate = elapsed / max(n_done, 1)
            eta = rate * (total - n_done)
            print(
                f"[{n_done:>2}/{total}] done {summary['energy_mev']:.4f} MeV | "
                f"dense bins={summary['dense_depth_bins']} | "
                f"single valid={summary['dense_single_valid']} | "
                f"σ MAE={summary['dense_single_sigma_mae_mm']:.4f} mm | "
                f"elapsed {elapsed/60:.1f} min | ETA {eta/60:.1f} min",
                flush=True,
            )

    results.sort(key=lambda item: float(item["energy_mev"]))
    summaries = [item["summary"] for item in results]

    base_mat_path = args.output_mat_path if args.output_mat_path.exists() else args.machine_mat_path
    raw = sio.loadmat(str(base_mat_path), struct_as_record=True, squeeze_me=False)
    machine = raw["machine"]
    entries = machine["data"][0, 0]
    for result in results:
        _update_machine_entry(entries, result)

    args.output_mat_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(args.output_mat_path), {"machine": machine}, do_compression=True)

    args.summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json_path.write_text(json.dumps(summaries, indent=2))

    print()
    print(f"Wrote LUT to {args.output_mat_path}")
    print(f"Wrote summary to {args.summary_json_path}")


if __name__ == "__main__":
    main()
