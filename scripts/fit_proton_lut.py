from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pydosert-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEAM_PARAMS_PATH = ROOT / "example_data" / "beam_parameters.json"
DEFAULT_MACHINE_MAT_PATH = ROOT / "example_data" / "pyradplan" / "protons_Generic.mat"
DEFAULT_OUTPUT_MAT_PATH = ROOT / "example_data" / "mc_fit" / "lut_test.mat"

FIT_RADIUS_MM = 100.0
KERNEL_WIDTH_MM = 74.0  # lateral integration width (full width) matching the pencil-beam kernel support
LOW_DOSE_ABS_THRESHOLD = 1e-8
LOW_DOSE_REL_THRESHOLD = 1e-5
SIGMA_MIN_MM = 0.05
SIGMA1_MAX_MM = 30.0
SIGMA2_MAX_MM = 220.0
WEIGHT_MAX = 1.0
TAIL_FLOOR_REL = 1e-7
LOG_RESIDUAL_WEIGHT = 0.20
SIGMA2_PRIOR_STRENGTH = 0.50
WEIGHT_PRIOR_STRENGTH = 0.10
MIN_DOUBLE_FIT_POINTS = 20
PROFILE_DEPTHS_TO_PLOT_MM = (1.0, 5.0, 10.0, 15.0, 20.0)
DEPTH_PLOT_XLIM_MM = 25.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edep-path", required=True, type=Path, help="Monte Carlo edep .mhd file (3D volume)")
    parser.add_argument(
        "--kernel-width-mm",
        type=float,
        default=KERNEL_WIDTH_MM,
        help="Full lateral width (mm) over which the MC dose is integrated for the depth dose, matching the pencil-beam kernel support.",
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
        "--energy-mev",
        type=float,
        default=None,
        help="Beam energy in MeV. If omitted, parsed from the edep filename.",
    )
    parser.add_argument(
        "--source-particles",
        type=float,
        default=None,
        help="Source particle count. If omitted, parsed from the edep filename prefix.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "out" / "lut_fit",
        help="Directory where plots are written.",
    )
    parser.add_argument(
        "--output-mat-path",
        type=Path,
        default=DEFAULT_OUTPUT_MAT_PATH,
        help="Output pyRadPlan-style LUT .mat path written as a drop-in replacement.",
    )
    parser.add_argument(
        "--export-lut",
        action="store_true",
        help="Enable dense-depth fitting and write a new LUT .mat. If omitted, only evaluate MC vs LUT at the original LUT depths.",
    )
    parser.add_argument(
        "--fit-mode",
        type=str,
        default="radial",
        choices=("radial", "centerline"),
        help="Lateral profile used for sigma/weight fits: 'radial' (2D area-weighted, default) or 'centerline' (z=0 cut).",
    )
    parser.add_argument(
        "--longitudinal-smoothing-mm",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma in mm applied along depth to exported dense sigma/sigma1/sigma2 curves. Set to 0 to disable.",
    )
    return parser.parse_args()


def parse_edep_filename(path: Path) -> tuple[float | None, float | None]:
    name = path.name

    number_pattern = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"

    energy_match = re.search(number_pattern + r"MeV", name, flags=re.IGNORECASE)
    energy_mev = float(energy_match.group(1)) if energy_match else None

    particles_match = re.match(number_pattern + r"_", name)
    source_particles = float(particles_match.group(1)) if particles_match else None

    return energy_mev, source_particles


def load_reference_entry(machine_mat_path: Path, energy_mev: float) -> dict[str, np.ndarray | float]:
    data = sio.loadmat(str(machine_mat_path), squeeze_me=True, struct_as_record=False)
    machine = data["machine"]
    entries = list(machine.data)
    best = min(entries, key=lambda entry: abs(float(entry.energy) - float(energy_mev)))

    depths = np.asarray(best.depths, dtype=np.float64).ravel()
    order = np.argsort(depths)

    return {
        "energy_mev": float(best.energy),
        "depths_mm": depths[order],
        "Z": np.asarray(best.Z, dtype=np.float64).ravel()[order],
        "sigma_mm": np.asarray(best.sigma, dtype=np.float64).ravel()[order],
        "sigma1_mm": np.asarray(best.sigma1, dtype=np.float64).ravel()[order],
        "sigma2_mm": np.asarray(best.sigma2, dtype=np.float64).ravel()[order],
        "weight": np.asarray(best.weight, dtype=np.float64).ravel()[order],
        "LET": np.asarray(best.LET, dtype=np.float64).ravel()[order],
    }


def find_machine_entry_index_from_struct(entries: np.ndarray, energy_mev: float) -> int:
    n_entries = int(entries.shape[1])
    return min(
        range(n_entries),
        key=lambda idx: abs(float(np.asarray(entries["energy"][0, idx], dtype=np.float64).ravel()[0]) - float(energy_mev)),
    )


def interp_reference_curve(depths_src_mm: np.ndarray, values_src: np.ndarray, depths_dst_mm: np.ndarray) -> np.ndarray:
    depths_src_mm = np.asarray(depths_src_mm, dtype=np.float64)
    values_src = np.asarray(values_src, dtype=np.float64)
    depths_dst_mm = np.asarray(depths_dst_mm, dtype=np.float64)
    return np.interp(depths_dst_mm, depths_src_mm, values_src, left=values_src[0], right=values_src[-1])


def fill_curve_from_valid(depths_mm: np.ndarray, values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    depths_mm = np.asarray(depths_mm, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    fallback = np.asarray(fallback, dtype=np.float64)

    out = values.copy()
    valid = np.isfinite(out)
    if valid.sum() >= 2:
        missing = ~valid
        out[missing] = np.interp(depths_mm[missing], depths_mm[valid], out[valid], left=out[valid][0], right=out[valid][-1])
    elif valid.sum() == 1:
        out[:] = out[valid][0]
    else:
        out[:] = fallback

    bad = ~np.isfinite(out)
    out[bad] = fallback[bad]
    return out


def smooth_curve_along_depth(
    depths_mm: np.ndarray,
    values: np.ndarray,
    smoothing_sigma_mm: float,
) -> np.ndarray:
    depths_mm = np.asarray(depths_mm, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    sigma_mm = float(smoothing_sigma_mm)
    if sigma_mm <= 0.0 or values.size < 3:
        return values.copy()

    diffs = np.diff(depths_mm)
    positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if positive_diffs.size == 0:
        return values.copy()

    step_mm = float(np.median(positive_diffs))
    sigma_bins = sigma_mm / max(step_mm, 1e-12)
    if sigma_bins <= 0.0:
        return values.copy()
    return gaussian_filter1d(values, sigma=sigma_bins, mode="nearest")


def as_mat_column(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.ascontiguousarray(values.reshape(-1, 1), dtype=np.float64)


def write_dense_lut_mat(
    reference_machine_mat_path: Path,
    output_mat_path: Path,
    energy_mev: float,
    depths_mm: np.ndarray,
    z_curve: np.ndarray,
    sigma_mm: np.ndarray,
    sigma1_mm: np.ndarray,
    sigma2_mm: np.ndarray,
    weight: np.ndarray,
    let_curve: np.ndarray,
) -> dict[str, float]:
    base_mat_path = output_mat_path if output_mat_path.exists() else reference_machine_mat_path
    raw = sio.loadmat(str(base_mat_path), struct_as_record=True, squeeze_me=False)
    machine = raw["machine"]
    entries = machine["data"][0, 0]
    entry_idx = find_machine_entry_index_from_struct(entries, energy_mev)

    depths_col = as_mat_column(depths_mm)
    z_col = as_mat_column(z_curve)
    sigma_col = as_mat_column(sigma_mm)
    sigma1_col = as_mat_column(sigma1_mm)
    sigma2_col = as_mat_column(sigma2_mm)
    weight_col = as_mat_column(weight)
    let_col = as_mat_column(let_curve)

    entries["depths"][0, entry_idx] = depths_col
    entries["Z"][0, entry_idx] = z_col
    entries["sigma"][0, entry_idx] = sigma_col
    entries["sigma1"][0, entry_idx] = sigma1_col
    entries["sigma2"][0, entry_idx] = sigma2_col
    entries["weight"][0, entry_idx] = weight_col
    entries["LET"][0, entry_idx] = let_col

    peak_idx = int(np.nanargmax(z_curve)) if np.isfinite(z_curve).any() else 0
    entries["peakPos"][0, entry_idx] = np.array([[float(depths_mm[peak_idx])]], dtype=np.float64)

    output_mat_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(output_mat_path), {"machine": machine}, do_compression=True)
    return {
        "entry_index": float(entry_idx),
        "n_depths": float(len(depths_mm)),
        "peak_pos_mm": float(depths_mm[peak_idx]),
    }


def dose_rad_sigma_spot_mm(beam_parameters: dict[str, Any], energy_mev: float) -> float:
    entries = beam_parameters["proton"]["energy_table"]
    energies = np.asarray([float(entry["energy_mev"]) for entry in entries], dtype=np.float64)
    sigmas = np.asarray([float(entry["sigma_spot_mm"]) for entry in entries], dtype=np.float64)
    return float(np.interp(float(energy_mev), energies, sigmas, left=sigmas[0], right=sigmas[-1]))


def lateral_window_bounds(n: int, spacing_mm: float, origin_mm: float, half_width_mm: float) -> tuple[int, int]:
    """Contiguous [lo, hi) index range over a lateral axis within +/- half_width of the beam axis.

    Returns slice bounds rather than a boolean mask so callers can index with a *view*
    (no large copy) when collapsing a multi-GB volume.
    """
    coords_mm = origin_mm + np.arange(int(n), dtype=np.float64) * float(spacing_mm)
    inside = np.flatnonzero(np.abs(coords_mm) <= float(half_width_mm))
    if inside.size == 0:
        raise ValueError(f"Lateral window +/-{half_width_mm} mm selects no voxels (n={n}, spacing={spacing_mm})")
    return int(inside[0]), int(inside[-1]) + 1


def build_windowed_integral_image(
    edep_img: sitk.Image,
    source_particles: float,
    kernel_width_mm: float,
) -> dict[str, np.ndarray | float]:
    """Collapse a 3D MC edep volume into a 2D integral image and depth-dose curve.

    The new simulations are true 3D volumes ``(lateral_z, lateral_y, depth)``. The
    pencil-beam kernel only redistributes the integral depth dose over its modeled
    lateral support (``kernel_width_mm``); integrating the MC dose over the full
    lateral field therefore over-counts large-angle scatter the kernel does not
    represent, which makes the reconstructed central dose too high. We restrict the
    lateral integration to a +/- kernel_width/2 window on *both* lateral axes (the
    kernel is 2D separable). The depth-dose curve is the full double lateral integral
    over that window -- not a single central-axis line. The collapsed integral image
    ``(lateral_y, depth)`` is used for the lateral Gaussian fits, mirroring the old
    pre-integrated image convention.
    """
    edep = sitk.GetArrayFromImage(edep_img).astype(np.float64, copy=False)
    if edep.ndim != 3:
        raise ValueError(f"Expected a 3D edep image, got shape {edep.shape}")

    # SimpleITK numpy axis order is reversed vs DimSize: (lateral_z, lateral_y, depth_x).
    spacing = edep_img.GetSpacing()  # (x=depth, y, z)
    origin = edep_img.GetOrigin()  # (x=depth, y, z)
    depth_step_mm = float(spacing[0])
    depth_step_cm = depth_step_mm / 10.0
    spacing_y_mm = float(spacing[1])
    spacing_z_mm = float(spacing[2])
    origin_y_mm = float(origin[1])
    origin_z_mm = float(origin[2])

    half_width_mm = 0.5 * float(kernel_width_mm)
    n_z, n_y, _n_depth = edep.shape
    z_lo, z_hi = lateral_window_bounds(n_z, spacing_z_mm, origin_z_mm, half_width_mm)
    y_lo, y_hi = lateral_window_bounds(n_y, spacing_y_mm, origin_y_mm, half_width_mm)

    # Collapse one lateral axis (z) within the kernel window -> integral image (y, depth).
    # Contiguous slice is a view: no multi-GB copy of the masked subvolume.
    integral_img = edep[z_lo:z_hi, :, :].sum(axis=0)
    # Integral depth dose: integrate the remaining lateral axis (y) within the same window.
    z_per_depth_bin = integral_img[y_lo:y_hi, :].sum(axis=0) / float(source_particles)
    z_per_cm = z_per_depth_bin / depth_step_cm
    depth_mm = np.arange(integral_img.shape[1], dtype=np.float64) * depth_step_mm
    y_mm = origin_y_mm + np.arange(n_y, dtype=np.float64) * spacing_y_mm

    # Centerline image (y, depth): thin z-band through the beam axis (z=0). The engine
    # double-Gaussian weight is a 2D *area* mixture weight applied to normalized 2D
    # Gaussians (ion_pencil_beam_model.evaluate_lateral_cell_weights), whose 1D form is
    # the centerline cut (amplitude ~ 1/sigma^2). The z-integrated profile instead scales
    # ~1/sigma and biases the recovered halo weight, so weight fits must use this cut.
    zc_idx = int(round(-origin_z_mm / spacing_z_mm))
    zb_lo, zb_hi = max(0, zc_idx - 3), min(n_z, zc_idx + 4)  # ~+/-0.6 mm band for SNR
    centerline_img = edep[zb_lo:zb_hi, :, :].mean(axis=0)

    # Radial profile image (r_bin, depth): the true 2D lateral profile of the radially
    # symmetric kernel. g(r) has the same functional form as the centerline cut, but
    # binning every lateral voxel by radius (area-weighted: more samples at large r)
    # constrains the broad halo (sigma2/weight) far better than a single line. Fed to the
    # same centerline-form fitters.
    zc_mm = -origin_z_mm  # beam axis lateral position (origin chosen so axis ~ 0)
    yc_mm = -origin_y_mm
    z_off = (np.arange(n_z, dtype=np.float64) * spacing_z_mm - zc_mm)
    y_off = (np.arange(n_y, dtype=np.float64) * spacing_y_mm - yc_mm)
    r_grid = np.sqrt(z_off[:, None] ** 2 + y_off[None, :] ** 2)  # (n_z, n_y)
    # 1 mm bins: the beam sits at a half-voxel so a sub-mm innermost bin would be empty
    # (spurious central dip); 1 mm guarantees every bin -- including r~0 -- has voxels.
    r_step = 1.0
    n_rbins = int(FIT_RADIUS_MM / r_step) + 2
    rb = np.clip(np.round(r_grid / r_step).astype(np.intp), 0, n_rbins - 1).ravel()
    counts = np.bincount(rb, minlength=n_rbins).astype(np.float64)
    edep_2d = edep.reshape(n_z * n_y, -1)  # (n_z*n_y, depth) view
    radial_sum = np.zeros((n_rbins, edep_2d.shape[1]), dtype=np.float64)
    for d in range(edep_2d.shape[1]):
        radial_sum[:, d] = np.bincount(rb, weights=edep_2d[:, d], minlength=n_rbins)
    radial_img = radial_sum / np.maximum(counts[:, None], 1.0)
    r_mm = np.arange(n_rbins, dtype=np.float64) * r_step

    return {
        "integral_img": integral_img,  # (lateral_y, depth)
        "centerline_img": centerline_img,  # (lateral_y, depth), z=0 cut
        "radial_img": radial_img,  # (r_bin, depth), area-weighted radial profile
        "r_mm": r_mm,
        "y_mm": y_mm,
        "depth_mm": depth_mm,
        "Z_est": z_per_cm,
        "depth_step_mm": depth_step_mm,
        "kernel_width_mm": float(kernel_width_mm),
        "n_z_window": int(z_hi - z_lo),
        "n_y_window": int(y_hi - y_lo),
    }


def gaussian_profile_1d(x_mm: np.ndarray, amplitude: float, sigma_mm: float) -> np.ndarray:
    x_mm = np.asarray(x_mm, dtype=np.float64)
    sigma_mm = max(float(sigma_mm), 1e-9)
    return float(amplitude) * np.exp(-0.5 * (x_mm / sigma_mm) ** 2)


def double_gaussian_profile_1d_amplitude_weight(
    x_mm: np.ndarray,
    amplitude: float,
    sigma1_mm: float,
    sigma2_mm: float,
    halo_weight: float,
) -> np.ndarray:
    x_mm = np.asarray(x_mm, dtype=np.float64)
    sigma1_mm = max(float(sigma1_mm), 1e-9)
    sigma2_mm = max(float(sigma2_mm), 1e-9)
    halo_weight = float(np.clip(halo_weight, 0.0, 1.0))

    # Match pyRadPlan Hong double-Gaussian semantics: the mixture weights are
    # applied to normalized 2D Gaussian kernels. Along a 1D centerline this
    # means the wider component has a lower central amplitude proportional to
    # 1 / sigma^2, not a direct peak-amplitude blend.
    core = ((1.0 - halo_weight) / (sigma1_mm**2)) * np.exp(-0.5 * (x_mm / sigma1_mm) ** 2)
    halo = (halo_weight / (sigma2_mm**2)) * np.exp(-0.5 * (x_mm / sigma2_mm) ** 2)
    return float(amplitude) * (core + halo)


def profile_eval_mask(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    radius_limit_mm: float | None = None,
    tail_floor_rel: float = 0.0,
) -> np.ndarray:
    x_mm = np.asarray(x_mm, dtype=np.float64)
    y_mm = np.asarray(y_mm, dtype=np.float64)

    peak = float(np.nanmax(y_mm))
    if not np.isfinite(peak) or peak <= 0.0:
        return np.zeros(y_mm.shape, dtype=bool)

    keep = np.isfinite(x_mm) & np.isfinite(y_mm)
    keep &= y_mm >= 0.0
    if radius_limit_mm is not None:
        keep &= np.abs(x_mm) <= float(radius_limit_mm)
    if tail_floor_rel > 0.0:
        keep &= y_mm > peak * float(tail_floor_rel)

    if keep.sum() < 3:
        keep = np.isfinite(x_mm) & np.isfinite(y_mm)
        keep &= y_mm >= 0.0
        if radius_limit_mm is not None:
            keep &= np.abs(x_mm) <= float(radius_limit_mm)

    return keep


def normalized_profile_shape_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float] | None:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    if keep.sum() < 3:
        return None

    y_true_keep = np.clip(y_true[keep], 0.0, None)
    y_pred_keep = np.clip(y_pred[keep], 0.0, None)

    peak_true = float(np.nanmax(y_true_keep))
    peak_pred = float(np.nanmax(y_pred_keep))
    if not np.isfinite(peak_true) or peak_true <= 0.0:
        return None
    if not np.isfinite(peak_pred) or peak_pred <= 0.0:
        return None

    y_true_norm = y_true_keep / peak_true
    y_pred_norm = y_pred_keep / peak_pred
    eps = 1e-12

    rmse_linear = float(np.sqrt(np.mean((y_pred_norm - y_true_norm) ** 2)))
    rmse_log = float(np.sqrt(np.mean((np.log(y_pred_norm + eps) - np.log(y_true_norm + eps)) ** 2)))
    return {
        "rmse_linear": rmse_linear,
        "rmse_log": rmse_log,
        "combined": float(rmse_linear + LOG_RESIDUAL_WEIGHT * rmse_log),
    }


def single_gaussian_shape_error(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    sigma_mm: float,
    radius_limit_mm: float | None = None,
    tail_floor_rel: float = 0.0,
) -> float:
    if not np.isfinite(sigma_mm) or float(sigma_mm) <= 0.0:
        return float("nan")

    keep = profile_eval_mask(x_mm, y_mm, radius_limit_mm=radius_limit_mm, tail_floor_rel=tail_floor_rel)
    if keep.sum() < 3:
        return float("nan")

    metrics = normalized_profile_shape_metrics(
        y_true=y_mm[keep],
        y_pred=gaussian_profile_1d(x_mm[keep], 1.0, sigma_mm),
    )
    return float(metrics["combined"]) if metrics is not None else float("nan")


def double_gaussian_shape_error(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    sigma1_mm: float,
    sigma2_mm: float,
    halo_weight: float,
    radius_limit_mm: float | None = None,
    tail_floor_rel: float = 0.0,
) -> float:
    if not np.isfinite(sigma1_mm) or float(sigma1_mm) <= 0.0:
        return float("nan")
    if not np.isfinite(sigma2_mm) or float(sigma2_mm) <= 0.0:
        return float("nan")
    if not np.isfinite(halo_weight):
        return float("nan")

    keep = profile_eval_mask(x_mm, y_mm, radius_limit_mm=radius_limit_mm, tail_floor_rel=tail_floor_rel)
    if keep.sum() < 3:
        return float("nan")

    metrics = normalized_profile_shape_metrics(
        y_true=y_mm[keep],
        y_pred=double_gaussian_profile_1d_amplitude_weight(x_mm[keep], 1.0, sigma1_mm, sigma2_mm, halo_weight),
    )
    return float(metrics["combined"]) if metrics is not None else float("nan")


def scan_single_sigma_error_curve(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    sigma_reference_mm: float | None = None,
    sigma_fit_mm: float | None = None,
    n_samples: int = 300,
    radius_limit_mm: float | None = None,
    tail_floor_rel: float = 0.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    candidates = []
    for value in (sigma_reference_mm, sigma_fit_mm):
        if value is None:
            continue
        if np.isfinite(value) and float(value) > 0.0:
            candidates.append(float(value))

    if candidates:
        sigma_min = max(SIGMA_MIN_MM, 0.5 * min(candidates))
        sigma_max = min(SIGMA2_MAX_MM, 2.0 * max(candidates))
    else:
        sigma_min = SIGMA_MIN_MM
        sigma_max = min(SIGMA2_MAX_MM, 50.0)

    if sigma_max <= sigma_min:
        sigma_max = min(SIGMA2_MAX_MM, sigma_min + 5.0)

    sigma_grid = np.linspace(sigma_min, sigma_max, int(max(n_samples, 32)), dtype=np.float64)
    errors = np.asarray(
        [
            single_gaussian_shape_error(
                x_mm,
                y_mm,
                sigma,
                radius_limit_mm=radius_limit_mm,
                tail_floor_rel=tail_floor_rel,
            )
            for sigma in sigma_grid
        ],
        dtype=np.float64,
    )
    if not np.isfinite(errors).any():
        return None
    return sigma_grid, errors


def quadrature_subtract_sigma0(sigma_total_mm: float, sigma0_mm: float) -> float:
    return float(np.sqrt(max(float(sigma_total_mm) ** 2 - float(sigma0_mm) ** 2, 0.0)))


def fit_single_gaussian(x_fit: np.ndarray, y_fit: np.ndarray, sigma_guess: float) -> tuple[float, float]:
    peak = float(np.nanmax(y_fit))
    popt, _ = curve_fit(
        gaussian_profile_1d,
        x_fit,
        y_fit,
        p0=[peak, max(float(sigma_guess), 0.5)],
        bounds=([0.0, SIGMA_MIN_MM], [np.inf, SIGMA2_MAX_MM]),
        maxfev=20000,
    )
    return float(popt[0]), float(popt[1])


def fit_single_gaussian_logspace(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    sigma_guess: float,
) -> tuple[float, float]:
    x_fit = np.asarray(x_fit, dtype=np.float64)
    y_fit = np.asarray(y_fit, dtype=np.float64)

    keep = np.isfinite(x_fit) & np.isfinite(y_fit)
    keep &= y_fit > 0.0
    if keep.sum() < 3:
        raise RuntimeError("Need at least 3 positive points for log-space Gaussian fit")

    x = x_fit[keep]
    y = y_fit[keep]
    x_sq = x * x
    log_y = np.log(np.clip(y, 1e-30, None))

    design = np.column_stack([np.ones_like(x_sq), x_sq])
    coeffs, _, _, _ = np.linalg.lstsq(design, log_y, rcond=None)
    intercept, slope = coeffs
    if not np.isfinite(intercept) or not np.isfinite(slope) or slope >= 0.0:
        raise RuntimeError("Invalid log-space Gaussian fit")

    amplitude = float(np.exp(intercept))
    sigma = float(np.sqrt(-1.0 / (2.0 * slope)))
    amplitude = max(amplitude, 0.0)
    sigma = float(np.clip(sigma, SIGMA_MIN_MM, SIGMA2_MAX_MM))

    # Refit amplitude in linear space for the chosen sigma.
    unit = gaussian_profile_1d(x_fit, 1.0, sigma)
    denom = float(np.dot(unit, unit))
    if np.isfinite(denom) and denom > 0.0:
        amplitude = max(float(np.dot(y_fit, unit) / denom), 0.0)

    return amplitude, sigma


def fit_double_gaussian_safe(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    sigma_single_total: float,
    sigma1_guess: float,
    sigma2_prior: float,
    weight_guess: float,
) -> dict[str, float] | None:
    """Stage a double-Gaussian fit: core first, halo second.

    This is intentionally much cheaper and more stable than the previous
    fully-free 4-parameter least-squares fit.
    """
    x_fit = np.asarray(x_fit, dtype=np.float64)
    y_fit = np.asarray(y_fit, dtype=np.float64)

    peak = float(np.nanmax(y_fit))
    if not np.isfinite(peak) or peak <= 0.0:
        return None

    keep = np.isfinite(x_fit) & np.isfinite(y_fit)
    keep &= np.abs(x_fit) <= FIT_RADIUS_MM
    keep &= y_fit >= 0.0
    if keep.sum() < MIN_DOUBLE_FIT_POINTS:
        return None

    x = x_fit[keep]
    y = y_fit[keep]

    sigma_single_total = float(np.clip(sigma_single_total, SIGMA_MIN_MM, SIGMA2_MAX_MM))
    if not np.isfinite(sigma1_guess):
        sigma1_guess = max(0.8 * sigma_single_total, SIGMA_MIN_MM)
    sigma1_guess = float(np.clip(sigma1_guess, SIGMA_MIN_MM, SIGMA1_MAX_MM))

    if not np.isfinite(sigma2_prior):
        sigma2_prior = max(20.0, 2.0 * sigma_single_total, sigma1_guess + 10.0)
    sigma2_prior = float(np.clip(sigma2_prior, sigma1_guess + SIGMA_MIN_MM, SIGMA2_MAX_MM))

    if not np.isfinite(weight_guess):
        weight_guess = 1e-4
    weight_guess = float(np.clip(weight_guess, 0.0, WEIGHT_MAX))

    core_radius_mm = 100#max(6.0, min(18.0, 2 * sigma_single_total))
    core_mask = np.abs(x) <= core_radius_mm
    if core_mask.sum() < 9:
        core_mask = np.abs(x) <= max(10.0, core_radius_mm)
    if core_mask.sum() < 9:
        return None

    try:
        amp1, sigma1 = fit_single_gaussian(
            x_fit=x[core_mask],
            y_fit=y[core_mask],
            sigma_guess=min(sigma1_guess, sigma_single_total),
        )
    except Exception:
        return None

    sigma1 = float(np.clip(sigma1, SIGMA_MIN_MM, SIGMA1_MAX_MM))
    core_full = gaussian_profile_1d(x, amp1, sigma1)
    residual = np.clip(y - core_full, 0.0, None)

    halo_min_radius_mm = max(8.0, 1.5 * sigma1)
    halo_mask = np.abs(x) >= halo_min_radius_mm

    try:
        amp2, sigma2 = fit_single_gaussian_logspace(
            x_fit=x[halo_mask],
            y_fit=residual[halo_mask],
            sigma_guess=max(sigma2_prior, sigma1 + 1.0),
        )
    except Exception:
        return None

    sigma2 = float(np.clip(max(sigma2, sigma1 + SIGMA_MIN_MM), sigma1 + SIGMA_MIN_MM, SIGMA2_MAX_MM))
    amp1 = max(float(amp1), 0.0)
    amp2 = max(float(amp2), 0.0)

    mix_norm = amp1 * sigma1**2 + amp2 * sigma2**2
    if not np.isfinite(mix_norm) or mix_norm <= 0.0:
        return None

    if amp2 <= 0.0:
        weight = weight_guess
    else:
        weight = float(np.clip((amp2 * sigma2**2) / mix_norm, 0.0, WEIGHT_MAX))
    amplitude = float(mix_norm)

    unit_profile = double_gaussian_profile_1d_amplitude_weight(x, 1.0, sigma1, sigma2, weight)
    denom = float(np.dot(unit_profile, unit_profile))
    if np.isfinite(denom) and denom > 0.0:
        amplitude = max(float(np.dot(y, unit_profile) / denom), 0.0)

    return {
        "amplitude": float(amplitude),
        "sigma1_total_mm": float(sigma1),
        "sigma2_total_mm": float(sigma2),
        "weight": float(weight),
        "cost": float("nan"),
    }


def fallback_double_gaussian_from_single(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    sigma_single_total: float,
    sigma0_mm: float,
    ref_sigma_total: float,
    ref_sigma1_total: float,
    ref_sigma2_total: float,
    ref_weight: float,
) -> dict[str, float] | None:
    x_fit = np.asarray(x_fit, dtype=np.float64)
    y_fit = np.asarray(y_fit, dtype=np.float64)

    peak = float(np.nanmax(y_fit))
    if not np.isfinite(peak) or peak <= 0.0:
        return None

    sigma_single_lut = quadrature_subtract_sigma0(sigma_single_total, sigma0_mm)
    ref_sigma_lut = quadrature_subtract_sigma0(ref_sigma_total, sigma0_mm)
    ref_sigma1_lut = quadrature_subtract_sigma0(ref_sigma1_total, sigma0_mm)
    ref_sigma2_lut = quadrature_subtract_sigma0(ref_sigma2_total, sigma0_mm)

    scale = 1.0
    if np.isfinite(ref_sigma_lut) and ref_sigma_lut > SIGMA_MIN_MM:
        scale = float(np.clip(sigma_single_lut / ref_sigma_lut, 0.5, 2.0))

    sigma1_total = float(np.sqrt(sigma0_mm**2 + (ref_sigma1_lut * scale) ** 2))
    sigma2_total = float(np.sqrt(sigma0_mm**2 + (ref_sigma2_lut * scale) ** 2))
    sigma2_total = max(sigma2_total, sigma1_total + SIGMA_MIN_MM)
    weight = float(np.clip(ref_weight, 0.0, WEIGHT_MAX))

    unit_profile = double_gaussian_profile_1d_amplitude_weight(
        x_fit,
        1.0,
        sigma1_total,
        sigma2_total,
        weight,
    )
    denom = float(np.dot(unit_profile, unit_profile))
    if np.isfinite(denom) and denom > 0.0:
        amplitude = max(float(np.dot(y_fit, unit_profile) / denom), 0.0)
    else:
        amplitude = peak

    return {
        "amplitude": float(amplitude),
        "sigma1_total_mm": sigma1_total,
        "sigma2_total_mm": sigma2_total,
        "weight": weight,
        "cost": float("nan"),
    }


def main() -> None:
    args = parse_args()

    beam_parameters = json.loads(args.beam_params_path.read_text())
    inferred_energy_mev, inferred_source_particles = parse_edep_filename(args.edep_path)

    energy_value = args.energy_mev if args.energy_mev is not None else inferred_energy_mev
    if energy_value is None:
        raise ValueError("Could not determine energy. Pass --energy-mev or include '<energy>MeV' in the edep filename.")
    energy_mev = float(energy_value)

    source_particle_value = args.source_particles if args.source_particles is not None else inferred_source_particles
    if source_particle_value is None:
        raise ValueError(
            "Could not determine source particles. Pass --source-particles or use the notebook filename convention '<particles>_<energy>MeV_...'."
        )
    source_particles = float(source_particle_value)

    ref = load_reference_entry(args.machine_mat_path, energy_mev)
    sigma0_mm = dose_rad_sigma_spot_mm(beam_parameters, energy_mev)

    edep_img = sitk.ReadImage(str(args.edep_path))
    z_from_edep = build_windowed_integral_image(edep_img, source_particles, args.kernel_width_mm)
    integral_img = np.asarray(z_from_edep["integral_img"], dtype=np.float64)  # (lateral_y, depth)
    # Lateral profile fits (sigma1/sigma2/weight). 'radial' (default) uses the true 2D
    # area-weighted radial profile -> best halo constraint; 'centerline' the z=0 cut.
    # Both share the centerline-form double-Gaussian model that matches the engine kernel.
    if args.fit_mode == "radial":
        fit_img = np.asarray(z_from_edep["radial_img"], dtype=np.float64)  # (r_bin, depth)
        fit_axis_mm = np.asarray(z_from_edep["r_mm"], dtype=np.float64)
    else:
        fit_img = np.asarray(z_from_edep["centerline_img"], dtype=np.float64)  # (lateral_y, depth)
        fit_axis_mm = np.asarray(z_from_edep["y_mm"], dtype=np.float64)
    n_depth_bins = integral_img.shape[1]

    ref_depths_mm = np.asarray(ref["depths_mm"], dtype=np.float64)
    ref_Z = np.asarray(ref["Z"], dtype=np.float64)
    ref_sigma = np.asarray(ref["sigma_mm"], dtype=np.float64)
    ref_sigma1 = np.asarray(ref["sigma1_mm"], dtype=np.float64)
    ref_sigma2 = np.asarray(ref["sigma2_mm"], dtype=np.float64)
    ref_weight = np.asarray(ref["weight"], dtype=np.float64)
    ref_let = np.asarray(ref["LET"], dtype=np.float64)

    ref_sigma_total = np.sqrt(sigma0_mm**2 + ref_sigma**2)
    ref_sigma1_total = np.sqrt(sigma0_mm**2 + ref_sigma1**2)
    ref_sigma2_total = np.sqrt(sigma0_mm**2 + ref_sigma2**2)

    depth_indices = np.clip(
        np.round(ref_depths_mm / float(z_from_edep["depth_step_mm"])).astype(int),
        0,
        n_depth_bins - 1,
    )

    x_profile = fit_axis_mm
    fit_mask = np.abs(fit_axis_mm) <= FIT_RADIUS_MM
    x_fit = fit_axis_mm[fit_mask]
    profile_plot_xlim_mm = float(np.nanmax(np.abs(x_profile)))

    n_depths = len(ref_depths_mm)
    single_amplitude = np.full(n_depths, np.nan, dtype=np.float64)
    single_sigma_total = np.full(n_depths, np.nan, dtype=np.float64)
    single_sigma_deconv = np.full(n_depths, np.nan, dtype=np.float64)
    double_amplitude = np.full(n_depths, np.nan, dtype=np.float64)
    double_sigma1_total = np.full(n_depths, np.nan, dtype=np.float64)
    double_sigma2_total = np.full(n_depths, np.nan, dtype=np.float64)
    double_sigma1_deconv = np.full(n_depths, np.nan, dtype=np.float64)
    double_sigma2_deconv = np.full(n_depths, np.nan, dtype=np.float64)
    double_weight_fit = np.full(n_depths, np.nan, dtype=np.float64)
    double_cost = np.full(n_depths, np.nan, dtype=np.float64)
    single_ref_shape_error = np.full(n_depths, np.nan, dtype=np.float64)
    single_fit_shape_error = np.full(n_depths, np.nan, dtype=np.float64)
    double_ref_shape_error = np.full(n_depths, np.nan, dtype=np.float64)
    double_fit_shape_error = np.full(n_depths, np.nan, dtype=np.float64)
    fit_ok_single = np.zeros(n_depths, dtype=bool)
    fit_ok_double = np.zeros(n_depths, dtype=bool)
    double_repr_ok = np.zeros(n_depths, dtype=bool)
    double_fallback_used = np.zeros(n_depths, dtype=bool)

    global_peak = float(np.nanmax(fit_img))
    low_dose_threshold = max(global_peak * LOW_DOSE_REL_THRESHOLD, LOW_DOSE_ABS_THRESHOLD)

    for i, depth_idx in enumerate(depth_indices):
        profile = fit_img[:, int(depth_idx)].astype(np.float64, copy=False)
        y_profile = profile
        y_fit = profile[fit_mask]
        peak = float(np.nanmax(y_fit))
        if not np.isfinite(peak) or peak <= low_dose_threshold:
            continue

        single_ref_shape_error[i] = single_gaussian_shape_error(x_profile, y_profile, ref_sigma_total[i])
        double_ref_shape_error[i] = double_gaussian_shape_error(
            x_profile,
            y_profile,
            ref_sigma1_total[i],
            ref_sigma2_total[i],
            ref_weight[i],
        )

        try:
            amplitude, sigma = fit_single_gaussian(x_fit=x_fit, y_fit=y_fit, sigma_guess=ref_sigma_total[i])
            single_amplitude[i] = amplitude
            single_sigma_total[i] = sigma
            single_sigma_deconv[i] = quadrature_subtract_sigma0(sigma, sigma0_mm)
            fit_ok_single[i] = True
            single_fit_shape_error[i] = single_gaussian_shape_error(x_profile, y_profile, sigma)

        except Exception:
            continue

        result = None
        try:
            result = fit_double_gaussian_safe(
                x_fit=x_fit,
                y_fit=y_fit,
                sigma_single_total=single_sigma_total[i],
                sigma1_guess=ref_sigma1_total[i],
                sigma2_prior=ref_sigma2_total[i],
                weight_guess=ref_weight[i],
            )
        except Exception:
            result = None

        if result is None:
            result = fallback_double_gaussian_from_single(
                x_fit=x_fit,
                y_fit=y_fit,
                sigma_single_total=single_sigma_total[i],
                sigma0_mm=sigma0_mm,
                ref_sigma_total=ref_sigma_total[i],
                ref_sigma1_total=ref_sigma1_total[i],
                ref_sigma2_total=ref_sigma2_total[i],
                ref_weight=ref_weight[i],
            )
            double_fallback_used[i] = result is not None
        else:
            fit_ok_double[i] = True

        if result is not None:
            double_amplitude[i] = result["amplitude"]
            double_sigma1_total[i] = result["sigma1_total_mm"]
            double_sigma2_total[i] = result["sigma2_total_mm"]
            double_weight_fit[i] = result["weight"]
            double_cost[i] = result["cost"]
            double_sigma1_deconv[i] = quadrature_subtract_sigma0(double_sigma1_total[i], sigma0_mm)
            double_sigma2_deconv[i] = quadrature_subtract_sigma0(double_sigma2_total[i], sigma0_mm)
            double_fit_shape_error[i] = double_gaussian_shape_error(
                x_profile,
                y_profile,
                double_sigma1_total[i],
                double_sigma2_total[i],
                double_weight_fit[i],
            )
            double_repr_ok[i] = True

    mc_depth_mm_all = np.asarray(z_from_edep["depth_mm"], dtype=np.float64)
    mc_z_all = np.asarray(z_from_edep["Z_est"], dtype=np.float64)
    depth_plot_xlim_mm = max(DEPTH_PLOT_XLIM_MM, float(np.nanmax(ref_depths_mm)))

    dense_depth_keep = None
    dense_depth_indices = None
    dense_depths_mm = None
    dense_z_curve = None
    dense_ref_sigma = None
    dense_ref_sigma1 = None
    dense_ref_sigma2 = None
    dense_ref_weight = None
    dense_ref_let = None
    dense_single_sigma_total_raw = None
    dense_double_sigma1_total_raw = None
    dense_double_sigma2_total_raw = None
    dense_double_weight_raw = None
    dense_fit_ok_single = None
    dense_fit_ok_double = None
    dense_sigma1_raw = None
    dense_sigma2_raw = None
    dense_single_sigma_lut = None
    dense_sigma1_lut = None
    dense_sigma2_lut = None
    dense_weight_lut = None
    dense_single_sigma_lut_unsmoothed = None
    dense_sigma1_lut_unsmoothed = None
    dense_sigma2_lut_unsmoothed = None
    dense_single_sigma_total_runtime = None
    dense_lut_write_info = None
    n_dense_depths = 0

    if args.export_lut:
        dense_depth_keep = mc_depth_mm_all <= (float(np.nanmax(ref_depths_mm)) + 0.5 * float(z_from_edep["depth_step_mm"]))
        dense_depth_indices = np.flatnonzero(dense_depth_keep)
        dense_depths_mm = mc_depth_mm_all[dense_depth_keep]
        dense_z_curve = np.clip(mc_z_all[dense_depth_keep], 0.0, None)

        dense_ref_sigma = interp_reference_curve(ref_depths_mm, ref_sigma, dense_depths_mm)
        dense_ref_sigma1 = interp_reference_curve(ref_depths_mm, ref_sigma1, dense_depths_mm)
        dense_ref_sigma2 = interp_reference_curve(ref_depths_mm, ref_sigma2, dense_depths_mm)
        dense_ref_weight = interp_reference_curve(ref_depths_mm, ref_weight, dense_depths_mm)
        dense_ref_let = interp_reference_curve(ref_depths_mm, ref_let, dense_depths_mm)
        dense_ref_sigma_total = np.sqrt(sigma0_mm**2 + dense_ref_sigma**2)
        dense_ref_sigma1_total = np.sqrt(sigma0_mm**2 + dense_ref_sigma1**2)
        dense_ref_sigma2_total = np.sqrt(sigma0_mm**2 + dense_ref_sigma2**2)

        n_dense_depths = len(dense_depths_mm)
        dense_single_sigma_total_raw = np.full(n_dense_depths, np.nan, dtype=np.float64)
        dense_double_sigma1_total_raw = np.full(n_dense_depths, np.nan, dtype=np.float64)
        dense_double_sigma2_total_raw = np.full(n_dense_depths, np.nan, dtype=np.float64)
        dense_double_weight_raw = np.full(n_dense_depths, np.nan, dtype=np.float64)
        dense_fit_ok_single = np.zeros(n_dense_depths, dtype=bool)
        dense_fit_ok_double = np.zeros(n_dense_depths, dtype=bool)

        prev_single_sigma_total = float("nan")
        prev_double_sigma1_total = float("nan")
        prev_double_weight = float("nan")

        for j, depth_idx in enumerate(dense_depth_indices):
            profile = fit_img[:, int(depth_idx)].astype(np.float64, copy=False)
            y_fit_dense = profile[fit_mask]
            peak = float(np.nanmax(y_fit_dense))
            if not np.isfinite(peak) or peak <= low_dose_threshold:
                continue

            sigma_single_guess = prev_single_sigma_total if np.isfinite(prev_single_sigma_total) else dense_ref_sigma_total[j]
            try:
                _, sigma_single_total = fit_single_gaussian(x_fit=x_fit, y_fit=y_fit_dense, sigma_guess=sigma_single_guess)
                dense_single_sigma_total_raw[j] = sigma_single_total
                dense_fit_ok_single[j] = True
                prev_single_sigma_total = sigma_single_total
            except Exception:
                pass

            sigma_for_double = dense_single_sigma_total_raw[j] if np.isfinite(dense_single_sigma_total_raw[j]) else sigma_single_guess
            sigma1_guess = prev_double_sigma1_total if np.isfinite(prev_double_sigma1_total) else dense_ref_sigma1_total[j]
            weight_guess = prev_double_weight if np.isfinite(prev_double_weight) else dense_ref_weight[j]
            try:
                result = fit_double_gaussian_safe(
                    x_fit=x_fit,
                    y_fit=y_fit_dense,
                    sigma_single_total=sigma_for_double,
                    sigma1_guess=sigma1_guess,
                    sigma2_prior=dense_ref_sigma2_total[j],
                    weight_guess=weight_guess,
                )
                if result is not None:
                    dense_double_sigma1_total_raw[j] = result["sigma1_total_mm"]
                    dense_double_sigma2_total_raw[j] = result["sigma2_total_mm"]
                    dense_double_weight_raw[j] = result["weight"]
                    dense_fit_ok_double[j] = True
                    prev_double_sigma1_total = result["sigma1_total_mm"]
                    prev_double_weight = result["weight"]
            except Exception:
                pass

        dense_single_sigma_raw = np.sqrt(np.maximum(dense_single_sigma_total_raw**2 - sigma0_mm**2, 0.0))
        dense_sigma1_raw = np.sqrt(np.maximum(dense_double_sigma1_total_raw**2 - sigma0_mm**2, 0.0))
        dense_sigma2_raw = np.sqrt(np.maximum(dense_double_sigma2_total_raw**2 - sigma0_mm**2, 0.0))

        dense_single_sigma_lut_unsmoothed = np.clip(fill_curve_from_valid(dense_depths_mm, dense_single_sigma_raw, dense_ref_sigma), 0.0, None)

        # LUT sigma/sigma1/sigma2 are transport terms combined in quadrature with the
        # entrance sigma at dose-calc time. Use the MC-fitted double-Gaussian halo
        # (sigma1/sigma2/weight) directly -- now that the fit reads the centerline cut,
        # the recovered 2D mixture weight matches the engine kernel. Fall back to the
        # reference curves scaled by the MC single-sigma ratio only where the fit failed.
        halo_scale = np.ones_like(dense_single_sigma_lut_unsmoothed)
        scale_mask = dense_ref_sigma > SIGMA_MIN_MM
        halo_scale[scale_mask] = dense_single_sigma_lut_unsmoothed[scale_mask] / dense_ref_sigma[scale_mask]
        halo_scale = np.clip(halo_scale, 0.5, 2.0)
        ref_sigma1_scaled = np.clip(dense_ref_sigma1 * halo_scale, 0.0, None)
        ref_sigma2_scaled = np.clip(dense_ref_sigma2 * halo_scale, 0.0, None)
        s1_fit = np.where(dense_fit_ok_double, dense_sigma1_raw, np.nan)
        s2_fit = np.where(dense_fit_ok_double, dense_sigma2_raw, np.nan)
        w_fit = np.where(dense_fit_ok_double, dense_double_weight_raw, np.nan)
        dense_sigma1_lut_unsmoothed = np.clip(fill_curve_from_valid(dense_depths_mm, s1_fit, ref_sigma1_scaled), 0.0, None)
        dense_sigma2_lut_unsmoothed = np.clip(fill_curve_from_valid(dense_depths_mm, s2_fit, ref_sigma2_scaled), 0.0, None)
        dense_weight_lut = np.clip(fill_curve_from_valid(dense_depths_mm, w_fit, dense_ref_weight), 0.0, WEIGHT_MAX)
        dense_sigma2_lut_unsmoothed = np.maximum(dense_sigma2_lut_unsmoothed, dense_sigma1_lut_unsmoothed + SIGMA_MIN_MM)

        dense_single_sigma_lut = np.clip(
            smooth_curve_along_depth(dense_depths_mm, dense_single_sigma_lut_unsmoothed, args.longitudinal_smoothing_mm),
            0.0,
            None,
        )
        dense_sigma1_lut = np.clip(
            smooth_curve_along_depth(dense_depths_mm, dense_sigma1_lut_unsmoothed, args.longitudinal_smoothing_mm),
            0.0,
            None,
        )
        dense_sigma2_lut = np.clip(
            smooth_curve_along_depth(dense_depths_mm, dense_sigma2_lut_unsmoothed, args.longitudinal_smoothing_mm),
            0.0,
            None,
        )
        dense_sigma2_lut = np.maximum(dense_sigma2_lut, dense_sigma1_lut + SIGMA_MIN_MM)

        dense_single_sigma_total_runtime = np.sqrt(sigma0_mm**2 + dense_single_sigma_lut**2)

        dense_lut_write_info = write_dense_lut_mat(
            reference_machine_mat_path=args.machine_mat_path,
            output_mat_path=args.output_mat_path,
            energy_mev=energy_mev,
            depths_mm=dense_depths_mm,
            z_curve=dense_z_curve,
            sigma_mm=dense_single_sigma_lut,
            sigma1_mm=dense_sigma1_lut,
            sigma2_mm=dense_sigma2_lut,
            weight=dense_weight_lut,
            let_curve=dense_ref_let,
        )
        depth_plot_xlim_mm = max(DEPTH_PLOT_XLIM_MM, float(np.nanmax(dense_depths_mm)))

    valid_depth_mask = ref_Z > 0.05 * np.nanmax(ref_Z)
    valid_single = fit_ok_single & valid_depth_mask
    valid_double = double_repr_ok & valid_depth_mask
    valid_single_quality = valid_depth_mask & np.isfinite(single_ref_shape_error) & np.isfinite(single_fit_shape_error)
    valid_double_quality = valid_depth_mask & np.isfinite(double_ref_shape_error) & np.isfinite(double_fit_shape_error)

    print(f"Requested energy            = {energy_mev:.6f} MeV")
    print(f"Matched reference energy    = {float(ref['energy_mev']):.6f} MeV")
    print(f"sigma0 from beam parameters = {sigma0_mm:.6f} mm")
    print(f"source particles            = {source_particles:.6g}")
    print(f"integral image shape        = {tuple(integral_img.shape)} (lateral_y, depth)")
    print(f"kernel integration width    = {float(z_from_edep['kernel_width_mm']):.3f} mm full width")
    print(f"lateral window voxels        = z:{int(z_from_edep['n_z_window'])}  y:{int(z_from_edep['n_y_window'])}")
    print(f"depth step from edep        = {float(z_from_edep['depth_step_mm']):.6f} mm")
    print()
    print(f"single-gaussian fits        = {fit_ok_single.sum()} / {n_depths}")
    print(f"double-gaussian staged      = {fit_ok_double.sum()} / {n_depths}")
    print(f"double-gaussian represented = {double_repr_ok.sum()} / {n_depths}")
    print(f"double fallback used        = {double_fallback_used.sum()} / {n_depths}")
    if args.export_lut:
        print(f"dense single fits           = {dense_fit_ok_single.sum()} / {n_dense_depths}")
        print(f"dense double fits           = {dense_fit_ok_double.sum()} / {n_dense_depths}")
        print(f"dense LUT depth bins        = {int(dense_lut_write_info['n_depths'])}")
        print("dense halo params           = reference halo scaled by MC single-sigma ratio")
        print(f"longitudinal smoothing      = {args.longitudinal_smoothing_mm:.3f} mm")
        print(f"output LUT path             = {args.output_mat_path}")

        valid_dense_depth_mask = dense_z_curve > 0.05 * np.nanmax(dense_z_curve)
        if np.any(valid_dense_depth_mask):
            mae_dense_sigma = np.nanmean(np.abs(dense_single_sigma_lut[valid_dense_depth_mask] - dense_ref_sigma[valid_dense_depth_mask]))
            mae_dense_sigma1 = np.nanmean(np.abs(dense_sigma1_lut[valid_dense_depth_mask] - dense_ref_sigma1[valid_dense_depth_mask]))
            mae_dense_sigma2 = np.nanmean(np.abs(dense_sigma2_lut[valid_dense_depth_mask] - dense_ref_sigma2[valid_dense_depth_mask]))
            mae_dense_weight = np.nanmean(np.abs(dense_weight_lut[valid_dense_depth_mask] - dense_ref_weight[valid_dense_depth_mask]))
            print(f"dense LUT sigma MAE         = {mae_dense_sigma:.6f} mm")
            print(f"dense LUT sigma1 MAE        = {mae_dense_sigma1:.6f} mm")
            print(f"dense LUT sigma2 MAE        = {mae_dense_sigma2:.6f} mm")
            print(f"dense LUT weight MAE        = {mae_dense_weight:.6f}")
    else:
        print("dense LUT export            = disabled (--export-lut not set)")
        print("dense fitting               = skipped; checking only original LUT depths")

    if valid_single.any():
        mae_single = np.nanmean(np.abs(single_sigma_total[valid_single] - ref_sigma_total[valid_single]))
        rmse_single = np.sqrt(np.nanmean((single_sigma_total[valid_single] - ref_sigma_total[valid_single]) ** 2))
        print(f"single total sigma MAE      = {mae_single:.6f} mm")
        print(f"single total sigma RMSE     = {rmse_single:.6f} mm")

    if valid_single_quality.any():
        mean_single_ref_error = np.nanmean(single_ref_shape_error[valid_single_quality])
        mean_single_fit_error = np.nanmean(single_fit_shape_error[valid_single_quality])
        n_single_fit_better = int(np.sum(single_fit_shape_error[valid_single_quality] < single_ref_shape_error[valid_single_quality]))
        print(f"single ref profile error    = {mean_single_ref_error:.6f} (full lateral range)")
        print(f"single fit profile error    = {mean_single_fit_error:.6f} (full lateral range)")
        print(f"single fit beats reference  = {n_single_fit_better} / {int(valid_single_quality.sum())}")

    if valid_double.any():
        mae_sigma1 = np.nanmean(np.abs(double_sigma1_total[valid_double] - ref_sigma1_total[valid_double]))
        mae_sigma2 = np.nanmean(np.abs(double_sigma2_total[valid_double] - ref_sigma2_total[valid_double]))
        mae_weight = np.nanmean(np.abs(double_weight_fit[valid_double] - ref_weight[valid_double]))
        rmse_sigma1 = np.sqrt(np.nanmean((double_sigma1_total[valid_double] - ref_sigma1_total[valid_double]) ** 2))
        rmse_sigma2 = np.sqrt(np.nanmean((double_sigma2_total[valid_double] - ref_sigma2_total[valid_double]) ** 2))
        rmse_weight = np.sqrt(np.nanmean((double_weight_fit[valid_double] - ref_weight[valid_double]) ** 2))
        print(f"double sigma1 MAE           = {mae_sigma1:.6f} mm")
        print(f"double sigma1 RMSE          = {rmse_sigma1:.6f} mm")
        print(f"double sigma2 MAE           = {mae_sigma2:.6f} mm")
        print(f"double sigma2 RMSE          = {rmse_sigma2:.6f} mm")
        print(f"double weight MAE           = {mae_weight:.6f}")
        print(f"double weight RMSE          = {rmse_weight:.6f}")

    if valid_double_quality.any():
        mean_double_ref_error = np.nanmean(double_ref_shape_error[valid_double_quality])
        mean_double_fit_error = np.nanmean(double_fit_shape_error[valid_double_quality])
        n_double_fit_better = int(np.sum(double_fit_shape_error[valid_double_quality] < double_ref_shape_error[valid_double_quality]))
        print(f"double ref profile error    = {mean_double_ref_error:.6f} (full lateral range)")
        print(f"double fit profile error    = {mean_double_fit_error:.6f} (full lateral range)")
        print(f"double fit beats reference  = {n_double_fit_better} / {int(valid_double_quality.sum())}")

    safe_energy = f"{energy_mev:.4f}".replace(".", "p")
    out_dir = args.out_dir / f"energy_{safe_energy}MeV"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes[0, 0].plot(ref_depths_mm, ref_sigma_total, label="reference total sigma")
    axes[0, 0].plot(ref_depths_mm, single_sigma_total, "--", label="fit total sigma")
    axes[0, 0].set_title("Single Gaussian total sigma")
    axes[0, 0].set_ylabel("sigma [mm]")
    axes[0, 0].legend()

    axes[0, 1].plot(ref_depths_mm, ref_sigma1_total, label="reference total sigma1")
    axes[0, 1].plot(ref_depths_mm, double_sigma1_total, "--", label="fit total sigma1")
    axes[0, 1].set_title("Double Gaussian total sigma1")
    axes[0, 1].set_ylabel("sigma [mm]")
    axes[0, 1].legend()

    axes[1, 0].plot(ref_depths_mm, ref_sigma2_total, label="reference total sigma2")
    axes[1, 0].plot(ref_depths_mm, double_sigma2_total, "--", label="fit total sigma2")
    axes[1, 0].set_title("Double Gaussian total sigma2")
    axes[1, 0].set_xlabel("depth [mm]")
    axes[1, 0].set_ylabel("sigma [mm]")
    axes[1, 0].legend()

    axes[1, 1].plot(ref_depths_mm, ref_weight, label="reference weight")
    axes[1, 1].plot(ref_depths_mm, double_weight_fit, "--", label="fit weight")
    axes[1, 1].set_title("Double Gaussian halo weight")
    axes[1, 1].set_xlabel("depth [mm]")
    axes[1, 1].set_ylabel("weight")
    axes[1, 1].legend()

    for ax in axes.ravel():
        ax.set_xlim(0.0, depth_plot_xlim_mm)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir / "parameter_comparison.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    axes[0].plot(ref_depths_mm, ref_sigma, label="reference sigma")
    axes[0].plot(ref_depths_mm, single_sigma_deconv, "--", label="fit deconvolved sigma")
    axes[0].set_title("Single deconvolved sigma")
    axes[0].set_ylabel("sigma [mm]")
    axes[0].legend()

    axes[1].plot(ref_depths_mm, ref_sigma1, label="reference sigma1")
    axes[1].plot(ref_depths_mm, double_sigma1_deconv, "--", label="fit deconvolved sigma1")
    axes[1].set_title("Double deconvolved sigma1")
    axes[1].legend()

    axes[2].plot(ref_depths_mm, ref_sigma2, label="reference sigma2")
    axes[2].plot(ref_depths_mm, double_sigma2_deconv, "--", label="fit deconvolved sigma2")
    axes[2].set_title("Double deconvolved sigma2")
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("depth [mm]")
        ax.set_xlim(0.0, depth_plot_xlim_mm)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir / "deconvolved_sigma_comparison.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    axes[0].plot(ref_depths_mm, single_ref_shape_error, label="reference single")
    axes[0].plot(ref_depths_mm, single_fit_shape_error, "--", label="fitted single")
    axes[0].set_title("Single-profile mismatch to MC (full lateral range)")
    axes[0].set_ylabel("normalized error [lower is better]")
    axes[0].legend()

    axes[1].plot(ref_depths_mm, double_ref_shape_error, label="reference double")
    axes[1].plot(ref_depths_mm, double_fit_shape_error, "--", label="fitted double")
    axes[1].set_title("Double-profile mismatch to MC (full lateral range)")
    axes[1].set_ylabel("normalized error [lower is better]")
    axes[1].legend()

    for ax in axes:
        ax.set_xlabel("depth [mm]")
        ax.set_xlim(0.0, depth_plot_xlim_mm)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir / "fit_quality_comparison.png", dpi=200)
    plt.close(fig)

    if args.export_lut:
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
        axes = axes.ravel()

        axes[0].plot(dense_depths_mm, dense_z_curve, label="dense MC Z", color="C0")
        axes[0].plot(ref_depths_mm, ref_Z, "o", ms=3, label="reference Z", color="black")
        axes[0].set_title("Depth dose Z")
        axes[0].set_ylabel("Z / cm")
        axes[0].legend()

        axes[1].plot(dense_depths_mm[dense_fit_ok_single], dense_single_sigma_total_raw[dense_fit_ok_single], ".", alpha=0.35, label="raw dense single total fits", color="C0")
        if args.longitudinal_smoothing_mm > 0.0:
            axes[1].plot(dense_depths_mm, np.sqrt(sigma0_mm**2 + dense_single_sigma_lut_unsmoothed**2), "--", label="unsmoothed LUT single σ at runtime", color="0.5")
        axes[1].plot(dense_depths_mm, dense_single_sigma_total_runtime, label="saved LUT single σ at runtime", color="C1")
        axes[1].plot(ref_depths_mm, ref_sigma_total, "o", ms=3, label="reference single σ at runtime", color="black")
        axes[1].set_title("Single sigma in runtime space")
        axes[1].set_ylabel("total sigma [mm]")
        axes[1].legend()

        axes[2].plot(dense_depths_mm[dense_fit_ok_double], dense_sigma1_raw[dense_fit_ok_double], ".", alpha=0.20, label="raw free sigma1 fits (unstable)", color="0.5")
        if args.longitudinal_smoothing_mm > 0.0:
            axes[2].plot(dense_depths_mm, dense_sigma1_lut_unsmoothed, "--", label="unsmoothed LUT sigma1", color="0.5")
        axes[2].plot(dense_depths_mm, dense_sigma1_lut, label="saved LUT sigma1", color="C1")
        axes[2].plot(ref_depths_mm, ref_sigma1, "o", ms=3, label="reference sigma1", color="black")
        axes[2].set_title("Double sigma1 in LUT space")
        axes[2].set_ylabel("transport sigma [mm]")
        axes[2].legend()

        axes[3].plot(dense_depths_mm[dense_fit_ok_double], dense_sigma2_raw[dense_fit_ok_double], ".", alpha=0.20, label="raw free sigma2 fits (unstable)", color="0.5")
        if args.longitudinal_smoothing_mm > 0.0:
            axes[3].plot(dense_depths_mm, dense_sigma2_lut_unsmoothed, "--", label="unsmoothed LUT sigma2", color="0.5")
        axes[3].plot(dense_depths_mm, dense_sigma2_lut, label="saved LUT sigma2", color="C1")
        axes[3].plot(ref_depths_mm, ref_sigma2, "o", ms=3, label="reference sigma2", color="black")
        axes[3].set_title("Double sigma2 in LUT space")
        axes[3].set_ylabel("transport sigma [mm]")
        axes[3].legend()

        axes[4].plot(dense_depths_mm[dense_fit_ok_double], dense_double_weight_raw[dense_fit_ok_double], ".", alpha=0.20, label="raw free weights (unstable)", color="0.5")
        axes[4].plot(dense_depths_mm, dense_weight_lut, label="saved LUT weight", color="C1")
        axes[4].plot(ref_depths_mm, ref_weight, "o", ms=3, label="reference weight", color="black")
        axes[4].set_title("Double weight in LUT space")
        axes[4].set_ylabel("weight")
        axes[4].legend()

        axes[5].plot(dense_depths_mm, dense_ref_let, label="saved LUT LET", color="C1")
        axes[5].plot(ref_depths_mm, ref_let, "o", ms=3, label="reference LET", color="black")
        axes[5].set_title("LET")
        axes[5].set_ylabel("LET")
        axes[5].legend()

        for ax in axes:
            ax.set_xlabel("depth [mm]")
            ax.set_xlim(0.0, depth_plot_xlim_mm)
            ax.grid(True, alpha=0.2)

        fig.tight_layout()
        fig.savefig(out_dir / "dense_lut_comparison.png", dpi=200)
        plt.close(fig)

    n_cols = len(PROFILE_DEPTHS_TO_PLOT_MM)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 8.0), sharex='col')
    if n_cols == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col, z_plot in enumerate(PROFILE_DEPTHS_TO_PLOT_MM):
        ax_profile = axes[0, col]
        ax_residual = axes[1, col]
        i = int(np.argmin(np.abs(ref_depths_mm - z_plot)))
        depth_idx = int(depth_indices[i])
        profile = fit_img[:, depth_idx].astype(np.float64, copy=False)
        y_profile = profile
        peak = float(np.nanmax(y_profile))
        if not np.isfinite(peak) or peak <= 0.0:
            continue

        mc_norm = y_profile / peak
        ax_profile.scatter(
            x_profile,
            mc_norm,
            s=18,
            alpha=0.5,
            facecolors="C0",
            edgecolors="black",
            linewidths=0.4,
            label="MC normalized",
        )

        y_ref_single = gaussian_profile_1d(x_profile, 1.0, ref_sigma_total[i])
        y_ref_single /= np.nanmax(y_ref_single)
        ax_profile.semilogy(
            x_profile,
            y_ref_single,
            label=(
                f"reference single σ={ref_sigma_total[i]:.3f} mm, "
                f"E={single_ref_shape_error[i]:.4f}"
            ),
        )
        ax_residual.plot(
            x_profile,
            y_ref_single - mc_norm,
            label="reference single - MC",
        )

        y_ref_double = double_gaussian_profile_1d_amplitude_weight(
            x_profile,
            1.0,
            ref_sigma1_total[i],
            ref_sigma2_total[i],
            ref_weight[i],
        )
        y_ref_double /= np.nanmax(y_ref_double)
        ax_profile.semilogy(
            x_profile,
            y_ref_double,
            ":",
            label=(
                f"reference double σ1={ref_sigma1_total[i]:.3f}, "
                f"σ2={ref_sigma2_total[i]:.3f}, "
                f"w={ref_weight[i]:.4f}, "
                f"E={double_ref_shape_error[i]:.4f}"
            ),
        )
        ax_residual.plot(
            x_profile,
            y_ref_double - mc_norm,
            ":",
            label="reference double - MC",
        )

        if fit_ok_single[i]:
            y_single = gaussian_profile_1d(x_profile, single_amplitude[i], single_sigma_total[i])
            y_single_norm = y_single / np.nanmax(y_single)
            ax_profile.semilogy(
                x_profile,
                y_single_norm,
                "--",
                label=f"single fit σ={single_sigma_total[i]:.3f} mm, E={single_fit_shape_error[i]:.4f}",
            )
            ax_residual.plot(
                x_profile,
                y_single_norm - mc_norm,
                "--",
                label="single fit - MC",
            )

        if double_repr_ok[i]:
            y_double = double_gaussian_profile_1d_amplitude_weight(
                x_profile,
                double_amplitude[i],
                double_sigma1_total[i],
                double_sigma2_total[i],
                double_weight_fit[i],
            )
            y_double_norm = y_double / np.nanmax(y_double)
            double_label_prefix = "double fallback" if double_fallback_used[i] else "double staged"
            ax_profile.semilogy(
                x_profile,
                y_double_norm,
                "--",
                color="black",
                label=(
                    f"{double_label_prefix} σ1={double_sigma1_total[i]:.3f}, "
                    f"σ2={double_sigma2_total[i]:.3f}, "
                    f"w={double_weight_fit[i]:.4f}, "
                    f"E={double_fit_shape_error[i]:.4f}"
                ),
            )
            ax_residual.plot(
                x_profile,
                np.abs(y_double_norm - mc_norm),
                "--",
                color="black",
                label=f"{double_label_prefix} - MC",
            )

        ax_profile.set_title(f"z = {ref_depths_mm[i]:.2f} mm")
        ax_profile.set_xlim(-profile_plot_xlim_mm, profile_plot_xlim_mm)
        ax_profile.set_ylim(1e-6, 2.0)
        ax_profile.grid(True, alpha=0.2)
        ax_profile.legend(fontsize=7)

        ax_residual.axhline(0.0, color="0.3", linewidth=0.8)
        ax_residual.set_xlim(-profile_plot_xlim_mm, profile_plot_xlim_mm)
        ax_residual.grid(True, alpha=0.2)
        ax_residual.set_xlabel("lateral y [mm]")
        ax_residual.set_ylabel("residual")
        ax_residual.set_yscale("log")

    axes[0, 0].set_ylabel("normalized profile")
    fig.tight_layout()
    fig.savefig(out_dir / "profile_diagnostics.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.asarray(z_from_edep["depth_mm"], dtype=np.float64), np.asarray(z_from_edep["Z_est"], dtype=np.float64), label="Z estimated from edep")
    ax.plot(ref_depths_mm, ref_Z, "--", label="reference MAT Z")
    ax.set_xlim(0.0, depth_plot_xlim_mm)
    ax.set_xlabel("depth [mm]")
    ax.set_ylabel("Z / cm")
    ax.set_title("Depth-dose comparison")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "depth_dose_comparison.png", dpi=200)
    plt.close(fig)

    print()
    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
