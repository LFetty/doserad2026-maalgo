"""Shared helpers for DoseRAD proton case scripts (plotting and evaluation)."""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pydosert-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import blosc2
import numpy as np
import SimpleITK as sitk
import torch

_SCRIPTS_DIR = Path(__file__).parent
ROOT = _SCRIPTS_DIR.parent
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pydose_rt.data.ion_beam import IonSpotBeam, IonSpotBeamSequence
from pydose_rt.physics.kernels.ion_lut import PyRadPlanIonLUT

DEFAULT_PYRADPLAN_MACHINE_MAT = ROOT / "example_data" / "pyradplan" / "protons_Generic.mat"
DEFAULT_MC_FIT_MAT = ROOT / "example_data" / "mc_fit_smooth" / "lut_fast_tail_safe_halo_mc_v1.mat"
DEFAULT_MC_FIT_3D_OPT_MAT = ROOT / "example_data" / "mc_fit_smooth" / "lut_fast_3d_opt.mat"


# ---------------------------------------------------------------------------
# Case file resolution
# ---------------------------------------------------------------------------

def _resolve_case_files(case_dir: Path) -> tuple[str, Path, Path, Path]:
    case_dir = case_dir.resolve()
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Case directory does not exist: {case_dir}")

    patient_id = case_dir.name
    canonical_plan_path = case_dir / f"{patient_id}.json"
    if canonical_plan_path.is_file():
        plan_path = canonical_plan_path
    else:
        plan_paths = sorted(
            path for path in case_dir.glob("*.json")
            if not path.name.endswith("_metrics.json")
        )
        if len(plan_paths) != 1:
            raise FileNotFoundError(
                f"Expected exactly one plan JSON in {case_dir}, found {len(plan_paths)}"
            )
        plan_path = plan_paths[0]
        patient_id = plan_path.stem

    ct_path = case_dir / "image" / "ct.mha"
    if not ct_path.is_file():
        raise FileNotFoundError(f"Missing CT image: {ct_path}")

    dose_dir = case_dir / "dose"
    if not dose_dir.is_dir():
        raise FileNotFoundError(f"Missing dose directory: {dose_dir}")

    return patient_id, plan_path, ct_path, dose_dir


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _xyz_to_zyx(coords_xyz: list[float] | tuple[float, float, float]) -> np.ndarray:
    coords = np.asarray(coords_xyz, dtype=np.float32)
    return coords[[2, 1, 0]]


def _origin_zyx(image: sitk.Image) -> np.ndarray:
    return _xyz_to_zyx(tuple(float(v) for v in image.GetOrigin()))


def _resolution_zyx(image: sitk.Image) -> tuple[float, float, float]:
    return _xyz_to_zyx(tuple(float(v) for v in image.GetSpacing()))


# ---------------------------------------------------------------------------
# Plan helpers
# ---------------------------------------------------------------------------

def _expected_dose_filename(beam_json: dict, ray_json: dict, beamlet_json: dict) -> str:
    return (
        f"Dose_B{beam_json['beam_idx']}_R{ray_json['ray_idx']}"
        f"_L{beamlet_json['beamlet_idx']}.mha"
    )


def _assert_case_is_complete(plan: dict, dose_dir: Path) -> None:
    expected = {
        _expected_dose_filename(beam_json, ray_json, beamlet_json)
        for beam_json in plan["beams"]
        for ray_json in beam_json["rays"]
        for beamlet_json in ray_json["beamlets"]
    }
    actual = {path.name for path in dose_dir.glob("*.mha")}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    if missing or extra:
        message = [f"DoseRAD case is incomplete under {dose_dir}"]
        if missing:
            message.append(f"Missing {len(missing)} files, first 20: {missing[:20]}")
        if extra:
            message.append(f"Unexpected {len(extra)} files, first 20: {extra[:20]}")
        raise FileNotFoundError(" | ".join(message))


# ---------------------------------------------------------------------------
# CT / beam parameter conversions
# ---------------------------------------------------------------------------

def _doserad_density_from_hu(ct_hu: np.ndarray, beam_parameters: dict) -> np.ndarray:
    entries = beam_parameters["hu_to_density"]["entries"]
    hu = np.asarray([float(e["hu"]) for e in entries], dtype=np.float32)
    density = np.asarray([float(e["density_g_cm3"]) for e in entries], dtype=np.float32)
    return np.interp(ct_hu, hu, density, left=density[0], right=density[-1]).astype(
        np.float32, copy=False
    )


def _dose_rad_sigma_spot_mm(beam_parameters: dict, energy_mev: float) -> float:
    entries = beam_parameters["proton"]["energy_table"]
    energies = np.asarray([float(e["energy_mev"]) for e in entries], dtype=np.float64)
    sigmas = np.asarray([float(e["sigma_spot_mm"]) for e in entries], dtype=np.float64)
    return float(
        np.interp(float(energy_mev), energies, sigmas, left=sigmas[0], right=sigmas[-1])
    )


def _ray_sad_mm(plan: dict, ray_json: dict) -> float:
    if "SAD" in plan:
        return float(plan["SAD"])
    if "ray_source" not in ray_json or "ray_target" not in ray_json:
        raise KeyError("DoseRAD proton plan is missing SAD and ray_source/ray_target")
    source = np.asarray(ray_json["ray_source"], dtype=np.float64)
    target = np.asarray(ray_json["ray_target"], dtype=np.float64)
    return float(np.linalg.norm(source - target))


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


def _propagate_point_source_sigma_mm(
    sigma_bams_mm: float,
    source_to_surface_mm: float,
    sad_mm: float,
    bams_to_iso_dist_mm: float,
) -> float:
    source_to_bams_mm = float(sad_mm) - float(bams_to_iso_dist_mm)
    if source_to_bams_mm <= 0.0:
        raise ValueError(
            f"Expected SAD > bams_to_iso_dist_mm, got SAD={sad_mm} and "
            f"bams_to_iso_dist_mm={bams_to_iso_dist_mm}"
        )
    return float(sigma_bams_mm) * (float(source_to_surface_mm) / source_to_bams_mm)


def _estimate_ssd_mm(
    ct_hu: np.ndarray,
    resolution_zyx: tuple[float, float, float],
    sad_mm: float,
    gantry_angle_deg: float,
    iso_center_mm_zyx: tuple[float, float, float],
    skin_hu_threshold: float,
) -> float:
    # Beam axis in z-y-x coordinates from the gantry angle.
    theta = math.radians(float(gantry_angle_deg))
    axis = np.array([0.0, math.cos(theta), -math.sin(theta)], dtype=np.float64)
    iso = np.array(iso_center_mm_zyx, dtype=np.float64)
    # Source position inferred from the isocenter and SAD.
    source = iso - float(sad_mm) * axis
    shape = ct_hu.shape

    # Determine the ray segment inside the CT volume.
    bounds_min = np.zeros(3)
    bounds_max = np.array(
        [(shape[i] - 1) * resolution_zyx[i] for i in range(3)], dtype=np.float64
    )
    eps = 1e-9
    inv = 1.0 / np.where(np.abs(axis) < eps, eps, axis)
    t0 = (bounds_min - source) * inv
    t1 = (bounds_max - source) * inv
    t_enter = float(max(np.minimum(t0, t1).max(), 0.0))
    t_exit = float(np.maximum(t0, t1).min())
    if t_exit <= t_enter:
        # If the ray misses the volume, fall back to source-to-isocenter distance.
        return float(np.linalg.norm(iso - source))

    # Sample the ray at a fixed spacing through the volume.
    step = 1.0
    n_steps = int(np.ceil((t_exit - t_enter) / step)) + 1
    ts = t_enter + np.arange(n_steps) * step
    ts = np.clip(ts, 0.0, t_exit)
    pts = source[None, :] + ts[:, None] * axis[None, :]
    # Map physical coordinates to voxel indices and clamp to bounds.
    idx = pts / np.asarray(resolution_zyx, dtype=np.float64)[None, :]
    idx = np.clip(
        idx, 0.0, np.asarray([shape[i] - 1 for i in range(3)], dtype=np.float64)[None, :]
    )
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(
        i0 + 1, np.asarray([shape[i] - 1 for i in range(3)], dtype=np.int64)[None, :]
    )
    frac = idx - i0

    def gather(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        return ct_hu[a, b, c].astype(np.float64, copy=False)

    # Trilinear interpolation of CT Hounsfield units along the sampled ray.
    c000 = gather(i0[:, 0], i0[:, 1], i0[:, 2])
    c001 = gather(i0[:, 0], i0[:, 1], i1[:, 2])
    c010 = gather(i0[:, 0], i1[:, 1], i0[:, 2])
    c011 = gather(i0[:, 0], i1[:, 1], i1[:, 2])
    c100 = gather(i1[:, 0], i0[:, 1], i0[:, 2])
    c101 = gather(i1[:, 0], i0[:, 1], i1[:, 2])
    c110 = gather(i1[:, 0], i1[:, 1], i0[:, 2])
    c111 = gather(i1[:, 0], i1[:, 1], i1[:, 2])
    c00 = c000 * (1.0 - frac[:, 2]) + c001 * frac[:, 2]
    c01 = c010 * (1.0 - frac[:, 2]) + c011 * frac[:, 2]
    c10 = c100 * (1.0 - frac[:, 2]) + c101 * frac[:, 2]
    c11 = c110 * (1.0 - frac[:, 2]) + c111 * frac[:, 2]
    c0 = c00 * (1.0 - frac[:, 1]) + c01 * frac[:, 1]
    c1 = c10 * (1.0 - frac[:, 1]) + c11 * frac[:, 1]
    hu = c0 * (1.0 - frac[:, 0]) + c1 * frac[:, 0]

    # The first point above threshold is used as the estimated surface location.
    above = np.nonzero(hu > float(skin_hu_threshold))[0]
    if above.size == 0:
        return float(t_enter)
    return float(ts[above[0]])


# ---------------------------------------------------------------------------
# Beam / ray sequence construction
# ---------------------------------------------------------------------------

def _match_ray_layer_sigmas_mm(
    beamlets: list[dict],
    beam_parameters: dict,
    beam_json: dict,
    ray_json: dict,
    origin_zyx: np.ndarray,
    ct_hu: np.ndarray,
    resolution_zyx: tuple[float, float, float],
    sad_mm: float,
    skin_hu_threshold: float,
    sigma_mode: str,
    bams_to_iso_dist_mm: float,
    lut: PyRadPlanIonLUT,
) -> tuple[list[list[float]], float | None]:
    iso_center = tuple((_xyz_to_zyx(ray_json["ray_target"]) - origin_zyx).tolist())
    ssd_mm = _estimate_ssd_mm(
        ct_hu=ct_hu,
        resolution_zyx=resolution_zyx,
        sad_mm=sad_mm,
        gantry_angle_deg=_ray_gantry_angle_deg(beam_json, ray_json),
        iso_center_mm_zyx=iso_center,
        skin_hu_threshold=skin_hu_threshold,
    )

    layer_sigmas_mm: list[list[float]] = []
    for beamlet_json in beamlets:
        energy_mev = float(beamlet_json["energy"])
        sigma_bams_mm = _dose_rad_sigma_spot_mm(beam_parameters, energy_mev)
        if sigma_mode == "focus":
            sigma_x_mm = float(lut.get_initial_sigma(energy_mev, float(ssd_mm)).detach().cpu())
            sigma_y_mm = sigma_x_mm
        elif sigma_mode == "beam_params":
            sigma_x_mm = sigma_bams_mm
            sigma_y_mm = sigma_x_mm
        else:
            sigma_x_mm = _propagate_point_source_sigma_mm(
                sigma_bams_mm=sigma_bams_mm,
                source_to_surface_mm=float(ssd_mm),
                sad_mm=sad_mm,
                bams_to_iso_dist_mm=bams_to_iso_dist_mm,
            )
            sigma_y_mm = sigma_x_mm
        layer_sigmas_mm.append([sigma_x_mm, sigma_y_mm])
    return layer_sigmas_mm, ssd_mm


def _make_beam_sequence(
    plan: dict,
    beam_parameters: dict,
    ct_hu: np.ndarray,
    origin_zyx: np.ndarray,
    resolution_zyx: tuple[float, float, float],
    beam_index: int,
    particles_per_beamlet: float,
    gantry_offset_deg: float,
    skin_hu_threshold: float,
    sigma_mode: str,
    bams_to_iso_dist_mm: float,
    lut: PyRadPlanIonLUT,
    device: torch.device,
    dtype: torch.dtype,
    beamlets_as_spots: bool = False,
) -> tuple[IonSpotBeamSequence, list[float] | None]:
    beam_json = plan["beams"][beam_index]
    if beamlets_as_spots:
        ray_angles = [
            _ray_gantry_angle_deg(beam_json, ray_json) + gantry_offset_deg
            for ray_json in beam_json["rays"]
        ]
        if ray_angles and max(abs(a - ray_angles[0]) for a in ray_angles) <= 1e-4:
            ref_ray = beam_json["rays"][0]
            ref_iso_zyx = _xyz_to_zyx(ref_ray["ray_target"]).astype(np.float64) - origin_zyx
            theta = math.radians(float(ray_angles[0]))
            lat_zyx = np.array([0.0, math.sin(theta), math.cos(theta)], dtype=np.float64)
            spot_positions: list[list[float]] = []
            layer_energies: list[float] = []
            layer_sigmas: list[list[float]] = []
            ray_ssd_values_mm: list[float] = []
            sad_values: list[float] = []

            for ray_json in beam_json["rays"]:
                sad_mm = _ray_sad_mm(plan, ray_json)
                sad_values.append(float(sad_mm))
                ray_iso_zyx = _xyz_to_zyx(ray_json["ray_target"]).astype(np.float64) - origin_zyx
                delta = ray_iso_zyx - ref_iso_zyx
                h_offset_mm = float(delta[0])
                w_offset_mm = float(np.dot(delta, lat_zyx))
                beamlets = ray_json["beamlets"]
                ray_layer_sigmas_mm, ssd_mm = _match_ray_layer_sigmas_mm(
                    beamlets=beamlets,
                    beam_parameters=beam_parameters,
                    beam_json=beam_json,
                    ray_json=ray_json,
                    origin_zyx=origin_zyx,
                    ct_hu=ct_hu,
                    resolution_zyx=resolution_zyx,
                    sad_mm=sad_mm,
                    skin_hu_threshold=skin_hu_threshold,
                    sigma_mode=sigma_mode,
                    bams_to_iso_dist_mm=bams_to_iso_dist_mm,
                    lut=lut,
                )
                if ssd_mm is not None:
                    ray_ssd_values_mm.append(float(ssd_mm))
                for beamlet_json, sigma_xy in zip(beamlets, ray_layer_sigmas_mm, strict=True):
                    spot_positions.append([w_offset_mm, h_offset_mm])
                    layer_energies.append(float(beamlet_json["energy"]))
                    layer_sigmas.append(sigma_xy)

            if layer_energies:
                num_beamlets = len(layer_energies)
                beam = IonSpotBeam.create(
                    gantry_angle_deg=float(ray_angles[0]),
                    spot_positions_mm=torch.tensor(spot_positions, device=device, dtype=dtype),
                    spot_weights=torch.full(
                        (num_beamlets,), float(particles_per_beamlet), device=device, dtype=dtype
                    ),
                    spot_layer_index=torch.arange(num_beamlets, device=device, dtype=torch.long),
                    layer_energies_mev=torch.tensor(layer_energies, device=device, dtype=dtype),
                    layer_sigmas_mm=torch.tensor(layer_sigmas, device=device, dtype=dtype),
                    iso_center=tuple(ref_iso_zyx.tolist()),
                    sad_mm=float(np.mean(sad_values)) if sad_values else _ray_sad_mm(plan, ref_ray),
                    requires_grad=False,
                )
                ssd_values_mm = [float(np.mean(ray_ssd_values_mm))] if ray_ssd_values_mm else None
                return IonSpotBeamSequence.from_beams([beam]), ssd_values_mm

    beams: list[IonSpotBeam] = []
    ray_ssd_values_mm: list[float] = []
    for ray_json in beam_json["rays"]:
        sad_mm = _ray_sad_mm(plan, ray_json)
        iso_center = tuple((_xyz_to_zyx(ray_json["ray_target"]) - origin_zyx).tolist())
        beamlets = ray_json["beamlets"]
        layer_energies_mev = torch.tensor(
            [float(b["energy"]) for b in beamlets], device=device, dtype=dtype
        )
        ray_layer_sigmas_mm, ssd_mm = _match_ray_layer_sigmas_mm(
            beamlets=beamlets,
            beam_parameters=beam_parameters,
            beam_json=beam_json,
            ray_json=ray_json,
            origin_zyx=origin_zyx,
            ct_hu=ct_hu,
            resolution_zyx=resolution_zyx,
            sad_mm=sad_mm,
            skin_hu_threshold=skin_hu_threshold,
            sigma_mode=sigma_mode,
            bams_to_iso_dist_mm=bams_to_iso_dist_mm,
            lut=lut,
        )
        layer_sigmas_mm = torch.tensor(ray_layer_sigmas_mm, device=device, dtype=dtype)
        if ssd_mm is not None:
            ray_ssd_values_mm.append(float(ssd_mm))
        num_beamlets = len(beamlets)
        beams.append(
            IonSpotBeam.create(
                gantry_angle_deg=_ray_gantry_angle_deg(beam_json, ray_json) + gantry_offset_deg,
                spot_positions_mm=torch.zeros((num_beamlets, 2), device=device, dtype=dtype),
                spot_weights=torch.full(
                    (num_beamlets,), float(particles_per_beamlet), device=device, dtype=dtype
                ),
                spot_layer_index=torch.arange(num_beamlets, device=device, dtype=torch.long),
                layer_energies_mev=layer_energies_mev,
                layer_sigmas_mm=layer_sigmas_mm,
                iso_center=iso_center,
                sad_mm=sad_mm,
                requires_grad=False,
            )
        )
    ssd_values_mm = ray_ssd_values_mm if ray_ssd_values_mm else None
    return IonSpotBeamSequence.from_beams(beams), ssd_values_mm


def _make_ray_sequence(
    plan: dict,
    beam_parameters: dict,
    ct_hu: np.ndarray,
    origin_zyx: np.ndarray,
    resolution_zyx: tuple[float, float, float],
    beam_index: int,
    ray_index: int,
    beamlet_index: int | None,
    particles_per_beamlet: float,
    gantry_offset_deg: float,
    skin_hu_threshold: float,
    sigma_mode: str,
    bams_to_iso_dist_mm: float,
    lut: PyRadPlanIonLUT,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[IonSpotBeamSequence, float | None]:
    beam_json = plan["beams"][beam_index]
    ray_json = beam_json["rays"][ray_index]
    sad_mm = _ray_sad_mm(plan, ray_json)
    iso_center = tuple((_xyz_to_zyx(ray_json["ray_target"]) - origin_zyx).tolist())
    all_beamlets = ray_json["beamlets"]
    if beamlet_index is None:
        beamlets = all_beamlets
    else:
        if beamlet_index < 0 or beamlet_index >= len(all_beamlets):
            raise ValueError(
                f"--beamlet-index must be in [0, {len(all_beamlets) - 1}] "
                f"for beam {beam_index}, ray {ray_index}; got {beamlet_index}"
            )
        beamlets = [all_beamlets[beamlet_index]]
    layer_energies_mev = torch.tensor(
        [float(b["energy"]) for b in beamlets], device=device, dtype=dtype
    )
    ray_layer_sigmas_mm, ssd_mm = _match_ray_layer_sigmas_mm(
        beamlets=beamlets,
        beam_parameters=beam_parameters,
        beam_json=beam_json,
        ray_json=ray_json,
        origin_zyx=origin_zyx,
        ct_hu=ct_hu,
        resolution_zyx=resolution_zyx,
        sad_mm=sad_mm,
        skin_hu_threshold=skin_hu_threshold,
        sigma_mode=sigma_mode,
        bams_to_iso_dist_mm=bams_to_iso_dist_mm,
        lut=lut,
    )
    layer_sigmas_mm = torch.tensor(ray_layer_sigmas_mm, device=device, dtype=dtype)
    num_beamlets = len(beamlets)
    beam = IonSpotBeam.create(
        gantry_angle_deg=_ray_gantry_angle_deg(beam_json, ray_json) + gantry_offset_deg,
        spot_positions_mm=torch.zeros((num_beamlets, 2), device=device, dtype=dtype),
        spot_weights=torch.full(
            (num_beamlets,), float(particles_per_beamlet), device=device, dtype=dtype
        ),
        spot_layer_index=torch.arange(num_beamlets, device=device, dtype=torch.long),
        layer_energies_mev=layer_energies_mev,
        layer_sigmas_mm=layer_sigmas_mm,
        iso_center=iso_center,
        sad_mm=sad_mm,
        requires_grad=False,
    )
    return IonSpotBeamSequence.from_beams([beam]), ssd_mm


def _make_beamlet_batch_sequence(
    plan: dict,
    beam_parameters: dict,
    ct_hu: np.ndarray,
    origin_zyx: np.ndarray,
    resolution_zyx: tuple[float, float, float],
    beam_index: int,
    particles_per_beamlet: float,
    gantry_offset_deg: float,
    skin_hu_threshold: float,
    sigma_mode: str,
    bams_to_iso_dist_mm: float,
    lut: PyRadPlanIonLUT,
    device: torch.device,
    dtype: torch.dtype,
    ray_indices: list[int] | None = None,
    beamlet_index: int | None = None,
) -> tuple[IonSpotBeamSequence, list[float] | None]:
    """Build one IonSpotBeam per DoseRAD beamlet for batched dense BEV crops."""
    beam_json = plan["beams"][beam_index]
    beams: list[IonSpotBeam] = []
    ssd_values: list[float] = []
    if ray_indices is None:
        ray_indices = list(range(len(beam_json["rays"])))
    for ray_index in ray_indices:
        ray_json = beam_json["rays"][ray_index]
        sad_mm = _ray_sad_mm(plan, ray_json)
        iso_center = tuple((_xyz_to_zyx(ray_json["ray_target"]) - origin_zyx).tolist())
        beamlets = _selected_beamlets(ray_json, beamlet_index, beam_index, ray_index)
        ray_layer_sigmas_mm, ssd_mm = _match_ray_layer_sigmas_mm(
            beamlets=beamlets,
            beam_parameters=beam_parameters,
            beam_json=beam_json,
            ray_json=ray_json,
            origin_zyx=origin_zyx,
            ct_hu=ct_hu,
            resolution_zyx=resolution_zyx,
            sad_mm=sad_mm,
            skin_hu_threshold=skin_hu_threshold,
            sigma_mode=sigma_mode,
            bams_to_iso_dist_mm=bams_to_iso_dist_mm,
            lut=lut,
        )
        for beamlet_json, sigma_xy in zip(beamlets, ray_layer_sigmas_mm, strict=True):
            beams.append(
                IonSpotBeam.create(
                    gantry_angle_deg=_ray_gantry_angle_deg(beam_json, ray_json) + gantry_offset_deg,
                    spot_positions_mm=torch.zeros((1, 2), device=device, dtype=dtype),
                    spot_weights=torch.full((1,), float(particles_per_beamlet), device=device, dtype=dtype),
                    spot_layer_index=torch.zeros((1,), device=device, dtype=torch.long),
                    layer_energies_mev=torch.tensor([float(beamlet_json["energy"])], device=device, dtype=dtype),
                    layer_sigmas_mm=torch.tensor([sigma_xy], device=device, dtype=dtype),
                    iso_center=iso_center,
                    sad_mm=sad_mm,
                    requires_grad=False,
                )
            )
            if ssd_mm is not None:
                ssd_values.append(float(ssd_mm))
    return IonSpotBeamSequence.from_beams(beams), ssd_values if ssd_values else None


def iter_single_beamlet_sequences(
    plan: dict,
    beam_parameters: dict,
    ct_hu: np.ndarray,
    origin_zyx: np.ndarray,
    resolution_zyx: tuple[float, float, float],
    beam_index: int,
    particles_per_beamlet: float,
    gantry_offset_deg: float,
    skin_hu_threshold: float,
    sigma_mode: str,
    bams_to_iso_dist_mm: float,
    lut: PyRadPlanIonLUT,
    device: torch.device,
    dtype: torch.dtype,
):
    """Yield one single-beamlet sequence per beamlet, in the SAME order as
    `_reference_paths_for_selection` (rays -> beamlets), so the k-th yield matches the
    k-th reference path 1:1 (correspondence guarantee). Each yield is a dict with the
    sequence plus identifying metadata (ray/beamlet index, energy, iso_center, gantry).
    """
    beam_json = plan["beams"][beam_index]
    for ray_index, ray_json in enumerate(beam_json["rays"]):
        sad_mm = _ray_sad_mm(plan, ray_json)
        iso_center = tuple((_xyz_to_zyx(ray_json["ray_target"]) - origin_zyx).tolist())
        beamlets = ray_json["beamlets"]
        gantry_deg = _ray_gantry_angle_deg(beam_json, ray_json) + gantry_offset_deg
        ray_layer_sigmas_mm, ssd_mm = _match_ray_layer_sigmas_mm(
            beamlets=beamlets,
            beam_parameters=beam_parameters,
            beam_json=beam_json,
            ray_json=ray_json,
            origin_zyx=origin_zyx,
            ct_hu=ct_hu,
            resolution_zyx=resolution_zyx,
            sad_mm=sad_mm,
            skin_hu_threshold=skin_hu_threshold,
            sigma_mode=sigma_mode,
            bams_to_iso_dist_mm=bams_to_iso_dist_mm,
            lut=lut,
        )
        for local_idx, (beamlet_json, sigma_xy) in enumerate(zip(beamlets, ray_layer_sigmas_mm, strict=True)):
            beam = IonSpotBeam.create(
                gantry_angle_deg=gantry_deg,
                spot_positions_mm=torch.zeros((1, 2), device=device, dtype=dtype),
                spot_weights=torch.full((1,), float(particles_per_beamlet), device=device, dtype=dtype),
                spot_layer_index=torch.zeros((1,), device=device, dtype=torch.long),
                layer_energies_mev=torch.tensor([float(beamlet_json["energy"])], device=device, dtype=dtype),
                layer_sigmas_mm=torch.tensor([sigma_xy], device=device, dtype=dtype),
                iso_center=iso_center,
                sad_mm=sad_mm,
                requires_grad=False,
            )
            yield {
                "ray_index": ray_index,
                "beamlet_index": local_idx,
                "energy_mev": float(beamlet_json["energy"]),
                "iso_center_zyx": iso_center,
                "gantry_deg": gantry_deg,
                "sequence": IonSpotBeamSequence.from_beams([beam]),
                "ssd_mm": float(ssd_mm) if ssd_mm is not None else None,
            }


# ---------------------------------------------------------------------------
# Reference dose I/O
# ---------------------------------------------------------------------------

def _read_reference_dose(dose_path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(dose_path))).astype(
        np.float32, copy=False
    )


# bbox tuple layout: (z0, z1, y0, y1, x0, x1)
RefBbox = tuple[int, int, int, int, int, int]


def _b2nd_path(mha_path: Path) -> Path:
    return mha_path.with_suffix(".b2nd")


def _read_reference_dose_b2nd(
    mha_path: Path,
) -> tuple[np.ndarray, RefBbox | None] | None:
    """Load bbox-cropped reference dose from .b2nd if it exists, else return None.

    Returns (bbox_array, (z0, z1, y0, y1, x0, x1)) where bbox_array covers only
    the nonzero region (plus PAD) of the original full-grid dose. If the .b2nd
    file is missing its bbox vlmeta (corrupted / older converter), falls back to
    reading the MHA and returns (full_arr, None) so the caller treats it as an
    uncropped reference.
    """
    b2nd = _b2nd_path(mha_path)
    if not b2nd.exists():
        return None
    na = blosc2.open(str(b2nd))
    try:
        coords = np.frombuffer(na.vlmeta["bbox"], dtype=np.int32)
    except KeyError:
        print(
            f"[ref-load] WARN: {b2nd.name} missing 'bbox' vlmeta; "
            "falling back to MHA",
            file=sys.stderr,
        )
        return _read_reference_dose(mha_path), None
    z0, z1, y0, y1, x0, x1 = (int(v) for v in coords)
    arr: np.ndarray = na[:]
    return arr, (z0, z1, y0, y1, x0, x1)


def _selected_ray_indices(beam_json: dict, ray_index: int | None, beam_index: int) -> list[int]:
    num_rays = len(beam_json["rays"])
    if ray_index is None:
        return list(range(num_rays))
    if ray_index < 0 or ray_index >= num_rays:
        raise ValueError(
            f"--ray-index must be in [0, {num_rays - 1}] for beam {beam_index}; got {ray_index}"
        )
    return [ray_index]


def _selected_beamlets(
    ray_json: dict, beamlet_index: int | None, beam_index: int, ray_index: int
) -> list[dict[str, Any]]:
    beamlets = ray_json["beamlets"]
    if beamlet_index is None:
        return list(beamlets)
    if beamlet_index < 0 or beamlet_index >= len(beamlets):
        raise ValueError(
            f"--beamlet-index must be in [0, {len(beamlets) - 1}] "
            f"for beam {beam_index}, ray {ray_index}; got {beamlet_index}"
        )
    return [beamlets[beamlet_index]]


def _reference_paths_for_selection(
    dose_dir: Path,
    beam_json: dict,
    beam_index: int,
    ray_indices: list[int],
    beamlet_index: int | None,
) -> list[Path]:
    dose_paths: list[Path] = []
    for ray_index in ray_indices:
        ray_json = beam_json["rays"][ray_index]
        for beamlet_json in _selected_beamlets(ray_json, beamlet_index, beam_index, ray_index):
            dose_paths.append(
                dose_dir / _expected_dose_filename(beam_json, ray_json, beamlet_json)
            )
    return dose_paths


def _load_reference_paths_sum(dose_paths: list[Path], io_workers: int = 1) -> np.ndarray:
    ref_sum: np.ndarray | None = None
    total = len(dose_paths)
    if io_workers <= 1:
        for idx, ref in enumerate(map(_read_reference_dose, dose_paths), start=1):
            if ref_sum is None:
                ref_sum = np.array(ref, copy=True)
            else:
                ref_sum += ref
            if idx == total or idx % 25 == 0:
                print(f"Loaded reference dose {idx}/{total}")
    else:
        with ThreadPoolExecutor(max_workers=io_workers) as pool:
            path_iter = iter(dose_paths)
            pending = {
                pool.submit(_read_reference_dose, next(path_iter))
                for _ in range(min(io_workers, total))
            }
            loaded = 0
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    ref = future.result()
                    loaded += 1
                    if ref_sum is None:
                        ref_sum = np.array(ref, copy=True)
                    else:
                        ref_sum += ref
                    if loaded == total or loaded % 25 == 0:
                        print(f"Loaded reference dose {loaded}/{total}")
                    try:
                        next_path = next(path_iter)
                    except StopIteration:
                        continue
                    pending.add(pool.submit(_read_reference_dose, next_path))
    if ref_sum is None:
        raise ValueError("No reference dose files were selected")
    return ref_sum


# ---------------------------------------------------------------------------
# Display utilities
# ---------------------------------------------------------------------------

def _robust_positive_max(array: np.ndarray, percentile: float) -> float:
    positive = array[array > 0.0]
    if positive.size == 0:
        return 1.0
    value = float(np.percentile(positive, percentile))
    return value if value > 0.0 else float(positive.max())


def _normalize_for_display(array: np.ndarray, percentile: float) -> np.ndarray:
    vmax = _robust_positive_max(array, percentile)
    return np.clip(array / vmax, 0.0, 1.0).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _masked_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    mask_fraction: float,
) -> dict[str, float]:
    ref_max = float(reference.max())
    if mask_fraction <= 0.0:
        mask = (reference != 0.0) | (prediction != 0.0)
    else:
        mask = reference > (mask_fraction * ref_max)
    if not np.any(mask):
        mask = np.ones_like(reference, dtype=bool)
    diff = prediction - reference
    mae = float(np.mean(np.abs(diff[mask])))
    rmse = float(np.sqrt(np.mean(diff[mask] ** 2)))
    high_dose_mask = reference > (0.10 * ref_max)
    mae_high10 = (
        float(np.mean(np.abs(diff[high_dose_mask])))
        if np.any(high_dose_mask)
        else 0.0
    )
    return {
        "mae": mae,
        "rmse": rmse,
        "mae_pct_max": (100.0 * mae / ref_max) if ref_max > 0.0 else 0.0,
        "mae_high10_pct_max": (100.0 * mae_high10 / ref_max) if ref_max > 0.0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Shared plotting — total comparison (used by both plot and evaluate scripts)
# ---------------------------------------------------------------------------

def _plot_total_comparison(
    patient_id: str,
    ct: np.ndarray,
    ref_total: np.ndarray,
    pred_total: np.ndarray,
    scale: float,
    mask_fraction: float,
    display_percentile: float,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt  # lazy: backend must be configured by the calling script

    pred_display = pred_total

    diff = pred_display - ref_total
    if mask_fraction <= 0.0:
        mask = (ref_total != 0.0) | (pred_display != 0.0)
        mask_label = "nonzero support"
    else:
        mask = ref_total > (mask_fraction * float(ref_total.max()))
        mask_label = f"mask>{100.0 * mask_fraction:.0f}% max"
    if not np.any(mask):
        mask = np.ones_like(ref_total, dtype=bool)
        mask_label = "full volume"
    mae_val = float(np.mean(np.abs(diff[mask])))
    rmse_val = float(np.sqrt(np.mean(diff[mask] ** 2)))
    nonzero_mask = (ref_total != 0.0) | (pred_display != 0.0)
    mae_nonzero = float(np.mean(np.abs(diff[nonzero_mask]))) if np.any(nonzero_mask) else 0.0
    ref_max = float(ref_total.max())
    high_dose_mask = ref_total > (0.10 * ref_max)
    mae_high10 = (
        float(np.mean(np.abs(diff[high_dose_mask])))
        if np.any(high_dose_mask)
        else 0.0
    )
    mae_pct_max = (100.0 * mae_val / ref_max) if ref_max > 0.0 else 0.0
    mae_nonzero_pct_max = (100.0 * mae_nonzero / ref_max) if ref_max > 0.0 else 0.0
    mae_high10_pct_max = (100.0 * mae_high10 / ref_max) if ref_max > 0.0 else 0.0

    max_pos = np.unravel_index(int(np.argmax(ref_total)), ref_total.shape)
    z_mid, y_mid, x_mid = int(max_pos[0]), int(max_pos[1]), int(max_pos[2])

    views = [
        ("Transversal (y,x)", pred_display[z_mid], ref_total[z_mid], ct[z_mid]),
        ("Coronal (z,x)", pred_display[:, y_mid], ref_total[:, y_mid], ct[:, y_mid]),
        ("Sagittal (z,y)", pred_display[..., x_mid], ref_total[..., x_mid], ct[..., x_mid]),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12.0))
    dose_vmax = max(float(ref_total.max()), float(pred_display.max()), 1e-8)

    for col, (title, pred_view, ref_view, ct_view) in enumerate(views):
        diff_view = pred_view - ref_view

        ax = axes[0, col]
        ax.imshow(ct_view, cmap="gray", vmin=ct.min(), vmax=ct.max())
        im = ax.imshow(
            pred_view, cmap="inferno", vmin=0.0, vmax=dose_vmax,
            interpolation="nearest", aspect="auto", alpha=0.4,
        )
        ax.set_title(f"Predicted {title}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        ax = axes[1, col]
        ax.imshow(ct_view, cmap="gray", vmin=ct.min(), vmax=ct.max())
        im = ax.imshow(
            ref_view, cmap="inferno", vmin=0.0, vmax=dose_vmax,
            interpolation="nearest", aspect="auto", alpha=0.4,
        )
        ax.set_title(f"Reference {title}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        diff_vmax = max(float(np.abs(diff_view).max()), 1e-8)
        ax = axes[2, col]
        ax.imshow(ct_view, cmap="gray", vmin=ct.min(), vmax=ct.max())
        im = ax.imshow(
            diff_view, cmap="bwr",
            vmin=-diff_vmax, vmax=diff_vmax,
            interpolation="nearest", aspect="auto", alpha=0.4,
        )
        ax.set_title(f"Difference {title}\nred=pred, blue=ref")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(
        f"DoseRAD {patient_id} Proton Total Dose Comparison\n"
        f"scale={scale:.6g}, "
        f"MAE={mae_val:.4f} Gy ({mae_pct_max:.2f}% max), "
        f"MAE_ref>10%={mae_high10_pct_max:.2f}% max, "
        f"RMSE={rmse_val:.4f} Gy, "
        f"MAE_nonzero={mae_nonzero:.4f} Gy ({mae_nonzero_pct_max:.2f}% max), "
        f"{mask_label}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
