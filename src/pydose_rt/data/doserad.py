from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import SimpleITK as sitk
import torch

from pydose_rt.data.beam import Beam, BeamSequence
from pydose_rt.data.ion_beam import IonSpotBeam, IonSpotBeamSequence
from pydose_rt.data.patient import Patient


DIR_IMAGE = "image"
DIR_DOSE = "dose"
BEAM_PARAMS_FILENAME = "beam_parameters.json"

_IMAGE_FILENAMES: dict[str, tuple[str, ...]] = {
    "ct": ("ct.mha", "ct.mhd"),
    "mri": ("mri.mha", "mr.mha", "mri.mhd", "mr.mhd"),
}

_PHOTON_DEFAULT_JAW_X_MM = (-200.0, 200.0)
_PHOTON_DEFAULT_JAW_Y_MM = (-200.0, 200.0)
_PHOTON_BODY_THRESHOLD_HU = -1024.0 + 1e-3
_PROTON_DEFAULT_FIELD_MARGIN_MM = 20.0
_PROTON_DEFAULT_MIN_FIELD_SIZE_MM = 129
_PROTON_DEFAULT_PARTICLES_PER_BEAMLET = 1_000_000.0


@dataclass(frozen=True)
class DoseRADPhotonSampleRef:
    split: str
    patient_id: str
    beam_index: int
    control_point_index: int
    beam_id: int
    control_point_id: int

    @property
    def dose_filename(self) -> str:
        return f"Dose_B{self.beam_id}_CP{self.control_point_id:03d}.mha"


@dataclass(frozen=True)
class DoseRADProtonSampleRef:
    split: str
    patient_id: str
    beam_index: int
    ray_index: int
    beamlet_index: int
    beam_id: int
    ray_id: int
    beamlet_id: int

    @property
    def dose_filename(self) -> str:
        return f"Dose_B{self.beam_id}_R{self.ray_id}_L{self.beamlet_id}.mha"


def _normalize_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _resolve_split_dir(dataset_root: str | Path, modality: Literal["photon", "proton"], split: str) -> Path:
    root = _normalize_path(dataset_root)
    candidates = (root / modality / split, root / split)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find DoseRAD split directory for modality='{modality}', split='{split}' under {root}"
    )


def _resolve_case_dir(dataset_root: str | Path, modality: Literal["photon", "proton"], split: str, patient_id: str) -> Path:
    case_dir = _resolve_split_dir(dataset_root, modality, split) / patient_id
    if not case_dir.is_dir():
        raise FileNotFoundError(f"DoseRAD case directory does not exist: {case_dir}")
    return case_dir


def _resolve_beam_parameters_path(dataset_root: str | Path) -> Path:
    root = _normalize_path(dataset_root)
    candidates = (
        root / BEAM_PARAMS_FILENAME,
        root.parent / BEAM_PARAMS_FILENAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find {BEAM_PARAMS_FILENAME} under {root} or {root.parent}")


@lru_cache(maxsize=None)
def _load_json_cached(path_str: str) -> Any:
    with Path(path_str).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_doserad_beam_parameters(dataset_root: str | Path) -> dict[str, Any]:
    path = _resolve_beam_parameters_path(dataset_root)
    data = _load_json_cached(str(path))
    if not isinstance(data, dict):
        raise ValueError(f"DoseRAD beam parameters must be a JSON object: {path}")
    return data


def load_doserad_plan(
    dataset_root: str | Path,
    modality: Literal["photon", "proton"],
    split: str,
    patient_id: str,
) -> dict[str, Any]:
    case_dir = _resolve_case_dir(dataset_root, modality, split, patient_id)
    plan_path = case_dir / f"{patient_id}.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"DoseRAD plan JSON does not exist: {plan_path}")

    data = _load_json_cached(str(plan_path))
    if not isinstance(data, dict):
        raise ValueError(f"DoseRAD plan JSON must be a JSON object: {plan_path}")
    return data


def _list_patient_ids(dataset_root: str | Path, modality: Literal["photon", "proton"], split: str) -> list[str]:
    split_dir = _resolve_split_dir(dataset_root, modality, split)
    return sorted(path.name for path in split_dir.iterdir() if path.is_dir())


def iter_doserad_photon_samples(
    dataset_root: str | Path,
    split: str = "train",
    patient_ids: Sequence[str] | None = None,
) -> list[DoseRADPhotonSampleRef]:
    if patient_ids is None:
        patient_ids = _list_patient_ids(dataset_root, "photon", split)

    refs: list[DoseRADPhotonSampleRef] = []
    for patient_id in patient_ids:
        plan = load_doserad_plan(dataset_root, "photon", split, patient_id)
        for beam_index, beam_json in enumerate(plan["beams"]):
            beam_id = int(beam_json.get("beam_idx", beam_index))
            for cp_index, cp_json in enumerate(beam_json["control_points"]):
                cp_id = int(cp_json.get("cp_idx", cp_index))
                refs.append(
                    DoseRADPhotonSampleRef(
                        split=split,
                        patient_id=patient_id,
                        beam_index=beam_index,
                        control_point_index=cp_index,
                        beam_id=beam_id,
                        control_point_id=cp_id,
                    )
                )
    return refs


def iter_doserad_proton_samples(
    dataset_root: str | Path,
    split: str = "train",
    patient_ids: Sequence[str] | None = None,
) -> list[DoseRADProtonSampleRef]:
    if patient_ids is None:
        patient_ids = _list_patient_ids(dataset_root, "proton", split)

    refs: list[DoseRADProtonSampleRef] = []
    for patient_id in patient_ids:
        plan = load_doserad_plan(dataset_root, "proton", split, patient_id)
        for beam_index, beam_json in enumerate(plan["beams"]):
            beam_id = int(beam_json.get("beam_idx", beam_index))
            for ray_index, ray_json in enumerate(beam_json["rays"]):
                ray_id = int(ray_json.get("ray_idx", ray_index))
                for beamlet_index, beamlet_json in enumerate(ray_json["beamlets"]):
                    beamlet_id = int(beamlet_json.get("beamlet_idx", beamlet_index))
                    refs.append(
                        DoseRADProtonSampleRef(
                            split=split,
                            patient_id=patient_id,
                            beam_index=beam_index,
                            ray_index=ray_index,
                            beamlet_index=beamlet_index,
                            beam_id=beam_id,
                            ray_id=ray_id,
                            beamlet_id=beamlet_id,
                        )
                    )
                refs[-len(ray_json["beamlets"]):]
    return refs


def _resolve_image_path(case_dir: Path, image_kind: Literal["ct", "mri"]) -> Path:
    candidates = {name.lower() for name in _IMAGE_FILENAMES[image_kind]}
    search_dirs = [case_dir / DIR_IMAGE, case_dir]
    available: list[str] = []
    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for path in sorted(search_dir.iterdir()):
            if path.is_file():
                available.append(str(path.relative_to(case_dir)))
                if path.name.lower() in candidates:
                    return path

    raise FileNotFoundError(
        f"Could not find a DoseRAD {image_kind} image in {case_dir}. Available files: {sorted(available)}"
    )


def _xyz_to_zyx(coords_xyz: Sequence[float]) -> np.ndarray:
    coords = np.asarray(coords_xyz, dtype=np.float32)
    if coords.shape != (3,):
        raise ValueError(f"Expected a 3D coordinate, got shape {coords.shape}")
    return coords[[2, 1, 0]]


def _origin_zyx(image: sitk.Image) -> np.ndarray:
    return _xyz_to_zyx(tuple(float(value) for value in image.GetOrigin()))


def _resolution_zyx(image: sitk.Image) -> tuple[float, float, float]:
    spacing = tuple(float(value) for value in image.GetSpacing())
    return (spacing[2], spacing[1], spacing[0])


def _same_image_metadata(reference: sitk.Image, other: sitk.Image) -> bool:
    return (
        reference.GetSize() == other.GetSize()
        and np.allclose(reference.GetSpacing(), other.GetSpacing())
        and np.allclose(reference.GetOrigin(), other.GetOrigin())
        and np.allclose(reference.GetDirection(), other.GetDirection())
    )


def _load_patient_image(
    image_path: Path,
    dose_path: Path | None,
    image_kind: Literal["ct", "mri"],
    add_body_mask: bool,
) -> tuple[Patient, np.ndarray]:
    image = sitk.ReadImage(str(image_path))
    image_array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    image_tensor = torch.from_numpy(image_array)

    structures: dict[str, torch.Tensor] = {}
    if add_body_mask:
        if image_kind == "ct":
            structures["body"] = image_tensor > _PHOTON_BODY_THRESHOLD_HU
        else:
            structures["body"] = image_tensor != 0.0

    dose_tensor = None
    if dose_path is not None:
        dose_image = sitk.ReadImage(str(dose_path))
        if not _same_image_metadata(image, dose_image):
            dose_image = sitk.Resample(
                dose_image,
                image,
                sitk.Transform(),
                sitk.sitkLinear,
                0.0,
                sitk.sitkFloat32,
            )
        else:
            dose_image = sitk.Cast(dose_image, sitk.sitkFloat32)
        dose_array = sitk.GetArrayFromImage(dose_image).astype(np.float32, copy=False)
        dose_tensor = torch.from_numpy(dose_array)

    patient_kwargs = {
        "structures": structures,
        "dose": dose_tensor,
        "resolution": _resolution_zyx(image),
        "number_of_fractions": 1,
    }
    if image_kind == "ct":
        patient_kwargs["ct_tensor"] = image_tensor
    else:
        # Patient has no MRI-specific slot, so MRI intensities live in attenuation_tensor.
        patient_kwargs["attenuation_tensor"] = image_tensor

    return Patient(**patient_kwargs), _origin_zyx(image)


def load_doserad_patient(
    dataset_root: str | Path,
    modality: Literal["photon", "proton"],
    split: str,
    patient_id: str,
    image_kind: Literal["ct", "mri"] = "ct",
    add_body_mask: bool = True,
) -> Patient:
    case_dir = _resolve_case_dir(dataset_root, modality, split, patient_id)
    image_path = _resolve_image_path(case_dir, image_kind)
    patient, _ = _load_patient_image(image_path=image_path, dose_path=None, image_kind=image_kind, add_body_mask=add_body_mask)
    return patient


def load_doserad_photon_sample(
    dataset_root: str | Path,
    split: str,
    patient_id: str,
    beam_index: int,
    control_point_index: int,
    image_kind: Literal["ct", "mri"] = "ct",
    add_body_mask: bool = True,
    apply_leaf_convention_fix: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> tuple[Patient, BeamSequence]:
    case_dir = _resolve_case_dir(dataset_root, "photon", split, patient_id)
    plan = load_doserad_plan(dataset_root, "photon", split, patient_id)
    beam_params = load_doserad_beam_parameters(dataset_root)

    beam_json = plan["beams"][beam_index]
    cp_json = beam_json["control_points"][control_point_index]
    beam_id = int(beam_json.get("beam_idx", beam_index))
    cp_id = int(cp_json.get("cp_idx", control_point_index))

    image_path = _resolve_image_path(case_dir, image_kind)
    dose_path = case_dir / DIR_DOSE / f"Dose_B{beam_id}_CP{cp_id:03d}.mha"
    patient, origin = _load_patient_image(
        image_path=image_path,
        dose_path=dose_path,
        image_kind=image_kind,
        add_body_mask=add_body_mask,
    )

    photon_params = beam_params.get("photon", {})
    jaw_x_mm = tuple(float(value) for value in photon_params.get("jaw_x_mm", _PHOTON_DEFAULT_JAW_X_MM))
    jaw_y_mm = tuple(float(value) for value in photon_params.get("jaw_y_mm", _PHOTON_DEFAULT_JAW_Y_MM))
    field_size = (
        int(round(jaw_y_mm[1] - jaw_y_mm[0])),
        int(round(jaw_x_mm[1] - jaw_x_mm[0])),
    )

    mlc_left = np.asarray(cp_json["mlc_left_int_mm"], dtype=np.float32)
    mlc_right = np.asarray(cp_json["mlc_right_int_mm"], dtype=np.float32)
    if mlc_left.shape != mlc_right.shape:
        raise ValueError("DoseRAD photon control point must have matching left/right MLC arrays")

    if apply_leaf_convention_fix:
        leaf_left = -np.flip(mlc_right)
        leaf_right = -np.flip(mlc_left)
    else:
        leaf_left = mlc_left
        leaf_right = mlc_right

    leaf_positions = torch.tensor(
        np.stack((leaf_left, leaf_right), axis=1),
        device=device,
        dtype=dtype,
    )
    jaw_positions = torch.tensor(jaw_y_mm, device=device, dtype=dtype)
    mu = torch.tensor(1.0, device=device, dtype=dtype)
    if requires_grad:
        leaf_positions.requires_grad_(True)
        jaw_positions.requires_grad_(True)
        mu.requires_grad_(True)

    iso_center = tuple((_xyz_to_zyx(beam_json["iso_center"]) - origin).tolist())
    beam = Beam(
        gantry_angle=math.radians(float(cp_json["gantry_angle"])),
        collimator_angle=0.0,
        mu=mu,
        leaf_positions=leaf_positions,
        jaw_positions=jaw_positions,
        field_size=field_size,
        iso_center=iso_center,
        sid=float(photon_params.get("SAD_mm", 1000.0)),
        ssd=None,
    )
    return patient, BeamSequence.from_beams([beam])


def _extract_proton_sigma_xy_mm(entry: dict[str, Any]) -> tuple[float, float]:
    sigma_xy_pairs = (
        ("sigma_x_mm", "sigma_y_mm"),
        ("sigma_spot_x_mm", "sigma_spot_y_mm"),
        ("sigma_spot_mm_x", "sigma_spot_mm_y"),
    )
    for key_x, key_y in sigma_xy_pairs:
        if key_x in entry and key_y in entry:
            return float(entry[key_x]), float(entry[key_y])

    if "sigma_spot_mm" in entry:
        sigma_mm = float(entry["sigma_spot_mm"])
        return sigma_mm, sigma_mm

    raise ValueError(
        "DoseRAD proton energy table entry must contain either "
        "'sigma_spot_mm' or an anisotropic sigma pair such as "
        "('sigma_x_mm', 'sigma_y_mm')"
    )


def _match_proton_sigma_xy_mm(
    beam_parameters: dict[str, Any],
    energy_mev: float,
    tolerance_mev: float = 1e-3,
) -> tuple[float, float]:
    proton_params = beam_parameters.get("proton", {})
    energy_table = proton_params.get("energy_table")
    if not energy_table:
        raise ValueError("DoseRAD beam parameters are missing proton.energy_table")

    energies = np.asarray([float(entry["energy_mev"]) for entry in energy_table], dtype=np.float32)
    idx = int(np.argmin(np.abs(energies - float(energy_mev))))
    matched_energy = float(energies[idx])
    if abs(matched_energy - float(energy_mev)) > tolerance_mev:
        raise ValueError(
            f"Could not match DoseRAD proton energy {energy_mev:.6f} MeV within {tolerance_mev} MeV"
        )
    return _extract_proton_sigma_xy_mm(energy_table[idx])


def load_doserad_proton_sample(
    dataset_root: str | Path,
    split: str,
    patient_id: str,
    beam_index: int,
    ray_index: int,
    beamlet_index: int,
    image_kind: Literal["ct", "mri"] = "ct",
    add_body_mask: bool = True,
    field_margin_mm: float = _PROTON_DEFAULT_FIELD_MARGIN_MM,
    min_field_size_mm: int = _PROTON_DEFAULT_MIN_FIELD_SIZE_MM,
    particles_per_beamlet: float = _PROTON_DEFAULT_PARTICLES_PER_BEAMLET,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> tuple[Patient, IonSpotBeamSequence]:
    case_dir = _resolve_case_dir(dataset_root, "proton", split, patient_id)
    plan = load_doserad_plan(dataset_root, "proton", split, patient_id)
    beam_parameters = load_doserad_beam_parameters(dataset_root)

    beam_json = plan["beams"][beam_index]
    ray_json = beam_json["rays"][ray_index]
    beamlet_json = ray_json["beamlets"][beamlet_index]

    beam_id = int(beam_json.get("beam_idx", beam_index))
    ray_id = int(ray_json.get("ray_idx", ray_index))
    beamlet_id = int(beamlet_json.get("beamlet_idx", beamlet_index))

    image_path = _resolve_image_path(case_dir, image_kind)
    dose_path = case_dir / DIR_DOSE / f"Dose_B{beam_id}_R{ray_id}_L{beamlet_id}.mha"
    patient, origin = _load_patient_image(
        image_path=image_path,
        dose_path=dose_path,
        image_kind=image_kind,
        add_body_mask=add_body_mask,
    )

    sigma_x_mm, sigma_y_mm = _match_proton_sigma_xy_mm(beam_parameters, float(beamlet_json["energy"]))
    field_extent = max(
        int(math.ceil(2.0 * field_margin_mm)) + 1,
        int(min_field_size_mm),
    )
    if field_extent % 2 == 0:
        field_extent += 1

    target_iso_center = tuple((_xyz_to_zyx(ray_json["ray_target"]) - origin).tolist())
    sad_mm = float(plan.get("SAD", 10_000.0))

    beam = IonSpotBeam.create(
        gantry_angle_deg=float(beam_json["gantry_angle"]),
        spot_positions_mm=torch.zeros((1, 2), device=device, dtype=dtype),
        spot_weights=torch.full((1,), float(particles_per_beamlet), device=device, dtype=dtype),
        spot_layer_index=torch.zeros(1, device=device, dtype=torch.long),
        layer_energies_mev=torch.tensor([float(beamlet_json["energy"])], device=device, dtype=dtype),
        layer_sigmas_mm=torch.tensor([[sigma_x_mm, sigma_y_mm]], device=device, dtype=dtype),
        field_size=(field_extent, field_extent),
        iso_center=target_iso_center,
        sad_mm=sad_mm,
        requires_grad=requires_grad,
    )
    return patient, IonSpotBeamSequence.from_beams([beam])


def load_doserad_sample(
    dataset_root: str | Path,
    sample_ref: DoseRADPhotonSampleRef | DoseRADProtonSampleRef,
    image_kind: Literal["ct", "mri"] = "ct",
    add_body_mask: bool = True,
    particles_per_beamlet: float = _PROTON_DEFAULT_PARTICLES_PER_BEAMLET,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> tuple[Patient, BeamSequence | IonSpotBeamSequence]:
    if isinstance(sample_ref, DoseRADPhotonSampleRef):
        return load_doserad_photon_sample(
            dataset_root=dataset_root,
            split=sample_ref.split,
            patient_id=sample_ref.patient_id,
            beam_index=sample_ref.beam_index,
            control_point_index=sample_ref.control_point_index,
            image_kind=image_kind,
            add_body_mask=add_body_mask,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )

    return load_doserad_proton_sample(
        dataset_root=dataset_root,
        split=sample_ref.split,
        patient_id=sample_ref.patient_id,
        beam_index=sample_ref.beam_index,
        ray_index=sample_ref.ray_index,
        beamlet_index=sample_ref.beamlet_index,
        image_kind=image_kind,
        add_body_mask=add_body_mask,
        particles_per_beamlet=particles_per_beamlet,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
