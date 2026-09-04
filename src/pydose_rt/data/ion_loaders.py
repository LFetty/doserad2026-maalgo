from pathlib import Path
from typing import List

import numpy as np
import SimpleITK as sitk
import torch

from pydose_rt.data.ion_beam import IonSpotBeamSequence
from pydose_rt.data.patient import Patient
from pydose_rt.data.utils.dicom_utils import load_ct_series, load_dose, load_structures
from pydose_rt.data.utils.ion_dicom_utils import fetch_ion_plan_data


def _as_path_list(paths: List[Path] | Path | None) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, Path):
        return [paths]
    return list(paths)


def load_ion_dicom(
    ct_folder: Path,
    dose_path: List[Path] | Path | None,
    plan_path: List[Path] | Path | None,
    struct_path: Path | None,
    struct_names: List[str] | None = None,
    new_spacing: tuple[float, float, float] = (2.0, 2.0, 2.0),
    crop_volume: bool = True,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
    field_size: tuple[int, int] | None = None,
    field_margin_mm: float = 20.0,
    min_field_size_mm: int = 129,
    particles_per_meterset: float | None = None,
    requires_grad: bool = False,
) -> tuple[Patient, list[IonSpotBeamSequence]]:
    plan_paths = _as_path_list(plan_path)
    dose_paths = _as_path_list(dose_path)
    if not plan_paths:
        raise ValueError("load_ion_dicom requires an RT Ion Plan path")

    plans = fetch_ion_plan_data(
        plan_paths[0],
        device=device,
        dtype=dtype,
        field_size=field_size,
        field_margin_mm=field_margin_mm,
        min_field_size_mm=min_field_size_mm,
        particles_per_meterset=particles_per_meterset,
        requires_grad=requires_grad,
    )
    _, num_fractions = list(plans.values())[0]
    patient, dose_ref, origin = _load_ion_patient_from_images(
        ct_folder=ct_folder,
        dose_paths=dose_paths,
        struct_path=struct_path,
        struct_names=struct_names,
        new_spacing=new_spacing,
        crop_volume=crop_volume,
        num_fractions=num_fractions,
    )

    beam_sequences = []
    for key, (seq, _) in plans.items():
        if dose_ref is not None and dose_ref in plans and dose_ref != key:
            continue

        beam_sequence = IonSpotBeamSequence.from_beams(seq).to(device).to(dtype)
        beam_sequence.iso_center = tuple(np.array(beam_sequence.iso_center) - np.array(origin))
        beam_sequences.append(beam_sequence)

    return patient, beam_sequences


def _load_ion_patient_from_images(
    ct_folder: Path,
    dose_paths: list[Path],
    struct_path: Path | None,
    struct_names: List[str] | None,
    new_spacing: tuple[float, float, float],
    crop_volume: bool,
    num_fractions: int,
) -> tuple[Patient, str | None, list[float]]:
    ct_series, _ = load_ct_series(ct_folder)
    structures = load_structures(ct_series, ct_folder, struct_path, struct_names=struct_names)

    doses = {}
    for path in dose_paths:
        dose, plan_ref = load_dose(path)
        doses[plan_ref] = dose

    new_spacing_sitk = (new_spacing[2], new_spacing[1], new_spacing[0])
    ct_resampled = _resample_image_to_spacing(
        ct_series,
        new_spacing=new_spacing_sitk,
        interpolator=sitk.sitkLinear,
    )
    if crop_volume:
        ct_resampled = _center_crop_axial(ct_resampled, max_size_cm=40.0)

    resampled_structures_torch = {}
    for name, struct_img in structures.items():
        struct_resampled = sitk.Resample(
            struct_img,
            ct_resampled,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            struct_img.GetPixelID(),
        )
        struct_array = sitk.GetArrayFromImage(struct_resampled) > 0
        resampled_structures_torch[name] = torch.from_numpy(struct_array)

    dose_ref = next(iter(doses), None)
    dose_tensor = None
    if dose_ref is not None:
        dose = doses[dose_ref]
        dose_resampled = sitk.Resample(
            dose,
            ct_resampled,
            sitk.Transform(),
            sitk.sitkLinear,
            0.0,
            dose.GetPixelID(),
        )
        dose_array = sitk.GetArrayFromImage(dose_resampled) / float(num_fractions)
        dose_tensor = torch.from_numpy(dose_array)

    ct_array = sitk.GetArrayFromImage(ct_resampled)
    ct_tensor = torch.from_numpy(ct_array)
    origin_xyz = ct_resampled.GetOrigin()
    origin = [origin_xyz[2], origin_xyz[1], origin_xyz[0]]

    patient = Patient(
        ct_tensor=ct_tensor,
        structures=resampled_structures_torch,
        dose=dose_tensor,
        resolution=new_spacing,
        number_of_fractions=num_fractions,
    )
    return patient, dose_ref, origin


def _resample_image_to_spacing(image, new_spacing, interpolator=sitk.sitkLinear):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * (osp / nsp)))
        for osz, osp, nsp in zip(original_size, original_spacing, new_spacing)
    ]

    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        interpolator,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        0.0,
        image.GetPixelID(),
    )


def _center_crop_axial(image, max_size_cm=40.0):
    max_size_mm = max_size_cm * 10.0

    spacing = image.GetSpacing()
    size = image.GetSize()
    origin = image.GetOrigin()

    new_size_x = min(size[0], int(max_size_mm / spacing[0]))
    new_size_y = min(size[1], int(max_size_mm / spacing[1]))
    new_size_z = size[2]

    if new_size_x == size[0] and new_size_y == size[1]:
        return image

    start_x = (size[0] - new_size_x) // 2
    start_y = (size[1] - new_size_y) // 2
    start_z = 0

    new_origin = (
        origin[0] + start_x * spacing[0],
        origin[1] + start_y * spacing[1],
        origin[2],
    )

    cropped = sitk.RegionOfInterest(
        image,
        size=[new_size_x, new_size_y, new_size_z],
        index=[start_x, start_y, start_z],
    )
    cropped.SetOrigin(new_origin)
    return cropped
