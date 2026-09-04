import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from pydose_rt.data.doserad import (
    iter_doserad_photon_samples,
    iter_doserad_proton_samples,
    load_doserad_patient,
    load_doserad_photon_sample,
    load_doserad_proton_sample,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_image(
    path: Path,
    array_zyx: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
) -> None:
    image = sitk.GetImageFromArray(array_zyx.astype(np.float32, copy=False))
    image.SetSpacing(spacing_xyz)
    image.SetOrigin(origin_xyz)
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path), useCompression=True)


def _beam_parameters_payload() -> dict:
    return {
        "photon": {
            "SAD_mm": 900.0,
            "jaw_x_mm": [-180.0, 180.0],
            "jaw_y_mm": [-150.0, 150.0],
        },
        "proton": {
            "energy_table": [
                {"energy_mev": 100.0, "sigma_energy_mev": 1.0, "sigma_spot_mm": 6.5},
                {"energy_mev": 110.0, "sigma_energy_mev": 1.0, "sigma_spot_mm": 5.5},
            ]
        },
    }


def test_load_doserad_photon_sample_parses_geometry_and_dose(default_device, default_dtype, tmp_path):
    root = tmp_path / "doserad"
    patient_id = "P001"
    case_dir = root / "photon" / "train" / patient_id

    image = np.array(
        [
            [[-1024.0, -824.0, 1.0, 2.0], [3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0]],
            [[11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0], [19.0, 20.0, 21.0, 22.0]],
        ],
        dtype=np.float32,
    )
    dose = np.full_like(image, 2.5, dtype=np.float32)

    _write_json(root / "beam_parameters.json", _beam_parameters_payload())
    _write_json(
        case_dir / f"{patient_id}.json",
        {
            "beams": [
                {
                    "beam_idx": 7,
                    "iso_center": [14.0, 26.0, 38.0],
                    "num_mlc_leaf_pairs": 2,
                    "control_points": [
                        {
                            "cp_idx": 3,
                            "gantry_angle": 90.0,
                            "mlc_left_int_mm": [1.0, 2.0],
                            "mlc_right_int_mm": [3.0, 4.0],
                        }
                    ],
                }
            ]
        },
    )
    _write_image(case_dir / "image" / "ct.mha", image, spacing_xyz=(2.0, 3.0, 4.0), origin_xyz=(10.0, 20.0, 30.0))
    _write_image(case_dir / "dose" / "Dose_B7_CP003.mha", dose, spacing_xyz=(2.0, 3.0, 4.0), origin_xyz=(10.0, 20.0, 30.0))

    patient, beam_sequence = load_doserad_photon_sample(
        dataset_root=root,
        split="train",
        patient_id=patient_id,
        beam_index=0,
        control_point_index=0,
        device=default_device,
        dtype=default_dtype,
    )

    assert patient.ct_image.shape == (2, 3, 4)
    assert patient.resolution == (4.0, 3.0, 2.0)
    assert torch.allclose(patient.dose, torch.full((2, 3, 4), 2.5))
    assert not patient.structures["body"][0, 0, 0]
    assert patient.structures["body"][0, 0, 1]

    beam = beam_sequence[0]
    assert beam.iso_center == (8.0, 6.0, 4.0)
    assert beam.sid == 900.0
    assert beam.field_size == (300, 360)
    assert torch.allclose(
        beam.leaf_positions,
        torch.tensor([[-4.0, -2.0], [-3.0, -1.0]], device=default_device, dtype=default_dtype),
    )
    assert torch.allclose(
        beam.jaw_positions,
        torch.tensor([-150.0, 150.0], device=default_device, dtype=default_dtype),
    )


def test_iter_doserad_photon_samples_uses_json_ids(tmp_path):
    root = tmp_path / "doserad"
    patient_id = "P002"
    case_dir = root / "photon" / "train" / patient_id

    _write_json(
        case_dir / f"{patient_id}.json",
        {
            "beams": [
                {
                    "beam_idx": 5,
                    "iso_center": [0.0, 0.0, 0.0],
                    "num_mlc_leaf_pairs": 2,
                    "control_points": [
                        {"cp_idx": 10, "gantry_angle": 0.0, "mlc_left_int_mm": [0.0, 0.0], "mlc_right_int_mm": [1.0, 1.0]},
                        {"cp_idx": 12, "gantry_angle": 2.0, "mlc_left_int_mm": [0.0, 0.0], "mlc_right_int_mm": [1.0, 1.0]},
                    ],
                }
            ]
        },
    )

    refs = iter_doserad_photon_samples(root, split="train")

    assert len(refs) == 2
    assert refs[0].beam_index == 0
    assert refs[0].beam_id == 5
    assert refs[0].control_point_index == 0
    assert refs[0].control_point_id == 10
    assert refs[0].dose_filename == "Dose_B5_CP010.mha"
    assert refs[1].dose_filename == "Dose_B5_CP012.mha"


def test_load_doserad_proton_sample_matches_energy_table_and_ray_target(default_device, default_dtype, tmp_path):
    root = tmp_path / "doserad"
    patient_id = "Q001"
    case_dir = root / "proton" / "train" / patient_id

    image = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    dose = np.full_like(image, 1.25, dtype=np.float32)

    _write_json(root / "beam_parameters.json", _beam_parameters_payload())
    _write_json(
        case_dir / f"{patient_id}.json",
        {
            "SAD": 1250.0,
            "iso_center": [100.0, 200.0, 300.0],
            "beams": [
                {
                    "beam_idx": 1,
                    "gantry_angle": 30.0,
                    "rays": [
                        {
                            "ray_idx": 2,
                            "ray_source": [15.0, -1224.0, 37.0],
                            "ray_target": [15.0, 26.0, 37.0],
                            "beamlets": [
                                {"beamlet_idx": 4, "energy": 100.0},
                                {"beamlet_idx": 5, "energy": 110.0},
                            ],
                        }
                    ],
                }
            ],
        },
    )
    _write_image(case_dir / "image" / "ct.mha", image, spacing_xyz=(1.0, 1.0, 3.0), origin_xyz=(5.0, 6.0, 7.0))
    _write_image(case_dir / "dose" / "Dose_B1_R2_L4.mha", dose, spacing_xyz=(1.0, 1.0, 3.0), origin_xyz=(5.0, 6.0, 7.0))

    patient, beam_sequence = load_doserad_proton_sample(
        dataset_root=root,
        split="train",
        patient_id=patient_id,
        beam_index=0,
        ray_index=0,
        beamlet_index=0,
        device=default_device,
        dtype=default_dtype,
    )

    assert patient.ct_image.shape == (2, 2, 2)
    assert patient.resolution == (3.0, 1.0, 1.0)
    assert torch.allclose(patient.dose, torch.full((2, 2, 2), 1.25))

    beam = beam_sequence[0]
    assert beam.iso_center == (30.0, 20.0, 10.0)
    assert beam.field_size == (129, 129)
    assert torch.allclose(
        beam.spot_positions_mm,
        torch.zeros((1, 2), device=default_device, dtype=default_dtype),
    )
    assert torch.allclose(
        beam.layer_energies_mev,
        torch.tensor([100.0], device=default_device, dtype=default_dtype),
    )
    assert torch.allclose(
        beam.layer_sigmas_mm,
        torch.tensor([[6.5, 6.5]], device=default_device, dtype=default_dtype),
    )
    assert torch.allclose(
        beam.spot_weights,
        torch.tensor([1_000_000.0], device=default_device, dtype=default_dtype),
    )


def test_load_doserad_proton_sample_uses_anisotropic_sigmas_when_present(default_device, default_dtype, tmp_path):
    root = tmp_path / "doserad"
    patient_id = "Q004"
    case_dir = root / "proton" / "train" / patient_id

    image = np.zeros((2, 2, 2), dtype=np.float32)
    dose = np.ones_like(image, dtype=np.float32)

    beam_parameters = _beam_parameters_payload()
    beam_parameters["proton"]["energy_table"] = [
        {"energy_mev": 100.0, "sigma_energy_mev": 1.0, "sigma_x_mm": 6.5, "sigma_y_mm": 4.5},
        {"energy_mev": 110.0, "sigma_energy_mev": 1.0, "sigma_spot_x_mm": 5.5, "sigma_spot_y_mm": 3.5},
    ]
    _write_json(root / "beam_parameters.json", beam_parameters)
    _write_json(
        case_dir / f"{patient_id}.json",
        {
            "beams": [
                {
                    "beam_idx": 1,
                    "gantry_angle": 0.0,
                    "rays": [
                        {
                            "ray_idx": 0,
                            "ray_source": [0.0, -1000.0, 0.0],
                            "ray_target": [0.0, 0.0, 0.0],
                            "beamlets": [
                                {"beamlet_idx": 0, "energy": 100.0},
                            ],
                        }
                    ],
                }
            ]
        },
    )
    _write_image(case_dir / "image" / "ct.mha", image, spacing_xyz=(1.0, 1.0, 1.0), origin_xyz=(0.0, 0.0, 0.0))
    _write_image(case_dir / "dose" / "Dose_B1_R0_L0.mha", dose, spacing_xyz=(1.0, 1.0, 1.0), origin_xyz=(0.0, 0.0, 0.0))

    _, beam_sequence = load_doserad_proton_sample(
        dataset_root=root,
        split="train",
        patient_id=patient_id,
        beam_index=0,
        ray_index=0,
        beamlet_index=0,
        device=default_device,
        dtype=default_dtype,
    )

    assert torch.allclose(
        beam_sequence[0].layer_sigmas_mm,
        torch.tensor([[6.5, 4.5]], device=default_device, dtype=default_dtype),
    )


def test_load_doserad_proton_sample_allows_custom_particle_count(default_device, default_dtype, tmp_path):
    root = tmp_path / "doserad"
    patient_id = "Q003"
    case_dir = root / "proton" / "train" / patient_id

    image = np.zeros((2, 2, 2), dtype=np.float32)
    dose = np.ones_like(image, dtype=np.float32)

    _write_json(root / "beam_parameters.json", _beam_parameters_payload())
    _write_json(
        case_dir / f"{patient_id}.json",
        {
            "beams": [
                {
                    "beam_idx": 1,
                    "gantry_angle": 0.0,
                    "rays": [
                        {
                            "ray_idx": 0,
                            "ray_source": [0.0, -1000.0, 0.0],
                            "ray_target": [0.0, 0.0, 0.0],
                            "beamlets": [
                                {"beamlet_idx": 0, "energy": 100.0},
                            ],
                        }
                    ],
                }
            ]
        },
    )
    _write_image(case_dir / "image" / "ct.mha", image, spacing_xyz=(1.0, 1.0, 1.0), origin_xyz=(0.0, 0.0, 0.0))
    _write_image(case_dir / "dose" / "Dose_B1_R0_L0.mha", dose, spacing_xyz=(1.0, 1.0, 1.0), origin_xyz=(0.0, 0.0, 0.0))

    _, beam_sequence = load_doserad_proton_sample(
        dataset_root=root,
        split="train",
        patient_id=patient_id,
        beam_index=0,
        ray_index=0,
        beamlet_index=0,
        particles_per_beamlet=42.0,
        device=default_device,
        dtype=default_dtype,
    )

    assert torch.allclose(
        beam_sequence[0].spot_weights,
        torch.tensor([42.0], device=default_device, dtype=default_dtype),
    )


def test_iter_doserad_proton_samples_uses_json_ids(tmp_path):
    root = tmp_path / "doserad"
    patient_id = "Q002"
    case_dir = root / "proton" / "train" / patient_id

    _write_json(
        case_dir / f"{patient_id}.json",
        {
            "SAD": 1000.0,
            "iso_center": [0.0, 0.0, 0.0],
            "beams": [
                {
                    "beam_idx": 8,
                    "gantry_angle": 0.0,
                    "rays": [
                        {
                            "ray_idx": 6,
                            "ray_source": [0.0, -1000.0, 0.0],
                            "ray_target": [0.0, 0.0, 0.0],
                            "beamlets": [
                                {"beamlet_idx": 2, "energy": 100.0},
                                {"beamlet_idx": 4, "energy": 110.0},
                            ],
                        }
                    ],
                }
            ],
        },
    )

    refs = iter_doserad_proton_samples(root, split="train")

    assert len(refs) == 2
    assert refs[0].beam_index == 0
    assert refs[0].beam_id == 8
    assert refs[0].ray_index == 0
    assert refs[0].ray_id == 6
    assert refs[0].beamlet_index == 0
    assert refs[0].beamlet_id == 2
    assert refs[0].dose_filename == "Dose_B8_R6_L2.mha"
    assert refs[1].dose_filename == "Dose_B8_R6_L4.mha"


def test_load_doserad_patient_loads_mri_into_attenuation_tensor(tmp_path):
    root = tmp_path / "doserad"
    patient_id = "M001"
    case_dir = root / "photon" / "train" / patient_id

    mri = np.zeros((1, 2, 3), dtype=np.float32)
    mri[0, 1, 2] = 42.0
    _write_image(case_dir / "image" / "mr.mha", mri, spacing_xyz=(1.5, 1.5, 2.0), origin_xyz=(0.0, 0.0, 0.0))
    _write_json(case_dir / f"{patient_id}.json", {"beams": []})

    patient = load_doserad_patient(root, modality="photon", split="train", patient_id=patient_id, image_kind="mri")

    assert patient._ct_tensor is None
    assert patient.density_image.shape == (1, 2, 3)
    assert float(patient.density_image[0, 1, 2]) == 42.0
