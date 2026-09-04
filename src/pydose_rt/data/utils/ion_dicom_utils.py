import math
import os

import numpy as np
import pydicom
import torch

from pydose_rt.data.ion_beam import IonSpotBeam


_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def infer_ion_field_size(
    spot_positions_mm: np.ndarray,
    margin_mm: float = 20.0,
    min_size_mm: int = 129,
) -> tuple[int, int]:
    if spot_positions_mm.size == 0:
        height = max(int(min_size_mm), 1)
        width = max(int(min_size_mm), 1)
    else:
        max_x = float(np.abs(spot_positions_mm[:, 0]).max())
        max_y = float(np.abs(spot_positions_mm[:, 1]).max())
        width = max(int(math.ceil(2.0 * (max_x + margin_mm))) + 1, int(min_size_mm))
        height = max(int(math.ceil(2.0 * (max_y + margin_mm))) + 1, int(min_size_mm))

    if width % 2 == 0:
        width += 1
    if height % 2 == 0:
        height += 1

    return (height, width)


def parse_ion_plan_dataset(
    ds: pydicom.Dataset,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    field_size: tuple[int, int] | None = None,
    field_margin_mm: float = 20.0,
    min_field_size_mm: int = 129,
    particles_per_meterset: float | None = None,
    requires_grad: bool = False,
) -> dict[str, tuple[list[IonSpotBeam], int]]:
    parameters = {}
    fraction_group = ds.FractionGroupSequence[0]
    number_of_fractions = int(fraction_group.NumberOfFractionsPlanned)

    for beam in ds.IonBeamSequence:
        layer_positions = []
        layer_weights = []
        layer_energies = []
        layer_sigmas = []
        iso_center = None
        gantry_angle = 0.0

        for cp in beam.IonControlPointSequence:
            if hasattr(cp, "GantryAngle"):
                gantry_angle = float(cp.GantryAngle)
            if iso_center is None and hasattr(cp, "IsocenterPosition"):
                iso_center = (
                    float(cp.IsocenterPosition[2]),
                    float(cp.IsocenterPosition[1]),
                    float(cp.IsocenterPosition[0]),
                )

            positions = getattr(cp, "ScanSpotPositionMap", None)
            weights = getattr(cp, "ScanSpotMetersetWeights", None)
            if positions is None or weights is None:
                continue

            flat_positions = np.asarray(positions, dtype=np.float32).reshape(-1)
            positions_xy = flat_positions.reshape(-1, 2)
            weights_np = np.asarray(weights, dtype=np.float32).reshape(-1)
            valid = weights_np > 0.0
            if not np.any(valid):
                continue

            scaled_weights = weights_np[valid]
            if particles_per_meterset is not None:
                scaled_weights = scaled_weights * float(particles_per_meterset)

            spot_size = getattr(cp, "ScanningSpotSize", None)
            if spot_size is None:
                raise ValueError("Ion control point is missing ScanningSpotSize")

            layer_positions.append(positions_xy[valid])
            layer_weights.append(scaled_weights)
            layer_energies.append(float(cp.NominalBeamEnergy))
            spot_size_arr = np.asarray(spot_size, dtype=np.float32).reshape(-1)
            if spot_size_arr.size == 1:
                sigma_x_mm = float(spot_size_arr[0]) * _FWHM_TO_SIGMA
                sigma_y_mm = sigma_x_mm
            else:
                sigma_x_mm = float(spot_size_arr[0]) * _FWHM_TO_SIGMA
                sigma_y_mm = float(spot_size_arr[1]) * _FWHM_TO_SIGMA
            layer_sigmas.append((sigma_x_mm, sigma_y_mm))

        if not layer_positions:
            continue

        spot_positions = np.concatenate(layer_positions, axis=0)
        spot_weights = np.concatenate(layer_weights, axis=0)
        spot_layer_index = np.concatenate(
            [
                np.full(weights.shape[0], layer_idx, dtype=np.int64)
                for layer_idx, weights in enumerate(layer_weights)
            ]
        )

        beam_field_size = field_size
        if beam_field_size is None:
            beam_field_size = infer_ion_field_size(
                spot_positions,
                margin_mm=field_margin_mm,
                min_size_mm=min_field_size_mm,
            )

        ion_beam = IonSpotBeam.create(
            gantry_angle_deg=gantry_angle,
            spot_positions_mm=torch.tensor(spot_positions, device=device, dtype=dtype),
            spot_weights=torch.tensor(spot_weights, device=device, dtype=dtype),
            spot_layer_index=torch.tensor(spot_layer_index, device=device, dtype=torch.long),
            layer_energies_mev=torch.tensor(layer_energies, device=device, dtype=dtype),
            layer_sigmas_mm=torch.tensor(layer_sigmas, device=device, dtype=dtype),
            field_size=beam_field_size,
            iso_center=(0.0, 0.0, 0.0) if iso_center is None else iso_center,
            requires_grad=requires_grad,
        )
        parameters[f"{ds.SOPInstanceUID}_{beam.BeamNumber}"] = ([ion_beam], number_of_fractions)

    return parameters


def fetch_ion_plan_data(
    plan_path: str | os.PathLike[str],
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    field_size: tuple[int, int] | None = None,
    field_margin_mm: float = 20.0,
    min_field_size_mm: int = 129,
    particles_per_meterset: float | None = None,
    requires_grad: bool = False,
) -> dict[str, tuple[list[IonSpotBeam], int]]:
    ds = pydicom.dcmread(plan_path)
    return parse_ion_plan_dataset(
        ds,
        device=device,
        dtype=dtype,
        field_size=field_size,
        field_margin_mm=field_margin_mm,
        min_field_size_mm=min_field_size_mm,
        particles_per_meterset=particles_per_meterset,
        requires_grad=requires_grad,
    )
