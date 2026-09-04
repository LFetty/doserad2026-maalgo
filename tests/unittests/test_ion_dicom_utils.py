import pytest
import torch
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from pydose_rt.data.utils.ion_dicom_utils import parse_ion_plan_dataset


def _make_control_point(
    energy_mev: float,
    spot_size_fwhm_mm: float | tuple[float, float],
    positions_xy_mm: list[tuple[float, float]],
    weights: list[float],
    gantry_angle: float | None = None,
    isocenter_xyz: tuple[float, float, float] | None = None,
) -> Dataset:
    cp = Dataset()
    cp.NominalBeamEnergy = energy_mev
    if isinstance(spot_size_fwhm_mm, tuple):
        cp.ScanningSpotSize = list(spot_size_fwhm_mm)
    else:
        cp.ScanningSpotSize = [spot_size_fwhm_mm, spot_size_fwhm_mm]
    cp.ScanSpotPositionMap = [coord for pos in positions_xy_mm for coord in pos]
    cp.ScanSpotMetersetWeights = weights
    if gantry_angle is not None:
        cp.GantryAngle = gantry_angle
    if isocenter_xyz is not None:
        cp.IsocenterPosition = list(isocenter_xyz)
    return cp


def test_parse_ion_plan_dataset_preserves_bev_spot_coordinates(default_device, default_dtype):
    ds = Dataset()
    ds.SOPInstanceUID = "1.2.3"

    fraction_group = Dataset()
    fraction_group.NumberOfFractionsPlanned = 3
    ds.FractionGroupSequence = Sequence([fraction_group])

    beam = Dataset()
    beam.BeamNumber = 7
    beam.VirtualSourceAxisDistances = [7400.0, 6700.0]
    beam.IonControlPointSequence = Sequence(
        [
            _make_control_point(
                energy_mev=120.0,
                spot_size_fwhm_mm=(11.774100225154747, 16.483740315216644),
                positions_xy_mm=[(-15.0, 0.0), (5.0, 10.0)],
                weights=[1.5, 0.0],
                gantry_angle=90.0,
                isocenter_xyz=(10.0, 20.0, -300.0),
            ),
            _make_control_point(
                energy_mev=125.0,
                spot_size_fwhm_mm=(14.128920270185696, 9.419280180123798),
                positions_xy_mm=[(30.0, -25.0)],
                weights=[0.75],
            ),
        ]
    )
    ds.IonBeamSequence = Sequence([beam])

    plans = parse_ion_plan_dataset(
        ds,
        device=default_device,
        dtype=default_dtype,
        field_margin_mm=0.0,
        min_field_size_mm=1,
        requires_grad=False,
    )

    beam_data, num_fractions = plans["1.2.3_7"]
    parsed_beam = beam_data[0]

    assert num_fractions == 3
    assert parsed_beam.gantry_angle_deg == pytest.approx(90.0)
    assert parsed_beam.iso_center == pytest.approx((-300.0, 20.0, 10.0))
    assert parsed_beam.field_size == (51, 61)
    assert torch.equal(
        parsed_beam.spot_positions_mm,
        torch.tensor([[-15.0, 0.0], [30.0, -25.0]], device=default_device, dtype=default_dtype),
    )
    assert torch.equal(
        parsed_beam.spot_layer_index,
        torch.tensor([0, 1], device=default_device, dtype=torch.long),
    )
    assert torch.allclose(
        parsed_beam.spot_weights,
        torch.tensor([1.5, 0.75], device=default_device, dtype=default_dtype),
    )
    assert torch.allclose(
        parsed_beam.layer_sigmas_mm,
        torch.tensor([[5.0, 7.0], [6.0, 4.0]], device=default_device, dtype=default_dtype),
        atol=1e-5,
    )


def test_parse_ion_plan_dataset_can_scale_dicom_meterset_weights_to_particles(default_device, default_dtype):
    ds = Dataset()
    ds.SOPInstanceUID = "1.2.4"

    fraction_group = Dataset()
    fraction_group.NumberOfFractionsPlanned = 1
    ds.FractionGroupSequence = Sequence([fraction_group])

    beam = Dataset()
    beam.BeamNumber = 1
    beam.IonControlPointSequence = Sequence(
        [
            _make_control_point(
                energy_mev=120.0,
                spot_size_fwhm_mm=10.0,
                positions_xy_mm=[(0.0, 0.0), (5.0, 5.0)],
                weights=[0.5, 1.25],
                gantry_angle=0.0,
                isocenter_xyz=(0.0, 0.0, 0.0),
            )
        ]
    )
    ds.IonBeamSequence = Sequence([beam])

    plans = parse_ion_plan_dataset(
        ds,
        device=default_device,
        dtype=default_dtype,
        particles_per_meterset=2_000_000.0,
        requires_grad=False,
    )

    parsed_beam = plans["1.2.4_1"][0][0]

    assert torch.allclose(
        parsed_beam.spot_weights,
        torch.tensor([1_000_000.0, 2_500_000.0], device=default_device, dtype=default_dtype),
    )
