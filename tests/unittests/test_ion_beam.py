import pytest
import torch

from pydose_rt.data.ion_beam import IonSpotBeam, IonSpotBeamSequence


def test_ion_spot_beam_rejects_invalid_layer_index(default_device, default_dtype):
    positions = torch.tensor([[0.0, 0.0]], device=default_device, dtype=default_dtype)
    weights = torch.ones(1, device=default_device, dtype=default_dtype)
    layer_energies = torch.tensor([100.0], device=default_device, dtype=default_dtype)
    layer_sigmas = torch.tensor([[2.0, 3.0]], device=default_device, dtype=default_dtype)
    layer_index = torch.tensor([1], device=default_device, dtype=torch.long)

    with pytest.raises(ValueError):
        IonSpotBeam(
            gantry_angle=0.0,
            spot_positions_mm=positions,
            spot_weights=weights,
            spot_layer_index=layer_index,
            layer_energies_mev=layer_energies,
            layer_sigmas_mm=layer_sigmas,
        )


def test_ion_spot_beam_sequence_from_beams_pads_spots_and_layers(default_device, default_dtype):
    beam_a = IonSpotBeam.create(
        gantry_angle_deg=0.0,
        spot_positions_mm=torch.tensor([[0.0, 0.0], [4.0, -3.0]], device=default_device, dtype=default_dtype),
        spot_weights=torch.tensor([1.0, 2.0], device=default_device, dtype=default_dtype),
        spot_layer_index=torch.tensor([0, 1], device=default_device, dtype=torch.long),
        layer_energies_mev=torch.tensor([100.0, 120.0], device=default_device, dtype=default_dtype),
        layer_sigmas_mm=torch.tensor([[2.0, 2.5], [3.0, 3.5]], device=default_device, dtype=default_dtype),
        requires_grad=False,
    )
    beam_b = IonSpotBeam.create(
        gantry_angle_deg=90.0,
        spot_positions_mm=torch.tensor([[1.0, 2.0]], device=default_device, dtype=default_dtype),
        spot_weights=torch.tensor([0.5], device=default_device, dtype=default_dtype),
        spot_layer_index=torch.tensor([0], device=default_device, dtype=torch.long),
        layer_energies_mev=torch.tensor([110.0], device=default_device, dtype=default_dtype),
        layer_sigmas_mm=torch.tensor([[1.5, 1.0]], device=default_device, dtype=default_dtype),
        requires_grad=False,
    )

    sequence = IonSpotBeamSequence.from_beams([beam_a, beam_b])

    assert sequence.spot_positions_mm.shape == (2, 2, 2)
    assert sequence.layer_energies_mev.shape == (2, 2)
    assert sequence.layer_sigmas_mm.shape == (2, 2, 2)
    assert torch.equal(sequence.spot_mask[0], torch.tensor([True, True], device=default_device))
    assert torch.equal(sequence.spot_mask[1], torch.tensor([True, False], device=default_device))
    assert torch.equal(sequence.layer_mask[0], torch.tensor([True, True], device=default_device))
    assert torch.equal(sequence.layer_mask[1], torch.tensor([True, False], device=default_device))
    assert torch.allclose(
        sequence.layer_sigmas_mm[0],
        torch.tensor([[2.0, 2.5], [3.0, 3.5]], device=default_device, dtype=default_dtype),
    )
    assert sequence[1].num_spots == 1
    assert sequence[1].num_layers == 1


def test_ion_spot_beam_create_tracks_energy_gradients(default_device, default_dtype):
    beam = IonSpotBeam.create(
        gantry_angle_deg=0.0,
        spot_positions_mm=torch.tensor([[0.0, 0.0]], device=default_device, dtype=default_dtype),
        spot_weights=torch.tensor([1.0], device=default_device, dtype=default_dtype),
        spot_layer_index=torch.tensor([0], device=default_device, dtype=torch.long),
        layer_energies_mev=torch.tensor([120.0], device=default_device, dtype=default_dtype),
        layer_sigmas_mm=torch.tensor([2.0], device=default_device, dtype=default_dtype),
        requires_grad=True,
    )

    assert beam.layer_energies_mev.requires_grad
    assert not beam.layer_sigmas_mm.requires_grad
    assert beam.requires_grad


def test_ion_spot_beam_sequence_preserves_per_beam_iso_centers(default_device, default_dtype):
    beam_a = IonSpotBeam.create(
        gantry_angle_deg=0.0,
        spot_positions_mm=torch.tensor([[0.0, 0.0]], device=default_device, dtype=default_dtype),
        spot_weights=torch.tensor([1.0], device=default_device, dtype=default_dtype),
        spot_layer_index=torch.tensor([0], device=default_device, dtype=torch.long),
        layer_energies_mev=torch.tensor([100.0], device=default_device, dtype=default_dtype),
        layer_sigmas_mm=torch.tensor([[2.0, 2.5]], device=default_device, dtype=default_dtype),
        iso_center=(10.0, 20.0, 30.0),
        requires_grad=False,
    )
    beam_b = IonSpotBeam.create(
        gantry_angle_deg=90.0,
        spot_positions_mm=torch.tensor([[1.0, 2.0]], device=default_device, dtype=default_dtype),
        spot_weights=torch.tensor([0.5], device=default_device, dtype=default_dtype),
        spot_layer_index=torch.tensor([0], device=default_device, dtype=torch.long),
        layer_energies_mev=torch.tensor([110.0], device=default_device, dtype=default_dtype),
        layer_sigmas_mm=torch.tensor([[1.5, 1.0]], device=default_device, dtype=default_dtype),
        iso_center=(40.0, 50.0, 60.0),
        requires_grad=False,
    )

    sequence = IonSpotBeamSequence.from_beams([beam_a, beam_b])

    assert sequence.iso_center is None
    assert torch.allclose(
        sequence.iso_centers,
        torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], device=default_device, dtype=default_dtype),
    )
    assert sequence[0].iso_center == (10.0, 20.0, 30.0)
    assert sequence[1].iso_center == (40.0, 50.0, 60.0)


def test_ion_spot_beam_sigma_gradients_are_opt_in(default_device, default_dtype):
    beam = IonSpotBeam.create(
        gantry_angle_deg=0.0,
        spot_positions_mm=torch.tensor([[0.0, 0.0]], device=default_device, dtype=default_dtype),
        spot_weights=torch.tensor([1.0], device=default_device, dtype=default_dtype),
        spot_layer_index=torch.tensor([0], device=default_device, dtype=torch.long),
        layer_energies_mev=torch.tensor([120.0], device=default_device, dtype=default_dtype),
        layer_sigmas_mm=torch.tensor([[2.0, 3.0]], device=default_device, dtype=default_dtype),
        requires_grad=True,
        sigma_requires_grad=True,
    )

    assert beam.layer_sigmas_mm.requires_grad
