"""Tests for IonDoseEngine lattice path (compute_dose_bev_lattice_sparse_batch)."""

from pathlib import Path

import torch

from pydose_rt.data.ion_beam import IonSpotBeam, IonSpotBeamSequence
from pydose_rt.engine.ion_dose_engine import IonDoseEngine
from pydose_rt.physics.kernels.ion_lut import PyRadPlanIonLUT
from pydose_rt.sparse.ions import IonSparseHooks


_PYRADPLAN_MAT = Path("example_data/pyradplan/protons_Generic.mat")


def _make_engine(machine_config, sequence, device, dtype, **kwargs):
    return IonDoseEngine(
        machine_config=machine_config,
        lut=PyRadPlanIonLUT(_PYRADPLAN_MAT),
        dose_grid_spacing=(2.0, 2.0, 2.0),
        dose_grid_shape=(24, 24, 24),
        beam_template=sequence,
        device=device,
        dtype=dtype,
        lateral_model="gauss",
        **kwargs,
    )


def _single_spot_sequence(device, dtype, angle_deg=0.0, energy_mev=100.0, weight=1.0):
    beam = IonSpotBeam.create(
        gantry_angle_deg=angle_deg,
        spot_positions_mm=torch.tensor([[0.0, 0.0]], device=device, dtype=dtype),
        spot_weights=torch.tensor([weight], device=device, dtype=dtype),
        spot_layer_index=torch.tensor([0], device=device, dtype=torch.long),
        layer_energies_mev=torch.tensor([energy_mev], device=device, dtype=dtype),
        layer_sigmas_mm=torch.tensor([[2.0, 2.0]], device=device, dtype=dtype),
        field_size=(64, 64),
        iso_center=(24.0, 24.0, 24.0),
        requires_grad=False,
    )
    return IonSpotBeamSequence.from_beams([beam])


def test_lattice_output_shape(default_machine_config, default_device, default_dtype):
    seq = _single_spot_sequence(default_device, default_dtype)
    engine = _make_engine(default_machine_config, seq, default_device, default_dtype)
    density = torch.ones((1, 24, 24, 24), device=default_device, dtype=default_dtype)

    dose = engine.compute_dose_bev_lattice_sparse_batch(seq, density)

    assert dose.shape == (1, 24, 24, 24)
    assert dose.sum() > 0.0
    assert torch.all(dose >= 0.0)


def test_lattice_dose_scales_linearly_with_weight(default_machine_config, default_device, default_dtype):
    seq1 = _single_spot_sequence(default_device, default_dtype, weight=1.0)
    seq2 = _single_spot_sequence(default_device, default_dtype, weight=2.0)
    engine = _make_engine(default_machine_config, seq1, default_device, default_dtype)
    density = torch.ones((1, 24, 24, 24), device=default_device, dtype=default_dtype)

    dose1 = engine.compute_dose_bev_lattice_sparse_batch(seq1, density)
    dose2 = engine.compute_dose_bev_lattice_sparse_batch(seq2, density)

    torch.testing.assert_close(dose2, dose1 * 2.0, rtol=1e-4, atol=1e-6)


def test_lattice_dose_is_non_negative_in_water(default_machine_config, default_device, default_dtype):
    seq = _single_spot_sequence(default_device, default_dtype, energy_mev=150.0)
    engine = _make_engine(default_machine_config, seq, default_device, default_dtype)
    density = torch.ones((1, 24, 24, 24), device=default_device, dtype=default_dtype)

    dose = engine.compute_dose_bev_lattice_sparse_batch(seq, density)

    assert torch.all(dose >= 0.0)


def test_lattice_dense_bev_hook_can_zero_dose(default_machine_config, default_device, default_dtype):
    class _ZeroHook(torch.nn.Module):
        def forward(self, payload, **_ctx):
            return {k: torch.zeros_like(v) if k == "deposited_energy" else v for k, v in payload.items()}

    seq = _single_spot_sequence(default_device, default_dtype)
    hooks = IonSparseHooks(dense_bev=_ZeroHook())
    engine = _make_engine(default_machine_config, seq, default_device, default_dtype, sparse_hooks=hooks)
    density = torch.ones((1, 24, 24, 24), device=default_device, dtype=default_dtype)

    dose = engine.compute_dose_bev_lattice_sparse_batch(seq, density)

    assert dose.sum().item() < 1e-5


def test_lattice_field_size_controls_bev_crop(default_machine_config, default_device, default_dtype):
    """Smaller field_size should still produce a valid patient-frame dose."""
    seq = _single_spot_sequence(default_device, default_dtype)
    engine = _make_engine(
        default_machine_config, seq, default_device, default_dtype, field_size=(16, 16)
    )
    density = torch.ones((1, 24, 24, 24), device=default_device, dtype=default_dtype)

    dose = engine.compute_dose_bev_lattice_sparse_batch(seq, density)

    assert dose.shape == (1, 24, 24, 24)
    assert dose.sum() > 0.0


