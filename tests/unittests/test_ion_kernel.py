from pathlib import Path

import torch

from pydose_rt.physics.kernels.ion_pencil_beam_model import IonPencilBeamModel
from pydose_rt.physics.kernels.ion_lut import PyRadPlanIonLUT


_PYRADPLAN_MAT = Path("example_data/pyradplan/protons_Generic.mat")


def _lut() -> PyRadPlanIonLUT:
    return PyRadPlanIonLUT(_PYRADPLAN_MAT, calculation_step_mm=1.0)


def test_ion_lut_loads_hong_machine_data():
    lut = _lut()

    assert len(lut.available_energies) > 10
    assert torch.as_tensor(lut.get_edep(100.0, 0.0)).item() >= 0.0
    assert torch.as_tensor(lut.get_sigma(100.0, 0.0)).item() >= 0.0
    assert lut.has_initial_focus


def test_ion_lut_initial_focus_matches_mat_reference_values():
    lut = _lut()

    assert torch.isclose(
        lut.get_initial_sigma(95.26045078341389, 9000.0),
        torch.tensor(6.69508647),
        atol=1e-5,
        rtol=0.0,
    )
    assert torch.isclose(
        lut.get_initial_sigma(135.14582670578662, 10000.0),
        torch.tensor(5.37118962),
        atol=1e-5,
        rtol=0.0,
    )


def test_ion_lut_interpolates_curves_on_common_depth_grid():
    lut = _lut()

    target_energy = 135.0
    depth, edep = lut.get_edep_curve(target_energy)
    lower_depth, lower_edep = lut.get_edep_curve(130.0)
    _, higher_edep = lut.get_edep_curve(140.0)

    assert depth.ndim == 1
    assert edep.shape == depth.shape
    assert depth[0] == 0.0
    assert torch.all(torch.diff(depth) > 0.0)
    assert depth[-1] > lower_depth[-1]
    assert torch.argmax(edep) >= min(int(torch.argmax(lower_edep)), int(torch.argmax(higher_edep)))


def test_ion_kernel_cell_weights_gradients_reach_depth_sigma_and_energy():
    lut = _lut()
    model = IonPencilBeamModel(
        lut=lut,
        energy_mev=100.0,
        resolution=(1.0, 1.0, 1.0),
        lateral_model="gauss",
    )

    energy = torch.tensor(135.0, dtype=torch.float32, requires_grad=True)
    depth = torch.tensor(20.0, dtype=torch.float32, requires_grad=True)
    sigma_x = torch.tensor(4.0, dtype=torch.float32, requires_grad=True)
    sigma_y = torch.tensor(2.5, dtype=torch.float32, requires_grad=True)

    weights = model.evaluate_lateral_cell_weights(
        depth,
        x_mm=torch.tensor(1.2),
        y_mm=torch.tensor(-0.7),
        sigma_x_mm=sigma_x,
        sigma_y_mm=sigma_y,
        cell_width_x_mm=torch.tensor(0.5),
        cell_width_y_mm=torch.tensor(0.5),
        energy_mev=energy,
    )
    edep = model.lut.get_edep(energy, depth)
    loss = (edep * weights).sum()
    loss.backward()

    assert depth.grad is not None
    assert sigma_x.grad is not None
    assert sigma_y.grad is not None
    assert energy.grad is not None
    assert torch.any(depth.grad != 0.0)
    assert torch.any(sigma_x.grad != 0.0)
    assert torch.any(sigma_y.grad != 0.0)
    assert torch.any(energy.grad != 0.0)


def test_ion_kernel_double_gaussian_is_available_when_lut_has_parameters():
    lut = _lut()
    if not lut.has_double_gauss:
        return

    model = IonPencilBeamModel(
        lut=lut,
        energy_mev=130.0,
        resolution=(1.0, 1.0, 1.0),
        lateral_model="gauss_double",
    )

    weights = model.evaluate_lateral_cell_weights(
        torch.tensor(24.0, dtype=torch.float32),
        x_mm=torch.tensor(1.5, dtype=torch.float32),
        y_mm=torch.tensor(-2.0, dtype=torch.float32),
        sigma_x_mm=torch.tensor(3.4, dtype=torch.float32),
        sigma_y_mm=torch.tensor(2.2, dtype=torch.float32),
        cell_width_x_mm=torch.tensor(0.5, dtype=torch.float32),
        cell_width_y_mm=torch.tensor(0.5, dtype=torch.float32),
        energy_mev=torch.tensor(130.0, dtype=torch.float32),
    )

    assert torch.isfinite(weights).all()
    assert torch.all(weights >= 0.0)
