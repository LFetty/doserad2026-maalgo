"""Hong-style ion pencil beam kernel model."""

from __future__ import annotations

import math

import torch


def _gaussian_cell_1d(
    x_mm: torch.Tensor,
    sigma_mm: torch.Tensor,
    cell_width_mm: torch.Tensor,
) -> torch.Tensor:
    sigma = sigma_mm.clamp_min(torch.finfo(x_mm.dtype).eps)
    half_width = 0.5 * cell_width_mm
    inv = 1.0 / (math.sqrt(2.0) * sigma)
    upper = (x_mm + half_width) * inv
    lower = (x_mm - half_width) * inv
    return 0.5 * (torch.erf(upper) - torch.erf(lower))


class IonPencilBeamModel:
    """Hong lateral kernel builder backed by pyRadPlan LUT data.

    Supported lateral models:
      ``gauss``        - single Gaussian
      ``gauss_double`` - double Gaussian from LUT ``sigma1/sigma2/weight``

    For the double-Gaussian model we follow pyRadPlan semantics exactly:
      ``sigma1_total^2 = sigma1_lut^2 + sigma_ini^2``
      ``sigma2_total^2 = sigma2_lut^2 + sigma_ini^2``
    where ``sigma_ini`` is the external entrance/focus/beam-parameter term.
    """

    def __init__(
        self,
        lut,
        energy_mev: float,
        resolution: tuple,
        lateral_model: str = "gauss",
    ) -> None:
        if lateral_model not in {"gauss", "gauss_double"}:
            raise ValueError("lateral_model must be 'gauss' or 'gauss_double', got " f"{lateral_model!r}")
        if lateral_model == "gauss_double" and not getattr(lut, "has_double_gauss", False):
            raise ValueError(
                "lateral_model='gauss_double' requires LUT sigma1/sigma2/weight parameters"
            )
        self.lateral_model = lateral_model
        self.lut = lut
        self.energy_mev = float(energy_mev)
        self.resolution = resolution
        if len(resolution) >= 2:
            self.lut.set_calculation_step_mm(float(resolution[1]))

        self.res_h = float(resolution[0])
        self.res_w = float(resolution[2])

    def get_double_gauss_total_sigmas(
        self,
        kernel_depth_mm: torch.Tensor | float,
        sigma_x_total_mm: torch.Tensor | float,
        sigma_y_total_mm: torch.Tensor | float,
        energy_mev: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if energy_mev is None:
            energy_mev = self.energy_mev

        energy = torch.as_tensor(energy_mev)
        depth = torch.as_tensor(kernel_depth_mm, device=energy.device, dtype=energy.dtype)
        sigma_x_total = torch.as_tensor(sigma_x_total_mm, device=energy.device, dtype=energy.dtype)
        sigma_y_total = torch.as_tensor(sigma_y_total_mm, device=energy.device, dtype=energy.dtype)
        depth, sigma_x_total, sigma_y_total = torch.broadcast_tensors(depth, sigma_x_total, sigma_y_total)

        sigma_transport = self.lut.get_sigma(energy, depth).clamp_min(0.0)
        sigma_ini_sq_x = (sigma_x_total.square() - sigma_transport.square()).clamp_min(0.0)
        sigma_ini_sq_y = (sigma_y_total.square() - sigma_transport.square()).clamp_min(0.0)

        s1, s2, w_halo = self.lut.get_double_gauss(energy, depth)
        sigma1_total_x = (s1.square() + sigma_ini_sq_x).sqrt().clamp_min(1e-6)
        sigma1_total_y = (s1.square() + sigma_ini_sq_y).sqrt().clamp_min(1e-6)
        sigma2_total_x = (s2.square() + sigma_ini_sq_x).sqrt().clamp_min(1e-6)
        sigma2_total_y = (s2.square() + sigma_ini_sq_y).sqrt().clamp_min(1e-6)
        return (
            sigma1_total_x,
            sigma1_total_y,
            sigma2_total_x,
            sigma2_total_y,
            w_halo.clamp(0.0, 1.0),
        )

    def evaluate_lateral_cell_weights(
        self,
        depth_water_mm: torch.Tensor | float,
        x_mm: torch.Tensor | float,
        y_mm: torch.Tensor | float,
        sigma_x_mm: torch.Tensor | float,
        sigma_y_mm: torch.Tensor | float | None = None,
        cell_width_x_mm: torch.Tensor | float | None = None,
        cell_width_y_mm: torch.Tensor | float | None = None,
        energy_mev: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        if cell_width_x_mm is None or cell_width_y_mm is None:
            raise ValueError("cell_width_x_mm and cell_width_y_mm are required")
        if energy_mev is None:
            energy_mev = self.energy_mev

        energy = torch.as_tensor(energy_mev)
        depth = torch.as_tensor(depth_water_mm, device=energy.device, dtype=energy.dtype)
        x = torch.as_tensor(x_mm, device=energy.device, dtype=energy.dtype)
        y = torch.as_tensor(y_mm, device=energy.device, dtype=energy.dtype)
        sigma_x = torch.as_tensor(sigma_x_mm, device=energy.device, dtype=energy.dtype)
        sigma_y = sigma_x if sigma_y_mm is None else torch.as_tensor(sigma_y_mm, device=energy.device, dtype=energy.dtype)
        cell_x = torch.as_tensor(cell_width_x_mm, device=energy.device, dtype=energy.dtype)
        cell_y = torch.as_tensor(cell_width_y_mm, device=energy.device, dtype=energy.dtype)
        depth, x, y, sigma_x, sigma_y, cell_x, cell_y = torch.broadcast_tensors(
            depth,
            x,
            y,
            sigma_x,
            sigma_y,
            cell_x,
            cell_y,
        )

        if self.lateral_model == "gauss":
            weight_y = _gaussian_cell_1d(y, sigma_y.clamp_min(1e-6), cell_y)
            weight_x = _gaussian_cell_1d(x, sigma_x.clamp_min(1e-6), cell_x)
            return weight_y * weight_x

        sx_n, sy_n, sx_b, sy_b, w_halo = self.get_double_gauss_total_sigmas(
            kernel_depth_mm=depth,
            sigma_x_total_mm=sigma_x,
            sigma_y_total_mm=sigma_y,
            energy_mev=energy,
        )
        narrow = _gaussian_cell_1d(y, sy_n, cell_y) * _gaussian_cell_1d(x, sx_n, cell_x)
        broad = _gaussian_cell_1d(y, sy_b, cell_y) * _gaussian_cell_1d(x, sx_b, cell_x)
        return (1.0 - w_halo) * narrow + w_halo * broad
