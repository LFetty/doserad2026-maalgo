"""Hong-style pyRadPlan machine-base-data LUT loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def _load_mat(mat_path: Path):
    try:
        import scipy.io as sio
    except ImportError as exc:
        raise RuntimeError("scipy is required to load pyRadPlan .mat machine data") from exc
    return sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)


def _interp1d(
    x: torch.Tensor,
    y: torch.Tensor,
    x_new: torch.Tensor,
    left: torch.Tensor | float | None = None,
    right: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """1D interpolation with edge-value extrapolation."""
    if x.numel() != y.numel():
        raise ValueError("x and y must have the same length")
    if x.numel() == 1:
        return torch.zeros_like(x_new, dtype=y.dtype, device=y.device) + y[0]

    x_new_flat = x_new.reshape(-1)
    indices = torch.searchsorted(x, x_new_flat, right=False).clamp(1, x.numel() - 1)

    x0 = x[indices - 1]
    x1 = x[indices]
    y0 = y[indices - 1]
    y1 = y[indices]
    slope = (y1 - y0) / (x1 - x0).clamp_min(torch.finfo(y.dtype).eps)
    y_new = y0 + slope * (x_new_flat - x0)

    if left is None:
        left = y[0]
    if right is None:
        right = y[-1]

    left_tensor = torch.as_tensor(left, device=y.device, dtype=y.dtype)
    right_tensor = torch.as_tensor(right, device=y.device, dtype=y.dtype)
    y_new = torch.where(x_new_flat < x[0], left_tensor, y_new)
    y_new = torch.where(x_new_flat > x[-1], right_tensor, y_new)
    return y_new.view_as(x_new)


class PyRadPlanIonLUT:
    """LUT wrapper driven by pyRadPlan ``*_Generic.mat`` machine data."""

    def __init__(
        self,
        mat_path: Path | str,
        calculation_step_mm: float | None = None,
        beam_particles: int = 1,
        source_particles: int = 1,
    ) -> None:
        self.mat_path = Path(mat_path)
        if not self.mat_path.is_file():
            raise FileNotFoundError(f"pyRadPlan machine data not found: {self.mat_path}")
        self.lut_dir = self.mat_path.parent
        self.material_table = None
        self.calculation_step_mm: float | None = None
        self.set_calculation_step_mm(calculation_step_mm)
        self.beam_particles = int(beam_particles)
        self.source_particles = int(source_particles)

        raw = _load_mat(self.mat_path)
        machine = raw["machine"]
        meta = machine.meta
        self.meta_sad_mm = float(getattr(meta, "SAD"))
        self.meta_bams_to_iso_mm = float(getattr(meta, "BAMStoIsoDist"))

        entries = list(machine.data)
        rows: list[
            tuple[
                float,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                float,
                np.ndarray | None,
                np.ndarray | None,
                np.ndarray | None,
                np.ndarray | None,
                np.ndarray | None,
            ]
        ] = []
        has_double = True
        for entry in entries:
            energy = float(getattr(entry, "energy"))
            depths = np.asarray(getattr(entry, "depths"), dtype=np.float64).ravel()
            idd = np.asarray(getattr(entry, "Z"), dtype=np.float64).ravel()
            sigma = np.asarray(getattr(entry, "sigma"), dtype=np.float64).ravel()
            offset = float(getattr(entry, "offset", 0.0))
            if depths.shape != idd.shape or depths.shape != sigma.shape:
                raise ValueError(
                    f"pyRadPlan entry at {energy:g} MeV has mismatched depth/Z/sigma shapes"
                )
            sigma1_raw = getattr(entry, "sigma1", None)
            sigma2_raw = getattr(entry, "sigma2", None)
            weight_raw = getattr(entry, "weight", None)
            init_focus = getattr(entry, "initFocus", None)
            if sigma1_raw is None or sigma2_raw is None or weight_raw is None:
                has_double = False
                sigma1_arr = sigma2_arr = weight_arr = None
            else:
                sigma1_arr = np.asarray(sigma1_raw, dtype=np.float64).ravel()
                sigma2_arr = np.asarray(sigma2_raw, dtype=np.float64).ravel()
                weight_arr = np.asarray(weight_raw, dtype=np.float64).ravel()
                if (
                    sigma1_arr.shape != depths.shape
                    or sigma2_arr.shape != depths.shape
                    or weight_arr.shape != depths.shape
                ):
                    raise ValueError(
                        f"pyRadPlan entry at {energy:g} MeV has mismatched sigma1/sigma2/weight shapes"
                    )
            if init_focus is None:
                focus_dist_arr = None
                focus_sigma_arr = None
            else:
                focus_dist_arr = np.asarray(getattr(init_focus, "dist"), dtype=np.float64).ravel()
                focus_sigma_arr = np.asarray(getattr(init_focus, "sigma"), dtype=np.float64).ravel()
                if focus_dist_arr.shape != focus_sigma_arr.shape:
                    raise ValueError(
                        f"pyRadPlan entry at {energy:g} MeV has mismatched initFocus dist/sigma shapes"
                    )
                focus_order = np.argsort(focus_dist_arr)
                focus_dist_arr = focus_dist_arr[focus_order].astype(np.float64, copy=False)
                focus_sigma_arr = focus_sigma_arr[focus_order].astype(np.float64, copy=False)
            order = np.argsort(depths)
            rows.append(
                (
                    energy,
                    depths[order].astype(np.float64, copy=False),
                    idd[order].astype(np.float64, copy=False),
                    sigma[order].astype(np.float64, copy=False),
                    offset,
                    None if sigma1_arr is None else sigma1_arr[order].astype(np.float64, copy=False),
                    None if sigma2_arr is None else sigma2_arr[order].astype(np.float64, copy=False),
                    None if weight_arr is None else weight_arr[order].astype(np.float64, copy=False),
                    focus_dist_arr,
                    focus_sigma_arr,
                )
            )
        rows.sort(key=lambda r: r[0])
        self._energies: List[float] = [r[0] for r in rows]
        self._depths: List[np.ndarray] = [r[1] for r in rows]
        self._idd: List[np.ndarray] = [r[2] for r in rows]
        self._sigma: List[np.ndarray] = [r[3] for r in rows]
        self._offsets: List[float] = [r[4] for r in rows]
        self._sigma1: List[np.ndarray | None] = [r[5] for r in rows]
        self._sigma2: List[np.ndarray | None] = [r[6] for r in rows]
        self._weight: List[np.ndarray | None] = [r[7] for r in rows]
        self._focus_dist: List[np.ndarray | None] = [r[8] for r in rows]
        self._focus_sigma: List[np.ndarray | None] = [r[9] for r in rows]
        self.has_double_gauss: bool = has_double
        self.has_initial_focus: bool = all(
            dist is not None and sigma is not None
            for dist, sigma in zip(self._focus_dist, self._focus_sigma)
        )

        self._edep_curve_cache: Dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self._sigma_curve_cache: Dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self._raw_cache: Dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._double_raw_cache: Dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._double_curve_cache: Dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._focus_raw_cache: Dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

        # Optional differentiable residual hook (see
        # training.proton.lut_residuals.DifferentiableLUTResiduals). Inert until
        # attached. When attached and ``active`` it rewrites the per-energy edep
        # and sigma curves with learnable smooth residuals; the detached curve
        # caches are bypassed so updated parameters are not masked by a stale
        # first-forward result.
        self.differentiable_residuals: Any = None

    def _residuals_active(self) -> bool:
        res = self.differentiable_residuals
        return res is not None and bool(getattr(res, "active", False))

    @property
    def available_energies(self) -> List[float]:
        return list(self._energies)

    def set_calculation_step_mm(self, step_mm: float | None) -> None:
        self.calculation_step_mm = None if step_mm is None else float(step_mm)

    def _bracket(self, energy_mev: float) -> Tuple[int, int, float]:
        energies = self._energies
        for i, e in enumerate(energies):
            if np.isclose(energy_mev, e):
                return i, i, 0.0
        if energy_mev <= energies[0]:
            return 0, 0, 0.0
        if energy_mev >= energies[-1]:
            n = len(energies) - 1
            return n, n, 1.0
        for i in range(len(energies) - 1):
            if energies[i] <= energy_mev <= energies[i + 1]:
                frac = (energy_mev - energies[i]) / (energies[i + 1] - energies[i])
                return i, i + 1, frac
        return 0, 0, 0.0

    def _load_raw(
        self,
        idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (idx, str(device), dtype)
        if key not in self._raw_cache:
            depth = torch.tensor(self._depths[idx], device=device, dtype=dtype)
            idd = torch.tensor(self._idd[idx], device=device, dtype=dtype)
            sigma = torch.tensor(self._sigma[idx], device=device, dtype=dtype)
            self._raw_cache[key] = (depth, idd, sigma)
        return self._raw_cache[key]

    def _load_double_raw(
        self,
        idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.has_double_gauss:
            raise RuntimeError("This LUT does not provide double-Gaussian parameters")
        key = (idx, str(device), dtype)
        if key not in self._double_raw_cache:
            sigma1 = torch.tensor(self._sigma1[idx], device=device, dtype=dtype)
            sigma2 = torch.tensor(self._sigma2[idx], device=device, dtype=dtype)
            weight = torch.tensor(self._weight[idx], device=device, dtype=dtype)
            self._double_raw_cache[key] = (sigma1, sigma2, weight)
        return self._double_raw_cache[key]

    def _load_focus_raw(
        self,
        idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.has_initial_focus:
            raise RuntimeError("This LUT does not provide initial focus parameters")
        key = (idx, str(device), dtype)
        if key not in self._focus_raw_cache:
            dist = torch.tensor(self._focus_dist[idx], device=device, dtype=dtype)
            sigma = torch.tensor(self._focus_sigma[idx], device=device, dtype=dtype)
            self._focus_raw_cache[key] = (dist, sigma)
        return self._focus_raw_cache[key]

    def get_double_gauss_curves(
        self,
        energy_mev: torch.Tensor | float,
        *,
        energy_value_hint: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.has_double_gauss:
            raise RuntimeError("This LUT does not provide double-Gaussian parameters")
        energy = torch.as_tensor(energy_mev)
        if energy_value_hint is None:
            energy_value_hint = float(energy.detach().cpu())
        cache_key = (round(energy_value_hint, 6), str(energy.device), energy.dtype)
        active = self._residuals_active()
        if not energy.requires_grad and not active and cache_key in self._double_curve_cache:
            return self._double_curve_cache[cache_key]

        lo, hi, _ = self._bracket(energy_value_hint)
        if lo == hi:
            depth, _, _ = self._load_raw(lo, energy.device, energy.dtype)
            s1, s2, w = self._load_double_raw(lo, energy.device, energy.dtype)
            if active:
                s1, s2, w = self.differentiable_residuals.apply_double_gauss(
                    lo, depth, s1, s2, w
                )
            result = (depth, s1, s2, w)
            if not energy.requires_grad and not active:
                self._double_curve_cache[cache_key] = result
            return result

        lo_depth, lo_idd, _ = self._load_raw(lo, energy.device, energy.dtype)
        hi_depth, hi_idd, _ = self._load_raw(hi, energy.device, energy.dtype)
        lo_s1, lo_s2, lo_w = self._load_double_raw(lo, energy.device, energy.dtype)
        hi_s1, hi_s2, hi_w = self._load_double_raw(hi, energy.device, energy.dtype)

        depth = self._common_depth_grid(lo_depth, hi_depth)
        lo_energy = torch.tensor(self._energies[lo], device=energy.device, dtype=energy.dtype)
        hi_energy = torch.tensor(self._energies[hi], device=energy.device, dtype=energy.dtype)
        frac = (energy - lo_energy) / (hi_energy - lo_energy).clamp_min(torch.finfo(energy.dtype).eps)

        lo_peak_depth = lo_depth[torch.argmax(lo_idd)]
        hi_peak_depth = hi_depth[torch.argmax(hi_idd)]
        interp_peak_depth = lo_peak_depth + frac * (hi_peak_depth - lo_peak_depth)
        safe_peak_depth = interp_peak_depth.clamp_min(torch.finfo(energy.dtype).eps)
        pos_scale = lo_peak_depth / safe_peak_depth
        scaled_positions = pos_scale * depth

        def _blend(lo_arr: torch.Tensor, hi_arr: torch.Tensor) -> torch.Tensor:
            lo_v = _interp1d(lo_depth, lo_arr, scaled_positions)
            hi_v = _interp1d(hi_depth, hi_arr, depth)
            return lo_v + frac * (hi_v - lo_v)

        sigma1 = _blend(lo_s1, hi_s1).clamp_min(0.0)
        sigma2 = _blend(lo_s2, hi_s2).clamp_min(0.0)
        weight = _blend(lo_w, hi_w).clamp(0.0, 1.0)
        result = (depth, sigma1, sigma2, weight)
        if not energy.requires_grad and not active:
            self._double_curve_cache[cache_key] = result
        return result

    def get_double_gauss(
        self,
        energy_mev: torch.Tensor | float,
        depth_mm: torch.Tensor | float,
        *,
        energy_value_hint: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy = torch.as_tensor(energy_mev)
        depth = torch.as_tensor(depth_mm, device=energy.device, dtype=energy.dtype)
        depth_curve, s1_curve, s2_curve, w_curve = self.get_double_gauss_curves(
            energy,
            energy_value_hint=energy_value_hint,
        )
        s1 = _interp1d(depth_curve, s1_curve, depth).clamp_min(0.0)
        s2 = _interp1d(depth_curve, s2_curve, depth).clamp_min(0.0)
        w = _interp1d(depth_curve, w_curve, depth).clamp(0.0, 1.0)
        return s1, s2, w

    def get_initial_sigma(
        self,
        energy_mev: torch.Tensor | float,
        source_to_surface_mm: torch.Tensor | float,
        *,
        energy_value_hint: float | None = None,
    ) -> torch.Tensor:
        energy = torch.as_tensor(energy_mev)
        source_to_surface = torch.as_tensor(
            source_to_surface_mm,
            device=energy.device,
            dtype=energy.dtype,
        )
        if energy_value_hint is None:
            energy_value_hint = float(energy.detach().cpu())

        if not self.has_initial_focus:
            raise RuntimeError("This LUT does not provide initial focus parameters")

        lo, hi, _ = self._bracket(energy_value_hint)
        if lo == hi:
            dist, sigma = self._load_focus_raw(lo, energy.device, energy.dtype)
            return _interp1d(dist, sigma, source_to_surface).clamp_min(0.0)

        lo_dist, lo_sigma = self._load_focus_raw(lo, energy.device, energy.dtype)
        hi_dist, hi_sigma = self._load_focus_raw(hi, energy.device, energy.dtype)
        lo_value = _interp1d(lo_dist, lo_sigma, source_to_surface)
        hi_value = _interp1d(hi_dist, hi_sigma, source_to_surface)
        lo_energy = torch.tensor(self._energies[lo], device=energy.device, dtype=energy.dtype)
        hi_energy = torch.tensor(self._energies[hi], device=energy.device, dtype=energy.dtype)
        frac = (energy - lo_energy) / (hi_energy - lo_energy).clamp_min(torch.finfo(energy.dtype).eps)
        return (lo_value + frac * (hi_value - lo_value)).clamp_min(0.0)

    def _edep_scale(self, depth: torch.Tensor) -> torch.Tensor:
        particle_scale = self.beam_particles / max(self.source_particles, 1)
        return torch.tensor(particle_scale, device=depth.device, dtype=depth.dtype)

    def _common_depth_grid(
        self,
        lo_depth: torch.Tensor,
        hi_depth: torch.Tensor,
    ) -> torch.Tensor:
        max_depth = float(max(lo_depth[-1].item(), hi_depth[-1].item()))
        diffs_lo = (lo_depth[1:] - lo_depth[:-1]).abs()
        diffs_hi = (hi_depth[1:] - hi_depth[:-1]).abs()
        pos = torch.cat([diffs_lo[diffs_lo > 0.0], diffs_hi[diffs_hi > 0.0]])
        step = float(pos.min().item()) if pos.numel() > 0 else 1.0
        n = int(np.ceil(max_depth / step)) + 1
        return torch.arange(n, device=lo_depth.device, dtype=lo_depth.dtype) * step

    def _peak_shift_interp(
        self,
        energy: torch.Tensor,
        lo_idx: int,
        hi_idx: int,
        pick: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lo_depth, lo_idd, lo_sigma = self._load_raw(lo_idx, energy.device, energy.dtype)
        hi_depth, hi_idd, hi_sigma = self._load_raw(hi_idx, energy.device, energy.dtype)
        depth = self._common_depth_grid(lo_depth, hi_depth)
        lo_energy = torch.tensor(self._energies[lo_idx], device=energy.device, dtype=energy.dtype)
        hi_energy = torch.tensor(self._energies[hi_idx], device=energy.device, dtype=energy.dtype)
        frac = (energy - lo_energy) / (hi_energy - lo_energy).clamp_min(torch.finfo(energy.dtype).eps)

        lo_peak_idx = torch.argmax(lo_idd)
        hi_peak_idx = torch.argmax(hi_idd)
        lo_peak_depth = lo_depth[lo_peak_idx]
        hi_peak_depth = hi_depth[hi_peak_idx]
        interp_peak_depth = lo_peak_depth + frac * (hi_peak_depth - lo_peak_depth)
        safe_peak_depth = interp_peak_depth.clamp_min(torch.finfo(energy.dtype).eps)
        pos_scale = lo_peak_depth / safe_peak_depth
        scaled_positions = pos_scale * depth

        if pick == "idd":
            lo_peak = lo_idd[lo_peak_idx].clamp_min(torch.finfo(energy.dtype).eps)
            hi_peak = hi_idd[hi_peak_idx]
            interp_peak = lo_peak + frac * (hi_peak - lo_peak)
            edep_scale = interp_peak / lo_peak
            scaled = _interp1d(lo_depth, lo_idd, scaled_positions) * edep_scale
            valid = (interp_peak_depth > 0.0) & (lo_peak > 0.0)
            scaled = torch.where(valid, scaled, torch.zeros_like(scaled))
            return depth, scaled.clamp_min(0.0)

        lo_interp = _interp1d(lo_depth, lo_sigma, scaled_positions)
        hi_interp = _interp1d(hi_depth, hi_sigma, depth)
        sigma = lo_interp + frac * (hi_interp - lo_interp)
        return depth, sigma.clamp_min(0.0)

    def get_edep_curve(
        self,
        energy_mev: torch.Tensor | float,
        *,
        energy_value_hint: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        energy = torch.as_tensor(energy_mev)
        if energy_value_hint is None:
            energy_value_hint = float(energy.detach().cpu())
        cache_key = (round(energy_value_hint, 6), str(energy.device), energy.dtype)
        active = self._residuals_active()
        if not energy.requires_grad and not active and cache_key in self._edep_curve_cache:
            return self._edep_curve_cache[cache_key]

        lo, hi, _ = self._bracket(energy_value_hint)
        if lo == hi:
            depth, idd, _ = self._load_raw(lo, energy.device, energy.dtype)
            result = (depth, idd * self._edep_scale(depth))
        else:
            depth, idd = self._peak_shift_interp(energy, lo, hi, pick="idd")
            result = (depth, idd * self._edep_scale(depth))
        if active and lo == hi:
            depth, edep = result
            result = (depth, self.differentiable_residuals.apply_edep(lo, depth, edep))
        if not energy.requires_grad and not active:
            self._edep_curve_cache[cache_key] = result
        return result

    def get_edep(
        self,
        energy_mev: torch.Tensor | float,
        depth_mm: torch.Tensor | float,
        *,
        energy_value_hint: float | None = None,
    ) -> torch.Tensor:
        energy = torch.as_tensor(energy_mev)
        depth = torch.as_tensor(depth_mm, device=energy.device, dtype=energy.dtype)
        depth_curve, edep_curve = self.get_edep_curve(energy, energy_value_hint=energy_value_hint)
        return _interp1d(depth_curve, edep_curve, depth)

    def get_kernel_offset(
        self,
        energy_mev: torch.Tensor | float,
        *,
        energy_value_hint: float | None = None,
    ) -> torch.Tensor:
        energy = torch.as_tensor(energy_mev)
        if energy_value_hint is None:
            energy_value_hint = float(energy.detach().cpu())

        lo, hi, _ = self._bracket(energy_value_hint)
        lo_offset = torch.tensor(self._offsets[lo], device=energy.device, dtype=energy.dtype)
        if lo == hi:
            return lo_offset

        hi_offset = torch.tensor(self._offsets[hi], device=energy.device, dtype=energy.dtype)
        lo_energy = torch.tensor(self._energies[lo], device=energy.device, dtype=energy.dtype)
        hi_energy = torch.tensor(self._energies[hi], device=energy.device, dtype=energy.dtype)
        frac = (energy - lo_energy) / (hi_energy - lo_energy).clamp_min(torch.finfo(energy.dtype).eps)
        return lo_offset + frac * (hi_offset - lo_offset)

    def get_sigma_curve(
        self,
        energy_mev: torch.Tensor | float,
        *,
        energy_value_hint: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        energy = torch.as_tensor(energy_mev)
        if energy_value_hint is None:
            energy_value_hint = float(energy.detach().cpu())
        cache_key = (round(energy_value_hint, 6), str(energy.device), energy.dtype)
        active = self._residuals_active()
        if not energy.requires_grad and not active and cache_key in self._sigma_curve_cache:
            return self._sigma_curve_cache[cache_key]

        lo, hi, _ = self._bracket(energy_value_hint)
        if lo == hi:
            depth, _, sigma = self._load_raw(lo, energy.device, energy.dtype)
            result = (depth, sigma)
        else:
            result = self._peak_shift_interp(energy, lo, hi, pick="sigma")
        if active and lo == hi:
            depth, sigma = result
            result = (depth, self.differentiable_residuals.apply_sigma(lo, depth, sigma))
        if not energy.requires_grad and not active:
            self._sigma_curve_cache[cache_key] = result
        return result

    def get_sigma(
        self,
        energy_mev: torch.Tensor | float,
        depth_mm: torch.Tensor | float,
        *,
        energy_value_hint: float | None = None,
    ) -> torch.Tensor:
        energy = torch.as_tensor(energy_mev)
        depth = torch.as_tensor(depth_mm, device=energy.device, dtype=energy.dtype)
        depth_curve, sigma_curve = self.get_sigma_curve(
            energy,
            energy_value_hint=energy_value_hint,
        )
        return _interp1d(depth_curve, sigma_curve, depth).clamp_min(0.0)
