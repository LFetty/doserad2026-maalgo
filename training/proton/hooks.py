from __future__ import annotations

from contextlib import nullcontext
import math
import os
from pathlib import Path
import time
from typing import Any

import copy
import torch
from torch import nn
import torch.nn.functional as F

from training.common.checkpoints import load_model_state_dict
from training.common.materials import GEANT4_HU_BOUNDS, GEANT4_NUM_MATERIALS
from training.common.ray_sequence_corrector import FanGridConvCorrector, RaySequenceCorrector
from training.common.repvgg_unet_corrector import RepVGGUNetCorrector
from training.common.separable_fan_grid_corrector import SeparableFanGridConvCorrector


def _cfg_with_support_overrides(
    cfg: dict[str, Any],
    max_support_rays: int | None = None,
    support_ray_selection: str | None = None,
) -> dict[str, Any]:
    if max_support_rays is None and support_ray_selection is None:
        return cfg
    cfg = copy.deepcopy(cfg)
    training_cfg = cfg.setdefault("training", {})
    if max_support_rays is not None:
        training_cfg["max_support_rays"] = int(max_support_rays)
    if support_ray_selection is not None:
        training_cfg["support_ray_selection"] = str(support_ray_selection)
    return cfg


def _normalised_decode_mode(mode: str | None, cfg: dict[str, Any]) -> str:
    if mode is None:
        training_cfg = cfg.get("training", {})
        mode = str(training_cfg.get("eval_decode_mode", training_cfg.get("decode_mode", "causal")))
    mode = str(mode).lower()
    if mode in {"one_shot", "oneshot"}:
        return "causal"
    if mode in {"ar", "auto_regressive"}:
        return "autoregressive"
    if mode == "teacher_forced":
        return "causal"
    if mode not in {"causal", "autoregressive"}:
        raise ValueError(f"decode_mode must be 'causal' or 'autoregressive', got {mode!r}")
    return mode


def _density_bin_midpoint(density: torch.Tensor) -> torch.Tensor:
    step = 0.001
    return torch.trunc(density.clamp_min(0.0) / step) * step + 0.5 * step


def _density_material_id(density: torch.Tensor, num_bins: int = 86, max_density: float = 2.5) -> torch.Tensor:
    scaled = density.clamp_min(0.0) / max(float(max_density), 1e-8)
    return torch.clamp((scaled * (int(num_bins) - 1)).long(), 0, int(num_bins) - 1)


def _density_material_id_from_bounds(density: torch.Tensor, bounds: torch.Tensor | None) -> torch.Tensor:
    if bounds is None:
        return _density_material_id(density)
    ids = torch.bucketize(density.contiguous(), bounds[1:].contiguous(), right=True)
    return torch.clamp(ids, 0, int(GEANT4_NUM_MATERIALS) - 1).long()


def _material_id_from_hu(hu: torch.Tensor) -> torch.Tensor:
    bounds = torch.as_tensor(GEANT4_HU_BOUNDS, device=hu.device, dtype=hu.dtype)
    clipped = hu.clamp(float(GEANT4_HU_BOUNDS[0]), float(GEANT4_HU_BOUNDS[-1]))
    ids = torch.bucketize(clipped.contiguous(), bounds[1:].contiguous(), right=True)
    return torch.clamp(ids, 0, int(GEANT4_NUM_MATERIALS) - 1).long()


def _crop_slices(center: float, size: int, half_width: int) -> tuple[slice, slice, slice]:
    center_i = int(round(float(center)))
    target_lo = center_i - int(half_width)
    target_hi = center_i + int(half_width)
    src_lo = max(target_lo, 0)
    src_hi = min(target_hi, int(size))
    dst_lo = src_lo - target_lo
    dst_hi = dst_lo + max(src_hi - src_lo, 0)
    return slice(src_lo, src_hi), slice(dst_lo, dst_hi), slice(target_lo, target_hi)


def _support_crop_1d(mask: torch.Tensor, size: int, margin: int = 0) -> tuple[slice, slice]:
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    if idx.numel() == 0:
        return slice(0, int(size)), slice(0, int(size))
    start = max(int(idx.min().item()) - int(margin), 0)
    stop = min(int(idx.max().item()) + int(margin) + 1, int(size))
    return slice(start, stop), slice(0, stop - start)


def _export_quantized(value: torch.Tensor) -> torch.Tensor:
    """Match the float16 per-sample storage used by proton ray-sequence exports."""
    return value.to(torch.float16).to(value.dtype)


def _radial_stratified_indices(
    radius: torch.Tensor,
    scores: torch.Tensor,
    max_count: int,
    central_idx: int,
    bins: int = 8,
) -> torch.Tensor:
    count = int(radius.numel())
    max_count = min(max(int(max_count), 1), count)
    if max_count >= count:
        return torch.arange(count, device=radius.device)

    order = torch.argsort(radius)
    selected: list[torch.Tensor] = [torch.as_tensor([central_idx], device=radius.device, dtype=torch.long)]
    remaining = max_count - 1
    n_bins = min(max(int(bins), 1), max_count, count)
    for bin_i in range(n_bins):
        start = int(round(bin_i * count / n_bins))
        end = int(round((bin_i + 1) * count / n_bins))
        bin_indices = order[start:end]
        if bin_indices.numel() == 0:
            continue
        quota = remaining // max(n_bins - bin_i, 1)
        if remaining % max(n_bins - bin_i, 1):
            quota += 1
        quota = min(quota, int(bin_indices.numel()), remaining)
        if quota <= 0:
            continue
        bin_scores = scores.index_select(0, bin_indices)
        picked = bin_indices[torch.topk(bin_scores, k=quota, largest=True, sorted=False).indices]
        selected.append(picked)
        remaining -= quota
        if remaining <= 0:
            break

    chosen = torch.unique(torch.cat(selected))
    shortfall = max_count - int(chosen.numel())
    if shortfall > 0:
        keep = torch.ones(count, device=radius.device, dtype=torch.bool)
        keep[chosen] = False
        candidates = torch.arange(count, device=radius.device)[keep]
        if candidates.numel() > 0:
            extra = candidates[
                torch.topk(
                    scores.index_select(0, candidates),
                    k=min(shortfall, int(candidates.numel())),
                    largest=True,
                    sorted=False,
                ).indices
            ]
            chosen = torch.unique(torch.cat((chosen, extra)))

    return torch.sort(chosen[:max_count]).values


class ProtonRaySequenceCorrectionHook(nn.Module):
    """Apply a trained ray-sequence model at the ion sparse deposition hook.

    The model predicts the corrected central-ray dose. The hook converts that to
    a depth-wise ratio and applies the ratio to every ray in the sparse fan before
    the dose engine scatters the samples into the volume. When ``trainable=True``,
    the wrapped model remains part of the dose-engine autograd graph.
    """

    def __init__(
        self,
        model: RaySequenceCorrector,
        cfg: dict[str, Any],
        decode_mode: str | None = None,
        trainable: bool = False,
        patient_density_threshold_g_cm3: float = 0.03,
        ratio_clip: tuple[float, float] | None = (0.02, 50.0),
        match_export_quantization: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.decode_mode = _normalised_decode_mode(decode_mode, cfg)
        self.trainable = bool(trainable)
        self.patient_density_threshold_g_cm3 = float(patient_density_threshold_g_cm3)
        self.ratio_clip = ratio_clip
        self.match_export_quantization = bool(match_export_quantization)
        self.eps = float(cfg.get("dose", {}).get("eps", getattr(model, "eps", 1e-3)))
        bounds = cfg.get("material", {}).get("density_bounds_g_cm3")
        self.material_density_bounds: torch.Tensor | None = None
        if bounds is not None:
            ref = next(model.parameters())
            self.material_density_bounds = torch.as_tensor(bounds, device=ref.device, dtype=ref.dtype)
        for parameter in self.model.parameters():
            parameter.requires_grad_(self.trainable)
        if self.trainable:
            self.model.train()
        else:
            self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        decode_mode: str | None = None,
        trainable: bool = False,
        max_support_rays: int | None = None,
        support_ray_selection: str | None = None,
    ) -> ProtonRaySequenceCorrectionHook:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        cfg = _cfg_with_support_overrides(
            checkpoint["config"],
            max_support_rays=max_support_rays,
            support_ray_selection=support_ray_selection,
        )
        central_dim = int(checkpoint.get("central_dim", 8))
        support_dim = int(checkpoint.get("support_dim", 10))
        model = RaySequenceCorrector.from_config(central_dim, support_dim, cfg)
        load_model_state_dict(model, checkpoint["model_state"])
        model = model.to(device=device, dtype=dtype)
        return cls(model=model, cfg=cfg, decode_mode=decode_mode, trainable=trainable)

    def _build_features(
        self,
        deposition: dict[str, torch.Tensor],
        energy: torch.Tensor,
        spot_sigma_mm: torch.Tensor | None,
        dose_pb_override: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        deposited_energy = deposition["deposited_energy"]
        density = deposition["density_profile_g_cm3"].to(device=deposited_energy.device, dtype=deposited_energy.dtype)
        valid = deposition["valid_steps"].bool()
        edep_to_gy = deposition["edep_to_gy"].to(device=deposited_energy.device, dtype=deposited_energy.dtype)
        if dose_pb_override is None:
            dose_pb = torch.where(
                density > self.patient_density_threshold_g_cm3,
                deposited_energy * edep_to_gy,
                torch.zeros_like(deposited_energy),
            )
        else:
            dose_pb = dose_pb_override.to(device=deposited_energy.device, dtype=deposited_energy.dtype)
            if dose_pb.shape != deposited_energy.shape:
                raise ValueError(
                    "dose_pb_override must match deposited_energy shape "
                    f"{tuple(deposited_energy.shape)}, got {tuple(dose_pb.shape)}"
                )
        t_samples = deposition["t_samples"].to(device=deposited_energy.device, dtype=deposited_energy.dtype)
        center_depth = deposition["center_depth_mm"].to(device=deposited_energy.device, dtype=deposited_energy.dtype)
        kernel_depth = deposition["kernel_depth_mm"].to(device=deposited_energy.device, dtype=deposited_energy.dtype)
        sigma_profile = deposition["sigma_profile_mm"].to(device=deposited_energy.device, dtype=deposited_energy.dtype)
        cell_weights = deposition["cell_weights"].to(device=deposited_energy.device, dtype=deposited_energy.dtype)
        if self.match_export_quantization:
            dose_pb = _export_quantized(dose_pb)
            density = _export_quantized(density)
            t_samples = _export_quantized(t_samples)
            center_depth = _export_quantized(center_depth)
            kernel_depth = _export_quantized(kernel_depth)
            sigma_profile = _export_quantized(sigma_profile)
            cell_weights = _export_quantized(cell_weights)

        fan_x = deposition["fan_x_mm"].to(device=deposited_energy.device, dtype=deposited_energy.dtype)
        fan_y = deposition["fan_y_mm"].to(device=deposited_energy.device, dtype=deposited_energy.dtype)
        radius = torch.sqrt(fan_x.square() + fan_y.square())
        central_idx = int(torch.argmin(radius).detach().cpu())
        support_indices = torch.arange(deposited_energy.shape[0], device=deposited_energy.device)
        max_support_rays = int(self.cfg.get("training", {}).get("max_support_rays", 0) or 0)
        if max_support_rays > 0 and support_indices.numel() > max_support_rays:
            support_scores = dose_pb.amax(dim=1)
            selection = str(self.cfg.get("training", {}).get("support_ray_selection", "topk")).lower()
            if selection == "radial":
                support_indices = _radial_stratified_indices(
                    radius=radius,
                    scores=support_scores,
                    max_count=max_support_rays,
                    central_idx=central_idx,
                )
            else:
                support_indices = torch.sort(
                    torch.topk(support_scores, k=max_support_rays, largest=True, sorted=False).indices
                ).values

        depth_scale = float(self.cfg.get("normalization", {}).get("depth_scale_mm", 100.0))
        energy_scale = float(self.cfg.get("normalization", {}).get("energy_scale_mev", 150.0))
        dose_scale = float(self.cfg.get("dose", {}).get("prescription_dose", 1.0))
        central_pb = dose_pb[central_idx]
        central_density = density[central_idx]
        central_material_bin = _density_bin_midpoint(central_density)
        central_t = t_samples[central_idx]
        central_wepl = center_depth[central_idx]
        central_kdepth = kernel_depth[central_idx]
        beam_energy = energy.to(device=deposited_energy.device, dtype=deposited_energy.dtype).expand_as(central_pb)
        if spot_sigma_mm is None:
            sigma = deposition["sigma_profile_mm"][central_idx].to(
                device=deposited_energy.device,
                dtype=deposited_energy.dtype,
            ).mean(dim=-1)
        else:
            sigma_tensor = spot_sigma_mm.to(device=deposited_energy.device, dtype=deposited_energy.dtype)
            sigma = sigma_tensor.mean().expand_as(central_pb) if sigma_tensor.ndim > 0 else sigma_tensor.expand_as(central_pb)

        central_features = torch.stack(
            (
                torch.log1p(central_pb.clamp_min(0.0) / max(dose_scale, 1e-8)),
                central_density - 1.0,
                central_material_bin,
                central_t / max(depth_scale, 1e-8),
                central_wepl / max(depth_scale, 1e-8),
                central_kdepth / max(depth_scale, 1e-8),
                beam_energy / max(energy_scale, 1e-8),
                sigma / max(depth_scale, 1e-8),
            ),
            dim=-1,
        ).unsqueeze(0).unsqueeze(0)

        dose_pb_support = dose_pb.index_select(0, support_indices)
        density_support = density.index_select(0, support_indices)
        center_depth_support = center_depth.index_select(0, support_indices)
        kernel_depth_support = kernel_depth.index_select(0, support_indices)
        sigma_profile_support = sigma_profile.index_select(0, support_indices)
        cell_weights_support = cell_weights.index_select(0, support_indices)
        valid_support = valid.index_select(0, support_indices)
        radius_support = radius.index_select(0, support_indices)

        support_pb = dose_pb_support.transpose(0, 1)
        support_density = density_support.transpose(0, 1)
        support_material_bin = _density_bin_midpoint(support_density)
        support_radius = radius_support.expand(deposited_energy.shape[1], -1)
        support_wepl = center_depth_support.transpose(0, 1)
        support_kdepth = kernel_depth_support.transpose(0, 1)
        support_sigma_profile = sigma_profile_support.transpose(0, 1)
        cell_weight = cell_weights_support.transpose(0, 1)
        support_features = torch.stack(
            (
                torch.log1p(support_pb.clamp_min(0.0) / max(dose_scale, 1e-8)),
                support_density - 1.0,
                support_material_bin,
                support_radius / max(depth_scale, 1e-8),
                support_wepl / max(depth_scale, 1e-8),
                support_kdepth / max(depth_scale, 1e-8),
                support_sigma_profile[..., 0] / max(depth_scale, 1e-8),
                support_sigma_profile[..., 1] / max(depth_scale, 1e-8),
                cell_weight,
            ),
            dim=-1,
        ).unsqueeze(0).unsqueeze(0)

        support_mask = valid_support.transpose(0, 1).unsqueeze(0).unsqueeze(0)
        bounds = self.material_density_bounds
        if bounds is not None:
            bounds = bounds.to(device=deposited_energy.device, dtype=deposited_energy.dtype)
        central_material_id = _density_material_id_from_bounds(central_density, bounds).unsqueeze(0).unsqueeze(0)
        support_material_id = _density_material_id_from_bounds(support_density, bounds).unsqueeze(0).unsqueeze(0)
        central_dose_pb = central_pb.unsqueeze(0).unsqueeze(0)
        support_dose_pb = support_pb.unsqueeze(0).unsqueeze(0)
        return (
            central_features,
            support_features,
            central_dose_pb,
            support_mask,
            central_material_id,
            support_material_id,
            support_dose_pb,
            support_indices,
        )

    def forward(self, deposition: dict[str, torch.Tensor], **context: Any) -> dict[str, torch.Tensor]:
        if "density_profile_g_cm3" not in deposition or "edep_to_gy" not in deposition:
            raise KeyError("Proton correction hook requires density_profile_g_cm3 and edep_to_gy in deposition")

        grad_context = nullcontext() if self.trainable else torch.no_grad()
        with grad_context:
            features = self._build_features(
                deposition,
                energy=context["energy"],
                spot_sigma_mm=context.get("spot_sigma_mm"),
            )
            *model_args, support_dose_pb, support_indices = features
            outputs = self.model(*model_args, support_dose_pb=support_dose_pb, decode_mode=self.decode_mode)

        residual_mode = getattr(self.model, "residual_mode", "multiplicative")
        corrected_deposited_energy = deposition["deposited_energy"].clone()
        if "support_r" in outputs and residual_mode == "multiplicative":
            ratio = torch.exp(outputs["support_r"].squeeze(0).squeeze(0)).transpose(0, 1)
        elif "support_dose_hat" in outputs:
            corrected = outputs["support_dose_hat"].squeeze(0).squeeze(0).clamp_min(0.0)
            if residual_mode == "additive":
                edep_to_gy = deposition["edep_to_gy"].to(device=corrected.device, dtype=corrected.dtype)
                if edep_to_gy.ndim == 0:
                    edep_to_gy = edep_to_gy.expand_as(deposition["deposited_energy"])
                edep_to_gy = edep_to_gy.index_select(0, support_indices)
                density = deposition["density_profile_g_cm3"].to(
                    device=corrected.device, dtype=corrected.dtype
                ).index_select(0, support_indices)
                patient_mask = density > self.patient_density_threshold_g_cm3
                corrected_support_energy = torch.where(
                    patient_mask,
                    corrected.transpose(0, 1) / edep_to_gy.clamp_min(torch.finfo(corrected.dtype).tiny),
                    torch.zeros_like(edep_to_gy),
                )
                corrected_deposited_energy.index_copy_(
                    0,
                    support_indices,
                    corrected_support_energy.to(dtype=corrected_deposited_energy.dtype),
                )
                ratio = None
            else:
                support_pb = support_dose_pb.squeeze(0).squeeze(0)
                ratio = ((corrected + self.eps) / (support_pb + self.eps).clamp_min(self.eps)).transpose(0, 1)
        else:
            central_pb = features[2].squeeze(0).squeeze(0)
            corrected = outputs["dose_hat"].squeeze(0).squeeze(0).clamp_min(0.0)
            ratio = torch.where(
                central_pb > 0.0,
                corrected / central_pb.clamp_min(torch.finfo(central_pb.dtype).tiny),
                torch.ones_like(central_pb),
            )
            support_indices = torch.arange(deposition["deposited_energy"].shape[0], device=ratio.device)
            ratio = ratio.unsqueeze(0).expand(support_indices.numel(), -1)
        if ratio is not None and self.ratio_clip is not None:
            lo, hi = self.ratio_clip
            ratio = ratio.clamp(float(lo), float(hi))
        if ratio is not None:
            corrected_deposited_energy.index_copy_(
                0,
                support_indices,
                deposition["deposited_energy"].index_select(0, support_indices) * ratio,
            )
        corrected_deposited_energy = torch.where(
            deposition["valid_steps"].bool(),
            corrected_deposited_energy,
            torch.zeros_like(corrected_deposited_energy),
        )
        return {**deposition, "deposited_energy": corrected_deposited_energy}


class ProtonBeamletVolumeCorrectionHook(ProtonRaySequenceCorrectionHook):
    """Single-pass beamlet-volume correction hook.

    Uses ``support_r`` (log correction ratio per depth sample) from the model
    directly, matching the per-beamlet volume training objective.  No pre-computed
    PB volume is required — one engine pass suffices.
    """

    def __init__(
        self,
        model: RaySequenceCorrector,
        cfg: dict[str, Any],
        decode_mode: str | None = None,
        log_ratio_clip: float | None = None,
        patient_density_threshold_g_cm3: float = 0.03,
        match_export_quantization: bool = True,
    ) -> None:
        super().__init__(
            model=model,
            cfg=cfg,
            decode_mode=decode_mode,
            trainable=False,
            patient_density_threshold_g_cm3=patient_density_threshold_g_cm3,
            ratio_clip=None,
            match_export_quantization=match_export_quantization,
        )
        if log_ratio_clip is None:
            log_ratio_clip = float(cfg.get("loss", {}).get("log_ratio_clip", 1.5))
        self.log_ratio_clip = float(log_ratio_clip)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        decode_mode: str | None = None,
        max_support_rays: int | None = None,
        support_ray_selection: str | None = None,
        **_ignored: Any,
    ) -> ProtonBeamletVolumeCorrectionHook:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        cfg = _cfg_with_support_overrides(
            checkpoint["config"],
            max_support_rays=max_support_rays,
            support_ray_selection=support_ray_selection,
        )
        central_dim = int(checkpoint.get("central_dim", 8))
        support_dim = int(checkpoint.get("support_dim", 10))
        model = RaySequenceCorrector.from_config(central_dim, support_dim, cfg)
        load_model_state_dict(model, checkpoint["model_state"])
        model = model.to(device=device, dtype=dtype)
        return cls(model=model, cfg=cfg, decode_mode=decode_mode)

    def forward(self, deposition: dict[str, torch.Tensor], **context: Any) -> dict[str, torch.Tensor]:
        if "density_profile_g_cm3" not in deposition or "edep_to_gy" not in deposition:
            raise KeyError("Proton correction hook requires density_profile_g_cm3 and edep_to_gy in deposition")

        with torch.no_grad():
            features = self._build_features(
                deposition,
                energy=context["energy"],
                spot_sigma_mm=context.get("spot_sigma_mm"),
            )
            *model_args, support_dose_pb, support_indices = features
            outputs = self.model(*model_args, support_dose_pb=support_dose_pb, decode_mode=self.decode_mode)

        residual_mode = getattr(self.model, "residual_mode", "multiplicative")
        if "support_r" not in outputs:
            raise RuntimeError(
                "ProtonBeamletVolumeCorrectionHook requires a checkpoint with "
                "model.predict_support=True (support_r output missing)"
            )

        corrected_deposited_energy = deposition["deposited_energy"].clone()
        if residual_mode == "additive":
            corrected = outputs["support_dose_hat"].squeeze(0).squeeze(0).clamp_min(0.0)
            edep_to_gy = deposition["edep_to_gy"].to(device=corrected.device, dtype=corrected.dtype)
            if edep_to_gy.ndim == 0:
                edep_to_gy = edep_to_gy.expand_as(deposition["deposited_energy"])
            edep_to_gy = edep_to_gy.index_select(0, support_indices)
            density = deposition["density_profile_g_cm3"].to(device=corrected.device, dtype=corrected.dtype).index_select(
                0, support_indices
            )
            patient_mask = density > self.patient_density_threshold_g_cm3
            corrected_support_energy = torch.where(
                patient_mask,
                corrected.transpose(0, 1) / edep_to_gy.clamp_min(torch.finfo(corrected.dtype).tiny),
                torch.zeros_like(edep_to_gy),
            )
            corrected_deposited_energy.index_copy_(
                0,
                support_indices,
                corrected_support_energy.to(dtype=corrected_deposited_energy.dtype),
            )
        else:
            support_r = outputs["support_r"].squeeze(0).squeeze(0)  # [depth, support]
            ratio = torch.exp(support_r.clamp(-self.log_ratio_clip, self.log_ratio_clip)).transpose(0, 1)
            corrected_deposited_energy.index_copy_(
                0,
                support_indices,
                deposition["deposited_energy"].index_select(0, support_indices) * ratio,
            )
        corrected_deposited_energy = torch.where(
            deposition["valid_steps"].bool(),
            corrected_deposited_energy,
            torch.zeros_like(corrected_deposited_energy),
        )
        return {**deposition, "deposited_energy": corrected_deposited_energy}


_D4_SYMMETRIES: tuple[tuple[bool, int], ...] = (
    (False, 0),
    (False, 1),
    (False, 2),
    (False, 3),
    (True, 0),
    (True, 1),
    (True, 2),
    (True, 3),
)


def _d4_apply(x: torch.Tensor, flip: bool, k_rot: int) -> torch.Tensor:
    """Apply a D4 element to a [..., H, W] tensor (H=dim-3, W=dim-4)."""
    if flip:
        x = x.flip(3)
    if k_rot:
        x = torch.rot90(x, k=k_rot, dims=(3, 4))
    return x


def _d4_inverse(x: torch.Tensor, flip: bool, k_rot: int) -> torch.Tensor:
    """Apply the inverse D4 element (reflections are involutions)."""
    inv_k = k_rot if flip else (4 - k_rot) % 4
    return _d4_apply(x, flip, inv_k)


class ProtonFanGridCorrectionHook(nn.Module):
    """Apply a dense fan-grid convolution model to every active sparse ray."""

    def __init__(
        self,
        model: FanGridConvCorrector,
        cfg: dict[str, Any],
        trainable: bool = False,
        patient_density_threshold_g_cm3: float = 0.03,
        match_export_quantization: bool = True,
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.trainable = bool(trainable)
        self.augment = bool(augment)
        self.patient_density_threshold_g_cm3 = float(patient_density_threshold_g_cm3)
        self.match_export_quantization = bool(match_export_quantization)
        self.eps = float(cfg.get("dose", {}).get("eps", getattr(model, "eps", 1e-3)))
        self._material_id_volume: torch.Tensor | None = None
        self._material_resolution_zyx: tuple[float, float, float] | None = None
        bounds = cfg.get("material", {}).get("density_bounds_g_cm3")
        self.material_density_bounds: torch.Tensor | None = None
        if bounds is not None:
            ref = next(model.parameters())
            self.material_density_bounds = torch.as_tensor(bounds, device=ref.device, dtype=ref.dtype)
        for parameter in self.model.parameters():
            parameter.requires_grad_(self.trainable)
        self.model.train(self.trainable)

    def set_hu_volume(
        self,
        hu_volume: torch.Tensor | None,
        resolution_zyx: tuple[float, float, float] | None,
    ) -> None:
        if hu_volume is None or resolution_zyx is None:
            self._material_id_volume = None
            self._material_resolution_zyx = None
            return
        ref = next(self.model.parameters())
        with torch.no_grad():
            material_id = _material_id_from_hu(hu_volume.to(device=ref.device, dtype=ref.dtype)).to(torch.long)
        self._material_id_volume = material_id
        self._material_resolution_zyx = tuple(float(v) for v in resolution_zyx)

    def _sample_material_id_profile(self, profile: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self._material_id_volume is None or self._material_resolution_zyx is None:
            return profile
        t_samples = profile["t_samples"]
        ray_dirs = profile["ray_dirs"].to(device=t_samples.device, dtype=t_samples.dtype)
        source = profile["source_position_mm"].to(device=t_samples.device, dtype=t_samples.dtype)
        coords = source.view(1, 1, 3) + ray_dirs[:, None, :] * t_samples[..., None]
        resolution = torch.as_tensor(self._material_resolution_zyx, device=t_samples.device, dtype=t_samples.dtype)
        index = torch.round(coords / resolution.view(1, 1, 3)).long()
        shape = self._material_id_volume.shape
        inside = (
            (index[..., 0] >= 0)
            & (index[..., 0] < int(shape[0]))
            & (index[..., 1] >= 0)
            & (index[..., 1] < int(shape[1]))
            & (index[..., 2] >= 0)
            & (index[..., 2] < int(shape[2]))
        )
        if "valid_steps" in profile:
            inside = inside & profile["valid_steps"].to(device=t_samples.device).bool()
        clipped = torch.stack(
            (
                index[..., 0].clamp(0, int(shape[0]) - 1),
                index[..., 1].clamp(0, int(shape[1]) - 1),
                index[..., 2].clamp(0, int(shape[2]) - 1),
            ),
            dim=-1,
        )
        sampled = self._material_id_volume[
            clipped[..., 0],
            clipped[..., 1],
            clipped[..., 2],
        ].to(device=t_samples.device)
        sampled = torch.where(inside, sampled, torch.zeros_like(sampled))
        return {**profile, "material_id_profile": sampled}

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        trainable: bool = False,
        match_export_quantization: bool = False,
        **_ignored: Any,
    ) -> ProtonFanGridCorrectionHook:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        cfg = checkpoint["config"]
        model = FanGridConvCorrector.from_config(int(checkpoint.get("fan_input_dim", 8)), cfg)
        load_model_state_dict(model, checkpoint["model_state"])
        model = model.to(device=device, dtype=dtype)
        return cls(
            model=model,
            cfg=cfg,
            trainable=trainable,
            match_export_quantization=match_export_quantization,
        )

    def _grid_indices(
        self,
        deposition: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if "fan_grid_i" in deposition and "fan_grid_j" in deposition and "fan_grid_shape" in deposition:
            grid_i = deposition["fan_grid_i"].long()
            grid_j = deposition["fan_grid_j"].long()
            h, w = deposition["fan_grid_shape"]
            return grid_i, grid_j, int(h), int(w)

        # Fallback for older engine outputs: recover the native 0.125-sigma grid
        # from physical fan offsets and cell widths.
        step = 0.125
        fan_x = deposition["fan_x_mm"]
        fan_y = deposition["fan_y_mm"]
        cell_x = deposition["cell_width_x_mm"].to(device=fan_x.device, dtype=fan_x.dtype)
        cell_y = deposition["cell_width_y_mm"].to(device=fan_y.device, dtype=fan_y.dtype)
        grid_x = fan_x / (cell_x / step).clamp_min(torch.finfo(fan_x.dtype).tiny)
        grid_y = fan_y / (cell_y / step).clamp_min(torch.finfo(fan_y.dtype).tiny)
        n_max = int(torch.ceil(torch.maximum(grid_x.abs().amax(), grid_y.abs().amax()) / step).detach().cpu())
        grid_i = torch.round(grid_x / step).long() + n_max
        grid_j = torch.round(grid_y / step).long() + n_max
        side = 2 * n_max + 1
        return grid_i, grid_j, side, side

    def _build_dense_features(
        self,
        deposition: dict[str, torch.Tensor],
        energy: torch.Tensor,
        spot_sigma_mm: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        deposited_energy = deposition["deposited_energy"]
        device = deposited_energy.device
        dtype = deposited_energy.dtype
        density = deposition["density_profile_g_cm3"].to(device=device, dtype=dtype)
        valid = deposition["valid_steps"].bool()
        edep_to_gy = deposition["edep_to_gy"].to(device=device, dtype=dtype)
        if edep_to_gy.ndim == 0:
            edep_to_gy = edep_to_gy.expand_as(deposited_energy)
        patient_valid = valid & (density > self.patient_density_threshold_g_cm3)
        dose_pb = torch.where(
            patient_valid,
            deposited_energy * edep_to_gy,
            torch.zeros_like(deposited_energy),
        )
        if self.match_export_quantization:
            dose_pb = _export_quantized(dose_pb)
            density = _export_quantized(density)

        t_samples = deposition["t_samples"].to(device=device, dtype=dtype)
        center_depth = deposition["center_depth_mm"].to(device=device, dtype=dtype)
        kernel_depth = deposition["kernel_depth_mm"].to(device=device, dtype=dtype)
        sigma_profile = deposition["sigma_profile_mm"].to(device=device, dtype=dtype)
        cell_weights = deposition["cell_weights"].to(device=device, dtype=dtype)
        fan_x = deposition["fan_x_mm"].to(device=device, dtype=dtype)
        fan_y = deposition["fan_y_mm"].to(device=device, dtype=dtype)
        if self.match_export_quantization:
            t_samples = _export_quantized(t_samples)
            center_depth = _export_quantized(center_depth)
            kernel_depth = _export_quantized(kernel_depth)
            sigma_profile = _export_quantized(sigma_profile)
            cell_weights = _export_quantized(cell_weights)

        grid_i, grid_j, height, width = self._grid_indices(deposition)
        grid_i = grid_i.to(device=device)
        grid_j = grid_j.to(device=device)
        flat_idx = grid_i * int(width) + grid_j

        depth_count = int(deposited_energy.shape[1])
        flat_count = int(height) * int(width)
        depth_scale = float(self.cfg.get("normalization", {}).get("depth_scale_mm", 100.0))
        if "material_id_profile" in deposition:
            material_id = deposition["material_id_profile"].to(device=device, dtype=torch.long)
        else:
            bounds = self.material_density_bounds
            if bounds is not None:
                bounds = bounds.to(device=device, dtype=dtype)
            material_id = _density_material_id_from_bounds(density, bounds)
        material_id = torch.where(patient_valid, material_id.to(device=device, dtype=torch.long), torch.zeros_like(material_id))
        radius = torch.sqrt(fan_x.square() + fan_y.square()).unsqueeze(1).expand_as(dose_pb)

        feature_list = (
            density - 1.0,
            radius / max(depth_scale, 1e-8),
            t_samples.expand_as(dose_pb) / max(depth_scale, 1e-8),
            center_depth / max(depth_scale, 1e-8),
            kernel_depth / max(depth_scale, 1e-8),
            sigma_profile[..., 0] / max(depth_scale, 1e-8),
            sigma_profile[..., 1] / max(depth_scale, 1e-8),
            cell_weights,
        )
        sparse_features = torch.stack(feature_list, dim=0)  # [C, N, D]
        channels = sparse_features.shape[0]
        dense_features_flat = sparse_features.new_zeros((channels, depth_count, flat_count))
        dense_features_flat.index_copy_(2, flat_idx, sparse_features.permute(0, 2, 1))
        dense_dose_flat = dose_pb.new_zeros((1, depth_count, flat_count))
        dense_dose_flat.index_copy_(2, flat_idx, dose_pb.transpose(0, 1).unsqueeze(0))
        dense_valid_flat = torch.zeros((1, depth_count, flat_count), device=device, dtype=torch.bool)
        dense_valid_flat.index_copy_(2, flat_idx, patient_valid.transpose(0, 1).unsqueeze(0))
        dense_material_flat = torch.zeros((1, depth_count, flat_count), device=device, dtype=torch.long)
        dense_material_flat.index_copy_(2, flat_idx, material_id.transpose(0, 1).unsqueeze(0))
        fan_mask_flat = torch.zeros((1, 1, flat_count), device=device, dtype=torch.bool)
        fan_mask_flat.index_fill_(2, flat_idx, True)

        features = dense_features_flat.view(1, channels, depth_count, height, width)
        dense_dose = dense_dose_flat.view(1, 1, depth_count, height, width)
        dense_valid = dense_valid_flat.view(1, 1, depth_count, height, width)
        dense_material = dense_material_flat.view(1, 1, depth_count, height, width)
        fan_mask = fan_mask_flat.view(1, 1, 1, height, width)
        return features, dense_dose, dense_valid, fan_mask, flat_idx, edep_to_gy, patient_valid, dense_material

    def _dense_feature_group_key(self, deposition: dict[str, torch.Tensor]) -> tuple[int, int, int]:
        _grid_i, _grid_j, height, width = self._grid_indices(deposition)
        return int(deposition["deposited_energy"].shape[1]), int(height), int(width)

    def _build_dense_feature_batch(
        self,
        depositions: list[dict[str, torch.Tensor]],
        contexts: list[dict[str, Any]],
        indices: list[int],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        torch.Tensor,
    ]:
        if not indices:
            raise ValueError("indices must be non-empty")

        first = depositions[indices[0]]
        depth_count, height, width = self._dense_feature_group_key(first)
        flat_count = int(height) * int(width)
        batch_count = len(indices)
        deposited_ref = first["deposited_energy"]
        device = deposited_ref.device
        dtype = deposited_ref.dtype
        depth_scale = float(self.cfg.get("normalization", {}).get("depth_scale_mm", 100.0))

        features_flat = deposited_ref.new_zeros((batch_count, 8, depth_count, flat_count))
        dense_dose_flat = deposited_ref.new_zeros((batch_count, 1, depth_count, flat_count))
        dense_valid_flat = torch.zeros((batch_count, 1, depth_count, flat_count), device=device, dtype=torch.bool)
        dense_material_flat = torch.zeros((batch_count, 1, depth_count, flat_count), device=device, dtype=torch.long)
        fan_mask_flat = torch.zeros((batch_count, 1, 1, flat_count), device=device, dtype=torch.bool)
        flat_indices: list[torch.Tensor] = []
        edep_to_gy_values: list[torch.Tensor] = []
        patient_valid_values: list[torch.Tensor] = []

        for local_i, item_idx in enumerate(indices):
            deposition = depositions[item_idx]
            deposited_energy = deposition["deposited_energy"]
            if deposited_energy.shape[1] != depth_count:
                raise ValueError("all depositions in a fan-grid batch group must have the same depth count")

            density = deposition["density_profile_g_cm3"].to(device=device, dtype=dtype)
            valid = deposition["valid_steps"].to(device=device).bool()
            edep_to_gy = deposition["edep_to_gy"].to(device=device, dtype=dtype)
            if edep_to_gy.ndim == 0:
                edep_to_gy = edep_to_gy.expand_as(deposited_energy)
            patient_valid = valid & (density > self.patient_density_threshold_g_cm3)
            dose_pb = torch.where(
                patient_valid,
                deposited_energy.to(device=device, dtype=dtype) * edep_to_gy,
                torch.zeros_like(deposited_energy, device=device, dtype=dtype),
            )
            if self.match_export_quantization:
                dose_pb = _export_quantized(dose_pb)
                density = _export_quantized(density)

            t_samples = deposition["t_samples"].to(device=device, dtype=dtype)
            center_depth = deposition["center_depth_mm"].to(device=device, dtype=dtype)
            kernel_depth = deposition["kernel_depth_mm"].to(device=device, dtype=dtype)
            sigma_profile = deposition["sigma_profile_mm"].to(device=device, dtype=dtype)
            cell_weights = deposition["cell_weights"].to(device=device, dtype=dtype)
            fan_x = deposition["fan_x_mm"].to(device=device, dtype=dtype)
            fan_y = deposition["fan_y_mm"].to(device=device, dtype=dtype)
            if self.match_export_quantization:
                t_samples = _export_quantized(t_samples)
                center_depth = _export_quantized(center_depth)
                kernel_depth = _export_quantized(kernel_depth)
                sigma_profile = _export_quantized(sigma_profile)
                cell_weights = _export_quantized(cell_weights)

            grid_i, grid_j, item_height, item_width = self._grid_indices(deposition)
            if int(item_height) != height or int(item_width) != width:
                raise ValueError("all depositions in a fan-grid batch group must have the same fan-grid shape")
            flat_idx = grid_i.to(device=device).long() * int(width) + grid_j.to(device=device).long()

            if "material_id_profile" in deposition:
                material_id = deposition["material_id_profile"].to(device=device, dtype=torch.long)
            else:
                bounds = self.material_density_bounds
                if bounds is not None:
                    bounds = bounds.to(device=device, dtype=dtype)
                material_id = _density_material_id_from_bounds(density, bounds)
            material_id = torch.where(
                patient_valid,
                material_id.to(device=device, dtype=torch.long),
                torch.zeros_like(material_id, device=device, dtype=torch.long),
            )
            radius = torch.sqrt(fan_x.square() + fan_y.square()).unsqueeze(1).expand_as(dose_pb)
            sparse_features = torch.stack(
                (
                    density - 1.0,
                    radius / max(depth_scale, 1e-8),
                    t_samples.expand_as(dose_pb) / max(depth_scale, 1e-8),
                    center_depth / max(depth_scale, 1e-8),
                    kernel_depth / max(depth_scale, 1e-8),
                    sigma_profile[..., 0] / max(depth_scale, 1e-8),
                    sigma_profile[..., 1] / max(depth_scale, 1e-8),
                    cell_weights,
                ),
                dim=0,
            )

            features_flat[local_i].index_copy_(2, flat_idx, sparse_features.permute(0, 2, 1))
            dense_dose_flat[local_i].index_copy_(2, flat_idx, dose_pb.transpose(0, 1).unsqueeze(0))
            dense_valid_flat[local_i].index_copy_(2, flat_idx, patient_valid.transpose(0, 1).unsqueeze(0))
            dense_material_flat[local_i].index_copy_(2, flat_idx, material_id.transpose(0, 1).unsqueeze(0))
            fan_mask_flat[local_i].index_fill_(2, flat_idx, True)
            flat_indices.append(flat_idx)
            edep_to_gy_values.append(edep_to_gy)
            patient_valid_values.append(patient_valid)

        return (
            features_flat.view(batch_count, 8, depth_count, height, width),
            dense_dose_flat.view(batch_count, 1, depth_count, height, width),
            dense_valid_flat.view(batch_count, 1, depth_count, height, width),
            fan_mask_flat.view(batch_count, 1, 1, height, width),
            flat_indices,
            edep_to_gy_values,
            patient_valid_values,
            dense_material_flat.view(batch_count, 1, depth_count, height, width),
        )

    def forward(self, deposition: dict[str, torch.Tensor], **context: Any) -> dict[str, torch.Tensor]:
        if "deposited_energy" not in deposition and "sampled_lines" in deposition:
            return self._sample_material_id_profile(deposition)
        if "density_profile_g_cm3" not in deposition or "edep_to_gy" not in deposition:
            raise KeyError("Proton fan-grid correction hook requires density_profile_g_cm3 and edep_to_gy")

        grad_context = nullcontext() if self.trainable else torch.inference_mode()
        with grad_context:
            features, dense_dose, dense_valid, fan_mask, flat_idx, edep_to_gy, patient_valid, dense_material = self._build_dense_features(
                deposition,
                energy=context["energy"],
                spot_sigma_mm=context.get("spot_sigma_mm"),
            )
            aug_flip, aug_k = False, 0
            if self.augment and self.model.training:
                sym = int(torch.randint(0, 8, (1,)).item())
                aug_flip, aug_k = _D4_SYMMETRIES[sym]
                if aug_flip or aug_k:
                    features = _d4_apply(features, aug_flip, aug_k)
                    dense_dose = _d4_apply(dense_dose, aug_flip, aug_k)
                    dense_valid = _d4_apply(dense_valid, aug_flip, aug_k)
                    fan_mask = _d4_apply(fan_mask, aug_flip, aug_k)
                    dense_material = _d4_apply(dense_material, aug_flip, aug_k)
            outputs = self.model(
                features,
                dense_dose,
                dense_valid,
                fan_mask,
                material_id=dense_material,
                energy=context["energy"],
            )
            if aug_flip or aug_k:
                outputs = {**outputs, "dose_hat": _d4_inverse(outputs["dose_hat"], aug_flip, aug_k)}

        dose_hat = outputs["dose_hat"].squeeze(0).squeeze(0)
        depth_count = int(deposition["deposited_energy"].shape[1])
        corrected_dose = dose_hat.reshape(depth_count, -1).index_select(1, flat_idx).transpose(0, 1)
        corrected_energy = corrected_dose / edep_to_gy.clamp_min(torch.finfo(corrected_dose.dtype).tiny)
        corrected_energy = torch.where(
            patient_valid,
            corrected_energy.to(dtype=deposition["deposited_energy"].dtype),
            torch.zeros_like(deposition["deposited_energy"]),
        )
        return {**deposition, "deposited_energy": corrected_energy}

    def forward_batch(
        self,
        depositions: list[dict[str, torch.Tensor]],
        contexts: list[dict[str, Any]],
    ) -> list[dict[str, torch.Tensor]]:
        if not depositions:
            return []
        for deposition in depositions:
            if "density_profile_g_cm3" not in deposition or "edep_to_gy" not in deposition:
                raise KeyError("Proton fan-grid correction hook requires density_profile_g_cm3 and edep_to_gy")

        grad_context = nullcontext() if self.trainable else torch.inference_mode()
        with grad_context:
            groups: dict[tuple[int, int, int], list[int]] = {}
            for item_idx, deposition in enumerate(depositions):
                key = self._dense_feature_group_key(deposition)
                groups.setdefault(key, []).append(item_idx)

            corrected: list[dict[str, torch.Tensor] | None] = [None] * len(depositions)
            for indices in groups.values():
                (
                    features,
                    dense_dose,
                    dense_valid,
                    fan_mask,
                    flat_indices,
                    edep_to_gy_values,
                    patient_valid_values,
                    dense_material,
                ) = self._build_dense_feature_batch(depositions, contexts, indices)
                energy = torch.stack(
                    [
                        torch.as_tensor(
                            contexts[item_idx]["energy"],
                            device=features.device,
                            dtype=features.dtype,
                        ).reshape(())
                        for item_idx in indices
                    ],
                    dim=0,
                )
                aug_flip, aug_k = False, 0
                if self.augment and self.model.training:
                    sym = int(torch.randint(0, 8, (1,)).item())
                    aug_flip, aug_k = _D4_SYMMETRIES[sym]
                    if aug_flip or aug_k:
                        features = _d4_apply(features, aug_flip, aug_k)
                        dense_dose = _d4_apply(dense_dose, aug_flip, aug_k)
                        dense_valid = _d4_apply(dense_valid, aug_flip, aug_k)
                        fan_mask = _d4_apply(fan_mask, aug_flip, aug_k)
                        dense_material = _d4_apply(dense_material, aug_flip, aug_k)
                outputs = self.model(
                    features,
                    dense_dose,
                    dense_valid,
                    fan_mask,
                    material_id=dense_material,
                    energy=energy,
                )
                dose_hat = outputs["dose_hat"]
                if aug_flip or aug_k:
                    dose_hat = _d4_inverse(dose_hat, aug_flip, aug_k)

                for local_i, item_idx in enumerate(indices):
                    deposition = depositions[item_idx]
                    flat_idx = flat_indices[local_i]
                    edep_to_gy = edep_to_gy_values[local_i]
                    patient_valid = patient_valid_values[local_i]
                    depth_count = int(deposition["deposited_energy"].shape[1])
                    corrected_dose = dose_hat[local_i, 0].reshape(depth_count, -1).index_select(1, flat_idx).transpose(0, 1)
                    corrected_energy = corrected_dose / edep_to_gy.clamp_min(torch.finfo(corrected_dose.dtype).tiny)
                    corrected_energy = torch.where(
                        patient_valid,
                        corrected_energy.to(dtype=deposition["deposited_energy"].dtype),
                        torch.zeros_like(deposition["deposited_energy"]),
                    )
                    corrected[item_idx] = {**deposition, "deposited_energy": corrected_energy}

        if any(item is None for item in corrected):
            raise RuntimeError("Fan-grid correction batch did not produce all requested depositions")
        return [item for item in corrected if item is not None]


# Indices into the 8-channel BEV feature tensor (see _build_bev_features) of the
# two *signed lateral coordinate* channels: the h/w offset of each voxel from the
# beam axis. These are positional encodings, not physical fields. Flipping them as
# plain fields (what _d4_apply does) makes the offset run backwards w.r.t. the
# spatial axis -- a convention that never occurs at inference (offsets always
# increase with position), so naively-augmented inputs are out-of-distribution in
# exactly these channels. They must instead keep the canonical sign convention,
# i.e. be negated according to the net reflection parity of each lateral axis.
_FEATURE_H_OFFSET_CH = 4
_FEATURE_W_OFFSET_CH = 5


def _d4_apply_features(x: torch.Tensor, flip: bool, k_rot: int) -> torch.Tensor:
    """Apply a D4 symmetry to the BEV feature tensor, sign-correcting the signed
    lateral-coordinate channels so augmented inputs stay in-distribution.

    Restricted to ``k_rot in {0, 2}`` (the repvgg_unet symmetry set, no 90-degree
    rotations). A 90-degree rotation would additionally have to *swap* the h/w
    offset channels and the sigma_h/sigma_w conditioning, which is not handled
    here -- so it raises rather than silently corrupting the features.
    """
    if k_rot % 2 != 0:
        raise ValueError(
            "_d4_apply_features only supports k_rot in {0, 2}; 90-degree rotations "
            "require swapping the h/w offset channels and sigma conditioning."
        )
    x = _d4_apply(x, flip, k_rot)
    parity_h = bool(flip) ^ (k_rot == 2)  # H axis (dim 3) net reflected
    parity_w = (k_rot == 2)               # W axis (dim 4) net reflected
    if parity_h or parity_w:
        x = x.clone()
        if parity_h:
            x[:, _FEATURE_H_OFFSET_CH].neg_()
        if parity_w:
            x[:, _FEATURE_W_OFFSET_CH].neg_()
    return x



#: Input-channel count per BEV feature set. ``v1`` is the historical 8-channel
#: stack and MUST stay the default: every checkpoint trained before 2026-07-26
#: was fitted to it, and ``from_checkpoint`` falls back to it when an older
#: checkpoint carries no ``bev_feature_set`` entry.
#:
#: ``v2`` appends one channel, ``(weq - R_peak(E)) / depth_scale``: water-equivalent
#: depth measured relative to *this beamlet's own* Bragg peak. The v1 stack gives
#: only absolute WEQ plus energy as a global embedding, so the net has to fuse a
#: spatial channel with a per-beamlet bias through the conv stack to work out
#: whether a voxel is proximal or distal to the peak. That is the coordinate the
#: residual physics is stationary in, and the cost of not having it shows up as an
#: energy-dependent error: corr(MAE, energy) = -0.34, with 32-81 MeV beamlets at
#: MAE 0.0084 against 0.0053 for 160-201 MeV.
#: ``v3`` appends a second channel, the lateral gradient magnitude of WEQ. Where WEQ
#: varies across the spot at a given depth, protons at different lateral positions have
#: different residual range and the Bragg peak smears -- "range mixing". A pencil-beam
#: model carries one depth-dose curve per beamlet and structurally cannot represent it,
#: so this is baseline-side error the net currently has no way to see. Measured
#: 2026-07-26: the correction net removes 69.3% of baseline MAE but the low-energy
#: deficit (1.61x low/high spread) is already present in the analytic baseline before
#: the net runs, and finer lateral sub-beam sampling does NOT touch it.
BEV_FEATURE_CHANNELS = {"v1": 8, "v2": 9, "v3": 10}

#: Test-time augmentation set: (flip, k_rot) over the four lateral reflections, i.e.
#: identity / flip-H / flip-both / flip-W. 90-degree rotations are excluded because the
#: BEV crop is anisotropic (26x74) and would additionally require swapping the h/w offset
#: channels and the sigma conditioning -- _d4_apply_features raises on odd k_rot.
TTA_TRANSFORMS = ((False, 0), (True, 0), (False, 2), (True, 2))


def _lateral_weq_gradient(weq: torch.Tensor) -> torch.Tensor:
    """Gradient magnitude of WEQ across the two lateral axes of a BEV crop.

    ``weq`` is ``[N, D, H, W]``; the gradient is taken over H and W (never depth) in
    index space, with one-sided differences at the borders so the result keeps the
    input shape. Axes of extent 1 contribute zero rather than erroring.
    """

    def _central(x: torch.Tensor, dim: int) -> torch.Tensor:
        if x.shape[dim] < 2:
            return torch.zeros_like(x)
        out = torch.zeros_like(x)
        upper = x.narrow(dim, 2, x.shape[dim] - 2)
        lower = x.narrow(dim, 0, x.shape[dim] - 2)
        if x.shape[dim] > 2:
            out.narrow(dim, 1, x.shape[dim] - 2).copy_(0.5 * (upper - lower))
        # One-sided at the two borders.
        out.narrow(dim, 0, 1).copy_(x.narrow(dim, 1, 1) - x.narrow(dim, 0, 1))
        out.narrow(dim, x.shape[dim] - 1, 1).copy_(
            x.narrow(dim, x.shape[dim] - 1, 1) - x.narrow(dim, x.shape[dim] - 2, 1)
        )
        return out

    gh = _central(weq, -2)
    gw = _central(weq, -1)
    return torch.sqrt(gh.square() + gw.square())


class ProtonDenseBevCorrectionHook(nn.Module):
    """Apply a dense BEV FanGridConvCorrector inside IonDoseEngine's dense path."""

    def __init__(
        self,
        model: FanGridConvCorrector,
        cfg: dict[str, Any],
        bev_crop_hw: int = 64,
        bev_crop_h: int | None = None,
        bev_crop_w: int | None = None,
        max_inference_batch_items: int = 8,
        inference_amp: bool = False,
        trainable: bool = False,
        bev_feature_set: str = "v1",
        tta: bool = False,
        pad_inference_batch: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.bev_crop_hw = int(bev_crop_hw)
        self.bev_crop_h = int(self.bev_crop_hw if bev_crop_h is None else bev_crop_h)
        self.bev_crop_w = int(self.bev_crop_hw if bev_crop_w is None else bev_crop_w)
        self.max_inference_batch_items = max(1, int(max_inference_batch_items))
        self.inference_amp = bool(inference_amp)
        self.trainable = bool(trainable)
        self.tta = bool(tta)
        # Pad the final ragged chunk up to max_inference_batch_items with zero items so the
        # compiled graph only ever sees one batch size. Without it the tail chunk is a new
        # shape and torch.compile recompiles *inside* the metered path. Safe because the
        # normalisation here is GroupNorm, whose statistics are per-sample -- a padded row
        # cannot perturb a real one -- and because the scatter-back loop iterates meta_items,
        # so padded rows are dropped without ever being read. Inference only: in training the
        # padded rows would still consume memory and backward time for no gradient.
        self.pad_inference_batch = bool(pad_inference_batch) and not self.trainable
        self.bev_feature_set = str(bev_feature_set).lower()
        if self.bev_feature_set not in BEV_FEATURE_CHANNELS:
            raise ValueError(
                f"Unknown bev_feature_set {bev_feature_set!r}; "
                f"expected one of {sorted(BEV_FEATURE_CHANNELS)}"
            )
        self._material_id_volume: torch.Tensor | None = None
        self._peak_depth_cache: dict[float, float] = {}
        for parameter in self.model.parameters():
            parameter.requires_grad_(self.trainable)
        self.model.train(self.trainable)

    def set_bev_crop_half_widths(self, crop_h: int, crop_w: int) -> None:
        self.bev_crop_h = max(1, int(crop_h))
        self.bev_crop_w = max(1, int(crop_w))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        available_energies: list[float] | torch.Tensor | None = None,
        bev_crop_hw: int | None = None,
        bev_crop_h: int | None = None,
        bev_crop_w: int | None = None,
        max_inference_batch_items: int = 8,
        inference_amp: bool = False,
        trainable: bool = False,
        tta: bool = False,
        pad_inference_batch: bool = False,
        **_ignored: Any,
    ) -> "ProtonDenseBevCorrectionHook":
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]
        model_kind = str(cfg.get("model", {}).get("kind", "fan_conv"))
        model_classes = {
            "fan_conv": FanGridConvCorrector,
            "separable_fan_conv": SeparableFanGridConvCorrector,
            "repvgg_unet": RepVGGUNetCorrector,
        }
        if model_kind not in model_classes:
            raise ValueError(f"Unsupported dense BEV correction model kind: {model_kind!r}")
        model_cls = model_classes[model_kind]
        model = model_cls.from_config(
            int(checkpoint.get("fan_input_dim", 8)),
            cfg,
            available_energies=available_energies,
        )
        load_model_state_dict(model, checkpoint["model_state"])
        model = model.to(device=device, dtype=dtype)
        if not trainable and hasattr(model, "fuse_repvgg"):
            model.fuse_repvgg()
        ckpt_args = checkpoint.get("args", {})
        if bev_crop_hw is None and isinstance(ckpt_args, dict):
            bev_crop_hw = int(ckpt_args.get("bev_crop_hw", 64))
        if bev_crop_h is None and isinstance(ckpt_args, dict):
            bev_crop_h = int(ckpt_args.get("bev_crop_h", bev_crop_hw))
        if bev_crop_w is None and isinstance(ckpt_args, dict):
            bev_crop_w = int(ckpt_args.get("bev_crop_w", bev_crop_hw))
        # Pre-v2 checkpoints carry no bev_feature_set; they are all v1.
        feature_set = "v1"
        if isinstance(ckpt_args, dict):
            feature_set = str(ckpt_args.get("bev_feature_set", "v1")).lower()
        expected_dim = BEV_FEATURE_CHANNELS.get(feature_set)
        got_dim = int(checkpoint.get("fan_input_dim", 8))
        if expected_dim is not None and got_dim != expected_dim:
            raise ValueError(
                f"Checkpoint declares bev_feature_set={feature_set!r} ({expected_dim} "
                f"channels) but fan_input_dim={got_dim}; refusing to build a hook whose "
                "features would not match the trained input projection."
            )
        return cls(
            model=model,
            cfg=cfg,
            bev_crop_hw=64 if bev_crop_hw is None else int(bev_crop_hw),
            bev_crop_h=bev_crop_h,
            bev_crop_w=bev_crop_w,
            max_inference_batch_items=max_inference_batch_items,
            inference_amp=inference_amp,
            trainable=trainable,
            bev_feature_set=feature_set,
            tta=tta,
            pad_inference_batch=pad_inference_batch,
        )

    def set_hu_volume(self, hu_volume: torch.Tensor | None) -> None:
        if hu_volume is None:
            self._material_id_volume = None
            return
        ref = next(self.model.parameters())
        with torch.no_grad():
            self._material_id_volume = _material_id_from_hu(
                hu_volume.to(device=ref.device, dtype=ref.dtype)
            ).to(torch.long)

    def _sample_material_bev(
        self,
        engine: Any,
        B: int,
        G: int,
        H: int,
        D: int,
        W: int,
        device: torch.device,
        bev_crop: dict | None = None,
    ) -> torch.Tensor:
        if self._material_id_volume is None:
            return torch.zeros((B, G, D, H, W), device=device, dtype=torch.long)
        if B != 1:
            raise ValueError("ProtonDenseBevCorrectionHook currently supports B=1")
        volume = self._material_id_volume.to(device=device).unsqueeze(0).float()

        if isinstance(bev_crop, list):
            out_h, out_w = bev_crop[0]["shape_hw"]
            sampled_out = torch.zeros((B, G, D, out_h, out_w), device=device, dtype=volume.dtype)
            for g_idx, crop in enumerate(bev_crop):
                h_src = crop["h_src"]
                h_dst = crop["h_dst"]
                w_src = crop["w_src"]
                w_dst = crop["w_dst"]
                _full_h, full_w = crop["full_shape_hw"]
                if h_src.stop <= h_src.start or w_src.stop <= w_src.start:
                    continue
                volume_src = volume[:, h_src, :, :]
                src_h = h_src.stop - h_src.start
                vol_flat = volume_src.reshape(B * src_h, 1, D, full_w)
                grid_g = engine.rad_depth_layer._inv_rot_grid[0, g_idx, 0, :, w_src].to(device=device)
                grid = grid_g.unsqueeze(0).expand(B * src_h, -1, -1, -1)
                sampled = F.grid_sample(
                    vol_flat,
                    grid,
                    mode="nearest",
                    padding_mode="zeros",
                    align_corners=False,
                )
                sampled = sampled.reshape(B, src_h, D, w_src.stop - w_src.start)
                sampled = sampled.permute(0, 2, 1, 3).contiguous()
                sampled_out[:, g_idx, :, h_dst, w_dst] = sampled
            return sampled_out.long()

        if bev_crop is None:
            vol_flat = volume.reshape(B * H, 1, D, W)
            bev_parts = []
            for g_idx in range(G):
                grid_g = engine.rad_depth_layer._inv_rot_grid[0, g_idx, 0].to(device=device)
                grid = grid_g.unsqueeze(0).expand(B * H, -1, -1, -1)
                sampled = F.grid_sample(
                    vol_flat,
                    grid,
                    mode="nearest",
                    padding_mode="zeros",
                    align_corners=False,
                )
                sampled = sampled.reshape(B, H, D, W).permute(0, 2, 1, 3).contiguous()
                bev_parts.append(sampled)
            return torch.stack(bev_parts, dim=1).long()

        h_src = bev_crop["h_src"]
        h_dst = bev_crop["h_dst"]
        w_src = bev_crop["w_src"]
        w_dst = bev_crop["w_dst"]
        _full_h, full_w = bev_crop["full_shape_hw"]
        out_h, out_w = bev_crop["shape_hw"]
        sampled_out = torch.zeros((B, G, D, out_h, out_w), device=device, dtype=volume.dtype)
        if h_src.stop <= h_src.start or w_src.stop <= w_src.start:
            return sampled_out.long()

        volume_src = volume[:, h_src, :, :]
        src_h = h_src.stop - h_src.start
        vol_flat = volume_src.reshape(B * src_h, 1, D, full_w)
        for g_idx in range(G):
            grid_g = engine.rad_depth_layer._inv_rot_grid[0, g_idx, 0, :, w_src].to(device=device)
            grid = grid_g.unsqueeze(0).expand(B * src_h, -1, -1, -1)
            sampled = F.grid_sample(
                vol_flat,
                grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled = sampled.reshape(B, src_h, D, w_src.stop - w_src.start)
            sampled = sampled.permute(0, 2, 1, 3).contiguous()
            sampled_out[:, g_idx, :, h_dst, w_dst] = sampled
        return sampled_out.long()

    def _model_forward_tta(
        self,
        features: torch.Tensor,
        dose: torch.Tensor,
        valid: torch.Tensor,
        fan: torch.Tensor,
        material_id: torch.Tensor,
        energy: torch.Tensor,
        sigma_mm: torch.Tensor,
    ) -> torch.Tensor:
        """Run the correction model, optionally averaging over lateral reflections.

        ``self.tta`` is off by default: every model in this project was trained with
        ``--no-augmentation``, so the net is NOT reflection-equivariant and averaging over
        transforms it never saw can just as easily hurt as help. Whether it does is an
        empirical question, which is why this is a switch and not a default.

        Note the transformed inputs are still in-distribution: ``_d4_apply_features``
        negates the signed h/w offset channels so the positional encoding keeps its
        canonical sign convention. ``energy`` and ``sigma_mm`` are per-item scalars and are
        untouched -- valid only because the transform set excludes 90-degree rotations,
        which would swap sigma_h/sigma_w.
        """
        if not self.tta:
            return self.model(
                features, dose, valid, fan,
                material_id=material_id, energy=energy, sigma_mm=sigma_mm,
            )["dose_hat"]

        acc: torch.Tensor | None = None
        for flip, k_rot in TTA_TRANSFORMS:
            out = self.model(
                _d4_apply_features(features, flip, k_rot),
                _d4_apply(dose, flip, k_rot),
                _d4_apply(valid, flip, k_rot),
                _d4_apply(fan, flip, k_rot),
                material_id=_d4_apply(material_id, flip, k_rot),
                energy=energy,
                sigma_mm=sigma_mm,
            )["dose_hat"]
            out = _d4_inverse(out, flip, k_rot)
            acc = out if acc is None else acc + out
        return acc / float(len(TTA_TRANSFORMS))

    def _peak_depth_mm(self, engine: Any, energy: torch.Tensor) -> float:
        """Water-equivalent depth of the Bragg peak for ``energy``, in mm.

        ``lut.get_edep_curve`` is itself cached per energy, so after the first
        beamlet at a given energy this costs a dict hit. Energies come from a
        discrete machine table, so the cache stays small.
        """
        key = round(float(energy.detach()), 4)
        cached = self._peak_depth_cache.get(key)
        if cached is None:
            depth, idd = engine.lut.get_edep_curve(energy.detach(), energy_value_hint=key)
            cached = float(depth[torch.argmax(idd)])
            self._peak_depth_cache[key] = cached
        return cached

    def _build_bev_features(
        self,
        spr_bev: torch.Tensor,
        weq_bev: torch.Tensor,
        dose_pb_bev: torch.Tensor,
        material_id_bev: torch.Tensor,
        crop_center_hw: tuple[float, float],
        peak_depth_mm: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, slice]]:
        N, D, H, W = dose_pb_bev.shape
        # Keep the same depth context used during training. Cropping to the PB
        # support changes convolution padding near the entrance and distal tail.
        d_src = slice(0, D)
        d_dst = slice(0, D)
        h_src, h_dst, h_target = _crop_slices(crop_center_hw[0], H, self.bev_crop_h)
        w_src, w_dst, w_target = _crop_slices(crop_center_hw[1], W, self.bev_crop_w)

        cD = d_src.stop - d_src.start
        cH, cW = h_target.stop - h_target.start, w_target.stop - w_target.start
        spr_src = spr_bev.expand(N, -1, -1, -1)
        weq_src = weq_bev.expand(N, -1, -1, -1)
        mat_src = material_id_bev.expand(N, -1, -1, -1)
        spr_c = spr_bev.new_zeros((N, cD, cH, cW))
        weq_c = weq_bev.new_zeros((N, cD, cH, cW))
        dose_c = dose_pb_bev.new_zeros((N, cD, cH, cW))
        mat_c = material_id_bev.new_zeros((N, cD, cH, cW))
        spr_c[:, d_dst, h_dst, w_dst] = spr_src[:, d_src, h_src, w_src]
        weq_c[:, d_dst, h_dst, w_dst] = weq_src[:, d_src, h_src, w_src]
        dose_c[:, d_dst, h_dst, w_dst] = dose_pb_bev[:, d_src, h_src, w_src]
        mat_c[:, d_dst, h_dst, w_dst] = mat_src[:, d_src, h_src, w_src]

        depth_scale = float(self.cfg.get("normalization", {}).get("depth_scale_mm", 100.0))
        h_offsets = torch.arange(h_target.start, h_target.stop, device=spr_c.device, dtype=spr_c.dtype) - float(crop_center_hw[0])
        w_offsets = torch.arange(w_target.start, w_target.stop, device=spr_c.device, dtype=spr_c.dtype) - float(crop_center_hw[1])
        h_grid = h_offsets.view(1, 1, cH, 1).expand(N, cD, cH, cW) / max(depth_scale, 1e-8)
        w_grid = w_offsets.view(1, 1, 1, cW).expand(N, cD, cH, cW) / max(depth_scale, 1e-8)
        t_depth = (torch.arange(d_src.start, d_src.stop, device=spr_c.device, dtype=spr_c.dtype) + 0.5).view(1, cD, 1, 1).expand(N, cD, cH, cW)
        dose_scale = dose_c.clamp_min(1e-8).amax(dim=(1, 2, 3), keepdim=True)

        feature_list = [
            spr_c - 1.0,
            torch.sqrt(h_grid.square() + w_grid.square()),
            t_depth / max(depth_scale, 1e-8),
            weq_c / max(depth_scale, 1e-8),
            h_grid,
            w_grid,
            spr_c.clamp_min(0.0),
            dose_c.clamp_min(0.0) / dose_scale,
        ]
        if self.bev_feature_set in ("v2", "v3"):
            if peak_depth_mm is None:
                raise ValueError(
                    f"bev_feature_set={self.bev_feature_set!r} requires peak_depth_mm"
                )
            # Residual range: WEQ measured from this beamlet's own Bragg peak, so
            # every energy lands on a common frame. Negative proximal, positive distal.
            feature_list.append((weq_c - float(peak_depth_mm)) / max(depth_scale, 1e-8))
        if self.bev_feature_set == "v3":
            # Range mixing: lateral WEQ structure the pencil-beam model cannot express.
            feature_list.append(_lateral_weq_gradient(weq_c) / max(depth_scale, 1e-8))
        features = torch.stack(feature_list, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        dose_pb_5d = dose_c.unsqueeze(1)
        valid_mask = (dose_c > 0.0).unsqueeze(1)
        material_id_5d = mat_c.unsqueeze(1)
        fan_mask = torch.ones(N, 1, 1, cH, cW, device=spr_c.device, dtype=torch.bool)
        crop_meta = {
            "d_src": d_src,
            "d_dst": d_dst,
            "h_src": h_src,
            "h_dst": h_dst,
            "w_src": w_src,
            "w_dst": w_dst,
        }
        return features, dose_pb_5d, valid_mask, material_id_5d, fan_mask, crop_meta

    def _axis_crop_center_hw(
        self,
        engine: Any,
        g_idx: int,
        H: int,
        W: int,
        bev_crop: dict | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[float, float]:
        """Return the ray-axis center in the current BEV tensor coordinates."""
        res_h, _res_d, res_w = (float(v) for v in engine.dose_grid_spacing)
        iso = engine.iso_centers.to(device=device, dtype=dtype)[int(g_idx)]
        target_h_start = 0.0
        target_w_start = 0.0
        if bev_crop is not None:
            target_h_start = float(bev_crop["target_h_start"])
            target_w_start = float(bev_crop["target_w_start"])
        center_h = float((iso[0] / res_h).detach()) - target_h_start
        center_w = float((iso[2] / res_w).detach()) - target_w_start
        if not math.isfinite(center_h):
            center_h = 0.5 * float(H - 1)
        if not math.isfinite(center_w):
            center_w = 0.5 * float(W - 1)
        return center_h, center_w

    def _layer_axis_crop_center_hw(
        self,
        engine: Any,
        payload: dict[str, torch.Tensor],
        g_idx: int,
        layer_idx: int,
        H: int,
        W: int,
        bev_crop: dict | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[float, float]:
        crop_centers_hw = payload.get("crop_centers_hw")
        if crop_centers_hw is not None:
            centers = crop_centers_hw.to(device=device, dtype=dtype)
            return float(centers[int(g_idx), 0].detach()), float(centers[int(g_idx), 1].detach())

        center_h, center_w = self._axis_crop_center_hw(
            engine=engine,
            g_idx=g_idx,
            H=H,
            W=W,
            bev_crop=bev_crop,
            device=device,
            dtype=dtype,
        )
        spot_positions = payload.get("spot_positions_mm")
        spot_layer_index = payload.get("spot_layer_index")
        spot_mask = payload.get("spot_mask")
        spot_weights = payload.get("spot_weights")
        if (
            spot_positions is None
            or spot_layer_index is None
            or spot_mask is None
            or spot_weights is None
        ):
            return center_h, center_w

        layer_mask = (
            spot_mask[0, g_idx].to(device=device).bool()
            & (spot_layer_index[0, g_idx].to(device=device, dtype=torch.long) == int(layer_idx))
            & (spot_weights[0, g_idx].to(device=device, dtype=dtype).abs() > 0)
        )
        if not layer_mask.any():
            return center_h, center_w

        positions = spot_positions[0, g_idx, layer_mask].to(device=device, dtype=dtype)
        weights = spot_weights[0, g_idx, layer_mask].to(device=device, dtype=dtype).abs()
        weight_sum = weights.sum().clamp_min(torch.finfo(dtype).tiny)
        offset_w = (positions[:, 0] * weights).sum() / weight_sum
        offset_h = (positions[:, 1] * weights).sum() / weight_sum
        res_h, _res_d, res_w = (float(v) for v in engine.dose_grid_spacing)
        return (
            center_h + float((offset_h / res_h).detach()),
            center_w + float((offset_w / res_w).detach()),
        )

    def forward(self, payload: dict[str, torch.Tensor], **context: Any) -> dict[str, torch.Tensor]:
        engine = context["engine"]
        edep_bev = payload["edep_bev"]
        B, G, D, H, W = edep_bev.shape
        if B != 1:
            raise ValueError("ProtonDenseBevCorrectionHook currently supports one sample at a time")

        layer_mask = payload["layer_mask"]

        edep_to_gy = payload["edep_to_gy"].to(device=edep_bev.device, dtype=edep_bev.dtype)
        spr_bev = payload["density_bev"].view(B, G, D, H, W)[0]
        weq_bev = payload["weq_bev"].view(B, G, D, H, W)[0]
        resolved_offset = payload.get("resolved_offset")
        if resolved_offset is not None:
            weq_bev = weq_bev + resolved_offset.view(G, 1, 1, 1)

        bev_crop = payload.get("bev_crop")
        material_id_bev = self._sample_material_bev(engine, B, G, H, D, W, edep_bev.device, bev_crop=bev_crop)[0]
        edep_by_layer = payload.get("edep_bev_by_layer")
        layer_indices = payload.get("edep_bev_layer_indices")
        if edep_by_layer is not None and layer_indices is None:
            raise ValueError("Dense BEV payload with edep_bev_by_layer also requires edep_bev_layer_indices")
        if layer_indices is not None:
            layer_indices = layer_indices.to(device=edep_bev.device, dtype=torch.long)

        edep_by_layer_t = (
            edep_by_layer.to(device=edep_bev.device, dtype=edep_bev.dtype)
            if edep_by_layer is not None
            else None
        )
        tasks: list[tuple[int, int, int, torch.Tensor]] = []

        for g_idx in range(G):
            active_layers = torch.nonzero(layer_mask[0, g_idx], as_tuple=False).flatten()
            if active_layers.numel() == 0:
                continue

            if edep_by_layer is None:
                if active_layers.numel() != 1:
                    raise ValueError(
                        "ProtonDenseBevCorrectionHook requires per-layer dense BEV payload when more than one energy layer is active"
                    )
                layer_idx_t = active_layers[0]
                energy_t = payload["layer_energies_mev"][0, g_idx, int(layer_idx_t.detach().item())].view(1)
                tasks.append((g_idx, int(layer_idx_t.detach().item()), -1, energy_t))
            else:
                for stack_pos, layer_idx_t in enumerate(layer_indices):
                    layer_idx = int(layer_idx_t.detach().item())
                    if not bool(layer_mask[0, g_idx, layer_idx].detach().item()):
                        continue
                    energy_t = payload["layer_energies_mev"][0, g_idx, layer_idx].view(1)
                    tasks.append((g_idx, layer_idx, int(stack_pos), energy_t))

        if not tasks:
            return payload

        profile_timing = os.environ.get("PYDOSERT_DENSE_HOOK_TIMING", "").lower() in {"1", "true", "yes", "on"}
        timings: dict[str, float] = {}

        def _sync_time(label: str, start_time: float) -> None:
            if edep_bev.device.type == "cuda":
                torch.cuda.synchronize(edep_bev.device)
            timings[label] = timings.get(label, 0.0) + time.perf_counter() - start_time

        corrected_dose_bev = edep_bev.new_zeros((G, D, H, W))
        grad_context = nullcontext() if self.trainable else torch.inference_mode()
        chunk_size = self.max_inference_batch_items
        with grad_context:
            for start in range(0, len(tasks), chunk_size):
                _t_chunk = time.perf_counter()
                chunk_tasks = tasks[start:start + chunk_size]
                features_items = []
                dose_items = []
                valid_items = []
                material_items = []
                fan_items = []
                energy_items = []
                sigma_items = []
                meta_items: list[tuple[int, torch.Tensor, slice, slice, slice, slice, slice, slice]] = []

                for g_idx, layer_idx, stack_pos, energy_t in chunk_tasks:
                    if stack_pos < 0:
                        dose_pb_layer = edep_bev[0, g_idx].unsqueeze(0) * edep_to_gy
                    else:
                        if edep_by_layer_t is None:
                            raise RuntimeError("Missing per-layer dense BEV tensor for correction task")
                        dose_pb_layer = edep_by_layer_t[stack_pos, 0, g_idx].unsqueeze(0) * edep_to_gy

                    crop_center_hw = self._layer_axis_crop_center_hw(
                        engine=engine,
                        payload=payload,
                        g_idx=g_idx,
                        layer_idx=layer_idx,
                        H=H,
                        W=W,
                        bev_crop=bev_crop,
                        device=edep_bev.device,
                        dtype=edep_bev.dtype,
                    )
                    features_i, dose_i, valid_i, material_i, fan_i, crop_meta = self._build_bev_features(
                        spr_bev[g_idx:g_idx + 1],
                        weq_bev[g_idx:g_idx + 1],
                        dose_pb_layer,
                        material_id_bev[g_idx:g_idx + 1],
                        crop_center_hw,
                        peak_depth_mm=(
                            self._peak_depth_mm(engine, energy_t)
                            if self.bev_feature_set in ("v2", "v3")
                            else None
                        ),
                    )
                    features_items.append(features_i)
                    dose_items.append(dose_i)
                    valid_items.append(valid_i)
                    material_items.append(material_i)
                    fan_items.append(fan_i)
                    energy_items.append(energy_t.to(device=edep_bev.device, dtype=edep_bev.dtype))
                    sigma_items.append(
                        payload["layer_sigmas_mm"][0, g_idx, layer_idx]
                        .to(device=edep_bev.device, dtype=edep_bev.dtype)
                        .view(1, -1)
                    )
                    meta_items.append((
                        g_idx,
                        dose_pb_layer,
                        crop_meta["d_src"],
                        crop_meta["d_dst"],
                        crop_meta["h_src"],
                        crop_meta["h_dst"],
                        crop_meta["w_src"],
                        crop_meta["w_dst"],
                    ))
                if profile_timing:
                    _sync_time("build_features", _t_chunk)

                _t_pack = time.perf_counter()
                batch_n = len(features_items)
                # Allocate the full batch even when the tail chunk is short: the padded rows
                # stay zero, are never scattered back (the loop below walks meta_items), and
                # keep the batch dimension constant so the compiled graph is reused instead
                # of recompiled on the last chunk of every beam.
                alloc_n = max(batch_n, self.max_inference_batch_items) if self.pad_inference_batch else batch_n
                _n, channels, _depth, height, width = features_items[0].shape
                max_depth = max(int(item.shape[2]) for item in features_items)
                features_chunk = features_items[0].new_zeros((alloc_n, channels, max_depth, height, width))
                dose_chunk = dose_items[0].new_zeros((alloc_n, 1, max_depth, height, width))
                valid_chunk = torch.zeros(
                    (alloc_n, 1, max_depth, height, width),
                    device=features_chunk.device,
                    dtype=torch.bool,
                )
                material_chunk = torch.zeros(
                    (alloc_n, 1, max_depth, height, width),
                    device=features_chunk.device,
                    dtype=torch.long,
                )
                depths: list[int] = []
                for local_i, (features_i, dose_i, valid_i, material_i) in enumerate(
                    zip(features_items, dose_items, valid_items, material_items, strict=True)
                ):
                    depth_i = int(features_i.shape[2])
                    depths.append(depth_i)
                    features_chunk[local_i, :, :depth_i] = features_i[0]
                    dose_chunk[local_i, :, :depth_i] = dose_i[0]
                    valid_chunk[local_i, :, :depth_i] = valid_i[0]
                    material_chunk[local_i, :, :depth_i] = material_i[0]

                fan_chunk = torch.cat(fan_items, dim=0)
                energy_chunk = torch.cat(energy_items, dim=0)
                sigma_chunk = torch.cat(sigma_items, dim=0)
                if alloc_n > batch_n:
                    # The per-item conditioning tensors must match the padded batch. Repeat
                    # the last real row rather than using zeros: energy/sigma feed FiLM-style
                    # conditioning, and a zero energy is out of distribution in a way that can
                    # produce non-finite activations, which would poison the real rows through
                    # AMP's shared gradient scaler if this ever ran under autocast in training.
                    pad_n = alloc_n - batch_n
                    fan_chunk = torch.cat([fan_chunk, fan_chunk[-1:].expand(pad_n, *fan_chunk.shape[1:])], dim=0)
                    energy_chunk = torch.cat(
                        [energy_chunk, energy_chunk[-1:].expand(pad_n, *energy_chunk.shape[1:])], dim=0
                    )
                    sigma_chunk = torch.cat(
                        [sigma_chunk, sigma_chunk[-1:].expand(pad_n, *sigma_chunk.shape[1:])], dim=0
                    )
                if profile_timing:
                    _sync_time("pack_batch", _t_pack)

                amp_enabled = (
                    self.inference_amp
                    and not self.trainable
                    and features_chunk.device.type == "cuda"
                    and features_chunk.dtype == torch.float32
                )
                _t_model = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                    dose_hat_5d = self._model_forward_tta(
                        features_chunk,
                        dose_chunk,
                        valid_chunk,
                        fan_chunk,
                        material_chunk,
                        energy_chunk,
                        sigma_chunk,
                    )
                dose_hat_chunk = dose_hat_5d.squeeze(1)
                if profile_timing:
                    _sync_time("model", _t_model)

                _t_scatter = time.perf_counter()
                for local_i, (g_idx, dose_pb_layer, d_src, d_dst, h_src, h_dst, w_src, w_dst) in enumerate(meta_items):
                    dose_hat = dose_hat_chunk[local_i:local_i + 1, :depths[local_i]]
                    corrected_layer = dose_pb_layer.clone()
                    corrected_layer[:, d_src, h_src, w_src] = dose_hat[:, d_dst, h_dst, w_dst]
                    corrected_dose_bev[g_idx] += corrected_layer.sum(dim=0)
                if profile_timing:
                    _sync_time("scatter_back", _t_scatter)

        _t_finish = time.perf_counter()
        corrected_edep_bev = corrected_dose_bev / edep_to_gy.clamp_min(torch.finfo(corrected_dose_bev.dtype).tiny)
        if profile_timing:
            _sync_time("finish", _t_finish)
            summary = " ".join(f"{name}={value * 1e3:.2f}ms" for name, value in sorted(timings.items()))
            print(f"[dense_hook_timing] tasks={len(tasks)} chunks={math.ceil(len(tasks) / chunk_size)} {summary}")
        return {**payload, "edep_bev": corrected_edep_bev.view(B, G, D, H, W)}
