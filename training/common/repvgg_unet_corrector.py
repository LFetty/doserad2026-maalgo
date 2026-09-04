from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from training.common.separable_fan_grid_corrector import (
    _SeparableFanBlock,
    _apply_norm3d,
    _conditioned_norm3d,
)


class _DownsampleProjection(nn.Module):
    """Cheap lateral-only reduction followed by channel projection."""

    def __init__(self, in_channels: int, out_channels: int, stride_hw: tuple[int, int]) -> None:
        super().__init__()
        self.stride_hw = tuple(int(v) for v in stride_hw)
        self.project = nn.Conv3d(int(in_channels), int(out_channels), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sh, sw = self.stride_hw
        x = F.avg_pool3d(x, kernel_size=(1, sh, sw), stride=(1, sh, sw), ceil_mode=True)
        return self.project(x)


class _GatedLatentDepthMixer(nn.Module):
    """Multi-dilation depth-only residual mixer for the deepest lateral latent map."""

    def __init__(
        self,
        channels: int,
        depth_kernel_size: int,
        dilations: tuple[int, ...],
        norm_kind: str,
        conditioning_dim: int,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        kernel = max(3, int(depth_kernel_size) | 1)
        self.norm = _conditioned_norm3d(norm_kind, self.channels, int(conditioning_dim))
        self.branches = nn.ModuleList(
            [
                nn.Conv3d(
                    self.channels,
                    self.channels,
                    kernel_size=(kernel, 1, 1),
                    padding=(kernel // 2 * int(dilation), 0, 0),
                    dilation=(int(dilation), 1, 1),
                    groups=self.channels,
                    bias=True,
                )
                for dilation in dilations
            ]
        )
        self.mix = nn.Sequential(
            nn.SiLU(),
            nn.Conv3d(self.channels, self.channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(self.channels, self.channels, kernel_size=1),
        )
        self.gate = nn.Conv3d(self.channels, self.channels, kernel_size=1)
        nn.init.constant_(self.gate.bias, -2.0)
        self.out = nn.Conv3d(self.channels, self.channels, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor | None = None) -> torch.Tensor:
        h = _apply_norm3d(self.norm, x, conditioning)
        context = torch.stack([branch(h) for branch in self.branches], dim=0).sum(dim=0)
        gate = torch.sigmoid(self.gate(h))
        return x + gate * self.out(self.mix(context))


class _RepVGGStage(nn.Module):
    def __init__(
        self,
        channels: int,
        num_blocks: int,
        depth_kernel_size: int,
        dropout: float,
        mix_ratio: float,
        use_repvgg: bool,
        grad_checkpoint: bool,
        norm_kind: str,
        conditioning_dim: int,
    ) -> None:
        super().__init__()
        self.grad_checkpoint = bool(grad_checkpoint)
        self.blocks = nn.ModuleList(
            [
                _SeparableFanBlock(
                    channels=int(channels),
                    depth_kernel_size=int(depth_kernel_size),
                    dropout=float(dropout),
                    mix_ratio=float(mix_ratio),
                    use_repvgg=bool(use_repvgg),
                    norm_kind=str(norm_kind),
                    conditioning_dim=int(conditioning_dim),
                )
                for _ in range(int(num_blocks))
            ]
        )

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            if self.grad_checkpoint and self.training and x.requires_grad:
                x, _maps = checkpoint(block, x, conditioning, use_reentrant=False)
            else:
                x, _maps = block(x, conditioning)
        return x

    def fuse_repvgg(self) -> None:
        for block in self.blocks:
            block.reparameterize()


class RepVGGUNetCorrector(nn.Module):
    """Fast BEV residual U-Net with lateral-only reductions.

    Depth remains at native resolution. The first reduction equalizes anisotropic
    lateral BEV spacing; subsequent reductions are isotropic in the BEV plane.
    """

    def __init__(
        self,
        input_dim: int,
        native_dim: int = 8,
        stage_dims: tuple[int, int, int] = (16, 24, 32),
        stage_blocks: tuple[int, int, int, int, int] = (1, 1, 1, 2, 1),
        depth_kernel_size: int = 11,
        dropout: float = 0.0,
        residual_mode: str = "additive",
        additive_scale_frac: float = 0.25,
        material_embedding_dim: int = 4,
        num_materials: int = 86,
        available_energies: list[float] | torch.Tensor | None = None,
        use_sigma_conditioning: bool = False,
        eps: float = 1e-3,
        grad_checkpoint: bool = True,
        mix_ratio: float = 0.5,
        use_repvgg: bool = True,
        equalize_axis: str = "w",
        equalize_factor: int = 3,
        extra_stage_dim: int = 0,
        extra_stage_blocks: int = 2,
        latent_depth_mixer: bool = False,
        latent_depth_mixer_kernel_size: int = 11,
        latent_depth_mixer_dilations: tuple[int, ...] = (1, 2, 4),
        deep_supervision: bool = True,
        energy_conditioning: str = "embedding",
        energy_fourier_bands: int = 4,
        norm_kind: str = "group",
        conditioning_injection: str = "entrance",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.native_dim = int(native_dim)
        self.stage_dims = tuple(int(v) for v in stage_dims)
        if len(self.stage_dims) != 3:
            raise ValueError("stage_dims must contain exactly three channel counts")
        self.stage_blocks = tuple(int(v) for v in stage_blocks)
        if len(self.stage_blocks) != 5:
            raise ValueError("stage_blocks must contain native, equal, encoder, bottleneck, and decoder counts")
        self.grad_checkpoint = bool(grad_checkpoint)
        self.residual_mode = str(residual_mode).lower()
        if self.residual_mode not in {"additive", "multiplicative"}:
            raise ValueError("residual_mode must be 'additive' or 'multiplicative'")
        self.additive_scale_frac = float(additive_scale_frac)
        self.material_embedding_dim = int(material_embedding_dim)
        self.num_materials = int(num_materials)
        self.eps = float(eps)
        self.use_repvgg = bool(use_repvgg)
        self.equalize_axis = str(equalize_axis).lower()
        if self.equalize_axis not in {"h", "w"}:
            raise ValueError("equalize_axis must be 'h' or 'w'")
        self.equalize_factor = max(1, int(equalize_factor))
        self.extra_stage_dim = max(0, int(extra_stage_dim))
        self.extra_stage_blocks = max(1, int(extra_stage_blocks))
        self.latent_depth_mixer_enabled = bool(latent_depth_mixer)
        self.latent_depth_mixer_kernel_size = max(3, int(latent_depth_mixer_kernel_size) | 1)
        self.latent_depth_mixer_dilations = tuple(
            max(1, int(dilation)) for dilation in latent_depth_mixer_dilations
        )
        if not self.latent_depth_mixer_dilations:
            raise ValueError("latent_depth_mixer_dilations must contain at least one dilation")
        self.deep_supervision = bool(deep_supervision)
        self.energy_conditioning = str(energy_conditioning).lower()
        if self.energy_conditioning not in {"none", "embedding", "scalar", "fourier"}:
            raise ValueError("energy_conditioning must be 'none', 'embedding', 'scalar', or 'fourier'")
        self.energy_fourier_bands = max(1, int(energy_fourier_bands))
        self.norm_kind = str(norm_kind).lower()
        if self.norm_kind not in {"group", "instance"}:
            raise ValueError("norm_kind must be 'group' or 'instance'")
        self.conditioning_injection = str(conditioning_injection).lower()
        if self.conditioning_injection not in {"entrance", "adagn", "entrance_adagn"}:
            raise ValueError("conditioning_injection must be 'entrance', 'adagn', or 'entrance_adagn'")
        self.use_entrance_conditioning = self.conditioning_injection in {"entrance", "entrance_adagn"}
        self.use_adagn_conditioning = self.conditioning_injection in {"adagn", "entrance_adagn"}

        self.material_embedding = (
            nn.Embedding(self.num_materials, self.material_embedding_dim)
            if self.material_embedding_dim > 0
            else None
        )
        if available_energies is not None:
            energies_t = torch.as_tensor(available_energies, dtype=torch.float32).sort().values
            self.register_buffer("available_energies", energies_t)
        else:
            self.register_buffer("available_energies", None)
        if self.energy_conditioning == "embedding" and self.available_energies is not None:
            self.energy_embed = nn.Embedding(len(self.available_energies), self.native_dim)
            if not self.use_adagn_conditioning:
                nn.init.zeros_(self.energy_embed.weight)
        else:
            self.energy_embed = None
        if self.energy_conditioning in {"scalar", "fourier"}:
            energy_input_dim = 1 if self.energy_conditioning == "scalar" else 1 + 2 * self.energy_fourier_bands
            self.energy_project = nn.Sequential(
                nn.Linear(energy_input_dim, self.native_dim),
                nn.SiLU(),
                nn.Linear(self.native_dim, self.native_dim),
            )
            if not self.use_adagn_conditioning:
                nn.init.zeros_(self.energy_project[-1].weight)
                nn.init.zeros_(self.energy_project[-1].bias)
        else:
            self.energy_project = None
        self.use_sigma_conditioning = bool(use_sigma_conditioning)
        self.sigma_project = (
            nn.Sequential(
                nn.Linear(2, self.native_dim),
                nn.SiLU(),
                nn.Linear(self.native_dim, self.native_dim),
            )
            if self.use_sigma_conditioning
            else None
        )
        if self.sigma_project is not None and not self.use_adagn_conditioning:
            nn.init.zeros_(self.sigma_project[-1].weight)
            nn.init.zeros_(self.sigma_project[-1].bias)

        total_input_dim = self.input_dim + max(self.material_embedding_dim, 0)
        equal_dim, encoder_dim, bottleneck_dim = self.stage_dims
        native_blocks, equal_blocks, encoder_blocks, bottleneck_blocks, decoder_blocks = self.stage_blocks
        stage_kwargs = {
            "depth_kernel_size": int(depth_kernel_size),
            "dropout": float(dropout),
            "mix_ratio": float(mix_ratio),
            "use_repvgg": self.use_repvgg,
            "grad_checkpoint": self.grad_checkpoint,
            "norm_kind": self.norm_kind,
            "conditioning_dim": self.native_dim if self.use_adagn_conditioning else 0,
        }

        self.in_proj = nn.Conv3d(total_input_dim, self.native_dim, kernel_size=1)
        self.native_stage = _RepVGGStage(self.native_dim, native_blocks, **stage_kwargs)
        equal_stride = (self.equalize_factor, 1) if self.equalize_axis == "h" else (1, self.equalize_factor)
        self.down_equal = _DownsampleProjection(self.native_dim, equal_dim, equal_stride)
        self.equal_stage = _RepVGGStage(equal_dim, equal_blocks, **stage_kwargs)
        self.down_encoder = _DownsampleProjection(equal_dim, encoder_dim, (2, 2))
        self.encoder_stage = _RepVGGStage(encoder_dim, encoder_blocks, **stage_kwargs)
        self.down_bottleneck = _DownsampleProjection(encoder_dim, bottleneck_dim, (2, 2))
        self.bottleneck_stage = _RepVGGStage(bottleneck_dim, bottleneck_blocks, **stage_kwargs)
        if self.extra_stage_dim > 0:
            self.down_extra = _DownsampleProjection(bottleneck_dim, self.extra_stage_dim, (2, 2))
            self.extra_stage = _RepVGGStage(self.extra_stage_dim, self.extra_stage_blocks, **stage_kwargs)
            self.latent_depth_mixer = (
                _GatedLatentDepthMixer(
                    channels=self.extra_stage_dim,
                    depth_kernel_size=self.latent_depth_mixer_kernel_size,
                    dilations=self.latent_depth_mixer_dilations,
                    norm_kind=self.norm_kind,
                    conditioning_dim=self.native_dim if self.use_adagn_conditioning else 0,
                )
                if self.latent_depth_mixer_enabled
                else None
            )
            self.up_bottleneck = nn.Conv3d(self.extra_stage_dim, bottleneck_dim, kernel_size=1)
            self.decoder_bottleneck_stage = _RepVGGStage(bottleneck_dim, decoder_blocks, **stage_kwargs)
            self.aux_bottleneck_head = nn.Conv3d(bottleneck_dim, 1, kernel_size=1)
            nn.init.zeros_(self.aux_bottleneck_head.weight)
            nn.init.zeros_(self.aux_bottleneck_head.bias)
        else:
            self.down_extra = None
            self.extra_stage = None
            self.latent_depth_mixer = None
            self.up_bottleneck = None
            self.decoder_bottleneck_stage = None
            self.aux_bottleneck_head = None

        self.up_encoder = nn.Conv3d(bottleneck_dim, encoder_dim, kernel_size=1)
        self.decoder_encoder_stage = _RepVGGStage(encoder_dim, decoder_blocks, **stage_kwargs)
        self.up_equal = nn.Conv3d(encoder_dim, equal_dim, kernel_size=1)
        self.decoder_equal_stage = _RepVGGStage(equal_dim, decoder_blocks, **stage_kwargs)
        self.up_native = nn.Conv3d(equal_dim, self.native_dim, kernel_size=1)
        self.norm = _conditioned_norm3d(
            self.norm_kind,
            self.native_dim,
            self.native_dim if self.use_adagn_conditioning else 0,
        )
        self.head = nn.Conv3d(self.native_dim, 1, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.aux_encoder_head = nn.Conv3d(encoder_dim, 1, kernel_size=1)
        self.aux_equal_head = nn.Conv3d(equal_dim, 1, kernel_size=1)
        nn.init.zeros_(self.aux_encoder_head.weight)
        nn.init.zeros_(self.aux_encoder_head.bias)
        nn.init.zeros_(self.aux_equal_head.weight)
        nn.init.zeros_(self.aux_equal_head.bias)

    @classmethod
    def from_config(
        cls,
        input_dim: int,
        cfg: dict[str, Any],
        available_energies: list[float] | torch.Tensor | None = None,
    ) -> "RepVGGUNetCorrector":
        model_cfg = cfg.get("model", {})
        return cls(
            input_dim=input_dim,
            native_dim=int(model_cfg.get("unet_native_dim", 8)),
            stage_dims=tuple(int(v) for v in model_cfg.get("unet_stage_dims", [16, 24, 32])),
            stage_blocks=tuple(int(v) for v in model_cfg.get("unet_stage_blocks", [1, 1, 1, 2, 1])),
            depth_kernel_size=int(model_cfg.get("depth_kernel_size", 11)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            residual_mode=str(model_cfg.get("residual_mode", "additive")),
            additive_scale_frac=float(model_cfg.get("additive_scale_frac", 0.25)),
            material_embedding_dim=int(model_cfg.get("material_embedding_dim", 4)),
            num_materials=int(model_cfg.get("num_materials", 86)),
            available_energies=available_energies,
            use_sigma_conditioning=bool(model_cfg.get("use_sigma_conditioning", False)),
            eps=float(cfg.get("dose", {}).get("eps", 1e-3)),
            grad_checkpoint=bool(model_cfg.get("grad_checkpoint", True)),
            mix_ratio=float(model_cfg.get("mix_ratio", 0.5)),
            use_repvgg=bool(model_cfg.get("use_repvgg", True)),
            equalize_axis=str(model_cfg.get("unet_equalize_axis", "w")),
            equalize_factor=int(model_cfg.get("unet_equalize_factor", 3)),
            extra_stage_dim=int(model_cfg.get("unet_extra_stage_dim", 0)),
            extra_stage_blocks=int(model_cfg.get("unet_extra_stage_blocks", 2)),
            latent_depth_mixer=bool(model_cfg.get("unet_latent_depth_mixer", False)),
            latent_depth_mixer_kernel_size=int(model_cfg.get("unet_latent_depth_mixer_kernel_size", 11)),
            latent_depth_mixer_dilations=tuple(
                int(v) for v in model_cfg.get("unet_latent_depth_mixer_dilations", [1, 2, 4])
            ),
            deep_supervision=bool(model_cfg.get("unet_deep_supervision", True)),
            energy_conditioning=str(model_cfg.get("unet_energy_conditioning", "embedding")),
            energy_fourier_bands=int(model_cfg.get("unet_energy_fourier_bands", 4)),
            norm_kind=str(model_cfg.get("unet_norm", "group")),
            conditioning_injection=str(model_cfg.get("unet_conditioning_injection", "entrance")),
        )

    def fuse_repvgg(self) -> None:
        for module in self.modules():
            if isinstance(module, _RepVGGStage):
                module.fuse_repvgg()

    def _material_features(self, material_id: torch.Tensor | None, features: torch.Tensor) -> torch.Tensor | None:
        if self.material_embedding is None:
            return None
        if material_id is None:
            material_id = torch.zeros(
                (features.shape[0], 1, features.shape[2], features.shape[3], features.shape[4]),
                device=features.device,
                dtype=torch.long,
            )
        material_id = material_id.to(device=features.device, dtype=torch.long).clamp(0, self.num_materials - 1)
        if material_id.ndim == 4:
            material_id = material_id.unsqueeze(1)
        if material_id.shape[1] != 1:
            raise ValueError(f"material_id must have one channel, got shape {tuple(material_id.shape)}")
        embedded = self.material_embedding(material_id.squeeze(1))
        return embedded.permute(0, 4, 1, 2, 3).to(dtype=features.dtype)

    def _energy_to_index(self, energy: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor | None:
        if self.energy_embed is None or self.available_energies is None:
            return None
        if energy is None:
            return torch.zeros(batch_size, device=device, dtype=torch.long)
        energy_flat = energy.to(device=device, dtype=torch.float32).reshape(-1)
        if energy_flat.shape[0] == 1 and batch_size != 1:
            energy_flat = energy_flat.expand(batch_size)
        return (energy_flat.unsqueeze(-1) - self.available_energies.unsqueeze(0)).abs().argmin(dim=-1)

    def _energy_features(self, energy: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if energy is None:
            energy_flat = torch.zeros(batch_size, device=device, dtype=torch.float32)
        else:
            energy_flat = energy.to(device=device, dtype=torch.float32).reshape(-1)
            if energy_flat.shape[0] == 1 and batch_size != 1:
                energy_flat = energy_flat.expand(batch_size)
        if self.available_energies is not None and self.available_energies.numel() > 1:
            e_min = self.available_energies[0]
            e_max = self.available_energies[-1]
            e_norm = 2.0 * (energy_flat - e_min) / (e_max - e_min).clamp_min(1e-6) - 1.0
        else:
            e_norm = energy_flat / 125.0 - 1.0
        if self.energy_conditioning == "scalar":
            return e_norm.unsqueeze(-1)
        frequencies = 2.0 ** torch.arange(self.energy_fourier_bands, device=device, dtype=e_norm.dtype)
        angles = math.pi * e_norm.unsqueeze(-1) * frequencies.unsqueeze(0)
        return torch.cat((e_norm.unsqueeze(-1), torch.sin(angles), torch.cos(angles)), dim=-1)

    def _correct(self, residual: torch.Tensor, dose_pb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if tuple(residual.shape[2:]) != tuple(dose_pb.shape[2:]):
            dose_pb = F.adaptive_avg_pool3d(dose_pb, residual.shape[2:])
            mask = F.adaptive_max_pool3d(mask.to(dtype=dose_pb.dtype), residual.shape[2:]) > 0.0
        if self.residual_mode == "additive":
            scale = dose_pb.detach().amax(dim=(2, 3, 4), keepdim=True).clamp_min(torch.finfo(dose_pb.dtype).tiny)
            dose_hat = dose_pb + residual * scale * self.additive_scale_frac
        else:
            dose_hat = (dose_pb + self.eps).clamp_min(self.eps) * torch.exp(residual) - self.eps
        return torch.where(mask, dose_hat.clamp_min(0.0), torch.zeros_like(dose_hat))

    @staticmethod
    def _upsample_project(x: torch.Tensor, project: nn.Module, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
        return project(x) + skip

    def forward(
        self,
        features: torch.Tensor,
        dose_pb: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        fan_mask: torch.Tensor | None = None,
        material_id: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
        sigma_mm: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        material_features = self._material_features(material_id, features)
        if material_features is not None:
            features = torch.cat((features, material_features), dim=1)
        native = self.in_proj(features)
        conditioning = native.new_zeros((int(features.shape[0]), self.native_dim))
        has_conditioning = False
        energy_idx = self._energy_to_index(energy, int(features.shape[0]), features.device)
        if energy_idx is not None:
            conditioning = conditioning + self.energy_embed(energy_idx).to(dtype=native.dtype)
            has_conditioning = True
        if self.energy_project is not None:
            energy_features = self._energy_features(energy, int(features.shape[0]), features.device)
            conditioning = conditioning + self.energy_project(energy_features).to(dtype=native.dtype)
            has_conditioning = True
        if self.sigma_project is not None and sigma_mm is not None:
            sigma_vec = sigma_mm.to(device=features.device, dtype=features.dtype).reshape(-1, 2)
            if sigma_vec.shape[0] == 1 and int(features.shape[0]) != 1:
                sigma_vec = sigma_vec.expand(int(features.shape[0]), -1)
            conditioning = conditioning + self.sigma_project(sigma_vec).to(dtype=native.dtype)
            has_conditioning = True
        if self.use_entrance_conditioning and has_conditioning:
            native = native + conditioning.view(-1, self.native_dim, 1, 1, 1)
        adagn_conditioning = conditioning if self.use_adagn_conditioning and has_conditioning else None

        native = self.native_stage(native, adagn_conditioning)
        equal = self.equal_stage(self.down_equal(native), adagn_conditioning)
        encoder = self.encoder_stage(self.down_encoder(equal), adagn_conditioning)
        bottleneck = self.bottleneck_stage(self.down_bottleneck(encoder), adagn_conditioning)
        decoder_bottleneck = bottleneck
        if self.extra_stage is not None:
            if self.down_extra is None or self.up_bottleneck is None or self.decoder_bottleneck_stage is None:
                raise RuntimeError("extra U-Net stage is incompletely configured")
            extra = self.extra_stage(self.down_extra(bottleneck), adagn_conditioning)
            if self.latent_depth_mixer is not None:
                extra = self.latent_depth_mixer(extra, adagn_conditioning)
            decoder_bottleneck = self.decoder_bottleneck_stage(
                self._upsample_project(extra, self.up_bottleneck, bottleneck),
                adagn_conditioning,
            )
        decoder_encoder = self.decoder_encoder_stage(
            self._upsample_project(decoder_bottleneck, self.up_encoder, encoder),
            adagn_conditioning,
        )
        decoder_equal = self.decoder_equal_stage(
            self._upsample_project(decoder_encoder, self.up_equal, equal),
            adagn_conditioning,
        )
        decoder_native = self._upsample_project(decoder_equal, self.up_native, native)

        mask = dose_pb > 0.0 if valid_mask is None else valid_mask
        if fan_mask is not None:
            mask = mask & fan_mask
        residual = self.head(F.silu(_apply_norm3d(self.norm, decoder_native, adagn_conditioning)))
        dose_hat = self._correct(residual, dose_pb, mask)
        deep_supervision: tuple[torch.Tensor, ...] = ()
        if self.training and self.deep_supervision:
            predictions = [
                self._correct(self.aux_equal_head(decoder_equal), dose_pb, mask),
                self._correct(self.aux_encoder_head(decoder_encoder), dose_pb, mask),
            ]
            if self.aux_bottleneck_head is not None:
                predictions.append(self._correct(self.aux_bottleneck_head(decoder_bottleneck), dose_pb, mask))
            deep_supervision = tuple(predictions)
        return {
            "residual": residual,
            "dose_hat": dose_hat,
            "deep_supervision": deep_supervision,
            "attn_maps": [],
        }
