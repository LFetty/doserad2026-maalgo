from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def _groups(channels: int) -> int:
    return max(1, min(8, int(channels) // 4))


def _norm3d(kind: str, channels: int) -> nn.Module:
    kind = str(kind).lower()
    if kind == "group":
        return nn.GroupNorm(_groups(channels), channels)
    if kind == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    raise ValueError("norm_kind must be 'group' or 'instance'")


class _AdaptiveNorm3d(nn.Module):
    """Apply a zero-initialized channel-wise adaptive scale and shift after normalization."""

    def __init__(self, kind: str, channels: int, conditioning_dim: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.norm = _norm3d(kind, self.channels)
        self.modulation = nn.Linear(int(conditioning_dim), 2 * self.channels)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm(x)
        if conditioning is None:
            return x
        scale, shift = self.modulation(conditioning).chunk(2, dim=-1)
        scale = scale.view(-1, self.channels, 1, 1, 1)
        shift = shift.view(-1, self.channels, 1, 1, 1)
        return x * (1.0 + scale) + shift


def _conditioned_norm3d(kind: str, channels: int, conditioning_dim: int) -> nn.Module:
    if int(conditioning_dim) > 0:
        return _AdaptiveNorm3d(kind, channels, conditioning_dim)
    return _norm3d(kind, channels)


def _apply_norm3d(norm: nn.Module, x: torch.Tensor, conditioning: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(norm, _AdaptiveNorm3d):
        return norm(x, conditioning)
    return norm(x)


class _SCSAM3D(nn.Module):
    """3-D Sequential Channel-Spatial Attention Module (paper-like SCSAM).

    Channel attention:  M_C = sigmoid(MLP(GAP(x) + GMP(x)))
      — single MLP pass on the *summed* descriptors, unlike CBAM's two separate passes.

    Spatial attention:  M_S = sigmoid(DC([avg_ch(fc), max_ch(fc)]))
      — dilated 3×3×3 conv on channel-pooled features; default dilation (2,1,1)
        dilates along the depth axis where Bragg-peak structure is longest.

    Residual:  out = x + gate_scale * (fc ⊗ M_S)
      — ``gate_scale`` is a learnable scalar initialised to **0**, so the module
        is an exact identity at step 0 (safe to add without hyper-param search).

    Note on zero-padded fan mask: ``amax`` is unaffected by zero-padding so
    GMP statistics are mask-safe; GAP will dilute by the masked fraction but
    the ``GAP + GMP`` sum is still dominated by the maximum, which is correct.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        dilation: tuple[int, int, int] = (2, 1, 1),
    ) -> None:
        super().__init__()
        hidden = max(4, int(channels) // max(1, int(reduction)))
        pad = tuple(int(d) for d in dilation)  # same-padding for a 3^3 kernel

        # Channel attention MLP (two pointwise convs, single eval)
        self.channel_mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
        )

        # Spatial attention: channel-pooled → dilated conv
        self.spatial_dc = nn.Conv3d(
            2,
            1,
            kernel_size=3,
            padding=pad,
            dilation=pad,
            bias=True,
        )

        # Learnable residual gate — init 0 → identity at step 0
        self.gate_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(output, spatial_attn_map)`` where ``spatial_attn_map`` is
        the sigmoid gate in ``[B, 1, D, H, W]``, useful for auxiliary supervision."""
        # --- Channel Attention ---
        gap = x.mean(dim=(2, 3, 4), keepdim=True)   # B,C,1,1,1
        gmp = x.amax(dim=(2, 3, 4), keepdim=True)   # B,C,1,1,1  (mask-safe)
        mc = torch.sigmoid(self.channel_mlp(gap + gmp))
        fc = x * mc

        # --- Spatial Attention (applied on channel-attended features) ---
        avg_s = fc.mean(dim=1, keepdim=True)         # B,1,D,H,W
        max_s = fc.amax(dim=1, keepdim=True)         # B,1,D,H,W
        ms = torch.sigmoid(self.spatial_dc(torch.cat([avg_s, max_s], dim=1)))
        fs = fc * ms

        return x + self.gate_scale * fs, ms


class _SeparableFanBlock(nn.Module):
    """Cheap longitudinal + lateral residual block for BEV dose tensors."""

    def __init__(
        self,
        channels: int,
        depth_kernel_size: int = 7,
        dropout: float = 0.0,
        mix_ratio: float = 0.5,
        use_depth_attention: bool = False,
        use_lateral_attention: bool = False,
        attention_heads: int = 1,
        attention_dim: int | None = None,
        use_se_attention: bool = False,
        se_ratio: float = 0.25,
        use_scsam: bool = False,
        scsam_reduction: int = 4,
        scsam_dilation: tuple[int, int, int] = (2, 1, 1),
        use_repvgg: bool = False,
        norm_kind: str = "group",
        conditioning_dim: int = 0,
    ) -> None:
        super().__init__()
        channels = int(channels)
        depth_kernel_size = max(3, int(depth_kernel_size) | 1)
        mix_channels = max(4, int(round(channels * float(mix_ratio))))
        attention_dim = channels if attention_dim is None else int(attention_dim)
        attention_dim = max(1, min(channels, attention_dim))
        attention_heads = max(1, int(attention_heads))
        if attention_dim % attention_heads != 0:
            raise ValueError(f"attention_dim={attention_dim} must be divisible by attention_heads={attention_heads}")
        self.use_depth_attention = bool(use_depth_attention)
        self.use_lateral_attention = bool(use_lateral_attention)
        self.use_se_attention = bool(use_se_attention)
        self.use_repvgg = bool(use_repvgg)
        self.reparam_deployed = False
        self.attention_heads = attention_heads
        self.attention_dim = attention_dim
        self.head_dim = attention_dim // attention_heads
        self.pre = nn.Sequential(
            _conditioned_norm3d(norm_kind, channels, conditioning_dim),
            nn.SiLU(),
        )
        if self.use_repvgg:
            self.depth_main = nn.Conv3d(
                channels,
                channels,
                kernel_size=(depth_kernel_size, 1, 1),
                padding=(depth_kernel_size // 2, 0, 0),
                groups=channels,
                bias=True,
            )
            self.depth_1x1 = nn.Conv3d(
                channels,
                channels,
                kernel_size=1,
                groups=channels,
                bias=False,
            )
            self.lat_main = nn.Conv3d(
                channels,
                channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                groups=channels,
                bias=True,
            )
            self.lat_1x1 = nn.Conv3d(
                channels,
                channels,
                kernel_size=1,
                groups=channels,
                bias=False,
            )
            nn.init.zeros_(self.depth_1x1.weight)
            nn.init.zeros_(self.lat_1x1.weight)
        else:
            self.depth = nn.Conv3d(
                channels,
                channels,
                kernel_size=(depth_kernel_size, 1, 1),
                padding=(depth_kernel_size // 2, 0, 0),
                groups=channels,
            )
            self.lateral = nn.Conv3d(
                channels,
                channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                groups=channels,
            )
        if self.use_depth_attention:
            self.depth_attn_norm = _norm3d(norm_kind, channels)
            self.depth_attn_qkv = nn.Conv3d(channels, attention_dim * 3, kernel_size=1)
            self.depth_attn_out = nn.Conv3d(attention_dim, channels, kernel_size=1)
            nn.init.zeros_(self.depth_attn_out.weight)
            nn.init.zeros_(self.depth_attn_out.bias)
        else:
            self.depth_attn_norm = None
            self.depth_attn_qkv = None
            self.depth_attn_out = None
        if self.use_lateral_attention:
            self.lateral_attn_norm = _norm3d(norm_kind, channels)
            self.lateral_attn_qkv = nn.Conv3d(channels, attention_dim * 3, kernel_size=1)
            self.lateral_attn_out = nn.Conv3d(attention_dim, channels, kernel_size=1)
            nn.init.zeros_(self.lateral_attn_out.weight)
            nn.init.zeros_(self.lateral_attn_out.bias)
        else:
            self.lateral_attn_norm = None
            self.lateral_attn_qkv = None
            self.lateral_attn_out = None
        if self.use_se_attention:
            se_channels = max(1, int(round(channels * float(se_ratio))))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(channels, se_channels, kernel_size=1),
                nn.SiLU(),
                nn.Conv3d(se_channels, channels, kernel_size=1),
                nn.Sigmoid(),
            )
            final = self.se[-2]
            if isinstance(final, nn.Conv3d):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
        else:
            self.se = None
        # SCSAM — when enabled, replaces SE in the forward pass
        self.use_scsam = bool(use_scsam)
        self.scsam: _SCSAM3D | None = (
            _SCSAM3D(
                channels=channels,
                reduction=int(scsam_reduction),
                dilation=tuple(int(d) for d in scsam_dilation),  # type: ignore[arg-type]
            )
            if self.use_scsam
            else None
        )
        self.dropout = nn.Dropout3d(float(dropout))
        self.mix = nn.Sequential(
            _conditioned_norm3d(norm_kind, channels, conditioning_dim),
            nn.SiLU(),
            nn.Conv3d(channels, mix_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(mix_channels, channels, kernel_size=1),
        )

    def _depth_attention(self, x: torch.Tensor) -> torch.Tensor:
        if self.depth_attn_norm is None or self.depth_attn_qkv is None or self.depth_attn_out is None:
            return torch.zeros_like(x)
        b, _c, d, h, w = x.shape
        qkv = self.depth_attn_qkv(F.silu(self.depth_attn_norm(x)))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.permute(0, 3, 4, 1, 2).reshape(b * h * w, self.attention_heads, self.head_dim, d).transpose(-1, -2)
        k = k.permute(0, 3, 4, 1, 2).reshape(b * h * w, self.attention_heads, self.head_dim, d).transpose(-1, -2)
        v = v.permute(0, 3, 4, 1, 2).reshape(b * h * w, self.attention_heads, self.head_dim, d).transpose(-1, -2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        y = y.transpose(-1, -2).reshape(b, h, w, self.attention_dim, d).permute(0, 3, 4, 1, 2).contiguous()
        return self.depth_attn_out(y)

    def _lateral_attention(self, x: torch.Tensor) -> torch.Tensor:
        if self.lateral_attn_norm is None or self.lateral_attn_qkv is None or self.lateral_attn_out is None:
            return torch.zeros_like(x)
        b, _c, d, h, w = x.shape
        n = h * w
        qkv = self.lateral_attn_qkv(F.silu(self.lateral_attn_norm(x)))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.permute(0, 2, 1, 3, 4).reshape(b * d, self.attention_heads, self.head_dim, n).transpose(-1, -2)
        k = k.permute(0, 2, 1, 3, 4).reshape(b * d, self.attention_heads, self.head_dim, n).transpose(-1, -2)
        v = v.permute(0, 2, 1, 3, 4).reshape(b * d, self.attention_heads, self.head_dim, n).transpose(-1, -2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        y = y.transpose(-1, -2).reshape(b, d, self.attention_dim, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return self.lateral_attn_out(y)

    @staticmethod
    def _pad_1x1_to_main(weight: torch.Tensor, main_weight: torch.Tensor) -> torch.Tensor:
        padded = torch.zeros_like(main_weight)
        center = tuple(size // 2 for size in main_weight.shape[2:])
        padded[(slice(None), slice(None), *center)] = weight.flatten(2).squeeze(-1)
        return padded

    def reparameterize(self) -> None:
        """Fold RepVGG auxiliary branches into their depthwise main convolutions."""
        if not self.use_repvgg or self.reparam_deployed:
            return
        with torch.no_grad():
            self.depth_main.weight.add_(self._pad_1x1_to_main(self.depth_1x1.weight, self.depth_main.weight))
            self.lat_main.weight.add_(self._pad_1x1_to_main(self.lat_1x1.weight, self.lat_main.weight))
        del self.depth_1x1
        del self.lat_1x1
        self.reparam_deployed = True

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Returns ``(output, attn_maps)`` where ``attn_maps`` is a (possibly
        empty) list of spatial attention tensors ``[B, 1, D, H, W]``."""
        h = self.pre[1](_apply_norm3d(self.pre[0], x, conditioning))
        if self.use_repvgg:
            spatial = self.depth_main(h) + self.lat_main(h)
            if not self.reparam_deployed:
                spatial = spatial + self.depth_1x1(h) + self.lat_1x1(h)
        else:
            spatial = self.depth(h) + self.lateral(h)
        if self.use_depth_attention:
            spatial = spatial + self._depth_attention(x)
        if self.use_lateral_attention:
            spatial = spatial + self._lateral_attention(x)
        y = x + self.dropout(spatial)
        attn_maps: list[torch.Tensor] = []
        if self.scsam is not None:
            y, ms = self.scsam(y)
            attn_maps.append(ms)
        elif self.se is not None:
            y = y * (1.0 + self.se(y))
        mixed = _apply_norm3d(self.mix[0], y, conditioning)
        for layer in self.mix[1:]:
            mixed = layer(mixed)
        return y + mixed, attn_maps


class SeparableFanGridConvCorrector(nn.Module):
    """Fast BEV correction model with separated longitudinal and lateral filters.

    The public forward API matches ``FanGridConvCorrector``:
    ``forward(features, dose_pb, valid_mask=None, fan_mask=None, material_id=None, energy=None)``.
    The expensive full-channel spatial convolutions are replaced by depthwise
    depth/lateral filters plus a small pointwise channel mixer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        num_layers: int = 4,
        depth_kernel_size: int = 7,
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
        use_depth_attention: bool = False,
        use_lateral_attention: bool = False,
        attention_heads: int = 1,
        attention_dim: int | None = None,
        attention_layers: str = "all",
        use_se_attention: bool = False,
        se_ratio: float = 0.25,
        use_scsam: bool = False,
        scsam_reduction: int = 4,
        scsam_dilation: tuple[int, int, int] = (2, 1, 1),
        use_repvgg: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.residual_mode = str(residual_mode).lower()
        if self.residual_mode not in {"additive", "multiplicative"}:
            raise ValueError("residual_mode must be 'additive' or 'multiplicative'")
        self.additive_scale_frac = float(additive_scale_frac)
        self.material_embedding_dim = int(material_embedding_dim)
        self.num_materials = int(num_materials)
        self.eps = float(eps)
        self.use_depth_attention = bool(use_depth_attention)
        self.use_lateral_attention = bool(use_lateral_attention)
        self.attention_heads = max(1, int(attention_heads))
        self.attention_dim = None if attention_dim is None else int(attention_dim)
        self.attention_layers = str(attention_layers).lower()
        if self.attention_layers not in {"all", "last"}:
            raise ValueError("attention_layers must be 'all' or 'last'")
        self.use_se_attention = bool(use_se_attention)
        self.se_ratio = float(se_ratio)
        self.use_scsam = bool(use_scsam)
        self.use_repvgg = bool(use_repvgg)
        self.scsam_reduction = int(scsam_reduction)
        self.scsam_dilation: tuple[int, int, int] = tuple(int(d) for d in scsam_dilation)  # type: ignore[assignment]
        self.material_embedding = (
            nn.Embedding(self.num_materials, self.material_embedding_dim)
            if self.material_embedding_dim > 0
            else None
        )
        if available_energies is not None:
            energies_t = torch.as_tensor(available_energies, dtype=torch.float32).sort().values
            self.register_buffer("available_energies", energies_t)
            self.energy_embed = nn.Embedding(len(energies_t), self.hidden_dim)
            nn.init.zeros_(self.energy_embed.weight)
        else:
            self.register_buffer("available_energies", None)
            self.energy_embed = None
        self.use_sigma_conditioning = bool(use_sigma_conditioning)
        self.sigma_project = nn.Sequential(
            nn.Linear(2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        ) if self.use_sigma_conditioning else None
        if self.sigma_project is not None:
            nn.init.zeros_(self.sigma_project[-1].weight)
            nn.init.zeros_(self.sigma_project[-1].bias)

        total_input_dim = self.input_dim + max(self.material_embedding_dim, 0)
        self.in_proj = nn.Conv3d(total_input_dim, self.hidden_dim, kernel_size=1)
        blocks = []
        n_layers = int(num_layers)
        for layer_idx in range(n_layers):
            use_attention_here = self.attention_layers == "all" or layer_idx == n_layers - 1
            blocks.append(
                _SeparableFanBlock(
                    self.hidden_dim,
                    depth_kernel_size=depth_kernel_size,
                    dropout=dropout,
                    mix_ratio=mix_ratio,
                    use_depth_attention=self.use_depth_attention and use_attention_here,
                    use_lateral_attention=self.use_lateral_attention and use_attention_here,
                    attention_heads=self.attention_heads,
                    attention_dim=self.attention_dim,
                    use_se_attention=self.use_se_attention and use_attention_here,
                    se_ratio=self.se_ratio,
                    use_scsam=self.use_scsam and use_attention_here,
                    scsam_reduction=self.scsam_reduction,
                    scsam_dilation=self.scsam_dilation,
                    use_repvgg=self.use_repvgg,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.GroupNorm(_groups(self.hidden_dim), self.hidden_dim)
        self.head = nn.Conv3d(self.hidden_dim, 1, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @classmethod
    def from_config(
        cls,
        input_dim: int,
        cfg: dict[str, Any],
        available_energies: list[float] | torch.Tensor | None = None,
    ) -> SeparableFanGridConvCorrector:
        model_cfg = cfg.get("model", {})
        return cls(
            input_dim=input_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 16)),
            num_layers=int(model_cfg.get("num_layers", 4)),
            depth_kernel_size=int(model_cfg.get("depth_kernel_size", 7)),
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
            use_depth_attention=bool(model_cfg.get("use_depth_attention", False)),
            use_lateral_attention=bool(model_cfg.get("use_lateral_attention", False)),
            attention_heads=int(model_cfg.get("attention_heads", 1)),
            attention_dim=model_cfg.get("attention_dim", None),
            attention_layers=str(model_cfg.get("attention_layers", "all")),
            use_se_attention=bool(model_cfg.get("use_se_attention", False)),
            se_ratio=float(model_cfg.get("se_ratio", 0.25)),
            use_scsam=bool(model_cfg.get("use_scsam", False)),
            scsam_reduction=int(model_cfg.get("scsam_reduction", 4)),
            scsam_dilation=tuple(int(d) for d in model_cfg.get("scsam_dilation", [2, 1, 1])),  # type: ignore[arg-type]
            use_repvgg=bool(model_cfg.get("use_repvgg", False)),
        )

    def fuse_repvgg(self) -> None:
        """Fold all enabled RepVGG blocks into their inference-time convolutions."""
        for block in self.blocks:
            block.reparameterize()

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
        idx = (energy_flat.unsqueeze(-1) - self.available_energies.unsqueeze(0)).abs().argmin(dim=-1)
        return idx

    def forward(
        self,
        features: torch.Tensor,
        dose_pb: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        fan_mask: torch.Tensor | None = None,
        material_id: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
        sigma_mm: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        material_features = self._material_features(material_id, features)
        if material_features is not None:
            features = torch.cat((features, material_features), dim=1)
        hidden = self.in_proj(features)
        energy_idx = self._energy_to_index(energy, int(features.shape[0]), features.device)
        if energy_idx is not None:
            energy_vec = self.energy_embed(energy_idx).view(-1, self.hidden_dim, 1, 1, 1)
            hidden = hidden + energy_vec
        if self.sigma_project is not None and sigma_mm is not None:
            sigma_vec = sigma_mm.to(device=features.device, dtype=features.dtype).reshape(-1, 2)
            if sigma_vec.shape[0] == 1 and int(features.shape[0]) != 1:
                sigma_vec = sigma_vec.expand(int(features.shape[0]), -1)
            sigma_cond = self.sigma_project(sigma_vec).view(-1, self.hidden_dim, 1, 1, 1)
            hidden = hidden + sigma_cond
        all_attn_maps: list[torch.Tensor] = []
        for block in self.blocks:
            if self.grad_checkpoint and self.training and hidden.requires_grad:
                hidden, block_maps = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden, block_maps = block(hidden)
            all_attn_maps.extend(block_maps)
        residual = self.head(torch.nn.functional.silu(self.norm(hidden)))
        if self.residual_mode == "additive":
            scale = dose_pb.detach().amax(dim=(2, 3, 4), keepdim=True).clamp_min(torch.finfo(dose_pb.dtype).tiny)
            dose_hat = dose_pb + residual * scale * self.additive_scale_frac
        else:
            dose_hat = (dose_pb + self.eps).clamp_min(self.eps) * torch.exp(residual) - self.eps
        mask = dose_pb > 0.0 if valid_mask is None else valid_mask
        if fan_mask is not None:
            mask = mask & fan_mask
        dose_hat = torch.where(mask, dose_hat.clamp_min(0.0), torch.zeros_like(dose_hat))
        return {"residual": residual, "dose_hat": dose_hat, "attn_maps": all_attn_maps}
