from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class DoseTransformerDirect(nn.Module):
    """Low-resolution DoTA-style direct beamlet dose predictor.

    The model encodes each BEV depth slice independently, attends causally over depth
    tokens with an energy token prepended, then decodes one dose slice per token.
    Input/output tensors use ``[B, C, D, H, W]`` and ``[B, 1, D, H, W]``.
    """

    def __init__(
        self,
        input_channels: int,
        depth: int = 150,
        height: int = 24,
        width: int = 24,
        latent_hw: tuple[int, int] = (6, 6),
        latent_channels: int = 12,
        num_heads: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
        energy_embed_dim: int | None = None,
        available_energies: list[float] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.depth = int(depth)
        self.height = int(height)
        self.width = int(width)
        self.latent_hw = tuple(int(v) for v in latent_hw)
        self.latent_channels = int(latent_channels)
        self.token_dim = self.latent_channels * self.latent_hw[0] * self.latent_hw[1]
        self.num_heads = int(num_heads)
        if self.token_dim % self.num_heads != 0:
            raise ValueError(f"token_dim={self.token_dim} must be divisible by num_heads={self.num_heads}")

        self.slice_encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, self.latent_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.depth + 1, self.token_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        if available_energies is not None:
            energies_t = torch.as_tensor(available_energies, dtype=torch.float32).sort().values
            self.register_buffer("available_energies", energies_t)
            self.energy_embed = nn.Embedding(len(energies_t), self.token_dim)
        else:
            self.register_buffer("available_energies", None)
            energy_embed_dim = self.token_dim if energy_embed_dim is None else int(energy_embed_dim)
            self.energy_mlp = nn.Sequential(
                nn.Linear(1, energy_embed_dim),
                nn.SiLU(),
                nn.Linear(energy_embed_dim, self.token_dim),
            )
            self.energy_embed = None

        layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=self.num_heads,
            dim_feedforward=self.token_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.token_norm = nn.LayerNorm(self.token_dim)

        self.slice_decoder = nn.Sequential(
            nn.Conv2d(self.latent_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Upsample(size=(self.height, self.width), mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        nn.init.zeros_(self.slice_decoder[-1].weight)
        nn.init.zeros_(self.slice_decoder[-1].bias)

    @classmethod
    def from_config(
        cls,
        input_channels: int,
        cfg: dict[str, Any],
        available_energies: list[float] | torch.Tensor | None = None,
    ) -> DoseTransformerDirect:
        model_cfg = cfg.get("model", {})
        return cls(
            input_channels=input_channels,
            depth=int(model_cfg.get("depth", 150)),
            height=int(model_cfg.get("height", 24)),
            width=int(model_cfg.get("width", 24)),
            latent_hw=tuple(int(v) for v in model_cfg.get("latent_hw", [6, 6])),
            latent_channels=int(model_cfg.get("latent_channels", 12)),
            num_heads=int(model_cfg.get("num_heads", 16)),
            num_layers=int(model_cfg.get("num_layers", 1)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            available_energies=available_energies,
        )

    def _energy_token(self, energy: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if energy is None:
            energy = torch.zeros(batch_size, device=device)
        energy = energy.to(device=device, dtype=torch.float32).flatten()
        if energy.numel() == 1 and batch_size > 1:
            energy = energy.expand(batch_size)
        if energy.numel() != batch_size:
            raise ValueError(f"energy must have {batch_size} values, got {energy.numel()}")
        if self.energy_embed is not None and self.available_energies is not None:
            idx = torch.argmin((energy[:, None] - self.available_energies[None, :].to(device=device)).abs(), dim=1)
            return self.energy_embed(idx)
        return self.energy_mlp((energy / 250.0).view(batch_size, 1))

    def forward(self, x: torch.Tensor, energy: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"x must be [B,C,D,H,W], got {tuple(x.shape)}")
        b, c, d, h, w = x.shape
        if c != self.input_channels:
            raise ValueError(f"expected {self.input_channels} input channels, got {c}")
        if d != self.depth or h != self.height or w != self.width:
            raise ValueError(
                f"expected spatial shape {(self.depth, self.height, self.width)}, got {(d, h, w)}"
            )
        slices = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        latent = self.slice_encoder(slices)
        latent = F.adaptive_avg_pool2d(latent, self.latent_hw)
        tokens = latent.reshape(b, d, self.token_dim)
        energy_token = self._energy_token(energy, b, x.device).unsqueeze(1)
        tokens = torch.cat([energy_token, tokens], dim=1) + self.pos_embed[:, : d + 1]

        causal_mask = torch.full((d + 1, d + 1), float("-inf"), device=x.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        hidden = self.transformer(tokens, mask=causal_mask)
        dose_tokens = self.token_norm(hidden[:, 1:])
        dose_latent = dose_tokens.reshape(b * d, self.latent_channels, self.latent_hw[0], self.latent_hw[1])
        dose_slices = self.slice_decoder(dose_latent)
        dose = dose_slices.reshape(b, d, 1, self.height, self.width).permute(0, 2, 1, 3, 4).contiguous()
        direct = F.softplus(dose) - torch.log(dose.new_tensor(2.0))
        return {"dose_hat": direct, "residual": dose}
