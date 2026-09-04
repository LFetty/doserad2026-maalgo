from __future__ import annotations

from typing import Any

import math
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class RaySequenceCorrector(nn.Module):
    """Causal Transformer central-ray dose corrector with fixed support context."""

    def __init__(
        self,
        central_dim: int,
        support_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.05,
        material_embedding_dim: int = 16,
        num_materials: int = 86,
        eps: float = 1e-3,
        predict_support: bool = False,
        support_only: bool = False,
        support_attention: str = "dense",
        support_depth_backend: str = "transformer",
        residual_mode: str = "multiplicative",
        additive_scale_frac: float = 0.25,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.hidden_dim = int(hidden_dim)
        self.predict_support = bool(predict_support)
        self.support_only = bool(support_only)
        self.support_attention_kind = str(support_attention).lower()
        self.support_depth_backend = str(support_depth_backend).lower()
        self.residual_mode = str(residual_mode).lower()
        self.additive_scale_frac = float(additive_scale_frac)
        if self.residual_mode not in {"multiplicative", "additive"}:
            raise ValueError("residual_mode must be 'multiplicative' or 'additive'")
        self.causal = bool(causal)
        self.central_embedding = nn.Sequential(
            nn.Linear(central_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.support_embedding = nn.Sequential(
            nn.Linear(support_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.material_embedding = nn.Embedding(int(num_materials), int(material_embedding_dim))
        self.material_project = nn.Linear(int(material_embedding_dim), hidden_dim)
        self.support_context_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.dose_embed = nn.Linear(2, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=hidden_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._zero_init_residual_head(self.head)
        if self.predict_support:
            if self.support_depth_backend == "sdpa":
                self.support_depth_transformer = _SDPADepthEncoder(
                    hidden_dim=hidden_dim,
                    num_heads=int(num_heads),
                    dropout=float(dropout),
                    num_layers=1,
                )
            elif self.support_depth_backend == "unet":
                self.support_depth_transformer = _ConvUNetDepthEncoder(hidden_dim=hidden_dim, dropout=float(dropout))
            elif self.support_depth_backend == "transformer":
                depth_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=int(num_heads),
                    dim_feedforward=hidden_dim * 4,
                    dropout=float(dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.support_depth_transformer = nn.TransformerEncoder(depth_layer, num_layers=1)
            else:
                raise ValueError("support_depth_backend must be 'transformer', 'sdpa', or 'unet'")
            if self.support_attention_kind == "linear":
                self.support_axis_attention = _SupportLinearAttention(hidden_dim, int(num_heads), dropout=float(dropout))
            elif self.support_attention_kind == "dense":
                self.support_axis_attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=int(num_heads),
                    dropout=float(dropout),
                    batch_first=True,
                )
            elif self.support_attention_kind in {"none", "depth", "longitudinal"}:
                self.support_attention_kind = "none"
            else:
                raise ValueError("support_attention must be 'dense', 'linear', or 'none'")
            self.support_norm = nn.LayerNorm(hidden_dim)
            self.support_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            self._zero_init_residual_head(self.support_head)

    @staticmethod
    def _zero_init_residual_head(head: nn.Sequential) -> None:
        final = head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    @classmethod
    def from_config(cls, central_dim: int, support_dim: int, cfg: dict[str, Any]) -> RaySequenceCorrector:
        model_cfg = cfg.get("model", {})
        return cls(
            central_dim=central_dim,
            support_dim=support_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            num_layers=int(model_cfg.get("num_layers", model_cfg.get("num_blocks", 4))),
            num_heads=int(model_cfg.get("num_heads", 4)),
            dropout=float(model_cfg.get("dropout", 0.05)),
            material_embedding_dim=int(model_cfg.get("material_embedding_dim", 16)),
            num_materials=int(model_cfg.get("num_materials", 86)),
            eps=float(cfg.get("dose", {}).get("eps", 1e-3)),
            predict_support=bool(model_cfg.get("predict_support", False)),
            support_only=bool(model_cfg.get("support_only", False)),
            support_attention=str(model_cfg.get("support_attention", "dense")),
            support_depth_backend=str(model_cfg.get("support_depth_backend", "transformer")),
            residual_mode=str(model_cfg.get("residual_mode", "multiplicative")),
            additive_scale_frac=float(model_cfg.get("additive_scale_frac", 0.25)),
            causal=bool(model_cfg.get("causal", True)),
        )

    def _position_encoding(self, depth_count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pos = torch.arange(depth_count, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.hidden_dim, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / max(self.hidden_dim, 1))
        )
        enc = torch.zeros((depth_count, self.hidden_dim), device=device, dtype=dtype)
        enc[:, 0::2] = torch.sin(pos * div)
        enc[:, 1::2] = torch.cos(pos * div[: enc[:, 1::2].shape[1]])
        return enc.unsqueeze(0)

    def _encode_context(
        self,
        central_features: torch.Tensor,
        support_features: torch.Tensor,
        support_mask: torch.Tensor,
        central_material_id: torch.Tensor,
        support_material_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, seq_count, depth_count, _ = central_features.shape
        flat_steps = b * seq_count
        num_emb = self.material_embedding.num_embeddings

        central_h = self.central_embedding(central_features)
        central_h = central_h + self.material_project(
            self.material_embedding(central_material_id.clamp(0, num_emb - 1))
        )

        support_h = self.support_embedding(support_features)
        support_h = support_h + self.material_project(
            self.material_embedding(support_material_id.clamp(0, num_emb - 1))
        )
        _, _, _, support_count, hidden_dim = support_h.shape
        support_flat = support_h.reshape(b * seq_count * depth_count, support_count, hidden_dim)
        central_query = central_h.reshape(b * seq_count * depth_count, 1, hidden_dim)
        support_key_padding = ~support_mask.reshape(b * seq_count * depth_count, support_count).bool()
        all_invalid = support_key_padding.all(dim=1)
        if all_invalid.any():
            support_key_padding = support_key_padding.clone()
            support_key_padding[all_invalid, 0] = False
        support_context, _ = self.support_context_attention(
            central_query,
            support_flat,
            support_flat,
            key_padding_mask=support_key_padding,
            need_weights=False,
        )
        support_h = support_context.reshape(b, seq_count, depth_count, hidden_dim)

        central_h = central_h.reshape(flat_steps, depth_count, -1)
        support_h = support_h.reshape(flat_steps, depth_count, -1)
        return central_h, support_h

    def _decode(
        self,
        central_h: torch.Tensor,
        support_h: torch.Tensor,
        dose_pb: torch.Tensor,
        previous_sequence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        prev_log = torch.log((previous_sequence + self.eps).clamp_min(self.eps)).unsqueeze(-1)
        pb_log = torch.log((dose_pb + self.eps).clamp_min(self.eps)).unsqueeze(-1)
        dose_h = self.dose_embed(torch.cat((prev_log, pb_log), dim=-1))
        tokens = central_h + support_h + dose_h
        tokens = tokens + self._position_encoding(tokens.shape[1], tokens.device, tokens.dtype)
        causal_mask = None
        if self.causal:
            causal_mask = torch.triu(
                torch.ones((tokens.shape[1], tokens.shape[1]), device=tokens.device, dtype=torch.bool),
                diagonal=1,
            )
        hidden = self.transformer(tokens, mask=causal_mask)
        r_pred = self.head(hidden).squeeze(-1)
        if self.residual_mode == "additive":
            scale = dose_pb.detach().amax(dim=1, keepdim=True).clamp_min(torch.finfo(dose_pb.dtype).tiny)
            dose_hat = dose_pb + r_pred * scale * self.additive_scale_frac
        else:
            dose_hat = (dose_pb + self.eps).clamp_min(self.eps) * torch.exp(r_pred) - self.eps
        return {"r": r_pred, "dose_hat": dose_hat}

    def _decode_autoregressive(
        self,
        central_h: torch.Tensor,
        support_h: torch.Tensor,
        dose_pb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        predicted: list[torch.Tensor] = []
        for depth_idx in range(dose_pb.shape[1]):
            if depth_idx == 0:
                previous_sequence = dose_pb[:, :1]
            else:
                previous_sequence = torch.cat((dose_pb[:, :1], torch.stack(predicted, dim=1)), dim=1)
            prefix = self._decode(
                central_h[:, : depth_idx + 1],
                support_h[:, : depth_idx + 1],
                dose_pb[:, : depth_idx + 1],
                previous_sequence,
            )
            predicted.append(prefix["dose_hat"][:, -1])

        dose_hat = torch.stack(predicted, dim=1)
        r_pred = torch.log((dose_hat + self.eps).clamp_min(self.eps) / (dose_pb + self.eps).clamp_min(self.eps))
        return {"r": r_pred, "dose_hat": dose_hat}

    def forward(
        self,
        central_features: torch.Tensor,
        support_features: torch.Tensor,
        central_dose_pb: torch.Tensor,
        support_mask: torch.Tensor,
        central_material_id: torch.Tensor,
        support_material_id: torch.Tensor,
        support_dose_pb: torch.Tensor | None = None,
        teacher_dose: torch.Tensor | None = None,
        teacher_forcing: bool = False,
        decode_mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        b, seq_count, depth_count, _ = central_features.shape
        if self.predict_support and self.support_only:
            if support_dose_pb is None:
                raise ValueError("support_dose_pb is required when model.predict_support=True")
            return self._decode_support(
                central_features=central_features,
                support_features=support_features,
                support_mask=support_mask,
                central_material_id=central_material_id,
                support_material_id=support_material_id,
                support_dose_pb=support_dose_pb,
            )

        flat_steps = b * seq_count
        central_h, support_h = self._encode_context(
            central_features,
            support_features,
            support_mask,
            central_material_id,
            support_material_id,
        )
        dose_pb = central_dose_pb.reshape(flat_steps, depth_count)

        mode = (decode_mode or ("teacher_forced" if teacher_forcing and teacher_dose is not None else "causal")).lower()
        if mode in {"one_shot", "oneshot"}:
            mode = "causal"
        if mode in {"ar", "auto_regressive"}:
            mode = "autoregressive"

        if mode == "autoregressive":
            out = self._decode_autoregressive(central_h, support_h, dose_pb)
        elif mode == "causal":
            out = self._decode(central_h, support_h, dose_pb, dose_pb)
        elif mode == "teacher_forced":
            if teacher_dose is None:
                raise ValueError("teacher_dose is required when decode_mode='teacher_forced'")
            teacher = teacher_dose.reshape(flat_steps, depth_count)
            previous_sequence = torch.cat((dose_pb[:, :1], teacher[:, :-1]), dim=1)
            out = self._decode(central_h, support_h, dose_pb, previous_sequence)
        else:
            raise ValueError(
                "decode_mode must be one of 'causal', 'autoregressive', or 'teacher_forced', "
                f"got {decode_mode!r}"
            )

        result = {
            "r": out["r"].reshape(b, seq_count, depth_count),
            "dose_hat": out["dose_hat"].reshape(b, seq_count, depth_count),
        }
        if self.predict_support:
            if support_dose_pb is None:
                raise ValueError("support_dose_pb is required when model.predict_support=True")
            result.update(
                self._decode_support(
                    central_features=central_features,
                    support_features=support_features,
                    support_mask=support_mask,
                    central_material_id=central_material_id,
                    support_material_id=support_material_id,
                    support_dose_pb=support_dose_pb,
                )
            )
        return result

    def _decode_support(
        self,
        central_features: torch.Tensor,
        support_features: torch.Tensor,
        support_mask: torch.Tensor,
        central_material_id: torch.Tensor,
        support_material_id: torch.Tensor,
        support_dose_pb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ray_chunk_size = int(os.environ.get("PYDOSERT_SUPPORT_RAY_CHUNK_SIZE", "0"))
        support_count = int(support_features.shape[3])
        if ray_chunk_size > 0 and support_count > ray_chunk_size:
            r_chunks = []
            hat_chunks = []
            for start in range(0, support_count, ray_chunk_size):
                stop = min(start + ray_chunk_size, support_count)
                out = self._decode_support_impl(
                    central_features=central_features,
                    support_features=support_features[:, :, :, start:stop, :],
                    support_mask=support_mask[:, :, :, start:stop],
                    central_material_id=central_material_id,
                    support_material_id=support_material_id[:, :, :, start:stop],
                    support_dose_pb=support_dose_pb[:, :, :, start:stop],
                )
                r_chunks.append(out["support_r"])
                hat_chunks.append(out["support_dose_hat"])
            return {
                "support_r": torch.cat(r_chunks, dim=3),
                "support_dose_hat": torch.cat(hat_chunks, dim=3),
            }
        return self._decode_support_impl(
            central_features=central_features,
            support_features=support_features,
            support_mask=support_mask,
            central_material_id=central_material_id,
            support_material_id=support_material_id,
            support_dose_pb=support_dose_pb,
        )

    def _decode_support_impl(
        self,
        central_features: torch.Tensor,
        support_features: torch.Tensor,
        support_mask: torch.Tensor,
        central_material_id: torch.Tensor,
        support_material_id: torch.Tensor,
        support_dose_pb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b, seq_count, depth_count, support_count, _ = support_features.shape
        num_emb = self.material_embedding.num_embeddings
        hidden_dim = self.hidden_dim

        support_h = self.support_embedding(support_features)
        support_h = support_h + self.material_project(
            self.material_embedding(support_material_id.clamp(0, num_emb - 1))
        )
        central_h = self.central_embedding(central_features)
        central_h = central_h + self.material_project(
            self.material_embedding(central_material_id.clamp(0, num_emb - 1))
        )
        support_h = support_h + central_h.unsqueeze(3)

        depth_tokens = support_h.permute(0, 1, 3, 2, 4).reshape(b * seq_count * support_count, depth_count, hidden_dim)
        depth_tokens = depth_tokens + self._position_encoding(depth_count, depth_tokens.device, depth_tokens.dtype)
        chunk_size = int(os.environ.get("PYDOSERT_SUPPORT_DEPTH_CHUNK_SIZE", "64"))
        if chunk_size > 0 and depth_tokens.shape[0] > chunk_size:
            depth_h = torch.cat(
                [
                    self.support_depth_transformer(depth_tokens[start : start + chunk_size])
                    for start in range(0, depth_tokens.shape[0], chunk_size)
                ],
                dim=0,
            )
        else:
            depth_h = self.support_depth_transformer(depth_tokens)
        support_h = depth_h.reshape(b, seq_count, support_count, depth_count, hidden_dim).permute(0, 1, 3, 2, 4)

        if self.support_attention_kind == "none":
            support_h = self.support_norm(support_h)
            support_r = self.support_head(support_h).squeeze(-1)
        else:
            support_tokens = support_h.reshape(b * seq_count * depth_count, support_count, hidden_dim)
            support_key_padding = ~support_mask.reshape(b * seq_count * depth_count, support_count).bool()
            all_invalid = support_key_padding.all(dim=1)
            if all_invalid.any():
                support_key_padding = support_key_padding.clone()
                support_key_padding[all_invalid, 0] = False
            axis_chunk_size = int(os.environ.get("PYDOSERT_SUPPORT_AXIS_CHUNK_SIZE", "64"))
            if axis_chunk_size > 0 and support_tokens.shape[0] > axis_chunk_size:
                support_r_chunks = []
                for start in range(0, support_tokens.shape[0], axis_chunk_size):
                    token_chunk = support_tokens[start : start + axis_chunk_size]
                    mask_chunk = support_key_padding[start : start + axis_chunk_size]
                    if self.support_attention_kind == "linear":
                        mixed_chunk = self.support_axis_attention(token_chunk, key_padding_mask=mask_chunk)
                    else:
                        mixed_chunk, _ = self.support_axis_attention(
                            token_chunk,
                            token_chunk,
                            token_chunk,
                            key_padding_mask=mask_chunk,
                            need_weights=False,
                        )
                    chunk_h = self.support_norm(token_chunk + mixed_chunk)
                    support_r_chunks.append(self.support_head(chunk_h).squeeze(-1))
                support_r = torch.cat(support_r_chunks, dim=0).reshape(b, seq_count, depth_count, support_count)
            elif self.support_attention_kind == "linear":
                mixed = self.support_axis_attention(support_tokens, key_padding_mask=support_key_padding)
                support_h = self.support_norm((support_tokens + mixed).reshape(b, seq_count, depth_count, support_count, hidden_dim))
                support_r = self.support_head(support_h).squeeze(-1)
            else:
                mixed, _ = self.support_axis_attention(
                    support_tokens,
                    support_tokens,
                    support_tokens,
                    key_padding_mask=support_key_padding,
                    need_weights=False,
                )
                support_h = self.support_norm((support_tokens + mixed).reshape(b, seq_count, depth_count, support_count, hidden_dim))
                support_r = self.support_head(support_h).squeeze(-1)
        if self.residual_mode == "additive":
            scale = support_dose_pb.detach().amax(dim=(2, 3), keepdim=True).clamp_min(
                torch.finfo(support_dose_pb.dtype).tiny
            )
            support_hat = support_dose_pb + support_r * scale * self.additive_scale_frac
        else:
            support_hat = (support_dose_pb + self.eps).clamp_min(self.eps) * torch.exp(support_r) - self.eps
        return {
            "support_r": support_r,
            "support_dose_hat": support_hat,
        }


class _SupportLinearAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = int(hidden_dim) // self.num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, s, c = x.shape
        qkv = self.qkv(x).view(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = torch.nn.functional.elu(q) + 1.0
        k = torch.nn.functional.elu(k) + 1.0
        if key_padding_mask is not None:
            valid = (~key_padding_mask).to(x.dtype).view(b, s, 1, 1)
            k = k * valid
            v = v * valid
        kv = torch.einsum("bshm,bshn->bhmn", k, v)
        k_sum = k.sum(dim=1)
        out = torch.einsum("bshm,bhmn->bshn", q, kv)
        denom = torch.einsum("bshm,bhm->bsh", q, k_sum).clamp_min(1e-6).unsqueeze(-1)
        out = (out / denom).reshape(b, s, c)
        return self.out(self.dropout(out))


class _SDPADepthLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = int(hidden_dim) // self.num_heads
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(float(dropout))
        self.attn_dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, depth_count, hidden_dim = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(b, depth_count, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).reshape(b, depth_count, hidden_dim)
        x = x + self.dropout(self.out(attn))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class _SDPADepthEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0, num_layers: int = 1) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_SDPADepthLayer(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(int(num_layers))]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(1, min(8, out_channels // 4)), num_channels=out_channels),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(1, min(8, out_channels // 4)), num_channels=out_channels),
            nn.SiLU(),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class _ConvUNetDepthEncoder(nn.Module):
    """Small per-ray 1D U-Net over depth samples.

    Input/output shape matches the depth transformer: [ray_batch, depth, hidden].
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        mid_dim = hidden_dim * 2
        bottleneck_dim = hidden_dim * 4
        self.enc1 = _ConvBlock1D(hidden_dim, hidden_dim, dropout=dropout)
        self.down1 = nn.Conv1d(hidden_dim, mid_dim, kernel_size=4, stride=2, padding=1)
        self.enc2 = _ConvBlock1D(mid_dim, mid_dim, dropout=dropout)
        self.down2 = nn.Conv1d(mid_dim, bottleneck_dim, kernel_size=4, stride=2, padding=1)
        self.bottleneck = _ConvBlock1D(bottleneck_dim, bottleneck_dim, dropout=dropout)
        self.dec2 = _ConvBlock1D(bottleneck_dim + mid_dim, mid_dim, dropout=dropout)
        self.dec1 = _ConvBlock1D(mid_dim + hidden_dim, hidden_dim, dropout=dropout)
        self.out = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))
        u2 = F.interpolate(b, size=e2.shape[-1], mode="linear", align_corners=False)
        d2 = self.dec2(torch.cat((u2, e2), dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-1], mode="linear", align_corners=False)
        d1 = self.dec1(torch.cat((u1, e1), dim=1))
        y = self.out(d1).transpose(1, 2)
        return self.norm(y + residual)


class _FanConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        groups = max(1, min(8, channels // 4))
        self.depth = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=(5, 1, 1), padding=(2, 0, 0)),
            nn.Dropout3d(float(dropout)),
        )
        self.lateral = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.Dropout3d(float(dropout)),
        )
        self.mix = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.depth(x)
        x = x + self.lateral(x)
        return x + self.mix(x)


class FanGridConvCorrector(nn.Module):
    """Dense all-ray fan-grid correction model.

    Input shape is ``[B, C, D, H, W]`` where ``H x W`` is the native lateral fan
    lattice, usually ``73 x 73`` with a circular mask. The model is deliberately
    small and local: depth and lateral convolutions are separated so all rays can
    be processed without attention over millions of samples.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        num_layers: int = 4,
        dropout: float = 0.0,
        residual_mode: str = "additive",
        additive_scale_frac: float = 0.25,
        material_embedding_dim: int = 4,
        num_materials: int = 86,
        available_energies: list[float] | torch.Tensor | None = None,
        use_sigma_conditioning: bool = False,
        eps: float = 1e-3,
        grad_checkpoint: bool = True,
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
        self.material_embedding = (
            nn.Embedding(self.num_materials, self.material_embedding_dim)
            if self.material_embedding_dim > 0
            else None
        )
        if available_energies is not None:
            energies_t = torch.as_tensor(available_energies, dtype=torch.float32).sort().values
            self.register_buffer("available_energies", energies_t)
            self.energy_embed = nn.Embedding(len(energies_t), int(hidden_dim))
            nn.init.zeros_(self.energy_embed.weight)
        else:
            self.register_buffer("available_energies", None)
            self.energy_embed = None
        self.use_sigma_conditioning = bool(use_sigma_conditioning)
        self.sigma_project = nn.Sequential(
            nn.Linear(2, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        ) if self.use_sigma_conditioning else None
        if self.sigma_project is not None:
            nn.init.zeros_(self.sigma_project[-1].weight)
            nn.init.zeros_(self.sigma_project[-1].bias)
        total_input_dim = int(input_dim) + max(self.material_embedding_dim, 0)
        self.in_proj = nn.Conv3d(total_input_dim, int(hidden_dim), kernel_size=1)
        self.blocks = nn.ModuleList(
            [_FanConvBlock(int(hidden_dim), dropout=float(dropout)) for _ in range(int(num_layers))]
        )
        self.norm = nn.GroupNorm(max(1, min(8, int(hidden_dim) // 4)), int(hidden_dim))
        self.head = nn.Conv3d(int(hidden_dim), 1, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @classmethod
    def from_config(
        cls,
        input_dim: int,
        cfg: dict[str, Any],
        available_energies: list[float] | torch.Tensor | None = None,
    ) -> FanGridConvCorrector:
        model_cfg = cfg.get("model", {})
        return cls(
            input_dim=input_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 16)),
            num_layers=int(model_cfg.get("num_layers", 4)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            residual_mode=str(model_cfg.get("residual_mode", "additive")),
            additive_scale_frac=float(model_cfg.get("additive_scale_frac", 0.25)),
            material_embedding_dim=int(model_cfg.get("material_embedding_dim", 4)),
            num_materials=int(model_cfg.get("num_materials", 86)),
            available_energies=available_energies,
            use_sigma_conditioning=bool(model_cfg.get("use_sigma_conditioning", False)),
            eps=float(cfg.get("dose", {}).get("eps", 1e-3)),
            grad_checkpoint=bool(model_cfg.get("grad_checkpoint", True)),
        )

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
        for block in self.blocks:
            if self.grad_checkpoint and self.training and hidden.requires_grad:
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
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
        return {"residual": residual, "dose_hat": dose_hat}
