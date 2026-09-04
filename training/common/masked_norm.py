"""GroupNorm that ignores distal depth padding, so a padded forward equals a native one.

Why this exists: padding the BEV depth to a fixed value is what lets ``torch.compile`` emit
static-shape kernels (native D varies per beam -- 36 gantry angles per case -- so dynamic
shapes are otherwise unavoidable). But plain ``nn.GroupNorm`` computes its statistics over
(C/G, D, H, W), so distal zero padding enlarges D and shifts the mean and variance,
rescaling every real voxel. Measured on the 887k geometry that moves the output by 26% of
the residual's own magnitude -- see tests/unittests/test_model_depth_padding.py.

Two things are needed to make padding a no-op, and only doing the first is not enough:

1. Compute the statistics over the valid region only.
2. Re-zero the padded region on the way out. Normalisation applies ``* w + b`` everywhere,
   so the pad would otherwise leave the norm holding ``b`` rather than zero, and the next
   convolution -- depth kernel 11, so a reach of +-5 voxels per layer -- would pull that
   back across the boundary into real voxels, compounding stage by stage.

``valid_depth`` is carried as a **tensor**, not a Python int, on purpose: it changes per
beam, and a Python int would be burned in as a compile-time constant and force a recompile
for every beam -- reintroducing exactly the thrash that padding is meant to remove. As a
tensor it is runtime data, so the compiled graph stays shape-static and value-agnostic.
"""
from __future__ import annotations

import torch
from torch import nn


class MaskedGroupNorm3d(nn.Module):
    """``nn.GroupNorm`` over ``[B, C, D, H, W]`` restricted to ``[:, :, :valid_depth]``.

    With ``valid_depth`` unset (or >= D) this is exactly ``nn.GroupNorm``, so a model
    converted by :func:`convert_group_norms` behaves identically until padding is used.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float, affine: bool):
        super().__init__()
        self.num_groups = int(num_groups)
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        # Shared across every instance in a model; set by set_valid_depth().
        self.valid_depth: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        G = self.num_groups
        xg = x.reshape(B, G, C // G, D, H, W)

        vd = self.valid_depth
        if vd is None:
            mean = xg.mean(dim=(2, 3, 4, 5), keepdim=True)
            var = xg.var(dim=(2, 3, 4, 5), keepdim=True, unbiased=False)
            mask = None
        else:
            # [1, 1, 1, D, 1, 1] float mask over the depth axis.
            idx = torch.arange(D, device=x.device).view(1, 1, 1, D, 1, 1)
            mask = (idx < vd.to(device=x.device)).to(dtype=x.dtype)
            n = mask.sum() * (C // G) * H * W
            mean = (xg * mask).sum(dim=(2, 3, 4, 5), keepdim=True) / n
            # Variance over the valid region only: the pad contributes no deviation.
            var = (((xg - mean) * mask) ** 2).sum(dim=(2, 3, 4, 5), keepdim=True) / n

        out = (xg - mean) * torch.rsqrt(var + self.eps)
        if mask is not None:
            out = out * mask
        out = out.reshape(B, C, D, H, W)
        if self.weight is not None:
            out = out * self.weight.view(1, C, 1, 1, 1) + self.bias.view(1, C, 1, 1, 1)
        if mask is not None:
            # Re-zero the pad: the affine shift above would otherwise leave `bias` there,
            # and the next convolution would read it back into the real region.
            out = out * mask.reshape(1, 1, D, 1, 1)
        return out

    def extra_repr(self) -> str:
        return f"{self.num_groups}, {self.num_channels}, eps={self.eps}, affine={self.weight is not None}"


def convert_group_norms(model: nn.Module) -> int:
    """Replace every ``nn.GroupNorm`` in ``model`` in place, preserving weights.

    Returns the number of modules converted. Safe on an already-converted model (0).
    """
    converted = 0
    for name, child in list(model.named_children()):
        if isinstance(child, nn.GroupNorm):
            repl = MaskedGroupNorm3d(
                child.num_groups, child.num_channels, child.eps, child.affine
            )
            if child.affine:
                with torch.no_grad():
                    repl.weight.copy_(child.weight)
                    repl.bias.copy_(child.bias)
            repl.to(device=child.weight.device if child.affine else None)
            setattr(model, name, repl)
            converted += 1
        else:
            converted += convert_group_norms(child)
    return converted


def set_valid_depth(model: nn.Module, valid_depth: torch.Tensor | int | None) -> None:
    """Point every masked norm in ``model`` at the current beam's real depth.

    Pass ``None`` to fall back to plain GroupNorm behaviour. An ``int`` is accepted for
    tests but converted to a tensor -- see the module docstring on why a Python int must
    not reach a compiled graph.
    """
    if isinstance(valid_depth, int):
        valid_depth = torch.tensor(valid_depth)
    for module in model.modules():
        if isinstance(module, MaskedGroupNorm3d):
            module.valid_depth = valid_depth
