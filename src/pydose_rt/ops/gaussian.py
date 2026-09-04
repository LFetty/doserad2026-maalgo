"""Gaussian-footprint scatter ops.

Anti-aliased alternative to the 8-corner trilinear point splat. Each sample
contributes to a ``(2W+1)^3`` voxel footprint with weights from a separable
normalised 3D Gaussian evaluated at integer offsets relative to the sample's
fractional voxel position. Total mass per sample is conserved (per-axis
weights are renormalised to sum to 1).

Choosing ``sigma_voxels = 0.5`` and ``half_window = 1`` gives a 3^3 = 27-voxel
footprint that anti-aliases the regular per-ray sample lattice without
materially over-smoothing the dose.

These ops are forward-only (no autograd). The trilinear path remains the
default and retains full autograd support.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional at runtime
    triton = None
    tl = None


def _scatter_inputs(
    coords_mm: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if coords_mm.dim() < 2 or coords_mm.shape[-1] != 3:
        raise ValueError(f"coords_mm must end in a xyz axis, got {tuple(coords_mm.shape)}")
    if values.shape != coords_mm.shape[:-1]:
        raise ValueError(
            "values must match coords_mm without the trailing xyz axis, got "
            f"{tuple(values.shape)} vs {tuple(coords_mm.shape[:-1])}"
        )
    return coords_mm.reshape(-1, 3), values.reshape(-1)


def gaussian_scatter_add_inplace(
    dose_row_flat: torch.Tensor,
    coords_mm: torch.Tensor,
    values: torch.Tensor,
    resolution: tuple[float, float, float],
    ct_array_shape: tuple[int, int, int],
    sigma_voxels: float,
    half_window: int,
) -> None:
    """In-place Gaussian-footprint scatter into a flat dose buffer.

    Footprint is ``(2*half_window + 1)^3`` voxels centred on the floor of each
    sample's voxel coordinate. Per-axis weights are normalised so total mass
    per sample is conserved exactly.
    """
    coords_flat_mm, values_flat = _scatter_inputs(coords_mm, values)
    if coords_flat_mm.numel() == 0 or values_flat.numel() == 0:
        return
    if sigma_voxels <= 0.0:
        raise ValueError(f"sigma_voxels must be > 0, got {sigma_voxels!r}")
    if half_window < 0:
        raise ValueError(f"half_window must be >= 0, got {half_window!r}")

    res_h, res_d, res_w = (float(v) for v in resolution)
    size_h, size_d, size_w = (int(v) for v in ct_array_shape)
    max_h, max_d, max_w = size_h - 1, size_d - 1, size_w - 1
    depth_stride = size_d * size_w
    width_stride = size_w
    eps = torch.finfo(values_flat.dtype).tiny

    h = coords_flat_mm[:, 0] / res_h
    d = coords_flat_mm[:, 1] / res_d
    w = coords_flat_mm[:, 2] / res_w
    h0 = torch.floor(h).long()
    d0 = torch.floor(d).long()
    w0 = torch.floor(w).long()
    fh = h - h0.to(values_flat.dtype)
    fd = d - d0.to(values_flat.dtype)
    fw = w - w0.to(values_flat.dtype)

    offsets = torch.arange(
        -half_window,
        half_window + 1,
        device=values_flat.device,
        dtype=values_flat.dtype,
    )
    inv_2_sigma_sq = 1.0 / (2.0 * float(sigma_voxels) * float(sigma_voxels))
    # per-axis weight matrices, shape (N, 2W+1)
    weight_h = torch.exp(-((offsets.view(1, -1) - fh.unsqueeze(-1)) ** 2) * inv_2_sigma_sq)
    weight_d = torch.exp(-((offsets.view(1, -1) - fd.unsqueeze(-1)) ** 2) * inv_2_sigma_sq)
    weight_w = torch.exp(-((offsets.view(1, -1) - fw.unsqueeze(-1)) ** 2) * inv_2_sigma_sq)
    weight_h = weight_h / weight_h.sum(dim=1, keepdim=True).clamp_min(eps)
    weight_d = weight_d / weight_d.sum(dim=1, keepdim=True).clamp_min(eps)
    weight_w = weight_w / weight_w.sum(dim=1, keepdim=True).clamp_min(eps)

    int_offsets = torch.arange(
        -half_window, half_window + 1, device=values_flat.device, dtype=torch.long
    )
    nW = int_offsets.numel()
    for ih in range(nW):
        for id_ in range(nW):
            for iw in range(nW):
                hi = h0 + int_offsets[ih]
                di = d0 + int_offsets[id_]
                wi = w0 + int_offsets[iw]
                inside = (
                    (hi >= 0) & (hi <= max_h)
                    & (di >= 0) & (di <= max_d)
                    & (wi >= 0) & (wi <= max_w)
                )
                hi_c = hi.clamp(0, max_h)
                di_c = di.clamp(0, max_d)
                wi_c = wi.clamp(0, max_w)
                lin = hi_c * depth_stride + di_c * width_stride + wi_c
                w_combined = weight_h[:, ih] * weight_d[:, id_] * weight_w[:, iw]
                contrib = torch.where(
                    inside,
                    values_flat * w_combined,
                    torch.zeros_like(values_flat),
                )
                dose_row_flat.scatter_add_(0, lin, contrib)


def gaussian_scatter_add(
    base_flat: torch.Tensor,
    coords_mm: torch.Tensor,
    values: torch.Tensor,
    res_h: float,
    res_d: float,
    res_w: float,
    size_h: int,
    size_d: int,
    size_w: int,
    sigma_voxels: float,
    half_window: int,
) -> torch.Tensor:
    """Functional Gaussian-footprint scatter that returns a new tensor.

    Mirrors the signature of ``ops.trilinear_scatter_add`` so it can be
    swapped in by the splat dispatchers. Forward-only (no autograd kernel
    is registered)."""
    out = base_flat.clone()
    gaussian_scatter_add_inplace(
        out,
        coords_mm,
        values,
        (float(res_h), float(res_d), float(res_w)),
        (int(size_h), int(size_d), int(size_w)),
        float(sigma_voxels),
        int(half_window),
    )
    return out


if triton is not None:
    @triton.jit
    def _gaussian_scatter_add_hw1_forward_kernel(
        out_ptr,
        coords_ptr,
        values_ptr,
        n_elements,
        res_h,
        res_d,
        res_w,
        max_h,
        max_d,
        max_w,
        depth_stride,
        width_stride,
        inv_2_sigma_sq,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        coord_base = offsets * 3
        h_mm = tl.load(coords_ptr + coord_base + 0, mask=mask, other=0.0)
        d_mm = tl.load(coords_ptr + coord_base + 1, mask=mask, other=0.0)
        w_mm = tl.load(coords_ptr + coord_base + 2, mask=mask, other=0.0)
        vals = tl.load(values_ptr + offsets, mask=mask, other=0.0)

        h = h_mm / res_h
        d = d_mm / res_d
        w = w_mm / res_w
        h0 = tl.floor(h).to(tl.int32)
        d0 = tl.floor(d).to(tl.int32)
        w0 = tl.floor(w).to(tl.int32)
        fh = h - h0.to(tl.float32)
        fd = d - d0.to(tl.float32)
        fw = w - w0.to(tl.float32)

        wh_m = tl.exp(-((-1.0 - fh) * (-1.0 - fh)) * inv_2_sigma_sq)
        wh_0 = tl.exp(-((0.0 - fh) * (0.0 - fh)) * inv_2_sigma_sq)
        wh_p = tl.exp(-((1.0 - fh) * (1.0 - fh)) * inv_2_sigma_sq)
        wd_m = tl.exp(-((-1.0 - fd) * (-1.0 - fd)) * inv_2_sigma_sq)
        wd_0 = tl.exp(-((0.0 - fd) * (0.0 - fd)) * inv_2_sigma_sq)
        wd_p = tl.exp(-((1.0 - fd) * (1.0 - fd)) * inv_2_sigma_sq)
        ww_m = tl.exp(-((-1.0 - fw) * (-1.0 - fw)) * inv_2_sigma_sq)
        ww_0 = tl.exp(-((0.0 - fw) * (0.0 - fw)) * inv_2_sigma_sq)
        ww_p = tl.exp(-((1.0 - fw) * (1.0 - fw)) * inv_2_sigma_sq)
        sh = wh_m + wh_0 + wh_p
        sd = wd_m + wd_0 + wd_p
        sw = ww_m + ww_0 + ww_p
        wh_m = wh_m / sh
        wh_0 = wh_0 / sh
        wh_p = wh_p / sh
        wd_m = wd_m / sd
        wd_0 = wd_0 / sd
        wd_p = wd_p / sd
        ww_m = ww_m / sw
        ww_0 = ww_0 / sw
        ww_p = ww_p / sw

        for ih in tl.static_range(0, 3):
            hi = h0 + (ih - 1)
            wh = tl.where(ih == 0, wh_m, tl.where(ih == 1, wh_0, wh_p))
            for id_ in tl.static_range(0, 3):
                di = d0 + (id_ - 1)
                wd = tl.where(id_ == 0, wd_m, tl.where(id_ == 1, wd_0, wd_p))
                for iw in tl.static_range(0, 3):
                    wi = w0 + (iw - 1)
                    ww = tl.where(iw == 0, ww_m, tl.where(iw == 1, ww_0, ww_p))
                    inside = (
                        mask
                        & (hi >= 0) & (hi <= max_h)
                        & (di >= 0) & (di <= max_d)
                        & (wi >= 0) & (wi <= max_w)
                    )
                    hi_c = tl.maximum(0, tl.minimum(hi, max_h))
                    di_c = tl.maximum(0, tl.minimum(di, max_d))
                    wi_c = tl.maximum(0, tl.minimum(wi, max_w))
                    lin = hi_c * depth_stride + di_c * width_stride + wi_c
                    tl.atomic_add(out_ptr + lin, vals * wh * wd * ww, mask=inside)

    @triton.jit
    def _gaussian_scatter_add_hw1_backward_values_kernel(
        grad_out_ptr,
        coords_ptr,
        grad_values_ptr,
        n_elements,
        res_h,
        res_d,
        res_w,
        max_h,
        max_d,
        max_w,
        depth_stride,
        width_stride,
        inv_2_sigma_sq,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        coord_base = offsets * 3
        h_mm = tl.load(coords_ptr + coord_base + 0, mask=mask, other=0.0)
        d_mm = tl.load(coords_ptr + coord_base + 1, mask=mask, other=0.0)
        w_mm = tl.load(coords_ptr + coord_base + 2, mask=mask, other=0.0)

        h = h_mm / res_h
        d = d_mm / res_d
        w = w_mm / res_w
        h0 = tl.floor(h).to(tl.int32)
        d0 = tl.floor(d).to(tl.int32)
        w0 = tl.floor(w).to(tl.int32)
        fh = h - h0.to(tl.float32)
        fd = d - d0.to(tl.float32)
        fw = w - w0.to(tl.float32)

        wh_m = tl.exp(-((-1.0 - fh) * (-1.0 - fh)) * inv_2_sigma_sq)
        wh_0 = tl.exp(-((0.0 - fh) * (0.0 - fh)) * inv_2_sigma_sq)
        wh_p = tl.exp(-((1.0 - fh) * (1.0 - fh)) * inv_2_sigma_sq)
        wd_m = tl.exp(-((-1.0 - fd) * (-1.0 - fd)) * inv_2_sigma_sq)
        wd_0 = tl.exp(-((0.0 - fd) * (0.0 - fd)) * inv_2_sigma_sq)
        wd_p = tl.exp(-((1.0 - fd) * (1.0 - fd)) * inv_2_sigma_sq)
        ww_m = tl.exp(-((-1.0 - fw) * (-1.0 - fw)) * inv_2_sigma_sq)
        ww_0 = tl.exp(-((0.0 - fw) * (0.0 - fw)) * inv_2_sigma_sq)
        ww_p = tl.exp(-((1.0 - fw) * (1.0 - fw)) * inv_2_sigma_sq)
        sh = wh_m + wh_0 + wh_p
        sd = wd_m + wd_0 + wd_p
        sw = ww_m + ww_0 + ww_p
        wh_m = wh_m / sh
        wh_0 = wh_0 / sh
        wh_p = wh_p / sh
        wd_m = wd_m / sd
        wd_0 = wd_0 / sd
        wd_p = wd_p / sd
        ww_m = ww_m / sw
        ww_0 = ww_0 / sw
        ww_p = ww_p / sw

        grad_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for ih in tl.static_range(0, 3):
            hi = h0 + (ih - 1)
            wh = tl.where(ih == 0, wh_m, tl.where(ih == 1, wh_0, wh_p))
            for id_ in tl.static_range(0, 3):
                di = d0 + (id_ - 1)
                wd = tl.where(id_ == 0, wd_m, tl.where(id_ == 1, wd_0, wd_p))
                for iw in tl.static_range(0, 3):
                    wi = w0 + (iw - 1)
                    ww = tl.where(iw == 0, ww_m, tl.where(iw == 1, ww_0, ww_p))
                    inside = (
                        mask
                        & (hi >= 0) & (hi <= max_h)
                        & (di >= 0) & (di <= max_d)
                        & (wi >= 0) & (wi <= max_w)
                    )
                    hi_c = tl.maximum(0, tl.minimum(hi, max_h))
                    di_c = tl.maximum(0, tl.minimum(di, max_d))
                    wi_c = tl.maximum(0, tl.minimum(wi, max_w))
                    lin = hi_c * depth_stride + di_c * width_stride + wi_c
                    gout = tl.load(grad_out_ptr + lin, mask=inside, other=0.0)
                    grad_val += gout * wh * wd * ww

        tl.store(grad_values_ptr + offsets, grad_val, mask=mask)


class _GaussianScatterAddHW1Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        base_flat: torch.Tensor,
        coords_mm: torch.Tensor,
        values: torch.Tensor,
        res_h: float,
        res_d: float,
        res_w: float,
        size_h: int,
        size_d: int,
        size_w: int,
        sigma_voxels: float,
    ) -> torch.Tensor:
        coords_flat, values_flat = _scatter_inputs(coords_mm, values)
        out = base_flat.contiguous().clone()
        coords_flat = coords_flat.contiguous()
        values_flat = values_flat.contiguous()
        n_elements = values_flat.numel()
        if n_elements:
            block_size = 256
            grid = (triton.cdiv(n_elements, block_size),)
            inv_2_sigma_sq = 1.0 / (2.0 * float(sigma_voxels) * float(sigma_voxels))
            _gaussian_scatter_add_hw1_forward_kernel[grid](
                out,
                coords_flat,
                values_flat,
                n_elements,
                float(res_h),
                float(res_d),
                float(res_w),
                int(size_h) - 1,
                int(size_d) - 1,
                int(size_w) - 1,
                int(size_d) * int(size_w),
                int(size_w),
                float(inv_2_sigma_sq),
                BLOCK_SIZE=block_size,
            )
        ctx.save_for_backward(coords_flat)
        ctx.values_shape = values.shape
        ctx.res_h = float(res_h)
        ctx.res_d = float(res_d)
        ctx.res_w = float(res_w)
        ctx.size_h = int(size_h)
        ctx.size_d = int(size_d)
        ctx.size_w = int(size_w)
        ctx.sigma_voxels = float(sigma_voxels)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (coords_flat,) = ctx.saved_tensors
        grad_base = grad_output
        grad_values_flat = torch.empty(
            coords_flat.shape[0],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        n_elements = coords_flat.shape[0]
        if n_elements:
            block_size = 256
            grid = (triton.cdiv(n_elements, block_size),)
            inv_2_sigma_sq = 1.0 / (2.0 * ctx.sigma_voxels * ctx.sigma_voxels)
            _gaussian_scatter_add_hw1_backward_values_kernel[grid](
                grad_output.reshape(-1).contiguous(),
                coords_flat,
                grad_values_flat,
                n_elements,
                ctx.res_h,
                ctx.res_d,
                ctx.res_w,
                ctx.size_h - 1,
                ctx.size_d - 1,
                ctx.size_w - 1,
                ctx.size_d * ctx.size_w,
                ctx.size_w,
                float(inv_2_sigma_sq),
                BLOCK_SIZE=block_size,
            )
        grad_values = grad_values_flat.reshape(ctx.values_shape)
        return grad_base, None, grad_values, None, None, None, None, None, None, None


def _can_use_triton_gaussian(
    base_flat: torch.Tensor,
    coords_mm: torch.Tensor,
    values: torch.Tensor,
    half_window: int,
) -> bool:
    return (
        triton is not None
        and half_window == 1
        and base_flat.device.type == "cuda"
        and base_flat.dtype == torch.float32
        and coords_mm.dtype == torch.float32
        and values.dtype == torch.float32
    )


def gaussian_scatter_add_autograd(
    base_flat: torch.Tensor,
    coords_mm: torch.Tensor,
    values: torch.Tensor,
    res_h: float,
    res_d: float,
    res_w: float,
    size_h: int,
    size_d: int,
    size_w: int,
    sigma_voxels: float,
    half_window: int,
) -> torch.Tensor:
    """Functional Gaussian scatter with Triton value-gradient fast path.

    The Triton path currently targets the training setting: CUDA float32 with
    ``half_window == 1``. Other cases fall back to the reference PyTorch
    implementation, which keeps full autograd semantics.
    """
    if _can_use_triton_gaussian(base_flat, coords_mm, values, int(half_window)):
        return _GaussianScatterAddHW1Function.apply(
            base_flat,
            coords_mm,
            values,
            float(res_h),
            float(res_d),
            float(res_w),
            int(size_h),
            int(size_d),
            int(size_w),
            float(sigma_voxels),
        )
    return gaussian_scatter_add(
        base_flat,
        coords_mm,
        values,
        res_h,
        res_d,
        res_w,
        size_h,
        size_d,
        size_w,
        sigma_voxels,
        half_window,
    )
