from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional at runtime
    triton = None
    tl = None


def _reshape_scatter_inputs(
    coords_mm: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Size, torch.Size]:
    coords_shape = coords_mm.shape
    values_shape = values.shape
    if coords_mm.dim() < 2 or coords_mm.shape[-1] != 3:
        raise ValueError(f"coords_mm must end in a xyz axis, got {tuple(coords_mm.shape)}")
    if values.shape != coords_mm.shape[:-1]:
        raise ValueError(
            "values must match coords_mm without the trailing xyz axis, got "
            f"{tuple(values.shape)} vs {tuple(coords_mm.shape[:-1])}"
        )
    return coords_mm.reshape(-1, 3), values.reshape(-1), coords_shape, values_shape


def _gather_trilinear_corners_flat(
    grid_flat: torch.Tensor,
    coords_flat_mm: torch.Tensor,
    res_h: float,
    res_d: float,
    res_w: float,
    size_h: int,
    size_d: int,
    size_w: int,
) -> dict[str, torch.Tensor]:
    h = coords_flat_mm[:, 0] / res_h
    d = coords_flat_mm[:, 1] / res_d
    w = coords_flat_mm[:, 2] / res_w

    h0 = torch.floor(h).long()
    d0 = torch.floor(d).long()
    w0 = torch.floor(w).long()
    h1 = h0 + 1
    d1 = d0 + 1
    w1 = w0 + 1

    fh = h - h0.to(grid_flat.dtype)
    fd = d - d0.to(grid_flat.dtype)
    fw = w - w0.to(grid_flat.dtype)

    max_h = size_h - 1
    max_d = size_d - 1
    max_w = size_w - 1
    depth_stride = size_d * size_w
    width_stride = size_w

    def gather(hi: torch.Tensor, di: torch.Tensor, wi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inside = (
            (hi >= 0) & (hi <= max_h)
            & (di >= 0) & (di <= max_d)
            & (wi >= 0) & (wi <= max_w)
        )
        hi_c = hi.clamp(0, max_h)
        di_c = di.clamp(0, max_d)
        wi_c = wi.clamp(0, max_w)
        linear_idx = hi_c * depth_stride + di_c * width_stride + wi_c
        values = torch.where(inside, grid_flat[linear_idx], torch.zeros_like(h, dtype=grid_flat.dtype))
        return values, inside

    c000, inside000 = gather(h0, d0, w0)
    c001, inside001 = gather(h0, d0, w1)
    c010, inside010 = gather(h0, d1, w0)
    c011, inside011 = gather(h0, d1, w1)
    c100, inside100 = gather(h1, d0, w0)
    c101, inside101 = gather(h1, d0, w1)
    c110, inside110 = gather(h1, d1, w0)
    c111, inside111 = gather(h1, d1, w1)

    return {
        "h0": h0,
        "d0": d0,
        "w0": w0,
        "h1": h1,
        "d1": d1,
        "w1": w1,
        "fh": fh,
        "fd": fd,
        "fw": fw,
        "c000": c000,
        "c001": c001,
        "c010": c010,
        "c011": c011,
        "c100": c100,
        "c101": c101,
        "c110": c110,
        "c111": c111,
        "inside000": inside000,
        "inside001": inside001,
        "inside010": inside010,
        "inside011": inside011,
        "inside100": inside100,
        "inside101": inside101,
        "inside110": inside110,
        "inside111": inside111,
    }


def _trilinear_sample_with_coord_grads_flat(
    grid_flat: torch.Tensor,
    coords_flat_mm: torch.Tensor,
    res_h: float,
    res_d: float,
    res_w: float,
    size_h: int,
    size_d: int,
    size_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    corners = _gather_trilinear_corners_flat(
        grid_flat,
        coords_flat_mm,
        res_h=res_h,
        res_d=res_d,
        res_w=res_w,
        size_h=size_h,
        size_d=size_d,
        size_w=size_w,
    )
    fh = corners["fh"]
    fd = corners["fd"]
    fw = corners["fw"]

    c000 = corners["c000"]
    c001 = corners["c001"]
    c010 = corners["c010"]
    c011 = corners["c011"]
    c100 = corners["c100"]
    c101 = corners["c101"]
    c110 = corners["c110"]
    c111 = corners["c111"]

    c00 = c000 * (1.0 - fw) + c001 * fw
    c01 = c010 * (1.0 - fw) + c011 * fw
    c10 = c100 * (1.0 - fw) + c101 * fw
    c11 = c110 * (1.0 - fw) + c111 * fw
    c0 = c00 * (1.0 - fd) + c01 * fd
    c1 = c10 * (1.0 - fd) + c11 * fd
    sampled = c0 * (1.0 - fh) + c1 * fh

    ds_dfh = c1 - c0
    ds_dfd = (c01 - c00) * (1.0 - fh) + (c11 - c10) * fh
    ds_dfw = (
        ((c001 - c000) * (1.0 - fd) + (c011 - c010) * fd) * (1.0 - fh)
        + ((c101 - c100) * (1.0 - fd) + (c111 - c110) * fd) * fh
    )
    grad_coords = torch.stack((ds_dfh / res_h, ds_dfd / res_d, ds_dfw / res_w), dim=-1)
    return sampled, grad_coords


def _trilinear_scatter_add_torch_impl(
    base_flat: torch.Tensor,
    coords_mm: torch.Tensor,
    values: torch.Tensor,
    res_h: float,
    res_d: float,
    res_w: float,
    size_h: int,
    size_d: int,
    size_w: int,
) -> torch.Tensor:
    coords_flat_mm, values_flat, _, _ = _reshape_scatter_inputs(coords_mm, values)
    if coords_flat_mm.numel() == 0 or values_flat.numel() == 0:
        return base_flat.clone()

    out = base_flat.clone()
    h = coords_flat_mm[:, 0] / res_h
    d = coords_flat_mm[:, 1] / res_d
    w = coords_flat_mm[:, 2] / res_w

    h0 = torch.floor(h).long()
    d0 = torch.floor(d).long()
    w0 = torch.floor(w).long()
    h1 = h0 + 1
    d1 = d0 + 1
    w1 = w0 + 1

    fh = h - h0.to(values_flat.dtype)
    fd = d - d0.to(values_flat.dtype)
    fw = w - w0.to(values_flat.dtype)

    max_h = size_h - 1
    max_d = size_d - 1
    max_w = size_w - 1
    depth_stride = size_d * size_w
    width_stride = size_w

    one_minus_fh = 1.0 - fh
    one_minus_fd = 1.0 - fd
    one_minus_fw = 1.0 - fw

    hi_all = torch.stack((h0, h0, h0, h0, h1, h1, h1, h1), dim=0)
    di_all = torch.stack((d0, d0, d1, d1, d0, d0, d1, d1), dim=0)
    wi_all = torch.stack((w0, w1, w0, w1, w0, w1, w0, w1), dim=0)
    weights_all = torch.stack(
        (
            one_minus_fh * one_minus_fd * one_minus_fw,
            one_minus_fh * one_minus_fd * fw,
            one_minus_fh * fd * one_minus_fw,
            one_minus_fh * fd * fw,
            fh * one_minus_fd * one_minus_fw,
            fh * one_minus_fd * fw,
            fh * fd * one_minus_fw,
            fh * fd * fw,
        ),
        dim=0,
    )
    inside_all = (
        (hi_all >= 0) & (hi_all <= max_h)
        & (di_all >= 0) & (di_all <= max_d)
        & (wi_all >= 0) & (wi_all <= max_w)
    )
    hi_c = hi_all.clamp(0, max_h)
    di_c = di_all.clamp(0, max_d)
    wi_c = wi_all.clamp(0, max_w)
    linear_idx_all = (hi_c * depth_stride + di_c * width_stride + wi_c).reshape(-1)
    contrib_all = torch.where(
        inside_all,
        values_flat.unsqueeze(0) * weights_all,
        torch.zeros_like(weights_all),
    ).reshape(-1)
    out.scatter_add_(0, linear_idx_all, contrib_all)
    return out


if triton is not None:
    @triton.jit
    def _trilinear_scatter_add_kernel(
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
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        coord_base = offsets * 3
        h_mm = tl.load(coords_ptr + coord_base + 0, mask=mask, other=0.0)
        d_mm = tl.load(coords_ptr + coord_base + 1, mask=mask, other=0.0)
        w_mm = tl.load(coords_ptr + coord_base + 2, mask=mask, other=0.0)
        values = tl.load(values_ptr + offsets, mask=mask, other=0.0)

        h = h_mm / res_h
        d = d_mm / res_d
        w = w_mm / res_w

        h0 = tl.floor(h).to(tl.int32)
        d0 = tl.floor(d).to(tl.int32)
        w0 = tl.floor(w).to(tl.int32)
        h1 = h0 + 1
        d1 = d0 + 1
        w1 = w0 + 1

        fh = h - h0.to(tl.float32)
        fd = d - d0.to(tl.float32)
        fw = w - w0.to(tl.float32)

        one_minus_fh = 1.0 - fh
        one_minus_fd = 1.0 - fd
        one_minus_fw = 1.0 - fw

        hi = h0
        di = d0
        wi = w0
        weight = one_minus_fh * one_minus_fd * one_minus_fw
        inside = mask & (hi >= 0) & (hi <= max_h) & (di >= 0) & (di <= max_d) & (wi >= 0) & (wi <= max_w)
        hi_c = tl.maximum(0, tl.minimum(hi, max_h))
        di_c = tl.maximum(0, tl.minimum(di, max_d))
        wi_c = tl.maximum(0, tl.minimum(wi, max_w))
        linear_idx = hi_c * depth_stride + di_c * width_stride + wi_c
        tl.atomic_add(out_ptr + linear_idx, values * weight, mask=inside)

        hi = h0
        di = d0
        wi = w1
        weight = one_minus_fh * one_minus_fd * fw
        inside = mask & (hi >= 0) & (hi <= max_h) & (di >= 0) & (di <= max_d) & (wi >= 0) & (wi <= max_w)
        hi_c = tl.maximum(0, tl.minimum(hi, max_h))
        di_c = tl.maximum(0, tl.minimum(di, max_d))
        wi_c = tl.maximum(0, tl.minimum(wi, max_w))
        linear_idx = hi_c * depth_stride + di_c * width_stride + wi_c
        tl.atomic_add(out_ptr + linear_idx, values * weight, mask=inside)

        hi = h0
        di = d1
        wi = w0
        weight = one_minus_fh * fd * one_minus_fw
        inside = mask & (hi >= 0) & (hi <= max_h) & (di >= 0) & (di <= max_d) & (wi >= 0) & (wi <= max_w)
        hi_c = tl.maximum(0, tl.minimum(hi, max_h))
        di_c = tl.maximum(0, tl.minimum(di, max_d))
        wi_c = tl.maximum(0, tl.minimum(wi, max_w))
        linear_idx = hi_c * depth_stride + di_c * width_stride + wi_c
        tl.atomic_add(out_ptr + linear_idx, values * weight, mask=inside)

        hi = h0
        di = d1
        wi = w1
        weight = one_minus_fh * fd * fw
        inside = mask & (hi >= 0) & (hi <= max_h) & (di >= 0) & (di <= max_d) & (wi >= 0) & (wi <= max_w)
        hi_c = tl.maximum(0, tl.minimum(hi, max_h))
        di_c = tl.maximum(0, tl.minimum(di, max_d))
        wi_c = tl.maximum(0, tl.minimum(wi, max_w))
        linear_idx = hi_c * depth_stride + di_c * width_stride + wi_c
        tl.atomic_add(out_ptr + linear_idx, values * weight, mask=inside)

        hi = h1
        di = d0
        wi = w0
        weight = fh * one_minus_fd * one_minus_fw
        inside = mask & (hi >= 0) & (hi <= max_h) & (di >= 0) & (di <= max_d) & (wi >= 0) & (wi <= max_w)
        hi_c = tl.maximum(0, tl.minimum(hi, max_h))
        di_c = tl.maximum(0, tl.minimum(di, max_d))
        wi_c = tl.maximum(0, tl.minimum(wi, max_w))
        linear_idx = hi_c * depth_stride + di_c * width_stride + wi_c
        tl.atomic_add(out_ptr + linear_idx, values * weight, mask=inside)

        hi = h1
        di = d0
        wi = w1
        weight = fh * one_minus_fd * fw
        inside = mask & (hi >= 0) & (hi <= max_h) & (di >= 0) & (di <= max_d) & (wi >= 0) & (wi <= max_w)
        hi_c = tl.maximum(0, tl.minimum(hi, max_h))
        di_c = tl.maximum(0, tl.minimum(di, max_d))
        wi_c = tl.maximum(0, tl.minimum(wi, max_w))
        linear_idx = hi_c * depth_stride + di_c * width_stride + wi_c
        tl.atomic_add(out_ptr + linear_idx, values * weight, mask=inside)

        hi = h1
        di = d1
        wi = w0
        weight = fh * fd * one_minus_fw
        inside = mask & (hi >= 0) & (hi <= max_h) & (di >= 0) & (di <= max_d) & (wi >= 0) & (wi <= max_w)
        hi_c = tl.maximum(0, tl.minimum(hi, max_h))
        di_c = tl.maximum(0, tl.minimum(di, max_d))
        wi_c = tl.maximum(0, tl.minimum(wi, max_w))
        linear_idx = hi_c * depth_stride + di_c * width_stride + wi_c
        tl.atomic_add(out_ptr + linear_idx, values * weight, mask=inside)

        hi = h1
        di = d1
        wi = w1
        weight = fh * fd * fw
        inside = mask & (hi >= 0) & (hi <= max_h) & (di >= 0) & (di <= max_d) & (wi >= 0) & (wi <= max_w)
        hi_c = tl.maximum(0, tl.minimum(hi, max_h))
        di_c = tl.maximum(0, tl.minimum(di, max_d))
        wi_c = tl.maximum(0, tl.minimum(wi, max_w))
        linear_idx = hi_c * depth_stride + di_c * width_stride + wi_c
        tl.atomic_add(out_ptr + linear_idx, values * weight, mask=inside)


if triton is not None:
    @triton.jit
    def _trilinear_scatter_rays_inplace_kernel(
        out_ptr,
        source_ptr,
        ray_dirs_ptr,
        t_samples_ptr,
        values_ptr,
        n_rays,
        n_steps,
        res_h,
        res_d,
        res_w,
        max_h,
        max_d,
        max_w,
        depth_stride,
        width_stride,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (n_rays * n_steps)

        ray_idx = offsets // n_steps
        step_idx = offsets % n_steps

        src_h = tl.load(source_ptr + 0)
        src_d = tl.load(source_ptr + 1)
        src_w = tl.load(source_ptr + 2)

        dir_base = ray_idx * 3
        dir_h = tl.load(ray_dirs_ptr + dir_base + 0, mask=mask, other=0.0)
        dir_d = tl.load(ray_dirs_ptr + dir_base + 1, mask=mask, other=0.0)
        dir_w = tl.load(ray_dirs_ptr + dir_base + 2, mask=mask, other=0.0)

        sample_idx = ray_idx * n_steps + step_idx
        t = tl.load(t_samples_ptr + sample_idx, mask=mask, other=0.0)
        values = tl.load(values_ptr + sample_idx, mask=mask, other=0.0)

        h = (src_h + dir_h * t) / res_h
        d = (src_d + dir_d * t) / res_d
        w = (src_w + dir_w * t) / res_w

        h0 = tl.floor(h).to(tl.int32)
        d0 = tl.floor(d).to(tl.int32)
        w0 = tl.floor(w).to(tl.int32)
        h1 = h0 + 1
        d1 = d0 + 1
        w1 = w0 + 1

        fh = h - h0.to(tl.float32)
        fd = d - d0.to(tl.float32)
        fw = w - w0.to(tl.float32)

        one_minus_fh = 1.0 - fh
        one_minus_fd = 1.0 - fd
        one_minus_fw = 1.0 - fw

        hi = h0
        di = d0
        wi = w0
        weight = one_minus_fh * one_minus_fd * one_minus_fw
        inside = mask & (hi >= 0) & (hi <= max_h) & (di >= 0) & (di <= max_d) & (wi >= 0) & (wi <= max_w)
        hi_c = tl.maximum(0, tl.minimum(hi, max_h))
        di_c = tl.maximum(0, tl.minimum(di, max_d))
        wi_c = tl.maximum(0, tl.minimum(wi, max_w))
        linear_idx = hi_c * depth_stride + di_c * width_stride + wi_c
        tl.atomic_add(out_ptr + linear_idx, values * weight, mask=inside)

        hi = h0
        di = d0
        wi = w1
        weight = one_minus_fh * one_minus_fd * fw
        inside = mask & (hi >= 0) & (hi <= max_h) & (di >= 0) & (di <= max_d) & (wi >= 0) & (wi <= max_w)
        hi_c = tl.maximum(0, tl.minimum(hi, max_h))
        di_c = tl.maximum(0, tl.minimum(di, max_d))
        wi_c = tl.maximum(0, tl.minimum(wi, max_w))
        linear_idx = hi_c * depth_stride + di_c * width_stride + wi_c
        tl.atomic_add(out_ptr + linear_idx, values * weight, mask=inside)

        hi = h0
        di = d1
        wi = w0
        weight = one_minus_fh * fd * one_minus_fw
        inside = mask & (hi >= 0) & (hi <= max_h) & (di >= 0) & (di <= max_d) & (wi >= 0) & (wi <= max_w)
        hi_c = tl.maximum(0, tl.minimum(hi, max_h))
        di_c = tl.maximum(0, tl.minimum(di, max_d))
        wi_c = tl.maximum(0, tl.minimum(wi, max_w))
        linear_idx = hi_c * depth_stride + di_c * width_stride + wi_c
        tl.atomic_add(out_ptr + linear_idx, values * weight, mask=inside)

        hi = h0
        di = d1
        wi = w1
        weight = one_minus_fh * fd * fw
        inside = mask & (hi >= 0) & (hi <= max_h) & (di >= 0) & (di <= max_d) & (wi >= 0) & (wi <= max_w)
        hi_c = tl.maximum(0, tl.minimum(hi, max_h))
        di_c = tl.maximum(0, tl.minimum(di, max_d))
        wi_c = tl.maximum(0, tl.minimum(wi, max_w))
        linear_idx = hi_c * depth_stride + di_c * width_stride + wi_c
        tl.atomic_add(out_ptr + linear_idx, values * weight, mask=inside)

        hi = h1
        di = d0
        wi = w0
        weight = fh * one_minus_fd * one_minus_fw
        inside = mask & (hi >= 0) & (hi <= max_h) & (di >= 0) & (di <= max_d) & (wi >= 0) & (wi <= max_w)
        hi_c = tl.maximum(0, tl.minimum(hi, max_h))
        di_c = tl.maximum(0, tl.minimum(di, max_d))
        wi_c = tl.maximum(0, tl.minimum(wi, max_w))
        linear_idx = hi_c * depth_stride + di_c * width_stride + wi_c
        tl.atomic_add(out_ptr + linear_idx, values * weight, mask=inside)

        hi = h1
        di = d0
        wi = w1
        weight = fh * one_minus_fd * fw
        inside = mask & (hi >= 0) & (hi <= max_h) & (di >= 0) & (di <= max_d) & (wi >= 0) & (wi <= max_w)
        hi_c = tl.maximum(0, tl.minimum(hi, max_h))
        di_c = tl.maximum(0, tl.minimum(di, max_d))
        wi_c = tl.maximum(0, tl.minimum(wi, max_w))
        linear_idx = hi_c * depth_stride + di_c * width_stride + wi_c
        tl.atomic_add(out_ptr + linear_idx, values * weight, mask=inside)

        hi = h1
        di = d1
        wi = w0
        weight = fh * fd * one_minus_fw
        inside = mask & (hi >= 0) & (hi <= max_h) & (di >= 0) & (di <= max_d) & (wi >= 0) & (wi <= max_w)
        hi_c = tl.maximum(0, tl.minimum(hi, max_h))
        di_c = tl.maximum(0, tl.minimum(di, max_d))
        wi_c = tl.maximum(0, tl.minimum(wi, max_w))
        linear_idx = hi_c * depth_stride + di_c * width_stride + wi_c
        tl.atomic_add(out_ptr + linear_idx, values * weight, mask=inside)

        hi = h1
        di = d1
        wi = w1
        weight = fh * fd * fw
        inside = mask & (hi >= 0) & (hi <= max_h) & (di >= 0) & (di <= max_d) & (wi >= 0) & (wi <= max_w)
        hi_c = tl.maximum(0, tl.minimum(hi, max_h))
        di_c = tl.maximum(0, tl.minimum(di, max_d))
        wi_c = tl.maximum(0, tl.minimum(wi, max_w))
        linear_idx = hi_c * depth_stride + di_c * width_stride + wi_c
        tl.atomic_add(out_ptr + linear_idx, values * weight, mask=inside)


def trilinear_scatter_rays_inplace(
    dose_row_flat: torch.Tensor,
    source: torch.Tensor,
    ray_dirs: torch.Tensor,
    t_samples: torch.Tensor,
    values: torch.Tensor,
    resolution: tuple[float, float, float],
    ct_array_shape: tuple[int, int, int],
) -> None:
    if ray_dirs.numel() == 0 or t_samples.numel() == 0 or values.numel() == 0:
        return
    if (
        triton is None
        or dose_row_flat.device.type != "cuda"
        or dose_row_flat.dtype != torch.float32
        or source.dtype != torch.float32
        or ray_dirs.dtype != torch.float32
        or t_samples.dtype != torch.float32
        or values.dtype != torch.float32
    ):
        coords = source.unsqueeze(0).unsqueeze(0) + ray_dirs.unsqueeze(1) * t_samples.unsqueeze(-1)
        updated = _trilinear_scatter_add_torch_impl(
            dose_row_flat,
            coords,
            values,
            res_h=resolution[0],
            res_d=resolution[1],
            res_w=resolution[2],
            size_h=ct_array_shape[0],
            size_d=ct_array_shape[1],
            size_w=ct_array_shape[2],
        )
        dose_row_flat.copy_(updated)
        return

    out = dose_row_flat.contiguous()
    src = source.contiguous()
    dirs = ray_dirs.contiguous()
    ts = t_samples.contiguous()
    vals = values.contiguous()
    n_rays, n_steps = ts.shape
    block_size = 256
    grid = (triton.cdiv(n_rays * n_steps, block_size),)
    _trilinear_scatter_rays_inplace_kernel[grid](
        out,
        src,
        dirs,
        ts,
        vals,
        n_rays,
        n_steps,
        float(resolution[0]),
        float(resolution[1]),
        float(resolution[2]),
        ct_array_shape[0] - 1,
        ct_array_shape[1] - 1,
        ct_array_shape[2] - 1,
        ct_array_shape[1] * ct_array_shape[2],
        ct_array_shape[2],
        BLOCK_SIZE=block_size,
    )


def trilinear_scatter_add_inplace(
    dose_row_flat: torch.Tensor,
    coords_mm: torch.Tensor,
    values: torch.Tensor,
    resolution: tuple[float, float, float],
    ct_array_shape: tuple[int, int, int],
) -> None:
    coords_flat_mm, values_flat, _, _ = _reshape_scatter_inputs(coords_mm, values)
    if coords_flat_mm.numel() == 0 or values_flat.numel() == 0:
        return

    if (
        triton is None
        or dose_row_flat.device.type != "cuda"
        or dose_row_flat.dtype != torch.float32
        or coords_flat_mm.dtype != torch.float32
        or values_flat.dtype != torch.float32
    ):
        updated = _trilinear_scatter_add_torch_impl(
            dose_row_flat,
            coords_mm,
            values,
            res_h=resolution[0],
            res_d=resolution[1],
            res_w=resolution[2],
            size_h=ct_array_shape[0],
            size_d=ct_array_shape[1],
            size_w=ct_array_shape[2],
        )
        dose_row_flat.copy_(updated)
        return

    if dose_row_flat.is_contiguous():
        out = dose_row_flat
    else:
        out = dose_row_flat.contiguous()
    coords_flat_mm = coords_flat_mm.contiguous()
    values_flat = values_flat.contiguous()
    n_elements = values_flat.numel()
    block_size = 256
    grid = (triton.cdiv(n_elements, block_size),)
    _trilinear_scatter_add_kernel[grid](
        out,
        coords_flat_mm,
        values_flat,
        n_elements,
        float(resolution[0]),
        float(resolution[1]),
        float(resolution[2]),
        ct_array_shape[0] - 1,
        ct_array_shape[1] - 1,
        ct_array_shape[2] - 1,
        ct_array_shape[1] * ct_array_shape[2],
        ct_array_shape[2],
        BLOCK_SIZE=block_size,
    )
    if out is not dose_row_flat:
        dose_row_flat.copy_(out)


def _trilinear_scatter_add_cuda_impl(
    base_flat: torch.Tensor,
    coords_mm: torch.Tensor,
    values: torch.Tensor,
    res_h: float,
    res_d: float,
    res_w: float,
    size_h: int,
    size_d: int,
    size_w: int,
) -> torch.Tensor:
    coords_flat_mm, values_flat, _, _ = _reshape_scatter_inputs(coords_mm, values)
    if coords_flat_mm.numel() == 0 or values_flat.numel() == 0:
        return base_flat.clone()
    if triton is None or base_flat.dtype != torch.float32 or coords_flat_mm.dtype != torch.float32 or values_flat.dtype != torch.float32:
        return _trilinear_scatter_add_torch_impl(
            base_flat,
            coords_mm,
            values,
            res_h=res_h,
            res_d=res_d,
            res_w=res_w,
            size_h=size_h,
            size_d=size_d,
            size_w=size_w,
        )

    out = base_flat.contiguous().clone()
    coords_flat_mm = coords_flat_mm.contiguous()
    values_flat = values_flat.contiguous()
    n_elements = values_flat.numel()
    block_size = 256
    grid = (triton.cdiv(n_elements, block_size),)
    _trilinear_scatter_add_kernel[grid](
        out,
        coords_flat_mm,
        values_flat,
        n_elements,
        float(res_h),
        float(res_d),
        float(res_w),
        size_h - 1,
        size_d - 1,
        size_w - 1,
        size_d * size_w,
        size_w,
        BLOCK_SIZE=block_size,
    )
    return out


@torch.library.custom_op("pydose_rt::trilinear_scatter_add", mutates_args=())
def trilinear_scatter_add(
    base_flat: torch.Tensor,
    coords_mm: torch.Tensor,
    values: torch.Tensor,
    res_h: float,
    res_d: float,
    res_w: float,
    size_h: int,
    size_d: int,
    size_w: int,
) -> torch.Tensor:
    return _trilinear_scatter_add_torch_impl(
        base_flat,
        coords_mm,
        values,
        res_h=res_h,
        res_d=res_d,
        res_w=res_w,
        size_h=size_h,
        size_d=size_d,
        size_w=size_w,
    )


@trilinear_scatter_add.register_kernel("cuda")
def _trilinear_scatter_add_cuda_kernel(
    base_flat: torch.Tensor,
    coords_mm: torch.Tensor,
    values: torch.Tensor,
    res_h: float,
    res_d: float,
    res_w: float,
    size_h: int,
    size_d: int,
    size_w: int,
) -> torch.Tensor:
    return _trilinear_scatter_add_cuda_impl(
        base_flat,
        coords_mm,
        values,
        res_h=res_h,
        res_d=res_d,
        res_w=res_w,
        size_h=size_h,
        size_d=size_d,
        size_w=size_w,
    )


def _trilinear_scatter_add_setup_context(ctx, inputs, output) -> None:
    base_flat, coords_mm, values, res_h, res_d, res_w, size_h, size_d, size_w = inputs
    ctx.save_for_backward(coords_mm, values)
    ctx.res_h = float(res_h)
    ctx.res_d = float(res_d)
    ctx.res_w = float(res_w)
    ctx.size_h = int(size_h)
    ctx.size_d = int(size_d)
    ctx.size_w = int(size_w)


def _trilinear_scatter_add_backward(ctx, grad_output: torch.Tensor):
    coords_mm, values = ctx.saved_tensors
    coords_flat_mm, values_flat, coords_shape, values_shape = _reshape_scatter_inputs(coords_mm, values)
    grad_output_flat = grad_output.reshape(-1)
    grad_values_flat, grad_coords_basis = _trilinear_sample_with_coord_grads_flat(
        grad_output_flat,
        coords_flat_mm,
        res_h=ctx.res_h,
        res_d=ctx.res_d,
        res_w=ctx.res_w,
        size_h=ctx.size_h,
        size_d=ctx.size_d,
        size_w=ctx.size_w,
    )
    grad_coords = (grad_coords_basis * values_flat.unsqueeze(-1)).reshape(coords_shape)
    grad_values = grad_values_flat.reshape(values_shape)
    return grad_output, grad_coords, grad_values, None, None, None, None, None, None


trilinear_scatter_add.register_autograd(
    _trilinear_scatter_add_backward,
    setup_context=_trilinear_scatter_add_setup_context,
)
