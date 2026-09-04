from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def local_gamma_pass_rate(
    ref: np.ndarray | torch.Tensor,
    pred: np.ndarray | torch.Tensor,
    voxel_size_mm: tuple[float, float, float],
    dose_threshold_pct: float,
    dist_threshold_mm: float,
    prescription_gy: float,
    lower_cutoff_pct: float = 10.0,
    max_gamma: float = 2.0,
    interp_fraction: int = 5,
    device: torch.device | None = None,
) -> float:
    if isinstance(ref, np.ndarray):
        ref = torch.from_numpy(ref)
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)

    ref = ref.to(device=device, dtype=torch.float32)
    pred = pred.to(device=device, dtype=torch.float32)

    mask = ref > lower_cutoff_pct / 100.0 * prescription_gy
    if not mask.any():
        return 100.0

    dz, dy, dx = float(voxel_size_mm[0]), float(voxel_size_mm[1]), float(voxel_size_mm[2])
    dd_crit = dose_threshold_pct / 100.0
    max_gamma2 = max_gamma ** 2
    Z, Y, X = ref.shape
    ref_local = (dd_crit * ref).clamp(min=1e-9)

    # Build candidate offsets on a uniform grid, filtered to the search sphere
    step = dist_threshold_mm / interp_fraction
    r = max_gamma * dist_threshold_mm
    t = torch.arange(-r, r + step * 0.5, step)
    oz, oy, ox = torch.meshgrid(t, t, t, indexing="ij")
    d2 = (oz / dist_threshold_mm).square() + (oy / dist_threshold_mm).square() + (ox / dist_threshold_mm).square()
    valid = d2 <= max_gamma2

    d2_v = d2[valid].to(ref.device)
    norm_off = torch.stack([
        ox[valid] / ((X - 1) * 0.5 * dx),
        oy[valid] / ((Y - 1) * 0.5 * dy),
        oz[valid] / ((Z - 1) * 0.5 * dz),
    ], dim=1).to(ref.device)  # (N_off, 3) in grid_sample (x,y,z) convention

    # Process nearest offsets first so we can exit early once all voxels pass
    order = d2_v.argsort()
    d2_v = d2_v[order]
    norm_off = norm_off[order]

    # Base identity sampling grid: (1, Z, Y, X, 3)
    gz = torch.linspace(-1.0, 1.0, Z, device=ref.device)
    gy = torch.linspace(-1.0, 1.0, Y, device=ref.device)
    gx = torch.linspace(-1.0, 1.0, X, device=ref.device)
    gzz, gyy, gxx = torch.meshgrid(gz, gy, gx, indexing="ij")
    base_grid = torch.stack([gxx, gyy, gzz], dim=-1).unsqueeze(0)  # (1, Z, Y, X, 3)

    pred_4d = pred[None, None]  # (1, 1, Z, Y, X)
    min_gamma2 = ref.new_full((Z, Y, X), float("inf"))

    # Chunk size: keep grids tensor (K, Z, Y, X, 3) under ~1 GB
    vox_bytes = Z * Y * X * 4
    chunk_size = max(1, int(1e9 // (vox_bytes * 3)))

    N_off = d2_v.shape[0]
    for i in range(0, N_off, chunk_size):
        off = norm_off[i : i + chunk_size]   # (K, 3)
        d2s = d2_v[i : i + chunk_size]       # (K,)
        K = off.shape[0]

        grids = base_grid + off[:, None, None, None, :]  # (K, Z, Y, X, 3)
        sampled = F.grid_sample(
            pred_4d.expand(K, 1, Z, Y, X),
            grids,
            mode="bilinear",
            align_corners=True,
            padding_mode="zeros",
        )[:, 0]  # (K, Z, Y, X)

        gamma2 = ((sampled - ref) / ref_local).square() + d2s[:, None, None, None]
        torch.minimum(min_gamma2, gamma2.min(dim=0).values, out=min_gamma2)

        if (min_gamma2[mask] <= 1.0).all():
            break

    return 100.0 * (min_gamma2[mask] <= 1.0).float().mean().item()
