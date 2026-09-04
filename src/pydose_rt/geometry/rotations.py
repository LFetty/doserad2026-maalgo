import torch
import math
import torch.nn.functional as F


def _normalize_iso_centers_tensor(
    iso_center,
    num_beams: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if iso_center is None:
        return None
    iso_tensor = torch.as_tensor(iso_center, device=device, dtype=dtype)
    if iso_tensor.shape == (3,):
        return iso_tensor.unsqueeze(0).expand(num_beams, -1)
    if iso_tensor.shape == (num_beams, 3):
        return iso_tensor
    raise ValueError(
        f"iso_center must be shape (3,) or ({num_beams}, 3), got {tuple(iso_tensor.shape)}"
    )


def _beam_axis_lat_units(angles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    axis_units = torch.stack(
        (
            torch.zeros_like(cos_angles),
            cos_angles,
            -sin_angles,
        ),
        dim=1,
    )
    lat_units = torch.stack(
        (
            torch.zeros_like(cos_angles),
            sin_angles,
            cos_angles,
        ),
        dim=1,
    )
    return axis_units, lat_units


def _central_ray_entry_mm(
    input_shape: tuple[int, int, int],
    angles: torch.Tensor,
    iso_centers: torch.Tensor,
    resolution: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h, d, w = input_shape
    rh, rd, rw = resolution
    axis_units, lat_units = _beam_axis_lat_units(angles)
    bounds_min = torch.zeros((3,), device=angles.device, dtype=angles.dtype)
    bounds_max = torch.tensor(
        ((h - 1) * rh, (d - 1) * rd, (w - 1) * rw),
        device=angles.device,
        dtype=angles.dtype,
    )
    source = iso_centers - axis_units * torch.as_tensor(1_000_000.0, device=angles.device, dtype=angles.dtype)
    eps = torch.finfo(angles.dtype).eps
    safe_axis = torch.where(axis_units.abs() < eps, torch.full_like(axis_units, eps), axis_units)
    inv_axis = 1.0 / safe_axis
    t0 = (bounds_min - source) * inv_axis
    t1 = (bounds_max - source) * inv_axis
    t_entry = torch.maximum(torch.minimum(t0, t1).amax(dim=1), torch.zeros_like(angles))
    entry = source + axis_units * t_entry[:, None]
    return entry, axis_units, lat_units


def get_radiological_depth_indices(
    input_shape,
    angles_rad,
    dtype,
    iso_center=None,
    resolution=None,
    depth_origin: str = "iso_depth",
):
    """
    Generate sampling coordinates for radiological depth calculation using ray tracing.

    For each angle, creates a ray through the isocenter (or volume center if not specified)
    with uniform voxel spacing in the rotated coordinate frame. Returns exactly D points per ray.

    Args:
        input_shape: (H, D, W) - shape of CT volume in voxels
        angles_rad: list/tensor of rotation angles in radians
        dtype: torch dtype for output
        iso_center: (X, Y, Z) - isocenter in physical coordinates (mm), where X=height, Y=depth, Z=width
        resolution: (rx, ry, rz) - voxel spacing in mm, where rx=res_height, ry=res_depth, rz=res_width

    Returns:
        indices: [1, G, D, 3] - floating point coordinates (x, y, z) for sampling
                 where x∈[0,W-1], y∈[0,D-1], z∈[0,H-1]
                 Each ray has exactly D points
    """
    H, D, W = input_shape

    # Calculate center in voxel coordinates
    device = angles_rad.device if isinstance(angles_rad, torch.Tensor) else torch.device("cpu")
    angles_tensor = torch.as_tensor(angles_rad, device=device, dtype=dtype)
    num_beams = int(angles_tensor.numel())

    if iso_center is not None and resolution is not None:
        iso_tensor = _normalize_iso_centers_tensor(iso_center, num_beams, device, dtype)
        rx, ry, rz = resolution
        center_z = iso_tensor[:, 0] / rx
        center_y = iso_tensor[:, 1] / ry
        center_x = iso_tensor[:, 2] / rz
    else:
        iso_tensor = None
        center_x = torch.full((num_beams,), W / 2.0, device=device, dtype=dtype)
        center_y = torch.full((num_beams,), D / 2.0, device=device, dtype=dtype)
        center_z = torch.full((num_beams,), H / 2.0, device=device, dtype=dtype)

    # Create a line of D points along the Y axis (depth direction)
    # This is the reference line at angle=0
    y_line = torch.linspace(0, D - 1, D, device=device, dtype=dtype)

    if depth_origin == "entry":
        if iso_tensor is None or resolution is None:
            raise ValueError("depth_origin='entry' requires iso_center and resolution")
        entry, axis_units, _lat_units = _central_ray_entry_mm(input_shape, angles_tensor, iso_tensor, resolution)
        step_mm = torch.as_tensor(resolution[1], device=device, dtype=dtype)
        points_mm = entry[:, None, :] + axis_units[:, None, :] * (y_line[None, :, None] * step_mm)
        coords = torch.stack(
            (
                points_mm[..., 2] / resolution[2],
                points_mm[..., 1] / resolution[1],
                points_mm[..., 0] / resolution[0],
            ),
            dim=-1,
        )
        return coords.unsqueeze(0)
    if depth_origin != "iso_depth":
        raise ValueError(f"depth_origin must be 'iso_depth' or 'entry', got {depth_origin!r}")

    if resolution is None:
        # Backwards-compatible voxel-space fallback when no physical spacing
        # is available.
        indices_list = []
        for beam_idx, angle in enumerate(angles_tensor):
            theta = float(angle)
            x_line = torch.full_like(y_line, center_x[beam_idx])

            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)

            x_shifted = x_line - center_x[beam_idx]
            y_shifted = y_line - center_y[beam_idx]

            x_rotated = x_shifted * cos_theta - y_shifted * sin_theta
            y_rotated = x_shifted * sin_theta + y_shifted * cos_theta

            x_coords = x_rotated + center_x[beam_idx]
            y_coords = y_rotated + center_y[beam_idx]
            z_coords = torch.full_like(x_coords, center_z[beam_idx])
            indices_list.append(torch.stack([x_coords, y_coords, z_coords], dim=-1))
        stacked_indices = torch.stack(indices_list, dim=0)  # [G, D, 3]
    else:
        rx, ry, rz = resolution
        depth_offsets_mm = (y_line.unsqueeze(0) - center_y.unsqueeze(1)) * ry
        cos_theta = torch.cos(angles_tensor).unsqueeze(1)
        sin_theta = torch.sin(angles_tensor).unsqueeze(1)
        y_coords = center_y.unsqueeze(1) + (depth_offsets_mm * cos_theta) / ry
        x_coords = center_x.unsqueeze(1) - (depth_offsets_mm * sin_theta) / rz
        z_coords = center_z.view(num_beams, 1).expand(num_beams, D)
        stacked_indices = torch.stack([x_coords, y_coords, z_coords], dim=-1)

    return stacked_indices.unsqueeze(0)  # [1, G, D, 3]

def rotate_2d_images(images, angles_rad, device, dtype):
    """
    Rotate 2D images by given angles using affine transformation.
    Args:
        images: [B*G, H, W] - batch of 2D images
        angles_rad: [G] - rotation angles in radians (one per control point)
        device: torch device
        dtype: torch dtype
    Returns:
        rotated_images: [B*G, H, W] - rotated images
    """
    BG, H, W = images.shape

    # Convert angles to tensor if needed
    if not isinstance(angles_rad, torch.Tensor):
        angles_rad = torch.tensor(angles_rad, device=device, dtype=dtype)
    else:
        angles_rad = angles_rad.to(device=device, dtype=dtype)

    # Flatten angles to [1, G] if needed
    if angles_rad.dim() == 2:
        angles_rad = angles_rad.view(-1)  # [G]
    G = angles_rad.shape[0]
    B = BG // G

    # Expand angles for batch dimension: [B*G]
    angles_expanded = angles_rad.unsqueeze(0).repeat(B, 1).view(BG)  # [B*G]

    cos_a = torch.cos(angles_expanded)
    sin_a = torch.sin(angles_expanded)

    # Create affine transformation matrices for rotation
    # Note: negative angle for counter-clockwise rotation in image space
    mats = torch.zeros((BG, 2, 3), device=device, dtype=dtype)
    mats[:, 0, 0] = cos_a
    mats[:, 0, 1] = sin_a
    mats[:, 1, 0] = -sin_a
    mats[:, 1, 1] = cos_a

    # Generate rotation grids
    grid = F.affine_grid(mats, size=(BG, 1, H, W), align_corners=False)  # [BG, H, W, 2]

    # Rotate images
    images_4d = images.unsqueeze(1)  # [BG, 1, H, W]
    rotated = F.grid_sample(images_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    rotated = rotated.squeeze(1)  # [BG, H, W]

    return rotated

def build_rotation_grids(
    input_shape,
    angles_rad,
    device,
    dtype,
    iso_center=None,
    resolution=None,
    depth_origin: str = "iso_depth",
):
    """
    Build rotation grids for rotating D×W images by given angles around a specified point.

    Args:
        input_shape: (B, G, D, H, W)
        angles_rad: Tensor of G rotation angles in radians
        device: torch device
        dtype: torch dtype
        iso_center: (X, Y, Z) - isocenter in physical coordinates (mm), where X=height, Y=depth, Z=width
        resolution: (rx, ry, rz) - voxel spacing in mm, where rx=res_height, ry=res_depth, rz=res_width

    Returns:
        grid2d: [B*G*H, D, W, 2] sampling grid for grid_sample
    """
    B, G, D, H, W = input_shape
    a = angles_rad.to(device=device, dtype=dtype)

    if depth_origin in {"entry", "entry_patient_to_bev"}:
        if iso_center is None or resolution is None:
            raise ValueError("depth_origin='entry' requires iso_center and resolution")
        iso_tensor = _normalize_iso_centers_tensor(iso_center, G, device, dtype)
        entry, axis_units, lat_units = _central_ray_entry_mm((H, D, W), a, iso_tensor, resolution)
        _rh, rd, rw = resolution

        d_coords = torch.arange(D, device=device, dtype=dtype) * rd
        w_coords = torch.arange(W, device=device, dtype=dtype) * rw
        d_grid, w_grid = torch.meshgrid(d_coords, w_coords, indexing="ij")
        points_dw = torch.stack((d_grid, w_grid), dim=-1).unsqueeze(0)  # [1, D, W, 2]

        entry_dw = entry[:, [1, 2]].view(G, 1, 1, 2)
        axis_dw = axis_units[:, [1, 2]].view(G, 1, 1, 2)
        lat_dw = lat_units[:, [1, 2]].view(G, 1, 1, 2)
        delta = points_dw - entry_dw
        depth_mm = (delta * axis_dw).sum(dim=-1)
        lateral_mm = (delta * lat_dw).sum(dim=-1)

        d_bev = depth_mm / rd
        w_bev = lateral_mm / rw + iso_tensor[:, 2].view(G, 1, 1) / rw
        grid_x = (2.0 * (w_bev + 0.5) / W) - 1.0
        grid_y = (2.0 * (d_bev + 0.5) / D) - 1.0
        return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).unsqueeze(2)
    if depth_origin == "entry_bev_to_patient":
        if iso_center is None or resolution is None:
            raise ValueError("depth_origin='entry_bev_to_patient' requires iso_center and resolution")
        iso_tensor = _normalize_iso_centers_tensor(iso_center, G, device, dtype)
        _rh, rd, rw = resolution
        cos_a = torch.cos(a)
        sin_a = torch.sin(a)
        axis_units = torch.stack(
            (
                torch.zeros_like(cos_a),
                cos_a,
                sin_a,
            ),
            dim=1,
        )
        lat_units = torch.stack(
            (
                torch.zeros_like(cos_a),
                -sin_a,
                cos_a,
            ),
            dim=1,
        )

        bounds_min = torch.zeros((3,), device=device, dtype=dtype)
        bounds_max = torch.tensor(
            ((H - 1) * resolution[0], (D - 1) * rd, (W - 1) * rw),
            device=device,
            dtype=dtype,
        )
        source = iso_tensor - axis_units * torch.as_tensor(1_000_000.0, device=device, dtype=dtype)
        eps = torch.finfo(dtype).eps
        safe_axis = torch.where(axis_units.abs() < eps, torch.full_like(axis_units, eps), axis_units)
        inv_axis = 1.0 / safe_axis
        t0 = (bounds_min - source) * inv_axis
        t1 = (bounds_max - source) * inv_axis
        t_entry = torch.maximum(torch.minimum(t0, t1).amax(dim=1), torch.zeros_like(a))
        entry = source + axis_units * t_entry[:, None]

        d_coords = torch.arange(D, device=device, dtype=dtype) * rd
        w_offsets = (torch.arange(W, device=device, dtype=dtype) - (iso_tensor[:, 2:3] / rw)) * rw
        d_grid = d_coords.view(1, D, 1)
        w_grid = w_offsets.view(G, 1, W)
        points = (
            entry[:, None, None, :]
            + axis_units[:, None, None, :] * d_grid[..., None]
            + lat_units[:, None, None, :] * w_grid[..., None]
        )
        sample_y = points[..., 1] / rd
        sample_x = points[..., 2] / rw
        grid_x = (2.0 * (sample_x + 0.5) / W) - 1.0
        grid_y = (2.0 * (sample_y + 0.5) / D) - 1.0
        return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).unsqueeze(2)
    if depth_origin != "iso_depth":
        raise ValueError(
            "depth_origin must be 'iso_depth', 'entry', 'entry_patient_to_bev', "
            f"or 'entry_bev_to_patient', got {depth_origin!r}"
        )

    if iso_center is None or resolution is None:
        cos_a = torch.cos(a)
        sin_a = torch.sin(a)
        mats = torch.zeros((G, 2, 3), device=device, dtype=dtype)
        mats[:, 0, 0] = cos_a
        mats[:, 0, 1] = sin_a
        mats[:, 1, 0] = -sin_a
        mats[:, 1, 1] = cos_a

        grid2d = F.affine_grid(mats, size=(G, 1, D, W), align_corners=False)
        return grid2d.unsqueeze(1).unsqueeze(0)

    iso_tensor = _normalize_iso_centers_tensor(iso_center, G, device, dtype)
    _rh, rd, rw = resolution
    center_y = iso_tensor[:, 1] / rd
    center_x = iso_tensor[:, 2] / rw

    y_idx = torch.arange(D, device=device, dtype=dtype)
    x_idx = torch.arange(W, device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(y_idx, x_idx, indexing="ij")

    y_offsets_mm = (y_grid.unsqueeze(0) - center_y.view(G, 1, 1)) * rd
    x_offsets_mm = (x_grid.unsqueeze(0) - center_x.view(G, 1, 1)) * rw

    cos_a = torch.cos(a).view(G, 1, 1)
    sin_a = torch.sin(a).view(G, 1, 1)
    sample_x = center_x.view(G, 1, 1) + (x_offsets_mm * cos_a + y_offsets_mm * sin_a) / rw
    sample_y = center_y.view(G, 1, 1) + (-x_offsets_mm * sin_a + y_offsets_mm * cos_a) / rd

    grid_x = (2.0 * (sample_x + 0.5) / W) - 1.0
    grid_y = (2.0 * (sample_y + 0.5) / D) - 1.0
    grid2d = torch.stack((grid_x, grid_y), dim=-1)
    return grid2d.unsqueeze(0).unsqueeze(2)
