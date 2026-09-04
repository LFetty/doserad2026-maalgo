import torch
import torch.nn as nn
import torch.nn.functional as F


class BevProjector(nn.Module):
    """Project patient-frame volumes into BEV with per-sample gantry angles.

    Unlike ``RadiologicalDepthLayer`` which precomputes rotation grids at init
    for fixed gantry angles, this class builds grids on the fly so each sample
    in a batch can have its own angle and isocenter.
    """

    def __init__(self, resolution: tuple[float, float, float]):
        super().__init__()
        self.resolution = resolution

    def _build_sampling_grid(
        self,
        angles_rad: torch.Tensor,
        iso_centers_mm: torch.Tensor,
        D: int,
        W: int,
    ) -> torch.Tensor:
        """Build a physical-mm sampling grid for ``grid_sample``.

        ``angles_rad`` follows the public API convention: positive gantry
        angles are inverted here for patient→BEV sampling.  Passing
        ``-gantry`` therefore builds the opposite BEV→patient grid.
        """
        a = -angles_rad
        cos_a = torch.cos(a).view(-1, 1, 1)
        sin_a = torch.sin(a).view(-1, 1, 1)

        B = angles_rad.shape[0]
        _res_h, res_d, res_w = self.resolution
        center_y = iso_centers_mm[:, 1] / res_d
        center_x = iso_centers_mm[:, 2] / res_w

        y_idx = torch.arange(D, device=angles_rad.device, dtype=angles_rad.dtype)
        x_idx = torch.arange(W, device=angles_rad.device, dtype=angles_rad.dtype)
        y_grid, x_grid = torch.meshgrid(y_idx, x_idx, indexing="ij")

        y_offsets_mm = (y_grid.unsqueeze(0) - center_y.view(B, 1, 1)) * res_d
        x_offsets_mm = (x_grid.unsqueeze(0) - center_x.view(B, 1, 1)) * res_w

        sample_x = center_x.view(B, 1, 1) + (x_offsets_mm * cos_a + y_offsets_mm * sin_a) / res_w
        sample_y = center_y.view(B, 1, 1) + (-x_offsets_mm * sin_a + y_offsets_mm * cos_a) / res_d

        grid_x = (2.0 * (sample_x + 0.5) / W) - 1.0
        grid_y = (2.0 * (sample_y + 0.5) / D) - 1.0
        return torch.stack((grid_x, grid_y), dim=-1)

    def sample_bev(
        self,
        volume: torch.Tensor,
        gantry_angles_rad: torch.Tensor,
        iso_centers_mm: torch.Tensor,
        mode: str = "bilinear",
    ) -> torch.Tensor:
        """Inverse-rotate a patient-frame volume into each sample's BEV.

        Args:
            volume: ``[B, H, D, W]`` patient-frame volume.
            gantry_angles_rad: ``[B]`` per-sample gantry angles in radians.
            iso_centers_mm: ``[B, 3]`` per-sample isocenters ``(h, d, w)`` in mm.
            mode: Interpolation mode — ``'bilinear'`` for continuous values
                (SPR), ``'nearest'`` for integer labels (material ID).

        Returns:
            ``[B, D, H, W]`` volume in BEV layout.
        """
        B, H, D, W = volume.shape
        grid = self._build_sampling_grid(gantry_angles_rad, iso_centers_mm, D, W)
        grid = grid.unsqueeze(1).expand(B, H, D, W, 2).reshape(B * H, D, W, 2)

        vol_flat = volume.reshape(B * H, 1, D, W)
        sampled = F.grid_sample(
            vol_flat, grid, mode=mode, padding_mode="zeros", align_corners=False,
        )
        sampled = sampled.reshape(B, H, D, W)
        return sampled.permute(0, 2, 1, 3).contiguous()

    def rotate_to_patient(
        self,
        bev_volume: torch.Tensor,
        gantry_angles_rad: torch.Tensor,
        iso_centers_mm: torch.Tensor,
    ) -> torch.Tensor:
        """Forward-rotate a BEV volume back to the patient frame.

        Args:
            bev_volume: ``[B, D, H, W]`` volume in BEV layout.
            gantry_angles_rad: ``[B]`` per-sample gantry angles in radians.
            iso_centers_mm: ``[B, 3]`` per-sample isocenters ``(h, d, w)`` in mm.

        Returns:
            ``[B, H, D, W]`` volume in patient layout.
        """
        B, D, H, W = bev_volume.shape
        grid = self._build_sampling_grid(-gantry_angles_rad, iso_centers_mm, D, W)
        grid = grid.unsqueeze(1).expand(B, H, D, W, 2).reshape(B * H, D, W, 2)

        vol_perm = bev_volume.permute(0, 2, 1, 3).contiguous()
        vol_flat = vol_perm.reshape(B * H, 1, D, W)
        sampled = F.grid_sample(
            vol_flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False,
        )
        return sampled.reshape(B, H, D, W)

    @torch.no_grad()
    def forward(
        self,
        spr_volume: torch.Tensor,
        gantry_angles_rad: torch.Tensor,
        iso_centers_mm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-voxel SPR and WEQ in BEV coordinates.

        Args:
            spr_volume: ``[B, H, D, W]`` patient-frame SPR (stopping power ratio).
            gantry_angles_rad: ``[B]`` per-sample gantry angles in radians.
            iso_centers_mm: ``[B, 3]`` per-sample isocenters ``(h, d, w)`` in mm.

        Returns:
            ``(spr_bev, weq_bev)`` each ``[B, D, H, W]``.
            WEQ uses the sparse convention: ``boundary_end - 0.5 * segment_weq``
            (depth to voxel center).
        """
        spr_bev = self.sample_bev(
            spr_volume, gantry_angles_rad, iso_centers_mm, mode="bilinear",
        )

        ry = self.resolution[1]
        segment_weq = spr_bev.clamp_min(0.0) * ry
        boundary_end = torch.cumsum(segment_weq, dim=1)
        weq_bev = boundary_end - 0.5 * segment_weq

        return spr_bev, weq_bev
