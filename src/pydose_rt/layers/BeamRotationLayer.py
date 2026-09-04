"""
CPPRotationLayer module for performing beam-wise 2D rotation of dose volumes using grid sampling.

This module provides the CPPRotationLayer class, which rotates accumulated dose volumes for each gantry angle
using PyTorch's grid sampling. The layer is designed to handle 5D tensors representing dose distributions across batches,
gantry angles, depth, height, and width.

Typical usage example::
    layer = BeamRotationLayer(machine_config, device, dtype, gantry_angles)
    rotated_dose = layer(accumulated_dose)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pydose_rt.data import MachineConfig
from pydose_rt.geometry.rotations import build_rotation_grids

class BeamRotationLayer(nn.Module):
    """
    PyTorch module for performing beam-wise 2D rotation of dose volumes using grid sampling.

    Attributes:        
        machine_config (MachineConfig): Stores configuration parameters.
        device (torch.device): Device on which computations are performed.
        dtype (type): Data type for tensors.
        verbose (bool): Verbosity flag.
        rot_angles_rad (torch.Tensor): Tensor of gantry angles in radians.
    """
    def __init__(self,
                 machine_config: MachineConfig,
                 ct_array_shape: tuple[float, float, float],
                 iso_center: tuple[float, float, float],                
                 resolution: tuple[float, float, float],
                 gantry_angles: list[float] | torch.Tensor = None,
                 depth_origin: str = "iso_depth",
                 device: torch.device | str | None = None,
                 dtype: torch.dtype = torch.float32,
                 verbose: bool = False,
                ) -> 'BeamRotationLayer':
        """
        Initializes the BeamRotationLayer.
        Args:            
            machine_config (MachineConfig): Configuration parameters for the layer.            
            ct_array_shape: (H, D, W) - shape of CT array in voxels
            iso_center: (X, Y, Z) - isocenter in physical coordinates (mm)
            resolution: (rx, ry, rz) - voxel spacing in mm
            device (torch.device): Device for computation (CPU or CUDA).
            dtype (type): Data type for tensors.
            gantry_angles (list[float] | torch.Tensor): Gantry angles in radians.
            verbose (bool, optional): If True, enables verbose output for debugging. Defaults to False.
        """
        super().__init__()
        # Handle device default
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.dtype = dtype
        self.machine_config = machine_config
        self.ct_array_shape = ct_array_shape        
        self.iso_center = iso_center
        self.resolution = resolution
        self.depth_origin = depth_origin
        self.verbose = verbose

        self.rot_angles_rad = gantry_angles.to(dtype=self.dtype, device=self.device)        
        rot_depth_origin = "entry_patient_to_bev" if depth_origin == "entry" else depth_origin
        self.rot_grid = build_rotation_grids(
            (1, self.rot_angles_rad.shape[0], self.ct_array_shape[1], self.ct_array_shape[0], self.ct_array_shape[2]),
            self.rot_angles_rad,
            self.device,
            self.dtype,
            iso_center=iso_center,
            resolution=resolution,
            depth_origin=rot_depth_origin,
        )

    def forward(self, accumulated_dose: torch.Tensor) -> torch.Tensor:
        """
        Rotates all [B, G, D, H, W] dose accumulated_dose for all gantry angles in parallel (fully vectorized).
        Args:
            accumulated_dose (torch.Tensor): [B, G, D, H, W]
            rot_angles_rad (torch.Tensor): tensor of G angles in radians
            center (tuple, optional): (cy, cx) voxel coordinates in [D, W] plane. If None, uses center of volume.
        Returns:
            torch.Tensor: Rotated [B, G, H, D, W]
        """

        B, G, D, H, W = accumulated_dose.shape
        dose_hdw = accumulated_dose.permute(0, 1, 3, 2, 4).contiguous()  # [B, G, H, D, W]
        rotated_by_beam = []
        for g in range(G):
            dose_g = dose_hdw[:, g].reshape(B * H, 1, D, W)
            grid_g = self.rot_grid[0, g, 0].unsqueeze(0).expand(B * H, -1, -1, -1)
            rotated_g = F.grid_sample(
                dose_g,
                grid_g,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            rotated_by_beam.append(rotated_g.reshape(B, H, D, W))
        return torch.stack(rotated_by_beam, dim=1)
