"""
Main class for radiotherapy dose calculation using pencil beam convolution and beam-wise rotation.

This class orchestrates the pipeline for dose calculation, including preprocessing, fluence modeling,
kernel generation, convolution, and geometric rotation of dose volumes. It supports batched inputs and
multiple beams, and can optionally perform upsampling and debugging visualizations.
"""
from contextlib import nullcontext

import torch
from torch import nn

from pydose_rt.layers.BeamValidationLayer import BeamValidationLayer
from pydose_rt.layers.FluenceMapLayer import FluenceMapLayer
from pydose_rt.layers.FluenceVolumeLayer import FluenceVolumeLayer
from pydose_rt.layers.RadiologicalDepthLayer import RadiologicalDepthLayer
from pydose_rt.layers.PencilBeamKernelLayer import PencilBeamKernelLayer
from pydose_rt.layers.BeamWiseConvolutionalLayer import BeamWiseConvolutionalLayer
from pydose_rt.layers.BeamRotationLayer import BeamRotationLayer
from pydose_rt.data import MachineConfig, Beam, BeamSequence
from pydose_rt.geometry.rotations import rotate_2d_images


class DoseEngine(nn.Module):
    """
    Implements the full dose calculation pipeline for radiotherapy.

    Usage:
        engine = DoseEngine(machine_config)
        dose = engine.forward(leaf_positions, mus, jaw_positions, density_image)

    Or with BeamSequence:
        dose = engine.forward_beam_sequence(beam_seq, density_image)

    Attributes:
        machine_config (MachineConfig): Machine physics parameters.
        device (torch.device): PyTorch device for computation.
    """
    machine_config: MachineConfig | None = None
    dose_grid_shape: tuple[int, int, int] | None = None
    dose_grid_spacing: tuple[float, float, float] | None = None
    number_of_beams: int | None = None
    layers_initialized: bool = False
    gantry_angles: torch.Tensor | None = None
    collimator_angles: torch.Tensor | None = None
    field_size: tuple[float, float] | None = None
    SID: float | None = None
    iso_center: tuple[float, float, float] | None = None

    def __init__(
        self,
        machine_config: MachineConfig,
        kernel_size: int,
        dose_grid_spacing: tuple[float, float, float],
        dose_grid_shape: tuple[int, int, int],
        beam_template: BeamSequence | Beam | None = None,
        adjust_values: bool = False, # Move to nn.Module
        device: torch.device | str | None = None,
        dtype: torch.dtype = None,
        verbose: bool = False,
    ) -> "DoseEngine":
        """
        Initializes the DoseEngine pipeline.

        Args:
            machine_config: Machine physics and MLC specifications.
            kernel_size: Size of the pencil beam dose kernel.
            dose_grid_spacing: Voxel spacing in mm (depth, height, width).
            dose_grid_shape: Shape of the output grid tensor (depth, height, width) in pixels.
            beam_input: Beam or BeamSequence defining the treatment geometry.
            adjust_values: Whether to validate and adjust parameter values (default: False).
            device: PyTorch device for computation.
            dtype: Data type for tensors.
            verbose: Enable verbose output (default: False).
        """
        super().__init__()
        self.kernel_size = kernel_size

        # Handle device default
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self._adjust_values = adjust_values

        self.machine_config = machine_config
        self.dose_grid_spacing = dose_grid_spacing
        self.dose_grid_shape = dose_grid_shape
        self._initialize_layers(beam_template)

    def _set_device_dtype(self, device, dtype) -> None:
        if self.dtype is None:
            self.dtype = dtype
        if self.device is None:
            self.device = device

    def _initialize_layers(self, new_beam_data: BeamSequence | Beam, overwrite: bool = False) -> None:
        if new_beam_data is None:
            return

        if self.dtype is None:
            self.dtype = new_beam_data.dtype
        if self.device is None:
            self.device = new_beam_data.device

        initialize_beam_validation_layer = overwrite or not hasattr(self, "beam_validation_layer")
        initialize_fluence_map_layer = overwrite or not hasattr(self, "fluence_map_layer")
        initialize_fluence_volume_layer = overwrite or not hasattr(self, "fluence_volume_layer")
        initialize_beam_wise_conv_layer = overwrite or not hasattr(self, "beam_wise_conv_layer")
        initialize_pencil_beam_kernel_layer = overwrite or not hasattr(self, "pencil_beam_kernel_layer")
        initialize_rad_depth_layer = overwrite or not hasattr(self, "rad_depth_layer")
        initialize_rotation_layer = overwrite or not hasattr(self, "rotation_layer")

        if isinstance(new_beam_data, Beam):
            number_of_beams = 1
            gantry_angles = torch.tensor(
                [new_beam_data.gantry_angle],
                device=self.device,
                dtype=self.dtype,
            )
            collimator_angles = torch.tensor(
                [new_beam_data.collimator_angle],
                device=self.device,
                dtype=self.dtype,
            )
        elif isinstance(new_beam_data, BeamSequence):
            number_of_beams = len(new_beam_data)
            gantry_angles = new_beam_data.gantry_angles.to(device=self.device, dtype=self.dtype)
            collimator_angles = new_beam_data.collimator_angles.to(device=self.device, dtype=self.dtype)
        else:
            raise TypeError(f"Unsupported beam data type: {type(new_beam_data)!r}")

        angles_changed = (
            self.number_of_beams is None
            or self.number_of_beams != number_of_beams
            or self.gantry_angles is None
            or self.gantry_angles.shape != gantry_angles.shape
            or not torch.allclose(self.gantry_angles.to(device=self.device, dtype=self.dtype), gantry_angles)
            or self.collimator_angles is None
            or self.collimator_angles.shape != collimator_angles.shape
            or not torch.allclose(self.collimator_angles.to(device=self.device, dtype=self.dtype), collimator_angles)
        )
        field_changed = self.field_size is None or self.field_size != new_beam_data.field_size
        sid_changed = self.SID is None or self.SID != new_beam_data.sid
        iso_changed = self.iso_center is None or self.iso_center != new_beam_data.iso_center

        if angles_changed:
            initialize_rad_depth_layer = True
            initialize_rotation_layer = True
        if field_changed:
            initialize_beam_validation_layer = True
            initialize_fluence_map_layer = True
            initialize_fluence_volume_layer = True
        if sid_changed:
            initialize_fluence_volume_layer = True
        if iso_changed:
            initialize_fluence_volume_layer = True
            initialize_rad_depth_layer = True
            initialize_rotation_layer = True

        self.number_of_beams = number_of_beams
        self.gantry_angles = gantry_angles
        self.collimator_angles = collimator_angles
        self.field_size = new_beam_data.field_size
        self.SID = new_beam_data.sid
        self.iso_center = new_beam_data.iso_center

        if self.dtype is None:
            return
        if self.device is None:
            return
        if self.dose_grid_shape is None:
            return
        if self.dose_grid_spacing is None:
            return
        if self.number_of_beams is None:
            return
        
        if self._adjust_values and initialize_beam_validation_layer:
            self.beam_validation_layer = BeamValidationLayer(
                self.machine_config,
                device = self.device,
                dtype=self.dtype,
                field_size=self.field_size,
            )

        if initialize_fluence_map_layer:
            self.fluence_map_layer = FluenceMapLayer(
                self.machine_config,
                device = self.device,
                dtype=self.dtype,
                field_size=self.field_size,
                verbose=self.verbose
            )
        
        if initialize_fluence_volume_layer:
            self.fluence_volume_layer = FluenceVolumeLayer(
                self.machine_config, 
                device = self.device,
                dtype=self.dtype,
                resolution=self.dose_grid_spacing,
                ct_array_shape=self.dose_grid_shape,
                sid=self.SID,
                iso_center=self.iso_center,
                field_size=self.field_size,
                verbose=self.verbose
            )

        if initialize_rad_depth_layer:
            self.rad_depth_layer = RadiologicalDepthLayer(
                self.machine_config, 
                device = self.device,
                dtype=self.dtype,
                resolution=self.dose_grid_spacing,
                ct_array_shape=self.dose_grid_shape,
                gantry_angles=self.gantry_angles,
                iso_center=self.iso_center,
                verbose=self.verbose
            )

        if initialize_pencil_beam_kernel_layer:
            self.pencil_beam_kernel_layer = PencilBeamKernelLayer(
                self.machine_config, 
                device = self.device,
                dtype=self.dtype,
                resolution=self.dose_grid_spacing,
                kernel_size=self.kernel_size,
                verbose=self.verbose
            )

        if initialize_beam_wise_conv_layer:
            self.beam_wise_conv_layer = BeamWiseConvolutionalLayer(
                self.device, 
                self.dtype,
                verbose=self.verbose
            )

        if initialize_rotation_layer:
            self.rotation_layer = BeamRotationLayer(
                self.machine_config, 
                device=self.device, 
                dtype=self.dtype,
                ct_array_shape=self.dose_grid_shape,
                gantry_angles=self.gantry_angles,
                iso_center=self.iso_center,            
                resolution=self.dose_grid_spacing,
                verbose=self.verbose
            )

        self.layers_initialized = True

    @property
    def iso_center_voxel(self) -> tuple[int, int, int]:
        if self.iso_center is None:
            return None

        sx, sy, sz = self.dose_grid_shape
        rx, ry, rz = self.dose_grid_spacing
        X, Y, Z = self.iso_center  # physical coords, origin at isocenter corner
        X_center, Y_center, Z_center = (X, Y, Z)

        # Convert physical coords to voxel indices and round to nearest voxel
        ix = int(X_center / rx)
        iy = int(Y_center / ry)
        iz = int(Z_center / rz)

        # Optionally clamp to valid voxel range
        ix = max(0, min(sx - 1, ix))
        iy = max(0, min(sy - 1, iy))
        iz = max(0, min(sz - 1, iz))

        return (ix, iy, iz)


    def _assert_sizes(self, density_image, leaf_positions, jaw_positions, mus, fluence_maps=None):
        """Validate input tensor sizes."""

        G = self.number_of_beams

        if fluence_maps is not None:
            # Derive B from fluence_maps; mus is optional in this path
            fm_h, fm_w = self.field_size
            if fluence_maps.dim() == 4:
                B = fluence_maps.shape[0]
                expected_fm = (B, G, fm_h, fm_w)
                assert fluence_maps.shape == expected_fm, \
                    f"Fluence maps shape mismatch: expected {expected_fm}, got {fluence_maps.shape}"
            elif fluence_maps.dim() == 3:
                assert fluence_maps.shape[0] % G == 0, \
                    f"Fluence maps leading dim {fluence_maps.shape[0]} is not divisible by G={G}"
                B = fluence_maps.shape[0] // G
                expected_fm = (B * G, fm_h, fm_w)
                assert fluence_maps.shape == expected_fm, \
                    f"Fluence maps shape mismatch: expected {expected_fm}, got {fluence_maps.shape}"
            else:
                raise ValueError(
                    f"fluence_maps must be 3D [B*G, H, W] or 4D [B, G, H, W], got {fluence_maps.dim()}D"
                )

            # Validate mus only when provided
            if mus is not None:
                assert mus.dim() == 2, \
                    f"MUs needs 2 dimensions [B, G], got {mus.dim()}D: {mus.shape}"
                expected_mus = (B, G)
                assert mus.shape == expected_mus, \
                    f"MUs shape mismatch: expected {expected_mus}, got {mus.shape}"

            devices = {fluence_maps.device}
            dtypes = {fluence_maps.dtype}
            if mus is not None:
                devices.add(mus.device)
                dtypes.add(mus.dtype)
        else:
            B = leaf_positions.shape[0]
            assert leaf_positions.dim() == 4, \
                f"Leaf positions needs 4 dimensions [B, 2, CP, N], got {leaf_positions.dim()}D: {leaf_positions.shape}"
            assert mus.dim() == 2, \
                f"MUs needs 2 dimensions [B, CP], got {mus.dim()}D: {mus.shape}"

            assert leaf_positions.shape[0] == B and mus.shape[0] == B, \
                f"Batch size mismatch: ct={B}, leaf_positions={leaf_positions.shape[0]}, mus={mus.shape[0]}"

            expected_leaf = (B, G, self.machine_config.number_of_leaf_pairs, 2)
            assert leaf_positions.shape == expected_leaf, \
                f"Leaf positions shape mismatch: expected {expected_leaf}, got {leaf_positions.shape}"

            expected_mus = (B, G)
            assert mus.shape == expected_mus, \
                f"MUs shape mismatch: expected {expected_mus}, got {mus.shape}"

            if jaw_positions is not None:
                assert jaw_positions.dim() == 3, \
                    f"Jaw positions needs 3 dimensions [B, 2, CP], got {jaw_positions.dim()}D: {jaw_positions.shape}"

                assert jaw_positions.shape[0] == B, \
                    f"Batch size mismatch: ct={B}, jaw_positions={jaw_positions.shape[0]}"

                expected_jaw = (B, G, 2)
                assert jaw_positions.shape == expected_jaw, \
                    f"Jaw positions shape mismatch: expected {expected_jaw}, got {jaw_positions.shape}"

            devices = {leaf_positions.device, mus.device}
            if jaw_positions is not None:
                devices.add(jaw_positions.device)
            dtypes = {leaf_positions.dtype, mus.dtype}
            if jaw_positions is not None:
                dtypes.add(jaw_positions.dtype)

        if density_image is None:
            raise ValueError("CT image must be provided.")
        assert density_image.dim() == 4, \
            f"CT image needs 4 dimensions [B, D, H, W], got {density_image.dim()}D: {density_image.shape}"

        expected_ct = (B, *self.dose_grid_shape)
        assert density_image.shape == expected_ct, \
            f"CT shape mismatch: expected {expected_ct}, got {density_image.shape}"

        devices.add(density_image.device)
        dtypes.add(density_image.dtype)

        if len(devices) != 1:
            raise ValueError(f"Device mismatch among tensors: {devices}")

        if len(dtypes) != 1:
            raise ValueError(f"Dtype mismatch among tensors: {dtypes}")

    def _reshape_fluence_maps(self, fluence_maps: torch.Tensor) -> tuple[torch.Tensor, int]:
        G = self.number_of_beams
        if fluence_maps.dim() == 4:
            batch_size = fluence_maps.shape[0]
            return fluence_maps.reshape(batch_size * G, fluence_maps.shape[2], fluence_maps.shape[3]), batch_size
        batch_size = fluence_maps.shape[0] // G
        return fluence_maps, batch_size

    def _autocast_context(self):
        if self.device is None:
            return nullcontext()
        if self.device.type == "cpu" and self.dtype not in {torch.bfloat16, torch.float16}:
            return nullcontext()
        return torch.amp.autocast(self.device.type, dtype=self.dtype)

    def _forward_dense(
        self,
        leaf_positions: torch.Tensor | None,
        mus: torch.Tensor | None,
        jaw_positions: torch.Tensor | None,
        density_image: torch.Tensor,
        return_intermediates: bool = False,
        fluence_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with self._autocast_context():
            if density_image.dim() == 3:
                density_image = density_image.unsqueeze(0)
            with torch.no_grad():
                batched_radiological_depths = self.rad_depth_layer(density_image).detach()
                batched_kernels = self.pencil_beam_kernel_layer(batched_radiological_depths).detach()

            G = self.number_of_beams

            if fluence_maps is not None:
                batched_fluence_maps, batch_size = self._reshape_fluence_maps(fluence_maps)
            else:
                if self._adjust_values:
                    leaf_positions, jaw_positions, mus = self.beam_validation_layer(
                        leaf_positions=leaf_positions, jaw_positions=jaw_positions, mus=mus
                    )

                batched_fluence_maps = self.fluence_map_layer(leaf_positions, jaw_positions)
                batch_size = leaf_positions.shape[0]

            if (self.collimator_angles != 0.0).any():
                batched_fluence_maps = rotate_2d_images(
                    batched_fluence_maps,
                    self.collimator_angles,
                    device=self.device,
                    dtype=self.dtype
                )

            batched_fluence_volumes = self.fluence_volume_layer(batched_fluence_maps)
            batched_accumulated_dose = self.beam_wise_conv_layer(
                batched_fluence_volumes, batched_kernels
            )
            batched_accumulated_dose.mul_(self.machine_config.mean_photon_energy_MeV)

            D_, H_, W_, _ = batched_accumulated_dose.shape[1:]
            batched_accumulated_dose = batched_accumulated_dose.view(batch_size, G, D_, H_, W_)
            if mus is not None:
                batched_accumulated_dose.mul_(mus[:, :, None, None, None])

            batched_accumulated_dose = self.rotation_layer(batched_accumulated_dose)
            batched_accumulated_dose = batched_accumulated_dose.sum(dim=1).to(self.dtype)

        if return_intermediates:
            return batched_radiological_depths, batched_fluence_maps, batched_fluence_volumes, batched_accumulated_dose
        return batched_accumulated_dose

    def forward(
        self,
        leaf_positions: torch.Tensor | None,
        mus: torch.Tensor | None,
        jaw_positions: torch.Tensor | None,
        density_image: torch.Tensor,
        return_intermediates: bool = False,
        fluence_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Runs the full dose calculation pipeline.

        Args:
            leaf_positions: Leaf positions [B, G, N, 2]. Not required when fluence_maps is provided.
            mus: Monitor units [B, G]. Optional when fluence_maps is provided; if supplied the dose
                is scaled by MUs, if omitted the fluence maps are used as-is.
            jaw_positions: Jaw positions [B, G, 2]. Not required when fluence_maps is provided.
            density_image: CT image tensor [B, D, H, W].
            return_intermediates: If True, also return intermediate tensors.
            fluence_maps: Optional pre-computed fluence maps [B, G, H, W] or [B*G, H, W].
                If provided, the FluenceMapLayer is skipped and leaf_positions/jaw_positions
                are ignored. The maps are used directly as input to the FluenceVolumeLayer.

        Returns:
            Dose tensor [B, D, H, W].
        """
        if fluence_maps is not None:
            self._set_device_dtype(fluence_maps.device, fluence_maps.dtype)
        else:
            self._set_device_dtype(leaf_positions.device, leaf_positions.dtype)

        if not self.layers_initialized:
            raise Exception("Layers haven't been initialized yet. Dose engine cannot perform dose calculations.")

        self._assert_sizes(density_image, leaf_positions, jaw_positions, mus, fluence_maps=fluence_maps)
        return self._forward_dense(
            leaf_positions=leaf_positions,
            mus=mus,
            jaw_positions=jaw_positions,
            density_image=density_image,
            return_intermediates=return_intermediates,
            fluence_maps=fluence_maps,
        )

    def compute_dose(
        self,
        beam_input: BeamSequence | Beam,
        density_image: torch.Tensor | None = None,
        return_intermediates: bool = False,
        overwrite: bool = False,
        fluence_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute dose from a BeamSequence or Beam.

        Args:
            beam_input: BeamSequence (shapes: mus [CP], leaf_positions [CP, N, 2], jaw_positions [CP, 2])
                or a single Beam. Always required for geometry (gantry angles, iso center, etc.).
            density_image: CT image tensor [1, D, H, W].
            return_intermediates: If True, also return intermediate tensors.
            overwrite: Re-initialize layers even if already set up.
            fluence_maps: Optional pre-computed fluence maps [1, G, H, W] or [G, H, W].
                If provided, the FluenceMapLayer is skipped and leaf/jaw positions from
                beam_input are ignored. G must equal the number of beams in beam_input.

        Returns:
            Dose tensor [1, D, H, W].
        """
        self._initialize_layers(beam_input, overwrite)

        # Add batching dimension to parameters
        if density_image is not None:
            ct_tensor = density_image
            if ct_tensor.dim() == 3:
                ct_tensor = ct_tensor.unsqueeze(0)
        else:
            ct_tensor = None

        if isinstance(beam_input, Beam):
            leaf_positions = beam_input.leaf_positions.unsqueeze(0).unsqueeze(0)
            mus = beam_input.mu.unsqueeze(0).unsqueeze(0)
            jaw_positions = beam_input.jaw_positions.unsqueeze(0).unsqueeze(0)
        elif isinstance(beam_input, BeamSequence):
            leaf_positions = beam_input.leaf_positions.unsqueeze(0)
            mus = beam_input.mus.unsqueeze(0)
            jaw_positions = beam_input.jaw_positions.unsqueeze(0)

        # Normalise fluence_maps to [1, G, H, W] so forward() can reshape to [B*G, H, W]
        if fluence_maps is not None and fluence_maps.dim() == 3:
            fluence_maps = fluence_maps.unsqueeze(0)  # [G, H, W] -> [1, G, H, W]

        return self.forward(
            leaf_positions=leaf_positions,
            mus=mus,
            jaw_positions=jaw_positions,
            density_image=ct_tensor,
            return_intermediates=return_intermediates,
            fluence_maps=fluence_maps,
        )

    def compute_dose_sequential(
        self,
        beam_sequence: BeamSequence,
        density_image: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute dose by processing beams sequentially or in batches (memory efficient).

        Args:
            beam_sequence: BeamSequence containing all control points
            density_image: CT image tensor [1, D, H, W]

        Returns:
            Accumulated dose tensor [1, H, D, W]
        """
        self._initialize_layers(beam_sequence)
        total_dose = None

        # Process beams one by one
        for beam in beam_sequence:
            beam_dose = self.compute_dose(
                beam,
                density_image=density_image,
                overwrite=True
            )

            if total_dose is None:
                total_dose = beam_dose
            else:
                total_dose = total_dose + beam_dose

        self._initialize_layers(beam_sequence, overwrite=True)
        return total_dose

    def calibrate(self, 
                  calibration_mu: float = None,
                  original_beam_template: BeamSequence | None = None,
                  verbose: bool = True) -> None: # Keep in dose engine
        if not self.layers_initialized:
            raise Exception("Layers must be fully initialized for calibration.")

        center_x, _, center_z = torch.tensor(self.dose_grid_spacing) * (torch.tensor(self.dose_grid_shape)) / 2
        iso_center = (center_x.item(), 100.0, center_z.item())
        beam = Beam.create(0.0, self.machine_config.number_of_leaf_pairs, 0.0, (100.0, 100.0), iso_center=iso_center, device=self.device, dtype=self.dtype)
        if calibration_mu is None:
            calibration_mu = self.machine_config.calibration_mu
        beam.mu = calibration_mu * beam.mu
        water_attenuation = torch.ones(self.dose_grid_shape).to(self.device).to(self.dtype)

        self.layers_initialized = False
        old_kernel_size = self.kernel_size

        self.kernel_size = max(self.dose_grid_shape)

        dose = self.compute_dose(
            beam,
            density_image=water_attenuation,
            overwrite=True
            )


        # Get center dose (at 10cm depth - index 50 for 100 voxels)
        center_dose = dose[0, *self.iso_center_voxel].detach().cpu().numpy().item()

        # Calculate calibration factor
        # This gives the factor to normalize to 1 Gy per MU at reference conditions
        calibration_factor = self.machine_config.mean_photon_energy_MeV / center_dose

        if abs(center_dose - 1.0) > 0.001:
            if verbose:
                print(f"Calibration failed. Adjusting calibration factor to: {calibration_factor}")
            self.machine_config.mean_photon_energy_MeV = calibration_factor

        self.kernel_size = old_kernel_size
        if original_beam_template is not None:
            self._initialize_layers(original_beam_template, True)
