"""
Standalone nnUNet Regression Model Inference

A minimalistic, self-contained inference class for nnUNet regression models.
No nnUNet dependencies required - just torch, numpy, and scipy.

Usage Example:
    # 1. Export a trained nnUNet model to standalone format
    from standalone_regression_inference import export_nnunet_model_to_standalone

    export_nnunet_model_to_standalone(
        checkpoint_path="nnUNet_results/Dataset015/nnUNetTrainerRegression_mae/fold_0/checkpoint_best.pth",
        output_dir="./model_bundle",
        example_input_shape=(128, 128, 128)
    )

    # 2. Use the standalone inference class
    from standalone_regression_inference import StandaloneRegressionInference

    predictor = StandaloneRegressionInference(
        model_path="./model_bundle",
        device="cuda"
    )

    # Predict (input should be a NumPy array with shape (H, W, D) or (1, H, W, D))
    output = predictor.predict(
        input_array=input_image,  # NumPy array
        apply_normalization=True,
        apply_denormalization=True,
        tile_step_size=0.5  # Optional: control overlap (lower = more overlap, slower but better quality)
    )

Dependencies:
    - torch >= 2.0
    - numpy
    - scipy (for Gaussian filter)

Author: Claude Code
License: Apache 2.0
"""

import json
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


# ============================================================================
# Helper Functions
# ============================================================================

def compute_steps_for_sliding_window(
    image_size: Tuple[int, ...],
    tile_size: Tuple[int, ...],
    tile_step_size: float
) -> List[List[int]]:
    """
    Calculate sliding window step positions for tiled inference.

    Args:
        image_size: Size of the input image (e.g., (512, 512, 300))
        tile_size: Size of each tile/patch (e.g., (128, 128, 128))
        tile_step_size: Step size as fraction of tile_size (0 < value <= 1)
                       0.5 = 50% overlap, 1.0 = no overlap

    Returns:
        List of lists containing step positions for each dimension

    Example:
        >>> compute_steps_for_sliding_window((110, 110, 110), (64, 64, 64), 0.5)
        [[0, 23, 46], [0, 23, 46], [0, 23, 46]]
    """
    assert all(i >= j for i, j in zip(image_size, tile_size)), \
        "Image size must be >= tile size in all dimensions"
    assert 0 < tile_step_size <= 1, "tile_step_size must be in range (0, 1]"

    # Calculate target step size in voxels
    target_step_sizes_in_voxels = [int(i * tile_step_size) for i in tile_size]

    # Calculate number of steps needed in each dimension
    num_steps = [
        int(np.ceil((i - k) / j)) + 1
        for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)
    ]

    # Compute actual step positions
    steps = []
    for dim in range(len(tile_size)):
        max_step_value = image_size[dim] - tile_size[dim]

        if num_steps[dim] > 1:
            # Evenly distribute steps
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # Doesn't matter, only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]
        steps.append(steps_here)

    return steps


@lru_cache(maxsize=4)
def compute_gaussian_weight(
    tile_size: Tuple[int, ...],
    sigma_scale: float = 1.0 / 8,
    value_scaling_factor: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "cuda"
) -> torch.Tensor:
    """
    Compute Gaussian importance map for blending overlapping predictions.

    Args:
        tile_size: Size of the tile/patch
        sigma_scale: Scale factor for Gaussian sigma (default: 1/8 of tile size)
        value_scaling_factor: Scale factor for the peak value
        dtype: PyTorch data type for the output tensor
        device: Device to place the tensor on

    Returns:
        Gaussian importance map as a PyTorch tensor

    Note:
        The result is cached for efficiency since patch size is constant
        during inference. The Gaussian cannot be 0 to avoid NaN divisions.
    """
    # Create zero array and set center to 1
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1

    # Apply Gaussian filter
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    # Convert to PyTorch tensor
    gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

    # Normalize by max value
    gaussian_importance_map /= (torch.max(gaussian_importance_map) / value_scaling_factor)
    gaussian_importance_map = gaussian_importance_map.to(device=device, dtype=dtype)

    # Ensure no zeros to prevent NaN when dividing
    mask = gaussian_importance_map == 0
    if mask.any():
        gaussian_importance_map[mask] = torch.min(gaussian_importance_map[~mask])

    return gaussian_importance_map


# ============================================================================
# Main Inference Class
# ============================================================================

class StandaloneRegressionInference:
    """
    Standalone inference class for nnUNet regression models.

    This class provides a self-contained solution for running inference with
    nnUNet regression models. It includes:
    - TorchScript model loading
    - Normalization and denormalization
    - Sliding window inference with Gaussian blending
    - No nnUNet dependencies required

    Attributes:
        model: TorchScript traced model
        metadata: Configuration and normalization parameters
        device: PyTorch device (cuda or cpu)
        patch_size: Size of inference patches
        tile_step_size: Default overlap for sliding window
        use_gaussian: Whether to use Gaussian weighting for blending
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Union[str, torch.device] = "cuda",
        inference_amp: bool = False,
        patch_batch_size: int = 1,
    ):
        """
        Initialize the standalone inference predictor.

        Args:
            model_path: Path to the model bundle directory containing:
                       - model.pt (TorchScript traced model)
                       - metadata.json (configuration and normalization params)
            device: Device to run inference on ('cuda', 'cpu', or torch.device)
            inference_amp: Run CUDA network forwards under float16 autocast. Sliding-window
                           accumulation and output denormalization remain float32.
            patch_batch_size: Number of same-sized sliding-window patches evaluated in one
                              model call. Blending is still applied one patch at a time in
                              the original traversal order.

        Raises:
            FileNotFoundError: If model or metadata files are not found
            ValueError: If metadata is invalid or incompatible
        """
        self.model_path = Path(model_path)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.inference_amp = bool(inference_amp) and self.device.type == "cuda"
        self.patch_batch_size = int(patch_batch_size)
        if self.patch_batch_size < 1:
            raise ValueError("patch_batch_size must be at least 1")

        # Load metadata
        metadata_path = self.model_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Load TorchScript model
        model_file = self.model_path / "model.pt"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        print(f"Loading TorchScript model from {model_file}...")
        self.model = torch.jit.load(str(model_file), map_location=self.device)
        self.model.eval()

        # Extract inference configuration
        self.patch_size = tuple(self.metadata['inference_config']['patch_size'])
        self.tile_step_size = self.metadata['inference_config']['tile_step_size']
        self.use_gaussian = self.metadata['inference_config']['use_gaussian']

        # Precompute Gaussian weights for efficiency
        if self.use_gaussian:
            self._gaussian_cache = compute_gaussian_weight(
                self.patch_size,
                sigma_scale=1.0 / 8,
                value_scaling_factor=10.0,  # nnUNet uses 10 for better blending
                dtype=torch.float32,
                device=self.device
            )
        else:
            self._gaussian_cache = None

        print(f"Loaded model: {self.metadata['model_info']['trainer_name']}")
        print(f"Patch size: {self.patch_size}, Tile step size: {self.tile_step_size}")
        print(f"Gaussian blending: {self.use_gaussian}")
        print(f"FP16 autocast: {self.inference_amp}")
        print(f"Patch batch size: {self.patch_batch_size}")

    def _model_forward(self, data: torch.Tensor) -> torch.Tensor:
        """Inference-only model call; AMP affects convolutions, not FP32 blending."""
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.inference_amp,
        ):
            return self.model(data)

    def _normalize(self, data: np.ndarray, channel: str = 'input') -> np.ndarray:
        """
        Normalize input data using stored normalization parameters.

        Args:
            data: Input array to normalize
            channel: 'input' or 'output' to select normalization params

        Returns:
            Normalized array
        """
        norm_config = self.metadata['normalization'][channel]
        scheme = norm_config['scheme']

        # Convert to float32 for normalization
        data = data.astype(np.float32, copy=False)

        if scheme == 'ZScoreNormalization':
            mean = norm_config['mean']
            std = norm_config['std']
            data = (data - mean) / max(std, 1e-8)

        elif scheme == 'CTNormalization':
            mean = norm_config['mean']
            std = norm_config['std']
            lower = norm_config['percentile_00_5']
            upper = norm_config['percentile_99_5']
            # Clip then normalize
            np.clip(data, lower, upper, out=data)
            data = (data - mean) / max(std, 1e-8)

        elif scheme == 'GlobalNormalization':
            mean = norm_config['mean']
            std = norm_config['std']
            data = (data - mean) / max(std, 1e-8)

        elif scheme == 'NoNormalization':
            # No normalization
            pass

        elif scheme == 'RescaleTo01Normalization':
            data = data - data.min()
            data = data / np.clip(data.max(), a_min=1e-8, a_max=None)

        else:
            raise ValueError(f"Unknown normalization scheme: {scheme}")

        return data

    def _denormalize(self, data: np.ndarray, channel: str = 'output') -> np.ndarray:
        """
        Denormalize output data back to original intensity range.

        Args:
            data: Normalized array to denormalize
            channel: 'input' or 'output' to select normalization params

        Returns:
            Denormalized array in original intensity range
        """
        norm_config = self.metadata['normalization'][channel]
        scheme = norm_config['scheme']

        if scheme in ['ZScoreNormalization', 'CTNormalization', 'GlobalNormalization']:
            mean = norm_config['mean']
            std = norm_config['std']
            # Reverse: x_orig = (x_norm * std) + mean
            data = data * std + mean

        elif scheme == 'NoNormalization':
            # No denormalization needed
            pass

        elif scheme == 'RescaleTo01Normalization':
            # This would need original min/max, which aren't stored
            # Return as-is
            pass

        else:
            raise ValueError(f"Unknown normalization scheme: {scheme}")

        return data

    def _sliding_window_inference(
        self,
        data: torch.Tensor,
        tile_step_size: Optional[float] = None
    ) -> torch.Tensor:
        """
        Perform sliding window inference with Gaussian blending.

        Args:
            data: Input tensor of shape (1, C, H, W, D) or (1, C, H, W)
            tile_step_size: Optional override for tile step size

        Returns:
            Prediction tensor of same spatial shape as input
        """
        # Use provided tile_step_size or default
        effective_step_size = tile_step_size if tile_step_size is not None else self.tile_step_size

        # Get spatial dimensions (excluding batch and channel)
        data_shape = data.shape[2:]
        num_dimensions = len(data_shape)

        # Check if image is smaller than patch size
        if any(i < j for i, j in zip(data_shape, self.patch_size)):
            # For small images, just run inference directly
            return self._model_forward(data)

        # Compute sliding window steps
        steps = compute_steps_for_sliding_window(data_shape, self.patch_size, effective_step_size)

        # Initialize output accumulators
        predicted_logits = torch.zeros(
            (1, 1) + data_shape,
            dtype=torch.float32,
            device=self.device
        )
        n_predictions = torch.zeros(
            data_shape,
            dtype=torch.float32,
            device=self.device
        )

        # Get Gaussian weights if enabled
        if self.use_gaussian and self._gaussian_cache is not None:
            gaussian = self._gaussian_cache
        else:
            gaussian = torch.ones(self.patch_size, dtype=torch.float32, device=self.device)

        if num_dimensions not in (2, 3):
            raise ValueError(f"Unsupported number of dimensions: {num_dimensions}")

        # Evaluate several patches per model call, then blend each prediction in the
        # original traversal order. Keeping accumulation order fixed avoids introducing
        # any additional FP32 summation differences beyond the model's batched forward.
        positions = list(product(*steps))
        for first in range(0, len(positions), self.patch_batch_size):
            batch_positions = positions[first:first + self.patch_batch_size]
            patches = []
            for position in batch_positions:
                spatial_slices = tuple(
                    slice(start, start + size)
                    for start, size in zip(position, self.patch_size)
                )
                patches.append(data[(slice(None), slice(None), *spatial_slices)])

            predictions = self._model_forward(torch.cat(patches, dim=0))
            for batch_index, position in enumerate(batch_positions):
                spatial_slices = tuple(
                    slice(start, start + size)
                    for start, size in zip(position, self.patch_size)
                )
                prediction = predictions[batch_index:batch_index + 1]
                if self.use_gaussian:
                    prediction = prediction * gaussian
                predicted_logits[(slice(None), slice(None), *spatial_slices)] += prediction
                n_predictions[spatial_slices] += gaussian

        # Normalize by cumulative weights
        predicted_logits = predicted_logits / n_predictions

        return predicted_logits

    def predict(
        self,
        input_array: np.ndarray,
        apply_normalization: bool = True,
        apply_denormalization: bool = True,
        tile_step_size: Optional[float] = None
    ) -> np.ndarray:
        """
        Run inference on an input image.

        Args:
            input_array: Input image as NumPy array
                        Shape: (H, W, D) or (1, H, W, D) for 3D
                               (H, W) or (1, H, W) for 2D
            apply_normalization: If True, normalize input using stored parameters
            apply_denormalization: If True, denormalize output to original range
            tile_step_size: Optional override for sliding window overlap
                          Lower values = more overlap = slower but better quality
                          Default: 0.5 (50% overlap)

        Returns:
            Prediction as NumPy array with same shape as input

        Example:
            >>> predictor = StandaloneRegressionInference("./model_bundle")
            >>> output = predictor.predict(input_cbct, tile_step_size=0.3)
        """
        # Ensure input has channel dimension
        if input_array.ndim == 3:
            # (H, W, D) -> (1, H, W, D)
            input_array = input_array[np.newaxis, ...]
        elif input_array.ndim == 2:
            # (H, W) -> (1, H, W)
            input_array = input_array[np.newaxis, ...]

        # Normalize if requested
        if apply_normalization:
            input_array = self._normalize(input_array, channel='input')

        # Convert to PyTorch tensor: (C, H, W, D) -> (1, C, H, W, D)
        input_tensor = torch.from_numpy(input_array).float()
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        # Run sliding window inference
        prediction = self._sliding_window_inference(input_tensor, tile_step_size)

        # Convert back to NumPy
        prediction_np = prediction.squeeze(0).cpu().numpy()  # (1, H, W, D) -> (C, H, W, D)

        # Denormalize if requested
        if apply_denormalization:
            prediction_np = self._denormalize(prediction_np, channel='output')

        # Remove channel dimension to match input: (1, H, W, D) -> (H, W, D)
        if prediction_np.shape[0] == 1:
            prediction_np = prediction_np[0]

        return prediction_np


# ============================================================================
# Model Export Function
# ============================================================================

def export_nnunet_model_to_standalone(
    checkpoint_path: Union[str, Path],
    output_dir: Union[str, Path],
    example_input_shape: Optional[Tuple[int, ...]] = None,
    device: Union[str, torch.device] = "cuda"
):
    """
    Export a trained nnUNet regression model to standalone format.

    This function converts an nnUNet checkpoint to a self-contained bundle
    that can be used with the StandaloneRegressionInference class without
    requiring any nnUNet dependencies.

    Args:
        checkpoint_path: Path to the nnUNet checkpoint file
                        (e.g., "fold_0/checkpoint_best.pth")
        output_dir: Directory to save the model bundle
        example_input_shape: Optional shape for tracing (default: use patch_size from config)
                           Should be (H, W, D) for 3D or (H, W) for 2D
        device: Device to use for tracing

    Creates:
        output_dir/
        ├── model.pt          # TorchScript traced model
        └── metadata.json     # Configuration and normalization parameters

    Example:
        >>> export_nnunet_model_to_standalone(
        ...     checkpoint_path="nnUNet_results/Dataset015/fold_0/checkpoint_best.pth",
        ...     output_dir="./model_bundle",
        ...     example_input_shape=(128, 128, 128)
        ... )
    """
    # Import nnUNet dependencies (only needed for export)
    try:
        import nnunetv2  # type: ignore
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager  # type: ignore
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class  # type: ignore
        from batchgenerators.utilities.file_and_folder_operations import load_json, join  # type: ignore
    except ImportError as e:
        raise ImportError(
            "nnUNet must be installed to export models. "
            "Install with: pip install nnunetv2"
        ) from e

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device) if isinstance(device, str) else device

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract checkpoint information
    trainer_name = checkpoint['trainer_name']
    init_args = checkpoint['init_args']
    network_weights = checkpoint['network_weights']

    print(f"Trainer: {trainer_name}")
    print(f"Configuration: {init_args['configuration']}")
    print(f"Fold: {init_args['fold']}")

    # Load plans and dataset.json from the same directory
    model_dir = checkpoint_path.parent.parent
    plans_path = model_dir / 'plans.json'
    dataset_json_path = model_dir / 'dataset.json'

    if not plans_path.exists():
        raise FileNotFoundError(f"plans.json not found at {plans_path}")
    if not dataset_json_path.exists():
        raise FileNotFoundError(f"dataset.json not found at {dataset_json_path}")

    plans = load_json(str(plans_path))
    dataset_json = load_json(str(dataset_json_path))

    # Extract configuration
    plans_manager = PlansManager(plans)
    config_manager = plans_manager.get_configuration(init_args['configuration'])

    patch_size = config_manager.patch_size
    norm_stats = plans_manager.foreground_intensity_properties_per_channel
    norm_schemes = config_manager.normalization_schemes

    print(f"Patch size: {patch_size}")
    print(f"Normalization schemes: {norm_schemes}")

    # Build network architecture
    print("Building network architecture...")
    trainer_class = recursive_find_python_class(
        folder=join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        class_name=trainer_name,
        current_module="nnunetv2.training.nnUNetTrainer"
    )

    if trainer_class is None:
        raise RuntimeError(
            f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. '
            f'Please ensure the trainer is available in the nnUNet installation.'
        )

    network = trainer_class.build_network_architecture(
        architecture_class_name=config_manager.network_arch_class_name,
        arch_init_kwargs=config_manager.network_arch_init_kwargs,
        arch_init_kwargs_req_import=config_manager.network_arch_init_kwargs_req_import,
        num_input_channels=1,  # Regression uses 1 input channel
        num_output_channels=1,  # Regression uses 1 output channel
        enable_deep_supervision=False
    )

    # Load weights
    network.load_state_dict(network_weights)
    network = network.to(device)
    network.eval()

    print("Network loaded successfully")

    # Determine input shape for tracing
    if example_input_shape is None:
        example_input_shape = tuple(patch_size)

    print(f"Tracing model with input shape: (1, 1, {example_input_shape})...")

    # Trace model to TorchScript
    with torch.no_grad():
        example_input = torch.randn(1, 1, *example_input_shape, device=device)
        traced_model = torch.jit.trace(network, example_input)

    # Save traced model
    model_save_path = output_dir / "model.pt"
    torch.jit.save(traced_model, str(model_save_path))
    print(f"Saved traced model to {model_save_path}")

    # Create metadata
    metadata = {
        "model_info": {
            "trainer_name": trainer_name,
            "dataset_name": dataset_json.get('name', 'Unknown'),
            "configuration": init_args['configuration'],
            "fold": init_args['fold']
        },
        "inference_config": {
            "patch_size": list(patch_size),
            "tile_step_size": 0.5,
            "use_gaussian": True
        },
        "normalization": {}
    }

    # Add normalization parameters for input and output channels
    for channel_idx, channel_name in [(0, 'input'), (1, 'output')]:
        if str(channel_idx) in norm_stats:
            stats = norm_stats[str(channel_idx)]
            scheme = norm_schemes[channel_idx] if channel_idx < len(norm_schemes) else 'ZScoreNormalization'

            metadata['normalization'][channel_name] = {
                'scheme': scheme,
                'mean': float(stats.get('mean', 0)),
                'std': float(stats.get('std', 1)),
                'percentile_00_5': float(stats.get('percentile_00_5', 0)),
                'percentile_99_5': float(stats.get('percentile_99_5', 0)),
                'median': float(stats.get('median', 0)),
                'min': float(stats.get('min', 0)),
                'max': float(stats.get('max', 0))
            }

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    # Verify the model can be loaded
    print("\nVerifying exported model...")
    try:
        predictor = StandaloneRegressionInference(output_dir, device=device)
        print("✓ Model bundle created successfully and can be loaded!")
        print(f"\nModel bundle saved to: {output_dir.absolute()}")
        print("\nYou can now use this model with:")
        print(f"  predictor = StandaloneRegressionInference('{output_dir}')")
        print(f"  output = predictor.predict(input_array)")
    except Exception as e:
        print(f"✗ Error verifying model: {e}")
        raise


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export nnUNet regression model to standalone format"
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help="Path to nnUNet checkpoint (e.g., fold_0/checkpoint_best.pth)"
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help="Output directory for model bundle"
    )
    parser.add_argument(
        '--input-shape',
        type=int,
        nargs='+',
        help="Input shape for tracing (e.g., 128 128 128). If not provided, uses patch_size from config"
    )
    parser.add_argument(
        '--device',
        default='cuda',
        help="Device to use for tracing (default: cuda)"
    )

    args = parser.parse_args()

    example_shape = tuple(args.input_shape) if args.input_shape else None

    export_nnunet_model_to_standalone(
        checkpoint_path=args.input,
        output_dir=args.output,
        example_input_shape=example_shape,
        device=args.device
    )
