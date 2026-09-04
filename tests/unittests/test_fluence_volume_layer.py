from pathlib import Path
import sys

from pydose_rt.utils.utils import get_shapes
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import pytest
import torch
from pydose_rt.data import MachineConfig
from pydose_rt.layers import FluenceVolumeLayer


# ---- Fixtures -----
@pytest.fixture
def fluence_volume_layer(default_machine_config, default_resolution, default_ct_array_shape):
    """Fixture to create a FluenceMapLayer instance"""
    return FluenceVolumeLayer(default_machine_config, default_resolution, default_ct_array_shape)


# ----- Tests -----
def test_fluence_volume_output_shape(fluence_volume_layer, default_machine_config, default_field_size, default_ct_array_shape, default_number_of_beams, default_dtype, default_device):
    """Test that fluence map behaves correctly based on input width."""
    # Arrange
    shapes = get_shapes(default_machine_config, 
                        number_of_beams=default_number_of_beams,
                        field_size=default_field_size,
                        ct_shape=default_ct_array_shape)
    fluence_map = torch.zeros(shapes["fluence_maps"], dtype=default_dtype, device=default_device)
    expected = shapes["fluence_volumes"]

    # Act
    fluence_volume = fluence_volume_layer(fluence_map)
    actual = fluence_volume.shape

    # Assert
    assert actual == expected, (
        f"Expected shape {expected}, but got {fluence_volume.shape}")


def test_fluence_volume_field_size_is_interpreted_in_physical_mm(default_machine_config, default_device, default_dtype):
    layer = FluenceVolumeLayer(
        default_machine_config,
        resolution=(1.0, 1.0, 2.0),
        ct_array_shape=(1, 1, 101),
        iso_center=(0.0, 0.0, 0.0),
        field_size=(400, 400),
        device=default_device,
        dtype=default_dtype,
    )

    # Voxel index 75 corresponds to +150 mm in width at 2 mm spacing.
    # With a 400 mm physical fluence field, the normalized sampling coordinate
    # must therefore be +150 / 200 = 0.75.
    assert torch.isclose(
        layer.sampling_grids[0, 75, 0, 0],
        torch.tensor(0.75, device=default_device, dtype=default_dtype),
        atol=1e-6,
    )
