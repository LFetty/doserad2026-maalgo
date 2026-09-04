import sys
import math

from pydose_rt.utils.utils import get_shapes
sys.path.append("../../")
import pytest
import numpy as np
import torch
from pydose_rt.data import MachineConfig
from pydose_rt.layers import RadiologicalDepthLayer


@pytest.fixture
def radiological_depth_layer(default_machine_config, default_resolution, default_ct_array_shape, default_gantry_angles, default_iso_center):
    """Fixture to create a FluenceMapLayer instance"""
    return RadiologicalDepthLayer(default_machine_config, default_resolution, default_ct_array_shape, default_gantry_angles, default_iso_center)


def test_radiological_depth_output_shape(radiological_depth_layer, default_machine_config, default_number_of_beams, default_ct_array_shape, default_device):
    """Test that fluence map behaves correctly based on input width."""
    expected = get_shapes(default_machine_config,
                          number_of_beams=default_number_of_beams,
                          ct_shape=default_ct_array_shape)["radiological_depths"]
    ct_array = torch.zeros(
        (
            1,
            default_ct_array_shape[0],
            default_ct_array_shape[1],
            default_ct_array_shape[2],
        ), dtype=torch.float32, device=default_device
    )

    radiological_depths = radiological_depth_layer(ct_array)

    assert (
        radiological_depths.shape == expected
    ), f"Expected shape {expected}, but got {radiological_depths.shape}"


class TestForwardBev:
    """Tests for RadiologicalDepthLayer.forward_bev (per-voxel BEV density & WEQ)."""

    @pytest.fixture
    def ct_shape(self):
        return (32, 40, 48)  # H, D, W

    @pytest.fixture
    def resolution(self):
        return (2.5, 2.5, 2.5)

    @pytest.fixture
    def iso_center(self):
        return (0.0, 0.0, 0.0)

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def layer_gantry0(self, default_machine_config, ct_shape, resolution, iso_center, device):
        return RadiologicalDepthLayer(
            default_machine_config, resolution, ct_shape,
            gantry_angles=[0.0], iso_center=iso_center,
            device=device, dtype=torch.float32,
        )

    @pytest.fixture
    def layer_multi(self, default_machine_config, ct_shape, resolution, iso_center, device):
        return RadiologicalDepthLayer(
            default_machine_config, resolution, ct_shape,
            gantry_angles=[0.0, 1.5708], iso_center=iso_center,
            device=device, dtype=torch.float32,
        )

    def test_output_shape(self, layer_gantry0, ct_shape, device):
        H, D, W = ct_shape
        ct = torch.ones((1, H, D, W), device=device, dtype=torch.float32)
        density_bev, weq_bev = layer_gantry0.forward_bev(ct)
        assert density_bev.shape == (1, D, H, W)  # B*G=1
        assert weq_bev.shape == (1, D, H, W)

    def test_output_shape_multi_beam(self, layer_multi, ct_shape, device):
        H, D, W = ct_shape
        ct = torch.ones((1, H, D, W), device=device, dtype=torch.float32)
        density_bev, weq_bev = layer_multi.forward_bev(ct)
        assert density_bev.shape == (2, D, H, W)  # B*G=2
        assert weq_bev.shape == (2, D, H, W)

    def test_gantry0_density_is_identity(self, layer_gantry0, ct_shape, device):
        """At gantry=0, BEV density should equal patient density (just permuted)."""
        H, D, W = ct_shape
        torch.manual_seed(42)
        ct = torch.rand((1, H, D, W), device=device, dtype=torch.float32)
        density_bev, _ = layer_gantry0.forward_bev(ct)
        # BEV [B*G, D, H, W] should match patient [B, H, D, W] permuted
        expected = ct[0].permute(1, 0, 2)  # [D, H, W]
        torch.testing.assert_close(density_bev[0], expected, atol=1e-4, rtol=1e-4)

    def test_weq_monotonically_increasing(self, layer_gantry0, ct_shape, device):
        """WEQ must increase monotonically along depth for positive density."""
        H, D, W = ct_shape
        torch.manual_seed(42)
        ct = torch.rand((1, H, D, W), device=device, dtype=torch.float32) + 0.1
        _, weq_bev = layer_gantry0.forward_bev(ct)
        diffs = weq_bev[0, 1:, :, :] - weq_bev[0, :-1, :, :]
        assert (diffs > 0).all()

    def test_rotated_density_sums_preserved(self, layer_multi, ct_shape, device):
        """Total density in BEV should approximate total density in patient frame."""
        H, D, W = ct_shape
        torch.manual_seed(42)
        # Place a sphere of high density in the center so rotation doesn't
        # lose mass to zero-padding at the edges.
        ct = torch.zeros((1, H, D, W), device=device, dtype=torch.float32)
        ch, cd, cw = H // 2, D // 2, W // 2
        r = min(H, D, W) // 4
        for h in range(H):
            for d in range(D):
                for w in range(W):
                    if (h - ch)**2 + (d - cd)**2 + (w - cw)**2 < r**2:
                        ct[0, h, d, w] = 1.0
        density_bev, _ = layer_multi.forward_bev(ct)
        patient_sum = ct.sum().item()
        # Beam 0 (gantry=0): should match closely
        bev0_sum = density_bev[0].sum().item()
        assert abs(bev0_sum - patient_sum) / patient_sum < 0.02

    def test_uniform_density_weq(self, layer_gantry0, ct_shape, device):
        """Uniform density=1 should give WEQ = physical depth at voxel center."""
        H, D, W = ct_shape
        ct = torch.ones((1, H, D, W), device=device, dtype=torch.float32)
        _, weq_bev = layer_gantry0.forward_bev(ct)

        ry = layer_gantry0.resolution[1]
        expected_center_depth = (torch.arange(D, device=device, dtype=torch.float32) + 0.5) * ry
        # Should be the same at every lateral position
        for h in [0, H // 2, H - 1]:
            for w in [0, W // 2, W - 1]:
                torch.testing.assert_close(
                    weq_bev[0, :, h, w], expected_center_depth, atol=1e-4, rtol=1e-4,
                )

    def test_zero_density_gives_zero_weq(self, layer_gantry0, ct_shape, device):
        H, D, W = ct_shape
        ct = torch.zeros((1, H, D, W), device=device, dtype=torch.float32)
        density_bev, weq_bev = layer_gantry0.forward_bev(ct)
        assert (weq_bev == 0).all()

    def test_negative_density_clamped(self, layer_gantry0, ct_shape, device):
        """Negative densities (air) should be clamped to 0 for WEQ."""
        H, D, W = ct_shape
        ct = -torch.ones((1, H, D, W), device=device, dtype=torch.float32)
        _, weq_bev = layer_gantry0.forward_bev(ct)
        assert (weq_bev == 0).all()

    def test_forward_bev_uses_physical_rotation_for_non_square_grid(
        self, default_machine_config, device
    ):
        """At 90 degrees, central BEV depth must advance one physical W voxel.

        This catches normalized-coordinate affine grids, where the same setup
        advanced by ``W / D`` voxels and shifted dense proton dose off ray.
        """
        H, D, W = 8, 50, 40
        resolution = (3.0, 3.0, 3.0)
        iso_center = (12.0, 75.0, 60.0)  # h=4, d=25, w=20 voxels
        layer = RadiologicalDepthLayer(
            default_machine_config,
            resolution,
            (H, D, W),
            gantry_angles=[math.pi / 2.0],
            iso_center=iso_center,
            device=device,
            dtype=torch.float32,
        )

        w_values = torch.arange(W, device=device, dtype=torch.float32)
        ct = w_values.view(1, 1, 1, W).expand(1, H, D, W).contiguous()
        density_bev, _ = layer.forward_bev(ct)

        h_idx = int(round(iso_center[0] / resolution[0]))
        w_idx = int(round(iso_center[2] / resolution[2]))
        d_idx = torch.tensor([20, 25, 30], device=device)
        expected = torch.tensor([25.0, 20.0, 15.0], device=device)
        torch.testing.assert_close(
            density_bev[0, d_idx, h_idx, w_idx],
            expected,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_forward_bev_entry_origin_starts_at_patient_outline(
        self, default_machine_config, device
    ):
        """Entry-origin dense BEV should include the upstream patient length.

        At 90 degrees the central ray enters through the high-W face.  The
        first BEV depth voxel must therefore sample W=39, not the isocenter
        depth window around W=20.
        """
        H, D, W = 8, 50, 40
        resolution = (3.0, 3.0, 3.0)
        iso_center = (12.0, 75.0, 60.0)  # h=4, d=25, w=20 voxels
        layer = RadiologicalDepthLayer(
            default_machine_config,
            resolution,
            (H, D, W),
            gantry_angles=[math.pi / 2.0],
            iso_center=iso_center,
            depth_origin="entry",
            device=device,
            dtype=torch.float32,
        )

        w_values = torch.arange(W, device=device, dtype=torch.float32)
        ct = w_values.view(1, 1, 1, W).expand(1, H, D, W).contiguous()
        density_bev, _ = layer.forward_bev(ct)

        h_idx = int(round(iso_center[0] / resolution[0]))
        w_idx = int(round(iso_center[2] / resolution[2]))
        d_idx = torch.tensor([0, 1, 34], device=device)
        expected_w = torch.tensor([39.0, 38.0, 5.0], device=device)
        torch.testing.assert_close(
            density_bev[0, d_idx, h_idx, w_idx],
            expected_w,
            atol=1e-4,
            rtol=1e-4,
        )

        _, weq_bev = layer.forward_bev(torch.ones_like(ct))
        expected_weq = (torch.arange(D, device=device, dtype=torch.float32) + 0.5) * resolution[1]
        torch.testing.assert_close(
            weq_bev[0, :W, h_idx, w_idx],
            expected_weq[:W],
            atol=1e-4,
            rtol=1e-4,
        )
