import pytest
import torch
from pydose_rt.layers import BevProjector


@pytest.fixture
def resolution():
    return (2.5, 2.5, 2.5)


@pytest.fixture
def ct_shape():
    return (32, 40, 48)  # H, D, W


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def projector(resolution):
    return BevProjector(resolution)


class TestSampleBev:

    def test_output_shape(self, projector, ct_shape, device):
        H, D, W = ct_shape
        vol = torch.ones((2, H, D, W), device=device)
        angles = torch.zeros(2, device=device)
        iso = torch.zeros((2, 3), device=device)
        out = projector.sample_bev(vol, angles, iso)
        assert out.shape == (2, D, H, W)

    def test_gantry0_is_permutation(self, projector, ct_shape, device):
        """At gantry=0 with iso=(0,0,0), BEV is just a permute of H and D."""
        H, D, W = ct_shape
        torch.manual_seed(42)
        vol = torch.rand((1, H, D, W), device=device)
        angles = torch.zeros(1, device=device)
        iso = torch.zeros((1, 3), device=device)
        bev = projector.sample_bev(vol, angles, iso)
        expected = vol[0].permute(1, 0, 2)  # [D, H, W]
        torch.testing.assert_close(bev[0], expected, atol=1e-4, rtol=1e-4)

    def test_nearest_mode_integer_values(self, projector, ct_shape, device):
        """Nearest mode preserves integer labels."""
        H, D, W = ct_shape
        vol = torch.randint(0, 5, (1, H, D, W), device=device, dtype=torch.float32)
        angles = torch.zeros(1, device=device)
        iso = torch.zeros((1, 3), device=device)
        bev = projector.sample_bev(vol, angles, iso, mode="nearest")
        unique = torch.unique(bev)
        assert all(v in range(5) for v in unique.int().tolist())

    def test_per_sample_angles(self, projector, device):
        """Different angles in a batch produce different outputs."""
        H, D, W = 16, 20, 24
        torch.manual_seed(7)
        vol = torch.rand((2, H, D, W), device=device)
        vol[1] = vol[0]
        angles = torch.tensor([0.0, 0.5], device=device)
        iso = torch.zeros((2, 3), device=device)
        bev = projector.sample_bev(vol, angles, iso)
        assert not torch.allclose(bev[0], bev[1], atol=1e-3)


class TestRotateToPatient:

    @pytest.mark.parametrize(
        "angle_rad,iso_mm",
        [
            (0.0, (0.0, 0.0, 0.0)),
            (0.5, (20.0, 15.0, -10.0)),
            (1.5708, (0.0, 25.0, 12.5)),
            (-0.75, (40.0, 50.0, 60.0)),
        ],
    )
    def test_matches_beam_rotation_layer(self, projector, ct_shape, device, angle_rad, iso_mm):
        from pydose_rt.data import MachineConfig
        from pydose_rt.layers.BeamRotationLayer import BeamRotationLayer

        H, D, W = ct_shape
        resolution = projector.resolution
        config = MachineConfig(
            preset="src/pydose_rt/data/machine_presets/test.json",
            head_scatter_amplitude=None,
            head_scatter_sigma=None,
            penumbra_fwhm=None,
            profile_corrections=None,
        )
        layer = BeamRotationLayer(
            config,
            ct_array_shape=ct_shape,
            iso_center=iso_mm,
            resolution=resolution,
            gantry_angles=torch.tensor([angle_rad], device=device, dtype=torch.float32),
            device=device,
            dtype=torch.float32,
        )

        torch.manual_seed(42)
        bev = torch.rand((1, 1, D, H, W), device=device, dtype=torch.float32)
        expected = layer(bev)[0, 0]
        got = projector.rotate_to_patient(
            bev[0],
            torch.tensor([angle_rad], device=device, dtype=torch.float32),
            torch.tensor([list(iso_mm)], device=device, dtype=torch.float32),
        )[0]

        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


class TestForward:

    def test_output_shapes(self, projector, ct_shape, device):
        H, D, W = ct_shape
        spr = torch.ones((1, H, D, W), device=device)
        angles = torch.zeros(1, device=device)
        iso = torch.zeros((1, 3), device=device)
        spr_bev, weq_bev = projector(spr, angles, iso)
        assert spr_bev.shape == (1, D, H, W)
        assert weq_bev.shape == (1, D, H, W)

    def test_uniform_spr_weq(self, projector, ct_shape, device):
        """Uniform SPR=1 gives WEQ = (i + 0.5) * step at each depth."""
        H, D, W = ct_shape
        spr = torch.ones((1, H, D, W), device=device)
        angles = torch.zeros(1, device=device)
        iso = torch.zeros((1, 3), device=device)
        _, weq_bev = projector(spr, angles, iso)

        ry = projector.resolution[1]
        expected = (torch.arange(D, device=device, dtype=torch.float32) + 0.5) * ry
        for h in [0, H // 2, H - 1]:
            for w in [0, W // 2, W - 1]:
                torch.testing.assert_close(
                    weq_bev[0, :, h, w], expected, atol=1e-4, rtol=1e-4,
                )

    def test_weq_monotonically_increasing(self, projector, ct_shape, device):
        H, D, W = ct_shape
        torch.manual_seed(42)
        spr = torch.rand((1, H, D, W), device=device) + 0.1
        angles = torch.zeros(1, device=device)
        iso = torch.zeros((1, 3), device=device)
        _, weq_bev = projector(spr, angles, iso)
        diffs = weq_bev[0, 1:] - weq_bev[0, :-1]
        assert (diffs > 0).all()

    def test_zero_spr_zero_weq(self, projector, ct_shape, device):
        H, D, W = ct_shape
        spr = torch.zeros((1, H, D, W), device=device)
        angles = torch.zeros(1, device=device)
        iso = torch.zeros((1, 3), device=device)
        _, weq_bev = projector(spr, angles, iso)
        assert (weq_bev == 0).all()

    def test_negative_spr_clamped(self, projector, ct_shape, device):
        H, D, W = ct_shape
        spr = -torch.ones((1, H, D, W), device=device)
        angles = torch.zeros(1, device=device)
        iso = torch.zeros((1, 3), device=device)
        _, weq_bev = projector(spr, angles, iso)
        assert (weq_bev == 0).all()

    def test_batch_different_angles(self, projector, device):
        """Batch with different angles produces consistent WEQ."""
        H, D, W = 16, 20, 24
        torch.manual_seed(99)
        spr = torch.rand((2, H, D, W), device=device) + 0.1
        angles = torch.tensor([0.0, 1.0], device=device)
        iso = torch.zeros((2, 3), device=device)
        spr_bev, weq_bev = projector(spr, angles, iso)
        assert spr_bev.shape == (2, D, H, W)
        assert weq_bev.shape == (2, D, H, W)
        diffs = weq_bev[:, 1:] - weq_bev[:, :-1]
        assert (diffs >= 0).all()

    def test_matches_radiological_depth_layer_forward_bev(
        self, projector, ct_shape, device
    ):
        """BevProjector at gantry=0 should match RadiologicalDepthLayer.forward_bev."""
        from pydose_rt.layers import RadiologicalDepthLayer
        from pydose_rt.data import MachineConfig

        H, D, W = ct_shape
        resolution = projector.resolution
        iso = (0.0, 0.0, 0.0)
        config = MachineConfig(
            preset="src/pydose_rt/data/machine_presets/test.json",
            head_scatter_amplitude=None,
            head_scatter_sigma=None,
            penumbra_fwhm=None,
            profile_corrections=None,
        )
        rdl = RadiologicalDepthLayer(
            config, resolution, ct_shape,
            gantry_angles=[0.0], iso_center=iso,
            device=device, dtype=torch.float32,
        )

        torch.manual_seed(42)
        ct = torch.rand((1, H, D, W), device=device, dtype=torch.float32) + 0.1
        _, weq_rdl = rdl.forward_bev(ct)

        angles_t = torch.zeros(1, device=device)
        iso_t = torch.zeros((1, 3), device=device)
        _, weq_bp = projector(ct, angles_t, iso_t)

        torch.testing.assert_close(weq_bp, weq_rdl, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize(
        "angle_rad,iso_mm",
        [
            (0.7854, (20.0, 15.0, -10.0)),
            (1.5708, (0.0, 25.0, 12.5)),
            (-0.5, (10.0, -5.0, 30.0)),
        ],
    )
    def test_matches_rdl_nonzero_angle_and_iso(
        self, projector, ct_shape, device, angle_rad, iso_mm
    ):
        """BevProjector must match RadiologicalDepthLayer.forward_bev with
        non-trivial gantry angle and isocenter."""
        from pydose_rt.layers import RadiologicalDepthLayer
        from pydose_rt.data import MachineConfig

        H, D, W = ct_shape
        resolution = projector.resolution
        config = MachineConfig(
            preset="src/pydose_rt/data/machine_presets/test.json",
            head_scatter_amplitude=None,
            head_scatter_sigma=None,
            penumbra_fwhm=None,
            profile_corrections=None,
        )
        rdl = RadiologicalDepthLayer(
            config, resolution, ct_shape,
            gantry_angles=[angle_rad], iso_center=iso_mm,
            device=device, dtype=torch.float32,
        )

        torch.manual_seed(42)
        ct = torch.rand((1, H, D, W), device=device, dtype=torch.float32) + 0.1
        den_rdl, weq_rdl = rdl.forward_bev(ct)

        angles_t = torch.tensor([angle_rad], device=device, dtype=torch.float32)
        iso_t = torch.tensor([list(iso_mm)], device=device, dtype=torch.float32)
        den_bp = projector.sample_bev(ct, angles_t, iso_t)
        _, weq_bp = projector(ct, angles_t, iso_t)

        torch.testing.assert_close(den_bp, den_rdl, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(weq_bp, weq_rdl, atol=1e-4, rtol=1e-4)
