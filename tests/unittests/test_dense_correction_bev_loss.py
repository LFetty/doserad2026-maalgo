from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from pydose_rt.data.ion_beam import IonSpotBeam, IonSpotBeamSequence
from pydose_rt.data.machine_config import MachineConfig
from pydose_rt.layers.BeamRotationLayer import BeamRotationLayer
from pydose_rt.layers.RadiologicalDepthLayer import RadiologicalDepthLayer
from training.proton.train_dense_correction import (
    _bev_deep_supervision_loss,
    _build_bev_features,
    _compute_lattice_edep_bev,
    _compute_lattice_edep_bev_batch,
    _crop_bev_volume,
    _engine_sample_bev,
    _loss,
    _pad_model_depth,
    _reference_patient_tensor,
    _trim_model_outputs,
)
from training.common.separable_fan_grid_corrector import SeparableFanGridConvCorrector
from training.proton.hooks import ProtonDenseBevCorrectionHook


def test_crop_bev_volume_pads_outside_source() -> None:
    bev = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(1, 3, 4, 5)

    cropped = _crop_bev_volume(bev, crop_center_hw=(0.0, 2.0), crop_h=2, crop_w=2)

    assert cropped.shape == (1, 3, 4, 4)
    assert torch.equal(cropped[:, :, 2:, :], bev[:, :, :2, :4])
    assert not cropped[:, :, :2, :].count_nonzero()


def test_model_depth_padding_and_output_trimming() -> None:
    x = torch.ones(2, 3, 5, 4, 6)

    padded, original_depth = _pad_model_depth(x, 8)

    assert padded.shape == (2, 3, 8, 4, 6)
    assert original_depth == 5
    assert torch.equal(padded[:, :, :5], x)
    assert not padded[:, :, 5:].count_nonzero()

    outputs = {
        "dose_hat": padded[:, :1],
        "residual": padded[:, :1],
        "deep_supervision": (padded[:, :1],),
        "attn_maps": [padded[:, :1]],
    }
    trimmed = _trim_model_outputs(outputs, original_depth)
    assert trimmed["dose_hat"].shape[2] == 5
    assert trimmed["residual"].shape[2] == 5
    assert trimmed["deep_supervision"][0].shape[2] == 5
    assert trimmed["attn_maps"][0].shape[2] == 5


def test_bev_deep_supervision_uses_dose_epsilon_for_low_dose_targets() -> None:
    prediction = torch.full((1, 1, 2, 2, 2), 1e-3)
    target = torch.zeros_like(prediction)
    valid_mask = torch.ones_like(prediction, dtype=torch.bool)

    loss = _bev_deep_supervision_loss((prediction,), target, valid_mask, eps=1e-3)

    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_reference_patient_tensor_expands_bbox_payload() -> None:
    ref_arr = np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2)

    full = _reference_patient_tensor(
        ref_arr,
        (1, 3, 2, 4, 3, 5),
        (4, 5, 6),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert full.shape == (4, 5, 6)
    assert torch.equal(full[1:3, 2:4, 3:5], torch.from_numpy(ref_arr))
    assert full.count_nonzero() == np.count_nonzero(ref_arr)


def test_bev_and_patient_l1_match_at_zero_rotation() -> None:
    shape = (9, 11, 13)
    resolution = (1.0, 1.0, 1.0)
    iso_center = (4.0, 5.0, 6.0)
    config = MachineConfig(tpr_20_10=0.7, number_of_leaf_pairs=40)
    rad_depth_layer = RadiologicalDepthLayer(
        config, resolution, shape, [0.0], iso_center,
        depth_origin="entry", device="cpu",
    )
    rotation_layer = BeamRotationLayer(
        config, shape, iso_center, resolution, torch.tensor([0.0]),
        depth_origin="entry", device="cpu",
    )
    engine = SimpleNamespace(rad_depth_layer=rad_depth_layer)
    torch.manual_seed(3)
    ref_bev = torch.rand(1, shape[1], shape[0], shape[2])
    pred_bev = ref_bev + 0.05 * torch.randn_like(ref_bev)
    ref_patient = rotation_layer(ref_bev.unsqueeze(1)).sum(dim=1)
    pred_patient = rotation_layer(pred_bev.unsqueeze(1)).sum(dim=1)

    ref_bev_resampled = _engine_sample_bev(engine, ref_patient)
    patient_l1 = (pred_patient - ref_patient).abs().mean()
    bev_l1 = (pred_bev - ref_bev_resampled).abs().mean()

    torch.testing.assert_close(bev_l1, patient_l1, atol=1e-6, rtol=1e-6)


def test_lattice_helper_delegates_to_split_kernel() -> None:
    beam = IonSpotBeam.create(
        gantry_angle_deg=0.0,
        spot_positions_mm=torch.tensor([[0.0, 0.0]]),
        spot_weights=torch.tensor([1.0]),
        spot_layer_index=torch.tensor([0]),
        layer_energies_mev=torch.tensor([100.0]),
        layer_sigmas_mm=torch.tensor([[2.0, 2.0]]),
        field_size=(4, 6),
        iso_center=(2.0, 3.0, 2.0),
        requires_grad=False,
    )
    seq = IonSpotBeamSequence.from_beams([beam])

    class _Engine:
        dose_grid_spacing = (1.0, 1.0, 1.0)

        def compute_layer_edep(self, *args, **kwargs):
            assert kwargs["splitting_mode"] == "split"
            assert kwargs["n_per_dim"] == 9
            assert args[6].shape == (1, 1, 5, 4, 6)
            return torch.ones(5, 4, 6)

    out = _compute_lattice_edep_bev(
        _Engine(), seq, torch.ones(1, 5, 4, 6), (2.0, 3.0), torch.zeros(1),
    )

    assert out.shape == (1, 5, 4, 6)


def test_lattice_batch_helper_delegates_each_beam_to_split_kernel() -> None:
    beams = [
        IonSpotBeam.create(
            gantry_angle_deg=float(idx * 5),
            spot_positions_mm=torch.tensor([[0.0, 0.0]]),
            spot_weights=torch.tensor([1.0]),
            spot_layer_index=torch.tensor([0]),
            layer_energies_mev=torch.tensor([100.0 + idx]),
            layer_sigmas_mm=torch.tensor([[2.0, 2.0]]),
            field_size=(4, 6),
            iso_center=(2.0, 3.0, 2.0),
            requires_grad=False,
        )
        for idx in range(2)
    ]
    seq = IonSpotBeamSequence.from_beams(beams)

    class _Engine:
        dose_grid_spacing = (1.0, 1.0, 1.0)

        def compute_layer_edep(self, *args, **kwargs):
            g_idx = int(args[0])
            assert kwargs["splitting_mode"] == "split"
            assert kwargs["n_per_dim"] == 9
            assert args[6].shape == (1, 2, 5, 4, 6)
            return torch.full((5, 4, 6), float(g_idx + 1))

    out = _compute_lattice_edep_bev_batch(
        _Engine(),
        seq,
        torch.ones(2, 5, 4, 6),
        [(2.0, 3.0), (2.0, 3.0)],
        torch.zeros(2),
    )

    assert out.shape == (2, 5, 4, 6)
    assert torch.equal(out[:, 0, 0, 0], torch.tensor([1.0, 2.0]))


def test_loss_reports_high_dose_mae_separately() -> None:
    pred = torch.zeros(1, 1, 2)
    ref = torch.tensor([[[1.0, 0.01]]])
    args = SimpleNamespace(
        w_dose=1.0,
        w_energy=0.0,
        w_profile=0.0,
        w_idd=0.0,
        w_peak=0.0,
        depth_bin_mm=1.0,
        huber_delta=0.05,
        peak_tau_frac=0.05,
        peak_scale_mm=2.0,
    )

    _, terms = _loss(pred, ref, torch.zeros_like(ref), args)

    assert terms["mae_pct"] == 50.5
    assert terms["mae_high10_pct"] == 100.0


def test_dense_hook_restores_anisotropic_crop_from_checkpoint(tmp_path) -> None:
    config = {
        "model": {
            "kind": "separable_fan_conv",
            "hidden_dim": 4,
            "num_layers": 1,
            "depth_kernel_size": 3,
            "material_embedding_dim": 0,
        },
    }
    model = SeparableFanGridConvCorrector.from_config(8, config)
    checkpoint_path = tmp_path / "dense.pt"
    torch.save(
        {
            "config": config,
            "fan_input_dim": 8,
            "model_state": model.state_dict(),
            "args": {"bev_crop_hw": 64, "bev_crop_h": 13, "bev_crop_w": 37},
        },
        checkpoint_path,
    )

    hook = ProtonDenseBevCorrectionHook.from_checkpoint(checkpoint_path, device="cpu")

    assert hook.bev_crop_hw == 64
    assert hook.bev_crop_h == 13
    assert hook.bev_crop_w == 37


def test_dense_hook_features_match_training_depth_context() -> None:
    config = {
        "model": {
            "kind": "separable_fan_conv",
            "hidden_dim": 4,
            "num_layers": 1,
            "depth_kernel_size": 3,
            "material_embedding_dim": 0,
        },
    }
    model = SeparableFanGridConvCorrector.from_config(8, config)
    hook = ProtonDenseBevCorrectionHook(model, config, bev_crop_h=2, bev_crop_w=3)
    torch.manual_seed(5)
    spr = torch.rand(1, 7, 6, 8)
    weq = torch.rand(1, 7, 6, 8)
    dose = torch.rand(1, 7, 6, 8)
    dose[:, :2] = 0.0
    material = torch.zeros_like(dose, dtype=torch.long)
    center = (3.0, 4.0)

    expected = _build_bev_features(
        spr, weq, dose, material,
        bev_crop_hw=64, crop_center_hw=center, bev_crop_h=2, bev_crop_w=3,
    )
    actual = hook._build_bev_features(spr, weq, dose, material, center)

    for expected_item, actual_item in zip(expected, actual[:5], strict=True):
        torch.testing.assert_close(actual_item, expected_item)
    assert actual[5]["d_src"] == slice(0, 7)
