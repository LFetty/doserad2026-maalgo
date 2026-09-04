from __future__ import annotations

import torch
from torch import nn

from training.common.repvgg_unet_corrector import RepVGGUNetCorrector, _GatedLatentDepthMixer
from training.common.separable_fan_grid_corrector import _AdaptiveNorm3d
from training.proton.train_dense_correction import _bev_deep_supervision_loss


def _inputs() -> dict[str, torch.Tensor]:
    shape = (2, 1, 7, 10, 14)
    return {
        "features": torch.randn(2, 8, 7, 10, 14),
        "dose_pb": torch.rand(shape),
        "valid_mask": torch.ones(shape, dtype=torch.bool),
        "fan_mask": torch.ones(2, 1, 1, 10, 14, dtype=torch.bool),
        "material_id": torch.randint(0, 86, shape),
        "energy": torch.tensor([100.0, 150.0]),
        "sigma_mm": torch.rand(2, 2),
    }


def _model(**kwargs) -> RepVGGUNetCorrector:
    return RepVGGUNetCorrector(
        input_dim=8,
        native_dim=4,
        stage_dims=(4, 8, 8),
        stage_blocks=(1, 1, 1, 1, 1),
        depth_kernel_size=3,
        available_energies=[100.0, 150.0],
        use_sigma_conditioning=True,
        use_repvgg=True,
        equalize_axis="w",
        equalize_factor=3,
        **kwargs,
    )


def test_repvgg_unet_keeps_depth_and_emits_lateral_deep_supervision() -> None:
    torch.manual_seed(10)
    model = _model()
    with torch.no_grad():
        model.head.weight.normal_()
        model.aux_equal_head.weight.normal_()
        model.aux_encoder_head.weight.normal_()
    model.train()
    inputs = _inputs()

    outputs = model(**inputs)

    assert outputs["dose_hat"].shape == (2, 1, 7, 10, 14)
    assert [item.shape for item in outputs["deep_supervision"]] == [
        (2, 1, 7, 10, 5),
        (2, 1, 7, 5, 3),
    ]
    target = torch.rand_like(outputs["dose_hat"])
    loss = outputs["dose_hat"].mean() + _bev_deep_supervision_loss(
        outputs["deep_supervision"],
        target,
        inputs["valid_mask"],
    )
    loss.backward()
    assert model.aux_equal_head.weight.grad is not None
    assert model.aux_encoder_head.weight.grad is not None


def test_repvgg_unet_optional_extra_stage_keeps_depth_and_adds_supervision() -> None:
    torch.manual_seed(12)
    model = _model(extra_stage_dim=12, extra_stage_blocks=2)
    with torch.no_grad():
        model.head.weight.normal_()
        model.aux_equal_head.weight.normal_()
        model.aux_encoder_head.weight.normal_()
        assert model.aux_bottleneck_head is not None
        model.aux_bottleneck_head.weight.normal_()
    model.train()
    inputs = _inputs()

    outputs = model(**inputs)

    assert outputs["dose_hat"].shape == (2, 1, 7, 10, 14)
    assert [item.shape for item in outputs["deep_supervision"]] == [
        (2, 1, 7, 10, 5),
        (2, 1, 7, 5, 3),
        (2, 1, 7, 3, 2),
    ]
    target = torch.rand_like(outputs["dose_hat"])
    loss = outputs["dose_hat"].mean() + _bev_deep_supervision_loss(
        outputs["deep_supervision"],
        target,
        inputs["valid_mask"],
    )
    loss.backward()
    assert model.aux_bottleneck_head.weight.grad is not None


def test_repvgg_unet_fold_preserves_inference_output() -> None:
    torch.manual_seed(11)
    model = _model()
    with torch.no_grad():
        model.head.weight.normal_()
        model.head.bias.normal_()
        for module in model.modules():
            if hasattr(module, "depth_1x1"):
                module.depth_1x1.weight.normal_()
                module.lat_1x1.weight.normal_()
    model.eval()
    inputs = _inputs()

    before = model(**inputs)["dose_hat"]
    model.fuse_repvgg()
    after = model(**inputs)["dose_hat"]

    torch.testing.assert_close(after, before, atol=1e-5, rtol=1e-4)


def test_repvgg_unet_supports_continuous_energy_conditioning() -> None:
    inputs = _inputs()
    for mode in ("none", "scalar", "fourier"):
        model = _model(energy_conditioning=mode)
        outputs = model(**inputs)
        assert outputs["dose_hat"].shape == inputs["dose_pb"].shape


def test_repvgg_unet_instance_norm_reaches_internal_blocks() -> None:
    model = _model(norm_kind="instance")
    assert isinstance(model.norm, nn.InstanceNorm3d)
    assert isinstance(model.native_stage.blocks[0].pre[0], nn.InstanceNorm3d)
    assert isinstance(model.native_stage.blocks[0].mix[0], nn.InstanceNorm3d)


def test_repvgg_unet_adagn_reaches_every_internal_block() -> None:
    model = _model(conditioning_injection="adagn")
    adaptive_norms = [module for module in model.modules() if isinstance(module, _AdaptiveNorm3d)]

    assert isinstance(model.norm, _AdaptiveNorm3d)
    assert len(adaptive_norms) == 2 * 6 + 1
    for stage in (
        model.native_stage,
        model.equal_stage,
        model.encoder_stage,
        model.bottleneck_stage,
        model.decoder_encoder_stage,
        model.decoder_equal_stage,
    ):
        for block in stage.blocks:
            assert isinstance(block.pre[0], _AdaptiveNorm3d)
            assert isinstance(block.mix[0], _AdaptiveNorm3d)
    for norm in adaptive_norms:
        assert torch.count_nonzero(norm.modulation.weight) == 0
        assert torch.count_nonzero(norm.modulation.bias) == 0


def test_repvgg_unet_adagn_modulates_blocks_from_energy() -> None:
    torch.manual_seed(13)
    model = _model(conditioning_injection="adagn")
    with torch.no_grad():
        model.head.weight.normal_()
        model.native_stage.blocks[0].pre[0].modulation.weight.normal_()
    model.eval()
    inputs = _inputs()
    inputs["sigma_mm"] = torch.zeros_like(inputs["sigma_mm"])
    inputs["energy"] = torch.tensor([100.0, 100.0])
    low = model(**inputs)["dose_hat"]
    inputs["energy"] = torch.tensor([150.0, 150.0])
    high = model(**inputs)["dose_hat"]

    assert not torch.equal(low, high)


def test_repvgg_unet_adagn_modulations_receive_gradients() -> None:
    torch.manual_seed(14)
    model = _model(conditioning_injection="adagn")
    with torch.no_grad():
        model.head.weight.normal_()
        model.aux_equal_head.weight.normal_()
        model.aux_encoder_head.weight.normal_()
    model.train()
    inputs = _inputs()

    outputs = model(**inputs)
    loss = outputs["dose_hat"].mean() + sum(prediction.mean() for prediction in outputs["deep_supervision"])
    loss.backward()

    adaptive_norms = [module for module in model.modules() if isinstance(module, _AdaptiveNorm3d)]
    assert adaptive_norms
    for norm in adaptive_norms:
        assert torch.count_nonzero(norm.modulation.weight.grad) > 0
        assert torch.count_nonzero(norm.modulation.bias.grad) > 0


def test_repvgg_unet_latent_depth_mixer_is_extra_stage_identity_at_init() -> None:
    torch.manual_seed(15)
    model = _model(
        extra_stage_dim=12,
        extra_stage_blocks=2,
        latent_depth_mixer=True,
        latent_depth_mixer_kernel_size=5,
        latent_depth_mixer_dilations=(1, 2),
    )
    assert isinstance(model.latent_depth_mixer, _GatedLatentDepthMixer)
    with torch.no_grad():
        model.head.weight.normal_()
    model.eval()
    inputs = _inputs()

    with_mixer = model(**inputs)["dose_hat"]
    mixer = model.latent_depth_mixer
    model.latent_depth_mixer = None
    without_mixer = model(**inputs)["dose_hat"]
    model.latent_depth_mixer = mixer

    torch.testing.assert_close(with_mixer, without_mixer, atol=0.0, rtol=0.0)


def test_repvgg_unet_latent_depth_mixer_receives_gradients_when_opened() -> None:
    torch.manual_seed(16)
    model = _model(
        extra_stage_dim=12,
        extra_stage_blocks=2,
        latent_depth_mixer=True,
        latent_depth_mixer_kernel_size=5,
        latent_depth_mixer_dilations=(1, 2),
    )
    assert isinstance(model.latent_depth_mixer, _GatedLatentDepthMixer)
    with torch.no_grad():
        model.head.weight.normal_()
        model.latent_depth_mixer.out.weight.normal_()
    model.train()
    inputs = _inputs()

    loss = model(**inputs)["dose_hat"].mean()
    loss.backward()

    assert torch.count_nonzero(model.latent_depth_mixer.branches[0].weight.grad) > 0
    assert torch.count_nonzero(model.latent_depth_mixer.gate.weight.grad) > 0
