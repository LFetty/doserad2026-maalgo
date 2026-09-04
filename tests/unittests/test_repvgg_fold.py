from __future__ import annotations

import torch

from training.common.separable_fan_grid_corrector import SeparableFanGridConvCorrector


def _inputs() -> dict[str, torch.Tensor]:
    shape = (2, 1, 7, 5, 5)
    return {
        "features": torch.randn(2, 8, 7, 5, 5),
        "dose_pb": torch.rand(shape),
        "valid_mask": torch.rand(shape) > 0.1,
        "fan_mask": torch.rand(shape) > 0.1,
        "material_id": torch.randint(0, 86, shape),
        "energy": torch.tensor([100.0, 150.0]),
        "sigma_mm": torch.rand(2, 2),
    }


def test_repvgg_fold_preserves_output() -> None:
    torch.manual_seed(1234)
    model = SeparableFanGridConvCorrector(
        input_dim=8,
        hidden_dim=8,
        num_layers=2,
        depth_kernel_size=5,
        available_energies=[100.0, 150.0],
        use_sigma_conditioning=True,
        use_repvgg=True,
    )
    with torch.no_grad():
        model.head.weight.normal_()
        model.head.bias.normal_()
        for block in model.blocks:
            assert block.depth_1x1.weight.count_nonzero().item() == 0
            assert block.lat_1x1.weight.count_nonzero().item() == 0
            block.depth_1x1.weight.normal_()
            block.lat_1x1.weight.normal_()
    model.eval()
    inputs = _inputs()

    before = model(**inputs)["dose_hat"]
    model.fuse_repvgg()
    after = model(**inputs)["dose_hat"]

    assert torch.allclose(before, after, atol=1e-5, rtol=1e-4)
    for block in model.blocks:
        assert block.reparam_deployed
        assert not hasattr(block, "depth_1x1")
        assert not hasattr(block, "lat_1x1")


def test_repvgg_disabled_preserves_default_model() -> None:
    torch.manual_seed(5678)
    default_model = SeparableFanGridConvCorrector(input_dim=8, hidden_dim=8, num_layers=2)
    torch.manual_seed(5678)
    disabled_model = SeparableFanGridConvCorrector(input_dim=8, hidden_dim=8, num_layers=2, use_repvgg=False)
    with torch.no_grad():
        default_model.head.weight.normal_()
        default_model.head.bias.normal_()
        disabled_model.head.load_state_dict(default_model.head.state_dict())
    default_model.eval()
    disabled_model.eval()
    inputs = _inputs()

    default = default_model(**inputs)["dose_hat"]
    disabled = disabled_model(**inputs)["dose_hat"]

    assert torch.equal(default, disabled)
