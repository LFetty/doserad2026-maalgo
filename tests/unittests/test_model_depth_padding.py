"""Is --model-depth padding free? No -- it changes what the model computes.

Padding BEV depth to a fixed 544 is what lets torch.compile see static shapes, and in the
training logs padded runs sit at ~0.5 s/step against ~1.0 for native depth. That is only
free speed if the padded forward is the SAME FUNCTION as the native one on the region we
keep. It is not.

The mechanism is normalisation: this net uses GroupNorm, whose statistics are computed
over (C/G, D, H, W) per sample. Distal zero padding enlarges D, so mean and variance are
taken over a volume that is part signal and part zeros, which rescales every real voxel.
Convolutions are fine -- zero padding at the D boundary is what an unpadded conv already
sees -- but the norm is not.

Consequence, and the reason these tests exist: a checkpoint trained with --model-depth
must be RUN with the same --model-depth. The inference path (hooks.py, container/
inference.py) implements no padding at all, so shipping a padded-trained checkpoint today
would silently be a train/inference mismatch -- the same class of bug as the BEV-crop one
that reached a scored submission.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training.common.repvgg_unet_corrector import RepVGGUNetCorrector  # noqa: E402
from training.proton.train_dense_correction import (  # noqa: E402
    _pad_model_depth,
    _trim_model_outputs,
)

# The shipped 887k geometry.
_CFG = dict(
    unet_native_dim=64,
    unet_stage_dims=[96, 128, 192],
    unet_stage_blocks=[2, 2, 2, 4, 2],
    unet_extra_stage_dim=256,
    unet_extra_stage_blocks=4,
    unet_equalize_axis="w",
    unet_equalize_factor=3,
    unet_energy_conditioning="embedding",
    unet_conditioning_injection="entrance",
    material_embedding_dim=4,
    use_sigma_conditioning=True,
    residual_mode="additive",
    additive_scale_frac=0.25,
    dropout=0.0,
    use_repvgg=True,
    depth_kernel_size=11,
    mix_ratio=0.5,
)

_D_NATIVE = 40
_D_PADDED = 56
_H, _W = 13, 37


def _build(norm_kind: str = "group", *, live_residual: bool = True):
    torch.manual_seed(0)
    model = RepVGGUNetCorrector.from_config(
        8, dict(_CFG, unet_norm=norm_kind), available_energies=[100.0, 150.0]
    ).eval()
    if live_residual:
        # The residual head is zero-init, so a freshly built model is EXACTLY the
        # identity and dose_hat == dose bit for bit. Comparing padded against native on
        # an identity model measures nothing; perturb the weights so the residual --
        # the model's entire contribution -- is actually nonzero.
        torch.manual_seed(7)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.05 * torch.randn_like(p))
    return model


def _inputs(depth: int):
    g = torch.Generator().manual_seed(1)
    return dict(
        features=torch.randn(1, 8, depth, _H, _W, generator=g),
        dose=torch.rand(1, 1, depth, _H, _W, generator=g),
        material_id=torch.zeros(1, 1, depth, _H, _W, dtype=torch.long),
        energy=torch.tensor([100.0]),
        sigma_mm=torch.tensor([5.0]),
    )


def _forward(model, inp, target_depth: int):
    feats, orig = _pad_model_depth(inp["features"], target_depth)
    dose, _ = _pad_model_depth(inp["dose"], target_depth)
    mat, _ = _pad_model_depth(inp["material_id"], target_depth)
    with torch.no_grad():
        out = model(
            feats, dose, None, None,
            material_id=mat, energy=inp["energy"], sigma_mm=inp["sigma_mm"],
        )
    return _trim_model_outputs(out, orig)["dose_hat"]


def test_trimmed_output_has_native_shape():
    """Whatever it does numerically, the trimmed output must have the native shape."""
    model = _build()
    assert _forward(model, _inputs(_D_NATIVE), _D_PADDED).shape[2] == _D_NATIVE


def test_freshly_built_model_is_the_identity():
    """Guards the trap that made the first version of this test vacuous: with a zero-init
    residual head, dose_hat == dose exactly and padding trivially looks free."""
    model = _build(live_residual=False)
    inp = _inputs(_D_NATIVE)
    torch.testing.assert_close(_forward(model, inp, 0), inp["dose"])


def test_padding_changes_the_output_by_a_large_fraction_of_the_residual():
    """Measured: max|native - padded| is ~26% of the residual's own magnitude.

    If this ever starts failing, padding became numerically free and --model-depth could
    be enabled at inference without re-validating the metrics. Until then it cannot.
    """
    model = _build()
    inp = _inputs(_D_NATIVE)
    native = _forward(model, inp, 0)
    padded = _forward(model, inp, _D_PADDED)
    residual = (native - inp["dose"]).abs().max()
    assert residual > 1e-3, "perturbation failed to make the residual live"
    rel = ((native - padded).abs().max() / residual).item()
    assert rel > 0.01, f"expected padding to shift the output materially, got {rel:.3%}"


def test_masked_group_norm_makes_padding_exact():
    """Masked GN is the fix: statistics over the valid region, and the pad re-zeroed.

    Measured 1.2e-07 (float32 roundoff) against plain GroupNorm's 26% of the residual.
    Note this buys correctness, not speed -- benchmarked, padded+masked+static compile
    came out 2.4% SLOWER than simply compiling the native path with dynamic shapes.
    """
    from training.common.masked_norm import convert_group_norms, set_valid_depth

    model = _build()
    inp = _inputs(_D_NATIVE)
    native = _forward(model, inp, 0)

    assert convert_group_norms(model) == 15
    set_valid_depth(model, _D_NATIVE)
    padded = _forward(model, inp, _D_PADDED)

    residual = (native - inp["dose"]).abs().max()
    rel = ((native - padded).abs().max() / residual).item()
    assert rel < 1e-5, f"masked GN should make padding exact, got {rel:.3%} of residual"


def test_masked_group_norm_without_valid_depth_is_plain_group_norm():
    """Converting a model must be a no-op until padding is actually used."""
    from training.common.masked_norm import convert_group_norms, set_valid_depth

    model = _build()
    inp = _inputs(_D_NATIVE)
    native = _forward(model, inp, 0)
    convert_group_norms(model)
    set_valid_depth(model, None)
    torch.testing.assert_close(_forward(model, inp, 0), native, atol=1e-6, rtol=1e-5)


def test_inference_path_has_no_depth_padding():
    """Pins the mismatch risk: the trainer pads, the inference hook does not, and the
    checkpoint does not record which was used. A padded-trained model must not be shipped
    until the hook can reproduce the same depth context."""
    hooks_src = (ROOT / "training" / "proton" / "hooks.py").read_text(encoding="utf-8")
    assert "_pad_model_depth" not in hooks_src
    assert "model_depth" not in hooks_src
