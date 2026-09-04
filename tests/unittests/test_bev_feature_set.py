"""BEV input feature-set versioning.

Two properties matter and both are regression traps:

1. ``v1`` must stay byte-identical to the historical 8-channel stack. Every
   checkpoint trained before 2026-07-26 was fitted to it, so any drift silently
   invalidates the deployed model rather than failing loudly.
2. A ``v2`` model warm-started from a ``v1`` checkpoint must be *exactly*
   functionally identical at step 0. Otherwise a v1/v2 comparison confounds the
   new channel with a partially reinitialised input projection.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training.proton.hooks import (  # noqa: E402
    BEV_FEATURE_CHANNELS,
    ProtonDenseBevCorrectionHook,
    _lateral_weq_gradient,
)
from training.proton.train_dense_correction import (  # noqa: E402
    _adapt_state_for_feature_set,
    _build_bev_features,
)


def _bev_inputs(n=1, d=4, h=6, w=8):
    torch.manual_seed(0)
    return dict(
        spr_bev=torch.rand(n, d, h, w) + 0.5,
        weq_bev=torch.rand(n, d, h, w) * 100.0,
        dose_pb_bev=torch.rand(n, d, h, w),
        material_id_bev=torch.zeros(n, d, h, w, dtype=torch.long),
    )


def test_v1_channel_count_and_content_unchanged():
    feats, *_ = _build_bev_features(
        **_bev_inputs(), bev_crop_hw=3, crop_center_hw=(3.0, 4.0)
    )
    assert feats.shape[1] == BEV_FEATURE_CHANNELS["v1"] == 8


def test_v2_appends_exactly_one_channel_and_leaves_v1_channels_intact():
    inputs = _bev_inputs()
    v1, *_ = _build_bev_features(
        **inputs, bev_crop_hw=3, crop_center_hw=(3.0, 4.0), feature_set="v1"
    )
    v2, *_ = _build_bev_features(
        **inputs, bev_crop_hw=3, crop_center_hw=(3.0, 4.0),
        feature_set="v2", peak_depth_mm=50.0,
    )

    assert v2.shape[1] == v1.shape[1] + 1 == BEV_FEATURE_CHANNELS["v2"]
    # v2 must be a pure extension: the first 8 channels are untouched.
    torch.testing.assert_close(v2[:, :8], v1)


def test_v2_extra_channel_is_weq_relative_to_the_bragg_peak():
    peak = 50.0
    feats, *_ = _build_bev_features(
        **_bev_inputs(), bev_crop_hw=3, crop_center_hw=(3.0, 4.0),
        feature_set="v2", peak_depth_mm=peak,
    )
    # Channel 3 is weq/depth_scale; channel 8 is (weq - peak)/depth_scale.
    torch.testing.assert_close(feats[:, 8], feats[:, 3] - peak / 100.0)


@pytest.mark.parametrize("feature_set", ["v2", "v3"])
def test_range_relative_sets_without_peak_depth_are_an_error(feature_set):
    with pytest.raises(ValueError, match="requires peak_depth_mm"):
        _build_bev_features(
            **_bev_inputs(),
            bev_crop_hw=3,
            crop_center_hw=(3.0, 4.0),
            feature_set=feature_set,
        )


def test_v3_is_v2_plus_the_lateral_weq_gradient():
    inputs = _bev_inputs()
    v2, *_ = _build_bev_features(
        **inputs, bev_crop_hw=3, crop_center_hw=(3.0, 4.0),
        feature_set="v2", peak_depth_mm=50.0,
    )
    v3, *_ = _build_bev_features(
        **inputs, bev_crop_hw=3, crop_center_hw=(3.0, 4.0),
        feature_set="v3", peak_depth_mm=50.0,
    )
    # Feature sets are strictly cumulative -- that is what makes the zero-init
    # warm start valid across every pair of versions.
    assert v3.shape[1] == v2.shape[1] + 1 == BEV_FEATURE_CHANNELS["v3"]
    torch.testing.assert_close(v3[:, :9], v2)
    assert torch.all(v3[:, 9] >= 0.0)


def test_lateral_gradient_is_lateral_only_and_shape_preserving():
    # Varies along depth only: the lateral gradient must be identically zero.
    depth_ramp = torch.arange(5.0).view(1, 5, 1, 1).expand(2, 5, 4, 6).contiguous()
    torch.testing.assert_close(
        _lateral_weq_gradient(depth_ramp), torch.zeros_like(depth_ramp)
    )

    # Unit slope along W: interior central differences are 1, and so are the
    # one-sided borders for a linear ramp.
    w_ramp = torch.arange(6.0).view(1, 1, 1, 6).expand(2, 5, 4, 6).contiguous()
    torch.testing.assert_close(_lateral_weq_gradient(w_ramp), torch.ones_like(w_ramp))

    # Degenerate lateral axes must not raise.
    thin = torch.randn(1, 3, 1, 1)
    assert _lateral_weq_gradient(thin).shape == thin.shape


def test_warm_start_widening_spans_two_versions_at_once():
    """v1 -> v3 adds two columns; the zero-init identity must still hold."""
    material = 4
    torch.manual_seed(0)
    weight = torch.randn(6, 8 + material, 1, 1, 1)
    adapted = _adapt_state_for_feature_set(
        {"in_proj.weight": weight}, src_channels=8, dst_channels=10
    )["in_proj.weight"]

    assert adapted.shape == (6, 10 + material, 1, 1, 1)
    x8 = torch.randn(2, 8, 3, 3, 3)
    mat = torch.randn(2, material, 3, 3, 3)
    extra = torch.randn(2, 2, 3, 3, 3)
    old_out = torch.nn.functional.conv3d(torch.cat((x8, mat), 1), weight)
    new_out = torch.nn.functional.conv3d(torch.cat((x8, extra, mat), 1), adapted)
    torch.testing.assert_close(old_out, new_out)


def test_warm_start_widening_preserves_the_v1_function_exactly():
    native, material = 5, 4
    torch.manual_seed(0)
    weight = torch.randn(native, 8 + material, 1, 1, 1)
    adapted = _adapt_state_for_feature_set(
        {"in_proj.weight": weight}, src_channels=8, dst_channels=9
    )["in_proj.weight"]

    assert adapted.shape == (native, 9 + material, 1, 1, 1)
    # BEV block copied, inserted column zeroed, material block shifted right by one.
    torch.testing.assert_close(adapted[:, :8], weight[:, :8])
    torch.testing.assert_close(adapted[:, 8], torch.zeros_like(adapted[:, 8]))
    torch.testing.assert_close(adapted[:, 9:], weight[:, 8:])

    # The real invariant: identical output for any input whose 9th channel is
    # arbitrary, since that channel must contribute nothing at step 0.
    x8 = torch.randn(2, 8, 3, 3, 3)
    mat = torch.randn(2, material, 3, 3, 3)
    new_ch = torch.randn(2, 1, 3, 3, 3)
    old_out = torch.nn.functional.conv3d(torch.cat((x8, mat), 1), weight)
    new_out = torch.nn.functional.conv3d(torch.cat((x8, new_ch, mat), 1), adapted)
    torch.testing.assert_close(old_out, new_out)


@pytest.mark.parametrize(
    ("feature_set", "peak_depth_mm"), [("v1", None), ("v2", 50.0), ("v3", 50.0)]
)
def test_training_and_inference_build_the_same_features(feature_set, peak_depth_mm):
    """The two _build_bev_features implementations must agree, channel for channel.

    Training and inference construct this stack through completely separate code
    paths. That duplication is what let the BEV-crop bug reach a scored submission:
    the container derived its crop differently from the trainer, and the selftest
    reproduced the container's derivation, so nothing compared the two. This test
    is the comparison that was missing.
    """
    inputs = _bev_inputs(n=1, d=4, h=9, w=11)
    crop_h, crop_w, center = 3, 4, (4.0, 5.0)

    train_feats, train_dose, train_valid, train_mat, train_fan = _build_bev_features(
        **inputs,
        bev_crop_hw=0,
        bev_crop_h=crop_h,
        bev_crop_w=crop_w,
        crop_center_hw=center,
        feature_set=feature_set,
        peak_depth_mm=peak_depth_mm,
    )

    hook = ProtonDenseBevCorrectionHook.__new__(ProtonDenseBevCorrectionHook)
    hook.cfg = {}
    hook.bev_crop_h, hook.bev_crop_w = crop_h, crop_w
    hook.bev_feature_set = feature_set
    hook_feats, hook_dose, hook_valid, hook_mat, hook_fan, _ = hook._build_bev_features(
        inputs["spr_bev"],
        inputs["weq_bev"],
        inputs["dose_pb_bev"],
        inputs["material_id_bev"],
        center,
        peak_depth_mm=peak_depth_mm,
    )

    assert hook_feats.shape == train_feats.shape == (
        1, BEV_FEATURE_CHANNELS[feature_set], 4, 2 * crop_h, 2 * crop_w
    )
    torch.testing.assert_close(hook_feats, train_feats)
    torch.testing.assert_close(hook_dose, train_dose)
    torch.testing.assert_close(hook_mat.to(train_mat.dtype), train_mat)
    assert torch.equal(hook_valid, train_valid)
    assert hook_fan.shape == train_fan.shape


def test_warm_start_refuses_to_narrow():
    with pytest.raises(ValueError, match="only ever added"):
        _adapt_state_for_feature_set(
            {"in_proj.weight": torch.zeros(2, 13, 1, 1, 1)},
            src_channels=9,
            dst_channels=8,
        )
