from __future__ import annotations

import pytest
import torch

from training.proton.train_dense_correction import (
    _FEATURE_H_OFFSET_CH,
    _FEATURE_W_OFFSET_CH,
    _d4_apply,
    _d4_apply_features,
)

# The repvgg_unet symmetry set: identity, H-flip, 180-degree, and (H-flip + 180) = W-flip.
# No 90-degree rotations, so the H/W axes are never swapped.
_REPVGG_SYMMETRIES = ((False, 0), (False, 2), (True, 0), (True, 2))


def _canonical_features() -> torch.Tensor:
    """An 8-channel BEV feature tensor whose signed coordinate channels follow the
    canonical convention: the h-offset increases along H (dim 3), the w-offset
    increases along W (dim 4). Symmetric ramps (even width) so flip == negate."""
    n, c, d, h, w = 1, 8, 2, 4, 4
    x = torch.zeros(n, c, d, h, w)
    h_ramp = torch.arange(h, dtype=torch.float32) - (h - 1) / 2.0  # [-1.5,-0.5,0.5,1.5]
    w_ramp = torch.arange(w, dtype=torch.float32) - (w - 1) / 2.0
    x[:, _FEATURE_H_OFFSET_CH] = h_ramp.view(1, 1, h, 1)
    x[:, _FEATURE_W_OFFSET_CH] = w_ramp.view(1, 1, 1, w)
    # A physical-field channel (e.g. dose) with arbitrary asymmetric content.
    x[:, 7] = torch.arange(h * w, dtype=torch.float32).view(1, 1, h, w)
    return x


@pytest.mark.parametrize("flip,k_rot", _REPVGG_SYMMETRIES)
def test_coordinate_channels_stay_canonical(flip: bool, k_rot: int) -> None:
    out = _d4_apply_features(_canonical_features(), flip, k_rot)
    h_off = out[:, _FEATURE_H_OFFSET_CH]
    w_off = out[:, _FEATURE_W_OFFSET_CH]
    # h-offset must still increase along H (dim 3); w-offset along W (dim 4).
    assert torch.all(h_off.diff(dim=2) > 0), f"h-offset not canonical for flip={flip} k={k_rot}"
    assert torch.all(w_off.diff(dim=3) > 0), f"w-offset not canonical for flip={flip} k={k_rot}"


def test_plain_apply_breaks_coordinate_convention() -> None:
    """Guard: the naive field-flip (the original bug) inverts the coordinate ramp,
    proving the test above is actually exercising the fix."""
    out = _d4_apply(_canonical_features(), True, 0)  # H-flip as plain fields
    h_off = out[:, _FEATURE_H_OFFSET_CH]
    assert torch.all(h_off.diff(dim=2) < 0), "expected naive flip to reverse the h-offset ramp"


@pytest.mark.parametrize("flip,k_rot", _REPVGG_SYMMETRIES)
def test_physical_field_channel_matches_plain_apply(flip: bool, k_rot: int) -> None:
    """Non-coordinate channels are true fields and must transform exactly like _d4_apply."""
    feats = _canonical_features()
    out = _d4_apply_features(feats, flip, k_rot)
    ref = _d4_apply(feats.clone(), flip, k_rot)
    assert torch.equal(out[:, 7], ref[:, 7])


@pytest.mark.parametrize("k_rot", (1, 3))
def test_odd_rotation_rejected(k_rot: int) -> None:
    with pytest.raises(ValueError):
        _d4_apply_features(_canonical_features(), False, k_rot)
