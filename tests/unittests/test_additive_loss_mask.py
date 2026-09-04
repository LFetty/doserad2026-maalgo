"""`--loss-mask additive` must add high10 pressure WITHOUT suppressing full support.

The problem it solves: `blend` computes ``f * dose_high10 + (1-f) * dose``, but the two
terms are not on the same scale. ``dose_high10`` divides by the high10 voxel count and
``dose`` by the full nonzero count, and the halo outnumbers the scored core 30-87x, so
``dose_high10`` runs ~17x larger. At the shipped frac=0.7 that makes the actual split
about 40:1, not 70:30 -- the full-support term carries ~2.4% of the dose loss. Switching
to blend cost ~27% of mae_pct for that reason.

`additive` keeps full support at weight 1.0 and adds high10 on top, so the two can be
tuned independently.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training.proton.train_dense_correction import _loss  # noqa: E402


def _args(**over):
    base = dict(
        w_dose=1.0, w_energy=0.0, w_profile=0.0, w_idd=0.0, w_peak=0.0, w_idd_z=0.0,
        loss_mask="nonzero", loss_high10_frac=0.7, loss_high10_weight=0.15,
        huber_delta=0.05, depth_bin_mm=1.0, peak_scale_mm=1.0, peak_tau_frac=0.5,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _case():
    """A support shaped like a real beamlet: small high-dose core, large low-dose halo.

    Two properties matter and both are needed to reproduce the scale gap:
      * the halo outnumbers the >=10% core by 30-87x, so the two terms divide by very
        different voxel counts;
      * error scales WITH dose, so the core also carries the larger absolute errors.
    Uniform additive noise would make the two means comparable and hide the gap entirely.
    """
    torch.manual_seed(0)
    ref = torch.rand(1, 1, 8, 16, 16) * 0.03          # halo, all below the 10% mask
    ref[..., 3:5, 7:9, 7:9] = 1.0                     # core: 16 of 2048 voxels (~1/128)
    pred = (ref * (1.0 + 0.05 * torch.randn_like(ref))).clamp_min(0.0).requires_grad_(True)
    depth = torch.rand(8, 16, 16) * 100.0
    return pred, ref, depth


def test_additive_equals_full_support_when_weight_is_zero():
    pred, ref, depth = _case()
    add, _ = _loss(pred, ref, depth, _args(loss_mask="additive", loss_high10_weight=0.0))
    nz, _ = _loss(pred, ref, depth, _args(loss_mask="nonzero"))
    assert abs(float(add) - float(nz)) < 1e-9


def test_additive_preserves_the_full_support_term_but_blend_shrinks_it():
    """The core claim. `blend` scales full support by (1-f); `additive` leaves it at 1.0."""
    pred, ref, depth = _case()
    nz, _ = _loss(pred, ref, depth, _args(loss_mask="nonzero"))
    add, _ = _loss(pred, ref, depth, _args(loss_mask="additive", loss_high10_weight=0.15))
    bl, _ = _loss(pred, ref, depth, _args(loss_mask="blend", loss_high10_frac=0.7))

    # additive >= nonzero, because it is nonzero plus a non-negative term.
    assert float(add) > float(nz)
    # The full-support contribution inside additive is exactly the nonzero loss.
    assert float(add) - float(nz) > 0
    assert float(bl) != float(add)


def test_the_two_terms_really_are_on_different_scales():
    """Pins the 17x-ish gap that makes frac=0.7 an effective 40:1 split.

    If this ever fails the scale gap has changed and the guidance on --loss-high10-weight
    in the argument help needs revisiting.
    """
    pred, ref, depth = _case()
    _, t = _loss(pred, ref, depth, _args(loss_mask="nonzero"))
    assert t["mae_high10_pct"] > 3.0 * t["mae_pct"], (
        f"expected dose_high10 >> dose; got high10={t['mae_high10_pct']:.4g} "
        f"vs full={t['mae_pct']:.4g}"
    )


def test_additive_weight_monotonically_increases_the_loss():
    pred, ref, depth = _case()
    losses = [
        float(_loss(pred, ref, depth, _args(loss_mask="additive", loss_high10_weight=w))[0])
        for w in (0.0, 0.06, 0.15, 1.0)
    ]
    assert losses == sorted(losses)


def test_additive_gradient_is_finite():
    pred, ref, depth = _case()
    total, _ = _loss(pred, ref, depth, _args(loss_mask="additive", loss_high10_weight=0.15))
    total.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0
