"""The --w-idd-z loss must mirror the reported idd_z metric, and the scored axis.

The trap this guards: `--w-idd` bins along `depth_mm`, the radiological BEAM axis, so it
is a Bragg-curve term. The challenge's Level-1.2 IDD profiles along numpy axis 0 = world
z, which is the BEV `h` axis -- perpendicular to the beam. Training on --w-idd therefore
does not touch the scored metric at all. These tests pin --w-idd-z to the right axis.
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
        loss_mask="nonzero", loss_high10_frac=0.7, huber_delta=0.05,
        depth_bin_mm=1.0, peak_scale_mm=1.0, peak_tau_frac=0.5,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _case():
    torch.manual_seed(0)
    # [N, C, D, H, W]; H is the BEV h axis == world z.
    ref = torch.rand(1, 1, 4, 5, 6).clamp_min(0.0)
    pred = (ref + 0.05 * torch.randn_like(ref)).clamp_min(0.0).requires_grad_(True)
    depth = torch.rand(4, 5, 6) * 100.0
    return pred, ref, depth


def test_idd_z_loss_matches_the_reported_metric():
    pred, ref, depth = _case()
    _, terms = _loss(pred, ref, depth, _args(w_idd_z=1.0))
    # The metric is computed unconditionally; the loss should agree with it closely.
    # (They differ only by the eps inside the sqrt, added for gradient stability.)
    assert abs(terms["idd_z_loss"] - terms["idd_z"]) < 1e-4


def test_idd_z_loss_is_zero_axis_correct():
    """A prediction that redistributes dose WITHIN a z-slice must not change idd_z."""
    torch.manual_seed(1)
    ref = torch.rand(1, 1, 3, 4, 5)
    depth = torch.rand(3, 4, 5) * 100.0
    # Permute along the two transverse axes only (D and W), leaving H (=z) sums intact.
    shuffled = ref.flip(dims=(-3,)).flip(dims=(-1,))
    torch.testing.assert_close(
        ref.sum(dim=(-3, -1)), shuffled.sum(dim=(-3, -1))
    )
    _, t = _loss(shuffled.clone().requires_grad_(True), ref, depth, _args(w_idd_z=1.0))
    assert t["idd_z_loss"] < 1e-3, "z-IDD must be blind to redistribution within a z-slice"


def test_idd_z_loss_penalises_shifting_dose_across_z():
    pred, ref, depth = _case()
    _, flat = _loss(pred, ref, depth, _args(w_idd_z=1.0))
    shifted = ref.roll(shifts=1, dims=-2)  # move dose to a different z slice
    _, moved = _loss(shifted.clone().requires_grad_(True), ref, depth, _args(w_idd_z=1.0))
    assert moved["idd_z_loss"] > flat["idd_z_loss"] * 2.0


def test_idd_z_contributes_gradient_and_changes_total():
    pred, ref, depth = _case()
    off, _ = _loss(pred, ref, depth, _args(w_idd_z=0.0))
    on, _ = _loss(pred, ref, depth, _args(w_idd_z=1.0))
    assert on.item() > off.item()
    on.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0
