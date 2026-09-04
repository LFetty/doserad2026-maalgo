"""best.pt must track the scored metric, not whichever epoch the run stopped on.

`latest.pt` is the last epoch, which stops meaning "best" once the curve flattens --
the long continuations wobble ~0.5% between adjacent epochs, and that is the same order
as the differences the objective screens are trying to resolve. These tests pin the
selection predicate: lower wins, non-finite never wins, and the incumbent survives a
resume.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training.proton.train_dense_correction import _is_new_best, _parse_args  # noqa: E402


def _state(value=None):
    return {"value": value, "epoch": -1, "step": -1}


def test_first_finite_value_always_wins():
    assert _is_new_best(_state(None), 0.9)


def test_lower_wins_and_higher_does_not():
    assert _is_new_best(_state(0.704), 0.655)
    assert not _is_new_best(_state(0.655), 0.704)


def test_equal_does_not_win():
    """Ties keep the earlier checkpoint -- rewriting best.pt on a tie is pure churn."""
    assert not _is_new_best(_state(0.7), 0.7)


def test_non_finite_never_wins():
    """An unguarded `<` against NaN is False both ways, freezing best.pt silently."""
    assert not _is_new_best(_state(0.7), float("nan"))
    assert not _is_new_best(_state(0.7), float("inf"))
    assert not _is_new_best(_state(None), float("nan"))
    assert not _is_new_best(_state(0.7), None)


def test_resumed_incumbent_is_not_beaten_by_a_worse_restart():
    """The failure this guards: resume forgets the incumbent, and the first post-restart
    epoch -- typically worse, since the LR schedule re-warms -- overwrites best.pt."""
    resumed = _state(0.655)
    assert not _is_new_best(resumed, 0.671)


def test_best_metric_defaults_to_the_scored_proxy():
    args = _parse_args_with([])
    assert args.best_metric == "mae_high10_pct"


def test_integral_ratio_is_rejected_as_a_best_metric():
    """Closest-to-1 is better for integral_ratio, so a lower-is-better selector would
    pick the worst conservation. It must not be selectable at all."""
    import pytest

    with pytest.raises(SystemExit):
        _parse_args_with(["--best-metric", "integral_ratio"])


def _parse_args_with(extra: list[str]):
    argv = sys.argv
    try:
        sys.argv = ["train_dense_correction.py", "--output-dir", "/tmp/unused", *extra]
        return _parse_args()
    finally:
        sys.argv = argv
