from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "container"))

from standalone_regression_inference import StandaloneRegressionInference  # noqa: E402


def _fake_predictor(patch_batch_size: int, calls: list[int]) -> StandaloneRegressionInference:
    predictor = StandaloneRegressionInference.__new__(StandaloneRegressionInference)
    predictor.device = torch.device("cpu")
    predictor.patch_size = (2, 2)
    predictor.tile_step_size = 0.5
    predictor.use_gaussian = False
    predictor._gaussian_cache = None
    predictor.patch_batch_size = patch_batch_size

    def forward(data: torch.Tensor) -> torch.Tensor:
        calls.append(int(data.shape[0]))
        return data[:, :1] * 2.0 + 1.0

    predictor._model_forward = forward
    return predictor


def test_patch_batching_preserves_blending_and_uses_requested_batches() -> None:
    data = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)

    serial_calls: list[int] = []
    serial = _fake_predictor(1, serial_calls)._sliding_window_inference(data)

    batched_calls: list[int] = []
    batched = _fake_predictor(3, batched_calls)._sliding_window_inference(data)

    torch.testing.assert_close(batched, serial, rtol=0, atol=0)
    assert serial_calls == [1, 1, 1, 1]
    assert batched_calls == [3, 1]
