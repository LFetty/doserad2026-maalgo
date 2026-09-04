"""Hook for the BEV lattice ion dose engine."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class _IdentityHook(nn.Module):
    def forward(self, value: Any, **_context: Any) -> Any:
        return value


class IonSparseHooks(nn.Module):
    """Correction hook for the BEV lattice ion dose engine.

    The ``dense_bev`` hook receives the BEV energy deposition payload before
    rotation into the patient frame and may return a corrected version.
    Pass ``dense_bev=<your_model>`` to plug in a trained correction model.
    """

    def __init__(self, dense_bev: nn.Module | None = None) -> None:
        super().__init__()
        self.dense_bev = dense_bev or _IdentityHook()

    def apply_dense_bev(self, payload: dict[str, torch.Tensor], **context: Any) -> dict[str, torch.Tensor]:
        return self.dense_bev(payload, **context)
