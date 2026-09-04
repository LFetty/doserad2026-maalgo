from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch
from torch import nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


_WRAPPER_PREFIXES = ("_orig_mod.", "module.")


def _copy_state_dict(state_dict: Mapping[str, Any]) -> OrderedDict[str, Any]:
    copied = OrderedDict(state_dict.items())
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        copied._metadata = deepcopy(metadata)  # type: ignore[attr-defined]
    return copied


def _add_prefix_to_state_dict(state_dict: OrderedDict[str, Any], prefix: str) -> OrderedDict[str, Any]:
    prefixed = OrderedDict((f"{prefix}{key}", value) for key, value in state_dict.items())
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        prefixed._metadata = OrderedDict(  # type: ignore[attr-defined]
            (f"{prefix}{key}" if key else key, value) for key, value in metadata.items()
        )
    return prefixed


def state_dict_for_model(model: nn.Module, state_dict: Mapping[str, Any]) -> OrderedDict[str, Any]:
    """Return a checkpoint state dict with wrapper prefixes matched to ``model``."""
    normalized = _copy_state_dict(state_dict)
    model_keys = tuple(model.state_dict().keys())

    for prefix in _WRAPPER_PREFIXES:
        state_keys = tuple(str(key) for key in normalized.keys())
        if not state_keys or not model_keys:
            break
        state_has_prefix = any(key.startswith(prefix) for key in state_keys)
        model_has_prefix = any(key.startswith(prefix) for key in model_keys)
        if state_has_prefix and not model_has_prefix:
            consume_prefix_in_state_dict_if_present(normalized, prefix)
        elif model_has_prefix and not state_has_prefix:
            normalized = _add_prefix_to_state_dict(normalized, prefix)

    return normalized


def load_model_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    strict: bool = True,
) -> torch.nn.modules.module._IncompatibleKeys:
    return model.load_state_dict(state_dict_for_model(model, state_dict), strict=strict)


def portable_model_state_dict(model: nn.Module) -> OrderedDict[str, Any] | Mapping[str, Any]:
    """Save the underlying module when torch.compile/DataParallel wrappers are present."""
    unwrapped = model
    seen: set[int] = set()
    while id(unwrapped) not in seen:
        seen.add(id(unwrapped))
        wrapped = getattr(unwrapped, "_orig_mod", None)
        if isinstance(wrapped, nn.Module):
            unwrapped = wrapped
            continue
        wrapped = getattr(unwrapped, "module", None)
        if isinstance(wrapped, nn.Module):
            unwrapped = wrapped
            continue
        break
    return unwrapped.state_dict()
