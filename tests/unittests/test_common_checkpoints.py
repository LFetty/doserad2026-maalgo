from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from training.common.checkpoints import load_model_state_dict, portable_model_state_dict


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)


class _Wrapped(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._orig_mod = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._orig_mod(x)


def test_load_model_state_dict_strips_orig_mod_prefix_for_plain_model() -> None:
    source = _Tiny()
    with torch.no_grad():
        source.linear.weight.fill_(2.0)
        source.linear.bias.fill_(3.0)
    prefixed = OrderedDict((f"_orig_mod.{key}", value.clone()) for key, value in source.state_dict().items())

    target = _Tiny()
    load_model_state_dict(target, prefixed)

    assert torch.equal(target.linear.weight, source.linear.weight)
    assert torch.equal(target.linear.bias, source.linear.bias)


def test_load_model_state_dict_adds_orig_mod_prefix_for_wrapped_model() -> None:
    source = _Tiny()
    with torch.no_grad():
        source.linear.weight.fill_(4.0)
        source.linear.bias.fill_(5.0)

    target = _Wrapped(_Tiny())
    load_model_state_dict(target, source.state_dict())

    assert torch.equal(target._orig_mod.linear.weight, source.linear.weight)
    assert torch.equal(target._orig_mod.linear.bias, source.linear.bias)


def test_portable_model_state_dict_unwraps_orig_mod() -> None:
    source = _Tiny()
    wrapped = _Wrapped(source)

    assert tuple(portable_model_state_dict(wrapped).keys()) == tuple(source.state_dict().keys())
