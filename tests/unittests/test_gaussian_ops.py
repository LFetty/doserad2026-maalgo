from __future__ import annotations

import pytest
import torch

from pydose_rt.ops.gaussian import (
    gaussian_scatter_add,
    gaussian_scatter_add_autograd,
    gaussian_scatter_add_inplace,
)


def _case(device: torch.device, requires_grad: bool = False):
    size_h, size_d, size_w = 7, 6, 8
    resolution = (1.7, 2.3, 3.1)
    base = torch.linspace(
        -0.1,
        0.2,
        size_h * size_d * size_w,
        device=device,
        dtype=torch.float32,
    )
    if requires_grad:
        base.requires_grad_(True)
    coords = torch.tensor(
        [
            [[3.2, 4.7, 5.1], [1.4, 5.9, 9.2], [-0.2, 2.1, 3.3]],
            [[6.3, 1.8, 7.7], [0.5, 0.5, 0.5], [9.9, 8.1, 7.2]],
        ],
        device=device,
        dtype=torch.float32,
    )
    values = torch.tensor(
        [[1.0, -0.5, 0.25], [0.75, 0.3, -0.2]],
        device=device,
        dtype=torch.float32,
        requires_grad=requires_grad,
    )
    return base, coords, values, resolution, (size_h, size_d, size_w)


def _torch_reference(base, coords, values, resolution, shape):
    return gaussian_scatter_add(
        base,
        coords,
        values,
        resolution[0],
        resolution[1],
        resolution[2],
        shape[0],
        shape[1],
        shape[2],
        0.5,
        1,
    )


def test_gaussian_scatter_add_inplace_matches_functional_reference() -> None:
    base, coords, values, resolution, shape = _case(torch.device("cpu"))
    expected = _torch_reference(base, coords, values, resolution, shape)
    actual = base.clone()
    gaussian_scatter_add_inplace(actual, coords, values, resolution, shape, 0.5, 1)
    assert torch.allclose(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gaussian_scatter_add_triton_forward_matches_torch_reference() -> None:
    device = torch.device("cuda")
    base, coords, values, resolution, shape = _case(device)
    expected = _torch_reference(base, coords, values, resolution, shape)
    actual = gaussian_scatter_add_autograd(
        base,
        coords,
        values,
        resolution[0],
        resolution[1],
        resolution[2],
        shape[0],
        shape[1],
        shape[2],
        0.5,
        1,
    )
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gaussian_scatter_add_triton_value_backward_matches_torch_reference() -> None:
    device = torch.device("cuda")
    base_ref, coords_ref, values_ref, resolution, shape = _case(device, requires_grad=True)
    base_tri = base_ref.detach().clone().requires_grad_(True)
    coords_tri = coords_ref.detach().clone()
    values_tri = values_ref.detach().clone().requires_grad_(True)

    expected = _torch_reference(base_ref, coords_ref, values_ref, resolution, shape)
    actual = gaussian_scatter_add_autograd(
        base_tri,
        coords_tri,
        values_tri,
        resolution[0],
        resolution[1],
        resolution[2],
        shape[0],
        shape[1],
        shape[2],
        0.5,
        1,
    )
    probe = torch.randn_like(expected)
    (expected * probe).sum().backward()
    (actual * probe).sum().backward()

    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)
    assert torch.allclose(base_tri.grad, base_ref.grad, atol=0.0, rtol=0.0)
    assert torch.allclose(values_tri.grad, values_ref.grad, atol=2e-5, rtol=2e-5)
