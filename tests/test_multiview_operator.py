"""Operator-level tests for utils/multiview_mri.py.

Covers: forward/adjoint inner-product (adjoint) test, observed-coefficient
counting, finite-difference gradient of the data fidelity, and CPU/GPU
consistency.  Run with ``pytest -q tests/test_multiview_operator.py``.
"""

import numpy as np
import pytest
import torch

from utils.multiview_mri import (
    MRIViewOperator,
    MultiViewMRI,
    channels_to_complex,
    complex_to_channels,
)


def _rand_maps(V, C, H, W, dtype=torch.complex128, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(V, C, H, W, generator=g, dtype=torch.float64)
            + 1j * torch.randn(V, C, H, W, generator=g, dtype=torch.float64)).to(dtype)


def _rand_mask(V, H, W, frac=0.4, seed=1):
    g = torch.Generator().manual_seed(seed)
    m = (torch.rand(V, 1, H, W, generator=g) < frac).to(torch.float64)
    # guarantee a non-empty mask per view
    m[:, :, H // 2, W // 2] = 1.0
    return m


def test_forward_adjoint_inner_product():
    """<A x, y> == <x, A^H y> for the single-view SENSE operator (complex)."""
    torch.manual_seed(0)
    C, H, W = 4, 12, 10
    maps = _rand_maps(1, C, H, W)[0]
    mask = _rand_mask(1, H, W)[0]
    op = MRIViewOperator(mask=mask, maps=maps)

    x = (torch.randn(1, 1, H, W, dtype=torch.float64)
         + 1j * torch.randn(1, 1, H, W, dtype=torch.float64))
    y = (torch.randn(1, C, H, W, dtype=torch.float64)
         + 1j * torch.randn(1, C, H, W, dtype=torch.float64))

    Ax = op.forward_complex(x)
    AHy = op.adjoint(y)
    lhs = torch.vdot(Ax.reshape(-1), y.reshape(-1))     # <Ax, y>
    rhs = torch.vdot(x.reshape(-1), AHy.reshape(-1))    # <x, A^H y>
    assert torch.allclose(lhs, rhs, rtol=1e-8, atol=1e-8), (lhs, rhs)


def test_num_observed_counts_coils_times_lines():
    C, H, W = 8, 16, 16
    mask = _rand_mask(2, H, W, frac=0.3)
    maps = _rand_maps(2, C, H, W)
    noise = torch.ones(2, dtype=torch.float64)
    mv = MultiViewMRI(mask, maps, noise)
    n_obs = mv.num_observed()  # [1, V]
    expected = (mask.abs() > 0).float().sum(dim=(-3, -2, -1)) * C
    assert torch.allclose(n_obs.reshape(-1), expected.reshape(-1))


def test_channels_roundtrip():
    x = torch.randn(3, 2, 5, 7, dtype=torch.float64)
    xc = channels_to_complex(x)
    xr = complex_to_channels(xc)
    assert torch.allclose(x, xr)
    assert xc.shape == (3, 1, 5, 7)


def test_finite_difference_gradient():
    """Autograd gradient of the multi-view data fidelity matches central diff."""
    torch.manual_seed(3)
    V, C, H, W = 2, 2, 8, 8
    maps = _rand_maps(V, C, H, W, seed=7)
    mask = _rand_mask(V, H, W, frac=0.5, seed=9)
    noise = torch.tensor([0.7, 1.3], dtype=torch.float64)
    mv = MultiViewMRI(mask, maps, noise)

    x_true = torch.randn(1, 2, H, W, dtype=torch.float64)
    y = mv.forward(x_true).detach()
    # perturb measurements so the fidelity is non-zero
    y = y + 0.1 * (torch.randn_like(y.real) + 1j * torch.randn_like(y.real))

    x = torch.randn(1, 2, H, W, dtype=torch.float64, requires_grad=True)
    dc = mv.data_fidelity_per_sample(x, y).sum()
    (grad_auto,) = torch.autograd.grad(dc, x)

    eps = 1e-6
    grad_fd = torch.zeros_like(x)
    x_d = x.detach().clone()
    for i in range(x.numel()):
        flat = x_d.reshape(-1).clone()
        flat[i] += eps
        fp = mv.data_fidelity_per_sample(flat.reshape(x.shape), y).sum()
        flat[i] -= 2 * eps
        fm = mv.data_fidelity_per_sample(flat.reshape(x.shape), y).sum()
        grad_fd.reshape(-1)[i] = (fp - fm) / (2 * eps)

    assert torch.allclose(grad_auto, grad_fd, rtol=1e-4, atol=1e-6), \
        (grad_auto.reshape(-1)[:5], grad_fd.reshape(-1)[:5])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cpu_gpu_consistency():
    torch.manual_seed(5)
    V, C, H, W = 2, 4, 16, 16
    maps = _rand_maps(V, C, H, W, seed=2)
    mask = _rand_mask(V, H, W, frac=0.4, seed=3)
    noise = torch.tensor([0.8, 1.1], dtype=torch.float64)
    x = torch.randn(2, 2, H, W, dtype=torch.float64)

    mv_cpu = MultiViewMRI(mask, maps, noise)
    y_cpu = mv_cpu.forward(x)
    d_cpu = mv_cpu.data_fidelity_per_sample(x, y_cpu + 0.05)

    dev = torch.device("cuda")
    mv_gpu = MultiViewMRI(mask.to(dev), maps.to(dev), noise.to(dev))
    y_gpu = mv_gpu.forward(x.to(dev))
    d_gpu = mv_gpu.data_fidelity_per_sample(x.to(dev), y_gpu + 0.05)

    assert torch.allclose(y_cpu, y_gpu.cpu(), rtol=1e-6, atol=1e-6)
    assert torch.allclose(d_cpu, d_gpu.cpu(), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
