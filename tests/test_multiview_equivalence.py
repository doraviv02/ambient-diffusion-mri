"""Equivalence tests for the multi-view MRI operator.

These tests decide whether the multi-view code is a correct generalization of
the published single-view baseline, and which multi-view settings are merely
data-merging versus genuinely multi-operator.  Run with
``pytest -q tests/test_multiview_equivalence.py``.
"""

import torch

from utils.multiview_mri import (
    MRIViewOperator,
    MultiViewMRI,
    channels_to_complex,
    fft2c,
    noise_weighted_merge,
)


# ---------------------------------------------------------------------------
# Literal copy of the original solve_inverse_adps.MRI_utils forward/adjoint so
# we can assert exact equivalence *without* importing that script (it executes
# an argparse + data-loading loop at import time).
# ---------------------------------------------------------------------------
class _OriginalMRIUtils:
    def __init__(self, mask, maps):
        self.mask = mask
        self.maps = maps

    @staticmethod
    def _fft(x):
        return torch.fft.fft2(x, dim=(-2, -1), norm="ortho")

    def forward(self, x):
        x_cplx = torch.view_as_complex(x.permute(0, -2, -1, 1).contiguous())[:, None, ...]
        coil_imgs = self.maps * x_cplx
        coil_ksp = self._fft(coil_imgs)
        return self.mask * coil_ksp


def _rand_maps(V, C, H, W, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(V, C, H, W, generator=g, dtype=torch.float64)
            + 1j * torch.randn(V, C, H, W, generator=g, dtype=torch.float64))


def _line_mask(V, H, W, sampled_cols_list):
    """Cartesian (column) masks: sampled_cols_list[v] is an iterable of columns."""
    m = torch.zeros(V, 1, H, W, dtype=torch.float64)
    for v, cols in enumerate(sampled_cols_list):
        m[v, :, :, list(cols)] = 1.0
    return m


# ---------------------------------------------------------------------------
# 1. Single-view equivalence to the original operator (forward + gradient).
# ---------------------------------------------------------------------------
def test_v1_forward_matches_original():
    C, H, W = 6, 12, 10
    maps = _rand_maps(1, C, H, W, seed=1)
    mask = _line_mask(1, H, W, [range(0, W, 2)])
    x = torch.randn(3, 2, H, W, dtype=torch.float64)

    orig = _OriginalMRIUtils(mask[0], maps[0])
    mv = MultiViewMRI(mask, maps, torch.ones(1, dtype=torch.float64))

    out_orig = orig.forward(x)             # [3, C, H, W]
    out_mv = mv.forward(x)[:, 0]           # [3, C, H, W]
    assert torch.allclose(out_orig, out_mv, rtol=1e-10, atol=1e-10)


def test_v1_gradient_matches_original():
    """grad of original SSE == N * grad of normalized fidelity (sigma=1, V=1)."""
    C, H, W = 4, 10, 8
    maps = _rand_maps(1, C, H, W, seed=2)
    mask = _line_mask(1, H, W, [range(0, W, 2)])
    noise = torch.ones(1, dtype=torch.float64)
    mv = MultiViewMRI(mask, maps, noise)
    orig = _OriginalMRIUtils(mask[0], maps[0])

    x_true = torch.randn(1, 2, H, W, dtype=torch.float64)
    y = (mask[0] * (fft2c(maps[0] * channels_to_complex(x_true)))).detach()
    y = y + mask[0] * 0.1 * (torch.randn_like(y.real) + 1j * torch.randn_like(y.real))

    # original data-fidelity gradient: sum ||y - A x||^2
    x1 = torch.randn(1, 2, H, W, dtype=torch.float64, requires_grad=True)
    resid = (y - orig.forward(x1)).reshape(1, -1)
    sse = torch.sum(torch.norm(resid, dim=-1) ** 2)
    (g_orig,) = torch.autograd.grad(sse, x1)

    # normalized fidelity gradient at the same point
    x2 = x1.detach().clone().requires_grad_(True)
    dc = mv.data_fidelity_per_sample(x2, y).sum()
    (g_mv,) = torch.autograd.grad(dc, x2)

    n_obs = mv.num_observed().reshape(-1).item()  # C * nnz
    assert torch.allclose(g_orig, n_obs * g_mv, rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# 2. Duplicated-view invariance.
# ---------------------------------------------------------------------------
def test_duplicate_view_invariance():
    C, H, W = 4, 12, 12
    maps1 = _rand_maps(1, C, H, W, seed=3)
    mask1 = _line_mask(1, H, W, [range(0, W, 3)])
    noise1 = torch.tensor([0.9], dtype=torch.float64)

    mv1 = MultiViewMRI(mask1, maps1, noise1)
    mv2 = MultiViewMRI(mask1.repeat(2, 1, 1, 1), maps1.repeat(2, 1, 1, 1),
                       noise1.repeat(2))

    x = torch.randn(1, 2, H, W, dtype=torch.float64, requires_grad=True)
    y1 = mv1.forward(x).detach() + 0.05
    y2 = torch.cat([y1, y1], dim=1)  # duplicate the measurement view

    d1 = mv1.data_fidelity_per_sample(x, y1).sum()
    (g1,) = torch.autograd.grad(d1, x, retain_graph=True)
    d2 = mv2.data_fidelity_per_sample(x, y2).sum()
    (g2,) = torch.autograd.grad(d2, x)

    assert torch.allclose(d1, d2, rtol=1e-10, atol=1e-10)
    assert torch.allclose(g1, g2, rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# 3. Joint likelihood == analytically noise-weighted merged measurement.
# ---------------------------------------------------------------------------
def test_joint_equals_weighted_merge_gradient():
    """With shared maps/geometry the joint weighted-SSE gradient equals the
    gradient of the analytically noise-weighted merged measurement, even for
    overlapping masks and unequal noise."""
    C, H, W = 4, 12, 12
    shared = _rand_maps(1, C, H, W, seed=4)[0]           # [C, H, W]
    maps = shared[None].repeat(2, 1, 1, 1)[None]         # [1, 2, C, H, W]

    # overlapping masks: shared center columns + disjoint outer columns
    center = list(range(4, 8))
    mask = _line_mask(2, H, W, [center + [0, 1, 2], center + [9, 10, 11]])
    noise = torch.tensor([0.7, 1.4], dtype=torch.float64)
    mv = MultiViewMRI(mask[None], maps, noise[None])

    # arbitrary masked measurements per view
    g = torch.Generator().manual_seed(11)
    y = (torch.randn(2, C, H, W, generator=g, dtype=torch.float64)
         + 1j * torch.randn(2, C, H, W, generator=g, dtype=torch.float64))
    y = mask * y                                         # [2,1,H,W]*[2,C,H,W]

    merged = noise_weighted_merge(y, mask, noise)
    w = merged["inv_var_map"]                            # [1,1,H,W] per-location inv-var
    y_merge = merged["ksp"]                              # [1,C,H,W]

    var = noise.square()

    def joint_wsse(xr):
        pred = mv.forward(xr)[0]                         # [2,C,H,W] masked
        resid = pred - y
        sse = (resid.real ** 2 + resid.imag ** 2).sum(dim=(-3, -2, -1))  # [2]
        return (sse / var).sum()

    def merged_wsse(xr):
        coilksp = fft2c(shared * channels_to_complex(xr))  # [1,C,H,W] unmasked
        diff = coilksp - y_merge
        return (w[0] * (diff.real ** 2 + diff.imag ** 2)).sum()

    x = torch.randn(1, 2, H, W, dtype=torch.float64, requires_grad=True)
    (g_joint,) = torch.autograd.grad(joint_wsse(x), x, retain_graph=True)
    (g_merge,) = torch.autograd.grad(merged_wsse(x), x)
    assert torch.allclose(g_joint, g_merge, rtol=1e-7, atol=1e-7), \
        (g_joint.reshape(-1)[:4], g_merge.reshape(-1)[:4])


# ---------------------------------------------------------------------------
# 4. Complementary-mask union equivalence (equal per-view sample counts).
# ---------------------------------------------------------------------------
def test_complementary_masks_equal_union_single_view():
    C, H, W = 4, 12, 12
    shared = _rand_maps(1, C, H, W, seed=6)              # [1,C,H,W]
    # disjoint masks with equal column counts -> equal N per view
    cols_a = list(range(0, 12, 2))                       # 6 columns
    cols_b = list(range(1, 12, 2))                       # 6 columns (disjoint)
    mask = _line_mask(2, H, W, [cols_a, cols_b])
    noise = torch.tensor([1.0, 1.0], dtype=torch.float64)

    maps2 = shared.repeat(2, 1, 1, 1)                    # [2,C,H,W]
    mv2 = MultiViewMRI(mask, maps2, noise)

    # single-view union operator
    union_mask = (mask[0:1] + mask[1:2]).clamp(max=1.0)  # [1,1,H,W]
    mv1 = MultiViewMRI(union_mask, shared, noise[0:1])

    x = torch.randn(1, 2, H, W, dtype=torch.float64, requires_grad=True)
    y2 = mv2.forward(x).detach()
    y2 = y2 + mask[None] * 0.05 * (torch.randn_like(y2.real) + 1j * torch.randn_like(y2.real))
    # disjoint supports -> the union measurement is just the sum of the views
    y_union = (y2[:, 0] + y2[:, 1])[:, None]             # [B, V=1, C, H, W]

    d2 = mv2.data_fidelity_per_sample(x, y2).sum()
    (g2,) = torch.autograd.grad(d2, x, retain_graph=True)
    d1 = mv1.data_fidelity_per_sample(x, y_union).sum()
    (g1,) = torch.autograd.grad(d1, x)

    assert torch.allclose(d1, d2, rtol=1e-8, atol=1e-8), (d1, d2)
    assert torch.allclose(g1, g2, rtol=1e-7, atol=1e-7)


# ---------------------------------------------------------------------------
# 5. Unequal-noise weighting: a high-noise view contributes less.
# ---------------------------------------------------------------------------
def test_unequal_noise_weighting():
    C, H, W = 4, 12, 12
    maps = _rand_maps(2, C, H, W, seed=8)
    maps[1] = maps[0]                                    # identical maps/geometry
    mask = _line_mask(2, H, W, [range(0, W, 2), range(0, W, 2)])
    noise = torch.tensor([0.5, 2.0], dtype=torch.float64)  # view0 low-noise, view1 high-noise
    mv = MultiViewMRI(mask, maps, noise)

    x = torch.randn(1, 2, H, W, dtype=torch.float64)
    clean = mv.forward(x).detach()                      # [1,2,C,H,W]
    err = mask[None] * 0.1 * (torch.randn_like(clean.real) + 1j * torch.randn_like(clean.real))

    # Case A: error only in the low-noise view (view 0)
    yA = clean.clone(); yA[:, 0] = clean[:, 0] + err[:, 0]
    dA = mv.data_fidelity_per_sample(x, yA)
    # Case B: same-magnitude error only in the high-noise view (view 1)
    yB = clean.clone(); yB[:, 1] = clean[:, 1] + err[:, 0]  # same error tensor
    dB = mv.data_fidelity_per_sample(x, yB)

    # An error in the low-noise view must penalize the fidelity more.
    assert (dA > dB).all(), (dA, dB)
    # Ratio should be ~ (sigma_high/sigma_low)^2 = 16.
    ratio = (dA / dB).item()
    assert 10.0 < ratio < 22.0, ratio


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
