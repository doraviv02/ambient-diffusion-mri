"""Multi-view MRI forward operator and noise-weighted data fidelity.

This module implements the measurement model described in
``MULTIVIEW_ULF_AMBIENT_DIFFUSION_PROJECT.md`` (Section 3 / 6) for the
seminar project.  It is written to be *numerically identical* to the original
single-view ``MRI_utils`` class in ``solve_inverse_adps.py`` when ``V=1`` so
that Stage-1 multi-view reconstruction reduces exactly to the published
baseline.

Conventions (must match the pretrained Ambient checkpoint / the repo):

* The common image ``x`` is stored as a 2-channel *real* tensor ``[B, 2, H, W]``
  with ``real = channel 0`` and ``imag = channel 1``.
* Coil sensitivity maps are complex ``[V, C, H, W]`` (or ``[B, V, C, H, W]``).
* Sampling masks are real/binary ``[V, 1, H, W]`` (or ``[B, V, 1, H, W]``),
  broadcast across the coil dimension.
* The FFT is the orthonormal, *un-shifted* 2-D FFT (``norm='ortho'``).  The
  repository centers the transform by applying ``fftmod`` (a checkerboard sign
  flip) to the k-space and the maps *outside* the operator; this module keeps
  that convention (it never applies ``fftmod`` itself), so callers must feed
  ``fftmod``-ed maps and k-space exactly like ``solve_inverse_adps.py`` does.

All new tensors are allocated with the device/dtype of the inputs -- there are
no ``.cuda()`` calls and no hard-coded shapes.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "fft2c",
    "ifft2c",
    "complex_to_channels",
    "channels_to_complex",
    "MRIViewOperator",
    "MultiViewMRI",
    "noise_weighted_merge",
]


# ---------------------------------------------------------------------------
# Centered (via external fftmod), orthonormal 2-D transforms.
# ---------------------------------------------------------------------------
def fft2c(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal 2-D FFT over the last two dims (matches repo ``fft``)."""
    return torch.fft.fft2(x, dim=(-2, -1), norm="ortho")


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal 2-D inverse FFT over the last two dims (matches repo ``ifft``)."""
    return torch.fft.ifft2(x, dim=(-2, -1), norm="ortho")


def complex_to_channels(x_cplx: torch.Tensor) -> torch.Tensor:
    """``[..., 1, H, W]`` complex -> ``[..., 2, H, W]`` real (real=0, imag=1)."""
    return torch.cat((x_cplx.real, x_cplx.imag), dim=-3)


def channels_to_complex(x_chan: torch.Tensor) -> torch.Tensor:
    """``[..., 2, H, W]`` real -> ``[..., 1, H, W]`` complex (real=0, imag=1)."""
    real = x_chan[..., 0:1, :, :]
    imag = x_chan[..., 1:2, :, :]
    return real + 1j * imag


# ---------------------------------------------------------------------------
# Single-view operator -- numerically identical to solve_inverse_adps.MRI_utils.
# ---------------------------------------------------------------------------
class MRIViewOperator:
    """SENSE forward/adjoint for one acquisition.

    Parameters
    ----------
    mask : ``[B, 1, H, W]`` or ``[1, H, W]`` real sampling mask.
    maps : ``[B, C, H, W]`` or ``[C, H, W]`` complex coil sensitivities.
    noise_std : optional scalar / ``[B]`` per-sample k-space noise std.
    """

    def __init__(self, mask: torch.Tensor, maps: torch.Tensor,
                 noise_std: Optional[torch.Tensor] = None):
        if mask.dim() == 3:
            mask = mask[None]
        if maps.dim() == 3:
            maps = maps[None]
        self.mask = mask
        self.maps = maps
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """2-channel real image ``[B, 2, H, W]`` -> masked coil k-space ``[B, C, H, W]``."""
        x_cplx = channels_to_complex(x)  # [B, 1, H, W]
        coil_imgs = self.maps * x_cplx
        coil_ksp = fft2c(coil_imgs)
        return self.mask * coil_ksp

    def forward_complex(self, x_cplx: torch.Tensor) -> torch.Tensor:
        """Complex image ``[B, 1, H, W]`` -> masked coil k-space ``[B, C, H, W]``."""
        coil_ksp = fft2c(self.maps * x_cplx)
        return self.mask * coil_ksp

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Masked coil k-space ``[B, C, H, W]`` -> complex image ``[B, 1, H, W]``."""
        sampled_ksp = self.mask * y
        coil_imgs = ifft2c(sampled_ksp)
        img_out = torch.sum(torch.conj(self.maps) * coil_imgs, dim=1)
        return img_out[:, None, ...]

    def adjoint_channels(self, y: torch.Tensor) -> torch.Tensor:
        """Adjoint returned as a 2-channel real image ``[B, 2, H, W]``."""
        return complex_to_channels(self.adjoint(y))

    def num_observed(self) -> torch.Tensor:
        """Number of observed *complex* coefficients per sample: C * nnz(mask)."""
        n_coils = self.maps.shape[-3]
        nnz = (self.mask.abs() > 0).float().sum(dim=(-3, -2, -1))  # [B]
        return nnz * n_coils


# ---------------------------------------------------------------------------
# Multi-view operator.
# ---------------------------------------------------------------------------
class MultiViewMRI:
    r"""Joint multi-view MRI operator and normalized data fidelity.

    The data fidelity implements (project doc Section 6.3):

    .. math::
        \mathcal{D}(x) = \frac{1}{V}\sum_{v=1}^V
            \frac{\lVert M_v(\mathcal{F}S_v x - y_v)\rVert_2^2}{\sigma_v^2 N_v}

    where :math:`N_v` is the number of observed complex coefficients of view
    :math:`v` (``C * nnz(mask_v)``).  Dividing by :math:`N_v` and averaging over
    views keeps the guidance magnitude approximately invariant to ``V``.

    Parameters
    ----------
    masks : ``[V, 1, H, W]`` or ``[B, V, 1, H, W]`` real.
    maps  : ``[V, C, H, W]`` or ``[B, V, C, H, W]`` complex.
    noise_std : ``[V]`` or ``[B, V]`` positive per-view k-space noise std.
    min_variance : lower bound on the noise variance (default 1e-8).
    """

    def __init__(self, masks: torch.Tensor, maps: torch.Tensor,
                 noise_std: torch.Tensor, min_variance: float = 1e-8):
        # Normalize to a leading batch axis of size 1 when a per-sample axis is
        # not provided.  Stored shapes: masks [B, V, 1, H, W], maps
        # [B, V, C, H, W], noise_std [B, V].
        if masks.dim() == 4:
            masks = masks[None]
        if maps.dim() == 4:
            maps = maps[None]
        if noise_std.dim() == 1:
            noise_std = noise_std[None]
        self.masks = masks
        self.maps = maps
        self.noise_std = noise_std
        self.min_variance = float(min_variance)

        self.num_views = masks.shape[1]
        self.num_coils = maps.shape[2]

    # -- helpers ----------------------------------------------------------
    def _match_batch(self, batch: int):
        masks = self.masks.expand(batch, -1, -1, -1, -1) if self.masks.shape[0] == 1 else self.masks
        maps = self.maps.expand(batch, -1, -1, -1, -1) if self.maps.shape[0] == 1 else self.maps
        noise = self.noise_std.expand(batch, -1) if self.noise_std.shape[0] == 1 else self.noise_std
        return masks, maps, noise

    def num_observed(self) -> torch.Tensor:
        """``[B, V]`` number of observed complex coefficients per view."""
        nnz = (self.masks.abs() > 0).float().sum(dim=(-3, -2, -1))  # [B|1, V]
        return nnz * self.num_coils

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Common image ``[B, 2, H, W]`` -> per-view coil k-space ``[B, V, C, H, W]``."""
        batch = x.shape[0]
        masks, maps, _ = self._match_batch(batch)
        x_cplx = channels_to_complex(x)          # [B, 1, H, W]
        x_cplx = x_cplx[:, None]                 # [B, 1, 1, H, W] -> broadcast over V, C
        coil_imgs = maps * x_cplx                # [B, V, C, H, W]
        coil_ksp = fft2c(coil_imgs)
        return masks * coil_ksp                  # [B, V, C, H, W]

    def data_fidelity_per_sample(self, x: torch.Tensor, y_views: torch.Tensor) -> torch.Tensor:
        """Return ``[B]`` normalized negative-log-likelihood (up to a constant).

        ``y_views`` is ``[V, C, H, W]`` or ``[B, V, C, H, W]`` masked k-space.
        """
        batch = x.shape[0]
        _, _, noise = self._match_batch(batch)
        pred = self.forward(x)                   # [B, V, C, H, W]
        if y_views.dim() == 4:
            y_views = y_views[None]
        resid = pred - y_views                   # y already masked
        sse = (resid.real ** 2 + resid.imag ** 2).sum(dim=(-3, -2, -1))  # [B, V]
        variance = noise.square().clamp_min(self.min_variance)          # [B, V]
        n_obs = self.num_observed().to(sse.dtype)                       # [B|1, V]
        per_view = sse / (variance * n_obs)                            # [B, V]
        return per_view.mean(dim=1)                                    # [B]

    def per_view_residual_norm(self, x: torch.Tensor, y_views: torch.Tensor) -> torch.Tensor:
        """``[B, V]`` normalized residual sqrt(sse / (sigma^2 N)) for logging."""
        batch = x.shape[0]
        _, _, noise = self._match_batch(batch)
        pred = self.forward(x)
        if y_views.dim() == 4:
            y_views = y_views[None]
        resid = pred - y_views
        sse = (resid.real ** 2 + resid.imag ** 2).sum(dim=(-3, -2, -1))
        variance = noise.square().clamp_min(self.min_variance)
        n_obs = self.num_observed().to(sse.dtype)
        return torch.sqrt((sse / (variance * n_obs)).clamp_min(0.0))


# ---------------------------------------------------------------------------
# Analytic noise-weighted k-space merge (for the ``merge_fixed`` baseline).
# ---------------------------------------------------------------------------
def noise_weighted_merge(ksp_views: torch.Tensor, mask_views: torch.Tensor,
                         noise_std: torch.Tensor, min_variance: float = 1e-8):
    r"""Combine aligned views into a single equivalent measurement.

    For shared coil maps / geometry, the maximum-likelihood combination of the
    per-view measurements at each k-space location is the inverse-variance
    weighted average over the views that sampled it:

    .. math::
        y_{\mathrm{merge}}(k) =
            \frac{\sum_v M_v(k)\, y_v(k)/\sigma_v^2}
                 {\sum_v M_v(k)/\sigma_v^2}, \qquad
        M_{\mathrm{merge}}(k) = \mathbb{1}\!\left[\sum_v M_v(k) > 0\right].

    The merged measurement has an *effective* per-location inverse variance
    :math:`\sum_v M_v(k)/\sigma_v^2`; the returned ``eff_noise_std`` is a single
    scalar-per-sample summary (harmonic-style) usable by the single-view
    fidelity, while ``inv_var_map`` gives the exact per-location weighting for
    tests / classical reconstruction.

    Parameters
    ----------
    ksp_views  : ``[V, C, H, W]`` or ``[B, V, C, H, W]`` complex, already masked.
    mask_views : ``[V, 1, H, W]`` or ``[B, V, 1, H, W]`` real.
    noise_std  : ``[V]`` or ``[B, V]``.

    Returns
    -------
    dict with ``ksp`` ``[B, C, H, W]``, ``mask`` ``[B, 1, H, W]``,
    ``inv_var_map`` ``[B, 1, H, W]``, ``eff_noise_std`` ``[B]``.
    """
    if ksp_views.dim() == 4:
        ksp_views = ksp_views[None]
    if mask_views.dim() == 4:
        mask_views = mask_views[None]
    if noise_std.dim() == 1:
        noise_std = noise_std[None]

    variance = noise_std.square().clamp_min(min_variance)       # [B, V]
    inv_var = (1.0 / variance)[..., None, None, None]           # [B, V, 1, 1, 1]

    weight = mask_views * inv_var                               # [B, V, 1, H, W]
    weight_sum = weight.sum(dim=1)                              # [B, 1, H, W]  (per-location inv-var)
    num = (weight * ksp_views).sum(dim=1)                       # [B, C, H, W]
    denom = weight_sum.clamp_min(1e-12)                         # avoid /0 at unsampled locations
    merged_ksp = num / denom                                    # [B, C, H, W]
    merged_mask = (weight_sum > 0).to(ksp_views.real.dtype)     # [B, 1, H, W]
    merged_ksp = merged_ksp * merged_mask                       # zero outside union support

    # A scalar effective noise std summarizing the merged measurement: the
    # merged coefficient at a sampled location has variance 1/weight_sum(k).
    sampled = merged_mask > 0
    per_loc_var = (1.0 / weight_sum.clamp_min(1e-12))
    eff_var = torch.stack([
        per_loc_var[b][sampled[b]].mean() if sampled[b].any() else per_loc_var.new_tensor(1.0)
        for b in range(per_loc_var.shape[0])
    ])
    eff_noise_std = eff_var.sqrt()                              # [B]

    return {
        "ksp": merged_ksp,
        "mask": merged_mask,
        "inv_var_map": weight_sum,
        "eff_noise_std": eff_noise_std,
    }
