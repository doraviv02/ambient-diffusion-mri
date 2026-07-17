"""Shared MRI FFT/transform helpers (numpy) for M4Raw preprocessing & analysis.

M4Raw k-space is stored center-DC (``fastmri.ifft2c(kspace)`` yields the image),
so the transforms here are *centered* (ifftshift -> FFT -> fftshift).  This is
the physical convention used by preprocessing.  The reconstruction *operator*
(utils.multiview_mri) instead uses the repo's plain-FFT convention on
``fftmod``-ed inputs; ``fftmod`` below converts between the two and is applied
by the inference/training loaders exactly like ``solve_inverse_adps.py``.
"""

from __future__ import annotations

import numpy as np


def ifft2c_np(x: np.ndarray) -> np.ndarray:
    """Centered orthonormal 2-D inverse FFT over the last two axes."""
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1))


def fft2c_np(x: np.ndarray) -> np.ndarray:
    """Centered orthonormal 2-D FFT over the last two axes."""
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1))


def rss(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Root-sum-of-squares coil combination."""
    return np.sqrt((np.abs(x) ** 2).sum(axis=axis))


def fftmod_np(x: np.ndarray) -> np.ndarray:
    """Repo checkerboard modulation on the last two axes (returns a copy).

    Equivalent to multiplying by (-1)^(i+j); converts between center-DC and
    corner-DC conventions so the plain-FFT operator matches ``MRI_utils``.
    """
    x = x.copy()
    x[..., ::2, :] *= -1
    x[..., :, ::2] *= -1
    return x


def central_acs_mask(shape_hw, acs_width: int) -> np.ndarray:
    """Binary [H, W] mask selecting a central ``acs_width`` block of columns/rows.

    M4Raw is phase-encoded along the last axis; we keep a central square ACS
    region (both dims) which is adequate for ESPIRiT / global-phase estimation.
    """
    H, W = shape_hw
    m = np.zeros((H, W), dtype=np.float32)
    r0 = (H - acs_width) // 2
    c0 = (W - acs_width) // 2
    m[r0:r0 + acs_width, c0:c0 + acs_width] = 1.0
    return m


def brain_mask_from_magnitude(mag: np.ndarray, rel_thresh: float = None) -> np.ndarray:
    """Deterministic brain mask: Otsu threshold + morphological cleanup.

    Otsu is used instead of a fixed fraction of the 99th percentile because at
    0.3 T the noisy air background sits close to a low relative threshold: the
    old ``0.08 * p99`` rule labelled ~97% of the FOV as brain (vs ~27% here),
    which silently made brain-masked metrics identical to full-FOV metrics.

    ``rel_thresh`` (optional) overrides Otsu with the legacy relative rule.
    """
    from scipy import ndimage
    if mag.size == 0 or not np.isfinite(mag).any() or mag.max() <= 0:
        return np.zeros_like(mag, dtype=bool)
    if rel_thresh is not None:
        thr = rel_thresh * np.percentile(mag, 99.0)
    else:
        from skimage.filters import threshold_otsu
        try:
            thr = threshold_otsu(mag)
        except Exception:
            thr = 0.1 * np.percentile(mag, 99.0)
    mask = mag > thr
    mask = ndimage.binary_opening(mask, iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    # largest connected component
    lbl, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
        keep = int(np.argmax(sizes)) + 1
        mask = lbl == keep
    return mask.astype(bool)
