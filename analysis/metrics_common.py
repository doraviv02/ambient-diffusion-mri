"""Shared metric computation for diffusion and classical reconstructions.

Both method families produce a *natural* complex image, so the same
``fftmod``-convention operator scores them identically.  ``compute_all_metrics``
returns image-domain metrics (vs a reference) plus normalized measured/held-out
k-space errors, so every method in the experiment is measured the same way.
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from torch_utils.ambient_diffusion import nrmse_np, psnr
from utils.multiview_mri import MRIViewOperator, complex_to_channels


def fftmod(x):
    x = x.clone()
    x[..., ::2, :] *= -1
    x[..., :, ::2] *= -1
    return x


def compute_image_metrics(recon_mag, ref_mag, brain_mask=None):
    """Image metrics, restricted to a brain mask when given.

    At 0.3 T the air background is pure noise in the reconstruction but is
    averaged-down in the multi-repetition reference, so full-FOV SSIM is
    dominated by structurally uncorrelated noise and understates quality by
    ~0.2. The mask is derived from the *reference*, so it is identical for every
    method on a given slice. Full-FOV values are also reported for transparency.
    """
    if ref_mag is None:
        return {}
    ref = np.asarray(ref_mag, dtype=np.float64)
    est = np.asarray(recon_mag, dtype=np.float64)
    dr = ref.max() - ref.min()
    dr = dr if dr > 0 else 1.0

    ssim_full, ssim_map = ssim(ref, est, data_range=dr, full=True)
    out = {
        "nrmse_fullfov": float(nrmse_np(ref, est)),
        "psnr_fullfov": float(psnr(gt=ref, est=est, max_pixel=ref.max())),
        "ssim_fullfov": float(ssim_full),
    }
    if brain_mask is None or not brain_mask.any():
        out.update(nrmse=out["nrmse_fullfov"], psnr=out["psnr_fullfov"],
                   ssim=out["ssim_fullfov"], brain_fraction=1.0)
        return out

    m = brain_mask.astype(bool)
    r, e = ref[m], est[m]
    num = float(np.linalg.norm(r - e)); den = float(np.linalg.norm(r))
    mse = float(np.mean((r - e) ** 2))
    out.update(
        ssim=float(ssim_map[m].mean()),
        nrmse=(num / den if den > 0 else float("nan")),
        psnr=(20 * np.log10(r.max() / np.sqrt(mse)) if mse > 0 else float("inf")),
        brain_fraction=float(m.mean()),
    )
    return out


def measured_kspace_error(mv_op, recon_ch, y_views):
    """Normalized error on the coefficients used in the fit."""
    with torch.no_grad():
        pred = mv_op.forward(recon_ch)
        yv = y_views[None] if y_views.dim() == 4 else y_views
        num = (pred - yv).abs().pow(2).sum().sqrt()
        den = yv.abs().pow(2).sum().sqrt().clamp_min(1e-12)
    return float(num / den)


def heldout_kspace_error(sample, recon_ch, device, heldout_mask=None):
    """Normalized error on k-space coefficients the method never observed.

    ``heldout_mask`` ([1,H,W] or [H,W]) should be the complement of the union of
    the condition's sampling masks. This matters: scoring over *all* coefficients
    (the old behaviour) includes the ones the method fitted, which rewards a
    reconstruction for reproducing view-0's particular noise realization. That
    biases the metric toward single-view methods and penalizes multi-view methods
    that average noise away. Restricting to unobserved coefficients removes the
    bias -- there, every method faces the same noise floor and a genuinely better
    (less noisy) reconstruction scores lower.

    Ground truth is view-0's full k-space (it contains noise, so the error floors
    at the inverse SNR for every method equally).
    """
    ksp0 = fftmod(sample["ksp_views"][0:1].to(device))
    maps0 = fftmod(sample["s_maps_views"][0:1].to(device))
    full_op = MRIViewOperator(mask=torch.ones_like(ksp0[:, :1].real), maps=maps0[0])
    with torch.no_grad():
        pred_full = full_op.forward(recon_ch.to(device))
        diff = pred_full - ksp0
        if heldout_mask is not None:
            hm = heldout_mask.to(device)
            if hm.dim() == 2:
                hm = hm[None]
            hm = hm[None] if hm.dim() == 3 else hm          # -> [1,1,H,W]
            diff = diff * hm
            ref = ksp0 * hm
        else:
            ref = ksp0
        num = diff.abs().pow(2).sum().sqrt()
        den = ref.abs().pow(2).sum().sqrt().clamp_min(1e-12)
    return float(num / den)


def compute_all_metrics(recon_cplx_hw, sample, mv_op, y_views, device, ref_mag=None):
    """recon_cplx_hw: complex numpy [H,W]. Returns a metrics dict."""
    recon_ch = complex_to_channels(
        torch.from_numpy(recon_cplx_hw)[None, None].to(device))  # [1,2,H,W]
    m = compute_image_metrics(np.abs(recon_cplx_hw), ref_mag,
                              brain_mask_for(ref_mag))
    m["measured_kspace_err"] = measured_kspace_error(mv_op, recon_ch, y_views)
    m["heldout_kspace_err"] = heldout_kspace_error(sample, recon_ch, device)
    return m


def brain_mask_for(ref_mag):
    """Deterministic brain mask from the reference (same for all methods)."""
    if ref_mag is None:
        return None
    from utils.mri_fft import brain_mask_from_magnitude
    try:
        return brain_mask_from_magnitude(np.asarray(ref_mag, dtype=np.float64))
    except Exception:
        return None
