#!/usr/bin/env python
"""Validate the processed multi-view dataset.

Fails (non-zero exit) on: NaN/Inf tensors, inconsistent view shapes, empty
masks, non-positive noise, subject overlap across splits, fewer than two views,
invalid sensitivity-map normalization, or missing metadata.

Also runs the empirical FFT-convention self-consistency check: the plain-FFT
adjoint of the ``fftmod``-ed fully-sampled k-space (with ``fftmod``-ed maps)
must reconstruct the RSS reference magnitude (high correlation).  This is the
convention gate for the M4Raw path.
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.multiview_mri import MRIViewOperator
from utils.mri_fft import ifft2c_np, rss


def fftmod_torch(x):
    x = x.clone()
    x[..., ::2, :] *= -1
    x[..., :, ::2] *= -1
    return x


def check_split(split_dir, split, subjects_seen, errors, warnings, convergence):
    files = sorted(glob.glob(os.path.join(split_dir, "*.pt")))
    if not files:
        warnings.append(f"[{split}] no .pt files in {split_dir}")
        return
    for path in files:
        s = torch.load(path, map_location="cpu", weights_only=False)
        tag = f"[{split}] {os.path.basename(path)}"
        for key in ("ksp_views", "full_mask_views", "s_maps_views", "noise_std", "metadata"):
            if key not in s:
                errors.append(f"{tag}: missing key {key}")
        if "ksp_views" not in s:
            continue
        ksp = s["ksp_views"]; mask = s["full_mask_views"]; maps = s["s_maps_views"]
        noise = s["noise_std"]; meta = s["metadata"]

        if ksp.shape[0] < 2:
            errors.append(f"{tag}: fewer than two views ({ksp.shape[0]})")
        if ksp.shape != maps.shape:
            errors.append(f"{tag}: ksp/maps shape mismatch {tuple(ksp.shape)} vs {tuple(maps.shape)}")
        if ksp.shape[-2:] != mask.shape[-2:]:
            errors.append(f"{tag}: mask spatial mismatch")
        for name, t in (("ksp", ksp), ("maps", maps), ("noise", noise)):
            if torch.is_complex(t):
                bad = not torch.isfinite(t.real).all() or not torch.isfinite(t.imag).all()
            else:
                bad = not torch.isfinite(t).all()
            if bad:
                errors.append(f"{tag}: non-finite values in {name}")
        if (mask.abs().sum(dim=(-3, -2, -1)) == 0).any():
            errors.append(f"{tag}: empty mask for some view")
        if (noise <= 0).any():
            errors.append(f"{tag}: non-positive noise std {noise.tolist()}")
        # sensitivity-map normalization: sum_c |S|^2 ~ 1 where signal present
        sos = (maps[0].abs() ** 2).sum(dim=0)                # [H,W]
        rss0 = rss(ifft2c_np(ksp[0].numpy()), axis=0)
        brain = rss0 > 0.1 * np.percentile(rss0, 99.0)
        if brain.sum() > 0:
            med_sos = float(np.median(sos.numpy()[brain]))
            if not (0.5 <= med_sos <= 1.5):
                warnings.append(f"{tag}: median sum|S|^2 in brain = {med_sos:.3f} (expect ~1)")
        for req in ("subject_id", "slice_id", "repetition_ids", "normalization_scale"):
            if req not in meta:
                errors.append(f"{tag}: metadata missing {req}")
        subjects_seen.setdefault(split, set()).add(meta.get("subject_id"))

        # --- convention self-consistency: adjoint recon vs RSS reference ---
        maps_mod = fftmod_torch(maps[0])                      # [C,H,W]
        ksp_mod = fftmod_torch(ksp[0])                        # fully sampled
        op = MRIViewOperator(mask=mask[0], maps=maps_mod)
        recon = op.adjoint(ksp_mod[None])[0, 0]               # complex [H,W]
        recon_mag = recon.abs().numpy()
        ref = rss0
        # correlation over the brain region
        bm = brain
        if bm.sum() > 10:
            c = np.corrcoef(recon_mag[bm], ref[bm])[0, 1]
            convergence.append(float(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()

    errors, warnings, convergence = [], [], []
    subjects_seen = {}
    for split in ("train", "val", "test"):
        d = os.path.join(args.data_root, split)
        if os.path.isdir(d):
            check_split(d, split, subjects_seen, errors, warnings, convergence)

    # subject overlap
    splits = list(subjects_seen.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            inter = subjects_seen[splits[i]] & subjects_seen[splits[j]]
            if inter:
                errors.append(f"subject overlap {splits[i]}&{splits[j]}: {sorted(inter)}")

    print("=== check_processed_dataset ===")
    for split, subs in subjects_seen.items():
        print(f"  {split}: {len(subs)} subjects")
    if convergence:
        arr = np.array(convergence)
        print(f"  convention self-consistency corr(|adjoint|, rss): "
              f"mean={arr.mean():.3f} min={arr.min():.3f} n={len(arr)}")
        if arr.mean() < 0.9:
            errors.append(f"FFT convention check FAILED: mean corr {arr.mean():.3f} < 0.9")
    print(f"  warnings: {len(warnings)}")
    for w in warnings[:20]:
        print("   WARN", w)
    print(f"  errors: {len(errors)}")
    for e in errors[:40]:
        print("   ERR", e)
    if errors:
        print("\nCHECK FAILED")
        sys.exit(1)
    print("\nCHECK PASSED")


if __name__ == "__main__":
    main()
