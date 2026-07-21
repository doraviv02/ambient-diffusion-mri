#!/usr/bin/env python
"""M4Raw multi-view preprocessing pipeline.

Produces one ``.pt`` per accepted T2 slice with two motion-screened,
phase-aligned repetitions as views, one shared ESPIRiT sensitivity-map set,
per-view k-space noise estimates and (val/test only) a reference magnitude.
Stored k-space/maps are in the *natural* centered convention; the reconstruction
loaders apply ``fftmod``.

Outputs under ``--output-root``::

    train/<subject>_sl<idx>.pt ...
    val/<subject>_sl<idx>.pt ...
    test/<subject>_sl<idx>.pt ...
    train_manifest.csv val_manifest.csv test_manifest.csv
    preprocessing_report.json
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import h5py
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mri_fft import ifft2c_np, rss, brain_mask_from_magnitude
from tools.inspect_m4raw import parse_subject_contrast_rep

try:
    import sigpy.mri
    _HAVE_SIGPY = True
except Exception:  # pragma: no cover
    _HAVE_SIGPY = False

from skimage.registration import phase_cross_correlation


# ---------------------------------------------------------------------------
def read_t2_files(root, split):
    """Return dict subject_id -> list of (rep_id, path) for T2 acquisitions."""
    groups = defaultdict(list)
    if not root or not os.path.isdir(root):
        return groups
    for dirpath, _dirs, files in os.walk(root):
        for fn in sorted(files):
            if not fn.endswith(".h5"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with h5py.File(path, "r") as hf:
                    acq = hf.attrs.get("acquisition")
            except Exception:
                continue
            stem = os.path.splitext(fn)[0]
            subj, contrast, rep = parse_subject_contrast_rep(stem, acq)
            if contrast != "T2":
                continue
            groups[subj].append((rep if rep is not None else 999, path))
    for subj in groups:
        groups[subj].sort(key=lambda t: t[0])
    return groups


def load_kspace(path):
    with h5py.File(path, "r") as hf:
        ksp = np.asarray(hf["kspace"][()])            # [S, C, H, W] complex
        rss_recon = np.asarray(hf["reconstruction_rss"][()]) if "reconstruction_rss" in hf else None
    return ksp.astype(np.complex64), (rss_recon.astype(np.float32) if rss_recon is not None else None)


def central_slice_indices(n_slices, policy):
    if isinstance(policy, str) and policy.startswith("central_"):
        k = int(policy.split("_")[1])
    else:
        k = int(policy)
    k = min(k, n_slices)
    start = (n_slices - k) // 2
    return list(range(start, start + k))


def estimate_global_phase(imgs_a, imgs_b, region):
    """Global scalar phase offset between two coil-image stacks over ``region``."""
    m = region[None].astype(bool)
    inner = np.sum(np.conj(imgs_a)[np.broadcast_to(m, imgs_a.shape)] *
                   imgs_b[np.broadcast_to(m, imgs_b.shape)])
    return float(np.angle(inner))


def residual_phase_std(imgs_ref, imgs_aligned, brain):
    """Weighted std of residual phase over HIGH-SNR voxels after global alignment.

    At 0.3 T the phase in low-signal voxels is dominated by noise, so a naive
    brain-wide std conflates noise with genuinely structured (spatially varying)
    residual phase.  We restrict to the upper-signal half of the brain, where
    phase is reliable, so this statistic responds to real phase structure rather
    than SNR.  The global scalar offset has already been removed by the caller.
    """
    combined = np.sum(np.conj(imgs_ref) * imgs_aligned, axis=0)   # [H, W]
    weight = np.abs(np.sum(np.conj(imgs_ref) * imgs_ref, axis=0))
    ph = np.angle(combined)
    if brain.sum() <= 0:
        return float("nan")
    thr = np.percentile(weight[brain], 50.0)
    hs = brain & (weight > thr)
    w = weight * hs
    if w.sum() <= 0:
        return float("nan")
    mean = np.average(ph, weights=w)
    var = np.average((np.angle(np.exp(1j * (ph - mean)))) ** 2, weights=w)
    return float(np.sqrt(var))


def normalized_xcorr(a, b, mask=None):
    a = a.astype(np.float64); b = b.astype(np.float64)
    if mask is not None:
        a = a[mask]; b = b[mask]
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def smoothed_xcorr(a, b, mask=None, sigma=1.5):
    """Structural NCC on lightly Gaussian-smoothed images (robust to per-pixel
    noise at low SNR).  Genuine misalignment is still caught independently by the
    phase-correlation translation estimate."""
    from scipy import ndimage
    a = ndimage.gaussian_filter(a.astype(np.float64), sigma)
    b = ndimage.gaussian_filter(b.astype(np.float64), sigma)
    return normalized_xcorr(a, b, mask)


def espirit_maps(ksp_multicoil, acs_width):
    """Shared ESPIRiT maps [C,H,W] from centered k-space; None on failure."""
    if not _HAVE_SIGPY:
        return None
    try:
        maps = sigpy.mri.app.EspiritCalib(
            ksp_multicoil, calib_width=acs_width, thresh=0.02, kernel_width=6,
            crop=0.8, max_iter=50, show_pbar=False).run()
        return np.asarray(maps).astype(np.complex64)
    except Exception:
        return None


def fallback_maps(imgs):
    """RSS-normalized coil images as a sensitivity fallback: S_c = I_c / sqrt(sum|I|^2)."""
    denom = np.sqrt((np.abs(imgs) ** 2).sum(axis=0, keepdims=True) + 1e-8)
    return (imgs / denom).astype(np.complex64)


def estimate_noise_repdiff(imgs0, imgs1, brain):
    """Per-view k-space noise std via repetition-difference MAD in the image
    background (air).  Under the orthonormal FFT the complex image-domain noise
    std equals the per-coil k-space noise std.  We use the image background
    rather than k-space corners because M4Raw uses circular k-space coverage
    (exact-zero corners), which would otherwise yield a degenerate estimate.
    ``imgs0``/``imgs1`` are phase-aligned complex coil images ``[C,H,W]``.
    """
    from scipy import ndimage
    bg = ~ndimage.binary_dilation(brain, iterations=4)
    if bg.sum() < 50:                                 # fall back to image corners
        bg = np.zeros_like(brain); m = max(4, brain.shape[0] // 16)
        bg[:m, :m] = bg[:m, -m:] = bg[-m:, :m] = bg[-m:, -m:] = True
    diff = (imgs0 - imgs1)[:, bg]                     # [C, npix]
    vals = np.concatenate([diff.real.ravel(), diff.imag.ravel()])
    mad = np.median(np.abs(vals - np.median(vals)))
    sigma_diff = 1.4826 * mad
    sigma = float(sigma_diff / np.sqrt(2.0))
    if not np.isfinite(sigma) or sigma <= 0:
        # last-resort: robust std of the background itself
        sigma = float(1.4826 * np.median(np.abs(vals - np.median(vals))) + 1e-6) or 1e-6
    return max(sigma, 1e-6)


def highfreq_region(shape_hw, frac=0.18):
    """Boolean [H,W] mask of the outer (high-frequency) k-space used for noise est."""
    H, W = shape_hw
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    r = np.sqrt(((yy - cy) / (H / 2)) ** 2 + ((xx - cx) / (W / 2)) ** 2)
    return r > (1.0 - frac) * np.sqrt(2)


def pad_to_multiple(arr, multiple, axes=(-2, -1)):
    """Center-pad the given axes to a multiple; return padded array + pad spec."""
    pads = [(0, 0)] * arr.ndim
    for ax in axes:
        n = arr.shape[ax]
        target = int(np.ceil(n / multiple) * multiple)
        total = target - n
        pads[ax % arr.ndim] = (total // 2, total - total // 2)
    if all(p == (0, 0) for p in pads):
        return arr, pads
    return np.pad(arr, pads, mode="constant"), pads


# ---------------------------------------------------------------------------
def process_subject(subj, reps, split, cfg, out_dir, report):
    """Process one subject's T2 repetitions; write accepted slices; return manifest rows."""
    rows = []
    if len(reps) < cfg["minimum_repetitions"]:
        report["skipped_subjects_few_reps"] += 1
        return rows

    # Load all repetition volumes.
    vols, rss_recons = [], []
    for rep_id, path in reps:
        ksp, rss_recon = load_kspace(path)
        vols.append((rep_id, ksp))
        rss_recons.append(rss_recon)
    n_slices = vols[0][1].shape[0]
    sl_idx = central_slice_indices(n_slices, cfg["slice_policy"])
    acs = cfg["acs_width"]

    for sidx in sl_idx:
        # per-rep coil images / k-space at this slice
        ksps = [v[sidx] for _, v in vols]                     # list of [C,H,W]
        imgs = [ifft2c_np(k) for k in ksps]                    # list of [C,H,W]
        rss_imgs = [rss(im, axis=0) for im in imgs]            # list of [H,W]
        brain = brain_mask_from_magnitude(rss_imgs[0])
        hf_mask = highfreq_region(rss_imgs[0].shape)

        # Select up to `num_views` repetitions: an anchor (view 0) plus every other
        # repetition that passes the motion + phase checks *against the anchor*.
        # num_views=2 reproduces the original pair behaviour.
        n = len(ksps)
        num_views = int(cfg.get("num_views", 2))
        chosen = None
        for a in range(n):
            accepted, extras = [a], []
            for b in range(n):
                if b == a or len(accepted) >= num_views:
                    continue
                shift, _, _ = phase_cross_correlation(
                    rss_imgs[a], rss_imgs[b], upsample_factor=10)
                trans = float(np.sqrt(shift[0] ** 2 + shift[1] ** 2))
                ncc = smoothed_xcorr(rss_imgs[a], rss_imgs[b], brain)
                if cfg["reject_motion"] and (
                        trans > cfg["max_translation_pixels"] or ncc < cfg["minimum_pair_correlation"]):
                    continue
                gphase = estimate_global_phase(imgs[a], imgs[b], brain)
                rp_std = residual_phase_std(imgs[a], imgs[b] * np.exp(-1j * gphase), brain)
                if cfg["phase_correction"] and rp_std > cfg.get("max_residual_phase_std", 0.6):
                    continue
                accepted.append(b); extras.append((gphase, trans, ncc, rp_std))
            if len(accepted) >= num_views:
                chosen = (accepted, extras)
                break

        if chosen is None:
            report["rejected_slices"] += 1
            continue
        accepted, extras = chosen
        a = accepted[0]
        # view 0 is the anchor; every other view is phase-aligned to it
        view_ksps = [ksps[a].copy()]
        view_imgs = [imgs[a]]
        for (b, (gphase, _t, _c, _r)) in zip(accepted[1:], extras):
            view_ksps.append(ksps[b] * np.exp(-1j * gphase))
            view_imgs.append(imgs[b] * np.exp(-1j * gphase))
        # per-slice diagnostics reported for the first partner (back-compat)
        gphase, trans, ncc, rp_std = extras[0]
        b = accepted[1]
        ksp0, ksp1 = view_ksps[0], view_ksps[1]
        imgs0, imgs1 = view_imgs[0], view_imgs[1]

        # shared ESPIRiT maps from the phase-aligned average k-space of all views
        ksp_avg = np.mean(np.stack(view_ksps, axis=0), axis=0)
        maps = espirit_maps(ksp_avg, acs)
        sens_method = "espirit"
        if maps is None or not np.isfinite(maps).all():
            maps = fallback_maps(np.mean(np.stack(view_imgs, axis=0), axis=0))
            sens_method = "rss_normalized_coil_images"
            report["fallback_maps"] += 1

        # noise estimate (equal-noise repetition-difference; same sigma for all views)
        sigma = estimate_noise_repdiff(imgs0, imgs1, brain)
        noise_std = np.full(len(view_ksps), sigma, dtype=np.float32)

        # normalization: percentile of the RSS reference magnitude
        norm_ref = rss(imgs0, axis=0)
        scale = float(np.percentile(norm_ref[brain] if brain.any() else norm_ref,
                                    cfg["normalization_percentile"]))
        scale = scale if scale > 0 else 1.0
        view_ksps = [k / scale for k in view_ksps]
        ksp0, ksp1 = view_ksps[0], view_ksps[1]
        noise_std = noise_std / scale

        # reference magnitude
        reference_mag = None
        reference_source = None
        if split in ("val", "test") and cfg["save_reference"]:
            if split == "val" and n >= 3:
                # unused third repetition as a noisy secondary reference
                unused = [i for i in range(n) if i not in (a, b)]
                ref_img = rss(imgs[unused[0]], axis=0)
                reference_source = f"held_out_rep_idx{unused[0]}"
            else:
                # test: multi-repetition averaged magnitude (released GT proxy)
                stack = [r for r in rss_recons if r is not None]
                if stack:
                    ref_img = np.mean([s[sidx] for s in stack], axis=0)
                    reference_source = "multi_rep_avg_rss"
                else:
                    ref_img = rss(imgs0, axis=0)
                    reference_source = "view0_rss"
            reference_mag = (ref_img / scale).astype(np.float32)

        # pad spatial dims to a multiple (usually a no-op for 256)
        V = len(view_ksps)
        stacked_ksp = np.stack(view_ksps, axis=0)              # [V,C,H,W]
        stacked_ksp, pads = pad_to_multiple(stacked_ksp, cfg["pad_multiple"])
        maps_p, _ = pad_to_multiple(maps, cfg["pad_multiple"])
        maps_stacked = np.stack([maps_p] * V, axis=0)          # shared -> [V,C,H,W]
        if reference_mag is not None:
            reference_mag, _ = pad_to_multiple(reference_mag, cfg["pad_multiple"])
        H, W = stacked_ksp.shape[-2:]
        full_mask = np.ones((V, 1, H, W), dtype=np.float32)

        sample = {
            "ksp_views": torch.from_numpy(stacked_ksp.astype(np.complex64)),
            "full_mask_views": torch.from_numpy(full_mask),
            "s_maps_views": torch.from_numpy(maps_stacked.astype(np.complex64)),
            "noise_std": torch.from_numpy(noise_std.astype(np.float32)),
            "reference_mag": (torch.from_numpy(reference_mag.astype(np.float32))
                              if reference_mag is not None else None),
            "metadata": {
                "split": split, "subject_id": subj, "slice_id": int(sidx),
                "contrast": "T2", "field_strength_t": 0.3,
                "repetition_ids": [int(reps[i][0]) for i in accepted],
                "num_views": int(len(view_ksps)),
                "normalization_scale": scale,
                "padding": pads,
                "translation_pixels": trans,
                "pair_correlation": ncc,
                "phase_correction": gphase,
                "residual_phase_std": rp_std,
                "sensitivity_method_used": sens_method,
                "reference_source": reference_source,
                "noise_std_unnormalized": float(sigma),
            },
        }
        fn = f"{subj}_sl{sidx:02d}.pt"
        torch.save(sample, os.path.join(out_dir, fn))
        report["accepted_slices"] += 1
        rows.append({
            "split": split, "subject_id": subj, "slice_id": sidx, "file": fn,
            "repetition_ids": f"{reps[a][0]}+{reps[b][0]}", "H": H, "W": W,
            "n_coils": stacked_ksp.shape[1], "noise_std": float(noise_std[0]),
            "translation_pixels": trans, "pair_correlation": ncc,
            "residual_phase_std": rp_std, "sensitivity_method_used": sens_method,
            "has_reference": reference_mag is not None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-root")
    ap.add_argument("--val-root")
    ap.add_argument("--test-root")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--max-subjects-train", type=int, default=60)
    ap.add_argument("--max-subjects-val", type=int, default=20)
    ap.add_argument("--max-subjects-test", type=int, default=25)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_root, exist_ok=True)
    report = defaultdict(int)
    report["config"] = cfg

    split_roots = {"train": args.train_root, "val": args.val_root, "test": args.test_root}
    max_subj = {"train": args.max_subjects_train, "val": args.max_subjects_val,
                "test": args.max_subjects_test}
    all_subjects = {}

    for split, root in split_roots.items():
        if not root:
            continue
        out_dir = os.path.join(args.output_root, split)
        os.makedirs(out_dir, exist_ok=True)
        groups = read_t2_files(root, split)
        subjects = sorted(groups.keys())[: max_subj[split]]
        all_subjects[split] = subjects
        manifest_rows = []
        for i, subj in enumerate(subjects):
            rows = process_subject(subj, groups[subj], split, cfg, out_dir, report)
            manifest_rows.extend(rows)
            print(f"[{split}] {i+1}/{len(subjects)} {subj}: {len(rows)} slices", flush=True)
        # write manifest
        mpath = os.path.join(args.output_root, f"{split}_manifest.csv")
        if manifest_rows:
            with open(mpath, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
                w.writeheader(); w.writerows(manifest_rows)
        report[f"{split}_slices"] = len(manifest_rows)
        report[f"{split}_subjects"] = len(subjects)

    # subject overlap sanity
    overlap = {}
    sp = list(all_subjects.keys())
    for i in range(len(sp)):
        for j in range(i + 1, len(sp)):
            inter = set(all_subjects[sp[i]]) & set(all_subjects[sp[j]])
            if inter:
                overlap[f"{sp[i]}&{sp[j]}"] = sorted(inter)
    report["subject_overlap"] = overlap
    total_candidate = report["accepted_slices"] + report["rejected_slices"]
    report["acceptance_rate"] = (report["accepted_slices"] / total_candidate
                                 if total_candidate else None)

    with open(os.path.join(args.output_root, "preprocessing_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\n=== preprocessing report ===")
    print(json.dumps({k: v for k, v in report.items() if k != "config"}, indent=2, default=str))


if __name__ == "__main__":
    main()
