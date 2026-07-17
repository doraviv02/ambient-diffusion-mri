#!/usr/bin/env python
"""Classical reconstruction baselines (runbook M0 / M1).

M0: noise-weighted adjoint (zero-filled SENSE) of the merged fixed-budget data.
M1: multi-coil L1-wavelet SENSE (SigPy) of the same merged measurement.

Outputs use the same result schema as the diffusion methods so the aggregation
and figures treat every method identically.
"""

import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mri_fft import ifft2c_np, fft2c_np
from utils.multiview_mri import noise_weighted_merge
from solve_inverse_mv_adps import build_condition_operator
from analysis.metrics_common import compute_all_metrics

try:
    import sigpy.mri
    _HAVE_SIGPY = True
except Exception:
    _HAVE_SIGPY = False


def natural_merge(sample, masks_bundle):
    """Noise-weighted merge of the two R=8 views in the NATURAL convention (for SigPy)."""
    cond = masks_bundle["conditions"]["merge_fixed"]  # [V,1,H,W]
    V = cond.shape[0]
    # take only as many repetitions as the condition needs (datasets may store more)
    ksp = sample["ksp_views"][:V]                   # [V,C,H,W]
    maps = sample["s_maps_views"][0].numpy()        # [C,H,W] (shared)
    noise = sample["noise_std"][:V]
    masked = cond * ksp
    merged = noise_weighted_merge(masked, cond, noise)
    return merged["ksp"][0].numpy(), merged["mask"][0].numpy(), maps  # [C,H,W],[1,H,W],[C,H,W]


def adjoint_recon(merged_ksp, maps):
    coil_imgs = ifft2c_np(merged_ksp)               # [C,H,W]
    return np.sum(np.conj(maps) * coil_imgs, axis=0)  # [H,W] complex


def l1wavelet_recon(merged_ksp, maps, mask, lamda, iters):
    if not _HAVE_SIGPY:
        raise RuntimeError("sigpy not available for L1-wavelet SENSE")
    weights = mask.astype(np.float32)               # [1,H,W]
    app = sigpy.mri.app.L1WaveletRecon(
        merged_ksp.astype(np.complex64), maps.astype(np.complex64),
        lamda=lamda, weights=weights, max_iter=iters, show_pbar=False)
    return np.asarray(app.run())                     # [H,W] complex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="selected_classical.yaml (for M1 lambda)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--methods", default="M0,M1")
    ap.add_argument("--lam", type=float, default=None, help="override L1-wavelet lambda")
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lam = args.lam
    if lam is None and args.config and os.path.exists(args.config):
        with open(args.config) as f:
            lam = yaml.safe_load(f).get("lambda", 3e-4)
    if lam is None:
        lam = 3e-4
    methods = args.methods.split(",")

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_id]
    if args.max_samples:
        rows = rows[: args.max_samples]
    os.makedirs(args.output_dir, exist_ok=True)

    n = 0
    for row in rows:
        masks_path = os.path.join(os.path.dirname(args.manifest), row["split"], row["mask_file"])
        masks_bundle = torch.load(masks_path, map_location="cpu", weights_only=False)
        sample = torch.load(masks_bundle["processed_path"], map_location="cpu", weights_only=False)
        meta = sample["metadata"]
        ref = sample.get("reference_mag")
        ref_np = ref.numpy() if ref is not None else None
        merged_ksp, merged_mask, maps = natural_merge(sample, masks_bundle)
        # fftmod-convention merged operator for consistent k-space error metrics
        mv_op, y_views, _ = build_condition_operator(sample, masks_bundle, "merge_fixed", device, 1e-8)

        for mname in methods:
            if mname == "MF":
                # Plain reconstruction of ONE complete k-space measurement (R=1,
                # single repetition) -- the "single full scan" comparison target.
                out_name = f"MF__{meta['subject_id']}_sl{meta['slice_id']:02d}_seed0.pt"
                out_path = os.path.join(args.output_dir, out_name)
                if os.path.exists(out_path):
                    continue
                t0 = time.time()
                k0 = sample["ksp_views"][0].numpy()          # natural, fully sampled
                mps0 = sample["s_maps_views"][0].numpy()
                recon = adjoint_recon(k0, mps0)
                runtime = time.time() - t0
                op_f, y_f, _ = build_condition_operator(sample, masks_bundle, "single_full",
                                                        device, 1e-8)
                metrics = compute_all_metrics(recon.astype(np.complex64), sample, op_f, y_f,
                                              device, ref_np)
                torch.save({
                    "method": "MF", "condition": "single_full", "checkpoint": None,
                    "reconstruction": recon.astype(np.complex64), "reference": ref_np,
                    "subject_id": meta["subject_id"], "slice_id": meta["slice_id"], "seed": 0,
                    "num_steps": 0, "l_ss": None, "likelihood_type": None, "lambda": None,
                    "runtime_seconds": runtime, "num_net_evals": 0, "metrics": metrics,
                    "normalization_scale": meta.get("normalization_scale"),
                    "mask_path": masks_path,
                }, out_path)
                n += 1
                print(f"[MF] {meta['subject_id']} sl{meta['slice_id']} "
                      f"ssim={metrics.get('ssim', float('nan')):.3f}", flush=True)
                continue
            out_name = f"{mname}__{meta['subject_id']}_sl{meta['slice_id']:02d}_seed0.pt"
            out_path = os.path.join(args.output_dir, out_name)
            if os.path.exists(out_path):
                continue
            t0 = time.time()
            if mname == "M0":
                recon = adjoint_recon(merged_ksp, maps)
                cond_label = "merge_fixed"; used_lam = None
            elif mname == "M1":
                recon = l1wavelet_recon(merged_ksp, maps, merged_mask, lam, args.iterations)
                cond_label = "merge_fixed"; used_lam = lam
            else:
                raise ValueError(f"unknown classical method {mname}")
            runtime = time.time() - t0
            metrics = compute_all_metrics(recon.astype(np.complex64), sample, mv_op, y_views,
                                          device, ref_np)
            out = {
                "method": mname, "condition": cond_label, "checkpoint": None,
                "reconstruction": recon.astype(np.complex64), "reference": ref_np,
                "subject_id": meta["subject_id"], "slice_id": meta["slice_id"], "seed": 0,
                "num_steps": args.iterations if mname == "M1" else 0,
                "l_ss": None, "likelihood_type": None, "lambda": used_lam,
                "runtime_seconds": runtime, "num_net_evals": 0, "metrics": metrics,
                "normalization_scale": meta.get("normalization_scale"),
                "mask_path": masks_path,
            }
            torch.save(out, out_path)
            n += 1
            msg = f"[{mname}] {meta['subject_id']} sl{meta['slice_id']} t={runtime:.2f}s " \
                  f"heldoutK={metrics['heldout_kspace_err']:.3f}"
            if "ssim" in metrics:
                msg += f" ssim={metrics['ssim']:.3f}"
            print(msg, flush=True)
    print(f"DONE classical shard {args.shard_id}/{args.num_shards}: {n} recons")


if __name__ == "__main__":
    main()
