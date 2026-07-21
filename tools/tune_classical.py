#!/usr/bin/env python
"""Tune the L1-wavelet SENSE lambda on the validation subset (runbook Section 14).

Selection: validation held-out k-space error, SSIM as tie-breaker.  Because
SigPy's regularization is scaled differently from the runbook's nominal grid, a
wider log-spaced grid is used and the final grid is recorded.
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.run_classical_recon import natural_merge, l1wavelet_recon
from analysis.metrics_common import compute_all_metrics
from solve_inverse_mv_adps import build_condition_operator
from tools.tune_inference import select_subset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n-subjects", type=int, default=6)
    ap.add_argument("--lambda-grid", default="1e-6,1e-5,1e-4,3e-4,1e-3,1e-2")
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subset, mask_root = select_subset(args.manifest, args.n_subjects)
    lam_grid = [float(x) for x in args.lambda_grid.split(",")]

    rows = []
    for lam in lam_grid:
        hk, ss = [], []
        for r in subset:
            mb = torch.load(os.path.join(mask_root, r["split"], r["mask_file"]),
                            map_location="cpu", weights_only=False)
            sample = torch.load(mb["processed_path"], map_location="cpu", weights_only=False)
            merged_ksp, merged_mask, maps = natural_merge(sample, mb)
            recon = l1wavelet_recon(merged_ksp, maps, merged_mask, lam, args.iterations)
            mv_op, y_views, _ = build_condition_operator(sample, mb, "merge_fixed", device, 1e-8)
            ref = sample.get("reference_mag")
            m = compute_all_metrics(recon.astype(np.complex64), sample, mv_op, y_views, device,
                                    ref.numpy() if ref is not None else None)
            hk.append(m["heldout_kspace_err"]); ss.append(m.get("ssim", np.nan))
        row = {"lambda": lam, "heldout_kspace_err": float(np.mean(hk)),
               "ssim": float(np.nanmean(ss)), "n": len(subset)}
        rows.append(row)
        print(f"lambda={lam:.1e}: heldoutK={row['heldout_kspace_err']:.4f} ssim={row['ssim']:.4f}",
              flush=True)

    best = sorted(rows, key=lambda r: (round(r["heldout_kspace_err"], 4), -round(r["ssim"], 4)))[0]
    out = args.output or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "configs", "project", "selected_classical.yaml")
    with open(out, "w") as f:
        yaml.safe_dump({"lambda": best["lambda"], "iterations": args.iterations,
                        "grid": lam_grid, "selection": "val_heldout_kspace_error"}, f, sort_keys=False)
    # also record the grid table
    tbl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "tables", "project", "classical_lambda_grid.csv")
    os.makedirs(os.path.dirname(tbl), exist_ok=True)
    with open(tbl, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSelected lambda={best['lambda']:.1e}; wrote {out} and {tbl}")


if __name__ == "__main__":
    main()
