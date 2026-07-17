#!/usr/bin/env python
"""Subject-level aggregation with bootstrap CIs (runbook Section 17.2).

Slices are first averaged within subject; subjects are the resampling units for
the 95% bootstrap confidence interval.  Slices are never treated as independent.
"""

import argparse
import os

import numpy as np
import pandas as pd

METRIC_COLS = ["nrmse", "psnr", "ssim", "measured_kspace_err", "heldout_kspace_err",
               "runtime_seconds", "num_net_evals", "heldout_kspace_err_allcoef",
               # full-FOV variants kept for transparency (brain-masked are primary)
               "ssim_fullfov", "nrmse_fullfov", "psnr_fullfov", "brain_fraction"]


def bootstrap_ci(values, n_boot, seed, alpha=0.05):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.RandomState(seed)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean()
                      for _ in range(n_boot)])
    return float(values.mean()), float(np.percentile(means, 100 * alpha / 2)), \
        float(np.percentile(means, 100 * (1 - alpha / 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--subject-output", required=True)
    ap.add_argument("--summary-output", required=True)
    ap.add_argument("--bootstrap-samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260716)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    # For the uncertainty/multi-seed case, average seeds first per slice.
    metrics = [m for m in METRIC_COLS if m in df.columns]
    slice_df = df.groupby(["method", "subject_id", "slice_id"], as_index=False)[metrics].mean()
    # average slices within subject
    subj_df = slice_df.groupby(["method", "subject_id"], as_index=False)[metrics].mean()
    os.makedirs(os.path.dirname(os.path.abspath(args.subject_output)), exist_ok=True)
    subj_df.to_csv(args.subject_output, index=False)

    rows = []
    for method, g in subj_df.groupby("method"):
        row = {"method": method, "n_subjects": g["subject_id"].nunique()}
        for m in metrics:
            mean, lo, hi = bootstrap_ci(g[m].values, args.bootstrap_samples, args.seed)
            row[f"{m}_mean"] = mean
            row[f"{m}_median"] = float(np.nanmedian(g[m].values))
            row[f"{m}_std"] = float(np.nanstd(g[m].values))
            row[f"{m}_ci_lo"] = lo
            row[f"{m}_ci_hi"] = hi
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.summary_output, index=False)
    print(f"Wrote {args.subject_output} ({len(subj_df)} subject rows)")
    print(f"Wrote {args.summary_output} ({len(summary)} methods)")
    show = [c for c in ["method", "n_subjects", "ssim_mean", "ssim_ci_lo", "ssim_ci_hi",
                        "heldout_kspace_err_mean", "nrmse_mean"] if c in summary.columns]
    print(summary[show].round(4).to_string(index=False))

    # paired comparisons (subject-level) between key method pairs, if present.
    pairs = [("M3", "M4"), ("M4", "M5"), ("M5", "M6"), ("M2", "M4"), ("M1", "M4")]
    piv = subj_df.pivot_table(index="subject_id", columns="method", values="heldout_kspace_err")
    print("\nPaired held-out k-space error (subject-level, mean diff a-b, frac a<b):")
    for a, b in pairs:
        if a in piv.columns and b in piv.columns:
            d = (piv[a] - piv[b]).dropna()
            if len(d):
                print(f"  {a}-{b}: n={len(d)} mean_diff={d.mean():+.4f} frac({a}<{b})={float((d<0).mean()):.2f}")


if __name__ == "__main__":
    main()
