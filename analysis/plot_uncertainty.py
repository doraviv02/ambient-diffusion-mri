#!/usr/bin/env python
"""Posterior uncertainty examples + calibration (Section 17.3).

Uses the multi-seed uncertainty runs (e.g. M5, seeds 0-3): posterior mean and
voxelwise std over seeds, error vs the reference, and an uncertainty/error
calibration plot with Spearman correlation.
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.plot_common import load_results, get_reference


def collect_posterior(res):
    """Group multi-seed recons by (method,subject,slice) -> stacked magnitudes."""
    groups = defaultdict(list)
    refs = {}
    for (m, subj, sl, seed), d in res.items():
        r = d.get("reconstruction")
        if r is None:
            continue
        groups[(m, subj, sl)].append(np.abs(np.asarray(r)))
        refs[(m, subj, sl)] = get_reference(d)
    return groups, refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    res = load_results(args.results_root)
    groups, refs = collect_posterior(res)
    # keep groups with >=2 seeds
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if not groups:
        print("No multi-seed groups for uncertainty; skipping.")
        # still emit placeholder figures
        for name in ["uncertainty_examples.png", "uncertainty_calibration.png"]:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.text(0.5, 0.5, "no multi-seed uncertainty runs", ha="center", va="center")
            ax.axis("off"); fig.savefig(os.path.join(args.output_dir, name), dpi=110)
            plt.close(fig)
        return

    items = list(groups.items())[:4]
    # ---- examples ----
    fig, axes = plt.subplots(len(items), 4, figsize=(9, 2.4 * len(items)), squeeze=False)
    all_std, all_err = [], []
    for r, ((m, subj, sl), mags) in enumerate(items):
        stack = np.stack(mags, 0)
        mean = stack.mean(0); std = stack.std(0)
        ref = refs[(m, subj, sl)]
        err = np.abs(mean - ref) if ref is not None else np.zeros_like(mean)
        vmax = np.percentile(ref, 99) if ref is not None else mean.max()
        panels = [("Reference", ref, "gray", vmax), ("Posterior mean", mean, "gray", vmax),
                  ("|error|", err, "magma", np.percentile(err, 99) if err.max() > 0 else 1),
                  ("Posterior std", std, "viridis", np.percentile(std, 99) if std.max() > 0 else 1)]
        for c, (title, img, cmap, vm) in enumerate(panels):
            ax = axes[r, c]
            if img is not None:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=vm)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{subj}\nsl{sl}", fontsize=8)
        if ref is not None:
            all_std.append(std.ravel()); all_err.append(err.ravel())
    fig.suptitle("Posterior uncertainty examples")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "uncertainty_examples.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- calibration ----
    fig, ax = plt.subplots(figsize=(6, 5))
    if all_std:
        s = np.concatenate(all_std); e = np.concatenate(all_err)
        # bin by posterior std
        nb = 12
        qs = np.quantile(s, np.linspace(0, 1, nb + 1))
        xs, ys = [], []
        for i in range(nb):
            mask = (s >= qs[i]) & (s <= qs[i + 1])
            if mask.sum() > 10:
                xs.append(s[mask].mean()); ys.append(e[mask].mean())
        ax.plot(xs, ys, "o-", label="binned mean")
        rho, _ = spearmanr(s, e)
        lim = max(max(xs) if xs else 1, max(ys) if ys else 1)
        ax.plot([0, lim], [0, lim], "k--", alpha=0.4, label="y=x")
        ax.set_xlabel("posterior std (binned)"); ax.set_ylabel("mean |error|")
        ax.set_title(f"Uncertainty calibration (Spearman rho={rho:.3f})")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no reference for calibration", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "uncertainty_calibration.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote uncertainty_examples.png and uncertainty_calibration.png ({len(items)} cases)")


if __name__ == "__main__":
    main()
