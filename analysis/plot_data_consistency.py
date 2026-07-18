#!/usr/bin/env python
"""Data-consistency trajectory curves vs diffusion step (Section 17.3)."""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.plot_common import load_results, METHOD_LABELS

CURVE_METHODS = ["single_r4", "fixed_split_merge", "fixed_split_ft", "dup2_ft"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    res = load_results(args.results_root)

    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = 0
    for m in CURVE_METHODS:
        trajs = [d["dc_trajectory"] for (mm, s, sl, sd), d in res.items()
                 if mm == m and d.get("dc_trajectory")]
        trajs = [t for t in trajs if t and len(t) > 1]
        if not trajs:
            continue
        L = min(len(t) for t in trajs)
        arr = np.array([t[:L] for t in trajs])
        mean = arr.mean(0)
        ax.plot(np.arange(L), mean, label=f"{METHOD_LABELS.get(m, m)} (n={len(trajs)})")
        ax.fill_between(np.arange(L), arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0), alpha=0.15)
        plotted += 1
    ax.set_yscale("log")
    ax.set_xlabel("diffusion step"); ax.set_ylabel("normalized data fidelity (per-view)")
    ax.set_title("Data-consistency trajectory")
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no dc_trajectory saved\n(run inference with --save_dc_trajectory)",
                ha="center", va="center", transform=ax.transAxes)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.tight_layout(); fig.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
