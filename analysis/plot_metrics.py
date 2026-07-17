#!/usr/bin/env python
"""Subject-level metric boxplots + fixed-vs-extra budget plot (Section 17.3)."""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.plot_common import METHOD_LABELS, BUDGET_COEFFS, methods_for, design_of, DESIGN_COLORS

# Acquired-coefficient budget per method (dup-ACS accounting; columns*H) now comes
# from analysis.experiments so it covers every condition, not just M0-M8.
DEFAULT_BUDGET = BUDGET_COEFFS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="metrics_per_subject.csv")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--budget-json", default=None)
    ap.add_argument("--set", dest="set_", default="main", choices=["main", "quad"],
                    help="which evaluation set this table is")
    ap.add_argument("--suffix", default="", help="appended to output filenames")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.metrics)
    methods = [m for m in methods_for(args.set_) if m in df["method"].unique()]

    # ---- boxplots ----
    metric_cols = [c for c in ["nrmse", "ssim", "heldout_kspace_err"] if c in df.columns]
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(0.62 * len(methods) * len(metric_cols) + 3,
                                                          5.6))
    axes = np.atleast_1d(axes)
    for ax, mc in zip(axes, metric_cols):
        data = [df[df.method == m][mc].dropna().values for m in methods]
        # a method with an all-NaN column (e.g. held-out error for full sampling)
        # would make boxplot raise; keep the slot but draw nothing.
        parts = ax.boxplot([d if len(d) else [np.nan] for d in data],
                           showmeans=True, patch_artist=True)
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels([f"{m}\n{METHOD_LABELS.get(m, m)}" for m in methods], fontsize=7)
        # colour each box by its acquisition design -- the study's main axis
        for pc, m in zip(parts["boxes"], methods):
            pc.set_facecolor(DESIGN_COLORS[design_of(m)])
            pc.set_alpha(0.75)
        # faint paired subject lines
        piv = df.pivot_table(index="subject_id", columns="method", values=mc)
        for _, rowv in piv.iterrows():
            xs, ys = [], []
            for i, m in enumerate(methods):
                if m in piv.columns and np.isfinite(rowv.get(m, np.nan)):
                    xs.append(i + 1); ys.append(rowv[m])
            if len(xs) > 1:
                ax.plot(xs, ys, color="gray", alpha=0.15, lw=0.6)
        ax.set_title(mc)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.suptitle("Subject-level metrics (box = IQR, triangle = mean, faint lines = paired subjects)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, f"metric_boxplots{args.suffix}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- fixed vs extra budget ----
    budget = dict(DEFAULT_BUDGET)
    if args.budget_json and os.path.exists(args.budget_json):
        import json
        try:
            b = json.load(open(args.budget_json))
            any_split = next(iter(b.values()))
            budget["M6"] = any_split.get("extra_sum", budget["M6"])  # noqa
            for m in ["M0", "M1", "M2", "M3", "M4", "M5"]:
                budget[m] = any_split.get("fixed_sum_dupACS", budget[m])
        except Exception:
            pass
    ycols = [c for c in ["ssim", "heldout_kspace_err"] if c in df.columns]
    fig, axes = plt.subplots(1, len(ycols), figsize=(5.5 * len(ycols), 4.5))
    axes = np.atleast_1d(axes)
    for ax, yc in zip(axes, ycols):
        for m in methods:
            g = df[df.method == m][yc].dropna()
            if len(g):
                ax.errorbar(budget.get(m, np.nan), g.mean(), yerr=g.std(),
                            fmt="o", capsize=3, label=METHOD_LABELS.get(m, m))
                ax.annotate(m, (budget.get(m, np.nan), g.mean()), fontsize=8,
                            textcoords="offset points", xytext=(5, 4))
        ax.set_xlabel("acquired coefficients (columns x H, dup-ACS)")
        ax.set_ylabel(yc); ax.set_title(f"{yc} vs acquisition budget")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, f"fixed_vs_extra_budget{args.suffix}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote metric_boxplots{args.suffix}.png and fixed_vs_extra_budget{args.suffix}.png")


if __name__ == "__main__":
    main()
