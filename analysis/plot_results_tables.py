#!/usr/bin/env python
"""Bar-chart figures for the results tables in reports/mvp_summary.md.

One grouped figure per report *section* (so the panels match how the tables are
presented together):
  results_2view.png : the 2-view section's two sub-tables (fixed budget; 2x budget)
  results_4view.png : the 4-view section's single table

Each table becomes a row of two horizontal-bar panels (SSIM, NRMSE) with 95%
bootstrap CIs, bars coloured by acquisition design. Numbers come straight from the
summary_metrics CSVs so the figures and the tables cannot disagree.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.experiments import EXPERIMENTS, DESIGN_COLORS, design_of

DESIGN_LABEL = {
    "1x": "one cheap scan", "fixed": "fixed budget (split one scan)",
    "dup": "extra scans, duplicated mask", "comp": "extra scans, complementary (shared ACS)",
    "fcomp": "extra scans, fully-complementary (split ACS)", "full": "one complete measurement",
    "dualfull": "two complete measurements (NEX=2)",
}

# The three tables exactly as they appear in mvp_summary.md.
T_FIXED = ["classical_adjoint", "classical_l1wav", "single_r4", "fixed_split_merge", "fixed_split_ft"]
T_2X = ["dup2_merge", "dup2_ft", "comp2_merge", "comp2_ft", "fcomp2_merge", "fcomp2_ft"]
T_QUAD = ["single_r4", "dup4_merge", "dup4_ft", "comp4_merge", "comp4_ft",
          "fcomp4_merge", "fcomp4_ft", "full_diffusion", "full_plain",
          "dualfull_merge", "dualfull_ft"]


def _row(summ, m, col):
    r = summ[summ.method == m]
    if r.empty or f"{col}_mean" not in r:
        return None
    v = r.iloc[0]
    return (float(v[f"{col}_mean"]), float(v[f"{col}_ci_lo"]), float(v[f"{col}_ci_hi"]))


def _panel(ax, summ, methods, col, better_high):
    ys = np.arange(len(methods))
    means, lo, hi, chis, colors, ylabels = [], [], [], [], [], []
    for m in methods:
        e = EXPERIMENTS[m]
        rr = _row(summ, m, col)
        mean, clo, chi = rr if rr else (np.nan, np.nan, np.nan)
        means.append(mean); lo.append(mean - clo); hi.append(chi - mean); chis.append(chi)
        colors.append(DESIGN_COLORS[design_of(m)])
        ylabels.append(f"{m}\n({e.cov:.0%} cov)")
    ax.barh(ys, means, xerr=[lo, hi], color=colors, edgecolor="black", linewidth=0.4,
            error_kw=dict(ecolor="#333", lw=1.0, capsize=2.5), height=0.72, zorder=3)
    # mark fine-tuned methods with a ring cue
    for y, m, mean in zip(ys, methods, means):
        if EXPERIMENTS[m].prior == "finetuned" and np.isfinite(mean):
            ax.barh(y, mean, height=0.72, fill=False, edgecolor="black", linewidth=1.6, zorder=4)
    ax.set_yticks(ys); ax.set_yticklabels(ylabels, fontsize=7.5)
    ax.invert_yaxis()
    finite = [v for v in means if np.isfinite(v)]
    span = (max(finite) - min(finite)) or 1.0
    lopad = min(finite) - span * 0.28
    hipad = max(finite) + span * 0.42          # room for value labels past the CI caps
    ax.set_xlim(max(0, lopad), hipad)
    for y, chi in zip(ys, chis):               # label just past the upper CI cap
        if np.isfinite(chi):
            ax.text(chi + span * 0.03, y, f"{means[int(y)]:.3f}", va="center", fontsize=7.5)
    arrow = "higher = better" if better_high else "lower = better"
    ax.set_xlabel(f"{col.upper()}  ({arrow})", fontsize=9)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.tick_params(axis="y", length=0)


def _legend(fig, methods_used):
    designs = []
    for m in methods_used:
        d = design_of(m)
        if d not in designs:
            designs.append(d)
    handles = [plt.Rectangle((0, 0), 1, 1, color=DESIGN_COLORS[d]) for d in designs]
    labels = [DESIGN_LABEL.get(d, d) for d in designs]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="white", ec="black", lw=1.6))
    labels.append("cross-view fine-tuned")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5)


def fig_2view(summ, out):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2),
                             gridspec_kw=dict(height_ratios=[5, 6]))
    _panel(axes[0, 0], summ, T_FIXED, "ssim", True)
    _panel(axes[0, 1], summ, T_FIXED, "nrmse", False)
    _panel(axes[1, 0], summ, T_2X, "ssim", True)
    _panel(axes[1, 1], summ, T_2X, "nrmse", False)
    axes[0, 0].set_title("Table 1 — classical baselines & fixed budget (64 lines)",
                         fontsize=10.5, fontweight="bold", loc="left")
    axes[1, 0].set_title("Table 2 — 2x budget (128 lines): duplicated vs complementary vs fully-comp",
                         fontsize=10.5, fontweight="bold", loc="left")
    fig.suptitle("2-view set results (25 subjects, 75 slices)  —  subject mean ± 95% CI",
                 fontsize=13, fontweight="bold")
    _legend(fig, T_FIXED + T_2X)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def fig_4view(summ, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6))
    _panel(axes[0], summ, T_QUAD, "ssim", True)
    _panel(axes[1], summ, T_QUAD, "nrmse", False)
    fig.suptitle("4-view set results (20 subjects, 59 slices)  —  subject mean ± 95% CI\n"
                 "the budget question: can 4 cheap R=4 scans match one full scan? "
                 "(held-out k-err is n/a at full coverage)",
                 fontsize=12.5, fontweight="bold")
    _legend(fig, T_QUAD)
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--summary-quad", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    fig_2view(pd.read_csv(args.summary), os.path.join(args.output_dir, "results_2view.png"))
    fig_4view(pd.read_csv(args.summary_quad), os.path.join(args.output_dir, "results_4view.png"))


if __name__ == "__main__":
    main()
