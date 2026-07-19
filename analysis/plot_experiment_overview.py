#!/usr/bin/env python
"""Cross-experiment figures: coverage, effect sizes, budget ladder, fine-tune decay.

These are the figures that carry the study's conclusions, as opposed to the
per-method montages.  Everything is computed from the per-subject metric tables
so the plots and the report cannot disagree.

Outputs (figures/mvp/):
  effect_sizes_forest.png     paired effects, acquisition design vs recon method
  budget_ladder.png           SSIM vs lines bought, duplicated vs complementary
  finetuning_vs_coverage.png  cross-view fine-tuning benefit decaying with coverage
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
from analysis.experiments import EXPERIMENTS, DESIGN_COLORS, design_of, methods_for

DESIGN_LABEL = {
    "fixed": "fixed budget (split one scan)",
    "1x": "one cheap scan",
    "dup": "extra scans, DUPLICATED mask",
    "comp": "extra scans, COMPLEMENTARY (shared ACS)",
    "fcomp": "extra scans, FULLY-COMP (split ACS, no overlap)",
    "full": "one complete measurement",
    "dualfull": "TWO complete measurements (NEX=2)",
}


def paired(df, a, b, col="ssim", n_boot=10000, seed=0):
    A = df[df.method == a].set_index("subject_id")[col]
    B = df[df.method == b].set_index("subject_id")[col]
    idx = A.index.intersection(B.index)
    if len(idx) == 0:
        return None
    d = (A.loc[idx] - B.loc[idx]).values
    rng = np.random.RandomState(seed)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n_boot)])
    return dict(delta=d.mean(), lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5),
                win=float((d > 0).mean()), n=len(d))


def subj_mean(df, m, col="ssim"):
    v = df[df.method == m][col]
    return float(v.mean()) if len(v) else None



# ---------------------------------------------------------------------------
# 2. Forest plot of paired effects
# ---------------------------------------------------------------------------
def fig_forest(main, quad, out):
    rows = []  # (label, result, group)
    D, M = "acquisition design", "reconstruction method"
    rows.append(("COMP vs dup, 2x, fine-tuned", paired(main, "comp2_ft", "dup2_ft"), D))
    rows.append(("COMP vs dup, 2x, merged", paired(main, "comp2_merge", "dup2_merge"), D))
    rows.append(("COMP vs dup, 4x", paired(quad, "comp4_ft", "dup4_ft"), D))
    rows.append(("extra cheap repetition (dup2-single)", paired(main, "dup2_merge", "single_r4"), D))
    rows.append(("4x COMP vs one full scan, same recon", paired(quad, "comp4_ft", "full_diffusion"), D))

    # Cross-view fine-tuning re-anchored to the merge baseline of each family
    # (merge == the removed original-prior joint, so this is the fine-tuning effect).
    rows.append(("cross-view FT @ 19% coverage", paired(main, "fixed_split_ft", "fixed_split_merge"), M))
    rows.append(("cross-view FT @ 25% coverage", paired(main, "dup2_ft", "dup2_merge"), M))
    rows.append(("cross-view FT @ 44% coverage", paired(main, "comp2_ft", "comp2_merge"), M))
    rows.append(("cross-view FT @ 81% coverage", paired(quad, "comp4_ft", "comp4_merge"), M))

    rows = [r for r in rows if r[1] is not None]
    fig, ax = plt.subplots(figsize=(11, 0.46 * len(rows) + 2.4))
    ypos = np.arange(len(rows))[::-1]
    for y, (lab, r, grp) in zip(ypos, rows):
        c = "#2a7f3f" if grp == "acquisition design" else "#8172b2"
        ax.plot([r["lo"], r["hi"]], [y, y], color=c, lw=2.6, solid_capstyle="round")
        ax.scatter(r["delta"], y, color=c, s=95, zorder=3)
        ax.text(r["hi"] + 0.002, y, f"  {r['delta']:+.3f}  ({r['win']:.0%} of {r['n']})",
                va="center", fontsize=8.5, color="#333")
    ax.axvline(0, color="black", lw=1.2, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_xlabel("paired SSIM difference (95% bootstrap CI over subjects)")
    ax.set_xlim(right=max(r["hi"] for _, r, _ in rows) * 1.55)
    n_design = sum(1 for r in rows if r[2] == "acquisition design")
    ax.axhline(len(rows) - n_design - 0.5, color="gray", lw=1, alpha=0.6)
    ax.text(0.985, 0.975, "ACQUISITION DESIGN", transform=ax.transAxes, ha="right",
            color="#2a7f3f", fontweight="bold", fontsize=10.5)
    ax.text(0.985, 0.40, "RECONSTRUCTION METHOD", transform=ax.transAxes, ha="right",
            color="#8172b2", fontweight="bold", fontsize=10.5)
    ax.set_title("Acquisition vs Reconstruction contributions to SSIM",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# 3. Budget ladder: what you get per line bought
# ---------------------------------------------------------------------------
def fig_budget(main, quad, out):
    """Every method as a bar, grouped by what it cost. Within a budget tier the
    only thing that varies is the design + method, so the tier makes the
    like-for-like comparison explicit."""
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 10.5))
    for ax, (df, set_, title) in zip(axes, [(main, "main", "2-view set (25 subjects, 75 slices)"),
                                            (quad, "quad", "4-view set (20 subjects, 59 slices)")]):
        present = [m for m in methods_for(set_) if not df[df.method == m].empty]
        # order: cheapest tier first, then by design, then by score
        present.sort(key=lambda m: (EXPERIMENTS[m].lines, EXPERIMENTS[m].cov, subj_mean(df, m)))
        xs, seen, gap = [], None, 0
        for m in present:
            tier = EXPERIMENTS[m].lines
            if seen is not None and tier != seen:
                gap += 1.0
            seen = tier
            xs.append(len(xs) + gap)
        vals = [subj_mean(df, m) for m in present]
        errs = [df[df.method == m]["ssim"].sem() * 1.96 for m in present]
        cols = [DESIGN_COLORS[design_of(m)] for m in present]
        ax.bar(xs, vals, color=cols, yerr=errs, capsize=3, width=0.82,
               edgecolor=["black" if EXPERIMENTS[m].prior == "finetuned" else "none"
                          for m in present],
               linewidth=1.8)
        for x, v, m in zip(xs, vals, present):
            ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=8, rotation=90)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{m}\n{EXPERIMENTS[m].label}" for m in present],
                           fontsize=7.6, rotation=38, ha="right")
        # tier separators + labels
        for tier in sorted({EXPERIMENTS[m].lines for m in present}):
            idx = [i for i, m in enumerate(present) if EXPERIMENTS[m].lines == tier]
            mid = (xs[idx[0]] + xs[idx[-1]]) / 2
            ax.text(mid, max(vals) * 1.14, f"{tier} lines acquired",
                    ha="center", fontsize=10, fontweight="bold", color="#444")
            if idx[0] > 0:
                ax.axvline(xs[idx[0]] - 0.9, color="gray", ls=":", lw=1.2)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.set_ylabel("brain-masked SSIM")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=DESIGN_COLORS[k], label=DESIGN_LABEL[k])
               for k in ["1x", "fixed", "dup", "comp", "fcomp", "full", "dualfull"]]
    handles += [plt.Rectangle((0, 0), 1, 1, fc="white", ec="black", lw=1.8,
                              label="cross-view fine-tuned")]
    # figure-level legend: bars reach y=0 so there is no free space inside an axes
    fig.legend(handles=handles, fontsize=9, loc="lower center", ncol=6, frameon=False)
    fig.suptitle("Every experiment, grouped by what it cost\n"
                 "within a budget tier, only the acquisition design and method change",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0.035, 1, 0.945])
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# 4. Fine-tuning benefit vs coverage
# ---------------------------------------------------------------------------
def fig_ft(main, quad, out):
    # fine-tuning effect vs the merge baseline of each family (merge == removed
    # original-prior joint, so this measures the cross-view fine-tuning benefit).
    pts = [(0.188, paired(main, "fixed_split_ft", "fixed_split_merge"), "fixed"),
           (0.250, paired(main, "dup2_ft", "dup2_merge"), "dup2"),
           (0.438, paired(main, "comp2_ft", "comp2_merge"), "comp2"),
           (0.812, paired(quad, "comp4_ft", "comp4_merge"), "comp4")]
    pts = [p for p in pts if p[1] is not None]
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    xs = [p[0] * 100 for p in pts]
    ys = [p[1]["delta"] for p in pts]
    lo = [p[1]["delta"] - p[1]["lo"] for p in pts]
    hi = [p[1]["hi"] - p[1]["delta"] for p in pts]
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="-o", color="#8172b2", ms=11, lw=2.2, capsize=6)
    for x, y, p in zip(xs, ys, pts):
        ax.annotate(f"{p[2]}\n{y:+.3f} ({p[1]['win']:.0%})", (x, y),
                    textcoords="offset points", xytext=(0, 16), ha="center", fontsize=9)
    ax.axhline(0, color="black", ls="--", lw=1.2)
    ax.fill_between([0, 100], -0.003, 0.003, color="gray", alpha=0.18)
    ax.text(88, 0.0035, "practically zero", fontsize=8.5, color="gray", ha="right")
    ax.set_xlim(10, 95)
    ax.set_xlabel("unique k-space coverage (%)")
    ax.set_ylabel("SSIM gain from cross-view fine-tuning\n(paired, 95% CI)")
    ax.set_title("cross-view fine tuning decays to zero at adequate k-space coverage",
                 fontsize=12.5, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subject", required=True)
    ap.add_argument("--per-subject-quad", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    m = pd.read_csv(args.per_subject)
    q = pd.read_csv(args.per_subject_quad)
    o = args.output_dir
    fig_forest(m, q, os.path.join(o, "effect_sizes_forest.png"))
    fig_budget(m, q, os.path.join(o, "budget_ladder.png"))
    fig_ft(m, q, os.path.join(o, "finetuning_vs_coverage.png"))


if __name__ == "__main__":
    main()
