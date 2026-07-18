#!/usr/bin/env python
"""Cross-experiment figures: coverage, effect sizes, budget ladder, fine-tune decay.

These are the figures that carry the study's conclusions, as opposed to the
per-method montages.  Everything is computed from the per-subject metric tables
so the plots and the report cannot disagree.

Outputs (figures/mvp/):
  experiment_overview.png     SSIM vs unique k-space coverage, all 20 methods
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
import matplotlib.patheffects as pe

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.experiments import EXPERIMENTS, DESIGN_COLORS, design_of, methods_for

MARKERS = {"single": "o", "merge": "s", "ft": "^", "plain": "D"}
DESIGN_LABEL = {
    "fixed": "fixed budget (split one scan)",
    "1x": "one cheap scan",
    "dup": "extra scans, DUPLICATED mask",
    "comp": "extra scans, COMPLEMENTARY masks",
    "full": "one complete measurement",
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
# 1. SSIM vs coverage -- the headline figure
# ---------------------------------------------------------------------------
def fig_overview(main, quad, out):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=False)
    for ax, (df, set_, title, n, jit) in zip(axes, [
            (main, "main", "2-view set", main.subject_id.nunique(), 0.85),
            (quad, "quad", "4-view set", quad.subject_id.nunique(), 2.2)]):
        # Methods sharing a coverage differ by <0.006 SSIM, so points (and their
        # labels) would overlap. Spread each coverage cluster horizontally.
        present = [m for m in methods_for(set_) if not df[df.method == m].empty]
        clusters = {}
        for m in present:
            clusters.setdefault(EXPERIMENTS[m].cov, []).append(m)
        for cov, ms in clusters.items():
            ms.sort(key=lambda m: subj_mean(df, m))
            for i, m in enumerate(ms):
                e = EXPERIMENTS[m]
                x = cov * 100 + (i - (len(ms) - 1) / 2) * jit
                y = subj_mean(df, m)
                c = DESIGN_COLORS[design_of(m)]
                ax.scatter(x, y, s=190, color=c, marker=MARKERS[e.combiner],
                           edgecolor="black" if e.prior == "finetuned" else "white",
                           linewidth=2.0 if e.prior == "finetuned" else 1.0, zorder=3)
                # stagger labels above/below: cluster members differ by <0.006 SSIM
                # so a single offset would overlap regardless of jitter. Colour the
                # text to match its marker so close pairs stay unambiguous.
                dy = 13 if i % 2 == 0 else -23
                ax.annotate(m, (x, y), textcoords="offset points", xytext=(0, dy),
                            ha="center", fontsize=8.5, fontweight="bold", color=c,
                            zorder=4, path_effects=[pe.withStroke(linewidth=2.2,
                                                                  foreground="white")])
        ax.set_xlabel("unique k-space coverage (% of PE columns)")
        ax.set_ylabel("brain-masked SSIM (subject mean)")
        ax.set_title(f"{title}  ({n} subjects)")
        ax.grid(alpha=0.3)

    # annotate the two big design jumps
    a = axes[0]
    y6, y13 = subj_mean(main, "dup2_ft"), subj_mean(main, "comp2_ft")
    if y6 and y13:
        a.annotate("", xy=(43.8, y13), xytext=(25, y6),
                   arrowprops=dict(arrowstyle="->", lw=2.4, color="#2a7f3f"))
        a.text(34, (y6 + y13) / 2 + 0.008, f"same 128 lines\n+{y13-y6:.3f} SSIM",
               color="#2a7f3f", fontsize=10, fontweight="bold", ha="center")
    b = axes[1]
    y9, y16, yMF = subj_mean(quad, "dup4_ft"), subj_mean(quad, "comp4_ft"), subj_mean(quad, "full_plain")
    if y9 and y16:
        b.annotate("", xy=(81.2, y16), xytext=(25, y9),
                   arrowprops=dict(arrowstyle="->", lw=2.4, color="#2a7f3f"))
        b.text(53, (y9 + y16) / 2 + 0.008, f"same 256 lines\n+{y16-y9:.3f} SSIM",
               color="#2a7f3f", fontsize=10, fontweight="bold", ha="center")
    if yMF:
        b.axhline(yMF, ls="--", color="#c44e52", lw=1.5, alpha=0.8)
        b.text(24, yMF - 0.007, "ceiling: one full scan, plain recon (full_plain)",
               color="#c44e52", fontsize=9, va="top")

    handles = [plt.Line2D([], [], marker="o", ls="", color=DESIGN_COLORS[k], ms=11,
                          label=DESIGN_LABEL[k]) for k in ["1x", "fixed", "dup", "comp", "full"]]
    handles += [plt.Line2D([], [], marker=MARKERS[k], ls="", color="gray", ms=11,
                           label=f"{k}") for k in ["single", "merge", "ft", "plain"]]
    handles += [plt.Line2D([], [], marker="o", ls="", mfc="white", mec="black", mew=2, ms=11,
                           label="cross-view fine-tuned")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle("Coverage, not method, drives reconstruction quality\n"
                 "colour = acquisition design | marker = combiner | black ring = fine-tuned prior",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.10, 1, 0.93])
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


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
    ax.set_title("What actually moves the needle\n"
                 "design effects are 5-10x larger than every method effect",
                 fontsize=12.5, fontweight="bold")
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
               for k in ["1x", "fixed", "dup", "comp", "full"]]
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
    ax.set_title("The project's contribution is a SPARSE-REGIME effect\n"
                 "cross-view fine-tuning decays to zero once coverage is adequate",
                 fontsize=12, fontweight="bold")
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
    fig_overview(m, q, os.path.join(o, "experiment_overview.png"))
    fig_forest(m, q, os.path.join(o, "effect_sizes_forest.png"))
    fig_budget(m, q, os.path.join(o, "budget_ladder.png"))
    fig_ft(m, q, os.path.join(o, "finetuning_vs_coverage.png"))


if __name__ == "__main__":
    main()
