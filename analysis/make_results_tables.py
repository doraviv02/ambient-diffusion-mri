#!/usr/bin/env python
"""Generate ONLY the results tables (no prose, no figures).

Reproduces exactly the three results tables in reports/project_summary.md from the
summary_metrics CSVs, and writes them as both Markdown and CSV:

  tables/project/results_tables.md          all three tables, Markdown
  tables/project/results_2view_fixed.csv    2-view table 1 (classical + fixed budget)
  tables/project/results_2view_2x.csv       2-view table 2 (2x budget)
  tables/project/results_4view.csv          4-view table

Numbers come straight from summary_metrics{,_quad}.csv, so these tables and the
report cannot disagree. Method identity/order/coverage come from
analysis.experiments (the single source of truth).
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.experiments import EXPERIMENTS

# the three tables, exactly as presented in project_summary.md
T_FIXED = ["classical_adjoint", "classical_l1wav", "single_r4", "fixed_split_merge", "fixed_split_ft"]
T_2X = ["dup2_merge", "dup2_ft", "comp2_merge", "comp2_ft", "fcomp2_merge", "fcomp2_ft"]
T_QUAD = ["single_r4", "dup4_merge", "dup4_ft", "comp4_merge", "comp4_ft",
          "fcomp4_merge", "fcomp4_ft", "full_diffusion", "full_plain",
          "dualfull_merge", "dualfull_ft"]

COLS = ["ssim", "nrmse", "heldout_kspace_err"]
COL_HDR = {"ssim": "SSIM", "nrmse": "NRMSE", "heldout_kspace_err": "Held-out k err"}


def _fmt(summ, m, col):
    r = summ[summ.method == m]
    if r.empty or f"{col}_mean" not in r:
        return "n/a"
    v = r.iloc[0]
    mu, lo, hi = v[f"{col}_mean"], v.get(f"{col}_ci_lo"), v.get(f"{col}_ci_hi")
    if pd.isna(mu):
        return "n/a"
    return f"{mu:.4f} [{lo:.4f}, {hi:.4f}]" if pd.notna(lo) and pd.notna(hi) else f"{mu:.4f}"


def _num(summ, m, col):
    r = summ[summ.method == m]
    if r.empty:
        return dict()
    v = r.iloc[0]
    out = {}
    for suf in ("mean", "ci_lo", "ci_hi"):
        out[f"{col}_{suf}"] = v.get(f"{col}_{suf}")
    return out


def markdown_table(summ, methods, title):
    lines = [f"### {title}\n",
             "| Method | Description | Views | Lines | Unique cols | Coverage | SSIM | NRMSE | Held-out k err |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in methods:
        if summ[summ.method == m].empty:
            continue
        e = EXPERIMENTS[m]
        lines.append(f"| `{m}` | {e.label} | {e.views} | {e.lines} | {e.uniq} | {e.cov:.0%} | "
                     f"{_fmt(summ, m, 'ssim')} | {_fmt(summ, m, 'nrmse')} | "
                     f"{_fmt(summ, m, 'heldout_kspace_err')} |")
    lines.append("")
    return "\n".join(lines)


def csv_table(summ, methods, path):
    rows = []
    for m in methods:
        if summ[summ.method == m].empty:
            continue
        e = EXPERIMENTS[m]
        row = {"method": m, "description": e.label, "views": e.views, "lines": e.lines,
               "unique_cols": e.uniq, "coverage": round(e.cov, 3)}
        for col in COLS:
            row.update(_num(summ, m, col))
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="summary_metrics.csv (2-view)")
    ap.add_argument("--summary-quad", required=True, help="summary_metrics_quad.csv (4-view)")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main_s = pd.read_csv(args.summary)
    quad_s = pd.read_csv(args.summary_quad)

    md = ["# Results tables (subject mean [95% bootstrap CI])\n",
          "## 2-view set (25 subjects, 75 slices)\n",
          markdown_table(main_s, T_FIXED, "Classical baselines & fixed budget (64 lines)"),
          markdown_table(main_s, T_2X, "2x budget (128 lines): duplicated vs complementary vs fully-comp"),
          "## 4-view set (20 subjects, 59 slices)\n",
          markdown_table(quad_s, T_QUAD, "Budget question: 4 cheap scans vs one full scan (+ NEX=2)")]
    md_path = os.path.join(args.output_dir, "results_tables.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {md_path}")

    csv_table(main_s, T_FIXED, os.path.join(args.output_dir, "results_2view_fixed.csv"))
    csv_table(main_s, T_2X, os.path.join(args.output_dir, "results_2view_2x.csv"))
    csv_table(quad_s, T_QUAD, os.path.join(args.output_dir, "results_4view.csv"))


if __name__ == "__main__":
    main()
