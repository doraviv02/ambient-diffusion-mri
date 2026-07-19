#!/usr/bin/env python
"""Assemble reports/mvp_summary.md: dataset/scope, the mask design, and the
results tables. Deliberately terse -- no narrative, no interpretation.

Covers both evaluation sets:
  * the 2-view set   (tables/mvp/summary_metrics.csv,      25 subjects / 75 slices)
  * the 4-view set   (tables/mvp/summary_metrics_quad.csv, 20 subjects / 59 slices)
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.experiments import EXPERIMENTS  # noqa: E402

# the three results tables, in report order
T_FIXED = ["classical_adjoint", "classical_l1wav", "single_r4", "fixed_split_merge", "fixed_split_ft"]
T_2X = ["dup2_merge", "dup2_ft", "comp2_merge", "comp2_ft", "fcomp2_merge", "fcomp2_ft"]
T_QUAD = ["single_r4", "dup4_merge", "dup4_ft", "comp4_merge", "comp4_ft",
          "fcomp4_merge", "fcomp4_ft", "full_diffusion", "full_plain",
          "dualfull_merge", "dualfull_ft"]


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _load_yaml(path):
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _fmt(summ, m, col):
    r = summ[summ.method == m]
    if r.empty or f"{col}_mean" not in r:
        return "n/a"
    v = r.iloc[0]
    mu, lo, hi = v[f"{col}_mean"], v.get(f"{col}_ci_lo"), v.get(f"{col}_ci_hi")
    if pd.isna(mu):
        return "n/a"
    return f"{mu:.4f} [{lo:.4f}, {hi:.4f}]" if pd.notna(lo) and pd.notna(hi) else f"{mu:.4f}"


def results_table(summ, methods, L):
    L.append("| Method | Description | Views | Lines | Unique cols | Coverage | SSIM | NRMSE | Held-out k err |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for m in methods:
        if summ.empty or summ[summ.method == m].empty:
            continue
        e = EXPERIMENTS[m]
        L.append(f"| `{m}` | {e.label} | {e.views} | {e.lines} | {e.uniq} | {e.cov:.0%} | "
                 f"{_fmt(summ, m, 'ssim')} | {_fmt(summ, m, 'nrmse')} | "
                 f"{_fmt(summ, m, 'heldout_kspace_err')} |")
    L.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--summary-quad", default=None)
    ap.add_argument("--per-subject", default=None)        # accepted, unused
    ap.add_argument("--per-subject-quad", default=None)   # accepted, unused
    ap.add_argument("--figures", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--budget-json", default=None)
    ap.add_argument("--preprocess-report", default=None)
    ap.add_argument("--selected-inference", default=None)
    ap.add_argument("--selected-classical", default=None)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    main_s = pd.read_csv(args.summary) if os.path.exists(args.summary) else pd.DataFrame()
    quad_s = (pd.read_csv(args.summary_quad)
              if args.summary_quad and os.path.exists(args.summary_quad) else pd.DataFrame())
    budget = _load_json(args.budget_json) if args.budget_json else None
    prep = _load_json(args.preprocess_report) if args.preprocess_report else None
    sel_inf = _load_yaml(args.selected_inference) if args.selected_inference else None
    sel_cls = _load_yaml(args.selected_classical) if args.selected_classical else None

    L = ["# MVP Summary: Cross-View Ambient Diffusion for Multi-Acquisition Low-Field MRI\n"]

    # ---------------- dataset and scope ----------------
    L.append("## Dataset and scope\n")
    L.append("- Dataset: **M4Raw 0.3 T** T2-weighted brain, multi-coil (4 coils), 256x256, 2D.")
    L.append("- This is **low-field (0.3 T)**, NOT ultra-low-field (<0.1 T).")
    if prep:
        counts = "/".join(str((budget or {}).get(s, {}).get("n_samples", "?"))
                          for s in ("train", "val", "test")) if budget else "?"
        L.append(f"- Preprocessing: test acceptance rate "
                 f"{prep.get('acceptance_rate', float('nan')):.1%} "
                 f"({prep.get('accepted_slices','?')} accepted / "
                 f"{prep.get('rejected_slices','?')} rejected, "
                 f"{prep.get('test_subjects','?')} subjects); train/val/test slices: {counts}.")
    L.append("- Two evaluation sets: the **2-view set** (25 test subjects, 75 central slices) and the "
             "**4-view set** (20 subjects, 59 slices — requiring 4 mutually motion-consistent "
             "repetitions accepts 188/300 slices).")
    L.append("- Reference: the average of all 6 M4Raw test repetitions (RSS magnitude).")
    L.append("- All metrics are **brain-masked** (Otsu on the reference, identical mask for every "
             "method on a slice) and aggregated **per subject** with 95% bootstrap CIs.")
    L.append("")

    # ---------------- mask design ----------------
    L.append("## Mask design and acquisition budget\n")
    L.append("Cartesian column undersampling with a fully-sampled 16-line ACS block. `Lines` is what "
             "you pay for (summed over views); `unique cols` is what you actually measure.\n")
    L.append("| Condition | Views | Lines bought | Unique cols | Coverage |")
    L.append("|---|---|---|---|---|")
    L.append("| `single_r4` | 1 | 64 | 64 | 25% |")
    L.append("| `merge_fixed` / `joint_fixed` (fixed budget, 2xR=8) | 2 | 64 | 48 | 19% |")
    L.append("| `*_extra` (2x, duplicated mask) | 2 | 128 | 64 | 25% |")
    L.append("| `*_extra_comp` (2x, complementary, shared ACS) | 2 | 128 | 112 | 44% |")
    L.append("| `*_extra_fullcomp` (2x, fully-complementary, split ACS) | 2 | 128 | 128 | 50% |")
    L.append("| `*_quad` (4x, duplicated mask) | 4 | 256 | 64 | 25% |")
    L.append("| `*_quad_comp` (4x, complementary, shared ACS) | 4 | 256 | 208 | 81% |")
    L.append("| `*_quad_fullcomp` (4x, fully-complementary, split ACS) | 4 | 256 | 256 | 100% |")
    L.append("| `single_full` (one complete measurement) | 1 | 256 | 256 | 100% |")
    L.append("| `dual_full` (two complete measurements, NEX=2) | 2 | 512 | 256 | 100% |")
    L.append("")
    if budget:
        L.append("Fixed-budget accounting (coefficients = columns x 256 rows):")
        for split, b in budget.items():
            L.append(f"- **{split}**: single_r4={b.get('single_r4')}, "
                     f"fixed_sum(dupACS)={b.get('fixed_sum_dupACS')} "
                     f"(within tolerance: {b.get('within_tolerance')}), "
                     f"fixed_union={b.get('fixed_union')}, extra_sum={b.get('extra_sum')}.")
        L.append("")

    # ---------------- hyperparameters ----------------
    L.append("## Selected hyperparameters\n")
    if sel_inf:
        L.append(f"- Diffusion inference: num_steps={sel_inf.get('num_steps')}, "
                 f"l_ss={sel_inf.get('l_ss')}, likelihood_type={sel_inf.get('likelihood_type')} "
                 f"(tuned on validation held-out k-space error).")
    if sel_cls:
        L.append(f"- L1-wavelet SENSE: lambda={sel_cls.get('lambda')}, "
                 f"iterations={sel_cls.get('iterations', 50)}.")
    L.append("- Cross-view fine-tune: 0.02 Mimg, lr=1e-4, EMA half-life 0.002 Mimg.")
    L.append("")

    # ---------------- results tables ----------------
    L.append("## Results: 2-view set (25 subjects, 75 slices)\n")
    L.append("Subject-level mean [95% bootstrap CI].\n")
    L.append("### Classical baselines and fixed budget (64 lines)\n")
    results_table(main_s, T_FIXED, L)
    L.append("### 2x budget (128 lines): duplicated vs complementary vs fully-complementary\n")
    results_table(main_s, T_2X, L)

    if not quad_s.empty:
        L.append("## Results: 4-view set (20 subjects, 59 slices)\n")
        L.append("Subject-level mean [95% bootstrap CI].\n")
        results_table(quad_s, T_QUAD, L)

    # ---------------- figures ----------------
    L.append("## Figures\n")
    groups = [
        ("Main results — reconstruction montages (per results table)", [
            "recon_montage_2view.png", "recon_montage_4view.png"]),
        ("Results tables (visualized)", [
            "results_2view.png", "results_4view.png"]),
        ("Cross-experiment", [
            "effect_sizes_forest.png", "budget_ladder.png", "finetuning_vs_coverage.png"]),
        ("Acquisition design and tuning", [
            "mask_design_all.png", "lss_tier_tuning.png"]),
        ("Supporting — absolute error maps", [
            "error_maps.png", "error_maps_quad.png"]),
    ]
    for gname, items in groups:
        L.append(f"**{gname}**\n")
        for fn in items:
            mark = "" if os.path.exists(os.path.join(args.figures, fn)) else "  _(missing)_"
            L.append(f"- `figures/mvp/{fn}`{mark}")
        L.append("")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
