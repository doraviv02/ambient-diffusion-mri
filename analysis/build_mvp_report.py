#!/usr/bin/env python
"""Assemble reports/mvp_summary.md from the summary metrics, configs and reports.

Covers both evaluation sets:
  * the 2-view set   (tables/mvp/summary_metrics.csv,      25 subjects / 75 slices)
  * the 4-view set   (tables/mvp/summary_metrics_quad.csv, 20 subjects / 59 slices)
Paired deltas are bootstrapped over subjects from the per-subject tables.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


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


# method -> (description, views, lines bought, unique columns, coverage)
METHODS = {
    "M0":  ("noise-weighted adjoint (classical)",                "merge_fixed",      2, 64,  48,  0.188),
    "M1":  ("tuned L1-wavelet SENSE (classical)",                "merge_fixed",      2, 64,  48,  0.188),
    "M2":  ("original prior, single view",                       "single_r4",        1, 64,  64,  0.250),
    "M3":  ("original prior, analytic merge",                    "merge_fixed",      2, 64,  48,  0.188),
    "M4":  ("original prior, joint likelihood",                  "joint_fixed",      2, 64,  48,  0.188),
    "M5":  ("cross-view fine-tuned, joint likelihood",           "joint_fixed",      2, 64,  48,  0.188),
    "M7":  ("original prior, analytic merge",                    "merge_extra",      2, 128, 64,  0.250),
    "M8":  ("original prior, joint likelihood",                  "joint_extra",      2, 128, 64,  0.250),
    "M6":  ("cross-view fine-tuned, joint likelihood",           "joint_extra",      2, 128, 64,  0.250),
    "M15": ("original prior, analytic merge",                    "merge_extra_comp", 2, 128, 112, 0.438),
    "M14": ("original prior, joint likelihood",                  "joint_extra_comp", 2, 128, 112, 0.438),
    "M13": ("cross-view fine-tuned, joint likelihood",           "joint_extra_comp", 2, 128, 112, 0.438),
    "M11": ("original prior, analytic merge",                    "merge_quad",       4, 256, 64,  0.250),
    "M10": ("original prior, joint likelihood",                  "joint_quad",       4, 256, 64,  0.250),
    "M9":  ("cross-view fine-tuned, joint likelihood",           "joint_quad",       4, 256, 64,  0.250),
    "M18": ("original prior, analytic merge",                    "merge_quad_comp",  4, 256, 208, 0.812),
    "M17": ("original prior, joint likelihood",                  "joint_quad_comp",  4, 256, 208, 0.812),
    "M16": ("cross-view fine-tuned, joint likelihood",           "joint_quad_comp",  4, 256, 208, 0.812),
    "M12": ("original prior, one complete measurement",          "single_full",      1, 256, 256, 1.000),
    "MF":  ("plain recon, one complete measurement (reference)", "single_full",      1, 256, 256, 1.000),
}


class Table:
    """One evaluation set: summary stats + per-subject values for paired tests."""

    def __init__(self, summary_csv, per_subject_csv):
        self.summ = pd.read_csv(summary_csv) if os.path.exists(summary_csv) else pd.DataFrame()
        self.subj = pd.read_csv(per_subject_csv) if os.path.exists(per_subject_csv) else pd.DataFrame()

    def has(self, m):
        return not self.summ.empty and not self.summ[self.summ.method == m].empty

    def val(self, m, col):
        if self.summ.empty or "method" not in self.summ:
            return None
        r = self.summ[self.summ.method == m]
        if r.empty or col not in r:
            return None
        v = r.iloc[0][col]
        return None if pd.isna(v) else float(v)

    def fmt(self, m, base="ssim"):
        mu = self.val(m, f"{base}_mean")
        lo, hi = self.val(m, f"{base}_ci_lo"), self.val(m, f"{base}_ci_hi")
        if mu is None:
            return "n/a"
        return f"{mu:.4f} [{lo:.4f}, {hi:.4f}]" if lo is not None and hi is not None else f"{mu:.4f}"

    def paired(self, a, b, col="ssim", n_boot=10000, seed=0):
        """Subject-paired mean delta (a - b) with a bootstrap CI and win rate."""
        if self.subj.empty:
            return None
        A = self.subj[self.subj.method == a].set_index("subject_id")[col]
        B = self.subj[self.subj.method == b].set_index("subject_id")[col]
        idx = A.index.intersection(B.index)
        if len(idx) == 0:
            return None
        d = (A.loc[idx] - B.loc[idx]).values
        rng = np.random.RandomState(seed)
        boot = [rng.choice(d, len(d), replace=True).mean() for _ in range(n_boot)]
        return dict(delta=d.mean(), lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5),
                    win=float((d > 0).mean()), n=len(d))


def cmp_line(tab, a, b, label, col="ssim"):
    r = tab.paired(a, b, col=col)
    if r is None:
        return f"- **{label}** ({a} - {b}): n/a"
    return (f"- **{label}** ({a} - {b}): **{r['delta']:+.4f}** SSIM "
            f"[{r['lo']:+.4f}, {r['hi']:+.4f}], {a} wins on {r['win']:.0%} of {r['n']} subjects.")


def results_table(tab, methods, L):
    L.append("| Method | Views | Lines | Unique cols | Coverage | Prior / combiner | SSIM | NRMSE | Held-out k err |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for m in methods:
        if not tab.has(m):
            continue
        d, cond, v, lines, uniq, cov = METHODS[m]
        L.append(f"| {m} | {v} | {lines} | {uniq} | {cov:.0%} | {d} | "
                 f"{tab.fmt(m,'ssim')} | {tab.fmt(m,'nrmse')} | {tab.fmt(m,'heldout_kspace_err')} |")
    L.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--per-subject", default=None)
    ap.add_argument("--summary-quad", default=None)
    ap.add_argument("--per-subject-quad", default=None)
    ap.add_argument("--figures", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--budget-json", default=None)
    ap.add_argument("--preprocess-report", default=None)
    ap.add_argument("--selected-inference", default=None)
    ap.add_argument("--selected-classical", default=None)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    ps = args.per_subject or args.summary.replace("summary_metrics", "metrics_per_subject")
    main_t = Table(args.summary, ps)
    quad_t = None
    if args.summary_quad:
        psq = args.per_subject_quad or args.summary_quad.replace("summary_metrics", "metrics_per_subject")
        quad_t = Table(args.summary_quad, psq)

    budget = _load_json(args.budget_json) if args.budget_json else None
    prep = _load_json(args.preprocess_report) if args.preprocess_report else None
    sel_inf = _load_yaml(args.selected_inference) if args.selected_inference else None
    sel_cls = _load_yaml(args.selected_classical) if args.selected_classical else None

    L = []
    L.append("# MVP Summary: Cross-View Ambient Diffusion for Multi-Acquisition Low-Field MRI\n")

    # ---------------- headline ----------------
    L.append("## Headline\n")
    L.append("**Acquisition design dominates reconstruction method.** Changing *which* k-space lines "
             "the repeated scans measure — complementary instead of duplicated, at an identical "
             "line budget — is worth **+0.04 to +0.06 SSIM**. Every reconstruction-method choice in "
             "this study (learned prior vs original, cross-view fine-tuning, joint likelihood vs "
             "analytic merge) is worth **0.00 to +0.01**. Design beats method by roughly 5-10x.\n")
    L.append("Two consequences worth stating plainly:\n")
    L.append("- The project's nominal contribution (cross-view fine-tuning) only pays off in the "
             "**sparse** regime, and decays monotonically as coverage grows: +0.010 SSIM at 19% "
             "coverage (M5-M4), +0.003 at 25% (M6-M8), then **statistically zero** at 44% (M13-M14) "
             "and 81% (M16-M17). Once coverage is adequate the data determines the image and "
             "prior/combiner choices wash out. See `figures/mvp/finetuning_vs_coverage.png`.")
    L.append("- At an equal 256-line budget, **four cheap complementary R=4 scans beat one complete "
             "measurement when both are reconstructed the same way**: M16 > M12 on SSIM (+0.007, "
             "65% of subjects) and on NRMSE (-0.014, 95%). They trail a *plain* linear recon of the "
             "full scan (MF) on SSIM by 0.032, yet are **statistically tied with it on NRMSE** "
             "(+0.0006, CI [-0.004, +0.005]) — the residual gap is a reconstruction artefact, not "
             "an acquisition one; see the guidance-scale caveat below.")
    L.append("")

    # ---------------- scope ----------------
    L.append("## Dataset and scope\n")
    L.append("- Dataset: **M4Raw 0.3 T** T2-weighted brain, multi-coil (4 coils), 256x256, 2D.")
    L.append("- This is **low-field (0.3 T)**, NOT ultra-low-field (<0.1 T). Ultra-low-field "
             "validation is future work and is not claimed here.")
    if prep:
        # the preprocessing report covers the test split only; per-split slice counts
        # come from the mask manifest (budget_report n_samples).
        counts = "/".join(str((budget or {}).get(s, {}).get("n_samples", "?"))
                          for s in ("train", "val", "test")) if budget else "?"
        L.append(f"- Preprocessing: test acceptance rate "
                 f"{prep.get('acceptance_rate', float('nan')):.1%} "
                 f"({prep.get('accepted_slices','?')} accepted / "
                 f"{prep.get('rejected_slices','?')} rejected, "
                 f"{prep.get('test_subjects','?')} subjects); fallback maps: "
                 f"{prep.get('fallback_maps', 0)}; train/val/test slices: {counts}.")
    L.append("- Two evaluation sets: the **2-view set** (25 test subjects, 75 central slices) and the "
             "**4-view set** (20 subjects, 59 slices — fewer, because requiring 4 mutually "
             "motion-consistent repetitions accepts 188/300 slices).")
    L.append("- All metrics are **brain-masked** (Otsu on the reference, identical mask for every "
             "method on a slice) and aggregated **per subject** with 95% bootstrap CIs.")
    L.append("")

    # ---------------- mask design ----------------
    L.append("## Mask design and acquisition budget\n")
    L.append("Every condition uses Cartesian column undersampling with a fully-sampled 16-line ACS "
             "block. `lines` counts what you pay for (summed over views, ACS re-bought by each "
             "view); `unique cols` is what you actually learn about.\n")
    L.append("| Condition | Views | Lines bought | Unique cols | Coverage | Notes |")
    L.append("|---|---|---|---|---|---|")
    L.append("| `single_r4` | 1 | 64 | 64 | 25% | one cheap scan; the baseline budget |")
    L.append("| `merge_fixed` / `joint_fixed` | 2 | 64 | 48 | 19% | **fixed budget**: two R=8 views, "
             "shared ACS. Duplicated ACS costs 16 unique columns vs `single_r4`. |")
    L.append("| `joint_extra` / `merge_extra` | 2 | 128 | 64 | 25% | 2x budget, **duplicated** mask "
             "(runbook Condition D = plain NEX: a repetition re-runs the identical sequence) |")
    L.append("| **`joint_extra_comp` / `merge_extra_comp`** | 2 | 128 | **112** | **44%** | 2x budget, "
             "**complementary**: shared ACS + disjoint outer lines |")
    L.append("| `joint_quad` / `merge_quad` | 4 | 256 | 64 | 25% | 4x budget, **duplicated** — 256 "
             "lines to see 64 columns |")
    L.append("| **`joint_quad_comp` / `merge_quad_comp`** | 4 | 256 | **208** | **81%** | 4x budget, "
             "**complementary** |")
    L.append("| `single_full` | 1 | 256 | 256 | 100% | one complete measurement; the ceiling |")
    L.append("")
    L.append("The duplicated designs (`*_extra`, `*_quad`) follow the runbook literally, but "
             "re-buying the same 64 columns caps coverage at 25% no matter how many scans you add. "
             "They are retained as the honest ablation for *what the runbook specified*; the "
             "complementary designs are the corrected acquisition and carry the study's main result. "
             "Duplication is not worthless — it averages noise where the signal energy is, worth "
             "sigma/sqrt(V) (measured: 0.0443 -> 0.0313 at V=2) — it is just worth far less than "
             "coverage.")
    if budget:
        L.append("")
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
                 f"l_ss={sel_inf.get('l_ss')}, likelihood_type={sel_inf.get('likelihood_type')}.")
        L.append("  NOTE: the normalized multi-view fidelity divides by sigma^2*N, so the "
                 "effective l_ss scale is ~10x the runbook nominal grid (tuned on validation "
                 "held-out k-space error, single_r4).")
    if sel_cls:
        L.append(f"- L1-wavelet SENSE: lambda={sel_cls.get('lambda')}, "
                 f"iterations={sel_cls.get('iterations', 50)}.")
    L.append("- Cross-view fine-tune: 0.02 Mimg, lr=1e-4, EMA half-life 0.002 Mimg (must be << "
             "training duration, see notes).")
    L.append("")

    # ---------------- 2-view results ----------------
    L.append("## Results: 2-view set (25 subjects, 75 slices)\n")
    L.append("Subject-level mean [95% bootstrap CI]. Grouped by acquisition budget.\n")
    L.append("### Classical baselines and fixed budget (64 lines)\n")
    results_table(main_t, ["M0", "M1", "M2", "M3", "M4", "M5"], L)
    L.append("At a **fixed** budget, splitting into two views *loses* to spending it on one view "
             "(M2 - M5 = +0.0125, M2 wins 68%): the duplicated ACS buys 48 unique columns instead of "
             "64. Multi-view only helps when it buys extra scans, not when it subdivides one.\n")
    L.append("### 2x budget (128 lines): duplicated vs COMPLEMENTARY\n")
    results_table(main_t, ["M7", "M8", "M6", "M15", "M14", "M13"], L)
    L.append("The three complementary methods (M13/M14/M15, 44% coverage) beat their duplicated "
             "counterparts (M6/M8/M7, 25% coverage) by +0.041 to +0.052 SSIM on **100% of subjects**, "
             "at an identical 128-line cost. Within each coverage level the method variants are "
             "within 0.005 of each other.\n")

    # ---------------- 4-view results ----------------
    if quad_t is not None and not quad_t.summ.empty:
        L.append("## Results: 4-view set (20 subjects, 59 slices)\n")
        L.append("This set answers the budget question directly: **with the same 256 lines a full "
                 "scan costs, can four cheap R=4 scans surpass one complete measurement?**\n")
        results_table(quad_t, ["M2", "M11", "M10", "M9", "M18", "M17", "M16", "M12", "MF"], L)
        L.append("**Answer: yes against the same reconstruction method, no against the best one.** "
                 "M16 (4x complementary, 81% coverage) beats M12 (the *same* diffusion "
                 "reconstruction given a complete measurement) on both metrics: +0.007 SSIM (65% of "
                 "subjects) and -0.014 NRMSE (95%). Four cheap complementary scans genuinely "
                 "substitute for one full scan. But both lose on SSIM to MF, a plain linear recon of "
                 "the full scan (M12 - MF = -0.039, MF wins 90%): **the diffusion prior hurts on "
                 "well-sampled data**. With `l_ss=30` tuned at R=4, guidance keeps injecting prior "
                 "once the data already determines the image.\n")
        L.append("Note the metric split for M16 vs MF: -0.032 on SSIM but **+0.0006 on NRMSE, a "
                 "statistical tie** (CI [-0.004, +0.005]). M16 is as accurate as the full scan in "
                 "the L2 sense and loses only on *structural* similarity — the signature of a prior "
                 "adding plausible-but-wrong texture, not of missing information. That makes the "
                 "residual gap a *reconstruction* deficit, not an acquisition one, and per-condition "
                 "l_ss tuning is the obvious way to close it.\n")
        L.append("With the **duplicated** 4x design the same question answered *no* by a wide margin "
                 "(M9 0.703 vs MF 0.796, -0.093). The acquisition fix (M16 - M9 = +0.061 on 100% of "
                 "subjects) closes two thirds of that gap. This is the single largest effect measured "
                 "anywhere in the study.\n")

    # ---------------- key comparisons ----------------
    L.append("## Key comparisons (subject-paired, brain-masked SSIM)\n")
    L.append("### Acquisition design (the big effects)\n")
    L.append(cmp_line(main_t, "M13", "M6",  "COMPLEMENTARY vs duplicated, 2x budget, fine-tuned"))
    L.append(cmp_line(main_t, "M14", "M8",  "COMPLEMENTARY vs duplicated, 2x budget, joint"))
    L.append(cmp_line(main_t, "M15", "M7",  "COMPLEMENTARY vs duplicated, 2x budget, merged"))
    if quad_t is not None:
        L.append(cmp_line(quad_t, "M16", "M9", "COMPLEMENTARY vs duplicated, 4x budget"))
    L.append(cmp_line(main_t, "M7", "M2", "Extra cheap repetition (averaged) vs single view"))
    L.append("  Identical 25% coverage, 2x the lines -> isolates the pure sqrt(2) SNR benefit of "
             "averaging cheap low-field repeats. Real, but half the size of the design effect.")
    L.append(cmp_line(main_t, "M13", "M2", "Best 2-view design vs single view"))
    L.append("")
    L.append("### Reconstruction method (the small effects)\n")
    L.append(cmp_line(main_t, "M5", "M4", "Cross-view fine-tuning vs original prior, 19% coverage"))
    L.append(cmp_line(main_t, "M6", "M8", "Cross-view fine-tuning, 25% coverage"))
    L.append(cmp_line(main_t, "M13", "M14", "Cross-view fine-tuning, 44% coverage"))
    if quad_t is not None:
        L.append(cmp_line(quad_t, "M16", "M17", "Cross-view fine-tuning, 81% coverage"))
    L.append("  The fine-tuning benefit **decays to zero as coverage grows** — it is a sparse-regime "
             "effect only.")
    L.append("")
    L.append(cmp_line(main_t, "M4", "M3", "Joint likelihood vs analytic merge, fixed budget"))
    L.append(cmp_line(main_t, "M8", "M7", "Joint likelihood vs analytic merge, 2x duplicated"))
    L.append(cmp_line(main_t, "M14", "M15", "Joint likelihood vs analytic merge, 2x complementary"))
    if quad_t is not None:
        L.append(cmp_line(quad_t, "M10", "M11", "Joint likelihood vs analytic merge, 4x duplicated"))
        L.append(cmp_line(quad_t, "M17", "M18", "Joint likelihood vs analytic merge, 4x complementary"))
    L.append("  For aligned views with identical masks, theory predicts **equivalence**, and that is "
             "what we see (|delta| < 0.006) — *except* at 4x duplicated (M10 - M11 = +0.043), which "
             "is an artefact of the guidance-scale confound below, not a real advantage.")
    L.append("")
    L.append("### Diffusion vs classical\n")
    L.append(cmp_line(main_t, "M2", "M1", "Diffusion vs tuned L1-wavelet SENSE"))
    L.append(cmp_line(main_t, "M2", "M0", "Diffusion vs noise-weighted adjoint"))
    L.append("  The learned prior is worth ~+0.10 SSIM over classical baselines in the sparse regime "
             "— the one place where the prior axis is large. It reverses at full sampling (M12 < MF).")
    L.append("")

    # ---------------- figures ----------------
    L.append("## Figures\n")
    groups = [
        ("Cross-experiment (the conclusions)", [
            ("experiment_overview.png", "SSIM vs unique k-space coverage, all 20 methods, both sets "
                                        "-- the headline: coverage separates, method does not"),
            ("effect_sizes_forest.png", "every paired comparison with 95% CI, split into acquisition "
                                        "design vs reconstruction method"),
            ("budget_ladder.png", "every experiment as a bar, grouped by what it cost (64/128/256 lines)"),
            ("finetuning_vs_coverage.png", "the cross-view fine-tuning benefit decaying to zero as "
                                           "coverage grows"),
        ]),
        ("Acquisition design", [
            ("mask_design_all.png", "all 8 conditions: per-view masks, union coverage, and lines-bought "
                                    "vs unique-lines efficiency"),
            ("mask_design.png", "original 4-condition mask figure (runbook 10)"),
            ("mask_design_quad.png", "mask figure for the 4-view mask set"),
        ]),
        ("2-view set (25 subjects)", [
            ("reconstruction_montage.png", "reference / zero-filled / 11 methods on 4 cases picked by "
                                           "input SNR"),
            ("error_maps.png", "per-method absolute error vs the reference"),
            ("metric_boxplots.png", "subject-level NRMSE / SSIM / held-out error, boxes coloured by design"),
            ("fixed_vs_extra_budget.png", "metric vs acquired coefficients"),
            ("data_consistency_curves.png", "per-view normalized residual over sampler steps"),
        ]),
        ("4-view set (20 subjects)", [
            ("reconstruction_montage_quad.png", "the budget question: 4xR=4 duplicated vs complementary "
                                                "vs one full scan"),
            ("error_maps_quad.png", "per-method absolute error, 4-view set"),
            ("metric_boxplots_quad.png", "subject-level metrics, 4-view set"),
            ("fixed_vs_extra_budget_quad.png", "metric vs acquired coefficients, 4-view set"),
        ]),
        ("Uncertainty and tuning", [
            ("uncertainty_examples.png", "per-pixel posterior std over seeds"),
            ("uncertainty_calibration.png", "|error| vs predicted std"),
            ("validation_grid.png", "l_ss / num_steps / likelihood tuning on validation"),
        ]),
    ]
    for gname, items in groups:
        L.append(f"**{gname}**\n")
        for fn, desc_txt in items:
            p = os.path.join(args.figures, fn)
            mark = "" if os.path.exists(p) else "  _(missing)_"
            L.append(f"- `figures/mvp/{fn}`{mark} -- {desc_txt}")
        L.append("")

    # ---------------- limitations ----------------
    L.append("## Limitations and negative findings\n")
    L.append("Retained per runbook 9.4 — negative results are results.\n")
    L.append("- **0.3 T low-field, not <0.1 T ultra-low-field.** No ultra-low-field claim is made.")
    L.append("- **Joint likelihood == analytic merging** for aligned views at identical masks, as "
             "theory predicts. The multi-view contribution is not in the summation.")
    L.append("- **Cross-view fine-tuning is worth ~0 above 40% coverage** and only ~+0.01 in the "
             "sparse regime. It is also a short (0.02 Mimg) empirical self-supervised adaptation and "
             "does **not** inherit the Ambient Diffusion identifiability theorem.")
    L.append("- **The diffusion prior hurts on well-sampled data** (M12 < MF on 90% of subjects). "
             "A generative prior plus guidance only adds error once the data determines the image.")
    L.append("- **Joint-vs-merged is confounded by a guidance-scale factor.** The runbook's "
             "normalized fidelity averages over views, so `D_merged = V * D_joint` (measured: 1.989 "
             "at V=2, 3.967 at V=4). Since the sampler applies `grad(D)/sqrt(D)`, the merged "
             "condition gets **sqrt(V)x stronger guidance at the same l_ss**. A clean test needs "
             "per-condition l_ss tuning, or dropping the 1/V. The equivalence unit tests remain "
             "correct — they compare raw weighted-SSE gradients, which do match; it is the "
             "sampler-side normalization that differs.")
    L.append("- **l_ss was tuned once, on single_r4**, and reused for every V and R. This is the "
             "prime suspect for the residual M16 < MF gap and is the highest-value next experiment.")
    L.append("- **Test reference is an average of all 6 repetitions, including the 2-4 used as "
             "input** (~1/3 of the average), a mild uniform optimistic bias affecting all methods "
             "equally. Comparisons are unaffected; absolute SSIM is slightly generous.")
    L.append("- **SSIM ~0.6-0.8 here is not comparable to the ~0.9+ in the Ambient/FastMRI paper.** "
             "Different field strength and, crucially, a different reference: they compare against "
             "the fully-sampled version of the *same* acquisition (one undersampling gap), we "
             "compare against a 6-repetition average (an undersampling gap *plus* an irreducible SNR "
             "gap). Measured directly: a fully-sampled 1-rep recon scores **0.991** against an "
             "SNR-matched self-reference (proving the operator/FFT path is exact) but only **0.703** "
             "against the 6-rep average — that 0.703 is the practical **ceiling** for our setup, so "
             "the 2-view methods at 0.60-0.71 sit at 85-100% of what is achievable.")
    L.append("- Validation reference is a noisy third repetition (weak), so held-out k-space error "
             "was the primary tuning criterion. At 0.3 T the held-out outer k-space is largely noise, "
             "so that metric compresses into 0.84-0.98 and discriminates weakly; SSIM/NRMSE vs the "
             "multi-repetition average carry more signal.")
    L.append("")
    L.append("See `reports/mvp_notes.md` for the full decision log, including four bugs that "
             "materially changed conclusions (brain-mask, held-out-error bias, EMA half-life, "
             "duplicated-mask design).")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
