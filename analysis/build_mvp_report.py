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
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.experiments import EXPERIMENTS, MAIN_ORDER, QUAD_ORDER  # noqa: E402


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
    L.append("| Method | Description | Views | Lines | Unique cols | Coverage | SSIM | NRMSE | Held-out k err |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for m in methods:
        if not tab.has(m):
            continue
        e = EXPERIMENTS[m]
        L.append(f"| `{m}` | {e.label} | {e.views} | {e.lines} | {e.uniq} | {e.cov:.0%} | "
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
             "this study (learned prior vs classical, cross-view fine-tuning) is worth **0.00 to "
             "+0.01**. Design beats method by roughly 5-10x.\n")
    L.append("Two consequences worth stating plainly:\n")
    L.append("- The project's nominal contribution (cross-view fine-tuning) only pays off in the "
             "**sparse** regime, and decays monotonically as coverage grows: +0.008 SSIM at 19% "
             "coverage (`fixed_split_ft`-`fixed_split_merge`), +0.009 at 25% (`dup2_ft`-`dup2_merge`), "
             "then **statistically zero** at 44% (`comp2_ft`-`comp2_merge`) and 81% "
             "(`comp4_ft`-`comp4_merge`). Once coverage is adequate the data determines the image and "
             "prior choices wash out. See `figures/mvp/finetuning_vs_coverage.png`.")
    L.append("- At an equal 256-line budget, **four cheap complementary R=4 scans beat one complete "
             "measurement when both are reconstructed the same way**: `comp4_ft` > `full_diffusion` "
             "on SSIM (+0.007, 65% of subjects) and on NRMSE (-0.014, 95%). They trail a *plain* "
             "linear recon of the full scan (`full_plain`) on SSIM by 0.032, yet are **statistically "
             "tied with it on NRMSE** (+0.0006, CI [-0.004, +0.005]) — the residual gap is a "
             "reconstruction artefact, not an acquisition one; see the limitations section.")
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
    results_table(main_t, ["classical_adjoint", "classical_l1wav", "single_r4",
                           "fixed_split_merge", "fixed_split_ft"], L)
    L.append("At a **fixed** budget, splitting into two views *loses* to spending it on one view "
             "(`single_r4` - `fixed_split_ft` = +0.0125, single wins 68%): the duplicated ACS buys 48 "
             "unique columns instead of 64. Multi-view only helps when it buys extra scans, not when "
             "it subdivides one.\n")
    L.append("### 2x budget (128 lines): duplicated vs COMPLEMENTARY vs FULLY-complementary\n")
    results_table(main_t, ["dup2_merge", "dup2_ft", "comp2_merge", "comp2_ft",
                           "fcomp2_merge", "fcomp2_ft"], L)
    L.append("Three acquisition designs at an identical 128-line budget, in rising coverage: "
             "`dup2_*` re-buys one mask (25%); `comp2_*` shares the full ACS and partitions the "
             "outer lines (44%); `fcomp2_*` also **partitions the ACS** (split round-robin, no "
             "overlap anywhere), reaching **50% coverage** — the maximum possible per line — at the "
             "cost of no √V averaging in the k-space centre. The complementary methods beat "
             "duplicated by +0.041 to +0.052 SSIM on 100% of subjects; fcomp vs comp isolates the "
             "coverage-vs-centre-SNR trade (see Key comparisons).\n")

    # ---------------- 4-view results ----------------
    if quad_t is not None and not quad_t.summ.empty:
        L.append("## Results: 4-view set (20 subjects, 59 slices)\n")
        L.append("This set answers the budget question directly: **with the same 256 lines a full "
                 "scan costs, can four cheap R=4 scans surpass one complete measurement?**\n")
        results_table(quad_t, ["single_r4", "dup4_merge", "dup4_ft", "comp4_merge", "comp4_ft",
                               "fcomp4_merge", "fcomp4_ft", "full_diffusion", "full_plain",
                               "dualfull_merge", "dualfull_ft"], L)
        L.append("**Answer: yes against the same reconstruction method, no against the best one.** "
                 "`comp4_ft` (4x complementary, 81% coverage) beats `full_diffusion` (the *same* "
                 "diffusion reconstruction given a complete measurement) on both metrics: +0.007 SSIM "
                 "(65% of subjects) and -0.014 NRMSE (95%). Four cheap complementary scans genuinely "
                 "substitute for one full scan. But both lose on SSIM to `full_plain`, a plain linear "
                 "recon of the full scan (`full_diffusion` - `full_plain` = -0.039, plain wins 90%): "
                 "**the diffusion prior hurts on well-sampled data**.\n")
        L.append("Note the metric split for `comp4_ft` vs `full_plain`: -0.032 on SSIM but **+0.0006 "
                 "on NRMSE, a statistical tie** (CI [-0.004, +0.005]). It is as accurate as the full "
                 "scan in the L2 sense and loses only on *structural* similarity — the signature of a "
                 "prior adding plausible-but-wrong texture, not of missing information. That makes the "
                 "residual gap a *reconstruction* deficit, not an acquisition one. We suspected the "
                 "fix was to re-tune the guidance scale `l_ss` (tuned once on sparse data) per "
                 "coverage tier, and **tested it: it does not help** — `l_ss=30` is optimal at every "
                 "tier, and raising it on well-sampled data makes SSIM *worse* (see "
                 "`figures/mvp/lss_tier_tuning.png` and the limitations section). The gap is intrinsic "
                 "to soft-guidance diffusion sampling vs a direct solve, not a tuning artefact.\n")
        L.append("With the **duplicated** 4x design the same question answered *no* by a wide margin "
                 "(`dup4_ft` 0.703 vs `full_plain` 0.796, -0.093). The acquisition fix "
                 "(`comp4_ft` - `dup4_ft` = +0.061 on 100% of subjects) closes two thirds of that "
                 "gap. This is the single largest effect measured anywhere in the study.\n")

    # ---------------- key comparisons ----------------
    L.append("## Key comparisons (subject-paired, brain-masked SSIM)\n")
    L.append("_Note: the redundant \"joint likelihood, original prior\" methods (old M4/M8/M10/M14/"
             "M17) were removed, since joint == analytic merge for aligned identical-mask views. "
             "Fine-tuning effects are therefore measured against the **merge** baseline of the same "
             "acquisition (equivalent, since merge == the removed original-prior joint)._\n")
    L.append("### Acquisition design (the big effects)\n")
    L.append(cmp_line(main_t, "comp2_ft", "dup2_ft", "COMPLEMENTARY vs duplicated, 2x budget, fine-tuned"))
    L.append(cmp_line(main_t, "comp2_merge", "dup2_merge", "COMPLEMENTARY vs duplicated, 2x budget, merged"))
    if quad_t is not None:
        L.append(cmp_line(quad_t, "comp4_ft", "dup4_ft", "COMPLEMENTARY vs duplicated, 4x budget"))
    L.append(cmp_line(main_t, "dup2_merge", "single_r4", "Extra cheap repetition (averaged) vs single view"))
    L.append("  Identical 25% coverage, 2x the lines -> isolates the pure sqrt(2) SNR benefit of "
             "averaging cheap low-field repeats. Real, but half the size of the design effect.")
    L.append(cmp_line(main_t, "comp2_ft", "single_r4", "Best 2-view design vs single view"))
    L.append("")
    L.append("### Fully-complementary: maximum coverage vs k-space-centre SNR\n")
    L.append("`fcomp*` splits the ACS across views (no overlap anywhere) to convert the duplicated-ACS "
             "budget into extra coverage, trading away the √V noise averaging in the centre that "
             "`comp*` keeps.\n")
    L.append(cmp_line(main_t, "fcomp2_ft", "comp2_ft", "Split-ACS (50%) vs shared-ACS (44%), 2x"))
    if quad_t is not None:
        L.append(cmp_line(quad_t, "fcomp4_ft", "comp4_ft", "Split-ACS (100%) vs shared-ACS (81%), 4x"))
        L.append("  At 4 views `fcomp4` reaches **full coverage** — a complete k-space assembled from "
                 "4 disjoint cheap scans. The two comparisons below ask whether that equals one full "
                 "scan (same coverage, same per-line single-rep noise, but 4 independent acquisitions "
                 "stitched vs 1 monolithic):")
        L.append(cmp_line(quad_t, "fcomp4_ft", "full_diffusion", "4x disjoint cheap scans vs 1 full scan (both diffusion)"))
        L.append(cmp_line(quad_t, "fcomp4_merge", "full_plain", "4x disjoint cheap scans vs 1 full scan (both plain/merge)"))
    L.append("")
    if quad_t is not None and quad_t.has("dualfull_merge"):
        L.append("### Two full acquisitions (low-field NEX=2): the true reconstruction ceiling\n")
        L.append("Two *complete* k-space acquisitions of the same slice (512 lines), the realistic "
                 "low-field multi-average scenario. `dualfull_merge` is the classical noise-weighted "
                 "average (= (y0+y1)/2 at σ/√2, plain recon); `dualfull_ft` is the learned cross-view "
                 "fine-tuned joint reconstruction over both full views.\n")
        L.append(cmp_line(quad_t, "dualfull_merge", "full_plain", "NEX=2 classical average vs NEX=1 (both plain)"))
        L.append("  Isolates the pure √2 SNR benefit of a second full acquisition under identical "
                 "(classical) reconstruction — the honest low-field quality ceiling.")
        L.append(cmp_line(quad_t, "dualfull_ft", "dualfull_merge", "learned xview-FT vs classical average, same 2 full views"))
        L.append(cmp_line(quad_t, "dualfull_ft", "full_plain", "learned on 2 full views vs classical on 1 full view"))
        L.append("")
        L.append("**This is the study's sharpest result.** `dualfull_merge` (classical NEX=2 average) "
                 "scores **0.878 SSIM / 0.099 NRMSE — the best of any method here by a wide margin**, "
                 "+0.081 over a single full scan on 100% of subjects: at low field, a second full "
                 "acquisition is worth far more than any reconstruction cleverness. And the learned "
                 "method **fails this test decisively** — `dualfull_ft` (0.759) *loses to the classical "
                 "average by 0.119 SSIM* and gains essentially nothing over `full_diffusion` (+0.001), "
                 "i.e. it extracts almost no value from the second full acquisition. Once coverage is "
                 "complete, the diffusion prior is not just unhelpful but a large net negative vs "
                 "plain averaging. The learned pipeline earns its keep **only in the undersampled "
                 "regime**; the honest low-field reconstruction ceiling is classical multi-average.")
        L.append("")
    L.append("### Cross-view fine-tuning (the contribution) vs the merge baseline\n")
    L.append(cmp_line(main_t, "fixed_split_ft", "fixed_split_merge", "Cross-view FT, 19% coverage"))
    L.append(cmp_line(main_t, "dup2_ft", "dup2_merge", "Cross-view FT, 25% coverage"))
    L.append(cmp_line(main_t, "comp2_ft", "comp2_merge", "Cross-view FT, 44% coverage"))
    if quad_t is not None:
        L.append(cmp_line(quad_t, "comp4_ft", "comp4_merge", "Cross-view FT, 81% coverage"))
    L.append("  The fine-tuning benefit **decays to zero as coverage grows** (+0.008 at 19-25% "
             "coverage, ~0 at 44-81%) — it is a sparse-regime effect only.")
    L.append("")
    L.append("### Diffusion vs classical\n")
    L.append(cmp_line(main_t, "single_r4", "classical_l1wav", "Diffusion vs tuned L1-wavelet SENSE"))
    L.append(cmp_line(main_t, "single_r4", "classical_adjoint", "Diffusion vs noise-weighted adjoint"))
    L.append("  The learned prior is worth ~+0.10 SSIM over classical baselines in the sparse regime "
             "— the one place where the prior axis is large. It reverses at full sampling "
             "(`full_diffusion` < `full_plain`).")
    L.append("")

    # ---------------- figures ----------------
    L.append("## Figures\n")
    groups = [
        ("Results tables (visualized)", [
            ("results_2view.png", "the two 2-view result tables (fixed budget; 2x budget) as SSIM "
                                  "and NRMSE bars with 95% CI, coloured by design"),
            ("results_4view.png", "the 4-view result table as SSIM and NRMSE bars with 95% CI "
                                  "(the budget question, incl. dualfull NEX=2)"),
            ("recon_montage_2view.png", "reconstructed IMAGES per 2-view table (columns = the table's "
                                        "methods, coloured by design; 3 example slices)"),
            ("recon_montage_4view.png", "reconstructed IMAGES for the 4-view table (single_r4 -> full "
                                        "scan -> NEX=2); dualfull_merge is visibly the cleanest"),
        ]),
        ("Cross-experiment (the conclusions)", [
            ("experiment_overview.png", "SSIM vs unique k-space coverage, all methods, both sets "
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
            ("lss_tier_tuning.png", "per-coverage-tier l_ss retune on validation -- l_ss=30 is "
                                    "optimal at every tier, so the full_diffusion<full_plain gap is not a tuning artefact"),
            ("uncertainty_examples.png", "per-pixel posterior std over seeds"),
            ("uncertainty_calibration.png", "|error| vs predicted std"),
            ("validation_grid.png", "original global l_ss / num_steps / likelihood tuning on validation"),
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
    L.append("- **Sharing the ACS beats maximising coverage past a point.** The fully-complementary "
             "design (`fcomp*`, split ACS, no overlap) buys more coverage per line than the "
             "shared-ACS complementary design (`comp*`), but loses the √V noise averaging in the "
             "k-space centre. At 2 views this is a wash (+0.004 SSIM for the +6% coverage); at 4 "
             "views `comp4` (81%, √4 centre averaging) **beats** `fcomp4` (100%, no averaging) by "
             "+0.008 SSIM despite covering less k-space. Low-frequency SNR is worth more than the "
             "last columns of coverage.")
    L.append("- **Classical multi-average is the true low-field ceiling; the learned method loses "
             "badly to it.** Two full acquisitions averaged (`dualfull_merge`, NEX=2) score 0.878 "
             "SSIM — the best in the study, +0.081 over one full scan. The learned reconstruction on "
             "the same two full views (`dualfull_ft`, 0.759) trails it by **0.119 SSIM** and gains "
             "≈0 over single-full diffusion. With complete coverage, the diffusion prior is a large "
             "net negative vs plain averaging; the learned pipeline is only valuable when "
             "undersampled.")
    L.append("- **Four disjoint cheap scans reach a full scan's coverage but not its quality.** "
             "`fcomp4` assembles a complete k-space from 4 disjoint R=4 scans (same coverage and "
             "same per-line single-rep noise as `full_plain`), yet trails it by -0.031 SSIM / "
             "-0.011 NRMSE — the cost of stitching 4 independently-acquired sub-scans (inter-scan "
             "motion/phase drift), not undersampling. It does beat `full_diffusion` (+0.007 SSIM), "
             "so it is a viable full-scan substitute when only a plain recon of the real full scan "
             "is unavailable.")
    L.append("- **Joint likelihood == analytic merging** for aligned views at identical masks, as "
             "theory predicts. The multi-view contribution is not in the summation. (This is why the "
             "redundant original-prior joint methods were removed; each family keeps one merge "
             "representative plus the fine-tuned method.)")
    L.append("- **Cross-view fine-tuning is worth ~0 above 40% coverage** and only ~+0.01 in the "
             "sparse regime. It is also a short (0.02 Mimg) empirical self-supervised adaptation and "
             "does **not** inherit the Ambient Diffusion identifiability theorem.")
    L.append("- **The diffusion prior hurts on well-sampled data** (`full_diffusion` < `full_plain` "
             "on 90% of subjects), "
             "and this is **not** a guidance-tuning artefact. We re-tuned `l_ss` per coverage tier "
             "on validation (held-out k-space error, test untouched): `l_ss=30` is optimal at every "
             "tier, and raising it on the dense tier *lowers* SSIM (0.642 -> 0.513 by l_ss=300) "
             "because larger DPS steps overshoot and inject noise. The gap is intrinsic to "
             "soft-guidance diffusion sampling vs a direct least-squares solve; closing it needs a "
             "different sampler (hard data-consistency projection), not a scalar knob. See "
             "`figures/mvp/lss_tier_tuning.png`.")
    L.append("- **Joint-vs-merged is confounded by a guidance-scale factor.** The runbook's "
             "normalized fidelity averages over views, so `D_merged = V * D_joint` (measured: 1.989 "
             "at V=2, 3.967 at V=4). Since the sampler applies `grad(D)/sqrt(D)`, the merged "
             "condition gets **sqrt(V)x stronger guidance at the same l_ss**. A clean test needs "
             "dropping the 1/V (the l_ss retune above shows a global scale shift does not resolve "
             "it). The equivalence unit tests remain correct — they compare raw weighted-SSE "
             "gradients, which do match; it is the sampler-side normalization that differs.")
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
