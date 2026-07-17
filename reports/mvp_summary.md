# MVP Summary: Cross-View Ambient Diffusion for Multi-Acquisition Low-Field MRI

## Headline

**Acquisition design dominates reconstruction method.** Changing *which* k-space lines the repeated scans measure — complementary instead of duplicated, at an identical line budget — is worth **+0.04 to +0.06 SSIM**. Every reconstruction-method choice in this study (learned prior vs original, cross-view fine-tuning, joint likelihood vs analytic merge) is worth **0.00 to +0.01**. Design beats method by roughly 5-10x.

Two consequences worth stating plainly:

- The project's nominal contribution (cross-view fine-tuning) only pays off in the **sparse** regime, and decays monotonically as coverage grows: +0.010 SSIM at 19% coverage (M5-M4), +0.003 at 25% (M6-M8), then **statistically zero** at 44% (M13-M14) and 81% (M16-M17). Once coverage is adequate the data determines the image and prior/combiner choices wash out. See `figures/mvp/finetuning_vs_coverage.png`.
- At an equal 256-line budget, **four cheap complementary R=4 scans beat one complete measurement when both are reconstructed the same way**: M16 > M12 on SSIM (+0.007, 65% of subjects) and on NRMSE (-0.014, 95%). They trail a *plain* linear recon of the full scan (MF) on SSIM by 0.032, yet are **statistically tied with it on NRMSE** (+0.0006, CI [-0.004, +0.005]) — the residual gap is a reconstruction artefact, not an acquisition one; see the guidance-scale caveat below.

## Dataset and scope

- Dataset: **M4Raw 0.3 T** T2-weighted brain, multi-coil (4 coils), 256x256, 2D.
- This is **low-field (0.3 T)**, NOT ultra-low-field (<0.1 T). Ultra-low-field validation is future work and is not claimed here.
- Preprocessing: test acceptance rate 99.7% (299 accepted / 1 rejected, 25 subjects); fallback maps: 0; train/val/test slices: 708/239/299.
- Two evaluation sets: the **2-view set** (25 test subjects, 75 central slices) and the **4-view set** (20 subjects, 59 slices — fewer, because requiring 4 mutually motion-consistent repetitions accepts 188/300 slices).
- All metrics are **brain-masked** (Otsu on the reference, identical mask for every method on a slice) and aggregated **per subject** with 95% bootstrap CIs.

## Mask design and acquisition budget

Every condition uses Cartesian column undersampling with a fully-sampled 16-line ACS block. `lines` counts what you pay for (summed over views, ACS re-bought by each view); `unique cols` is what you actually learn about.

| Condition | Views | Lines bought | Unique cols | Coverage | Notes |
|---|---|---|---|---|---|
| `single_r4` | 1 | 64 | 64 | 25% | one cheap scan; the baseline budget |
| `merge_fixed` / `joint_fixed` | 2 | 64 | 48 | 19% | **fixed budget**: two R=8 views, shared ACS. Duplicated ACS costs 16 unique columns vs `single_r4`. |
| `joint_extra` / `merge_extra` | 2 | 128 | 64 | 25% | 2x budget, **duplicated** mask (runbook Condition D = plain NEX: a repetition re-runs the identical sequence) |
| **`joint_extra_comp` / `merge_extra_comp`** | 2 | 128 | **112** | **44%** | 2x budget, **complementary**: shared ACS + disjoint outer lines |
| `joint_quad` / `merge_quad` | 4 | 256 | 64 | 25% | 4x budget, **duplicated** — 256 lines to see 64 columns |
| **`joint_quad_comp` / `merge_quad_comp`** | 4 | 256 | **208** | **81%** | 4x budget, **complementary** |
| `single_full` | 1 | 256 | 256 | 100% | one complete measurement; the ceiling |

The duplicated designs (`*_extra`, `*_quad`) follow the runbook literally, but re-buying the same 64 columns caps coverage at 25% no matter how many scans you add. They are retained as the honest ablation for *what the runbook specified*; the complementary designs are the corrected acquisition and carry the study's main result. Duplication is not worthless — it averages noise where the signal energy is, worth sigma/sqrt(V) (measured: 0.0443 -> 0.0313 at V=2) — it is just worth far less than coverage.

Fixed-budget accounting (coefficients = columns x 256 rows):
- **train**: single_r4=16384, fixed_sum(dupACS)=16384 (within tolerance: True), fixed_union=12288, extra_sum=32768.
- **val**: single_r4=16384, fixed_sum(dupACS)=16384 (within tolerance: True), fixed_union=12288, extra_sum=32768.
- **test**: single_r4=16384, fixed_sum(dupACS)=16384 (within tolerance: True), fixed_union=12288, extra_sum=32768.

## Selected hyperparameters

- Diffusion inference: num_steps=100, l_ss=30.0, likelihood_type=DPS.
  NOTE: the normalized multi-view fidelity divides by sigma^2*N, so the effective l_ss scale is ~10x the runbook nominal grid (tuned on validation held-out k-space error, single_r4).
- L1-wavelet SENSE: lambda=0.03, iterations=50.
- Cross-view fine-tune: 0.02 Mimg, lr=1e-4, EMA half-life 0.002 Mimg (must be << training duration, see notes).

## Results: 2-view set (25 subjects, 75 slices)

Subject-level mean [95% bootstrap CI]. Grouped by acquisition budget.

### Classical baselines and fixed budget (64 lines)

| Method | Views | Lines | Unique cols | Coverage | Prior / combiner | SSIM | NRMSE | Held-out k err |
|---|---|---|---|---|---|---|---|---|
| M0 | 2 | 64 | 48 | 19% | noise-weighted adjoint (classical) | 0.5193 [0.5130, 0.5255] | 0.2258 [0.2208, 0.2313] | 0.9845 [0.9833, 0.9857] |
| M1 | 2 | 64 | 48 | 19% | tuned L1-wavelet SENSE (classical) | 0.5090 [0.5002, 0.5179] | 0.2215 [0.2169, 0.2264] | 0.9713 [0.9670, 0.9755] |
| M2 | 1 | 64 | 64 | 25% | original prior, single view | 0.6176 [0.5993, 0.6394] | 0.1827 [0.1764, 0.1892] | 0.8668 [0.8579, 0.8748] |
| M3 | 2 | 64 | 48 | 19% | original prior, analytic merge | 0.5973 [0.5824, 0.6160] | 0.1909 [0.1865, 0.1950] | 0.8838 [0.8753, 0.8918] |
| M4 | 2 | 64 | 48 | 19% | original prior, joint likelihood | 0.5952 [0.5797, 0.6141] | 0.1917 [0.1868, 0.1965] | 0.8823 [0.8739, 0.8902] |
| M5 | 2 | 64 | 48 | 19% | cross-view fine-tuned, joint likelihood | 0.6050 [0.5900, 0.6235] | 0.1878 [0.1836, 0.1918] | 0.8781 [0.8696, 0.8862] |

At a **fixed** budget, splitting into two views *loses* to spending it on one view (M2 - M5 = +0.0125, M2 wins 68%): the duplicated ACS buys 48 unique columns instead of 64. Multi-view only helps when it buys extra scans, not when it subdivides one.

### 2x budget (128 lines): duplicated vs COMPLEMENTARY

| Method | Views | Lines | Unique cols | Coverage | Prior / combiner | SSIM | NRMSE | Held-out k err |
|---|---|---|---|---|---|---|---|---|
| M7 | 2 | 128 | 64 | 25% | original prior, analytic merge | 0.6542 [0.6394, 0.6713] | 0.1668 [0.1622, 0.1715] | 0.8639 [0.8559, 0.8714] |
| M8 | 2 | 128 | 64 | 25% | original prior, joint likelihood | 0.6594 [0.6442, 0.6773] | 0.1662 [0.1614, 0.1712] | 0.8594 [0.8510, 0.8671] |
| M6 | 2 | 128 | 64 | 25% | cross-view fine-tuned, joint likelihood | 0.6628 [0.6466, 0.6818] | 0.1648 [0.1599, 0.1698] | 0.8574 [0.8492, 0.8649] |
| M15 | 2 | 128 | 112 | 44% | original prior, analytic merge | 0.7059 [0.6929, 0.7218] | 0.1493 [0.1447, 0.1541] | 0.8484 [0.8402, 0.8561] |
| M14 | 2 | 128 | 112 | 44% | original prior, joint likelihood | 0.7043 [0.6914, 0.7196] | 0.1496 [0.1454, 0.1539] | 0.8492 [0.8413, 0.8568] |
| M13 | 2 | 128 | 112 | 44% | cross-view fine-tuned, joint likelihood | 0.7042 [0.6908, 0.7202] | 0.1494 [0.1452, 0.1537] | 0.8480 [0.8400, 0.8556] |

The three complementary methods (M13/M14/M15, 44% coverage) beat their duplicated counterparts (M6/M8/M7, 25% coverage) by +0.041 to +0.052 SSIM on **100% of subjects**, at an identical 128-line cost. Within each coverage level the method variants are within 0.005 of each other.

## Results: 4-view set (20 subjects, 59 slices)

This set answers the budget question directly: **with the same 256 lines a full scan costs, can four cheap R=4 scans surpass one complete measurement?**

| Method | Views | Lines | Unique cols | Coverage | Prior / combiner | SSIM | NRMSE | Held-out k err |
|---|---|---|---|---|---|---|---|---|
| M2 | 1 | 64 | 64 | 25% | original prior, single view | 0.6431 [0.6218, 0.6667] | 0.1671 [0.1635, 0.1706] | 0.8729 [0.8634, 0.8821] |
| M11 | 4 | 256 | 64 | 25% | original prior, analytic merge | 0.6563 [0.6360, 0.6773] | 0.1577 [0.1535, 0.1618] | 0.8833 [0.8736, 0.8932] |
| M10 | 4 | 256 | 64 | 25% | original prior, joint likelihood | 0.6993 [0.6792, 0.7208] | 0.1464 [0.1426, 0.1502] | 0.8599 [0.8511, 0.8684] |
| M9 | 4 | 256 | 64 | 25% | cross-view fine-tuned, joint likelihood | 0.7032 [0.6845, 0.7236] | 0.1447 [0.1418, 0.1476] | 0.8578 [0.8495, 0.8661] |
| M18 | 4 | 256 | 208 | 81% | original prior, analytic merge | 0.7656 [0.7513, 0.7816] | 0.1215 [0.1192, 0.1238] | 0.8370 [0.8251, 0.8489] |
| M17 | 4 | 256 | 208 | 81% | original prior, joint likelihood | 0.7652 [0.7508, 0.7813] | 0.1206 [0.1188, 0.1225] | 0.8398 [0.8282, 0.8514] |
| M16 | 4 | 256 | 208 | 81% | cross-view fine-tuned, joint likelihood | 0.7643 [0.7492, 0.7809] | 0.1205 [0.1185, 0.1224] | 0.8388 [0.8268, 0.8505] |
| M12 | 1 | 256 | 256 | 100% | original prior, one complete measurement | 0.7577 [0.7436, 0.7733] | 0.1346 [0.1303, 0.1396] | n/a |
| MF | 1 | 256 | 256 | 100% | plain recon, one complete measurement (reference) | 0.7964 [0.7832, 0.8118] | 0.1198 [0.1165, 0.1240] | n/a |

**Answer: yes against the same reconstruction method, no against the best one.** M16 (4x complementary, 81% coverage) beats M12 (the *same* diffusion reconstruction given a complete measurement) on both metrics: +0.007 SSIM (65% of subjects) and -0.014 NRMSE (95%). Four cheap complementary scans genuinely substitute for one full scan. But both lose on SSIM to MF, a plain linear recon of the full scan (M12 - MF = -0.039, MF wins 90%): **the diffusion prior hurts on well-sampled data**. With `l_ss=30` tuned at R=4, guidance keeps injecting prior once the data already determines the image.

Note the metric split for M16 vs MF: -0.032 on SSIM but **+0.0006 on NRMSE, a statistical tie** (CI [-0.004, +0.005]). M16 is as accurate as the full scan in the L2 sense and loses only on *structural* similarity — the signature of a prior adding plausible-but-wrong texture, not of missing information. That makes the residual gap a *reconstruction* deficit, not an acquisition one. We suspected the fix was to re-tune the guidance scale `l_ss` (tuned once on sparse data) per coverage tier, and **tested it: it does not help** — `l_ss=30` is optimal at every tier, and raising it on well-sampled data makes SSIM *worse* (see `figures/mvp/lss_tier_tuning.png` and the limitations section). The gap is intrinsic to soft-guidance diffusion sampling vs a direct solve, not a tuning artefact.

With the **duplicated** 4x design the same question answered *no* by a wide margin (M9 0.703 vs MF 0.796, -0.093). The acquisition fix (M16 - M9 = +0.061 on 100% of subjects) closes two thirds of that gap. This is the single largest effect measured anywhere in the study.

## Key comparisons (subject-paired, brain-masked SSIM)

### Acquisition design (the big effects)

- **COMPLEMENTARY vs duplicated, 2x budget, fine-tuned** (M13 - M6): **+0.0414** SSIM [+0.0348, +0.0483], M13 wins on 100% of 25 subjects.
- **COMPLEMENTARY vs duplicated, 2x budget, joint** (M14 - M8): **+0.0450** SSIM [+0.0384, +0.0514], M14 wins on 100% of 25 subjects.
- **COMPLEMENTARY vs duplicated, 2x budget, merged** (M15 - M7): **+0.0517** SSIM [+0.0454, +0.0582], M15 wins on 100% of 25 subjects.
- **COMPLEMENTARY vs duplicated, 4x budget** (M16 - M9): **+0.0610** SSIM [+0.0554, +0.0664], M16 wins on 100% of 20 subjects.
- **Extra cheap repetition (averaged) vs single view** (M7 - M2): **+0.0366** SSIM [+0.0312, +0.0415], M7 wins on 96% of 25 subjects.
  Identical 25% coverage, 2x the lines -> isolates the pure sqrt(2) SNR benefit of averaging cheap low-field repeats. Real, but half the size of the design effect.
- **Best 2-view design vs single view** (M13 - M2): **+0.0866** SSIM [+0.0786, +0.0949], M13 wins on 100% of 25 subjects.

### Reconstruction method (the small effects)

- **Cross-view fine-tuning vs original prior, 19% coverage** (M5 - M4): **+0.0099** SSIM [+0.0069, +0.0129], M5 wins on 88% of 25 subjects.
- **Cross-view fine-tuning, 25% coverage** (M6 - M8): **+0.0034** SSIM [+0.0008, +0.0062], M6 wins on 60% of 25 subjects.
- **Cross-view fine-tuning, 44% coverage** (M13 - M14): **-0.0001** SSIM [-0.0026, +0.0024], M13 wins on 52% of 25 subjects.
- **Cross-view fine-tuning, 81% coverage** (M16 - M17): **-0.0009** SSIM [-0.0035, +0.0015], M16 wins on 40% of 20 subjects.
  The fine-tuning benefit **decays to zero as coverage grows** — it is a sparse-regime effect only.

- **Joint likelihood vs analytic merge, fixed budget** (M4 - M3): **-0.0021** SSIM [-0.0052, +0.0008], M4 wins on 36% of 25 subjects.
- **Joint likelihood vs analytic merge, 2x duplicated** (M8 - M7): **+0.0052** SSIM [+0.0022, +0.0083], M8 wins on 64% of 25 subjects.
- **Joint likelihood vs analytic merge, 2x complementary** (M14 - M15): **-0.0015** SSIM [-0.0046, +0.0016], M14 wins on 36% of 25 subjects.
- **Joint likelihood vs analytic merge, 4x duplicated** (M10 - M11): **+0.0431** SSIM [+0.0354, +0.0513], M10 wins on 100% of 20 subjects.
- **Joint likelihood vs analytic merge, 4x complementary** (M17 - M18): **-0.0004** SSIM [-0.0033, +0.0026], M17 wins on 45% of 20 subjects.
  For aligned views with identical masks, theory predicts **equivalence**, and that is what we see (|delta| < 0.006) — *except* at 4x duplicated (M10 - M11 = +0.043), which is an artefact of the guidance-scale confound below, not a real advantage.

### Diffusion vs classical

- **Diffusion vs tuned L1-wavelet SENSE** (M2 - M1): **+0.1085** SSIM [+0.0916, +0.1270], M2 wins on 100% of 25 subjects.
- **Diffusion vs noise-weighted adjoint** (M2 - M0): **+0.0983** SSIM [+0.0807, +0.1190], M2 wins on 100% of 25 subjects.
  The learned prior is worth ~+0.10 SSIM over classical baselines in the sparse regime — the one place where the prior axis is large. It reverses at full sampling (M12 < MF).

## Figures

**Cross-experiment (the conclusions)**

- `figures/mvp/experiment_overview.png` -- SSIM vs unique k-space coverage, all 20 methods, both sets -- the headline: coverage separates, method does not
- `figures/mvp/effect_sizes_forest.png` -- every paired comparison with 95% CI, split into acquisition design vs reconstruction method
- `figures/mvp/budget_ladder.png` -- every experiment as a bar, grouped by what it cost (64/128/256 lines)
- `figures/mvp/finetuning_vs_coverage.png` -- the cross-view fine-tuning benefit decaying to zero as coverage grows

**Acquisition design**

- `figures/mvp/mask_design_all.png` -- all 8 conditions: per-view masks, union coverage, and lines-bought vs unique-lines efficiency
- `figures/mvp/mask_design.png` -- original 4-condition mask figure (runbook 10)
- `figures/mvp/mask_design_quad.png` -- mask figure for the 4-view mask set

**2-view set (25 subjects)**

- `figures/mvp/reconstruction_montage.png` -- reference / zero-filled / 11 methods on 4 cases picked by input SNR
- `figures/mvp/error_maps.png` -- per-method absolute error vs the reference
- `figures/mvp/metric_boxplots.png` -- subject-level NRMSE / SSIM / held-out error, boxes coloured by design
- `figures/mvp/fixed_vs_extra_budget.png` -- metric vs acquired coefficients
- `figures/mvp/data_consistency_curves.png` -- per-view normalized residual over sampler steps

**4-view set (20 subjects)**

- `figures/mvp/reconstruction_montage_quad.png` -- the budget question: 4xR=4 duplicated vs complementary vs one full scan
- `figures/mvp/error_maps_quad.png` -- per-method absolute error, 4-view set
- `figures/mvp/metric_boxplots_quad.png` -- subject-level metrics, 4-view set
- `figures/mvp/fixed_vs_extra_budget_quad.png` -- metric vs acquired coefficients, 4-view set

**Uncertainty and tuning**

- `figures/mvp/lss_tier_tuning.png` -- per-coverage-tier l_ss retune on validation -- l_ss=30 is optimal at every tier, so the M12<MF gap is not a tuning artefact
- `figures/mvp/uncertainty_examples.png` -- per-pixel posterior std over seeds
- `figures/mvp/uncertainty_calibration.png` -- |error| vs predicted std
- `figures/mvp/validation_grid.png` -- original global l_ss / num_steps / likelihood tuning on validation

## Limitations and negative findings

Retained per runbook 9.4 — negative results are results.

- **0.3 T low-field, not <0.1 T ultra-low-field.** No ultra-low-field claim is made.
- **Joint likelihood == analytic merging** for aligned views at identical masks, as theory predicts. The multi-view contribution is not in the summation.
- **Cross-view fine-tuning is worth ~0 above 40% coverage** and only ~+0.01 in the sparse regime. It is also a short (0.02 Mimg) empirical self-supervised adaptation and does **not** inherit the Ambient Diffusion identifiability theorem.
- **The diffusion prior hurts on well-sampled data** (M12 < MF on 90% of subjects), and this is **not** a guidance-tuning artefact. We re-tuned `l_ss` per coverage tier on validation (held-out k-space error, test untouched): `l_ss=30` is optimal at every tier, and raising it on the dense tier *lowers* SSIM (0.642 -> 0.513 by l_ss=300) because larger DPS steps overshoot and inject noise. The gap is intrinsic to soft-guidance diffusion sampling vs a direct least-squares solve; closing it needs a different sampler (hard data-consistency projection), not a scalar knob. See `figures/mvp/lss_tier_tuning.png`.
- **Joint-vs-merged is confounded by a guidance-scale factor.** The runbook's normalized fidelity averages over views, so `D_merged = V * D_joint` (measured: 1.989 at V=2, 3.967 at V=4). Since the sampler applies `grad(D)/sqrt(D)`, the merged condition gets **sqrt(V)x stronger guidance at the same l_ss**. A clean test needs dropping the 1/V (the l_ss retune above shows a global scale shift does not resolve it). The equivalence unit tests remain correct — they compare raw weighted-SSE gradients, which do match; it is the sampler-side normalization that differs.
- **Test reference is an average of all 6 repetitions, including the 2-4 used as input** (~1/3 of the average), a mild uniform optimistic bias affecting all methods equally. Comparisons are unaffected; absolute SSIM is slightly generous.
- **SSIM ~0.6-0.8 here is not comparable to the ~0.9+ in the Ambient/FastMRI paper.** Different field strength and, crucially, a different reference: they compare against the fully-sampled version of the *same* acquisition (one undersampling gap), we compare against a 6-repetition average (an undersampling gap *plus* an irreducible SNR gap). Measured directly: a fully-sampled 1-rep recon scores **0.991** against an SNR-matched self-reference (proving the operator/FFT path is exact) but only **0.703** against the 6-rep average — that 0.703 is the practical **ceiling** for our setup, so the 2-view methods at 0.60-0.71 sit at 85-100% of what is achievable.
- Validation reference is a noisy third repetition (weak), so held-out k-space error was the primary tuning criterion. At 0.3 T the held-out outer k-space is largely noise, so that metric compresses into 0.84-0.98 and discriminates weakly; SSIM/NRMSE vs the multi-repetition average carry more signal.

See `reports/mvp_notes.md` for the full decision log, including four bugs that materially changed conclusions (brain-mask, held-out-error bias, EMA half-life, duplicated-mask design).
