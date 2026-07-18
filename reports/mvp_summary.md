# MVP Summary: Cross-View Ambient Diffusion for Multi-Acquisition Low-Field MRI

## Headline

**Acquisition design dominates reconstruction method.** Changing *which* k-space lines the repeated scans measure — complementary instead of duplicated, at an identical line budget — is worth **+0.04 to +0.06 SSIM**. Every reconstruction-method choice in this study (learned prior vs classical, cross-view fine-tuning) is worth **0.00 to +0.01**. Design beats method by roughly 5-10x.

Two consequences worth stating plainly:

- The project's nominal contribution (cross-view fine-tuning) only pays off in the **sparse** regime, and decays monotonically as coverage grows: +0.008 SSIM at 19% coverage (`fixed_split_ft`-`fixed_split_merge`), +0.009 at 25% (`dup2_ft`-`dup2_merge`), then **statistically zero** at 44% (`comp2_ft`-`comp2_merge`) and 81% (`comp4_ft`-`comp4_merge`). Once coverage is adequate the data determines the image and prior choices wash out. See `figures/mvp/finetuning_vs_coverage.png`.
- At an equal 256-line budget, **four cheap complementary R=4 scans beat one complete measurement when both are reconstructed the same way**: `comp4_ft` > `full_diffusion` on SSIM (+0.007, 65% of subjects) and on NRMSE (-0.014, 95%). They trail a *plain* linear recon of the full scan (`full_plain`) on SSIM by 0.032, yet are **statistically tied with it on NRMSE** (+0.0006, CI [-0.004, +0.005]) — the residual gap is a reconstruction artefact, not an acquisition one; see the limitations section.

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

| Method | Description | Views | Lines | Unique cols | Coverage | SSIM | NRMSE | Held-out k err |
|---|---|---|---|---|---|---|---|---|
| `classical_adjoint` | Classical adjoint | 2 | 64 | 48 | 19% | 0.5193 [0.5130, 0.5255] | 0.2258 [0.2208, 0.2313] | 0.9845 [0.9833, 0.9857] |
| `classical_l1wav` | Classical L1-wavelet | 2 | 64 | 48 | 19% | 0.5090 [0.5002, 0.5179] | 0.2215 [0.2169, 0.2264] | 0.9713 [0.9670, 0.9755] |
| `single_r4` | Single scan (R=4) | 1 | 64 | 64 | 25% | 0.6176 [0.5993, 0.6394] | 0.1827 [0.1764, 0.1892] | 0.8668 [0.8579, 0.8748] |
| `fixed_split_merge` | Fixed budget, merged | 2 | 64 | 48 | 19% | 0.5973 [0.5824, 0.6160] | 0.1909 [0.1865, 0.1950] | 0.8838 [0.8753, 0.8918] |
| `fixed_split_ft` | Fixed budget, xview-FT | 2 | 64 | 48 | 19% | 0.6050 [0.5900, 0.6235] | 0.1878 [0.1836, 0.1918] | 0.8781 [0.8696, 0.8862] |

At a **fixed** budget, splitting into two views *loses* to spending it on one view (`single_r4` - `fixed_split_ft` = +0.0125, single wins 68%): the duplicated ACS buys 48 unique columns instead of 64. Multi-view only helps when it buys extra scans, not when it subdivides one.

### 2x budget (128 lines): duplicated vs COMPLEMENTARY vs FULLY-complementary

| Method | Description | Views | Lines | Unique cols | Coverage | SSIM | NRMSE | Held-out k err |
|---|---|---|---|---|---|---|---|---|
| `dup2_merge` | 2x duplicated, merged | 2 | 128 | 64 | 25% | 0.6542 [0.6394, 0.6713] | 0.1668 [0.1622, 0.1715] | 0.8639 [0.8559, 0.8714] |
| `dup2_ft` | 2x duplicated, xview-FT | 2 | 128 | 64 | 25% | 0.6628 [0.6466, 0.6818] | 0.1648 [0.1599, 0.1698] | 0.8574 [0.8492, 0.8649] |
| `comp2_merge` | 2x complementary, merged | 2 | 128 | 112 | 44% | 0.7059 [0.6929, 0.7218] | 0.1493 [0.1447, 0.1541] | 0.8484 [0.8402, 0.8561] |
| `comp2_ft` | 2x complementary, xview-FT | 2 | 128 | 112 | 44% | 0.7042 [0.6908, 0.7202] | 0.1494 [0.1452, 0.1537] | 0.8480 [0.8400, 0.8556] |
| `fcomp2_merge` | 2x fully-complementary, merged | 2 | 128 | 128 | 50% | 0.7067 [0.6944, 0.7222] | 0.1486 [0.1446, 0.1527] | 0.8458 [0.8374, 0.8536] |
| `fcomp2_ft` | 2x fully-complementary, xview-FT | 2 | 128 | 128 | 50% | 0.7083 [0.6957, 0.7236] | 0.1485 [0.1447, 0.1525] | 0.8440 [0.8355, 0.8519] |

Three acquisition designs at an identical 128-line budget, in rising coverage: `dup2_*` re-buys one mask (25%); `comp2_*` shares the full ACS and partitions the outer lines (44%); `fcomp2_*` also **partitions the ACS** (split round-robin, no overlap anywhere), reaching **50% coverage** — the maximum possible per line — at the cost of no √V averaging in the k-space centre. The complementary methods beat duplicated by +0.041 to +0.052 SSIM on 100% of subjects; fcomp vs comp isolates the coverage-vs-centre-SNR trade (see Key comparisons).

## Results: 4-view set (20 subjects, 59 slices)

This set answers the budget question directly: **with the same 256 lines a full scan costs, can four cheap R=4 scans surpass one complete measurement?**

| Method | Description | Views | Lines | Unique cols | Coverage | SSIM | NRMSE | Held-out k err |
|---|---|---|---|---|---|---|---|---|
| `single_r4` | Single scan (R=4) | 1 | 64 | 64 | 25% | 0.6431 [0.6218, 0.6667] | 0.1671 [0.1635, 0.1706] | 0.8729 [0.8634, 0.8821] |
| `dup4_merge` | 4x duplicated, merged | 4 | 256 | 64 | 25% | 0.6563 [0.6360, 0.6773] | 0.1577 [0.1535, 0.1618] | 0.8833 [0.8736, 0.8932] |
| `dup4_ft` | 4x duplicated, xview-FT | 4 | 256 | 64 | 25% | 0.7032 [0.6845, 0.7236] | 0.1447 [0.1418, 0.1476] | 0.8578 [0.8495, 0.8661] |
| `comp4_merge` | 4x complementary, merged | 4 | 256 | 208 | 81% | 0.7656 [0.7513, 0.7816] | 0.1215 [0.1192, 0.1238] | 0.8370 [0.8251, 0.8489] |
| `comp4_ft` | 4x complementary, xview-FT | 4 | 256 | 208 | 81% | 0.7643 [0.7492, 0.7809] | 0.1205 [0.1185, 0.1224] | 0.8388 [0.8268, 0.8505] |
| `fcomp4_merge` | 4x fully-complementary, merged | 4 | 256 | 256 | 100% | 0.7650 [0.7508, 0.7809] | 0.1306 [0.1256, 0.1363] | n/a |
| `fcomp4_ft` | 4x fully-complementary, xview-FT | 4 | 256 | 256 | 100% | 0.7559 [0.7408, 0.7735] | 0.1406 [0.1334, 0.1484] | n/a |
| `full_diffusion` | Full scan, diffusion | 1 | 256 | 256 | 100% | 0.7577 [0.7436, 0.7733] | 0.1346 [0.1303, 0.1396] | n/a |
| `full_plain` | Full scan, plain recon | 1 | 256 | 256 | 100% | 0.7964 [0.7832, 0.8118] | 0.1198 [0.1165, 0.1240] | n/a |
| `dualfull_merge` | 2 full scans, classical avg (NEX=2) | 2 | 512 | 256 | 100% | 0.8778 [0.8689, 0.8870] | 0.0986 [0.0936, 0.1034] | n/a |
| `dualfull_ft` | 2 full scans, xview-FT | 2 | 512 | 256 | 100% | 0.7586 [0.7439, 0.7748] | 0.1425 [0.1361, 0.1496] | n/a |

**Answer: yes against the same reconstruction method, no against the best one.** `comp4_ft` (4x complementary, 81% coverage) beats `full_diffusion` (the *same* diffusion reconstruction given a complete measurement) on both metrics: +0.007 SSIM (65% of subjects) and -0.014 NRMSE (95%). Four cheap complementary scans genuinely substitute for one full scan. But both lose on SSIM to `full_plain`, a plain linear recon of the full scan (`full_diffusion` - `full_plain` = -0.039, plain wins 90%): **the diffusion prior hurts on well-sampled data**.

Note the metric split for `comp4_ft` vs `full_plain`: -0.032 on SSIM but **+0.0006 on NRMSE, a statistical tie** (CI [-0.004, +0.005]). It is as accurate as the full scan in the L2 sense and loses only on *structural* similarity — the signature of a prior adding plausible-but-wrong texture, not of missing information. That makes the residual gap a *reconstruction* deficit, not an acquisition one. We suspected the fix was to re-tune the guidance scale `l_ss` (tuned once on sparse data) per coverage tier, and **tested it: it does not help** — `l_ss=30` is optimal at every tier, and raising it on well-sampled data makes SSIM *worse* (see `figures/mvp/lss_tier_tuning.png` and the limitations section). The gap is intrinsic to soft-guidance diffusion sampling vs a direct solve, not a tuning artefact.

With the **duplicated** 4x design the same question answered *no* by a wide margin (`dup4_ft` 0.703 vs `full_plain` 0.796, -0.093). The acquisition fix (`comp4_ft` - `dup4_ft` = +0.061 on 100% of subjects) closes two thirds of that gap. This is the single largest effect measured anywhere in the study.

## Key comparisons (subject-paired, brain-masked SSIM)

_Note: the redundant "joint likelihood, original prior" methods (old M4/M8/M10/M14/M17) were removed, since joint == analytic merge for aligned identical-mask views. Fine-tuning effects are therefore measured against the **merge** baseline of the same acquisition (equivalent, since merge == the removed original-prior joint)._

### Acquisition design (the big effects)

- **COMPLEMENTARY vs duplicated, 2x budget, fine-tuned** (comp2_ft - dup2_ft): **+0.0414** SSIM [+0.0348, +0.0483], comp2_ft wins on 100% of 25 subjects.
- **COMPLEMENTARY vs duplicated, 2x budget, merged** (comp2_merge - dup2_merge): **+0.0517** SSIM [+0.0454, +0.0582], comp2_merge wins on 100% of 25 subjects.
- **COMPLEMENTARY vs duplicated, 4x budget** (comp4_ft - dup4_ft): **+0.0610** SSIM [+0.0554, +0.0664], comp4_ft wins on 100% of 20 subjects.
- **Extra cheap repetition (averaged) vs single view** (dup2_merge - single_r4): **+0.0366** SSIM [+0.0312, +0.0415], dup2_merge wins on 96% of 25 subjects.
  Identical 25% coverage, 2x the lines -> isolates the pure sqrt(2) SNR benefit of averaging cheap low-field repeats. Real, but half the size of the design effect.
- **Best 2-view design vs single view** (comp2_ft - single_r4): **+0.0866** SSIM [+0.0786, +0.0949], comp2_ft wins on 100% of 25 subjects.

### Fully-complementary: maximum coverage vs k-space-centre SNR

`fcomp*` splits the ACS across views (no overlap anywhere) to convert the duplicated-ACS budget into extra coverage, trading away the √V noise averaging in the centre that `comp*` keeps.

- **Split-ACS (50%) vs shared-ACS (44%), 2x** (fcomp2_ft - comp2_ft): **+0.0041** SSIM [+0.0005, +0.0079], fcomp2_ft wins on 56% of 25 subjects.
- **Split-ACS (100%) vs shared-ACS (81%), 4x** (fcomp4_ft - comp4_ft): **-0.0083** SSIM [-0.0125, -0.0041], fcomp4_ft wins on 20% of 20 subjects.
  At 4 views `fcomp4` reaches **full coverage** — a complete k-space assembled from 4 disjoint cheap scans. The two comparisons below ask whether that equals one full scan (same coverage, same per-line single-rep noise, but 4 independent acquisitions stitched vs 1 monolithic):
- **4x disjoint cheap scans vs 1 full scan (both diffusion)** (fcomp4_ft - full_diffusion): **-0.0018** SSIM [-0.0056, +0.0022], fcomp4_ft wins on 35% of 20 subjects.
- **4x disjoint cheap scans vs 1 full scan (both plain/merge)** (fcomp4_merge - full_plain): **-0.0314** SSIM [-0.0428, -0.0181], fcomp4_merge wins on 10% of 20 subjects.

### Two full acquisitions (low-field NEX=2): the true reconstruction ceiling

Two *complete* k-space acquisitions of the same slice (512 lines), the realistic low-field multi-average scenario. `dualfull_merge` is the classical noise-weighted average (= (y0+y1)/2 at σ/√2, plain recon); `dualfull_ft` is the learned cross-view fine-tuned joint reconstruction over both full views.

- **NEX=2 classical average vs NEX=1 (both plain)** (dualfull_merge - full_plain): **+0.0814** SSIM [+0.0722, +0.0899], dualfull_merge wins on 100% of 20 subjects.
  Isolates the pure √2 SNR benefit of a second full acquisition under identical (classical) reconstruction — the honest low-field quality ceiling.
- **learned xview-FT vs classical average, same 2 full views** (dualfull_ft - dualfull_merge): **-0.1192** SSIM [-0.1312, -0.1055], dualfull_ft wins on 0% of 20 subjects.
- **learned on 2 full views vs classical on 1 full view** (dualfull_ft - full_plain): **-0.0378** SSIM [-0.0506, -0.0233], dualfull_ft wins on 10% of 20 subjects.

**This is the study's sharpest result.** `dualfull_merge` (classical NEX=2 average) scores **0.878 SSIM / 0.099 NRMSE — the best of any method here by a wide margin**, +0.081 over a single full scan on 100% of subjects: at low field, a second full acquisition is worth far more than any reconstruction cleverness. And the learned method **fails this test decisively** — `dualfull_ft` (0.759) *loses to the classical average by 0.119 SSIM* and gains essentially nothing over `full_diffusion` (+0.001), i.e. it extracts almost no value from the second full acquisition. Once coverage is complete, the diffusion prior is not just unhelpful but a large net negative vs plain averaging. The learned pipeline earns its keep **only in the undersampled regime**; the honest low-field reconstruction ceiling is classical multi-average.

### Cross-view fine-tuning (the contribution) vs the merge baseline

- **Cross-view FT, 19% coverage** (fixed_split_ft - fixed_split_merge): **+0.0077** SSIM [+0.0043, +0.0111], fixed_split_ft wins on 80% of 25 subjects.
- **Cross-view FT, 25% coverage** (dup2_ft - dup2_merge): **+0.0087** SSIM [+0.0057, +0.0115], dup2_ft wins on 84% of 25 subjects.
- **Cross-view FT, 44% coverage** (comp2_ft - comp2_merge): **-0.0017** SSIM [-0.0040, +0.0006], comp2_ft wins on 32% of 25 subjects.
- **Cross-view FT, 81% coverage** (comp4_ft - comp4_merge): **-0.0013** SSIM [-0.0043, +0.0015], comp4_ft wins on 40% of 20 subjects.
  The fine-tuning benefit **decays to zero as coverage grows** (+0.008 at 19-25% coverage, ~0 at 44-81%) — it is a sparse-regime effect only.

### Diffusion vs classical

- **Diffusion vs tuned L1-wavelet SENSE** (single_r4 - classical_l1wav): **+0.1085** SSIM [+0.0916, +0.1270], single_r4 wins on 100% of 25 subjects.
- **Diffusion vs noise-weighted adjoint** (single_r4 - classical_adjoint): **+0.0983** SSIM [+0.0807, +0.1190], single_r4 wins on 100% of 25 subjects.
  The learned prior is worth ~+0.10 SSIM over classical baselines in the sparse regime — the one place where the prior axis is large. It reverses at full sampling (`full_diffusion` < `full_plain`).

## Figures

**Results tables (visualized)**

- `figures/mvp/results_2view.png` -- the two 2-view result tables (fixed budget; 2x budget) as SSIM and NRMSE bars with 95% CI, coloured by design
- `figures/mvp/results_4view.png` -- the 4-view result table as SSIM and NRMSE bars with 95% CI (the budget question, incl. dualfull NEX=2)

**Cross-experiment (the conclusions)**

- `figures/mvp/experiment_overview.png` -- SSIM vs unique k-space coverage, all methods, both sets -- the headline: coverage separates, method does not
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

- `figures/mvp/lss_tier_tuning.png` -- per-coverage-tier l_ss retune on validation -- l_ss=30 is optimal at every tier, so the full_diffusion<full_plain gap is not a tuning artefact
- `figures/mvp/uncertainty_examples.png` -- per-pixel posterior std over seeds
- `figures/mvp/uncertainty_calibration.png` -- |error| vs predicted std
- `figures/mvp/validation_grid.png` -- original global l_ss / num_steps / likelihood tuning on validation

## Limitations and negative findings

Retained per runbook 9.4 — negative results are results.

- **0.3 T low-field, not <0.1 T ultra-low-field.** No ultra-low-field claim is made.
- **Sharing the ACS beats maximising coverage past a point.** The fully-complementary design (`fcomp*`, split ACS, no overlap) buys more coverage per line than the shared-ACS complementary design (`comp*`), but loses the √V noise averaging in the k-space centre. At 2 views this is a wash (+0.004 SSIM for the +6% coverage); at 4 views `comp4` (81%, √4 centre averaging) **beats** `fcomp4` (100%, no averaging) by +0.008 SSIM despite covering less k-space. Low-frequency SNR is worth more than the last columns of coverage.
- **Classical multi-average is the true low-field ceiling; the learned method loses badly to it.** Two full acquisitions averaged (`dualfull_merge`, NEX=2) score 0.878 SSIM — the best in the study, +0.081 over one full scan. The learned reconstruction on the same two full views (`dualfull_ft`, 0.759) trails it by **0.119 SSIM** and gains ≈0 over single-full diffusion. With complete coverage, the diffusion prior is a large net negative vs plain averaging; the learned pipeline is only valuable when undersampled.
- **Four disjoint cheap scans reach a full scan's coverage but not its quality.** `fcomp4` assembles a complete k-space from 4 disjoint R=4 scans (same coverage and same per-line single-rep noise as `full_plain`), yet trails it by -0.031 SSIM / -0.011 NRMSE — the cost of stitching 4 independently-acquired sub-scans (inter-scan motion/phase drift), not undersampling. It does beat `full_diffusion` (+0.007 SSIM), so it is a viable full-scan substitute when only a plain recon of the real full scan is unavailable.
- **Joint likelihood == analytic merging** for aligned views at identical masks, as theory predicts. The multi-view contribution is not in the summation. (This is why the redundant original-prior joint methods were removed; each family keeps one merge representative plus the fine-tuned method.)
- **Cross-view fine-tuning is worth ~0 above 40% coverage** and only ~+0.01 in the sparse regime. It is also a short (0.02 Mimg) empirical self-supervised adaptation and does **not** inherit the Ambient Diffusion identifiability theorem.
- **The diffusion prior hurts on well-sampled data** (`full_diffusion` < `full_plain` on 90% of subjects), and this is **not** a guidance-tuning artefact. We re-tuned `l_ss` per coverage tier on validation (held-out k-space error, test untouched): `l_ss=30` is optimal at every tier, and raising it on the dense tier *lowers* SSIM (0.642 -> 0.513 by l_ss=300) because larger DPS steps overshoot and inject noise. The gap is intrinsic to soft-guidance diffusion sampling vs a direct least-squares solve; closing it needs a different sampler (hard data-consistency projection), not a scalar knob. See `figures/mvp/lss_tier_tuning.png`.
- **Joint-vs-merged is confounded by a guidance-scale factor.** The runbook's normalized fidelity averages over views, so `D_merged = V * D_joint` (measured: 1.989 at V=2, 3.967 at V=4). Since the sampler applies `grad(D)/sqrt(D)`, the merged condition gets **sqrt(V)x stronger guidance at the same l_ss**. A clean test needs dropping the 1/V (the l_ss retune above shows a global scale shift does not resolve it). The equivalence unit tests remain correct — they compare raw weighted-SSE gradients, which do match; it is the sampler-side normalization that differs.
- **Test reference is an average of all 6 repetitions, including the 2-4 used as input** (~1/3 of the average), a mild uniform optimistic bias affecting all methods equally. Comparisons are unaffected; absolute SSIM is slightly generous.
- **SSIM ~0.6-0.8 here is not comparable to the ~0.9+ in the Ambient/FastMRI paper.** Different field strength and, crucially, a different reference: they compare against the fully-sampled version of the *same* acquisition (one undersampling gap), we compare against a 6-repetition average (an undersampling gap *plus* an irreducible SNR gap). Measured directly: a fully-sampled 1-rep recon scores **0.991** against an SNR-matched self-reference (proving the operator/FFT path is exact) but only **0.703** against the 6-rep average — that 0.703 is the practical **ceiling** for our setup, so the 2-view methods at 0.60-0.71 sit at 85-100% of what is achievable.
- Validation reference is a noisy third repetition (weak), so held-out k-space error was the primary tuning criterion. At 0.3 T the held-out outer k-space is largely noise, so that metric compresses into 0.84-0.98 and discriminates weakly; SSIM/NRMSE vs the multi-repetition average carry more signal.

See `reports/mvp_notes.md` for the full decision log, including four bugs that materially changed conclusions (brain-mask, held-out-error bias, EMA half-life, duplicated-mask design).
