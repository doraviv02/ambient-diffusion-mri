# MVP running notes / key decisions (feeds the final report)

Decision log for the cross-view ambient diffusion MVP. Contains every choice that
changed a number, and every bug that changed a conclusion. Results tables live in
`reports/mvp_summary.md`; this file records *why* they look the way they do.

All paired statistics quoted here are **subject-level** (bootstrap over subjects),
matching `analysis/build_mvp_report.py`. Earlier drafts of this file quoted
slice-level win rates for a few comparisons; those have been recomputed.

---

# 1. Setup

## Environment
- Cluster: 4× RTX 2080 Ti (11 GB each). Using GPUs 2,3 (0 is used by another job).
- No conda initially → installed Miniconda; env `ambient-mv`.
- environment.yml pins pytorch=2.0.0 but the pip section pulled **torch 2.8.0+cu128
  / python 3.9.25 / numpy 2.0.2** (native for the driver 590 / CUDA 12.8). Kept it:
  the published R=4 checkpoint reconstructs and strict-loads **638/638 tensors
  (100%)**, so architecture compatibility is exact.
- GPUs are 11 GB, not the runbook's assumed 12 GB → memory settings adapt
  (microbatch 1, fp16, no torch.compile).

## FFT / tensor convention (Stage-0 gate)
- Repo `MRI_utils` uses **plain-ortho FFT on `fftmod`-ed maps + k-space**; the
  image is 2-channel real (real=ch0, imag=ch1).
- Decision: preprocessing stores **natural** (un-fftmod) centered k-space + maps;
  inference/loss apply `fftmod` at load (byte-identical to `solve_inverse_adps.py`).
- Empirically validated on real M4Raw: `corr(|adjoint|, RSS) = 0.980` (min 0.972)
  across 36 slices → convention correct.

## M4Raw facts (confirmed by inspector)
- 0.3 T brain, k-space `[slice=18, coil=4, PE=256, FE=256]`, order [S,C,PE,FE].
- Files `<study>_T2<rep>.h5`; train/val have 3 T2 reps/subject, test has 6.
- Off-center phase correction already applied in released k-space (don't reapply).

## Preprocessing threshold review (runbook §9.4 permits reviewing)
- Raw NCC between two aligned 0.3 T reps ≈ 0.95 (right at threshold) and naive
  brain-wide residual-phase-std ≈ 0.67 rad — both **SNR artifacts**, not motion
  (translation is 0.1–0.22 px). Fixes:
  - NCC computed on lightly Gaussian-smoothed RSS (structural) → ≈0.99 for aligned.
  - Residual phase std measured only over **high-SNR** brain voxels → ≈0.47–0.53.
  - Result: ~100% central-slice acceptance (was 2.8%).
- Noise: estimate repetition-difference MAD in the **image background** (air),
  not k-space corners (M4Raw uses circular k-space coverage → exact-zero corners).
- 4-view preprocessing (`num_views=4`) accepts **188/300 slices (62.7%)**, 20/25
  subjects — requiring 4 mutually motion-consistent repetitions is stricter than 2.

## Guidance scale (Stage-1 tuning)
- The normalized multi-view fidelity divides by σ²·N; with normalized σ≈0.037 and
  N≈C·nnz, the effective `l_ss` is ~10× the runbook's nominal [0.1,0.3,1,3] grid.
- Empirical single_r4 sweep (1 val slice, 100 steps): l_ss=10 → heldoutK 0.48
  SSIM 0.30; l_ss=30 → heldoutK 0.47 SSIM 0.33; l_ss=3 underfits (SSIM 0.11).
- Decision: tune `l_ss` over **[1,3,10,30]** (log-spaced, brackets the optimum);
  document the shift. Validation reference is a *noisy 3rd repetition* (weak;
  full-sampled adjoint only reaches SSIM≈0.40), so **held-out k-space error is the
  primary tuning criterion**, SSIM secondary — exactly as the runbook specifies.

## Tuning outcomes (validation, 6 subjects, single_r4)
- Inference grid selected **DPS, l_ss=30, num_steps=100** (heldoutK 0.477, SSIM 0.330).
  DPS beat ALD at every l_ss; l_ss=3 underfits (heldoutK 0.83).
- L1-wavelet SENSE lambda grid recentred to [1e-3..3e-1]; selected **lambda=0.03**
  (val heldoutK 0.534). SigPy's regularizer is scaled ~30x above the runbook nominal.
- **fp16 inference** matches fp32 quality (heldoutK 0.473 vs 0.483) at 3.67 GB vs
  ~10.5 GB (fp32 OOM'd on 11 GB). Guidance FFTs stay fp64 for accuracy.

---

# 2. Bugs that changed conclusions

Four defects materially altered results. Three were surfaced by user questions.

## 2.1 Brain-mask bug in metrics (fixed) — changed every number by ~+0.2 SSIM
Runbook 17.1 requires brain-masked metrics; the first pass scored the **full FOV**
including air, which is pure noise in the recon but averaged-down in the reference
=> structurally uncorrelated => SSIM deflated by ~0.2 (e.g. M4 0.402 -> 0.598).
`brain_mask_from_magnitude` was itself broken (`0.08*p99` labelled **97% of the FOV**
as brain at 0.3 T); switched to **Otsu** (brain fraction ~0.25). Metrics are now
recomputed from the stored reconstructions with a mask derived from the
*reference* (identical mask for every method on a slice); full-FOV variants are
retained as `*_fullfov` for transparency. Masking also flipped M0 vs M1 — the
L1-wavelet baseline had been rewarded for smoothing the noisy background.
*Surfaced by: user asking why SSIM was 0.4 when the montage looked good.*

## 2.2 Held-out k-space error was biased (fixed) — inverted the method ranking
The first implementation scored **all** coefficients against view-0's *noisy* k-space,
including the ones the method fitted. That rewards reproducing view-0's particular
noise realization -> biased toward single-view methods and *penalized* multi-view
methods that average noise away. Symptom: M7 (2x budget) scored *worse* than M2
despite winning SSIM by 0.037. Fixed to score only coefficients outside the union of
the condition's masks; the corrected metric now ranks **consistently with SSIM**.
Old values retained as `heldout_kspace_err_allcoef`.
Caveat: at 0.3 T the held-out (outer) k-space is largely noise, so the corrected
values compress into 0.84-0.98 and discriminate weakly; SSIM/NRMSE vs the
multi-repetition average carry more signal at low field.
*Surfaced by: the M7-vs-M2 contradiction, which the user's provenance question prompted.*

## 2.3 EMA half-life pitfall (fixed) — would have faked a null result
First fine-tune run used the default **EMA half-life = 0.5 Mimg (500 kimg)** while
training only 0.02 Mimg (20 kimg). The saved/evaluated model is the EMA, which
therefore stayed pinned at the pretrained weights: validation cross_view was
**identical at kimg 5 and 10 (27.8105 vs 27.8109)**. This would have made M5==M4 as an
artifact, not a real null result. Also lr=1e-5 left the raw training loss flat at
this (σ²·N-normalized) loss scale.
**Fix**: `--ema 0.002` (2 kimg half-life, so the EMA tracks the fine-tuned weights)
and `--lr 1e-4` -> 27.32→17.23, frac_improved=1.00. First run archived under
`runs/.../checkpoints/crossview_t2_emaartifact`.
Lesson: for short fine-tunes, EMA half-life must be << training duration or the
snapshot never reflects the fine-tuning.

## 2.4 Extra-repetition conditions duplicated one mask — the study's main finding
See §4. *Surfaced by: user asking why four scans would re-measure the same lines.*

## Other fixes (non-conclusion-changing)
- fp32 inference + fragmentation across many recons OOM'd 11 GB GPUs -> fp16
  denoiser + `torch.cuda.empty_cache()` between recons.
- **Checkpoint resolution**: SongUNet layer names embed the spatial resolution
  (`dec.96x96_...`). The published model was built at img_resolution=384; M4Raw is
  256 px. The net runs fine at 256 (fully-conv), but the fine-tuned checkpoint MUST
  be *reconstructed* at 384 to match its state_dict keys. `checkpoint_arch` now
  honors a stored `img_resolution_override=384` -> loads 638/638 (100%).
  Building at 256 gave 4/638 (would have broken M5/M6).
- `merge_fixed` operator needed a 4-D `[V=1,1,H,W]` mask (not `[1,H,W]`).
- `natural_merge` sliced `ksp[:V]` for 4-view data.
- Uncertainty manifest had to live in the masks root, or mask paths resolved
  relative to `configs/mvp/` and 0 recons were produced.

---

# 3. Validity checks

## Multi-view sample provenance (verified)
The two views are genuinely two independent acquisitions, not a copy: M4Raw ships
each repetition as its own H5 (`<subj>_T201.h5`, `_T202.h5`, ...); preprocessing
loads them as separate volumes and view 1 is only *phase-aligned* to view 0.
Empirical check on test slices: `repetition_ids=[1,2]`, k-space rel. diff 0.65-0.70,
image corr 0.88-0.93 (same anatomy), and `diff_bg_std/(sigma*sqrt2) = 1.21`
(~1 as expected for independent noise; the 21% excess is the MAD estimator being
robust to non-Gaussian tails). The 4-view set likewise uses 4 *distinct* repetitions
(e.g. `repetition_ids=[4,3,5,6]`).
Consequently duplicating the ACS **does** buy a real sqrt(2) SNR gain in the k-space
centre — measured directly: merging two identical-mask R=4 views drops the effective
noise 0.0443 -> 0.0313 (= sigma/sqrt2). Duplication is not worthless; it is just
worth much less than coverage (§4).

## Why SSIM is ~0.6-0.8 here and >0.9 in the Ambient/FastMRI paper (NOT a defect)
Three measurements settle it (6 test slices, proper Otsu brain mask):
- **Fully-sampled 1-rep recon vs an SNR-matched 1-rep reference: SSIM 0.991.**
  This is the paper's regime (reconstruct undersampled, compare to the
  *fully-sampled version of the same acquisition* -> only an undersampling gap).
  0.991 also proves the operator/FFT-convention/reconstruction path is exact.
- **Fully-sampled 1-rep recon vs the 6-repetition-average reference: SSIM 0.703.**
  This is the practical **ceiling** for our 2-view setup: the test reference averages
  6 repetitions, so anything reconstructed from 1-2 noisy 0.3 T repetitions carries
  an *irreducible SNR gap* on top of the undersampling gap.
- Averaging ladder toward the reference: 1 rep -> 0.72, 2 -> 0.84, 4 -> 0.96, 6 -> 1.0.
So our SSIM is not comparable to the paper's: different field strength (0.3 T vs
3 T), different reference (multi-rep average vs self fully-sampled), two gaps vs one.

## Reference bias (known, accepted)
The test reference averages all 6 repetitions **including the 2-4 used as input**
(~1/3 of the average), a mild *uniform* optimistic bias. All methods share it, so
comparisons are unaffected; absolute SSIM is slightly generous. Re-scoring against
4 held-out reps was offered and declined.

## Joint-vs-merged comparisons are confounded by a guidance-scale factor (verified)
The runbook's normalized fidelity averages over views, `D_joint = (1/V) sum_v
||M_v(FSx-y_v)||^2/(sigma^2 N)`, whereas the analytic merge uses `sigma_eff =
sigma/sqrt(V)`, giving `D_merged = V * D_joint`. **Measured ratio: 1.989 (V=2),
3.967 (V=4)** -- exactly V. Since the sampler applies `grad(D)/sqrt(D)`, the merged
condition receives **sqrt(V)x stronger guidance at the same l_ss**. Consequences:
- M3-vs-M4 and M7-vs-M8 ("joint == merge") were compared at a sqrt(2) guidance
  difference and *happened* to agree; at V=4 the 2x factor makes them diverge
  (M10 0.699 vs M11 0.656) despite carrying identical information.
- The equivalence unit tests remain correct: they compare the *raw* weighted-SSE
  gradients, which do match. It is the sampler-side normalization that differs.
- A clean joint-vs-merge test needs per-condition l_ss tuning (or dropping the 1/V).
Note the 1/V averaging is deliberate in the runbook (to keep l_ss interpretable as V
changes), but it *dilutes* the statistically-correct multi-view likelihood by V.

---

# 4. CORRECTION: extra-repetition conditions duplicated one mask (25% coverage)

The runbook's Condition D defines `joint_extra` as "both views use the **same** R=4
mask" (modelling plain NEX: a repetition re-runs the identical sequence). I extended
that to 4 views, so `joint_quad` spent **256 lines to see only 64 unique columns**
(25% coverage) — re-buying the same lines four times. Comparing *that* against a full
scan is a strawman. Separate scans should measure **different** lines.
Added complementary designs (shared ACS + disjoint outer partitions), same budget
(coverage verified directly from the stored mask tensors):

| condition | views | lines | unique cols | coverage |
|---|---|---|---|---|
| single_r4 | 1 | 64 | 64 | 25% |
| merge/joint_fixed | 2 | 64 | 48 | 19% |
| joint_extra (dup) | 2 | 128 | 64 | 25% |
| **joint_extra_comp** | 2 | 128 | **112** | **44%** |
| joint_quad (dup) | 4 | 256 | 64 | 25% |
| **joint_quad_comp** | 4 | 256 | **208** | **81%** |
| single_full | 1 | 256 | 256 | 100% |

The duplicated conditions are **retained**, not replaced: they are the honest
ablation for what the runbook specified, and they isolate the pure SNR benefit of
averaging (§3). The new rng draws come *after* the existing ones in
`build_conditions`, so all previously generated conditions are bit-identical and
earlier results remain valid.

## 4.1 2xR=4: duplicated vs complementary (25 subjects, same 128-line budget)
| method | SSIM | coverage |
|---|---|---|
| M2 single-view R=4 (64 lines) | 0.618 | 25% |
| M7 merged 2xR=4 duplicated | 0.654 | 25% |
| M8 joint 2xR=4 duplicated | 0.659 | 25% |
| M6 xview-FT joint 2xR=4 duplicated | 0.663 | 25% |
| **M15 merged 2xR=4 COMPLEMENTARY** | **0.706** | 44% |
| **M14 joint 2xR=4 COMPLEMENTARY** | **0.704** | 44% |
| **M13 xview-FT joint 2xR=4 COMPLEMENTARY** | **0.704** | 44% |

Subject-paired: **M13-M6 +0.041 (100%)**, **M14-M8 +0.045 (100%)**,
**M15-M7 +0.052 (100%)**, M13-M2 +0.087 (100%), **M13-M14 = -0.0001 (52%)**.
This supersedes the main table's "best fixed-prior method" (M6 0.663 -> 0.706 at the
same budget).

## 4.2 4xR=4 vs ONE FULL measurement (20 subjects) — "same budget, can we win?"
Extra experiment (beyond the runbook), on the 4-view set (59 slices):

| method | SSIM [95% CI] | lines | coverage |
|---|---|---|---|
| MF plain recon, 1 FULL k-space | **0.796 [0.783,0.812]** | 256 | 100% |
| **M18 merged 4xR=4 COMPLEMENTARY** | 0.766 [0.751,0.782] | 256 | 81% |
| **M17 joint 4xR=4 COMPLEMENTARY** | 0.765 [0.751,0.781] | 256 | 81% |
| **M16 xview-FT joint 4xR=4 COMPLEMENTARY** | 0.764 [0.749,0.781] | 256 | 81% |
| M12 diffusion, 1 FULL k-space | 0.758 [0.744,0.773] | 256 | 100% |
| M9 joint 4xR=4 + xview-FT (duplicated) | 0.703 [0.685,0.724] | 256 | 25% |
| M10 joint 4xR=4, original prior (duplicated) | 0.699 [0.679,0.721] | 256 | 25% |
| M11 merged 4xR=4 (duplicated) | 0.656 [0.636,0.677] | 256 | 25% |
| M2 single-view R=4 | 0.643 [0.622,0.667] | 64 | 25% |

**Answer: yes against the same reconstruction method, no against the best one.**
Subject-paired: **M16-M12 = +0.007 SSIM (M16 wins 65%) and -0.014 NRMSE (95%)** ->
against the *same* reconstruction method, four cheap complementary scans **beat** one
full scan on both metrics.
But **M12-MF = -0.039 SSIM (MF wins 90%)**: the diffusion prior **hurts** on
fully-sampled data — when the data determines the image, a generative prior +
guidance only adds error.

The M16-vs-MF metric split is the informative one:
- SSIM: **-0.032** [-0.045, -0.017], M16 wins only 10%.
- NRMSE: **+0.0006** [-0.0041, +0.0045] — a **statistical tie** (0.1205 vs 0.1198).

So M16 is as accurate as a full scan in the L2 sense and loses only on *structural*
similarity. That is the signature of a prior painting plausible-but-wrong texture,
not of missing information — i.e. **M16-MF is a reconstruction deficit, not an
acquisition one**. `l_ss=30` was tuned on single_r4 and never re-tuned for V=4 or
R=1; per-condition tuning is the prime suspect for closing that gap.
(An earlier draft of this file mis-attributed the NRMSE tie to M12; M12's NRMSE is
0.135, clearly *worse* than M16's 0.120. The tie is with MF.)

The acquisition fix is what moved this: **M16-M9 = +0.061 on 100% of subjects** —
the largest single effect measured anywhere in the study. With the duplicated design
the answer was *no* by 0.093; complementary masks close two thirds of that.
Also: at 81% coverage **all method variants converge** (M16~M17~M18 within 0.002).

---

# 5. HEADLINE: acquisition design dominates reconstruction method

| what you change | SSIM gain |
|---|---|
| **acquisition design** (complementary vs duplicated masks, same budget) | **+0.041 .. +0.061** |
| extra cheap repetition at same coverage (pure sqrt(2) SNR) | +0.037 |
| learned prior vs classical baselines (sparse regime only) | +0.10 |
| prior / cross-view fine-tuning / joint-vs-merged | **-0.001 .. +0.010** |

Acquisition design is worth ~5-10x more than every reconstruction-method choice in
this study combined. The cross-view fine-tuning (the project's nominal contribution)
only pays off in the **sparse** regime, and decays monotonically to zero as coverage
grows:

| coverage | comparison | SSIM delta | wins |
|---|---|---|---|
| 19% | M5-M4 | +0.010 | 88% |
| 25% | M6-M8 | +0.003 | 60% |
| 44% | M13-M14 | -0.000 | 52% |
| 81% | M16-M17 | -0.001 | 40% |

Once coverage is adequate the data dominates and prior/combiner choices wash out.
Credit: this was surfaced by the user questioning why four scans would re-measure
the same lines.

## Negative findings retained (runbook §9.4)
- Joint likelihood == analytic merging for aligned identical-mask views (theory
  predicted it; we confirm it, and do not claim the summation as a contribution).
- Cross-view fine-tuning is worth ~0 above 40% coverage.
- The diffusion prior *hurts* on well-sampled data (M12 < MF).
- At a **fixed** budget, multi-view *loses* to single-view (M2-M5 = +0.013, 68%):
  the duplicated ACS buys 48 unique columns instead of 64. Multi-view only helps
  when it buys extra scans, not when it subdivides one.
- Cross-view fine-tuning is a short (0.02 Mimg) empirical self-supervised adaptation;
  it does not inherit the Ambient Diffusion identifiability theorem.

## Duplicated-ACS in the fixed-budget design (known limitation)
Conditions B/C follow runbook 10 ("counting duplicated ACS coefficients twice"),
i.e. each of the two R=8 views re-acquires the 16-line ACS, mirroring *realistic
repeated scanning*. Cost: at the same 64-line budget the two views cover only **48
unique columns** vs single-view R=4's **64** — which is why M2 beats M3/M4/M5.
An **asymmetric** design (view A = ACS + outer, view B = outer only) would give 64
unique columns at the same cost, trading √2 SNR in the k-space centre for 2x more
unique outer coverage. Largely superseded by the complementary designs (§4), which
apply the same principle at the extra-repetition budgets.

---

# 6. Open threads (not run)
1. **Re-tune `l_ss` per condition.** It was tuned once at V=1/R=4 and reused
   everywhere. This is the only thing keeping M16 below MF, and it also confounds
   every joint-vs-merged comparison (§3). Highest-value next experiment.
2. **Drop the 1/V normalization** in the multi-view fidelity — it verifiably dilutes
   the statistically-correct likelihood by a factor V.
3. Ultra-low-field (<0.1 T) validation. Nothing here is claimed below 0.3 T.
