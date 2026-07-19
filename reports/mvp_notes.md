# MVP configuration record

Factual record of the setup behind `reports/mvp_summary.md` (environment, dataset,
preprocessing, mask conditions, tuning outcomes). Results tables live in the summary.

---

## Method naming

Experiments use descriptive slugs. The five redundant "joint likelihood, original
prior" methods were removed (for aligned identical-mask views the joint likelihood
equals the analytic noise-weighted merge); each family keeps one `*_merge`
representative plus the cross-view fine-tuned `*_ft` method. Removed result files are
backed up under `runs/.../results/_deleted_joint_original/`.

| slug | old id | | slug | old id |
|---|---|---|---|---|
| `classical_adjoint` | M0 | | `dup2_ft` | M6 |
| `classical_l1wav` | M1 | | `comp2_merge` | M15 |
| `single_r4` | M2 | | `comp2_ft` | M13 |
| `fixed_split_merge` | M3 | | `dup4_merge` | M11 |
| `fixed_split_ft` | M5 | | `dup4_ft` | M9 |
| `dup2_merge` | M7 | | `comp4_merge` | M18 |
| `full_diffusion` | M12 | | `comp4_ft` | M16 |
| `full_plain` | MF | | **removed** | M4, M8, M10, M14, M17 |

`fcomp*` (fully-complementary, split ACS) and `dualfull_*` (two full acquisitions)
were added later and have no legacy id.

---

## Environment

- Cluster: 4× RTX 2080 Ti (11 GB each); GPUs 2,3 used.
- Miniconda env `ambient-mv`: torch 2.8.0+cu128, python 3.9.25, numpy 2.0.2.
- The published R=4 checkpoint strict-loads **638/638 tensors (100%)**.
- Inference runs fp16 (3.67 GB vs ~10.5 GB fp32, which OOM'd); guidance FFTs stay fp64.
- SongUNet layer names embed the spatial resolution, so the fine-tuned checkpoint is
  rebuilt at `img_resolution_override=384` to match its state_dict keys.

## Dataset (M4Raw)

- 0.3 T brain, k-space `[slice=18, coil=4, PE=256, FE=256]`, order [S,C,PE,FE].
- Files `<study>_T2<rep>.h5`. **train/val have 3 T2 repetitions per subject; test has 6.**
  The 4-view experiments therefore run on the test split (4 of its 6 reps).
- Off-center phase correction is already applied in the released k-space.

## FFT / tensor convention

- Plain-ortho FFT on `fftmod`-ed maps + k-space; image is 2-channel real (real=ch0, imag=ch1).
- Preprocessing stores **natural** (un-fftmod) centered k-space + maps; inference and
  loss apply `fftmod` at load.
- Validated on real M4Raw: `corr(|adjoint|, RSS) = 0.980` (min 0.972) across 36 slices.

## Preprocessing

- Motion screening: structural NCC on lightly Gaussian-smoothed RSS (≈0.99 for aligned
  reps) and residual phase std over **high-SNR** brain voxels (≈0.47–0.53).
  Result: ~100% central-slice acceptance for 2 views.
- Noise: repetition-difference MAD in the **image background** (M4Raw k-space corners
  are exactly zero due to circular coverage).
- 4-view preprocessing (`num_views=4`) accepts **188/300 slices (62.7%)**, 20/25 subjects.
- Reference (test): average of all 6 released RSS reconstructions per slice.
- Metrics are brain-masked with **Otsu** on the reference (brain fraction ~0.25);
  full-FOV variants retained as `*_fullfov`.
- Held-out k-space error scores only coefficients outside the union of the condition's
  masks; the all-coefficient variant is retained as `heldout_kspace_err_allcoef`.

---

## Tuning outcomes (validation only; test never used for tuning)

**Inference grid** (6 subjects, `single_r4`) — selected **DPS, l_ss=30, num_steps=100**
(held-out k-space error 0.477, SSIM 0.330). DPS beat ALD at every l_ss.

**L1-wavelet SENSE** — lambda grid [1e-3 .. 3e-1]; selected **lambda=0.03**
(val held-out k-space error 0.534).

**Cross-view fine-tune** — `--ema 0.002` (half-life must be << the 0.02 Mimg training
duration or the EMA snapshot never reflects the fine-tuning) and `--lr 1e-4`;
validation cross-view loss 27.32 → 17.23, frac_improved 1.00.

**Per-tier l_ss retune** (6 subjects; dense tier via an 80%-holdout proxy, since
held-out error is undefined at 100% sampling). Validation SSIM:

| tier | l_ss=3 | 10 | 30 | 100 | 300 | selected |
|---|---|---|---|---|---|---|
| sparse (≤25%) | 0.153 | 0.545 | **0.582** | 0.556 | — | **30** |
| mid (44%) | 0.155 | 0.610 | **0.637** | 0.603 | — | **30** |
| dense (≥80%) | 0.118 | 0.441 | **0.642** | 0.632 | 0.513 | **30** |

`l_ss=30` is optimal at every tier, so no test result changes. The solver supports
per-tier l_ss (`cfg["l_ss_by_tier"]`, `configs/mvp/l_ss_by_tier.yaml`); it resolves to 30.
Grid: `tables/mvp/lss_tier_grid.csv`; figure: `figures/mvp/lss_tier_tuning.png`.

---

## Measured quantities used to interpret the tables

- Merging two identical-mask R=4 views drops the effective noise **0.0443 → 0.0313**
  (= σ/√2), confirming the views are independent acquisitions.
- Multi-view provenance verified: `repetition_ids=[1,2]` (2-view) and `[4,3,5,6]`
  (4-view); k-space rel. diff 0.65–0.70, image corr 0.88–0.93,
  `diff_bg_std/(σ√2) = 1.21`.
- Guidance-scale factor: `D_merged = V · D_joint` (measured 1.989 at V=2, 3.967 at V=4),
  so the merged condition receives √V× stronger guidance at the same `l_ss`.
- Averaging ladder toward the 6-rep reference: 1 rep → 0.72, 2 → 0.84, 4 → 0.96, 6 → 1.0.
  A fully-sampled 1-rep recon scores **0.991** against an SNR-matched 1-rep reference
  and **0.703** against the 6-rep average.
