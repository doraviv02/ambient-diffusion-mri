# Cross-View Ambient Diffusion for Multi-Acquisition Low-Field MRI (MVP)

A seminar MVP built on the [Ambient Diffusion Posterior Sampling for MRI](https://github.com/utcsilab/ambient-diffusion-mri)
repository. It studies whether a learned diffusion prior plus **multiple cheap
acquisitions** can improve low-field reconstruction on **M4Raw 0.3 T** T2-weighted
brain data, and — more importantly — asks how much the **acquisition design**
(which k-space lines the repeated scans measure) matters versus the reconstruction
method.

**The headline finding:** acquisition design dominates reconstruction method by
5–10×, and the honest low-field ceiling is classical multi-average (NEX), not the
learned prior. See [`reports/mvp_summary.md`](reports/mvp_summary.md) for the full
write-up and [`reports/mvp_notes.md`](reports/mvp_notes.md) for the decision log
(including four bugs that changed conclusions).

> This README is the MVP submission guide. The original upstream README is kept as
> [`README_upstream.md`](README_upstream.md).

---

## Main results (start here)

| Deliverable | File |
|---|---|
| **Results tables** (SSIM / NRMSE / held-out k-error, per method, both sets) | [`reports/mvp_summary.md`](reports/mvp_summary.md) |
| Results tables, standalone (Markdown + CSV) | `tables/mvp/results_tables.md`, `tables/mvp/results_*.csv` |
| **Reconstruction montages** — the reconstructed images per results table | `figures/mvp/recon_montage_2view.png`, `figures/mvp/recon_montage_4view.png` |
| Results tables visualised (bars ± 95% CI) | `figures/mvp/results_2view.png`, `figures/mvp/results_4view.png` |
| Cross-experiment conclusions | `figures/mvp/experiment_overview.png`, `effect_sizes_forest.png`, `budget_ladder.png`, `finetuning_vs_coverage.png` |
| Mask designs / l_ss tuning | `figures/mvp/mask_design_all.png`, `lss_tier_tuning.png` |

If you only want to regenerate the tables and figures from the reconstructions that
are already on disk, jump to [Regenerate tables & figures only](#regenerate-tables--figures-only).

---

## The experiment grid

All methods are defined in one place — [`analysis/experiments.py`](analysis/experiments.py).
Each is a combination of **acquisition design** × **reconstruction method**, evaluated
on two sets: the **2-view set** (25 subjects, 75 slices) and the **4-view set**
(20 subjects, 59 slices — only slices with 4 motion-consistent M4Raw test repetitions).

| slug family | acquisition | combiner variants |
|---|---|---|
| `classical_{adjoint,l1wav}` | fixed budget (2×R=8) | classical (no prior) |
| `single_r4` | one cheap R=4 scan | diffusion |
| `fixed_split_{merge,ft}` | fixed budget (2×R=8, split one scan) | analytic merge / cross-view FT |
| `dup2_{merge,ft}`, `dup4_{merge,ft}` | 2× / 4× **duplicated** mask (25% cov) | merge / FT |
| `comp2_{merge,ft}`, `comp4_{merge,ft}` | 2× / 4× **complementary**, shared ACS (44% / 81% cov) | merge / FT |
| `fcomp2_{merge,ft}`, `fcomp4_{merge,ft}` | 2× / 4× **fully-complementary**, split ACS (50% / 100% cov) | merge / FT |
| `full_{plain,diffusion}` | one complete measurement (R=1) | plain / diffusion |
| `dualfull_{merge,ft}` | **two** complete measurements (NEX=2) | classical average / cross-view FT |

`merge` = analytic noise-weighted k-space merge then diffusion; `ft` = cross-view
fine-tuned prior with the joint multi-view likelihood; `plain` = direct SENSE recon.

---

## Setup

```bash
# 1. environment (Miniconda). The env used here is `ambient-mv`:
conda env create -f environment.yml            # or: conda create -n ambient-mv python=3.9 && pip install ...
conda activate ambient-mv

# 2. paths — edit and source .env.mvp (defines PROJECT_ROOT, DATA_ROOT, RUN_ROOT,
#    MODEL_ROOT, AMBIENT_R4_DIR, MVP_GPUS). Every script below reads these.
source ../.env.mvp        # adjust the path to wherever you keep .env.mvp
```

**Prerequisites**
- **Data:** M4Raw (`train` / `val` / `test` multicoil H5) under `$DATA_ROOT/raw/`.
- **Prior checkpoint:** the published Ambient R=4 model at `$AMBIENT_R4_DIR`
  (`network-snapshot.pkl` + `training_options.json`).
- **GPUs:** two are enough (`MVP_GPUS=2,3`); 11 GB each. Inference runs fp16.

---

## Full reproduction pipeline

Every stage writes into `$DATA_ROOT/processed`, `$RUN_ROOT`, `tables/mvp`, and
`figures/mvp`. Stages are ordered; later stages consume earlier outputs.

### 1. Preprocess M4Raw into multi-view samples

Screens repetitions for motion, estimates per-view noise, computes shared ESPIRiT
maps, and stores a multi-rep-average reference. Run once per view count.

```bash
# 2-view set (num_views=2, from configs/mvp/data_m4raw_t2.yaml)
python tools/prepare_m4raw_multiview.py --config configs/mvp/data_m4raw_t2.yaml \
  --train-root "$DATA_ROOT/raw/train/multicoil_train" \
  --val-root   "$DATA_ROOT/raw/val/multicoil_val" \
  --test-root  "$DATA_ROOT/raw/test/multicoil_test" \
  --output-root "$DATA_ROOT/processed/mvp_t2"

# 4-view set (num_views=4; test only has enough repetitions — M4Raw test = 6 reps)
python tools/prepare_m4raw_multiview.py --config configs/mvp/data_m4raw_t2_quad.yaml \
  --test-root "$DATA_ROOT/raw/test/multicoil_test" \
  --output-root "$DATA_ROOT/processed/mvp_t2_quad"
```

### 2. Generate the experiment masks

Deterministic per-subject/slice masks for every acquisition condition (adds no RNG
draws for later conditions, so re-running never perturbs earlier ones).

```bash
python tools/generate_mvp_masks.py --config configs/mvp/masks.yaml \
  --data-root "$DATA_ROOT/processed/mvp_t2"      --output-root "$DATA_ROOT/processed/mvp_t2_masks"
python tools/generate_mvp_masks.py --config configs/mvp/masks.yaml \
  --data-root "$DATA_ROOT/processed/mvp_t2_quad" --output-root "$DATA_ROOT/processed/mvp_t2_quad_masks"

# evaluation manifests (the central-slice subset the paper scores)
python tools/make_eval_subset.py --masks-root "$DATA_ROOT/processed/mvp_t2_masks"      --out test_eval_manifest.csv
python tools/make_eval_subset.py --masks-root "$DATA_ROOT/processed/mvp_t2_quad_masks" --out quad_eval_manifest.csv
```

### 3. (Optional) Cross-view fine-tuning of the prior

Trains the `*_ft` prior. Skippable — the published R=4 checkpoint is the base, and
the fine-tuning benefit is small (see the report). Uses `train.py --precond ambient_mv`.

```bash
bash cluster/run_finetune.sh          # 2-GPU torchrun; ~0.02 Mimg, EMA half-life 0.002
# best checkpoint -> $RUN_ROOT/checkpoints/crossview_t2/best (via tools/select_best_checkpoint.py)
```

### 4. Inference (produces the reconstruction `.pt` files)

```bash
# 4a. 2-view diffusion methods (sharded over MVP_GPUS) -> $RUN_ROOT/results/final
bash cluster/run_final_diffusion.sh

# 4b. 4-view diffusion methods -> $RUN_ROOT/results/final_quad
CONFIG=configs/mvp/selected_inference.yaml
QMAN="$DATA_ROOT/processed/mvp_t2_quad_masks/quad_eval_manifest.csv"
python solve_inverse_mv_adps.py --config $CONFIG --manifest "$QMAN" \
  --methods single_r4,dup4_merge,dup4_ft,comp4_merge,comp4_ft,fcomp4_merge,fcomp4_ft,full_diffusion,dualfull_ft \
  --output_dir "$RUN_ROOT/results/final_quad"

# 4c. classical baselines (no network): 2-view (adjoint, L1-wavelet) and 4-view (full_plain, NEX=2 avg)
python analysis/run_classical_recon.py --config configs/mvp/selected_classical.yaml \
  --manifest "$DATA_ROOT/processed/mvp_t2_masks/test_eval_manifest.csv" \
  --methods classical_adjoint,classical_l1wav --output_dir "$RUN_ROOT/results/final"
python analysis/run_classical_recon.py --manifest "$QMAN" \
  --methods full_plain,dualfull_merge --output_dir "$RUN_ROOT/results/final_quad"
```

Hyperparameters (`l_ss=30`, `num_steps=100`, DPS; L1-wavelet `lambda=0.03`) were
tuned on **validation** only — see `tools/tune_inference.py`, `tools/tune_classical.py`,
and `tools/tune_lss_by_tier.py` (the per-tier l_ss retune, a documented negative result).

### 5. Metrics, tables, figures, report

One command aggregates the reconstructions and produces everything:

```bash
bash cluster/run_final_analysis.sh
```

This runs: `compute_metrics.py` → `aggregate_subject_metrics.py` (subject-level
means + 95 % bootstrap CIs, both sets) → `make_results_tables.py` (standalone
tables) → the canonical figures → `build_mvp_report.py` (`reports/mvp_summary.md`).

---

## Regenerate tables & figures only

If the reconstruction `.pt` files already exist under `$RUN_ROOT/results/{final,final_quad}`,
you do **not** need GPUs or re-inference:

```bash
source ../.env.mvp
bash cluster/run_final_analysis.sh          # metrics + tables + figures + report
```

**Just the results tables** (Markdown + CSV, no prose, no figures):

```bash
python analysis/make_results_tables.py \
  --summary tables/mvp/summary_metrics.csv \
  --summary-quad tables/mvp/summary_metrics_quad.csv \
  --output-dir tables/mvp
# -> tables/mvp/results_tables.md + results_2view_fixed.csv / results_2view_2x.csv / results_4view.csv
```

**Just the reconstruction montages** (the main visual result):

```bash
python analysis/plot_recon_montage_tables.py \
  --final "$RUN_ROOT/results/final" --final-quad "$RUN_ROOT/results/final_quad" \
  --output-dir figures/mvp --cases 3
```

**Just the results-table bar charts:**

```bash
python analysis/plot_results_tables.py \
  --summary tables/mvp/summary_metrics.csv \
  --summary-quad tables/mvp/summary_metrics_quad.csv --output-dir figures/mvp
```

---

## What is MVP vs upstream

**MVP additions (this work):**
- `utils/multiview_mri.py` — multi-view MRI operator + noise-weighted merge.
- `solve_inverse_mv_adps.py` — multi-view Ambient DPS sampler.
- `training/loss.py::CrossViewAmbientLoss`, `training/dataset.py::MultiViewKspaceDataset` — cross-view fine-tuning.
- `analysis/` — metrics, experiment registry, table + figure generation, the report builder.
- `tools/` — preprocessing, mask generation, tuning, manifests.
- `configs/mvp/`, `cluster/`, `reports/`, `tables/mvp/`, `figures/mvp/`.

**Unchanged upstream** (the diffusion architecture and original single-view sampler
are byte-for-byte intact): `training/networks.py`, `torch_utils/`, `generate.py`,
`solve_inverse_adps.py`. The `ambient-diffusion-inverse/` git submodule is the
original reference code.

---

## Repository map

```
analysis/         metrics, experiments.py (method registry), plotting, report + tables builders
tools/            preprocessing, mask generation, tuning, manifests, name migration
configs/mvp/      data / mask / inference / training / classical configs
cluster/          run_finetune.sh, run_final_diffusion.sh, run_final_analysis.sh
utils/            multiview_mri.py, mri_fft.py, checkpoint_arch.py, train_masks.py
reports/          mvp_summary.md (results), mvp_notes.md (decision log)
tables/mvp/       metric CSVs + results tables (md/csv)
figures/mvp/      the canonical figures (montages, results bars, conclusions)
solve_inverse_mv_adps.py    multi-view inference entry point
```
