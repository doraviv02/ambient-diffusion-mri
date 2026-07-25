# Cross-View Ambient Diffusion for Multi-Acquisition Low-Field MRI

## Dor Aviv, Ori Hagler

Built on the [Ambient Diffusion Posterior Sampling for MRI](https://github.com/utcsilab/ambient-diffusion-mri)
repository. It studies whether a learned diffusion prior plus **multiple cheap
acquisitions** can improve low-field reconstruction on **M4Raw 0.3 T** T2-weighted
brain data, and - more importantly - asks how much the **acquisition design**
(which k-space lines the repeated scans measure) matters versus the reconstruction
method.


> This README is the project submission guide. The original upstream README is kept as
> [`README_upstream.md`](README_upstream.md).

---

## Main results

| Output | File |
|---|---|
| **Results tables** (SSIM / NRMSE / held-out k-error, per method, both sets) + dataset & scope | [`reports/project_summary.md`](reports/project_summary.md) |
| Results tables, standalone (Markdown + CSV) | `tables/project/results_tables.md`, `tables/project/results_*.csv` |
| **Reconstruction montages** - the reconstructed images per results table | `figures/project/recon_montage_2view.png`, `figures/project/recon_montage_4view.png` |
| Results tables visualised (bars ± 95% CI) | `figures/project/results_2view.png`, `figures/project/results_4view.png` |
| Cross-experiment conclusions | `figures/project/effect_sizes_forest.png`, `budget_ladder.png`, `finetuning_vs_coverage.png` |
| Mask designs / l_ss tuning | `figures/project/mask_design_all.png`, `lss_tier_tuning.png` |

If you only want to regenerate the tables and figures from the reconstructions that
are already on disk, jump to [Regenerate tables & figures only](#regenerate-tables--figures-only).

---

## The experiment grid

All methods are defined in one place - [`analysis/experiments.py`](analysis/experiments.py).
Each is a combination of **acquisition design** × **reconstruction method**, evaluated
on two sets: the **2-view set** (25 subjects, 75 slices) and the **4-view set**
(20 subjects, 59 slices - only slices with 4 motion-consistent M4Raw test repetitions).

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
fine-tuned prior with the joint multi-view likelihood; `plain` = direct
coil-combined (adjoint SENSE) recon, no prior and no iterations.

---

## Data & inputs

**Dataset - M4Raw** (0.3 T, 4-coil, 256×256, multi-repetition brain k-space; this work
uses the T2-weighted subset). Download V1.6 from Zenodo:
[doi.org/10.5281/zenodo.8056074](https://doi.org/10.5281/zenodo.8056074) - paper:
Lyu, Mei, Huang et al., *M4Raw*, Sci Data 10, 264 (2023). Unpack so that:

```
$DATA_ROOT/raw/train/multicoil_train/*.h5
$DATA_ROOT/raw/val/multicoil_val/*.h5
$DATA_ROOT/raw/test/multicoil_test/*.h5     # 6 repetitions per contrast
```

**Prior checkpoint - Ambient R=4**, published by the upstream authors:
[utexas.box.com/s/axofnwib9kukdpa92ge4ays87dmuvpf7](https://utexas.box.com/s/axofnwib9kukdpa92ge4ays87dmuvpf7)
(`wget -v -O ambient_models.zip -L https://utexas.box.com/shared/static/axofnwib9kukdpa92ge4ays87dmuvpf7.zip`).
Point `$AMBIENT_R4_DIR` at the folder holding `network-snapshot.pkl` + `training_options.json`.

**File types read/written**

| Type | Role |
|---|---|
| `.h5` | raw M4Raw multicoil k-space - the only external input |
| `.pt` | per-slice multi-view samples, masks, and reconstructions (all intermediates) |
| `.pkl` / `.json` | prior checkpoint weights + its `training_options.json` |
| `.yaml` | configs under `configs/project/` (data, masks, inference, training) |
| `.csv` | dataset + eval manifests (next to the masks under `$DATA_ROOT/processed/`) and metric tables (`tables/project/`) |
| `.png` / `.md` | figures and the final report |

---

## Setup

```bash
# 1. environment (Miniconda). `environment.yml` is the *upstream* file: it creates an
#    env named `ambient2.0`, pins torch 2.0.0, and does not install h5py, pyyaml,
#    pandas, matplotlib, scikit-image or sigpy, all of which this project imports.
#    The env used here is `ambient-mv`:
conda create -n ambient-mv python=3.9 && conda activate ambient-mv
pip install torch torchvision numpy scipy h5py pyyaml pandas matplotlib \
            scikit-image sigpy pillow click psutil tqdm pyspng wandb

# 2. paths - copy the template, edit it, source it. It defines PROJECT_ROOT,
#    DATA_ROOT, RUN_ROOT, MODEL_ROOT, AMBIENT_R4_DIR, PROJECT_GPUS, CONDA_ROOT.
#    Every Python entry point below takes its paths from these.
cp .env.project.example ../.env.project
$EDITOR ../.env.project
source ../.env.project        # adjust the path to wherever you keep .env.project
```

Exact versions behind the reported results: python 3.9.25, torch 2.8.0+cu128,
numpy 2.0.2, scipy 1.13.1, h5py 3.14.0, pyyaml 6.0.3, pandas 2.3.3,
matplotlib 3.9.4, scikit-image 0.24.0, sigpy 0.1.27.

**Not covered by the env file** - edit these three spots for a different machine:
the two `source` lines at the top of each `cluster/*.sh` (conda root and the
location of `.env.project`), the `checkpoints:` block in
`configs/project/inference.yaml` and `configs/project/selected_inference.yaml`
(absolute checkpoint dirs), and the `CUDA_VISIBLE_DEVICES=2` / `=3` literals in
`cluster/run_final_diffusion.sh` and `cluster/run_finetune.sh`. `PROJECT_GPUS` is a
record of which GPUs were used, not a variable the scripts read.

**Also needed**
- **GPUs:** we used two RTX 2080 Ti for fine-tuning. Inference runs fp16
  (~3.7 GB per process, ~14 s per 100-step reconstruction on an RTX 2080 Ti).

---

## Full reproduction pipeline

Every stage writes into `$DATA_ROOT/processed`, `$RUN_ROOT`, `tables/project`, and
`figures/project`. Stages are ordered; later stages consume earlier outputs.

### 1. Preprocess M4Raw into multi-view samples

Screens repetitions for motion, estimates per-view noise, computes shared ESPIRiT
maps, and stores a multi-rep-average reference. Run once per view count.

```bash
# 2-view set (num_views=2, from configs/project/data_m4raw_t2.yaml)
python tools/prepare_m4raw_multiview.py --config configs/project/data_m4raw_t2.yaml \
  --train-root "$DATA_ROOT/raw/train/multicoil_train" \
  --val-root   "$DATA_ROOT/raw/val/multicoil_val" \
  --test-root  "$DATA_ROOT/raw/test/multicoil_test" \
  --output-root "$DATA_ROOT/processed/project_t2"

# 4-view set (num_views=4; test only has enough repetitions - M4Raw test = 6 reps)
python tools/prepare_m4raw_multiview.py --config configs/project/data_m4raw_t2_quad.yaml \
  --test-root "$DATA_ROOT/raw/test/multicoil_test" \
  --output-root "$DATA_ROOT/processed/project_t2_quad"
```

Note: the 2-view `processed/project_t2` currently on disk was written
before the brain mask switched from the legacy relative-threshold rule to Otsu, so
re-running this stage rewrites it with slightly different per-slice normalisation and
noise statistics (the 4-view set already matches the current code). A check showing the difference reproduces SSIM within 0.015.

### 2. Generate the experiment masks

Deterministic per-subject/slice masks for every acquisition condition (adds no RNG
draws for later conditions, so re-running never perturbs earlier ones).

```bash
python tools/generate_project_masks.py --config configs/project/masks.yaml \
  --data-root "$DATA_ROOT/processed/project_t2"      --output-root "$DATA_ROOT/processed/project_t2_masks"
python tools/generate_project_masks.py --config configs/project/masks.yaml \
  --data-root "$DATA_ROOT/processed/project_t2_quad" --output-root "$DATA_ROOT/processed/project_t2_quad_masks"

# evaluation manifests (the central-slice subset the paper scores: all subjects x 3
# central slices) -- written next to the masks, where stages 4a/4b read them from
python tools/make_eval_subset.py \
  --manifest "$DATA_ROOT/processed/project_t2_masks/test_manifest.csv" \
  --output   "$DATA_ROOT/processed/project_t2_masks/test_eval_manifest.csv"
python tools/make_eval_subset.py \
  --manifest "$DATA_ROOT/processed/project_t2_quad_masks/test_manifest.csv" \
  --output   "$DATA_ROOT/processed/project_t2_quad_masks/quad_eval_manifest.csv"

# uncertainty subset (10 subjects x 1 central slice) - cluster/run_final_diffusion.sh
# needs this file for its last stage (fixed_split_ft, seeds 0-3)
python tools/make_uncertainty_subset.py \
  --manifest "$DATA_ROOT/processed/project_t2_masks/test_manifest.csv" \
  --output   "$DATA_ROOT/processed/project_t2_masks/uncertainty_subset.csv"
```

### 3. Cross-view fine-tuning of the prior

Trains the `*_ft` prior (the published R=4 checkpoint is the base). Uses
`train.py --precond ambient_mv`. Skip it only if you drop every `*_ft` method:
`cluster/run_final_diffusion.sh` starts by selecting and load-checking this
checkpoint, and `configs/project/selected_inference.yaml` resolves `checkpoint:
finetuned` to it. The fine-tuning benefit itself is small (see the report).

```bash
bash cluster/run_finetune.sh          # 2-GPU torchrun; ~0.02 Mimg, EMA half-life 0.002
# best checkpoint -> $RUN_ROOT/checkpoints/crossview_t2/best (via tools/select_best_checkpoint.py)
```

### 4. Inference (produces the reconstruction `.pt` files)

```bash
# 4a. 2-view diffusion methods, sharded across two GPUs (hardcoded CUDA_VISIBLE_DEVICES
#     =2 / =3 inside the script). Also selects the fine-tuned checkpoint and runs the
#     uncertainty subset -> $RUN_ROOT/results/final and $RUN_ROOT/results/uncertainty
bash cluster/run_final_diffusion.sh

# 4b. 4-view diffusion methods -> $RUN_ROOT/results/final_quad
CONFIG=configs/project/selected_inference.yaml
QMAN="$DATA_ROOT/processed/project_t2_quad_masks/quad_eval_manifest.csv"
CUDA_VISIBLE_DEVICES=2 python solve_inverse_mv_adps.py --config $CONFIG --manifest "$QMAN" \
  --methods single_r4,dup4_merge,dup4_ft,comp4_merge,comp4_ft,fcomp4_merge,fcomp4_ft,full_diffusion,dualfull_ft \
  --output_dir "$RUN_ROOT/results/final_quad"

# 4c. classical baselines (no network): 2-view (adjoint, L1-wavelet) and 4-view (full_plain, NEX=2 avg)
CUDA_VISIBLE_DEVICES=2 python analysis/run_classical_recon.py --config configs/project/selected_classical.yaml \
  --manifest "$DATA_ROOT/processed/project_t2_masks/test_eval_manifest.csv" \
  --methods classical_adjoint,classical_l1wav --output_dir "$RUN_ROOT/results/final"
CUDA_VISIBLE_DEVICES=2 python analysis/run_classical_recon.py --manifest "$QMAN" \
  --methods full_plain,dualfull_merge --output_dir "$RUN_ROOT/results/final_quad"
```

Hyperparameters (`l_ss=30`, `num_steps=100`, DPS; L1-wavelet `lambda=0.03`) were
tuned on **validation** only - see `tools/tune_inference.py`, `tools/tune_classical.py`,
and `tools/tune_lss_by_tier.py` (the per-tier l_ss retune, a documented negative result).

### 5. Metrics, tables, figures, report

One command aggregates the reconstructions and produces everything:

```bash
bash cluster/run_final_analysis.sh
```

This runs: `compute_metrics.py` → `aggregate_subject_metrics.py` (subject-level
means + 95 % bootstrap CIs, both sets) → `make_results_tables.py` (standalone
tables) → the canonical figures → `build_project_report.py` (`reports/project_summary.md`).
The one figure it does not rebuild is `lss_tier_tuning.png`, which comes from the
tuning stage (`tools/tune_lss_by_tier.py`).

---

## Regenerate tables & figures only

If the reconstruction `.pt` files already exist under `$RUN_ROOT/results/{final,final_quad}`,
you do **not** need GPUs or re-inference:

```bash
source ../.env.project
bash cluster/run_final_analysis.sh          # metrics + tables + figures + report
```

**Just the results tables** (Markdown + CSV, no prose, no figures):

```bash
python analysis/make_results_tables.py \
  --summary tables/project/summary_metrics.csv \
  --summary-quad tables/project/summary_metrics_quad.csv \
  --output-dir tables/project
# -> tables/project/results_tables.md + results_2view_fixed.csv / results_2view_2x.csv / results_4view.csv
```

**Just the reconstruction montages** (the main visual result):

```bash
python analysis/plot_recon_montage_tables.py \
  --final "$RUN_ROOT/results/final" --final-quad "$RUN_ROOT/results/final_quad" \
  --output-dir figures/project --cases 3
```

**Just the results-table bar charts:**

```bash
python analysis/plot_results_tables.py \
  --summary tables/project/summary_metrics.csv \
  --summary-quad tables/project/summary_metrics_quad.csv --output-dir figures/project
```

---

## Comparing against upstream

**Project additions (this work):**
- `utils/multiview_mri.py` - multi-view MRI operator + noise-weighted merge.
- `solve_inverse_mv_adps.py` - multi-view Ambient DPS sampler.
- `training/loss.py::CrossViewAmbientLoss`, `training/dataset.py::MultiViewKspaceDataset` - cross-view fine-tuning.
- `analysis/` - metrics, experiment registry, table + figure generation, the report builder.
- `tools/` - preprocessing, mask generation, tuning, manifests.
- `tests/` - multi-view operator + merge/joint-equivalence checks (`python -m pytest tests`).
- `configs/project/`, `cluster/`, `reports/`, `tables/project/`, `figures/project/`.

**Upstream files this work modifies:** `train.py` and `training/training_loop.py`
(the `--precond ambient_mv` / `--dataset-mode multiview_kspace` hooks), plus
`training/loss.py` and `training/dataset.py` for the two classes above.

**Unchanged upstream** - every other file is byte-for-byte identical to the upstream
base commit `75feb2f`, including the diffusion architecture and the original
single-view sampler: `training/networks.py`, `torch_utils/`, `solve_inverse_adps.py`,
`solve_inverse_1step.py`, `prior.py`, `dataset_tool.py`, `fid.py`. The
`ambient-diffusion-inverse/` git submodule is the original reference code.

---

## Repository map

```
analysis/         metrics, experiments.py (method registry), plotting, report + tables builders
tools/            preprocessing, mask generation, tuning, manifests, name migration
tests/            pytest checks for the multi-view operator and merge/joint equivalence
configs/project/      data / mask / inference / training / classical configs
cluster/          run_finetune.sh, run_final_diffusion.sh, run_final_analysis.sh
utils/            multiview_mri.py, mri_fft.py, checkpoint_arch.py, train_masks.py
reports/          project_summary.md (dataset/scope + results tables), project_notes.md (config record)
tables/project/       metric CSVs + results tables (md/csv)
figures/project/      the canonical figures (montages, results bars, conclusions)
solve_inverse_mv_adps.py    multi-view inference entry point
```


## Note

While the seminar project idea and concepts were solely created by us, the implementation itself was assisted by Claude.

Experiments and results were checked and verified before submission.