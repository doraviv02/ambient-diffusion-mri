#!/bin/bash
# Aggregate all reconstructions into the metric tables, then build the canonical
# figures and the final report. Assumes inference has already produced the .pt
# result files (see cluster/run_final_diffusion.sh and analysis/run_classical_recon.py).
set -euo pipefail
source /home/doraviv/miniconda3/etc/profile.d/conda.sh
conda activate ambient-mv
source /home/doraviv/Lowfield_Ambient/.env.project
cd "$PROJECT_ROOT"

FINAL="$RUN_ROOT/results/final"                 # 2-view result .pt files
FINAL_QUAD="$RUN_ROOT/results/final_quad"        # 4-view result .pt files
MASKS="$DATA_ROOT/processed/project_t2_masks"
QUAD_MASKS="$DATA_ROOT/processed/project_t2_quad_masks"

echo "=== metrics: per-slice -> subject-level (both evaluation sets) ==="
python analysis/compute_metrics.py --results-root "$FINAL"      --output tables/project/metrics_per_slice.csv
python analysis/compute_metrics.py --results-root "$FINAL_QUAD" --output tables/project/metrics_per_slice_quad.csv
python analysis/aggregate_subject_metrics.py --input tables/project/metrics_per_slice.csv \
  --subject-output tables/project/metrics_per_subject.csv \
  --summary-output tables/project/summary_metrics.csv --bootstrap-samples 10000 --seed 20260716
python analysis/aggregate_subject_metrics.py --input tables/project/metrics_per_slice_quad.csv \
  --subject-output tables/project/metrics_per_subject_quad.csv \
  --summary-output tables/project/summary_metrics_quad.csv --bootstrap-samples 10000 --seed 20260716

echo "=== results tables (markdown + CSV) ==="
python analysis/make_results_tables.py --summary tables/project/summary_metrics.csv \
  --summary-quad tables/project/summary_metrics_quad.csv --output-dir tables/project

echo "=== figures (canonical set) ==="
# MAIN results: reconstruction-image montages per results table
python analysis/plot_recon_montage_tables.py --final "$FINAL" --final-quad "$FINAL_QUAD" \
  --output-dir figures/project --cases 3
# results tables visualised (SSIM / NRMSE bars with 95% CI)
python analysis/plot_results_tables.py --summary tables/project/summary_metrics.csv \
  --summary-quad tables/project/summary_metrics_quad.csv --output-dir figures/project
# cross-experiment conclusions
python analysis/plot_experiment_overview.py \
  --per-subject tables/project/metrics_per_subject.csv \
  --per-subject-quad tables/project/metrics_per_subject_quad.csv --output-dir figures/project
# acquisition design (all mask conditions)
python analysis/plot_mask_design.py --masks-root "$QUAD_MASKS" --output figures/project/mask_design_all.png
# supporting: per-method absolute error maps (errors only; the montages above are the main recon view)
python analysis/plot_reconstructions.py --results-root "$FINAL" --output-dir figures/project --emit errors
python analysis/plot_reconstructions.py --results-root "$FINAL_QUAD" --output-dir figures/project --emit errors \
  --methods single_r4,dup4_merge,dup4_ft,comp4_merge,comp4_ft,fcomp4_merge,fcomp4_ft,full_diffusion,full_plain,dualfull_merge,dualfull_ft \
  --suffix _quad

echo "=== final report ==="
python analysis/build_project_report.py --summary tables/project/summary_metrics.csv --figures figures/project \
  --per-subject tables/project/metrics_per_subject.csv \
  --summary-quad tables/project/summary_metrics_quad.csv \
  --per-subject-quad tables/project/metrics_per_subject_quad.csv \
  --output reports/project_summary.md --budget-json "$MASKS/budget_report.json" \
  --preprocess-report "$DATA_ROOT/processed/project_t2/preprocessing_report.json" \
  --selected-inference configs/project/selected_inference.yaml \
  --selected-classical configs/project/selected_classical.yaml
echo "FINAL_ANALYSIS_DONE"
