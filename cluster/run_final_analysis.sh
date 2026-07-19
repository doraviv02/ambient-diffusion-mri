#!/bin/bash
# Aggregate all reconstructions into the metric tables, then build the canonical
# figures and the final report. Assumes inference has already produced the .pt
# result files (see cluster/run_final_diffusion.sh and analysis/run_classical_recon.py).
set -euo pipefail
source /home/doraviv/miniconda3/etc/profile.d/conda.sh
conda activate ambient-mv
source /home/doraviv/Lowfield_Ambient/.env.mvp
cd "$PROJECT_ROOT"

FINAL="$RUN_ROOT/results/final"                 # 2-view result .pt files
FINAL_QUAD="$RUN_ROOT/results/final_quad"        # 4-view result .pt files
MASKS="$DATA_ROOT/processed/mvp_t2_masks"
QUAD_MASKS="$DATA_ROOT/processed/mvp_t2_quad_masks"

echo "=== metrics: per-slice -> subject-level (both evaluation sets) ==="
python analysis/compute_metrics.py --results-root "$FINAL"      --output tables/mvp/metrics_per_slice.csv
python analysis/compute_metrics.py --results-root "$FINAL_QUAD" --output tables/mvp/metrics_per_slice_quad.csv
python analysis/aggregate_subject_metrics.py --input tables/mvp/metrics_per_slice.csv \
  --subject-output tables/mvp/metrics_per_subject.csv \
  --summary-output tables/mvp/summary_metrics.csv --bootstrap-samples 10000 --seed 20260716
python analysis/aggregate_subject_metrics.py --input tables/mvp/metrics_per_slice_quad.csv \
  --subject-output tables/mvp/metrics_per_subject_quad.csv \
  --summary-output tables/mvp/summary_metrics_quad.csv --bootstrap-samples 10000 --seed 20260716

echo "=== results tables (markdown + CSV) ==="
python analysis/make_results_tables.py --summary tables/mvp/summary_metrics.csv \
  --summary-quad tables/mvp/summary_metrics_quad.csv --output-dir tables/mvp

echo "=== figures (canonical set) ==="
# MAIN results: reconstruction-image montages per results table
python analysis/plot_recon_montage_tables.py --final "$FINAL" --final-quad "$FINAL_QUAD" \
  --output-dir figures/mvp --cases 3
# results tables visualised (SSIM / NRMSE bars with 95% CI)
python analysis/plot_results_tables.py --summary tables/mvp/summary_metrics.csv \
  --summary-quad tables/mvp/summary_metrics_quad.csv --output-dir figures/mvp
# cross-experiment conclusions
python analysis/plot_experiment_overview.py \
  --per-subject tables/mvp/metrics_per_subject.csv \
  --per-subject-quad tables/mvp/metrics_per_subject_quad.csv --output-dir figures/mvp
# acquisition design (all mask conditions)
python analysis/plot_mask_design.py --masks-root "$QUAD_MASKS" --output figures/mvp/mask_design_all.png
# supporting: per-method absolute error maps (errors only; the montages above are the main recon view)
python analysis/plot_reconstructions.py --results-root "$FINAL" --output-dir figures/mvp --emit errors
python analysis/plot_reconstructions.py --results-root "$FINAL_QUAD" --output-dir figures/mvp --emit errors \
  --methods single_r4,dup4_merge,dup4_ft,comp4_merge,comp4_ft,fcomp4_merge,fcomp4_ft,full_diffusion,full_plain,dualfull_merge,dualfull_ft \
  --suffix _quad

echo "=== final report ==="
python analysis/build_mvp_report.py --summary tables/mvp/summary_metrics.csv --figures figures/mvp \
  --per-subject tables/mvp/metrics_per_subject.csv \
  --summary-quad tables/mvp/summary_metrics_quad.csv \
  --per-subject-quad tables/mvp/metrics_per_subject_quad.csv \
  --output reports/mvp_summary.md --budget-json "$MASKS/budget_report.json" \
  --preprocess-report "$DATA_ROOT/processed/mvp_t2/preprocessing_report.json" \
  --selected-inference configs/mvp/selected_inference.yaml \
  --selected-classical configs/mvp/selected_classical.yaml
echo "FINAL_ANALYSIS_DONE"
