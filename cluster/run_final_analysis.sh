#!/bin/bash
# Aggregate all results and build every figure + the final report + manifest.
set -euo pipefail
source /home/doraviv/miniconda3/etc/profile.d/conda.sh
conda activate ambient-mv
source /home/doraviv/Lowfield_Ambient/.env.mvp
cd "$PROJECT_ROOT"

FINAL="$RUN_ROOT/results/final"
UNC="$RUN_ROOT/results/uncertainty"
MASKS="$DATA_ROOT/processed/mvp_t2_masks"

echo "=== metrics (final: M0-M6) ==="
python analysis/compute_metrics.py --results-root "$FINAL" --output tables/mvp/metrics_per_slice.csv

echo "=== subject-level aggregation + bootstrap ==="
python analysis/aggregate_subject_metrics.py --input tables/mvp/metrics_per_slice.csv \
  --subject-output tables/mvp/metrics_per_subject.csv \
  --summary-output tables/mvp/summary_metrics.csv --bootstrap-samples 10000 --seed 20260716

echo "=== figures ==="
FINAL_QUAD="$RUN_ROOT/results/final_quad"
QUAD_MASKS="$DATA_ROOT/processed/mvp_t2_quad_masks"

# cross-experiment figures (the ones carrying the conclusions)
python analysis/plot_experiment_overview.py \
  --per-subject tables/mvp/metrics_per_subject.csv \
  --per-subject-quad tables/mvp/metrics_per_subject_quad.csv \
  --output-dir figures/mvp
python analysis/plot_mask_design.py --masks-root "$QUAD_MASKS" \
  --output figures/mvp/mask_design_all.png

# per-set montages / boxplots
python analysis/plot_reconstructions.py --results-root "$FINAL" --output-dir figures/mvp
python analysis/plot_reconstructions.py --results-root "$FINAL_QUAD" --output-dir figures/mvp \
  --methods M2,M11,M10,M9,M18,M17,M16,M12,MF --suffix _quad --zf-condition joint_quad_comp
python analysis/plot_metrics.py --metrics tables/mvp/metrics_per_subject.csv --output-dir figures/mvp \
  --set main --budget-json "$MASKS/budget_report.json"
python analysis/plot_metrics.py --metrics tables/mvp/metrics_per_subject_quad.csv \
  --output-dir figures/mvp --set quad --suffix _quad

python analysis/plot_data_consistency.py --results-root "$FINAL" \
  --output figures/mvp/data_consistency_curves.png
python analysis/plot_uncertainty.py --results-root "$UNC" --output-dir figures/mvp

echo "=== report + manifest ==="
python analysis/build_mvp_report.py --summary tables/mvp/summary_metrics.csv --figures figures/mvp \
  --per-subject tables/mvp/metrics_per_subject.csv \
  --summary-quad tables/mvp/summary_metrics_quad.csv \
  --per-subject-quad tables/mvp/metrics_per_subject_quad.csv \
  --output reports/mvp_summary.md --budget-json "$MASKS/budget_report.json" \
  --preprocess-report "$DATA_ROOT/processed/mvp_t2/preprocessing_report.json" \
  --selected-inference configs/mvp/selected_inference.yaml \
  --selected-classical configs/mvp/selected_classical.yaml
python tools/write_run_manifest.py --output RUN_MANIFEST.json --repo . \
  --upstream-commit "$RUN_ROOT/upstream_commit.txt" \
  --checkpoint "$AMBIENT_R4_DIR/network-snapshot.pkl" \
  --dataset-manifest "$MASKS/test_manifest.csv" \
  --config configs/mvp/selected_inference.yaml --seeds 0 --status complete \
  --components "$RUN_ROOT/ambient_r4_checkpoint.sha256"
echo "FINAL_ANALYSIS_DONE"
