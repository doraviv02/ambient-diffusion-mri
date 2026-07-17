#!/bin/bash
# Two-GPU deterministic sharded multi-view inference (runbook Section 16).
# Uses GPUs 2,3 on this node. Adjust CONFIG/MANIFEST/METHODS as needed.
set -euo pipefail
source "$(dirname "$0")/.env.mvp" 2>/dev/null || true

CONFIG=${CONFIG:-configs/mvp/selected_inference.yaml}
MANIFEST=${MANIFEST:-$DATA_ROOT/processed/mvp_t2_masks/test_manifest.csv}
METHODS=${METHODS:-M2,M3,M4,M5,M6}
OUTDIR=${OUTDIR:-$RUN_ROOT/results/final}
GPUS=${GPUS:-2,3}
IFS=',' read -r G0 G1 <<< "$GPUS"

CUDA_VISIBLE_DEVICES=$G0 python solve_inverse_mv_adps.py \
  --config "$CONFIG" --manifest "$MANIFEST" --methods "$METHODS" \
  --shard_id 0 --num_shards 2 --save_dc_trajectory --output_dir "$OUTDIR" &
CUDA_VISIBLE_DEVICES=$G1 python solve_inverse_mv_adps.py \
  --config "$CONFIG" --manifest "$MANIFEST" --methods "$METHODS" \
  --shard_id 1 --num_shards 2 --save_dc_trajectory --output_dir "$OUTDIR" &
wait
echo "sharded inference complete -> $OUTDIR"
