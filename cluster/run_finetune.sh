#!/bin/bash
# Direct (non-SLURM) 2-GPU cross-view fine-tune launcher for this node (GPUs 2,3).
set -euo pipefail
source /home/doraviv/miniconda3/etc/profile.d/conda.sh
conda activate ambient-mv
source /home/doraviv/Lowfield_Ambient/.env.project
cd "$PROJECT_ROOT"
export WANDB_MODE=disabled

# validation cadence aligned with snapshot cadence (both 5 kimg) so the best
# checkpoint can be selected directly from the validation curve.
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 \
  train.py \
  --data "$DATA_ROOT/processed/project_t2_masks/train" \
  --outdir "$RUN_ROOT/checkpoints/crossview_t2" \
  --experiment_name crossview_t2 \
  --dataset-mode multiview_kspace \
  --precond ambient_mv \
  --transfer "$AMBIENT_R4_DIR/network-snapshot.pkl" \
  --init-options "$AMBIENT_R4_DIR/training_options.json" \
  --val-data "$DATA_ROOT/processed/project_t2_masks/val" \
  --validation-every-kimg 5 \
  --batch 4 --batch-gpu 1 --lr 1e-4 --duration 0.02 \
  --ema 0.002 \
  --fp16 True --compile_network False --cache False --tracking disabled \
  --workers 2 --max_grad_norm 1.0 --dropout 0.1 --augment 0 \
  --snap 5 --dump 100 --tick 1 --seed 20260716
# NOTE: EMA half-life reduced to 2 kimg (default 500 kimg leaves the saved EMA
# pinned at the pretrained weights over a 20-kimg fine-tune); lr raised to 1e-4
# so the cross-view loss actually decreases (1e-5 left it flat at this loss scale).
