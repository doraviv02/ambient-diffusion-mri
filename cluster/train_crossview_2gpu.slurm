#!/bin/bash
#SBATCH --job-name=mv-ambient
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=REPLACE_ME
#SBATCH --account=REPLACE_ME

set -euo pipefail

# --- site-specific: adapt only conda init + project/log paths + partition/account ---
source /home/doraviv/miniconda3/etc/profile.d/conda.sh
conda activate ambient-mv
source /home/doraviv/Lowfield_Ambient/.env.project

cd "$PROJECT_ROOT"

nvidia-smi
export WANDB_MODE=disabled

# This node has 4 GPUs; the project uses the last two (2,3). On a real SLURM
# allocation with --gres=gpu:2, drop CUDA_VISIBLE_DEVICES (SLURM sets it).
CUDA_VISIBLE_DEVICES=2,3 \
torchrun --standalone --nproc_per_node=2 \
  train.py \
  --data "$DATA_ROOT/processed/project_t2_masks/train" \
  --outdir "$RUN_ROOT/checkpoints/crossview_t2" \
  --experiment_name crossview_t2 \
  --dataset-mode multiview_kspace \
  --precond ambient_mv \
  --transfer "$AMBIENT_R4_DIR/network-snapshot.pkl" \
  --init-options "$AMBIENT_R4_DIR/training_options.json" \
  --val-data "$DATA_ROOT/processed/project_t2_masks/val" \
  --validation-every-kimg 2 \
  --batch 4 \
  --batch-gpu 1 \
  --lr 1e-5 \
  --duration 0.02 \
  --fp16 True \
  --compile_network False \
  --cache False \
  --tracking disabled \
  --workers 2 \
  --max_grad_norm 1.0 \
  --dropout 0.1 \
  --augment 0 \
  --snap 5 \
  --dump 5 \
  --tick 1 \
  --seed 20260716
