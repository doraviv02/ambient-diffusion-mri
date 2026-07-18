#!/bin/bash
# Final diffusion experiment: select best fine-tuned checkpoint, verify it loads,
# run 2-view diffusion methods sharded on GPUs 2,3, then the uncertainty subset (fixed_split_ft, seeds 0-3).
set -euo pipefail
source /home/doraviv/miniconda3/etc/profile.d/conda.sh
conda activate ambient-mv
source /home/doraviv/Lowfield_Ambient/.env.mvp
cd "$PROJECT_ROOT"

CONFIG=configs/mvp/selected_inference.yaml
EVAL_MAN="$DATA_ROOT/processed/mvp_t2_masks/test_eval_manifest.csv"
UNC_MAN="$DATA_ROOT/processed/mvp_t2_masks/uncertainty_subset.csv"
FINAL="$RUN_ROOT/results/final"
UNC="$RUN_ROOT/results/uncertainty"
mkdir -p "$FINAL" "$UNC"

echo "=== [1/4] select best fine-tuned checkpoint ==="
python tools/select_best_checkpoint.py --run-dir "$RUN_ROOT/checkpoints/crossview_t2" \
  --out "$RUN_ROOT/checkpoints/crossview_t2/best"

echo "=== [2/4] verify fine-tuned checkpoint loads ==="
CUDA_VISIBLE_DEVICES=2 python -c "
import sys, torch; sys.path.insert(0,'.')
from solve_inverse_mv_adps import load_network
load_network('$RUN_ROOT/checkpoints/crossview_t2/best', torch.device('cuda'))
print('FINETUNED_LOADS_OK')"

echo "=== [3/4] 2-view diffusion sharded on GPUs 2,3 ==="
CUDA_VISIBLE_DEVICES=2 python solve_inverse_mv_adps.py --config "$CONFIG" --manifest "$EVAL_MAN" \
  --methods single_r4,fixed_split_merge,fixed_split_ft,dup2_merge,dup2_ft,comp2_merge,comp2_ft,fcomp2_merge,fcomp2_ft --shard_id 0 --num_shards 2 --save_dc_trajectory --output_dir "$FINAL" &
CUDA_VISIBLE_DEVICES=3 python solve_inverse_mv_adps.py --config "$CONFIG" --manifest "$EVAL_MAN" \
  --methods single_r4,fixed_split_merge,fixed_split_ft,dup2_merge,dup2_ft,comp2_merge,comp2_ft,fcomp2_merge,fcomp2_ft --shard_id 1 --num_shards 2 --save_dc_trajectory --output_dir "$FINAL" &
wait
echo "2-view diffusion done"

echo "=== [4/4] uncertainty fixed_split_ft seeds 0-3 sharded ==="
CUDA_VISIBLE_DEVICES=2 python solve_inverse_mv_adps.py --config "$CONFIG" --manifest "$UNC_MAN" \
  --methods fixed_split_ft --seeds 0,1,2,3 --shard_id 0 --num_shards 2 --output_dir "$UNC" &
CUDA_VISIBLE_DEVICES=3 python solve_inverse_mv_adps.py --config "$CONFIG" --manifest "$UNC_MAN" \
  --methods fixed_split_ft --seeds 0,1,2,3 --shard_id 1 --num_shards 2 --output_dir "$UNC" &
wait
echo "FINAL_DIFFUSION_DONE"
echo "final results: $(ls $FINAL/*.pt | wc -l), uncertainty: $(ls $UNC/*.pt | wc -l)"
