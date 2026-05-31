#!/usr/bin/env bash
set -e
CS570_ROOT=${CS570_ROOT:-$HOME/cs570}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CKPT_DIR="$CS570_ROOT/ckpt_bundle"
SCRIPT="$CS570_ROOT/project/scripts/eval_deis.py"
TASKS=(lift can square transport tool_hang)
STEPS=(20 6 3 1)
N=${1:-50}

for task in "${TASKS[@]}"; do
  for s in "${STEPS[@]}"; do
    echo "============================================================"
    echo "TASK: $task | DEIS steps: $s | rollouts: $N"
    echo "============================================================"
    python "$SCRIPT" \
      --agent "$CKPT_DIR/$task/last.pth" \
      --task "$task" \
      --steps "$s" \
      --n_rollouts "$N" \
      --seed 0
  done
done

echo ""
echo "ALL DONE. Results: $CS570_ROOT/project/outputs/eval_deis/"
