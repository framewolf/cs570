#!/usr/bin/env bash
# UniPC NFE sweep across all 5 robomimic tasks.
# Iterates STEPS x TASKS and writes one JSON per (task, step) into
# project/outputs/eval_unipc/. Mirrors run_all_deis.sh's layout.
set -e

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CS570_ROOT=${CS570_ROOT:-$(cd "$PROJECT_DIR/.." && pwd)}
export CS570_ROOT
export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

SCRIPT="$PROJECT_DIR/scripts/eval_unipc.py"
TASKS=(can lift square transport tool_hang)
STEPS=(6 3 1)
N=${1:-50}

# resolve the most recent dp_lowdim_full run dir per task and use its 2000-epoch ckpt
ckpt_for() {
  ls -dt "$PROJECT_DIR/results/dp_$1_dp_lowdim_full/"*/ 2>/dev/null | head -1 \
    | xargs -I {} echo "{}models/model_epoch_2000.pth" | sed 's:/\+:/:g'
}

for task in "${TASKS[@]}"; do
  ckpt=$(ckpt_for "$task")
  if [ ! -f "$ckpt" ]; then
    echo "[skip] $task: no checkpoint found ($ckpt)"
    continue
  fi
  for s in "${STEPS[@]}"; do
    echo "============================================================"
    echo "TASK: $task | UniPC steps: $s | rollouts: $N"
    echo "ckpt: $ckpt"
    echo "============================================================"
    python "$SCRIPT" \
      --agent "$ckpt" \
      --task "$task" \
      --steps "$s" \
      --n_rollouts "$N" \
      --seed 0
  done
done

echo ""
echo "ALL DONE. Results: $PROJECT_DIR/outputs/eval_unipc/"
