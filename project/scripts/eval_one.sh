#!/usr/bin/env bash
set -e

task=$1
ckpt=$2
n=${3:-50}
gpu=${4:-0}

if [ -z "$task" ] || [ -z "$ckpt" ]; then
  echo "Usage: bash scripts/eval_one.sh <task> <checkpoint_path> [n_rollouts] [gpu]"
  echo "Example: bash scripts/eval_one.sh lift results/.../last.pth 50 0"
  exit 1
fi

case "$task" in
  lift|can|square) horizon=400 ;;
  transport|tool_hang) horizon=700 ;;
  *) echo "Unknown task: $task"; exit 1 ;;
esac

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CS570_ROOT=${CS570_ROOT:-$HOME/cs570}
ROBOMIMIC_DIR=${ROBOMIMIC_DIR:-$CS570_ROOT/robomimic}

mkdir -p "${PROJECT_DIR}/outputs/eval"

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=$gpu

echo "TASK: ${task}"
echo "CKPT: ${ckpt}"
echo "N_ROLLOUTS: ${n}"
echo "HORIZON: ${horizon}"
echo "GPU: ${gpu}"

python "${ROBOMIMIC_DIR}/robomimic/scripts/run_trained_agent.py" \
  --agent "${ckpt}" \
  --n_rollouts "${n}" \
  --horizon "${horizon}" \
  --seed 0 \
  --video_path "${PROJECT_DIR}/outputs/eval/${task}_eval.mp4" \
  --camera_names agentview \
  2>&1 | tee "${PROJECT_DIR}/outputs/eval/${task}_eval.log"

echo ""
echo "Video saved to: ${PROJECT_DIR}/outputs/eval/${task}_eval.mp4"
echo "Log saved to: ${PROJECT_DIR}/outputs/eval/${task}_eval.log"
