#!/usr/bin/env bash
# Progressive Distillation — one level
#
# Usage:
#   bash scripts/distill_one.sh <task> <level> [gpu]
#
# <task>  : lift | can | square | transport | tool_hang
# <level> : 1..6
#   level 1 → 50 inference steps  (teacher: ckpt_bundle/<task>/last.pth)
#   level 2 → 25 steps            (teacher: results/distill_<task>_level1/...)
#   level 3 → 12 steps
#   level 4 →  6 steps
#   level 5 →  3 steps
#   level 6 →  1 step
# [gpu]   : GPU id (default 0)
set -e

task=$1
level=${2:-1}
gpu=${3:-0}

if [ -z "$task" ]; then
  echo "Usage: bash scripts/distill_one.sh <task> [level] [gpu]"
  echo "Example: bash scripts/distill_one.sh lift 1 0"
  exit 1
fi

case "$task" in
  lift|can|square|transport|tool_hang) ;;
  *) echo "Unknown task: $task"; exit 1 ;;
esac

# ---- paths ----
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CS570_ROOT=${CS570_ROOT:-$HOME/cs570-project}
ROBOMIMIC_DIR=${ROBOMIMIC_DIR:-$CS570_ROOT/robomimic}
OUTPUT_DIR="${PROJECT_DIR}/results"
CONFIG_TEMPLATE="${PROJECT_DIR}/configs/dp_lowdim_distill.json"

# ---- derived values ----
# teacher checkpoint
if [ "$level" -eq 1 ]; then
  TEACHER_CKPT="${PROJECT_DIR}/ckpt_bundle/${task}/last.pth"
else
  prev_level=$((level - 1))
  # find the best (or last) checkpoint from the previous level run
  PREV_DIR="${OUTPUT_DIR}/distill_${task}_level${prev_level}"
  # prefer best_model.pth (saved on best rollout SR), fall back to models/model_epoch_*.pth (latest)
  if [ -f "${PREV_DIR}/models/best_model.pth" ]; then
    TEACHER_CKPT="${PREV_DIR}/models/best_model.pth"
  else
    TEACHER_CKPT=$(ls "${PREV_DIR}/models/model_epoch_"*.pth 2>/dev/null | sort -V | tail -n 1)
    if [ -z "$TEACHER_CKPT" ]; then
      echo "ERROR: no checkpoint found in ${PREV_DIR}/models/"
      echo "Make sure level $((level-1)) has been trained first."
      exit 1
    fi
  fi
fi

# num_inference_timesteps = 100 // 2^level  (floor)
NUM_TRAIN=100
NUM_INFER=$(python3 -c "print(max(1, $NUM_TRAIN // (2**$level)))")

echo "============================================"
echo "TASK          : ${task}"
echo "DISTILL LEVEL : ${level}"
echo "TEACHER CKPT  : ${TEACHER_CKPT}"
echo "INFER STEPS   : ${NUM_INFER}"
echo "GPU           : ${gpu}"
echo "============================================"

# ---- build runtime config (inject dynamic values) ----
RUNTIME_CONFIG=$(mktemp)
trap 'rm -f "$RUNTIME_CONFIG"' EXIT

python3 - "$CONFIG_TEMPLATE" "$RUNTIME_CONFIG" "$OUTPUT_DIR" \
         "$TEACHER_CKPT" "$level" "$NUM_INFER" << 'PY'
import json, sys

src, dst, output_dir, teacher_ckpt, level, num_infer = sys.argv[1:7]
level = int(level)
num_infer = int(num_infer)

with open(src) as f:
    cfg = json.load(f)

cfg["train"]["output_dir"] = output_dir
cfg["algo"]["distill"]["teacher_ckpt_path"] = teacher_ckpt
cfg["algo"]["distill"]["distill_level"] = level
cfg["algo"]["ddpm"]["num_inference_timesteps"] = num_infer

with open(dst, "w") as f:
    json.dump(cfg, f, indent=4)
PY

# ---- rollout horizon per task ----
case "$task" in
  lift|can|square) horizon=400 ;;
  transport|tool_hang) horizon=700 ;;
esac

# patch rollout horizon in config
python3 - "$RUNTIME_CONFIG" "$horizon" << 'PY'
import json, sys
path, horizon = sys.argv[1], int(sys.argv[2])
with open(path) as f:
    cfg = json.load(f)
cfg["experiment"]["rollout"]["horizon"] = horizon
with open(path, "w") as f:
    json.dump(cfg, f, indent=4)
PY

# ---- dataset ----
dataset=$(ls "${ROBOMIMIC_DIR}/datasets/${task}/ph/low_dim"*.hdf5 | head -n 1)
if [ -z "$dataset" ]; then
  echo "ERROR: dataset not found at ${ROBOMIMIC_DIR}/datasets/${task}/ph/"
  exit 1
fi

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=$gpu

cd "$PROJECT_DIR"

python "${ROBOMIMIC_DIR}/robomimic/scripts/train.py" \
  --config "${RUNTIME_CONFIG}" \
  --dataset "${dataset}" \
  --name "distill_${task}_level${level}"
