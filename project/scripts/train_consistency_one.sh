#!/usr/bin/env bash
set -e

task=$1
gpu=${2:-0}
config=${3:-configs/cd_lowdim.json}

if [ -z "$task" ]; then
  echo "Usage: bash scripts/train_consistency_one.sh <task> [gpu] [config] [teacher_ckpt]"
  echo "Example: bash scripts/train_consistency_one.sh lift 0 configs/cd_lowdim.json"
  exit 1
fi

case "$task" in
  lift|can|square|transport|tool_hang) ;;
  *) echo "Unknown task: $task"; exit 1 ;;
esac

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CS570_ROOT=${CS570_ROOT:-$HOME/cs570}
ROBOMIMIC_DIR=${ROBOMIMIC_DIR:-$CS570_ROOT/robomimic}
OUTPUT_DIR="${PROJECT_DIR}/results"
CONFIG_PATH="${PROJECT_DIR}/${config}"
teacher=${4:-"${CS570_ROOT}/ckpt_bundle/${task}/last.pth"}
RUNTIME_CONFIG=$(mktemp)
trap 'rm -f "$RUNTIME_CONFIG"' EXIT

if [ ! -f "$teacher" ]; then
  echo "Teacher checkpoint not found: ${teacher}"
  exit 1
fi

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=$gpu

dataset=$(ls "${ROBOMIMIC_DIR}/datasets/${task}/ph/low_dim"*.hdf5 | head -n 1)

${PYTHON:-python} - "$CONFIG_PATH" "$RUNTIME_CONFIG" "$OUTPUT_DIR" "$teacher" <<'PY'
import json
import sys

src, dst, output_dir, teacher = sys.argv[1:5]
with open(src, "r", encoding="utf-8") as f:
    config = json.load(f)
config["train"]["output_dir"] = output_dir
config["algo"]["consistency"]["teacher_checkpoint_path"] = teacher
with open(dst, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)
PY

echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "ROBOMIMIC_DIR: ${ROBOMIMIC_DIR}"
echo "TASK: ${task}"
echo "GPU: ${gpu}"
echo "CONFIG: ${CONFIG_PATH}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "DATASET: ${dataset}"
echo "TEACHER: ${teacher}"

cd "$PROJECT_DIR"

${PYTHON:-python} "${ROBOMIMIC_DIR}/robomimic/scripts/train.py" \
  --config "${RUNTIME_CONFIG}" \
  --dataset "${dataset}" \
  --name "cd_${task}_$(basename "$config" .json)"
