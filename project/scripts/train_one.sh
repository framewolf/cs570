#!/usr/bin/env bash
set -e

task=$1
gpu=${2:-0}
config=${3:-configs/dp_lowdim_sanity.json}

if [ -z "$task" ]; then
  echo "Usage: bash scripts/train_one.sh <task> [gpu] [config]"
  echo "Example: bash scripts/train_one.sh lift 0 configs/dp_lowdim_sanity.json"
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
RUNTIME_CONFIG=$(mktemp)
trap 'rm -f "$RUNTIME_CONFIG"' EXIT

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=$gpu

dataset=$(ls "${ROBOMIMIC_DIR}/datasets/${task}/ph/low_dim"*.hdf5 | head -n 1)

python - "$CONFIG_PATH" "$RUNTIME_CONFIG" "$OUTPUT_DIR" <<'PY'
import json
import sys

src, dst, output_dir = sys.argv[1:4]
with open(src, "r", encoding="utf-8") as f:
    config = json.load(f)
config["train"]["output_dir"] = output_dir
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

cd "$PROJECT_DIR"

python "${ROBOMIMIC_DIR}/robomimic/scripts/train.py" \
  --config "${RUNTIME_CONFIG}" \
  --dataset "${dataset}" \
  --name "dp_${task}_$(basename "$config" .json)"
