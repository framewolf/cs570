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

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=$gpu

dataset=$(ls "${ROBOMIMIC_DIR}/datasets/${task}/ph/low_dim"*.hdf5 | head -n 1)

echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "ROBOMIMIC_DIR: ${ROBOMIMIC_DIR}"
echo "TASK: ${task}"
echo "GPU: ${gpu}"
echo "CONFIG: ${PROJECT_DIR}/${config}"
echo "DATASET: ${dataset}"

cd "$PROJECT_DIR"

python "${ROBOMIMIC_DIR}/robomimic/scripts/train.py" \
  --config "${PROJECT_DIR}/${config}" \
  --dataset "${dataset}" \
  --name "dp_${task}_$(basename "$config" .json)"
