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
resolve_config_path() {
  local path="$1"
  local dir

  if [ -f "$path" ]; then
    dir=$(cd "$(dirname "$path")" && pwd)
    echo "${dir}/$(basename "$path")"
  elif [ -f "${PROJECT_DIR}/${path}" ]; then
    echo "${PROJECT_DIR}/${path}"
  elif [ -f "${CS570_ROOT}/${path}" ]; then
    echo "${CS570_ROOT}/${path}"
  else
    echo "${PROJECT_DIR}/${path}"
  fi
}

CONFIG_PATH=$(resolve_config_path "$config")
RUNTIME_CONFIG=$(mktemp)
trap 'rm -f "$RUNTIME_CONFIG"' EXIT

mkdir -p "${CS570_ROOT}/.cache" "${CS570_ROOT}/.hf" "${CS570_ROOT}/.mplconfig"

export MUJOCO_GL=${MUJOCO_GL:-egl}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${CS570_ROOT}/.cache}
export HF_HOME=${HF_HOME:-${CS570_ROOT}/.hf}
export MPLCONFIGDIR=${MPLCONFIGDIR:-${CS570_ROOT}/.mplconfig}
export PYTHONPATH="${ROBOMIMIC_DIR}:${CS570_ROOT}/robosuite${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=$gpu

shopt -s nullglob
datasets=("${ROBOMIMIC_DIR}/datasets/${task}/ph/low_dim"*.hdf5)
shopt -u nullglob

if [ ${#datasets[@]} -eq 0 ]; then
  if [ "${AUTO_DOWNLOAD_DATA:-0}" = "1" ]; then
    echo "Dataset not found. Downloading ${task} PH low_dim dataset..."
    python "${ROBOMIMIC_DIR}/robomimic/scripts/download_datasets.py" \
      --download_dir "${ROBOMIMIC_DIR}/datasets" \
      --tasks "${task}" \
      --dataset_types ph \
      --hdf5_types low_dim
    shopt -s nullglob
    datasets=("${ROBOMIMIC_DIR}/datasets/${task}/ph/low_dim"*.hdf5)
    shopt -u nullglob
  fi
fi

if [ ${#datasets[@]} -eq 0 ]; then
  echo "Dataset not found: ${ROBOMIMIC_DIR}/datasets/${task}/ph/low_dim*.hdf5"
  echo "Download it with one of:"
  echo "  AUTO_DOWNLOAD_DATA=1 bash scripts/train_one.sh ${task} ${gpu} ${config}"
  echo "  bash scripts/download_lowdim_ph.sh"
  exit 1
fi

dataset="${datasets[0]}"

python - "$CONFIG_PATH" "$RUNTIME_CONFIG" "$OUTPUT_DIR" <<'PY'
import json
import os
import sys

src, dst, output_dir = sys.argv[1:4]
with open(src, "r", encoding="utf-8") as f:
    config = json.load(f)
config["train"]["output_dir"] = output_dir
wandb_flag = os.environ.get("WANDB", os.environ.get("LOG_WANDB", "")).lower()
if wandb_flag in {"1", "true", "yes", "on"}:
    config["experiment"]["logging"]["log_wandb"] = True
if os.environ.get("WANDB_PROJECT"):
    config["experiment"]["logging"]["wandb_proj_name"] = os.environ["WANDB_PROJECT"]
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
echo "WANDB: ${WANDB:-${LOG_WANDB:-off}}"

cd "$PROJECT_DIR"

python "${ROBOMIMIC_DIR}/robomimic/scripts/train.py" \
  --config "${RUNTIME_CONFIG}" \
  --dataset "${dataset}" \
  --name "dp_${task}_$(basename "$config" .json)"
