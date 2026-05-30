#!/usr/bin/env bash
set -euo pipefail

task=${1:-}
gpu=${2:-0}
config=${3:-configs/dmd2_dp_lowdim_sanity.json}
teacher_ckpt=${4:-}

if [ -z "$task" ]; then
  echo "Usage: bash scripts/train_dmd2_one.sh <task> [gpu|cpu] [config] [teacher_ckpt]"
  echo "Example: bash scripts/train_dmd2_one.sh lift 0 configs/dmd2_dp_lowdim_4step.json"
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
config_base=$(basename "$CONFIG_PATH" .json)
config_tag=${config_base#dmd2_dp_lowdim_}
run_name=${DMD2_RUN_NAME:-${RUN_NAME:-dmd2_${task}_${config_tag}}}

bundle_parent=${TEACHER_CKPT_DIR:-${CS570_ROOT}/dp_teacher_checkpoints_5tasks}
bundle_root=${TEACHER_CKPT_BUNDLE:-${bundle_parent}/ckpt_bundle}
tarball=${TEACHER_CKPT_TARBALL:-${CS570_ROOT}/dp_teacher_checkpoints_5tasks.tar.gz}

mkdir -p "${CS570_ROOT}/.cache" "${CS570_ROOT}/.hf" "${CS570_ROOT}/.mplconfig"

export MUJOCO_GL=${MUJOCO_GL:-egl}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${CS570_ROOT}/.cache}
export HF_HOME=${HF_HOME:-${CS570_ROOT}/.hf}
export MPLCONFIGDIR=${MPLCONFIGDIR:-${CS570_ROOT}/.mplconfig}
export PYTHONPATH="${ROBOMIMIC_DIR}:${CS570_ROOT}/robosuite${PYTHONPATH:+:${PYTHONPATH}}"

if [ -z "$teacher_ckpt" ]; then
  teacher_ckpt="${bundle_root}/${task}/last.pth"
fi

if [ ! -f "$teacher_ckpt" ] && [ -f "$tarball" ]; then
  echo "Teacher checkpoint not found, extracting ${tarball}"
  mkdir -p "$bundle_parent"
  tar -xzf "$tarball" -C "$bundle_parent"
fi

if [ ! -f "$teacher_ckpt" ]; then
  echo "Teacher checkpoint not found: ${teacher_ckpt}"
  echo "Expected tarball: ${tarball}"
  exit 1
fi

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
  echo "  AUTO_DOWNLOAD_DATA=1 bash scripts/train_dmd2_one.sh ${task} ${gpu} ${config}"
  echo "  bash scripts/download_lowdim_ph.sh"
  exit 1
fi

dataset="${datasets[0]}"

if [ "$gpu" = "cpu" ] || [ "$gpu" = "-1" ]; then
  export CUDA_VISIBLE_DEVICES=""
else
  export CUDA_VISIBLE_DEVICES=$gpu
fi

python - "$CONFIG_PATH" "$RUNTIME_CONFIG" "$OUTPUT_DIR" "$teacher_ckpt" <<'PY'
import json
import os
import sys

src, dst, output_dir, teacher_ckpt = sys.argv[1:5]
with open(src, "r", encoding="utf-8") as f:
    config = json.load(f)

config["train"]["output_dir"] = output_dir
config["algo"]["dmd2"]["teacher"]["ckpt_path"] = teacher_ckpt

if os.environ.get("EPOCHS"):
    config["train"]["num_epochs"] = int(os.environ["EPOCHS"])

if os.environ.get("SAVE_EVERY_N_EPOCHS"):
    config["experiment"]["save"]["every_n_epochs"] = int(os.environ["SAVE_EVERY_N_EPOCHS"])
else:
    config["experiment"]["save"]["every_n_epochs"] = int(config["train"]["num_epochs"])

wandb_flag = os.environ.get("WANDB", os.environ.get("LOG_WANDB", "")).lower()
if wandb_flag in {"1", "true", "yes", "on"}:
    config["experiment"]["logging"]["log_wandb"] = True
if os.environ.get("WANDB_PROJECT"):
    config["experiment"]["logging"]["wandb_proj_name"] = os.environ["WANDB_PROJECT"]

rollout_flag = os.environ.get("ROLLOUT", os.environ.get("EVAL_ROLLOUT", "")).lower()
if rollout_flag in {"1", "true", "yes", "on"}:
    config["experiment"]["rollout"]["enabled"] = True
if os.environ.get("ROLLOUT_RATE"):
    config["experiment"]["rollout"]["rate"] = int(os.environ["ROLLOUT_RATE"])
if os.environ.get("ROLLOUT_N"):
    config["experiment"]["rollout"]["n"] = int(os.environ["ROLLOUT_N"])
if os.environ.get("ROLLOUT_HORIZON"):
    config["experiment"]["rollout"]["horizon"] = int(os.environ["ROLLOUT_HORIZON"])
if os.environ.get("ROLLOUT_WARMSTART"):
    config["experiment"]["rollout"]["warmstart"] = int(os.environ["ROLLOUT_WARMSTART"])
if os.environ.get("ROLLOUT_VIDEO"):
    config["experiment"]["render_video"] = os.environ["ROLLOUT_VIDEO"].lower() in {"1", "true", "yes", "on"}

gan_flag = os.environ.get("DMD2_GAN", "").lower()
if gan_flag in {"1", "true", "yes", "on"}:
    config["algo"]["dmd2"]["gan"]["enabled"] = True
if os.environ.get("DMD2_GAN_GEN_WEIGHT"):
    config["algo"]["dmd2"]["gan"]["generator_loss_weight"] = float(os.environ["DMD2_GAN_GEN_WEIGHT"])
if os.environ.get("DMD2_GAN_DISC_WEIGHT"):
    config["algo"]["dmd2"]["gan"]["discriminator_loss_weight"] = float(os.environ["DMD2_GAN_DISC_WEIGHT"])
if os.environ.get("DMD2_GAN_MAX_STEP"):
    config["algo"]["dmd2"]["gan"]["max_step"] = int(os.environ["DMD2_GAN_MAX_STEP"])

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
echo "TEACHER: ${teacher_ckpt}"
echo "RUN_NAME: ${run_name}"
echo "EPOCHS_OVERRIDE: ${EPOCHS:-none}"
echo "DMD2_GAN_OVERRIDE: ${DMD2_GAN:-none}"
python - "$RUNTIME_CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    config = json.load(f)

print(f"CONFIG_EPOCHS: {config['train']['num_epochs']}")
print(f"CONFIG_SAVE_EVERY_N_EPOCHS: {config['experiment']['save']['every_n_epochs']}")
print(f"CONFIG_DMD2_GAN_ENABLED: {config['algo']['dmd2']['gan']['enabled']}")
PY
echo "WANDB: ${WANDB:-${LOG_WANDB:-off}}"
echo "ROLLOUT: ${ROLLOUT:-${EVAL_ROLLOUT:-off}}"

cd "$PROJECT_DIR"

python "${ROBOMIMIC_DIR}/robomimic/scripts/train.py" \
  --config "${RUNTIME_CONFIG}" \
  --dataset "${dataset}" \
  --name "${run_name}"
