#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CS570_ROOT=${CS570_ROOT:-$HOME/cs570}

GPU=${GPU:-0}
STEPS=${STEPS:-"3 1"}
TASKS=${TASKS:-"lift can square transport tool_hang"}
EVAL_N=${EVAL_N:-10}
VIDEO_SKIP=${VIDEO_SKIP:-5}
WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-cs570-dmd2}
ROLLOUT=${ROLLOUT:-1}
ROLLOUT_RATE=${ROLLOUT_RATE:-100}
ROLLOUT_N=${ROLLOUT_N:-10}
EPOCHS=${EPOCHS:-500}
WAIT_POLL_SECONDS=${WAIT_POLL_SECONDS:-120}

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" >&2
}

latest_run_dir() {
  local run_name="$1"
  if [ ! -d "${PROJECT_DIR}/results/${run_name}" ]; then
    return 0
  fi
  find "${PROJECT_DIR}/results/${run_name}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR==1 {print $2}'
}

preferred_checkpoint() {
  local run_name="$1"
  local run_dir
  run_dir=$(latest_run_dir "$run_name")
  if [ -z "$run_dir" ]; then
    return 1
  fi

  if [ -f "${run_dir}/last.pth" ]; then
    echo "${run_dir}/last.pth"
    return 0
  fi
  if [ -f "${run_dir}/models/model_epoch_${EPOCHS}.pth" ]; then
    echo "${run_dir}/models/model_epoch_${EPOCHS}.pth"
    return 0
  fi
  return 1
}

cleanup_last_backups() {
  local run_name="$1"
  find "${PROJECT_DIR}/results/${run_name}" -name 'last_bak.pth' -delete 2>/dev/null || true
}

final_checkpoint() {
  local run_name="$1"
  local run_dir
  run_dir=$(latest_run_dir "$run_name")
  if [ -z "$run_dir" ]; then
    return 1
  fi

  if [ -f "${run_dir}/last.pth" ]; then
    echo "${run_dir}/last.pth"
    return 0
  fi
  return 1
}

wait_for_existing_run() {
  local run_name="$1"
  local ckpt=""

  while true; do
    ckpt=$(final_checkpoint "$run_name" || true)
    if [ -n "$ckpt" ]; then
      echo "$ckpt"
      return 0
    fi
    log "waiting for ${run_name} last.pth"
    sleep "$WAIT_POLL_SECONDS"
  done
}

train_if_needed() {
  local task="$1"
  local step="$2"
  local run_name="dmd2_${task}_${step}step_gan"
  local config="configs/dmd2_dp_lowdim_${step}step_gan.json"
  local ckpt=""
  local run_dir=""
  CKPT_RESULT=""

  ckpt=$(final_checkpoint "$run_name" || true)
  if [ -n "$ckpt" ]; then
    log "found completed ${run_name}: ${ckpt}"
    cleanup_last_backups "$run_name"
    CKPT_RESULT="$ckpt"
    return 0
  fi

  run_dir=$(latest_run_dir "$run_name")
  if [ -n "$run_dir" ]; then
    log "found existing ${run_name} run without final checkpoint: ${run_dir}"
    ckpt=$(preferred_checkpoint "$run_name" || true)
    if [ -n "$ckpt" ] && [ "$task" = "lift" ] && [ "$step" = "3" ]; then
      log "assuming current lift 3step run is still active; waiting for final checkpoint"
      CKPT_RESULT=$(wait_for_existing_run "$run_name")
      return 0
    fi
  fi

  log "training ${run_name}"
  WANDB="$WANDB" \
  WANDB_PROJECT="$WANDB_PROJECT" \
  ROLLOUT="$ROLLOUT" \
  ROLLOUT_RATE="$ROLLOUT_RATE" \
  ROLLOUT_N="$ROLLOUT_N" \
  EPOCHS="$EPOCHS" \
  bash "${PROJECT_DIR}/scripts/train_dmd2_one.sh" "$task" "$GPU" "$config"

  ckpt=$(final_checkpoint "$run_name" || true)
  if [ -z "$ckpt" ]; then
    ckpt=$(preferred_checkpoint "$run_name" || true)
  fi
  if [ -z "$ckpt" ]; then
    log "no checkpoint found after training ${run_name}"
    return 1
  fi
  cleanup_last_backups "$run_name"
  CKPT_RESULT="$ckpt"
}

eval_checkpoint() {
  local task="$1"
  local step="$2"
  local ckpt="$3"
  local run_name="dmd2_${task}_${step}step_gan"
  local log_path="${PROJECT_DIR}/outputs/eval/${run_name}_eval.log"

  if [ -f "$log_path" ] && rg -q "Average Rollout Stats" "$log_path"; then
    log "found completed eval for ${run_name}: ${log_path}"
    return 0
  fi

  log "evaluating ${run_name}: ${ckpt}"
  EVAL_TAG="$run_name" \
  bash "${PROJECT_DIR}/scripts/eval_one.sh" "$task" "$ckpt" "$EVAL_N" "$GPU" "$VIDEO_SKIP"
}

log "starting DMD2 step sweep inside ${CS570_ROOT}"
log "steps=${STEPS}; tasks=${TASKS}; eval_n=${EVAL_N}; gpu=${GPU}"

for step in $STEPS; do
  for task in $TASKS; do
    train_if_needed "$task" "$step"
    eval_checkpoint "$task" "$step" "$CKPT_RESULT"
  done
done

log "done"
