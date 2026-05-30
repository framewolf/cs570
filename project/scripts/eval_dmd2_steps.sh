#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

GPU=${GPU:-0}
STEPS=${STEPS:-"6 3 1"}
TASKS=${TASKS:-"lift can square transport tool_hang"}
EVAL_N=${EVAL_N:-50}
VIDEO_SKIP=${VIDEO_SKIP:-5}

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
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

checkpoint_for() {
  local task="$1"
  local step="$2"
  local run_name="dmd2_${task}_${step}step_gan"
  local run_dir=""

  run_dir=$(latest_run_dir "$run_name")
  if [ -z "$run_dir" ] && [ "$task" = "lift" ] && [ "$step" = "6" ]; then
    run_dir=$(latest_run_dir "dmd2_lift_dmd2_dp_lowdim_6step_gan")
  fi

  if [ -z "$run_dir" ] || [ ! -f "${run_dir}/last.pth" ]; then
    return 1
  fi

  echo "${run_dir}/last.pth"
}

log "starting DMD2 eval sweep"
log "steps=${STEPS}; tasks=${TASKS}; eval_n=${EVAL_N}; gpu=${GPU}"

for step in $STEPS; do
  for task in $TASKS; do
    tag="dmd2_${task}_${step}step_gan"
    ckpt=$(checkpoint_for "$task" "$step")
    log "evaluating ${tag}: ${ckpt}"
    EVAL_TAG="$tag" \
    bash "${PROJECT_DIR}/scripts/eval_one.sh" "$task" "$ckpt" "$EVAL_N" "$GPU" "$VIDEO_SKIP"
  done
done

"${PROJECT_DIR}/scripts/summarize_dmd2_eval.py"
log "done"
