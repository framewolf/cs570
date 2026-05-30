#!/usr/bin/env bash
set -euo pipefail

task=${1:-lift}
n=${2:-1}
gpu=${3:-0}
video_skip=${4:-${VIDEO_SKIP:-5}}

case "$task" in
  lift|can|square|transport|tool_hang) ;;
  *)
    echo "Unknown task: ${task}"
    echo "Usage: bash project/scripts/eval_teacher_one.sh [lift|can|square|transport|tool_hang] [n_rollouts] [gpu|cpu] [video_skip]"
    exit 1
    ;;
esac

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CS570_ROOT=${CS570_ROOT:-$HOME/cs570}

bundle_parent=${TEACHER_CKPT_DIR:-${CS570_ROOT}/dp_teacher_checkpoints_5tasks}
bundle_root=${TEACHER_CKPT_BUNDLE:-${bundle_parent}/ckpt_bundle}
tarball=${TEACHER_CKPT_TARBALL:-${CS570_ROOT}/dp_teacher_checkpoints_5tasks.tar.gz}
ckpt="${bundle_root}/${task}/last.pth"

if [ ! -f "$ckpt" ] && [ -f "$tarball" ]; then
  echo "Checkpoint not found, extracting ${tarball}"
  mkdir -p "$bundle_parent"
  tar -xzf "$tarball" -C "$bundle_parent"
fi

if [ ! -f "$ckpt" ]; then
  echo "Checkpoint not found: ${ckpt}"
  echo "Expected tarball: ${tarball}"
  exit 1
fi

exec bash "${PROJECT_DIR}/scripts/eval_one.sh" "$task" "$ckpt" "$n" "$gpu" "$video_skip"
