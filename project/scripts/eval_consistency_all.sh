#!/usr/bin/env bash
set -euo pipefail

n=${1:-50}
if [ "$#" -gt 0 ]; then
  shift
fi

gpu=${1:-0}
if [ "$#" -gt 0 ]; then
  shift
fi

if [ "$#" -gt 0 ]; then
  steps=("$@")
else
  steps=(1 3 6)
fi

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CS570_ROOT=${CS570_ROOT:-$(cd "${PROJECT_DIR}/.." && pwd)}
OUT_ROOT=${OUT_ROOT:-"${PROJECT_DIR}/outputs/eval_consistency_all/$(date +%Y%m%d_%H%M%S)"}

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES="${gpu}"

tasks=(lift can square transport tool_hang)

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/json" "${OUT_ROOT}/videos"

echo "OUT_ROOT: ${OUT_ROOT}"
echo "N_ROLLOUTS: ${n}"
echo "GPU: ${gpu}"
echo "STEPS: ${steps[*]}"
echo ""

for task in "${tasks[@]}"; do
  case "${task}" in
    lift|can|square) horizon=400 ;;
    transport|tool_hang) horizon=700 ;;
    *) echo "Unknown task: ${task}"; exit 1 ;;
  esac

  ckpt=$(find "${PROJECT_DIR}/results/cd_${task}_cd_lowdim" -mindepth 2 -maxdepth 2 -name last.pth | sort | tail -n 1)
  if [ -z "${ckpt}" ]; then
    echo "Missing checkpoint for ${task}: ${PROJECT_DIR}/results/cd_${task}_cd_lowdim/*/last.pth" >&2
    exit 1
  fi
  if [ ! -f "${ckpt}" ]; then
    echo "Missing checkpoint for ${task}: ${ckpt}" >&2
    exit 1
  fi

  echo "============================================================"
  echo "TASK: ${task}"
  echo "CKPT: ${ckpt}"
  echo "HORIZON: ${horizon}"
  echo "============================================================"

  python3 "${PROJECT_DIR}/scripts/eval_consistency_steps.py" \
    --agent "${ckpt}" \
    --n_rollouts "${n}" \
    --horizon "${horizon}" \
    --seed 0 \
    --steps "${steps[@]}" \
    --output "${OUT_ROOT}/json/${task}.json" \
    --video_dir "${OUT_ROOT}/videos/${task}" \
    --camera_names agentview \
    2>&1 | tee "${OUT_ROOT}/logs/${task}.log"
done

python3 "${PROJECT_DIR}/scripts/summarize_consistency_results.py" "${OUT_ROOT}/json" "${OUT_ROOT}/summary.csv" "${OUT_ROOT}/summary.json"

echo ""
echo "Done."
echo "Logs: ${OUT_ROOT}/logs"
echo "Per-task JSON: ${OUT_ROOT}/json"
echo "Videos: ${OUT_ROOT}/videos"
echo "Summary CSV: ${OUT_ROOT}/summary.csv"
echo "Summary JSON: ${OUT_ROOT}/summary.json"
