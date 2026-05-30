#!/usr/bin/env bash
set -euo pipefail

task=${1:-}
ckpt=${2:-}
n=${3:-50}
gpu=${4:-0}
video_skip=${5:-${VIDEO_SKIP:-5}}
eval_tag=${EVAL_TAG:-$task}
eval_tag=${eval_tag//\//_}

if [ -z "$task" ] || [ -z "$ckpt" ]; then
  echo "Usage: bash scripts/eval_one.sh <task> <checkpoint_path> [n_rollouts] [gpu|cpu] [video_skip]"
  echo "Example: bash scripts/eval_one.sh lift results/.../last.pth 1 0"
  exit 1
fi

case "$task" in
  lift|can|square) horizon=400 ;;
  transport|tool_hang) horizon=700 ;;
  *) echo "Unknown task: $task"; exit 1 ;;
esac

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CS570_ROOT=${CS570_ROOT:-$HOME/cs570}
ROBOMIMIC_DIR=${ROBOMIMIC_DIR:-$CS570_ROOT/robomimic}

if [ ! -f "$ckpt" ]; then
  echo "Checkpoint not found: ${ckpt}"
  exit 1
fi

mkdir -p "${PROJECT_DIR}/outputs/eval"
mkdir -p "${CS570_ROOT}/.cache" "${CS570_ROOT}/.hf" "${CS570_ROOT}/.mplconfig"
video_path="${PROJECT_DIR}/outputs/eval/${eval_tag}_eval.mp4"
log_path="${PROJECT_DIR}/outputs/eval/${eval_tag}_eval.log"

export MUJOCO_GL=${MUJOCO_GL:-egl}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${CS570_ROOT}/.cache}
export HF_HOME=${HF_HOME:-${CS570_ROOT}/.hf}
export MPLCONFIGDIR=${MPLCONFIGDIR:-${CS570_ROOT}/.mplconfig}
export PYTHONPATH="${ROBOMIMIC_DIR}:${CS570_ROOT}/robosuite${PYTHONPATH:+:${PYTHONPATH}}"

if [ "$gpu" = "cpu" ] || [ "$gpu" = "-1" ]; then
  export CUDA_VISIBLE_DEVICES=""
else
  export CUDA_VISIBLE_DEVICES=$gpu
fi

echo "TASK: ${task}"
echo "CKPT: ${ckpt}"
echo "N_ROLLOUTS: ${n}"
echo "HORIZON: ${horizon}"
echo "GPU: ${gpu}"
echo "VIDEO_SKIP: ${video_skip}"
echo "EVAL_TAG: ${eval_tag}"

python "${ROBOMIMIC_DIR}/robomimic/scripts/run_trained_agent.py" \
  --agent "${ckpt}" \
  --n_rollouts "${n}" \
  --horizon "${horizon}" \
  --seed 0 \
  --video_path "${video_path}" \
  --video_skip "${video_skip}" \
  --camera_names agentview \
  2>&1 | tee "${log_path}"

python - "$log_path" <<'PY' | tee -a "$log_path"
import ast
import json
import re
import sys

log_path = sys.argv[1]
with open(log_path, "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

match = re.search(r"Average Rollout Stats\s*\n(\{.*?\n\})", text, re.S)
if match:
    stats = ast.literal_eval(match.group(1))
    control_freq = 20.0
    horizon = float(stats["Horizon"])
    summary = {
        "Control_Frequency_Hz": control_freq,
        "Approx_Success_Time_Sec": horizon / control_freq,
    }
    print("Derived Eval Stats")
    print(json.dumps(summary, indent=4))
PY

echo ""
echo "Video saved to: ${video_path}"
echo "Log saved to: ${log_path}"
