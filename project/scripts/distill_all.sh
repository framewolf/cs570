#!/usr/bin/env bash
# Progressive Distillation — full pipeline for ONE task
#
# Usage:
#   bash scripts/distill_all.sh <task> [gpu] [num_iters] [batch_size] [save_every] [wandb_project]
#
# Example:
#   bash scripts/distill_all.sh lift 0 20000 64 2000 cs570-distill
#
# Folder layout created:
#   project/distill_ckpts/<task>/
#     teacher_step100/  teacher.pt           (converted from ckpt_bundle)
#     student_step050/  student.pt  iter_*.pt
#     student_step025/  student.pt  iter_*.pt
#     student_step012/  student.pt  iter_*.pt
#     student_step006/  student.pt  iter_*.pt
#     student_step003/  student.pt  iter_*.pt
#     student_step001/  student.pt  iter_*.pt
set -e

task=${1}
gpu=${2:-0}
num_iters=${3:-8000}
batch_size=${4:-64}
save_every=${5:-2000}   # save intermediate ckpt every N iters (0 = final only)
wandb_project=${6:-"diffusion-policy-pd"}  # wandb project name

if [ -z "$task" ]; then
  echo "Usage: bash scripts/distill_all.sh <task> [gpu] [num_iters] [batch_size] [save_every]"
  echo "  task: lift | can | square | transport | tool_hang"
  exit 1
fi

case "$task" in
  lift|can|square|transport|tool_hang) ;;
  *) echo "Unknown task: $task"; exit 1 ;;
esac

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CS570_ROOT=${CS570_ROOT:-$(dirname "$PROJECT_DIR")}
ROBOMIMIC_DIR=${ROBOMIMIC_DIR:-$CS570_ROOT/robomimic}
DISTILLER_DIR=$CS570_ROOT/diffusion_distiller

CKPT_ROOT="${PROJECT_DIR}/distill_ckpts/${task}"
HDF5="${ROBOMIMIC_DIR}/datasets/${task}/ph/$(ls "${ROBOMIMIC_DIR}/datasets/${task}/ph/" | grep low_dim | head -1)"

export CUDA_VISIBLE_DEVICES=$gpu
export MUJOCO_GL=${MUJOCO_GL:-egl}
export CS570_ROOT ROBOMIMIC_DIR

PYTHON="conda run -n robomimic_dp python"

echo "============================================"
echo "TASK        : $task"
echo "GPU         : $gpu"
echo "NUM_ITERS   : $num_iters  (per round)"
echo "BATCH_SIZE  : $batch_size"
echo "SAVE_EVERY  : $save_every"
echo "WANDB       : ${wandb_project:-disabled}"
echo "HDF5        : $HDF5"
echo "CKPT_ROOT   : $CKPT_ROOT"
echo "============================================"

# ── Step 0: Convert robomimic teacher checkpoint ──────────────────────────────
TEACHER_DIR="${CKPT_ROOT}/teacher_step100"
TEACHER_PT="${TEACHER_DIR}/teacher.pt"
mkdir -p "$TEACHER_DIR"

if [ -f "$TEACHER_PT" ]; then
  echo "[Step 0] Teacher already exists: $TEACHER_PT"
else
  echo "[Step 0] Converting robomimic checkpoint -> distiller format ..."
  $PYTHON "${DISTILLER_DIR}/convert_ckpt.py" \
    --robomimic_ckpt "${PROJECT_DIR}/ckpt_bundle/${task}/last.pth" \
    --out "$TEACHER_PT"
  echo "[Step 0] Done."
fi

# ── Progressive Distillation Rounds ──────────────────────────────────────────
# Compute the step schedule: 100 -> 50 -> 25 -> 12 -> 6 -> 3 -> 1
steps=(100)
n=100
while [ "$n" -gt 1 ]; do
  n=$(( n / 2 ))
  steps+=($n)
done
# steps = (100 50 25 12 6 3 1)

prev_pt="$TEACHER_PT"

for i in "${!steps[@]}"; do
  if [ "$i" -eq 0 ]; then
    continue   # skip 100 (that's the teacher)
  fi

  s="${steps[$i]}"
  s_fmt=$(printf "%03d" "$s")
  STUDENT_DIR="${CKPT_ROOT}/student_step${s_fmt}"
  STUDENT_PT="${STUDENT_DIR}/student.pt"
  mkdir -p "$STUDENT_DIR"

  echo ""
  echo "──────────────────────────────────────────"
  echo "[Round $i] ${steps[$((i-1))]} -> ${s} steps"
  echo "  teacher  : $prev_pt"
  echo "  out_ckpt : $STUDENT_PT"
  echo "──────────────────────────────────────────"

  if [ -f "$STUDENT_PT" ]; then
    echo "  Already exists, skipping."
    prev_pt="$STUDENT_PT"
    continue
  fi

  WANDB_ARG=""
  [ -n "$wandb_project" ] && WANDB_ARG="--wandb_project $wandb_project"

  $PYTHON "${PROJECT_DIR}/scripts/run_distill.py" \
    --teacher_ckpt "$prev_pt" \
    --out_ckpt     "$STUDENT_PT" \
    --hdf5_path    "$HDF5" \
    --num_iters    "$num_iters" \
    --batch_size   "$batch_size" \
    --save_every   "$save_every" \
    --num_workers  4 \
    $WANDB_ARG

  echo "[Round $i] Done -> $STUDENT_PT"
  prev_pt="$STUDENT_PT"
done

# ── Step Final: Convert 1-step student back to robomimic format for eval ──────
FINAL_PT="${CKPT_ROOT}/student_step001/student.pt"
EVAL_PTH="${CKPT_ROOT}/student_step001/student_robomimic.pth"

if [ -f "$FINAL_PT" ]; then
  echo ""
  echo "[Final] Converting 1-step student -> robomimic .pth for eval_one.sh ..."
  $PYTHON "${DISTILLER_DIR}/v_to_eps_wrapper.py" \
    --student_pt "$FINAL_PT" \
    --out_pth    "$EVAL_PTH"
  echo "[Final] eval_one.sh ready:"
  echo "  bash scripts/eval_one.sh $task $EVAL_PTH 50 $gpu"
fi

echo ""
echo "============================================"
echo "All distillation rounds complete for: $task"
echo "Checkpoints:"
find "$CKPT_ROOT" -name "*.pt" -o -name "*.pth" | sort
echo "============================================"
