#!/usr/bin/env bash
# Distillation + Eval for all 5 tasks
# Usage: bash scripts/run_all_tasks.sh [gpu] [num_iters]
set -e

gpu=${1:-0}
num_iters=${2:-20000}
batch_size=64
save_every=5000   # save checkpoint every 5k iters in its own subfolder
wandb_project="diffusion-policy-pd"

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CS570_ROOT=${CS570_ROOT:-$(dirname "$PROJECT_DIR")}
ROBOMIMIC_DIR=${ROBOMIMIC_DIR:-$CS570_ROOT/robomimic}
PYTHON="conda run -n robomimic_dp python"

export CUDA_VISIBLE_DEVICES=$gpu
export MUJOCO_GL=${MUJOCO_GL:-egl}
export MUJOCO_EGL_DEVICE_ID=0
export CS570_ROOT ROBOMIMIC_DIR

TASKS="lift can square transport tool_hang"

echo "============================================"
echo "Running distillation + eval for all tasks"
echo "GPU=$gpu  NUM_ITERS=$num_iters  WANDB=$wandb_project"
echo "============================================"

# ── Step 1: Distillation for all tasks ───────────────────────────────────────
for task in $TASKS; do
    echo ""
    echo "##############################"
    echo "# DISTILL: $task"
    echo "##############################"
    bash "$PROJECT_DIR/scripts/distill_all.sh" \
        "$task" "$gpu" "$num_iters" "$batch_size" "$save_every" "$wandb_project"
done

# ── Step 2: Eval all steps for all tasks ──────────────────────────────────────
RESULT_FILE="$PROJECT_DIR/outputs/distill_eval_results.txt"
mkdir -p "$PROJECT_DIR/outputs"
echo "task,steps,success_rate,num_success" > "$RESULT_FILE"

declare -A HORIZONS
HORIZONS[lift]=400; HORIZONS[can]=400; HORIZONS[square]=400
HORIZONS[transport]=700; HORIZONS[tool_hang]=700

for task in $TASKS; do
    echo ""
    echo "##############################"
    echo "# EVAL: $task"
    echo "##############################"
    CKPT_ROOT="$PROJECT_DIR/distill_ckpts/$task"
    ORIG_CKPT="$PROJECT_DIR/ckpt_bundle/$task/last.pth"
    horizon=${HORIZONS[$task]}

    for step_dir in "$CKPT_ROOT"/student_step*/; do
        step=$(basename "$step_dir" | sed 's/student_step//')
        student_pt="$step_dir/student.pt"
        [ -f "$student_pt" ] || continue

        echo ""
        echo "  -- $task / step${step} --"
        result=$($PYTHON "$PROJECT_DIR/scripts/eval_student.py" \
            --student_pt "$student_pt" \
            --orig_ckpt  "$ORIG_CKPT" \
            --n_rollouts 50 \
            --horizon    "$horizon" \
            --seed 0 \
            2>&1 | grep -E "\"Success_Rate\"|\"Num_Success\"")

        sr=$(echo "$result"  | grep Success_Rate | grep -oP '[\d.]+')
        ns=$(echo "$result"  | grep Num_Success  | grep -oP '[\d.]+')
        echo "    $task step${step}: SR=$sr ($ns/50)"
        echo "$task,$step,$sr,$ns" >> "$RESULT_FILE"
    done
done

# ── Step 3: Print summary table ───────────────────────────────────────────────
echo ""
echo "============================================"
echo "FINAL RESULTS"
echo "============================================"
printf "%-12s %6s %6s %6s %6s %6s %6s\n" "task" "050" "025" "012" "006" "003" "001"
echo "------------------------------------------------------------"
for task in $TASKS; do
    row="$(printf '%-12s' $task)"
    for step in 050 025 012 006 003 001; do
        sr=$(grep "^${task},${step}," "$RESULT_FILE" | cut -d',' -f3)
        sr=${sr:-"N/A"}
        row="$row $(printf '%6s' $sr)"
    done
    echo "$row"
done
echo ""
echo "Full results saved to: $RESULT_FILE"
