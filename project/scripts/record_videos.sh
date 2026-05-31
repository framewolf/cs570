#!/usr/bin/env bash
# Record rollout videos for each task × step level
# Saves to project/test/{task}/step{NNN}.mp4
#
# Usage: bash scripts/record_videos.sh [gpu] [n_rollouts_per_video]
#   gpu              : CUDA device id (default 0)
#   n_rollouts_per_video : rollouts to record per video (default 3)

set -e

gpu=${1:-0}
n_rollouts=${2:-3}

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON="conda run -n robomimic_dp python"

export CUDA_VISIBLE_DEVICES=$gpu
export MUJOCO_GL=${MUJOCO_GL:-egl}
export MUJOCO_EGL_DEVICE_ID=0

TASKS="lift can square transport tool_hang"
STEPS="001 003 006 012 025 050"

declare -A HORIZONS
HORIZONS[lift]=400; HORIZONS[can]=400; HORIZONS[square]=400
HORIZONS[transport]=700; HORIZONS[tool_hang]=700

echo "============================================"
echo "Recording rollout videos"
echo "GPU=$gpu  N_ROLLOUTS=$n_rollouts"
echo "Output: $PROJECT_DIR/distill_ckpts/{task}/student_step{NNN}/rollout.mp4"
echo "============================================"

for task in $TASKS; do
    ORIG_CKPT="$PROJECT_DIR/ckpt_bundle/$task/last.pth"
    horizon=${HORIZONS[$task]}

    for step in $STEPS; do
        ckpt_dir="$PROJECT_DIR/distill_ckpts/$task/student_step${step}"
        student_pt="$ckpt_dir/student.pt"
        video_path="$ckpt_dir/rollout.mp4"

        [ -f "$student_pt" ] || { echo "  SKIP $task/step$step (no checkpoint)"; continue; }
        [ -f "$video_path" ] && { echo "  SKIP $task/step$step (video exists)"; continue; }

        echo ""
        echo "  >> $task / step${step}"
        $PYTHON "$PROJECT_DIR/scripts/eval_student.py" \
            --student_pt  "$student_pt" \
            --orig_ckpt   "$ORIG_CKPT" \
            --n_rollouts  "$n_rollouts" \
            --horizon     "$horizon" \
            --video_path  "$video_path" \
            --seed 42 \
            2>&1 | grep -v "^$" | grep -v "^Created\|^Action\|^number of\|^Environment\|^Loaded\|^Device\|^Using\|^obs modality\|^normaliz\|^[Ii]nit"
        echo "  Saved: $video_path"
    done
done

echo ""
echo "============================================"
echo "All videos saved."
echo "============================================"
echo ""
echo "Directory structure:"
find "$PROJECT_DIR/distill_ckpts" -name "rollout.mp4" | sort | while read f; do
    sz=$(du -h "$f" | cut -f1)
    echo "  $sz  $f"
done
