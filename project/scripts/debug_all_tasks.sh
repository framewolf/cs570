#!/usr/bin/env bash
set -e

CS570_ROOT=${CS570_ROOT:-$HOME/cs570}
ROBOMIMIC_DIR=${ROBOMIMIC_DIR:-$CS570_ROOT/robomimic}
export MUJOCO_GL=${MUJOCO_GL:-egl}

TASKS=(lift can square transport tool_hang)

cd "$ROBOMIMIC_DIR"

for task in "${TASKS[@]}"; do
  dataset=$(ls datasets/${task}/ph/low_dim*.hdf5 | head -n 1)
  echo ""
  echo "============================================================"
  echo "DEBUG TASK: ${task}"
  echo "DATASET: ${dataset}"
  echo "============================================================"

  python robomimic/scripts/train.py \
    --config robomimic/exps/templates/diffusion_policy.json \
    --dataset "${dataset}" \
    --name "dp_debug_${task}" \
    --debug
done
