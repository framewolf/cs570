#!/usr/bin/env bash
set -e

CS570_ROOT=${CS570_ROOT:-$HOME/cs570}
ROBOMIMIC_DIR=${ROBOMIMIC_DIR:-$CS570_ROOT/robomimic}

cd "$ROBOMIMIC_DIR"

python robomimic/scripts/download_datasets.py \
  --tasks sim \
  --dataset_types ph \
  --hdf5_types low_dim

echo ""
echo "Downloaded datasets:"
find datasets -path "*/ph/*low_dim*.hdf5" | sort
