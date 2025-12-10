#!/usr/bin/env bash
set -euo pipefail

# Usage: bash compute_for_GLG.sh ./globalscale ./globalscale_GLG_raw
# - First arg: base input dir containing train, out_of_domain
# - Second arg: base output dir to store computed GLG (raw/intermediate)

BASE_INPUT_DIR=${1:-./globalscale}
BASE_OUTPUT_DIR=${2:-./globalscale_GLG_raw}

PYTHON_EXE=${PYTHON_EXE:-python3}
DATASET_SCRIPT=${DATASET_SCRIPT:-"$(dirname "$0")/dataset.py"}
RUNNER_SCRIPT=${RUNNER_SCRIPT:-"$(dirname "$0")/run_precompute_parallel.py"}

# adjust the number of jobs based on the available memory and CPU cores
NUM_JOBS=${NUM_JOBS:-16}

# GlobalScale only needs to process these two folders
folders=(
  "train"
  "out_of_domain"
)

mkdir -p "$BASE_OUTPUT_DIR"

for rel in "${folders[@]}"; do
  in_dir="$BASE_INPUT_DIR/$rel"
  out_dir="$BASE_OUTPUT_DIR/${rel}_GLG"
  mkdir -p "$out_dir"

  echo "[INFO] Processing $in_dir -> $out_dir"
  if [ ! -d "$in_dir" ]; then
    echo "[WARN] Skip: input directory not found: $in_dir"
    continue
  fi

  # Auto-detect total tiles
  "$PYTHON_EXE" "$RUNNER_SCRIPT" \
    --num_jobs "$NUM_JOBS" \
    --dataset_name "globalscale" \
    --total_tiles -1 \
    --start_offset 0 \
    --base_output_dir "$out_dir" \
    --python_exe "$PYTHON_EXE" \
    --dataset_script "$DATASET_SCRIPT" \
    --input_dir "$in_dir"
done

echo "All submissions done."
