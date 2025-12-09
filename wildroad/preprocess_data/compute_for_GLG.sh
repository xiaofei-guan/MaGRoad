#!/usr/bin/env bash
set -euo pipefail

# Usage: bash compute_for_GLG.sh ./wildroad ./wildroad_GLG
# - First arg: base input dir containing train_patches, val_patches, test_patches
# - Second arg: base output dir to store computed GLG

BASE_INPUT_DIR=${1:-./wildroad}
BASE_OUTPUT_DIR=${2:-./wildroad_GLG}

PYTHON_EXE=${PYTHON_EXE:-python3}
DATASET_SCRIPT=${DATASET_SCRIPT:-"$(dirname "$0")/dataset.py"}
RUNNER_SCRIPT=${RUNNER_SCRIPT:-"$(dirname "$0")/run_precompute_parallel.py"}

NUM_JOBS=${NUM_JOBS:-24}

folders=(
  # "train_patches/train_A"
  "train_patches/train_AB"
  # "val_patches/val_A"
  "val_patches/val_AB"
  # "test_patches/test_A"
  "test_patches/test_AB"
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

  # Auto-detect total tiles by counting contiguous gt_graph_{i}.pickle
  # Pass -1 to runner to trigger auto-detection
  "$PYTHON_EXE" "$RUNNER_SCRIPT" \
    --num_jobs "$NUM_JOBS" \
    --dataset_name "wildroad" \
    --total_tiles -1 \
    --start_offset 0 \
    --base_output_dir "$out_dir" \
    --python_exe "$PYTHON_EXE" \
    --dataset_script "$DATASET_SCRIPT" \
    --input_dir "$in_dir"
done

echo "All submissions done. Check each GLG_* subfolder logs for progress."

