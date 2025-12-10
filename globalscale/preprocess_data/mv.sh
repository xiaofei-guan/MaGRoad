#!/usr/bin/env bash
set -euo pipefail

# This script moves tile folders from raw GLG directories to the final clean directory.

# Usage:
#   bash mv.sh [BASE_INPUT_DIR] [BASE_OUTPUT_DIR]
# Defaults:
#   BASE_INPUT_DIR=./globalscale_GLG_raw
#   BASE_OUTPUT_DIR=./globalscale_GLG

# get the raw input and output directories
RAW_INPUT_DIR=${1:-./globalscale_GLG_raw}
RAW_OUTPUT_DIR=${2:-./globalscale_GLG}

PYTHON_EXE=${PYTHON_EXE:-python3}

# CRITICAL FIX: Convert paths to absolute paths BEFORE changing directory
# Use python to resolve absolute paths reliably
BASE_INPUT_DIR=$("$PYTHON_EXE" -c "import os; print(os.path.abspath('$RAW_INPUT_DIR'))")
BASE_OUTPUT_DIR=$("$PYTHON_EXE" -c "import os; print(os.path.abspath('$RAW_OUTPUT_DIR'))")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

folders=(
  "train"
  "out_of_domain"
)

echo "[INFO] Base input (Raw):  $BASE_INPUT_DIR"
echo "[INFO] Base output (Final): $BASE_OUTPUT_DIR"

for folder in "${folders[@]}"; do
    in_dir="$BASE_INPUT_DIR/${folder}_GLG"
    out_dir="$BASE_OUTPUT_DIR/${folder}_GLG"

    echo
    echo "===================="
    echo "[PROCESS] $in_dir -> $out_dir"
    echo "===================="

    if [ ! -d "$in_dir" ]; then
      echo "[WARN] Input directory not found: $in_dir (skip)"
      continue
    fi

    mkdir -p "$out_dir"

    # Pass paths as environment variables to the python script
    IN_DIR="$in_dir" OUT_DIR="$out_dir" "$PYTHON_EXE" - <<'PY'
import os
import sys
# Now we are in SCRIPT_DIR, so we can import local modules
from mv_files import move_tile_folders, verify_move_result

source = os.environ['IN_DIR']
output = os.environ['OUT_DIR']

print(f"[PY] moving from {source} to {output}")
move_tile_folders(source, output)
verify_move_result(output)
PY

done

echo
echo "[DONE] All merges completed."
