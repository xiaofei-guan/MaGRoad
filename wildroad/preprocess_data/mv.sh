#!/usr/bin/env bash
set -euo pipefail

# This script leverages mv_files.py (move_tile_folders + verify_move_result)
# to merge tile_* folders from GLG_* subdirectories into all_* aggregation dirs.

# Usage:
#   bash mv.sh [BASE_INPUT_DIR] [BASE_OUTPUT_DIR]
# Defaults:
#   BASE_INPUT_DIR=./wildroad_GLG
#   BASE_OUTPUT_DIR=./wildroad_GLG

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_EXE=${PYTHON_EXE:-python3}

BASE_INPUT_DIR=${1:-./wildroad_GLG}
BASE_OUTPUT_DIR=${2:-./wildroad_GLG}

splits=(train val test)
kinds=(A AB)

echo "[INFO] Base input:  $BASE_INPUT_DIR"
echo "[INFO] Base output: $BASE_OUTPUT_DIR"

for split in "${splits[@]}"; do
  for kind in "${kinds[@]}"; do
    in_dir="$BASE_INPUT_DIR/${split}_patches/${split}_${kind}_GLG"
    out_dir="$BASE_OUTPUT_DIR/${split}_patches/all_${split}_${kind}_GLG"

    echo
    echo "===================="
    echo "[PROCESS] $in_dir -> $out_dir"
    echo "===================="

    if [ ! -d "$in_dir" ]; then
      echo "[WARN] Input directory not found: $in_dir (skip)"
      continue
    fi

    mkdir -p "$out_dir"

    IN_DIR="$in_dir" OUT_DIR="$out_dir" "$PYTHON_EXE" - <<'PY'
import os
import sys
from mv_files import move_tile_folders, verify_move_result

source = os.environ['IN_DIR']
output = os.environ['OUT_DIR']

print(f"[PY] moving from {source} to {output}")
move_tile_folders(source, output)
verify_move_result(output)
PY

  done
done

echo
echo "[DONE] All merges completed."

