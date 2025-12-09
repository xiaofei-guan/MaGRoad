#!/usr/bin/env bash
set -e

# This script coordinates the GLG precomputation (Stage 1) and file aggregation (Stage 2) and generate masks (Stage 3).
# It allows specifying input and output directories as arguments.

# Usage: bash preprocess.sh [INPUT_DATA_DIR] [OUTPUT_GLG_DIR] [OUTPUT_MASK_DIR] [NUM_JOBS]
# Example: bash preprocess.sh ./wildroad ./wildroad_GLG ./wildroad_mask 16

# Default paths and settings
INPUT_DATA_DIR=${1:-"./wildroad"}
OUTPUT_GLG_DIR=${2:-"./wildroad_GLG"}
OUTPUT_MASK_DIR=${3:-"./wildroad_mask"}
NUM_JOBS=${4:-16}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================================="
echo "Starting Preprocessing Pipeline"
echo "Input Data Directory:  $INPUT_DATA_DIR"
echo "Output GLG Directory:  $OUTPUT_GLG_DIR"
echo "Output Mask Directory: $OUTPUT_MASK_DIR"
echo "Parallel Jobs:         $NUM_JOBS"
echo "=========================================================="

# ------------------------------------------------------------------
# Stage 1: Compute GLG for patches
# ------------------------------------------------------------------
echo ""
echo "[Stage 1] Computing GLG..."
# Passes input/output directories to the computation script.
# Export NUM_JOBS so compute_for_GLG.sh can pick it up or pass it explicitly if supported.
export NUM_JOBS="$NUM_JOBS"
bash "$SCRIPT_DIR/compute_for_GLG.sh" "$INPUT_DATA_DIR" "$OUTPUT_GLG_DIR"

# ------------------------------------------------------------------
# Stage 2: Move and Aggregate Files
# ------------------------------------------------------------------
echo ""
echo "[Stage 2] Aggregating GLG files..."
bash "$SCRIPT_DIR/mv.sh" "$OUTPUT_GLG_DIR" "$OUTPUT_GLG_DIR"

# ------------------------------------------------------------------
# Stage 3: Generate Masks
# ------------------------------------------------------------------
echo ""
echo "[Stage 3] Generating Masks..."
python3 "$SCRIPT_DIR/generate_masks_for_wild_road.py" \
  --input_dir "$INPUT_DATA_DIR" \
  --output_dir "$OUTPUT_MASK_DIR"

echo ""
echo "=========================================================="
echo "All stages completed successfully."
echo "=========================================================="

