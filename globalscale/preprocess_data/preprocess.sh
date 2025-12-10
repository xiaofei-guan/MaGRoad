#!/usr/bin/env bash
set -e

# Usage: bash preprocess.sh [INPUT_DATA_DIR] [OUTPUT_GLG_DIR] [OUTPUT_MASK_DIR] [NUM_JOBS]
# Example: bash preprocess.sh ./globalscale ./globalscale_GLG ./globalscale_mask 16

# Default paths and settings for GlobalScale
INPUT_DATA_DIR=${1:-"./Globalscale"}
OUTPUT_GLG_DIR=${2:-"./Globalscale_GLG"}
OUTPUT_MASK_DIR=${3:-"./Globalscale_mask"}
NUM_JOBS=${4:-16}


OUTPUT_GLG_RAW_DIR="${OUTPUT_GLG_DIR}_raw"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================================="
echo "Starting Preprocessing Pipeline for GlobalScale"
echo "Input Data Directory:  $INPUT_DATA_DIR"
echo "Output GLG (Final):    $OUTPUT_GLG_DIR"
echo "Output GLG (Raw):      $OUTPUT_GLG_RAW_DIR"
echo "Output Mask Directory: $OUTPUT_MASK_DIR"
echo "Parallel Jobs:         $NUM_JOBS"
echo "=========================================================="

# ------------------------------------------------------------------
# Stage 1: Compute GLG for patches (into RAW directory)
# ------------------------------------------------------------------
echo ""
echo "[Stage 1] Computing GLG..."
export NUM_JOBS="$NUM_JOBS"
# attention: here output to RAW directory
bash "$SCRIPT_DIR/compute_for_GLG.sh" "$INPUT_DATA_DIR" "$OUTPUT_GLG_RAW_DIR"

# ------------------------------------------------------------------
# Stage 2: Move and Aggregate Files (RAW -> Final)
# ------------------------------------------------------------------
echo ""
echo "[Stage 2] Aggregating GLG files..."
# move from RAW to Final
bash "$SCRIPT_DIR/mv.sh" "$OUTPUT_GLG_RAW_DIR" "$OUTPUT_GLG_DIR"

# ------------------------------------------------------------------
# Stage 3: Generate Masks
# ------------------------------------------------------------------
echo ""
echo "[Stage 3] Generating Masks..."

python3 "$SCRIPT_DIR/generate_masks_for_globalscale.py" \
  --input_dir "$INPUT_DATA_DIR" \
  --output_dir "$OUTPUT_MASK_DIR"
