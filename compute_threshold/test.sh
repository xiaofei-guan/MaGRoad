#!/bin/bash

# Script to run testing and compute optimal thresholds for road segmentation model
# Usage: ./test.sh --config <config_path> --checkpoint <checkpoint_path> [--precision <precision>]

# Parse command line arguments
CONFIG=""
CHECKPOINT=""
PRECISION="16-mixed"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --config <config_path> --checkpoint <checkpoint_path> [--precision <precision>]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$CONFIG" ] || [ -z "$CHECKPOINT" ]; then
    echo "Error: --config and --checkpoint are required arguments"
    echo "Usage: $0 --config <config_path> --checkpoint <checkpoint_path> [--precision <precision>]"
    exit 1
fi

echo "=========================================="
echo "Starting testing and threshold computation"
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Precision: $PRECISION"
echo "=========================================="

# Step 1: Run model testing
echo ""
echo "[Step 1/3] Running model testing..."
python compute_threshold/test.py --config "$CONFIG" --checkpoint "$CHECKPOINT" --precision "$PRECISION"

if [ $? -ne 0 ]; then
    echo "Error: Model testing failed"
    exit 1
fi

echo "[Step 1/3] Model testing completed successfully"

# Step 2: Compute best threshold for keypoint (kp)
echo ""
echo "[Step 2/3] Computing best threshold for keypoint (kp)..."
python compute_threshold/compute_best_threshold.py --target kp

if [ $? -ne 0 ]; then
    echo "Error: Failed to compute threshold for keypoint"
    exit 1
fi

echo "[Step 2/3] Keypoint threshold computed successfully"

# Step 3: Compute best threshold for road
echo ""
echo "[Step 3/3] Computing best threshold for road..."
python compute_threshold/compute_best_threshold.py --target road

if [ $? -ne 0 ]; then
    echo "Error: Failed to compute threshold for road"
    exit 1
fi

echo "[Step 3/3] Road threshold computed successfully"

# Complete
echo ""
echo "=========================================="
echo "All steps completed successfully!"
echo "Three thresholds have been computed:"
echo "  1. Model test threshold"
echo "  2. Keypoint (kp) threshold"
echo "  3. Road threshold"
echo "=========================================="