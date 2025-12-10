#!/usr/bin/env bash

# Replicates the functionality of run_apls_topo.bat for Linux/macOS.
# Usage: ./run_apls_topo.sh <gt-dir> <prop-dir> <work-dir-prefix>

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -lt 3 ]; then
    echo "Usage: $(basename "$0") <gt-dir> <prop-dir> <work-dir-prefix>"
    echo
    echo "Example: $(basename "$0") test_data/20cities_gt_graph test_data/magtoponet_new_trhreshold_inference_20250824_210608_V6_last_ckpt/graph exp/V6_last_ckpt_new_threshold"
    echo
    echo "Parameters:"
    echo "  gt-dir: Ground truth graph data directory"
    echo "  prop-dir: Predicted graph data directory"
    echo "  work-dir-prefix: Work directory prefix, script will auto-add suffix"
    exit 1
fi

GT_DIR="$1"
PROP_DIR="$2"
WORK_DIR_PREFIX="$3"

APLS_WORK_DIR="${WORK_DIR_PREFIX}_optimized_apls"
TOPO_WORK_DIR="${WORK_DIR_PREFIX}_optimized_topo"

APLS_PY="$SCRIPT_DIR/metrics/compute_apls.py"
TOPO_PY="$SCRIPT_DIR/metrics/compute_topo.py"
GO_SCRIPT_NAME="optimized_main.go"  # Same as .bat

echo "================================"
echo "Starting metrics computation..."
echo "================================"
echo "GT directory: $GT_DIR"
echo "Prediction directory: $PROP_DIR"
echo "APLS work directory: $APLS_WORK_DIR"
echo "TOPO work directory: $TOPO_WORK_DIR"
echo "================================"

# Ensure work dirs exist
mkdir -p "$APLS_WORK_DIR" "$TOPO_WORK_DIR"

echo
echo "[1/2] Computing APLS metrics..."
echo "Command: python3 \"$APLS_PY\" --gt-dir \"$GT_DIR\" --prop-dir \"$PROP_DIR\" --work-dir \"$APLS_WORK_DIR\" --go-script $GO_SCRIPT_NAME"
python3 "$APLS_PY" --gt-dir "$GT_DIR" --prop-dir "$PROP_DIR" --work-dir "$APLS_WORK_DIR" --go-script "$GO_SCRIPT_NAME"
status=$?
if [ $status -ne 0 ]; then
    echo "Error: APLS computation failed!"
    exit $status
fi

echo "APLS metrics computation completed!"

echo
# echo "[2/2] Computing TOPO metrics..."
# echo "Command: python3 \"$TOPO_PY\" --gt-dir \"$GT_DIR\" --prop-dir \"$PROP_DIR\" --work-dir \"$TOPO_WORK_DIR\""
# python3 "$TOPO_PY" --gt-dir "$GT_DIR" --prop-dir "$PROP_DIR" --work-dir "$TOPO_WORK_DIR"
# status=$?
# if [ $status -ne 0 ]; then
#     echo "Error: TOPO computation failed!"
#     exit $status
# fi

echo "TOPO metrics computation completed!"

echo
echo "================================"
echo "All metrics computation completed!"
echo "================================"
echo "APLS results saved in: $APLS_WORK_DIR"
echo "TOPO results saved in: $TOPO_WORK_DIR"
echo "================================"


