#!/usr/bin/env bash

# Parallel APLS + TOPO computation for a SINGLE prediction result
# Usage: ./apls_topo_parallel.sh <gt-dir> <pred-dir> <result-dir> <n-parallel-apls> <n-parallel-topo>
#
# Structure:
#   gt-dir: Ground truth pickle files directory
#   pred-dir: Specific prediction folder containing a "graph" subdirectory with pickle files
#   result-dir: Base directory where results will be saved
#   n-parallel-apls: Number of parallel processes for APLS computation
#   n-parallel-topo: Number of parallel processes for TOPO computation
#
# Example:
#   ./apls_topo_parallel.sh \
#     test_data/20cities_gt_graph \
#     test_data/predictions/magtoponet_v1 \
#     exp/results \
#     16 \
#     36


set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -lt 5 ]; then
    echo "Usage: $(basename "$0") <gt-dir> <pred-dir> <result-dir> <n-parallel-apls> <n-parallel-topo>"
    echo
    echo "Parameters:"
    echo "  gt-dir:           Ground truth graph pickle directory"
    echo "  pred-dir:         Specific prediction folder (must contain 'graph' subdir)"
    echo "  result-dir:       Base directory where results will be saved"
    echo "  n-parallel-apls:  Number of parallel processes for APLS (e.g., 8, 16, 32)"
    echo "  n-parallel-topo:  Number of parallel processes for TOPO (e.g., 16, 36, 48)"
    echo
    echo "Notes:"
    echo "  - Both APLS and TOPO use parallel computation with random chunk distribution"
    echo "  - APLS uses main.go (non-optimized), TOPO uses optimized version"
    echo "  - Recommend n-parallel <= number of CPU cores for best performance"
    exit 1
fi

GT_DIR="$1"
PRED_DIR="$2"
RESULT_DIR="$3"
N_PARALLEL_APLS="$4"
N_PARALLEL_TOPO="$5"

# Validate parallel parameters are positive integers
if ! [[ "$N_PARALLEL_APLS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: n-parallel-apls must be a positive integer (got: $N_PARALLEL_APLS)"
    exit 1
fi

if ! [[ "$N_PARALLEL_TOPO" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: n-parallel-topo must be a positive integer (got: $N_PARALLEL_TOPO)"
    exit 1
fi

# Script paths
APLS_PY="$SCRIPT_DIR/metrics/compute_apls_parallel.py" # faster with minimal difference in results
# TOPO_PY="$SCRIPT_DIR/metrics/compute_topo_parallel.py" # use optimized TOPO implementation
TOPO_PY="$SCRIPT_DIR/metrics/original_compute_topo_parallel.py" # use original TOPO implementation

echo "========================================"
echo "Parallel APLS + TOPO Computation"
echo "========================================"
echo "GT directory: $GT_DIR"
echo "Prediction directory: $PRED_DIR"
echo "Result directory: $RESULT_DIR"
echo "APLS parallel processes: $N_PARALLEL_APLS"
echo "TOPO parallel processes: $N_PARALLEL_TOPO"
echo "========================================"

# Validate GT directory
if [ ! -d "$GT_DIR" ]; then
    echo "Error: GT directory does not exist: $GT_DIR"
    exit 1
fi

# Validate prediction directory
if [ ! -d "$PRED_DIR" ]; then
    echo "Error: Prediction directory does not exist: $PRED_DIR"
    exit 1
fi

# Validate graph subdirectory
pred_graph_dir="$PRED_DIR/graph"
if [ ! -d "$pred_graph_dir" ]; then
    echo "Error: Prediction graph directory does not exist: $pred_graph_dir"
    echo "Expected 'graph' subdirectory inside prediction folder."
    exit 1
fi

# Validate required scripts exist
if [ ! -f "$APLS_PY" ]; then
    echo "Error: Parallel APLS computation script not found: $APLS_PY"
    exit 1
fi

if [ ! -f "$TOPO_PY" ]; then
    echo "Error: Parallel TOPO computation script not found: $TOPO_PY"
    exit 1
fi

# Create result directory
mkdir -p "$RESULT_DIR"

pred_name=$(basename "$PRED_DIR")
work_prefix="$RESULT_DIR/$pred_name"

apls_work_dir="${work_prefix}_apls_parallel"
topo_work_dir="${work_prefix}_optimized_topo_parallel"

echo "Prediction name: $pred_name"
echo "Prediction graph directory: $pred_graph_dir"
echo "APLS work directory: $apls_work_dir"
echo "TOPO work directory: $topo_work_dir"
echo

# Create work directories
mkdir -p "$apls_work_dir" "$topo_work_dir"

# ========================================
# Step 1: Compute APLS (Parallel)
# ========================================
echo "----------------------------------------"
echo "Step 1: Computing APLS for $pred_name (parallel=$N_PARALLEL_APLS)"
echo "----------------------------------------"

apls_start=$(date +%s)

python3 "$APLS_PY" \
    --gt-dir "$GT_DIR" \
    --prop-dir "$pred_graph_dir" \
    --work-dir "$apls_work_dir" \
    --n-parallel "$N_PARALLEL_APLS" \
    --go-script main.go

apls_status=$?
apls_end=$(date +%s)
apls_elapsed=$((apls_end - apls_start))

if [ $apls_status -ne 0 ]; then
    echo "Error: APLS computation failed for $pred_name!"
    echo "Aborting TOPO computation."
    exit 1
fi

echo "APLS computation completed in ${apls_elapsed} seconds"

# Display APLS result if available
apls_json="$apls_work_dir/apls.json"
if [ -f "$apls_json" ]; then
    if command -v python3 &> /dev/null; then
        final_apls=$(python3 -c "import json; print(json.load(open('$apls_json'))['final_APLS'])" 2>/dev/null || echo "N/A")
        echo "Final APLS: $final_apls"
    fi
fi

echo

# ========================================
# Step 2: Compute TOPO (Parallel)
# ========================================
echo "----------------------------------------"
echo "Step 2: Computing TOPO for $pred_name (parallel=$N_PARALLEL_TOPO)"
echo "----------------------------------------"

topo_start=$(date +%s)

python3 "$TOPO_PY" \
    --gt-dir "$GT_DIR" \
    --prop-dir "$pred_graph_dir" \
    --work-dir "$topo_work_dir" \
    --n-parallel "$N_PARALLEL_TOPO" \
    --workers 1

topo_status=$?
topo_end=$(date +%s)
topo_elapsed=$((topo_end - topo_start))

if [ $topo_status -ne 0 ]; then
    echo "Error: TOPO computation failed for $pred_name!"
    exit 1
fi

echo "TOPO computation completed in ${topo_elapsed} seconds"

# Display TOPO result if available
topo_json="$topo_work_dir/topo.json"
if [ -f "$topo_json" ]; then
    if command -v python3 &> /dev/null; then
        mean_topo=$(python3 -c "import json; data=json.load(open('$topo_json')); print('F1={:.6f}, P={:.6f}, R={:.6f}'.format(*data['mean topo']))" 2>/dev/null || echo "N/A")
        echo "Mean TOPO: $mean_topo"
    fi
fi

echo
echo "========================================"
echo "COMPUTATION COMPLETED"
echo "========================================"
echo "APLS time: ${apls_elapsed}s, TOPO time: ${topo_elapsed}s, Total: $((apls_elapsed + topo_elapsed))s"
echo "Results saved in:"
echo "  $RESULT_DIR"
echo
