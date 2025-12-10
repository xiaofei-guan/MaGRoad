# APLS & TOPO Parallel Computation

This script computes APLS (Average Path Length Similarity) and TOPO (Topology) metrics for road network graph predictions in parallel.

## Initialize Go Environment

```bash
# Navigte to the optimized APLS directory
cd metrics/optimized_apls/

# Initialize Go module (first time only)
go mod init apls

# Install required dependencies
go get github.com/dhconnelly/rtreego@latest
```

## Usage

### Basic Command

```bash
./apls_topo_parallel.sh <gt-dir> <pred-dir> <result-dir> <n-parallel-apls> <n-parallel-topo>
```

### Parameters

- **gt-dir**: Ground truth graph pickle directory
- **pred-dir**: Parent directory containing `graph `folder
- **result-dir**: Output directory for results
- **n-parallel-apls**: Number of parallel processes for APLS (e.g., 8, 16, 32)
- **n-parallel-topo**: Number of parallel processes for TOPO (e.g., 16, 36, 48)

*Note: set `n-parallel-apls` and `n-parallel-topo` to ≤ number of CPU cores*

### Example

```bash
./apls_topo_parallel.sh \
    test_data/gt_graph \
    test_data/predictions \
    exp/results \
    16 \
    36
```