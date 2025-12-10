# APLS Metrics

This directory contains tools for computing Average Path Length Similarity (APLS) metrics to evaluate graph topology accuracy.

## Quick Start

### Option 1: Batch Processing (Recommended)

Use the Python script for automated batch processing:

```bash
# Use optimized version (faster)
python compute_apls.py \
  --gt-dir test_data/20cities_test_gt_graph \
  --prop-dir test_data/magToponetgraph \
  --work-dir runs/apls_optimized \
  --go-script optimized_main.go

# Process specific regions only
python compute_apls.py \
  --gt-dir test_data/20cities_test_gt_graph \
  --prop-dir test_data/magToponetgraph \
  --work-dir runs/apls_subset \
  --ids 108,109,119
```

### Option 2: Manual Processing

For individual files or custom workflows:

```bash
# 1. Initialize Go module (first time only)
cd apls/
go mod init apls
go get github.com/dhconnelly/rtreego@latest

# 2. Convert pickle files to JSON
python apls/convert.py path/to/gt.p gt.json
python apls/convert.py path/to/prop.p prop.json

# 3. Run APLS computation
go run apls/optimized_main.go gt.json prop.json result.txt
```

## Output Structure

The batch script creates organized outputs in your work directory:

```
work-dir/
├── json/               # Intermediate JSON files
│   ├── gt_8.json
│   ├── prop_8.json
│   └── ...
├── results/            # Per-region APLS results
│   ├── apls_8.txt
│   └── ...
├── summary.csv         # Tabular summary
└── apls.json          # Final statistics
```

## File Formats

### Input
- **GT files**: `region_<id>_*_gt_*.{p,pickle}` (ground truth graphs)
- **Proposal files**: `<id>.p` (predicted graphs)

### Output
- **Individual results**: `apls_<id>.txt` contains three values: `apls_gt apls_prop final_apls`
- **Summary CSV**: All results in tabular format
- **Statistics JSON**: Final average and per-region values

## Parameters

### Tile Size Parameters
- **Default**: 2048×2048 meter tiles
- **Small tiles**: Use `--small-tiles` flag for 352×352 meter tiles
