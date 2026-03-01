# WildRoad Dataset Processing Pipeline

**Note on Dataset Size:** The fully processed dataset is quite large. If downloading the processed patches is inconvenient, you can download the raw source data instead and use the scripts provided here to perfectly reproduce the dataset.

This repository contains scripts to process large-scale remote sensing images and their corresponding road network graphs into smaller, trainable patches.

## Overview

The processing pipeline employs two strategies to crop the large map into patches:
- **Strategy A (Non-overlapping):** Crops the image using a regular, non-overlapping grid.
- **Strategy B (Overlapping):** Crops the image using an overlapping sliding window.

**Filtering & De-duplication:**
All patches must meet a minimum road length density threshold. To avoid data redundancy, the pipeline uses the Weisfeiler-Lehman (WL) topological similarity algorithm. If an overlapping patch (Strategy B) contains a road topology that is too similar to the existing non-overlapping patches (Strategy A), it will be discarded. Otherwise, it will be kept as a valuable topological supplement.

## Folder Structure

Before running the script, ensure your raw data is organized into split folders. Inside each folder, images (`.jpg`) and their corresponding graph files (`.pickle`) should be paired by name (e.g., `data0.jpg` and `data0.pickle`).

```text
Project Root/
├── script/
│   ├── process_single_split.py
│   ├── crop_patch_from_pickle_parallel.py
│   └── ...
├── train/
├── val/
└── test/
```

## How It Works

The main entry point is `process_single_split.py`. When you run it on a target folder (e.g., `test`), the script will:
1. Find all image-graph pairs in the folder.
2. Parallelly crop the large data into candidates and save them temporarily in `{split}_processed/`.
3. Filter redundant topological patches using WL similarity.
4. Collect the final valid patches into `{split}_patches/` directory.

The output will contain two subdirectories for each split:
- `{split}_A`: Contains strictly non-overlapping patches.
- `{split}_AB`: Contains both non-overlapping and selected overlapping patches.

*Note: Only the RGB image and the graph data are kept in the final output. The debug masks are ignored to save disk space.*

## Usage

Process each split sequentially by running the following commands from the **Project Root**:

```bash
# 1. Process the training set
python script/process_single_split.py train --workers 4

# 2. Process the validation set
python script/process_single_split.py val --workers 4

# 3. Process the test set
python script/process_single_split.py test --workers 4
```

*Optional Arguments:*
- `--workers`: Number of parallel threads to speed up the cropping process (default is 4).
- `--patch_size`: Output patch size (default is 1024).
- `--sim_threshold`: WL similarity threshold to discard redundant B patches (default is 0.7).

## Verification (Expected Patch Counts)

After running the processing scripts, you can verify your results by checking the number of generated patches. The expected counts of data pairs (image + graph) for each split are as follows:

| Split | Strategy A (`_A`) | Strategy A+B (`_AB`) |
|-------|-------------------|----------------------|
| train | 5566              | 12896                |
| val   | 1306              | 2986                 |
| test  | 1146              | 2666                 |