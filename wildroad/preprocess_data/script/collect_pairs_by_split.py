#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect paired (rgb, graph.pickle) samples from processed data folders into
train/ and valid/ directories following a split JSON.

Input layout (source_dir):
  source_dir/
    data{i}_tiled_candidates/
      A/
        cropped_..._rgb.png
        cropped_..._graph.pickle
        ... (mask present but ignored)
      B/  (optional, included only with --include_B)
        ...

Behavior:
  - Read split indices from a JSON like:
      {"train": [0,1,...], "valid": [3,9,...]}
    Indices refer to the "i" in folder names data{i}_tiled_candidates.
  - Copy only matching pairs (same prefix) of rgb and graph files; ignore mask.
  - Rename sequentially within each split:
      train/data_{idx}.png, train/gt_graph_{idx}.pickle
      valid/data_{idx}.png, valid/gt_graph_{idx}.pickle
    where idx starts at 0 separately for each split.

Usage example:
  python script/collect_pairs_by_split.py \
    --source_dir processed \
    --split_json raw_data/data_split.json \
    --output_dir dataset_out \
    --include_B
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from typing import Dict, List, Tuple


DATA_DIR_PATTERN = re.compile(r"^data(\d+)_tiled_candidates$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect (rgb,pickle) pairs into train/valid according to split JSON"
    )
    parser.add_argument(
        "--source_dir",
        required=True,
        help="Directory containing multiple data{i}_tiled_candidates folders",
    )
    parser.add_argument(
        "--split_json",
        required=True,
        help="Path to data_split.json specifying train/valid dataset indices",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output root; train/ and valid/ will be created under this directory",
    )
    parser.add_argument(
        "--include_B",
        action="store_true",
        help="If set, copy from both A and B; otherwise only A",
    )
    return parser.parse_args()


def find_dataset_dirs(source_dir: str) -> Dict[int, str]:
    """Map dataset index i -> absolute folder path for data{i}_tiled_candidates.

    Only include directories matching the expected naming pattern.
    """
    mapping: Dict[int, str] = {}
    try:
        for name in os.listdir(source_dir):
            full = os.path.join(source_dir, name)
            if not os.path.isdir(full):
                continue
            m = DATA_DIR_PATTERN.match(name)
            if not m:
                continue
            idx = int(m.group(1))
            mapping[idx] = full
    except FileNotFoundError:
        pass
    return mapping


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def iter_pairs_in_subdir(subdir: str) -> List[Tuple[str, str]]:
    """Return list of (rgb_path, graph_path) pairs found in a given directory.

    Strategy: scan for *_graph.pickle, and for each such file ensure that a matching
    *_rgb.png exists with the same prefix. Mask files are ignored.
    """
    pairs: List[Tuple[str, str]] = []
    if not os.path.isdir(subdir):
        return pairs

    try:
        for fname in os.listdir(subdir):
            if not fname.endswith("_graph.pickle"):
                continue
            prefix = fname[: -len("_graph.pickle")]
            rgb_name = prefix + "_rgb.png"
            graph_path = os.path.join(subdir, fname)
            rgb_path = os.path.join(subdir, rgb_name)
            if os.path.exists(rgb_path):
                pairs.append((rgb_path, graph_path))
            else:
                # Optional: log missing rgb for visibility
                print(f"[WARN] Missing RGB for graph: {graph_path}")
    except FileNotFoundError:
        pass
    return pairs


def copy_pairs(
    pairs: List[Tuple[str, str]],
    dest_dir: str,
    start_idx: int,
) -> int:
    """Copy (rgb,graph) pairs into dest_dir with sequential names starting at start_idx.

    Returns the next index after the last written item.
    """
    ensure_dir(dest_dir)
    idx = start_idx
    for rgb_path, graph_path in pairs:
        dst_rgb = os.path.join(dest_dir, f"data_{idx}.png")
        dst_graph = os.path.join(dest_dir, f"gt_graph_{idx}.pickle")
        shutil.copy2(rgb_path, dst_rgb)
        shutil.copy2(graph_path, dst_graph)
        idx += 1
    return idx


def collect_for_split(
    split_name: str,
    indices: List[int],
    id2path: Dict[int, str],
    include_B: bool,
    output_root: str,
) -> int:
    """Collect samples for a specific split (train or valid).

    Returns the total number of copied pairs for this split.
    """
    out_dir = os.path.join(output_root, split_name)
    ensure_dir(out_dir)
    next_idx = 0
    copied = 0

    for i in indices:
        dpath = id2path.get(i)
        if not dpath:
            print(f"[WARN] Missing dataset directory for index {i}; expected data{i}_tiled_candidates under source_dir")
            continue
        # Always copy A; optionally also B
        a_dir = os.path.join(dpath, "A")
        b_dir = os.path.join(dpath, "B")

        a_pairs = iter_pairs_in_subdir(a_dir)
        if a_pairs:
            next_idx_after_a = copy_pairs(a_pairs, out_dir, next_idx)
            copied += next_idx_after_a - next_idx
            next_idx = next_idx_after_a
        else:
            print(f"[INFO] No pairs found in {a_dir}")

        if include_B:
            b_pairs = iter_pairs_in_subdir(b_dir)
            if b_pairs:
                next_idx_after_b = copy_pairs(b_pairs, out_dir, next_idx)
                copied += next_idx_after_b - next_idx
                next_idx = next_idx_after_b
            else:
                print(f"[INFO] No pairs found in {b_dir}")

    print(f"{split_name}: copied {copied} pairs -> {out_dir}")
    return copied


def main() -> int:
    args = parse_args()

    source_dir = os.path.abspath(args.source_dir)
    split_json_path = os.path.abspath(args.split_json)
    output_dir = os.path.abspath(args.output_dir)

    # Build index -> path mapping
    id2path = find_dataset_dirs(source_dir)
    if not id2path:
        print(f"[ERR] No data{{i}}_tiled_candidates directories found under: {source_dir}")
        return 1

    # Load split JSON
    try:
        with open(split_json_path, "r", encoding="utf-8") as f:
            split = json.load(f)
    except Exception as e:
        print(f"[ERR] Failed to read split JSON: {split_json_path} | {e}")
        return 1

    train_ids: List[int] = list(split.get("train", []))
    valid_ids: List[int] = list(split.get("valid", []))

    # Create output root and subfolders
    ensure_dir(output_dir)

    # Process splits separately; indices restart from 0 in each split
    train_count = collect_for_split(
        split_name="train",
        indices=train_ids,
        id2path=id2path,
        include_B=args.include_B,
        output_root=output_dir,
    )
    valid_count = collect_for_split(
        split_name="valid",
        indices=valid_ids,
        id2path=id2path,
        include_B=args.include_B,
        output_root=output_dir,
    )

    total = train_count + valid_count
    print(f"\n✓ Done. Total copied pairs: {total} | train={train_count} | valid={valid_count}")
    print(f"Output structure: {os.path.join(output_dir, 'train')} , {os.path.join(output_dir, 'valid')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


