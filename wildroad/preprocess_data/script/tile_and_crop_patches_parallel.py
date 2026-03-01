#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tile and crop big RGB/graph into A (canonical grid) and B (overlap) candidates
using GraphPatchCropper, and save results to disk with manifests.

Outputs:
- <output_dir>/A_candidates/ cropped_*_{rgb|mask|graph}
- <output_dir>/B_candidates/ ...
- <output_dir>/A_candidates.csv, B_candidates.csv
- <output_dir>/meta.json (image size, patch size, overlaps)

This script ONLY generates candidates. Selection will be done by select_by_wl_similarity.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

# Ensure we can import sibling modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from crop_patch_from_pickle_parallel import ParallelGraphPatchCropper


PatchRect = Tuple[int, int, int, int]


def generate_A_patches(W: int, H: int, patch_size: int) -> List[PatchRect]:
    patches: List[PatchRect] = []

    # 横向坐标（保持原有方式）
    x_coords = list(range(0, W - patch_size + 1, patch_size))
    if x_coords[-1] + patch_size < W:  # 如果最后一个patch没覆盖到右边界
        x_coords.append(W - patch_size)

    # 纵向坐标（均匀分布，确保覆盖上下边界）
    if H <= patch_size:
        y_coords = [0]
    else:
        n_patches_y = max(1, round(H / patch_size))
        if n_patches_y == 1:
            y_coords = [0]
        else:
            step = (H - patch_size) / (n_patches_y - 1)
            y_coords = [round(step * i) for i in range(n_patches_y)]

    # 生成 patch
    for top in y_coords:
        for left in x_coords:
            patches.append((
                left,
                top,
                left + patch_size,
                top + patch_size
            ))

    return patches


def generate_B_patches(W: int, H: int, patch_size: int, overlaps: List[int]) -> List[PatchRect]:
    seen: Set[PatchRect] = set()
    patches: List[PatchRect] = []
    for m in overlaps:
        stride = max(1, patch_size - m)
        top = 0
        while top + patch_size <= H:
            left = 0
            while left + patch_size <= W:
                rect = (left, top, left + patch_size, top + patch_size)
                if rect not in seen:
                    seen.add(rect)
                    patches.append(rect)
                left += stride
            top += stride
    return patches


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tile and crop A/B candidates using GraphPatchCropper")
    p.add_argument("rgb_path", help="Path to big RGB image")
    p.add_argument("graph_path", help="Path to big graph pickle (adjacency dict)")
    p.add_argument("--output", "-o", default="./tiled_candidates", help="Output directory")
    p.add_argument("--patch_size", type=int, default=1024, help="Patch size")
    p.add_argument("--overlaps", type=int, nargs="*", default=[512, 256], help="Overlap m values for B (stride=patch_size-m)")
    # parameters for GraphPatchCropper (mask thickness etc.)
    p.add_argument("--edge_width", type=int, default=6, help="Road raster width for mask")
    p.add_argument("--inner_offset", type=int, default=5, help="Inner offset for boundary points")
    p.add_argument("--min_edge_ratio", type=float, default=0.05, help="Min edge length ratio in cropper")
    p.add_argument("--workers", type=int, default=4, help="Thread workers for parallel cropping")
    p.add_argument("--seed", type=int, default=42, help="Random seed (not used here, reserved for determinism)")
    return p.parse_args()


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def crop_and_save_set(cropper: ParallelGraphPatchCropper, patches: List[PatchRect], out_dir: str, workers: int) -> None:
    ensure_dir(out_dir)
    # Run true multithreading with thread pool
    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_patch = {}
        for (left, top, right, bottom) in patches:
            patch = [left, top, right, bottom]
            fut = ex.submit(cropper.crop_patch, patch)
            future_to_patch[fut] = patch

        total = len(future_to_patch)
        done_count = 0
        for fut in as_completed(future_to_patch):
            patch = future_to_patch[fut]
            try:
                cropped_image, adjacency, road_mask = fut.result()
                cropper.save_results(out_dir, patch, cropped_image, adjacency, road_mask)
            except Exception as e:
                print(f"Skip patch {patch} due to error: {e}")
            finally:
                done_count += 1
                if done_count % 20 == 0 or done_count == total:
                    print(f"Completed {done_count}/{total} patches -> {out_dir}")


def write_manifest_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    args = parse_args()

    # image size
    with Image.open(args.rgb_path) as im:
        W, H = im.size

    out_root = args.output
    dir_A = os.path.join(out_root, "A_candidates")
    dir_B = os.path.join(out_root, "B_candidates")
    ensure_dir(dir_A)
    ensure_dir(dir_B)

    # Build cropper once
    # cropper = GraphPatchCropper(
    #     args.rgb_path,
    #     args.graph_path,
    #     edge_width=args.edge_width,
    #     inner_offset=args.inner_offset,
    #     min_edge_length_ratio=args.min_edge_ratio,
    # )
    cropper = ParallelGraphPatchCropper(
        args.rgb_path,
        args.graph_path,
        edge_width=args.edge_width,
        inner_offset=args.inner_offset,
        min_edge_length_ratio=args.min_edge_ratio,
    )

    # Generate patches
    A_rects = generate_A_patches(W, H, args.patch_size)
    B_rects = generate_B_patches(W, H, args.patch_size, args.overlaps)
    start_time = time.time()
    # Crop and save
    print(f"Cropping A candidates: {len(A_rects)} with {args.workers} workers")
    crop_and_save_set(cropper, A_rects, dir_A, args.workers)
    print(f"Cropping B candidates: {len(B_rects)} with {args.workers} workers")
    crop_and_save_set(cropper, B_rects, dir_B, args.workers)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    # Manifests
    A_rows: List[Dict[str, object]] = []
    for (left, top, right, bottom) in A_rects:
        A_rows.append({
            "kind": "A",
            "left": left, "top": top, "right": right, "bottom": bottom,
            "row": top // args.patch_size, "col": left // args.patch_size,
        })
    B_rows: List[Dict[str, object]] = []
    for (left, top, right, bottom) in B_rects:
        B_rows.append({
            "kind": "B",
            "left": left, "top": top, "right": right, "bottom": bottom,
        })
    write_manifest_csv(os.path.join(out_root, "A_candidates.csv"), A_rows)
    write_manifest_csv(os.path.join(out_root, "B_candidates.csv"), B_rows)

    # Meta
    meta = {
        "image_width": W,
        "image_height": H,
        "patch_size": args.patch_size,
        "overlaps": args.overlaps,
        "edge_width": args.edge_width,
        "inner_offset": args.inner_offset,
        "min_edge_ratio": args.min_edge_ratio,
    }
    with open(os.path.join(out_root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✓ Done. A: {len(A_rects)} candidates -> {dir_A}; B: {len(B_rects)} candidates -> {dir_B}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



