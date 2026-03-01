#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch pipeline runner for raw_data.

Scans raw_data for data{i} image/pickle pairs and processes each pair:
1) Tiling + cropping candidates into processed/data{i}_tiled_candidates
2) Filtering A by length density and selecting B by WL similarity

Directory layout (expected):
project/
  - raw_data/
      - data{i}.(png|jpg|jpeg|tif|tiff|bmp)
      - data{i}*.pickle (e.g., data_i.pickle or data_i_gt_graph.pickle)
  - script/
      - crop_patch_from_pickle.py
      - tile_and_crop_patches.py
      - tile_and_crop_patches_parallel.py
      - select_by_wl_similarity.py
      - topology_similarity.py
      - process_all_datasets.py  ← this script
  - processed/  (output directory, same level as script/)

Outputs under: processed/data{i}_tiled_candidates/
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import subprocess
from typing import Dict, List, Optional, Tuple


IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def is_image(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in IMG_EXTS


def find_pairs(raw_root: str) -> List[Tuple[str, str, str]]:
    """Find (data_id, image_path, graph_path) pairs under raw_root.

    - data_id is the leading basename like 'data0' or 'data_0' captured as 'data<number>'
    - image must be one of IMG_EXTS
    - graph is a .pickle file starting with the same data_id prefix
    """
    files = [f for f in os.listdir(raw_root) if os.path.isfile(os.path.join(raw_root, f))]
    images_by_id: Dict[str, str] = {}
    graphs_by_id: Dict[str, str] = {}

    pat = re.compile(r"^(data\d+)")

    for fname in files:
        m = pat.match(fname)
        if not m:
            continue
        data_id = m.group(1)
        full = os.path.join(raw_root, fname)
        if is_image(full):
            # prefer first-found; if multiple, keep the earliest
            images_by_id.setdefault(data_id, full)
        elif fname.lower().endswith(".pickle"):
            # prefer *_gt_graph.pickle or anything starting with id; last write wins
            # but we store only one; prioritize *_gt_graph.pickle if both exist
            prev = graphs_by_id.get(data_id)
            if prev is None or fname.endswith("_gt_graph.pickle"):
                graphs_by_id[data_id] = full

    pairs: List[Tuple[str, str, str]] = []
    for data_id, img in images_by_id.items():
        g = graphs_by_id.get(data_id)
        if g:
            pairs.append((data_id, img, g))
    # Sort by numeric id within data_id
    def id_key(t: Tuple[str, str, str]) -> int:
        m = re.search(r"data(\d+)", t[0])
        return int(m.group(1)) if m else 0
    pairs.sort(key=id_key)
    return pairs


def run_cmd(cmd: List[str], cwd: Optional[str] = None) -> int:
    print("[RUN]", " ".join(cmd))
    try:
        res = subprocess.run(cmd, cwd=cwd, check=False)
        return res.returncode
    except Exception as e:
        print(f"[ERR] Failed to run: {' '.join(cmd)} | {e}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Process all data{i} pairs under raw_data")
    parser.add_argument("--raw_root", default=None, help="Path to raw_data root (default: ../raw_data relative to script dir)")
    parser.add_argument("--processed_dirname", default="processed", help="Name of output dir (default: ../processed relative to script dir)")
    # tiling params
    parser.add_argument("--patch_size", type=int, default=1024)
    parser.add_argument("--overlaps", type=int, nargs="*", default=[256, 384])
    parser.add_argument("--edge_width", type=int, default=6)
    parser.add_argument("--inner_offset", type=int, default=5)
    parser.add_argument("--min_edge_ratio", type=float, default=0.05)
    # selection params
    parser.add_argument("--a_density_min", type=float, default=0.001)
    parser.add_argument("--b_density_min", type=float, default=0.001)
    parser.add_argument("--wl_iterations", type=int, default=3)
    parser.add_argument("--sim_threshold", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=2025)
    # speed
    parser.add_argument("--use_parallel_tiler", action="store_true", help="Use tile_and_crop_patches_parallel.py if available")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # parent directory of script/
    raw_root = args.raw_root or os.path.join(project_root, "raw_data")
    processed_root = os.path.join(project_root, args.processed_dirname)
    os.makedirs(processed_root, exist_ok=True)

    # Resolve scripts
    tiler_parallel = os.path.join(script_dir, "tile_and_crop_patches_parallel.py")
    tiler_serial = os.path.join(script_dir, "tile_and_crop_patches.py")
    selector = os.path.join(script_dir, "select_by_wl_similarity.py")

    tiler = tiler_parallel if (args.use_parallel_tiler and os.path.exists(tiler_parallel)) else tiler_serial
    if not os.path.exists(tiler):
        print(f"[ERR] Tiling script not found: {tiler}")
        return 1
    if not os.path.exists(selector):
        print(f"[ERR] Selection script not found: {selector}")
        return 1

    pairs = find_pairs(raw_root)
    if not pairs:
        print(f"[WARN] No data<i> pairs found under: {raw_root}")
        return 0

    print(f"Discovered {len(pairs)} pairs under {raw_root}")

    for data_id, img_path, graph_path in pairs:
        out_dir = os.path.join(processed_root, f"{data_id}_tiled_candidates")
        os.makedirs(out_dir, exist_ok=True)
        img_path = os.path.join(project_root, img_path)
        graph_path = os.path.join(project_root, graph_path)
        print(f"\n=== Processing {data_id} ===")
        print(f"Image: {img_path}")
        print(f"Graph: {graph_path}")
        print(f"Output: {out_dir}")

        # 1) tiling and cropping
        tile_cmd = [
            sys.executable, tiler,
            img_path, graph_path,
            "--output", out_dir,
            "--patch_size", str(args.patch_size),
            "--overlaps", *[str(m) for m in args.overlaps],
            "--edge_width", str(args.edge_width),
            "--inner_offset", str(args.inner_offset),
            "--min_edge_ratio", str(args.min_edge_ratio),
        ]
        t0 = time.time()
        rc = run_cmd(tile_cmd, cwd=script_dir)
        if rc != 0:
            print(f"[ERR] Tiling failed for {data_id}, skipping selection.")
            continue
        t1 = time.time()
        print(f"Tiling done in {t1 - t0:.2f}s")

        # 2) selection by WL similarity
        sel_cmd = [
            sys.executable, selector,
            out_dir,
            "--a_density_min", str(args.a_density_min),
            "--b_density_min", str(args.b_density_min),
            "--wl_iterations", str(args.wl_iterations),
            "--sim_threshold", str(args.sim_threshold),
            "--seed", str(args.seed),
            "--debug_max", "0",  # concise
        ]
        t2 = time.time()
        rc = run_cmd(sel_cmd, cwd=script_dir)
        if rc != 0:
            print(f"[ERR] Selection failed for {data_id}")
            continue
        t3 = time.time()
        print(f"Selection done in {t3 - t2:.2f}s")
        print(f"Completed {data_id}")

    print("\n✓ All datasets processed. See:")
    print(f"  {processed_root}/*_tiled_candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


