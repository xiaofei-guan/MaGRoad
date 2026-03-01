#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select filtered A and B patches using length density and WL topology similarity.

Input (produced by tile_and_crop_patches.py):
- <root>/A_candidates/ cropped_*_{rgb.png,mask.png,graph.pickle}
- <root>/B_candidates/ ...
- <root>/A_candidates.csv, <root>/B_candidates.csv, <root>/meta.json

Output:
- <root>/A/ (filtered A)
- <root>/B/ (selected B after WL-sim NMS)
- <root>/selected_manifest.csv

Key rules:
- A: filter by length density >= a_density_min
- B: filter by length density >= b_density_min, sort by density desc
- Neighborhood: for a B patch, find its most-overlapping A cell (row,col),
  take its 8-neighborhood (9 cells), construct the minimal bounding rectangle (MBR)
  of these A cells, and compare only with Keep patches whose rect is fully contained in MBR.
- Similarity: WL histogram similarity (1-JS) over cropped graphs loaded from graph.pickle
- Caching: WL hist per patch; pairwise similarity computed from hist directly
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from topology_similarity import build_networkx_graph, compute_wl_histograms, wl_similarity_from_histograms, compute_length_density


PatchRect = Tuple[int, int, int, int]


@dataclass
class PatchInfo:
    kind: str  # 'A' or 'B'
    left: int
    top: int
    right: int
    bottom: int
    row: int = -1  # only for A; for B we will infer the best-overlap A cell
    col: int = -1
    density: float = 0.0
    hist: Optional[List[Dict[str, int]]] = None
    graph_path: Optional[str] = None

    def rect(self) -> PatchRect:
        return (self.left, self.top, self.right, self.bottom)


def load_meta(root: str) -> Dict:
    with open(os.path.join(root, "meta.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_candidates_csv(path: str, kind: str) -> List[PatchInfo]:
    rows: List[PatchInfo] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            left = int(r["left"])
            top = int(r["top"])
            right = int(r["right"])
            bottom = int(r["bottom"])
            row = int(r.get("row", -1)) if "row" in r else -1
            col = int(r.get("col", -1)) if "col" in r else -1
            rows.append(PatchInfo(kind=kind, left=left, top=top, right=right, bottom=bottom, row=row, col=col))
    return rows


def graph_path_for_rect(dir_path: str, rect: PatchRect) -> Optional[str]:
    """Return path to cropped graph file for a rect."""
    left, top, right, bottom = rect
    pattern = os.path.join(dir_path, f"cropped_{left}_{top}_{right}_{bottom}_graph.pickle")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def load_graph_adjacency(graph_path: str) -> Dict[Tuple[float, float], List[Tuple[float, float]]]:
    import pickle
    with open(graph_path, "rb") as f:
        g = pickle.load(f)
    if not isinstance(g, dict):
        return {}
    return g


def rect_iou(a: PatchRect, b: PatchRect) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / (area_a + area_b - inter)


def best_overlap_A_cell(b: PatchInfo, A: List[PatchInfo], patch_size: int) -> Tuple[int, int]:
    # A cells are on canonical grid; find A patch with max IoU
    best_iou = -1.0
    best_rowcol = (-1, -1)
    b_rect = b.rect()
    for a in A:
        iou = rect_iou(b_rect, a.rect())
        if iou > best_iou:
            best_iou = iou
            best_rowcol = (a.row, a.col)
    return best_rowcol


def neighborhood_mbr(row: int, col: int, grid_rows: int, grid_cols: int, A: List[PatchInfo], patch_size: int) -> PatchRect:
    """Compute MBR of the 3x3 A-neighborhood around (row,col) using actual A patch coordinates.

    Rationale: A patches may not be strictly contiguous; we must derive the bounding box
    from real saved patch rectangles rather than assuming a perfect grid.

    Fallback: if no A patch exists in the neighborhood (should not happen),
    approximate by the ideal grid-based rectangle.
    """
    # neighborhood index ranges
    r0 = max(0, row - 1)
    r1 = min(grid_rows - 1, row + 1)
    c0 = max(0, col - 1)
    c1 = min(grid_cols - 1, col + 1)

    # collect actual A patches within 3x3 neighborhood
    neighbors = [a for a in A if (r0 <= a.row <= r1) and (c0 <= a.col <= c1)]
    if neighbors:
        left = min(a.left for a in neighbors)
        top = min(a.top for a in neighbors)
        right = max(a.right for a in neighbors)
        bottom = max(a.bottom for a in neighbors)
        return (left, top, right, bottom)

    # fallback to grid-based bounding box
    left = c0 * patch_size
    top = r0 * patch_size
    right = (c1 + 1) * patch_size
    bottom = (r1 + 1) * patch_size
    return (left, top, right, bottom)


def rect_contains(outer: PatchRect, inner: PatchRect) -> bool:
    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = inner
    return (ix0 >= ox0) and (iy0 >= oy0) and (ix1 <= ox1) and (iy1 <= oy1)


def main() -> int:
    p = argparse.ArgumentParser(description="Filter A by density, select B by WL similarity (neighborhood-contained)")
    p.add_argument("root", help="Root directory produced by tile_and_crop_patches.py")
    p.add_argument("--a_density_min", type=float, default=0.01, help="Min length density for A")
    p.add_argument("--b_density_min", type=float, default=0.005, help="Min length density for B")
    p.add_argument("--sim_threshold", type=float, default=0.8, help="WL similarity threshold (>= drop)")
    p.add_argument("--wl_iterations", type=int, default=3, help="WL iterations for histogram")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--debug", action="store_true", help="Print detailed similarity stats for tuning threshold")
    p.add_argument("--debug_max", type=int, default=20, help="Max number of B patches to print detailed stats for")
    p.add_argument("--print_topk", type=int, default=3, help="Top-k similarities to print in debug mode")
    p.add_argument("--log_every", type=int, default=0, help="If >0, log every k-th B record in debug mode")
    args = p.parse_args()

    np.random.seed(args.seed)

    meta = load_meta(args.root)
    patch_size = int(meta["patch_size"])
    W = int(meta["image_width"])
    H = int(meta["image_height"])
    grid_cols = (W - patch_size) // patch_size + 1
    grid_rows = (H - patch_size) // patch_size + 1

    A_dir = os.path.join(args.root, "A_candidates")
    B_dir = os.path.join(args.root, "B_candidates")

    A_list = load_candidates_csv(os.path.join(args.root, "A_candidates.csv"), kind="A")
    B_list = load_candidates_csv(os.path.join(args.root, "B_candidates.csv"), kind="B")

    if args.debug:
        print(f"Config | a_density_min={args.a_density_min} | b_density_min={args.b_density_min} | sim_threshold={args.sim_threshold} | wl_iter={args.wl_iterations} | seed={args.seed}")

    # Attach graph path and compute density for A
    for a in A_list:
        a.graph_path = graph_path_for_rect(A_dir, a.rect())
        if not a.graph_path:
            a.density = 0.0
            continue
        adj = load_graph_adjacency(a.graph_path)
        a.density = compute_length_density(adj, patch_size)

    A_keep = [a for a in A_list if a.density >= args.a_density_min and a.graph_path]
    if args.debug:
        print(f"A stats | total={len(A_list)} | with_graph={sum(1 for x in A_list if x.graph_path)} | kept={len(A_keep)} (>= {args.a_density_min})")

    # Prepare Keep directory
    out_A = os.path.join(args.root, "A")
    out_B = os.path.join(args.root, "B")
    os.makedirs(out_A, exist_ok=True)
    os.makedirs(out_B, exist_ok=True)

    # Copy files for A_keep
    import shutil
    for a in A_keep:
        base = f"cropped_{a.left}_{a.top}_{a.right}_{a.bottom}"
        # RGB
        src = os.path.join(A_dir, base + "_rgb.png")
        if os.path.exists(src):
            shutil.copy2(src, out_A)
        # MASK
        src = os.path.join(A_dir, base + "_mask.png")
        if os.path.exists(src):
            shutil.copy2(src, out_A)
        # GRAPH
        src = os.path.join(A_dir, base + "_graph.pickle")
        if os.path.exists(src):
            shutil.copy2(src, out_A)

    # WL hist cache for patches in Keep
    for a in A_keep:
        if a.graph_path:
            adj = load_graph_adjacency(a.graph_path)
            G = build_networkx_graph(adj)
            a.hist = compute_wl_histograms(G, iterations=args.wl_iterations)

    # B: compute density and sort by density desc
    for b in B_list:
        b.graph_path = graph_path_for_rect(B_dir, b.rect())
        if not b.graph_path:
            b.density = 0.0
            continue
        adj = load_graph_adjacency(b.graph_path)
        b.density = compute_length_density(adj, patch_size)

    B_cand = [b for b in B_list if b.density >= args.b_density_min and b.graph_path]
    B_cand.sort(key=lambda x: (-x.density, x.top, x.left))
    if args.debug:
        print(f"B stats | total={len(B_list)} | with_graph={sum(1 for x in B_list if x.graph_path)} | kept_by_density={len(B_cand)} (>= {args.b_density_min})")

    # Selection with neighborhood-contained filter
    Keep: List[PatchInfo] = list(A_keep)
    max_sims_all: List[float] = []
    accepted_B = 0
    dropped_B = 0

    for idx_b, b in enumerate(B_cand):
        # find best-overlap A cell
        r_best, c_best = best_overlap_A_cell(b, A_list, patch_size)
        if r_best < 0 or c_best < 0:
            # fallback: accept if no A cells (degenerate case)
            # but still require WL compared to A_keep if any Keep exists in full image
            pass
        # neighborhood MBR based on actual A patch rectangles (fully-contained filter will use this)
        mbr = neighborhood_mbr(r_best, c_best, grid_rows, grid_cols, A_list, patch_size)

        # fully-contained Keep patches inside MBR
        contenders = [k for k in Keep if rect_contains(mbr, k.rect())]
        if not contenders:
            # accept directly
            # compute and cache WL hist for b
            adj_b = load_graph_adjacency(b.graph_path)
            G_b = build_networkx_graph(adj_b)
            b.hist = compute_wl_histograms(G_b, iterations=args.wl_iterations)
            Keep.append(b)
            accepted_B += 1
            max_sims_all.append(0.0)
            if args.debug and (idx_b < args.debug_max or (args.log_every and (idx_b % args.log_every == 0))):
                print(f"B[{idx_b}] ACCEPT (no contenders) | rect={b.rect()} | density={b.density:.6f}")
            continue

        # prepare b hist
        if b.hist is None:
            adj_b = load_graph_adjacency(b.graph_path)
            G_b = build_networkx_graph(adj_b)
            b.hist = compute_wl_histograms(G_b, iterations=args.wl_iterations)

        # compute max similarity
        max_sim = 0.0
        sims_debug: List[Tuple[PatchRect, float]] = []
        for k in contenders:
            if k.hist is None and k.graph_path:
                adj_k = load_graph_adjacency(k.graph_path)
                G_k = build_networkx_graph(adj_k)
                k.hist = compute_wl_histograms(G_k, iterations=args.wl_iterations)
            if k.hist is None:
                continue
            sim = wl_similarity_from_histograms(b.hist, k.hist)
            if args.debug:
                sims_debug.append((k.rect(), sim))
            if sim > max_sim:
                max_sim = sim
            if not args.debug and max_sim >= args.sim_threshold:
                break

        max_sims_all.append(max_sim)
        if max_sim >= args.sim_threshold:
            # Too similar, drop
            dropped_B += 1
            if args.debug and (idx_b < args.debug_max or (args.log_every and (idx_b % args.log_every == 0))):
                sims_debug.sort(key=lambda x: x[1], reverse=True)
                topk = sims_debug[:args.print_topk]
                print(f"B[{idx_b}] DROP | max_sim={max_sim:.4f} >= thr={args.sim_threshold} | rect={b.rect()} | density={b.density:.6f}")
                print(f"  contenders={len(contenders)} | MBR={mbr} | bestA=({r_best},{c_best}) | top{args.print_topk} sims:")
                for rect_k, s in topk:
                    print(f"    k_rect={rect_k} | sim={s:.4f}")
            continue
        else:
            Keep.append(b)
            accepted_B += 1
            if args.debug and (idx_b < args.debug_max or (args.log_every and (idx_b % args.log_every == 0))):
                sims_debug.sort(key=lambda x: x[1], reverse=True)
                topk = sims_debug[:args.print_topk]
                print(f"B[{idx_b}] ACCEPT | max_sim={max_sim:.4f} < thr={args.sim_threshold} | rect={b.rect()} | density={b.density:.6f}")
                print(f"  contenders={len(contenders)} | MBR={mbr} | bestA=({r_best},{c_best}) | top{args.print_topk} sims:")
                for rect_k, s in topk:
                    print(f"    k_rect={rect_k} | sim={s:.4f}")

    # Save selected to A/ and B/ already copied A; now copy B
    for k in Keep:
        if k.kind != 'B':
            continue
        base = f"cropped_{k.left}_{k.top}_{k.right}_{k.bottom}"
        src_dir = B_dir
        # RGB
        src = os.path.join(src_dir, base + "_rgb.png")
        if os.path.exists(src):
            shutil.copy2(src, out_B)
        # MASK
        src = os.path.join(src_dir, base + "_mask.png")
        if os.path.exists(src):
            shutil.copy2(src, out_B)
        # GRAPH
        src = os.path.join(src_dir, base + "_graph.pickle")
        if os.path.exists(src):
            shutil.copy2(src, out_B)

    # Write manifest
    manifest = os.path.join(args.root, "selected_manifest.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "left", "top", "right", "bottom", "density"]) 
        for k in Keep:
            w.writerow([k.kind, k.left, k.top, k.right, k.bottom, f"{k.density:.8f}"])

    print(f"✓ Selected: total={len(Keep)}, A={len([x for x in Keep if x.kind=='A'])}, B={len([x for x in Keep if x.kind=='B'])}")
    print(f"A dir: {out_A}\nB dir: {out_B}\nManifest: {manifest}")
    if max_sims_all:
        arr = np.array(max_sims_all, dtype=np.float64)
        qs = np.quantile(arr, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        print("B max_sim summary:")
        print(f"  processed={len(max_sims_all)} | accepted={accepted_B} | dropped={dropped_B}")
        print(f"  quantiles: min={qs[0]:.4f} p10={qs[1]:.4f} p25={qs[2]:.4f} median={qs[3]:.4f} p75={qs[4]:.4f} p90={qs[5]:.4f} max={qs[6]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


