#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行版路网裁剪脚本（不修改原有实现文件，作为独立脚本提供）

目标：
- 保持与原有 `GraphPatchCropper` 一致的裁剪逻辑与输出（rgb、mask、graph），
  但在实现层面做两点优化：
  1) 避免在构图流程中产生孤立节点，从而无需在每个patch末尾再次做“孤立点过滤”开销；
  2) 使用多线程对多个patch并行裁剪与保存（CPU核较多时可加速）。

说明：
- 为保证线程安全，本脚本实现一个“无共享可变状态”的裁剪器，所有中间结果都在局部变量中构建，
  不复用实例字段，避免不同patch之间的写入冲突。
- PIL裁剪操作加锁，以回避潜在的线程不安全读。
"""

import argparse
import math
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from shapely.geometry import LineString, box
from skimage.draw import line as bresenham_line


Point = Tuple[float, float]
PatchRect = List[int]  # [left, top, right, bottom]
Adjacency = Dict[Point, List[Point]]


def is_in_patch(point: Point, patch: PatchRect) -> bool:
    row, col = point
    return (patch[0] <= col <= patch[2]) and (patch[1] <= row <= patch[3])


def calculate_intersection(point1: Point, point2: Point, patch: PatchRect) -> List[Tuple[float, float, str]]:
    l = LineString([(point1[1], point1[0]), (point2[1], point2[0])])
    patch_box = box(patch[0], patch[1], patch[2], patch[3])
    if not l.intersects(patch_box.boundary):
        return []

    left_bound = LineString([(patch[0], patch[1]), (patch[0], patch[3])])
    right_bound = LineString([(patch[2], patch[1]), (patch[2], patch[3])])
    top_bound = LineString([(patch[0], patch[1]), (patch[2], patch[1])])
    bottom_bound = LineString([(patch[0], patch[3]), (patch[2], patch[3])])
    boundaries = [
        (left_bound, 'left'),
        (right_bound, 'right'),
        (top_bound, 'top'),
        (bottom_bound, 'bottom'),
    ]

    intersection_points: List[Tuple[float, float, str]] = []
    for boundary, btype in boundaries:
        if l.intersects(boundary):
            inter = l.intersection(boundary)
            if not inter.is_empty:
                if inter.geom_type == 'Point':
                    intersection_points.append((float(inter.y), float(inter.x), btype))
                elif inter.geom_type == 'MultiPoint':
                    for p in inter.geoms:
                        intersection_points.append((float(p.y), float(p.x), btype))
    return intersection_points


def create_offset_point(intersection: Tuple[float, float, str], inner_offset: float) -> Point:
    row, col, btype = intersection
    if btype == 'left':
        return (float(row), float(col + inner_offset))
    if btype == 'right':
        return (float(row), float(col - inner_offset))
    if btype == 'top':
        return (float(row + inner_offset), float(col))
    if btype == 'bottom':
        return (float(row - inner_offset), float(col))
    return (float(row), float(col))


def euclidean_distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class ParallelGraphPatchCropper:
    def __init__(self, rgb_path: str, graph_path: str, edge_width: int = 4, inner_offset: int = 5,
                 min_edge_length_ratio: float = 0.08) -> None:
        self.rgb_path = rgb_path
        self.graph_path = graph_path
        self.edge_width = int(edge_width)
        self.inner_offset = float(inner_offset)
        self.min_edge_length_ratio = float(min_edge_length_ratio)

        # Load once, share for threads (read-only)
        self.rgb_image = Image.open(rgb_path)
        with open(graph_path, 'rb') as f:
            self.graph: Adjacency = pickle.load(f)

        # PIL image read lock
        self._pil_lock = threading.Lock()

    def crop_patch(self, patch: PatchRect):
        # Local containers; do not touch instance fields (thread-safe)
        left, top, right, bottom = patch
        width, height = right - left, bottom - top

        # Crop RGB safely
        with self._pil_lock:
            cropped_image = self.rgb_image.crop((left, top, right, bottom))

        # Build edges first, then adjacency, to avoid isolated nodes
        edges: List[Tuple[Point, Point]] = []
        boundary_points: List[Tuple[Point, Point]] = []  # (boundary_new_point_rel, neighbor_rel)

        # 1) nodes inside patch
        for node, neighbors in self.graph.items():
            if not is_in_patch(node, patch):
                continue
            node_rel = (float(node[0] - top), float(node[1] - left))
            for nb in neighbors:
                if is_in_patch(nb, patch):
                    nb_rel = (float(nb[0] - top), float(nb[1] - left))
                    edges.append((node_rel, nb_rel))
                else:
                    # one inside, one outside → find intersection; ensure min length
                    intersections = calculate_intersection(node, nb, patch)
                    if not intersections:
                        continue
                    inter = intersections[0]
                    if euclidean_distance(node, (inter[0], inter[1])) < self.min_edge_length_ratio * width:
                        continue
                    new_pt = create_offset_point(inter, self.inner_offset)
                    new_pt_rel = (float(new_pt[0] - top), float(new_pt[1] - left))
                    edges.append((node_rel, new_pt_rel))
                    boundary_points.append((new_pt_rel, node_rel))

        # 2) both endpoints outside but crossing patch
        processed_edges = set()
        for node, neighbors in self.graph.items():
            if is_in_patch(node, patch):
                continue
            for nb in neighbors:
                if is_in_patch(nb, patch):
                    continue
                key = tuple(sorted([node, nb]))
                if key in processed_edges:
                    continue
                processed_edges.add(key)

                inters = calculate_intersection(node, nb, patch)
                if len(inters) < 2:
                    continue
                int1, int2 = inters[:2]
                dist = euclidean_distance((int1[0], int1[1]), (int2[0], int2[1]))
                if dist < self.min_edge_length_ratio * width:
                    continue
                p1 = create_offset_point(int1, self.inner_offset)
                p2 = create_offset_point(int2, self.inner_offset)
                p1_rel = (float(p1[0] - top), float(p1[1] - left))
                p2_rel = (float(p2[0] - top), float(p2[1] - left))
                edges.append((p1_rel, p2_rel))
                boundary_points.append((p1_rel, p2_rel))
                boundary_points.append((p2_rel, p1_rel))

        # Build adjacency from unique undirected edges
        adjacency: Adjacency = {}
        seen = set()
        for a, b in edges:
            if a == b:
                continue
            key = (a, b) if a <= b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)

        # Create road mask
        road_mask = np.zeros((height, width), dtype=np.uint8)
        for a, nbrs in adjacency.items():
            for b in nbrs:
                r0, c0 = map(int, a)
                r1, c1 = map(int, b)
                rr, cc = bresenham_line(r0, c0, r1, c1)
                if self.edge_width > 1:
                    y = np.clip(rr, 0, road_mask.shape[0] - 1)
                    x = np.clip(cc, 0, road_mask.shape[1] - 1)
                    half = int(self.edge_width // 2)
                    for dy in range(-half, half + 1):
                        for dx in range(-half, half + 1):
                            ny = np.clip(y + dy, 0, road_mask.shape[0] - 1)
                            nx = np.clip(x + dx, 0, road_mask.shape[1] - 1)
                            road_mask[ny, nx] = 1
                else:
                    rr = np.clip(rr, 0, road_mask.shape[0] - 1)
                    cc = np.clip(cc, 0, road_mask.shape[1] - 1)
                    road_mask[rr, cc] = 1

        # 返回全部局部结果
        return cropped_image, adjacency, road_mask

    @staticmethod
    def save_results(output_dir: str, patch: PatchRect, cropped_image: Image.Image,
                     adjacency: Adjacency, road_mask: np.ndarray) -> None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        name_prefix = f"cropped_{patch[0]}_{patch[1]}_{patch[2]}_{patch[3]}"
        cropped_image.save(os.path.join(output_dir, name_prefix + '_rgb.png'))
        Image.fromarray(road_mask * 255).save(os.path.join(output_dir, name_prefix + '_mask.png'))
        with open(os.path.join(output_dir, name_prefix + '_graph.pickle'), 'wb') as f:
            pickle.dump(adjacency, f)


def generate_patches(left: int, top: int, right: int, bottom: int, patch_size: int,
                     num_patches_rows: Optional[int] = None, num_patches_cols: Optional[int] = None) -> List[PatchRect]:
    patches: List[PatchRect] = []
    region_width = right - left
    region_height = bottom - top
    if num_patches_rows is None:
        num_patches_rows = (region_height + patch_size - 1) // patch_size
    if num_patches_cols is None:
        num_patches_cols = (region_width + patch_size - 1) // patch_size
    row_stride = 0 if num_patches_rows == 1 else (region_height - patch_size) / (num_patches_rows - 1)
    col_stride = 0 if num_patches_cols == 1 else (region_width - patch_size) / (num_patches_cols - 1)
    for i in range(num_patches_rows):
        for j in range(num_patches_cols):
            patch_left = min(int(left + j * col_stride), right - patch_size)
            patch_top = min(int(top + i * row_stride), bottom - patch_size)
            patch_right = min(patch_left + patch_size, right)
            patch_bottom = min(patch_top + patch_size, bottom)
            patches.append([patch_left, patch_top, patch_right, patch_bottom])
    return patches


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='并行裁剪路网与RGB图像（保持原有逻辑，线程安全实现）')
    p.add_argument('rgb_path', help='RGB图像路径')
    p.add_argument('graph_path', help='路网图pickle路径')
    p.add_argument('--output', '-o', default='./output_parallel', help='输出目录')
    p.add_argument('--patch', nargs=4, type=int, help='剪裁区域 [left top right bottom]')
    p.add_argument('--region', nargs=4, type=int, help='生成patch的区域范围 [left top right bottom]')
    p.add_argument('--patch_size', type=int, help='每个patch的大小（正方形）')
    p.add_argument('--num_patches_rows', type=int, help='行方向上的patch数量')
    p.add_argument('--num_patches_cols', type=int, help='列方向上的patch数量')
    p.add_argument('--edge_width', type=int, default=4, help='道路宽度')
    p.add_argument('--inner_offset', type=int, default=5, help='边界点向内偏移距离')
    p.add_argument('--min_edge_ratio', type=float, default=0.08, help='最小边长度比例')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) - 1), help='线程数（默认CPU核数-1）')
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Prepare patch list
    patches: List[PatchRect] = []
    if args.patch:
        patches = [args.patch]
    elif args.patch_size and args.region:
        patches = generate_patches(args.region[0], args.region[1], args.region[2], args.region[3],
                                   args.patch_size, args.num_patches_rows, args.num_patches_cols)
        print(f"在区域 {args.region} 内生成了 {len(patches)} 个patch")
    elif args.patch_size:
        with Image.open(args.rgb_path) as im:
            W, H = im.size
        patches = generate_patches(0, 0, W, H, args.patch_size, args.num_patches_rows, args.num_patches_cols)
        print(f"在整张图像内生成了 {len(patches)} 个patch")
    else:
        with Image.open(args.rgb_path) as im:
            W, H = im.size
        patches = [[0, 0, W, H]]

    cropper = ParallelGraphPatchCropper(
        args.rgb_path,
        args.graph_path,
        edge_width=args.edge_width,
        inner_offset=args.inner_offset,
        min_edge_length_ratio=args.min_edge_ratio,
    )

    os.makedirs(args.output, exist_ok=True)

    # Parallel processing
    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for patch in patches:
            futures.append(ex.submit(cropper.crop_patch, patch))

        for i, fut in enumerate(as_completed(futures), 1):
            try:
                cropped_image, adjacency, road_mask = fut.result()
                # We need the patch that corresponds to this future: re-submit with index
                # Simplify: index mapping by position in list
                idx = futures.index(fut)
                patch = patches[idx]
                ParallelGraphPatchCropper.save_results(
                    args.output, patch, cropped_image, adjacency, road_mask)
            except Exception as e:
                print(f"处理patch时出错: {e}")
            if i % 20 == 0 or i == len(futures):
                print(f"已完成 {i}/{len(futures)} 个patch")

    print(f"✓ 处理完成，结果已保存到 {args.output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


