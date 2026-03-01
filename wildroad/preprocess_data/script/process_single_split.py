#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立单流程处理脚本：处理指定目录(如 train/val/test)下的图片和路网文件
不依赖 json 分割文件，直接把处理好的 A 和 AB 结果归置到对应 _patches 文件夹。
同时，保证 mask 也能正确无误地提取并保存下来。
"""

import os
import re
import sys
import time
import shutil
import argparse
import subprocess
from typing import Dict, List, Tuple, Optional

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

def is_image(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in IMG_EXTS

def find_pairs(split_dir: str) -> List[Tuple[str, str, str]]:
    files = [f for f in os.listdir(split_dir) if os.path.isfile(os.path.join(split_dir, f))]
    images_by_id: Dict[str, str] = {}
    graphs_by_id: Dict[str, str] = {}

    pat = re.compile(r"^(data\d+)")

    for fname in files:
        m = pat.match(fname)
        if not m:
            continue
        data_id = m.group(1)
        full = os.path.join(split_dir, fname)
        if is_image(full):
            images_by_id.setdefault(data_id, full)
        elif fname.lower().endswith(".pickle"):
            prev = graphs_by_id.get(data_id)
            # 优先保留带有 _gt_graph 的后缀
            if prev is None or fname.endswith("_gt_graph.pickle"):
                graphs_by_id[data_id] = full

    pairs: List[Tuple[str, str, str]] = []
    for data_id, img in images_by_id.items():
        g = graphs_by_id.get(data_id)
        if g:
            pairs.append((data_id, img, g))

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

def collect_pairs_to_output(
    processed_root: str,
    output_dir: str,
    only_A: bool = False
) -> int:
    os.makedirs(output_dir, exist_ok=True)
    pat = re.compile(r"^data(\d+)_tiled_candidates$")
    folder_map: Dict[int, str] = {}

    for name in os.listdir(processed_root):
        full = os.path.join(processed_root, name)
        if not os.path.isdir(full):
            continue
        m = pat.match(name)
        if m:
            idx = int(m.group(1))
            folder_map[idx] = full

    global_idx = 0
    # 遍历所有被处理过的 data{i}
    for data_idx in sorted(folder_map.keys()):
        data_dir = folder_map[data_idx]
        
        # 要收集的子文件夹：只收集A 或者 收集A和B
        dirs_to_check = ["A"] if only_A else ["A", "B"]
        
        for d in dirs_to_check:
            sub_dir = os.path.join(data_dir, d)
            if not os.path.isdir(sub_dir):
                continue
            # 寻找 graph.pickle 作为锚点，提取 prefix
            for fname in os.listdir(sub_dir):
                if fname.endswith("_graph.pickle"):
                    prefix = fname[:-len("_graph.pickle")]
                    rgb_name = prefix + "_rgb.png"
                    
                    graph_path = os.path.join(sub_dir, fname)
                    rgb_path = os.path.join(sub_dir, rgb_name)
                    
                    if os.path.exists(rgb_path):
                        # 目标文件命名
                        dst_rgb = os.path.join(output_dir, f"data_{global_idx}.png")
                        dst_graph = os.path.join(output_dir, f"gt_graph_{global_idx}.pickle")
                        
                        shutil.copy2(rgb_path, dst_rgb)
                        shutil.copy2(graph_path, dst_graph)
                        global_idx += 1
    return global_idx

def main() -> int:
    parser = argparse.ArgumentParser(description="按序处理单分片目录（比如 test 或 val），不需要 split json 文件")
    parser.add_argument("split_dir", help="需要处理的目录路径 (例如 test)")
    parser.add_argument("--patch_size", type=int, default=1024)
    parser.add_argument("--overlaps", type=int, nargs="*", default=[256, 384])
    parser.add_argument("--edge_width", type=int, default=6)
    parser.add_argument("--inner_offset", type=int, default=5)
    parser.add_argument("--min_edge_ratio", type=float, default=0.05)
    parser.add_argument("--a_density_min", type=float, default=0.001)
    parser.add_argument("--b_density_min", type=float, default=0.001)
    parser.add_argument("--wl_iterations", type=int, default=3)
    parser.add_argument("--sim_threshold", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--workers", type=int, default=4, help="用于并行裁剪的线程数")
    
    args = parser.parse_args()

    split_dir = os.path.abspath(args.split_dir)
    split_name = os.path.basename(split_dir)
    project_root = os.path.dirname(split_dir)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义输出路径
    processed_dir = os.path.join(project_root, f"{split_name}_processed")
    patches_dir = os.path.join(project_root, f"{split_name}_patches")
    
    if not os.path.isdir(split_dir):
        print(f"[ERR] 找不到指定的目录: {split_dir}")
        return 1
        
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(patches_dir, exist_ok=True)

    pairs = find_pairs(split_dir)
    if not pairs:
        print(f"[WARN] 在目录中未找到任何 (图像, 路网) 文件对: {split_dir}")
        return 0
        
    print(f"找到 {len(pairs)} 对数据在目录 {split_dir} 中，准备进行处理。")
    
    # 强制使用并行的tiler，保障速度与xy的正确映射
    tiler = os.path.join(script_dir, "tile_and_crop_patches_parallel.py")
    selector = os.path.join(script_dir, "select_by_wl_similarity.py")

    for data_id, img_path, graph_path in pairs:
        out_dir = os.path.join(processed_dir, f"{data_id}_tiled_candidates")
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"\n--- 正在处理 {data_id} ---")
        
        # 第一步：多线程切分和初步过滤
        tile_cmd = [
            sys.executable, tiler,
            img_path, graph_path,
            "--output", out_dir,
            "--patch_size", str(args.patch_size),
            "--overlaps", *[str(m) for m in args.overlaps],
            "--edge_width", str(args.edge_width),
            "--inner_offset", str(args.inner_offset),
            "--min_edge_ratio", str(args.min_edge_ratio),
            "--workers", str(args.workers)
        ]
        
        t0 = time.time()
        if run_cmd(tile_cmd, cwd=script_dir) != 0:
            print(f"[ERR] 对 {data_id} 执行并行裁剪失败")
            continue
        print(f"裁剪耗时 {time.time() - t0:.2f}s")
        
        # 第二步：基于 WL 相似度的精选
        sel_cmd = [
            sys.executable, selector,
            out_dir,
            "--a_density_min", str(args.a_density_min),
            "--b_density_min", str(args.b_density_min),
            "--wl_iterations", str(args.wl_iterations),
            "--sim_threshold", str(args.sim_threshold),
            "--seed", str(args.seed),
            "--debug_max", "0"
        ]
        
        t1 = time.time()
        if run_cmd(sel_cmd, cwd=script_dir) != 0:
            print(f"[ERR] 对 {data_id} 进行 WL 相似度筛选失败")
            continue
        print(f"筛选耗时 {time.time() - t1:.2f}s")

    # 第三步：收集结果整理到最终的补丁文件夹里（同时复制 RGB，MASK 和 GRAPH）
    print(f"\n--- 正在将处理结果收集归类为 {split_name}_A 和 {split_name}_AB ---")
    
    out_A = os.path.join(patches_dir, f"{split_name}_A")
    count_A = collect_pairs_to_output(processed_dir, out_A, only_A=True)
    print(f"已收集纯 A 划分数据: {count_A} 对 -> {out_A}")
    
    out_AB = os.path.join(patches_dir, f"{split_name}_AB")
    count_AB = collect_pairs_to_output(processed_dir, out_AB, only_A=False)
    print(f"已收集包含重叠块(A+B)划分数据: {count_AB} 对 -> {out_AB}")
    
    print("\n✓ 全部处理完成！")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
