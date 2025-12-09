#!/usr/bin/env python3
"""
Parallel TOPO computation for large datasets.

Splits region IDs into n chunks and runs them in parallel to speed up computation.
Each chunk runs in a separate subprocess, writing results to individual directories.
Finally aggregates all results into a unified summary.

Example:
  python metrics/compute_topo_paralism.py \
    --gt-dir test_data/20cities_test_gt_graph \
    --prop-dir test_data/magToponetgraph \
    --work-dir runs/topo_20cities_parallel \
    --n-parallel 8 \
    --interval 0.00005 --matching-threshold 0.00010

Notes:
- Splits discovered IDs into n chunks (as evenly as possible)
- Each chunk creates results in work-dir/chunk_X/results/
- Final aggregation creates work-dir/summary.csv and work-dir/topo.json
- Use --workers 1 per chunk to avoid threading issues within each process
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel batch compute TOPO for GT/prop folders.")
    parser.add_argument("--gt-dir", required=True,
                        help="Directory containing GT pickles (e.g., region_108_refine_gt_graph.p)")
    parser.add_argument("--prop-dir", required=True,
                        help="Directory containing proposal pickles (e.g., 108.p)")
    parser.add_argument("--work-dir", required=True,
                        help="Output working directory for results")
    parser.add_argument("--n-parallel", type=int, required=True,
                        help="Number of parallel processes to run")
    parser.add_argument("--ids", default="", help="Comma-separated region IDs to limit, e.g. 108,109")
    parser.add_argument("--interval", type=float, default=0.00005, help="topo marble-hole interval")
    parser.add_argument("--matching-threshold", type=float, default=0.00010, help="matching distance threshold")
    parser.add_argument("--lat-top-left", type=float, default=41.0)
    parser.add_argument("--lon-top-left", type=float, default=-71.0)
    parser.add_argument("--r", type=float, default=None, help="propagation distance override (optional)")
    parser.add_argument("--workers", type=int, default=1, help="max worker threads per chunk (recommend 1)")
    return parser.parse_args()


def discover_ids(gt_dir: Path, prop_dir: Path, ids_filter: Optional[Iterable[str]]) -> List[str]:
    """Same ID discovery logic as original compute_topo.py"""
    # gt_re = re.compile(r"region_(\d+)_.*\.(?:p|pickle)$") # for globalscale
    gt_re = re.compile(r"data(\d+).*\.(?:p|pickle)$") # for wild_data
    prop_re = re.compile(r"(\d+)\.p$")

    gt_ids = set()
    for p in gt_dir.iterdir():
        if p.is_file():
            m = gt_re.match(p.name)
            if m:
                gt_ids.add(m.group(1))

    prop_ids = set()
    for p in prop_dir.iterdir():
        if p.is_file():
            m = prop_re.match(p.name)
            if m:
                prop_ids.add(m.group(1))

    ids = gt_ids & prop_ids
    if ids_filter:
        ids = ids & set(ids_filter)
    return sorted(ids, key=lambda s: int(s))


def split_ids_into_chunks(ids: List[str], n_chunks: int) -> List[List[str]]:
    """Split IDs into n chunks as evenly as possible"""
    if n_chunks <= 0:
        raise ValueError("n_chunks must be positive")
    if n_chunks >= len(ids):
        # Each chunk gets at most 1 ID
        return [[id_] for id_ in ids]
    
    chunk_size = len(ids) // n_chunks
    remainder = len(ids) % n_chunks
    
    chunks = []
    start_idx = 0
    for i in range(n_chunks):
        # First 'remainder' chunks get one extra item
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        chunks.append(ids[start_idx:end_idx])
        start_idx = end_idx
    
    return chunks


def run_chunk(chunk_args: Tuple[int, List[str], Path, Path, Path, argparse.Namespace]) -> Tuple[int, List[str], List[float], List[float]]:
    """Run TOPO computation for a single chunk of IDs"""
    chunk_id, ids_chunk, gt_dir, prop_dir, work_dir, args = chunk_args
    
    # Create chunk-specific work directory
    chunk_work_dir = work_dir / f"chunk_{chunk_id}"
    chunk_results_dir = chunk_work_dir / "results"
    chunk_results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Chunk {chunk_id}] Processing {len(ids_chunk)} IDs: {ids_chunk[:5]}{'...' if len(ids_chunk) > 5 else ''}")
    
    # Path to optimized_topo.main
    script_dir = Path(__file__).resolve().parent
    optimized_main = (script_dir / "optimized_topo" / "main.py").resolve()
    
    precisions: List[float] = []
    recalls: List[float] = []
    processed_ids: List[str] = []
    
    for rid in ids_chunk:
        # Locate input files (same logic as original)
        # gt_candidates = list(gt_dir.glob(f"region_{rid}_*.*")) # for globalscale
        gt_candidates = list(gt_dir.glob(f"data{rid}.*")) # for wild_data
        gt_file = None
        for c in gt_candidates:
            if c.suffix in (".p", ".pickle"):
                gt_file = c
                break
        prop_file = prop_dir / f"{rid}.p"
        if not gt_file or not prop_file.exists():
            print(f"[Chunk {chunk_id}] [Skip] Missing files for {rid}: GT={gt_file} PROP={prop_file}")
            continue

        out_txt = chunk_results_dir / f"topo_{rid}.txt"

        cmd = [
            sys.executable,
            str(optimized_main),
            "-graph_gt", str(gt_file),
            "-graph_prop", str(prop_file),
            "-matching_threshold", str(args.matching_threshold),
            "-interval", str(args.interval),
            "-lat_top_left", str(args.lat_top_left),
            "-lon_top_left", str(args.lon_top_left),
            "-output", str(out_txt),
        ]
        if args.r is not None:
            cmd.extend(["-r", str(args.r)])
        if args.workers is not None and int(args.workers) > 0:
            cmd.extend(["-workers", str(int(args.workers))])

        print(f"[Chunk {chunk_id}] [Eval] {rid} -> {out_txt.name}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"[Chunk {chunk_id}] [Error] Failed to process {rid}: {e}")
            print(f"[Chunk {chunk_id}] [Error] stderr: {e.stderr}")
            continue

        # Parse result
        try:
            txt = out_txt.read_text(encoding="utf-8", errors="ignore")
            pair = parse_result_line(txt)
            if pair is None:
                print(f"[Chunk {chunk_id}] [Warn] Could not parse result for {rid}: {out_txt}")
                continue
            p, r = pair
            precisions.append(p)
            recalls.append(r)
            processed_ids.append(rid)
        except Exception as e:
            print(f"[Chunk {chunk_id}] [Warn] Failed to read/parse {out_txt}: {e}")
            continue
    
    print(f"[Chunk {chunk_id}] Completed {len(processed_ids)}/{len(ids_chunk)} regions")
    return chunk_id, processed_ids, precisions, recalls


def parse_result_line(text: str) -> Optional[Tuple[float, float]]:
    """Same result parsing logic as original compute_topo.py"""
    # expecting: "precision=... overall-recall=..."
    parts = text.strip().split()
    if len(parts) >= 2:
        try:
            p = float(parts[0].split("=")[-1])
            r = float(parts[-1].split("=")[-1])
            return p, r
        except ValueError:
            return None
    return None


def aggregate_results(work_dir: Path, all_precisions: List[float], all_recalls: List[float], all_processed_ids: List[str]) -> None:
    """Aggregate all chunk results into final summary files"""
    if not all_precisions:
        print("No TOPO results to aggregate; nothing to summarize.")
        return

    # Calculate per-region F1 scores and filter out invalid data
    valid_f1_list: List[float] = []
    all_f1_list: List[float] = []  # Including invalid ones for JSON output
    valid_p_list: List[float] = []
    valid_r_list: List[float] = []
    skipped_count = 0
    
    for rid, p, r in zip(all_processed_ids, all_precisions, all_recalls):
        # Calculate F1 for this region
        if (p + r) == 0:
            f1 = 0.0
            all_f1_list.append(f1)
            skipped_count += 1
            print(f"[Warning] Region {rid} has p={p}, r={r} (both zero), skipping from average calculation")
        else:
            f1 = 2 * p * r / (p + r)
            all_f1_list.append(f1)
            # Only include in average if not zero
            valid_f1_list.append(f1)
            valid_p_list.append(p)
            valid_r_list.append(r)
    
    # Calculate averages from valid data only
    if valid_f1_list:
        avg_f1 = sum(valid_f1_list) / len(valid_f1_list)
        avg_p = sum(valid_p_list) / len(valid_p_list)
        avg_r = sum(valid_r_list) / len(valid_r_list)
    else:
        avg_f1 = 0.0
        avg_p = 0.0
        avg_r = 0.0
        print("[Warning] No valid regions found (all have p+r=0)")

    # Save summary CSV
    summary_csv = work_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["region_id", "precision", "overall_recall", "f1"])
        for rid, p, r, f1 in zip(all_processed_ids, all_precisions, all_recalls, all_f1_list):
            writer.writerow([rid, p, r, f1])
        writer.writerow(["AVERAGE", avg_p, avg_r, avg_f1])
        writer.writerow(["VALID_COUNT", len(valid_f1_list), "", ""])
        writer.writerow(["SKIPPED_COUNT", skipped_count, "", ""])

    # Save summary JSON
    summary_json = work_dir / "topo.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump({
            "mean topo": [avg_f1, avg_p, avg_r],
            "prec": all_precisions,
            "recall": all_recalls,
            "f1": all_f1_list,
            "valid_count": len(valid_f1_list),
            "skipped_count": skipped_count,
            "total_count": len(all_processed_ids)
        }, f, indent=2)

    print(f"\n=== Final Results ===")
    print(f"Total regions: {len(all_processed_ids)}")
    print(f"Valid regions: {len(valid_f1_list)}")
    print(f"Skipped regions (p+r=0): {skipped_count}")
    print(f"Averages (from valid regions only):")
    print(f"  Precision: {avg_p:.6f}")
    print(f"  Recall: {avg_r:.6f}")
    print(f"  F1: {avg_f1:.6f}")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote summary JSON: {summary_json}")


def main() -> None:
    args = parse_args()

    gt_dir = Path(args.gt_dir).resolve()
    prop_dir = Path(args.prop_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if not gt_dir.exists() or not prop_dir.exists():
        raise SystemExit(f"GT dir or Prop dir does not exist:\n  GT:   {gt_dir}\n  Prop: {prop_dir}")

    # Discover all matching IDs
    ids_filter = [s for s in args.ids.split(",") if s.strip()] if args.ids else []
    ids = discover_ids(gt_dir, prop_dir, ids_filter)
    if not ids:
        raise SystemExit("No matching region IDs found between GT and proposal folders.")

    print(f"Found {len(ids)} regions: {', '.join(ids[:10])}{'...' if len(ids) > 10 else ''}")
    print(f"Work directory: {work_dir}")
    print(f"Splitting into {args.n_parallel} parallel chunks")

    # Split IDs into chunks
    chunks = split_ids_into_chunks(ids, args.n_parallel)
    print(f"Chunk sizes: {[len(chunk) for chunk in chunks]}")

    # Prepare arguments for each chunk
    chunk_args_list = []
    for i, chunk in enumerate(chunks):
        chunk_args_list.append((i, chunk, gt_dir, prop_dir, work_dir, args))

    # Run chunks in parallel
    start_time = time.time()
    all_results: List[Tuple[str, float, float]] = []  # (region_id, precision, recall)

    with ProcessPoolExecutor(max_workers=args.n_parallel) as executor:
        future_to_chunk = {executor.submit(run_chunk, chunk_args): chunk_args[0] for chunk_args in chunk_args_list}
        
        for future in as_completed(future_to_chunk):
            chunk_id = future_to_chunk[future]
            try:
                chunk_id_result, processed_ids, precisions, recalls = future.result()
                print(f"[Main] Chunk {chunk_id_result} finished: {len(processed_ids)} regions processed")
                
                # CRITICAL: Keep ID, precision, recall in sync by zipping them together
                if len(processed_ids) != len(precisions) or len(processed_ids) != len(recalls):
                    print(f"[Main] [ERROR] Chunk {chunk_id_result} data mismatch: {len(processed_ids)} IDs, {len(precisions)} precisions, {len(recalls)} recalls")
                    continue
                
                # Aggregate results from this chunk, maintaining correspondence
                for rid, p, r in zip(processed_ids, precisions, recalls):
                    all_results.append((rid, p, r))
                            
            except Exception as e:
                print(f"[Main] Chunk {chunk_id} generated an exception: {e}")

    elapsed_time = time.time() - start_time
    print(f"\nParallel processing completed in {elapsed_time:.2f} seconds")

    # Sort results by region ID to maintain consistency
    if all_results:
        # Sort by numeric region ID
        all_results.sort(key=lambda x: int(x[0]))
        # Unpack sorted results
        all_processed_ids = [r[0] for r in all_results]
        all_precisions = [r[1] for r in all_results]
        all_recalls = [r[2] for r in all_results]
    else:
        all_processed_ids = []
        all_precisions = []
        all_recalls = []

    # Aggregate final results
    aggregate_results(work_dir, all_precisions, all_recalls, all_processed_ids)


if __name__ == "__main__":
    main()

