#!/usr/bin/env python3
"""
Parallel APLS computation for large datasets.

Splits region IDs into n chunks and runs them in parallel to speed up computation.
Each chunk runs in a separate subprocess, writing results to individual directories.
Finally aggregates all results into a unified summary.

Example:
  python metrics/compute_apls_parallel.py \
    --gt-dir test_data/20cities_test_gt_graph \
    --prop-dir test_data/magToponetgraph \
    --work-dir runs/apls_20cities_parallel \
    --n-parallel 8 \
    --go-script optimized_main.go

Notes:
- Splits discovered IDs into n chunks (as evenly as possible)
- Each chunk creates results in work-dir/chunk_X/json/ and work-dir/chunk_X/results/
- Final aggregation creates work-dir/summary.csv and work-dir/apls.json
- Uses main.go (non-optimized) as the core computation engine
"""

from __future__ import annotations

import argparse
import csv
import json
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
    parser = argparse.ArgumentParser(description="Parallel batch compute APLS for GT/prop folders.")
    parser.add_argument("--gt-dir", required=True,
                        help="Directory containing GT pickles (e.g., region_108_refine_gt_graph.p)")
    parser.add_argument("--prop-dir", required=True,
                        help="Directory containing proposal pickles (e.g., 108.p)")
    parser.add_argument("--work-dir", required=True,
                        help="Output working directory for results")
    parser.add_argument("--n-parallel", type=int, required=True,
                        help="Number of parallel processes to run")
    parser.add_argument("--go-script", default="main.go", choices=["main.go", "optimized_main.go"],
                        help="Go script to run: main.go (apls/) or optimized_main.go (optimized_apls/)")
    parser.add_argument("--ids", default="", help="Comma-separated region IDs to limit, e.g. 108,109")
    parser.add_argument("--small-tiles", action="store_true",
                        help="Use small tile parameters (passes a 5th arg to Go script)")
    parser.add_argument("--go-bin", default=os.environ.get("GO_BIN", "go"),
                        help="Go binary name/path (default: 'go' or $GO_BIN)")
    parser.add_argument("--goproxy", default=os.environ.get("GOPROXY", ""),
                        help="Optional GOPROXY to set in env for dependency download")
    parser.add_argument("--offset", type=int, default=2714,
                        help="Offset to add to GT IDs to match Prop IDs (default: 2714)")
    return parser.parse_args()


def discover_ids(gt_dir: Path, prop_dir: Path, ids_filter: Optional[Iterable[str]], offset: int = 0) -> List[str]:
    """Same ID discovery logic as original compute_apls.py"""
    gt_re = re.compile(r"region_(\d+)_.*\.(?:p|pickle)$") # for globalscale
    # gt_re = re.compile(r"data(\d+).*\.(?:p|pickle)$") # for wild_data
    prop_re = re.compile(r"(\d+)\.p$")

    gt_ids = set()
    for p in gt_dir.iterdir():
        if p.is_file():
            m = gt_re.match(p.name)
            if m:
                gt_ids.add(m.group(1))
    
    gt_ids = {str(int(id) + offset) for id in gt_ids}

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
    """
    Split IDs into n chunks as evenly as possible.
    IDs are randomly shuffled before splitting to balance workload across chunks.
    """
    if n_chunks <= 0:
        raise ValueError("n_chunks must be positive")
    if n_chunks >= len(ids):
        # Each chunk gets at most 1 ID
        return [[id_] for id_ in ids]
    
    # IMPORTANT: Shuffle IDs randomly for load balancing
    # This prevents slow chunks caused by consecutive difficult regions
    shuffled_ids = ids.copy()  # Create a copy to avoid modifying original
    random.shuffle(shuffled_ids)
    
    chunk_size = len(shuffled_ids) // n_chunks
    remainder = len(shuffled_ids) % n_chunks
    
    chunks = []
    start_idx = 0
    for i in range(n_chunks):
        # First 'remainder' chunks get one extra item
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        chunks.append(shuffled_ids[start_idx:end_idx])
        start_idx = end_idx
    
    return chunks


def run_convert(convert_py: Path, p_in: Path, j_out: Path) -> None:
    """Convert pickle to JSON"""
    cmd = [sys.executable, str(convert_py), str(p_in), str(j_out)]
    subprocess.run(cmd, check=True, cwd=str(convert_py.parent), capture_output=True, text=True)


def run_go(go_bin: str, module_dir: Path, go_script: str, gt_json: Path, prop_json: Path, 
           out_txt: Path, small: bool, goproxy: str) -> None:
    """Run Go APLS evaluator"""
    if go_script == "optimized_main.go":
        script_name = "optimized_main.go"
    else:
        script_name = "main.go"
    
    args = [go_bin, "run", script_name, str(gt_json), str(prop_json), str(out_txt)]
    if small:
        args.append("1")
    env = os.environ.copy()
    if goproxy:
        env["GOPROXY"] = goproxy
    subprocess.run(args, check=True, cwd=str(module_dir), env=env, capture_output=True, text=True)


def parse_result_line(text: str) -> Optional[Tuple[float, float, float]]:
    """Parse APLS result line: apls_gt apls_prop apls"""
    parts = text.strip().split()
    if len(parts) >= 3:
        try:
            return float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            return None
    return None


def run_chunk(chunk_args: Tuple[int, List[str], Path, Path, Path, argparse.Namespace]) -> Tuple[int, List[str], List[float], List[float], List[float]]:
    """Run APLS computation for a single chunk of IDs"""
    chunk_id, ids_chunk, gt_dir, prop_dir, work_dir, args = chunk_args
    
    # Create chunk-specific work directory
    chunk_work_dir = work_dir / f"chunk_{chunk_id}"
    chunk_json_dir = chunk_work_dir / "json"
    chunk_results_dir = chunk_work_dir / "results"
    chunk_json_dir.mkdir(parents=True, exist_ok=True)
    chunk_results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Chunk {chunk_id}] Processing {len(ids_chunk)} IDs: {ids_chunk[:5]}{'...' if len(ids_chunk) > 5 else ''}")
    
    # Path to converter and Go module
    script_dir = Path(__file__).resolve().parent
    if args.go_script == "optimized_main.go":
        subdir = "optimized_apls"
    else:
        subdir = "apls"
    
    convert_py = (script_dir / subdir / "convert.py").resolve()
    module_dir = (script_dir / subdir).resolve()
    
    apls_gt_list: List[float] = []
    apls_prop_list: List[float] = []
    apls_list: List[float] = []
    processed_ids: List[str] = []

    for rid in ids_chunk:
        # Locate input files (same logic as original)
        gt_id = str(int(rid) - args.offset)
        gt_candidates = list(gt_dir.glob(f"region_{gt_id}_*.*")) # for globalscale
        # gt_candidates = list(gt_dir.glob(f"data{rid}.*")) # for wild_data
        gt_file = None
        for c in gt_candidates:
            if c.suffix in (".p", ".pickle"):
                gt_file = c
                break
        prop_file = prop_dir / f"{rid}.p"
        
        if not gt_file or not prop_file.exists():
            print(f"[Chunk {chunk_id}] [Skip] Missing files for {rid}: GT={gt_file} PROP={prop_file}")
            continue
        
        # Convert pickle to JSON
        gt_json = chunk_json_dir / f"gt_{gt_id}.json"
        prop_json = chunk_json_dir / f"prop_{rid}.json"
        out_txt = chunk_results_dir / f"apls_{rid}.txt"
        
        try:
            # Convert GT and Prop pickles to JSON
            run_convert(convert_py, gt_file, gt_json)
            run_convert(convert_py, prop_file, prop_json)
        except subprocess.CalledProcessError as e:
            print(f"[Chunk {chunk_id}] [Error] Conversion failed for {rid}: {e}")
            print(f"[Chunk {chunk_id}] [Error] stderr: {e.stderr}")
            continue
        
        try:
            # Run Go APLS evaluator
            run_go(args.go_bin, module_dir, args.go_script, gt_json, prop_json, 
                   out_txt, args.small_tiles, args.goproxy)
        except subprocess.CalledProcessError as e:
            print(f"[Chunk {chunk_id}] [Error] APLS computation failed for {rid}: {e}")
            print(f"[Chunk {chunk_id}] [Error] stderr: {e.stderr}")
            continue
        
        # Parse result
        try:
            txt = out_txt.read_text(encoding="utf-8", errors="ignore")
            triple = parse_result_line(txt)
            if triple is None:
                print(f"[Chunk {chunk_id}] [Warn] Could not parse result for {rid}: {out_txt}")
                continue
            apls_gt, apls_prop, apls = triple
            apls_gt_list.append(apls_gt)
            apls_prop_list.append(apls_prop)
            apls_list.append(apls)
            processed_ids.append(rid)
        except Exception as e:
            print(f"[Chunk {chunk_id}] [Warn] Failed to read/parse {out_txt}: {e}")
            continue
    
    print(f"[Chunk {chunk_id}] Completed {len(processed_ids)}/{len(ids_chunk)} regions")
    return chunk_id, processed_ids, apls_gt_list, apls_prop_list, apls_list


def aggregate_results(work_dir: Path, all_apls_gt: List[float], all_apls_prop: List[float], 
                      all_apls: List[float], all_processed_ids: List[str]) -> None:
    """Aggregate all chunk results into final summary files"""
    if not all_apls:
        print("No APLS results to aggregate; nothing to summarize.")
        return

    # Filter out zero values
    valid_apls = [apls for apls in all_apls if apls != 0]
    
    if not valid_apls:
        print("No valid APLS results after filtering zeros.")
        return
    
    # Calculate average
    final_apls = sum(valid_apls) / len(valid_apls)
    
    print(f"Filtered {len(all_apls) - len(valid_apls)} samples with APLS = 0")
    print(f"Final APLS (avg of {len(valid_apls)} samples): {final_apls:.6f}")

    # Save summary CSV
    summary_csv = work_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["region_id", "apls_gt", "apls_prop", "apls"])
        for rid, apls_gt, apls_prop, apls in zip(all_processed_ids, all_apls_gt, all_apls_prop, all_apls):
            writer.writerow([rid, apls_gt, apls_prop, apls])

    # Save summary JSON
    summary_json = work_dir / "apls.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump({
            "apls": all_apls,
            "final_APLS": final_apls
        }, f, indent=2)

    print(f"\n=== Final Results ===")
    print(f"Processed regions: {len(all_processed_ids)}")
    print(f"Final APLS: {final_apls:.6f}")
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
    ids = discover_ids(gt_dir, prop_dir, ids_filter, args.offset)
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
    all_results: List[Tuple[str, float, float, float]] = []  # (region_id, apls_gt, apls_prop, apls)

    with ProcessPoolExecutor(max_workers=args.n_parallel) as executor:
        future_to_chunk = {executor.submit(run_chunk, chunk_args): chunk_args[0] for chunk_args in chunk_args_list}
        
        for future in as_completed(future_to_chunk):
            chunk_id = future_to_chunk[future]
            try:
                chunk_id_result, processed_ids, apls_gt_list, apls_prop_list, apls_list = future.result()
                print(f"[Main] Chunk {chunk_id_result} finished: {len(processed_ids)} regions processed")
                
                # CRITICAL: Keep ID and all three APLS values in sync by zipping them together
                if len(processed_ids) != len(apls_gt_list) or len(processed_ids) != len(apls_prop_list) or len(processed_ids) != len(apls_list):
                    print(f"[Main] [ERROR] Chunk {chunk_id_result} data mismatch: {len(processed_ids)} IDs, {len(apls_gt_list)} apls_gt, {len(apls_prop_list)} apls_prop, {len(apls_list)} apls")
                    continue
                
                # Aggregate results from this chunk, maintaining correspondence
                for rid, apls_gt, apls_prop, apls in zip(processed_ids, apls_gt_list, apls_prop_list, apls_list):
                    all_results.append((rid, apls_gt, apls_prop, apls))
                            
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
        all_apls_gt = [r[1] for r in all_results]
        all_apls_prop = [r[2] for r in all_results]
        all_apls = [r[3] for r in all_results]
    else:
        all_processed_ids = []
        all_apls_gt = []
        all_apls_prop = []
        all_apls = []

    # Aggregate final results
    aggregate_results(work_dir, all_apls_gt, all_apls_prop, all_apls, all_processed_ids)


if __name__ == "__main__":
    main()

