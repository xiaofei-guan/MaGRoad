#!/usr/bin/env python3
"""
Parallel TOPO computation script for original (non-optimized) TOPO implementation.
Splits region IDs into chunks, computes them in parallel, and aggregates results.
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple


def discover_region_ids(gt_dir: Path) -> List[str]:
    """Discover all region IDs from GT directory."""
    region_ids = []
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")
    
    for item in sorted(gt_dir.iterdir()):
        if item.is_file() and (item.suffix == '.p' or item.suffix == '.pickle'):
            region_id = item.stem
            region_ids.append(region_id)
    
    if not region_ids:
        raise ValueError(f"No .p or .pickle files found in {gt_dir}")
    
    print(f"Discovered {len(region_ids)} regions")
    return region_ids


def split_ids_into_chunks(ids: List[str], n_chunks: int) -> List[List[str]]:
    """Split IDs into n chunks with random shuffling for load balancing."""
    shuffled_ids = ids.copy()
    random.shuffle(shuffled_ids)
    
    chunk_size = len(shuffled_ids) // n_chunks
    remainder = len(shuffled_ids) % n_chunks
    
    chunks = []
    start = 0
    for i in range(n_chunks):
        # Distribute remainder across first chunks
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        chunks.append(shuffled_ids[start:end])
        start = end
    
    return chunks


def run_chunk(
    chunk_id: int,
    region_ids: List[str],
    gt_dir: Path,
    prop_dir: Path,
    chunk_work_dir: Path,
    topo_main_py: Path,
    workers: int = 1
) -> Tuple[int, List[str], List[float], List[float]]:
    """
    Run TOPO computation for a chunk of region IDs.
    Returns: (chunk_id, processed_ids, precisions, recalls)
    """
    print(f"[Chunk {chunk_id}] Processing {len(region_ids)} regions...")
    
    chunk_work_dir.mkdir(parents=True, exist_ok=True)
    
    processed_ids = []
    precisions = []
    recalls = []
    
    for region_id in region_ids:
        # Try both .pickle and .p extensions for GT
        gt_file = gt_dir / f"{region_id}.pickle"
        if not gt_file.exists():
            gt_file = gt_dir / f"{region_id}.p"
        
        # Extract numeric index from region_id (e.g., "data123" -> "123")
        prop_id = region_id.replace("data", "")
        
        # Try both .pickle and .p extensions for Prop
        prop_file = prop_dir / f"{prop_id}.pickle"
        if not prop_file.exists():
            prop_file = prop_dir / f"{prop_id}.p"
        
        if not gt_file.exists():
            print(f"[Chunk {chunk_id}] WARNING: GT file not found: {region_id}")
            continue
        
        if not prop_file.exists():
            print(f"[Chunk {chunk_id}] WARNING: Prop file not found: {prop_id}")
            continue
        
        out_txt = chunk_work_dir / f"topo_{prop_id}.txt"
        
        cmd = [
            sys.executable,
            str(topo_main_py),
            "-graph_gt", str(gt_file),
            "-graph_prop", str(prop_file),
            "-output", str(out_txt),
        ]
        
        if workers is not None and workers > 0:
            cmd.extend(["-workers", str(workers)])
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=str(topo_main_py.parent)
            )
            
            # Parse output: look for "precision=X overall-recall=Y"
            output_lines = result.stdout.strip().split('\n')
            precision = None
            recall = None
            
            for line in output_lines:
                if 'precision=' in line and 'overall-recall=' in line:
                    parts = line.split()
                    for part in parts:
                        if part.startswith('precision='):
                            precision = float(part.split('=')[1])
                        elif part.startswith('overall-recall='):
                            recall = float(part.split('=')[1])
            
            if precision is not None and recall is not None:
                processed_ids.append(region_id)
                precisions.append(precision)
                recalls.append(recall)
                print(f"[Chunk {chunk_id}] {region_id}: P={precision:.6f}, R={recall:.6f}")
            else:
                print(f"[Chunk {chunk_id}] WARNING: Could not parse results for {region_id}")
        
        except subprocess.CalledProcessError as e:
            print(f"[Chunk {chunk_id}] ERROR processing {region_id}: {e}")
            print(f"[Chunk {chunk_id}] STDERR: {e.stderr}")
            continue
    
    print(f"[Chunk {chunk_id}] Completed {len(processed_ids)}/{len(region_ids)} regions")
    return chunk_id, processed_ids, precisions, recalls


def aggregate_results(
    all_processed_ids: List[str],
    all_precisions: List[float],
    all_recalls: List[float],
    output_dir: Path
) -> None:
    """Aggregate all chunk results and compute F1 scores."""
    if not all_processed_ids:
        print("ERROR: No results to aggregate!")
        return
    
    # Calculate F1 for each region
    valid_f1_list: List[float] = []
    all_f1_list: List[float] = []
    skipped_count = 0
    
    for rid, p, r in zip(all_processed_ids, all_precisions, all_recalls):
        if (p + r) == 0:
            f1 = 0.0
            skipped_count += 1
        else:
            f1 = 2 * p * r / (p + r)
            valid_f1_list.append(f1)
        all_f1_list.append(f1)
    
    # Compute averages (only from valid F1 scores)
    if valid_f1_list:
        avg_f1 = sum(valid_f1_list) / len(valid_f1_list)
        # Also compute avg P and R from valid entries
        valid_p_list = [all_precisions[i] for i in range(len(all_precisions)) if all_f1_list[i] > 0]
        valid_r_list = [all_recalls[i] for i in range(len(all_recalls)) if all_f1_list[i] > 0]
        avg_p = sum(valid_p_list) / len(valid_p_list) if valid_p_list else 0.0
        avg_r = sum(valid_r_list) / len(valid_r_list) if valid_r_list else 0.0
    else:
        avg_f1 = 0.0
        avg_p = 0.0
        avg_r = 0.0
    
    valid_count = len(valid_f1_list)
    total_count = len(all_processed_ids)
    
    print(f"\n{'='*60}")
    print(f"TOPO Results Summary (Original Implementation)")
    print(f"{'='*60}")
    print(f"Total regions:    {total_count}")
    print(f"Valid F1:         {valid_count}")
    print(f"Skipped (P+R=0):  {skipped_count}")
    print(f"Average Precision: {avg_p:.6f}")
    print(f"Average Recall:    {avg_r:.6f}")
    print(f"Average F1:        {avg_f1:.6f}")
    print(f"{'='*60}\n")
    
    # Write CSV
    csv_path = output_dir / "summary.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["region_id", "precision", "recall", "f1"])
        for rid, p, r, f1 in zip(all_processed_ids, all_precisions, all_recalls, all_f1_list):
            writer.writerow([rid, f"{p:.6f}", f"{r:.6f}", f"{f1:.6f}"])
        writer.writerow(["AVERAGE", f"{avg_p:.6f}", f"{avg_r:.6f}", f"{avg_f1:.6f}"])
        writer.writerow(["VALID_COUNT", valid_count, "", ""])
        writer.writerow(["SKIPPED_COUNT", skipped_count, "", ""])
    
    print(f"CSV summary written to: {csv_path}")
    
    # Write JSON
    json_path = output_dir / "topo.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "mean topo": [avg_f1, avg_p, avg_r],
            "prec": all_precisions,
            "recall": all_recalls,
            "f1": all_f1_list,
            "valid_count": len(valid_f1_list),
            "skipped_count": skipped_count,
            "total_count": len(all_processed_ids)
        }, f, indent=2)
    
    print(f"JSON summary written to: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel TOPO computation using original (non-optimized) TOPO implementation"
    )
    parser.add_argument("--gt-dir", required=True, type=Path, help="Ground truth graphs directory")
    parser.add_argument("--prop-dir", required=True, type=Path, help="Proposed graphs directory")
    parser.add_argument("--work-dir", required=True, type=Path, help="Working directory for outputs")
    parser.add_argument("--n-parallel", type=int, default=8, help="Number of parallel processes")
    parser.add_argument("--workers", type=int, default=1, help="Worker threads per chunk (set 1 for original topo)")
    
    args = parser.parse_args()
    
    gt_dir = args.gt_dir.resolve()
    prop_dir = args.prop_dir.resolve()
    work_dir = args.work_dir.resolve()
    n_parallel = args.n_parallel
    workers = args.workers
    
    # Find topo/main.py
    script_dir = Path(__file__).parent.resolve()
    topo_main_py = script_dir / "topo" / "main.py"
    
    if not topo_main_py.exists():
        raise FileNotFoundError(f"TOPO main.py not found at: {topo_main_py}")
    
    print(f"Using original TOPO implementation: {topo_main_py}")
    print(f"GT dir: {gt_dir}")
    print(f"Prop dir: {prop_dir}")
    print(f"Work dir: {work_dir}")
    print(f"Parallel processes: {n_parallel}")
    print(f"Workers per chunk: {workers}")
    
    # Discover regions
    region_ids = discover_region_ids(gt_dir)
    
    # Split into chunks
    chunks = split_ids_into_chunks(region_ids, n_parallel)
    print(f"\nSplit {len(region_ids)} regions into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk)} regions")
    
    # Create work directory
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Run chunks in parallel
    all_results: List[Tuple[str, float, float]] = []  # (region_id, precision, recall)
    
    with ProcessPoolExecutor(max_workers=n_parallel) as executor:
        futures = []
        for chunk_id, chunk in enumerate(chunks):
            chunk_work_dir = work_dir / f"chunk_{chunk_id}"
            future = executor.submit(
                run_chunk,
                chunk_id,
                chunk,
                gt_dir,
                prop_dir,
                chunk_work_dir,
                topo_main_py,
                workers
            )
            futures.append((future, chunk_id))
        
        # Collect results
        for future, chunk_id in futures:
            try:
                chunk_id_result, processed_ids, precisions, recalls = future.result()
                
                # Data consistency check
                if not (len(processed_ids) == len(precisions) == len(recalls)):
                    raise ValueError(
                        f"Chunk {chunk_id_result} data mismatch: "
                        f"IDs={len(processed_ids)}, P={len(precisions)}, R={len(recalls)}"
                    )
                
                for rid, p, r in zip(processed_ids, precisions, recalls):
                    rid = rid.replace("data", "")
                    all_results.append((rid, p, r))
                
                print(f"[Main] Collected {len(processed_ids)} results from chunk {chunk_id_result}")
            
            except Exception as e:
                print(f"[Main] ERROR: Chunk {chunk_id} failed: {e}")
                continue
    
    # Sort by region ID for consistency
    if all_results:
        all_results.sort(key=lambda x: int(x[0]))
        all_processed_ids = [r[0] for r in all_results]
        all_precisions = [r[1] for r in all_results]
        all_recalls = [r[2] for r in all_results]
    else:
        all_processed_ids = []
        all_precisions = []
        all_recalls = []
    
    # Aggregate results
    aggregate_results(all_processed_ids, all_precisions, all_recalls, work_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
