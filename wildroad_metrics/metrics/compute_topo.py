#!/usr/bin/env python3
"""
Compute TOPO for a folder of graphs without touching original data.

- Discovers matching region IDs from GT and proposal folders
- Calls optimized_topo.main with absolute paths and per-region output path
- Stores per-region result files and a summary CSV/JSON inside the work directory

Example:
  python metrics/compute_topo.py \
    --gt-dir test_data/20cities_test_gt_graph \
    --prop-dir test_data/magToponetgraph \
    --work-dir runs/topo_20cities \
    --interval 0.00005 --matching-threshold 0.00010

Notes:
- This script never modifies input folders. All outputs live under --work-dir.
- Uses the optimized, single-pair evaluator. Ensure Python deps installed: rtree, hopcroftkarp, numpy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch compute TOPO for GT/prop folders.")
    parser.add_argument("--gt-dir", required=True,
                        help="Directory containing GT pickles (e.g., region_108_refine_gt_graph.p)")
    parser.add_argument("--prop-dir", required=True,
                        help="Directory containing proposal pickles (e.g., 108.p)")
    parser.add_argument("--work-dir", required=True,
                        help="Output working directory for results")
    parser.add_argument("--ids", default="", help="Comma-separated region IDs to limit, e.g. 108,109")
    parser.add_argument("--interval", type=float, default=0.00005, help="topo marble-hole interval")
    parser.add_argument("--matching-threshold", type=float, default=0.00010, help="matching distance threshold")
    parser.add_argument("--lat-top-left", type=float, default=41.0)
    parser.add_argument("--lon-top-left", type=float, default=-71.0)
    parser.add_argument("--r", type=float, default=None, help="propagation distance override (optional)")
    parser.add_argument("--workers", type=int, default=1, help="max worker threads for TOPO, set 1 on Linux if hangs")
    return parser.parse_args()


def discover_ids(gt_dir: Path, prop_dir: Path, ids_filter: Optional[Iterable[str]]) -> List[str]:
    gt_re = re.compile(r"region_(\d+)_.*\.(?:p|pickle)$")
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


def parse_result_line(text: str) -> Optional[Tuple[float, float]]:
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


def main() -> None:
    args = parse_args()

    gt_dir = Path(args.gt_dir).resolve()
    prop_dir = Path(args.prop_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    results_dir = work_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not gt_dir.exists() or not prop_dir.exists():
        raise SystemExit(f"GT dir or Prop dir does not exist:\n  GT:   {gt_dir}\n  Prop: {prop_dir}")

    ids_filter = [s for s in args.ids.split(",") if s.strip()] if args.ids else []
    ids = discover_ids(gt_dir, prop_dir, ids_filter)
    if not ids:
        raise SystemExit("No matching region IDs found between GT and proposal folders.")

    print(f"Found {len(ids)} regions: {', '.join(ids)}")
    print(f"Work directory: {work_dir}")

    # module path for optimized_topo.main
    script_dir = Path(__file__).resolve().parent
    optimized_main = (script_dir / "optimized_topo" / "main.py").resolve()

    precisions: List[float] = []
    recalls: List[float] = []

    for rid in ids:
        # Locate input files
        gt_candidates = list(gt_dir.glob(f"region_{rid}_*.*"))
        gt_file = None
        for c in gt_candidates:
            if c.suffix in (".p", ".pickle"):
                gt_file = c
                break
        prop_file = prop_dir / f"{rid}.p"
        if not gt_file or not prop_file.exists():
            print(f"[Skip] Missing files for {rid}: GT={gt_file} PROP={prop_file}")
            continue

        out_txt = results_dir / f"topo_{rid}.txt"

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

        print(f"[Eval] {rid} -> {out_txt.name}")
        subprocess.run(cmd, check=True)

        try:
            txt = out_txt.read_text(encoding="utf-8", errors="ignore")
            pair = parse_result_line(txt)
            if pair is None:
                print(f"[Warn] Could not parse result for {rid}: {out_txt}")
                continue
            p, r = pair
            precisions.append(p)
            recalls.append(r)
        except Exception as e:
            print(f"[Warn] Failed to read/parse {out_txt}: {e}")
            continue

    if not precisions:
        print("No TOPO results parsed; nothing to summarize.")
        return

    # Averages
    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)
    if (avg_p + avg_r) == 0:
        avg_f1 = 0.0
    else:
        avg_f1 = 2 * avg_p * avg_r / (avg_p + avg_r)

    # Save summary CSV and JSON
    summary_csv = work_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["region_id", "precision", "overall_recall"])
        for rid, p, r in zip(ids, precisions, recalls):
            writer.writerow([rid, p, r])
        writer.writerow(["AVERAGE", avg_p, avg_r])
        writer.writerow(["F1", avg_f1])

    summary_json = work_dir / "topo.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump({
            "mean topo": [avg_f1, avg_p, avg_r],
            "prec": precisions,
            "recall": recalls,
            "f1": [
                (0.0 if (p + r) == 0 else 2 * p * r / (p + r))
                for p, r in zip(precisions, recalls)
            ]
        }, f, indent=2)

    print(f"Averages -> Precision: {avg_p}, Recall: {avg_r}, F1: {avg_f1}")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote summary JSON: {summary_json}")


if __name__ == "__main__":
    main()

