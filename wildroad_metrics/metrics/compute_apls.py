#!/usr/bin/env python3
"""
Compute APLS for a folder of graphs without touching original data.

- Discovers matching region IDs from GT and proposal folders
- Converts pickles to JSON into a clean work directory
- Runs the Go evaluator (metrics/apls/main.go) with absolute paths
- Stores per-region result files and a summary CSV inside the work directory

Example:
  python metrics/compute_apls.py \
    --gt-dir test_data/20cities_test_gt_graph \
    --prop-dir test_data/magToponetgraph \
    --work-dir runs/apls_20cities \
    --go-script main.go

  # Use optimized version (from optimized_apls/):
  python metrics/compute_apls.py \
    --gt-dir test_data/20cities_test_gt_graph \
    --prop-dir test_data/magToponetgraph \
    --work-dir runs/apls_optimized \
    --go-script optimized_main.go

Notes:
- This script never modifies input folders. All intermediates/results live under --work-dir.
- On Windows, ensure Go is installed and available as 'go'.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch compute APLS for GT/prop folders.")
    parser.add_argument("--gt-dir", required=True,
                        help="Directory containing GT pickles (e.g., region_108_refine_gt_graph.p)")
    parser.add_argument("--prop-dir", required=True,
                        help="Directory containing proposal pickles (e.g., 108.p)")
    parser.add_argument("--work-dir", required=True,
                        help="Output working directory for intermediates and results")
    parser.add_argument("--go-script", default="main.go", choices=["main.go", "optimized_main.go"],
                        help="Go script to run: main.go (apls/) or optimized_main.go (optimized_apls/) (default: main.go)")
    parser.add_argument("--ids", default="", help="Comma-separated region IDs to limit, e.g. 108,109")
    parser.add_argument("--small-tiles", action="store_true",
                        help="Use small tile parameters (passes a 5th arg to Go script)")
    parser.add_argument("--go-bin", default=os.environ.get("GO_BIN", "go"),
                        help="Go binary name/path (default: 'go' or $GO_BIN)")
    parser.add_argument("--goproxy", default=os.environ.get("GOPROXY", ""),
                        help="Optional GOPROXY to set in env for dependency download")
    return parser.parse_args()


def discover_ids(gt_dir: Path, prop_dir: Path, ids_filter: Optional[Iterable[str]]) -> List[str]:
    # GT: region_<id>_... .p or .pickle
    gt_re = re.compile(r"region_(\d+)_.*\.(?:p|pickle)$")
    # PROP: <id>.p
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


def ensure_dirs(base: Path) -> Tuple[Path, Path]:
    json_dir = base / "json"
    results_dir = base / "results"
    json_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return json_dir, results_dir


def run_convert(convert_py: Path, p_in: Path, j_out: Path) -> None:
    cmd = [sys.executable, str(convert_py), str(p_in), str(j_out)]
    # Run convert with cwd at the script's directory to avoid stray outputs
    subprocess.run(cmd, check=True, cwd=str(convert_py.parent))


def run_go(go_bin: str, module_dir: Path, go_script: str, gt_json: Path, prop_json: Path, out_txt: Path, small: bool,
           goproxy: str) -> None:
    # For optimized_apls, the script is optimized_main.go, for apls it's main.go
    # But we run with the actual filename in the respective directory
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
    subprocess.run(args, check=True, cwd=str(module_dir), env=env)


def parse_result_line(text: str) -> Optional[Tuple[float, float, float]]:
    parts = text.strip().split()
    if len(parts) >= 3:
        try:
            return float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            return None
    return None


def main() -> None:
    args = parse_args()

    gt_dir = Path(args.gt_dir).resolve()
    prop_dir = Path(args.prop_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    json_dir, results_dir = ensure_dirs(work_dir)

    # Paths to converter and module (Go main + go.mod live here)
    # This script resides in metrics/, choose folder based on go_script
    script_dir = Path(__file__).resolve().parent
    if args.go_script == "optimized_main.go":
        subdir = "optimized_apls"
    else:
        subdir = "apls"
    
    convert_py = (script_dir / subdir / "convert.py").resolve()
    module_dir = (script_dir / subdir).resolve()

    if not gt_dir.exists() or not prop_dir.exists():
        raise SystemExit(f"GT dir or Prop dir does not exist:\n  GT:   {gt_dir}\n  Prop: {prop_dir}")

    ids_filter = [s for s in args.ids.split(",") if s.strip()] if args.ids else []
    ids = discover_ids(gt_dir, prop_dir, ids_filter)
    if not ids:
        raise SystemExit("No matching region IDs found between GT and proposal folders.")

    print(f"Found {len(ids)} regions: {', '.join(ids)}")
    print(f"Work directory: {work_dir}")

    # Convert + Evaluate per ID
    for rid in ids:
        # Locate input files
        # GT could be 'region_<id>_refine_gt_graph.p' or 'region_<id>_graph_gt.pickle'
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

        gt_json = json_dir / f"gt_{rid}.json"
        prop_json = json_dir / f"prop_{rid}.json"
        out_txt = results_dir / f"apls_{rid}.txt"

        print(f"gt_file: {gt_file}")
        print(f"prop_file: {prop_file}")
        print(f"[Convert] {rid} -> {gt_json.name}, {prop_json.name}")
        run_convert(convert_py, gt_file, gt_json)
        run_convert(convert_py, prop_file, prop_json)

        print(f"[Eval] {rid} -> {out_txt.name} (using {args.go_script} from {subdir}/)")
        run_go(args.go_bin, module_dir, args.go_script, gt_json, prop_json, out_txt, args.small_tiles, args.goproxy)

    # Aggregate results into CSV and compute final statistics
    summary_csv = work_dir / "summary.csv"
    apls_json = work_dir / "apls.json"
    rows: List[Tuple[str, float, float, float]] = []
    apls_values = []  # For final APLS computation
    
    for rid in ids:
        out_txt = results_dir / f"apls_{rid}.txt"
        if not out_txt.exists():
            continue
        try:
            txt = out_txt.read_text(encoding="utf-8", errors="ignore")
            triple = parse_result_line(txt)
            if triple is None:
                continue
            rows.append((rid, *triple))
            apls_values.append(triple[2])  # Final APLS value (third column)
        except Exception:
            continue

    if rows:
        # Write CSV summary
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["region_id", "apls_gt", "apls_prop", "apls"])
            writer.writerows(rows)
        print(f"Wrote summary CSV: {summary_csv}")
        
        # Compute and save final APLS statistics (like apls.py)
        if apls_values:
            import json
            final_apls = sum(apls_values) / len(apls_values)
            apls_data = {
                "apls": apls_values,
                "final_APLS": final_apls
            }
            with apls_json.open("w", encoding="utf-8") as f:
                json.dump(apls_data, f, indent=2)
            
            print(f"Final APLS: {final_apls:.6f}")
            print(f"Processed {len(apls_values)} regions")
            print(f"Wrote APLS statistics: {apls_json}")
        else:
            print("No valid APLS values found for final computation.")
    else:
        print("No result files found to summarize.")


if __name__ == "__main__":
    # Do not auto-execute heavy jobs on import; this runs only when called directly.
    main()


