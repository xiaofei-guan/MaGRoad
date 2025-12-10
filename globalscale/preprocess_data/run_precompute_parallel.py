import argparse
import os
import subprocess
import sys
from typing import List, Tuple


def compute_ranges(total: int, num_jobs: int, start_offset: int = 0) -> List[Tuple[int, int]]:
    if num_jobs <= 0:
        raise ValueError('num_jobs must be > 0')
    if total < 0:
        raise ValueError('total must be >= 0')
    base = total // num_jobs
    rem = total % num_jobs
    ranges: List[Tuple[int, int]] = []
    begin = start_offset
    for i in range(num_jobs):
        size = base + (1 if i < rem else 0)
        end = begin + size
        ranges.append((begin, end))
        begin = end
    return ranges


def count_contiguous_ids(input_dir: str, dataset_name: str = 'wildroad') -> int:
    """Count how many gt graph files exist from 0..N-1 contiguously.

    Stops at the first missing id.
    """
    if dataset_name == 'globalscale':
        # Pattern from dataset.py line 361
        filename_tmpl = 'region_{}_refine_gt_graph.p'
    else:
        # Default wildroad pattern
        filename_tmpl = 'gt_graph_{}.pickle'

    i = 0
    while True:
        p = os.path.join(input_dir, filename_tmpl.format(i))
        if not os.path.exists(p):
            # Try checking if it's just a limit issue or actual end
            # For robustness, check next one too? No, assumes contiguous.
            break
        i += 1
    
    if i == 0:
        print(f"[WARN] No files found in {input_dir} with pattern {filename_tmpl.format('0')}")
        
    return i


def main():
    parser = argparse.ArgumentParser(description='Run multiple GLG precompute jobs in parallel.')
    parser.add_argument('--num_jobs', type=int, default=24, help='number of parallel jobs')
    parser.add_argument('--total_tiles', type=int, default=3338, help='total number of tiles to process')
    parser.add_argument('--start_offset', type=int, default=0, help='starting tile id (inclusive)')
    parser.add_argument('--base_output_dir', type=str, required=True, help='base directory to store outputs')
    parser.add_argument('--input_dir', type=str, required=True, help='directory containing gt_graph_{id}.pickle files')
    parser.add_argument('--python_exe', type=str, default=sys.executable, help='python executable to run dataset.py')
    parser.add_argument('--dataset_script', type=str, default=None, help='path to dataset.py; defaults to sibling file')
    parser.add_argument('--dataset_name', type=str, default="wildroad", help='dataset name')
    parser.add_argument('--overwrite', action='store_true', help='pass --overwrite to dataset.py jobs')
    parser.add_argument('--dry_run', action='store_true', help='print commands without executing')
    args = parser.parse_args()

    dataset_script = args.dataset_script
    if dataset_script is None:
        dataset_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset.py')
    if not os.path.exists(dataset_script):
        raise FileNotFoundError(f'dataset.py not found at {dataset_script}')

    os.makedirs(args.base_output_dir, exist_ok=True)

    # Auto-detect total tiles if not provided or negative
    total_tiles = args.total_tiles
    if total_tiles < 0:
        total_tiles = count_contiguous_ids(args.input_dir, dataset_name=args.dataset_name)
    print(f"[INFO] Total tiles detected: {total_tiles} (dataset: {args.dataset_name})")
    ranges = compute_ranges(total_tiles, args.num_jobs, start_offset=args.start_offset)

    procs: List[subprocess.Popen] = []
    log_files = []
    try:
        for job_id, (start_id, end_id) in enumerate(ranges):
            out_dir = os.path.join(args.base_output_dir, f'GLG_{job_id}')
            os.makedirs(out_dir, exist_ok=True)
            cmd = [
                args.python_exe,
                dataset_script,
                '--dataset_name', args.dataset_name,
                '--start_id', str(start_id),
                '--end_id', str(end_id),
                '--output_dir', out_dir,
                '--input_dir', args.input_dir,
            ]
            if args.overwrite:
                cmd.append('--overwrite')

            print(' '.join(cmd))
            if args.dry_run:
                continue

            log_path = os.path.join(out_dir, 'precompute.log')
            f = open(log_path, 'w')
            log_files.append(f)
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            procs.append(p)

        if args.dry_run:
            return

        exit_codes = []
        for p in procs:
            exit_codes.append(p.wait())

        failed = [(i, c) for i, c in enumerate(exit_codes) if c != 0]
        if failed:
            print(f'Completed with failures: {failed}')
            sys.exit(1)
        else:
            print('All jobs completed successfully.')
    except KeyboardInterrupt:
        print('Interrupted, terminating all jobs...')
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                pass
        raise
    finally:
        for f in log_files:
            try:
                f.close()
            except Exception:
                pass


if __name__ == '__main__':
    main()


