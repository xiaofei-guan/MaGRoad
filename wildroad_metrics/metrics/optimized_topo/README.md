Optimized TOPO (Single-Pair, No extra I/O)

Usage
- Input graphs are the same dict format as the original (`x -> [neighbors]` in XY pixel coordinates).
- This implementation handles only one pair of graphs per run and prints exactly one line:
  - precision=... overall-recall=...

Run
```bash
python -m metrics.optimized_topo.main \
  -graph_gt path/to/gt.p \
  -graph_prop path/to/prop.p \
  -matching_threshold 0.00010 \
  -interval 0.00005 \
  -output runs/topo_108.txt
```

Notes
- Fully self-contained: no dependency on the old codebase. Core logic ported and optimized.
- Hot paths vectorized and parallelized (ThreadPool for per-start computations).
- No intermediates; prints the final line, and if `-output` is given, also saves it to the specified txt.
- Matches original algorithm semantics, including direction thresholds, tunnel handling hooks, and coverage-based overall recall.


