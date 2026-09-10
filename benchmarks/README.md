# Reproducible optimizer benchmarks

`tools/run_optimizer_benchmarks.py` is the canonical entry point for TNHVP
paper data. It runs the current matrix-free optimizer and nonredundant L-BFGS
from identical input decks, preserves the complete program output, writes the
accepted-step TNHVP diagnostics, and records the Git revision, executable and
input checksums, host, thread settings, wall time, and maximum resident memory.

Example:

```bash
python3 tools/run_optimizer_benchmarks.py \
  testdata/vbscf/F2.xmi \
  testdata/vbscf/241_VBSCF.xmi \
  testdata/vbscf/MnF2.xmi \
  testdata/vbscf/FeCl2.xmi \
  --output benchmarks/results/<machine>-<commit> \
  --backends tnhvp lbfgs \
  --repeats 5 \
  --omp-threads 32 \
  --blas-threads 1
```

The output directory must be empty. A completed dataset contains
`manifest.json`, a unified `summary.tsv`, raw `.out`, `.err`, and `.time`
files, plus one `.tnhvp.tsv` file for every matrix-free run. Raw files are the
source of record; article tables should be generated from them rather than
edited by hand.

The tracked `local_baseline_20260907` directory is a historical debugging
record from before the corrected quotient-space implementation. It must not be
used as article data.
