# Historical VBSCF optimizer runs

`paired_runs.csv` is a machine-readable inventory of 261 Hanhai25 inputs, each
run with block-LBFGS and NEO. Of the 261 pairs, 258 have two completed,
converged results. The remaining three pairs have interrupted/incomplete logs
and are retained with blank result fields. `transition_metals.csv` holds a
separate six-run comparison with TNHVP. Each row points to its remote `run.out`
directory. Blank fields mean that a quantity was unavailable in the original
log; they are not zeros. These are single-run development measurements,
**not** publication-ready timings.

The `all-optimizers-c2776c4` panel pairs the two methods on the same inputs and
code revision. All 258 complete pairs used 32 threads, Davidson, and exact
Libcint integrals, and have matching initial energies and problem settings.
NEO takes fewer accepted steps in all 258 pairs (median reduction 2.4-fold),
but has lower end-to-end wall time in only 76 pairs (median wall-time speedup
0.81-fold). The median absolute final-energy difference is 2.1e-8 hartree; the
largest is 1.3e-5 hartree (`69333_vb`). Energy agreement must be checked before
interpreting each pair as an equivalent-solution speed comparison.

The `transition-metals-unified` panel additionally contains TNHVP, but its
exact executable revision was not recorded. It is a **separate** comparison;
do not pool wall times across panels without controlling version, node,
integral mode, eigensolver, and thread count. `accepted_steps` counts accepted
orbital updates, not the initial point. `peak_rss_kb` comes from GNU `time` and
is the maximum resident set size of the measured process. NEO and TNHVP action
fields name different operations and must not be treated as interchangeable.

The collector is [`tools/collect_optimizer_results.py`](../../tools/collect_optimizer_results.py).
The Python/Matplotlib plotter is
[`tools/plot_optimizer_comparison.py`](../../tools/plot_optimizer_comparison.py).
It reads `paired_runs.csv` and plots accepted iterations and end-to-end wall
time as point-and-line curves for ten named, converged reference cases. It
produces `iteration_time_curves.png` and `iteration_time_curves.svg`. Case
selection and sorting are encoded in the script; no numbers are hand-entered.
For example:

```sh
uv run --no-project --with matplotlib python tools/plot_optimizer_comparison.py \
  results/history/paired_runs.csv results/history/iteration_time_curves
```

The source scripts are retained so the CSV and figure can be regenerated and
audited. The raw Slurm outputs remain on Hanhai25; they are not copied into
this repository.
