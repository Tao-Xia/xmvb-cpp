# TNHVP versus nonredundant L-BFGS benchmark

## Reproducibility contract

This benchmark was run on Hanhai25 on 2026-09-19 with the executable built
from code revision `45e8e48`.  Revision `e23a6f3` changes documentation only
and therefore has the same executable code.  Every calculation used:

- 32 CPU cores;
- the Davidson structure eigensolver;
- a projected-gradient infinity-norm tolerance of `1e-3`;
- an absolute adjacent-step energy tolerance of `1e-7` hartree;
- the same input orbitals and integral representation for both optimizers;
- nonredundant L-BFGS for `ISCF=5` and matrix-free TNHVP for `ISCF=7`.

The optimizer backend was selected through the command line without editing
the input deck.  The initial energy agrees exactly between the two methods in
every reported decimal digit:

| System | TNHVP initial energy | L-BFGS initial energy |
|---|---:|---:|
| F2 | -198.4883735546 | -198.4883735546 |
| 241 | -230.4744717289 | -230.4744717289 |
| MnF2 | -1348.2999515874 | -1348.2999515874 |
| FeCl2 | -2181.6176319113 | -2181.6176319113 |
| 240 | -343.0756452126 | -343.0756452126 |

## Complete convergence results

`SCF wall` is the optimizer time reported internally by the executable.  Peak
RSS is from `/usr/bin/time -v`.  The raw wall ratio divides the complete
L-BFGS wall time by the TNHVP wall time.  MnF2 did not converge at the 2000-step
input limit, so its ratio is a strict lower bound rather than a converged
speedup.

| System | TN steps | TN SCF wall (s) | TN final energy | TN projected gradient | L-BFGS steps | L-BFGS SCF wall (s) | L-BFGS final energy | L-BFGS projected gradient | Raw wall ratio | TN RSS (KiB) | L-BFGS RSS (KiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F2 | 6 | 0.565847 | -198.7511558305 | 2.10997520e-06 | 177 | 6.332379 | -198.7511553889 | 9.50643094e-04 | 11.19 | 67208 | 69632 |
| 241 | 11 | 8.468393 | -230.7205903929 | 2.80391992e-06 | 585 | 408.715558 | -230.7205850716 | 9.58983833e-04 | 48.26 | 996784 | 976020 |
| MnF2 | 15 | 58.210345 | -1348.8933532242 | 4.51189144e-06 | 2000, not converged | 4068.776327 | -1348.8587180937 | 2.97286272e-01 | greater than 69.90 | 578524 | 578268 |
| FeCl2 | 10 | 17.021230 | -2181.6176356835 | 5.56753711e-04 | 266 | 118.591612 | -2181.6176358328 | 8.09492595e-04 | 6.97 | 457008 | 457308 |
| 240 | 6 | 86.637815 | -343.4463023533 | 4.81382556e-07 | 819 | 3034.909086 | -343.4463017827 | 7.49850358e-04 | 35.03 | 1874760 | 1857784 |

The long L-BFGS calculations ran in independent Slurm allocations after an
initial paired 200-step panel.  Their raw wall ratios therefore include node
performance variation.  Iteration counts, convergence status, energies, and
gradients are independent of that variation.

## Same-node paired lower bounds

Each system in the initial panel ran both methods sequentially in the same
Slurm allocation.  Method order was alternated between systems.  L-BFGS was
stopped at 200 steps in this paired panel; a nonconverged ratio is consequently
a lower bound on the time needed for convergence.

| System | TN status | TN wall (s) | L-BFGS status at 200 steps | L-BFGS wall (s) | Paired wall lower bound |
|---|---|---:|---|---:|---:|
| F2 | converged in 6 | 0.565847 | converged in 177 | 6.332379 | 11.19 |
| 241 | converged in 11 | 8.468393 | not converged | 136.201362 | greater than 16.08 |
| MnF2 | converged in 15 | 58.210345 | not converged | 404.616980 | greater than 6.95 |
| FeCl2 | converged in 10 | 17.021230 | not converged | 71.688134 | greater than 4.21 |
| 240 | converged in 6 | 86.637815 | not converged | 538.604596 | greater than 6.22 |

Using the paired first-200-step L-BFGS cost to normalize away node speed and
multiplying by the complete L-BFGS iteration count gives estimated total
ratios of 47.04 for 241, 5.60 for FeCl2, and 25.46 for 240.  The corresponding
MnF2 lower bound is 69.51.  These estimates agree in scale with the raw
complete-run ratios and show that the conclusion is not caused by a favorable
TNHVP compute node.

## Interpretation

TNHVP outer steps are not uniformly cheaper.  From the same-node panel, one
TNHVP step costs approximately 1.1--5.4 times one L-BFGS step, depending on the
system.  The overall gain comes from reducing the number of outer iterations
by one to two orders of magnitude:

- F2: 6 versus 177 steps;
- 241: 11 versus 585 steps;
- FeCl2: 10 versus 266 steps;
- 240: 6 versus 819 steps;
- MnF2: 15 versus more than 2000 steps.

The 240 trajectory also demonstrates why both convergence criteria are
necessary.  L-BFGS reached adjacent energy changes below `1e-7` hartree while
its projected gradient remained several times above `1e-3`; accepting an
energy-only criterion would have produced false convergence.  TNHVP reached
both criteria in six steps.

Peak RSS differs by no more than a few percent between the methods for this
panel.  The accepted-point integral and structure data dominate memory, so the
matrix-free second-order subspace does not introduce a material peak-memory
penalty at these sizes.

The wall times are single-run measurements.  Publication tables should repeat
the converged calculations on otherwise idle nodes and report a median and
dispersion, while retaining the deterministic iteration counts and final
accuracy checks reported here.
