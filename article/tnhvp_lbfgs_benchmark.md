# TNHVP versus nonredundant L-BFGS benchmark

## Benchmark status

The benchmark produced from revision `45e8e48` is withdrawn. Although its
command line selected `nonredundant_lbfgspp`, the implementation applied the
L-BFGS inverse-Hessian history in the full packed sparse-coefficient space.
It then subtracted projected gradients expressed in different accepted-point
quotient charts without transporting either the primal step or the dual
gradient change. Gauge canonicalization therefore corrupted the secant
history, and the resulting iteration counts were not a valid nonredundant
L-BFGS baseline.

The corrected implementation stores the accepted finite retraction
displacement and the full ambient gradient-covector change in the common
packed embedding. When a secant is used in a new accepted-point chart, it:

1. transports a step with the new chart's vector projection;
2. transports a gradient covector with the new chart's covector pullback;
3. checks positive curvature only after both quantities occupy the same
   target quotient chart; and
4. clears history only if the quotient rank changes, not when an equivalent
   orbital gauge representative changes.

## Corrected paired benchmark

The corrected benchmark used revision `09579bd` and binary SHA-256
`fc216e2c62c51bbce9878253bf439d703fb3924913a1355af124060c1134bf43`.
Slurm job `245912` ran on Hanhai25 with 32 CPU cores per calculation. Each
system ran TNHVP and L-BFGS sequentially on the same node, with alternating
method order between systems. Both methods used Davidson, the same input
orbitals, a projected-gradient infinity-norm threshold of `1e-3`, and an
adjacent-step energy threshold of `1e-7` hartree. Their initial energies agree
exactly.

| System | TNHVP steps | TNHVP SCF wall (s) | TNHVP final energy | TNHVP projected gradient | L-BFGS steps | L-BFGS SCF wall (s) | L-BFGS final energy | L-BFGS projected gradient | TNHVP/L-BFGS wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F2 | 6 | 0.383526 | -198.751155830526 | 2.10997520e-06 | 8 | 0.132126 | -198.751155821808 | 3.41375526e-04 | 2.90 |
| 241 | 11 | 8.699289 | -230.720590392874 | 2.80391992e-06 | 24 | 4.368567 | -230.720590368221 | 1.08715828e-04 | 1.99 |
| MnF2 | 15 | 55.976787 | -1348.893353224166 | 4.51189144e-06 | 98 | 27.725153 | -1348.893352802026 | 7.24461861e-04 | 2.02 |
| FeCl2 | 10 | 22.156342 | -2181.617635683505 | 5.56753711e-04 | 7 | 11.518017 | -2181.617634827449 | 9.14020185e-04 | 1.92 |
| 240 | 6 | 93.643589 | -343.446302353263 | 4.81382556e-07 | 10 | 4.972273 | -343.446302350844 | 3.26154740e-05 | 18.83 |

Peak RSS remains comparable:

| System | TNHVP peak RSS (KiB) | L-BFGS peak RSS (KiB) |
|---|---:|---:|
| F2 | 67456 | 55296 |
| 241 | 991608 | 1002048 |
| MnF2 | 575804 | 564488 |
| FeCl2 | 455352 | 436456 |
| 240 | 1901152 | 1856368 |

## Interpretation

The corrected result reverses the conclusion of the withdrawn benchmark.
TNHVP reduces the outer iteration count for F2, 241, MnF2, and 240, and reaches
a substantially tighter final gradient under the same stopping thresholds.
However, its accepted steps are currently too expensive: total TNHVP SCF time
is about twice the L-BFGS time for four systems and about nineteen times the
L-BFGS time for 240. FeCl2 also needs more TNHVP outer steps than L-BFGS.

Therefore the present data support a convergence-quality advantage, but not a
wall-time advantage, for TNHVP. A publication performance claim requires
reducing the HVP/coupled-response cost or avoiding response work whose expected
reduction cannot amortize its measured wall time. The corrected nonredundant
L-BFGS implementation is the baseline for all subsequent comparisons.

## Accuracy-triggered forcing regression

A 32-core Hanhai25 regression on 2026-09-19 tested the production forcing rule
derived in Section 11.7 of
`matrix_free_vbscf_orbital_optimization_theory.md`. These runs used the same
Davidson inputs and stopping thresholds as the corrected benchmark, but were
not executed as a new paired TNHVP/L-BFGS timing experiment; they are recorded
as an implementation regression rather than final article data.

| System | TNHVP steps | SCF wall (s) | Final energy | Projected gradient infinity norm | Peak RSS (KiB) |
|---|---:|---:|---:|---:|---:|
| F2 | 5 | 0.255586 | -198.751155830509 | 1.18593384e-05 | 67584 |
| MnF2 | 15 | 25.204288 | -1348.893353224099 | 2.49837891e-06 | 555384 |
| FeCl2 | 5 | 2.501624 | -2181.617636436763 | 5.06137414e-04 | 425048 |
| 240 | 7 | 20.601759 | -343.446302353175 | 3.80186083e-06 | 1911848 |

The FeCl2 iteration count decreases from 10 in the earlier TNHVP benchmark to
5, below the 7-step nonredundant L-BFGS baseline, while its SCF wall time falls
from 22.16 s to 2.50 s. MnF2 retains a 15-step count while its SCF time falls
from 55.98 s to 25.20 s. The 240 calculation requires one additional accepted
step but decreases from 93.64 s to 20.60 s. A fresh interleaved paired run is
still required before these wall-time changes are used as publication claims.
