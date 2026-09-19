# TNHVP versus L-BFGS benchmark

## Baseline definition

The production `lbfgs` backend is the nonredundant, support-preserving
first-order baseline. Its two-loop recursion uses the conventional scalar
initial inverse Hessian

$$
H_k^{(0)} = \gamma_k I,
\qquad
\gamma_k = \frac{s_{k-1}^{\mathrm T}y_{k-1}}
{y_{k-1}^{\mathrm T}y_{k-1}}.
$$

The accepted displacement and gradient-covector difference are stored in the
common packed embedding and transported into the current quotient chart before
the secant recursion is applied. Positive-curvature pairs satisfy

$$
s_i^{\mathrm T}y_i >
\sqrt{\epsilon_{\mathrm{mach}}}\,\lVert s_i\rVert_2\lVert y_i\rVert_2.
$$

TNHVP uses the same transported secant history, but its inverse action is
initialized by the positive local orbital-curvature block. This block is a
second-order preconditioner and is intentionally not part of the L-BFGS
baseline. TNHVP subsequently enriches the preconditioned L-BFGS direction with
matrix-free relaxed Hessian-vector products.

## Baseline audit

Earlier comparisons are withdrawn. Although the earlier backend used the
correct nonredundant chart and transported secants, it initialized the L-BFGS
two-loop recursion with the inverse local orbital-curvature block rather than
with $\gamma_k I$. The reported method was therefore preconditioned L-BFGS,
not a conventional first-order L-BFGS baseline.

An isolated calculation in which only this initial inverse action was changed
gave the following outer iteration counts:

| System | Orbital-block L-BFGS | Standard L-BFGS | XMVB L-BFGS |
|---|---:|---:|---:|
| F2 HAO | 8 | 26 | 26 |
| 7975 HAO | 9 | 57 | 89 |

The exact agreement for F2 identifies the hidden block preconditioner as the
cause of the anomalously small eight-step count. The remaining 7975 difference
is consistent with the use of a nonredundant quotient chart, a physical
retraction, and a monotone Armijo line search in XMVB-CPP; it is not caused by
the stopping thresholds, which are identical.

## Corrected paired validation

Slurm job `246480` ran the corrected standard L-BFGS and TNHVP implementations
sequentially on the same Hanhai25 node (`anode018`) with 32 CPU cores. Both
methods used the same input orbitals, exact Libcint integrals, dense structure
diagonalization, a projected-gradient infinity-norm threshold of `1e-3`, and
an adjacent-step energy threshold of `1e-7` hartree.

| System | Method | Outer steps | SCF wall / s | End-to-end wall / s | Final energy / hartree | Final projected gradient |
|---|---|---:|---:|---:|---:|---:|
| F2 HAO | standard L-BFGS | 26 | 0.244541 | 0.622779 | -198.751155768900 | 7.13992385e-4 |
| F2 HAO | TNHVP | 7 | 0.163298 | 0.522691 | -198.751155830497 | 2.94905488e-5 |
| 7975 HAO | standard L-BFGS | 57 | 15.570932 | 16.248405 | -285.733616613173 | 8.53372951e-4 |
| 7975 HAO | TNHVP | 7 | 2.854029 | 3.558416 | -285.733616922951 | 7.40072187e-5 |

TNHVP reduces the iteration count by factors of 3.7 and 8.1 for F2 and 7975,
respectively. It is also faster in this paired validation: the SCF-region
speedups are 1.50 and 5.46, while the end-to-end speedups are 1.19 and 4.57.
TNHVP used five block-HVP applications for each system.

These two systems validate the corrected method separation, but they are not a
publication benchmark set. Final article data require the complete molecular
suite, at least five measured repetitions after a warm-up, a fixed executable
checksum, identical eigensolver choices, and peak-RSS measurements from the
batch scheduler.

## Reporting rule

Future tables must distinguish the following methods explicitly:

1. standard nonredundant L-BFGS with $H_k^{(0)}=\gamma_k I$;
2. orbital-block-preconditioned L-BFGS, if retained as an additional strong
   quasi-Newton reference; and
3. TNHVP with the orbital-block inverse, transported secants, and exact
   matrix-free curvature correction.

An orbital-block-preconditioned calculation must never be labelled simply as
L-BFGS in an article table.
