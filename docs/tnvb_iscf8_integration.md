# TNVB / ISCF=8 integration

> Historical validation record. Since the 2026-10-17 merge of master
> `03d469cc`, the current optimizer is `Plbfgs` / `--optimizer-backend plbfgs`;
> native ISCF=5 is removed. Use ISCF=8 for current TNVB decks. The measurements
> below retain their original revision and are not revalidation of this merge.
> Current usage is documented in `tnvb_usage.md`.

Status (2026-10-12): implemented and verified for the scope below. Full C14
ISCF=8 convergence and reproducibility of independently optimized OEO weights
are not claimed.

## Scope and invariants

The accepted ISCF=5 baseline is commit `d8434be` on branch `tnvb`.
This extension permits the independent TNVB electronic backend to serve the
existing ISCF=8 nonredundant orbital-block L-BFGS optimizer. ISCF=7 (TNHVP)
and ISCF=9 (NEO) are not included. No tensor algorithm, truncation policy,
ordinary optimizer, preconditioner, line search, convergence default, or
integral kernel is replaced.

The existing input `ISCF=8` selects `BlockLbfgs`; the CLI override is
`--optimizer-backend block_lbfgs`. The TNVB CLI and optimizer guards allow
ISCF=5/8 while retaining one singlet root, HAO/OEO, dense H/S, and the
optimization requirement `TNSUMBOND>=TNBOND`. Only the same three TNVB
keywords are registered: `TNVB`, `TNBOND`, `TNSUMBOND`.

An evaluation still constructs unexpanded selected covalent/ionic VB MPSs,
H/S and structure coefficients. Its DIRECT total state supplies FULL
1/2/3-RDM and the overlap/integral adjoints. The ordinary C++ pullback returns
the orbital gradient. Exact and RI share this boundary; RI retains the
factorized AO reverse propagation.

The block optimizer uses the physical orbital frame, a selected-space
nonredundant chart and a one-electron orbital-block initial inverse.
`complete_active_space` remains false. Its transported secant history needs
no determinant expansion or `AcceptedPointContext`. Trial evaluations retain
the existing isolation and explicit accepted-point commit semantics.

## Stopping criteria and output

Defaults remain absolute accepted energy change `<1e-7 Eh` and gradient
tolerance `1e-3`. ISCF=8 applies the gradient tolerance to the **projected
reduced-gradient infinity norm**. ISCF=5 uses its existing raw-coordinate
L2 criterion. These are different norms, so equal numerical thresholds do
not mean identical stopping accuracy or terminal orbitals. The existing
block optimizer also accepts an initially small projected gradient; this
behavior is not modified.

Normal runs use `bin/xmvb-cpp.exe INPUT.xmi > OUTPUT.xmo 2>&1`. H/S,
coefficients, weights, orbitals, natural orbitals and population analyses
remain available. `ITMAX=0` constructs no total state/RDM and does not report
orbital convergence. `optimize_tnvb` remains a supplementary test executable,
not a production entry point. It selects input ISCF=5/8 explicitly, accepts
no tolerance overrides, and reports which gradient metric is used.

All current block-LBFGS Armijo trial points calculate a full gradient. An
eventual performance advantage comes from convergence/accepted-step behavior,
not from claiming rejected trials already skip 3-RDM. Energy screening of
trials would be a separate change.

## Verification procedure

Results and an installed matched executable/worker are isolated under:

`/pool1/home/chentz/apps/xmvb4.0/build/cpp-tnvb-iscf8-validation`

The original installation and ISCF=5 C14 acceptance files are not overwritten.

Fast CTest covers input/CLI selection, CLI override, ENERGY without 3-PDM,
full-gradient postprocessing/Molden, continued finite fit warnings,
unsupported TNHVP/NEO, failed-trial isolation, and an actual energy-lowering
accepted block step. The accepted-step test checks that no determinants/CAS
are created and that projected gradients and TNVB diagnostics are present.

The explicit numerical driver `tests/tnvb/validate_iscf8.py` runs:

1. Benzene, five covalent/ionic structures, HAO and OEO, D/S=128/256,
   ordinary block-LBFGS versus TNVB block-LBFGS, up to 500 accepted steps,
   using unchanged defaults. Exact and RI are separate tests.
2. C14, four Kekule structures, HAO, D/S=200/400, **one accepted step**
   with both electronic backends. This checks orbital retraction/pullback
   consistency at larger active space, not full C14 convergence.
3. A complete HAO benzene run through the installed production executable
   in each integral model, retaining normal `.xmo` and analyses.

Required evidence includes actual accepted steps, monotonic accepted energies,
original thresholds, benzene convergence, endpoint H/S/E/gradient and
phase-aligned coefficient/weight comparisons, final normalized-orbital energy
consistency, fit error and RDM identities. C14 matrix errors are separately
reported under the existing user-approved C14 profile. A deliberate one-step
C14 `passed=false` in the full-convergence comparison is not relabeled as
convergence: the driver applies separate bounded-step requirements and records
`convergence_required=false`.

The numerical optimized-energy comparison target stays `1e-6 Eh`. A full
C14 ISCF=8 optimization or larger systems are not proven by these tests.

## Current results

The expanded CLI and accepted-trial smoke tests passed (2/2, 121.98 seconds).
After rebuilding all targets, ON CTest passed 15/15 (125.69 seconds), OFF
passed 10/10 (1.97 seconds). The original C14 snapshot's five SHA256 checks
also passed. Numerical jobs `2108440` (exact) and `2108441` (RI) use separate
16-CPU, 64-GiB allocations; each uses four one-thread H/S workers and 16 RDM
threads. Both jobs completed with exit 0 in 15:08 and 17:12, respectively.
Each saved summary records four passed cases: benzene HAO/OEO full
optimization, bounded C14 and the normal production benzene CLI. These are
eight separate numerical/output checks in addition to the 15/15 ON tests.

All four benzene comparisons completed and converged at the original
thresholds:

| Orbitals / model | Ordinary / TNVB steps | Terminal energy difference (Eh) | Terminal coefficient difference | Terminal weight difference |
|---|---:|---:|---:|---:|
| HAO / exact | 9 / 9 | 1.7053e-13 | 4.7601e-15 | 1.5543e-15 |
| HAO / RI | 9 / 9 | 1.1369e-13 | 4.0523e-15 | 1.9984e-15 |
| OEO / exact | 52 / 57 | 5.7055e-9 | 1.2799e-3 | 9.3134e-4 |
| OEO / RI | 113 / 124 | 2.0330e-7 | 4.7708e-2 | 3.5653e-2 |

The OEO coefficient/weight columns compare **independently optimized orbital
sets**, not two backends evaluated at the same orbitals. Their terminal
coefficients and weights are not equal despite passing the approved energy
target. At either endpoint, same-orbital energy, H/S, gradient and
coefficients/weights match at about 1e-14 or better. The RI OEO initial
accepted iterations 0--3 also agree in energy/gradient to roundoff. This
supports a numerically sensitive optimization trajectory rather than an
electronic-backend mismatch; it does not prove the endpoint weight variation
is harmless or fully identify its cause. Energy/gradient stopping thresholds
alone do not constrain all structural weights. No convergence parameter was
tightened to hide the variation.

RI and exact are evaluated within their own integral models. Their absolute
energies here must not be used as an RI accuracy benchmark. Small-benzene TNVB
runs are slower than their tiny determinant references; these tests establish
integration, not large-active-space speedup.

### C14 one-step comparison

Both ordinary and TNVB block-LBFGS accepted exactly one step, reduced the
energy by about 1.09 Eh and terminated at the intentional `ITMAX=1` cap.
Neither run is labeled converged.

| Model | Ordinary / TNVB optimization wall (s) | Step-end energy difference (Eh) | Coefficient / weight difference | Same-orbital gradient error |
|---|---:|---:|---:|---:|
| Exact | 325.009 / 59.111 | 8.7084e-11 | 2.9841e-9 / 3.9434e-9 | 2.4092e-9 |
| RI | 365.139 / 56.932 | 7.6398e-11 | 2.9780e-9 / 3.9359e-9 | 2.4091e-9 |

The optimization times include the initial gradient and one accepted orbital
step; they exclude the supplementary two-endpoint cross-checks. A TNVB
FULL RDM evaluation takes about 25--27 seconds, of which 21--23 seconds are
3-RDM. Much of the complete 15--17 minute validation job is ordinary
determinant work, not TNVB worker latency. These timings do not establish
full-optimization speedup or a universal per-step cost.

At either endpoint, energy errors are approximately 9.5e-11 Eh, coefficient
errors approximately 3.1e-9 and weight errors approximately 4.1e-9. Fit energy
errors are approximately 1.7e-12 Eh, and all RDM identity errors are below
1e-8. H error is approximately 1.52e-8 Eh and S error approximately 6.96e-10;
they are explicitly retained diagnostics under the user-approved C14 profile,
not substituted for the benzene matrix tolerances. Final normalized-orbital
export energy errors are below 1e-12 Eh.

`c14_hao_{exact,ri}_iscf8.json` retains `passed=false` in its **full convergence**
field because neither one-step run is converged. `summary_{exact,ri}.json`
separately records the successful bounded-step test with
`convergence_required=false`. No convergence threshold or matrix tolerance
was changed to obtain these results.

### Normal C14 xmi / xmo

The supplemental `2108457` job invokes the installed normal
`bin/xmvb-cpp.exe` directly on `inputs/c14_hao_{exact,ri}_iscf8.xmi`, with
`ITMAX=1`, and writes `c14_hao_{exact,ri}_iscf8_cli.xmo`. It performs no
determinant reference optimization or endpoint comparisons. The command is:

```sh
./bin/xmvb-cpp.exe inputs/c14_hao_exact_iscf8.xmi \
  --standard-two-electron-mode exact > c14_hao_exact_iscf8_cli.xmo 2>&1
```

This deliberate one-step run returns the unchanged normal CLI's exit 2 and
prints `VBSCF did not converge` / `Termination reason: max_iterations`.
Those messages mean the iteration cap was reached, not a failed tensor kernel.
Slurm consequently marks each `srun` step `FAILED (2:0)`; the validation batch
job is `COMPLETED (0:0)` only after checking the actual accepted step, energy,
original convergence settings, no structure expansion, H/S,
coefficients/weights, orbitals, natural orbitals, population analyses and
Molden. It does not merely ignore the nonzero exit. Detailed checks are in
`c14_hao_{exact,ri}_iscf8_cli.json`.

Both CLI checks passed; the complete batch took 2:04. End-to-end exact/RI
times are 62.710/60.601 seconds. Energies differ from the comparison tool's
TNVB endpoints by 3.41e-13/4.55e-13 Eh. Both Molden files contain all 54
inactive and active orbitals. Normal orbital, natural-orbital and population
sections were inspected in both `.xmo` files.

Binary/worker, six generated deck and four relevant source SHA256 hashes are
recorded in `snapshot.sha256` in the validation directory. Both worker hashes
are unchanged from the ISCF=5 baseline. The original acceptance installation
is not replaced; the ISCF=8-capable executable is in the new validation
directory above. The ISCF=8 extension remains uncommitted after the accepted
ISCF=5 commit `d8434be`.

## Completion audit

| Requirement | Evidence |
|---|---|
| Input ISCF=8 / CLI block override reaches TNVB optimizer | Normal C6/C14 xmi runs, extended `tnvb_cli` CTest |
| Actual block step, selected VB space, no determinant/HVP substitution | `tnvb_trials` accepted-step assertions, zero expansion in normal C14 output, shared one-electron chart path |
| HAO/OEO, exact/RI energy/gradient correctness | Four converged benzene comparisons; same-orbital endpoint H/S/E/gradient/coefficient/weight checks |
| Larger active-space retraction/pullback | Both bounded C14 comparisons and normal C14 main-program runs |
| Preserve original convergence defaults and ISCF=5 / ordinary paths | Clean diffs for optimizer backends/options, preconditioners, line search, tensor/integral kernels; ON 15/15 and OFF 10/10; original C14 snapshot intact |
| ENERGY skips total MPS/RDM; retain gradient-mode analyses | Extended CLI CTest with forbidden 3-PDM; normal C6/C14 xmo and Molden outputs |
| Fit/identity/export consistency | Numeric JSONs and the two C14 CLI verification JSONs |
| Document use and limitations without claiming untested C14 convergence | This report and `tnvb_usage.md`; one-step cap and OEO terminal weight variation explicitly retained |
