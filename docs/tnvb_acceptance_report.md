# Independent TNVB final acceptance report — 2026-10-12

> Historical acceptance evidence, not acceptance of the 2026-10-17 master
> merge. Current TNVB uses ISCF=8/P-LBFGS; native ISCF=5 has been removed by
> master. See `tnvb_usage.md` for the current interfaces and input examples.

Status: implementation and acceptance complete under the user-confirmed
acceptance policy below. All results refer to the `tnvb` branch, based on `df85712`.
Changes remain uncommitted. No legacy XMVB numerical source is modified by
this integration. No C20 or larger acceptance calculation has been run.

The user explicitly confirmed the optimized terminal-energy difference target
is 1e-6 Eh, without changing optimizer convergence parameters. The original
L-BFGS energy-change/gradient-L2 limits remain 1e-7 Eh / 1e-3. Benzene
same-orbital gates remain unchanged. C14 H/S errors are separately reported
diagnostics, under the user's explicit approval; finite values and its energy,
gradient, coefficient/weight gates remain required. No stricter-convergence
historical diagnostic supplies formal acceptance. Original numerical evidence
and its old pass/fail flags are retained without rewriting.

## Evidence locations

| Item | Path |
|---|---|
| ON build and JSON/log evidence | `/pool1/home/chentz/apps/xmvb4.0/build/cpp-tnvb` |
| OFF build and ordinary regression | `/pool1/home/chentz/apps/xmvb4.0/build/cpp-tnvb-baseline` |
| Separate install | `/pool1/home/chentz/apps/xmvb4.0/build/install-cpp-tnvb` |
| Frozen C14 validation binaries/worker/inputs/jobs | `/pool1/home/chentz/apps/xmvb4.0/build/cpp-tnvb-c14-validation` |
| Benzene HAO/OEO fixtures | `testdata/tnvb/C6_random5_{hao,oeo}.xmi` |
| C14 four-Kekulé fixture | `testdata/tnvb/C14_4k_hao.xmi` |
| Gradient derivation and conventions | `docs/tnvb_gradient_conventions.md` |
| Build/run instructions and tools | `docs/tnvb_usage.md` |
| Final numerical audit, all 24 required checks passed | `build/cpp-tnvb/numerical_acceptance_summary.json` under the evidence root |
| Refreshed OFF/ON regressions | `test-final-delivery.log` in each independent build |

JSON has 17-digit values. H/S comparisons use individually normalized structure
coordinates; the root's overall phase is aligned before comparing coefficients.
Weights include Coulson–Chirgwin, Löwdin, inverse-overlap and renormalized values.
Reported differences are absolute maxima, except total energy differences.

## Implementation audit

| Requirement | Current evidence/status |
|---|---|
| Independent optional target; OFF has no Python/Block2 requirements | `src/tnvb`, `xmvb_tnvb`, root default-OFF option; final ordinary OFF regression passes all 10 tests (2.04 s). |
| Exactly three new input keywords | Parser registers `TNVB`, `TNBOND`, `TNSUMBOND`; no old-keyword dispatch/compatibility added. |
| Preserve ordinary defaults | `tnvb_input` checks ordinary ITMAX default 2000, Davidson and expanded determinants. |
| TNVB raw structures, no expansion and not CAS | Loader condition; `tnvb_input` and all comparison tools assert empty determinant occupations/terms and zero expansion time. |
| Final ITMAX=0 ENERGY, no total MPS/RDM | `tnvb_cli`: missing S accepted, CLI override tested with forbidden 3-PDM, no SUM/RDM logs. |
| Real dense solver algorithm/capabilities | Worker dense generalized solve; C++ norm/residual checks; CLI rejects explicit unsupported solver/optimizer before worker computation. |
| Public active derivatives and common HAO/OEO/RI pullback | Mathematical adapter plus four same-orbital and four finite-difference cases below. |
| No determinant HVP cache for TNVB | Evaluator and trial tests assert absent `AcceptedPointContext`; ordinary default keeps its original forward/cache route. |
| FULL 3-RDM stays inside worker | Boundary transfers P (K²), Pi (K⁴), D3G (K²); no rank-six response buffer. |
| Fit discrepancy warning only | `tnvb_cli` deliberately truncates the total rank-21 benzene MPS to 16: warning appears, RDM/gradient continue; old sum-energy environment setting does not stop it. |
| RDM 1e-8 not relaxed | Child forces 1e-8 and C++ rechecks; passing tests include inherited loose setting ignored. |
| Trial/accepted state and failure isolation | `tnvb_trials`: energy-only trial, lazy completion, rejected nonfinite trial, actual FULL worker failure after H/S, subsequent successful completion and explicit commit. Accepted orbitals, energy/history and H/S remain unchanged until commit. |
| Worker and environment isolation | Own staged/installed worker, posix_spawn private env, unique scratch/cache; preserve inherited preloads, child-only matched numerical runtime. |
| Installed executable usable outside source/build | Full CLI ENERGY/gradient/output regression passes from `/tmp` with LD_LIBRARY_PATH absent; finds installed worker and basis assets. Build/install executable and both worker hashes match. |
| Preserve production structure/orbital output | CLI tests check H/S, coefficients, weights and orbital sections; ENERGY has a distinct completion label and no gradient/convergence claim. Extended CLI tests also retain Molden export (all 24 benzene orbitals), natural orbitals and population analysis after full gradients, and skip density analysis in ENERGY. |
| Full benchmark/report deliverables | Benzene HAO/OEO and C14 exact/RI complete optimizations pass the user-confirmed policy. Both C14 JSON reports say `passed=true`; the independent numerical audit passes 24/24. Original historical flags are retained. |

ON `ctest` final measured result: **15/15 passed**, 87.45 s. The five added tests
are adapter, input, CLI, trial-isolation and saved-evidence audit tests. The ordinary ten-test set is
unchanged. An earlier trial-isolation test deliberately supplied NaN and exposed
an OpenMP-region exception abort; TNVB now rejects nonfinite coordinates before
parallel preparation, and the same test passes. Ordinary numerical kernels
were not changed to obtain that pass.

After removing the unused legacy protocol/entrypoint, ON regressions again
pass 14/14 (54.90 s). A separate reversed-pair test exposed a genuine phase
adapter defect: energy/weights matched, but H and S rows changed sign relative
to the C++ CSF convention. The new adapter canonicalizes each pair's two
members without changing its state or pair order. The same test now passes
with max H/S errors 2.0e-15/5.0e-16 and gradient error 3.5e-15; it is also
covered by `tnvb_trials`. No ordinary expansion/kernel was changed.

The execution host has glibc 2.17 and an AVX2 CPU. A fixed SKYLAKEX OpenBLAS
caused SIGILL; the existing LP64 dynamic-architecture OpenBLAS resolves it.
GCC 13 and configured matching runtimes/dependencies produce executable tests
on this host. The separate install does not overwrite an existing installation.
The extended CLI output/export regression passed independently in 48.16 s.
A subsequent OFF/ON test refresh without the GCC module exposed a shared
ordinary `cli_options` test segfault; `ldd` showed the shell's Anaconda C++
runtime. Loading the configured GCC 13 runtime makes that test pass without
source changes. Both attempted test logs are retained; the matched-runtime
full-suite refresh passed OFF 10/10 (3.09 s) and ON 14/14 (77.60 s), recorded
separately in `test-final-matched-runtime.log` in each build. The failed
`test-final-output-audit.log` files remain intact. The independent installed
executable additionally passes the expanded CLI suite from `/tmp` without
LD_LIBRARY_PATH, including ENERGY's absent RDM and full-gradient density
analysis/Molden output. No production source changed during this output audit.
The later `test-worker-failure-audit.log` records 14/14 passes in 68.98 s
after extending trial isolation to a real child failure: an explicitly
forbidden FULL 3-PDM call makes the worker fail after constructing H/S. The
exception propagates without overwriting accepted data, and a subsequent
unrestricted gradient evaluation completes successfully. This is distinct
from the earlier NaN rejection before any worker is launched. Only the test
source changed; the C14 validation installation retained its frozen snapshot.

## Benzene: same-orbital ENERGY/gradient

Five explicit structures contain one covalent and four ionic structures, with
1, 2 and 3 doubly occupied active orbitals among the latter. D/S=128/256.
The explicit determinant reference has 19 unique determinants; TNVB has zero.

| Orbitals/integrals | ΔE (Eh) | max ΔH | max ΔS | max Δc | max Δweight | max Δgradient |
|---|---:|---:|---:|---:|---:|---:|
| HAO exact | 0 | 2.00e-15 | 5.00e-16 | 4.82e-15 | 2.00e-15 | 4.94e-15 |
| HAO RI | 0 | 3.55e-15 | 5.00e-16 | 4.15e-15 | 2.11e-15 | 5.27e-15 |
| OEO exact | 0 | 6.22e-15 | 8.33e-16 | 2.55e-15 | 1.11e-15 | 5.75e-15 |
| OEO RI | 0 | 4.72e-15 | 8.33e-16 | 5.26e-15 | 1.55e-15 | 5.88e-15 |

All four meet the requested fixed-orbital gates. Separate HAO ENERGY reports
also pass. The OEO fixture exports the actual HAO physical guess into full-AO
GUS; merely replacing ORBTYP in the original autoVB guess produces duplicate
active columns and a singular overlap, and was correctly rejected.

## Benzene: directional finite differences

Every case checks S, h, packed g, inactive orbitals and active orbitals at
1e-4, 3e-5 and 1e-5. At least two adjacent steps must have error <=1e-6;
the table gives the worst of all 15 measurements, not a chosen best step.

| Case | Worst directional error | Outcome |
|---|---:|---|
| HAO exact | 3.94e-8 | Pass |
| HAO RI | 4.52e-8 | Pass |
| OEO exact | 1.72e-8 | Pass |
| OEO RI | 2.35e-8 | Pass |

The frozen legacy worker interop test receives identical C++ active integrals.
All four HAO/OEO × exact/RI cases have zero differences in H/S, root,
coefficients, P, Pi and D3G. They use an explicit independent test bridge,
not a production fallback to the old source.

## Benzene: full ISCF=5 optimization

Identical initial physical orbitals, dense solving and original L-BFGS settings:
energy tolerance 1e-7 Eh, raw gradient L2 tolerance 1e-3. No line-search,
history or convergence algorithm modification.

| Case | Reference / TNVB accepted steps | Final ΔE | Final max Δc | Final max Δweight | Reference / TNVB wall time |
|---|---:|---:|---:|---:|---:|
| HAO exact | 52 / 52 | 2.84e-13 | 1.74e-12 | 4.02e-13 | 20.62 / 177.76 s |
| HAO RI | 54 / 54 | 3.41e-13 | 1.79e-13 | 1.01e-13 | 3.09 / 157.05 s |

Both HAO cases converge under the unchanged criteria and pass endpoint
cross-evaluation. Re-evaluating the exported TNVB physical orbitals reproduces
the reported energies within 5.7e-14 Eh (exact) and zero displayed difference
(RI) in machine-readable results. These tiny systems verify correctness;
they do not demonstrate a TNVB speed advantage.

At a 200-step cap, **both ordinary and TNVB OEO remain unconverged**:

| Case | Reference / TNVB gradient L2 | Final ΔE | Classification |
|---|---:|---:|---|
| OEO exact | 2.29e-3 / 3.12e-3 | 4.13e-8 | Not passed |
| OEO RI | 3.00e-3 / 2.81e-3 | 3.23e-7 | Not passed |

At both endpoints of each OEO run, cross-backend H/S and full gradients still
agree: maximum gradient discrepancies are below 1.6e-14. This distinguishes
slow/path-sensitive raw-coordinate optimization from an electronic-backend
gradient mismatch. The finite-difference and endpoint results do not establish
full optimization convergence. The separately bounded 500-step OEO checks have
now completed under the original tolerances:

| Case | Reference / TNVB steps | Reference / TNVB gradient L2 | Final ΔE | Classification |
|---|---:|---:|---:|---|
| OEO exact | 243 / 239 | 6.76e-4 / 8.10e-4 | 1.27e-7 | Both converged; passes user-updated 1e-6 target |
| OEO RI | 239 / 256 | 9.40e-4 / 8.60e-4 | 7.68e-7 | Both converged; passes user-updated 1e-6 target |

Both endpoints pass same-orbital cross-evaluation (gradient differences below
1.3e-12); exported physical orbitals reproduce each backend's own final energy.
This establishes different finite-tolerance terminal points, not distinct true
minima. The original reports retain `passed=false` under the former 1e-8 Eh
energy-match target; their data are not rewritten. The user has explicitly
confirmed the terminal-energy difference target is 1e-6 Eh, with convergence
parameters unchanged. Independent reassessment therefore passes both original-
setting OEO optimizations: their convergence, endpoint gradients, fit/RDM and
export checks also pass. This update affects only the cross-backend optimized-
energy comparison, not same-orbital gates or optimizer stopping conditions.
A former test-only
extension exercised stricter existing energy/gradient options identically for
both benzene backends, explicitly labeled `stricter_diagnostic`; production
defaults and the ordinary numerical kernels were unchanged. The user has now
explicitly required unchanged convergence parameters. That extra-argument
interface is removed: these saved diagnostics are historical only and cannot
supply formal acceptance. Original results and the C14 200-step cap remain.

The supplementary OEO runs with the existing tighter options E=1e-10 Eh and
gradient L2=1e-5 have now completed:

| Case | Final ΔE | Final max Δc | Final max Δweight | Classification |
|---|---:|---:|---:|---|
| OEO exact | 3.75e-11 | 1.73e-5 | 8.75e-6 | Supplementary optimization energy match passed |
| OEO RI | 4.83e-12 | 1.23e-5 | 7.13e-6 | Supplementary optimization energy match passed |

Both converge and pass endpoint full-gradient cross-evaluation. Final c/weight
differences between independently optimized orbitals are not the same-orbital
coefficient test: the latter still passes its 1e-8 limit. The RI diagnostic
uses 424/423 reference/TNVB steps and 70.75/1161.87 s. Reports are explicitly
`stricter_diagnostic`, not replacements for original-tolerance results.

A read-only comparison of the original accepted-energy histories finds that
the exact trajectories first differ by more than 1e-10 Eh at accepted step 94,
and by more than 1e-8 Eh at step 136; RI first crosses those values at steps
102 and 144. Initial energies are identical and early exact differences are
approximately 1e-13 Eh. Matching step indices does not establish identical
orbitals, so these are trajectory diagnostics, not electronic-backend errors.
This supports investigating late numerical/line-search sensitivity without
changing convergence settings; it does not establish an algorithmic fix.
Endpoint same-orbital checks and the former energy-match failures remain
independently reported, alongside acceptance under the user-updated target.

## C14: initial same-orbital verification

The fixture contains the requested existing four Kekulé structures and same
HAO initial guess, 14e/14o, cc-pVDZ, D/S=200/400. No adaptive/contracted inputs.
The exact and RI same-orbital comparisons have completed. The original reports
retain `passed=false` because the first tool applied benzene's stricter S gate
to both systems. The user has now confirmed C14 is assessed by its energy,
gradient and coefficient/weight requirements, with S discrepancies retained
separately. Under that confirmed scope, both C14 same-orbital checks pass:

| Model | ΔE (Eh) | max ΔH | max ΔS | max Δc | max Δweight | max Δgradient |
|---|---:|---:|---:|---:|---:|---:|
| exact | 3.23e-11 | 3.17e-9 | 1.524e-10 | 7.04e-10 | 9.10e-10 | 1.75e-9 |
| RI | 3.23e-11 | 3.17e-9 | 1.524e-10 | 7.04e-10 | 9.10e-10 | 1.75e-9 |

The C14 profile is ΔE<=1e-6 Eh, max Δgradient<=1e-7, and max Δc/weight<=1e-4;
H/S differences remain finite, explicitly printed diagnostics. Benzene keeps
all its original strict H/S gates. A regression test distinguishes these
profiles and rejects nonfinite diagnostics in either one. No optimizer or RDM
checking tolerance is changed. The reference
has 416 determinants; TNVB has zero. Tightening only the pair norm-fit tolerance
and SVD cutoff (1e-12 / 1e-16) does not remove the S discrepancy: max ΔS remains
1.520e-10. The pair-fit diagnostics below were not adopted; D/S=200/400 and
the ordinary reference remain unchanged. The same-orbital numerical
reports are retained unchanged; an explicit command checks their measured
errors against the user-confirmed C14 profile before scheduling optimization.

An independent small-system batched-LU overlap oracle reproduces the D=200
error (1.5238e-10); it is not a reference cofactor artifact. A separately
labeled D=256 sensitivity diagnostic reduces it to 3.6679e-11. This diagnostic
does not change the prescribed C14 acceptance D/S=200/400. The attempt with
12 initial sweeps and a 1e-14 norm tolerance did not finish its first state in
more than six minutes; a live stack was in Block2 MPS-center movement. That
extra diagnostic worker was terminated, its failed log retained, and the
stable default pair controls left unchanged. Reordering commuting pairs by
right endpoint gives 1.490e-10, not a substantive accuracy improvement. A
six-sweep diagnostic also timed out at its 180 s bound. Neither change was
adopted. These sensitivity runs are not substitutions for the fixed 200/400
optimization acceptance.

Measured TNVB worker stages (4 CPUs, one H/S worker):

| Model | Total RDM time | FULL dm3 time | fit_dE | RDM identity max |
|---|---:|---:|---:|---:|
| exact | 48.77 s | 41.37 s | 3.30e-13 | 2.74e-10 |
| RI | 45.51 s | 38.37 s | 3.34e-13 | 3.24e-10 |

These identity errors meet 1e-8; they are not substitutes for the C++ reference
gradient gate. A live exact reference reached approximately 9.4 GiB RSS; resource
planning must include it, rather than quoting only the Block2 stack.
The expensive reference stage was the existing opposite-spin gradient backward
contraction (`consume_primary_tiles`), verified by a live stack trace; it was
CPU-active, not waiting for the worker.

An existing legacy auxiliary record is
`test/tnvb_adaptive_control/c14_4str_plateau_sweep_window2/fixed/`
`C14_4str_fixed_D200_S400.xmo`: converged at 100 iterations, printed energy
-535.63699788 Eh; the associated `.time` records 5283 s and exit 0.
Its optimizer/gradient threshold and printed precision differ from the C++
acceptance run. It is auxiliary context, not an exact terminal-energy oracle.

## C14 complete optimization: final results

The user-confirmed same-orbital gates permit proceeding with the original
D/S=200/400 and original optimizer tolerances. Both jobs completed on 2026-10-08:

| Model | Slurm job | CPUs / job memory | Node at first live check | Step cap |
|---|---:|---:|---|---:|
| exact | 2104966 | 16 / 64 GiB | pc008 | 200 per backend |
| RI | 2104967 | 16 / 64 GiB | pc009 | 200 per backend |

Both jobs passed a same-node benzene ENERGY backend/dependency preflight and
entered the full optimizer. Each uses a frozen independent installation,
worker and input snapshot in `build/cpp-tnvb-c14-validation`; SHA256 checks
record the binaries, worker and C14 input. Subsequent source edits/rebuilds do
not affect these jobs. Both original C++ reference and TNVB optimizations
converged within their 200-step bounds. Reference optimization ran first, with a saved JSON
checkpoint, followed by TNVB from the same initial orbitals and endpoint
cross-evaluation. No extension, retry, different initial guess, higher bond
dimension or alternative convergence limit was used for these complete runs.

| Metric | Exact | RI | Required limit/status |
|---|---:|---:|---|
| Reference / TNVB accepted steps | 104 / 104 | 106 / 107 | <=200 per backend; both converged |
| Reference total energy (Eh) | -535.636997588905 | -535.636448578363 | Same integral model as TNVB |
| TNVB total energy (Eh) | -535.636997601999 | -535.636448696347 | Original optimization settings |
| Final energy difference (Eh) | 1.30938e-8 | 1.17984e-7 | <=1e-6; pass |
| Final max coefficient difference | 1.05249e-6 | 9.74999e-6 | <=1e-4; pass |
| Final max weight difference | 1.09477e-6 | 1.05349e-5 | <=1e-4; pass |
| Reference / TNVB final gradient L2 | 6.06088e-4 / 6.01950e-4 | 9.07054e-4 / 7.97459e-4 | Both <1e-3; pass |
| Reference / TNVB final absolute accepted dE (Eh) | 4.87976e-8 / 4.98710e-8 | 8.33675e-8 / 4.43740e-8 | Both <1e-7; pass |
| TNVB fit_dE (Eh) | 8.06466e-13 | 8.13571e-13 | <=1e-7 verification gate; pass |
| TNVB RDM identity max | 3.86083e-10 | 4.15754e-10 | <=1e-8; pass |
| Worst endpoint same-orbital gradient difference | 1.87762e-9 | 1.87776e-9 | <=1e-7; pass |
| Worst endpoint same-orbital energy difference (Eh) | 6.16183e-11 | 6.13909e-11 | <=1e-6; pass |
| Worst endpoint H / S differences | 8.39628e-9 / 3.76412e-10 | 8.40090e-9 / 3.76577e-10 | Finite diagnostics, separately reported |
| Exported normalized-orbital energy error (both backends, Eh) | 0 | 2.27374e-13 | <=1e-9; pass |
| Comparison JSON | passed=true | passed=true | Complete verification passed |

The endpoint S differences exceed benzene's 1e-10 limit; this is explicitly
retained, not hidden as a strict-matrix pass. The requested C14 energy,
gradient and coefficient/weight checks pass at the original D/S=200/400.
Independent optimizations end at slightly different finite-tolerance points;
both endpoint cross-evaluations agree between electronic backends.

| Timing/resource measurement | Exact | RI |
|---|---:|---:|
| Ordinary C++ reference optimizer wall time | 18177.648 s | 20618.213 s |
| TNVB optimizer wall time | 3238.403 s (54.0 min) | 3389.709 s (56.5 min) |
| Reference / TNVB optimizer time ratio | 5.6 | 6.1 |
| Whole comparison job elapsed | 06:03:38 | 06:47:15 |
| Compute-step elapsed | 06:03:35 | 06:47:12 |
| Slurm compute-step MaxRSS | 20963508 KiB (~20.0 GiB) | 2333888 KiB (~2.23 GiB) |
| Parent ru_maxrss | 20210748 KiB | 1371736 KiB |
| Largest reaped child ru_maxrss | 20210748 KiB | 1371736 KiB |
| Slurm job / compute-step state, exit code | COMPLETED / COMPLETED, 0:0 | COMPLETED / COMPLETED, 0:0 |

These timings compare the same 16-CPU allocation, initial guess and stopping
criteria for this four-structure C14 fixture. They are not a general scaling
or larger-system speed claim. Whole-job elapsed includes both optimizers and
the endpoint comparisons. Slurm MaxRSS is the recorded compute-task metric;
it is not presented as a separately measured concurrent sum over all worker
processes or as TNVB-only memory. Tool ru_maxrss is also a per-process metric.
Block2's configured 8 GiB stack and the requested 64 GiB job budget are different
quantities. The launcher's `/usr/bin/time` saw only approximately 8 MiB RSS and
must not be used as the compute-task peak. Final accepted TNVB evaluations
report about 26.0 s (exact) / 26.9 s (RI) for total-state/RDM work, with FULL dm3
about 22.0 / 22.7 s; all stage timings remain in the original logs.

Evidence files are `c14_{exact,ri}_opt200.{json,log,time}` and
`c14_{exact,ri}_opt200.json.reference.json` within that snapshot. Final Slurm
accounting and tool parent/child RSS are recorded above; the launcher's
`time` memory alone is not the concurrent compute-step peak.

## Final completion audit

The final audit inspected current source, build caches, fixtures, installed
artifacts, regression logs, high-precision JSON and completed Slurm accounting.
It did not infer completion merely from process exit or a source pass flag.

| Goal requirement group | Authoritative final evidence |
|---|---|
| Minimal three-keyword interface and effective ITMAX | Parser/loader/CLI source; `tnvb_input` and `tnvb_cli`, including CLI override and omitted-S ENERGY |
| Singlet selected covalent/ionic structures, HAO/OEO and exact/RI | Actual fixtures (one covalent plus ionic double-occupancy counts 1/1/2/3), four benzene comparisons, four FD/legacy cases, complete optimizations |
| Raw structure preservation, no expansion/CAS/HVP fiction | Loader condition, independent backend contracts and trial assertions; TNVB determinant count zero versus reference 19/416 |
| ENERGY without total-state/RDM; FULL gradient, DIRECT and SU(2) | Owned `evaluate.py` mode branches, hard-coded core SU(2) route, FULL/DIRECT call; ENERGY forbidden-3PDM tests and complete C14 runs |
| Shared electronic interface and ordinary-path isolation | `ElectronicStateBackend`, default determinant adapter and original default forward route; OFF/ON ordinary regression sets |
| Correct normalization/phase/root/energy composition | Worker dense solve and retained RDM coefficients; C++ norm/residual checks; reversed-pair test; same-orbital and endpoint comparisons |
| Formula-derived As/Ah/Ag and shared orbital/RI pullback | Gradient conventions document, adapter source, independent S/h/packed-g and active/inactive orbital FD, existing RI factor pullback unchanged |
| No full AO ERI restoration or rank-six IPC | Worker boundary's active-pair RI reconstruction and P/Pi/D3G buffers; existing C++ RI representation retained |
| Original ISCF=5 optimizer and stopping settings | Clean diff for optimizer backends/options/integral kernels; unchanged 1e-7/1e-3 values in all formal optimization reports |
| Failed trial/worker isolation and numerical errors | Actual child FULL failure plus NaN tests; accepted data unchanged; private forward objects; hard 1e-8 RDM checks |
| Finite-fit warning and approximation disclosure | Deliberately truncated CLI regression continues with warning; finite-D limitation stated in usage/derivation |
| Optional build, compatible runtime and independent install | OFF=10/10, ON=15/15; installed executable/owned worker hashes match; install CLI suite passed from /tmp without LD_LIBRARY_PATH |
| Child environment/scratch/cache and conservative resources | Source copies parent environment to posix_spawn, mkdtemp ownership, current-integral re-evaluation, Slurm/affinity/memory phase-layout code and printed C14 layout |
| Normal report, weights/orbitals/analyses and no legacy AIM/WFN port | Production CLI output/Molden regression; H/S, coefficients, four weights, orbitals, natural/population output preserved |
| Stage A/B/C numerical gates and bounded C14 work | All 24 required checks passed; both C14 optimizations <200 steps; historical failures/diagnostics retained separately |
| Deliverables and unchanged scope | Independent src/tnvb target, fixtures/minimal examples, submit.slurm, conventions/usage/final report, comparison/FD/optimization/audit tools; no C20+ acceptance jobs |

Ordinary determinant numerical changes are limited to the `minor` variable
rename needed to avoid a system macro collision; the contractions themselves
are unchanged. Original L-BFGS/line-search/options and integral kernels have
no diff. The copied core retains dormant numerical helpers, but its old binary
protocol/CLI entrypoint is removed and no unsupported mode is exposed by the
production adapter. No commit, push or existing installation replacement was
performed; all source changes remain on the independent `tnvb` worktree branch.

## Independent saved-evidence audit

`tests/tnvb/summarize_acceptance.py` rechecks saved high-precision numerical
reports without running a calculation or modifying source evidence. It checks
finite values, original convergence settings, accepted energy changes and
gradient norms, endpoint errors, final coefficient/weight arrays, RDM limits,
exported-energy consistency and the C14 step cap. A passing supplementary
diagnostic cannot replace an original failed optimization. Missing evidence is
not interpreted as a running or successful job. The historical C14 S error
and original `passed=false` flag remain explicitly visible under the approved
C14 matrix-diagnostic policy.

The current `build/cpp-tnvb/numerical_acceptance_summary.json` applies the
user-confirmed 1e-6 optimized-energy target without changing optimizer
convergence parameters. It has **24 passed required numerical checks**, with
no failed or pending required result. Historical OEO flags under the
former 1e-8 target remain visible; stricter results are historical only.
This summary deliberately does not claim
full implementation/build/install acceptance. The audit's lightweight logic
tests are registered only in the TNVB-enabled build and do not change any
production numerical code or the frozen C14 snapshot.

The audit classifier's 14 lightweight unit cases pass, including the approved
1e-6 optimized-energy comparison and unchanged stricter same-orbital gates.
Final ON CTest is 15/15 (87.45 s), and final OFF regression is 10/10 (2.04 s),
recorded in each build's `test-final-delivery.log`. The installed CLI suite
again passes from /tmp with LD_LIBRARY_PATH absent. The rebuilt acceptance optimizer rejects
extra convergence arguments before loading input or creating a report. The
main executable and both worker files
retain the same hashes as the independently validated install; all five frozen
C14 snapshot hashes still pass. These test/document-only changes do not alter
the two completed optimization jobs or their convergence criteria.

## Production output and acceptance-tool output

`optimize_tnvb` is a newly written, optional test driver under
`tests/tnvb/optimization.cpp`, built only when its excluded target is explicitly
requested. It calls the existing optimizer for the two backends and writes
17-digit JSON plus progress logs; it does not call the production text formatter.
Consequently the frozen C14 `.log` files are comparison logs, not `.xmo` reports.
The JSON retains optimized orbital tables, coefficients, weights and histories.

The production interface is the independently installed `bin/xmvb-cpp.exe`.
It already prints normal H/S, coefficients, weights, orbitals and applicable
density analyses. Use the production submit template to capture stdout as
`.xmo`; renaming a test log cannot recreate missing report sections. No
acceptance-driver dependency or extra keyword is required for normal TNVB runs.
