# Independent TNVB: build, run and numerical acceptance

## Production worker surface

The owned worker contains the SU(2) paired-structure constructor, DIRECT
total-state SVD, full 1/2/3-RDM evaluation, DENSE/Davidson structure solvers
and the one-shot ENERGY-to-gradient state handoff. It accepts the C++
orthogonal integral frame directly. Input keywords and wire protocol v3
are unchanged.

Unused SZ/determinant constructors, variational total-state fitting,
CONTRACTED/VALIDATE implementations (including explicit-product, hole-state,
sigma, weighted-NPDM and complementary-operator prototypes), their process
pools and backend-selection arguments have been removed. They have no
compatibility entrypoints or environment-controlled fallback. Raw-frame
density conversion is retained only as an independent test reference.

The production truncation thresholds, total-state fitting warning at
`1e-7 Eh`, RDM warning at `1e-8` and fatal RDM consistency limit at `1e-7`
are unchanged. Full 3-RDM reduction still avoids a second rank-six copy.
Execution-policy controls for scratch, memory, threads, H/S construction
and NPDM profiling remain available. This cleanup changes code ownership
and maintenance cost; it does not claim a different computational scaling.

## Same-point ENERGY to FULL-gradient handoff: 2026-10-18

The lazy-gradient interface now retains the final structure MPS files from
an ENERGY trial and consumes them when that same trial needs its gradient.
This applies to both DENSE and Davidson. The second worker does not rebuild
the structures, recompute H/S, or solve CI again. It uses the original
selected eigenpair and orthogonal integral context for the FULL 1/2/3-RDM
calculation. The C++ forward object also owns the original orbital/integral
preparation result, so completing its gradient does not repeat preparation
or serialize a second integral request. No input keyword is added.

Each forward object owns a private directory lease. Only final state files
are moved into its retained bank; pair intermediates are excluded. RDM work
consumes that bank directly instead of copying it into another workspace.
Davidson's private native-operator child can likewise use its caller-owned
bank without another file-bank copy. These changes remove redundant phase
transfers; native tensor loading, common-basis conversion and total-state
construction still perform their required numerical work.

The handoff is one-shot and checks the scientific input, original request,
worker settings/code, retained file inventory and solved response before
RDM evaluation. Changed orbitals, integrals, structures or numerical settings
cannot count as a cache hit. Failed or discarded trials release their private
directories without modifying the accepted electronic state. Ordinary VBSCF,
orbital stopping conditions and the optimizer algorithm are unchanged.
Standalone `ITMAX=0` ENERGY does not retain a bank or compute RDMs.
Python workers still have separate native lifetimes; this is not a persistent
worker service or a cross-orbital numerical-result cache.

Scratch parent selection is `XMVB_BLOCK2_SCRATCH_PARENT`, then `SLURM_TMPDIR`,
then the system temporary directory. A user-selected parent retains
precedence. Node-local scratch can independently improve small-file access;
measure old/new handoff timings with the same placement and resources.
Validation inputs, normal CLI XMO outputs, numerical comparisons and retained
failure diagnostics are under `cpptest/tnvb_energy_gradient_handoff_20261018`.
The matched development build is `xmvb4.0/build/cpp-tnvb-master-20261017`;
this change does not overwrite `install-tnvb`.

All six same-orbital handoff checks passed: C6 exact and C14 exact/RI, with
both solvers. C14's maximum energy, weight and orbital-gradient differences
were `2.3e-13 Eh`, `9.7e-14` and `8.8e-12`, respectively. Ten focused CTests
passed, including changed-input rejection and failed/discarded-trial isolation.
Matched pc001/16-core/local-scratch C6 full optimizations converged in 9 steps
with both versions. Across two repeats, DENSE mean wall time decreased from
14.991 s to 10.274 s and Davidson from 13.211 s to 8.755 s. MPS builds fell
from 19 to 10; RDM evaluations remained 10. These are small-system timings,
not a predicted speedup for 3-RDM-dominated large systems. Full evidence and
measurement conditions are in that validation directory's README.md.

## Generalized Davidson integration: 2026-10-18

`TNVB EIGENSOLVER=DAVIDSON` uses one fixed common SU(2) family and
one reusable native effective Hamiltonian for the lowest singlet root.
Writing the structure family as `B = L P`, each CI action is
`Hv = P.T H_eff P v`, `Sv = P.T P v`. Only structure coefficients vary;
the solver does not perform unrestricted DMRG or enlarge the VB space.
The common family uses lossless block QR; no sigma-MPS fit is involved.

The generalized solver uses S-orthogonal corrections, diagonal H/S
preconditioning, low-Ritz-vector restart and a fresh final-action residual
check. Its CI residual budget is at most `1e-10` Eh and never looser than
the requested outer energy/gradient accuracy. Orbital stopping conditions
and the P-LBFGS algorithm are unchanged. A rank-deficient normalized metric
or a failed solve is an error, not a silent structure deletion or dense
fallback. The same converged coefficients feed the FULL RDM/gradient path;
there is no second dense CI solve in that path.

Dense remains the default. Both modes support `ITMAX=0` ENERGY and ISCF=8
orbital optimization. ENERGY constructs no total-state MPS or RDM. In
Davidson mode, full H/S are exported **after** CI convergence for the normal
XMO matrices, weights and analysis. Accordingly the solve is matrix-free,
but current result export and postprocessing still have quadratic storage;
this is not an end-to-end linear-memory implementation.

Common-family ranks can exceed individual `TNBOND` substantially. The
`XMVB_BLOCK2_COMMON_GB` resource control (default 2 GiB) bounds estimated
Python family workspace and the center map, not total process RSS or the
native environment. Native stack remains governed by `XMVB_BLOCK2_STACK_GB`;
when unset, the common-operator phase has an 8-GiB default floor (an
individual-structure 512-MiB budget proved insufficient for C14).
Increasing these is not an accuracy setting. Exhausting the budget reports
an error; select DENSE when common-basis growth is unfavorable. No universal
speedup over pairwise contraction is assumed. The native environment is
private to one child process and is released before RDM construction.

Numerical integration evidence is recorded under
`cpptest/tnvb_davidson_integration_20261018`; do not infer a completed test
from the mere presence of an input file. The matched development build is
`xmvb4.0/build/cpp-tnvb-davidson-20261018`; existing installations are not
automatically overwritten.

Initial numerical gates passed for frozen C10 4/64 mixed covalent/ionic
states, normal benzene 175-structure ENERGY, benzene exact/RI same-orbital
FULL gradients and matched normal-CLI benzene orbital optimizations (both
solvers converged in 9 steps with unchanged stopping settings). In the
175-structure single-node comparison, H/S+CI wall time was 84.809 s for
shared-readonly DENSE versus 21.656 s for common-basis Davidson; total wall
time was 137.663 s versus 83.435 s. MPS build times fluctuated and were not
optimized. C14 exact ENERGY passed its confirmed acceptance profile; full
exact/RI gradient-reference comparisons are tracked separately and are not
yet claimed complete. See the validation directory's RESULTS.md for scope
and the retained dispatch/stack failures.

## Shared-readonly ENERGY integration: 2026-10-18

Parallel fixed-orbital ENERGY now uses `hs_shared.py`: prepare a common
one-site/right-center MPS file bank once, then read it from process-private
Block2 drivers with MPS metadata writes disabled. Bank contents are checked
before/after contraction. Mutable tensors, MPOs and environments are not
shared between processes. Serial ENERGY and FULL gradients retain their
existing H/S path. No input keyword or optimizer setting is added.

The SU(2) pair-construction cutoff defaults to `1e-14` while `TNBOND`
continues to cap the bond dimension. This trims negligible subspaces before
structure summation and high-rank RDM environment construction. The warm-cache
manifest and ENERGY-to-GRADIENT handoff signature use the same default, so
zero-cutoff states are not reused under the default setting. The existing
`XMVB_BLOCK2_SU2_CUTOFF=0` override remains available for diagnosing pair-fit
stagnation or small-subspace loss; it can substantially increase stored bond
dimensions and RDM memory. No production input option is added.

The verified C10 256/512 ENERGY prototype evidence is under
`cpptest/tnvb_c10_ionic_hs_scaling_validation_20261017/development_20261018`.
Production integration and C14 exact/RI same-orbital regressions are tracked
separately under `cpptest/tnvb_energy_integration_c14_20261018`; that directory
retains the pre-integration executable/worker. This is not a C14 full-space
or full-optimization claim. Common-prefix reuse is not enabled in production.

## Master synchronization: 2026-10-17

Master `03d469cc` is integrated with TNVB restricted to ISCF=8/P-LBFGS.
The C++23 build includes the production CLI and all TNVB numerical tools.
Five focused CTests pass: input invariants, gradient adapter, RDM policy,
acceptance-summary logic, and trial isolation with an accepted orbital step.
The normal benzene ENERGY CLI run and fixed-orbital exact/RI energy-gradient
comparisons also pass; maximum same-orbital gradient differences are
5.11e-15 / 4.94e-15. This is not a new full C14 optimization acceptance.
The isolated verification build is
`/pool1/home/chentz/apps/xmvb4.0/build/cpp-tnvb-master-20261017`.
Its LAPACK and LAPACKE explicitly use the previously verified dynamic
OpenBLAS library recorded in the older `cpp-tnvb/CMakeCache.txt`; automatic
selection of the older static library caused SIGILL on this host.
No existing installation or submission script is replaced.

## Scope

The `tnvb` branch supplies a separate `xmvb_tnvb` library. Production uses
singlet VB structures, SU(2) Block2, fixed bond dimensions, DIRECT
total-state construction and FULL 3-RDM. HAO and OEO use the same electronic
adapter. Selected structures remain a selected space, not a complete CAS.
The default ordinary determinant path and its second-order caches remain
available; a TNVB evaluation creates neither determinants nor an
`AcceptedPointContext`. TNVB uses ISCF=8, the shared nonredundant orbital-block
P-LBFGS optimizer (`Plbfgs`), after merging master `03d469cc` on 2026-10-17.
It uses the same first-order electronic adapter and does not require an
`AcceptedPointContext`, a determinant HVP, or a new input keyword. Its original
stopping metric is the projected reduced-gradient infinity norm. Enabling TNVB
does not change that optimizer's numerical stopping thresholds or algorithm.
Master removed native ISCF=5; this branch does not restore or alias it.

Only these new input keywords are registered:

| Keyword | Rule |
|---|---|
| `TNVB` | Enable the independent electronic backend. |
| `TNBOND` | Required positive integer; cap for each structure's SU(2) MPS. |
| `TNSUMBOND` | Required for optimization, positive and at least `TNBOND`. |

For ENERGY, use `VBSCF ISCF=8 ITMAX=0 TNVB TNBOND=128`. No total MPS or
1/2/3-RDM is constructed. For optimization, use
`VBSCF ISCF=8 ITMAX=200 TNVB TNBOND=128 TNSUMBOND=256`.
The CLI override is `--optimizer-backend plbfgs`. In ENERGY (`ITMAX=0`),
the fixed orbitals are evaluated without optimization.
These are additions to the existing `$CTRL`; geometry, orbital guess and
VB structures are still needed. Structures can be supplied explicitly through
`$STR` or generated using the existing supported `STR=...` structure classes,
including `STR=COV`. Both sources feed the same TNVB MPS construction without
determinant expansion. Final command-line `--max-iterations`
overrides `ITMAX`. Omitted `ITMAX` retains the project default 2000; specify
zero explicitly for ENERGY. TNVB defaults to dense H/S solving;
`EIGENSOLVER=DAVIDSON` selects the common-basis iterative CI path described
above. An unsupported optimizer request is a capability error.

No old TNVB keywords, adaptive scheduling, contracted backend, TNHVP, CIAH,
state averaging, open-shell treatment or nuclear derivatives are registered
by this module. Generic unknown-keyword handling remains the existing
project behavior, without compatibility or old-keyword rejection branches.

## Build isolation

The project requires CMake >=3.20 and a C++23 compiler.
`XMVB_ENABLE_TNVB` defaults to OFF. This build does not require Python or
Block2. Configure an independent build directory and dependency roots:

```sh
cmake -S /path/to/xmvb-cpp -B /path/to/build-off \
  -DCMAKE_BUILD_TYPE=Release -DXMVB_ENABLE_TNVB=OFF \
  -DLAPACK_ROOT_DIR=/path/to/openblas \
  -DCINT_ROOT_DIR=/path/to/libcint -DEIGEN3_ROOT_DIR=/path/to/eigen
cmake --build /path/to/build-off --parallel 4
ctest --test-dir /path/to/build-off --output-on-failure
```

For ON, also configure matching paths once:

```sh
cmake -S /path/to/xmvb-cpp -B /path/to/build-tnvb \
  -DCMAKE_BUILD_TYPE=Release -DXMVB_ENABLE_TNVB=ON \
  -DLAPACK_ROOT_DIR=/path/to/openblas \
  -DCINT_ROOT_DIR=/path/to/libcint -DEIGEN3_ROOT_DIR=/path/to/eigen \
  -DXMVB_TNVB_PYTHON=/path/to/python \
  -DXMVB_TNVB_BLOCK2_MODULE=/path/to/compiled-block2 \
  -DXMVB_TNVB_BLOCK2_SOURCE=/path/to/block2-source \
  -DXMVB_TNVB_LIBRARY_DIR=/path/to/matching-gcc/lib64
cmake --build /path/to/build-tnvb --parallel 4
ctest --test-dir /path/to/build-tnvb --output-on-failure
cmake --install /path/to/build-tnvb --prefix /path/to/new-install
```

Use a BLAS built for the execution node, preferably dynamic CPU dispatch.
The original fixed SKYLAKEX OpenBLAS caused SIGILL on this host's AVX2 CPU;
the existing dynamic-architecture LP64 OpenBLAS solved that issue. GCC 13
and matched dependencies avoid the earlier incompatible-binary problem.
The local validated cache and exact dependency paths are under
`/pool1/home/chentz/apps/xmvb4.0/build/cpp-tnvb/CMakeCache.txt`.

Keep the matching compiler runtime active when building and running the
original project's test binaries. On this machine that is:

```sh
source /etc/profile.d/modules.sh
module load gcc/13.2.0
```

Without it, the shared ordinary `cli_options` test can load the shell's
Anaconda C++ runtime and segfault in both OFF and ON builds. The installed
TNVB executable and new acceptance tools have explicit matching runtime
paths; this shell requirement is not another TNVB input or worker setting.

Installation contains `bin/xmvb-cpp.exe`, `lib/libxmvb_tnvb.a`, its own
`libexec/xmvb-cpp/tnvb/` (including `forward_state.py` and the common-basis
CI modules), and basis assets in
`share/xmvb-cpp/basis`. Configured external Python/Block2/runtime dependencies
must remain accessible. This is not a self-contained binary distribution.
Existing project/local basis files retain precedence over installed assets.
Only the ON executable gains this independent install rule.

The build stages an owned worker copy; production never imports the legacy
XMVB worker. Installation resolves its worker relative to the executable.
Child-only Python paths and matched C++/OpenMP preloads do not change the
parent's environment; inherited preloads are retained. Each evaluator has
unique scratch/cache ownership and each request has its own files. A warm
state is only an initial guess: H/S and RDM are recomputed for current
integrals. Trial numerical results cannot replace the accepted point until
the optimizer explicitly commits them.

## Run and resources

```sh
/path/to/new-install/bin/xmvb-cpp.exe calculation.xmi > calculation.xmo 2>&1
```

RI is selected using the existing `RI`/`AUXBASIS` input conventions or
`--standard-two-electron-mode ri`. The worker receives active-space ERIs;
the C++ backpropagation retains RI factors and does not reconstruct AO ERIs.
RI and exact calculations are separate integral models: agreement between
the two electronic backends is tested within each model.

The example `src/tnvb/templates/submit.slurm` needs only the input and installed
executable. It runs in the submission directory. Adjust the Slurm CPU/memory
request for the system; no Python/worker/backend switches are needed.
Worker processes and Block2 threads are chosen from Slurm limits/CPU affinity,
the useful work and available Slurm memory. The actual state/H/S/RDM layout is
printed. `XMVB_BLOCK2_WORKERS`, `XMVB_BLOCK2_THREADS`,
`XMVB_BLOCK2_RDM_THREADS`, `XMVB_BLOCK2_STACK_GB`,
`XMVB_BLOCK2_RDM_STACK_GB` and `XMVB_BLOCK2_SCRATCH_PARENT` remain advanced
resource overrides, not ordinary input keywords. A Block2 stack is not the
whole job's memory budget. Multi-process copies and C++ AO/response work also
consume memory.

Finite `fit_dE>1e-7 Eh` emits a warning and continues. It is not a user-configured
convergence limit. Nonfinite data, invalid structure metric, worker failure,
eigenpair norm/residual failure and RDM check residuals above `1e-7` are errors.
Finite RDM check residuals above `1e-8` and at most `1e-7` emit a warning and
continue. Both limits are fixed in the C++/worker production path; inherited
`XMVB_BLOCK2_RDM_CHECK_TOL` cannot override them. This changes numerical
consistency handling, not orbital convergence parameters or RDM algorithms.
At finite bond dimensions, the gradient does not differentiate all truncation
operations. Check bond convergence before interpreting an approximate optimum.
ENERGY output reports no orbital convergence or gradient norms. Optimization
retains H/S, all existing structure weights, orbital and density analyses.
The CLI regression also checks Molden export in ENERGY and gradient modes,
all 24 orbitals of the benzene fixture, natural-orbital and population sections
after a full-gradient evaluation, and their absence in ENERGY output.
ISCF=8 integration checks and their explicitly bounded C14 scope are recorded
separately in `tnvb_iscf8_integration.md`; the original ISCF=5 acceptance archive
is preserved as historical evidence. The P-LBFGS optimizer's Armijo trials evaluate full
gradients, including FULL 3-RDM, even for rejected trials. Joining this optimizer
does not by itself reduce the cost of a gradient evaluation.

## Reproducible acceptance tools

Build optional numerical tools explicitly:

```sh
cmake --build /path/to/build-tnvb --target \
  compare_tnvb check_tnvb_directions optimize_tnvb \
  prepare_tnvb_oeo export_tnvb_request --parallel 4
```

All results are JSON at 17-digit precision, with explicit pass/fail, not tests
of rounded `.xmo` energies:

```sh
compare_tnvb INPUT REPORT.json exact gradient
compare_tnvb INPUT REPORT.json ri energy
check_tnvb_directions INPUT FD.json exact
optimize_tnvb INPUT OPT.json exact 200
```

`compare_tnvb` accepts an optional final-orbital plain-text table. It expands
determinants only in its explicit small-system reference branch, normalizes
both structure bases, aligns the root's overall phase, compares H/S,
coefficients and four weight definitions, and applies the common orbital
pullback for gradients. `check_tnvb_directions` checks independent S, h,
packed-g, inactive-orbital and active-orbital directions at three step sizes;
two adjacent passing step sizes are required.

`optimize_tnvb` uses the same guess, ISCF=8 and convergence settings for both
backends, retains failed/unconverged outcomes, compares both endpoints and
cross-evaluates their exported physical orbitals. A different stationary point
is not labeled as a passing optimization match. C14 is capped at 200 accepted
steps. Parent peak RSS and largest reaped child peak RSS are recorded separately;
neither is claimed to be the aggregate concurrent job peak. Slurm/cgroup
accounting is needed for that quantity.

The benzene profile retains its strict H/S, energy, coefficient, weight and
gradient gates. The user-confirmed C14 profile uses energy <=1e-6 Eh,
gradient <=1e-7, and coefficient/weight errors <=1e-4. Its finite-D H/S
differences are reported separately, including whether they satisfy benzene's
stricter matrix limits. Nonfinite matrices remain errors for both profiles;
The historical numerical acceptance profile retains RDM identity <=1e-8;
production now warns above 1e-8 and stops above 1e-7. Warning-band continuation
alone does not establish that stricter acceptance criterion. Eigenpair/metric
checks are never disabled. C14 final
optimization additionally requires both backends to converge under the
original optimizer criteria, fit <=1e-7 and exported-energy consistency.

`optimize_tnvb` uses only the original optimizer convergence settings:
energy tolerance 1e-7 Eh and raw gradient L2 tolerance 1e-3. There are no
additional tolerance arguments. The user explicitly requires unchanged
convergence parameters. Earlier reports labeled `stricter_diagnostic` are
retained solely as historical diagnostics and cannot supply formal acceptance.
C14 retains its original tolerances and 200-step limit.

The user-confirmed cross-backend optimized terminal-energy difference target
is 1e-6 Eh for both benzene and C14. This comparison target does not change the
optimizer's 1e-7 energy-change or 1e-3 gradient L2 convergence settings. Benzene
same-orbital energy/H/S/coefficient/weight/gradient gates remain unchanged.

`prepare_tnvb_oeo HAO.xmi OEO.xmi` exports the actual supported, normalized
HAO physical guess into full-AO OEO coefficients. The original autoVB GUS
contains repeated full-AO columns which are valid only after HAO support
projection; replacing the `ORBTYP` token alone gives a singular OEO guess.
The checked-in benzene OEO fixture includes this conversion.

`compare_legacy_worker.py` is test-only: explicitly provide a frozen legacy
worker and a request from `export_tnvb_request`. It compares H/S, root,
coefficients and all transferred density/contraction buffers. It has no
production import or fallback role.

`diagnose_overlap.py REQUEST WORKER REPORT` is a small-system (K<=14),
test-only batched-LU oracle for structure overlaps. Optional `--bond` and
`--pair-order right-endpoint` diagnose finite-D and pair-product-order effects
without changing the production input. Its explicit CSF enumeration belongs
only to the test reference; production TNVB remains unexpanded.

See `tnvb_gradient_conventions.md` for the mathematical interface and
`tnvb_acceptance_report.md` for the completed acceptance results and limitations.

`tests/tnvb/summarize_acceptance.py BUILD_EVIDENCE C14_EVIDENCE --output SUMMARY.json`
independently rechecks the saved numerical errors, finite-difference interval,
original optimizer criteria, endpoint gradients, C14 fit/export checks and
iteration cap. It never edits its input evidence or jobs. Missing reports are
`pending_evidence`, not proof that a job is running or successful. Original
optimization checks and supplementary stricter diagnostics are separate;
passing supplementary checks cannot turn an original failure into a pass.
Historical C14 matrix-only failures retain their original flag and measured
H/S errors while applying the explicitly user-confirmed C14 profile. The
old benzene optimization flags are likewise retained when reassessing their
measured errors against the user-updated 1e-6 terminal-energy target; changed
convergence settings and unexplained failures cannot be reclassified. The
summary is numerical evidence only, not a substitute for the final source,
build, installation, isolation and requirement-by-requirement audit.
