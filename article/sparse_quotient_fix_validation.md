# Sparse quotient correction: validation record

## Scope and status

This record accompanies the correction to `NonredundantOrbitalSpace` based on
eqs 27a and 27b of
[the theory note](matrix_free_vbscf_orbital_optimization_theory.md).
The reference source is commit `1233a4c`; measurements below use the corrected
working tree on `node45`, 2026-09-07.

The correction passes the algebraic, physical-map, and HVP checks below.
The latest inactive-projected preconditioning stage reduces HVP work on all
three larger inputs, leaves F2 HVP work unchanged, and reaches smaller final
projected gradients on all four at the same default tolerances. This does not
establish asymptotically quadratic convergence or second-order stationarity.
The coordinate-only correction
took 119 MnF2 iterations. A subsequently identified HVP-subspace consistency bug
has been fixed, reducing this to 44 iterations. Correct spectral-endpoint handling
in the projected trust-region solver subsequently reduces this to 37 iterations
and 865 HVP directions. The next coordinate-consistent preconditioning and
curvature-recycling stage gives 21 iterations and 437 HVP directions, versus
18 iterations and 413 HVP directions in the matched old-coordinate reference.
Those intermediate changes had mixed default-tolerance results on the other
inputs. Incorporating the inactive projection into the local surrogate
subsequently gives 19 MnF2 iterations and 255 HVP directions. Each stage and
its limitations are distinguished below. Gauge removal alone does not
establish near-quadratic convergence.

## Construction

For each target orbital, the implementation forms gauge sources from the full
inactive span, adding the target's scaling direction for an active orbital.
It first enforces zero variation on forbidden AO entries, including stored but
fixed coefficients, and only then takes the complement on differentiable slots.
The numerical rank threshold is machine precision times the largest matrix
dimension times the largest singular value. No molecule-specific rank thresholds
or legacy construction switches are used.

The old occupied/virtual generator and its empirical rank cutoffs have been
removed. The resulting basis is stored in per-orbital blocks; the production
optimizer does not construct a global dense Hessian or global dense quotient
basis. The full global audit is a diagnostic only. The current coefficient
metric remains Euclidean, not a physical-metric whitening.

## Independent initial-point audit

The diagnostic compares the production basis with both a global
support-constrained gauge construction and an independently differentiated
physical map (inactive projector and projected active rays). The three larger
inputs use `--nonredundant-adapt true`, matching the optimizer's representation.

| Input | Packed dimension | Old reduced dimension | Corrected reduced dimension | Gauge rank | Retained gauge | Missing physical | Gauge/basis overlap norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| F2 | 60 | 46 | 42 | 18 | 0 | 0 | $5.73\times10^{-16}$ |
| 241 | 480 | 456 | 432 | 48 | 0 | 0 | $1.61\times10^{-15}$ |
| MnF2 | 1140 | 1019 | 957 | 183 | 0 | 0 | $7.80\times10^{-15}$ |
| FeCl2 | 4488 | 3855 | 3655 | 833 | 0 | 0 | $9.35\times10^{-15}$ |

The physical-Jacobian rank equals the corrected reduced dimension in each case.
Relative gauge-annihilation residuals are respectively
$1.40\times10^{-16}$, $1.21\times10^{-16}$,
$5.04\times10^{-15}$, and $2.55\times10^{-16}$.
These are finite-precision observations at the initial points. The large audit
uses a Jacobian Gram matrix, which squares conditioning; it is not a reliable
classifier of arbitrarily small nonzero singular values and does not establish
constant rank along an entire optimization trajectory.

## Synthetic regression and memory safety

`test_sparse_orbital_quotient` checks seven cases:

| Case | Packed dimension | Reduced dimension |
|---|---:|---:|
| Unequal supports with off-support cancellation | 12 | 7 |
| Stored but frozen active tail | 11 | 7 |
| Full support | 24 | 14 |
| Nonsingular change of inactive basis | 24 | 14 |
| No inactive orbitals | 6 | 4 |
| Inactive orbitals only | 6 | 4 |
| All coordinates are scaling gauge | 2 | 0 |

Checks include orthonormality, gauge/complement orthogonality, physical-image
completeness, adjoint projection and expansion, additive retraction, and
preservation of fixed coefficients. A separate finite physical map is centrally
differenced and its reduced Jacobian is checked by direct SVD. Its active rays
are represented by normalized rank-one projectors to avoid sign ambiguity.

All seven cases pass in Release and in a standalone build with AddressSanitizer,
UndefinedBehaviorSanitizer, Eigen assertions, and leak detection. The tests also
exposed two LAPACKE interface hazards fixed here: passing an unmaterialized
transpose into SVD, and invoking an eigensolver on an empty Gram matrix.

## Relaxed HVP finite differences

Central differences use a common accepted-point coordinate chart. Each input
passes the same relative-error limit, $10^{-7}$. Batched and scalar HVPs agree
to the printed precision (maximum absolute difference zero).

| Input | Finite-difference step | Maximum absolute error | Relative error |
|---|---:|---:|---:|
| F2 | $10^{-4}$ | $4.0617\times10^{-9}$ | $1.1257\times10^{-9}$ |
| 241 | $10^{-4}$ | $2.8727\times10^{-8}$ | $4.8481\times10^{-9}$ |
| MnF2 | $10^{-5}$ | $1.1312\times10^{-4}$ | $7.0016\times10^{-8}$ |
| FeCl2 | $10^{-5}$ | $3.3955\times10^{-6}$ | $5.6890\times10^{-9}$ |

At step $10^{-4}$, MnF2 and FeCl2 have relative errors
$7.0040\times10^{-6}$ and $5.6894\times10^{-7}$, respectively. Reducing the
step tenfold reduces the error approximately one hundredfold, consistent with
second-order central-difference truncation error. This changes the diagnostic
step, not an optimizer parameter or the acceptance threshold.

## Complete TNHVP runs before resolving the performance regression

All inputs terminate by the existing dual energy/gradient criterion. These
results establish end-to-end operability, not satisfactory convergence speed.

| Input | Outer iterations | HVP directions | Final total energy (Eh) | Final projected gradient infinity norm |
|---|---:|---:|---:|---:|
| F2 | 6 | 20 | -198.751155830527 | $1.3398\times10^{-7}$ |
| 241 | 9 | 108 | -230.720590392865 | $6.9696\times10^{-7}$ |
| MnF2 | 119 | 927 | -1348.893351713981 | $9.1369\times10^{-4}$ |
| FeCl2 | 6 | 193 | -2181.617637314785 | $3.3388\times10^{-4}$ |

F2 used one OpenMP thread; the other runs used four. OpenBLAS used one thread.
The larger runs overlapped in time, so their measured times are not suitable
for a controlled before/after performance comparison.

For MnF2, the energy is -1348.893080435312 Eh after iteration 21; a further
98 iterations are required. Recent pre-fix logs use the same energy tolerance
$10^{-7}$ Eh and gradient tolerance $10^{-3}$ and converge in 18 iterations,
with energy approximately -1348.893353206084 Eh. The corrected-coordinate run
therefore has both a much longer tail and a less accurate final energy at the
same stopping thresholds. A matched coordinate-only comparison and
trust-region/subproblem diagnostics are required to isolate the cause.

## Investigation of the MnF2 regression

A matched comparison used the same executable objects, input, stopping
tolerances, and four OpenMP threads, substituting only the old coordinate
implementation from `1233a4c`. Its unused `BlockBasis::n_virtual` assignment was
omitted for compatibility with the cleaned storage layout; its numerical
algorithm was unchanged. The comparison reproduced 18 iterations and 413 HVPs
for the old coordinates versus 119 iterations and 927 HVPs for the corrected
coordinates with the original subspace accumulation.

The coordinate-only run revealed a second, independent correctness bug. The old
subspace builder orthogonalized a reduced vector, its packed tangent, and its
HVP separately by subtracting cached vectors. For nearly dependent search
directions, these separately evolved quantities no longer represented the same
direction. Dividing the residual HVP by the small orthogonalization norm further
amplified cancellation error. Orthonormality in the separately cached tangent
array did not imply orthonormality of the actual reduced basis.

At outer iteration 22, direct checks gave

| Quantity | Old accumulation | Corrected accumulation |
|---|---:|---:|
| Reduced-basis orthogonality defect | $1.3371\times10^{-1}$ | $2.6317\times10^{-15}$ |
| Cached versus fresh step-HVP relative discrepancy | $1.97411$ | $1.12159\times10^{-13}$ |
| Relative skew of the projected Hessian before symmetrization | $8.84447\times10^{-2}$ | $2.37724\times10^{-15}$ |

The two trajectories differ after the fix; this table compares the same outer
iteration number, not identical orbital coefficients. The HVP discrepancy is
the Euclidean norm of the difference divided by the larger of one and the norm
of the fresh HVP. The orthogonality defect is the Frobenius norm of the reduced
basis Gram matrix minus the identity.

The corrected builder scales and orthogonalizes the **reduced direction first**,
applies the HVP to the resulting normalized basis vector, and constructs its
packed tangent afresh. It reconstructs the original search-direction HVP by
linearity, so each admitted independent direction still costs one HVP. This is
valid because the present strictly sparse additive chart has an orthonormal
Euclidean basis; a future non-Euclidean metric requires the corresponding
consistent construction. The formulas are recorded after eq 65 in the theory
note. A dedicated regression uses 48 nearly dependent directions with raw
amplitudes spanning 160 orders of magnitude, tests all three invariants, and
passes AddressSanitizer and UndefinedBehaviorSanitizer with leak detection.

| MnF2 implementation | Outer iterations | HVP directions | Final energy (Eh) |
|---|---:|---:|---:|
| Matched old-coordinate reference | 18 | 413 | -1348.893353206083 |
| Corrected coordinates, old HVP-subspace accumulation | 119 | 927 | -1348.893351713981 |
| Corrected coordinates and consistent HVP-subspace accumulation | 44 | 911 | -1348.893352938175 |

The final projected gradient infinity norm in the third run is
$4.05099\times10^{-4}$. This is a correctness improvement and a substantial
reduction in outer iterations relative to the broken intermediate version,
but **not** a restoration of the old computational efficiency. The third run
overlapped diagnostics, so its wall time is not used as a controlled speedup
measurement.

The complete four-input regression after the subspace correction is:

| Input | Coordinate-only iterations / HVPs | Consistent-subspace iterations / HVPs | Final energy with consistent subspace (Eh) |
|---|---:|---:|---:|
| F2 | 6 / 20 | 6 / 20 | -198.751155830527 |
| 241 | 9 / 108 | 9 / 108 | -230.720590392864 |
| MnF2 | 119 / 927 | 44 / 911 | -1348.893352938175 |
| FeCl2 | 6 / 193 | 18 / 353 | -2181.617641557797 |

FeCl2 therefore also has a performance regression, although it reaches a lower
energy at the unchanged stopping tolerances. All four runs converge, but this
does **not** satisfy an across-system performance acceptance criterion. The
consistency correction must not be described as an overall optimizer speedup.

Trust-radius diagnostics also show slow radius recovery after descent fallback:
in the 119-step run, iteration 22 reduced the radius from about 0.11 to 0.0017.
The maximum-curvature Cauchy length used to limit subsequent expansion can be
very small along soft modes. Two scratch-only model-remainder radius experiments
with the old, inconsistent subspace cache took 76 and 80 iterations (825 and
1019 HVPs), respectively. Neither has been adopted. Correcting subspace
consistency takes precedence over changing the radius policy. Remaining work
includes finite negative-curvature subproblem handling, Newton residual accuracy,
and preconditioner/metric quality with the exact quotient; no molecule-specific
limits or looser convergence tolerances were introduced.

## Projected trust-region consistency correction

The next retained correction replaces the small spectral trust-region solver,
not the matrix-free orbital Hessian or the CG subspace generator. The former
solver could reject a valid singular endpoint because its positive shift offset
was smaller than its denominator rejection threshold. For example, the model
with diagonal Hessian $(-2,3)$, gradient $(0,1)$, and radius one has the global
solutions $(\pm\sqrt{0.96},-0.2)$ with multiplier two. The old endpoint logic
could reject this model instead of constructing its negative-curvature boundary
component. This counterexample is independent of a molecular benchmark.

The replacement uses the pseudoinverse at a compatible spectral endpoint and
adds the minimum-eigenvalue component in the indefinite hard case. Otherwise,
it brackets the secular root, retaining the positive excess shift separately
from the spectral lower bound. Normalizing the radius and energy scale avoids
squaring a very large radius; the right-hand side also protects against an
underflowing intermediate at a very small radius. Tiny positive eigenvalues
are not replaced by an empirical curvature floor. Pure negative-curvature
steps are eligible when their predicted reduction is positive, even when their
linear energy term is zero. The projected-Hessian symmetrization now explicitly
materializes its expression to avoid Eigen transpose aliasing.

Equations 65c and 65d in the theory note give the projected-model optimality
conditions and their global sufficiency proof. This is a guarantee for the
small quadratic model in exact arithmetic, not a global second-order guarantee
for the nonlinear VBSCF optimizer. The outer stopping rule remains first order.

`test_spectral_trust_region` passes 17 cases covering positive-definite interior
and boundary solutions, a zero-multiplier boundary, compatible and incompatible
semidefinite models, indefinite regular and hard cases, pure negative curvature,
zero and linear models, a repeated minimum eigenvalue, a tiny positive mode,
large-radius near-pole behavior, small radii with ordinary and disparate scales,
and two energy rescalings. Checks include feasibility, stationarity, shifted
positive semidefiniteness, and complementarity. The disparate-scale small-radius
case failed before the underflow protection and passes afterward. The same
17 cases pass AddressSanitizer and UndefinedBehaviorSanitizer with leak detection.
The four selected CTests, including the quotient, HVP-basis, and F2 finite-
difference regressions, also pass after the final source edit.

The retained implementation gives the following complete runs at unchanged
input and convergence settings:

| Input | Previous consistent-subspace iterations / HVPs | Spectral correction iterations / HVPs | Final energy (Eh) | Full KKT target met / fresh inner solves |
|---|---:|---:|---:|---:|
| F2 | 6 / 20 | 6 / 20 | -198.751155830527 | 5 / 6 |
| 241 | 9 / 108 | 9 / 105 | -230.720590392802 | 3 / 9 |
| MnF2 | 44 / 911 | 37 / 865 | -1348.893352987854 | 5 / 37 |
| FeCl2 | 18 / 353 | 18 / 345 | -2181.617641518378 | 0 / 18 |

The post-hoc KKT diagnostic evaluates the full reduced-coordinate residual of
the final step, including the trust-region shift, against the current forcing
fraction times the gradient 2-norm (eq 65e). The production CG stopping test
still uses an infinity norm, so these counts are not counts of failures of the
existing CG criterion. Cached radius retries are excluded from the denominator.
The low fractions identify inner-solve accuracy and preconditioning as issues
to investigate; they do not alone prove the cause of every rejected outer step.

These logs are under `/tmp/xmvb-spectral-validation.eM56b8/final/` on the test
machine. F2 used one OpenMP thread and the other inputs used four; OpenBLAS used
one thread. Large runs overlapped, so no wall-time speedup is claimed. The final
underflow guard was added after these runs; it leaves ordinary-scale arithmetic
unchanged and was verified by the final test suite, not another full four-input
run. The final MnF2 projected gradient infinity norm is
$6.48237683\times10^{-4}$, versus $4.05099\times10^{-4}$ in the prior 44-step
run, so the iteration reduction must not be described as higher final gradient
accuracy. Both runs satisfy the same existing stopping tolerances.

### Residual-expanded candidate not adopted

A separate matrix-free candidate expanded the subspace using the full
preconditioned trust-region KKT residual and solved the small trust-region model
after each expansion. It passed synthetic quadratic-model tests, but complete
runs gave 5 / 16, 9 / 149, 41 / 1238, and 16 / 512 iterations / HVP directions
for F2, 241, MnF2, and FeCl2, respectively. In particular, MnF2 used more HVP
work than the retained implementation. Its final energy was
-1348.893352864050 Eh; FeCl2 reached -2181.617643876471 Eh, so endpoint accuracy
also differed. This experiment does not demonstrate a general efficiency gain.
The candidate source and logs are preserved only in the scratch directory
`/tmp/xmvb-kkt-validation.KotrQs/`; no unused production branch or configuration
switch remains in the repository.

This stage therefore retains an independently testable correctness repair with
a modest reduction in HVP work, not the proposed replacement of CG. The older
18-step MnF2 reference is still not matched. Coordinate-consistent metric and
preconditioner construction, together with the relationship between the actual
final-step residual and inner stopping policy, remain the next algorithmic work.
No molecule-specific budget, relaxed outer tolerance, or new empirical tuning
parameter was introduced.

## Coordinate-consistent preconditioning and curvature reuse

### Retained changes and mathematical checks

The local preconditioner previously used the projected effective-Fock difference
without the complete derivatives of orbital normalization. A Euclidean quotient
basis need not be orthogonal to the orbital in the AO-overlap metric, so the
normalization derivative terms do not generally vanish. The old local orbital
norm and energy also excluded stored but frozen coefficients. The replacement
implements the full additive-chart Hessian of a frozen normalized one-electron
Rayleigh quotient, eqs 66a--66c of the theory note. This is an exact derivative
of an **approximate surrogate**, not a new exact many-electron Hessian.

`test_normalized_orbital_curvature` independently differences the analytic
surrogate gradient, checks orbital-rescaling and quotient-basis covariance,
and includes a fixed-coefficient tail. Its fixtures explicitly distinguish the
new expression from both an omitted-normalization-term expression and an
incorrectly omitted frozen tail. Empty tangents and invalid orbital norms are
also covered. Only per-orbital surrogate blocks are constructed; no dense
global Hessian is introduced.

The inner forcing and CG residual now use the Euclidean 2-norm, consistent with
the spherical quotient trust region. The previous infinity norm depended on an
arbitrary orthogonal choice of quotient basis. Outer stopping tolerances and
their infinity-norm gradient criterion are unchanged. The inherited forcing
bounds and 32-direction safety limit are also unchanged; neither establishes
quadratic convergence.

Two-pass full conjugacy restoration replaces the three-term CG direction
recurrence, using already evaluated direction images (eqs 66d and 66e). The
residual is recomputed from the accumulated step image rather than updated by
successive residual subtractions. `test_positive_conjugate_basis` checks
positive-definite quadratic models at condition numbers 100 and one million,
verifying conjugacy, a fresh final residual, and exactly one HVP per direction.
Negative curvature is rejected by the positive-direction cache, not discarded
from the trust-region model. This adds linear-in-dimension subspace storage
and vector work, but no new HVPs per admitted direction.

Before discarding an accepted-point HVP subspace, its soft positive Ritz modes
are converted to preconditioning secants (eqs 66f and 66g). Full HVP images,
including components outside the Ritz subspace, are retained. These pairs are
packed and transported with the existing step/covector transformations before
projection into the next quotient. They are approximate next-point
preconditioning data; the next quadratic model still uses fresh exact HVPs.
The existing history capacity of eight is unchanged, with at most seven Ritz
pairs added to leave room for an accepted-step secant. Rank changes clear the
history and positive curvature is checked again after projection.

`test_positive_ritz_secants` covers indefinite and zero modes, memory limits,
off-subspace HVP components, invariance of a nondegenerate selected subspace
under a change of its orthogonal basis, and exact inverse-BFGS recovery of a
complete small positive-definite quadratic model. All three new test programs
pass Release, AddressSanitizer, UndefinedBehaviorSanitizer, and leak detection.
All seven selected CTests pass in the integrated final build.

### Default-tolerance results and limitations

The complete four-input comparison uses the same inputs, exact-context HVP
mode, outer thresholds, history capacity, and inner work limit as the preceding
spectral-correction implementation:

| Input | Previous iterations / HVPs | New iterations / HVPs | New final total energy (Eh) | Full KKT target met / fresh inner solves |
|---|---:|---:|---:|---:|
| F2 | 6 / 20 | 5 / 14 | -198.751155830471 | 5 / 5 |
| 241 | 9 / 105 | 10 / 106 | -230.720590392883 | 6 / 10 |
| MnF2 | 37 / 865 | 21 / 437 | -1348.893352973132 | 6 / 21 |
| FeCl2 | 18 / 345 | 18 / 576 | -2181.617644595395 | 0 / 18 |

MnF2 uses 43% fewer outer iterations and 49% fewer HVP directions than the
preceding correct-coordinate implementation. Its final projected gradient
infinity norm improves from $6.48237683\times10^{-4}$ to
$2.61846077\times10^{-4}$, while its final energy is higher by approximately
$1.47\times10^{-8}$ Eh. This is close to, but does not beat, the old-coordinate
18-iteration, 413-HVP reference.

241 is slightly worse at default settings: one more outer iteration and one
more HVP. FeCl2 has substantially more HVP work at these settings, but reaches
an energy lower by approximately $3.08\times10^{-6}$ Eh and reduces its final
projected gradient infinity norm from $2.33502033\times10^{-4}$ to
$7.51042511\times10^{-5}$. Its 18 default iterations are therefore **not** a
same-accuracy speedup. The post-hoc KKT count remains zero: the final steps are
limited by inner work rather than certified at the requested forcing accuracy.
This remains an algorithmic bottleneck, not a reason to change a molecule's
iteration budget.

F2 also illustrates why iteration counts need endpoint information: its new
final projected gradient infinity norm is $1.75633661\times10^{-5}$, versus
$1.33975067\times10^{-7}$ previously. Both satisfy the unchanged gradient
tolerance, and their energies differ by only about $5.6\times10^{-11}$ Eh, but
the five-step run is not a comparison at equal final gradient accuracy.

The final extracted-helper executable reproduces the prototype's default
counts and energies. The four relaxed-HVP finite-difference checks pass after
integration; the three larger systems use step $10^{-5}$ and the same
$10^{-7}$ relative-error limit. Scalar and batched images still agree to the
printed precision. A sequential before/after MnF2 timing pair using four OpenMP
threads and one OpenBLAS thread measured 155.39 s and 85.07 s of SCF time.
Other diagnostic runs overlapped on the node, so this is indicative timing,
not an isolated or statistically replicated speedup measurement. Logs and
executable snapshots are in `/tmp/xmvb-preconditioner-validation.PrNFrI/`.

### Stricter FeCl2 diagnostic: different trajectories, not a speedup claim

An additional paired diagnostic set the same gradient tolerance $10^{-4}$ and
energy-change tolerance $10^{-8}$ Eh for both implementations. This did not
alter production defaults. The preceding spectral-correction implementation
met those criteria after 86 iterations and 2082 HVP directions, at
-2181.617645882917 Eh with projected gradient infinity norm
$3.82142411\times10^{-5}$.

The new implementation initially approached that energy plateau, then continued
into a substantially lower-energy region with a growing gradient. This no
longer provides an equal-endpoint timing comparison. The optional diagnostic
was deliberately terminated with SIGTERM after accepted iteration 70; its last
accepted energy was -2181.620692903083 Eh and its printed gradient infinity norm
was about $5.9993\times10^{-2}$. It is **not a converged result**, does not
certify a minimum, and is excluded from convergence and speedup tables. The
post-plateau numerical conditioning and physical interpretation have not been
separately audited. The full partial trajectory remains in
`FeCl2.combined-tight.log`; the completed reference is in
`FeCl2.baseline-tight.log` under the scratch directory above. No diagnostic
process was left running.

This observation reinforces a limitation of the existing dual first-order
stopping rule: small energy changes and a small coordinate gradient do not
certify second-order stationarity or rule out later progress in soft directions.
It does not justify loosening a threshold, claiming a successful strict run, or
comparing runs at different endpoints as equivalent convergence performance.

### Isolated alternatives and why they were not retained separately

All prototypes used the same four inputs and default outer tolerances. Entries
below are outer iterations / HVP directions; differing endpoint accuracy must
be considered before interpreting them as efficiency comparisons.

| Prototype | F2 | 241 | MnF2 | FeCl2 |
|---|---:|---:|---:|---:|
| Latest-secant scaling of the old base inverse only | 6 / 24 | 10 / 128 | 30 / 730 | 18 / 347 |
| Soft positive Ritz recycling with the old local model | 6 / 22 | 8 / 99 | 34 / 798 | 18 / 326 |
| Generalized Ritz recycling relative to the old base model | 6 / 21 | 9 / 129 | 37 / 837 | 18 / 348 |
| Euclidean inner norms only | 5 / 16 | 10 / 122 | 37 / 860 | 18 / 345 |
| Complete normalized surrogate plus Euclidean inner norms | 6 / 18 | 11 / 125 | 18 / 452 | 8 / 169 |
| Previous row plus conjugacy restoration | 6 / 18 | 11 / 125 | 18 / 447 | 19 / 608 |
| Retained combination, including soft positive Ritz recycling | 5 / 14 | 10 / 106 | 21 / 437 | 18 / 576 |

In particular, the apparently fast eight-iteration FeCl2 prototype stops at
-2181.617639243877 Eh, which is materially above both the previous production
endpoint and the retained result. It was not selected based on its short run.
The scalar-rescaling and generalized-Ritz variants were not retained as
optional code paths. Scratch executables and logs for these experiments are
under `/tmp/xmvb-preconditioner-validation.PrNFrI/` on the test machine.
The retained helpers separate surrogate mathematics, conjugacy restoration,
and Ritz-pair selection from the optimizer driver.

## Inactive-projected preconditioner

### Correction and verification

The preceding normalized one-electron surrogate still described the raw
orbital ray, whereas the physical variables also involve the inactive-space
projector. The new local model projects out the other inactive orbitals for
an inactive target, and the full inactive span for an active target. It then
applies the complete normalization derivatives to that projected representative.
The target's strict sparse support and stored/frozen coefficient distinction
are unchanged. Equations 66h--66j in the theory note define the model and prove
its frozen-target projector-trace interpretation by a block-diagonal Gram
decomposition.

`projected_orbital_surrogate.hpp` contains the support-column projection and
local matrix construction. The excluded-space Gram matrix is factored, not
explicitly inverted. AO matrices are passed as Eigen references to avoid
copying a global overlap matrix for every target. The obsolete raw block-Fock
extraction helpers and their unused construction have been removed. There is
no alternate unprojected production branch. Only the preconditioner changed;
the exact relaxed HVP, CG and recycling policies, inner limit, forcing rule,
trust-radius update, and outer stopping thresholds are unchanged in this stage.

`test_projected_orbital_surrogate` independently evaluates the full occupied
projector trace, rather than differentiating the implementation's local
matrices as its reference. Tests check energy agreement, directional energy
second differences, excluded-span gauge annihilation, nonsingular changes of
the excluded inactive basis, target additions from that span, strict supports,
frozen tails, exclusion of the target, and an empty excluded span. A singular
excluded span is rejected. The same fixture explicitly shows that the prior
unprojected normalized curvature does not annihilate the fixed inactive gauge.
The test passes Release, ASan, UBSan, and leak detection. All eight selected
integrated CTests pass. These are surrogate identities and numerical tests,
not a proof that a block-diagonal surrogate equals the relaxed VBSCF Hessian.

### Complete default-tolerance comparison

| Input | Previous iterations / HVPs | Projected-model iterations / HVPs | Final energy (Eh) | Final projected gradient infinity norm | KKT target met / fresh solves |
|---|---:|---:|---:|---:|---:|
| F2 | 5 / 14 | 5 / 14 | -198.751155830527 | $5.00511623\times10^{-8}$ | 5 / 5 |
| 241 | 10 / 106 | 10 / 39 | -230.720590392894 | $1.89165638\times10^{-8}$ | 8 / 10 |
| MnF2 | 21 / 437 | 19 / 255 | -1348.893353224063 | $1.11509905\times10^{-5}$ | 8 / 19 |
| FeCl2 | 18 / 576 | 7 / 224 | -2181.617645388120 | $6.92748201\times10^{-5}$ | 0 / 7 |

Relative to the immediately preceding implementation, HVP work decreases by
approximately 63%, 42%, and 61% for 241, MnF2, and FeCl2, respectively. No input
uses more outer iterations or HVP directions, and every final projected
gradient infinity norm is smaller. Final energies are also lower to the
reported precision. MnF2 is now close to the old-coordinate reference in outer
iterations (19 versus 18), with substantially fewer HVPs (255 versus 413).
That comparison does not imply that the old coordinates were mathematically
valid or establish a general near-quadratic convergence rate.

FeCl2 still uses the full 32-direction inner budget on every default iteration
and does not meet the post-hoc KKT forcing target. Its flat-region stopping
limitations from the stricter diagnostic above remain unresolved. The new
preconditioner does not modify or certify the first-order outer stopping rule.
These results support this preconditioning correction on the four inputs,
not an across-problem guarantee or a full replacement of the inner algorithm.

The prototype and final executable snapshots, input copies, and logs are under
`/tmp/xmvb-augmented-validation.kARnFJ/`. The final executable reproduces the
table's counts, energies, and projected gradients on all four inputs; every
final run exits successfully. The initial projected-model runs overlap
and their wall times are not used as isolated speedup measurements. The final
build also repeats the relaxed-HVP finite-difference checks on the larger
inputs at step $10^{-5}$ with the same relative-error threshold $10^{-7}$;
the F2 check remains in CTest at step $10^{-4}$.
The final full-HVP relative errors are $6.9951\times10^{-8}$,
$5.6890\times10^{-9}$, and $1.6913\times10^{-9}$ for MnF2, FeCl2, and 241,
respectively, with zero printed batch-versus-scalar discrepancy.

A sequential matched MnF2 timing pair used four OpenMP threads and one
OpenBLAS thread, with no other agent-run molecular calculation overlapping
either member of the pair. The preceding executable took 75.935746 s of SCF
time (21 iterations, 437 HVP directions); the final projected-model executable
took 47.026873 s (19 iterations, 255 HVP directions). This is approximately 38%
less SCF time in a single matched pair, not a statistically replicated timing
study or a claim of exclusive access to the host. The logs are
`MnF2.baseline-timed.log` and `MnF2.final.log` in the scratch directory above.

### Refreshed starting-subspace experiment not adopted

A separate prototype transported historical directions into the current
quotient, refreshed their images with current HVPs, solved an initial coarse
trust-region model, and continued CG in its positive-curvature conjugate
complement. The refreshed directions counted against the existing total inner
budget, not an added allowance. It used the preceding unprojected local model.
The results were 5 / 37, 10 / 166, 22 / 536, and 16 / 512 iterations / HVPs for
F2, 241, MnF2, and FeCl2. Most inputs used more HVP work, so it was not retained.
A drafted block-refresh helper was also removed without being adopted; it has
no claimed molecular benchmark result. Their scratch records are preserved,
but no unused algorithm switch or helper remains in production. This experiment
does not rule out adaptive coarse spaces with better selection or a better
base model; it does show that refreshing all available history is not by itself
an efficiency improvement on these inputs.

## Inactive double-occupancy weighting

The preceding local model used an unweighted projector trace for every
orbital. The inactive projector density in the implementation does not itself
contain a factor of two. Differentiating the actual reference-energy
convention gives $\mathrm dE_{11}=2\operatorname{Tr}(\mathbf F_{11}\,
\mathrm d\mathbf P_{\mathrm I})$. Therefore the frozen-field, target-local
curvature contribution is twice the unweighted Rayleigh curvature. Production
now applies this factor to inactive blocks before positive spectral
regularization; active blocks retain the unit-ray approximation. The derivation
and the separate field-response term are in section 10.5, eqs 66k--66m, of the
theory note. The factor is fixed by double occupancy, not estimated from a
molecule or fitted to these runs. This still does not supply exact active
occupations, field response, cross-orbital coupling, or the relaxed VB Hessian.

No quotient, HVP formula, L-BFGS history size, trust-radius rule, inner budget,
or outer stopping tolerance changed in this retained step. The default
energy and projected-gradient thresholds remain $10^{-7}$ Eh and $10^{-3}$.
All runs use four OpenMP threads and one OpenBLAS thread. The experimental
outer safety cap was 100 rather than 2000; every run converged well below it.

| Input | Previous iterations / HVPs | Weighted-model iterations / HVPs | Final energy (Eh) | Final projected gradient infinity norm | KKT target met / fresh solves |
|---|---:|---:|---:|---:|---:|
| F2 | 5 / 14 | 5 / 11 | -198.751155830527 | $2.15458526\times10^{-7}$ | 5 / 5 |
| 241 | 10 / 39 | 10 / 39 | -230.720590392895 | $8.65153897\times10^{-8}$ | 6 / 10 |
| MnF2 | 19 / 255 | 14 / 165 | -1348.893353224164 | $6.07151757\times10^{-6}$ | 9 / 14 |
| FeCl2 | 7 / 224 | 6 / 192 | -2181.617645346998 | $3.35497053\times10^{-5}$ | 0 / 6 |

MnF2 uses approximately 26% fewer outer iterations and 35% fewer HVPs, while
reaching a smaller gradient. F2 and 241 stop with somewhat larger gradients
than the preceding implementation, still far below the unchanged threshold.
FeCl2's endpoint is higher in energy by approximately $4.1\times10^{-8}$ Eh
and has a smaller gradient. Its six inner solves still exhaust 32 directions
without satisfying the full residual target. Its first-order plateau and
minimum-certification limitations remain unresolved. These are comparisons
at identical stopping settings, not identical terminal gradients or proof
of asymptotically quadratic convergence.

Two independent regression additions verify the weighting:

- `projected_orbital_surrogate` differentiates an independently evaluated
  projector-density quadratic energy with a self-adjoint interaction map. It
  checks the factor two in the first derivative and separates frozen-field
  geometry from field response in the second derivative.
- `sparse_orbital_quotient` constructs production inactive and active local
  blocks at a stationary one-electron ray. Their actions and inverse actions
  match the independent occupation-weighted gap formula. This test detects
  a missing inactive factor without fitting any molecular result.

All eight existing selected CTests pass, including the two extended tests and
the relaxed exact-HVP F2 finite-difference regression. Executable snapshots,
input copies, and all experimental logs are preserved in
`/tmp/xmvb-multisecant-validation.BKJqrb/`; `final.exe` is the preceding
implementation and `occupation.exe` is the retained weighted implementation.

A sequential MnF2 timing pair, run after compilation completed with no other
agent-run molecular calculation overlapping, gave 47.242690 s for the
preceding implementation and 32.198546 s for the weighted implementation
(approximately 32% less SCF time). Counts reproduced 19 / 255 and 14 / 165.
The logs are `MnF2.baseline-isolated.log` and `MnF2.occupation-isolated.log`.
The filename does not imply exclusive access to the shared host; this is a
single matched pair, not a statistically replicated timing study. An earlier
pair overlapped compilation and is not used for this comparison.

### Simultaneous multisecant experiment not adopted

Before changing the local weighting, a block inverse-BFGS prototype attempted
to preserve all transported secants simultaneously. It orthonormalized their
direction span, removed the in-span skew part of the images to obtain a
symmetric compatibility matrix, and kept its positive modes. Independent
tests passed simultaneous interpolation, positive definiteness, compatible
basis covariance, duplicate-rank handling, skew correction, and complete SPD
inverse recovery. No additional HVPs were needed to build that preconditioner.

Nevertheless, the molecular results were 5 / 12, 9 / 35, 19 / 308, and 7 / 224
iterations / HVPs for F2, 241, MnF2, and FeCl2. In particular, MnF2 regressed
from 255 to 308 HVPs, and FeCl2 still met no inner residual targets. The
prototype was not retained or combined with the occupation weighting. Its
source and logs remain only in scratch; its production helper, test source,
and build registrations were removed. Sequential transported L-BFGS remains
the production history update. This experiment does not establish that
multisecant updates are generally inferior; it rules out this particular
replacement as a uniform improvement over the present baseline.

### HVP cost observation

A three-repeat FeCl2 initial-point profile with one warmup measured about
0.285 s per full HVP, of which about 0.258 s was the outer-response stage.
Within that stage, active-integral variations, structure-matrix variations,
and active-gradient response dominated; the eigensystem response was small.
The profile overlapped other molecular tests and is not an isolated timing
comparison. The tested two-column batch was only about 1.03 times faster than
two scalar calls. This motivates studying the response contractions and
their reusable algebra, not assuming that a block interface alone provides
a large speedup. A small response on this single direction is not evidence
that response can safely be omitted in soft optimization directions.

## Subsequent direction-resolved HVP audit

The [curvature decomposition and consistency audit](matrix_free_curvature_decomposition_diagnostics.md)
finds an unresolved FeCl2 endpoint discrepancy: a fresh HVP of a sampled
direction combination differs from the combination of cached HVPs by about
1.2%, localized to the outer-response contribution. The core contribution
passes the same test near roundoff. Direction-resolved finite differences
also expose a larger error than the earlier gradient-direction checks.
Thus the earlier checks and successful default runs must not be read as a
general certificate of exact-Hessian consistency. No optimizer or HVP fix
was made in the diagnostic turn; the new audit and reproduction commands
document the failure and its current localization.

## Reproduction

From the repository root, build and run the automated regressions:

```bash
cmake --build build --target test_projected_orbital_surrogate test_positive_ritz_secants test_positive_conjugate_basis test_normalized_orbital_curvature test_spectral_trust_region test_orthonormal_hvp_basis test_sparse_orbital_quotient check_exact_ctx_hvp audit_sparse_orbital_gauge run_cpp_vbscf -j 4
ctest --test-dir build -R 'projected_orbital_surrogate|positive_ritz_secants|positive_conjugate_basis|normalized_orbital_curvature|spectral_trust_region|orthonormal_hvp_basis|sparse_orbital_quotient|exact_ctx_hvp_f2_finite_difference' --output-on-failure
```

Audit each input (`src/test_molecule/F2.xmi`, `test/241_VBSCF.xmi`,
`test/MnF2.xmi`, `test/FeCl2.xmi`):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build/src/audit_sparse_orbital_gauge INPUT.xmi --nonredundant-adapt true
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build/src/check_exact_ctx_hvp INPUT.xmi --nonredundant-adapt true --step 1e-5 --probe full --max-rel-error 1e-7
```

For full runs, copy the input to a scratch directory first because requested
Molden output is written next to the input. Keep thread counts and solver
tolerances identical in performance comparisons:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 build/src/xmvb-cpp.exe /path/to/scratch/INPUT.xmi --optimizer-backend nonredundant_truncated_newton --nonredundant-truncated-newton-hvp-mode exact_ctx
```
