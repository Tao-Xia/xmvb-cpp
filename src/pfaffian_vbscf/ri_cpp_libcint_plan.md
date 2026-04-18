# RI/DF Plan On A Pure C++/libcint Path

## Constraint

This plan intentionally does **not** extend or depend on the legacy
`runtime_c` RI/COSX/THC machinery.

The RI implementation should be built on top of the existing clean C++ path:

- `CppVbInput::libcint_input`
- `runtime/libcint_direct_shell_evaluator.*`
- `runtime/libcint_materialized_integral_provider.*`
- `vb/orbital/active_space_two_electron_builder.*`
- `vb/orbital/active_space_two_electron_backpropagator.*`
- `pfaffian_vbscf/kernel/*`

## Why This Direction Is The Right One

For the current closed-shell Pf-VBSCF implementation, the exact active-space
two-electron path spends time in two different places:

1. rebuilding active-space `ggo` from AO four-index ERIs every evaluation
2. contracting the Pf kernel against packed active-space four-index data

On the current `C6H6_full / k=175 / OMP_NUM_THREADS=1` benchmark, the
single-evaluation timing is roughly

- `active_2e_build_dt ~ 0.28 s`
- `pf_matrix_forward_dt ~ 0.28 s`
- `pf_adjoint_dt ~ 0.58 s`

So RI should not be treated as only an AO-integral build optimization. It
should become the production **representation** for the two-electron part:

- AO-side RI data should be built once per molecule
- active-space RI factors should be rebuilt each orbital evaluation
- Pf two-electron forward and adjoint should contract **directly** against the
  RI factors rather than reconstructing packed `ggo`

This is the only path that has a realistic chance to materially beat the
current exact four-index route.

## Core Observation

The AO RI objects are molecule-static, while orbital optimization changes only
the active-orbital transformation.

That means we can split the two-electron pipeline into:

1. **Static molecule-side RI data**
   - auxiliary basis
   - auxiliary metric `(A|B)`
   - metric-whitened three-center AO-pair factors

2. **Dynamic active-space transform**
   - active-orbital coefficients `C_{mu p}`
   - active pair factors `L_{A,P}`

This is much stronger than the earlier "pair-level cache" idea:
the expensive AO-side RI data can be computed once and reused across all SCF
iterations and all line-search objective evaluations.

## Target Mathematical Form

Let `A, B` label auxiliary functions, `mu, nu` AO indices, and `p, q` active
orbital indices.

We introduce the Coulomb metric

```text
J_AB = (A|B)
```

and the metric-whitened AO-pair RI factors

```text
W_A,(mu nu) = sum_B J^(-1/2)_(A B) (B|mu nu)
```

where `(mu nu)` denotes a symmetric AO pair.

Let `P = (p <= q)` be a packed active pair, and define the symmetric AO-pair
to active-pair transform

```text
T_(mu nu),P = C_(mu p) C_(nu q) + C_(mu q) C_(nu p)   for p != q
T_(mu mu),(p p) = C_(mu p) C_(mu p)
```

Then the active-space RI factors are

```text
L_A,P = sum_(mu <= nu) W_A,(mu nu) T_(mu nu),P
```

and the active-space ERI matrix in packed pair form is

```text
g_(P,Q) ~= sum_A L_A,P L_A,Q
```

This factorization is enough to:

- reconstruct packed `ggo` for validation against the exact path
- drive a direct RI two-electron Pf kernel without materializing `ggo`

## Production Strategy

The production implementation should proceed in four phases.

### Phase 1: Pure C++ RI Infrastructure

Add a clean C++ RI provider that takes only primary-basis `LibcintInput` and
returns molecule-static AO-side RI data.

Recommended new components:

- `runtime/libcint_auxiliary_basis_builder.*`
- `runtime/libcint_ri_integral_provider.*`
- `vb/orbital/ri_active_space_two_electron_builder.*`

This phase should **not** touch `runtime_c`.

### Phase 2: Validation Path

Before changing the Pf kernel, validate the RI active transform by rebuilding
packed active-space `ggo` from `L_A,P`:

```text
g_(P,Q) = sum_A L_A,P L_A,Q
```

and compare this reconstructed active-space tensor with the existing exact
`ActiveSpaceTwoElectronBuilder`.

This phase proves that:

- the auxiliary basis generation is correct
- the 2c/3c libcint wrapper is correct
- the AO-pair to active-pair transform is correct

without changing any Pf code yet.

### Phase 3: Closed-Shell Pf RI Kernel

After validation, implement a closed-shell production RI fast path inside
`src/pfaffian_vbscf/kernel/`.

Recommended direction:

- keep the current exact packed-ERI path untouched as the reference
- add a closed-shell RI path specialized to the current production formula
- do **not** first RI-generalize the legacy tensor-term API

The reason is practical: the current production path already bypasses the
generic tensor-term machinery and evaluates only the closed-shell total
two-electron functional. The RI implementation should follow the same choice.

### Phase 4: RI Adjoint

Backpropagation should proceed in the reverse order:

1. Pf RI kernel returns `dE / dL_A,P`
2. active-space RI builder backpropagates `dE / dL_A,P` to
   `dE / dC_(mu p)`
3. orbital backprop continues exactly as in the current C++ pipeline

Because the AO-side RI factors are molecule-static, the first production
version does **not** need RI integral derivatives. Orbital optimization only
needs derivatives through the active-space transform.

This is a major simplification.

## Recommended Data Model

The current `ActiveSpaceTwoElectronResult` has already been extended to support
both exact and RI representations.

The intended production use is:

- `representation == PackedExact`
  - use `packed_active_two_electron_integrals`
- `representation == ResolutionOfIdentity`
  - use `n_auxiliary_functions`
  - use `ri_active_pair_factors`

The RI factor buffer should be stored row-major as

```text
[auxiliary_function][packed_active_pair]
```

so that:

- rebuilding packed `ggo` is a Gram product
- direct RI Pf contractions can stream over auxiliary rows
- reverse-mode accumulation into `dL` remains contiguous

## Auxiliary Basis Policy

Since this plan avoids `runtime_c`, the auxiliary basis must also come from the
clean C++ side.

The recommended first implementation is:

1. implement a C++ auxiliary-basis generator that mirrors the current
   `GEN-A_n` style construction used by the old code
2. build auxiliary `atm/bas/basidx/env` arrays in the same shape as
   `LibcintInput`
3. feed those arrays into libcint 2c/3c evaluators

This avoids:

- exporting hidden legacy runtime state
- coupling RI production code to the old runtime
- correctness risk from mixed C/C++ integral ownership

For the first C++ production version, it is acceptable to support only one
well-defined auxiliary policy, for example a generated Coulomb-fitting basis,
and fail fast for unsupported cases.

## Shell Evaluator Extensions

`LibcintDirectShellEvaluator` currently exposes:

- overlap shell pairs
- core-H shell pairs
- four-center shell quartets

The RI implementation needs two additional shell evaluators:

- auxiliary metric shell pairs `(A|B)`
- three-center shell blocks `(A|mu nu)` or `(mu nu|A)`

These should be added as new C++ methods on top of libcint’s 2c2e/3c2e entry
points.

This keeps all integral generation inside the same C++ ownership model as the
existing materialized libcint provider.

## Active-Space RI Builder

The RI active-space builder should have the same role as the current exact
`ActiveSpaceTwoElectronBuilder`, but with different outputs.

Its responsibilities should be:

1. build or reuse dense active-orbital AO coefficients
2. build symmetric AO-pair to active-pair coefficients
3. contract AO-side RI factors into active-space RI factors `L_A,P`
4. optionally reconstruct packed `ggo` in validation mode

The dynamic cost per objective evaluation then becomes:

- build active-pair transform from orbital coefficients
- contract molecule-static AO RI factors into active-pair RI factors

The AO-side 2c/3c integral generation should no longer appear in every SCF
iteration.

## Backpropagation Strategy

For orbital optimization, the reverse-mode path should avoid any derivative of
the AO integrals themselves.

The dynamic reverse graph is only:

```text
orbital coefficients -> active pair transform -> active RI factors -> Pf energy
```

So the production adjoint should:

1. receive `dE / dL`
2. backpropagate to the symmetric AO-pair transform
3. backpropagate to dense active AO coefficients
4. expand back to the full auxiliary orbital matrix, just as the current exact
   backpropagator does

This means the RI transition can reuse most of the existing orbital-gradient
framework.

## Suggested File-Level Integration Points

### Runtime / Integral Layer

- `src/runtime/libcint_direct_shell_evaluator.hpp`
- `src/runtime/libcint_direct_shell_evaluator.cpp`
- new `src/runtime/libcint_auxiliary_basis_builder.*`
- new `src/runtime/libcint_ri_integral_provider.*`

### Active-Space Layer

- `src/vb/orbital/active_space_two_electron_result.hpp`
- `src/vb/orbital/active_space_two_electron_builder.hpp`
- `src/vb/orbital/active_space_two_electron_builder.cpp`
- `src/vb/orbital/active_space_two_electron_backpropagator.hpp`
- `src/vb/orbital/active_space_two_electron_backpropagator.cpp`

### Pf Layer

- `src/pfaffian_vbscf/kernel/pf_closed_shell_spatial_kernel.cpp`
- `src/pfaffian_vbscf/kernel/pf_adjoint_kernel.cpp`
- optionally new `src/pfaffian_vbscf/kernel/pf_closed_shell_ri_kernel.*`

### SCF Wiring

- `src/pfaffian_vbscf/scf/pf_active_grad_eval.cpp`
- `src/pfaffian_vbscf/scf/pf_spin_adapted_active_grad_eval.cpp`

## Recommended Execution Order

The most defensible order is:

1. add pure C++ auxiliary-basis builder
2. add libcint 2c/3c RI provider
3. add active-space RI factor builder
4. add exact-vs-RI active-space validation tool
5. add closed-shell Pf RI forward
6. add closed-shell Pf RI adjoint
7. wire RI into `pf_active_grad_eval`

This order keeps every stage testable against the current exact path.

## Immediate Next Step

The next code step should be:

1. implement the pure C++ auxiliary-basis builder
2. extend `LibcintDirectShellEvaluator` with 2c/3c evaluators
3. build a standalone validation tool that compares
   RI-reconstructed active `ggo` against the current exact `ggo`

Only after that should the Pf kernel itself be switched to RI.
