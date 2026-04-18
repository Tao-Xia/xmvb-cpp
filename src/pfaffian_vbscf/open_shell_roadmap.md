# Arbitrary-Spin Pfaffian-VBSCF Roadmap

## 1. Goal

The next major objective of `pfaffian_vbscf` is to extend the current
closed-shell singlet implementation to open-shell systems while preserving the
same basic asymptotic cost profile:

- pairwise overlap / Hamiltonian matrix elements should remain free of
  determinant expansion;
- the dominant two-electron work should remain `O(M^4)` in the number of
  active spatial orbitals `M`;
- analytic active-space and orbital gradients should be derived from the same
  collapsed forward graph;
- spin adaptation should be treated as a later outer layer, not mixed into the
  first open-shell production kernel.

The recommended development order is therefore:

1. support fixed-`M_s` open-shell states;
2. derive exact forward and adjoint matrix elements for those states;
3. optimize the code architecture around generic spin-block kernels;
4. only then introduce spin-adapted recoupling.

---

## 2. Current Status and Main Constraints

The present code already contains enough structure to motivate the next step,
but it is still explicitly closed-shell in the production path.

### 2.1 What already exists

- `PfBasisData` already stores `n_alpha` and `n_beta`, so the top-level basis
  object is not intrinsically closed-shell:
  `src/pfaffian_vbscf/types/pf_basis_data.hpp`.
- The legacy prototype code contains a spin-resolved overlap projection idea
  based on `alpha/beta` sector sampling:
  `src/pfaffian_vbscf/matrices/pf_pair_kernel_common.hpp`.
- The current production kernel has a validated `O(M^4)` closed-shell
  two-electron fast path and its analytic adjoint:
  `src/pfaffian_vbscf/kernel/pf_forward_kernel.cpp`,
  `src/pfaffian_vbscf/kernel/pf_adjoint_kernel.cpp`.
- `op_spin.md` already shows that mixed-spin two-electron trace terms can be
  collapsed to a finite number of `O(M^4)` contractions.

### 2.2 What is still fundamentally closed-shell

- `PfState` stores only one packed antisymmetric spin-orbital matrix:
  `src/pfaffian_vbscf/data/pf_state.hpp`.
  This is sufficient for the current singlet `AB/BA` representation, but it
  does not explicitly encode blocked open-shell orbitals or later spin-coupling
  data.
- `build_structure_pf_basis(...)` explicitly rejects `n_alpha != n_beta`:
  `src/pfaffian_vbscf/scf/pf_basis_factory.cpp`.
- `PfKernelCache` contains many `closed_shell_*` intermediates instead of a
  generic spin-block description:
  `src/pfaffian_vbscf/kernel/pf_kernel_cache.hpp`.
- `TraceProjector` is now a clean 1D projector for pair number. That is good
  and should be preserved. The old Vandermonde spin-grid route should be kept
  only as a validation oracle, not revived as the production formalism:
  `src/pfaffian_vbscf/math/trace_projector.hpp`.

---

## 3. Recommended Scientific Strategy

### 3.1 Stage A: Fixed-`M_s` open-shell first

The first production target should not be "fully spin-adapted arbitrary spin".
That would mix two separate problems:

- orbital / Pfaffian algebra for open-shell states;
- spin recoupling / spin adaptation between different couplings.

The right first target is:

- exact matrix elements for fixed-`M_s` open-shell Pfaffian-VB states;
- first high-spin `M_s = S`;
- then general fixed `M_s` with both alpha and beta blocked electrons;
- only after that, add a spin-adapted outer layer.

This keeps the first expansion mathematically controlled and code-wise
tractable.

### 3.2 State representation before tensor formulas

The central design principle is:

- do not start by hacking the closed-shell two-electron formulas;
- first introduce a state representation that can distinguish
  singlet-pair structure from blocked open-shell structure.

Without this, the open-shell forward path and its gradient will be unstable and
hard to organize.

### 3.3 Preserve the current 1D pair-number projection

The current 1D trace recursion is one of the strongest parts of the present
implementation. The recommended open-shell theory should keep the same pair
projection whenever possible:

- the singlet-pair sector remains projected by the same `t^n` recursion;
- blocked open-shell electrons enter as low-rank boundary data, not as a new
  multidimensional interpolation problem;
- the old spin-grid/Vandermonde machinery should remain only as a test oracle.

### 3.4 Accept a constant-factor increase, but not a scaling increase

Open-shell support will almost certainly destroy the very special closed-shell
collapse to a single spatial matrix `G/H`.

That is acceptable.

The real requirement is:

- the number of spin blocks and blocked-orbital auxiliaries must remain finite
  and independent of `M`;
- the matrix-polynomial work must remain `O(n M^3)`;
- the dominant two-electron contractions must still be reducible to a fixed
  number of packed `O(M^4)` kernels.

The target is therefore not "one matrix again", but "constant-size spin-block
algebra".

---

## 4. Recommended Mathematical Program

### Phase 0. Freeze the current closed-shell path as a reference

Before open-shell development starts, preserve the current production singlet
path as a regression baseline.

Deliverables:

- keep the present closed-shell pairwise forward and adjoint kernels intact;
- keep `F2` and `C6H6` as regression tests;
- add small unit-style checks that compare closed-shell generic-block formulas
  to the current specialized path.

Purpose:

- later open-shell refactors can be validated against the existing singlet
  implementation without ambiguity.

### Phase 1. Introduce a blocked open-shell state model

For a fixed-`M_s` VB structure, separate:

- the singlet-pair sector;
- the blocked open-shell orbitals.

Recommended data model:

- one pair descriptor for the singlet-pair part;
- one explicit list or coefficient matrix for unpaired alpha orbitals;
- one explicit list or coefficient matrix for unpaired beta orbitals;
- no spin-adapted coefficients yet.

Immediate code implication:

- `PfState` must stop being "only one packed antisymmetric matrix";
- it should become a structured state object with both pair and blocked data.

### Phase 2. Derive fixed-`M_s` overlap and 1-RDM

The first theoretical milestone should be:

- exact pairwise overlap for blocked open-shell Pfaffian-VB states;
- exact transition 1-RDM;
- exact one-electron matrix element;
- exact active-space overlap derivative.

Recommended route:

- factor the blocked open-shell contribution through a low-rank overlap block;
- reduce the remaining paired sector to the same kind of 1D pair-number
  projection already used by `TraceProjector`;
- validate first against determinant-expanded oracle code on tiny systems.

This phase is where the key structural question gets answered:

- whether the blocked orbitals can be removed by a Schur complement or
  equivalent low-rank elimination, leaving an effective pair kernel.

### Phase 3. Derive the full two-electron fixed-`M_s` kernel

Only after overlap and 1-RDM are stable should the two-electron path be
implemented.

The open-shell two-electron matrix element should be decomposed into:

- blocked-blocked terms;
- pair-pair terms;
- pair-blocked cross terms.

Requirements:

- each class must reduce to a finite number of `m x m` block products and
  `O(M^4)` tensor contractions;
- same-spin and opposite-spin contractions should reuse the existing packed
  contraction machinery where possible;
- the formulas in `op_spin.md` should be treated as the prototype for the
  mixed-spin cross part.

### Phase 4. Reverse differentiation of the fixed-`M_s` kernel

The adjoint should not be bolted on afterward. As soon as the open-shell
forward graph is finalized, derive the reverse sweep through:

- blocked-overlap factors;
- Schur-complement or effective-metric intermediates;
- pair-sector trace recursion;
- open-shell matrix polynomials;
- packed two-electron contractions.

Validation:

- compare active-space derivatives against finite differences on small
  open-shell molecules;
- verify orbital gradients before trying full geometry or large SCF studies.

### Phase 5. Integrate the new kernel into SCF/orbital optimization

Only after the pairwise forward and adjoint objects are stable should they be
connected to the production SCF code path.

Key checks:

- generalized eigenproblem stability with non-orthogonal open-shell Pfaffian
  bases;
- orbital optimization convergence for simple radicals / high-spin test cases;
- performance profile versus determinant-expanded reference calculations.

### Phase 6. Add spin adaptation as an outer recoupling layer

Spin adaptation should be treated as a separate layer above the fixed-`M_s`
kernel.

Recommended principle:

- keep the orbital kernel unaware of spin recoupling coefficients;
- compute fixed-`M_s` matrix elements first;
- assemble spin-adapted matrix elements through a small recoupling transform.

This is cleaner mathematically and avoids contaminating the `O(M^4)` kernel
with combinatorics that belong to spin algebra, not orbital algebra.

### Phase 7. Selected-VB after open-shell forward/adjoint are stable

Selected-VB should be postponed until the fixed-`M_s` forward and gradient
framework is working.

Reason:

- selected-VB is a structure-space problem;
- arbitrary-spin support is a kernel problem;
- combining both too early will make debugging nearly impossible.

---

## 5. Recommended Code-Architecture Refactor

### 5.1 Data model

The present `PfState` should evolve toward something like:

- pair-sector payload;
- blocked-alpha payload;
- blocked-beta payload;
- later optional spin-coupling metadata.

The state object should describe chemistry-level structure, not only a decoded
antisymmetric matrix.

### 5.2 Kernel cache

`PfKernelCache` should be split conceptually into:

- generic spin-orbital kernel data;
- generic spin-block / blocked-orbital intermediates;
- optional specialized closed-shell cache fields.

The existing `closed_shell_*` fields should not be used as the master design
for open-shell support. They should become one specialization inside a more
general cache architecture.

### 5.3 Forward kernel API

The current API should eventually distinguish:

- generic fixed-`M_s` cache building;
- closed-shell specialization;
- later spin-adapted assembly.

The important design point is that the production interface should expose a
generic pairwise kernel object, not force all paths through the current
closed-shell specialization.

### 5.4 Tensor contractions

The tensor contractor is already close to what is needed conceptually.

Recommended direction:

- keep the packed `O(M^4)` contraction backend;
- add generic helpers for low-rank / blocked cross terms;
- avoid reintroducing any loop nesting that mixes four-index integral loops
  with `O(M^2)` or `O(M^3)` matrix builds.

### 5.5 Validation oracles

Keep the following as separate reference layers:

- determinant-expanded oracle code for tiny systems;
- old spin-grid / Vandermonde overlap prototype;
- current closed-shell exact kernel.

These are validation tools, not production formulas.

---

## 6. Validation Plan

### 6.1 Mathematical validation

For small active spaces, compare the new fixed-`M_s` formulas against exact
determinant-expanded matrix elements for:

- overlap;
- one-electron Hamiltonian part;
- total Hamiltonian;
- active-space overlap / one-electron / two-electron derivatives.

### 6.2 Incremental molecule set

Recommended order:

1. atomic or diatomic high-spin toy systems;
2. simple radicals with one blocked alpha electron;
3. two-unpaired high-spin systems;
4. mixed alpha/beta blocked cases at fixed `M_s`;
5. only then, spin-adapted combinations.

### 6.3 Performance validation

For each phase, record:

- pairwise kernel wall time;
- gradient wall time;
- scaling with active-space size;
- number of tensor contractions used by the open-shell path.

The core acceptance criterion is not equal constant factor relative to the
closed-shell code. The acceptance criterion is preserving `O(M^4)` leading
behavior with a controlled constant increase.

---

## 7. Main Risks

### Risk 1. Confusing fixed-`M_s` support with full spin adaptation

These are different problems. The first should be solved cleanly before the
second is attempted.

### Risk 2. Forcing open-shell formulas into the closed-shell `G/H/C/D` mold

That will likely fail. Open-shell theory should be expected to require a small
family of spin blocks and blocked-orbital intermediates, not a single master
spatial matrix.

### Risk 3. Reviving the old Vandermonde spin projection as the production path

It is valuable as an oracle, but it is not the right final architecture for
exact matrix elements and gradients.

### Risk 4. Interleaving selected-VB too early

Selected-VB is important, but it should be built on top of a correct generic
kernel. Otherwise debugging the kernel and debugging the structure selection
will become entangled.

---

## 8. Immediate Next Steps

The most productive near-term sequence is:

1. define the new open-shell state representation;
2. derive and validate fixed-`M_s` overlap + 1-RDM;
3. derive the full fixed-`M_s` two-electron `O(M^4)` kernel;
4. derive the corresponding adjoint;
5. refactor the cache / API around a generic spin-block formalism;
6. only then, add spin-adapted assembly.

This is the cleanest path to an implementation that is both publishable and
maintainable.
