# Davidson Structure-Operator Design

## Separation of concerns

The structure eigensolver and the representation of the Hamiltonian and
overlap operators are independent choices. Davidson requires exact block
actions, but those actions may be supplied by either:

1. materialized structure-space matrices followed by dense block products; or
2. a matrix-free support-local contraction.

Selecting Davidson must not force the construction of a larger intermediate
representation than the structure matrices themselves.

## Exact representation selection

Let `n_str` be the number of VB structures and let `n_u_alpha` and
`n_u_beta` be the numbers of unique alpha and beta strings. A global
Cartesian-factor action must retain at least the same-spin Hamiltonian and
overlap matrices,

$$
N_{\mathrm{factor,min}}
=2\left(n_{u,\alpha}^{2}+n_{u,\beta}^{2}\right),
$$

whereas materialized structure-space Hamiltonian and overlap matrices require

$$
N_{\mathrm{matrix}}=2n_{\mathrm{str}}^{2}.
$$

When `N_matrix <= N_factor,min`, the materialized representation is selected.
This comparison is an exact storage-dominance test, not a molecule-specific
threshold. Opposite-spin channel storage can only make the global factorized
representation larger.

Davidson still computes only the requested roots in this branch. The
materialized matrices provide its block actions and do not imply a dense
eigensolve.

## Matrix-free large-structure branch

The intended large-scale operator is support-local. Each structure owns a
`StructureCoefficientBlock` containing its active unique-string support and
local coefficient matrix. For a block of Davidson vectors, the operator must:

1. traverse structure-pair tiles;
2. obtain only the unique-string-pair kernels touched by the two local
   supports;
3. contract the local Hamiltonian and overlap contribution with every
   right-hand side in the block; and
4. immediately accumulate the result in structure space.

The implementation must not form a global
`n_u_alpha * n_u_beta * block_width` Cartesian image and must not retain one
global `n_u^2` matrix per active-pair channel. The existing dense H/S builder
already implements the validated support-local pair kernels and is the
reference algebra for this streamed block action.

## Accepted-point lifetime

The structure action is derivative-only state. A core-only TNHVP step does not
construct it. Dense and matrix-backed forward solves leave the accepted action
empty; the exact HVP operator creates the matrix-free action only when an outer
structure response is actually admitted. A forward matrix-free Davidson solve
may transfer its already-built action into the accepted-point context.

## Accuracy contract

Davidson convergence is governed by both the outer energy accuracy and the
eigen-equation residual needed by the orbital gradient. The solver must use a
Ritz-energy error certificate together with the requested residual bound. A
linear conversion of absolute energy tolerance to relative residual by
dividing by the eigenvalue magnitude is not a valid near-solution error model
and can over-solve the structure problem by many orders of magnitude.

## Required validation

- Compare dense and Davidson selected energies, eigen-equation residuals,
  orbital gradients, structure coefficients, and structure weights.
- Test cold and recycled solves separately.
- Verify block action against column-wise action and explicit H/S products.
- Record operator construction time, action count, action block width,
  cumulative action time, and peak resident memory.
- Cover sparse HAO, full-AO OEO, open-shell, large unique-string, and large
  structure-space cases.
