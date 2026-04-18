# Exact Separator Notes

This directory now keeps only the documents that still match the current code
state or the next intended implementation direction.

## Current Exact Value Kernel

- [summary.md](./summary.md): current exact overlap / Hamiltonian algorithm,
  validation status, and current bottlenecks.
- [formulas.md](./formulas.md): exact overlap, one-electron, and two-electron
  formulas used by the separator implementation.
- [root_closure.md](./root_closure.md): root-closure formulas for the recursive
  Hamiltonian bundle.
- [status_2n_2m.md](./status_2n_2m.md): current status of the `2^n -> 2^m`
  transition and what is still missing structurally.

## Future Exact Message Redesign

- [boundary_repr.md](./boundary_repr.md): the exact boundary-only message
  representation needed for a true separator-width exponential.
- [boundary_impl.md](./boundary_impl.md): implementation-level design for the
  boundary-only message recurrence.
- [boundary_data.md](./boundary_data.md): concrete payload / sector data layout
  needed by the recursive driver.

## Optional Approximation Path

- [truncation.md](./truncation.md): low-rank / truncated one-leaf design for
  star-dominated systems when chemical accuracy is acceptable.

## Gradient Design

- [gradient.md](./gradient.md): exact-separator active-space gradient design,
  formulas, reverse-mode decomposition, and staged implementation plan.

## Gradient Next Step

The gradient work should start from the current exact value kernel rather than
from the deleted historical phase documents.

Recommended order:

1. Differentiate the current overlap / one-electron boundary messages on top of
   `component_tree.cpp`.
2. Extend the same typed message algebra to the Hamiltonian channels already
   used by the exact value kernel.
3. Add root-closure derivatives using the formulas in [root_closure.md](./root_closure.md).
4. Reuse the same message basis for both value and gradient code so the future
   exact `2^n -> 2^m` redesign does not need a second gradient rewrite.
