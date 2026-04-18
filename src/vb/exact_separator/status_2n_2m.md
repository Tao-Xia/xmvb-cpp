# Exact Separator Current `2^n -> 2^m` Status

## 1. Purpose

This note records the code status as of the current `component_tree`
implementation, without preserving earlier intermediate conclusions that are
no longer true.

Here:

- `n` means the internal combinatorial size of a subtree or component pair;
- `m` means the exposed separator / boundary width.

The important question is:

> in the production path that is actually used today, which parts already live
> on the boundary-width side of the redesign, and which parts still do not?

---

## 2. Short Answer

The current codebase is now in a more useful mixed state.

1. the production `one_leaf_star` Hamiltonian path is exact and uses the
   dedicated one-leaf boundary-message implementation;
2. the generic rooted-tree Hamiltonian path now runs through the typed
   recursive boundary bundle
   `HamiltonianBoundaryPayload`
   instead of the older root-level
   `JointDeletionPayload`
   closure;
3. current validation shows that this generic recursive path is exact on the
   existing synthetic chain / branched subtree checks and on the currently
   exercised molecular tree-pair validations;
4. however, the implementation is still not the final performance-oriented
   sector-and-family data layout described in the design notes, and it is
   still slower than determinant-pair contraction on the current star-heavy
   benchmarks.

---

## 3. What Is Exact In Production Today

For rooted trees with exactly one root component and one leaf component,
`evaluate_rooted_component_tree_hamiltonian_collapsed(...)` now dispatches to
the validated one-leaf separator kernels:

- one-electron:
  `evaluate_component_ordered_open_state_star_pair_one_electron_collapsed(...)`
- two-electron:
  `evaluate_component_ordered_open_state_star_pair_two_electron_one_leaf_collapsed(...)`

This path is:

1. exact to numerical precision;
2. separator-based rather than determinant-pair based;
3. the active production path for the recent `C6H6` one-leaf-star benchmarks.

So for the currently dominant star hot path, the code is no longer falling back
to the old determinant-pair Hamiltonian builder.

For rooted trees with more than one node outside the specialized `one_leaf_star`
case, the generic Hamiltonian path now uses the typed recursive Hamiltonian
bundle in `component_tree.cpp`.

Current exactness evidence for that path is:

1. the synthetic `chain_rooted_subtree` check is exact to numerical noise;
2. the synthetic `branched_rooted_subtree` check is exact to numerical noise;
3. the current `validate_rooted_component_tree_exact_hamiltonian` molecular
   validations remain exact on the exercised closed-shell tree pairs.

---

## 4. What Still Remains Open

The main remaining problem is no longer "generic exactness is missing".
The main remaining problem is that the exact recursive representation is still
stored in a payload layout that is mathematically complete but not yet the
cleanest or fastest final layout.

The code still contains older experimental machinery for:

- generic joint deleted-minor subtree messages;
- direct deleted-minor-map oracles and debug validators;
- interface-block reorder transforms.

Those routines are still useful as derivation / oracle infrastructure, but they
are no longer the intended long-term hot path.

The remaining structural work is now:

1. reorganize the exact recursive bundle into the explicit boundary-sector /
   family layout from the design documents;
2. reduce the polynomial payload cost and outer mask combinatorics so the exact
   separator actually beats determinant-pair contraction;
3. remove obsolete debug-only generic closure code once the typed path has
   enough validation coverage.

---

## 5. Current Meaning Of `2^n -> 2^m`

The intended exact separator complexity target remains

$$
T \sim \mathrm{poly}(n)\, 2^{O(m)}.
$$

The current production situation should now be interpreted as follows:

1. the one-leaf-star path already uses boundary-message combinatorics rather
   than determinant-pair Cartesian products;
2. the generic recursive Hamiltonian path is now exact at the representation
   level on the currently validated cases;
3. the remaining gap to the practical `2^n -> 2^m` goal is therefore mainly
   a **performance / representation-layout** gap rather than a missing exact
   recursive closure.

---

## 6. Next Technical Step

The next structural step is **not** to revisit determinant-pair fallback, and
it is no longer to "make the generic recursion exact" in the first place.

The next structural step is to keep the now-exact generic recursive bundle,
but change its representation so that the computation is cheaper:

1. move from generic deleted-label payload storage toward explicit
   boundary-sector / family message blocks;
2. reduce outer mask / term combinatorics on top of that exact recursive
   representation;
3. only after that, revisit compression / truncation options for chemical
   accuracy rather than exact arithmetic.
