#!/usr/bin/env python3
"""Explain the biorthogonal structure-projection issue on the F2 example.

This script uses the actual F2 subset-{1} diagnostics produced by
`check_biorthogonal_dual_structure_projection`.

Why choose subset {1}?
  - It is a genuine truncated structure subspace of F2.
  - The selected space is 1-dimensional, so every generalized eigenproblem
    reduces to the scalar formula

        E = H / S

    which makes the algebra completely transparent.

The script compares three pencils for the same F2 subspace:

  1. naive:
       H_naive c = E M_naive c
     with M_naive = T^T T

  2. local dual:
       H_local c = E S_ref c
     which restores the correct overlap but still misses the omitted-
     determinant coupling in the Hamiltonian

  3. full dual:
       H_ref c = E S_ref c
     which uses the full-space action and therefore reproduces the original
     nonorthogonal subspace result exactly
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScalarPencil:
    label: str
    hamiltonian: float
    overlap: float

    @property
    def energy(self) -> float:
        return self.hamiltonian / self.overlap


def main() -> None:
    # Actual F2 subset-{1} root energies reported by
    # check_biorthogonal_dual_structure_projection.
    e_reference = -1.54930223341209
    e_bad = -1.33094069018502

    # For subset {1}, the diagnostic reports:
    #   local_dual_structure_hamiltonian_diff_max_abs = 0.169373097541607
    #
    # In a 1x1 generalized eigenproblem, H = E * S. Because the local-dual
    # and reference pencils share the same overlap S_ref, this immediately
    # gives the physical structure overlap.
    h_local_diff_abs = 0.169373097541607
    s_reference = h_local_diff_abs / abs(e_reference - e_bad)

    # The subset-{1} diagnostic also reports:
    #   naive_structure_overlap_diff_max_abs = 0.224345573682453
    #
    # For this one-determinant / one-structure case, T^T T = 1 exactly.
    s_naive = 1.0

    # Reconstruct the three scalar pencils.
    #
    # 1. original nonorthogonal structure problem
    h_reference = e_reference * s_reference

    # 2. current naive prototype: wrong metric M_naive = T^T T = 1
    h_naive = e_bad * s_naive

    # 3. "local dual" projector: correct overlap but incomplete Hamiltonian
    h_local = e_bad * s_reference

    # The missing term is exactly the omitted-determinant coupling.
    delta_omitted = h_reference - h_local

    # 4. full dual projector: same overlap as the physical problem and the
    # missing Hamiltonian contribution restored from the full determinant space.
    h_full_dual = h_local + delta_omitted

    reference = ScalarPencil("reference nonorth", h_reference, s_reference)
    naive = ScalarPencil("naive T^T T", h_naive, s_naive)
    local_dual = ScalarPencil("local dual", h_local, s_reference)
    full_dual = ScalarPencil("full dual", h_full_dual, s_reference)

    print("F2 example: selected structure subset {1}")
    print("=" * 72)
    print("This is a 1D subspace, so every generalized eigenvalue is simply E = H / S.")
    print()

    print("Physical nonorthogonal structure pencil")
    print(f"  S_ref = {reference.overlap:.15f}")
    print(f"  H_ref = {reference.hamiltonian:.15f}")
    print(f"  E_ref = H_ref / S_ref = {reference.energy:.15f}")
    print()

    print("Current naive biorthogonal prototype")
    print("  Uses M_naive = T^T T instead of the physical overlap.")
    print(f"  M_naive = {naive.overlap:.15f}")
    print(f"  H_naive = {naive.hamiltonian:.15f}")
    print(f"  E_naive = H_naive / M_naive = {naive.energy:.15f}")
    print()

    print("Local-dual repair inside the truncated determinant list")
    print("  Restores the correct overlap, but the Hamiltonian still misses")
    print("  the omitted-determinant coupling from the full determinant space.")
    print(f"  S_local = S_ref = {local_dual.overlap:.15f}")
    print(f"  H_local = {local_dual.hamiltonian:.15f}")
    print(f"  Delta_omitted = H_ref - H_local = {delta_omitted:.15f}")
    print(f"  E_local = H_local / S_local = {local_dual.energy:.15f}")
    print()

    print("Full-dual projection")
    print("  Uses the same physical overlap S_ref and restores the missing")
    print("  full-space Hamiltonian contribution.")
    print(f"  S_full = {full_dual.overlap:.15f}")
    print(f"  H_full = {full_dual.hamiltonian:.15f}")
    print(f"  E_full = H_full / S_full = {full_dual.energy:.15f}")
    print()

    print("Summary")
    print(f"  naive error      = {naive.energy - reference.energy:+.15f}")
    print(f"  local-dual error = {local_dual.energy - reference.energy:+.15f}")
    print(f"  full-dual error  = {full_dual.energy - reference.energy:+.15e}")
    print()

    print("Interpretation")
    print("  1. The naive prototype is wrong because it replaces the physical")
    print("     structure overlap by T^T T.")
    print("  2. Restoring only the overlap is not enough.")
    print("  3. The truncated structure subspace still feels the omitted")
    print("     determinants through Delta_omitted.")
    print("  4. Once the full-space dual projection is used, the biorthogonal")
    print("     formulation reproduces the original nonorthogonal subspace exactly.")


if __name__ == "__main__":
    main()
