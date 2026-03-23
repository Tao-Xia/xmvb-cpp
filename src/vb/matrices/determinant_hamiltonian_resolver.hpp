#pragma once

#include <vector>

#include "vb/matrices/determinant_hamiltonian_result.hpp"
#include "vb/matrices/determinant_overlap_resolver.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace xmvb::vb {

/**
 * @brief Computes determinant-level Hamiltonian matrix elements from explicit inputs.
 *
 * This class is a new C++ implementation of the core numerical work performed
 * by the legacy `Hamhd0` routine for a single determinant pair. It assumes that
 * the caller has already assembled the determinant overlap submatrix.
 */
class DeterminantHamiltonianResolver {
public:
  /**
   * @brief Constructs a resolver with the default overlap resolver.
   */
  explicit DeterminantHamiltonianResolver(
      VbScfAlgorithm algorithm = VbScfAlgorithm::Original);

  /**
   * @brief Constructs a resolver using an overlap resolver instance.
   *
   * @param overlap_resolver Resolver used to compute determinant overlap
   *   determinant, nullity, and first-order cofactors.
   */
  explicit DeterminantHamiltonianResolver(
      DeterminantOverlapResolver overlap_resolver,
      VbScfAlgorithm algorithm = VbScfAlgorithm::Original);

  /**
   * @brief Evaluates determinant-level Hamiltonian and overlap quantities.
   *
   * @param occupied_orbitals_left Zero-based occupied orbital indices of the
   *   left determinant.
   * @param occupied_orbitals_right Zero-based occupied orbital indices of the
   *   right determinant.
   * @param determinant_overlap_submatrix Column-major overlap submatrix between
   *   occupied orbitals of the two determinants.
   * @param one_electron_matrix Column-major one-electron integral matrix over
   *   the full orbital basis.
   * @param n_orbitals Total number of orbitals represented in
   *   `one_electron_matrix`.
   * @param packed_two_electron_integrals Packed two-electron integral storage
   *   using the legacy `Ggo` indexing convention.
   * @return DeterminantHamiltonianResult Determinant overlap and Hamiltonian.
   */
  DeterminantHamiltonianResult resolve(
      const std::vector<int>& occupied_orbitals_left,
      const std::vector<int>& occupied_orbitals_right,
      const std::vector<double>& determinant_overlap_submatrix,
      const std::vector<double>& one_electron_matrix,
      int n_orbitals,
      const std::vector<double>& packed_two_electron_integrals) const;

  /**
   * @brief Evaluates determinant Hamiltonian using a precomputed overlap result.
   *
   * This overload lets callers reuse determinant/cofactor work across overlap
   * and Hamiltonian evaluation so the overlap matrix does not need to be
   * factorized twice for the same determinant pair.
   *
   * @param occupied_orbitals_left Zero-based occupied orbital indices of the
   *   left determinant.
   * @param occupied_orbitals_right Zero-based occupied orbital indices of the
   *   right determinant.
   * @param determinant_overlap_submatrix Column-major overlap submatrix between
   *   occupied orbitals of the two determinants.
   * @param overlap_result Precomputed determinant overlap quantities for
   *   `determinant_overlap_submatrix`.
   * @param one_electron_matrix Column-major one-electron integral matrix over
   *   the full orbital basis.
   * @param n_orbitals Total number of orbitals represented in
   *   `one_electron_matrix`.
   * @param packed_two_electron_integrals Packed two-electron integral storage
   *   using the legacy `Ggo` indexing convention.
   * @return DeterminantHamiltonianResult Determinant overlap and Hamiltonian.
   */
  DeterminantHamiltonianResult resolve(
      const std::vector<int>& occupied_orbitals_left,
      const std::vector<int>& occupied_orbitals_right,
      const std::vector<double>& determinant_overlap_submatrix,
      const DeterminantOverlapResult& overlap_result,
      const std::vector<double>& one_electron_matrix,
      int n_orbitals,
      const std::vector<double>& packed_two_electron_integrals) const;

private:
  DeterminantOverlapResolver overlap_resolver_;
  VbScfAlgorithm algorithm_ = VbScfAlgorithm::Original;
};

}  // namespace xmvb::vb
