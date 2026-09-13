#pragma once

#include <Eigen/Core>

#include <vector>

#include "vbscf/determinants/contracts/types.hpp"
#include "vbscf/determinants/algebra/overlap.hpp"
#include "vbscf/integrals/active/two_electron/construction/result.hpp"

namespace xmvb::vb {

/**
 * @brief Computes determinant-level Hamiltonian matrix elements from explicit inputs.
 *
 * The caller supplies an assembled determinant overlap submatrix; this class
 * evaluates the corresponding one- and two-electron matrix elements.
 */
class DeterminantHamiltonianResolver {
public:
  /**
   * @brief Constructs a resolver with the default overlap resolver.
   */
  DeterminantHamiltonianResolver();

  /**
   * @brief Constructs a resolver using an overlap resolver instance.
   *
   * @param overlap_resolver Resolver used to compute determinant overlap
   *   determinant, nullity, and first-order cofactors.
   */
  explicit DeterminantHamiltonianResolver(
      DeterminantOverlapResolver overlap_resolver);

  /**
   * @brief Evaluates determinant-level Hamiltonian and overlap quantities.
   *
   * @param occ_L Zero-based occupied orbital indices of the
   *   left determinant.
   * @param occ_R Zero-based occupied orbital indices of the
   *   right determinant.
   * @param det_ovlp_mat Column-major overlap submatrix between
   *   occupied orbitals of the two determinants.
   * @param h1e_act Column-major one-electron integral matrix over
   *   the full orbital basis.
   * @param n_orbitals Total number of orbitals represented in
   *   `h1e_act`.
   * @param eri_act Packed two-electron integral storage.
   * @return DeterminantHamiltonianResult Determinant overlap and Hamiltonian.
   */
  DeterminantHamiltonianResult resolve(
      const std::vector<int>& occ_L,
      const std::vector<int>& occ_R,
      const Eigen::Ref<const Eigen::MatrixXd>& det_ovlp_mat,
      const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
      int n_orbitals,
      const std::vector<double>& eri_act) const;

  /**
   * @brief Evaluates determinant Hamiltonian using either packed or RI active ERIs.
   */
  DeterminantHamiltonianResult resolve(
      const std::vector<int>& occ_L,
      const std::vector<int>& occ_R,
      const Eigen::Ref<const Eigen::MatrixXd>& det_ovlp_mat,
      const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
      int n_orbitals,
      const ActiveSpaceTwoElectronResult& active_space_two_electron_result) const;

  /**
   * @brief Evaluates determinant Hamiltonian using a precomputed overlap result.
   *
   * This overload lets callers reuse determinant/cofactor work across overlap
   * and Hamiltonian evaluation so the overlap matrix does not need to be
   * factorized twice for the same determinant pair.
   *
   * @param occ_L Zero-based occupied orbital indices of the
   *   left determinant.
   * @param occ_R Zero-based occupied orbital indices of the
   *   right determinant.
   * @param det_ovlp_result Precomputed determinant overlap quantities for
   *   the occupied-overlap submatrix of this determinant pair.
   * @param h1e_act Column-major one-electron integral matrix over
   *   the full orbital basis.
   * @param n_orbitals Total number of orbitals represented in
   *   `h1e_act`.
   * @param eri_act Packed two-electron integral storage
   *   using the packed two-electron indexing convention.
   * @return DeterminantHamiltonianResult Determinant overlap and Hamiltonian.
   */
  DeterminantHamiltonianResult resolve(
      const std::vector<int>& occ_L,
      const std::vector<int>& occ_R,
      const DeterminantOverlapResult& det_ovlp_result,
      const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
      int n_orbitals,
      const std::vector<double>& eri_act) const;

  /**
   * @brief Evaluates determinant Hamiltonian using either packed or RI active ERIs.
   */
  DeterminantHamiltonianResult resolve(
      const std::vector<int>& occ_L,
      const std::vector<int>& occ_R,
      const DeterminantOverlapResult& det_ovlp_result,
      const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
      int n_orbitals,
      const ActiveSpaceTwoElectronResult& active_space_two_electron_result) const;

private:
  DeterminantOverlapResolver overlap_resolver_;
};

}  // namespace xmvb::vb
