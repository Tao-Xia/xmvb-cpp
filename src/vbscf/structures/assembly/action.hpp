#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"
#include "vbscf/structures/expansion/types.hpp"

namespace xmvb::vb {

/**
 * @brief Result of applying the structure Hamiltonian and overlap to a vector block.
 *
 * Both matrices have shape `(n_structures, block_width)`. No dense
 * `n_structures x n_structures` matrix is formed.
 */
struct StructureActionResult {
  Eigen::MatrixXd hamiltonian;
  Eigen::MatrixXd overlap;
};

/**
 * @brief Exact diagonals used by a matrix-free generalized eigensolver.
 */
struct StructureDiagonal {
  Eigen::VectorXd hamiltonian;
  Eigen::VectorXd overlap;
};

/**
 * @brief Matrix-free structure-space Hamiltonian and overlap action.
 *
 * Let `B` be the sparse determinant-to-structure expansion. This class
 * evaluates
 *
 * `Y_H = B^T H_det B X` and `Y_S = B^T S_det B X`
 *
 * directly from the ordered same-spin determinant-pair cache. Its working
 * memory is linear in `n_determinants * block_width` and
 * `n_structures * block_width`; it never allocates dense structure matrices.
 * A block of vectors shares every determinant-pair scalar evaluation.
 */
class StructureAction {
public:
  StructureAction(
      const std::vector<std::vector<StructureExpansionTerm>>&
          determinant_to_structure_terms,
      int n_structures,
      const SameSpinPairCacheContext& same_spin_pair_cache,
      const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
      int n_active_orbitals);

  /**
   * @brief Applies both structure matrices to one or more vectors.
   *
   * @param vectors Dense `(n_structures, block_width)` input block.
   * @return Hamiltonian and overlap images with the same shape.
   */
  StructureActionResult apply(
      const Eigen::Ref<const Eigen::MatrixXd>& vectors) const;

  /**
   * @brief Evaluates the exact structure-space H/S diagonals without full matrices.
   */
  StructureDiagonal diagonal() const;

  int n_determinants() const noexcept;
  int n_structures() const noexcept;

private:
  struct DeterminantTerm {
    int determinant = 0;
    double coefficient = 0.0;
  };

  const SameSpinPairCacheContext* same_spin_pair_cache_ = nullptr;
  ActiveSpaceTwoElectronView two_electron_view_;
  const std::vector<std::vector<StructureExpansionTerm>>*
      determinant_to_structure_terms_ = nullptr;
  std::vector<std::vector<DeterminantTerm>> structure_to_determinant_terms_;
  int n_determinants_ = 0;
  int n_structures_ = 0;
  int n_packed_pairs_ = 0;
  int n_active_orbitals_ = 0;
};

}  // namespace xmvb::vb
