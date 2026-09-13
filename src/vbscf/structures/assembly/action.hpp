#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

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

struct StructureActionStorage {
  /** Numerical payload retained by the spin-factorized action. */
  std::size_t factor_bytes = 0;
  /** Opposite-spin channels whose raw side is stored densely. */
  int dense_channels = 0;
  /** Opposite-spin channels whose raw side is stored as exact nonzeros. */
  int sparse_channels = 0;
  /** Exact nonzeros before a sparse channel is optionally densified. */
  std::size_t channel_nonzeros = 0;
  /** Dense values required to store the same active channel matrices. */
  std::size_t channel_dense_values = 0;
};

/**
 * @brief Matrix-free structure-space Hamiltonian and overlap action.
 *
 * Let `B` be the sparse determinant-to-structure expansion and let `X` be a
 * determinant vector reshaped over the unique alpha/beta spin spaces. The
 * determinant Hamiltonian action is factorized as
 *
 * `H_alpha X S_beta^T + S_alpha X H_beta^T`
 *
 * plus one factored alpha/beta product per active-orbital pair channel. This
 * replaces the quadratic full-determinant pair traversal with dense spin-space
 * contractions. The sparse `B` expansion is applied before and after these
 * contractions, and dense structure-space matrices are never allocated.
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

  /** @brief Reports the retained factor storage and channel representations. */
  StructureActionStorage storage() const noexcept;

private:
  struct DeterminantTerm {
    int determinant = 0;
    double coefficient = 0.0;
  };

  struct OppositeSpinChannel {
    Eigen::MatrixXd projected;
    Eigen::MatrixXd dense;
    std::vector<Eigen::Triplet<double>> sparse;
  };

  const SameSpinPairCacheContext* same_spin_pair_cache_ = nullptr;
  ActiveSpaceTwoElectronView two_electron_view_;
  const std::vector<std::vector<StructureExpansionTerm>>*
      determinant_to_structure_terms_ = nullptr;
  std::vector<std::vector<DeterminantTerm>> structure_to_determinant_terms_;
  Eigen::MatrixXd alpha_overlap_;
  Eigen::MatrixXd alpha_hamiltonian_;
  Eigen::MatrixXd beta_overlap_;
  Eigen::MatrixXd beta_hamiltonian_;
  std::vector<OppositeSpinChannel> opposite_spin_channels_;
  std::vector<int> determinant_to_spin_product_;
  bool alpha_projection_is_dense_ = false;
  int n_determinants_ = 0;
  int n_structures_ = 0;
  int n_unique_alpha_ = 0;
  int n_unique_beta_ = 0;
  int n_packed_pairs_ = 0;
  int n_active_orbitals_ = 0;
  std::size_t channel_nonzeros_ = 0;
  std::size_t channel_dense_values_ = 0;
};

}  // namespace xmvb::vb
