#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/structures/expansion/types.hpp"

namespace xmvb::vb {

/**
 * @brief Per-selected-state coefficients in unique alpha/beta string space.
 *
 * `coefficient_matrix(alpha,beta)` stores the structure eigenvector expanded
 * directly into unique spin-string coordinates. Determinant-order copies are
 * deliberately not retained: all matrix elements and responses contract in
 * unique-string or structure space.
 *
 * In close-shell expansions `alpha_id == beta_id` for every determinant, so
 * `coefficient_matrix` is diagonal. In that case `close_shell_diagonal` is set
 * and `diagonal_coefficients` / `local_diagonal_coefficients` cache the global
 * and trimmed diagonal entries directly for hot-path contractions.
 *
 * The row-major sparse local matrices store the same coefficients on the
 * trimmed `alpha_support x beta_support` block.  Both orientations are kept so
 * alpha- and beta-major contractions traverse determinant connections without
 * materializing a Cartesian support block.
 */
struct SelectedStateDeterminantCoefficients {
  int state_index = 0;
  double normalized_state_weight = 0.0;
  Eigen::MatrixXd coefficient_matrix;
  bool close_shell_diagonal = false;
  std::vector<double> diagonal_coefficients;
  std::vector<int> alpha_support;
  std::vector<int> beta_support;
  /** @brief Row-major sparse local coefficient matrix for string contractions. */
  Eigen::SparseMatrix<double, Eigen::RowMajor, int>
      local_sparse_coefficient_matrix;
  /** @brief Row-major sparse transpose for beta-oriented contractions. */
  Eigen::SparseMatrix<double, Eigen::RowMajor, int>
      local_sparse_coefficient_transpose;
  std::vector<double> local_diagonal_coefficients;
  int nonzero_coefficient_count = 0;
};

/**
 * @brief Selected-state determinant coefficient bundle for matrix-form rewrites.
 *
 * The two index maps are used only while expanding structure coefficients into
 * unique `(alpha_id, beta_id)` coordinates.
 */
struct SelectedStateDeterminantMatrices {
  int n_structures = 0;
  int n_determinants = 0;
  int n_unique_alpha = 0;
  int n_unique_beta = 0;

  std::vector<int> selected_state_indices;
  std::vector<double> normalized_state_weights;

  std::vector<int> determinant_to_unique_alpha_id;
  std::vector<int> determinant_to_unique_beta_id;

  std::vector<SelectedStateDeterminantCoefficients> states;
};

/**
 * @brief Normalizes non-negative state-average weights to unit sum.
 */
std::vector<double> normalize_state_average_weights(
    const std::vector<double>& state_average_weights);

/**
 * @brief Builds `c_d^(n)` and `C^(n)` from raw (possibly unnormalized) weights.
 */
SelectedStateDeterminantMatrices build_selected_state_determinant_matrices(
    const FullDeterminantStructureData& full_determinant_data,
    const std::vector<double>& eigenvector_matrix,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    const SameSpinPairCacheContext& same_spin_pair_cache);

/**
 * @brief Builds `c_d^(n)` and `C^(n)` from already normalized weights.
 *
 * This overload requires the provided weights to sum to 1 within tolerance.
 */
SelectedStateDeterminantMatrices
build_selected_state_determinant_matrices_from_normalized_weights(
    const FullDeterminantStructureData& full_determinant_data,
    const std::vector<double>& eigenvector_matrix,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& normalized_state_weights,
    const SameSpinPairCacheContext& same_spin_pair_cache);

/**
 * @brief Builds `c_d^(n)` and `C^(n)` from explicitly provided selected-state columns.
 *
 * `selected_state_columns` must have shape
 * `(full_determinant_data.n_structures, selected_state_indices.size())`, with
 * column `k` already equal to the structure-basis coefficients for
 * `selected_state_indices[k]`.
 */
SelectedStateDeterminantMatrices
build_selected_state_determinant_matrices_from_selected_columns(
    const FullDeterminantStructureData& full_determinant_data,
    const Eigen::MatrixXd& selected_state_columns,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& normalized_state_weights,
    const SameSpinPairCacheContext& same_spin_pair_cache);

/**
 * @brief Returns whether sparse selected-state contractions should run.
 *
 * A tiled sweep is considered only when at least one selected-state support is
 * actually trimmed, then dense unique-string work is compared with the
 * determinant-connectivity traversal count. No molecule-specific override is
 * used.
 */
bool should_use_sparse_selected_state_contractions(
    const SelectedStateDeterminantMatrices& selected_state_matrices);

/**
 * @brief Estimates the selected-state unique-string contraction work.
 *
 * The returned count follows the same dense-versus-sparse decision used by
 * the production kernels. It is an operation-count proxy, not a wall-time
 * fit, and therefore remains independent of molecule names and hardware.
 */
double estimate_selected_state_contraction_work(
    const SelectedStateDeterminantMatrices& selected_state_matrices);

/**
 * @brief Extracts energies for selected states from the full eigenvalue vector.
 */
std::vector<double> gather_selected_state_energies(
    const std::vector<double>& eigenvalues,
    const std::vector<int>& selected_state_indices);

}  // namespace xmvb::vb
