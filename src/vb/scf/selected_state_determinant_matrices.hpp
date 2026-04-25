#pragma once

#include <vector>

#include <Eigen/Core>

#include "vb/matrices/same_spin_pair_cache.hpp"
#include "vb/matrices/structure_types.hpp"

namespace xmvb::vb {

/**
 * @brief Per-selected-state determinant coefficients and alpha/beta matrix view.
 *
 * For selected state `n`, `determinant_coefficients[d]` stores:
 *
 * `c_d^(n) = sum_s T_{d,s} * u_s^(n)`
 *
 * where `T_{d,s}` is the determinant-to-structure expansion coefficient and
 * `u_s^(n)` is the structure eigenvector coefficient.
 *
 * `coefficient_matrix` is the same state reshaped to unique spin-string space
 * with dimensions `(n_unique_alpha, n_unique_beta)`.
 *
 * In close-shell expansions `alpha_id == beta_id` for every determinant, so
 * `coefficient_matrix` is diagonal. In that case `close_shell_diagonal` is set
 * and `diagonal_coefficients` / `local_diagonal_coefficients` cache the global
 * and trimmed diagonal entries directly for hot-path contractions.
 *
 * `local_coefficient_matrix` stores the same coefficients restricted to the
 * trimmed support block `alpha_support x beta_support`. This exact support-aware
 * view lets backward contractions skip global zero rows/columns for each
 * selected state.
 */
struct SelectedStateDeterminantCoefficients {
  int state_index = 0;
  double normalized_state_weight = 0.0;
  std::vector<double> determinant_coefficients;
  Eigen::MatrixXd coefficient_matrix;
  bool close_shell_diagonal = false;
  std::vector<double> diagonal_coefficients;
  std::vector<int> alpha_support;
  std::vector<int> beta_support;
  Eigen::MatrixXd local_coefficient_matrix;
  std::vector<double> local_diagonal_coefficients;
  int nonzero_coefficient_count = 0;
};

/**
 * @brief Selected-state determinant coefficient bundle for matrix-form rewrites.
 *
 * This object exposes both:
 * 1. per-state determinant vectors `c_d^(n)`,
 * 2. per-state unique-spin matrices `C^(n)`.
 *
 * The two index maps allow callers to convert determinant index `d` into
 * unique `(alpha_id, beta_id)` coordinates without touching cache internals.
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
 * @brief Exact determinant-pair weights reconstructed from `c_d^(n)`.
 *
 * `ordered_*` are dense row-major `n_determinants x n_determinants` tables:
 *
 * `W_H(dL,dR) = sum_n w_n c_dL^(n) c_dR^(n)`
 * `W_S(dL,dR) = -sum_n w_n E_n c_dL^(n) c_dR^(n)`
 *
 * `unordered_combined_*` uses canonical storage (`left >= right`) and combines
 * off-diagonal ordered orientations:
 *
 * `W_unordered(left,right) = W(left,right) + W(right,left)` for `left != right`.
 *
 * This is the quantity that should match the current unordered pair logic after
 * adding its direct and swapped adjoints.
 */
struct DeterminantPairWeightTablesFromCoefficients {
  int n_determinants = 0;
  std::vector<double> ordered_hamiltonian_weights;
  std::vector<double> ordered_overlap_weights;
  std::vector<double> unordered_combined_hamiltonian_weights;
  std::vector<double> unordered_combined_overlap_weights;
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
 * @brief Returns whether support-sparse selected-state contractions should run.
 *
 * The decision is automatic by default and compares the full unique-spin
 * contraction footprint against the trimmed per-state support blocks. Set
 * `XMVB_CPP_SELECTED_STATE_SUPPORT_SPARSE=on|off|auto` to override.
 */
bool should_use_support_sparse_selected_state_contractions(
    const SelectedStateDeterminantMatrices& selected_state_matrices);

/**
 * @brief Extracts energies for selected states from the full eigenvalue vector.
 */
std::vector<double> gather_selected_state_energies(
    const std::vector<double>& eigenvalues,
    const std::vector<int>& selected_state_indices);

/**
 * @brief Builds exact determinant-pair Hamiltonian/overlap weights from `c_d^(n)`.
 *
 * `selected_state_energies[k]` must correspond to
 * `selected_state_indices[k]` in the coefficient bundle.
 */
DeterminantPairWeightTablesFromCoefficients
build_exact_determinant_pair_weight_tables_from_coefficients(
    const SelectedStateDeterminantMatrices& selected_state_matrices,
    const std::vector<double>& selected_state_energies);

/**
 * @brief Builds the directional determinant-pair weights from `\delta c_d^(n)`.
 *
 * The returned ordered tables store the exact first-order response
 *
 * `\delta W_H(dL,dR) = sum_n w_n [\delta c_dL^(n) c_dR^(n) + c_dL^(n) \delta c_dR^(n)]`
 *
 * `\delta W_S(dL,dR) = -sum_n w_n [\delta E_n c_dL^(n) c_dR^(n) +`
 * `                                 E_n (\delta c_dL^(n) c_dR^(n) +`
 * `                                      c_dL^(n) \delta c_dR^(n))]`
 *
 * with the unordered buffers combining the two ordered orientations on the
 * canonical `left >= right` storage.
 */
DeterminantPairWeightTablesFromCoefficients
build_directional_determinant_pair_weight_tables_from_coefficients(
    const SelectedStateDeterminantMatrices& selected_state_matrices,
    const SelectedStateDeterminantMatrices& directional_selected_state_matrices,
    const std::vector<double>& selected_state_energies,
    const std::vector<double>& directional_selected_state_energies);

/**
 * @brief Convenience overload that pulls selected energies from all eigenvalues.
 */
DeterminantPairWeightTablesFromCoefficients
build_exact_determinant_pair_weight_tables_from_eigenvalues(
    const SelectedStateDeterminantMatrices& selected_state_matrices,
    const std::vector<double>& eigenvalues);

}  // namespace xmvb::vb
