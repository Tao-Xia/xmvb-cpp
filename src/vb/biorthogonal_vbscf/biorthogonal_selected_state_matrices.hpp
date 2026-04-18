#pragma once

#include <vector>

#include <Eigen/Core>

#include "vb/biorthogonal_vbscf/biorthogonal_orbital_integrals.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_exact_selected_structure.hpp"
#include "vb/matrices/same_spin_pair_cache.hpp"
#include "vb/scf/selected_state_determinant_matrices.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

/**
 * @brief Per-selected-state determinant coefficients for the exact biorthogonal backward.
 *
 * For selected state `n`, the three determinant families are:
 *
 * `r_I^(n)`:
 *   right determinant coefficients from `C_R = T_sel C`.
 *
 * `l_J^(n)`:
 *   dual-left determinant coefficients from `C_L = U_sel C`.
 *
 * `q_J^(n)`:
 *   overlap residual-left coefficients
 *   `q^(n) = h_bi T_sel c_n - E_n r^(n)`.
 *
 * Each family is also regrouped onto the touched unique alpha/beta support so
 * later matrix-form contractions can work directly on unique spin strings
 * without storing a full dense `(n_unique_alpha, n_unique_beta)` image per
 * selected state.
 */
struct BiorthogonalSelectedStateDeterminantCoefficients {
  int state_index = 0;
  double eigenvalue = 0.0;
  double normalized_state_weight = 0.0;

  std::vector<double> right_determinant_coefficients;
  std::vector<double> left_determinant_coefficients;
  std::vector<double> residual_determinant_coefficients;

  std::vector<int> alpha_support;
  std::vector<int> beta_support;
  Eigen::MatrixXd local_right_coefficient_matrix;
  Eigen::MatrixXd local_left_coefficient_matrix;
  Eigen::MatrixXd local_residual_coefficient_matrix;

  int nonzero_right_coefficient_count = 0;
  int nonzero_left_coefficient_count = 0;
  int nonzero_residual_coefficient_count = 0;
};

/**
 * @brief Exact biorthogonal selected-state bundle on the full determinant topology.
 *
 * This mirrors the nonorthogonal `SelectedStateDeterminantMatrices` helper, but
 * keeps the three coefficient families required by the exact biorthogonal
 * determinant-level backward:
 *
 * 1. `R^(n)` for Hamiltonian right coefficients,
 * 2. `L^(n)` for Hamiltonian left coefficients,
 * 3. `Q^(n)` for overlap left coefficients.
 */
struct BiorthogonalSelectedStateMatrices {
  int n_structures = 0;
  int n_determinants = 0;
  int n_unique_alpha = 0;
  int n_unique_beta = 0;

  std::vector<int> selected_state_indices;
  std::vector<double> normalized_state_weights;

  std::vector<int> determinant_to_unique_alpha_id;
  std::vector<int> determinant_to_unique_beta_id;

  std::vector<BiorthogonalSelectedStateDeterminantCoefficients> states;
};

/**
 * @brief Exact ordered and unordered determinant-pair weights from `R/L/Q`.
 *
 * The ordered tables store:
 *
 * `W_H(J,I) = sum_n w_n l_J^(n) r_I^(n)`,
 * `W_S(J,I) = sum_n w_n q_J^(n) r_I^(n)`.
 *
 * The unordered tables use canonical `left >= right` storage and combine the
 * two ordered orientations on off-diagonal pairs.
 */
struct BiorthogonalDeterminantPairWeightTablesFromCoefficients {
  int n_determinants = 0;
  std::vector<double> ordered_hamiltonian_weights;
  std::vector<double> ordered_overlap_weights;
  std::vector<double> unordered_combined_hamiltonian_weights;
  std::vector<double> unordered_combined_overlap_weights;
};

/**
 * @brief Normalizes non-negative state-average weights to unit sum.
 */
std::vector<double> normalize_biorthogonal_state_average_weights(
    const std::vector<double>& state_average_weights);

/**
 * @brief Builds the exact biorthogonal `R/L/Q` bundle from raw weights.
 */
BiorthogonalSelectedStateMatrices build_biorthogonal_selected_state_matrices(
    const BiorthogonalExactSelectedStructureEvaluationResult& evaluation_result,
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights);

/**
 * @brief Builds the exact biorthogonal `R/L/Q` bundle from normalized weights.
 *
 * The supplied weights must sum to 1 within tolerance.
 */
BiorthogonalSelectedStateMatrices
build_biorthogonal_selected_state_matrices_from_normalized_weights(
    const BiorthogonalExactSelectedStructureEvaluationResult& evaluation_result,
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& normalized_state_weights);

/**
 * @brief Builds the exact biorthogonal `R/L/Q` bundle from a matrix-only forward solve.
 *
 * This path is intended for accepted-point SCF gradients. It first solves the
 * selected-space generalized eigenproblem on the physical structure matrices,
 * then constructs only the requested state bundles directly:
 *
 * `r^(n) = T_sel c_n`,
 * `l^(n) = S_det T_sel c_n`,
 * `q^(n) = h_bi T_sel c_n - E_n r^(n)`.
 *
 * Crucially, it does not materialize the full selected-column determinant
 * actions `U_sel` and `Y_sel` when only a small number of states is needed.
 */
BiorthogonalSelectedStateMatrices
build_biorthogonal_selected_state_matrices_from_structure_problem(
    const xmvb::vb::FullDeterminantStructureData& full_structure_data,
    const std::vector<int>& selected_structure_indices,
    const Eigen::MatrixXd& structure_coefficient_matrix,
    const std::vector<double>& eigenvalues,
    const xmvb::vb::SameSpinPairCacheContext& same_spin_pair_cache,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view);

/**
 * @brief Builds exact determinant-pair adjoint weights from the `R/L/Q` bundle.
 */
BiorthogonalDeterminantPairWeightTablesFromCoefficients
build_biorthogonal_exact_determinant_pair_weight_tables_from_coefficients(
    const BiorthogonalSelectedStateMatrices& selected_state_matrices);

}  // namespace xmvb::vb::biorthogonal_vbscf
