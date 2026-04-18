#pragma once

#include <Eigen/Core>

#include <vector>

#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"
#include "vb/scf/cpp_active_space_gradient_result.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace xmvb::vb {

/**
 * @brief Reusable exact selected-state on-top pair-density contraction context.
 *
 * This object stores the selected-state coefficient matrix together with the
 * physical occupied-orbital blocks and determinant-pair first cofactors needed
 * to evaluate the exact physical on-top pair density
 *
 * `Pi(r) = <Psi| rho_alpha(r) rho_beta(r) |Psi> / <Psi|Psi>`
 *
 * directly from an AO value vector `chi(r)`. The context avoids materializing
 * a full physical four-index `2-RDM`; later grid code can instead call the
 * contraction routine point-by-point or block-by-block.
 */
struct SelectedStateExactPhysicalOnTopPairDensityContext {
  /**
   * @brief Selected-state index in the generalized eigenspectrum.
   */
  int state_index = 0;

  /**
   * @brief Number of AO basis functions.
   */
  int n_basis_functions = 0;

  /**
   * @brief Number of supported unique alpha determinants in the selected state.
   */
  int n_alpha_support_determinants = 0;

  /**
   * @brief Number of supported unique beta determinants in the selected state.
   */
  int n_beta_support_determinants = 0;

  /**
   * @brief Number of occupied physical alpha orbitals per support determinant.
   */
  int n_alpha_occupied_orbitals = 0;

  /**
   * @brief Number of occupied physical beta orbitals per support determinant.
   */
  int n_beta_occupied_orbitals = 0;

  /**
   * @brief Selected-state overlap normalization `\langle \Psi_n | \Psi_n \rangle`.
   */
  double selected_state_overlap_normalization = 0.0;

  /**
   * @brief Absolute difference between alpha-side and beta-side normalization contractions.
   */
  double alpha_beta_normalization_abs_error = 0.0;

  /**
   * @brief Column-major selected-state coefficient matrix on support-restricted unique-spin space.
   *
   * Dimensions: `n_alpha_support_determinants x n_beta_support_determinants`.
   */
  Eigen::MatrixXd local_coefficient_matrix;

  /**
   * @brief Occupied physical alpha-orbital blocks, one matrix per support determinant.
   *
   * Each block has dimensions `n_basis_functions x n_alpha_occupied_orbitals`.
   */
  std::vector<Eigen::MatrixXd> alpha_support_occupied_physical_orbitals;

  /**
   * @brief Occupied physical beta-orbital blocks, one matrix per support determinant.
   *
   * Each block has dimensions `n_basis_functions x n_beta_occupied_orbitals`.
   */
  std::vector<Eigen::MatrixXd> beta_support_occupied_physical_orbitals;

  /**
   * @brief Ordered alpha determinant-pair first cofactors.
   *
   * Storage index is `left * n_alpha_support_determinants + right`. Each block
   * has dimensions `n_alpha_occupied_orbitals x n_alpha_occupied_orbitals`.
   */
  std::vector<Eigen::MatrixXd> alpha_pair_first_order_cofactors;

  /**
   * @brief Ordered beta determinant-pair first cofactors.
   *
   * Storage index is `left * n_beta_support_determinants + right`. Each block
   * has dimensions `n_beta_occupied_orbitals x n_beta_occupied_orbitals`.
   */
  std::vector<Eigen::MatrixXd> beta_pair_first_order_cofactors;
};

/**
 * @brief Builds reusable exact physical on-top pair-density contraction contexts.
 *
 * The builder reuses the selected-state determinant coefficients from the
 * current C++ VBSCF eigensystem, but keeps the contraction in the physical
 * orbital frame rather than in the auxiliary/projected active-space frame.
 */
class SelectedStateExactPhysicalOnTopPairDensityBuilder {
public:
  /**
   * @brief Creates a builder with the default active-space gradient evaluator.
   */
  explicit SelectedStateExactPhysicalOnTopPairDensityBuilder(
      VBSCFAlgorithm algorithm = VBSCFAlgorithm::Original);

  /**
   * @brief Creates a builder with an explicit gradient evaluator.
   */
  explicit SelectedStateExactPhysicalOnTopPairDensityBuilder(
      CppActiveSpaceGradientEvaluator gradient_evaluator);

  /**
   * @brief Builds the exact physical on-top contraction context by running a unit-weight state-specific gradient.
   */
  SelectedStateExactPhysicalOnTopPairDensityContext build(
      const CppVbInput& input,
      int state_index,
      double nuclear_repulsion_energy = 0.0) const;

  /**
   * @brief Builds the exact physical on-top contraction context from an existing state-specific gradient result.
   *
   * The supplied gradient result must correspond to exactly one selected state
   * with normalized weight 1.
   */
  SelectedStateExactPhysicalOnTopPairDensityContext build_from_state_specific_gradient(
      const CppVbInput& input,
      const CppActiveSpaceGradientResult& gradient_result) const;

private:
  CppActiveSpaceGradientEvaluator gradient_evaluator_;
};

/**
 * @brief Evaluates the exact physical selected-state on-top pair density for one AO value vector.
 *
 * `ao_values[mu] = chi_mu(r)` corresponds to one real-space point or any other
 * AO-space probe vector used for diagnostics.
 */
double evaluate_selected_state_exact_physical_on_top_pair_density(
    const SelectedStateExactPhysicalOnTopPairDensityContext& context,
    const Eigen::Ref<const Eigen::VectorXd>& ao_values);

/**
 * @brief Evaluates the exact physical selected-state on-top pair density for a block of AO probe vectors.
 *
 * `ao_value_matrix` stores `n_probe_vectors` AO vectors in column-major order
 * with dimensions `n_basis_functions x n_probe_vectors`.
 */
Eigen::VectorXd evaluate_selected_state_exact_physical_on_top_pair_density_batch(
    const SelectedStateExactPhysicalOnTopPairDensityContext& context,
    const Eigen::Ref<const Eigen::MatrixXd>& ao_value_matrix);

}  // namespace xmvb::vb
