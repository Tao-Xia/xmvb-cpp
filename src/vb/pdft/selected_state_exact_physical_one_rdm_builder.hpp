#pragma once

#include <Eigen/Core>

#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"
#include "vb/scf/cpp_active_space_gradient_result.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace xmvb::vb {

/**
 * @brief Exact selected-state AO one-particle density in the physical orbital frame.
 *
 * Unlike the current matrix-form `1-RDM` builder, this result is constructed
 * directly from the selected-state determinant coefficients together with the
 * normalized physical orbitals stored before the inactive-space projection
 * step. It therefore targets the real-space density needed by a future
 * `VB-PDFT` implementation rather than the auxiliary/projected `HHO`
 * convention used by the present matrix-form VBSCF path.
 */
struct SelectedStateExactPhysicalOneRdmResult {
  /**
   * @brief Selected-state index in the generalized eigenspectrum.
   */
  int state_index = 0;

  /**
   * @brief Number of AO basis functions in the physical density matrix.
   */
  int n_basis_functions = 0;

  /**
   * @brief Column-major symmetrized total AO density matrix in the physical frame.
   *
   * Dimensions: `n_basis_functions x n_basis_functions`.
   */
  Eigen::MatrixXd physical_ao_total_density_matrix;

  /**
   * @brief Maximum antisymmetric component before final symmetrization.
   */
  double ao_density_asymmetry_max_abs = 0.0;

  /**
   * @brief Selected-state normalization `\langle \Psi_n | \Psi_n \rangle`.
   *
   * The builder divides the raw transition-density sum by this value so the
   * returned AO density is normalized even if the determinant-coefficient
   * reconstruction carries small numerical drift.
   */
  double selected_state_overlap_normalization = 0.0;

  /**
   * @brief Absolute difference between alpha-side and beta-side normalization contractions.
   */
  double alpha_beta_normalization_abs_error = 0.0;

  /**
   * @brief Metric trace `Tr(P_AO * S_AO)` of the normalized physical AO density.
   */
  double ao_total_metric_trace = 0.0;
};

/**
 * @brief Builds exact selected-state physical AO densities from current C++ VBSCF objects.
 *
 * The builder reuses the validated selected-state coefficients from the current
 * C++ VBSCF eigensystem, but reconstructs the density in the physical
 * orbital frame rather than through the auxiliary/projected active-space
 * matrices.
 */
class SelectedStateExactPhysicalOneRdmBuilder {
public:
  /**
   * @brief Creates a builder with the default active-space gradient evaluator.
   */
  explicit SelectedStateExactPhysicalOneRdmBuilder(
      VBSCFAlgorithm algorithm = VBSCFAlgorithm::Original);

  /**
   * @brief Creates a builder with an explicit gradient evaluator.
   */
  explicit SelectedStateExactPhysicalOneRdmBuilder(
      CppActiveSpaceGradientEvaluator gradient_evaluator);

  /**
   * @brief Builds the exact physical selected-state AO density by running a unit-weight state-specific gradient.
   */
  SelectedStateExactPhysicalOneRdmResult build(
      const CppVbInput& input,
      int state_index,
      double nuclear_repulsion_energy = 0.0) const;

  /**
   * @brief Builds the exact physical selected-state AO density from an existing state-specific gradient result.
   *
   * The supplied gradient result must correspond to exactly one selected state
   * with normalized weight 1.
   */
  SelectedStateExactPhysicalOneRdmResult build_from_state_specific_gradient(
      const CppVbInput& input,
      const CppActiveSpaceGradientResult& gradient_result) const;

private:
  CppActiveSpaceGradientEvaluator gradient_evaluator_;
};

/**
 * @brief Evaluates the exact physical selected-state density for one AO value vector.
 *
 * If `ao_values[mu] = chi_mu(r)`, then this routine returns
 * `rho(r) = chi(r)^T P_AO chi(r)`.
 */
double evaluate_selected_state_exact_physical_density(
    const SelectedStateExactPhysicalOneRdmResult& density_result,
    const Eigen::Ref<const Eigen::VectorXd>& ao_values);

/**
 * @brief Evaluates the exact physical selected-state density for a block of AO probe vectors.
 *
 * `ao_value_matrix` stores `n_probe_vectors` AO vectors in column-major order
 * with dimensions `n_basis_functions x n_probe_vectors`.
 */
Eigen::VectorXd evaluate_selected_state_exact_physical_density_batch(
    const SelectedStateExactPhysicalOneRdmResult& density_result,
    const Eigen::Ref<const Eigen::MatrixXd>& ao_value_matrix);

}  // namespace xmvb::vb
