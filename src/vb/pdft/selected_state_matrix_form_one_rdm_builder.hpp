#pragma once

#include <vector>

#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"
#include "vb/scf/cpp_active_space_gradient_result.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace xmvb::vb {

/**
 * @brief Selected-state one-particle density aligned with the current matrix-form VBSCF path.
 *
 * This result intentionally follows the present managed C++ active-space
 * conventions:
 *
 * - `matrix_form_active_one_rdm` is the selected-state one-particle density in
 *   the current nonorthogonal active-orbital basis that contracts directly
 *   with `HHO`;
 * - the AO-space matrices below are reconstructed from the current auxiliary
 *   active orbitals used by the matrix-form VBSCF path, not from the future
 *   physical-orbital `VB-PDFT` frame.
 *
 * The active-space electron count is therefore recovered through the overlap
 * metric
 *
 * `N_active = Tr(gamma_active * SSO)`
 *
 * rather than through a raw matrix trace.
 */
struct SelectedStateMatrixFormOneRdmResult {
  /**
   * @brief Selected-state index in the generalized eigenspectrum.
   */
  int state_index = 0;

  /**
   * @brief Symmetrized selected-state one-particle density in the current active basis.
   *
   * Column-major `n_active_orbitals x n_active_orbitals`.
   */
  std::vector<double> matrix_form_active_one_rdm;

  /**
   * @brief Active-orbital overlap matrix `SSO` used to interpret the active-space density.
   *
   * Column-major `n_active_orbitals x n_active_orbitals`.
   */
  std::vector<double> active_orbital_overlap_matrix;

  /**
   * @brief AO density reconstructed from the current auxiliary active orbitals only.
   *
   * Column-major `n_basis_functions x n_basis_functions`.
   */
  std::vector<double> matrix_form_ao_active_density_matrix;

  /**
   * @brief Total AO matrix-form density used by the present AO-H1E backprop route.
   *
   * This is
   *
   * `2 * P11 + T_active * sym(gamma_active) * T_active^T`
   *
   * in the current auxiliary-orbital convention.
   */
  std::vector<double> matrix_form_ao_total_density_matrix;

  /**
   * @brief Maximum antisymmetric component of the raw active-space gradient before symmetrization.
   */
  double active_one_rdm_asymmetry_max_abs = 0.0;

  /**
   * @brief Metric trace `Tr(gamma_active * SSO)`.
   */
  double active_metric_trace = 0.0;

  /**
   * @brief Metric trace of the auxiliary-active AO density `Tr(P_active * S_AO)`.
   */
  double ao_active_metric_trace = 0.0;

  /**
   * @brief Metric trace of the full matrix-form AO density `Tr(P_total * S_AO)`.
   */
  double ao_total_metric_trace = 0.0;
};

/**
 * @brief Builds selected-state matrix-form one-particle densities from current C++ VBSCF objects.
 *
 * The first implementation intentionally reuses the validated active-space
 * gradient evaluator: for a single selected state with unit weight,
 * `dE/dHHO` is the state-specific matrix-form one-particle density aligned
 * with the current nonorthogonal active-space Hamiltonian convention.
 */
class SelectedStateMatrixFormOneRdmBuilder {
public:
  /**
   * @brief Creates a builder with the default active-space gradient evaluator.
   */
  explicit SelectedStateMatrixFormOneRdmBuilder(
      VBSCFAlgorithm algorithm = VBSCFAlgorithm::Original);

  /**
   * @brief Creates a builder with an explicit gradient evaluator.
   */
  explicit SelectedStateMatrixFormOneRdmBuilder(
      CppActiveSpaceGradientEvaluator gradient_evaluator);

  /**
   * @brief Builds the selected-state matrix-form density by running a unit-weight state-specific gradient.
   */
  SelectedStateMatrixFormOneRdmResult build(
      const CppVbInput& input,
      int state_index,
      double nuclear_repulsion_energy = 0.0) const;

  /**
   * @brief Builds the selected-state matrix-form density from an existing state-specific gradient result.
   *
   * The supplied gradient result must correspond to exactly one selected state
   * with normalized weight 1.
   */
  SelectedStateMatrixFormOneRdmResult build_from_state_specific_gradient(
      const CppVbInput& input,
      const CppActiveSpaceGradientResult& gradient_result) const;

private:
  CppActiveSpaceGradientEvaluator gradient_evaluator_;
};

}  // namespace xmvb::vb
