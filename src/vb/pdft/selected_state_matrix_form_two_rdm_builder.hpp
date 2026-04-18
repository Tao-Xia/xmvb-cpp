#pragma once

#include <vector>

#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"
#include "vb/scf/cpp_active_space_gradient_result.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace xmvb::vb {

/**
 * @brief Selected-state two-particle density aligned with the current matrix-form VBSCF path.
 *
 * This result intentionally follows the present managed C++ active-space
 * conventions:
 *
 * - `packed_matrix_form_active_two_rdm` is the selected-state derivative
 *   `dE / dGGO` in the current packed active-space pair-of-pairs storage;
 * - `matrix_form_active_pair_density_matrix` is the equivalent symmetric
 *   pair-basis matrix over packed active pairs `P = (p, q)` with `p >= q`.
 *
 * The unpacked pair matrix doubles diagonal entries so that its full symmetric
 * contraction reproduces the packed energy contraction:
 *
 * `E_2 = sum_{P>=Q} G_{PQ} * Gamma_{PQ}`
 *
 * `    = 0.5 * sum_{P,Q} G_pair(P,Q) * Gamma_pair(P,Q)`.
 *
 * As with the current `1-RDM` builder, this is a matrix-form object aligned
 * with the current `GGO` convention, not yet a final physical-orbital
 * `VB-PDFT` `2-RDM`.
 */
struct SelectedStateMatrixFormTwoRdmResult {
  /**
   * @brief Selected-state index in the generalized eigenspectrum.
   */
  int state_index = 0;

  /**
   * @brief Number of packed active pairs `P = (p, q)` with `p >= q`.
   */
  int n_active_pairs = 0;

  /**
   * @brief Selected-state packed matrix-form `2-RDM` in the legacy `GGO` layout.
   *
   * Storage size matches `packed_active_two_electron_integral_count`.
   */
  std::vector<double> packed_matrix_form_active_two_rdm;

  /**
   * @brief Full symmetric pair-density matrix over packed active-pair indices.
   *
   * Column-major `n_active_pairs x n_active_pairs`. Diagonal entries are
   * doubled relative to the packed storage so that a half-trace with the full
   * pair kernel reproduces the packed contraction exactly.
   */
  std::vector<double> matrix_form_active_pair_density_matrix;

  /**
   * @brief Maximum antisymmetric component of the unpacked pair matrix.
   */
  double active_pair_density_symmetry_max_abs = 0.0;

  /**
   * @brief Packed contraction `sum_{P>=Q} G_{PQ} * Gamma_{PQ}` at the current point.
   */
  double packed_active_two_electron_energy = 0.0;

  /**
   * @brief Equivalent unpacked half-trace `0.5 * Tr(G_pair * Gamma_pair)`.
   */
  double unpacked_active_pair_energy = 0.0;
};

/**
 * @brief Builds selected-state matrix-form two-particle densities from current C++ VBSCF objects.
 *
 * The first implementation intentionally reuses the validated active-space
 * gradient evaluator: for a single selected state with unit weight,
 * `dE/dGGO` is the selected-state matrix-form two-particle density aligned
 * with the current packed active-space pair-kernel convention.
 */
class SelectedStateMatrixFormTwoRdmBuilder {
public:
  /**
   * @brief Creates a builder with the default active-space gradient evaluator.
   */
  explicit SelectedStateMatrixFormTwoRdmBuilder(
      VBSCFAlgorithm algorithm = VBSCFAlgorithm::Original);

  /**
   * @brief Creates a builder with an explicit gradient evaluator.
   */
  explicit SelectedStateMatrixFormTwoRdmBuilder(
      CppActiveSpaceGradientEvaluator gradient_evaluator);

  /**
   * @brief Builds the selected-state matrix-form `2-RDM` by running a unit-weight state-specific gradient.
   */
  SelectedStateMatrixFormTwoRdmResult build(
      const CppVbInput& input,
      int state_index,
      double nuclear_repulsion_energy = 0.0) const;

  /**
   * @brief Builds the selected-state matrix-form `2-RDM` from an existing state-specific gradient result.
   *
   * The supplied gradient result must correspond to exactly one selected state
   * with normalized weight 1.
   */
  SelectedStateMatrixFormTwoRdmResult build_from_state_specific_gradient(
      const CppVbInput& input,
      const CppActiveSpaceGradientResult& gradient_result) const;

private:
  CppActiveSpaceGradientEvaluator gradient_evaluator_;
};

}  // namespace xmvb::vb
