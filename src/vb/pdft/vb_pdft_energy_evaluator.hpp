#pragma once

#include <limits>

#include "vb/pdft/vb_pdft_config.hpp"
#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"
#include "vb/scf/cpp_active_space_gradient_result.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace xmvb::vb::pdft {

/**
 * @brief Evaluates VB-PDFT energy for a selected VBSCF state.
 *
 * This is the top-level interface for VB-PDFT energy evaluation.
 * It orchestrates all the components:
 * - state-specific active-space gradient evaluation
 * - molecular grid generation
 * - AO value evaluation
 * - exact physical `1-RDM` and on-top pair density construction
 * - real-space density evaluation
 * - translated spin density computation
 * - LDA on-top functional integration
 *
 * The implementation follows the energy expression:
 *   E_n^{VB-PDFT} = V_nn + E_one[gamma] + J[rho] + E_ot[rho, Pi]
 *
 * where:
 * - V_nn: nuclear repulsion
 * - E_one: one-electron energy from 1-RDM
 * - J[rho]: classical Coulomb energy
 * - E_ot: on-top functional energy
 */
class VbPdftEnergyEvaluator {
public:
  /**
   * @brief Constructs evaluator with specified configuration and algorithm.
   */
  explicit VbPdftEnergyEvaluator(
      VbPdftConfig config,
      vb::VBSCFAlgorithm algorithm = vb::VBSCFAlgorithm::Original);

  /**
   * @brief Constructs evaluator with an explicit gradient evaluator.
   */
  explicit VbPdftEnergyEvaluator(
      VbPdftConfig config,
      vb::CppActiveSpaceGradientEvaluator gradient_evaluator);

  /**
   * @brief Evaluates VB-PDFT energy for one absolute selected-state index.
   *
   * @param input VBSCF input with molecular geometry and basis set.
   * @param state_index Absolute generalized-eigenstate index (0-based).
   * @param nuclear_repulsion_energy Optional nuclear repulsion energy.
   *        If omitted, the internally computed state-specific SCF result value
   *        is used.
   * @return VB-PDFT energy components and total energy.
   */
  VbPdftEnergyResult evaluate(
      const vb::CppVbInput& input,
      int state_index,
      double nuclear_repulsion_energy =
          std::numeric_limits<double>::quiet_NaN()) const;

  /**
   * @brief Evaluates VB-PDFT energy from an existing state-specific gradient result.
   *
   * The supplied gradient result must contain exactly one selected state with
   * unit weight. Reusing that result avoids repeating the determinant/structure
   * adjoint work when VB-PDFT is called after a gradient evaluation.
   */
  VbPdftEnergyResult evaluate_from_state_specific_gradient(
      const vb::CppVbInput& input,
      const vb::CppActiveSpaceGradientResult& gradient_result,
      double nuclear_repulsion_energy =
          std::numeric_limits<double>::quiet_NaN()) const;

private:
  VbPdftConfig config_;
  vb::CppActiveSpaceGradientEvaluator gradient_evaluator_;
};

}  // namespace xmvb::vb::pdft
