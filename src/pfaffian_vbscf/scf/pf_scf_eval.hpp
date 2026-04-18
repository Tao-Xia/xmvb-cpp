#pragma once

#include <vector>

#include "pfaffian_vbscf/matrices/pf_act_builder.hpp"
#include "pfaffian_vbscf/matrices/pf_matrix_builder.hpp"
#include "pfaffian_vbscf/types/pf_active_space_data.hpp"
#include "vb/matrices/cpp_vb_input.hpp"

namespace xmvb::pfaffian_vbscf {

/**
 * @brief End-to-end molecule-level Pfaffian-VBSCF evaluation result.
 */
struct PfScfResult {
  double e_ref = 0.0;
  double e_ele = 0.0;
  double e_tot = 0.0;
  double avg_diag_s = 0.0;
  double matrix_dt = 0.0;
  double total_dt = 0.0;
  std::vector<double> evals;
  PfMatrixBuildResult mats;
};

/**
 * @brief Runs one molecule-level Pfaffian-VBSCF matrix build and generalized
 * eigensolve on a supplied Pfaffian basis.
 */
class PfScfEval {
public:
  /**
   * @brief Creates the evaluator with default helper components.
   */
  PfScfEval();

  /**
   * @brief Creates the evaluator with explicit helper components.
   */
  PfScfEval(
      PfActBuilder active_space_builder,
      PfMatrixBuilder matrix_builder);

  /**
   * @brief Evaluates the lowest Pfaffian-VBSCF state on the current orbitals.
   *
   * @param input Molecule-level C++ VB input bundle.
   * @param basis Pfaffian basis used to span the CI-like subspace.
   * @param nuclear_repulsion_energy Nuclear repulsion energy.
   * @return PfScfResult Reference energy, active-space eigenvalues, and total energy.
   */
  PfScfResult eval(
      const xmvb::vb::CppVbInput& input,
      const PfBasisData& basis,
      double nuclear_repulsion_energy = 0.0) const;

  PfScfResult eval_active_space(
      const PfActiveSpaceData& active_space,
      const PfBasisData& basis,
      double nuclear_repulsion_energy = 0.0,
      double reference_energy = 0.0) const;

private:
  PfActBuilder active_space_builder_;
  PfMatrixBuilder matrix_builder_;
};

}  // namespace xmvb::pfaffian_vbscf
