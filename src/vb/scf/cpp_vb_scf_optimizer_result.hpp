#pragma once

#include <string>
#include <vector>

#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/scf/cpp_vb_scf_result.hpp"

namespace xmvb::vb {

/**
 * @brief Accepted optimizer iterate with the quantities needed for DeepVBSCF tracing.
 *
 * The trace stores the initial point at index 0 and then each accepted outer
 * optimization step. Static metadata such as sparse basis indices and raw
 * structure definitions are intentionally not duplicated here.
 */
struct CppVbScfAcceptedIterationSnapshot {
  /**
   * @brief Accepted iteration index, with the initial point stored as 0.
   */
  int accepted_iteration_index = 0;

  /**
   * @brief Sparse orbital coefficient table at this accepted iterate.
   */
  std::vector<double> orbital_value_table;

  /**
   * @brief Exact structure overlap and Hamiltonian matrices at this iterate.
   */
  StructureAccumulationResult structure_matrices;

  /**
   * @brief Column-major active-orbital overlap matrix `SSO` at this iterate.
   */
  std::vector<double> active_orbital_overlap_matrix;

  /**
   * @brief Column-major active-space effective one-electron matrix `HHO`.
   */
  std::vector<double> active_one_electron_integrals;

  /**
   * @brief Packed active-space two-electron integrals for reconstructing `J` and `K`.
   */
  std::vector<double> packed_active_two_electron_integrals;

  /**
   * @brief Exact energy gradient with respect to `orbital_value_table`.
   */
  std::vector<double> sparse_orbital_energy_gradient;

  /**
   * @brief Exact `E11` gradient with respect to `orbital_value_table`.
   */
  std::vector<double> sparse_orbital_reference_energy_gradient;

  /**
   * @brief Total energy associated with this accepted iterate.
   */
  double total_energy = 0.0;

  /**
   * @brief Inactive-space reference energy `E11` at this accepted iterate.
   */
  double one_electron_reference_energy = 0.0;

  /**
   * @brief Scalar summary of the structure overlap matrix at this iterate.
   */
  double average_structure_overlap = 0.0;
};

/**
 * @brief Result of the C++ VBSCF orbital optimization loop.
 */
struct CppVbScfOptimizerResult {
  /**
   * @brief Whether the optimization satisfied the convergence criterion.
   */
  bool converged = false;

  /**
   * @brief Human-readable termination reason.
   */
  std::string termination_reason;

  /**
   * @brief Number of accepted optimization iterations.
   */
  int n_iterations = 0;

  /**
   * @brief Initial total energy.
   */
  double initial_total_energy = 0.0;

  /**
   * @brief Final total energy.
   */
  double final_total_energy = 0.0;

  /**
   * @brief Final gradient infinity norm over differentiable parameters.
   */
  double final_gradient_inf_norm = 0.0;

  /**
   * @brief Final gradient Euclidean norm reported by the legacy L-BFGS driver.
   */
  double final_gradient_l2_norm = 0.0;

  /**
   * @brief Total energy after each objective/gradient evaluation, including the initial value.
   */
  std::vector<double> total_energy_history;

  /**
   * @brief Gradient infinity norm after each objective/gradient evaluation, including the initial value.
   */
  std::vector<double> gradient_inf_norm_history;

  /**
   * @brief Wall-clock time in seconds for each objective/gradient evaluation.
   */
  std::vector<double> iteration_time_history_seconds;

  /**
   * @brief Total wall-clock optimization time in seconds.
   */
  double total_wall_time_seconds = 0.0;

  /**
   * @brief Accepted-iterate trace with orbital coefficients and exact structure matrices.
   *
   * Entry 0 stores the initial point before any accepted optimization step.
   */
  std::vector<CppVbScfAcceptedIterationSnapshot> accepted_iteration_trace;

  /**
   * @brief Last accepted single-step VBSCF result.
   */
  CppVbScfResult scf_result;

  /**
   * @brief Optimized matrix-builder input, including orbital parameters.
   */
  CppVbInput optimized_input;
};

}  // namespace xmvb::vb
