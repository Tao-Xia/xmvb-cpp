#pragma once

#include <cstddef>

#include <Eigen/Core>

#include <optional>
#include <string>
#include <vector>

#include "vbscf/core/contracts/input.hpp"
#include "vbscf/core/contracts/result.hpp"

namespace xmvb::vb {

/**
 * @brief Solver diagnostics for one accepted TNHVP outer iteration.
 *
 * Counts include every trust-region attempt made from the same accepted
 * source point, while model quantities describe the step that was ultimately
 * accepted. This distinction makes rejected-radius retries visible without
 * confusing them with additional outer iterations.
 */
struct TnhvpIterationRecord {
  int accepted_iteration_index = 0;
  int reduced_dimension = 0;
  int subspace_dimension = 0;
  int rejected_trial_count = 0;

  std::size_t hvp_direction_count = 0;
  std::size_t hvp_batch_count = 0;
  std::size_t subproblem_count = 0;
  double hvp_wall_time_seconds = 0.0;

  double source_gradient_l2_norm = 0.0;
  double accepted_gradient_l2_norm = 0.0;
  double forcing_term = 0.0;
  bool has_kkt_residual = false;
  double kkt_relative_residual = 0.0;

  double initial_trust_radius = 0.0;
  double accepted_trial_radius = 0.0;
  double next_trust_radius = 0.0;
  double step_norm = 0.0;

  double predicted_decrease = 0.0;
  double actual_decrease = 0.0;
  double trust_ratio = 0.0;
  double model_spectral_radius = 0.0;
  double trust_region_shift = 0.0;

  bool reached_boundary = false;
  bool encountered_negative_curvature = false;
  bool reused_subspace = false;
};

/**
 * @brief Accepted optimizer iterate with the quantities needed for DeepVBSCF tracing.
 *
 * The trace stores the initial point at index 0 and then each accepted outer
 * optimization step. Static metadata such as sparse basis indices and raw
 * structure definitions are intentionally not duplicated here.
 */
struct VbScfAcceptedIterationSnapshot {
  /**
   * @brief Accepted iteration index, with the initial point stored as 0.
   */
  int accepted_iteration_index = 0;

  /** TNHVP solver data, absent at the initial point and for other backends. */
  std::optional<TnhvpIterationRecord> tnhvp;

  /**
   * @brief Whether this snapshot carries the heavyweight matrix/integral payloads.
   *
   * Lightweight callback paths such as the terminal logger only need scalar
   * summaries, so they can skip deep copies of structure matrices, orbital
   * tables, and integral buffers by leaving this flag `false`.
   */
  bool has_full_payload = true;

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
  Eigen::MatrixXd active_one_electron_integrals;

  /**
   * @brief Packed active-space two-electron integrals for reconstructing `J` and `K`.
   */
  std::vector<double> packed_active_two_electron_integrals;

  /**
   * @brief Exact energy gradient with respect to `orbital_value_table`.
   */
  std::vector<double> sparse_orbital_energy_gradient;

  /**
   * @brief Infinity norm of the exact energy gradient at this accepted iterate.
   *
   * This scalar is always populated, even when the heavyweight gradient vector
   * itself is omitted from a lightweight callback snapshot.
   */
  double sparse_orbital_energy_gradient_inf_norm = 0.0;

  /**
   * @brief Euclidean norm of the exact energy gradient at this accepted iterate.
   *
   * This scalar is always populated, even when the heavyweight gradient vector
   * itself is omitted from a lightweight callback snapshot.
   */
  double sparse_orbital_energy_gradient_l2_norm = 0.0;

  /** Whether nonredundant projected-gradient norms are available. */
  bool has_projected_gradient = false;

  /** Infinity norm of the gradient in the nonredundant orbital chart. */
  double projected_gradient_inf_norm = 0.0;

  /** Euclidean norm of the gradient in the nonredundant orbital chart. */
  double projected_gradient_l2_norm = 0.0;

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
 * @brief Result of the VBSCF orbital optimization loop.
 */
struct VbScfOptimizerResult {
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
   * @brief Initial inactive-space reference energy `E11`.
   */
  double initial_one_electron_reference_energy = 0.0;

  /**
   * @brief Final inactive-space reference energy `E11`.
   */
  double final_one_electron_reference_energy = 0.0;

  /**
   * @brief Final gradient infinity norm over differentiable parameters.
   */
  double final_gradient_inf_norm = 0.0;

  /**
   * @brief Final projected gradient infinity norm in the nonredundant reduced space.
   */
  double final_projected_gradient_inf_norm = 0.0;

  /**
   * @brief Final gradient Euclidean norm reported by the optimizer.
   */
  double final_gradient_l2_norm = 0.0;

  /**
   * @brief Final projected gradient Euclidean norm in the nonredundant reduced space.
   */
  double final_projected_gradient_l2_norm = 0.0;

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

  /** Number of reduced directions evaluated by the matrix-free Hessian. */
  std::size_t matrix_free_hvp_direction_count = 0;

  /** Number of block-HVP calls (each may contain multiple directions). */
  std::size_t matrix_free_hvp_batch_count = 0;

  /** Fresh HVP-subspace solves; same-point cached-radius retries are excluded. */
  std::size_t matrix_free_subproblem_count = 0;
  /** Fresh subspace solves whose trust-region solution is interior. */
  std::size_t matrix_free_interior_subproblem_count = 0;
  /** Interior solves passing a post-hoc full 2-norm Newton residual check. */
  std::size_t matrix_free_residual_converged_count = 0;

  /** Wall time spent inside exact reduced Hessian actions. */
  double matrix_free_hvp_wall_time_seconds = 0.0;

  /** Number of matrix-free structure-energy trial screens. */
  std::size_t energy_only_evaluation_count = 0;

  /** Wall time spent in structure-energy trial screens. */
  double energy_only_wall_time_seconds = 0.0;

  /** Lightweight per-accepted-step diagnostics for the TNHVP backend. */
  std::vector<TnhvpIterationRecord> tnhvp_iteration_trace;

  /**
   * @brief Explicitly requested accepted-iterate trace.
   *
   * Entry 0 stores the initial point before any accepted optimization step.
   * Retention is disabled by default because full snapshots materialize dense
   * structure matrices and scale quadratically with the structure count.
   */
  std::vector<VbScfAcceptedIterationSnapshot> accepted_iteration_trace;

  /**
   * @brief Last accepted single-step VBSCF result.
   */
  VbScfResult scf_result;

  /**
   * @brief Final spin-summed AO one-particle density matrix.
   *
   * This contravariant AO density satisfies `trace(P * S) = N_e` and is kept
   * for natural-orbital and population analyses in the output layer.
   */
  Eigen::MatrixXd one_particle_density_matrix;

  /**
   * @brief Optimized matrix-builder input, including orbital parameters.
   */
  VbScfInput optimized_input;
};

}  // namespace xmvb::vb
