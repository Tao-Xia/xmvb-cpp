#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vbscf/core/contracts/input.hpp"
#include "vbscf/core/contracts/eigensolver.hpp"
#include "vbscf/orbitals/charts/layout.hpp"
#include "vbscf/derivatives/gradient/orbital/evaluator.hpp"
#include "vbscf/derivatives/gradient/orbital/result.hpp"

namespace xmvb::vb {

struct AcceptedPointContext;

/**
 * @brief Result of evaluating an orbital point without accepting it.
 */
struct VbScfObjectiveTrialEvaluation {
  OrbitalPreparationInput orbital_preparation_input;
  OrbitalGradientResult gradient_result;
  Eigen::VectorXd gradient;
  double energy = 0.0;
  double gradient_inf_norm = 0.0;
  double wall_time_seconds = 0.0;
  bool chart_changed = false;
  bool gradient_ready = false;
  bool valid = false;

  std::optional<ActiveSpaceForwardEvaluation> forward_evaluation;
};

/** @brief Exact coupled gradient at a prescribed orbital/structure keyframe. */
struct VbScfCoupledKeyframeEvaluation {
  Eigen::VectorXd orbital_gradient;
  Eigen::MatrixXd normalized_structure_coefficients;
  Eigen::MatrixXd structure_residuals;
  double energy = 0.0;
};

/**
 * @brief Stateful VBSCF objective on a sparse orbital chart.
 *
 * The object owns one complete input. Trial evaluations replace only its
 * orbital state and restore the accepted state before returning. Large,
 * molecule-static integral data are therefore never duplicated for a trial.
 */
class VbScfObjective {
 public:
  using TrialEvaluation = VbScfObjectiveTrialEvaluation;

  VbScfObjective(
      VbScfInput input,
      SparseParameterLayout layout,
      const std::vector<int>& selected_state_indices,
      const std::vector<double>& state_average_weights,
      double nuclear_repulsion_energy,
      StructureEigensolver structure_eigensolver,
      StructureSolveAccuracy structure_solve_accuracy,
      const OrbitalGradientEvaluator* orbital_gradient_evaluator);

  VbScfObjective(const VbScfObjective&) = delete;
  VbScfObjective& operator=(const VbScfObjective&) = delete;
  VbScfObjective(VbScfObjective&&) = default;
  VbScfObjective& operator=(VbScfObjective&&) = default;

  /** @brief Evaluates and accepts an orbital parameter vector. */
  double operator()(const Eigen::VectorXd& parameter_vector,
                    Eigen::VectorXd& gradient);

  /** @brief Returns the current accepted VBSCF input. */
  const VbScfInput& input() const { return input_; }

  /** @brief Transfers the accepted input out after optimization. */
  VbScfInput take_input() && { return std::move(input_); }

  /** @brief Returns the gradient data at the accepted point. */
  const OrbitalGradientResult& gradient_result() const {
    return gradient_result_;
  }

  /** @brief Returns the exact-HVP context at the accepted point. */
  const std::shared_ptr<AcceptedPointContext>&
  second_order_context() const {
    return gradient_result_.second_order_context;
  }

  /** @brief Populates the reference-energy gradient at the accepted point. */
  void ensure_reference_gradient();

  /** @brief Materializes exact structure norms once for final reporting. */
  void ensure_exact_structure_overlap_diagonal();

  const std::vector<double>& energy_history() const { return energy_history_; }
  const std::vector<double>& gradient_inf_norm_history() const {
    return gradient_inf_norm_history_;
  }
  const std::vector<double>& iteration_time_history_seconds() const {
    return iteration_time_history_seconds_;
  }
  std::size_t call_count() const { return energy_history_.size(); }
  double objective_wall_time_seconds() const {
    return objective_wall_time_seconds_;
  }
  double last_gradient_inf_norm() const {
    if (gradient_inf_norm_history_.empty()) {
      return 0.0;
    }
    return gradient_inf_norm_history_.back();
  }

  /** @brief Evaluates a trial point without changing the accepted point. */
  TrialEvaluation evaluate_trial(
      const Eigen::VectorXd& parameter_vector,
      bool canonicalize_sparse_gauge = false) const;

  /** @brief Evaluates only the trial energy and retains reusable forward work. */
  TrialEvaluation evaluate_trial_energy(
      const Eigen::VectorXd& parameter_vector,
      bool canonicalize_sparse_gauge = false) const;

  /**
   * @brief Evaluates a nonstationary coupled keyframe without rediagonalizing.
   *
   * This mirrors the CIAH keyframe operation: both the orbital displacement
   * and the current approximate structure displacement are retained.
   */
  VbScfCoupledKeyframeEvaluation evaluate_coupled_keyframe(
      const Eigen::VectorXd& parameter_vector,
      const Eigen::Ref<const Eigen::MatrixXd>& structure_coefficients) const;

  /** @brief Solves the trial structure problem from a coupled CIAH estimate. */
  TrialEvaluation evaluate_trial_with_structure_guess(
      const Eigen::VectorXd& parameter_vector,
      const Eigen::Ref<const Eigen::MatrixXd>& selected_structure_coefficients) const;

  /** @brief Completes the exact gradient for an energy-screened trial. */
  void complete_trial(TrialEvaluation* evaluation) const;

  /** @brief Makes a previously evaluated trial the accepted point. */
  void commit(TrialEvaluation evaluation);

private:
  mutable VbScfInput input_;
  SparseParameterLayout layout_;
  std::vector<int> state_indices_;
  std::vector<double> state_weights_;
  double nuclear_repulsion_ = 0.0;
  StructureEigensolver structure_eigensolver_ =
      StructureEigensolver::Davidson;
  StructureSolveAccuracy structure_solve_accuracy_;
  const OrbitalGradientEvaluator* gradient_evaluator_ = nullptr;

  OrbitalGradientResult gradient_result_;
  std::vector<double> energy_history_;
  std::vector<double> gradient_inf_norm_history_;
  std::vector<double> iteration_time_history_seconds_;
  double objective_wall_time_seconds_ = 0.0;

  TrialEvaluation evaluate_trial_energy_with_initial_eigenvectors(
      const Eigen::VectorXd& parameter_vector,
      bool canonicalize_sparse_gauge,
      const Eigen::Ref<const Eigen::MatrixXd>& initial_eigenvectors) const;
};

}  // namespace xmvb::vb
