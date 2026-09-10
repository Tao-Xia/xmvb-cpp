#pragma once

// VbScfObjective wraps the SCF + orbital-gradient evaluators into the
// L-BFGS / truncated-Newton objective surface used by vbscf_optimizer.
// The class owns the accepted orbital point, the last committed gradient
// result, and the energy / gradient-norm histories; trial evaluations are
// staged into a scratch buffer so rejected trust-region steps do not pay for
// full gradient, adjoint, and second-order-context construction.

#include <chrono>
#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "vbscf/core/vbscf_input.hpp"
#include "vbscf/orbitals/charts/layout.hpp"
#include "vbscf/derivatives/gradient/orbital/evaluator.hpp"
#include "vbscf/derivatives/gradient/orbital/result.hpp"
#include "vbscf/workflow/vbscf_evaluator.hpp"
#include "vbscf/optimization/driver/types.hpp"

namespace xmvb::vb {

struct AcceptedPointContext;

// One accepted or trialed orbital point evaluated without committing state.
// Carries everything the optimizer needs to decide whether to accept the
// step: gradient in packed coordinates, SCF total energy, infinity-norm of
// the gradient, wall time, and the inputs that produced it.
struct VbScfObjectiveTrialEvaluation {
  OrbitalPreparationInput orbital_preparation_input;
  OrbitalGradientResult gradient_result;
  Eigen::VectorXd gradient;
  double energy = 0.0;
  double gradient_inf_norm = 0.0;
  double wall_time_seconds = 0.0;
  bool valid = false;
};

class VbScfObjective {
 public:
  using TrialEvaluation = VbScfObjectiveTrialEvaluation;

  VbScfObjective(
      const VbScfInput& input,
      SparseParameterLayout parameter_view,
      const std::vector<int>& selected_state_indices,
      const std::vector<double>& state_average_weights,
      double nuclear_repulsion_energy,
      const OrbitalGradientEvaluator* orbital_gradient_evaluator,
      const VbScfEvaluator* scf_evaluator);

  double operator()(const Eigen::VectorXd& parameter_vector,
                    Eigen::VectorXd& gradient);

  const VbScfInput& last_input() const { return working_input_; }
  const OrbitalGradientResult& last_gradient_result() const {
    return last_gradient_result_;
  }
  const std::shared_ptr<AcceptedPointContext>&
  last_second_order_context() const {
    return last_gradient_result_.second_order_context;
  }

  void ensure_last_reference_energy_gradient();

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
  std::size_t energy_only_call_count() const { return energy_only_call_count_; }
  double energy_only_wall_time_seconds() const {
    return energy_only_wall_time_seconds_;
  }
  double last_energy_only_wall_time_seconds() const {
    return last_energy_only_wall_time_seconds_;
  }
  double last_gradient_inf_norm() const {
    if (gradient_inf_norm_history_.empty()) {
      return 0.0;
    }
    return gradient_inf_norm_history_.back();
  }

  TrialEvaluation evaluate_trial_without_committing(
      const Eigen::VectorXd& parameter_vector) const;

  void commit_trial_evaluation(TrialEvaluation evaluation);

  double evaluate_energy_only(const Eigen::VectorXd& parameter_vector) const;

  bool canonicalize_orbital_chart_at_current_point(
      Eigen::VectorXd* parameter_vector,
      Eigen::VectorXd* gradient,
      std::vector<PackedSecantPair>* packed_secant_history = nullptr);

  VbScfObjective make_probe_copy() const;

 private:
  VbScfInput working_input_;
  mutable VbScfInput probe_input_buffer_;
  SparseParameterLayout parameter_view_;
  std::vector<int> selected_state_indices_;
  std::vector<double> state_average_weights_;
  double nuclear_repulsion_energy_ = 0.0;
  const OrbitalGradientEvaluator* orbital_gradient_evaluator_ = nullptr;
  const VbScfEvaluator* scf_evaluator_ = nullptr;

  OrbitalGradientResult last_gradient_result_;
  std::vector<double> energy_history_;
  std::vector<double> gradient_inf_norm_history_;
  std::vector<double> iteration_time_history_seconds_;
  double objective_wall_time_seconds_ = 0.0;
  mutable std::size_t energy_only_call_count_ = 0;
  mutable double energy_only_wall_time_seconds_ = 0.0;
  mutable double last_energy_only_wall_time_seconds_ = 0.0;
};

}  // namespace xmvb::vb
