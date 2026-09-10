#include "vbscf/optimization/vbscf_objective.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vbscf/orbitals/charts/canonicalization.hpp"
#include "vbscf/orbitals/gauge/support_preserving.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/optimization/optimization_checks.hpp"

namespace xmvb::vb {

VbScfObjective::VbScfObjective(
    const VbScfInput& input,
    SparseParameterLayout parameter_view,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy,
    const OrbitalGradientEvaluator* orbital_gradient_evaluator,
    const VbScfEvaluator* scf_evaluator)
    : working_input_(input),
      probe_input_buffer_(input),
      parameter_view_(std::move(parameter_view)),
      selected_state_indices_(selected_state_indices),
      state_average_weights_(state_average_weights),
      nuclear_repulsion_energy_(nuclear_repulsion_energy),
      orbital_gradient_evaluator_(orbital_gradient_evaluator),
      scf_evaluator_(scf_evaluator) {}

double VbScfObjective::operator()(
    const Eigen::VectorXd& parameter_vector,
    Eigen::VectorXd& gradient) {
  TrialEvaluation evaluation =
      evaluate_trial_without_committing(parameter_vector);
  gradient = std::move(evaluation.gradient);
  const double energy = evaluation.energy;
  commit_trial_evaluation(std::move(evaluation));
  return energy;
}

void VbScfObjective::ensure_last_reference_energy_gradient() {
  orbital_gradient_evaluator_->populate_reference_energy_gradient(
      working_input_,
      &last_gradient_result_);
}

VbScfObjective::TrialEvaluation
VbScfObjective::evaluate_trial_without_committing(
    const Eigen::VectorXd& parameter_vector) const {
  const auto iteration_start_time = std::chrono::steady_clock::now();
  // TN trial acceptance only needs a scratch orbital point.  Do not copy the
  // whole objective state here: the accepted-point second-order context can
  // be large, and rejected trust-region trials must leave it untouched.
  probe_input_buffer_.orbital_preparation_input =
      working_input_.orbital_preparation_input;
  parameter_view_.unpack(
      parameter_vector,
      &probe_input_buffer_.orbital_preparation_input);

  TrialEvaluation evaluation;
  evaluation.orbital_preparation_input =
      probe_input_buffer_.orbital_preparation_input;
  evaluation.gradient_result =
      orbital_gradient_evaluator_->evaluate_without_reference_energy_gradient(
          probe_input_buffer_,
          selected_state_indices_,
          state_average_weights_,
          nuclear_repulsion_energy_);
  if (evaluation.gradient_result.second_order_context == nullptr) {
    throw std::runtime_error(
        "relaxed orbital gradient did not populate the accepted-point second-order context");
  }

  evaluation.gradient = parameter_view_.gather_from_full(
      evaluation.gradient_result.sparse_orbital_energy_gradient);
  evaluation.energy = evaluation.gradient_result.scf_result.total_energy;
  evaluation.gradient_inf_norm =
      gradient_infinity_norm(evaluation.gradient);
  evaluation.wall_time_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - iteration_start_time)
          .count();
  evaluation.valid = true;
  return evaluation;
}

void VbScfObjective::commit_trial_evaluation(TrialEvaluation evaluation) {
  if (!evaluation.valid) {
    throw std::invalid_argument("cannot commit an invalid orbital trial evaluation");
  }
  // A committed trial becomes the new accepted point used by later gradients,
  // exact-ctx HVPs, chart canonicalization, and accepted-iteration snapshots.
  working_input_.orbital_preparation_input =
      std::move(evaluation.orbital_preparation_input);
  last_gradient_result_ = std::move(evaluation.gradient_result);
  probe_input_buffer_.orbital_preparation_input =
      working_input_.orbital_preparation_input;
  energy_history_.push_back(evaluation.energy);
  gradient_inf_norm_history_.push_back(evaluation.gradient_inf_norm);
  iteration_time_history_seconds_.push_back(evaluation.wall_time_seconds);
  objective_wall_time_seconds_ += evaluation.wall_time_seconds;
}

double VbScfObjective::evaluate_energy_only(
    const Eigen::VectorXd& parameter_vector) const {
  if (scf_evaluator_ == nullptr) {
    throw std::runtime_error("energy-only objective evaluation requires a live SCF evaluator");
  }
  // Trust-region trial rejection only needs the relaxed energy. Rebuild the
  // current orbital point in a scratch input buffer so rejected steps do not
  // pay for full gradient, adjoint, and second-order-context construction.
  probe_input_buffer_.orbital_preparation_input =
      working_input_.orbital_preparation_input;
  parameter_view_.unpack(
      parameter_vector,
      &probe_input_buffer_.orbital_preparation_input);
  const auto evaluation_start_time = std::chrono::steady_clock::now();
  const double energy = scf_evaluator_->evaluate_energy_only(
      probe_input_buffer_,
      selected_state_indices_,
      state_average_weights_,
      nuclear_repulsion_energy_);
  const double elapsed_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - evaluation_start_time)
          .count();
  ++energy_only_call_count_;
  energy_only_wall_time_seconds_ += elapsed_seconds;
  last_energy_only_wall_time_seconds_ = elapsed_seconds;
  return energy;
}

bool VbScfObjective::canonicalize_orbital_chart_at_current_point(
    Eigen::VectorXd* parameter_vector,
    Eigen::VectorXd* gradient,
    std::vector<PackedSecantPair>* packed_secant_history) {
  bool chart_changed = false;
  const auto transform =
      apply_support_preserving_inactive_gauge(
          &working_input_.orbital_preparation_input);
  if (transform.chart_changed) {
    transform_sparse_inactive_orbital_gradient(
        transform,
        working_input_.orbital_preparation_input,
        &last_gradient_result_.sparse_orbital_energy_gradient);

    if (!last_gradient_result_.sparse_orbital_reference_energy_gradient.empty()) {
      transform_sparse_inactive_orbital_gradient(
          transform,
          working_input_.orbital_preparation_input,
          &last_gradient_result_.sparse_orbital_reference_energy_gradient);
    }
    transport_packed_secant_history_with_support_aware_inactive_gauge(
        transform,
        working_input_.orbital_preparation_input,
        parameter_view_,
        packed_secant_history);
    refresh_cached_localized_representative_selector(
        working_input_.orbital_preparation_input,
        &last_gradient_result_.orbital_preparation_result);
    if (last_gradient_result_.second_order_context != nullptr) {
      refresh_cached_localized_representative_selector(
          working_input_.orbital_preparation_input,
          &last_gradient_result_
               .second_order_context
               ->prepared_active_space
               .orbital_result);
    }
    chart_changed = true;
  }

  if (!chart_changed) {
    return false;
  }
  probe_input_buffer_.orbital_preparation_input =
      working_input_.orbital_preparation_input;
  if (parameter_vector != nullptr) {
    *parameter_vector =
        parameter_view_.pack(working_input_.orbital_preparation_input);
  }
  if (gradient != nullptr) {
    *gradient = parameter_view_.gather_from_full(
        last_gradient_result_.sparse_orbital_energy_gradient);
    if (!gradient_inf_norm_history_.empty()) {
      gradient_inf_norm_history_.back() =
          gradient_infinity_norm(*gradient);
    }
  }
  return true;
}

VbScfObjective VbScfObjective::make_probe_copy() const {
  // HVP finite-difference probes and TN trial evaluations only need the
  // current immutable inputs plus the evaluator handles. Reconstruct a fresh
  // objective instead of copying the last accepted gradient result and
  // second-order context into another large object.
  VbScfObjective copy(
      working_input_,
      parameter_view_,
      selected_state_indices_,
      state_average_weights_,
      nuclear_repulsion_energy_,
      orbital_gradient_evaluator_,
      scf_evaluator_);
  copy.probe_input_buffer_.orbital_preparation_input =
      working_input_.orbital_preparation_input;
  return copy;
}

}  // namespace xmvb::vb
