#include "vbscf/optimization/objective/reduced_hvp.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "vbscf/optimization/trust_region/retraction.hpp"

namespace xmvb::vb {
namespace {

const char* bool_name(bool value) {
  return value ? "true" : "false";
}

}  // namespace

Eigen::MatrixXd ReducedHvpOperator::apply_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions) {
  Eigen::MatrixXd responses(
      reduced_directions.rows(),
      reduced_directions.cols());
  for (Eigen::Index column = 0;
       column < reduced_directions.cols();
       ++column) {
    responses.col(column) = apply(reduced_directions.col(column));
  }
  return responses;
}

FullFiniteDifferenceReducedHvpOperator::
    FullFiniteDifferenceReducedHvpOperator(
        const VbScfObjective& objective,
        const OrbitalChart& current_space,
        const OrbitalChart::ProjectionResult& current_projection,
        const OrbitalPreparationInput& current_orbital_input,
        const SparseParameterLayout& parameter_view,
        double hvp_step_size)
    : probe_objective_(objective.make_probe()),
      current_space_(current_space),
      current_reduced_gradient_(current_projection.reduced_gradient),
      current_orbital_input_(current_orbital_input),
      parameter_view_(parameter_view),
      hvp_step_size_(hvp_step_size) {}

Eigen::VectorXd FullFiniteDifferenceReducedHvpOperator::apply(
    const Eigen::VectorXd& reduced_direction) {
  if (reduced_direction.size() == 0) {
    return Eigen::VectorXd::Zero(0);
  }

  const Eigen::VectorXd packed_direction =
      gather_nonredundant_retract_tangent(
          current_orbital_input_,
          current_space_,
          parameter_view_,
          reduced_direction);
  const double packed_direction_norm = packed_direction.norm();
  if (!(packed_direction_norm > 0.0) ||
      !std::isfinite(packed_direction_norm)) {
    return Eigen::VectorXd::Zero(reduced_direction.size());
  }

  const double epsilon =
      hvp_step_size_ / std::max(1.0, packed_direction_norm);
  if (!(epsilon > 0.0) || !std::isfinite(epsilon)) {
    throw std::runtime_error("invalid finite-difference step for reduced HVP");
  }

  const OrbitalPreparationInput trial_orbital_input =
      current_space_.retract_step(
          current_orbital_input_,
          reduced_direction,
          epsilon);
  const Eigen::VectorXd trial_parameters =
      parameter_view_.pack(trial_orbital_input);
  const VbScfObjective::TrialEvaluation trial_evaluation =
      probe_objective_.evaluate_trial(trial_parameters);
  return
      (current_space_.project_reduced_gradient(trial_evaluation.gradient) -
       current_reduced_gradient_) /
      epsilon;
}

ExactContextReducedHvpOperator::ExactContextReducedHvpOperator(
    const VbScfObjective& objective,
    const OrbitalChart& current_space)
    : exact_operator_(
          objective.second_order_context(),
          &objective.input(),
          SparseParameterLayout(
              objective.input().orbital_preparation_input),
          &current_space) {}

Eigen::VectorXd ExactContextReducedHvpOperator::apply(
    const Eigen::VectorXd& reduced_direction) {
  return exact_operator_.apply_reduced(reduced_direction);
}

Eigen::MatrixXd ExactContextReducedHvpOperator::apply_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions) {
  return exact_operator_.apply_reduced_batch(reduced_directions);
}

bool ExactContextReducedHvpOperator::supports_analytic_core_model()
    const noexcept {
  return exact_operator_.supports_analytic_core_model();
}

ExactHvpOperator::Diagnostics ExactContextReducedHvpOperator::diagnostics()
    const {
  return exact_operator_.diagnostics();
}

std::string build_exact_ctx_unavailable_message(
    const ExactContextReducedHvpOperator& hvp_operator) {
  const auto info = hvp_operator.diagnostics();
  std::ostringstream message;
  message << "exact_ctx HVP is unavailable"
          << ": supports_analytic_core_model="
          << bool_name(info.supports_analytic_core_model)
          << " outer_response_enabled="
          << bool_name(info.outer_response_enabled)
          << " has_same_spin_matrix_form="
          << bool_name(info.has_same_spin_matrix_form)
          << " has_opposite_spin_matrix_form="
          << bool_name(info.has_opposite_spin_matrix_form)
          << " n_selected_states=" << info.n_selected_states
          << " n_active_orbitals=" << info.n_active_orbitals
          << " n_blocks=" << info.n_blocks;
  return message.str();
}

}  // namespace xmvb::vb
