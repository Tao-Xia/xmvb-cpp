#include "vbscf/optimization/objective/function.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vbscf/orbitals/charts/canonicalization.hpp"
#include "vbscf/orbitals/gauge/support_preserving.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/optimization/driver/checks.hpp"

namespace xmvb::vb {
namespace {

class ScopedTrialOrbitals {
 public:
  ScopedTrialOrbitals(
      VbScfInput* input,
      OrbitalPreparationInput trial_orbitals)
      : input_(input),
        accepted_orbitals_(std::move(input->orbital_preparation_input)) {
    input_->orbital_preparation_input = std::move(trial_orbitals);
  }

  ScopedTrialOrbitals(const ScopedTrialOrbitals&) = delete;
  ScopedTrialOrbitals& operator=(const ScopedTrialOrbitals&) = delete;

  ~ScopedTrialOrbitals() {
    input_->orbital_preparation_input = std::move(accepted_orbitals_);
  }

 private:
  VbScfInput* input_;
  OrbitalPreparationInput accepted_orbitals_;
};

}  // namespace

VbScfObjective::VbScfObjective(
    VbScfInput input,
    SparseParameterLayout layout,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy,
    StructureEigensolver structure_eigensolver,
    StructureSolveAccuracy structure_solve_accuracy,
    const OrbitalGradientEvaluator* orbital_gradient_evaluator,
    const VbScfEvaluator* scf_evaluator)
    : input_(std::move(input)),
      layout_(std::move(layout)),
      state_indices_(selected_state_indices),
      state_weights_(state_average_weights),
      nuclear_repulsion_(nuclear_repulsion_energy),
      structure_eigensolver_(structure_eigensolver),
      structure_solve_accuracy_(structure_solve_accuracy),
      gradient_evaluator_(orbital_gradient_evaluator),
      scf_(scf_evaluator) {}

double VbScfObjective::operator()(
    const Eigen::VectorXd& parameter_vector,
    Eigen::VectorXd& gradient) {
  TrialEvaluation evaluation =
      evaluate_trial(parameter_vector);
  gradient = std::move(evaluation.gradient);
  const double energy = evaluation.energy;
  commit(std::move(evaluation));
  return energy;
}

void VbScfObjective::ensure_reference_gradient() {
  gradient_evaluator_->populate_reference_energy_gradient(
      input_,
      &gradient_result_);
}

VbScfObjective::TrialEvaluation
VbScfObjective::evaluate_trial(
    const Eigen::VectorXd& parameter_vector) const {
  const auto iteration_start_time = std::chrono::steady_clock::now();
  OrbitalPreparationInput trial_orbitals = input_.orbital_preparation_input;
  layout_.unpack(parameter_vector, &trial_orbitals);
  ScopedTrialOrbitals trial_scope(&input_, std::move(trial_orbitals));

  TrialEvaluation evaluation;
  evaluation.orbital_preparation_input =
      input_.orbital_preparation_input;
  Eigen::MatrixXd initial_eigenvectors;
  if (gradient_result_.second_order_context != nullptr) {
    initial_eigenvectors =
        gradient_result_.second_order_context->root_eigenvectors;
  }
  evaluation.gradient_result =
      gradient_evaluator_->evaluate_without_reference_energy_gradient(
          input_,
          state_indices_,
          state_weights_,
          nuclear_repulsion_,
          structure_eigensolver_,
          structure_solve_accuracy_,
          initial_eigenvectors);
  if (evaluation.gradient_result.second_order_context == nullptr) {
    throw std::runtime_error(
        "relaxed orbital gradient did not populate the accepted-point second-order context");
  }

  evaluation.gradient = layout_.gather_from_full(
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

void VbScfObjective::commit(TrialEvaluation evaluation) {
  if (!evaluation.valid) {
    throw std::invalid_argument("cannot commit an invalid orbital trial evaluation");
  }
  input_.orbital_preparation_input =
      std::move(evaluation.orbital_preparation_input);
  gradient_result_ = std::move(evaluation.gradient_result);
  energy_history_.push_back(evaluation.energy);
  gradient_inf_norm_history_.push_back(evaluation.gradient_inf_norm);
  iteration_time_history_seconds_.push_back(evaluation.wall_time_seconds);
  objective_wall_time_seconds_ += evaluation.wall_time_seconds;
}

double VbScfObjective::evaluate_energy_only(
    const Eigen::VectorXd& parameter_vector) const {
  if (scf_ == nullptr) {
    throw std::runtime_error("energy-only objective evaluation requires a live SCF evaluator");
  }
  if (gradient_result_.second_order_context == nullptr) {
    throw std::runtime_error(
        "energy-only objective evaluation requires accepted structure eigenvectors");
  }
  const Eigen::MatrixXd& accepted_eigenvectors =
      gradient_result_.second_order_context->root_eigenvectors;
  OrbitalPreparationInput trial_orbitals = input_.orbital_preparation_input;
  layout_.unpack(parameter_vector, &trial_orbitals);
  ScopedTrialOrbitals trial_scope(&input_, std::move(trial_orbitals));
  const auto evaluation_start_time = std::chrono::steady_clock::now();
  const double energy = scf_->evaluate_energy_only(
      input_,
      state_indices_,
      state_weights_,
      structure_eigensolver_,
      structure_solve_accuracy_,
      accepted_eigenvectors,
      nuclear_repulsion_);
  const double elapsed_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - evaluation_start_time)
          .count();
  ++energy_only_call_count_;
  energy_only_wall_time_seconds_ += elapsed_seconds;
  last_energy_only_wall_time_seconds_ = elapsed_seconds;
  return energy;
}

bool VbScfObjective::canonicalize_chart(
    Eigen::VectorXd* parameter_vector,
    Eigen::VectorXd* gradient,
    std::vector<PackedSecantPair>* packed_secant_history) {
  bool chart_changed = false;
  const auto transform =
      apply_support_preserving_inactive_gauge(
          &input_.orbital_preparation_input);
  if (transform.chart_changed) {
    transform_sparse_inactive_orbital_gradient(
        transform,
        input_.orbital_preparation_input,
        &gradient_result_.sparse_orbital_energy_gradient);

    if (!gradient_result_.sparse_orbital_reference_energy_gradient.empty()) {
      transform_sparse_inactive_orbital_gradient(
          transform,
          input_.orbital_preparation_input,
          &gradient_result_.sparse_orbital_reference_energy_gradient);
    }
    transport_packed_secant_history_with_support_aware_inactive_gauge(
        transform,
        input_.orbital_preparation_input,
        layout_,
        packed_secant_history);
    refresh_cached_localized_representative_selector(
        input_.orbital_preparation_input,
        &gradient_result_.orbital_preparation_result);
    if (gradient_result_.second_order_context != nullptr) {
      refresh_cached_localized_representative_selector(
          input_.orbital_preparation_input,
          &gradient_result_
               .second_order_context
               ->prepared_active_space
               .orbital_result);
    }
    chart_changed = true;
  }

  if (!chart_changed) {
    return false;
  }
  if (parameter_vector != nullptr) {
    *parameter_vector =
        layout_.pack(input_.orbital_preparation_input);
  }
  if (gradient != nullptr) {
    *gradient = layout_.gather_from_full(
        gradient_result_.sparse_orbital_energy_gradient);
    if (!gradient_inf_norm_history_.empty()) {
      gradient_inf_norm_history_.back() =
          gradient_infinity_norm(*gradient);
    }
  }
  return true;
}

}  // namespace xmvb::vb
