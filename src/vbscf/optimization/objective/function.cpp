#include "vbscf/optimization/objective/function.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vbscf/orbitals/gauge/support_preserving.hpp"
#include "vbscf/orbitals/charts/canonicalization.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/optimization/driver/checks.hpp"
#include "vbscf/structures/assembly/hamiltonian_overlap.hpp"

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
    const OrbitalGradientEvaluator* orbital_gradient_evaluator)
    : input_(std::move(input)),
      layout_(std::move(layout)),
      state_indices_(selected_state_indices),
      state_weights_(state_average_weights),
      nuclear_repulsion_(nuclear_repulsion_energy),
      structure_eigensolver_(structure_eigensolver),
      structure_solve_accuracy_(structure_solve_accuracy),
      gradient_evaluator_(orbital_gradient_evaluator) {}

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

void VbScfObjective::ensure_exact_structure_overlap_diagonal() {
  VbScfResult& scf_result = gradient_result_.scf_result;
  if (scf_result.structure_overlap_diagonal_exact) {
    return;
  }
  const std::shared_ptr<AcceptedPointContext>& accepted =
      gradient_result_.second_order_context;
  if (accepted == nullptr) {
    throw std::runtime_error(
        "final structure normalization requires an accepted-point context");
  }
  const auto& prepared = accepted->prepared_active_space;
  const FullDeterminantStructureHamiltonianOverlapBuilder builder;
  const Eigen::VectorXd exact_diagonal = builder.build_exact_overlap_diagonal(
      input_.structure_data.alpha_det,
      input_.structure_data.beta_det,
      input_.structure_data.determinant_to_structure_terms,
      prepared.orbital_result.active_orbital_overlap_matrix,
      input_.orbital_preparation_input.n_active_orbitals,
      input_.structure_data.n_structures,
      accepted->same_spin_pair_cache);
  scf_result.structure_overlap_diagonal.assign(
      exact_diagonal.data(), exact_diagonal.data() + exact_diagonal.size());
  scf_result.average_structure_overlap = exact_diagonal.mean();
  scf_result.structure_overlap_diagonal_exact = true;
}

VbScfObjective::TrialEvaluation
VbScfObjective::evaluate_trial(
    const Eigen::VectorXd& parameter_vector,
    bool canonicalize_sparse_gauge) const {
  TrialEvaluation evaluation = evaluate_trial_energy(
      parameter_vector,
      canonicalize_sparse_gauge);
  complete_trial(&evaluation);
  return evaluation;
}

VbScfObjective::TrialEvaluation
VbScfObjective::evaluate_trial_energy(
    const Eigen::VectorXd& parameter_vector,
    bool canonicalize_sparse_gauge) const {
  const auto iteration_start_time = std::chrono::steady_clock::now();
  OrbitalPreparationInput trial_orbitals = input_.orbital_preparation_input;
  layout_.unpack(parameter_vector, &trial_orbitals);
  bool chart_changed = false;
  if (canonicalize_sparse_gauge) {
    chart_changed =
        apply_support_preserving_inactive_gauge(&trial_orbitals);
    chart_changed = balance_active_gauge(&trial_orbitals) || chart_changed;
  }
  ScopedTrialOrbitals trial_scope(&input_, std::move(trial_orbitals));

  TrialEvaluation evaluation;
  evaluation.orbital_preparation_input =
      input_.orbital_preparation_input;
  Eigen::MatrixXd initial_eigenvectors;
  if (gradient_result_.second_order_context != nullptr) {
    initial_eigenvectors =
        gradient_result_.second_order_context->root_eigenvectors;
  }
  evaluation.forward_evaluation.emplace(
      gradient_evaluator_->evaluate_forward(
          input_,
          state_indices_,
          state_weights_,
          nuclear_repulsion_,
          structure_eigensolver_,
          structure_solve_accuracy_,
          initial_eigenvectors));
  evaluation.energy = evaluation.forward_evaluation->total_energy();
  evaluation.chart_changed = chart_changed;
  evaluation.wall_time_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - iteration_start_time)
          .count();
  evaluation.valid = true;
  return evaluation;
}

void VbScfObjective::complete_trial(TrialEvaluation* evaluation) const {
  if (evaluation == nullptr || !evaluation->valid ||
      evaluation->gradient_ready ||
      !evaluation->forward_evaluation.has_value()) {
    throw std::invalid_argument(
        "trial completion requires a valid energy-only evaluation");
  }
  const auto completion_start_time = std::chrono::steady_clock::now();
  ScopedTrialOrbitals trial_scope(
      &input_,
      evaluation->orbital_preparation_input);
  evaluation->gradient_result = gradient_evaluator_->complete_gradient(
      input_,
      std::move(*evaluation->forward_evaluation));
  evaluation->forward_evaluation.reset();
  if (evaluation->gradient_result.second_order_context == nullptr) {
    throw std::runtime_error(
        "relaxed orbital gradient did not populate the accepted-point second-order context");
  }
  evaluation->gradient = layout_.gather_from_full(
      evaluation->gradient_result.sparse_orbital_energy_gradient);
  evaluation->energy = evaluation->gradient_result.scf_result.total_energy;
  evaluation->gradient_inf_norm =
      gradient_infinity_norm(evaluation->gradient);
  evaluation->wall_time_seconds +=
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - completion_start_time)
          .count();
  evaluation->gradient_ready = true;
}

void VbScfObjective::commit(TrialEvaluation evaluation) {
  if (!evaluation.valid || !evaluation.gradient_ready) {
    throw std::invalid_argument(
        "cannot commit a trial evaluation without an exact gradient");
  }
  input_.orbital_preparation_input =
      std::move(evaluation.orbital_preparation_input);
  gradient_result_ = std::move(evaluation.gradient_result);
  energy_history_.push_back(evaluation.energy);
  gradient_inf_norm_history_.push_back(evaluation.gradient_inf_norm);
  iteration_time_history_seconds_.push_back(evaluation.wall_time_seconds);
  objective_wall_time_seconds_ += evaluation.wall_time_seconds;
}

}  // namespace xmvb::vb
