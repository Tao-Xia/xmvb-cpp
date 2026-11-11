#include "vbscf/optimization/objective/function.hpp"
#include "core/timing.hpp"
#include "mcscf/optimization/trace.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vbscf/orbitals/gauge/support_preserving.hpp"
#include "vbscf/orbitals/charts/canonicalization.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/optimization/driver/checks.hpp"
#include "vbscf/optimization/coupled/ci_transport.hpp"

namespace xmvb::vb {
namespace {

class ScopedTrialOrbitals {
  public:
    ScopedTrialOrbitals(VbScfInput* input, OrbitalPreparationInput trial_orbitals)
        : input_(input), accepted_orbitals_(std::move(input->orbital_preparation_input)) {
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

} // namespace

VbScfObjective::VbScfObjective(VbScfInput input,
                               SparseParameterLayout layout,
                               const std::vector<int>& selected_state_indices,
                               const std::vector<double>& state_average_weights,
                               double nuclear_repulsion_energy,
                               StructureEigensolver structure_eigensolver,
                               StructureSolveAccuracy structure_solve_accuracy,
                               const OrbitalGradientEvaluator* orbital_gradient_evaluator,
                               bool active_inactive_orthogonal)
    : input_(std::move(input)), layout_(std::move(layout)), state_indices_(selected_state_indices),
      state_weights_(state_average_weights), nuclear_repulsion_(nuclear_repulsion_energy),
      structure_eigensolver_(structure_eigensolver),
      structure_solve_accuracy_(structure_solve_accuracy),
      gradient_evaluator_(orbital_gradient_evaluator),
      active_inactive_orthogonal_(active_inactive_orthogonal) {}

double VbScfObjective::operator()(const Eigen::VectorXd& parameter_vector,
                                  Eigen::VectorXd& gradient) {
    TrialEvaluation evaluation = evaluate_trial(parameter_vector);
    gradient = std::move(evaluation.gradient);
    const double energy = evaluation.energy;
    commit(std::move(evaluation));
    return energy;
}

void VbScfObjective::ensure_reference_gradient() {
    gradient_evaluator_->populate_reference_energy_gradient(input_, &gradient_result_);
}

VbScfObjective::TrialEvaluation
VbScfObjective::evaluate_trial(const Eigen::VectorXd& parameter_vector,
                               bool canonicalize_sparse_gauge) const {
    TrialEvaluation evaluation = evaluate_trial_energy(parameter_vector, canonicalize_sparse_gauge);
    complete_trial(&evaluation);
    return evaluation;
}

VbScfObjective::TrialEvaluation
VbScfObjective::evaluate_trial_energy(const Eigen::VectorXd& parameter_vector,
                                      bool canonicalize_sparse_gauge) const {
    Eigen::MatrixXd initial_eigenvectors;
    if (gradient_result_.second_order_context != nullptr) {
        initial_eigenvectors = gradient_result_.second_order_context->root_eigenvectors;
    }
    return evaluate_trial_energy_with_initial_eigenvectors(
        parameter_vector, canonicalize_sparse_gauge, initial_eigenvectors);
}

VbScfObjective::TrialEvaluation VbScfObjective::evaluate_trial_energy_with_initial_eigenvectors(
    const Eigen::VectorXd& parameter_vector,
    bool canonicalize_sparse_gauge,
    const Eigen::Ref<const Eigen::MatrixXd>& initial_eigenvectors,
    double exploratory_stationarity) const {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::TrialForward, true);
    const auto iteration_start_time = std::chrono::steady_clock::now();
    OrbitalPreparationInput trial_orbitals = input_.orbital_preparation_input;
    layout_.unpack(parameter_vector, &trial_orbitals);
    bool chart_changed = false;
    bool active_inactive_orthogonal = false;
    if (canonicalize_sparse_gauge) {
        chart_changed = apply_support_preserving_inactive_gauge(&trial_orbitals);
        chart_changed =
            balance_active_gauge(&trial_orbitals, &active_inactive_orthogonal) || chart_changed;
    } else if (active_inactive_orthogonal_) {
        active_inactive_orthogonal =
            trial_orbitals.orbital_value_table == input_.orbital_preparation_input.orbital_value_table;
    }
    ScopedTrialOrbitals trial_scope(&input_, std::move(trial_orbitals));

    TrialEvaluation evaluation;
    evaluation.orbital_preparation_input = input_.orbital_preparation_input;
    auto accuracy = structure_solve_accuracy_;
    accuracy.exploratory_stationarity = exploratory_stationarity;
    evaluation.forward_evaluation.emplace(
        gradient_evaluator_->evaluate_forward(input_,
                                              state_indices_,
                                              state_weights_,
                                              nuclear_repulsion_,
                                              structure_eigensolver_,
                                              accuracy,
                                              initial_eigenvectors,
                                              second_order_context().get()));
    evaluation.energy = evaluation.forward_evaluation->total_energy();
    evaluation.chart_changed = chart_changed;
    evaluation.active_inactive_orthogonal = active_inactive_orthogonal;
    evaluation.wall_time_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - iteration_start_time)
            .count();
    evaluation.valid = true;
    return evaluation;
}

VbScfCoupledKeyframeEvaluation VbScfObjective::evaluate_coupled_keyframe(
    const Eigen::VectorXd& parameter_vector,
    const Eigen::Ref<const Eigen::MatrixXd>& structure_coefficients,
    const SelectedMetric* metric) const {
    OrbitalPreparationInput trial_orbitals = input_.orbital_preparation_input;
    layout_.unpack(parameter_vector, &trial_orbitals);
    return evaluate_coupled_keyframe_at(
        std::move(trial_orbitals), structure_coefficients, metric, nullptr);
}

VbScfCoupledKeyframeEvaluation VbScfObjective::evaluate_coupled_keyframe(
    const CiTransportPoint& point) const {
    const PreparedActiveOrbitalView geometry{point.prepared, point.frame};
    return evaluate_coupled_keyframe_at(
        point.input, point.coefficients, point.selected_metric.get(), &geometry);
}

VbScfCoupledKeyframeEvaluation VbScfObjective::evaluate_coupled_keyframe_at(
    OrbitalPreparationInput trial_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& structure_coefficients,
    const SelectedMetric* metric,
    const PreparedActiveOrbitalView* prepared_orbitals) const {
    const auto start = std::chrono::steady_clock::now();
    ScopedTrialOrbitals trial_scope(&input_, std::move(trial_orbitals));
    OrbitalFixedStructureGradient fixed =
        gradient_evaluator_->evaluate_fixed_structure(input_,
                                                      state_indices_,
                                                      state_weights_,
                                                      nuclear_repulsion_,
                                                      structure_coefficients,
                                                      metric,
                                                      false,
                                                      second_order_context().get(),
                                                      prepared_orbitals);
    VbScfCoupledKeyframeEvaluation result;
    result.orbital_gradient =
        layout_.gather_from_full(fixed.gradient.sparse_orbital_energy_gradient);
    result.normalized_structure_coefficients = std::move(fixed.normalized_coefficients);
    result.structure_residuals = std::move(fixed.structure_residuals);
    result.coefficient_gradient = std::move(fixed.coefficient_gradient);
    result.energy = fixed.gradient.scf_result.total_energy;
    mcscf::diagnostics::ciah_trace(
        "ciah_keyframe_cost",
        "seconds=%.17g,active=%.17g,two_electron_backward=%.17g,ao_backward=%.17g,orbital_backward="
        "%.17g",
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
        fixed.gradient.active_space_gradient_wall_time_seconds,
        fixed.gradient.active_space_two_electron_backpropagation_wall_time_seconds,
        fixed.gradient.ao_effective_one_electron_backpropagation_wall_time_seconds,
        fixed.gradient.orbital_backpropagation_wall_time_seconds);
    return result;
}

VbScfObjective::TrialEvaluation VbScfObjective::evaluate_trial_energy_with_structure_guess(
    const Eigen::VectorXd& parameter_vector,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_structure_coefficients,
    double exploratory_stationarity) const {
    if (gradient_result_.second_order_context == nullptr) {
        throw std::runtime_error("coupled trial requires an accepted-point structure eigensystem");
    }
    Eigen::MatrixXd initial_eigenvectors = gradient_result_.second_order_context->root_eigenvectors;
    if (selected_structure_coefficients.rows() != initial_eigenvectors.rows() ||
        selected_structure_coefficients.cols() !=
            static_cast<Eigen::Index>(state_indices_.size()) ||
        !selected_structure_coefficients.allFinite()) {
        throw std::invalid_argument(
            "coupled trial structure guess has inconsistent dimensions or values");
    }
    for (std::size_t state = 0; state < state_indices_.size(); ++state) {
        const int root = state_indices_[state];
        if (root < 0 || root >= initial_eigenvectors.cols()) {
            throw std::runtime_error(
                "accepted structure eigensystem does not contain a selected root");
        }
        initial_eigenvectors.col(root) =
            selected_structure_coefficients.col(static_cast<Eigen::Index>(state));
    }
    return evaluate_trial_energy_with_initial_eigenvectors(
        parameter_vector, true, initial_eigenvectors, exploratory_stationarity);
}

double VbScfObjective::strict_accepted_energy() const {
    const auto& point = *gradient_result_.second_order_context;
    if (structure_eigensolver_ == StructureEigensolver::Davidson &&
        point.structure_action && point.structure_metric_inverse)
        return refine_complete_ci_energy(point, nuclear_repulsion_);
    return evaluate_trial_energy(layout_.pack(input_.orbital_preparation_input)).energy;
}

void VbScfObjective::refine_trial_ci(TrialEvaluation& evaluation) const {
    const Eigen::VectorXd parameters = layout_.pack(evaluation.orbital_preparation_input);
    const Eigen::MatrixXd roots = evaluation.forward_evaluation->root_coefficients();
    evaluation.forward_evaluation.reset();
    auto strict = evaluate_trial_energy_with_initial_eigenvectors(parameters, false, roots);
    strict.chart_changed = evaluation.chart_changed;
    strict.active_inactive_orthogonal = evaluation.active_inactive_orthogonal;
    strict.wall_time_seconds += evaluation.wall_time_seconds;
    evaluation = std::move(strict);
}

void VbScfObjective::complete_trial(TrialEvaluation* evaluation) const {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::TrialGradient, true);
    if (evaluation == nullptr || !evaluation->valid || evaluation->gradient_ready ||
        !evaluation->forward_evaluation.has_value()) {
        throw std::invalid_argument("trial completion requires a valid energy-only evaluation");
    }
    const auto completion_start_time = std::chrono::steady_clock::now();
    ScopedTrialOrbitals trial_scope(&input_, evaluation->orbital_preparation_input);
    evaluation->gradient_result =
        gradient_evaluator_->complete_gradient(input_, std::move(*evaluation->forward_evaluation));
    evaluation->forward_evaluation.reset();
    if (evaluation->gradient_result.second_order_context == nullptr) {
        throw std::runtime_error(
            "relaxed orbital gradient did not populate the accepted-point second-order context");
    }
    evaluation->gradient =
        layout_.gather_from_full(evaluation->gradient_result.sparse_orbital_energy_gradient);
    evaluation->energy = evaluation->gradient_result.scf_result.total_energy;
    evaluation->gradient_inf_norm = gradient_infinity_norm(evaluation->gradient);
    evaluation->wall_time_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - completion_start_time)
            .count();
    evaluation->gradient_ready = true;
}

void VbScfObjective::commit(TrialEvaluation evaluation) {
    if (!evaluation.valid || !evaluation.gradient_ready) {
        throw std::invalid_argument("cannot commit a trial evaluation without an exact gradient");
    }
    input_.orbital_preparation_input = std::move(evaluation.orbital_preparation_input);
    active_inactive_orthogonal_ = evaluation.active_inactive_orthogonal;
    gradient_result_ = std::move(evaluation.gradient_result);
    energy_history_.push_back(evaluation.energy);
    gradient_inf_norm_history_.push_back(evaluation.gradient_inf_norm);
    iteration_time_history_seconds_.push_back(evaluation.wall_time_seconds);
}

} // namespace xmvb::vb
