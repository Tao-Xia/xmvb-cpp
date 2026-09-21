#include "vbscf/optimization/backends/neo.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/derivatives/hessian/coupled/structure.hpp"
#include "vbscf/optimization/driver/checks.hpp"
#include "vbscf/optimization/driver/options.hpp"
#include "vbscf/optimization/driver/result.hpp"
#include "vbscf/optimization/driver/session.hpp"
#include "vbscf/optimization/globalization/line_search.hpp"
#include "vbscf/optimization/neo/globalization.hpp"
#include "vbscf/optimization/neo/response_solver.hpp"
#include "vbscf/optimization/objective/function.hpp"
#include "vbscf/optimization/trust_region/retraction.hpp"
#include "vbscf/orbitals/charts/layout.hpp"

namespace xmvb::vb::optimizer_detail {
namespace {

double neo_forcing_term(
    double gradient_norm,
    Eigen::Index dimension) {
  // eta = ||g_k||/(1+||g_k||) is below one globally and is O(||g_k||)
  // locally, as required for an inexact Newton method to retain quadratic
  // convergence. Unlike normalization by the initial gradient, it remains
  // meaningful when an initial guess happens to lie near a saddle point.
  // The lower bound only prevents asking for accuracy below accumulated
  // floating-point roundoff.
  const double forcing = gradient_norm / (1.0 + gradient_norm);
  const double roundoff = std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max<Eigen::Index>(1, dimension));
  return std::max(roundoff, forcing);
}

}  // namespace

BackendRunResult run_neo_backend(
    VbScfObjective* objective,
    const SparseParameterLayout& parameter_view,
    const VbScfOptimizerOptions& options,
    const Eigen::VectorXd& initial_parameters,
    const Eigen::VectorXd& initial_gradient,
    double initial_energy,
    VbScfOptimizerResult* result) {
  BackendRunResult run_result;
  Eigen::VectorXd parameters = initial_parameters;
  Eigen::VectorXd gradient = initial_gradient;
  double energy = initial_energy;
  double previous_energy = initial_energy;
  double trust_radius =
      std::max(options.minimum_step_size, options.initial_step_size);
  Eigen::VectorXd curvature_probe_packed;

  OrbitalChart chart = build_orbital_chart(*objective, parameter_view);
  auto projected = chart.project_gradient(gradient);
  double final_gradient_inf =
      gradient_infinity_norm(projected.reduced_gradient);
  double final_gradient_l2 = projected.reduced_gradient.norm();

  while (run_result.n_iterations < options.max_iterations) {
    if (chart.reduced_size() == 0) {
      result->termination_reason = "neo_nonredundant_space_empty";
      final_gradient_inf = 0.0;
      final_gradient_l2 = 0.0;
      break;
    }
    final_gradient_inf = gradient_infinity_norm(projected.reduced_gradient);
    final_gradient_l2 = projected.reduced_gradient.norm();
    if (run_result.n_iterations == 0 &&
        final_gradient_inf < options.gradient_tolerance) {
      result->converged = true;
      result->termination_reason = "neo_initial_tolerance";
      break;
    }

    bool accepted = false;
    VbScfObjective::TrialEvaluation trial;
    Eigen::VectorXd trial_parameters;
    Eigen::VectorXd accepted_curvature_probe_packed;
    NeoIterationRecord iteration_record;
    iteration_record.accepted_iteration_index = run_result.n_iterations + 1;
    iteration_record.initial_trust_radius = trust_radius;
    {
      const OrbitalPreparationInput accepted_orbitals =
          objective->input().orbital_preparation_input;
      NonredundantRetractionMetric orbital_metric(
          chart, parameter_view, accepted_orbitals);
      ExactHvpOperator orbital_hessian(
          objective->second_order_context(),
          &objective->input(),
          parameter_view,
          &chart);
      if (!orbital_hessian.supports_analytic_core_model()) {
        result->termination_reason = "neo_analytic_hessian_unavailable";
        break;
      }
      StructureTangentOperator structure_hessian(
          objective->second_order_context(),
          orbital_hessian.structure_action());
      iteration_record.model_dimension =
          chart.reduced_size() + structure_hessian.tangent_size();
      Eigen::VectorXd curvature_probe;
      if (curvature_probe_packed.size() ==
          static_cast<Eigen::Index>(parameter_view.size())) {
        curvature_probe =
            chart.project_vector(curvature_probe_packed).reduced_gradient;
      }
      double selected_energy_scale = 0.0;
      for (const double selected_energy :
           objective->second_order_context()->selected_state_energies) {
        selected_energy_scale =
            std::max(selected_energy_scale, std::abs(selected_energy));
      }
      const double operator_relative_accuracy =
          objective->second_order_context()
              ->structure_solve_accuracy
              .response_backward_error_tolerance(selected_energy_scale);
      ResponseNeoProblem problem(
          projected.reduced_gradient,
          structure_hessian.tangent_size(),
          [&orbital_hessian, &structure_hessian](
              const Eigen::VectorXd& vector) {
            const OrbitalCouplingAction image =
                orbital_hessian.apply_orbital_coupling(vector);
            return ResponseNeoDirection{
                image.orbital_hessian,
                structure_hessian.coordinates(
                    StructureTangent{image.scaled_structure_forcing})};
          },
          [&orbital_hessian, &structure_hessian](
              const Eigen::VectorXd& vector) {
            const StructureCouplingAction image =
                structure_hessian.apply_coupling(
                    structure_hessian.expand(vector));
            return ResponseNeoDirection{
                orbital_hessian.apply_structure_coupling_adjoint(
                    image.coefficient_response,
                    image.adjoint_multipliers),
                structure_hessian.coordinates(image.hessian)};
          },
          [&orbital_metric](const Eigen::VectorXd& vector) {
            return orbital_metric.apply(vector);
          },
          [&chart](const Eigen::VectorXd& residual) {
            return chart.apply_inverse_reduced_block_preconditioner(residual);
          },
          std::move(curvature_probe),
          operator_relative_accuracy);

      while (!accepted) {
        const double trial_radius = trust_radius;
        NeoOptions neo_options;
        neo_options.trust_radius = trial_radius;
        neo_options.relative_residual_tolerance = neo_forcing_term(
            final_gradient_l2, chart.reduced_size());
        neo_options.maximum_subspace_dimension = 0;
        const ResponseNeoResult step = solve_response_neo(problem, neo_options);
        iteration_record.micro_iterations += step.iterations;
        iteration_record.coupled_block_actions += step.coupled_actions;
        if (!step.converged()) {
          result->termination_reason = "neo_microproblem_not_converged";
          break;
        }

        const Eigen::VectorXd candidate_parameters =
            build_nonredundant_lifted_trial_parameters(
                accepted_orbitals,
                chart,
                parameter_view,
                step.step.orbital);
        if (is_effectively_zero_step(
                candidate_parameters - parameters, parameters)) {
          result->termination_reason = "neo_zero_orbital_step";
          break;
        }

        trial = objective->evaluate_trial_energy(candidate_parameters, true);
        const double gradient_dot_step =
            problem.orbital_gradient().dot(step.step.orbital);
        const double step_dot_hessian_step =
            step.step.orbital.dot(step.hessian_step.orbital) +
            step.step.structure.dot(step.hessian_step.structure);
        const NeoGlobalizationResult globalization = globalize_neo_trial(
            energy,
            trial.energy,
            gradient_dot_step,
            step_dot_hessian_step,
            step.boundary,
            trial_radius,
            options.minimum_step_size);
        if (!globalization.accepted) {
          ++iteration_record.rejected_trial_count;
          if (!(globalization.next_radius < trust_radius)) {
            result->termination_reason = "neo_trust_radius_exhausted";
            break;
          }
          trust_radius = globalization.next_radius;
          continue;
        }

        iteration_record.kkt_residual_norm = step.residual_norm;
        iteration_record.kkt_residual_target = step.residual_target;
        iteration_record.curvature_residual_norm =
            step.curvature_residual_norm;
        iteration_record.curvature_residual_target =
            step.curvature_residual_target;
        iteration_record.global_curvature_certified =
            step.global_curvature_certified;
        iteration_record.accepted_trial_radius = trial_radius;
        iteration_record.next_trust_radius = globalization.next_radius;
        iteration_record.gradient_dot_step = gradient_dot_step;
        iteration_record.step_dot_hessian_step = step_dot_hessian_step;
        iteration_record.predicted_reduction =
            globalization.predicted_reduction;
        iteration_record.actual_reduction = globalization.actual_reduction;
        iteration_record.trust_ratio = globalization.rho;
        iteration_record.reached_boundary = step.boundary;
        trust_radius = globalization.next_radius;
        trial_parameters =
            parameter_view.pack(trial.orbital_preparation_input);
        accepted_curvature_probe_packed =
            chart.expand_step(step.minimum_curvature_orbital);
        accepted = true;
      }
    }
    if (!accepted) break;

    objective->complete_trial(&trial);
    trial_parameters = parameter_view.pack(trial.orbital_preparation_input);
    gradient = trial.gradient;
    energy = trial.energy;
    objective->commit(std::move(trial));
    parameters = std::move(trial_parameters);
    curvature_probe_packed = std::move(accepted_curvature_probe_packed);
    ++run_result.n_iterations;
    result->neo_iteration_trace.push_back(iteration_record);
    sync_result_from_objective(*objective, result);

    OrbitalChart next_chart = build_orbital_chart(*objective, parameter_view);
    auto next_projected = next_chart.project_gradient(gradient);
    final_gradient_inf =
        gradient_infinity_norm(next_projected.reduced_gradient);
    final_gradient_l2 = next_projected.reduced_gradient.norm();
    record_accepted_iteration_snapshot(
        objective,
        run_result.n_iterations,
        options,
        result,
        nullptr,
        &next_projected.reduced_gradient);

    const double energy_change = energy - previous_energy;
    previous_energy = energy;
    if (std::abs(energy_change) < options.energy_tolerance &&
        final_gradient_inf < options.gradient_tolerance) {
      result->converged = true;
      result->termination_reason = "neo_dual_tolerance";
      break;
    }
    chart = std::move(next_chart);
    projected = std::move(next_projected);
  }

  result->final_projected_gradient_inf_norm = final_gradient_inf;
  result->final_projected_gradient_l2_norm = final_gradient_l2;
  run_result.final_gradient_l2_norm = final_gradient_l2;
  run_result.final_projected_gradient_ready = true;
  return run_result;
}

}  // namespace xmvb::vb::optimizer_detail
