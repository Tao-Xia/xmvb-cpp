#include "vbscf/optimization/backends/neo.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
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
#include "vbscf/optimization/preconditioners/shifted_metric.hpp"
#include "vbscf/optimization/trust_region/retraction.hpp"
#include "vbscf/orbitals/charts/layout.hpp"

namespace xmvb::vb::optimizer_detail {
namespace {

OrbitalPreconditioner resolve_neo_preconditioner(
    const VbScfObjective& objective,
    OrbitalPreconditioner requested) {
  if (requested != OrbitalPreconditioner::Automatic) return requested;
  static_cast<void>(objective);
  return OrbitalPreconditioner::OneElectron;
}

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

struct AcceptedNeoKeyframe {
  VbScfObjective::TrialEvaluation trial;
  Eigen::VectorXd parameters;
  Eigen::VectorXd next_orbital_guess_packed;
};

bool build_accepted_neo_keyframe(
    VbScfObjective* objective,
    const SparseParameterLayout& parameter_view,
    const VbScfOptimizerOptions& options,
    const OrbitalChart& chart,
    const Eigen::VectorXd& reduced_gradient,
    const Eigen::VectorXd& current_parameters,
    const Eigen::VectorXd& orbital_guess_packed,
    double gradient_l2,
    double energy,
    double* trust_radius,
    NeoIterationRecord* record,
    VbScfOptimizerResult* result,
    AcceptedNeoKeyframe* accepted) {
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
    return false;
  }
  StructureTangentOperator structure_hessian(
      objective->second_order_context(), orbital_hessian.structure_action());
  const bool use_structure_response =
      structure_hessian.tangent_size() != 0;
  record->model_dimension = std::max(
      record->model_dimension,
      static_cast<int>(
          chart.reduced_size() +
          (use_structure_response ? structure_hessian.tangent_size() : 0)));
  Eigen::VectorXd orbital_guess;
  if (orbital_guess_packed.size() ==
      static_cast<Eigen::Index>(parameter_view.size())) {
    orbital_guess =
        chart.project_vector(orbital_guess_packed).reduced_gradient;
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
  std::function<ResponseNeoResult(const NeoOptions&)> solve_step;
  if (use_structure_response) {
    auto problem = std::make_shared<ResponseNeoProblem>(
        reduced_gradient,
        structure_hessian.tangent_size(),
        [&orbital_hessian, &structure_hessian](
            const Eigen::VectorXd& vector) {
          const OrbitalCouplingAction image =
              orbital_hessian.apply_orbital_coupling(vector);
          return ResponseNeoDirection{
              image.orbital_hessian,
              structure_hessian.coordinates(
                  structure_hessian.project(image.scaled_structure_forcing))};
        },
        [&orbital_hessian, &structure_hessian](
            const Eigen::VectorXd& vector) {
          const StructureCouplingAction image =
              structure_hessian.apply_coupling(
                  structure_hessian.expand(vector));
          return ResponseNeoDirection{
              orbital_hessian.apply_structure_coupling_adjoint(
                  image.coefficient_response, image.adjoint_multipliers),
              structure_hessian.coordinates(image.hessian)};
        },
        [&orbital_metric](const Eigen::VectorXd& vector) {
          return orbital_metric.apply(vector);
        },
        [&chart, &orbital_metric](
            const Eigen::VectorXd& residual, double shift) {
          return apply_inverse_shifted_metric_model(
              residual,
              shift,
              [&chart](const Eigen::VectorXd& vector) {
                return chart.apply_reduced_curvature(vector);
              },
              [&orbital_metric](const Eigen::VectorXd& vector) {
                return orbital_metric.apply(vector);
              },
              [&chart, shift](const Eigen::VectorXd& vector) {
                return chart.apply_inverse_reduced_shifted_block_preconditioner(
                    vector, shift);
              });
        },
        std::move(orbital_guess),
        operator_relative_accuracy,
        [&orbital_hessian, &structure_hessian, &chart](
            const Eigen::MatrixXd& vectors) {
          const StructureCoordinateCouplingBlock images =
              structure_hessian.apply_coupling_coordinate_block(vectors);
          ResponseNeoDirectionBlock result;
          result.orbital.resize(
              chart.reduced_size(), vectors.cols());
          result.structure = images.hessian_coordinates;
          for (Eigen::Index column = 0; column < vectors.cols(); ++column) {
            result.orbital.col(column) =
                orbital_hessian.apply_structure_coupling_adjoint(
                    images.coefficient_responses[column],
                    images.adjoint_multipliers[column]);
          }
          return result;
        });
    auto workspace = std::make_shared<ResponseNeoWorkspace>(*problem);
    solve_step = [problem = std::move(problem),
                  workspace = std::move(workspace)](
                     const NeoOptions& neo_options) {
      // The workspace stores a reference; retain its immutable problem here.
      static_cast<void>(problem);
      return workspace->solve(neo_options);
    };
  } else {
    HvpComponents core_components;
    core_components.local_active_response = false;
    core_components.structure_response = false;
    auto problem = std::make_shared<NeoProblem>(
        reduced_gradient,
        [&orbital_hessian, core_components](
            const Eigen::VectorXd& vector) {
          return orbital_hessian.apply_reduced(vector, core_components);
        },
        [&orbital_metric](const Eigen::VectorXd& vector) {
          return orbital_metric.apply(vector);
        },
        [&chart](const Eigen::VectorXd& residual) {
          return chart.apply_inverse_reduced_block_preconditioner(residual);
        });
    solve_step = [problem = std::move(problem)](
                     const NeoOptions& neo_options) {
      const NeoResult core = solve_neo(*problem, neo_options);
      ResponseNeoResult result;
      result.step = {core.step, Eigen::VectorXd{}};
      result.hessian_step = {core.hessian_step, Eigen::VectorXd{}};
      result.orbital_metric_step = core.metric_step;
      result.kkt_residual = {core.kkt_residual, Eigen::VectorXd{}};
      result.minimum_curvature_orbital =
          Eigen::VectorXd::Zero(problem->size());
      result.shift = core.shift;
      result.predicted_reduction = core.predicted_reduction;
      result.step_norm = core.step_norm;
      result.residual_norm = core.residual_norm;
      result.residual_target = core.residual_target;
      result.curvature_residual_norm = core.curvature_residual_norm;
      result.curvature_residual_target = core.curvature_residual_target;
      result.iterations = core.iterations;
      result.coupled_actions = core.hessian_actions;
      result.orbital_actions = core.hessian_actions;
      result.boundary = core.boundary;
      result.hard_case = core.hard_case;
      result.global_curvature_certified =
          core.global_curvature_certified;
      result.stop_reason = core.stop_reason;
      return result;
    };
  }

  while (true) {
    const double trial_radius = *trust_radius;
    NeoOptions neo_options;
    neo_options.trust_radius = trial_radius;
    neo_options.relative_residual_tolerance = neo_forcing_term(
        gradient_l2, chart.reduced_size());
    // A linear solve below the requested nonlinear stationarity is unusable
    // accuracy, especially for the final energy-confirmation keyframe.
    neo_options.absolute_residual_tolerance = options.gradient_tolerance;
    neo_options.maximum_subspace_dimension = 0;
    neo_options.require_curvature_certificate = false;
    const ResponseNeoResult step = solve_step(neo_options);
    record->micro_iterations += step.iterations;
    record->coupled_block_actions += step.coupled_actions;
    record->orbital_hvp_actions += step.orbital_actions;
    record->structure_response_actions += step.structure_actions;
    if (!step.converged()) {
      result->termination_reason = "neo_microproblem_not_converged";
      return false;
    }

    const Eigen::VectorXd candidate_parameters =
        build_nonredundant_lifted_trial_parameters(
            accepted_orbitals,
            chart,
            parameter_view,
            step.step.orbital);
    if (is_effectively_zero_step(
            candidate_parameters - current_parameters,
            current_parameters)) {
      result->termination_reason = "neo_zero_orbital_step";
      return false;
    }

    auto trial = objective->evaluate_trial_energy(candidate_parameters, true);
    const double gradient_dot_step =
        reduced_gradient.dot(step.step.orbital);
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
      ++record->rejected_trial_count;
      if (!(globalization.next_radius < *trust_radius)) {
        result->termination_reason = "neo_trust_radius_exhausted";
        return false;
      }
      *trust_radius = globalization.next_radius;
      continue;
    }

    record->kkt_residual_norm = step.residual_norm;
    record->kkt_residual_target = step.residual_target;
    record->curvature_residual_norm = step.curvature_residual_norm;
    record->curvature_residual_target = step.curvature_residual_target;
    record->global_curvature_certified = step.global_curvature_certified;
    record->accepted_trial_radius = trial_radius;
    record->next_trust_radius = globalization.next_radius;
    record->gradient_dot_step += gradient_dot_step;
    record->step_dot_hessian_step += step_dot_hessian_step;
    record->predicted_reduction += globalization.predicted_reduction;
    record->actual_reduction += globalization.actual_reduction;
    record->reached_boundary = record->reached_boundary || step.boundary;
    *trust_radius = globalization.next_radius;

    accepted->parameters =
        parameter_view.pack(trial.orbital_preparation_input);
    accepted->next_orbital_guess_packed =
        chart.expand_step(step.step.orbital);
    accepted->trial = std::move(trial);
    return true;
  }
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
  const OrbitalPreconditioner preconditioner =
      resolve_neo_preconditioner(*objective, options.orbital_preconditioner);
  OrbitalChart chart = build_orbital_chart(
      *objective, parameter_view, OrbitalPreconditioner::Identity);
  auto projected = chart.project_gradient(gradient);
  double final_gradient_inf =
      gradient_infinity_norm(projected.reduced_gradient);
  double final_gradient_l2 = projected.reduced_gradient.norm();
  if (final_gradient_inf >= options.gradient_tolerance) {
    chart = build_orbital_chart(
        *objective, parameter_view, preconditioner);
  }

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

    NeoIterationRecord iteration_record;
    iteration_record.accepted_iteration_index = run_result.n_iterations + 1;
    iteration_record.initial_trust_radius = trust_radius;
    const double macro_gradient_target = neo_forcing_term(
        final_gradient_l2, chart.reduced_size()) * final_gradient_l2;
    bool macro_accepted = false;
    bool backend_failed = false;
    Eigen::VectorXd keyframe_orbital_guess_packed;

    // A macro step may contain several exact-gradient keyframes.  Each
    // keyframe discards the old quadratic model and rebuilds H, B, and C at
    // the newly accepted nonlinear point.  The macro ends once the exact
    // projected gradient satisfies the same Eisenstat--Walker forcing target
    // used by the matrix-free Newton equation.
    while (!result->converged && final_gradient_l2 > macro_gradient_target) {
      AcceptedNeoKeyframe accepted;
      if (!build_accepted_neo_keyframe(
              objective,
              parameter_view,
              options,
              chart,
              projected.reduced_gradient,
              parameters,
              keyframe_orbital_guess_packed,
              final_gradient_l2,
              energy,
              &trust_radius,
              &iteration_record,
              result,
              &accepted)) {
        backend_failed = true;
        break;
      }

      objective->complete_trial(&accepted.trial);
      accepted.parameters =
          parameter_view.pack(accepted.trial.orbital_preparation_input);
      gradient = accepted.trial.gradient;
      energy = accepted.trial.energy;
      objective->commit(std::move(accepted.trial));
      parameters = std::move(accepted.parameters);
      keyframe_orbital_guess_packed =
          std::move(accepted.next_orbital_guess_packed);
      ++iteration_record.keyframes;
      macro_accepted = true;

      // Stationarity depends only on the quotient geometry. Do not build an
      // integral-dependent preconditioner until another NEO solve is needed.
      OrbitalChart next_geometry = build_orbital_chart(
          *objective, parameter_view, OrbitalPreconditioner::Identity);
      auto next_projected = next_geometry.project_gradient(gradient);
      final_gradient_inf =
          gradient_infinity_norm(next_projected.reduced_gradient);
      final_gradient_l2 = next_projected.reduced_gradient.norm();
      const double energy_change = energy - previous_energy;
      previous_energy = energy;
      if (std::abs(energy_change) < options.energy_tolerance &&
          final_gradient_inf < options.gradient_tolerance) {
        result->converged = true;
        result->termination_reason = "neo_projected_gradient_tolerance";
      }
      // Once external first-order stationarity is reached, expose this
      // keyframe as the end of the macro step.  If its energy change is still
      // too large, the next macro supplies the required independent energy
      // confirmation instead of hiding it inside the current reported step.
      if (!result->converged &&
          final_gradient_inf < options.gradient_tolerance) {
        chart = build_orbital_chart(
            *objective, parameter_view, preconditioner);
        projected = std::move(next_projected);
        break;
      }
      if (!result->converged) {
        OrbitalChart next_chart =
            build_orbital_chart(
                *objective, parameter_view, preconditioner);
        if (next_chart.reduced_size() != next_geometry.reduced_size() ||
            next_chart.rank_signature() != next_geometry.rank_signature()) {
          throw std::runtime_error(
              "preconditioner changed the nonredundant orbital geometry");
        }
        chart = std::move(next_chart);
      } else {
        chart = std::move(next_geometry);
      }
      projected = std::move(next_projected);
    }

    if (macro_accepted) {
      ++run_result.n_iterations;
      iteration_record.trust_ratio =
          iteration_record.predicted_reduction > 0.0
              ? iteration_record.actual_reduction /
                    iteration_record.predicted_reduction
              : -std::numeric_limits<double>::infinity();
      result->neo_iteration_trace.push_back(iteration_record);
      sync_result_from_objective(*objective, result);
      record_accepted_iteration_snapshot(
          objective,
          run_result.n_iterations,
          options,
          result,
          nullptr,
          &projected.reduced_gradient);
    }
    if (result->converged || backend_failed || !macro_accepted) break;
  }

  result->final_projected_gradient_inf_norm = final_gradient_inf;
  result->final_projected_gradient_l2_norm = final_gradient_l2;
  run_result.final_gradient_l2_norm = final_gradient_l2;
  run_result.final_projected_gradient_ready = true;
  return run_result;
}

}  // namespace xmvb::vb::optimizer_detail
