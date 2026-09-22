#include "vbscf/optimization/backends/neo.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

#include "vbscf/derivatives/hessian/coupled/operator.hpp"
#include "vbscf/derivatives/hessian/coupled/structure.hpp"
#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/optimization/driver/checks.hpp"
#include "vbscf/optimization/driver/options.hpp"
#include "vbscf/optimization/driver/result.hpp"
#include "vbscf/optimization/driver/session.hpp"
#include "vbscf/optimization/globalization/line_search.hpp"
#include "vbscf/optimization/neo/augmented_hessian.hpp"
#include "vbscf/optimization/neo/coordinates.hpp"
#include "vbscf/optimization/neo/globalization.hpp"
#include "vbscf/optimization/neo/problem.hpp"
#include "vbscf/optimization/objective/function.hpp"
#include "vbscf/optimization/trust_region/retraction.hpp"
#include "vbscf/orbitals/charts/layout.hpp"

namespace xmvb::vb::optimizer_detail {
namespace {

OrbitalPreconditioner resolve_neo_preconditioner(
    const VbScfObjective& objective,
    OrbitalPreconditioner requested) {
  if (requested != OrbitalPreconditioner::Automatic) return requested;
  static_cast<void>(objective);
  return OrbitalPreconditioner::HessianDiagonal;
}

struct AcceptedNeoKeyframe {
  VbScfObjective::TrialEvaluation trial;
  Eigen::VectorXd parameters;
  Eigen::VectorXd next_orbital_guess_packed;
  Eigen::MatrixXd next_structure_guess;
};

struct NeoCarryGuess {
  Eigen::VectorXd orbital_packed;
  Eigen::MatrixXd structure_response;
};

constexpr double kAhGradientTrust = 3.0;
constexpr double kMaximumAhCoordinateStep = 0.03;
constexpr double kAhLevelShift = 1.0e-8;
constexpr int kBaseMicroIterations = 10;
constexpr int kKeyframeInterval = 5;

struct AcceptedNeoModelScalars {
  double trial_radius = 0.0;
  double gradient_dot_step = 0.0;
  double step_dot_hessian_step = 0.0;
  NeoGlobalizationResult globalization;
  double kkt_residual_norm = 0.0;
  double kkt_residual_target = 0.0;
  double curvature_residual_norm = 0.0;
  double curvature_residual_target = 0.0;
  bool global_curvature_certified = false;
  bool reached_boundary = false;
};

void accept_neo_keyframe_record(
    const AcceptedNeoModelScalars& model,
    NeoIterationRecord* record) {
  record->kkt_residual_norm = model.kkt_residual_norm;
  record->kkt_residual_target = model.kkt_residual_target;
  record->curvature_residual_norm = model.curvature_residual_norm;
  record->curvature_residual_target = model.curvature_residual_target;
  record->global_curvature_certified = model.global_curvature_certified;
  record->accepted_trial_radius = model.trial_radius;
  record->next_trust_radius = model.globalization.next_radius;
  record->gradient_dot_step += model.gradient_dot_step;
  record->step_dot_hessian_step += model.step_dot_hessian_step;
  record->predicted_reduction += model.globalization.predicted_reduction;
  record->actual_reduction += model.globalization.actual_reduction;
  record->reached_boundary =
      record->reached_boundary || model.reached_boundary;
}

bool build_accepted_neo_keyframe(
    VbScfObjective* objective,
    const SparseParameterLayout& parameter_view,
    const VbScfOptimizerOptions& options,
    const OrbitalChart& chart,
    const Eigen::VectorXd& reduced_gradient,
    const Eigen::VectorXd& current_parameters,
    const NeoCarryGuess& carry_guess,
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
  record->model_dimension = std::max(
      record->model_dimension,
      static_cast<int>(
          chart.reduced_size() + structure_hessian.tangent_size()));
  Eigen::VectorXd orbital_guess;
  if (carry_guess.orbital_packed.size() ==
      static_cast<Eigen::Index>(parameter_view.size())) {
    orbital_guess =
        chart.project_vector(carry_guess.orbital_packed).reduced_gradient;
  }
  CoupledHessianOperator coupled_hessian(orbital_hessian, structure_hessian);
  CoupledNeoCoordinates coordinates(chart, orbital_metric, structure_hessian);
  Eigen::VectorXd joint_gradient = Eigen::VectorXd::Zero(coordinates.size());
  joint_gradient.head(coordinates.orbital_size()) = reduced_gradient;
  Eigen::VectorXd initial_guess;
  const bool has_orbital_guess =
      orbital_guess.size() == coordinates.orbital_size();
  const bool has_structure_guess =
      carry_guess.structure_response.rows() == structure_hessian.n_structures() &&
      carry_guess.structure_response.cols() == structure_hessian.n_states() &&
      carry_guess.structure_response.allFinite();
  if (has_orbital_guess || has_structure_guess) {
    initial_guess = Eigen::VectorXd::Zero(coordinates.size());
    if (has_orbital_guess) {
      initial_guess.head(coordinates.orbital_size()) = orbital_guess;
    }
    if (has_structure_guess) {
      initial_guess.tail(coordinates.structure_size()) =
          structure_hessian.coordinates(
              structure_hessian.from_coefficient_response(
                  carry_guess.structure_response));
    }
  }
  NeoProblem problem(
      std::move(joint_gradient),
      [&coupled_hessian, &coordinates](const Eigen::VectorXd& vector) {
        return coordinates.flatten(
            coupled_hessian.apply(coordinates.unflatten(vector)));
      },
      [&coordinates](const Eigen::VectorXd& vector) {
        return coordinates.apply_metric(vector);
      },
      [&chart, &structure_hessian, &coordinates](
          const Eigen::VectorXd& covector, double eigenvalue) {
        CoupledDirection blocks = coordinates.unflatten(covector);
        const double shifted_eigenvalue = eigenvalue - kAhLevelShift;
        blocks.orbital = chart.apply_inverse_augmented_hessian_diagonal(
            blocks.orbital, shifted_eigenvalue);
        const Eigen::VectorXd structure_coordinates =
            structure_hessian.coordinates(blocks.structure);
        blocks.structure = structure_hessian.expand(
            structure_hessian.apply_inverse_augmented_hessian_diagonal(
                structure_coordinates, shifted_eigenvalue));
        return coordinates.flatten(blocks);
      },
      std::nullopt,
      std::move(initial_guess));
  AugmentedHessianWorkspace workspace(problem);
  const auto joint_gradient_norm =
      [&coordinates, &structure_hessian](const Eigen::VectorXd& covector) {
        const CoupledDirection blocks = coordinates.unflatten(covector);
        return std::hypot(
            blocks.orbital.stableNorm(),
            structure_hessian.diagonal_dual_norm(
                structure_hessian.coordinates(blocks.structure)));
      };
  OrbitalPreparationInput keyframe_orbitals = accepted_orbitals;
  Eigen::MatrixXd keyframe_structure_coefficients =
      objective->second_order_context()->selected_state_eigenvectors;
  Eigen::VectorXd keyframe_step = Eigen::VectorXd::Zero(coordinates.size());
  Eigen::VectorXd keyframe_hessian_step =
      Eigen::VectorXd::Zero(coordinates.size());
  Eigen::VectorXd model_gradient = problem.gradient();
  double completed_gradient_dot_step = 0.0;
  double completed_step_dot_hessian_step = 0.0;
  const double keyframe_convergence_tolerance =
      0.3 * options.gradient_tolerance;
  const double micro_convergence_tolerance =
      0.3 * keyframe_convergence_tolerance;
  double keyframe_gradient_norm = joint_gradient_norm(problem.gradient());
  int iterations_since_keyframe = 0;
  Eigen::VectorXd last_increment;
  Eigen::VectorXd last_hessian_increment;
  AugmentedHessianStep last_ah_step;
  bool reached_boundary = false;
  while (true) {
    AugmentedHessianOptions ah_options;
    last_ah_step =
        workspace.next(model_gradient, ah_options);
    ++record->micro_iterations;
    record->coupled_block_actions += last_ah_step.new_hessian_actions;
    record->orbital_hvp_actions += last_ah_step.new_hessian_actions;
    record->structure_response_actions += last_ah_step.new_hessian_actions;

    last_increment = last_ah_step.step;
    last_hessian_increment = last_ah_step.hessian_step;
    const CoupledDirection increment_blocks =
        coordinates.unflatten(last_increment);
    double increment_max =
        chart.maximum_rotation_component(increment_blocks.orbital);
    increment_max = std::max(
        increment_max,
        structure_hessian.maximum_coefficient_component(
            increment_blocks.structure));
    if (increment_max > kMaximumAhCoordinateStep) {
      const double scale = kMaximumAhCoordinateStep / increment_max;
      last_increment *= scale;
      last_hessian_increment *= scale;
      reached_boundary = true;
    }
    keyframe_step += last_increment;
    keyframe_hessian_step += last_hessian_increment;
    model_gradient += last_hessian_increment;
    ++iterations_since_keyframe;
    const double model_gradient_norm = joint_gradient_norm(model_gradient);
    const double keyframe_step_norm =
        std::sqrt(coordinates.squared_norm(keyframe_step));
    if (record->micro_iterations > 3 &&
        model_gradient_norm > kAhGradientTrust * keyframe_gradient_norm) {
      keyframe_step -= last_increment;
      keyframe_hessian_step -= last_hessian_increment;
      model_gradient -= last_hessian_increment;
      break;
    }
    const int maximum_microsteps = std::max(
        kBaseMicroIterations,
        kBaseMicroIterations - static_cast<int>(
            2.0 * std::log(keyframe_gradient_norm + 1.0e-7)));
    if (record->micro_iterations >= maximum_microsteps ||
        model_gradient_norm < micro_convergence_tolerance) {
      break;
    }

    const double keyframe_threshold = std::max(
        static_cast<double>(kKeyframeInterval),
        static_cast<double>(kKeyframeInterval) -
            std::log(keyframe_step_norm + 1.0e-7));
    const bool keyframe_due =
        iterations_since_keyframe >= keyframe_threshold ||
        model_gradient_norm < keyframe_gradient_norm / kAhGradientTrust;
    if (keyframe_due) {
      const CoupledDirection keyframe_increment =
          coordinates.unflatten(keyframe_step);
      const Eigen::VectorXd keyframe_parameters =
          build_nonredundant_lifted_trial_parameters(
              keyframe_orbitals,
              chart,
              parameter_view,
              keyframe_increment.orbital);
      const Eigen::MatrixXd keyframe_coefficients =
          keyframe_structure_coefficients +
          structure_hessian.coefficient_response(keyframe_increment.structure);
      // CIAH keyframes retain both approximate components of the joint step.
      // Rediagonalizing H/S here would erase the structure microiterations and
      // turn the coupled method into a different, repeatedly relaxed method.
      auto keyframe = objective->evaluate_coupled_keyframe(
          keyframe_parameters, keyframe_coefficients);
      Eigen::VectorXd exact_gradient =
          Eigen::VectorXd::Zero(coordinates.size());
      exact_gradient.head(coordinates.orbital_size()) =
          chart.project_gradient(keyframe.orbital_gradient).reduced_gradient;
      Eigen::MatrixXd scaled_structure_residuals =
          std::move(keyframe.structure_residuals);
      const auto& state_weights =
          objective->second_order_context()->normalized_state_weights;
      for (int state = 0; state < scaled_structure_residuals.cols(); ++state) {
        scaled_structure_residuals.col(state) *= std::sqrt(
            2.0 * state_weights[static_cast<std::size_t>(state)]);
      }
      exact_gradient.tail(coordinates.structure_size()) =
          structure_hessian.project_coordinates(
              scaled_structure_residuals);
      const double exact_gradient_norm = joint_gradient_norm(exact_gradient);
      const double correction_norm =
          joint_gradient_norm(exact_gradient - model_gradient);
      ++record->keyframes;
      iterations_since_keyframe = 0;
      if (correction_norm < kAhGradientTrust * model_gradient_norm ||
          exact_gradient_norm <
              kAhGradientTrust * keyframe_convergence_tolerance) {
        completed_gradient_dot_step +=
            model_gradient.dot(keyframe_step) -
            keyframe_step.dot(keyframe_hessian_step);
        completed_step_dot_hessian_step +=
            keyframe_step.dot(keyframe_hessian_step);
        parameter_view.unpack(keyframe_parameters, &keyframe_orbitals);
        keyframe_structure_coefficients =
            std::move(keyframe.normalized_structure_coefficients);
        keyframe_step.setZero();
        keyframe_hessian_step.setZero();
        model_gradient = std::move(exact_gradient);
        keyframe_gradient_norm = exact_gradient_norm;
      } else {
        keyframe_step -= last_increment;
        keyframe_hessian_step -= last_hessian_increment;
        model_gradient -= last_hessian_increment;
        ++record->rejected_trial_count;
        break;
      }
    }
  }

  CoupledDirection joint_step = coordinates.unflatten(keyframe_step);
  Eigen::VectorXd candidate_parameters =
      build_nonredundant_lifted_trial_parameters(
          keyframe_orbitals, chart, parameter_view, joint_step.orbital);
  if (is_effectively_zero_step(
          candidate_parameters - current_parameters, current_parameters)) {
    result->termination_reason = "neo_zero_orbital_step";
    return false;
  }
  const Eigen::MatrixXd candidate_structure_coefficients =
      keyframe_structure_coefficients +
      structure_hessian.coefficient_response(joint_step.structure);
  auto trial = objective->evaluate_trial_with_structure_guess(
      candidate_parameters, candidate_structure_coefficients);

  const double gradient_dot_step =
      completed_gradient_dot_step + model_gradient.dot(keyframe_step) -
      keyframe_step.dot(keyframe_hessian_step);
  const double step_dot_hessian_step =
      completed_step_dot_hessian_step +
      keyframe_step.dot(keyframe_hessian_step);
  NeoGlobalizationResult globalization = globalize_neo_trial(
      energy,
      trial.energy,
      gradient_dot_step,
      step_dot_hessian_step,
      reached_boundary,
      kMaximumAhCoordinateStep,
      options.minimum_step_size);
  globalization.accepted = true;
  globalization.next_radius = kMaximumAhCoordinateStep;
  *trust_radius = globalization.next_radius;

  accepted->parameters =
      parameter_view.pack(trial.orbital_preparation_input);
  accepted->next_orbital_guess_packed =
      chart.expand_step(
          coordinates.unflatten(last_increment).orbital);
  accepted->next_structure_guess =
      structure_hessian.coefficient_response(
          coordinates.unflatten(last_increment).structure);
  accepted->trial = std::move(trial);
  accept_neo_keyframe_record(
      AcceptedNeoModelScalars{
          kMaximumAhCoordinateStep,
          gradient_dot_step,
          step_dot_hessian_step,
          globalization,
          last_ah_step.residual_norm,
          1.0e-6,
          0.0,
          0.0,
          false,
          reached_boundary},
      record);
  return true;
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
  NeoCarryGuess carry_guess;
  double energy = initial_energy;
  double previous_energy = initial_energy;
  double trust_radius =
      std::max(options.minimum_step_size, options.initial_step_size);
  const OrbitalPreconditioner preconditioner =
      resolve_neo_preconditioner(*objective, options.orbital_preconditioner);
  OrbitalChart chart = build_orbital_chart(
      *objective, parameter_view, OrbitalPreconditioner::Identity);
  auto projected = chart.project_gradient(gradient);
  NonredundantRetractionMetric initial_metric(
      chart, parameter_view, objective->input().orbital_preparation_input);
  double final_gradient_inf =
      gradient_infinity_norm(projected.reduced_gradient);
  double final_gradient_l2 = projected.reduced_gradient.norm();
  if (initial_metric.dual_norm(projected.reduced_gradient) >=
      options.gradient_tolerance) {
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
    const NonredundantRetractionMetric metric(
        chart,
        parameter_view,
        objective->input().orbital_preparation_input);
    double physical_gradient_norm =
        metric.dual_norm(projected.reduced_gradient);
    if (run_result.n_iterations == 0 &&
        physical_gradient_norm < options.gradient_tolerance) {
      result->converged = true;
      result->termination_reason = "neo_initial_tolerance";
      break;
    }

    NeoIterationRecord iteration_record;
    iteration_record.accepted_iteration_index = run_result.n_iterations + 1;
    iteration_record.initial_trust_radius = trust_radius;
    bool macro_accepted = false;
    bool backend_failed = false;
    AcceptedNeoKeyframe accepted;
    if (!build_accepted_neo_keyframe(
            objective,
            parameter_view,
            options,
            chart,
            projected.reduced_gradient,
            parameters,
            carry_guess,
            energy,
            &trust_radius,
            &iteration_record,
            result,
            &accepted)) {
      backend_failed = true;
    } else {
      accepted.parameters =
          parameter_view.pack(accepted.trial.orbital_preparation_input);
      gradient = accepted.trial.gradient;
      energy = accepted.trial.energy;
      objective->commit(std::move(accepted.trial));
      parameters = std::move(accepted.parameters);
      // CIAH reuses the last accepted microstep as the next macro's seed.
      // A gauge change alters its coordinates, not the physical tangent: the
      // next chart projects this packed tangent into its own quotient basis.
      carry_guess.orbital_packed =
          std::move(accepted.next_orbital_guess_packed);
      carry_guess.structure_response =
          std::move(accepted.next_structure_guess);
      ++iteration_record.keyframes;
      macro_accepted = true;

      OrbitalChart next_geometry = build_orbital_chart(
          *objective, parameter_view, OrbitalPreconditioner::Identity);
      auto next_projected = next_geometry.project_gradient(gradient);
      const NonredundantRetractionMetric next_metric(
          next_geometry,
          parameter_view,
          objective->input().orbital_preparation_input);
      final_gradient_inf =
          gradient_infinity_norm(next_projected.reduced_gradient);
      final_gradient_l2 = next_projected.reduced_gradient.norm();
      physical_gradient_norm =
          next_metric.dual_norm(next_projected.reduced_gradient);
      const double energy_change = energy - previous_energy;
      previous_energy = energy;
      if (std::abs(energy_change) < options.energy_tolerance &&
          physical_gradient_norm < options.gradient_tolerance) {
        result->converged = true;
        result->termination_reason = "neo_dual_tolerance";
      }
      chart = result->converged
          ? std::move(next_geometry)
          : build_orbital_chart(*objective, parameter_view, preconditioner);
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
