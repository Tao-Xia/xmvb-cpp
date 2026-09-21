#include "vbscf/optimization/backends/neo.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
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
#include "vbscf/optimization/neo/keyframe_policy.hpp"
#include "vbscf/optimization/neo/response_solver.hpp"
#include "vbscf/optimization/objective/function.hpp"
#include "vbscf/optimization/preconditioners/structure_response_woodbury.hpp"
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

/**
 * @brief Shift-local inverse of the recycled relaxed orbital model.
 *
 * The response revision identifies @f$J@f$ and @f$K@f$; the NEO shift
 * identifies the base inverse @f$P_\lambda@f$.  The small Woodbury factor is
 * rebuilt only when either quantity changes.
 */
class NeoWoodburyOrbitalPreconditioner {
public:
  NeoWoodburyOrbitalPreconditioner(
      const ExactHvpOperator& orbital_hessian,
      const OrbitalChart& chart)
      : orbital_hessian_(orbital_hessian), chart_(chart) {}

  Eigen::VectorXd apply(
      const Eigen::VectorXd& covector,
      double shift) const {
    const std::uint64_t revision =
        orbital_hessian_.response_model_revision();
    if (revision != revision_ || shift != shift_) rebuild(revision, shift);
    if (!woodbury_) {
      return chart_.apply_inverse_reduced_shifted_block_preconditioner(
          covector, shift);
    }
    return woodbury_->apply(covector);
  }

private:
  void rebuild(std::uint64_t revision, double shift) const {
    const StructureResponseSchurModel model =
        orbital_hessian_.structure_response_schur_model();
    revision_ = revision;
    shift_ = shift;
    woodbury_.reset();
    if (model.orbital_couplings.cols() == 0) return;
    woodbury_ = std::make_unique<
        StructureResponseWoodburyPreconditioner>(
        model.orbital_couplings,
        model.projected_operator,
        [this, shift](const Eigen::MatrixXd& covectors) {
          Eigen::MatrixXd result(covectors.rows(), covectors.cols());
          for (Eigen::Index column = 0; column < covectors.cols(); ++column) {
            result.col(column) =
                chart_.apply_inverse_reduced_shifted_block_preconditioner(
                    covectors.col(column), shift);
          }
          return result;
        });
    if (!woodbury_->available()) {
      throw std::runtime_error(
          "NEO Woodbury orbital preconditioner is unavailable: " +
          woodbury_->unavailability_reason());
    }
  }

  const ExactHvpOperator& orbital_hessian_;
  const OrbitalChart& chart_;
  mutable std::uint64_t revision_ =
      std::numeric_limits<std::uint64_t>::max();
  mutable double shift_ = std::numeric_limits<double>::quiet_NaN();
  mutable std::unique_ptr<StructureResponseWoodburyPreconditioner> woodbury_;
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
  record->model_dimension = std::max(
      record->model_dimension,
      static_cast<int>(
          chart.reduced_size() + structure_hessian.tangent_size()));
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
  NeoWoodburyOrbitalPreconditioner orbital_preconditioner(
      orbital_hessian, chart);
  ResponseNeoProblem problem(
      reduced_gradient,
      structure_hessian.tangent_size(),
      [&orbital_hessian, &structure_hessian](
          const Eigen::VectorXd& vector) {
        const OrbitalCouplingAction image =
            orbital_hessian.apply_orbital_coupling(vector);
        return ResponseNeoDirection{
            image.orbital_hessian,
            structure_hessian.project_coordinates(
                image.scaled_structure_forcing)};
      },
      [&orbital_hessian, &structure_hessian, &reduced_gradient](
          const Eigen::Ref<const Eigen::MatrixXd>& forcing,
          double relative_tolerance) {
        std::vector<Eigen::MatrixXd> ambient_forcing;
        ambient_forcing.reserve(static_cast<std::size_t>(forcing.cols()));
        for (Eigen::Index column = 0; column < forcing.cols(); ++column) {
          ambient_forcing.push_back(
              structure_hessian.expand(forcing.col(column))
                  .scaled_coefficients);
        }
        const StructureResponseBlock response =
            orbital_hessian.solve_structure_response_block(
                ambient_forcing, relative_tolerance);
        Eigen::MatrixXd coordinates(
            structure_hessian.tangent_size(), forcing.cols());
        Eigen::MatrixXd orbital_images(
            reduced_gradient.size(), forcing.cols());
        Eigen::MatrixXd equation_residuals(
            structure_hessian.tangent_size(), forcing.cols());
        for (Eigen::Index column = 0; column < forcing.cols(); ++column) {
          coordinates.col(column) = structure_hessian.project_coordinates(
              response.scaled_coefficients[static_cast<std::size_t>(column)]);
          orbital_images.col(column) =
              orbital_hessian.apply_structure_coupling_adjoint(
                  response.coefficient_responses[
                      static_cast<std::size_t>(column)],
                  response.adjoint_multipliers[
                      static_cast<std::size_t>(column)]);
          equation_residuals.col(column) =
              structure_hessian.project_coordinates(
                  response.scaled_equation_residuals[
                      static_cast<std::size_t>(column)]);
        }
        return ResponseNeoStructureResponse{
            std::move(coordinates), std::move(orbital_images),
            std::move(equation_residuals), response.revision,
            response.block_actions,
            response.max_relative_residual};
      },
      [&orbital_metric](const Eigen::VectorXd& vector) {
        return orbital_metric.apply(vector);
      },
      [&orbital_preconditioner](
          const Eigen::VectorXd& residual, double shift) {
        return orbital_preconditioner.apply(residual, shift);
      },
      std::move(orbital_guess),
      operator_relative_accuracy);
  ResponseNeoWorkspace workspace(problem);
  NeoKeyframePolicy keyframe_policy;
  std::optional<AcceptedNeoKeyframe> last_accepted;
  std::optional<AcceptedNeoModelScalars> last_accepted_model;

  const auto finish_last_accepted = [&]() {
    if (!last_accepted.has_value() || !last_accepted_model.has_value()) {
      return false;
    }
    accept_neo_keyframe_record(*last_accepted_model, record);
    *accepted = std::move(*last_accepted);
    return true;
  };

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
    const ResponseNeoResult step = workspace.solve(neo_options);
    record->micro_iterations += step.iterations;
    record->coupled_block_actions += step.coupled_actions;
    record->orbital_hvp_actions += step.orbital_actions;
    record->structure_response_actions += step.structure_actions;
    if (!step.converged()) {
      if (finish_last_accepted()) return true;
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
      if (finish_last_accepted()) return true;
      result->termination_reason = "neo_zero_orbital_step";
      return false;
    }

    auto trial = objective->evaluate_trial_energy(candidate_parameters, true);
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
      ++record->rejected_trial_count;
      const bool radius_contracts =
          globalization.next_radius < *trust_radius;
      *trust_radius = globalization.next_radius;
      // A farther frozen-model candidate may fail even though the previous
      // radius produced a valid keyframe. Retain that last energy-accepted
      // point and refresh its exact gradient instead of discarding it.
      if (last_accepted_model.has_value()) {
        last_accepted_model->globalization.next_radius = *trust_radius;
      }
      if (finish_last_accepted()) return true;
      if (!radius_contracts) {
        result->termination_reason = "neo_trust_radius_exhausted";
        return false;
      }
      continue;
    }

    *trust_radius = globalization.next_radius;

    AcceptedNeoKeyframe candidate;
    candidate.parameters =
        parameter_view.pack(trial.orbital_preparation_input);
    candidate.next_orbital_guess_packed =
        chart.expand_step(step.step.orbital);
    candidate.trial = std::move(trial);
    last_accepted = std::move(candidate);
    last_accepted_model = AcceptedNeoModelScalars{
        trial_radius,
        gradient_dot_step,
        step_dot_hessian_step,
        globalization,
        step.residual_norm,
        step.residual_target,
        step.curvature_residual_norm,
        step.curvature_residual_target,
        step.global_curvature_certified,
        step.boundary};

    const double estimated_gradient_norm =
        (problem.orbital_gradient() + step.hessian_step.orbital).norm();
    if (keyframe_policy.observe(
            {problem.orbital_gradient().norm(),
             estimated_gradient_norm,
             globalization.rho,
             trial_radius,
             globalization.next_radius,
             step.boundary}) == NeoKeyframePolicy::Decision::Refresh) {
      return finish_last_accepted();
    }
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
        result->termination_reason = "neo_dual_tolerance";
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
