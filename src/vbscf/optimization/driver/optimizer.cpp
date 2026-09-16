#include "vbscf/optimization/driver/optimizer.hpp"

#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vbscf/orbitals/gauge/localized.hpp"
#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/canonicalization.hpp"
#include "vbscf/orbitals/charts/layout.hpp"
#include "vbscf/orbitals/gauge/support_preserving.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/optimization/objective/function.hpp"
#include "vbscf/optimization/backends/lbfgs.hpp"
#include "vbscf/optimization/backends/projected_gradient.hpp"
#include "vbscf/optimization/backends/truncated_newton.hpp"
#include "vbscf/optimization/driver/session.hpp"
#include "vbscf/optimization/driver/checks.hpp"

namespace xmvb::vb {

namespace {

Eigen::MatrixXd build_one_particle_density_matrix(
    const OrbitalGradientResult& gradient_result,
    const OrbitalPreparationInput& orbital_input) {
  const auto& orbital_result = gradient_result.orbital_preparation_result;
  const auto& second_order_context = gradient_result.second_order_context;
  if (second_order_context == nullptr) {
    throw std::invalid_argument(
        "final one-particle density requires an accepted-point context");
  }

  const int n_bf = static_cast<int>(orbital_input.n_basis_functions);
  const int n_active = static_cast<int>(orbital_input.n_active_orbitals);
  const int n_inactive =
      static_cast<int>(
          (orbital_input.n_total_electrons - orbital_input.n_active_electrons) /
          2);
  if (orbital_result.auxiliary_orbital_matrix.rows() != n_bf ||
      orbital_result.auxiliary_orbital_matrix.cols() != n_bf ||
      orbital_result.inactive_density_matrix.rows() != n_bf ||
      orbital_result.inactive_density_matrix.cols() != n_bf ||
      second_order_context->active_one_electron_gradient.size() !=
          static_cast<std::size_t>(n_active * n_active)) {
    throw std::invalid_argument(
        "final one-particle density inputs have inconsistent dimensions");
  }

  const Eigen::Map<const Eigen::MatrixXd> active_density_adjoint(
      second_order_context->active_one_electron_gradient.data(),
      n_active,
      n_active);
  const auto active_orbitals =
      orbital_result.auxiliary_orbital_matrix.middleCols(n_inactive, n_active);
  Eigen::MatrixXd density = 2.0 * orbital_result.inactive_density_matrix;
  density.noalias() +=
      active_orbitals * active_density_adjoint.transpose() *
      active_orbitals.transpose();
  return 0.5 * (density + density.transpose());
}

}  // namespace

using optimizer_detail::build_orbital_chart;
using optimizer_detail::record_accepted_iteration_snapshot;
using optimizer_detail::sync_result_from_objective;
using optimizer_detail::uses_nonredundant_space;

VbScfOptimizer::VbScfOptimizer(
    VbScfOptimizerOptions options)
    : orbital_gradient_evaluator_(),
      scf_evaluator_(),
      options_(options) {}

VbScfOptimizer::VbScfOptimizer(
    OrbitalGradientEvaluator orbital_gradient_evaluator,
    VbScfEvaluator scf_evaluator,
    VbScfOptimizerOptions options)
    : orbital_gradient_evaluator_(std::move(orbital_gradient_evaluator)),
      scf_evaluator_(std::move(scf_evaluator)),
      options_(options) {}

VbScfOptimizerResult VbScfOptimizer::optimize(
    VbScfInput input,
    double nuclear_repulsion_energy) const {
  return optimize(std::move(input), {0}, {1.0}, nuclear_repulsion_energy);
}

VbScfOptimizerResult VbScfOptimizer::optimize(
    VbScfInput input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy) const {
  if (options_.max_iterations <= 0) {
    throw std::invalid_argument("max_iterations must be positive");
  }
  if (options_.gradient_tolerance <= 0.0 ||
      options_.energy_tolerance <= 0.0 ||
      options_.initial_step_size <= 0.0 ||
      options_.minimum_step_size <= 0.0) {
    throw std::invalid_argument("optimizer tolerances and step sizes must be positive");
  }
  if (options_.minimum_step_size > options_.initial_step_size) {
    throw std::invalid_argument("minimum_step_size must not exceed initial_step_size");
  }
  if (options_.history_size <= 0) {
    throw std::invalid_argument("history_size must be positive");
  }
  if (options_.tnhvp_max_subspace_dimension < 0) {
    throw std::invalid_argument(
        "tnhvp_max_subspace_dimension must be nonnegative");
  }
  if (options_.nonredundant_truncated_newton_transport_history_size < 0) {
    throw std::invalid_argument(
        "nonredundant_truncated_newton_transport_history_size must be nonnegative");
  }
  VbScfOptimizerResult result;
  const auto optimization_start_time = std::chrono::steady_clock::now();

  // Keep the source in the same support-preserving section as accepted trials.
  // Loaded guesses already carry an inactive section; inputs without one need
  // its initial selection before the active representative is balanced.
  if (!input.orbital_preparation_input.maintain_inactive_gauge) {
    apply_support_preserving_inactive_gauge(&input.orbital_preparation_input);
  }
  balance_active_gauge(&input.orbital_preparation_input);
  const SparseParameterLayout parameter_view(
      input.orbital_preparation_input);
  Eigen::VectorXd parameter_vector =
      parameter_view.pack(input.orbital_preparation_input);
  Eigen::MatrixXd initial_normalized_orbital_matrix;

  VbScfObjective objective(
      std::move(input),
      parameter_view,
      selected_state_indices,
      state_average_weights,
      nuclear_repulsion_energy,
      options_.structure_eigensolver,
      StructureSolveAccuracy{
          options_.energy_tolerance,
          options_.gradient_tolerance},
      &orbital_gradient_evaluator_,
      &scf_evaluator_);
  const int n = static_cast<int>(parameter_vector.size());
  int n_iterations = 0;
  double final_gradient_l2_norm = 0.0;
  bool final_projected_gradient_ready = false;

  try {
    Eigen::VectorXd gradient(parameter_vector.size());
    double energy = objective(parameter_vector, gradient);
    initial_normalized_orbital_matrix =
        objective.gradient_result()
            .orbital_preparation_result
            .physical_orbital_frame
            .normalized_orbital_matrix;
    sync_result_from_objective(objective, &result);
    record_accepted_iteration_snapshot(&objective, 0, options_, &result);
    result.initial_total_energy = energy;
    result.initial_one_electron_reference_energy =
        objective.gradient_result().scf_result.one_electron_reference_energy;
    double previous_energy = energy;
    final_gradient_l2_norm = gradient.norm();
    switch (options_.backend) {

      case VbScfOptimizerBackend::Lbfgspp: {
        const auto backend_result =
            optimizer_detail::run_full_space_lbfgs_backend(
                &objective,
                options_,
                parameter_vector,
                gradient,
                energy,
                &result);
        n_iterations = backend_result.n_iterations;
        final_gradient_l2_norm = backend_result.final_gradient_l2_norm;
        break;
      }

      case VbScfOptimizerBackend::NonredundantProjectedGradient: {
        const auto backend_result =
            optimizer_detail::run_projected_gradient_backend(
                &objective,
                parameter_view,
                options_,
                parameter_vector,
                gradient,
                energy,
                &result);
        n_iterations = backend_result.n_iterations;
        final_gradient_l2_norm = backend_result.final_gradient_l2_norm;
        break;
      }

      case VbScfOptimizerBackend::NonredundantLbfgspp: {
        const auto backend_result =
            optimizer_detail::run_nonredundant_lbfgs_backend(
                &objective,
                parameter_view,
                options_,
                parameter_vector,
                gradient,
                energy,
                &result);
        n_iterations = backend_result.n_iterations;
        final_gradient_l2_norm = backend_result.final_gradient_l2_norm;
        break;
      }

      case VbScfOptimizerBackend::NonredundantTruncatedNewton: {
        const auto backend_result =
            optimizer_detail::run_truncated_newton_backend(
                &objective,
                parameter_view,
                options_,
                parameter_vector,
                gradient,
                energy,
                &result);
        n_iterations = backend_result.n_iterations;
        final_gradient_l2_norm = backend_result.final_gradient_l2_norm;
        final_projected_gradient_ready =
            backend_result.final_projected_gradient_ready;
        break;
      }

    }
  } catch (const std::exception& error) {
    result.termination_reason = error.what();
  }

  if (result.total_energy_history.empty()) {
    if (!result.termination_reason.empty()) {
      throw std::runtime_error(
          std::string("optimizer did not evaluate the objective: ") +
          result.termination_reason);
    }
    throw std::runtime_error("optimizer did not evaluate the objective");
  }

  result.n_iterations = n_iterations;
  result.final_total_energy = result.total_energy_history.back();
  result.final_one_electron_reference_energy =
      result.scf_result.one_electron_reference_energy;
  result.final_gradient_inf_norm = result.gradient_inf_norm_history.back();
  result.final_gradient_l2_norm = final_gradient_l2_norm;
  if (uses_nonredundant_space(options_.backend) &&
      !final_projected_gradient_ready) {
    const OrbitalChart final_space =
        build_orbital_chart(objective, parameter_view);
    const Eigen::VectorXd final_packed_gradient =
        parameter_view.gather_from_full(
            objective.gradient_result().sparse_orbital_energy_gradient);
    const auto final_projection =
        final_space.project_gradient(final_packed_gradient);
    result.final_projected_gradient_inf_norm =
        gradient_infinity_norm(final_projection.reduced_gradient);
    result.final_projected_gradient_l2_norm =
        final_projection.reduced_gradient.norm();
  }
  result.one_particle_density_matrix = build_one_particle_density_matrix(
      objective.gradient_result(),
      objective.input().orbital_preparation_input);
  result.energy_only_evaluation_count =
      objective.energy_only_call_count();
  result.energy_only_wall_time_seconds =
      objective.energy_only_wall_time_seconds();
  Eigen::MatrixXd final_normalized_orbital_matrix =
      objective.gradient_result()
          .orbital_preparation_result
          .physical_orbital_frame
          .normalized_orbital_matrix;
  result.optimized_input = std::move(objective).take_input();
  if (final_normalized_orbital_matrix.size() != 0) {
    // The evaluator always works with the normalized physical orbital frame,
    // Store the normalized accepted frame in the final
    // sparse slots before exporting so Molden / restart artifacts see the
    // actual accepted physical orbitals rather than a pre-normalization raw
    // parameter vector.
    //
    // The OEO representative is an export gauge, not an optimizer mutation.
    // Rebuild it once from the converged auxiliary block for Molden/restart
    // output without perturbing the accepted-point tangent chart.
    final_normalized_orbital_matrix =
        build_metric_preserving_oeo_repaired_normalized_orbital_matrix(
            result.optimized_input.orbital_preparation_input,
            objective.gradient_result().orbital_preparation_result,
            initial_normalized_orbital_matrix);
    overwrite_sparse_orbitals_from_dense_physical_frame(
        final_normalized_orbital_matrix,
        &result.optimized_input.orbital_preparation_input);
  }
  enforce_strict_sparse_orbital_support(
      &result.optimized_input.orbital_preparation_input);

  if (result.termination_reason.empty()) {
    if (result.n_iterations >= options_.max_iterations) {
      result.termination_reason = "max_iterations";
    } else {
      result.termination_reason = "stopped";
    }
  }

  const auto optimization_end_time = std::chrono::steady_clock::now();
  const std::chrono::duration<double> total_elapsed_seconds =
      optimization_end_time - optimization_start_time;
  result.total_wall_time_seconds = total_elapsed_seconds.count();

  return result;
}

}  // namespace xmvb::vb
