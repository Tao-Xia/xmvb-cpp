#include "vbscf/optimization/vbscf_optimizer.hpp"

#include <chrono>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vbscf/orbitals/gauge/localized_representative.hpp"
#include "vbscf/orbitals/charts/support_layout_adapter.hpp"
#include "vbscf/orbitals/charts/orbital_chart.hpp"
#include "vbscf/orbitals/charts/orbital_chart_canonicalization.hpp"
#include "vbscf/orbitals/charts/sparse_parameter_layout.hpp"
#include "vbscf/orbitals/gauge/support_preserving_gauge.hpp"
#include "vbscf/optimization/vbscf_objective.hpp"
#include "vbscf/optimization/backends/lbfgs_backends.hpp"
#include "vbscf/optimization/backends/projected_gradient_backend.hpp"
#include "vbscf/optimization/backends/truncated_newton_backend.hpp"
#include "vbscf/optimization/optimizer_session.hpp"
#include "vbscf/optimization/optimization_checks.hpp"

namespace xmvb::vb {

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
    const VbScfInput& input,
    double nuclear_repulsion_energy) const {
  return optimize(input, {0}, {1.0}, nuclear_repulsion_energy);
}

VbScfOptimizerResult VbScfOptimizer::optimize(
    const VbScfInput& input,
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
  if (options_.nonredundant_truncated_newton_max_cg_iterations < 0) {
    throw std::invalid_argument(
        "nonredundant_truncated_newton_max_cg_iterations must be nonnegative");
  }
  if (options_.nonredundant_truncated_newton_hvp_step_size <= 0.0) {
    throw std::invalid_argument(
        "nonredundant_truncated_newton_hvp_step_size must be positive");
  }
  if (options_.nonredundant_truncated_newton_transport_history_size < 0) {
    throw std::invalid_argument(
        "nonredundant_truncated_newton_transport_history_size must be nonnegative");
  }
  VbScfOptimizerResult result;
  const auto optimization_start_time = std::chrono::steady_clock::now();

  // For `guess=mo`, numerical parity with the legacy VBSCF implementation is
  // more important than any temporary convergence-speed heuristic. Keep the
  // nonredundant optimizer on the original legacy sparse chart and exact-
  // support block partition so the reduced coordinates, projected gradients,
  // and exact-context orbital derivatives all live on the same variational
  // manifold as the reference `.xmo` calculation.
  std::optional<VbScfInput> adapted_optimizer_input;
  const VbScfInput* optimizer_input = &input;
  if (uses_nonredundant_space(options_.backend)) {
    adapted_optimizer_input = build_nonredundant_optimizer_input(input);
    optimizer_input = &adapted_optimizer_input.value();
  }
  const SparseParameterLayout parameter_view(
      optimizer_input->orbital_preparation_input);
  Eigen::VectorXd parameter_vector =
      parameter_view.pack(optimizer_input->orbital_preparation_input);
  Eigen::MatrixXd initial_normalized_orbital_matrix;

  VbScfObjective objective(
      *optimizer_input,
      parameter_view,
      selected_state_indices,
      state_average_weights,
      nuclear_repulsion_energy,
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
        objective.last_gradient_result()
            .orbital_preparation_result
            .physical_orbital_frame
            .normalized_orbital_matrix;
    sync_result_from_objective(objective, &result);
    record_accepted_iteration_snapshot(&objective, 0, options_, &result);
    result.initial_total_energy = energy;
    result.initial_one_electron_reference_energy =
        objective.last_gradient_result().scf_result.one_electron_reference_energy;
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
            objective.last_gradient_result().sparse_orbital_energy_gradient);
    const auto final_projection =
        final_space.project_gradient(final_packed_gradient);
    result.final_projected_gradient_inf_norm =
        gradient_infinity_norm(final_projection.reduced_gradient);
    result.final_projected_gradient_l2_norm =
        final_projection.reduced_gradient.norm();
  }
  result.optimized_input = objective.last_input();
  Eigen::MatrixXd final_normalized_orbital_matrix =
      objective.last_gradient_result()
          .orbital_preparation_result
          .physical_orbital_frame
          .normalized_orbital_matrix;
  if (final_normalized_orbital_matrix.size() != 0) {
    // The evaluator always works with the normalized physical orbital frame,
    // mirroring legacy `normalize(...)`. Store that same frame in the final
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
            objective.last_gradient_result().orbital_preparation_result,
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
