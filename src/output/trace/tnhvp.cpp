#include "output/trace/tnhvp.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#include "vbscf/optimization/driver/result.hpp"

namespace xmvb::output {

void write_tnhvp_trace(
    const std::filesystem::path& output_path,
    const vb::VbScfOptimizerResult& result) {
  if (output_path.empty()) {
    throw std::invalid_argument("TNHVP trace path must not be empty");
  }
  if (!output_path.parent_path().empty()) {
    std::filesystem::create_directories(output_path.parent_path());
  }

  std::ofstream stream(output_path);
  if (!stream) {
    throw std::runtime_error(
        "failed to open TNHVP trace: " + output_path.string());
  }
  stream
      << "iteration\treduced_dimension"
      << "\tcoupled_orbital_subspace_dimension"
      << "\tcoupled_response_subspace_dimension"
      << "\tcoupled_expansions\tcoupled_orbital_hessian_block_actions"
      << "\tcoupled_orbital_to_response_block_actions"
      << "\tcoupled_response_to_orbital_block_actions"
      << "\tcoupled_response_hessian_block_actions"
      << "\tcoupled_orbital_metric_block_actions"
      << "\tpreconditioner_history_size\trejected_trials"
      << "\touter_iteration_seconds"
      << "\tgradient_log_progress_per_second"
      << "\tsource_gradient_l2\taccepted_gradient_l2\tforcing_term"
      << "\torbital_backward_error\tresponse_backward_error"
      << "\tinitial_trust_radius\taccepted_trial_radius\tnext_trust_radius"
      << "\tstep_norm\tlinear_decrease\tpredicted_decrease"
      << "\tactual_decrease\ttrust_ratio"
      << "\tminimum_ritz_value\tminimum_shifted_ritz_value"
      << "\ttrust_region_shift\treached_boundary"
      << "\tnegative_curvature\treused_subspace\tchart_changed\n";
  stream << std::setprecision(17);
  for (const auto& step : result.tnhvp_iteration_trace) {
    stream
        << step.accepted_iteration_index << '\t'
        << step.reduced_dimension << '\t'
        << step.coupled_orbital_subspace_dimension << '\t'
        << step.coupled_response_subspace_dimension << '\t'
        << step.coupled_expansion_count << '\t'
        << step.coupled_orbital_hessian_block_actions << '\t'
        << step.coupled_orbital_to_response_block_actions << '\t'
        << step.coupled_response_to_orbital_block_actions << '\t'
        << step.coupled_response_hessian_block_actions << '\t'
        << step.coupled_orbital_metric_block_actions << '\t'
        << step.preconditioner_history_size << '\t'
        << step.rejected_trial_count << '\t'
        << step.outer_iteration_wall_time_seconds << '\t'
        << step.gradient_log_progress_per_second << '\t'
        << step.source_gradient_l2_norm << '\t'
        << step.accepted_gradient_l2_norm << '\t'
        << step.forcing_term << '\t'
        << step.orbital_backward_error << '\t'
        << step.response_backward_error << '\t'
        << step.initial_trust_radius << '\t'
        << step.accepted_trial_radius << '\t'
        << step.next_trust_radius << '\t'
        << step.step_norm << '\t'
        << step.linear_decrease << '\t'
        << step.predicted_decrease << '\t'
        << step.actual_decrease << '\t'
        << step.trust_ratio << '\t'
        << step.minimum_ritz_value << '\t'
        << step.minimum_shifted_ritz_value << '\t'
        << step.trust_region_shift << '\t'
        << (step.reached_boundary ? 1 : 0) << '\t'
        << (step.encountered_negative_curvature ? 1 : 0) << '\t'
        << (step.reused_subspace ? 1 : 0) << '\t'
        << (step.chart_changed ? 1 : 0) << '\n';
  }
  if (!stream) {
    throw std::runtime_error(
        "failed to write TNHVP trace: " + output_path.string());
  }
}

}  // namespace xmvb::output
