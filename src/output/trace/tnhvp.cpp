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
      << "\tcurvature_subspace_dimension"
      << "\texact_hvp_block_actions"
      << "\tstructure_response_block_actions"
      << "\tresponse_low_rank_new_columns"
      << "\tpreconditioner_history_size\trejected_trials"
      << "\touter_iteration_seconds"
      << "\taccepted_point_setup_seconds"
      << "\texact_hvp_seconds"
      << "\touter_response_seconds"
      << "\tresponse_low_rank_seconds"
      << "\tresponse_low_rank_structure_action_seconds"
      << "\tresponse_low_rank_adjoint_seconds"
      << "\ttrial_objective_seconds"
      << "\tgradient_log_progress_per_second"
      << "\tsource_gradient_l2\taccepted_gradient_l2\tforcing_term"
      << "\tmodel_kkt_relative_residual"
      << "\tmax_structure_response_relative_residual"
      << "\tinitial_trust_radius\taccepted_trial_radius\tnext_trust_radius"
      << "\tstep_norm\tlinear_decrease\tpredicted_decrease"
      << "\tactual_decrease\ttrust_ratio"
      << "\tminimum_ritz_value\tminimum_shifted_ritz_value"
      << "\ttrust_region_shift\treached_boundary"
      << "\tnegative_curvature\tchart_changed"
      << "\tnewton_trial_evaluated\taccepted_newton_step"
      << "\tnewton_trial_actual_decrease\tnewton_trial_predicted_decrease"
      << "\tnewton_trial_step_norm\n";
  stream << std::setprecision(17);
  for (const auto& step : result.tnhvp_iteration_trace) {
    stream
        << step.accepted_iteration_index << '\t'
        << step.reduced_dimension << '\t'
        << step.curvature_subspace_dimension << '\t'
        << step.exact_hvp_block_actions << '\t'
        << step.structure_response_block_actions << '\t'
        << step.response_low_rank_new_columns << '\t'
        << step.preconditioner_history_size << '\t'
        << step.rejected_trial_count << '\t'
        << step.outer_iteration_wall_time_seconds << '\t'
        << step.accepted_point_setup_wall_time_seconds << '\t'
        << step.exact_hvp_wall_time_seconds << '\t'
        << step.outer_response_wall_time_seconds << '\t'
        << step.response_low_rank_wall_time_seconds << '\t'
        << step.response_low_rank_structure_action_wall_time_seconds << '\t'
        << step.response_low_rank_adjoint_wall_time_seconds << '\t'
        << step.trial_objective_wall_time_seconds << '\t'
        << step.gradient_log_progress_per_second << '\t'
        << step.source_gradient_l2_norm << '\t'
        << step.accepted_gradient_l2_norm << '\t'
        << step.forcing_term << '\t'
        << step.model_kkt_relative_residual << '\t'
        << step.max_structure_response_relative_residual << '\t'
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
        << (step.chart_changed ? 1 : 0) << '\t'
        << (step.newton_trial_evaluated ? 1 : 0) << '\t'
        << (step.accepted_newton_step ? 1 : 0) << '\t'
        << step.newton_trial_actual_decrease << '\t'
        << step.newton_trial_predicted_decrease << '\t'
        << step.newton_trial_step_norm << '\n';
  }
  if (!stream) {
    throw std::runtime_error(
        "failed to write TNHVP trace: " + output_path.string());
  }
}

}  // namespace xmvb::output
