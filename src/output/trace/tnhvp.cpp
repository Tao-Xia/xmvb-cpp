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
      << "iteration\treduced_dimension\tsubspace_dimension"
      << "\tsecant_correction_size\trejected_trials"
      << "\thvp_directions\thvp_batches\tcore_hvp_directions"
      << "\touter_response_directions\tsubproblems\thvp_seconds"
      << "\touter_response_seconds\tgradient_log_progress_per_second"
      << "\tused_outer_response\tresponse_probe_performed"
      << "\tresponse_probe_relative_residual\tresponse_work_ratio"
      << "\tresponse_scale_affordable\tresponse_deferred_for_cost"
      << "\tused_full_hvp"
      << "\tsource_gradient_l2\taccepted_gradient_l2\tforcing_term"
      << "\thas_kkt_residual\tkkt_relative_residual\tkkt_inf_norm"
      << "\ttrust_model_error_order\tsubproblem_met_model_kkt"
      << "\tsubproblem_stopped_by_work_budget"
      << "\tinitial_trust_radius\taccepted_trial_radius\tnext_trust_radius"
      << "\tstep_norm\tlinear_decrease\tpredicted_decrease"
      << "\tactual_decrease\ttrust_ratio"
      << "\tmodel_spectral_radius\ttrust_region_shift\treached_boundary"
      << "\tnegative_curvature\treused_subspace\tchart_changed\n";
  stream << std::setprecision(17);
  for (const auto& step : result.tnhvp_iteration_trace) {
    stream
        << step.accepted_iteration_index << '\t'
        << step.reduced_dimension << '\t'
        << step.subspace_dimension << '\t'
        << step.secant_correction_size << '\t'
        << step.rejected_trial_count << '\t'
        << step.hvp_direction_count << '\t'
        << step.hvp_batch_count << '\t'
        << step.core_hvp_direction_count << '\t'
        << step.outer_response_direction_count << '\t'
        << step.subproblem_count << '\t'
        << step.hvp_wall_time_seconds << '\t'
        << step.outer_response_wall_time_seconds << '\t'
        << step.gradient_log_progress_per_second << '\t'
        << (step.used_outer_response ? 1 : 0) << '\t'
        << (step.response_probe_performed ? 1 : 0) << '\t'
        << step.response_probe_relative_residual << '\t'
        << step.response_work_ratio << '\t'
        << (step.response_scale_affordable ? 1 : 0) << '\t'
        << (step.response_deferred_for_cost ? 1 : 0) << '\t'
        << (step.used_full_hvp ? 1 : 0) << '\t'
        << step.source_gradient_l2_norm << '\t'
        << step.accepted_gradient_l2_norm << '\t'
        << step.forcing_term << '\t'
        << (step.has_kkt_residual ? 1 : 0) << '\t'
        << step.kkt_relative_residual << '\t'
        << step.kkt_inf_norm << '\t'
        << step.trust_model_error_order << '\t'
        << (step.subproblem_met_model_kkt ? 1 : 0) << '\t'
        << (step.subproblem_stopped_by_work_budget ? 1 : 0) << '\t'
        << step.initial_trust_radius << '\t'
        << step.accepted_trial_radius << '\t'
        << step.next_trust_radius << '\t'
        << step.step_norm << '\t'
        << step.linear_decrease << '\t'
        << step.predicted_decrease << '\t'
        << step.actual_decrease << '\t'
        << step.trust_ratio << '\t'
        << step.model_spectral_radius << '\t'
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
