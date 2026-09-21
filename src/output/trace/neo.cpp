#include "output/trace/neo.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#include "vbscf/optimization/driver/result.hpp"

namespace xmvb::output {

void write_neo_trace(
    const std::filesystem::path& output_path,
    const vb::VbScfOptimizerResult& result) {
  if (output_path.empty()) {
    throw std::invalid_argument("NEO trace path must not be empty");
  }
  if (!output_path.parent_path().empty()) {
    std::filesystem::create_directories(output_path.parent_path());
  }

  std::ofstream stream(output_path);
  if (!stream) {
    throw std::runtime_error(
        "failed to open NEO trace: " + output_path.string());
  }
  stream
      << "iteration\tkeyframes\tmodel_dimension\tmicro_iterations"
      << "\tcoupled_block_actions"
      << "\trejected_trials"
      << "\tkkt_residual_norm\tkkt_residual_target"
      << "\tcurvature_residual_norm\tcurvature_residual_target"
      << "\tglobal_curvature_certified"
      << "\tinitial_trust_radius\taccepted_trial_radius\tnext_trust_radius"
      << "\tgradient_dot_step\tstep_dot_hessian_step"
      << "\tpredicted_reduction\tactual_reduction\ttrust_ratio"
      << "\treached_boundary\n";
  stream << std::setprecision(17);
  for (const auto& step : result.neo_iteration_trace) {
    stream
        << step.accepted_iteration_index << '\t'
        << step.keyframes << '\t'
        << step.model_dimension << '\t'
        << step.micro_iterations << '\t'
        << step.coupled_block_actions << '\t'
        << step.rejected_trial_count << '\t'
        << step.kkt_residual_norm << '\t'
        << step.kkt_residual_target << '\t'
        << step.curvature_residual_norm << '\t'
        << step.curvature_residual_target << '\t'
        << (step.global_curvature_certified ? 1 : 0) << '\t'
        << step.initial_trust_radius << '\t'
        << step.accepted_trial_radius << '\t'
        << step.next_trust_radius << '\t'
        << step.gradient_dot_step << '\t'
        << step.step_dot_hessian_step << '\t'
        << step.predicted_reduction << '\t'
        << step.actual_reduction << '\t'
        << step.trust_ratio << '\t'
        << (step.reached_boundary ? 1 : 0) << '\n';
  }
  if (!stream) {
    throw std::runtime_error(
        "failed to write NEO trace: " + output_path.string());
  }
}

}  // namespace xmvb::output
