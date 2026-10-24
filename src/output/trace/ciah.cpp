#include "output/trace/ciah.hpp"

#include <filesystem>
#include <iomanip>
#include <stdexcept>

#include "vbscf/optimization/driver/result.hpp"

namespace xmvb::output {

CiahTraceWriter::CiahTraceWriter(const std::filesystem::path& output_path)
    : output_path_(output_path) {
    if (output_path.empty()) {
        throw std::invalid_argument("CIAH trace path must not be empty");
    }
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    stream_.open(output_path);
    if (!stream_) {
        throw std::runtime_error("failed to open CIAH trace: " + output_path.string());
    }
    stream_ << "iteration\tkeyframes\tmodel_dimension\tmicro_iterations"
            << "\tcoupled_block_actions"
            << "\torbital_hvp_actions\tstructure_response_actions"
            << "\trejected_trials"
            << "\tkkt_residual_norm\tkkt_residual_target"
            << "\tcurvature_residual_norm\tcurvature_residual_target"
            << "\tglobal_curvature_certified"
            << "\tinitial_trust_radius\taccepted_trial_radius\tnext_trust_radius"
            << "\tgradient_dot_step\tstep_dot_hessian_step"
            << "\tpredicted_reduction\tactual_reduction\ttrust_ratio"
            << "\treached_boundary"
            << "\tsource_gradient_norm\tsource_physical_gradient_norm"
            << "\tphysical_orbital_kkt_norm\torbital_step_norm\tshift"
            << "\taccepted_physical_gradient_norm\tapproximate_step\n";
    stream_ << std::setprecision(17);
    stream_.flush();
    if (!stream_) {
        throw std::runtime_error("failed to write CIAH trace: " + output_path.string());
    }
}

void CiahTraceWriter::write(const vb::CiahIterationRecord& step) {
    stream_ << step.accepted_iteration_index << '\t' << step.keyframes << '\t'
            << step.model_dimension << '\t' << step.micro_iterations << '\t'
            << step.coupled_block_actions << '\t' << step.orbital_hvp_actions << '\t'
            << step.structure_response_actions << '\t' << step.rejected_trial_count << '\t'
            << step.kkt_residual_norm << '\t' << step.kkt_residual_target << '\t'
            << step.curvature_residual_norm << '\t' << step.curvature_residual_target << '\t'
            << (step.global_curvature_certified ? 1 : 0) << '\t' << step.initial_trust_radius
            << '\t' << step.accepted_trial_radius << '\t' << step.next_trust_radius << '\t'
            << step.gradient_dot_step << '\t' << step.step_dot_hessian_step << '\t'
            << step.predicted_reduction << '\t' << step.actual_reduction << '\t' << step.trust_ratio
            << '\t' << (step.reached_boundary ? 1 : 0) << '\t' << step.source_gradient_norm << '\t'
            << step.source_physical_gradient_norm << '\t' << step.physical_orbital_kkt_norm << '\t'
            << step.orbital_step_norm << '\t' << step.shift << '\t'
            << step.accepted_physical_gradient_norm << '\t' << step.approximate_step << '\n';
    stream_.flush();
    if (!stream_) {
        throw std::runtime_error("failed to write CIAH trace: " + output_path_.string());
    }
}

} // namespace xmvb::output
