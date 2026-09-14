#pragma once

#include <Eigen/Core>

#include "vbscf/optimization/driver/options.hpp"
#include "vbscf/orbitals/charts/chart.hpp"

namespace xmvb::vb {

class SparseParameterLayout;
class VbScfObjective;
struct VbScfOptimizerResult;
struct TnhvpIterationRecord;

namespace optimizer_detail {

bool uses_nonredundant_space(VbScfOptimizerBackend backend);

OrbitalChart build_orbital_chart(
    const VbScfObjective& objective,
    const SparseParameterLayout& parameter_view);

int choose_truncated_newton_max_cg_iterations(
    const VbScfOptimizerOptions& options,
    int reduced_size);

int choose_truncated_newton_transport_history_size(
    const VbScfOptimizerOptions& options);

void sync_result_from_objective(
    const VbScfObjective& objective,
    VbScfOptimizerResult* result);

void record_accepted_iteration_snapshot(
    VbScfObjective* objective,
    int accepted_iteration_index,
    const VbScfOptimizerOptions& options,
    VbScfOptimizerResult* result,
    const TnhvpIterationRecord* tnhvp = nullptr,
    const Eigen::VectorXd* reduced_gradient = nullptr);

}  // namespace optimizer_detail

}  // namespace xmvb::vb
