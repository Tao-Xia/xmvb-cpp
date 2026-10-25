#include "vbscf/optimization/driver/session.hpp"
#include "integrals/active/raw.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>

#include "vbscf/optimization/objective/function.hpp"
#include "vbscf/optimization/driver/checks.hpp"
#include "vbscf/optimization/driver/result.hpp"
#include "vbscf/orbitals/charts/layout.hpp"
#include "vbscf/optimization/preconditioners/hessian_diagonal.hpp"
#include "vbscf/optimization/trust_region/retraction.hpp"

namespace xmvb::vb::optimizer_detail {

OrbitalChart build_orbital_geometry(const VbScfObjective& objective,
                                    const SparseParameterLayout& parameter_view,
                                    OrbitalChart::Coordinates coordinates) {
    return build_orbital_geometry(objective.input().orbital_preparation_input,
                                  objective.gradient_result().orbital_preparation_result,
                                  parameter_view,
                                  objective.input().complete_active_space,
                                  coordinates);
}

OrbitalChart build_orbital_geometry(const OrbitalPreparationInput& orbital_preparation_input,
                                    const OrbitalPreparationResult& orbital_preparation_result,
                                    const SparseParameterLayout& parameter_view,
                                    bool complete_active_space,
                                    OrbitalChart::Coordinates coordinates) {
    const auto& normalized_orbital_matrix =
        orbital_preparation_result.physical_orbital_frame.normalized_orbital_matrix;
    if (normalized_orbital_matrix.size() == 0) {
        throw std::runtime_error("nonredundant space requires the cached physical orbital frame");
    }
    const int n_inactive_doubly_occupied_orbitals = (orbital_preparation_input.n_total_electrons -
                                                     orbital_preparation_input.n_active_electrons) /
                                                    2;
    const int n_occupied_orbitals =
        n_inactive_doubly_occupied_orbitals + orbital_preparation_input.n_active_orbitals;
    return OrbitalChart(
        orbital_preparation_input,
        parameter_view,
        orbital_preparation_result.auxiliary_orbital_matrix.leftCols(n_occupied_orbitals),
        normalized_orbital_matrix,
        nullptr,
        false,
        complete_active_space,
        nullptr,
        coordinates);
}

void prepare_orbital_curvature(const VbScfObjective& objective,
                               OrbitalChart& chart,
                               OrbitalPreconditioner preconditioner,
                               bool need_ah_mapping) {
    if (preconditioner == OrbitalPreconditioner::Automatic) {
        throw std::invalid_argument(
            "orbital-chart construction requires a resolved preconditioner");
    }
    std::optional<AnalyticOrbitalDiagonal> hessian_diagonal;
    if (preconditioner == OrbitalPreconditioner::HessianDiagonal || need_ah_mapping) {
        const auto& context = objective.second_order_context();
        if (!context) {
            throw std::invalid_argument(
                "analytic Hessian diagonal requires an accepted-point context");
        }
        hessian_diagonal = build_analytic_orbital_diagonal(objective.input(), *context);
    }
    const Eigen::MatrixXd* one_electron =
        preconditioner == OrbitalPreconditioner::OneElectron
            ? &objective.gradient_result().ao_effective_one_electron_result.ao_effective_h1e
            : nullptr;
    chart.set_curvature(objective.input().orbital_preparation_input,
                        one_electron,
                        hessian_diagonal ? &*hessian_diagonal : nullptr,
                        need_ah_mapping && preconditioner == OrbitalPreconditioner::HessianDiagonal
                            ? OrbitalChart::CurvatureUse::AugmentedHessian
                            : OrbitalChart::CurvatureUse::PositiveBlock);
}

OrbitalChart build_orbital_chart(const VbScfObjective& objective,
                                 const SparseParameterLayout& parameter_view,
                                 OrbitalPreconditioner preconditioner,
                                 bool need_ah_mapping) {
    auto chart = build_orbital_geometry(objective, parameter_view);
    prepare_orbital_curvature(objective, chart, preconditioner, need_ah_mapping);
    return chart;
}

bool orbital_gradient_converged(const OrbitalPreparationInput& orbitals,
                                const SparseParameterLayout& parameter_view,
                                const OrbitalChart& chart,
                                const Eigen::VectorXd& reduced_gradient,
                                double tolerance) {
    if (chart.uses_explicit_whitening()) {
        if (gradient_infinity_norm(reduced_gradient) >= tolerance)
            return false;
    } else {
        // ||g_white||_infinity >= ||g_white||_2/sqrt(d). Reject obviously
        // unconverged points cheaply, then use the EXACT established reference
        // infinity criterion. This does not substitute a different threshold.
        if (chart.coordinate_dual_norm(reduced_gradient) >=
            tolerance * std::sqrt(static_cast<double>(chart.reduced_size())))
            return false;
        if (chart.reference_gradient_inf_norm(reduced_gradient) >= tolerance)
            return false;
    }
    if (!chart.uses_active_subspace_gauge())
        return true;
    const NonredundantRetractionMetric metric(chart, parameter_view, orbitals);
    return metric.dual_norm(reduced_gradient) < tolerance;
}

void sync_result_from_objective(const VbScfObjective& objective, VbScfOptimizerResult* result) {
    result->total_energy_history = objective.energy_history();
    result->gradient_inf_norm_history = objective.gradient_inf_norm_history();
    result->iteration_time_history_seconds = objective.iteration_time_history_seconds();
    result->scf_result = objective.gradient_result().scf_result;
}

void record_accepted_iteration_snapshot(VbScfObjective* objective,
                                        int accepted_iteration_index,
                                        const VbScfOptimizerOptions& options,
                                        VbScfOptimizerResult* result,
                                        const TnhvpIterationRecord* tnhvp,
                                        const Eigen::VectorXd* reduced_gradient,
                                        const OrbitalChart* chart) {
    if (!options.retain_accepted_iteration_trace && !options.accepted_iteration_callback) {
        return;
    }
    const bool include_reference_energy_gradient =
        options.retain_accepted_iteration_trace ||
        options.accepted_iteration_callback_requires_reference_gradient;
    const bool include_full_payload = options.retain_accepted_iteration_trace ||
                                      options.accepted_iteration_callback_requires_full_snapshot;
    if (include_reference_energy_gradient) {
        objective->ensure_reference_gradient();
    }
    const auto& gradient_result = objective->gradient_result();
    VbScfAcceptedIterationSnapshot snapshot;
    snapshot.accepted_iteration_index = accepted_iteration_index;
    if (tnhvp != nullptr) {
        snapshot.tnhvp = *tnhvp;
    }
    snapshot.has_full_payload = include_full_payload;
    for (const double value : gradient_result.sparse_orbital_energy_gradient) {
        snapshot.sparse_orbital_energy_gradient_inf_norm =
            std::max(snapshot.sparse_orbital_energy_gradient_inf_norm, std::abs(value));
        snapshot.sparse_orbital_energy_gradient_l2_norm += value * value;
    }
    snapshot.sparse_orbital_energy_gradient_l2_norm =
        std::sqrt(snapshot.sparse_orbital_energy_gradient_l2_norm);
    Eigen::VectorXd computed_reduced_gradient;
    std::optional<OrbitalChart> snapshot_chart;
    if (reduced_gradient == nullptr) {
        const SparseParameterLayout parameter_view(objective->input().orbital_preparation_input);
        // A gradient snapshot requires the quotient pullback only. Do not build
        // one-electron curvature blocks that no snapshot consumer uses.
        snapshot_chart.emplace(build_orbital_geometry(*objective,
                                                      parameter_view,
                                                      options.backend == VbScfOptimizerBackend::Ciah
                                                          ? OrbitalChart::Coordinates::Unwhitened
                                                          : OrbitalChart::Coordinates::Whitened));
        chart = &*snapshot_chart;
        const Eigen::VectorXd packed_gradient =
            parameter_view.gather_from_full(gradient_result.sparse_orbital_energy_gradient);
        computed_reduced_gradient = chart->project_gradient(packed_gradient).reduced_gradient;
        reduced_gradient = &computed_reduced_gradient;
    }
    snapshot.has_projected_gradient = true;
    snapshot.projected_gradient_inf_norm = gradient_infinity_norm(*reduced_gradient);
    snapshot.projected_gradient_l2_norm =
        chart ? chart->coordinate_dual_norm(*reduced_gradient) : reduced_gradient->norm();
    if (include_full_payload) {
        snapshot.orbital_value_table.assign(
            objective->input().orbital_preparation_input.orbital_value_table.begin(),
            objective->input().orbital_preparation_input.orbital_value_table.end());
        snapshot.structure_matrices = gradient_result.scf_result.structure_matrices;
        snapshot.active_orbital_overlap_matrix = gradient_result.active_orbital_overlap_matrix;
        snapshot.active_one_electron_integrals = integrals::materialize_raw_one_electron(
            gradient_result.active_one_electron_integrals, gradient_result.two_electron_frame);
        snapshot.packed_active_two_electron_integrals = integrals::materialize_raw_pair_kernel(
            gradient_result.packed_active_two_electron_integrals,
            gradient_result.two_electron_frame);
        snapshot.sparse_orbital_energy_gradient = gradient_result.sparse_orbital_energy_gradient;
        if (include_reference_energy_gradient) {
            snapshot.sparse_orbital_reference_energy_gradient =
                gradient_result.sparse_orbital_reference_energy_gradient;
        }
    }
    snapshot.total_energy = gradient_result.scf_result.total_energy;
    snapshot.one_electron_reference_energy =
        gradient_result.scf_result.one_electron_reference_energy;
    snapshot.average_structure_overlap = gradient_result.scf_result.average_structure_overlap;
    if (options.retain_accepted_iteration_trace) {
        result->accepted_iteration_trace.push_back(snapshot);
    }
    if (options.accepted_iteration_callback) {
        options.accepted_iteration_callback(snapshot);
    }
}

} // namespace xmvb::vb::optimizer_detail
