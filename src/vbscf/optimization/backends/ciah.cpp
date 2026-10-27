#include "vbscf/optimization/backends/ciah.hpp"
#include "vbscf/optimization/backends/ciah_step.hpp"
#include "mcscf/optimization/trace.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/derivatives/hessian/coupled/structure.hpp"
#include "vbscf/optimization/driver/checks.hpp"
#include "vbscf/optimization/driver/options.hpp"
#include "vbscf/optimization/driver/result.hpp"
#include "vbscf/optimization/driver/session.hpp"
#include "vbscf/optimization/objective/function.hpp"
#include "vbscf/optimization/trust_region/retraction.hpp"
#include "vbscf/orbitals/charts/layout.hpp"
#include "vbscf/structures/assembly/selected_coefficients.hpp"

namespace xmvb::vb::optimizer_detail {
namespace {

OrbitalPreconditioner resolve_ciah_preconditioner(OrbitalPreconditioner requested) {
    if (requested != OrbitalPreconditioner::Automatic)
        return requested;
    return OrbitalPreconditioner::HessianDiagonal;
}

bool admit_structure_response(const VbScfObjective& objective,
                              const StructureTangentOperator& structure_hessian,
                              const StructureAction& structure_action,
                              Eigen::Index orbital_dimension) {
    if (structure_hessian.tangent_size() == 0)
        return false;

    const auto& context = *objective.second_order_context();
    const auto& orbitals = objective.input().orbital_preparation_input;
    const long double n_ao = orbitals.n_basis_functions;
    const long double n_active = orbitals.n_active_orbitals;
    const long double n_ao_pairs = n_ao * (n_ao + 1.0L) / 2.0L;
    const long double n_active_pairs = n_active * (n_active + 1.0L) / 2.0L;
    const auto& graph = objective.input().ao_integral_input.pair_graph;
    // Exact HVP applies the AO-pair operator to active-pair columns. Its work
    // scales with graph edges, not just the number of AO-pair rows.
    const long double ao_pair_actions =
        graph.empty()           ? n_ao_pairs
        : graph.has_row_graph() ? static_cast<long double>(graph.values.size())
                                : 2.0L * static_cast<long double>(graph.integral_count());
    const long double core_pair_work = ao_pair_actions * n_active_pairs;
    // The orthogonal direct-CI branch uses pair-channel GEMM on spin products,
    // not the U_alpha^2 U_beta + U_alpha U_beta^2 cofactor sandwiches.
    // These are mathematical dimensions, independent of whether any numerical
    // product array is resident at the accepted point.
    const auto& selected = context.selected_state_matrices;
    const long double response_work =
        structure_action.supports_integral_direction()
            ? static_cast<long double>(context.same_spin_pair_cache.n_unique_alpha) *
                  context.same_spin_pair_cache.n_unique_beta * n_active_pairs * n_active_pairs *
                  context.selected_state_indices.size()
            : estimate_selected_state_contraction_work(selected);

    // A coupled orbital basis vector evaluates Bp in addition to the core HVP.
    // Retain the complete coupled model when either its horizontal response
    // space does not exceed the orbital tangent space or its leading
    // unique-string contraction is no more expensive than the core AO-pair
    // contraction.  Otherwise use orbital-only CIAH and let the exact trial
    // energy and gradient globalize it.  Both tests compare mathematical work
    // dimensions; neither contains a molecule, active-space, or timing cutoff.
    return structure_hessian.tangent_size() <= orbital_dimension || response_work <= core_pair_work;
}

bool build_accepted_ciah_keyframe(VbScfObjective* objective,
                                  const SparseParameterLayout& parameter_view,
                                  const VbScfOptimizerOptions& options,
                                  const OrbitalChart& chart,
                                  const Eigen::VectorXd& reduced_gradient,
                                  const Eigen::VectorXd& orbital_guess_packed,
                                  const Eigen::MatrixXd& structure_guess,
                                  double energy,
                                  double* trust_radius,
                                  CiahIterationRecord* record,
                                  VbScfOptimizerResult* result,
                                  CiahMacroStep* accepted,
                                  bool seek_curvature) {
    const OrbitalPreparationInput accepted_orbitals = objective->input().orbital_preparation_input;
    NonredundantRetractionMetric orbital_metric(chart, parameter_view, accepted_orbitals);
    record->source_gradient_norm = chart.coordinate_dual_norm(reduced_gradient);
    const double physical_gradient_norm = orbital_metric.dual_norm(reduced_gradient);
    record->source_physical_gradient_norm = physical_gradient_norm;
    ExactHvpOperator orbital_hessian(
        objective->second_order_context(), &objective->input(), parameter_view, &chart);
    if (!orbital_hessian.supports_analytic_core_model()) {
        result->termination_reason = "ciah_analytic_hessian_unavailable";
        return false;
    }
    StructureTangentOperator structure_hessian(objective->second_order_context(),
                                               orbital_hessian.structure_action());
    const bool use_structure_response = admit_structure_response(
        *objective, structure_hessian, orbital_hessian.structure_action(), chart.reduced_size());
    mcscf::diagnostics::ciah_trace_context = {record->accepted_iteration_index, 0, "ciah"};
    return build_ciah_macro_step(objective,
                                 parameter_view,
                                 options,
                                 chart,
                                 orbital_metric,
                                 orbital_hessian,
                                 structure_hessian,
                                 reduced_gradient,
                                 physical_gradient_norm,
                                 orbital_guess_packed,
                                 structure_guess,
                                 use_structure_response,
                                 energy,
                                 trust_radius,
                                 record,
                                 result,
                                 accepted,
                                 seek_curvature);
}

} // namespace

BackendRunResult run_ciah_backend(VbScfObjective* objective,
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
    double trust_radius = std::max(options.minimum_step_size, options.initial_step_size);
    const OrbitalPreconditioner preconditioner =
        resolve_ciah_preconditioner(options.orbital_preconditioner);
    OrbitalChart chart =
        build_orbital_geometry(*objective, parameter_view, OrbitalChart::Coordinates::Unwhitened);
    auto projected = chart.project_gradient(gradient);
    double final_gradient_l2 = chart.coordinate_dual_norm(projected.reduced_gradient);
    Eigen::VectorXd orbital_guess_packed;
    Eigen::MatrixXd structure_guess;
    bool seek_curvature = false;

    while (run_result.n_iterations < options.max_iterations) {
        if (chart.reduced_size() == 0) {
            result->termination_reason = "ciah_nonredundant_space_empty";
            final_gradient_l2 = 0.0;
            break;
        }
        final_gradient_l2 = chart.coordinate_dual_norm(projected.reduced_gradient);
        if (run_result.n_iterations == 0 &&
            orbital_gradient_converged(objective->input().orbital_preparation_input,
                                       parameter_view,
                                       chart,
                                       projected.reduced_gradient,
                                       options.gradient_tolerance)) {
            seek_curvature = true;
        }
        // Independent probing is reserved for a point meeting the ordinary
        // energy/gradient criteria; normal steps reuse their current AH span.
        prepare_orbital_curvature(*objective, chart, preconditioner, true);

        CiahIterationRecord iteration_record;
        iteration_record.accepted_iteration_index = run_result.n_iterations + 1;
        iteration_record.initial_trust_radius = trust_radius;
        CiahMacroStep accepted;
        if (!build_accepted_ciah_keyframe(objective,
                                          parameter_view,
                                          options,
                                          chart,
                                          projected.reduced_gradient,
                                          orbital_guess_packed,
                                          structure_guess,
                                          energy,
                                          &trust_radius,
                                          &iteration_record,
                                          result,
                                          &accepted,
                                          seek_curvature)) {
            if (seek_curvature)
                result->ciah_terminal_probe_hvp_actions += iteration_record.orbital_hvp_actions;
            break;
        }

        seek_curvature = false;
        if (!accepted.trial.gradient_ready)
            objective->complete_trial(&accepted.trial);
        accepted.parameters = parameter_view.pack(accepted.trial.orbital_preparation_input);
        gradient = accepted.trial.gradient;
        energy = accepted.trial.energy;
        objective->commit(std::move(accepted.trial));
        parameters = std::move(accepted.parameters);
        orbital_guess_packed = std::move(accepted.next_orbital_guess_packed);
        structure_guess = std::move(accepted.next_structure_guess);
        // CIAH may evaluate several fixed-structure keyframes before this final
        // relaxed point. Include the macro endpoint without discarding that work.
        ++iteration_record.keyframes;

        // Every accepted point has a new exact gradient and a new local model.
        // It is therefore an outer iteration, including for iteration limits and
        // progress output; grouping such points obscures the actual work.
        // Preconditioning changes only curvature surrogates, not the quotient
        // maps. Use one chart for stationarity, reporting, and the next solve.
        const bool terminal_checked = accepted.converged_geometry.has_value();
        OrbitalChart next_geometry =
            terminal_checked ? std::move(*accepted.converged_geometry)
                             : build_orbital_geometry(*objective,
                                                      parameter_view,
                                                      OrbitalChart::Coordinates::Unwhitened);
        auto next_projected = next_geometry.project_gradient(gradient);
        final_gradient_l2 = next_geometry.coordinate_dual_norm(next_projected.reduced_gradient);
        if (options.ciah_iteration_callback) {
            const NonredundantRetractionMetric next_metric(
                next_geometry, parameter_view, objective->input().orbital_preparation_input);
            iteration_record.accepted_physical_gradient_norm =
                next_metric.dual_norm(next_projected.reduced_gradient);
        }
        const double energy_change = energy - previous_energy;
        previous_energy = energy;
        if (std::abs(energy_change) < options.energy_tolerance &&
            (terminal_checked ||
             orbital_gradient_converged(objective->input().orbital_preparation_input,
                                        parameter_view,
                                        next_geometry,
                                        next_projected.reduced_gradient,
                                        options.gradient_tolerance))) {
            seek_curvature = true;
        }
        chart = std::move(next_geometry);
        projected = std::move(next_projected);

        ++run_result.n_iterations;
        iteration_record.trust_ratio =
            iteration_record.predicted_reduction > 0.0
                ? iteration_record.actual_reduction / iteration_record.predicted_reduction
                : -std::numeric_limits<double>::infinity();
        result->ciah_iteration_trace.push_back(iteration_record);
        if (options.ciah_iteration_callback) {
            options.ciah_iteration_callback(iteration_record);
        }
        sync_result_from_objective(*objective, result);
        record_accepted_iteration_snapshot(objective,
                                           run_result.n_iterations,
                                           options,
                                           result,
                                           nullptr,
                                           &projected.reduced_gradient,
                                           &chart);
        if (result->converged)
            break;
    }

    result->final_projected_gradient_inf_norm =
        chart.reference_gradient_inf_norm(projected.reduced_gradient);
    result->final_projected_gradient_l2_norm = final_gradient_l2;
    run_result.final_gradient_l2_norm = final_gradient_l2;
    run_result.final_projected_gradient_ready = true;
    return run_result;
}

} // namespace xmvb::vb::optimizer_detail
