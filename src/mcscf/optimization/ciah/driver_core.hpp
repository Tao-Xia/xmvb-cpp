#pragma once

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "core/timing.hpp"
#include "mcscf/optimization/ciah/solver.hpp"
#include "mcscf/optimization/trace.hpp"

namespace xmvb::mcscf::detail {

// PySCF newton_casscf.update_orb_ci policies for approximate microsteps.
inline constexpr double ciah_gradient_trust = 3.0;
inline constexpr int ciah_ah_start_cycle = 3;
inline constexpr int ciah_keyframe_interval = 5;
inline constexpr int ciah_base_micro_cycles = 10;

/** One policy loop for host and device microsteps; Backend owns vector operations. */
template <class Backend, class Workspace>
auto run_ciah_microsteps_core(Backend backend,
                              Workspace& workspace,
                              const typename Backend::Vector& initial_gradient,
                              double energy,
                              double outer_gradient_tolerance) {
    using Vector = typename Backend::Vector;
    auto result = backend.make_result(energy, initial_gradient);
    auto& displacement = result.displacement;
    auto& hessian_displacement = result.hessian_displacement;
    auto& predicted_gradient = result.predicted_gradient;
    auto& last_step = result.last_step;
    auto& candidate = result.candidate;
    auto& energy_model = result.energy_model;
    auto& keyframes = result.keyframes;
    using Candidate = std::remove_cvref_t<decltype(candidate)>;
    CiahStepPathT<Candidate, Backend> path(backend);
    double keyframe_norm = backend.coordinate_norm(initial_gradient);
    const double micro_gradient_tolerance = 0.3 * outer_gradient_tolerance;
    int since_keyframe = 0;
    bool audit_next_keyframe_step = false;
    bool audited_keyframe_step = false;
    bool audited_origin_step = false;
    bool audited_keyframe = false;
    while (true) {
        bool replayed = path.pending();
        const double rhs_norm = backend.coordinate_norm(predicted_gradient);
        double replay_residual = 0.0;
        if (replayed) {
            Vector remaining_residual = path.residual_at(predicted_gradient);
            replay_residual = backend.coordinate_norm(remaining_residual);
            if (!(replay_residual <= rhs_norm / ciah_gradient_trust)) {
                diagnostics::ciah_trace(
                    "ciah_replay_reset",
                    "basis=%d,remaining_fraction=%.17g,current_residual=%.17g,target=%.17g",
                    candidate.basis_size,
                    path.remaining_fraction(),
                    replay_residual,
                    rhs_norm / ciah_gradient_trust);
                path.clear();
                workspace.reset_rhs();
                replayed = false;
            }
        }
        if (!replayed) {
            candidate = backend.next(workspace, predicted_gradient);
            ++result.micro_iterations;
        }
        const bool forcing_resolved =
            replayed || ciah_result_resolved(candidate, rhs_norm, 1.0 / ciah_gradient_trust);
        const bool ready = replayed || candidate.converged || candidate.exhaustion ||
                           (candidate.basis_size >= ciah_ah_start_cycle &&
                            (forcing_resolved || candidate.new_hvp_actions != 0));
        if (!ready)
            continue;

        if (!replayed)
            path.set(candidate);
        else
            candidate.converged = false;
        double largest = backend.maximum_component(path.direction());
        largest *= path.remaining_fraction();
        const double step_scale =
            largest > ciah_step_max_component ? ciah_step_max_component / largest : 1.0;
        auto [step, image] = path.take(step_scale);
        if (step_scale < 1.0)
            candidate.converged = false;
        backend.audit_step("micro_origin", displacement, step, result.micro_steps + 1);
        backend.audit_step("micro_increment", displacement, step, result.micro_steps + 1);
        if (!audited_origin_step && backend.is_zero(displacement)) {
            backend.audit_origin(displacement, step, image, predicted_gradient);
            audited_origin_step = true;
        }
        if (audit_next_keyframe_step) {
            backend.audit_resumed(
                displacement, step, image, predicted_gradient, candidate.basis_size);
            audit_next_keyframe_step = false;
            audited_keyframe_step = true;
        }
        Vector previous_gradient = backend.clone(predicted_gradient);
        Vector previous_step = backend.clone(last_step);
        backend.axpy(displacement, 1.0, step);
        backend.axpy(hessian_displacement, 1.0, image);
        energy_model.add(step, image);
        backend.axpy(predicted_gradient, 1.0, image);
        last_step = backend.clone(step);
        ++result.micro_steps;
        ++since_keyframe;
        const double predicted_norm = backend.coordinate_norm(predicted_gradient);
        if (diagnostics::ciah_trace_enabled())
            diagnostics::ciah_trace(
                "ciah_micro",
                "basis=%d,micro=%d,omega=%.17g,alpha=%.17g,residual=%.17g,gradient=%.17g,step_"
                "scale=%.17g,max_component=%.17g,orbital_residual_euclidean=%.17g,structure_"
                "residual_euclidean=%.17g,rhs_norm=%.17g,forcing_target=%.17g,forcing_resolved=%d,"
                "replayed=%d,new_hvp=%d,remaining_fraction=%.17g,residual_rhs_current=%d",
                candidate.basis_size,
                result.micro_steps,
                candidate.eigenvalue,
                candidate.reference_component,
                candidate.residual_norm,
                predicted_norm,
                step_scale,
                largest,
                backend.orbital_residual_norm(candidate.residual),
                backend.structure_residual_norm(candidate.residual),
                rhs_norm,
                rhs_norm / ciah_gradient_trust,
                static_cast<int>(forcing_resolved),
                static_cast<int>(replayed),
                replayed ? 0 : candidate.new_hvp_actions,
                path.remaining_fraction(),
                static_cast<int>(!replayed));
        if (result.micro_steps > 3 && predicted_norm > ciah_gradient_trust * keyframe_norm) {
            backend.axpy(displacement, -1.0, step);
            backend.axpy(hessian_displacement, -1.0, image);
            energy_model.remove(step, image);
            predicted_gradient = std::move(previous_gradient);
            last_step = std::move(previous_step);
            ++result.rejected_trial_count;
            result.stop_reason = "gradient_growth";
            break;
        }
        const int micro_limit = std::max(
            ciah_base_micro_cycles,
            ciah_base_micro_cycles - static_cast<int>(2.0 * std::log(keyframe_norm + 1.0e-7)));
        if (result.micro_steps >= micro_limit || predicted_norm < 0.3 * micro_gradient_tolerance ||
            candidate.exhaustion) {
            result.stop_reason = predicted_norm < 0.3 * micro_gradient_tolerance ? "micro_gradient"
                                 : candidate.exhaustion ? "subspace_exhaustion"
                                                        : "micro_budget";
            diagnostics::ciah_trace("ciah_micro_stop",
                                    "reason=%s,micro=%d,budget=%d,basis=%d,keyframe_norm=%.17g,"
                                    "predicted=%.17g,target=%.17g,residual=%.17g,ah_converged=%d",
                                    result.stop_reason,
                                    result.micro_steps,
                                    micro_limit,
                                    candidate.basis_size,
                                    keyframe_norm,
                                    predicted_norm,
                                    0.3 * micro_gradient_tolerance,
                                    candidate.residual_norm,
                                    static_cast<int>(candidate.converged));
            break;
        }

        const Vector& keyframe_step = energy_model.displacement();
        Vector metric_step = backend.metric_action(keyframe_step);
        const double length = std::sqrt(std::max(0.0, backend.dot(keyframe_step, metric_step)));
        const double interval = std::max<double>(
            ciah_keyframe_interval, ciah_keyframe_interval - std::log(length + 1.0e-7));
        if (since_keyframe >= interval || predicted_norm < keyframe_norm / ciah_gradient_trust) {
            backend.audit_step("keyframe", displacement, displacement, result.micro_steps);
            auto exact = backend.evaluate_keyframe(displacement);
            ++result.keyframe_evaluations;
            if (!audited_keyframe) {
                backend.audit_keyframe(displacement, hessian_displacement, exact);
                audited_keyframe = true;
            }
            const double actual_norm = backend.coordinate_norm(exact.gradient);
            Vector correction = backend.clone(exact.gradient);
            backend.axpy(correction, -1.0, predicted_gradient);
            const double mismatch_norm = backend.coordinate_norm(correction);
            const bool small_gradient =
                actual_norm < ciah_gradient_trust * micro_gradient_tolerance;
            const bool mismatch_small = mismatch_norm < ciah_gradient_trust * predicted_norm;
            if (diagnostics::ciah_event_trace_enabled()) {
                const double captured = backend.projected_dual_norm(workspace, correction);
                const double secant = backend.dot(correction, keyframe_step);
                diagnostics::ciah_trace("ciah_keyframe_secant",
                                        "mismatch_norm=%.17g,captured=%.17g,displacement_norm=%."
                                        "17g,secant=%.17g,mismatch_small=%d",
                                        mismatch_norm,
                                        captured,
                                        length,
                                        secant,
                                        static_cast<int>(mismatch_small));
            }
            const bool keep_keyframe = small_gradient || mismatch_small;
            keyframes.save(exact.energy, displacement, hessian_displacement, last_step, candidate);
            diagnostics::ciah_trace("ciah_keyframe",
                                    "actual=%.17g,predicted=%.17g,mismatch=%.17g,mismatch_small=%d,"
                                    "energy=%.17g,physical_energy=%.17g,accepted=%d",
                                    actual_norm,
                                    predicted_norm,
                                    mismatch_norm,
                                    static_cast<int>(mismatch_small),
                                    exact.energy,
                                    exact.physical_energy,
                                    static_cast<int>(keep_keyframe));
            if (diagnostics::ciah_event_trace_enabled()) {
                const double actual_orbital = backend.orbital_norm(exact.gradient);
                const double predicted_orbital = backend.orbital_norm(predicted_gradient);
                const double mismatch_orbital = backend.orbital_norm(correction);
                diagnostics::ciah_trace(
                    "ciah_keyframe_blocks",
                    "actual_orbital=%.17g,actual_structure_squared=%.17g,predicted_orbital=%.17g,"
                    "predicted_structure_squared=%.17g,mismatch_orbital=%.17g,mismatch_structure_"
                    "squared=%.17g",
                    actual_orbital,
                    std::max(0.0, actual_norm * actual_norm - actual_orbital * actual_orbital),
                    predicted_orbital,
                    std::max(0.0,
                             predicted_norm * predicted_norm -
                                 predicted_orbital * predicted_orbital),
                    mismatch_orbital,
                    std::max(0.0,
                             mismatch_norm * mismatch_norm - mismatch_orbital * mismatch_orbital));
            }
            if (keep_keyframe) {
                if (energy_model.descent_mismatch(exact.energy)) {
                    result.energy_model_stopped = true;
                    result.stop_reason = "keyframe_energy_model";
                    diagnostics::ciah_trace("ciah_keyframe_energy_model",
                                            "anchor=%.17g,predicted=%.17g,actual=%.17g,stop=1",
                                            energy_model.anchor_energy(),
                                            energy_model.energy(),
                                            exact.energy);
                    break;
                }
                xmvb::profiling::ScopedPhase refresh_phase(xmvb::profiling::Phase::CiahRefresh);
                predicted_gradient = backend.clone(exact.gradient);
                energy_model.refresh(exact.energy, exact.gradient);
                path.clear();
                if (diagnostics::ciah_trace_enabled()) {
                    Vector origin_prediction = backend.clone(initial_gradient);
                    backend.scal(origin_prediction, -1.0);
                    const double origin_value =
                        backend.dot(origin_prediction, displacement) -
                        0.5 * backend.dot(displacement, hessian_displacement);
                    diagnostics::ciah_trace("ciah_model_refresh",
                                            "energy=%.17g,source_energy=%.17g,anchored_prediction=%"
                                            ".17g,origin_prediction=%.17g",
                                            exact.energy,
                                            energy,
                                            energy - exact.energy,
                                            origin_value);
                }
                keyframe_norm = actual_norm;
                since_keyframe = 0;
                audit_next_keyframe_step = !audited_keyframe_step && backend.has_audit_resumed();
            } else {
                backend.axpy(displacement, -1.0, step);
                backend.axpy(hessian_displacement, -1.0, image);
                energy_model.remove(step, image);
                predicted_gradient = std::move(previous_gradient);
                last_step = std::move(previous_step);
                ++result.rejected_trial_count;
                result.stop_reason = "keyframe_mismatch";
                break;
            }
        }
        if (!path.pending())
            workspace.reset_rhs();
    }

    if (diagnostics::ciah_event_trace_enabled()) {
        Vector frozen_gradient = backend.clone(initial_gradient);
        backend.axpy(frozen_gradient, 1.0, hessian_displacement);
        Vector correction = backend.clone(predicted_gradient);
        backend.axpy(correction, -1.0, frozen_gradient);
        diagnostics::ciah_trace(
            "ciah_micro_end",
            "reason=%s,source_coordinate_norm=%.17g,predicted_coordinate_norm=%.17g,frozen_model_"
            "coordinate_norm=%.17g,keyframe_correction_norm=%.17g,last_ritz=%.17g",
            result.stop_reason,
            backend.coordinate_norm(initial_gradient),
            backend.coordinate_norm(predicted_gradient),
            backend.coordinate_norm(frozen_gradient),
            backend.coordinate_norm(correction),
            candidate.residual_norm);
    }
    return result;
}

} // namespace xmvb::mcscf::detail
