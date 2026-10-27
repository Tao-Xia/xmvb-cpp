#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <utility>

#include "mcscf/optimization/ciah/solver.hpp"

namespace xmvb::mcscf {

namespace detail {
/** @brief Interior minimum of a quadratic fitted to a relaxed energy ray. */
inline std::optional<double>
ciah_ray_fraction(double source, double trial, double slope, double energy_resolution) {
    const double curvature = trial - source - slope;
    if (!(slope < 0.0 && curvature > 0.0))
        return std::nullopt;
    const double fraction = -0.5 * slope / curvature;
    const double remaining = 1.0 - fraction;
    const double gain = curvature * remaining * remaining;
    const double roundoff = std::numeric_limits<double>::epsilon() *
                            (std::abs(source) + std::abs(trial) + std::abs(slope));
    if (fraction > 0.0 && fraction < 1.0 && gain > std::max(roundoff, energy_resolution))
        return fraction;
    return std::nullopt;
}
} // namespace detail

struct CiahMacroObservation {
    double scale = 1.0, linear = 0.0, quadratic = 0.0;
    double predicted = 0.0, actual = 0.0, trial_energy = 0.0;
    bool accepted = false, terminal = false;
};

/** Physical callbacks; the acceptance policy does not depend on vector storage. */
template <class Ops, class Trial> struct CiahMacroCallbacks {
    using Vector = typename Ops::Vector;
    std::function<Trial(const Vector&, double)> evaluate;
    std::function<double(const Trial&)> energy;
    /** @brief Origin derivative of the CI-relaxed trial ray, without its CI seed. */
    std::function<double(const Vector&)> relaxed_slope;
    std::function<bool(Trial&)> terminal;
    std::function<double(const Vector&)> step_norm;
    std::function<double(const Vector&)> maximum_component;
    std::function<bool(const Vector&)> changes_parameters;
    std::function<void(const CiahMacroObservation&)> observe;
};

template <class Trial> struct CiahMacroResult {
    std::optional<Trial> trial;
    const char* reason = "ciah_macro_energy_rejected";
    double scale = 1.0, predicted_reduction = 0.0, actual_reduction = 0.0;
    double residual_norm = 0.0, eigenvalue = 0.0;
    int rejected_trials = 0;
    bool ritz_converged = false, keyframe_backtracked = false, origin_restarted = false;
};

/** Shared nonlinear CIAH globalization, including retained true keyframes and
 * an origin-root restart in the existing AH span. No physical action is added
 * by restart; only a trial's own gradient can accept a terminal non-descent.
 * On success micro's vectors are the accepted, scaled displacement/images.
 */
template <class Ops, class Workspace, class Micro, class Trial>
CiahMacroResult<Trial> run_ciah_macro(Workspace& workspace,
                                      Micro& micro,
                                      const typename Ops::Vector& initial_gradient,
                                      double source_energy,
                                      double energy_tolerance,
                                      double minimum_step_size,
                                      Ops ops,
                                      const CiahMacroCallbacks<Ops, Trial>& callbacks) {
    auto& displacement = micro.displacement;
    auto& image = micro.hessian_displacement;
    auto& last_step = micro.last_step;
    CiahMacroResult<Trial> result;
    double final_reduction = source_energy - micro.energy_model.energy();
    double linear = ops.dot(initial_gradient, displacement);
    double quadratic = ops.dot(displacement, image);
    result.residual_norm = micro.candidate.residual_norm;
    result.eigenvalue = micro.candidate.eigenvalue;
    result.ritz_converged = micro.candidate.converged;
    const auto restart_origin = [&]() {
        if (result.origin_restarted)
            return false;
        auto ray = workspace.origin_ray(initial_gradient);
        if (!ray)
            return false;
        const double largest = callbacks.maximum_component(ray->step);
        if (!(largest > 0.0))
            return false;
        const double cap = detail::ciah_step_max_component;
        const double factor = largest > cap * ray->reference_component
                                  ? cap / largest
                                  : 1.0 / ray->reference_component;
        displacement = ops.clone(ray->step);
        ops.scal(displacement, factor);
        image = ops.clone(ray->hessian_step);
        ops.scal(image, factor);
        last_step = ops.clone(displacement);
        linear = ops.dot(initial_gradient, displacement);
        quadratic = ops.dot(displacement, image);
        final_reduction = -linear - 0.5 * quadratic;
        result.residual_norm = ray->residual_norm;
        result.eigenvalue = ray->eigenvalue;
        result.ritz_converged = false;
        result.scale = 1.0;
        result.origin_restarted = true;
        return true;
    };
    if (!callbacks.changes_parameters(displacement) && !micro.keyframes.available() &&
        !restart_origin()) {
        result.reason = "ciah_zero_macro_step";
        return result;
    }
    while (true) {
        auto trial = callbacks.evaluate(displacement, result.scale);
        double trial_energy = callbacks.energy(trial);
        double model_linear =
            result.scale == 1.0 ? -final_reduction - 0.5 * quadratic : result.scale * linear;
        double prediction = -model_linear - 0.5 * result.scale * result.scale * quadratic;
        double reduction = source_energy - trial_energy;
        bool keep = reduction > 0.0, terminal = false;
        if (!keep && std::abs(reduction) < energy_tolerance) {
            terminal = true;
            keep = callbacks.terminal(trial);
        }
        bool observed = false;
        // E(s*t) = E0 + d*t + a*t^2 interpolates the relaxed endpoint and
        // its origin slope d=s*E'(0). Only a convex interpolant with an interior
        // minimum proposes a shorter candidate. It is compared to the full
        // trial, never substituted on the strength of a model prediction alone.
        // The CI seed is not a coordinate of this relaxed energy function.
        // Its predicted and actual benefit must exceed the existing energy
        // resolution; this adds no independent line-search tolerance.
        if (keep && !terminal) {
            const double slope = result.scale * callbacks.relaxed_slope(displacement);
            if (const auto fraction = detail::ciah_ray_fraction(
                    source_energy, trial_energy, slope, energy_tolerance)) {
                const double refined_scale = result.scale * *fraction;
                auto refined = callbacks.evaluate(displacement, refined_scale);
                const double refined_energy = callbacks.energy(refined);
                // A sub-resolution energy difference cannot justify replacing
                // the already valid candidate and perturbing its recycled step.
                const bool better = trial_energy - refined_energy > energy_tolerance;
                if (callbacks.observe)
                    callbacks.observe({result.scale,
                                       model_linear,
                                       result.scale * result.scale * quadratic,
                                       prediction,
                                       reduction,
                                       trial_energy,
                                       !better,
                                       false});
                if (better) {
                    trial = std::move(refined);
                    trial_energy = refined_energy;
                    result.scale = refined_scale;
                    model_linear = result.scale * linear;
                    prediction = -model_linear - 0.5 * result.scale * result.scale * quadratic;
                    reduction = source_energy - trial_energy;
                } else {
                    if (callbacks.observe)
                        callbacks.observe({refined_scale,
                                           refined_scale * linear,
                                           refined_scale * refined_scale * quadratic,
                                           -refined_scale * linear -
                                               0.5 * refined_scale * refined_scale * quadratic,
                                           source_energy - refined_energy,
                                           refined_energy,
                                           false,
                                           false});
                    ++result.rejected_trials;
                    observed = true;
                }
            }
        }
        if (!observed && callbacks.observe)
            callbacks.observe({result.scale,
                               model_linear,
                               result.scale * result.scale * quadratic,
                               prediction,
                               reduction,
                               trial_energy,
                               keep,
                               terminal});
        if (keep) {
            result.predicted_reduction = prediction;
            result.actual_reduction = reduction;
            result.trial.emplace(std::move(trial));
            result.reason = "accepted";
            ops.scal(displacement, result.scale);
            ops.scal(image, result.scale);
            ops.scal(last_step, result.scale);
            return result;
        }
        ++result.rejected_trials;
        if (micro.keyframes.available()) {
            auto point = micro.keyframes.take();
            displacement = std::move(point.displacement);
            image = std::move(point.image);
            last_step = std::move(point.last_step);
            result.residual_norm = point.residual_norm;
            result.eigenvalue = point.eigenvalue;
            result.ritz_converged = point.converged;
            final_reduction = source_energy - point.energy;
            linear = ops.dot(initial_gradient, displacement);
            quadratic = ops.dot(displacement, image);
            result.scale = 1.0;
            result.keyframe_backtracked = true;
            continue;
        }
        if ((micro.energy_model_stopped || linear >= 0.0) && restart_origin())
            continue;
        result.scale *= 0.5;
        if (result.scale * callbacks.step_norm(displacement) <= minimum_step_size) {
            if (restart_origin())
                continue;
            return result;
        }
    }
}

} // namespace xmvb::mcscf
