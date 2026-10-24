#pragma once

#include <functional>

#include <Eigen/Core>

#include "mcscf/optimization/ciah/solver.hpp"

namespace xmvb::mcscf {

inline constexpr double ciah_max_component = detail::ciah_step_max_component;

/** Exact nonlinear observation in the immutable origin coordinates. */
struct CiahKeyframe {
    Eigen::VectorXd gradient;
    double energy = 0.0;
    double physical_energy = 0.0;
};

/** Problem-specific geometry and optional diagnostics for the CIAH microsteps. */
struct CiahMicroCallbacks {
    std::function<double(const Eigen::VectorXd&)> coordinate_norm;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> metric_action;
    std::function<double(const Eigen::VectorXd&)> maximum_component;
    std::function<CiahKeyframe(const Eigen::VectorXd&)> evaluate_keyframe;
    /** Orbital prefix of the coupled coordinate vector, for split traces. */
    Eigen::Index orbital_size = 0;
    std::function<double(const Eigen::VectorXd&)> orbital_coordinate_norm;
    std::function<void(const char*, const Eigen::VectorXd&, const Eigen::VectorXd&, int)>
        audit_step;
    std::function<void(const Eigen::VectorXd&,
                       const Eigen::VectorXd&,
                       const Eigen::VectorXd&,
                       const Eigen::VectorXd&)>
        audit_origin;
    std::function<void(const Eigen::VectorXd&,
                       const Eigen::VectorXd&,
                       const Eigen::VectorXd&,
                       const Eigen::VectorXd&,
                       int)>
        audit_resumed;
    std::function<void(const Eigen::VectorXd&, const Eigen::VectorXd&, const CiahKeyframe&)>
        audit_keyframe;
};

/** Kept microstep path and exact checkpoints for caller-owned macro acceptance. */
struct CiahMicroResult {
    CiahMicroResult(double energy, const Eigen::VectorXd& initial_gradient)
        : displacement(Eigen::VectorXd::Zero(initial_gradient.size())),
          hessian_displacement(displacement), predicted_gradient(initial_gradient),
          last_step(displacement), energy_model(energy, initial_gradient), keyframes(energy) {}

    Eigen::VectorXd displacement, hessian_displacement, predicted_gradient, last_step;
    CiahResult candidate;
    CiahEnergyModel energy_model;
    CiahKeyframes keyframes;
    int micro_steps = 0;
    int micro_iterations = 0;
    int keyframe_evaluations = 0;
    int rejected_trial_count = 0;
    bool energy_model_stopped = false;
    const char* stop_reason = "unset";
};

/** Advance a single frozen-Hessian CIAH microstep sequence. */
CiahMicroResult run_ciah_microsteps(CiahWorkspace& workspace,
                                    const Eigen::VectorXd& initial_gradient,
                                    double energy,
                                    double outer_gradient_tolerance,
                                    const CiahMicroCallbacks& callbacks);

} // namespace xmvb::mcscf
