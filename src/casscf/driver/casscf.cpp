#include "casscf/driver/casscf.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include "casscf/derivatives/coupled.hpp"
#include "mcscf/ci/strings.hpp"
#include "mcscf/optimization/problem.hpp"
#include "mcscf/optimization/preconditioners/coarse.hpp"
#include "mcscf/optimization/trace.hpp"

namespace xmvb::casscf {
namespace {
using Clock = std::chrono::steady_clock;

Eigen::MatrixXd initial_ci(int nactive, int electrons, int spin, std::size_t workspace_bytes) {
    if (nactive < 1 || nactive > 63 || electrons < 1 || electrons > 2 * nactive || spin < 0 ||
        spin > std::min(electrons, 2 * nactive - electrons) || (electrons - spin) % 2)
        throw std::invalid_argument("invalid complete-CAS electron/spin space");
    const auto na = mcscf::string_count(nactive, (electrons + spin) / 2);
    const auto nb = mcscf::string_count(nactive, (electrons - spin) / 2);
    if (!na || !nb || na > static_cast<std::size_t>(std::numeric_limits<int>::max()) / nb ||
        na > workspace_bytes / sizeof(double) / nb)
        throw std::length_error("initial CAS coefficient carrier exceeds workspace");
    // Nested alpha/beta occupations make the first determinant a pure highest-Ms
    // state. Davidson subsequently finds the lowest root in this total-spin sector.
    Eigen::MatrixXd c = Eigen::MatrixXd::Zero(na, nb);
    c(0, 0) = 1;
    return c;
}
} // namespace

CasscfResult solve_casscf(const integrals::AoIntegralInput& ao,
                          const Eigen::MatrixXd& orbitals,
                          int ncore,
                          int nactive,
                          int electrons,
                          int spin,
                          double nuclear,
                          const CasscfOptions& options) {
    if (!(options.energy_tolerance > 0) || !(options.gradient_tolerance > 0) ||
        !std::isfinite(options.energy_tolerance) || !std::isfinite(options.gradient_tolerance) ||
        options.max_iterations < 1 || options.ci.n_roots != 1)
        throw std::invalid_argument("CASSCF requires positive tolerances and one CI root");
    CasscfResult result;
    const auto prepare = [&](const Eigen::MatrixXd& u, const Eigen::MatrixXd& c) {
        auto point =
            std::make_unique<FixedCiPoint>(ao, u, ncore, nactive, nuclear, electrons, spin, c);
        const auto ci = solve_casci(point->sigma(),
                                    point->spin_space(),
                                    point->integrals().active.one_electron,
                                    spin,
                                    point->core_energy(),
                                    options.ci,
                                    c);
        result.ci_actions += ci.active_ci.block_actions;
        const auto& values = ci.active_ci.eigenpairs.eigenvector_matrix;
        Eigen::MatrixXd relaxed =
            Eigen::Map<const Eigen::MatrixXd>(values.data(), c.rows(), c.cols());
        // Fix the arbitrary eigenvector sign against the finite CI predictor.
        // This is a phase choice, needed when recycling coefficient directions.
        if ((relaxed.array() * c.array()).sum() < 0)
            relaxed = -relaxed;
        point->set_coefficients(relaxed);
        return point;
    };
    auto point =
        prepare(orbitals, initial_ci(nactive, electrons, spin, options.ci.workspace_bytes));
    result.initial_energy = point->energy();
    const RotationChart chart(ao.n_basis_functions, ncore, nactive);
    const auto gradient_norm = [&](const FixedCiPoint& value) {
        const double orbital = chart
                                   .pullback(value.orbitals(),
                                             value.orbital_gradient_raw(),
                                             Eigen::VectorXd::Zero(chart.size()))
                                   .norm();
        return std::hypot(orbital, std::sqrt(2.0) * value.ci_residual().norm());
    };
    result.initial_gradient_norm = gradient_norm(*point);

    if (chart.size() == 0) {
        result.gradient_norm = std::sqrt(2.0) * point->ci_residual().norm();
        result.converged = result.gradient_norm < options.gradient_tolerance;
        result.termination_reason = result.converged ? "complete_casci" : "ci_residual_tolerance";
        result.energy = point->energy();
        result.orbitals = point->orbitals();
        result.coefficients = point->coefficients();
        return result;
    }

    Eigen::VectorXd direction_guess;
    for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
        const auto start = Clock::now();
        const int ci_before = result.ci_actions;
        const double energy = point->energy();
        CoupledModel model(*point, ao, ncore, nactive);
        const Eigen::Index no = model.orbital_size();
        CoupledImages images;
        const auto action = [&](const Eigen::VectorXd& direction) {
            ++result.hvp_actions;
            images = model.apply_split(direction);
            return Eigen::VectorXd(images.orbital + images.ci);
        };
        const auto norm = [](const Eigen::VectorXd& vector) { return vector.norm(); };
        const auto identity = [](const Eigen::VectorXd& vector) -> Eigen::VectorXd {
            return vector;
        };
        mcscf::QuadraticProblem problem(model.gradient(), action, identity, norm);
        mcscf::CiahOptions inner;
        mcscf::CoarsePreconditioner coarse(static_cast<int>(model.size()),
                                           2 * inner.max_cycle,
                                           inner.lindep,
                                           inner.operator_relative_accuracy);
        inner.coordinate_norm = norm;
        inner.residual_preconditioner = [&](const Eigen::VectorXd& r, double shift) {
            return coarse.apply(r, shift, [&](const Eigen::VectorXd& value) {
                return model.precondition(value, shift - 1e-8);
            });
        };
        inner.observe_images = [&](const Eigen::VectorXd& direction, const Eigen::VectorXd&) {
            // The physical action already computed both split images. The separate
            // block samples add coupling information beyond the mixed Ritz vectors.
            Eigen::VectorXd orbital = Eigen::VectorXd::Zero(model.size());
            orbital.head(no) = direction.head(no);
            coarse.observe(orbital, images.orbital, orbital);
            Eigen::VectorXd ci = Eigen::VectorXd::Zero(model.size());
            ci.tail(model.size() - no) = direction.tail(model.size() - no);
            coarse.observe(ci, images.ci, ci);
        };
        inner.residual_direction = identity;
        Eigen::VectorXd seed = direction_guess.size() != 0
                                   ? model.project_direction(direction_guess)
                                   : -model.precondition(model.gradient(), 0);
        mcscf::CiahWorkspace workspace(problem, std::move(seed), inner);
        mcscf::CiahMicroCallbacks callbacks;
        callbacks.coordinate_norm = norm;
        callbacks.metric_action = identity;
        callbacks.maximum_component = [&](const Eigen::VectorXd& vector) {
            return model.maximum_component(vector);
        };
        callbacks.orbital_size = model.orbital_size();
        callbacks.orbital_coordinate_norm = norm;
        callbacks.evaluate_keyframe = [&](const Eigen::VectorXd& displacement) {
            FixedCiPoint trial(ao,
                               model.orbitals(displacement),
                               ncore,
                               nactive,
                               nuclear,
                               electrons,
                               spin,
                               model.coefficients(displacement));
            return model.keyframe(trial, displacement);
        };
        const int hvp_before = result.hvp_actions;
        auto micro = mcscf::run_ciah_microsteps(
            workspace, model.gradient(), energy, options.gradient_tolerance, callbacks);

        // Physical macro acceptance: relax CI in the finite rotated orbitals.
        // The small AH space remains available for an origin descent/curvature
        // restart; this uses only its existing H images, never another HVP.
        Eigen::VectorXd displacement = micro.displacement;
        Eigen::VectorXd last_step = micro.last_step;
        bool restarted = false;
        const auto restart_origin = [&]() {
            if (restarted)
                return false;
            restarted = true;
            const auto ray = workspace.origin_ray(model.gradient());
            if (!ray)
                return false;
            const double largest = model.maximum_component(ray->step);
            if (!(largest > 0))
                return false;
            const double scale = largest > mcscf::ciah_max_component * ray->reference_component
                                     ? mcscf::ciah_max_component / largest
                                     : 1 / ray->reference_component;
            displacement = scale * ray->step;
            last_step = displacement;
            return true;
        };
        std::unique_ptr<FixedCiPoint> accepted;
        double accepted_norm = 0;
        while (true) {
            auto trial = prepare(model.orbitals(displacement), model.coefficients(displacement));
            const double change = trial->energy() - energy;
            if (change < 0 || std::abs(change) < options.energy_tolerance) {
                accepted_norm = gradient_norm(*trial);
                if (change < 0 || accepted_norm < options.gradient_tolerance) {
                    accepted = std::move(trial);
                    break;
                }
            }
            if (micro.keyframes.available()) {
                auto checkpoint = micro.keyframes.take();
                displacement = std::move(checkpoint.displacement);
                last_step = std::move(checkpoint.last_step);
                continue;
            }
            if ((micro.energy_model_stopped || model.gradient().dot(displacement) >= 0) &&
                restart_origin())
                continue;
            displacement *= 0.5;
            last_step *= 0.5;
            if (displacement.norm() <= std::numeric_limits<double>::epsilon()) {
                if (restart_origin())
                    continue;
                result.termination_reason = "ciah_macro_energy_rejected";
                break;
            }
        }
        if (!accepted)
            break;
        const double change = accepted->energy() - energy;
        result.iterations.push_back({iteration,
                                     result.hvp_actions - hvp_before,
                                     micro.keyframe_evaluations,
                                     result.ci_actions - ci_before,
                                     micro.micro_iterations,
                                     accepted->energy(),
                                     change,
                                     accepted_norm,
                                     std::chrono::duration<double>(Clock::now() - start).count()});
        if (options.observe_iteration)
            options.observe_iteration(result.iterations.back());
        mcscf::diagnostics::ciah_trace("casscf_macro",
                                       "iteration=%d,energy=%.17g,de=%.17g,gradient=%.17g,hvp=%d,"
                                       "ci_actions=%d,coarse=%d,micro=%d,stop=%s",
                                       iteration,
                                       accepted->energy(),
                                       change,
                                       accepted_norm,
                                       result.hvp_actions - hvp_before,
                                       result.ci_actions - ci_before,
                                       coarse.size(),
                                       micro.micro_steps,
                                       micro.stop_reason);
        // As in PySCF's dr0 and VB's last-step predictor, recycle only a direction.
        // MO labels are unchanged (no canonicalization); the next model projects
        // its CI tail at the relaxed point and computes all curvature anew.
        direction_guess = std::move(last_step);
        point = std::move(accepted);
        if (std::abs(change) < options.energy_tolerance &&
            accepted_norm < options.gradient_tolerance) {
            result.converged = true;
            result.termination_reason = "ciah_gradient_tolerance";
            break;
        }
    }
    if (result.termination_reason.empty())
        result.termination_reason = "maximum_iterations";
    result.energy = point->energy();
    result.gradient_norm = gradient_norm(*point);
    result.orbitals = point->orbitals();
    result.coefficients = point->coefficients();
    return result;
}

} // namespace xmvb::casscf
