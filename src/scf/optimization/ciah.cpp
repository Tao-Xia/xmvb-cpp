#include "scf/optimization/ciah.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include "mcscf/optimization/ciah/solver.hpp"
#include "mcscf/optimization/preconditioners/coarse.hpp"
#include "mcscf/optimization/trace.hpp"
#include "scf/mixing/diis.hpp"

namespace xmvb::scf {

std::optional<OrbitalTrial> ciah_step(const OrbitalModel& model,
                                      const EvaluateUnrestricted& evaluate,
                                      const SpinMatrices& reference_density,
                                      const UnrestrictedEvaluation& reference,
                                      const Eigen::Ref<const Eigen::MatrixXd>& overlap,
                                      const Eigen::VectorXd& history_direction,
                                      int* response_actions) {
    const auto& g = model.gradient();
    if (g.size() == 0 || !(g.stableNorm() > 0.0))
        return std::nullopt;
    const auto identity = [](const Eigen::VectorXd& p) -> Eigen::VectorXd { return p; };
    const auto norm = [](const Eigen::VectorXd& p) { return p.stableNorm(); };
    Eigen::VectorXd last_image;
    mcscf::QuadraticProblem problem(
        g,
        [&](const Eigen::VectorXd& p) {
            ++*response_actions;
            last_image = model.apply(p);
            mcscf::diagnostics::ciah_trace(
                "scf_response_direction",
                "gradient_projection=%.17g,curvature=%.17g,image_norm=%.17g",
                p.dot(g),
                p.dot(last_image),
                last_image.stableNorm());
            return last_image;
        },
        identity,
        norm);
    mcscf::CiahOptions options;
    options.tolerance = std::min(options.tolerance, g.squaredNorm());
    options.coordinate_norm = norm;
    options.residual_direction = identity;
    mcscf::CoarsePreconditioner coarse(
        g.size(), options.max_cycle, options.lindep, options.operator_relative_accuracy);
    options.observe_images = [&](const Eigen::VectorXd& p, const Eigen::VectorXd& metric) {
        coarse.observe(p, last_image, metric);
    };
    options.residual_preconditioner = [&](const Eigen::VectorXd& r, double shift) {
        return coarse.apply(
            r, shift, [&](const Eigen::VectorXd& v) { return model.precondition(v, shift); });
    };
    const Eigen::VectorXd seed =
        history_direction.stableNorm() > 0.0 ? history_direction : model.precondition(-g, 0.0);
    mcscf::CiahWorkspace workspace(problem, seed, options);
    std::optional<OrbitalTrial> cached;
    Eigen::VectorXd cached_displacement;
    const auto observe = [&](const Eigen::VectorXd& p) -> const OrbitalTrial& {
        if (!cached || cached_displacement.size() != p.size() ||
            !(cached_displacement.array() == p.array()).all()) {
            OrbitalTrial trial;
            trial.orbitals = model.rotate(p);
            trial.density = model.density(trial.orbitals);
            trial.physical = evaluate(trial.density);
            cached = std::move(trial);
            cached_displacement = p;
        }
        return *cached;
    };
    mcscf::CiahResult candidate;
    while (true) {
        candidate = workspace.next(g);
        // Same full-residual forcing as the common CIAH microstep driver. A new
        // image alone is not a reason to advance a nonlinear SCF orbital step.
        if (candidate.resolved(g.stableNorm(), 1.0 / 3.0) || candidate.exhaustion)
            break;
        if (candidate.converged)
            workspace.reset_rhs();
    }
    const auto residual = [&](const SpinMatrices& d, const SpinMatrices& f) {
        double value = 0.0;
        for (int spin = 0; spin < 2; ++spin)
            if (d[spin].size() != 0)
                value += commutator_error(f[spin], d[spin], overlap).squaredNorm();
        return value;
    };
    const double source_residual = residual(reference_density, reference.fock);
    // A finite-precision energy tie is judged by the unchanged physical residual,
    // not by a loosened SCF tolerance or by the projected/model gradient.
    const double energy_noise = 32.0 * std::numeric_limits<double>::epsilon() *
                                std::max(1.0, std::abs(reference.electronic_energy));
    Eigen::VectorXd p = candidate.step;
    mcscf::diagnostics::ciah_trace(
        "scf_orbital_proposal",
        "basis=%d,gradient=%.17g,residual=%.17g,step=%.17g,predicted_gradient=%.17g",
        candidate.basis_size,
        g.stableNorm(),
        candidate.residual_norm,
        p.stableNorm(),
        (g + candidate.hessian_step).stableNorm());
    const auto acceptable = [&](const OrbitalTrial& trial) {
        const double de = trial.physical.electronic_energy - reference.electronic_energy;
        return de < 0.0 || (de <= energy_noise &&
                            residual(trial.density, trial.physical.fock) < source_residual);
    };
    // A soft tangent step can leave a curved low-energy valley at second order.
    // Correct transverse nonlinearity before shortening the useful soft motion.
    // AH's lowest eigenvalue lies below the sampled Hessian spectrum. Its
    // shifted coarse inverse with a positive gap base is positive, unlike the
    // unshifted signed coarse inverse. These corrections need no new HVP.
    const Eigen::VectorXd proposal = p;
    while (p.stableNorm() > 0.0) {
        const Eigen::VectorXd u = proposal / proposal.stableNorm();
        const auto inverse = [&](const Eigen::VectorXd& v) {
            return coarse.apply(v, candidate.eigenvalue, [&](const Eigen::VectorXd& r) {
                return model.positive_precondition(r, candidate.eigenvalue);
            });
        };
        const Eigen::VectorXd pu = inverse(u);
        const double denominator = u.dot(pu);
        for (int correction = 0; correction < candidate.basis_size; ++correction) {
            const auto& trial = observe(p);
            if (acceptable(trial))
                return std::move(cached);
            const double energy = trial.physical.electronic_energy;
            const auto gradient = model.pullback(trial.physical.fock, p, trial.orbitals);
            const Eigen::VectorXd normal = gradient - u * u.dot(gradient);
            Eigen::VectorXd direction = -inverse(gradient);
            if (denominator > 0.0)
                direction -= pu * (u.dot(direction) / denominator);
            direction -= u * u.dot(direction);
            if (!(gradient.dot(direction) < 0.0))
                direction = -normal;
            if (!(direction.stableNorm() > 0.0))
                break;
            bool advanced = false;
            while (direction.stableNorm() > 0.0) {
                const Eigen::VectorXd next = p + direction;
                if ((next.array() == p.array()).all())
                    break;
                const auto& test = observe(next);
                const double de = test.physical.electronic_energy - energy;
                const auto next_gradient = model.pullback(test.physical.fock, next, test.orbitals);
                const double next_normal = (next_gradient - u * u.dot(next_gradient)).stableNorm();
                if (de < 0.0 || (de <= energy_noise && next_normal < normal.stableNorm())) {
                    p = next;
                    advanced = true;
                    mcscf::diagnostics::ciah_trace(
                        "scf_normal_correction",
                        "index=%d,energy_change=%.17g,normal_before=%.17g,normal_after=%.17g",
                        correction + 1,
                        de,
                        normal.stableNorm(),
                        next_normal);
                    break;
                }
                direction *= 0.5;
            }
            if (!advanced)
                break;
        }
        const auto& trial = observe(p);
        const double de = trial.physical.electronic_energy - reference.electronic_energy;
        mcscf::diagnostics::ciah_trace(
            "scf_orbital_trial",
            "step=%.17g,energy_change=%.17g,residual_before=%.17g,residual_after=%.17g",
            p.stableNorm(),
            de,
            source_residual,
            residual(trial.density, trial.physical.fock));
        if (acceptable(trial))
            return std::move(cached);
        // Restart transverse relaxation at each shorter soft displacement. Halving
        // an already corrected point would scale its quadratic normal correction
        // linearly and put it off the curved valley again.
        const Eigen::VectorXd half = 0.5 * u.dot(p) * u;
        // Stop when the matrix exponential no longer changes a density.
        const auto next = model.density(model.rotate(half));
        bool changed = false;
        for (int spin = 0; spin < 2; ++spin)
            if (next[spin].size() != 0)
                changed = changed || !(next[spin].array() == reference_density[spin].array()).all();
        if (!changed || (half.array() == p.array()).all())
            break;
        p = half;
    }
    return std::nullopt;
}

} // namespace xmvb::scf
