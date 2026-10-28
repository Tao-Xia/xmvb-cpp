#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <Eigen/Core>

#include "mcscf/optimization/trust_region/spectral.hpp"

namespace xmvb::mcscf {

/** @brief One sampled shifted Newton step satisfying the physical component cap. */
struct BoundedCiahStep {
    Eigen::VectorXd coefficients;
    double shift = 0;
    bool hard_case = false;
};

/** @brief Restrict an AH model before selecting its finite direction.
 * V contains generalized eigenvectors normalized in the projected metric.
 * At each trial radius the existing spectral trust solver minimizes the same
 * quadratic, including its hard case. Only small coefficient arrays are owned.
 * The callback measures the actual physical lift; no triangle-inequality bound
 * or diagonal surrogate replaces it. The lower shift may be the original AH
 * shift, so restriction need never weaken that model's regularization.
 */
inline BoundedCiahStep bound_ciah_step(const Eigen::VectorXd& eigenvalues,
                                       const Eigen::MatrixXd& eigenvectors,
                                       const Eigen::VectorXd& gradient,
                                       const std::function<double(const Eigen::VectorXd&)>& maximum,
                                       double cap,
                                       double minimum_shift = 0) {
    const Eigen::Index n = gradient.size();
    if (!n || eigenvalues.size() != n || eigenvectors.rows() != n || eigenvectors.cols() != n ||
        !(cap > 0) || !std::isfinite(cap) || minimum_shift < 0 || !std::isfinite(minimum_shift))
        throw std::invalid_argument("invalid bounded CIAH model");
    const Eigen::VectorXd rhs = eigenvectors.transpose() * gradient;
    const double lower_shift = std::max({0.0, minimum_shift, -eigenvalues.minCoeff()});
    double upper_radius = 1;
    if (eigenvalues.minCoeff() + lower_shift > 0) {
        const Eigen::VectorXd spectral =
            -rhs.cwiseQuotient((eigenvalues.array() + lower_shift).matrix());
        Eigen::VectorXd step = eigenvectors * spectral;
        if (maximum(step) <= cap)
            return {std::move(step), lower_shift, false};
        upper_radius = spectral.stableNorm();
    }
    const auto at_radius = [&](double radius) {
        auto solution = solve_spectral_trust_region(eigenvalues, rhs, radius);
        return BoundedCiahStep{eigenvectors * solution.step, solution.shift, solution.hard_case};
    };
    double lower_radius = 0;
    BoundedCiahStep feasible{Eigen::VectorXd::Zero(n), lower_shift, false};
    auto upper = at_radius(upper_radius);
    while (maximum(upper.coefficients) <= cap) {
        // A convex model may reach its finite unconstrained optimum first.
        if (upper.shift == 0)
            return upper;
        lower_radius = upper_radius;
        feasible = std::move(upper);
        upper_radius *= 2;
        upper = at_radius(upper_radius);
    }
    // Continuity gives a feasible/infeasible bracket; monotonicity of an
    // infinity norm along the spectral solution path is not assumed.
    for (;;) {
        const double mid = std::midpoint(lower_radius, upper_radius);
        if (mid == lower_radius || mid == upper_radius)
            return feasible;
        auto trial = at_radius(mid);
        if (maximum(trial.coefficients) <= cap) {
            lower_radius = mid;
            feasible = std::move(trial);
        } else {
            upper_radius = mid;
        }
    }
}
} // namespace xmvb::mcscf
