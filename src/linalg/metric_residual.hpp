#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <Eigen/Core>
#include <Eigen/SVD>

namespace xmvb::linalg {

struct MetricSpectrum {
    double lower, upper;
};

/** Spectrum bounds for (wedge^na R) tensor (wedge^nb R), squared.
 * Use the retained orbital factor R, not R^T R: forming its Gram would
 * square the conditioning before estimating the smallest singular value.
 * A missing bound preserves the caller's existing stopping policy.
 */
inline std::optional<MetricSpectrum>
exterior_metric_bounds(const Eigen::Ref<const Eigen::MatrixXd>& r, int na, int nb) {
    if (r.rows() != r.cols() || !r.allFinite() || na < 0 || nb < 0 || na > r.rows() ||
        nb > r.rows())
        throw std::invalid_argument("invalid exterior metric dimensions");
    if (na + nb == 0)
        return MetricSpectrum{1, 1};
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(r);
    if (svd.info() != Eigen::Success)
        return {};
    const double error = 64 * r.rows() * std::numeric_limits<double>::epsilon() * r.norm();
    long double lo = 0, hi = 0;
    for (int electrons : {na, nb})
        for (int i = 0; i < electrons; ++i) {
            const double small = svd.singularValues()[r.rows() - 1 - i] - error;
            if (!(small > 0))
                return {};
            lo += 2 * std::log(static_cast<long double>(small));
            hi += 2 * std::log(static_cast<long double>(svd.singularValues()[i] + error));
        }
    MetricSpectrum bounds{static_cast<double>(std::exp(lo)), static_cast<double>(std::exp(hi))};
    // Outward rounding of the final conversion; never fabricate a positive
    // lower bound when a severely ill-conditioned exterior power underflows.
    bounds.lower = std::nextafter(bounds.lower, 0.0);
    bounds.upper = std::nextafter(bounds.upper, std::numeric_limits<double>::infinity());
    if (!(bounds.lower > 0) || !std::isfinite(bounds.upper))
        return {};
    return bounds;
}

struct MetricResidualResult {
    double lower = 0, upper = 0;
    Eigen::Index actions = 0;
};

/** Bound ||r||_(M^-1) using l G <= M <= u G and an existing G inverse.
 * Refine only an undecided threshold crossing. No dense factor, H action,
 * Ritz-space projection or high-accuracy inverse is needed. O(n) scratch.
 */
template <class Action, class Inverse>
MetricResidualResult metric_residual_bounds(const Eigen::VectorXd& rhs,
                                            double target,
                                            double lower,
                                            double upper,
                                            const Action& action,
                                            const Inverse& inverse) {
    if (!(lower > 0 && upper >= lower && target > 0) || !std::isfinite(upper) ||
        !std::isfinite(target) || !rhs.allFinite())
        throw std::invalid_argument("invalid metric residual bound");
    MetricResidualResult result;
    if (rhs.isZero(0.0))
        return result;
    const double roundoff =
        64 * std::numeric_limits<double>::epsilon() * std::max<Eigen::Index>(1, rhs.size());
    Eigen::VectorXd z = inverse(rhs);
    double rz = rhs.dot(z);
    if (!(rz > 0) || !std::isfinite(rz))
        throw std::runtime_error("invalid residual reference metric");
    result.lower = std::sqrt(std::max(0.0, rz * (1 - roundoff) / upper));
    result.upper = std::sqrt(rz * (1 + roundoff) / lower);
    const auto decided = [&] { return result.upper <= target || result.lower > target; };
    if (decided())
        return result;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(rhs.size()), mx = x, p = z, residual = rhs;
    // In exact arithmetic CG resolves the entire space in n steps. In finite
    // precision an unresolved interval is returned conservatively, never accepted.
    for (Eigen::Index k = 0; k < rhs.size(); ++k) {
        const Eigen::VectorXd mp = action(p);
        ++result.actions;
        const double curvature = p.dot(mp);
        if (!(curvature > 0) || !std::isfinite(curvature))
            break;
        const double alpha = rz / curvature;
        x.noalias() += alpha * p;
        mx.noalias() += alpha * mp;
        residual = rhs - mx;
        z = inverse(residual);
        const double next = residual.dot(z);
        if (!(next >= 0) || !std::isfinite(next))
            break;
        const double rx = rhs.dot(x), xmx = x.dot(mx);
        const double value = 2 * rx - xmx;
        const double guard = roundoff * (2 * std::abs(rx) + std::abs(xmx));
        result.lower = std::sqrt(std::max(0.0, value - guard));
        result.upper = std::sqrt(std::max(0.0, value + guard) + next * (1 + roundoff) / lower);
        if (result.upper <= target) {
            // A positive stopping decision uses a fresh physical image, not
            // the accumulated CG image. This is the sole extra certification.
            mx = action(x);
            ++result.actions;
            residual = rhs - mx;
            z = inverse(residual);
            const double defect = residual.dot(z);
            const double actual = 2 * rhs.dot(x) - x.dot(mx);
            const double error = roundoff * (2 * std::abs(rhs.dot(x)) + std::abs(x.dot(mx)));
            if (!(defect >= 0) || !std::isfinite(defect))
                throw std::runtime_error("invalid metric residual certification");
            result.lower = std::sqrt(std::max(0.0, actual - error));
            result.upper =
                std::sqrt(std::max(0.0, actual + error) + defect * (1 + roundoff) / lower);
            // An unresolved roundoff boundary is rejected conservatively.
            break;
        }
        if (decided() || next == 0)
            break;
        p = z + (next / rz) * p;
        rz = next;
    }
    return result;
}

} // namespace xmvb::linalg
