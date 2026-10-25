#include "vbscf/orbitals/charts/whitening.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <lapacke.h>

#include "core/timing.hpp"

namespace xmvb::vb {
namespace {

/** @brief Extreme eigenvalues of diag(d)-z*z^T from scalar secular roots.
 * Retains the spectral rank criterion near its boundary without forming
 * another eigensystem. Bisection ends at floating-point adjacency.
 */
std::pair<double, double> downdated_extremes(const Eigen::VectorXd& diagonal,
                                             const Eigen::VectorXd& update) {
    const Eigen::Index n = diagonal.size();
    if (n == 1) {
        const double value = static_cast<double>(static_cast<long double>(diagonal[0]) -
                                                 static_cast<long double>(update[0]) * update[0]);
        return {value, value};
    }
    const auto secular = [&](long double x) {
        long double result = 1.0L;
        for (Eigen::Index i = 0; i < n; ++i) {
            if (update[i] == 0.0)
                continue;
            result -= static_cast<long double>(update[i]) * update[i] /
                      (static_cast<long double>(diagonal[i]) - x);
        }
        return result;
    };
    const auto root = [&](long double lower, long double upper) {
        while (true) {
            const long double middle = lower + 0.5L * (upper - lower);
            if (!(middle > lower && middle < upper))
                return static_cast<double>(middle);
            if (secular(middle) > 0.0L)
                lower = middle;
            else
                upper = middle;
        }
    };
    const long double infinity = std::numeric_limits<long double>::infinity();
    long double weight = 0.0L;
    for (const double value : update)
        weight += static_cast<long double>(value) * value;
    double minimum = diagonal[0], maximum = diagonal[n - 1];
    const long double upper = std::nextafter(static_cast<long double>(minimum), -infinity);
    if (secular(upper) < 0.0L)
        minimum =
            root(std::nextafter(static_cast<long double>(minimum) - weight, -infinity), upper);
    if (diagonal[n - 2] != maximum && update[n - 1] != 0.0) {
        const long double lower =
            std::nextafter(static_cast<long double>(diagonal[n - 2]), infinity);
        const long double last = std::nextafter(static_cast<long double>(maximum), -infinity);
        if (secular(lower) <= 0.0L)
            maximum = diagonal[n - 2];
        else if (secular(last) < 0.0L)
            maximum = root(lower, last);
    }
    return {minimum, maximum};
}

} // namespace

TangentWhitening::TangentWhitening(const Eigen::Ref<const Eigen::MatrixXd>& quotient,
                                   const Eigen::Ref<const Eigen::MatrixXd>& overlap) {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::ChartWhitening);
    if (overlap.rows() != quotient.rows() || overlap.cols() != quotient.rows())
        throw std::invalid_argument("quotient overlap dimensions are inconsistent");
    metric_.noalias() = quotient.transpose() * overlap * quotient;
    metric_ = (0.5 * (metric_ + metric_.transpose())).eval();
    if (!quotient.cols()) {
        basis_ = quotient;
        return;
    }
    eigenvectors_ = metric_;
    eigenvalues_.resize(metric_.rows());
    const lapack_int n = static_cast<lapack_int>(metric_.rows());
    const lapack_int info =
        LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'L', n, eigenvectors_.data(), n, eigenvalues_.data());
    if (info != 0 || !eigenvalues_.allFinite() || !eigenvectors_.allFinite() ||
        !(eigenvalues_[0] > 0.0))
        throw std::runtime_error("failed to factor accepted-point quotient metric");
    const Eigen::MatrixXd inverse_root = eigenvectors_ *
                                         eigenvalues_.cwiseSqrt().cwiseInverse().asDiagonal() *
                                         eigenvectors_.transpose();
    basis_.noalias() = quotient * inverse_root;
}

WhitenedTangent TangentWhitening::apply(const Eigen::Ref<const Eigen::VectorXd>& coefficients,
                                        const Eigen::Ref<const Eigen::MatrixXd>& overlap,
                                        bool sphere) const {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::ChartGram);
    if (coefficients.size() != basis_.rows() || overlap.rows() != basis_.rows() ||
        overlap.cols() != basis_.rows())
        throw std::invalid_argument("normalized-orbital tangent dimensions are inconsistent");
    if (!basis_.cols())
        return {basis_, metric_};
    const Eigen::VectorXd overlap_raw = overlap * coefficients;
    const double squared_norm = coefficients.dot(overlap_raw);
    if (!(squared_norm > 0.0) || !std::isfinite(squared_norm))
        throw std::runtime_error("normalized-orbital quotient metric has nonpositive orbital norm");
    const double norm = std::sqrt(squared_norm);
    const Eigen::VectorXd c = coefficients / norm;
    const Eigen::VectorXd u = basis_.transpose() * (overlap_raw / norm);
    long double squared = 0.0L;
    for (const double value : u)
        squared += static_cast<long double>(value) * value;
    const double delta = static_cast<double>(1.0L - squared);
    if (!(delta > 0.0) || !std::isfinite(delta))
        throw std::runtime_error("normalized-orbital quotient metric is rank deficient");
    const double rank_factor = 64.0 * std::numeric_limits<double>::epsilon() * basis_.cols();
    // Congruence bounds certify normal cases. Near the boundary evaluate the
    // same min/max test by the diagonal rank-one secular equation.
    if (!(eigenvalues_[0] * delta > rank_factor * eigenvalues_.tail(1)[0])) {
        const Eigen::VectorXd z =
            eigenvalues_.cwiseSqrt().asDiagonal() * (eigenvectors_.transpose() * u);
        const auto [minimum, maximum] = downdated_extremes(eigenvalues_, z);
        if (!(maximum > 0.0) || !(minimum > rank_factor * maximum))
            throw std::runtime_error("normalized-orbital quotient metric is rank deficient");
    }
    const double root = std::sqrt(delta);
    const double beta = 1.0 / (root * (1.0 + root));
    const Eigen::VectorXd image = basis_ * u;
    WhitenedTangent result;
    result.basis = basis_;
    result.basis.noalias() += beta * image * u.transpose();
    if (sphere)
        result.basis.noalias() -= (c / root) * u.transpose();
    result.basis *= norm;
    // A sphere representative has L^T S L=||b||^2 I. Without removing
    // the radial representative its inverse is (I-u*u^T)/||b||^2.
    result.inverse_gram = Eigen::MatrixXd::Identity(u.size(), u.size());
    if (!sphere)
        result.inverse_gram.noalias() -= u * u.transpose();
    result.inverse_gram /= squared_norm;
    result.inverse_gram = (0.5 * (result.inverse_gram + result.inverse_gram.transpose())).eval();
    if (!result.basis.allFinite() || !result.inverse_gram.allFinite())
        throw std::runtime_error("non-finite normalized-orbital tangent factor");
    return result;
}

} // namespace xmvb::vb
