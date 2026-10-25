#include "vbscf/orbitals/charts/raw.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <Eigen/Cholesky>
#include <lapacke.h>

#include "core/timing.hpp"

namespace xmvb::vb {

RawTangentMetric::RawTangentMetric(const Eigen::Ref<const Eigen::MatrixXd>& quotient,
                                   const Eigen::Ref<const Eigen::MatrixXd>& overlap) {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::ChartGram);
    metric_.noalias() = quotient.transpose() * overlap * quotient;
    metric_ = (0.5 * (metric_ + metric_.transpose())).eval();
    if (!quotient.cols()) {
        inverse_ = metric_;
        return;
    }
    const Eigen::LLT<Eigen::MatrixXd> factor(metric_);
    if (factor.info() != Eigen::Success)
        throw std::runtime_error("raw quotient AO metric is not positive definite");
    inverse_ = factor.solve(Eigen::MatrixXd::Identity(metric_.rows(), metric_.cols()));
    lower_bound_ = 1.0 / inverse_.cwiseAbs().rowwise().sum().maxCoeff();
    upper_bound_ = metric_.cwiseAbs().rowwise().sum().maxCoeff();
}

RawTangent RawTangentMetric::apply(const Eigen::Ref<const Eigen::MatrixXd>& quotient,
                                   const Eigen::Ref<const Eigen::VectorXd>& coefficients,
                                   const Eigen::Ref<const Eigen::MatrixXd>& overlap,
                                   bool sphere) const {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::ChartGram);
    if (!quotient.cols())
        return {quotient, metric_};
    const Eigen::VectorXd sb = overlap * coefficients;
    const double squared_norm = coefficients.dot(sb);
    if (!(squared_norm > 0.0) || !std::isfinite(squared_norm))
        throw std::runtime_error("raw quotient has nonpositive orbital norm");
    const double norm = std::sqrt(squared_norm);
    const Eigen::VectorXd c = coefficients / norm;
    const Eigen::VectorXd r = quotient.transpose() * (sb / norm);
    const Eigen::VectorXd image = inverse_ * r;
    long double squared = 0.0L;
    for (Eigen::Index i = 0; i < r.size(); ++i)
        squared += static_cast<long double>(r[i]) * image[i];
    const double delta = static_cast<double>(1.0L - squared);
    if (!(delta > 0.0) || !std::isfinite(delta))
        throw std::runtime_error("raw normalized-orbital metric is rank deficient");
    RawTangent result;
    // The factor norm preserves scale invariance without metric whitening.
    result.basis = quotient;
    if (sphere)
        result.basis.noalias() -= c * r.transpose();
    result.basis *= norm;
    result.projection = r;
    result.inverse_projection = image;
    result.radial_gap = delta;

    // Same spectral rank cutoff as explicit whitening. Norm bounds suffice
    // normally; an inconclusive bound needs eigenvalues only, not a new basis.
    const double rank_factor = 64.0 * std::numeric_limits<double>::epsilon() * r.size();
    if (!(delta * lower_bound_ > rank_factor * upper_bound_)) {
        Eigen::MatrixXd work = metric_ - r * r.transpose();
        Eigen::VectorXd spectrum(r.size());
        const lapack_int n = static_cast<lapack_int>(r.size());
        const lapack_int info =
            LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'N', 'L', n, work.data(), n, spectrum.data());
        if (info != 0 || !(spectrum[0] > rank_factor * spectrum[n - 1]))
            throw std::runtime_error("raw normalized-orbital metric is rank deficient");
    }
    // Inverse of L^T S L, not the Euclidean raw Gram. This supplies quotient
    // vector coordinates and their dual covector lift without another factor.
    result.inverse_gram = inverse_;
    if (sphere) {
        result.inverse_gram.noalias() += image * image.transpose() / delta;
    }
    result.inverse_gram /= squared_norm;
    if (!result.basis.allFinite() || !result.inverse_projection.allFinite() ||
        !result.inverse_gram.allFinite())
        throw std::runtime_error("non-finite raw tangent metric");
    return result;
}

Eigen::VectorXd RawTangentMetric::reference_covector(const Eigen::VectorXd& covector,
                                                     const Eigen::VectorXd& projection) const {
    if (!reference_root_.size()) {
        xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::ChartWhitening);
        Eigen::MatrixXd vectors = metric_;
        Eigen::VectorXd values(metric_.rows());
        const lapack_int n = static_cast<lapack_int>(metric_.rows());
        const lapack_int info =
            LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'L', n, vectors.data(), n, values.data());
        if (info != 0 || !(values[0] > 0.0) || !values.allFinite())
            throw std::runtime_error("failed to factor reference coordinate metric");
        reference_root_ =
            vectors * values.cwiseSqrt().cwiseInverse().asDiagonal() * vectors.transpose();
    }
    const Eigen::VectorXd u = reference_root_ * projection;
    long double squared = 0.0L;
    for (const double value : u)
        squared += static_cast<long double>(value) * value;
    const double delta = static_cast<double>(1.0L - squared);
    if (!(delta > 0.0))
        throw std::runtime_error("singular reference coordinate metric");
    const double root = std::sqrt(delta);
    Eigen::VectorXd result = reference_root_ * covector;
    result.noalias() += u * (u.dot(result) / (root * (1.0 + root)));
    return result;
}

} // namespace xmvb::vb
