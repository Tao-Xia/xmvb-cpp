#include "core/index.hpp"
#include "mcscf/optimization/preconditioners/coarse.hpp"
#include "linalg/range_inverse.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include "mcscf/optimization/trace.hpp"
#include "linalg/symmetry.hpp"

namespace xmvb::mcscf {

CoarsePreconditioner::CoarsePreconditioner(Eigen::Index dimension,
                                           int capacity,
                                           double lindep,
                                           double action_accuracy)
    : lindep_(lindep), action_accuracy_(action_accuracy) {
    if (dimension <= 0 || capacity < 0 || !(lindep > 0) || !std::isfinite(action_accuracy) ||
        action_accuracy < 0)
        throw std::invalid_argument("invalid coarse preconditioner dimensions");
    basis_.resize(dimension, std::min<Eigen::Index>(dimension, capacity));
    images_.resize(dimension, basis_.cols());
    metrics_.resize(dimension, basis_.cols());
    hessian_projection_ = Eigen::MatrixXd::Zero(basis_.cols(), basis_.cols());
    metric_projection_ = Eigen::MatrixXd::Zero(basis_.cols(), basis_.cols());
    sample_factor_ = Eigen::MatrixXd::Zero(basis_.cols(), basis_.cols());
}

void CoarsePreconditioner::observe(const Eigen::VectorXd& direction,
                                   const Eigen::VectorXd& image,
                                   const Eigen::VectorXd& metric_image) {
    if (direction.size() != basis_.rows() || image.size() != basis_.rows() ||
        metric_image.size() != basis_.rows() || !direction.allFinite() || !image.allFinite() ||
        !metric_image.allFinite())
        throw std::invalid_argument("invalid coarse operator sample");
    if (size_ == basis_.cols())
        return;
    Eigen::VectorXd v = direction, h = image, g = metric_image;
    const double initial = v.stableNorm();
    if (!(initial > 0))
        return;
    Eigen::VectorXd coefficients = Eigen::VectorXd::Zero(size_);
    if (diagnostics::ciah_event_trace_enabled() && size_) {
        const Eigen::VectorXd forward = basis_.leftCols(size_).transpose() * image;
        const Eigen::VectorXd backward = images_.leftCols(size_).transpose() * direction;
        diagnostics::ciah_trace("coarse_source_reciprocity",
                                "size=%d,direction_norm=%.17g,image_norm=%.17g,skew=%.17g",
                                size_,
                                initial,
                                image.norm(),
                                (forward - backward).norm());
    }
    for (int pass = 0; pass < 2; ++pass)
        for (int i = 0; i < size_; ++i) {
            const double coefficient = basis_.col(i).dot(v);
            coefficients[i] += coefficient;
            v.noalias() -= coefficient * basis_.col(i);
            h.noalias() -= coefficient * images_.col(i);
            g.noalias() -= coefficient * metrics_.col(i);
        }
    const double norm = v.stableNorm();
    if (diagnostics::ciah_event_trace_enabled())
        diagnostics::ciah_trace(
            "coarse_sample_condition",
            "size=%d,remaining_fraction=%.17g,source_image=%.17g,transformed_image=%.17g",
            size_,
            norm / initial,
            image.norm(),
            norm > 0.0 ? h.norm() / norm : 0.0);
    // Certify the entire original sample Gram, not just the new column
    // against an already orthonormalized basis. Images are transformed after
    // their evaluation: recursively ill-conditioned QR factors amplify their
    // original errors even if every individual orthogonalization looks safe.
    Eigen::MatrixXd factor = Eigen::MatrixXd::Zero(size_ + 1, size_ + 1);
    factor.topLeftCorner(size_, size_) = sample_factor_.topLeftCorner(size_, size_);
    factor.col(size_).head(size_) = coefficients / initial;
    factor(size_, size_) = norm / initial;
    Eigen::JacobiSVD<Eigen::MatrixXd> spectrum(factor);
    const double singular_ratio =
        spectrum.singularValues().tail(1)[0] / spectrum.singularValues()[0];
    const double ratio = singular_ratio * singular_ratio;
    diagnostics::ciah_trace("coarse_sample_gram", "size=%d,gram_ratio=%.17g", size_, ratio);
    if (!(ratio > lindep_)) {
        diagnostics::ciah_trace(
            "coarse_sample_rank_rejected",
            "size=%d,remaining_fraction=%.17g,gram_ratio=%.17g,rank_tolerance=%.17g",
            size_,
            norm / initial,
            ratio,
            lindep_);
        return;
    }
    basis_.col(size_) = v / norm;
    images_.col(size_) = h / norm;
    metrics_.col(size_) = g / norm;
    const int candidate = size_;
    if (candidate) {
        hessian_projection_.col(candidate).head(candidate) =
            basis_.leftCols(candidate).transpose() * images_.col(candidate);
        hessian_projection_.row(candidate).head(candidate) =
            (images_.leftCols(candidate).transpose() * basis_.col(candidate)).transpose();
        metric_projection_.col(candidate).head(candidate) =
            basis_.leftCols(candidate).transpose() * metrics_.col(candidate);
        metric_projection_.row(candidate).head(candidate) =
            (metrics_.leftCols(candidate).transpose() * basis_.col(candidate)).transpose();
    }
    hessian_projection_(candidate, candidate) = basis_.col(candidate).dot(images_.col(candidate));
    metric_projection_(candidate, candidate) = basis_.col(candidate).dot(metrics_.col(candidate));
    const auto certified = [&](const auto& projected,
                               const auto& action,
                               double action_accuracy,
                               const char* operator_name) {
        const Eigen::MatrixXd raw = projected.topLeftCorner(candidate + 1, candidate + 1);
        const double skew = (raw - raw.transpose()).norm();
        double bound = 256 * std::numeric_limits<double>::epsilon() * (candidate + 1) *
                       std::max(1.0, raw.norm());
        if (action_accuracy > 0)
            bound += action_accuracy * basis_.leftCols(candidate + 1).norm() *
                     action.leftCols(candidate + 1).norm();
        double projection_roundoff = 0;
        if (skew > bound) {
            projection_roundoff = linalg::projection_skew_roundoff(
                basis_.leftCols(candidate + 1), action.leftCols(candidate + 1), raw);
            bound += projection_roundoff;
        }
        if (!std::isfinite(skew) || !std::isfinite(bound) || skew > bound) {
            diagnostics::ciah_trace("coarse_sample_image_rejected",
                                    "size=%d,operator=%s,skew=%.17g,bound=%.17g,projection_"
                                    "roundoff=%.17g,image_norm=%.17g",
                                    candidate,
                                    operator_name,
                                    skew,
                                    bound,
                                    projection_roundoff,
                                    action.leftCols(candidate + 1).norm());
            return false;
        }
        return true;
    };
    if (!certified(hessian_projection_, images_, action_accuracy_, "hessian") ||
        !certified(metric_projection_, metrics_, 0.0, "metric"))
        return;
    for (int i = 0; i < candidate; ++i) {
        const double hessian =
            0.5 * (hessian_projection_(i, candidate) + hessian_projection_(candidate, i));
        const double metric =
            0.5 * (metric_projection_(i, candidate) + metric_projection_(candidate, i));
        hessian_projection_(i, candidate) = hessian_projection_(candidate, i) = hessian;
        metric_projection_(i, candidate) = metric_projection_(candidate, i) = metric;
    }
    sample_factor_.topLeftCorner(size_ + 1, size_ + 1) = factor;
    ++size_;
}

Eigen::VectorXd CoarsePreconditioner::apply(
    const Eigen::VectorXd& residual,
    double eigenvalue,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& inverse) const {
    if (residual.size() != basis_.rows() || !residual.allFinite() || !std::isfinite(eigenvalue))
        throw std::invalid_argument("invalid coarse inverse residual");
    if (!size_)
        return inverse(residual);
    const auto v = basis_.leftCols(size_);
    const Eigen::MatrixXd lv = images_.leftCols(size_) - eigenvalue * metrics_.leftCols(size_);
    const linalg::RangeInverse factor(hessian_projection_.topLeftCorner(size_, size_) -
                                      eigenvalue * metric_projection_.topLeftCorner(size_, size_));
    if (factor.rank() != size_)
        diagnostics::ciah_trace("coarse_shift_range",
                                "size=%d,rank=%ld,shift=%.17g",
                                size_,
                                static_cast<long>(factor.rank()),
                                eigenvalue);
    const Eigen::VectorXd a = factor.solve(v.transpose() * residual);
    const Eigen::VectorXd fine = inverse(residual - lv * a);
    const Eigen::VectorXd b = factor.solve(lv.transpose() * fine);
    return fine + v * (a - b);
}

SchurSpectrum CoarsePreconditioner::schur_spectrum(Eigen::Index no) const {
    if (no <= 0 || no >= basis_.rows())
        throw std::invalid_argument("invalid sampled Schur orbital dimension");
    std::vector<int> oi, si;
    for (int i = 0; i < size_; ++i) {
        const double o = basis_.col(i).head(no).squaredNorm();
        const double s = basis_.col(i).tail(basis_.rows() - no).squaredNorm();
        if (o > 0.0 && s == 0.0)
            oi.push_back(i);
        if (s > 0.0 && o == 0.0)
            si.push_back(i);
    }
    SchurSpectrum result;
    result.orbital_dimension = oi.size();
    result.structure_dimension = si.size();
    if (oi.empty() || si.empty())
        return result;
    Eigen::MatrixXd a(oi.size(), oi.size()), c(si.size(), si.size());
    Eigen::MatrixXd b(si.size(), oi.size()), mo(a.rows(), a.cols()), ms(c.rows(), c.cols());
    for (int i = 0; i < a.rows(); ++i)
        for (int j = 0; j < a.cols(); ++j) {
            a(i, j) = basis_.col(oi[i]).dot(images_.col(oi[j]));
            mo(i, j) = basis_.col(oi[i]).dot(metrics_.col(oi[j]));
        }
    for (int i = 0; i < c.rows(); ++i) {
        for (int j = 0; j < c.cols(); ++j) {
            c(i, j) = basis_.col(si[i]).dot(images_.col(si[j]));
            ms(i, j) = basis_.col(si[i]).dot(metrics_.col(si[j]));
        }
        for (int j = 0; j < a.cols(); ++j)
            b(i, j) = 0.5 * (basis_.col(si[i]).dot(images_.col(oi[j])) +
                             basis_.col(oi[j]).dot(images_.col(si[i])));
    }
    a = (0.5 * (a + a.transpose())).eval();
    c = (0.5 * (c + c.transpose())).eval();
    mo = (0.5 * (mo + mo.transpose())).eval();
    ms = (0.5 * (ms + ms.transpose())).eval();
    const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> as(
        a, mo, Eigen::EigenvaluesOnly);
    const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> cs(
        c, ms, Eigen::EigenvaluesOnly);
    if (as.info() != Eigen::Success || cs.info() != Eigen::Success)
        throw std::runtime_error("sampled Schur diagonal spectrum failed");
    result.orbital_min = as.eigenvalues()[0];
    result.orbital_max = as.eigenvalues().tail(1)[0];
    result.structure_min = cs.eigenvalues()[0];
    result.structure_max = cs.eigenvalues().tail(1)[0];
    result.structure_positive = result.structure_min > lindep_ * result.structure_max;
    result.orbital_positive = result.orbital_min > lindep_ * result.orbital_max;
    if (!result.structure_positive)
        return result;
    const Eigen::LLT<Eigen::MatrixXd> factor(c);
    if (factor.info() != Eigen::Success)
        throw std::runtime_error("sampled positive structure factorization failed");
    const Eigen::MatrixXd inverse_b = factor.solve(b);
    const double bound = 256 * std::numeric_limits<double>::epsilon() * c.rows() *
                         std::max(1.0, c.norm() * inverse_b.norm() + b.norm());
    if (!inverse_b.allFinite() || (c * inverse_b - b).norm() > bound)
        throw std::runtime_error("sampled structure elimination failed its residual certificate");
    Eigen::MatrixXd removed = b.transpose() * inverse_b;
    removed = (0.5 * (removed + removed.transpose())).eval();
    const Eigen::MatrixXd relaxed = a - removed;
    const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> rs(
        relaxed, mo, Eigen::EigenvaluesOnly);
    if (rs.info() != Eigen::Success)
        throw std::runtime_error("sampled relaxed orbital spectrum failed");
    result.relaxed_min = rs.eigenvalues()[0];
    result.relaxed_max = rs.eigenvalues().tail(1)[0];
    if (result.orbital_positive) {
        const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> loss(
            removed, a, Eigen::EigenvaluesOnly);
        if (loss.info() != Eigen::Success)
            throw std::runtime_error("sampled coupling fraction spectrum failed");
        result.maximum_fraction_removed = loss.eigenvalues().tail(1)[0];
    }
    return result;
}

CoarseSpectrum CoarsePreconditioner::orbital_spectrum(
    Eigen::Index no, const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& inverse) const {
    if (no <= 0 || no > basis_.rows())
        throw std::invalid_argument("invalid sampled orbital dimension");
    std::vector<int> indices;
    for (int i = 0; i < size_; ++i)
        if (basis_.col(i).head(no).squaredNorm() > 0.0 &&
            basis_.col(i).tail(basis_.rows() - no).squaredNorm() == 0.0)
            indices.push_back(i);
    CoarseSpectrum result;
    result.dimension = static_cast<int>(indices.size());
    if (!result.dimension)
        return result;
    Eigen::MatrixXd v(no, result.dimension), av(no, result.dimension), pav(no, result.dimension);
    for (int j = 0; j < result.dimension; ++j) {
        v.col(j) = basis_.col(indices[j]).head(no);
        av.col(j) = images_.col(indices[j]).head(no);
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(basis_.rows());
        rhs.head(no) = av.col(j);
        const Eigen::VectorXd image = inverse(rhs);
        if (image.size() != basis_.rows() || !image.allFinite())
            throw std::runtime_error("invalid sampled base inverse image");
        pav.col(j) = image.head(no);
    }
    Eigen::MatrixXd e = v.transpose() * av;
    e = (0.5 * (e + e.transpose())).eval();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> curvature(e, Eigen::EigenvaluesOnly);
    if (curvature.info() != Eigen::Success)
        throw std::runtime_error("sampled orbital spectrum failed");
    result.curvature_min = curvature.eigenvalues()[0];
    result.curvature_max = curvature.eigenvalues().tail(1)[0];
    result.positive_curvature = result.curvature_min > lindep_ * result.curvature_max;
    if (!result.positive_curvature)
        return result;
    Eigen::MatrixXd t = av.transpose() * pav;
    t = (0.5 * (t + t.transpose())).eval();
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(
        t, e, Eigen::EigenvaluesOnly);
    if (spectrum.info() != Eigen::Success)
        throw std::runtime_error("sampled preconditioned spectrum failed");
    result.preconditioned_min = spectrum.eigenvalues()[0];
    result.preconditioned_max = spectrum.eigenvalues().tail(1)[0];
    return result;
}

} // namespace xmvb::mcscf
