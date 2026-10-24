#pragma once

#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include "core/timing.hpp"

namespace xmvb::vb {

struct ProjectedOrbitalSurrogate {
    Eigen::MatrixXd one_electron;
    Eigen::MatrixXd overlap;
};

/**
 * @brief Projects a one-electron model out of the fixed inactive span.
 *
 * The sparse support is unchanged. The result is a local preconditioning
 * model, not the relaxed VBSCF Hessian.
 */
class InactiveOrbitalProjector {
  public:
    /** @brief Factors the common inactive span once per accepted orbital point. */
    InactiveOrbitalProjector(const Eigen::Ref<const Eigen::MatrixXd>& f,
                             const Eigen::Ref<const Eigen::MatrixXd>& s,
                             const Eigen::Ref<const Eigen::MatrixXd>& fixed_inactive)
        : f_(f), s_(s), inactive_(fixed_inactive) {
        xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::Projection);
        if (f.rows() != f.cols() || s.rows() != f.rows() || s.cols() != f.cols() ||
            fixed_inactive.rows() != f.rows() || !f.allFinite() || !s.allFinite() ||
            !fixed_inactive.allFinite()) {
            throw std::invalid_argument("invalid projected orbital surrogate input");
        }
        if (fixed_inactive.cols() == 0) {
            weights_.resize(0, f.rows());
            inverse_.resize(0, 0);
            return;
        }
        const Eigen::MatrixXd sc = s * fixed_inactive;
        Eigen::MatrixXd gram = fixed_inactive.transpose() * sc;
        gram = (0.5 * (gram + gram.transpose())).eval();
        Eigen::LDLT<Eigen::MatrixXd> factor(gram);
        if (factor.info() != Eigen::Success || !factor.vectorD().allFinite() ||
            !(factor.vectorD().minCoeff() > 0.0)) {
            throw std::runtime_error("rank-deficient fixed inactive surrogate span");
        }
        weights_ = factor.solve(sc.transpose());
        inverse_ = factor.solve(Eigen::MatrixXd::Identity(gram.rows(), gram.cols()));
    }

    ~InactiveOrbitalProjector() {
        Eigen::Index entries = weights_.size() + inverse_.size();
        if (full_model_)
            entries += full_model_->one_electron.size() + full_model_->overlap.size();
        xmvb::profiling::record_projector_bytes(entries * sizeof(double));
    }

    /**
   * @brief Projects a local support, optionally excluding one inactive orbital.
   *
   * Removing column j adds d_j d_j^T S / (G^-1)_jj to the common
   * projector, where G=C^T S C and d_j=C G^-1 e_j. Full-AO models
   * reuse the common contraction and apply only rank-one corrections.
   */
    ProjectedOrbitalSurrogate model(const std::vector<int>& support, int excluded = -1) {
        xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::Projection);
        if (excluded < -1 || excluded >= inactive_.cols())
            throw std::invalid_argument("invalid excluded inactive orbital");
        bool full_support = support.size() == static_cast<std::size_t>(f_.rows());
        for (std::size_t column = 0; column < support.size(); ++column) {
            if (support[column] < 0 || support[column] >= f_.rows())
                throw std::invalid_argument("invalid projected orbital support");
            full_support = full_support && support[column] == static_cast<int>(column);
        }
        if (full_support) {
            if (!full_model_) {
                Eigen::MatrixXd projected = Eigen::MatrixXd::Identity(f_.rows(), f_.rows());
                projected.noalias() -= inactive_ * weights_;
                full_model_ = contract(projected);
            }
            auto result = *full_model_;
            if (excluded >= 0) {
                const Eigen::VectorXd d = inactive_ * inverse_.col(excluded);
                const Eigen::VectorXd u =
                    weights_.row(excluded).transpose() / inverse_(excluded, excluded);
                add_excluded(f_, d, u, &result.one_electron);
                add_excluded(s_, d, u, &result.overlap);
            }
            return result;
        }
        Eigen::MatrixXd projected = Eigen::MatrixXd::Zero(f_.rows(), support.size());
        Eigen::MatrixXd weights(inactive_.cols(), support.size());
        for (std::size_t column = 0; column < support.size(); ++column) {
            projected(support[column], column) = 1.0;
            weights.col(column) = weights_.col(support[column]);
        }
        projected.noalias() -= inactive_ * weights;
        if (excluded >= 0) {
            const Eigen::VectorXd d = inactive_ * inverse_.col(excluded);
            projected.noalias() += d * (weights.row(excluded) / inverse_(excluded, excluded));
        }
        return contract(projected);
    }

  private:
    ProjectedOrbitalSurrogate contract(const Eigen::MatrixXd& projected) const {
        ProjectedOrbitalSurrogate result;
        result.one_electron = projected.transpose() * f_ * projected;
        result.overlap = projected.transpose() * s_ * projected;
        result.one_electron =
            (0.5 * (result.one_electron + result.one_electron.transpose())).eval();
        result.overlap = (0.5 * (result.overlap + result.overlap.transpose())).eval();
        return result;
    }

    void add_excluded(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
                      const Eigen::VectorXd& d,
                      const Eigen::VectorXd& u,
                      Eigen::MatrixXd* result) const {
        const Eigen::VectorXd md = matrix * d;
        const Eigen::VectorXd mtd = matrix.transpose() * d;
        const Eigen::VectorXd left = md - weights_.transpose() * (inactive_.transpose() * md);
        const Eigen::VectorXd right = mtd - weights_.transpose() * (inactive_.transpose() * mtd);
        result->noalias() +=
            left * u.transpose() + u * right.transpose() + d.dot(md) * (u * u.transpose());
        *result = (0.5 * (*result + result->transpose())).eval();
    }

    const Eigen::Ref<const Eigen::MatrixXd> f_, s_, inactive_;
    Eigen::MatrixXd weights_, inverse_;
    std::optional<ProjectedOrbitalSurrogate> full_model_;
};

inline ProjectedOrbitalSurrogate
projected_orbital_surrogate(const Eigen::Ref<const Eigen::MatrixXd>& f,
                            const Eigen::Ref<const Eigen::MatrixXd>& s,
                            const Eigen::Ref<const Eigen::MatrixXd>& fixed_inactive,
                            const std::vector<int>& support) {
    return InactiveOrbitalProjector(f, s, fixed_inactive).model(support);
}

/**
 * @brief Exact pullback Hessian of a normalized one-electron Rayleigh quotient.
 *
 * This evaluates the Hessian of
 * \f$e(d)=(x+Ud)^T F(x+Ud)/(x+Ud)^T S(x+Ud)\f$ at \f$d=0\f$.
 * Frozen stored coefficients remain in `x` and have zero rows in `tangent`.
 */
inline Eigen::MatrixXd normalized_orbital_curvature(const Eigen::MatrixXd& f,
                                                    const Eigen::MatrixXd& overlap,
                                                    const Eigen::VectorXd& x,
                                                    const Eigen::MatrixXd& tangent) {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::Curvature);
    const Eigen::Index n = x.size();
    if (f.rows() != n || f.cols() != n || overlap.rows() != n || overlap.cols() != n ||
        tangent.rows() != n || !f.allFinite() || !overlap.allFinite() || !x.allFinite() ||
        !tangent.allFinite()) {
        throw std::invalid_argument("invalid normalized-orbital curvature input");
    }
    const Eigen::VectorXd sx = overlap * x;
    const Eigen::VectorXd fx = f * x;
    const double norm_squared = x.dot(sx);
    if (!(norm_squared > 0.0) || !std::isfinite(norm_squared)) {
        throw std::invalid_argument("nonpositive normalized-orbital metric norm");
    }
    const double energy = x.dot(fx) / norm_squared;
    const Eigen::VectorXd a = tangent.transpose() * ((fx - energy * sx) / norm_squared);
    const Eigen::VectorXd b = tangent.transpose() * (sx / norm_squared);
    Eigen::MatrixXd curvature =
        (2.0 / norm_squared) * (tangent.transpose() * (f * tangent - energy * overlap * tangent));
    curvature.noalias() -= 4.0 * (a * b.transpose() + b * a.transpose());
    curvature = (0.5 * (curvature + curvature.transpose())).eval();
    if (!curvature.allFinite()) {
        throw std::runtime_error("non-finite normalized-orbital curvature");
    }
    return curvature;
}

} // namespace xmvb::vb
