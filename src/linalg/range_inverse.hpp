#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

namespace xmvb::linalg {

/** @brief Signed inverse on the numerically resolved range of a small symmetric matrix.
 *
 * A zero coarse eigenvalue is not a loss of a physical variational direction.
 * Its coarse correction vanishes; the balancing preconditioner's base action
 * remains available. The numerical rank uses only the matrix's floating-point
 * scale, not an optimization tolerance or an absolute curvature cutoff.
 */
class RangeInverse {
  public:
    explicit RangeInverse(Eigen::MatrixXd matrix, const Eigen::VectorXd& metric_diagonal)
        : matrix_(std::move(matrix)) {
        if (matrix_.rows() != matrix_.cols() || !matrix_.allFinite())
            throw std::invalid_argument("range inverse requires a finite symmetric matrix");
        if (metric_diagonal.size() != matrix_.rows() || !metric_diagonal.allFinite() ||
            (metric_diagonal.array() <= 0.0).any())
            throw std::invalid_argument("range inverse requires positive physical metric norms");
        // E a=b, a=D z, D_ii=(v_i^T M v_i)^(-1/2): solve
        // (D E D) z=D b. Numerical rank must not depend on the units of
        // the orbital/CI samples. This changes neither E nor its full-rank inverse.
        scale_ = metric_diagonal.array().sqrt().inverse();
        matrix_ = (scale_.asDiagonal() * matrix_ * scale_.asDiagonal()).eval();
        if (!matrix_.allFinite())
            throw std::runtime_error("non-finite physically scaled coarse matrix");
        factor_.compute(matrix_);
        rank_ = factor_.rank();
        if (rank_ == matrix_.rows())
            return;
        rank_ = 0;
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(matrix_);
        if (spectrum.info() != Eigen::Success)
            throw std::runtime_error("coarse range eigensolver failed");
        vectors_ = spectrum.eigenvectors();
        inverse_ = spectrum.eigenvalues();
        const double scale = inverse_.size() ? inverse_.cwiseAbs().maxCoeff() : 0.0;
        const double cutoff = std::numeric_limits<double>::epsilon() *
                              std::max<Eigen::Index>(1, inverse_.size()) * scale;
        for (Eigen::Index i = 0; i < inverse_.size(); ++i) {
            if (std::abs(inverse_[i]) > cutoff) {
                inverse_[i] = 1.0 / inverse_[i];
                ++rank_;
            } else {
                inverse_[i] = 0.0;
            }
        }
    }

    /** @brief Solve the range component, retaining signed resolved curvatures. */
    Eigen::VectorXd solve(const Eigen::VectorXd& rhs) const {
        const Eigen::VectorXd scaled_rhs = scale_.cwiseProduct(rhs);
        Eigen::VectorXd result, target = scaled_rhs;
        if (vectors_.size() == 0) {
            result = factor_.solve(scaled_rhs);
        } else {
            const Eigen::VectorXd coordinates = vectors_.transpose() * scaled_rhs;
            result.noalias() = vectors_ * inverse_.cwiseProduct(coordinates);
            Eigen::VectorXd projected = coordinates;
            for (Eigen::Index i = 0; i < projected.size(); ++i)
                if (inverse_[i] == 0.0)
                    projected[i] = 0.0;
            target.noalias() = vectors_ * projected;
        }
        const double bound = 256 * std::numeric_limits<double>::epsilon() * matrix_.rows() *
                             std::max(1.0, matrix_.norm() * result.norm() + scaled_rhs.norm());
        if (!result.allFinite() || (matrix_ * result - target).norm() > bound)
            throw std::runtime_error("coarse range inverse failed its residual certificate");
        return scale_.cwiseProduct(result);
    }

    Eigen::Index rank() const noexcept {
        return rank_;
    }

  private:
    Eigen::MatrixXd matrix_, vectors_;
    Eigen::FullPivLU<Eigen::MatrixXd> factor_;
    Eigen::VectorXd inverse_, scale_;
    Eigen::Index rank_ = 0;
};

} // namespace xmvb::linalg
