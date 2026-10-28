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
    explicit RangeInverse(Eigen::MatrixXd matrix) : matrix_(std::move(matrix)) {
        if (matrix_.rows() != matrix_.cols() || !matrix_.allFinite())
            throw std::invalid_argument("range inverse requires a finite symmetric matrix");
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
        Eigen::VectorXd result, target = rhs;
        if (vectors_.size() == 0) {
            result = factor_.solve(rhs);
        } else {
            const Eigen::VectorXd coordinates = vectors_.transpose() * rhs;
            result.noalias() = vectors_ * inverse_.cwiseProduct(coordinates);
            Eigen::VectorXd projected = coordinates;
            for (Eigen::Index i = 0; i < projected.size(); ++i)
                if (inverse_[i] == 0.0)
                    projected[i] = 0.0;
            target.noalias() = vectors_ * projected;
        }
        const double bound = 256 * std::numeric_limits<double>::epsilon() * matrix_.rows() *
                             std::max(1.0, matrix_.norm() * result.norm() + rhs.norm());
        if (!result.allFinite() || (matrix_ * result - target).norm() > bound)
            throw std::runtime_error("coarse range inverse failed its residual certificate");
        return result;
    }

    Eigen::Index rank() const noexcept {
        return rank_;
    }

  private:
    Eigen::MatrixXd matrix_, vectors_;
    Eigen::FullPivLU<Eigen::MatrixXd> factor_;
    Eigen::VectorXd inverse_;
    Eigen::Index rank_ = 0;
};

} // namespace xmvb::linalg
