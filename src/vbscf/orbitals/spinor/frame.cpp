#include "vbscf/orbitals/spinor/frame.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include <Eigen/Cholesky>

namespace xmvb::vb::spinor {

Frame::Frame(const Eigen::Ref<const Eigen::MatrixXcd>& coefficients,
             const Eigen::Ref<const Eigen::MatrixXd>& ao_overlap) {
    const Eigen::Index n_bf = ao_overlap.rows();
    const Eigen::Index n_spinor = coefficients.cols();
    if (n_bf <= 0 || ao_overlap.cols() != n_bf || n_spinor <= 0 ||
        coefficients.rows() != 2 * n_bf || n_spinor > 2 * n_bf || !coefficients.allFinite() ||
        !ao_overlap.allFinite()) {
        throw std::invalid_argument("invalid spinor frame dimensions or coefficients");
    }
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon();
    if ((ao_overlap - ao_overlap.transpose()).norm() >
        tolerance * n_bf * std::max(1.0, ao_overlap.norm())) {
        throw std::invalid_argument("spinor AO overlap must be symmetric");
    }
    Eigen::LLT<Eigen::MatrixXd> ao_factor(ao_overlap);
    if (ao_factor.info() != Eigen::Success) {
        throw std::invalid_argument("spinor AO overlap must be positive definite");
    }

    // Avoid a second, explicitly allocated (2*n_bf)^2 AO metric.
    gram_ = coefficients.topRows(n_bf).adjoint() * ao_overlap * coefficients.topRows(n_bf);
    gram_.noalias() +=
        coefficients.bottomRows(n_bf).adjoint() * ao_overlap * coefficients.bottomRows(n_bf);
    if (!gram_.allFinite()) {
        throw std::invalid_argument("spinor Gram matrix is not finite");
    }
    Eigen::LLT<Eigen::MatrixXcd> factor(gram_);
    if (factor.info() != Eigen::Success) {
        throw std::invalid_argument("spinor frame is linearly dependent");
    }
    upper_ = factor.matrixU();
    const double smallest_pivot = upper_.diagonal().cwiseAbs2().minCoeff();
    const double gram_scale = gram_.diagonal().real().maxCoeff();
    if (smallest_pivot <= std::numeric_limits<double>::epsilon() * n_spinor * gram_scale) {
        throw std::invalid_argument("spinor frame is numerically rank deficient");
    }

    // Solve R^T * tilde_b^T = b^T. This is a transpose solve, not R^dagger.
    coefficients_ = upper_.transpose()
                        .triangularView<Eigen::Lower>()
                        .solve(coefficients.transpose())
                        .transpose();
    if (!coefficients_.allFinite()) {
        throw std::runtime_error("spinor computational frame is not finite");
    }
}

} // namespace xmvb::vb::spinor
