#pragma once

#include <stdexcept>
#include <Eigen/Cholesky>
#include <Eigen/Core>

namespace xmvb::vb {

/** @brief Positive upper factor O = R^T R of the active overlap. */
inline Eigen::MatrixXd overlap_upper(const Eigen::Ref<const Eigen::MatrixXd>& overlap) {
    Eigen::LLT<Eigen::MatrixXd> factor(overlap);
    if (factor.info() != Eigen::Success)
        throw std::runtime_error("active overlap is singular");
    return factor.matrixU();
}

inline Eigen::MatrixXd upper_inverse(const Eigen::Ref<const Eigen::MatrixXd>& r) {
    return r.triangularView<Eigen::Upper>().solve(Eigen::MatrixXd::Identity(r.rows(), r.cols()));
}

/** @brief First derivative of R along a symmetric overlap direction. */
inline Eigen::MatrixXd overlap_upper_direction(const Eigen::Ref<const Eigen::MatrixXd>& r,
                                               const Eigen::Ref<const Eigen::MatrixXd>& direction) {
    const Eigen::MatrixXd inverse = upper_inverse(r);
    const Eigen::MatrixXd normalized = inverse.transpose() * direction * inverse;
    Eigen::MatrixXd generator = normalized.triangularView<Eigen::Upper>();
    generator.diagonal() *= 0.5;
    return generator * r;
}

/** @brief Symmetric Frobenius adjoint of the upper Cholesky direction. */
inline Eigen::MatrixXd
overlap_upper_gradient(const Eigen::Ref<const Eigen::MatrixXd>& r,
                       const Eigen::Ref<const Eigen::MatrixXd>& upper_gradient) {
    const Eigen::MatrixXd inverse = upper_inverse(r);
    Eigen::MatrixXd weighted = upper_gradient * r.transpose();
    weighted.triangularView<Eigen::StrictlyLower>().setZero();
    weighted.diagonal().setZero();
    Eigen::MatrixXd normalized = 0.5 * (weighted + weighted.transpose());
    normalized.diagonal() = 0.5 * (upper_gradient * r.transpose()).diagonal();
    return inverse * normalized * inverse.transpose();
}

/** @brief Full directional derivative of the Cholesky adjoint. */
inline Eigen::MatrixXd
overlap_upper_gradient_direction(const Eigen::Ref<const Eigen::MatrixXd>& r,
                                 const Eigen::Ref<const Eigen::MatrixXd>& dr,
                                 const Eigen::Ref<const Eigen::MatrixXd>& gradient,
                                 const Eigen::Ref<const Eigen::MatrixXd>& dgradient) {
    const Eigen::MatrixXd inverse = upper_inverse(r);
    const Eigen::MatrixXd inverse_direction = -inverse * dr * inverse;
    const auto symmetric_upper = [](const Eigen::MatrixXd& x) {
        Eigen::MatrixXd upper = x.triangularView<Eigen::StrictlyUpper>();
        Eigen::MatrixXd result = 0.5 * (upper + upper.transpose());
        result.diagonal() = 0.5 * x.diagonal();
        return result;
    };
    const Eigen::MatrixXd normalized = symmetric_upper(gradient * r.transpose());
    const Eigen::MatrixXd dnormalized =
        symmetric_upper(dgradient * r.transpose() + gradient * dr.transpose());
    return inverse_direction * normalized * inverse.transpose() +
           inverse * dnormalized * inverse.transpose() +
           inverse * normalized * inverse_direction.transpose();
}

} // namespace xmvb::vb
