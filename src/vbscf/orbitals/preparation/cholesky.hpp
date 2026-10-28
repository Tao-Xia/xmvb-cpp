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

/** @brief Metric QR with a second Cholesky pass, without factoring the AO metric.
 * The second pass measures the physical frame rather than reusing its rounded
 * Gram matrix. Both passes use triangular solves; no orbital rank is removed.
 */
struct ActiveFrame {
    Eigen::MatrixXd upper;
    Eigen::MatrixXd coefficients;
};

inline ActiveFrame active_frame(const Eigen::Ref<const Eigen::MatrixXd>& coefficients,
                                const Eigen::Ref<const Eigen::MatrixXd>& ao_overlap,
                                const Eigen::Ref<const Eigen::MatrixXd>& overlap) {
    ActiveFrame frame;
    frame.upper = overlap_upper(overlap);
    frame.coefficients = frame.upper.transpose()
                             .triangularView<Eigen::Lower>()
                             .solve(coefficients.transpose())
                             .transpose();
    const Eigen::MatrixXd metric = frame.coefficients.transpose() * ao_overlap * frame.coefficients;
    const Eigen::MatrixXd correction = overlap_upper(metric);
    frame.coefficients = correction.transpose()
                             .triangularView<Eigen::Lower>()
                             .solve(frame.coefficients.transpose())
                             .transpose()
                             .eval();
    frame.upper = (correction * frame.upper).eval();
    return frame;
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
