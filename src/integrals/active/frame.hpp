#pragma once

#include <vector>
#include <Eigen/Core>

namespace xmvb::integrals {

/** @brief Fixed-frame tangent dC R^-1; an empty R denotes the raw frame. */
inline Eigen::MatrixXd active_frame_tangent(const Eigen::Ref<const Eigen::MatrixXd>& tangent,
                                            const Eigen::MatrixXd& upper) {
    if (upper.size() == 0)
        return tangent;
    return upper.transpose().triangularView<Eigen::Lower>().solve(tangent.transpose()).transpose();
}

/** @brief Pull back a coefficient cotangent by R^-T, without a four-index map. */
inline Eigen::MatrixXd active_frame_pullback(const Eigen::Ref<const Eigen::MatrixXd>& gradient,
                                             const Eigen::MatrixXd& upper) {
    if (upper.size() == 0)
        return gradient;
    return upper.triangularView<Eigen::Upper>().solve(gradient.transpose()).transpose();
}

/** @brief Express a raw one-electron cotangent with Cbar=C R^-1. */
inline std::vector<double> active_frame_one_electron_weight(const std::vector<double>& raw_weight,
                                                            const Eigen::MatrixXd& upper) {
    if (upper.size() == 0)
        return raw_weight;
    const int n = upper.rows();
    const Eigen::Map<const Eigen::MatrixXd> raw(raw_weight.data(), n, n);
    const Eigen::MatrixXd transformed = upper * raw * upper.transpose();
    return {transformed.data(), transformed.data() + transformed.size()};
}

} // namespace xmvb::integrals
