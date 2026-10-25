#pragma once

#include <stdexcept>
#include <utility>

#include <Eigen/Cholesky>

namespace xmvb::vb {

/** @brief Bounded selected-structure Gram and its shared positive factor.
 * Constructed only when the selected Gram is already resident. This owns no
 * spin-product array and is shared across CI and derivative consumers at one
 * fixed orbital chart; a trial chart requires its own instance.
 */
struct SelectedMetric {
    explicit SelectedMetric(Eigen::MatrixXd matrix) : overlap(std::move(matrix)) {
        factor.compute(overlap);
        if (factor.info() != Eigen::Success)
            throw std::runtime_error("selected CI metric is not positive definite");
    }

    Eigen::MatrixXd overlap;
    Eigen::LLT<Eigen::MatrixXd> factor;
};

} // namespace xmvb::vb
