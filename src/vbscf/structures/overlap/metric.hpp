#pragma once

#include <stdexcept>
#include <utility>

#include <Eigen/Cholesky>

namespace xmvb::vb {

/** @brief Bounded selected-structure Gram and its shared triangular factor.
 * Constructed only when the selected Gram is already resident. This owns no
 * spin-product array and is shared across CI and derivative consumers at one
 * fixed orbital chart; a trial chart requires its own instance.
 */
struct SelectedMetric {
    explicit SelectedMetric(Eigen::MatrixXd matrix) : overlap(std::move(matrix)) {
        const Eigen::LLT<Eigen::MatrixXd> factor(overlap);
        if (factor.info() != Eigen::Success)
            throw std::runtime_error("selected CI metric is not positive definite");
        upper = factor.matrixU();
    }

    /** @brief Retain a physical-image QR factor without forming normal equations first. */
    static SelectedMetric from_upper(Eigen::MatrixXd upper) {
        if (upper.rows() != upper.cols() || !upper.allFinite() ||
            (upper.diagonal().array() == 0).any())
            throw std::runtime_error("selected physical-image factor is singular");
        Eigen::MatrixXd matrix = upper.transpose() * upper;
        return SelectedMetric(std::move(matrix), std::move(upper));
    }

    Eigen::MatrixXd apply(const Eigen::Ref<const Eigen::MatrixXd>& c) const {
        return upper.transpose() * (upper * c);
    }

    Eigen::MatrixXd solve(const Eigen::Ref<const Eigen::MatrixXd>& rhs) const {
        const Eigen::MatrixXd first = upper.transpose().triangularView<Eigen::Lower>().solve(rhs);
        return upper.triangularView<Eigen::Upper>().solve(first);
    }

    Eigen::MatrixXd overlap;
    Eigen::MatrixXd upper;

  private:
    SelectedMetric(Eigen::MatrixXd matrix, Eigen::MatrixXd factor)
        : overlap(std::move(matrix)), upper(std::move(factor)) {}
};

} // namespace xmvb::vb
