#pragma once

#include <stdexcept>
#include <utility>

#include <Eigen/Cholesky>
#include "core/index.hpp"
#include "core/workspace.hpp"

namespace xmvb::vb {

/** @brief Bounded selected-structure Gram and its shared triangular factor.
 * Constructed only when the selected Gram is already resident. This owns no
 * spin-product array and is shared across CI and derivative consumers at one
 * fixed orbital chart; a trial chart requires its own instance.
 */
struct SelectedMetric {
  private:
    workspace::Reservation residency_;
    static workspace::Reservation reserve(Eigen::Index dimension, std::size_t matrices = 2) {
        const auto n = core::checked_index<std::size_t>(dimension);
        const auto bytes =
            core::checked_multiply(matrices * sizeof(double), core::checked_multiply(n, n));
        workspace::Reservation memory;
        if (!memory.try_resize(bytes))
            throw std::length_error("selected metric exceeds shared workspace");
        return memory;
    }

  public:
    explicit SelectedMetric(Eigen::MatrixXd matrix)
        : residency_(reserve(matrix.rows(), 3)), overlap(std::move(matrix)) {
        {
            const Eigen::LLT<Eigen::MatrixXd> factor(overlap);
            if (factor.info() != Eigen::Success)
                throw std::runtime_error("selected CI metric is not positive definite");
            upper = factor.matrixU();
        }
        residency_.try_resize(residency_.bytes() / 3 * 2);
    }

    /** @brief Retain a physical-image QR factor without forming normal equations first. */
    static SelectedMetric from_upper(Eigen::MatrixXd upper, workspace::Reservation memory = {}) {
        if (upper.rows() != upper.cols() || !upper.allFinite() ||
            (upper.diagonal().array() == 0).any())
            throw std::runtime_error("selected physical-image factor is singular");
        if (!memory.bytes())
            memory = reserve(upper.rows());
        Eigen::MatrixXd matrix = upper.transpose() * upper;
        return SelectedMetric(std::move(matrix), std::move(upper), std::move(memory));
    }

    Eigen::MatrixXd apply(const Eigen::Ref<const Eigen::MatrixXd>& c) const {
        return upper.transpose() * (upper * c);
    }

    /** @brief Physical state Gram, without cancellation through C^T M C. */
    Eigen::MatrixXd gram(const Eigen::Ref<const Eigen::MatrixXd>& c) const {
        const Eigen::MatrixXd images = upper * c;
        return images.transpose() * images;
    }

    Eigen::MatrixXd solve(const Eigen::Ref<const Eigen::MatrixXd>& rhs) const {
        const Eigen::MatrixXd first = upper.transpose().triangularView<Eigen::Lower>().solve(rhs);
        return upper.triangularView<Eigen::Upper>().solve(first);
    }

    Eigen::MatrixXd overlap;
    Eigen::MatrixXd upper;

  private:
    SelectedMetric(Eigen::MatrixXd matrix, Eigen::MatrixXd factor, workspace::Reservation memory)
        : residency_(std::move(memory)), overlap(std::move(matrix)), upper(std::move(factor)) {}
};

} // namespace xmvb::vb
