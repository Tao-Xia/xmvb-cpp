#pragma once

#include <vector>

#include <Eigen/Core>

namespace xmvb::pfaffian_vbscf {

/**
 * @brief Canonical dense matrix type used by the Pfaffian-VBSCF modules.
 *
 * Col-major storage matches Eigen's default BLAS-friendly layout and keeps the
 * tensor contraction kernels compatible with the rest of the codebase.
 */
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

/**
 * @brief Row-major dense matrix alias for explicit layout-sensitive utilities.
 */
using RowMajorMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/**
 * @brief Canonical dense vector type used by the Pfaffian-VBSCF modules.
 */
using Vector = Eigen::VectorXd;

/**
 * @brief Mutable Eigen matrix view.
 */
using MatrixRef = Eigen::Ref<Matrix>;

/**
 * @brief Read-only Eigen matrix view.
 */
using ConstMatrixRef = Eigen::Ref<const Matrix>;

/**
 * @brief Canonical scalar buffer type for packed tensors and gradients.
 */
using ScalarBuffer = std::vector<double>;

}  // namespace xmvb::pfaffian_vbscf
