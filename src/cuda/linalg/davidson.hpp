#pragma once

#include <functional>

#include <Eigen/Core>

#include "cuda/memory/context.hpp"
#include "linalg/eigensolver.hpp"
#include "linalg/views.hpp"

namespace xmvb::cuda {

/**
 * Apply H and S to a block already on the device. The output views have the
 * same shape as the input. For identity_overlap, overlap_images is empty and
 * the callback must leave it untouched. All work must use Context::stream().
 */
using DeviceEigenAction = std::function<void(
    linalg::ConstMatrixView vectors,
    linalg::MatrixView hamiltonian_images,
    linalg::MatrixView overlap_images)>;

/** @brief Positive inverse image of a residual, entirely on Context's stream.
 * The shared Davidson loop supplies the minus sign. Implementations must not
 * download the residual or invoke a host numerical preconditioner.
 */
using DeviceEigenPreconditioner = std::function<void(
    linalg::ConstMatrixView residual, double energy, linalg::MatrixView image)>;

/** The device backend shares the host Davidson iteration and stopping logic. */
linalg::DavidsonResult solve_davidson(
    Context& context, const DeviceEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
    const linalg::DavidsonOptions& options,
    const DeviceEigenPreconditioner& preconditioner = {});

linalg::DavidsonResult solve_davidson(
    Context& context, const DeviceEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
    const Eigen::Ref<const Eigen::MatrixXd>& initial_vectors,
    const linalg::DavidsonOptions& options,
    const DeviceEigenPreconditioner& preconditioner = {});

}  // namespace xmvb::cuda
