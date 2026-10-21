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

/** The device backend shares the host Davidson iteration and stopping logic. */
linalg::DavidsonResult solve_davidson(
    Context& context, const DeviceEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
    const linalg::DavidsonOptions& options);

linalg::DavidsonResult solve_davidson(
    Context& context, const DeviceEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
    const Eigen::Ref<const Eigen::MatrixXd>& initial_vectors,
    const linalg::DavidsonOptions& options);

}  // namespace xmvb::cuda
