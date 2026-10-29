#pragma once

#include <vector>
#include <Eigen/Core>
#include "core/workspace.hpp"

namespace xmvb::linalg {

/** @brief Reusable LAPACK workspace for in-place column-major QR.
 * The upper triangle becomes R; the lower triangle holds Householder vectors.
 * No copy of the input matrix or explicit Q is allocated.
 */
class QrWorkspace {
  public:
    explicit QrWorkspace(std::size_t maximum_workspace) : maximum_workspace_(maximum_workspace) {}
    void compute(Eigen::Ref<Eigen::MatrixXd> matrix);

  private:
    Eigen::Index columns_ = -1;
    std::vector<double> tau_, work_;
    std::size_t maximum_workspace_;
    workspace::Temporary residency_;
};

} // namespace xmvb::linalg
