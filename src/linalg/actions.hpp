#pragma once

#include <functional>
#include <Eigen/Core>

namespace xmvb::linalg {

/** @brief Matrix-free action on caller-owned vector coordinates. */
using VectorAction = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

} // namespace xmvb::linalg
