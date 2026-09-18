#pragma once

#include <Eigen/Core>

namespace xmvb::vb {

/** @brief Matrix-free Hessian action in nonredundant orbital coordinates. */
class ReducedHvp {
public:
  virtual ~ReducedHvp() = default;
  virtual Eigen::VectorXd apply(const Eigen::VectorXd& reduced_direction) = 0;
  virtual Eigen::MatrixXd apply_batch(
      const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions);
};

}  // namespace xmvb::vb
