#pragma once

#include <cstdint>

#include <Eigen/Core>

namespace xmvb::vb {

/** @brief Matrix-free Hessian action in nonredundant orbital coordinates. */
class ReducedHvp {
public:
  virtual ~ReducedHvp() = default;
  virtual Eigen::VectorXd apply(const Eigen::VectorXd& reduced_direction) = 0;
  virtual Eigen::MatrixXd apply_batch(
      const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions);

  /** @brief Frozen linear action, without changing the response model. */
  virtual Eigen::MatrixXd apply_frozen_batch(
      const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions);

  /** @brief Changes whenever old Hessian images require refresh. */
  virtual std::uint64_t model_revision() const noexcept { return 0; }
};

}  // namespace xmvb::vb
