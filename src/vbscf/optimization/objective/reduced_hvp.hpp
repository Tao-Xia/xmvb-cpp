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

  /**
   * @brief Updates stored Hessian images after a model revision.
   *
   * Implementations return false when no exact incremental update is
   * available, in which case the caller must replay the frozen action.
   */
  virtual bool update_images(
      std::uint64_t previous_revision,
      const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions,
      Eigen::MatrixXd* images);

  /** @brief Changes whenever old Hessian images require refresh. */
  virtual std::uint64_t model_revision() const noexcept { return 0; }
};

}  // namespace xmvb::vb
