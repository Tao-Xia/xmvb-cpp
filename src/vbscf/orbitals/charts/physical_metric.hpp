#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/orbitals/charts/layout.hpp"
#include "vbscf/orbitals/preparation/input.hpp"

namespace xmvb::vb {

/**
 * @brief Accepted-point metric of the inactive subspace and projected active rays.
 *
 * This is the coupled quotient metric of Eq. (31) in the orbital-optimization
 * theory note. It measures the physical derivative of the same raw-coefficient
 * curve used by the optimizer and annihilates support-admissible orbital gauge
 * directions. No orbital Hessian or AO-by-AO directional projector is formed.
 */
class OrbitalPhysicalMetric {
 public:
  explicit OrbitalPhysicalMetric(const OrbitalPreparationInput& input);

  /** @brief Squared physical length of one packed raw-coefficient direction. */
  double squared_norm(
      const SparseParameterLayout& layout,
      const Eigen::Ref<const Eigen::VectorXd>& packed_direction) const;

  /** @brief Matrix-free cotangent action of the coupled pullback metric. */
  Eigen::VectorXd apply(
      const SparseParameterLayout& layout,
      const Eigen::Ref<const Eigen::VectorXd>& packed_direction) const;

 private:
  Eigen::VectorXd physical_feature(
      const SparseParameterLayout& layout,
      const Eigen::Ref<const Eigen::VectorXd>& packed_direction) const;
  Eigen::MatrixXd overlap_;
  Eigen::MatrixXd ao_metric_upper_;
  Eigen::MatrixXd inactive_;
  Eigen::MatrixXd active_;
  Eigen::MatrixXd inactive_inverse_;
  Eigen::MatrixXd inactive_inverse_upper_;
  Eigen::MatrixXd complement_;
  Eigen::MatrixXd projected_active_;
  Eigen::VectorXd projected_norm_squared_;
  std::vector<std::vector<int>> orbital_supports_;
  int n_basis_functions_ = 0;
  int n_inactive_ = 0;
  int n_active_ = 0;
};

}  // namespace xmvb::vb
