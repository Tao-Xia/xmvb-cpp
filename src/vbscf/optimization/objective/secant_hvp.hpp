#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/optimization/driver/types.hpp"
#include "vbscf/optimization/objective/reduced_hvp.hpp"
#include "vbscf/orbitals/charts/chart.hpp"

namespace xmvb::vb {

/**
 * @brief Symmetric least-change correction assembled from secant residuals.
 *
 * Each update is a Powell-symmetric-Broyden correction. The newest admitted
 * pair is satisfied exactly while the complete operator remains symmetric;
 * indefinite missing curvature is retained for the trust-region solver.
 */
class SymmetricSecantCorrection {
public:
  bool add_pair(
      const Eigen::VectorXd& step,
      const Eigen::VectorXd& target_image);
  Eigen::VectorXd apply(const Eigen::VectorXd& vector) const;
  Eigen::MatrixXd apply_batch(
      const Eigen::Ref<const Eigen::MatrixXd>& vectors) const;
  int size() const noexcept;

private:
  struct Update {
    Eigen::VectorXd step;
    Eigen::VectorXd residual;
    double inverse_step_norm_squared = 0.0;
    double residual_step_inner_product = 0.0;
  };

  std::vector<Update> updates_;
};

/** @brief Core HVP augmented by transported accepted-step secants. */
class SecantCorrectedCoreHvp final : public ReducedHvp {
public:
  SecantCorrectedCoreHvp(
      ExactReducedHvp* exact_hvp,
      const SymmetricSecantCorrection* correction);

  Eigen::VectorXd apply(const Eigen::VectorXd& reduced_direction) override;
  Eigen::MatrixXd apply_batch(
      const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions) override;
  int correction_size() const noexcept;

private:
  ExactReducedHvp* exact_hvp_ = nullptr;
  const SymmetricSecantCorrection* correction_ = nullptr;
};

/**
 * @brief Builds the accepted-point secant correction with one core-HVP block.
 *
 * The result depends only on the accepted orbital point and transported
 * history, so trust-radius retries at that point must reuse it.
 */
SymmetricSecantCorrection build_symmetric_secant_correction(
    ExactReducedHvp* exact_hvp,
    const OrbitalChart& current_space,
    const std::vector<PackedSecantPair>& packed_secant_history,
    int max_history_size);

}  // namespace xmvb::vb
