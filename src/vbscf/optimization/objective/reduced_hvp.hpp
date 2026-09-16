#pragma once

#include <string>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "vbscf/optimization/objective/function.hpp"
#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/layout.hpp"

namespace xmvb::vb {

/** @brief Matrix-free Hessian action in nonredundant orbital coordinates. */
class ReducedHvp {
public:
  virtual ~ReducedHvp() = default;
  virtual Eigen::VectorXd apply(const Eigen::VectorXd& reduced_direction) = 0;
  virtual Eigen::MatrixXd apply_batch(
      const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions);
};

/** @brief Analytic reduced Hessian action at the accepted VBSCF point. */
class ExactReducedHvp final : public ReducedHvp {
public:
  ExactReducedHvp(
      const VbScfObjective& objective,
      const OrbitalChart& current_space);

  Eigen::VectorXd apply(const Eigen::VectorXd& reduced_direction) override;
  Eigen::MatrixXd apply_batch(
      const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions) override;
  bool supports_analytic_core_model() const noexcept;
  ExactHvpOperator::Diagnostics diagnostics() const;

private:
  ExactHvpOperator exact_operator_;
};

std::string build_hvp_error(const ExactReducedHvp& hvp);

}  // namespace xmvb::vb
