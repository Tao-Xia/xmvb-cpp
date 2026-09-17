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
  Eigen::VectorXd apply_core(const Eigen::VectorXd& reduced_direction);
  Eigen::MatrixXd apply_core_batch(
      const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions);
  Eigen::VectorXd apply_outer(const Eigen::VectorXd& reduced_direction);
  std::size_t core_direction_count() const noexcept;
  std::size_t outer_response_direction_count() const noexcept;
  bool supports_analytic_core_model() const noexcept;
  ExactHvpOperator::Diagnostics diagnostics() const;

private:
  ExactHvpOperator exact_operator_;
  std::size_t core_direction_count_ = 0;
  std::size_t outer_response_direction_count_ = 0;
};

/** @brief Core-only reduced HVP used by the inexpensive fidelity level. */
class CoreReducedHvp final : public ReducedHvp {
public:
  explicit CoreReducedHvp(ExactReducedHvp* exact_hvp);

  Eigen::VectorXd apply(const Eigen::VectorXd& reduced_direction) override;
  Eigen::MatrixXd apply_batch(
      const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions) override;

private:
  ExactReducedHvp* exact_hvp_ = nullptr;
};

std::string build_hvp_error(const ExactReducedHvp& hvp);

}  // namespace xmvb::vb
