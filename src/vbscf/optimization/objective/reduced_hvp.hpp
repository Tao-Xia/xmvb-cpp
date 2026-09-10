#pragma once

#include <string>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "vbscf/optimization/objective/function.hpp"
#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/layout.hpp"

namespace xmvb::vb {

class ReducedHvpOperator {
public:
  virtual ~ReducedHvpOperator() = default;
  virtual Eigen::VectorXd apply(const Eigen::VectorXd& reduced_direction) = 0;
  virtual Eigen::MatrixXd apply_batch(
      const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions);
};

class FullFiniteDifferenceReducedHvpOperator final : public ReducedHvpOperator {
public:
  FullFiniteDifferenceReducedHvpOperator(
      const VbScfObjective& objective,
      const OrbitalChart& current_space,
      const OrbitalChart::ProjectionResult& current_projection,
      const OrbitalPreparationInput& current_orbital_input,
      const SparseParameterLayout& parameter_view,
      double hvp_step_size);

  Eigen::VectorXd apply(const Eigen::VectorXd& reduced_direction) override;

private:
  VbScfObjective probe_objective_;
  const OrbitalChart& current_space_;
  Eigen::VectorXd current_reduced_gradient_;
  OrbitalPreparationInput current_orbital_input_;
  SparseParameterLayout parameter_view_;
  double hvp_step_size_ = 0.0;
};

class ExactContextReducedHvpOperator final : public ReducedHvpOperator {
public:
  ExactContextReducedHvpOperator(
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

std::string build_exact_ctx_unavailable_message(
    const ExactContextReducedHvpOperator& hvp_operator);

}  // namespace xmvb::vb
