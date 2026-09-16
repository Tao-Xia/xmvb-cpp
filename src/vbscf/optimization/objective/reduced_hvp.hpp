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

/**
 * @brief Symmetric matrix-free interpolation of expensive response actions.
 *
 * Orthonormal sampled directions Q and exact response images Y define the
 * least-change symmetric operator
 *
 *     B = Q Yc^T + Yc Q^T - Q (Q^T Yc) Q^T,
 *
 * where Yc is the closest sampled image block whose projected cross matrix is
 * symmetric.  The representation stores O(n r) values for rank r and never
 * forms an n-by-n Hessian.
 */
class SymmetricResponseModel {
public:
  /** @brief Adds an independent exact response sample. */
  bool add(
      const Eigen::VectorXd& direction,
      const Eigen::VectorXd& exact_response);

  Eigen::VectorXd apply(const Eigen::VectorXd& direction) const;
  Eigen::MatrixXd apply_batch(
      const Eigen::Ref<const Eigen::MatrixXd>& directions) const;

  Eigen::Index dimension() const noexcept;
  Eigen::Index rank() const noexcept;

private:
  void rebuild_symmetric_images();

  Eigen::MatrixXd directions_;
  Eigen::MatrixXd exact_images_;
  Eigen::MatrixXd symmetric_images_;
  Eigen::MatrixXd projected_response_;
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
  Eigen::MatrixXd apply_outer_batch(
      const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions);
  bool supports_analytic_core_model() const noexcept;
  ExactHvpOperator::Diagnostics diagnostics() const;

private:
  ExactHvpOperator exact_operator_;
};

/** @brief Cheap core HVP augmented by audited outer-response secants. */
class ResponseCorrectedHvp final : public ReducedHvp {
public:
  explicit ResponseCorrectedHvp(ExactReducedHvp* exact_hvp);

  Eigen::VectorXd apply(const Eigen::VectorXd& reduced_direction) override;
  Eigen::MatrixXd apply_batch(
      const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions) override;

  Eigen::VectorXd exact_outer_response(
      const Eigen::VectorXd& reduced_direction);
  Eigen::VectorXd modeled_outer_response(
      const Eigen::VectorXd& reduced_direction) const;
  bool add_outer_sample(
      const Eigen::VectorXd& reduced_direction,
      const Eigen::VectorXd& exact_outer_response);
  Eigen::Index response_rank() const noexcept;

private:
  ExactReducedHvp* exact_hvp_;
  SymmetricResponseModel response_model_;
};

std::string build_hvp_error(const ExactReducedHvp& hvp);

}  // namespace xmvb::vb
