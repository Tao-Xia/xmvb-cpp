#pragma once

#include <cstddef>
#include <functional>
#include <limits>

#include <Eigen/Core>

#include "vbscf/optimization/neo/solver.hpp"

namespace xmvb::vb {

/** @brief Vector in separated orbital and horizontal-structure coordinates. */
struct ResponseNeoDirection {
  Eigen::VectorXd orbital;
  Eigen::VectorXd structure;
};

/** @brief Common-revision solutions of @f$CZ=-F@f$ for a forcing block. */
struct ResponseNeoStructureResponse {
  Eigen::MatrixXd coordinates;
  /** @brief Columns are the matching @f$B^Tz@f$ images. */
  Eigen::MatrixXd orbital_images;
  /** @brief Exact horizontal @f$F+Cz@f$ bordered residual columns. */
  Eigen::MatrixXd equation_residuals;
  std::uint64_t revision = 0;
  int block_actions = 0;
  double max_relative_residual = 0.0;
};

/** @brief One block column of the coupled orbital--structure Hessian. */
using ResponseNeoBlockAction =
    std::function<ResponseNeoDirection(const Eigen::VectorXd&)>;
using ResponseNeoPreconditioner =
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, double)>;
using ResponseNeoStructureSolver = std::function<ResponseNeoStructureResponse(
    const Eigen::Ref<const Eigen::MatrixXd>&, double)>;

/**
 * @brief Quadratic model for orbital-trust NEO with projected response.
 *
 * The coupled action represents
 * @f$[\bar A p+B^Tz,\;Bp+Cz]@f$.  Only the orbital metric defines the
 * trust region.  The optional preconditioner acts only on orbital covectors.
 */
class ResponseNeoProblem {
public:
  ResponseNeoProblem(
      Eigen::VectorXd orbital_gradient,
      Eigen::Index structure_size,
      ResponseNeoBlockAction apply_orbital_coupling,
      ResponseNeoStructureSolver solve_structure_response,
      NeoAction apply_orbital_metric,
      ResponseNeoPreconditioner apply_orbital_preconditioner = {},
      Eigen::VectorXd initial_orbital_guess = {},
      double operator_relative_accuracy = 0.0);

  Eigen::Index orbital_size() const noexcept {
    return orbital_gradient_.size();
  }
  Eigen::Index structure_size() const noexcept { return structure_size_; }
  const Eigen::VectorXd& orbital_gradient() const noexcept {
    return orbital_gradient_;
  }

  ResponseNeoDirection apply_orbital_coupling(
      const Eigen::VectorXd& direction) const;
  ResponseNeoStructureResponse solve_structure_response(
      const Eigen::Ref<const Eigen::MatrixXd>& forcing,
      double relative_tolerance) const;
  Eigen::VectorXd apply_orbital_metric(
      const Eigen::VectorXd& direction) const;
  bool has_orbital_preconditioner() const noexcept {
    return static_cast<bool>(apply_orbital_preconditioner_);
  }
  Eigen::VectorXd apply_orbital_preconditioner(
      const Eigen::VectorXd& covector,
      double shift) const;
  /** @brief Initial Davidson direction recycled within one nonlinear macro. */
  const Eigen::VectorXd& initial_orbital_guess() const noexcept {
    return initial_orbital_guess_;
  }
  double operator_relative_accuracy() const noexcept {
    return operator_relative_accuracy_;
  }

private:
  Eigen::VectorXd apply_orbital_checked(
      const NeoAction& action,
      const Eigen::VectorXd& vector,
      const char* message) const;

  const Eigen::VectorXd orbital_gradient_;
  const Eigen::Index structure_size_;
  ResponseNeoDirection apply_block_checked(
      const ResponseNeoBlockAction& action,
      const Eigen::VectorXd& vector,
      Eigen::Index expected_size) const;

  const ResponseNeoBlockAction apply_orbital_coupling_;
  const ResponseNeoStructureSolver solve_structure_response_;
  const NeoAction apply_orbital_metric_;
  const ResponseNeoPreconditioner apply_orbital_preconditioner_;
  const Eigen::VectorXd initial_orbital_guess_;
  const double operator_relative_accuracy_;
};

/** @brief Step and explicit residuals of projected-response NEO. */
struct ResponseNeoResult {
  ResponseNeoDirection step;
  ResponseNeoDirection hessian_step;
  Eigen::VectorXd orbital_metric_step;
  ResponseNeoDirection kkt_residual;
  Eigen::VectorXd minimum_curvature_orbital;
  double shift = 0.0;
  double predicted_reduction = 0.0;
  double step_norm = 0.0;
  double residual_norm = 0.0;
  double residual_target = 0.0;
  double curvature_residual_norm = 0.0;
  double curvature_residual_target = 0.0;
  int iterations = 0;
  int coupled_actions = 0;
  int orbital_actions = 0;
  int structure_actions = 0;
  int projected_model_builds = 0;
  bool boundary = false;
  bool hard_case = false;
  bool global_curvature_certified = false;
  NeoStopReason stop_reason = NeoStopReason::SubspaceLimit;

  bool converged() const noexcept {
    return stop_reason == NeoStopReason::Converged;
  }
};

/**
 * @brief Reusable projected-response NEO subspace at one accepted point.
 *
 * The workspace owns an orbital basis and the common-revision structure
 * responses to all of its coupling columns. Changing only the trust radius
 * therefore requires no repeated large-space action.
 */
class ResponseNeoWorkspace {
public:
  explicit ResponseNeoWorkspace(const ResponseNeoProblem& problem);

  /** @brief Solves one trust-region subproblem using the cached subspace. */
  ResponseNeoResult solve(const NeoOptions& options);

  Eigen::Index orbital_basis_size() const noexcept {
    return orbital_basis_size_;
  }
  std::uint64_t structure_response_revision() const noexcept {
    return structure_response_revision_;
  }

private:
  struct ProjectedModel {
    std::size_t revision = std::numeric_limits<std::size_t>::max();
    Eigen::MatrixXd metric;
    Eigen::MatrixXd whitening;
    Eigen::MatrixXd eigenvectors;
    Eigen::VectorXd eigenvalues;
    Eigen::VectorXd white_gradient;
    Eigen::VectorXd minimum_curvature_orbital;
    Eigen::VectorXd curvature_orbital_residual;
    Eigen::VectorXd curvature_structure_residual;
    double response_closure_norm = 0.0;
    double response_closure_scale = 0.0;
    double curvature_orbital_scale = 0.0;
    double curvature_structure_scale = 0.0;
  };

  void reserve_orbital_column();
  void refresh_structure_response(
      double relative_tolerance,
      int* structure_actions);
  /**
   * @brief Re-solves one combined forcing to an explicit residual target.
   *
   * Retained response columns are deliberately only model-accurate.  This
   * routine is used for the current Newton or curvature combination when its
   * actual bordered-equation residual is too large for certification.
   */
  ResponseNeoStructureResponse refine_structure_direction(
      const Eigen::VectorXd& forcing,
      double residual_target,
      int* structure_actions) const;
  /** @brief Adds a certified combined response as a symmetric secant sample. */
  bool update_response_model(
      const Eigen::VectorXd& coefficients,
      const Eigen::VectorXd& refined_orbital_image,
      const Eigen::VectorXd& model_orbital_image);
  bool rebuild_projected_model();
  bool append_orbital(
      Eigen::VectorXd direction,
      int* actions,
      int* orbital_actions);

  const ResponseNeoProblem& problem_;
  Eigen::MatrixXd orbital_basis_;
  Eigen::MatrixXd orbital_metric_images_;
  Eigen::MatrixXd orbital_hessian_images_;
  Eigen::MatrixXd orbital_structure_images_;
  Eigen::MatrixXd structure_response_coordinates_;
  Eigen::MatrixXd structure_response_orbital_images_;
  Eigen::MatrixXd structure_response_residuals_;
  Eigen::MatrixXd response_model_correction_;
  Eigen::MatrixXd response_sample_coefficients_;
  Eigen::MatrixXd response_sample_orbital_defects_;
  Eigen::MatrixXd projected_orbital_hessian_;
  Eigen::MatrixXd projected_orbital_metric_;
  Eigen::Index orbital_basis_size_ = 0;
  Eigen::Index orbital_canonical_ = 0;
  Eigen::Index structure_response_columns_ = 0;
  std::uint64_t structure_response_revision_ = 0;
  std::size_t revision_ = 0;
  ProjectedModel projected_model_;
};

/**
 * @brief Solves the coupled model after projected structure-response removal.
 *
 * Every orbital basis column carries a common-revision response
 * @f$z=-C^{-1}Bp@f$, its adjoint image @f$B^Tz@f$, and its true bordered
 * residual. The projected relaxed Hessian is therefore formed directly,
 * while convergence still requires both coupled KKT residuals.
 */
ResponseNeoResult solve_response_neo(
    const ResponseNeoProblem& problem,
    const NeoOptions& options);

}  // namespace xmvb::vb
