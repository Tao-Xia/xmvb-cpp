#include "vbscf/optimization/coupled/solver.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <Eigen/LU>

namespace xmvb::vb {
namespace {

double backward_error(
    const Eigen::Ref<const Eigen::VectorXd>& residual,
    std::initializer_list<double> term_norms) {
  double scale = 0.0;
  for (const double norm : term_norms) scale += norm;
  const double residual_norm = residual.stableNorm();
  if (scale > 0.0) return residual_norm / scale;
  return residual_norm == 0.0
      ? 0.0
      : std::numeric_limits<double>::infinity();
}

CoupledSubspaceStatus expansion_status(
    bool orbital_converged,
    bool response_converged) {
  if (orbital_converged && response_converged) {
    return CoupledSubspaceStatus::Converged;
  }
  if (!orbital_converged && !response_converged) {
    return CoupledSubspaceStatus::ExpandBothSpaces;
  }
  return orbital_converged
      ? CoupledSubspaceStatus::ExpandResponseSpace
      : CoupledSubspaceStatus::ExpandOrbitalSpace;
}

}  // namespace

CoupledSubspaceSolver::CoupledSubspaceSolver(
    const CoupledNewtonOperator& coupled_operator,
    Eigen::VectorXd orbital_gradient,
    Eigen::VectorXd response_residual)
    : coupled_operator_(&coupled_operator),
      orbital_gradient_(std::move(orbital_gradient)),
      response_residual_(std::move(response_residual)),
      cache_(coupled_operator) {
  if (orbital_gradient_.size() !=
          coupled_operator.n_orbital_coordinates() ||
      response_residual_.size() !=
          coupled_operator.n_response_coordinates() ||
      !orbital_gradient_.allFinite() ||
      !response_residual_.allFinite()) {
    throw std::invalid_argument(
        "coupled subspace gradients have invalid dimensions or values");
  }
}

const CoupledProjectionCache& CoupledSubspaceSolver::cache() const noexcept {
  return cache_;
}

int CoupledSubspaceSolver::append_orbital_block(
    const Eigen::Ref<const Eigen::MatrixXd>& candidates) {
  return cache_.append_orbital_block(candidates);
}

int CoupledSubspaceSolver::append_response_block(
    const Eigen::Ref<const Eigen::MatrixXd>& candidates) {
  return cache_.append_response_block(candidates);
}

CoupledSubspaceStep CoupledSubspaceSolver::solve(
    double trust_radius,
    const CoupledKktTolerances& tolerances,
    double incumbent_predicted_decrease) const {
  if (!(trust_radius > 0.0) || !std::isfinite(trust_radius) ||
      !(tolerances.orbital >= 0.0) ||
      !std::isfinite(tolerances.orbital) ||
      !(tolerances.response >= 0.0) ||
      !std::isfinite(tolerances.response) ||
      !(incumbent_predicted_decrease >= 0.0) ||
      !std::isfinite(incumbent_predicted_decrease)) {
    throw std::invalid_argument(
        "coupled subspace radius or KKT tolerances are invalid");
  }

  CoupledSubspaceStep result;
  if (cache_.orbital_subspace_size() == 0) {
    result.status = CoupledSubspaceStatus::EmptyOrbitalSpace;
    return result;
  }
  const Eigen::VectorXd projected_orbital_gradient =
      cache_.orbital_basis().transpose() * orbital_gradient_;
  const Eigen::VectorXd projected_response_residual =
      cache_.response_basis().transpose() * response_residual_;
  result.projected = solve_coupled_projected_model(
      cache_.projected_orbital_hessian(),
      cache_.projected_response_hessian(),
      cache_.projected_coupling(),
      cache_.projected_orbital_metric(),
      projected_orbital_gradient,
      projected_response_residual,
      trust_radius);
  switch (result.projected.status) {
    case CoupledProjectedModelStatus::ResponseProjectionSingular:
      {
        Eigen::FullPivLU<Eigen::MatrixXd> response_factor(
            cache_.projected_response_hessian());
        const Eigen::MatrixXd null_vectors = response_factor.kernel();
        if (null_vectors.cols() != 0) {
          result.response_expansion_candidate =
              cache_.response_hessian_images() * null_vectors.col(0);
          result.response_expansion_candidate.noalias() -=
              cache_.response_basis() *
              (cache_.response_basis().transpose() *
               result.response_expansion_candidate);
        }
        const double image_scale = cache_.response_hessian_images().norm();
        const double roundoff = std::numeric_limits<double>::epsilon() *
            std::max(1, coupled_operator_->n_response_coordinates()) *
            image_scale;
        result.status = result.response_expansion_candidate.stableNorm() >
                roundoff
            ? CoupledSubspaceStatus::ExpandResponseSpace
            : CoupledSubspaceStatus::ResponseOperatorSingular;
      }
      return result;
    case CoupledProjectedModelStatus::OrbitalTrustRegionFailure:
      result.status = CoupledSubspaceStatus::ProjectedTrustRegionFailure;
      return result;
    case CoupledProjectedModelStatus::NumericalFailure:
      result.status = CoupledSubspaceStatus::NumericalFailure;
      return result;
    case CoupledProjectedModelStatus::Converged:
      break;
  }

  const Eigen::VectorXd& y = result.projected.orbital_coordinates;
  const Eigen::VectorXd& u = result.projected.response_coordinates;
  result.orbital_step = cache_.orbital_basis() * y;
  result.response_step = cache_.response_basis() * u;
  const CoupledCachedBlocks unshifted = cache_.reconstruct_image(y, u);
  const Eigen::VectorXd metric_step =
      cache_.orbital_metric_images() * y;
  const double shift = result.projected.orbital_solution.shift;
  result.orbital_kkt_residual = orbital_gradient_ + unshifted.orbital +
      shift * metric_step;
  result.response_kkt_residual = response_residual_ + unshifted.response;

  const Eigen::VectorXd orbital_hessian_term =
      cache_.orbital_hessian_images() * y;
  const Eigen::VectorXd response_to_orbital_term =
      cache_.response_to_orbital_images() * u;
  const Eigen::VectorXd orbital_to_response_term =
      cache_.orbital_to_response_images() * y;
  const Eigen::VectorXd response_hessian_term =
      cache_.response_hessian_images() * u;
  result.orbital_backward_error = backward_error(
      result.orbital_kkt_residual,
      {orbital_gradient_.stableNorm(),
       orbital_hessian_term.stableNorm(),
       response_to_orbital_term.stableNorm(),
       std::abs(shift) * metric_step.stableNorm()});
  result.response_backward_error = backward_error(
      result.response_kkt_residual,
      {response_residual_.stableNorm(),
       orbital_to_response_term.stableNorm(),
       response_hessian_term.stableNorm()});

  result.model_change = orbital_gradient_.dot(result.orbital_step) +
      response_residual_.dot(result.response_step) +
      0.5 * (result.orbital_step.dot(unshifted.orbital) +
             result.response_step.dot(unshifted.response));
  result.predicted_decrease = -result.model_change;
  if (!result.orbital_step.allFinite() ||
      !result.response_step.allFinite() ||
      !result.orbital_kkt_residual.allFinite() ||
      !result.response_kkt_residual.allFinite() ||
      !std::isfinite(result.orbital_backward_error) ||
      !std::isfinite(result.response_backward_error) ||
      !std::isfinite(result.model_change) ||
      !std::isfinite(result.predicted_decrease)) {
    result.status = CoupledSubspaceStatus::NumericalFailure;
    return result;
  }

  const double feasibility_roundoff =
      32.0 * std::numeric_limits<double>::epsilon() *
      std::max(1, cache_.orbital_subspace_size());
  const double projected_roundoff =
      256.0 * std::numeric_limits<double>::epsilon() *
      std::max(1, cache_.orbital_subspace_size());
  if (result.projected.orbital_solution.metric_norm >
          trust_radius * (1.0 + feasibility_roundoff) ||
      result.projected.orbital_solution.stationarity_backward_error >
          projected_roundoff ||
      result.projected.orbital_solution.minimum_shifted_ritz_value <
          -projected_roundoff * std::max(
              std::abs(
                  result.projected.orbital_solution.minimum_ritz_value),
              std::abs(result.projected.orbital_solution.shift))) {
    result.status = CoupledSubspaceStatus::NumericalFailure;
    return result;
  }
  const double decrease_roundoff =
      64.0 * std::numeric_limits<double>::epsilon() *
      std::max({std::abs(result.predicted_decrease),
                incumbent_predicted_decrease});
  if (result.predicted_decrease < -decrease_roundoff ||
      result.predicted_decrease + decrease_roundoff <
          incumbent_predicted_decrease) {
    result.status = CoupledSubspaceStatus::InsufficientPredictedDecrease;
    return result;
  }

  result.status = expansion_status(
      result.orbital_backward_error <= tolerances.orbital,
      result.response_backward_error <= tolerances.response);
  return result;
}

CoupledSubspaceExpansion CoupledSubspaceSolver::expand(
    const CoupledSubspaceStep& step,
    const SymmetricOperatorAction& apply_inverse_orbital_preconditioner,
    const SymmetricOperatorAction& apply_inverse_response_preconditioner) {
  const bool expand_orbital =
      step.status == CoupledSubspaceStatus::ExpandOrbitalSpace ||
      step.status == CoupledSubspaceStatus::ExpandBothSpaces;
  const bool expand_response =
      step.status == CoupledSubspaceStatus::ExpandResponseSpace ||
      step.status == CoupledSubspaceStatus::ExpandBothSpaces;
  if (!expand_orbital && !expand_response) {
    throw std::invalid_argument(
        "coupled subspace expansion requires an unresolved KKT block");
  }

  CoupledSubspaceExpansion expansion;
  if (expand_response) {
    Eigen::VectorXd candidate;
    if (step.response_expansion_candidate.size() != 0) {
      candidate = step.response_expansion_candidate;
    } else {
      if (!apply_inverse_response_preconditioner ||
          step.response_kkt_residual.size() !=
              coupled_operator_->n_response_coordinates()) {
        throw std::invalid_argument(
            "response expansion requires its full residual and inverse action");
      }
      candidate = -apply_inverse_response_preconditioner(
          step.response_kkt_residual);
    }
    if (candidate.size() != coupled_operator_->n_response_coordinates() ||
        !candidate.allFinite()) {
      throw std::runtime_error(
          "response expansion inverse action returned invalid values");
    }
    expansion.response_rank = append_response_block(candidate);
  }
  if (expand_orbital) {
    if (!apply_inverse_orbital_preconditioner ||
        step.orbital_kkt_residual.size() !=
            coupled_operator_->n_orbital_coordinates()) {
      throw std::invalid_argument(
          "orbital expansion requires its full residual and inverse action");
    }
    Eigen::VectorXd candidate = -apply_inverse_orbital_preconditioner(
        step.orbital_kkt_residual);
    if (candidate.size() != coupled_operator_->n_orbital_coordinates() ||
        !candidate.allFinite()) {
      throw std::runtime_error(
          "orbital expansion inverse action returned invalid values");
    }
    expansion.orbital_rank = append_orbital_block(candidate);
  }
  return expansion;
}

}  // namespace xmvb::vb
