#include "vbscf/optimization/coupled/workspace.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace xmvb::vb {
namespace {

Eigen::MatrixXd apply_inverse_block(
    const SymmetricOperatorAction& apply_inverse,
    const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
  Eigen::MatrixXd images(vectors.rows(), vectors.cols());
  for (Eigen::Index column = 0; column < vectors.cols(); ++column) {
    const Eigen::VectorXd image = apply_inverse(vectors.col(column));
    if (image.size() != vectors.rows() || !image.allFinite()) {
      throw std::runtime_error(
          "coupled workspace inverse action returned invalid values");
    }
    images.col(column) = image;
  }
  return images;
}

Eigen::MatrixXd limited_columns(
    const Eigen::Ref<const Eigen::MatrixXd>& candidates,
    int available) {
  return candidates.leftCols(std::min<int>(available, candidates.cols()));
}

bool requests_orbital_expansion(CoupledSubspaceStatus status) {
  return status == CoupledSubspaceStatus::ExpandOrbitalSpace ||
      status == CoupledSubspaceStatus::ExpandBothSpaces;
}

bool requests_response_expansion(CoupledSubspaceStatus status) {
  return status == CoupledSubspaceStatus::ExpandResponseSpace ||
      status == CoupledSubspaceStatus::ExpandBothSpaces;
}

}  // namespace

AcceptedPointCoupledWorkspace::AcceptedPointCoupledWorkspace(
    AcceptedPointCoupledModel accepted_model,
    Eigen::VectorXd orbital_gradient,
    SymmetricOperatorAction apply_inverse_orbital_preconditioner)
    : accepted_model_(std::move(accepted_model)),
      orbital_gradient_(std::move(orbital_gradient)),
      apply_inverse_orbital_preconditioner_(
          std::move(apply_inverse_orbital_preconditioner)),
      apply_inverse_response_preconditioner_(
          accepted_model_.response_inverse_preconditioner),
      subspace_solver_(
          accepted_model_.newton_operator,
          orbital_gradient_,
          accepted_model_.structure_kkt_residual) {
  if (orbital_gradient_.size() !=
          accepted_model_.newton_operator.n_orbital_coordinates() ||
      !orbital_gradient_.allFinite() ||
      !apply_inverse_orbital_preconditioner_ ||
      !apply_inverse_response_preconditioner_) {
    throw std::invalid_argument(
        "accepted-point coupled workspace inputs are inconsistent");
  }
}

const AcceptedPointCoupledModel&
AcceptedPointCoupledWorkspace::accepted_model() const noexcept {
  return accepted_model_;
}

const CoupledSubspaceSolver&
AcceptedPointCoupledWorkspace::subspace_solver() const noexcept {
  return subspace_solver_;
}

int AcceptedPointCoupledWorkspace::orbital_limit(
    const CoupledWorkspaceLimits& limits) const {
  const int dimension =
      accepted_model_.newton_operator.n_orbital_coordinates();
  if (limits.maximum_orbital_dimension < 0 ||
      limits.maximum_orbital_dimension > dimension) {
    throw std::invalid_argument("invalid coupled orbital work limit");
  }
  return limits.maximum_orbital_dimension == 0
      ? dimension
      : limits.maximum_orbital_dimension;
}

int AcceptedPointCoupledWorkspace::response_limit(
    const CoupledWorkspaceLimits& limits) const {
  const int dimension =
      accepted_model_.newton_operator.n_response_coordinates();
  if (limits.maximum_response_dimension < 0 ||
      limits.maximum_response_dimension > dimension) {
    throw std::invalid_argument("invalid coupled response work limit");
  }
  return limits.maximum_response_dimension == 0
      ? dimension
      : limits.maximum_response_dimension;
}

bool AcceptedPointCoupledWorkspace::initialize(
    const CoupledWorkspaceLimits& limits) {
  if (initialized_) return true;
  const int maximum_orbitals = orbital_limit(limits);
  const int maximum_responses = response_limit(limits);

  const Eigen::VectorXd& structure_residual =
      accepted_model_.structure_kkt_residual;
  if (orbital_gradient_.isZero(0.0) && structure_residual.isZero(0.0)) {
    initialized_ = true;
    return true;
  }
  if (!structure_residual.isZero(0.0) && maximum_responses != 0) {
    const Eigen::VectorXd response_image =
        -apply_inverse_response_preconditioner_(structure_residual);
    if (response_image.size() != structure_residual.size() ||
        !response_image.allFinite()) {
      throw std::runtime_error(
          "response seed inverse action returned invalid values");
    }
    Eigen::MatrixXd response_seed(structure_residual.size(), 1);
    response_seed.col(0) = response_image;
    subspace_solver_.append_response_block(response_seed);
  }

  const auto& cache = subspace_solver_.cache();
  const int n_response_seeds = cache.response_subspace_size();
  Eigen::MatrixXd orbital_rhs(
      orbital_gradient_.size(), 1 + n_response_seeds);
  orbital_rhs.col(0) = orbital_gradient_;
  if (n_response_seeds != 0) {
    orbital_rhs.rightCols(n_response_seeds) =
        cache.response_to_orbital_images();
  }
  Eigen::MatrixXd orbital_seed = -apply_inverse_block(
      apply_inverse_orbital_preconditioner_, orbital_rhs);
  const int orbital_room = maximum_orbitals -
      cache.orbital_subspace_size();
  if (orbital_room > 0) {
    subspace_solver_.append_orbital_block(
        limited_columns(orbital_seed, orbital_room));
  }

  const int n_orbital_seeds = cache.orbital_subspace_size();
  const int response_room = maximum_responses -
      cache.response_subspace_size();
  if (n_orbital_seeds != 0 && response_room > 0) {
    const Eigen::MatrixXd response_seed = -apply_inverse_block(
        apply_inverse_response_preconditioner_,
        cache.orbital_to_response_images());
    subspace_solver_.append_response_block(
        limited_columns(response_seed, response_room));
  }
  initialized_ = true;
  return cache.orbital_subspace_size() != 0 ||
      (orbital_gradient_.isZero(0.0) && structure_residual.isZero(0.0));
}

CoupledWorkspaceResult AcceptedPointCoupledWorkspace::result(
    CoupledWorkspaceStatus status,
    CoupledSubspaceStep step,
    int expansions) const {
  CoupledWorkspaceResult workspace_result;
  workspace_result.status = status;
  workspace_result.step = std::move(step);
  workspace_result.action_counts =
      subspace_solver_.cache().action_counts();
  workspace_result.orbital_dimension =
      subspace_solver_.cache().orbital_subspace_size();
  workspace_result.response_dimension =
      subspace_solver_.cache().response_subspace_size();
  workspace_result.expansions = expansions;
  return workspace_result;
}

CoupledWorkspaceResult AcceptedPointCoupledWorkspace::solve(
    double trust_radius,
    const CoupledKktTolerances& tolerances,
    const CoupledWorkspaceLimits& limits,
    double incumbent_predicted_decrease) {
  const int maximum_orbitals = orbital_limit(limits);
  const int maximum_responses = response_limit(limits);
  if (!initialize(limits)) {
    CoupledSubspaceStep empty;
    empty.status = CoupledSubspaceStatus::EmptyOrbitalSpace;
    return result(
        CoupledWorkspaceStatus::WorkLimit, std::move(empty), 0);
  }

  if (orbital_gradient_.isZero(0.0) &&
      accepted_model_.structure_kkt_residual.isZero(0.0) &&
      subspace_solver_.cache().orbital_subspace_size() == 0) {
    CoupledSubspaceStep zero;
    zero.status = CoupledSubspaceStatus::Converged;
    zero.orbital_step = Eigen::VectorXd::Zero(
        accepted_model_.newton_operator.n_orbital_coordinates());
    zero.response_step = Eigen::VectorXd::Zero(
        accepted_model_.newton_operator.n_response_coordinates());
    zero.orbital_kkt_residual = orbital_gradient_;
    zero.response_kkt_residual =
        accepted_model_.structure_kkt_residual;
    zero.orbital_backward_error = 0.0;
    zero.response_backward_error = 0.0;
    zero.model_change = 0.0;
    zero.predicted_decrease = 0.0;
    return result(
        CoupledWorkspaceStatus::Converged, std::move(zero), 0);
  }

  int expansions = 0;
  while (true) {
    CoupledSubspaceStep step = subspace_solver_.solve(
        trust_radius, tolerances, incumbent_predicted_decrease);
    if (step.converged()) {
      return result(
          CoupledWorkspaceStatus::Converged,
          std::move(step),
          expansions);
    }
    bool expand_orbital = requests_orbital_expansion(step.status);
    bool expand_response = requests_response_expansion(step.status);
    if (step.status ==
        CoupledSubspaceStatus::InsufficientPredictedDecrease) {
      expand_orbital =
          step.orbital_backward_error > tolerances.orbital;
      expand_response =
          step.response_backward_error > tolerances.response;
      if (expand_orbital && expand_response) {
        step.status = CoupledSubspaceStatus::ExpandBothSpaces;
      } else if (expand_orbital) {
        step.status = CoupledSubspaceStatus::ExpandOrbitalSpace;
      } else if (expand_response) {
        step.status = CoupledSubspaceStatus::ExpandResponseSpace;
      }
    }
    if (!expand_orbital && !expand_response) {
      const CoupledWorkspaceStatus status =
          step.status == CoupledSubspaceStatus::NumericalFailure
          ? CoupledWorkspaceStatus::NumericalFailure
          : CoupledWorkspaceStatus::SubproblemFailure;
      return result(
          status, std::move(step), expansions);
    }
    if ((expand_orbital &&
         subspace_solver_.cache().orbital_subspace_size() >=
             maximum_orbitals) ||
        (expand_response &&
         subspace_solver_.cache().response_subspace_size() >=
             maximum_responses)) {
      return result(
          CoupledWorkspaceStatus::WorkLimit,
          std::move(step),
          expansions);
    }

    const CoupledSubspaceExpansion expansion = subspace_solver_.expand(
        step,
        apply_inverse_orbital_preconditioner_,
        apply_inverse_response_preconditioner_);
    ++expansions;
    if (!expansion.progressed()) {
      return result(
          CoupledWorkspaceStatus::NumericalFailure,
          std::move(step),
          expansions);
    }
  }
}

}  // namespace xmvb::vb
