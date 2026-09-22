#include "vbscf/optimization/neo/augmented_hessian.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/Eigenvalues>

namespace xmvb::vb {
AugmentedHessianWorkspace::AugmentedHessianWorkspace(
    const NeoProblem& problem)
    : problem_(&problem) {}

bool AugmentedHessianWorkspace::append(
    Eigen::VectorXd direction,
    double spectral_shift,
    bool precondition) {
  if (direction.size() != problem_->size() || !direction.allFinite()) {
    throw std::invalid_argument("invalid augmented-Hessian basis direction");
  }
  if (precondition && problem_->has_preconditioner()) {
    direction = problem_->apply_preconditioner(direction, spectral_shift);
  }
  const double initial_norm = direction.stableNorm();
  if (!(initial_norm > 0.0) || !std::isfinite(initial_norm)) return false;
  for (int pass = 0; pass < 2; ++pass) {
    if (vectors_.cols() > 0) {
      direction.noalias() -=
          vectors_ * (vectors_.transpose() * direction);
    }
  }
  const double norm = direction.stableNorm();
  const double threshold = 64.0 * std::numeric_limits<double>::epsilon() *
      initial_norm;
  if (!(norm > threshold) || !std::isfinite(norm)) return false;
  direction /= norm;
  NeoOperatorImages images = problem_->apply_hessian_metric(direction);
  const Eigen::Index column = vectors_.cols();
  if (column == 0) {
    vectors_.resize(direction.size(), 1);
    hessian_images_.resize(direction.size(), 1);
    metric_images_.resize(direction.size(), 1);
  } else {
    vectors_.conservativeResize(Eigen::NoChange, column + 1);
    hessian_images_.conservativeResize(Eigen::NoChange, column + 1);
    metric_images_.conservativeResize(Eigen::NoChange, column + 1);
  }
  vectors_.col(column) = std::move(direction);
  hessian_images_.col(column) = std::move(images.hessian);
  metric_images_.col(column) = std::move(images.metric);
  return true;
}

AugmentedHessianStep AugmentedHessianWorkspace::next(
    const Eigen::VectorXd& gradient,
    const AugmentedHessianOptions& options) {
  if (gradient.size() != problem_->size() || !gradient.allFinite() ||
      !(options.convergence_tolerance > 0.0) ||
      !(options.linear_dependence_tolerance > 0.0) ||
      !(options.start_tolerance > 0.0) ||
      options.start_cycle < 1 || options.maximum_subspace_dimension < 1) {
    throw std::invalid_argument("invalid augmented-Hessian solve input");
  }
  const int action_start = hessian_actions();
  if (has_pending_residual_) {
    Eigen::VectorXd residual = std::move(pending_residual_);
    has_pending_residual_ = false;
    if (!append(std::move(residual), pending_eigenvalue_, true)) {
      throw std::runtime_error(
          "augmented-Hessian residual is linearly dependent");
    }
  }
  if (vectors_.cols() == 0) {
    Eigen::VectorXd seed = problem_->initial_guess().size() == problem_->size()
        ? problem_->initial_guess()
        : gradient;
    // PySCF CIAH inserts x0 directly.  Only Davidson residuals are
    // preconditioned; preconditioning the seed changes the first projected
    // model and can select a different nonlinear basin.
    if (!append(std::move(seed), 0.0, false)) {
      throw std::runtime_error(
          "augmented-Hessian solver could not construct its first direction");
    }
  }

  while (true) {
    const Eigen::MatrixXd& q = vectors_;
    const Eigen::MatrixXd& hq = hessian_images_;
    const Eigen::MatrixXd& mq = metric_images_;
    const Eigen::Index n = q.cols();
    Eigen::MatrixXd augmented = Eigen::MatrixXd::Zero(n + 1, n + 1);
    Eigen::MatrixXd overlap = Eigen::MatrixXd::Zero(n + 1, n + 1);
    augmented.block(1, 1, n, n) =
        0.5 * (q.transpose() * hq + hq.transpose() * q);
    augmented.block(0, 1, 1, n) = gradient.transpose() * q;
    augmented.block(1, 0, n, 1) = q.transpose() * gradient;
    overlap(0, 0) = 1.0;
    overlap.block(1, 1, n, n) =
        0.5 * (q.transpose() * mq + mq.transpose() * q);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> overlap_spectrum(overlap);
    if (overlap_spectrum.info() != Eigen::Success) {
      throw std::runtime_error("augmented-Hessian overlap diagonalization failed");
    }
    const double overlap_minimum = overlap_spectrum.eigenvalues().minCoeff();
    if (!(overlap_minimum > 0.0)) {
      throw std::runtime_error("augmented-Hessian overlap is not positive definite");
    }
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(
        augmented, overlap);
    if (spectrum.info() != Eigen::Success) {
      throw std::runtime_error("augmented-Hessian diagonalization failed");
    }

    Eigen::Index selected = -1;
    for (Eigen::Index root = 0; root < spectrum.eigenvectors().cols(); ++root) {
      if (std::abs(spectrum.eigenvectors()(0, root)) > 0.1) {
        selected = root;
        break;
      }
    }
    if (selected < 0) {
      throw std::runtime_error(
          "augmented-Hessian subspace has no regular root");
    }
    const double eigenvalue = spectrum.eigenvalues()[selected];
    const Eigen::VectorXd eigenvector = spectrum.eigenvectors().col(selected);
    const Eigen::VectorXd coefficients =
        eigenvector.tail(n) / eigenvector[0];

    AugmentedHessianStep result;
    result.step = q * coefficients;
    result.hessian_step = hq * coefficients;
    result.metric_step = mq * coefficients;
    result.augmented_component = eigenvector[0];
    result.residual = eigenvector[0] *
        (gradient + result.hessian_step - eigenvalue * result.metric_step);
    result.eigenvalue = eigenvalue;
    result.residual_norm = result.residual.stableNorm();
    result.overlap_minimum = overlap_minimum;
    result.subspace_dimension = static_cast<int>(n);
    const bool eigenvalue_converged = has_previous_eigenvalue_ &&
        std::abs(eigenvalue - previous_eigenvalue_) <
            options.convergence_tolerance;
    result.converged = eigenvalue_converged &&
        result.residual_norm < std::sqrt(options.convergence_tolerance);
    previous_eigenvalue_ = eigenvalue;
    has_previous_eigenvalue_ = true;

    const bool ready = result.converged ||
        (result.residual_norm < options.start_tolerance &&
         result.subspace_dimension >= options.start_cycle) ||
        overlap_minimum < options.linear_dependence_tolerance ||
        result.subspace_dimension >= options.maximum_subspace_dimension;
    if (ready) {
      if (!result.converged &&
          result.subspace_dimension < options.maximum_subspace_dimension &&
          overlap_minimum >= options.linear_dependence_tolerance) {
        pending_residual_ = result.residual;
        pending_eigenvalue_ = eigenvalue;
        has_pending_residual_ = true;
      }
      result.new_hessian_actions = hessian_actions() - action_start;
      return result;
    }
    if (!append(result.residual, eigenvalue, true)) {
      result.new_hessian_actions = hessian_actions() - action_start;
      return result;
    }
  }
}

}  // namespace xmvb::vb
