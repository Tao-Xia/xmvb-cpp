#include "vbscf/optimization/neo/response_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include "vbscf/optimization/trust_region/spectral.hpp"

namespace xmvb::vb {
namespace {

bool orthonormalize(
    Eigen::VectorXd* direction,
    const Eigen::Ref<const Eigen::MatrixXd>& basis) {
  const double initial_norm = direction->stableNorm();
  if (!(initial_norm > 0.0) || !std::isfinite(initial_norm)) return false;
  for (int pass = 0; pass < 2; ++pass) {
    direction->noalias() -= basis * (basis.transpose() * *direction);
  }
  const double norm = direction->stableNorm();
  const double threshold = 64.0 * std::numeric_limits<double>::epsilon() *
      initial_norm;
  if (!(norm > threshold) || !std::isfinite(norm)) return false;
  *direction /= norm;
  return true;
}

void append_column(Eigen::MatrixXd* matrix, const Eigen::VectorXd& column) {
  if (matrix->cols() == 0) {
    matrix->resize(column.size(), 1);
  } else {
    matrix->conservativeResize(Eigen::NoChange, matrix->cols() + 1);
  }
  matrix->col(matrix->cols() - 1) = column;
}

std::uint64_t splitmix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

Eigen::VectorXd generic_probe(Eigen::Index size) {
  Eigen::VectorXd probe(size);
  for (Eigen::Index j = 0; j < size; ++j) {
    const std::uint64_t bits = splitmix64(static_cast<std::uint64_t>(j));
    const double unit = static_cast<double>(bits >> 11U) * 0x1.0p-53;
    probe[j] = 2.0 * unit - 1.0;
  }
  return probe;
}

Eigen::MatrixXd symmetric_part(
    const Eigen::MatrixXd& matrix,
    const char* message,
    double operator_relative_accuracy = 0.0) {
  const double factor = std::max(
      2048.0 * std::numeric_limits<double>::epsilon() *
          static_cast<double>(std::max<Eigen::Index>(1, matrix.rows())),
      operator_relative_accuracy);
  const double defect = (matrix - matrix.transpose()).stableNorm();
  const double scale = std::max(1.0, matrix.stableNorm());
  if (defect > factor * scale) {
    std::ostringstream detail;
    detail.precision(17);
    detail << message << ": relative defect=" << defect / scale;
    throw std::runtime_error(detail.str());
  }
  return 0.5 * (matrix + matrix.transpose());
}

struct StructureInverse {
  Eigen::MatrixXd value;
  Eigen::MatrixXd null_vectors;
};

StructureInverse symmetric_pseudoinverse(const Eigen::MatrixXd& matrix) {
  if (matrix.rows() == 0) {
    return {matrix, Eigen::MatrixXd(matrix.rows(), 0)};
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(matrix);
  if (spectrum.info() != Eigen::Success) {
    throw std::runtime_error("NEO structure eigensystem failed");
  }
  const double cutoff = 256.0 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, spectrum.eigenvalues().cwiseAbs().maxCoeff()) *
      static_cast<double>(matrix.rows());
  Eigen::VectorXd inverse = spectrum.eigenvalues();
  Eigen::Index nullity = 0;
  for (Eigen::Index j = 0; j < inverse.size(); ++j) {
    if (std::abs(inverse[j]) > cutoff) {
      inverse[j] = 1.0 / inverse[j];
    } else {
      inverse[j] = 0.0;
      ++nullity;
    }
  }
  StructureInverse result;
  result.value = spectrum.eigenvectors() * inverse.asDiagonal() *
      spectrum.eigenvectors().transpose();
  result.null_vectors.resize(matrix.rows(), nullity);
  Eigen::Index column = 0;
  for (Eigen::Index j = 0; j < inverse.size(); ++j) {
    if (inverse[j] == 0.0) {
      result.null_vectors.col(column++) = spectrum.eigenvectors().col(j);
    }
  }
  return result;
}

Eigen::VectorXd preconditioned_orbital(
    const ResponseNeoProblem& problem,
    const Eigen::VectorXd& residual,
    double shift = 0.0) {
  return problem.has_orbital_preconditioner()
      ? problem.apply_orbital_preconditioner(residual, shift)
      : residual;
}

}  // namespace

ResponseNeoProblem::ResponseNeoProblem(
    Eigen::VectorXd orbital_gradient,
    Eigen::Index structure_size,
    ResponseNeoBlockAction apply_orbital_coupling,
    ResponseNeoBlockAction apply_structure_coupling,
    NeoAction apply_orbital_metric,
    ResponseNeoPreconditioner apply_orbital_preconditioner,
    Eigen::VectorXd initial_orbital_guess,
    double operator_relative_accuracy,
    ResponseNeoMatrixAction apply_structure_coupling_block)
    : orbital_gradient_(std::move(orbital_gradient)),
      structure_size_(structure_size),
      apply_orbital_coupling_(std::move(apply_orbital_coupling)),
      apply_structure_coupling_(std::move(apply_structure_coupling)),
      apply_structure_coupling_block_(
          std::move(apply_structure_coupling_block)),
      apply_orbital_metric_(std::move(apply_orbital_metric)),
      apply_orbital_preconditioner_(std::move(apply_orbital_preconditioner)),
      initial_orbital_guess_(std::move(initial_orbital_guess)),
      operator_relative_accuracy_(operator_relative_accuracy) {
  if (orbital_gradient_.size() == 0 || !orbital_gradient_.allFinite() ||
      structure_size_ < 0) {
    throw std::invalid_argument("response NEO requires finite dimensions");
  }
  if (!apply_orbital_coupling_ || !apply_structure_coupling_ ||
      !apply_orbital_metric_) {
    throw std::invalid_argument("response NEO requires coupling and metric actions");
  }
  if (initial_orbital_guess_.size() != 0 &&
      (initial_orbital_guess_.size() != orbital_size() ||
       !initial_orbital_guess_.allFinite())) {
    throw std::invalid_argument(
        "response NEO initial orbital guess is invalid");
  }
  if (!std::isfinite(operator_relative_accuracy_) ||
      operator_relative_accuracy_ < 0.0) {
    throw std::invalid_argument(
        "response NEO operator accuracy must be finite and nonnegative");
  }
}

ResponseNeoDirection ResponseNeoProblem::apply_block_checked(
    const ResponseNeoBlockAction& action,
    const Eigen::VectorXd& vector,
    Eigen::Index expected_size) const {
  if (vector.size() != expected_size || !vector.allFinite()) {
    throw std::invalid_argument("invalid vector passed to NEO block action");
  }
  ResponseNeoDirection result = action(vector);
  if (result.orbital.size() != orbital_size() ||
      result.structure.size() != structure_size() ||
      !result.orbital.allFinite() || !result.structure.allFinite()) {
    throw std::runtime_error("coupled NEO action returned an invalid vector");
  }
  return result;
}

ResponseNeoDirection ResponseNeoProblem::apply_orbital_coupling(
    const Eigen::VectorXd& direction) const {
  return apply_block_checked(
      apply_orbital_coupling_, direction, orbital_size());
}

ResponseNeoDirection ResponseNeoProblem::apply_structure_coupling(
    const Eigen::VectorXd& direction) const {
  return apply_block_checked(
      apply_structure_coupling_, direction, structure_size());
}

ResponseNeoDirectionBlock ResponseNeoProblem::apply_structure_coupling_block(
    const Eigen::MatrixXd& directions) const {
  if (directions.rows() != structure_size() || directions.cols() == 0 ||
      !directions.allFinite()) {
    throw std::invalid_argument(
        "invalid matrix passed to NEO structure block action");
  }
  ResponseNeoDirectionBlock result;
  if (apply_structure_coupling_block_) {
    result = apply_structure_coupling_block_(directions);
  } else {
    result.orbital.resize(orbital_size(), directions.cols());
    result.structure.resize(structure_size(), directions.cols());
    for (Eigen::Index column = 0; column < directions.cols(); ++column) {
      const ResponseNeoDirection image =
          apply_structure_coupling(directions.col(column));
      result.orbital.col(column) = image.orbital;
      result.structure.col(column) = image.structure;
    }
  }
  if (result.orbital.rows() != orbital_size() ||
      result.structure.rows() != structure_size() ||
      result.orbital.cols() != directions.cols() ||
      result.structure.cols() != directions.cols() ||
      !result.orbital.allFinite() || !result.structure.allFinite()) {
    throw std::runtime_error(
        "coupled NEO block action returned an invalid matrix");
  }
  return result;
}

Eigen::VectorXd ResponseNeoProblem::apply_orbital_checked(
    const NeoAction& action,
    const Eigen::VectorXd& vector,
    const char* message) const {
  if (vector.size() != orbital_size() || !vector.allFinite()) {
    throw std::invalid_argument("invalid orbital vector passed to response NEO");
  }
  Eigen::VectorXd result = action(vector);
  if (result.size() != orbital_size() || !result.allFinite()) {
    throw std::runtime_error(message);
  }
  return result;
}

Eigen::VectorXd ResponseNeoProblem::apply_orbital_metric(
    const Eigen::VectorXd& direction) const {
  return apply_orbital_checked(
      apply_orbital_metric_, direction,
      "response NEO orbital metric returned an invalid vector");
}

Eigen::VectorXd ResponseNeoProblem::apply_orbital_preconditioner(
    const Eigen::VectorXd& covector,
    double shift) const {
  if (!apply_orbital_preconditioner_) {
    throw std::logic_error("response NEO has no orbital preconditioner");
  }
  if (!(shift >= 0.0) || !std::isfinite(shift)) {
    throw std::invalid_argument(
        "response NEO preconditioner shift must be finite and nonnegative");
  }
  if (covector.size() != orbital_size() || !covector.allFinite()) {
    throw std::invalid_argument(
        "invalid vector passed to response NEO preconditioner");
  }
  Eigen::VectorXd result = apply_orbital_preconditioner_(covector, shift);
  if (result.size() != orbital_size() || !result.allFinite()) {
    throw std::runtime_error(
        "response NEO orbital preconditioner returned an invalid vector");
  }
  return result;
}

bool ResponseNeoWorkspace::append_orbital(
    Eigen::VectorXd direction,
    int* actions,
    int* orbital_actions) {
  if (!orthonormalize(&direction, orbital_basis_)) return false;
  const Eigen::VectorXd metric = problem_.apply_orbital_metric(direction);
  const ResponseNeoDirection image =
      problem_.apply_orbital_coupling(direction);
  append_column(&orbital_basis_, direction);
  append_column(&orbital_metric_images_, metric);
  append_column(&orbital_orbital_images_, image.orbital);
  append_column(&orbital_structure_images_, image.structure);
  const Eigen::Index n_orbital = orbital_basis_.cols();
  const Eigen::Index n_structure = structure_basis_.cols();
  projected_orbital_.conservativeResize(n_orbital, n_orbital);
  projected_orbital_.col(n_orbital - 1).noalias() =
      orbital_basis_.transpose() * image.orbital;
  projected_orbital_.row(n_orbital - 1).noalias() =
      direction.transpose() * orbital_orbital_images_;
  projected_metric_.conservativeResize(n_orbital, n_orbital);
  projected_metric_.col(n_orbital - 1).noalias() =
      orbital_basis_.transpose() * metric;
  projected_metric_.row(n_orbital - 1).noalias() =
      direction.transpose() * orbital_metric_images_;
  projected_coupling_.conservativeResize(n_structure, n_orbital);
  if (n_structure != 0) {
    projected_coupling_.col(n_orbital - 1).noalias() =
        structure_basis_.transpose() * image.structure;
  }
  projected_coupling_adjoint_.conservativeResize(n_orbital, n_structure);
  if (n_structure != 0) {
    projected_coupling_adjoint_.row(n_orbital - 1).noalias() =
        direction.transpose() * structure_orbital_images_;
  }
  ++*actions;
  ++*orbital_actions;
  return true;
}

bool ResponseNeoWorkspace::append_structure(
    Eigen::VectorXd direction,
    int* actions,
    int* structure_actions) {
  Eigen::MatrixXd directions(direction.size(), 1);
  directions.col(0) = std::move(direction);
  return append_structure_block(
      std::move(directions), actions, structure_actions);
}

void ResponseNeoWorkspace::store_structure(
    const Eigen::VectorXd& direction,
    const ResponseNeoDirection& image) {
  append_column(&structure_basis_, direction);
  append_column(&structure_orbital_images_, image.orbital);
  append_column(&structure_structure_images_, image.structure);
  const Eigen::Index n_orbital = orbital_basis_.cols();
  const Eigen::Index n_structure = structure_basis_.cols();
  projected_coupling_.conservativeResize(n_structure, n_orbital);
  if (n_orbital != 0) {
    projected_coupling_.row(n_structure - 1).noalias() =
        direction.transpose() * orbital_structure_images_;
  }
  projected_coupling_adjoint_.conservativeResize(n_orbital, n_structure);
  if (n_orbital != 0) {
    projected_coupling_adjoint_.col(n_structure - 1).noalias() =
        orbital_basis_.transpose() * image.orbital;
  }
  projected_structure_.conservativeResize(n_structure, n_structure);
  projected_structure_.col(n_structure - 1).noalias() =
      structure_basis_.transpose() * image.structure;
  projected_structure_.row(n_structure - 1).noalias() =
      direction.transpose() * structure_structure_images_;
}

bool ResponseNeoWorkspace::append_structure_block(
    Eigen::MatrixXd directions,
    int* actions,
    int* structure_actions) {
  Eigen::MatrixXd accepted(problem_.structure_size(), 0);
  for (Eigen::Index column = 0; column < directions.cols(); ++column) {
    Eigen::VectorXd direction = directions.col(column);
    if (!orthonormalize(&direction, structure_basis_) ||
        !orthonormalize(&direction, accepted)) {
      continue;
    }
    append_column(&accepted, direction);
  }
  if (accepted.cols() == 0) return false;
  const ResponseNeoDirectionBlock images =
      problem_.apply_structure_coupling_block(accepted);
  for (Eigen::Index column = 0; column < accepted.cols(); ++column) {
    store_structure(
        accepted.col(column),
        ResponseNeoDirection{
            images.orbital.col(column), images.structure.col(column)});
  }
  *actions += static_cast<int>(accepted.cols());
  *structure_actions += static_cast<int>(accepted.cols());
  return true;
}

ResponseNeoResult ResponseNeoWorkspace::solve(const NeoOptions& options) {
  if (!(options.trust_radius > 0.0) ||
      !std::isfinite(options.trust_radius) ||
      !(options.relative_residual_tolerance > 0.0) ||
      !std::isfinite(options.relative_residual_tolerance) ||
      options.absolute_residual_tolerance < 0.0 ||
      !std::isfinite(options.absolute_residual_tolerance) ||
      options.maximum_subspace_dimension < 0) {
    throw std::invalid_argument("invalid response NEO solver options");
  }

  const int full_dimension = static_cast<int>(
      problem_.orbital_size() + problem_.structure_size());
  const int maximum_dimension = options.maximum_subspace_dimension == 0
      ? full_dimension
      : std::min(full_dimension, options.maximum_subspace_dimension);
  const double gradient_norm = problem_.orbital_gradient().stableNorm();
  const double numerical_curvature_tolerance =
      std::sqrt(std::numeric_limits<double>::epsilon()) *
      static_cast<double>(std::max(1, full_dimension));
  const double curvature_relative_tolerance = std::min(
      options.relative_residual_tolerance,
      numerical_curvature_tolerance);
  ResponseNeoResult result;
  result.step.orbital = Eigen::VectorXd::Zero(problem_.orbital_size());
  result.step.structure = Eigen::VectorXd::Zero(problem_.structure_size());
  result.hessian_step = result.step;
  result.orbital_metric_step = result.step.orbital;
  result.kkt_residual = result.step;
  result.minimum_curvature_orbital = result.step.orbital;

  if (orbital_basis_.cols() == 0) {
    Eigen::VectorXd first =
        problem_.initial_orbital_guess().size() == problem_.orbital_size()
            ? problem_.initial_orbital_guess()
            : preconditioned_orbital(problem_, -problem_.orbital_gradient());
    if (!append_orbital(
            std::move(first), &result.coupled_actions,
            &result.orbital_actions)) {
      append_orbital(
          generic_probe(problem_.orbital_size()), &result.coupled_actions,
          &result.orbital_actions);
    }
  }
  if (orbital_basis_.cols() == 0) {
    throw std::runtime_error("response NEO could not construct an orbital basis");
  }
  if (options.require_curvature_certificate &&
      orbital_basis_.cols() == 1 && maximum_dimension > 1 &&
      problem_.orbital_size() > 1) {
    append_orbital(
        generic_probe(problem_.orbital_size()),
        &result.coupled_actions,
        &result.orbital_actions);
  }
  while (true) {
    ++result.iterations;
    const Eigen::MatrixXd& qo = orbital_basis_;
    const Eigen::MatrixXd& mo = orbital_metric_images_;
    const Eigen::MatrixXd& ao = orbital_orbital_images_;
    const Eigen::MatrixXd& bo = orbital_structure_images_;
    const Eigen::MatrixXd& qs = structure_basis_;
    const Eigen::MatrixXd& bt = structure_orbital_images_;
    const Eigen::MatrixXd& cs = structure_structure_images_;

    const Eigen::MatrixXd a = symmetric_part(
        projected_orbital_, "response NEO orbital block is not symmetric",
        problem_.operator_relative_accuracy());
    const Eigen::MatrixXd metric = symmetric_part(
        projected_metric_, "response NEO orbital metric is not symmetric");
    Eigen::LLT<Eigen::MatrixXd> metric_factor(metric);
    if (metric_factor.info() != Eigen::Success) {
      throw std::runtime_error("response NEO orbital metric is not positive definite");
    }
    const Eigen::MatrixXd whitening = metric_factor.matrixU().solve(
        Eigen::MatrixXd::Identity(metric.rows(), metric.cols()));

    Eigen::MatrixXd b(qs.cols(), qo.cols());
    Eigen::MatrixXd c(qs.cols(), qs.cols());
    Eigen::MatrixXd c_inverse(qs.cols(), qs.cols());
    if (qs.cols() == 0) {
      b.setZero();
      c.setZero();
      c_inverse.setZero();
    } else {
      b = projected_coupling_;
      const Eigen::MatrixXd& projected_adjoint =
          projected_coupling_adjoint_;
      const double adjoint_scale = std::max({
          1.0, b.stableNorm(), projected_adjoint.stableNorm()});
      const double adjoint_tolerance = std::max(
          256.0 * std::numeric_limits<double>::epsilon() *
              static_cast<double>(std::max(b.rows(), b.cols())),
          problem_.operator_relative_accuracy());
      if ((b.transpose() - projected_adjoint).stableNorm() >
          adjoint_tolerance * adjoint_scale) {
        throw std::runtime_error("response NEO coupling blocks are not adjoints");
      }
      b = 0.5 * (b + projected_adjoint.transpose());
      c = symmetric_part(
          projected_structure_,
          "response NEO structure block is not symmetric",
          problem_.operator_relative_accuracy());
      const StructureInverse inverse = symmetric_pseudoinverse(c);
      c_inverse = inverse.value;
      if (inverse.null_vectors.cols() != 0) {
        const Eigen::MatrixXd range_defect =
            inverse.null_vectors.transpose() * b;
        const double range_error = range_defect.stableNorm();
        const double range_tolerance = std::max(
            256.0 * std::numeric_limits<double>::epsilon() *
                static_cast<double>(std::max(c.rows(), b.cols())),
            problem_.operator_relative_accuracy()) *
            std::max(1.0, b.stableNorm());
        if (range_error > range_tolerance) {
          Eigen::Index unresolved = 0;
          range_defect.rowwise().squaredNorm().maxCoeff(&unresolved);
          if (structure_basis_.cols() == problem_.structure_size()) {
            throw std::runtime_error(
                "response NEO coupling is outside the range of C");
          }
          const int basis_size = static_cast<int>(
              orbital_basis_.cols() + structure_basis_.cols());
          if (basis_size >= maximum_dimension) {
            result.stop_reason = NeoStopReason::SubspaceLimit;
            return result;
          }
          if (append_structure(
                  cs * inverse.null_vectors.col(unresolved),
                  &result.coupled_actions,
                  &result.structure_actions)) {
            continue;
          }
          throw std::runtime_error(
              "response NEO coupling is outside the range of C");
        }
      }
    }

    const Eigen::MatrixXd response_coefficients = -c_inverse * b;
    const Eigen::MatrixXd relaxed = a + b.transpose() * response_coefficients;
    const Eigen::MatrixXd whitened =
        whitening.transpose() * relaxed * whitening;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(whitened);
    if (spectrum.info() != Eigen::Success) {
      result.stop_reason = NeoStopReason::NumericalFailure;
      return result;
    }
    const Eigen::VectorXd gradient =
        qo.transpose() * problem_.orbital_gradient();
    const Eigen::VectorXd white_gradient = whitening.transpose() * gradient;
    const SpectralTrustRegionSolution projected = solve_spectral_trust_region(
        spectrum.eigenvalues(),
        spectrum.eigenvectors().transpose() * white_gradient,
        options.trust_radius);
    const Eigen::VectorXd p_coefficients = whitening *
        spectrum.eigenvectors() * projected.step;
    const Eigen::VectorXd q_coefficients =
        response_coefficients * p_coefficients;

    result.step.orbital.noalias() = qo * p_coefficients;
    result.step.structure.noalias() = qs * q_coefficients;
    const Eigen::VectorXd orbital_a = ao * p_coefficients;
    const Eigen::VectorXd orbital_bt = bt * q_coefficients;
    const Eigen::VectorXd structure_b = bo * p_coefficients;
    const Eigen::VectorXd structure_c = cs * q_coefficients;
    result.hessian_step.orbital = orbital_a + orbital_bt;
    result.hessian_step.structure = structure_b + structure_c;
    result.orbital_metric_step.noalias() = mo * p_coefficients;
    result.shift = projected.shift;
    result.boundary = projected.boundary;
    result.hard_case = projected.hard_case;
    result.step_norm = std::sqrt(std::max(
        0.0, result.step.orbital.dot(result.orbital_metric_step)));
    result.kkt_residual.orbital = problem_.orbital_gradient() +
        result.hessian_step.orbital +
        result.shift * result.orbital_metric_step;
    result.kkt_residual.structure = result.hessian_step.structure;
    // Inexact Newton convergence is controlled relative to the current
    // gradient, not to the much larger cancelling Hessian terms.  Scaling by
    // those terms would permit an O(||g||) residual and destroy the local
    // Newton rate in ill-conditioned coordinates.
    const double kkt_residual_target =
        options.absolute_residual_tolerance +
        options.relative_residual_tolerance * gradient_norm;
    const double orbital_residual_norm =
        result.kkt_residual.orbital.stableNorm();
    const double structure_residual_norm =
        result.kkt_residual.structure.stableNorm();
    result.residual_norm = std::hypot(
        orbital_residual_norm, structure_residual_norm);
    result.residual_target = kkt_residual_target;
    result.predicted_reduction =
        -problem_.orbital_gradient().dot(result.step.orbital) -
        0.5 * (result.step.orbital.dot(result.hessian_step.orbital) +
               result.step.structure.dot(result.hessian_step.structure));

    const Eigen::VectorXd minimum_coefficients = whitening *
        spectrum.eigenvectors().col(0);
    result.minimum_curvature_orbital.noalias() =
        qo * minimum_coefficients;
    const Eigen::VectorXd minimum_structure =
        response_coefficients * minimum_coefficients;
    const Eigen::VectorXd curvature_a = ao * minimum_coefficients;
    const Eigen::VectorXd curvature_bt = bt * minimum_structure;
    const Eigen::VectorXd curvature_metric =
        spectrum.eigenvalues()[0] * mo * minimum_coefficients;
    const Eigen::VectorXd curvature_b = bo * minimum_coefficients;
    const Eigen::VectorXd curvature_c = cs * minimum_structure;
    const Eigen::VectorXd curvature_orbital =
        curvature_a + curvature_bt - curvature_metric;
    const Eigen::VectorXd curvature_structure = curvature_b + curvature_c;
    result.curvature_residual_norm = std::hypot(
        curvature_orbital.stableNorm(), curvature_structure.stableNorm());
    const double curvature_orbital_target =
        options.absolute_residual_tolerance +
        curvature_relative_tolerance * std::max({
            curvature_a.stableNorm(),
            curvature_bt.stableNorm(),
            curvature_metric.stableNorm()});
    const double curvature_structure_target =
        options.absolute_residual_tolerance +
        curvature_relative_tolerance * std::max(
            curvature_b.stableNorm(), curvature_c.stableNorm());
    result.curvature_residual_target = std::hypot(
        curvature_orbital_target, curvature_structure_target);

    const double positivity_tolerance =
        curvature_relative_tolerance * std::max({
            1.0, result.shift, std::abs(spectrum.eigenvalues()[0])});
    const bool stationary = result.residual_norm <= kkt_residual_target;
    const bool curvature_converged =
        curvature_orbital.stableNorm() <= curvature_orbital_target &&
        curvature_structure.stableNorm() <= curvature_structure_target;
    const bool shifted_positive =
        spectrum.eigenvalues()[0] + result.shift >= -positivity_tolerance;
    const bool orbital_complete =
        orbital_basis_.cols() == problem_.orbital_size();
    result.global_curvature_certified =
        orbital_complete && curvature_converged;
    // A nonlinear keyframe certifies a regular boundary candidate by its
    // actual/predicted decrease.  Resolving an unrelated lowest Ritz vector
    // is therefore unnecessary unless explicitly requested.  A hard case is
    // different: its step itself depends on the minimum-curvature direction.
    const bool need_curvature_certificate =
        options.require_curvature_certificate || result.hard_case;
    const bool required_curvature_converged =
        !need_curvature_certificate || curvature_converged;
    if (stationary && required_curvature_converged && shifted_positive) {
      result.stop_reason = NeoStopReason::Converged;
      return result;
    }

    const int basis_size = static_cast<int>(
        orbital_basis_.cols() + structure_basis_.cols());
    if (basis_size >= maximum_dimension) {
      result.stop_reason = NeoStopReason::SubspaceLimit;
      return result;
    }

    // Refine only the structure response required by the current candidate.
    // Closing C Z = -B Q for every retained orbital basis column performs
    // response solves that do not enter p = Q y.  The complete coupled KKT
    // residual below certifies the direction-local response without that
    // basis-wide work.
    bool expanded = false;
    if (!stationary && structure_residual_norm >= orbital_residual_norm) {
      const bool add_curvature = need_curvature_certificate &&
          curvature_structure.stableNorm() > curvature_structure_target;
      Eigen::MatrixXd directions(
          problem_.structure_size(), add_curvature ? 2 : 1);
      directions.col(0) = -result.kkt_residual.structure;
      if (add_curvature) directions.col(1) = -curvature_structure;
      expanded = append_structure_block(
          std::move(directions), &result.coupled_actions,
          &result.structure_actions);
    }
    if (!expanded && need_curvature_certificate &&
        curvature_structure.stableNorm() >
        curvature_structure_target) {
      expanded = append_structure(
          -curvature_structure, &result.coupled_actions,
          &result.structure_actions);
    }
    if (!expanded && !stationary) {
      expanded = append_orbital(
          preconditioned_orbital(
              problem_, -result.kkt_residual.orbital, result.shift),
          &result.coupled_actions,
          &result.orbital_actions);
    }
    if (!expanded && !stationary && structure_residual_norm > 0.0) {
      expanded = append_structure(
          -result.kkt_residual.structure, &result.coupled_actions,
          &result.structure_actions);
    }
    if (!expanded && need_curvature_certificate &&
        curvature_orbital.stableNorm() >
        curvature_orbital_target) {
      expanded = append_orbital(
          preconditioned_orbital(problem_, -curvature_orbital),
          &result.coupled_actions,
          &result.orbital_actions);
    }
    while (!expanded && orbital_canonical_ < problem_.orbital_size()) {
      Eigen::VectorXd direction =
          Eigen::VectorXd::Zero(problem_.orbital_size());
      direction[orbital_canonical_++] = 1.0;
      expanded = append_orbital(
          std::move(direction), &result.coupled_actions,
          &result.orbital_actions);
    }
    while (!expanded && structure_canonical_ < problem_.structure_size()) {
      Eigen::VectorXd direction =
          Eigen::VectorXd::Zero(problem_.structure_size());
      direction[structure_canonical_++] = 1.0;
      expanded = append_structure(
          std::move(direction), &result.coupled_actions,
          &result.structure_actions);
    }
    if (!expanded) {
      result.stop_reason = NeoStopReason::NumericalFailure;
      return result;
    }
  }
}

ResponseNeoResult solve_response_neo(
    const ResponseNeoProblem& problem,
    const NeoOptions& options) {
  ResponseNeoWorkspace workspace(problem);
  return workspace.solve(options);
}

}  // namespace xmvb::vb
