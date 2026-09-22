#include "vbscf/optimization/neo/response_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

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
    detail << message << ": relative defect=" << defect / scale
           << ", dimension=" << matrix.rows();
    throw std::runtime_error(detail.str());
  }
  return 0.5 * (matrix + matrix.transpose());
}

Eigen::VectorXd preconditioned_orbital(
    const ResponseNeoProblem& problem,
    const Eigen::VectorXd& residual,
    double shift = 0.0) {
  return problem.has_orbital_preconditioner()
      ? problem.apply_orbital_preconditioner(residual, shift)
      : residual;
}

double coarse_response_tolerance(double final_relative_tolerance) {
  // The square-root forcing is the conventional inexact-Newton first tier:
  // it is accurate enough to propose a useful projected step without paying
  // final-stationarity cost for every retained Davidson column.
  return std::min(
      0.5,
      std::max(
          final_relative_tolerance,
          std::min(1.0e-2, std::sqrt(final_relative_tolerance))));
}

}  // namespace

ResponseNeoProblem::ResponseNeoProblem(
    Eigen::VectorXd orbital_gradient,
    Eigen::Index structure_size,
    ResponseNeoBlockAction apply_orbital_coupling,
    ResponseNeoStructureSolver solve_structure_response,
    NeoAction apply_orbital_metric,
    ResponseNeoPreconditioner apply_orbital_preconditioner,
    Eigen::VectorXd initial_orbital_guess,
    double operator_relative_accuracy)
    : orbital_gradient_(std::move(orbital_gradient)),
      structure_size_(structure_size),
      apply_orbital_coupling_(std::move(apply_orbital_coupling)),
      solve_structure_response_(std::move(solve_structure_response)),
      apply_orbital_metric_(std::move(apply_orbital_metric)),
      apply_orbital_preconditioner_(std::move(apply_orbital_preconditioner)),
      initial_orbital_guess_(std::move(initial_orbital_guess)),
      operator_relative_accuracy_(operator_relative_accuracy) {
  if (orbital_gradient_.size() == 0 || !orbital_gradient_.allFinite() ||
      structure_size_ < 0) {
    throw std::invalid_argument("response NEO requires finite dimensions");
  }
  if (!apply_orbital_coupling_ ||
      (structure_size_ > 0 && !solve_structure_response_) ||
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

ResponseNeoStructureResponse ResponseNeoProblem::solve_structure_response(
    const Eigen::Ref<const Eigen::MatrixXd>& forcing,
    double relative_tolerance) const {
  if (forcing.rows() != structure_size_ || forcing.cols() <= 0 ||
      !forcing.allFinite() || !(relative_tolerance > 0.0) ||
      !std::isfinite(relative_tolerance)) {
    throw std::invalid_argument("invalid NEO structure-response block");
  }
  if (!solve_structure_response_) {
    throw std::logic_error("response NEO has no structure-response solver");
  }
  ResponseNeoStructureResponse result =
      solve_structure_response_(forcing, relative_tolerance);
  if (result.coordinates.rows() != structure_size_ ||
      result.coordinates.cols() != forcing.cols() ||
      result.orbital_images.rows() != orbital_size() ||
      result.orbital_images.cols() != forcing.cols() ||
      result.equation_residuals.rows() != structure_size_ ||
      result.equation_residuals.cols() != forcing.cols() ||
      !result.coordinates.allFinite() || !result.orbital_images.allFinite() ||
      !result.equation_residuals.allFinite() ||
      result.block_actions < 0 ||
      !std::isfinite(result.max_relative_residual) ||
      result.max_relative_residual < 0.0) {
    throw std::runtime_error(
        "NEO structure-response solver returned an invalid block");
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

ResponseNeoWorkspace::ResponseNeoWorkspace(
    const ResponseNeoProblem& problem)
    : problem_(problem),
      orbital_basis_(problem.orbital_size(), 0),
      orbital_metric_images_(problem.orbital_size(), 0),
      orbital_hessian_images_(problem.orbital_size(), 0),
      orbital_structure_images_(problem.structure_size(), 0),
      structure_response_coordinates_(problem.structure_size(), 0),
      structure_response_orbital_images_(problem.orbital_size(), 0),
      structure_response_residuals_(problem.structure_size(), 0),
      response_model_correction_(0, 0),
      response_sample_coefficients_(0, 0),
      response_sample_orbital_defects_(problem.orbital_size(), 0),
      projected_orbital_hessian_(0, 0),
      projected_orbital_metric_(0, 0) {}

void ResponseNeoWorkspace::reserve_orbital_column() {
  if (orbital_basis_size_ < orbital_basis_.cols()) return;
  const Eigen::Index capacity = std::max<Eigen::Index>(
      2, 2 * orbital_basis_.cols());
  orbital_basis_.conservativeResize(Eigen::NoChange, capacity);
  orbital_metric_images_.conservativeResize(Eigen::NoChange, capacity);
  orbital_hessian_images_.conservativeResize(Eigen::NoChange, capacity);
  orbital_structure_images_.conservativeResize(Eigen::NoChange, capacity);
}

bool ResponseNeoWorkspace::append_orbital(
    Eigen::VectorXd direction,
    int* actions,
    int* orbital_actions) {
  if (!orthonormalize(
          &direction, orbital_basis_.leftCols(orbital_basis_size_))) {
    return false;
  }
  const Eigen::VectorXd metric_image =
      problem_.apply_orbital_metric(direction);
  const ResponseNeoDirection hessian_image =
      problem_.apply_orbital_coupling(direction);
  const Eigen::Index old_size = orbital_basis_size_;
  const Eigen::Index new_size = old_size + 1;

  projected_orbital_hessian_.conservativeResize(new_size, new_size);
  projected_orbital_metric_.conservativeResize(new_size, new_size);
  response_model_correction_.conservativeResize(new_size, new_size);
  response_model_correction_.row(old_size).setZero();
  response_model_correction_.col(old_size).setZero();
  response_sample_coefficients_.conservativeResize(
      new_size, Eigen::NoChange);
  response_sample_coefficients_.row(old_size).setZero();
  if (old_size > 0) {
    projected_orbital_hessian_.block(0, old_size, old_size, 1).noalias() =
        orbital_basis_.leftCols(old_size).transpose() * hessian_image.orbital;
    projected_orbital_hessian_.block(old_size, 0, 1, old_size).noalias() =
        direction.transpose() * orbital_hessian_images_.leftCols(old_size);
    projected_orbital_metric_.block(0, old_size, old_size, 1).noalias() =
        orbital_basis_.leftCols(old_size).transpose() * metric_image;
    projected_orbital_metric_.block(old_size, 0, 1, old_size).noalias() =
        direction.transpose() * orbital_metric_images_.leftCols(old_size);
  }
  projected_orbital_hessian_(old_size, old_size) =
      direction.dot(hessian_image.orbital);
  projected_orbital_metric_(old_size, old_size) =
      direction.dot(metric_image);

  reserve_orbital_column();
  orbital_basis_.col(old_size) = direction;
  orbital_metric_images_.col(old_size) = metric_image;
  orbital_hessian_images_.col(old_size) = hessian_image.orbital;
  orbital_structure_images_.col(old_size) = hessian_image.structure;
  orbital_basis_size_ = new_size;
  ++revision_;
  ++*actions;
  ++*orbital_actions;
  return true;
}

void ResponseNeoWorkspace::refresh_structure_response(
    double relative_tolerance,
    int* structure_actions) {
  if (problem_.structure_size() == 0) {
    structure_response_coordinates_.resize(0, orbital_basis_size_);
    structure_response_orbital_images_.resize(
        problem_.orbital_size(), orbital_basis_size_);
    structure_response_orbital_images_.setZero();
    structure_response_residuals_.resize(0, orbital_basis_size_);
    structure_response_columns_ = orbital_basis_size_;
    return;
  }
  const auto forcing =
      orbital_structure_images_.leftCols(orbital_basis_size_);
  const Eigen::Index old_columns = structure_response_columns_;
  ResponseNeoStructureResponse solved = problem_.solve_structure_response(
      forcing.rightCols(orbital_basis_size_ - old_columns),
      relative_tolerance);
  *structure_actions += solved.block_actions;
  if (old_columns > 0 && solved.revision != structure_response_revision_) {
    solved = problem_.solve_structure_response(forcing, relative_tolerance);
    *structure_actions += solved.block_actions;
    invalidate_response_model(solved.revision);
    structure_response_coordinates_ = std::move(solved.coordinates);
    structure_response_orbital_images_ = std::move(solved.orbital_images);
    structure_response_residuals_ = std::move(solved.equation_residuals);
  } else {
    structure_response_coordinates_.conservativeResize(
        Eigen::NoChange, orbital_basis_size_);
    structure_response_orbital_images_.conservativeResize(
        Eigen::NoChange, orbital_basis_size_);
    structure_response_residuals_.conservativeResize(
        Eigen::NoChange, orbital_basis_size_);
    structure_response_coordinates_.rightCols(solved.coordinates.cols()) =
        solved.coordinates;
    structure_response_orbital_images_.rightCols(
        solved.orbital_images.cols()) = solved.orbital_images;
    structure_response_residuals_.rightCols(
        solved.equation_residuals.cols()) = solved.equation_residuals;
  }
  structure_response_columns_ = orbital_basis_size_;
  structure_response_revision_ = solved.revision;
  response_model_invalidated_ = false;
  ++revision_;
}

void ResponseNeoWorkspace::invalidate_response_model(
    std::uint64_t revision) {
  structure_response_columns_ = 0;
  structure_response_revision_ = revision;
  response_model_correction_.setZero(
      orbital_basis_size_, orbital_basis_size_);
  response_sample_coefficients_.resize(orbital_basis_size_, 0);
  response_sample_orbital_defects_.resize(problem_.orbital_size(), 0);
  response_model_invalidated_ = true;
  ++revision_;
}

ResponseNeoStructureResponse
ResponseNeoWorkspace::refine_structure_direction(
    const Eigen::VectorXd& forcing,
    double residual_target,
    int* structure_actions) {
  if (forcing.size() != problem_.structure_size() || !forcing.allFinite() ||
      !(residual_target > 0.0) || !std::isfinite(residual_target)) {
    throw std::invalid_argument(
        "invalid NEO structure-response refinement target");
  }
  Eigen::MatrixXd forcing_block(forcing.size(), 1);
  forcing_block.col(0) = forcing;
  const double relative_tolerance = std::min(
      0.5, 0.5 * residual_target / std::max(1.0, forcing.stableNorm()));
  ResponseNeoStructureResponse refined =
      problem_.solve_structure_response(forcing_block, relative_tolerance);
  *structure_actions += refined.block_actions;
  if (refined.revision != structure_response_revision_) {
    invalidate_response_model(refined.revision);
  }
  const double residual_norm = refined.equation_residuals.col(0).stableNorm();
  if (residual_norm > residual_target) {
    std::ostringstream message;
    message.precision(17);
    message << "refined structure response failed full Bp+Cz certification: "
            << residual_norm << " > " << residual_target;
    throw std::runtime_error(message.str());
  }
  return refined;
}

bool ResponseNeoWorkspace::update_response_model(
    const Eigen::VectorXd& coefficients,
    const Eigen::VectorXd& refined_orbital_image,
    const Eigen::VectorXd& model_orbital_image) {
  if (response_model_invalidated_) return false;
  const double norm_squared = coefficients.squaredNorm();
  if (!(norm_squared > 0.0)) return false;
  const Eigen::Index sample = response_sample_coefficients_.cols();
  response_sample_coefficients_.conservativeResize(
      Eigen::NoChange, sample + 1);
  response_sample_orbital_defects_.conservativeResize(
      Eigen::NoChange, sample + 1);
  response_sample_coefficients_.col(sample) = coefficients;
  response_sample_orbital_defects_.col(sample) =
      refined_orbital_image - model_orbital_image;
  const auto basis = orbital_basis_.leftCols(orbital_basis_size_);
  const Eigen::MatrixXd projected_defects =
      basis.transpose() * response_sample_orbital_defects_;
  const Eigen::MatrixXd fitted =
      response_sample_coefficients_.transpose()
          .completeOrthogonalDecomposition()
          .solve(projected_defects.transpose())
          .transpose();
  const Eigen::MatrixXd corrected = 0.5 * (fitted + fitted.transpose());
  const double change =
      (corrected - response_model_correction_).stableNorm();
  const double threshold = 64.0 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, response_model_correction_.stableNorm());
  response_model_correction_ = corrected;
  if (!(change > threshold)) return false;
  ++revision_;
  return true;
}

bool ResponseNeoWorkspace::rebuild_projected_model() {
  const Eigen::Index no = orbital_basis_size_;
  const auto qo = orbital_basis_.leftCols(no);
  const auto mo = orbital_metric_images_.leftCols(no);
  const auto ao = orbital_hessian_images_.leftCols(no);
  const auto bo = orbital_structure_images_.leftCols(no);
  const auto btz = structure_response_orbital_images_.leftCols(no);
  const auto response_residual = structure_response_residuals_.leftCols(no);
  projected_model_.metric = symmetric_part(
      projected_orbital_metric_,
      "response NEO orbital metric is not symmetric");
  Eigen::LLT<Eigen::MatrixXd> metric_factor(projected_model_.metric);
  if (metric_factor.info() != Eigen::Success) {
    throw std::runtime_error(
        "response NEO orbital metric is not positive definite");
  }
  projected_model_.whitening = metric_factor.matrixU().solve(
      Eigen::MatrixXd::Identity(no, no));

  projected_model_.response_closure_scale = bo.stableNorm();
  projected_model_.response_closure_norm = response_residual.stableNorm();
  const Eigen::MatrixXd relaxed_images =
      ao + btz + qo * response_model_correction_;
  const Eigen::MatrixXd projected_response = qo.transpose() * btz;
  const auto relative_skew = [](const Eigen::MatrixXd& matrix) {
    return (matrix - matrix.transpose()).stableNorm() /
        std::max(1.0, matrix.stableNorm());
  };
  std::ostringstream symmetry_context;
  symmetry_context.precision(17);
  symmetry_context
      << "response NEO relaxed orbital block is not symmetric"
      << " [orbital_skew=" << relative_skew(projected_orbital_hessian_)
      << ", response_skew=" << relative_skew(projected_response)
      << ", adjoint_defect="
      << (projected_response - bo.transpose() *
              structure_response_coordinates_.leftCols(no)).stableNorm() /
          std::max(1.0, projected_response.stableNorm());
  symmetry_context << "]";
  const std::string symmetry_message = symmetry_context.str();
  const Eigen::MatrixXd relaxed = symmetric_part(
      qo.transpose() * (ao + btz) + response_model_correction_,
      symmetry_message.c_str(),
      problem_.operator_relative_accuracy());
  const Eigen::MatrixXd whitened =
      projected_model_.whitening.transpose() * relaxed *
      projected_model_.whitening;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(whitened);
  if (spectrum.info() != Eigen::Success) return false;
  projected_model_.eigenvalues = spectrum.eigenvalues();
  projected_model_.eigenvectors = spectrum.eigenvectors();
  const Eigen::VectorXd gradient =
      orbital_basis_.leftCols(no).transpose() * problem_.orbital_gradient();
  projected_model_.white_gradient =
      projected_model_.whitening.transpose() * gradient;

  const Eigen::VectorXd minimum_coefficients = projected_model_.whitening *
      projected_model_.eigenvectors.col(0);
  projected_model_.minimum_curvature_orbital.noalias() =
      qo * minimum_coefficients;
  const Eigen::VectorXd curvature_a = relaxed_images * minimum_coefficients;
  const Eigen::VectorXd curvature_metric =
      projected_model_.eigenvalues[0] * mo * minimum_coefficients;
  projected_model_.curvature_orbital_residual =
      curvature_a - curvature_metric;
  projected_model_.curvature_structure_residual =
      response_residual * minimum_coefficients;
  projected_model_.curvature_orbital_scale = std::max({
      curvature_a.stableNorm(),
      curvature_metric.stableNorm()});
  projected_model_.curvature_structure_scale = std::max(
      (bo * minimum_coefficients).stableNorm(),
      projected_model_.curvature_structure_residual.stableNorm());
  projected_model_.revision = revision_;
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

  const int full_dimension = static_cast<int>(problem_.orbital_size());
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
  const double model_response_tolerance = coarse_response_tolerance(
      options.relative_residual_tolerance);

  ResponseNeoResult result;
  result.step.orbital = Eigen::VectorXd::Zero(problem_.orbital_size());
  result.step.structure = Eigen::VectorXd::Zero(problem_.structure_size());
  result.hessian_step = result.step;
  result.orbital_metric_step = result.step.orbital;
  result.kkt_residual = result.step;
  result.minimum_curvature_orbital = result.step.orbital;

  if (orbital_basis_size_ == 0) {
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
  if (orbital_basis_size_ == 0) {
    throw std::runtime_error("response NEO could not construct an orbital basis");
  }
  if (options.require_curvature_certificate &&
      orbital_basis_size_ == 1 && maximum_dimension > 1 &&
      problem_.orbital_size() > 1) {
    append_orbital(
        generic_probe(problem_.orbital_size()),
        &result.coupled_actions,
        &result.orbital_actions);
  }
  while (true) {
    ++result.iterations;
    if (structure_response_columns_ != orbital_basis_size_) {
      refresh_structure_response(
          model_response_tolerance,
          &result.structure_actions);
    }
    const auto qo = orbital_basis_.leftCols(orbital_basis_size_);
    const auto mo = orbital_metric_images_.leftCols(orbital_basis_size_);
    const auto ao = orbital_hessian_images_.leftCols(orbital_basis_size_);
    const auto bo = orbital_structure_images_.leftCols(orbital_basis_size_);
    const auto z =
        structure_response_coordinates_.leftCols(orbital_basis_size_);
    const auto btz =
        structure_response_orbital_images_.leftCols(orbital_basis_size_);
    const auto response_residual =
        structure_response_residuals_.leftCols(orbital_basis_size_);

    if (projected_model_.revision != revision_) {
      ++result.projected_model_builds;
      if (!rebuild_projected_model()) {
        result.stop_reason = NeoStopReason::NumericalFailure;
        return result;
      }
    }
    const SpectralTrustRegionSolution projected = solve_spectral_trust_region(
        projected_model_.eigenvalues,
        projected_model_.eigenvectors.transpose() *
            projected_model_.white_gradient,
        options.trust_radius);
    const Eigen::VectorXd p_coefficients = projected_model_.whitening *
        projected_model_.eigenvectors * projected.step;
    result.step.orbital.noalias() = qo * p_coefficients;
    result.step.structure.noalias() = z * p_coefficients;
    const Eigen::VectorXd orbital_a = ao * p_coefficients;
    Eigen::VectorXd orbital_bt = btz * p_coefficients;
    const Eigen::VectorXd structure_b = bo * p_coefficients;
    Eigen::VectorXd structure_c =
        -structure_b + response_residual * p_coefficients;
    result.shift = projected.shift;
    result.boundary = projected.boundary;
    result.hard_case = projected.hard_case;
    result.orbital_metric_step.noalias() = mo * p_coefficients;
    result.step_norm = std::sqrt(std::max(
        0.0, result.step.orbital.dot(result.orbital_metric_step)));
    // Inexact Newton convergence is controlled relative to the current
    // gradient, not to the much larger cancelling Hessian terms.  Scaling by
    // those terms would permit an O(||g||) residual and destroy the local
    // Newton rate in ill-conditioned coordinates.
    const double component_residual_target =
        options.absolute_residual_tolerance +
        options.relative_residual_tolerance * gradient_norm;
    const double orbital_residual_target = component_residual_target;
    const double structure_residual_target = component_residual_target;
    if ((structure_b + structure_c).stableNorm() >
        structure_residual_target) {
      ResponseNeoStructureResponse refined = refine_structure_direction(
          structure_b, structure_residual_target,
          &result.structure_actions);
      result.step.structure = refined.coordinates.col(0);
      update_response_model(
          p_coefficients, refined.orbital_images.col(0), orbital_bt);
      orbital_bt = refined.orbital_images.col(0);
      structure_c = -structure_b + refined.equation_residuals.col(0);
      if (response_model_invalidated_) continue;
    }
    result.hessian_step.orbital = orbital_a + orbital_bt;
    result.hessian_step.structure = structure_b + structure_c;
    result.kkt_residual.orbital = problem_.orbital_gradient() +
        result.hessian_step.orbital +
        result.shift * result.orbital_metric_step;
    result.kkt_residual.structure = result.hessian_step.structure;
    const double certified_structure_residual =
        result.kkt_residual.structure.stableNorm();
    result.residual_norm = std::hypot(
        result.kkt_residual.orbital.stableNorm(),
        certified_structure_residual);
    result.residual_target = std::hypot(
        orbital_residual_target, structure_residual_target);
    result.predicted_reduction =
        -problem_.orbital_gradient().dot(result.step.orbital) -
        0.5 * (result.step.orbital.dot(result.hessian_step.orbital) +
               result.step.structure.dot(result.hessian_step.structure));

    result.minimum_curvature_orbital =
        projected_model_.minimum_curvature_orbital;
    const double curvature_orbital_target =
        options.absolute_residual_tolerance +
        curvature_relative_tolerance *
            projected_model_.curvature_orbital_scale;
    const double curvature_structure_target =
        options.absolute_residual_tolerance +
        curvature_relative_tolerance *
            projected_model_.curvature_structure_scale;
    const double positivity_tolerance =
        curvature_relative_tolerance * std::max({
            1.0,
            result.shift,
            std::abs(projected_model_.eigenvalues[0])});
    const bool stationary =
        result.kkt_residual.orbital.stableNorm() <= orbital_residual_target &&
        certified_structure_residual <= structure_residual_target;
    const bool need_curvature_certificate =
        options.require_curvature_certificate || result.boundary ||
        result.hard_case;
    Eigen::VectorXd curvature_orbital =
        projected_model_.curvature_orbital_residual;
    Eigen::VectorXd curvature_structure =
        projected_model_.curvature_structure_residual;
    if (need_curvature_certificate &&
        curvature_structure.stableNorm() > curvature_structure_target) {
      const Eigen::VectorXd minimum_coefficients =
          projected_model_.whitening *
          projected_model_.eigenvectors.col(0);
      const Eigen::VectorXd curvature_forcing = bo * minimum_coefficients;
      ResponseNeoStructureResponse refined = refine_structure_direction(
          curvature_forcing, curvature_structure_target,
          &result.structure_actions);
      update_response_model(
          minimum_coefficients, refined.orbital_images.col(0),
          btz * minimum_coefficients);
      if (response_model_invalidated_) continue;
      curvature_orbital = ao * minimum_coefficients +
          refined.orbital_images.col(0) -
          projected_model_.eigenvalues[0] * mo * minimum_coefficients;
      curvature_structure = refined.equation_residuals.col(0);
    }
    const double certified_curvature_structure_residual =
        curvature_structure.stableNorm();
    result.curvature_residual_norm = std::hypot(
        curvature_orbital.stableNorm(),
        certified_curvature_structure_residual);
    result.curvature_residual_target = std::hypot(
        curvature_orbital_target, curvature_structure_target);
    const bool curvature_converged =
        curvature_orbital.stableNorm() <= curvature_orbital_target &&
        certified_curvature_structure_residual <=
            curvature_structure_target;
    const bool shifted_positive =
        projected_model_.eigenvalues[0] + result.shift >=
        -positivity_tolerance;
    const bool orbital_complete =
        orbital_basis_size_ == problem_.orbital_size();
    result.global_curvature_certified =
        orbital_complete && curvature_converged;
    const bool required_curvature_converged =
        !need_curvature_certificate || curvature_converged;
    if (stationary && required_curvature_converged && shifted_positive) {
      result.stop_reason = NeoStopReason::Converged;
      return result;
    }

    const int basis_size = static_cast<int>(orbital_basis_size_);
    if (basis_size >= maximum_dimension) {
      // A certified combined response supplies a new small-space secant even
      // when no further orbital direction can be appended.  Re-solve that
      // updated projected model before declaring the explicit work budget
      // exhausted.
      if (projected_model_.revision != revision_) continue;
      result.stop_reason = NeoStopReason::SubspaceLimit;
      return result;
    }

    bool expanded = false;
    if (result.kkt_residual.orbital.stableNorm() >
        orbital_residual_target) {
      expanded = append_orbital(
          preconditioned_orbital(
              problem_, -result.kkt_residual.orbital, result.shift),
          &result.coupled_actions,
          &result.orbital_actions);
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
