#include "vbscf/optimization/coupled/projection_cache.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace xmvb::vb {
namespace {

double accumulation_roundoff(int dimension, int n_vectors) {
  const double epsilon = std::numeric_limits<double>::epsilon();
  const double operations = 8.0 * std::max(1, dimension) *
      std::max(1, n_vectors);
  const double product = operations * epsilon;
  return product < 1.0
      ? product / (1.0 - product)
      : std::numeric_limits<double>::infinity();
}

void fix_column_sign(
    Eigen::VectorXd* vector,
    Eigen::VectorXd* image = nullptr) {
  Eigen::Index pivot = 0;
  vector->cwiseAbs().maxCoeff(&pivot);
  if ((*vector)[pivot] < 0.0) {
    *vector = -*vector;
    if (image != nullptr) *image = -*image;
  }
}

void append_columns(
    Eigen::MatrixXd* destination,
    const Eigen::Ref<const Eigen::MatrixXd>& columns) {
  if (columns.cols() == 0) return;
  const Eigen::Index old_columns = destination->cols();
  destination->conservativeResize(Eigen::NoChange, old_columns + columns.cols());
  destination->rightCols(columns.cols()) = columns;
}

Eigen::MatrixXd symmetrized(const Eigen::MatrixXd& matrix) {
  return 0.5 * (matrix + matrix.transpose());
}

}  // namespace

Eigen::VectorXd CoupledCachedBlocks::packed() const {
  Eigen::VectorXd result(orbital.size() + response.size());
  result.head(orbital.size()) = orbital;
  result.tail(response.size()) = response;
  return result;
}

CoupledProjectionCache::CoupledProjectionCache(
    const CoupledNewtonOperator& coupled_operator)
    : coupled_operator_(&coupled_operator),
      orbital_basis_(coupled_operator.n_orbital_coordinates(), 0),
      response_basis_(coupled_operator.n_response_coordinates(), 0),
      orbital_hessian_images_(
          coupled_operator.n_orbital_coordinates(), 0),
      orbital_to_response_images_(
          coupled_operator.n_response_coordinates(), 0),
      orbital_metric_images_(
          coupled_operator.n_orbital_coordinates(), 0),
      response_to_orbital_images_(
          coupled_operator.n_orbital_coordinates(), 0),
      response_hessian_images_(
          coupled_operator.n_response_coordinates(), 0) {}

int CoupledProjectionCache::orbital_subspace_size() const noexcept {
  return orbital_basis_.cols();
}

int CoupledProjectionCache::response_subspace_size() const noexcept {
  return response_basis_.cols();
}

const Eigen::MatrixXd&
CoupledProjectionCache::orbital_basis() const noexcept {
  return orbital_basis_;
}

const Eigen::MatrixXd&
CoupledProjectionCache::response_basis() const noexcept {
  return response_basis_;
}

const Eigen::MatrixXd&
CoupledProjectionCache::orbital_hessian_images() const noexcept {
  return orbital_hessian_images_;
}

const Eigen::MatrixXd&
CoupledProjectionCache::orbital_to_response_images() const noexcept {
  return orbital_to_response_images_;
}

const Eigen::MatrixXd&
CoupledProjectionCache::orbital_metric_images() const noexcept {
  return orbital_metric_images_;
}

const Eigen::MatrixXd&
CoupledProjectionCache::response_to_orbital_images() const noexcept {
  return response_to_orbital_images_;
}

const Eigen::MatrixXd&
CoupledProjectionCache::response_hessian_images() const noexcept {
  return response_hessian_images_;
}

int CoupledProjectionCache::append_orbital_block(
    const Eigen::Ref<const Eigen::MatrixXd>& candidates) {
  const int n_orbitals = coupled_operator_->n_orbital_coordinates();
  if (candidates.rows() != n_orbitals || !candidates.allFinite()) {
    throw std::invalid_argument(
        "orbital projection candidates have invalid dimensions or values");
  }
  if (candidates.cols() == 0) return 0;
  const Eigen::MatrixXd candidate_metric_images =
      coupled_operator_->apply_orbital_metric(candidates);
  Eigen::MatrixXd accepted(n_orbitals, candidates.cols());
  Eigen::MatrixXd accepted_metric(n_orbitals, candidates.cols());
  int n_accepted = 0;
  const double rank_roundoff = accumulation_roundoff(
      n_orbitals,
      orbital_subspace_size() + candidates.cols());
  for (Eigen::Index column = 0; column < candidates.cols(); ++column) {
    Eigen::VectorXd vector = candidates.col(column);
    Eigen::VectorXd metric_image = candidate_metric_images.col(column);
    const double original_squared_norm = vector.dot(metric_image);
    const double original_absolute_product =
        vector.cwiseAbs().dot(metric_image.cwiseAbs());
    if (original_squared_norm <
        -rank_roundoff * original_absolute_product) {
      throw std::runtime_error(
          "orbital metric is not positive on a projection candidate");
    }
    if (!(original_squared_norm > 0.0)) continue;

    for (int pass = 0; pass < 2; ++pass) {
      if (orbital_basis_.cols() != 0) {
        const Eigen::VectorXd projection =
            orbital_basis_.transpose() * metric_image;
        vector.noalias() -= orbital_basis_ * projection;
        metric_image.noalias() -= orbital_metric_images_ * projection;
      }
      if (n_accepted != 0) {
        const auto basis = accepted.leftCols(n_accepted);
        const auto basis_metric = accepted_metric.leftCols(n_accepted);
        const Eigen::VectorXd projection =
            basis.transpose() * metric_image;
        vector.noalias() -= basis * projection;
        metric_image.noalias() -= basis_metric * projection;
      }
    }
    const double squared_norm = vector.dot(metric_image);
    const double absolute_product =
        vector.cwiseAbs().dot(metric_image.cwiseAbs());
    if (squared_norm < -rank_roundoff * absolute_product) {
      throw std::runtime_error(
          "orbital metric lost positivity during numerical-rank filtering");
    }
    if (!(squared_norm >
          rank_roundoff * rank_roundoff * original_squared_norm)) {
      continue;
    }
    const double inverse_norm = 1.0 / std::sqrt(squared_norm);
    vector *= inverse_norm;
    metric_image *= inverse_norm;
    fix_column_sign(&vector, &metric_image);
    accepted.col(n_accepted) = vector;
    accepted_metric.col(n_accepted) = metric_image;
    ++n_accepted;
  }
  if (n_accepted == 0) return 0;

  accepted.conservativeResize(Eigen::NoChange, n_accepted);
  accepted_metric.conservativeResize(Eigen::NoChange, n_accepted);
  const Eigen::MatrixXd hessian_images =
      coupled_operator_->apply_orbital_hessian(accepted);
  const Eigen::MatrixXd response_images =
      coupled_operator_->apply_orbital_to_response(accepted);
  append_columns(&orbital_basis_, accepted);
  append_columns(&orbital_metric_images_, accepted_metric);
  append_columns(&orbital_hessian_images_, hessian_images);
  append_columns(&orbital_to_response_images_, response_images);
  return n_accepted;
}

int CoupledProjectionCache::append_response_block(
    const Eigen::Ref<const Eigen::MatrixXd>& candidates) {
  const int n_response = coupled_operator_->n_response_coordinates();
  if (candidates.rows() != n_response || !candidates.allFinite()) {
    throw std::invalid_argument(
        "response projection candidates have invalid dimensions or values");
  }
  if (candidates.cols() == 0) return 0;
  Eigen::MatrixXd accepted(n_response, candidates.cols());
  int n_accepted = 0;
  const double rank_roundoff = accumulation_roundoff(
      n_response,
      response_subspace_size() + candidates.cols());
  for (Eigen::Index column = 0; column < candidates.cols(); ++column) {
    Eigen::VectorXd vector = candidates.col(column);
    const double original_norm = vector.stableNorm();
    if (!(original_norm > 0.0)) continue;
    for (int pass = 0; pass < 2; ++pass) {
      if (response_basis_.cols() != 0) {
        const Eigen::VectorXd projection =
            response_basis_.transpose() * vector;
        vector.noalias() -= response_basis_ * projection;
      }
      if (n_accepted != 0) {
        const auto basis = accepted.leftCols(n_accepted);
        const Eigen::VectorXd projection = basis.transpose() * vector;
        vector.noalias() -= basis * projection;
      }
    }
    const double norm = vector.stableNorm();
    if (!(norm > rank_roundoff * original_norm)) continue;
    vector /= norm;
    fix_column_sign(&vector);
    accepted.col(n_accepted) = vector;
    ++n_accepted;
  }
  if (n_accepted == 0) return 0;

  accepted.conservativeResize(Eigen::NoChange, n_accepted);
  const Eigen::MatrixXd orbital_images =
      coupled_operator_->apply_response_to_orbital(accepted);
  const Eigen::MatrixXd hessian_images =
      coupled_operator_->apply_response_hessian(accepted);
  append_columns(&response_basis_, accepted);
  append_columns(&response_to_orbital_images_, orbital_images);
  append_columns(&response_hessian_images_, hessian_images);
  return n_accepted;
}

Eigen::MatrixXd CoupledProjectionCache::projected_orbital_hessian() const {
  return symmetrized(orbital_basis_.transpose() * orbital_hessian_images_);
}

Eigen::MatrixXd CoupledProjectionCache::projected_orbital_metric() const {
  return symmetrized(orbital_basis_.transpose() * orbital_metric_images_);
}

Eigen::MatrixXd CoupledProjectionCache::projected_response_hessian() const {
  return symmetrized(response_basis_.transpose() * response_hessian_images_);
}

Eigen::MatrixXd CoupledProjectionCache::projected_coupling() const {
  return response_basis_.transpose() * orbital_to_response_images_;
}

double CoupledProjectionCache::projected_coupling_adjoint_error() const {
  const Eigen::MatrixXd forward = projected_coupling();
  const Eigen::MatrixXd adjoint =
      (orbital_basis_.transpose() * response_to_orbital_images_).transpose();
  return (forward - adjoint).stableNorm() /
      std::max({1.0, forward.stableNorm(), adjoint.stableNorm()});
}

CoupledCachedBlocks CoupledProjectionCache::reconstruct_image(
    const Eigen::Ref<const Eigen::VectorXd>& orbital_coordinates,
    const Eigen::Ref<const Eigen::VectorXd>& response_coordinates,
    double orbital_shift) const {
  if (orbital_coordinates.size() != orbital_subspace_size() ||
      response_coordinates.size() != response_subspace_size() ||
      !orbital_coordinates.allFinite() ||
      !response_coordinates.allFinite() || !std::isfinite(orbital_shift)) {
    throw std::invalid_argument(
        "cached coupled-image coordinates are invalid");
  }
  CoupledCachedBlocks image;
  image.orbital = orbital_hessian_images_ * orbital_coordinates +
      response_to_orbital_images_ * response_coordinates +
      orbital_shift * orbital_metric_images_ * orbital_coordinates;
  image.response = orbital_to_response_images_ * orbital_coordinates +
      response_hessian_images_ * response_coordinates;
  return image;
}

CoupledCachedBlocks CoupledProjectionCache::reconstruct_ritz_residual(
    const Eigen::Ref<const Eigen::VectorXd>& orbital_coordinates,
    const Eigen::Ref<const Eigen::VectorXd>& response_coordinates,
    double ritz_value) const {
  return reconstruct_image(
      orbital_coordinates,
      response_coordinates,
      -ritz_value);
}

}  // namespace xmvb::vb
