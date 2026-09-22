#include "vbscf/derivatives/hessian/coupled/structure.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/QR>
#include <Eigen/LU>

#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/structures/assembly/action.hpp"

namespace xmvb::vb {
namespace {

bool weights_are_equal(const std::vector<double>& weights) {
  if (weights.empty()) return false;
  const double reference = weights.front();
  const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, std::abs(reference));
  return std::all_of(
      weights.begin(),
      weights.end(),
      [&](double weight) {
        return std::abs(weight - reference) <= tolerance;
      });
}

}  // namespace

StructureTangentOperator::StructureTangentOperator(
    std::shared_ptr<const AcceptedPointContext> accepted_point,
    const StructureAction& action)
    : accepted_point_(std::move(accepted_point)), action_(&action) {
  if (accepted_point_ == nullptr) {
    throw std::invalid_argument(
        "coupled structure operator requires an accepted point");
  }
  n_structures_ = accepted_point_->n_structures;
  n_states_ = static_cast<int>(accepted_point_->selected_state_indices.size());
  if (n_structures_ <= 0 || n_states_ <= 0 ||
      action_->n_structures() != n_structures_ ||
      accepted_point_->selected_state_eigenvectors.rows() != n_structures_ ||
      accepted_point_->selected_state_eigenvectors.cols() != n_states_ ||
      accepted_point_->selected_state_energies.size() !=
          static_cast<std::size_t>(n_states_) ||
      accepted_point_->normalized_state_weights.size() !=
          static_cast<std::size_t>(n_states_)) {
    throw std::invalid_argument(
        "coupled structure operator dimensions are inconsistent");
  }
  if (n_states_ > 1 &&
      !weights_are_equal(accepted_point_->normalized_state_weights)) {
    throw std::invalid_argument(
        "coupled structure operator supports only equal-weight state averages");
  }

  energies_ = Eigen::Map<const Eigen::VectorXd>(
      accepted_point_->selected_state_energies.data(), n_states_);
  coordinate_scales_.resize(n_states_);
  for (int state = 0; state < n_states_; ++state) {
    const double weight =
        accepted_point_->normalized_state_weights[static_cast<std::size_t>(state)];
    if (!(weight > 0.0) || !std::isfinite(weight) ||
        !std::isfinite(energies_[state])) {
      throw std::invalid_argument(
          "coupled structure operator requires positive finite state weights and energies");
    }
    coordinate_scales_[state] = std::sqrt(2.0 * weight);
  }

  const StructureDiagonal& diagonal = action_->preconditioner_diagonal();
  if (diagonal.hamiltonian.size() != n_structures_ ||
      diagonal.overlap.size() != n_structures_ ||
      !diagonal.hamiltonian.allFinite() || !diagonal.overlap.allFinite() ||
      (diagonal.overlap.array() <= 0.0).any()) {
    throw std::invalid_argument(
        "coupled structure operator requires finite positive H/S diagonals");
  }
  metric_diagonal_ = diagonal.overlap;
  hessian_diagonal_.resize(n_structures_, n_states_);
  absolute_hessian_diagonal_.resize(n_structures_, n_states_);
  for (int state = 0; state < n_states_; ++state) {
    hessian_diagonal_.col(state) =
        diagonal.hamiltonian - energies_[state] * diagonal.overlap;
    absolute_hessian_diagonal_.col(state) =
        hessian_diagonal_.col(state).cwiseAbs();
  }

  selected_ = accepted_point_->selected_state_eigenvectors;
  if (!accepted_point_->selected_state_structure_images.has_value()) {
    throw std::invalid_argument(
        "coupled structure operator requires cached selected-state structure images");
  }
  const StructureActionResult& selected_images =
      accepted_point_->selected_state_structure_images.value();
  if (selected_images.hamiltonian.rows() != n_structures_ ||
      selected_images.hamiltonian.cols() != n_states_ ||
      !selected_images.hamiltonian.allFinite()) {
    throw std::runtime_error(
        "coupled structure operator received invalid selected Hamiltonian images");
  }
  if (selected_images.overlap.rows() != n_structures_ ||
      selected_images.overlap.cols() != n_states_ ||
      !selected_images.overlap.allFinite()) {
    throw std::runtime_error(
        "coupled structure operator received invalid selected overlap images");
  }
  const Eigen::MatrixXd selected_metric =
      selected_.transpose() * selected_images.overlap;
  const double metric_error =
      (selected_metric - Eigen::MatrixXd::Identity(n_states_, n_states_)).norm();
  const double metric_tolerance = 1.0e3 *
      std::numeric_limits<double>::epsilon() *
      std::max(1, n_structures_);
  if (!std::isfinite(metric_error) || metric_error > metric_tolerance) {
    throw std::invalid_argument(
        "coupled structure operator requires S-orthonormal selected states");
  }
  selected_metric_inverse_ = selected_metric.inverse();
  if (!selected_metric_inverse_.allFinite()) {
    throw std::runtime_error(
        "coupled structure selected metric inverse is not finite");
  }

  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> factor(selected_images.overlap);
  factor.setThreshold(
      std::numeric_limits<double>::epsilon() *
      std::max(n_structures_, n_states_));
  if (factor.rank() != n_states_) {
    throw std::invalid_argument(
        "selected-state metric images are linearly dependent");
  }
  constraint_qr_.compute(selected_images.overlap);
}

int StructureTangentOperator::n_structures() const noexcept {
  return n_structures_;
}

int StructureTangentOperator::n_states() const noexcept {
  return n_states_;
}

int StructureTangentOperator::tangent_size() const noexcept {
  return (n_structures_ - n_states_) * n_states_;
}

void StructureTangentOperator::validate_shape(
    const Eigen::MatrixXd& coefficients) const {
  if (coefficients.rows() != n_structures_ ||
      coefficients.cols() != n_states_ ||
      !coefficients.allFinite()) {
    throw std::invalid_argument(
        "structure tangent has incompatible dimensions or non-finite values");
  }
}

void StructureTangentOperator::project_in_place(
    Eigen::MatrixXd* coefficients) const {
  if (coefficients == nullptr) {
    throw std::invalid_argument("structure tangent output must not be null");
  }
  validate_shape(*coefficients);
  Eigen::MatrixXd transformed =
      constraint_qr_.householderQ().adjoint() * (*coefficients);
  transformed.topRows(n_states_).setZero();
  *coefficients = constraint_qr_.householderQ() * transformed;
}

Eigen::VectorXd StructureTangentOperator::coordinates(
    const StructureTangent& tangent) const {
  validate_shape(tangent.scaled_coefficients);
  const Eigen::MatrixXd transformed =
      constraint_qr_.householderQ().adjoint() *
      tangent.scaled_coefficients;
  const double tolerance = 1024.0 * std::numeric_limits<double>::epsilon() *
      std::max(1, n_structures_ * n_states_) *
      std::max(1.0, tangent.scaled_coefficients.norm());
  if (transformed.topRows(n_states_).norm() > tolerance) {
    throw std::invalid_argument("structure tangent is not horizontal");
  }
  const Eigen::MatrixXd independent = transformed.bottomRows(
      n_structures_ - n_states_);
  return Eigen::Map<const Eigen::VectorXd>(
      independent.data(), independent.size());
}

Eigen::VectorXd StructureTangentOperator::project_coordinates(
    const Eigen::Ref<const Eigen::MatrixXd>& scaled_coefficients) const {
  validate_shape(scaled_coefficients);
  const Eigen::MatrixXd transformed =
      constraint_qr_.householderQ().adjoint() * scaled_coefficients;
  const Eigen::MatrixXd independent =
      transformed.bottomRows(n_structures_ - n_states_);
  return Eigen::Map<const Eigen::VectorXd>(
      independent.data(), independent.size());
}

StructureTangent StructureTangentOperator::expand(
    const Eigen::VectorXd& coordinates) const {
  if (coordinates.size() != tangent_size() || !coordinates.allFinite()) {
    throw std::invalid_argument(
        "structure coordinates have incompatible dimensions or values");
  }
  Eigen::MatrixXd transformed =
      Eigen::MatrixXd::Zero(n_structures_, n_states_);
  const Eigen::Index horizontal_rows = n_structures_ - n_states_;
  const Eigen::Map<const Eigen::MatrixXd> independent(
      coordinates.data(), horizontal_rows, n_states_);
  transformed.bottomRows(horizontal_rows) = independent;
  return StructureTangent{
      constraint_qr_.householderQ() * transformed};
}

StructureTangent StructureTangentOperator::project(
    const Eigen::Ref<const Eigen::MatrixXd>& scaled_coefficients) const {
  StructureTangent result{scaled_coefficients};
  project_in_place(&result.scaled_coefficients);
  return result;
}

StructureTangent StructureTangentOperator::from_coefficient_response(
    const Eigen::Ref<const Eigen::MatrixXd>& coefficient_response) const {
  validate_shape(coefficient_response);
  Eigen::MatrixXd scaled = coefficient_response;
  for (int state = 0; state < n_states_; ++state) {
    scaled.col(state) *= coordinate_scales_[state];
  }
  return project(scaled);
}

Eigen::MatrixXd StructureTangentOperator::coefficient_response(
    const StructureTangent& tangent) const {
  StructureTangent horizontal = project(tangent.scaled_coefficients);
  for (int state = 0; state < n_states_; ++state) {
    horizontal.scaled_coefficients.col(state) /= coordinate_scales_[state];
  }
  return horizontal.scaled_coefficients;
}

double StructureTangentOperator::maximum_coefficient_component(
    const StructureTangent& tangent) const {
  const Eigen::MatrixXd response = coefficient_response(tangent);
  if (response.size() == 0) return 0.0;
  if (!action_->supports_integral_direction()) {
    return response.cwiseAbs().maxCoeff();
  }
  const Eigen::MatrixXd orthogonal =
      action_->orthogonalize_structure_block(response);
  return orthogonal.size() == 0
      ? 0.0
      : orthogonal.cwiseAbs().maxCoeff();
}

StructureTangent StructureTangentOperator::apply_hessian(
    const StructureTangent& tangent) const {
  return apply_coupling(tangent).hessian;
}

Eigen::VectorXd StructureTangentOperator::apply_hessian_coordinates(
    const Eigen::VectorXd& coordinates) const {
  return apply_coupling_coordinates(coordinates).hessian_coordinates;
}

StructureCouplingAction StructureTangentOperator::apply_coupling(
    const StructureTangent& tangent) const {
  const Eigen::VectorXd tangent_coordinates = coordinates(tangent);
  StructureCoordinateCouplingAction coordinate_action =
      apply_coupling_coordinates(tangent_coordinates);
  return StructureCouplingAction{
      expand(coordinate_action.hessian_coordinates),
      std::move(coordinate_action.coefficient_response),
      std::move(coordinate_action.adjoint_multipliers)};
}

StructureCoordinateCouplingAction
StructureTangentOperator::apply_coupling_coordinates(
    const Eigen::VectorXd& coordinates) const {
  StructureTangent horizontal = expand(coordinates);
  Eigen::MatrixXd raw = std::move(horizontal.scaled_coefficients);
  for (int state = 0; state < n_states_; ++state) {
    raw.col(state) /= coordinate_scales_[state];
  }
  const StructureActionResult images = action_->apply(raw);
  validate_shape(images.hamiltonian);
  validate_shape(images.overlap);
  Eigen::MatrixXd shifted = images.hamiltonian;
  shifted.noalias() -= images.overlap * energies_.asDiagonal();

  StructureCoordinateCouplingAction result;
  result.coefficient_response = std::move(raw);
  result.adjoint_multipliers =
      -selected_metric_inverse_ * (selected_.transpose() * shifted);
  Eigen::MatrixXd scaled_shifted = std::move(shifted);
  for (int state = 0; state < n_states_; ++state) {
    scaled_shifted.col(state) *= coordinate_scales_[state];
  }
  result.hessian_coordinates = project_coordinates(scaled_shifted);
  return result;
}

StructureTangent StructureTangentOperator::apply_metric(
    const StructureTangent& tangent) const {
  return expand(apply_metric_coordinates(coordinates(tangent)));
}

Eigen::VectorXd StructureTangentOperator::apply_metric_coordinates(
    const Eigen::VectorXd& coordinates) const {
  const StructureTangent horizontal = expand(coordinates);
  const StructureActionResult images =
      action_->apply(horizontal.scaled_coefficients);
  validate_shape(images.overlap);
  return project_coordinates(images.overlap);
}

double StructureTangentOperator::metric_inner_product(
    const Eigen::VectorXd& left,
    const Eigen::VectorXd& right) const {
  if (left.size() != tangent_size() || right.size() != tangent_size() ||
      !left.allFinite() || !right.allFinite()) {
    throw std::invalid_argument(
        "structure metric operands have incompatible dimensions or values");
  }
  const double product = left.dot(apply_metric_coordinates(right));
  if (!std::isfinite(product)) {
    throw std::runtime_error("structure metric inner product is not finite");
  }
  return product;
}

double StructureTangentOperator::diagonal_dual_norm(
    const Eigen::VectorXd& covector) const {
  if (covector.size() != tangent_size() || !covector.allFinite()) {
    throw std::invalid_argument(
        "structure dual-norm covector has incompatible dimensions or values");
  }
  Eigen::MatrixXd ambient = expand(covector).scaled_coefficients;
  for (int state = 0; state < n_states_; ++state) {
    ambient.col(state).array() /= metric_diagonal_.array();
  }
  const Eigen::VectorXd inverse_image = project_coordinates(ambient);
  const double squared_norm = covector.dot(inverse_image);
  const double tolerance = 1024.0 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, covector.squaredNorm());
  if (!std::isfinite(squared_norm) || squared_norm < -tolerance) {
    throw std::runtime_error(
        "structure diagonal dual norm is not positive semidefinite");
  }
  return std::sqrt(std::max(0.0, squared_norm));
}

Eigen::VectorXd
StructureTangentOperator::apply_inverse_shifted_preconditioner(
    const Eigen::VectorXd& covector,
    double shift) const {
  if (covector.size() != tangent_size() || !covector.allFinite()) {
    throw std::invalid_argument(
        "structure preconditioner covector has incompatible dimensions or values");
  }
  if (!std::isfinite(shift) || shift < 0.0) {
    throw std::invalid_argument(
        "structure preconditioner shift must be finite and nonnegative");
  }

  Eigen::MatrixXd ambient = expand(covector).scaled_coefficients;
  for (int state = 0; state < n_states_; ++state) {
    Eigen::VectorXd denominator =
        absolute_hessian_diagonal_.col(state) + shift * metric_diagonal_;
    const double scale = denominator.maxCoeff();
    if (scale == 0.0) {
      continue;
    }
    const double floor = std::max(
        std::numeric_limits<double>::min(),
        std::numeric_limits<double>::epsilon() *
            static_cast<double>(std::max(1, n_structures_)) * scale);
    denominator = denominator.cwiseMax(floor);
    ambient.col(state).array() /= denominator.array();
  }
  return project_coordinates(ambient);
}

Eigen::VectorXd
StructureTangentOperator::apply_inverse_augmented_hessian_diagonal(
    const Eigen::VectorXd& covector,
    double eigenvalue) const {
  if (covector.size() != tangent_size() || !covector.allFinite()) {
    throw std::invalid_argument(
        "structure AH preconditioner covector has incompatible dimensions or values");
  }
  if (!std::isfinite(eigenvalue)) {
    throw std::invalid_argument(
        "structure AH preconditioner eigenvalue must be finite");
  }

  Eigen::MatrixXd ambient = expand(covector).scaled_coefficients;
  constexpr double kDenominatorFloor = 1.0e-8;
  for (int state = 0; state < n_states_; ++state) {
    Eigen::VectorXd denominator = hessian_diagonal_.col(state) -
        eigenvalue * metric_diagonal_;
    for (Eigen::Index i = 0; i < denominator.size(); ++i) {
      if (std::abs(denominator[i]) < kDenominatorFloor) {
        denominator[i] = kDenominatorFloor;
      }
    }
    ambient.col(state).array() /= denominator.array();
  }
  return project_coordinates(ambient);
}

double StructureTangentOperator::squared_norm(
    const StructureTangent& tangent) const {
  const Eigen::VectorXd horizontal = coordinates(tangent);
  const double norm = metric_inner_product(horizontal, horizontal);
  const double tolerance = 1.0e3 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, horizontal.squaredNorm());
  if (!std::isfinite(norm) || norm < -tolerance) {
    throw std::runtime_error("structure tangent metric is not positive semidefinite");
  }
  return std::max(0.0, norm);
}

}  // namespace xmvb::vb
