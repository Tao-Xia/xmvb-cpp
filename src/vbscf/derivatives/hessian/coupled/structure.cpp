#include "vbscf/derivatives/hessian/coupled/structure.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/QR>

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

  const Eigen::MatrixXd& selected =
      accepted_point_->selected_state_eigenvectors;
  const StructureActionResult selected_images = action_->apply(selected);
  if (selected_images.overlap.rows() != n_structures_ ||
      selected_images.overlap.cols() != n_states_ ||
      !selected_images.overlap.allFinite()) {
    throw std::runtime_error(
        "coupled structure operator received invalid selected overlap images");
  }
  const Eigen::MatrixXd selected_metric =
      selected.transpose() * selected_images.overlap;
  const double metric_error =
      (selected_metric - Eigen::MatrixXd::Identity(n_states_, n_states_)).norm();
  const double metric_tolerance = 1.0e3 *
      std::numeric_limits<double>::epsilon() *
      std::max(1, n_structures_);
  if (!std::isfinite(metric_error) || metric_error > metric_tolerance) {
    throw std::invalid_argument(
        "coupled structure operator requires S-orthonormal selected states");
  }

  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> factor(selected_images.overlap);
  factor.setThreshold(
      std::numeric_limits<double>::epsilon() *
      std::max(n_structures_, n_states_));
  if (factor.rank() != n_states_) {
    throw std::invalid_argument(
        "selected-state metric images are linearly dependent");
  }
  constraint_units_ = factor.householderQ() *
      Eigen::MatrixXd::Identity(n_structures_, n_states_);
}

int StructureTangentOperator::n_structures() const noexcept {
  return n_structures_;
}

int StructureTangentOperator::n_states() const noexcept {
  return n_states_;
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
  coefficients->noalias() -= constraint_units_ *
      (constraint_units_.transpose() * (*coefficients));
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

StructureTangent StructureTangentOperator::apply_hessian(
    const StructureTangent& tangent) const {
  StructureTangent horizontal = project(tangent.scaled_coefficients);
  const StructureActionResult images =
      action_->apply(horizontal.scaled_coefficients);
  validate_shape(images.hamiltonian);
  validate_shape(images.overlap);
  Eigen::MatrixXd result = images.hamiltonian;
  result.noalias() -= images.overlap * energies_.asDiagonal();
  return project(result);
}

StructureTangent StructureTangentOperator::apply_metric(
    const StructureTangent& tangent) const {
  StructureTangent horizontal = project(tangent.scaled_coefficients);
  const StructureActionResult images =
      action_->apply(horizontal.scaled_coefficients);
  validate_shape(images.overlap);
  return project(images.overlap);
}

double StructureTangentOperator::squared_norm(
    const StructureTangent& tangent) const {
  const StructureTangent horizontal = project(tangent.scaled_coefficients);
  const StructureActionResult images =
      action_->apply(horizontal.scaled_coefficients);
  validate_shape(images.overlap);
  const double norm =
      (horizontal.scaled_coefficients.array() * images.overlap.array()).sum();
  const double tolerance = 1.0e3 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, horizontal.scaled_coefficients.squaredNorm());
  if (!std::isfinite(norm) || norm < -tolerance) {
    throw std::runtime_error("structure tangent metric is not positive semidefinite");
  }
  return std::max(0.0, norm);
}

}  // namespace xmvb::vb
