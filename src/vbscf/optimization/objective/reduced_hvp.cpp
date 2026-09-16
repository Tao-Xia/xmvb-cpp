#include "vbscf/optimization/objective/reduced_hvp.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace xmvb::vb {
namespace {

const char* bool_name(bool value) {
  return value ? "true" : "false";
}

}  // namespace

bool SymmetricResponseModel::add(
    const Eigen::VectorXd& direction,
    const Eigen::VectorXd& exact_response) {
  if (direction.size() == 0 ||
      exact_response.size() != direction.size() ||
      !direction.allFinite() ||
      !exact_response.allFinite()) {
    throw std::invalid_argument(
        "response sample has invalid dimensions or non-finite values");
  }
  if (directions_.rows() != 0 && directions_.rows() != direction.size()) {
    throw std::invalid_argument("response sample dimension changed");
  }

  Eigen::VectorXd orthogonal_direction = direction;
  Eigen::VectorXd orthogonal_response = exact_response;
  if (directions_.cols() > 0) {
    // Two passes keep the sampled basis orthogonal without changing the exact
    // linear combination represented by its paired response image.
    for (int pass = 0; pass < 2; ++pass) {
      const Eigen::VectorXd coefficients =
          directions_.transpose() * orthogonal_direction;
      orthogonal_direction.noalias() -= directions_ * coefficients;
      orthogonal_response.noalias() -= exact_images_ * coefficients;
    }
  }

  const double sample_norm = direction.stableNorm();
  const double orthogonal_norm = orthogonal_direction.stableNorm();
  const double scaled_epsilon =
      static_cast<double>(std::max<Eigen::Index>(1, direction.size())) *
      std::numeric_limits<double>::epsilon();
  const double roundoff_limit =
      scaled_epsilon < 1.0
          ? scaled_epsilon / (1.0 - scaled_epsilon)
          : 1.0;
  if (!(sample_norm > 0.0) ||
      !(orthogonal_norm > roundoff_limit * sample_norm)) {
    return false;
  }

  orthogonal_direction /= orthogonal_norm;
  orthogonal_response /= orthogonal_norm;
  const Eigen::Index old_rank = directions_.cols();
  directions_.conservativeResize(direction.size(), old_rank + 1);
  exact_images_.conservativeResize(direction.size(), old_rank + 1);
  directions_.col(old_rank) = orthogonal_direction;
  exact_images_.col(old_rank) = orthogonal_response;
  rebuild_symmetric_images();
  return true;
}

void SymmetricResponseModel::rebuild_symmetric_images() {
  if (directions_.cols() == 0) {
    symmetric_images_.resize(directions_.rows(), 0);
    projected_response_.resize(0, 0);
    return;
  }
  const Eigen::MatrixXd cross = directions_.transpose() * exact_images_;
  const Eigen::MatrixXd skew = 0.5 * (cross - cross.transpose());
  symmetric_images_ = exact_images_ - directions_ * skew;
  projected_response_ = 0.5 * (cross + cross.transpose());
}

Eigen::VectorXd SymmetricResponseModel::apply(
    const Eigen::VectorXd& direction) const {
  if (directions_.rows() == 0) {
    return Eigen::VectorXd::Zero(direction.size());
  }
  if (direction.size() != directions_.rows() || !direction.allFinite()) {
    throw std::invalid_argument("response-model direction is invalid");
  }
  const Eigen::VectorXd coordinates = directions_.transpose() * direction;
  return directions_ * (symmetric_images_.transpose() * direction) +
      symmetric_images_ * coordinates -
      directions_ * (projected_response_ * coordinates);
}

Eigen::MatrixXd SymmetricResponseModel::apply_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& directions) const {
  if (directions_.rows() == 0) {
    return Eigen::MatrixXd::Zero(directions.rows(), directions.cols());
  }
  if (directions.rows() != directions_.rows() || !directions.allFinite()) {
    throw std::invalid_argument("response-model direction block is invalid");
  }
  const Eigen::MatrixXd coordinates = directions_.transpose() * directions;
  return directions_ * (symmetric_images_.transpose() * directions) +
      symmetric_images_ * coordinates -
      directions_ * (projected_response_ * coordinates);
}

Eigen::Index SymmetricResponseModel::dimension() const noexcept {
  return directions_.rows();
}

Eigen::Index SymmetricResponseModel::rank() const noexcept {
  return directions_.cols();
}

Eigen::MatrixXd ReducedHvp::apply_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions) {
  Eigen::MatrixXd responses(
      reduced_directions.rows(),
      reduced_directions.cols());
  for (Eigen::Index column = 0;
       column < reduced_directions.cols();
       ++column) {
    responses.col(column) = apply(reduced_directions.col(column));
  }
  return responses;
}

ExactReducedHvp::ExactReducedHvp(
    const VbScfObjective& objective,
    const OrbitalChart& current_space)
    : exact_operator_(
          objective.second_order_context(),
          &objective.input(),
          SparseParameterLayout(
              objective.input().orbital_preparation_input),
          &current_space) {}

Eigen::VectorXd ExactReducedHvp::apply(
    const Eigen::VectorXd& reduced_direction) {
  return exact_operator_.apply_reduced(reduced_direction);
}

Eigen::MatrixXd ExactReducedHvp::apply_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions) {
  return exact_operator_.apply_reduced_batch(reduced_directions);
}

Eigen::VectorXd ExactReducedHvp::apply_core(
    const Eigen::VectorXd& reduced_direction) {
  return exact_operator_.apply_reduced(
      reduced_direction,
      {.direct_core_response = true,
       .fixed_upstream_pullback = true,
       .local_active_response = false,
       .structure_response = false});
}

Eigen::MatrixXd ExactReducedHvp::apply_core_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions) {
  return exact_operator_.apply_reduced_batch(
      reduced_directions,
      {.direct_core_response = true,
       .fixed_upstream_pullback = true,
       .local_active_response = false,
       .structure_response = false});
}

Eigen::VectorXd ExactReducedHvp::apply_outer(
    const Eigen::VectorXd& reduced_direction) {
  return exact_operator_.apply_reduced(
      reduced_direction,
      {.direct_core_response = false,
       .fixed_upstream_pullback = false,
       .local_active_response = true,
       .structure_response = true});
}

Eigen::MatrixXd ExactReducedHvp::apply_outer_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions) {
  return exact_operator_.apply_reduced_batch(
      reduced_directions,
      {.direct_core_response = false,
       .fixed_upstream_pullback = false,
       .local_active_response = true,
       .structure_response = true});
}

bool ExactReducedHvp::supports_analytic_core_model()
    const noexcept {
  return exact_operator_.supports_analytic_core_model();
}

ExactHvpOperator::Diagnostics ExactReducedHvp::diagnostics()
    const {
  return exact_operator_.diagnostics();
}

ResponseCorrectedHvp::ResponseCorrectedHvp(ExactReducedHvp* exact_hvp)
    : exact_hvp_(exact_hvp) {
  if (exact_hvp_ == nullptr) {
    throw std::invalid_argument("exact HVP must not be null");
  }
}

Eigen::VectorXd ResponseCorrectedHvp::apply(
    const Eigen::VectorXd& reduced_direction) {
  return exact_hvp_->apply_core(reduced_direction) +
      response_model_.apply(reduced_direction);
}

Eigen::MatrixXd ResponseCorrectedHvp::apply_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions) {
  return exact_hvp_->apply_core_batch(reduced_directions) +
      response_model_.apply_batch(reduced_directions);
}

Eigen::VectorXd ResponseCorrectedHvp::exact_outer_response(
    const Eigen::VectorXd& reduced_direction) {
  return exact_hvp_->apply_outer(reduced_direction);
}

Eigen::VectorXd ResponseCorrectedHvp::modeled_outer_response(
    const Eigen::VectorXd& reduced_direction) const {
  return response_model_.apply(reduced_direction);
}

bool ResponseCorrectedHvp::add_outer_sample(
    const Eigen::VectorXd& reduced_direction,
    const Eigen::VectorXd& exact_outer_response) {
  return response_model_.add(reduced_direction, exact_outer_response);
}

Eigen::Index ResponseCorrectedHvp::response_rank() const noexcept {
  return response_model_.rank();
}

std::string build_hvp_error(const ExactReducedHvp& hvp) {
  const auto info = hvp.diagnostics();
  std::ostringstream message;
  message << "analytic HVP is unavailable"
          << ": supports_analytic_core_model="
          << bool_name(info.supports_analytic_core_model)
          << " outer_response_enabled="
          << bool_name(info.outer_response_enabled)
          << " has_same_spin_matrix_form="
          << bool_name(info.has_same_spin_matrix_form)
          << " has_opposite_spin_matrix_form="
          << bool_name(info.has_opposite_spin_matrix_form)
          << " n_selected_states=" << info.n_selected_states
          << " n_active_orbitals=" << info.n_active_orbitals
          << " n_blocks=" << info.n_blocks;
  return message.str();
}

}  // namespace xmvb::vb
