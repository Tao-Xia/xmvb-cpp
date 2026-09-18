#include "vbscf/optimization/objective/reduced_hvp.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace xmvb::vb {
namespace {

const char* bool_name(bool value) {
  return value ? "true" : "false";
}

}  // namespace

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
    const OrbitalChart& current_space,
    double response_relative_residual_tolerance)
    : exact_operator_(std::make_shared<ExactHvpOperator>(
          objective.second_order_context(),
          &objective.input(),
          SparseParameterLayout(
              objective.input().orbital_preparation_input),
          &current_space)),
      response_relative_residual_tolerance_(
          response_relative_residual_tolerance) {
  if (!std::isfinite(response_relative_residual_tolerance_) ||
      response_relative_residual_tolerance_ < 0.0 ||
      response_relative_residual_tolerance_ >= 1.0) {
    throw std::invalid_argument(
        "reduced HVP response tolerance must be finite in [0, 1)");
  }
}

Eigen::VectorXd ExactReducedHvp::apply(
    const Eigen::VectorXd& reduced_direction) {
  ++core_direction_count_;
  ++outer_response_direction_count_;
  return exact_operator_->apply_reduced(
      reduced_direction,
      {.direct_core_response = true,
       .fixed_upstream_pullback = true,
       .local_active_response = true,
       .structure_response = true,
       .response_relative_residual_tolerance =
           response_relative_residual_tolerance_});
}

Eigen::MatrixXd ExactReducedHvp::apply_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions) {
  const std::size_t n_directions =
      static_cast<std::size_t>(reduced_directions.cols());
  core_direction_count_ += n_directions;
  outer_response_direction_count_ += n_directions;
  return exact_operator_->apply_reduced_batch(
      reduced_directions,
      {.direct_core_response = true,
       .fixed_upstream_pullback = true,
       .local_active_response = true,
       .structure_response = true,
       .response_relative_residual_tolerance =
           response_relative_residual_tolerance_});
}

Eigen::VectorXd ExactReducedHvp::apply_core(
    const Eigen::VectorXd& reduced_direction) {
  ++core_direction_count_;
  return exact_operator_->apply_reduced(
      reduced_direction,
      {.direct_core_response = true,
       .fixed_upstream_pullback = true,
       .local_active_response = false,
       .structure_response = false});
}

Eigen::MatrixXd ExactReducedHvp::apply_core_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions) {
  core_direction_count_ +=
      static_cast<std::size_t>(reduced_directions.cols());
  return exact_operator_->apply_reduced_batch(
      reduced_directions,
      {.direct_core_response = true,
       .fixed_upstream_pullback = true,
       .local_active_response = false,
       .structure_response = false});
}

Eigen::VectorXd ExactReducedHvp::apply_outer(
    const Eigen::VectorXd& reduced_direction) {
  ++outer_response_direction_count_;
  return exact_operator_->apply_reduced(
      reduced_direction,
      {.direct_core_response = false,
       .fixed_upstream_pullback = false,
       .local_active_response = true,
       .structure_response = true});
}

std::size_t ExactReducedHvp::core_direction_count() const noexcept {
  return core_direction_count_;
}

std::size_t ExactReducedHvp::outer_response_direction_count() const noexcept {
  return outer_response_direction_count_;
}

bool ExactReducedHvp::supports_analytic_core_model()
    const noexcept {
  return exact_operator_->supports_analytic_core_model();
}

ExactHvpOperator::Diagnostics ExactReducedHvp::diagnostics()
    const {
  return exact_operator_->diagnostics();
}

std::shared_ptr<const ExactHvpOperator>
ExactReducedHvp::exact_operator() const noexcept {
  return exact_operator_;
}

CoreReducedHvp::CoreReducedHvp(ExactReducedHvp* exact_hvp)
    : exact_hvp_(exact_hvp) {
  if (exact_hvp_ == nullptr) {
    throw std::invalid_argument("core HVP requires an exact HVP operator");
  }
}

Eigen::VectorXd CoreReducedHvp::apply(
    const Eigen::VectorXd& reduced_direction) {
  return exact_hvp_->apply_core(reduced_direction);
}

Eigen::MatrixXd CoreReducedHvp::apply_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions) {
  return exact_hvp_->apply_core_batch(reduced_directions);
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
          << " has_opposite_spin_pair_graph="
          << bool_name(info.has_opposite_spin_pair_graph)
          << " n_selected_states=" << info.n_selected_states
          << " n_active_orbitals=" << info.n_active_orbitals
          << " n_blocks=" << info.n_blocks;
  return message.str();
}

bool outer_response_scale_is_affordable(
    const ExactHvpOperator::Diagnostics& diagnostics) {
  return diagnostics.estimated_outer_string_contraction_work > 0.0 &&
      diagnostics.estimated_core_pair_contraction_work > 0.0 &&
      diagnostics.estimated_outer_string_contraction_work <=
          diagnostics.estimated_core_pair_contraction_work;
}

}  // namespace xmvb::vb
