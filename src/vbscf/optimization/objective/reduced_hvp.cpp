#include "vbscf/optimization/objective/reduced_hvp.hpp"

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

bool ExactReducedHvp::supports_analytic_core_model()
    const noexcept {
  return exact_operator_.supports_analytic_core_model();
}

ExactHvpOperator::Diagnostics ExactReducedHvp::diagnostics()
    const {
  return exact_operator_.diagnostics();
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
