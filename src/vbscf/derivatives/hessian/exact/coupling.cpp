#include "vbscf/derivatives/hessian/exact/operator.hpp"

#include "vbscf/derivatives/hessian/exact/ao_one_electron_internal.hpp"
#include "vbscf/derivatives/hessian/exact/apply_internal.hpp"
#include "vbscf/derivatives/hessian/exact/state_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/active_space/integral_direction.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"
#include "vbscf/derivatives/hessian/responses/structure/directional.hpp"
#include "vbscf/integrals/ao/one_electron/ri_operator.hpp"

namespace xmvb::vb {
namespace {

SelectedStructureDirection copy_selected_images(
    SelectedStateDirectionalStructureImages images) {
  return SelectedStructureDirection{
      std::move(images.delta_hamiltonian_selected),
      std::move(images.delta_overlap_selected)};
}

bool same_weight(double left, double right) {
  const double scale = std::max({1.0, std::abs(left), std::abs(right)});
  return std::abs(left - right) <=
      64.0 * std::numeric_limits<double>::epsilon() * scale;
}

}  // namespace

SelectedStructureDirection
ExactHvpOperator::apply_selected_structure_direction(
    const Eigen::VectorXd& reduced_direction) const {
  return state_->apply_selected_structure_direction(reduced_direction);
}

std::vector<SelectedStructureDirection>
ExactHvpOperator::apply_selected_structure_direction_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions) const {
  std::vector<SelectedStructureDirection> images;
  images.reserve(static_cast<std::size_t>(reduced_directions.cols()));
  for (Eigen::Index column = 0;
       column < reduced_directions.cols();
       ++column) {
    images.push_back(
        state_->apply_selected_structure_direction(
            reduced_directions.col(column)));
  }
  return images;
}

std::vector<SelectedStructureResponse>
ExactHvpOperator::apply_selected_structure_response_batch(
    const std::vector<SelectedStructureResponse>& responses) const {
  return state_->apply_selected_structure_response_batch(responses);
}

const std::vector<double>&
ExactHvpOperator::selected_state_weights() const noexcept {
  return state_->selected_state_weights();
}

int ExactHvpOperator::n_structures() const noexcept {
  return state_->n_structures();
}

SelectedStructureDirection
ExactHvpOperator::State::apply_selected_structure_direction(
    const Eigen::VectorXd& reduced_direction) const {
  if (!supports_analytic_core_model()) {
    throw std::runtime_error(
        "orbital-to-structure action requires the analytic accepted-point context");
  }

  const int n_selected_states = static_cast<int>(
      accepted_point_context_->selected_state_indices.size());
  const int n_structures = accepted_point_context_->n_structures;
  if (reduced_direction.size() == 0 || !reduced_direction.allFinite() ||
      n_selected_states <= 0 || n_structures <= 0) {
    throw std::invalid_argument(
        "orbital-to-structure direction or accepted eigensystem is invalid");
  }
  const Eigen::VectorXd packed_direction =
      nonredundant_space_->expand_step(reduced_direction);
  if (packed_direction.norm() == 0.0) {
    return SelectedStructureDirection{
        Eigen::MatrixXd::Zero(n_structures, n_selected_states),
        Eigen::MatrixXd::Zero(n_structures, n_selected_states)};
  }

  const auto& orbital_input = current_input_->orbital_preparation_input;
  const int n_bf = orbital_input.n_basis_functions;
  const int n_active = orbital_input.n_active_orbitals;
  const PrecomputedDirection prepared_direction =
      prepare_direction(packed_direction);
  const OrbitalPreparationDirectionalResult& orbital_direction =
      prepared_direction.orbital_preparation_directional_result;

  // Bp needs only the forward AO-H1E derivative G[delta D]. The ordinary HVP
  // fuses this action with a reverse pullback, but carrying that adjoint work
  // into the explicit coupled block would nearly double this stage.
  if (accepted_ri_factorization_ != nullptr) {
    const std::vector<double> density_direction(
        orbital_direction.delta_inactive_density.data(),
        orbital_direction.delta_inactive_density.data() +
            orbital_direction.delta_inactive_density.size());
    ao_h1e_delta_h1e_workspace_ =
        apply_ao_effective_one_electron_ri_operator(
            density_direction,
            *accepted_ri_factorization_,
            n_bf,
            {.attempt_spectral_factorization = true});
  } else {
    detail::apply_forward_exact_ao_one_electron_response(
        orbital_direction.delta_inactive_density,
        current_input_->ao_integral_input,
        orbital_input,
        &ao_h1e_delta_h1e_workspace_);
  }
  const Eigen::Map<const Eigen::MatrixXd> delta_h1e(
      ao_h1e_delta_h1e_workspace_.data(), n_bf, n_bf);
  const Eigen::MatrixXd delta_h1e_times_active =
      delta_h1e * accepted_active_auxiliary_orbitals_;

  Eigen::VectorXd ri_delta_two_electron;
  const ActiveSpaceIntegralDirectionView integral_direction =
      build_active_integral_direction(
          orbital_direction,
          delta_h1e_times_active,
          nullptr,
          &ri_delta_two_electron,
          &outer_response_integral_direction_workspace_);

  const StructureAction* structure_action = outer_response_context()
      .selected_state_eigen_response_operator.structure_action;
  SameSpinDirectionalPairCache pair_cache;
  if (!structure_action->supports_integral_direction()) {
    pair_cache = build_same_spin_directional_pair_cache(
        accepted_point_context_->same_spin_pair_cache,
        n_active,
        integral_direction);
  }
  return copy_selected_images(
      build_selected_structure_direction(
          outer_response_context(),
          integral_direction,
          pair_cache));
}

std::vector<SelectedStructureResponse>
ExactHvpOperator::State::apply_selected_structure_response_batch(
    const std::vector<SelectedStructureResponse>& responses) const {
  if (responses.empty()) {
    throw std::invalid_argument(
        "selected-subspace response action requires at least one direction");
  }
  const int n_structures = accepted_point_context_->n_structures;
  const int n_states = static_cast<int>(
      accepted_point_context_->selected_state_indices.size());
  const auto& weights = accepted_point_context_->normalized_state_weights;
  if (n_structures <= 0 || n_states <= 0 ||
      weights.size() != static_cast<std::size_t>(n_states)) {
    throw std::logic_error(
        "accepted selected-subspace response dimensions are inconsistent");
  }

  Eigen::MatrixXd coefficient_block(
      n_structures,
      n_states * static_cast<int>(responses.size()));
  for (std::size_t direction = 0; direction < responses.size(); ++direction) {
    const SelectedStructureResponse& response = responses[direction];
    if (response.coefficients.rows() != n_structures ||
        response.coefficients.cols() != n_states ||
        response.multipliers.rows() != n_states ||
        response.multipliers.cols() != n_states ||
        !response.coefficients.allFinite() ||
        !response.multipliers.allFinite()) {
      throw std::invalid_argument(
          "selected-subspace response has incompatible dimensions or values");
    }
    for (int column = 0; column < n_states; ++column) {
      for (int row = 0; row < n_states; ++row) {
        if (response.multipliers(row, column) != 0.0 &&
            !same_weight(weights[row], weights[column])) {
          throw std::invalid_argument(
              "selected-subspace response multipliers may couple only equal-weight states");
        }
      }
    }
    coefficient_block.middleCols(
        static_cast<int>(direction) * n_states,
        n_states) = response.coefficients;
  }

  const auto& response_operator =
      outer_response_context().selected_state_eigen_response_operator;
  if (response_operator.structure_action == nullptr ||
      response_operator.selected_eigenvalues.size() != n_states ||
      response_operator.selected_eigenvectors.rows() != n_structures ||
      response_operator.selected_eigenvectors.cols() != n_states ||
      response_operator.overlap_selected.rows() != n_structures ||
      response_operator.overlap_selected.cols() != n_states) {
    throw std::logic_error(
        "accepted selected-subspace response action is unavailable");
  }
  const StructureActionResult action =
      response_operator.structure_action->apply(coefficient_block);

  std::vector<SelectedStructureResponse> images;
  images.reserve(responses.size());
  for (std::size_t direction = 0; direction < responses.size(); ++direction) {
    const int first = static_cast<int>(direction) * n_states;
    const auto h_z = action.hamiltonian.middleCols(first, n_states);
    const auto s_z = action.overlap.middleCols(first, n_states);
    SelectedStructureResponse image;
    image.coefficients =
        h_z - s_z * response_operator.selected_eigenvalues.asDiagonal() +
        response_operator.overlap_selected * responses[direction].multipliers;
    image.multipliers =
        response_operator.selected_eigenvectors.transpose() * s_z;
    images.push_back(std::move(image));
  }
  return images;
}

const std::vector<double>&
ExactHvpOperator::State::selected_state_weights() const noexcept {
  return accepted_point_context_->normalized_state_weights;
}

int ExactHvpOperator::State::n_structures() const noexcept {
  return accepted_point_context_->n_structures;
}

bool ExactHvpOperator::State::represents_accepted_point(
    const AcceptedPointContext& accepted_point) const noexcept {
  return accepted_point_context_.get() == &accepted_point;
}

}  // namespace xmvb::vb
