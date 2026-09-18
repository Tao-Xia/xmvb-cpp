#include "vbscf/derivatives/hessian/exact/operator.hpp"

#include "vbscf/derivatives/hessian/exact/ao_one_electron_internal.hpp"
#include "vbscf/derivatives/hessian/exact/apply_internal.hpp"
#include "vbscf/derivatives/hessian/exact/state_internal.hpp"

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

}  // namespace xmvb::vb
