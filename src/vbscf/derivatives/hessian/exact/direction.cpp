#include "vbscf/derivatives/hessian/exact/apply_internal.hpp"
#include "vbscf/derivatives/hessian/exact/state_internal.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/active_space/integral_direction.hpp"
#include "vbscf/derivatives/hessian/responses/orbital/preparation.hpp"
#include "vbscf/integrals/active/two_electron/response/ri.hpp"

namespace xmvb::vb {

ExactHvpOperator::State::PrecomputedDirection
ExactHvpOperator::State::prepare_direction(
    Eigen::VectorXd packed_direction) const {
  const auto& orbital_input = current_input_->orbital_preparation_input;
  const int n_inactive =
      (orbital_input.n_total_electrons - orbital_input.n_active_electrons) / 2;
  PrecomputedDirection result;
  result.packed_direction = std::move(packed_direction);
  result.dense_orbital_tangent_context =
      build_dense_orbital_tangent_context(
          orbital_input,
          parameter_view_,
          result.packed_direction,
          *accepted_orbital_preparation_cache_);
  result.orbital_preparation_directional_result =
      build_orbital_preparation_directional_result(
          orbital_input,
          result.dense_orbital_tangent_context,
          n_inactive,
          orbital_input.n_active_orbitals,
          *accepted_orbital_preparation_cache_);
  return result;
}

ActiveSpaceIntegralDirectionView
ExactHvpOperator::State::build_active_integral_direction(
    const OrbitalPreparationDirectionalResult& orbital_direction,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_h1e_times_active,
    const Eigen::VectorXd* precomputed_delta_packed_two_electron,
    Eigen::VectorXd* ri_delta_packed_two_electron,
    ActiveSpaceIntegralDirectionWorkspace* workspace) const {
  const Eigen::VectorXd* delta_packed_two_electron =
      precomputed_delta_packed_two_electron;
  if (delta_packed_two_electron == nullptr &&
      accepted_ri_two_electron_cache_.has_value()) {
    if (ri_delta_packed_two_electron == nullptr) {
      throw std::invalid_argument(
          "RI active-integral direction requires an output workspace");
    }
    const Eigen::MatrixXd delta_factors =
        compute_ri_active_pair_factor_directional_derivative(
            *accepted_ri_two_electron_cache_,
            orbital_direction.delta_active_auxiliary_orbitals);
    const std::vector<double> packed =
        compute_ri_packed_active_two_electron_integral_directional_derivative(
            *accepted_ri_two_electron_cache_,
            delta_factors);
    *ri_delta_packed_two_electron = Eigen::Map<const Eigen::VectorXd>(
        packed.data(), static_cast<Eigen::Index>(packed.size()));
    delta_packed_two_electron = ri_delta_packed_two_electron;
  }

  const auto& accepted_two_electron = accepted_point_context_
      ->prepared_active_space.active_space_two_electron_result;
  const ActiveSpaceIntegralDirectionContext integral_context{
      *current_input_,
      accepted_ri_two_electron_cache_.has_value()
          ? nullptr
          : &accepted_exact_two_electron_cache_,
      accepted_active_auxiliary_orbitals_,
      accepted_basis_overlap_times_active_auxiliary_orbitals_,
      accepted_ao_effective_one_electron_times_active_auxiliary_orbitals_,
      accepted_ao_effective_one_electron_transpose_times_active_auxiliary_orbitals_,
      accepted_two_electron.dense_active_coefficients};
  return xmvb::vb::build_active_space_integral_direction(
      integral_context,
      ActiveSpaceIntegralTangent{
          orbital_direction.delta_active_auxiliary_orbitals,
          orbital_direction.delta_active_auxiliary_orbitals,
          delta_h1e_times_active,
          delta_packed_two_electron},
      workspace);
}

}  // namespace xmvb::vb
