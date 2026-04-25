#include "pfaffian_vbscf/matrices/pf_act_builder.hpp"

#include <stdexcept>

#include "pfaffian_vbscf/scf/pf_prepared_active_space.hpp"
#include "pfaffian_vbscf/types/pf_active_space_utils.hpp"
#include "vb/matrices/prepared_active_space_context.hpp"

namespace xmvb::pfaffian_vbscf {

PfPreparedActiveSpaceData PfActBuilder::prepare(
    const xmvb::vb::CppVbInput& input) const {
  const auto& structure_data = input.structure_data;
  if (input.orbital_preparation_input.n_active_orbitals <= 0) {
    throw std::invalid_argument("input.orbital_preparation_input.n_active_orbitals must be positive");
  }
  if (structure_data.alpha_det.empty() || structure_data.beta_det.empty()) {
    throw std::invalid_argument("determinant list must not be empty");
  }

  const auto prepared_active_space =
      prepare_pf_timed_active_space_context(input).prepared_active_space;

  PfPreparedActiveSpaceData result;
  PfActiveSpaceData& active_space = result.active_space;
  active_space.n_active_orbitals =
      input.orbital_preparation_input.n_active_orbitals;
  active_space.n_alpha =
      static_cast<int>(structure_data.alpha_det.front().size());
  active_space.n_beta =
      static_cast<int>(structure_data.beta_det.front().size());
  active_space.sso.assign(
      prepared_active_space.orbital_result.active_orbital_overlap_matrix.begin(),
      prepared_active_space.orbital_result.active_orbital_overlap_matrix.end());
  active_space.hho.assign(
      prepared_active_space.active_space_one_electron_result.h1e_act.data(),
      prepared_active_space.active_space_one_electron_result.h1e_act.data() +
          prepared_active_space.active_space_one_electron_result.h1e_act.size());
  active_space.two_electron_representation =
      prepared_active_space.active_space_two_electron_result.representation;
  active_space.n_auxiliary_functions =
      prepared_active_space.active_space_two_electron_result.n_auxiliary_functions;
  active_space.ggo.assign(
      prepared_active_space.active_space_two_electron_result
          .packed_active_two_electron_integrals.begin(),
      prepared_active_space.active_space_two_electron_result
          .packed_active_two_electron_integrals.end());
  active_space.ri_active_pair_factors =
      flatten_column_major_matrix(
          prepared_active_space.active_space_two_electron_result
              .ri_active_pair_factors);
  result.reference_energy = prepared_active_space.one_electron_reference_energy;
  return result;
}

PfActiveSpaceData PfActBuilder::build(const xmvb::vb::CppVbInput& input) const {
  return prepare(input).active_space;
}

}  // namespace xmvb::pfaffian_vbscf
