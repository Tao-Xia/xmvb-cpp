#pragma once

#include "vb/orbital/active_space_two_electron_result.hpp"
#include "pfaffian_vbscf/types/eigen_types.hpp"

namespace xmvb::pfaffian_vbscf {

/**
 * @brief Active-space payload consumed by the Pfaffian-VBSCF forward builders.
 *
 * The spatial overlap `sso` and one-electron Hamiltonian `hho` use the same
 * column-major storage convention as the legacy VB builders. `ggo` reuses the
 * packed pair-of-pairs storage already handled by `PfTensorContractor`. When
 * `two_electron_representation == ResolutionOfIdentity`, the Pf closed-shell
 * kernel can consume `ri_active_pair_factors` directly without materializing
 * packed `ggo` on the hot path.
 */
struct PfActiveSpaceData {
  int n_active_orbitals = 0;
  int n_alpha = 0;
  int n_beta = 0;
  xmvb::vb::ActiveSpaceTwoElectronRepresentation two_electron_representation =
      xmvb::vb::ActiveSpaceTwoElectronRepresentation::PackedExact;
  int n_auxiliary_functions = 0;
  ScalarBuffer sso;
  ScalarBuffer hho;
  ScalarBuffer ggo;
  ScalarBuffer ri_active_pair_factors;
};

}  // namespace xmvb::pfaffian_vbscf
