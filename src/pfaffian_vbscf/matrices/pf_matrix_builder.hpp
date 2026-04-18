#pragma once

#include <vector>

#include "pfaffian_vbscf/kernel/pf_kernel_cache.hpp"
#include "pfaffian_vbscf/types/pf_active_space_data.hpp"
#include "pfaffian_vbscf/types/pf_basis_data.hpp"
#include "pfaffian_vbscf/types/pf_pair_profile.hpp"

namespace xmvb::pfaffian_vbscf {

/**
 * @brief Full Pfaffian basis matrices used by the molecule-level tests.
 */
struct PfMatrixBuildResult {
  int n_states = 0;
  ScalarBuffer s;
  ScalarBuffer h;
  PfPairProfile pair_profile;
  std::vector<PfKernelCache> lower_triangle_pair_caches;
};

/**
 * @brief Builds overlap and Hamiltonian matrices between Pfaffian basis states.
 */
class PfMatrixBuilder {
public:
  /**
   * @brief Builds the basis overlap and active Hamiltonian matrices.
   *
   * @param basis Pfaffian basis states.
   * @param active_space Active-space integrals and electron counts.
   * @return PfMatrixBuildResult Column-major overlap and total Hamiltonian matrices.
   */
  PfMatrixBuildResult build(
      const PfBasisData& basis,
      const PfActiveSpaceData& active_space,
      bool store_lower_triangle_pair_caches = false) const;
};

}  // namespace xmvb::pfaffian_vbscf
