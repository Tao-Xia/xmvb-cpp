#include "vbscf/derivatives/hessian/responses/same_spin_backward.hpp"

#include "vbscf/derivatives/hessian/responses/same_spin_pair_response_internal.hpp"

namespace xmvb::vb {

SameSpinDirectionalPairCache build_same_spin_directional_pair_cache(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    int n_active_orbitals,
    const ActiveSpaceIntegralDirectionView& direction) {
  SameSpinDirectionalPairCache result;
  result.close_shell_same_spin =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();
  result.alpha = detail::build_directional_pair_scalar_matrices(
      same_spin_pair_cache.alpha_reuse_table.unique_determinants,
      same_spin_pair_cache.alpha_pair_cache_ref(),
      static_cast<int>(
          same_spin_pair_cache.alpha_reuse_table.unique_determinants.size()),
      n_active_orbitals,
      direction);
  if (!result.close_shell_same_spin) {
    result.beta = detail::build_directional_pair_scalar_matrices(
        same_spin_pair_cache.beta_reuse_table.unique_determinants,
        same_spin_pair_cache.beta_pair_cache_ref(),
        static_cast<int>(
            same_spin_pair_cache.beta_reuse_table.unique_determinants.size()),
        n_active_orbitals,
        direction);
  }
  return result;
}

}  // namespace xmvb::vb
