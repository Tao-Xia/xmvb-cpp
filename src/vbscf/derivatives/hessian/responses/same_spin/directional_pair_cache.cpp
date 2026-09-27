#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"

#include "vbscf/derivatives/hessian/responses/same_spin/pair_response_internal.hpp"

namespace xmvb::vb {

SameSpinDirectionalPairCache build_same_spin_directional_pair_cache(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    int n_active_orbitals,
    const ActiveSpaceIntegralDirectionView& direction,
    const Eigen::MatrixXd* accepted_active_one_electron,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors,
    const Eigen::MatrixXd* directional_ri_active_pair_factors) {
  SameSpinDirectionalPairCache result;
  result.close_shell_same_spin =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();
  result.alpha = detail::build_directional_pair_scalar_matrices(
      same_spin_pair_cache.alpha_reuse_table.unique_determinants,
      same_spin_pair_cache.alpha_pair_cache_ref(),
      static_cast<int>(
          same_spin_pair_cache.alpha_reuse_table.unique_determinants.size()),
      n_active_orbitals,
      direction,
      accepted_active_one_electron,
      accepted_ri_active_pair_factors,
      directional_ri_active_pair_factors);
  if (!result.close_shell_same_spin) {
    result.beta = detail::build_directional_pair_scalar_matrices(
        same_spin_pair_cache.beta_reuse_table.unique_determinants,
        same_spin_pair_cache.beta_pair_cache_ref(),
        static_cast<int>(
            same_spin_pair_cache.beta_reuse_table.unique_determinants.size()),
        n_active_orbitals,
        direction,
        accepted_active_one_electron,
        accepted_ri_active_pair_factors,
        directional_ri_active_pair_factors);
  }
  return result;
}

std::vector<SameSpinDirectionalPairCache>
build_same_spin_directional_pair_cache_batch(
    const SameSpinPairCacheContext& same_spin_pair_cache,
    int n_active_orbitals,
    const std::vector<ActiveSpaceIntegralDirectionView>& directions,
    const Eigen::MatrixXd* accepted_active_one_electron,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors,
    const std::vector<Eigen::MatrixXd>* directional_ri_active_pair_factors) {
  std::vector<SameSpinDirectionalPairCache> results(directions.size());
  const bool close_shell =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();
  auto alpha = detail::build_directional_pair_scalar_matrices_batch(
      same_spin_pair_cache.alpha_reuse_table.unique_determinants,
      same_spin_pair_cache.alpha_pair_cache_ref(),
      static_cast<int>(
          same_spin_pair_cache.alpha_reuse_table.unique_determinants.size()),
      n_active_orbitals,
      directions,
      accepted_active_one_electron,
      accepted_ri_active_pair_factors,
      directional_ri_active_pair_factors);
  std::vector<SameSpinDirectionalScalarMatrices> beta;
  if (!close_shell) {
    beta = detail::build_directional_pair_scalar_matrices_batch(
        same_spin_pair_cache.beta_reuse_table.unique_determinants,
        same_spin_pair_cache.beta_pair_cache_ref(),
        static_cast<int>(
            same_spin_pair_cache.beta_reuse_table.unique_determinants.size()),
        n_active_orbitals,
        directions,
        accepted_active_one_electron,
        accepted_ri_active_pair_factors,
        directional_ri_active_pair_factors);
  }
  for (std::size_t direction = 0; direction < directions.size(); ++direction) {
    results[direction].close_shell_same_spin = close_shell;
    results[direction].alpha = std::move(alpha[direction]);
    if (!close_shell) {
      results[direction].beta = std::move(beta[direction]);
    }
  }
  return results;
}

}  // namespace xmvb::vb
