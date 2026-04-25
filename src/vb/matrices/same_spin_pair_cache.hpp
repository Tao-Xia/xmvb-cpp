#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <vector>

#include "vb/matrices/full_determinant_pair_evaluator.hpp"

namespace xmvb::vb {

/**
 * @brief Maps each spin determinant in the full space onto a unique occupied string.
 */
struct SpinDeterminantReuseTable {
  std::vector<std::vector<int>> unique_determinants;
  std::vector<int> determinant_to_unique_id;
};

/**
 * @brief Controls which opposite-spin payloads are materialized in the cache.
 *
 * The sparse packed-pair coefficients are always stored. Callers can disable
 * `projected_pair_values` when a streamed matrix-form path only needs sparse
 * projections and can rebuild selected packed-pair rows on demand.
 */
struct SameSpinPairCacheBuildOptions {
  bool materialize_projected_pair_values = true;
};

/**
 * @brief Reusable ordered same-spin determinant kernels for one active-space evaluation.
 */
struct SameSpinPairCacheContext {
  SpinDeterminantReuseTable alpha_reuse_table;
  SpinDeterminantReuseTable beta_reuse_table;
  std::vector<SpinDeterminantPairEvaluation> alpha_pair_cache;
  // When alpha/beta unique determinant spaces are identical, the ordered
  // beta-beta kernel is identical to the alpha-alpha kernel after the beta
  // determinant ids are expressed in the shared basis. In that case
  // `beta_pair_cache_ref()` aliases `alpha_pair_cache` and this storage
  // remains empty.
  std::vector<SpinDeterminantPairEvaluation> beta_pair_cache;
  bool beta_reuses_alpha_pair_cache = false;
  bool close_shell_diagonal_reuses_same_spin_pair_cache = false;

  // Optional scalar opposite-spin cache for legacy pair-by-pair callers.
  // The matrix-form forward/backward paths consume the per-spin projected
  // payloads inside `alpha_pair_cache` / `beta_pair_cache` directly and do not
  // require this `N_alpha^2 N_beta^2` table to be materialized.
  std::vector<double> opposite_spin_cache;
  std::vector<bool> opposite_spin_cache_computed;
  int n_unique_alpha = 0;
  int n_unique_beta = 0;

  std::size_t cached_same_spin_evaluation_count = 0;
  std::size_t estimated_cache_bytes = 0;
  bool use_same_spin_pair_cache = false;

  bool enabled() const {
    return use_same_spin_pair_cache;
  }

  bool close_shell_reuses_same_spin_pair_cache() const {
    return close_shell_diagonal_reuses_same_spin_pair_cache;
  }

  bool shares_same_spin_pair_cache_between_spins() const {
    return beta_reuses_alpha_pair_cache;
  }

  const std::vector<SpinDeterminantPairEvaluation>& alpha_pair_cache_ref() const {
    return alpha_pair_cache;
  }

  const std::vector<SpinDeterminantPairEvaluation>& beta_pair_cache_ref() const {
    return beta_reuses_alpha_pair_cache ? alpha_pair_cache : beta_pair_cache;
  }

  std::vector<SpinDeterminantPairEvaluation>* mutable_alpha_pair_cache_ref() {
    return &alpha_pair_cache;
  }

  std::vector<SpinDeterminantPairEvaluation>* mutable_beta_pair_cache_ref() {
    return beta_reuses_alpha_pair_cache ? &alpha_pair_cache : &beta_pair_cache;
  }

  // Compute the flattened `(alpha_L, alpha_R, beta_L, beta_R)` storage slot.
  std::size_t opposite_spin_cache_index(
      int unique_alpha_L, int unique_alpha_R,
      int unique_beta_L, int unique_beta_R) const {
    return unique_alpha_L * n_unique_alpha * n_unique_alpha * n_unique_beta +
           unique_alpha_R * n_unique_alpha * n_unique_beta +
           unique_beta_L * n_unique_beta +
           unique_beta_R;
  }
};

/**
 * @brief Compresses repeated same-spin occupied strings into a unique table.
 */
SpinDeterminantReuseTable build_spin_determinant_reuse_table(
    const std::vector<std::vector<int>>& spin_determinants);

/**
 * @brief Returns the dense row-major storage slot for one ordered unique pair.
 */
std::size_t ordered_spin_pair_storage_index(
    int left_index,
    int right_index,
    int n_unique_determinants);

/**
 * @brief Overestimates the memory footprint of one ordered same-spin cache table.
 */
std::size_t estimate_same_spin_pair_cache_bytes(
    const std::vector<std::vector<int>>& unique_spin_determinants,
    int n_orbitals);

/**
 * @brief Builds ordered alpha/beta same-spin caches for one determinant expansion.
 */
SameSpinPairCacheContext build_same_spin_pair_cache_context(
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    const FullDeterminantPairEvaluator& pair_evaluator,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act);

SameSpinPairCacheContext build_same_spin_pair_cache_context(
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    const FullDeterminantPairEvaluator& pair_evaluator,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act,
    SameSpinPairCacheBuildOptions build_options);

/**
 * @brief Builds ordered alpha/beta same-spin caches from packed or RI active ERIs.
 */
SameSpinPairCacheContext build_same_spin_pair_cache_context(
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    const FullDeterminantPairEvaluator& pair_evaluator,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result);

SameSpinPairCacheContext build_same_spin_pair_cache_context(
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    const FullDeterminantPairEvaluator& pair_evaluator,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    SameSpinPairCacheBuildOptions build_options);

/**
 * @brief Populates cached same-spin `\phi` payloads on one ordered unique-spin cache.
 *
 * The matrix-form same-spin backward needs, for each ordered unique-spin pair,
 * the accepted-point scalar
 *
 * `\phi = \phi_{1e} + \phi_{2e}`
 *
 * together with its derivative with respect to the inverse overlap submatrix.
 * This routine computes those quantities once per ordered unique-spin pair and
 * stores them inside `SpinDeterminantPairEvaluation` so later matrix-form
 * contractions can reuse the payload without revisiting the active-space
 * kernels on every determinant-pair adjoint.
 */
void populate_same_spin_phi_cache(
    SameSpinPairCacheContext* same_spin_pair_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result);

/**
 * @brief Evaluates one full determinant pair, optionally via cached same-spin kernels.
 */
FullDeterminantPairEvaluation evaluate_full_determinant_pair_with_optional_same_spin_cache(
    const SameSpinPairCacheContext* same_spin_pair_cache,
    const FullDeterminantPairEvaluator& pair_evaluator,
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    int determinant_index_left,
    int determinant_index_right,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act,
    bool retain_spin_pair_evaluations = true);

/**
 * @brief Evaluates one full determinant pair from packed or RI active ERIs.
 */
FullDeterminantPairEvaluation evaluate_full_determinant_pair_with_optional_same_spin_cache(
    const SameSpinPairCacheContext* same_spin_pair_cache,
    const FullDeterminantPairEvaluator& pair_evaluator,
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    int determinant_index_left,
    int determinant_index_right,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    bool retain_spin_pair_evaluations = true);

}  // namespace xmvb::vb
