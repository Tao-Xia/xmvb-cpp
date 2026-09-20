#pragma once

#include <cstddef>
#include <vector>

namespace xmvb::vb {

/**
 * @brief Operation-count model for an orthogonalized direct-CI action.
 *
 * Counts include one Hamiltonian/overlap block action after transformation to
 * and from a complete fixed-spin determinant basis. They exclude the common
 * sparse structure-to-spin-product scatter and gather.
 */
struct DirectCiActionPlan {
  bool alpha_space_complete = false;
  bool beta_space_complete = false;
  int n_active_orbitals = 0;
  int n_alpha_electrons = 0;
  int n_beta_electrons = 0;
  std::size_t n_alpha_strings = 0;
  std::size_t n_beta_strings = 0;
  std::size_t alpha_same_spin_connections = 0;
  std::size_t beta_same_spin_connections = 0;
  std::size_t opposite_spin_connections = 0;
  long double kronecker_flops = 0.0L;
  long double direct_ci_flops = 0.0L;
  long double exterior_transform_flops = 0.0L;
  std::size_t n_structures = 0;
  std::size_t n_structure_expansion_terms = 0;
  long double structure_scatter_gather_flops = 0.0L;
  long double materialized_action_flops = 0.0L;
  long double direct_ci_action_live_values = 0.0L;
  long double materialized_action_live_values = 0.0L;

  bool complete() const noexcept {
    return alpha_space_complete && beta_space_complete;
  }

  long double total_direct_ci_flops() const noexcept {
    return direct_ci_flops + exterior_transform_flops;
  }

  bool favors_direct_ci() const noexcept {
    return complete() && total_direct_ci_flops() < kronecker_flops;
  }

  /**
   * @brief Whether dense structure H/S dominates one direct-CI block action.
   *
   * The comparison is deliberately Pareto based: materialization is preferred
   * only when it needs no more live scalar values and no more action FLOPs.
   * This avoids a machine- or molecule-specific conversion between memory and
   * work while preventing a small structure problem from being expanded into
   * a much larger complete spin-product workspace.
   */
  bool materialized_structure_action_dominates() const noexcept {
    return complete() && n_structures != 0 &&
        materialized_action_live_values <= direct_ci_action_live_values &&
        materialized_action_flops <=
            total_direct_ci_flops() + structure_scatter_gather_flops;
  }

  /** @brief Selects direct CI for Davidson after considering all exact forms. */
  bool favors_direct_ci_for_davidson() const noexcept {
    return favors_direct_ci() &&
        !materialized_structure_action_dominates();
  }
};

/**
 * @brief Builds a molecule-independent FLOP plan for one Davidson block.
 *
 * Direct CI is exact only when both unique-string sets are complete fixed-spin
 * Fock spaces. Completeness is verified from the occupied-orbital lists, not
 * inferred from their counts alone.
 */
DirectCiActionPlan plan_orthogonal_direct_ci_action(
    const std::vector<std::vector<int>>& alpha_strings,
    const std::vector<std::vector<int>>& beta_strings,
    int n_active_orbitals,
    int block_width,
    int n_structures = 0,
    std::size_t n_structure_expansion_terms = 0);

}  // namespace xmvb::vb
