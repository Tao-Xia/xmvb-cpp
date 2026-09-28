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
  bool alpha_carrier_complete = false;
  bool beta_carrier_complete = false;
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
  /** Minimum coefficient workspace for one direct-CI action vector. */
  std::size_t minimum_action_workspace_bytes = 0;

  bool carrier_complete() const noexcept {
    return alpha_carrier_complete && beta_carrier_complete;
  }

  bool fits_action_workspace(std::size_t workspace_bytes) const noexcept {
    return carrier_complete() &&
        minimum_action_workspace_bytes <= workspace_bytes;
  }

  long double total_direct_ci_flops() const noexcept {
    return direct_ci_flops + exterior_transform_flops;
  }

  bool favors_direct_ci() const noexcept {
    return carrier_complete() && total_direct_ci_flops() < kronecker_flops;
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
    int block_width);

}  // namespace xmvb::vb
