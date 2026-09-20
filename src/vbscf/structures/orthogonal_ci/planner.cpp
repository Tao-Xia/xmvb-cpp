#include "vbscf/structures/orthogonal_ci/planner.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace xmvb::vb {
namespace {

std::size_t binomial(int n, int k) {
  if (n < 0 || k < 0 || k > n) {
    return 0;
  }
  k = std::min(k, n - k);
  std::size_t value = 1;
  for (int index = 1; index <= k; ++index) {
    const std::size_t numerator = static_cast<std::size_t>(n - k + index);
    if (value > std::numeric_limits<std::size_t>::max() / numerator) {
      throw std::overflow_error("fixed-spin determinant count overflow");
    }
    value = value * numerator / static_cast<std::size_t>(index);
  }
  return value;
}

struct SpinSpaceAnalysis {
  bool complete = false;
  int n_electrons = 0;
  std::size_t dimension = 0;
};

SpinSpaceAnalysis analyze_spin_space(
    const std::vector<std::vector<int>>& strings,
    int n_active_orbitals) {
  SpinSpaceAnalysis result;
  if (strings.empty() || n_active_orbitals <= 0 || n_active_orbitals > 63) {
    return result;
  }
  result.n_electrons = static_cast<int>(strings.front().size());
  result.dimension = binomial(n_active_orbitals, result.n_electrons);
  if (strings.size() != result.dimension) {
    return result;
  }

  std::unordered_set<std::uint64_t> masks;
  masks.reserve(strings.size());
  for (const auto& occupied : strings) {
    if (static_cast<int>(occupied.size()) != result.n_electrons) {
      return result;
    }
    std::uint64_t mask = 0;
    int previous = -1;
    for (const int orbital : occupied) {
      if (orbital <= previous || orbital < 0 || orbital >= n_active_orbitals) {
        return result;
      }
      mask |= std::uint64_t{1} << orbital;
      previous = orbital;
    }
    if (!masks.insert(mask).second) {
      return result;
    }
  }
  result.complete = masks.size() == result.dimension;
  return result;
}

std::size_t same_spin_connections(int n_orbitals, int n_electrons) {
  const std::size_t occupied = static_cast<std::size_t>(n_electrons);
  const std::size_t virtuals =
      static_cast<std::size_t>(n_orbitals - n_electrons);
  return 1 + occupied * virtuals +
      binomial(n_electrons, 2) * binomial(n_orbitals - n_electrons, 2);
}

std::size_t single_excitation_count(int n_orbitals, int n_electrons) {
  return static_cast<std::size_t>(n_electrons) *
      static_cast<std::size_t>(n_orbitals - n_electrons);
}

long double exterior_transform_flops(
    int n_orbitals,
    int n_electrons,
    std::size_t spin_dimension,
    std::size_t partner_dimension,
    int block_width) {
  if (n_electrons == 0 || n_electrons == n_orbitals) {
    return 2.0L * spin_dimension * partner_dimension * block_width;
  }
  const long double shear_count =
      static_cast<long double>(n_orbitals) * (n_orbitals - 1) / 2.0L;
  const long double determinant_pairs = static_cast<long double>(
      binomial(n_orbitals - 2, n_electrons - 1));
  const long double entries_per_shear =
      determinant_pairs * partner_dimension * block_width;
  // Each shear is one multiply-add. Forward and adjoint transforms are both
  // required. Diagonal factors are also applied in both directions.
  return 4.0L * shear_count * entries_per_shear +
      2.0L * spin_dimension * partner_dimension * block_width;
}

}  // namespace

DirectCiActionPlan plan_orthogonal_direct_ci_action(
    const std::vector<std::vector<int>>& alpha_strings,
    const std::vector<std::vector<int>>& beta_strings,
    int n_active_orbitals,
    int block_width,
    int n_structures,
    std::size_t n_structure_expansion_terms) {
  if (n_active_orbitals <= 0 || block_width <= 0 || n_structures < 0) {
    throw std::invalid_argument(
        "direct-CI planner requires valid orbital, block, and structure "
        "dimensions");
  }
  const SpinSpaceAnalysis alpha =
      analyze_spin_space(alpha_strings, n_active_orbitals);
  const SpinSpaceAnalysis beta =
      analyze_spin_space(beta_strings, n_active_orbitals);

  DirectCiActionPlan plan;
  plan.alpha_space_complete = alpha.complete;
  plan.beta_space_complete = beta.complete;
  plan.n_active_orbitals = n_active_orbitals;
  plan.n_alpha_electrons = alpha.n_electrons;
  plan.n_beta_electrons = beta.n_electrons;
  plan.n_alpha_strings = alpha_strings.size();
  plan.n_beta_strings = beta_strings.size();
  plan.n_structures = static_cast<std::size_t>(n_structures);
  plan.n_structure_expansion_terms = n_structure_expansion_terms;
  if (!plan.complete()) {
    return plan;
  }

  plan.alpha_same_spin_connections =
      same_spin_connections(n_active_orbitals, alpha.n_electrons);
  plan.beta_same_spin_connections =
      same_spin_connections(n_active_orbitals, beta.n_electrons);
  plan.opposite_spin_connections =
      single_excitation_count(n_active_orbitals, alpha.n_electrons) *
      single_excitation_count(n_active_orbitals, beta.n_electrons);

  const long double n_alpha = static_cast<long double>(alpha.dimension);
  const long double n_beta = static_cast<long double>(beta.dimension);
  const long double width = static_cast<long double>(block_width);
  // Five dense GEMMs: two beta-side images and three alpha-side images.
  plan.kronecker_flops = width *
      (4.0L * n_alpha * n_beta * n_beta +
       6.0L * n_alpha * n_alpha * n_beta);

  // A target-driven Slater--Condon action aggregates diagonal and single
  // opposite-spin contractions into each nonzero Hamiltonian connection.
  const std::size_t full_connections =
      plan.alpha_same_spin_connections +
      plan.beta_same_spin_connections - 1 +
      plan.opposite_spin_connections;
  plan.direct_ci_flops =
      2.0L * n_alpha * n_beta * width * full_connections;
  plan.exterior_transform_flops =
      exterior_transform_flops(
          n_active_orbitals,
          alpha.n_electrons,
          alpha.dimension,
          beta.dimension,
          block_width) +
      exterior_transform_flops(
          n_active_orbitals,
          beta.n_electrons,
          beta.dimension,
          alpha.dimension,
          block_width);

  if (n_structures > 0) {
    const long double n_product = n_alpha * n_beta;
    const long double n_structure =
        static_cast<long double>(n_structures);
    const long double expansion_terms =
        static_cast<long double>(n_structure_expansion_terms);

    // One structure-to-product scatter and two product-to-structure gathers.
    // Each sparse coefficient application is one multiply-add.
    plan.structure_scatter_gather_flops =
        6.0L * expansion_terms * width;

    // Applying materialized H and S consists of two dense matrix-block
    // products. The live-value model includes both retained matrices and both
    // output blocks. The direct-CI lower bound includes the simultaneous
    // coefficient/Hamiltonian product blocks and the two structure outputs;
    // persistent connection graphs can only strengthen materialization's
    // memory advantage.
    plan.materialized_action_flops =
        4.0L * n_structure * n_structure * width;
    plan.materialized_action_live_values =
        2.0L * n_structure * n_structure +
        2.0L * n_structure * width;
    plan.direct_ci_action_live_values =
        2.0L * n_product * width +
        2.0L * n_structure * width;
  }
  return plan;
}

}  // namespace xmvb::vb
