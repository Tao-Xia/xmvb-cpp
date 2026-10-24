#include "mcscf/ci/planner.hpp"
#include "mcscf/ci/strings.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace xmvb::mcscf {
namespace {

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
  result.dimension = string_count(n_active_orbitals, result.n_electrons);
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
      string_count(n_electrons, 2) * string_count(n_orbitals - n_electrons, 2);
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
      string_count(n_orbitals - 2, n_electrons - 1));
  const long double entries_per_shear =
      determinant_pairs * partner_dimension * block_width;
  // Each shear is one multiply-add. Forward and adjoint transforms are both
  // required. Diagonal factors are also applied in both directions.
  return 4.0L * shear_count * entries_per_shear +
      2.0L * spin_dimension * partner_dimension * block_width;
}

}  // namespace

namespace {
DirectCiActionPlan make_plan(const SpinSpaceAnalysis& alpha,const SpinSpaceAnalysis& beta,
    int n_active_orbitals,int block_width) {
  if (n_active_orbitals<=0 || block_width<=0)
    throw std::invalid_argument("direct-CI planner requires positive dimensions");
  DirectCiActionPlan plan;
  plan.alpha_carrier_complete = alpha.complete;
  plan.beta_carrier_complete = beta.complete;
  plan.n_active_orbitals = n_active_orbitals;
  plan.n_alpha_electrons = alpha.n_electrons;
  plan.n_beta_electrons = beta.n_electrons;
  plan.n_alpha_strings = alpha.dimension;
  plan.n_beta_strings = beta.dimension;
  if (!plan.carrier_complete()) {
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

  return plan;
}

} // namespace

DirectCiActionPlan plan_orthogonal_direct_ci_action(
    const std::vector<std::vector<int>>& alpha,const std::vector<std::vector<int>>& beta,
    int n,int width) {
  auto a=analyze_spin_space(alpha,n),b=analyze_spin_space(beta,n);
  a.dimension=alpha.size();b.dimension=beta.size();
  return make_plan(a,b,n,width);
}
DirectCiActionPlan plan_orthogonal_direct_ci_action(
    const SpinStrings& alpha,const SpinStrings& beta,int n,int width) {
  return make_plan({alpha.complete() && alpha.orbitals()==n,alpha.electrons(),static_cast<std::size_t>(alpha.size())},
      {beta.complete() && beta.orbitals()==n,beta.electrons(),static_cast<std::size_t>(beta.size())},n,width);
}
DirectCiActionPlan plan_orthogonal_direct_ci_action(
    const StringTable& alpha,const StringTable& beta,int n,int width) {
  if (n<1 || n>63 || alpha.empty() || beta.empty())
    return make_plan({}, {},n,width);
  return plan_orthogonal_direct_ci_action(alpha.basis(n),beta.basis(n),n,width);
}

} // namespace xmvb::mcscf
