#include "vb/exact_separator/one_leaf_packed_bundle.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "vb/exact_separator/bundle_channels.hpp"
#include "vb/exact_separator/hamiltonian_bundle.hpp"
#include "vb/exact_separator/leaf_boundary_bundle.hpp"
#include "vb/exact_separator/leaf_boundary_bundle_table.hpp"

namespace xmvb::vb::exact_separator {

namespace {

std::uint64_t pack_state_pair_indices(
    int root_state_index,
    int leaf_state_index) {
  if (root_state_index < 0 || leaf_state_index < 0) {
    throw std::invalid_argument("state indices must be non-negative");
  }
  return
      (static_cast<std::uint64_t>(static_cast<std::uint32_t>(root_state_index)) << 32U) |
      static_cast<std::uint64_t>(static_cast<std::uint32_t>(leaf_state_index));
}

struct IndexedOneLeafSpinBundleCache {
  int root_state_count = 0;
  int leaf_state_count = 0;
  BoundarySpinBundleLayout layout;
  std::vector<BoundarySpinBundle> bundles;
  std::unordered_map<std::uint64_t, int> bundle_index_by_state_pair;

  int index(int root_state_index, int leaf_state_index) const {
    if (root_state_index < 0 || root_state_index >= root_state_count) {
      throw std::invalid_argument("root_state_index is out of range");
    }
    if (leaf_state_index < 0 || leaf_state_index >= leaf_state_count) {
      throw std::invalid_argument("leaf_state_index is out of range");
    }
    const auto iterator = bundle_index_by_state_pair.find(
        pack_state_pair_indices(root_state_index, leaf_state_index));
    if (iterator == bundle_index_by_state_pair.end()) {
      throw std::invalid_argument("requested one-leaf packed-bundle state pair was not cached");
    }
    return iterator->second;
  }
};

std::uint64_t count_message_sectors(const OneLeafBoundarySpinMessage& message) {
  std::uint64_t total = 0;
  for (const auto& family : message.families) {
    total += static_cast<std::uint64_t>(family.sectors.size());
  }
  return total;
}

IndexedOneLeafSpinBundleCache build_indexed_one_leaf_spin_bundle_cache(
    const std::vector<SpinPairStateKey>& root_states,
    const std::vector<SpinPairStateKey>& leaf_states,
    const std::vector<OneLeafSpinStatePairIndex>& state_pairs,
    const std::vector<double>& support_overlap_storage,
    int support_size,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* total_sector_count,
    std::uint64_t* subdeterminant_evaluations) {
  if (total_sector_count == nullptr || subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("bundle cache counters must not be null");
  }

  IndexedOneLeafSpinBundleCache cache;
  cache.root_state_count = static_cast<int>(root_states.size());
  cache.leaf_state_count = static_cast<int>(leaf_states.size());
  cache.bundles.reserve(state_pairs.size());
  cache.bundle_index_by_state_pair.reserve(state_pairs.size());

  bool have_layout = false;
  for (const OneLeafSpinStatePairIndex& state_pair : state_pairs) {
    if (state_pair.root_state_index < 0 ||
        state_pair.root_state_index >= cache.root_state_count ||
        state_pair.leaf_state_index < 0 ||
        state_pair.leaf_state_index >= cache.leaf_state_count) {
      throw std::invalid_argument("one-leaf packed bundle cache state pair is out of range");
    }
    const SpinPairStateKey& root_state =
        root_states[xmvb::to_size(state_pair.root_state_index)];
    const SpinPairStateKey& leaf_state =
        leaf_states[xmvb::to_size(state_pair.leaf_state_index)];
    const OneLeafBoundarySpinMessage message =
        build_one_leaf_spin_boundary_message(
            root_state.left_occ,
            leaf_state.left_occ,
            root_state.right_occ,
            leaf_state.right_occ,
            support_overlap_storage,
            support_size,
            overlap_resolver,
            subdeterminant_evaluations);
    *total_sector_count += count_message_sectors(message);

    BoundarySpinBundle bundle = build_one_leaf_boundary_spin_bundle(message);
    if (!have_layout) {
      cache.layout = bundle.layout;
      have_layout = true;
    } else if (!have_same_boundary_spin_bundle_layout(
                   cache.layout,
                   bundle.layout)) {
      throw std::runtime_error(
          "one-leaf packed bundle cache encountered inconsistent bundle layouts");
    }
    const int bundle_index = static_cast<int>(cache.bundles.size());
    cache.bundles.push_back(std::move(bundle));
    cache.bundle_index_by_state_pair.emplace(
        pack_state_pair_indices(
            state_pair.root_state_index,
            state_pair.leaf_state_index),
        bundle_index);
  }
  return cache;
}

}  // namespace

PackedOneLeafBundle build_one_leaf_packed_bundle(
    const ComponentSpinCoefficientOperator& root_operator,
    const ComponentSpinCoefficientOperator& leaf_operator,
    const std::vector<double>& support_overlap_storage,
    int support_size,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  // Builds the exact one-leaf Hamiltonian message directly on the dense
  // boundary-bundle basis.
  //
  // The current experimental path already has exact one-spin bundle builders
  // for every referenced root/leaf state pair. The structural step here is to
  // cache only the state pairs actually touched by the sparse root/leaf
  // coefficient operators, then accumulate one packed Hamiltonian bundle:
  //
  // - alpha channel weighted by spectator beta overlap,
  // - beta channel weighted by spectator alpha overlap,
  // - mixed alpha/beta degree-1 second moment.
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }
  if (support_size <= 0) {
    throw std::invalid_argument("support_size must be positive");
  }

  PackedOneLeafBundle packed;
  packed.support_size = support_size;
  std::uint64_t alpha_total_sector_count = 0;
  std::uint64_t beta_total_sector_count = 0;
  const std::vector<OneLeafSpinStatePairIndex> alpha_state_pairs =
      collect_referenced_one_leaf_spin_state_pairs(
          root_operator,
          leaf_operator,
          true);
  const std::vector<OneLeafSpinStatePairIndex> beta_state_pairs =
      collect_referenced_one_leaf_spin_state_pairs(
          root_operator,
          leaf_operator,
          false);
  const IndexedOneLeafSpinBundleCache alpha_cache =
      build_indexed_one_leaf_spin_bundle_cache(
          root_operator.alpha_states,
          leaf_operator.alpha_states,
          alpha_state_pairs,
          support_overlap_storage,
          support_size,
          overlap_resolver,
          &alpha_total_sector_count,
          subdeterminant_evaluations);
  const IndexedOneLeafSpinBundleCache beta_cache =
      build_indexed_one_leaf_spin_bundle_cache(
          root_operator.beta_states,
          leaf_operator.beta_states,
          beta_state_pairs,
          support_overlap_storage,
          support_size,
          overlap_resolver,
          &beta_total_sector_count,
          subdeterminant_evaluations);
  packed.alpha_state_pair_count = alpha_state_pairs.size();
  packed.beta_state_pair_count = beta_state_pairs.size();
  packed.total_sector_count =
      static_cast<std::uint64_t>(alpha_cache.layout.total_sector_count()) +
      static_cast<std::uint64_t>(beta_cache.layout.total_sector_count());
  packed.alpha = make_zero_boundary_spin_bundle(alpha_cache.layout);
  packed.beta = make_zero_boundary_spin_bundle(beta_cache.layout);
  packed.mixed = make_zero_boundary_mixed_bundle(
      alpha_cache.layout.support_pair_count() *
          alpha_cache.layout.total_sector_count(),
      beta_cache.layout.support_pair_count() *
          beta_cache.layout.total_sector_count());

  for (const auto& root_entry : root_operator.entries) {
    for (const auto& leaf_entry : leaf_operator.entries) {
      const double coefficient =
          root_entry.coefficient * leaf_entry.coefficient;
      if (std::abs(coefficient) <= 1.0e-15) {
        continue;
      }
      const int alpha_index = alpha_cache.index(
          root_entry.alpha_state_index,
          leaf_entry.alpha_state_index);
      const int beta_index = beta_cache.index(
          root_entry.beta_state_index,
          leaf_entry.beta_state_index);
      const BoundarySpinBundle& alpha_bundle =
          alpha_cache.bundles[xmvb::to_size(alpha_index)];
      const BoundarySpinBundle& beta_bundle =
          beta_cache.bundles[xmvb::to_size(beta_index)];
      accumulate_weighted_spin_pair_into_hamiltonian_channels(
          alpha_bundle,
          beta_bundle,
          coefficient,
          &packed.overlap,
          &packed.alpha,
          &packed.beta,
          &packed.mixed);
    }
  }

  return packed;
}

PackedOneLeafBundleContractionResult contract_one_leaf_packed_bundle(
    const PackedOneLeafBundle& packed,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals) {
  // Reads the exact one-leaf Hamiltonian channels from the packed dense
  // Hamiltonian bundle.
  //
  // `packed.alpha` and `packed.beta` already include the matching spectator
  // overlap weights, so the final one-electron and same-spin channels are
  // direct contractions of those weighted bundles. The mixed channel stores
  // the full degree-1 second moment and contracts directly against the packed
  // opposite-spin ERIs.
  PackedOneLeafBundleContractionResult result;
  result.overlap = packed.overlap;
  result.one_electron =
      contract_boundary_spin_bundle_one_electron(
          packed.alpha,
          support_one_electron_storage,
          packed.support_size) +
      contract_boundary_spin_bundle_one_electron(
          packed.beta,
          support_one_electron_storage,
          packed.support_size);
  result.same_spin_alpha_two_electron =
      contract_boundary_spin_bundle_same_spin(
          packed.alpha,
          packed_active_two_electron_integrals);
  result.same_spin_beta_two_electron =
      contract_boundary_spin_bundle_same_spin(
          packed.beta,
          packed_active_two_electron_integrals);
  result.opposite_spin_two_electron =
      contract_boundary_mixed_bundle_opposite_spin(
          packed.mixed,
          packed.alpha.layout.total_sector_count(),
          packed.beta.layout.total_sector_count(),
          packed.support_size,
          packed_active_two_electron_integrals);
  return result;
}

}  // namespace xmvb::vb::exact_separator
