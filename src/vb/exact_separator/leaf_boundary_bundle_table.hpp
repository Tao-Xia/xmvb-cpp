#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "vb/exact_separator/bundle.hpp"
#include "vb/exact_separator/leaf_boundary_message.hpp"
#include "vb/exact_separator/leaf_coefficient_operator.hpp"

namespace xmvb::vb::exact_separator {

struct OneLeafSpinStatePairIndex {
  int root_state_index = -1;
  int leaf_state_index = -1;
};

/**
 * @brief Dense one-leaf table of exact one-spin boundary bundles.
 *
 * Each table entry stores one exact `BoundarySpinBundle` for one unique
 * root/leaf one-spin state pair. The bundle is built directly from the
 * exported exact one-leaf boundary-sector message, so the table keeps the
 * canonical dense sector basis instead of collapsing immediately to overlap /
 * cofactor aggregates.
 */
struct IndexedOneLeafBoundarySpinBundleTable {
  int root_state_count = 0;
  int leaf_state_count = 0;
  std::vector<OneLeafSpinStatePairIndex> state_pairs;
  std::vector<BoundarySpinBundle> bundles;
  std::unordered_map<std::uint64_t, int> bundle_index_by_state_pair;
  std::uint64_t total_state_count = 0;
  std::uint64_t total_sector_count = 0;

  int index(int root_state_index, int leaf_state_index) const;
};

/**
 * @brief Collects the exact one-spin state pairs actually touched by the sparse
 * root/leaf coefficient operators.
 *
 * The current one-leaf experimental path should only materialize the state
 * pairs that appear in at least one `root_entry × leaf_entry` contribution.
 * Building the full Cartesian product of unique root and leaf states is a
 * large structural waste on sparse coefficient operators.
 */
std::vector<OneLeafSpinStatePairIndex>
collect_referenced_one_leaf_spin_state_pairs(
    const ComponentSpinCoefficientOperator& root_operator,
    const ComponentSpinCoefficientOperator& leaf_operator,
    bool alpha_channel);

/**
 * @brief Fused one-leaf Hamiltonian totals contracted from boundary bundles.
 *
 * The fields mirror the one-leaf production Hamiltonian decomposition. The
 * difference from the removed aggregate path is that the per-state data now
 * stay in canonical bundle form until this final contraction.
 */
struct OneLeafBoundaryBundleContractionResult {
  double overlap = 0.0;
  double one_electron = 0.0;
  double same_spin_alpha_two_electron = 0.0;
  double same_spin_beta_two_electron = 0.0;
  double opposite_spin_two_electron = 0.0;
};

/**
 * @brief Builds exact one-leaf one-spin bundles from exported sector messages.
 *
 * For each unique root/leaf one-spin state pair, this routine materializes the
 * exact public `OneLeafBoundarySpinMessage` and reindexes it to the canonical
 * dense `BoundarySpinBundle` storage. No direct full-state aggregate is built.
 */
IndexedOneLeafBoundarySpinBundleTable
build_indexed_one_leaf_boundary_spin_bundle_table(
    const std::vector<SpinPairStateKey>& root_states,
    const std::vector<SpinPairStateKey>& leaf_states,
    const std::vector<OneLeafSpinStatePairIndex>& state_pairs,
    const std::vector<double>& support_overlap_storage,
    int support_size,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations);

/**
 * @brief Contracts one-leaf Hamiltonian channels directly from bundle tables.
 *
 * The scalar overlap / one-electron / same-spin channels are read from the
 * dense bundle carrier with `bundle_channels.cpp`. The opposite-spin channel
 * reconstructs exact support-space first-cofactor matrices from the bundle
 * degree-1 data, then reuses the established direct mixed-spin contraction.
 */
OneLeafBoundaryBundleContractionResult
contract_one_leaf_boundary_bundle_channels(
    const ComponentSpinCoefficientOperator& root_operator,
    const ComponentSpinCoefficientOperator& leaf_operator,
    const IndexedOneLeafBoundarySpinBundleTable& alpha_table,
    const IndexedOneLeafBoundarySpinBundleTable& beta_table,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size);

}  // namespace xmvb::vb::exact_separator
