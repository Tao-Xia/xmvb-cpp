#include "vb/exact_separator/leaf_boundary_bundle_table.hpp"

#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vb/exact_separator/bundle_channels.hpp"
#include "vb/exact_separator/leaf_boundary_bundle.hpp"
#include "vb/exact_separator/spin_state_aggregate.hpp"

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

struct OneLeafBoundarySpinBundleChannels {
  double overlap = 0.0;
  double one_electron = 0.0;
  double same_spin_two_electron = 0.0;
  Eigen::MatrixXd first_cofactor;
};

std::uint64_t count_one_leaf_boundary_sectors(
    const OneLeafBoundarySpinMessage& message) {
  std::uint64_t total = 0;
  for (const auto& family : message.families) {
    total += static_cast<std::uint64_t>(family.sectors.size());
  }
  return total;
}

Eigen::MatrixXd build_support_first_cofactor_from_bundle(
    const BoundarySpinBundle& bundle,
    int support_size) {
  if (bundle.layout.support_size() != support_size) {
    throw std::invalid_argument("bundle support size is inconsistent");
  }

  Eigen::MatrixXd first_cofactor = Eigen::MatrixXd::Zero(support_size, support_size);
  for (int support_pair_index = 0;
       support_pair_index < bundle.layout.support_pair_count();
       ++support_pair_index) {
    int row_orbital = -1;
    int col_orbital = -1;
    decode_support_pair_index(
        support_pair_index,
        support_size,
        &row_orbital,
        &col_orbital);
    double value = 0.0;
    for (int flat_sector_index = 0;
         flat_sector_index < bundle.layout.total_sector_count();
         ++flat_sector_index) {
      value += bundle.degree1[xmvb::to_size(
          bundle.layout.flat_degree1_index(
              support_pair_index,
              flat_sector_index))];
    }
    first_cofactor(row_orbital, col_orbital) = value;
  }
  return first_cofactor;
}

std::vector<OneLeafBoundarySpinBundleChannels>
build_one_leaf_boundary_spin_bundle_channels(
    const IndexedOneLeafBoundarySpinBundleTable& table,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size) {
  std::vector<OneLeafBoundarySpinBundleChannels> channels;
  channels.reserve(table.bundles.size());
  for (const BoundarySpinBundle& bundle : table.bundles) {
    OneLeafBoundarySpinBundleChannels state_channels;
    state_channels.overlap = contract_boundary_spin_bundle_overlap(bundle);
    state_channels.one_electron =
        contract_boundary_spin_bundle_one_electron(
            bundle,
            support_one_electron_storage,
            support_size);
    state_channels.same_spin_two_electron =
        contract_boundary_spin_bundle_same_spin(
            bundle,
            packed_active_two_electron_integrals);
    state_channels.first_cofactor =
        build_support_first_cofactor_from_bundle(bundle, support_size);
    channels.push_back(std::move(state_channels));
  }
  return channels;
}

}  // namespace

int IndexedOneLeafBoundarySpinBundleTable::index(
    int root_state_index,
    int leaf_state_index) const {
  if (root_state_index < 0 || root_state_index >= root_state_count) {
    throw std::invalid_argument("root_state_index is out of range");
  }
  if (leaf_state_index < 0 || leaf_state_index >= leaf_state_count) {
    throw std::invalid_argument("leaf_state_index is out of range");
  }
  const auto iterator = bundle_index_by_state_pair.find(
      pack_state_pair_indices(root_state_index, leaf_state_index));
  if (iterator == bundle_index_by_state_pair.end()) {
    throw std::invalid_argument("requested one-leaf bundle state pair was not materialized");
  }
  return iterator->second;
}

std::vector<OneLeafSpinStatePairIndex>
collect_referenced_one_leaf_spin_state_pairs(
    const ComponentSpinCoefficientOperator& root_operator,
    const ComponentSpinCoefficientOperator& leaf_operator,
    bool alpha_channel) {
  std::unordered_set<std::uint64_t> visited_pairs;
  std::vector<OneLeafSpinStatePairIndex> state_pairs;
  visited_pairs.reserve(root_operator.entries.size() * leaf_operator.entries.size());
  state_pairs.reserve(root_operator.entries.size() * leaf_operator.entries.size());
  for (const auto& root_entry : root_operator.entries) {
    const int root_state_index =
        alpha_channel ? root_entry.alpha_state_index : root_entry.beta_state_index;
    for (const auto& leaf_entry : leaf_operator.entries) {
      const int leaf_state_index =
          alpha_channel ? leaf_entry.alpha_state_index : leaf_entry.beta_state_index;
      const std::uint64_t packed_key =
          pack_state_pair_indices(root_state_index, leaf_state_index);
      if (!visited_pairs.insert(packed_key).second) {
        continue;
      }
      state_pairs.push_back(OneLeafSpinStatePairIndex{
          .root_state_index = root_state_index,
          .leaf_state_index = leaf_state_index,
      });
    }
  }
  return state_pairs;
}

IndexedOneLeafBoundarySpinBundleTable
build_indexed_one_leaf_boundary_spin_bundle_table(
    const std::vector<SpinPairStateKey>& root_states,
    const std::vector<SpinPairStateKey>& leaf_states,
    const std::vector<OneLeafSpinStatePairIndex>& state_pairs,
    const std::vector<double>& support_overlap_storage,
    int support_size,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }
  if (support_size <= 0) {
    throw std::invalid_argument("support_size must be positive");
  }

  IndexedOneLeafBoundarySpinBundleTable table;
  table.root_state_count = static_cast<int>(root_states.size());
  table.leaf_state_count = static_cast<int>(leaf_states.size());
  table.state_pairs = state_pairs;
  table.bundles.reserve(table.state_pairs.size());
  table.bundle_index_by_state_pair.reserve(table.state_pairs.size());

  for (const OneLeafSpinStatePairIndex& state_pair : table.state_pairs) {
    if (state_pair.root_state_index < 0 ||
        state_pair.root_state_index >= table.root_state_count ||
        state_pair.leaf_state_index < 0 ||
        state_pair.leaf_state_index >= table.leaf_state_count) {
      throw std::invalid_argument("one-leaf bundle table state pair is out of range");
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
    table.total_sector_count += count_one_leaf_boundary_sectors(message);
    const int bundle_index = static_cast<int>(table.bundles.size());
    table.bundles.push_back(build_one_leaf_boundary_spin_bundle(message));
    table.bundle_index_by_state_pair.emplace(
        pack_state_pair_indices(
            state_pair.root_state_index,
            state_pair.leaf_state_index),
        bundle_index);
    ++table.total_state_count;
  }
  return table;
}

OneLeafBoundaryBundleContractionResult
contract_one_leaf_boundary_bundle_channels(
    const ComponentSpinCoefficientOperator& root_operator,
    const ComponentSpinCoefficientOperator& leaf_operator,
    const IndexedOneLeafBoundarySpinBundleTable& alpha_table,
    const IndexedOneLeafBoundarySpinBundleTable& beta_table,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size) {
  if (alpha_table.root_state_count != static_cast<int>(root_operator.alpha_states.size()) ||
      alpha_table.leaf_state_count != static_cast<int>(leaf_operator.alpha_states.size()) ||
      beta_table.root_state_count != static_cast<int>(root_operator.beta_states.size()) ||
      beta_table.leaf_state_count != static_cast<int>(leaf_operator.beta_states.size())) {
    throw std::invalid_argument(
        "boundary bundle contraction received inconsistent state dimensions");
  }

  const std::vector<OneLeafBoundarySpinBundleChannels> alpha_channels =
      build_one_leaf_boundary_spin_bundle_channels(
          alpha_table,
          support_one_electron_storage,
          packed_active_two_electron_integrals,
          support_size);
  const std::vector<OneLeafBoundarySpinBundleChannels> beta_channels =
      build_one_leaf_boundary_spin_bundle_channels(
          beta_table,
          support_one_electron_storage,
          packed_active_two_electron_integrals,
          support_size);

  std::vector<Eigen::MatrixXd> beta_potential_first_cofactors;
  beta_potential_first_cofactors.reserve(beta_channels.size());
  for (const auto& beta_channel : beta_channels) {
    beta_potential_first_cofactors.push_back(
        apply_opposite_spin_kernel_to_first_cofactor(
            beta_channel.first_cofactor,
            support_size,
            packed_active_two_electron_integrals));
  }

  OneLeafBoundaryBundleContractionResult result;
  for (const auto& root_entry : root_operator.entries) {
    for (const auto& leaf_entry : leaf_operator.entries) {
      const double coefficient =
          root_entry.coefficient * leaf_entry.coefficient;
      const int alpha_index = alpha_table.index(
          root_entry.alpha_state_index,
          leaf_entry.alpha_state_index);
      const int beta_index = beta_table.index(
          root_entry.beta_state_index,
          leaf_entry.beta_state_index);
      const OneLeafBoundarySpinBundleChannels& alpha_channel =
          alpha_channels[xmvb::to_size(alpha_index)];
      const OneLeafBoundarySpinBundleChannels& beta_channel =
          beta_channels[xmvb::to_size(beta_index)];

      result.overlap +=
          coefficient * alpha_channel.overlap * beta_channel.overlap;
      result.one_electron +=
          coefficient *
          (alpha_channel.one_electron * beta_channel.overlap +
           alpha_channel.overlap * beta_channel.one_electron);
      result.same_spin_alpha_two_electron +=
          coefficient *
          alpha_channel.same_spin_two_electron *
          beta_channel.overlap;
      result.same_spin_beta_two_electron +=
          coefficient *
          alpha_channel.overlap *
          beta_channel.same_spin_two_electron;
      result.opposite_spin_two_electron +=
          coefficient *
          contract_direct_opposite_spin_channel(
              alpha_channel.first_cofactor,
              beta_potential_first_cofactors[xmvb::to_size(beta_index)]);
    }
  }
  return result;
}

}  // namespace xmvb::vb::exact_separator
