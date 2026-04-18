#include "vb/exact_separator/component_tree.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/LU>

#include "vb/exact_separator/bundle.hpp"
#include "vb/exact_separator/bundle_channels.hpp"
#include "vb/exact_separator/boundary_sector.hpp"
#include "vb/exact_separator/component_terms.hpp"
#include "vb/exact_separator/one_leaf_packed_bundle.hpp"
#include "vb/exact_separator/spin_state_aggregate.hpp"
#include "vb/exact_separator/two_electron.hpp"
#include "vb/matrices/determinant_hamiltonian_resolver.hpp"
#include "vb/matrices/full_determinant_pair_evaluator.hpp"
#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace xmvb::vb::exact_separator {

namespace {


enum class PartialSide {
  LeftFrontier,
  RightComplement,
};

struct SpinPayloadCacheKey {
  std::vector<int> left_occ;
  std::vector<int> right_occ;
  int internal_row_begin = 0;
  int internal_col_begin = 0;
  bool zero_selected_root_block = false;
  PartialSide side = PartialSide::LeftFrontier;
};

bool operator==(
    const SpinPayloadCacheKey& left,
    const SpinPayloadCacheKey& right) {
  return left.left_occ == right.left_occ &&
      left.right_occ == right.right_occ &&
      left.internal_row_begin == right.internal_row_begin &&
      left.internal_col_begin == right.internal_col_begin &&
      left.zero_selected_root_block == right.zero_selected_root_block &&
      left.side == right.side;
}

struct SpinPayloadCacheKeyHasher {
  std::size_t operator()(const SpinPayloadCacheKey& key) const;
};

struct SpinDeletionKey {
  std::vector<int> row_labels;
  std::vector<int> col_labels;
};

bool operator<(const SpinDeletionKey& left, const SpinDeletionKey& right) {
  return std::tie(left.row_labels, left.col_labels) <
      std::tie(right.row_labels, right.col_labels);
}

bool operator==(const SpinDeletionKey& left, const SpinDeletionKey& right) {
  return left.row_labels == right.row_labels &&
      left.col_labels == right.col_labels;
}

bool operator!=(const SpinDeletionKey& left, const SpinDeletionKey& right) {
  return !(left == right);
}

struct SpinDeletionPayload {
  int row_count = 0;
  int col_count = 0;
  std::vector<SpinDeletionKey> basis_keys;
  std::map<SpinDeletionKey, double> sectors;
};

struct JointDeletionKey {
  SpinDeletionKey alpha_key;
  SpinDeletionKey beta_key;
};

bool operator<(const JointDeletionKey& left, const JointDeletionKey& right) {
  return std::tie(left.alpha_key, left.beta_key) <
      std::tie(right.alpha_key, right.beta_key);
}

bool operator==(const JointDeletionKey& left, const JointDeletionKey& right) {
  return left.alpha_key == right.alpha_key &&
      left.beta_key == right.beta_key;
}

bool operator!=(const JointDeletionKey& left, const JointDeletionKey& right) {
  return !(left == right);
}

struct JointDeletionPayload {
  // Coupled alpha/beta deleted-minor sectors.
  //
  // This is the minimal exact state needed by the current rooted-tree driver.
  // Keeping alpha and beta together preserves the term-level correlation that
  // was lost in the earlier incorrect "sum alpha and beta separately, then
  // multiply" design.
  int alpha_row_count = 0;
  int alpha_col_count = 0;
  int beta_row_count = 0;
  int beta_col_count = 0;
  std::vector<JointDeletionKey> basis_keys;
  std::map<JointDeletionKey, double> sectors;
};

struct ComponentTreeLeafMessage {
  std::uint32_t alpha_row_mask = 0;
  std::uint32_t alpha_col_mask = 0;
  std::uint32_t beta_row_mask = 0;
  std::uint32_t beta_col_mask = 0;
  JointDeletionPayload payload;
};

struct SameSpinDeletedSectorValue {
  SpinDeletionKey key;
  double value = 0.0;
};

struct HamiltonianMixedDeletedSectorValue {
  JointDeletionKey key;
  double value = 0.0;
};

struct HamiltonianBoundaryPayload {
  // Exact rooted-tree Hamiltonian boundary-message bundle.
  //
  // This compactly stores the exact payload required by the production root
  // closure:
  // - `overlap` is the closed `(0,0;0,0)` sector,
  // - `alpha_sectors` carry the full alpha degree-<=2 deleted-minor families
  //   scaled by closed beta overlap,
  // - `beta_sectors` are the symmetric beta-active bundle,
  // - `mixed_sectors` carry the exact mixed alpha/beta degree-1 moments used
  //   by opposite-spin contraction.
  int alpha_row_count = 0;
  int alpha_col_count = 0;
  int beta_row_count = 0;
  int beta_col_count = 0;
  bool has_overlap_basis = false;
  std::vector<SpinDeletionKey> alpha_basis_keys;
  std::vector<SpinDeletionKey> beta_basis_keys;
  std::vector<JointDeletionKey> mixed_basis_keys;
  double overlap = 0.0;
  std::vector<SameSpinDeletedSectorValue> alpha_sectors;
  std::vector<SameSpinDeletedSectorValue> beta_sectors;
  std::vector<HamiltonianMixedDeletedSectorValue> mixed_sectors;
};

struct HamiltonianEntryDensePayload {
  int alpha_row_count = 0;
  int alpha_col_count = 0;
  int beta_row_count = 0;
  int beta_col_count = 0;
  int support_size = 0;
  double overlap = 0.0;
  std::vector<double> alpha_degree1;
  std::vector<double> alpha_degree2;
  std::vector<double> beta_degree1;
  std::vector<double> beta_degree2;
  std::vector<double> mixed_degree1;
};

struct DenseBoundaryHamiltonianSectorEntry {
  int alpha_flat_sector_index = -1;
  int beta_flat_sector_index = -1;
  int flat_index = -1;
};

struct DenseBoundaryHamiltonianMessage {
  BoundarySpinBundleLayout alpha_layout;
  BoundarySpinBundleLayout beta_layout;
  int support_size = 0;
  std::vector<HamiltonianEntryDensePayload> payload_values;
  std::vector<DenseBoundaryHamiltonianSectorEntry> nonzero_entries;

  int flat_index(int alpha_flat_sector_index, int beta_flat_sector_index) const {
    return
        alpha_flat_sector_index * beta_layout.total_sector_count() +
        beta_flat_sector_index;
  }
};

SpinDeletionKey make_spin_key(
    std::vector<int> row_labels = {},
    std::vector<int> col_labels = {});
int sector_degree(const SpinDeletionKey& key);
SpinDeletionKey canonicalize_spin_key(
    const SpinDeletionKey& key,
    int* parity);
void finalize_dense_boundary_hamiltonian_message(
    DenseBoundaryHamiltonianMessage* message);

struct ComponentSpinSizes {
  int left_alpha = 0;
  int left_beta = 0;
  int right_alpha = 0;
  int right_beta = 0;
};

struct SubtreeExpansion {
  std::vector<int> preorder_nodes;
  std::vector<ComponentData> preorder_components;
  ComponentSpinSizes spin_sizes;
};

struct ChildSpinSizeVectors {
  std::vector<int> left_alpha;
  std::vector<int> left_beta;
  std::vector<int> right_alpha;
  std::vector<int> right_beta;
};

struct SubtreeMessageCacheKey {
  int node = -1;
  std::vector<int> left_parent_alpha_occ;
  std::vector<int> left_parent_beta_occ;
  std::vector<int> right_parent_alpha_occ;
  std::vector<int> right_parent_beta_occ;
  bool preserve_structural_zero_messages = false;
};

ChildSpinSizeVectors build_child_spin_size_vectors(
    const std::vector<int>& children,
    const std::vector<SubtreeExpansion>& subtree_expansions) {
  ChildSpinSizeVectors result;
  result.left_alpha.reserve(children.size());
  result.left_beta.reserve(children.size());
  result.right_alpha.reserve(children.size());
  result.right_beta.reserve(children.size());
  for (const int child : children) {
    const ComponentSpinSizes& child_sizes =
        subtree_expansions[xmvb::to_size(child)].spin_sizes;
    result.left_alpha.push_back(child_sizes.left_alpha);
    result.left_beta.push_back(child_sizes.left_beta);
    result.right_alpha.push_back(child_sizes.right_alpha);
    result.right_beta.push_back(child_sizes.right_beta);
  }
  return result;
}

bool is_one_leaf_star_tree(const ComponentTree& tree) {
  const std::vector<int>& root_children =
      tree.children[xmvb::to_size(tree.root_index)];
  return tree.components.size() == 2U &&
      root_children.size() == 1U &&
      tree.children[xmvb::to_size(root_children.front())].empty();
}

ComponentTreeOppositeSpinResult build_opposite_spin_result(
    const ComponentTreeHamiltonianResult& full_result) {
  ComponentTreeOppositeSpinResult result;
  result.overlap = full_result.overlap;
  result.one_electron = full_result.one_electron;
  result.opposite_spin_two_electron = full_result.opposite_spin_two_electron;
  result.subtree_message_state_count = full_result.subtree_message_state_count;
  result.subtree_term_pair_count = full_result.subtree_term_pair_count;
  result.subdeterminant_evaluations = full_result.subdeterminant_evaluations;
  result.dp_transition_count = full_result.dp_transition_count;
  return result;
}

void assign_opposite_spin_debug_result(
    const ComponentTreeHamiltonianResult& full_result,
    ComponentTreeOppositeSpinDebugResult* result) {
  if (result == nullptr) {
    throw std::invalid_argument("opposite-spin debug result must not be null");
  }
  result->overlap = full_result.overlap;
  result->one_electron = full_result.one_electron;
  result->opposite_spin_two_electron = full_result.opposite_spin_two_electron;
  result->subtree_message_state_count = full_result.subtree_message_state_count;
  result->subtree_term_pair_count = full_result.subtree_term_pair_count;
  result->subdeterminant_evaluations = full_result.subdeterminant_evaluations;
  result->dp_transition_count = full_result.dp_transition_count;
}

ComponentTreeSameSpinResult build_same_spin_result(
    const ComponentTreeHamiltonianResult& full_result) {
  ComponentTreeSameSpinResult result;
  result.overlap = full_result.overlap;
  result.one_electron = full_result.one_electron;
  result.same_spin_alpha_two_electron =
      full_result.same_spin_alpha_two_electron;
  result.same_spin_beta_two_electron =
      full_result.same_spin_beta_two_electron;
  result.subtree_message_state_count = full_result.subtree_message_state_count;
  result.subtree_term_pair_count = full_result.subtree_term_pair_count;
  result.subdeterminant_evaluations = full_result.subdeterminant_evaluations;
  result.dp_transition_count = full_result.dp_transition_count;
  return result;
}

bool operator==(
    const SubtreeMessageCacheKey& left,
    const SubtreeMessageCacheKey& right) {
  return left.node == right.node &&
      left.left_parent_alpha_occ == right.left_parent_alpha_occ &&
      left.left_parent_beta_occ == right.left_parent_beta_occ &&
      left.right_parent_alpha_occ == right.right_parent_alpha_occ &&
      left.right_parent_beta_occ == right.right_parent_beta_occ &&
      left.preserve_structural_zero_messages ==
          right.preserve_structural_zero_messages;
}

struct SubtreeMessageCacheKeyHasher {
  std::size_t operator()(const SubtreeMessageCacheKey& key) const;
};

struct BoundaryHamiltonianMessage;

struct OverlapGradientReverseContext {
  const ComponentTree& tree;
  const std::vector<SubtreeExpansion>& subtree_expansions;
  const std::vector<double>& support_overlap_storage;
  const std::vector<double>& support_one_electron_storage;
  const std::vector<double>& packed_active_two_electron_integrals;
  int support_size = 0;
  const DeterminantOverlapResolver& overlap_resolver;
  std::uint64_t* subtree_term_pair_count = nullptr;
  std::uint64_t* subtree_message_state_count = nullptr;
  std::uint64_t* subdeterminant_evaluations = nullptr;
  std::uint64_t* dp_transition_count = nullptr;
  std::unordered_map<SpinPayloadCacheKey,
                     SpinDeletionPayload,
                     SpinPayloadCacheKeyHasher>* spin_payload_cache = nullptr;
  std::unordered_map<SubtreeMessageCacheKey,
                     std::vector<ComponentTreeLeafMessage>,
                     SubtreeMessageCacheKeyHasher>* subtree_message_cache = nullptr;
  std::unordered_map<SubtreeMessageCacheKey,
                     BoundaryHamiltonianMessage,
                     SubtreeMessageCacheKeyHasher>* hamiltonian_message_cache = nullptr;
  std::vector<double>* active_orbital_overlap_gradient = nullptr;
};

struct SpinOverlapCacheKey {
  std::vector<int> left_occ;
  std::vector<int> right_occ;
  int interface_row_count = 0;
  int interface_col_count = 0;
  bool zero_selected_interface_block = false;
};

bool operator==(
    const SpinOverlapCacheKey& left,
    const SpinOverlapCacheKey& right) {
  return left.left_occ == right.left_occ &&
      left.right_occ == right.right_occ &&
      left.interface_row_count == right.interface_row_count &&
      left.interface_col_count == right.interface_col_count &&
      left.zero_selected_interface_block == right.zero_selected_interface_block;
}

struct SpinOverlapCacheKeyHasher {
  std::size_t operator()(const SpinOverlapCacheKey& key) const;
};

struct SpinScalarChannelValues {
  double overlap = 0.0;
  double one_electron = 0.0;
};

struct BoundaryScalarSectorEntry {
  int alpha_sector_index = -1;
  int beta_sector_index = -1;
  double overlap = 0.0;
  double one_electron = 0.0;
};

struct BoundaryScalarMessage {
  BoundarySectorIndexer alpha_indexer;
  BoundarySectorIndexer beta_indexer;
  std::vector<double> overlap_values;
  std::vector<double> one_electron_values;
  std::vector<BoundaryScalarSectorEntry> nonzero_entries;

  int flat_index(int alpha_sector_index, int beta_sector_index) const {
    return alpha_sector_index * beta_indexer.sector_count() + beta_sector_index;
  }
};

struct BoundaryHamiltonianSectorEntry {
  int alpha_flat_sector_index = -1;
  int beta_flat_sector_index = -1;
  int flat_index = -1;
};

struct BoundaryHamiltonianMessage {
  BoundarySpinBundleLayout alpha_layout;
  BoundarySpinBundleLayout beta_layout;
  std::vector<HamiltonianBoundaryPayload> payload_values;
  std::vector<BoundaryHamiltonianSectorEntry> nonzero_entries;

  int flat_index(int alpha_flat_sector_index, int beta_flat_sector_index) const {
    return
        alpha_flat_sector_index * beta_layout.total_sector_count() +
        beta_flat_sector_index;
  }
};

struct FrontierHamiltonianMaskKey {
  std::uint32_t alpha_row_mask = 0U;
  std::uint32_t alpha_col_mask = 0U;
  std::uint32_t beta_row_mask = 0U;
  std::uint32_t beta_col_mask = 0U;
};

bool operator<(
    const FrontierHamiltonianMaskKey& left,
    const FrontierHamiltonianMaskKey& right) {
  return std::tie(
             left.alpha_row_mask,
             left.alpha_col_mask,
             left.beta_row_mask,
             left.beta_col_mask) <
      std::tie(
             right.alpha_row_mask,
             right.alpha_col_mask,
             right.beta_row_mask,
             right.beta_col_mask);
}

bool operator==(
    const FrontierHamiltonianMaskKey& left,
    const FrontierHamiltonianMaskKey& right) {
  return left.alpha_row_mask == right.alpha_row_mask &&
      left.alpha_col_mask == right.alpha_col_mask &&
      left.beta_row_mask == right.beta_row_mask &&
      left.beta_col_mask == right.beta_col_mask;
}

struct FrontierHamiltonianEntry {
  FrontierHamiltonianMaskKey key;
  HamiltonianBoundaryPayload payload;
};

struct FrontierHamiltonianMessage {
  int alpha_row_count = 0;
  int alpha_col_count = 0;
  int beta_row_count = 0;
  int beta_col_count = 0;
  std::vector<FrontierHamiltonianEntry> entries;
};

struct DenseFrontierHamiltonianEntry {
  FrontierHamiltonianMaskKey key;
  HamiltonianEntryDensePayload payload;
};

struct DenseFrontierHamiltonianMessage {
  int alpha_row_count = 0;
  int alpha_col_count = 0;
  int beta_row_count = 0;
  int beta_col_count = 0;
  int support_size = 0;
  std::vector<DenseFrontierHamiltonianEntry> entries;
};

struct FrontierJointEntry {
  FrontierHamiltonianMaskKey key;
  JointDeletionPayload payload;
};

struct FrontierJointMessage {
  int alpha_row_count = 0;
  int alpha_col_count = 0;
  int beta_row_count = 0;
  int beta_col_count = 0;
  std::vector<FrontierJointEntry> entries;
};

const BoundarySector& boundary_layout_sector(
    const BoundarySpinBundleLayout& layout,
    int flat_sector_index) {
  const BoundaryFlatSectorIndex decoded =
      decode_boundary_flat_sector_index(layout, flat_sector_index);
  return layout.family(decoded.family_index).sector_indexer.sector(decoded.sector_index);
}

int count_mask_extraction_inversions(
    int n_root_occ,
    std::uint32_t prefix_mask,
    std::uint32_t extracted_mask) {
  const std::uint32_t full_mask =
      (n_root_occ == 0)
          ? 0U
          : ((static_cast<std::uint32_t>(1U) << n_root_occ) - 1U);
  const std::uint32_t trailing_mask =
      full_mask ^ (prefix_mask | extracted_mask);
  int inversion_count = 0;
  for (int root_position = 0; root_position < n_root_occ; ++root_position) {
    const std::uint32_t bit =
        static_cast<std::uint32_t>(1U) << root_position;
    if ((extracted_mask & bit) == 0U) {
      continue;
    }
    const std::uint32_t lower_mask =
        (root_position == 0)
            ? 0U
            : (bit - 1U);
    inversion_count +=
        __builtin_popcount(trailing_mask & lower_mask);
  }
  return inversion_count;
}

int incremental_component_block_parity(
    int n_root_occ,
    std::uint32_t prefix_mask,
    std::uint32_t child_mask,
    int child_leaf_size) {
  const std::uint32_t full_mask =
      (n_root_occ == 0)
          ? 0U
          : ((static_cast<std::uint32_t>(1U) << n_root_occ) - 1U);
  const std::uint32_t trailing_mask =
      full_mask ^ (prefix_mask | child_mask);
  return
      count_mask_extraction_inversions(
          n_root_occ,
          prefix_mask,
          child_mask) +
      child_leaf_size * __builtin_popcount(trailing_mask);
}

constexpr double kHamiltonianDenseZeroTolerance = 1.0e-15;

int dense_antisym_support_pair_count(int support_size) {
  if (support_size < 0) {
    throw std::invalid_argument("support_size must be non-negative");
  }
  return support_size * (support_size - 1) / 2;
}

int dense_spin_degree_delta(int row_count, int col_count) {
  return row_count - col_count;
}

int dense_spin_degree1_size(
    int row_count,
    int col_count,
    int support_size) {
  const int delta = dense_spin_degree_delta(row_count, col_count);
  switch (delta) {
    case -1:
    case 1:
      return support_size;
    case 0:
      return support_size * support_size;
    default:
      return 0;
  }
}

int dense_spin_degree2_size(
    int row_count,
    int col_count,
    int support_size) {
  const int delta = dense_spin_degree_delta(row_count, col_count);
  const int pair_count = dense_antisym_support_pair_count(support_size);
  switch (delta) {
    case -2:
    case 2:
      return pair_count;
    case -1:
    case 1:
      return pair_count * support_size;
    case 0:
      return pair_count * pair_count;
    default:
      return 0;
  }
}

int dense_mixed_degree1_size(
    int alpha_row_count,
    int alpha_col_count,
    int beta_row_count,
    int beta_col_count,
    int support_size) {
  return
      dense_spin_degree1_size(
          alpha_row_count,
          alpha_col_count,
          support_size) *
      dense_spin_degree1_size(
          beta_row_count,
          beta_col_count,
          support_size);
}

int dense_mixed_degree1_flat_index(
    const HamiltonianEntryDensePayload& payload,
    int alpha_degree1_index,
    int beta_degree1_index) {
  const int alpha_degree1_size = dense_spin_degree1_size(
      payload.alpha_row_count,
      payload.alpha_col_count,
      payload.support_size);
  const int beta_degree1_size = dense_spin_degree1_size(
      payload.beta_row_count,
      payload.beta_col_count,
      payload.support_size);
  if (alpha_degree1_index < 0 ||
      alpha_degree1_index >= alpha_degree1_size ||
      beta_degree1_index < 0 ||
      beta_degree1_index >= beta_degree1_size) {
    throw std::out_of_range("dense mixed degree-1 index is out of range");
  }
  return beta_degree1_index * alpha_degree1_size + alpha_degree1_index;
}

HamiltonianEntryDensePayload make_zero_hamiltonian_entry_dense_payload(
    int alpha_row_count,
    int alpha_col_count,
    int beta_row_count,
    int beta_col_count,
    int support_size) {
  HamiltonianEntryDensePayload payload;
  payload.alpha_row_count = alpha_row_count;
  payload.alpha_col_count = alpha_col_count;
  payload.beta_row_count = beta_row_count;
  payload.beta_col_count = beta_col_count;
  payload.support_size = support_size;
  payload.alpha_degree1.assign(
      xmvb::to_size(dense_spin_degree1_size(
          alpha_row_count,
          alpha_col_count,
          support_size)),
      0.0);
  payload.alpha_degree2.assign(
      xmvb::to_size(dense_spin_degree2_size(
          alpha_row_count,
          alpha_col_count,
          support_size)),
      0.0);
  payload.beta_degree1.assign(
      xmvb::to_size(dense_spin_degree1_size(
          beta_row_count,
          beta_col_count,
          support_size)),
      0.0);
  payload.beta_degree2.assign(
      xmvb::to_size(dense_spin_degree2_size(
          beta_row_count,
          beta_col_count,
          support_size)),
      0.0);
  payload.mixed_degree1.assign(
      xmvb::to_size(dense_mixed_degree1_size(
          alpha_row_count,
          alpha_col_count,
          beta_row_count,
          beta_col_count,
          support_size)),
      0.0);
  return payload;
}

bool have_same_hamiltonian_entry_dense_payload_layout(
    const HamiltonianEntryDensePayload& left,
    const HamiltonianEntryDensePayload& right) {
  return left.alpha_row_count == right.alpha_row_count &&
      left.alpha_col_count == right.alpha_col_count &&
      left.beta_row_count == right.beta_row_count &&
      left.beta_col_count == right.beta_col_count &&
      left.support_size == right.support_size &&
      left.alpha_degree1.size() == right.alpha_degree1.size() &&
      left.alpha_degree2.size() == right.alpha_degree2.size() &&
      left.beta_degree1.size() == right.beta_degree1.size() &&
      left.beta_degree2.size() == right.beta_degree2.size() &&
      left.mixed_degree1.size() == right.mixed_degree1.size();
}

bool is_hamiltonian_entry_dense_payload_nonzero(
    const HamiltonianEntryDensePayload& payload) {
  if (std::abs(payload.overlap) > kHamiltonianDenseZeroTolerance) {
    return true;
  }
  auto has_nonzero = [](const std::vector<double>& values) {
    for (const double value : values) {
      if (std::abs(value) > kHamiltonianDenseZeroTolerance) {
        return true;
      }
    }
    return false;
  };
  return
      has_nonzero(payload.alpha_degree1) ||
      has_nonzero(payload.alpha_degree2) ||
      has_nonzero(payload.beta_degree1) ||
      has_nonzero(payload.beta_degree2) ||
      has_nonzero(payload.mixed_degree1);
}

void add_scaled_dense_values(
    const std::vector<double>& source,
    double scale,
    std::vector<double>* destination) {
  if (destination == nullptr) {
    throw std::invalid_argument("dense destination must not be null");
  }
  if (source.size() != destination->size()) {
    throw std::invalid_argument("dense source/destination sizes do not match");
  }
  if (std::abs(scale) <= kHamiltonianDenseZeroTolerance) {
    return;
  }
  for (std::size_t index = 0; index < source.size(); ++index) {
    (*destination)[index] += scale * source[index];
  }
}

void add_scaled_hamiltonian_entry_dense_payload(
    const HamiltonianEntryDensePayload& source,
    double scale,
    HamiltonianEntryDensePayload* destination) {
  if (destination == nullptr) {
    throw std::invalid_argument("dense destination payload must not be null");
  }
  const bool destination_empty =
      destination->alpha_row_count == 0 &&
      destination->alpha_col_count == 0 &&
      destination->beta_row_count == 0 &&
      destination->beta_col_count == 0 &&
      destination->support_size == 0 &&
      std::abs(destination->overlap) <= kHamiltonianDenseZeroTolerance &&
      destination->alpha_degree1.empty() &&
      destination->alpha_degree2.empty() &&
      destination->beta_degree1.empty() &&
      destination->beta_degree2.empty() &&
      destination->mixed_degree1.empty();
  if (destination_empty) {
    *destination = make_zero_hamiltonian_entry_dense_payload(
        source.alpha_row_count,
        source.alpha_col_count,
        source.beta_row_count,
        source.beta_col_count,
        source.support_size);
  }
  if (!have_same_hamiltonian_entry_dense_payload_layout(source, *destination)) {
    throw std::invalid_argument("dense payload layouts do not match");
  }
  if (std::abs(scale) <= kHamiltonianDenseZeroTolerance) {
    return;
  }
  destination->overlap += scale * source.overlap;
  add_scaled_dense_values(source.alpha_degree1, scale, &destination->alpha_degree1);
  add_scaled_dense_values(source.alpha_degree2, scale, &destination->alpha_degree2);
  add_scaled_dense_values(source.beta_degree1, scale, &destination->beta_degree1);
  add_scaled_dense_values(source.beta_degree2, scale, &destination->beta_degree2);
  add_scaled_dense_values(source.mixed_degree1, scale, &destination->mixed_degree1);
}

void cleanup_dense_values(std::vector<double>* values) {
  if (values == nullptr) {
    throw std::invalid_argument("dense values must not be null");
  }
  for (double& value : *values) {
    if (std::abs(value) <= kHamiltonianDenseZeroTolerance) {
      value = 0.0;
    }
  }
}

void cleanup_hamiltonian_entry_dense_payload(
    HamiltonianEntryDensePayload* payload) {
  if (payload == nullptr) {
    throw std::invalid_argument("dense payload must not be null");
  }
  if (std::abs(payload->overlap) <= kHamiltonianDenseZeroTolerance) {
    payload->overlap = 0.0;
  }
  cleanup_dense_values(&payload->alpha_degree1);
  cleanup_dense_values(&payload->alpha_degree2);
  cleanup_dense_values(&payload->beta_degree1);
  cleanup_dense_values(&payload->beta_degree2);
  cleanup_dense_values(&payload->mixed_degree1);
}

bool encode_dense_spin_degree1_key(
    const SpinDeletionKey& key,
    int row_count,
    int col_count,
    int support_size,
    int* dense_index,
    int* canonical_parity) {
  if (dense_index == nullptr || canonical_parity == nullptr) {
    throw std::invalid_argument("dense degree-1 outputs must not be null");
  }
  SpinDeletionKey canonical_key = canonicalize_spin_key(key, canonical_parity);
  const int delta = dense_spin_degree_delta(row_count, col_count);
  switch (delta) {
    case -1:
      if (canonical_key.row_labels.size() != 0U ||
          canonical_key.col_labels.size() != 1U) {
        return false;
      }
      *dense_index = canonical_key.col_labels[0];
      return true;
    case 0:
      if (canonical_key.row_labels.size() != 1U ||
          canonical_key.col_labels.size() != 1U) {
        return false;
      }
      *dense_index = flatten_support_pair_index(
          canonical_key.row_labels[0],
          canonical_key.col_labels[0],
          support_size);
      return true;
    case 1:
      if (canonical_key.row_labels.size() != 1U ||
          canonical_key.col_labels.size() != 0U) {
        return false;
      }
      *dense_index = canonical_key.row_labels[0];
      return true;
    default:
      return false;
  }
}

bool encode_dense_spin_degree2_key(
    const SpinDeletionKey& key,
    int row_count,
    int col_count,
    int support_size,
    int* dense_index,
    int* canonical_parity) {
  if (dense_index == nullptr || canonical_parity == nullptr) {
    throw std::invalid_argument("dense degree-2 outputs must not be null");
  }
  SpinDeletionKey canonical_key = canonicalize_spin_key(key, canonical_parity);
  const int delta = dense_spin_degree_delta(row_count, col_count);
  const int pair_count = dense_antisym_support_pair_count(support_size);
  switch (delta) {
    case -2:
      if (canonical_key.row_labels.size() != 0U ||
          canonical_key.col_labels.size() != 2U) {
        return false;
      }
      *dense_index = flatten_antisym_support_pair_index(
          canonical_key.col_labels[0],
          canonical_key.col_labels[1],
          support_size);
      return true;
    case -1: {
      if (canonical_key.row_labels.size() != 1U ||
          canonical_key.col_labels.size() != 2U) {
        return false;
      }
      const int col_pair_index = flatten_antisym_support_pair_index(
          canonical_key.col_labels[0],
          canonical_key.col_labels[1],
          support_size);
      *dense_index =
          col_pair_index * support_size + canonical_key.row_labels[0];
      return true;
    }
    case 0:
      if (canonical_key.row_labels.size() != 2U ||
          canonical_key.col_labels.size() != 2U) {
        return false;
      }
      *dense_index = flatten_support_pair_pair_index(
          canonical_key.row_labels[0],
          canonical_key.row_labels[1],
          canonical_key.col_labels[0],
          canonical_key.col_labels[1],
          support_size);
      return true;
    case 1: {
      if (canonical_key.row_labels.size() != 2U ||
          canonical_key.col_labels.size() != 1U) {
        return false;
      }
      const int row_pair_index = flatten_antisym_support_pair_index(
          canonical_key.row_labels[0],
          canonical_key.row_labels[1],
          support_size);
      *dense_index =
          canonical_key.col_labels[0] * pair_count + row_pair_index;
      return true;
    }
    case 2:
      if (canonical_key.row_labels.size() != 2U ||
          canonical_key.col_labels.size() != 0U) {
        return false;
      }
      *dense_index = flatten_antisym_support_pair_index(
          canonical_key.row_labels[0],
          canonical_key.row_labels[1],
          support_size);
      return true;
    default:
      return false;
  }
}

void decode_dense_spin_degree1_key(
    int row_count,
    int col_count,
    int dense_index,
    int support_size,
    SpinDeletionKey* key) {
  if (key == nullptr) {
    throw std::invalid_argument("dense degree-1 decode output must not be null");
  }
  const int degree1_size = dense_spin_degree1_size(
      row_count,
      col_count,
      support_size);
  if (dense_index < 0 || dense_index >= degree1_size) {
    throw std::out_of_range("dense degree-1 index is out of range");
  }
  const int delta = dense_spin_degree_delta(row_count, col_count);
  switch (delta) {
    case -1:
      *key = make_spin_key({}, {dense_index});
      return;
    case 0: {
      int row_orbital = -1;
      int col_orbital = -1;
      decode_support_pair_index(
          dense_index,
          support_size,
          &row_orbital,
          &col_orbital);
      *key = make_spin_key({row_orbital}, {col_orbital});
      return;
    }
    case 1:
      *key = make_spin_key({dense_index}, {});
      return;
    default:
      throw std::logic_error("dense degree-1 decode requested for unsupported delta");
  }
}

void decode_dense_spin_degree2_key(
    int row_count,
    int col_count,
    int dense_index,
    int support_size,
    SpinDeletionKey* key) {
  if (key == nullptr) {
    throw std::invalid_argument("dense degree-2 decode output must not be null");
  }
  const int degree2_size = dense_spin_degree2_size(
      row_count,
      col_count,
      support_size);
  if (dense_index < 0 || dense_index >= degree2_size) {
    throw std::out_of_range("dense degree-2 index is out of range");
  }
  const int delta = dense_spin_degree_delta(row_count, col_count);
  const int pair_count = dense_antisym_support_pair_count(support_size);
  switch (delta) {
    case -2: {
      const DecodedAntisymSupportPair col_pair =
          decode_antisym_support_pair_index(dense_index, support_size);
      *key = make_spin_key({}, {col_pair.first_orbital, col_pair.second_orbital});
      return;
    }
    case -1: {
      const int row_orbital = dense_index % support_size;
      const int col_pair_index = dense_index / support_size;
      const DecodedAntisymSupportPair col_pair =
          decode_antisym_support_pair_index(col_pair_index, support_size);
      *key = make_spin_key(
          {row_orbital},
          {col_pair.first_orbital, col_pair.second_orbital});
      return;
    }
    case 0: {
      const int row_pair_index = dense_index % pair_count;
      const int col_pair_index = dense_index / pair_count;
      const DecodedAntisymSupportPair row_pair =
          decode_antisym_support_pair_index(row_pair_index, support_size);
      const DecodedAntisymSupportPair col_pair =
          decode_antisym_support_pair_index(col_pair_index, support_size);
      *key = make_spin_key(
          {row_pair.first_orbital, row_pair.second_orbital},
          {col_pair.first_orbital, col_pair.second_orbital});
      return;
    }
    case 1: {
      const int row_pair_index = dense_index % pair_count;
      const int col_orbital = dense_index / pair_count;
      const DecodedAntisymSupportPair row_pair =
          decode_antisym_support_pair_index(row_pair_index, support_size);
      *key = make_spin_key(
          {row_pair.first_orbital, row_pair.second_orbital},
          {col_orbital});
      return;
    }
    case 2: {
      const DecodedAntisymSupportPair row_pair =
          decode_antisym_support_pair_index(dense_index, support_size);
      *key = make_spin_key(
          {row_pair.first_orbital, row_pair.second_orbital},
          {});
      return;
    }
    default:
      throw std::logic_error("dense degree-2 decode requested for unsupported delta");
  }
}

bool accumulate_dense_spin_deleted_sector(
    const SpinDeletionKey& key,
    double value,
    int row_count,
    int col_count,
    int support_size,
    std::vector<double>* degree1,
    std::vector<double>* degree2) {
  if (degree1 == nullptr || degree2 == nullptr) {
    throw std::invalid_argument("dense spin degree outputs must not be null");
  }
  if (std::abs(value) <= kHamiltonianDenseZeroTolerance) {
    return false;
  }
  const int degree = sector_degree(key);
  if (degree == 1) {
    int dense_index = -1;
    int canonical_parity = 0;
    if (!encode_dense_spin_degree1_key(
            key,
            row_count,
            col_count,
            support_size,
            &dense_index,
            &canonical_parity)) {
      return false;
    }
    (*degree1)[xmvb::to_size(dense_index)] +=
        parity_sign(canonical_parity) * value;
    return true;
  }
  if (degree == 2) {
    int dense_index = -1;
    int canonical_parity = 0;
    if (!encode_dense_spin_degree2_key(
            key,
            row_count,
            col_count,
            support_size,
            &dense_index,
            &canonical_parity)) {
      return false;
    }
    (*degree2)[xmvb::to_size(dense_index)] +=
        parity_sign(canonical_parity) * value;
    return true;
  }
  return false;
}

bool accumulate_dense_mixed_deleted_sector(
    const SpinDeletionKey& alpha_key,
    const SpinDeletionKey& beta_key,
    double value,
    HamiltonianEntryDensePayload* payload) {
  if (payload == nullptr) {
    throw std::invalid_argument("dense mixed payload must not be null");
  }
  if (std::abs(value) <= kHamiltonianDenseZeroTolerance) {
    return false;
  }
  int alpha_index = -1;
  int beta_index = -1;
  int alpha_parity = 0;
  int beta_parity = 0;
  if (!encode_dense_spin_degree1_key(
          alpha_key,
          payload->alpha_row_count,
          payload->alpha_col_count,
          payload->support_size,
          &alpha_index,
          &alpha_parity) ||
      !encode_dense_spin_degree1_key(
          beta_key,
          payload->beta_row_count,
          payload->beta_col_count,
          payload->support_size,
          &beta_index,
          &beta_parity)) {
    return false;
  }
  payload->mixed_degree1[xmvb::to_size(
      dense_mixed_degree1_flat_index(*payload, alpha_index, beta_index))] +=
      parity_sign(alpha_parity ^ beta_parity) * value;
  return true;
}

struct WeightedOrientationTermPair {
  OrientationTerm left_term;
  OrientationTerm right_term;
  double coefficient = 0.0;
};

JointDeletionPayload merge_joint_deletion_payloads_limited(
    const JointDeletionPayload& left,
    const JointDeletionPayload& right,
    int max_spin_degree);

HamiltonianBoundaryPayload merge_hamiltonian_boundary_payloads_limited(
    const HamiltonianBoundaryPayload& left,
    const HamiltonianBoundaryPayload& right,
    int max_spin_degree);

HamiltonianBoundaryPayload make_zero_hamiltonian_payload(
    int alpha_row_count,
    int alpha_col_count,
    int beta_row_count,
    int beta_col_count);

void cleanup_hamiltonian_payload(HamiltonianBoundaryPayload* payload);

void append_hamiltonian_mixed_merge_terms(
    const HamiltonianBoundaryPayload& left,
    const HamiltonianBoundaryPayload& right,
    int max_spin_degree,
    std::vector<HamiltonianMixedDeletedSectorValue>* destination);

void reverse_merge_hamiltonian_boundary_payloads_limited(
    const HamiltonianBoundaryPayload& left,
    const HamiltonianBoundaryPayload& right,
    int max_spin_degree,
    const HamiltonianBoundaryPayload& merged_adjoint,
    HamiltonianBoundaryPayload* left_adjoint,
    HamiltonianBoundaryPayload* right_adjoint);

HamiltonianBoundaryPayload transform_hamiltonian_interface_block_to_front(
    const HamiltonianBoundaryPayload& natural_payload,
    int alpha_interface_rows,
    int alpha_interface_cols,
    const std::vector<int>& alpha_frontier_interface_rows,
    const std::vector<int>& alpha_frontier_interface_cols,
    const std::vector<int>& alpha_local_remainder_rows,
    const std::vector<int>& alpha_local_remainder_cols,
    int alpha_frontier_rows,
    int alpha_frontier_cols,
    int beta_interface_rows,
    int beta_interface_cols,
    const std::vector<int>& beta_frontier_interface_rows,
    const std::vector<int>& beta_frontier_interface_cols,
    const std::vector<int>& beta_local_remainder_rows,
    const std::vector<int>& beta_local_remainder_cols,
    int beta_frontier_rows,
    int beta_frontier_cols);

HamiltonianBoundaryPayload transform_hamiltonian_interface_block_to_front_values(
    const HamiltonianBoundaryPayload& natural_payload,
    int alpha_interface_rows,
    int alpha_interface_cols,
    const std::vector<int>& alpha_frontier_interface_rows,
    const std::vector<int>& alpha_frontier_interface_cols,
    const std::vector<int>& alpha_local_remainder_rows,
    const std::vector<int>& alpha_local_remainder_cols,
    int alpha_frontier_rows,
    int alpha_frontier_cols,
    int beta_interface_rows,
    int beta_interface_cols,
    const std::vector<int>& beta_frontier_interface_rows,
    const std::vector<int>& beta_frontier_interface_cols,
    const std::vector<int>& beta_local_remainder_rows,
    const std::vector<int>& beta_local_remainder_cols,
    int beta_frontier_rows,
    int beta_frontier_cols);

HamiltonianEntryDensePayload
transform_hamiltonian_entry_dense_payload_interface_block_to_front_values(
    const HamiltonianEntryDensePayload& natural_payload,
    int alpha_interface_rows,
    int alpha_interface_cols,
    const std::vector<int>& alpha_frontier_interface_rows,
    const std::vector<int>& alpha_frontier_interface_cols,
    const std::vector<int>& alpha_local_remainder_rows,
    const std::vector<int>& alpha_local_remainder_cols,
    int alpha_frontier_rows,
    int alpha_frontier_cols,
    int beta_interface_rows,
    int beta_interface_cols,
    const std::vector<int>& beta_frontier_interface_rows,
    const std::vector<int>& beta_frontier_interface_cols,
    const std::vector<int>& beta_local_remainder_rows,
    const std::vector<int>& beta_local_remainder_cols,
    int beta_frontier_rows,
    int beta_frontier_cols);

HamiltonianBoundaryPayload project_dense_hamiltonian_entry_payload_to_boundary_values(
    const HamiltonianEntryDensePayload& payload);

std::string format_dense_hamiltonian_entry_payload_summary(
    const HamiltonianEntryDensePayload& payload,
    std::size_t max_entries);

void validate_dense_projected_hamiltonian_payload_against_reference(
    const char* stage,
    const HamiltonianEntryDensePayload& dense_payload,
    const HamiltonianBoundaryPayload& reference_payload,
    const std::string& context);

SpinDeletionPayload build_exact_frontier_spin_payload_limited(
    const std::vector<int>& selected_root_cols,
    const std::vector<int>& leaf_left_occ,
    const std::vector<int>& selected_root_rows,
    const std::vector<int>& leaf_right_occ,
    int max_deleted_rank,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    std::unordered_map<SpinPayloadCacheKey,
                       SpinDeletionPayload,
                       SpinPayloadCacheKeyHasher>* payload_cache);

void finalize_boundary_hamiltonian_message(
    BoundaryHamiltonianMessage* message,
    bool preserve_structural_zero_messages);

void hash_combine(std::size_t* seed, std::size_t value) {
  if (seed == nullptr) {
    throw std::invalid_argument("hash seed must not be null");
  }
  *seed ^= value + 0x9e3779b97f4a7c15ULL + (*seed << 6) + (*seed >> 2);
}

std::size_t hash_occ_list(const std::vector<int>& occ) {
  std::size_t seed = occ.size();
  for (const int orbital : occ) {
    hash_combine(&seed, xmvb::to_size(orbital + 0x10000));
  }
  return seed;
}

std::vector<WeightedOrientationTermPair> build_weighted_component_orientation_pairs(
    const ComponentData& component) {
  std::map<std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>>,
           double>
      pair_coefficients;
  for (const auto& left_term : component.left_orientation_terms) {
    for (const auto& right_term : component.right_orientation_terms) {
      const double coefficient =
          left_term.coefficient * right_term.coefficient;
      if (std::abs(coefficient) <= 1.0e-15) {
        continue;
      }
      pair_coefficients[std::make_tuple(
          left_term.alpha_occ,
          left_term.beta_occ,
          right_term.alpha_occ,
          right_term.beta_occ)] += coefficient;
    }
  }

  std::vector<WeightedOrientationTermPair> pairs;
  pairs.reserve(pair_coefficients.size());
  for (auto& [key, coefficient] : pair_coefficients) {
    if (std::abs(coefficient) <= 1.0e-15) {
      continue;
    }
    WeightedOrientationTermPair pair;
    pair.left_term.alpha_occ = std::move(std::get<0>(key));
    pair.left_term.beta_occ = std::move(std::get<1>(key));
    pair.left_term.coefficient = 1.0;
    pair.right_term.alpha_occ = std::move(std::get<2>(key));
    pair.right_term.beta_occ = std::move(std::get<3>(key));
    pair.right_term.coefficient = 1.0;
    pair.coefficient = coefficient;
    pairs.push_back(std::move(pair));
  }
  return pairs;
}

std::size_t SpinPayloadCacheKeyHasher::operator()(
    const SpinPayloadCacheKey& key) const {
  std::size_t seed = 0;
  hash_combine(&seed, hash_occ_list(key.left_occ));
  hash_combine(&seed, hash_occ_list(key.right_occ));
  hash_combine(&seed, xmvb::to_size(key.internal_row_begin));
  hash_combine(&seed, xmvb::to_size(key.internal_col_begin));
  hash_combine(&seed, key.zero_selected_root_block ? 1U : 0U);
  hash_combine(&seed, xmvb::to_size(key.side));
  return seed;
}

std::size_t SubtreeMessageCacheKeyHasher::operator()(
    const SubtreeMessageCacheKey& key) const {
  std::size_t seed = xmvb::to_size(key.node + 0x4000);
  hash_combine(&seed, hash_occ_list(key.left_parent_alpha_occ));
  hash_combine(&seed, hash_occ_list(key.left_parent_beta_occ));
  hash_combine(&seed, hash_occ_list(key.right_parent_alpha_occ));
  hash_combine(&seed, hash_occ_list(key.right_parent_beta_occ));
  hash_combine(&seed, key.preserve_structural_zero_messages ? 1U : 0U);
  return seed;
}

std::size_t SpinOverlapCacheKeyHasher::operator()(
    const SpinOverlapCacheKey& key) const {
  std::size_t seed = 0;
  hash_combine(&seed, hash_occ_list(key.left_occ));
  hash_combine(&seed, hash_occ_list(key.right_occ));
  hash_combine(&seed, xmvb::to_size(key.interface_row_count + 0x400));
  hash_combine(&seed, xmvb::to_size(key.interface_col_count + 0x800));
  hash_combine(&seed, key.zero_selected_interface_block ? 1U : 0U);
  return seed;
}

std::uint32_t open_state_mask_limit(int n_bits) {
  if (n_bits < 0 || n_bits >= 31) {
    throw std::invalid_argument("open-state mask width is out of range");
  }
  return (n_bits == 0) ? 1U : (static_cast<std::uint32_t>(1U) << n_bits);
}

std::vector<int> select_occ_by_mask(
    const std::vector<int>& occ,
    std::uint32_t mask) {
  std::vector<int> selected;
  selected.reserve(occ.size());
  for (int index = 0; index < static_cast<int>(occ.size()); ++index) {
    if ((mask & (static_cast<std::uint32_t>(1U) << index)) != 0U) {
      selected.push_back(occ[xmvb::to_size(index)]);
    }
  }
  return selected;
}

std::vector<std::vector<int>> build_mask_occ_table(const std::vector<int>& occ) {
  const std::uint32_t limit = open_state_mask_limit(static_cast<int>(occ.size()));
  std::vector<std::vector<int>> occ_by_mask(xmvb::to_size(limit));
  for (std::uint32_t mask = 0; mask < limit; ++mask) {
    occ_by_mask[xmvb::to_size(mask)] = select_occ_by_mask(occ, mask);
  }
  return occ_by_mask;
}

void assign_concatenated_occ(
    const std::vector<int>& first,
    const std::vector<int>& second,
    std::vector<int>* result) {
  if (result == nullptr) {
    throw std::invalid_argument("result must not be null");
  }
  result->clear();
  result->reserve(first.size() + second.size());
  result->insert(result->end(), first.begin(), first.end());
  result->insert(result->end(), second.begin(), second.end());
}

Eigen::MatrixXd build_overlap_block(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& overlap_storage,
    int n_orbitals) {
  Eigen::MatrixXd overlap_block(
      static_cast<int>(right_occ.size()),
      static_cast<int>(left_occ.size()));
  for (int column = 0; column < static_cast<int>(left_occ.size()); ++column) {
    const int left_orbital = left_occ[xmvb::to_size(column)];
    for (int row = 0; row < static_cast<int>(right_occ.size()); ++row) {
      const int right_orbital = right_occ[xmvb::to_size(row)];
      overlap_block(row, column) =
          overlap_storage[xmvb::to_size(left_orbital) *
                              xmvb::to_size(n_orbitals) +
                          xmvb::to_size(right_orbital)];
    }
  }
  return overlap_block;
}

double determinant_of_dense_matrix(
    const Eigen::MatrixXd& matrix,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("determinant_of_dense_matrix requires a square matrix");
  }
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }
  ++(*subdeterminant_evaluations);
  switch (matrix.rows()) {
    case 0:
      return 1.0;
    case 1:
      return matrix(0, 0);
    default:
      break;
  }
  return overlap_resolver.resolve_matrix(matrix).overlap_determinant;
}

DeterminantOverlapResult resolve_overlap_result_counted(
    const Eigen::MatrixXd& matrix,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }
  ++(*subdeterminant_evaluations);

  DeterminantOverlapResult result;
  result.n_electrons = matrix.rows();
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("resolve_overlap_result_counted requires a square matrix");
  }
  switch (matrix.rows()) {
    case 0:
      result.nullity = 0;
      result.overlap_determinant = 1.0;
      return result;
    case 1:
      result.nullity = (std::abs(matrix(0, 0)) <= 1.0e-14) ? 1 : 0;
      result.overlap_determinant = matrix(0, 0);
      if (result.nullity == 0) {
        result.inverse_overlap_submatrix = Eigen::MatrixXd::Constant(1, 1, 1.0 / matrix(0, 0));
      } else {
        // The nullity-1 cofactor path requires a consistent rank-revealing
        // representation even for the scalar singular case. For the `1 x 1`
        // zero matrix, the exact first cofactor is the `0 x 0` determinant,
        // i.e. `1`, which is recovered from the null singular vectors below.
        result.singular_values = Eigen::VectorXd::Zero(1);
        result.matrix_U = Eigen::MatrixXd::Identity(1, 1);
        result.matrix_V = Eigen::MatrixXd::Identity(1, 1);
        result.parity = 1.0;
      }
      return result;
    default:
      return overlap_resolver.resolve_matrix(matrix);
  }
}

SpinScalarChannelValues build_spin_overlap_one_electron_cached(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    int interface_row_count,
    int interface_col_count,
    bool zero_selected_interface_block,
    const std::vector<double>& overlap_storage,
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    std::unordered_map<SpinOverlapCacheKey,
                       SpinScalarChannelValues,
                       SpinOverlapCacheKeyHasher>* overlap_cache) {
  // Evaluates one exact overlap determinant and one-electron scalar for a
  // block whose first
  // `interface_row_count` rows and first `interface_col_count` columns form the
  // boundary-interface block. When `zero_selected_interface_block` is true,
  // that leading block is zeroed before factorization. This is the exact local
  // scalar kernel used by the boundary-only overlap / one-electron recurrence.
  if (subdeterminant_evaluations == nullptr || overlap_cache == nullptr) {
    throw std::invalid_argument("spin overlap cache inputs must not be null");
  }
  const SpinOverlapCacheKey cache_key{
      left_occ,
      right_occ,
      interface_row_count,
      interface_col_count,
      zero_selected_interface_block,
  };
  auto cache_iterator = overlap_cache->find(cache_key);
  if (cache_iterator != overlap_cache->end()) {
    return cache_iterator->second;
  }

  Eigen::MatrixXd overlap_block =
      build_overlap_block(left_occ, right_occ, overlap_storage, n_orbitals);
  if (zero_selected_interface_block &&
      interface_row_count > 0 &&
      interface_col_count > 0) {
    overlap_block.topLeftCorner(interface_row_count, interface_col_count).setZero();
  }
  const DeterminantOverlapResult overlap_result = resolve_overlap_result_counted(
      overlap_block,
      overlap_resolver,
      subdeterminant_evaluations);

  double one_electron = 0.0;
  if (!left_occ.empty() && !one_electron_storage.empty()) {
    const Eigen::MatrixXd first_cofactor = calc_cofactor_1st(overlap_result);
    for (int column = 0; column < static_cast<int>(left_occ.size()); ++column) {
      const bool masked_interface_column =
          zero_selected_interface_block && column < interface_col_count;
      const int left_orbital = left_occ[xmvb::to_size(column)];
      for (int row = 0; row < static_cast<int>(right_occ.size()); ++row) {
        const bool masked_interface_row =
            zero_selected_interface_block && row < interface_row_count;
        if (masked_interface_row && masked_interface_column) {
          continue;
        }
        const int right_orbital = right_occ[xmvb::to_size(row)];
        one_electron +=
            one_electron_storage[xmvb::to_size(left_orbital) *
                                     xmvb::to_size(n_orbitals) +
                                 xmvb::to_size(right_orbital)] *
            first_cofactor(row, column);
      }
    }
  }

  const SpinScalarChannelValues values{
      .overlap = overlap_result.overlap_determinant,
      .one_electron = one_electron,
  };
  return overlap_cache->emplace(cache_key, values).first->second;
}

double build_spin_overlap_cached(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    int interface_row_count,
    int interface_col_count,
    bool zero_selected_interface_block,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    std::unordered_map<SpinOverlapCacheKey,
                       SpinScalarChannelValues,
                       SpinOverlapCacheKeyHasher>* overlap_cache) {
  static const std::vector<double> zero_one_electron_storage;
  return build_spin_overlap_one_electron_cached(
             left_occ,
             right_occ,
             interface_row_count,
             interface_col_count,
             zero_selected_interface_block,
             overlap_storage,
             zero_one_electron_storage,
             n_orbitals,
             overlap_resolver,
             subdeterminant_evaluations,
             overlap_cache)
      .overlap;
}

SpinScalarChannelValues build_exact_frontier_spin_overlap_one_electron(
    const std::vector<int>& selected_root_cols,
    const std::vector<int>& local_left_occ,
    const std::vector<int>& selected_root_rows,
    const std::vector<int>& local_right_occ,
    const std::vector<double>& overlap_storage,
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    std::unordered_map<SpinOverlapCacheKey,
                       SpinScalarChannelValues,
                       SpinOverlapCacheKeyHasher>* overlap_cache) {
  std::vector<int> frontier_left_occ;
  std::vector<int> frontier_right_occ;
  assign_concatenated_occ(selected_root_cols, local_left_occ, &frontier_left_occ);
  assign_concatenated_occ(selected_root_rows, local_right_occ, &frontier_right_occ);
  return build_spin_overlap_one_electron_cached(
      frontier_left_occ,
      frontier_right_occ,
      static_cast<int>(selected_root_rows.size()),
      static_cast<int>(selected_root_cols.size()),
      true,
      overlap_storage,
      one_electron_storage,
      n_orbitals,
      overlap_resolver,
      subdeterminant_evaluations,
      overlap_cache);
}

double build_exact_frontier_spin_overlap(
    const std::vector<int>& selected_root_cols,
    const std::vector<int>& local_left_occ,
    const std::vector<int>& selected_root_rows,
    const std::vector<int>& local_right_occ,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    std::unordered_map<SpinOverlapCacheKey,
                       SpinScalarChannelValues,
                       SpinOverlapCacheKeyHasher>* overlap_cache) {
  static const std::vector<double> zero_one_electron_storage;
  return build_exact_frontier_spin_overlap_one_electron(
             selected_root_cols,
             local_left_occ,
             selected_root_rows,
             local_right_occ,
             overlap_storage,
             zero_one_electron_storage,
             n_orbitals,
             overlap_resolver,
             subdeterminant_evaluations,
             overlap_cache)
      .overlap;
}

double build_exact_spin_overlap(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    std::unordered_map<SpinOverlapCacheKey,
                       SpinScalarChannelValues,
                       SpinOverlapCacheKeyHasher>* overlap_cache) {
  return build_spin_overlap_cached(
      left_occ,
      right_occ,
      0,
      0,
      false,
      overlap_storage,
      n_orbitals,
      overlap_resolver,
      subdeterminant_evaluations,
      overlap_cache);
}


void for_each_index_subset(
    int begin,
    int end,
    int rank,
    const std::function<void(const std::vector<int>&)>& callback) {
  // Enumerates ordered local-index deletion sets from one contiguous block.
  //
  // The returned vectors are always in ascending local-index order, which is
  // exactly the block order required by the deleted-minor sign convention used
  // throughout the separator recurrence.
  if (!callback) {
    throw std::invalid_argument("subset callback must not be empty");
  }
  if (rank < 0 || rank > 2) {
    throw std::invalid_argument("only rank-0/1/2 subsets are supported");
  }
  if (begin < 0 || end < begin) {
    throw std::invalid_argument("subset enumeration range is invalid");
  }
  if (rank > end - begin) {
    return;
  }

  if (rank == 0) {
    callback({});
    return;
  }
  for (int first = begin; first < end; ++first) {
    if (rank == 1) {
      callback({first});
      continue;
    }
    for (int second = first + 1; second < end; ++second) {
      callback({first, second});
    }
  }
}

int deleted_index_sum(const std::vector<int>& deleted_indices) {
  return std::accumulate(deleted_indices.begin(), deleted_indices.end(), 0);
}

Eigen::MatrixXd build_deleted_minor_matrix(
    const Eigen::MatrixXd& overlap_block,
    const std::vector<int>& deleted_rows,
    const std::vector<int>& deleted_cols) {
  if (overlap_block.rows() < static_cast<int>(deleted_rows.size()) ||
      overlap_block.cols() < static_cast<int>(deleted_cols.size())) {
    throw std::invalid_argument("deleted-minor request is larger than the source matrix");
  }

  Eigen::MatrixXd minor(
      overlap_block.rows() - static_cast<int>(deleted_rows.size()),
      overlap_block.cols() - static_cast<int>(deleted_cols.size()));
  int minor_row = 0;
  for (int row = 0; row < overlap_block.rows(); ++row) {
    if (std::find(deleted_rows.begin(), deleted_rows.end(), row) != deleted_rows.end()) {
      continue;
    }
    int minor_col = 0;
    for (int col = 0; col < overlap_block.cols(); ++col) {
      if (std::find(deleted_cols.begin(), deleted_cols.end(), col) != deleted_cols.end()) {
        continue;
      }
      minor(minor_row, minor_col) = overlap_block(row, col);
      ++minor_col;
    }
    ++minor_row;
  }
  return minor;
}

double deleted_minor_sign(
    int n_rows,
    int n_cols,
    const std::vector<int>& deleted_rows,
    const std::vector<int>& deleted_cols,
    PartialSide side) {
  // Generic sign convention for one-spin deleted minors.
  //
  // `RightComplement` uses the ordinary cofactor parity
  //   (-1)^(sum deleted rows + sum deleted cols).
  //
  // `LeftFrontier` is the separator zeroed-block orientation. The unmatched
  // row/column imbalance of the original frontier contributes an additional
  // permutation of parity
  //   min(n_rows, n_cols) * (n_rows - n_cols).
  // This reproduces the tested old cases:
  // - square `(1,1)` and `(2,2)` sectors,
  // - row-open `(1,0)` sectors with extra `n_cols`,
  // - col-open `(0,1)` sectors with extra `n_rows`,
  // and gives the correct non-star mixed `(2,1)` / `(1,2)` same-spin signs.
  int parity =
      deleted_index_sum(deleted_rows) +
      deleted_index_sum(deleted_cols);
  if (side == PartialSide::LeftFrontier) {
    parity += std::min(n_rows, n_cols) * (n_rows - n_cols);
  }
  return parity_sign(parity);
}

double determinant_of_dense_matrix(
    const Eigen::MatrixXd& matrix,
    const DeterminantOverlapResolver& overlap_resolver) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("determinant_of_dense_matrix requires a square matrix");
  }
  if (matrix.rows() == 0) {
    return 1.0;
  }
  if (matrix.rows() == 1) {
    return matrix(0, 0);
  }
  return overlap_resolver.resolve_matrix(matrix).overlap_determinant;
}

void append_spin_deleted_payload_basis_keys(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    int internal_row_begin,
    int internal_col_begin,
    int max_deleted_rank,
    SpinDeletionPayload* payload) {
  if (payload == nullptr) {
    throw std::invalid_argument("payload must not be null");
  }
  payload->basis_keys.clear();

  const int n_rows = static_cast<int>(right_occ.size());
  const int n_cols = static_cast<int>(left_occ.size());
  for (int deleted_row_rank = 0; deleted_row_rank <= max_deleted_rank; ++deleted_row_rank) {
    if (deleted_row_rank > n_rows - internal_row_begin) {
      continue;
    }
    for_each_index_subset(
        internal_row_begin,
        n_rows,
        deleted_row_rank,
        [&](const std::vector<int>& deleted_rows) {
          for (int deleted_col_rank = 0;
               deleted_col_rank <= max_deleted_rank;
               ++deleted_col_rank) {
            if (deleted_col_rank > n_cols - internal_col_begin) {
              continue;
            }
            if (n_rows - deleted_row_rank != n_cols - deleted_col_rank) {
              continue;
            }
            for_each_index_subset(
                internal_col_begin,
                n_cols,
                deleted_col_rank,
                [&](const std::vector<int>& deleted_cols) {
                  std::vector<int> deleted_row_labels;
                  std::vector<int> deleted_col_labels;
                  deleted_row_labels.reserve(deleted_rows.size());
                  deleted_col_labels.reserve(deleted_cols.size());
                  for (const int row : deleted_rows) {
                    deleted_row_labels.push_back(
                        right_occ[xmvb::to_size(row)]);
                  }
                  for (const int col : deleted_cols) {
                    deleted_col_labels.push_back(
                        left_occ[xmvb::to_size(col)]);
                  }
                  payload->basis_keys.push_back(SpinDeletionKey{
                      std::move(deleted_row_labels),
                      std::move(deleted_col_labels)});
                });
          }
        });
  }
}

void cleanup_spin_payload(SpinDeletionPayload* payload);

bool try_build_square_spin_deleted_minor_payload_fast(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const Eigen::MatrixXd& overlap_block,
    int internal_row_begin,
    int internal_col_begin,
    int max_deleted_rank,
    PartialSide side,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    SpinDeletionPayload* payload) {
  if (subdeterminant_evaluations == nullptr || payload == nullptr) {
    throw std::invalid_argument("square spin deleted-minor outputs must not be null");
  }

  const int n_rows = static_cast<int>(right_occ.size());
  const int n_cols = static_cast<int>(left_occ.size());
  if (n_rows != n_cols) {
    return false;
  }

  const DeterminantOverlapResult overlap_result = resolve_overlap_result_counted(
      overlap_block,
      overlap_resolver,
      subdeterminant_evaluations);
  const bool can_build_second_order =
      max_deleted_rank >= 2 &&
      overlap_result.nullity == 0 &&
      std::abs(overlap_result.overlap_determinant) > 1.0e-15;
  const bool can_build_first_order =
      max_deleted_rank >= 1 &&
      overlap_result.nullity <= 1;
  if (max_deleted_rank >= 2 && !can_build_second_order) {
    return false;
  }
  if (max_deleted_rank == 1 && !can_build_first_order) {
    return false;
  }

  payload->row_count = n_rows;
  payload->col_count = n_cols;
  payload->sectors.clear();
  payload->sectors[SpinDeletionKey{}] = overlap_result.overlap_determinant;

  if (!can_build_first_order && !can_build_second_order) {
    cleanup_spin_payload(payload);
    return true;
  }

  const Eigen::MatrixXd first_cofactor = calc_cofactor_1st(overlap_result);
  if (can_build_first_order) {
    for (int row = internal_row_begin; row < n_rows; ++row) {
      for (int col = internal_col_begin; col < n_cols; ++col) {
        const double value = first_cofactor(row, col);
        if (std::abs(value) <= 1.0e-15) {
          continue;
        }
        payload->sectors[SpinDeletionKey{
            {right_occ[xmvb::to_size(row)]},
            {left_occ[xmvb::to_size(col)]}}] = value;
      }
    }
  }

  if (can_build_second_order) {
    const double inverse_overlap = 1.0 / overlap_result.overlap_determinant;
    for (int row_first = internal_row_begin; row_first + 1 < n_rows; ++row_first) {
      for (int row_second = row_first + 1; row_second < n_rows; ++row_second) {
        for (int col_first = internal_col_begin; col_first + 1 < n_cols; ++col_first) {
          for (int col_second = col_first + 1; col_second < n_cols; ++col_second) {
            const double c11 = first_cofactor(row_first, col_first);
            const double c12 = first_cofactor(row_first, col_second);
            const double c21 = first_cofactor(row_second, col_first);
            const double c22 = first_cofactor(row_second, col_second);
            const double value =
                (c11 * c22 - c12 * c21) * inverse_overlap;
            if (std::abs(value) <= 1.0e-15) {
              continue;
            }
            payload->sectors[SpinDeletionKey{
                {
                    right_occ[xmvb::to_size(row_first)],
                    right_occ[xmvb::to_size(row_second)],
                },
                {
                    left_occ[xmvb::to_size(col_first)],
                    left_occ[xmvb::to_size(col_second)],
                }}] = value;
          }
        }
      }
    }
  }

  // For square blocks the frontier-vs-complement convention shift is zero, so
  // the standard cofactor signs already match the deleted-minor payload
  // convention for both `RightComplement` and `LeftFrontier`.
  static_cast<void>(side);
  cleanup_spin_payload(payload);
  return true;
}

SpinDeletionPayload build_spin_deleted_minor_payload_limited(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    int internal_row_begin,
    int internal_col_begin,
    int max_deleted_rank,
    bool zero_selected_root_block,
    PartialSide side,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  // Builds the complete rank<=2 scalar deleted-minor payload for one spin
  // block.
  //
  // `left_occ` and `right_occ` define the block-order bra/ket orbitals of the
  // current subtree/root matrix block. Rows before `internal_row_begin` and
  // columns before `internal_col_begin` belong to the parent interface and are
  // not allowed to appear in subtree-local deleted sectors. The returned map
  // contains every sector `(dr, dc)` with `dr <= 2`, `dc <= 2`, and
  //   n_rows - dr == n_cols - dc,
  // because only those deletions reduce the block to a square determinant.
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }
  if (max_deleted_rank < 0 || max_deleted_rank > 2) {
    throw std::invalid_argument("max_deleted_rank must be in [0, 2]");
  }

  const int n_rows = static_cast<int>(right_occ.size());
  const int n_cols = static_cast<int>(left_occ.size());
  SpinDeletionPayload payload;
  payload.row_count = n_rows;
  payload.col_count = n_cols;
  append_spin_deleted_payload_basis_keys(
      left_occ,
      right_occ,
      internal_row_begin,
      internal_col_begin,
      max_deleted_rank,
      &payload);

  Eigen::MatrixXd overlap_block =
      build_overlap_block(left_occ, right_occ, overlap_storage, n_orbitals);
  if (zero_selected_root_block && internal_row_begin > 0 && internal_col_begin > 0) {
    overlap_block.topLeftCorner(internal_row_begin, internal_col_begin).setZero();
  }

  if (try_build_square_spin_deleted_minor_payload_fast(
          left_occ,
          right_occ,
          overlap_block,
          internal_row_begin,
          internal_col_begin,
          max_deleted_rank,
          side,
          overlap_resolver,
          subdeterminant_evaluations,
          &payload)) {
    return payload;
  }

  for (int deleted_row_rank = 0; deleted_row_rank <= max_deleted_rank; ++deleted_row_rank) {
    if (deleted_row_rank > n_rows - internal_row_begin) {
      continue;
    }
    for_each_index_subset(
        internal_row_begin,
        n_rows,
        deleted_row_rank,
        [&](const std::vector<int>& deleted_rows) {
          for (int deleted_col_rank = 0;
               deleted_col_rank <= max_deleted_rank;
               ++deleted_col_rank) {
            if (deleted_col_rank > n_cols - internal_col_begin) {
              continue;
            }
            if (n_rows - deleted_row_rank != n_cols - deleted_col_rank) {
              continue;
            }
            for_each_index_subset(
                internal_col_begin,
                n_cols,
                deleted_col_rank,
                [&](const std::vector<int>& deleted_cols) {
                  const Eigen::MatrixXd minor = build_deleted_minor_matrix(
                      overlap_block,
                      deleted_rows,
                      deleted_cols);
                  const double determinant = determinant_of_dense_matrix(
                      minor,
                      overlap_resolver,
                      subdeterminant_evaluations);
                  if (std::abs(determinant) <= 1.0e-15) {
                    return;
                  }

                  std::vector<int> deleted_row_labels;
                  std::vector<int> deleted_col_labels;
                  deleted_row_labels.reserve(deleted_rows.size());
                  deleted_col_labels.reserve(deleted_cols.size());
                  for (const int row : deleted_rows) {
                    deleted_row_labels.push_back(
                        right_occ[xmvb::to_size(row)]);
                  }
                  for (const int col : deleted_cols) {
                    deleted_col_labels.push_back(
                        left_occ[xmvb::to_size(col)]);
                  }
                  payload.sectors[SpinDeletionKey{
                      std::move(deleted_row_labels),
                      std::move(deleted_col_labels)}] +=
                      deleted_minor_sign(
                          n_rows,
                          n_cols,
                          deleted_rows,
                          deleted_cols,
                          side) *
                      determinant;
                });
          }
        });
  }
  return payload;
}

const SpinDeletionPayload& build_spin_deleted_minor_payload_cached_limited(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    int internal_row_begin,
    int internal_col_begin,
    int max_deleted_rank,
    bool zero_selected_root_block,
    PartialSide side,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    std::unordered_map<SpinPayloadCacheKey,
                       SpinDeletionPayload,
                       SpinPayloadCacheKeyHasher>* payload_cache) {
  if (payload_cache == nullptr) {
    throw std::invalid_argument("payload_cache must not be null");
  }

  const SpinPayloadCacheKey cache_key{
      left_occ,
      right_occ,
      internal_row_begin,
      internal_col_begin,
      zero_selected_root_block,
      side,
  };
  auto iterator = payload_cache->find(cache_key);
  if (iterator == payload_cache->end()) {
    iterator = payload_cache
                   ->emplace(
                       cache_key,
                       build_spin_deleted_minor_payload_limited(
                           left_occ,
                           right_occ,
                           overlap_storage,
                           n_orbitals,
                           internal_row_begin,
                           internal_col_begin,
                           max_deleted_rank,
                           zero_selected_root_block,
                           side,
                           overlap_resolver,
                           subdeterminant_evaluations))
                   .first;
    cleanup_spin_payload(&iterator->second);
  }
  return iterator->second;
}

void cleanup_spin_payload(SpinDeletionPayload* payload) {
  if (payload == nullptr) {
    throw std::invalid_argument("payload must not be null");
  }
  for (auto iterator = payload->sectors.begin(); iterator != payload->sectors.end();) {
    if (std::abs(iterator->second) <= 1.0e-15) {
      iterator = payload->sectors.erase(iterator);
    } else {
      ++iterator;
    }
  }
}

SpinDeletionKey make_spin_key(
    std::vector<int> row_labels,
    std::vector<int> col_labels) {
  return SpinDeletionKey{std::move(row_labels), std::move(col_labels)};
}

int sector_degree(const SpinDeletionKey& key) {
  return std::max(
      static_cast<int>(key.row_labels.size()),
      static_cast<int>(key.col_labels.size()));
}

int spin_sector_merge_parity(
    int left_row_count,
    int left_col_count,
    const SpinDeletionKey& left_key,
    int right_row_count,
    int right_col_count,
    const SpinDeletionKey& right_key) {
  const int left_row_rank = static_cast<int>(left_key.row_labels.size());
  const int left_col_rank = static_cast<int>(left_key.col_labels.size());
  const int left_row_excess = std::max(0, left_row_rank - left_col_rank);
  const int left_col_excess = std::max(0, left_col_rank - left_row_rank);
  const int right_degree = sector_degree(right_key);
  return
      left_row_excess * right_col_count +
      left_col_excess * right_row_count +
      right_degree * (left_row_count + left_col_count);
}

JointDeletionKey make_joint_key(
    SpinDeletionKey alpha_key = {},
    SpinDeletionKey beta_key = {}) {
  return JointDeletionKey{std::move(alpha_key), std::move(beta_key)};
}

bool is_empty_spin_key(const SpinDeletionKey& key) {
  return key.row_labels.empty() && key.col_labels.empty();
}

bool combine_spin_keys_with_limit(
    const SpinDeletionKey& left_key,
    const SpinDeletionKey& right_key,
    int max_rank,
    SpinDeletionKey* combined_key) {
  if (combined_key == nullptr) {
    throw std::invalid_argument("combined_key must not be null");
  }
  if (max_rank < 0 || max_rank > 2) {
    throw std::invalid_argument("max_rank must be in [0, 2]");
  }
  const std::size_t combined_row_rank =
      left_key.row_labels.size() + right_key.row_labels.size();
  const std::size_t combined_col_rank =
      left_key.col_labels.size() + right_key.col_labels.size();
  if (combined_row_rank > xmvb::to_size(max_rank) ||
      combined_col_rank > xmvb::to_size(max_rank)) {
    return false;
  }
  combined_key->row_labels.clear();
  combined_key->col_labels.clear();
  combined_key->row_labels.reserve(combined_row_rank);
  combined_key->col_labels.reserve(combined_col_rank);
  combined_key->row_labels.insert(
      combined_key->row_labels.end(),
      left_key.row_labels.begin(),
      left_key.row_labels.end());
  combined_key->row_labels.insert(
      combined_key->row_labels.end(),
      right_key.row_labels.begin(),
      right_key.row_labels.end());
  combined_key->col_labels.insert(
      combined_key->col_labels.end(),
      left_key.col_labels.begin(),
      left_key.col_labels.end());
  combined_key->col_labels.insert(
      combined_key->col_labels.end(),
      right_key.col_labels.begin(),
      right_key.col_labels.end());
  return true;
}

bool combine_spin_keys(
    const SpinDeletionKey& left_key,
    const SpinDeletionKey& right_key,
    SpinDeletionKey* combined_key) {
  return combine_spin_keys_with_limit(left_key, right_key, 2, combined_key);
}

void cleanup_joint_payload(JointDeletionPayload* payload);
SpinDeletionPayload make_zero_spin_payload_like(const SpinDeletionPayload& payload);
JointDeletionPayload make_zero_joint_payload_like(const JointDeletionPayload& payload);
JointDeletionPayload scale_joint_payload(
    const JointDeletionPayload& payload,
    double scale);
void reverse_build_joint_payload(
    const SpinDeletionPayload& alpha_payload,
    const SpinDeletionPayload& beta_payload,
    const JointDeletionPayload& joint_adjoint,
    SpinDeletionPayload* alpha_adjoint,
    SpinDeletionPayload* beta_adjoint);
void reverse_merge_joint_deletion_payloads_limited(
    const JointDeletionPayload& left,
    const JointDeletionPayload& right,
    int max_spin_degree,
    const JointDeletionPayload& merged_adjoint,
    JointDeletionPayload* left_adjoint,
    JointDeletionPayload* right_adjoint);
void reverse_transform_interface_block_to_front(
    int alpha_interface_rows,
    int alpha_interface_cols,
    const std::vector<int>& alpha_frontier_interface_rows,
    const std::vector<int>& alpha_frontier_interface_cols,
    const std::vector<int>& alpha_local_remainder_rows,
    const std::vector<int>& alpha_local_remainder_cols,
    int alpha_frontier_rows,
    int alpha_frontier_cols,
    int beta_interface_rows,
    int beta_interface_cols,
    const std::vector<int>& beta_frontier_interface_rows,
    const std::vector<int>& beta_frontier_interface_cols,
    const std::vector<int>& beta_local_remainder_rows,
    const std::vector<int>& beta_local_remainder_cols,
    int beta_frontier_rows,
    int beta_frontier_cols,
    const JointDeletionPayload& transformed_adjoint,
    JointDeletionPayload* natural_adjoint);
void accumulate_direct_subtree_message_overlap_gradient_exact(
    const OverlapGradientReverseContext& context,
    const SubtreeExpansion& subtree_expansion,
    const OrientationTerm& left_parent_term,
    const OrientationTerm& right_parent_term,
    std::uint32_t target_alpha_row_mask,
    std::uint32_t target_alpha_col_mask,
    std::uint32_t target_beta_row_mask,
    std::uint32_t target_beta_col_mask,
    const JointDeletionPayload& target_adjoint,
    std::vector<double>* active_orbital_overlap_gradient);
void reverse_interface_hamiltonian_payload_overlap_gradient(
    const OverlapGradientReverseContext& context,
    const std::vector<int>& selected_parent_alpha_cols,
    const std::vector<int>& alpha_local_remainder_cols,
    const std::vector<int>& selected_parent_alpha_rows,
    const std::vector<int>& alpha_local_remainder_rows,
    const std::vector<int>& selected_parent_beta_cols,
    const std::vector<int>& beta_local_remainder_cols,
    const std::vector<int>& selected_parent_beta_rows,
    const std::vector<int>& beta_local_remainder_rows,
    const HamiltonianBoundaryPayload& interface_adjoint);
const BoundaryHamiltonianMessage& build_recursive_subtree_hamiltonian_messages_cached(
    const ComponentTree& tree,
    const std::vector<SubtreeExpansion>& subtree_expansions,
    int node,
    const OrientationTerm& left_parent_term,
    const OrientationTerm& right_parent_term,
    bool preserve_structural_zero_messages,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subtree_term_pair_count,
    std::uint64_t* subtree_message_state_count,
    std::uint64_t* subdeterminant_evaluations,
    std::uint64_t* dp_transition_count,
    std::unordered_map<SpinPayloadCacheKey,
                       SpinDeletionPayload,
                       SpinPayloadCacheKeyHasher>* payload_cache,
    std::unordered_map<SubtreeMessageCacheKey,
                       BoundaryHamiltonianMessage,
                       SubtreeMessageCacheKeyHasher>* message_cache);

const DenseBoundaryHamiltonianMessage&
build_recursive_subtree_dense_hamiltonian_messages_cached(
    const ComponentTree& tree,
    const std::vector<SubtreeExpansion>& subtree_expansions,
    int node,
    const OrientationTerm& left_parent_term,
    const OrientationTerm& right_parent_term,
    const std::vector<double>& overlap_storage,
    int support_size,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subtree_term_pair_count,
    std::uint64_t* subtree_message_state_count,
    std::uint64_t* subdeterminant_evaluations,
    std::uint64_t* dp_transition_count,
    std::unordered_map<SpinPayloadCacheKey,
                       SpinDeletionPayload,
                       SpinPayloadCacheKeyHasher>* payload_cache,
    std::unordered_map<SubtreeMessageCacheKey,
                       DenseBoundaryHamiltonianMessage,
                       SubtreeMessageCacheKeyHasher>* message_cache);

SpinDeletionKey canonicalize_spin_key(
    const SpinDeletionKey& key,
    int* parity) {
  if (parity == nullptr) {
    throw std::invalid_argument("parity output must not be null");
  }
  SpinDeletionKey canonical_key = key;
  *parity =
      canonicalization_parity(canonical_key.row_labels) ^
      canonicalization_parity(canonical_key.col_labels);
  std::sort(canonical_key.row_labels.begin(), canonical_key.row_labels.end());
  std::sort(canonical_key.col_labels.begin(), canonical_key.col_labels.end());
  return canonical_key;
}

void canonicalize_spin_payload_basis_keys(SpinDeletionPayload* payload) {
  if (payload == nullptr) {
    throw std::invalid_argument("payload must not be null");
  }

  for (auto& key : payload->basis_keys) {
    int parity = 0;
    key = canonicalize_spin_key(key, &parity);
    static_cast<void>(parity);
  }
  std::sort(payload->basis_keys.begin(), payload->basis_keys.end());
  payload->basis_keys.erase(
      std::unique(payload->basis_keys.begin(), payload->basis_keys.end()),
      payload->basis_keys.end());
}

void canonicalize_joint_payload_basis_keys(JointDeletionPayload* payload) {
  if (payload == nullptr) {
    throw std::invalid_argument("payload must not be null");
  }

  for (auto& key : payload->basis_keys) {
    int alpha_parity = 0;
    int beta_parity = 0;
    key = JointDeletionKey{
        canonicalize_spin_key(key.alpha_key, &alpha_parity),
        canonicalize_spin_key(key.beta_key, &beta_parity),
    };
    static_cast<void>(alpha_parity);
    static_cast<void>(beta_parity);
  }
  std::sort(payload->basis_keys.begin(), payload->basis_keys.end());
  payload->basis_keys.erase(
      std::unique(payload->basis_keys.begin(), payload->basis_keys.end()),
      payload->basis_keys.end());
}

void canonicalize_joint_payload_keys(JointDeletionPayload* payload) {
  if (payload == nullptr) {
    throw std::invalid_argument("payload must not be null");
  }

  std::map<JointDeletionKey, double> canonical_sectors;
  for (const auto& [key, value] : payload->sectors) {
    if (std::abs(value) <= 1.0e-15) {
      continue;
    }

    int alpha_parity = 0;
    int beta_parity = 0;
    const JointDeletionKey canonical_key{
        canonicalize_spin_key(key.alpha_key, &alpha_parity),
        canonicalize_spin_key(key.beta_key, &beta_parity),
    };
    canonical_sectors[canonical_key] +=
        parity_sign(alpha_parity ^ beta_parity) * value;
  }
  payload->sectors = std::move(canonical_sectors);
  canonicalize_joint_payload_basis_keys(payload);
  cleanup_joint_payload(payload);
}

JointDeletionPayload make_identity_joint_payload() {
  JointDeletionPayload payload;
  payload.basis_keys.push_back(make_joint_key(make_spin_key(), make_spin_key()));
  payload.sectors[make_joint_key(make_spin_key(), make_spin_key())] = 1.0;
  return payload;
}

bool is_joint_payload_nonzero(const JointDeletionPayload& payload) {
  for (const auto& [key, value] : payload.sectors) {
    static_cast<void>(key);
    if (std::abs(value) > 1.0e-15) {
      return true;
    }
  }
  return false;
}

void cleanup_joint_payload(JointDeletionPayload* payload) {
  if (payload == nullptr) {
    throw std::invalid_argument("payload must not be null");
  }
  for (auto iterator = payload->sectors.begin(); iterator != payload->sectors.end();) {
    if (std::abs(iterator->second) <= 1.0e-15) {
      iterator = payload->sectors.erase(iterator);
    } else {
      ++iterator;
    }
  }
}

JointDeletionPayload build_joint_payload(
    const SpinDeletionPayload& alpha_payload,
    const SpinDeletionPayload& beta_payload) {
  JointDeletionPayload joint_payload;
  joint_payload.alpha_row_count = alpha_payload.row_count;
  joint_payload.alpha_col_count = alpha_payload.col_count;
  joint_payload.beta_row_count = beta_payload.row_count;
  joint_payload.beta_col_count = beta_payload.col_count;
  joint_payload.basis_keys.reserve(
      alpha_payload.basis_keys.size() * beta_payload.basis_keys.size());
  for (const auto& alpha_key : alpha_payload.basis_keys) {
    for (const auto& beta_key : beta_payload.basis_keys) {
      joint_payload.basis_keys.push_back(make_joint_key(alpha_key, beta_key));
    }
  }
  for (const auto& [alpha_key, alpha_value] : alpha_payload.sectors) {
    if (std::abs(alpha_value) <= 1.0e-15) {
      continue;
    }
    for (const auto& [beta_key, beta_value] : beta_payload.sectors) {
      if (std::abs(beta_value) <= 1.0e-15) {
        continue;
      }
      joint_payload.sectors[make_joint_key(alpha_key, beta_key)] +=
          alpha_value * beta_value;
    }
  }
  canonicalize_joint_payload_basis_keys(&joint_payload);
  cleanup_joint_payload(&joint_payload);
  return joint_payload;
}

double spin_payload_overlap_value(const SpinDeletionPayload& payload) {
  const auto iterator = payload.sectors.find(make_spin_key());
  return (iterator == payload.sectors.end()) ? 0.0 : iterator->second;
}

void cleanup_same_spin_basis_keys(
    std::vector<SpinDeletionKey>* basis_keys) {
  if (basis_keys == nullptr) {
    throw std::invalid_argument("basis_keys must not be null");
  }
  for (auto& key : *basis_keys) {
    int parity = 0;
    key = canonicalize_spin_key(key, &parity);
    static_cast<void>(parity);
  }
  std::sort(basis_keys->begin(), basis_keys->end());
  basis_keys->erase(
      std::unique(basis_keys->begin(), basis_keys->end()),
      basis_keys->end());
  basis_keys->erase(
      std::remove_if(
          basis_keys->begin(),
          basis_keys->end(),
          [](const SpinDeletionKey& key) { return is_empty_spin_key(key); }),
      basis_keys->end());
}

void cleanup_mixed_basis_keys(
    std::vector<JointDeletionKey>* basis_keys) {
  if (basis_keys == nullptr) {
    throw std::invalid_argument("basis_keys must not be null");
  }
  for (auto& key : *basis_keys) {
    int alpha_parity = 0;
    int beta_parity = 0;
    key = JointDeletionKey{
        canonicalize_spin_key(key.alpha_key, &alpha_parity),
        canonicalize_spin_key(key.beta_key, &beta_parity),
    };
    static_cast<void>(alpha_parity);
    static_cast<void>(beta_parity);
  }
  std::sort(basis_keys->begin(), basis_keys->end());
  basis_keys->erase(
      std::unique(basis_keys->begin(), basis_keys->end()),
      basis_keys->end());
  basis_keys->erase(
      std::remove_if(
          basis_keys->begin(),
          basis_keys->end(),
          [](const JointDeletionKey& key) {
            return is_empty_spin_key(key.alpha_key) ||
                is_empty_spin_key(key.beta_key);
          }),
      basis_keys->end());
}

void cleanup_same_spin_deleted_sectors(
    std::vector<SameSpinDeletedSectorValue>* sectors) {
  // Canonicalizes row/column ordering inside each active same-spin sector and
  // compresses duplicate deleted-label patterns into one accumulated value.
  if (sectors == nullptr) {
    throw std::invalid_argument("sectors must not be null");
  }

  for (auto& sector : *sectors) {
    if (std::abs(sector.value) <= 1.0e-15) {
      continue;
    }
    int parity = 0;
    sector.key = canonicalize_spin_key(sector.key, &parity);
    sector.value *= parity_sign(parity);
  }

  std::sort(
      sectors->begin(),
      sectors->end(),
      [](const SameSpinDeletedSectorValue& left,
         const SameSpinDeletedSectorValue& right) {
        return left.key < right.key;
      });

  std::vector<SameSpinDeletedSectorValue> compacted;
  compacted.reserve(sectors->size());
  for (const auto& sector : *sectors) {
    if (std::abs(sector.value) <= 1.0e-15) {
      continue;
    }
    if (!compacted.empty() &&
        !(sector.key < compacted.back().key) &&
        !(compacted.back().key < sector.key)) {
      compacted.back().value += sector.value;
      continue;
    }
    compacted.push_back(sector);
  }

  sectors->clear();
  sectors->reserve(compacted.size());
  for (const auto& sector : compacted) {
    if (std::abs(sector.value) <= 1.0e-15) {
      continue;
    }
    sectors->push_back(sector);
  }
}

void append_same_spin_merge_cross_terms(
    const std::vector<SameSpinDeletedSectorValue>& left_sectors,
    int left_row_count,
    int left_col_count,
    const std::vector<SameSpinDeletedSectorValue>& right_sectors,
    int right_row_count,
    int right_col_count,
    int max_spin_degree,
    std::vector<SameSpinDeletedSectorValue>* destination) {
  // Forms the exact same-spin active-active merge terms that raise degree from
  // two first-order sectors into one second-order sector, or combine open
  // one-side sectors into the exact merged active channel.
  if (destination == nullptr) {
    throw std::invalid_argument("destination must not be null");
  }
  for (const auto& left_sector : left_sectors) {
    if (std::abs(left_sector.value) <= 1.0e-15) {
      continue;
    }
    for (const auto& right_sector : right_sectors) {
      if (std::abs(right_sector.value) <= 1.0e-15) {
        continue;
      }
      SpinDeletionKey combined_key;
      if (!combine_spin_keys_with_limit(
              left_sector.key,
              right_sector.key,
              max_spin_degree,
              &combined_key)) {
        continue;
      }
      const int parity = spin_sector_merge_parity(
          left_row_count,
          left_col_count,
          left_sector.key,
          right_row_count,
          right_col_count,
          right_sector.key);
      destination->push_back(SameSpinDeletedSectorValue{
          .key = std::move(combined_key),
          .value = parity_sign(parity) * left_sector.value * right_sector.value,
      });
    }
  }
}

void append_same_spin_overlap_merge_terms(
    const std::vector<SameSpinDeletedSectorValue>& source_sectors,
    int source_row_count,
    int source_col_count,
    int other_row_count,
    int other_col_count,
    bool source_is_left,
    int max_spin_degree,
    double overlap_scale,
    std::vector<SameSpinDeletedSectorValue>* destination) {
  // Merges one active channel against the closed overlap sector of the other
  // factor. Open sectors carry nontrivial separator parity even in this
  // active-vs-overlap case, so this helper preserves the exact sign from the
  // deleted-minor merge law.
  if (destination == nullptr) {
    throw std::invalid_argument("destination must not be null");
  }
  if (std::abs(overlap_scale) <= 1.0e-15) {
    return;
  }
  for (const auto& sector : source_sectors) {
    if (std::abs(sector.value) <= 1.0e-15 ||
        sector_degree(sector.key) > max_spin_degree) {
      continue;
    }
    const int parity =
        source_is_left
            ? spin_sector_merge_parity(
                  source_row_count,
                  source_col_count,
                  sector.key,
                  other_row_count,
                  other_col_count,
                  make_spin_key())
            : spin_sector_merge_parity(
                  other_row_count,
                  other_col_count,
                  make_spin_key(),
                  source_row_count,
                  source_col_count,
                  sector.key);
    destination->push_back(SameSpinDeletedSectorValue{
        .key = sector.key,
        .value = parity_sign(parity) * overlap_scale * sector.value,
    });
  }
}

HamiltonianBoundaryPayload merge_hamiltonian_boundary_payloads_limited_values(
    const HamiltonianBoundaryPayload& left,
    const HamiltonianBoundaryPayload& right,
    int max_spin_degree) {
  // Fast exact forward merge on the typed Hamiltonian channels.
  //
  // This is the same algebra as the joint deleted-minor merge projected onto
  // the carried production channels, but done directly on those channels
  // instead of round-tripping through `JointDeletionPayload`.
  if (max_spin_degree < 0 || max_spin_degree > 2) {
    throw std::invalid_argument("max_spin_degree must be in [0, 2]");
  }

  HamiltonianBoundaryPayload merged = make_zero_hamiltonian_payload(
      left.alpha_row_count + right.alpha_row_count,
      left.alpha_col_count + right.alpha_col_count,
      left.beta_row_count + right.beta_row_count,
      left.beta_col_count + right.beta_col_count);
  merged.overlap = left.overlap * right.overlap;
  if (max_spin_degree == 0) {
    cleanup_hamiltonian_payload(&merged);
    return merged;
  }

  append_same_spin_overlap_merge_terms(
      left.alpha_sectors,
      left.alpha_row_count,
      left.alpha_col_count,
      right.alpha_row_count,
      right.alpha_col_count,
      true,
      max_spin_degree,
      right.overlap,
      &merged.alpha_sectors);
  append_same_spin_overlap_merge_terms(
      right.alpha_sectors,
      right.alpha_row_count,
      right.alpha_col_count,
      left.alpha_row_count,
      left.alpha_col_count,
      false,
      max_spin_degree,
      left.overlap,
      &merged.alpha_sectors);
  append_same_spin_merge_cross_terms(
      left.alpha_sectors,
      left.alpha_row_count,
      left.alpha_col_count,
      right.alpha_sectors,
      right.alpha_row_count,
      right.alpha_col_count,
      max_spin_degree,
      &merged.alpha_sectors);

  append_same_spin_overlap_merge_terms(
      left.beta_sectors,
      left.beta_row_count,
      left.beta_col_count,
      right.beta_row_count,
      right.beta_col_count,
      true,
      max_spin_degree,
      right.overlap,
      &merged.beta_sectors);
  append_same_spin_overlap_merge_terms(
      right.beta_sectors,
      right.beta_row_count,
      right.beta_col_count,
      left.beta_row_count,
      left.beta_col_count,
      false,
      max_spin_degree,
      left.overlap,
      &merged.beta_sectors);
  append_same_spin_merge_cross_terms(
      left.beta_sectors,
      left.beta_row_count,
      left.beta_col_count,
      right.beta_sectors,
      right.beta_row_count,
      right.beta_col_count,
      max_spin_degree,
      &merged.beta_sectors);

  append_hamiltonian_mixed_merge_terms(
      left,
      right,
      max_spin_degree,
      &merged.mixed_sectors);
  cleanup_hamiltonian_payload(&merged);
  return merged;
}

struct DenseSameSpinMergeSourceTerm {
  SpinDeletionKey key;
  double value = 0.0;
};

std::vector<DenseSameSpinMergeSourceTerm>
build_dense_same_spin_merge_source_terms(
    const std::vector<double>& degree1,
    const std::vector<double>& degree2,
    int row_count,
    int col_count,
    int support_size,
    double overlap) {
  std::vector<DenseSameSpinMergeSourceTerm> terms;
  terms.reserve(1U + degree1.size() + degree2.size());
  if (std::abs(overlap) > kHamiltonianDenseZeroTolerance) {
    terms.push_back(DenseSameSpinMergeSourceTerm{
        .key = make_spin_key(),
        .value = overlap,
    });
  }
  for (int index = 0; index < static_cast<int>(degree1.size()); ++index) {
    const double value = degree1[xmvb::to_size(index)];
    if (std::abs(value) <= kHamiltonianDenseZeroTolerance) {
      continue;
    }
    SpinDeletionKey key;
    decode_dense_spin_degree1_key(
        row_count,
        col_count,
        index,
        support_size,
        &key);
    terms.push_back(DenseSameSpinMergeSourceTerm{
        .key = std::move(key),
        .value = value,
    });
  }
  for (int index = 0; index < static_cast<int>(degree2.size()); ++index) {
    const double value = degree2[xmvb::to_size(index)];
    if (std::abs(value) <= kHamiltonianDenseZeroTolerance) {
      continue;
    }
    SpinDeletionKey key;
    decode_dense_spin_degree2_key(
        row_count,
        col_count,
        index,
        support_size,
        &key);
    terms.push_back(DenseSameSpinMergeSourceTerm{
        .key = std::move(key),
        .value = value,
    });
  }
  return terms;
}

void append_dense_exact_same_spin_merge_terms(
    const std::vector<double>& left_degree1,
    const std::vector<double>& left_degree2,
    int left_row_count,
    int left_col_count,
    double left_overlap,
    const std::vector<double>& right_degree1,
    const std::vector<double>& right_degree2,
    int right_row_count,
    int right_col_count,
    double right_overlap,
    int max_spin_degree,
    int support_size,
    std::vector<double>* destination_degree1,
    std::vector<double>* destination_degree2) {
  if (destination_degree1 == nullptr || destination_degree2 == nullptr) {
    throw std::invalid_argument("dense same-spin merge destinations must not be null");
  }

  const std::vector<DenseSameSpinMergeSourceTerm> left_terms =
      build_dense_same_spin_merge_source_terms(
          left_degree1,
          left_degree2,
          left_row_count,
          left_col_count,
          support_size,
          left_overlap);
  const std::vector<DenseSameSpinMergeSourceTerm> right_terms =
      build_dense_same_spin_merge_source_terms(
          right_degree1,
          right_degree2,
          right_row_count,
          right_col_count,
          support_size,
          right_overlap);
  for (const auto& left_term : left_terms) {
    if (std::abs(left_term.value) <= kHamiltonianDenseZeroTolerance) {
      continue;
    }
    for (const auto& right_term : right_terms) {
      if (std::abs(right_term.value) <= kHamiltonianDenseZeroTolerance) {
        continue;
      }

      SpinDeletionKey combined_key;
      if (!combine_spin_keys_with_limit(
              left_term.key,
              right_term.key,
              max_spin_degree,
              &combined_key) ||
          is_empty_spin_key(combined_key)) {
        continue;
      }
      const int parity = spin_sector_merge_parity(
          left_row_count,
          left_col_count,
          left_term.key,
          right_row_count,
          right_col_count,
          right_term.key);
      static_cast<void>(accumulate_dense_spin_deleted_sector(
          combined_key,
          parity_sign(parity) * left_term.value * right_term.value,
          left_row_count + right_row_count,
          left_col_count + right_col_count,
          support_size,
          destination_degree1,
          destination_degree2));
    }
  }
}

struct DenseHamiltonianMixedMergeSourceTerm {
  SpinDeletionKey alpha_key;
  SpinDeletionKey beta_key;
  double value = 0.0;
};

std::vector<DenseHamiltonianMixedMergeSourceTerm>
build_dense_hamiltonian_mixed_merge_source_terms(
    const HamiltonianEntryDensePayload& payload) {
  std::vector<DenseHamiltonianMixedMergeSourceTerm> terms;
  terms.reserve(
      1U +
      payload.alpha_degree1.size() +
      payload.beta_degree1.size() +
      payload.mixed_degree1.size());
  if (std::abs(payload.overlap) > kHamiltonianDenseZeroTolerance) {
    terms.push_back(DenseHamiltonianMixedMergeSourceTerm{
        .alpha_key = make_spin_key(),
        .beta_key = make_spin_key(),
        .value = payload.overlap,
    });
  }

  for (int index = 0; index < static_cast<int>(payload.alpha_degree1.size()); ++index) {
    const double value = payload.alpha_degree1[xmvb::to_size(index)];
    if (std::abs(value) <= kHamiltonianDenseZeroTolerance) {
      continue;
    }
    SpinDeletionKey alpha_key;
    decode_dense_spin_degree1_key(
        payload.alpha_row_count,
        payload.alpha_col_count,
        index,
        payload.support_size,
        &alpha_key);
    terms.push_back(DenseHamiltonianMixedMergeSourceTerm{
        .alpha_key = std::move(alpha_key),
        .beta_key = make_spin_key(),
        .value = value,
    });
  }

  for (int index = 0; index < static_cast<int>(payload.beta_degree1.size()); ++index) {
    const double value = payload.beta_degree1[xmvb::to_size(index)];
    if (std::abs(value) <= kHamiltonianDenseZeroTolerance) {
      continue;
    }
    SpinDeletionKey beta_key;
    decode_dense_spin_degree1_key(
        payload.beta_row_count,
        payload.beta_col_count,
        index,
        payload.support_size,
        &beta_key);
    terms.push_back(DenseHamiltonianMixedMergeSourceTerm{
        .alpha_key = make_spin_key(),
        .beta_key = std::move(beta_key),
        .value = value,
    });
  }

  const int alpha_degree1_size = dense_spin_degree1_size(
      payload.alpha_row_count,
      payload.alpha_col_count,
      payload.support_size);
  const int beta_degree1_size = dense_spin_degree1_size(
      payload.beta_row_count,
      payload.beta_col_count,
      payload.support_size);
  for (int beta_index = 0; beta_index < beta_degree1_size; ++beta_index) {
    for (int alpha_index = 0; alpha_index < alpha_degree1_size; ++alpha_index) {
      const double value = payload.mixed_degree1[xmvb::to_size(
          dense_mixed_degree1_flat_index(payload, alpha_index, beta_index))];
      if (std::abs(value) <= kHamiltonianDenseZeroTolerance) {
        continue;
      }
      SpinDeletionKey alpha_key;
      SpinDeletionKey beta_key;
      decode_dense_spin_degree1_key(
          payload.alpha_row_count,
          payload.alpha_col_count,
          alpha_index,
          payload.support_size,
          &alpha_key);
      decode_dense_spin_degree1_key(
          payload.beta_row_count,
          payload.beta_col_count,
          beta_index,
          payload.support_size,
          &beta_key);
      terms.push_back(DenseHamiltonianMixedMergeSourceTerm{
          .alpha_key = std::move(alpha_key),
          .beta_key = std::move(beta_key),
          .value = value,
      });
    }
  }
  return terms;
}

void append_dense_exact_mixed_merge_terms(
    const HamiltonianEntryDensePayload& left,
    const HamiltonianEntryDensePayload& right,
    HamiltonianEntryDensePayload* destination) {
  if (destination == nullptr) {
    throw std::invalid_argument("dense mixed merge destination must not be null");
  }

  const std::vector<DenseHamiltonianMixedMergeSourceTerm> left_terms =
      build_dense_hamiltonian_mixed_merge_source_terms(left);
  const std::vector<DenseHamiltonianMixedMergeSourceTerm> right_terms =
      build_dense_hamiltonian_mixed_merge_source_terms(right);
  for (const auto& left_term : left_terms) {
    if (std::abs(left_term.value) <= kHamiltonianDenseZeroTolerance) {
      continue;
    }
    for (const auto& right_term : right_terms) {
      if (std::abs(right_term.value) <= kHamiltonianDenseZeroTolerance) {
        continue;
      }

      SpinDeletionKey combined_alpha_key;
      SpinDeletionKey combined_beta_key;
      if (!combine_spin_keys_with_limit(
              left_term.alpha_key,
              right_term.alpha_key,
              1,
              &combined_alpha_key) ||
          !combine_spin_keys_with_limit(
              left_term.beta_key,
              right_term.beta_key,
              1,
              &combined_beta_key) ||
          is_empty_spin_key(combined_alpha_key) ||
          is_empty_spin_key(combined_beta_key)) {
        continue;
      }

      const int parity =
          spin_sector_merge_parity(
              left.alpha_row_count,
              left.alpha_col_count,
              left_term.alpha_key,
              right.alpha_row_count,
              right.alpha_col_count,
              right_term.alpha_key) ^
          spin_sector_merge_parity(
              left.beta_row_count,
              left.beta_col_count,
              left_term.beta_key,
              right.beta_row_count,
              right.beta_col_count,
              right_term.beta_key);
      static_cast<void>(accumulate_dense_mixed_deleted_sector(
          combined_alpha_key,
          combined_beta_key,
          parity_sign(parity) * left_term.value * right_term.value,
          destination));
    }
  }
}

HamiltonianEntryDensePayload merge_hamiltonian_entry_dense_payloads_limited_values(
    const HamiltonianEntryDensePayload& left,
    const HamiltonianEntryDensePayload& right) {
  if (left.support_size != right.support_size) {
    throw std::invalid_argument("dense payload support sizes do not match");
  }
  HamiltonianEntryDensePayload merged =
      make_zero_hamiltonian_entry_dense_payload(
          left.alpha_row_count + right.alpha_row_count,
          left.alpha_col_count + right.alpha_col_count,
          left.beta_row_count + right.beta_row_count,
          left.beta_col_count + right.beta_col_count,
          left.support_size);
  merged.overlap = left.overlap * right.overlap;

  append_dense_exact_same_spin_merge_terms(
      left.alpha_degree1,
      left.alpha_degree2,
      left.alpha_row_count,
      left.alpha_col_count,
      left.overlap,
      right.alpha_degree1,
      right.alpha_degree2,
      right.alpha_row_count,
      right.alpha_col_count,
      right.overlap,
      2,
      left.support_size,
      &merged.alpha_degree1,
      &merged.alpha_degree2);

  append_dense_exact_same_spin_merge_terms(
      left.beta_degree1,
      left.beta_degree2,
      left.beta_row_count,
      left.beta_col_count,
      left.overlap,
      right.beta_degree1,
      right.beta_degree2,
      right.beta_row_count,
      right.beta_col_count,
      right.overlap,
      2,
      left.support_size,
      &merged.beta_degree1,
      &merged.beta_degree2);

  append_dense_exact_mixed_merge_terms(left, right, &merged);
  cleanup_hamiltonian_entry_dense_payload(&merged);
  std::ostringstream merge_context;
  merge_context << "left="
                << format_dense_hamiltonian_entry_payload_summary(left, 16U)
                << "; right="
                << format_dense_hamiltonian_entry_payload_summary(right, 16U);
  validate_dense_projected_hamiltonian_payload_against_reference(
      "merge_hamiltonian_entry_dense_payloads_limited_values",
      merged,
      merge_hamiltonian_boundary_payloads_limited_values(
          project_dense_hamiltonian_entry_payload_to_boundary_values(left),
          project_dense_hamiltonian_entry_payload_to_boundary_values(right),
          2),
      merge_context.str());
  return merged;
}

void cleanup_hamiltonian_mixed_sectors(
    std::vector<HamiltonianMixedDeletedSectorValue>* sectors) {
  // Canonicalizes each mixed alpha/beta first-order sector and compresses
  // duplicate joint deleted-label patterns.
  if (sectors == nullptr) {
    throw std::invalid_argument("sectors must not be null");
  }

  for (auto& sector : *sectors) {
    if (std::abs(sector.value) <= 1.0e-15) {
      continue;
    }
    int alpha_parity = 0;
    int beta_parity = 0;
    sector.key = make_joint_key(
        canonicalize_spin_key(sector.key.alpha_key, &alpha_parity),
        canonicalize_spin_key(sector.key.beta_key, &beta_parity));
    sector.value *= parity_sign(alpha_parity ^ beta_parity);
  }

  std::sort(
      sectors->begin(),
      sectors->end(),
      [](const HamiltonianMixedDeletedSectorValue& left,
         const HamiltonianMixedDeletedSectorValue& right) {
        return left.key < right.key;
      });

  std::vector<HamiltonianMixedDeletedSectorValue> compacted;
  compacted.reserve(sectors->size());
  for (const auto& sector : *sectors) {
    if (std::abs(sector.value) <= 1.0e-15) {
      continue;
    }
    if (!compacted.empty() &&
        !(sector.key < compacted.back().key) &&
        !(compacted.back().key < sector.key)) {
      compacted.back().value += sector.value;
      continue;
    }
    compacted.push_back(sector);
  }

  sectors->clear();
  sectors->reserve(compacted.size());
  for (const auto& sector : compacted) {
    if (std::abs(sector.value) <= 1.0e-15) {
      continue;
    }
    sectors->push_back(sector);
  }
}

void cleanup_hamiltonian_payload(HamiltonianBoundaryPayload* payload) {
  if (payload == nullptr) {
    throw std::invalid_argument("payload must not be null");
  }
  if (std::abs(payload->overlap) <= 1.0e-15) {
    payload->overlap = 0.0;
  }
  cleanup_same_spin_basis_keys(&payload->alpha_basis_keys);
  cleanup_same_spin_basis_keys(&payload->beta_basis_keys);
  cleanup_mixed_basis_keys(&payload->mixed_basis_keys);
  cleanup_same_spin_deleted_sectors(&payload->alpha_sectors);
  cleanup_same_spin_deleted_sectors(&payload->beta_sectors);
  cleanup_hamiltonian_mixed_sectors(&payload->mixed_sectors);
}

bool is_hamiltonian_payload_nonzero(const HamiltonianBoundaryPayload& payload) {
  return std::abs(payload.overlap) > 1.0e-15 ||
      !payload.alpha_sectors.empty() ||
      !payload.beta_sectors.empty() ||
      !payload.mixed_sectors.empty();
}

bool has_hamiltonian_payload_basis(const HamiltonianBoundaryPayload& payload) {
  return payload.has_overlap_basis ||
      !payload.alpha_basis_keys.empty() ||
      !payload.beta_basis_keys.empty() ||
      !payload.mixed_basis_keys.empty() ||
      is_hamiltonian_payload_nonzero(payload);
}

HamiltonianBoundaryPayload make_zero_hamiltonian_payload(
    int alpha_row_count,
    int alpha_col_count,
    int beta_row_count,
    int beta_col_count) {
  HamiltonianBoundaryPayload payload;
  payload.alpha_row_count = alpha_row_count;
  payload.alpha_col_count = alpha_col_count;
  payload.beta_row_count = beta_row_count;
  payload.beta_col_count = beta_col_count;
  return payload;
}

HamiltonianBoundaryPayload make_zero_hamiltonian_payload_like(
    const HamiltonianBoundaryPayload& payload) {
  HamiltonianBoundaryPayload zero = make_zero_hamiltonian_payload(
      payload.alpha_row_count,
      payload.alpha_col_count,
      payload.beta_row_count,
      payload.beta_col_count);
  zero.has_overlap_basis = payload.has_overlap_basis;
  zero.alpha_basis_keys = payload.alpha_basis_keys;
  zero.beta_basis_keys = payload.beta_basis_keys;
  zero.mixed_basis_keys = payload.mixed_basis_keys;
  return zero;
}

HamiltonianBoundaryPayload make_identity_hamiltonian_payload() {
  HamiltonianBoundaryPayload payload;
  payload.has_overlap_basis = true;
  payload.overlap = 1.0;
  return payload;
}

HamiltonianBoundaryPayload scale_hamiltonian_payload(
    const HamiltonianBoundaryPayload& payload,
    double scale) {
  HamiltonianBoundaryPayload scaled = make_zero_hamiltonian_payload_like(payload);
  if (std::abs(scale) <= 1.0e-15) {
    return scaled;
  }
  scaled.has_overlap_basis = payload.has_overlap_basis;
  scaled.overlap = scale * payload.overlap;
  scaled.alpha_basis_keys = payload.alpha_basis_keys;
  scaled.beta_basis_keys = payload.beta_basis_keys;
  scaled.mixed_basis_keys = payload.mixed_basis_keys;
  scaled.alpha_sectors.reserve(payload.alpha_sectors.size());
  for (const auto& sector : payload.alpha_sectors) {
    if (std::abs(sector.value) <= 1.0e-15) {
      continue;
    }
    scaled.alpha_sectors.push_back(SameSpinDeletedSectorValue{
        .key = sector.key,
        .value = scale * sector.value,
    });
  }
  scaled.beta_sectors.reserve(payload.beta_sectors.size());
  for (const auto& sector : payload.beta_sectors) {
    if (std::abs(sector.value) <= 1.0e-15) {
      continue;
    }
    scaled.beta_sectors.push_back(SameSpinDeletedSectorValue{
        .key = sector.key,
        .value = scale * sector.value,
    });
  }
  scaled.mixed_sectors.reserve(payload.mixed_sectors.size());
  for (const auto& sector : payload.mixed_sectors) {
    if (std::abs(sector.value) <= 1.0e-15) {
      continue;
    }
    scaled.mixed_sectors.push_back(HamiltonianMixedDeletedSectorValue{
        .key = sector.key,
        .value = scale * sector.value,
    });
  }
  cleanup_hamiltonian_payload(&scaled);
  return scaled;
}

JointDeletionPayload convert_hamiltonian_payload_to_joint(
    const HamiltonianBoundaryPayload& payload) {
  JointDeletionPayload joint_payload;
  joint_payload.alpha_row_count = payload.alpha_row_count;
  joint_payload.alpha_col_count = payload.alpha_col_count;
  joint_payload.beta_row_count = payload.beta_row_count;
  joint_payload.beta_col_count = payload.beta_col_count;

  if (payload.has_overlap_basis) {
    const JointDeletionKey key = make_joint_key(make_spin_key(), make_spin_key());
    joint_payload.basis_keys.push_back(key);
  }
  for (const auto& key : payload.alpha_basis_keys) {
    joint_payload.basis_keys.push_back(make_joint_key(key, make_spin_key()));
  }
  for (const auto& key : payload.beta_basis_keys) {
    joint_payload.basis_keys.push_back(make_joint_key(make_spin_key(), key));
  }
  for (const auto& key : payload.mixed_basis_keys) {
    joint_payload.basis_keys.push_back(key);
  }

  if (std::abs(payload.overlap) > 1.0e-15) {
    joint_payload.sectors[make_joint_key(make_spin_key(), make_spin_key())] +=
        payload.overlap;
  }
  for (const auto& sector : payload.alpha_sectors) {
    if (std::abs(sector.value) <= 1.0e-15) {
      continue;
    }
    joint_payload.sectors[make_joint_key(sector.key, make_spin_key())] += sector.value;
  }
  for (const auto& sector : payload.beta_sectors) {
    if (std::abs(sector.value) <= 1.0e-15) {
      continue;
    }
    joint_payload.sectors[make_joint_key(make_spin_key(), sector.key)] += sector.value;
  }
  for (const auto& sector : payload.mixed_sectors) {
    if (std::abs(sector.value) <= 1.0e-15) {
      continue;
    }
    joint_payload.sectors[sector.key] += sector.value;
  }
  canonicalize_joint_payload_basis_keys(&joint_payload);
  cleanup_joint_payload(&joint_payload);
  return joint_payload;
}

HamiltonianBoundaryPayload project_joint_payload_to_hamiltonian(
    const JointDeletionPayload& payload) {
  HamiltonianBoundaryPayload projected = make_zero_hamiltonian_payload(
      payload.alpha_row_count,
      payload.alpha_col_count,
      payload.beta_row_count,
      payload.beta_col_count);
  for (const auto& key : payload.basis_keys) {
    if (is_empty_spin_key(key.alpha_key) && is_empty_spin_key(key.beta_key)) {
      projected.has_overlap_basis = true;
      continue;
    }
    if (is_empty_spin_key(key.beta_key) &&
        !is_empty_spin_key(key.alpha_key) &&
        sector_degree(key.alpha_key) <= 2) {
      projected.alpha_basis_keys.push_back(key.alpha_key);
      continue;
    }
    if (is_empty_spin_key(key.alpha_key) &&
        !is_empty_spin_key(key.beta_key) &&
        sector_degree(key.beta_key) <= 2) {
      projected.beta_basis_keys.push_back(key.beta_key);
      continue;
    }
    if (!is_empty_spin_key(key.alpha_key) &&
        !is_empty_spin_key(key.beta_key) &&
        sector_degree(key.alpha_key) <= 1 &&
        sector_degree(key.beta_key) <= 1) {
      projected.mixed_basis_keys.push_back(key);
    }
  }
  for (const auto& [key, value] : payload.sectors) {
    if (std::abs(value) <= 1.0e-15) {
      continue;
    }
    if (is_empty_spin_key(key.alpha_key) && is_empty_spin_key(key.beta_key)) {
      projected.overlap += value;
      continue;
    }
    if (is_empty_spin_key(key.beta_key) &&
        !is_empty_spin_key(key.alpha_key) &&
        sector_degree(key.alpha_key) <= 2) {
      projected.alpha_sectors.push_back(SameSpinDeletedSectorValue{
          .key = key.alpha_key,
          .value = value,
      });
      continue;
    }
    if (is_empty_spin_key(key.alpha_key) &&
        !is_empty_spin_key(key.beta_key) &&
        sector_degree(key.beta_key) <= 2) {
      projected.beta_sectors.push_back(SameSpinDeletedSectorValue{
          .key = key.beta_key,
          .value = value,
      });
      continue;
    }
    if (!is_empty_spin_key(key.alpha_key) &&
        !is_empty_spin_key(key.beta_key) &&
        sector_degree(key.alpha_key) <= 1 &&
        sector_degree(key.beta_key) <= 1) {
      projected.mixed_sectors.push_back(HamiltonianMixedDeletedSectorValue{
          .key = key,
          .value = value,
      });
    }
  }
  cleanup_hamiltonian_payload(&projected);
  return projected;
}

HamiltonianBoundaryPayload build_hamiltonian_boundary_payload_values(
    const SpinDeletionPayload& alpha_payload,
    const SpinDeletionPayload& beta_payload) {
  HamiltonianBoundaryPayload payload = make_zero_hamiltonian_payload(
      alpha_payload.row_count,
      alpha_payload.col_count,
      beta_payload.row_count,
      beta_payload.col_count);
  const double alpha_overlap = spin_payload_overlap_value(alpha_payload);
  const double beta_overlap = spin_payload_overlap_value(beta_payload);
  payload.overlap = alpha_overlap * beta_overlap;

  if (std::abs(beta_overlap) > 1.0e-15) {
    payload.alpha_sectors.reserve(alpha_payload.sectors.size());
    for (const auto& [key, value] : alpha_payload.sectors) {
      if (is_empty_spin_key(key) ||
          sector_degree(key) > 2 ||
          std::abs(value) <= 1.0e-15) {
        continue;
      }
      payload.alpha_sectors.push_back(SameSpinDeletedSectorValue{
          .key = key,
          .value = beta_overlap * value,
      });
    }
  }
  if (std::abs(alpha_overlap) > 1.0e-15) {
    payload.beta_sectors.reserve(beta_payload.sectors.size());
    for (const auto& [key, value] : beta_payload.sectors) {
      if (is_empty_spin_key(key) ||
          sector_degree(key) > 2 ||
          std::abs(value) <= 1.0e-15) {
        continue;
      }
      payload.beta_sectors.push_back(SameSpinDeletedSectorValue{
          .key = key,
          .value = alpha_overlap * value,
      });
    }
  }

  payload.mixed_sectors.reserve(
      alpha_payload.sectors.size() * beta_payload.sectors.size());
  for (const auto& [alpha_key, alpha_value] : alpha_payload.sectors) {
    if (is_empty_spin_key(alpha_key) ||
        sector_degree(alpha_key) > 1 ||
        std::abs(alpha_value) <= 1.0e-15) {
      continue;
    }
    for (const auto& [beta_key, beta_value] : beta_payload.sectors) {
      if (is_empty_spin_key(beta_key) ||
          sector_degree(beta_key) > 1 ||
          std::abs(beta_value) <= 1.0e-15) {
        continue;
      }
      payload.mixed_sectors.push_back(HamiltonianMixedDeletedSectorValue{
          .key = make_joint_key(alpha_key, beta_key),
          .value = alpha_value * beta_value,
      });
    }
  }

  cleanup_hamiltonian_payload(&payload);
  return payload;
}

HamiltonianBoundaryPayload build_hamiltonian_boundary_payload(
    const SpinDeletionPayload& alpha_payload,
    const SpinDeletionPayload& beta_payload) {
  return project_joint_payload_to_hamiltonian(
      build_joint_payload(alpha_payload, beta_payload));
}

struct HamiltonianSpinProjectedPayloadValues {
  int row_count = 0;
  int col_count = 0;
  double overlap = 0.0;
  std::vector<SameSpinDeletedSectorValue> same_spin_sectors;
  std::vector<SameSpinDeletedSectorValue> first_order_sectors;
};

struct HamiltonianParentSectorSelection {
  const std::vector<int>* selected_rows = nullptr;
  const std::vector<int>* selected_cols = nullptr;
  int row_count = 0;
  int col_count = 0;
};

std::size_t flatten_state_sector_index(
    int state_index,
    int sector_count,
    int flat_sector_index) {
  if (state_index < 0 || sector_count < 0 || flat_sector_index < 0) {
    throw std::invalid_argument("state/sector indices must be non-negative");
  }
  return xmvb::to_size(state_index) *
      xmvb::to_size(sector_count) +
      xmvb::to_size(flat_sector_index);
}

HamiltonianSpinProjectedPayloadValues project_spin_payload_to_hamiltonian_values(
    const SpinDeletionPayload& payload) {
  // Converts one cached one-spin deleted-minor payload into the exact typed
  // values needed by the forward Hamiltonian hot path:
  // - closed overlap,
  // - degree-<=2 same-spin sectors,
  // - degree-1 sectors for mixed alpha/beta products.
  //
  // The input payload is already canonicalized by the deleted-minor builder, so
  // the projected vectors can stay in sparse vector form without another map
  // cleanup pass.
  HamiltonianSpinProjectedPayloadValues projected;
  projected.row_count = payload.row_count;
  projected.col_count = payload.col_count;
  projected.overlap = spin_payload_overlap_value(payload);
  projected.same_spin_sectors.reserve(payload.sectors.size());
  projected.first_order_sectors.reserve(payload.sectors.size());
  for (const auto& [key, value] : payload.sectors) {
    if (is_empty_spin_key(key) || std::abs(value) <= 1.0e-15) {
      continue;
    }
    const int degree = sector_degree(key);
    if (degree > 2) {
      continue;
    }
    projected.same_spin_sectors.push_back(SameSpinDeletedSectorValue{
        .key = key,
        .value = value,
    });
    if (degree <= 1) {
      projected.first_order_sectors.push_back(SameSpinDeletedSectorValue{
          .key = key,
          .value = value,
      });
    }
  }
  return projected;
}

bool has_projected_spin_payload_values(
    const HamiltonianSpinProjectedPayloadValues& projected) {
  return std::abs(projected.overlap) > 1.0e-15 ||
      !projected.same_spin_sectors.empty() ||
      !projected.first_order_sectors.empty();
}

HamiltonianBoundaryPayload build_hamiltonian_boundary_payload_from_projected_spin_values(
    const HamiltonianSpinProjectedPayloadValues& alpha_payload,
    const HamiltonianSpinProjectedPayloadValues& beta_payload) {
  // Fast exact spin-product projection used by the leaf base case.
  //
  // This is the typed forward Hamiltonian algebra specialized to one already
  // projected alpha payload and one projected beta payload. It avoids repeatedly
  // scanning the underlying `std::map` deleted-minor storage when the same
  // one-spin payload is reused across many local alpha/beta state combinations.
  HamiltonianBoundaryPayload payload = make_zero_hamiltonian_payload(
      alpha_payload.row_count,
      alpha_payload.col_count,
      beta_payload.row_count,
      beta_payload.col_count);
  payload.overlap = alpha_payload.overlap * beta_payload.overlap;

  if (std::abs(beta_payload.overlap) > 1.0e-15) {
    payload.alpha_sectors.reserve(alpha_payload.same_spin_sectors.size());
    for (const auto& sector : alpha_payload.same_spin_sectors) {
      payload.alpha_sectors.push_back(SameSpinDeletedSectorValue{
          .key = sector.key,
          .value = beta_payload.overlap * sector.value,
      });
    }
  }
  if (std::abs(alpha_payload.overlap) > 1.0e-15) {
    payload.beta_sectors.reserve(beta_payload.same_spin_sectors.size());
    for (const auto& sector : beta_payload.same_spin_sectors) {
      payload.beta_sectors.push_back(SameSpinDeletedSectorValue{
          .key = sector.key,
          .value = alpha_payload.overlap * sector.value,
      });
    }
  }

  payload.mixed_sectors.reserve(
      alpha_payload.first_order_sectors.size() *
      beta_payload.first_order_sectors.size());
  for (const auto& alpha_sector : alpha_payload.first_order_sectors) {
    for (const auto& beta_sector : beta_payload.first_order_sectors) {
      payload.mixed_sectors.push_back(HamiltonianMixedDeletedSectorValue{
          .key = make_joint_key(alpha_sector.key, beta_sector.key),
          .value = alpha_sector.value * beta_sector.value,
      });
    }
  }
  return payload;
}

HamiltonianEntryDensePayload
build_hamiltonian_entry_dense_payload_from_projected_spin_values(
    const HamiltonianSpinProjectedPayloadValues& alpha_payload,
    const HamiltonianSpinProjectedPayloadValues& beta_payload,
    int support_size) {
  HamiltonianEntryDensePayload payload =
      make_zero_hamiltonian_entry_dense_payload(
          alpha_payload.row_count,
          alpha_payload.col_count,
          beta_payload.row_count,
          beta_payload.col_count,
          support_size);
  payload.overlap = alpha_payload.overlap * beta_payload.overlap;

  if (std::abs(beta_payload.overlap) > kHamiltonianDenseZeroTolerance) {
    for (const auto& sector : alpha_payload.same_spin_sectors) {
      static_cast<void>(accumulate_dense_spin_deleted_sector(
          sector.key,
          beta_payload.overlap * sector.value,
          payload.alpha_row_count,
          payload.alpha_col_count,
          payload.support_size,
          &payload.alpha_degree1,
          &payload.alpha_degree2));
    }
  }
  if (std::abs(alpha_payload.overlap) > kHamiltonianDenseZeroTolerance) {
    for (const auto& sector : beta_payload.same_spin_sectors) {
      static_cast<void>(accumulate_dense_spin_deleted_sector(
          sector.key,
          alpha_payload.overlap * sector.value,
          payload.beta_row_count,
          payload.beta_col_count,
          payload.support_size,
          &payload.beta_degree1,
          &payload.beta_degree2));
    }
  }

  for (const auto& alpha_sector : alpha_payload.first_order_sectors) {
    if (std::abs(alpha_sector.value) <= kHamiltonianDenseZeroTolerance) {
      continue;
    }
    for (const auto& beta_sector : beta_payload.first_order_sectors) {
      if (std::abs(beta_sector.value) <= kHamiltonianDenseZeroTolerance) {
        continue;
      }
      static_cast<void>(accumulate_dense_mixed_deleted_sector(
          alpha_sector.key,
          beta_sector.key,
          alpha_sector.value * beta_sector.value,
          &payload));
    }
  }
  return payload;
}

HamiltonianBoundaryPayload project_dense_hamiltonian_entry_payload_to_boundary_values(
    const HamiltonianEntryDensePayload& payload) {
  HamiltonianBoundaryPayload projected = make_zero_hamiltonian_payload(
      payload.alpha_row_count,
      payload.alpha_col_count,
      payload.beta_row_count,
      payload.beta_col_count);
  projected.overlap = payload.overlap;

  for (int index = 0; index < static_cast<int>(payload.alpha_degree1.size()); ++index) {
    const double value = payload.alpha_degree1[xmvb::to_size(index)];
    if (std::abs(value) <= kHamiltonianDenseZeroTolerance) {
      continue;
    }
    SpinDeletionKey key;
    decode_dense_spin_degree1_key(
        payload.alpha_row_count,
        payload.alpha_col_count,
        index,
        payload.support_size,
        &key);
    projected.alpha_sectors.push_back(SameSpinDeletedSectorValue{
        .key = std::move(key),
        .value = value,
    });
  }
  for (int index = 0; index < static_cast<int>(payload.alpha_degree2.size()); ++index) {
    const double value = payload.alpha_degree2[xmvb::to_size(index)];
    if (std::abs(value) <= kHamiltonianDenseZeroTolerance) {
      continue;
    }
    SpinDeletionKey key;
    decode_dense_spin_degree2_key(
        payload.alpha_row_count,
        payload.alpha_col_count,
        index,
        payload.support_size,
        &key);
    projected.alpha_sectors.push_back(SameSpinDeletedSectorValue{
        .key = std::move(key),
        .value = value,
    });
  }
  for (int index = 0; index < static_cast<int>(payload.beta_degree1.size()); ++index) {
    const double value = payload.beta_degree1[xmvb::to_size(index)];
    if (std::abs(value) <= kHamiltonianDenseZeroTolerance) {
      continue;
    }
    SpinDeletionKey key;
    decode_dense_spin_degree1_key(
        payload.beta_row_count,
        payload.beta_col_count,
        index,
        payload.support_size,
        &key);
    projected.beta_sectors.push_back(SameSpinDeletedSectorValue{
        .key = std::move(key),
        .value = value,
    });
  }
  for (int index = 0; index < static_cast<int>(payload.beta_degree2.size()); ++index) {
    const double value = payload.beta_degree2[xmvb::to_size(index)];
    if (std::abs(value) <= kHamiltonianDenseZeroTolerance) {
      continue;
    }
    SpinDeletionKey key;
    decode_dense_spin_degree2_key(
        payload.beta_row_count,
        payload.beta_col_count,
        index,
        payload.support_size,
        &key);
    projected.beta_sectors.push_back(SameSpinDeletedSectorValue{
        .key = std::move(key),
        .value = value,
    });
  }

  const int alpha_degree1_size = dense_spin_degree1_size(
      payload.alpha_row_count,
      payload.alpha_col_count,
      payload.support_size);
  const int beta_degree1_size = dense_spin_degree1_size(
      payload.beta_row_count,
      payload.beta_col_count,
      payload.support_size);
  for (int beta_index = 0; beta_index < beta_degree1_size; ++beta_index) {
    for (int alpha_index = 0; alpha_index < alpha_degree1_size; ++alpha_index) {
      const double value = payload.mixed_degree1[xmvb::to_size(
          dense_mixed_degree1_flat_index(payload, alpha_index, beta_index))];
      if (std::abs(value) <= kHamiltonianDenseZeroTolerance) {
        continue;
      }
      SpinDeletionKey alpha_key;
      SpinDeletionKey beta_key;
      decode_dense_spin_degree1_key(
          payload.alpha_row_count,
          payload.alpha_col_count,
          alpha_index,
          payload.support_size,
          &alpha_key);
      decode_dense_spin_degree1_key(
          payload.beta_row_count,
          payload.beta_col_count,
          beta_index,
          payload.support_size,
          &beta_key);
      projected.mixed_sectors.push_back(HamiltonianMixedDeletedSectorValue{
          .key = make_joint_key(std::move(alpha_key), std::move(beta_key)),
          .value = value,
      });
    }
  }

  cleanup_hamiltonian_payload(&projected);
  return projected;
}

void add_scaled_hamiltonian_payload(
    const HamiltonianBoundaryPayload& source,
    double scale,
    HamiltonianBoundaryPayload* destination) {
  // Accumulates one typed Hamiltonian payload into another while keeping all
  // active sector lists sparse until the final cleanup.
  if (destination == nullptr) {
    throw std::invalid_argument("destination must not be null");
  }
  if (std::abs(scale) <= 1.0e-15 || !is_hamiltonian_payload_nonzero(source)) {
    return;
  }

  const bool destination_empty =
      destination->alpha_row_count == 0 &&
      destination->alpha_col_count == 0 &&
      destination->beta_row_count == 0 &&
      destination->beta_col_count == 0 &&
      !destination->has_overlap_basis &&
      destination->alpha_basis_keys.empty() &&
      destination->beta_basis_keys.empty() &&
      destination->mixed_basis_keys.empty() &&
      std::abs(destination->overlap) <= 1.0e-15 &&
      destination->alpha_sectors.empty() &&
      destination->beta_sectors.empty() &&
      destination->mixed_sectors.empty();
  if (destination_empty) {
    destination->alpha_row_count = source.alpha_row_count;
    destination->alpha_col_count = source.alpha_col_count;
    destination->beta_row_count = source.beta_row_count;
    destination->beta_col_count = source.beta_col_count;
  }

  destination->has_overlap_basis =
      destination->has_overlap_basis || source.has_overlap_basis;
  destination->alpha_basis_keys.insert(
      destination->alpha_basis_keys.end(),
      source.alpha_basis_keys.begin(),
      source.alpha_basis_keys.end());
  destination->beta_basis_keys.insert(
      destination->beta_basis_keys.end(),
      source.beta_basis_keys.begin(),
      source.beta_basis_keys.end());
  destination->mixed_basis_keys.insert(
      destination->mixed_basis_keys.end(),
      source.mixed_basis_keys.begin(),
      source.mixed_basis_keys.end());
  destination->overlap += scale * source.overlap;
  destination->alpha_sectors.reserve(
      destination->alpha_sectors.size() + source.alpha_sectors.size());
  for (const auto& sector : source.alpha_sectors) {
    if (std::abs(sector.value) <= 1.0e-15) {
      continue;
    }
    destination->alpha_sectors.push_back(SameSpinDeletedSectorValue{
        .key = sector.key,
        .value = scale * sector.value,
    });
  }
  destination->beta_sectors.reserve(
      destination->beta_sectors.size() + source.beta_sectors.size());
  for (const auto& sector : source.beta_sectors) {
    if (std::abs(sector.value) <= 1.0e-15) {
      continue;
    }
    destination->beta_sectors.push_back(SameSpinDeletedSectorValue{
        .key = sector.key,
        .value = scale * sector.value,
    });
  }
  destination->mixed_sectors.reserve(
      destination->mixed_sectors.size() + source.mixed_sectors.size());
  for (const auto& sector : source.mixed_sectors) {
    if (std::abs(sector.value) <= 1.0e-15) {
      continue;
    }
    destination->mixed_sectors.push_back(HamiltonianMixedDeletedSectorValue{
        .key = sector.key,
        .value = scale * sector.value,
    });
  }

}

void add_scaled_hamiltonian_payload_preserve_basis(
    const HamiltonianBoundaryPayload& source,
    double scale,
    bool preserve_structural_zero_messages,
    HamiltonianBoundaryPayload* destination) {
  if (destination == nullptr) {
    throw std::invalid_argument("destination must not be null");
  }

  const bool contribute_basis =
      preserve_structural_zero_messages &&
      has_hamiltonian_payload_basis(source);
  const bool contribute_value =
      std::abs(scale) > 1.0e-15 &&
      is_hamiltonian_payload_nonzero(source);
  if (!contribute_basis && !contribute_value) {
    return;
  }

  const bool destination_empty =
      destination->alpha_row_count == 0 &&
      destination->alpha_col_count == 0 &&
      destination->beta_row_count == 0 &&
      destination->beta_col_count == 0 &&
      !destination->has_overlap_basis &&
      destination->alpha_basis_keys.empty() &&
      destination->beta_basis_keys.empty() &&
      destination->mixed_basis_keys.empty() &&
      std::abs(destination->overlap) <= 1.0e-15 &&
      destination->alpha_sectors.empty() &&
      destination->beta_sectors.empty() &&
      destination->mixed_sectors.empty();
  if (destination_empty) {
    destination->alpha_row_count = source.alpha_row_count;
    destination->alpha_col_count = source.alpha_col_count;
    destination->beta_row_count = source.beta_row_count;
    destination->beta_col_count = source.beta_col_count;
  }

  if (contribute_basis) {
    destination->has_overlap_basis =
        destination->has_overlap_basis || source.has_overlap_basis;
    destination->alpha_basis_keys.insert(
        destination->alpha_basis_keys.end(),
        source.alpha_basis_keys.begin(),
        source.alpha_basis_keys.end());
    destination->beta_basis_keys.insert(
        destination->beta_basis_keys.end(),
        source.beta_basis_keys.begin(),
        source.beta_basis_keys.end());
    destination->mixed_basis_keys.insert(
        destination->mixed_basis_keys.end(),
        source.mixed_basis_keys.begin(),
        source.mixed_basis_keys.end());
  }

  if (contribute_value) {
    add_scaled_hamiltonian_payload(source, scale, destination);
  }
}

BoundaryHamiltonianMessage make_zero_boundary_hamiltonian_message_like(
    const BoundaryHamiltonianMessage& source) {
  BoundaryHamiltonianMessage message;
  message.alpha_layout = source.alpha_layout;
  message.beta_layout = source.beta_layout;
  message.payload_values.resize(
      xmvb::to_size(message.alpha_layout.total_sector_count()) *
      xmvb::to_size(message.beta_layout.total_sector_count()));
  return message;
}

bool has_frontier_hamiltonian_payload(
    const HamiltonianBoundaryPayload& payload,
    bool preserve_structural_zero_messages) {
  return
      preserve_structural_zero_messages
          ? has_hamiltonian_payload_basis(payload)
          : is_hamiltonian_payload_nonzero(payload);
}

FrontierHamiltonianMessage finalize_frontier_hamiltonian_message(
    int alpha_row_count,
    int alpha_col_count,
    int beta_row_count,
    int beta_col_count,
    std::map<FrontierHamiltonianMaskKey, HamiltonianBoundaryPayload>* payloads,
    bool preserve_structural_zero_messages) {
  if (payloads == nullptr) {
    throw std::invalid_argument("frontier payload map must not be null");
  }

  FrontierHamiltonianMessage message;
  message.alpha_row_count = alpha_row_count;
  message.alpha_col_count = alpha_col_count;
  message.beta_row_count = beta_row_count;
  message.beta_col_count = beta_col_count;
  message.entries.reserve(payloads->size());
  for (auto& [key, payload] : *payloads) {
    cleanup_hamiltonian_payload(&payload);
    if (!has_frontier_hamiltonian_payload(
            payload,
            preserve_structural_zero_messages)) {
      continue;
    }
    message.entries.push_back(FrontierHamiltonianEntry{
        .key = key,
        .payload = std::move(payload),
    });
  }
  return message;
}

FrontierHamiltonianMessage make_identity_frontier_hamiltonian_message(
    int alpha_row_count,
    int alpha_col_count,
    int beta_row_count,
    int beta_col_count) {
  FrontierHamiltonianMessage message;
  message.alpha_row_count = alpha_row_count;
  message.alpha_col_count = alpha_col_count;
  message.beta_row_count = beta_row_count;
  message.beta_col_count = beta_col_count;
  message.entries.push_back(FrontierHamiltonianEntry{
      .key = FrontierHamiltonianMaskKey{},
      .payload = make_identity_hamiltonian_payload(),
  });
  return message;
}

int frontier_child_merge_parity(
    const FrontierHamiltonianMaskKey& prefix_key,
    const FrontierHamiltonianMaskKey& child_key,
    int alpha_row_count,
    int alpha_col_count,
    int beta_row_count,
    int beta_col_count,
    int child_right_alpha_size,
    int child_left_alpha_size,
    int child_right_beta_size,
    int child_left_beta_size) {
  int parity = 0;
  parity ^= incremental_component_block_parity(
      alpha_row_count,
      prefix_key.alpha_row_mask,
      child_key.alpha_row_mask,
      child_right_alpha_size);
  parity ^= incremental_component_block_parity(
      alpha_col_count,
      prefix_key.alpha_col_mask,
      child_key.alpha_col_mask,
      child_left_alpha_size);
  parity ^= incremental_component_block_parity(
      beta_row_count,
      prefix_key.beta_row_mask,
      child_key.beta_row_mask,
      child_right_beta_size);
  parity ^= incremental_component_block_parity(
      beta_col_count,
      prefix_key.beta_col_mask,
      child_key.beta_col_mask,
      child_left_beta_size);
  return parity;
}

FrontierHamiltonianMessage merge_frontier_hamiltonian_message_with_child(
    const FrontierHamiltonianMessage& frontier,
    const BoundaryHamiltonianMessage& child_message,
    int child_right_alpha_size,
    int child_left_alpha_size,
    int child_right_beta_size,
    int child_left_beta_size,
    bool preserve_structural_zero_messages,
    std::uint64_t* dp_transition_count) {
  if (dp_transition_count == nullptr) {
    throw std::invalid_argument("dp_transition_count must not be null");
  }

  std::map<FrontierHamiltonianMaskKey, HamiltonianBoundaryPayload> merged_payloads;
  for (const FrontierHamiltonianEntry& prefix_entry : frontier.entries) {
    for (const auto& child_entry : child_message.nonzero_entries) {
      const BoundarySector& child_alpha_sector =
          boundary_layout_sector(
              child_message.alpha_layout,
              child_entry.alpha_flat_sector_index);
      const BoundarySector& child_beta_sector =
          boundary_layout_sector(
              child_message.beta_layout,
              child_entry.beta_flat_sector_index);
      if ((prefix_entry.key.alpha_row_mask & child_alpha_sector.row_mask) != 0U ||
          (prefix_entry.key.alpha_col_mask & child_alpha_sector.col_mask) != 0U ||
          (prefix_entry.key.beta_row_mask & child_beta_sector.row_mask) != 0U ||
          (prefix_entry.key.beta_col_mask & child_beta_sector.col_mask) != 0U) {
        continue;
      }

      ++(*dp_transition_count);
      const FrontierHamiltonianMaskKey child_key{
          .alpha_row_mask = child_alpha_sector.row_mask,
          .alpha_col_mask = child_alpha_sector.col_mask,
          .beta_row_mask = child_beta_sector.row_mask,
          .beta_col_mask = child_beta_sector.col_mask,
      };
      const FrontierHamiltonianMaskKey merged_key{
          .alpha_row_mask =
              prefix_entry.key.alpha_row_mask | child_alpha_sector.row_mask,
          .alpha_col_mask =
              prefix_entry.key.alpha_col_mask | child_alpha_sector.col_mask,
          .beta_row_mask =
              prefix_entry.key.beta_row_mask | child_beta_sector.row_mask,
          .beta_col_mask =
              prefix_entry.key.beta_col_mask | child_beta_sector.col_mask,
      };
      const int parity = frontier_child_merge_parity(
          prefix_entry.key,
          child_key,
          frontier.alpha_row_count,
          frontier.alpha_col_count,
          frontier.beta_row_count,
          frontier.beta_col_count,
          child_right_alpha_size,
          child_left_alpha_size,
          child_right_beta_size,
          child_left_beta_size);
      const HamiltonianBoundaryPayload merged_payload =
          preserve_structural_zero_messages
              ? merge_hamiltonian_boundary_payloads_limited(
                    prefix_entry.payload,
                    child_message.payload_values[xmvb::to_size(
                        child_entry.flat_index)],
                    2)
              : merge_hamiltonian_boundary_payloads_limited_values(
                    prefix_entry.payload,
                    child_message.payload_values[xmvb::to_size(
                        child_entry.flat_index)],
                    2);
      add_scaled_hamiltonian_payload_preserve_basis(
          merged_payload,
          parity_sign(parity),
          preserve_structural_zero_messages,
          &merged_payloads[merged_key]);
    }
  }
  return finalize_frontier_hamiltonian_message(
      frontier.alpha_row_count,
      frontier.alpha_col_count,
      frontier.beta_row_count,
      frontier.beta_col_count,
      &merged_payloads,
      preserve_structural_zero_messages);
}

FrontierHamiltonianMessage build_frontier_hamiltonian_message(
    const std::vector<const BoundaryHamiltonianMessage*>& child_messages,
    const ChildSpinSizeVectors& child_spin_sizes,
    int alpha_row_count,
    int alpha_col_count,
    int beta_row_count,
    int beta_col_count,
    bool preserve_structural_zero_messages,
    std::uint64_t* dp_transition_count) {
  if (dp_transition_count == nullptr) {
    throw std::invalid_argument("dp_transition_count must not be null");
  }

  FrontierHamiltonianMessage frontier = make_identity_frontier_hamiltonian_message(
      alpha_row_count,
      alpha_col_count,
      beta_row_count,
      beta_col_count);
  for (std::size_t child_index = 0; child_index < child_messages.size(); ++child_index) {
    frontier = merge_frontier_hamiltonian_message_with_child(
        frontier,
        *child_messages[child_index],
        child_spin_sizes.right_alpha[child_index],
        child_spin_sizes.left_alpha[child_index],
        child_spin_sizes.right_beta[child_index],
        child_spin_sizes.left_beta[child_index],
        preserve_structural_zero_messages,
        dp_transition_count);
  }
  return frontier;
}

DenseFrontierHamiltonianMessage finalize_dense_frontier_hamiltonian_message(
    int alpha_row_count,
    int alpha_col_count,
    int beta_row_count,
    int beta_col_count,
    int support_size,
    std::map<FrontierHamiltonianMaskKey, HamiltonianEntryDensePayload>* payloads) {
  if (payloads == nullptr) {
    throw std::invalid_argument("dense frontier payload map must not be null");
  }
  DenseFrontierHamiltonianMessage message;
  message.alpha_row_count = alpha_row_count;
  message.alpha_col_count = alpha_col_count;
  message.beta_row_count = beta_row_count;
  message.beta_col_count = beta_col_count;
  message.support_size = support_size;
  message.entries.reserve(payloads->size());
  for (auto& [key, payload] : *payloads) {
    cleanup_hamiltonian_entry_dense_payload(&payload);
    if (!is_hamiltonian_entry_dense_payload_nonzero(payload)) {
      continue;
    }
    message.entries.push_back(DenseFrontierHamiltonianEntry{
        .key = key,
        .payload = std::move(payload),
    });
  }
  return message;
}

DenseFrontierHamiltonianMessage make_identity_dense_frontier_hamiltonian_message(
    int alpha_row_count,
    int alpha_col_count,
    int beta_row_count,
    int beta_col_count,
    int support_size) {
  DenseFrontierHamiltonianMessage message;
  message.alpha_row_count = alpha_row_count;
  message.alpha_col_count = alpha_col_count;
  message.beta_row_count = beta_row_count;
  message.beta_col_count = beta_col_count;
  message.support_size = support_size;
  HamiltonianEntryDensePayload identity =
      make_zero_hamiltonian_entry_dense_payload(
          0,
          0,
          0,
          0,
          support_size);
  identity.overlap = 1.0;
  message.entries.push_back(DenseFrontierHamiltonianEntry{
      .key = FrontierHamiltonianMaskKey{},
      .payload = std::move(identity),
  });
  return message;
}

DenseFrontierHamiltonianMessage merge_frontier_dense_hamiltonian_message_with_child(
    const DenseFrontierHamiltonianMessage& frontier,
    const DenseBoundaryHamiltonianMessage& child_message,
    int child_right_alpha_size,
    int child_left_alpha_size,
    int child_right_beta_size,
    int child_left_beta_size,
    std::uint64_t* dp_transition_count) {
  if (dp_transition_count == nullptr) {
    throw std::invalid_argument("dp_transition_count must not be null");
  }
  std::map<FrontierHamiltonianMaskKey, HamiltonianEntryDensePayload> merged_payloads;
  for (const DenseFrontierHamiltonianEntry& prefix_entry : frontier.entries) {
    for (const auto& child_entry : child_message.nonzero_entries) {
      const BoundarySector& child_alpha_sector =
          boundary_layout_sector(
              child_message.alpha_layout,
              child_entry.alpha_flat_sector_index);
      const BoundarySector& child_beta_sector =
          boundary_layout_sector(
              child_message.beta_layout,
              child_entry.beta_flat_sector_index);
      if ((prefix_entry.key.alpha_row_mask & child_alpha_sector.row_mask) != 0U ||
          (prefix_entry.key.alpha_col_mask & child_alpha_sector.col_mask) != 0U ||
          (prefix_entry.key.beta_row_mask & child_beta_sector.row_mask) != 0U ||
          (prefix_entry.key.beta_col_mask & child_beta_sector.col_mask) != 0U) {
        continue;
      }

      ++(*dp_transition_count);
      const FrontierHamiltonianMaskKey child_key{
          .alpha_row_mask = child_alpha_sector.row_mask,
          .alpha_col_mask = child_alpha_sector.col_mask,
          .beta_row_mask = child_beta_sector.row_mask,
          .beta_col_mask = child_beta_sector.col_mask,
      };
      const FrontierHamiltonianMaskKey merged_key{
          .alpha_row_mask =
              prefix_entry.key.alpha_row_mask | child_alpha_sector.row_mask,
          .alpha_col_mask =
              prefix_entry.key.alpha_col_mask | child_alpha_sector.col_mask,
          .beta_row_mask =
              prefix_entry.key.beta_row_mask | child_beta_sector.row_mask,
          .beta_col_mask =
              prefix_entry.key.beta_col_mask | child_beta_sector.col_mask,
      };
      const int parity = frontier_child_merge_parity(
          prefix_entry.key,
          child_key,
          frontier.alpha_row_count,
          frontier.alpha_col_count,
          frontier.beta_row_count,
          frontier.beta_col_count,
          child_right_alpha_size,
          child_left_alpha_size,
          child_right_beta_size,
          child_left_beta_size);
      const HamiltonianEntryDensePayload merged_payload =
          merge_hamiltonian_entry_dense_payloads_limited_values(
              prefix_entry.payload,
              child_message.payload_values[xmvb::to_size(
                  child_entry.flat_index)]);
      add_scaled_hamiltonian_entry_dense_payload(
          merged_payload,
          parity_sign(parity),
          &merged_payloads[merged_key]);
    }
  }
  return finalize_dense_frontier_hamiltonian_message(
      frontier.alpha_row_count,
      frontier.alpha_col_count,
      frontier.beta_row_count,
      frontier.beta_col_count,
      frontier.support_size,
      &merged_payloads);
}

DenseFrontierHamiltonianMessage build_frontier_dense_hamiltonian_message(
    const std::vector<const DenseBoundaryHamiltonianMessage*>& child_messages,
    const ChildSpinSizeVectors& child_spin_sizes,
    int alpha_row_count,
    int alpha_col_count,
    int beta_row_count,
    int beta_col_count,
    int support_size,
    std::uint64_t* dp_transition_count) {
  if (dp_transition_count == nullptr) {
    throw std::invalid_argument("dp_transition_count must not be null");
  }
  DenseFrontierHamiltonianMessage frontier =
      make_identity_dense_frontier_hamiltonian_message(
          alpha_row_count,
          alpha_col_count,
          beta_row_count,
          beta_col_count,
          support_size);
  for (std::size_t child_index = 0; child_index < child_messages.size(); ++child_index) {
    frontier = merge_frontier_dense_hamiltonian_message_with_child(
        frontier,
        *child_messages[child_index],
        child_spin_sizes.right_alpha[child_index],
        child_spin_sizes.left_alpha[child_index],
        child_spin_sizes.right_beta[child_index],
        child_spin_sizes.left_beta[child_index],
        dp_transition_count);
  }
  return frontier;
}

BoundaryHamiltonianMessage build_leaf_hamiltonian_subtree_message_values(
    const ComponentData& node_component,
    const BoundarySpinBundleLayout& alpha_layout,
    const BoundarySpinBundleLayout& beta_layout,
    const std::vector<std::vector<int>>& left_parent_alpha_occ_by_mask,
    const std::vector<std::vector<int>>& right_parent_alpha_occ_by_mask,
    const std::vector<std::vector<int>>& left_parent_beta_occ_by_mask,
    const std::vector<std::vector<int>>& right_parent_beta_occ_by_mask,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subtree_term_pair_count,
    std::uint64_t* subtree_message_state_count,
    std::uint64_t* subdeterminant_evaluations,
    std::unordered_map<SpinPayloadCacheKey,
                       SpinDeletionPayload,
                       SpinPayloadCacheKeyHasher>* payload_cache) {
  // Exact leaf base case in unique one-spin state form.
  //
  // A leaf subtree has no child frontier payload, so its message is purely the
  // local alpha/beta interface factor against the parent boundary. Instead of
  // rebuilding the same one-spin deleted-minor payload for every full
  // alpha-beta state pair, build each unique one-spin state/sector payload once
  // and reuse it through the sparse component coefficient operator.
  if (subtree_term_pair_count == nullptr ||
      subtree_message_state_count == nullptr ||
      subdeterminant_evaluations == nullptr ||
      payload_cache == nullptr) {
    throw std::invalid_argument("leaf subtree-message scratch outputs must not be null");
  }

  BoundaryHamiltonianMessage message;
  message.alpha_layout = alpha_layout;
  message.beta_layout = beta_layout;
  message.payload_values.resize(
      xmvb::to_size(alpha_layout.total_sector_count()) *
      xmvb::to_size(beta_layout.total_sector_count()));

  const ComponentSpinCoefficientOperator component_operator =
      build_component_spin_coefficient_operator(node_component);
  *subtree_term_pair_count += static_cast<std::uint64_t>(component_operator.entries.size());
  if (component_operator.entries.empty()) {
    finalize_boundary_hamiltonian_message(&message, false);
    *subtree_message_state_count +=
        static_cast<std::uint64_t>(message.nonzero_entries.size());
    return message;
  }

  std::vector<HamiltonianParentSectorSelection> alpha_parent_selections(
      xmvb::to_size(alpha_layout.total_sector_count()));
  for (int alpha_flat_sector_index = 0;
       alpha_flat_sector_index < alpha_layout.total_sector_count();
       ++alpha_flat_sector_index) {
    const BoundarySector& alpha_sector =
        boundary_layout_sector(alpha_layout, alpha_flat_sector_index);
    HamiltonianParentSectorSelection selection;
    selection.selected_rows =
        &right_parent_alpha_occ_by_mask[xmvb::to_size(alpha_sector.row_mask)];
    selection.selected_cols =
        &left_parent_alpha_occ_by_mask[xmvb::to_size(alpha_sector.col_mask)];
    selection.row_count = static_cast<int>(selection.selected_rows->size());
    selection.col_count = static_cast<int>(selection.selected_cols->size());
    alpha_parent_selections[xmvb::to_size(alpha_flat_sector_index)] = selection;
  }
  std::vector<HamiltonianParentSectorSelection> beta_parent_selections(
      xmvb::to_size(beta_layout.total_sector_count()));
  for (int beta_flat_sector_index = 0;
       beta_flat_sector_index < beta_layout.total_sector_count();
       ++beta_flat_sector_index) {
    const BoundarySector& beta_sector =
        boundary_layout_sector(beta_layout, beta_flat_sector_index);
    HamiltonianParentSectorSelection selection;
    selection.selected_rows =
        &right_parent_beta_occ_by_mask[xmvb::to_size(beta_sector.row_mask)];
    selection.selected_cols =
        &left_parent_beta_occ_by_mask[xmvb::to_size(beta_sector.col_mask)];
    selection.row_count = static_cast<int>(selection.selected_rows->size());
    selection.col_count = static_cast<int>(selection.selected_cols->size());
    beta_parent_selections[xmvb::to_size(beta_flat_sector_index)] = selection;
  }

  const int alpha_state_count =
      static_cast<int>(component_operator.alpha_states.size());
  const int beta_state_count =
      static_cast<int>(component_operator.beta_states.size());
  std::vector<HamiltonianSpinProjectedPayloadValues> alpha_projected_payloads(
      xmvb::to_size(alpha_state_count) *
      xmvb::to_size(alpha_layout.total_sector_count()));
  std::vector<HamiltonianSpinProjectedPayloadValues> beta_projected_payloads(
      xmvb::to_size(beta_state_count) *
      xmvb::to_size(beta_layout.total_sector_count()));
  std::vector<std::vector<int>> alpha_nonzero_sector_indices(
      xmvb::to_size(alpha_state_count));
  std::vector<std::vector<int>> beta_nonzero_sector_indices(
      xmvb::to_size(beta_state_count));

  for (int alpha_state_index = 0; alpha_state_index < alpha_state_count; ++alpha_state_index) {
    const auto& alpha_state =
        component_operator.alpha_states[xmvb::to_size(alpha_state_index)];
    auto& nonzero_indices =
        alpha_nonzero_sector_indices[xmvb::to_size(alpha_state_index)];
    nonzero_indices.reserve(alpha_layout.total_sector_count());
    for (int alpha_flat_sector_index = 0;
         alpha_flat_sector_index < alpha_layout.total_sector_count();
         ++alpha_flat_sector_index) {
      const auto& selection =
          alpha_parent_selections[xmvb::to_size(alpha_flat_sector_index)];
      const SpinDeletionPayload alpha_payload =
          build_exact_frontier_spin_payload_limited(
              *selection.selected_cols,
              alpha_state.left_occ,
              *selection.selected_rows,
              alpha_state.right_occ,
              2,
              overlap_storage,
              n_orbitals,
              overlap_resolver,
              subdeterminant_evaluations,
              payload_cache);
      const HamiltonianSpinProjectedPayloadValues projected =
          project_spin_payload_to_hamiltonian_values(alpha_payload);
      alpha_projected_payloads[flatten_state_sector_index(
          alpha_state_index,
          alpha_layout.total_sector_count(),
          alpha_flat_sector_index)] = projected;
      if (has_projected_spin_payload_values(projected)) {
        nonzero_indices.push_back(alpha_flat_sector_index);
      }
    }
  }

  for (int beta_state_index = 0; beta_state_index < beta_state_count; ++beta_state_index) {
    const auto& beta_state =
        component_operator.beta_states[xmvb::to_size(beta_state_index)];
    auto& nonzero_indices =
        beta_nonzero_sector_indices[xmvb::to_size(beta_state_index)];
    nonzero_indices.reserve(beta_layout.total_sector_count());
    for (int beta_flat_sector_index = 0;
         beta_flat_sector_index < beta_layout.total_sector_count();
         ++beta_flat_sector_index) {
      const auto& selection =
          beta_parent_selections[xmvb::to_size(beta_flat_sector_index)];
      const SpinDeletionPayload beta_payload =
          build_exact_frontier_spin_payload_limited(
              *selection.selected_cols,
              beta_state.left_occ,
              *selection.selected_rows,
              beta_state.right_occ,
              2,
              overlap_storage,
              n_orbitals,
              overlap_resolver,
              subdeterminant_evaluations,
              payload_cache);
      const HamiltonianSpinProjectedPayloadValues projected =
          project_spin_payload_to_hamiltonian_values(beta_payload);
      beta_projected_payloads[flatten_state_sector_index(
          beta_state_index,
          beta_layout.total_sector_count(),
          beta_flat_sector_index)] = projected;
      if (has_projected_spin_payload_values(projected)) {
        nonzero_indices.push_back(beta_flat_sector_index);
      }
    }
  }

  const std::vector<int> empty_occ;
  for (const auto& entry : component_operator.entries) {
    const auto& alpha_state =
        component_operator.alpha_states[xmvb::to_size(entry.alpha_state_index)];
    const auto& beta_state =
        component_operator.beta_states[xmvb::to_size(entry.beta_state_index)];
    const auto& alpha_sector_indices =
        alpha_nonzero_sector_indices[xmvb::to_size(entry.alpha_state_index)];
    const auto& beta_sector_indices =
        beta_nonzero_sector_indices[xmvb::to_size(entry.beta_state_index)];
    for (const int alpha_flat_sector_index : alpha_sector_indices) {
      const auto& alpha_selection = alpha_parent_selections[xmvb::to_size(
          alpha_flat_sector_index)];
      const HamiltonianSpinProjectedPayloadValues& alpha_payload =
          alpha_projected_payloads[flatten_state_sector_index(
              entry.alpha_state_index,
              alpha_layout.total_sector_count(),
              alpha_flat_sector_index)];
      for (const int beta_flat_sector_index : beta_sector_indices) {
        const auto& beta_selection = beta_parent_selections[xmvb::to_size(
            beta_flat_sector_index)];
        const HamiltonianSpinProjectedPayloadValues& beta_payload =
            beta_projected_payloads[flatten_state_sector_index(
                entry.beta_state_index,
                beta_layout.total_sector_count(),
                beta_flat_sector_index)];
        HamiltonianBoundaryPayload interface_payload =
            build_hamiltonian_boundary_payload_from_projected_spin_values(
                alpha_payload,
                beta_payload);
        if (!is_hamiltonian_payload_nonzero(interface_payload)) {
          continue;
        }

        const HamiltonianBoundaryPayload front_convention_payload =
            transform_hamiltonian_interface_block_to_front_values(
                interface_payload,
                alpha_selection.row_count,
                alpha_selection.col_count,
                empty_occ,
                empty_occ,
                alpha_state.right_occ,
                alpha_state.left_occ,
                0,
                0,
                beta_selection.row_count,
                beta_selection.col_count,
                empty_occ,
                empty_occ,
                beta_state.right_occ,
                beta_state.left_occ,
                0,
                0);
        if (!is_hamiltonian_payload_nonzero(front_convention_payload)) {
          continue;
        }

        add_scaled_hamiltonian_payload_preserve_basis(
            front_convention_payload,
            entry.coefficient,
            false,
            &message.payload_values[xmvb::to_size(
                message.flat_index(alpha_flat_sector_index, beta_flat_sector_index))]);
      }
    }
  }

  finalize_boundary_hamiltonian_message(&message, false);
  *subtree_message_state_count +=
      static_cast<std::uint64_t>(message.nonzero_entries.size());
  return message;
}

DenseBoundaryHamiltonianMessage build_leaf_dense_hamiltonian_subtree_message_values(
    const ComponentData& node_component,
    const BoundarySpinBundleLayout& alpha_layout,
    const BoundarySpinBundleLayout& beta_layout,
    const std::vector<std::vector<int>>& left_parent_alpha_occ_by_mask,
    const std::vector<std::vector<int>>& right_parent_alpha_occ_by_mask,
    const std::vector<std::vector<int>>& left_parent_beta_occ_by_mask,
    const std::vector<std::vector<int>>& right_parent_beta_occ_by_mask,
    const std::vector<double>& overlap_storage,
    int support_size,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subtree_term_pair_count,
    std::uint64_t* subtree_message_state_count,
    std::uint64_t* subdeterminant_evaluations,
    std::unordered_map<SpinPayloadCacheKey,
                       SpinDeletionPayload,
                       SpinPayloadCacheKeyHasher>* payload_cache) {
  if (subtree_term_pair_count == nullptr ||
      subtree_message_state_count == nullptr ||
      subdeterminant_evaluations == nullptr ||
      payload_cache == nullptr) {
    throw std::invalid_argument("dense leaf subtree-message scratch outputs must not be null");
  }

  DenseBoundaryHamiltonianMessage message;
  message.alpha_layout = alpha_layout;
  message.beta_layout = beta_layout;
  message.support_size = support_size;
  message.payload_values.resize(
      xmvb::to_size(alpha_layout.total_sector_count()) *
      xmvb::to_size(beta_layout.total_sector_count()));

  const ComponentSpinCoefficientOperator component_operator =
      build_component_spin_coefficient_operator(node_component);
  *subtree_term_pair_count +=
      static_cast<std::uint64_t>(component_operator.entries.size());
  if (component_operator.entries.empty()) {
    finalize_dense_boundary_hamiltonian_message(&message);
    *subtree_message_state_count +=
        static_cast<std::uint64_t>(message.nonzero_entries.size());
    return message;
  }

  std::vector<HamiltonianParentSectorSelection> alpha_parent_selections(
      xmvb::to_size(alpha_layout.total_sector_count()));
  for (int alpha_flat_sector_index = 0;
       alpha_flat_sector_index < alpha_layout.total_sector_count();
       ++alpha_flat_sector_index) {
    const BoundarySector& alpha_sector =
        boundary_layout_sector(alpha_layout, alpha_flat_sector_index);
    HamiltonianParentSectorSelection selection;
    selection.selected_rows =
        &right_parent_alpha_occ_by_mask[xmvb::to_size(alpha_sector.row_mask)];
    selection.selected_cols =
        &left_parent_alpha_occ_by_mask[xmvb::to_size(alpha_sector.col_mask)];
    selection.row_count = static_cast<int>(selection.selected_rows->size());
    selection.col_count = static_cast<int>(selection.selected_cols->size());
    alpha_parent_selections[xmvb::to_size(alpha_flat_sector_index)] = selection;
  }
  std::vector<HamiltonianParentSectorSelection> beta_parent_selections(
      xmvb::to_size(beta_layout.total_sector_count()));
  for (int beta_flat_sector_index = 0;
       beta_flat_sector_index < beta_layout.total_sector_count();
       ++beta_flat_sector_index) {
    const BoundarySector& beta_sector =
        boundary_layout_sector(beta_layout, beta_flat_sector_index);
    HamiltonianParentSectorSelection selection;
    selection.selected_rows =
        &right_parent_beta_occ_by_mask[xmvb::to_size(beta_sector.row_mask)];
    selection.selected_cols =
        &left_parent_beta_occ_by_mask[xmvb::to_size(beta_sector.col_mask)];
    selection.row_count = static_cast<int>(selection.selected_rows->size());
    selection.col_count = static_cast<int>(selection.selected_cols->size());
    beta_parent_selections[xmvb::to_size(beta_flat_sector_index)] = selection;
  }

  const int alpha_state_count =
      static_cast<int>(component_operator.alpha_states.size());
  const int beta_state_count =
      static_cast<int>(component_operator.beta_states.size());
  std::vector<HamiltonianSpinProjectedPayloadValues> alpha_projected_payloads(
      xmvb::to_size(alpha_state_count) *
      xmvb::to_size(alpha_layout.total_sector_count()));
  std::vector<HamiltonianSpinProjectedPayloadValues> beta_projected_payloads(
      xmvb::to_size(beta_state_count) *
      xmvb::to_size(beta_layout.total_sector_count()));
  std::vector<std::vector<int>> alpha_nonzero_sector_indices(
      xmvb::to_size(alpha_state_count));
  std::vector<std::vector<int>> beta_nonzero_sector_indices(
      xmvb::to_size(beta_state_count));

  for (int alpha_state_index = 0; alpha_state_index < alpha_state_count; ++alpha_state_index) {
    const auto& alpha_state =
        component_operator.alpha_states[xmvb::to_size(alpha_state_index)];
    auto& nonzero_indices =
        alpha_nonzero_sector_indices[xmvb::to_size(alpha_state_index)];
    nonzero_indices.reserve(alpha_layout.total_sector_count());
    for (int alpha_flat_sector_index = 0;
         alpha_flat_sector_index < alpha_layout.total_sector_count();
         ++alpha_flat_sector_index) {
      const auto& selection =
          alpha_parent_selections[xmvb::to_size(alpha_flat_sector_index)];
      const SpinDeletionPayload alpha_payload =
          build_exact_frontier_spin_payload_limited(
              *selection.selected_cols,
              alpha_state.left_occ,
              *selection.selected_rows,
              alpha_state.right_occ,
              2,
              overlap_storage,
              support_size,
              overlap_resolver,
              subdeterminant_evaluations,
              payload_cache);
      const HamiltonianSpinProjectedPayloadValues projected =
          project_spin_payload_to_hamiltonian_values(alpha_payload);
      alpha_projected_payloads[flatten_state_sector_index(
          alpha_state_index,
          alpha_layout.total_sector_count(),
          alpha_flat_sector_index)] = projected;
      if (has_projected_spin_payload_values(projected)) {
        nonzero_indices.push_back(alpha_flat_sector_index);
      }
    }
  }

  for (int beta_state_index = 0; beta_state_index < beta_state_count; ++beta_state_index) {
    const auto& beta_state =
        component_operator.beta_states[xmvb::to_size(beta_state_index)];
    auto& nonzero_indices =
        beta_nonzero_sector_indices[xmvb::to_size(beta_state_index)];
    nonzero_indices.reserve(beta_layout.total_sector_count());
    for (int beta_flat_sector_index = 0;
         beta_flat_sector_index < beta_layout.total_sector_count();
         ++beta_flat_sector_index) {
      const auto& selection =
          beta_parent_selections[xmvb::to_size(beta_flat_sector_index)];
      const SpinDeletionPayload beta_payload =
          build_exact_frontier_spin_payload_limited(
              *selection.selected_cols,
              beta_state.left_occ,
              *selection.selected_rows,
              beta_state.right_occ,
              2,
              overlap_storage,
              support_size,
              overlap_resolver,
              subdeterminant_evaluations,
              payload_cache);
      const HamiltonianSpinProjectedPayloadValues projected =
          project_spin_payload_to_hamiltonian_values(beta_payload);
      beta_projected_payloads[flatten_state_sector_index(
          beta_state_index,
          beta_layout.total_sector_count(),
          beta_flat_sector_index)] = projected;
      if (has_projected_spin_payload_values(projected)) {
        nonzero_indices.push_back(beta_flat_sector_index);
      }
    }
  }

  const std::vector<int> empty_occ;
  for (const auto& entry : component_operator.entries) {
    const auto& alpha_state =
        component_operator.alpha_states[xmvb::to_size(entry.alpha_state_index)];
    const auto& beta_state =
        component_operator.beta_states[xmvb::to_size(entry.beta_state_index)];
    const auto& alpha_sector_indices =
        alpha_nonzero_sector_indices[xmvb::to_size(entry.alpha_state_index)];
    const auto& beta_sector_indices =
        beta_nonzero_sector_indices[xmvb::to_size(entry.beta_state_index)];
    for (const int alpha_flat_sector_index : alpha_sector_indices) {
      const auto& alpha_selection = alpha_parent_selections[xmvb::to_size(
          alpha_flat_sector_index)];
      const HamiltonianSpinProjectedPayloadValues& alpha_payload =
          alpha_projected_payloads[flatten_state_sector_index(
              entry.alpha_state_index,
              alpha_layout.total_sector_count(),
              alpha_flat_sector_index)];
      for (const int beta_flat_sector_index : beta_sector_indices) {
        const auto& beta_selection = beta_parent_selections[xmvb::to_size(
            beta_flat_sector_index)];
        const HamiltonianSpinProjectedPayloadValues& beta_payload =
            beta_projected_payloads[flatten_state_sector_index(
                entry.beta_state_index,
                beta_layout.total_sector_count(),
                beta_flat_sector_index)];
        HamiltonianEntryDensePayload interface_payload =
            build_hamiltonian_entry_dense_payload_from_projected_spin_values(
                alpha_payload,
                beta_payload,
                support_size);
        if (!is_hamiltonian_entry_dense_payload_nonzero(interface_payload)) {
          continue;
        }

        const HamiltonianEntryDensePayload front_convention_payload =
            transform_hamiltonian_entry_dense_payload_interface_block_to_front_values(
                interface_payload,
                alpha_selection.row_count,
                alpha_selection.col_count,
                empty_occ,
                empty_occ,
                alpha_state.right_occ,
                alpha_state.left_occ,
                0,
                0,
                beta_selection.row_count,
                beta_selection.col_count,
                empty_occ,
                empty_occ,
                beta_state.right_occ,
                beta_state.left_occ,
                0,
                0);
        if (!is_hamiltonian_entry_dense_payload_nonzero(front_convention_payload)) {
          continue;
        }

        add_scaled_hamiltonian_entry_dense_payload(
            front_convention_payload,
            entry.coefficient,
            &message.payload_values[xmvb::to_size(
                message.flat_index(alpha_flat_sector_index, beta_flat_sector_index))]);
      }
    }
  }

  finalize_dense_boundary_hamiltonian_message(&message);
  *subtree_message_state_count +=
      static_cast<std::uint64_t>(message.nonzero_entries.size());
  return message;
}

void reverse_frontier_hamiltonian_message(
    const std::vector<const BoundaryHamiltonianMessage*>& child_messages,
    const ChildSpinSizeVectors& child_spin_sizes,
    const std::vector<FrontierHamiltonianMessage>& prefix_messages,
    const FrontierHamiltonianMessage& frontier_adjoint,
    std::vector<BoundaryHamiltonianMessage>* child_adjoint_messages,
    FrontierHamiltonianMessage* identity_adjoint) {
  if (child_adjoint_messages == nullptr || identity_adjoint == nullptr) {
    throw std::invalid_argument("frontier reverse outputs must not be null");
  }
  if (child_messages.size() != prefix_messages.size() - 1U ||
      child_messages.size() != child_adjoint_messages->size()) {
    throw std::invalid_argument("frontier reverse inputs have inconsistent sizes");
  }

  std::map<FrontierHamiltonianMaskKey, HamiltonianBoundaryPayload> current_adjoint_map;
  for (const FrontierHamiltonianEntry& entry : frontier_adjoint.entries) {
    add_scaled_hamiltonian_payload_preserve_basis(
        entry.payload,
        1.0,
        true,
        &current_adjoint_map[entry.key]);
  }

  for (std::size_t child_offset = child_messages.size(); child_offset > 0U; --child_offset) {
    const std::size_t child_index = child_offset - 1U;
    const FrontierHamiltonianMessage& prefix_message =
        prefix_messages[child_index];
    const BoundaryHamiltonianMessage& child_message =
        *child_messages[child_index];

    std::map<FrontierHamiltonianMaskKey, HamiltonianBoundaryPayload> next_prefix_adjoint_map;
    BoundaryHamiltonianMessage child_adjoint =
        make_zero_boundary_hamiltonian_message_like(child_message);
    for (const FrontierHamiltonianEntry& prefix_entry : prefix_message.entries) {
      for (const auto& child_entry : child_message.nonzero_entries) {
        const BoundarySector& child_alpha_sector =
            boundary_layout_sector(
                child_message.alpha_layout,
                child_entry.alpha_flat_sector_index);
        const BoundarySector& child_beta_sector =
            boundary_layout_sector(
                child_message.beta_layout,
                child_entry.beta_flat_sector_index);
        if ((prefix_entry.key.alpha_row_mask & child_alpha_sector.row_mask) != 0U ||
            (prefix_entry.key.alpha_col_mask & child_alpha_sector.col_mask) != 0U ||
            (prefix_entry.key.beta_row_mask & child_beta_sector.row_mask) != 0U ||
            (prefix_entry.key.beta_col_mask & child_beta_sector.col_mask) != 0U) {
          continue;
        }

        const FrontierHamiltonianMaskKey merged_key{
            .alpha_row_mask =
                prefix_entry.key.alpha_row_mask | child_alpha_sector.row_mask,
            .alpha_col_mask =
                prefix_entry.key.alpha_col_mask | child_alpha_sector.col_mask,
            .beta_row_mask =
                prefix_entry.key.beta_row_mask | child_beta_sector.row_mask,
            .beta_col_mask =
                prefix_entry.key.beta_col_mask | child_beta_sector.col_mask,
        };
        const auto merged_iterator = current_adjoint_map.find(merged_key);
        if (merged_iterator == current_adjoint_map.end()) {
          continue;
        }
        cleanup_hamiltonian_payload(&merged_iterator->second);
        if (!has_hamiltonian_payload_basis(merged_iterator->second)) {
          continue;
        }

        const FrontierHamiltonianMaskKey child_key{
            .alpha_row_mask = child_alpha_sector.row_mask,
            .alpha_col_mask = child_alpha_sector.col_mask,
            .beta_row_mask = child_beta_sector.row_mask,
            .beta_col_mask = child_beta_sector.col_mask,
        };
        const int parity = frontier_child_merge_parity(
            prefix_entry.key,
            child_key,
            prefix_message.alpha_row_count,
            prefix_message.alpha_col_count,
            prefix_message.beta_row_count,
            prefix_message.beta_col_count,
            child_spin_sizes.right_alpha[child_index],
            child_spin_sizes.left_alpha[child_index],
            child_spin_sizes.right_beta[child_index],
            child_spin_sizes.left_beta[child_index]);
        HamiltonianBoundaryPayload prefix_branch_adjoint =
            make_zero_hamiltonian_payload_like(prefix_entry.payload);
        HamiltonianBoundaryPayload child_branch_adjoint =
            make_zero_hamiltonian_payload_like(
                child_message.payload_values[xmvb::to_size(
                    child_entry.flat_index)]);
        reverse_merge_hamiltonian_boundary_payloads_limited(
            prefix_entry.payload,
            child_message.payload_values[xmvb::to_size(
                child_entry.flat_index)],
            2,
            scale_hamiltonian_payload(merged_iterator->second, parity_sign(parity)),
            &prefix_branch_adjoint,
            &child_branch_adjoint);
        add_scaled_hamiltonian_payload_preserve_basis(
            prefix_branch_adjoint,
            1.0,
            true,
            &next_prefix_adjoint_map[prefix_entry.key]);
        add_scaled_hamiltonian_payload_preserve_basis(
            child_branch_adjoint,
            1.0,
            true,
            &child_adjoint.payload_values[xmvb::to_size(
                child_entry.flat_index)]);
      }
    }

    finalize_boundary_hamiltonian_message(&child_adjoint, true);
    (*child_adjoint_messages)[child_index] = std::move(child_adjoint);
    current_adjoint_map = std::move(next_prefix_adjoint_map);
  }

  *identity_adjoint = finalize_frontier_hamiltonian_message(
      prefix_messages.front().alpha_row_count,
      prefix_messages.front().alpha_col_count,
      prefix_messages.front().beta_row_count,
      prefix_messages.front().beta_col_count,
      &current_adjoint_map,
      true);
}

void append_hamiltonian_mixed_merge_terms(
    const HamiltonianBoundaryPayload& left,
    const HamiltonianBoundaryPayload& right,
    int max_spin_degree,
    std::vector<HamiltonianMixedDeletedSectorValue>* destination) {
  // Exact targeted merge for the mixed alpha-beta Hamiltonian channel.
  //
  // The final mixed sector is first-order in each spin, so the exact source
  // terms are precisely the typed Hamiltonian sectors whose per-spin
  // row/column ranks are at most one. This helper keeps the typed payload
  // representation, but applies the same deleted-minor merge law as the old
  // generic joint payload on this reduced source set.
  if (destination == nullptr) {
    throw std::invalid_argument("destination must not be null");
  }
  if (max_spin_degree < 1) {
    return;
  }

  struct HamiltonianMixedMergeSourceTerm {
    const SpinDeletionKey* alpha_key = nullptr;
    const SpinDeletionKey* beta_key = nullptr;
    double value = 0.0;
  };
  const SpinDeletionKey empty_key = make_spin_key();
  auto build_source_terms =
      [&empty_key](const HamiltonianBoundaryPayload& payload) {
        std::vector<HamiltonianMixedMergeSourceTerm> terms;
        terms.reserve(
            1U + payload.alpha_sectors.size() + payload.beta_sectors.size() +
            payload.mixed_sectors.size());
        if (std::abs(payload.overlap) > 1.0e-15) {
          terms.push_back(HamiltonianMixedMergeSourceTerm{
              .alpha_key = &empty_key,
              .beta_key = &empty_key,
              .value = payload.overlap,
          });
        }
        for (const auto& sector : payload.alpha_sectors) {
          if (std::abs(sector.value) <= 1.0e-15 ||
              is_empty_spin_key(sector.key) ||
              sector_degree(sector.key) > 1) {
            continue;
          }
          terms.push_back(HamiltonianMixedMergeSourceTerm{
              .alpha_key = &sector.key,
              .beta_key = &empty_key,
              .value = sector.value,
          });
        }
        for (const auto& sector : payload.beta_sectors) {
          if (std::abs(sector.value) <= 1.0e-15 ||
              is_empty_spin_key(sector.key) ||
              sector_degree(sector.key) > 1) {
            continue;
          }
          terms.push_back(HamiltonianMixedMergeSourceTerm{
              .alpha_key = &empty_key,
              .beta_key = &sector.key,
              .value = sector.value,
          });
        }
        for (const auto& sector : payload.mixed_sectors) {
          if (std::abs(sector.value) <= 1.0e-15 ||
              is_empty_spin_key(sector.key.alpha_key) ||
              is_empty_spin_key(sector.key.beta_key) ||
              sector_degree(sector.key.alpha_key) > 1 ||
              sector_degree(sector.key.beta_key) > 1) {
            continue;
          }
          terms.push_back(HamiltonianMixedMergeSourceTerm{
              .alpha_key = &sector.key.alpha_key,
              .beta_key = &sector.key.beta_key,
              .value = sector.value,
          });
        }
        return terms;
      };

  const std::vector<HamiltonianMixedMergeSourceTerm> left_terms = build_source_terms(left);
  const std::vector<HamiltonianMixedMergeSourceTerm> right_terms = build_source_terms(right);

  destination->reserve(
      destination->size() + left_terms.size() * right_terms.size());
  for (const auto& left_term : left_terms) {
    if (std::abs(left_term.value) <= 1.0e-15) {
      continue;
    }
    for (const auto& right_term : right_terms) {
      if (std::abs(right_term.value) <= 1.0e-15) {
        continue;
      }

      SpinDeletionKey combined_alpha_key;
      SpinDeletionKey combined_beta_key;
      if (!combine_spin_keys_with_limit(
              *left_term.alpha_key,
              *right_term.alpha_key,
              1,
              &combined_alpha_key) ||
          !combine_spin_keys_with_limit(
              *left_term.beta_key,
              *right_term.beta_key,
              1,
              &combined_beta_key) ||
          is_empty_spin_key(combined_alpha_key) ||
          is_empty_spin_key(combined_beta_key)) {
        continue;
      }

      const int parity =
          spin_sector_merge_parity(
              left.alpha_row_count,
              left.alpha_col_count,
              *left_term.alpha_key,
              right.alpha_row_count,
              right.alpha_col_count,
              *right_term.alpha_key) ^
          spin_sector_merge_parity(
              left.beta_row_count,
              left.beta_col_count,
              *left_term.beta_key,
              right.beta_row_count,
              right.beta_col_count,
              *right_term.beta_key);
      destination->push_back(HamiltonianMixedDeletedSectorValue{
          .key = make_joint_key(
              std::move(combined_alpha_key),
              std::move(combined_beta_key)),
          .value = parity_sign(parity) * left_term.value * right_term.value,
      });
    }
  }
}

HamiltonianBoundaryPayload merge_hamiltonian_boundary_payloads_limited(
    const HamiltonianBoundaryPayload& left,
    const HamiltonianBoundaryPayload& right,
    int max_spin_degree) {
  // Exact typed Hamiltonian boundary-message merge.
  //
  // The typed forward path must stay bitwise consistent with the joint-payload
  // reference used by the reverse pass. Delegate the forward merge to the
  // exact joint deleted-minor algebra first, then project the exact result
  // back to the typed Hamiltonian channels.
  if (max_spin_degree < 0 || max_spin_degree > 2) {
    throw std::invalid_argument("max_spin_degree must be in [0, 2]");
  }
  return project_joint_payload_to_hamiltonian(
      merge_joint_deletion_payloads_limited(
          convert_hamiltonian_payload_to_joint(left),
          convert_hamiltonian_payload_to_joint(right),
          max_spin_degree));
}

void reverse_build_hamiltonian_boundary_payload(
    const SpinDeletionPayload& alpha_payload,
    const SpinDeletionPayload& beta_payload,
    const HamiltonianBoundaryPayload& payload_adjoint,
    SpinDeletionPayload* alpha_adjoint,
    SpinDeletionPayload* beta_adjoint) {
  if (alpha_adjoint == nullptr || beta_adjoint == nullptr) {
    throw std::invalid_argument("hamiltonian payload reverse outputs must not be null");
  }
  reverse_build_joint_payload(
      alpha_payload,
      beta_payload,
      convert_hamiltonian_payload_to_joint(payload_adjoint),
      alpha_adjoint,
      beta_adjoint);
}

void reverse_merge_hamiltonian_boundary_payloads_limited(
    const HamiltonianBoundaryPayload& left,
    const HamiltonianBoundaryPayload& right,
    int max_spin_degree,
    const HamiltonianBoundaryPayload& merged_adjoint,
    HamiltonianBoundaryPayload* left_adjoint,
    HamiltonianBoundaryPayload* right_adjoint) {
  if (left_adjoint == nullptr || right_adjoint == nullptr) {
    throw std::invalid_argument("reverse hamiltonian merge outputs must not be null");
  }
  if (max_spin_degree < 0 || max_spin_degree > 2) {
    throw std::invalid_argument("max_spin_degree must be in [0, 2]");
  }
  JointDeletionPayload left_joint_adjoint =
      convert_hamiltonian_payload_to_joint(*left_adjoint);
  JointDeletionPayload right_joint_adjoint =
      convert_hamiltonian_payload_to_joint(*right_adjoint);
  reverse_merge_joint_deletion_payloads_limited(
      convert_hamiltonian_payload_to_joint(left),
      convert_hamiltonian_payload_to_joint(right),
      max_spin_degree,
      convert_hamiltonian_payload_to_joint(merged_adjoint),
      &left_joint_adjoint,
      &right_joint_adjoint);
  *left_adjoint = project_joint_payload_to_hamiltonian(left_joint_adjoint);
  *right_adjoint = project_joint_payload_to_hamiltonian(right_joint_adjoint);
}

void add_scaled_joint_payload(
    const JointDeletionPayload& source,
    double scale,
    JointDeletionPayload* destination) {
  if (destination == nullptr) {
    throw std::invalid_argument("destination must not be null");
  }
  if (std::abs(scale) <= 1.0e-15) {
    return;
  }
  if (destination->sectors.empty()) {
    destination->alpha_row_count = source.alpha_row_count;
    destination->alpha_col_count = source.alpha_col_count;
    destination->beta_row_count = source.beta_row_count;
    destination->beta_col_count = source.beta_col_count;
  }
  destination->basis_keys.insert(
      destination->basis_keys.end(),
      source.basis_keys.begin(),
      source.basis_keys.end());
  for (const auto& [key, value] : source.sectors) {
    if (std::abs(value) <= 1.0e-15) {
      continue;
    }
    destination->sectors[key] += scale * value;
  }
  canonicalize_joint_payload_basis_keys(destination);
}

void add_scaled_joint_payload_preserve_basis(
    const JointDeletionPayload& source,
    double scale,
    bool preserve_structural_zero_messages,
    JointDeletionPayload* destination) {
  if (destination == nullptr) {
    throw std::invalid_argument("destination must not be null");
  }

  const bool contribute_basis =
      preserve_structural_zero_messages && !source.basis_keys.empty();
  const bool contribute_value =
      std::abs(scale) > 1.0e-15 && is_joint_payload_nonzero(source);
  if (!contribute_basis && !contribute_value) {
    return;
  }

  const bool destination_empty =
      destination->alpha_row_count == 0 &&
      destination->alpha_col_count == 0 &&
      destination->beta_row_count == 0 &&
      destination->beta_col_count == 0 &&
      destination->basis_keys.empty() &&
      destination->sectors.empty();
  if (destination_empty) {
    destination->alpha_row_count = source.alpha_row_count;
    destination->alpha_col_count = source.alpha_col_count;
    destination->beta_row_count = source.beta_row_count;
    destination->beta_col_count = source.beta_col_count;
  }

  if (contribute_basis) {
    destination->basis_keys.insert(
        destination->basis_keys.end(),
        source.basis_keys.begin(),
        source.basis_keys.end());
    canonicalize_joint_payload_basis_keys(destination);
  }
  if (contribute_value) {
    add_scaled_joint_payload(source, scale, destination);
  }
}

bool has_frontier_joint_payload(
    const JointDeletionPayload& payload,
    bool preserve_structural_zero_messages) {
  return preserve_structural_zero_messages
      ? !payload.basis_keys.empty()
      : is_joint_payload_nonzero(payload);
}

std::vector<ComponentTreeLeafMessage> finalize_component_tree_leaf_messages(
    std::map<FrontierHamiltonianMaskKey, JointDeletionPayload>* payloads,
    bool preserve_structural_zero_messages) {
  if (payloads == nullptr) {
    throw std::invalid_argument("payload map must not be null");
  }

  std::vector<ComponentTreeLeafMessage> messages;
  messages.reserve(payloads->size());
  for (auto& [key, payload] : *payloads) {
    cleanup_joint_payload(&payload);
    if (!has_frontier_joint_payload(payload, preserve_structural_zero_messages)) {
      continue;
    }
    messages.push_back(ComponentTreeLeafMessage{
        .alpha_row_mask = key.alpha_row_mask,
        .alpha_col_mask = key.alpha_col_mask,
        .beta_row_mask = key.beta_row_mask,
        .beta_col_mask = key.beta_col_mask,
        .payload = std::move(payload),
    });
  }
  return messages;
}

FrontierJointMessage finalize_frontier_joint_message(
    int alpha_row_count,
    int alpha_col_count,
    int beta_row_count,
    int beta_col_count,
    std::map<FrontierHamiltonianMaskKey, JointDeletionPayload>* payloads,
    bool preserve_structural_zero_messages) {
  if (payloads == nullptr) {
    throw std::invalid_argument("frontier joint payload map must not be null");
  }

  FrontierJointMessage message;
  message.alpha_row_count = alpha_row_count;
  message.alpha_col_count = alpha_col_count;
  message.beta_row_count = beta_row_count;
  message.beta_col_count = beta_col_count;
  message.entries.reserve(payloads->size());
  for (auto& [key, payload] : *payloads) {
    cleanup_joint_payload(&payload);
    if (!has_frontier_joint_payload(payload, preserve_structural_zero_messages)) {
      continue;
    }
    message.entries.push_back(FrontierJointEntry{
        .key = key,
        .payload = std::move(payload),
    });
  }
  return message;
}

FrontierJointMessage make_identity_frontier_joint_message(
    int alpha_row_count,
    int alpha_col_count,
    int beta_row_count,
    int beta_col_count) {
  FrontierJointMessage message;
  message.alpha_row_count = alpha_row_count;
  message.alpha_col_count = alpha_col_count;
  message.beta_row_count = beta_row_count;
  message.beta_col_count = beta_col_count;
  message.entries.push_back(FrontierJointEntry{
      .key = FrontierHamiltonianMaskKey{},
      .payload = make_identity_joint_payload(),
  });
  return message;
}

FrontierJointMessage merge_frontier_joint_message_with_child(
    const FrontierJointMessage& frontier,
    const std::vector<ComponentTreeLeafMessage>& child_messages,
    int child_right_alpha_size,
    int child_left_alpha_size,
    int child_right_beta_size,
    int child_left_beta_size,
    int max_spin_degree,
    bool preserve_structural_zero_messages,
    std::uint64_t* dp_transition_count) {
  if (dp_transition_count == nullptr) {
    throw std::invalid_argument("dp_transition_count must not be null");
  }

  std::map<FrontierHamiltonianMaskKey, JointDeletionPayload> merged_payloads;
  for (const FrontierJointEntry& prefix_entry : frontier.entries) {
    for (const ComponentTreeLeafMessage& child_message : child_messages) {
      if ((prefix_entry.key.alpha_row_mask & child_message.alpha_row_mask) != 0U ||
          (prefix_entry.key.alpha_col_mask & child_message.alpha_col_mask) != 0U ||
          (prefix_entry.key.beta_row_mask & child_message.beta_row_mask) != 0U ||
          (prefix_entry.key.beta_col_mask & child_message.beta_col_mask) != 0U) {
        continue;
      }

      ++(*dp_transition_count);
      const FrontierHamiltonianMaskKey child_key{
          .alpha_row_mask = child_message.alpha_row_mask,
          .alpha_col_mask = child_message.alpha_col_mask,
          .beta_row_mask = child_message.beta_row_mask,
          .beta_col_mask = child_message.beta_col_mask,
      };
      const FrontierHamiltonianMaskKey merged_key{
          .alpha_row_mask =
              prefix_entry.key.alpha_row_mask | child_message.alpha_row_mask,
          .alpha_col_mask =
              prefix_entry.key.alpha_col_mask | child_message.alpha_col_mask,
          .beta_row_mask =
              prefix_entry.key.beta_row_mask | child_message.beta_row_mask,
          .beta_col_mask =
              prefix_entry.key.beta_col_mask | child_message.beta_col_mask,
      };
      const int parity = frontier_child_merge_parity(
          prefix_entry.key,
          child_key,
          frontier.alpha_row_count,
          frontier.alpha_col_count,
          frontier.beta_row_count,
          frontier.beta_col_count,
          child_right_alpha_size,
          child_left_alpha_size,
          child_right_beta_size,
          child_left_beta_size);
      const JointDeletionPayload merged_payload =
          merge_joint_deletion_payloads_limited(
              prefix_entry.payload,
              child_message.payload,
              max_spin_degree);
      add_scaled_joint_payload_preserve_basis(
          merged_payload,
          parity_sign(parity),
          preserve_structural_zero_messages,
          &merged_payloads[merged_key]);
    }
  }
  return finalize_frontier_joint_message(
      frontier.alpha_row_count,
      frontier.alpha_col_count,
      frontier.beta_row_count,
      frontier.beta_col_count,
      &merged_payloads,
      preserve_structural_zero_messages);
}

FrontierJointMessage build_frontier_joint_message(
    const std::vector<const std::vector<ComponentTreeLeafMessage>*>& child_messages,
    const ChildSpinSizeVectors& child_spin_sizes,
    int alpha_row_count,
    int alpha_col_count,
    int beta_row_count,
    int beta_col_count,
    int max_spin_degree,
    bool preserve_structural_zero_messages,
    std::uint64_t* dp_transition_count) {
  FrontierJointMessage frontier = make_identity_frontier_joint_message(
      alpha_row_count,
      alpha_col_count,
      beta_row_count,
      beta_col_count);
  for (std::size_t child_index = 0; child_index < child_messages.size(); ++child_index) {
    frontier = merge_frontier_joint_message_with_child(
        frontier,
        *child_messages[child_index],
        child_spin_sizes.right_alpha[child_index],
        child_spin_sizes.left_alpha[child_index],
        child_spin_sizes.right_beta[child_index],
        child_spin_sizes.left_beta[child_index],
        max_spin_degree,
        preserve_structural_zero_messages,
        dp_transition_count);
  }
  return frontier;
}

void reverse_frontier_joint_message(
    const std::vector<const std::vector<ComponentTreeLeafMessage>*>& child_messages,
    const ChildSpinSizeVectors& child_spin_sizes,
    const std::vector<FrontierJointMessage>& prefix_messages,
    const FrontierJointMessage& frontier_adjoint,
    int max_spin_degree,
    std::vector<std::vector<ComponentTreeLeafMessage>>* child_adjoint_messages,
    FrontierJointMessage* identity_adjoint) {
  if (child_adjoint_messages == nullptr || identity_adjoint == nullptr) {
    throw std::invalid_argument("frontier joint reverse outputs must not be null");
  }
  if (child_messages.size() != prefix_messages.size() - 1U ||
      child_messages.size() != child_adjoint_messages->size()) {
    throw std::invalid_argument("frontier joint reverse inputs have inconsistent sizes");
  }

  std::map<FrontierHamiltonianMaskKey, JointDeletionPayload> current_adjoint_map;
  for (const FrontierJointEntry& entry : frontier_adjoint.entries) {
    add_scaled_joint_payload_preserve_basis(
        entry.payload,
        1.0,
        true,
        &current_adjoint_map[entry.key]);
  }

  for (std::size_t child_offset = child_messages.size(); child_offset > 0U; --child_offset) {
    const std::size_t child_index = child_offset - 1U;
    const FrontierJointMessage& prefix_message =
        prefix_messages[child_index];
    const std::vector<ComponentTreeLeafMessage>& child_message =
        *child_messages[child_index];
    std::map<FrontierHamiltonianMaskKey, JointDeletionPayload> next_prefix_adjoint_map;
    std::map<FrontierHamiltonianMaskKey, JointDeletionPayload> child_adjoint_map;
    for (const FrontierJointEntry& prefix_entry : prefix_message.entries) {
      for (const ComponentTreeLeafMessage& child_entry : child_message) {
        if ((prefix_entry.key.alpha_row_mask & child_entry.alpha_row_mask) != 0U ||
            (prefix_entry.key.alpha_col_mask & child_entry.alpha_col_mask) != 0U ||
            (prefix_entry.key.beta_row_mask & child_entry.beta_row_mask) != 0U ||
            (prefix_entry.key.beta_col_mask & child_entry.beta_col_mask) != 0U) {
          continue;
        }

        const FrontierHamiltonianMaskKey merged_key{
            .alpha_row_mask =
                prefix_entry.key.alpha_row_mask | child_entry.alpha_row_mask,
            .alpha_col_mask =
                prefix_entry.key.alpha_col_mask | child_entry.alpha_col_mask,
            .beta_row_mask =
                prefix_entry.key.beta_row_mask | child_entry.beta_row_mask,
            .beta_col_mask =
                prefix_entry.key.beta_col_mask | child_entry.beta_col_mask,
        };
        const auto merged_iterator = current_adjoint_map.find(merged_key);
        if (merged_iterator == current_adjoint_map.end()) {
          continue;
        }
        cleanup_joint_payload(&merged_iterator->second);
        if (merged_iterator->second.basis_keys.empty()) {
          continue;
        }

        const FrontierHamiltonianMaskKey child_key{
            .alpha_row_mask = child_entry.alpha_row_mask,
            .alpha_col_mask = child_entry.alpha_col_mask,
            .beta_row_mask = child_entry.beta_row_mask,
            .beta_col_mask = child_entry.beta_col_mask,
        };
        const int parity = frontier_child_merge_parity(
            prefix_entry.key,
            child_key,
            prefix_message.alpha_row_count,
            prefix_message.alpha_col_count,
            prefix_message.beta_row_count,
            prefix_message.beta_col_count,
            child_spin_sizes.right_alpha[child_index],
            child_spin_sizes.left_alpha[child_index],
            child_spin_sizes.right_beta[child_index],
            child_spin_sizes.left_beta[child_index]);
        JointDeletionPayload prefix_branch_adjoint =
            make_zero_joint_payload_like(prefix_entry.payload);
        JointDeletionPayload child_branch_adjoint =
            make_zero_joint_payload_like(child_entry.payload);
        reverse_merge_joint_deletion_payloads_limited(
            prefix_entry.payload,
            child_entry.payload,
            max_spin_degree,
            scale_joint_payload(merged_iterator->second, parity_sign(parity)),
            &prefix_branch_adjoint,
            &child_branch_adjoint);
        add_scaled_joint_payload_preserve_basis(
            prefix_branch_adjoint,
            1.0,
            true,
            &next_prefix_adjoint_map[prefix_entry.key]);
        add_scaled_joint_payload_preserve_basis(
            child_branch_adjoint,
            1.0,
            true,
            &child_adjoint_map[child_key]);
      }
    }

    (*child_adjoint_messages)[child_index] =
        finalize_component_tree_leaf_messages(&child_adjoint_map, true);
    current_adjoint_map = std::move(next_prefix_adjoint_map);
  }

  *identity_adjoint = finalize_frontier_joint_message(
      prefix_messages.front().alpha_row_count,
      prefix_messages.front().alpha_col_count,
      prefix_messages.front().beta_row_count,
      prefix_messages.front().beta_col_count,
      &current_adjoint_map,
      true);
}

JointDeletionPayload merge_joint_deletion_payloads_limited(
    const JointDeletionPayload& left,
    const JointDeletionPayload& right,
    int max_spin_degree) {
  if (max_spin_degree < 0 || max_spin_degree > 2) {
    throw std::invalid_argument("max_spin_degree must be in [0, 2]");
  }
  JointDeletionPayload merged;
  merged.alpha_row_count = left.alpha_row_count + right.alpha_row_count;
  merged.alpha_col_count = left.alpha_col_count + right.alpha_col_count;
  merged.beta_row_count = left.beta_row_count + right.beta_row_count;
  merged.beta_col_count = left.beta_col_count + right.beta_col_count;
  for (const auto& left_key : left.basis_keys) {
    for (const auto& right_key : right.basis_keys) {
      JointDeletionKey combined_key;
      if (!combine_spin_keys_with_limit(
              left_key.alpha_key,
              right_key.alpha_key,
              max_spin_degree,
              &combined_key.alpha_key) ||
          !combine_spin_keys_with_limit(
              left_key.beta_key,
              right_key.beta_key,
              max_spin_degree,
              &combined_key.beta_key)) {
        continue;
      }
      merged.basis_keys.push_back(combined_key);
    }
  }

  for (const auto& left_key : left.basis_keys) {
    const auto left_iterator = left.sectors.find(left_key);
    const double left_value =
        (left_iterator == left.sectors.end()) ? 0.0 : left_iterator->second;
    for (const auto& right_key : right.basis_keys) {
      const auto right_iterator = right.sectors.find(right_key);
      const double right_value =
          (right_iterator == right.sectors.end()) ? 0.0 : right_iterator->second;
      JointDeletionKey combined_key;
      if (!combine_spin_keys_with_limit(
              left_key.alpha_key,
              right_key.alpha_key,
              max_spin_degree,
              &combined_key.alpha_key) ||
          !combine_spin_keys_with_limit(
              left_key.beta_key,
              right_key.beta_key,
              max_spin_degree,
              &combined_key.beta_key)) {
        continue;
      }

      const int parity =
          spin_sector_merge_parity(
              left.alpha_row_count,
              left.alpha_col_count,
              left_key.alpha_key,
              right.alpha_row_count,
              right.alpha_col_count,
              right_key.alpha_key) ^
          spin_sector_merge_parity(
              left.beta_row_count,
              left.beta_col_count,
              left_key.beta_key,
              right.beta_row_count,
              right.beta_col_count,
              right_key.beta_key);
      merged.sectors[combined_key] +=
          parity_sign(parity) * left_value * right_value;
    }
  }
  canonicalize_joint_payload_keys(&merged);
  cleanup_joint_payload(&merged);
  return merged;
}

double overlap_sector_value(const JointDeletionPayload& payload) {
  const auto iterator = payload.sectors.find(
      make_joint_key(make_spin_key(), make_spin_key()));
  return (iterator == payload.sectors.end()) ? 0.0 : iterator->second;
}

double contract_spin_one_electron_first_sectors(
    const JointDeletionPayload& payload,
    bool alpha_channel,
    const std::vector<double>& support_one_electron_storage,
    int support_size) {
  // Contracts the exact first-cofactor sectors of one spin channel against the
  // support-space one-electron operator.
  //
  // For the active spin, the joint payload stores the exact signed deleted
  // minor labeled by one deleted ket/right row orbital and one deleted
  // bra/left column orbital. The spectator spin must remain in the closed
  // `(0,0)` overlap sector. This is exactly the determinant one-electron
  // matrix-element formula
  //   sum_{r,c} h_{r c} C_{r c}.
  double total = 0.0;
  for (const auto& [key, value] : payload.sectors) {
    const SpinDeletionKey& active_key =
        alpha_channel ? key.alpha_key : key.beta_key;
    const SpinDeletionKey& spectator_key =
        alpha_channel ? key.beta_key : key.alpha_key;
    if (active_key.row_labels.size() != 1U ||
        active_key.col_labels.size() != 1U ||
        !is_empty_spin_key(spectator_key) ||
        std::abs(value) <= 1.0e-15) {
      continue;
    }

    const int right_orbital = active_key.row_labels[0];
    const int left_orbital = active_key.col_labels[0];
    total +=
        support_one_electron_storage[xmvb::to_size(left_orbital) *
                                         xmvb::to_size(support_size) +
                                     xmvb::to_size(right_orbital)] *
        value;
  }
  return total;
}

double contract_first_cofactor_matrix(
    const Eigen::MatrixXd& first_cofactor,
    const std::vector<double>& support_one_electron_storage,
    int support_size) {
  // Contracts one exact support-space first-cofactor matrix against the
  // one-electron operator in the repository storage convention
  //   storage[col * support_size + row].
  double total = 0.0;
  for (int col = 0; col < first_cofactor.cols(); ++col) {
    for (int row = 0; row < first_cofactor.rows(); ++row) {
      const double value = first_cofactor(row, col);
      if (std::abs(value) <= 1.0e-15) {
        continue;
      }
      total +=
          support_one_electron_storage[xmvb::to_size(col) *
                                           xmvb::to_size(support_size) +
                                       xmvb::to_size(row)] *
          value;
    }
  }
  return total;
}

double overlap_sector_value(const JointDeletionPayload& payload);
double overlap_sector_value(const HamiltonianBoundaryPayload& payload);

void accumulate_spin_first_sectors(
    const JointDeletionPayload& payload,
    bool alpha_channel,
    double scale,
    Eigen::MatrixXd* accumulator) {
  if (accumulator == nullptr) {
    throw std::invalid_argument("first-sector accumulator must not be null");
  }
  if (std::abs(scale) <= 1.0e-15) {
    return;
  }

  for (const auto& [key, value] : payload.sectors) {
    const SpinDeletionKey& active_key =
        alpha_channel ? key.alpha_key : key.beta_key;
    const SpinDeletionKey& spectator_key =
        alpha_channel ? key.beta_key : key.alpha_key;
    if (active_key.row_labels.size() != 1U ||
        active_key.col_labels.size() != 1U ||
        !is_empty_spin_key(spectator_key) ||
        std::abs(value) <= 1.0e-15) {
      continue;
    }

    const int right_orbital = active_key.row_labels[0];
    const int left_orbital = active_key.col_labels[0];
    (*accumulator)(right_orbital, left_orbital) += scale * value;
  }
}

void accumulate_spin_first_sectors(
    const HamiltonianBoundaryPayload& payload,
    bool alpha_channel,
    double scale,
    Eigen::MatrixXd* accumulator) {
  // Rebuilds the exact support-space first-cofactor matrix carried by one
  // typed Hamiltonian payload channel. The mixed alpha-beta sectors are not
  // needed here because the total opposite-spin contraction is rebuilt later
  // from the separate alpha and beta first-cofactor matrices.
  if (accumulator == nullptr) {
    throw std::invalid_argument("first-sector accumulator must not be null");
  }
  if (std::abs(scale) <= 1.0e-15) {
    return;
  }

  const std::vector<SameSpinDeletedSectorValue>& sectors =
      alpha_channel ? payload.alpha_sectors : payload.beta_sectors;
  for (const auto& sector : sectors) {
    if (sector.key.row_labels.size() != 1U ||
        sector.key.col_labels.size() != 1U ||
        std::abs(sector.value) <= 1.0e-15) {
      continue;
    }

    const int right_orbital = sector.key.row_labels[0];
    const int left_orbital = sector.key.col_labels[0];
    (*accumulator)(right_orbital, left_orbital) += scale * sector.value;
  }
}

void accumulate_same_spin_second_sector_gradient(
    const HamiltonianBoundaryPayload& payload,
    bool alpha_channel,
    double scale,
    std::vector<double>* packed_active_two_electron_gradient) {
  // Scatters one exact degree-2 deleted-minor family into the packed ERI
  // gradient storage. The payload already stores the exact antisymmetrized
  // same-spin coefficient for each `(2,2)` deleted sector, so the reverse map
  // only needs the direct/exchange index pair.
  if (packed_active_two_electron_gradient == nullptr) {
    throw std::invalid_argument("packed_active_two_electron_gradient must not be null");
  }
  if (std::abs(scale) <= 1.0e-15) {
    return;
  }

  const std::vector<SameSpinDeletedSectorValue>& sectors =
      alpha_channel ? payload.alpha_sectors : payload.beta_sectors;
  for (const auto& sector : sectors) {
    if (sector.key.row_labels.size() != 2U ||
        sector.key.col_labels.size() != 2U ||
        std::abs(sector.value) <= 1.0e-15) {
      continue;
    }

    const double weighted_value = scale * sector.value;
    const int direct_index =
        xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
            sector.key.row_labels[0],
            sector.key.col_labels[0],
            sector.key.row_labels[1],
            sector.key.col_labels[1]);
    const int exchange_index =
        xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
            sector.key.row_labels[0],
            sector.key.col_labels[1],
            sector.key.row_labels[1],
            sector.key.col_labels[0]);
    (*packed_active_two_electron_gradient)[xmvb::to_size(direct_index)] +=
        weighted_value;
    (*packed_active_two_electron_gradient)[xmvb::to_size(exchange_index)] -=
        weighted_value;
  }
}

void accumulate_opposite_spin_first_sector_gradient(
    const HamiltonianBoundaryPayload& payload,
    double scale,
    std::vector<double>* packed_active_two_electron_gradient) {
  // Scatters the mixed alpha/beta degree-1 sector directly to the packed ERI
  // gradient. Each payload entry already equals the exact coefficient of
  // `g[(beta_row,beta_col),(alpha_row,alpha_col)]`.
  if (packed_active_two_electron_gradient == nullptr) {
    throw std::invalid_argument("packed_active_two_electron_gradient must not be null");
  }
  if (std::abs(scale) <= 1.0e-15) {
    return;
  }

  for (const auto& sector : payload.mixed_sectors) {
    if (sector.key.alpha_key.row_labels.size() != 1U ||
        sector.key.alpha_key.col_labels.size() != 1U ||
        sector.key.beta_key.row_labels.size() != 1U ||
        sector.key.beta_key.col_labels.size() != 1U ||
        std::abs(sector.value) <= 1.0e-15) {
      continue;
    }

    const int eri_index =
        xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
            sector.key.beta_key.row_labels[0],
            sector.key.beta_key.col_labels[0],
            sector.key.alpha_key.row_labels[0],
            sector.key.alpha_key.col_labels[0]);
    (*packed_active_two_electron_gradient)[xmvb::to_size(eri_index)] +=
        scale * sector.value;
  }
}

void accumulate_local_first_cofactor_gradient(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const Eigen::MatrixXd& first_cofactor,
    double scale,
    Eigen::MatrixXd* active_one_electron_gradient) {
  // Lifts one determinant-level first cofactor from occupied-block coordinates
  // to the support-space matrix gradient with repository ordering
  // `gradient(right_orbital, left_orbital)`.
  if (active_one_electron_gradient == nullptr) {
    throw std::invalid_argument("active_one_electron_gradient must not be null");
  }
  if (std::abs(scale) <= 1.0e-15) {
    return;
  }
  if (first_cofactor.rows() != static_cast<int>(right_occ.size()) ||
      first_cofactor.cols() != static_cast<int>(left_occ.size())) {
    throw std::invalid_argument("first_cofactor dimensions do not match occupied lists");
  }

  for (int left_column = 0; left_column < static_cast<int>(left_occ.size()); ++left_column) {
    const int left_orbital = left_occ[xmvb::to_size(left_column)];
    for (int right_row = 0; right_row < static_cast<int>(right_occ.size()); ++right_row) {
      const int right_orbital = right_occ[xmvb::to_size(right_row)];
      (*active_one_electron_gradient)(right_orbital, left_orbital) +=
          scale * first_cofactor(right_row, left_column);
    }
  }
}

double exact_second_deleted_minor(
    const Eigen::MatrixXd& overlap_block,
    int deleted_row_first,
    int deleted_row_second,
    int deleted_col_first,
    int deleted_col_second,
    const DeterminantOverlapResolver& overlap_resolver) {
  const std::vector<int> deleted_rows{
      deleted_row_first,
      deleted_row_second,
  };
  const std::vector<int> deleted_cols{
      deleted_col_first,
      deleted_col_second,
  };
  const Eigen::MatrixXd minor =
      build_deleted_minor_matrix(overlap_block, deleted_rows, deleted_cols);
  return
      deleted_minor_sign(
          overlap_block.rows(),
          overlap_block.cols(),
          deleted_rows,
          deleted_cols,
          PartialSide::RightComplement) *
      determinant_of_dense_matrix(minor, overlap_resolver);
}

void accumulate_local_same_spin_two_electron_gradient(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const Eigen::MatrixXd& overlap_block,
    const DeterminantOverlapResult& overlap_result,
    const Eigen::MatrixXd& first_cofactor,
    double scale,
    const DeterminantOverlapResolver& overlap_resolver,
    std::vector<double>* packed_active_two_electron_gradient) {
  // Accumulates the exact same-spin ERI gradient of one determinant pair.
  //
  // Regular square overlaps reuse the fast first-cofactor identity
  //   Cof^(2) = (C11*C22 - C12*C21) / det(S),
  // while singular pairs fall back to the literal deleted-minor definition so
  // the G1 reference remains exact beyond the invertible case.
  if (packed_active_two_electron_gradient == nullptr) {
    throw std::invalid_argument("packed_active_two_electron_gradient must not be null");
  }
  if (std::abs(scale) <= 1.0e-15) {
    return;
  }

  const int n_electrons = static_cast<int>(left_occ.size());
  if (static_cast<int>(right_occ.size()) != n_electrons) {
    throw std::invalid_argument("same-spin occupied lists must have the same size");
  }
  if (n_electrons < 2 || overlap_result.nullity >= 3) {
    return;
  }

  const bool use_regular_formula =
      overlap_result.nullity == 0 && overlap_result.overlap_determinant != 0.0;
  const double cofactor_scale =
      use_regular_formula ? (scale / overlap_result.overlap_determinant) : 0.0;
  for (int left_first = 0; left_first < n_electrons - 1; ++left_first) {
    const int left_orbital_first = left_occ[xmvb::to_size(left_first)];
    for (int right_first = 0; right_first < n_electrons - 1; ++right_first) {
      const int right_orbital_first = right_occ[xmvb::to_size(right_first)];
      const double c11 = use_regular_formula ? first_cofactor(right_first, left_first) : 0.0;
      for (int left_second = left_first + 1; left_second < n_electrons; ++left_second) {
        const int left_orbital_second = left_occ[xmvb::to_size(left_second)];
        const double c12 = use_regular_formula ? first_cofactor(right_first, left_second) : 0.0;
        for (int right_second = right_first + 1; right_second < n_electrons; ++right_second) {
          const int right_orbital_second = right_occ[xmvb::to_size(right_second)];
          const double second_cofactor =
              use_regular_formula
                  ? cofactor_scale *
                      (c11 * first_cofactor(right_second, left_second) -
                       c12 * first_cofactor(right_second, left_first))
                  : scale *
                      exact_second_deleted_minor(
                          overlap_block,
                          right_first,
                          right_second,
                          left_first,
                          left_second,
                          overlap_resolver);
          if (std::abs(second_cofactor) <= 1.0e-15) {
            continue;
          }

          const int direct_index = TwoElectronIndexer::two_electron_storage_index(
              right_orbital_first,
              left_orbital_first,
              right_orbital_second,
              left_orbital_second);
          const int exchange_index = TwoElectronIndexer::two_electron_storage_index(
              right_orbital_first,
              left_orbital_second,
              right_orbital_second,
              left_orbital_first);
          (*packed_active_two_electron_gradient)[xmvb::to_size(direct_index)] +=
              second_cofactor;
          (*packed_active_two_electron_gradient)[xmvb::to_size(exchange_index)] -=
              second_cofactor;
        }
      }
    }
  }
}

void accumulate_local_opposite_spin_two_electron_gradient(
    const std::vector<int>& alpha_left_occ,
    const std::vector<int>& alpha_right_occ,
    const Eigen::MatrixXd& alpha_first_cofactor,
    const std::vector<int>& beta_left_occ,
    const std::vector<int>& beta_right_occ,
    const Eigen::MatrixXd& beta_first_cofactor,
    double scale,
    std::vector<double>* packed_active_two_electron_gradient) {
  // Accumulates the exact mixed alpha/beta ERI gradient from the determinant
  // first-cofactor products. `calc_cofactor_1st(...)` already returns the
  // exact deleted-minor matrix for nullity 0 and 1, and vanishes automatically
  // when the determinant rank is too small to support degree-1 minors.
  if (packed_active_two_electron_gradient == nullptr) {
    throw std::invalid_argument("packed_active_two_electron_gradient must not be null");
  }
  if (std::abs(scale) <= 1.0e-15) {
    return;
  }

  for (int alpha_left_column = 0;
       alpha_left_column < static_cast<int>(alpha_left_occ.size());
       ++alpha_left_column) {
    const int alpha_left_orbital =
        alpha_left_occ[xmvb::to_size(alpha_left_column)];
    for (int alpha_right_row = 0;
         alpha_right_row < static_cast<int>(alpha_right_occ.size());
         ++alpha_right_row) {
      const int alpha_right_orbital =
          alpha_right_occ[xmvb::to_size(alpha_right_row)];
      const double weighted_alpha_value =
          scale * alpha_first_cofactor(alpha_right_row, alpha_left_column);
      if (std::abs(weighted_alpha_value) <= 1.0e-15) {
        continue;
      }
      for (int beta_left_column = 0;
           beta_left_column < static_cast<int>(beta_left_occ.size());
           ++beta_left_column) {
        const int beta_left_orbital =
            beta_left_occ[xmvb::to_size(beta_left_column)];
        for (int beta_right_row = 0;
             beta_right_row < static_cast<int>(beta_right_occ.size());
             ++beta_right_row) {
          const int beta_right_orbital =
              beta_right_occ[xmvb::to_size(beta_right_row)];
          const double beta_value =
              beta_first_cofactor(beta_right_row, beta_left_column);
          if (std::abs(beta_value) <= 1.0e-15) {
            continue;
          }
          const int eri_index = TwoElectronIndexer::two_electron_storage_index(
              beta_right_orbital,
              beta_left_orbital,
              alpha_right_orbital,
              alpha_left_orbital);
          (*packed_active_two_electron_gradient)[xmvb::to_size(eri_index)] +=
              weighted_alpha_value * beta_value;
        }
      }
    }
  }
}

void accumulate_determinant_pair_hamiltonian_contribution(
    const GlobalOrientationTerm& left_term,
    const GlobalOrientationTerm& right_term,
    double pair_weight,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    int support_size,
    const std::vector<double>& packed_active_two_electron_integrals,
    const DeterminantOverlapResolver& overlap_resolver,
    const FullDeterminantPairEvaluator& determinant_pair_evaluator,
    ComponentTreeHamiltonianResult* result,
    std::vector<double>* active_orbital_overlap_gradient,
    Eigen::MatrixXd* active_one_electron_gradient,
    std::vector<double>* packed_active_two_electron_gradient,
    double hamiltonian_weight = 1.0,
    double overlap_weight = 0.0) {
  // Shared exact determinant-pair contribution used by the scalar exact
  // reference and the local active-space gradient oracles.
  //
  // `pair_weight` always scales the actual rooted-tree matrix element value.
  // The gradient blocks are optionally weighted by
  //   hamiltonian_weight * H_total + overlap_weight * S.
  if (std::abs(pair_weight) <= 1.0e-15) {
    return;
  }

  const FullDeterminantPairEvaluation determinant_pair =
      determinant_pair_evaluator.evaluate(
          left_term.alpha_occ,
          right_term.alpha_occ,
          left_term.beta_occ,
          right_term.beta_occ,
          support_overlap_storage,
          support_one_electron_storage,
          support_size,
          packed_active_two_electron_integrals);
  const auto& alpha_result = determinant_pair.alpha.overlap_result;
  const auto& beta_result = determinant_pair.beta.overlap_result;
  const double alpha_overlap = alpha_result.overlap_determinant;
  const double beta_overlap = beta_result.overlap_determinant;
  const double raw_same_spin_alpha =
      (determinant_pair.alpha.total_hamiltonian -
       determinant_pair.alpha.one_electron_hamiltonian) *
      beta_overlap;
  const double raw_same_spin_beta =
      (determinant_pair.beta.total_hamiltonian -
       determinant_pair.beta.one_electron_hamiltonian) *
      alpha_overlap;
  const double raw_opposite_spin =
      determinant_pair.total_hamiltonian -
      determinant_pair.one_electron_hamiltonian -
      raw_same_spin_alpha -
      raw_same_spin_beta;
  const double same_spin_alpha = raw_same_spin_alpha;
  const double same_spin_beta = raw_same_spin_beta;
  const double opposite_spin = raw_opposite_spin;
  const double filtered_total_hamiltonian =
      determinant_pair.one_electron_hamiltonian +
      same_spin_alpha +
      same_spin_beta +
      opposite_spin;
  if (result != nullptr) {
    result->overlap += pair_weight * determinant_pair.overlap_determinant;
    result->one_electron += pair_weight * determinant_pair.one_electron_hamiltonian;
    result->same_spin_alpha_two_electron += pair_weight * same_spin_alpha;
    result->same_spin_beta_two_electron += pair_weight * same_spin_beta;
    result->opposite_spin_two_electron += pair_weight * opposite_spin;
    result->two_electron += pair_weight * (same_spin_alpha + same_spin_beta + opposite_spin);
    result->total_electronic_hamiltonian += pair_weight * filtered_total_hamiltonian;
  }

  const bool need_overlap_gradient =
      active_orbital_overlap_gradient != nullptr &&
      (std::abs(hamiltonian_weight) > 1.0e-15 ||
       std::abs(overlap_weight) > 1.0e-15);
  const bool need_hamiltonian_gradient =
      std::abs(hamiltonian_weight) > 1.0e-15 &&
      (active_one_electron_gradient != nullptr ||
       packed_active_two_electron_gradient != nullptr);
  if (!need_overlap_gradient && !need_hamiltonian_gradient) {
    return;
  }

  Eigen::MatrixXd alpha_first_cofactor;
  Eigen::MatrixXd beta_first_cofactor;
  if (need_hamiltonian_gradient) {
    alpha_first_cofactor = calc_cofactor_1st(alpha_result);
    beta_first_cofactor = calc_cofactor_1st(beta_result);
  }

  const double alpha_scale = pair_weight * hamiltonian_weight * beta_overlap;
  const double beta_scale = pair_weight * hamiltonian_weight * alpha_overlap;
  if (active_one_electron_gradient != nullptr &&
      std::abs(hamiltonian_weight) > 1.0e-15) {
    accumulate_local_first_cofactor_gradient(
        left_term.alpha_occ,
        right_term.alpha_occ,
        alpha_first_cofactor,
        alpha_scale,
        active_one_electron_gradient);
    accumulate_local_first_cofactor_gradient(
        left_term.beta_occ,
        right_term.beta_occ,
        beta_first_cofactor,
        beta_scale,
        active_one_electron_gradient);
  }

  if (need_overlap_gradient) {
    if (alpha_result.nullity != 0 || beta_result.nullity != 0) {
      throw std::runtime_error(
          "weighted exact overlap gradient requires nullity == 0");
    }

    const int n_alpha_electrons = static_cast<int>(left_term.alpha_occ.size());
    const int n_beta_electrons = static_cast<int>(left_term.beta_occ.size());
    Eigen::MatrixXd alpha_inverse_overlap_gradient =
        Eigen::MatrixXd::Zero(n_alpha_electrons, n_alpha_electrons);
    Eigen::MatrixXd beta_inverse_overlap_gradient =
        Eigen::MatrixXd::Zero(n_beta_electrons, n_beta_electrons);
    double alpha_phi = 0.0;
    double beta_phi = 0.0;
    double opposite_spin_phi = 0.0;

    if (std::abs(hamiltonian_weight) > 1.0e-15) {
      Eigen::MatrixXd alpha_same_spin_inverse_overlap_gradient =
          Eigen::MatrixXd::Zero(n_alpha_electrons, n_alpha_electrons);
      Eigen::MatrixXd beta_same_spin_inverse_overlap_gradient =
          Eigen::MatrixXd::Zero(n_beta_electrons, n_beta_electrons);
      Eigen::MatrixXd alpha_opposite_spin_inverse_overlap_gradient =
          Eigen::MatrixXd::Zero(n_alpha_electrons, n_alpha_electrons);
      Eigen::MatrixXd beta_opposite_spin_inverse_overlap_gradient =
          Eigen::MatrixXd::Zero(n_beta_electrons, n_beta_electrons);

      SameSpinPhiResult alpha_phi_result;
      SameSpinPhiResult beta_phi_result;
      alpha_phi_result = compute_same_spin_original_phi(
          left_term.alpha_occ,
          right_term.alpha_occ,
          support_one_electron_storage,
          support_size,
          packed_active_two_electron_integrals,
          alpha_result,
          &alpha_same_spin_inverse_overlap_gradient);
      beta_phi_result = compute_same_spin_original_phi(
          left_term.beta_occ,
          right_term.beta_occ,
          support_one_electron_storage,
          support_size,
          packed_active_two_electron_integrals,
          beta_result,
          &beta_same_spin_inverse_overlap_gradient);
      if (n_alpha_electrons > 0 &&
          n_beta_electrons > 0) {
        opposite_spin_phi = compute_opposite_spin_original_phi(
            left_term.alpha_occ,
            right_term.alpha_occ,
            alpha_result,
            left_term.beta_occ,
            right_term.beta_occ,
            beta_result,
            packed_active_two_electron_integrals,
            &alpha_opposite_spin_inverse_overlap_gradient,
            &beta_opposite_spin_inverse_overlap_gradient);
      }
      alpha_inverse_overlap_gradient =
          alpha_same_spin_inverse_overlap_gradient +
          alpha_opposite_spin_inverse_overlap_gradient;
      beta_inverse_overlap_gradient =
          beta_same_spin_inverse_overlap_gradient +
          beta_opposite_spin_inverse_overlap_gradient;
      alpha_phi = alpha_phi_result.total_phi;
      beta_phi = beta_phi_result.total_phi;
    }

    const double phi_sum = alpha_phi + beta_phi + opposite_spin_phi;
    const double alpha_determinant_weight =
        pair_weight *
        (overlap_weight * beta_overlap +
         hamiltonian_weight * beta_overlap * phi_sum);
    const double beta_determinant_weight =
        pair_weight *
        (overlap_weight * alpha_overlap +
         hamiltonian_weight * alpha_overlap * phi_sum);
    accumulate_spin_overlap_gradient(
        left_term.alpha_occ,
        right_term.alpha_occ,
        alpha_result,
        alpha_determinant_weight,
        pair_weight * hamiltonian_weight * beta_overlap *
            alpha_inverse_overlap_gradient,
        support_size,
        active_orbital_overlap_gradient);
    accumulate_spin_overlap_gradient(
        left_term.beta_occ,
        right_term.beta_occ,
        beta_result,
        beta_determinant_weight,
        pair_weight * hamiltonian_weight * alpha_overlap *
            beta_inverse_overlap_gradient,
        support_size,
        active_orbital_overlap_gradient);
  }

  if (packed_active_two_electron_gradient == nullptr ||
      std::abs(hamiltonian_weight) <= 1.0e-15) {
    return;
  }

  if (left_term.alpha_occ.size() >= 2U &&
      right_term.alpha_occ.size() >= 2U &&
      alpha_result.nullity < 3) {
    const std::vector<double> alpha_overlap_storage = build_overlap_submatrix(
        left_term.alpha_occ,
        right_term.alpha_occ,
        support_overlap_storage,
        support_size);
    const Eigen::Map<const Eigen::MatrixXd> alpha_overlap_block(
        alpha_overlap_storage.data(),
        static_cast<int>(left_term.alpha_occ.size()),
        static_cast<int>(right_term.alpha_occ.size()));
    accumulate_local_same_spin_two_electron_gradient(
        left_term.alpha_occ,
        right_term.alpha_occ,
        alpha_overlap_block,
        alpha_result,
        alpha_first_cofactor,
        alpha_scale,
        overlap_resolver,
        packed_active_two_electron_gradient);
  }
  if (left_term.beta_occ.size() >= 2U &&
      right_term.beta_occ.size() >= 2U &&
      beta_result.nullity < 3) {
    const std::vector<double> beta_overlap_storage = build_overlap_submatrix(
        left_term.beta_occ,
        right_term.beta_occ,
        support_overlap_storage,
        support_size);
    const Eigen::Map<const Eigen::MatrixXd> beta_overlap_block(
        beta_overlap_storage.data(),
        static_cast<int>(left_term.beta_occ.size()),
        static_cast<int>(right_term.beta_occ.size()));
    accumulate_local_same_spin_two_electron_gradient(
        left_term.beta_occ,
        right_term.beta_occ,
        beta_overlap_block,
        beta_result,
        beta_first_cofactor,
        beta_scale,
        overlap_resolver,
        packed_active_two_electron_gradient);
  }
  if (!left_term.alpha_occ.empty() &&
      !right_term.alpha_occ.empty() &&
      !left_term.beta_occ.empty() &&
      !right_term.beta_occ.empty()) {
    accumulate_local_opposite_spin_two_electron_gradient(
        left_term.alpha_occ,
        right_term.alpha_occ,
        alpha_first_cofactor,
        left_term.beta_occ,
        right_term.beta_occ,
        beta_first_cofactor,
        pair_weight * hamiltonian_weight,
        packed_active_two_electron_gradient);
  }
}

ComponentTreeHamiltonianResult evaluate_component_tree_hamiltonian_exact_reference(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const DeterminantOverlapResolver& overlap_resolver) {
  if (support_size <= 0) {
    throw std::invalid_argument("support_size must be positive");
  }
  if (ordered_components.empty()) {
    throw std::invalid_argument("ordered_components must not be empty");
  }

  ComponentTreeHamiltonianResult result;
  const std::vector<GlobalOrientationTerm> left_global_terms =
      build_global_orientation_terms(ordered_components, true);
  const std::vector<GlobalOrientationTerm> right_global_terms =
      build_global_orientation_terms(ordered_components, false);
  const DeterminantHamiltonianResolver hamiltonian_resolver(overlap_resolver);
  const FullDeterminantPairEvaluator determinant_pair_evaluator(
      overlap_resolver,
      hamiltonian_resolver);

  for (const auto& left_term : left_global_terms) {
    for (const auto& right_term : right_global_terms) {
      const double pair_weight = left_term.coefficient * right_term.coefficient;
      accumulate_determinant_pair_hamiltonian_contribution(
          left_term,
          right_term,
          pair_weight,
          support_overlap_storage,
          support_one_electron_storage,
          support_size,
          packed_active_two_electron_integrals,
          overlap_resolver,
          determinant_pair_evaluator,
          &result,
          nullptr,
          nullptr,
          nullptr);
    }
  }
  return result;
}

ComponentTreeHamiltonianGradientResult
evaluate_component_tree_active_space_gradient_exact_reference(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const DeterminantOverlapResolver& overlap_resolver,
    double hamiltonian_weight,
    double overlap_weight) {
  if (support_size <= 0) {
    throw std::invalid_argument("support_size must be positive");
  }
  if (ordered_components.empty()) {
    throw std::invalid_argument("ordered_components must not be empty");
  }

  ComponentTreeHamiltonianGradientResult result;
  result.active_orbital_overlap_gradient.assign(
      xmvb::to_size(support_size * support_size),
      0.0);
  result.active_one_electron_gradient.assign(
      xmvb::to_size(support_size * support_size),
      0.0);
  result.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integrals.size(),
      0.0);
  Eigen::MatrixXd active_one_electron_gradient = Eigen::MatrixXd::Zero(support_size, support_size);

  const std::vector<GlobalOrientationTerm> left_global_terms =
      build_global_orientation_terms(ordered_components, true);
  const std::vector<GlobalOrientationTerm> right_global_terms =
      build_global_orientation_terms(ordered_components, false);
  const DeterminantHamiltonianResolver hamiltonian_resolver(overlap_resolver);
  const FullDeterminantPairEvaluator determinant_pair_evaluator(
      overlap_resolver,
      hamiltonian_resolver);

  for (const auto& left_term : left_global_terms) {
    for (const auto& right_term : right_global_terms) {
      const double pair_weight = left_term.coefficient * right_term.coefficient;
      accumulate_determinant_pair_hamiltonian_contribution(
          left_term,
          right_term,
          pair_weight,
          support_overlap_storage,
          support_one_electron_storage,
          support_size,
          packed_active_two_electron_integrals,
          overlap_resolver,
          determinant_pair_evaluator,
          &result.hamiltonian,
          &result.active_orbital_overlap_gradient,
          &active_one_electron_gradient,
          &result.packed_active_two_electron_gradient,
          hamiltonian_weight,
          overlap_weight);
    }
  }

  result.active_one_electron_gradient.assign(
      active_one_electron_gradient.data(),
      active_one_electron_gradient.data() + active_one_electron_gradient.size());
  return result;
}

double overlap_sector_value(const HamiltonianBoundaryPayload& payload) {
  return payload.overlap;
}

double contract_spin_one_electron_first_sectors(
    const HamiltonianBoundaryPayload& payload,
    bool alpha_channel,
    const std::vector<double>& support_one_electron_storage,
    int support_size) {
  const std::vector<SameSpinDeletedSectorValue>& sectors =
      alpha_channel ? payload.alpha_sectors : payload.beta_sectors;
  double total = 0.0;
  for (const auto& sector : sectors) {
    if (sector.key.row_labels.size() != 1U ||
        sector.key.col_labels.size() != 1U ||
        std::abs(sector.value) <= 1.0e-15) {
      continue;
    }
    const int right_orbital = sector.key.row_labels[0];
    const int left_orbital = sector.key.col_labels[0];
    total +=
        support_one_electron_storage[xmvb::to_size(left_orbital) *
                                         xmvb::to_size(support_size) +
                                     xmvb::to_size(right_orbital)] *
        sector.value;
  }
  return total;
}

double contract_total_one_electron_first_sectors(
    const HamiltonianBoundaryPayload& payload,
    const std::vector<double>& support_one_electron_storage,
    int support_size) {
  return
      contract_spin_one_electron_first_sectors(
          payload,
          true,
          support_one_electron_storage,
          support_size) +
      contract_spin_one_electron_first_sectors(
          payload,
          false,
          support_one_electron_storage,
          support_size);
}

double contract_same_spin_second_sectors(
    const HamiltonianBoundaryPayload& payload,
    bool alpha_channel,
    const std::vector<double>& packed_active_two_electron_integrals) {
  const std::vector<SameSpinDeletedSectorValue>& sectors =
      alpha_channel ? payload.alpha_sectors : payload.beta_sectors;
  double total = 0.0;
  for (const auto& sector : sectors) {
    if (sector.key.row_labels.size() != 2U ||
        sector.key.col_labels.size() != 2U ||
        std::abs(sector.value) <= 1.0e-15) {
      continue;
    }
    const int direct_index =
        xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
            sector.key.row_labels[0],
            sector.key.col_labels[0],
            sector.key.row_labels[1],
            sector.key.col_labels[1]);
    const int exchange_index =
        xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
            sector.key.row_labels[0],
            sector.key.col_labels[1],
            sector.key.row_labels[1],
            sector.key.col_labels[0]);
    total +=
        (packed_active_two_electron_integrals[xmvb::to_size(direct_index)] -
         packed_active_two_electron_integrals[xmvb::to_size(exchange_index)]) *
        sector.value;
  }
  return total;
}

double contract_opposite_spin_first_sectors(
    const HamiltonianBoundaryPayload& payload,
    const std::vector<double>& packed_active_two_electron_integrals) {
  double total = 0.0;
  for (const auto& sector : payload.mixed_sectors) {
    if (sector.key.alpha_key.row_labels.size() != 1U ||
        sector.key.alpha_key.col_labels.size() != 1U ||
        sector.key.beta_key.row_labels.size() != 1U ||
        sector.key.beta_key.col_labels.size() != 1U ||
        std::abs(sector.value) <= 1.0e-15) {
      continue;
    }
    const int eri_index =
        xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
            sector.key.beta_key.row_labels[0],
            sector.key.beta_key.col_labels[0],
            sector.key.alpha_key.row_labels[0],
            sector.key.alpha_key.col_labels[0]);
    total +=
        packed_active_two_electron_integrals[xmvb::to_size(eri_index)] *
        sector.value;
  }
  return total;
}

double contract_dense_spin_one_electron_first_sectors(
    const std::vector<double>& degree1,
    int row_count,
    int col_count,
    int support_size,
    const std::vector<double>& support_one_electron_storage) {
  double total = 0.0;
  for (int index = 0; index < static_cast<int>(degree1.size()); ++index) {
    const double value = degree1[xmvb::to_size(index)];
    if (std::abs(value) <= kHamiltonianDenseZeroTolerance) {
      continue;
    }
    SpinDeletionKey key;
    decode_dense_spin_degree1_key(
        row_count,
        col_count,
        index,
        support_size,
        &key);
    if (key.row_labels.size() != 1U || key.col_labels.size() != 1U) {
      continue;
    }
    total +=
        support_one_electron_storage[xmvb::to_size(key.col_labels[0]) *
                                         xmvb::to_size(support_size) +
                                     xmvb::to_size(key.row_labels[0])] *
        value;
  }
  return total;
}

double contract_dense_same_spin_second_sectors(
    const std::vector<double>& degree2,
    int row_count,
    int col_count,
    int support_size,
    const std::vector<double>& packed_active_two_electron_integrals) {
  double total = 0.0;
  for (int index = 0; index < static_cast<int>(degree2.size()); ++index) {
    const double value = degree2[xmvb::to_size(index)];
    if (std::abs(value) <= kHamiltonianDenseZeroTolerance) {
      continue;
    }
    SpinDeletionKey key;
    decode_dense_spin_degree2_key(
        row_count,
        col_count,
        index,
        support_size,
        &key);
    if (key.row_labels.size() != 2U || key.col_labels.size() != 2U) {
      continue;
    }
    const int direct_index =
        xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
            key.row_labels[0],
            key.col_labels[0],
            key.row_labels[1],
            key.col_labels[1]);
    const int exchange_index =
        xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
            key.row_labels[0],
            key.col_labels[1],
            key.row_labels[1],
            key.col_labels[0]);
    total +=
        (packed_active_two_electron_integrals[xmvb::to_size(direct_index)] -
         packed_active_two_electron_integrals[xmvb::to_size(exchange_index)]) *
        value;
  }
  return total;
}

double contract_dense_opposite_spin_first_sectors(
    const HamiltonianEntryDensePayload& payload,
    const std::vector<double>& packed_active_two_electron_integrals) {
  const int alpha_degree1_size = dense_spin_degree1_size(
      payload.alpha_row_count,
      payload.alpha_col_count,
      payload.support_size);
  const int beta_degree1_size = dense_spin_degree1_size(
      payload.beta_row_count,
      payload.beta_col_count,
      payload.support_size);
  double total = 0.0;
  for (int beta_index = 0; beta_index < beta_degree1_size; ++beta_index) {
    SpinDeletionKey beta_key;
    decode_dense_spin_degree1_key(
        payload.beta_row_count,
        payload.beta_col_count,
        beta_index,
        payload.support_size,
        &beta_key);
    if (beta_key.row_labels.size() != 1U || beta_key.col_labels.size() != 1U) {
      continue;
    }
    for (int alpha_index = 0; alpha_index < alpha_degree1_size; ++alpha_index) {
      const double value = payload.mixed_degree1[xmvb::to_size(
          dense_mixed_degree1_flat_index(payload, alpha_index, beta_index))];
      if (std::abs(value) <= kHamiltonianDenseZeroTolerance) {
        continue;
      }
      SpinDeletionKey alpha_key;
      decode_dense_spin_degree1_key(
          payload.alpha_row_count,
          payload.alpha_col_count,
          alpha_index,
          payload.support_size,
          &alpha_key);
      if (alpha_key.row_labels.size() != 1U ||
          alpha_key.col_labels.size() != 1U) {
        continue;
      }
      const int eri_index =
          xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
              beta_key.row_labels[0],
              beta_key.col_labels[0],
              alpha_key.row_labels[0],
              alpha_key.col_labels[0]);
      total +=
          packed_active_two_electron_integrals[xmvb::to_size(eri_index)] *
          value;
    }
  }
  return total;
}

void accumulate_dense_spin_first_sectors(
    const std::vector<double>& degree1,
    int row_count,
    int col_count,
    int support_size,
    double scale,
    Eigen::MatrixXd* accumulator) {
  if (accumulator == nullptr) {
    throw std::invalid_argument("dense first-sector accumulator must not be null");
  }
  if (std::abs(scale) <= kHamiltonianDenseZeroTolerance) {
    return;
  }
  for (int index = 0; index < static_cast<int>(degree1.size()); ++index) {
    const double value = degree1[xmvb::to_size(index)];
    if (std::abs(value) <= kHamiltonianDenseZeroTolerance) {
      continue;
    }
    SpinDeletionKey key;
    decode_dense_spin_degree1_key(
        row_count,
        col_count,
        index,
        support_size,
        &key);
    if (key.row_labels.size() != 1U || key.col_labels.size() != 1U) {
      continue;
    }
    (*accumulator)(
        key.row_labels[0],
        key.col_labels[0]) += scale * value;
  }
}

void accumulate_dense_same_spin_second_sector_gradient(
    const std::vector<double>& degree2,
    int row_count,
    int col_count,
    int support_size,
    double scale,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (packed_active_two_electron_gradient == nullptr) {
    throw std::invalid_argument("dense two-electron gradient must not be null");
  }
  if (std::abs(scale) <= kHamiltonianDenseZeroTolerance) {
    return;
  }
  for (int index = 0; index < static_cast<int>(degree2.size()); ++index) {
    const double value = degree2[xmvb::to_size(index)];
    if (std::abs(value) <= kHamiltonianDenseZeroTolerance) {
      continue;
    }
    SpinDeletionKey key;
    decode_dense_spin_degree2_key(
        row_count,
        col_count,
        index,
        support_size,
        &key);
    if (key.row_labels.size() != 2U || key.col_labels.size() != 2U) {
      continue;
    }
    const double weighted_value = scale * value;
    const int direct_index =
        xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
            key.row_labels[0],
            key.col_labels[0],
            key.row_labels[1],
            key.col_labels[1]);
    const int exchange_index =
        xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
            key.row_labels[0],
            key.col_labels[1],
            key.row_labels[1],
            key.col_labels[0]);
    (*packed_active_two_electron_gradient)[xmvb::to_size(direct_index)] +=
        weighted_value;
    (*packed_active_two_electron_gradient)[xmvb::to_size(exchange_index)] -=
        weighted_value;
  }
}

void accumulate_dense_opposite_spin_first_sector_gradient(
    const HamiltonianEntryDensePayload& payload,
    double scale,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (packed_active_two_electron_gradient == nullptr) {
    throw std::invalid_argument("dense mixed gradient must not be null");
  }
  if (std::abs(scale) <= kHamiltonianDenseZeroTolerance) {
    return;
  }
  const int alpha_degree1_size = dense_spin_degree1_size(
      payload.alpha_row_count,
      payload.alpha_col_count,
      payload.support_size);
  const int beta_degree1_size = dense_spin_degree1_size(
      payload.beta_row_count,
      payload.beta_col_count,
      payload.support_size);
  for (int beta_index = 0; beta_index < beta_degree1_size; ++beta_index) {
    SpinDeletionKey beta_key;
    decode_dense_spin_degree1_key(
        payload.beta_row_count,
        payload.beta_col_count,
        beta_index,
        payload.support_size,
        &beta_key);
    if (beta_key.row_labels.size() != 1U || beta_key.col_labels.size() != 1U) {
      continue;
    }
    for (int alpha_index = 0; alpha_index < alpha_degree1_size; ++alpha_index) {
      const double value = payload.mixed_degree1[xmvb::to_size(
          dense_mixed_degree1_flat_index(payload, alpha_index, beta_index))];
      if (std::abs(value) <= kHamiltonianDenseZeroTolerance) {
        continue;
      }
      SpinDeletionKey alpha_key;
      decode_dense_spin_degree1_key(
          payload.alpha_row_count,
          payload.alpha_col_count,
          alpha_index,
          payload.support_size,
          &alpha_key);
      if (alpha_key.row_labels.size() != 1U ||
          alpha_key.col_labels.size() != 1U) {
        continue;
      }
      const int eri_index =
          xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
              beta_key.row_labels[0],
              beta_key.col_labels[0],
              alpha_key.row_labels[0],
              alpha_key.col_labels[0]);
      (*packed_active_two_electron_gradient)[xmvb::to_size(eri_index)] +=
          scale * value;
    }
  }
}

SpinDeletionPayload build_root_spin_payload_limited(
    const std::vector<int>& left_root_occ,
    const std::vector<int>& right_root_occ,
    std::uint32_t used_row_mask,
    std::uint32_t used_col_mask,
    int max_deleted_rank,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    std::unordered_map<SpinPayloadCacheKey,
                       SpinDeletionPayload,
                       SpinPayloadCacheKeyHasher>* payload_cache) {
  const std::uint32_t row_full_mask =
      right_root_occ.empty()
          ? 0U
          : (open_state_mask_limit(static_cast<int>(right_root_occ.size())) - 1U);
  const std::uint32_t col_full_mask =
      left_root_occ.empty()
          ? 0U
          : (open_state_mask_limit(static_cast<int>(left_root_occ.size())) - 1U);
  const auto root_rows = select_occ_by_mask(right_root_occ, row_full_mask ^ used_row_mask);
  const auto root_cols = select_occ_by_mask(left_root_occ, col_full_mask ^ used_col_mask);
  return build_spin_deleted_minor_payload_cached_limited(
      root_cols,
      root_rows,
      overlap_storage,
      n_orbitals,
      0,
      0,
      max_deleted_rank,
      false,
      PartialSide::RightComplement,
      overlap_resolver,
      subdeterminant_evaluations,
      payload_cache);
}

SpinDeletionPayload build_exact_frontier_spin_payload_limited(
    const std::vector<int>& selected_root_cols,
    const std::vector<int>& leaf_left_occ,
    const std::vector<int>& selected_root_rows,
    const std::vector<int>& leaf_right_occ,
    int max_deleted_rank,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    std::unordered_map<SpinPayloadCacheKey,
                       SpinDeletionPayload,
                       SpinPayloadCacheKeyHasher>* payload_cache) {
  std::vector<int> frontier_left_occ;
  std::vector<int> frontier_right_occ;
  assign_concatenated_occ(selected_root_cols, leaf_left_occ, &frontier_left_occ);
  assign_concatenated_occ(selected_root_rows, leaf_right_occ, &frontier_right_occ);
  return build_spin_deleted_minor_payload_cached_limited(
      frontier_left_occ,
      frontier_right_occ,
      overlap_storage,
      n_orbitals,
      static_cast<int>(selected_root_rows.size()),
      static_cast<int>(selected_root_cols.size()),
      max_deleted_rank,
      true,
      PartialSide::LeftFrontier,
      overlap_resolver,
      subdeterminant_evaluations,
      payload_cache);
}

SpinDeletionPayload build_exact_frontier_spin_payload(
    const std::vector<int>& selected_root_cols,
    const std::vector<int>& leaf_left_occ,
    const std::vector<int>& selected_root_rows,
    const std::vector<int>& leaf_right_occ,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    std::unordered_map<SpinPayloadCacheKey,
                       SpinDeletionPayload,
                       SpinPayloadCacheKeyHasher>* payload_cache) {
  return build_exact_frontier_spin_payload_limited(
      selected_root_cols,
      leaf_left_occ,
      selected_root_rows,
      leaf_right_occ,
      2,
      overlap_storage,
      n_orbitals,
      overlap_resolver,
      subdeterminant_evaluations,
      payload_cache);
}

HamiltonianBoundaryPayload build_root_hamiltonian_boundary_payload(
    const OrientationTerm& left_root_term,
    const OrientationTerm& right_root_term,
    std::uint32_t used_alpha_row_mask,
    std::uint32_t used_alpha_col_mask,
    std::uint32_t used_beta_row_mask,
    std::uint32_t used_beta_col_mask,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    std::unordered_map<SpinPayloadCacheKey,
                       SpinDeletionPayload,
                       SpinPayloadCacheKeyHasher>* payload_cache) {
  const SpinDeletionPayload alpha_payload =
      build_root_spin_payload_limited(
          left_root_term.alpha_occ,
          right_root_term.alpha_occ,
          used_alpha_row_mask,
          used_alpha_col_mask,
          2,
          overlap_storage,
          n_orbitals,
          overlap_resolver,
          subdeterminant_evaluations,
          payload_cache);
  const SpinDeletionPayload beta_payload =
      build_root_spin_payload_limited(
          left_root_term.beta_occ,
          right_root_term.beta_occ,
          used_beta_row_mask,
          used_beta_col_mask,
          2,
          overlap_storage,
          n_orbitals,
          overlap_resolver,
          subdeterminant_evaluations,
          payload_cache);
  return build_hamiltonian_boundary_payload_values(alpha_payload, beta_payload);
}

HamiltonianBoundaryPayload close_root_hamiltonian_payload(
    const HamiltonianBoundaryPayload& frontier_payload,
    const HamiltonianBoundaryPayload& root_payload,
    const OrientationTerm& left_root_term,
    const OrientationTerm& right_root_term,
    std::uint32_t used_alpha_row_mask,
    std::uint32_t used_alpha_col_mask,
    std::uint32_t used_beta_row_mask,
    std::uint32_t used_beta_col_mask) {
  // Root closure differs from an ordinary subtree append:
  // the merged natural order is
  //   [selected-root-interface, frontier-body, root-remainder],
  // while the final closed root determinant order is
  //   [selected-root-interface, root-remainder, frontier-body].
  //
  // So after the ordinary deleted-minor append merge we must move the local
  // root remainder block left across the frontier body, but not across the
  // already extracted root-interface labels.
  const HamiltonianBoundaryPayload natural_payload =
      merge_hamiltonian_boundary_payloads_limited_values(
          frontier_payload,
          root_payload,
          2);
  if (!is_hamiltonian_payload_nonzero(natural_payload)) {
    return natural_payload;
  }

  static const std::vector<int> empty_labels;
  return transform_hamiltonian_interface_block_to_front_values(
      natural_payload,
      root_payload.alpha_row_count,
      root_payload.alpha_col_count,
      select_occ_by_mask(right_root_term.alpha_occ, used_alpha_row_mask),
      select_occ_by_mask(left_root_term.alpha_occ, used_alpha_col_mask),
      empty_labels,
      empty_labels,
      frontier_payload.alpha_row_count,
      frontier_payload.alpha_col_count,
      root_payload.beta_row_count,
      root_payload.beta_col_count,
      select_occ_by_mask(right_root_term.beta_occ, used_beta_row_mask),
      select_occ_by_mask(left_root_term.beta_occ, used_beta_col_mask),
      empty_labels,
      empty_labels,
      frontier_payload.beta_row_count,
      frontier_payload.beta_col_count);
}

HamiltonianEntryDensePayload close_root_hamiltonian_entry_dense_payload(
    const HamiltonianEntryDensePayload& frontier_payload,
    const HamiltonianEntryDensePayload& root_payload,
    const OrientationTerm& left_root_term,
    const OrientationTerm& right_root_term,
    std::uint32_t used_alpha_row_mask,
    std::uint32_t used_alpha_col_mask,
    std::uint32_t used_beta_row_mask,
    std::uint32_t used_beta_col_mask) {
  const HamiltonianEntryDensePayload natural_payload =
      merge_hamiltonian_entry_dense_payloads_limited_values(
          frontier_payload,
          root_payload);
  if (!is_hamiltonian_entry_dense_payload_nonzero(natural_payload)) {
    return natural_payload;
  }

  static const std::vector<int> empty_labels;
  return transform_hamiltonian_entry_dense_payload_interface_block_to_front_values(
      natural_payload,
      root_payload.alpha_row_count,
      root_payload.alpha_col_count,
      select_occ_by_mask(right_root_term.alpha_occ, used_alpha_row_mask),
      select_occ_by_mask(left_root_term.alpha_occ, used_alpha_col_mask),
      empty_labels,
      empty_labels,
      frontier_payload.alpha_row_count,
      frontier_payload.alpha_col_count,
      root_payload.beta_row_count,
      root_payload.beta_col_count,
      select_occ_by_mask(right_root_term.beta_occ, used_beta_row_mask),
      select_occ_by_mask(left_root_term.beta_occ, used_beta_col_mask),
      empty_labels,
      empty_labels,
      frontier_payload.beta_row_count,
      frontier_payload.beta_col_count);
}

struct SingleNodeCollapsedAggregateBundle {
  ComponentSpinCoefficientOperator component_operator;
  std::vector<DirectSpinStateAggregate> alpha_aggregates;
  std::vector<DirectSpinStateAggregate> beta_aggregates;
  std::vector<Eigen::MatrixXd> beta_potential_first_cofactors;
  std::uint64_t subdeterminant_evaluations = 0;
};

SingleNodeCollapsedAggregateBundle build_single_node_collapsed_aggregate_bundle(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentData& node_component,
    const DeterminantOverlapResolver& overlap_resolver) {
  // Exact one-node closure in compressed full-state form.
  //
  // Unlike the old typed deleted-minor root closure, this path first
  // compresses the component into unique alpha/beta left-right state pairs,
  // builds each one-spin exact direct aggregate once, and then contracts the
  // Hamiltonian channels over the sparse coefficient operator. This removes
  // repeated payload construction across orientation-term pairs.
  if (support_size <= 0) {
    throw std::invalid_argument("support_size must be positive");
  }

  SingleNodeCollapsedAggregateBundle bundle;
  bundle.component_operator = build_component_spin_coefficient_operator(node_component);
  bundle.alpha_aggregates.reserve(bundle.component_operator.alpha_states.size());
  bundle.beta_aggregates.reserve(bundle.component_operator.beta_states.size());
  bundle.beta_potential_first_cofactors.reserve(
      bundle.component_operator.beta_states.size());

  const DeterminantHamiltonianResolver hamiltonian_resolver(overlap_resolver);
  for (const auto& alpha_state : bundle.component_operator.alpha_states) {
    bundle.alpha_aggregates.push_back(
        build_direct_spin_state_aggregate(
            alpha_state.left_occ,
            alpha_state.right_occ,
            support_overlap_storage,
            support_one_electron_storage,
            packed_active_two_electron_integrals,
            support_size,
            true,
            overlap_resolver,
            hamiltonian_resolver,
            &bundle.subdeterminant_evaluations));
  }
  for (const auto& beta_state : bundle.component_operator.beta_states) {
    bundle.beta_aggregates.push_back(
        build_direct_spin_state_aggregate(
            beta_state.left_occ,
            beta_state.right_occ,
            support_overlap_storage,
            support_one_electron_storage,
            packed_active_two_electron_integrals,
            support_size,
            true,
            overlap_resolver,
            hamiltonian_resolver,
            &bundle.subdeterminant_evaluations));
    bundle.beta_potential_first_cofactors.push_back(
        apply_opposite_spin_kernel_to_first_cofactor(
            bundle.beta_aggregates.back().first_cofactor,
            support_size,
            packed_active_two_electron_integrals));
  }

  return bundle;
}

ComponentTreeHamiltonianResult evaluate_single_node_hamiltonian_boundary_collapsed(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentData& node_component,
    const DeterminantOverlapResolver& overlap_resolver) {
  // Exact one-node closure in compressed direct-state form.
  //
  // A single component has no separator boundary at all, so the typed root
  // deleted-minor payload is unnecessary overhead. The compressed direct-state
  // bundle builds each unique alpha/beta determinant pair once and reuses it
  // across all Hamiltonian channels.
  ComponentTreeHamiltonianResult result;
  const SingleNodeCollapsedAggregateBundle bundle =
      build_single_node_collapsed_aggregate_bundle(
          support_overlap_storage,
          support_one_electron_storage,
          packed_active_two_electron_integrals,
          support_size,
          node_component,
          overlap_resolver);
  result.subtree_term_pair_count = bundle.component_operator.raw_nonzero_pair_count;
  result.subdeterminant_evaluations = bundle.subdeterminant_evaluations;

  for (const auto& entry : bundle.component_operator.entries) {
    const DirectSpinStateAggregate& alpha =
        bundle.alpha_aggregates[xmvb::to_size(entry.alpha_state_index)];
    const DirectSpinStateAggregate& beta =
        bundle.beta_aggregates[xmvb::to_size(entry.beta_state_index)];
    const Eigen::MatrixXd& beta_potential =
        bundle.beta_potential_first_cofactors[xmvb::to_size(
            entry.beta_state_index)];

    result.overlap +=
        entry.coefficient *
        alpha.overlap *
        beta.overlap;
    result.one_electron +=
        entry.coefficient *
        (alpha.one_electron * beta.overlap +
         alpha.overlap * beta.one_electron);
    result.same_spin_alpha_two_electron +=
        entry.coefficient *
        alpha.same_spin_two_electron *
        beta.overlap;
    result.same_spin_beta_two_electron +=
        entry.coefficient *
        alpha.overlap *
        beta.same_spin_two_electron;
    result.opposite_spin_two_electron +=
        entry.coefficient *
        contract_direct_opposite_spin_channel(
            alpha.first_cofactor,
            beta_potential);
  }

  result.two_electron =
      result.same_spin_alpha_two_electron +
      result.same_spin_beta_two_electron +
      result.opposite_spin_two_electron;
  result.total_electronic_hamiltonian =
      result.one_electron + result.two_electron;
  return result;
}

ComponentTreeHamiltonianResult evaluate_one_leaf_hamiltonian_boundary_bundle_collapsed(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const DeterminantOverlapResolver& overlap_resolver) {
  // Structural one-leaf boundary path:
  // 1. keep the current compressed root/leaf coefficient operators;
  // 2. materialize each unique one-spin root/leaf pair as the exported exact
  //    one-leaf sector message, then reorder it into the canonical dense
  //    `BoundarySpinBundle` carrier;
  // 3. contract the final Hamiltonian channels from those dense bundles.
  //
  // This is still an intermediate step toward the final `2^n -> 2^m` design
  // because the generic rooted-tree forward path is still payload-centric. But
  // it removes the older aggregate-only scaffold and keeps the one-leaf cache
  // on referenced state pairs instead of the full Cartesian product.
  if (support_size <= 0) {
    throw std::invalid_argument("support_size must be positive");
  }
  if (ordered_components.size() != 2U) {
    throw std::invalid_argument(
        "one-leaf boundary bundle path requires exactly two ordered components");
  }

  ComponentTreeHamiltonianResult result;
  const ComponentSpinCoefficientOperator root_operator =
      build_component_spin_coefficient_operator(ordered_components.front());
  const ComponentSpinCoefficientOperator leaf_operator =
      build_component_spin_coefficient_operator(ordered_components.back());
  result.subtree_term_pair_count =
      root_operator.raw_nonzero_pair_count *
      leaf_operator.raw_nonzero_pair_count;

  const PackedOneLeafBundle packed_bundle =
      build_one_leaf_packed_bundle(
          root_operator,
          leaf_operator,
          support_overlap_storage,
          support_size,
          overlap_resolver,
          &result.subdeterminant_evaluations);
  result.subtree_message_state_count = packed_bundle.total_sector_count;

  const PackedOneLeafBundleContractionResult contraction =
      contract_one_leaf_packed_bundle(
          packed_bundle,
          support_one_electron_storage,
          packed_active_two_electron_integrals);
  result.overlap = contraction.overlap;
  result.one_electron = contraction.one_electron;
  result.same_spin_alpha_two_electron =
      contraction.same_spin_alpha_two_electron;
  result.same_spin_beta_two_electron =
      contraction.same_spin_beta_two_electron;
  result.opposite_spin_two_electron =
      contraction.opposite_spin_two_electron;
  result.two_electron =
      result.same_spin_alpha_two_electron +
      result.same_spin_beta_two_electron +
      result.opposite_spin_two_electron;
  result.total_electronic_hamiltonian =
      result.one_electron + result.two_electron;
  return result;
}

int component_ordered_block_parity(
    int n_root_occ,
    const std::vector<std::uint32_t>& selected_masks,
    const std::vector<int>& leaf_occ_sizes) {
  if (selected_masks.size() != leaf_occ_sizes.size()) {
    throw std::invalid_argument("selected_masks and leaf_occ_sizes must have the same length");
  }
  std::vector<int> block_order;
  block_order.reserve(
      xmvb::to_size(n_root_occ) +
      std::accumulate(leaf_occ_sizes.begin(), leaf_occ_sizes.end(), 0));

  std::uint32_t used_mask = 0U;
  int next_leaf_label = n_root_occ;
  for (std::size_t leaf_index = 0; leaf_index < selected_masks.size(); ++leaf_index) {
    const std::uint32_t mask = selected_masks[leaf_index];
    used_mask |= mask;
    for (int root_position = 0; root_position < n_root_occ; ++root_position) {
      if ((mask & (static_cast<std::uint32_t>(1U) << root_position)) != 0U) {
        block_order.push_back(root_position);
      }
    }
    for (int leaf_position = 0;
         leaf_position < leaf_occ_sizes[xmvb::to_size(leaf_index)];
         ++leaf_position) {
      block_order.push_back(next_leaf_label++);
    }
  }

  const std::uint32_t full_mask =
      (n_root_occ == 0) ? 0U
                        : ((static_cast<std::uint32_t>(1U) << n_root_occ) - 1U);
  const std::uint32_t remainder_mask = full_mask ^ used_mask;
  for (int root_position = 0; root_position < n_root_occ; ++root_position) {
    if ((remainder_mask & (static_cast<std::uint32_t>(1U) << root_position)) != 0U) {
      block_order.push_back(root_position);
    }
  }
  return canonicalization_parity(block_order);
}

void validate_tree(const ComponentTree& tree) {
  if (tree.components.empty()) {
    throw std::invalid_argument("component tree must not be empty");
  }
  if (tree.root_index < 0 ||
      tree.root_index >= static_cast<int>(tree.components.size())) {
    throw std::invalid_argument("component tree root index is out of range");
  }
  if (tree.children.size() != tree.components.size()) {
    throw std::invalid_argument("component tree children size mismatch");
  }

  std::vector<int> parent_count(tree.components.size(), 0);
  for (int node = 0; node < static_cast<int>(tree.children.size()); ++node) {
    for (const int child : tree.children[xmvb::to_size(node)]) {
      if (child < 0 || child >= static_cast<int>(tree.components.size())) {
        throw std::invalid_argument("component tree child index is out of range");
      }
      ++parent_count[xmvb::to_size(child)];
    }
  }
  if (parent_count[xmvb::to_size(tree.root_index)] != 0) {
    throw std::invalid_argument("component tree root must not have a parent");
  }
  for (int node = 0; node < static_cast<int>(parent_count.size()); ++node) {
    if (node == tree.root_index) {
      continue;
    }
    if (parent_count[xmvb::to_size(node)] != 1) {
      throw std::invalid_argument("component tree must assign exactly one parent to each non-root node");
    }
  }

  std::vector<char> visited(tree.components.size(), 0);
  const std::function<void(int)> dfs =
      [&](int node) {
        if (visited[xmvb::to_size(node)] != 0) {
          throw std::invalid_argument("component tree contains a cycle or repeated child");
        }
        visited[xmvb::to_size(node)] = 1;
        for (const int child : tree.children[xmvb::to_size(node)]) {
          dfs(child);
        }
      };
  dfs(tree.root_index);
  for (const char seen : visited) {
    if (seen == 0) {
      throw std::invalid_argument("component tree contains unreachable nodes");
    }
  }
}

ComponentSpinSizes component_spin_sizes(const ComponentData& component) {
  if (component.left_orientation_terms.empty() ||
      component.right_orientation_terms.empty()) {
    throw std::invalid_argument("component orientation terms must not be empty");
  }
  ComponentSpinSizes sizes{
      static_cast<int>(component.left_orientation_terms.front().alpha_occ.size()),
      static_cast<int>(component.left_orientation_terms.front().beta_occ.size()),
      static_cast<int>(component.right_orientation_terms.front().alpha_occ.size()),
      static_cast<int>(component.right_orientation_terms.front().beta_occ.size()),
  };

  const auto validate_side =
      [&](const std::vector<OrientationTerm>& terms,
          int expected_alpha,
          int expected_beta) {
        for (const auto& term : terms) {
          if (static_cast<int>(term.alpha_occ.size()) != expected_alpha ||
              static_cast<int>(term.beta_occ.size()) != expected_beta) {
            throw std::invalid_argument(
                "all orientation terms on one component side must have the same spin occupation counts");
          }
        }
      };
  validate_side(
      component.left_orientation_terms,
      sizes.left_alpha,
      sizes.left_beta);
  validate_side(
      component.right_orientation_terms,
      sizes.right_alpha,
      sizes.right_beta);
  return sizes;
}

void collect_subtree_preorder_nodes(
    const ComponentTree& tree,
    int node,
    std::vector<int>* preorder_nodes) {
  if (preorder_nodes == nullptr) {
    throw std::invalid_argument("preorder_nodes must not be null");
  }
  preorder_nodes->push_back(node);
  for (const int child : tree.children[xmvb::to_size(node)]) {
    collect_subtree_preorder_nodes(tree, child, preorder_nodes);
  }
}

std::vector<SubtreeExpansion> build_subtree_expansions(const ComponentTree& tree) {
  std::vector<SubtreeExpansion> expansions(tree.components.size());
  const std::function<ComponentSpinSizes(int)> build_one =
      [&](int node) -> ComponentSpinSizes {
        SubtreeExpansion& expansion = expansions[xmvb::to_size(node)];
        collect_subtree_preorder_nodes(tree, node, &expansion.preorder_nodes);
        expansion.preorder_components.reserve(expansion.preorder_nodes.size());
        for (const int preorder_node : expansion.preorder_nodes) {
          expansion.preorder_components.push_back(
              tree.components[xmvb::to_size(preorder_node)]);
        }

        const ComponentSpinSizes local_sizes =
            component_spin_sizes(tree.components[xmvb::to_size(node)]);
        ComponentSpinSizes subtree_sizes = local_sizes;
        for (const int child : tree.children[xmvb::to_size(node)]) {
          const ComponentSpinSizes child_sizes = build_one(child);
          subtree_sizes.left_alpha += child_sizes.left_alpha;
          subtree_sizes.left_beta += child_sizes.left_beta;
          subtree_sizes.right_alpha += child_sizes.right_alpha;
          subtree_sizes.right_beta += child_sizes.right_beta;
        }
        expansion.spin_sizes = subtree_sizes;
        return subtree_sizes;
      };
  build_one(tree.root_index);
  return expansions;
}

void finalize_boundary_scalar_message(BoundaryScalarMessage* message) {
  // Compacts the dense sector tables into a sparse traversal list. Entries are
  // retained whenever either overlap or one-electron is nonzero, because
  // nullity-1 sectors can carry a nonzero one-electron contribution even when
  // the overlap determinant itself vanishes.
  if (message == nullptr) {
    throw std::invalid_argument("boundary scalar message must not be null");
  }
  message->nonzero_entries.clear();
  const int alpha_sector_count = message->alpha_indexer.sector_count();
  const int beta_sector_count = message->beta_indexer.sector_count();
  for (int alpha_sector_index = 0;
       alpha_sector_index < alpha_sector_count;
       ++alpha_sector_index) {
    for (int beta_sector_index = 0;
         beta_sector_index < beta_sector_count;
         ++beta_sector_index) {
      const int flat_index =
          message->flat_index(alpha_sector_index, beta_sector_index);
      const double overlap = message->overlap_values[xmvb::to_size(flat_index)];
      const double one_electron =
          message->one_electron_values[xmvb::to_size(flat_index)];
      if (std::abs(overlap) <= 1.0e-15 &&
          std::abs(one_electron) <= 1.0e-15) {
        continue;
      }
      message->nonzero_entries.push_back(BoundaryScalarSectorEntry{
          .alpha_sector_index = alpha_sector_index,
          .beta_sector_index = beta_sector_index,
          .overlap = overlap,
          .one_electron = one_electron,
      });
    }
  }
}

void finalize_boundary_hamiltonian_message(
    BoundaryHamiltonianMessage* message,
    bool preserve_structural_zero_messages) {
  // Compacts the dense joint sector table into one traversal list. In reverse
  // mode we still need structurally zero entries whenever the sector basis is
  // present, so that the recursive pullback can revisit the same combinatorial
  // states as the forward pass.
  if (message == nullptr) {
    throw std::invalid_argument("boundary hamiltonian message must not be null");
  }
  message->nonzero_entries.clear();
  const int alpha_sector_count = message->alpha_layout.total_sector_count();
  const int beta_sector_count = message->beta_layout.total_sector_count();
  for (int alpha_flat_sector_index = 0;
       alpha_flat_sector_index < alpha_sector_count;
       ++alpha_flat_sector_index) {
    for (int beta_flat_sector_index = 0;
         beta_flat_sector_index < beta_sector_count;
         ++beta_flat_sector_index) {
      const int flat_index =
          message->flat_index(alpha_flat_sector_index, beta_flat_sector_index);
      HamiltonianBoundaryPayload& payload =
          message->payload_values[xmvb::to_size(flat_index)];
      cleanup_hamiltonian_payload(&payload);
      const bool keep_entry =
          preserve_structural_zero_messages
              ? has_hamiltonian_payload_basis(payload)
              : is_hamiltonian_payload_nonzero(payload);
      if (!keep_entry) {
        continue;
      }
      message->nonzero_entries.push_back(BoundaryHamiltonianSectorEntry{
          .alpha_flat_sector_index = alpha_flat_sector_index,
          .beta_flat_sector_index = beta_flat_sector_index,
          .flat_index = flat_index,
      });
    }
  }
}

DenseBoundaryHamiltonianMessage make_zero_dense_boundary_hamiltonian_message_like(
    const DenseBoundaryHamiltonianMessage& source) {
  DenseBoundaryHamiltonianMessage message;
  message.alpha_layout = source.alpha_layout;
  message.beta_layout = source.beta_layout;
  message.support_size = source.support_size;
  message.payload_values.resize(
      xmvb::to_size(message.alpha_layout.total_sector_count()) *
      xmvb::to_size(message.beta_layout.total_sector_count()));
  return message;
}

void finalize_dense_boundary_hamiltonian_message(
    DenseBoundaryHamiltonianMessage* message) {
  if (message == nullptr) {
    throw std::invalid_argument("dense boundary hamiltonian message must not be null");
  }
  message->nonzero_entries.clear();
  const int alpha_sector_count = message->alpha_layout.total_sector_count();
  const int beta_sector_count = message->beta_layout.total_sector_count();
  for (int alpha_flat_sector_index = 0;
       alpha_flat_sector_index < alpha_sector_count;
       ++alpha_flat_sector_index) {
    for (int beta_flat_sector_index = 0;
         beta_flat_sector_index < beta_sector_count;
         ++beta_flat_sector_index) {
      const int flat_index =
          message->flat_index(alpha_flat_sector_index, beta_flat_sector_index);
      HamiltonianEntryDensePayload& payload =
          message->payload_values[xmvb::to_size(flat_index)];
      cleanup_hamiltonian_entry_dense_payload(&payload);
      if (!is_hamiltonian_entry_dense_payload_nonzero(payload)) {
        continue;
      }
      message->nonzero_entries.push_back(DenseBoundaryHamiltonianSectorEntry{
          .alpha_flat_sector_index = alpha_flat_sector_index,
          .beta_flat_sector_index = beta_flat_sector_index,
          .flat_index = flat_index,
      });
    }
  }
}

const BoundaryScalarMessage& build_recursive_subtree_boundary_scalar_cached(
    const ComponentTree& tree,
    const std::vector<SubtreeExpansion>& subtree_expansions,
    int node,
    const OrientationTerm& left_parent_term,
    const OrientationTerm& right_parent_term,
    const std::vector<double>& overlap_storage,
    const std::vector<double>& one_electron_storage,
    int n_orbitals,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subtree_term_pair_count,
    std::uint64_t* subtree_message_state_count,
    std::uint64_t* subdeterminant_evaluations,
    std::uint64_t* dp_transition_count,
    std::unordered_map<SpinOverlapCacheKey,
                       SpinScalarChannelValues,
                       SpinOverlapCacheKeyHasher>* overlap_cache,
    std::unordered_map<SubtreeMessageCacheKey,
                       BoundaryScalarMessage,
                       SubtreeMessageCacheKeyHasher>* message_cache) {
  // Exact boundary-only node->parent scalar message for overlap and total
  // one-electron.
  //
  // For the fixed parent determinant-term pair, the returned table is indexed
  // only by the selected parent-interface boundary masks. The subtree interior
  // is fully integrated out; no full-state payload or internal deleted-sector
  // map is carried upward.
  if (subtree_term_pair_count == nullptr ||
      subtree_message_state_count == nullptr ||
      subdeterminant_evaluations == nullptr ||
      dp_transition_count == nullptr ||
      overlap_cache == nullptr ||
      message_cache == nullptr) {
    throw std::invalid_argument("boundary overlap subtree inputs must not be null");
  }

  const SubtreeMessageCacheKey cache_key{
      node,
      left_parent_term.alpha_occ,
      left_parent_term.beta_occ,
      right_parent_term.alpha_occ,
      right_parent_term.beta_occ,
      false,
  };
  auto cache_iterator = message_cache->find(cache_key);
  if (cache_iterator != message_cache->end()) {
    return cache_iterator->second;
  }

  const ComponentData& node_component =
      tree.components[xmvb::to_size(node)];
  const std::vector<int>& children = tree.children[xmvb::to_size(node)];
  const ComponentSpinSizes& subtree_sizes =
      subtree_expansions[xmvb::to_size(node)].spin_sizes;

  BoundaryScalarMessage message;
  message.alpha_indexer = BoundarySectorIndexer(
      static_cast<int>(right_parent_term.alpha_occ.size()),
      static_cast<int>(left_parent_term.alpha_occ.size()),
      subtree_sizes.left_alpha - subtree_sizes.right_alpha);
  message.beta_indexer = BoundarySectorIndexer(
      static_cast<int>(right_parent_term.beta_occ.size()),
      static_cast<int>(left_parent_term.beta_occ.size()),
      subtree_sizes.left_beta - subtree_sizes.right_beta);
  message.overlap_values.assign(
      xmvb::to_size(message.alpha_indexer.sector_count()) *
          xmvb::to_size(message.beta_indexer.sector_count()),
      0.0);
  message.one_electron_values.assign(
      xmvb::to_size(message.alpha_indexer.sector_count()) *
          xmvb::to_size(message.beta_indexer.sector_count()),
      0.0);

  const ChildSpinSizeVectors child_spin_sizes =
      build_child_spin_size_vectors(children, subtree_expansions);

  const std::vector<WeightedOrientationTermPair> node_term_pairs =
      build_weighted_component_orientation_pairs(node_component);
  for (const auto& node_term_pair : node_term_pairs) {
      const OrientationTerm& left_node_term = node_term_pair.left_term;
      const OrientationTerm& right_node_term = node_term_pair.right_term;
      const double local_coefficient = node_term_pair.coefficient;
      ++(*subtree_term_pair_count);

      std::vector<const BoundaryScalarMessage*> child_messages(children.size(), nullptr);
      for (std::size_t child_index = 0; child_index < children.size(); ++child_index) {
        const int child = children[child_index];
        child_messages[child_index] =
            &build_recursive_subtree_boundary_scalar_cached(
                tree,
                subtree_expansions,
                child,
                left_node_term,
                right_node_term,
                overlap_storage,
                one_electron_storage,
                n_orbitals,
                overlap_resolver,
                subtree_term_pair_count,
                subtree_message_state_count,
                subdeterminant_evaluations,
                dp_transition_count,
                overlap_cache,
                message_cache);
      }

      const std::uint32_t local_alpha_row_full_mask =
          right_node_term.alpha_occ.empty()
              ? 0U
              : (open_state_mask_limit(static_cast<int>(right_node_term.alpha_occ.size())) - 1U);
      const std::uint32_t local_alpha_col_full_mask =
          left_node_term.alpha_occ.empty()
              ? 0U
              : (open_state_mask_limit(static_cast<int>(left_node_term.alpha_occ.size())) - 1U);
      const std::uint32_t local_beta_row_full_mask =
          right_node_term.beta_occ.empty()
              ? 0U
              : (open_state_mask_limit(static_cast<int>(right_node_term.beta_occ.size())) - 1U);
      const std::uint32_t local_beta_col_full_mask =
          left_node_term.beta_occ.empty()
              ? 0U
              : (open_state_mask_limit(static_cast<int>(left_node_term.beta_occ.size())) - 1U);

      std::vector<std::uint32_t> selected_alpha_row_masks(children.size(), 0U);
      std::vector<std::uint32_t> selected_alpha_col_masks(children.size(), 0U);
      std::vector<std::uint32_t> selected_beta_row_masks(children.size(), 0U);
      std::vector<std::uint32_t> selected_beta_col_masks(children.size(), 0U);

      for (int alpha_sector_index = 0;
           alpha_sector_index < message.alpha_indexer.sector_count();
           ++alpha_sector_index) {
        const BoundarySector& alpha_sector =
            message.alpha_indexer.sector(alpha_sector_index);
        const std::vector<int> selected_parent_alpha_rows = select_occ_by_mask(
            right_parent_term.alpha_occ,
            alpha_sector.row_mask);
        const std::vector<int> selected_parent_alpha_cols = select_occ_by_mask(
            left_parent_term.alpha_occ,
            alpha_sector.col_mask);

        for (int beta_sector_index = 0;
             beta_sector_index < message.beta_indexer.sector_count();
             ++beta_sector_index) {
          const BoundarySector& beta_sector =
              message.beta_indexer.sector(beta_sector_index);
          const std::vector<int> selected_parent_beta_rows = select_occ_by_mask(
              right_parent_term.beta_occ,
              beta_sector.row_mask);
          const std::vector<int> selected_parent_beta_cols = select_occ_by_mask(
              left_parent_term.beta_occ,
              beta_sector.col_mask);

          const auto accumulate_children =
              [&](const auto& self,
                  std::size_t child_index,
                  std::uint32_t used_alpha_row_mask,
                  std::uint32_t used_alpha_col_mask,
                  std::uint32_t used_beta_row_mask,
                  std::uint32_t used_beta_col_mask,
                  double child_overlap,
                  double child_one_electron) -> void {
                if (std::abs(child_overlap) <= 1.0e-15 &&
                    std::abs(child_one_electron) <= 1.0e-15) {
                  return;
                }

                if (child_index == children.size()) {
                  const std::vector<int> alpha_local_remainder_rows = select_occ_by_mask(
                      right_node_term.alpha_occ,
                      local_alpha_row_full_mask ^ used_alpha_row_mask);
                  const std::vector<int> alpha_local_remainder_cols = select_occ_by_mask(
                      left_node_term.alpha_occ,
                      local_alpha_col_full_mask ^ used_alpha_col_mask);
                  const std::vector<int> beta_local_remainder_rows = select_occ_by_mask(
                      right_node_term.beta_occ,
                      local_beta_row_full_mask ^ used_beta_row_mask);
                  const std::vector<int> beta_local_remainder_cols = select_occ_by_mask(
                      left_node_term.beta_occ,
                      local_beta_col_full_mask ^ used_beta_col_mask);

                  const SpinScalarChannelValues alpha_values =
                      build_exact_frontier_spin_overlap_one_electron(
                      selected_parent_alpha_cols,
                      alpha_local_remainder_cols,
                      selected_parent_alpha_rows,
                      alpha_local_remainder_rows,
                      overlap_storage,
                      one_electron_storage,
                      n_orbitals,
                      overlap_resolver,
                      subdeterminant_evaluations,
                      overlap_cache);
                  const SpinScalarChannelValues beta_values =
                      build_exact_frontier_spin_overlap_one_electron(
                      selected_parent_beta_cols,
                      beta_local_remainder_cols,
                      selected_parent_beta_rows,
                      beta_local_remainder_rows,
                      overlap_storage,
                      one_electron_storage,
                      n_orbitals,
                      overlap_resolver,
                      subdeterminant_evaluations,
                      overlap_cache);
                  const double local_overlap =
                      alpha_values.overlap * beta_values.overlap;
                  const double local_one_electron =
                      alpha_values.one_electron * beta_values.overlap +
                      beta_values.one_electron * alpha_values.overlap;
                  if (std::abs(local_overlap) <= 1.0e-15 &&
                      std::abs(local_one_electron) <= 1.0e-15) {
                    return;
                  }

                  int parity = 0;
                  parity ^= component_ordered_block_parity(
                      static_cast<int>(right_node_term.alpha_occ.size()),
                      selected_alpha_row_masks,
                      child_spin_sizes.right_alpha);
                  parity ^= component_ordered_block_parity(
                      static_cast<int>(left_node_term.alpha_occ.size()),
                      selected_alpha_col_masks,
                      child_spin_sizes.left_alpha);
                  parity ^= component_ordered_block_parity(
                      static_cast<int>(right_node_term.beta_occ.size()),
                      selected_beta_row_masks,
                      child_spin_sizes.right_beta);
                  parity ^= component_ordered_block_parity(
                      static_cast<int>(left_node_term.beta_occ.size()),
                      selected_beta_col_masks,
                      child_spin_sizes.left_beta);

                  const double combined_overlap = child_overlap * local_overlap;
                  const double combined_one_electron =
                      child_one_electron * local_overlap +
                      child_overlap * local_one_electron;
                  const double signed_scale =
                      local_coefficient * parity_sign(parity);
                  const int flat_index =
                      message.flat_index(alpha_sector_index, beta_sector_index);
                  message.overlap_values[xmvb::to_size(flat_index)] +=
                      signed_scale * combined_overlap;
                  message.one_electron_values[xmvb::to_size(flat_index)] +=
                      signed_scale * combined_one_electron;
                  return;
                }

                for (const auto& child_entry :
                     child_messages[xmvb::to_size(child_index)]->nonzero_entries) {
                  const BoundarySector& child_alpha_sector =
                      child_messages[xmvb::to_size(child_index)]
                          ->alpha_indexer.sector(child_entry.alpha_sector_index);
                  const BoundarySector& child_beta_sector =
                      child_messages[xmvb::to_size(child_index)]
                          ->beta_indexer.sector(child_entry.beta_sector_index);
                  if ((used_alpha_row_mask & child_alpha_sector.row_mask) != 0U ||
                      (used_alpha_col_mask & child_alpha_sector.col_mask) != 0U ||
                      (used_beta_row_mask & child_beta_sector.row_mask) != 0U ||
                      (used_beta_col_mask & child_beta_sector.col_mask) != 0U) {
                    continue;
                  }

                  selected_alpha_row_masks[child_index] = child_alpha_sector.row_mask;
                  selected_alpha_col_masks[child_index] = child_alpha_sector.col_mask;
                  selected_beta_row_masks[child_index] = child_beta_sector.row_mask;
                  selected_beta_col_masks[child_index] = child_beta_sector.col_mask;

                  ++(*dp_transition_count);
                  const double next_overlap =
                      child_overlap * child_entry.overlap;
                  const double next_one_electron =
                      child_one_electron * child_entry.overlap +
                      child_overlap * child_entry.one_electron;
                  self(
                      self,
                      child_index + 1,
                      used_alpha_row_mask | child_alpha_sector.row_mask,
                      used_alpha_col_mask | child_alpha_sector.col_mask,
                      used_beta_row_mask | child_beta_sector.row_mask,
                      used_beta_col_mask | child_beta_sector.col_mask,
                      next_overlap,
                      next_one_electron);

                  selected_alpha_row_masks[child_index] = 0U;
                  selected_alpha_col_masks[child_index] = 0U;
                  selected_beta_row_masks[child_index] = 0U;
                  selected_beta_col_masks[child_index] = 0U;
                }
              };

          accumulate_children(
              accumulate_children,
              0U,
              0U,
              0U,
              0U,
              0U,
              1.0,
              0.0);
        }
      }
  }

  finalize_boundary_scalar_message(&message);
  *subtree_message_state_count +=
      static_cast<std::uint64_t>(message.overlap_values.size());
  return message_cache->emplace(cache_key, std::move(message)).first->second;
}

bool recursive_message_debug_validation_enabled() {
  const char* debug_flag = std::getenv("XMVB_DEBUG_EXACT_SEPARATOR_MESSAGES");
  return debug_flag != nullptr && debug_flag[0] != '\0' && debug_flag[0] != '0';
}

bool typed_hamiltonian_production_fallback_enabled() {
  const char* debug_flag =
      std::getenv("XMVB_EXACT_SEPARATOR_FORCE_TYPED_HAMILTONIAN_PRODUCTION");
  return debug_flag != nullptr && debug_flag[0] != '\0' && debug_flag[0] != '0';
}

bool dense_forward_hamiltonian_message_compare_enabled() {
  const char* debug_flag =
      std::getenv("XMVB_DEBUG_EXACT_SEPARATOR_DENSE_HAMILTONIAN_COMPARE");
  return debug_flag != nullptr && debug_flag[0] != '\0' && debug_flag[0] != '0';
}

std::string format_label_list(const std::vector<int>& labels) {
  std::ostringstream stream;
  stream << '[';
  for (std::size_t index = 0; index < labels.size(); ++index) {
    if (index > 0U) {
      stream << ',';
    }
    stream << labels[index];
  }
  stream << ']';
  return stream.str();
}

std::string format_spin_key(const SpinDeletionKey& key) {
  std::ostringstream stream;
  stream << "{rows=" << format_label_list(key.row_labels)
         << ", cols=" << format_label_list(key.col_labels) << '}';
  return stream.str();
}

std::string format_joint_key(const JointDeletionKey& key) {
  std::ostringstream stream;
  stream << "{alpha=" << format_spin_key(key.alpha_key)
         << ", beta=" << format_spin_key(key.beta_key) << '}';
  return stream.str();
}

std::string format_message_mask(const ComponentTreeLeafMessage& message) {
  std::ostringstream stream;
  stream << "(ar=" << message.alpha_row_mask
         << ", ac=" << message.alpha_col_mask
         << ", br=" << message.beta_row_mask
         << ", bc=" << message.beta_col_mask << ')';
  return stream.str();
}

std::string format_joint_payload_summary(
    const JointDeletionPayload& payload,
    std::size_t max_entries = 16U) {
  std::ostringstream stream;
  stream << "dims=("
         << payload.alpha_row_count << ','
         << payload.alpha_col_count << ','
         << payload.beta_row_count << ','
         << payload.beta_col_count << "); sectors={";
  std::size_t printed = 0U;
  for (const auto& [key, value] : payload.sectors) {
    if (printed > 0U) {
      stream << "; ";
    }
    stream << format_joint_key(key) << '=' << value;
    ++printed;
    if (printed >= max_entries) {
      if (payload.sectors.size() > printed) {
        stream << "; ...";
      }
      break;
    }
  }
  stream << '}';
  return stream.str();
}

std::string format_hamiltonian_boundary_payload_summary(
    const HamiltonianBoundaryPayload& payload,
    std::size_t max_entries = 16U) {
  return format_joint_payload_summary(
      convert_hamiltonian_payload_to_joint(payload),
      max_entries);
}

std::string format_dense_hamiltonian_entry_payload_summary(
    const HamiltonianEntryDensePayload& payload,
    std::size_t max_entries = 16U) {
  return format_hamiltonian_boundary_payload_summary(
      project_dense_hamiltonian_entry_payload_to_boundary_values(payload),
      max_entries);
}

void validate_dense_projected_hamiltonian_payload_against_reference(
    const char* stage,
    const HamiltonianEntryDensePayload& dense_payload,
    const HamiltonianBoundaryPayload& reference_payload,
    const std::string& context) {
  if (!dense_forward_hamiltonian_message_compare_enabled()) {
    return;
  }

  JointDeletionPayload dense_joint = convert_hamiltonian_payload_to_joint(
      project_dense_hamiltonian_entry_payload_to_boundary_values(dense_payload));
  JointDeletionPayload reference_joint =
      convert_hamiltonian_payload_to_joint(reference_payload);
  cleanup_joint_payload(&dense_joint);
  cleanup_joint_payload(&reference_joint);

  constexpr double tolerance = 1.0e-12;
  if (dense_joint.alpha_row_count != reference_joint.alpha_row_count ||
      dense_joint.alpha_col_count != reference_joint.alpha_col_count ||
      dense_joint.beta_row_count != reference_joint.beta_row_count ||
      dense_joint.beta_col_count != reference_joint.beta_col_count) {
    std::ostringstream stream;
    stream << stage << " dimension mismatch";
    if (!context.empty()) {
      stream << " (" << context << ')';
    }
    stream << ": dense=("
           << dense_joint.alpha_row_count << ','
           << dense_joint.alpha_col_count << ','
           << dense_joint.beta_row_count << ','
           << dense_joint.beta_col_count << "), reference=("
           << reference_joint.alpha_row_count << ','
           << reference_joint.alpha_col_count << ','
           << reference_joint.beta_row_count << ','
           << reference_joint.beta_col_count << ')';
    throw std::runtime_error(stream.str());
  }

  std::map<JointDeletionKey, double> all_sectors = dense_joint.sectors;
  for (const auto& [key, value] : reference_joint.sectors) {
    all_sectors.emplace(key, value);
  }
  for (const auto& [key, value] : all_sectors) {
    const auto dense_iterator = dense_joint.sectors.find(key);
    const auto reference_iterator = reference_joint.sectors.find(key);
    const double dense_value =
        (dense_iterator == dense_joint.sectors.end()) ? 0.0 : dense_iterator->second;
    const double reference_value =
        (reference_iterator == reference_joint.sectors.end())
            ? 0.0
            : reference_iterator->second;
    if (std::abs(dense_value - reference_value) > tolerance) {
      std::ostringstream stream;
      stream << stage << " sector mismatch";
      if (!context.empty()) {
        stream << " (" << context << ')';
      }
      stream << ": key " << format_joint_key(key)
             << ", dense=" << dense_value
             << ", reference=" << reference_value
             << "\ndense_payload: "
             << format_dense_hamiltonian_entry_payload_summary(dense_payload)
             << "\nreference_payload: "
             << format_hamiltonian_boundary_payload_summary(reference_payload);
      throw std::runtime_error(stream.str());
    }
  }
}

std::vector<ComponentTreeLeafMessage> build_direct_subtree_messages_exact(
    const SubtreeExpansion& subtree_expansion,
    const OrientationTerm& left_parent_term,
    const OrientationTerm& right_parent_term,
    bool preserve_structural_zero_messages,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    std::unordered_map<SpinPayloadCacheKey,
                       SpinDeletionPayload,
                       SpinPayloadCacheKeyHasher>* payload_cache) {
  // Rebuilds the old "exact subtree as leaf" reference message for one
  // oriented edge `(node -> parent)`.
  //
  // The subtree is flattened into canonical global determinant terms over all
  // preorder components below `node`, then treated exactly like one large leaf
  // attached to the parent interface. This helper is intentionally expensive
  // and is used only as a correctness oracle for debugging the recursive
  // message recurrence.
  if (subdeterminant_evaluations == nullptr || payload_cache == nullptr) {
    throw std::invalid_argument("direct subtree-message scratch outputs must not be null");
  }

  const std::vector<GlobalOrientationTerm> left_global_terms =
      build_global_orientation_terms(subtree_expansion.preorder_components, true);
  const std::vector<GlobalOrientationTerm> right_global_terms =
      build_global_orientation_terms(subtree_expansion.preorder_components, false);
  const auto left_parent_alpha_occ_by_mask = build_mask_occ_table(left_parent_term.alpha_occ);
  const auto right_parent_alpha_occ_by_mask = build_mask_occ_table(right_parent_term.alpha_occ);
  const auto left_parent_beta_occ_by_mask = build_mask_occ_table(left_parent_term.beta_occ);
  const auto right_parent_beta_occ_by_mask = build_mask_occ_table(right_parent_term.beta_occ);

  const std::uint32_t alpha_row_limit =
      open_state_mask_limit(static_cast<int>(right_parent_term.alpha_occ.size()));
  const std::uint32_t alpha_col_limit =
      open_state_mask_limit(static_cast<int>(left_parent_term.alpha_occ.size()));
  const std::uint32_t beta_row_limit =
      open_state_mask_limit(static_cast<int>(right_parent_term.beta_occ.size()));
  const std::uint32_t beta_col_limit =
      open_state_mask_limit(static_cast<int>(left_parent_term.beta_occ.size()));

  std::map<
      std::tuple<std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>,
      JointDeletionPayload>
      aggregated_messages;
  for (const auto& left_global_term : left_global_terms) {
    for (const auto& right_global_term : right_global_terms) {
      const double coefficient =
          left_global_term.coefficient * right_global_term.coefficient;
      if (std::abs(coefficient) <= 1.0e-15) {
        continue;
      }

      for (std::uint32_t alpha_row_mask = 0; alpha_row_mask < alpha_row_limit; ++alpha_row_mask) {
        for (std::uint32_t alpha_col_mask = 0;
             alpha_col_mask < alpha_col_limit;
             ++alpha_col_mask) {
          for (std::uint32_t beta_row_mask = 0; beta_row_mask < beta_row_limit; ++beta_row_mask) {
            for (std::uint32_t beta_col_mask = 0;
                 beta_col_mask < beta_col_limit;
                 ++beta_col_mask) {
              const auto& selected_parent_alpha_rows =
                  right_parent_alpha_occ_by_mask[xmvb::to_size(alpha_row_mask)];
              const auto& selected_parent_alpha_cols =
                  left_parent_alpha_occ_by_mask[xmvb::to_size(alpha_col_mask)];
              const auto& selected_parent_beta_rows =
                  right_parent_beta_occ_by_mask[xmvb::to_size(beta_row_mask)];
              const auto& selected_parent_beta_cols =
                  left_parent_beta_occ_by_mask[xmvb::to_size(beta_col_mask)];

              const SpinDeletionPayload alpha_payload =
                  build_exact_frontier_spin_payload(
                      selected_parent_alpha_cols,
                      left_global_term.alpha_occ,
                      selected_parent_alpha_rows,
                      right_global_term.alpha_occ,
                      overlap_storage,
                      n_orbitals,
                      overlap_resolver,
                      subdeterminant_evaluations,
                      payload_cache);
              const SpinDeletionPayload beta_payload =
                  build_exact_frontier_spin_payload(
                      selected_parent_beta_cols,
                      left_global_term.beta_occ,
                      selected_parent_beta_rows,
                      right_global_term.beta_occ,
                      overlap_storage,
                      n_orbitals,
                      overlap_resolver,
                      subdeterminant_evaluations,
                      payload_cache);
              const JointDeletionPayload joint_payload =
                  build_joint_payload(alpha_payload, beta_payload);
              add_scaled_joint_payload(
                  joint_payload,
                  coefficient,
                  &aggregated_messages[std::make_tuple(
                      alpha_row_mask,
                      alpha_col_mask,
                      beta_row_mask,
                      beta_col_mask)]);
            }
          }
        }
      }
    }
  }

  std::vector<ComponentTreeLeafMessage> messages;
  messages.reserve(aggregated_messages.size());
  for (auto& [key, payload] : aggregated_messages) {
    cleanup_joint_payload(&payload);
    if (!preserve_structural_zero_messages &&
        !is_joint_payload_nonzero(payload)) {
      continue;
    }
    if (preserve_structural_zero_messages && payload.basis_keys.empty()) {
      continue;
    }
    messages.push_back(ComponentTreeLeafMessage{
        .alpha_row_mask = std::get<0>(key),
        .alpha_col_mask = std::get<1>(key),
        .beta_row_mask = std::get<2>(key),
        .beta_col_mask = std::get<3>(key),
        .payload = std::move(payload),
    });
  }
  return messages;
}

void validate_recursive_subtree_messages_against_direct_reference(
    const SubtreeExpansion& subtree_expansion,
    int node,
    const OrientationTerm& left_parent_term,
    const OrientationTerm& right_parent_term,
    bool preserve_structural_zero_messages,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const DeterminantOverlapResolver& overlap_resolver,
    const std::vector<ComponentTreeLeafMessage>& recursive_messages) {
  // This debug-only validator compares the recursive subtree message against
  // the previous exact oracle that flattens the whole subtree into one giant
  // leaf. Matching these two objects is the precise requirement for replacing
  // the old implementation with a true recursive recurrence.
  std::uint64_t debug_subdeterminant_evaluations = 0;
  std::unordered_map<SpinPayloadCacheKey,
                     SpinDeletionPayload,
                     SpinPayloadCacheKeyHasher> debug_payload_cache;
  const std::vector<ComponentTreeLeafMessage> direct_messages =
      build_direct_subtree_messages_exact(
          subtree_expansion,
          left_parent_term,
          right_parent_term,
          preserve_structural_zero_messages,
          overlap_storage,
          n_orbitals,
          overlap_resolver,
          &debug_subdeterminant_evaluations,
          &debug_payload_cache);

  if (recursive_messages.size() != direct_messages.size()) {
    std::ostringstream stream;
    stream << "recursive subtree-message count mismatch at node " << node
           << ": recursive=" << recursive_messages.size()
           << ", direct=" << direct_messages.size();
    throw std::runtime_error(stream.str());
  }

  constexpr double tolerance = 1.0e-12;
  for (std::size_t message_index = 0;
       message_index < recursive_messages.size();
       ++message_index) {
    const ComponentTreeLeafMessage& recursive_message =
        recursive_messages[message_index];
    const ComponentTreeLeafMessage& direct_message =
        direct_messages[message_index];
    if (recursive_message.alpha_row_mask != direct_message.alpha_row_mask ||
        recursive_message.alpha_col_mask != direct_message.alpha_col_mask ||
        recursive_message.beta_row_mask != direct_message.beta_row_mask ||
        recursive_message.beta_col_mask != direct_message.beta_col_mask) {
      std::ostringstream stream;
      stream << "recursive subtree-message mask mismatch at node " << node
             << ": recursive=" << format_message_mask(recursive_message)
             << ", direct=" << format_message_mask(direct_message);
      throw std::runtime_error(stream.str());
    }

    const JointDeletionPayload& recursive_payload = recursive_message.payload;
    const JointDeletionPayload& direct_payload = direct_message.payload;
    if (recursive_payload.alpha_row_count != direct_payload.alpha_row_count ||
        recursive_payload.alpha_col_count != direct_payload.alpha_col_count ||
        recursive_payload.beta_row_count != direct_payload.beta_row_count ||
        recursive_payload.beta_col_count != direct_payload.beta_col_count) {
      std::ostringstream stream;
      stream << "recursive subtree-message dimension mismatch at node " << node
             << " for mask " << format_message_mask(recursive_message)
             << ": recursive=("
             << recursive_payload.alpha_row_count << ','
             << recursive_payload.alpha_col_count << ','
             << recursive_payload.beta_row_count << ','
             << recursive_payload.beta_col_count << "), direct=("
             << direct_payload.alpha_row_count << ','
             << direct_payload.alpha_col_count << ','
             << direct_payload.beta_row_count << ','
             << direct_payload.beta_col_count << ')';
      throw std::runtime_error(stream.str());
    }

    if (recursive_payload.basis_keys != direct_payload.basis_keys) {
      std::vector<JointDeletionKey> missing_in_recursive;
      std::vector<JointDeletionKey> extra_in_recursive;
      std::set_difference(
          direct_payload.basis_keys.begin(),
          direct_payload.basis_keys.end(),
          recursive_payload.basis_keys.begin(),
          recursive_payload.basis_keys.end(),
          std::back_inserter(missing_in_recursive));
      std::set_difference(
          recursive_payload.basis_keys.begin(),
          recursive_payload.basis_keys.end(),
          direct_payload.basis_keys.begin(),
          direct_payload.basis_keys.end(),
          std::back_inserter(extra_in_recursive));
      std::ostringstream stream;
      stream << "recursive subtree-message basis mismatch at node " << node
             << " for mask " << format_message_mask(recursive_message);
      if (!missing_in_recursive.empty()) {
        stream << "\nmissing_in_recursive=";
        for (std::size_t index = 0; index < missing_in_recursive.size(); ++index) {
          if (index > 0U) {
            stream << "; ";
          }
          stream << format_joint_key(missing_in_recursive[index]);
        }
      }
      if (!extra_in_recursive.empty()) {
        stream << "\nextra_in_recursive=";
        for (std::size_t index = 0; index < extra_in_recursive.size(); ++index) {
          if (index > 0U) {
            stream << "; ";
          }
          stream << format_joint_key(extra_in_recursive[index]);
        }
      }
      stream
             << "\nrecursive_payload: "
             << format_joint_payload_summary(recursive_payload)
             << "\ndirect_payload: "
             << format_joint_payload_summary(direct_payload);
      throw std::runtime_error(stream.str());
    }

    std::map<JointDeletionKey, double> all_sectors = recursive_payload.sectors;
    for (const auto& [key, value] : direct_payload.sectors) {
      all_sectors.emplace(key, value);
    }
    for (const auto& [key, value] : all_sectors) {
      const auto recursive_iterator = recursive_payload.sectors.find(key);
      const auto direct_iterator = direct_payload.sectors.find(key);
      const double recursive_value =
          (recursive_iterator == recursive_payload.sectors.end())
              ? 0.0
              : recursive_iterator->second;
      const double direct_value =
          (direct_iterator == direct_payload.sectors.end())
              ? 0.0
              : direct_iterator->second;
      if (std::abs(recursive_value - direct_value) > tolerance) {
        std::ostringstream stream;
        stream << "recursive subtree-message sector mismatch at node " << node
               << " for mask " << format_message_mask(recursive_message)
               << ", sector " << format_joint_key(key)
               << ": recursive=" << recursive_value
               << ", direct=" << direct_value
               << "\nrecursive_payload: "
               << format_joint_payload_summary(recursive_payload)
               << "\ndirect_payload: "
               << format_joint_payload_summary(direct_payload);
        throw std::runtime_error(stream.str());
      }
    }
  }
}

void validate_dense_boundary_hamiltonian_message_against_typed_reference(
    const ComponentTree& tree,
    const std::vector<SubtreeExpansion>& subtree_expansions,
    int node,
    const OrientationTerm& left_parent_term,
    const OrientationTerm& right_parent_term,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const DeterminantOverlapResolver& overlap_resolver,
    const DenseBoundaryHamiltonianMessage& dense_message) {
  if (!dense_forward_hamiltonian_message_compare_enabled()) {
    return;
  }

  std::uint64_t typed_subtree_term_pair_count = 0;
  std::uint64_t typed_subtree_message_state_count = 0;
  std::uint64_t typed_subdeterminant_evaluations = 0;
  std::uint64_t typed_dp_transition_count = 0;
  std::unordered_map<SpinPayloadCacheKey,
                     SpinDeletionPayload,
                     SpinPayloadCacheKeyHasher> typed_payload_cache;
  std::unordered_map<SubtreeMessageCacheKey,
                     BoundaryHamiltonianMessage,
                     SubtreeMessageCacheKeyHasher> typed_message_cache;
  const BoundaryHamiltonianMessage& typed_message =
      build_recursive_subtree_hamiltonian_messages_cached(
          tree,
          subtree_expansions,
          node,
          left_parent_term,
          right_parent_term,
          false,
          overlap_storage,
          n_orbitals,
          overlap_resolver,
          &typed_subtree_term_pair_count,
          &typed_subtree_message_state_count,
          &typed_subdeterminant_evaluations,
          &typed_dp_transition_count,
          &typed_payload_cache,
          &typed_message_cache);

  if (dense_message.alpha_layout.total_sector_count() !=
          typed_message.alpha_layout.total_sector_count() ||
      dense_message.beta_layout.total_sector_count() !=
          typed_message.beta_layout.total_sector_count()) {
    std::ostringstream stream;
    stream << "dense boundary Hamiltonian layout count mismatch at node " << node
           << ": dense=("
           << dense_message.alpha_layout.total_sector_count() << ','
           << dense_message.beta_layout.total_sector_count() << "), typed=("
           << typed_message.alpha_layout.total_sector_count() << ','
           << typed_message.beta_layout.total_sector_count() << ')';
    throw std::runtime_error(stream.str());
  }

  constexpr double tolerance = 1.0e-12;
  for (int alpha_flat_sector_index = 0;
       alpha_flat_sector_index < dense_message.alpha_layout.total_sector_count();
       ++alpha_flat_sector_index) {
    const BoundarySector& dense_alpha_sector =
        boundary_layout_sector(dense_message.alpha_layout, alpha_flat_sector_index);
    const BoundarySector& typed_alpha_sector =
        boundary_layout_sector(typed_message.alpha_layout, alpha_flat_sector_index);
    if (dense_alpha_sector.row_mask != typed_alpha_sector.row_mask ||
        dense_alpha_sector.col_mask != typed_alpha_sector.col_mask) {
      std::ostringstream stream;
      stream << "dense boundary Hamiltonian alpha sector mismatch at node "
             << node << " index " << alpha_flat_sector_index
             << ": dense=(row_mask=" << dense_alpha_sector.row_mask
             << ", col_mask=" << dense_alpha_sector.col_mask
             << "), typed=(row_mask=" << typed_alpha_sector.row_mask
             << ", col_mask=" << typed_alpha_sector.col_mask << ')';
      throw std::runtime_error(stream.str());
    }
    for (int beta_flat_sector_index = 0;
         beta_flat_sector_index < dense_message.beta_layout.total_sector_count();
         ++beta_flat_sector_index) {
      const BoundarySector& dense_beta_sector =
          boundary_layout_sector(dense_message.beta_layout, beta_flat_sector_index);
      const BoundarySector& typed_beta_sector =
          boundary_layout_sector(typed_message.beta_layout, beta_flat_sector_index);
      if (dense_beta_sector.row_mask != typed_beta_sector.row_mask ||
          dense_beta_sector.col_mask != typed_beta_sector.col_mask) {
        std::ostringstream stream;
        stream << "dense boundary Hamiltonian beta sector mismatch at node "
               << node << " index " << beta_flat_sector_index
               << ": dense=(row_mask=" << dense_beta_sector.row_mask
               << ", col_mask=" << dense_beta_sector.col_mask
               << "), typed=(row_mask=" << typed_beta_sector.row_mask
               << ", col_mask=" << typed_beta_sector.col_mask << ')';
        throw std::runtime_error(stream.str());
      }

      const int flat_index =
          dense_message.flat_index(alpha_flat_sector_index, beta_flat_sector_index);
      JointDeletionPayload dense_joint = convert_hamiltonian_payload_to_joint(
          project_dense_hamiltonian_entry_payload_to_boundary_values(
              dense_message.payload_values[xmvb::to_size(flat_index)]));
      JointDeletionPayload typed_joint = convert_hamiltonian_payload_to_joint(
          typed_message.payload_values[xmvb::to_size(flat_index)]);
      cleanup_joint_payload(&dense_joint);
      cleanup_joint_payload(&typed_joint);

      if (dense_joint.alpha_row_count != typed_joint.alpha_row_count ||
          dense_joint.alpha_col_count != typed_joint.alpha_col_count ||
          dense_joint.beta_row_count != typed_joint.beta_row_count ||
          dense_joint.beta_col_count != typed_joint.beta_col_count) {
        std::ostringstream stream;
        stream << "dense boundary Hamiltonian payload dimension mismatch at node "
               << node
               << " for sector (alpha=" << alpha_flat_sector_index
               << ", beta=" << beta_flat_sector_index << ")"
               << ": dense=("
               << dense_joint.alpha_row_count << ','
               << dense_joint.alpha_col_count << ','
               << dense_joint.beta_row_count << ','
               << dense_joint.beta_col_count << "), typed=("
               << typed_joint.alpha_row_count << ','
               << typed_joint.alpha_col_count << ','
               << typed_joint.beta_row_count << ','
               << typed_joint.beta_col_count << ')';
        throw std::runtime_error(stream.str());
      }

      std::map<JointDeletionKey, double> all_sectors = dense_joint.sectors;
      for (const auto& [key, value] : typed_joint.sectors) {
        all_sectors.emplace(key, value);
      }
      for (const auto& [key, value] : all_sectors) {
        const auto dense_iterator = dense_joint.sectors.find(key);
        const auto typed_iterator = typed_joint.sectors.find(key);
        const double dense_value =
            (dense_iterator == dense_joint.sectors.end()) ? 0.0 : dense_iterator->second;
        const double typed_value =
            (typed_iterator == typed_joint.sectors.end()) ? 0.0 : typed_iterator->second;
        if (std::abs(dense_value - typed_value) > tolerance) {
          std::ostringstream stream;
          stream << "dense boundary Hamiltonian payload sector mismatch at node "
                 << node
                 << " for sector (alpha=" << alpha_flat_sector_index
                 << ", beta=" << beta_flat_sector_index << ")"
                 << ", key " << format_joint_key(key)
                 << ": dense=" << dense_value
                 << ", typed=" << typed_value
                 << "\ndense_payload: "
                 << format_dense_hamiltonian_entry_payload_summary(
                        dense_message.payload_values[xmvb::to_size(flat_index)])
                 << "\ntyped_payload: "
                 << format_hamiltonian_boundary_payload_summary(
                        typed_message.payload_values[xmvb::to_size(flat_index)]);
          throw std::runtime_error(stream.str());
        }
      }
    }
  }
}

int count_deleted_labels_in_frontier(
    const std::vector<int>& deleted_labels,
    const std::vector<int>& local_remainder_labels) {
  // In the recursive node->parent message, the final natural block order is
  //   [child-frontier blocks..., parent-interface, local remainder].
  // Deleted labels can therefore belong either to the child-frontier part or
  // to the local remainder. The parent-interface labels are never deletable.
  //
  // Counting labels not belonging to the local remainder gives the number of
  // deleted rows/columns in the child-frontier portion, which is exactly the
  // quantity needed to move the parent-interface block from its natural
  // position to the standard returned message convention with the interface
  // block in front.
  int count = 0;
  for (const int label : deleted_labels) {
    if (!std::binary_search(
            local_remainder_labels.begin(),
            local_remainder_labels.end(),
            label)) {
      ++count;
    }
  }
  return count;
}

int count_deleted_labels_in_set(
    const std::vector<int>& deleted_labels,
    const std::vector<int>& label_set) {
  int count = 0;
  for (const int label : deleted_labels) {
    if (std::binary_search(label_set.begin(), label_set.end(), label)) {
      ++count;
    }
  }
  return count;
}

JointDeletionPayload transform_interface_block_to_front(
    const JointDeletionPayload& natural_payload,
    int alpha_interface_rows,
    int alpha_interface_cols,
    const std::vector<int>& alpha_frontier_interface_rows,
    const std::vector<int>& alpha_frontier_interface_cols,
    const std::vector<int>& alpha_local_remainder_rows,
    const std::vector<int>& alpha_local_remainder_cols,
    int alpha_frontier_rows,
    int alpha_frontier_cols,
    int beta_interface_rows,
    int beta_interface_cols,
    const std::vector<int>& beta_frontier_interface_rows,
    const std::vector<int>& beta_frontier_interface_cols,
    const std::vector<int>& beta_local_remainder_rows,
    const std::vector<int>& beta_local_remainder_cols,
    int beta_frontier_rows,
    int beta_frontier_cols) {
  // Converts the recursive node-local natural order
  //   [child-frontier blocks..., parent-interface, local remainder]
  // into the external child-message convention used by the parent DP:
  //   [parent-interface, child-frontier blocks..., local remainder].
  //
  // The interface rows/columns are never deleted. However, they must be moved
  // only across the descendant-body part of the already-built child frontier,
  // not across the child-exposed node-interface labels that belong to the same
  // node-local block as the new parent interface itself.
  //
  // The current recursive frontier order is
  //   [child-interface, child-body, child-interface, child-body, ...].
  // The returned node->parent message instead needs
  //   [parent-interface, all node-local labels, descendant bodies].
  // Therefore the parent interface crosses only the surviving child-body
  // rows/columns for the present deleted sector.
  JointDeletionPayload transformed = natural_payload;
  for (auto& [key, value] : transformed.sectors) {
    const int alpha_deleted_rows_in_frontier =
        count_deleted_labels_in_frontier(
            key.alpha_key.row_labels,
            alpha_local_remainder_rows);
    const int alpha_deleted_cols_in_frontier =
        count_deleted_labels_in_frontier(
            key.alpha_key.col_labels,
            alpha_local_remainder_cols);
    const int beta_deleted_rows_in_frontier =
        count_deleted_labels_in_frontier(
            key.beta_key.row_labels,
            beta_local_remainder_rows);
    const int beta_deleted_cols_in_frontier =
        count_deleted_labels_in_frontier(
            key.beta_key.col_labels,
            beta_local_remainder_cols);

    const int alpha_deleted_frontier_interface_rows =
        count_deleted_labels_in_set(
            key.alpha_key.row_labels,
            alpha_frontier_interface_rows);
    const int alpha_deleted_frontier_interface_cols =
        count_deleted_labels_in_set(
            key.alpha_key.col_labels,
            alpha_frontier_interface_cols);
    const int beta_deleted_frontier_interface_rows =
        count_deleted_labels_in_set(
            key.beta_key.row_labels,
            beta_frontier_interface_rows);
    const int beta_deleted_frontier_interface_cols =
        count_deleted_labels_in_set(
            key.beta_key.col_labels,
            beta_frontier_interface_cols);

    const int alpha_frontier_body_rows =
        alpha_frontier_rows -
        static_cast<int>(alpha_frontier_interface_rows.size());
    const int alpha_frontier_body_cols =
        alpha_frontier_cols -
        static_cast<int>(alpha_frontier_interface_cols.size());
    const int beta_frontier_body_rows =
        beta_frontier_rows -
        static_cast<int>(beta_frontier_interface_rows.size());
    const int beta_frontier_body_cols =
        beta_frontier_cols -
        static_cast<int>(beta_frontier_interface_cols.size());

    const int alpha_deleted_frontier_body_rows =
        alpha_deleted_rows_in_frontier - alpha_deleted_frontier_interface_rows;
    const int alpha_deleted_frontier_body_cols =
        alpha_deleted_cols_in_frontier - alpha_deleted_frontier_interface_cols;
    const int beta_deleted_frontier_body_rows =
        beta_deleted_rows_in_frontier - beta_deleted_frontier_interface_rows;
    const int beta_deleted_frontier_body_cols =
        beta_deleted_cols_in_frontier - beta_deleted_frontier_interface_cols;

    const int alpha_surviving_frontier_body_rows =
        alpha_frontier_body_rows - alpha_deleted_frontier_body_rows;
    const int alpha_surviving_frontier_body_cols =
        alpha_frontier_body_cols - alpha_deleted_frontier_body_cols;
    const int beta_surviving_frontier_body_rows =
        beta_frontier_body_rows - beta_deleted_frontier_body_rows;
    const int beta_surviving_frontier_body_cols =
        beta_frontier_body_cols - beta_deleted_frontier_body_cols;

    const int parity =
        alpha_interface_rows *
            (alpha_surviving_frontier_body_rows + alpha_deleted_frontier_body_cols) +
        alpha_interface_cols *
            (alpha_surviving_frontier_body_cols + alpha_deleted_frontier_body_rows) +
        beta_interface_rows *
            (beta_surviving_frontier_body_rows + beta_deleted_frontier_body_cols) +
        beta_interface_cols *
            (beta_surviving_frontier_body_cols + beta_deleted_frontier_body_rows);
    value *= parity_sign(parity);
  }
  cleanup_joint_payload(&transformed);
  return transformed;
}

int interface_block_front_transform_parity(
    int interface_rows,
    int interface_cols,
    const std::vector<int>& frontier_interface_rows,
    const std::vector<int>& frontier_interface_cols,
    const std::vector<int>& local_remainder_rows,
    const std::vector<int>& local_remainder_cols,
    int frontier_rows,
    int frontier_cols,
    const SpinDeletionKey& key) {
  // Computes the one-spin parity needed to move the newly created
  // parent-interface block in front of the already merged child frontier for
  // one active deleted sector.
  const int deleted_rows_in_frontier =
      count_deleted_labels_in_frontier(key.row_labels, local_remainder_rows);
  const int deleted_cols_in_frontier =
      count_deleted_labels_in_frontier(key.col_labels, local_remainder_cols);
  const int deleted_frontier_interface_rows =
      count_deleted_labels_in_set(key.row_labels, frontier_interface_rows);
  const int deleted_frontier_interface_cols =
      count_deleted_labels_in_set(key.col_labels, frontier_interface_cols);
  const int frontier_body_rows =
      frontier_rows - static_cast<int>(frontier_interface_rows.size());
  const int frontier_body_cols =
      frontier_cols - static_cast<int>(frontier_interface_cols.size());
  const int deleted_frontier_body_rows =
      deleted_rows_in_frontier - deleted_frontier_interface_rows;
  const int deleted_frontier_body_cols =
      deleted_cols_in_frontier - deleted_frontier_interface_cols;
  const int surviving_frontier_body_rows =
      frontier_body_rows - deleted_frontier_body_rows;
  const int surviving_frontier_body_cols =
      frontier_body_cols - deleted_frontier_body_cols;
  return
      interface_rows *
          (surviving_frontier_body_rows + deleted_frontier_body_cols) +
      interface_cols *
          (surviving_frontier_body_cols + deleted_frontier_body_rows);
}

HamiltonianBoundaryPayload transform_hamiltonian_interface_block_to_front_values(
    const HamiltonianBoundaryPayload& natural_payload,
    int alpha_interface_rows,
    int alpha_interface_cols,
    const std::vector<int>& alpha_frontier_interface_rows,
    const std::vector<int>& alpha_frontier_interface_cols,
    const std::vector<int>& alpha_local_remainder_rows,
    const std::vector<int>& alpha_local_remainder_cols,
    int alpha_frontier_rows,
    int alpha_frontier_cols,
    int beta_interface_rows,
    int beta_interface_cols,
    const std::vector<int>& beta_frontier_interface_rows,
    const std::vector<int>& beta_frontier_interface_cols,
    const std::vector<int>& beta_local_remainder_rows,
    const std::vector<int>& beta_local_remainder_cols,
    int beta_frontier_rows,
    int beta_frontier_cols) {
  HamiltonianBoundaryPayload transformed = natural_payload;
  transformed.has_overlap_basis = false;
  transformed.alpha_basis_keys.clear();
  transformed.beta_basis_keys.clear();
  transformed.mixed_basis_keys.clear();

  for (auto& sector : transformed.alpha_sectors) {
    sector.value *= parity_sign(
        interface_block_front_transform_parity(
            alpha_interface_rows,
            alpha_interface_cols,
            alpha_frontier_interface_rows,
            alpha_frontier_interface_cols,
            alpha_local_remainder_rows,
            alpha_local_remainder_cols,
            alpha_frontier_rows,
            alpha_frontier_cols,
            sector.key));
  }
  for (auto& sector : transformed.beta_sectors) {
    sector.value *= parity_sign(
        interface_block_front_transform_parity(
            beta_interface_rows,
            beta_interface_cols,
            beta_frontier_interface_rows,
            beta_frontier_interface_cols,
            beta_local_remainder_rows,
            beta_local_remainder_cols,
            beta_frontier_rows,
            beta_frontier_cols,
            sector.key));
  }
  for (auto& sector : transformed.mixed_sectors) {
    const int alpha_parity =
        interface_block_front_transform_parity(
            alpha_interface_rows,
            alpha_interface_cols,
            alpha_frontier_interface_rows,
            alpha_frontier_interface_cols,
            alpha_local_remainder_rows,
            alpha_local_remainder_cols,
            alpha_frontier_rows,
            alpha_frontier_cols,
            sector.key.alpha_key);
    const int beta_parity =
        interface_block_front_transform_parity(
            beta_interface_rows,
            beta_interface_cols,
            beta_frontier_interface_rows,
            beta_frontier_interface_cols,
            beta_local_remainder_rows,
            beta_local_remainder_cols,
            beta_frontier_rows,
            beta_frontier_cols,
            sector.key.beta_key);
    sector.value *= parity_sign(alpha_parity ^ beta_parity);
  }

  cleanup_hamiltonian_payload(&transformed);
  return transformed;
}

std::vector<int> build_dense_spin_interface_parities(
    int row_count,
    int col_count,
    int support_size,
    int degree,
    int interface_rows,
    int interface_cols,
    const std::vector<int>& frontier_interface_rows,
    const std::vector<int>& frontier_interface_cols,
    const std::vector<int>& local_remainder_rows,
    const std::vector<int>& local_remainder_cols,
    int frontier_rows,
    int frontier_cols) {
  const int entry_count =
      (degree == 1)
          ? dense_spin_degree1_size(row_count, col_count, support_size)
          : dense_spin_degree2_size(row_count, col_count, support_size);
  std::vector<int> parities(
      xmvb::to_size(entry_count),
      0);
  for (int entry_index = 0; entry_index < entry_count; ++entry_index) {
    SpinDeletionKey key;
    if (degree == 1) {
      decode_dense_spin_degree1_key(
          row_count,
          col_count,
          entry_index,
          support_size,
          &key);
    } else {
      decode_dense_spin_degree2_key(
          row_count,
          col_count,
          entry_index,
          support_size,
          &key);
    }
    parities[xmvb::to_size(entry_index)] =
        interface_block_front_transform_parity(
            interface_rows,
            interface_cols,
            frontier_interface_rows,
            frontier_interface_cols,
            local_remainder_rows,
            local_remainder_cols,
            frontier_rows,
            frontier_cols,
            key);
  }
  return parities;
}

void apply_dense_parity_vector(
    std::vector<double>* values,
    const std::vector<int>& parities) {
  if (values == nullptr) {
    throw std::invalid_argument("dense parity values must not be null");
  }
  if (values->size() != parities.size()) {
    throw std::invalid_argument("dense parity vector size mismatch");
  }
  for (std::size_t index = 0; index < values->size(); ++index) {
    (*values)[index] *= parity_sign(parities[index]);
  }
}

HamiltonianEntryDensePayload
transform_hamiltonian_entry_dense_payload_interface_block_to_front_values(
    const HamiltonianEntryDensePayload& natural_payload,
    int alpha_interface_rows,
    int alpha_interface_cols,
    const std::vector<int>& alpha_frontier_interface_rows,
    const std::vector<int>& alpha_frontier_interface_cols,
    const std::vector<int>& alpha_local_remainder_rows,
    const std::vector<int>& alpha_local_remainder_cols,
    int alpha_frontier_rows,
    int alpha_frontier_cols,
    int beta_interface_rows,
    int beta_interface_cols,
    const std::vector<int>& beta_frontier_interface_rows,
    const std::vector<int>& beta_frontier_interface_cols,
    const std::vector<int>& beta_local_remainder_rows,
    const std::vector<int>& beta_local_remainder_cols,
    int beta_frontier_rows,
    int beta_frontier_cols) {
  HamiltonianEntryDensePayload transformed = natural_payload;
  const std::vector<int> alpha_degree1_parities =
      build_dense_spin_interface_parities(
          natural_payload.alpha_row_count,
          natural_payload.alpha_col_count,
          natural_payload.support_size,
          1,
          alpha_interface_rows,
          alpha_interface_cols,
          alpha_frontier_interface_rows,
          alpha_frontier_interface_cols,
          alpha_local_remainder_rows,
          alpha_local_remainder_cols,
          alpha_frontier_rows,
          alpha_frontier_cols);
  const std::vector<int> alpha_degree2_parities =
      build_dense_spin_interface_parities(
          natural_payload.alpha_row_count,
          natural_payload.alpha_col_count,
          natural_payload.support_size,
          2,
          alpha_interface_rows,
          alpha_interface_cols,
          alpha_frontier_interface_rows,
          alpha_frontier_interface_cols,
          alpha_local_remainder_rows,
          alpha_local_remainder_cols,
          alpha_frontier_rows,
          alpha_frontier_cols);
  const std::vector<int> beta_degree1_parities =
      build_dense_spin_interface_parities(
          natural_payload.beta_row_count,
          natural_payload.beta_col_count,
          natural_payload.support_size,
          1,
          beta_interface_rows,
          beta_interface_cols,
          beta_frontier_interface_rows,
          beta_frontier_interface_cols,
          beta_local_remainder_rows,
          beta_local_remainder_cols,
          beta_frontier_rows,
          beta_frontier_cols);
  const std::vector<int> beta_degree2_parities =
      build_dense_spin_interface_parities(
          natural_payload.beta_row_count,
          natural_payload.beta_col_count,
          natural_payload.support_size,
          2,
          beta_interface_rows,
          beta_interface_cols,
          beta_frontier_interface_rows,
          beta_frontier_interface_cols,
          beta_local_remainder_rows,
          beta_local_remainder_cols,
          beta_frontier_rows,
          beta_frontier_cols);
  apply_dense_parity_vector(
      &transformed.alpha_degree1,
      alpha_degree1_parities);
  apply_dense_parity_vector(
      &transformed.alpha_degree2,
      alpha_degree2_parities);
  apply_dense_parity_vector(
      &transformed.beta_degree1,
      beta_degree1_parities);
  apply_dense_parity_vector(
      &transformed.beta_degree2,
      beta_degree2_parities);

  const int alpha_degree1_size = dense_spin_degree1_size(
      natural_payload.alpha_row_count,
      natural_payload.alpha_col_count,
      natural_payload.support_size);
  const int beta_degree1_size = dense_spin_degree1_size(
      natural_payload.beta_row_count,
      natural_payload.beta_col_count,
      natural_payload.support_size);
  for (int beta_index = 0; beta_index < beta_degree1_size; ++beta_index) {
    for (int alpha_index = 0; alpha_index < alpha_degree1_size; ++alpha_index) {
      transformed.mixed_degree1[xmvb::to_size(
          dense_mixed_degree1_flat_index(transformed, alpha_index, beta_index))] *=
          parity_sign(
              alpha_degree1_parities[xmvb::to_size(alpha_index)] ^
              beta_degree1_parities[xmvb::to_size(beta_index)]);
    }
  }
  cleanup_hamiltonian_entry_dense_payload(&transformed);
  validate_dense_projected_hamiltonian_payload_against_reference(
      "transform_hamiltonian_entry_dense_payload_interface_block_to_front_values",
      transformed,
      transform_hamiltonian_interface_block_to_front_values(
          project_dense_hamiltonian_entry_payload_to_boundary_values(natural_payload),
          alpha_interface_rows,
          alpha_interface_cols,
          alpha_frontier_interface_rows,
          alpha_frontier_interface_cols,
          alpha_local_remainder_rows,
          alpha_local_remainder_cols,
          alpha_frontier_rows,
          alpha_frontier_cols,
          beta_interface_rows,
          beta_interface_cols,
          beta_frontier_interface_rows,
          beta_frontier_interface_cols,
          beta_local_remainder_rows,
          beta_local_remainder_cols,
          beta_frontier_rows,
          beta_frontier_cols),
      "");
  return transformed;
}

HamiltonianBoundaryPayload transform_hamiltonian_interface_block_to_front(
    const HamiltonianBoundaryPayload& natural_payload,
    int alpha_interface_rows,
    int alpha_interface_cols,
    const std::vector<int>& alpha_frontier_interface_rows,
    const std::vector<int>& alpha_frontier_interface_cols,
    const std::vector<int>& alpha_local_remainder_rows,
    const std::vector<int>& alpha_local_remainder_cols,
    int alpha_frontier_rows,
    int alpha_frontier_cols,
    int beta_interface_rows,
    int beta_interface_cols,
    const std::vector<int>& beta_frontier_interface_rows,
    const std::vector<int>& beta_frontier_interface_cols,
    const std::vector<int>& beta_local_remainder_rows,
    const std::vector<int>& beta_local_remainder_cols,
    int beta_frontier_rows,
    int beta_frontier_cols) {
  // Typed Hamiltonian analogue of `transform_interface_block_to_front(...)`.
  //
  // As with the typed merge, keep the forward map exactly aligned with the
  // joint deleted-minor reference by transforming in joint space first and
  // projecting back afterwards.
  return project_joint_payload_to_hamiltonian(
      transform_interface_block_to_front(
          convert_hamiltonian_payload_to_joint(natural_payload),
          alpha_interface_rows,
          alpha_interface_cols,
          alpha_frontier_interface_rows,
          alpha_frontier_interface_cols,
          alpha_local_remainder_rows,
          alpha_local_remainder_cols,
          alpha_frontier_rows,
          alpha_frontier_cols,
          beta_interface_rows,
          beta_interface_cols,
          beta_frontier_interface_rows,
          beta_frontier_interface_cols,
          beta_local_remainder_rows,
          beta_local_remainder_cols,
          beta_frontier_rows,
          beta_frontier_cols));
}

void reverse_transform_hamiltonian_interface_block_to_front(
    int alpha_interface_rows,
    int alpha_interface_cols,
    const std::vector<int>& alpha_frontier_interface_rows,
    const std::vector<int>& alpha_frontier_interface_cols,
    const std::vector<int>& alpha_local_remainder_rows,
    const std::vector<int>& alpha_local_remainder_cols,
    int alpha_frontier_rows,
    int alpha_frontier_cols,
    int beta_interface_rows,
    int beta_interface_cols,
    const std::vector<int>& beta_frontier_interface_rows,
    const std::vector<int>& beta_frontier_interface_cols,
    const std::vector<int>& beta_local_remainder_rows,
    const std::vector<int>& beta_local_remainder_cols,
    int beta_frontier_rows,
    int beta_frontier_cols,
    const HamiltonianBoundaryPayload& transformed_adjoint,
    HamiltonianBoundaryPayload* natural_adjoint) {
  if (natural_adjoint == nullptr) {
    throw std::invalid_argument("natural_adjoint must not be null");
  }
  JointDeletionPayload natural_joint_adjoint =
      convert_hamiltonian_payload_to_joint(*natural_adjoint);
  reverse_transform_interface_block_to_front(
      alpha_interface_rows,
      alpha_interface_cols,
      alpha_frontier_interface_rows,
      alpha_frontier_interface_cols,
      alpha_local_remainder_rows,
      alpha_local_remainder_cols,
      alpha_frontier_rows,
      alpha_frontier_cols,
      beta_interface_rows,
      beta_interface_cols,
      beta_frontier_interface_rows,
      beta_frontier_interface_cols,
      beta_local_remainder_rows,
      beta_local_remainder_cols,
      beta_frontier_rows,
      beta_frontier_cols,
      convert_hamiltonian_payload_to_joint(transformed_adjoint),
      &natural_joint_adjoint);
  *natural_adjoint = project_joint_payload_to_hamiltonian(natural_joint_adjoint);
}

const std::vector<ComponentTreeLeafMessage>& build_recursive_subtree_messages_limited_cached(
    const ComponentTree& tree,
    const std::vector<SubtreeExpansion>& subtree_expansions,
    int node,
    const OrientationTerm& left_parent_term,
    const OrientationTerm& right_parent_term,
    int max_spin_degree,
    bool preserve_structural_zero_messages,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subtree_term_pair_count,
    std::uint64_t* subtree_message_state_count,
    std::uint64_t* subdeterminant_evaluations,
    std::uint64_t* dp_transition_count,
    std::unordered_map<SpinPayloadCacheKey,
                       SpinDeletionPayload,
                       SpinPayloadCacheKeyHasher>* payload_cache,
    std::unordered_map<SubtreeMessageCacheKey,
                       std::vector<ComponentTreeLeafMessage>,
                       SubtreeMessageCacheKeyHasher>* message_cache) {
  // Exact recursive node->parent message builder.
  //
  // The returned vector contains all mask-indexed messages for subtree `node`
  // conditioned on the fixed parent determinant-term pair. Each message is
  // already expressed in the external convention expected by the parent merge:
  // the selected parent-interface orbitals are the undeletable leading block of
  // the message, followed by the recursively collapsed subtree body.
  if (subtree_term_pair_count == nullptr ||
      subtree_message_state_count == nullptr ||
      subdeterminant_evaluations == nullptr ||
      dp_transition_count == nullptr ||
      payload_cache == nullptr ||
      message_cache == nullptr) {
    throw std::invalid_argument("recursive subtree-message inputs must not be null");
  }
  if (max_spin_degree < 0 || max_spin_degree > 2) {
    throw std::invalid_argument("max_spin_degree must be in [0, 2]");
  }

  const SubtreeMessageCacheKey cache_key{
      node,
      left_parent_term.alpha_occ,
      left_parent_term.beta_occ,
      right_parent_term.alpha_occ,
      right_parent_term.beta_occ,
      preserve_structural_zero_messages,
  };
  auto cache_iterator = message_cache->find(cache_key);
  if (cache_iterator != message_cache->end()) {
    return cache_iterator->second;
  }

  const ComponentData& node_component =
      tree.components[xmvb::to_size(node)];
  const std::vector<int>& children = tree.children[xmvb::to_size(node)];
  if (children.empty()) {
    std::vector<ComponentTreeLeafMessage> messages =
        build_direct_subtree_messages_exact(
            subtree_expansions[xmvb::to_size(node)],
            left_parent_term,
            right_parent_term,
            preserve_structural_zero_messages,
            overlap_storage,
            n_orbitals,
            overlap_resolver,
            subdeterminant_evaluations,
            payload_cache);
    *subtree_message_state_count += static_cast<std::uint64_t>(messages.size());
    return message_cache->emplace(cache_key, std::move(messages)).first->second;
  }
  const auto left_parent_alpha_occ_by_mask = build_mask_occ_table(left_parent_term.alpha_occ);
  const auto right_parent_alpha_occ_by_mask = build_mask_occ_table(right_parent_term.alpha_occ);
  const auto left_parent_beta_occ_by_mask = build_mask_occ_table(left_parent_term.beta_occ);
  const auto right_parent_beta_occ_by_mask = build_mask_occ_table(right_parent_term.beta_occ);

  const ChildSpinSizeVectors child_spin_sizes =
      build_child_spin_size_vectors(children, subtree_expansions);

  std::map<
      FrontierHamiltonianMaskKey,
      JointDeletionPayload>
      aggregated_messages;

  const std::vector<WeightedOrientationTermPair> node_term_pairs =
      build_weighted_component_orientation_pairs(node_component);
  for (const auto& node_term_pair : node_term_pairs) {
      const OrientationTerm& left_node_term = node_term_pair.left_term;
      const OrientationTerm& right_node_term = node_term_pair.right_term;
      const double local_coefficient = node_term_pair.coefficient;
      ++(*subtree_term_pair_count);

      std::vector<const std::vector<ComponentTreeLeafMessage>*> child_messages(
          children.size(),
          nullptr);
      for (std::size_t child_index = 0; child_index < children.size(); ++child_index) {
        const int child = children[child_index];
        child_messages[child_index] =
            &build_recursive_subtree_messages_limited_cached(
                tree,
                subtree_expansions,
                child,
                left_node_term,
                right_node_term,
                max_spin_degree,
                preserve_structural_zero_messages,
                overlap_storage,
                n_orbitals,
                overlap_resolver,
                subtree_term_pair_count,
                subtree_message_state_count,
                subdeterminant_evaluations,
                dp_transition_count,
                payload_cache,
                message_cache);
      }

      const FrontierJointMessage frontier_message =
          build_frontier_joint_message(
              child_messages,
              child_spin_sizes,
              static_cast<int>(right_node_term.alpha_occ.size()),
              static_cast<int>(left_node_term.alpha_occ.size()),
              static_cast<int>(right_node_term.beta_occ.size()),
              static_cast<int>(left_node_term.beta_occ.size()),
              max_spin_degree,
              preserve_structural_zero_messages,
              dp_transition_count);

      const std::uint32_t alpha_row_limit =
          open_state_mask_limit(static_cast<int>(right_parent_term.alpha_occ.size()));
      const std::uint32_t alpha_col_limit =
          open_state_mask_limit(static_cast<int>(left_parent_term.alpha_occ.size()));
      const std::uint32_t beta_row_limit =
          open_state_mask_limit(static_cast<int>(right_parent_term.beta_occ.size()));
      const std::uint32_t beta_col_limit =
          open_state_mask_limit(static_cast<int>(left_parent_term.beta_occ.size()));
      const std::uint32_t local_alpha_row_full_mask =
          right_node_term.alpha_occ.empty()
              ? 0U
              : (open_state_mask_limit(static_cast<int>(right_node_term.alpha_occ.size())) - 1U);
      const std::uint32_t local_alpha_col_full_mask =
          left_node_term.alpha_occ.empty()
              ? 0U
              : (open_state_mask_limit(static_cast<int>(left_node_term.alpha_occ.size())) - 1U);
      const std::uint32_t local_beta_row_full_mask =
          right_node_term.beta_occ.empty()
              ? 0U
              : (open_state_mask_limit(static_cast<int>(right_node_term.beta_occ.size())) - 1U);
      const std::uint32_t local_beta_col_full_mask =
          left_node_term.beta_occ.empty()
              ? 0U
              : (open_state_mask_limit(static_cast<int>(left_node_term.beta_occ.size())) - 1U);

      for (std::uint32_t alpha_row_mask = 0; alpha_row_mask < alpha_row_limit; ++alpha_row_mask) {
        for (std::uint32_t alpha_col_mask = 0;
             alpha_col_mask < alpha_col_limit;
             ++alpha_col_mask) {
          for (std::uint32_t beta_row_mask = 0; beta_row_mask < beta_row_limit; ++beta_row_mask) {
            for (std::uint32_t beta_col_mask = 0;
                 beta_col_mask < beta_col_limit;
                 ++beta_col_mask) {
              const auto& selected_parent_alpha_rows =
                  right_parent_alpha_occ_by_mask[xmvb::to_size(alpha_row_mask)];
              const auto& selected_parent_alpha_cols =
                  left_parent_alpha_occ_by_mask[xmvb::to_size(alpha_col_mask)];
              const auto& selected_parent_beta_rows =
                  right_parent_beta_occ_by_mask[xmvb::to_size(beta_row_mask)];
              const auto& selected_parent_beta_cols =
                  left_parent_beta_occ_by_mask[xmvb::to_size(beta_col_mask)];
              for (const FrontierJointEntry& frontier_entry : frontier_message.entries) {
                const std::uint32_t used_alpha_row_mask =
                    frontier_entry.key.alpha_row_mask;
                const std::uint32_t used_alpha_col_mask =
                    frontier_entry.key.alpha_col_mask;
                const std::uint32_t used_beta_row_mask =
                    frontier_entry.key.beta_row_mask;
                const std::uint32_t used_beta_col_mask =
                    frontier_entry.key.beta_col_mask;
                const auto alpha_local_remainder_rows = select_occ_by_mask(
                    right_node_term.alpha_occ,
                    local_alpha_row_full_mask ^ used_alpha_row_mask);
                const auto alpha_local_remainder_cols = select_occ_by_mask(
                    left_node_term.alpha_occ,
                    local_alpha_col_full_mask ^ used_alpha_col_mask);
                const auto alpha_frontier_interface_rows = select_occ_by_mask(
                    right_node_term.alpha_occ,
                    used_alpha_row_mask);
                const auto alpha_frontier_interface_cols = select_occ_by_mask(
                    left_node_term.alpha_occ,
                    used_alpha_col_mask);
                const auto beta_local_remainder_rows = select_occ_by_mask(
                    right_node_term.beta_occ,
                    local_beta_row_full_mask ^ used_beta_row_mask);
                const auto beta_local_remainder_cols = select_occ_by_mask(
                    left_node_term.beta_occ,
                    local_beta_col_full_mask ^ used_beta_col_mask);
                const auto beta_frontier_interface_rows = select_occ_by_mask(
                    right_node_term.beta_occ,
                    used_beta_row_mask);
                const auto beta_frontier_interface_cols = select_occ_by_mask(
                    left_node_term.beta_occ,
                    used_beta_col_mask);

                const SpinDeletionPayload alpha_interface_payload =
                    build_exact_frontier_spin_payload_limited(
                        selected_parent_alpha_cols,
                        alpha_local_remainder_cols,
                        selected_parent_alpha_rows,
                        alpha_local_remainder_rows,
                        max_spin_degree,
                        overlap_storage,
                        n_orbitals,
                        overlap_resolver,
                        subdeterminant_evaluations,
                        payload_cache);
                const SpinDeletionPayload beta_interface_payload =
                    build_exact_frontier_spin_payload_limited(
                        selected_parent_beta_cols,
                        beta_local_remainder_cols,
                        selected_parent_beta_rows,
                        beta_local_remainder_rows,
                        max_spin_degree,
                        overlap_storage,
                        n_orbitals,
                        overlap_resolver,
                        subdeterminant_evaluations,
                        payload_cache);
                const JointDeletionPayload interface_payload =
                    build_joint_payload(
                        alpha_interface_payload,
                        beta_interface_payload);
                const JointDeletionPayload natural_payload =
                    merge_joint_deletion_payloads_limited(
                        frontier_entry.payload,
                        interface_payload,
                        max_spin_degree);
                const JointDeletionPayload front_convention_payload =
                    transform_interface_block_to_front(
                        natural_payload,
                        static_cast<int>(selected_parent_alpha_rows.size()),
                        static_cast<int>(selected_parent_alpha_cols.size()),
                        alpha_frontier_interface_rows,
                        alpha_frontier_interface_cols,
                        alpha_local_remainder_rows,
                        alpha_local_remainder_cols,
                        frontier_entry.payload.alpha_row_count,
                        frontier_entry.payload.alpha_col_count,
                        static_cast<int>(selected_parent_beta_rows.size()),
                        static_cast<int>(selected_parent_beta_cols.size()),
                        beta_frontier_interface_rows,
                        beta_frontier_interface_cols,
                        beta_local_remainder_rows,
                        beta_local_remainder_cols,
                        frontier_entry.payload.beta_row_count,
                        frontier_entry.payload.beta_col_count);
                if ((!preserve_structural_zero_messages &&
                     !is_joint_payload_nonzero(front_convention_payload)) ||
                    (preserve_structural_zero_messages &&
                     front_convention_payload.basis_keys.empty())) {
                  continue;
                }

                add_scaled_joint_payload_preserve_basis(
                    front_convention_payload,
                    local_coefficient,
                    preserve_structural_zero_messages,
                    &aggregated_messages[FrontierHamiltonianMaskKey{
                        .alpha_row_mask = alpha_row_mask,
                        .alpha_col_mask = alpha_col_mask,
                        .beta_row_mask = beta_row_mask,
                        .beta_col_mask = beta_col_mask,
                    }]);
              }
            }
          }
        }
      }
  }
  std::vector<ComponentTreeLeafMessage> messages =
      finalize_component_tree_leaf_messages(
          &aggregated_messages,
          preserve_structural_zero_messages);
  if (max_spin_degree == 2 && recursive_message_debug_validation_enabled()) {
    validate_recursive_subtree_messages_against_direct_reference(
        subtree_expansions[xmvb::to_size(node)],
        node,
        left_parent_term,
        right_parent_term,
        preserve_structural_zero_messages,
        overlap_storage,
        n_orbitals,
        overlap_resolver,
        messages);
  }
  *subtree_message_state_count += static_cast<std::uint64_t>(messages.size());
  return message_cache->emplace(cache_key, std::move(messages)).first->second;
}

void backprop_recursive_subtree_hamiltonian_message(
    const OverlapGradientReverseContext& context,
    int node,
    const OrientationTerm& left_parent_term,
    const OrientationTerm& right_parent_term,
    std::uint32_t target_alpha_row_mask,
    std::uint32_t target_alpha_col_mask,
    std::uint32_t target_beta_row_mask,
    std::uint32_t target_beta_col_mask,
    const HamiltonianBoundaryPayload& target_adjoint) {
  if (!is_hamiltonian_payload_nonzero(target_adjoint)) {
    return;
  }
  const ComponentData& node_component =
      context.tree.components[xmvb::to_size(node)];
  const std::vector<int>& children =
      context.tree.children[xmvb::to_size(node)];
  if (children.empty()) {
    const SubtreeExpansion& subtree_expansion =
        context.subtree_expansions[xmvb::to_size(node)];
    const std::vector<GlobalOrientationTerm> left_global_terms =
        build_global_orientation_terms(subtree_expansion.preorder_components, true);
    const std::vector<GlobalOrientationTerm> right_global_terms =
        build_global_orientation_terms(subtree_expansion.preorder_components, false);
    const auto left_parent_alpha_occ_by_mask =
        build_mask_occ_table(left_parent_term.alpha_occ);
    const auto right_parent_alpha_occ_by_mask =
        build_mask_occ_table(right_parent_term.alpha_occ);
    const auto left_parent_beta_occ_by_mask =
        build_mask_occ_table(left_parent_term.beta_occ);
    const auto right_parent_beta_occ_by_mask =
        build_mask_occ_table(right_parent_term.beta_occ);

    const auto& selected_parent_alpha_rows =
        right_parent_alpha_occ_by_mask[xmvb::to_size(target_alpha_row_mask)];
    const auto& selected_parent_alpha_cols =
        left_parent_alpha_occ_by_mask[xmvb::to_size(target_alpha_col_mask)];
    const auto& selected_parent_beta_rows =
        right_parent_beta_occ_by_mask[xmvb::to_size(target_beta_row_mask)];
    const auto& selected_parent_beta_cols =
        left_parent_beta_occ_by_mask[xmvb::to_size(target_beta_col_mask)];

    for (const auto& left_global_term : left_global_terms) {
      for (const auto& right_global_term : right_global_terms) {
        const double coefficient =
            left_global_term.coefficient * right_global_term.coefficient;
        if (std::abs(coefficient) <= 1.0e-15) {
          continue;
        }

        reverse_interface_hamiltonian_payload_overlap_gradient(
            context,
            selected_parent_alpha_cols,
            left_global_term.alpha_occ,
            selected_parent_alpha_rows,
            right_global_term.alpha_occ,
            selected_parent_beta_cols,
            left_global_term.beta_occ,
            selected_parent_beta_rows,
            right_global_term.beta_occ,
            scale_hamiltonian_payload(target_adjoint, coefficient));
      }
    }
    return;
  }

  const auto left_parent_alpha_occ_by_mask =
      build_mask_occ_table(left_parent_term.alpha_occ);
  const auto right_parent_alpha_occ_by_mask =
      build_mask_occ_table(right_parent_term.alpha_occ);
  const auto left_parent_beta_occ_by_mask =
      build_mask_occ_table(left_parent_term.beta_occ);
  const auto right_parent_beta_occ_by_mask =
      build_mask_occ_table(right_parent_term.beta_occ);
  const ChildSpinSizeVectors child_spin_sizes =
      build_child_spin_size_vectors(children, context.subtree_expansions);

  const auto& selected_parent_alpha_rows =
      right_parent_alpha_occ_by_mask[xmvb::to_size(target_alpha_row_mask)];
  const auto& selected_parent_alpha_cols =
      left_parent_alpha_occ_by_mask[xmvb::to_size(target_alpha_col_mask)];
  const auto& selected_parent_beta_rows =
      right_parent_beta_occ_by_mask[xmvb::to_size(target_beta_row_mask)];
  const auto& selected_parent_beta_cols =
      left_parent_beta_occ_by_mask[xmvb::to_size(target_beta_col_mask)];

  const std::vector<WeightedOrientationTermPair> node_term_pairs =
      build_weighted_component_orientation_pairs(node_component);
  for (const auto& node_term_pair : node_term_pairs) {
    const OrientationTerm& left_node_term = node_term_pair.left_term;
    const OrientationTerm& right_node_term = node_term_pair.right_term;
    const double local_coefficient = node_term_pair.coefficient;
    if (std::abs(local_coefficient) <= 1.0e-15) {
      continue;
    }

    std::vector<const BoundaryHamiltonianMessage*> child_messages(
        children.size(),
        nullptr);
    for (std::size_t child_index = 0; child_index < children.size(); ++child_index) {
      const int child = children[child_index];
      child_messages[child_index] =
          &build_recursive_subtree_hamiltonian_messages_cached(
              context.tree,
              context.subtree_expansions,
              child,
              left_node_term,
              right_node_term,
              true,
              context.support_overlap_storage,
              context.support_size,
              context.overlap_resolver,
              context.subtree_term_pair_count,
              context.subtree_message_state_count,
              context.subdeterminant_evaluations,
              context.dp_transition_count,
              context.spin_payload_cache,
              context.hamiltonian_message_cache);
    }

    const std::uint32_t local_alpha_row_full_mask =
        right_node_term.alpha_occ.empty()
            ? 0U
            : (open_state_mask_limit(static_cast<int>(right_node_term.alpha_occ.size())) - 1U);
    const std::uint32_t local_alpha_col_full_mask =
        left_node_term.alpha_occ.empty()
            ? 0U
            : (open_state_mask_limit(static_cast<int>(left_node_term.alpha_occ.size())) - 1U);
    const std::uint32_t local_beta_row_full_mask =
        right_node_term.beta_occ.empty()
            ? 0U
            : (open_state_mask_limit(static_cast<int>(right_node_term.beta_occ.size())) - 1U);
    const std::uint32_t local_beta_col_full_mask =
        left_node_term.beta_occ.empty()
            ? 0U
            : (open_state_mask_limit(static_cast<int>(left_node_term.beta_occ.size())) - 1U);
    std::vector<FrontierHamiltonianMessage> prefix_messages;
    prefix_messages.reserve(children.size() + 1U);
    prefix_messages.push_back(make_identity_frontier_hamiltonian_message(
        static_cast<int>(right_node_term.alpha_occ.size()),
        static_cast<int>(left_node_term.alpha_occ.size()),
        static_cast<int>(right_node_term.beta_occ.size()),
        static_cast<int>(left_node_term.beta_occ.size())));
    std::uint64_t reverse_frontier_transition_count = 0;
    for (std::size_t child_index = 0; child_index < children.size(); ++child_index) {
      prefix_messages.push_back(
          merge_frontier_hamiltonian_message_with_child(
              prefix_messages.back(),
              *child_messages[child_index],
              child_spin_sizes.right_alpha[child_index],
              child_spin_sizes.left_alpha[child_index],
              child_spin_sizes.right_beta[child_index],
              child_spin_sizes.left_beta[child_index],
              true,
              &reverse_frontier_transition_count));
    }
    const FrontierHamiltonianMessage& frontier_message = prefix_messages.back();

    std::map<FrontierHamiltonianMaskKey, HamiltonianBoundaryPayload> frontier_adjoint_map;
    for (const FrontierHamiltonianEntry& frontier_entry : frontier_message.entries) {
      const std::uint32_t used_alpha_row_mask =
          frontier_entry.key.alpha_row_mask;
      const std::uint32_t used_alpha_col_mask =
          frontier_entry.key.alpha_col_mask;
      const std::uint32_t used_beta_row_mask =
          frontier_entry.key.beta_row_mask;
      const std::uint32_t used_beta_col_mask =
          frontier_entry.key.beta_col_mask;
      const std::vector<int> alpha_local_remainder_rows = select_occ_by_mask(
          right_node_term.alpha_occ,
          local_alpha_row_full_mask ^ used_alpha_row_mask);
      const std::vector<int> alpha_local_remainder_cols = select_occ_by_mask(
          left_node_term.alpha_occ,
          local_alpha_col_full_mask ^ used_alpha_col_mask);
      const std::vector<int> alpha_frontier_interface_rows = select_occ_by_mask(
          right_node_term.alpha_occ,
          used_alpha_row_mask);
      const std::vector<int> alpha_frontier_interface_cols = select_occ_by_mask(
          left_node_term.alpha_occ,
          used_alpha_col_mask);
      const std::vector<int> beta_local_remainder_rows = select_occ_by_mask(
          right_node_term.beta_occ,
          local_beta_row_full_mask ^ used_beta_row_mask);
      const std::vector<int> beta_local_remainder_cols = select_occ_by_mask(
          left_node_term.beta_occ,
          local_beta_col_full_mask ^ used_beta_col_mask);
      const std::vector<int> beta_frontier_interface_rows = select_occ_by_mask(
          right_node_term.beta_occ,
          used_beta_row_mask);
      const std::vector<int> beta_frontier_interface_cols = select_occ_by_mask(
          left_node_term.beta_occ,
          used_beta_col_mask);

      const SpinDeletionPayload alpha_interface_payload =
          build_exact_frontier_spin_payload_limited(
              selected_parent_alpha_cols,
              alpha_local_remainder_cols,
              selected_parent_alpha_rows,
              alpha_local_remainder_rows,
              2,
              context.support_overlap_storage,
              context.support_size,
              context.overlap_resolver,
              context.subdeterminant_evaluations,
              context.spin_payload_cache);
      const SpinDeletionPayload beta_interface_payload =
          build_exact_frontier_spin_payload_limited(
              selected_parent_beta_cols,
              beta_local_remainder_cols,
              selected_parent_beta_rows,
              beta_local_remainder_rows,
              2,
              context.support_overlap_storage,
              context.support_size,
              context.overlap_resolver,
              context.subdeterminant_evaluations,
              context.spin_payload_cache);
      const HamiltonianBoundaryPayload interface_payload =
          build_hamiltonian_boundary_payload(
              alpha_interface_payload,
              beta_interface_payload);
      const HamiltonianBoundaryPayload natural_payload =
          merge_hamiltonian_boundary_payloads_limited(
              frontier_entry.payload,
              interface_payload,
              2);
      const HamiltonianBoundaryPayload front_convention_payload =
          transform_hamiltonian_interface_block_to_front(
              natural_payload,
              static_cast<int>(selected_parent_alpha_rows.size()),
              static_cast<int>(selected_parent_alpha_cols.size()),
              alpha_frontier_interface_rows,
              alpha_frontier_interface_cols,
              alpha_local_remainder_rows,
              alpha_local_remainder_cols,
              frontier_entry.payload.alpha_row_count,
              frontier_entry.payload.alpha_col_count,
              static_cast<int>(selected_parent_beta_rows.size()),
              static_cast<int>(selected_parent_beta_cols.size()),
              beta_frontier_interface_rows,
              beta_frontier_interface_cols,
              beta_local_remainder_rows,
              beta_local_remainder_cols,
              frontier_entry.payload.beta_row_count,
              frontier_entry.payload.beta_col_count);
      if (!has_hamiltonian_payload_basis(front_convention_payload)) {
        continue;
      }

      const HamiltonianBoundaryPayload front_adjoint =
          scale_hamiltonian_payload(
              target_adjoint,
              local_coefficient);
      HamiltonianBoundaryPayload natural_adjoint =
          make_zero_hamiltonian_payload_like(natural_payload);
      reverse_transform_hamiltonian_interface_block_to_front(
          static_cast<int>(selected_parent_alpha_rows.size()),
          static_cast<int>(selected_parent_alpha_cols.size()),
          alpha_frontier_interface_rows,
          alpha_frontier_interface_cols,
          alpha_local_remainder_rows,
          alpha_local_remainder_cols,
          frontier_entry.payload.alpha_row_count,
          frontier_entry.payload.alpha_col_count,
          static_cast<int>(selected_parent_beta_rows.size()),
          static_cast<int>(selected_parent_beta_cols.size()),
          beta_frontier_interface_rows,
          beta_frontier_interface_cols,
          beta_local_remainder_rows,
          beta_local_remainder_cols,
          frontier_entry.payload.beta_row_count,
          frontier_entry.payload.beta_col_count,
          front_adjoint,
          &natural_adjoint);

      HamiltonianBoundaryPayload interface_adjoint =
          make_zero_hamiltonian_payload_like(interface_payload);
      HamiltonianBoundaryPayload frontier_branch_adjoint =
          make_zero_hamiltonian_payload_like(frontier_entry.payload);
      reverse_merge_hamiltonian_boundary_payloads_limited(
          frontier_entry.payload,
          interface_payload,
          2,
          natural_adjoint,
          &frontier_branch_adjoint,
          &interface_adjoint);
      add_scaled_hamiltonian_payload_preserve_basis(
          frontier_branch_adjoint,
          1.0,
          true,
          &frontier_adjoint_map[frontier_entry.key]);
      reverse_interface_hamiltonian_payload_overlap_gradient(
          context,
          selected_parent_alpha_cols,
          alpha_local_remainder_cols,
          selected_parent_alpha_rows,
          alpha_local_remainder_rows,
          selected_parent_beta_cols,
          beta_local_remainder_cols,
          selected_parent_beta_rows,
          beta_local_remainder_rows,
          interface_adjoint);
    }

    FrontierHamiltonianMessage frontier_adjoint =
        finalize_frontier_hamiltonian_message(
            frontier_message.alpha_row_count,
            frontier_message.alpha_col_count,
            frontier_message.beta_row_count,
            frontier_message.beta_col_count,
            &frontier_adjoint_map,
            true);
    std::vector<BoundaryHamiltonianMessage> child_adjoint_messages(
        children.size());
    FrontierHamiltonianMessage identity_adjoint;
    reverse_frontier_hamiltonian_message(
        child_messages,
        child_spin_sizes,
        prefix_messages,
        frontier_adjoint,
        &child_adjoint_messages,
        &identity_adjoint);
    static_cast<void>(identity_adjoint);

    for (std::size_t child_index = 0; child_index < children.size(); ++child_index) {
      const BoundaryHamiltonianMessage& child_adjoint =
          child_adjoint_messages[child_index];
      for (const auto& child_entry : child_adjoint.nonzero_entries) {
        backprop_recursive_subtree_hamiltonian_message(
            context,
            children[child_index],
            left_node_term,
            right_node_term,
            boundary_layout_sector(
                child_adjoint.alpha_layout,
                child_entry.alpha_flat_sector_index)
                .row_mask,
            boundary_layout_sector(
                child_adjoint.alpha_layout,
                child_entry.alpha_flat_sector_index)
                .col_mask,
            boundary_layout_sector(
                child_adjoint.beta_layout,
                child_entry.beta_flat_sector_index)
                .row_mask,
            boundary_layout_sector(
                child_adjoint.beta_layout,
                child_entry.beta_flat_sector_index)
                .col_mask,
            child_adjoint.payload_values[xmvb::to_size(
                child_entry.flat_index)]);
      }
    }
  }
}

SpinDeletionPayload make_zero_spin_payload(int row_count, int col_count) {
  SpinDeletionPayload payload;
  payload.row_count = row_count;
  payload.col_count = col_count;
  return payload;
}

SpinDeletionPayload make_zero_spin_payload_like(const SpinDeletionPayload& payload) {
  SpinDeletionPayload zero = make_zero_spin_payload(payload.row_count, payload.col_count);
  zero.basis_keys = payload.basis_keys;
  return zero;
}

JointDeletionPayload make_zero_joint_payload(
    int alpha_row_count,
    int alpha_col_count,
    int beta_row_count,
    int beta_col_count) {
  JointDeletionPayload payload;
  payload.alpha_row_count = alpha_row_count;
  payload.alpha_col_count = alpha_col_count;
  payload.beta_row_count = beta_row_count;
  payload.beta_col_count = beta_col_count;
  return payload;
}

JointDeletionPayload make_zero_joint_payload_like(const JointDeletionPayload& payload) {
  JointDeletionPayload zero = make_zero_joint_payload(
      payload.alpha_row_count,
      payload.alpha_col_count,
      payload.beta_row_count,
      payload.beta_col_count);
  zero.basis_keys = payload.basis_keys;
  return zero;
}

JointDeletionPayload scale_joint_payload(
    const JointDeletionPayload& payload,
    double scale) {
  JointDeletionPayload scaled = make_zero_joint_payload_like(payload);
  if (std::abs(scale) <= 1.0e-15) {
    return scaled;
  }
  for (const auto& [key, value] : payload.sectors) {
    if (std::abs(value) <= 1.0e-15) {
      continue;
    }
    scaled.sectors[key] = scale * value;
  }
  cleanup_joint_payload(&scaled);
  return scaled;
}

bool try_find_occ_position(
    const std::vector<int>& occ,
    int orbital_label,
    int* position) {
  if (position == nullptr) {
    throw std::invalid_argument("position must not be null");
  }
  const auto iterator = std::find(occ.begin(), occ.end(), orbital_label);
  if (iterator == occ.end()) {
    return false;
  }
  *position = static_cast<int>(iterator - occ.begin());
  return true;
}

void reverse_build_joint_payload(
    const SpinDeletionPayload& alpha_payload,
    const SpinDeletionPayload& beta_payload,
    const JointDeletionPayload& joint_adjoint,
    SpinDeletionPayload* alpha_adjoint,
    SpinDeletionPayload* beta_adjoint) {
  if (alpha_adjoint == nullptr || beta_adjoint == nullptr) {
    throw std::invalid_argument("joint payload reverse outputs must not be null");
  }

  if (alpha_adjoint->row_count == 0 && alpha_adjoint->col_count == 0) {
    alpha_adjoint->row_count = alpha_payload.row_count;
    alpha_adjoint->col_count = alpha_payload.col_count;
  }
  if (beta_adjoint->row_count == 0 && beta_adjoint->col_count == 0) {
    beta_adjoint->row_count = beta_payload.row_count;
    beta_adjoint->col_count = beta_payload.col_count;
  }

  for (const auto& [joint_key, joint_value_adjoint] : joint_adjoint.sectors) {
    if (std::abs(joint_value_adjoint) <= 1.0e-15) {
      continue;
    }
    const auto alpha_iterator = alpha_payload.sectors.find(joint_key.alpha_key);
    const auto beta_iterator = beta_payload.sectors.find(joint_key.beta_key);
    const double alpha_value =
        (alpha_iterator == alpha_payload.sectors.end()) ? 0.0 : alpha_iterator->second;
    const double beta_value =
        (beta_iterator == beta_payload.sectors.end()) ? 0.0 : beta_iterator->second;
    if (std::abs(beta_value) > 1.0e-15) {
      alpha_adjoint->sectors[joint_key.alpha_key] +=
          joint_value_adjoint * beta_value;
    }
    if (std::abs(alpha_value) > 1.0e-15) {
      beta_adjoint->sectors[joint_key.beta_key] +=
          joint_value_adjoint * alpha_value;
    }
  }
  cleanup_spin_payload(alpha_adjoint);
  cleanup_spin_payload(beta_adjoint);
}

void reverse_merge_joint_deletion_payloads_limited(
    const JointDeletionPayload& left,
    const JointDeletionPayload& right,
    int max_spin_degree,
    const JointDeletionPayload& merged_adjoint,
    JointDeletionPayload* left_adjoint,
    JointDeletionPayload* right_adjoint) {
  if (left_adjoint == nullptr || right_adjoint == nullptr) {
    throw std::invalid_argument("reverse merge outputs must not be null");
  }
  if (max_spin_degree < 0 || max_spin_degree > 2) {
    throw std::invalid_argument("max_spin_degree must be in [0, 2]");
  }

  if (left_adjoint->alpha_row_count == 0 && left_adjoint->alpha_col_count == 0 &&
      left_adjoint->beta_row_count == 0 && left_adjoint->beta_col_count == 0) {
    *left_adjoint = make_zero_joint_payload_like(left);
  }
  if (right_adjoint->alpha_row_count == 0 && right_adjoint->alpha_col_count == 0 &&
      right_adjoint->beta_row_count == 0 && right_adjoint->beta_col_count == 0) {
    *right_adjoint = make_zero_joint_payload_like(right);
  }

  for (const auto& left_key : left.basis_keys) {
    const auto left_iterator = left.sectors.find(left_key);
    const double left_value =
        (left_iterator == left.sectors.end()) ? 0.0 : left_iterator->second;
    for (const auto& right_key : right.basis_keys) {
      const auto right_iterator = right.sectors.find(right_key);
      const double right_value =
          (right_iterator == right.sectors.end()) ? 0.0 : right_iterator->second;
      if (std::abs(left_value) <= 1.0e-15 &&
          std::abs(right_value) <= 1.0e-15) {
        continue;
      }
      JointDeletionKey combined_key;
      if (!combine_spin_keys_with_limit(
              left_key.alpha_key,
              right_key.alpha_key,
              max_spin_degree,
              &combined_key.alpha_key) ||
          !combine_spin_keys_with_limit(
              left_key.beta_key,
              right_key.beta_key,
              max_spin_degree,
              &combined_key.beta_key)) {
        continue;
      }

      const int raw_parity =
          spin_sector_merge_parity(
              left.alpha_row_count,
              left.alpha_col_count,
              left_key.alpha_key,
              right.alpha_row_count,
              right.alpha_col_count,
              right_key.alpha_key) ^
          spin_sector_merge_parity(
              left.beta_row_count,
              left.beta_col_count,
              left_key.beta_key,
              right.beta_row_count,
              right.beta_col_count,
              right_key.beta_key);
      int alpha_canonical_parity = 0;
      int beta_canonical_parity = 0;
      const JointDeletionKey canonical_key{
          canonicalize_spin_key(
              combined_key.alpha_key,
              &alpha_canonical_parity),
          canonicalize_spin_key(
              combined_key.beta_key,
              &beta_canonical_parity),
      };
      const auto merged_iterator = merged_adjoint.sectors.find(canonical_key);
      if (merged_iterator == merged_adjoint.sectors.end() ||
          std::abs(merged_iterator->second) <= 1.0e-15) {
        continue;
      }

      const double signed_adjoint =
          parity_sign(raw_parity ^ alpha_canonical_parity ^ beta_canonical_parity) *
          merged_iterator->second;
      if (std::abs(right_value) > 1.0e-15) {
        left_adjoint->sectors[left_key] += signed_adjoint * right_value;
      }
      if (std::abs(left_value) > 1.0e-15) {
        right_adjoint->sectors[right_key] += signed_adjoint * left_value;
      }
    }
  }
  cleanup_joint_payload(left_adjoint);
  cleanup_joint_payload(right_adjoint);
}

void reverse_transform_interface_block_to_front(
    int alpha_interface_rows,
    int alpha_interface_cols,
    const std::vector<int>& alpha_frontier_interface_rows,
    const std::vector<int>& alpha_frontier_interface_cols,
    const std::vector<int>& alpha_local_remainder_rows,
    const std::vector<int>& alpha_local_remainder_cols,
    int alpha_frontier_rows,
    int alpha_frontier_cols,
    int beta_interface_rows,
    int beta_interface_cols,
    const std::vector<int>& beta_frontier_interface_rows,
    const std::vector<int>& beta_frontier_interface_cols,
    const std::vector<int>& beta_local_remainder_rows,
    const std::vector<int>& beta_local_remainder_cols,
    int beta_frontier_rows,
    int beta_frontier_cols,
    const JointDeletionPayload& transformed_adjoint,
    JointDeletionPayload* natural_adjoint) {
  if (natural_adjoint == nullptr) {
    throw std::invalid_argument("natural_adjoint must not be null");
  }
  if (natural_adjoint->alpha_row_count == 0 && natural_adjoint->alpha_col_count == 0 &&
      natural_adjoint->beta_row_count == 0 && natural_adjoint->beta_col_count == 0) {
    *natural_adjoint = make_zero_joint_payload_like(transformed_adjoint);
  }

  for (const auto& [key, value_adjoint] : transformed_adjoint.sectors) {
    if (std::abs(value_adjoint) <= 1.0e-15) {
      continue;
    }
    const int alpha_deleted_rows_in_frontier =
        count_deleted_labels_in_frontier(
            key.alpha_key.row_labels,
            alpha_local_remainder_rows);
    const int alpha_deleted_cols_in_frontier =
        count_deleted_labels_in_frontier(
            key.alpha_key.col_labels,
            alpha_local_remainder_cols);
    const int beta_deleted_rows_in_frontier =
        count_deleted_labels_in_frontier(
            key.beta_key.row_labels,
            beta_local_remainder_rows);
    const int beta_deleted_cols_in_frontier =
        count_deleted_labels_in_frontier(
            key.beta_key.col_labels,
            beta_local_remainder_cols);

    const int alpha_deleted_frontier_interface_rows =
        count_deleted_labels_in_set(
            key.alpha_key.row_labels,
            alpha_frontier_interface_rows);
    const int alpha_deleted_frontier_interface_cols =
        count_deleted_labels_in_set(
            key.alpha_key.col_labels,
            alpha_frontier_interface_cols);
    const int beta_deleted_frontier_interface_rows =
        count_deleted_labels_in_set(
            key.beta_key.row_labels,
            beta_frontier_interface_rows);
    const int beta_deleted_frontier_interface_cols =
        count_deleted_labels_in_set(
            key.beta_key.col_labels,
            beta_frontier_interface_cols);

    const int alpha_frontier_body_rows =
        alpha_frontier_rows -
        static_cast<int>(alpha_frontier_interface_rows.size());
    const int alpha_frontier_body_cols =
        alpha_frontier_cols -
        static_cast<int>(alpha_frontier_interface_cols.size());
    const int beta_frontier_body_rows =
        beta_frontier_rows -
        static_cast<int>(beta_frontier_interface_rows.size());
    const int beta_frontier_body_cols =
        beta_frontier_cols -
        static_cast<int>(beta_frontier_interface_cols.size());

    const int alpha_deleted_frontier_body_rows =
        alpha_deleted_rows_in_frontier - alpha_deleted_frontier_interface_rows;
    const int alpha_deleted_frontier_body_cols =
        alpha_deleted_cols_in_frontier - alpha_deleted_frontier_interface_cols;
    const int beta_deleted_frontier_body_rows =
        beta_deleted_rows_in_frontier - beta_deleted_frontier_interface_rows;
    const int beta_deleted_frontier_body_cols =
        beta_deleted_cols_in_frontier - beta_deleted_frontier_interface_cols;

    const int alpha_surviving_frontier_body_rows =
        alpha_frontier_body_rows - alpha_deleted_frontier_body_rows;
    const int alpha_surviving_frontier_body_cols =
        alpha_frontier_body_cols - alpha_deleted_frontier_body_cols;
    const int beta_surviving_frontier_body_rows =
        beta_frontier_body_rows - beta_deleted_frontier_body_rows;
    const int beta_surviving_frontier_body_cols =
        beta_frontier_body_cols - beta_deleted_frontier_body_cols;

    const int parity =
        alpha_interface_rows *
            (alpha_surviving_frontier_body_rows + alpha_deleted_frontier_body_cols) +
        alpha_interface_cols *
            (alpha_surviving_frontier_body_cols + alpha_deleted_frontier_body_rows) +
        beta_interface_rows *
            (beta_surviving_frontier_body_rows + beta_deleted_frontier_body_cols) +
        beta_interface_cols *
            (beta_surviving_frontier_body_cols + beta_deleted_frontier_body_rows);
    natural_adjoint->sectors[key] += parity_sign(parity) * value_adjoint;
  }
  cleanup_joint_payload(natural_adjoint);
}

void accumulate_spin_deleted_payload_overlap_gradient(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    int internal_row_begin,
    int internal_col_begin,
    bool zero_selected_root_block,
    PartialSide side,
    const SpinDeletionPayload& sector_adjoint,
    const std::vector<double>& support_overlap_storage,
    int support_size,
    const DeterminantOverlapResolver& overlap_resolver,
    std::vector<double>* active_orbital_overlap_gradient) {
  // Pulls a reverse adjoint on exact deleted-minor sectors back to the
  // support-space overlap matrix of one spin block.
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }
  if (sector_adjoint.sectors.empty()) {
    return;
  }

  Eigen::MatrixXd overlap_block =
      build_overlap_block(left_occ, right_occ, support_overlap_storage, support_size);
  if (zero_selected_root_block && internal_row_begin > 0 && internal_col_begin > 0) {
    overlap_block.topLeftCorner(internal_row_begin, internal_col_begin).setZero();
  }

  const int n_rows = static_cast<int>(right_occ.size());
  const int n_cols = static_cast<int>(left_occ.size());
  std::uint64_t local_subdeterminant_evaluations = 0;
  for (const auto& [key, value_adjoint] : sector_adjoint.sectors) {
    if (std::abs(value_adjoint) <= 1.0e-15) {
      continue;
    }

    std::vector<int> deleted_rows;
    std::vector<int> deleted_cols;
    deleted_rows.reserve(key.row_labels.size());
    deleted_cols.reserve(key.col_labels.size());
    bool key_is_compatible = true;
    for (const int row_label : key.row_labels) {
      int row_position = -1;
      if (!try_find_occ_position(right_occ, row_label, &row_position)) {
        key_is_compatible = false;
        break;
      }
      deleted_rows.push_back(row_position);
    }
    if (!key_is_compatible) {
      continue;
    }
    for (const int col_label : key.col_labels) {
      int col_position = -1;
      if (!try_find_occ_position(left_occ, col_label, &col_position)) {
        key_is_compatible = false;
        break;
      }
      deleted_cols.push_back(col_position);
    }
    if (!key_is_compatible) {
      continue;
    }
    std::sort(deleted_rows.begin(), deleted_rows.end());
    std::sort(deleted_cols.begin(), deleted_cols.end());

    const Eigen::MatrixXd minor =
        build_deleted_minor_matrix(overlap_block, deleted_rows, deleted_cols);
    const DeterminantOverlapResult minor_result =
        resolve_overlap_result_counted(
            minor,
            overlap_resolver,
            &local_subdeterminant_evaluations);
    const Eigen::MatrixXd minor_first_cofactor = calc_cofactor_1st(minor_result);
    const double signed_adjoint =
        value_adjoint *
        deleted_minor_sign(
            n_rows,
            n_cols,
            deleted_rows,
            deleted_cols,
            side);

    int minor_row = 0;
    for (int row = 0; row < n_rows; ++row) {
      if (std::binary_search(deleted_rows.begin(), deleted_rows.end(), row)) {
        continue;
      }
      int minor_col = 0;
      for (int col = 0; col < n_cols; ++col) {
        if (std::binary_search(deleted_cols.begin(), deleted_cols.end(), col)) {
          continue;
        }
        if (!(zero_selected_root_block &&
              row < internal_row_begin &&
              col < internal_col_begin)) {
          (*active_orbital_overlap_gradient)[xmvb::to_size(
              left_occ[xmvb::to_size(col)]) *
                  xmvb::to_size(support_size) +
              xmvb::to_size(right_occ[xmvb::to_size(row)])] +=
              signed_adjoint * minor_first_cofactor(minor_row, minor_col);
        }
        ++minor_col;
      }
      ++minor_row;
    }
  }
}

void reverse_root_joint_payload_overlap_gradient(
    const OverlapGradientReverseContext& context,
    const OrientationTerm& left_root_term,
    const OrientationTerm& right_root_term,
    std::uint32_t used_alpha_row_mask,
    std::uint32_t used_alpha_col_mask,
    std::uint32_t used_beta_row_mask,
    std::uint32_t used_beta_col_mask,
    const JointDeletionPayload& root_adjoint) {
  if (!is_joint_payload_nonzero(root_adjoint)) {
    return;
  }

  const SpinDeletionPayload alpha_payload =
      build_root_spin_payload_limited(
          left_root_term.alpha_occ,
          right_root_term.alpha_occ,
          used_alpha_row_mask,
          used_alpha_col_mask,
          2,
          context.support_overlap_storage,
          context.support_size,
          context.overlap_resolver,
          context.subdeterminant_evaluations,
          context.spin_payload_cache);
  const SpinDeletionPayload beta_payload =
      build_root_spin_payload_limited(
          left_root_term.beta_occ,
          right_root_term.beta_occ,
          used_beta_row_mask,
          used_beta_col_mask,
          2,
          context.support_overlap_storage,
          context.support_size,
          context.overlap_resolver,
          context.subdeterminant_evaluations,
          context.spin_payload_cache);

  SpinDeletionPayload alpha_adjoint = make_zero_spin_payload_like(alpha_payload);
  SpinDeletionPayload beta_adjoint = make_zero_spin_payload_like(beta_payload);
  reverse_build_joint_payload(
      alpha_payload,
      beta_payload,
      root_adjoint,
      &alpha_adjoint,
      &beta_adjoint);

  const std::uint32_t alpha_row_full_mask =
      right_root_term.alpha_occ.empty()
          ? 0U
          : (open_state_mask_limit(static_cast<int>(right_root_term.alpha_occ.size())) - 1U);
  const std::uint32_t alpha_col_full_mask =
      left_root_term.alpha_occ.empty()
          ? 0U
          : (open_state_mask_limit(static_cast<int>(left_root_term.alpha_occ.size())) - 1U);
  const std::uint32_t beta_row_full_mask =
      right_root_term.beta_occ.empty()
          ? 0U
          : (open_state_mask_limit(static_cast<int>(right_root_term.beta_occ.size())) - 1U);
  const std::uint32_t beta_col_full_mask =
      left_root_term.beta_occ.empty()
          ? 0U
          : (open_state_mask_limit(static_cast<int>(left_root_term.beta_occ.size())) - 1U);

  const std::vector<int> alpha_root_rows = select_occ_by_mask(
      right_root_term.alpha_occ,
      alpha_row_full_mask ^ used_alpha_row_mask);
  const std::vector<int> alpha_root_cols = select_occ_by_mask(
      left_root_term.alpha_occ,
      alpha_col_full_mask ^ used_alpha_col_mask);
  const std::vector<int> beta_root_rows = select_occ_by_mask(
      right_root_term.beta_occ,
      beta_row_full_mask ^ used_beta_row_mask);
  const std::vector<int> beta_root_cols = select_occ_by_mask(
      left_root_term.beta_occ,
      beta_col_full_mask ^ used_beta_col_mask);

  accumulate_spin_deleted_payload_overlap_gradient(
      alpha_root_cols,
      alpha_root_rows,
      0,
      0,
      false,
      PartialSide::RightComplement,
      alpha_adjoint,
      context.support_overlap_storage,
      context.support_size,
      context.overlap_resolver,
      context.active_orbital_overlap_gradient);
  accumulate_spin_deleted_payload_overlap_gradient(
      beta_root_cols,
      beta_root_rows,
      0,
      0,
      false,
      PartialSide::RightComplement,
      beta_adjoint,
      context.support_overlap_storage,
      context.support_size,
      context.overlap_resolver,
      context.active_orbital_overlap_gradient);
}

void reverse_interface_joint_payload_overlap_gradient(
    const OverlapGradientReverseContext& context,
    const std::vector<int>& selected_parent_alpha_cols,
    const std::vector<int>& alpha_local_remainder_cols,
    const std::vector<int>& selected_parent_alpha_rows,
    const std::vector<int>& alpha_local_remainder_rows,
    const std::vector<int>& selected_parent_beta_cols,
    const std::vector<int>& beta_local_remainder_cols,
    const std::vector<int>& selected_parent_beta_rows,
    const std::vector<int>& beta_local_remainder_rows,
    const JointDeletionPayload& interface_adjoint) {
  if (!is_joint_payload_nonzero(interface_adjoint)) {
    return;
  }

  const SpinDeletionPayload alpha_payload =
      build_exact_frontier_spin_payload_limited(
          selected_parent_alpha_cols,
          alpha_local_remainder_cols,
          selected_parent_alpha_rows,
          alpha_local_remainder_rows,
          2,
          context.support_overlap_storage,
          context.support_size,
          context.overlap_resolver,
          context.subdeterminant_evaluations,
          context.spin_payload_cache);
  const SpinDeletionPayload beta_payload =
      build_exact_frontier_spin_payload_limited(
          selected_parent_beta_cols,
          beta_local_remainder_cols,
          selected_parent_beta_rows,
          beta_local_remainder_rows,
          2,
          context.support_overlap_storage,
          context.support_size,
          context.overlap_resolver,
          context.subdeterminant_evaluations,
          context.spin_payload_cache);

  SpinDeletionPayload alpha_adjoint = make_zero_spin_payload_like(alpha_payload);
  SpinDeletionPayload beta_adjoint = make_zero_spin_payload_like(beta_payload);
  reverse_build_joint_payload(
      alpha_payload,
      beta_payload,
      interface_adjoint,
      &alpha_adjoint,
      &beta_adjoint);

  std::vector<int> alpha_frontier_left_occ;
  std::vector<int> alpha_frontier_right_occ;
  assign_concatenated_occ(
      selected_parent_alpha_cols,
      alpha_local_remainder_cols,
      &alpha_frontier_left_occ);
  assign_concatenated_occ(
      selected_parent_alpha_rows,
      alpha_local_remainder_rows,
      &alpha_frontier_right_occ);
  std::vector<int> beta_frontier_left_occ;
  std::vector<int> beta_frontier_right_occ;
  assign_concatenated_occ(
      selected_parent_beta_cols,
      beta_local_remainder_cols,
      &beta_frontier_left_occ);
  assign_concatenated_occ(
      selected_parent_beta_rows,
      beta_local_remainder_rows,
      &beta_frontier_right_occ);

  accumulate_spin_deleted_payload_overlap_gradient(
      alpha_frontier_left_occ,
      alpha_frontier_right_occ,
      static_cast<int>(selected_parent_alpha_rows.size()),
      static_cast<int>(selected_parent_alpha_cols.size()),
      true,
      PartialSide::LeftFrontier,
      alpha_adjoint,
      context.support_overlap_storage,
      context.support_size,
      context.overlap_resolver,
      context.active_orbital_overlap_gradient);
  accumulate_spin_deleted_payload_overlap_gradient(
      beta_frontier_left_occ,
      beta_frontier_right_occ,
      static_cast<int>(selected_parent_beta_rows.size()),
      static_cast<int>(selected_parent_beta_cols.size()),
      true,
      PartialSide::LeftFrontier,
      beta_adjoint,
      context.support_overlap_storage,
      context.support_size,
      context.overlap_resolver,
      context.active_orbital_overlap_gradient);
}

void reverse_root_hamiltonian_payload_overlap_gradient(
    const OverlapGradientReverseContext& context,
    const OrientationTerm& left_root_term,
    const OrientationTerm& right_root_term,
    std::uint32_t used_alpha_row_mask,
    std::uint32_t used_alpha_col_mask,
    std::uint32_t used_beta_row_mask,
    std::uint32_t used_beta_col_mask,
    const HamiltonianBoundaryPayload& root_adjoint) {
  if (!is_hamiltonian_payload_nonzero(root_adjoint)) {
    return;
  }

  const SpinDeletionPayload alpha_payload =
      build_root_spin_payload_limited(
          left_root_term.alpha_occ,
          right_root_term.alpha_occ,
          used_alpha_row_mask,
          used_alpha_col_mask,
          2,
          context.support_overlap_storage,
          context.support_size,
          context.overlap_resolver,
          context.subdeterminant_evaluations,
          context.spin_payload_cache);
  const SpinDeletionPayload beta_payload =
      build_root_spin_payload_limited(
          left_root_term.beta_occ,
          right_root_term.beta_occ,
          used_beta_row_mask,
          used_beta_col_mask,
          2,
          context.support_overlap_storage,
          context.support_size,
          context.overlap_resolver,
          context.subdeterminant_evaluations,
          context.spin_payload_cache);

  SpinDeletionPayload alpha_adjoint = make_zero_spin_payload_like(alpha_payload);
  SpinDeletionPayload beta_adjoint = make_zero_spin_payload_like(beta_payload);
  reverse_build_hamiltonian_boundary_payload(
      alpha_payload,
      beta_payload,
      root_adjoint,
      &alpha_adjoint,
      &beta_adjoint);

  const std::uint32_t alpha_row_full_mask =
      right_root_term.alpha_occ.empty()
          ? 0U
          : (open_state_mask_limit(static_cast<int>(right_root_term.alpha_occ.size())) - 1U);
  const std::uint32_t alpha_col_full_mask =
      left_root_term.alpha_occ.empty()
          ? 0U
          : (open_state_mask_limit(static_cast<int>(left_root_term.alpha_occ.size())) - 1U);
  const std::uint32_t beta_row_full_mask =
      right_root_term.beta_occ.empty()
          ? 0U
          : (open_state_mask_limit(static_cast<int>(right_root_term.beta_occ.size())) - 1U);
  const std::uint32_t beta_col_full_mask =
      left_root_term.beta_occ.empty()
          ? 0U
          : (open_state_mask_limit(static_cast<int>(left_root_term.beta_occ.size())) - 1U);

  const std::vector<int> alpha_root_rows = select_occ_by_mask(
      right_root_term.alpha_occ,
      alpha_row_full_mask ^ used_alpha_row_mask);
  const std::vector<int> alpha_root_cols = select_occ_by_mask(
      left_root_term.alpha_occ,
      alpha_col_full_mask ^ used_alpha_col_mask);
  const std::vector<int> beta_root_rows = select_occ_by_mask(
      right_root_term.beta_occ,
      beta_row_full_mask ^ used_beta_row_mask);
  const std::vector<int> beta_root_cols = select_occ_by_mask(
      left_root_term.beta_occ,
      beta_col_full_mask ^ used_beta_col_mask);

  accumulate_spin_deleted_payload_overlap_gradient(
      alpha_root_cols,
      alpha_root_rows,
      0,
      0,
      false,
      PartialSide::RightComplement,
      alpha_adjoint,
      context.support_overlap_storage,
      context.support_size,
      context.overlap_resolver,
      context.active_orbital_overlap_gradient);
  accumulate_spin_deleted_payload_overlap_gradient(
      beta_root_cols,
      beta_root_rows,
      0,
      0,
      false,
      PartialSide::RightComplement,
      beta_adjoint,
      context.support_overlap_storage,
      context.support_size,
      context.overlap_resolver,
      context.active_orbital_overlap_gradient);
}

void reverse_interface_hamiltonian_payload_overlap_gradient(
    const OverlapGradientReverseContext& context,
    const std::vector<int>& selected_parent_alpha_cols,
    const std::vector<int>& alpha_local_remainder_cols,
    const std::vector<int>& selected_parent_alpha_rows,
    const std::vector<int>& alpha_local_remainder_rows,
    const std::vector<int>& selected_parent_beta_cols,
    const std::vector<int>& beta_local_remainder_cols,
    const std::vector<int>& selected_parent_beta_rows,
    const std::vector<int>& beta_local_remainder_rows,
    const HamiltonianBoundaryPayload& interface_adjoint) {
  if (!is_hamiltonian_payload_nonzero(interface_adjoint)) {
    return;
  }

  const SpinDeletionPayload alpha_payload =
      build_exact_frontier_spin_payload_limited(
          selected_parent_alpha_cols,
          alpha_local_remainder_cols,
          selected_parent_alpha_rows,
          alpha_local_remainder_rows,
          2,
          context.support_overlap_storage,
          context.support_size,
          context.overlap_resolver,
          context.subdeterminant_evaluations,
          context.spin_payload_cache);
  const SpinDeletionPayload beta_payload =
      build_exact_frontier_spin_payload_limited(
          selected_parent_beta_cols,
          beta_local_remainder_cols,
          selected_parent_beta_rows,
          beta_local_remainder_rows,
          2,
          context.support_overlap_storage,
          context.support_size,
          context.overlap_resolver,
          context.subdeterminant_evaluations,
          context.spin_payload_cache);

  SpinDeletionPayload alpha_adjoint = make_zero_spin_payload_like(alpha_payload);
  SpinDeletionPayload beta_adjoint = make_zero_spin_payload_like(beta_payload);
  reverse_build_hamiltonian_boundary_payload(
      alpha_payload,
      beta_payload,
      interface_adjoint,
      &alpha_adjoint,
      &beta_adjoint);

  std::vector<int> alpha_frontier_left_occ;
  std::vector<int> alpha_frontier_right_occ;
  assign_concatenated_occ(
      selected_parent_alpha_cols,
      alpha_local_remainder_cols,
      &alpha_frontier_left_occ);
  assign_concatenated_occ(
      selected_parent_alpha_rows,
      alpha_local_remainder_rows,
      &alpha_frontier_right_occ);
  std::vector<int> beta_frontier_left_occ;
  std::vector<int> beta_frontier_right_occ;
  assign_concatenated_occ(
      selected_parent_beta_cols,
      beta_local_remainder_cols,
      &beta_frontier_left_occ);
  assign_concatenated_occ(
      selected_parent_beta_rows,
      beta_local_remainder_rows,
      &beta_frontier_right_occ);

  accumulate_spin_deleted_payload_overlap_gradient(
      alpha_frontier_left_occ,
      alpha_frontier_right_occ,
      static_cast<int>(selected_parent_alpha_rows.size()),
      static_cast<int>(selected_parent_alpha_cols.size()),
      true,
      PartialSide::LeftFrontier,
      alpha_adjoint,
      context.support_overlap_storage,
      context.support_size,
      context.overlap_resolver,
      context.active_orbital_overlap_gradient);
  accumulate_spin_deleted_payload_overlap_gradient(
      beta_frontier_left_occ,
      beta_frontier_right_occ,
      static_cast<int>(selected_parent_beta_rows.size()),
      static_cast<int>(selected_parent_beta_cols.size()),
      true,
      PartialSide::LeftFrontier,
      beta_adjoint,
      context.support_overlap_storage,
      context.support_size,
      context.overlap_resolver,
      context.active_orbital_overlap_gradient);
}

JointDeletionPayload build_root_joint_payload_limited(
    const OrientationTerm& left_root_term,
    const OrientationTerm& right_root_term,
    std::uint32_t used_alpha_row_mask,
    std::uint32_t used_alpha_col_mask,
    std::uint32_t used_beta_row_mask,
    std::uint32_t used_beta_col_mask,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    std::unordered_map<SpinPayloadCacheKey,
                       SpinDeletionPayload,
                       SpinPayloadCacheKeyHasher>* payload_cache) {
  const SpinDeletionPayload alpha_payload =
      build_root_spin_payload_limited(
          left_root_term.alpha_occ,
          right_root_term.alpha_occ,
          used_alpha_row_mask,
          used_alpha_col_mask,
          2,
          overlap_storage,
          n_orbitals,
          overlap_resolver,
          subdeterminant_evaluations,
          payload_cache);
  const SpinDeletionPayload beta_payload =
      build_root_spin_payload_limited(
          left_root_term.beta_occ,
          right_root_term.beta_occ,
          used_beta_row_mask,
          used_beta_col_mask,
          2,
          overlap_storage,
          n_orbitals,
          overlap_resolver,
          subdeterminant_evaluations,
          payload_cache);
  return build_joint_payload(alpha_payload, beta_payload);
}

JointDeletionPayload build_weighted_root_joint_seed(
    const JointDeletionPayload& total_payload,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    double hamiltonian_weight,
    double overlap_weight) {
  // Builds the exact root adjoint on the generic deleted-minor bundle for the
  // scalar
  //   hamiltonian_weight * (H^(1) + H^(2)) + overlap_weight * S.
  JointDeletionPayload seed = make_zero_joint_payload_like(total_payload);
  if (std::abs(overlap_weight) > 1.0e-15) {
    seed.sectors[make_joint_key(make_spin_key(), make_spin_key())] += overlap_weight;
  }
  if (std::abs(hamiltonian_weight) <= 1.0e-15) {
    cleanup_joint_payload(&seed);
    return seed;
  }

  for (const auto& key : total_payload.basis_keys) {
    if (key.alpha_key.row_labels.size() == 1U &&
        key.alpha_key.col_labels.size() == 1U &&
        is_empty_spin_key(key.beta_key)) {
      const int right_orbital = key.alpha_key.row_labels[0];
      const int left_orbital = key.alpha_key.col_labels[0];
      seed.sectors[key] +=
          hamiltonian_weight *
          support_one_electron_storage[xmvb::to_size(left_orbital) *
                                           xmvb::to_size(support_size) +
                                       xmvb::to_size(right_orbital)];
      continue;
    }
    if (key.beta_key.row_labels.size() == 1U &&
        key.beta_key.col_labels.size() == 1U &&
        is_empty_spin_key(key.alpha_key)) {
      const int right_orbital = key.beta_key.row_labels[0];
      const int left_orbital = key.beta_key.col_labels[0];
      seed.sectors[key] +=
          hamiltonian_weight *
          support_one_electron_storage[xmvb::to_size(left_orbital) *
                                           xmvb::to_size(support_size) +
                                       xmvb::to_size(right_orbital)];
      continue;
    }
    if (key.alpha_key.row_labels.size() == 2U &&
        key.alpha_key.col_labels.size() == 2U &&
        is_empty_spin_key(key.beta_key)) {
      const int direct_index = TwoElectronIndexer::two_electron_storage_index(
          key.alpha_key.row_labels[0],
          key.alpha_key.col_labels[0],
          key.alpha_key.row_labels[1],
          key.alpha_key.col_labels[1]);
      const int exchange_index = TwoElectronIndexer::two_electron_storage_index(
          key.alpha_key.row_labels[0],
          key.alpha_key.col_labels[1],
          key.alpha_key.row_labels[1],
          key.alpha_key.col_labels[0]);
      seed.sectors[key] +=
          hamiltonian_weight *
          (packed_active_two_electron_integrals[xmvb::to_size(direct_index)] -
           packed_active_two_electron_integrals[xmvb::to_size(exchange_index)]);
      continue;
    }
    if (key.beta_key.row_labels.size() == 2U &&
        key.beta_key.col_labels.size() == 2U &&
        is_empty_spin_key(key.alpha_key)) {
      const int direct_index = TwoElectronIndexer::two_electron_storage_index(
          key.beta_key.row_labels[0],
          key.beta_key.col_labels[0],
          key.beta_key.row_labels[1],
          key.beta_key.col_labels[1]);
      const int exchange_index = TwoElectronIndexer::two_electron_storage_index(
          key.beta_key.row_labels[0],
          key.beta_key.col_labels[1],
          key.beta_key.row_labels[1],
          key.beta_key.col_labels[0]);
      seed.sectors[key] +=
          hamiltonian_weight *
          (packed_active_two_electron_integrals[xmvb::to_size(direct_index)] -
           packed_active_two_electron_integrals[xmvb::to_size(exchange_index)]);
      continue;
    }
    if (key.alpha_key.row_labels.size() == 1U &&
        key.alpha_key.col_labels.size() == 1U &&
        key.beta_key.row_labels.size() == 1U &&
        key.beta_key.col_labels.size() == 1U) {
      const int eri_index = TwoElectronIndexer::two_electron_storage_index(
          key.beta_key.row_labels[0],
          key.beta_key.col_labels[0],
          key.alpha_key.row_labels[0],
          key.alpha_key.col_labels[0]);
      seed.sectors[key] +=
          hamiltonian_weight *
          packed_active_two_electron_integrals[xmvb::to_size(eri_index)];
    }
  }
  cleanup_joint_payload(&seed);
  return seed;
}

HamiltonianBoundaryPayload build_weighted_root_hamiltonian_seed(
    const HamiltonianBoundaryPayload& total_payload,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    double hamiltonian_weight,
    double overlap_weight) {
  return project_joint_payload_to_hamiltonian(
      build_weighted_root_joint_seed(
          convert_hamiltonian_payload_to_joint(total_payload),
          support_one_electron_storage,
          packed_active_two_electron_integrals,
          support_size,
          hamiltonian_weight,
          overlap_weight));
}

bool recursive_reverse_debug_validation_enabled() {
  const char* debug_flag = std::getenv("XMVB_DEBUG_EXACT_SEPARATOR_REVERSE");
  return debug_flag != nullptr && debug_flag[0] != '\0' && debug_flag[0] != '0';
}

void accumulate_direct_subtree_message_overlap_gradient_exact(
    const OverlapGradientReverseContext& context,
    const SubtreeExpansion& subtree_expansion,
    const OrientationTerm& left_parent_term,
    const OrientationTerm& right_parent_term,
    std::uint32_t target_alpha_row_mask,
    std::uint32_t target_alpha_col_mask,
    std::uint32_t target_beta_row_mask,
    std::uint32_t target_beta_col_mask,
    const JointDeletionPayload& target_adjoint,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }
  if (!is_joint_payload_nonzero(target_adjoint)) {
    return;
  }

  const std::vector<GlobalOrientationTerm> left_global_terms =
      build_global_orientation_terms(subtree_expansion.preorder_components, true);
  const std::vector<GlobalOrientationTerm> right_global_terms =
      build_global_orientation_terms(subtree_expansion.preorder_components, false);
  const auto left_parent_alpha_occ_by_mask =
      build_mask_occ_table(left_parent_term.alpha_occ);
  const auto right_parent_alpha_occ_by_mask =
      build_mask_occ_table(right_parent_term.alpha_occ);
  const auto left_parent_beta_occ_by_mask =
      build_mask_occ_table(left_parent_term.beta_occ);
  const auto right_parent_beta_occ_by_mask =
      build_mask_occ_table(right_parent_term.beta_occ);

  const auto& selected_parent_alpha_rows =
      right_parent_alpha_occ_by_mask[xmvb::to_size(target_alpha_row_mask)];
  const auto& selected_parent_alpha_cols =
      left_parent_alpha_occ_by_mask[xmvb::to_size(target_alpha_col_mask)];
  const auto& selected_parent_beta_rows =
      right_parent_beta_occ_by_mask[xmvb::to_size(target_beta_row_mask)];
  const auto& selected_parent_beta_cols =
      left_parent_beta_occ_by_mask[xmvb::to_size(target_beta_col_mask)];

  OverlapGradientReverseContext exact_context = context;
  exact_context.active_orbital_overlap_gradient = active_orbital_overlap_gradient;
  for (const auto& left_global_term : left_global_terms) {
    for (const auto& right_global_term : right_global_terms) {
      const double coefficient =
          left_global_term.coefficient * right_global_term.coefficient;
      if (std::abs(coefficient) <= 1.0e-15) {
        continue;
      }

      reverse_interface_joint_payload_overlap_gradient(
          exact_context,
          selected_parent_alpha_cols,
          left_global_term.alpha_occ,
          selected_parent_alpha_rows,
          right_global_term.alpha_occ,
          selected_parent_beta_cols,
          left_global_term.beta_occ,
          selected_parent_beta_rows,
          right_global_term.beta_occ,
          scale_joint_payload(target_adjoint, coefficient));
    }
  }
}

void backprop_recursive_subtree_message_collapsed(
    const OverlapGradientReverseContext& context,
    int node,
    const OrientationTerm& left_parent_term,
    const OrientationTerm& right_parent_term,
    std::uint32_t target_alpha_row_mask,
    std::uint32_t target_alpha_col_mask,
    std::uint32_t target_beta_row_mask,
    std::uint32_t target_beta_col_mask,
    const JointDeletionPayload& target_adjoint) {
  if (!is_joint_payload_nonzero(target_adjoint)) {
    return;
  }

  const ComponentData& node_component =
      context.tree.components[xmvb::to_size(node)];
  const std::vector<int>& children =
      context.tree.children[xmvb::to_size(node)];
  const SubtreeExpansion& subtree_expansion =
      context.subtree_expansions[xmvb::to_size(node)];
  if (children.empty()) {
    accumulate_direct_subtree_message_overlap_gradient_exact(
        context,
        subtree_expansion,
        left_parent_term,
        right_parent_term,
        target_alpha_row_mask,
        target_alpha_col_mask,
        target_beta_row_mask,
        target_beta_col_mask,
        target_adjoint,
        context.active_orbital_overlap_gradient);
    return;
  }

  const bool validate_recursive_reverse =
      recursive_reverse_debug_validation_enabled();
  std::vector<double> gradient_before;
  if (validate_recursive_reverse) {
    gradient_before = *context.active_orbital_overlap_gradient;
  }

  const auto left_parent_alpha_occ_by_mask =
      build_mask_occ_table(left_parent_term.alpha_occ);
  const auto right_parent_alpha_occ_by_mask =
      build_mask_occ_table(right_parent_term.alpha_occ);
  const auto left_parent_beta_occ_by_mask =
      build_mask_occ_table(left_parent_term.beta_occ);
  const auto right_parent_beta_occ_by_mask =
      build_mask_occ_table(right_parent_term.beta_occ);
  const ChildSpinSizeVectors child_spin_sizes =
      build_child_spin_size_vectors(children, context.subtree_expansions);

  const auto& selected_parent_alpha_rows =
      right_parent_alpha_occ_by_mask[xmvb::to_size(target_alpha_row_mask)];
  const auto& selected_parent_alpha_cols =
      left_parent_alpha_occ_by_mask[xmvb::to_size(target_alpha_col_mask)];
  const auto& selected_parent_beta_rows =
      right_parent_beta_occ_by_mask[xmvb::to_size(target_beta_row_mask)];
  const auto& selected_parent_beta_cols =
      left_parent_beta_occ_by_mask[xmvb::to_size(target_beta_col_mask)];

  const std::vector<WeightedOrientationTermPair> node_term_pairs =
      build_weighted_component_orientation_pairs(node_component);
  for (const auto& node_term_pair : node_term_pairs) {
    const OrientationTerm& left_node_term = node_term_pair.left_term;
    const OrientationTerm& right_node_term = node_term_pair.right_term;
    const double local_coefficient = node_term_pair.coefficient;
    if (std::abs(local_coefficient) <= 1.0e-15) {
      continue;
    }

    std::vector<const std::vector<ComponentTreeLeafMessage>*> child_messages(
        children.size(),
        nullptr);
    for (std::size_t child_index = 0; child_index < children.size(); ++child_index) {
      const int child = children[child_index];
      child_messages[child_index] =
          &build_recursive_subtree_messages_limited_cached(
              context.tree,
              context.subtree_expansions,
              child,
              left_node_term,
              right_node_term,
              2,
              true,
              context.support_overlap_storage,
              context.support_size,
              context.overlap_resolver,
              context.subtree_term_pair_count,
              context.subtree_message_state_count,
              context.subdeterminant_evaluations,
              context.dp_transition_count,
              context.spin_payload_cache,
              context.subtree_message_cache);
    }

    const std::uint32_t local_alpha_row_full_mask =
        right_node_term.alpha_occ.empty()
            ? 0U
            : (open_state_mask_limit(static_cast<int>(right_node_term.alpha_occ.size())) - 1U);
    const std::uint32_t local_alpha_col_full_mask =
        left_node_term.alpha_occ.empty()
            ? 0U
            : (open_state_mask_limit(static_cast<int>(left_node_term.alpha_occ.size())) - 1U);
    const std::uint32_t local_beta_row_full_mask =
        right_node_term.beta_occ.empty()
            ? 0U
            : (open_state_mask_limit(static_cast<int>(right_node_term.beta_occ.size())) - 1U);
    const std::uint32_t local_beta_col_full_mask =
        left_node_term.beta_occ.empty()
            ? 0U
            : (open_state_mask_limit(static_cast<int>(left_node_term.beta_occ.size())) - 1U);
    std::vector<FrontierJointMessage> prefix_messages;
    prefix_messages.reserve(children.size() + 1U);
    prefix_messages.push_back(make_identity_frontier_joint_message(
        static_cast<int>(right_node_term.alpha_occ.size()),
        static_cast<int>(left_node_term.alpha_occ.size()),
        static_cast<int>(right_node_term.beta_occ.size()),
        static_cast<int>(left_node_term.beta_occ.size())));
    std::uint64_t reverse_frontier_transition_count = 0;
    for (std::size_t child_index = 0; child_index < children.size(); ++child_index) {
      prefix_messages.push_back(
          merge_frontier_joint_message_with_child(
              prefix_messages.back(),
              *child_messages[child_index],
              child_spin_sizes.right_alpha[child_index],
              child_spin_sizes.left_alpha[child_index],
              child_spin_sizes.right_beta[child_index],
              child_spin_sizes.left_beta[child_index],
              2,
              true,
              &reverse_frontier_transition_count));
    }
    const FrontierJointMessage& frontier_message = prefix_messages.back();

    std::map<FrontierHamiltonianMaskKey, JointDeletionPayload> frontier_adjoint_map;
    for (const FrontierJointEntry& frontier_entry : frontier_message.entries) {
      const std::uint32_t used_alpha_row_mask =
          frontier_entry.key.alpha_row_mask;
      const std::uint32_t used_alpha_col_mask =
          frontier_entry.key.alpha_col_mask;
      const std::uint32_t used_beta_row_mask =
          frontier_entry.key.beta_row_mask;
      const std::uint32_t used_beta_col_mask =
          frontier_entry.key.beta_col_mask;
      const std::vector<int> alpha_local_remainder_rows = select_occ_by_mask(
          right_node_term.alpha_occ,
          local_alpha_row_full_mask ^ used_alpha_row_mask);
      const std::vector<int> alpha_local_remainder_cols = select_occ_by_mask(
          left_node_term.alpha_occ,
          local_alpha_col_full_mask ^ used_alpha_col_mask);
      const std::vector<int> alpha_frontier_interface_rows = select_occ_by_mask(
          right_node_term.alpha_occ,
          used_alpha_row_mask);
      const std::vector<int> alpha_frontier_interface_cols = select_occ_by_mask(
          left_node_term.alpha_occ,
          used_alpha_col_mask);
      const std::vector<int> beta_local_remainder_rows = select_occ_by_mask(
          right_node_term.beta_occ,
          local_beta_row_full_mask ^ used_beta_row_mask);
      const std::vector<int> beta_local_remainder_cols = select_occ_by_mask(
          left_node_term.beta_occ,
          local_beta_col_full_mask ^ used_beta_col_mask);
      const std::vector<int> beta_frontier_interface_rows = select_occ_by_mask(
          right_node_term.beta_occ,
          used_beta_row_mask);
      const std::vector<int> beta_frontier_interface_cols = select_occ_by_mask(
          left_node_term.beta_occ,
          used_beta_col_mask);

      const SpinDeletionPayload alpha_interface_payload =
          build_exact_frontier_spin_payload_limited(
              selected_parent_alpha_cols,
              alpha_local_remainder_cols,
              selected_parent_alpha_rows,
              alpha_local_remainder_rows,
              2,
              context.support_overlap_storage,
              context.support_size,
              context.overlap_resolver,
              context.subdeterminant_evaluations,
              context.spin_payload_cache);
      const SpinDeletionPayload beta_interface_payload =
          build_exact_frontier_spin_payload_limited(
              selected_parent_beta_cols,
              beta_local_remainder_cols,
              selected_parent_beta_rows,
              beta_local_remainder_rows,
              2,
              context.support_overlap_storage,
              context.support_size,
              context.overlap_resolver,
              context.subdeterminant_evaluations,
              context.spin_payload_cache);
      const JointDeletionPayload interface_payload =
          build_joint_payload(alpha_interface_payload, beta_interface_payload);
      const JointDeletionPayload natural_payload =
          merge_joint_deletion_payloads_limited(
              frontier_entry.payload,
              interface_payload,
              2);
      const JointDeletionPayload front_convention_payload =
          transform_interface_block_to_front(
              natural_payload,
              static_cast<int>(selected_parent_alpha_rows.size()),
              static_cast<int>(selected_parent_alpha_cols.size()),
              alpha_frontier_interface_rows,
              alpha_frontier_interface_cols,
              alpha_local_remainder_rows,
              alpha_local_remainder_cols,
              frontier_entry.payload.alpha_row_count,
              frontier_entry.payload.alpha_col_count,
              static_cast<int>(selected_parent_beta_rows.size()),
              static_cast<int>(selected_parent_beta_cols.size()),
              beta_frontier_interface_rows,
              beta_frontier_interface_cols,
              beta_local_remainder_rows,
              beta_local_remainder_cols,
              frontier_entry.payload.beta_row_count,
              frontier_entry.payload.beta_col_count);
      if (front_convention_payload.basis_keys.empty()) {
        continue;
      }

      const JointDeletionPayload front_adjoint =
          scale_joint_payload(target_adjoint, local_coefficient);
      JointDeletionPayload natural_adjoint =
          make_zero_joint_payload_like(natural_payload);
      reverse_transform_interface_block_to_front(
          static_cast<int>(selected_parent_alpha_rows.size()),
          static_cast<int>(selected_parent_alpha_cols.size()),
          alpha_frontier_interface_rows,
          alpha_frontier_interface_cols,
          alpha_local_remainder_rows,
          alpha_local_remainder_cols,
          frontier_entry.payload.alpha_row_count,
          frontier_entry.payload.alpha_col_count,
          static_cast<int>(selected_parent_beta_rows.size()),
          static_cast<int>(selected_parent_beta_cols.size()),
          beta_frontier_interface_rows,
          beta_frontier_interface_cols,
          beta_local_remainder_rows,
          beta_local_remainder_cols,
          frontier_entry.payload.beta_row_count,
          frontier_entry.payload.beta_col_count,
          front_adjoint,
          &natural_adjoint);

      JointDeletionPayload interface_adjoint =
          make_zero_joint_payload_like(interface_payload);
      JointDeletionPayload frontier_branch_adjoint =
          make_zero_joint_payload_like(frontier_entry.payload);
      reverse_merge_joint_deletion_payloads_limited(
          frontier_entry.payload,
          interface_payload,
          2,
          natural_adjoint,
          &frontier_branch_adjoint,
          &interface_adjoint);
      add_scaled_joint_payload_preserve_basis(
          frontier_branch_adjoint,
          1.0,
          true,
          &frontier_adjoint_map[frontier_entry.key]);
      reverse_interface_joint_payload_overlap_gradient(
          context,
          selected_parent_alpha_cols,
          alpha_local_remainder_cols,
          selected_parent_alpha_rows,
          alpha_local_remainder_rows,
          selected_parent_beta_cols,
          beta_local_remainder_cols,
          selected_parent_beta_rows,
          beta_local_remainder_rows,
          interface_adjoint);
    }

    FrontierJointMessage frontier_adjoint =
        finalize_frontier_joint_message(
            frontier_message.alpha_row_count,
            frontier_message.alpha_col_count,
            frontier_message.beta_row_count,
            frontier_message.beta_col_count,
            &frontier_adjoint_map,
            true);
    std::vector<std::vector<ComponentTreeLeafMessage>> child_adjoint_messages(
        children.size());
    FrontierJointMessage identity_adjoint;
    reverse_frontier_joint_message(
        child_messages,
        child_spin_sizes,
        prefix_messages,
        frontier_adjoint,
        2,
        &child_adjoint_messages,
        &identity_adjoint);
    static_cast<void>(identity_adjoint);

    for (std::size_t child_index = 0; child_index < children.size(); ++child_index) {
      for (const ComponentTreeLeafMessage& child_adjoint :
           child_adjoint_messages[child_index]) {
        backprop_recursive_subtree_message_collapsed(
            context,
            children[child_index],
            left_node_term,
            right_node_term,
            child_adjoint.alpha_row_mask,
            child_adjoint.alpha_col_mask,
            child_adjoint.beta_row_mask,
            child_adjoint.beta_col_mask,
            child_adjoint.payload);
      }
    }
  }

  if (!validate_recursive_reverse) {
    return;
  }

  std::vector<double> exact_gradient(
      xmvb::to_size(context.support_size * context.support_size),
      0.0);
  accumulate_direct_subtree_message_overlap_gradient_exact(
      context,
      subtree_expansion,
      left_parent_term,
      right_parent_term,
      target_alpha_row_mask,
      target_alpha_col_mask,
      target_beta_row_mask,
      target_beta_col_mask,
      target_adjoint,
      &exact_gradient);

  constexpr double tolerance = 1.0e-10;
  for (std::size_t index = 0; index < exact_gradient.size(); ++index) {
    const double recursive_value =
        (*context.active_orbital_overlap_gradient)[index] - gradient_before[index];
    const double exact_value = exact_gradient[index];
    if (std::abs(recursive_value - exact_value) > tolerance) {
      std::ostringstream stream;
      stream << "recursive subtree reverse mismatch at node " << node
             << " masks=("
             << target_alpha_row_mask << ','
             << target_alpha_col_mask << ','
             << target_beta_row_mask << ','
             << target_beta_col_mask << ") index=" << index
             << " recursive=" << recursive_value
             << " exact=" << exact_value
             << " target_adjoint="
             << format_joint_payload_summary(target_adjoint);
      throw std::runtime_error(stream.str());
    }
  }
}

void accumulate_rooted_component_tree_weighted_overlap_gradient_boundary_collapsed(
    const ComponentTree& tree,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const DeterminantOverlapResolver& overlap_resolver,
    double hamiltonian_weight,
    double overlap_weight,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }
  active_orbital_overlap_gradient->assign(
      xmvb::to_size(support_size * support_size),
      0.0);
  if (std::abs(hamiltonian_weight) <= 1.0e-15 &&
      std::abs(overlap_weight) <= 1.0e-15) {
    return;
  }

  const std::vector<SubtreeExpansion> subtree_expansions =
      build_subtree_expansions(tree);
  const ComponentData& root_component =
      tree.components[xmvb::to_size(tree.root_index)];
  const std::vector<int>& root_children =
      tree.children[xmvb::to_size(tree.root_index)];
  const ChildSpinSizeVectors child_spin_sizes =
      build_child_spin_size_vectors(root_children, subtree_expansions);

  std::uint64_t subtree_term_pair_count = 0;
  std::uint64_t subtree_message_state_count = 0;
  std::uint64_t subdeterminant_evaluations = 0;
  std::uint64_t dp_transition_count = 0;
  std::unordered_map<SpinPayloadCacheKey,
                     SpinDeletionPayload,
                     SpinPayloadCacheKeyHasher> spin_payload_cache;
  std::unordered_map<SubtreeMessageCacheKey,
                     std::vector<ComponentTreeLeafMessage>,
                     SubtreeMessageCacheKeyHasher> subtree_message_cache;
  std::unordered_map<SubtreeMessageCacheKey,
                     BoundaryHamiltonianMessage,
                     SubtreeMessageCacheKeyHasher> hamiltonian_message_cache;
  const OverlapGradientReverseContext context{
      tree,
      subtree_expansions,
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      support_size,
      overlap_resolver,
      &subtree_term_pair_count,
      &subtree_message_state_count,
      &subdeterminant_evaluations,
      &dp_transition_count,
      &spin_payload_cache,
      &subtree_message_cache,
      &hamiltonian_message_cache,
      active_orbital_overlap_gradient,
  };

  const std::vector<WeightedOrientationTermPair> root_term_pairs =
      build_weighted_component_orientation_pairs(root_component);
  for (const auto& root_term_pair : root_term_pairs) {
    const OrientationTerm& left_root_term = root_term_pair.left_term;
    const OrientationTerm& right_root_term = root_term_pair.right_term;
    const double root_coefficient = root_term_pair.coefficient;
    if (std::abs(root_coefficient) <= 1.0e-15) {
      continue;
    }

    std::vector<const std::vector<ComponentTreeLeafMessage>*> child_messages(
        root_children.size(),
        nullptr);
    for (std::size_t child_index = 0; child_index < root_children.size();
         ++child_index) {
      const int child = root_children[child_index];
      child_messages[child_index] =
          &build_recursive_subtree_messages_limited_cached(
              tree,
              subtree_expansions,
              child,
              left_root_term,
              right_root_term,
              2,
              true,
              support_overlap_storage,
              support_size,
              overlap_resolver,
              &subtree_term_pair_count,
              &subtree_message_state_count,
              &subdeterminant_evaluations,
              &dp_transition_count,
              &spin_payload_cache,
              &subtree_message_cache);
    }

    const FrontierJointMessage frontier_message =
        build_frontier_joint_message(
            child_messages,
            child_spin_sizes,
            static_cast<int>(right_root_term.alpha_occ.size()),
            static_cast<int>(left_root_term.alpha_occ.size()),
            static_cast<int>(right_root_term.beta_occ.size()),
            static_cast<int>(left_root_term.beta_occ.size()),
            2,
            true,
            &dp_transition_count);

    std::map<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>,
             JointDeletionPayload>
        root_payload_cache;
    std::map<FrontierHamiltonianMaskKey, JointDeletionPayload> frontier_adjoint_map;
    for (const FrontierJointEntry& frontier_entry : frontier_message.entries) {
      const auto cache_key = std::make_tuple(
          frontier_entry.key.alpha_row_mask,
          frontier_entry.key.alpha_col_mask,
          frontier_entry.key.beta_row_mask,
          frontier_entry.key.beta_col_mask);
      auto cache_iterator = root_payload_cache.find(cache_key);
      if (cache_iterator == root_payload_cache.end()) {
        cache_iterator = root_payload_cache.emplace(
            cache_key,
            build_root_joint_payload_limited(
                left_root_term,
                right_root_term,
                frontier_entry.key.alpha_row_mask,
                frontier_entry.key.alpha_col_mask,
                frontier_entry.key.beta_row_mask,
                frontier_entry.key.beta_col_mask,
                support_overlap_storage,
                support_size,
                overlap_resolver,
                &subdeterminant_evaluations,
                &spin_payload_cache))
                             .first;
      }
      const JointDeletionPayload& root_payload = cache_iterator->second;
      if (root_payload.basis_keys.empty()) {
        continue;
      }

      const JointDeletionPayload total_payload =
          merge_joint_deletion_payloads_limited(
              frontier_entry.payload,
              root_payload,
              2);
      if (total_payload.basis_keys.empty()) {
        continue;
      }

      const JointDeletionPayload total_adjoint =
          scale_joint_payload(
              build_weighted_root_joint_seed(
                  total_payload,
                  support_one_electron_storage,
                  packed_active_two_electron_integrals,
                  support_size,
                  hamiltonian_weight,
                  overlap_weight),
              root_coefficient);
      if (!is_joint_payload_nonzero(total_adjoint)) {
        continue;
      }

      JointDeletionPayload root_adjoint =
          make_zero_joint_payload_like(root_payload);
      JointDeletionPayload frontier_branch_adjoint =
          make_zero_joint_payload_like(frontier_entry.payload);
      reverse_merge_joint_deletion_payloads_limited(
          frontier_entry.payload,
          root_payload,
          2,
          total_adjoint,
          &frontier_branch_adjoint,
          &root_adjoint);
      add_scaled_joint_payload_preserve_basis(
          frontier_branch_adjoint,
          1.0,
          true,
          &frontier_adjoint_map[frontier_entry.key]);
      reverse_root_joint_payload_overlap_gradient(
          context,
          left_root_term,
          right_root_term,
          frontier_entry.key.alpha_row_mask,
          frontier_entry.key.alpha_col_mask,
          frontier_entry.key.beta_row_mask,
          frontier_entry.key.beta_col_mask,
          root_adjoint);
    }

    FrontierJointMessage frontier_adjoint =
        finalize_frontier_joint_message(
            frontier_message.alpha_row_count,
            frontier_message.alpha_col_count,
            frontier_message.beta_row_count,
            frontier_message.beta_col_count,
            &frontier_adjoint_map,
            true);
    std::vector<FrontierJointMessage> prefix_messages;
    prefix_messages.reserve(root_children.size() + 1U);
    prefix_messages.push_back(make_identity_frontier_joint_message(
        static_cast<int>(right_root_term.alpha_occ.size()),
        static_cast<int>(left_root_term.alpha_occ.size()),
        static_cast<int>(right_root_term.beta_occ.size()),
        static_cast<int>(left_root_term.beta_occ.size())));
    std::uint64_t reverse_frontier_transition_count = 0;
    for (std::size_t child_index = 0; child_index < root_children.size(); ++child_index) {
      prefix_messages.push_back(
          merge_frontier_joint_message_with_child(
              prefix_messages.back(),
              *child_messages[child_index],
              child_spin_sizes.right_alpha[child_index],
              child_spin_sizes.left_alpha[child_index],
              child_spin_sizes.right_beta[child_index],
              child_spin_sizes.left_beta[child_index],
              2,
              true,
              &reverse_frontier_transition_count));
    }
    std::vector<std::vector<ComponentTreeLeafMessage>> child_adjoint_messages(
        root_children.size());
    FrontierJointMessage identity_adjoint;
    reverse_frontier_joint_message(
        child_messages,
        child_spin_sizes,
        prefix_messages,
        frontier_adjoint,
        2,
        &child_adjoint_messages,
        &identity_adjoint);
    static_cast<void>(identity_adjoint);

    for (std::size_t child_index = 0; child_index < root_children.size(); ++child_index) {
      for (const ComponentTreeLeafMessage& child_adjoint :
           child_adjoint_messages[child_index]) {
        backprop_recursive_subtree_message_collapsed(
            context,
            root_children[child_index],
            left_root_term,
            right_root_term,
            child_adjoint.alpha_row_mask,
            child_adjoint.alpha_col_mask,
            child_adjoint.beta_row_mask,
            child_adjoint.beta_col_mask,
            child_adjoint.payload);
      }
    }
  }
}

const DenseBoundaryHamiltonianMessage&
build_recursive_subtree_dense_hamiltonian_messages_cached(
    const ComponentTree& tree,
    const std::vector<SubtreeExpansion>& subtree_expansions,
    int node,
    const OrientationTerm& left_parent_term,
    const OrientationTerm& right_parent_term,
    const std::vector<double>& overlap_storage,
    int support_size,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subtree_term_pair_count,
    std::uint64_t* subtree_message_state_count,
    std::uint64_t* subdeterminant_evaluations,
    std::uint64_t* dp_transition_count,
    std::unordered_map<SpinPayloadCacheKey,
                       SpinDeletionPayload,
                       SpinPayloadCacheKeyHasher>* payload_cache,
    std::unordered_map<SubtreeMessageCacheKey,
                       DenseBoundaryHamiltonianMessage,
                       SubtreeMessageCacheKeyHasher>* message_cache) {
  if (subtree_term_pair_count == nullptr ||
      subtree_message_state_count == nullptr ||
      subdeterminant_evaluations == nullptr ||
      dp_transition_count == nullptr ||
      payload_cache == nullptr ||
      message_cache == nullptr) {
    throw std::invalid_argument("dense recursive subtree-message inputs must not be null");
  }

  const SubtreeMessageCacheKey cache_key{
      node,
      left_parent_term.alpha_occ,
      left_parent_term.beta_occ,
      right_parent_term.alpha_occ,
      right_parent_term.beta_occ,
      false,
  };
  auto cache_iterator = message_cache->find(cache_key);
  if (cache_iterator != message_cache->end()) {
    return cache_iterator->second;
  }

  const ComponentData& node_component =
      tree.components[xmvb::to_size(node)];
  const std::vector<int>& children = tree.children[xmvb::to_size(node)];
  const ComponentSpinSizes& subtree_sizes =
      subtree_expansions[xmvb::to_size(node)].spin_sizes;
  const auto left_parent_alpha_occ_by_mask = build_mask_occ_table(left_parent_term.alpha_occ);
  const auto right_parent_alpha_occ_by_mask = build_mask_occ_table(right_parent_term.alpha_occ);
  const auto left_parent_beta_occ_by_mask = build_mask_occ_table(left_parent_term.beta_occ);
  const auto right_parent_beta_occ_by_mask = build_mask_occ_table(right_parent_term.beta_occ);
  const ChildSpinSizeVectors child_spin_sizes =
      build_child_spin_size_vectors(children, subtree_expansions);

  DenseBoundaryHamiltonianMessage message;
  message.alpha_layout = BoundarySpinBundleLayout(
      static_cast<int>(right_parent_term.alpha_occ.size()),
      static_cast<int>(left_parent_term.alpha_occ.size()),
      subtree_sizes.left_alpha - subtree_sizes.right_alpha,
      -2,
      2,
      support_size);
  message.beta_layout = BoundarySpinBundleLayout(
      static_cast<int>(right_parent_term.beta_occ.size()),
      static_cast<int>(left_parent_term.beta_occ.size()),
      subtree_sizes.left_beta - subtree_sizes.right_beta,
      -2,
      2,
      support_size);
  message.support_size = support_size;
  message.payload_values.resize(
      xmvb::to_size(message.alpha_layout.total_sector_count()) *
      xmvb::to_size(message.beta_layout.total_sector_count()));

  if (children.empty()) {
    message = build_leaf_dense_hamiltonian_subtree_message_values(
        node_component,
        message.alpha_layout,
        message.beta_layout,
        left_parent_alpha_occ_by_mask,
        right_parent_alpha_occ_by_mask,
        left_parent_beta_occ_by_mask,
        right_parent_beta_occ_by_mask,
        overlap_storage,
        support_size,
        overlap_resolver,
        subtree_term_pair_count,
        subtree_message_state_count,
        subdeterminant_evaluations,
        payload_cache);
    return message_cache->emplace(cache_key, std::move(message)).first->second;
  }

  const std::vector<WeightedOrientationTermPair> node_term_pairs =
      build_weighted_component_orientation_pairs(node_component);
  for (const auto& node_term_pair : node_term_pairs) {
    const OrientationTerm& left_node_term = node_term_pair.left_term;
    const OrientationTerm& right_node_term = node_term_pair.right_term;
    const double local_coefficient = node_term_pair.coefficient;
    ++(*subtree_term_pair_count);

    std::vector<const DenseBoundaryHamiltonianMessage*> child_messages(
        children.size(),
        nullptr);
    for (std::size_t child_index = 0; child_index < children.size(); ++child_index) {
      const int child = children[child_index];
      child_messages[child_index] =
          &build_recursive_subtree_dense_hamiltonian_messages_cached(
              tree,
              subtree_expansions,
              child,
              left_node_term,
              right_node_term,
              overlap_storage,
              support_size,
              overlap_resolver,
              subtree_term_pair_count,
              subtree_message_state_count,
              subdeterminant_evaluations,
              dp_transition_count,
              payload_cache,
              message_cache);
    }

    const DenseFrontierHamiltonianMessage frontier_message =
        build_frontier_dense_hamiltonian_message(
            child_messages,
            child_spin_sizes,
            static_cast<int>(right_node_term.alpha_occ.size()),
            static_cast<int>(left_node_term.alpha_occ.size()),
            static_cast<int>(right_node_term.beta_occ.size()),
            static_cast<int>(left_node_term.beta_occ.size()),
            support_size,
            dp_transition_count);

    const std::uint32_t local_alpha_row_full_mask =
        right_node_term.alpha_occ.empty()
            ? 0U
            : (open_state_mask_limit(static_cast<int>(right_node_term.alpha_occ.size())) - 1U);
    const std::uint32_t local_alpha_col_full_mask =
        left_node_term.alpha_occ.empty()
            ? 0U
            : (open_state_mask_limit(static_cast<int>(left_node_term.alpha_occ.size())) - 1U);
    const std::uint32_t local_beta_row_full_mask =
        right_node_term.beta_occ.empty()
            ? 0U
            : (open_state_mask_limit(static_cast<int>(right_node_term.beta_occ.size())) - 1U);
    const std::uint32_t local_beta_col_full_mask =
        left_node_term.beta_occ.empty()
            ? 0U
            : (open_state_mask_limit(static_cast<int>(left_node_term.beta_occ.size())) - 1U);

    for (int alpha_flat_sector_index = 0;
         alpha_flat_sector_index < message.alpha_layout.total_sector_count();
         ++alpha_flat_sector_index) {
      const BoundarySector& alpha_sector =
          boundary_layout_sector(message.alpha_layout, alpha_flat_sector_index);
      const auto& selected_parent_alpha_rows =
          right_parent_alpha_occ_by_mask[xmvb::to_size(alpha_sector.row_mask)];
      const auto& selected_parent_alpha_cols =
          left_parent_alpha_occ_by_mask[xmvb::to_size(alpha_sector.col_mask)];

      for (int beta_flat_sector_index = 0;
           beta_flat_sector_index < message.beta_layout.total_sector_count();
           ++beta_flat_sector_index) {
        const BoundarySector& beta_sector =
            boundary_layout_sector(message.beta_layout, beta_flat_sector_index);
        const auto& selected_parent_beta_rows =
            right_parent_beta_occ_by_mask[xmvb::to_size(beta_sector.row_mask)];
        const auto& selected_parent_beta_cols =
            left_parent_beta_occ_by_mask[xmvb::to_size(beta_sector.col_mask)];
        for (const DenseFrontierHamiltonianEntry& frontier_entry : frontier_message.entries) {
          const std::uint32_t used_alpha_row_mask =
              frontier_entry.key.alpha_row_mask;
          const std::uint32_t used_alpha_col_mask =
              frontier_entry.key.alpha_col_mask;
          const std::uint32_t used_beta_row_mask =
              frontier_entry.key.beta_row_mask;
          const std::uint32_t used_beta_col_mask =
              frontier_entry.key.beta_col_mask;
          const auto alpha_local_remainder_rows = select_occ_by_mask(
              right_node_term.alpha_occ,
              local_alpha_row_full_mask ^ used_alpha_row_mask);
          const auto alpha_local_remainder_cols = select_occ_by_mask(
              left_node_term.alpha_occ,
              local_alpha_col_full_mask ^ used_alpha_col_mask);
          const auto alpha_frontier_interface_rows = select_occ_by_mask(
              right_node_term.alpha_occ,
              used_alpha_row_mask);
          const auto alpha_frontier_interface_cols = select_occ_by_mask(
              left_node_term.alpha_occ,
              used_alpha_col_mask);
          const auto beta_local_remainder_rows = select_occ_by_mask(
              right_node_term.beta_occ,
              local_beta_row_full_mask ^ used_beta_row_mask);
          const auto beta_local_remainder_cols = select_occ_by_mask(
              left_node_term.beta_occ,
              local_beta_col_full_mask ^ used_beta_col_mask);
          const auto beta_frontier_interface_rows = select_occ_by_mask(
              right_node_term.beta_occ,
              used_beta_row_mask);
          const auto beta_frontier_interface_cols = select_occ_by_mask(
              left_node_term.beta_occ,
              used_beta_col_mask);

          const SpinDeletionPayload alpha_interface_payload =
              build_exact_frontier_spin_payload_limited(
                  selected_parent_alpha_cols,
                  alpha_local_remainder_cols,
                  selected_parent_alpha_rows,
                  alpha_local_remainder_rows,
                  2,
                  overlap_storage,
                  support_size,
                  overlap_resolver,
                  subdeterminant_evaluations,
                  payload_cache);
          const SpinDeletionPayload beta_interface_payload =
              build_exact_frontier_spin_payload_limited(
                  selected_parent_beta_cols,
                  beta_local_remainder_cols,
                  selected_parent_beta_rows,
                  beta_local_remainder_rows,
                  2,
                  overlap_storage,
                  support_size,
                  overlap_resolver,
                  subdeterminant_evaluations,
                  payload_cache);
          const HamiltonianEntryDensePayload interface_payload =
              build_hamiltonian_entry_dense_payload_from_projected_spin_values(
                  project_spin_payload_to_hamiltonian_values(alpha_interface_payload),
                  project_spin_payload_to_hamiltonian_values(beta_interface_payload),
                  support_size);
          HamiltonianEntryDensePayload natural_payload =
              merge_hamiltonian_entry_dense_payloads_limited_values(
                  frontier_entry.payload,
                  interface_payload);
          HamiltonianEntryDensePayload front_convention_payload =
              transform_hamiltonian_entry_dense_payload_interface_block_to_front_values(
                  natural_payload,
                  static_cast<int>(selected_parent_alpha_rows.size()),
                  static_cast<int>(selected_parent_alpha_cols.size()),
                  alpha_frontier_interface_rows,
                  alpha_frontier_interface_cols,
                  alpha_local_remainder_rows,
                  alpha_local_remainder_cols,
                  frontier_entry.payload.alpha_row_count,
                  frontier_entry.payload.alpha_col_count,
                  static_cast<int>(selected_parent_beta_rows.size()),
                  static_cast<int>(selected_parent_beta_cols.size()),
                  beta_frontier_interface_rows,
                  beta_frontier_interface_cols,
                  beta_local_remainder_rows,
                  beta_local_remainder_cols,
                  frontier_entry.payload.beta_row_count,
                  frontier_entry.payload.beta_col_count);
          if (!is_hamiltonian_entry_dense_payload_nonzero(front_convention_payload)) {
            continue;
          }

          add_scaled_hamiltonian_entry_dense_payload(
              front_convention_payload,
              local_coefficient,
              &message.payload_values[xmvb::to_size(
                  message.flat_index(alpha_flat_sector_index, beta_flat_sector_index))]);
        }
      }
    }
  }
  finalize_dense_boundary_hamiltonian_message(&message);
  validate_dense_boundary_hamiltonian_message_against_typed_reference(
      tree,
      subtree_expansions,
      node,
      left_parent_term,
      right_parent_term,
      overlap_storage,
      support_size,
      overlap_resolver,
      message);
  *subtree_message_state_count +=
      static_cast<std::uint64_t>(message.nonzero_entries.size());
  return message_cache->emplace(cache_key, std::move(message)).first->second;
}

const BoundaryHamiltonianMessage& build_recursive_subtree_hamiltonian_messages_cached(
    const ComponentTree& tree,
    const std::vector<SubtreeExpansion>& subtree_expansions,
    int node,
    const OrientationTerm& left_parent_term,
    const OrientationTerm& right_parent_term,
    bool preserve_structural_zero_messages,
    const std::vector<double>& overlap_storage,
    int n_orbitals,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subtree_term_pair_count,
    std::uint64_t* subtree_message_state_count,
    std::uint64_t* subdeterminant_evaluations,
    std::uint64_t* dp_transition_count,
    std::unordered_map<SpinPayloadCacheKey,
                       SpinDeletionPayload,
                       SpinPayloadCacheKeyHasher>* payload_cache,
    std::unordered_map<SubtreeMessageCacheKey,
                       BoundaryHamiltonianMessage,
                       SubtreeMessageCacheKeyHasher>* message_cache) {
  // Exact recursive node->parent boundary message specialized for the full
  // Hamiltonian channel set:
  // - overlap,
  // - one-electron,
  // - same-spin degree-2 sectors,
  // - opposite-spin mixed degree-1 sectors.
  //
  // This is the typed boundary-message analogue of the old hamiltonian
  // `JointDeletionPayload`, with overlap, single-spin active channels, and
  // mixed alpha-beta first-order channels stored explicitly.
  if (subtree_term_pair_count == nullptr ||
      subtree_message_state_count == nullptr ||
      subdeterminant_evaluations == nullptr ||
      dp_transition_count == nullptr ||
      payload_cache == nullptr ||
      message_cache == nullptr) {
    throw std::invalid_argument("hamiltonian recursive subtree-message inputs must not be null");
  }

  const SubtreeMessageCacheKey cache_key{
      node,
      left_parent_term.alpha_occ,
      left_parent_term.beta_occ,
      right_parent_term.alpha_occ,
      right_parent_term.beta_occ,
      preserve_structural_zero_messages,
  };
  auto cache_iterator = message_cache->find(cache_key);
  if (cache_iterator != message_cache->end()) {
    return cache_iterator->second;
  }

  const ComponentData& node_component =
      tree.components[xmvb::to_size(node)];
  const std::vector<int>& children = tree.children[xmvb::to_size(node)];
  const ComponentSpinSizes& subtree_sizes =
      subtree_expansions[xmvb::to_size(node)].spin_sizes;
  const auto left_parent_alpha_occ_by_mask = build_mask_occ_table(left_parent_term.alpha_occ);
  const auto right_parent_alpha_occ_by_mask = build_mask_occ_table(right_parent_term.alpha_occ);
  const auto left_parent_beta_occ_by_mask = build_mask_occ_table(left_parent_term.beta_occ);
  const auto right_parent_beta_occ_by_mask = build_mask_occ_table(right_parent_term.beta_occ);

  const ChildSpinSizeVectors child_spin_sizes =
      build_child_spin_size_vectors(children, subtree_expansions);

  BoundaryHamiltonianMessage message;
  message.alpha_layout = BoundarySpinBundleLayout(
      static_cast<int>(right_parent_term.alpha_occ.size()),
      static_cast<int>(left_parent_term.alpha_occ.size()),
      subtree_sizes.left_alpha - subtree_sizes.right_alpha,
      -2,
      2,
      n_orbitals);
  message.beta_layout = BoundarySpinBundleLayout(
      static_cast<int>(right_parent_term.beta_occ.size()),
      static_cast<int>(left_parent_term.beta_occ.size()),
      subtree_sizes.left_beta - subtree_sizes.right_beta,
      -2,
      2,
      n_orbitals);
  message.payload_values.resize(
      xmvb::to_size(message.alpha_layout.total_sector_count()) *
      xmvb::to_size(message.beta_layout.total_sector_count()));

  if (children.empty() && !preserve_structural_zero_messages) {
    message = build_leaf_hamiltonian_subtree_message_values(
        node_component,
        message.alpha_layout,
        message.beta_layout,
        left_parent_alpha_occ_by_mask,
        right_parent_alpha_occ_by_mask,
        left_parent_beta_occ_by_mask,
        right_parent_beta_occ_by_mask,
        overlap_storage,
        n_orbitals,
        overlap_resolver,
        subtree_term_pair_count,
        subtree_message_state_count,
        subdeterminant_evaluations,
        payload_cache);
    return message_cache->emplace(cache_key, std::move(message)).first->second;
  }

  const std::vector<WeightedOrientationTermPair> node_term_pairs =
      build_weighted_component_orientation_pairs(node_component);
  for (const auto& node_term_pair : node_term_pairs) {
    const OrientationTerm& left_node_term = node_term_pair.left_term;
    const OrientationTerm& right_node_term = node_term_pair.right_term;
    const double local_coefficient = node_term_pair.coefficient;
    ++(*subtree_term_pair_count);

    std::vector<const BoundaryHamiltonianMessage*> child_messages(
        children.size(),
        nullptr);
    for (std::size_t child_index = 0; child_index < children.size(); ++child_index) {
      const int child = children[child_index];
      child_messages[child_index] =
          &build_recursive_subtree_hamiltonian_messages_cached(
              tree,
              subtree_expansions,
              child,
              left_node_term,
              right_node_term,
              preserve_structural_zero_messages,
              overlap_storage,
              n_orbitals,
              overlap_resolver,
              subtree_term_pair_count,
              subtree_message_state_count,
              subdeterminant_evaluations,
              dp_transition_count,
              payload_cache,
              message_cache);
    }

    const FrontierHamiltonianMessage frontier_message =
        build_frontier_hamiltonian_message(
            child_messages,
            child_spin_sizes,
            static_cast<int>(right_node_term.alpha_occ.size()),
            static_cast<int>(left_node_term.alpha_occ.size()),
            static_cast<int>(right_node_term.beta_occ.size()),
            static_cast<int>(left_node_term.beta_occ.size()),
            preserve_structural_zero_messages,
            dp_transition_count);

    const std::uint32_t local_alpha_row_full_mask =
        right_node_term.alpha_occ.empty()
            ? 0U
            : (open_state_mask_limit(static_cast<int>(right_node_term.alpha_occ.size())) - 1U);
    const std::uint32_t local_alpha_col_full_mask =
        left_node_term.alpha_occ.empty()
            ? 0U
            : (open_state_mask_limit(static_cast<int>(left_node_term.alpha_occ.size())) - 1U);
    const std::uint32_t local_beta_row_full_mask =
        right_node_term.beta_occ.empty()
            ? 0U
            : (open_state_mask_limit(static_cast<int>(right_node_term.beta_occ.size())) - 1U);
    const std::uint32_t local_beta_col_full_mask =
        left_node_term.beta_occ.empty()
            ? 0U
            : (open_state_mask_limit(static_cast<int>(left_node_term.beta_occ.size())) - 1U);

    for (int alpha_flat_sector_index = 0;
         alpha_flat_sector_index < message.alpha_layout.total_sector_count();
         ++alpha_flat_sector_index) {
      const BoundarySector& alpha_sector =
          boundary_layout_sector(message.alpha_layout, alpha_flat_sector_index);
      const auto& selected_parent_alpha_rows =
          right_parent_alpha_occ_by_mask[xmvb::to_size(alpha_sector.row_mask)];
      const auto& selected_parent_alpha_cols =
          left_parent_alpha_occ_by_mask[xmvb::to_size(alpha_sector.col_mask)];

      for (int beta_flat_sector_index = 0;
           beta_flat_sector_index < message.beta_layout.total_sector_count();
           ++beta_flat_sector_index) {
        const BoundarySector& beta_sector =
            boundary_layout_sector(message.beta_layout, beta_flat_sector_index);
        const auto& selected_parent_beta_rows =
            right_parent_beta_occ_by_mask[xmvb::to_size(beta_sector.row_mask)];
        const auto& selected_parent_beta_cols =
            left_parent_beta_occ_by_mask[xmvb::to_size(beta_sector.col_mask)];
        for (const FrontierHamiltonianEntry& frontier_entry : frontier_message.entries) {
          const std::uint32_t used_alpha_row_mask =
              frontier_entry.key.alpha_row_mask;
          const std::uint32_t used_alpha_col_mask =
              frontier_entry.key.alpha_col_mask;
          const std::uint32_t used_beta_row_mask =
              frontier_entry.key.beta_row_mask;
          const std::uint32_t used_beta_col_mask =
              frontier_entry.key.beta_col_mask;
          const auto alpha_local_remainder_rows = select_occ_by_mask(
              right_node_term.alpha_occ,
              local_alpha_row_full_mask ^ used_alpha_row_mask);
          const auto alpha_local_remainder_cols = select_occ_by_mask(
              left_node_term.alpha_occ,
              local_alpha_col_full_mask ^ used_alpha_col_mask);
          const auto alpha_frontier_interface_rows = select_occ_by_mask(
              right_node_term.alpha_occ,
              used_alpha_row_mask);
          const auto alpha_frontier_interface_cols = select_occ_by_mask(
              left_node_term.alpha_occ,
              used_alpha_col_mask);
          const auto beta_local_remainder_rows = select_occ_by_mask(
              right_node_term.beta_occ,
              local_beta_row_full_mask ^ used_beta_row_mask);
          const auto beta_local_remainder_cols = select_occ_by_mask(
              left_node_term.beta_occ,
              local_beta_col_full_mask ^ used_beta_col_mask);
          const auto beta_frontier_interface_rows = select_occ_by_mask(
              right_node_term.beta_occ,
              used_beta_row_mask);
          const auto beta_frontier_interface_cols = select_occ_by_mask(
              left_node_term.beta_occ,
              used_beta_col_mask);

          const SpinDeletionPayload alpha_interface_payload =
              build_exact_frontier_spin_payload_limited(
                  selected_parent_alpha_cols,
                  alpha_local_remainder_cols,
                  selected_parent_alpha_rows,
                  alpha_local_remainder_rows,
                  2,
                  overlap_storage,
                  n_orbitals,
                  overlap_resolver,
                  subdeterminant_evaluations,
                  payload_cache);
          const SpinDeletionPayload beta_interface_payload =
              build_exact_frontier_spin_payload_limited(
                  selected_parent_beta_cols,
                  beta_local_remainder_cols,
                  selected_parent_beta_rows,
                  beta_local_remainder_rows,
                  2,
                  overlap_storage,
                  n_orbitals,
                  overlap_resolver,
                  subdeterminant_evaluations,
                  payload_cache);
          const HamiltonianBoundaryPayload interface_payload =
              preserve_structural_zero_messages
                  ? build_hamiltonian_boundary_payload(
                        alpha_interface_payload,
                        beta_interface_payload)
                  : build_hamiltonian_boundary_payload_values(
                        alpha_interface_payload,
                        beta_interface_payload);
          HamiltonianBoundaryPayload natural_payload =
              preserve_structural_zero_messages
                  ? merge_hamiltonian_boundary_payloads_limited(
                        frontier_entry.payload,
                        interface_payload,
                        2)
                  : merge_hamiltonian_boundary_payloads_limited_values(
                        frontier_entry.payload,
                        interface_payload,
                        2);
          HamiltonianBoundaryPayload front_convention_payload =
              preserve_structural_zero_messages
                  ? transform_hamiltonian_interface_block_to_front(
                        natural_payload,
                        static_cast<int>(selected_parent_alpha_rows.size()),
                        static_cast<int>(selected_parent_alpha_cols.size()),
                        alpha_frontier_interface_rows,
                        alpha_frontier_interface_cols,
                        alpha_local_remainder_rows,
                        alpha_local_remainder_cols,
                        frontier_entry.payload.alpha_row_count,
                        frontier_entry.payload.alpha_col_count,
                        static_cast<int>(selected_parent_beta_rows.size()),
                        static_cast<int>(selected_parent_beta_cols.size()),
                        beta_frontier_interface_rows,
                        beta_frontier_interface_cols,
                        beta_local_remainder_rows,
                        beta_local_remainder_cols,
                        frontier_entry.payload.beta_row_count,
                        frontier_entry.payload.beta_col_count)
                  : transform_hamiltonian_interface_block_to_front_values(
                        natural_payload,
                        static_cast<int>(selected_parent_alpha_rows.size()),
                        static_cast<int>(selected_parent_alpha_cols.size()),
                        alpha_frontier_interface_rows,
                        alpha_frontier_interface_cols,
                        alpha_local_remainder_rows,
                        alpha_local_remainder_cols,
                        frontier_entry.payload.alpha_row_count,
                        frontier_entry.payload.alpha_col_count,
                        static_cast<int>(selected_parent_beta_rows.size()),
                        static_cast<int>(selected_parent_beta_cols.size()),
                        beta_frontier_interface_rows,
                        beta_frontier_interface_cols,
                        beta_local_remainder_rows,
                        beta_local_remainder_cols,
                        frontier_entry.payload.beta_row_count,
                        frontier_entry.payload.beta_col_count);
          if ((!preserve_structural_zero_messages &&
               !is_hamiltonian_payload_nonzero(front_convention_payload)) ||
              (preserve_structural_zero_messages &&
               !has_hamiltonian_payload_basis(front_convention_payload))) {
            continue;
          }

          add_scaled_hamiltonian_payload_preserve_basis(
              front_convention_payload,
              local_coefficient,
              preserve_structural_zero_messages,
              &message.payload_values[xmvb::to_size(
                  message.flat_index(alpha_flat_sector_index, beta_flat_sector_index))]);
        }
      }
    }
  }
  finalize_boundary_hamiltonian_message(&message, preserve_structural_zero_messages);
  *subtree_message_state_count +=
      static_cast<std::uint64_t>(message.nonzero_entries.size());
  return message_cache->emplace(cache_key, std::move(message)).first->second;
}

ComponentTreeHamiltonianResult
evaluate_rooted_component_tree_hamiltonian_bundle_collapsed_forward_dense_internal(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver,
    Eigen::MatrixXd* alpha_first_cofactor,
    Eigen::MatrixXd* beta_first_cofactor,
    Eigen::MatrixXd* active_one_electron_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (alpha_first_cofactor != nullptr) {
    *alpha_first_cofactor = Eigen::MatrixXd::Zero(support_size, support_size);
  }
  if (beta_first_cofactor != nullptr) {
    *beta_first_cofactor = Eigen::MatrixXd::Zero(support_size, support_size);
  }
  if (active_one_electron_gradient != nullptr) {
    *active_one_electron_gradient = Eigen::MatrixXd::Zero(support_size, support_size);
  }
  if (packed_active_two_electron_gradient != nullptr) {
    packed_active_two_electron_gradient->assign(
        packed_active_two_electron_integrals.size(),
        0.0);
  }

  const std::vector<SubtreeExpansion> subtree_expansions =
      build_subtree_expansions(tree);
  const ComponentData& root_component =
      tree.components[xmvb::to_size(tree.root_index)];
  const std::vector<int>& root_children =
      tree.children[xmvb::to_size(tree.root_index)];
  const ChildSpinSizeVectors child_spin_sizes =
      build_child_spin_size_vectors(root_children, subtree_expansions);

  ComponentTreeHamiltonianResult result;
  std::unordered_map<SpinPayloadCacheKey,
                     SpinDeletionPayload,
                     SpinPayloadCacheKeyHasher> spin_payload_cache;
  std::unordered_map<SubtreeMessageCacheKey,
                     DenseBoundaryHamiltonianMessage,
                     SubtreeMessageCacheKeyHasher> subtree_message_cache;
  const std::vector<WeightedOrientationTermPair> root_term_pairs =
      build_weighted_component_orientation_pairs(root_component);
  for (const auto& root_term_pair : root_term_pairs) {
    const OrientationTerm& left_root_term = root_term_pair.left_term;
    const OrientationTerm& right_root_term = root_term_pair.right_term;
    const double root_coefficient = root_term_pair.coefficient;
    ++result.subtree_term_pair_count;

    std::vector<const DenseBoundaryHamiltonianMessage*> child_messages(
        root_children.size(),
        nullptr);
    for (std::size_t child_index = 0; child_index < root_children.size();
         ++child_index) {
      const int child = root_children[child_index];
      child_messages[child_index] =
          &build_recursive_subtree_dense_hamiltonian_messages_cached(
              tree,
              subtree_expansions,
              child,
              left_root_term,
              right_root_term,
              support_overlap_storage,
              support_size,
              overlap_resolver,
              &result.subtree_term_pair_count,
              &result.subtree_message_state_count,
              &result.subdeterminant_evaluations,
              &result.dp_transition_count,
              &spin_payload_cache,
              &subtree_message_cache);
    }

    const DenseFrontierHamiltonianMessage frontier_message =
        build_frontier_dense_hamiltonian_message(
            child_messages,
            child_spin_sizes,
            static_cast<int>(right_root_term.alpha_occ.size()),
            static_cast<int>(left_root_term.alpha_occ.size()),
            static_cast<int>(right_root_term.beta_occ.size()),
            static_cast<int>(left_root_term.beta_occ.size()),
            support_size,
            &result.dp_transition_count);

    std::map<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>,
             HamiltonianEntryDensePayload>
        root_payload_cache;
    for (const DenseFrontierHamiltonianEntry& frontier_entry : frontier_message.entries) {
      if (!is_hamiltonian_entry_dense_payload_nonzero(frontier_entry.payload)) {
        continue;
      }
      const auto cache_key = std::make_tuple(
          frontier_entry.key.alpha_row_mask,
          frontier_entry.key.alpha_col_mask,
          frontier_entry.key.beta_row_mask,
          frontier_entry.key.beta_col_mask);
      auto cache_iterator = root_payload_cache.find(cache_key);
      if (cache_iterator == root_payload_cache.end()) {
        const SpinDeletionPayload alpha_root_payload =
            build_root_spin_payload_limited(
                left_root_term.alpha_occ,
                right_root_term.alpha_occ,
                frontier_entry.key.alpha_row_mask,
                frontier_entry.key.alpha_col_mask,
                2,
                support_overlap_storage,
                support_size,
                overlap_resolver,
                &result.subdeterminant_evaluations,
                &spin_payload_cache);
        const SpinDeletionPayload beta_root_payload =
            build_root_spin_payload_limited(
                left_root_term.beta_occ,
                right_root_term.beta_occ,
                frontier_entry.key.beta_row_mask,
                frontier_entry.key.beta_col_mask,
                2,
                support_overlap_storage,
                support_size,
                overlap_resolver,
                &result.subdeterminant_evaluations,
                &spin_payload_cache);
        cache_iterator = root_payload_cache.emplace(
            cache_key,
            build_hamiltonian_entry_dense_payload_from_projected_spin_values(
                project_spin_payload_to_hamiltonian_values(alpha_root_payload),
                project_spin_payload_to_hamiltonian_values(beta_root_payload),
                support_size))
                             .first;
      }
      const HamiltonianEntryDensePayload& root_payload = cache_iterator->second;
      if (!is_hamiltonian_entry_dense_payload_nonzero(root_payload)) {
        continue;
      }

      const HamiltonianEntryDensePayload total_payload =
          close_root_hamiltonian_entry_dense_payload(
              frontier_entry.payload,
              root_payload,
              left_root_term,
              right_root_term,
              frontier_entry.key.alpha_row_mask,
              frontier_entry.key.alpha_col_mask,
              frontier_entry.key.beta_row_mask,
              frontier_entry.key.beta_col_mask);
      if (!is_hamiltonian_entry_dense_payload_nonzero(total_payload)) {
        continue;
      }

      const double signed_coefficient = root_coefficient;
      result.overlap += signed_coefficient * total_payload.overlap;
      result.one_electron +=
          signed_coefficient *
          (contract_dense_spin_one_electron_first_sectors(
               total_payload.alpha_degree1,
               total_payload.alpha_row_count,
               total_payload.alpha_col_count,
               support_size,
               support_one_electron_storage) +
           contract_dense_spin_one_electron_first_sectors(
               total_payload.beta_degree1,
               total_payload.beta_row_count,
               total_payload.beta_col_count,
               support_size,
               support_one_electron_storage));
      result.same_spin_alpha_two_electron +=
          signed_coefficient *
          contract_dense_same_spin_second_sectors(
              total_payload.alpha_degree2,
              total_payload.alpha_row_count,
              total_payload.alpha_col_count,
              support_size,
              packed_active_two_electron_integrals);
      result.same_spin_beta_two_electron +=
          signed_coefficient *
          contract_dense_same_spin_second_sectors(
              total_payload.beta_degree2,
              total_payload.beta_row_count,
              total_payload.beta_col_count,
              support_size,
              packed_active_two_electron_integrals);
      result.opposite_spin_two_electron +=
          signed_coefficient *
          contract_dense_opposite_spin_first_sectors(
              total_payload,
              packed_active_two_electron_integrals);
      if (alpha_first_cofactor != nullptr) {
        accumulate_dense_spin_first_sectors(
            total_payload.alpha_degree1,
            total_payload.alpha_row_count,
            total_payload.alpha_col_count,
            support_size,
            signed_coefficient,
            alpha_first_cofactor);
      }
      if (beta_first_cofactor != nullptr) {
        accumulate_dense_spin_first_sectors(
            total_payload.beta_degree1,
            total_payload.beta_row_count,
            total_payload.beta_col_count,
            support_size,
            signed_coefficient,
            beta_first_cofactor);
      }
      if (active_one_electron_gradient != nullptr) {
        accumulate_dense_spin_first_sectors(
            total_payload.alpha_degree1,
            total_payload.alpha_row_count,
            total_payload.alpha_col_count,
            support_size,
            signed_coefficient,
            active_one_electron_gradient);
        accumulate_dense_spin_first_sectors(
            total_payload.beta_degree1,
            total_payload.beta_row_count,
            total_payload.beta_col_count,
            support_size,
            signed_coefficient,
            active_one_electron_gradient);
      }
      if (packed_active_two_electron_gradient != nullptr) {
        accumulate_dense_same_spin_second_sector_gradient(
            total_payload.alpha_degree2,
            total_payload.alpha_row_count,
            total_payload.alpha_col_count,
            support_size,
            signed_coefficient,
            packed_active_two_electron_gradient);
        accumulate_dense_same_spin_second_sector_gradient(
            total_payload.beta_degree2,
            total_payload.beta_row_count,
            total_payload.beta_col_count,
            support_size,
            signed_coefficient,
            packed_active_two_electron_gradient);
        accumulate_dense_opposite_spin_first_sector_gradient(
            total_payload,
            signed_coefficient,
            packed_active_two_electron_gradient);
      }
    }
  }

  result.two_electron =
      result.same_spin_alpha_two_electron +
      result.same_spin_beta_two_electron +
      result.opposite_spin_two_electron;
  result.total_electronic_hamiltonian =
      result.one_electron + result.two_electron;
  return result;
}

ComponentTreeHamiltonianResult
evaluate_rooted_component_tree_hamiltonian_bundle_collapsed_internal(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver,
    Eigen::MatrixXd* alpha_first_cofactor,
    Eigen::MatrixXd* beta_first_cofactor,
    Eigen::MatrixXd* active_one_electron_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  // Exact rooted-tree Hamiltonian closure from the typed boundary bundle.
  //
  // Each subtree message remains boundary-only, but the carried polynomial
  // payload is the explicit Hamiltonian bundle:
  // - closed overlap,
  // - one-spin degree-<= 2 deleted-minor families,
  // - mixed alpha/beta degree-1 moments.
  // This is the family-complete recursive payload we want the generic rooted
  // tree path to use instead of the older generic joint deleted-minor map.
  if (alpha_first_cofactor != nullptr) {
    *alpha_first_cofactor = Eigen::MatrixXd::Zero(support_size, support_size);
  }
  if (beta_first_cofactor != nullptr) {
    *beta_first_cofactor = Eigen::MatrixXd::Zero(support_size, support_size);
  }
  if (active_one_electron_gradient != nullptr) {
    *active_one_electron_gradient = Eigen::MatrixXd::Zero(support_size, support_size);
  }
  if (packed_active_two_electron_gradient != nullptr) {
    packed_active_two_electron_gradient->assign(
        packed_active_two_electron_integrals.size(),
        0.0);
  }

  const std::vector<SubtreeExpansion> subtree_expansions =
      build_subtree_expansions(tree);
  const ComponentData& root_component =
      tree.components[xmvb::to_size(tree.root_index)];
  const std::vector<int>& root_children =
      tree.children[xmvb::to_size(tree.root_index)];

  const ChildSpinSizeVectors child_spin_sizes =
      build_child_spin_size_vectors(root_children, subtree_expansions);

  ComponentTreeHamiltonianResult result;
  std::unordered_map<SpinPayloadCacheKey,
                     SpinDeletionPayload,
                     SpinPayloadCacheKeyHasher> spin_payload_cache;
  std::unordered_map<SubtreeMessageCacheKey,
                     BoundaryHamiltonianMessage,
                     SubtreeMessageCacheKeyHasher> subtree_message_cache;
  const std::vector<WeightedOrientationTermPair> root_term_pairs =
      build_weighted_component_orientation_pairs(root_component);
  for (const auto& root_term_pair : root_term_pairs) {
    // Root closure is not an ordinary append merge.
    // The recursive frontier payload is already in
    //   [selected-root-interface, frontier-body]
    // convention, while the local root payload is the remaining root block.
    // `close_root_hamiltonian_payload(...)` inserts that root remainder between
    // the selected root interface and the frontier body before the final
    // Hamiltonian contractions.
    const OrientationTerm& left_root_term = root_term_pair.left_term;
    const OrientationTerm& right_root_term = root_term_pair.right_term;
    const double root_coefficient = root_term_pair.coefficient;
    ++result.subtree_term_pair_count;

    std::vector<const BoundaryHamiltonianMessage*> child_messages(
        root_children.size(),
        nullptr);
    for (std::size_t child_index = 0; child_index < root_children.size();
         ++child_index) {
      const int child = root_children[child_index];
      child_messages[child_index] =
          &build_recursive_subtree_hamiltonian_messages_cached(
              tree,
              subtree_expansions,
              child,
              left_root_term,
              right_root_term,
              false,
              support_overlap_storage,
              support_size,
              overlap_resolver,
              &result.subtree_term_pair_count,
              &result.subtree_message_state_count,
              &result.subdeterminant_evaluations,
              &result.dp_transition_count,
              &spin_payload_cache,
              &subtree_message_cache);
    }

    const FrontierHamiltonianMessage frontier_message =
        build_frontier_hamiltonian_message(
            child_messages,
            child_spin_sizes,
            static_cast<int>(right_root_term.alpha_occ.size()),
            static_cast<int>(left_root_term.alpha_occ.size()),
            static_cast<int>(right_root_term.beta_occ.size()),
            static_cast<int>(left_root_term.beta_occ.size()),
            false,
            &result.dp_transition_count);

    std::map<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>,
             HamiltonianBoundaryPayload>
        root_payload_cache;
    for (const FrontierHamiltonianEntry& frontier_entry : frontier_message.entries) {
      if (!is_hamiltonian_payload_nonzero(frontier_entry.payload)) {
        continue;
      }
      const auto cache_key = std::make_tuple(
          frontier_entry.key.alpha_row_mask,
          frontier_entry.key.alpha_col_mask,
          frontier_entry.key.beta_row_mask,
          frontier_entry.key.beta_col_mask);
      auto cache_iterator = root_payload_cache.find(cache_key);
      if (cache_iterator == root_payload_cache.end()) {
        cache_iterator = root_payload_cache.emplace(
            cache_key,
            build_root_hamiltonian_boundary_payload(
                left_root_term,
                right_root_term,
                frontier_entry.key.alpha_row_mask,
                frontier_entry.key.alpha_col_mask,
                frontier_entry.key.beta_row_mask,
                frontier_entry.key.beta_col_mask,
                support_overlap_storage,
                support_size,
                overlap_resolver,
                &result.subdeterminant_evaluations,
                &spin_payload_cache))
                             .first;
      }
      const HamiltonianBoundaryPayload& root_payload = cache_iterator->second;
      if (!is_hamiltonian_payload_nonzero(root_payload)) {
        continue;
      }

      const HamiltonianBoundaryPayload total_payload =
          close_root_hamiltonian_payload(
              frontier_entry.payload,
              root_payload,
              left_root_term,
              right_root_term,
              frontier_entry.key.alpha_row_mask,
              frontier_entry.key.alpha_col_mask,
              frontier_entry.key.beta_row_mask,
              frontier_entry.key.beta_col_mask);
      if (!is_hamiltonian_payload_nonzero(total_payload)) {
        continue;
      }

      const double signed_coefficient = root_coefficient;
      result.overlap +=
          signed_coefficient * overlap_sector_value(total_payload);
      result.one_electron +=
          signed_coefficient *
          contract_total_one_electron_first_sectors(
              total_payload,
              support_one_electron_storage,
              support_size);
      result.same_spin_alpha_two_electron +=
          signed_coefficient *
          contract_same_spin_second_sectors(
              total_payload,
              true,
              packed_active_two_electron_integrals);
      result.same_spin_beta_two_electron +=
          signed_coefficient *
          contract_same_spin_second_sectors(
              total_payload,
              false,
              packed_active_two_electron_integrals);
      result.opposite_spin_two_electron +=
          signed_coefficient *
          contract_opposite_spin_first_sectors(
              total_payload,
              packed_active_two_electron_integrals);
      if (alpha_first_cofactor != nullptr) {
        accumulate_spin_first_sectors(
            total_payload,
            true,
            signed_coefficient,
            alpha_first_cofactor);
      }
      if (beta_first_cofactor != nullptr) {
        accumulate_spin_first_sectors(
            total_payload,
            false,
            signed_coefficient,
            beta_first_cofactor);
      }
      if (active_one_electron_gradient != nullptr) {
        accumulate_spin_first_sectors(
            total_payload,
            true,
            signed_coefficient,
            active_one_electron_gradient);
        accumulate_spin_first_sectors(
            total_payload,
            false,
            signed_coefficient,
            active_one_electron_gradient);
      }
      if (packed_active_two_electron_gradient != nullptr) {
        accumulate_same_spin_second_sector_gradient(
            total_payload,
            true,
            signed_coefficient,
            packed_active_two_electron_gradient);
        accumulate_same_spin_second_sector_gradient(
            total_payload,
            false,
            signed_coefficient,
            packed_active_two_electron_gradient);
        accumulate_opposite_spin_first_sector_gradient(
            total_payload,
            signed_coefficient,
            packed_active_two_electron_gradient);
      }
    }
  }

  result.two_electron =
      result.same_spin_alpha_two_electron +
      result.same_spin_beta_two_electron +
      result.opposite_spin_two_electron;
  result.total_electronic_hamiltonian =
      result.one_electron + result.two_electron;
  return result;
}

ComponentTreeHamiltonianResult
evaluate_rooted_component_tree_hamiltonian_bundle_collapsed_production_internal(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver,
    Eigen::MatrixXd* alpha_first_cofactor,
    Eigen::MatrixXd* beta_first_cofactor,
    Eigen::MatrixXd* active_one_electron_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  // Production Hamiltonian wrappers now use the dense forward bundle path by
  // default. The older typed internal remains available behind an environment
  // fallback while the pre-existing branched-tree generic correctness issue is
  // still being debugged.
  if (typed_hamiltonian_production_fallback_enabled()) {
    return evaluate_rooted_component_tree_hamiltonian_bundle_collapsed_internal(
        support_overlap_storage,
        support_one_electron_storage,
        packed_active_two_electron_integrals,
        support_size,
        tree,
        overlap_resolver,
        alpha_first_cofactor,
        beta_first_cofactor,
        active_one_electron_gradient,
        packed_active_two_electron_gradient);
  }
  return evaluate_rooted_component_tree_hamiltonian_bundle_collapsed_forward_dense_internal(
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      support_size,
      tree,
      overlap_resolver,
      alpha_first_cofactor,
      beta_first_cofactor,
      active_one_electron_gradient,
      packed_active_two_electron_gradient);
}

}  // namespace

void validate_rooted_component_tree_hamiltonian_inputs(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    int support_size,
    const ComponentTree& tree) {
  if (support_size <= 0) {
    throw std::invalid_argument("support_size must be positive");
  }
  if (support_overlap_storage.size() !=
      xmvb::to_size(support_size * support_size)) {
    throw std::invalid_argument(
        "support_overlap_storage must hold a square support-space matrix");
  }
  if (support_one_electron_storage.size() !=
      xmvb::to_size(support_size * support_size)) {
    throw std::invalid_argument(
        "support_one_electron_storage must hold a square support-space matrix");
  }
  validate_tree(tree);
}

void validate_rooted_component_tree_overlap_inputs(
    const std::vector<double>& support_overlap_storage,
    int support_size,
    const ComponentTree& tree) {
  if (support_size <= 0) {
    throw std::invalid_argument("support_size must be positive");
  }
  if (support_overlap_storage.size() !=
      xmvb::to_size(support_size * support_size)) {
    throw std::invalid_argument(
        "support_overlap_storage must hold a square support-space matrix");
  }
  validate_tree(tree);
}

CollapsedHamiltonianComponentTreeStats build_hamiltonian_comparison_stats(
    const ComponentTreeHamiltonianResult& exact,
    const ComponentTreeHamiltonianResult& collapsed) {
  CollapsedHamiltonianComponentTreeStats stats;
  stats.exact_overlap = exact.overlap;
  stats.collapsed_overlap = collapsed.overlap;
  stats.overlap_absolute_error = std::abs(collapsed.overlap - exact.overlap);
  stats.exact_one_electron = exact.one_electron;
  stats.collapsed_one_electron = collapsed.one_electron;
  stats.one_electron_absolute_error =
      std::abs(collapsed.one_electron - exact.one_electron);
  stats.exact_same_spin_alpha_two_electron = exact.same_spin_alpha_two_electron;
  stats.collapsed_same_spin_alpha_two_electron =
      collapsed.same_spin_alpha_two_electron;
  stats.same_spin_alpha_absolute_error = std::abs(
      collapsed.same_spin_alpha_two_electron -
      exact.same_spin_alpha_two_electron);
  stats.exact_same_spin_beta_two_electron = exact.same_spin_beta_two_electron;
  stats.collapsed_same_spin_beta_two_electron =
      collapsed.same_spin_beta_two_electron;
  stats.same_spin_beta_absolute_error = std::abs(
      collapsed.same_spin_beta_two_electron -
      exact.same_spin_beta_two_electron);
  stats.exact_opposite_spin_two_electron = exact.opposite_spin_two_electron;
  stats.collapsed_opposite_spin_two_electron =
      collapsed.opposite_spin_two_electron;
  stats.opposite_spin_absolute_error = std::abs(
      collapsed.opposite_spin_two_electron -
      exact.opposite_spin_two_electron);
  stats.exact_two_electron = exact.two_electron;
  stats.collapsed_two_electron = collapsed.two_electron;
  stats.total_two_electron_absolute_error =
      std::abs(collapsed.two_electron - exact.two_electron);
  stats.exact_total_electronic_hamiltonian = exact.total_electronic_hamiltonian;
  stats.collapsed_total_electronic_hamiltonian =
      collapsed.total_electronic_hamiltonian;
  stats.total_electronic_hamiltonian_absolute_error = std::abs(
      collapsed.total_electronic_hamiltonian -
      exact.total_electronic_hamiltonian);
  stats.subtree_message_state_count = collapsed.subtree_message_state_count;
  stats.subtree_term_pair_count = collapsed.subtree_term_pair_count;
  stats.subdeterminant_evaluations = collapsed.subdeterminant_evaluations;
  stats.dp_transition_count = collapsed.dp_transition_count;
  return stats;
}

ComponentTreeOverlapResult evaluate_component_tree_overlap_exact_reference(
    const std::vector<double>& support_overlap_storage,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const DeterminantOverlapResolver& overlap_resolver) {
  if (support_size <= 0) {
    throw std::invalid_argument("support_size must be positive");
  }
  if (ordered_components.empty()) {
    throw std::invalid_argument("ordered_components must not be empty");
  }

  ComponentTreeOverlapResult result;
  const std::vector<GlobalOrientationTerm> left_global_terms =
      build_global_orientation_terms(ordered_components, true);
  const std::vector<GlobalOrientationTerm> right_global_terms =
      build_global_orientation_terms(ordered_components, false);
  std::unordered_map<SpinOverlapCacheKey,
                     SpinScalarChannelValues,
                     SpinOverlapCacheKeyHasher> overlap_cache;

  for (const auto& left_term : left_global_terms) {
    for (const auto& right_term : right_global_terms) {
      const double pair_weight = left_term.coefficient * right_term.coefficient;
      if (std::abs(pair_weight) <= 1.0e-15) {
        continue;
      }
      const double alpha_overlap = build_exact_spin_overlap(
          left_term.alpha_occ,
          right_term.alpha_occ,
          support_overlap_storage,
          support_size,
          overlap_resolver,
          &result.subdeterminant_evaluations,
          &overlap_cache);
      if (std::abs(alpha_overlap) <= 1.0e-15) {
        continue;
      }
      const double beta_overlap = build_exact_spin_overlap(
          left_term.beta_occ,
          right_term.beta_occ,
          support_overlap_storage,
          support_size,
          overlap_resolver,
          &result.subdeterminant_evaluations,
          &overlap_cache);
      result.overlap += pair_weight * alpha_overlap * beta_overlap;
    }
  }
  return result;
}

ComponentTreeOverlapResult
evaluate_rooted_component_tree_overlap_exact(
    const std::vector<double>& support_overlap_storage,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver) {
  validate_rooted_component_tree_overlap_inputs(
      support_overlap_storage,
      support_size,
      tree);

  const std::vector<SubtreeExpansion> subtree_expansions =
      build_subtree_expansions(tree);
  const SubtreeExpansion& root_expansion =
      subtree_expansions[xmvb::to_size(tree.root_index)];
  return evaluate_component_tree_overlap_exact_reference(
      support_overlap_storage,
      support_size,
      root_expansion.preorder_components,
      overlap_resolver);
}

ComponentTreeOverlapResult
evaluate_rooted_component_tree_overlap_boundary_collapsed(
    const std::vector<double>& support_overlap_storage,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver) {
  validate_rooted_component_tree_overlap_inputs(
      support_overlap_storage,
      support_size,
      tree);

  const std::vector<SubtreeExpansion> subtree_expansions =
      build_subtree_expansions(tree);
  const ComponentData& root_component =
      tree.components[xmvb::to_size(tree.root_index)];
  const std::vector<int>& root_children =
      tree.children[xmvb::to_size(tree.root_index)];

  const ChildSpinSizeVectors child_spin_sizes =
      build_child_spin_size_vectors(root_children, subtree_expansions);

  ComponentTreeOverlapResult result;
  std::unordered_map<SpinOverlapCacheKey,
                     SpinScalarChannelValues,
                     SpinOverlapCacheKeyHasher> overlap_cache;
  std::unordered_map<SubtreeMessageCacheKey,
                     BoundaryScalarMessage,
                     SubtreeMessageCacheKeyHasher> subtree_message_cache;

  const std::vector<WeightedOrientationTermPair> root_term_pairs =
      build_weighted_component_orientation_pairs(root_component);
  for (const auto& root_term_pair : root_term_pairs) {
      const OrientationTerm& left_root_term = root_term_pair.left_term;
      const OrientationTerm& right_root_term = root_term_pair.right_term;
      const double root_coefficient = root_term_pair.coefficient;
      ++result.subtree_term_pair_count;

      std::vector<const BoundaryScalarMessage*> child_messages(root_children.size(), nullptr);
      for (std::size_t child_index = 0; child_index < root_children.size(); ++child_index) {
        const int child = root_children[child_index];
        child_messages[child_index] =
            &build_recursive_subtree_boundary_scalar_cached(
                tree,
                subtree_expansions,
                child,
                left_root_term,
                right_root_term,
                support_overlap_storage,
                {},
                support_size,
                overlap_resolver,
                &result.subtree_term_pair_count,
                &result.subtree_message_state_count,
                &result.subdeterminant_evaluations,
                &result.dp_transition_count,
                &overlap_cache,
                &subtree_message_cache);
      }

      const std::uint32_t local_alpha_row_full_mask =
          right_root_term.alpha_occ.empty()
              ? 0U
              : (open_state_mask_limit(static_cast<int>(right_root_term.alpha_occ.size())) - 1U);
      const std::uint32_t local_alpha_col_full_mask =
          left_root_term.alpha_occ.empty()
              ? 0U
              : (open_state_mask_limit(static_cast<int>(left_root_term.alpha_occ.size())) - 1U);
      const std::uint32_t local_beta_row_full_mask =
          right_root_term.beta_occ.empty()
              ? 0U
              : (open_state_mask_limit(static_cast<int>(right_root_term.beta_occ.size())) - 1U);
      const std::uint32_t local_beta_col_full_mask =
          left_root_term.beta_occ.empty()
              ? 0U
              : (open_state_mask_limit(static_cast<int>(left_root_term.beta_occ.size())) - 1U);

      std::vector<std::uint32_t> selected_alpha_row_masks(root_children.size(), 0U);
      std::vector<std::uint32_t> selected_alpha_col_masks(root_children.size(), 0U);
      std::vector<std::uint32_t> selected_beta_row_masks(root_children.size(), 0U);
      std::vector<std::uint32_t> selected_beta_col_masks(root_children.size(), 0U);

      const auto accumulate_children =
          [&](const auto& self,
              std::size_t child_index,
              std::uint32_t used_alpha_row_mask,
              std::uint32_t used_alpha_col_mask,
              std::uint32_t used_beta_row_mask,
              std::uint32_t used_beta_col_mask,
              double child_overlap_product,
              double child_one_electron) -> void {
            static_cast<void>(child_one_electron);
            if (std::abs(child_overlap_product) <= 1.0e-15) {
              return;
            }

            if (child_index == root_children.size()) {
              const std::vector<int> alpha_local_remainder_rows = select_occ_by_mask(
                  right_root_term.alpha_occ,
                  local_alpha_row_full_mask ^ used_alpha_row_mask);
              const std::vector<int> alpha_local_remainder_cols = select_occ_by_mask(
                  left_root_term.alpha_occ,
                  local_alpha_col_full_mask ^ used_alpha_col_mask);
              const std::vector<int> beta_local_remainder_rows = select_occ_by_mask(
                  right_root_term.beta_occ,
                  local_beta_row_full_mask ^ used_beta_row_mask);
              const std::vector<int> beta_local_remainder_cols = select_occ_by_mask(
                  left_root_term.beta_occ,
                  local_beta_col_full_mask ^ used_beta_col_mask);

              const double alpha_overlap = build_exact_frontier_spin_overlap(
                  {},
                  alpha_local_remainder_cols,
                  {},
                  alpha_local_remainder_rows,
                  support_overlap_storage,
                  support_size,
                  overlap_resolver,
                  &result.subdeterminant_evaluations,
                  &overlap_cache);
              if (std::abs(alpha_overlap) <= 1.0e-15) {
                return;
              }
              const double beta_overlap = build_exact_frontier_spin_overlap(
                  {},
                  beta_local_remainder_cols,
                  {},
                  beta_local_remainder_rows,
                  support_overlap_storage,
                  support_size,
                  overlap_resolver,
                  &result.subdeterminant_evaluations,
                  &overlap_cache);
              if (std::abs(beta_overlap) <= 1.0e-15) {
                return;
              }

              int parity = 0;
              parity ^= component_ordered_block_parity(
                  static_cast<int>(right_root_term.alpha_occ.size()),
                  selected_alpha_row_masks,
                child_spin_sizes.right_alpha);
              parity ^= component_ordered_block_parity(
                  static_cast<int>(left_root_term.alpha_occ.size()),
                  selected_alpha_col_masks,
                child_spin_sizes.left_alpha);
              parity ^= component_ordered_block_parity(
                  static_cast<int>(right_root_term.beta_occ.size()),
                  selected_beta_row_masks,
                child_spin_sizes.right_beta);
              parity ^= component_ordered_block_parity(
                  static_cast<int>(left_root_term.beta_occ.size()),
                  selected_beta_col_masks,
                child_spin_sizes.left_beta);

              result.overlap +=
                  root_coefficient *
                  parity_sign(parity) *
                  child_overlap_product *
                  alpha_overlap *
                  beta_overlap;
              return;
            }

            for (const auto& child_entry :
                 child_messages[xmvb::to_size(child_index)]->nonzero_entries) {
              const BoundarySector& child_alpha_sector =
                  child_messages[xmvb::to_size(child_index)]
                      ->alpha_indexer.sector(child_entry.alpha_sector_index);
              const BoundarySector& child_beta_sector =
                  child_messages[xmvb::to_size(child_index)]
                      ->beta_indexer.sector(child_entry.beta_sector_index);
              if ((used_alpha_row_mask & child_alpha_sector.row_mask) != 0U ||
                  (used_alpha_col_mask & child_alpha_sector.col_mask) != 0U ||
                  (used_beta_row_mask & child_beta_sector.row_mask) != 0U ||
                  (used_beta_col_mask & child_beta_sector.col_mask) != 0U) {
                continue;
              }

              selected_alpha_row_masks[child_index] = child_alpha_sector.row_mask;
              selected_alpha_col_masks[child_index] = child_alpha_sector.col_mask;
              selected_beta_row_masks[child_index] = child_beta_sector.row_mask;
              selected_beta_col_masks[child_index] = child_beta_sector.col_mask;

              ++result.dp_transition_count;
              self(
                  self,
                  child_index + 1,
                  used_alpha_row_mask | child_alpha_sector.row_mask,
                  used_alpha_col_mask | child_alpha_sector.col_mask,
                  used_beta_row_mask | child_beta_sector.row_mask,
                  used_beta_col_mask | child_beta_sector.col_mask,
                  child_overlap_product * child_entry.overlap,
                  child_one_electron * child_entry.overlap +
                      child_overlap_product * child_entry.one_electron);

              selected_alpha_row_masks[child_index] = 0U;
              selected_alpha_col_masks[child_index] = 0U;
              selected_beta_row_masks[child_index] = 0U;
              selected_beta_col_masks[child_index] = 0U;
            }
          };

      accumulate_children(
          accumulate_children,
          0U,
          0U,
          0U,
          0U,
          0U,
          1.0,
          0.0);
  }

  return result;
}

ComponentTreeOneElectronResult evaluate_component_tree_one_electron_exact_reference(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const DeterminantOverlapResolver& overlap_resolver) {
  if (support_size <= 0) {
    throw std::invalid_argument("support_size must be positive");
  }
  if (ordered_components.empty()) {
    throw std::invalid_argument("ordered_components must not be empty");
  }

  ComponentTreeOneElectronResult result;
  const std::vector<GlobalOrientationTerm> left_global_terms =
      build_global_orientation_terms(ordered_components, true);
  const std::vector<GlobalOrientationTerm> right_global_terms =
      build_global_orientation_terms(ordered_components, false);
  std::unordered_map<SpinOverlapCacheKey,
                     SpinScalarChannelValues,
                     SpinOverlapCacheKeyHasher> scalar_cache;

  for (const auto& left_term : left_global_terms) {
    for (const auto& right_term : right_global_terms) {
      const double pair_weight = left_term.coefficient * right_term.coefficient;
      if (std::abs(pair_weight) <= 1.0e-15) {
        continue;
      }
      const SpinScalarChannelValues alpha_values =
          build_spin_overlap_one_electron_cached(
              left_term.alpha_occ,
              right_term.alpha_occ,
              0,
              0,
              false,
              support_overlap_storage,
              support_one_electron_storage,
              support_size,
              overlap_resolver,
              &result.subdeterminant_evaluations,
              &scalar_cache);
      const SpinScalarChannelValues beta_values =
          build_spin_overlap_one_electron_cached(
              left_term.beta_occ,
              right_term.beta_occ,
              0,
              0,
              false,
              support_overlap_storage,
              support_one_electron_storage,
              support_size,
              overlap_resolver,
              &result.subdeterminant_evaluations,
              &scalar_cache);
      const double overlap =
          alpha_values.overlap * beta_values.overlap;
      const double one_electron =
          alpha_values.one_electron * beta_values.overlap +
          beta_values.one_electron * alpha_values.overlap;
      result.overlap += pair_weight * overlap;
      result.one_electron += pair_weight * one_electron;
    }
  }
  return result;
}

ComponentTreeOneElectronResult
evaluate_rooted_component_tree_one_electron_exact(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver) {
  validate_rooted_component_tree_hamiltonian_inputs(
      support_overlap_storage,
      support_one_electron_storage,
      support_size,
      tree);

  const std::vector<SubtreeExpansion> subtree_expansions =
      build_subtree_expansions(tree);
  const SubtreeExpansion& root_expansion =
      subtree_expansions[xmvb::to_size(tree.root_index)];
  return evaluate_component_tree_one_electron_exact_reference(
      support_overlap_storage,
      support_one_electron_storage,
      support_size,
      root_expansion.preorder_components,
      overlap_resolver);
}

ComponentTreeOneElectronResult
evaluate_rooted_component_tree_one_electron_boundary_collapsed(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver) {
  validate_rooted_component_tree_hamiltonian_inputs(
      support_overlap_storage,
      support_one_electron_storage,
      support_size,
      tree);

  const std::vector<SubtreeExpansion> subtree_expansions =
      build_subtree_expansions(tree);
  const ComponentData& root_component =
      tree.components[xmvb::to_size(tree.root_index)];
  const std::vector<int>& root_children =
      tree.children[xmvb::to_size(tree.root_index)];

  const ChildSpinSizeVectors child_spin_sizes =
      build_child_spin_size_vectors(root_children, subtree_expansions);

  ComponentTreeOneElectronResult result;
  std::unordered_map<SpinOverlapCacheKey,
                     SpinScalarChannelValues,
                     SpinOverlapCacheKeyHasher> scalar_cache;
  std::unordered_map<SubtreeMessageCacheKey,
                     BoundaryScalarMessage,
                     SubtreeMessageCacheKeyHasher> subtree_message_cache;

  const std::vector<WeightedOrientationTermPair> one_electron_root_term_pairs =
      build_weighted_component_orientation_pairs(root_component);
  for (const auto& root_term_pair : one_electron_root_term_pairs) {
      const OrientationTerm& left_root_term = root_term_pair.left_term;
      const OrientationTerm& right_root_term = root_term_pair.right_term;
      const double root_coefficient = root_term_pair.coefficient;
      ++result.subtree_term_pair_count;

      std::vector<const BoundaryScalarMessage*> child_messages(root_children.size(), nullptr);
      for (std::size_t child_index = 0; child_index < root_children.size(); ++child_index) {
        const int child = root_children[child_index];
        child_messages[child_index] =
            &build_recursive_subtree_boundary_scalar_cached(
                tree,
                subtree_expansions,
                child,
                left_root_term,
                right_root_term,
                support_overlap_storage,
                support_one_electron_storage,
                support_size,
                overlap_resolver,
                &result.subtree_term_pair_count,
                &result.subtree_message_state_count,
                &result.subdeterminant_evaluations,
                &result.dp_transition_count,
                &scalar_cache,
                &subtree_message_cache);
      }

      const std::uint32_t local_alpha_row_full_mask =
          right_root_term.alpha_occ.empty()
              ? 0U
              : (open_state_mask_limit(static_cast<int>(right_root_term.alpha_occ.size())) - 1U);
      const std::uint32_t local_alpha_col_full_mask =
          left_root_term.alpha_occ.empty()
              ? 0U
              : (open_state_mask_limit(static_cast<int>(left_root_term.alpha_occ.size())) - 1U);
      const std::uint32_t local_beta_row_full_mask =
          right_root_term.beta_occ.empty()
              ? 0U
              : (open_state_mask_limit(static_cast<int>(right_root_term.beta_occ.size())) - 1U);
      const std::uint32_t local_beta_col_full_mask =
          left_root_term.beta_occ.empty()
              ? 0U
              : (open_state_mask_limit(static_cast<int>(left_root_term.beta_occ.size())) - 1U);

      std::vector<std::uint32_t> selected_alpha_row_masks(root_children.size(), 0U);
      std::vector<std::uint32_t> selected_alpha_col_masks(root_children.size(), 0U);
      std::vector<std::uint32_t> selected_beta_row_masks(root_children.size(), 0U);
      std::vector<std::uint32_t> selected_beta_col_masks(root_children.size(), 0U);

      const auto accumulate_children =
          [&](const auto& self,
              std::size_t child_index,
              std::uint32_t used_alpha_row_mask,
              std::uint32_t used_alpha_col_mask,
              std::uint32_t used_beta_row_mask,
              std::uint32_t used_beta_col_mask,
              double child_overlap,
              double child_one_electron) -> void {
            if (std::abs(child_overlap) <= 1.0e-15 &&
                std::abs(child_one_electron) <= 1.0e-15) {
              return;
            }

            if (child_index == root_children.size()) {
              const std::vector<int> alpha_local_remainder_rows = select_occ_by_mask(
                  right_root_term.alpha_occ,
                  local_alpha_row_full_mask ^ used_alpha_row_mask);
              const std::vector<int> alpha_local_remainder_cols = select_occ_by_mask(
                  left_root_term.alpha_occ,
                  local_alpha_col_full_mask ^ used_alpha_col_mask);
              const std::vector<int> beta_local_remainder_rows = select_occ_by_mask(
                  right_root_term.beta_occ,
                  local_beta_row_full_mask ^ used_beta_row_mask);
              const std::vector<int> beta_local_remainder_cols = select_occ_by_mask(
                  left_root_term.beta_occ,
                  local_beta_col_full_mask ^ used_beta_col_mask);

              const SpinScalarChannelValues alpha_values =
                  build_exact_frontier_spin_overlap_one_electron(
                      {},
                      alpha_local_remainder_cols,
                      {},
                      alpha_local_remainder_rows,
                      support_overlap_storage,
                      support_one_electron_storage,
                      support_size,
                      overlap_resolver,
                      &result.subdeterminant_evaluations,
                      &scalar_cache);
              const SpinScalarChannelValues beta_values =
                  build_exact_frontier_spin_overlap_one_electron(
                      {},
                      beta_local_remainder_cols,
                      {},
                      beta_local_remainder_rows,
                      support_overlap_storage,
                      support_one_electron_storage,
                      support_size,
                      overlap_resolver,
                      &result.subdeterminant_evaluations,
                      &scalar_cache);
              const double local_overlap =
                  alpha_values.overlap * beta_values.overlap;
              const double local_one_electron =
                  alpha_values.one_electron * beta_values.overlap +
                  beta_values.one_electron * alpha_values.overlap;
              if (std::abs(local_overlap) <= 1.0e-15 &&
                  std::abs(local_one_electron) <= 1.0e-15) {
                return;
              }

              int parity = 0;
              parity ^= component_ordered_block_parity(
                  static_cast<int>(right_root_term.alpha_occ.size()),
                  selected_alpha_row_masks,
                  child_spin_sizes.right_alpha);
              parity ^= component_ordered_block_parity(
                  static_cast<int>(left_root_term.alpha_occ.size()),
                  selected_alpha_col_masks,
                  child_spin_sizes.left_alpha);
              parity ^= component_ordered_block_parity(
                  static_cast<int>(right_root_term.beta_occ.size()),
                  selected_beta_row_masks,
                  child_spin_sizes.right_beta);
              parity ^= component_ordered_block_parity(
                  static_cast<int>(left_root_term.beta_occ.size()),
                  selected_beta_col_masks,
                  child_spin_sizes.left_beta);

              const double combined_overlap = child_overlap * local_overlap;
              const double combined_one_electron =
                  child_one_electron * local_overlap +
                  child_overlap * local_one_electron;
              const double signed_scale =
                  root_coefficient * parity_sign(parity);
              result.overlap += signed_scale * combined_overlap;
              result.one_electron += signed_scale * combined_one_electron;
              return;
            }

            for (const auto& child_entry :
                 child_messages[xmvb::to_size(child_index)]->nonzero_entries) {
              const BoundarySector& child_alpha_sector =
                  child_messages[xmvb::to_size(child_index)]
                      ->alpha_indexer.sector(child_entry.alpha_sector_index);
              const BoundarySector& child_beta_sector =
                  child_messages[xmvb::to_size(child_index)]
                      ->beta_indexer.sector(child_entry.beta_sector_index);
              if ((used_alpha_row_mask & child_alpha_sector.row_mask) != 0U ||
                  (used_alpha_col_mask & child_alpha_sector.col_mask) != 0U ||
                  (used_beta_row_mask & child_beta_sector.row_mask) != 0U ||
                  (used_beta_col_mask & child_beta_sector.col_mask) != 0U) {
                continue;
              }

              selected_alpha_row_masks[child_index] = child_alpha_sector.row_mask;
              selected_alpha_col_masks[child_index] = child_alpha_sector.col_mask;
              selected_beta_row_masks[child_index] = child_beta_sector.row_mask;
              selected_beta_col_masks[child_index] = child_beta_sector.col_mask;

              ++result.dp_transition_count;
              self(
                  self,
                  child_index + 1,
                  used_alpha_row_mask | child_alpha_sector.row_mask,
                  used_alpha_col_mask | child_alpha_sector.col_mask,
                  used_beta_row_mask | child_beta_sector.row_mask,
                  used_beta_col_mask | child_beta_sector.col_mask,
                  child_overlap * child_entry.overlap,
                  child_one_electron * child_entry.overlap +
                      child_overlap * child_entry.one_electron);

              selected_alpha_row_masks[child_index] = 0U;
              selected_alpha_col_masks[child_index] = 0U;
              selected_beta_row_masks[child_index] = 0U;
              selected_beta_col_masks[child_index] = 0U;
            }
          };

      accumulate_children(
          accumulate_children,
          0U,
          0U,
          0U,
          0U,
          0U,
          1.0,
          0.0);
  }

  return result;
}

ComponentTreeOppositeSpinResult
evaluate_rooted_component_tree_opposite_spin_exact(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver) {
  const ComponentTreeHamiltonianResult full_result =
      evaluate_rooted_component_tree_hamiltonian_exact(
          support_overlap_storage,
          support_one_electron_storage,
          packed_active_two_electron_integrals,
          support_size,
          tree,
          overlap_resolver);
  return build_opposite_spin_result(full_result);
}

ComponentTreeOppositeSpinDebugResult
evaluate_rooted_component_tree_opposite_spin_boundary_debug(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver) {
  validate_rooted_component_tree_hamiltonian_inputs(
      support_overlap_storage,
      support_one_electron_storage,
      support_size,
      tree);
  ComponentTreeOppositeSpinDebugResult result;
  const ComponentTreeHamiltonianResult full_result =
      evaluate_rooted_component_tree_hamiltonian_bundle_collapsed_production_internal(
          support_overlap_storage,
          support_one_electron_storage,
          packed_active_two_electron_integrals,
          support_size,
          tree,
          overlap_resolver,
          &result.alpha_first_cofactor,
          &result.beta_first_cofactor,
          nullptr,
          nullptr);
  assign_opposite_spin_debug_result(full_result, &result);
  return result;
}

ComponentTreeOppositeSpinResult
evaluate_rooted_component_tree_opposite_spin_boundary_collapsed(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver) {
  validate_rooted_component_tree_hamiltonian_inputs(
      support_overlap_storage,
      support_one_electron_storage,
      support_size,
      tree);
  if (is_one_leaf_star_tree(tree)) {
    const ComponentTreeHamiltonianResult full_result =
        evaluate_rooted_component_tree_hamiltonian_collapsed(
            support_overlap_storage,
            support_one_electron_storage,
            packed_active_two_electron_integrals,
            support_size,
            tree,
            overlap_resolver);
    return build_opposite_spin_result(full_result);
  }
  const ComponentTreeHamiltonianResult full_result =
      evaluate_rooted_component_tree_hamiltonian_bundle_collapsed_production_internal(
          support_overlap_storage,
          support_one_electron_storage,
          packed_active_two_electron_integrals,
          support_size,
          tree,
          overlap_resolver,
          nullptr,
          nullptr,
          nullptr,
          nullptr);
  return build_opposite_spin_result(full_result);
}

ComponentTreeSameSpinResult
evaluate_rooted_component_tree_same_spin_exact(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver) {
  const ComponentTreeHamiltonianResult full_result =
      evaluate_rooted_component_tree_hamiltonian_exact(
          support_overlap_storage,
          support_one_electron_storage,
          packed_active_two_electron_integrals,
          support_size,
          tree,
          overlap_resolver);
  return build_same_spin_result(full_result);
}

ComponentTreeSameSpinResult
evaluate_rooted_component_tree_same_spin_boundary_collapsed(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver) {
  validate_rooted_component_tree_hamiltonian_inputs(
      support_overlap_storage,
      support_one_electron_storage,
      support_size,
      tree);
  if (is_one_leaf_star_tree(tree)) {
    const ComponentTreeHamiltonianResult full_result =
        evaluate_rooted_component_tree_hamiltonian_collapsed(
            support_overlap_storage,
            support_one_electron_storage,
            packed_active_two_electron_integrals,
            support_size,
            tree,
            overlap_resolver);
    return build_same_spin_result(full_result);
  }
  const ComponentTreeHamiltonianResult full_result =
      evaluate_rooted_component_tree_hamiltonian_bundle_collapsed_production_internal(
          support_overlap_storage,
          support_one_electron_storage,
          packed_active_two_electron_integrals,
          support_size,
          tree,
          overlap_resolver,
          nullptr,
          nullptr,
          nullptr,
          nullptr);
  return build_same_spin_result(full_result);
}

ComponentTreeHamiltonianResult
evaluate_rooted_component_tree_hamiltonian_exact(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver) {
  validate_rooted_component_tree_hamiltonian_inputs(
      support_overlap_storage,
      support_one_electron_storage,
      support_size,
      tree);

  const std::vector<SubtreeExpansion> subtree_expansions =
      build_subtree_expansions(tree);
  const SubtreeExpansion& root_expansion =
      subtree_expansions[xmvb::to_size(tree.root_index)];
  return evaluate_component_tree_hamiltonian_exact_reference(
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      support_size,
      root_expansion.preorder_components,
      overlap_resolver);
}

ComponentTreeHamiltonianResult
evaluate_rooted_component_tree_hamiltonian_collapsed(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver) {
  validate_rooted_component_tree_hamiltonian_inputs(
      support_overlap_storage,
      support_one_electron_storage,
      support_size,
      tree);
  if (tree.children[xmvb::to_size(tree.root_index)].empty()) {
    return evaluate_single_node_hamiltonian_boundary_collapsed(
        support_overlap_storage,
        support_one_electron_storage,
        packed_active_two_electron_integrals,
        support_size,
        tree.components[xmvb::to_size(tree.root_index)],
        overlap_resolver);
  }
  if (is_one_leaf_star_tree(tree)) {
    const std::vector<SubtreeExpansion> subtree_expansions =
        build_subtree_expansions(tree);
    const SubtreeExpansion& root_expansion =
        subtree_expansions[xmvb::to_size(tree.root_index)];
    const OneLeafHamiltonianStarPairResult one_leaf_result =
        evaluate_component_ordered_open_state_star_pair_hamiltonian_one_leaf_collapsed(
            support_overlap_storage,
            support_one_electron_storage,
            packed_active_two_electron_integrals,
            support_size,
            root_expansion.preorder_components,
            overlap_resolver);

    ComponentTreeHamiltonianResult result;
    result.overlap = one_leaf_result.overlap;
    result.one_electron = one_leaf_result.one_electron;
    result.same_spin_alpha_two_electron =
        one_leaf_result.same_spin_alpha_two_electron;
    result.same_spin_beta_two_electron =
        one_leaf_result.same_spin_beta_two_electron;
    result.opposite_spin_two_electron =
        one_leaf_result.opposite_spin_two_electron;
    result.two_electron = one_leaf_result.two_electron;
    result.total_electronic_hamiltonian =
        result.one_electron + result.two_electron;
    result.subtree_message_state_count = one_leaf_result.collapsed_leaf_state_count;
    result.subtree_term_pair_count = one_leaf_result.hypercube_assignment_count;
    result.subdeterminant_evaluations = one_leaf_result.subdeterminant_evaluations;
    result.dp_transition_count = one_leaf_result.dp_transition_count;
    return result;
  }
  return evaluate_rooted_component_tree_hamiltonian_bundle_collapsed_production_internal(
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      support_size,
      tree,
      overlap_resolver,
      nullptr,
      nullptr,
      nullptr,
      nullptr);
}

ComponentTreeHamiltonianResult
evaluate_rooted_component_tree_hamiltonian_bundle_boundary_collapsed(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver) {
  validate_rooted_component_tree_hamiltonian_inputs(
      support_overlap_storage,
      support_one_electron_storage,
      support_size,
      tree);
  if (tree.children[xmvb::to_size(tree.root_index)].empty() ||
      is_one_leaf_star_tree(tree)) {
    if (tree.children[xmvb::to_size(tree.root_index)].empty()) {
      return evaluate_rooted_component_tree_hamiltonian_collapsed(
          support_overlap_storage,
          support_one_electron_storage,
          packed_active_two_electron_integrals,
          support_size,
          tree,
          overlap_resolver);
    }
    const std::vector<SubtreeExpansion> subtree_expansions =
        build_subtree_expansions(tree);
    const SubtreeExpansion& root_expansion =
        subtree_expansions[xmvb::to_size(tree.root_index)];
    return evaluate_one_leaf_hamiltonian_boundary_bundle_collapsed(
        support_overlap_storage,
        support_one_electron_storage,
        packed_active_two_electron_integrals,
        support_size,
        root_expansion.preorder_components,
        overlap_resolver);
  }
  return evaluate_rooted_component_tree_hamiltonian_bundle_collapsed_production_internal(
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      support_size,
      tree,
      overlap_resolver,
      nullptr,
      nullptr,
      nullptr,
      nullptr);
}

ComponentTreeHamiltonianGradientResult
evaluate_rooted_component_tree_active_space_gradient_exact(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver,
    double hamiltonian_weight,
    double overlap_weight) {
  validate_rooted_component_tree_hamiltonian_inputs(
      support_overlap_storage,
      support_one_electron_storage,
      support_size,
      tree);

  const std::vector<SubtreeExpansion> subtree_expansions =
      build_subtree_expansions(tree);
  const SubtreeExpansion& root_expansion =
      subtree_expansions[xmvb::to_size(tree.root_index)];
  return evaluate_component_tree_active_space_gradient_exact_reference(
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      support_size,
      root_expansion.preorder_components,
      overlap_resolver,
      hamiltonian_weight,
      overlap_weight);
}

ComponentTreeHamiltonianGradientResult
evaluate_rooted_component_tree_active_space_gradient_boundary_collapsed(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver,
    double hamiltonian_weight,
    double overlap_weight) {
  validate_rooted_component_tree_hamiltonian_inputs(
      support_overlap_storage,
      support_one_electron_storage,
      support_size,
      tree);

  ComponentTreeHamiltonianGradientResult result;
  if (std::abs(hamiltonian_weight) > 1.0e-15) {
    Eigen::MatrixXd active_one_electron_gradient;
    result.hamiltonian =
        evaluate_rooted_component_tree_hamiltonian_bundle_collapsed_internal(
            support_overlap_storage,
            support_one_electron_storage,
            packed_active_two_electron_integrals,
            support_size,
            tree,
            overlap_resolver,
            nullptr,
            nullptr,
            &active_one_electron_gradient,
            &result.packed_active_two_electron_gradient);
    if (std::abs(hamiltonian_weight - 1.0) > 1.0e-15) {
      active_one_electron_gradient *= hamiltonian_weight;
      for (double& value : result.packed_active_two_electron_gradient) {
        value *= hamiltonian_weight;
      }
    }
    result.active_one_electron_gradient.assign(
        active_one_electron_gradient.data(),
        active_one_electron_gradient.data() + active_one_electron_gradient.size());
  } else {
    result.hamiltonian =
        evaluate_rooted_component_tree_hamiltonian_bundle_collapsed_internal(
            support_overlap_storage,
            support_one_electron_storage,
            packed_active_two_electron_integrals,
            support_size,
            tree,
            overlap_resolver,
            nullptr,
            nullptr,
            nullptr,
            nullptr);
    result.active_one_electron_gradient.assign(
        xmvb::to_size(support_size * support_size),
        0.0);
    result.packed_active_two_electron_gradient.assign(
        packed_active_two_electron_integrals.size(),
        0.0);
  }
  accumulate_rooted_component_tree_weighted_overlap_gradient_boundary_collapsed(
      tree,
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      support_size,
      overlap_resolver,
      hamiltonian_weight,
      overlap_weight,
      &result.active_orbital_overlap_gradient);
  return result;
}

ComponentTreeHamiltonianGradientResult
evaluate_rooted_component_tree_hamiltonian_gradient_exact(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver) {
  return evaluate_rooted_component_tree_active_space_gradient_exact(
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      support_size,
      tree,
      overlap_resolver,
      1.0,
      0.0);
}

ComponentTreeHamiltonianGradientResult
evaluate_rooted_component_tree_hamiltonian_gradient_boundary_collapsed(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver) {
  return evaluate_rooted_component_tree_active_space_gradient_boundary_collapsed(
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      support_size,
      tree,
      overlap_resolver,
      1.0,
      0.0);
}

CollapsedHamiltonianComponentTreeStats
evaluate_rooted_component_tree_hamiltonian(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver) {
  const ComponentTreeHamiltonianResult exact =
      evaluate_rooted_component_tree_hamiltonian_exact(
          support_overlap_storage,
          support_one_electron_storage,
          packed_active_two_electron_integrals,
          support_size,
          tree,
          overlap_resolver);
  const ComponentTreeHamiltonianResult collapsed =
      evaluate_rooted_component_tree_hamiltonian_collapsed(
          support_overlap_storage,
          support_one_electron_storage,
          packed_active_two_electron_integrals,
          support_size,
          tree,
          overlap_resolver);
  return build_hamiltonian_comparison_stats(exact, collapsed);
}

CollapsedTwoElectronComponentTreeStats
evaluate_rooted_component_tree_two_electron(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const ComponentTree& tree,
    const DeterminantOverlapResolver& overlap_resolver) {
  return evaluate_rooted_component_tree_hamiltonian(
      support_overlap_storage,
      std::vector<double>(xmvb::to_size(support_size * support_size), 0.0),
      packed_active_two_electron_integrals,
      support_size,
      tree,
      overlap_resolver);
}
}  // namespace xmvb::vb::exact_separator
