#include "vb/exact_separator/bundle.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace xmvb::vb::exact_separator {

namespace {

int triangular_pair_count(int support_size) {
  if (support_size < 0) {
    throw std::invalid_argument("support_size must be non-negative");
  }
  return support_size * (support_size - 1) / 2;
}

int find_label_position(
    const std::vector<int>& boundary_labels,
    int label) {
  for (int index = 0; index < static_cast<int>(boundary_labels.size()); ++index) {
    if (boundary_labels[xmvb::to_size(index)] == label) {
      return index;
    }
  }
  return -1;
}

}  // namespace

BoundarySpinBundleLayout::BoundarySpinBundleLayout(
    int row_count,
    int col_count,
    int base_selected_row_minus_col,
    int min_family_delta,
    int max_family_delta,
    int support_size)
    : family_range_(
          row_count,
          col_count,
          base_selected_row_minus_col,
          min_family_delta,
          max_family_delta),
      support_size_(support_size),
      support_pair_count_(support_size * support_size),
      antisym_support_pair_count_(triangular_pair_count(support_size)),
      support_pair_pair_count_(
          triangular_pair_count(support_size) *
          triangular_pair_count(support_size)) {
  // The bundle layout fixes the exact dense storage order used later by the
  // bundle merge algebra. The family offsets are precomputed once here so
  // forward / reverse kernels can use integer indexing only.
  if (support_size_ < 0) {
    throw std::invalid_argument("support_size must be non-negative");
  }
  family_sector_offsets_.reserve(
      xmvb::to_size(family_range_.family_count()));
  int running_offset = 0;
  for (int family_index_value = 0;
       family_index_value < family_range_.family_count();
       ++family_index_value) {
    family_sector_offsets_.push_back(running_offset);
    running_offset +=
        family_range_.family(family_index_value).sector_indexer.sector_count();
  }
  total_sector_count_ = running_offset;
}

int BoundarySpinBundleLayout::row_count() const {
  return family_range_.row_count();
}

int BoundarySpinBundleLayout::col_count() const {
  return family_range_.col_count();
}

int BoundarySpinBundleLayout::base_selected_row_minus_col() const {
  return family_range_.base_selected_row_minus_col();
}

int BoundarySpinBundleLayout::min_family_delta() const {
  return family_range_.min_family_delta();
}

int BoundarySpinBundleLayout::max_family_delta() const {
  return family_range_.max_family_delta();
}

int BoundarySpinBundleLayout::support_size() const {
  return support_size_;
}

int BoundarySpinBundleLayout::family_count() const {
  return family_range_.family_count();
}

int BoundarySpinBundleLayout::total_sector_count() const {
  return total_sector_count_;
}

int BoundarySpinBundleLayout::support_pair_count() const {
  return support_pair_count_;
}

int BoundarySpinBundleLayout::antisym_support_pair_count() const {
  return antisym_support_pair_count_;
}

int BoundarySpinBundleLayout::support_pair_pair_count() const {
  return support_pair_pair_count_;
}

const BoundarySectorFamilyIndexerRange& BoundarySpinBundleLayout::family_range() const {
  return family_range_;
}

const BoundarySectorFamilyIndexer& BoundarySpinBundleLayout::family(int index) const {
  return family_range_.family(index);
}

int BoundarySpinBundleLayout::family_index(int family_delta) const {
  return family_range_.family_index(family_delta);
}

int BoundarySpinBundleLayout::family_sector_offset(int family_index_value) const {
  if (family_index_value < 0 || family_index_value >= family_count()) {
    throw std::out_of_range("family_index is out of range");
  }
  return family_sector_offsets_[xmvb::to_size(family_index_value)];
}

int BoundarySpinBundleLayout::flat_sector_index(
    int family_index_value,
    int sector_index) const {
  const BoundarySectorFamilyIndexer& family_value = family(family_index_value);
  if (sector_index < 0 ||
      sector_index >= family_value.sector_indexer.sector_count()) {
    throw std::out_of_range("sector_index is out of range");
  }
  return family_sector_offset(family_index_value) + sector_index;
}

int BoundarySpinBundleLayout::flat_degree1_index(
    int support_pair_index,
    int flat_sector_index_value) const {
  if (support_pair_index < 0 || support_pair_index >= support_pair_count_) {
    throw std::out_of_range("support_pair_index is out of range");
  }
  if (flat_sector_index_value < 0 || flat_sector_index_value >= total_sector_count_) {
    throw std::out_of_range("flat_sector_index is out of range");
  }
  return
      support_pair_index * total_sector_count_ +
      flat_sector_index_value;
}

int BoundarySpinBundleLayout::flat_degree2_index(
    int support_pair_pair_index,
    int flat_sector_index_value) const {
  if (support_pair_pair_index < 0 ||
      support_pair_pair_index >= support_pair_pair_count_) {
    throw std::out_of_range("support_pair_pair_index is out of range");
  }
  if (flat_sector_index_value < 0 || flat_sector_index_value >= total_sector_count_) {
    throw std::out_of_range("flat_sector_index is out of range");
  }
  return
      support_pair_pair_index * total_sector_count_ +
      flat_sector_index_value;
}

BoundarySpinBundle make_zero_boundary_spin_bundle(
    const BoundarySpinBundleLayout& layout) {
  // The dense bundle stores every degree in one contiguous array so later
  // merge kernels can contract whole channels without rebuilding sparse maps.
  BoundarySpinBundle bundle;
  bundle.layout = layout;
  bundle.degree0.assign(
      xmvb::to_size(layout.total_sector_count()),
      0.0);
  bundle.degree1.assign(
      xmvb::to_size(layout.support_pair_count()) *
          xmvb::to_size(layout.total_sector_count()),
      0.0);
  bundle.degree2.assign(
      xmvb::to_size(layout.support_pair_pair_count()) *
          xmvb::to_size(layout.total_sector_count()),
      0.0);
  return bundle;
}

BoundaryMixedBundle make_zero_boundary_mixed_bundle(
    int alpha_degree1_entry_count,
    int beta_degree1_entry_count) {
  if (alpha_degree1_entry_count < 0 || beta_degree1_entry_count < 0) {
    throw std::invalid_argument("mixed bundle dimensions must be non-negative");
  }
  BoundaryMixedBundle bundle;
  bundle.alpha_degree1_entry_count = alpha_degree1_entry_count;
  bundle.beta_degree1_entry_count = beta_degree1_entry_count;
  bundle.values.assign(
      xmvb::to_size(alpha_degree1_entry_count) *
          xmvb::to_size(beta_degree1_entry_count),
      0.0);
  return bundle;
}

BoundaryBundle make_zero_boundary_bundle(
    const BoundarySpinBundleLayout& alpha_layout,
    const BoundarySpinBundleLayout& beta_layout) {
  BoundaryBundle bundle;
  bundle.alpha = make_zero_boundary_spin_bundle(alpha_layout);
  bundle.beta = make_zero_boundary_spin_bundle(beta_layout);
  bundle.mixed = make_zero_boundary_mixed_bundle(
      alpha_layout.support_pair_count() * alpha_layout.total_sector_count(),
      beta_layout.support_pair_count() * beta_layout.total_sector_count());
  return bundle;
}

bool has_compatible_layout(const BoundarySpinBundle& bundle) {
  return
      bundle.degree0.size() ==
          xmvb::to_size(bundle.layout.total_sector_count()) &&
      bundle.degree1.size() ==
          xmvb::to_size(bundle.layout.support_pair_count()) *
              xmvb::to_size(bundle.layout.total_sector_count()) &&
      bundle.degree2.size() ==
          xmvb::to_size(bundle.layout.support_pair_pair_count()) *
              xmvb::to_size(bundle.layout.total_sector_count());
}

bool has_compatible_layout(const BoundaryBundle& bundle) {
  if (!has_compatible_layout(bundle.alpha) || !has_compatible_layout(bundle.beta)) {
    return false;
  }
  const int expected_alpha_degree1_entry_count =
      bundle.alpha.layout.support_pair_count() *
      bundle.alpha.layout.total_sector_count();
  const int expected_beta_degree1_entry_count =
      bundle.beta.layout.support_pair_count() *
      bundle.beta.layout.total_sector_count();
  return
      bundle.mixed.alpha_degree1_entry_count == expected_alpha_degree1_entry_count &&
      bundle.mixed.beta_degree1_entry_count == expected_beta_degree1_entry_count &&
      bundle.mixed.values.size() ==
          xmvb::to_size(expected_alpha_degree1_entry_count) *
              xmvb::to_size(expected_beta_degree1_entry_count);
}

int flatten_support_pair_index(
    int row_orbital,
    int col_orbital,
    int support_size) {
  if (support_size < 0) {
    throw std::invalid_argument("support_size must be non-negative");
  }
  if (row_orbital < 0 || row_orbital >= support_size ||
      col_orbital < 0 || col_orbital >= support_size) {
    throw std::out_of_range("support-space orbital index is out of range");
  }
  return col_orbital * support_size + row_orbital;
}

int flatten_antisym_support_pair_index(
    int first_orbital,
    int second_orbital,
    int support_size) {
  if (support_size < 0) {
    throw std::invalid_argument("support_size must be non-negative");
  }
  if (first_orbital < 0 || second_orbital < 0 ||
      first_orbital >= support_size || second_orbital >= support_size) {
    throw std::out_of_range("support-space orbital index is out of range");
  }
  if (first_orbital >= second_orbital) {
    throw std::invalid_argument("antisymmetrized support pair requires first < second");
  }
  return
      first_orbital * (2 * support_size - first_orbital - 1) / 2 +
      (second_orbital - first_orbital - 1);
}

int flatten_support_pair_pair_index(
    int row_first_orbital,
    int row_second_orbital,
    int col_first_orbital,
    int col_second_orbital,
    int support_size) {
  const int pair_count = triangular_pair_count(support_size);
  const int row_pair_index = flatten_antisym_support_pair_index(
      row_first_orbital,
      row_second_orbital,
      support_size);
  const int col_pair_index = flatten_antisym_support_pair_index(
      col_first_orbital,
      col_second_orbital,
      support_size);
  return col_pair_index * pair_count + row_pair_index;
}

std::uint32_t encode_boundary_selection_mask(
    const std::vector<int>& boundary_labels,
    const std::vector<int>& selected_labels) {
  if (boundary_labels.size() >= 31U) {
    throw std::invalid_argument("boundary width exceeds mask representation");
  }

  std::unordered_set<int> seen_labels;
  std::uint32_t mask = 0U;
  for (const int label : selected_labels) {
    if (!seen_labels.insert(label).second) {
      throw std::invalid_argument("selected_labels contains duplicates");
    }
    const int position = find_label_position(boundary_labels, label);
    if (position < 0) {
      throw std::invalid_argument("selected label is not part of the boundary order");
    }
    mask |= (static_cast<std::uint32_t>(1U) << static_cast<std::uint32_t>(position));
  }
  return mask;
}

int find_boundary_flat_sector_index(
    const BoundarySpinBundleLayout& layout,
    const std::vector<int>& boundary_row_labels,
    const std::vector<int>& selected_row_labels,
    const std::vector<int>& boundary_col_labels,
    const std::vector<int>& selected_col_labels) {
  // This helper is the bridge between the old label-set view of deleted-minor
  // sectors and the new canonical dense sector basis. Future adapters can use
  // it to project legacy payload entries into the bundle layout without
  // rebuilding per-call search tables.
  if (static_cast<int>(boundary_row_labels.size()) != layout.row_count() ||
      static_cast<int>(boundary_col_labels.size()) != layout.col_count()) {
    throw std::invalid_argument("boundary labels do not match the layout dimensions");
  }

  const std::uint32_t row_mask =
      encode_boundary_selection_mask(boundary_row_labels, selected_row_labels);
  const std::uint32_t col_mask =
      encode_boundary_selection_mask(boundary_col_labels, selected_col_labels);
  const int selected_row_minus_col =
      static_cast<int>(selected_row_labels.size()) -
      static_cast<int>(selected_col_labels.size());
  const int family_delta =
      selected_row_minus_col - layout.base_selected_row_minus_col();
  const int family_index_value = layout.family_index(family_delta);
  if (family_index_value < 0) {
    return -1;
  }
  const int sector_index_value =
      layout.family(family_index_value).sector_indexer.index(row_mask, col_mask);
  if (sector_index_value < 0) {
    return -1;
  }
  return layout.flat_sector_index(family_index_value, sector_index_value);
}

BoundaryFlatSectorIndex decode_boundary_flat_sector_index(
    const BoundarySpinBundleLayout& layout,
    int flat_sector_index_value) {
  if (flat_sector_index_value < 0 ||
      flat_sector_index_value >= layout.total_sector_count()) {
    throw std::out_of_range("flat_sector_index is out of range");
  }

  BoundaryFlatSectorIndex decoded;
  for (int family_index_value = 0;
       family_index_value < layout.family_count();
       ++family_index_value) {
    const int offset = layout.family_sector_offset(family_index_value);
    const int family_sector_count =
        layout.family(family_index_value).sector_indexer.sector_count();
    if (flat_sector_index_value < offset + family_sector_count) {
      decoded.family_index = family_index_value;
      decoded.sector_index = flat_sector_index_value - offset;
      return decoded;
    }
  }

  throw std::logic_error("flat sector index could not be decoded");
}

}  // namespace xmvb::vb::exact_separator
