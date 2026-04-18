#include "vb/exact_separator/bundle_merge.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace xmvb::vb::exact_separator {

namespace {

constexpr double kZeroTolerance = 1.0e-15;

struct FlatSectorState {
  int family_index = -1;
  int sector_index = -1;
  int family_delta = 0;
  int selected_row_count = 0;
  int selected_col_count = 0;
  std::uint32_t row_mask = 0U;
  std::uint32_t col_mask = 0U;
};

struct Degree1FlatIndex {
  int support_pair_index = -1;
  int flat_sector_index = -1;
};

struct SupportPair {
  int row_orbital = -1;
  int col_orbital = -1;
};

bool same_spin_layout(
    const BoundarySpinBundleLayout& left,
    const BoundarySpinBundleLayout& right) {
  return
      left.row_count() == right.row_count() &&
      left.col_count() == right.col_count() &&
      left.base_selected_row_minus_col() ==
          right.base_selected_row_minus_col() &&
      left.min_family_delta() == right.min_family_delta() &&
      left.max_family_delta() == right.max_family_delta() &&
      left.support_size() == right.support_size() &&
      left.total_sector_count() == right.total_sector_count();
}

int degree1_entry_count(const BoundarySpinBundleLayout& layout) {
  return layout.support_pair_count() * layout.total_sector_count();
}

int mixed_flat_index(
    const BoundaryMixedBundle& bundle,
    int alpha_degree1_flat_index,
    int beta_degree1_flat_index) {
  if (alpha_degree1_flat_index < 0 ||
      alpha_degree1_flat_index >= bundle.alpha_degree1_entry_count ||
      beta_degree1_flat_index < 0 ||
      beta_degree1_flat_index >= bundle.beta_degree1_entry_count) {
    throw std::out_of_range("mixed flat index is out of range");
  }
  return
      beta_degree1_flat_index * bundle.alpha_degree1_entry_count +
      alpha_degree1_flat_index;
}

double mixed_value(
    const BoundaryMixedBundle& bundle,
    int alpha_degree1_flat_index,
    int beta_degree1_flat_index) {
  return bundle.values[xmvb::to_size(
      mixed_flat_index(bundle, alpha_degree1_flat_index, beta_degree1_flat_index))];
}

void add_mixed_value(
    BoundaryMixedBundle* bundle,
    int alpha_degree1_flat_index,
    int beta_degree1_flat_index,
    double value) {
  if (bundle == nullptr) {
    throw std::invalid_argument("mixed bundle output must not be null");
  }
  bundle->values[xmvb::to_size(
      mixed_flat_index(*bundle, alpha_degree1_flat_index, beta_degree1_flat_index))] +=
      value;
}

FlatSectorState flat_sector_state(
    const BoundarySpinBundleLayout& layout,
    int flat_sector_index) {
  const BoundaryFlatSectorIndex decoded =
      decode_boundary_flat_sector_index(layout, flat_sector_index);
  const BoundarySectorFamilyIndexer& family =
      layout.family(decoded.family_index);
  const BoundarySector& sector =
      family.sector_indexer.sector(decoded.sector_index);
  return FlatSectorState{
      .family_index = decoded.family_index,
      .sector_index = decoded.sector_index,
      .family_delta = family.family_delta,
      .selected_row_count = sector.selected_row_count,
      .selected_col_count = sector.selected_col_count,
      .row_mask = sector.row_mask,
      .col_mask = sector.col_mask,
  };
}

Degree1FlatIndex decode_degree1_flat_index(
    const BoundarySpinBundleLayout& layout,
    int flat_degree1_index) {
  if (flat_degree1_index < 0 ||
      flat_degree1_index >= degree1_entry_count(layout)) {
    throw std::out_of_range("degree-1 flat index is out of range");
  }
  return Degree1FlatIndex{
      .support_pair_index = flat_degree1_index / layout.total_sector_count(),
      .flat_sector_index = flat_degree1_index % layout.total_sector_count(),
  };
}

SupportPair decode_support_pair(
    int support_pair_index,
    int support_size) {
  if (support_pair_index < 0 ||
      support_pair_index >= support_size * support_size) {
    throw std::out_of_range("support pair index is out of range");
  }
  return SupportPair{
      .row_orbital = support_pair_index % support_size,
      .col_orbital = support_pair_index / support_size,
  };
}

int spin_sector_merge_parity_from_states(
    const BoundarySpinBundleLayout& left_layout,
    const FlatSectorState& left_state,
    const BoundarySpinBundleLayout& right_layout,
    const FlatSectorState& right_state) {
  const int left_row_excess =
      std::max(0, left_state.selected_row_count - left_state.selected_col_count);
  const int left_col_excess =
      std::max(0, left_state.selected_col_count - left_state.selected_row_count);
  const int right_degree =
      std::max(right_state.selected_row_count, right_state.selected_col_count);
  return
      left_row_excess * right_layout.col_count() +
      left_col_excess * right_layout.row_count() +
      right_degree * (left_layout.row_count() + left_layout.col_count());
}

bool merge_flat_spin_sectors(
    const BoundarySpinBundleLayout& left_layout,
    int left_flat_sector_index,
    const BoundarySpinBundleLayout& right_layout,
    int right_flat_sector_index,
    const BoundarySpinBundleLayout& merged_layout,
    int* merged_flat_sector_index,
    int* merge_parity) {
  if (merged_flat_sector_index == nullptr || merge_parity == nullptr) {
    throw std::invalid_argument("merge_flat_spin_sectors outputs must not be null");
  }
  const FlatSectorState left_state =
      flat_sector_state(left_layout, left_flat_sector_index);
  const FlatSectorState right_state =
      flat_sector_state(right_layout, right_flat_sector_index);
  const int merged_family_delta =
      left_state.family_delta + right_state.family_delta;
  const int merged_family_index = merged_layout.family_index(merged_family_delta);
  if (merged_family_index < 0) {
    return false;
  }
  const std::uint32_t merged_row_mask =
      left_state.row_mask |
      (right_state.row_mask << static_cast<std::uint32_t>(left_layout.row_count()));
  const std::uint32_t merged_col_mask =
      left_state.col_mask |
      (right_state.col_mask << static_cast<std::uint32_t>(left_layout.col_count()));
  const int merged_sector_index =
      merged_layout.family(merged_family_index).sector_indexer.index(
          merged_row_mask,
          merged_col_mask);
  if (merged_sector_index < 0) {
    return false;
  }
  *merged_flat_sector_index =
      merged_layout.flat_sector_index(merged_family_index, merged_sector_index);
  *merge_parity = spin_sector_merge_parity_from_states(
      left_layout,
      left_state,
      right_layout,
      right_state);
  return true;
}

bool merge_left_degree1_with_right_degree0_sector(
    const BoundarySpinBundleLayout& left_degree1_layout,
    int left_degree1_flat_index,
    const BoundarySpinBundleLayout& right_degree0_layout,
    int right_degree0_flat_sector_index,
    const BoundarySpinBundleLayout& merged_layout,
    int* merged_degree1_flat_index,
    int* merge_parity) {
  if (merged_degree1_flat_index == nullptr || merge_parity == nullptr) {
    throw std::invalid_argument(
        "merge_left_degree1_with_right_degree0_sector outputs must not be null");
  }
  const Degree1FlatIndex decoded_left =
      decode_degree1_flat_index(left_degree1_layout, left_degree1_flat_index);
  int merged_flat_sector_index = -1;
  if (!merge_flat_spin_sectors(
          left_degree1_layout,
          decoded_left.flat_sector_index,
          right_degree0_layout,
          right_degree0_flat_sector_index,
          merged_layout,
          &merged_flat_sector_index,
          merge_parity)) {
    return false;
  }
  *merged_degree1_flat_index =
      merged_layout.flat_degree1_index(
          decoded_left.support_pair_index,
          merged_flat_sector_index);
  return true;
}

bool merge_left_degree0_with_right_degree1_sector(
    const BoundarySpinBundleLayout& left_degree0_layout,
    int left_degree0_flat_sector_index,
    const BoundarySpinBundleLayout& right_degree1_layout,
    int right_degree1_flat_index,
    const BoundarySpinBundleLayout& merged_layout,
    int* merged_degree1_flat_index,
    int* merge_parity) {
  if (merged_degree1_flat_index == nullptr || merge_parity == nullptr) {
    throw std::invalid_argument(
        "merge_left_degree0_with_right_degree1_sector outputs must not be null");
  }
  const Degree1FlatIndex decoded_right =
      decode_degree1_flat_index(right_degree1_layout, right_degree1_flat_index);
  int merged_flat_sector_index = -1;
  if (!merge_flat_spin_sectors(
          left_degree0_layout,
          left_degree0_flat_sector_index,
          right_degree1_layout,
          decoded_right.flat_sector_index,
          merged_layout,
          &merged_flat_sector_index,
          merge_parity)) {
    return false;
  }
  *merged_degree1_flat_index =
      merged_layout.flat_degree1_index(
          decoded_right.support_pair_index,
          merged_flat_sector_index);
  return true;
}

bool build_same_spin_cross_pair_index(
    int left_support_pair_index,
    int right_support_pair_index,
    int support_size,
    int* support_pair_pair_index,
    int* pair_parity) {
  if (support_pair_pair_index == nullptr || pair_parity == nullptr) {
    throw std::invalid_argument(
        "build_same_spin_cross_pair_index outputs must not be null");
  }
  SupportPair left_pair = decode_support_pair(left_support_pair_index, support_size);
  SupportPair right_pair = decode_support_pair(right_support_pair_index, support_size);
  if (left_pair.row_orbital == right_pair.row_orbital ||
      left_pair.col_orbital == right_pair.col_orbital) {
    return false;
  }

  *pair_parity = 0;
  int row_first = left_pair.row_orbital;
  int row_second = right_pair.row_orbital;
  int col_first = left_pair.col_orbital;
  int col_second = right_pair.col_orbital;
  if (row_first > row_second) {
    std::swap(row_first, row_second);
    *pair_parity ^= 1;
  }
  if (col_first > col_second) {
    std::swap(col_first, col_second);
    *pair_parity ^= 1;
  }
  *support_pair_pair_index = flatten_support_pair_pair_index(
      row_first,
      row_second,
      col_first,
      col_second,
      support_size);
  return true;
}

void validate_spin_bundle_merge_inputs(
    const BoundarySpinBundle& left,
    const BoundarySpinBundle& right,
    const BoundarySpinBundleLayout& merged_layout) {
  if (!has_compatible_layout(left) || !has_compatible_layout(right)) {
    throw std::invalid_argument("input one-spin bundle layout is inconsistent");
  }
  if (left.layout.support_size() != right.layout.support_size() ||
      left.layout.support_size() != merged_layout.support_size()) {
    throw std::invalid_argument("merged one-spin bundles must share support size");
  }
  if (merged_layout.row_count() != left.layout.row_count() + right.layout.row_count() ||
      merged_layout.col_count() != left.layout.col_count() + right.layout.col_count()) {
    throw std::invalid_argument("merged one-spin layout has inconsistent boundary size");
  }
  if (merged_layout.base_selected_row_minus_col() !=
      left.layout.base_selected_row_minus_col() +
          right.layout.base_selected_row_minus_col()) {
    throw std::invalid_argument("merged one-spin layout has inconsistent base family");
  }
}

void validate_spin_bundle_reverse_inputs(
    const BoundarySpinBundle& left,
    const BoundarySpinBundle& right,
    const BoundarySpinBundle& merged_adjoint,
    const BoundarySpinBundle* left_adjoint,
    const BoundarySpinBundle* right_adjoint) {
  if (left_adjoint == nullptr || right_adjoint == nullptr) {
    throw std::invalid_argument("spin bundle reverse outputs must not be null");
  }
  validate_spin_bundle_merge_inputs(left, right, merged_adjoint.layout);
  if (!has_compatible_layout(merged_adjoint) ||
      !has_compatible_layout(*left_adjoint) ||
      !has_compatible_layout(*right_adjoint)) {
    throw std::invalid_argument("spin bundle reverse layout is inconsistent");
  }
  if (!same_spin_layout(left.layout, left_adjoint->layout) ||
      !same_spin_layout(right.layout, right_adjoint->layout)) {
    throw std::invalid_argument("spin bundle reverse output layouts do not match inputs");
  }
}

void validate_boundary_bundle_merge_inputs(
    const BoundaryBundle& left,
    const BoundaryBundle& right,
    const BoundarySpinBundleLayout& merged_alpha_layout,
    const BoundarySpinBundleLayout& merged_beta_layout) {
  if (!has_compatible_layout(left) || !has_compatible_layout(right)) {
    throw std::invalid_argument("input boundary bundle layout is inconsistent");
  }
  validate_spin_bundle_merge_inputs(left.alpha, right.alpha, merged_alpha_layout);
  validate_spin_bundle_merge_inputs(left.beta, right.beta, merged_beta_layout);
}

void validate_boundary_bundle_reverse_inputs(
    const BoundaryBundle& left,
    const BoundaryBundle& right,
    const BoundaryBundle& merged_adjoint,
    const BoundaryBundle* left_adjoint,
    const BoundaryBundle* right_adjoint) {
  if (left_adjoint == nullptr || right_adjoint == nullptr) {
    throw std::invalid_argument("boundary bundle reverse outputs must not be null");
  }
  if (!has_compatible_layout(left) || !has_compatible_layout(right) ||
      !has_compatible_layout(merged_adjoint) ||
      !has_compatible_layout(*left_adjoint) ||
      !has_compatible_layout(*right_adjoint)) {
    throw std::invalid_argument("boundary bundle layout is inconsistent");
  }
  validate_spin_bundle_reverse_inputs(
      left.alpha,
      right.alpha,
      merged_adjoint.alpha,
      &left_adjoint->alpha,
      &right_adjoint->alpha);
  validate_spin_bundle_reverse_inputs(
      left.beta,
      right.beta,
      merged_adjoint.beta,
      &left_adjoint->beta,
      &right_adjoint->beta);
}

void merge_boundary_mixed_bundle(
    const BoundaryBundle& left,
    const BoundaryBundle& right,
    BoundaryBundle* merged) {
  if (merged == nullptr) {
    throw std::invalid_argument("merged bundle must not be null");
  }

  // The mixed alpha/beta degree-1 moment expands into four exact source
  // patterns:
  //   1. left mixed times right overlap,
  //   2. right mixed times left overlap,
  //   3. left alpha-degree1 with right beta-degree1,
  //   4. right alpha-degree1 with left beta-degree1.
  //
  // Each spin channel uses the same exact degree-1/degree-0 boundary-sector
  // merge as the one-electron closure, and the total sign is the product of
  // the alpha and beta merge signs.

  for (int left_alpha_degree1 = 0;
       left_alpha_degree1 < left.mixed.alpha_degree1_entry_count;
       ++left_alpha_degree1) {
    for (int left_beta_degree1 = 0;
         left_beta_degree1 < left.mixed.beta_degree1_entry_count;
         ++left_beta_degree1) {
      const double left_mixed_value = mixed_value(
          left.mixed,
          left_alpha_degree1,
          left_beta_degree1);
      if (std::abs(left_mixed_value) <= kZeroTolerance) {
        continue;
      }
      for (int right_alpha_sector = 0;
           right_alpha_sector < right.alpha.layout.total_sector_count();
           ++right_alpha_sector) {
        const double right_alpha_degree0 =
            right.alpha.degree0[xmvb::to_size(right_alpha_sector)];
        if (std::abs(right_alpha_degree0) <= kZeroTolerance) {
          continue;
        }
        int merged_alpha_degree1 = -1;
        int alpha_parity = 0;
        if (!merge_left_degree1_with_right_degree0_sector(
                left.alpha.layout,
                left_alpha_degree1,
                right.alpha.layout,
                right_alpha_sector,
                merged->alpha.layout,
                &merged_alpha_degree1,
                &alpha_parity)) {
          continue;
        }
        for (int right_beta_sector = 0;
             right_beta_sector < right.beta.layout.total_sector_count();
             ++right_beta_sector) {
          const double right_beta_degree0 =
              right.beta.degree0[xmvb::to_size(right_beta_sector)];
          if (std::abs(right_beta_degree0) <= kZeroTolerance) {
            continue;
          }
          int merged_beta_degree1 = -1;
          int beta_parity = 0;
          if (!merge_left_degree1_with_right_degree0_sector(
                  left.beta.layout,
                  left_beta_degree1,
                  right.beta.layout,
                  right_beta_sector,
                  merged->beta.layout,
                  &merged_beta_degree1,
                  &beta_parity)) {
            continue;
          }
          const double total_sign =
              ((alpha_parity + beta_parity) & 1) == 0 ? 1.0 : -1.0;
          add_mixed_value(
              &merged->mixed,
              merged_alpha_degree1,
              merged_beta_degree1,
              total_sign * left_mixed_value * right_alpha_degree0 * right_beta_degree0);
        }
      }
    }
  }

  for (int right_alpha_degree1 = 0;
       right_alpha_degree1 < right.mixed.alpha_degree1_entry_count;
       ++right_alpha_degree1) {
    for (int right_beta_degree1 = 0;
         right_beta_degree1 < right.mixed.beta_degree1_entry_count;
         ++right_beta_degree1) {
      const double right_mixed_value = mixed_value(
          right.mixed,
          right_alpha_degree1,
          right_beta_degree1);
      if (std::abs(right_mixed_value) <= kZeroTolerance) {
        continue;
      }
      for (int left_alpha_sector = 0;
           left_alpha_sector < left.alpha.layout.total_sector_count();
           ++left_alpha_sector) {
        const double left_alpha_degree0 =
            left.alpha.degree0[xmvb::to_size(left_alpha_sector)];
        if (std::abs(left_alpha_degree0) <= kZeroTolerance) {
          continue;
        }
        int merged_alpha_degree1 = -1;
        int alpha_parity = 0;
        if (!merge_left_degree0_with_right_degree1_sector(
                left.alpha.layout,
                left_alpha_sector,
                right.alpha.layout,
                right_alpha_degree1,
                merged->alpha.layout,
                &merged_alpha_degree1,
                &alpha_parity)) {
          continue;
        }
        for (int left_beta_sector = 0;
             left_beta_sector < left.beta.layout.total_sector_count();
             ++left_beta_sector) {
          const double left_beta_degree0 =
              left.beta.degree0[xmvb::to_size(left_beta_sector)];
          if (std::abs(left_beta_degree0) <= kZeroTolerance) {
            continue;
          }
          int merged_beta_degree1 = -1;
          int beta_parity = 0;
          if (!merge_left_degree0_with_right_degree1_sector(
                  left.beta.layout,
                  left_beta_sector,
                  right.beta.layout,
                  right_beta_degree1,
                  merged->beta.layout,
                  &merged_beta_degree1,
                  &beta_parity)) {
            continue;
          }
          const double total_sign =
              ((alpha_parity + beta_parity) & 1) == 0 ? 1.0 : -1.0;
          add_mixed_value(
              &merged->mixed,
              merged_alpha_degree1,
              merged_beta_degree1,
              total_sign * left_alpha_degree0 * left_beta_degree0 * right_mixed_value);
        }
      }
    }
  }

  for (int left_alpha_degree1 = 0;
       left_alpha_degree1 < degree1_entry_count(left.alpha.layout);
       ++left_alpha_degree1) {
    const double left_alpha_value =
        left.alpha.degree1[xmvb::to_size(left_alpha_degree1)];
    if (std::abs(left_alpha_value) <= kZeroTolerance) {
      continue;
    }
    for (int right_alpha_sector = 0;
         right_alpha_sector < right.alpha.layout.total_sector_count();
         ++right_alpha_sector) {
      const double right_alpha_degree0 =
          right.alpha.degree0[xmvb::to_size(right_alpha_sector)];
      if (std::abs(right_alpha_degree0) <= kZeroTolerance) {
        continue;
      }
      int merged_alpha_degree1 = -1;
      int alpha_parity = 0;
      if (!merge_left_degree1_with_right_degree0_sector(
              left.alpha.layout,
              left_alpha_degree1,
              right.alpha.layout,
              right_alpha_sector,
              merged->alpha.layout,
              &merged_alpha_degree1,
              &alpha_parity)) {
        continue;
      }
      for (int left_beta_sector = 0;
           left_beta_sector < left.beta.layout.total_sector_count();
           ++left_beta_sector) {
        const double left_beta_degree0 =
            left.beta.degree0[xmvb::to_size(left_beta_sector)];
        if (std::abs(left_beta_degree0) <= kZeroTolerance) {
          continue;
        }
        for (int right_beta_degree1 = 0;
             right_beta_degree1 < degree1_entry_count(right.beta.layout);
             ++right_beta_degree1) {
          const double right_beta_value =
              right.beta.degree1[xmvb::to_size(right_beta_degree1)];
          if (std::abs(right_beta_value) <= kZeroTolerance) {
            continue;
          }
          int merged_beta_degree1 = -1;
          int beta_parity = 0;
          if (!merge_left_degree0_with_right_degree1_sector(
                  left.beta.layout,
                  left_beta_sector,
                  right.beta.layout,
                  right_beta_degree1,
                  merged->beta.layout,
                  &merged_beta_degree1,
                  &beta_parity)) {
            continue;
          }
          const double total_sign =
              ((alpha_parity + beta_parity) & 1) == 0 ? 1.0 : -1.0;
          add_mixed_value(
              &merged->mixed,
              merged_alpha_degree1,
              merged_beta_degree1,
              total_sign *
                  left_alpha_value *
                  left_beta_degree0 *
                  right_alpha_degree0 *
                  right_beta_value);
        }
      }
    }
  }

  for (int left_alpha_sector = 0;
       left_alpha_sector < left.alpha.layout.total_sector_count();
       ++left_alpha_sector) {
    const double left_alpha_degree0 =
        left.alpha.degree0[xmvb::to_size(left_alpha_sector)];
    if (std::abs(left_alpha_degree0) <= kZeroTolerance) {
      continue;
    }
    for (int right_alpha_degree1 = 0;
         right_alpha_degree1 < degree1_entry_count(right.alpha.layout);
         ++right_alpha_degree1) {
      const double right_alpha_value =
          right.alpha.degree1[xmvb::to_size(right_alpha_degree1)];
      if (std::abs(right_alpha_value) <= kZeroTolerance) {
        continue;
      }
      int merged_alpha_degree1 = -1;
      int alpha_parity = 0;
      if (!merge_left_degree0_with_right_degree1_sector(
              left.alpha.layout,
              left_alpha_sector,
              right.alpha.layout,
              right_alpha_degree1,
              merged->alpha.layout,
              &merged_alpha_degree1,
              &alpha_parity)) {
        continue;
      }
      for (int left_beta_degree1 = 0;
           left_beta_degree1 < degree1_entry_count(left.beta.layout);
           ++left_beta_degree1) {
        const double left_beta_value =
            left.beta.degree1[xmvb::to_size(left_beta_degree1)];
        if (std::abs(left_beta_value) <= kZeroTolerance) {
          continue;
        }
        for (int right_beta_sector = 0;
             right_beta_sector < right.beta.layout.total_sector_count();
             ++right_beta_sector) {
          const double right_beta_degree0 =
              right.beta.degree0[xmvb::to_size(right_beta_sector)];
          if (std::abs(right_beta_degree0) <= kZeroTolerance) {
            continue;
          }
          int merged_beta_degree1 = -1;
          int beta_parity = 0;
          if (!merge_left_degree1_with_right_degree0_sector(
                  left.beta.layout,
                  left_beta_degree1,
                  right.beta.layout,
                  right_beta_sector,
                  merged->beta.layout,
                  &merged_beta_degree1,
                  &beta_parity)) {
            continue;
          }
          const double total_sign =
              ((alpha_parity + beta_parity) & 1) == 0 ? 1.0 : -1.0;
          add_mixed_value(
              &merged->mixed,
              merged_alpha_degree1,
              merged_beta_degree1,
              total_sign *
                  left_alpha_degree0 *
                  left_beta_value *
                  right_alpha_value *
                  right_beta_degree0);
        }
      }
    }
  }
}

void reverse_merge_boundary_mixed_bundle(
    const BoundaryBundle& left,
    const BoundaryBundle& right,
    const BoundaryBundle& merged_adjoint,
    BoundaryBundle* left_adjoint,
    BoundaryBundle* right_adjoint) {
  if (left_adjoint == nullptr || right_adjoint == nullptr) {
    throw std::invalid_argument("boundary mixed reverse outputs must not be null");
  }

  for (int left_alpha_degree1 = 0;
       left_alpha_degree1 < left.mixed.alpha_degree1_entry_count;
       ++left_alpha_degree1) {
    for (int left_beta_degree1 = 0;
         left_beta_degree1 < left.mixed.beta_degree1_entry_count;
         ++left_beta_degree1) {
      const double left_mixed_value = mixed_value(
          left.mixed,
          left_alpha_degree1,
          left_beta_degree1);
      if (std::abs(left_mixed_value) <= kZeroTolerance) {
        continue;
      }
      for (int right_alpha_sector = 0;
           right_alpha_sector < right.alpha.layout.total_sector_count();
           ++right_alpha_sector) {
        const double right_alpha_degree0 =
            right.alpha.degree0[xmvb::to_size(right_alpha_sector)];
        if (std::abs(right_alpha_degree0) <= kZeroTolerance) {
          continue;
        }
        int merged_alpha_degree1 = -1;
        int alpha_parity = 0;
        if (!merge_left_degree1_with_right_degree0_sector(
                left.alpha.layout,
                left_alpha_degree1,
                right.alpha.layout,
                right_alpha_sector,
                merged_adjoint.alpha.layout,
                &merged_alpha_degree1,
                &alpha_parity)) {
          continue;
        }
        for (int right_beta_sector = 0;
             right_beta_sector < right.beta.layout.total_sector_count();
             ++right_beta_sector) {
          const double right_beta_degree0 =
              right.beta.degree0[xmvb::to_size(right_beta_sector)];
          if (std::abs(right_beta_degree0) <= kZeroTolerance) {
            continue;
          }
          int merged_beta_degree1 = -1;
          int beta_parity = 0;
          if (!merge_left_degree1_with_right_degree0_sector(
                  left.beta.layout,
                  left_beta_degree1,
                  right.beta.layout,
                  right_beta_sector,
                  merged_adjoint.beta.layout,
                  &merged_beta_degree1,
                  &beta_parity)) {
            continue;
          }
          const double total_sign =
              ((alpha_parity + beta_parity) & 1) == 0 ? 1.0 : -1.0;
          const double merged_value = mixed_value(
              merged_adjoint.mixed,
              merged_alpha_degree1,
              merged_beta_degree1);
          if (std::abs(merged_value) <= kZeroTolerance) {
            continue;
          }
          add_mixed_value(
              &left_adjoint->mixed,
              left_alpha_degree1,
              left_beta_degree1,
              total_sign * merged_value * right_alpha_degree0 * right_beta_degree0);
          right_adjoint->alpha.degree0[xmvb::to_size(right_alpha_sector)] +=
              total_sign * merged_value * left_mixed_value * right_beta_degree0;
          right_adjoint->beta.degree0[xmvb::to_size(right_beta_sector)] +=
              total_sign * merged_value * left_mixed_value * right_alpha_degree0;
        }
      }
    }
  }

  for (int right_alpha_degree1 = 0;
       right_alpha_degree1 < right.mixed.alpha_degree1_entry_count;
       ++right_alpha_degree1) {
    for (int right_beta_degree1 = 0;
         right_beta_degree1 < right.mixed.beta_degree1_entry_count;
         ++right_beta_degree1) {
      const double right_mixed_value = mixed_value(
          right.mixed,
          right_alpha_degree1,
          right_beta_degree1);
      if (std::abs(right_mixed_value) <= kZeroTolerance) {
        continue;
      }
      for (int left_alpha_sector = 0;
           left_alpha_sector < left.alpha.layout.total_sector_count();
           ++left_alpha_sector) {
        const double left_alpha_degree0 =
            left.alpha.degree0[xmvb::to_size(left_alpha_sector)];
        if (std::abs(left_alpha_degree0) <= kZeroTolerance) {
          continue;
        }
        int merged_alpha_degree1 = -1;
        int alpha_parity = 0;
        if (!merge_left_degree0_with_right_degree1_sector(
                left.alpha.layout,
                left_alpha_sector,
                right.alpha.layout,
                right_alpha_degree1,
                merged_adjoint.alpha.layout,
                &merged_alpha_degree1,
                &alpha_parity)) {
          continue;
        }
        for (int left_beta_sector = 0;
             left_beta_sector < left.beta.layout.total_sector_count();
             ++left_beta_sector) {
          const double left_beta_degree0 =
              left.beta.degree0[xmvb::to_size(left_beta_sector)];
          if (std::abs(left_beta_degree0) <= kZeroTolerance) {
            continue;
          }
          int merged_beta_degree1 = -1;
          int beta_parity = 0;
          if (!merge_left_degree0_with_right_degree1_sector(
                  left.beta.layout,
                  left_beta_sector,
                  right.beta.layout,
                  right_beta_degree1,
                  merged_adjoint.beta.layout,
                  &merged_beta_degree1,
                  &beta_parity)) {
            continue;
          }
          const double total_sign =
              ((alpha_parity + beta_parity) & 1) == 0 ? 1.0 : -1.0;
          const double merged_value = mixed_value(
              merged_adjoint.mixed,
              merged_alpha_degree1,
              merged_beta_degree1);
          if (std::abs(merged_value) <= kZeroTolerance) {
            continue;
          }
          add_mixed_value(
              &right_adjoint->mixed,
              right_alpha_degree1,
              right_beta_degree1,
              total_sign * merged_value * left_alpha_degree0 * left_beta_degree0);
          left_adjoint->alpha.degree0[xmvb::to_size(left_alpha_sector)] +=
              total_sign * merged_value * left_beta_degree0 * right_mixed_value;
          left_adjoint->beta.degree0[xmvb::to_size(left_beta_sector)] +=
              total_sign * merged_value * left_alpha_degree0 * right_mixed_value;
        }
      }
    }
  }

  for (int left_alpha_degree1 = 0;
       left_alpha_degree1 < degree1_entry_count(left.alpha.layout);
       ++left_alpha_degree1) {
    const double left_alpha_value =
        left.alpha.degree1[xmvb::to_size(left_alpha_degree1)];
    if (std::abs(left_alpha_value) <= kZeroTolerance) {
      continue;
    }
    for (int right_alpha_sector = 0;
         right_alpha_sector < right.alpha.layout.total_sector_count();
         ++right_alpha_sector) {
      const double right_alpha_degree0 =
          right.alpha.degree0[xmvb::to_size(right_alpha_sector)];
      if (std::abs(right_alpha_degree0) <= kZeroTolerance) {
        continue;
      }
      int merged_alpha_degree1 = -1;
      int alpha_parity = 0;
      if (!merge_left_degree1_with_right_degree0_sector(
              left.alpha.layout,
              left_alpha_degree1,
              right.alpha.layout,
              right_alpha_sector,
              merged_adjoint.alpha.layout,
              &merged_alpha_degree1,
              &alpha_parity)) {
        continue;
      }
      for (int left_beta_sector = 0;
           left_beta_sector < left.beta.layout.total_sector_count();
           ++left_beta_sector) {
        const double left_beta_degree0 =
            left.beta.degree0[xmvb::to_size(left_beta_sector)];
        if (std::abs(left_beta_degree0) <= kZeroTolerance) {
          continue;
        }
        for (int right_beta_degree1 = 0;
             right_beta_degree1 < degree1_entry_count(right.beta.layout);
             ++right_beta_degree1) {
          const double right_beta_value =
              right.beta.degree1[xmvb::to_size(right_beta_degree1)];
          if (std::abs(right_beta_value) <= kZeroTolerance) {
            continue;
          }
          int merged_beta_degree1 = -1;
          int beta_parity = 0;
          if (!merge_left_degree0_with_right_degree1_sector(
                  left.beta.layout,
                  left_beta_sector,
                  right.beta.layout,
                  right_beta_degree1,
                  merged_adjoint.beta.layout,
                  &merged_beta_degree1,
                  &beta_parity)) {
            continue;
          }
          const double total_sign =
              ((alpha_parity + beta_parity) & 1) == 0 ? 1.0 : -1.0;
          const double merged_value = mixed_value(
              merged_adjoint.mixed,
              merged_alpha_degree1,
              merged_beta_degree1);
          if (std::abs(merged_value) <= kZeroTolerance) {
            continue;
          }
          left_adjoint->alpha.degree1[xmvb::to_size(left_alpha_degree1)] +=
              total_sign *
              merged_value *
              left_beta_degree0 *
              right_alpha_degree0 *
              right_beta_value;
          left_adjoint->beta.degree0[xmvb::to_size(left_beta_sector)] +=
              total_sign *
              merged_value *
              left_alpha_value *
              right_alpha_degree0 *
              right_beta_value;
          right_adjoint->alpha.degree0[xmvb::to_size(right_alpha_sector)] +=
              total_sign *
              merged_value *
              left_alpha_value *
              left_beta_degree0 *
              right_beta_value;
          right_adjoint->beta.degree1[xmvb::to_size(right_beta_degree1)] +=
              total_sign *
              merged_value *
              left_alpha_value *
              left_beta_degree0 *
              right_alpha_degree0;
        }
      }
    }
  }

  for (int left_alpha_sector = 0;
       left_alpha_sector < left.alpha.layout.total_sector_count();
       ++left_alpha_sector) {
    const double left_alpha_degree0 =
        left.alpha.degree0[xmvb::to_size(left_alpha_sector)];
    if (std::abs(left_alpha_degree0) <= kZeroTolerance) {
      continue;
    }
    for (int right_alpha_degree1 = 0;
         right_alpha_degree1 < degree1_entry_count(right.alpha.layout);
         ++right_alpha_degree1) {
      const double right_alpha_value =
          right.alpha.degree1[xmvb::to_size(right_alpha_degree1)];
      if (std::abs(right_alpha_value) <= kZeroTolerance) {
        continue;
      }
      int merged_alpha_degree1 = -1;
      int alpha_parity = 0;
      if (!merge_left_degree0_with_right_degree1_sector(
              left.alpha.layout,
              left_alpha_sector,
              right.alpha.layout,
              right_alpha_degree1,
              merged_adjoint.alpha.layout,
              &merged_alpha_degree1,
              &alpha_parity)) {
        continue;
      }
      for (int left_beta_degree1 = 0;
           left_beta_degree1 < degree1_entry_count(left.beta.layout);
           ++left_beta_degree1) {
        const double left_beta_value =
            left.beta.degree1[xmvb::to_size(left_beta_degree1)];
        if (std::abs(left_beta_value) <= kZeroTolerance) {
          continue;
        }
        for (int right_beta_sector = 0;
             right_beta_sector < right.beta.layout.total_sector_count();
             ++right_beta_sector) {
          const double right_beta_degree0 =
              right.beta.degree0[xmvb::to_size(right_beta_sector)];
          if (std::abs(right_beta_degree0) <= kZeroTolerance) {
            continue;
          }
          int merged_beta_degree1 = -1;
          int beta_parity = 0;
          if (!merge_left_degree1_with_right_degree0_sector(
                  left.beta.layout,
                  left_beta_degree1,
                  right.beta.layout,
                  right_beta_sector,
                  merged_adjoint.beta.layout,
                  &merged_beta_degree1,
                  &beta_parity)) {
            continue;
          }
          const double total_sign =
              ((alpha_parity + beta_parity) & 1) == 0 ? 1.0 : -1.0;
          const double merged_value = mixed_value(
              merged_adjoint.mixed,
              merged_alpha_degree1,
              merged_beta_degree1);
          if (std::abs(merged_value) <= kZeroTolerance) {
            continue;
          }
          left_adjoint->alpha.degree0[xmvb::to_size(left_alpha_sector)] +=
              total_sign *
              merged_value *
              left_beta_value *
              right_alpha_value *
              right_beta_degree0;
          left_adjoint->beta.degree1[xmvb::to_size(left_beta_degree1)] +=
              total_sign *
              merged_value *
              left_alpha_degree0 *
              right_alpha_value *
              right_beta_degree0;
          right_adjoint->alpha.degree1[xmvb::to_size(right_alpha_degree1)] +=
              total_sign *
              merged_value *
              left_alpha_degree0 *
              left_beta_value *
              right_beta_degree0;
          right_adjoint->beta.degree0[xmvb::to_size(right_beta_sector)] +=
              total_sign *
              merged_value *
              left_alpha_degree0 *
              left_beta_value *
              right_alpha_value;
        }
      }
    }
  }
}

}  // namespace

BoundarySpinBundleLayout make_merged_boundary_spin_bundle_layout(
    const BoundarySpinBundleLayout& left,
    const BoundarySpinBundleLayout& right,
    int min_family_delta,
    int max_family_delta) {
  if (left.support_size() != right.support_size()) {
    throw std::invalid_argument("merged one-spin layout requires equal support sizes");
  }
  return BoundarySpinBundleLayout(
      left.row_count() + right.row_count(),
      left.col_count() + right.col_count(),
      left.base_selected_row_minus_col() + right.base_selected_row_minus_col(),
      min_family_delta,
      max_family_delta,
      left.support_size());
}

BoundarySpinBundleLayout make_full_merged_boundary_spin_bundle_layout(
    const BoundarySpinBundleLayout& left,
    const BoundarySpinBundleLayout& right) {
  return make_merged_boundary_spin_bundle_layout(
      left,
      right,
      left.min_family_delta() + right.min_family_delta(),
      left.max_family_delta() + right.max_family_delta());
}

BoundarySpinBundle merge_boundary_spin_bundles(
    const BoundarySpinBundle& left,
    const BoundarySpinBundle& right,
    const BoundarySpinBundleLayout& merged_layout) {
  validate_spin_bundle_merge_inputs(left, right, merged_layout);
  BoundarySpinBundle merged = make_zero_boundary_spin_bundle(merged_layout);

  for (int left_flat_sector = 0;
       left_flat_sector < left.layout.total_sector_count();
       ++left_flat_sector) {
    const double left_degree0 = left.degree0[xmvb::to_size(left_flat_sector)];
    for (int right_flat_sector = 0;
         right_flat_sector < right.layout.total_sector_count();
         ++right_flat_sector) {
      const double right_degree0 =
          right.degree0[xmvb::to_size(right_flat_sector)];
      int merged_flat_sector = -1;
      int merge_parity = 0;
      if (!merge_flat_spin_sectors(
              left.layout,
              left_flat_sector,
              right.layout,
              right_flat_sector,
              merged_layout,
              &merged_flat_sector,
              &merge_parity)) {
        continue;
      }
      const double merge_sign = (merge_parity & 1) == 0 ? 1.0 : -1.0;

      merged.degree0[xmvb::to_size(merged_flat_sector)] +=
          merge_sign * left_degree0 * right_degree0;

      for (int support_pair_index = 0;
           support_pair_index < merged_layout.support_pair_count();
           ++support_pair_index) {
        const int left_degree1_index = left.layout.flat_degree1_index(
            support_pair_index,
            left_flat_sector);
        const int right_degree1_index = right.layout.flat_degree1_index(
            support_pair_index,
            right_flat_sector);
        const int merged_degree1_index = merged_layout.flat_degree1_index(
            support_pair_index,
            merged_flat_sector);
        merged.degree1[xmvb::to_size(merged_degree1_index)] +=
            merge_sign *
            (left.degree1[xmvb::to_size(left_degree1_index)] * right_degree0 +
             left_degree0 *
                 right.degree1[xmvb::to_size(right_degree1_index)]);
      }

      for (int support_pair_pair_index = 0;
           support_pair_pair_index < merged_layout.support_pair_pair_count();
           ++support_pair_pair_index) {
        const int left_degree2_index = left.layout.flat_degree2_index(
            support_pair_pair_index,
            left_flat_sector);
        const int right_degree2_index = right.layout.flat_degree2_index(
            support_pair_pair_index,
            right_flat_sector);
        const int merged_degree2_index = merged_layout.flat_degree2_index(
            support_pair_pair_index,
            merged_flat_sector);
        merged.degree2[xmvb::to_size(merged_degree2_index)] +=
            merge_sign *
            (left.degree2[xmvb::to_size(left_degree2_index)] * right_degree0 +
             left_degree0 *
                 right.degree2[xmvb::to_size(right_degree2_index)]);
      }

      for (int left_support_pair_index = 0;
           left_support_pair_index < left.layout.support_pair_count();
           ++left_support_pair_index) {
        const double left_value = left.degree1[xmvb::to_size(
            left.layout.flat_degree1_index(left_support_pair_index, left_flat_sector))];
        if (std::abs(left_value) <= kZeroTolerance) {
          continue;
        }
        for (int right_support_pair_index = 0;
             right_support_pair_index < right.layout.support_pair_count();
             ++right_support_pair_index) {
          const double right_value = right.degree1[xmvb::to_size(
              right.layout.flat_degree1_index(right_support_pair_index, right_flat_sector))];
          if (std::abs(right_value) <= kZeroTolerance) {
            continue;
          }
          int support_pair_pair_index = -1;
          int pair_parity = 0;
          if (!build_same_spin_cross_pair_index(
                  left_support_pair_index,
                  right_support_pair_index,
                  merged_layout.support_size(),
                  &support_pair_pair_index,
                  &pair_parity)) {
            continue;
          }
          const double pair_sign = (pair_parity & 1) == 0 ? 1.0 : -1.0;
          const int merged_degree2_index = merged_layout.flat_degree2_index(
              support_pair_pair_index,
              merged_flat_sector);
          merged.degree2[xmvb::to_size(merged_degree2_index)] +=
              merge_sign * pair_sign * left_value * right_value;
        }
      }
    }
  }

  return merged;
}

void reverse_merge_boundary_spin_bundles(
    const BoundarySpinBundle& left,
    const BoundarySpinBundle& right,
    const BoundarySpinBundle& merged_adjoint,
    BoundarySpinBundle* left_adjoint,
    BoundarySpinBundle* right_adjoint) {
  validate_spin_bundle_reverse_inputs(
      left,
      right,
      merged_adjoint,
      left_adjoint,
      right_adjoint);

  for (int left_flat_sector = 0;
       left_flat_sector < left.layout.total_sector_count();
       ++left_flat_sector) {
    const double left_degree0 = left.degree0[xmvb::to_size(left_flat_sector)];
    for (int right_flat_sector = 0;
         right_flat_sector < right.layout.total_sector_count();
         ++right_flat_sector) {
      const double right_degree0 =
          right.degree0[xmvb::to_size(right_flat_sector)];
      int merged_flat_sector = -1;
      int merge_parity = 0;
      if (!merge_flat_spin_sectors(
              left.layout,
              left_flat_sector,
              right.layout,
              right_flat_sector,
              merged_adjoint.layout,
              &merged_flat_sector,
              &merge_parity)) {
        continue;
      }
      const double merge_sign = (merge_parity & 1) == 0 ? 1.0 : -1.0;

      const double degree0_adjoint =
          merged_adjoint.degree0[xmvb::to_size(merged_flat_sector)];
      left_adjoint->degree0[xmvb::to_size(left_flat_sector)] +=
          merge_sign * degree0_adjoint * right_degree0;
      right_adjoint->degree0[xmvb::to_size(right_flat_sector)] +=
          merge_sign * degree0_adjoint * left_degree0;

      for (int support_pair_index = 0;
           support_pair_index < left.layout.support_pair_count();
           ++support_pair_index) {
        const int left_degree1_index = left.layout.flat_degree1_index(
            support_pair_index,
            left_flat_sector);
        const int right_degree1_index = right.layout.flat_degree1_index(
            support_pair_index,
            right_flat_sector);
        const int merged_degree1_index = merged_adjoint.layout.flat_degree1_index(
            support_pair_index,
            merged_flat_sector);
        const double merged_value =
            merged_adjoint.degree1[xmvb::to_size(merged_degree1_index)];
        const double left_value =
            left.degree1[xmvb::to_size(left_degree1_index)];
        const double right_value =
            right.degree1[xmvb::to_size(right_degree1_index)];
        left_adjoint->degree1[xmvb::to_size(left_degree1_index)] +=
            merge_sign * merged_value * right_degree0;
        right_adjoint->degree1[xmvb::to_size(right_degree1_index)] +=
            merge_sign * merged_value * left_degree0;
        left_adjoint->degree0[xmvb::to_size(left_flat_sector)] +=
            merge_sign * merged_value * right_value;
        right_adjoint->degree0[xmvb::to_size(right_flat_sector)] +=
            merge_sign * merged_value * left_value;
      }

      for (int support_pair_pair_index = 0;
           support_pair_pair_index < left.layout.support_pair_pair_count();
           ++support_pair_pair_index) {
        const int left_degree2_index = left.layout.flat_degree2_index(
            support_pair_pair_index,
            left_flat_sector);
        const int right_degree2_index = right.layout.flat_degree2_index(
            support_pair_pair_index,
            right_flat_sector);
        const int merged_degree2_index = merged_adjoint.layout.flat_degree2_index(
            support_pair_pair_index,
            merged_flat_sector);
        const double merged_value =
            merged_adjoint.degree2[xmvb::to_size(merged_degree2_index)];
        const double left_value =
            left.degree2[xmvb::to_size(left_degree2_index)];
        const double right_value =
            right.degree2[xmvb::to_size(right_degree2_index)];
        left_adjoint->degree2[xmvb::to_size(left_degree2_index)] +=
            merge_sign * merged_value * right_degree0;
        right_adjoint->degree2[xmvb::to_size(right_degree2_index)] +=
            merge_sign * merged_value * left_degree0;
        left_adjoint->degree0[xmvb::to_size(left_flat_sector)] +=
            merge_sign * merged_value * right_value;
        right_adjoint->degree0[xmvb::to_size(right_flat_sector)] +=
            merge_sign * merged_value * left_value;
      }

      for (int left_support_pair_index = 0;
           left_support_pair_index < left.layout.support_pair_count();
           ++left_support_pair_index) {
        const int left_degree1_index = left.layout.flat_degree1_index(
            left_support_pair_index,
            left_flat_sector);
        const double left_value =
            left.degree1[xmvb::to_size(left_degree1_index)];
        if (std::abs(left_value) <= kZeroTolerance) {
          continue;
        }
        for (int right_support_pair_index = 0;
             right_support_pair_index < right.layout.support_pair_count();
             ++right_support_pair_index) {
          const int right_degree1_index = right.layout.flat_degree1_index(
              right_support_pair_index,
              right_flat_sector);
          const double right_value =
              right.degree1[xmvb::to_size(right_degree1_index)];
          if (std::abs(right_value) <= kZeroTolerance) {
            continue;
          }
          int support_pair_pair_index = -1;
          int pair_parity = 0;
          if (!build_same_spin_cross_pair_index(
                  left_support_pair_index,
                  right_support_pair_index,
                  merged_adjoint.layout.support_size(),
                  &support_pair_pair_index,
                  &pair_parity)) {
            continue;
          }
          const int merged_degree2_index = merged_adjoint.layout.flat_degree2_index(
              support_pair_pair_index,
              merged_flat_sector);
          const double merged_value =
              merged_adjoint.degree2[xmvb::to_size(merged_degree2_index)];
          const double total_sign =
              merge_sign * ((pair_parity & 1) == 0 ? 1.0 : -1.0);
          left_adjoint->degree1[xmvb::to_size(left_degree1_index)] +=
              total_sign * merged_value * right_value;
          right_adjoint->degree1[xmvb::to_size(right_degree1_index)] +=
              total_sign * merged_value * left_value;
        }
      }
    }
  }
}

BoundaryBundle merge_boundary_bundles(
    const BoundaryBundle& left,
    const BoundaryBundle& right,
    const BoundarySpinBundleLayout& merged_alpha_layout,
    const BoundarySpinBundleLayout& merged_beta_layout) {
  validate_boundary_bundle_merge_inputs(
      left,
      right,
      merged_alpha_layout,
      merged_beta_layout);

  BoundaryBundle merged = make_zero_boundary_bundle(
      merged_alpha_layout,
      merged_beta_layout);
  merged.alpha = merge_boundary_spin_bundles(
      left.alpha,
      right.alpha,
      merged_alpha_layout);
  merged.beta = merge_boundary_spin_bundles(
      left.beta,
      right.beta,
      merged_beta_layout);
  merge_boundary_mixed_bundle(left, right, &merged);
  return merged;
}

void reverse_merge_boundary_bundles(
    const BoundaryBundle& left,
    const BoundaryBundle& right,
    const BoundaryBundle& merged_adjoint,
    BoundaryBundle* left_adjoint,
    BoundaryBundle* right_adjoint) {
  validate_boundary_bundle_reverse_inputs(
      left,
      right,
      merged_adjoint,
      left_adjoint,
      right_adjoint);

  reverse_merge_boundary_spin_bundles(
      left.alpha,
      right.alpha,
      merged_adjoint.alpha,
      &left_adjoint->alpha,
      &right_adjoint->alpha);
  reverse_merge_boundary_spin_bundles(
      left.beta,
      right.beta,
      merged_adjoint.beta,
      &left_adjoint->beta,
      &right_adjoint->beta);
  reverse_merge_boundary_mixed_bundle(
      left,
      right,
      merged_adjoint,
      left_adjoint,
      right_adjoint);
}

}  // namespace xmvb::vb::exact_separator
