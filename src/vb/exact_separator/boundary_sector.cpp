#include "vb/exact_separator/boundary_sector.hpp"

#include <stdexcept>

namespace xmvb::vb::exact_separator {

namespace {

std::uint32_t mask_limit(int n_bits) {
  if (n_bits < 0 || n_bits >= 31) {
    throw std::invalid_argument("boundary sector width is out of range");
  }
  return (n_bits == 0) ? 1U : (static_cast<std::uint32_t>(1U) << n_bits);
}

std::uint32_t full_mask(int n_bits) {
  const std::uint32_t limit = mask_limit(n_bits);
  return (limit == 1U) ? 0U : (limit - 1U);
}

int popcount(std::uint32_t mask) {
  return __builtin_popcount(mask);
}

std::size_t packed_index(
    std::uint32_t row_mask,
    std::uint32_t col_mask,
    std::uint32_t col_limit) {
  return xmvb::to_size(row_mask) * xmvb::to_size(col_limit) +
      xmvb::to_size(col_mask);
}

}  // namespace

BoundarySectorIndexer::BoundarySectorIndexer(
    int row_count,
    int col_count,
    int selected_row_minus_col)
    : row_count_(row_count),
      col_count_(col_count),
      selected_row_minus_col_(selected_row_minus_col),
      full_row_mask_(full_mask(row_count)),
      full_col_mask_(full_mask(col_count)) {
  if (row_count_ < 0 || col_count_ < 0) {
    throw std::invalid_argument("boundary sector counts must be non-negative");
  }

  const std::uint32_t row_limit = mask_limit(row_count_);
  const std::uint32_t col_limit = mask_limit(col_count_);
  packed_index_table_.assign(
      xmvb::to_size(row_limit) * xmvb::to_size(col_limit),
      -1);

  for (int selected_rows = 0; selected_rows <= row_count_; ++selected_rows) {
    const int selected_cols = selected_rows - selected_row_minus_col_;
    if (selected_cols < 0 || selected_cols > col_count_) {
      continue;
    }

    for (std::uint32_t row_mask = 0; row_mask < row_limit; ++row_mask) {
      if (popcount(row_mask) != selected_rows) {
        continue;
      }
      for (std::uint32_t col_mask = 0; col_mask < col_limit; ++col_mask) {
        if (popcount(col_mask) != selected_cols) {
          continue;
        }
        packed_index_table_[packed_index(row_mask, col_mask, col_limit)] =
            static_cast<int>(sectors_.size());
        sectors_.push_back(BoundarySector{
            .row_mask = row_mask,
            .col_mask = col_mask,
            .selected_row_count = selected_rows,
            .selected_col_count = selected_cols,
        });
      }
    }
  }
}

int BoundarySectorIndexer::row_count() const {
  return row_count_;
}

int BoundarySectorIndexer::col_count() const {
  return col_count_;
}

int BoundarySectorIndexer::selected_row_minus_col() const {
  return selected_row_minus_col_;
}

int BoundarySectorIndexer::sector_count() const {
  return static_cast<int>(sectors_.size());
}

std::uint32_t BoundarySectorIndexer::full_row_mask() const {
  return full_row_mask_;
}

std::uint32_t BoundarySectorIndexer::full_col_mask() const {
  return full_col_mask_;
}

const BoundarySector& BoundarySectorIndexer::sector(int index) const {
  if (index < 0 || index >= sector_count()) {
    throw std::out_of_range("boundary sector index is out of range");
  }
  return sectors_[xmvb::to_size(index)];
}

int BoundarySectorIndexer::index(
    std::uint32_t row_mask,
    std::uint32_t col_mask) const {
  const std::uint32_t row_limit = mask_limit(row_count_);
  const std::uint32_t col_limit = mask_limit(col_count_);
  if (row_mask >= row_limit || col_mask >= col_limit) {
    return -1;
  }
  return packed_index_table_[packed_index(row_mask, col_mask, col_limit)];
}

int BoundarySectorIndexer::complement_index(int index) const {
  const BoundarySector& sector_value = sector(index);
  return this->index(
      full_row_mask_ ^ sector_value.row_mask,
      full_col_mask_ ^ sector_value.col_mask);
}

BoundarySectorFamilyIndexerRange::BoundarySectorFamilyIndexerRange(
    int row_count,
    int col_count,
    int base_selected_row_minus_col,
    int min_family_delta,
    int max_family_delta)
    : row_count_(row_count),
      col_count_(col_count),
      base_selected_row_minus_col_(base_selected_row_minus_col),
      min_family_delta_(min_family_delta),
      max_family_delta_(max_family_delta) {
  if (row_count_ < 0 || col_count_ < 0) {
    throw std::invalid_argument("boundary family counts must be non-negative");
  }
  if (min_family_delta_ > max_family_delta_) {
    throw std::invalid_argument("boundary family delta range is invalid");
  }

  families_.reserve(
      xmvb::to_size(max_family_delta_ - min_family_delta_ + 1));
  for (int family_delta = min_family_delta_;
       family_delta <= max_family_delta_;
       ++family_delta) {
    BoundarySectorFamilyIndexer family;
    family.family_delta = family_delta;
    family.sector_indexer = BoundarySectorIndexer(
        row_count_,
        col_count_,
        base_selected_row_minus_col_ + family_delta);
    if (family.sector_indexer.sector_count() == 0) {
      continue;
    }
    families_.push_back(std::move(family));
  }
}

int BoundarySectorFamilyIndexerRange::row_count() const {
  return row_count_;
}

int BoundarySectorFamilyIndexerRange::col_count() const {
  return col_count_;
}

int BoundarySectorFamilyIndexerRange::base_selected_row_minus_col() const {
  return base_selected_row_minus_col_;
}

int BoundarySectorFamilyIndexerRange::min_family_delta() const {
  return min_family_delta_;
}

int BoundarySectorFamilyIndexerRange::max_family_delta() const {
  return max_family_delta_;
}

int BoundarySectorFamilyIndexerRange::family_count() const {
  return static_cast<int>(families_.size());
}

const BoundarySectorFamilyIndexer& BoundarySectorFamilyIndexerRange::family(
    int index) const {
  if (index < 0 || index >= family_count()) {
    throw std::out_of_range("boundary family index is out of range");
  }
  return families_[xmvb::to_size(index)];
}

int BoundarySectorFamilyIndexerRange::family_index(int family_delta) const {
  for (int family_index_value = 0;
       family_index_value < family_count();
       ++family_index_value) {
    if (families_[xmvb::to_size(family_index_value)].family_delta ==
        family_delta) {
      return family_index_value;
    }
  }
  return -1;
}

}  // namespace xmvb::vb::exact_separator
