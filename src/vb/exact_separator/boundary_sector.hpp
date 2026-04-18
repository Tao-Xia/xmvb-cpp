#pragma once

#include <cstdint>
#include <vector>

namespace xmvb::vb::exact_separator {

/**
 * @brief One admissible boundary sector in the selected-mask basis.
 *
 * The current rooted-tree overlap prototype uses the exact boundary state
 * basis already implicit in the separator masks: one sector is a selected
 * boundary-row mask together with a selected boundary-column mask. Only mask
 * pairs whose popcount difference matches the fixed subtree row/column
 * imbalance are retained.
 */
struct BoundarySector {
  std::uint32_t row_mask = 0U;
  std::uint32_t col_mask = 0U;
  int selected_row_count = 0;
  int selected_col_count = 0;
};

/**
 * @brief Indexes exact selected-boundary sectors for one spin channel.
 *
 * For a subtree with fixed row/column imbalance
 * `selected_row_count - selected_col_count = delta`, this class enumerates
 * all admissible boundary mask pairs on the parent interface. The resulting
 * sector count is the exact `2^m`-type boundary basis used by the new
 * overlap-only prototype.
 */
class BoundarySectorIndexer {
 public:
  BoundarySectorIndexer() = default;
  BoundarySectorIndexer(
      int row_count,
      int col_count,
      int selected_row_minus_col);

  int row_count() const;
  int col_count() const;
  int selected_row_minus_col() const;
  int sector_count() const;

  std::uint32_t full_row_mask() const;
  std::uint32_t full_col_mask() const;

  const BoundarySector& sector(int index) const;
  int index(std::uint32_t row_mask, std::uint32_t col_mask) const;
  int complement_index(int index) const;

 private:
  int row_count_ = 0;
  int col_count_ = 0;
  int selected_row_minus_col_ = 0;
  std::uint32_t full_row_mask_ = 0U;
  std::uint32_t full_col_mask_ = 0U;
  std::vector<BoundarySector> sectors_;
  std::vector<int> packed_index_table_;
};

/**
 * @brief One exact imbalance family in the boundary selected-mask basis.
 *
 * `family_delta` is the local row-minus-column shift relative to the base
 * subtree imbalance. For the one-leaf degree-1/2 algebra this is exactly the
 * frontier shape difference `f_r - f_c`.
 */
struct BoundarySectorFamilyIndexer {
  int family_delta = 0;
  BoundarySectorIndexer sector_indexer;
};

/**
 * @brief Enumerates a contiguous range of imbalance families.
 *
 * This is the reusable helper for exact degree-1/2 boundary messages: the
 * boundary state still lives in mask sectors, but degree-1/2 channels require
 * multiple row/column-imbalance families around the base overlap family.
 */
class BoundarySectorFamilyIndexerRange {
 public:
  BoundarySectorFamilyIndexerRange() = default;
  BoundarySectorFamilyIndexerRange(
      int row_count,
      int col_count,
      int base_selected_row_minus_col,
      int min_family_delta,
      int max_family_delta);

  int row_count() const;
  int col_count() const;
  int base_selected_row_minus_col() const;
  int min_family_delta() const;
  int max_family_delta() const;
  int family_count() const;

  const BoundarySectorFamilyIndexer& family(int index) const;
  int family_index(int family_delta) const;

 private:
  int row_count_ = 0;
  int col_count_ = 0;
  int base_selected_row_minus_col_ = 0;
  int min_family_delta_ = 0;
  int max_family_delta_ = -1;
  std::vector<BoundarySectorFamilyIndexer> families_;
};

}  // namespace xmvb::vb::exact_separator
