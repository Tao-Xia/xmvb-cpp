#pragma once

#include <cstdint>
#include <vector>

#include "vb/exact_separator/boundary_sector.hpp"

namespace xmvb::vb::exact_separator {

/**
 * @brief Canonical dense layout of one-spin boundary sectors and operator bases.
 *
 * The current rooted-tree code still stores subtree messages as sparse maps on
 * deleted-minor keys and then enumerates compatible mask combinations at merge
 * time. The long-term `2^n -> 2^m` rewrite needs one canonical dense basis so
 * every subtree can be represented by one bundle and every merge can be
 * implemented as a contraction on that bundle.
 *
 * This layout fixes three index spaces for one spin channel:
 *
 * 1. boundary selected-mask sectors grouped into imbalance families;
 * 2. the degree-1 support-pair basis `(row_orbital, col_orbital)`;
 * 3. the degree-2 antisymmetrized support pair-pair basis.
 *
 * The actual forward / reverse algebra is implemented later; this class only
 * defines the exact canonical storage order needed by that algebra.
 */
class BoundarySpinBundleLayout {
 public:
  BoundarySpinBundleLayout() = default;
  BoundarySpinBundleLayout(
      int row_count,
      int col_count,
      int base_selected_row_minus_col,
      int min_family_delta,
      int max_family_delta,
      int support_size);

  int row_count() const;
  int col_count() const;
  int base_selected_row_minus_col() const;
  int min_family_delta() const;
  int max_family_delta() const;
  int support_size() const;

  int family_count() const;
  int total_sector_count() const;
  int support_pair_count() const;
  int antisym_support_pair_count() const;
  int support_pair_pair_count() const;

  const BoundarySectorFamilyIndexerRange& family_range() const;
  const BoundarySectorFamilyIndexer& family(int index) const;
  int family_index(int family_delta) const;
  int family_sector_offset(int family_index) const;

  int flat_sector_index(int family_index, int sector_index) const;
  int flat_degree1_index(int support_pair_index, int flat_sector_index) const;
  int flat_degree2_index(
      int support_pair_pair_index,
      int flat_sector_index) const;

 private:
  BoundarySectorFamilyIndexerRange family_range_;
  int support_size_ = 0;
  int support_pair_count_ = 0;
  int antisym_support_pair_count_ = 0;
  int support_pair_pair_count_ = 0;
  int total_sector_count_ = 0;
  std::vector<int> family_sector_offsets_;
};

/**
 * @brief One canonical one-spin boundary bundle.
 *
 * `degree0`, `degree1`, and `degree2` store the exact degree-0/1/2 message
 * coefficients in the canonical dense storage induced by
 * `BoundarySpinBundleLayout`.
 *
 * Storage conventions:
 *
 * - `degree0[flat_sector_index]`
 * - `degree1[layout.flat_degree1_index(mu, flat_sector_index)]`
 * - `degree2[layout.flat_degree2_index(nu, flat_sector_index)]`
 *
 * where `mu` is a support-pair index and `nu` is an antisymmetrized
 * support-pair-pair index.
 */
struct BoundarySpinBundle {
  BoundarySpinBundleLayout layout;
  std::vector<double> degree0;
  std::vector<double> degree1;
  std::vector<double> degree2;
};

/**
 * @brief Mixed alpha/beta second-order moment on the flattened degree-1 basis.
 *
 * The exact opposite-spin root closure acts on the second-order moment of the
 * alpha and beta degree-1 bundles. The row and column dimensions are the
 * flattened degree-1 entry counts of the alpha and beta bundles respectively.
 */
struct BoundaryMixedBundle {
  int alpha_degree1_entry_count = 0;
  int beta_degree1_entry_count = 0;
  std::vector<double> values;
};

/**
 * @brief Full canonical boundary bundle used by the planned bundle algebra.
 *
 * The intended exact rooted-tree production message is:
 *
 * - one alpha one-spin bundle;
 * - one beta one-spin bundle;
 * - one mixed alpha/beta degree-1 second-order moment.
 *
 * This matches the bundle notation introduced in `gradient.md`.
 */
struct BoundaryBundle {
  BoundarySpinBundle alpha;
  BoundarySpinBundle beta;
  BoundaryMixedBundle mixed;
};

/**
 * @brief Dense coordinate of one flattened boundary sector.
 *
 * `family_index` indexes one imbalance family inside the layout and
 * `sector_index` indexes one selected-mask sector inside that family.
 */
struct BoundaryFlatSectorIndex {
  int family_index = -1;
  int sector_index = -1;
};

/**
 * @brief Allocates a zero-filled one-spin bundle for the given layout.
 */
BoundarySpinBundle make_zero_boundary_spin_bundle(
    const BoundarySpinBundleLayout& layout);

/**
 * @brief Allocates a zero-filled mixed alpha/beta bundle.
 */
BoundaryMixedBundle make_zero_boundary_mixed_bundle(
    int alpha_degree1_entry_count,
    int beta_degree1_entry_count);

/**
 * @brief Allocates a zero-filled full boundary bundle.
 */
BoundaryBundle make_zero_boundary_bundle(
    const BoundarySpinBundleLayout& alpha_layout,
    const BoundarySpinBundleLayout& beta_layout);

/**
 * @brief Checks that a one-spin bundle has the exact dense sizes of its layout.
 */
bool has_compatible_layout(const BoundarySpinBundle& bundle);

/**
 * @brief Checks that a full bundle has mutually compatible alpha/beta/mixed sizes.
 */
bool has_compatible_layout(const BoundaryBundle& bundle);

/**
 * @brief Flattens one support-space pair `(row_orbital, col_orbital)`.
 *
 * The storage order matches the repository-wide column-major convention used
 * for support-space one-electron matrices:
 *
 * `pair_index = col_orbital * support_size + row_orbital`.
 */
int flatten_support_pair_index(
    int row_orbital,
    int col_orbital,
    int support_size);

/**
 * @brief Flattens one antisymmetrized support-space pair `(first, second)`.
 *
 * The orbitals must already satisfy `first < second`. The returned index is a
 * dense triangular-number encoding on the canonical sorted pair basis.
 */
int flatten_antisym_support_pair_index(
    int first_orbital,
    int second_orbital,
    int support_size);

/**
 * @brief Flattens one antisymmetrized support pair-pair basis element.
 *
 * The row pair indexes the ket/right side and the column pair indexes the
 * bra/left side. The combined storage uses the same column-major convention as
 * the degree-1 basis:
 *
 * `pair_pair_index = col_pair_index * pair_count + row_pair_index`.
 */
int flatten_support_pair_pair_index(
    int row_first_orbital,
    int row_second_orbital,
    int col_first_orbital,
    int col_second_orbital,
    int support_size);

/**
 * @brief Encodes one selected-label set as a mask in a fixed boundary order.
 *
 * `boundary_labels` defines the canonical boundary order. Every label in
 * `selected_labels` must occur exactly once in `boundary_labels`.
 */
std::uint32_t encode_boundary_selection_mask(
    const std::vector<int>& boundary_labels,
    const std::vector<int>& selected_labels);

/**
 * @brief Converts selected boundary labels to one flattened sector index.
 *
 * This is the low-level adapter primitive needed by the later legacy-payload
 * bridge: the old code stores deleted boundary labels explicitly, while the
 * new bundle algebra stores one canonical flattened sector index.
 *
 * The function returns `-1` if the selected row/column counts do not belong to
 * any family in `layout`.
 */
int find_boundary_flat_sector_index(
    const BoundarySpinBundleLayout& layout,
    const std::vector<int>& boundary_row_labels,
    const std::vector<int>& selected_row_labels,
    const std::vector<int>& boundary_col_labels,
    const std::vector<int>& selected_col_labels);

/**
 * @brief Decodes one flattened sector index back to `(family, sector)`.
 */
BoundaryFlatSectorIndex decode_boundary_flat_sector_index(
    const BoundarySpinBundleLayout& layout,
    int flat_sector_index);

}  // namespace xmvb::vb::exact_separator
