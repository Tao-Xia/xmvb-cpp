#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "vb/exact_separator/boundary_sector.hpp"
#include "vb/matrices/determinant_overlap_resolver.hpp"
#include "vb/matrices/spin_pair_utils.hpp"

namespace xmvb::vb::exact_separator {

/**
 * @brief One exact `(2,2)` deleted-minor entry inside one boundary sector.
 *
 * Row pairs and column pairs are stored in canonical support-orbital order
 * with the corresponding antisymmetry sign already absorbed into `value`.
 */
struct OneLeafBoundarySpinSecondCofactorEntry {
  int row_first_orbital = -1;
  int row_second_orbital = -1;
  int col_first_orbital = -1;
  int col_second_orbital = -1;
  double value = 0.0;
};

/**
 * @brief One exact first-cofactor entry inside one boundary sector.
 *
 * Rows index ket/right orbitals and columns index bra/left orbitals in the
 * same support-space convention used throughout the exact separator code.
 */
struct OneLeafBoundarySpinFirstCofactorEntry {
  int row_orbital = -1;
  int col_orbital = -1;
  double value = 0.0;
};

/**
 * @brief Exact one-spin one-leaf message for one boundary sector.
 *
 * `row_mask` and `col_mask` index one exact selected-boundary sector inside
 * one fixed frontier-shape family.
 * `overlap`, `first_cofactor_entries`, and `second_cofactor_entries` are the
 * already merged exact degree-0/1/2 sector amplitudes:
 *
 *   M^{(0)}(s), M^{(1)}(\mu; s), M^{(2)}(\nu; s)
 *
 * in support-space coordinates with the full one-leaf block-order sign
 * convention already applied.
 */
struct OneLeafBoundarySpinSectorMessage {
  int sector_index = -1;
  std::uint32_t row_mask = 0U;
  std::uint32_t col_mask = 0U;
  double overlap = 0.0;
  std::vector<OneLeafBoundarySpinFirstCofactorEntry> first_cofactor_entries;
  std::vector<OneLeafBoundarySpinSecondCofactorEntry> second_cofactor_entries;
};

/**
 * @brief One exact imbalance family of the one-leaf boundary message.
 *
 * `frontier_row_minus_col` is the local frontier shape difference
 * `f_r - f_c`. The exact degree-0/1/2 one-leaf algebra needs the family
 * range `-2 <= f_r - f_c <= 2`, because degree-1 and degree-2 channels couple
 * row-open / col-open / excess-two local sectors in addition to the balanced
 * overlap family.
 */
struct OneLeafBoundarySpinSectorFamilyMessage {
  BoundarySectorFamilyIndexer family_indexer;
  std::vector<OneLeafBoundarySpinSectorMessage> sectors;
};

/**
 * @brief Exact one-spin one-leaf boundary message over all imbalance families.
 *
 * The message is indexed only by boundary selected-mask sectors, but those
 * sectors are grouped into imbalance families labeled by the frontier shape
 * difference `f_r - f_c`. Each sector stores the fully merged degree-0/1/2
 * amplitude in support-space coordinates, so no reduced-state closure
 * metadata is needed after construction.
 */
struct OneLeafBoundarySpinMessage {
  int support_size = 0;
  int base_selected_row_minus_col = 0;
  std::vector<OneLeafBoundarySpinSectorFamilyMessage> families;
};

/**
 * @brief Exact reconstruction from the exported one-leaf boundary sectors.
 *
 * `overlap`, `first_cofactor`, and `same_spin_two_electron` are obtained by
 * summing the exact degree-0/1/2 sector amplitudes stored in the message.
 */
struct OneLeafBoundarySpinAggregate {
  double overlap = 0.0;
  Eigen::MatrixXd first_cofactor;
  double same_spin_two_electron = 0.0;
  std::vector<OneLeafBoundarySpinSecondCofactorEntry> second_cofactor_entries;
};

/**
 * @brief Builds the exact one-spin one-leaf boundary message in sector form.
 *
 * `left_root_occ` / `right_root_occ` are the root-side occupied orbital lists
 * for one spin in component order. `left_leaf_occ` / `right_leaf_occ` are the
 * corresponding leaf-side occupied orbital lists. The returned message stores,
 * for every admissible selected root boundary sector, the exact merged
 * degree-0/1/2 boundary amplitudes in support-space coordinates.
 */
OneLeafBoundarySpinMessage build_one_leaf_spin_boundary_message(
    const std::vector<int>& left_root_occ,
    const std::vector<int>& left_leaf_occ,
    const std::vector<int>& right_root_occ,
    const std::vector<int>& right_leaf_occ,
    const std::vector<double>& support_overlap_storage,
    int support_size,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations);

/**
 * @brief Applies the exact one-spin sector merge to the message.
 *
 * The returned overlap, first-cofactor matrix, and same-spin two-electron
 * scalar are numerically identical to the direct exact one-spin builders for
 * the same occupied-orbital lists.
 */
OneLeafBoundarySpinAggregate contract_one_leaf_spin_boundary_message(
    const OneLeafBoundarySpinMessage& message,
    const std::vector<double>& packed_active_two_electron_integrals);

}  // namespace xmvb::vb::exact_separator
