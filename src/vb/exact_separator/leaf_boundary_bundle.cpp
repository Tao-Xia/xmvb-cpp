#include "vb/exact_separator/leaf_boundary_bundle.hpp"

#include <stdexcept>

namespace xmvb::vb::exact_separator {

BoundarySpinBundleLayout make_one_leaf_boundary_spin_bundle_layout(
    const OneLeafBoundarySpinMessage& message) {
  if (message.support_size <= 0) {
    throw std::invalid_argument("boundary spin message support_size must be positive");
  }
  if (message.families.empty()) {
    throw std::invalid_argument("boundary spin message must contain at least one family");
  }

  const BoundarySectorFamilyIndexer& first_family =
      message.families.front().family_indexer;
  const BoundarySectorFamilyIndexer& last_family =
      message.families.back().family_indexer;
  return BoundarySpinBundleLayout(
      first_family.sector_indexer.row_count(),
      first_family.sector_indexer.col_count(),
      message.base_selected_row_minus_col,
      first_family.family_delta,
      last_family.family_delta,
      message.support_size);
}

BoundarySpinBundle build_one_leaf_boundary_spin_bundle(
    const OneLeafBoundarySpinMessage& message) {
  // The one-leaf exported boundary message already stores exact degree-0/1/2
  // amplitudes sector-by-sector. This adapter only reindexes that sparse
  // message into the canonical dense bundle basis used by the planned
  // `BoundaryBundle` carrier.
  const BoundarySpinBundleLayout layout =
      make_one_leaf_boundary_spin_bundle_layout(message);
  BoundarySpinBundle bundle = make_zero_boundary_spin_bundle(layout);
  for (int family_index = 0;
       family_index < static_cast<int>(message.families.size());
       ++family_index) {
    const OneLeafBoundarySpinSectorFamilyMessage& family =
        message.families[xmvb::to_size(family_index)];
    for (const auto& sector : family.sectors) {
      const int flat_sector_index =
          layout.flat_sector_index(family_index, sector.sector_index);
      bundle.degree0[xmvb::to_size(flat_sector_index)] +=
          sector.overlap;
      for (const auto& entry : sector.first_cofactor_entries) {
        const int support_pair_index = flatten_support_pair_index(
            entry.row_orbital,
            entry.col_orbital,
            message.support_size);
        const int flat_degree1_index =
            layout.flat_degree1_index(support_pair_index, flat_sector_index);
        bundle.degree1[xmvb::to_size(flat_degree1_index)] +=
            entry.value;
      }
      for (const auto& entry : sector.second_cofactor_entries) {
        const int support_pair_pair_index = flatten_support_pair_pair_index(
            entry.row_first_orbital,
            entry.row_second_orbital,
            entry.col_first_orbital,
            entry.col_second_orbital,
            message.support_size);
        const int flat_degree2_index =
            layout.flat_degree2_index(
                support_pair_pair_index,
                flat_sector_index);
        bundle.degree2[xmvb::to_size(flat_degree2_index)] +=
            entry.value;
      }
    }
  }
  return bundle;
}

}  // namespace xmvb::vb::exact_separator
