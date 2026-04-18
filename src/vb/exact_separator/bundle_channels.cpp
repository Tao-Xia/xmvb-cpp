#include "vb/exact_separator/bundle_channels.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>

#include "vb/matrices/two_electron_indexer.hpp"

namespace xmvb::vb::exact_separator {

bool have_same_boundary_spin_bundle_layout(
    const BoundarySpinBundleLayout& left,
    const BoundarySpinBundleLayout& right) {
  return left.row_count() == right.row_count() &&
      left.col_count() == right.col_count() &&
      left.base_selected_row_minus_col() == right.base_selected_row_minus_col() &&
      left.min_family_delta() == right.min_family_delta() &&
      left.max_family_delta() == right.max_family_delta() &&
      left.support_size() == right.support_size();
}

void decode_support_pair_index(
    int support_pair_index,
    int support_size,
    int* row_orbital,
    int* col_orbital) {
  if (row_orbital == nullptr || col_orbital == nullptr) {
    throw std::invalid_argument("decoded support pair outputs must not be null");
  }
  if (support_size <= 0) {
    throw std::invalid_argument("support_size must be positive");
  }
  if (support_pair_index < 0 || support_pair_index >= support_size * support_size) {
    throw std::out_of_range("support pair index is out of range");
  }
  *row_orbital = support_pair_index % support_size;
  *col_orbital = support_pair_index / support_size;
}

DecodedAntisymSupportPair decode_antisym_support_pair_index(
    int pair_index,
    int support_size) {
  if (support_size <= 0) {
    throw std::invalid_argument("support_size must be positive");
  }
  int running_index = 0;
  for (int first_orbital = 0; first_orbital < support_size; ++first_orbital) {
    for (int second_orbital = first_orbital + 1;
         second_orbital < support_size;
         ++second_orbital) {
      if (running_index == pair_index) {
        return DecodedAntisymSupportPair{
            .first_orbital = first_orbital,
            .second_orbital = second_orbital,
        };
      }
      ++running_index;
    }
  }
  throw std::out_of_range("antisymmetrized support pair index is out of range");
}

std::vector<int> collect_nonzero_boundary_spin_degree1_indices(
    const BoundarySpinBundle& bundle) {
  std::vector<int> indices;
  indices.reserve(bundle.degree1.size());
  for (int flat_index = 0;
       flat_index < static_cast<int>(bundle.degree1.size());
       ++flat_index) {
    if (std::abs(bundle.degree1[xmvb::to_size(flat_index)]) <= 1.0e-15) {
      continue;
    }
    indices.push_back(flat_index);
  }
  return indices;
}

double contract_boundary_spin_bundle_overlap(
    const BoundarySpinBundle& bundle) {
  return std::accumulate(bundle.degree0.begin(), bundle.degree0.end(), 0.0);
}

double contract_boundary_spin_bundle_one_electron(
    const BoundarySpinBundle& bundle,
    const std::vector<double>& support_one_electron_storage,
    int support_size) {
  if (bundle.layout.support_size() != support_size) {
    throw std::invalid_argument("one-electron contraction received inconsistent support size");
  }

  double total = 0.0;
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
    const double integral =
        support_one_electron_storage[xmvb::to_size(col_orbital) *
                                         xmvb::to_size(support_size) +
                                     xmvb::to_size(row_orbital)];
    if (std::abs(integral) <= 1.0e-15) {
      continue;
    }
    for (int flat_sector_index = 0;
         flat_sector_index < bundle.layout.total_sector_count();
         ++flat_sector_index) {
      const int flat_degree1_index =
          bundle.layout.flat_degree1_index(
              support_pair_index,
              flat_sector_index);
      total +=
          integral *
          bundle.degree1[xmvb::to_size(flat_degree1_index)];
    }
  }
  return total;
}

double contract_boundary_spin_bundle_same_spin(
    const BoundarySpinBundle& bundle,
    const std::vector<double>& packed_active_two_electron_integrals) {
  const int support_size = bundle.layout.support_size();
  const int pair_count = support_size * (support_size - 1) / 2;
  double total = 0.0;
  for (int support_pair_pair_index = 0;
       support_pair_pair_index < bundle.layout.support_pair_pair_count();
       ++support_pair_pair_index) {
    const int row_pair_index = support_pair_pair_index % pair_count;
    const int col_pair_index = support_pair_pair_index / pair_count;
    const DecodedAntisymSupportPair row_pair =
        decode_antisym_support_pair_index(row_pair_index, support_size);
    const DecodedAntisymSupportPair col_pair =
        decode_antisym_support_pair_index(col_pair_index, support_size);
    const int direct_index =
        xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
            row_pair.first_orbital,
            col_pair.first_orbital,
            row_pair.second_orbital,
            col_pair.second_orbital);
    const int exchange_index =
        xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
            row_pair.first_orbital,
            col_pair.second_orbital,
            row_pair.second_orbital,
            col_pair.first_orbital);
    const double integral =
        packed_active_two_electron_integrals[xmvb::to_size(direct_index)] -
        packed_active_two_electron_integrals[xmvb::to_size(exchange_index)];
    if (std::abs(integral) <= 1.0e-15) {
      continue;
    }
    for (int flat_sector_index = 0;
         flat_sector_index < bundle.layout.total_sector_count();
         ++flat_sector_index) {
      const int flat_degree2_index =
          bundle.layout.flat_degree2_index(
              support_pair_pair_index,
              flat_sector_index);
      total +=
          integral *
          bundle.degree2[xmvb::to_size(flat_degree2_index)];
    }
  }
  return total;
}

double contract_boundary_mixed_bundle_opposite_spin(
    const BoundaryMixedBundle& mixed,
    int alpha_total_sector_count,
    int beta_total_sector_count,
    int support_size,
    const std::vector<double>& packed_active_two_electron_integrals) {
  if (alpha_total_sector_count <= 0 || beta_total_sector_count <= 0) {
    throw std::invalid_argument("bundle opposite-spin contraction requires positive sector counts");
  }

  double total = 0.0;
  for (int beta_flat_degree1_index = 0;
       beta_flat_degree1_index < mixed.beta_degree1_entry_count;
       ++beta_flat_degree1_index) {
    const int beta_support_pair_index =
        beta_flat_degree1_index / beta_total_sector_count;
    int beta_row_orbital = -1;
    int beta_col_orbital = -1;
    decode_support_pair_index(
        beta_support_pair_index,
        support_size,
        &beta_row_orbital,
        &beta_col_orbital);
    for (int alpha_flat_degree1_index = 0;
         alpha_flat_degree1_index < mixed.alpha_degree1_entry_count;
         ++alpha_flat_degree1_index) {
      const double mixed_value =
          mixed.values[xmvb::to_size(beta_flat_degree1_index) *
                           xmvb::to_size(mixed.alpha_degree1_entry_count) +
                       xmvb::to_size(alpha_flat_degree1_index)];
      if (std::abs(mixed_value) <= 1.0e-15) {
        continue;
      }
      const int alpha_support_pair_index =
          alpha_flat_degree1_index / alpha_total_sector_count;
      int alpha_row_orbital = -1;
      int alpha_col_orbital = -1;
      decode_support_pair_index(
          alpha_support_pair_index,
          support_size,
          &alpha_row_orbital,
          &alpha_col_orbital);
      const int eri_index =
          xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
              beta_row_orbital,
              beta_col_orbital,
              alpha_row_orbital,
              alpha_col_orbital);
      total +=
          packed_active_two_electron_integrals[xmvb::to_size(eri_index)] *
          mixed_value;
    }
  }
  return total;
}

}  // namespace xmvb::vb::exact_separator
