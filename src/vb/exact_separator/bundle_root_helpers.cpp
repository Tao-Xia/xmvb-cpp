#include "vb/exact_separator/bundle_root_helpers.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "vb/exact_separator/bundle.hpp"
#include "vb/exact_separator/bundle_channels.hpp"

namespace xmvb::vb::exact_separator {

namespace {

int parity_sign(int parity) {
  return ((parity & 1) == 0) ? 1 : -1;
}

int count_deleted_labels_in_frontier(
    const std::vector<int>& deleted_labels,
    const std::vector<int>& local_remainder_labels) {
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

int compute_interface_block_parity(
    int interface_rows,
    int interface_cols,
    const SpinInterfaceLayout& layout,
    const std::vector<int>& row_labels,
    const std::vector<int>& col_labels) {
  const int deleted_rows_in_frontier =
      count_deleted_labels_in_frontier(row_labels, layout.local_remainder_rows);
  const int deleted_cols_in_frontier =
      count_deleted_labels_in_frontier(col_labels, layout.local_remainder_cols);
  const int deleted_frontier_interface_rows =
      count_deleted_labels_in_set(row_labels, layout.frontier_interface_rows);
  const int deleted_frontier_interface_cols =
      count_deleted_labels_in_set(col_labels, layout.frontier_interface_cols);
  const int frontier_body_rows =
      layout.frontier_rows - static_cast<int>(layout.frontier_interface_rows.size());
  const int frontier_body_cols =
      layout.frontier_cols - static_cast<int>(layout.frontier_interface_cols.size());
  const int deleted_frontier_body_rows =
      deleted_rows_in_frontier - deleted_frontier_interface_rows;
  const int deleted_frontier_body_cols =
      deleted_cols_in_frontier - deleted_frontier_interface_cols;
  const int surviving_frontier_body_rows =
      frontier_body_rows - deleted_frontier_body_rows;
  const int surviving_frontier_body_cols =
      frontier_body_cols - deleted_frontier_body_cols;
  return interface_rows *
          (surviving_frontier_body_rows + deleted_frontier_body_cols) +
      interface_cols *
          (surviving_frontier_body_cols + deleted_frontier_body_rows);
}

void sort_layout_sets(SpinInterfaceLayout* layout) {
  std::sort(layout->frontier_interface_rows.begin(), layout->frontier_interface_rows.end());
  std::sort(layout->frontier_interface_cols.begin(), layout->frontier_interface_cols.end());
  std::sort(layout->local_remainder_rows.begin(), layout->local_remainder_rows.end());
  std::sort(layout->local_remainder_cols.begin(), layout->local_remainder_cols.end());
}

void make_support_pair_labels(
    int support_pair_index,
    int support_size,
    std::vector<int>* row_labels,
    std::vector<int>* col_labels) {
  const int row_orbital = support_pair_index % support_size;
  const int col_orbital = support_pair_index / support_size;
  row_labels->assign(1, row_orbital);
  col_labels->assign(1, col_orbital);
}

void make_support_pair_pair_labels(
    int support_pair_pair_index,
    int support_size,
    std::vector<int>* row_labels,
    std::vector<int>* col_labels) {
  if (support_size < 2) {
    row_labels->clear();
    col_labels->clear();
    return;
  }
  const int pair_count = support_size * (support_size - 1) / 2;
  const int row_pair_index = support_pair_pair_index % pair_count;
  const int col_pair_index = support_pair_pair_index / pair_count;
  const DecodedAntisymSupportPair row_pair =
      decode_antisym_support_pair_index(row_pair_index, support_size);
  const DecodedAntisymSupportPair col_pair =
      decode_antisym_support_pair_index(col_pair_index, support_size);
  row_labels->assign({row_pair.first_orbital, row_pair.second_orbital});
  col_labels->assign({col_pair.first_orbital, col_pair.second_orbital});
}

SpinInterfaceLayout normalized_spin_layout(SpinInterfaceLayout layout) {
  sort_layout_sets(&layout);
  return layout;
}

std::vector<int> build_support_pair_parities(
    int support_pair_count,
    int support_size,
    const SpinInterfaceLayout& layout) {
  if (support_pair_count <= 0 || support_size <= 0) {
    return {};
  }
  const int interface_rows =
      static_cast<int>(layout.frontier_interface_rows.size());
  const int interface_cols =
      static_cast<int>(layout.frontier_interface_cols.size());
  std::vector<int> row_labels;
  std::vector<int> col_labels;
  std::vector<int> parities;
  parities.reserve(xmvb::to_size(support_pair_count));
  for (int support_pair_index = 0;
       support_pair_index < support_pair_count;
       ++support_pair_index) {
    make_support_pair_labels(
        support_pair_index,
        support_size,
        &row_labels,
        &col_labels);
    parities.push_back(compute_interface_block_parity(
        interface_rows,
        interface_cols,
        layout,
        row_labels,
        col_labels));
  }
  return parities;
}

std::vector<int> build_support_pair_pair_parities(
    int support_pair_pair_count,
    int support_size,
    const SpinInterfaceLayout& layout) {
  if (support_pair_pair_count <= 0 || support_size <= 1) {
    return {};
  }
  const int interface_rows =
      static_cast<int>(layout.frontier_interface_rows.size());
  const int interface_cols =
      static_cast<int>(layout.frontier_interface_cols.size());
  std::vector<int> row_labels;
  std::vector<int> col_labels;
  std::vector<int> parities;
  parities.reserve(xmvb::to_size(support_pair_pair_count));
  for (int support_pair_pair_index = 0;
       support_pair_pair_index < support_pair_pair_count;
       ++support_pair_pair_index) {
    make_support_pair_pair_labels(
        support_pair_pair_index,
        support_size,
        &row_labels,
        &col_labels);
    if (row_labels.empty() || col_labels.empty()) {
      parities.push_back(0);
      continue;
    }
    parities.push_back(compute_interface_block_parity(
        interface_rows,
        interface_cols,
        layout,
        row_labels,
        col_labels));
  }
  return parities;
}

std::vector<int> expand_parities_for_sectors(
    const std::vector<int>& base_parities,
    int sectors_per_support_pair) {
  if (base_parities.empty() || sectors_per_support_pair <= 0) {
    return {};
  }
  std::vector<int> expanded;
  expanded.reserve(
      xmvb::to_size(base_parities.size()) *
      xmvb::to_size(sectors_per_support_pair));
  for (const int base_parity : base_parities) {
    for (int sector_index = 0;
         sector_index < sectors_per_support_pair;
         ++sector_index) {
      expanded.push_back(base_parity);
    }
  }
  return expanded;
}

void apply_parity_vector(
    std::vector<double>* values,
    const std::vector<int>& parities) {
  if (values->empty() || parities.empty()) {
    return;
  }
  if (values->size() != parities.size()) {
    throw std::logic_error("boundary bundle parity vector size mismatch");
  }
  for (std::size_t index = 0;
       index < values->size();
       ++index) {
    (*values)[index] *= parity_sign(parities[index]);
  }
}

void apply_spin_bundle_interface_parity(
    BoundarySpinBundle* bundle,
    SpinInterfaceLayout layout) {
  if (layout.frontier_interface_rows.empty() &&
      layout.frontier_interface_cols.empty()) {
    return;
  }
  const SpinInterfaceLayout sorted_layout = normalized_spin_layout(layout);
  const int support_size = bundle->layout.support_size();
  const int total_sectors = bundle->layout.total_sector_count();
  const std::vector<int> degree1_parities = expand_parities_for_sectors(
      build_support_pair_parities(
          bundle->layout.support_pair_count(),
          support_size,
          sorted_layout),
      total_sectors);
  apply_parity_vector(&bundle->degree1, degree1_parities);
  const std::vector<int> degree2_parities = expand_parities_for_sectors(
      build_support_pair_pair_parities(
          bundle->layout.support_pair_pair_count(),
          support_size,
          sorted_layout),
      total_sectors);
  apply_parity_vector(&bundle->degree2, degree2_parities);
}

void apply_mixed_bundle_interface_parity(
    BoundaryMixedBundle* mixed,
    const BoundarySpinBundleLayout& alpha_bundle_layout,
    const BoundarySpinBundleLayout& beta_bundle_layout,
    SpinInterfaceLayout alpha_layout,
    SpinInterfaceLayout beta_layout) {
  if (alpha_layout.frontier_interface_rows.empty() &&
      alpha_layout.frontier_interface_cols.empty() &&
      beta_layout.frontier_interface_rows.empty() &&
      beta_layout.frontier_interface_cols.empty()) {
    return;
  }
  const SpinInterfaceLayout sorted_alpha_layout =
      normalized_spin_layout(alpha_layout);
  const SpinInterfaceLayout sorted_beta_layout =
      normalized_spin_layout(beta_layout);
  const int support_size = alpha_bundle_layout.support_size();
  if (support_size != beta_bundle_layout.support_size()) {
    throw std::logic_error("mixed bundle spin layouts must share support_size");
  }
  const int alpha_total_sectors = alpha_bundle_layout.total_sector_count();
  const int beta_total_sectors = beta_bundle_layout.total_sector_count();
  const int alpha_support_pair_count = alpha_bundle_layout.support_pair_count();
  const int beta_support_pair_count = beta_bundle_layout.support_pair_count();
  const std::vector<int> alpha_degree_parities = expand_parities_for_sectors(
      build_support_pair_parities(
          alpha_support_pair_count,
          support_size,
          sorted_alpha_layout),
      alpha_total_sectors);
  const std::vector<int> beta_degree_parities = expand_parities_for_sectors(
      build_support_pair_parities(
          beta_support_pair_count,
          support_size,
          sorted_beta_layout),
      beta_total_sectors);
  if (alpha_degree_parities.empty() || beta_degree_parities.empty()) {
    return;
  }
  if (static_cast<int>(alpha_degree_parities.size()) !=
          mixed->alpha_degree1_entry_count ||
      static_cast<int>(beta_degree_parities.size()) !=
          mixed->beta_degree1_entry_count) {
    throw std::logic_error("mixed bundle parity vector size mismatch");
  }
  const int alpha_degree_count =
      static_cast<int>(alpha_degree_parities.size());
  const int beta_degree_count =
      static_cast<int>(beta_degree_parities.size());
  for (int beta_index = 0; beta_index < beta_degree_count; ++beta_index) {
    for (int alpha_index = 0; alpha_index < alpha_degree_count; ++alpha_index) {
      const int total_parity =
          alpha_degree_parities[alpha_index] + beta_degree_parities[beta_index];
      const int flat_index =
          static_cast<int64_t>(beta_index) * alpha_degree_count + alpha_index;
      mixed->values[xmvb::to_size(flat_index)] *=
          parity_sign(total_parity);
    }
  }
}

}  // namespace

BoundaryBundle reorder_root_bundle(
    const BoundaryBundle& natural_payload,
    const RootInterfaceLayout& layout) {
  BoundaryBundle reordered = natural_payload;
  apply_spin_bundle_interface_parity(&reordered.alpha, layout.alpha);
  apply_spin_bundle_interface_parity(&reordered.beta, layout.beta);
  apply_mixed_bundle_interface_parity(
      &reordered.mixed,
      reordered.alpha.layout,
      reordered.beta.layout,
      layout.alpha,
      layout.beta);
  return reordered;
}

HamiltonianBoundaryBundle reorder_root_hamiltonian_bundle(
    const HamiltonianBoundaryBundle& natural_bundle,
    const RootInterfaceLayout& layout) {
  HamiltonianBoundaryBundle reordered = natural_bundle;
  apply_spin_bundle_interface_parity(&reordered.alpha, layout.alpha);
  apply_spin_bundle_interface_parity(&reordered.beta, layout.beta);
  apply_mixed_bundle_interface_parity(
      &reordered.mixed,
      reordered.alpha.layout,
      reordered.beta.layout,
      layout.alpha,
      layout.beta);
  return reordered;
}

BundleRootReadout contract_root_bundle_channels(
    const BoundaryBundle& bundle,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size) {
  BundleRootReadout result;
  const double alpha_overlap =
      contract_boundary_spin_bundle_overlap(bundle.alpha);
  const double beta_overlap =
      contract_boundary_spin_bundle_overlap(bundle.beta);
  const double alpha_one_electron =
      contract_boundary_spin_bundle_one_electron(
          bundle.alpha,
          support_one_electron_storage,
          support_size);
  const double beta_one_electron =
      contract_boundary_spin_bundle_one_electron(
          bundle.beta,
          support_one_electron_storage,
          support_size);
  result.overlap = alpha_overlap * beta_overlap;
  result.one_electron =
      alpha_one_electron * beta_overlap + alpha_overlap * beta_one_electron;
  result.same_spin_alpha_two_electron =
      contract_boundary_spin_bundle_same_spin(
          bundle.alpha,
          packed_active_two_electron_integrals) *
      beta_overlap;
  result.same_spin_beta_two_electron =
      contract_boundary_spin_bundle_same_spin(
          bundle.beta,
          packed_active_two_electron_integrals) *
      alpha_overlap;
  result.opposite_spin_two_electron =
      contract_boundary_mixed_bundle_opposite_spin(
          bundle.mixed,
          bundle.alpha.layout.total_sector_count(),
          bundle.beta.layout.total_sector_count(),
          support_size,
          packed_active_two_electron_integrals);
  return result;
}

}  // namespace xmvb::vb::exact_separator
