#include "vb/exact_separator/hamiltonian_bundle.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vb/exact_separator/bundle_channels.hpp"

namespace xmvb::vb::exact_separator {

namespace {

constexpr double kZeroTolerance = 1.0e-15;

void validate_same_spin_layout(
    const BoundarySpinBundle& source,
    const BoundarySpinBundle& destination) {
  if (!have_same_boundary_spin_bundle_layout(
          source.layout,
          destination.layout)) {
    throw std::invalid_argument("boundary spin bundle layouts do not match");
  }
}

void add_scaled_boundary_spin_bundle(
    const BoundarySpinBundle& source,
    double scale,
    BoundarySpinBundle* destination) {
  if (destination == nullptr) {
    throw std::invalid_argument("destination must not be null");
  }
  if (std::abs(scale) <= kZeroTolerance) {
    return;
  }
  validate_same_spin_layout(source, *destination);
  for (std::size_t index = 0; index < source.degree0.size(); ++index) {
    destination->degree0[index] += scale * source.degree0[index];
  }
  for (std::size_t index = 0; index < source.degree1.size(); ++index) {
    destination->degree1[index] += scale * source.degree1[index];
  }
  for (std::size_t index = 0; index < source.degree2.size(); ++index) {
    destination->degree2[index] += scale * source.degree2[index];
  }
}

int parity_of_permutation(std::vector<int>* values) {
  if (values == nullptr) {
    throw std::invalid_argument("values must not be null");
  }
  int parity = 0;
  for (int i = 0; i < static_cast<int>(values->size()); ++i) {
    for (int j = i + 1; j < static_cast<int>(values->size()); ++j) {
      if ((*values)[xmvb::to_size(i)] >
          (*values)[xmvb::to_size(j)]) {
        parity ^= 1;
      }
    }
  }
  std::sort(values->begin(), values->end());
  return parity;
}

int validate_and_canonicalize_sector_labels(
    const std::vector<int>& row_orbitals,
    const std::vector<int>& col_orbitals,
    std::vector<int>* canonical_rows,
    std::vector<int>* canonical_cols) {
  if (canonical_rows == nullptr || canonical_cols == nullptr) {
    throw std::invalid_argument("canonical orbital outputs must not be null");
  }
  if (row_orbitals.size() != col_orbitals.size()) {
    throw std::invalid_argument("projected sector row/col sizes must match");
  }
  if (row_orbitals.size() > 2U) {
    throw std::invalid_argument("projected sector degree must be <= 2");
  }
  *canonical_rows = row_orbitals;
  *canonical_cols = col_orbitals;
  if (canonical_rows->size() == 2U) {
    if ((*canonical_rows)[0] == (*canonical_rows)[1] ||
        (*canonical_cols)[0] == (*canonical_cols)[1]) {
      throw std::invalid_argument("degree-2 projected sector orbitals must be distinct");
    }
  }
  const int row_parity = parity_of_permutation(canonical_rows);
  const int col_parity = parity_of_permutation(canonical_cols);
  return row_parity ^ col_parity;
}

void validate_hamiltonian_bundle_layout_compatibility(
    const HamiltonianBoundaryBundle& source,
    const HamiltonianBoundaryBundle& destination) {
  validate_same_spin_layout(source.alpha, destination.alpha);
  validate_same_spin_layout(source.beta, destination.beta);
  if (source.mixed.alpha_degree1_entry_count !=
          destination.mixed.alpha_degree1_entry_count ||
      source.mixed.beta_degree1_entry_count !=
          destination.mixed.beta_degree1_entry_count) {
    throw std::invalid_argument("mixed bundle dimensions do not match");
  }
}

void add_scaled_boundary_mixed_bundle(
    const BoundaryMixedBundle& source,
    double scale,
    BoundaryMixedBundle* destination) {
  if (destination == nullptr) {
    throw std::invalid_argument("destination must not be null");
  }
  if (std::abs(scale) <= kZeroTolerance) {
    return;
  }
  if (source.alpha_degree1_entry_count != destination->alpha_degree1_entry_count ||
      source.beta_degree1_entry_count != destination->beta_degree1_entry_count ||
      source.values.size() != destination->values.size()) {
    throw std::invalid_argument("mixed bundle dimensions do not match");
  }
  for (std::size_t index = 0; index < source.values.size(); ++index) {
    destination->values[index] += scale * source.values[index];
  }
}

std::vector<std::pair<int, double>> collect_projected_degree1_entries(
    const HamiltonianProjectedSpinValues& projected,
    int flat_sector_index,
    const BoundarySpinBundleLayout& layout) {
  if (flat_sector_index < 0 || flat_sector_index >= layout.total_sector_count()) {
    throw std::invalid_argument("flat_sector_index is out of range");
  }
  if (layout.support_size() <= 0) {
    throw std::invalid_argument("layout support_size must be positive");
  }
  std::vector<std::pair<int, double>> entries;
  entries.reserve(projected.first_order_sectors.size());
  for (const auto& sector : projected.first_order_sectors) {
    if (std::abs(sector.value) <= kZeroTolerance) {
      continue;
    }
    if (sector.row_orbitals.size() != 1U || sector.col_orbitals.size() != 1U) {
      throw std::invalid_argument("first_order_sectors entries must have degree 1");
    }
    const int support_pair_index = flatten_support_pair_index(
        sector.row_orbitals[0],
        sector.col_orbitals[0],
        layout.support_size());
    const int flat_degree1_index =
        layout.flat_degree1_index(support_pair_index, flat_sector_index);
    entries.emplace_back(flat_degree1_index, sector.value);
  }
  return entries;
}

}  // namespace

HamiltonianBoundaryBundle make_zero_hamiltonian_boundary_bundle(
    const BoundarySpinBundleLayout& alpha_layout,
    const BoundarySpinBundleLayout& beta_layout) {
  HamiltonianBoundaryBundle bundle;
  bundle.overlap = 0.0;
  bundle.alpha = make_zero_boundary_spin_bundle(alpha_layout);
  bundle.beta = make_zero_boundary_spin_bundle(beta_layout);
  bundle.mixed = make_zero_boundary_mixed_bundle(
      alpha_layout.support_pair_count() * alpha_layout.total_sector_count(),
      beta_layout.support_pair_count() * beta_layout.total_sector_count());
  return bundle;
}

bool has_compatible_layout(const HamiltonianBoundaryBundle& bundle) {
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

void add_scaled_hamiltonian_boundary_bundle(
    const HamiltonianBoundaryBundle& source,
    double scale,
    HamiltonianBoundaryBundle* destination) {
  if (destination == nullptr) {
    throw std::invalid_argument("destination must not be null");
  }
  if (std::abs(scale) <= kZeroTolerance) {
    return;
  }
  validate_hamiltonian_bundle_layout_compatibility(source, *destination);
  destination->overlap += scale * source.overlap;
  add_scaled_boundary_spin_bundle(source.alpha, scale, &destination->alpha);
  add_scaled_boundary_spin_bundle(source.beta, scale, &destination->beta);
  add_scaled_boundary_mixed_bundle(source.mixed, scale, &destination->mixed);
}

void scatter_projected_spin_values_to_flat_sector(
    const HamiltonianProjectedSpinValues& projected,
    int flat_sector_index,
    double scale,
    BoundarySpinBundle* destination) {
  // Scatters one projected one-spin payload into one already selected
  // boundary sector. This is the dense-carrier adapter used by the planned
  // generic leaf migration from sparse deleted-sector payloads.
  if (destination == nullptr) {
    throw std::invalid_argument("destination must not be null");
  }
  if (std::abs(scale) <= kZeroTolerance) {
    return;
  }
  if (!has_compatible_layout(*destination)) {
    throw std::invalid_argument("destination spin bundle has incompatible layout");
  }
  if (flat_sector_index < 0 ||
      flat_sector_index >= destination->layout.total_sector_count()) {
    throw std::invalid_argument("flat_sector_index is out of range");
  }
  if (destination->layout.support_size() <= 0) {
    throw std::invalid_argument("destination layout support_size must be positive");
  }

  destination->degree0[xmvb::to_size(flat_sector_index)] +=
      scale * projected.overlap;

  for (const auto& sector : projected.same_spin_sectors) {
    if (std::abs(sector.value) <= kZeroTolerance) {
      continue;
    }
    std::vector<int> canonical_rows;
    std::vector<int> canonical_cols;
    const int parity = validate_and_canonicalize_sector_labels(
        sector.row_orbitals,
        sector.col_orbitals,
        &canonical_rows,
        &canonical_cols);
    const double signed_value =
        ((parity == 0) ? 1.0 : -1.0) * scale * sector.value;
    const int degree = static_cast<int>(canonical_rows.size());
    if (degree == 0) {
      destination->degree0[xmvb::to_size(flat_sector_index)] += signed_value;
      continue;
    }
    if (degree == 1) {
      const int support_pair_index = flatten_support_pair_index(
          canonical_rows[0],
          canonical_cols[0],
          destination->layout.support_size());
      const int flat_degree1_index =
          destination->layout.flat_degree1_index(
              support_pair_index,
              flat_sector_index);
      destination->degree1[xmvb::to_size(flat_degree1_index)] += signed_value;
      continue;
    }
    const int support_pair_pair_index = flatten_support_pair_pair_index(
        canonical_rows[0],
        canonical_rows[1],
        canonical_cols[0],
        canonical_cols[1],
        destination->layout.support_size());
    const int flat_degree2_index =
        destination->layout.flat_degree2_index(
            support_pair_pair_index,
            flat_sector_index);
    destination->degree2[xmvb::to_size(flat_degree2_index)] += signed_value;
  }
}

void accumulate_projected_mixed_first_order_outer_product(
    const HamiltonianProjectedSpinValues& alpha_projected,
    int alpha_flat_sector_index,
    const BoundarySpinBundleLayout& alpha_layout,
    const HamiltonianProjectedSpinValues& beta_projected,
    int beta_flat_sector_index,
    const BoundarySpinBundleLayout& beta_layout,
    double scale,
    BoundaryMixedBundle* destination) {
  // Builds the dense mixed degree-1 second moment for one fixed pair of
  // boundary sectors from projected alpha and beta first-order channels.
  if (destination == nullptr) {
    throw std::invalid_argument("destination must not be null");
  }
  if (std::abs(scale) <= kZeroTolerance) {
    return;
  }
  const int expected_alpha_degree1_entry_count =
      alpha_layout.support_pair_count() * alpha_layout.total_sector_count();
  const int expected_beta_degree1_entry_count =
      beta_layout.support_pair_count() * beta_layout.total_sector_count();
  if (destination->alpha_degree1_entry_count != expected_alpha_degree1_entry_count ||
      destination->beta_degree1_entry_count != expected_beta_degree1_entry_count ||
      destination->values.size() !=
          xmvb::to_size(expected_alpha_degree1_entry_count) *
              xmvb::to_size(expected_beta_degree1_entry_count)) {
    throw std::invalid_argument("destination mixed bundle dimensions do not match layouts");
  }

  const std::vector<std::pair<int, double>> alpha_entries =
      collect_projected_degree1_entries(
          alpha_projected,
          alpha_flat_sector_index,
          alpha_layout);
  const std::vector<std::pair<int, double>> beta_entries =
      collect_projected_degree1_entries(
          beta_projected,
          beta_flat_sector_index,
          beta_layout);
  for (const auto& [beta_index, beta_value] : beta_entries) {
    if (std::abs(beta_value) <= kZeroTolerance) {
      continue;
    }
    const std::size_t offset =
        xmvb::to_size(beta_index) *
        xmvb::to_size(destination->alpha_degree1_entry_count);
    for (const auto& [alpha_index, alpha_value] : alpha_entries) {
      if (std::abs(alpha_value) <= kZeroTolerance) {
        continue;
      }
      destination->values[offset + xmvb::to_size(alpha_index)] +=
          scale * alpha_value * beta_value;
    }
  }
}

void accumulate_projected_spin_pair_into_hamiltonian_channels(
    const HamiltonianProjectedSpinValues& alpha_projected,
    int alpha_flat_sector_index,
    const BoundarySpinBundleLayout& alpha_layout,
    const HamiltonianProjectedSpinValues& beta_projected,
    int beta_flat_sector_index,
    const BoundarySpinBundleLayout& beta_layout,
    double scale,
    double* overlap,
    BoundarySpinBundle* alpha,
    BoundarySpinBundle* beta,
    BoundaryMixedBundle* mixed) {
  if (overlap == nullptr ||
      alpha == nullptr ||
      beta == nullptr ||
      mixed == nullptr) {
    throw std::invalid_argument("Hamiltonian channel outputs must not be null");
  }
  if (std::abs(scale) <= kZeroTolerance) {
    return;
  }

  *overlap += scale * alpha_projected.overlap * beta_projected.overlap;
  scatter_projected_spin_values_to_flat_sector(
      alpha_projected,
      alpha_flat_sector_index,
      scale * beta_projected.overlap,
      alpha);
  scatter_projected_spin_values_to_flat_sector(
      beta_projected,
      beta_flat_sector_index,
      scale * alpha_projected.overlap,
      beta);
  accumulate_projected_mixed_first_order_outer_product(
      alpha_projected,
      alpha_flat_sector_index,
      alpha_layout,
      beta_projected,
      beta_flat_sector_index,
      beta_layout,
      scale,
      mixed);
}

void accumulate_weighted_spin_pair_into_hamiltonian_channels(
    const BoundarySpinBundle& alpha_bundle,
    const BoundarySpinBundle& beta_bundle,
    double coefficient,
    double* overlap,
    BoundarySpinBundle* alpha,
    BoundarySpinBundle* beta,
    BoundaryMixedBundle* mixed) {
  if (overlap == nullptr ||
      alpha == nullptr ||
      beta == nullptr ||
      mixed == nullptr) {
    throw std::invalid_argument("Hamiltonian channel outputs must not be null");
  }
  if (std::abs(coefficient) <= kZeroTolerance) {
    return;
  }
  validate_same_spin_layout(alpha_bundle, *alpha);
  validate_same_spin_layout(beta_bundle, *beta);
  const int expected_alpha_degree1_entry_count =
      alpha_bundle.layout.support_pair_count() *
      alpha_bundle.layout.total_sector_count();
  const int expected_beta_degree1_entry_count =
      beta_bundle.layout.support_pair_count() *
      beta_bundle.layout.total_sector_count();
  if (mixed->alpha_degree1_entry_count != expected_alpha_degree1_entry_count ||
      mixed->beta_degree1_entry_count != expected_beta_degree1_entry_count) {
    throw std::invalid_argument("destination mixed bundle dimensions do not match spin bundles");
  }

  const double alpha_overlap = contract_boundary_spin_bundle_overlap(alpha_bundle);
  const double beta_overlap = contract_boundary_spin_bundle_overlap(beta_bundle);
  *overlap += coefficient * alpha_overlap * beta_overlap;
  add_scaled_boundary_spin_bundle(
      alpha_bundle,
      coefficient * beta_overlap,
      alpha);
  add_scaled_boundary_spin_bundle(
      beta_bundle,
      coefficient * alpha_overlap,
      beta);

  const std::vector<int> alpha_nonzero_degree1_indices =
      collect_nonzero_boundary_spin_degree1_indices(alpha_bundle);
  const std::vector<int> beta_nonzero_degree1_indices =
      collect_nonzero_boundary_spin_degree1_indices(beta_bundle);
  for (const int beta_index : beta_nonzero_degree1_indices) {
    const double beta_value = beta_bundle.degree1[xmvb::to_size(beta_index)];
    if (std::abs(beta_value) <= kZeroTolerance) {
      continue;
    }
    const std::size_t offset =
        xmvb::to_size(beta_index) *
        xmvb::to_size(mixed->alpha_degree1_entry_count);
    for (const int alpha_index : alpha_nonzero_degree1_indices) {
      const double alpha_value = alpha_bundle.degree1[xmvb::to_size(alpha_index)];
      if (std::abs(alpha_value) <= kZeroTolerance) {
        continue;
      }
      mixed->values[offset + xmvb::to_size(alpha_index)] +=
          coefficient * beta_value * alpha_value;
    }
  }
}

void accumulate_weighted_spin_pair_into_hamiltonian_bundle(
    const BoundarySpinBundle& alpha_bundle,
    const BoundarySpinBundle& beta_bundle,
    double coefficient,
    HamiltonianBoundaryBundle* destination) {
  if (destination == nullptr) {
    throw std::invalid_argument("destination must not be null");
  }
  if (std::abs(coefficient) <= kZeroTolerance) {
    return;
  }
  validate_same_spin_layout(alpha_bundle, destination->alpha);
  validate_same_spin_layout(beta_bundle, destination->beta);
  const int expected_alpha_degree1_entry_count =
      alpha_bundle.layout.support_pair_count() *
      alpha_bundle.layout.total_sector_count();
  const int expected_beta_degree1_entry_count =
      beta_bundle.layout.support_pair_count() *
      beta_bundle.layout.total_sector_count();
  if (destination->mixed.alpha_degree1_entry_count != expected_alpha_degree1_entry_count ||
      destination->mixed.beta_degree1_entry_count != expected_beta_degree1_entry_count) {
    throw std::invalid_argument("destination mixed bundle dimensions do not match spin bundles");
  }

  accumulate_weighted_spin_pair_into_hamiltonian_channels(
      alpha_bundle,
      beta_bundle,
      coefficient,
      &destination->overlap,
      &destination->alpha,
      &destination->beta,
      &destination->mixed);
}

HamiltonianBoundaryBundleReadout contract_hamiltonian_boundary_bundle_channels(
    const HamiltonianBoundaryBundle& bundle,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size) {
  if (!has_compatible_layout(bundle)) {
    throw std::invalid_argument("bundle has incompatible dense channel layout");
  }
  if (bundle.alpha.layout.support_size() != support_size ||
      bundle.beta.layout.support_size() != support_size) {
    throw std::invalid_argument("bundle support_size is inconsistent with input support_size");
  }
  HamiltonianBoundaryBundleReadout result;
  result.overlap = bundle.overlap;
  result.one_electron =
      contract_boundary_spin_bundle_one_electron(
          bundle.alpha,
          support_one_electron_storage,
          support_size) +
      contract_boundary_spin_bundle_one_electron(
          bundle.beta,
          support_one_electron_storage,
          support_size);
  result.same_spin_alpha_two_electron =
      contract_boundary_spin_bundle_same_spin(
          bundle.alpha,
          packed_active_two_electron_integrals);
  result.same_spin_beta_two_electron =
      contract_boundary_spin_bundle_same_spin(
          bundle.beta,
          packed_active_two_electron_integrals);
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
