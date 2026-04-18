#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "vb/exact_separator/two_electron.hpp"
#include "vb/matrices/legacy_structure_overlap.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace {

using xmvb::vb::DeterminantOverlapResolver;
using xmvb::vb::OrbitalPair;
using xmvb::vb::TwoElectronIndexer;
using xmvb::vb::exact_separator::CollapsedTwoElectronStarPairStats;
using xmvb::vb::exact_separator::ComponentData;
using xmvb::vb::exact_separator::OrientationTerm;

std::vector<OrientationTerm> enumerate_orientation_terms(
    const std::vector<OrbitalPair>& pairs) {
  const auto legacy_terms = xmvb::vb::enumerate_legacy_determinant_terms(pairs);
  std::vector<OrientationTerm> terms;
  terms.reserve(legacy_terms.size());
  for (const auto& legacy_term : legacy_terms) {
    terms.push_back(OrientationTerm{
        .alpha_occ = legacy_term.alpha_occ,
        .beta_occ = legacy_term.beta_occ,
        .coefficient = legacy_term.coefficient,
    });
  }
  return terms;
}

std::vector<double> flatten_column_major(
    const std::vector<std::vector<double>>& row_major_matrix) {
  if (row_major_matrix.empty()) {
    return {};
  }
  const int n_rows = static_cast<int>(row_major_matrix.size());
  const int n_cols = static_cast<int>(row_major_matrix.front().size());
  std::vector<double> storage(
      xmvb::to_size(n_rows * n_cols),
      0.0);
  for (int row = 0; row < n_rows; ++row) {
    if (static_cast<int>(row_major_matrix[xmvb::to_size(row)].size()) != n_cols) {
      throw std::invalid_argument("row_major_matrix must be rectangular");
    }
    for (int col = 0; col < n_cols; ++col) {
      storage[xmvb::to_size(col * n_rows + row)] =
          row_major_matrix[xmvb::to_size(row)][xmvb::to_size(col)];
    }
  }
  return storage;
}

int main_impl() {
  // Exact three-component star:
  //   root  = (0,1)
  //   leaf1 = (2,3)
  //   leaf2 = (4,5)
  //
  // All three components use the exact local legacy determinant expansion, so
  // the general multi-leaf star recurrence is validated directly against the
  // merged determinant-space two-electron baseline carried by the library.
  ComponentData root_component;
  root_component.graph_node = 0;
  root_component.left_pairs = {{0, 1}};
  root_component.right_pairs = {{0, 1}};
  root_component.left_orientation_terms =
      enumerate_orientation_terms(root_component.left_pairs);
  root_component.right_orientation_terms =
      enumerate_orientation_terms(root_component.right_pairs);

  ComponentData leaf1_component;
  leaf1_component.graph_node = 1;
  leaf1_component.left_pairs = {{2, 3}};
  leaf1_component.right_pairs = {{2, 3}};
  leaf1_component.left_orientation_terms =
      enumerate_orientation_terms(leaf1_component.left_pairs);
  leaf1_component.right_orientation_terms =
      enumerate_orientation_terms(leaf1_component.right_pairs);

  ComponentData leaf2_component;
  leaf2_component.graph_node = 2;
  leaf2_component.left_pairs = {{4, 5}};
  leaf2_component.right_pairs = {{4, 5}};
  leaf2_component.left_orientation_terms =
      enumerate_orientation_terms(leaf2_component.left_pairs);
  leaf2_component.right_orientation_terms =
      enumerate_orientation_terms(leaf2_component.right_pairs);

  const std::vector<ComponentData> ordered_components{
      root_component,
      leaf1_component,
      leaf2_component,
  };

  // Support-overlap matrix for a genuine star separator:
  // - root component `(0,1)` couples to both leaves;
  // - leaf1 `(2,3)` and leaf2 `(4,5)` have zero direct cross blocks.
  const std::vector<double> support_overlap_storage = flatten_column_major({
      {1.00, 0.08, 0.12, 0.05, 0.09, 0.04},
      {0.11, 0.97, 0.07, 0.10, 0.03, 0.06},
      {0.14, 0.09, 0.95, 0.11, 0.00, 0.00},
      {0.06, 0.12, 0.10, 0.98, 0.00, 0.00},
      {0.07, 0.04, 0.00, 0.00, 0.96, 0.10},
      {0.05, 0.07, 0.00, 0.00, 0.09, 0.99},
  });

  const int packed_eri_size =
      TwoElectronIndexer::two_electron_storage_index(5, 5, 5, 5) + 1;
  std::vector<double> packed_active_two_electron_integrals(
      xmvb::to_size(packed_eri_size),
      0.0);
  for (int p = 0; p < 6; ++p) {
    for (int q = 0; q < 6; ++q) {
      for (int r = 0; r < 6; ++r) {
        for (int s = 0; s < 6; ++s) {
          const double value =
              0.25 + 0.03 * (p + 1) + 0.02 * (q + 1) + 0.015 * (r + 1) +
              0.01 * (s + 1);
          packed_active_two_electron_integrals[xmvb::to_size(
              TwoElectronIndexer::two_electron_storage_index(p, q, r, s))] = value;
        }
      }
    }
  }

  const CollapsedTwoElectronStarPairStats stats =
      xmvb::vb::exact_separator::evaluate_component_ordered_open_state_star_pair_two_electron(
          support_overlap_storage,
          packed_active_two_electron_integrals,
          6,
          ordered_components,
          DeterminantOverlapResolver());
  std::cout << std::setprecision(15);
  std::cout << "exact_overlap = " << stats.exact_overlap << '\n';
  std::cout << "collapsed_overlap = " << stats.collapsed_overlap << '\n';
  std::cout << "exact_same_spin_alpha = "
            << stats.exact_same_spin_alpha_two_electron << '\n';
  std::cout << "collapsed_same_spin_alpha = "
            << stats.collapsed_same_spin_alpha_two_electron << '\n';
  std::cout << "exact_same_spin_beta = "
            << stats.exact_same_spin_beta_two_electron << '\n';
  std::cout << "collapsed_same_spin_beta = "
            << stats.collapsed_same_spin_beta_two_electron << '\n';
  std::cout << "exact_opposite_spin = "
            << stats.exact_opposite_spin_two_electron << '\n';
  std::cout << "collapsed_opposite_spin = "
            << stats.collapsed_opposite_spin_two_electron << '\n';
  std::cout << "exact_total = " << stats.exact_two_electron << '\n';
  std::cout << "collapsed_total = " << stats.collapsed_two_electron << '\n';
  std::cout << "overlap_absolute_error = " << stats.overlap_absolute_error << '\n';
  std::cout << "same_spin_alpha_absolute_error = "
            << stats.same_spin_alpha_absolute_error << '\n';
  std::cout << "same_spin_beta_absolute_error = "
            << stats.same_spin_beta_absolute_error << '\n';
  std::cout << "opposite_spin_absolute_error = "
            << stats.opposite_spin_absolute_error << '\n';
  std::cout << "total_two_electron_absolute_error = "
            << stats.total_two_electron_absolute_error << '\n';
  std::cout << "collapsed_leaf_state_count = "
            << stats.collapsed_leaf_state_count << '\n';
  std::cout << "hypercube_assignment_count = "
            << stats.hypercube_assignment_count << '\n';
  std::cout << "subdeterminant_evaluations = "
            << stats.subdeterminant_evaluations << '\n';
  std::cout << "dp_transition_count = " << stats.dp_transition_count << '\n';

  const bool overlap_ok = stats.overlap_absolute_error <= 1.0e-10;
  const bool alpha_ok = stats.same_spin_alpha_absolute_error <= 1.0e-10;
  const bool beta_ok = stats.same_spin_beta_absolute_error <= 1.0e-10;
  const bool opposite_ok = stats.opposite_spin_absolute_error <= 1.0e-10;
  const bool total_ok = stats.total_two_electron_absolute_error <= 1.0e-10;
  if (!overlap_ok || !alpha_ok || !beta_ok || !opposite_ok || !total_ok) {
    std::cerr << "collapsed multi-leaf two-electron separator test failed\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

}  // namespace

int main() {
  try {
    return main_impl();
  } catch (const std::exception& exception) {
    std::cerr << "error: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
}
