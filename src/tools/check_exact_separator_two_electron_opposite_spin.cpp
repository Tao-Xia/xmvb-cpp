#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "vb/exact_separator/two_electron.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace {

using xmvb::vb::DeterminantOverlapResolver;
using xmvb::vb::TwoElectronIndexer;
using xmvb::vb::exact_separator::CollapsedOppositeSpinOneLeafStarPairStats;
using xmvb::vb::exact_separator::ComponentData;
using xmvb::vb::exact_separator::OrientationTerm;

int main_impl() {
  // Nontrivial one-root/one-leaf opposite-spin validation case.
  //
  // The root and leaf components together span two alpha orbitals and two beta
  // orbitals, with the alpha bra side intentionally ordered as `[1] + [0]` so
  // the collapsed path must apply the same canonicalization parity as the
  // exact determinant-space baseline.
  ComponentData root_component;
  root_component.graph_node = 0;
  root_component.left_orientation_terms.push_back(OrientationTerm{{1}, {0}, 1.0});
  root_component.right_orientation_terms.push_back(OrientationTerm{{0}, {0}, 1.0});

  ComponentData leaf_component;
  leaf_component.graph_node = 1;
  leaf_component.left_orientation_terms.push_back(OrientationTerm{{0}, {1}, 1.0});
  leaf_component.right_orientation_terms.push_back(OrientationTerm{{1}, {1}, 1.0});

  const std::vector<ComponentData> ordered_components{
      root_component,
      leaf_component,
  };

  const std::vector<double> support_overlap_storage{
      1.0, 0.15,
      0.25, 0.95,
  };

  const int packed_eri_size =
      TwoElectronIndexer::two_electron_storage_index(1, 1, 1, 1) + 1;
  std::vector<double> packed_active_two_electron_integrals(
      xmvb::to_size(packed_eri_size),
      0.0);
  packed_active_two_electron_integrals[xmvb::to_size(
      TwoElectronIndexer::two_electron_storage_index(0, 0, 0, 0))] = 1.2;
  packed_active_two_electron_integrals[xmvb::to_size(
      TwoElectronIndexer::two_electron_storage_index(1, 1, 1, 1))] = 1.4;
  packed_active_two_electron_integrals[xmvb::to_size(
      TwoElectronIndexer::two_electron_storage_index(0, 0, 1, 1))] = 0.8;
  packed_active_two_electron_integrals[xmvb::to_size(
      TwoElectronIndexer::two_electron_storage_index(1, 1, 0, 0))] = 0.8;
  packed_active_two_electron_integrals[xmvb::to_size(
      TwoElectronIndexer::two_electron_storage_index(0, 1, 0, 1))] = 0.4;
  packed_active_two_electron_integrals[xmvb::to_size(
      TwoElectronIndexer::two_electron_storage_index(1, 0, 1, 0))] = 0.45;
  packed_active_two_electron_integrals[xmvb::to_size(
      TwoElectronIndexer::two_electron_storage_index(1, 0, 0, 1))] = 0.35;

  const CollapsedOppositeSpinOneLeafStarPairStats stats =
      xmvb::vb::exact_separator::
          evaluate_component_ordered_open_state_star_pair_two_electron_opposite_spin_one_leaf(
              support_overlap_storage,
              packed_active_two_electron_integrals,
              2,
              ordered_components,
              DeterminantOverlapResolver());

  std::cout << std::setprecision(15);
  std::cout << "exact_overlap = " << stats.exact_overlap << '\n';
  std::cout << "collapsed_overlap = " << stats.collapsed_overlap << '\n';
  std::cout << "exact_opposite_spin = "
            << stats.exact_opposite_spin_two_electron << '\n';
  std::cout << "collapsed_opposite_spin = "
            << stats.collapsed_opposite_spin_two_electron << '\n';
  std::cout << "overlap_absolute_error = " << stats.overlap_absolute_error << '\n';
  std::cout << "opposite_spin_absolute_error = "
            << stats.opposite_spin_absolute_error << '\n';
  std::cout << "processed_term_quadruple_count = "
            << stats.processed_term_quadruple_count << '\n';
  std::cout << "alpha_mask_state_count = " << stats.alpha_mask_state_count << '\n';
  std::cout << "beta_mask_state_count = " << stats.beta_mask_state_count << '\n';
  std::cout << "subdeterminant_evaluations = "
            << stats.subdeterminant_evaluations << '\n';

  const bool overlap_ok = stats.overlap_absolute_error <= 1.0e-12;
  const bool opposite_spin_ok = stats.opposite_spin_absolute_error <= 1.0e-12;
  const bool nontrivial_values =
      std::abs(stats.exact_overlap) > 1.0e-10 &&
      std::abs(stats.exact_opposite_spin_two_electron) > 1.0e-10;
  if (!overlap_ok || !opposite_spin_ok || !nontrivial_values) {
    std::cerr << "collapsed one-leaf opposite-spin separator test failed\n";
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
