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
using xmvb::vb::exact_separator::ComponentData;
using xmvb::vb::exact_separator::ExactTwoElectronStarPairStats;
using xmvb::vb::exact_separator::OrientationTerm;

int main_impl() {
  // Smoke test 1: opposite-spin single-orbital pair.
  // One alpha and one beta electron occupy the same orbital on both sides,
  // with unit overlap and `(0 0 | 0 0) = 2.5`. The pure two-electron matrix
  // element must be entirely opposite-spin.
  ComponentData opposite_spin_component;
  opposite_spin_component.graph_node = 0;
  opposite_spin_component.left_orientation_terms.push_back(OrientationTerm{{0}, {0}, 1.0});
  opposite_spin_component.right_orientation_terms.push_back(OrientationTerm{{0}, {0}, 1.0});
  const std::vector<ComponentData> opposite_spin_components{opposite_spin_component};
  const std::vector<double> one_orbital_overlap_storage{1.0};
  std::vector<double> one_orbital_two_electron_integrals(1, 0.0);
  one_orbital_two_electron_integrals[xmvb::to_size(
      TwoElectronIndexer::two_electron_storage_index(0, 0, 0, 0))] = 2.5;

  const ExactTwoElectronStarPairStats opposite_spin_result =
      xmvb::vb::exact_separator::evaluate_component_ordered_open_state_star_pair_two_electron_exact(
          one_orbital_overlap_storage,
          one_orbital_two_electron_integrals,
          1,
          opposite_spin_components,
          DeterminantOverlapResolver());

  // Smoke test 2: same-spin two-orbital pair.
  // Two alpha electrons occupy orbitals 0 and 1 on both sides, with unit
  // overlap. Setting `(0 0 | 1 1) = 3.0` and `(0 1 | 1 0) = 0.5` gives the
  // exact same-spin contribution `3.0 - 0.5 = 2.5`.
  ComponentData same_spin_component;
  same_spin_component.graph_node = 0;
  same_spin_component.left_orientation_terms.push_back(OrientationTerm{{0, 1}, {}, 1.0});
  same_spin_component.right_orientation_terms.push_back(OrientationTerm{{0, 1}, {}, 1.0});
  const std::vector<ComponentData> same_spin_components{same_spin_component};
  const std::vector<double> two_orbital_overlap_storage{
      1.0, 0.0,
      0.0, 1.0,
  };
  const int two_orbital_eri_size =
      TwoElectronIndexer::two_electron_storage_index(1, 1, 1, 1) + 1;
  std::vector<double> two_orbital_two_electron_integrals(
      xmvb::to_size(two_orbital_eri_size),
      0.0);
  two_orbital_two_electron_integrals[xmvb::to_size(
      TwoElectronIndexer::two_electron_storage_index(0, 0, 1, 1))] = 3.0;
  two_orbital_two_electron_integrals[xmvb::to_size(
      TwoElectronIndexer::two_electron_storage_index(0, 1, 1, 0))] = 0.5;

  const ExactTwoElectronStarPairStats same_spin_result =
      xmvb::vb::exact_separator::evaluate_component_ordered_open_state_star_pair_two_electron_exact(
          two_orbital_overlap_storage,
          two_orbital_two_electron_integrals,
          2,
          same_spin_components,
          DeterminantOverlapResolver());

  std::cout << std::setprecision(15);
  std::cout << "opposite_spin_overlap = " << opposite_spin_result.exact_overlap << '\n';
  std::cout << "opposite_spin_total = " << opposite_spin_result.exact_two_electron << '\n';
  std::cout << "opposite_spin_same_spin_alpha = "
            << opposite_spin_result.exact_same_spin_alpha_two_electron << '\n';
  std::cout << "opposite_spin_same_spin_beta = "
            << opposite_spin_result.exact_same_spin_beta_two_electron << '\n';
  std::cout << "opposite_spin_channel = "
            << opposite_spin_result.exact_opposite_spin_two_electron << '\n';
  std::cout << "same_spin_overlap = " << same_spin_result.exact_overlap << '\n';
  std::cout << "same_spin_total = " << same_spin_result.exact_two_electron << '\n';
  std::cout << "same_spin_alpha_channel = "
            << same_spin_result.exact_same_spin_alpha_two_electron << '\n';
  std::cout << "same_spin_beta_channel = "
            << same_spin_result.exact_same_spin_beta_two_electron << '\n';
  std::cout << "same_spin_opposite_channel = "
            << same_spin_result.exact_opposite_spin_two_electron << '\n';

  const bool opposite_overlap_ok =
      std::abs(opposite_spin_result.exact_overlap - 1.0) <= 1.0e-12;
  const bool opposite_total_ok =
      std::abs(opposite_spin_result.exact_two_electron - 2.5) <= 1.0e-12;
  const bool opposite_channel_ok =
      std::abs(opposite_spin_result.exact_same_spin_alpha_two_electron) <= 1.0e-12 &&
      std::abs(opposite_spin_result.exact_same_spin_beta_two_electron) <= 1.0e-12 &&
      std::abs(opposite_spin_result.exact_opposite_spin_two_electron - 2.5) <= 1.0e-12;
  const bool same_overlap_ok =
      std::abs(same_spin_result.exact_overlap - 1.0) <= 1.0e-12;
  const bool same_total_ok =
      std::abs(same_spin_result.exact_two_electron - 2.5) <= 1.0e-12;
  const bool same_channel_ok =
      std::abs(same_spin_result.exact_same_spin_alpha_two_electron - 2.5) <= 1.0e-12 &&
      std::abs(same_spin_result.exact_same_spin_beta_two_electron) <= 1.0e-12 &&
      std::abs(same_spin_result.exact_opposite_spin_two_electron) <= 1.0e-12;
  if (!opposite_overlap_ok || !opposite_total_ok || !opposite_channel_ok ||
      !same_overlap_ok || !same_total_ok || !same_channel_ok) {
    std::cerr << "exact-separator two-electron smoke test failed\n";
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
