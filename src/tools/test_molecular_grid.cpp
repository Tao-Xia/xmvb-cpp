#include <cmath>
#include <iostream>
#include <iomanip>

#include <Eigen/Core>

#include "vb/pdft/molecular_grid.hpp"

using namespace xmvb::vb::pdft;

void test_molecular_grid_h2() {
  std::cout << "=== Testing Molecular Grid for H2 ===" << std::endl;

  // H2 molecule at equilibrium distance
  Eigen::MatrixXd atomic_coords(2, 3);
  atomic_coords << 0.0, 0.0, 0.0,
                   0.0, 0.0, 1.4;

  Eigen::VectorXd atomic_charges(2);
  atomic_charges << 1.0, 1.0;

  // Build grid with coarse settings for testing
  MolecularGridConfig config;
  config.radial_points = 20;
  config.angular_points = 26;
  config.becke_exponent = 3;

  MolecularGridBuilder builder(config);

  std::cout << "Building grid with:" << std::endl;
  std::cout << "  Radial points: " << config.radial_points << std::endl;
  std::cout << "  Angular points: " << config.angular_points << std::endl;
  std::cout << "  Expected total: " << 2 * config.radial_points * config.angular_points
            << " points" << std::endl;

  MolecularGrid grid = builder.build(atomic_coords, atomic_charges);

  std::cout << "\nGrid generated:" << std::endl;
  std::cout << "  Actual points: " << grid.n_points() << std::endl;

  // Check weight sum (should integrate to volume, but for molecular grid
  // we mainly care that weights are positive and reasonable)
  double weight_sum = grid.weights.sum();
  std::cout << "  Total weight: " << std::scientific << weight_sum << std::endl;

  // Check that all weights are positive
  bool all_positive = true;
  double min_weight = grid.weights.minCoeff();
  double max_weight = grid.weights.maxCoeff();

  if (min_weight < 0.0) {
    all_positive = false;
  }

  std::cout << "  Min weight: " << min_weight << std::endl;
  std::cout << "  Max weight: " << max_weight << std::endl;
  std::cout << "  All positive: " << (all_positive ? "YES" : "NO") << std::endl;

  // Check atom assignments
  int n_atom0 = 0;
  int n_atom1 = 0;
  for (int atom : grid.atom_assignments) {
    if (atom == 0) n_atom0++;
    else if (atom == 1) n_atom1++;
  }

  std::cout << "\nAtom assignments:" << std::endl;
  std::cout << "  Atom 0: " << n_atom0 << " points" << std::endl;
  std::cout << "  Atom 1: " << n_atom1 << " points" << std::endl;

  if (n_atom0 == n_atom1) {
    std::cout << "  Symmetry: PASS (equal points per atom)" << std::endl;
  } else {
    std::cout << "  Symmetry: Note - unequal distribution" << std::endl;
  }

  std::cout << std::endl;
}

void test_molecular_grid_performance() {
  std::cout << "=== Testing Molecular Grid Performance ===" << std::endl;

  // H2 molecule
  Eigen::MatrixXd atomic_coords(2, 3);
  atomic_coords << 0.0, 0.0, 0.0,
                   0.0, 0.0, 1.4;

  Eigen::VectorXd atomic_charges(2);
  atomic_charges << 1.0, 1.0;

  // Test different grid sizes
  const std::vector<std::pair<int, int>> grid_sizes = {
      {20, 26},   // Coarse
      {50, 50},   // Medium
      {50, 302},  // Fine
  };

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "\nGrid size tests:" << std::endl;
  std::cout << "  Radial  Angular  Total Points  Weight Sum" << std::endl;
  std::cout << "  ------  -------  ------------  ----------" << std::endl;

  for (const auto& size : grid_sizes) {
    MolecularGridConfig config;
    config.radial_points = size.first;
    config.angular_points = size.second;

    MolecularGridBuilder builder(config);
    MolecularGrid grid = builder.build(atomic_coords, atomic_charges);

    std::cout << "  " << std::setw(6) << config.radial_points
              << "  " << std::setw(7) << config.angular_points
              << "  " << std::setw(12) << grid.n_points()
              << "  " << std::scientific << std::setprecision(6)
              << grid.weights.sum() << std::endl;
  }

  std::cout << std::endl;
}

int main() {
  try {
    test_molecular_grid_h2();
    test_molecular_grid_performance();

    std::cout << "All molecular grid tests completed." << std::endl;
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
}
