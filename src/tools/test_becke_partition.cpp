#include <cmath>
#include <iostream>
#include <iomanip>

#include <Eigen/Core>

#include "vb/pdft/becke_partition.hpp"

using namespace xmvb::vb::pdft;

void test_becke_partition_h2() {
  std::cout << "=== Testing Becke Partition for H2 ===" << std::endl;

  // H2 molecule at equilibrium distance (1.4 Bohr)
  Eigen::MatrixXd atomic_coords(2, 3);
  atomic_coords << 0.0, 0.0, 0.0,
                   0.0, 0.0, 1.4;

  Eigen::VectorXi atomic_numbers(2);
  atomic_numbers << 1, 1;  // H, H

  // Test points along the bond axis
  const int n_test_points = 11;
  Eigen::MatrixXd test_points(n_test_points, 3);

  std::cout << "\nTest points along z-axis from -1.0 to 2.4 Bohr:" << std::endl;
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "    z        w_A        w_B      sum" << std::endl;
  std::cout << "---------------------------------------" << std::endl;

  for (int i = 0; i < n_test_points; ++i) {
    const double z = -1.0 + i * 0.35;
    test_points.row(i) << 0.0, 0.0, z;
  }

  // Compute Becke weights
  Eigen::MatrixXd weights = BeckePartition::compute_weights(
      test_points, atomic_coords, atomic_numbers, 3);

  // Check results
  bool all_pass = true;
  for (int i = 0; i < n_test_points; ++i) {
    const double z = test_points(i, 2);
    const double w_A = weights(i, 0);
    const double w_B = weights(i, 1);
    const double sum = w_A + w_B;

    std::cout << std::setw(7) << z << "  "
              << std::setw(9) << w_A << "  "
              << std::setw(9) << w_B << "  "
              << std::setw(9) << sum;

    // Check normalization
    if (std::abs(sum - 1.0) > 1.0e-10) {
      std::cout << " [FAIL: sum != 1]";
      all_pass = false;
    } else {
      std::cout << " [OK]";
    }
    std::cout << std::endl;
  }

  std::cout << "\nNormalization test: " << (all_pass ? "PASS" : "FAIL") << std::endl;

  // Check symmetry at midpoint
  const double z_mid = 0.7;  // Midpoint between atoms
  Eigen::MatrixXd midpoint(1, 3);
  midpoint << 0.0, 0.0, z_mid;

  Eigen::MatrixXd mid_weights = BeckePartition::compute_weights(
      midpoint, atomic_coords, atomic_numbers, 3);

  const double w_mid_A = mid_weights(0, 0);
  const double w_mid_B = mid_weights(0, 1);

  std::cout << "\nSymmetry test at midpoint (z=" << z_mid << "):" << std::endl;
  std::cout << "  w_A = " << w_mid_A << std::endl;
  std::cout << "  w_B = " << w_mid_B << std::endl;
  std::cout << "  |w_A - w_B| = " << std::abs(w_mid_A - w_mid_B) << std::endl;

  if (std::abs(w_mid_A - w_mid_B) < 1.0e-10) {
    std::cout << "  Symmetry: PASS" << std::endl;
  } else {
    std::cout << "  Symmetry: FAIL (should be equal for symmetric molecule)" << std::endl;
  }

  std::cout << std::endl;
}

void test_becke_partition_grid() {
  std::cout << "=== Testing Becke Partition on 3D Grid ===" << std::endl;

  // H2 molecule
  Eigen::MatrixXd atomic_coords(2, 3);
  atomic_coords << 0.0, 0.0, 0.0,
                   0.0, 0.0, 1.4;

  Eigen::VectorXi atomic_numbers(2);
  atomic_numbers << 1, 1;

  // Create a small 3D grid
  const int n_grid = 5;
  const double grid_range = 2.0;
  const int n_points = n_grid * n_grid * n_grid;

  Eigen::MatrixXd grid_points(n_points, 3);
  int idx = 0;
  for (int ix = 0; ix < n_grid; ++ix) {
    for (int iy = 0; iy < n_grid; ++iy) {
      for (int iz = 0; iz < n_grid; ++iz) {
        const double x = -grid_range + ix * (2.0 * grid_range / (n_grid - 1));
        const double y = -grid_range + iy * (2.0 * grid_range / (n_grid - 1));
        const double z = -grid_range + iz * (2.0 * grid_range / (n_grid - 1));
        grid_points.row(idx++) << x, y, z;
      }
    }
  }

  std::cout << "Computing weights for " << n_points << " grid points..." << std::endl;

  Eigen::MatrixXd weights = BeckePartition::compute_weights(
      grid_points, atomic_coords, atomic_numbers, 3);

  // Check normalization for all points
  bool all_normalized = true;
  double max_error = 0.0;

  for (int i = 0; i < n_points; ++i) {
    const double sum = weights.row(i).sum();
    const double error = std::abs(sum - 1.0);
    max_error = std::max(max_error, error);

    if (error > 1.0e-10) {
      all_normalized = false;
    }
  }

  std::cout << "Normalization check:" << std::endl;
  std::cout << "  Max error: " << std::scientific << max_error << std::endl;
  std::cout << "  Result: " << (all_normalized ? "PASS" : "FAIL") << std::endl;

  std::cout << std::endl;
}

int main() {
  try {
    test_becke_partition_h2();
    test_becke_partition_grid();

    std::cout << "All Becke partition tests completed." << std::endl;
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
}
