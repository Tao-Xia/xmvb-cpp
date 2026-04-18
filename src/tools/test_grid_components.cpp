#include <cmath>
#include <iostream>
#include <iomanip>

#include "vb/pdft/lebedev_grid.hpp"
#include "vb/pdft/radial_grid.hpp"

using namespace xmvb::vb::pdft;

void test_lebedev_grid() {
  std::cout << "=== Testing Lebedev Grids ===" << std::endl;

  const std::vector<int> test_sizes = {6, 14, 26, 38, 50, 302};

  for (int size : test_sizes) {
    auto grid = LebedevGrid::get_grid(size);

    // Check number of points
    if (static_cast<int>(grid.size()) != size) {
      std::cout << "ERROR: Expected " << size << " points, got "
                << grid.size() << std::endl;
      continue;
    }

    // Check weight normalization (should sum to 4*pi)
    double weight_sum = 0.0;
    for (const auto& point : grid) {
      weight_sum += point.weight;
    }

    const double expected_sum = 4.0 * M_PI;
    const double error = std::abs(weight_sum - expected_sum);

    std::cout << "Lebedev-" << std::setw(3) << size
              << ": weight_sum = " << std::setprecision(12) << weight_sum
              << ", error = " << std::scientific << error;

    if (error < 1.0e-10) {
      std::cout << " [PASS]" << std::endl;
    } else {
      std::cout << " [FAIL]" << std::endl;
    }
  }

  std::cout << std::endl;
}

void test_radial_grid() {
  std::cout << "=== Testing Radial Grids ===" << std::endl;

  const int n_points = 50;
  const std::vector<int> test_atoms = {1, 6, 8};  // H, C, O

  for (int Z : test_atoms) {
    auto grid = RadialGridBuilder::build(
        n_points, Z, RadialGridScheme::MuraKnowles);

    if (static_cast<int>(grid.size()) != n_points) {
      std::cout << "ERROR: Expected " << n_points << " points, got "
                << grid.size() << std::endl;
      continue;
    }

    // Check that radii are increasing
    bool monotonic = true;
    for (size_t i = 1; i < grid.size(); ++i) {
      if (grid[i].r <= grid[i-1].r) {
        monotonic = false;
        break;
      }
    }

    // Check that all weights are positive
    bool positive_weights = true;
    for (const auto& point : grid) {
      if (point.weight <= 0.0) {
        positive_weights = false;
        break;
      }
    }

    std::cout << "Radial grid for Z=" << Z
              << " (" << n_points << " points):"
              << " r_min=" << std::scientific << grid.front().r
              << ", r_max=" << grid.back().r
              << ", monotonic=" << (monotonic ? "yes" : "no")
              << ", positive_weights=" << (positive_weights ? "yes" : "no");

    if (monotonic && positive_weights) {
      std::cout << " [PASS]" << std::endl;
    } else {
      std::cout << " [FAIL]" << std::endl;
    }
  }

  std::cout << std::endl;
}

int main() {
  std::cout << std::fixed;

  try {
    test_lebedev_grid();
    test_radial_grid();

    std::cout << "All tests completed." << std::endl;
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
}
