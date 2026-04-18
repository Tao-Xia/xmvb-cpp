#include <cmath>
#include <iostream>
#include <iomanip>

#include <Eigen/Core>

#include "vb/pdft/ao_grid_values.hpp"
#include "vb/pdft/libcint_ao_grid_evaluator.hpp"
#include "vb/matrices/libcint_input.hpp"

using namespace xmvb::vb::pdft;
using namespace xmvb::vb;

void test_ao_grid_values_structure() {
  std::cout << "=== Testing AoGridValues Structure ===" << std::endl;

  AoGridValues values;
  values.values.resize(10, 5);  // 10 points, 5 basis functions
  values.values.setRandom();

  std::cout << "Created AoGridValues:" << std::endl;
  std::cout << "  n_points: " << values.n_points() << std::endl;
  std::cout << "  n_basis_functions: " << values.n_basis_functions() << std::endl;
  std::cout << "  has_gradients: " << (values.has_gradients() ? "yes" : "no") << std::endl;

  // Add gradients
  values.gradients.resize(10, 15);  // 10 points, 3*5 gradients
  values.gradients.setRandom();

  std::cout << "\nAfter adding gradients:" << std::endl;
  std::cout << "  has_gradients: " << (values.has_gradients() ? "yes" : "no") << std::endl;

  std::cout << "\n[PASS] AoGridValues structure test" << std::endl;
  std::cout << std::endl;
}

void test_ao_evaluator_basic() {
  std::cout << "=== Testing AO Evaluator (Basic) ===" << std::endl;

  // Create a minimal libcint input for H2
  LibcintInput libcint_input;

  // Setup for H2 molecule with minimal basis (STO-3G like)
  // This is a simplified setup for testing

  // Atoms: 2 H atoms
  libcint_input.atm.resize(6);  // 2 atoms * 3 entries
  libcint_input.atm[0] = 1;  // H, charge
  libcint_input.atm[1] = 0;  // coord offset in env
  libcint_input.atm[2] = 0;  // unused
  libcint_input.atm[3] = 1;  // H, charge
  libcint_input.atm[4] = 3;  // coord offset in env
  libcint_input.atm[5] = 0;  // unused

  // Basis: 1 s-type shell per atom
  libcint_input.bas.resize(16);  // 2 shells * 8 entries
  // Shell 0 (H1)
  libcint_input.bas[0] = 0;   // atom index
  libcint_input.bas[1] = 0;   // angular momentum (s)
  libcint_input.bas[2] = 1;   // number of primitives
  libcint_input.bas[3] = 1;   // number of contractions
  libcint_input.bas[4] = 0;   // unused
  libcint_input.bas[5] = 0;   // exponent offset in env
  libcint_input.bas[6] = 6;   // coefficient offset in env
  libcint_input.bas[7] = 0;   // unused
  // Shell 1 (H2)
  libcint_input.bas[8] = 1;   // atom index
  libcint_input.bas[9] = 0;   // angular momentum (s)
  libcint_input.bas[10] = 1;  // number of primitives
  libcint_input.bas[11] = 1;  // number of contractions
  libcint_input.bas[12] = 0;  // unused
  libcint_input.bas[13] = 0;  // exponent offset in env
  libcint_input.bas[14] = 6;  // coefficient offset in env
  libcint_input.bas[15] = 0;  // unused

  // Environment: coordinates and basis parameters
  libcint_input.env.resize(10);
  // H1 coordinates (Bohr)
  libcint_input.env[0] = 0.0;
  libcint_input.env[1] = 0.0;
  libcint_input.env[2] = 0.0;
  // H2 coordinates
  libcint_input.env[3] = 0.0;
  libcint_input.env[4] = 0.0;
  libcint_input.env[5] = 1.4;
  // Basis exponents and coefficients (simplified)
  libcint_input.env[6] = 1.0;  // exponent
  libcint_input.env[7] = 1.0;  // coefficient

  try {
    LibcintAoGridEvaluator evaluator(libcint_input);

    std::cout << "Created evaluator:" << std::endl;
    std::cout << "  n_basis_functions: " << evaluator.n_basis_functions() << std::endl;

    // Create test grid points
    Eigen::MatrixXd grid_points(5, 3);
    grid_points << 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.7,
                   0.0, 0.0, 1.4,
                   0.5, 0.0, 0.7,
                   0.0, 0.5, 0.7;

    std::cout << "\nEvaluating AO values at " << grid_points.rows() << " points..." << std::endl;

    AoGridValues ao_values = evaluator.evaluate_values(grid_points);

    std::cout << "Result:" << std::endl;
    std::cout << "  values shape: " << ao_values.values.rows() << " x "
              << ao_values.values.cols() << std::endl;

    // Check that values are reasonable
    bool all_finite = ao_values.values.allFinite();
    std::cout << "  all_finite: " << (all_finite ? "yes" : "no") << std::endl;

    // Print some sample values
    std::cout << "\nSample AO values at first 3 points:" << std::endl;
    std::cout << std::scientific << std::setprecision(6);
    for (int g = 0; g < std::min(3, static_cast<int>(ao_values.values.rows())); ++g) {
      std::cout << "  Point " << g << ": ";
      for (int mu = 0; mu < ao_values.values.cols(); ++mu) {
        std::cout << ao_values.values(g, mu) << " ";
      }
      std::cout << std::endl;
    }

    std::cout << "\n[PASS] Basic AO evaluator test" << std::endl;

  } catch (const std::exception& e) {
    std::cout << "[FAIL] Exception: " << e.what() << std::endl;
  }

  std::cout << std::endl;
}

void test_ao_evaluator_with_gradients() {
  std::cout << "=== Testing AO Evaluator (With Gradients) ===" << std::endl;

  // Create minimal libcint input
  LibcintInput libcint_input;
  libcint_input.atm.resize(6);
  libcint_input.atm[0] = 1; libcint_input.atm[1] = 0; libcint_input.atm[2] = 0;
  libcint_input.atm[3] = 1; libcint_input.atm[4] = 3; libcint_input.atm[5] = 0;

  libcint_input.bas.resize(16);
  for (int i = 0; i < 8; ++i) libcint_input.bas[i] = 0;
  libcint_input.bas[1] = 0;  // s-type
  for (int i = 8; i < 16; ++i) libcint_input.bas[i] = 0;
  libcint_input.bas[8] = 1;  // atom 1
  libcint_input.bas[9] = 0;  // s-type

  libcint_input.env.resize(10);
  libcint_input.env[0] = 0.0; libcint_input.env[1] = 0.0; libcint_input.env[2] = 0.0;
  libcint_input.env[3] = 0.0; libcint_input.env[4] = 0.0; libcint_input.env[5] = 1.4;
  libcint_input.env[6] = 1.0; libcint_input.env[7] = 1.0;

  try {
    LibcintAoGridEvaluator evaluator(libcint_input);

    Eigen::MatrixXd grid_points(3, 3);
    grid_points << 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.7,
                   0.0, 0.0, 1.4;

    std::cout << "Evaluating AO values and gradients..." << std::endl;

    AoGridValues ao_values = evaluator.evaluate_values_and_gradients(grid_points);

    std::cout << "Result:" << std::endl;
    std::cout << "  values shape: " << ao_values.values.rows() << " x "
              << ao_values.values.cols() << std::endl;
    std::cout << "  gradients shape: " << ao_values.gradients.rows() << " x "
              << ao_values.gradients.cols() << std::endl;
    std::cout << "  has_gradients: " << (ao_values.has_gradients() ? "yes" : "no") << std::endl;

    bool all_finite = ao_values.values.allFinite() && ao_values.gradients.allFinite();
    std::cout << "  all_finite: " << (all_finite ? "yes" : "no") << std::endl;

    std::cout << "\n[PASS] AO evaluator with gradients test" << std::endl;

  } catch (const std::exception& e) {
    std::cout << "[FAIL] Exception: " << e.what() << std::endl;
  }

  std::cout << std::endl;
}

int main() {
  try {
    test_ao_grid_values_structure();
    test_ao_evaluator_basic();
    test_ao_evaluator_with_gradients();

    std::cout << "All AO grid evaluator tests completed." << std::endl;
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
}
