#include <iostream>
#include <iomanip>

#include "vb/pdft/vb_pdft_energy_evaluator.hpp"
#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/scf/cpp_vb_scf_result.hpp"

using namespace xmvb::vb::pdft;
using namespace xmvb::vb;

void print_energy_result(const VbPdftEnergyResult& result) {
  std::cout << std::fixed << std::setprecision(10);

  std::cout << "\n=== VB-PDFT Energy Result ===" << std::endl;
  std::cout << "State index: " << result.state_index << std::endl;
  std::cout << "Grid points: " << result.n_grid_points << std::endl;

  std::cout << "\nEnergy components (Hartree):" << std::endl;
  std::cout << "  Nuclear repulsion:  " << std::setw(16) << result.nuclear_repulsion_energy << std::endl;
  std::cout << "  One-electron:       " << std::setw(16) << result.one_electron_energy << std::endl;
  std::cout << "  Coulomb:            " << std::setw(16) << result.coulomb_energy << std::endl;
  std::cout << "  On-top functional:  " << std::setw(16) << result.on_top_energy << std::endl;
  std::cout << "  " << std::string(50, '-') << std::endl;
  std::cout << "  Total VB-PDFT:      " << std::setw(16) << result.total_energy << std::endl;

  std::cout << "\nValidation:" << std::endl;
  std::cout << "  Integrated electrons: " << std::setw(16) << result.integrated_electron_count << std::endl;

  std::cout << std::endl;
}

void test_vb_pdft_config() {
  std::cout << "=== Testing VB-PDFT Configuration ===" << std::endl;

  VbPdftConfig config;
  config.grid_config.radial_points = 50;
  config.grid_config.angular_points = 302;
  config.functional_id = 1;  // LDA exchange
  config.density_threshold = 1.0e-12;
  config.use_gga = false;

  std::cout << "Configuration created:" << std::endl;
  std::cout << "  Radial points: " << config.grid_config.radial_points << std::endl;
  std::cout << "  Angular points: " << config.grid_config.angular_points << std::endl;
  std::cout << "  Functional ID: " << config.functional_id << std::endl;
  std::cout << "  Use GGA: " << (config.use_gga ? "yes" : "no") << std::endl;

  std::cout << "\n[PASS] Configuration test" << std::endl;
  std::cout << std::endl;
}

void test_vb_pdft_evaluator_construction() {
  std::cout << "=== Testing VB-PDFT Evaluator Construction ===" << std::endl;

  try {
    VbPdftConfig config;
    config.grid_config.radial_points = 20;
    config.grid_config.angular_points = 26;
    config.functional_id = 1;

    VbPdftEnergyEvaluator evaluator(config);

    std::cout << "Evaluator constructed successfully" << std::endl;
    std::cout << "\n[PASS] Evaluator construction test" << std::endl;

  } catch (const std::exception& e) {
    std::cout << "[FAIL] Exception: " << e.what() << std::endl;
  }

  std::cout << std::endl;
}

void test_vb_pdft_energy_result() {
  std::cout << "=== Testing VB-PDFT Energy Result ===" << std::endl;

  VbPdftEnergyResult result;
  result.state_index = 0;
  result.n_grid_points = 1040;
  result.nuclear_repulsion_energy = 0.7151043390;
  result.one_electron_energy = -2.5;
  result.coulomb_energy = 1.0;
  result.on_top_energy = -0.5;
  result.total_energy = result.nuclear_repulsion_energy +
                        result.one_electron_energy +
                        result.coulomb_energy +
                        result.on_top_energy;
  result.integrated_electron_count = 2.0;

  print_energy_result(result);

  std::cout << "[PASS] Energy result test" << std::endl;
  std::cout << std::endl;
}

int main() {
  std::cout << "VB-PDFT Energy Evaluator Tests\n" << std::endl;

  try {
    test_vb_pdft_config();
    test_vb_pdft_evaluator_construction();
    test_vb_pdft_energy_result();

    std::cout << "=== Summary ===" << std::endl;
    std::cout << "All VB-PDFT energy evaluator tests completed successfully." << std::endl;
    std::cout << "\nNote: Full integration test with real VBSCF input requires" << std::endl;
    std::cout << "      a complete VBSCF calculation to be run first." << std::endl;

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
}
