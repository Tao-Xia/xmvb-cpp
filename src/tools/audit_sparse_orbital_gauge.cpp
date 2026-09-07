#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/orbital/nonredundant_optimizer_input_adapter.hpp"
#include "vb/orbital/sparse_orbital_gauge_audit.hpp"
#include "vb/orbital/sparse_orbital_parameter_view.hpp"

namespace {

bool parse_bool(const std::string& value) {
  if (value == "true" || value == "1") return true;
  if (value == "false" || value == "0") return false;
  throw std::invalid_argument("invalid Boolean value: " + value);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 && argc != 4) {
    std::cerr
        << "usage: audit_sparse_orbital_gauge <input.xmi> "
        << "[--nonredundant-adapt true|false]\n";
    return EXIT_FAILURE;
  }

  try {
    bool nonredundant_adapt = true;
    if (argc == 4) {
      if (std::string(argv[2]) != "--nonredundant-adapt") {
        throw std::invalid_argument("unknown option: " + std::string(argv[2]));
      }
      nonredundant_adapt = parse_bool(argv[3]);
    }

    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.ao_integral_source = xmvb::vb::AoIntegralSource::Auto;
    load_options.standard_two_electron_mode =
        xmvb::vb::StandardTwoElectronMode::Exact;
    const auto loaded =
        xmvb::vb::load_cpp_vb_input_with_timings(argv[1], load_options);
    const xmvb::vb::CppVbInput input = nonredundant_adapt
        ? xmvb::vb::build_nonredundant_optimizer_input(loaded.input)
        : loaded.input;
    const xmvb::vb::SparseOrbitalParameterView parameter_view(
        input.orbital_preparation_input);
    const auto audit = xmvb::vb::audit_sparse_orbital_gauge(
        input.orbital_preparation_input, parameter_view);

    std::cout << std::setprecision(12);
    std::cout << "input = " << argv[1] << '\n';
    std::cout << "nonredundant_adapt = "
              << (nonredundant_adapt ? "true" : "false") << '\n';
    std::cout << "packed_dimension = " << audit.packed_dimension << '\n';
    std::cout << "gauge_parameter_dimension = "
              << audit.gauge_parameter_dimension << '\n';
    std::cout << "support_constraint_rank = "
              << audit.support_constraint_rank << '\n';
    std::cout << "admissible_gauge_parameter_dimension = "
              << audit.admissible_gauge_parameter_dimension << '\n';
    std::cout << "gauge_rank = " << audit.gauge_rank << '\n';
    std::cout << "quotient_dimension = " << audit.quotient_dimension << '\n';
    std::cout << "physical_jacobian_rank = "
              << audit.physical_jacobian_rank << '\n';
    std::cout << "physical_jacobian_nullity = "
              << audit.physical_jacobian_nullity << '\n';
    std::cout << "unmapped_parameter_count = "
              << audit.unmapped_parameter_count << '\n';
    std::cout << "relative_gauge_annihilation_residual = "
              << audit.relative_gauge_annihilation_residual << '\n';
    std::cout << "maximum_principal_angle_sine = "
              << audit.maximum_gauge_kernel_principal_angle_sine << '\n';
  } catch (const std::exception& exception) {
    std::cerr << "audit_sparse_orbital_gauge failed: "
              << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
