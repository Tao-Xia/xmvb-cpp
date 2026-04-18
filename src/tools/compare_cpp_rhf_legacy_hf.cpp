#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "runtime/cpp_closed_shell_fock_builder.hpp"
#include "runtime/cpp_restricted_hartree_fock.hpp"
#include "runtime/cpp_vb_input_loader.hpp"
#include "runtime_c/cpp_runtime_extractor.h"

namespace {

struct Options {
  std::string input_path;
};

void print_usage() {
  std::cerr << "usage: compare_cpp_rhf_legacy_hf <input.xmi>\n";
}

Options parse_arguments(int argc, char** argv) {
  if (argc != 2) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }
  return {.input_path = argv[1]};
}

double max_abs_difference(
    const double* left,
    const std::vector<double>& right,
    std::size_t count) {
  double max_difference = 0.0;
  for (std::size_t index = 0; index < count; ++index) {
    max_difference =
        std::max(max_difference, std::abs(left[index] - right[index]));
  }
  return max_difference;
}

double max_abs_value(const double* values, std::size_t count) {
  double max_value = 0.0;
  for (std::size_t index = 0; index < count; ++index) {
    max_value = std::max(max_value, std::abs(values[index]));
  }
  return max_value;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);

    CppRuntimeExtractionOptions extraction_options{};
    init_cpp_runtime_extraction_options(&extraction_options);
    CppRuntimeSnapshot snapshot{};
    init_cpp_runtime_snapshot(&snapshot);
    CppRuntimeExtractionTimings timings{};
    char error_message[1024] = {0};
    const int status = extract_cpp_runtime_snapshot_with_options(
        options.input_path.c_str(),
        &extraction_options,
        &snapshot,
        &timings,
        error_message,
        sizeof(error_message));
    if (status != 0) {
      throw std::runtime_error(
          error_message[0] != '\0' ? error_message : "legacy runtime extraction failed");
    }

    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.ao_integral_source = xmvb::vb::AoIntegralSource::LibcintMaterializedCpp;
    load_options.orbital_guess_source = xmvb::vb::OrbitalGuessSource::LegacyRuntime;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(options.input_path, load_options);
    const std::size_t matrix_size =
        xmvb::to_size(snapshot.n_basis_functions) * snapshot.n_basis_functions;

    xmvb::vb::CppRestrictedHartreeFockSolver solver;
    const auto rhf_result = solver.solve(
        load_result.input.orbital_preparation_input.n_total_electrons,
        load_result.input.orbital_preparation_input.active_orbital_overlap_matrix.vector(),
        load_result.input.ao_integral_input);
    xmvb::vb::CppClosedShellFockBuilder fock_builder;
    const auto cpp_fock_from_legacy_density = fock_builder.build(
        std::vector<double>(
            snapshot.hf_density_matrix,
            snapshot.hf_density_matrix + matrix_size),
        load_result.input.ao_integral_input);
    std::vector<double> doubled_legacy_density(
        snapshot.hf_density_matrix,
        snapshot.hf_density_matrix + matrix_size);
    for (double& value : doubled_legacy_density) {
      value *= 2.0;
    }
    const auto cpp_fock_from_doubled_legacy_density = fock_builder.build(
        doubled_legacy_density,
        load_result.input.ao_integral_input);
    const double density_max_abs_diff = max_abs_difference(
        snapshot.hf_density_matrix,
        rhf_result.density_projector,
        matrix_size);
    const double fock_max_abs_diff = max_abs_difference(
        snapshot.hf_fock_matrix,
        rhf_result.fock_matrix,
        matrix_size);
    const double builder_on_legacy_density_fock_max_abs_diff = max_abs_difference(
        snapshot.hf_fock_matrix,
        cpp_fock_from_legacy_density,
        matrix_size);
    const double builder_on_doubled_legacy_density_fock_max_abs_diff = max_abs_difference(
        snapshot.hf_fock_matrix,
        cpp_fock_from_doubled_legacy_density,
        matrix_size);

    std::cout << std::setprecision(12);
    std::cout << "n_basis_functions = " << snapshot.n_basis_functions << '\n';
    std::cout << "legacy_hf_fock_max_abs_value = "
              << max_abs_value(snapshot.hf_fock_matrix, matrix_size) << '\n';
    std::cout << "legacy_hf_density_max_abs_value = "
              << max_abs_value(snapshot.hf_density_matrix, matrix_size) << '\n';
    std::cout << "cpp_rhf_converged = " << (rhf_result.converged ? 1 : 0) << '\n';
    std::cout << "cpp_rhf_iterations = " << rhf_result.iterations << '\n';
    std::cout << "cpp_vs_legacy_density_max_abs_diff = "
              << density_max_abs_diff << '\n';
    std::cout << "cpp_vs_legacy_fock_max_abs_diff = "
              << fock_max_abs_diff << '\n';
    std::cout << "cpp_builder_on_legacy_density_fock_max_abs_diff = "
              << builder_on_legacy_density_fock_max_abs_diff << '\n';
    std::cout << "cpp_builder_on_doubled_legacy_density_fock_max_abs_diff = "
              << builder_on_doubled_legacy_density_fock_max_abs_diff << '\n';

    free_cpp_runtime_snapshot(&snapshot);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
