#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/structure_matrix_evaluator.hpp"
#include "vb/matrices/legacy_hamiltonian_overlap_calculator.hpp"

namespace {

struct Options {
  std::string input_path;
};

void print_usage() {
  std::cerr << "usage: compare_cpp_legacy_vb_energy <input.xmi>\n";
}

Options parse_arguments(int argc, char** argv) {
  if (argc != 2) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }
  return {.input_path = argv[1]};
}

double max_abs_difference(
    const std::vector<double>& left,
    const std::vector<double>& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("matrix sizes differ");
  }
  double max_difference = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    max_difference =
        std::max(max_difference, std::abs(left[index] - right[index]));
  }
  return max_difference;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const bool initial_only =
        std::getenv("XMVB_CPP_COMPARE_LEGACY_INITIAL_ONLY") != nullptr;

    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.ao_integral_source = xmvb::vb::AoIntegralSource::LegacyRuntime;
    load_options.orbital_guess_source = xmvb::vb::OrbitalGuessSource::Cpp;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(options.input_path, load_options);

    xmvb::vb::StructureMatrixEvaluator structure_evaluator;
    const auto prepared_active_space =
        structure_evaluator.prepare_active_space(load_result.input);
    const auto cpp_structure_matrices = structure_evaluator.evaluate(
        load_result.input,
        prepared_active_space);

    xmvb::core::GeneralizedEigensolver generalized_eigensolver;
    const auto cpp_eigen_result = generalized_eigensolver.solve(
        cpp_structure_matrices.hamiltonian_matrix,
        cpp_structure_matrices.overlap_matrix,
        load_result.input.structure_data.n_structures);
    const double cpp_valence_structure_energy =
        cpp_eigen_result.eigenvalues.front() + load_result.nuclear_repulsion_energy;
    const double cpp_total_energy =
        prepared_active_space.one_electron_reference_energy +
        cpp_valence_structure_energy;

    xmvb::vb::LegacyHamiltonianOverlapCalculator legacy_calculator;
    const auto legacy_initial_result =
        legacy_calculator.calculate_from_input_file(options.input_path, 1, false);
    const double legacy_initial_valence_structure_energy =
        legacy_initial_result.total_energy - legacy_initial_result.one_electron_energy;
    if (initial_only) {
      std::cout << std::setprecision(12);
      std::cout << "cpp_initial_total_energy = " << cpp_total_energy << '\n';
      std::cout << "legacy_initial_total_energy = "
                << legacy_initial_result.total_energy << '\n';
      std::cout << "legacy_initial_one_electron_reference_energy = "
                << legacy_initial_result.one_electron_energy << '\n';
      std::cout << "legacy_initial_valence_structure_energy = "
                << legacy_initial_valence_structure_energy << '\n';
      return 0;
    }
    const auto legacy_final_result =
        legacy_calculator.calculate_from_input_file(options.input_path, 1, true);
    const double legacy_final_valence_structure_energy =
        legacy_final_result.total_energy - legacy_final_result.one_electron_energy;

    std::cout << std::setprecision(12);
    std::cout << "cpp_n_structures = "
              << load_result.input.structure_data.n_structures << '\n';
    std::cout << "legacy_initial_n_structures = "
              << legacy_initial_result.n_structures << '\n';
    std::cout << "legacy_final_n_structures = "
              << legacy_final_result.n_structures << '\n';
    std::cout << "nuclear_repulsion_energy = "
              << load_result.nuclear_repulsion_energy << '\n';
    std::cout << "cpp_one_electron_reference_energy = "
              << prepared_active_space.one_electron_reference_energy << '\n';
    std::cout << "legacy_initial_one_electron_reference_energy = "
              << legacy_initial_result.one_electron_energy << '\n';
    std::cout << "legacy_final_one_electron_reference_energy = "
              << legacy_final_result.one_electron_energy << '\n';
    std::cout << "initial_one_electron_reference_energy_diff = "
              << (prepared_active_space.one_electron_reference_energy -
                  legacy_initial_result.one_electron_energy)
              << '\n';
    std::cout << "cpp_valence_structure_energy = "
              << cpp_valence_structure_energy << '\n';
    std::cout << "legacy_initial_valence_structure_energy = "
              << legacy_initial_valence_structure_energy << '\n';
    std::cout << "legacy_final_valence_structure_energy = "
              << legacy_final_valence_structure_energy << '\n';
    std::cout << "initial_valence_structure_energy_diff = "
              << (cpp_valence_structure_energy - legacy_initial_valence_structure_energy)
              << '\n';
    std::cout << "cpp_initial_total_energy = " << cpp_total_energy << '\n';
    std::cout << "legacy_initial_total_energy = "
              << legacy_initial_result.total_energy << '\n';
    std::cout << "legacy_final_total_energy = "
              << legacy_final_result.total_energy << '\n';
    std::cout << "initial_total_energy_diff = "
              << (cpp_total_energy - legacy_initial_result.total_energy) << '\n';
    std::cout << "cpp_vs_legacy_final_total_energy_diff = "
              << (cpp_total_energy - legacy_final_result.total_energy) << '\n';

    if (cpp_structure_matrices.hamiltonian_matrix.size() ==
            legacy_initial_result.hamiltonian_matrix.size() &&
        cpp_structure_matrices.overlap_matrix.size() ==
            legacy_initial_result.overlap_matrix.size()) {
      std::cout << "initial_hamiltonian_max_abs_diff = "
                << max_abs_difference(
                       cpp_structure_matrices.hamiltonian_matrix,
                       legacy_initial_result.hamiltonian_matrix)
                << '\n';
      std::cout << "initial_overlap_max_abs_diff = "
                << max_abs_difference(
                       cpp_structure_matrices.overlap_matrix,
                       legacy_initial_result.overlap_matrix)
                << '\n';
    } else {
      std::cout << "initial_hamiltonian_max_abs_diff = size_mismatch\n";
      std::cout << "initial_overlap_max_abs_diff = size_mismatch\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
