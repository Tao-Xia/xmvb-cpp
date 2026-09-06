#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/approx/approx_vbscf_evaluator.hpp"
#include "vb/approx/approx_vbscf_resonance.hpp"
#include "vb/matrices/structure_matrix_evaluator.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace {

struct Options {
  std::string input_path;
  xmvb::vb::StandardTwoElectronMode standard_two_electron_mode =
      xmvb::vb::StandardTwoElectronMode::Auto;
};

void print_usage() {
  std::cerr << "usage: analyze_approx_vbscf_resonance <input.xmi> "
               "[--standard-two-electron-mode auto|exact|ri]\n";
}

xmvb::vb::StandardTwoElectronMode parse_standard_two_electron_mode(
    const std::string& value) {
  if (value == "auto") {
    return xmvb::vb::StandardTwoElectronMode::Auto;
  }
  if (value == "exact") {
    return xmvb::vb::StandardTwoElectronMode::Exact;
  }
  if (value == "ri") {
    return xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
  }
  throw std::invalid_argument(
      "invalid standard two-electron mode: " + value);
}

Options parse_arguments(int argc, char** argv) {
  if (argc < 2) {
    print_usage();
    throw std::invalid_argument("missing input path");
  }

  Options options;
  options.input_path = argv[1];
  for (int argument_index = 2; argument_index < argc; ++argument_index) {
    const std::string argument = argv[argument_index];
    if (argument == "--standard-two-electron-mode") {
      if (argument_index + 1 >= argc) {
        throw std::invalid_argument(
            "--standard-two-electron-mode requires a value");
      }
      options.standard_two_electron_mode =
          parse_standard_two_electron_mode(argv[++argument_index]);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument);
  }
  return options;
}

double occupied_active_pair_count(
    int n_active_electrons,
    int spin_multiplicity) {
  const int n_open_shell_electrons = spin_multiplicity - 1;
  return 0.5 * static_cast<double>(
      n_active_electrons - n_open_shell_electrons);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.standard_two_electron_mode =
        options.standard_two_electron_mode;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(
            options.input_path,
            load_options);

    xmvb::vb::StructureMatrixEvaluator matrix_evaluator(
        xmvb::vb::VBSCFAlgorithm::Original);
    const auto prepared_active_space =
        matrix_evaluator.prepare_active_space(load_result.input);
    const auto structure_matrices =
        matrix_evaluator.evaluate(
            load_result.input,
            prepared_active_space);

    xmvb::core::GeneralizedEigensolver eigensolver;
    const auto eigen_result =
        eigensolver.solve(
            structure_matrices.hamiltonian_matrix,
            structure_matrices.overlap_matrix,
            structure_matrices.n_structures);

    const int n_active_orbitals =
        load_result.input.orbital_preparation_input.n_active_orbitals;
    xmvb::vb::ApproxVbScfModelInput model_input;
    model_input.n_active_orbitals = n_active_orbitals;
    model_input.one_electron_reference_energy =
        prepared_active_space.one_electron_reference_energy;
    model_input.nuclear_repulsion_energy =
        load_result.nuclear_repulsion_energy;
    model_input.active_one_electron_integrals =
        &prepared_active_space.active_space_one_electron_result.h1e_act;
    model_input.active_orbital_overlap_matrix =
        &prepared_active_space.orbital_result.active_orbital_overlap_matrix;
    model_input.active_two_electron_result =
        &prepared_active_space.active_space_two_electron_result;
    const auto model =
        xmvb::vb::build_approx_vbscf_model(model_input);

    xmvb::vb::ApproxVbScfStateMappingInput mapping_input;
    mapping_input.n_active_orbitals = n_active_orbitals;
    mapping_input.raw_structure_data = &load_result.raw_structure_data;
    mapping_input.structure_overlap_matrix =
        &structure_matrices.overlap_matrix;
    mapping_input.eigenvector_matrix =
        &eigen_result.eigenvector_matrix;
    const auto exact_state =
        xmvb::vb::map_selected_state_to_pair_state(mapping_input);
    const auto approx_result =
        xmvb::vb::evaluate_approx_vbscf(
            model,
            exact_state.pair_occupations,
            exact_state.pair_pair_occupations);

    xmvb::vb::ApproxVbScfResonanceDecompositionInput resonance_input;
    resonance_input.n_active_orbitals = n_active_orbitals;
    resonance_input.raw_structure_data = &load_result.raw_structure_data;
    resonance_input.structure_hamiltonian_matrix =
        &structure_matrices.hamiltonian_matrix;
    resonance_input.structure_overlap_matrix =
        &structure_matrices.overlap_matrix;
    resonance_input.eigenvector_matrix =
        &eigen_result.eigenvector_matrix;
    const auto resonance =
        xmvb::vb::decompose_approx_vbscf_resonance(resonance_input);

    const double exact_active_energy = eigen_result.eigenvalues.front();
    const double cluster_missing_active_energy =
        exact_active_energy - approx_result.cluster_quotient_active_energy;
    const double exact_offdiagonal_energy =
        resonance.same_pattern_energy + resonance.resonance_energy;
    const double cluster_offdiagonal_like_energy =
        approx_result.cluster_quotient_active_energy -
        resonance.diagonal_energy;
    const double offdiagonal_residual_energy =
        exact_offdiagonal_energy - cluster_offdiagonal_like_energy;
    double cluster_offdiagonal_capture_fraction = 0.0;
    if (std::abs(exact_offdiagonal_energy) > 1.0e-14) {
      cluster_offdiagonal_capture_fraction =
          cluster_offdiagonal_like_energy / exact_offdiagonal_energy;
    }
    const double occupied_pair_count =
        occupied_active_pair_count(
            load_result.input.orbital_preparation_input.n_active_electrons,
            load_result.raw_structure_data.spin_multiplicity);

    std::cout << std::setprecision(12);
    std::cout << "input = " << options.input_path << '\n';
    std::cout << "n_structures = " << resonance.n_structures << '\n';
    std::cout << "n_active_orbitals = " << n_active_orbitals << '\n';
    std::cout << "occupied_pair_count = "
              << occupied_pair_count << '\n';
    std::cout << "denominator = "
              << resonance.denominator << '\n';
    std::cout << "exact_active_energy = "
              << exact_active_energy << '\n';
    std::cout << "cluster_quotient_active_energy = "
              << approx_result.cluster_quotient_active_energy << '\n';
    std::cout << "cluster_missing_active_energy = "
              << cluster_missing_active_energy << '\n';
    std::cout << "diagonal_energy = "
              << resonance.diagonal_energy << '\n';
    std::cout << "same_pattern_energy = "
              << resonance.same_pattern_energy << '\n';
    std::cout << "resonance_energy = "
              << resonance.resonance_energy << '\n';
    std::cout << "diagonal_plus_same_pattern_energy = "
              << resonance.diagonal_energy +
                     resonance.same_pattern_energy << '\n';
    std::cout << "exact_offdiagonal_energy = "
              << exact_offdiagonal_energy << '\n';
    std::cout << "cluster_offdiagonal_like_energy = "
              << cluster_offdiagonal_like_energy << '\n';
    std::cout << "offdiagonal_residual_energy = "
              << offdiagonal_residual_energy << '\n';
    std::cout << "cluster_offdiagonal_capture_fraction = "
              << cluster_offdiagonal_capture_fraction << '\n';
    std::cout << "diagonal_terms = "
              << resonance.diagonal_terms << '\n';
    std::cout << "same_pattern_terms = "
              << resonance.same_pattern_terms << '\n';
    std::cout << "resonance_terms = "
              << resonance.resonance_terms << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
