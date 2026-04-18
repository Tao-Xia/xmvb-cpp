#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/full_structure_builder.hpp"
#include "vb/matrices/prepared_active_space_context.hpp"
#include "vb/vb_model_flags.hpp"

namespace {

struct Options {
  std::string input_path;
  xmvb::vb::StandardTwoElectronMode standard_two_electron_mode =
      xmvb::vb::StandardTwoElectronMode::Exact;
  int matrix_print_limit = 5;
};

enum class SignMode {
  CurrentAlphaBeta,
  PairSwapAlphaBeta,
  PairSwapAlphaOnly,
  PairSwapBetaOnly,
  PairSwapOnly,
  KeepOrderUnit,
  KeepOrderPairSwap,
};

struct DeterminantKey {
  std::vector<int> alpha_orbitals;
  std::vector<int> beta_orbitals;

  bool operator==(const DeterminantKey& other) const {
    return alpha_orbitals == other.alpha_orbitals &&
        beta_orbitals == other.beta_orbitals;
  }
};

struct DeterminantKeyHasher {
  std::size_t operator()(const DeterminantKey& determinant_key) const {
    std::size_t hash_value = 0;
    for (const int orbital_index : determinant_key.alpha_orbitals) {
      hash_value = hash_value * 1315423911u + xmvb::to_size(orbital_index + 257);
    }
    hash_value = hash_value * 2654435761u + 17u;
    for (const int orbital_index : determinant_key.beta_orbitals) {
      hash_value = hash_value * 1315423911u + xmvb::to_size(orbital_index + 257);
    }
    return hash_value;
  }
};

struct PairedDeterminantAssignment {
  std::vector<int> occupied_orbitals;
  int pair_swap_sign = 1;
};

struct MatrixOffDiagonalStats {
  int nnz = 0;
  double max_abs = 0.0;
};

void print_usage() {
  std::cerr << "usage: check_structure_expansion_signs <input.xmi>"
               " [--standard-two-electron-mode auto|exact|ri]"
               " [--matrix-print-limit N]\n";
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
      const std::string mode = argv[++argument_index];
      if (mode == "auto") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::Auto;
      } else if (mode == "exact") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::Exact;
      } else if (mode == "ri") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
      } else {
        throw std::invalid_argument(
            "unknown --standard-two-electron-mode value: " + mode);
      }
      continue;
    }
    if (argument == "--matrix-print-limit") {
      if (argument_index + 1 >= argc) {
        throw std::invalid_argument("--matrix-print-limit requires a value");
      }
      options.matrix_print_limit = std::stoi(argv[++argument_index]);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument);
  }

  if (options.matrix_print_limit < 1) {
    throw std::invalid_argument("--matrix-print-limit must be positive");
  }
  return options;
}

const char* sign_mode_name(SignMode sign_mode) {
  switch (sign_mode) {
    case SignMode::CurrentAlphaBeta:
      return "current_alpha_beta";
    case SignMode::PairSwapAlphaBeta:
      return "pair_swap_alpha_beta";
    case SignMode::PairSwapAlphaOnly:
      return "pair_swap_alpha_only";
    case SignMode::PairSwapBetaOnly:
      return "pair_swap_beta_only";
    case SignMode::PairSwapOnly:
      return "pair_swap_only";
    case SignMode::KeepOrderUnit:
      return "keep_order_unit";
    case SignMode::KeepOrderPairSwap:
      return "keep_order_pair_swap";
  }
  return "unknown";
}

int canonicalize_spin_string(std::vector<int>* occupied_orbitals) {
  if (occupied_orbitals == nullptr) {
    throw std::invalid_argument("occupied_orbitals must not be null");
  }
  int permutation_sign = 1;
  for (std::size_t left_index = 0;
       left_index + 1 < occupied_orbitals->size();
       ++left_index) {
    for (std::size_t right_index = left_index + 1;
         right_index < occupied_orbitals->size();
         ++right_index) {
      if ((*occupied_orbitals)[left_index] > (*occupied_orbitals)[right_index]) {
        std::swap(
            (*occupied_orbitals)[left_index],
            (*occupied_orbitals)[right_index]);
        permutation_sign = -permutation_sign;
      }
    }
  }
  return permutation_sign;
}

void enumerate_paired_determinants_recursive(
    const std::vector<int>& paired_active_orbitals,
    int n_active_beta_electrons,
    int pair_index,
    int pair_swap_sign,
    std::vector<int>* occupied_orbitals,
    std::vector<PairedDeterminantAssignment>* assignments) {
  if (occupied_orbitals == nullptr || assignments == nullptr) {
    throw std::invalid_argument("enumeration buffers must not be null");
  }
  if (pair_index == n_active_beta_electrons) {
    assignments->push_back(
        PairedDeterminantAssignment{*occupied_orbitals, pair_swap_sign});
    return;
  }

  (*occupied_orbitals)[xmvb::to_size(pair_index)] =
      paired_active_orbitals[xmvb::to_size(2 * pair_index)];
  (*occupied_orbitals)[xmvb::to_size(pair_index + n_active_beta_electrons)] =
      paired_active_orbitals[xmvb::to_size(2 * pair_index + 1)];
  enumerate_paired_determinants_recursive(
      paired_active_orbitals,
      n_active_beta_electrons,
      pair_index + 1,
      pair_swap_sign,
      occupied_orbitals,
      assignments);

  if (paired_active_orbitals[xmvb::to_size(2 * pair_index)] ==
      paired_active_orbitals[xmvb::to_size(2 * pair_index + 1)]) {
    return;
  }

  (*occupied_orbitals)[xmvb::to_size(pair_index)] =
      paired_active_orbitals[xmvb::to_size(2 * pair_index + 1)];
  (*occupied_orbitals)[xmvb::to_size(pair_index + n_active_beta_electrons)] =
      paired_active_orbitals[xmvb::to_size(2 * pair_index)];
  enumerate_paired_determinants_recursive(
      paired_active_orbitals,
      n_active_beta_electrons,
      pair_index + 1,
      -pair_swap_sign,
      occupied_orbitals,
      assignments);
}

std::vector<PairedDeterminantAssignment> expand_paired_active_orbitals(
    const std::vector<int>& paired_active_orbitals,
    int n_active_beta_electrons,
    int wavefunction_type) {
  if (static_cast<int>(paired_active_orbitals.size()) != 2 * n_active_beta_electrons) {
    throw std::invalid_argument("paired active orbital count mismatch");
  }
  if (n_active_beta_electrons == 0) {
    return {PairedDeterminantAssignment{{}, 1}};
  }
  if (wavefunction_type ==
      xmvb::vb::wavefunction_type_code(xmvb::vb::WavefunctionType::Determinant)) {
    return {PairedDeterminantAssignment{paired_active_orbitals, 1}};
  }

  std::vector<PairedDeterminantAssignment> assignments;
  std::vector<int> occupied_orbitals(xmvb::to_size(2 * n_active_beta_electrons), 0);
  enumerate_paired_determinants_recursive(
      paired_active_orbitals,
      n_active_beta_electrons,
      0,
      1,
      &occupied_orbitals,
      &assignments);
  return assignments;
}

double expansion_coefficient(
    SignMode sign_mode,
    int pair_swap_sign,
    int alpha_sign,
    int beta_sign) {
  switch (sign_mode) {
    case SignMode::CurrentAlphaBeta:
      return static_cast<double>(alpha_sign * beta_sign);
    case SignMode::PairSwapAlphaBeta:
      return static_cast<double>(pair_swap_sign * alpha_sign * beta_sign);
    case SignMode::PairSwapAlphaOnly:
      return static_cast<double>(pair_swap_sign * alpha_sign);
    case SignMode::PairSwapBetaOnly:
      return static_cast<double>(pair_swap_sign * beta_sign);
    case SignMode::PairSwapOnly:
      return static_cast<double>(pair_swap_sign);
    case SignMode::KeepOrderUnit:
      return 1.0;
    case SignMode::KeepOrderPairSwap:
      return static_cast<double>(pair_swap_sign);
  }
  throw std::invalid_argument("unknown sign mode");
}

bool should_canonicalize_spin_strings(SignMode sign_mode) {
  switch (sign_mode) {
    case SignMode::CurrentAlphaBeta:
    case SignMode::PairSwapAlphaBeta:
    case SignMode::PairSwapAlphaOnly:
    case SignMode::PairSwapBetaOnly:
    case SignMode::PairSwapOnly:
      return true;
    case SignMode::KeepOrderUnit:
    case SignMode::KeepOrderPairSwap:
      return false;
  }
  throw std::invalid_argument("unknown sign mode");
}

xmvb::vb::FullDeterminantStructureData expand_with_sign_mode(
    const xmvb::vb::RawStructureData& raw_structure_data,
    SignMode sign_mode) {
  if (raw_structure_data.n_total_electrons <= 0 || raw_structure_data.n_active_electrons <= 0) {
    throw std::invalid_argument("raw structure dimensions must be positive");
  }

  const int n_inactive_doubly_occupied_orbitals =
      (raw_structure_data.n_total_electrons - raw_structure_data.n_active_electrons) / 2;
  const int n_spin_open_shell_electrons = raw_structure_data.spin_multiplicity - 1;
  const int n_active_beta_electrons =
      (raw_structure_data.n_active_electrons - n_spin_open_shell_electrons) / 2;

  xmvb::vb::FullDeterminantStructureData result;
  result.n_structures = raw_structure_data.n_structures;

  std::unordered_map<DeterminantKey, int, DeterminantKeyHasher> determinant_to_index;
  for (int structure_index = 0;
       structure_index < raw_structure_data.n_structures;
       ++structure_index) {
    const int* raw_structure = raw_structure_data.structure_orbitals_data(structure_index);

    std::vector<int> active_structure_orbitals(
        xmvb::to_size(raw_structure_data.n_active_electrons),
        0);
    for (int active_index = 0;
         active_index < raw_structure_data.n_active_electrons;
         ++active_index) {
      active_structure_orbitals[xmvb::to_size(active_index)] =
          raw_structure[xmvb::to_size(
              active_index + 2 * n_inactive_doubly_occupied_orbitals)] -
          n_inactive_doubly_occupied_orbitals - 1;
    }

    const std::vector<int> paired_active_orbitals(
        active_structure_orbitals.begin(),
        active_structure_orbitals.begin() + 2 * n_active_beta_electrons);
    const auto paired_assignments = expand_paired_active_orbitals(
        paired_active_orbitals,
        n_active_beta_electrons,
        raw_structure_data.wavefunction_type);
    for (const auto& assignment : paired_assignments) {
      std::vector<int> alpha_orbitals(
          assignment.occupied_orbitals.begin(),
          assignment.occupied_orbitals.begin() + n_active_beta_electrons);
      alpha_orbitals.insert(
          alpha_orbitals.end(),
          active_structure_orbitals.begin() + 2 * n_active_beta_electrons,
          active_structure_orbitals.end());
      std::vector<int> beta_orbitals(
          assignment.occupied_orbitals.begin() + n_active_beta_electrons,
          assignment.occupied_orbitals.end());

      int alpha_sign = 1;
      int beta_sign = 1;
      if (should_canonicalize_spin_strings(sign_mode)) {
        alpha_sign = canonicalize_spin_string(&alpha_orbitals);
        beta_sign = canonicalize_spin_string(&beta_orbitals);
      }
      const double coefficient =
          expansion_coefficient(
              sign_mode,
              assignment.pair_swap_sign,
              alpha_sign,
              beta_sign);

      DeterminantKey determinant_key{alpha_orbitals, beta_orbitals};
      const auto [determinant_iterator, inserted] =
          determinant_to_index.emplace(
              determinant_key,
              static_cast<int>(result.alpha_det.size()));
      if (inserted) {
        result.alpha_det.push_back(alpha_orbitals);
        result.beta_det.push_back(beta_orbitals);
        result.determinant_to_structure_terms.push_back({});
      }
      result.determinant_to_structure_terms[xmvb::to_size(determinant_iterator->second)]
          .push_back(
              xmvb::vb::StructureExpansionTerm{
                  structure_index,
                  coefficient});
    }
  }

  return result;
}

MatrixOffDiagonalStats matrix_off_diagonal_stats(
    const std::vector<double>& matrix,
    int dimension) {
  if (dimension <= 0 || static_cast<int>(matrix.size()) != dimension * dimension) {
    throw std::invalid_argument("matrix dimension mismatch");
  }
  MatrixOffDiagonalStats stats;
  for (int column = 0; column < dimension; ++column) {
    for (int row = 0; row < dimension; ++row) {
      if (row == column) {
        continue;
      }
      const double value = matrix[xmvb::col_major_index(row, column, dimension)];
      const double abs_value = std::abs(value);
      stats.max_abs = std::max(stats.max_abs, abs_value);
      if (abs_value > 1.0e-12) {
        ++stats.nnz;
      }
    }
  }
  return stats;
}

void print_matrix_block(
    const std::vector<double>& matrix,
    int dimension,
    int print_limit,
    const char* label) {
  const int block_size = std::min(dimension, print_limit);
  std::cout << label << "_top_left_" << block_size << "x" << block_size << ":\n";
  for (int row = 0; row < block_size; ++row) {
    for (int column = 0; column < block_size; ++column) {
      std::cout << std::setw(14)
                << matrix[xmvb::col_major_index(row, column, dimension)];
    }
    std::cout << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);

    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.standard_two_electron_mode = options.standard_two_electron_mode;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(options.input_path, load_options);

    xmvb::vb::ActiveSpaceOrbitalPreparer orbital_preparer;
    xmvb::vb::AoEffectiveOneElectronBuilder ao_effective_one_electron_builder;
    xmvb::vb::ActiveSpaceOneElectronBuilder active_space_one_electron_builder;
    xmvb::vb::ActiveSpaceTwoElectronBuilder active_space_two_electron_builder;
    const auto prepared_active_space = xmvb::vb::prepare_active_space_context(
        load_result.input,
        orbital_preparer,
        ao_effective_one_electron_builder,
        active_space_one_electron_builder,
        active_space_two_electron_builder);

    xmvb::core::GeneralizedEigensolver generalized_eigensolver;
    xmvb::vb::FullDeterminantStructureHamiltonianOverlapBuilder structure_builder(
        xmvb::vb::VBSCFAlgorithm::Original);

    std::cout << std::setprecision(12);
    std::cout << "input_path = " << options.input_path << '\n';
    std::cout << "n_structures = " << load_result.raw_structure_data.n_structures << '\n';
    std::cout << "n_total_electrons = " << load_result.raw_structure_data.n_total_electrons << '\n';
    std::cout << "n_active_electrons = " << load_result.raw_structure_data.n_active_electrons << '\n';
    std::cout << "spin_multiplicity = " << load_result.raw_structure_data.spin_multiplicity << '\n';
    std::cout << "wavefunction_type = " << load_result.raw_structure_data.wavefunction_type << '\n';
    std::cout << "vb_function_type = " << load_result.raw_structure_data.vb_function_type << '\n';
    std::cout << "one_electron_reference_energy = "
              << prepared_active_space.one_electron_reference_energy << '\n';
    std::cout << "nuclear_repulsion_energy = "
              << load_result.nuclear_repulsion_energy << '\n';

    const std::vector<SignMode> sign_modes = {
        SignMode::CurrentAlphaBeta,
        SignMode::PairSwapAlphaBeta,
        SignMode::PairSwapAlphaOnly,
        SignMode::PairSwapBetaOnly,
        SignMode::PairSwapOnly,
        SignMode::KeepOrderUnit,
        SignMode::KeepOrderPairSwap,
    };
    for (const SignMode sign_mode : sign_modes) {
      const auto structure_data =
          expand_with_sign_mode(load_result.raw_structure_data, sign_mode);
      auto structure_matrices = structure_builder.build(
          structure_data.alpha_det,
          structure_data.beta_det,
          structure_data.determinant_to_structure_terms,
          prepared_active_space.orbital_result.active_orbital_overlap_matrix,
          prepared_active_space.active_space_one_electron_result.h1e_act,
          load_result.input.orbital_preparation_input.n_active_orbitals,
          prepared_active_space.active_space_two_electron_result,
          structure_data.n_structures);
      const auto eigen_result = generalized_eigensolver.solve(
          structure_matrices.hamiltonian_matrix,
          structure_matrices.overlap_matrix,
          structure_data.n_structures);
      const double valence_structure_energy = eigen_result.eigenvalues.front();
      const double total_energy =
          prepared_active_space.one_electron_reference_energy +
          load_result.nuclear_repulsion_energy +
          valence_structure_energy;
      const MatrixOffDiagonalStats overlap_stats =
          matrix_off_diagonal_stats(
              structure_matrices.overlap_matrix,
              structure_data.n_structures);
      const MatrixOffDiagonalStats hamiltonian_stats =
          matrix_off_diagonal_stats(
              structure_matrices.hamiltonian_matrix,
              structure_data.n_structures);

      std::cout << '\n';
      std::cout << "sign_mode = " << sign_mode_name(sign_mode) << '\n';
      std::cout << "expanded_determinant_count = " << structure_data.alpha_det.size() << '\n';
      std::cout << "overlap_offdiag_nnz = " << overlap_stats.nnz << '\n';
      std::cout << "overlap_max_abs_offdiag = " << overlap_stats.max_abs << '\n';
      std::cout << "hamiltonian_offdiag_nnz = " << hamiltonian_stats.nnz << '\n';
      std::cout << "hamiltonian_max_abs_offdiag = " << hamiltonian_stats.max_abs << '\n';
      std::cout << "valence_structure_eigenvalue = " << valence_structure_energy << '\n';
      std::cout << "total_energy = " << total_energy << '\n';
      print_matrix_block(
          structure_matrices.overlap_matrix,
          structure_data.n_structures,
          options.matrix_print_limit,
          "overlap");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
