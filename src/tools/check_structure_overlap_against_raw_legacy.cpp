#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/legacy_structure_overlap.hpp"
#include "vb/matrices/structure_matrix_evaluator.hpp"

namespace {

using Matrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

struct Options {
  std::string input_path;
  xmvb::vb::StandardTwoElectronMode standard_two_electron_mode =
      xmvb::vb::StandardTwoElectronMode::Exact;
};

void print_usage() {
  std::cerr
      << "usage: check_structure_overlap_against_raw_legacy <input.xmi>"
      << " [--standard-two-electron-mode exact|auto|ri]\n";
}

Options parse_arguments(int argc, char** argv) {
  if (argc < 2 || ((argc - 2) % 2 != 0)) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }

  Options options;
  options.input_path = argv[1];
  for (int argument_index = 2; argument_index < argc; argument_index += 2) {
    const std::string name = argv[argument_index];
    const std::string value = argv[argument_index + 1];
    if (name == "--standard-two-electron-mode") {
      if (value == "auto") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::Auto;
      } else if (value == "exact") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::Exact;
      } else if (value == "ri") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
      } else {
        throw std::invalid_argument(
            "invalid --standard-two-electron-mode value: " + value);
      }
      continue;
    }
    throw std::invalid_argument("unknown argument: " + name);
  }
  return options;
}

double max_abs_difference(
    const std::vector<double>& left,
    const std::vector<double>& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("matrix size mismatch");
  }
  double max_abs_diff = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    max_abs_diff = std::max(max_abs_diff, std::abs(left[index] - right[index]));
  }
  return max_abs_diff;
}

std::vector<xmvb::vb::OrbitalPair> build_active_pairs_for_structure(
    const xmvb::vb::RawStructureData& raw_structure_data,
    int structure_index) {
  const int n_inactive_doubly_occupied_orbitals =
      (raw_structure_data.n_total_electrons - raw_structure_data.n_active_electrons) / 2;
  const int n_open_shell_electrons = raw_structure_data.spin_multiplicity - 1;
  const int n_active_beta_electrons =
      (raw_structure_data.n_active_electrons - n_open_shell_electrons) / 2;
  if (n_open_shell_electrons != 0) {
    throw std::invalid_argument(
        "raw-legacy overlap diagnostic currently assumes closed-shell active spaces");
  }

  const int active_offset = 2 * n_inactive_doubly_occupied_orbitals;
  const int* structure_orbitals =
      raw_structure_data.structure_orbitals_data(structure_index);
  std::vector<xmvb::vb::OrbitalPair> pairs;
  pairs.reserve(n_active_beta_electrons);
  for (int pair_index = 0; pair_index < n_active_beta_electrons; ++pair_index) {
    const int left_orbital =
        structure_orbitals[active_offset + 2 * pair_index] -
        n_inactive_doubly_occupied_orbitals - 1;
    const int right_orbital =
        structure_orbitals[active_offset + 2 * pair_index + 1] -
        n_inactive_doubly_occupied_orbitals - 1;
    pairs.push_back({left_orbital, right_orbital});
  }
  return pairs;
}

std::vector<double> build_raw_legacy_structure_overlap_matrix(
    const xmvb::vb::RawStructureData& raw_structure_data,
    const std::vector<double>& active_overlap_matrix,
    int n_active_orbitals) {
  const Eigen::Map<const Matrix> active_overlap(
      active_overlap_matrix.data(),
      n_active_orbitals,
      n_active_orbitals);
  xmvb::vb::DeterminantOverlapResolver overlap_resolver;
  const int n_structures = raw_structure_data.n_structures;
  std::vector<std::vector<xmvb::vb::LegacyStructureDeterminantTerm>> structure_terms(
      n_structures);
  for (int structure_index = 0; structure_index < n_structures; ++structure_index) {
    structure_terms[structure_index] =
        xmvb::vb::enumerate_legacy_determinant_terms(
            build_active_pairs_for_structure(raw_structure_data, structure_index));
  }

  std::vector<double> overlap_matrix(
      n_structures * n_structures,
      0.0);
  for (int left_structure = 0; left_structure < n_structures; ++left_structure) {
    for (int right_structure = 0; right_structure <= left_structure; ++right_structure) {
      const double overlap_value = xmvb::vb::legacy_structure_overlap(
          structure_terms[left_structure],
          structure_terms[right_structure],
          active_overlap,
          overlap_resolver);
      overlap_matrix[left_structure * n_structures + right_structure] =
          overlap_value;
      overlap_matrix[right_structure * n_structures + left_structure] =
          overlap_value;
    }
  }
  return overlap_matrix;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.standard_two_electron_mode = options.standard_two_electron_mode;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(options.input_path, load_options);

    xmvb::vb::StructureMatrixEvaluator evaluator;
    const auto prepared_active_space = evaluator.prepare_active_space(load_result.input);
    const auto structure_matrices =
        evaluator.evaluate(load_result.input, prepared_active_space);
    const auto raw_legacy_overlap =
        build_raw_legacy_structure_overlap_matrix(
            load_result.raw_structure_data,
            prepared_active_space.orbital_result.active_orbital_overlap_matrix,
            load_result.input.orbital_preparation_input.n_active_orbitals);

    std::cout << std::setprecision(16);
    std::cout << "n_structures = " << load_result.raw_structure_data.n_structures << '\n';
    std::cout << "n_active_orbitals = "
              << load_result.input.orbital_preparation_input.n_active_orbitals << '\n';
    std::cout << "max_abs_structure_overlap_diff = "
              << max_abs_difference(structure_matrices.overlap_matrix, raw_legacy_overlap)
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
