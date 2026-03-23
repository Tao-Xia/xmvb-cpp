#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace {

struct Options {
  std::string input_path;
  xmvb::vb::VBSCFAlgorithm algorithm = xmvb::vb::VBSCFAlgorithm::Original;
  int count = 8;
  double step = 1.0e-6;
};

void print_usage() {
  std::cerr << "usage: check_structure_overlap_weight <input.xmi> "
               "[--algorithm original|biorthogonal] "
               "[--count N] [--step h]\n";
}

Options parse_arguments(int argc, char** argv) {
  if (argc < 2 || ((argc - 2) % 2 != 0)) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }

  Options options;
  options.input_path = argv[1];
  for (int argument_index = 2; argument_index < argc; argument_index += 2) {
    const std::string argument_name = argv[argument_index];
    const std::string argument_value = argv[argument_index + 1];
    if (argument_name == "--algorithm") {
      if (argument_value == "original") {
        options.algorithm = xmvb::vb::VBSCFAlgorithm::Original;
      } else if (argument_value == "biorthogonal") {
        options.algorithm = xmvb::vb::VBSCFAlgorithm::Biorthogonal;
      } else {
        throw std::invalid_argument("invalid algorithm: " + argument_value);
      }
      continue;
    }
    if (argument_name == "--count") {
      options.count = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--step") {
      options.step = std::stod(argument_value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument_name);
  }

  if (options.count <= 0) {
    throw std::invalid_argument("--count must be positive");
  }
  if (options.step <= 0.0) {
    throw std::invalid_argument("--step must be positive");
  }
  return options;
}

double structure_upper_overlap_weight(
    const std::vector<double>& eigenvector_matrix,
    const std::vector<double>& eigenvalues,
    int n_structures,
    int structure_row,
    int structure_column) {
  const double state_energy = eigenvalues.front();
  const double coefficient_row =
      eigenvector_matrix[static_cast<std::size_t>(0) * n_structures + structure_row];
  const double coefficient_column =
      eigenvector_matrix[static_cast<std::size_t>(0) * n_structures + structure_column];
  if (structure_row == structure_column) {
    return -state_energy * coefficient_row * coefficient_column;
  }
  return -2.0 * state_energy * coefficient_row * coefficient_column;
}

double evaluate_ground_state_energy(
    const std::vector<double>& hamiltonian_matrix,
    const std::vector<double>& overlap_matrix,
    int n_structures) {
  xmvb::core::GeneralizedEigensolver eigensolver;
  return eigensolver.solve(hamiltonian_matrix, overlap_matrix, n_structures).eigenvalues.front();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
    xmvb::vb::CppActiveSpaceGradientEvaluator evaluator(options.algorithm);
    const auto result = evaluator.evaluate(load_result.input, load_result.nuclear_repulsion_energy);
    const int n_structures = result.scf_result.n_structures;

    std::vector<std::pair<double, int>> ranked_entries;
    for (int row = 0; row < n_structures; ++row) {
      for (int column = 0; column <= row; ++column) {
        const double weight = structure_upper_overlap_weight(
            result.scf_result.eigenvector_matrix,
            result.scf_result.electronic_state_energies,
            n_structures,
            row,
            column);
        ranked_entries.emplace_back(std::abs(weight), row * n_structures + column);
      }
    }
    std::sort(
        ranked_entries.begin(),
        ranked_entries.end(),
        [](const auto& left, const auto& right) {
          if (left.first != right.first) {
            return left.first > right.first;
          }
          return left.second < right.second;
        });

    const int n_to_report =
        std::min(options.count, static_cast<int>(ranked_entries.size()));
    std::cout << std::setprecision(12);
    std::cout << "algorithm = " << xmvb::vb::vb_scf_algorithm_name(options.algorithm) << '\n';
    std::cout << "initial_electronic_energy = " << result.scf_result.electronic_energy << '\n';
    std::cout << "finite_difference_step = " << options.step << '\n';
    std::cout << "reported_entries = " << n_to_report << '\n';

    for (int report_index = 0; report_index < n_to_report; ++report_index) {
      const int packed_index = ranked_entries[static_cast<std::size_t>(report_index)].second;
      const int row = packed_index / n_structures;
      const int column = packed_index % n_structures;
      const double analytic = structure_upper_overlap_weight(
          result.scf_result.eigenvector_matrix,
          result.scf_result.electronic_state_energies,
          n_structures,
          row,
          column);

      std::vector<double> plus_overlap = result.scf_result.structure_matrices.overlap_matrix;
      std::vector<double> minus_overlap = result.scf_result.structure_matrices.overlap_matrix;
      plus_overlap[static_cast<std::size_t>(row) * n_structures + column] += options.step;
      minus_overlap[static_cast<std::size_t>(row) * n_structures + column] -= options.step;
      if (row != column) {
        plus_overlap[static_cast<std::size_t>(column) * n_structures + row] += options.step;
        minus_overlap[static_cast<std::size_t>(column) * n_structures + row] -= options.step;
      }

      const double plus_energy = evaluate_ground_state_energy(
          result.scf_result.structure_matrices.hamiltonian_matrix,
          plus_overlap,
          n_structures);
      const double minus_energy = evaluate_ground_state_energy(
          result.scf_result.structure_matrices.hamiltonian_matrix,
          minus_overlap,
          n_structures);
      const double finite_difference = (plus_energy - minus_energy) / (2.0 * options.step);
      const double absolute_error = std::abs(analytic - finite_difference);
      const double relative_error =
          absolute_error / std::max(1.0, std::abs(finite_difference));

      std::cout << "entry[" << report_index << "]"
                << " row=" << row
                << " column=" << column
                << " analytic=" << analytic
                << " fd=" << finite_difference
                << " abs_error=" << absolute_error
                << " rel_error=" << relative_error
                << '\n';
    }

    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
