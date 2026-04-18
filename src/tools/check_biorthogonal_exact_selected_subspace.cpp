#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_exact_selected_structure_scf.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_prepared_input.hpp"
#include "vb/scf/cpp_vb_scf_evaluator.hpp"

namespace {

using DenseMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

struct Options {
  std::string input_path;
  int print_roots = 6;
  int min_subspace_size = 1;
  int max_subspace_size = std::numeric_limits<int>::max();
};

struct MatrixDifferenceStats {
  double frobenius_norm = 0.0;
  double max_abs = 0.0;
};

struct RootDifferenceStats {
  int compared_root_count = 0;
  double max_abs = 0.0;
};

void print_usage() {
  std::cerr << "usage: check_biorthogonal_exact_selected_subspace <input.xmi>"
               " [--print-roots N]"
               " [--min-subspace-size N]"
               " [--max-subspace-size N]\n";
}

Options parse_options(int argc, char** argv) {
  if (argc < 2 || ((argc - 2) % 2 != 0)) {
    print_usage();
    throw std::invalid_argument("invalid arguments");
  }

  Options options;
  options.input_path = argv[1];
  for (int argument_index = 2; argument_index < argc; argument_index += 2) {
    const std::string argument_name = argv[argument_index];
    const std::string argument_value = argv[argument_index + 1];
    if (argument_name == "--print-roots") {
      options.print_roots = std::stoi(argument_value);
    } else if (argument_name == "--min-subspace-size") {
      options.min_subspace_size = std::stoi(argument_value);
    } else if (argument_name == "--max-subspace-size") {
      options.max_subspace_size = std::stoi(argument_value);
    } else {
      throw std::invalid_argument("unknown argument: " + argument_name);
    }
  }

  if (options.print_roots <= 0) {
    throw std::invalid_argument("--print-roots must be positive");
  }
  if (options.min_subspace_size <= 0) {
    throw std::invalid_argument("--min-subspace-size must be positive");
  }
  if (options.max_subspace_size <= 0) {
    throw std::invalid_argument("--max-subspace-size must be positive");
  }
  if (options.min_subspace_size > options.max_subspace_size) {
    throw std::invalid_argument(
        "--min-subspace-size must not exceed --max-subspace-size");
  }
  return options;
}

DenseMatrix map_square_matrix(
    const std::vector<double>& matrix_data,
    int dimension) {
  if (static_cast<int>(matrix_data.size()) != dimension * dimension) {
    throw std::invalid_argument("matrix size does not match dimension");
  }
  DenseMatrix matrix(dimension, dimension);
  Eigen::Map<const DenseMatrix> mapped_matrix(
      matrix_data.data(),
      dimension,
      dimension);
  matrix = mapped_matrix;
  return matrix;
}

MatrixDifferenceStats compute_difference_stats(
    const DenseMatrix& left_matrix,
    const DenseMatrix& right_matrix) {
  if (left_matrix.rows() != right_matrix.rows() ||
      left_matrix.cols() != right_matrix.cols()) {
    throw std::invalid_argument("matrix dimensions do not match");
  }

  const DenseMatrix difference = left_matrix - right_matrix;
  MatrixDifferenceStats stats;
  stats.frobenius_norm = difference.norm();
  if (difference.size() > 0) {
    stats.max_abs = difference.cwiseAbs().maxCoeff();
  }
  return stats;
}

RootDifferenceStats compute_root_difference_stats(
    const std::vector<double>& reference_roots,
    const std::vector<double>& candidate_roots) {
  RootDifferenceStats stats;
  stats.compared_root_count = std::min(
      static_cast<int>(reference_roots.size()),
      static_cast<int>(candidate_roots.size()));
  for (int root_index = 0; root_index < stats.compared_root_count; ++root_index) {
    const double diff = std::abs(
        candidate_roots[xmvb::to_size(root_index)] -
        reference_roots[xmvb::to_size(root_index)]);
    stats.max_abs = std::max(stats.max_abs, diff);
  }
  return stats;
}

std::vector<std::vector<int>> enumerate_structure_subspaces(
    int n_structures,
    int min_subspace_size,
    int max_subspace_size) {
  if (n_structures <= 0) {
    throw std::invalid_argument("n_structures must be positive");
  }
  if (n_structures > 20) {
    throw std::invalid_argument(
        "refusing to enumerate all structure subspaces for n_structures > 20");
  }

  std::vector<std::vector<int>> structure_subspaces;
  const std::uint64_t mask_limit = std::uint64_t{1} << n_structures;
  for (std::uint64_t mask = 1; mask < mask_limit; ++mask) {
    const int subset_size = static_cast<int>(__builtin_popcountll(mask));
    if (subset_size < min_subspace_size || subset_size > max_subspace_size) {
      continue;
    }

    std::vector<int> selected_structure_indices;
    selected_structure_indices.reserve(xmvb::to_size(subset_size));
    for (int structure_index = 0; structure_index < n_structures; ++structure_index) {
      if ((mask & (std::uint64_t{1} << structure_index)) != 0) {
        selected_structure_indices.push_back(structure_index);
      }
    }
    structure_subspaces.push_back(std::move(selected_structure_indices));
  }
  return structure_subspaces;
}

std::string format_structure_index_list(
    const std::vector<int>& structure_indices) {
  std::ostringstream stream;
  stream << '{';
  for (std::size_t index = 0; index < structure_indices.size(); ++index) {
    if (index > 0) {
      stream << ',';
    }
    stream << structure_indices[index];
  }
  stream << '}';
  return stream.str();
}

void print_root_comparison(
    const std::vector<double>& reference_roots,
    const std::vector<double>& candidate_roots,
    int print_roots,
    const char* label) {
  const int n_roots = std::min(
      print_roots,
      std::min(
          static_cast<int>(reference_roots.size()),
          static_cast<int>(candidate_roots.size())));
  std::cout << label << "_root_comparison_count = " << n_roots << '\n';
  for (int root_index = 0; root_index < n_roots; ++root_index) {
    const double diff = candidate_roots[xmvb::to_size(root_index)] -
        reference_roots[xmvb::to_size(root_index)];
    std::cout << label << "_root[" << root_index << "]"
              << " reference=" << reference_roots[xmvb::to_size(root_index)]
              << " candidate=" << candidate_roots[xmvb::to_size(root_index)]
              << " diff=" << diff << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::cout << std::setprecision(15);

    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.ao_integral_source = xmvb::vb::AoIntegralSource::LibcintMaterializedCpp;
    load_options.orbital_guess_source = xmvb::vb::OrbitalGuessSource::Cpp;
    load_options.standard_two_electron_mode =
        xmvb::vb::StandardTwoElectronMode::Exact;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(options.input_path, load_options);
    const auto prepared_input =
        xmvb::vb::biorthogonal_vbscf::prepare_biorthogonal_input(
            load_result.input);

    const xmvb::vb::CppVbScfEvaluator nonorth_evaluator;
    const auto structure_subspaces = enumerate_structure_subspaces(
        load_result.input.structure_data.n_structures,
        options.min_subspace_size,
        std::min(
            options.max_subspace_size,
            load_result.input.structure_data.n_structures));

    std::cout << "input = " << options.input_path << '\n';
    std::cout << "full_n_structures = "
              << load_result.input.structure_data.n_structures << '\n';
    std::cout << "enumerated_subspace_count = " << structure_subspaces.size() << '\n';

    double worst_root_diff = 0.0;
    double worst_overlap_diff = 0.0;
    double worst_hamiltonian_diff = 0.0;
    double worst_overlap_symmetry = 0.0;
    double worst_hamiltonian_symmetry = 0.0;
    double worst_total_energy_diff = 0.0;
    double worst_one_electron_reference_diff = 0.0;
    std::string worst_root_subset;

    for (const auto& selected_structure_indices : structure_subspaces) {
      const xmvb::vb::CppVbScfResult nonorth_result =
          nonorth_evaluator.evaluate_subspace(
              load_result.input,
              selected_structure_indices,
              load_result.nuclear_repulsion_energy);
      const auto exact_scf_result =
          xmvb::vb::biorthogonal_vbscf::
              evaluate_biorthogonal_exact_selected_structure_scf(
                  prepared_input,
                  selected_structure_indices,
                  load_result.nuclear_repulsion_energy,
                  1.0e-8);
      const auto& exact_result = exact_scf_result.exact_evaluation;

      const DenseMatrix reference_overlap =
          map_square_matrix(
              nonorth_result.structure_matrices.overlap_matrix,
              static_cast<int>(selected_structure_indices.size()));
      const DenseMatrix reference_hamiltonian =
          map_square_matrix(
              nonorth_result.structure_matrices.hamiltonian_matrix,
              static_cast<int>(selected_structure_indices.size()));

      const MatrixDifferenceStats overlap_difference =
          compute_difference_stats(
              reference_overlap,
              exact_result.physical_structure_overlap);
      const MatrixDifferenceStats hamiltonian_difference =
          compute_difference_stats(
              reference_hamiltonian,
              exact_result.physical_structure_hamiltonian);
      const RootDifferenceStats root_difference =
          compute_root_difference_stats(
              nonorth_result.electronic_state_energies,
              exact_result.eigenvalues);

      worst_overlap_diff = std::max(worst_overlap_diff, overlap_difference.max_abs);
      worst_hamiltonian_diff = std::max(
          worst_hamiltonian_diff,
          hamiltonian_difference.max_abs);
      worst_overlap_symmetry = std::max(
          worst_overlap_symmetry,
          exact_result.structure_overlap_symmetry_residual_max_abs);
      worst_hamiltonian_symmetry = std::max(
          worst_hamiltonian_symmetry,
          exact_result.structure_hamiltonian_symmetry_residual_max_abs);
      worst_total_energy_diff = std::max(
          worst_total_energy_diff,
          std::abs(exact_scf_result.total_energy - nonorth_result.total_energy));
      worst_one_electron_reference_diff = std::max(
          worst_one_electron_reference_diff,
          std::abs(
              exact_scf_result.one_electron_reference_energy -
              nonorth_result.one_electron_reference_energy));
      if (root_difference.max_abs > worst_root_diff) {
        worst_root_diff = root_difference.max_abs;
        worst_root_subset = format_structure_index_list(selected_structure_indices);
      }

      std::cout << "subspace = "
                << format_structure_index_list(selected_structure_indices) << '\n';
      std::cout << "subspace_size = " << selected_structure_indices.size() << '\n';
      std::cout << "overlap_diff_fro = "
                << overlap_difference.frobenius_norm << '\n';
      std::cout << "overlap_diff_max_abs = "
                << overlap_difference.max_abs << '\n';
      std::cout << "hamiltonian_diff_fro = "
                << hamiltonian_difference.frobenius_norm << '\n';
      std::cout << "hamiltonian_diff_max_abs = "
                << hamiltonian_difference.max_abs << '\n';
      std::cout << "root_diff_max_abs = "
                << root_difference.max_abs << '\n';
      std::cout << "exact_overlap_symmetry_residual_max_abs = "
                << exact_result.structure_overlap_symmetry_residual_max_abs << '\n';
      std::cout << "exact_hamiltonian_symmetry_residual_max_abs = "
                << exact_result.structure_hamiltonian_symmetry_residual_max_abs << '\n';
      std::cout << "exact_ground_total_energy_diff = "
                << std::abs(exact_scf_result.total_energy - nonorth_result.total_energy)
                << '\n';
      std::cout << "exact_one_electron_reference_diff = "
                << std::abs(
                       exact_scf_result.one_electron_reference_energy -
                       nonorth_result.one_electron_reference_energy)
                << '\n';
      print_root_comparison(
          nonorth_result.electronic_state_energies,
          exact_result.eigenvalues,
          options.print_roots,
          "exact_selected");
    }

    std::cout << "worst_root_diff_max_abs = " << worst_root_diff << '\n';
    std::cout << "worst_root_diff_subspace = " << worst_root_subset << '\n';
    std::cout << "worst_overlap_diff_max_abs = " << worst_overlap_diff << '\n';
    std::cout << "worst_hamiltonian_diff_max_abs = "
              << worst_hamiltonian_diff << '\n';
    std::cout << "worst_overlap_symmetry_residual_max_abs = "
              << worst_overlap_symmetry << '\n';
    std::cout << "worst_hamiltonian_symmetry_residual_max_abs = "
              << worst_hamiltonian_symmetry << '\n';
    std::cout << "worst_ground_total_energy_diff = "
              << worst_total_energy_diff << '\n';
    std::cout << "worst_one_electron_reference_diff = "
              << worst_one_electron_reference_diff << '\n';
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "error: " << exception.what() << '\n';
    return 1;
  }
}
