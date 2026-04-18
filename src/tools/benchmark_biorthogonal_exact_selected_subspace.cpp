#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_exact_selected_structure.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_exact_selected_structure_scf.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_prepared_input.hpp"
#include "vb/matrices/structure_matrix_evaluator.hpp"
#include "vb/matrices/structure_subspace_builder.hpp"
#include "vb/scf/cpp_vb_scf_evaluator.hpp"

namespace {

using DenseMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

enum class BenchmarkMode {
  Matrix,
  Scf,
};

struct Options {
  std::string input_path;
  int repeats = 3;
  int subspace_size = 0;
  int print_roots = 3;
  double symmetry_tolerance = 1.0e-8;
  BenchmarkMode mode = BenchmarkMode::Matrix;
};

struct MatrixDifferenceStats {
  double frobenius_norm = 0.0;
  double max_abs = 0.0;
};

struct RootDifferenceStats {
  int compared_root_count = 0;
  double max_abs = 0.0;
};

const char* benchmark_mode_name(BenchmarkMode mode) {
  switch (mode) {
    case BenchmarkMode::Matrix:
      return "matrix";
    case BenchmarkMode::Scf:
      return "scf";
  }
  throw std::invalid_argument("unsupported benchmark mode");
}

BenchmarkMode parse_benchmark_mode(
    const std::string& mode_text) {
  if (mode_text == "matrix") {
    return BenchmarkMode::Matrix;
  }
  if (mode_text == "scf") {
    return BenchmarkMode::Scf;
  }
  throw std::invalid_argument("unknown benchmark mode: " + mode_text);
}

void print_usage() {
  std::cerr << "usage: benchmark_biorthogonal_exact_selected_subspace <input.xmi>"
               " [--mode matrix|scf]"
               " [--repeats N]"
               " [--subspace-size N]"
               " [--symmetry-tolerance X]"
               " [--print-roots N]\n";
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
    if (argument_name == "--mode") {
      options.mode = parse_benchmark_mode(argument_value);
    } else if (argument_name == "--repeats") {
      options.repeats = std::stoi(argument_value);
    } else if (argument_name == "--subspace-size") {
      options.subspace_size = std::stoi(argument_value);
    } else if (argument_name == "--symmetry-tolerance") {
      options.symmetry_tolerance = std::stod(argument_value);
    } else if (argument_name == "--print-roots") {
      options.print_roots = std::stoi(argument_value);
    } else {
      throw std::invalid_argument("unknown argument: " + argument_name);
    }
  }

  if (options.repeats <= 0) {
    throw std::invalid_argument("--repeats must be positive");
  }
  if (options.subspace_size < 0) {
    throw std::invalid_argument("--subspace-size must be non-negative");
  }
  if (options.print_roots <= 0) {
    throw std::invalid_argument("--print-roots must be positive");
  }
  if (options.symmetry_tolerance < 0.0) {
    throw std::invalid_argument("--symmetry-tolerance must be non-negative");
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

std::vector<double> dense_matrix_to_vector(
    const DenseMatrix& matrix) {
  return std::vector<double>(
      matrix.data(),
      matrix.data() + matrix.size());
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

std::vector<int> build_selected_structure_indices(
    int n_structures,
    int requested_subspace_size) {
  if (n_structures <= 0) {
    throw std::invalid_argument("n_structures must be positive");
  }
  const int selected_count =
      (requested_subspace_size == 0) ? n_structures : requested_subspace_size;
  if (selected_count <= 0 || selected_count > n_structures) {
    throw std::invalid_argument("requested subspace size is out of range");
  }

  std::vector<int> selected_structure_indices;
  selected_structure_indices.reserve(xmvb::to_size(selected_count));
  for (int structure_index = 0; structure_index < selected_count; ++structure_index) {
    selected_structure_indices.push_back(structure_index);
  }
  return selected_structure_indices;
}

template <typename Callback>
double benchmark_repeated_wall_time_seconds(
    int repeats,
    Callback&& callback) {
  volatile double benchmark_sink = 0.0;
  const auto started_at = std::chrono::steady_clock::now();
  for (int repeat_index = 0; repeat_index < repeats; ++repeat_index) {
    benchmark_sink += callback();
  }
  static_cast<void>(benchmark_sink);
  return std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started_at).count();
}

std::vector<double> solve_generalized_roots(
    const DenseMatrix& hamiltonian,
    const DenseMatrix& overlap) {
  const xmvb::core::GeneralizedEigensolver generalized_eigensolver;
  return generalized_eigensolver.solve(
      dense_matrix_to_vector(hamiltonian),
      dense_matrix_to_vector(overlap),
      hamiltonian.rows()).eigenvalues;
}

void print_root_comparison(
    const std::vector<double>& reference_roots,
    const std::vector<double>& candidate_roots,
    int print_roots) {
  const int n_roots = std::min(
      print_roots,
      std::min(
          static_cast<int>(reference_roots.size()),
          static_cast<int>(candidate_roots.size())));
  std::cout << "printed_root_count = " << n_roots << '\n';
  for (int root_index = 0; root_index < n_roots; ++root_index) {
    const double diff =
        candidate_roots[xmvb::to_size(root_index)] -
        reference_roots[xmvb::to_size(root_index)];
    std::cout << "root[" << root_index << "]"
              << " nonorth=" << reference_roots[xmvb::to_size(root_index)]
              << " exact_biorth=" << candidate_roots[xmvb::to_size(root_index)]
              << " diff=" << diff << '\n';
  }
}

void print_common_header(
    const Options& options,
    const std::vector<int>& selected_structure_indices,
    int full_n_structures,
    int nonorth_n_determinants,
    int exact_n_determinants) {
  std::cout << "input = " << options.input_path << '\n';
  std::cout << "mode = " << benchmark_mode_name(options.mode) << '\n';
  std::cout << "full_n_structures = " << full_n_structures << '\n';
  std::cout << "selected_n_structures = "
            << selected_structure_indices.size() << '\n';
  std::cout << "nonorth_n_determinants = " << nonorth_n_determinants << '\n';
  std::cout << "exact_n_determinants = " << exact_n_determinants << '\n';
  std::cout << "repeats = " << options.repeats << '\n';
  std::cout << "symmetry_tolerance = " << options.symmetry_tolerance << '\n';
  std::cout << "selection_mode = "
            << ((options.subspace_size == 0) ? "full" : "prefix_subspace") << '\n';
}

double first_matrix_entry_or_zero(
    const DenseMatrix& matrix) {
  return (matrix.size() == 0) ? 0.0 : matrix(0, 0);
}

double first_vector_entry_or_zero(
    const std::vector<double>& values) {
  return values.empty() ? 0.0 : values.front();
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
    const std::vector<int> selected_structure_indices =
        build_selected_structure_indices(
            load_result.input.structure_data.n_structures,
            options.subspace_size);

    if (options.mode == BenchmarkMode::Matrix) {
      const xmvb::vb::StructureMatrixEvaluator nonorth_matrix_evaluator;
      const xmvb::vb::StructureSubspaceBuilder subspace_builder;
      const xmvb::vb::PreparedActiveSpaceContext prepared_active_space =
          nonorth_matrix_evaluator.prepare_active_space(load_result.input);
      const xmvb::vb::CppVbInput nonorth_input =
          (static_cast<int>(selected_structure_indices.size()) ==
           load_result.input.structure_data.n_structures)
              ? load_result.input
              : subspace_builder.build(
                    load_result.input,
                    selected_structure_indices);
      const xmvb::vb::FullDeterminantStructureData exact_structure_data =
          xmvb::vb::biorthogonal_vbscf::build_biorthogonal_full_structure_data(
              load_result.input,
              prepared_active_space);

      auto run_nonorth = [&]() {
        return nonorth_matrix_evaluator.evaluate(
            nonorth_input,
            prepared_active_space);
      };
      auto run_exact = [&]() {
        return xmvb::vb::biorthogonal_vbscf::
            build_biorthogonal_exact_selected_structure_matrices(
                exact_structure_data,
                selected_structure_indices,
                options.symmetry_tolerance);
      };

      const xmvb::vb::StructureAccumulationResult warmup_nonorth = run_nonorth();
      const auto warmup_exact = run_exact();
      static_cast<void>(first_vector_entry_or_zero(
          warmup_nonorth.hamiltonian_matrix));
      static_cast<void>(first_matrix_entry_or_zero(
          warmup_exact.physical_structure_hamiltonian));

      const double nonorth_seconds = benchmark_repeated_wall_time_seconds(
          options.repeats,
          [&]() {
            const xmvb::vb::StructureAccumulationResult result = run_nonorth();
            return first_vector_entry_or_zero(result.hamiltonian_matrix) +
                   first_vector_entry_or_zero(result.overlap_matrix);
          });
      const double exact_seconds = benchmark_repeated_wall_time_seconds(
          options.repeats,
          [&]() {
            const auto result = run_exact();
            return first_matrix_entry_or_zero(result.physical_structure_hamiltonian) +
                   first_matrix_entry_or_zero(result.physical_structure_overlap);
          });

      const xmvb::vb::StructureAccumulationResult nonorth_result = run_nonorth();
      const auto exact_result = run_exact();
      const DenseMatrix nonorth_overlap =
          map_square_matrix(
              nonorth_result.overlap_matrix,
              nonorth_result.n_structures);
      const DenseMatrix nonorth_hamiltonian =
          map_square_matrix(
              nonorth_result.hamiltonian_matrix,
              nonorth_result.n_structures);
      const DenseMatrix exact_overlap =
          exact_result.physical_structure_overlap;
      const DenseMatrix exact_hamiltonian =
          exact_result.physical_structure_hamiltonian;
      const MatrixDifferenceStats overlap_diff =
          compute_difference_stats(nonorth_overlap, exact_overlap);
      const MatrixDifferenceStats hamiltonian_diff =
          compute_difference_stats(nonorth_hamiltonian, exact_hamiltonian);
      const std::vector<double> nonorth_roots =
          solve_generalized_roots(nonorth_hamiltonian, nonorth_overlap);
      const std::vector<double> exact_roots =
          solve_generalized_roots(exact_hamiltonian, exact_overlap);
      const RootDifferenceStats root_diff =
          compute_root_difference_stats(nonorth_roots, exact_roots);

      print_common_header(
          options,
          selected_structure_indices,
          load_result.input.structure_data.n_structures,
          static_cast<int>(nonorth_input.structure_data.alpha_det.size()),
          exact_result.n_determinants);
      std::cout << "nonorth_total_seconds = " << nonorth_seconds << '\n';
      std::cout << "nonorth_avg_seconds = "
                << (nonorth_seconds / static_cast<double>(options.repeats)) << '\n';
      std::cout << "exact_biorth_total_seconds = " << exact_seconds << '\n';
      std::cout << "exact_biorth_avg_seconds = "
                << (exact_seconds / static_cast<double>(options.repeats)) << '\n';
      std::cout << "exact_biorth_speedup_over_nonorth = "
                << (nonorth_seconds / exact_seconds) << '\n';
      std::cout << "overlap_diff_fro = " << overlap_diff.frobenius_norm << '\n';
      std::cout << "overlap_diff_max_abs = " << overlap_diff.max_abs << '\n';
      std::cout << "hamiltonian_diff_fro = " << hamiltonian_diff.frobenius_norm << '\n';
      std::cout << "hamiltonian_diff_max_abs = " << hamiltonian_diff.max_abs << '\n';
      std::cout << "root_diff_compared_count = "
                << root_diff.compared_root_count << '\n';
      std::cout << "root_diff_max_abs = " << root_diff.max_abs << '\n';
      print_root_comparison(
          nonorth_roots,
          exact_roots,
          options.print_roots);
      return 0;
    }

    const auto prepared_input =
        xmvb::vb::biorthogonal_vbscf::prepare_biorthogonal_input(
            load_result.input);
    const xmvb::vb::CppVbScfEvaluator nonorth_evaluator;

    auto run_nonorth = [&]() {
      if (static_cast<int>(selected_structure_indices.size()) ==
          load_result.input.structure_data.n_structures) {
        return nonorth_evaluator.evaluate(
            load_result.input,
            load_result.nuclear_repulsion_energy);
      }
      return nonorth_evaluator.evaluate_subspace(
          load_result.input,
          selected_structure_indices,
          load_result.nuclear_repulsion_energy);
    };

    auto run_exact = [&]() {
      return xmvb::vb::biorthogonal_vbscf::
          evaluate_biorthogonal_exact_selected_structure_scf(
              prepared_input,
              selected_structure_indices,
              load_result.nuclear_repulsion_energy,
              options.symmetry_tolerance);
    };

    const xmvb::vb::CppVbScfResult warmup_nonorth = run_nonorth();
    const auto warmup_exact = run_exact();
    static_cast<void>(warmup_nonorth.total_energy);
    static_cast<void>(warmup_exact.total_energy);

    const double nonorth_seconds = benchmark_repeated_wall_time_seconds(
        options.repeats,
        [&]() {
          const xmvb::vb::CppVbScfResult result = run_nonorth();
          return result.total_energy + result.average_structure_overlap;
        });
    const double exact_seconds = benchmark_repeated_wall_time_seconds(
        options.repeats,
        [&]() {
          const auto result = run_exact();
          return result.total_energy + result.average_structure_overlap;
        });

    const xmvb::vb::CppVbScfResult nonorth_result = run_nonorth();
    const auto exact_result = run_exact();
    const DenseMatrix nonorth_overlap =
        map_square_matrix(
            nonorth_result.structure_matrices.overlap_matrix,
            nonorth_result.n_structures);
    const DenseMatrix nonorth_hamiltonian =
        map_square_matrix(
            nonorth_result.structure_matrices.hamiltonian_matrix,
            nonorth_result.n_structures);
    const DenseMatrix exact_overlap =
        map_square_matrix(
            exact_result.structure_matrices.overlap_matrix,
            exact_result.exact_evaluation.n_selected_structures);
    const DenseMatrix exact_hamiltonian =
        map_square_matrix(
            exact_result.structure_matrices.hamiltonian_matrix,
            exact_result.exact_evaluation.n_selected_structures);
    const MatrixDifferenceStats overlap_diff =
        compute_difference_stats(nonorth_overlap, exact_overlap);
    const MatrixDifferenceStats hamiltonian_diff =
        compute_difference_stats(nonorth_hamiltonian, exact_hamiltonian);
    const RootDifferenceStats root_diff =
        compute_root_difference_stats(
            nonorth_result.electronic_state_energies,
            exact_result.exact_evaluation.eigenvalues);

    print_common_header(
        options,
        selected_structure_indices,
        load_result.input.structure_data.n_structures,
        static_cast<int>(
            xmvb::vb::StructureSubspaceBuilder()
                .build(load_result.input, selected_structure_indices)
                .structure_data.alpha_det.size()),
        static_cast<int>(prepared_input.structure_data.alpha_det.size()));
    std::cout << "nonorth_total_seconds = " << nonorth_seconds << '\n';
    std::cout << "nonorth_avg_seconds = "
              << (nonorth_seconds / static_cast<double>(options.repeats)) << '\n';
    std::cout << "exact_biorth_total_seconds = " << exact_seconds << '\n';
    std::cout << "exact_biorth_avg_seconds = "
              << (exact_seconds / static_cast<double>(options.repeats)) << '\n';
    std::cout << "exact_biorth_speedup_over_nonorth = "
              << (nonorth_seconds / exact_seconds) << '\n';
    std::cout << "overlap_diff_fro = " << overlap_diff.frobenius_norm << '\n';
    std::cout << "overlap_diff_max_abs = " << overlap_diff.max_abs << '\n';
    std::cout << "hamiltonian_diff_fro = " << hamiltonian_diff.frobenius_norm << '\n';
    std::cout << "hamiltonian_diff_max_abs = " << hamiltonian_diff.max_abs << '\n';
    std::cout << "root_diff_compared_count = "
              << root_diff.compared_root_count << '\n';
    std::cout << "root_diff_max_abs = " << root_diff.max_abs << '\n';
    std::cout << "nonorth_total_energy = " << nonorth_result.total_energy << '\n';
    std::cout << "exact_biorth_total_energy = " << exact_result.total_energy << '\n';
    std::cout << "total_energy_diff = "
              << (exact_result.total_energy - nonorth_result.total_energy) << '\n';
    print_root_comparison(
        nonorth_result.electronic_state_energies,
        exact_result.exact_evaluation.eigenvalues,
        options.print_roots);
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "benchmark_biorthogonal_exact_selected_subspace failed: "
              << exception.what() << '\n';
    return 1;
  }
}
