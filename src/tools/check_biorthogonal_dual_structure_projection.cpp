#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_determinant_hamiltonian.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_orbital_integrals.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_prepared_input.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_structure_expansion.hpp"
#include "vb/matrices/full_structure_builder.hpp"
#include "vb/matrices/structure_subspace_builder.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"
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
  std::cerr << "usage: check_biorthogonal_dual_structure_projection <input.xmi>"
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

double compute_max_abs_symmetry_residual(const DenseMatrix& matrix) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("matrix must be square");
  }
  if (matrix.size() == 0) {
    return 0.0;
  }
  return (matrix - matrix.transpose()).cwiseAbs().maxCoeff();
}

std::vector<double> solve_real_nonsymmetric_eigenvalues(
    const DenseMatrix& matrix,
    double imaginary_tolerance) {
  Eigen::EigenSolver<DenseMatrix> solver(matrix, false);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("failed to diagonalize non-symmetric matrix");
  }

  std::vector<double> eigenvalues;
  eigenvalues.reserve(xmvb::to_size(matrix.rows()));
  for (int eigenvalue_index = 0; eigenvalue_index < matrix.rows(); ++eigenvalue_index) {
    const std::complex<double> eigenvalue = solver.eigenvalues()(eigenvalue_index);
    if (std::abs(eigenvalue.imag()) > imaginary_tolerance) {
      throw std::runtime_error(
          "non-negligible imaginary eigenvalue encountered in generalized pencil");
    }
    eigenvalues.push_back(eigenvalue.real());
  }
  std::sort(eigenvalues.begin(), eigenvalues.end());
  return eigenvalues;
}

std::vector<double> solve_real_generalized_eigenvalues(
    const DenseMatrix& hamiltonian,
    const DenseMatrix& overlap,
    double imaginary_tolerance) {
  if (hamiltonian.rows() != hamiltonian.cols() ||
      overlap.rows() != overlap.cols() ||
      hamiltonian.rows() != overlap.rows()) {
    throw std::invalid_argument("generalized eigenproblem dimensions are inconsistent");
  }

  Eigen::LLT<DenseMatrix> overlap_factor(overlap);
  if (overlap_factor.info() != Eigen::Success) {
    throw std::runtime_error("generalized eigenproblem overlap is not positive definite");
  }

  const DenseMatrix operator_matrix = overlap_factor.solve(hamiltonian);
  return solve_real_nonsymmetric_eigenvalues(operator_matrix, imaginary_tolerance);
}

DenseMatrix gather_columns(
    const DenseMatrix& matrix,
    const std::vector<int>& column_indices) {
  DenseMatrix gathered(
      matrix.rows(),
      static_cast<int>(column_indices.size()));
  for (int column_offset = 0;
       column_offset < static_cast<int>(column_indices.size());
       ++column_offset) {
    const int source_column = column_indices[xmvb::to_size(column_offset)];
    if (source_column < 0 || source_column >= matrix.cols()) {
      throw std::out_of_range("column index is out of range");
    }
    gathered.col(column_offset) = matrix.col(source_column);
  }
  return gathered;
}

void fill_determinant_matrices(
    const xmvb::vb::FullDeterminantStructureBuildResult&
        build_result,
    DenseMatrix* overlap,
    DenseMatrix* hamiltonian) {
  if (overlap == nullptr || hamiltonian == nullptr) {
    throw std::invalid_argument("matrix output pointers must not be null");
  }
  if (overlap->rows() != overlap->cols() ||
      hamiltonian->rows() != hamiltonian->cols() ||
      overlap->rows() != hamiltonian->rows()) {
    throw std::invalid_argument("determinant matrices must be square and aligned");
  }
  const int n_determinants = overlap->rows();
  for (int column_index = 0; column_index < n_determinants; ++column_index) {
    for (int row_index = 0; row_index < n_determinants; ++row_index) {
      const auto& pair_result = build_result.pair_evaluation(row_index, column_index);
      (*overlap)(row_index, column_index) = pair_result.overlap_determinant;
      (*hamiltonian)(row_index, column_index) = pair_result.total_hamiltonian;
    }
  }
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
    const xmvb::vb::FullDeterminantStructureData& complete_structure_data =
        prepared_input.structure_data;

    const xmvb::vb::CppVbScfEvaluator nonorth_evaluator;
    const xmvb::vb::StructureSubspaceBuilder subspace_builder;
    const xmvb::vb::FullDeterminantStructureHamiltonianOverlapBuilder
        determinant_builder;

    const auto full_structure_expansion =
        xmvb::vb::biorthogonal_vbscf::build_biorthogonal_structure_expansion(
            complete_structure_data);
    const auto full_determinant_build_result =
        determinant_builder.build_with_pair_evaluations(complete_structure_data);
    const int full_n_determinants =
        static_cast<int>(complete_structure_data.alpha_det.size());
    DenseMatrix full_determinant_overlap =
        DenseMatrix::Zero(full_n_determinants, full_n_determinants);
    DenseMatrix full_determinant_hamiltonian =
        DenseMatrix::Zero(full_n_determinants, full_n_determinants);
    fill_determinant_matrices(
        full_determinant_build_result,
        &full_determinant_overlap,
        &full_determinant_hamiltonian);

    const auto full_determinant_orbital_integrals =
        xmvb::vb::biorthogonal_vbscf::build_biorthogonal_orbital_integrals(
            complete_structure_data.n_active_orbitals,
            complete_structure_data.ovlp_act,
            complete_structure_data.h1e_act);
    const DenseMatrix full_determinant_biorth_hamiltonian =
        xmvb::vb::biorthogonal_vbscf::build_biorthogonal_determinant_hamiltonian_matrix(
            full_structure_expansion.determinants,
            full_determinant_orbital_integrals,
            xmvb::vb::make_active_space_two_electron_view(
                complete_structure_data.eri_act));

    const MatrixDifferenceStats full_determinant_relation_difference =
        compute_difference_stats(
            full_determinant_hamiltonian,
            full_determinant_overlap * full_determinant_biorth_hamiltonian);

    const auto structure_subspaces = enumerate_structure_subspaces(
        complete_structure_data.n_structures,
        options.min_subspace_size,
        std::min(options.max_subspace_size, complete_structure_data.n_structures));

    std::cout << "input = " << options.input_path << '\n';
    std::cout << "full_n_structures = " << complete_structure_data.n_structures << '\n';
    std::cout << "full_n_determinants = " << full_n_determinants << '\n';
    std::cout << "prepared_n_active_orbitals = "
              << complete_structure_data.n_active_orbitals << '\n';
    std::cout << "full_determinant_relation_diff_fro = "
              << full_determinant_relation_difference.frobenius_norm << '\n';
    std::cout << "full_determinant_relation_diff_max_abs = "
              << full_determinant_relation_difference.max_abs << '\n';
    std::cout << "enumerated_subspace_count = " << structure_subspaces.size() << '\n';

    double worst_naive_root_diff = 0.0;
    double worst_local_dual_root_diff = 0.0;
    double worst_full_dual_root_diff = 0.0;
    double worst_naive_overlap_diff = 0.0;
    double worst_local_dual_overlap_diff = 0.0;
    double worst_full_dual_overlap_diff = 0.0;
    double worst_naive_hamiltonian_diff = 0.0;
    double worst_local_dual_hamiltonian_diff = 0.0;
    double worst_full_dual_hamiltonian_diff = 0.0;

    for (std::size_t subset_index = 0;
         subset_index < structure_subspaces.size();
         ++subset_index) {
      const auto& selected_structure_indices = structure_subspaces[subset_index];
      const xmvb::vb::CppVbScfResult nonorth_result =
          nonorth_evaluator.evaluate_subspace(
              load_result.input,
              selected_structure_indices,
              load_result.nuclear_repulsion_energy);
      const xmvb::vb::FullDeterminantStructureData subspace_structure_data =
          subspace_builder.build(
              complete_structure_data,
              selected_structure_indices);
      const auto local_structure_expansion =
          xmvb::vb::biorthogonal_vbscf::build_biorthogonal_structure_expansion(
              subspace_structure_data);
      const auto local_determinant_build_result =
          determinant_builder.build_with_pair_evaluations(subspace_structure_data);
      const int local_n_determinants =
          static_cast<int>(subspace_structure_data.alpha_det.size());
      DenseMatrix local_determinant_overlap =
          DenseMatrix::Zero(local_n_determinants, local_n_determinants);
      DenseMatrix local_determinant_hamiltonian =
          DenseMatrix::Zero(local_n_determinants, local_n_determinants);
      fill_determinant_matrices(
          local_determinant_build_result,
          &local_determinant_overlap,
          &local_determinant_hamiltonian);

      const DenseMatrix local_determinant_biorth_hamiltonian =
          xmvb::vb::biorthogonal_vbscf::build_biorthogonal_determinant_hamiltonian_matrix(
              local_structure_expansion.determinants,
              full_determinant_orbital_integrals,
              xmvb::vb::make_active_space_two_electron_view(
                  subspace_structure_data.eri_act));

      const DenseMatrix original_structure_overlap =
          map_square_matrix(
              nonorth_result.structure_matrices.overlap_matrix,
              subspace_structure_data.n_structures);
      const DenseMatrix original_structure_hamiltonian =
          map_square_matrix(
              nonorth_result.structure_matrices.hamiltonian_matrix,
              subspace_structure_data.n_structures);

      const DenseMatrix local_structure_to_determinant =
          local_structure_expansion.structure_to_determinant;
      const DenseMatrix full_structure_to_determinant =
          gather_columns(
              full_structure_expansion.structure_to_determinant,
              selected_structure_indices);

      // The current prototype freezes identical left/right structure coefficients,
      // so its selected-space pencil is built entirely inside the filtered
      // determinant list with the Euclidean metric T_k^T T_k.
      const DenseMatrix naive_structure_metric =
          local_structure_to_determinant.transpose() *
          local_structure_to_determinant;
      const DenseMatrix naive_structure_hamiltonian =
          local_structure_to_determinant.transpose() *
          local_determinant_biorth_hamiltonian *
          local_structure_to_determinant;

      // Replacing the left structure coefficients by S_kk T_k restores the
      // physical subspace metric, but it still misses the omitted-determinant
      // coupling carried by the full biorthogonal Hamiltonian.
      const DenseMatrix local_dual_left_projector =
          local_determinant_overlap * local_structure_to_determinant;
      const DenseMatrix local_dual_structure_metric =
          local_dual_left_projector.transpose() *
          local_structure_to_determinant;
      const DenseMatrix local_dual_structure_hamiltonian =
          local_dual_left_projector.transpose() *
          local_determinant_biorth_hamiltonian *
          local_structure_to_determinant;

      // Using the full determinant biorthogonal operator together with the full
      // left projector S_full T_hat reproduces the original nonorthogonal
      // structure pencil exactly, even after selecting only a subset of
      // structures on the right.
      const DenseMatrix full_dual_left_projector =
          full_determinant_overlap * full_structure_to_determinant;
      const DenseMatrix full_dual_structure_metric =
          full_dual_left_projector.transpose() *
          full_structure_to_determinant;
      const DenseMatrix full_dual_structure_hamiltonian =
          full_dual_left_projector.transpose() *
          full_determinant_biorth_hamiltonian *
          full_structure_to_determinant;

      const MatrixDifferenceStats naive_overlap_difference =
          compute_difference_stats(
              original_structure_overlap,
              naive_structure_metric);
      const MatrixDifferenceStats local_dual_overlap_difference =
          compute_difference_stats(
              original_structure_overlap,
              local_dual_structure_metric);
      const MatrixDifferenceStats full_dual_overlap_difference =
          compute_difference_stats(
              original_structure_overlap,
              full_dual_structure_metric);
      const MatrixDifferenceStats naive_hamiltonian_difference =
          compute_difference_stats(
              original_structure_hamiltonian,
              naive_structure_hamiltonian);
      const MatrixDifferenceStats local_dual_hamiltonian_difference =
          compute_difference_stats(
              original_structure_hamiltonian,
              local_dual_structure_hamiltonian);
      const MatrixDifferenceStats full_dual_hamiltonian_difference =
          compute_difference_stats(
              original_structure_hamiltonian,
              full_dual_structure_hamiltonian);

      const std::vector<double> naive_roots =
          solve_real_generalized_eigenvalues(
              naive_structure_hamiltonian,
              naive_structure_metric,
              1.0e-8);
      const std::vector<double> local_dual_roots =
          solve_real_generalized_eigenvalues(
              local_dual_structure_hamiltonian,
              local_dual_structure_metric,
              1.0e-8);
      const std::vector<double> full_dual_roots =
          solve_real_generalized_eigenvalues(
              full_dual_structure_hamiltonian,
              full_dual_structure_metric,
              1.0e-8);

      const RootDifferenceStats naive_root_difference =
          compute_root_difference_stats(
              nonorth_result.electronic_state_energies,
              naive_roots);
      const RootDifferenceStats local_dual_root_difference =
          compute_root_difference_stats(
              nonorth_result.electronic_state_energies,
              local_dual_roots);
      const RootDifferenceStats full_dual_root_difference =
          compute_root_difference_stats(
              nonorth_result.electronic_state_energies,
              full_dual_roots);

      worst_naive_root_diff = std::max(
          worst_naive_root_diff,
          naive_root_difference.max_abs);
      worst_local_dual_root_diff = std::max(
          worst_local_dual_root_diff,
          local_dual_root_difference.max_abs);
      worst_full_dual_root_diff = std::max(
          worst_full_dual_root_diff,
          full_dual_root_difference.max_abs);
      worst_naive_overlap_diff = std::max(
          worst_naive_overlap_diff,
          naive_overlap_difference.max_abs);
      worst_local_dual_overlap_diff = std::max(
          worst_local_dual_overlap_diff,
          local_dual_overlap_difference.max_abs);
      worst_full_dual_overlap_diff = std::max(
          worst_full_dual_overlap_diff,
          full_dual_overlap_difference.max_abs);
      worst_naive_hamiltonian_diff = std::max(
          worst_naive_hamiltonian_diff,
          naive_hamiltonian_difference.max_abs);
      worst_local_dual_hamiltonian_diff = std::max(
          worst_local_dual_hamiltonian_diff,
          local_dual_hamiltonian_difference.max_abs);
      worst_full_dual_hamiltonian_diff = std::max(
          worst_full_dual_hamiltonian_diff,
          full_dual_hamiltonian_difference.max_abs);

      std::cout << "subset[" << subset_index << "] = "
                << format_structure_index_list(selected_structure_indices) << '\n';
      std::cout << "subset_n_structures = "
                << subspace_structure_data.n_structures << '\n';
      std::cout << "subset_n_local_determinants = "
                << local_n_determinants << '\n';
      std::cout << "naive_structure_overlap_diff_max_abs = "
                << naive_overlap_difference.max_abs << '\n';
      std::cout << "local_dual_structure_overlap_diff_max_abs = "
                << local_dual_overlap_difference.max_abs << '\n';
      std::cout << "full_dual_structure_overlap_diff_max_abs = "
                << full_dual_overlap_difference.max_abs << '\n';
      std::cout << "naive_structure_hamiltonian_diff_max_abs = "
                << naive_hamiltonian_difference.max_abs << '\n';
      std::cout << "local_dual_structure_hamiltonian_diff_max_abs = "
                << local_dual_hamiltonian_difference.max_abs << '\n';
      std::cout << "full_dual_structure_hamiltonian_diff_max_abs = "
                << full_dual_hamiltonian_difference.max_abs << '\n';
      std::cout << "naive_structure_hamiltonian_symmetry_max_abs = "
                << compute_max_abs_symmetry_residual(naive_structure_hamiltonian) << '\n';
      std::cout << "local_dual_structure_hamiltonian_symmetry_max_abs = "
                << compute_max_abs_symmetry_residual(local_dual_structure_hamiltonian) << '\n';
      std::cout << "full_dual_structure_hamiltonian_symmetry_max_abs = "
                << compute_max_abs_symmetry_residual(full_dual_structure_hamiltonian) << '\n';
      std::cout << "naive_structure_root_max_abs = "
                << naive_root_difference.max_abs << '\n';
      std::cout << "local_dual_structure_root_max_abs = "
                << local_dual_root_difference.max_abs << '\n';
      std::cout << "full_dual_structure_root_max_abs = "
                << full_dual_root_difference.max_abs << '\n';
      print_root_comparison(
          nonorth_result.electronic_state_energies,
          naive_roots,
          options.print_roots,
          "naive_structure");
      print_root_comparison(
          nonorth_result.electronic_state_energies,
          local_dual_roots,
          options.print_roots,
          "local_dual_structure");
      print_root_comparison(
          nonorth_result.electronic_state_energies,
          full_dual_roots,
          options.print_roots,
          "full_dual_structure");
    }

    std::cout << "worst_naive_structure_root_max_abs = "
              << worst_naive_root_diff << '\n';
    std::cout << "worst_local_dual_structure_root_max_abs = "
              << worst_local_dual_root_diff << '\n';
    std::cout << "worst_full_dual_structure_root_max_abs = "
              << worst_full_dual_root_diff << '\n';
    std::cout << "worst_naive_structure_overlap_matrix_max_abs = "
              << worst_naive_overlap_diff << '\n';
    std::cout << "worst_local_dual_structure_overlap_matrix_max_abs = "
              << worst_local_dual_overlap_diff << '\n';
    std::cout << "worst_full_dual_structure_overlap_matrix_max_abs = "
              << worst_full_dual_overlap_diff << '\n';
    std::cout << "worst_naive_structure_hamiltonian_matrix_max_abs = "
              << worst_naive_hamiltonian_diff << '\n';
    std::cout << "worst_local_dual_structure_hamiltonian_matrix_max_abs = "
              << worst_local_dual_hamiltonian_diff << '\n';
    std::cout << "worst_full_dual_structure_hamiltonian_matrix_max_abs = "
              << worst_full_dual_hamiltonian_diff << '\n';
  } catch (const std::exception& error) {
    std::cerr << "check_biorthogonal_dual_structure_projection failed: "
              << error.what() << '\n';
    return 1;
  }

  return 0;
}
