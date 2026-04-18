#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_determinant_hamiltonian.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_forward_evaluator.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_orbital_integrals.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_prepared_input.hpp"
#include "vb/matrices/full_structure_builder.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"
#include "vb/scf/cpp_vb_scf_evaluator.hpp"

namespace {

using DenseMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

struct Options {
  std::string input_path;
  int print_roots = 6;
  int determinant_threshold = 128;
};

void print_usage() {
  std::cerr << "usage: compare_biorthogonal_nonorth_forward <input.xmi>"
               " [--print-roots N]"
               " [--determinant-threshold N]\n";
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
    } else if (argument_name == "--determinant-threshold") {
      options.determinant_threshold = std::stoi(argument_value);
    } else {
      throw std::invalid_argument("unknown argument: " + argument_name);
    }
  }

  if (options.print_roots <= 0) {
    throw std::invalid_argument("--print-roots must be positive");
  }
  if (options.determinant_threshold <= 0) {
    throw std::invalid_argument("--determinant-threshold must be positive");
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

struct MatrixDifferenceStats {
  double frobenius_norm = 0.0;
  double max_abs = 0.0;
};

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
          "non-negligible imaginary eigenvalue encountered in determinant-level biorthogonal matrix");
    }
    eigenvalues.push_back(eigenvalue.real());
  }
  std::sort(eigenvalues.begin(), eigenvalues.end());
  return eigenvalues;
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
    const auto& structure_data = load_result.input.structure_data;

    std::cout << "input = " << options.input_path << '\n';
    std::cout << "n_structures = " << structure_data.n_structures << '\n';
    std::cout << "n_determinants = " << structure_data.alpha_det.size() << '\n';
    std::cout << "raw_structure_data_n_active_orbitals = "
              << structure_data.n_active_orbitals << '\n';

    const xmvb::vb::CppVbScfEvaluator nonorth_evaluator;
    const auto prepared_input =
        xmvb::vb::biorthogonal_vbscf::prepare_biorthogonal_input(
            load_result.input);
    const xmvb::vb::FullDeterminantStructureData& complete_structure_data =
        prepared_input.structure_data;
    std::cout << "prepared_n_active_orbitals = "
              << complete_structure_data.n_active_orbitals << '\n';
    const xmvb::vb::CppVbScfResult nonorth_result =
        nonorth_evaluator.evaluate(
            load_result.input,
            load_result.nuclear_repulsion_energy);
    const auto biorth_result =
        xmvb::vb::biorthogonal_vbscf::evaluate_biorthogonal_forward(
            complete_structure_data);

    const DenseMatrix original_structure_overlap =
        map_square_matrix(
            nonorth_result.structure_matrices.overlap_matrix,
            structure_data.n_structures);
    const DenseMatrix original_structure_hamiltonian =
        map_square_matrix(
            nonorth_result.structure_matrices.hamiltonian_matrix,
            structure_data.n_structures);
    const DenseMatrix biorth_structure_metric =
        biorth_result.structure_space.fixed_metric;
    const DenseMatrix biorth_structure_hamiltonian =
        biorth_result.structure_hamiltonian_build_result.selected_structure_hamiltonian;

    const MatrixDifferenceStats structure_overlap_difference =
        compute_difference_stats(
            original_structure_overlap,
            biorth_structure_metric);
    const MatrixDifferenceStats structure_hamiltonian_difference =
        compute_difference_stats(
            original_structure_hamiltonian,
            biorth_structure_hamiltonian);

    std::cout << "n_unique_alpha = "
              << biorth_result.structure_hamiltonian_build_result.n_unique_alpha << '\n';
    std::cout << "n_unique_beta = "
              << biorth_result.structure_hamiltonian_build_result.n_unique_beta << '\n';
    std::cout << "structure_overlap_diff_fro = "
              << structure_overlap_difference.frobenius_norm << '\n';
    std::cout << "structure_overlap_diff_max_abs = "
              << structure_overlap_difference.max_abs << '\n';
    std::cout << "structure_hamiltonian_diff_fro = "
              << structure_hamiltonian_difference.frobenius_norm << '\n';
    std::cout << "structure_hamiltonian_diff_max_abs = "
              << structure_hamiltonian_difference.max_abs << '\n';

    std::cout << "original_one_electron_reference_energy = "
              << nonorth_result.one_electron_reference_energy << '\n';
    print_root_comparison(
        nonorth_result.electronic_state_energies,
        biorth_result.solve_result.eigenvalues,
        options.print_roots,
        "structure_electronic");

    const int n_total_roots = std::min(
        static_cast<int>(nonorth_result.electronic_state_energies.size()),
        static_cast<int>(biorth_result.solve_result.eigenvalues.size()));
    for (int root_index = 0;
         root_index < std::min(options.print_roots, n_total_roots);
         ++root_index) {
      const double original_total =
          nonorth_result.one_electron_reference_energy +
          nonorth_result.electronic_state_energies[xmvb::to_size(root_index)] +
          load_result.nuclear_repulsion_energy;
      const double biorth_total =
          nonorth_result.one_electron_reference_energy +
          biorth_result.solve_result.eigenvalues[xmvb::to_size(root_index)] +
          load_result.nuclear_repulsion_energy;
      std::cout << "structure_total_root[" << root_index << "]"
                << " original=" << original_total
                << " biorth=" << biorth_total
                << " diff=" << (biorth_total - original_total) << '\n';
    }

    const int n_determinants = static_cast<int>(structure_data.alpha_det.size());
    if (n_determinants <= options.determinant_threshold) {
      const xmvb::vb::FullDeterminantStructureHamiltonianOverlapBuilder
          determinant_builder;
      const auto determinant_build_result =
          determinant_builder.build_with_pair_evaluations(complete_structure_data);
      DenseMatrix determinant_overlap =
          DenseMatrix::Zero(n_determinants, n_determinants);
      DenseMatrix determinant_hamiltonian =
          DenseMatrix::Zero(n_determinants, n_determinants);
      for (int column_index = 0; column_index < n_determinants; ++column_index) {
        for (int row_index = 0; row_index < n_determinants; ++row_index) {
          const auto& pair_result =
              determinant_build_result.pair_evaluation(row_index, column_index);
          determinant_overlap(row_index, column_index) =
              pair_result.overlap_determinant;
          determinant_hamiltonian(row_index, column_index) =
              pair_result.total_hamiltonian;
        }
      }

      const auto determinant_orbital_integrals =
          xmvb::vb::biorthogonal_vbscf::build_biorthogonal_orbital_integrals(
              complete_structure_data.n_active_orbitals,
              complete_structure_data.ovlp_act,
              complete_structure_data.h1e_act);
      const DenseMatrix determinant_biorth_hamiltonian =
          xmvb::vb::biorthogonal_vbscf::build_biorthogonal_determinant_hamiltonian_matrix(
              biorth_result.structure_expansion.determinants,
              determinant_orbital_integrals,
              xmvb::vb::make_active_space_two_electron_view(
                  complete_structure_data.eri_act));

      const MatrixDifferenceStats determinant_relation_difference =
          compute_difference_stats(
              determinant_hamiltonian,
              determinant_overlap * determinant_biorth_hamiltonian);
      std::cout << "determinant_relation_diff_fro = "
                << determinant_relation_difference.frobenius_norm << '\n';
      std::cout << "determinant_relation_diff_max_abs = "
                << determinant_relation_difference.max_abs << '\n';

      xmvb::core::GeneralizedEigensolver generalized_eigensolver;
      const auto original_determinant_eigen_result =
          generalized_eigensolver.solve(
              dense_matrix_to_vector(determinant_hamiltonian),
              dense_matrix_to_vector(determinant_overlap),
              n_determinants);
      const auto biorth_determinant_eigenvalues =
          solve_real_nonsymmetric_eigenvalues(
              determinant_biorth_hamiltonian,
              1.0e-8);
      print_root_comparison(
          original_determinant_eigen_result.eigenvalues,
          biorth_determinant_eigenvalues,
          options.print_roots,
          "determinant_electronic");
    } else {
      std::cout << "determinant_level_skipped = true\n";
      std::cout << "determinant_level_threshold = "
                << options.determinant_threshold << '\n';
    }
  } catch (const std::exception& error) {
    std::cerr << "compare_biorthogonal_nonorth_forward failed: "
              << error.what() << '\n';
    return 1;
  }

  return 0;
}
