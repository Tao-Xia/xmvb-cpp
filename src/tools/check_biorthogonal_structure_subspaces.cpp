#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
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
#include "vb/matrices/structure_subspace_builder.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"
#include "vb/scf/cpp_vb_scf_evaluator.hpp"

namespace {

using DenseMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

struct Options {
  std::string input_path;
  int print_roots = 6;
  int determinant_threshold = 128;
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
  std::cerr << "usage: check_biorthogonal_structure_subspaces <input.xmi>"
               " [--print-roots N]"
               " [--determinant-threshold N]"
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
    } else if (argument_name == "--determinant-threshold") {
      options.determinant_threshold = std::stoi(argument_value);
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
  if (options.determinant_threshold <= 0) {
    throw std::invalid_argument("--determinant-threshold must be positive");
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

DenseMatrix gather_submatrix(
    const DenseMatrix& matrix,
    const std::vector<int>& row_indices,
    const std::vector<int>& column_indices) {
  DenseMatrix submatrix(
      static_cast<int>(row_indices.size()),
      static_cast<int>(column_indices.size()));
  for (int column_index = 0;
       column_index < static_cast<int>(column_indices.size());
       ++column_index) {
    for (int row_index = 0;
         row_index < static_cast<int>(row_indices.size());
         ++row_index) {
      submatrix(row_index, column_index) =
          matrix(
              row_indices[xmvb::to_size(row_index)],
              column_indices[xmvb::to_size(column_index)]);
    }
  }
  return submatrix;
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
          "non-negligible imaginary eigenvalue encountered in biorthogonal matrix");
    }
    eigenvalues.push_back(eigenvalue.real());
  }
  std::sort(eigenvalues.begin(), eigenvalues.end());
  return eigenvalues;
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

std::vector<int> build_kept_determinant_indices(
    const xmvb::vb::FullDeterminantStructureData& full_structure_data,
    const std::vector<int>& selected_structure_indices) {
  std::vector<bool> keep_structure(
      xmvb::to_size(full_structure_data.n_structures),
      false);
  for (const int structure_index : selected_structure_indices) {
    keep_structure[xmvb::to_size(structure_index)] = true;
  }

  std::vector<int> kept_determinant_indices;
  for (int determinant_index = 0;
       determinant_index < static_cast<int>(full_structure_data.determinant_to_structure_terms.size());
       ++determinant_index) {
    bool keep_determinant = false;
    for (const auto& term :
         full_structure_data.determinant_to_structure_terms[xmvb::to_size(determinant_index)]) {
      if (keep_structure[xmvb::to_size(term.structure_index)]) {
        keep_determinant = true;
        break;
      }
    }
    if (keep_determinant) {
      kept_determinant_indices.push_back(determinant_index);
    }
  }
  return kept_determinant_indices;
}

std::vector<int> build_omitted_determinant_indices(
    int n_determinants,
    const std::vector<int>& kept_determinant_indices) {
  std::vector<bool> is_kept(xmvb::to_size(n_determinants), false);
  for (const int determinant_index : kept_determinant_indices) {
    is_kept[xmvb::to_size(determinant_index)] = true;
  }

  std::vector<int> omitted_determinant_indices;
  for (int determinant_index = 0; determinant_index < n_determinants; ++determinant_index) {
    if (!is_kept[xmvb::to_size(determinant_index)]) {
      omitted_determinant_indices.push_back(determinant_index);
    }
  }
  return omitted_determinant_indices;
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
        full_determinant_builder;
    const auto full_determinant_build_result =
        full_determinant_builder.build_with_pair_evaluations(complete_structure_data);
    const int full_n_determinants =
        static_cast<int>(complete_structure_data.alpha_det.size());
    DenseMatrix full_determinant_overlap =
        DenseMatrix::Zero(full_n_determinants, full_n_determinants);
    DenseMatrix full_determinant_hamiltonian =
        DenseMatrix::Zero(full_n_determinants, full_n_determinants);
    for (int column_index = 0; column_index < full_n_determinants; ++column_index) {
      for (int row_index = 0; row_index < full_n_determinants; ++row_index) {
        const auto& pair_result =
            full_determinant_build_result.pair_evaluation(row_index, column_index);
        full_determinant_overlap(row_index, column_index) =
            pair_result.overlap_determinant;
        full_determinant_hamiltonian(row_index, column_index) =
            pair_result.total_hamiltonian;
      }
    }
    const auto full_determinant_orbital_integrals =
        xmvb::vb::biorthogonal_vbscf::build_biorthogonal_orbital_integrals(
            complete_structure_data.n_active_orbitals,
            complete_structure_data.ovlp_act,
            complete_structure_data.h1e_act);
    const DenseMatrix full_determinant_biorth_hamiltonian =
        xmvb::vb::biorthogonal_vbscf::build_biorthogonal_determinant_hamiltonian_matrix(
            xmvb::vb::biorthogonal_vbscf::build_biorthogonal_structure_expansion(
                complete_structure_data)
                .determinants,
            full_determinant_orbital_integrals,
            xmvb::vb::make_active_space_two_electron_view(
                complete_structure_data.eri_act));
    const auto structure_subspaces = enumerate_structure_subspaces(
        complete_structure_data.n_structures,
        options.min_subspace_size,
        std::min(options.max_subspace_size, complete_structure_data.n_structures));

    std::cout << "input = " << options.input_path << '\n';
    std::cout << "full_n_structures = " << complete_structure_data.n_structures << '\n';
    std::cout << "full_n_determinants = " << complete_structure_data.alpha_det.size() << '\n';
    std::cout << "prepared_n_active_orbitals = "
              << complete_structure_data.n_active_orbitals << '\n';
    std::cout << "enumerated_subspace_count = " << structure_subspaces.size() << '\n';

    double worst_structure_root_diff = 0.0;
    double worst_total_root_diff = 0.0;
    double worst_determinant_relation_diff = 0.0;
    double worst_omitted_coupling_diff = 0.0;
    double worst_omitted_coupling_reconstruction_diff = 0.0;
    double worst_structure_overlap_matrix_diff = 0.0;
    double worst_structure_hamiltonian_matrix_diff = 0.0;

    for (std::size_t subset_index = 0;
         subset_index < structure_subspaces.size();
         ++subset_index) {
      const auto& selected_structure_indices =
          structure_subspaces[subset_index];
      const xmvb::vb::CppVbScfResult nonorth_result =
          nonorth_evaluator.evaluate_subspace(
              load_result.input,
              selected_structure_indices,
              load_result.nuclear_repulsion_energy);
      const xmvb::vb::FullDeterminantStructureData subspace_structure_data =
          subspace_builder.build(
              complete_structure_data,
              selected_structure_indices);
      const auto biorth_result =
          xmvb::vb::biorthogonal_vbscf::evaluate_biorthogonal_forward(
              subspace_structure_data);

      const DenseMatrix original_structure_overlap =
          map_square_matrix(
              nonorth_result.structure_matrices.overlap_matrix,
              subspace_structure_data.n_structures);
      const DenseMatrix original_structure_hamiltonian =
          map_square_matrix(
              nonorth_result.structure_matrices.hamiltonian_matrix,
              subspace_structure_data.n_structures);
      const MatrixDifferenceStats structure_overlap_difference =
          compute_difference_stats(
              original_structure_overlap,
              biorth_result.structure_space.fixed_metric);
      const MatrixDifferenceStats structure_hamiltonian_difference =
          compute_difference_stats(
              original_structure_hamiltonian,
              biorth_result.structure_hamiltonian_build_result.selected_structure_hamiltonian);
      const RootDifferenceStats structure_root_difference =
          compute_root_difference_stats(
              nonorth_result.electronic_state_energies,
              biorth_result.solve_result.eigenvalues);

      worst_structure_root_diff = std::max(
          worst_structure_root_diff,
          structure_root_difference.max_abs);
      worst_structure_overlap_matrix_diff = std::max(
          worst_structure_overlap_matrix_diff,
          structure_overlap_difference.max_abs);
      worst_structure_hamiltonian_matrix_diff = std::max(
          worst_structure_hamiltonian_matrix_diff,
          structure_hamiltonian_difference.max_abs);

      const int n_total_roots = std::min(
          static_cast<int>(nonorth_result.electronic_state_energies.size()),
          static_cast<int>(biorth_result.solve_result.eigenvalues.size()));
      double subset_worst_total_root_diff = 0.0;
      for (int root_index = 0; root_index < n_total_roots; ++root_index) {
        const double original_total =
            nonorth_result.one_electron_reference_energy +
            nonorth_result.electronic_state_energies[xmvb::to_size(root_index)] +
            load_result.nuclear_repulsion_energy;
        const double biorth_total =
            nonorth_result.one_electron_reference_energy +
            biorth_result.solve_result.eigenvalues[xmvb::to_size(root_index)] +
            load_result.nuclear_repulsion_energy;
        subset_worst_total_root_diff = std::max(
            subset_worst_total_root_diff,
            std::abs(biorth_total - original_total));
      }
      worst_total_root_diff = std::max(
          worst_total_root_diff,
          subset_worst_total_root_diff);

      MatrixDifferenceStats determinant_relation_difference;
      MatrixDifferenceStats determinant_omitted_coupling_reconstruction_difference;
      double determinant_omitted_coupling_max_abs = 0.0;
      bool determinant_level_checked =
          static_cast<int>(subspace_structure_data.alpha_det.size()) <=
          options.determinant_threshold;
      if (determinant_level_checked) {
        // This determinant-level check distinguishes a true implementation bug
        // from a selected-space projection effect. If
        // `H_det = S_det * H_det^(bi)` still holds in the filtered determinant
        // list but the selected-space roots differ, the discrepancy must come
        // from the structure-level projection rather than from the dual
        // determinant algebra itself.
        const xmvb::vb::FullDeterminantStructureHamiltonianOverlapBuilder
            determinant_builder;
        const auto determinant_build_result =
            determinant_builder.build_with_pair_evaluations(subspace_structure_data);
        const int n_determinants =
            static_cast<int>(subspace_structure_data.alpha_det.size());
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
                subspace_structure_data.n_active_orbitals,
                subspace_structure_data.ovlp_act,
                subspace_structure_data.h1e_act);
        const DenseMatrix determinant_biorth_hamiltonian =
            xmvb::vb::biorthogonal_vbscf::build_biorthogonal_determinant_hamiltonian_matrix(
                biorth_result.structure_expansion.determinants,
                determinant_orbital_integrals,
                xmvb::vb::make_active_space_two_electron_view(
                    subspace_structure_data.eri_act));
        determinant_relation_difference =
            compute_difference_stats(
                determinant_hamiltonian,
                determinant_overlap * determinant_biorth_hamiltonian);
        worst_determinant_relation_diff = std::max(
            worst_determinant_relation_diff,
            determinant_relation_difference.max_abs);

        const std::vector<int> kept_determinant_indices =
            build_kept_determinant_indices(
                complete_structure_data,
                selected_structure_indices);
        const std::vector<int> omitted_determinant_indices =
            build_omitted_determinant_indices(
                full_n_determinants,
                kept_determinant_indices);
        const DenseMatrix full_overlap_kk =
            gather_submatrix(
                full_determinant_overlap,
                kept_determinant_indices,
                kept_determinant_indices);
        const DenseMatrix full_hamiltonian_kk =
            gather_submatrix(
                full_determinant_hamiltonian,
                kept_determinant_indices,
                kept_determinant_indices);
        const DenseMatrix full_biorth_hamiltonian_kk =
            gather_submatrix(
                full_determinant_biorth_hamiltonian,
                kept_determinant_indices,
                kept_determinant_indices);
        DenseMatrix omitted_coupling =
            DenseMatrix::Zero(
                static_cast<int>(kept_determinant_indices.size()),
                static_cast<int>(kept_determinant_indices.size()));
        if (!omitted_determinant_indices.empty()) {
          const DenseMatrix full_overlap_ko =
              gather_submatrix(
                  full_determinant_overlap,
                  kept_determinant_indices,
                  omitted_determinant_indices);
          const DenseMatrix full_biorth_hamiltonian_ok =
              gather_submatrix(
                  full_determinant_biorth_hamiltonian,
                  omitted_determinant_indices,
                  kept_determinant_indices);
          omitted_coupling = full_overlap_ko * full_biorth_hamiltonian_ok;
        }
        determinant_omitted_coupling_max_abs =
            omitted_coupling.size() > 0
            ? omitted_coupling.cwiseAbs().maxCoeff()
            : 0.0;
        const DenseMatrix determinant_relation_residual =
            full_hamiltonian_kk -
            full_overlap_kk * full_biorth_hamiltonian_kk;
        determinant_omitted_coupling_reconstruction_difference =
            compute_difference_stats(
                determinant_relation_residual,
                omitted_coupling);
        worst_omitted_coupling_diff = std::max(
            worst_omitted_coupling_diff,
            determinant_omitted_coupling_max_abs);
        worst_omitted_coupling_reconstruction_diff = std::max(
            worst_omitted_coupling_reconstruction_diff,
            determinant_omitted_coupling_reconstruction_difference.max_abs);
      }

      std::cout << "subset[" << subset_index << "] = "
                << format_structure_index_list(selected_structure_indices) << '\n';
      std::cout << "subset_n_structures = "
                << subspace_structure_data.n_structures << '\n';
      std::cout << "subset_n_determinants = "
                << subspace_structure_data.alpha_det.size() << '\n';
      std::cout << "subset_structure_overlap_diff_fro = "
                << structure_overlap_difference.frobenius_norm << '\n';
      std::cout << "subset_structure_overlap_diff_max_abs = "
                << structure_overlap_difference.max_abs << '\n';
      std::cout << "subset_structure_hamiltonian_diff_fro = "
                << structure_hamiltonian_difference.frobenius_norm << '\n';
      std::cout << "subset_structure_hamiltonian_diff_max_abs = "
                << structure_hamiltonian_difference.max_abs << '\n';
      std::cout << "subset_structure_root_max_abs = "
                << structure_root_difference.max_abs << '\n';
      std::cout << "subset_total_root_max_abs = "
                << subset_worst_total_root_diff << '\n';
      if (determinant_level_checked) {
        std::cout << "subset_determinant_relation_diff_fro = "
                  << determinant_relation_difference.frobenius_norm << '\n';
        std::cout << "subset_determinant_relation_diff_max_abs = "
                  << determinant_relation_difference.max_abs << '\n';
        std::cout << "subset_omitted_determinant_coupling_max_abs = "
                  << determinant_omitted_coupling_max_abs << '\n';
        std::cout << "subset_omitted_determinant_coupling_reconstruction_max_abs = "
                  << determinant_omitted_coupling_reconstruction_difference.max_abs << '\n';
      } else {
        std::cout << "subset_determinant_level_skipped = true\n";
      }
      print_root_comparison(
          nonorth_result.electronic_state_energies,
          biorth_result.solve_result.eigenvalues,
          options.print_roots,
          "subset_structure_electronic");
    }

    std::cout << "worst_subset_structure_root_max_abs = "
              << worst_structure_root_diff << '\n';
    std::cout << "worst_subset_total_root_max_abs = "
              << worst_total_root_diff << '\n';
    std::cout << "worst_subset_determinant_relation_max_abs = "
              << worst_determinant_relation_diff << '\n';
    std::cout << "worst_subset_omitted_determinant_coupling_max_abs = "
              << worst_omitted_coupling_diff << '\n';
    std::cout << "worst_subset_omitted_determinant_coupling_reconstruction_max_abs = "
              << worst_omitted_coupling_reconstruction_diff << '\n';
    std::cout << "worst_subset_structure_overlap_matrix_max_abs = "
              << worst_structure_overlap_matrix_diff << '\n';
    std::cout << "worst_subset_structure_hamiltonian_matrix_max_abs = "
              << worst_structure_hamiltonian_matrix_diff << '\n';
  } catch (const std::exception& error) {
    std::cerr << "check_biorthogonal_structure_subspaces failed: "
              << error.what() << '\n';
    return 1;
  }

  return 0;
}
