#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include "runtime/cpp_block_guess_builder.hpp"
#include "runtime/cpp_vb_input_loader.hpp"
#include "runtime_c/cpp_runtime_extractor.h"

namespace {

struct Options {
  std::string input_path;
};

struct GuessMetrics {
  double sign_aligned_orbital_max_abs_diff = 0.0;
  double block_projector_max_abs_diff = 0.0;
  double block_subspace_min_singular_value = 1.0;
};

void print_usage() {
  std::cerr << "usage: compare_legacy_block_guess <input.xmi>\n";
}

Options parse_arguments(int argc, char** argv) {
  if (argc != 2) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }
  return {.input_path = argv[1]};
}

std::vector<double> expand_sparse_orbital(
    const xmvb::vb::OrbitalPreparationInput& orbital_preparation_input,
    const std::vector<double>& orbital_value_table,
    int orbital_index) {
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  std::vector<double> dense_orbital(xmvb::to_size(n_basis_functions), 0.0);
  const int basis_count =
      xmvb::vb::get_orbital_basis_count(orbital_preparation_input, orbital_index);
  for (int coefficient_index = 0; coefficient_index < basis_count; ++coefficient_index) {
    const int basis_function_index =
        orbital_preparation_input.orbital_basis_index_table
            [xmvb::to_size(orbital_index) * n_basis_functions + coefficient_index] -
        1;
    if (basis_function_index < 0) {
      break;
    }
    dense_orbital[xmvb::to_size(basis_function_index)] =
        orbital_value_table[xmvb::to_size(orbital_index) * n_basis_functions +
                            coefficient_index];
  }
  return dense_orbital;
}

double dot_product(const std::vector<double>& left, const std::vector<double>& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("dot_product requires equal-sized vectors");
  }
  double result = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    result += left[index] * right[index];
  }
  return result;
}

double max_abs_difference(
    const std::vector<double>& left,
    const std::vector<double>& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("max_abs_difference requires equal-sized vectors");
  }
  double result = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    result = std::max(result, std::abs(left[index] - right[index]));
  }
  return result;
}

std::vector<double> build_block_projector(
    const xmvb::vb::OrbitalPreparationInput& orbital_preparation_input,
    const std::vector<double>& orbital_value_table,
    const std::vector<int>& block_orbitals) {
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  std::vector<double> projector(
      xmvb::to_size(n_basis_functions) * n_basis_functions,
      0.0);
  for (int orbital_index : block_orbitals) {
    const auto dense_orbital = expand_sparse_orbital(
        orbital_preparation_input,
        orbital_value_table,
        orbital_index);
    for (int column = 0; column < n_basis_functions; ++column) {
      const double column_value = dense_orbital[xmvb::to_size(column)];
      for (int row = 0; row < n_basis_functions; ++row) {
        projector[xmvb::to_size(column) * n_basis_functions + row] +=
            dense_orbital[xmvb::to_size(row)] * column_value;
      }
    }
  }
  return projector;
}

Eigen::MatrixXd build_dense_block_matrix(
    const xmvb::vb::OrbitalPreparationInput& orbital_preparation_input,
    const std::vector<double>& orbital_value_table,
    const std::vector<int>& block_orbitals) {
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  Eigen::MatrixXd block_matrix =
      Eigen::MatrixXd::Zero(n_basis_functions, static_cast<int>(block_orbitals.size()));
  for (std::size_t column = 0; column < block_orbitals.size(); ++column) {
    const auto dense_orbital = expand_sparse_orbital(
        orbital_preparation_input,
        orbital_value_table,
        block_orbitals[column]);
    for (int row = 0; row < n_basis_functions; ++row) {
      block_matrix(row, static_cast<int>(column)) = dense_orbital[xmvb::to_size(row)];
    }
  }
  return block_matrix;
}

Eigen::MatrixXd metric_orthonormalize(
    const Eigen::MatrixXd& coefficient_matrix,
    const Eigen::MatrixXd& overlap_matrix) {
  Eigen::MatrixXd gram_matrix =
      coefficient_matrix.transpose() * overlap_matrix * coefficient_matrix;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(gram_matrix);
  if (eigensolver.info() != Eigen::Success) {
    throw std::runtime_error("failed to diagonalize block Gram matrix");
  }

  Eigen::VectorXd inverse_sqrt = eigensolver.eigenvalues();
  for (int index = 0; index < inverse_sqrt.size(); ++index) {
    if (!(inverse_sqrt(index) > 1.0e-12)) {
      throw std::runtime_error("encountered rank-deficient block Gram matrix");
    }
    inverse_sqrt(index) = 1.0 / std::sqrt(inverse_sqrt(index));
  }

  return coefficient_matrix *
         eigensolver.eigenvectors() *
         inverse_sqrt.asDiagonal() *
         eigensolver.eigenvectors().transpose();
}

GuessMetrics evaluate_guess_metrics(
    const xmvb::vb::OrbitalPreparationInput& orbital_preparation_input,
    const std::vector<double>& legacy_orbital_value_table,
    const std::vector<double>& rebuilt_orbital_value_table,
    const std::vector<std::vector<int>>& blocks,
    const Eigen::MatrixXd& overlap_matrix) {
  GuessMetrics metrics;

  for (int orbital_index = 0; orbital_index < orbital_preparation_input.n_orbitals; ++orbital_index) {
    const auto legacy_dense_orbital = expand_sparse_orbital(
        orbital_preparation_input,
        legacy_orbital_value_table,
        orbital_index);
    auto rebuilt_dense_orbital = expand_sparse_orbital(
        orbital_preparation_input,
        rebuilt_orbital_value_table,
        orbital_index);
    if (dot_product(legacy_dense_orbital, rebuilt_dense_orbital) < 0.0) {
      for (double& value : rebuilt_dense_orbital) {
        value = -value;
      }
    }
    metrics.sign_aligned_orbital_max_abs_diff = std::max(
        metrics.sign_aligned_orbital_max_abs_diff,
        max_abs_difference(legacy_dense_orbital, rebuilt_dense_orbital));
  }

  for (const auto& block : blocks) {
    const auto legacy_projector = build_block_projector(
        orbital_preparation_input,
        legacy_orbital_value_table,
        block);
    const auto rebuilt_projector = build_block_projector(
        orbital_preparation_input,
        rebuilt_orbital_value_table,
        block);
    metrics.block_projector_max_abs_diff = std::max(
        metrics.block_projector_max_abs_diff,
        max_abs_difference(legacy_projector, rebuilt_projector));

    const Eigen::MatrixXd legacy_block =
        build_dense_block_matrix(
            orbital_preparation_input,
            legacy_orbital_value_table,
            block);
    const Eigen::MatrixXd rebuilt_block =
        build_dense_block_matrix(
            orbital_preparation_input,
            rebuilt_orbital_value_table,
            block);
    const Eigen::MatrixXd legacy_orthonormal =
        metric_orthonormalize(legacy_block, overlap_matrix);
    const Eigen::MatrixXd rebuilt_orthonormal =
        metric_orthonormalize(rebuilt_block, overlap_matrix);
    const Eigen::MatrixXd subspace_overlap =
        legacy_orthonormal.transpose() * overlap_matrix * rebuilt_orthonormal;
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(subspace_overlap);
    if (svd.singularValues().size() > 0) {
      metrics.block_subspace_min_singular_value = std::min(
          metrics.block_subspace_min_singular_value,
          svd.singularValues().minCoeff());
    }
  }

  return metrics;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);

    CppRuntimeSnapshot snapshot{};
    init_cpp_runtime_snapshot(&snapshot);
    char error_message[1024] = {0};
    if (extract_cpp_runtime_snapshot(
            options.input_path.c_str(),
            &snapshot,
            nullptr,
            error_message,
            sizeof(error_message)) != 0) {
      throw std::runtime_error(
          error_message[0] != '\0' ? error_message : "legacy runtime extraction failed");
    }

    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.ao_integral_source = xmvb::vb::AoIntegralSource::LibcintMaterializedCpp;
    load_options.orbital_guess_source = xmvb::vb::OrbitalGuessSource::LegacyRuntime;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(options.input_path, load_options);

    const auto& orbital_preparation_input = load_result.input.orbital_preparation_input;
    const auto& legacy_orbital_value_table =
        orbital_preparation_input.orbital_value_table.vector();
    const std::vector<double> legacy_hf_fock_matrix(
        snapshot.hf_fock_matrix,
        snapshot.hf_fock_matrix +
            xmvb::to_size(snapshot.n_basis_functions) * snapshot.n_basis_functions);
    const std::vector<double> legacy_hf_overlap_matrix(
        snapshot.hf_overlap_matrix,
        snapshot.hf_overlap_matrix +
            xmvb::to_size(snapshot.n_basis_functions) * snapshot.n_basis_functions);
    const auto rebuilt_with_vb_overlap = xmvb::vb::build_block_matrix_guess(
        load_result.input.libcint_input,
        legacy_hf_fock_matrix,
        orbital_preparation_input);
    const auto rebuilt_with_hf_overlap = xmvb::vb::build_block_matrix_guess(
        load_result.input.libcint_input,
        legacy_hf_fock_matrix,
        legacy_hf_overlap_matrix,
        orbital_preparation_input);

    const auto blocks = xmvb::vb::detect_orbital_blocks(orbital_preparation_input);
    const int n_basis_functions = orbital_preparation_input.n_basis_functions;
    Eigen::MatrixXd vb_overlap_matrix =
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            orbital_preparation_input.active_orbital_overlap_matrix.data(),
            n_basis_functions,
            n_basis_functions);
    Eigen::MatrixXd hf_overlap_matrix =
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            legacy_hf_overlap_matrix.data(),
            n_basis_functions,
            n_basis_functions);
    const GuessMetrics vb_overlap_metrics = evaluate_guess_metrics(
        orbital_preparation_input,
        legacy_orbital_value_table,
        rebuilt_with_vb_overlap,
        blocks,
        vb_overlap_matrix);
    const GuessMetrics hf_overlap_metrics = evaluate_guess_metrics(
        orbital_preparation_input,
        legacy_orbital_value_table,
        rebuilt_with_hf_overlap,
        blocks,
        hf_overlap_matrix);

    std::cout << std::setprecision(12);
    std::cout << "n_basis_functions = " << snapshot.n_basis_functions << '\n';
    std::cout << "n_orbitals = " << orbital_preparation_input.n_orbitals << '\n';
    std::cout << "n_blocks = " << blocks.size() << '\n';
    std::cout << "vb_overlap_sign_aligned_orbital_max_abs_diff = "
              << vb_overlap_metrics.sign_aligned_orbital_max_abs_diff << '\n';
    std::cout << "vb_overlap_block_projector_max_abs_diff = "
              << vb_overlap_metrics.block_projector_max_abs_diff << '\n';
    std::cout << "vb_overlap_block_subspace_min_singular_value = "
              << vb_overlap_metrics.block_subspace_min_singular_value << '\n';
    std::cout << "hf_overlap_sign_aligned_orbital_max_abs_diff = "
              << hf_overlap_metrics.sign_aligned_orbital_max_abs_diff << '\n';
    std::cout << "hf_overlap_block_projector_max_abs_diff = "
              << hf_overlap_metrics.block_projector_max_abs_diff << '\n';
    std::cout << "hf_overlap_block_subspace_min_singular_value = "
              << hf_overlap_metrics.block_subspace_min_singular_value << '\n';

    free_cpp_runtime_snapshot(&snapshot);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
