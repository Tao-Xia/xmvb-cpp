#include "vb/orbital/ao_effective_one_electron_ri_operator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cblas.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xmvb::vb {

namespace {

std::size_t packed_pair_count(int n_basis_functions) {
  const std::size_t n = xmvb::to_size(n_basis_functions);
  return n * (n + 1) / 2;
}

struct SpectralFactorization {
  Eigen::MatrixXd scaled_eigenvectors;
  int n_positive_components = 0;
  int n_negative_components = 0;
  int effective_rank = 0;
  bool use_low_rank_path = false;
};

void unpack_packed_factor_row_lower_triangle(
    const Eigen::Ref<const Eigen::MatrixXd>& packed_factor_matrix,
    int auxiliary_index,
    int n_basis_functions,
    Eigen::MatrixXd* factor_matrix) {
  if (factor_matrix == nullptr) {
    throw std::invalid_argument("RI factor-row unpack received null matrix output");
  }

  factor_matrix->resize(n_basis_functions, n_basis_functions);
  std::size_t packed_index = 0;
  for (int column = 0; column < n_basis_functions; ++column) {
    for (int row = 0; row <= column; ++row) {
      const double value =
          packed_factor_matrix(auxiliary_index, static_cast<Eigen::Index>(packed_index++));
      (*factor_matrix)(column, row) = value;
    }
  }
}

void add_scaled_packed_factor_row_to_lower_triangle(
    const Eigen::Ref<const Eigen::MatrixXd>& packed_factor_matrix,
    int auxiliary_index,
    double scale,
    double* output_storage,
    int n_basis_functions) {
  if (output_storage == nullptr) {
    throw std::invalid_argument("packed-row lower-triangle accumulation received null storage");
  }
  std::size_t packed_index = 0;
  for (int column = 0; column < n_basis_functions; ++column) {
    const std::size_t column_offset =
        xmvb::to_size(column) * n_basis_functions;
    for (int row = 0; row <= column; ++row) {
      output_storage[column_offset + xmvb::to_size(row)] +=
          scale * packed_factor_matrix(
              auxiliary_index,
              static_cast<Eigen::Index>(packed_index++));
    }
  }
}

void subtract_lower_triangle_in_place(
    const Eigen::MatrixXd& exchange_matrix,
    double* output_storage,
    int n_basis_functions) {
  if (output_storage == nullptr) {
    throw std::invalid_argument("output_storage must not be null");
  }
  for (int column = 0; column < n_basis_functions; ++column) {
    const std::size_t column_offset =
        xmvb::to_size(column) * n_basis_functions;
    for (int row = 0; row <= column; ++row) {
      output_storage[column_offset + xmvb::to_size(row)] -=
          exchange_matrix(column, row);
    }
  }
}

double packed_row_symmetric_matrix_dot(
    const Eigen::Ref<const Eigen::MatrixXd>& packed_factor_matrix,
    int auxiliary_index,
    const std::vector<double>& weighted_packed_matrix) {
  if (weighted_packed_matrix.empty()) {
    return 0.0;
  }

  double result = 0.0;
  for (std::size_t packed_index = 0;
       packed_index < weighted_packed_matrix.size();
       ++packed_index) {
    result +=
        packed_factor_matrix(auxiliary_index, static_cast<Eigen::Index>(packed_index)) *
        weighted_packed_matrix[packed_index];
  }
  return result;
}

std::vector<double> build_weighted_packed_symmetric_matrix(
    const Eigen::Ref<const Eigen::MatrixXd>& input_matrix) {
  const int n_basis_functions = static_cast<int>(input_matrix.rows());
  std::vector<double> weighted_packed_matrix(
      packed_pair_count(n_basis_functions),
      0.0);
  std::size_t packed_index = 0;
  for (int column = 0; column < n_basis_functions; ++column) {
    for (int row = 0; row <= column; ++row) {
      if (row == column) {
        weighted_packed_matrix[packed_index++] = input_matrix(row, column);
      } else {
        weighted_packed_matrix[packed_index++] =
            input_matrix(row, column) + input_matrix(column, row);
      }
    }
  }
  return weighted_packed_matrix;
}

SpectralFactorization build_spectral_factorization(
    const Eigen::Ref<const Eigen::MatrixXd>& input_matrix,
    const AoEffectiveOneElectronRiOperatorOptions& options) {
  SpectralFactorization factorization;
  if (!options.attempt_spectral_factorization) {
    return factorization;
  }
  if (!(options.spectral_eigenvalue_cutoff >= 0.0)) {
    throw std::invalid_argument("spectral_eigenvalue_cutoff must be non-negative");
  }
  if (!(options.low_rank_fraction_cutoff > 0.0 &&
        options.low_rank_fraction_cutoff <= 1.0)) {
    throw std::invalid_argument("low_rank_fraction_cutoff must lie in (0, 1]");
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(input_matrix);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("failed to diagonalize RI AO-H1E input matrix");
  }

  const auto eigenvalues = solver.eigenvalues();
  const double max_abs_eigenvalue =
      eigenvalues.cwiseAbs().size() > 0 ? eigenvalues.cwiseAbs().maxCoeff() : 0.0;
  const double effective_cutoff =
      std::max(options.spectral_eigenvalue_cutoff,
               std::numeric_limits<double>::epsilon() *
                   static_cast<double>(input_matrix.rows()) *
                   max_abs_eigenvalue);

  int n_positive_components = 0;
  int n_negative_components = 0;
  for (int index = 0; index < eigenvalues.size(); ++index) {
    const double eigenvalue = eigenvalues(index);
    if (eigenvalue > effective_cutoff) {
      ++n_positive_components;
    } else if (eigenvalue < -effective_cutoff) {
      ++n_negative_components;
    }
  }
  const int effective_rank = n_positive_components + n_negative_components;
  factorization.n_positive_components = n_positive_components;
  factorization.n_negative_components = n_negative_components;
  factorization.effective_rank = effective_rank;
  if (effective_rank == 0) {
    factorization.use_low_rank_path = true;
    return factorization;
  }
  if (static_cast<double>(effective_rank) >
      options.low_rank_fraction_cutoff * static_cast<double>(input_matrix.rows())) {
    return factorization;
  }

  factorization.scaled_eigenvectors.resize(input_matrix.rows(), effective_rank);
  factorization.use_low_rank_path = true;

  int positive_column = 0;
  int negative_column = n_positive_components;
  for (int index = 0; index < eigenvalues.size(); ++index) {
    const double eigenvalue = eigenvalues(index);
    if (eigenvalue > effective_cutoff) {
      factorization.scaled_eigenvectors.col(positive_column++) =
          solver.eigenvectors().col(index) * std::sqrt(eigenvalue);
    } else if (eigenvalue < -effective_cutoff) {
      factorization.scaled_eigenvectors.col(negative_column++) =
          solver.eigenvectors().col(index) * std::sqrt(-eigenvalue);
    }
  }
  return factorization;
}

SpectralFactorization build_prefactorized_spectral_factorization(
    const AoEffectiveOneElectronRiLowRankFactors& low_rank_factors,
    int n_basis_functions) {
  SpectralFactorization factorization;
  if (low_rank_factors.scaled_factor_matrix.rows() != n_basis_functions) {
    throw std::invalid_argument("prefactorized RI operator row count mismatch");
  }
  if (low_rank_factors.n_positive_components < 0 ||
      low_rank_factors.n_negative_components < 0) {
    throw std::invalid_argument("prefactorized RI operator component counts must be non-negative");
  }

  const int effective_rank =
      low_rank_factors.n_positive_components + low_rank_factors.n_negative_components;
  if (low_rank_factors.scaled_factor_matrix.cols() != effective_rank) {
    throw std::invalid_argument("prefactorized RI operator factor column count mismatch");
  }

  factorization.n_positive_components = low_rank_factors.n_positive_components;
  factorization.n_negative_components = low_rank_factors.n_negative_components;
  factorization.effective_rank = effective_rank;
  factorization.use_low_rank_path = true;
  if (effective_rank == 0) {
    factorization.scaled_eigenvectors.resize(n_basis_functions, 0);
    return factorization;
  }

  factorization.scaled_eigenvectors = low_rank_factors.scaled_factor_matrix;
  return factorization;
}

void symmetrize_in_place(
    std::vector<double>* matrix_storage,
    int n_basis_functions) {
  if (matrix_storage == nullptr) {
    throw std::invalid_argument("matrix_storage must not be null");
  }
  Eigen::Map<Eigen::MatrixXd> matrix(
      matrix_storage->data(),
      n_basis_functions,
      n_basis_functions);
  for (int column = 0; column < n_basis_functions; ++column) {
    for (int row = 0; row < column; ++row) {
      // The RI sweep accumulates complementary contributions into opposite
      // triangles: Coulomb updates are written in packed-pair order while the
      // exchange kernels update a selfadjoint view. Finalization therefore
      // must merge both triangles, mirroring the exact AO path, rather than
      // averaging them away.
      const double symmetrized_sum = matrix(column, row) + matrix(row, column);
      matrix(column, row) = symmetrized_sum;
      matrix(row, column) = symmetrized_sum;
    }
  }
}

std::vector<double> apply_low_rank_ri_operator(
    const LibcintRiIntegralProviderResult& ri_integral_provider_result,
    int n_basis_functions,
    const SpectralFactorization& factorization) {
  const std::size_t matrix_size =
      xmvb::to_size(n_basis_functions) * n_basis_functions;
  std::vector<double> output_storage(matrix_size, 0.0);
  if (factorization.scaled_eigenvectors.cols() == 0) {
    return output_storage;
  }

  const auto& packed_factor_matrix =
      ri_integral_provider_result.metric_whitened_ao_pair_factors;
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  std::vector<std::vector<double>> partial_outputs(
      xmvb::to_size(n_threads),
      std::vector<double>(matrix_size, 0.0));

#pragma omp parallel
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    Eigen::Map<Eigen::MatrixXd> local_output(
        partial_outputs[xmvb::to_size(thread_index)].data(),
        n_basis_functions,
        n_basis_functions);
    Eigen::MatrixXd factor_matrix(n_basis_functions, n_basis_functions);
    Eigen::MatrixXd transformed_eigenvectors(
        n_basis_functions,
        factorization.scaled_eigenvectors.cols());

#pragma omp for schedule(static)
    for (std::ptrdiff_t auxiliary_offset = 0;
         auxiliary_offset < ri_integral_provider_result.n_auxiliary_functions;
         ++auxiliary_offset) {
      unpack_packed_factor_row_lower_triangle(
          packed_factor_matrix,
          static_cast<int>(auxiliary_offset),
          n_basis_functions,
          &factor_matrix);

      // The inactive-density path is typically low-rank. We therefore apply the
      // symmetric RI factor to the retained spectral columns and form the
      // exchange term as signed rank updates instead of dense `L_A X L_A`.
      //
      // A molecule-static dense-factor cache was prototyped here, but for the
      // production `C6H6_full` case it regressed wall time because streaming
      // one large `[aux][ao][ao]` buffer from memory was slower than unpacking
      // each packed row into a thread-local matrix and immediately reusing it.
      // Keeping the small local buffer preserves cache locality while still
      // letting BLAS handle the expensive symmetric multiplies.
      cblas_dsymm(
          CblasColMajor,
          CblasLeft,
          CblasLower,
          n_basis_functions,
          factorization.scaled_eigenvectors.cols(),
          1.0,
          factor_matrix.data(),
          n_basis_functions,
          factorization.scaled_eigenvectors.data(),
          n_basis_functions,
          0.0,
          transformed_eigenvectors.data(),
          n_basis_functions);

      double coulomb_projection = 0.0;
      for (int column = 0; column < factorization.n_positive_components; ++column) {
        coulomb_projection +=
            cblas_ddot(
                n_basis_functions,
                factorization.scaled_eigenvectors.col(column).data(),
                1,
                transformed_eigenvectors.col(column).data(),
                1);
      }
      for (int column = factorization.n_positive_components;
           column < transformed_eigenvectors.cols();
           ++column) {
        coulomb_projection -=
            cblas_ddot(
                n_basis_functions,
                factorization.scaled_eigenvectors.col(column).data(),
                1,
                transformed_eigenvectors.col(column).data(),
                1);
      }
      add_scaled_packed_factor_row_to_lower_triangle(
          packed_factor_matrix,
          static_cast<int>(auxiliary_offset),
          2.0 * coulomb_projection,
          local_output.data(),
          n_basis_functions);
      if (factorization.n_positive_components > 0) {
        cblas_dsyrk(
            CblasColMajor,
            CblasLower,
            CblasNoTrans,
            n_basis_functions,
            factorization.n_positive_components,
            -1.0,
            transformed_eigenvectors.data(),
            n_basis_functions,
            1.0,
            local_output.data(),
            n_basis_functions);
      }
      if (factorization.n_negative_components > 0) {
        cblas_dsyrk(
            CblasColMajor,
            CblasLower,
            CblasNoTrans,
            n_basis_functions,
            factorization.n_negative_components,
            1.0,
            transformed_eigenvectors.data() +
                xmvb::to_size(factorization.n_positive_components) *
                    n_basis_functions,
            n_basis_functions,
            1.0,
            local_output.data(),
            n_basis_functions);
      }
    }
  }

  for (const auto& partial_output : partial_outputs) {
    for (std::size_t index = 0; index < output_storage.size(); ++index) {
      output_storage[index] += partial_output[index];
    }
  }
  symmetrize_in_place(&output_storage, n_basis_functions);
  return output_storage;
}

std::vector<double> apply_dense_ri_operator(
    const Eigen::Ref<const Eigen::MatrixXd>& input_matrix,
    const LibcintRiIntegralProviderResult& ri_integral_provider_result,
    int n_basis_functions) {
  const std::size_t matrix_size =
      xmvb::to_size(n_basis_functions) * n_basis_functions;
  std::vector<double> output_storage(matrix_size, 0.0);
  const auto weighted_packed_input =
      build_weighted_packed_symmetric_matrix(input_matrix);
  const auto& packed_factor_matrix =
      ri_integral_provider_result.metric_whitened_ao_pair_factors;

  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  std::vector<std::vector<double>> partial_outputs(
      xmvb::to_size(n_threads),
      std::vector<double>(matrix_size, 0.0));

#pragma omp parallel
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    Eigen::Map<Eigen::MatrixXd> local_output(
        partial_outputs[xmvb::to_size(thread_index)].data(),
        n_basis_functions,
        n_basis_functions);
    Eigen::MatrixXd factor_matrix(n_basis_functions, n_basis_functions);
    Eigen::MatrixXd left_workspace(n_basis_functions, n_basis_functions);
    Eigen::MatrixXd exchange_matrix(n_basis_functions, n_basis_functions);

#pragma omp for schedule(static)
    for (std::ptrdiff_t auxiliary_offset = 0;
         auxiliary_offset < ri_integral_provider_result.n_auxiliary_functions;
         ++auxiliary_offset) {
      const double coulomb_projection =
          packed_row_symmetric_matrix_dot(
              packed_factor_matrix,
              static_cast<int>(auxiliary_offset),
              weighted_packed_input);
      unpack_packed_factor_row_lower_triangle(
          packed_factor_matrix,
          static_cast<int>(auxiliary_offset),
          n_basis_functions,
          &factor_matrix);

      // Even in the dense fallback we only expose the lower triangle of `L_A`
      // to BLAS symmetric kernels, which avoids treating the RI factor as a
      // generic dense matrix and matches the dense-lower cache layout.
      cblas_dsymm(
          CblasColMajor,
          CblasLeft,
          CblasLower,
          n_basis_functions,
          n_basis_functions,
          1.0,
          factor_matrix.data(),
          n_basis_functions,
          input_matrix.data(),
          n_basis_functions,
          0.0,
          left_workspace.data(),
          n_basis_functions);
      cblas_dsymm(
          CblasColMajor,
          CblasRight,
          CblasLower,
          n_basis_functions,
          n_basis_functions,
          1.0,
          factor_matrix.data(),
          n_basis_functions,
          left_workspace.data(),
          n_basis_functions,
          0.0,
          exchange_matrix.data(),
          n_basis_functions);
      add_scaled_packed_factor_row_to_lower_triangle(
          packed_factor_matrix,
          static_cast<int>(auxiliary_offset),
          2.0 * coulomb_projection,
          local_output.data(),
          n_basis_functions);
      subtract_lower_triangle_in_place(
          exchange_matrix,
          local_output.data(),
          n_basis_functions);
    }
  }

  for (const auto& partial_output : partial_outputs) {
    for (std::size_t index = 0; index < output_storage.size(); ++index) {
      output_storage[index] += partial_output[index];
    }
  }
  symmetrize_in_place(&output_storage, n_basis_functions);
  return output_storage;
}

}  // namespace

std::vector<double> apply_ao_effective_one_electron_ri_operator(
    const std::vector<double>& input_matrix,
    const LibcintRiIntegralProviderResult& ri_integral_provider_result,
    int n_basis_functions,
    const AoEffectiveOneElectronRiOperatorOptions& options) {
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }
  if (ri_integral_provider_result.n_basis_functions != n_basis_functions) {
    throw std::invalid_argument("RI basis-function count mismatch");
  }

  const std::size_t matrix_size =
      xmvb::to_size(n_basis_functions) * n_basis_functions;
  if (input_matrix.size() != matrix_size) {
    throw std::invalid_argument("input_matrix size mismatch");
  }

  const std::size_t n_packed_pairs = packed_pair_count(n_basis_functions);
  if (ri_integral_provider_result.metric_whitened_ao_pair_factors.rows() !=
          ri_integral_provider_result.n_auxiliary_functions ||
      ri_integral_provider_result.metric_whitened_ao_pair_factors.cols() !=
          static_cast<Eigen::Index>(n_packed_pairs)) {
    throw std::invalid_argument("RI AO pair-factor matrix shape mismatch");
  }

  const Eigen::Map<const Eigen::MatrixXd> input(
      input_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const auto spectral_factorization =
      build_spectral_factorization(input, options);
  if (spectral_factorization.use_low_rank_path) {
    return apply_low_rank_ri_operator(
        ri_integral_provider_result,
        n_basis_functions,
        spectral_factorization);
  }
  return apply_dense_ri_operator(
      input,
      ri_integral_provider_result,
      n_basis_functions);
}

std::vector<double> apply_ao_effective_one_electron_ri_operator(
    const AoEffectiveOneElectronRiLowRankFactors& low_rank_factors,
    const LibcintRiIntegralProviderResult& ri_integral_provider_result,
    int n_basis_functions) {
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }
  if (ri_integral_provider_result.n_basis_functions != n_basis_functions) {
    throw std::invalid_argument("RI basis-function count mismatch");
  }

  const std::size_t n_packed_pairs = packed_pair_count(n_basis_functions);
  if (ri_integral_provider_result.metric_whitened_ao_pair_factors.rows() !=
          ri_integral_provider_result.n_auxiliary_functions ||
      ri_integral_provider_result.metric_whitened_ao_pair_factors.cols() !=
          static_cast<Eigen::Index>(n_packed_pairs)) {
    throw std::invalid_argument("RI AO pair-factor matrix shape mismatch");
  }

  const auto spectral_factorization =
      build_prefactorized_spectral_factorization(
          low_rank_factors,
          n_basis_functions);
  return apply_low_rank_ri_operator(
      ri_integral_provider_result,
      n_basis_functions,
      spectral_factorization);
}

}  // namespace xmvb::vb
