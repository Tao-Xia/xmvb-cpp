#include "vbscf/integrals/ao/one_electron/ri_operator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cblas.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/openmp.hpp"

namespace xmvb::vb {

namespace {

std::size_t packed_pair_count(int n_basis_functions) {
  const std::size_t n = n_basis_functions;
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
        column * n_basis_functions;
    for (int row = 0; row <= column; ++row) {
      output_storage[column_offset + row] +=
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
        column * n_basis_functions;
    for (int row = 0; row <= column; ++row) {
      output_storage[column_offset + row] -=
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

void accumulate_dense_ri_factor_action(
    const Eigen::Ref<const Eigen::MatrixXd>& packed_factor_matrix,
    int auxiliary_index,
    const Eigen::Ref<const Eigen::MatrixXd>& input_matrix,
    const std::vector<double>& weighted_packed_input,
    const Eigen::MatrixXd& factor_matrix,
    Eigen::MatrixXd* left_workspace,
    Eigen::MatrixXd* exchange_matrix,
    double* local_output) {
  if (left_workspace == nullptr || exchange_matrix == nullptr ||
      local_output == nullptr) {
    throw std::invalid_argument("dense RI factor action received null workspace");
  }
  const int n_basis_functions = static_cast<int>(input_matrix.rows());
  const double coulomb_projection =
      packed_row_symmetric_matrix_dot(
          packed_factor_matrix,
          auxiliary_index,
          weighted_packed_input);

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
      left_workspace->data(),
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
      left_workspace->data(),
      n_basis_functions,
      0.0,
      exchange_matrix->data(),
      n_basis_functions);
  add_scaled_packed_factor_row_to_lower_triangle(
      packed_factor_matrix,
      auxiliary_index,
      2.0 * coulomb_projection,
      local_output,
      n_basis_functions);
  subtract_lower_triangle_in_place(
      *exchange_matrix,
      local_output,
      n_basis_functions);
}

void build_weighted_packed_symmetric_matrix(
    const Eigen::Ref<const Eigen::MatrixXd>& input_matrix,
    std::vector<double>* weighted_packed_matrix) {
  if (weighted_packed_matrix == nullptr) {
    throw std::invalid_argument("weighted packed matrix output must not be null");
  }
  const int n_basis_functions = static_cast<int>(input_matrix.rows());
  weighted_packed_matrix->resize(packed_pair_count(n_basis_functions));
  std::size_t packed_index = 0;
  for (int column = 0; column < n_basis_functions; ++column) {
    for (int row = 0; row <= column; ++row) {
      if (row == column) {
        (*weighted_packed_matrix)[packed_index++] = input_matrix(row, column);
      } else {
        (*weighted_packed_matrix)[packed_index++] =
            input_matrix(row, column) + input_matrix(column, row);
      }
    }
  }
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

int ri_auxiliary_thread_count(int n_auxiliary_functions) {
  // RI application partitions by auxiliary factor. The reduction buffers are
  // full AO matrices, so launching more workers than factors, or launching an
  // inner team under an existing exact_ctx parallel region, only increases
  // memory traffic without adding useful work.
  int n_threads = xmvb::effective_openmp_thread_count();
  if (n_auxiliary_functions <= 1) {
    return 1;
  }
  if (n_threads > n_auxiliary_functions) {
    n_threads = n_auxiliary_functions;
  }
  return std::max(1, n_threads);
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

void symmetrize_in_place(Eigen::MatrixXd* matrix) {
  if (matrix == nullptr || matrix->rows() != matrix->cols()) {
    throw std::invalid_argument("invalid RI matrix symmetrization target");
  }
  for (int column = 0; column < matrix->cols(); ++column) {
    for (int row = 0; row < column; ++row) {
      const double symmetrized_sum =
          (*matrix)(column, row) + (*matrix)(row, column);
      (*matrix)(column, row) = symmetrized_sum;
      (*matrix)(row, column) = symmetrized_sum;
    }
  }
}

void validate_ri_factorization(
    const RiAoFactorization& ri_factorization,
    int n_basis_functions) {
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }
  if (ri_factorization.n_basis_functions != n_basis_functions) {
    throw std::invalid_argument("RI basis-function count mismatch");
  }
  if (ri_factorization.n_auxiliary_functions < 0) {
    throw std::invalid_argument("RI auxiliary-function count must be non-negative");
  }

  const std::size_t n_packed_pairs = packed_pair_count(n_basis_functions);
  if (ri_factorization.n_packed_ao_pairs != 0 &&
      ri_factorization.n_packed_ao_pairs !=
          static_cast<int>(n_packed_pairs)) {
    throw std::invalid_argument("RI packed AO pair count mismatch");
  }
  if (ri_factorization.metric_whitened_ao_pair_factors.rows() !=
          ri_factorization.n_auxiliary_functions ||
      ri_factorization.metric_whitened_ao_pair_factors.cols() !=
          static_cast<Eigen::Index>(n_packed_pairs)) {
    throw std::invalid_argument("RI AO pair-factor matrix shape mismatch");
  }
}

void resize_fused_workspace(
    AoEffectiveOneElectronRiFusedWorkspace* workspace,
    int n_threads,
    int n_basis_functions) {
  if (workspace == nullptr) {
    throw std::invalid_argument("RI fused workspace must not be null");
  }
  workspace->partial_forward.resize(n_threads);
  workspace->partial_adjoint.resize(n_threads);
  workspace->factor_matrices.resize(n_threads);
  workspace->left_products.resize(n_threads);
  workspace->exchange_products.resize(n_threads);
  for (int thread = 0; thread < n_threads; ++thread) {
    workspace->partial_forward[thread].setZero(
        n_basis_functions,
        n_basis_functions);
    workspace->partial_adjoint[thread].setZero(
        n_basis_functions,
        n_basis_functions);
    workspace->factor_matrices[thread].resize(
        n_basis_functions,
        n_basis_functions);
    workspace->left_products[thread].resize(
        n_basis_functions,
        n_basis_functions);
    workspace->exchange_products[thread].resize(
        n_basis_functions,
        n_basis_functions);
  }
}

void apply_fused_dense_ri_operator(
    const Eigen::Ref<const Eigen::MatrixXd>& source,
    const Eigen::Ref<const Eigen::MatrixXd>& adjoint,
    const RiAoFactorization& ri_factorization,
    AoEffectiveOneElectronRiFusedWorkspace* workspace,
    Eigen::MatrixXd* forward,
    Eigen::MatrixXd* transpose) {
  const int n_basis_functions = ri_factorization.n_basis_functions;
  const int n_threads = ri_auxiliary_thread_count(
      ri_factorization.n_auxiliary_functions);
  resize_fused_workspace(
      workspace,
      n_threads,
      n_basis_functions);
  build_weighted_packed_symmetric_matrix(
      source,
      &workspace->weighted_packed_forward);
  build_weighted_packed_symmetric_matrix(
      adjoint,
      &workspace->weighted_packed_adjoint);

  const auto& packed_factor_matrix =
      ri_factorization.metric_whitened_ao_pair_factors;
#pragma omp parallel num_threads(n_threads)
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    Eigen::MatrixXd& factor_matrix =
        workspace->factor_matrices[thread_index];
    Eigen::MatrixXd& left_product =
        workspace->left_products[thread_index];
    Eigen::MatrixXd& exchange_product =
        workspace->exchange_products[thread_index];
    Eigen::MatrixXd& local_forward =
        workspace->partial_forward[thread_index];
    Eigen::MatrixXd& local_adjoint =
        workspace->partial_adjoint[thread_index];

#pragma omp for schedule(static)
    for (std::ptrdiff_t auxiliary_offset = 0;
         auxiliary_offset < ri_factorization.n_auxiliary_functions;
         ++auxiliary_offset) {
      const int auxiliary_index = static_cast<int>(auxiliary_offset);
      unpack_packed_factor_row_lower_triangle(
          packed_factor_matrix,
          auxiliary_index,
          n_basis_functions,
          &factor_matrix);
      accumulate_dense_ri_factor_action(
          packed_factor_matrix,
          auxiliary_index,
          source,
          workspace->weighted_packed_forward,
          factor_matrix,
          &left_product,
          &exchange_product,
          local_forward.data());
      accumulate_dense_ri_factor_action(
          packed_factor_matrix,
          auxiliary_index,
          adjoint,
          workspace->weighted_packed_adjoint,
          factor_matrix,
          &left_product,
          &exchange_product,
          local_adjoint.data());
    }
  }

  forward->setZero(n_basis_functions, n_basis_functions);
  transpose->setZero(n_basis_functions, n_basis_functions);
  for (int thread = 0; thread < n_threads; ++thread) {
    *forward += workspace->partial_forward[thread];
    *transpose += workspace->partial_adjoint[thread];
  }
  symmetrize_in_place(forward);
  symmetrize_in_place(transpose);
}

std::vector<double> apply_low_rank_ri_operator(
    const RiAoFactorization& ri_factorization,
    int n_basis_functions,
    const SpectralFactorization& factorization) {
  const std::size_t matrix_size =
      n_basis_functions * n_basis_functions;
  std::vector<double> output_storage(matrix_size, 0.0);
  if (factorization.scaled_eigenvectors.cols() == 0) {
    return output_storage;
  }

  const auto& packed_factor_matrix =
      ri_factorization.metric_whitened_ao_pair_factors;
  const int n_auxiliary = ri_factorization.n_auxiliary_functions;
  const int rank = factorization.effective_rank;

  // Store Z_(A,k),mu = sum_nu L_(A,mu,nu) U_(nu,k) as a
  // (n_aux * rank)-by-n_bf matrix.  Each AO slice is obtained by one GEMM from
  // a contiguous auxiliary-major view of the packed RI tensor.  This replaces
  // n_aux small symmetric matrix products and rank updates by n_bf level-3
  // transformations followed by two large SYRK contractions.
  Eigen::MatrixXd transformed(n_auxiliary * rank, n_basis_functions);
  const int n_threads = std::min(
      xmvb::effective_openmp_thread_count(),
      n_basis_functions);
#pragma omp parallel num_threads(n_threads)
  {
    Eigen::MatrixXd ao_slice(n_auxiliary, n_basis_functions);
#pragma omp for schedule(static)
    for (int first = 0; first < n_basis_functions; ++first) {
      for (int second = 0; second < n_basis_functions; ++second) {
        const int larger = std::max(first, second);
        const int smaller = std::min(first, second);
        const Eigen::Index packed_index =
            static_cast<Eigen::Index>(larger) * (larger + 1) / 2 + smaller;
        ao_slice.col(second) = packed_factor_matrix.col(packed_index);
      }
      Eigen::Map<Eigen::MatrixXd> transformed_slice(
          transformed.col(first).data(),
          n_auxiliary,
          rank);
      transformed_slice.noalias() =
          ao_slice * factorization.scaled_eigenvectors;
    }
  }

  std::vector<Eigen::VectorXd> partial_projections;
  partial_projections.reserve(n_threads);
  for (int thread = 0; thread < n_threads; ++thread) {
    partial_projections.emplace_back(Eigen::VectorXd::Zero(n_auxiliary));
  }
#pragma omp parallel num_threads(n_threads)
  {
    int thread = 0;
#ifdef _OPENMP
    thread = omp_get_thread_num();
#endif
#pragma omp for schedule(static)
    for (int orbital = 0; orbital < n_basis_functions; ++orbital) {
      const Eigen::Map<const Eigen::MatrixXd> transformed_slice(
          transformed.col(orbital).data(),
          n_auxiliary,
          rank);
      for (int component = 0;
           component < factorization.n_positive_components;
           ++component) {
        partial_projections[thread].noalias() +=
            factorization.scaled_eigenvectors(orbital, component) *
            transformed_slice.col(component);
      }
      for (int component = factorization.n_positive_components;
           component < rank;
           ++component) {
        partial_projections[thread].noalias() -=
            factorization.scaled_eigenvectors(orbital, component) *
            transformed_slice.col(component);
      }
    }
  }
  Eigen::VectorXd auxiliary_projection =
      Eigen::VectorXd::Zero(n_auxiliary);
  for (const auto& partial_projection : partial_projections) {
    auxiliary_projection += partial_projection;
  }

  Eigen::Map<Eigen::MatrixXd> output(
      output_storage.data(),
      n_basis_functions,
      n_basis_functions);
  std::vector<Eigen::MatrixXd> partial_exchange;
  partial_exchange.reserve(n_threads);
  for (int thread = 0; thread < n_threads; ++thread) {
    partial_exchange.emplace_back(
        Eigen::MatrixXd::Zero(n_basis_functions, n_basis_functions));
  }
#pragma omp parallel num_threads(n_threads)
  {
    int thread = 0;
#ifdef _OPENMP
    thread = omp_get_thread_num();
#endif
    Eigen::MatrixXd& local_exchange = partial_exchange[thread];
    const int positive_rows =
        n_auxiliary * factorization.n_positive_components;
    const int positive_begin = positive_rows * thread / n_threads;
    const int positive_end = positive_rows * (thread + 1) / n_threads;
    if (positive_end > positive_begin) {
      cblas_dsyrk(
          CblasColMajor,
          CblasLower,
          CblasTrans,
          n_basis_functions,
          positive_end - positive_begin,
          -1.0,
          transformed.data() + positive_begin,
          transformed.rows(),
          0.0,
          local_exchange.data(),
          n_basis_functions);
    }
    const int negative_rows =
        n_auxiliary * factorization.n_negative_components;
    const int negative_begin = negative_rows * thread / n_threads;
    const int negative_end = negative_rows * (thread + 1) / n_threads;
    if (negative_end > negative_begin) {
      cblas_dsyrk(
          CblasColMajor,
          CblasLower,
          CblasTrans,
          n_basis_functions,
          negative_end - negative_begin,
          1.0,
          transformed.data() + positive_rows + negative_begin,
          transformed.rows(),
          positive_end > positive_begin ? 1.0 : 0.0,
          local_exchange.data(),
          n_basis_functions);
    }
  }
  for (const auto& local_exchange : partial_exchange) {
    output += local_exchange;
  }

  const Eigen::VectorXd packed_coulomb =
      2.0 * packed_factor_matrix.transpose() * auxiliary_projection;
  Eigen::Index packed_index = 0;
  for (int column = 0; column < n_basis_functions; ++column) {
    for (int row = 0; row <= column; ++row) {
      output(row, column) += packed_coulomb(packed_index++);
    }
  }
  symmetrize_in_place(&output_storage, n_basis_functions);
  return output_storage;
}

std::vector<double> apply_dense_ri_operator(
    const Eigen::Ref<const Eigen::MatrixXd>& input_matrix,
    const RiAoFactorization& ri_factorization,
    int n_basis_functions) {
  const std::size_t matrix_size =
      n_basis_functions * n_basis_functions;
  std::vector<double> output_storage(matrix_size, 0.0);
  std::vector<double> weighted_packed_input;
  build_weighted_packed_symmetric_matrix(
      input_matrix,
      &weighted_packed_input);
  const auto& packed_factor_matrix =
      ri_factorization.metric_whitened_ao_pair_factors;

  int n_threads = 1;
  n_threads = ri_auxiliary_thread_count(
      ri_factorization.n_auxiliary_functions);
  std::vector<std::vector<double>> partial_outputs(
      n_threads,
      std::vector<double>(matrix_size, 0.0));

  // The dense RI contraction has the same reduction shape as the low-rank path:
  // local AO matrices avoid write conflicts, while the capped team width keeps
  // the memory footprint proportional to useful auxiliary work.
#pragma omp parallel num_threads(n_threads)
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    Eigen::Map<Eigen::MatrixXd> local_output(
        partial_outputs[thread_index].data(),
        n_basis_functions,
        n_basis_functions);
    Eigen::MatrixXd factor_matrix(n_basis_functions, n_basis_functions);
    Eigen::MatrixXd left_workspace(n_basis_functions, n_basis_functions);
    Eigen::MatrixXd exchange_matrix(n_basis_functions, n_basis_functions);

#pragma omp for schedule(static)
    for (std::ptrdiff_t auxiliary_offset = 0;
         auxiliary_offset < ri_factorization.n_auxiliary_functions;
         ++auxiliary_offset) {
      unpack_packed_factor_row_lower_triangle(
          packed_factor_matrix,
          static_cast<int>(auxiliary_offset),
          n_basis_functions,
          &factor_matrix);

      // In the dense contraction we expose only the lower triangle of `L_A`
      // to BLAS symmetric kernels, which avoids treating the RI factor as a
      // generic dense matrix and matches the dense-lower cache layout.
      accumulate_dense_ri_factor_action(
          packed_factor_matrix,
          static_cast<int>(auxiliary_offset),
          input_matrix,
          weighted_packed_input,
          factor_matrix,
          &left_workspace,
          &exchange_matrix,
          local_output.data());
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
    const RiAoFactorization& ri_factorization,
    int n_basis_functions,
    const AoEffectiveOneElectronRiOperatorOptions& options) {
  validate_ri_factorization(ri_factorization, n_basis_functions);

  const std::size_t matrix_size =
      n_basis_functions * n_basis_functions;
  if (input_matrix.size() != matrix_size) {
    throw std::invalid_argument("input_matrix size mismatch");
  }

  const Eigen::Map<const Eigen::MatrixXd> input(
      input_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const auto spectral_factorization =
      build_spectral_factorization(input, options);
  if (spectral_factorization.use_low_rank_path) {
    return apply_low_rank_ri_operator(
        ri_factorization,
        n_basis_functions,
        spectral_factorization);
  }
  return apply_dense_ri_operator(
      input,
      ri_factorization,
      n_basis_functions);
}

std::vector<double> apply_ao_effective_one_electron_ri_operator(
    const AoEffectiveOneElectronRiLowRankFactors& low_rank_factors,
    const RiAoFactorization& ri_factorization,
    int n_basis_functions) {
  validate_ri_factorization(ri_factorization, n_basis_functions);

  const auto spectral_factorization =
      build_prefactorized_spectral_factorization(
          low_rank_factors,
          n_basis_functions);
  return apply_low_rank_ri_operator(
      ri_factorization,
      n_basis_functions,
      spectral_factorization);
}

void apply_ao_effective_one_electron_ri_operator_fused(
    const Eigen::Ref<const Eigen::MatrixXd>& source,
    const Eigen::Ref<const Eigen::MatrixXd>& adjoint,
    const RiAoFactorization& ri_factorization,
    AoEffectiveOneElectronRiFusedWorkspace* workspace,
    Eigen::MatrixXd* forward,
    Eigen::MatrixXd* transpose) {
  const int n_basis_functions = ri_factorization.n_basis_functions;
  validate_ri_factorization(ri_factorization, n_basis_functions);
  if (source.rows() != n_basis_functions ||
      source.cols() != n_basis_functions ||
      adjoint.rows() != n_basis_functions ||
      adjoint.cols() != n_basis_functions) {
    throw std::invalid_argument("RI fused input matrix shape mismatch");
  }
  if (workspace == nullptr || forward == nullptr || transpose == nullptr) {
    throw std::invalid_argument("RI fused buffers must not be null");
  }
  if (forward == transpose || forward->data() == source.data() ||
      forward->data() == adjoint.data() ||
      transpose->data() == source.data() ||
      transpose->data() == adjoint.data()) {
    throw std::invalid_argument("RI fused outputs must not alias inputs or each other");
  }

  apply_fused_dense_ri_operator(
      source,
      adjoint,
      ri_factorization,
      workspace,
      forward,
      transpose);
}

void apply_ao_effective_one_electron_ri_operator_adaptive(
    const Eigen::Ref<const Eigen::MatrixXd>& source,
    const Eigen::Ref<const Eigen::MatrixXd>& adjoint,
    const RiAoFactorization& ri_factorization,
    AoEffectiveOneElectronRiFusedWorkspace* workspace,
    AoEffectiveOneElectronRiStrategy* strategy,
    Eigen::MatrixXd* forward,
    Eigen::MatrixXd* transpose) {
  if (strategy == nullptr) {
    throw std::invalid_argument("RI adaptive strategy must not be null");
  }
  const int n_basis_functions = ri_factorization.n_basis_functions;
  const auto run_dense = [&]() {
    apply_ao_effective_one_electron_ri_operator_fused(
        source,
        adjoint,
        ri_factorization,
        workspace,
        forward,
        transpose);
  };
  const auto run_spectral = [&]() {
    const std::vector<double> source_storage(
        source.data(), source.data() + source.size());
    const std::vector<double> adjoint_storage(
        adjoint.data(), adjoint.data() + adjoint.size());
    constexpr AoEffectiveOneElectronRiOperatorOptions options{
        .attempt_spectral_factorization = true};
    const std::vector<double> forward_storage =
        apply_ao_effective_one_electron_ri_operator(
            source_storage,
            ri_factorization,
            n_basis_functions,
            options);
    const std::vector<double> transpose_storage =
        apply_ao_effective_one_electron_ri_operator(
            adjoint_storage,
            ri_factorization,
            n_basis_functions,
            options);
    *forward = Eigen::Map<const Eigen::MatrixXd>(
        forward_storage.data(), n_basis_functions, n_basis_functions);
    *transpose = Eigen::Map<const Eigen::MatrixXd>(
        transpose_storage.data(), n_basis_functions, n_basis_functions);
  };

  if (*strategy == AoEffectiveOneElectronRiStrategy::DenseFused) {
    run_dense();
    return;
  }
  if (*strategy == AoEffectiveOneElectronRiStrategy::Spectral) {
    run_spectral();
    return;
  }

  const auto dense_start = std::chrono::steady_clock::now();
  run_dense();
  const double dense_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - dense_start).count();
  Eigen::MatrixXd dense_forward = *forward;
  Eigen::MatrixXd dense_transpose = *transpose;

  const auto spectral_start = std::chrono::steady_clock::now();
  run_spectral();
  const double spectral_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - spectral_start).count();
  const double forward_scale =
      std::max(1.0, dense_forward.cwiseAbs().maxCoeff());
  const double transpose_scale =
      std::max(1.0, dense_transpose.cwiseAbs().maxCoeff());
  if ((dense_forward - *forward).cwiseAbs().maxCoeff() >
          1.0e-8 * forward_scale ||
      (dense_transpose - *transpose).cwiseAbs().maxCoeff() >
          1.0e-8 * transpose_scale) {
    throw std::runtime_error(
        "RI AO-H1E spectral contraction disagrees with dense action");
  }
  if (spectral_seconds < dense_seconds) {
    *strategy = AoEffectiveOneElectronRiStrategy::Spectral;
  } else {
    *strategy = AoEffectiveOneElectronRiStrategy::DenseFused;
    *forward = std::move(dense_forward);
    *transpose = std::move(dense_transpose);
  }
}

}  // namespace xmvb::vb
