#include "vbscf/derivatives/hessian/exact_hvp_ao_one_electron_internal.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "vbscf/integrals/ao/ao_effective_one_electron_graph_operator.hpp"
#include "vbscf/integrals/ao/ao_integral_input.hpp"
#include "vbscf/orbitals/orbital_preparation_input.hpp"

namespace xmvb::vb::detail {
namespace {

std::vector<std::size_t> build_ao_matrix_column_offsets(
    int n_basis_functions) {
  std::vector<std::size_t> column_offsets(n_basis_functions, 0);
  for (int column = 0; column < n_basis_functions; ++column) {
    column_offsets[column] =
        column * n_basis_functions;
  }
  return column_offsets;
}

int exact_ctx_current_openmp_max_threads_local() {
  int omp_max_threads = 1;
#ifdef _OPENMP
  omp_max_threads = omp_get_max_threads();
#endif
  return std::max(1, omp_max_threads);
}

std::size_t ao_pair_index_packed(int first, int second) {
  if (first >= second) {
    return first * (first + 1) / 2 + second;
  }
  return second * (second + 1) / 2 + first;
}

template <bool ValidateIntegralIndices, bool UsePrecomputedSymmetryShifts, bool UseLinearIndexCache>
inline void accumulate_fused_ao_effective_one_electron_integral(
    std::size_t integral_index,
    const double* inactive_density_matrix,
    const double* unsymmetrized_gradient_storage,
    const double* ao_two_electron_integral_values,
    const int* ao_two_electron_integral_indices,
    const std::uint8_t* ao_two_electron_integral_symmetry_shifts,
    const int* ao_effective_one_electron_linear_indices,
    const std::size_t* column_offsets,
    int n_basis_functions,
    std::atomic<int>& invalid_integral_index,
    double* local_delta_ao_effective_h1e,
    double* local_inactive_density_gradient) {
  const int* integral_indices =
      ao_two_electron_integral_indices + integral_index * 4;
  const int i = integral_indices[0];
  const int j = integral_indices[1];
  const int k = integral_indices[2];
  const int l = integral_indices[3];

  if constexpr (ValidateIntegralIndices) {
    if (i < 0 || i >= n_basis_functions ||
        j < 0 || j >= n_basis_functions ||
        k < 0 || k >= n_basis_functions ||
        l < 0 || l >= n_basis_functions) {
      int expected = -1;
      invalid_integral_index.compare_exchange_strong(
          expected,
          static_cast<int>(integral_index));
      return;
    }
  }

  double two_electron_value = ao_two_electron_integral_values[integral_index];
  if constexpr (UsePrecomputedSymmetryShifts) {
    constexpr double kIntegralSymmetryMultipliers[] = {
        1.0,
        0.5,
        0.25,
        0.125,
    };
    two_electron_value *= kIntegralSymmetryMultipliers[
        ao_two_electron_integral_symmetry_shifts[integral_index]];
  } else {
    if (i == j) {
      two_electron_value *= 0.5;
    }
    if (k == l) {
      two_electron_value *= 0.5;
    }
    if (i == k && j == l) {
      two_electron_value *= 0.5;
    }
  }

  std::size_t ij_index = 0;
  std::size_t kl_index = 0;
  std::size_t ik_index = 0;
  std::size_t jl_index = 0;
  std::size_t il_index = 0;
  std::size_t jk_index = 0;
  std::size_t lj_index = 0;
  std::size_t ki_index = 0;
  std::size_t kj_index = 0;
  std::size_t li_index = 0;
  if constexpr (UseLinearIndexCache) {
    const int* linear_indices =
        ao_effective_one_electron_linear_indices + integral_index * 10;
    ij_index = linear_indices[0];
    kl_index = linear_indices[1];
    ik_index = linear_indices[2];
    jl_index = linear_indices[3];
    il_index = linear_indices[4];
    jk_index = linear_indices[5];
    lj_index = linear_indices[6];
    ki_index = linear_indices[7];
    kj_index = linear_indices[8];
    li_index = linear_indices[9];
  } else {
    const std::size_t col_i = column_offsets[i];
    const std::size_t col_j = column_offsets[j];
    const std::size_t col_k = column_offsets[k];
    const std::size_t col_l = column_offsets[l];

    ij_index = col_j + i;
    kl_index = col_l + k;
    ik_index = col_k + i;
    jl_index = col_l + j;
    il_index = col_l + i;
    jk_index = col_k + j;
    lj_index = col_j + l;
    ki_index = col_i + k;
    kj_index = col_j + k;
    li_index = col_i + l;
  }

  const double density_ij = inactive_density_matrix[ij_index];
  const double density_kl = inactive_density_matrix[kl_index];
  const double density_lj = inactive_density_matrix[lj_index];
  const double density_ki = inactive_density_matrix[ki_index];
  const double density_kj = inactive_density_matrix[kj_index];
  const double density_li = inactive_density_matrix[li_index];

  const double gradient_ij = unsymmetrized_gradient_storage[ij_index];
  const double gradient_kl = unsymmetrized_gradient_storage[kl_index];
  const double gradient_ik = unsymmetrized_gradient_storage[ik_index];
  const double gradient_jl = unsymmetrized_gradient_storage[jl_index];
  const double gradient_il = unsymmetrized_gradient_storage[il_index];
  const double gradient_jk = unsymmetrized_gradient_storage[jk_index];
  const double scaled_two_electron_value = two_electron_value * 4.0;

  local_delta_ao_effective_h1e[ij_index] +=
      density_kl * scaled_two_electron_value;
  local_delta_ao_effective_h1e[kl_index] +=
      density_ij * scaled_two_electron_value;
  local_delta_ao_effective_h1e[ik_index] -= density_lj * two_electron_value;
  local_delta_ao_effective_h1e[jl_index] -= density_ki * two_electron_value;
  local_delta_ao_effective_h1e[il_index] -= density_kj * two_electron_value;
  local_delta_ao_effective_h1e[jk_index] -= density_li * two_electron_value;

  local_inactive_density_gradient[ij_index] +=
      gradient_kl * scaled_two_electron_value;
  local_inactive_density_gradient[kl_index] +=
      gradient_ij * scaled_two_electron_value;
  local_inactive_density_gradient[lj_index] -=
      gradient_ik * two_electron_value;
  local_inactive_density_gradient[ki_index] -=
      gradient_jl * two_electron_value;
  local_inactive_density_gradient[kj_index] -=
      gradient_il * two_electron_value;
  local_inactive_density_gradient[li_index] -=
      gradient_jk * two_electron_value;
}

template <bool ValidateIntegralIndices, bool UsePrecomputedSymmetryShifts, bool UseLinearIndexCache>
inline void apply_fused_ao_effective_one_electron_integral_loop_single_thread(
    const double* inactive_density_matrix,
    const double* unsymmetrized_gradient_storage,
    const AoIntegralInput& ao_integral_input,
    const std::size_t* column_offsets,
    int n_basis_functions,
    std::atomic<int>& invalid_integral_index,
    double* delta_ao_effective_h1e,
    double* inactive_density_gradient) {
  const std::uint8_t* symmetry_shifts =
      UsePrecomputedSymmetryShifts
          ? ao_integral_input.ao_two_electron_integral_symmetry_shifts.data()
          : nullptr;
  const int* linear_indices =
      UseLinearIndexCache
          ? ao_integral_input.ao_effective_one_electron_linear_indices.data()
          : nullptr;
  for (std::size_t integral_index = 0;
       integral_index < ao_integral_input.ao_two_electron_integral_values.size();
       ++integral_index) {
    accumulate_fused_ao_effective_one_electron_integral<
        ValidateIntegralIndices,
        UsePrecomputedSymmetryShifts,
        UseLinearIndexCache>(
        integral_index,
        inactive_density_matrix,
        unsymmetrized_gradient_storage,
        ao_integral_input.ao_two_electron_integral_values.data(),
        ao_integral_input.ao_two_electron_integral_indices.data(),
        symmetry_shifts,
        linear_indices,
        column_offsets,
        n_basis_functions,
        invalid_integral_index,
        delta_ao_effective_h1e,
        inactive_density_gradient);
  }
}

template <bool ValidateIntegralIndices, bool UsePrecomputedSymmetryShifts, bool UseLinearIndexCache>
inline void apply_fused_ao_effective_one_electron_integral_loop_parallel(
    const double* inactive_density_matrix,
    const double* unsymmetrized_gradient_storage,
    const AoIntegralInput& ao_integral_input,
    const std::size_t* column_offsets,
    int n_basis_functions,
    int n_threads,
    std::atomic<int>& invalid_integral_index,
    std::vector<Eigen::MatrixXd>* partial_delta_h1e,
    std::vector<Eigen::MatrixXd>* partial_density_gradients) {
  const std::uint8_t* symmetry_shifts =
      UsePrecomputedSymmetryShifts
          ? ao_integral_input.ao_two_electron_integral_symmetry_shifts.data()
          : nullptr;
  const int* linear_indices =
      UseLinearIndexCache
          ? ao_integral_input.ao_effective_one_electron_linear_indices.data()
          : nullptr;

#pragma omp parallel num_threads(n_threads)
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    Eigen::MatrixXd& local_delta_h1e =
        (*partial_delta_h1e)[thread_index];
    Eigen::MatrixXd& local_density_gradient =
        (*partial_density_gradients)[thread_index];

#pragma omp for schedule(static)
    for (std::ptrdiff_t integral_offset = 0;
         integral_offset <
             static_cast<std::ptrdiff_t>(
                 ao_integral_input.ao_two_electron_integral_values.size());
         ++integral_offset) {
      const std::size_t integral_index = integral_offset;
      accumulate_fused_ao_effective_one_electron_integral<
          ValidateIntegralIndices,
          UsePrecomputedSymmetryShifts,
          UseLinearIndexCache>(
          integral_index,
          inactive_density_matrix,
          unsymmetrized_gradient_storage,
          ao_integral_input.ao_two_electron_integral_values.data(),
          ao_integral_input.ao_two_electron_integral_indices.data(),
          symmetry_shifts,
          linear_indices,
          column_offsets,
          n_basis_functions,
          invalid_integral_index,
          local_delta_h1e.data(),
          local_density_gradient.data());
    }
  }
}

void resize_zero_fused_ao_h1e_partial_workspaces(
    int n_threads,
    int n_basis_functions,
    std::vector<Eigen::MatrixXd>* partial_delta_h1e,
    std::vector<Eigen::MatrixXd>* partial_density_gradients) {
  const Eigen::Index basis_dim =
      static_cast<Eigen::Index>(n_basis_functions);
  const std::size_t thread_count =
      static_cast<std::size_t>(n_threads);
  partial_delta_h1e->resize(thread_count);
  partial_density_gradients->resize(thread_count);

  // Each Krylov matvec touches the same AO square shape. Keep one dense AO
  // accumulator per OpenMP worker and just zero/reuse it instead of rebuilding
  // `n_threads * n_basis^2` heap buffers on every apply.
  for (std::size_t thread_index = 0;
       thread_index < thread_count;
       ++thread_index) {
    Eigen::MatrixXd& local_delta_h1e =
        (*partial_delta_h1e)[thread_index];
    if (local_delta_h1e.rows() != basis_dim ||
        local_delta_h1e.cols() != basis_dim) {
      local_delta_h1e.resize(basis_dim, basis_dim);
    }
    local_delta_h1e.setZero();

    Eigen::MatrixXd& local_density_gradient =
        (*partial_density_gradients)[thread_index];
    if (local_density_gradient.rows() != basis_dim ||
        local_density_gradient.cols() != basis_dim) {
      local_density_gradient.resize(basis_dim, basis_dim);
    }
    local_density_gradient.setZero();
  }
}

}  // namespace

int choose_exact_ao_h1e_thread_count(
    const OrbitalPreparationInput& orbital_preparation_input) {
  const int n_basis_functions = static_cast<int>(std::min<std::size_t>(
      orbital_preparation_input.n_basis_functions,
      static_cast<std::size_t>(std::numeric_limits<int>::max())));
  return std::min(
      exact_ctx_current_openmp_max_threads_local(),
      std::max(1, n_basis_functions));
}

void apply_fused_exact_ao_one_electron_response(
    const Eigen::MatrixXd& inactive_density_matrix,
    const Eigen::MatrixXd& ao_effective_one_electron_gradient,
    const AoIntegralInput& ao_integral_input,
    const OrbitalPreparationInput& orbital_preparation_input,
    Eigen::MatrixXd* symmetrized_pullback_source,
    std::vector<double>* delta_ao_effective_h1e_storage,
    std::vector<double>* inactive_density_gradient_storage,
    std::vector<Eigen::MatrixXd>* partial_delta_h1e_workspaces,
    std::vector<Eigen::MatrixXd>* partial_density_gradient_workspaces) {
  const int n_basis_functions = ao_integral_input.n_basis_functions;
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("AO-H1E fused exact operator requires positive dimensions");
  }
  const std::size_t matrix_size =
      n_basis_functions * n_basis_functions;
  if (inactive_density_matrix.size() != matrix_size ||
      ao_effective_one_electron_gradient.size() != matrix_size) {
    throw std::invalid_argument("AO-H1E fused exact operator matrix size mismatch");
  }
  if (ao_integral_input.ao_two_electron_integral_indices.size() !=
      ao_integral_input.ao_two_electron_integral_values.size() * 4) {
    throw std::invalid_argument("AO two-electron index/value sizes are inconsistent");
  }
  if (!ao_integral_input.ao_two_electron_integral_symmetry_shifts.empty() &&
      ao_integral_input.ao_two_electron_integral_symmetry_shifts.size() !=
          ao_integral_input.ao_two_electron_integral_values.size()) {
    throw std::invalid_argument("AO two-electron symmetry-shift/value sizes are inconsistent");
  }
  if (!ao_integral_input.ao_effective_one_electron_linear_indices.empty() &&
      ao_integral_input.ao_effective_one_electron_linear_indices.size() !=
          ao_integral_input.ao_two_electron_integral_values.size() * 10) {
    throw std::invalid_argument(
        "AO effective one-electron linear-index cache size is inconsistent");
  }
  symmetrized_pullback_source->resizeLike(ao_effective_one_electron_gradient);
  *symmetrized_pullback_source = ao_effective_one_electron_gradient;
  *symmetrized_pullback_source += ao_effective_one_electron_gradient.transpose();
  delta_ao_effective_h1e_storage->assign(matrix_size, 0.0);
  inactive_density_gradient_storage->assign(matrix_size, 0.0);

  std::atomic<int> invalid_integral_index(-1);
  const bool validate_integral_indices =
      ao_integral_input.ao_two_electron_pair_indices.empty() &&
      ao_integral_input.ao_two_electron_pair_graph_row_offsets.empty();
  std::vector<std::size_t> column_offsets;
  if (ao_integral_input.ao_effective_one_electron_linear_indices.empty()) {
    column_offsets = build_ao_matrix_column_offsets(n_basis_functions);
  }
  const std::size_t* column_offsets_data =
      ao_integral_input.ao_effective_one_electron_linear_indices.empty()
          ? column_offsets.data()
          : nullptr;
  const std::uint8_t* symmetry_shifts =
      ao_integral_input.ao_two_electron_integral_symmetry_shifts.empty()
          ? nullptr
          : ao_integral_input.ao_two_electron_integral_symmetry_shifts.data();
  const int* linear_indices =
      ao_integral_input.ao_effective_one_electron_linear_indices.empty()
          ? nullptr
          : ao_integral_input.ao_effective_one_electron_linear_indices.data();

  const int n_threads =
      choose_exact_ao_h1e_thread_count(orbital_preparation_input);
  if (ao_effective_one_electron_graph_available(ao_integral_input)) {
    apply_fused_ao_effective_one_electron_graph(
        inactive_density_matrix.data(),
        symmetrized_pullback_source->data(),
        ao_integral_input,
        n_threads,
        delta_ao_effective_h1e_storage,
        inactive_density_gradient_storage);
  } else if (n_threads <= 1) {
    auto* delta_h1e = delta_ao_effective_h1e_storage->data();
    auto* density_gradient = inactive_density_gradient_storage->data();
    if (linear_indices != nullptr) {
      if (validate_integral_indices) {
        if (symmetry_shifts != nullptr) {
          apply_fused_ao_effective_one_electron_integral_loop_single_thread<
              true,
              true,
              true>(
              inactive_density_matrix.data(),
              symmetrized_pullback_source->data(),
              ao_integral_input,
              column_offsets_data,
              n_basis_functions,
              invalid_integral_index,
              delta_h1e,
              density_gradient);
        } else {
          apply_fused_ao_effective_one_electron_integral_loop_single_thread<
              true,
              false,
              true>(
              inactive_density_matrix.data(),
              symmetrized_pullback_source->data(),
              ao_integral_input,
              column_offsets_data,
              n_basis_functions,
              invalid_integral_index,
              delta_h1e,
              density_gradient);
        }
      } else if (symmetry_shifts != nullptr) {
        apply_fused_ao_effective_one_electron_integral_loop_single_thread<
            false,
            true,
            true>(
            inactive_density_matrix.data(),
            symmetrized_pullback_source->data(),
            ao_integral_input,
            column_offsets_data,
            n_basis_functions,
            invalid_integral_index,
            delta_h1e,
            density_gradient);
      } else {
        apply_fused_ao_effective_one_electron_integral_loop_single_thread<
            false,
            false,
            true>(
            inactive_density_matrix.data(),
            symmetrized_pullback_source->data(),
            ao_integral_input,
            column_offsets_data,
            n_basis_functions,
            invalid_integral_index,
            delta_h1e,
            density_gradient);
      }
    } else if (validate_integral_indices) {
      if (symmetry_shifts != nullptr) {
        apply_fused_ao_effective_one_electron_integral_loop_single_thread<
            true,
            true,
            false>(
            inactive_density_matrix.data(),
            symmetrized_pullback_source->data(),
            ao_integral_input,
            column_offsets_data,
            n_basis_functions,
            invalid_integral_index,
            delta_h1e,
            density_gradient);
      } else {
        apply_fused_ao_effective_one_electron_integral_loop_single_thread<
            true,
            false,
            false>(
            inactive_density_matrix.data(),
            symmetrized_pullback_source->data(),
            ao_integral_input,
            column_offsets_data,
            n_basis_functions,
            invalid_integral_index,
            delta_h1e,
            density_gradient);
      }
    } else if (symmetry_shifts != nullptr) {
      apply_fused_ao_effective_one_electron_integral_loop_single_thread<
          false,
          true,
          false>(
          inactive_density_matrix.data(),
          symmetrized_pullback_source->data(),
          ao_integral_input,
          column_offsets_data,
          n_basis_functions,
          invalid_integral_index,
          delta_h1e,
          density_gradient);
    } else {
      apply_fused_ao_effective_one_electron_integral_loop_single_thread<
          false,
          false,
          false>(
          inactive_density_matrix.data(),
          symmetrized_pullback_source->data(),
          ao_integral_input,
          column_offsets_data,
          n_basis_functions,
          invalid_integral_index,
          delta_h1e,
          density_gradient);
    }
  } else {
    resize_zero_fused_ao_h1e_partial_workspaces(
        n_threads,
        n_basis_functions,
        partial_delta_h1e_workspaces,
        partial_density_gradient_workspaces);

    if (linear_indices != nullptr) {
      if (validate_integral_indices) {
        if (symmetry_shifts != nullptr) {
          apply_fused_ao_effective_one_electron_integral_loop_parallel<
              true,
              true,
              true>(
              inactive_density_matrix.data(),
              symmetrized_pullback_source->data(),
              ao_integral_input,
              column_offsets_data,
              n_basis_functions,
              n_threads,
              invalid_integral_index,
              partial_delta_h1e_workspaces,
              partial_density_gradient_workspaces);
        } else {
          apply_fused_ao_effective_one_electron_integral_loop_parallel<
              true,
              false,
              true>(
              inactive_density_matrix.data(),
              symmetrized_pullback_source->data(),
              ao_integral_input,
              column_offsets_data,
              n_basis_functions,
              n_threads,
              invalid_integral_index,
              partial_delta_h1e_workspaces,
              partial_density_gradient_workspaces);
        }
      } else if (symmetry_shifts != nullptr) {
        apply_fused_ao_effective_one_electron_integral_loop_parallel<
            false,
            true,
            true>(
            inactive_density_matrix.data(),
            symmetrized_pullback_source->data(),
            ao_integral_input,
            column_offsets_data,
            n_basis_functions,
            n_threads,
            invalid_integral_index,
            partial_delta_h1e_workspaces,
            partial_density_gradient_workspaces);
      } else {
        apply_fused_ao_effective_one_electron_integral_loop_parallel<
            false,
            false,
            true>(
            inactive_density_matrix.data(),
            symmetrized_pullback_source->data(),
            ao_integral_input,
            column_offsets_data,
            n_basis_functions,
            n_threads,
            invalid_integral_index,
            partial_delta_h1e_workspaces,
            partial_density_gradient_workspaces);
      }
    } else if (validate_integral_indices) {
      if (symmetry_shifts != nullptr) {
        apply_fused_ao_effective_one_electron_integral_loop_parallel<
            true,
            true,
            false>(
            inactive_density_matrix.data(),
            symmetrized_pullback_source->data(),
            ao_integral_input,
            column_offsets_data,
            n_basis_functions,
            n_threads,
            invalid_integral_index,
            partial_delta_h1e_workspaces,
            partial_density_gradient_workspaces);
      } else {
        apply_fused_ao_effective_one_electron_integral_loop_parallel<
            true,
            false,
            false>(
            inactive_density_matrix.data(),
            symmetrized_pullback_source->data(),
            ao_integral_input,
            column_offsets_data,
            n_basis_functions,
            n_threads,
            invalid_integral_index,
            partial_delta_h1e_workspaces,
            partial_density_gradient_workspaces);
      }
    } else if (symmetry_shifts != nullptr) {
      apply_fused_ao_effective_one_electron_integral_loop_parallel<
          false,
          true,
          false>(
          inactive_density_matrix.data(),
          symmetrized_pullback_source->data(),
          ao_integral_input,
          column_offsets_data,
          n_basis_functions,
          n_threads,
          invalid_integral_index,
          partial_delta_h1e_workspaces,
          partial_density_gradient_workspaces);
    } else {
      apply_fused_ao_effective_one_electron_integral_loop_parallel<
          false,
          false,
          false>(
          inactive_density_matrix.data(),
          symmetrized_pullback_source->data(),
          ao_integral_input,
          column_offsets_data,
          n_basis_functions,
          n_threads,
          invalid_integral_index,
          partial_delta_h1e_workspaces,
          partial_density_gradient_workspaces);
    }

    Eigen::Map<Eigen::MatrixXd> delta_ao_effective_h1e(
        delta_ao_effective_h1e_storage->data(),
        n_basis_functions,
        n_basis_functions);
    Eigen::Map<Eigen::MatrixXd> inactive_density_gradient(
        inactive_density_gradient_storage->data(),
        n_basis_functions,
        n_basis_functions);
    for (const Eigen::MatrixXd& partial_delta_h1e_matrix :
         *partial_delta_h1e_workspaces) {
      delta_ao_effective_h1e += partial_delta_h1e_matrix;
    }
    for (const Eigen::MatrixXd& partial_density_gradient_matrix :
         *partial_density_gradient_workspaces) {
      inactive_density_gradient += partial_density_gradient_matrix;
    }
  }

  if (validate_integral_indices && invalid_integral_index.load() >= 0) {
    throw std::invalid_argument("AO two-electron index out of range");
  }
  Eigen::Map<Eigen::MatrixXd> delta_ao_effective_h1e(
      delta_ao_effective_h1e_storage->data(),
      n_basis_functions,
      n_basis_functions);
  for (int row = 0; row < n_basis_functions; ++row) {
    for (int column = 0; column <= row; ++column) {
      delta_ao_effective_h1e(row, column) =
          delta_ao_effective_h1e(row, column) +
          delta_ao_effective_h1e(column, row);
      delta_ao_effective_h1e(column, row) =
          delta_ao_effective_h1e(row, column);
    }
  }
}

}  // namespace xmvb::vb::detail
