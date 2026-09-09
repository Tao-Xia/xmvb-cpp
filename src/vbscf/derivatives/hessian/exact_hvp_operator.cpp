#include "vbscf/derivatives/hessian/exact_hvp_operator.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "vbscf/structures/coefficient_blocks.hpp"
#include "vbscf/integrals/active/two_electron_indexer.hpp"
#include "vbscf/integrals/active/active_space_two_electron_response.hpp"
#include "vbscf/integrals/ao/ao_effective_one_electron_graph_operator.hpp"
#include "vbscf/derivatives/hessian/responses/active_space_outer_response.hpp"
#include "vbscf/derivatives/hessian/responses/orbital_preparation_response.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin_backward.hpp"
#include "vbscf/derivatives/hessian/responses/structure_directional_response.hpp"
#include "vbscf/structures/selected_state_coefficients.hpp"

namespace xmvb::vb {

namespace {

struct FusedAoEffectiveOneElectronDirectionalResult {
  std::vector<double> delta_ao_effective_h1e;
  std::vector<double> inactive_density_gradient;
};

/**
 * @brief Low-rank AO matrix represented as `left_factors * right_factors^T`.
 *
 * The exact-context HVP repeatedly differentiates the inactive-space
 * projector.  That derivative has rank at most twice the number of inactive
 * orbitals even though its AO representation is dense.  Keeping the factor
 * form lets orbital-preparation and fixed-upstream chain-rule terms multiply
 * by the projector derivative without paying for avoidable AO-by-AO products.
 */
void add_orbital_value_gradient_in_place(
    std::vector<double>* target,
    const std::vector<double>& contribution,
    const char* /*label*/) {
  if (target->empty()) {
    *target = contribution;
    return;
  }
  for (std::size_t index = 0; index < target->size(); ++index) {
    (*target)[index] += contribution[index];
  }
}

void resize_for_overwrite(
    std::vector<double>* values,
    std::size_t size) {
  if (values->size() != size) {
    values->resize(size);
  }
}

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

int choose_exact_ctx_ao_h1e_threads(
    const OrbitalPreparationInput& orbital_preparation_input,
    bool compute_outer_response) {
  (void)compute_outer_response;
  const int n_basis_functions = static_cast<int>(std::min<std::size_t>(
      orbital_preparation_input.n_basis_functions,
      static_cast<std::size_t>(std::numeric_limits<int>::max())));
  return std::min(
      exact_ctx_current_openmp_max_threads_local(),
      std::max(1, n_basis_functions));
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

/**
 * The exact-context HVP needs both `delta G11[delta P11]` and the transpose
 * pullback `G11^T[delta Lambda]` against the same accepted AO ERI list.
 * Sweeping the materialized AO integrals once and accumulating both outputs
 * removes one full exact AO-H1E pass per matvec without introducing any new
 * spin-string-sized cache.
 */
void apply_fused_exact_ao_effective_one_electron_directional_operator(
    const Eigen::MatrixXd& inactive_density_matrix,
    const Eigen::MatrixXd& ao_effective_one_electron_gradient,
    const AoIntegralInput& ao_integral_input,
    const OrbitalPreparationInput& orbital_preparation_input,
    bool compute_outer_response,
    Eigen::MatrixXd* symmetrized_pullback_source,
    std::vector<double>* delta_ao_effective_h1e_storage,
    std::vector<double>* inactive_density_gradient_storage,
    std::vector<Eigen::MatrixXd>* partial_delta_h1e_workspaces = nullptr,
    std::vector<Eigen::MatrixXd>* partial_density_gradient_workspaces = nullptr) {
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
      choose_exact_ctx_ao_h1e_threads(
          orbital_preparation_input,
          compute_outer_response);
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
    std::vector<Eigen::MatrixXd> owned_partial_delta_h1e_workspaces;
    std::vector<Eigen::MatrixXd> owned_partial_density_gradient_workspaces;
    if (partial_delta_h1e_workspaces == nullptr) {
      partial_delta_h1e_workspaces =
          &owned_partial_delta_h1e_workspaces;
    }
    if (partial_density_gradient_workspaces == nullptr) {
      partial_density_gradient_workspaces =
          &owned_partial_density_gradient_workspaces;
    }
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

FusedAoEffectiveOneElectronDirectionalResult
apply_fused_exact_ao_effective_one_electron_directional_operator(
    const Eigen::MatrixXd& inactive_density_matrix,
    const Eigen::MatrixXd& ao_effective_one_electron_gradient,
    const AoIntegralInput& ao_integral_input,
    const OrbitalPreparationInput& orbital_preparation_input,
    bool compute_outer_response) {
  FusedAoEffectiveOneElectronDirectionalResult result;
  Eigen::MatrixXd symmetrized_pullback_source;
  apply_fused_exact_ao_effective_one_electron_directional_operator(
      inactive_density_matrix,
      ao_effective_one_electron_gradient,
      ao_integral_input,
      orbital_preparation_input,
      compute_outer_response,
      &symmetrized_pullback_source,
      &result.delta_ao_effective_h1e,
      &result.inactive_density_gradient);
  return result;
}

double elapsed_wall_time_seconds(
    const std::chrono::steady_clock::time_point& start_time) {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now() - start_time)
      .count();
}

void throw_if_nonfinite(
    const std::vector<double>& values,
    const char* label) {
  const auto nonfinite_it =
      std::find_if(
          values.begin(),
          values.end(),
          [](double value) { return !std::isfinite(value); });
  if (nonfinite_it == values.end()) {
    return;
  }
  throw std::runtime_error(
      std::string(label) + " contains non-finite values");
}

void throw_if_nonfinite(
    const Eigen::MatrixXd& values,
    const char* label) {
  if (values.allFinite()) {
    return;
  }
  throw std::runtime_error(
      std::string(label) + " contains non-finite values");
}

void throw_if_nonfinite(
    const Eigen::VectorXd& values,
    const char* label) {
  if (values.allFinite()) {
    return;
  }
  throw std::runtime_error(
      std::string(label) + " contains non-finite values");
}

void build_active_space_directional_integrals(
    const VbScfInput& input,
    const ActiveSpaceTwoElectronResult& accepted_active_space_two_electron_result,
    const Eigen::MatrixXd& accepted_active_auxiliary_orbitals,
    const Eigen::MatrixXd& accepted_basis_overlap_times_active_auxiliary_orbitals,
    const Eigen::MatrixXd& accepted_ao_effective_one_electron_times_active_auxiliary_orbitals,
    const Eigen::MatrixXd&
        accepted_ao_effective_one_electron_transpose_times_active_auxiliary_orbitals,
    const Eigen::MatrixXd& delta_active_auxiliary_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& accepted_dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_dense_active_coefficients,
    const Eigen::MatrixXd& delta_ao_effective_h1e_times_active_auxiliary_orbitals,
    std::vector<double>* delta_ao_overlap_matrix_storage,
    std::vector<double>* delta_active_one_electron_matrix_storage,
    ExactPackedActiveTwoElectronDirectionalDerivativeWorkspace* exact_2e_workspace,
    std::vector<double>* delta_packed_active_two_electron_integrals,
    const Eigen::VectorXd* precomputed_delta_packed_active_two_electron) {
  const int n_basis_functions =
      input.orbital_preparation_input.n_basis_functions;
  const int n_active_orbitals =
      input.orbital_preparation_input.n_active_orbitals;
  const std::size_t active_matrix_size =
      n_active_orbitals * n_active_orbitals;
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument(
        "active-space directional integrals require positive dimensions");
  }
  resize_for_overwrite(
      delta_ao_overlap_matrix_storage,
      active_matrix_size);
  resize_for_overwrite(
      delta_active_one_electron_matrix_storage,
      active_matrix_size);
  Eigen::Map<Eigen::MatrixXd> delta_ao_overlap_matrix(
      delta_ao_overlap_matrix_storage->data(),
      n_active_orbitals,
      n_active_orbitals);
  Eigen::Map<Eigen::MatrixXd> delta_active_one_electron_matrix(
      delta_active_one_electron_matrix_storage->data(),
      n_active_orbitals,
      n_active_orbitals);

  // The accepted-point active auxiliary block A is fixed during one Newton
  // linear solve.  Reusing cached `S * A`, `F * A`, and `F^T * A` contractions
  // leaves only active-sized temporaries here while writing the directional
  // `delta SSO` and `delta HHO` directly into caller-owned column-major buffers.
  const Eigen::MatrixXd delta_active_overlap_left =
      delta_active_auxiliary_orbitals.transpose() *
      accepted_basis_overlap_times_active_auxiliary_orbitals;
  delta_ao_overlap_matrix.noalias() =
      delta_active_overlap_left;
  delta_ao_overlap_matrix.noalias() +=
      delta_active_overlap_left.transpose();

  const Eigen::MatrixXd delta_active_one_electron_left =
      delta_active_auxiliary_orbitals.transpose() *
      accepted_ao_effective_one_electron_times_active_auxiliary_orbitals;
  delta_active_one_electron_matrix.noalias() =
      delta_active_one_electron_left;
  delta_active_one_electron_matrix.noalias() +=
      accepted_active_auxiliary_orbitals.transpose() *
      delta_ao_effective_h1e_times_active_auxiliary_orbitals;
  delta_active_one_electron_matrix.noalias() +=
      accepted_ao_effective_one_electron_transpose_times_active_auxiliary_orbitals
          .transpose() *
      delta_active_auxiliary_orbitals;
  if (accepted_dense_active_coefficients.rows() != n_basis_functions ||
      accepted_dense_active_coefficients.cols() != n_active_orbitals ||
      delta_dense_active_coefficients.rows() != n_basis_functions ||
      delta_dense_active_coefficients.cols() != n_active_orbitals) {
    throw std::invalid_argument(
        "dense active coefficient shapes are inconsistent in active-space directional integrals");
  }

  if (precomputed_delta_packed_active_two_electron != nullptr) {
    const std::size_t packed_size =
        packed_active_two_electron_integral_count(n_active_orbitals);
    if (precomputed_delta_packed_active_two_electron->size() !=
        static_cast<Eigen::Index>(packed_size)) {
      throw std::invalid_argument(
          "precomputed block delta GGO has inconsistent dimensions");
    }
    delta_packed_active_two_electron_integrals->assign(
        precomputed_delta_packed_active_two_electron->data(),
        precomputed_delta_packed_active_two_electron->data() + packed_size);
  } else {
    compute_exact_packed_active_two_electron_integral_directional_derivative(
        accepted_dense_active_coefficients,
        delta_dense_active_coefficients,
        input.ao_integral_input,
        n_active_orbitals,
        exact_2e_workspace,
        delta_packed_active_two_electron_integrals,
        &accepted_active_space_two_electron_result);
  }
}

bool exact_ctx_stage1_analytic_core_enabled() {
  return true;
}

}  // namespace

struct ExactHvpOperator::PrecomputedDirection {
  Eigen::VectorXd packed_direction;
  DenseOrbitalTangentContext dense_orbital_tangent_context;
  OrbitalPreparationDirectionalResult orbital_preparation_directional_result;
};

ExactHvpOperator::ExactHvpOperator(
    std::shared_ptr<const AcceptedPointContext> accepted_point_context,
    const VbScfInput* current_input,
    SparseParameterLayout parameter_view,
    const OrbitalChart* nonredundant_space)
    : accepted_point_context_(std::move(accepted_point_context)),
      current_input_(current_input),
      parameter_view_(std::move(parameter_view)),
      nonredundant_space_(nonredundant_space) {
  if (accepted_point_context_ == nullptr) {
    throw std::invalid_argument(
        "accepted-point second-order context must not be null");
  }
  if (current_input_ == nullptr) {
    throw std::invalid_argument("current_input must not be null");
  }
  if (nonredundant_space_ == nullptr) {
    throw std::invalid_argument("nonredundant_space must not be null");
  }
  if (supports_analytic_core_model()) {
    const int n_basis_functions =
        current_input_->orbital_preparation_input.n_basis_functions;
    const int n_active_orbitals =
        current_input_->orbital_preparation_input.n_active_orbitals;
    const int n_inactive_doubly_occupied_orbitals =
        (current_input_->orbital_preparation_input.n_total_electrons -
         current_input_->orbital_preparation_input.n_active_electrons) /
        2;
    const std::size_t ao_matrix_size =
        n_basis_functions * n_basis_functions;
    const std::size_t active_matrix_size =
        n_active_orbitals * n_active_orbitals;
    if (accepted_point_context_->prepared_active_space.orbital_result
                .auxiliary_orbital_matrix.size() == ao_matrix_size &&
        accepted_point_context_->active_orbital_overlap_gradient.size() ==
            active_matrix_size &&
        accepted_point_context_->active_one_electron_gradient.size() ==
            active_matrix_size) {
      const Eigen::Map<const Eigen::MatrixXd> basis_overlap(
          current_input_->orbital_preparation_input.ao_overlap_matrix.data(),
          n_basis_functions,
          n_basis_functions);
      const Eigen::Map<const Eigen::MatrixXd> accepted_auxiliary_matrix(
          accepted_point_context_->prepared_active_space.orbital_result
              .auxiliary_orbital_matrix.data(),
          n_basis_functions,
          n_basis_functions);
      const Eigen::Map<const Eigen::MatrixXd> accepted_ao_effective_h1e(
          accepted_point_context_->prepared_active_space.ao_effective_one_electron_result
              .ao_effective_h1e.data(),
          n_basis_functions,
          n_basis_functions);
      accepted_active_auxiliary_orbitals_ =
          accepted_auxiliary_matrix.middleCols(
              n_inactive_doubly_occupied_orbitals,
              n_active_orbitals);
      accepted_basis_overlap_times_active_auxiliary_orbitals_ =
          basis_overlap * accepted_active_auxiliary_orbitals_;
      accepted_ao_effective_one_electron_times_active_auxiliary_orbitals_ =
          accepted_ao_effective_h1e * accepted_active_auxiliary_orbitals_;
      accepted_ao_effective_one_electron_transpose_times_active_auxiliary_orbitals_ =
          accepted_ao_effective_h1e.transpose() *
          accepted_active_auxiliary_orbitals_;
      const Eigen::Map<const Eigen::MatrixXd> accepted_sso_gradient(
          accepted_point_context_->active_orbital_overlap_gradient.data(),
          n_active_orbitals,
          n_active_orbitals);
      accepted_sso_gradient_symmetric_ =
          accepted_sso_gradient + accepted_sso_gradient.transpose();
      const Eigen::Map<const Eigen::MatrixXd> accepted_hho_gradient(
          accepted_point_context_->active_one_electron_gradient.data(),
          n_active_orbitals,
          n_active_orbitals);
      accepted_hho_gradient_symmetric_ =
          accepted_hho_gradient + accepted_hho_gradient.transpose();
      accepted_active_auxiliary_orbitals_times_hho_gradient_symmetric_ =
          accepted_active_auxiliary_orbitals_ *
          accepted_hho_gradient_symmetric_;
      const auto& accepted_active_space_two_electron_result =
          accepted_point_context_->prepared_active_space.active_space_two_electron_result;
      if (accepted_active_space_two_electron_result.dense_active_coefficients.size() != 0) {
        accepted_dense_active_coefficients_ =
            accepted_active_space_two_electron_result.dense_active_coefficients;
      } else {
        accepted_dense_active_coefficients_ = accepted_active_auxiliary_orbitals_;
      }
      zero_core_hamiltonian_ =
          Eigen::MatrixXd::Zero(n_basis_functions, n_basis_functions);
      accepted_exact_two_electron_cache_ =
          build_exact_packed_active_two_electron_adjoint_cache(
              accepted_point_context_->packed_active_two_electron_gradient,
              accepted_dense_active_coefficients_,
              current_input_->ao_integral_input,
              n_active_orbitals,
              &accepted_point_context_->prepared_active_space
                   .active_space_two_electron_result);
    }

    const auto accepted_orbital_backprop_inputs =
        build_accepted_orbital_backprop_inputs(
            *accepted_point_context_,
            *current_input_);
    accepted_total_active_auxiliary_gradient_ =
        accepted_orbital_backprop_inputs.total_active_auxiliary_gradient;
    accepted_total_inactive_density_gradient_ =
        accepted_orbital_backprop_inputs.total_inactive_density_gradient;

    accepted_orbital_preparation_cache_ =
        std::make_unique<AcceptedOrbitalPreparationCache>(
            build_accepted_orbital_preparation_cache(
                current_input_->orbital_preparation_input,
                accepted_total_active_auxiliary_gradient_,
                accepted_total_inactive_density_gradient_));

    if (accepted_point_context_->same_spin_pair_cache.enabled()) {
      structure_coefficient_blocks_ =
          build_structure_coefficient_blocks(
              current_input_->structure_data.determinant_to_structure_terms,
              current_input_->structure_data.n_structures,
              accepted_point_context_->same_spin_pair_cache.alpha_reuse_table,
              accepted_point_context_->same_spin_pair_cache.beta_reuse_table,
              true);
      accepted_outer_response_context_ =
          build_accepted_outer_response_context(
              current_input_,
              accepted_point_context_.get(),
              &structure_coefficient_blocks_);
    }
  }
}

ExactHvpOperator::~ExactHvpOperator() = default;

Eigen::MatrixXd ExactHvpOperator::apply_reduced_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions,
    HvpComponents components) const {
  ++apply_timing_totals_.batch_apply_count;
  Eigen::MatrixXd responses(
      reduced_directions.rows(),
      reduced_directions.cols());
  if (reduced_directions.cols() <= 1 ||
      !ao_effective_one_electron_graph_available(
          current_input_->ao_integral_input)) {
    for (Eigen::Index column = 0;
         column < reduced_directions.cols();
         ++column) {
      responses.col(column) =
          apply_reduced(reduced_directions.col(column), components);
    }
    return responses;
  }
  if (!supports_analytic_core_model()) {
    throw std::runtime_error(
        "exact_ctx analytic core HVP is unavailable for the current accepted point");
  }

  const int n_basis_functions =
      current_input_->orbital_preparation_input.n_basis_functions;
  const int n_active_orbitals =
      current_input_->orbital_preparation_input.n_active_orbitals;
  const int n_inactive_doubly_occupied_orbitals =
      (current_input_->orbital_preparation_input.n_total_electrons -
       current_input_->orbital_preparation_input.n_active_electrons) /
      2;
  const Eigen::Index ao_matrix_size =
      static_cast<Eigen::Index>(n_basis_functions) * n_basis_functions;
  const Eigen::Index n_directions = reduced_directions.cols();
  Eigen::MatrixXd inactive_density_columns(ao_matrix_size, n_directions);
  Eigen::MatrixXd symmetrized_pullback_columns(ao_matrix_size, n_directions);
  std::vector<PrecomputedDirection> precomputed_directions;
  precomputed_directions.reserve(n_directions);
  std::vector<Eigen::MatrixXd> dense_active_directions;
  dense_active_directions.reserve(n_directions);
  for (Eigen::Index column = 0; column < n_directions; ++column) {
    PrecomputedDirection precomputed;
    precomputed.packed_direction =
        nonredundant_space_->expand_step(reduced_directions.col(column));
    precomputed.dense_orbital_tangent_context =
        build_dense_orbital_tangent_context(
            current_input_->orbital_preparation_input,
            parameter_view_,
            precomputed.packed_direction,
            *accepted_orbital_preparation_cache_);
    precomputed.orbital_preparation_directional_result =
        build_orbital_preparation_directional_result(
            current_input_->orbital_preparation_input,
            precomputed.dense_orbital_tangent_context,
            n_inactive_doubly_occupied_orbitals,
            n_active_orbitals,
            *accepted_orbital_preparation_cache_);
    const auto& directional_result =
        precomputed.orbital_preparation_directional_result;
    dense_active_directions.push_back(
        directional_result.delta_active_auxiliary_orbitals);
    Eigen::MatrixXd pullback_source = directional_result.delta_inactive_density;
    pullback_source.noalias() +=
        (directional_result.delta_active_auxiliary_orbitals *
         accepted_hho_gradient_symmetric_) *
        accepted_active_auxiliary_orbitals_.transpose();
    inactive_density_columns.col(column) = Eigen::Map<const Eigen::VectorXd>(
        directional_result.delta_inactive_density.data(), ao_matrix_size);
    const Eigen::MatrixXd symmetrized_pullback =
        pullback_source + pullback_source.transpose();
    symmetrized_pullback_columns.col(column) =
        Eigen::Map<const Eigen::VectorXd>(
            symmetrized_pullback.data(), ao_matrix_size);
    precomputed_directions.push_back(std::move(precomputed));
  }

  const auto batch_h1e_start_time = std::chrono::steady_clock::now();
  Eigen::MatrixXd delta_h1e_columns;
  Eigen::MatrixXd inactive_density_gradient_columns;
  apply_fused_ao_effective_one_electron_graph_batch(
      inactive_density_columns,
      symmetrized_pullback_columns,
      current_input_->ao_integral_input,
      choose_exact_ctx_ao_h1e_threads(
          current_input_->orbital_preparation_input,
          components.outer_response),
      &delta_h1e_columns,
      &inactive_density_gradient_columns);
  for (Eigen::Index column = 0; column < n_directions; ++column) {
    Eigen::Map<Eigen::MatrixXd> delta_h1e(
        delta_h1e_columns.col(column).data(),
        n_basis_functions,
        n_basis_functions);
    for (int row = 0; row < n_basis_functions; ++row) {
      for (int matrix_column = 0; matrix_column <= row; ++matrix_column) {
        delta_h1e(row, matrix_column) += delta_h1e(matrix_column, row);
        delta_h1e(matrix_column, row) = delta_h1e(row, matrix_column);
      }
    }
  }
  const double batch_h1e_seconds =
      elapsed_wall_time_seconds(batch_h1e_start_time);
  apply_timing_totals_.ao_effective_one_electron_fused_wall_time_seconds +=
      batch_h1e_seconds;
  apply_timing_totals_.total_apply_wall_time_seconds += batch_h1e_seconds;

  Eigen::MatrixXd delta_packed_active_two_electron_columns;
  std::vector<ExactCtxPairMatrix> directional_pair_products;
  if (components.outer_response) {
    const auto batch_active_two_electron_start_time =
        std::chrono::steady_clock::now();
    delta_packed_active_two_electron_columns =
        compute_exact_packed_active_two_electron_integral_directional_derivative_batch(
            accepted_exact_two_electron_cache_,
            dense_active_directions,
            current_input_->ao_integral_input,
            &directional_pair_products);
    const double batch_active_two_electron_seconds =
        elapsed_wall_time_seconds(batch_active_two_electron_start_time);
    apply_timing_totals_
        .outer_response_active_space_integrals_wall_time_seconds +=
        batch_active_two_electron_seconds;
    apply_timing_totals_.total_apply_wall_time_seconds +=
        batch_active_two_electron_seconds;
  }

  for (Eigen::Index column = 0;
       column < reduced_directions.cols();
       ++column) {
    const Eigen::VectorXd delta_h1e = delta_h1e_columns.col(column);
    const Eigen::VectorXd inactive_density_gradient =
        inactive_density_gradient_columns.col(column);
    const Eigen::VectorXd delta_packed_active_two_electron =
        components.outer_response
            ? delta_packed_active_two_electron_columns.col(column)
            : Eigen::VectorXd();
    responses.col(column) = apply_reduced_impl(
        reduced_directions.col(column),
        components,
        &delta_h1e,
        &inactive_density_gradient,
        components.outer_response
            ? &delta_packed_active_two_electron
            : nullptr,
        components.outer_response
            ? &directional_pair_products[column]
            : nullptr,
        &precomputed_directions[column]);
  }
  return responses;
}

Eigen::VectorXd ExactHvpOperator::apply_reduced(
    const Eigen::VectorXd& reduced_direction,
    HvpComponents components) const {
  return apply_reduced_impl(
      reduced_direction,
      components,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr);
}

Eigen::VectorXd ExactHvpOperator::apply_reduced_impl(
    const Eigen::VectorXd& reduced_direction,
    HvpComponents components,
    const Eigen::VectorXd* precomputed_delta_ao_effective_h1e,
    const Eigen::VectorXd* precomputed_inactive_density_gradient,
    const Eigen::VectorXd* precomputed_delta_packed_active_two_electron,
    const ExactCtxPairMatrix* precomputed_directional_pair_products,
    const PrecomputedDirection* precomputed_direction) const {
  const auto apply_start_time = std::chrono::steady_clock::now();
  auto record_apply_wall_time = [&]() {
    ++apply_timing_totals_.apply_count;
    apply_timing_totals_.total_apply_wall_time_seconds +=
        elapsed_wall_time_seconds(apply_start_time);
  };
  Eigen::VectorXd response =
      Eigen::VectorXd::Zero(reduced_direction.size());

  if (!supports_analytic_core_model()) {
    throw std::runtime_error(
        "exact_ctx analytic core HVP is unavailable for the current accepted point");
  }

  const int n_basis_functions =
      current_input_->orbital_preparation_input.n_basis_functions;
  const int n_active_orbitals =
      current_input_->orbital_preparation_input.n_active_orbitals;
  const int n_inactive_doubly_occupied_orbitals =
      (current_input_->orbital_preparation_input.n_total_electrons -
       current_input_->orbital_preparation_input.n_active_electrons) /
      2;
  const std::size_t ao_matrix_size =
      n_basis_functions * n_basis_functions;

  const auto core_setup_start_time = std::chrono::steady_clock::now();
  Eigen::VectorXd local_packed_direction;
  if (precomputed_direction == nullptr) {
    local_packed_direction =
        nonredundant_space_->expand_step(reduced_direction);
  }
  const Eigen::VectorXd& packed_direction =
      precomputed_direction != nullptr
          ? precomputed_direction->packed_direction
          : local_packed_direction;
  if (packed_direction.norm() == 0.0) {
    record_apply_wall_time();
    return Eigen::VectorXd::Zero(reduced_direction.size());
  }

  const AcceptedOrbitalPreparationCache* orbital_preparation_cache =
      accepted_orbital_preparation_cache_.get();
  DenseOrbitalTangentContext local_dense_orbital_tangent_context;
  if (precomputed_direction == nullptr) {
    local_dense_orbital_tangent_context =
        build_dense_orbital_tangent_context(
            current_input_->orbital_preparation_input,
            parameter_view_,
            packed_direction,
            *orbital_preparation_cache);
  }
  const DenseOrbitalTangentContext& dense_orbital_tangent_context =
      precomputed_direction != nullptr
          ? precomputed_direction->dense_orbital_tangent_context
          : local_dense_orbital_tangent_context;
  const Eigen::VectorXd input_retract_tangent =
      components.fixed_upstream_pullback
          ? nonredundant_space_->expand_retract_input_tangent(
                current_input_->orbital_preparation_input,
                reduced_direction)
          : Eigen::VectorXd();
  OrbitalPreparationDirectionalResult
      local_orbital_preparation_directional_result;
  if (precomputed_direction == nullptr) {
    local_orbital_preparation_directional_result =
        build_orbital_preparation_directional_result(
            current_input_->orbital_preparation_input,
            dense_orbital_tangent_context,
            n_inactive_doubly_occupied_orbitals,
            n_active_orbitals,
            *orbital_preparation_cache);
  }
  const OrbitalPreparationDirectionalResult&
      orbital_preparation_directional_result =
          precomputed_direction != nullptr
              ? precomputed_direction->orbital_preparation_directional_result
              : local_orbital_preparation_directional_result;

  const Eigen::Map<const Eigen::MatrixXd> basis_overlap(
      current_input_->orbital_preparation_input.ao_overlap_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const Eigen::MatrixXd> ao_effective_h1e(
      accepted_point_context_->prepared_active_space.ao_effective_one_electron_result
          .ao_effective_h1e.data(),
      n_basis_functions,
      n_basis_functions);
  const auto& delta_dense_active_coefficients =
      orbital_preparation_directional_result.delta_active_auxiliary_orbitals;
  apply_timing_totals_.core_setup_wall_time_seconds +=
      elapsed_wall_time_seconds(core_setup_start_time);
  // `F11` and `delta F11` are explicitly symmetrized in the AO-H1E builder, so
  // the directional active-space matrix gradient only needs the two distinct
  // left contractions `delta F11 * T_active` and `F11 * delta T_active`.
  const Eigen::MatrixXd delta_active_times_hho_symmetric =
      orbital_preparation_directional_result.delta_active_auxiliary_orbitals *
      accepted_hho_gradient_symmetric_;
  Eigen::MatrixXd total_ao_effective_one_electron_direction =
      orbital_preparation_directional_result.delta_inactive_density;
  total_ao_effective_one_electron_direction.noalias() +=
      delta_active_times_hho_symmetric *
      accepted_active_auxiliary_orbitals_.transpose();
  const double* delta_ao_effective_h1e_data = nullptr;
  const double* inactive_density_gradient_data = nullptr;
  if (precomputed_delta_ao_effective_h1e != nullptr ||
      precomputed_inactive_density_gradient != nullptr) {
    if (precomputed_delta_ao_effective_h1e == nullptr ||
        precomputed_inactive_density_gradient == nullptr ||
        precomputed_delta_ao_effective_h1e->size() !=
            static_cast<Eigen::Index>(ao_matrix_size) ||
        precomputed_inactive_density_gradient->size() !=
            static_cast<Eigen::Index>(ao_matrix_size)) {
      throw std::invalid_argument(
          "precomputed block AO-H1E response has inconsistent dimensions");
    }
    delta_ao_effective_h1e_data =
        precomputed_delta_ao_effective_h1e->data();
    inactive_density_gradient_data =
        precomputed_inactive_density_gradient->data();
  } else {
    const auto ao_effective_one_electron_fused_start_time =
        std::chrono::steady_clock::now();
    apply_fused_exact_ao_effective_one_electron_directional_operator(
        orbital_preparation_directional_result.delta_inactive_density,
        total_ao_effective_one_electron_direction,
        current_input_->ao_integral_input,
        current_input_->orbital_preparation_input,
        components.outer_response,
        &ao_h1e_symmetrized_gradient_workspace_,
        &ao_h1e_delta_h1e_workspace_,
        &ao_h1e_inactive_density_gradient_workspace_,
        &ao_h1e_partial_delta_h1e_workspaces_,
        &ao_h1e_partial_inactive_density_gradient_workspaces_);
    delta_ao_effective_h1e_data = ao_h1e_delta_h1e_workspace_.data();
    inactive_density_gradient_data =
        ao_h1e_inactive_density_gradient_workspace_.data();
    apply_timing_totals_.ao_effective_one_electron_fused_wall_time_seconds +=
        elapsed_wall_time_seconds(ao_effective_one_electron_fused_start_time);
  }
  const Eigen::Map<const Eigen::MatrixXd> delta_ao_effective_h1e(
      delta_ao_effective_h1e_data,
      n_basis_functions,
      n_basis_functions);

  const bool compute_outer_response = components.outer_response;
  std::vector<double> combined_core_orbital_value_gradient;
  Eigen::MatrixXd delta_ao_effective_h1e_times_active_auxiliary_orbitals;
  if (compute_outer_response) {
    // Full HVPs need `delta F11 * A_active` both for the direct-core
    // `delta F11 * A_active * G_hho` pullback and for outer-response
    // `delta HHO = A_active^T * delta F11 * A_active`.  Materializing it once
    // avoids repeating the same AO-by-active contraction in the two stages.
    delta_ao_effective_h1e_times_active_auxiliary_orbitals.noalias() =
        delta_ao_effective_h1e * accepted_active_auxiliary_orbitals_;
  }

  if (compute_outer_response) {
    const auto outer_response_start_time =
        std::chrono::steady_clock::now();

    const auto active_space_integrals_start_time =
        std::chrono::steady_clock::now();
    build_active_space_directional_integrals(
        *current_input_,
        accepted_point_context_->prepared_active_space.active_space_two_electron_result,
        accepted_active_auxiliary_orbitals_,
        accepted_basis_overlap_times_active_auxiliary_orbitals_,
        accepted_ao_effective_one_electron_times_active_auxiliary_orbitals_,
        accepted_ao_effective_one_electron_transpose_times_active_auxiliary_orbitals_,
        orbital_preparation_directional_result.delta_active_auxiliary_orbitals,
        accepted_dense_active_coefficients_,
        delta_dense_active_coefficients,
        delta_ao_effective_h1e_times_active_auxiliary_orbitals,
        &outer_response_delta_ao_overlap_matrix_workspace_,
        &outer_response_delta_active_one_electron_matrix_workspace_,
        &outer_response_exact_two_electron_directional_workspace_,
        &outer_response_delta_packed_active_two_electron_workspace_,
        precomputed_delta_packed_active_two_electron);
    apply_timing_totals_.outer_response_active_space_integrals_wall_time_seconds +=
        elapsed_wall_time_seconds(active_space_integrals_start_time);

    const auto structure_matrices_start_time =
        std::chrono::steady_clock::now();
    // The polynomial pair response is consumed first by the projected
    // structure action and later by the local same-spin adjoint. Build it once
    // for this HVP direction so both stages share the same cofactor actions.
    const ActiveSpaceIntegralDirectionView active_space_integral_direction{
        outer_response_delta_ao_overlap_matrix_workspace_,
        outer_response_delta_active_one_electron_matrix_workspace_,
        outer_response_delta_packed_active_two_electron_workspace_};
    const SameSpinDirectionalPairCache directional_pair_cache =
        build_same_spin_directional_pair_cache(
            accepted_point_context_->same_spin_pair_cache,
            current_input_->orbital_preparation_input.n_active_orbitals,
            active_space_integral_direction);
    const auto projected_directional_structure_matrices =
        build_projected_structure_direction(
            accepted_outer_response_context_,
            active_space_integral_direction,
            directional_pair_cache);
    apply_timing_totals_.outer_response_structure_matrices_wall_time_seconds +=
        elapsed_wall_time_seconds(structure_matrices_start_time);

    const auto eigensystem_start_time = std::chrono::steady_clock::now();
    const auto directional_selected_state_response =
        accepted_outer_response_context_.selected_state_eigen_response_operator.apply(
            projected_directional_structure_matrices);
    apply_timing_totals_.outer_response_eigensystem_wall_time_seconds +=
        elapsed_wall_time_seconds(eigensystem_start_time);

    const auto active_gradient_start_time = std::chrono::steady_clock::now();
    const SelectedStateDeterminantMatrices directional_selected_states =
        build_selected_state_determinant_matrices_from_selected_columns(
            current_input_->structure_data,
            directional_selected_state_response.delta_selected_eigenvector_matrix,
            accepted_point_context_->selected_state_indices,
            accepted_point_context_->normalized_state_weights,
            accepted_point_context_->same_spin_pair_cache);
    ActiveSpaceGradientDirection directional_active_space_gradient =
        build_active_space_gradient_direction_from_outer_response(
            *current_input_, *accepted_point_context_,
            active_space_integral_direction,
            directional_selected_states,
            directional_selected_state_response.delta_selected_eigenvalues,
            directional_pair_cache);
    apply_timing_totals_.outer_response_active_gradient_wall_time_seconds +=
        elapsed_wall_time_seconds(active_gradient_start_time);
    validate_outer_response_active_gradient(
        directional_active_space_gradient);

    const auto orbital_pullback_start_time =
        std::chrono::steady_clock::now();
    const std::vector<double> outer_response_orbital_value_gradient =
        build_orbital_value_gradient_from_active_space_gradient_direction(
            *current_input_,
            *accepted_point_context_,
            directional_active_space_gradient,
            *orbital_preparation_cache,
            accepted_exact_two_electron_cache_.n_basis_functions > 0
                ? &accepted_exact_two_electron_cache_
                : nullptr,
            &outer_response_symmetric_active_overlap_gradient_workspace_,
            &outer_response_symmetric_active_one_electron_gradient_workspace_);
    add_orbital_value_gradient_in_place(
        &combined_core_orbital_value_gradient,
        outer_response_orbital_value_gradient,
        "outer-response");
    apply_timing_totals_.outer_response_orbital_pullback_wall_time_seconds +=
        elapsed_wall_time_seconds(orbital_pullback_start_time);

    apply_timing_totals_.outer_response_wall_time_seconds +=
        elapsed_wall_time_seconds(outer_response_start_time);
  }
  if (components.direct_core_response) {
    Eigen::MatrixXd delta_auxiliary_active_gradient =
        basis_overlap *
        orbital_preparation_directional_result.delta_active_auxiliary_orbitals *
        accepted_sso_gradient_symmetric_;
    if (compute_outer_response) {
      delta_auxiliary_active_gradient.noalias() +=
          delta_ao_effective_h1e_times_active_auxiliary_orbitals *
          accepted_hho_gradient_symmetric_;
    } else {
      delta_auxiliary_active_gradient.noalias() +=
          delta_ao_effective_h1e *
          accepted_active_auxiliary_orbitals_times_hho_gradient_symmetric_;
    }
    delta_auxiliary_active_gradient.noalias() +=
        ao_effective_h1e *
        delta_active_times_hho_symmetric;
    const auto active_two_electron_start_time =
        std::chrono::steady_clock::now();
    Eigen::MatrixXd dense_active_two_electron_gradient_direction_storage;
    const Eigen::MatrixXd* dense_active_two_electron_gradient_direction =
        nullptr;
    if (accepted_exact_two_electron_cache_.n_basis_functions > 0) {
      // Fused path: reuse forward K*mixed from outer response to skip
      // one apply_exact_ao_pair_kernel call (~500M FLOPs) per HVP.
      const ExactCtxPairMatrix* directional_pair_products =
          precomputed_directional_pair_products != nullptr
              ? precomputed_directional_pair_products
              : &outer_response_exact_two_electron_directional_workspace_
                     .directional_pair_products;
      const bool can_fuse =
          compute_outer_response && directional_pair_products->size() > 0;
      if (can_fuse) {
        apply_exact_packed_active_two_electron_adjoint_hessian_vector_fused(
            accepted_exact_two_electron_cache_,
            delta_dense_active_coefficients,
            current_input_->ao_integral_input,
            *directional_pair_products,
            &accepted_exact_two_electron_apply_workspace_,
            &accepted_exact_two_electron_apply_workspace_
                 .dense_active_gradient_direction);
      } else {
        apply_exact_packed_active_two_electron_adjoint_hessian_vector(
            accepted_exact_two_electron_cache_,
            delta_dense_active_coefficients,
            current_input_->ao_integral_input,
            &accepted_exact_two_electron_apply_workspace_,
            &accepted_exact_two_electron_apply_workspace_
                 .dense_active_gradient_direction);
      }
      dense_active_two_electron_gradient_direction =
          &accepted_exact_two_electron_apply_workspace_
               .dense_active_gradient_direction;
    } else {
      dense_active_two_electron_gradient_direction_storage =
          apply_exact_packed_active_two_electron_adjoint_hessian_vector(
              accepted_point_context_->packed_active_two_electron_gradient,
              accepted_dense_active_coefficients_,
              delta_dense_active_coefficients,
              current_input_->ao_integral_input,
              n_active_orbitals,
              &accepted_point_context_->prepared_active_space
                   .active_space_two_electron_result);
      dense_active_two_electron_gradient_direction =
          &dense_active_two_electron_gradient_direction_storage;
    }
    apply_timing_totals_.active_two_electron_wall_time_seconds +=
        elapsed_wall_time_seconds(active_two_electron_start_time);
    if (dense_active_two_electron_gradient_direction != nullptr) {
      delta_auxiliary_active_gradient.noalias() +=
          *dense_active_two_electron_gradient_direction;
    }

    Eigen::MatrixXd total_inactive_density_direction =
        delta_ao_effective_h1e;
    const Eigen::Map<const Eigen::MatrixXd> ao_backpropagated_inactive_density(
        inactive_density_gradient_data,
        n_basis_functions,
        n_basis_functions);
    total_inactive_density_direction.noalias() +=
        ao_backpropagated_inactive_density;

    const auto orbital_backprop_start_time =
        std::chrono::steady_clock::now();
    const std::vector<double> orbital_value_gradient =
        backpropagate_active_space_orbital_gradient(
            delta_auxiliary_active_gradient,
            total_inactive_density_direction,
            current_input_->orbital_preparation_input,
            accepted_point_context_->prepared_active_space.orbital_result,
            *orbital_preparation_cache);
    add_orbital_value_gradient_in_place(
        &combined_core_orbital_value_gradient,
        orbital_value_gradient,
        "direct-core");
    apply_timing_totals_.orbital_backprop_wall_time_seconds +=
        elapsed_wall_time_seconds(orbital_backprop_start_time);

  }

  if (components.fixed_upstream_pullback &&
      accepted_total_active_auxiliary_gradient_.rows() == n_basis_functions &&
      accepted_total_active_auxiliary_gradient_.cols() == n_active_orbitals &&
      accepted_total_inactive_density_gradient_.size() == ao_matrix_size) {
      const auto fixed_upstream_pullback_start_time =
          std::chrono::steady_clock::now();
      const std::vector<double> fixed_upstream_orbital_value_gradient =
              apply_fixed_upstream_orbital_pullback_direction(
                  current_input_->orbital_preparation_input,
                  dense_orbital_tangent_context,
                  accepted_total_active_auxiliary_gradient_,
                  orbital_preparation_directional_result
                      .basis_overlap_times_delta_active_orbitals,
                  accepted_total_inactive_density_gradient_,
                  input_retract_tangent,
                  *orbital_preparation_cache);
      add_orbital_value_gradient_in_place(
          &combined_core_orbital_value_gradient,
          fixed_upstream_orbital_value_gradient,
          "fixed-upstream");
      apply_timing_totals_.fixed_upstream_pullback_wall_time_seconds +=
          elapsed_wall_time_seconds(fixed_upstream_pullback_start_time);
  }

  if (!combined_core_orbital_value_gradient.empty()) {
    // Full exact_ctx matvecs used to project the direct-core/fixed-upstream
    // pullback and the outer-response pullback separately.  Both contributions
    // live in the same raw sparse-orbital coefficient chart, so combining them
    // before `gather_from_full + project_reduced_gradient` removes one full
    // packed/reduced projection from every HVP apply without changing the
    // accepted-point nonredundant semantics.
    const Eigen::VectorXd packed_response =
        parameter_view_.gather_from_full(combined_core_orbital_value_gradient);
    response += nonredundant_space_->project_reduced_gradient(packed_response);

    // Geometric pullback (retraction-induced Hessian) is not needed for the
    // production U_p tangent path.  Per-orbital physical retraction is linear:
    // OLD code (retired): gradient scatter/gather + apply_geometric_pullback call.
  }

  record_apply_wall_time();
  return response;
}

bool ExactHvpOperator::supports_analytic_core_model() const noexcept {
  if (accepted_point_context_ == nullptr ||
      current_input_ == nullptr ||
      nonredundant_space_ == nullptr) {
    return false;
  }
  if (!exact_ctx_stage1_analytic_core_enabled()) {
    return false;
  }
  if (current_input_->standard_two_electron_mode ==
      StandardTwoElectronMode::ResolutionOfIdentity) {
    return false;
  }
  if (current_input_->ao_integral_input.ao_two_electron_integral_values.empty() ||
      current_input_->ao_integral_input.ao_two_electron_integral_indices.empty()) {
    return false;
  }
  const int n_active_orbitals =
      current_input_->orbital_preparation_input.n_active_orbitals;
  return n_active_orbitals > 0 &&
      has_active_matrix_gradient(*accepted_point_context_, n_active_orbitals);
}

ExactHvpOperator::Diagnostics
ExactHvpOperator::diagnostics() const {
  Diagnostics info;
  if (accepted_point_context_ == nullptr || nonredundant_space_ == nullptr) {
    return info;
  }
  info.supports_analytic_core_model = supports_analytic_core_model();
  info.outer_response_enabled = true;
  info.outer_response_local_only_approximation = false;
  info.outer_response_energy_only_approximation = false;
  info.used_reduced_curvature_diagonal =
      nonredundant_space_->has_reduced_curvature_diagonal();
  info.has_same_spin_matrix_form =
      accepted_point_context_->use_full_matrix_form_adjoint;
  info.has_opposite_spin_matrix_form =
      accepted_point_context_->use_matrix_form_opposite_spin;
  info.n_selected_states =
      static_cast<int>(accepted_point_context_->selected_state_indices.size());
  info.n_active_orbitals = accepted_point_context_->n_active_orbitals;
  info.n_blocks = nonredundant_space_->n_blocks();
  info.apply_count = apply_timing_totals_.apply_count;
  info.batch_apply_count = apply_timing_totals_.batch_apply_count;
  info.total_apply_wall_time_seconds =
      apply_timing_totals_.total_apply_wall_time_seconds;
  info.core_setup_wall_time_seconds =
      apply_timing_totals_.core_setup_wall_time_seconds;
  info.ao_effective_one_electron_build_wall_time_seconds =
      apply_timing_totals_.ao_effective_one_electron_build_wall_time_seconds;
  info.ao_effective_one_electron_fused_wall_time_seconds =
      apply_timing_totals_.ao_effective_one_electron_fused_wall_time_seconds;
  info.active_two_electron_wall_time_seconds =
      apply_timing_totals_.active_two_electron_wall_time_seconds;
  info.ao_effective_one_electron_backprop_wall_time_seconds =
      apply_timing_totals_.ao_effective_one_electron_backprop_wall_time_seconds;
  info.orbital_backprop_wall_time_seconds =
      apply_timing_totals_.orbital_backprop_wall_time_seconds;
  info.fixed_upstream_pullback_wall_time_seconds =
      apply_timing_totals_.fixed_upstream_pullback_wall_time_seconds;
  info.outer_response_wall_time_seconds =
      apply_timing_totals_.outer_response_wall_time_seconds;
  info.outer_response_active_space_integrals_wall_time_seconds =
      apply_timing_totals_.outer_response_active_space_integrals_wall_time_seconds;
  info.outer_response_structure_matrices_wall_time_seconds =
      apply_timing_totals_.outer_response_structure_matrices_wall_time_seconds;
  info.outer_response_eigensystem_wall_time_seconds =
      apply_timing_totals_.outer_response_eigensystem_wall_time_seconds;
  info.outer_response_pair_weights_wall_time_seconds =
      apply_timing_totals_.outer_response_pair_weights_wall_time_seconds;
  info.outer_response_active_gradient_wall_time_seconds =
      apply_timing_totals_.outer_response_active_gradient_wall_time_seconds;
  info.outer_response_orbital_pullback_wall_time_seconds =
      apply_timing_totals_.outer_response_orbital_pullback_wall_time_seconds;
  return info;
}

}  // namespace xmvb::vb
