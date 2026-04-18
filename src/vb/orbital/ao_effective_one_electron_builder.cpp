#include "vb/orbital/ao_effective_one_electron_builder.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "vb/matrices/eigen_matrix_storage_utils.hpp"
#include "vb/orbital/ao_effective_one_electron_graph_operator.hpp"
#include "vb/orbital/ao_effective_one_electron_ri_operator.hpp"

namespace xmvb::vb {

namespace {

constexpr double kIntegralSymmetryMultipliers[] = {
    1.0,
    0.5,
    0.25,
    0.125,
};

std::vector<std::size_t> build_column_offsets(int n_basis_functions) {
  std::vector<std::size_t> column_offsets(xmvb::to_size(n_basis_functions), 0);
  for (int column = 0; column < n_basis_functions; ++column) {
    column_offsets[xmvb::to_size(column)] =
        xmvb::to_size(column) * n_basis_functions;
  }
  return column_offsets;
}

int choose_ao_effective_one_electron_graph_threads(
    int requested_threads,
    int n_basis_functions) {
  if (requested_threads <= 1) {
    return 1;
  }

  // Medium AO spaces are bandwidth-bound. A small row-parallel graph sweep is
  // usually faster than the legacy integral scatter with dozens of threads and
  // avoids allocating one dense AO matrix per OpenMP worker.
  const int workload_limited_threads =
      std::max(1, n_basis_functions / 32);
  return std::max(1, std::min(requested_threads, workload_limited_threads));
}

template <bool ValidateIntegralIndices, bool UsePrecomputedSymmetryShifts, bool UseLinearIndexCache>
inline void accumulate_ao_effective_one_electron_integral(
    std::size_t integral_index,
    const double* inactive_density_matrix,
    const double* ao_two_electron_integral_values,
    const int* ao_two_electron_integral_indices,
    const std::uint8_t* ao_two_electron_integral_symmetry_shifts,
    const int* ao_effective_one_electron_linear_indices,
    const std::size_t* column_offsets,
    int n_basis_functions,
    std::atomic<int>& invalid_integral_index,
    double* local_g11) {
  int i = 0;
  int j = 0;
  int k = 0;
  int l = 0;
  if constexpr (ValidateIntegralIndices ||
                !UsePrecomputedSymmetryShifts ||
                !UseLinearIndexCache) {
    const int* integral_indices =
        ao_two_electron_integral_indices + integral_index * 4;
    i = integral_indices[0];
    j = integral_indices[1];
    k = integral_indices[2];
    l = integral_indices[3];
  }

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
    ij_index = xmvb::to_size(linear_indices[0]);
    kl_index = xmvb::to_size(linear_indices[1]);
    ik_index = xmvb::to_size(linear_indices[2]);
    jl_index = xmvb::to_size(linear_indices[3]);
    il_index = xmvb::to_size(linear_indices[4]);
    jk_index = xmvb::to_size(linear_indices[5]);
    lj_index = xmvb::to_size(linear_indices[6]);
    ki_index = xmvb::to_size(linear_indices[7]);
    kj_index = xmvb::to_size(linear_indices[8]);
    li_index = xmvb::to_size(linear_indices[9]);
  } else {
    const std::size_t col_i = column_offsets[xmvb::to_size(i)];
    const std::size_t col_j = column_offsets[xmvb::to_size(j)];
    const std::size_t col_k = column_offsets[xmvb::to_size(k)];
    const std::size_t col_l = column_offsets[xmvb::to_size(l)];

    ij_index = col_j + xmvb::to_size(i);
    kl_index = col_l + xmvb::to_size(k);
    ik_index = col_k + xmvb::to_size(i);
    jl_index = col_l + xmvb::to_size(j);
    il_index = col_l + xmvb::to_size(i);
    jk_index = col_k + xmvb::to_size(j);
    lj_index = col_j + xmvb::to_size(l);
    ki_index = col_i + xmvb::to_size(k);
    kj_index = col_j + xmvb::to_size(k);
    li_index = col_i + xmvb::to_size(l);
  }

  const double density_ij = inactive_density_matrix[ij_index];
  const double density_kl = inactive_density_matrix[kl_index];
  const double density_lj = inactive_density_matrix[lj_index];
  const double density_ki = inactive_density_matrix[ki_index];
  const double density_kj = inactive_density_matrix[kj_index];
  const double density_li = inactive_density_matrix[li_index];
  const double scaled_two_electron_value = two_electron_value * 4.0;

  local_g11[ij_index] += density_kl * scaled_two_electron_value;
  local_g11[kl_index] += density_ij * scaled_two_electron_value;
  local_g11[ik_index] -= density_lj * two_electron_value;
  local_g11[jl_index] -= density_ki * two_electron_value;
  local_g11[il_index] -= density_kj * two_electron_value;
  local_g11[jk_index] -= density_li * two_electron_value;
}

AoEffectiveOneElectronResult build_ao_effective_one_electron_ri(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_matrix,
    const std::vector<double>& ao_core_hamiltonian_matrix,
    const LibcintRiIntegralProviderResult& ri_integral_provider_result,
    int n_basis_functions) {
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }

  const std::size_t matrix_size =
      xmvb::to_size(n_basis_functions) * n_basis_functions;
  if (inactive_density_matrix.rows() != n_basis_functions ||
      inactive_density_matrix.cols() != n_basis_functions ||
      ao_core_hamiltonian_matrix.size() != matrix_size) {
    throw std::invalid_argument("AO matrix sizes do not match n_basis_functions");
  }
  if (ri_integral_provider_result.n_basis_functions != n_basis_functions) {
    throw std::invalid_argument("RI basis-function count mismatch");
  }

  const Eigen::Map<const Eigen::MatrixXd> core_hamiltonian(
      ao_core_hamiltonian_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  std::vector<double> g11_storage =
      apply_ao_effective_one_electron_ri_operator(
          flatten_matrix_column_major(inactive_density_matrix),
          ri_integral_provider_result,
          n_basis_functions,
          {.attempt_spectral_factorization = true});
  const Eigen::Map<const Eigen::MatrixXd> g11(
      g11_storage.data(),
      n_basis_functions,
      n_basis_functions);

  AoEffectiveOneElectronResult result;
  result.ao_coulomb_exchange_matrix.assign(
      g11.data(),
      g11.data() + g11.size());

  Eigen::MatrixXd ao_effective_h1e = core_hamiltonian;
  ao_effective_h1e.noalias() += g11;
  result.ao_effective_h1e.assign(
      ao_effective_h1e.data(),
      ao_effective_h1e.data() + ao_effective_h1e.size());
  return result;
}

AoEffectiveOneElectronResult build_ao_effective_one_electron_ri(
    const AoEffectiveOneElectronRiLowRankFactors& inactive_density_factors,
    const std::vector<double>& ao_core_hamiltonian_matrix,
    const LibcintRiIntegralProviderResult& ri_integral_provider_result,
    int n_basis_functions) {
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }

  const std::size_t matrix_size =
      xmvb::to_size(n_basis_functions) * n_basis_functions;
  if (ao_core_hamiltonian_matrix.size() != matrix_size) {
    throw std::invalid_argument("AO core Hamiltonian size does not match n_basis_functions");
  }
  if (ri_integral_provider_result.n_basis_functions != n_basis_functions) {
    throw std::invalid_argument("RI basis-function count mismatch");
  }

  const Eigen::Map<const Eigen::MatrixXd> core_hamiltonian(
      ao_core_hamiltonian_matrix.data(),
      n_basis_functions,
      n_basis_functions);

  // The inactive density is already available as `P11 = F F^T` from orbital
  // preparation.  Feeding those occupied-space factors directly into the RI
  // operator avoids the otherwise redundant AO-level eigendecomposition.
  std::vector<double> g11_storage =
      apply_ao_effective_one_electron_ri_operator(
          inactive_density_factors,
          ri_integral_provider_result,
          n_basis_functions);
  const Eigen::Map<const Eigen::MatrixXd> g11(
      g11_storage.data(),
      n_basis_functions,
      n_basis_functions);

  AoEffectiveOneElectronResult result;
  result.ao_coulomb_exchange_matrix.assign(
      g11.data(),
      g11.data() + g11.size());

  Eigen::MatrixXd ao_effective_h1e = core_hamiltonian;
  ao_effective_h1e.noalias() += g11;
  result.ao_effective_h1e.assign(
      ao_effective_h1e.data(),
      ao_effective_h1e.data() + ao_effective_h1e.size());
  return result;
}

}  // namespace

AoEffectiveOneElectronResult build_ao_effective_one_electron(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_matrix,
    const std::vector<double>& ao_core_hamiltonian_matrix,
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_integral_indices,
    const std::uint8_t* ao_two_electron_integral_symmetry_shifts,
    const int* ao_effective_one_electron_linear_indices,
    int n_basis_functions,
    bool validate_integral_indices) {
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }

  const std::size_t matrix_size =
      xmvb::to_size(n_basis_functions) * n_basis_functions;
  if (inactive_density_matrix.rows() != n_basis_functions ||
      inactive_density_matrix.cols() != n_basis_functions ||
      ao_core_hamiltonian_matrix.size() != matrix_size) {
    throw std::invalid_argument("AO matrix sizes do not match n_basis_functions");
  }
  if (ao_two_electron_integral_indices.size() != ao_two_electron_integral_values.size() * 4) {
    throw std::invalid_argument("AO two-electron index/value sizes are inconsistent");
  }

  Eigen::Map<const Eigen::MatrixXd> hhf(
      ao_core_hamiltonian_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  std::vector<double> g11_storage(matrix_size, 0.0);
  std::atomic<int> invalid_integral_index(-1);
  const double* inactive_density_matrix_data = inactive_density_matrix.data();
  const double* ao_two_electron_integral_values_data =
      ao_two_electron_integral_values.data();
  const int* ao_two_electron_integral_indices_data =
      ao_two_electron_integral_indices.data();
  if (ao_two_electron_integral_symmetry_shifts == nullptr &&
      !validate_integral_indices) {
    // Trusted input can still omit symmetry-shift caches. Fall back to the
    // branchy scaling path without changing numerical behavior.
  }
  std::vector<std::size_t> column_offsets;
  if (ao_effective_one_electron_linear_indices == nullptr) {
    column_offsets = build_column_offsets(n_basis_functions);
  }
  const std::size_t* column_offsets_data =
      ao_effective_one_electron_linear_indices == nullptr
          ? column_offsets.data()
          : nullptr;

  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  std::vector<std::vector<double>> partial_g11;
  if (n_threads <= 1) {
    if (ao_effective_one_electron_linear_indices != nullptr &&
        validate_integral_indices &&
        ao_two_electron_integral_symmetry_shifts != nullptr) {
      for (std::size_t integral_index = 0;
           integral_index < ao_two_electron_integral_values.size();
           ++integral_index) {
        accumulate_ao_effective_one_electron_integral<true, true, true>(
            integral_index,
            inactive_density_matrix_data,
            ao_two_electron_integral_values_data,
            ao_two_electron_integral_indices_data,
            ao_two_electron_integral_symmetry_shifts,
            ao_effective_one_electron_linear_indices,
            column_offsets_data,
            n_basis_functions,
            invalid_integral_index,
            g11_storage.data());
      }
    } else if (ao_effective_one_electron_linear_indices != nullptr &&
               validate_integral_indices) {
      for (std::size_t integral_index = 0;
           integral_index < ao_two_electron_integral_values.size();
           ++integral_index) {
        accumulate_ao_effective_one_electron_integral<true, false, true>(
            integral_index,
            inactive_density_matrix_data,
            ao_two_electron_integral_values_data,
            ao_two_electron_integral_indices_data,
            nullptr,
            ao_effective_one_electron_linear_indices,
            column_offsets_data,
            n_basis_functions,
            invalid_integral_index,
            g11_storage.data());
      }
    } else if (ao_effective_one_electron_linear_indices != nullptr &&
               ao_two_electron_integral_symmetry_shifts != nullptr) {
      for (std::size_t integral_index = 0;
           integral_index < ao_two_electron_integral_values.size();
           ++integral_index) {
        accumulate_ao_effective_one_electron_integral<false, true, true>(
            integral_index,
            inactive_density_matrix_data,
            ao_two_electron_integral_values_data,
            ao_two_electron_integral_indices_data,
            ao_two_electron_integral_symmetry_shifts,
            ao_effective_one_electron_linear_indices,
            column_offsets_data,
            n_basis_functions,
            invalid_integral_index,
            g11_storage.data());
      }
    } else if (ao_effective_one_electron_linear_indices != nullptr) {
      for (std::size_t integral_index = 0;
           integral_index < ao_two_electron_integral_values.size();
           ++integral_index) {
        accumulate_ao_effective_one_electron_integral<false, false, true>(
            integral_index,
            inactive_density_matrix_data,
            ao_two_electron_integral_values_data,
            ao_two_electron_integral_indices_data,
            nullptr,
            ao_effective_one_electron_linear_indices,
            column_offsets_data,
            n_basis_functions,
            invalid_integral_index,
            g11_storage.data());
      }
    } else if (validate_integral_indices && ao_two_electron_integral_symmetry_shifts != nullptr) {
      for (std::size_t integral_index = 0;
           integral_index < ao_two_electron_integral_values.size();
           ++integral_index) {
        accumulate_ao_effective_one_electron_integral<true, true, false>(
            integral_index,
            inactive_density_matrix_data,
            ao_two_electron_integral_values_data,
            ao_two_electron_integral_indices_data,
            ao_two_electron_integral_symmetry_shifts,
            nullptr,
            column_offsets_data,
            n_basis_functions,
            invalid_integral_index,
            g11_storage.data());
      }
    } else if (validate_integral_indices) {
      for (std::size_t integral_index = 0;
           integral_index < ao_two_electron_integral_values.size();
           ++integral_index) {
        accumulate_ao_effective_one_electron_integral<true, false, false>(
            integral_index,
            inactive_density_matrix_data,
            ao_two_electron_integral_values_data,
            ao_two_electron_integral_indices_data,
            nullptr,
            nullptr,
            column_offsets_data,
            n_basis_functions,
            invalid_integral_index,
            g11_storage.data());
      }
    } else if (ao_two_electron_integral_symmetry_shifts != nullptr) {
      for (std::size_t integral_index = 0;
           integral_index < ao_two_electron_integral_values.size();
           ++integral_index) {
        accumulate_ao_effective_one_electron_integral<false, true, false>(
            integral_index,
            inactive_density_matrix_data,
            ao_two_electron_integral_values_data,
            ao_two_electron_integral_indices_data,
            ao_two_electron_integral_symmetry_shifts,
            nullptr,
            column_offsets_data,
            n_basis_functions,
            invalid_integral_index,
            g11_storage.data());
      }
    } else {
      for (std::size_t integral_index = 0;
           integral_index < ao_two_electron_integral_values.size();
           ++integral_index) {
        accumulate_ao_effective_one_electron_integral<false, false, false>(
            integral_index,
            inactive_density_matrix_data,
            ao_two_electron_integral_values_data,
            ao_two_electron_integral_indices_data,
            nullptr,
            nullptr,
            column_offsets_data,
            n_basis_functions,
            invalid_integral_index,
            g11_storage.data());
      }
    }
  } else {
    partial_g11.assign(
        xmvb::to_size(n_threads),
        std::vector<double>(matrix_size, 0.0));

#pragma omp parallel
    {
      int thread_index = 0;
#ifdef _OPENMP
      thread_index = omp_get_thread_num();
#endif
      auto& local_g11 = partial_g11[xmvb::to_size(thread_index)];

#pragma omp for schedule(static)
      for (std::ptrdiff_t integral_offset = 0;
           integral_offset < static_cast<std::ptrdiff_t>(ao_two_electron_integral_values.size());
           ++integral_offset) {
        const std::size_t integral_index = xmvb::to_size(integral_offset);
        if (ao_effective_one_electron_linear_indices != nullptr &&
            validate_integral_indices &&
            ao_two_electron_integral_symmetry_shifts != nullptr) {
          accumulate_ao_effective_one_electron_integral<true, true, true>(
              integral_index,
              inactive_density_matrix_data,
              ao_two_electron_integral_values_data,
              ao_two_electron_integral_indices_data,
              ao_two_electron_integral_symmetry_shifts,
              ao_effective_one_electron_linear_indices,
              column_offsets_data,
              n_basis_functions,
              invalid_integral_index,
              local_g11.data());
        } else if (ao_effective_one_electron_linear_indices != nullptr &&
                   validate_integral_indices) {
          accumulate_ao_effective_one_electron_integral<true, false, true>(
              integral_index,
              inactive_density_matrix_data,
              ao_two_electron_integral_values_data,
              ao_two_electron_integral_indices_data,
              nullptr,
              ao_effective_one_electron_linear_indices,
              column_offsets_data,
              n_basis_functions,
              invalid_integral_index,
              local_g11.data());
        } else if (ao_effective_one_electron_linear_indices != nullptr &&
                   ao_two_electron_integral_symmetry_shifts != nullptr) {
          accumulate_ao_effective_one_electron_integral<false, true, true>(
              integral_index,
              inactive_density_matrix_data,
              ao_two_electron_integral_values_data,
              ao_two_electron_integral_indices_data,
              ao_two_electron_integral_symmetry_shifts,
              ao_effective_one_electron_linear_indices,
              column_offsets_data,
              n_basis_functions,
              invalid_integral_index,
              local_g11.data());
        } else if (ao_effective_one_electron_linear_indices != nullptr) {
          accumulate_ao_effective_one_electron_integral<false, false, true>(
              integral_index,
              inactive_density_matrix_data,
              ao_two_electron_integral_values_data,
              ao_two_electron_integral_indices_data,
              nullptr,
              ao_effective_one_electron_linear_indices,
              column_offsets_data,
              n_basis_functions,
              invalid_integral_index,
              local_g11.data());
        } else if (validate_integral_indices &&
                   ao_two_electron_integral_symmetry_shifts != nullptr) {
          accumulate_ao_effective_one_electron_integral<true, true, false>(
              integral_index,
              inactive_density_matrix_data,
              ao_two_electron_integral_values_data,
              ao_two_electron_integral_indices_data,
              ao_two_electron_integral_symmetry_shifts,
              nullptr,
              column_offsets_data,
              n_basis_functions,
              invalid_integral_index,
              local_g11.data());
        } else if (validate_integral_indices) {
          accumulate_ao_effective_one_electron_integral<true, false, false>(
              integral_index,
              inactive_density_matrix_data,
              ao_two_electron_integral_values_data,
              ao_two_electron_integral_indices_data,
              nullptr,
              nullptr,
              column_offsets_data,
              n_basis_functions,
              invalid_integral_index,
              local_g11.data());
        } else if (ao_two_electron_integral_symmetry_shifts != nullptr) {
          accumulate_ao_effective_one_electron_integral<false, true, false>(
              integral_index,
              inactive_density_matrix_data,
              ao_two_electron_integral_values_data,
              ao_two_electron_integral_indices_data,
              ao_two_electron_integral_symmetry_shifts,
              nullptr,
              column_offsets_data,
              n_basis_functions,
              invalid_integral_index,
              local_g11.data());
        } else {
          accumulate_ao_effective_one_electron_integral<false, false, false>(
              integral_index,
              inactive_density_matrix_data,
              ao_two_electron_integral_values_data,
              ao_two_electron_integral_indices_data,
              nullptr,
              nullptr,
              column_offsets_data,
              n_basis_functions,
              invalid_integral_index,
              local_g11.data());
        }
      }
    }
  }

  if (validate_integral_indices && invalid_integral_index.load() >= 0) {
    throw std::invalid_argument("AO two-electron index out of range");
  }

  if (n_threads > 1) {
    for (const auto& partial_matrix : partial_g11) {
      for (std::size_t index = 0; index < g11_storage.size(); ++index) {
        g11_storage[index] += partial_matrix[index];
      }
    }
  }

  Eigen::Map<Eigen::MatrixXd> g11(
      g11_storage.data(),
      n_basis_functions,
      n_basis_functions);

  for (int row = 0; row < n_basis_functions; ++row) {
    for (int column = 0; column <= row; ++column) {
      g11(row, column) = g11(row, column) + g11(column, row);
      g11(column, row) = g11(row, column);
    }
  }

  const Eigen::MatrixXd f11 = g11 + hhf;

  AoEffectiveOneElectronResult result;
  result.ao_coulomb_exchange_matrix.assign(
      g11.data(),
      g11.data() + g11.size());
  result.ao_effective_h1e.assign(
      f11.data(),
      f11.data() + f11.size());
  return result;
}

AoEffectiveOneElectronResult AoEffectiveOneElectronBuilder::build(
    const std::vector<double>& inactive_density_matrix,
    const std::vector<double>& ao_core_hamiltonian_matrix,
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_integral_indices,
    int n_basis_functions) const {
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }
  const std::size_t matrix_size =
      xmvb::to_size(n_basis_functions) * n_basis_functions;
  if (inactive_density_matrix.size() != matrix_size) {
    throw std::invalid_argument("inactive density matrix size mismatch");
  }
  const Eigen::Map<const Eigen::MatrixXd> inactive_density_matrix_map(
      inactive_density_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  return build(
      inactive_density_matrix_map,
      ao_core_hamiltonian_matrix,
      ao_two_electron_integral_values,
      ao_two_electron_integral_indices,
      n_basis_functions);
}

AoEffectiveOneElectronResult AoEffectiveOneElectronBuilder::build(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_matrix,
    const std::vector<double>& ao_core_hamiltonian_matrix,
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_integral_indices,
    int n_basis_functions) const {
  return build_ao_effective_one_electron(
      inactive_density_matrix,
      ao_core_hamiltonian_matrix,
      ao_two_electron_integral_values,
      ao_two_electron_integral_indices,
      nullptr,
      nullptr,
      n_basis_functions,
      true);
}

AoEffectiveOneElectronResult AoEffectiveOneElectronBuilder::build(
    const std::vector<double>& inactive_density_matrix,
    const std::vector<double>& ao_core_hamiltonian_matrix,
    const LibcintRiIntegralProviderResult& ri_integral_provider_result,
    int n_basis_functions) const {
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }
  const std::size_t matrix_size =
      xmvb::to_size(n_basis_functions) * n_basis_functions;
  if (inactive_density_matrix.size() != matrix_size) {
    throw std::invalid_argument("inactive density matrix size mismatch");
  }
  const Eigen::Map<const Eigen::MatrixXd> inactive_density_matrix_map(
      inactive_density_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  return build(
      inactive_density_matrix_map,
      ao_core_hamiltonian_matrix,
      ri_integral_provider_result,
      n_basis_functions);
}

AoEffectiveOneElectronResult AoEffectiveOneElectronBuilder::build(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_matrix,
    const std::vector<double>& ao_core_hamiltonian_matrix,
    const LibcintRiIntegralProviderResult& ri_integral_provider_result,
    int n_basis_functions) const {
  return build_ao_effective_one_electron_ri(
      inactive_density_matrix,
      ao_core_hamiltonian_matrix,
      ri_integral_provider_result,
      n_basis_functions);
}

AoEffectiveOneElectronResult AoEffectiveOneElectronBuilder::build(
    const OrbitalPreparationResult& orbital_result,
    const std::vector<double>& ao_core_hamiltonian_matrix,
    const LibcintRiIntegralProviderResult& ri_integral_provider_result,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals) const {
  if (n_inactive_doubly_occupied_orbitals < 0 ||
      n_inactive_doubly_occupied_orbitals > n_basis_functions) {
    throw std::invalid_argument("invalid inactive-orbital count for AO RI builder");
  }
  if (orbital_result.inactive_density_low_rank_factors.rows() != n_basis_functions ||
      orbital_result.inactive_density_low_rank_factors.cols() !=
          n_inactive_doubly_occupied_orbitals) {
    throw std::invalid_argument("inactive density low-rank factor shape mismatch");
  }

  return build_ao_effective_one_electron_ri(
      AoEffectiveOneElectronRiLowRankFactors{
          .scaled_factor_matrix = orbital_result.inactive_density_low_rank_factors,
          .n_positive_components = n_inactive_doubly_occupied_orbitals,
          .n_negative_components = 0,
      },
      ao_core_hamiltonian_matrix,
      ri_integral_provider_result,
      n_basis_functions);
}

AoEffectiveOneElectronResult AoEffectiveOneElectronBuilder::build(
    const std::vector<double>& inactive_density_matrix,
    const std::vector<double>& ao_core_hamiltonian_matrix,
    const AoIntegralInput& ao_integral_input) const {
  const std::size_t matrix_size =
      xmvb::to_size(ao_integral_input.n_basis_functions) *
      ao_integral_input.n_basis_functions;
  if (inactive_density_matrix.size() != matrix_size) {
    throw std::invalid_argument(
        "inactive density matrix size does not match the AO-H1E graph dimensions");
  }
  const Eigen::Map<const Eigen::MatrixXd> inactive_density_matrix_map(
      inactive_density_matrix.data(),
      ao_integral_input.n_basis_functions,
      ao_integral_input.n_basis_functions);
  return build(
      inactive_density_matrix_map,
      ao_core_hamiltonian_matrix,
      ao_integral_input);
}

AoEffectiveOneElectronResult AoEffectiveOneElectronBuilder::build(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_matrix,
    const std::vector<double>& ao_core_hamiltonian_matrix,
    const AoIntegralInput& ao_integral_input) const {
  const std::size_t matrix_size =
      xmvb::to_size(ao_integral_input.n_basis_functions) *
      ao_integral_input.n_basis_functions;
  if (inactive_density_matrix.rows() != ao_integral_input.n_basis_functions ||
      inactive_density_matrix.cols() != ao_integral_input.n_basis_functions ||
      ao_core_hamiltonian_matrix.size() != matrix_size) {
    throw std::invalid_argument(
        "AO matrix sizes do not match the AO-H1E graph dimensions");
  }

  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  const int graph_threads =
      choose_ao_effective_one_electron_graph_threads(
          n_threads,
          ao_integral_input.n_basis_functions);
  if (ao_effective_one_electron_graph_available(ao_integral_input)) {
    std::vector<double> g11_storage =
        apply_ao_effective_one_electron_graph_forward(
            inactive_density_matrix.data(),
            ao_integral_input,
            graph_threads);
    Eigen::Map<const Eigen::MatrixXd> hhf(
        ao_core_hamiltonian_matrix.data(),
        ao_integral_input.n_basis_functions,
        ao_integral_input.n_basis_functions);
    Eigen::Map<Eigen::MatrixXd> g11(
        g11_storage.data(),
        ao_integral_input.n_basis_functions,
        ao_integral_input.n_basis_functions);
    for (int row = 0; row < ao_integral_input.n_basis_functions; ++row) {
      for (int column = 0; column <= row; ++column) {
        g11(row, column) = g11(row, column) + g11(column, row);
        g11(column, row) = g11(row, column);
      }
    }
    const Eigen::MatrixXd f11 = g11 + hhf;

    AoEffectiveOneElectronResult result;
    result.ao_coulomb_exchange_matrix.assign(
        g11.data(),
        g11.data() + g11.size());
    result.ao_effective_h1e.assign(
        f11.data(),
        f11.data() + f11.size());
    return result;
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
  const bool validate_integral_indices =
      ao_integral_input.ao_two_electron_pair_indices.empty() &&
      ao_integral_input.ao_two_electron_pair_graph_row_offsets.empty();
  return build_ao_effective_one_electron(
      inactive_density_matrix,
      ao_core_hamiltonian_matrix,
      ao_integral_input.ao_two_electron_integral_values.vector(),
      ao_integral_input.ao_two_electron_integral_indices.vector(),
      ao_integral_input.ao_two_electron_integral_symmetry_shifts.empty()
          ? nullptr
          : ao_integral_input.ao_two_electron_integral_symmetry_shifts.data(),
      ao_integral_input.ao_effective_one_electron_linear_indices.empty()
          ? nullptr
          : ao_integral_input.ao_effective_one_electron_linear_indices.data(),
      ao_integral_input.n_basis_functions,
      validate_integral_indices);
}

AoEffectiveOneElectronResult AoEffectiveOneElectronBuilder::build(
    const std::vector<double>& inactive_density_matrix,
    const AoIntegralInput& ao_integral_input) const {
  const std::size_t matrix_size =
      xmvb::to_size(ao_integral_input.n_basis_functions) *
      ao_integral_input.n_basis_functions;
  if (inactive_density_matrix.size() != matrix_size) {
    throw std::invalid_argument(
        "inactive density matrix size does not match the AO-H1E graph dimensions");
  }
  const Eigen::Map<const Eigen::MatrixXd> inactive_density_matrix_map(
      inactive_density_matrix.data(),
      ao_integral_input.n_basis_functions,
      ao_integral_input.n_basis_functions);
  return build(inactive_density_matrix_map, ao_integral_input);
}

AoEffectiveOneElectronResult AoEffectiveOneElectronBuilder::build(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_matrix,
    const AoIntegralInput& ao_integral_input) const {
  return build(
      inactive_density_matrix,
      ao_integral_input.ao_core_hamiltonian_matrix.vector(),
      ao_integral_input);
}

}  // namespace xmvb::vb
