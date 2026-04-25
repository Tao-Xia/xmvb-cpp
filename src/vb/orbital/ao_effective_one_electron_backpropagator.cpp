#include "vb/orbital/ao_effective_one_electron_backpropagator.hpp"

#include <atomic>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/openmp_utils.hpp"
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
  std::vector<std::size_t> column_offsets(n_basis_functions, 0);
  for (int column = 0; column < n_basis_functions; ++column) {
    column_offsets[column] =
        column * n_basis_functions;
  }
  return column_offsets;
}

std::vector<double> encode_symmetric_gradient_as_unsymmetrized_storage(
    const std::vector<double>& symmetric_gradient,
    int n_basis_functions) {
  const std::size_t matrix_size =
      n_basis_functions * n_basis_functions;
  if (symmetric_gradient.size() != matrix_size) {
    throw std::invalid_argument("symmetric_gradient size mismatch");
  }

  const Eigen::Map<const Eigen::MatrixXd> symmetric_gradient_matrix(
      symmetric_gradient.data(),
      n_basis_functions,
      n_basis_functions);
  Eigen::MatrixXd unsymmetrized_gradient_matrix =
      symmetric_gradient_matrix.triangularView<Eigen::Lower>();

  // The orbital backpropagator expects a column-major matrix representation G
  // whose symmetric contribution is formed later as G + G^T.  The RI operator
  // already returns the fully symmetric derivative dE/dP, so we store only one
  // triangular half and halve the diagonal to preserve the same downstream
  // convention as the exact integral path.
  unsymmetrized_gradient_matrix.diagonal() *= 0.5;

  return std::vector<double>(
      unsymmetrized_gradient_matrix.data(),
      unsymmetrized_gradient_matrix.data() + unsymmetrized_gradient_matrix.size());
}

struct SignedSmallMatrixFactors {
  Eigen::MatrixXd scaled_eigenvectors;
  int n_positive_components = 0;
  int n_negative_components = 0;
};

SignedSmallMatrixFactors build_signed_small_matrix_factors(
    const Eigen::Ref<const Eigen::MatrixXd>& symmetric_matrix) {
  if (symmetric_matrix.rows() != symmetric_matrix.cols()) {
    throw std::invalid_argument("small-matrix RI factorization requires a square matrix");
  }

  SignedSmallMatrixFactors factors;
  if (symmetric_matrix.rows() == 0) {
    factors.scaled_eigenvectors.resize(0, 0);
    return factors;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(symmetric_matrix);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("failed to diagonalize active one-electron gradient");
  }

  const auto eigenvalues = solver.eigenvalues();
  const double max_abs_eigenvalue =
      eigenvalues.cwiseAbs().size() > 0 ? eigenvalues.cwiseAbs().maxCoeff() : 0.0;
  const double effective_cutoff =
      std::numeric_limits<double>::epsilon() *
      static_cast<double>(symmetric_matrix.rows()) * max_abs_eigenvalue;

  for (int index = 0; index < eigenvalues.size(); ++index) {
    const double eigenvalue = eigenvalues(index);
    if (eigenvalue > effective_cutoff) {
      ++factors.n_positive_components;
    } else if (eigenvalue < -effective_cutoff) {
      ++factors.n_negative_components;
    }
  }

  const int effective_rank =
      factors.n_positive_components + factors.n_negative_components;
  factors.scaled_eigenvectors.resize(symmetric_matrix.rows(), effective_rank);

  int positive_column = 0;
  int negative_column = factors.n_positive_components;
  for (int index = 0; index < eigenvalues.size(); ++index) {
    const double eigenvalue = eigenvalues(index);
    if (eigenvalue > effective_cutoff) {
      factors.scaled_eigenvectors.col(positive_column++) =
          solver.eigenvectors().col(index) * std::sqrt(eigenvalue);
    } else if (eigenvalue < -effective_cutoff) {
      factors.scaled_eigenvectors.col(negative_column++) =
          solver.eigenvectors().col(index) * std::sqrt(-eigenvalue);
    }
  }
  return factors;
}

AoEffectiveOneElectronRiLowRankFactors
build_ao_effective_one_electron_ri_backpropagator_factors(
    const std::vector<double>& active_one_electron_gradient,
    const OrbitalPreparationResult& orbital_result,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) {
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument("RI AO-H1E backprop dimensions must be positive");
  }
  if (n_inactive_doubly_occupied_orbitals < 0 ||
      n_inactive_doubly_occupied_orbitals + n_active_orbitals > n_basis_functions) {
    throw std::invalid_argument("invalid inactive/active partition for RI AO-H1E backprop");
  }
  const std::size_t expected_active_gradient_size =
      n_active_orbitals * n_active_orbitals;
  if (active_one_electron_gradient.size() != expected_active_gradient_size) {
    throw std::invalid_argument("active one-electron gradient size mismatch");
  }
  const std::size_t expected_auxiliary_size =
      n_basis_functions * n_basis_functions;
  if (orbital_result.auxiliary_orbital_matrix.size() != expected_auxiliary_size) {
    throw std::invalid_argument("auxiliary orbital matrix size mismatch");
  }
  if (orbital_result.inactive_density_low_rank_factors.rows() != n_basis_functions ||
      orbital_result.inactive_density_low_rank_factors.cols() !=
          n_inactive_doubly_occupied_orbitals) {
    throw std::invalid_argument("inactive density factor shape mismatch");
  }

  const Eigen::Map<const Eigen::MatrixXd> active_gradient(
      active_one_electron_gradient.data(),
      n_active_orbitals,
      n_active_orbitals);
  const Eigen::MatrixXd active_gradient_symmetric =
      active_gradient + active_gradient.transpose();
  const SignedSmallMatrixFactors active_factors =
      build_signed_small_matrix_factors(active_gradient_symmetric);

  const int n_positive_components =
      n_inactive_doubly_occupied_orbitals + active_factors.n_positive_components;
  const int n_negative_components = active_factors.n_negative_components;
  Eigen::MatrixXd scaled_factor_matrix =
      Eigen::MatrixXd::Zero(
          n_basis_functions,
          n_positive_components + n_negative_components);

  if (n_inactive_doubly_occupied_orbitals > 0) {
    // The matrix-based RI backprop symmetrizes `P11` as `P11 + P11^T = 2 P11`.
    // Scaling the occupied-space factors by `sqrt(2)` reproduces that same AO
    // operator input without ever materializing the full AO matrix.
    scaled_factor_matrix.leftCols(n_inactive_doubly_occupied_orbitals) =
        std::sqrt(2.0) * orbital_result.inactive_density_low_rank_factors;
  }

  if (active_factors.scaled_eigenvectors.cols() > 0) {
    const Eigen::Map<const Eigen::MatrixXd> auxiliary_matrix(
        orbital_result.auxiliary_orbital_matrix.data(),
        n_basis_functions,
        n_basis_functions);
    const auto active_auxiliary_orbitals = auxiliary_matrix.middleCols(
        n_inactive_doubly_occupied_orbitals,
        n_active_orbitals);
    // The AO adjoint entering `G[P11]` is
    // `T_active * (dE/dHHO + dE/dHHO^T) * T_active^T`.  Diagonalizing only the
    // small active-space matrix exposes the signed low-rank factors needed by
    // the AO RI operator while completely avoiding an `n_basis x n_basis`
    // eigendecomposition.
    scaled_factor_matrix.middleCols(
        n_inactive_doubly_occupied_orbitals,
        active_factors.scaled_eigenvectors.cols()).noalias() =
        active_auxiliary_orbitals * active_factors.scaled_eigenvectors;
  }

  AoEffectiveOneElectronRiLowRankFactors result;
  result.scaled_factor_matrix = scaled_factor_matrix;
  result.n_positive_components = n_positive_components;
  result.n_negative_components = n_negative_components;
  return result;
}

template <bool ValidateIntegralIndices, bool UsePrecomputedSymmetryShifts, bool UseLinearIndexCache>
inline void accumulate_ao_effective_one_electron_backprop_integral(
    std::size_t integral_index,
    const double* unsymmetrized_gradient_storage,
    const double* ao_two_electron_integral_values,
    const int* ao_two_electron_integral_indices,
    const std::uint8_t* ao_two_electron_integral_symmetry_shifts,
    const int* ao_effective_one_electron_linear_indices,
    const std::size_t* column_offsets,
    int n_basis_functions,
    std::atomic<int>& invalid_integral_index,
    double* local_inactive_density_gradient) {
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

  const double gradient_ij = unsymmetrized_gradient_storage[ij_index];
  const double gradient_kl = unsymmetrized_gradient_storage[kl_index];
  const double gradient_ik = unsymmetrized_gradient_storage[ik_index];
  const double gradient_jl = unsymmetrized_gradient_storage[jl_index];
  const double gradient_il = unsymmetrized_gradient_storage[il_index];
  const double gradient_jk = unsymmetrized_gradient_storage[jk_index];
  const double scaled_two_electron_value = two_electron_value * 4.0;

  local_inactive_density_gradient[ij_index] +=
      gradient_kl * scaled_two_electron_value;
  local_inactive_density_gradient[kl_index] +=
      gradient_ij * scaled_two_electron_value;
  local_inactive_density_gradient[lj_index] -= gradient_ik * two_electron_value;
  local_inactive_density_gradient[ki_index] -= gradient_jl * two_electron_value;
  local_inactive_density_gradient[kj_index] -= gradient_il * two_electron_value;
  local_inactive_density_gradient[li_index] -= gradient_jk * two_electron_value;
}

AoEffectiveOneElectronBackpropagationResult
backpropagate_ao_effective_one_electron_ri(
    const std::vector<double>& ao_effective_one_electron_gradient,
    const LibcintRiIntegralProviderResult& ri_integral_provider_result,
    int n_basis_functions) {
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }
  const std::size_t matrix_size =
      n_basis_functions * n_basis_functions;
  if (ao_effective_one_electron_gradient.size() != matrix_size) {
    throw std::invalid_argument("ao_effective_one_electron_gradient size mismatch");
  }
  if (ri_integral_provider_result.n_basis_functions != n_basis_functions) {
    throw std::invalid_argument("RI basis-function count mismatch");
  }

  const Eigen::Map<const Eigen::MatrixXd> ao_effective_gradient(
      ao_effective_one_electron_gradient.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::MatrixXd unsymmetrized_g11_gradient =
      ao_effective_gradient + ao_effective_gradient.transpose();
  std::vector<double> inactive_density_gradient_storage =
      apply_ao_effective_one_electron_ri_operator(
          std::vector<double>(
              unsymmetrized_g11_gradient.data(),
              unsymmetrized_g11_gradient.data() + unsymmetrized_g11_gradient.size()),
          ri_integral_provider_result,
          n_basis_functions,
          {.attempt_spectral_factorization = true});
  inactive_density_gradient_storage =
      encode_symmetric_gradient_as_unsymmetrized_storage(
          inactive_density_gradient_storage,
          n_basis_functions);

  AoEffectiveOneElectronBackpropagationResult result;
  result.inactive_density_gradient = std::move(inactive_density_gradient_storage);
  return result;
}

AoEffectiveOneElectronBackpropagationResult
backpropagate_ao_effective_one_electron_ri(
    const std::vector<double>& active_one_electron_gradient,
    const OrbitalPreparationResult& orbital_result,
    const LibcintRiIntegralProviderResult& ri_integral_provider_result,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) {
  const auto low_rank_factors =
      build_ao_effective_one_electron_ri_backpropagator_factors(
          active_one_electron_gradient,
          orbital_result,
          n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);
  std::vector<double> inactive_density_gradient_storage =
      apply_ao_effective_one_electron_ri_operator(
          low_rank_factors,
          ri_integral_provider_result,
          n_basis_functions);
  inactive_density_gradient_storage =
      encode_symmetric_gradient_as_unsymmetrized_storage(
          inactive_density_gradient_storage,
          n_basis_functions);

  AoEffectiveOneElectronBackpropagationResult result;
  result.inactive_density_gradient = std::move(inactive_density_gradient_storage);
  return result;
}

}  // namespace

AoEffectiveOneElectronBackpropagationResult
backpropagate_ao_effective_one_electron(
    const std::vector<double>& ao_effective_one_electron_gradient,
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
      n_basis_functions * n_basis_functions;
  if (ao_effective_one_electron_gradient.size() != matrix_size) {
    throw std::invalid_argument("ao_effective_one_electron_gradient size mismatch");
  }
  if (ao_two_electron_integral_indices.size() != ao_two_electron_integral_values.size() * 4) {
    throw std::invalid_argument("AO two-electron index/value sizes are inconsistent");
  }

  const Eigen::Map<const Eigen::MatrixXd> ao_effective_gradient(
      ao_effective_one_electron_gradient.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::MatrixXd unsymmetrized_g11_gradient =
      ao_effective_gradient + ao_effective_gradient.transpose();
  const std::vector<double> unsymmetrized_gradient_storage(
      unsymmetrized_g11_gradient.data(),
      unsymmetrized_g11_gradient.data() + unsymmetrized_g11_gradient.size());
  std::vector<double> inactive_density_gradient_storage(matrix_size, 0.0);
  std::atomic<int> invalid_integral_index(-1);
  const double* unsymmetrized_gradient_storage_data =
      unsymmetrized_gradient_storage.data();
  const double* ao_two_electron_integral_values_data =
      ao_two_electron_integral_values.data();
  const int* ao_two_electron_integral_indices_data =
      ao_two_electron_integral_indices.data();
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
  n_threads = xmvb::effective_openmp_thread_count();
#endif
  std::vector<std::vector<double>> partial_inactive_density_gradients;
  if (n_threads <= 1) {
    if (ao_effective_one_electron_linear_indices != nullptr &&
        validate_integral_indices &&
        ao_two_electron_integral_symmetry_shifts != nullptr) {
      for (std::size_t integral_index = 0;
           integral_index < ao_two_electron_integral_values.size();
           ++integral_index) {
        accumulate_ao_effective_one_electron_backprop_integral<true, true, true>(
            integral_index,
            unsymmetrized_gradient_storage_data,
            ao_two_electron_integral_values_data,
            ao_two_electron_integral_indices_data,
            ao_two_electron_integral_symmetry_shifts,
            ao_effective_one_electron_linear_indices,
            column_offsets_data,
            n_basis_functions,
            invalid_integral_index,
            inactive_density_gradient_storage.data());
      }
    } else if (ao_effective_one_electron_linear_indices != nullptr &&
               validate_integral_indices) {
      for (std::size_t integral_index = 0;
           integral_index < ao_two_electron_integral_values.size();
           ++integral_index) {
        accumulate_ao_effective_one_electron_backprop_integral<true, false, true>(
            integral_index,
            unsymmetrized_gradient_storage_data,
            ao_two_electron_integral_values_data,
            ao_two_electron_integral_indices_data,
            nullptr,
            ao_effective_one_electron_linear_indices,
            column_offsets_data,
            n_basis_functions,
            invalid_integral_index,
            inactive_density_gradient_storage.data());
      }
    } else if (ao_effective_one_electron_linear_indices != nullptr &&
               ao_two_electron_integral_symmetry_shifts != nullptr) {
      for (std::size_t integral_index = 0;
           integral_index < ao_two_electron_integral_values.size();
           ++integral_index) {
        accumulate_ao_effective_one_electron_backprop_integral<false, true, true>(
            integral_index,
            unsymmetrized_gradient_storage_data,
            ao_two_electron_integral_values_data,
            ao_two_electron_integral_indices_data,
            ao_two_electron_integral_symmetry_shifts,
            ao_effective_one_electron_linear_indices,
            column_offsets_data,
            n_basis_functions,
            invalid_integral_index,
            inactive_density_gradient_storage.data());
      }
    } else if (ao_effective_one_electron_linear_indices != nullptr) {
      for (std::size_t integral_index = 0;
           integral_index < ao_two_electron_integral_values.size();
           ++integral_index) {
        accumulate_ao_effective_one_electron_backprop_integral<false, false, true>(
            integral_index,
            unsymmetrized_gradient_storage_data,
            ao_two_electron_integral_values_data,
            ao_two_electron_integral_indices_data,
            nullptr,
            ao_effective_one_electron_linear_indices,
            column_offsets_data,
            n_basis_functions,
            invalid_integral_index,
            inactive_density_gradient_storage.data());
      }
    } else if (validate_integral_indices && ao_two_electron_integral_symmetry_shifts != nullptr) {
      for (std::size_t integral_index = 0;
           integral_index < ao_two_electron_integral_values.size();
           ++integral_index) {
        accumulate_ao_effective_one_electron_backprop_integral<true, true, false>(
            integral_index,
            unsymmetrized_gradient_storage_data,
            ao_two_electron_integral_values_data,
            ao_two_electron_integral_indices_data,
            ao_two_electron_integral_symmetry_shifts,
            nullptr,
            column_offsets_data,
            n_basis_functions,
            invalid_integral_index,
            inactive_density_gradient_storage.data());
      }
    } else if (validate_integral_indices) {
      for (std::size_t integral_index = 0;
           integral_index < ao_two_electron_integral_values.size();
           ++integral_index) {
        accumulate_ao_effective_one_electron_backprop_integral<true, false, false>(
            integral_index,
            unsymmetrized_gradient_storage_data,
            ao_two_electron_integral_values_data,
            ao_two_electron_integral_indices_data,
            nullptr,
            nullptr,
            column_offsets_data,
            n_basis_functions,
            invalid_integral_index,
            inactive_density_gradient_storage.data());
      }
    } else if (ao_two_electron_integral_symmetry_shifts != nullptr) {
      for (std::size_t integral_index = 0;
           integral_index < ao_two_electron_integral_values.size();
           ++integral_index) {
        accumulate_ao_effective_one_electron_backprop_integral<false, true, false>(
            integral_index,
            unsymmetrized_gradient_storage_data,
            ao_two_electron_integral_values_data,
            ao_two_electron_integral_indices_data,
            ao_two_electron_integral_symmetry_shifts,
            nullptr,
            column_offsets_data,
            n_basis_functions,
            invalid_integral_index,
            inactive_density_gradient_storage.data());
      }
    } else {
      for (std::size_t integral_index = 0;
           integral_index < ao_two_electron_integral_values.size();
           ++integral_index) {
        accumulate_ao_effective_one_electron_backprop_integral<false, false, false>(
            integral_index,
            unsymmetrized_gradient_storage_data,
            ao_two_electron_integral_values_data,
            ao_two_electron_integral_indices_data,
            nullptr,
            nullptr,
            column_offsets_data,
            n_basis_functions,
            invalid_integral_index,
            inactive_density_gradient_storage.data());
      }
    }
  } else {
    partial_inactive_density_gradients.assign(
        n_threads,
        std::vector<double>(matrix_size, 0.0));

#pragma omp parallel
    {
      int thread_index = 0;
#ifdef _OPENMP
      thread_index = omp_get_thread_num();
#endif
      auto& local_inactive_density_gradient =
          partial_inactive_density_gradients[thread_index];

#pragma omp for schedule(static)
      for (std::ptrdiff_t integral_offset = 0;
           integral_offset < static_cast<std::ptrdiff_t>(ao_two_electron_integral_values.size());
           ++integral_offset) {
        const std::size_t integral_index = integral_offset;
        if (ao_effective_one_electron_linear_indices != nullptr &&
            validate_integral_indices &&
            ao_two_electron_integral_symmetry_shifts != nullptr) {
          accumulate_ao_effective_one_electron_backprop_integral<true, true, true>(
              integral_index,
              unsymmetrized_gradient_storage_data,
              ao_two_electron_integral_values_data,
              ao_two_electron_integral_indices_data,
              ao_two_electron_integral_symmetry_shifts,
              ao_effective_one_electron_linear_indices,
              column_offsets_data,
              n_basis_functions,
              invalid_integral_index,
              local_inactive_density_gradient.data());
        } else if (ao_effective_one_electron_linear_indices != nullptr &&
                   validate_integral_indices) {
          accumulate_ao_effective_one_electron_backprop_integral<true, false, true>(
              integral_index,
              unsymmetrized_gradient_storage_data,
              ao_two_electron_integral_values_data,
              ao_two_electron_integral_indices_data,
              nullptr,
              ao_effective_one_electron_linear_indices,
              column_offsets_data,
              n_basis_functions,
              invalid_integral_index,
              local_inactive_density_gradient.data());
        } else if (ao_effective_one_electron_linear_indices != nullptr &&
                   ao_two_electron_integral_symmetry_shifts != nullptr) {
          accumulate_ao_effective_one_electron_backprop_integral<false, true, true>(
              integral_index,
              unsymmetrized_gradient_storage_data,
              ao_two_electron_integral_values_data,
              ao_two_electron_integral_indices_data,
              ao_two_electron_integral_symmetry_shifts,
              ao_effective_one_electron_linear_indices,
              column_offsets_data,
              n_basis_functions,
              invalid_integral_index,
              local_inactive_density_gradient.data());
        } else if (ao_effective_one_electron_linear_indices != nullptr) {
          accumulate_ao_effective_one_electron_backprop_integral<false, false, true>(
              integral_index,
              unsymmetrized_gradient_storage_data,
              ao_two_electron_integral_values_data,
              ao_two_electron_integral_indices_data,
              nullptr,
              ao_effective_one_electron_linear_indices,
              column_offsets_data,
              n_basis_functions,
              invalid_integral_index,
              local_inactive_density_gradient.data());
        } else if (validate_integral_indices &&
                   ao_two_electron_integral_symmetry_shifts != nullptr) {
          accumulate_ao_effective_one_electron_backprop_integral<true, true, false>(
              integral_index,
              unsymmetrized_gradient_storage_data,
              ao_two_electron_integral_values_data,
              ao_two_electron_integral_indices_data,
              ao_two_electron_integral_symmetry_shifts,
              nullptr,
              column_offsets_data,
              n_basis_functions,
              invalid_integral_index,
              local_inactive_density_gradient.data());
        } else if (validate_integral_indices) {
          accumulate_ao_effective_one_electron_backprop_integral<true, false, false>(
              integral_index,
              unsymmetrized_gradient_storage_data,
              ao_two_electron_integral_values_data,
              ao_two_electron_integral_indices_data,
              nullptr,
              nullptr,
              column_offsets_data,
              n_basis_functions,
              invalid_integral_index,
              local_inactive_density_gradient.data());
        } else if (ao_two_electron_integral_symmetry_shifts != nullptr) {
          accumulate_ao_effective_one_electron_backprop_integral<false, true, false>(
              integral_index,
              unsymmetrized_gradient_storage_data,
              ao_two_electron_integral_values_data,
              ao_two_electron_integral_indices_data,
              ao_two_electron_integral_symmetry_shifts,
              nullptr,
              column_offsets_data,
              n_basis_functions,
              invalid_integral_index,
              local_inactive_density_gradient.data());
        } else {
          accumulate_ao_effective_one_electron_backprop_integral<false, false, false>(
              integral_index,
              unsymmetrized_gradient_storage_data,
              ao_two_electron_integral_values_data,
              ao_two_electron_integral_indices_data,
              nullptr,
              nullptr,
              column_offsets_data,
              n_basis_functions,
              invalid_integral_index,
              local_inactive_density_gradient.data());
        }
      }
    }
  }

  if (validate_integral_indices && invalid_integral_index.load() >= 0) {
    throw std::invalid_argument("AO two-electron index out of range");
  }

  if (n_threads > 1) {
    for (const auto& partial_gradient : partial_inactive_density_gradients) {
      for (std::size_t index = 0; index < inactive_density_gradient_storage.size(); ++index) {
        inactive_density_gradient_storage[index] += partial_gradient[index];
      }
    }
  }

  AoEffectiveOneElectronBackpropagationResult result;
  result.inactive_density_gradient.assign(
      inactive_density_gradient_storage.begin(),
      inactive_density_gradient_storage.end());
  return result;
}

AoEffectiveOneElectronBackpropagationResult
AoEffectiveOneElectronBackpropagator::backpropagate(
    const std::vector<double>& ao_effective_one_electron_gradient,
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_integral_indices,
    int n_basis_functions) const {
  return backpropagate_ao_effective_one_electron(
      ao_effective_one_electron_gradient,
      ao_two_electron_integral_values,
      ao_two_electron_integral_indices,
      nullptr,
      nullptr,
      n_basis_functions,
      true);
}

AoEffectiveOneElectronBackpropagationResult
AoEffectiveOneElectronBackpropagator::backpropagate(
    const Eigen::Ref<const Eigen::MatrixXd>& ao_effective_one_electron_gradient,
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<int>& ao_two_electron_integral_indices,
    int n_basis_functions) const {
  return backpropagate_ao_effective_one_electron(
      flatten_matrix_column_major(ao_effective_one_electron_gradient),
      ao_two_electron_integral_values,
      ao_two_electron_integral_indices,
      nullptr,
      nullptr,
      n_basis_functions,
      true);
}

AoEffectiveOneElectronBackpropagationResult
AoEffectiveOneElectronBackpropagator::backpropagate(
    const std::vector<double>& ao_effective_one_electron_gradient,
    const LibcintRiIntegralProviderResult& ri_integral_provider_result,
    int n_basis_functions) const {
  return backpropagate_ao_effective_one_electron_ri(
      ao_effective_one_electron_gradient,
      ri_integral_provider_result,
      n_basis_functions);
}

AoEffectiveOneElectronBackpropagationResult
AoEffectiveOneElectronBackpropagator::backpropagate(
    const Eigen::Ref<const Eigen::MatrixXd>& ao_effective_one_electron_gradient,
    const LibcintRiIntegralProviderResult& ri_integral_provider_result,
    int n_basis_functions) const {
  return backpropagate_ao_effective_one_electron_ri(
      flatten_matrix_column_major(ao_effective_one_electron_gradient),
      ri_integral_provider_result,
      n_basis_functions);
}

AoEffectiveOneElectronBackpropagationResult
AoEffectiveOneElectronBackpropagator::backpropagate(
    const std::vector<double>& active_one_electron_gradient,
    const OrbitalPreparationResult& orbital_result,
    const LibcintRiIntegralProviderResult& ri_integral_provider_result,
    int n_basis_functions,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) const {
  return backpropagate_ao_effective_one_electron_ri(
      active_one_electron_gradient,
      orbital_result,
      ri_integral_provider_result,
      n_basis_functions,
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);
}

AoEffectiveOneElectronBackpropagationResult
AoEffectiveOneElectronBackpropagator::backpropagate(
    const std::vector<double>& ao_effective_one_electron_gradient,
    const AoIntegralInput& ao_integral_input) const {
  const std::size_t matrix_size =
      ao_integral_input.n_basis_functions *
      ao_integral_input.n_basis_functions;
  if (ao_effective_one_electron_gradient.size() != matrix_size) {
    throw std::invalid_argument(
        "ao_effective_one_electron_gradient size mismatch");
  }

  int n_threads = 1;
#ifdef _OPENMP
  n_threads = xmvb::effective_openmp_thread_count();
#endif
  if (ao_effective_one_electron_graph_available(ao_integral_input)) {
    const Eigen::Map<const Eigen::MatrixXd> ao_effective_gradient(
        ao_effective_one_electron_gradient.data(),
        ao_integral_input.n_basis_functions,
        ao_integral_input.n_basis_functions);
    // The AO-H1E graph stores the unsymmetrized `G11` map, so the adjoint must
    // enter as `Lambda + Lambda^T` before applying `K^T`. The graph transpose
    // now has a multithreaded implementation, which keeps LBFGS/SCF gradient
    // paths on the same molecule-static operator used by exact_ctx.
    const Eigen::MatrixXd unsymmetrized_g11_gradient =
        ao_effective_gradient + ao_effective_gradient.transpose();
    AoEffectiveOneElectronBackpropagationResult result;
    result.inactive_density_gradient =
        apply_ao_effective_one_electron_graph_transpose(
            unsymmetrized_g11_gradient.data(),
            ao_integral_input,
            n_threads);
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
  return backpropagate_ao_effective_one_electron(
      ao_effective_one_electron_gradient,
      ao_integral_input.ao_two_electron_integral_values,
      ao_integral_input.ao_two_electron_integral_indices,
      ao_integral_input.ao_two_electron_integral_symmetry_shifts.empty()
          ? nullptr
          : ao_integral_input.ao_two_electron_integral_symmetry_shifts.data(),
      ao_integral_input.ao_effective_one_electron_linear_indices.empty()
          ? nullptr
          : ao_integral_input.ao_effective_one_electron_linear_indices.data(),
      ao_integral_input.n_basis_functions,
      validate_integral_indices);
}

AoEffectiveOneElectronBackpropagationResult
AoEffectiveOneElectronBackpropagator::backpropagate(
    const Eigen::Ref<const Eigen::MatrixXd>& ao_effective_one_electron_gradient,
    const AoIntegralInput& ao_integral_input) const {
  if (ao_effective_one_electron_gradient.rows() != ao_integral_input.n_basis_functions ||
      ao_effective_one_electron_gradient.cols() != ao_integral_input.n_basis_functions) {
    throw std::invalid_argument(
        "ao_effective_one_electron_gradient size mismatch");
  }
  return backpropagate(
      flatten_matrix_column_major(ao_effective_one_electron_gradient),
      ao_integral_input);
}

}  // namespace xmvb::vb
