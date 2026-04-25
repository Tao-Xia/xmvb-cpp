#include "pfaffian_vbscf/matrices/pf_matrix_builder.hpp"

#include <chrono>
#include <stdexcept>

#include "pfaffian_vbscf/kernel/pf_fixed_ms_open_shell_kernel.hpp"
#include "pfaffian_vbscf/kernel/pf_forward_kernel.hpp"
#include "pfaffian_vbscf/kernel/pf_high_spin_open_shell_hamiltonian.hpp"
#include "pfaffian_vbscf/math/antisymm_codec.hpp"
#include "pfaffian_vbscf/math/dense_utils.hpp"
#include "pfaffian_vbscf/matrices/pf_pair_kernels.hpp"

namespace xmvb::pfaffian_vbscf {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(const Clock::time_point& start_time) {
  return std::chrono::duration<double>(Clock::now() - start_time).count();
}

Matrix dense_mat(
    const ScalarBuffer& data,
    int dimension,
    const char* label) {
  const std::size_t expected_size =
      dimension * dimension;
  if (data.size() != expected_size) {
    throw std::invalid_argument(std::string(label) + " size does not match the active-space dimension");
  }

  Matrix matrix = Matrix::Zero(dimension, dimension);
  for (int col = 0; col < dimension; ++col) {
    for (int row = 0; row < dimension; ++row) {
      matrix(row, col) = data[col * dimension + row];
    }
  }
  return matrix;
}

std::size_t linear_index(
    int row,
    int col,
    int dimension) {
  return col * dimension + row;
}

std::size_t lower_triangle_index(
    int row,
    int col) {
  return row * (row + 1) / 2 + col;
}

bool basis_contains_open_shell_states(const PfBasisData& basis) {
  for (const PfState& state : basis.states) {
    if (state.has_blocked_open_shell()) {
      return true;
    }
  }
  return basis.has_blocked_open_shell();
}

bool supports_high_spin_blocked_alpha_basis(const PfBasisData& basis) {
  if (!basis_contains_open_shell_states(basis)) {
    return false;
  }
  if (basis.n_blocked_beta != 0 || basis.n_beta > basis.n_alpha) {
    return false;
  }
  if (basis.n_blocked_alpha != basis.n_alpha - basis.n_beta) {
    return false;
  }
  for (const PfState& state : basis.states) {
    if (!state.blocked_beta_orbitals.empty() ||
        state.n_blocked_alpha() != basis.n_blocked_alpha ||
        state.n_singlet_pairs != basis.n_singlet_pairs) {
      return false;
    }
  }
  return true;
}

bool supports_general_fixed_ms_open_shell_basis(const PfBasisData& basis) {
  if (!basis_contains_open_shell_states(basis)) {
    return false;
  }
  if (basis.n_singlet_pairs < 0 ||
      basis.n_blocked_alpha < 0 ||
      basis.n_blocked_beta < 0) {
    return false;
  }
  if (basis.n_alpha != basis.n_singlet_pairs + basis.n_blocked_alpha ||
      basis.n_beta != basis.n_singlet_pairs + basis.n_blocked_beta) {
    return false;
  }
  for (const PfState& state : basis.states) {
    if (state.n_blocked_alpha() != basis.n_blocked_alpha ||
        state.n_blocked_beta() != basis.n_blocked_beta ||
        state.n_singlet_pairs != basis.n_singlet_pairs) {
      return false;
    }
  }
  return true;
}

PfPairKernelResult evaluate_pf_pair_kernel_from_cache(
    const PfKernelCache& cache,
    const ConstMatrixRef& one_electron_matrix,
    const PfActiveSpaceData& active_space) {
  PfPairKernelResult result;
  result.overlap = cache.overlap_value;
  result.one_electron_hamiltonian =
      cache.one_rdm.cwiseProduct(one_electron_matrix).sum();
  if (active_space.two_electron_representation ==
      xmvb::vb::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity) {
    result.total_hamiltonian =
        result.one_electron_hamiltonian +
        PfForwardKernel::evaluate_closed_shell_two_electron(
            cache,
            active_space.ggo);
  } else {
    result.total_hamiltonian =
        result.one_electron_hamiltonian +
        PfForwardKernel::evaluate_closed_shell_two_electron(
            cache,
            active_space.ggo);
  }
  return result;
}

}  // namespace

PfMatrixBuildResult PfMatrixBuilder::build(
    const PfBasisData& basis,
    const PfActiveSpaceData& active_space,
    bool store_lower_triangle_pair_caches) const {
  if (basis.n_states <= 0) {
    throw std::invalid_argument("basis.n_states must be positive");
  }
  if (static_cast<int>(basis.states.size()) != basis.n_states) {
    throw std::invalid_argument("basis.states size does not match basis.n_states");
  }
  if (basis.n_active_orbitals != active_space.n_active_orbitals) {
    throw std::invalid_argument("basis and active-space dimensions do not match");
  }
  if (basis.n_alpha != active_space.n_alpha || basis.n_beta != active_space.n_beta) {
    throw std::invalid_argument("basis electron counts do not match the active-space payload");
  }
  const bool use_closed_shell_cache_path = !basis_contains_open_shell_states(basis);
  const bool use_high_spin_open_shell_path =
      !use_closed_shell_cache_path &&
      supports_high_spin_blocked_alpha_basis(basis);
  const bool use_general_fixed_ms_open_shell_path =
      !use_closed_shell_cache_path &&
      !use_high_spin_open_shell_path &&
      supports_general_fixed_ms_open_shell_basis(basis);
  if (!use_closed_shell_cache_path &&
      !use_high_spin_open_shell_path &&
      !use_general_fixed_ms_open_shell_path) {
    throw std::invalid_argument(
        "PfMatrixBuilder currently supports closed-shell states and "
        "fixed-M_s open-shell states with consistent blocked alpha/beta counts only");
  }

  const int n_states = basis.n_states;
  const int n_active_orbitals = active_space.n_active_orbitals;
  const int n_pairs = basis.n_singlet_pairs;
  const Matrix spatial_overlap_matrix =
      dense_mat(active_space.sso, n_active_orbitals, "active_space.sso");
  const Matrix one_electron_matrix =
      dense_mat(active_space.hho, n_active_orbitals, "active_space.hho");
  if (!use_closed_shell_cache_path &&
      active_space.two_electron_representation ==
          xmvb::vb::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity) {
    throw std::invalid_argument(
        "PfMatrixBuilder RI two-electron representation is currently supported "
        "only on the closed-shell cache path");
  }

  std::vector<Matrix> pairing_matrices;
  std::vector<Matrix> pairing_matrices_transpose;
  std::vector<Matrix> closed_shell_right_ab_blocks;
  std::vector<Matrix> closed_shell_left_ba_blocks;
  if (use_closed_shell_cache_path) {
    if (store_lower_triangle_pair_caches) {
      pairing_matrices.reserve(n_states);
      pairing_matrices_transpose.reserve(n_states);
    } else {
      closed_shell_right_ab_blocks.reserve(n_states);
      closed_shell_left_ba_blocks.reserve(n_states);
    }
  }
  for (const PfState& state : basis.states) {
    if (state.n_active_orbitals != n_active_orbitals ||
        state.n_spin_orbitals != 2 * n_active_orbitals) {
      throw std::invalid_argument("PfState dimensions do not match the basis");
    }
    if (use_closed_shell_cache_path) {
      if (state.has_blocked_open_shell()) {
        throw std::invalid_argument(
            "PfMatrixBuilder encountered a blocked open-shell PfState while the basis is closed-shell");
      }
      if (store_lower_triangle_pair_caches) {
        pairing_matrices.push_back(decode_pf_state(state));
        pairing_matrices_transpose.push_back(
            pairing_matrices.back().transpose().eval());
      } else {
        closed_shell_right_ab_blocks.push_back(decode_pf_alpha_beta_block(state));
        closed_shell_left_ba_blocks.push_back(
            closed_shell_right_ab_blocks.back().transpose().eval());
      }
    } else {
      if (state.n_blocked_alpha() != basis.n_blocked_alpha ||
          state.n_blocked_beta() != basis.n_blocked_beta ||
          state.n_singlet_pairs != basis.n_singlet_pairs) {
        throw std::invalid_argument(
            "PfMatrixBuilder encountered an unsupported open-shell PfState");
      }
    }
  }

  PfMatrixBuildResult result;
  result.n_states = n_states;
  result.s.assign(n_states * n_states, 0.0);
  result.h.assign(n_states * n_states, 0.0);
  const std::size_t lower_triangle_size =
      n_states * (n_states + 1) / 2;
  if (use_closed_shell_cache_path && store_lower_triangle_pair_caches) {
    result.lower_triangle_pair_caches.resize(lower_triangle_size);
  }

  const Clock::time_point start_time = Clock::now();
  if (use_closed_shell_cache_path) {
    if (store_lower_triangle_pair_caches) {
      const Matrix spin_metric =
          build_spin_block_diagonal_metric(spatial_overlap_matrix);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int row = 0; row < n_states; ++row) {
        const Matrix& left_pairing =
            pairing_matrices_transpose[row];
        for (int col = 0; col <= row; ++col) {
          const std::size_t pair_index = lower_triangle_index(row, col);
          const Matrix& right_pairing =
              pairing_matrices[col];
          PfKernelCache cache =
              PfForwardKernel::build_closed_shell_exact_cache(
                  left_pairing,
                  spin_metric,
                  right_pairing,
                  n_pairs);
          const PfPairKernelResult pair_result =
              evaluate_pf_pair_kernel_from_cache(
                  cache,
                  one_electron_matrix,
                  active_space);
          const std::size_t upper_index = linear_index(row, col, n_states);
          const std::size_t lower_index = linear_index(col, row, n_states);
          result.s[upper_index] = pair_result.overlap;
          result.h[upper_index] = pair_result.total_hamiltonian;
          result.s[lower_index] = pair_result.overlap;
          result.h[lower_index] = pair_result.total_hamiltonian;
          result.lower_triangle_pair_caches[pair_index] = std::move(cache);
        }
      }
    } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int row = 0; row < n_states; ++row) {
        const Matrix& left_ba =
            closed_shell_left_ba_blocks[row];
        for (int col = 0; col <= row; ++col) {
          const Matrix& right_ab =
              closed_shell_right_ab_blocks[col];
          const PfKernelCache cache =
              PfForwardKernel::build_closed_shell_exact_spatial_cache(
                  left_ba,
                  spatial_overlap_matrix,
                  right_ab,
                  n_pairs);
          const PfPairKernelResult pair_result =
              evaluate_pf_pair_kernel_from_cache(
                  cache,
                  one_electron_matrix,
                  active_space);
          const std::size_t upper_index = linear_index(row, col, n_states);
          const std::size_t lower_index = linear_index(col, row, n_states);
          result.s[upper_index] = pair_result.overlap;
          result.h[upper_index] = pair_result.total_hamiltonian;
          result.s[lower_index] = pair_result.overlap;
          result.h[lower_index] = pair_result.total_hamiltonian;
        }
      }
    }
    result.pair_profile.forward_pair_calls = lower_triangle_size;
    result.pair_profile.forward_sample_calls = lower_triangle_size;
  } else if (use_high_spin_open_shell_path) {
    for (int row = 0; row < n_states; ++row) {
      for (int col = 0; col <= row; ++col) {
        const PfHighSpinOpenShellHamiltonianResult pair_result =
            evaluate_high_spin_open_shell_pf_state_pair_hamiltonian(
                basis.states[row],
                basis.states[col],
                spatial_overlap_matrix,
                one_electron_matrix,
                active_space.ggo,
                false);
        const std::size_t upper_index = linear_index(row, col, n_states);
        const std::size_t lower_index = linear_index(col, row, n_states);
        result.s[upper_index] = pair_result.overlap;
        result.h[upper_index] = pair_result.total_hamiltonian;
        result.s[lower_index] = pair_result.overlap;
        result.h[lower_index] = pair_result.total_hamiltonian;
        ++result.pair_profile.forward_pair_calls;
        result.pair_profile.forward_sample_calls +=
            pair_result.interpolation_degree + 1;
      }
    }
  } else {
    for (int row = 0; row < n_states; ++row) {
      for (int col = 0; col <= row; ++col) {
        const PfFixedMsOpenShellHamiltonianResult pair_result =
            evaluate_fixed_ms_open_shell_pf_state_pair_hamiltonian(
                basis.states[row],
                basis.states[col],
                spatial_overlap_matrix,
                one_electron_matrix,
                active_space.ggo);
        const std::size_t upper_index = linear_index(row, col, n_states);
        const std::size_t lower_index = linear_index(col, row, n_states);
        result.s[upper_index] = pair_result.overlap;
        result.h[upper_index] = pair_result.total_hamiltonian;
        result.s[lower_index] = pair_result.overlap;
        result.h[lower_index] = pair_result.total_hamiltonian;
        ++result.pair_profile.forward_pair_calls;
        result.pair_profile.forward_sample_calls +=
            pair_result.interpolation_degree + 1;
      }
    }
  }
  result.pair_profile.forward_pair_total_dt = seconds_since(start_time);
  return result;
}

}  // namespace xmvb::pfaffian_vbscf
