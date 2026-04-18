#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "pfaffian_vbscf/kernel/pf_fixed_ms_open_shell_kernel.hpp"
#include "pfaffian_vbscf/math/antisymm_codec.hpp"
#include "pfaffian_vbscf/matrices/pf_matrix_builder.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace {

using Matrix = xmvb::pfaffian_vbscf::Matrix;
using PfActiveSpaceData = xmvb::pfaffian_vbscf::PfActiveSpaceData;
using PfBasisData = xmvb::pfaffian_vbscf::PfBasisData;
using PfMatrixBuilder = xmvb::pfaffian_vbscf::PfMatrixBuilder;
using PfState = xmvb::pfaffian_vbscf::PfState;
using ScalarBuffer = xmvb::pfaffian_vbscf::ScalarBuffer;

std::size_t linear_index(
    int row,
    int col,
    int dimension) {
  return xmvb::to_size(col) * dimension + row;
}

ScalarBuffer flatten_matrix(const Matrix& matrix) {
  ScalarBuffer data(
      xmvb::to_size(matrix.rows()) * matrix.cols(),
      0.0);
  for (int col = 0; col < matrix.cols(); ++col) {
    for (int row = 0; row < matrix.rows(); ++row) {
      data[xmvb::to_size(col) * matrix.rows() + row] = matrix(row, col);
    }
  }
  return data;
}

Matrix random_spd_matrix(
    int dimension,
    std::mt19937* generator,
    std::normal_distribution<double>* distribution) {
  Matrix random_matrix = Matrix::Zero(dimension, dimension);
  for (int col = 0; col < dimension; ++col) {
    for (int row = 0; row < dimension; ++row) {
      random_matrix(row, col) = (*distribution)(*generator);
    }
  }
  const Matrix spd =
      random_matrix.transpose() * random_matrix +
      0.5 * Matrix::Identity(dimension, dimension);
  return 0.5 * (spd + spd.transpose());
}

Matrix random_symmetric_matrix(
    int dimension,
    std::mt19937* generator,
    std::normal_distribution<double>* distribution) {
  Matrix matrix = Matrix::Zero(dimension, dimension);
  for (int col = 0; col < dimension; ++col) {
    for (int row = col; row < dimension; ++row) {
      const double value = (*distribution)(*generator);
      matrix(row, col) = value;
      matrix(col, row) = value;
    }
  }
  return matrix;
}

Matrix random_pair_block(
    int dimension,
    const std::vector<int>& blocked_alpha_orbitals,
    const std::vector<int>& blocked_beta_orbitals,
    std::mt19937* generator,
    std::normal_distribution<double>* distribution) {
  Matrix pair_block = Matrix::Zero(dimension, dimension);
  std::vector<bool> blocked_mask(xmvb::to_size(dimension), false);
  for (const int orbital : blocked_alpha_orbitals) {
    blocked_mask[xmvb::to_size(orbital)] = true;
  }
  for (const int orbital : blocked_beta_orbitals) {
    blocked_mask[xmvb::to_size(orbital)] = true;
  }
  for (int col = 0; col < dimension; ++col) {
    if (blocked_mask[xmvb::to_size(col)]) {
      continue;
    }
    for (int row = col; row < dimension; ++row) {
      if (blocked_mask[xmvb::to_size(row)]) {
        continue;
      }
      const double value = (*distribution)(*generator);
      pair_block(row, col) = value;
      pair_block(col, row) = value;
    }
  }
  return pair_block;
}

Matrix build_pairing_matrix(const Matrix& alpha_beta_block) {
  const int n_active_orbitals = static_cast<int>(alpha_beta_block.rows());
  Matrix pairing_matrix =
      Matrix::Zero(2 * n_active_orbitals, 2 * n_active_orbitals);
  pairing_matrix.topRightCorner(n_active_orbitals, n_active_orbitals) =
      alpha_beta_block;
  pairing_matrix.bottomLeftCorner(n_active_orbitals, n_active_orbitals) =
      -alpha_beta_block.transpose();
  return pairing_matrix;
}

PfState build_state(
    int n_active_orbitals,
    int n_singlet_pairs,
    const std::vector<int>& blocked_alpha_orbitals,
    const std::vector<int>& blocked_beta_orbitals,
    const Matrix& alpha_beta_block) {
  PfState state;
  state.n_active_orbitals = n_active_orbitals;
  state.n_spin_orbitals = 2 * n_active_orbitals;
  state.n_singlet_pairs = n_singlet_pairs;
  state.blocked_alpha_orbitals = blocked_alpha_orbitals;
  state.blocked_beta_orbitals = blocked_beta_orbitals;
  state.packed_entries =
      xmvb::pfaffian_vbscf::encode_antisymm(build_pairing_matrix(alpha_beta_block));
  return state;
}

ScalarBuffer random_packed_two_electron(
    int n_active_orbitals,
    std::mt19937* generator,
    std::normal_distribution<double>* distribution) {
  const int last_pair_index =
      xmvb::vb::TwoElectronIndexer::packed_pair_index(
          n_active_orbitals - 1,
          n_active_orbitals - 1);
  const int storage_size =
      xmvb::vb::TwoElectronIndexer::packed_pair_of_pairs_index(
          last_pair_index,
          last_pair_index) +
      1;
  ScalarBuffer packed_integrals(xmvb::to_size(storage_size), 0.0);
  for (double& value : packed_integrals) {
    value = (*distribution)(*generator);
  }
  return packed_integrals;
}

void run_case(
    const std::string& name,
    const Matrix& spatial_overlap_matrix,
    std::mt19937* generator,
    std::normal_distribution<double>* distribution) {
  const int n_active_orbitals = static_cast<int>(spatial_overlap_matrix.rows());
  const int n_singlet_pairs = 1;
  const std::vector<std::vector<int>> blocked_alpha_list = {
      {0},
      {2},
      {1},
  };
  const std::vector<std::vector<int>> blocked_beta_list = {
      {1},
      {3},
      {2},
  };

  PfBasisData basis;
  basis.n_states = static_cast<int>(blocked_alpha_list.size());
  basis.n_active_orbitals = n_active_orbitals;
  basis.n_singlet_pairs = n_singlet_pairs;
  basis.n_blocked_alpha = 1;
  basis.n_blocked_beta = 1;
  basis.n_alpha = basis.n_singlet_pairs + basis.n_blocked_alpha;
  basis.n_beta = basis.n_singlet_pairs + basis.n_blocked_beta;
  basis.states.reserve(xmvb::to_size(basis.n_states));

  for (int state_index = 0; state_index < basis.n_states; ++state_index) {
    const Matrix alpha_beta_block =
        random_pair_block(
            n_active_orbitals,
            blocked_alpha_list[xmvb::to_size(state_index)],
            blocked_beta_list[xmvb::to_size(state_index)],
            generator,
            distribution);
    basis.states.push_back(
        build_state(
            n_active_orbitals,
            n_singlet_pairs,
            blocked_alpha_list[xmvb::to_size(state_index)],
            blocked_beta_list[xmvb::to_size(state_index)],
            alpha_beta_block));
  }

  PfActiveSpaceData active_space;
  active_space.n_active_orbitals = n_active_orbitals;
  active_space.n_alpha = basis.n_alpha;
  active_space.n_beta = basis.n_beta;
  const Matrix one_electron_matrix =
      random_symmetric_matrix(n_active_orbitals, generator, distribution);
  active_space.sso = flatten_matrix(spatial_overlap_matrix);
  active_space.hho = flatten_matrix(one_electron_matrix);
  active_space.ggo =
      random_packed_two_electron(n_active_orbitals, generator, distribution);

  PfMatrixBuilder matrix_builder;
  const auto mats = matrix_builder.build(basis, active_space, false);

  double max_overlap_error = 0.0;
  double max_total_error = 0.0;
  int expected_sample_calls = 0;
  for (int row = 0; row < basis.n_states; ++row) {
    for (int col = 0; col <= row; ++col) {
      const auto pair_result =
          xmvb::pfaffian_vbscf::evaluate_fixed_ms_open_shell_pf_state_pair_hamiltonian(
              basis.states[xmvb::to_size(row)],
              basis.states[xmvb::to_size(col)],
              spatial_overlap_matrix,
              one_electron_matrix,
              active_space.ggo);
      expected_sample_calls += pair_result.interpolation_degree + 1;

      const std::size_t upper_index = linear_index(row, col, basis.n_states);
      const std::size_t lower_index = linear_index(col, row, basis.n_states);
      max_overlap_error = std::max(
          max_overlap_error,
          std::abs(mats.s[upper_index] - pair_result.overlap));
      max_overlap_error = std::max(
          max_overlap_error,
          std::abs(mats.s[lower_index] - pair_result.overlap));
      max_total_error = std::max(
          max_total_error,
          std::abs(mats.h[upper_index] - pair_result.total_hamiltonian));
      max_total_error = std::max(
          max_total_error,
          std::abs(mats.h[lower_index] - pair_result.total_hamiltonian));
    }
  }

  if (mats.pair_profile.forward_pair_calls != 6) {
    throw std::runtime_error("unexpected forward_pair_calls in matrix-builder dispatch");
  }
  if (mats.pair_profile.forward_sample_calls != expected_sample_calls) {
    throw std::runtime_error("unexpected forward_sample_calls in matrix-builder dispatch");
  }
  if (max_overlap_error > 1.0e-12 || max_total_error > 1.0e-12) {
    throw std::runtime_error("matrix-builder dispatch does not match direct fixed-M_s evaluation");
  }

  std::cout << name << " max_overlap_error = " << max_overlap_error << '\n';
  std::cout << name << " max_total_error = " << max_total_error << '\n';
}

}  // namespace

int main() {
  try {
    std::mt19937 generator(20260329);
    std::normal_distribution<double> distribution(0.0, 1.0);

    std::cout << std::setprecision(15);
    run_case(
        "spd",
        random_spd_matrix(4, &generator, &distribution),
        &generator,
        &distribution);

    const Matrix near_identity =
        Matrix::Identity(4, 4) +
        1.0e-5 * random_symmetric_matrix(4, &generator, &distribution);
    run_case(
        "near_identity",
        near_identity,
        &generator,
        &distribution);
    return 0;
  } catch (const std::exception& err) {
    std::cerr << err.what() << '\n';
    return 1;
  }
}
