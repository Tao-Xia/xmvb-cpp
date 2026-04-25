#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "pfaffian_vbscf/kernel/pf_fixed_ms_open_shell_kernel.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace {

using Matrix = xmvb::pfaffian_vbscf::Matrix;
using ScalarBuffer = xmvb::pfaffian_vbscf::ScalarBuffer;

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
  std::vector<bool> blocked_mask(dimension, false);
  for (const int orbital : blocked_alpha_orbitals) {
    blocked_mask[orbital] = true;
  }
  for (const int orbital : blocked_beta_orbitals) {
    blocked_mask[orbital] = true;
  }
  for (int col = 0; col < dimension; ++col) {
    if (blocked_mask[col]) {
      continue;
    }
    for (int row = col; row < dimension; ++row) {
      if (blocked_mask[row]) {
        continue;
      }
      const double value = (*distribution)(*generator);
      pair_block(row, col) = value;
      pair_block(col, row) = value;
    }
  }
  return pair_block;
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
  ScalarBuffer packed_integrals(storage_size, 0.0);
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
  constexpr double step = 1.0e-6;

  const int n_active_orbitals = static_cast<int>(spatial_overlap_matrix.rows());
  const int n_singlet_pairs = 1;
  const std::vector<int> left_blocked_alpha = {0};
  const std::vector<int> left_blocked_beta = {1};
  const std::vector<int> right_blocked_alpha = {2};
  const std::vector<int> right_blocked_beta = {3};
  const Matrix left_pair_ba =
      random_pair_block(
          n_active_orbitals,
          left_blocked_alpha,
          left_blocked_beta,
          generator,
          distribution);
  const Matrix right_pair_ab =
      random_pair_block(
          n_active_orbitals,
          right_blocked_alpha,
          right_blocked_beta,
          generator,
          distribution);
  const Matrix one_electron_matrix =
      random_symmetric_matrix(n_active_orbitals, generator, distribution);
  const ScalarBuffer packed_two_electron_integrals =
      random_packed_two_electron(n_active_orbitals, generator, distribution);

  const auto baseline =
      xmvb::pfaffian_vbscf::evaluate_fixed_ms_open_shell_pair_hamiltonian(
          left_pair_ba,
          right_pair_ab,
          left_blocked_alpha,
          left_blocked_beta,
          right_blocked_alpha,
          right_blocked_beta,
          spatial_overlap_matrix,
          one_electron_matrix,
          packed_two_electron_integrals,
          n_singlet_pairs,
          true);
  if (!baseline.gradients_computed) {
    throw std::runtime_error("fixed-M_s gradient result is not populated");
  }

  double max_overlap_spatial_error = 0.0;
  double max_total_spatial_error = 0.0;
  double max_one_electron_error = 0.0;
  double max_two_electron_error = 0.0;

  for (int col = 0; col < n_active_orbitals; ++col) {
    for (int row = 0; row < n_active_orbitals; ++row) {
      Matrix plus_overlap = spatial_overlap_matrix;
      Matrix minus_overlap = spatial_overlap_matrix;
      plus_overlap(row, col) += step;
      minus_overlap(row, col) -= step;

      const auto plus =
          xmvb::pfaffian_vbscf::evaluate_fixed_ms_open_shell_pair_hamiltonian(
              left_pair_ba,
              right_pair_ab,
              left_blocked_alpha,
              left_blocked_beta,
              right_blocked_alpha,
              right_blocked_beta,
              plus_overlap,
              one_electron_matrix,
              packed_two_electron_integrals,
              n_singlet_pairs);
      const auto minus =
          xmvb::pfaffian_vbscf::evaluate_fixed_ms_open_shell_pair_hamiltonian(
              left_pair_ba,
              right_pair_ab,
              left_blocked_alpha,
              left_blocked_beta,
              right_blocked_alpha,
              right_blocked_beta,
              minus_overlap,
              one_electron_matrix,
              packed_two_electron_integrals,
              n_singlet_pairs);

      const double fd_overlap =
          (plus.overlap - minus.overlap) / (2.0 * step);
      const double fd_total =
          (plus.total_hamiltonian - minus.total_hamiltonian) / (2.0 * step);
      max_overlap_spatial_error = std::max(
          max_overlap_spatial_error,
          std::abs(baseline.overlap_spatial_gradient(row, col) - fd_overlap));
      max_total_spatial_error = std::max(
          max_total_spatial_error,
          std::abs(baseline.total_hamiltonian_spatial_gradient(row, col) - fd_total));
    }
  }

  for (int col = 0; col < n_active_orbitals; ++col) {
    for (int row = 0; row < n_active_orbitals; ++row) {
      Matrix plus_one = one_electron_matrix;
      Matrix minus_one = one_electron_matrix;
      plus_one(row, col) += step;
      minus_one(row, col) -= step;

      const auto plus =
          xmvb::pfaffian_vbscf::evaluate_fixed_ms_open_shell_pair_hamiltonian(
              left_pair_ba,
              right_pair_ab,
              left_blocked_alpha,
              left_blocked_beta,
              right_blocked_alpha,
              right_blocked_beta,
              spatial_overlap_matrix,
              plus_one,
              packed_two_electron_integrals,
              n_singlet_pairs);
      const auto minus =
          xmvb::pfaffian_vbscf::evaluate_fixed_ms_open_shell_pair_hamiltonian(
              left_pair_ba,
              right_pair_ab,
              left_blocked_alpha,
              left_blocked_beta,
              right_blocked_alpha,
              right_blocked_beta,
              spatial_overlap_matrix,
              minus_one,
              packed_two_electron_integrals,
              n_singlet_pairs);

      const double fd_total =
          (plus.total_hamiltonian - minus.total_hamiltonian) / (2.0 * step);
      max_one_electron_error = std::max(
          max_one_electron_error,
          std::abs(baseline.one_electron_matrix_gradient(row, col) - fd_total));
    }
  }

  for (std::size_t index = 0; index < packed_two_electron_integrals.size(); ++index) {
    ScalarBuffer plus_two = packed_two_electron_integrals;
    ScalarBuffer minus_two = packed_two_electron_integrals;
    plus_two[index] += step;
    minus_two[index] -= step;

    const auto plus =
        xmvb::pfaffian_vbscf::evaluate_fixed_ms_open_shell_pair_hamiltonian(
            left_pair_ba,
            right_pair_ab,
            left_blocked_alpha,
            left_blocked_beta,
            right_blocked_alpha,
            right_blocked_beta,
            spatial_overlap_matrix,
            one_electron_matrix,
            plus_two,
            n_singlet_pairs);
    const auto minus =
        xmvb::pfaffian_vbscf::evaluate_fixed_ms_open_shell_pair_hamiltonian(
            left_pair_ba,
            right_pair_ab,
            left_blocked_alpha,
            left_blocked_beta,
            right_blocked_alpha,
            right_blocked_beta,
            spatial_overlap_matrix,
            one_electron_matrix,
            minus_two,
            n_singlet_pairs);

    const double fd_total =
        (plus.total_hamiltonian - minus.total_hamiltonian) / (2.0 * step);
    max_two_electron_error = std::max(
        max_two_electron_error,
        std::abs(baseline.packed_two_electron_gradient[index] - fd_total));
  }

  std::cout << name << " max_overlap_spatial_error = "
            << max_overlap_spatial_error << '\n';
  std::cout << name << " max_total_spatial_error = "
            << max_total_spatial_error << '\n';
  std::cout << name << " max_one_electron_error = "
            << max_one_electron_error << '\n';
  std::cout << name << " max_two_electron_error = "
            << max_two_electron_error << '\n';
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
