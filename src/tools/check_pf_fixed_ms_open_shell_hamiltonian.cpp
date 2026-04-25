#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/LU>

#include "pfaffian_vbscf/kernel/pf_fixed_ms_open_shell_kernel.hpp"
#include "vb/matrices/full_determinant_pair_evaluator.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace {

using Matrix = xmvb::pfaffian_vbscf::Matrix;
using ScalarBuffer = xmvb::pfaffian_vbscf::ScalarBuffer;

struct TestCase {
  std::string name;
  std::vector<int> left_blocked_alpha;
  std::vector<int> left_blocked_beta;
  std::vector<int> right_blocked_alpha;
  std::vector<int> right_blocked_beta;
  int n_singlet_pairs = 0;
};

struct DeterminantTerm {
  std::vector<int> alpha_occ;
  std::vector<int> beta_occ;
  double coefficient = 0.0;
};

struct ExactStatePairResult {
  double overlap = 0.0;
  double one_electron_hamiltonian = 0.0;
  double total_hamiltonian = 0.0;
};

int canonicalize_spin_string(std::vector<int>* occupied_orbitals) {
  if (occupied_orbitals == nullptr) {
    throw std::invalid_argument("occupied_orbitals must not be null");
  }
  int permutation_sign = 1;
  for (std::size_t left_index = 0; left_index + 1 < occupied_orbitals->size(); ++left_index) {
    for (std::size_t right_index = left_index + 1;
         right_index < occupied_orbitals->size();
         ++right_index) {
      if ((*occupied_orbitals)[left_index] > (*occupied_orbitals)[right_index]) {
        std::swap((*occupied_orbitals)[left_index], (*occupied_orbitals)[right_index]);
        permutation_sign = -permutation_sign;
      }
    }
  }
  return permutation_sign;
}

void enumerate_combinations_recursive(
    int start,
    int remaining,
    int dimension,
    std::vector<int>* current,
    std::vector<std::vector<int>>* combinations) {
  if (current == nullptr || combinations == nullptr) {
    throw std::invalid_argument("enumeration buffers must not be null");
  }
  if (remaining == 0) {
    combinations->push_back(*current);
    return;
  }
  for (int value = start; value <= dimension - remaining; ++value) {
    current->push_back(value);
    enumerate_combinations_recursive(
        value + 1,
        remaining - 1,
        dimension,
        current,
        combinations);
    current->pop_back();
  }
}

std::vector<std::vector<int>> enumerate_combinations(
    int dimension,
    int choose) {
  if (choose < 0 || choose > dimension) {
    throw std::invalid_argument("invalid combination size");
  }
  std::vector<std::vector<int>> combinations;
  std::vector<int> current;
  enumerate_combinations_recursive(
      0,
      choose,
      dimension,
      &current,
      &combinations);
  return combinations;
}

Matrix extract_submatrix(
    const Matrix& matrix,
    const std::vector<int>& rows,
    const std::vector<int>& cols) {
  Matrix submatrix =
      Matrix::Zero(
          static_cast<int>(rows.size()),
          static_cast<int>(cols.size()));
  for (int col = 0; col < static_cast<int>(cols.size()); ++col) {
    for (int row = 0; row < static_cast<int>(rows.size()); ++row) {
      submatrix(row, col) =
          matrix(rows[row],
                 cols[col]);
    }
  }
  return submatrix;
}

std::vector<double> flatten_matrix(const Matrix& matrix) {
  std::vector<double> data(
      matrix.rows() * matrix.cols(),
      0.0);
  for (int col = 0; col < matrix.cols(); ++col) {
    for (int row = 0; row < matrix.rows(); ++row) {
      data[col * matrix.rows() + row] = matrix(row, col);
    }
  }
  return data;
}

std::vector<DeterminantTerm> enumerate_state_terms(
    const Matrix& pair_matrix,
    const std::vector<int>& blocked_alpha_orbitals,
    const std::vector<int>& blocked_beta_orbitals,
    int n_singlet_pairs) {
  const int n_active_orbitals = static_cast<int>(pair_matrix.rows());
  if (pair_matrix.rows() != pair_matrix.cols()) {
    throw std::invalid_argument("pair_matrix must be square");
  }

  std::vector<DeterminantTerm> terms;
  const auto alpha_combinations =
      enumerate_combinations(n_active_orbitals, n_singlet_pairs);
  const auto beta_combinations =
      enumerate_combinations(n_active_orbitals, n_singlet_pairs);

  for (const auto& alpha_subset : alpha_combinations) {
    for (const auto& beta_subset : beta_combinations) {
      const Matrix pair_submatrix =
          extract_submatrix(pair_matrix, alpha_subset, beta_subset);
      const double pair_coefficient =
          Eigen::FullPivLU<Matrix>(pair_submatrix).determinant();
      if (std::abs(pair_coefficient) <= 1.0e-12) {
        continue;
      }

      DeterminantTerm term;
      term.alpha_occ = blocked_alpha_orbitals;
      term.alpha_occ.insert(
          term.alpha_occ.end(),
          alpha_subset.begin(),
          alpha_subset.end());
      term.beta_occ = blocked_beta_orbitals;
      term.beta_occ.insert(
          term.beta_occ.end(),
          beta_subset.begin(),
          beta_subset.end());
      term.coefficient = pair_coefficient;
      const int alpha_sign = canonicalize_spin_string(&term.alpha_occ);
      const int beta_sign = canonicalize_spin_string(&term.beta_occ);
      term.coefficient *= static_cast<double>(alpha_sign * beta_sign);
      terms.push_back(std::move(term));
    }
  }
  return terms;
}

ExactStatePairResult exact_state_pair_evaluation(
    const std::vector<DeterminantTerm>& left_terms,
    const std::vector<DeterminantTerm>& right_terms,
    const Matrix& spatial_overlap,
    const Matrix& one_electron,
    const ScalarBuffer& packed_two_electron_integrals) {
  const int n_active_orbitals = static_cast<int>(spatial_overlap.rows());
  const auto flat_overlap = flatten_matrix(spatial_overlap);
  const auto flat_one_electron = flatten_matrix(one_electron);
  const xmvb::vb::FullDeterminantPairEvaluator evaluator;

  ExactStatePairResult result;
  for (const auto& left_term : left_terms) {
    for (const auto& right_term : right_terms) {
      const auto determinant_pair =
          evaluator.evaluate(
              left_term.alpha_occ,
              right_term.alpha_occ,
              left_term.beta_occ,
              right_term.beta_occ,
              flat_overlap,
              flat_one_electron,
              n_active_orbitals,
              packed_two_electron_integrals);
      const double prefactor =
          left_term.coefficient * right_term.coefficient;
      result.overlap +=
          prefactor * determinant_pair.overlap_determinant;
      result.one_electron_hamiltonian +=
          prefactor * determinant_pair.one_electron_hamiltonian;
      result.total_hamiltonian +=
          prefactor * determinant_pair.total_hamiltonian;
    }
  }
  return result;
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

Matrix random_pair_matrix(
    int dimension,
    const std::vector<int>& blocked_alpha_orbitals,
    const std::vector<int>& blocked_beta_orbitals,
    std::mt19937* generator,
    std::normal_distribution<double>* distribution) {
  Matrix pair_matrix = Matrix::Zero(dimension, dimension);
  std::vector<bool> blocked_mask(dimension, false);
  for (const int orbital : blocked_alpha_orbitals) {
    blocked_mask[orbital] = true;
  }
  for (const int orbital : blocked_beta_orbitals) {
    blocked_mask[orbital] = true;
  }
  for (int col = 0; col < dimension; ++col) {
    for (int row = 0; row < dimension; ++row) {
      if (blocked_mask[row] ||
          blocked_mask[col]) {
        continue;
      }
      pair_matrix(row, col) = (*distribution)(*generator);
    }
  }
  return pair_matrix;
}

ScalarBuffer random_packed_two_electron_integrals(
    int n_active_orbitals,
    std::mt19937* generator,
    std::normal_distribution<double>* distribution) {
  const int size =
      xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
          n_active_orbitals - 1,
          n_active_orbitals - 1,
          n_active_orbitals - 1,
          n_active_orbitals - 1) +
      1;
  ScalarBuffer packed(size, 0.0);
  for (double& value : packed) {
    value = (*distribution)(*generator);
  }
  return packed;
}

Matrix build_left_pairing_matrix(const Matrix& left_pair_ba) {
  const int n_active_orbitals = static_cast<int>(left_pair_ba.rows());
  Matrix pairing_matrix =
      Matrix::Zero(2 * n_active_orbitals, 2 * n_active_orbitals);
  pairing_matrix.topRightCorner(
      n_active_orbitals,
      n_active_orbitals) =
      left_pair_ba.transpose();
  pairing_matrix.bottomLeftCorner(
      n_active_orbitals,
      n_active_orbitals) =
      -left_pair_ba;
  return pairing_matrix;
}

Matrix build_right_pairing_matrix(const Matrix& right_pair_ab) {
  const int n_active_orbitals = static_cast<int>(right_pair_ab.rows());
  Matrix pairing_matrix =
      Matrix::Zero(2 * n_active_orbitals, 2 * n_active_orbitals);
  pairing_matrix.topRightCorner(
      n_active_orbitals,
      n_active_orbitals) =
      right_pair_ab;
  pairing_matrix.bottomLeftCorner(
      n_active_orbitals,
      n_active_orbitals) =
      -right_pair_ab.transpose();
  return pairing_matrix;
}

void run_suite(bool use_near_identity_overlap) {
  constexpr int kDimension = 5;
  constexpr int kTrialsPerCase = 4;
  const std::vector<TestCase> test_cases = {
      {"closed", {}, {}, {}, {}, 2},
      {"high", {0}, {}, {2}, {}, 2},
      {"beta_only", {}, {0}, {}, {2}, 2},
      {"general", {0}, {1}, {2}, {3}, 2},
  };

  std::mt19937 generator(7 + (use_near_identity_overlap ? 17 : 0));
  std::normal_distribution<double> distribution(0.0, 1.0);

  double max_overlap_error = 0.0;
  double max_one_error = 0.0;
  double max_two_error = 0.0;
  double max_total_error = 0.0;
  double max_internal_overlap_error = 0.0;
  double max_internal_one_error = 0.0;

  for (const TestCase& test_case : test_cases) {
    for (int trial = 0; trial < kTrialsPerCase; ++trial) {
      const Matrix spatial_overlap =
          use_near_identity_overlap
              ? (Matrix::Identity(kDimension, kDimension) +
                 1.0e-5 *
                     random_symmetric_matrix(
                         kDimension,
                         &generator,
                         &distribution))
              : random_spd_matrix(kDimension, &generator, &distribution);
      const Matrix one_electron =
          random_symmetric_matrix(kDimension, &generator, &distribution);
      const ScalarBuffer packed_two_electron =
          random_packed_two_electron_integrals(
              kDimension,
              &generator,
              &distribution);

      const Matrix left_pair_ba =
          random_pair_matrix(
              kDimension,
              test_case.left_blocked_alpha,
              test_case.left_blocked_beta,
              &generator,
              &distribution);
      const Matrix right_pair_ab =
          random_pair_matrix(
              kDimension,
              test_case.right_blocked_alpha,
              test_case.right_blocked_beta,
              &generator,
              &distribution);

      const auto exact =
          exact_state_pair_evaluation(
              enumerate_state_terms(
                  left_pair_ba.transpose(),
                  test_case.left_blocked_alpha,
                  test_case.left_blocked_beta,
                  test_case.n_singlet_pairs),
              enumerate_state_terms(
                  right_pair_ab,
                  test_case.right_blocked_alpha,
                  test_case.right_blocked_beta,
                  test_case.n_singlet_pairs),
              spatial_overlap,
              one_electron,
              packed_two_electron);
      const auto candidate =
          xmvb::pfaffian_vbscf::evaluate_fixed_ms_open_shell_pair_hamiltonian(
              left_pair_ba,
              right_pair_ab,
              test_case.left_blocked_alpha,
              test_case.left_blocked_beta,
              test_case.right_blocked_alpha,
              test_case.right_blocked_beta,
              spatial_overlap,
              one_electron,
              packed_two_electron,
              test_case.n_singlet_pairs);
      const auto overlap_helper =
          xmvb::pfaffian_vbscf::evaluate_fixed_ms_open_shell_overlap_and_one_electron(
              left_pair_ba,
              right_pair_ab,
              test_case.left_blocked_alpha,
              test_case.left_blocked_beta,
              test_case.right_blocked_alpha,
              test_case.right_blocked_beta,
              spatial_overlap,
              one_electron,
              test_case.n_singlet_pairs);

      const double exact_two_electron =
          exact.total_hamiltonian -
          exact.one_electron_hamiltonian;
      const double overlap_error =
          std::abs(candidate.overlap - exact.overlap);
      const double one_error =
          std::abs(
              candidate.one_electron_hamiltonian -
              exact.one_electron_hamiltonian);
      const double two_error =
          std::abs(
              candidate.two_electron_hamiltonian -
              exact_two_electron);
      const double total_error =
          std::abs(
              candidate.total_hamiltonian -
              exact.total_hamiltonian);
      const double internal_overlap_error =
          std::abs(candidate.overlap - overlap_helper.overlap);
      const double internal_one_error =
          std::abs(
              candidate.one_electron_hamiltonian -
              overlap_helper.one_electron_hamiltonian);

      max_overlap_error = std::max(max_overlap_error, overlap_error);
      max_one_error = std::max(max_one_error, one_error);
      max_two_error = std::max(max_two_error, two_error);
      max_total_error = std::max(max_total_error, total_error);
      max_internal_overlap_error =
          std::max(max_internal_overlap_error, internal_overlap_error);
      max_internal_one_error =
          std::max(max_internal_one_error, internal_one_error);
    }
  }

      std::cout
      << (use_near_identity_overlap ? "near_identity" : "spd")
      << " max_overlap_error = " << std::setprecision(16)
      << max_overlap_error << '\n'
      << (use_near_identity_overlap ? "near_identity" : "spd")
      << " max_one_error = " << std::setprecision(16)
      << max_one_error << '\n'
      << (use_near_identity_overlap ? "near_identity" : "spd")
      << " max_two_error = " << std::setprecision(16)
      << max_two_error << '\n'
      << (use_near_identity_overlap ? "near_identity" : "spd")
      << " max_total_error = " << std::setprecision(16)
      << max_total_error << '\n'
      << (use_near_identity_overlap ? "near_identity" : "spd")
      << " max_internal_overlap_error = " << std::setprecision(16)
      << max_internal_overlap_error << '\n'
      << (use_near_identity_overlap ? "near_identity" : "spd")
      << " max_internal_one_error = " << std::setprecision(16)
      << max_internal_one_error << '\n';

  if (max_overlap_error > 1.0e-7 ||
      max_one_error > 1.0e-6 ||
      max_two_error > 1.0e-6 ||
      max_total_error > 1.0e-6) {
    throw std::runtime_error("fixed-M_s open-shell Hamiltonian checker failed");
  }
}

}  // namespace

int main() {
  run_suite(false);
  run_suite(true);
  return 0;
}
