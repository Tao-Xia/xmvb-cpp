#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/LU>

#include "pfaffian_vbscf/kernel/pf_fixed_ms_open_shell_kernel.hpp"

namespace {

using Matrix = xmvb::pfaffian_vbscf::Matrix;

struct DeterminantTerm {
  std::vector<int> alpha_occ;
  std::vector<int> beta_occ;
  double coefficient = 0.0;
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
          matrix(rows[xmvb::to_size(row)],
                 cols[xmvb::to_size(col)]);
    }
  }
  return submatrix;
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

struct DeterminantOverlapResult {
  double overlap = 0.0;
  Matrix gradient;
};

DeterminantOverlapResult determinant_overlap_and_gradient(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const Matrix& spatial_overlap_matrix) {
  if (left_occ.size() != right_occ.size()) {
    throw std::invalid_argument("left/right determinant occupations must match");
  }

  DeterminantOverlapResult result;
  result.gradient =
      Matrix::Zero(
          spatial_overlap_matrix.rows(),
          spatial_overlap_matrix.cols());
  if (left_occ.empty()) {
    result.overlap = 1.0;
    return result;
  }

  const Matrix overlap_submatrix =
      extract_submatrix(spatial_overlap_matrix, left_occ, right_occ);
  const Eigen::FullPivLU<Matrix> lu(overlap_submatrix);
  result.overlap = lu.determinant();
  if (std::abs(result.overlap) <= 1.0e-14) {
    return result;
  }

  const Matrix inverse = overlap_submatrix.inverse();
  for (int col = 0; col < static_cast<int>(right_occ.size()); ++col) {
    for (int row = 0; row < static_cast<int>(left_occ.size()); ++row) {
      result.gradient(
          left_occ[xmvb::to_size(row)],
          right_occ[xmvb::to_size(col)]) +=
          result.overlap * inverse(col, row);
    }
  }
  return result;
}

struct ExactStatePairResult {
  double overlap = 0.0;
  Matrix gradient;
};

ExactStatePairResult exact_overlap_and_gradient(
    const std::vector<DeterminantTerm>& left_terms,
    const std::vector<DeterminantTerm>& right_terms,
    const Matrix& spatial_overlap_matrix) {
  ExactStatePairResult result;
  result.gradient =
      Matrix::Zero(
          spatial_overlap_matrix.rows(),
          spatial_overlap_matrix.cols());

  for (const auto& left_term : left_terms) {
    for (const auto& right_term : right_terms) {
      const DeterminantOverlapResult alpha =
          determinant_overlap_and_gradient(
              left_term.alpha_occ,
              right_term.alpha_occ,
              spatial_overlap_matrix);
      const DeterminantOverlapResult beta =
          determinant_overlap_and_gradient(
              left_term.beta_occ,
              right_term.beta_occ,
              spatial_overlap_matrix);
      const double prefactor =
          left_term.coefficient * right_term.coefficient;
      result.overlap += prefactor * alpha.overlap * beta.overlap;
      result.gradient.noalias() +=
          prefactor *
          (alpha.gradient * beta.overlap + beta.gradient * alpha.overlap);
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

Matrix random_pair_matrix(
    int dimension,
    const std::vector<int>& blocked_alpha_orbitals,
    const std::vector<int>& blocked_beta_orbitals,
    std::mt19937* generator,
    std::normal_distribution<double>* distribution) {
  Matrix pair_matrix = Matrix::Zero(dimension, dimension);
  std::vector<bool> blocked_mask(xmvb::to_size(dimension), false);
  for (const int orbital : blocked_alpha_orbitals) {
    blocked_mask[xmvb::to_size(orbital)] = true;
  }
  for (const int orbital : blocked_beta_orbitals) {
    blocked_mask[xmvb::to_size(orbital)] = true;
  }
  for (int col = 0; col < dimension; ++col) {
    for (int row = 0; row < dimension; ++row) {
      if (blocked_mask[xmvb::to_size(row)] ||
          blocked_mask[xmvb::to_size(col)]) {
        continue;
      }
      pair_matrix(row, col) = (*distribution)(*generator);
    }
  }
  return pair_matrix;
}

double max_abs_diff(
    const Matrix& left,
    const Matrix& right) {
  if (left.rows() != right.rows() || left.cols() != right.cols()) {
    throw std::invalid_argument("matrix dimensions must match");
  }
  return (left - right).cwiseAbs().maxCoeff();
}

void run_case(
    int n_active_orbitals,
    int n_singlet_pairs,
    const std::vector<int>& left_blocked_alpha_orbitals,
    const std::vector<int>& left_blocked_beta_orbitals,
    const std::vector<int>& right_blocked_alpha_orbitals,
    const std::vector<int>& right_blocked_beta_orbitals,
    std::mt19937* generator,
    std::normal_distribution<double>* distribution,
    double* max_overlap_error,
    double* max_gradient_error,
    double* max_one_electron_error) {
  if (max_overlap_error == nullptr ||
      max_gradient_error == nullptr ||
      max_one_electron_error == nullptr) {
    throw std::invalid_argument("error trackers must not be null");
  }

  const Matrix spatial_overlap =
      random_spd_matrix(n_active_orbitals, generator, distribution);
  const Matrix one_electron =
      random_spd_matrix(n_active_orbitals, generator, distribution);
  const Matrix left_pair =
      random_pair_matrix(
          n_active_orbitals,
          left_blocked_alpha_orbitals,
          left_blocked_beta_orbitals,
          generator,
          distribution);
  const Matrix right_pair =
      random_pair_matrix(
          n_active_orbitals,
          right_blocked_alpha_orbitals,
          right_blocked_beta_orbitals,
          generator,
          distribution);

  const auto left_terms =
      enumerate_state_terms(
          left_pair,
          left_blocked_alpha_orbitals,
          left_blocked_beta_orbitals,
          n_singlet_pairs);
  const auto right_terms =
      enumerate_state_terms(
          right_pair,
          right_blocked_alpha_orbitals,
          right_blocked_beta_orbitals,
          n_singlet_pairs);
  const ExactStatePairResult exact =
      exact_overlap_and_gradient(
          left_terms,
          right_terms,
          spatial_overlap);
  const auto candidate =
      xmvb::pfaffian_vbscf::evaluate_fixed_ms_open_shell_overlap_and_one_electron(
          left_pair.transpose(),
          right_pair,
          left_blocked_alpha_orbitals,
          left_blocked_beta_orbitals,
          right_blocked_alpha_orbitals,
          right_blocked_beta_orbitals,
          spatial_overlap,
          one_electron,
          n_singlet_pairs);
  const double exact_one_electron =
      exact.gradient.cwiseProduct(one_electron).sum();

  *max_overlap_error =
      std::max(
          *max_overlap_error,
          std::abs(candidate.overlap - exact.overlap));
  *max_gradient_error =
      std::max(
          *max_gradient_error,
          max_abs_diff(candidate.spatial_overlap_gradient, exact.gradient));
  *max_one_electron_error =
      std::max(
          *max_one_electron_error,
          std::abs(candidate.one_electron_hamiltonian - exact_one_electron));
}

}  // namespace

int main() {
  try {
    std::mt19937 generator(20260329);
    std::normal_distribution<double> distribution(0.0, 1.0);

    double max_overlap_error = 0.0;
    double max_gradient_error = 0.0;
    double max_one_electron_error = 0.0;

    for (int repeat = 0; repeat < 8; ++repeat) {
      run_case(4, 1, {}, {}, {}, {}, &generator, &distribution, &max_overlap_error, &max_gradient_error, &max_one_electron_error);
      run_case(4, 1, {0}, {}, {1}, {}, &generator, &distribution, &max_overlap_error, &max_gradient_error, &max_one_electron_error);
      run_case(4, 1, {}, {0}, {}, {1}, &generator, &distribution, &max_overlap_error, &max_gradient_error, &max_one_electron_error);
      run_case(4, 1, {0}, {1}, {2}, {3}, &generator, &distribution, &max_overlap_error, &max_gradient_error, &max_one_electron_error);
      run_case(5, 2, {}, {}, {}, {}, &generator, &distribution, &max_overlap_error, &max_gradient_error, &max_one_electron_error);
      run_case(5, 1, {0}, {1}, {2}, {3}, &generator, &distribution, &max_overlap_error, &max_gradient_error, &max_one_electron_error);
      run_case(6, 2, {0}, {1}, {2}, {3}, &generator, &distribution, &max_overlap_error, &max_gradient_error, &max_one_electron_error);
    }

    std::cout << std::setprecision(16);
    std::cout << "max_overlap_error = " << max_overlap_error << '\n';
    std::cout << "max_gradient_error = " << max_gradient_error << '\n';
    std::cout << "max_one_electron_error = " << max_one_electron_error << '\n';

    const double tolerance = 1.0e-6;
    if (max_overlap_error > tolerance ||
        max_gradient_error > tolerance ||
        max_one_electron_error > tolerance) {
      std::cerr << "fixed-M_s open-shell overlap validation failed\n";
      return 1;
    }
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "error: " << exception.what() << '\n';
    return 1;
  }
}
