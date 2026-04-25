#include "pfaffian_vbscf/scf/pf_spin_coupling_builder.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Eigenvalues>

namespace xmvb::pfaffian_vbscf {

namespace {

std::vector<std::vector<int>> enumerate_combinations(
    int n_items,
    int n_selected) {
  if (n_items < 0 || n_selected < 0 || n_selected > n_items) {
    throw std::invalid_argument("invalid spin-string combination request");
  }

  std::vector<std::vector<int>> combinations;
  std::vector<int> current;
  current.reserve(n_selected);

  const auto recurse =
      [&](const auto& self, int start_index, int remaining) -> void {
    if (remaining == 0) {
      combinations.push_back(current);
      return;
    }
    for (int index = start_index; index <= n_items - remaining; ++index) {
      current.push_back(index);
      self(self, index + 1, remaining - 1);
      current.pop_back();
    }
  };
  recurse(recurse, 0, n_selected);
  return combinations;
}

void validate_spin_sector(
    int n_open_shell_electrons,
    int spin_multiplicity,
    int ms_twice) {
  if (n_open_shell_electrons < 0) {
    throw std::invalid_argument("n_open_shell_electrons must be non-negative");
  }
  if (spin_multiplicity <= 0 || spin_multiplicity > n_open_shell_electrons + 1) {
    throw std::invalid_argument("target spin multiplicity is outside the open-shell sector");
  }
  if (((n_open_shell_electrons - (spin_multiplicity - 1)) % 2) != 0) {
    throw std::invalid_argument(
        "target spin multiplicity is incompatible with the open-shell electron parity");
  }
  if (ms_twice < 0 || ms_twice > spin_multiplicity - 1) {
    throw std::invalid_argument("target 2*M_s is outside the pure-spin sector");
  }
  if (((n_open_shell_electrons - ms_twice) % 2) != 0) {
    throw std::invalid_argument(
        "target 2*M_s is incompatible with the open-shell electron parity");
  }
}

double target_s2_eigenvalue(int spin_multiplicity) {
  return 0.25 * static_cast<double>(spin_multiplicity * spin_multiplicity - 1);
}

bool differs_by_opposite_spin_swap(
    const std::vector<bool>& beta_mask_left,
    const std::vector<bool>& beta_mask_right) {
  int mismatch_count = 0;
  int left_beta_to_alpha = 0;
  int left_alpha_to_beta = 0;
  for (std::size_t index = 0; index < beta_mask_left.size(); ++index) {
    if (beta_mask_left[index] == beta_mask_right[index]) {
      continue;
    }
    ++mismatch_count;
    if (beta_mask_left[index]) {
      ++left_beta_to_alpha;
    } else {
      ++left_alpha_to_beta;
    }
    if (mismatch_count > 2) {
      return false;
    }
  }
  return mismatch_count == 2 &&
      left_beta_to_alpha == 1 &&
      left_alpha_to_beta == 1;
}

Matrix build_s2_matrix(
    int n_open_shell_electrons,
    int ms_twice,
    const std::vector<std::vector<int>>& beta_position_combinations) {
  const int dimension =
      static_cast<int>(beta_position_combinations.size());
  Matrix s2_matrix = Matrix::Zero(dimension, dimension);
  if (dimension == 0) {
    return s2_matrix;
  }

  const double ms = 0.5 * static_cast<double>(ms_twice);
  const double diagonal_value =
      ms * ms + 0.5 * static_cast<double>(n_open_shell_electrons);

  std::vector<std::vector<bool>> beta_masks;
  beta_masks.reserve(dimension);
  for (const auto& beta_positions : beta_position_combinations) {
    std::vector<bool> beta_mask(
        n_open_shell_electrons,
        false);
    for (const int beta_position : beta_positions) {
      beta_mask[beta_position] = true;
    }
    beta_masks.push_back(std::move(beta_mask));
  }

  for (int row = 0; row < dimension; ++row) {
    s2_matrix(row, row) = diagonal_value;
    for (int col = 0; col < row; ++col) {
      if (differs_by_opposite_spin_swap(
              beta_masks[row],
              beta_masks[col])) {
        s2_matrix(row, col) = 1.0;
        s2_matrix(col, row) = 1.0;
      }
    }
  }
  return s2_matrix;
}

}  // namespace

PfSpinCouplingBlock PfSpinCouplingBuilder::build(
    int n_open_shell_electrons,
    int spin_multiplicity,
    int ms_twice) const {
  validate_spin_sector(
      n_open_shell_electrons,
      spin_multiplicity,
      ms_twice);
  const int n_blocked_beta =
      (n_open_shell_electrons - ms_twice) / 2;
  const int n_blocked_alpha =
      n_open_shell_electrons - n_blocked_beta;

  PfSpinCouplingBlock block;
  block.n_open_shell_electrons = n_open_shell_electrons;
  block.n_blocked_alpha = n_blocked_alpha;
  block.n_blocked_beta = n_blocked_beta;
  block.spin_multiplicity = spin_multiplicity;
  block.ms_twice = ms_twice;

  const auto beta_position_combinations =
      enumerate_combinations(
          n_open_shell_electrons,
          n_blocked_beta);
  block.primitive_spin_strings.reserve(beta_position_combinations.size());
  for (const auto& beta_positions : beta_position_combinations) {
    block.primitive_spin_strings.push_back({beta_positions});
  }

  if (beta_position_combinations.empty()) {
    throw std::runtime_error("spin-string enumeration produced an empty basis");
  }

  const Matrix s2_matrix =
      build_s2_matrix(
          n_open_shell_electrons,
          ms_twice,
          beta_position_combinations);
  Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(s2_matrix);
  if (eigensolver.info() != Eigen::Success) {
    throw std::runtime_error("failed to diagonalize the open-shell S^2 matrix");
  }

  const double expected_eigenvalue =
      target_s2_eigenvalue(spin_multiplicity);
  const double tolerance =
      1.0e-8 * std::max(1.0, std::abs(expected_eigenvalue));
  std::vector<int> selected_columns;
  for (int column = 0; column < eigensolver.eigenvalues().size(); ++column) {
    if (std::abs(eigensolver.eigenvalues()[column] - expected_eigenvalue) <=
        tolerance) {
      selected_columns.push_back(column);
    }
  }
  if (selected_columns.empty()) {
    throw std::runtime_error(
        "S^2 diagonalization did not produce the requested pure-spin eigenspace");
  }

  block.primitive_to_adapted_coefficients =
      Matrix::Zero(
          static_cast<int>(beta_position_combinations.size()),
          static_cast<int>(selected_columns.size()));
  for (int adapted_index = 0;
       adapted_index < static_cast<int>(selected_columns.size());
       ++adapted_index) {
    block.primitive_to_adapted_coefficients.col(adapted_index) =
        eigensolver.eigenvectors().col(
            selected_columns[adapted_index]);
  }
  return block;
}

}  // namespace xmvb::pfaffian_vbscf
