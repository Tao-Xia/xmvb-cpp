#include "pfaffian_vbscf/kernel/pf_closed_shell_ri_kernel.hpp"

#include <algorithm>
#include <stdexcept>

namespace xmvb::pfaffian_vbscf {

namespace {

using RowMajorMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

std::size_t packed_pair_count(int n_active_orbitals) {
  const std::size_t n = xmvb::to_size(n_active_orbitals);
  return n * (n + 1) / 2;
}

void validate_ri_inputs(
    const PfKernelCache& cache,
    int n_auxiliary_functions,
    const ScalarBuffer& ri_active_pair_factors) {
  if (n_auxiliary_functions < 0) {
    throw std::invalid_argument("n_auxiliary_functions must be non-negative");
  }
  const std::size_t n_pairs =
      packed_pair_count(cache.n_active_orbitals);
  const std::size_t expected_size =
      xmvb::to_size(n_auxiliary_functions) * n_pairs;
  if (ri_active_pair_factors.size() != expected_size) {
    throw std::invalid_argument(
        "ri_active_pair_factors size does not match [n_auxiliary_functions][packed_pair]");
  }
}

void validate_packed_two_electron_grad(
    int n_active_orbitals,
    const ScalarBuffer& packed_two_electron_grad) {
  const std::size_t n_pairs = packed_pair_count(n_active_orbitals);
  const std::size_t expected_size =
      n_pairs * (n_pairs + 1) / 2;
  if (packed_two_electron_grad.size() != expected_size) {
    throw std::invalid_argument(
        "packed_two_electron_grad size does not match the packed active-space Gram dimension");
  }
}

void unpack_pair_factor_row(
    const double* packed_row,
    int n_active_orbitals,
    Matrix* factor_matrix) {
  factor_matrix->resize(n_active_orbitals, n_active_orbitals);
  int packed_index = 0;
  for (int column = 0; column < n_active_orbitals; ++column) {
    for (int row = 0; row <= column; ++row) {
      const double value = packed_row[packed_index++];
      (*factor_matrix)(column, row) = value;
      (*factor_matrix)(row, column) = value;
    }
  }
}

double symmetric_factor_trace(
    const ConstMatrixRef& factor_matrix,
    const ConstMatrixRef& operand) {
  return factor_matrix.cwiseProduct(operand).sum();
}

double evaluate_same_spin_asym_value(
    const ConstMatrixRef& factor_matrix,
    const ConstMatrixRef& x,
    const ConstMatrixRef& c,
    const ConstMatrixRef& u) {
  const int n = static_cast<int>(factor_matrix.rows());
  double value = 0.0;
  for (int p = 0; p < n - 1; ++p) {
    for (int q = 0; q < n - 1; ++q) {
      const double x_qp = x(q, p);
      const double c_qp = c(q, p);
      const double f_qp = factor_matrix(q, p);
      for (int r = p + 1; r < n; ++r) {
        const double x_qr = x(q, r);
        const double c_qr = c(q, r);
        const double f_qr = factor_matrix(q, r);
        for (int s = q + 1; s < n; ++s) {
          const double interaction =
              f_qp * factor_matrix(s, r) -
              f_qr * factor_matrix(s, p);
          if (interaction == 0.0) {
            continue;
          }
          value +=
              2.0 * interaction *
              (x_qp * c(s, r) -
               x_qr * c(s, p) +
               c_qp * u(s, r) -
               c_qr * u(s, p));
        }
      }
    }
  }
  return value;
}

}  // namespace

double evaluate_closed_shell_two_electron_spatial_ri(
    const PfKernelCache& cache,
    int n_auxiliary_functions,
    const ScalarBuffer& ri_active_pair_factors) {
  if (cache.trace_order <= 0) {
    return 0.0;
  }

  validate_ri_inputs(cache, n_auxiliary_functions, ri_active_pair_factors);
  const int n = cache.n_active_orbitals;
  if (n <= 0 || n_auxiliary_functions == 0) {
    return 0.0;
  }

  const std::size_t n_pairs = packed_pair_count(n);
  const Eigen::Map<const RowMajorMatrix> ri_pair_factor_matrix(
      ri_active_pair_factors.data(),
      n_auxiliary_functions,
      static_cast<int>(n_pairs));

  const Matrix& b = cache.closed_shell_right_ab_block;
  const Matrix& c = cache.closed_shell_pair_core;
  const Matrix& d = cache.closed_shell_d;
  const Matrix& pair_term_matrix = cache.closed_shell_pair_term_matrix;
  const Matrix& c_poly_h_n2 = cache.closed_shell_c_poly_h_n2;
  const Matrix& frechet_h_c_h_n2 = cache.closed_shell_frechet_h_c_h_n2;
  const Matrix& d_poly_h_n2 = cache.closed_shell_d_poly_h_n2;
  const Matrix& opposite_bridge_h_split_matrix =
      cache.closed_shell_opposite_bridge_h_split_matrix;
  const Matrix& a = cache.closed_shell_left_ba_block;

  Matrix factor_matrix;
  Matrix left_workspace = Matrix::Zero(n, n);
  Matrix right_workspace = Matrix::Zero(n, n);
  double value = 0.0;
  for (int auxiliary_index = 0;
       auxiliary_index < n_auxiliary_functions;
       ++auxiliary_index) {
    const double* packed_row =
        ri_pair_factor_matrix.data() +
        xmvb::to_size(auxiliary_index) * n_pairs;
    unpack_pair_factor_row(packed_row, n, &factor_matrix);

    left_workspace.noalias() = factor_matrix * pair_term_matrix.transpose();
    right_workspace.noalias() = left_workspace * factor_matrix;
    if (cache.closed_shell_coeff_n2.empty()) {
      value += 2.0 * b.cwiseProduct(right_workspace).sum();
      continue;
    }

    value += 2.0 * b.cwiseProduct(right_workspace).sum();

    left_workspace.noalias() = factor_matrix * a.transpose();
    right_workspace.noalias() = left_workspace * factor_matrix;
    value -= d_poly_h_n2.cwiseProduct(right_workspace).sum();

    left_workspace.noalias() = factor_matrix * opposite_bridge_h_split_matrix.transpose();
    right_workspace.noalias() = left_workspace * factor_matrix;
    value -= d.cwiseProduct(right_workspace).sum();

    const double x_trace =
        symmetric_factor_trace(factor_matrix, c_poly_h_n2);
    const double c_trace =
        symmetric_factor_trace(factor_matrix, c);
    const double u_trace =
        symmetric_factor_trace(factor_matrix, frechet_h_c_h_n2);
    value += x_trace * c_trace + c_trace * u_trace;
    value +=
        evaluate_same_spin_asym_value(
            factor_matrix,
            c_poly_h_n2,
            c,
            frechet_h_c_h_n2);
  }
  return value;
}

void backpropagate_closed_shell_ri_factor_gram(
    int n_active_orbitals,
    int n_auxiliary_functions,
    const ScalarBuffer& ri_active_pair_factors,
    const ScalarBuffer& packed_two_electron_grad,
    ScalarBuffer* ri_active_pair_factor_grad) {
  if (n_active_orbitals < 0) {
    throw std::invalid_argument("n_active_orbitals must be non-negative");
  }
  if (n_auxiliary_functions < 0) {
    throw std::invalid_argument("n_auxiliary_functions must be non-negative");
  }
  if (ri_active_pair_factor_grad == nullptr) {
    throw std::invalid_argument("ri_active_pair_factor_grad must not be null");
  }

  const std::size_t n_pairs = packed_pair_count(n_active_orbitals);
  const std::size_t expected_factor_size =
      xmvb::to_size(n_auxiliary_functions) * n_pairs;
  if (ri_active_pair_factors.size() != expected_factor_size) {
    throw std::invalid_argument(
        "ri_active_pair_factors size does not match [n_auxiliary_functions][packed_pair]");
  }
  validate_packed_two_electron_grad(
      n_active_orbitals,
      packed_two_electron_grad);

  if (ri_active_pair_factor_grad->size() != expected_factor_size) {
    ri_active_pair_factor_grad->assign(expected_factor_size, 0.0);
  } else {
    std::fill(
        ri_active_pair_factor_grad->begin(),
        ri_active_pair_factor_grad->end(),
        0.0);
  }
  if (n_pairs == 0 || n_auxiliary_functions == 0) {
    return;
  }

  Matrix symmetric_gram_grad =
      Matrix::Zero(static_cast<int>(n_pairs), static_cast<int>(n_pairs));
  std::size_t packed_index = 0;
  for (std::size_t pair_col = 0; pair_col < n_pairs; ++pair_col) {
    for (std::size_t pair_row = 0; pair_row <= pair_col; ++pair_row) {
      const double gradient = packed_two_electron_grad[packed_index++];
      if (pair_row == pair_col) {
        symmetric_gram_grad(
            static_cast<Eigen::Index>(pair_row),
            static_cast<Eigen::Index>(pair_col)) = 2.0 * gradient;
      } else {
        symmetric_gram_grad(
            static_cast<Eigen::Index>(pair_row),
            static_cast<Eigen::Index>(pair_col)) = gradient;
        symmetric_gram_grad(
            static_cast<Eigen::Index>(pair_col),
            static_cast<Eigen::Index>(pair_row)) = gradient;
      }
    }
  }

  const Eigen::Map<const RowMajorMatrix> factor_matrix(
      ri_active_pair_factors.data(),
      n_auxiliary_functions,
      static_cast<int>(n_pairs));
  Eigen::Map<RowMajorMatrix> factor_grad_matrix(
      ri_active_pair_factor_grad->data(),
      n_auxiliary_functions,
      static_cast<int>(n_pairs));
  factor_grad_matrix.noalias() = factor_matrix * symmetric_gram_grad;
}

}  // namespace xmvb::pfaffian_vbscf
