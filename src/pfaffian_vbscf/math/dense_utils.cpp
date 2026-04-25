#include "pfaffian_vbscf/math/dense_utils.hpp"

#include <stdexcept>
#include <string>

namespace xmvb::pfaffian_vbscf {

namespace {

void validate_square_matrix(const char* label, const ConstMatrixRef& matrix) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument(std::string(label) + " must be square");
  }
}

void validate_even_dimension(const char* label, Eigen::Index dimension) {
  if ((dimension % 2) != 0) {
    throw std::invalid_argument(std::string(label) + " must have even dimension");
  }
}

}  // namespace

int packed_antisymmetric_size(int dimension) {
  if (dimension < 0) {
    throw std::invalid_argument("dimension must be non-negative");
  }
  return dimension * (dimension - 1) / 2;
}

Matrix decode_antisymmetric_matrix(
    const ScalarBuffer& packed_entries,
    int dimension) {
  if (dimension < 0) {
    throw std::invalid_argument("dimension must be non-negative");
  }
  if (static_cast<int>(packed_entries.size()) != packed_antisymmetric_size(dimension)) {
    throw std::invalid_argument("packed_entries size does not match the matrix dimension");
  }

  Matrix matrix = Matrix::Zero(dimension, dimension);
  std::size_t offset = 0;
  for (int col = 1; col < dimension; ++col) {
    for (int row = 0; row < col; ++row) {
      const double value = packed_entries[offset++];
      matrix(row, col) = value;
      matrix(col, row) = -value;
    }
  }
  return matrix;
}

Matrix decode_pf_state(const PfState& state) {
  if (state.n_spin_orbitals <= 0) {
    throw std::invalid_argument("state.n_spin_orbitals must be positive");
  }
  return decode_antisymmetric_matrix(state.packed_entries, state.n_spin_orbitals);
}

Matrix decode_pf_alpha_beta_block(const PfState& state) {
  if (state.n_active_orbitals <= 0 || state.n_spin_orbitals <= 0) {
    throw std::invalid_argument("PfState dimensions must be positive");
  }
  if (state.n_spin_orbitals != 2 * state.n_active_orbitals) {
    throw std::invalid_argument(
        "PfState spin-orbital dimension does not match twice the active-space dimension");
  }
  if (static_cast<int>(state.packed_entries.size()) !=
      packed_antisymmetric_size(state.n_spin_orbitals)) {
    throw std::invalid_argument(
        "PfState packed entries do not match the declared spin-orbital dimension");
  }

  const int n_active_orbitals = state.n_active_orbitals;
  const int n_spin_orbitals = state.n_spin_orbitals;
  Matrix alpha_beta_block = Matrix::Zero(n_active_orbitals, n_active_orbitals);
  std::size_t offset = 0;
  for (int col = 1; col < n_spin_orbitals; ++col) {
    for (int row = 0; row < col; ++row) {
      const double value = state.packed_entries[offset++];
      if (col >= n_active_orbitals && row < n_active_orbitals) {
        alpha_beta_block(row, col - n_active_orbitals) = value;
      }
    }
  }
  return alpha_beta_block;
}

Matrix build_spin_block_diagonal_metric(const ConstMatrixRef& spatial_matrix) {
  validate_square_matrix("spatial_matrix", spatial_matrix);

  const Eigen::Index n = spatial_matrix.rows();
  Matrix spin_metric = Matrix::Zero(2 * n, 2 * n);
  spin_metric.topLeftCorner(n, n) = spatial_matrix;
  spin_metric.bottomRightCorner(n, n) = spatial_matrix;
  return spin_metric;
}

Matrix build_kernel(
    const ConstMatrixRef& left,
    const ConstMatrixRef& sigma,
    const ConstMatrixRef& right) {
  validate_square_matrix("left", left);
  validate_square_matrix("sigma", sigma);
  validate_square_matrix("right", right);
  if (left.rows() != sigma.rows() || sigma.rows() != right.rows()) {
    throw std::invalid_argument("left/sigma/right dimensions must match");
  }

  return left * sigma * right * sigma.transpose();
}

KernelTraceData compute_kernel_trace_data(
    const ConstMatrixRef& kernel,
    int order) 
{
  // spatial kernal matrix :shape (n, n)
  validate_square_matrix("kernel", kernel);
  if (order < 0) {
    throw std::invalid_argument("order must be non-negative");
  }

  KernelTraceData data;
  // storige power kernal matrix
  data.kernel_powers.resize(order);
  data.traces.assign(order, 0.0);

  if (order == 0) {
    return data;
  }

  Matrix current_power = Matrix::Identity(kernel.rows(), kernel.cols());
  data.kernel_powers[0] = current_power;

  for (int power = 1; power <= order; ++power) {
    Matrix next_power = current_power * kernel;
    current_power.swap(next_power);
    data.traces[power - 1] = current_power.trace();
    if (power < order) {
      data.kernel_powers[power] = current_power;
    }
  }

  return data;
}

Matrix build_kernel_trace_adjoint(
    const std::vector<Matrix>& kernel_powers,
    const ScalarBuffer& trace_weights) {
  if (kernel_powers.size() < trace_weights.size()) {
    throw std::invalid_argument("kernel_powers does not cover all trace weights");
  }
  if (trace_weights.empty()) {
    return Matrix();
  }

  const Eigen::Index dimension = kernel_powers.front().rows();
  Matrix kernel_adjoint = Matrix::Zero(dimension, dimension);
  for (std::size_t power = 0; power < trace_weights.size(); ++power) {
    const Matrix& kernel_power = kernel_powers[power];
    validate_square_matrix("kernel_power", kernel_power);
    if (kernel_power.rows() != dimension) {
      throw std::invalid_argument("kernel_powers dimensions are inconsistent");
    }
    kernel_adjoint.noalias() +=
        trace_weights[power] * static_cast<double>(power + 1) * kernel_power.transpose();
  }
  return kernel_adjoint;
}

Matrix backpropagate_sigma(
    const ConstMatrixRef& left,
    const ConstMatrixRef& sigma,
    const ConstMatrixRef& right,
    const ConstMatrixRef& kernel_adjoint) {
  validate_square_matrix("left", left);
  validate_square_matrix("sigma", sigma);
  validate_square_matrix("right", right);
  validate_square_matrix("kernel_adjoint", kernel_adjoint);
  if (left.rows() != sigma.rows() || sigma.rows() != right.rows() ||
      right.rows() != kernel_adjoint.rows()) {
    throw std::invalid_argument("left/sigma/right/kernel_adjoint dimensions must match");
  }

  Matrix sigma_adjoint = left.transpose() * kernel_adjoint * sigma * right.transpose();
  // The second Sigma enters through K = L Sigma R Sigma^T, so this contribution
  // comes from Tr(K_bar^T L Sigma R dSigma^T) = Tr((K_bar^T L Sigma R)^T dSigma).
  sigma_adjoint.noalias() +=
      kernel_adjoint.transpose() * left * sigma * right;
  return sigma_adjoint;
}

Matrix collapse_spin_diagonal_blocks(const ConstMatrixRef& spin_orbital_matrix) {
  validate_square_matrix("spin_orbital_matrix", spin_orbital_matrix);
  validate_even_dimension("spin_orbital_matrix", spin_orbital_matrix.rows());

  const Eigen::Index n_active = spin_orbital_matrix.rows() / 2;
  return spin_orbital_matrix.topLeftCorner(n_active, n_active) +
      spin_orbital_matrix.bottomRightCorner(n_active, n_active);
}

}  // namespace xmvb::pfaffian_vbscf
