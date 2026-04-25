#include "pfaffian_vbscf/tensor/pf_tensor_contractor.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "pfaffian_vbscf/tensor/pf_packed_index_cache.hpp"

namespace xmvb::pfaffian_vbscf {

namespace {

std::size_t checked_packed_pair_count(int n_active_orbitals) {
  if (n_active_orbitals < 0) {
    throw std::invalid_argument("n_active_orbitals must be non-negative");
  }
  const std::size_t n = n_active_orbitals;
  return n * (n + 1) / 2;
}

std::size_t checked_packed_two_electron_count(int n_active_orbitals) {
  const std::size_t n_pairs = checked_packed_pair_count(n_active_orbitals);
  return n_pairs * (n_pairs + 1) / 2;
}

void validate_square_matrix(const char* label, const ConstMatrixRef& matrix) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument(std::string(label) + " must be square");
  }
}

bool is_effectively_zero(double value) {
  return std::abs(value) < 1.0e-15;
}

int validate_contract_inputs(
    const ScalarBuffer& ggo,
    int n_active_orbitals,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right) {
  validate_square_matrix("left", left);
  validate_square_matrix("right", right);
  if (left.rows() != right.rows()) {
    throw std::invalid_argument("left/right matrix size mismatch");
  }
  if (left.rows() != n_active_orbitals) {
    throw std::invalid_argument("matrix dimensions do not match n_active_orbitals");
  }

  const std::size_t expected_size = checked_packed_two_electron_count(n_active_orbitals);
  if (ggo.size() != expected_size) {
    throw std::invalid_argument("ggo size does not match packed active-space tensor size");
  }
  return n_active_orbitals;
}

int validate_outer_product_inputs(
    const ConstMatrixRef& left,
    const ConstMatrixRef& right) {
  validate_square_matrix("left", left);
  validate_square_matrix("right", right);
  if (left.rows() != right.rows()) {
    throw std::invalid_argument("left/right matrix size mismatch");
  }
  return static_cast<int>(left.rows());
}

int validate_operand_adjoint_inputs(
    double weight,
    const ScalarBuffer& ggo,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right,
    MatrixRef left_adjoint,
    MatrixRef right_adjoint) {
  if (is_effectively_zero(weight)) {
    return 0;
  }

  const int n = validate_contract_inputs(ggo, static_cast<int>(left.rows()), left, right);
  validate_square_matrix("left_adjoint", left_adjoint);
  validate_square_matrix("right_adjoint", right_adjoint);
  if (left_adjoint.rows() != n || right_adjoint.rows() != n) {
    throw std::invalid_argument("operand adjoint dimensions do not match the operands");
  }
  return n;
}

void validate_matching_square_matrix(
    const char* label,
    const ConstMatrixRef& matrix,
    int n_active_orbitals) {
  validate_square_matrix(label, matrix);
  if (matrix.rows() != n_active_orbitals) {
    throw std::invalid_argument(
        std::string(label) + " dimensions do not match n_active_orbitals");
  }
}

double* prepare_gradient_buffer(ScalarBuffer* ggo_grad, int n_active_orbitals) {
  if (ggo_grad == nullptr) {
    throw std::invalid_argument("ggo_grad must not be null");
  }

  const std::size_t expected_size = checked_packed_two_electron_count(n_active_orbitals);
  if (ggo_grad->empty()) {
    ggo_grad->assign(expected_size, 0.0);
  } else if (ggo_grad->size() != expected_size) {
    throw std::invalid_argument("ggo_grad size does not match packed active-space tensor size");
  }
  return ggo_grad->data();
}

}  // namespace

std::size_t PfTensorContractor::packed_pair_count(int n_active_orbitals) {
  return checked_packed_pair_count(n_active_orbitals);
}

std::size_t PfTensorContractor::packed_two_electron_count(int n_active_orbitals) {
  return checked_packed_two_electron_count(n_active_orbitals);
}

double PfTensorContractor::contract_exchange(
    const ScalarBuffer& ggo,
    int n_active_orbitals,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right) {
  const int n = validate_contract_inputs(ggo, n_active_orbitals, left, right);
  const double* ggo_data = ggo.data();
  const double* left_data = left.data();
  const double* right_data = right.data();
  const Eigen::Index left_stride = left.outerStride();
  const Eigen::Index right_stride = right.outerStride();
  const PfPackedIndexCache& index_cache = get_packed_index_cache(n);

  double value = 0.0;
  for (int q = 0; q < n; ++q) {
    const double* right_col_q = right_data + static_cast<Eigen::Index>(q) * right_stride;
    for (int r = 0; r < n; ++r) {
      const double* left_col_r = left_data + static_cast<Eigen::Index>(r) * left_stride;
      for (int s = 0; s < n; ++s) {
        const double right_sq = right_col_q[s];
        if (std::abs(right_sq) < 1.0e-15) {
          continue;
        }
        for (int p = 0; p < n; ++p) {
          value += ggo_data[quartet_index(index_cache, q, p, s, r)] *
              left_col_r[p] * right_sq;
        }
      }
    }
  }
  return value;
}

double PfTensorContractor::contract_direct(
    const ScalarBuffer& ggo,
    int n_active_orbitals,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right) {
  const int n = validate_contract_inputs(ggo, n_active_orbitals, left, right);
  const double* ggo_data = ggo.data();
  const double* left_data = left.data();
  const double* right_data = right.data();
  const Eigen::Index left_stride = left.outerStride();
  const Eigen::Index right_stride = right.outerStride();
  const PfPackedIndexCache& index_cache = get_packed_index_cache(n);

  double value = 0.0;
  for (int q = 0; q < n; ++q) {
    for (int p = 0; p < n; ++p) {
      const double left_qp =
          left_data[q + static_cast<Eigen::Index>(p) * left_stride];
      if (is_effectively_zero(left_qp)) {
        continue;
      }
      for (int s = 0; s < n; ++s) {
        for (int r = 0; r < n; ++r) {
          const double right_sr =
              right_data[s + static_cast<Eigen::Index>(r) * right_stride];
          if (is_effectively_zero(right_sr)) {
            continue;
          }
          value += ggo_data[quartet_index(index_cache, q, p, s, r)] *
              left_qp * right_sr;
        }
      }
    }
  }
  return value;
}

double PfTensorContractor::contract_coulomb(
    const ScalarBuffer& ggo,
    int n_active_orbitals,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right) {
  const int n = validate_contract_inputs(ggo, n_active_orbitals, left, right);
  const double* ggo_data = ggo.data();
  const double* left_data = left.data();
  const double* right_data = right.data();
  const Eigen::Index left_stride = left.outerStride();
  const Eigen::Index right_stride = right.outerStride();
  const PfPackedIndexCache& index_cache = get_packed_index_cache(n);

  double value = 0.0;
  for (int q = 0; q < n; ++q) {
    for (int p = 0; p < n; ++p) {
      const double* left_col_p = left_data + static_cast<Eigen::Index>(p) * left_stride;
      for (int s = 0; s < n; ++s) {
        const double right_qs = right_data[q + static_cast<Eigen::Index>(s) * right_stride];
        if (std::abs(right_qs) < 1.0e-15) {
          continue;
        }
        for (int r = 0; r < n; ++r) {
          value += ggo_data[quartet_index(index_cache, q, p, s, r)] *
              left_col_p[r] * right_qs;
        }
      }
    }
  }
  return value;
}

void PfTensorContractor::compute_exchange_operand_adjoints(
    double weight,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right,
    const ScalarBuffer& ggo,
    MatrixRef left_adjoint,
    MatrixRef right_adjoint) {
  const int n = validate_operand_adjoint_inputs(
      weight, ggo, left, right, left_adjoint, right_adjoint);
  if (n == 0) {
    return;
  }

  const double* ggo_data = ggo.data();
  const double* left_data = left.data();
  const double* right_data = right.data();
  double* left_bar_data = left_adjoint.data();
  double* right_bar_data = right_adjoint.data();
  const Eigen::Index left_stride = left.outerStride();
  const Eigen::Index right_stride = right.outerStride();
  const Eigen::Index left_bar_stride = left_adjoint.outerStride();
  const Eigen::Index right_bar_stride = right_adjoint.outerStride();
  const PfPackedIndexCache& index_cache = get_packed_index_cache(n);

  for (int q = 0; q < n; ++q) {
    const double* right_col_q = right_data + static_cast<Eigen::Index>(q) * right_stride;
    double* right_bar_col_q = right_bar_data + static_cast<Eigen::Index>(q) * right_bar_stride;
    for (int r = 0; r < n; ++r) {
      const double* left_col_r = left_data + static_cast<Eigen::Index>(r) * left_stride;
      double* left_bar_col_r = left_bar_data + static_cast<Eigen::Index>(r) * left_bar_stride;
      for (int s = 0; s < n; ++s) {
        const int idx_sr = packed_pair_index(s, r);
        const double right_sq = right_col_q[s];
        const bool update_left = !is_effectively_zero(right_sq);
        const double w_right_sq = weight * right_sq;
        double right_bar_sq = 0.0;
        for (int p = 0; p < n; ++p) {
          const int idx_qp = packed_pair_index(q, p);
          const double g = ggo_data[pair_of_pairs_index(index_cache, idx_qp, idx_sr)];
          if (is_effectively_zero(g)) {
            continue;
          }
          const double weighted_g = weight * g;
          if (update_left) {
            left_bar_col_r[p] += g * w_right_sq;
          }
          const double left_pr = left_col_r[p];
          if (!is_effectively_zero(left_pr)) {
            right_bar_sq += weighted_g * left_pr;
          }
        }
        right_bar_col_q[s] += right_bar_sq;
      }
    }
  }
}

void PfTensorContractor::compute_coulomb_operand_adjoints(
    double weight,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right,
    const ScalarBuffer& ggo,
    MatrixRef left_adjoint,
    MatrixRef right_adjoint) {
  const int n = validate_operand_adjoint_inputs(
      weight, ggo, left, right, left_adjoint, right_adjoint);
  if (n == 0) {
    return;
  }

  const double* ggo_data = ggo.data();
  const double* left_data = left.data();
  const double* right_data = right.data();
  double* left_bar_data = left_adjoint.data();
  double* right_bar_data = right_adjoint.data();
  const Eigen::Index left_stride = left.outerStride();
  const Eigen::Index right_stride = right.outerStride();
  const Eigen::Index left_bar_stride = left_adjoint.outerStride();
  const Eigen::Index right_bar_stride = right_adjoint.outerStride();
  const PfPackedIndexCache& index_cache = get_packed_index_cache(n);

  for (int q = 0; q < n; ++q) {
    for (int p = 0; p < n; ++p) {
      const int idx_qp = packed_pair_index(q, p);
      const double* left_col_p = left_data + static_cast<Eigen::Index>(p) * left_stride;
      double* left_bar_col_p = left_bar_data + static_cast<Eigen::Index>(p) * left_bar_stride;
      for (int s = 0; s < n; ++s) {
        const double right_qs = right_data[q + static_cast<Eigen::Index>(s) * right_stride];
        const bool update_left = !is_effectively_zero(right_qs);
        const double w_right_qs = weight * right_qs;
        double* right_bar_col_s = right_bar_data + static_cast<Eigen::Index>(s) * right_bar_stride;
        double right_bar_qs = 0.0;
        for (int r = 0; r < n; ++r) {
          const int idx_sr = packed_pair_index(s, r);
          const double g = ggo_data[pair_of_pairs_index(index_cache, idx_qp, idx_sr)];
          if (is_effectively_zero(g)) {
            continue;
          }
          const double weighted_g = weight * g;
          if (update_left) {
            left_bar_col_p[r] += g * w_right_qs;
          }
          const double left_rp = left_col_p[r];
          if (!is_effectively_zero(left_rp)) {
            right_bar_qs += weighted_g * left_rp;
          }
        }
        right_bar_col_s[q] += right_bar_qs;
      }
    }
  }
}

void PfTensorContractor::compute_direct_operand_adjoints(
    double weight,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right,
    const ScalarBuffer& ggo,
    MatrixRef left_adjoint,
    MatrixRef right_adjoint) {
  const int n = validate_operand_adjoint_inputs(
      weight, ggo, left, right, left_adjoint, right_adjoint);
  if (n == 0) {
    return;
  }

  const double* ggo_data = ggo.data();
  const double* left_data = left.data();
  const double* right_data = right.data();
  double* left_bar_data = left_adjoint.data();
  double* right_bar_data = right_adjoint.data();
  const Eigen::Index left_stride = left.outerStride();
  const Eigen::Index right_stride = right.outerStride();
  const Eigen::Index left_bar_stride = left_adjoint.outerStride();
  const Eigen::Index right_bar_stride = right_adjoint.outerStride();
  const PfPackedIndexCache& index_cache = get_packed_index_cache(n);

  for (int q = 0; q < n; ++q) {
    for (int p = 0; p < n; ++p) {
      const int idx_qp = packed_pair_index(q, p);
      const double left_qp =
          left_data[q + static_cast<Eigen::Index>(p) * left_stride];
      double left_bar_qp = 0.0;
      for (int s = 0; s < n; ++s) {
        for (int r = 0; r < n; ++r) {
          const int idx_sr = packed_pair_index(s, r);
          const double g = ggo_data[pair_of_pairs_index(index_cache, idx_qp, idx_sr)];
          if (is_effectively_zero(g)) {
            continue;
          }
          const double weighted_g = weight * g;
          const double right_sr =
              right_data[s + static_cast<Eigen::Index>(r) * right_stride];
          if (!is_effectively_zero(right_sr)) {
            left_bar_qp += weighted_g * right_sr;
          }
          if (!is_effectively_zero(left_qp)) {
            right_bar_data[s + static_cast<Eigen::Index>(r) * right_bar_stride] +=
                weighted_g * left_qp;
          }
        }
      }
      left_bar_data[q + static_cast<Eigen::Index>(p) * left_bar_stride] += left_bar_qp;
    }
  }
}

double PfTensorContractor::contract_same_spin_exchange(
    const ScalarBuffer& ggo,
    int n_active_orbitals,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right) {
  const int n = validate_contract_inputs(ggo, n_active_orbitals, left, right);
  const double* ggo_data = ggo.data();
  const double* left_data = left.data();
  const double* right_data = right.data();
  const Eigen::Index left_stride = left.outerStride();
  const Eigen::Index right_stride = right.outerStride();
  const PfPackedIndexCache& index_cache = get_packed_index_cache(n);

  double value = 0.0;
  for (int q = 0; q < n; ++q) {
    const double* right_col_q = right_data + static_cast<Eigen::Index>(q) * right_stride;
    for (int r = 0; r < n; ++r) {
      const double* left_col_r = left_data + static_cast<Eigen::Index>(r) * left_stride;
      for (int s = 0; s < n; ++s) {
        const int idx_sr = packed_pair_index(s, r);
        const double right_sq = right_col_q[s];
        if (std::abs(right_sq) < 1.0e-15) {
          continue;
        }
        for (int p = 0; p < n; ++p) {
          const int idx_qp = packed_pair_index(q, p);
          value += ggo_data[pair_of_pairs_index(index_cache, idx_qp, idx_sr)] *
              (left_col_r[p] - left_data[r + static_cast<Eigen::Index>(p) * left_stride]) *
              right_sq;
        }
      }
    }
  }
  return value;
}

void PfTensorContractor::compute_same_spin_exchange_operand_adjoints(
    double weight,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right,
    const ScalarBuffer& ggo,
    MatrixRef left_adjoint,
    MatrixRef right_adjoint) {
  const int n = validate_operand_adjoint_inputs(
      weight, ggo, left, right, left_adjoint, right_adjoint);
  if (n == 0) {
    return;
  }

  const double* ggo_data = ggo.data();
  const double* left_data = left.data();
  const double* right_data = right.data();
  double* left_bar_data = left_adjoint.data();
  double* right_bar_data = right_adjoint.data();
  const Eigen::Index left_stride = left.outerStride();
  const Eigen::Index right_stride = right.outerStride();
  const Eigen::Index left_bar_stride = left_adjoint.outerStride();
  const Eigen::Index right_bar_stride = right_adjoint.outerStride();
  const PfPackedIndexCache& index_cache = get_packed_index_cache(n);

  for (int q = 0; q < n; ++q) {
    const double* right_col_q = right_data + static_cast<Eigen::Index>(q) * right_stride;
    double* right_bar_col_q = right_bar_data + static_cast<Eigen::Index>(q) * right_bar_stride;
    for (int r = 0; r < n; ++r) {
      const double* left_col_r = left_data + static_cast<Eigen::Index>(r) * left_stride;
      double* left_bar_col_r = left_bar_data + static_cast<Eigen::Index>(r) * left_bar_stride;
      for (int s = 0; s < n; ++s) {
        const int idx_sr = packed_pair_index(s, r);
        const double right_sq = right_col_q[s];
        const bool update_left = !is_effectively_zero(right_sq);
        const double w_right_sq = weight * right_sq;
        double right_bar_sq = 0.0;
        for (int p = 0; p < n; ++p) {
          const int idx_qp = packed_pair_index(q, p);
          const double g = ggo_data[pair_of_pairs_index(index_cache, idx_qp, idx_sr)];
          if (is_effectively_zero(g)) {
            continue;
          }
          const double left_pr = left_col_r[p];
          const double left_rp = left_data[r + static_cast<Eigen::Index>(p) * left_stride];
          if (update_left) {
            const double delta = g * w_right_sq;
            left_bar_col_r[p] += delta;
            left_bar_data[r + static_cast<Eigen::Index>(p) * left_bar_stride] -= delta;
          }
          const double left_delta = left_pr - left_rp;
          if (!is_effectively_zero(left_delta)) {
            right_bar_sq += weight * g * left_delta;
          }
        }
        right_bar_col_q[s] += right_bar_sq;
      }
    }
  }
}

double PfTensorContractor::contract_same_spin_coulomb(
    const ScalarBuffer& ggo,
    int n_active_orbitals,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right) {
  const int n = validate_contract_inputs(ggo, n_active_orbitals, left, right);
  const double* ggo_data = ggo.data();
  const double* left_data = left.data();
  const double* right_data = right.data();
  const Eigen::Index left_stride = left.outerStride();
  const Eigen::Index right_stride = right.outerStride();
  const PfPackedIndexCache& index_cache = get_packed_index_cache(n);

  double value = 0.0;
  for (int q = 0; q < n; ++q) {
    for (int p = 0; p < n; ++p) {
      const int idx_qp = packed_pair_index(q, p);
      const double* left_col_p = left_data + static_cast<Eigen::Index>(p) * left_stride;
      for (int s = 0; s < n; ++s) {
        const double right_qs = right_data[q + static_cast<Eigen::Index>(s) * right_stride];
        if (std::abs(right_qs) < 1.0e-15) {
          continue;
        }
        for (int r = 0; r < n; ++r) {
          const int idx_sr = packed_pair_index(s, r);
          value += ggo_data[pair_of_pairs_index(index_cache, idx_qp, idx_sr)] *
              (left_col_p[r] - left_data[p + static_cast<Eigen::Index>(r) * left_stride]) *
              right_qs;
        }
      }
    }
  }
  return value;
}

double PfTensorContractor::contract_same_spin_separable(
    const ScalarBuffer& ggo,
    int n_active_orbitals,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right) {
  const int n = validate_contract_inputs(ggo, n_active_orbitals, left, right);
  const double* ggo_data = ggo.data();
  const double* left_data = left.data();
  const double* right_data = right.data();
  const Eigen::Index left_stride = left.outerStride();
  const Eigen::Index right_stride = right.outerStride();
  const PfPackedIndexCache& index_cache = get_packed_index_cache(n);

  double value = 0.0;
  for (int left_first = 0; left_first < n - 1; ++left_first) {
    for (int right_first = 0; right_first < n - 1; ++right_first) {
      const double left_entry =
          left_data[right_first + static_cast<Eigen::Index>(left_first) * left_stride];
      if (is_effectively_zero(left_entry)) {
        continue;
      }
      for (int left_second = left_first + 1; left_second < n; ++left_second) {
        for (int right_second = right_first + 1; right_second < n; ++right_second) {
          const double right_entry =
              right_data[right_second + static_cast<Eigen::Index>(left_second) * right_stride];
          if (is_effectively_zero(right_entry)) {
            continue;
          }
          const double interaction =
              ggo_data[quartet_index(
                  index_cache,
                  right_first,
                  left_first,
                  right_second,
                  left_second)] -
              ggo_data[quartet_index(
                  index_cache,
                  right_first,
                  left_second,
                  right_second,
                  left_first)];
          value += interaction * left_entry * right_entry;
        }
      }
    }
  }
  return value;
}

double PfTensorContractor::contract_same_spin_bridge(
    const ScalarBuffer& ggo,
    int n_active_orbitals,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right) {
  const int n = validate_contract_inputs(ggo, n_active_orbitals, left, right);
  const double* ggo_data = ggo.data();
  const double* left_data = left.data();
  const double* right_data = right.data();
  const Eigen::Index left_stride = left.outerStride();
  const Eigen::Index right_stride = right.outerStride();
  const PfPackedIndexCache& index_cache = get_packed_index_cache(n);

  double value = 0.0;
  for (int left_first = 0; left_first < n - 1; ++left_first) {
    for (int right_first = 0; right_first < n - 1; ++right_first) {
      for (int left_second = left_first + 1; left_second < n; ++left_second) {
        const double left_entry =
            left_data[right_first + static_cast<Eigen::Index>(left_second) * left_stride];
        if (is_effectively_zero(left_entry)) {
          continue;
        }
        const double weighted_left = left_entry;
        for (int right_second = right_first + 1; right_second < n; ++right_second) {
          const double right_entry =
              right_data[right_second + static_cast<Eigen::Index>(left_first) * right_stride];
          if (is_effectively_zero(right_entry)) {
            continue;
          }
          const double interaction =
              ggo_data[quartet_index(
                  index_cache,
                  right_first,
                  left_first,
                  right_second,
                  left_second)] -
              ggo_data[quartet_index(
                  index_cache,
                  right_first,
                  left_second,
                  right_second,
                  left_first)];
          value += interaction * weighted_left * right_entry;
        }
      }
    }
  }
  return value;
}

double PfTensorContractor::contract_closed_shell_full_linear_combo(
    const ScalarBuffer& ggo,
    int n_active_orbitals,
    const ConstMatrixRef& b,
    const ConstMatrixRef& m_pair,
    const ConstMatrixRef& d_r,
    const ConstMatrixRef& a,
    const ConstMatrixRef& d,
    const ConstMatrixRef& v,
    const ConstMatrixRef& x,
    const ConstMatrixRef& c,
    const ConstMatrixRef& u) {
  const int n = validate_contract_inputs(ggo, n_active_orbitals, b, m_pair);
  validate_matching_square_matrix("d_r", d_r, n);
  validate_matching_square_matrix("a", a, n);
  validate_matching_square_matrix("d", d, n);
  validate_matching_square_matrix("v", v, n);
  validate_matching_square_matrix("x", x, n);
  validate_matching_square_matrix("c", c, n);
  validate_matching_square_matrix("u", u, n);

  const double* ggo_data = ggo.data();
  const double* b_data = b.data();
  const double* m_pair_data = m_pair.data();
  const double* d_r_data = d_r.data();
  const double* a_data = a.data();
  const double* d_data = d.data();
  const double* v_data = v.data();
  const double* x_data = x.data();
  const double* c_data = c.data();
  const double* u_data = u.data();
  const Eigen::Index b_stride = b.outerStride();
  const Eigen::Index m_pair_stride = m_pair.outerStride();
  const Eigen::Index d_r_stride = d_r.outerStride();
  const Eigen::Index a_stride = a.outerStride();
  const Eigen::Index d_stride = d.outerStride();
  const Eigen::Index v_stride = v.outerStride();
  const Eigen::Index x_stride = x.outerStride();
  const Eigen::Index c_stride = c.outerStride();
  const Eigen::Index u_stride = u.outerStride();
  const PfPackedIndexCache& index_cache = get_packed_index_cache(n);

  double value = 0.0;
  for (int q = 0; q < n; ++q) {
    const double* m_pair_col_q =
        m_pair_data + static_cast<Eigen::Index>(q) * m_pair_stride;
    const double* a_col_q =
        a_data + static_cast<Eigen::Index>(q) * a_stride;
    const double* v_col_q =
        v_data + static_cast<Eigen::Index>(q) * v_stride;
    for (int s = 0; s < n; ++s) {
      const double m_sq = m_pair_col_q[s];
      const double m_qs =
          m_pair_data[q + static_cast<Eigen::Index>(s) * m_pair_stride];
      const double a_sq = a_col_q[s];
      const double a_qs =
          a_data[q + static_cast<Eigen::Index>(s) * a_stride];
      const double v_sq = v_col_q[s];
      const double v_qs =
          v_data[q + static_cast<Eigen::Index>(s) * v_stride];
      for (int p = 0; p < n; ++p) {
        const double x_qp =
            x_data[q + static_cast<Eigen::Index>(p) * x_stride];
        const double c_qp =
            c_data[q + static_cast<Eigen::Index>(p) * c_stride];
        const double* b_col_p =
            b_data + static_cast<Eigen::Index>(p) * b_stride;
        const double* d_r_col_p =
            d_r_data + static_cast<Eigen::Index>(p) * d_r_stride;
        const double* d_col_p =
            d_data + static_cast<Eigen::Index>(p) * d_stride;
        for (int r = 0; r < n; ++r) {
          const double coefficient =
              b_data[p + static_cast<Eigen::Index>(r) * b_stride] * m_sq +
              b_col_p[r] * m_qs -
              0.5 *
                  (d_r_data[p + static_cast<Eigen::Index>(r) * d_r_stride] * a_sq +
                   d_r_col_p[r] * a_qs +
                   d_data[p + static_cast<Eigen::Index>(r) * d_stride] * v_sq +
                   d_col_p[r] * v_qs) +
              x_qp * c_data[s + static_cast<Eigen::Index>(r) * c_stride] +
              c_qp * u_data[s + static_cast<Eigen::Index>(r) * u_stride];
          if (is_effectively_zero(coefficient)) {
            continue;
          }
          value += ggo_data[quartet_index(index_cache, q, p, s, r)] * coefficient;
        }
      }
    }
  }
  return value;
}

double PfTensorContractor::contract_closed_shell_same_spin_asym_linear_combo(
    const ScalarBuffer& ggo,
    int n_active_orbitals,
    const ConstMatrixRef& x,
    const ConstMatrixRef& c,
    const ConstMatrixRef& u) {
  const int n = validate_contract_inputs(ggo, n_active_orbitals, x, c);
  validate_matching_square_matrix("u", u, n);

  const double* ggo_data = ggo.data();
  const double* x_data = x.data();
  const double* c_data = c.data();
  const double* u_data = u.data();
  const Eigen::Index x_stride = x.outerStride();
  const Eigen::Index c_stride = c.outerStride();
  const Eigen::Index u_stride = u.outerStride();
  const PfPackedIndexCache& index_cache = get_packed_index_cache(n);

  double value = 0.0;
  for (int p = 0; p < n - 1; ++p) {
    const double* c_col_p =
        c_data + static_cast<Eigen::Index>(p) * c_stride;
    const double* u_col_p =
        u_data + static_cast<Eigen::Index>(p) * u_stride;
    for (int q = 0; q < n - 1; ++q) {
      const double x_qp =
          x_data[q + static_cast<Eigen::Index>(p) * x_stride];
      const double c_qp =
          c_data[q + static_cast<Eigen::Index>(p) * c_stride];
      const int direct_qp = packed_pair_index(q, p);
      for (int r = p + 1; r < n; ++r) {
        const double x_qr =
            x_data[q + static_cast<Eigen::Index>(r) * x_stride];
        const double c_qr =
            c_data[q + static_cast<Eigen::Index>(r) * c_stride];
        if (is_effectively_zero(x_qp) &&
            is_effectively_zero(c_qp) &&
            is_effectively_zero(x_qr) &&
            is_effectively_zero(c_qr)) {
          continue;
        }
        const int exchange_qr = packed_pair_index(q, r);
        for (int s = q + 1; s < n; ++s) {
          const double coefficient =
              2.0 *
              (x_qp * c_data[s + static_cast<Eigen::Index>(r) * c_stride] -
               x_qr * c_col_p[s] +
               c_qp * u_data[s + static_cast<Eigen::Index>(r) * u_stride] -
               c_qr * u_col_p[s]);
          if (is_effectively_zero(coefficient)) {
            continue;
          }
          const int direct_sr = packed_pair_index(s, r);
          const int exchange_sp = packed_pair_index(s, p);
          const double interaction =
              ggo_data[pair_of_pairs_index(index_cache, direct_qp, direct_sr)] -
              ggo_data[pair_of_pairs_index(index_cache, exchange_qr, exchange_sp)];
          if (is_effectively_zero(interaction)) {
            continue;
          }
          value += interaction * coefficient;
        }
      }
    }
  }
  return value;
}

void PfTensorContractor::compute_same_spin_coulomb_operand_adjoints(
    double weight,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right,
    const ScalarBuffer& ggo,
    MatrixRef left_adjoint,
    MatrixRef right_adjoint) {
  const int n = validate_operand_adjoint_inputs(
      weight, ggo, left, right, left_adjoint, right_adjoint);
  if (n == 0) {
    return;
  }

  const double* ggo_data = ggo.data();
  const double* left_data = left.data();
  const double* right_data = right.data();
  double* left_bar_data = left_adjoint.data();
  double* right_bar_data = right_adjoint.data();
  const Eigen::Index left_stride = left.outerStride();
  const Eigen::Index right_stride = right.outerStride();
  const Eigen::Index left_bar_stride = left_adjoint.outerStride();
  const Eigen::Index right_bar_stride = right_adjoint.outerStride();
  const PfPackedIndexCache& index_cache = get_packed_index_cache(n);

  for (int q = 0; q < n; ++q) {
    for (int p = 0; p < n; ++p) {
      const int idx_qp = packed_pair_index(q, p);
      const double* left_col_p = left_data + static_cast<Eigen::Index>(p) * left_stride;
      double* left_bar_col_p = left_bar_data + static_cast<Eigen::Index>(p) * left_bar_stride;
      for (int s = 0; s < n; ++s) {
        const double right_qs = right_data[q + static_cast<Eigen::Index>(s) * right_stride];
        const bool update_left = !is_effectively_zero(right_qs);
        const double w_right_qs = weight * right_qs;
        double* right_bar_col_s = right_bar_data + static_cast<Eigen::Index>(s) * right_bar_stride;
        double right_bar_qs = 0.0;
        for (int r = 0; r < n; ++r) {
          const int idx_sr = packed_pair_index(s, r);
          const double g = ggo_data[pair_of_pairs_index(index_cache, idx_qp, idx_sr)];
          if (is_effectively_zero(g)) {
            continue;
          }
          const double left_rp = left_col_p[r];
          const double left_pr = left_data[p + static_cast<Eigen::Index>(r) * left_stride];
          if (update_left) {
            const double delta = g * w_right_qs;
            left_bar_col_p[r] += delta;
            left_bar_data[p + static_cast<Eigen::Index>(r) * left_bar_stride] -= delta;
          }
          const double left_delta = left_rp - left_pr;
          if (!is_effectively_zero(left_delta)) {
            right_bar_qs += weight * g * left_delta;
          }
        }
        right_bar_col_s[q] += right_bar_qs;
      }
    }
  }
}

void PfTensorContractor::compute_same_spin_separable_operand_adjoints(
    double weight,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right,
    const ScalarBuffer& ggo,
    MatrixRef left_adjoint,
    MatrixRef right_adjoint) {
  const int n = validate_operand_adjoint_inputs(
      weight, ggo, left, right, left_adjoint, right_adjoint);
  if (n == 0) {
    return;
  }

  const double* ggo_data = ggo.data();
  const double* left_data = left.data();
  const double* right_data = right.data();
  double* left_bar_data = left_adjoint.data();
  double* right_bar_data = right_adjoint.data();
  const Eigen::Index left_stride = left.outerStride();
  const Eigen::Index right_stride = right.outerStride();
  const Eigen::Index left_bar_stride = left_adjoint.outerStride();
  const Eigen::Index right_bar_stride = right_adjoint.outerStride();
  const PfPackedIndexCache& index_cache = get_packed_index_cache(n);

  for (int left_first = 0; left_first < n - 1; ++left_first) {
    for (int right_first = 0; right_first < n - 1; ++right_first) {
      const double left_entry =
          left_data[right_first + static_cast<Eigen::Index>(left_first) * left_stride];
      for (int left_second = left_first + 1; left_second < n; ++left_second) {
        for (int right_second = right_first + 1; right_second < n; ++right_second) {
          const double right_entry =
              right_data[right_second + static_cast<Eigen::Index>(left_second) * right_stride];
          const int direct_qp = packed_pair_index(right_first, left_first);
          const int direct_sr = packed_pair_index(right_second, left_second);
          const int exchange_qr = packed_pair_index(right_first, left_second);
          const int exchange_sp = packed_pair_index(right_second, left_first);
          const double interaction =
              ggo_data[pair_of_pairs_index(index_cache, direct_qp, direct_sr)] -
              ggo_data[pair_of_pairs_index(index_cache, exchange_qr, exchange_sp)];
          if (is_effectively_zero(interaction)) {
            continue;
          }
          if (!is_effectively_zero(right_entry)) {
            left_bar_data[right_first +
                          static_cast<Eigen::Index>(left_first) * left_bar_stride] +=
                weight * interaction * right_entry;
          }
          if (!is_effectively_zero(left_entry)) {
            right_bar_data[right_second +
                           static_cast<Eigen::Index>(left_second) * right_bar_stride] +=
                weight * interaction * left_entry;
          }
        }
      }
    }
  }
}

void PfTensorContractor::compute_same_spin_bridge_operand_adjoints(
    double weight,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right,
    const ScalarBuffer& ggo,
    MatrixRef left_adjoint,
    MatrixRef right_adjoint) {
  const int n = validate_operand_adjoint_inputs(
      weight, ggo, left, right, left_adjoint, right_adjoint);
  if (n == 0) {
    return;
  }

  const double* ggo_data = ggo.data();
  const double* left_data = left.data();
  const double* right_data = right.data();
  double* left_bar_data = left_adjoint.data();
  double* right_bar_data = right_adjoint.data();
  const Eigen::Index left_stride = left.outerStride();
  const Eigen::Index right_stride = right.outerStride();
  const Eigen::Index left_bar_stride = left_adjoint.outerStride();
  const Eigen::Index right_bar_stride = right_adjoint.outerStride();
  const PfPackedIndexCache& index_cache = get_packed_index_cache(n);

  for (int left_first = 0; left_first < n - 1; ++left_first) {
    for (int right_first = 0; right_first < n - 1; ++right_first) {
      for (int left_second = left_first + 1; left_second < n; ++left_second) {
        const double left_entry =
            left_data[right_first + static_cast<Eigen::Index>(left_second) * left_stride];
        for (int right_second = right_first + 1; right_second < n; ++right_second) {
          const double right_entry =
              right_data[right_second + static_cast<Eigen::Index>(left_first) * right_stride];
          const int direct_qp = packed_pair_index(right_first, left_first);
          const int direct_sr = packed_pair_index(right_second, left_second);
          const int exchange_qr = packed_pair_index(right_first, left_second);
          const int exchange_sp = packed_pair_index(right_second, left_first);
          const double interaction =
              ggo_data[pair_of_pairs_index(index_cache, direct_qp, direct_sr)] -
              ggo_data[pair_of_pairs_index(index_cache, exchange_qr, exchange_sp)];
          if (is_effectively_zero(interaction)) {
            continue;
          }
          if (!is_effectively_zero(right_entry)) {
            left_bar_data[right_first +
                          static_cast<Eigen::Index>(left_second) * left_bar_stride] +=
                weight * interaction * right_entry;
          }
          if (!is_effectively_zero(left_entry)) {
            right_bar_data[right_second +
                           static_cast<Eigen::Index>(left_first) * right_bar_stride] +=
                weight * interaction * left_entry;
          }
        }
      }
    }
  }
}

void PfTensorContractor::add_exchange_outer_product(
    double weight,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right,
    ScalarBuffer* ggo_grad) {
  if (weight == 0.0) {
    return;
  }
  const int n = validate_outer_product_inputs(left, right);
  double* ggo_grad_data = prepare_gradient_buffer(ggo_grad, n);
  const double* left_data = left.data();
  const double* right_data = right.data();
  const Eigen::Index left_stride = left.outerStride();
  const Eigen::Index right_stride = right.outerStride();
  const PfPackedIndexCache& index_cache = get_packed_index_cache(n);

  for (int q = 0; q < n; ++q) {
    const double* right_col_q = right_data + static_cast<Eigen::Index>(q) * right_stride;
    for (int r = 0; r < n; ++r) {
      const double* left_col_r = left_data + static_cast<Eigen::Index>(r) * left_stride;
      for (int s = 0; s < n; ++s) {
        const int idx_sr = packed_pair_index(s, r);
        const double right_sq = right_col_q[s];
        if (std::abs(right_sq) < 1.0e-15) {
          continue;
        }
        const double w_right_sq = weight * right_sq;
        for (int p = 0; p < n; ++p) {
          const int idx_qp = packed_pair_index(q, p);
          ggo_grad_data[pair_of_pairs_index(index_cache, idx_qp, idx_sr)] +=
              w_right_sq * left_col_r[p];
        }
      }
    }
  }
}

void PfTensorContractor::add_coulomb_outer_product(
    double weight,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right,
    ScalarBuffer* ggo_grad) {
  if (weight == 0.0) {
    return;
  }
  const int n = validate_outer_product_inputs(left, right);
  double* ggo_grad_data = prepare_gradient_buffer(ggo_grad, n);
  const double* left_data = left.data();
  const double* right_data = right.data();
  const Eigen::Index left_stride = left.outerStride();
  const Eigen::Index right_stride = right.outerStride();
  const PfPackedIndexCache& index_cache = get_packed_index_cache(n);

  for (int q = 0; q < n; ++q) {
    for (int p = 0; p < n; ++p) {
      const int idx_qp = packed_pair_index(q, p);
      const double* left_col_p = left_data + static_cast<Eigen::Index>(p) * left_stride;
      for (int s = 0; s < n; ++s) {
        const double right_qs = right_data[q + static_cast<Eigen::Index>(s) * right_stride];
        if (std::abs(right_qs) < 1.0e-15) {
          continue;
        }
        const double w_right_qs = weight * right_qs;
        for (int r = 0; r < n; ++r) {
          const int idx_sr = packed_pair_index(s, r);
          ggo_grad_data[pair_of_pairs_index(index_cache, idx_qp, idx_sr)] +=
              w_right_qs * left_col_p[r];
        }
      }
    }
  }
}

void PfTensorContractor::add_direct_outer_product(
    double weight,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right,
    ScalarBuffer* ggo_grad) {
  if (weight == 0.0) {
    return;
  }
  const int n = validate_outer_product_inputs(left, right);
  double* ggo_grad_data = prepare_gradient_buffer(ggo_grad, n);
  const double* left_data = left.data();
  const double* right_data = right.data();
  const Eigen::Index left_stride = left.outerStride();
  const Eigen::Index right_stride = right.outerStride();
  const PfPackedIndexCache& index_cache = get_packed_index_cache(n);

  for (int q = 0; q < n; ++q) {
    for (int p = 0; p < n; ++p) {
      const int idx_qp = packed_pair_index(q, p);
      const double left_qp =
          left_data[q + static_cast<Eigen::Index>(p) * left_stride];
      if (is_effectively_zero(left_qp)) {
        continue;
      }
      const double weighted_left = weight * left_qp;
      for (int s = 0; s < n; ++s) {
        for (int r = 0; r < n; ++r) {
          const int idx_sr = packed_pair_index(s, r);
          ggo_grad_data[pair_of_pairs_index(index_cache, idx_qp, idx_sr)] +=
              weighted_left *
              right_data[s + static_cast<Eigen::Index>(r) * right_stride];
        }
      }
    }
  }
}

void PfTensorContractor::add_same_spin_separable_outer_product(
    double weight,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right,
    ScalarBuffer* ggo_grad) {
  if (weight == 0.0) {
    return;
  }
  const int n = validate_outer_product_inputs(left, right);
  double* ggo_grad_data = prepare_gradient_buffer(ggo_grad, n);
  const double* left_data = left.data();
  const double* right_data = right.data();
  const Eigen::Index left_stride = left.outerStride();
  const Eigen::Index right_stride = right.outerStride();
  const PfPackedIndexCache& index_cache = get_packed_index_cache(n);

  for (int left_first = 0; left_first < n - 1; ++left_first) {
    for (int right_first = 0; right_first < n - 1; ++right_first) {
      const double left_entry =
          left_data[right_first + static_cast<Eigen::Index>(left_first) * left_stride];
      if (is_effectively_zero(left_entry)) {
        continue;
      }
      const double weighted_left = weight * left_entry;
      for (int left_second = left_first + 1; left_second < n; ++left_second) {
        for (int right_second = right_first + 1; right_second < n; ++right_second) {
          const double right_entry =
              right_data[right_second + static_cast<Eigen::Index>(left_second) * right_stride];
          if (is_effectively_zero(right_entry)) {
            continue;
          }
          const int direct_qp = packed_pair_index(right_first, left_first);
          const int direct_sr = packed_pair_index(right_second, left_second);
          const int exchange_qr = packed_pair_index(right_first, left_second);
          const int exchange_sp = packed_pair_index(right_second, left_first);
          const double coeff = weighted_left * right_entry;
          ggo_grad_data[pair_of_pairs_index(index_cache, direct_qp, direct_sr)] += coeff;
          ggo_grad_data[pair_of_pairs_index(index_cache, exchange_qr, exchange_sp)] -= coeff;
        }
      }
    }
  }
}

void PfTensorContractor::add_same_spin_bridge_outer_product(
    double weight,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right,
    ScalarBuffer* ggo_grad) {
  if (weight == 0.0) {
    return;
  }
  const int n = validate_outer_product_inputs(left, right);
  double* ggo_grad_data = prepare_gradient_buffer(ggo_grad, n);
  const double* left_data = left.data();
  const double* right_data = right.data();
  const Eigen::Index left_stride = left.outerStride();
  const Eigen::Index right_stride = right.outerStride();
  const PfPackedIndexCache& index_cache = get_packed_index_cache(n);

  for (int left_first = 0; left_first < n - 1; ++left_first) {
    for (int right_first = 0; right_first < n - 1; ++right_first) {
      for (int left_second = left_first + 1; left_second < n; ++left_second) {
        const double left_entry =
            left_data[right_first + static_cast<Eigen::Index>(left_second) * left_stride];
        if (is_effectively_zero(left_entry)) {
          continue;
        }
        const double weighted_left = weight * left_entry;
        for (int right_second = right_first + 1; right_second < n; ++right_second) {
          const double right_entry =
              right_data[right_second + static_cast<Eigen::Index>(left_first) * right_stride];
          if (is_effectively_zero(right_entry)) {
            continue;
          }
          const int direct_qp = packed_pair_index(right_first, left_first);
          const int direct_sr = packed_pair_index(right_second, left_second);
          const int exchange_qr = packed_pair_index(right_first, left_second);
          const int exchange_sp = packed_pair_index(right_second, left_first);
          const double coeff = weighted_left * right_entry;
          ggo_grad_data[pair_of_pairs_index(index_cache, direct_qp, direct_sr)] += coeff;
          ggo_grad_data[pair_of_pairs_index(index_cache, exchange_qr, exchange_sp)] -= coeff;
        }
      }
    }
  }
}

void PfTensorContractor::add_same_spin_exchange_outer_product(
    double weight,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right,
    ScalarBuffer* ggo_grad) {
  if (weight == 0.0) {
    return;
  }
  const int n = validate_outer_product_inputs(left, right);
  double* ggo_grad_data = prepare_gradient_buffer(ggo_grad, n);
  const double* left_data = left.data();
  const double* right_data = right.data();
  const Eigen::Index left_stride = left.outerStride();
  const Eigen::Index right_stride = right.outerStride();
  const PfPackedIndexCache& index_cache = get_packed_index_cache(n);

  for (int q = 0; q < n; ++q) {
    const double* right_col_q = right_data + static_cast<Eigen::Index>(q) * right_stride;
    for (int r = 0; r < n; ++r) {
      const int idx_qr = packed_pair_index(q, r);
      const double* left_col_r = left_data + static_cast<Eigen::Index>(r) * left_stride;
      for (int s = 0; s < n; ++s) {
        const int idx_sr = packed_pair_index(s, r);
        const double right_sq = right_col_q[s];
        if (std::abs(right_sq) < 1.0e-15) {
          continue;
        }
        const double w_right_sq = weight * right_sq;
        for (int p = 0; p < n; ++p) {
          const int idx_qp = packed_pair_index(q, p);
          const int idx_sp = packed_pair_index(s, p);
          const double coeff = w_right_sq * left_col_r[p];
          ggo_grad_data[pair_of_pairs_index(index_cache, idx_qp, idx_sr)] += coeff;
          ggo_grad_data[pair_of_pairs_index(index_cache, idx_qr, idx_sp)] -= coeff;
        }
      }
    }
  }
}

void PfTensorContractor::add_same_spin_coulomb_outer_product(
    double weight,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right,
    ScalarBuffer* ggo_grad) {
  if (weight == 0.0) {
    return;
  }
  const int n = validate_outer_product_inputs(left, right);
  double* ggo_grad_data = prepare_gradient_buffer(ggo_grad, n);
  const double* left_data = left.data();
  const double* right_data = right.data();
  const Eigen::Index left_stride = left.outerStride();
  const Eigen::Index right_stride = right.outerStride();
  const PfPackedIndexCache& index_cache = get_packed_index_cache(n);

  for (int q = 0; q < n; ++q) {
    for (int p = 0; p < n; ++p) {
      const int idx_qp = packed_pair_index(q, p);
      const double* left_col_p = left_data + static_cast<Eigen::Index>(p) * left_stride;
      for (int s = 0; s < n; ++s) {
        const int idx_sp = packed_pair_index(s, p);
        const double right_qs = right_data[q + static_cast<Eigen::Index>(s) * right_stride];
        if (std::abs(right_qs) < 1.0e-15) {
          continue;
        }
        const double w_right_qs = weight * right_qs;
        for (int r = 0; r < n; ++r) {
          const int idx_sr = packed_pair_index(s, r);
          const int idx_qr = packed_pair_index(q, r);
          const double coeff = w_right_qs * left_col_p[r];
          ggo_grad_data[pair_of_pairs_index(index_cache, idx_qp, idx_sr)] += coeff;
          ggo_grad_data[pair_of_pairs_index(index_cache, idx_qr, idx_sp)] -= coeff;
        }
      }
    }
  }
}

}  // namespace xmvb::pfaffian_vbscf
