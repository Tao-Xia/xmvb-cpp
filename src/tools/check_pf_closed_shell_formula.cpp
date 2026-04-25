#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "pfaffian_vbscf/kernel/pf_adjoint_kernel.hpp"
#include "pfaffian_vbscf/kernel/pf_forward_kernel.hpp"
#include "pfaffian_vbscf/kernel/pf_hamiltonian_tensor_terms.hpp"
#include "pfaffian_vbscf/kernel/pf_operand_views.hpp"
#include "pfaffian_vbscf/math/dense_utils.hpp"
#include "pfaffian_vbscf/math/pf_matrix_polynomial.hpp"
#include "pfaffian_vbscf/matrices/pf_act_builder.hpp"
#include "pfaffian_vbscf/matrices/pf_pair_kernel_common.hpp"
#include "pfaffian_vbscf/scf/pf_basis_factory.hpp"
#include "pfaffian_vbscf/tensor/pf_tensor_contractor.hpp"
#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace {

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using GenericMatrix = xmvb::pfaffian_vbscf::detail::GenericMatrix<double>;

struct Options {
  std::string input_path;
  int k = 2;
  int row = 0;
  int col = 1;
  int seed = 20260328;
};

struct ExactTwoElectronChannels {
  double total = 0.0;
  double same_spin_total = 0.0;
  double opposite_spin_total = 0.0;
};

void print_usage() {
  std::cerr << "usage: check_pf_closed_shell_formula <input.xmi> "
               "[--k K] [--row I] [--col J] [--seed S]\n";
}

Options parse_args(int argc, char** argv) {
  if (argc < 2 || ((argc - 2) % 2) != 0) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }

  Options opt;
  opt.input_path = argv[1];
  for (int index = 2; index < argc; index += 2) {
    const std::string name = argv[index];
    const std::string value = argv[index + 1];
    if (name == "--k") {
      opt.k = std::stoi(value);
      continue;
    }
    if (name == "--row") {
      opt.row = std::stoi(value);
      continue;
    }
    if (name == "--col") {
      opt.col = std::stoi(value);
      continue;
    }
    if (name == "--seed") {
      opt.seed = std::stoi(value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + name);
  }

  if (opt.k < 0) {
    throw std::invalid_argument("--k must be non-negative");
  }
  if (opt.row < 0 || opt.col < 0) {
    throw std::invalid_argument("--row/--col must be non-negative");
  }
  return opt;
}

Matrix dense_mat(
    const std::vector<double>& data,
    int dim) {
  Matrix mat = Matrix::Zero(dim, dim);
  for (int col = 0; col < dim; ++col) {
    for (int row = 0; row < dim; ++row) {
      mat(row, col) = data[col * dim + row];
    }
  }
  return mat;
}

double max_abs_diff(
    const Matrix& left,
    const Matrix& right) {
  if (left.rows() != right.rows() || left.cols() != right.cols()) {
    throw std::invalid_argument("matrix dimensions do not match");
  }

  double value = 0.0;
  for (int col = 0; col < left.cols(); ++col) {
    for (int row = 0; row < left.rows(); ++row) {
      value = std::max(value, std::abs(left(row, col) - right(row, col)));
    }
  }
  return value;
}

double max_abs_diff_buffer(
    const xmvb::pfaffian_vbscf::ScalarBuffer& left,
    const xmvb::pfaffian_vbscf::ScalarBuffer& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("buffer sizes do not match");
  }
  double value = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    value = std::max(value, std::abs(left[index] - right[index]));
  }
  return value;
}

xmvb::pfaffian_vbscf::ScalarBuffer scale_outer_product_buffer(
    const xmvb::pfaffian_vbscf::ScalarBuffer& buffer,
    double scale) {
  xmvb::pfaffian_vbscf::ScalarBuffer scaled = buffer;
  for (double& value : scaled) {
    value *= scale;
  }
  return scaled;
}

Matrix build_weighted_sum(
    const std::vector<Matrix>& sequence,
    const xmvb::pfaffian_vbscf::ScalarBuffer& weights) {
  if (sequence.size() != weights.size()) {
    throw std::invalid_argument("sequence/weights size mismatch");
  }
  Matrix value = Matrix::Zero(sequence.front().rows(), sequence.front().cols());
  for (std::size_t index = 0; index < sequence.size(); ++index) {
    const double weight = weights[index];
    if (weight == 0.0) {
      continue;
    }
    value.noalias() += weight * sequence[index];
  }
  return value;
}

Matrix apply_right_matrix_polynomial(
    const Matrix& kernel,
    const xmvb::pfaffian_vbscf::ScalarBuffer& coefficients,
    const Matrix& source) {
  if (coefficients.empty()) {
    return Matrix::Zero(source.rows(), source.cols());
  }
  return xmvb::pfaffian_vbscf::apply_left_matrix_polynomial(
             kernel.transpose(),
             coefficients,
             source.transpose())
      .transpose();
}

xmvb::pfaffian_vbscf::ScalarBuffer build_projected_order_coefficients(
    const xmvb::pfaffian_vbscf::PfKernelCache& cache,
    int order) {
  if (order < 0) {
    return {};
  }
  if (static_cast<int>(cache.projected_overlap_coefficients.size()) !=
      cache.trace_order + 1) {
    throw std::invalid_argument(
        "cache.projected_overlap_coefficients size does not match "
        "cache.trace_order + 1");
  }

  xmvb::pfaffian_vbscf::ScalarBuffer coefficients(
      order + 1,
      0.0);
  for (int power = 0; power <= order; ++power) {
    const double sign = ((power % 2) == 0) ? 1.0 : -1.0;
    coefficients[power] =
        sign *
        cache.projected_overlap_coefficients[
            order - power];
  }
  return coefficients;
}

xmvb::pfaffian_vbscf::ScalarBuffer scale_buffer(
    const xmvb::pfaffian_vbscf::ScalarBuffer& buffer,
    double scale) {
  xmvb::pfaffian_vbscf::ScalarBuffer scaled = buffer;
  for (double& value : scaled) {
    value *= scale;
  }
  return scaled;
}

xmvb::pfaffian_vbscf::ScalarBuffer pad_weights(
    const xmvb::pfaffian_vbscf::ScalarBuffer& buffer,
    int size) {
  if (size < 0) {
    throw std::invalid_argument("size must be non-negative");
  }
  xmvb::pfaffian_vbscf::ScalarBuffer padded(
      size,
      0.0);
  for (std::size_t index = 0; index < buffer.size(); ++index) {
    padded[index] = buffer[index];
  }
  return padded;
}

Matrix copy_spin_block(
    const xmvb::pfaffian_vbscf::PfKernelCache& cache,
    const Matrix& source,
    xmvb::pfaffian_vbscf::PfSpinBlock block) {
  return xmvb::pfaffian_vbscf::detail::copy_spin_block(cache, source, block).eval();
}

std::vector<Matrix> build_left_power_sequence(
    const Matrix& kernel,
    const Matrix& source,
    int count) {
  if (count < 0) {
    throw std::invalid_argument("count must be non-negative");
  }
  if (kernel.rows() != kernel.cols()) {
    throw std::invalid_argument("kernel must be square");
  }
  if (source.rows() != kernel.rows()) {
    throw std::invalid_argument("kernel/source dimensions do not match");
  }

  std::vector<Matrix> sequence;
  sequence.reserve(count);
  Matrix current = source;
  for (int index = 0; index < count; ++index) {
    sequence.push_back(current);
    current = kernel * current;
  }
  return sequence;
}

double evaluate_cross_from_n2_sequence(
    const xmvb::pfaffian_vbscf::ScalarBuffer& ggo,
    int n_active_orbitals,
    const std::vector<Matrix>& gc_sequence,
    const xmvb::pfaffian_vbscf::ScalarBuffer& coeff_n2) {
  if (gc_sequence.empty() || coeff_n2.empty()) {
    return 0.0;
  }
  if (static_cast<int>(gc_sequence.size()) < static_cast<int>(coeff_n2.size())) {
    throw std::invalid_argument("gc_sequence does not cover coeff_n2");
  }

  double value = 0.0;
  for (std::size_t right_power = 0; right_power < coeff_n2.size(); ++right_power) {
    for (std::size_t left_power = 0;
         left_power + right_power < coeff_n2.size();
         ++left_power) {
      const double scale = coeff_n2[left_power + right_power];
      if (scale == 0.0) {
        continue;
      }
      value +=
          2.0 *
          scale *
          xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_separable(
              ggo,
              n_active_orbitals,
              gc_sequence[left_power],
              gc_sequence[right_power]);
      value +=
          scale *
          xmvb::pfaffian_vbscf::PfTensorContractor::contract_direct(
              ggo,
              n_active_orbitals,
              gc_sequence[left_power],
              gc_sequence[right_power]);
    }
  }
  return value;
}

double evaluate_cross_from_anchor(
    const xmvb::pfaffian_vbscf::ScalarBuffer& ggo,
    int n_active_orbitals,
    const Matrix& left,
    const Matrix& right) {
  return
      2.0 *
          xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_separable(
              ggo,
              n_active_orbitals,
              left,
              right) +
      xmvb::pfaffian_vbscf::PfTensorContractor::contract_direct(
          ggo,
          n_active_orbitals,
          left,
          right);
}

double evaluate_exchange_coulomb(
    const xmvb::pfaffian_vbscf::ScalarBuffer& ggo,
    int n_active_orbitals,
    const Matrix& left,
    const Matrix& right) {
  return
      xmvb::pfaffian_vbscf::PfTensorContractor::contract_exchange(
          ggo,
          n_active_orbitals,
          left,
          right) +
      xmvb::pfaffian_vbscf::PfTensorContractor::contract_coulomb(
          ggo,
          n_active_orbitals,
          left,
          right);
}

void add_exchange_coulomb_outer_product(
    double weight,
    const Matrix& left,
    const Matrix& right,
    xmvb::pfaffian_vbscf::ScalarBuffer* buffer) {
  xmvb::pfaffian_vbscf::PfTensorContractor::add_exchange_outer_product(
      weight,
      left,
      right,
      buffer);
  xmvb::pfaffian_vbscf::PfTensorContractor::add_coulomb_outer_product(
      weight,
      left,
      right,
      buffer);
}

double evaluate_same_spin_bridge_from_n2_sequence(
    const xmvb::pfaffian_vbscf::ScalarBuffer& ggo,
    int n_active_orbitals,
    const std::vector<Matrix>& gc_sequence,
    const xmvb::pfaffian_vbscf::ScalarBuffer& coeff_n2) {
  if (gc_sequence.empty() || coeff_n2.empty()) {
    return 0.0;
  }
  if (static_cast<int>(gc_sequence.size()) < static_cast<int>(coeff_n2.size())) {
    throw std::invalid_argument("gc_sequence does not cover coeff_n2");
  }

  double value = 0.0;
  for (std::size_t right_power = 0; right_power < coeff_n2.size(); ++right_power) {
    for (std::size_t left_power = 0;
         left_power + right_power < coeff_n2.size();
         ++left_power) {
      const double scale = -2.0 * coeff_n2[left_power + right_power];
      if (scale == 0.0) {
        continue;
      }
      value +=
          scale *
          xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_bridge(
              ggo,
              n_active_orbitals,
              gc_sequence[left_power],
              gc_sequence[right_power]);
    }
  }
  return value;
}

void add_same_spin_bridge_from_n2_sequence_outer_product(
    int n_active_orbitals,
    const std::vector<Matrix>& gc_sequence,
    const xmvb::pfaffian_vbscf::ScalarBuffer& coeff_n2,
    xmvb::pfaffian_vbscf::ScalarBuffer* buffer) {
  if (buffer == nullptr) {
    throw std::invalid_argument("buffer must not be null");
  }
  if (gc_sequence.empty() || coeff_n2.empty()) {
    return;
  }
  if (static_cast<int>(gc_sequence.size()) < static_cast<int>(coeff_n2.size())) {
    throw std::invalid_argument("gc_sequence does not cover coeff_n2");
  }

  for (std::size_t right_power = 0; right_power < coeff_n2.size(); ++right_power) {
    for (std::size_t left_power = 0;
         left_power + right_power < coeff_n2.size();
         ++left_power) {
      const double scale = -2.0 * coeff_n2[left_power + right_power];
      if (scale == 0.0) {
        continue;
      }
      xmvb::pfaffian_vbscf::PfTensorContractor::add_same_spin_bridge_outer_product(
          scale,
          gc_sequence[left_power],
          gc_sequence[right_power],
          buffer);
    }
  }
}

double evaluate_opposite_bridge_from_n2_sequence(
    const xmvb::pfaffian_vbscf::ScalarBuffer& ggo,
    int n_active_orbitals,
    const std::vector<Matrix>& bs_gc_sequence,
    const std::vector<Matrix>& ga_sequence,
    const xmvb::pfaffian_vbscf::ScalarBuffer& coeff_n2) {
  if (bs_gc_sequence.empty() || ga_sequence.empty() || coeff_n2.empty()) {
    return 0.0;
  }
  if (static_cast<int>(bs_gc_sequence.size()) < static_cast<int>(coeff_n2.size()) ||
      static_cast<int>(ga_sequence.size()) < static_cast<int>(coeff_n2.size())) {
    throw std::invalid_argument("bridge sequences do not cover coeff_n2");
  }

  double value = 0.0;
  for (std::size_t right_power = 0; right_power < coeff_n2.size(); ++right_power) {
    for (std::size_t left_power = 0;
         left_power + right_power < coeff_n2.size();
         ++left_power) {
      const double scale = -0.5 * coeff_n2[left_power + right_power];
      if (scale == 0.0) {
        continue;
      }
      value +=
          scale *
          evaluate_exchange_coulomb(
              ggo,
              n_active_orbitals,
              bs_gc_sequence[left_power],
              ga_sequence[right_power]);
    }
  }
  return value;
}

void add_opposite_bridge_from_n2_sequence_outer_product(
    int n_active_orbitals,
    const std::vector<Matrix>& bs_gc_sequence,
    const std::vector<Matrix>& ga_sequence,
    const xmvb::pfaffian_vbscf::ScalarBuffer& coeff_n2,
    xmvb::pfaffian_vbscf::ScalarBuffer* buffer) {
  if (buffer == nullptr) {
    throw std::invalid_argument("buffer must not be null");
  }
  if (bs_gc_sequence.empty() || ga_sequence.empty() || coeff_n2.empty()) {
    return;
  }
  if (static_cast<int>(bs_gc_sequence.size()) < static_cast<int>(coeff_n2.size()) ||
      static_cast<int>(ga_sequence.size()) < static_cast<int>(coeff_n2.size())) {
    throw std::invalid_argument("bridge sequences do not cover coeff_n2");
  }

  for (std::size_t right_power = 0; right_power < coeff_n2.size(); ++right_power) {
    for (std::size_t left_power = 0;
         left_power + right_power < coeff_n2.size();
         ++left_power) {
      const double scale = -0.5 * coeff_n2[left_power + right_power];
      if (scale == 0.0) {
        continue;
      }
      add_exchange_coulomb_outer_product(
          scale,
          bs_gc_sequence[left_power],
          ga_sequence[right_power],
          buffer);
    }
  }
}

void add_cross_from_n2_sequence_outer_product(
    int n_active_orbitals,
    const std::vector<Matrix>& gc_sequence,
    const xmvb::pfaffian_vbscf::ScalarBuffer& coeff_n2,
    xmvb::pfaffian_vbscf::ScalarBuffer* buffer) {
  if (buffer == nullptr) {
    throw std::invalid_argument("buffer must not be null");
  }
  if (gc_sequence.empty() || coeff_n2.empty()) {
    return;
  }
  if (static_cast<int>(gc_sequence.size()) < static_cast<int>(coeff_n2.size())) {
    throw std::invalid_argument("gc_sequence does not cover coeff_n2");
  }

  for (std::size_t right_power = 0; right_power < coeff_n2.size(); ++right_power) {
    for (std::size_t left_power = 0;
         left_power + right_power < coeff_n2.size();
         ++left_power) {
      const double scale = coeff_n2[left_power + right_power];
      if (scale == 0.0) {
        continue;
      }
      xmvb::pfaffian_vbscf::PfTensorContractor::add_same_spin_separable_outer_product(
          2.0 * scale,
          gc_sequence[left_power],
          gc_sequence[right_power],
          buffer);
      xmvb::pfaffian_vbscf::PfTensorContractor::add_direct_outer_product(
          scale,
          gc_sequence[left_power],
          gc_sequence[right_power],
          buffer);
    }
  }
}

ExactTwoElectronChannels evaluate_exact_two_electron_channels_generic(
    const GenericMatrix& left_pairing_matrix,
    const GenericMatrix& right_pairing_matrix,
    const GenericMatrix& spatial_overlap_matrix,
    const xmvb::pfaffian_vbscf::ScalarBuffer& packed_active_two_electron_integrals,
    int n_pairs) {
  namespace detail = xmvb::pfaffian_vbscf::detail;

  const int n_active_orbitals = static_cast<int>(spatial_overlap_matrix.rows());
  const int n_spatial_entries = n_active_orbitals * n_active_orbitals;

  GenericMatrix spin_orbital_overlap_matrix =
      GenericMatrix::Zero(2 * n_active_orbitals, 2 * n_active_orbitals);
  spin_orbital_overlap_matrix.topLeftCorner(
      n_active_orbitals,
      n_active_orbitals) = spatial_overlap_matrix;
  spin_orbital_overlap_matrix.bottomRightCorner(
      n_active_orbitals,
      n_active_orbitals) = spatial_overlap_matrix;
  const GenericMatrix left_pairing_transpose =
      left_pairing_matrix.transpose();
  const GenericMatrix kernel_matrix =
      detail::build_kernel_generic(
          left_pairing_transpose,
          spin_orbital_overlap_matrix,
          right_pairing_matrix);
  const std::vector<GenericMatrix> matrix_powers =
      detail::build_matrix_powers_including_identity_generic(
          kernel_matrix,
          n_pairs);
  const std::vector<double> traces =
      detail::build_power_traces_generic(kernel_matrix, n_pairs);

  std::vector<GenericMatrix> alpha_source_directions(
      n_spatial_entries);
  std::vector<GenericMatrix> beta_source_directions(
      n_spatial_entries);
  std::vector<GenericMatrix> alpha_first_direction_matrices(
      n_spatial_entries);
  std::vector<GenericMatrix> beta_first_direction_matrices(
      n_spatial_entries);
  std::vector<std::vector<double>> alpha_first_derivative_traces(
      n_spatial_entries);
  std::vector<std::vector<double>> beta_first_derivative_traces(
      n_spatial_entries);

  const GenericMatrix left_transpose = left_pairing_matrix.transpose();
  const GenericMatrix right_times_source_transpose =
      right_pairing_matrix * spin_orbital_overlap_matrix.transpose();
  const GenericMatrix source_times_right =
      spin_orbital_overlap_matrix * right_pairing_matrix;

  for (int left_column = 0; left_column < n_active_orbitals; ++left_column) {
    for (int right_row = 0; right_row < n_active_orbitals; ++right_row) {
      const int entry_index =
          detail::spatial_source_entry_index(
              right_row,
              left_column,
              n_active_orbitals);
      alpha_source_directions[entry_index] =
          detail::build_spin_source_direction_matrix_generic<double>(
              n_active_orbitals,
              true,
              right_row,
              left_column);
      beta_source_directions[entry_index] =
          detail::build_spin_source_direction_matrix_generic<double>(
              n_active_orbitals,
              false,
              right_row,
              left_column);
      alpha_first_direction_matrices[entry_index] =
          left_transpose *
              alpha_source_directions[entry_index] *
              right_times_source_transpose +
          left_transpose *
              source_times_right *
              alpha_source_directions[entry_index].transpose();
      beta_first_direction_matrices[entry_index] =
          left_transpose *
              beta_source_directions[entry_index] *
              right_times_source_transpose +
          left_transpose *
              source_times_right *
              beta_source_directions[entry_index].transpose();
      alpha_first_derivative_traces[entry_index] =
          detail::build_first_derivative_power_traces_generic(
              matrix_powers,
              alpha_first_direction_matrices[entry_index],
              n_pairs);
      beta_first_derivative_traces[entry_index] =
          detail::build_first_derivative_power_traces_generic(
              matrix_powers,
              beta_first_direction_matrices[entry_index],
              n_pairs);
    }
  }

  ExactTwoElectronChannels channels;
  for (int left_first = 0; left_first < n_active_orbitals - 1; ++left_first) {
    for (int right_first = 0; right_first < n_active_orbitals - 1; ++right_first) {
      const int first_entry_index =
          detail::spatial_source_entry_index(
              right_first,
              left_first,
              n_active_orbitals);
      for (int left_second = left_first + 1;
           left_second < n_active_orbitals;
           ++left_second) {
        for (int right_second = right_first + 1;
             right_second < n_active_orbitals;
             ++right_second) {
          const int second_entry_index =
              detail::spatial_source_entry_index(
                  right_second,
                  left_second,
                  n_active_orbitals);
          const int direct_index =
              xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
                  right_first,
                  left_first,
                  right_second,
                  left_second);
          const int exchange_index =
              xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
                  right_first,
                  left_second,
                  right_second,
                  left_first);
          const double same_spin_interaction =
              packed_active_two_electron_integrals[direct_index] -
              packed_active_two_electron_integrals[exchange_index];

          const GenericMatrix alpha_second_direction_matrix =
              left_transpose *
                  alpha_source_directions[first_entry_index] *
                  right_pairing_matrix *
                  alpha_source_directions[second_entry_index].transpose() +
              left_transpose *
                  alpha_source_directions[second_entry_index] *
                  right_pairing_matrix *
                  alpha_source_directions[first_entry_index].transpose();
          const std::vector<double> alpha_second_derivative_traces =
              detail::build_second_derivative_power_traces_generic(
                  matrix_powers,
                  alpha_first_direction_matrices[first_entry_index],
                  alpha_first_direction_matrices[second_entry_index],
                  alpha_second_direction_matrix,
                  n_pairs);
          const double alpha_total =
              detail::projected_overlap_second_derivative_coefficient_from_traces_generic(
                  traces,
                  alpha_first_derivative_traces[first_entry_index],
                  alpha_first_derivative_traces[second_entry_index],
                  alpha_second_derivative_traces,
                  n_pairs);

          const GenericMatrix beta_second_direction_matrix =
              left_transpose *
                  beta_source_directions[first_entry_index] *
                  right_pairing_matrix *
                  beta_source_directions[second_entry_index].transpose() +
              left_transpose *
                  beta_source_directions[second_entry_index] *
                  right_pairing_matrix *
                  beta_source_directions[first_entry_index].transpose();
          const std::vector<double> beta_second_derivative_traces =
              detail::build_second_derivative_power_traces_generic(
                  matrix_powers,
                  beta_first_direction_matrices[first_entry_index],
                  beta_first_direction_matrices[second_entry_index],
                  beta_second_direction_matrix,
                  n_pairs);
          const double beta_total =
              detail::projected_overlap_second_derivative_coefficient_from_traces_generic(
                  traces,
                  beta_first_derivative_traces[first_entry_index],
                  beta_first_derivative_traces[second_entry_index],
                  beta_second_derivative_traces,
                  n_pairs);

          channels.same_spin_total += same_spin_interaction * alpha_total;
          channels.same_spin_total += same_spin_interaction * beta_total;
        }
      }
    }
  }

  for (int alpha_left_column = 0;
       alpha_left_column < n_active_orbitals;
       ++alpha_left_column) {
    for (int alpha_right_row = 0;
         alpha_right_row < n_active_orbitals;
         ++alpha_right_row) {
      const int alpha_entry_index =
          detail::spatial_source_entry_index(
              alpha_right_row,
              alpha_left_column,
              n_active_orbitals);
      for (int beta_left_column = 0;
           beta_left_column < n_active_orbitals;
           ++beta_left_column) {
        for (int beta_right_row = 0;
             beta_right_row < n_active_orbitals;
             ++beta_right_row) {
          const int beta_entry_index =
              detail::spatial_source_entry_index(
                  beta_right_row,
                  beta_left_column,
                  n_active_orbitals);
          const int opposite_spin_index =
              xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
                  beta_right_row,
                  beta_left_column,
                  alpha_right_row,
                  alpha_left_column);
          const GenericMatrix mixed_second_direction_matrix =
              left_transpose *
                  alpha_source_directions[alpha_entry_index] *
                  right_pairing_matrix *
                  beta_source_directions[beta_entry_index].transpose() +
              left_transpose *
                  beta_source_directions[beta_entry_index] *
                  right_pairing_matrix *
                  alpha_source_directions[alpha_entry_index].transpose();
          const std::vector<double> mixed_second_derivative_traces =
              detail::build_second_derivative_power_traces_generic(
                  matrix_powers,
                  alpha_first_direction_matrices[alpha_entry_index],
                  beta_first_direction_matrices[beta_entry_index],
                  mixed_second_direction_matrix,
                  n_pairs);
          channels.opposite_spin_total +=
              packed_active_two_electron_integrals[opposite_spin_index] *
              detail::projected_overlap_second_derivative_coefficient_from_traces_generic(
                  traces,
                  alpha_first_derivative_traces[alpha_entry_index],
                  beta_first_derivative_traces[beta_entry_index],
                  mixed_second_derivative_traces,
                  n_pairs);
        }
      }
    }
  }

  channels.total = channels.same_spin_total + channels.opposite_spin_total;
  return channels;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parse_args(argc, argv);
    const auto load = xmvb::vb::load_cpp_vb_input_with_timings(opt.input_path);
    xmvb::pfaffian_vbscf::PfActBuilder act_builder;
    const auto act = act_builder.build(load.input);

    xmvb::pfaffian_vbscf::PfBasisFactoryOptions basis_options;
    basis_options.n_states = opt.k;
    basis_options.seed = opt.seed;
    const auto basis =
        xmvb::pfaffian_vbscf::build_structure_pf_basis(
            load.input,
            load.raw_structure_data,
            basis_options);
    if (opt.row >= basis.n_states || opt.col >= basis.n_states) {
      throw std::invalid_argument("selected state pair is out of range");
    }

    const Matrix spatial_overlap = dense_mat(act.sso, act.n_active_orbitals);
    const Matrix spin_metric =
        xmvb::pfaffian_vbscf::build_spin_block_diagonal_metric(spatial_overlap);
    const Matrix left_pairing =
        xmvb::pfaffian_vbscf::decode_antisymmetric_matrix(
            basis.states[opt.row].packed_entries,
            basis.states[opt.row].n_spin_orbitals);
    const Matrix right_pairing =
        xmvb::pfaffian_vbscf::decode_antisymmetric_matrix(
            basis.states[opt.col].packed_entries,
            basis.states[opt.col].n_spin_orbitals);
    const Matrix left = left_pairing.transpose();
    const Matrix right = right_pairing;
    const auto cache =
        xmvb::pfaffian_vbscf::PfForwardKernel::build_cache(
            left,
            spin_metric,
            right,
            basis.n_beta);
    const ExactTwoElectronChannels exact_channels =
        evaluate_exact_two_electron_channels_generic(
            left_pairing,
            right_pairing,
            spatial_overlap,
            act.ggo,
            basis.n_beta);

    const int n = cache.n_active_orbitals;
    const Matrix left_ab = left.topRightCorner(n, n);
    const Matrix right_ab = right.topRightCorner(n, n);
    const Matrix a = -left_ab;
    const Matrix b = right_ab;
    const Matrix g = cache.kernel.topLeftCorner(n, n);
    const Matrix c = a * spatial_overlap * b;
    const Matrix sigma_adjoint =
        xmvb::pfaffian_vbscf::PfAdjointKernel::build_sigma_adjoint(
            cache,
            xmvb::pfaffian_vbscf::ScalarBuffer{},
            std::vector<xmvb::pfaffian_vbscf::PfTensorTerm>{});
    const Matrix alpha_block = sigma_adjoint.topLeftCorner(n, n);
    const Matrix beta_block = sigma_adjoint.bottomRightCorner(n, n);
    const double overlap_value = cache.overlap_value;

    Matrix p = Matrix::Zero(n, n);
    for (int power = 0; power < cache.trace_order; ++power) {
      const double scale =
          cache.trace_weights[power] *
          static_cast<double>(power + 1);
      if (scale == 0.0) {
        continue;
      }
      p.noalias() +=
          scale *
          cache.kernel_powers[power].topLeftCorner(n, n);
    }

    const Matrix candidate_pg = 2.0 * (p * g);
    const Matrix candidate_gp = 2.0 * (g * p);
    const Matrix candidate_pc_cp = 2.0 * (p * c + c * p);
    const Matrix candidate_alpha = 2.0 * (p * c);
    const Matrix candidate_beta = 2.0 * (c * p);
    const Matrix candidate_beta_transpose = candidate_beta.transpose();
    const Matrix gamma = alpha_block;
    const auto terms =
        xmvb::pfaffian_vbscf::build_two_electron_hamiltonian_tensor_terms(cache);
    std::vector<xmvb::pfaffian_vbscf::PfTensorTerm> cross_terms;
    for (const auto& term : terms) {
      if (term.left_operand.source ==
              xmvb::pfaffian_vbscf::PfMatrixSource::TraceRDM &&
          term.right_operand.source ==
              xmvb::pfaffian_vbscf::PfMatrixSource::TraceRDM) {
        cross_terms.push_back(term);
      }
    }
    const double exact_cross =
        xmvb::pfaffian_vbscf::PfForwardKernel::evaluate_tensor_terms(
            cache,
            act.ggo,
            cross_terms);
    const double candidate_cross =
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_separable(
            act.ggo,
            n,
            alpha_block,
            alpha_block) +
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_separable(
            act.ggo,
            n,
            beta_block,
            beta_block) +
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_direct(
            act.ggo,
            n,
            alpha_block,
            beta_block);
    std::vector<Matrix> kpl_ba;
    std::vector<Matrix> kpr_ab;
    kpl_ba.reserve(cache.trace_order);
    kpr_ab.reserve(cache.trace_order);
    for (int power = 0; power < cache.trace_order; ++power) {
      kpl_ba.push_back(
          copy_spin_block(
              cache,
              cache.kernel_power_times_left[power],
              xmvb::pfaffian_vbscf::PfSpinBlock::BetaAlpha));
      kpr_ab.push_back(
          copy_spin_block(
              cache,
              cache.kernel_power_times_right[power],
              xmvb::pfaffian_vbscf::PfSpinBlock::AlphaBeta));
    }
    const Matrix bs = b * spatial_overlap;
    const std::vector<Matrix> ga_sequence =
        build_left_power_sequence(g, a, cache.trace_order);
    const std::vector<Matrix> gc_sequence =
        build_left_power_sequence(g, c, cache.trace_order);
    std::vector<Matrix> bs_gc_sequence;
    bs_gc_sequence.reserve(gc_sequence.size());
    for (const Matrix& gc_power : gc_sequence) {
      bs_gc_sequence.push_back((bs * gc_power).eval());
    }
    const auto pair_coeff_n1 =
        scale_buffer(
            build_projected_order_coefficients(cache, cache.trace_order - 1),
            0.5);
    xmvb::pfaffian_vbscf::ScalarBuffer pair_coeff_n1_tail;
    if (pair_coeff_n1.size() > 1) {
      pair_coeff_n1_tail.assign(
          pair_coeff_n1.begin() + 1,
          pair_coeff_n1.end());
    }
    const auto coeff_n2 =
        build_projected_order_coefficients(cache, cache.trace_order - 2);
    const auto pair_coeff_n2 = pad_weights(
        scale_buffer(coeff_n2, 0.5),
        cache.trace_order);
    const Matrix weighted_kpl_ba =
        build_weighted_sum(kpl_ba, pair_coeff_n1);
    const Matrix weighted_kpr_ab =
        build_weighted_sum(kpr_ab, pair_coeff_n1);
    const Matrix weighted_kpl_ba_n2 =
        pair_coeff_n2.empty()
            ? Matrix::Zero(n, n)
            : build_weighted_sum(kpl_ba, pair_coeff_n2);
    const Matrix weighted_kpr_ab_n2 =
        pair_coeff_n2.empty()
            ? Matrix::Zero(n, n)
            : build_weighted_sum(kpr_ab, pair_coeff_n2);
    const Matrix left_poly_gc_n2 =
        coeff_n2.empty()
            ? Matrix::Zero(n, n)
            : xmvb::pfaffian_vbscf::apply_left_matrix_polynomial(
                  g,
                  coeff_n2,
                  c);
    const Matrix bilateral_gc_n2 =
        coeff_n2.empty()
            ? Matrix::Zero(n, n)
            : xmvb::pfaffian_vbscf::apply_bilateral_matrix_polynomial(
                  g,
                  coeff_n2,
                  c);
    Matrix bilateral_ga_n2 = Matrix::Zero(n, n);
    if (!coeff_n2.empty()) {
      bilateral_ga_n2 =
          (0.5 *
           xmvb::pfaffian_vbscf::apply_bilateral_matrix_polynomial(
               g,
               coeff_n2,
               a))
              .eval();
    }
    Matrix bilateral_gb_n2 = Matrix::Zero(n, n);
    if (!coeff_n2.empty()) {
      bilateral_gb_n2 =
          (0.5 *
           xmvb::pfaffian_vbscf::apply_bilateral_matrix_polynomial(
               g,
               coeff_n2,
               b))
              .eval();
    }
    const Matrix bilateral_ga_n2_transpose = bilateral_ga_n2.transpose();
    const Matrix bs_bilateral_gc_n2 = bs * bilateral_gc_n2;
    const Matrix bs_left_poly_gc_n2 = bs * left_poly_gc_n2;
    const Matrix h = spatial_overlap * c;
    const Matrix d = bs * c;
    Matrix pair_term_h_exact_candidate =
        pair_coeff_n1.empty()
            ? Matrix::Zero(n, n)
            : (pair_coeff_n1.front() * a).eval();
    if (!pair_coeff_n1_tail.empty()) {
      pair_term_h_exact_candidate +=
          (c * xmvb::pfaffian_vbscf::apply_left_matrix_polynomial(
                   h,
                   pair_coeff_n1_tail,
                   spatial_overlap * a))
              .eval();
    }
    const Matrix d_poly_h_n2 =
        coeff_n2.empty()
            ? Matrix::Zero(n, n)
            : apply_right_matrix_polynomial(h, coeff_n2, d);
    const Matrix c_poly_h_n2 =
        coeff_n2.empty()
            ? Matrix::Zero(n, n)
            : apply_right_matrix_polynomial(h, coeff_n2, c);
    const Matrix frechet_h_c_n2 =
        coeff_n2.empty()
            ? Matrix::Zero(n, n)
            : xmvb::pfaffian_vbscf::apply_left_matrix_polynomial_frechet(
                  h,
                  coeff_n2,
                  c);
    const Matrix frechet_h_c_h_n2 =
        coeff_n2.empty()
            ? Matrix::Zero(n, n)
            : (frechet_h_c_n2 * h).eval();
    const Matrix opposite_bridge_h_split_matrix =
        (frechet_h_c_n2 * (spatial_overlap * a)).eval();
    const double candidate_same_spin_total_gamma =
        2.0 *
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_separable(
            act.ggo,
            n,
            gamma,
            gamma);
    const double candidate_opposite_total_gamma =
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_direct(
            act.ggo,
            n,
            gamma,
            gamma);
    const double candidate_pair_term =
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_exchange(
            act.ggo,
            n,
            right_ab,
            weighted_kpl_ba) +
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_coulomb(
            act.ggo,
            n,
            right_ab,
            weighted_kpl_ba);
    const double candidate_projected_pair_term =
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_exchange(
            act.ggo,
            n,
            weighted_kpr_ab,
            weighted_kpl_ba) +
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_coulomb(
            act.ggo,
            n,
            weighted_kpr_ab,
            weighted_kpl_ba);
    const double candidate_pair_term_n2 =
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_exchange(
            act.ggo,
            n,
            right_ab,
            weighted_kpl_ba_n2) +
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_coulomb(
            act.ggo,
            n,
            right_ab,
            weighted_kpl_ba_n2);
    const double candidate_projected_pair_term_n2 =
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_exchange(
            act.ggo,
            n,
            weighted_kpr_ab_n2,
            weighted_kpl_ba_n2) +
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_coulomb(
            act.ggo,
            n,
            weighted_kpr_ab_n2,
            weighted_kpl_ba_n2);
    const double candidate_cross_n2_exact_sequence =
        coeff_n2.empty()
            ? 0.0
            : evaluate_cross_from_n2_sequence(
                  act.ggo,
                  n,
                  gc_sequence,
                  coeff_n2);
    const double candidate_cross_left_poly_n2 =
        evaluate_cross_from_anchor(
            act.ggo,
            n,
            left_poly_gc_n2,
            c);
    const double candidate_cross_left_poly_n2_swapped =
        evaluate_cross_from_anchor(
            act.ggo,
            n,
            c,
            left_poly_gc_n2);
    const double candidate_cross_bilateral_n2 =
        evaluate_cross_from_anchor(
            act.ggo,
            n,
            bilateral_gc_n2,
            c);
    const double candidate_cross_bilateral_n2_swapped =
        evaluate_cross_from_anchor(
            act.ggo,
            n,
            c,
            bilateral_gc_n2);
    const double candidate_cross_h_split_n2 =
        evaluate_cross_from_anchor(
            act.ggo,
            n,
            c_poly_h_n2,
            c) +
        evaluate_cross_from_anchor(
            act.ggo,
            n,
            c,
            frechet_h_c_h_n2);
    const double candidate_pair_term_bilateral_n2 =
        evaluate_exchange_coulomb(
            act.ggo,
            n,
            b,
            bilateral_ga_n2);
    const double candidate_pair_term_bilateral_left_n2 =
        evaluate_exchange_coulomb(
            act.ggo,
            n,
            bilateral_gb_n2,
            a);
    const double candidate_pair_term_bilateral_transpose_n2 =
        evaluate_exchange_coulomb(
            act.ggo,
            n,
            b,
            bilateral_ga_n2_transpose);
    const double candidate_pair_term_h_exact_n1 =
        evaluate_exchange_coulomb(
            act.ggo,
            n,
            b,
            pair_term_h_exact_candidate);
    const double candidate_pair_term_bilateral_avg_n2 =
        0.5 *
        (candidate_pair_term_bilateral_n2 +
         candidate_pair_term_bilateral_left_n2);
    const double candidate_same_spin_cumulant_n2_exact_sequence =
        evaluate_same_spin_bridge_from_n2_sequence(
            act.ggo,
            n,
            gc_sequence,
            coeff_n2);
    const double candidate_opposite_bridge_n2_exact_sequence =
        evaluate_opposite_bridge_from_n2_sequence(
            act.ggo,
            n,
            bs_gc_sequence,
            ga_sequence,
            coeff_n2);
    const double candidate_opposite_cumulant_n2_exact_sequence =
        candidate_pair_term + candidate_opposite_bridge_n2_exact_sequence;
    const double candidate_cumulant_n2_exact_sequence =
        candidate_same_spin_cumulant_n2_exact_sequence +
        candidate_opposite_cumulant_n2_exact_sequence;
    const double candidate_same_spin_bridge_bilateral_n2 =
        2.0 *
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_bridge(
            act.ggo,
            n,
            bilateral_gc_n2,
            c);
    const double candidate_same_spin_bridge_bilateral_n2_swapped =
        2.0 *
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_bridge(
            act.ggo,
            n,
            c,
            bilateral_gc_n2);
    const double candidate_same_spin_bridge_left_poly_n2 =
        2.0 *
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_bridge(
            act.ggo,
            n,
            left_poly_gc_n2,
            c);
    const double candidate_same_spin_bridge_left_poly_n2_swapped =
        2.0 *
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_bridge(
            act.ggo,
            n,
            c,
            left_poly_gc_n2);
    const double candidate_same_spin_bridge_h_split_n2 =
        -2.0 *
        (xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_bridge(
             act.ggo,
             n,
             c_poly_h_n2,
             c) +
         xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_bridge(
             act.ggo,
             n,
             c,
             frechet_h_c_h_n2));
    const double candidate_opposite_bridge_bilateral_n2 =
        evaluate_exchange_coulomb(
            act.ggo,
            n,
            bs_bilateral_gc_n2,
            a);
    const double candidate_opposite_bridge_bilateral_right_n2 =
        evaluate_exchange_coulomb(
            act.ggo,
            n,
            bs * c,
            bilateral_ga_n2);
    const double candidate_opposite_bridge_bilateral_right_transpose_n2 =
        evaluate_exchange_coulomb(
            act.ggo,
            n,
            bs * c,
            bilateral_ga_n2_transpose);
    const double candidate_opposite_bridge_bilateral_avg_n2 =
        0.5 *
        (candidate_opposite_bridge_bilateral_n2 +
         candidate_opposite_bridge_bilateral_right_n2);
    const double candidate_opposite_bridge_h_split_n2 =
        -0.5 *
        (evaluate_exchange_coulomb(
             act.ggo,
             n,
             d_poly_h_n2,
             a) +
         evaluate_exchange_coulomb(
             act.ggo,
             n,
             d,
             opposite_bridge_h_split_matrix));
    const double candidate_opposite_bridge_left_poly_n2 =
        evaluate_exchange_coulomb(
            act.ggo,
            n,
            bs_left_poly_gc_n2,
            a);
    const double candidate_total_cross_exact_plus_pair_bilateral =
        candidate_cross_n2_exact_sequence + candidate_pair_term_bilateral_n2;
    const double candidate_total_cross_exact_plus_bilateral_bridges =
        candidate_cross_n2_exact_sequence +
        candidate_pair_term_bilateral_n2 +
        candidate_same_spin_bridge_bilateral_n2 +
        candidate_opposite_bridge_bilateral_n2;
    const double candidate_total_cross_exact_plus_left_poly_bridges =
        candidate_cross_n2_exact_sequence +
        candidate_pair_term_bilateral_n2 +
        candidate_same_spin_bridge_left_poly_n2 +
        candidate_opposite_bridge_left_poly_n2;
    xmvb::pfaffian_vbscf::ScalarBuffer exact_cross_outer;
    add_cross_from_n2_sequence_outer_product(
        n,
        gc_sequence,
        coeff_n2,
        &exact_cross_outer);
    xmvb::pfaffian_vbscf::ScalarBuffer candidate_cross_bilateral_outer;
    xmvb::pfaffian_vbscf::PfTensorContractor::add_same_spin_separable_outer_product(
        2.0,
        bilateral_gc_n2,
        c,
        &candidate_cross_bilateral_outer);
    xmvb::pfaffian_vbscf::PfTensorContractor::add_direct_outer_product(
        1.0,
        bilateral_gc_n2,
        c,
        &candidate_cross_bilateral_outer);
    xmvb::pfaffian_vbscf::ScalarBuffer candidate_cross_bilateral_swapped_outer;
    xmvb::pfaffian_vbscf::PfTensorContractor::add_same_spin_separable_outer_product(
        2.0,
        c,
        bilateral_gc_n2,
        &candidate_cross_bilateral_swapped_outer);
    xmvb::pfaffian_vbscf::PfTensorContractor::add_direct_outer_product(
        1.0,
        c,
        bilateral_gc_n2,
        &candidate_cross_bilateral_swapped_outer);
    xmvb::pfaffian_vbscf::ScalarBuffer candidate_cross_h_split_outer;
    xmvb::pfaffian_vbscf::PfTensorContractor::add_same_spin_separable_outer_product(
        2.0,
        c_poly_h_n2,
        c,
        &candidate_cross_h_split_outer);
    xmvb::pfaffian_vbscf::PfTensorContractor::add_direct_outer_product(
        1.0,
        c_poly_h_n2,
        c,
        &candidate_cross_h_split_outer);
    xmvb::pfaffian_vbscf::PfTensorContractor::add_same_spin_separable_outer_product(
        2.0,
        c,
        frechet_h_c_h_n2,
        &candidate_cross_h_split_outer);
    xmvb::pfaffian_vbscf::PfTensorContractor::add_direct_outer_product(
        1.0,
        c,
        frechet_h_c_h_n2,
        &candidate_cross_h_split_outer);

    xmvb::pfaffian_vbscf::ScalarBuffer exact_same_spin_cumulant_outer;
    add_same_spin_bridge_from_n2_sequence_outer_product(
        n,
        gc_sequence,
        coeff_n2,
        &exact_same_spin_cumulant_outer);
    xmvb::pfaffian_vbscf::ScalarBuffer candidate_same_spin_bilateral_outer;
    xmvb::pfaffian_vbscf::PfTensorContractor::add_same_spin_bridge_outer_product(
        2.0,
        bilateral_gc_n2,
        c,
        &candidate_same_spin_bilateral_outer);
    const xmvb::pfaffian_vbscf::ScalarBuffer candidate_same_spin_bilateral_outer_neg =
        scale_outer_product_buffer(candidate_same_spin_bilateral_outer, -1.0);
    xmvb::pfaffian_vbscf::ScalarBuffer candidate_same_spin_h_split_outer;
    xmvb::pfaffian_vbscf::PfTensorContractor::add_same_spin_bridge_outer_product(
        -2.0,
        c_poly_h_n2,
        c,
        &candidate_same_spin_h_split_outer);
    xmvb::pfaffian_vbscf::PfTensorContractor::add_same_spin_bridge_outer_product(
        -2.0,
        c,
        frechet_h_c_h_n2,
        &candidate_same_spin_h_split_outer);

    xmvb::pfaffian_vbscf::ScalarBuffer exact_opposite_bridge_outer;
    add_opposite_bridge_from_n2_sequence_outer_product(
        n,
        bs_gc_sequence,
        ga_sequence,
        coeff_n2,
        &exact_opposite_bridge_outer);
    xmvb::pfaffian_vbscf::ScalarBuffer candidate_opposite_bridge_left_outer;
    add_exchange_coulomb_outer_product(
        1.0,
        bs_bilateral_gc_n2,
        a,
        &candidate_opposite_bridge_left_outer);
    xmvb::pfaffian_vbscf::ScalarBuffer candidate_opposite_bridge_right_outer;
    add_exchange_coulomb_outer_product(
        1.0,
        bs * c,
        bilateral_ga_n2,
        &candidate_opposite_bridge_right_outer);
    xmvb::pfaffian_vbscf::ScalarBuffer candidate_opposite_bridge_right_transpose_outer;
    add_exchange_coulomb_outer_product(
        1.0,
        bs * c,
        bilateral_ga_n2_transpose,
        &candidate_opposite_bridge_right_transpose_outer);
    xmvb::pfaffian_vbscf::ScalarBuffer candidate_opposite_bridge_h_split_outer;
    add_exchange_coulomb_outer_product(
        -0.5,
        d_poly_h_n2,
        a,
        &candidate_opposite_bridge_h_split_outer);
    add_exchange_coulomb_outer_product(
        -0.5,
        d,
        opposite_bridge_h_split_matrix,
        &candidate_opposite_bridge_h_split_outer);
    const xmvb::pfaffian_vbscf::ScalarBuffer candidate_opposite_bridge_left_outer_neg =
        scale_outer_product_buffer(candidate_opposite_bridge_left_outer, -1.0);
    const xmvb::pfaffian_vbscf::ScalarBuffer candidate_opposite_bridge_right_outer_neg =
        scale_outer_product_buffer(candidate_opposite_bridge_right_outer, -1.0);
    const xmvb::pfaffian_vbscf::ScalarBuffer
        candidate_opposite_bridge_right_transpose_outer_neg =
            scale_outer_product_buffer(
                candidate_opposite_bridge_right_transpose_outer,
                -1.0);

    xmvb::pfaffian_vbscf::ScalarBuffer exact_pair_term_outer;
    add_exchange_coulomb_outer_product(
        1.0,
        b,
        weighted_kpl_ba,
        &exact_pair_term_outer);
    xmvb::pfaffian_vbscf::ScalarBuffer candidate_pair_term_h_exact_outer;
    add_exchange_coulomb_outer_product(
        1.0,
        b,
        pair_term_h_exact_candidate,
        &candidate_pair_term_h_exact_outer);
    xmvb::pfaffian_vbscf::ScalarBuffer exact_opposite_cumulant_outer =
        exact_pair_term_outer;
    if (exact_opposite_cumulant_outer.empty()) {
      exact_opposite_cumulant_outer = exact_opposite_bridge_outer;
    } else if (!exact_opposite_bridge_outer.empty()) {
      if (exact_opposite_cumulant_outer.size() != exact_opposite_bridge_outer.size()) {
        throw std::invalid_argument("outer-product buffer sizes do not match");
      }
      for (std::size_t index = 0; index < exact_opposite_cumulant_outer.size(); ++index) {
        exact_opposite_cumulant_outer[index] += exact_opposite_bridge_outer[index];
      }
    }
    xmvb::pfaffian_vbscf::ScalarBuffer exact_total_outer = exact_cross_outer;
    if (exact_total_outer.empty()) {
      exact_total_outer = exact_same_spin_cumulant_outer;
    } else if (!exact_same_spin_cumulant_outer.empty()) {
      for (std::size_t index = 0; index < exact_total_outer.size(); ++index) {
        exact_total_outer[index] += exact_same_spin_cumulant_outer[index];
      }
    }
    if (exact_total_outer.empty()) {
      exact_total_outer = exact_opposite_cumulant_outer;
    } else if (!exact_opposite_cumulant_outer.empty()) {
      for (std::size_t index = 0; index < exact_total_outer.size(); ++index) {
        exact_total_outer[index] += exact_opposite_cumulant_outer[index];
      }
    }

    xmvb::pfaffian_vbscf::ScalarBuffer candidate_pair_term_n2_outer;
    add_exchange_coulomb_outer_product(
        1.0,
        right_ab,
        weighted_kpl_ba_n2,
        &candidate_pair_term_n2_outer);
    xmvb::pfaffian_vbscf::ScalarBuffer candidate_total_pair_n2_plus_same_outer =
        candidate_pair_term_n2_outer;
    if (candidate_total_pair_n2_plus_same_outer.empty()) {
      candidate_total_pair_n2_plus_same_outer =
          candidate_same_spin_bilateral_outer_neg;
    } else if (!candidate_same_spin_bilateral_outer_neg.empty()) {
      for (std::size_t index = 0;
           index < candidate_total_pair_n2_plus_same_outer.size();
           ++index) {
        candidate_total_pair_n2_plus_same_outer[index] +=
            candidate_same_spin_bilateral_outer_neg[index];
      }
    }
    xmvb::pfaffian_vbscf::ScalarBuffer
        candidate_total_pair_n2_plus_same_plus_right_outer =
            candidate_total_pair_n2_plus_same_outer;
    if (candidate_total_pair_n2_plus_same_plus_right_outer.empty()) {
      candidate_total_pair_n2_plus_same_plus_right_outer =
          candidate_opposite_bridge_right_outer_neg;
    } else if (!candidate_opposite_bridge_right_outer_neg.empty()) {
      for (std::size_t index = 0;
           index < candidate_total_pair_n2_plus_same_plus_right_outer.size();
           ++index) {
        candidate_total_pair_n2_plus_same_plus_right_outer[index] +=
            candidate_opposite_bridge_right_outer_neg[index];
      }
    }
    xmvb::pfaffian_vbscf::ScalarBuffer candidate_opposite_cumulant_right_model_outer =
        exact_pair_term_outer;
    if (candidate_opposite_cumulant_right_model_outer.empty()) {
      candidate_opposite_cumulant_right_model_outer =
          candidate_opposite_bridge_right_outer_neg;
    } else if (!candidate_opposite_bridge_right_outer_neg.empty()) {
      if (candidate_opposite_cumulant_right_model_outer.size() !=
          candidate_opposite_bridge_right_outer_neg.size()) {
        throw std::invalid_argument("outer-product buffer sizes do not match");
      }
      for (std::size_t index = 0;
           index < candidate_opposite_cumulant_right_model_outer.size();
           ++index) {
        candidate_opposite_cumulant_right_model_outer[index] +=
            candidate_opposite_bridge_right_outer_neg[index];
      }
    }
    const double candidate_total_gamma_only =
        candidate_same_spin_total_gamma +
        candidate_opposite_total_gamma;
    const double candidate_total_gamma_plus_pair =
        candidate_total_gamma_only + candidate_pair_term;
    const double candidate_total_gamma_plus_projected_pair =
        candidate_total_gamma_only + candidate_projected_pair_term;
    const Matrix frechet_c =
        xmvb::pfaffian_vbscf::apply_left_matrix_polynomial_frechet(
            g,
            pair_coeff_n1,
            c);
    const Matrix frechet_c_n2 =
        pair_coeff_n2.empty()
            ? Matrix::Zero(n, n)
            : xmvb::pfaffian_vbscf::apply_left_matrix_polynomial_frechet(
                  g,
                  pair_coeff_n2,
                  c);
    const Matrix bs_frechet_c = b * spatial_overlap * frechet_c;
    const Matrix bs_frechet_c_n2 = b * spatial_overlap * frechet_c_n2;
    const double candidate_same_spin_bridge_frechet =
        2.0 *
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_bridge(
            act.ggo,
            n,
            frechet_c,
            c);
    const double candidate_opposite_frechet =
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_exchange(
            act.ggo,
            n,
            bs_frechet_c,
            a) +
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_coulomb(
            act.ggo,
            n,
            bs_frechet_c,
            a);
    const double candidate_same_spin_bridge_frechet_n2 =
        2.0 *
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_bridge(
            act.ggo,
            n,
            frechet_c_n2,
            c);
    const double candidate_opposite_frechet_n2 =
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_exchange(
            act.ggo,
            n,
            bs_frechet_c_n2,
            a) +
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_coulomb(
            act.ggo,
            n,
            bs_frechet_c_n2,
            a);
    const double candidate_total_gamma_plus_frechet =
        candidate_total_gamma_only +
        candidate_same_spin_bridge_frechet +
        candidate_opposite_frechet;
    const double candidate_total_gamma_pair_frechet =
        candidate_total_gamma_plus_pair +
        candidate_same_spin_bridge_frechet +
        candidate_opposite_frechet;
    const double candidate_total_pair_n1_plus_frechet_n2 =
        candidate_pair_term +
        candidate_same_spin_bridge_frechet_n2 +
        candidate_opposite_frechet_n2;
    const double candidate_total_pair_n1_plus_projected_pair_n2 =
        candidate_pair_term + candidate_projected_pair_term_n2;
    const double candidate_total_normalized_gamma_only =
        overlap_value == 0.0
            ? 0.0
            : candidate_total_gamma_only / overlap_value;
    const double candidate_total_normalized_gamma_plus_projected_pair =
        overlap_value == 0.0
            ? 0.0
            : (candidate_total_gamma_only + candidate_projected_pair_term) /
                  overlap_value;
    const double exact_total_forward =
        xmvb::pfaffian_vbscf::PfForwardKernel::evaluate_closed_shell_two_electron(
            cache,
            act.ggo);

    std::cout << std::setprecision(15);
    std::cout << "k = " << basis.n_states << '\n';
    std::cout << "row = " << opt.row << '\n';
    std::cout << "col = " << opt.col << '\n';
    std::cout << "overlap_value = " << overlap_value << '\n';
    std::cout << "trace_one_rdm = " << cache.one_rdm.trace() << '\n';
    std::cout << "diff_alpha_vs_half_one_rdm = "
              << max_abs_diff(alpha_block, 0.5 * cache.one_rdm) << '\n';
    std::cout << "trace_candidate_pg = " << candidate_pg.trace() << '\n';
    std::cout << "trace_candidate_gp = " << candidate_gp.trace() << '\n';
    std::cout << "trace_candidate_pc_cp = " << candidate_pc_cp.trace() << '\n';
    std::cout << "diff_candidate_pg = "
              << max_abs_diff(cache.one_rdm, candidate_pg) << '\n';
    std::cout << "diff_candidate_gp = "
              << max_abs_diff(cache.one_rdm, candidate_gp) << '\n';
    std::cout << "diff_candidate_pc_cp = "
              << max_abs_diff(cache.one_rdm, candidate_pc_cp) << '\n';
    std::cout << "diff_alpha_block = "
              << max_abs_diff(alpha_block, candidate_alpha) << '\n';
    std::cout << "diff_beta_vs_alpha = "
              << max_abs_diff(beta_block, alpha_block) << '\n';
    std::cout << "diff_beta_block = "
              << max_abs_diff(beta_block, candidate_beta) << '\n';
    std::cout << "diff_beta_block_transpose = "
              << max_abs_diff(beta_block, candidate_beta_transpose) << '\n';
    for (int power = 0; power < cache.trace_order; ++power) {
      const Matrix trace_alpha =
          cache.trace_rdms[power].topLeftCorner(n, n);
      const Matrix trace_beta =
          cache.trace_rdms[power].bottomRightCorner(n, n);
      const Matrix candidate_trace_alpha =
          2.0 * static_cast<double>(power + 1) *
          cache.kernel_powers[power].topLeftCorner(n, n) *
          c;
      const Matrix candidate_trace_beta =
          2.0 * static_cast<double>(power + 1) *
          c *
          cache.kernel_powers[power].topLeftCorner(n, n);
      const Matrix candidate_trace_beta_transpose = candidate_trace_beta.transpose();
      std::cout << "diff_trace_alpha[" << power << "] = "
                << max_abs_diff(trace_alpha, candidate_trace_alpha) << '\n';
      std::cout << "diff_trace_beta_vs_alpha[" << power << "] = "
                << max_abs_diff(trace_beta, trace_alpha) << '\n';
      std::cout << "diff_trace_beta[" << power << "] = "
                << max_abs_diff(trace_beta, candidate_trace_beta) << '\n';
      std::cout << "diff_trace_beta_transpose[" << power << "] = "
                << max_abs_diff(trace_beta, candidate_trace_beta_transpose) << '\n';
      std::cout << "diff_kpl_ba[" << power << "] = "
                << max_abs_diff(
                       kpl_ba[power],
                       ga_sequence[power])
                << '\n';
      std::cout << "diff_kpr_ab[" << power << "] = "
                << max_abs_diff(
                       kpr_ab[power],
                       -ga_sequence[power])
                << '\n';
      const Matrix kplsr_alpha =
          copy_spin_block(
              cache,
              cache.kernel_power_times_left_sigma_right[power],
              xmvb::pfaffian_vbscf::PfSpinBlock::AlphaAlpha);
      std::cout << "diff_kplsr_aa[" << power << "] = "
                << max_abs_diff(
                       kplsr_alpha,
                       gc_sequence[power])
                << '\n';
      const Matrix rskplsr_ab =
          copy_spin_block(
              cache,
              cache.right_sigma_times_kernel_power_times_left_sigma_right[
                  power],
              xmvb::pfaffian_vbscf::PfSpinBlock::AlphaBeta);
      std::cout << "diff_rskplsr_ab[" << power << "] = "
                << max_abs_diff(
                       rskplsr_ab,
                       bs * gc_sequence[power])
                << '\n';
    }
    std::cout << "exact_total = " << exact_channels.total << '\n';
    std::cout << "exact_total_forward = " << exact_total_forward << '\n';
    std::cout << "diff_total_forward = "
              << std::abs(exact_channels.total - exact_total_forward) << '\n';
    std::cout << "exact_same_spin_total = " << exact_channels.same_spin_total << '\n';
    std::cout << "exact_opposite_spin_total = "
              << exact_channels.opposite_spin_total << '\n';
    std::cout << "candidate_same_spin_total_gamma = "
              << candidate_same_spin_total_gamma << '\n';
    std::cout << "candidate_opposite_total_gamma = "
              << candidate_opposite_total_gamma << '\n';
    std::cout << "candidate_pair_term = " << candidate_pair_term << '\n';
    std::cout << "candidate_projected_pair_term = "
              << candidate_projected_pair_term << '\n';
    std::cout << "candidate_pair_term_n2 = " << candidate_pair_term_n2 << '\n';
    std::cout << "candidate_projected_pair_term_n2 = "
              << candidate_projected_pair_term_n2 << '\n';
    std::cout << "candidate_cross_n2_exact_sequence = "
              << candidate_cross_n2_exact_sequence << '\n';
    std::cout << "candidate_cross_left_poly_n2 = "
              << candidate_cross_left_poly_n2 << '\n';
    std::cout << "candidate_cross_left_poly_n2_swapped = "
              << candidate_cross_left_poly_n2_swapped << '\n';
    std::cout << "candidate_cross_bilateral_n2 = "
              << candidate_cross_bilateral_n2 << '\n';
    std::cout << "candidate_cross_bilateral_n2_swapped = "
              << candidate_cross_bilateral_n2_swapped << '\n';
    std::cout << "candidate_cross_h_split_n2 = "
              << candidate_cross_h_split_n2 << '\n';
    std::cout << "candidate_pair_term_bilateral_n2 = "
              << candidate_pair_term_bilateral_n2 << '\n';
    std::cout << "candidate_pair_term_bilateral_left_n2 = "
              << candidate_pair_term_bilateral_left_n2 << '\n';
    std::cout << "candidate_pair_term_bilateral_transpose_n2 = "
              << candidate_pair_term_bilateral_transpose_n2 << '\n';
    std::cout << "candidate_pair_term_h_exact_n1 = "
              << candidate_pair_term_h_exact_n1 << '\n';
    std::cout << "candidate_pair_term_bilateral_avg_n2 = "
              << candidate_pair_term_bilateral_avg_n2 << '\n';
    std::cout << "candidate_same_spin_cumulant_n2_exact_sequence = "
              << candidate_same_spin_cumulant_n2_exact_sequence << '\n';
    std::cout << "candidate_opposite_bridge_n2_exact_sequence = "
              << candidate_opposite_bridge_n2_exact_sequence << '\n';
    std::cout << "candidate_opposite_cumulant_n2_exact_sequence = "
              << candidate_opposite_cumulant_n2_exact_sequence << '\n';
    std::cout << "candidate_cumulant_n2_exact_sequence = "
              << candidate_cumulant_n2_exact_sequence << '\n';
    std::cout << "candidate_same_spin_bridge_left_poly_n2 = "
              << candidate_same_spin_bridge_left_poly_n2 << '\n';
    std::cout << "candidate_same_spin_bridge_left_poly_n2_swapped = "
              << candidate_same_spin_bridge_left_poly_n2_swapped << '\n';
    std::cout << "candidate_same_spin_bridge_bilateral_n2 = "
              << candidate_same_spin_bridge_bilateral_n2 << '\n';
    std::cout << "candidate_same_spin_bridge_bilateral_n2_swapped = "
              << candidate_same_spin_bridge_bilateral_n2_swapped << '\n';
    std::cout << "candidate_same_spin_bridge_h_split_n2 = "
              << candidate_same_spin_bridge_h_split_n2 << '\n';
    std::cout << "candidate_opposite_bridge_left_poly_n2 = "
              << candidate_opposite_bridge_left_poly_n2 << '\n';
    std::cout << "candidate_opposite_bridge_bilateral_n2 = "
              << candidate_opposite_bridge_bilateral_n2 << '\n';
    std::cout << "candidate_opposite_bridge_bilateral_right_n2 = "
              << candidate_opposite_bridge_bilateral_right_n2 << '\n';
    std::cout << "candidate_opposite_bridge_bilateral_right_transpose_n2 = "
              << candidate_opposite_bridge_bilateral_right_transpose_n2 << '\n';
    std::cout << "candidate_opposite_bridge_h_split_n2 = "
              << candidate_opposite_bridge_h_split_n2 << '\n';
    std::cout << "candidate_opposite_bridge_bilateral_avg_n2 = "
              << candidate_opposite_bridge_bilateral_avg_n2 << '\n';
    std::cout << "candidate_total_cross_exact_plus_pair_bilateral = "
              << candidate_total_cross_exact_plus_pair_bilateral << '\n';
    std::cout << "candidate_total_cross_exact_plus_bilateral_bridges = "
              << candidate_total_cross_exact_plus_bilateral_bridges << '\n';
    std::cout << "candidate_total_cross_exact_plus_left_poly_bridges = "
              << candidate_total_cross_exact_plus_left_poly_bridges << '\n';
    std::cout << "candidate_same_spin_bridge_frechet = "
              << candidate_same_spin_bridge_frechet << '\n';
    std::cout << "candidate_opposite_frechet = "
              << candidate_opposite_frechet << '\n';
    std::cout << "candidate_same_spin_bridge_frechet_n2 = "
              << candidate_same_spin_bridge_frechet_n2 << '\n';
    std::cout << "candidate_opposite_frechet_n2 = "
              << candidate_opposite_frechet_n2 << '\n';
    std::cout << "candidate_total_gamma_only = "
              << candidate_total_gamma_only << '\n';
    std::cout << "candidate_total_gamma_plus_pair = "
              << candidate_total_gamma_plus_pair << '\n';
    std::cout << "candidate_total_gamma_plus_projected_pair = "
              << candidate_total_gamma_plus_projected_pair << '\n';
    std::cout << "candidate_total_gamma_plus_frechet = "
              << candidate_total_gamma_plus_frechet << '\n';
    std::cout << "candidate_total_gamma_pair_frechet = "
              << candidate_total_gamma_pair_frechet << '\n';
    std::cout << "candidate_total_pair_n1_plus_frechet_n2 = "
              << candidate_total_pair_n1_plus_frechet_n2 << '\n';
    std::cout << "candidate_total_pair_n1_plus_projected_pair_n2 = "
              << candidate_total_pair_n1_plus_projected_pair_n2 << '\n';
    std::cout << "candidate_total_normalized_gamma_only = "
              << candidate_total_normalized_gamma_only << '\n';
    std::cout << "candidate_total_normalized_gamma_plus_projected_pair = "
              << candidate_total_normalized_gamma_plus_projected_pair << '\n';
    std::cout << "diff_same_spin_total_gamma = "
              << std::abs(exact_channels.same_spin_total -
                          candidate_same_spin_total_gamma)
              << '\n';
    std::cout << "diff_opposite_total_gamma = "
              << std::abs(exact_channels.opposite_spin_total -
                          candidate_opposite_total_gamma)
              << '\n';
    std::cout << "diff_total_gamma_only = "
              << std::abs(exact_channels.total - candidate_total_gamma_only) << '\n';
    std::cout << "diff_total_gamma_plus_pair = "
              << std::abs(exact_channels.total - candidate_total_gamma_plus_pair)
              << '\n';
    std::cout << "diff_total_gamma_plus_projected_pair = "
              << std::abs(exact_channels.total -
                          candidate_total_gamma_plus_projected_pair)
              << '\n';
    std::cout << "diff_total_gamma_plus_frechet = "
              << std::abs(exact_channels.total - candidate_total_gamma_plus_frechet)
              << '\n';
    std::cout << "diff_total_gamma_pair_frechet = "
              << std::abs(exact_channels.total - candidate_total_gamma_pair_frechet)
              << '\n';
    std::cout << "diff_total_pair_n1_plus_frechet_n2 = "
              << std::abs(exact_channels.total -
                          candidate_total_pair_n1_plus_frechet_n2)
              << '\n';
    std::cout << "diff_total_pair_n1_plus_projected_pair_n2 = "
              << std::abs(exact_channels.total -
                          candidate_total_pair_n1_plus_projected_pair_n2)
              << '\n';
    std::cout << "diff_cross_n2_exact_sequence = "
              << std::abs(exact_cross - candidate_cross_n2_exact_sequence) << '\n';
    std::cout << "diff_cross_left_poly_n2 = "
              << std::abs(exact_cross - candidate_cross_left_poly_n2) << '\n';
    std::cout << "diff_cross_left_poly_n2_swapped = "
              << std::abs(exact_cross - candidate_cross_left_poly_n2_swapped)
              << '\n';
    std::cout << "diff_cross_bilateral_n2 = "
              << std::abs(exact_cross - candidate_cross_bilateral_n2) << '\n';
    std::cout << "diff_cross_bilateral_n2_swapped = "
              << std::abs(exact_cross - candidate_cross_bilateral_n2_swapped)
              << '\n';
    std::cout << "diff_cross_h_split_n2 = "
              << std::abs(exact_cross - candidate_cross_h_split_n2) << '\n';
    std::cout << "diff_cross_outer_bilateral_n2 = "
              << max_abs_diff_buffer(
                     exact_cross_outer,
                     candidate_cross_bilateral_outer)
              << '\n';
    std::cout << "diff_cross_outer_bilateral_n2_swapped = "
              << max_abs_diff_buffer(
                     exact_cross_outer,
                     candidate_cross_bilateral_swapped_outer)
              << '\n';
    std::cout << "diff_cross_outer_h_split = "
              << max_abs_diff_buffer(
                     exact_cross_outer,
                     candidate_cross_h_split_outer)
              << '\n';
    std::cout << "diff_cumulant_n2_exact_sequence = "
              << std::abs(
                     (exact_channels.total - exact_cross) -
                     candidate_cumulant_n2_exact_sequence)
              << '\n';
    std::cout << "diff_same_spin_cumulant_outer_bilateral_neg = "
              << max_abs_diff_buffer(
                     exact_same_spin_cumulant_outer,
                     candidate_same_spin_bilateral_outer_neg)
              << '\n';
    std::cout << "diff_same_spin_cumulant_h_split = "
              << std::abs(
                     candidate_same_spin_cumulant_n2_exact_sequence -
                     candidate_same_spin_bridge_h_split_n2)
              << '\n';
    std::cout << "diff_same_spin_cumulant_outer_h_split = "
              << max_abs_diff_buffer(
                     exact_same_spin_cumulant_outer,
                     candidate_same_spin_h_split_outer)
              << '\n';
    std::cout << "diff_opposite_bridge_outer_bilateral_left = "
              << max_abs_diff_buffer(
                     exact_opposite_bridge_outer,
                     candidate_opposite_bridge_left_outer)
              << '\n';
    std::cout << "diff_opposite_bridge_outer_bilateral_left_neg = "
              << max_abs_diff_buffer(
                     exact_opposite_bridge_outer,
                     candidate_opposite_bridge_left_outer_neg)
              << '\n';
    std::cout << "diff_opposite_bridge_outer_bilateral_right = "
              << max_abs_diff_buffer(
                     exact_opposite_bridge_outer,
                     candidate_opposite_bridge_right_outer)
              << '\n';
    std::cout << "diff_opposite_bridge_outer_bilateral_right_neg = "
              << max_abs_diff_buffer(
                     exact_opposite_bridge_outer,
                     candidate_opposite_bridge_right_outer_neg)
              << '\n';
    std::cout << "diff_opposite_bridge_outer_bilateral_right_transpose_neg = "
              << max_abs_diff_buffer(
                     exact_opposite_bridge_outer,
                     candidate_opposite_bridge_right_transpose_outer_neg)
              << '\n';
    std::cout << "diff_opposite_bridge_outer_h_split = "
              << max_abs_diff_buffer(
                     exact_opposite_bridge_outer,
                     candidate_opposite_bridge_h_split_outer)
              << '\n';
    std::cout << "diff_opposite_cumulant_outer_right_model = "
              << max_abs_diff_buffer(
                     exact_opposite_cumulant_outer,
                     candidate_opposite_cumulant_right_model_outer)
              << '\n';
    std::cout << "diff_pair_term_outer_h_exact_n1 = "
              << max_abs_diff_buffer(
                     exact_pair_term_outer,
                     candidate_pair_term_h_exact_outer)
              << '\n';
    std::cout << "diff_total_outer_pair_term_n2 = "
              << max_abs_diff_buffer(
                     exact_total_outer,
                     candidate_pair_term_n2_outer)
              << '\n';
    std::cout << "diff_total_outer_pair_term_n2_plus_same = "
              << max_abs_diff_buffer(
                     exact_total_outer,
                     candidate_total_pair_n2_plus_same_outer)
              << '\n';
    std::cout << "diff_total_outer_pair_term_n2_plus_same_plus_right = "
              << max_abs_diff_buffer(
                     exact_total_outer,
                     candidate_total_pair_n2_plus_same_plus_right_outer)
              << '\n';
    std::cout << "diff_total_cross_exact_plus_pair_bilateral = "
              << std::abs(
                     exact_channels.total -
                     candidate_total_cross_exact_plus_pair_bilateral)
              << '\n';
    std::cout << "diff_total_cross_exact_plus_bilateral_bridges = "
              << std::abs(
                     exact_channels.total -
                     candidate_total_cross_exact_plus_bilateral_bridges)
              << '\n';
    std::cout << "diff_total_cross_exact_plus_left_poly_bridges = "
              << std::abs(
                     exact_channels.total -
                     candidate_total_cross_exact_plus_left_poly_bridges)
              << '\n';
    std::cout << "diff_total_normalized_gamma_only = "
              << std::abs(exact_channels.total -
                          candidate_total_normalized_gamma_only)
              << '\n';
    std::cout << "diff_total_normalized_gamma_plus_projected_pair = "
              << std::abs(exact_channels.total -
                          candidate_total_normalized_gamma_plus_projected_pair)
              << '\n';
    std::cout << "exact_cross = " << exact_cross << '\n';
    std::cout << "candidate_cross = " << candidate_cross << '\n';
    std::cout << "diff_cross = " << std::abs(exact_cross - candidate_cross) << '\n';
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << '\n';
    return 1;
  }
}
