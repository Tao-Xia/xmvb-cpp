#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pfaffian_vbscf/kernel/pf_adjoint_kernel.hpp"
#include "pfaffian_vbscf/kernel/pf_forward_kernel.hpp"
#include "pfaffian_vbscf/math/dense_utils.hpp"
#include "pfaffian_vbscf/math/pf_matrix_polynomial.hpp"
#include "pfaffian_vbscf/matrices/pf_act_builder.hpp"
#include "pfaffian_vbscf/scf/pf_basis_factory.hpp"
#include "pfaffian_vbscf/scf/pf_overlap_metric_gradient.hpp"
#include "runtime/cpp_vb_input_loader.hpp"

namespace {

using Matrix = xmvb::pfaffian_vbscf::Matrix;
using ScalarBuffer = xmvb::pfaffian_vbscf::ScalarBuffer;

struct Options {
  std::string input_path;
  int k = 3;
  int row = 0;
  int col = 0;
  int seed = 20260328;
  int count = 8;
  double step = 1.0e-6;
};

struct PairValues {
  double overlap = 0.0;
  double one_electron = 0.0;
  double two_electron = 0.0;
  double total_hamiltonian = 0.0;
};

void print_usage() {
  std::cerr << "usage: check_pf_pair_sigma_grad <input.xmi> "
               "[--k K] [--row I] [--col J] [--seed S] [--count N] [--step h]\n";
}

Options parse_args(int argc, char** argv) {
  if (argc < 2 || ((argc - 2) % 2 != 0)) {
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
    if (name == "--count") {
      opt.count = std::stoi(value);
      continue;
    }
    if (name == "--step") {
      opt.step = std::stod(value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + name);
  }

  if (opt.k <= 0) {
    throw std::invalid_argument("--k must be positive");
  }
  if (opt.row < 0 || opt.col < 0) {
    throw std::invalid_argument("--row/--col must be non-negative");
  }
  if (opt.count <= 0) {
    throw std::invalid_argument("--count must be positive");
  }
  if (opt.step <= 0.0) {
    throw std::invalid_argument("--step must be positive");
  }
  return opt;
}

Matrix dense_mat(
    const ScalarBuffer& data,
    int dim,
    const char* label) {
  const std::size_t expected_size =
      dim * dim;
  if (data.size() != expected_size) {
    throw std::invalid_argument(std::string(label) + " size does not match dimension");
  }

  Matrix matrix = Matrix::Zero(dim, dim);
  for (int col = 0; col < dim; ++col) {
    for (int row = 0; row < dim; ++row) {
      matrix(row, col) = data[col * dim + row];
    }
  }
  return matrix;
}

ScalarBuffer column_major_storage(const Matrix& matrix) {
  ScalarBuffer data(
      matrix.rows() * matrix.cols(),
      0.0);
  for (int col = 0; col < matrix.cols(); ++col) {
    for (int row = 0; row < matrix.rows(); ++row) {
      data[col * matrix.rows() + row] = matrix(row, col);
    }
  }
  return data;
}

PairValues eval_pair_values(
    const Matrix& left_pairing,
    const Matrix& right_pairing,
    const Matrix& spatial_overlap,
    const Matrix& one_electron_matrix,
    const ScalarBuffer& ggo,
    int n_pairs) {
  const Matrix spin_metric =
      xmvb::pfaffian_vbscf::build_spin_block_diagonal_metric(spatial_overlap);
  const auto cache =
      xmvb::pfaffian_vbscf::PfForwardKernel::build_closed_shell_exact_cache(
          left_pairing.transpose(),
          spin_metric,
          right_pairing,
          n_pairs);

  PairValues values;
  values.overlap = cache.overlap_value;
  values.one_electron = cache.one_rdm.cwiseProduct(one_electron_matrix).sum();
  values.two_electron =
      xmvb::pfaffian_vbscf::PfForwardKernel::evaluate_closed_shell_two_electron(
          cache,
          ggo);
  values.total_hamiltonian = values.one_electron + values.two_electron;
  return values;
}

ScalarBuffer build_density_coefficients(
    const xmvb::pfaffian_vbscf::PfKernelCache& cache) {
  const int order = cache.trace_order - 1;
  if (order < 0) {
    return {};
  }

  ScalarBuffer coefficients(order + 1, 0.0);
  for (int power = 0; power <= order; ++power) {
    const double sign = ((power % 2) == 0) ? 1.0 : -1.0;
    coefficients[power] =
        2.0 * sign *
        cache.projected_overlap_coefficients[order - power];
  }
  return coefficients;
}

double eval_one_electron_with_frozen_coefficients(
    const Matrix& left_pairing,
    const Matrix& right_pairing,
    const Matrix& spatial_overlap,
    const Matrix& one_electron_matrix,
    const ScalarBuffer& density_coefficients) {
  const int n = static_cast<int>(spatial_overlap.rows());
  const Matrix left_ba =
      left_pairing.topRightCorner(n, n).transpose().eval();
  const Matrix right_ab =
      right_pairing.topRightCorner(n, n).eval();
  const Matrix pair_core =
      left_ba.transpose() * spatial_overlap * right_ab.transpose();
  const Matrix spatial_kernel = pair_core * spatial_overlap;
  const Matrix one_rdm =
      xmvb::pfaffian_vbscf::apply_left_matrix_polynomial(
          spatial_kernel,
          density_coefficients,
          pair_core);
  return one_rdm.cwiseProduct(one_electron_matrix).sum();
}

Matrix build_one_electron_direct_spatial_sigma(
    const xmvb::pfaffian_vbscf::PfKernelCache& cache,
    const Matrix& one_electron_matrix) {
  const ScalarBuffer density_coefficients = build_density_coefficients(cache);
  const auto poly_adjoint =
      xmvb::pfaffian_vbscf::backpropagate_left_matrix_polynomial(
          cache.closed_shell_spatial_kernel,
          density_coefficients,
          cache.closed_shell_pair_core,
          one_electron_matrix);

  Matrix spatial_sigma =
      cache.closed_shell_pair_core.transpose() *
      poly_adjoint.kernel_adjoint;
  Matrix pair_core_adjoint = poly_adjoint.source_adjoint;
  pair_core_adjoint.noalias() +=
      poly_adjoint.kernel_adjoint *
      cache.closed_shell_spatial_overlap.transpose();
  spatial_sigma.noalias() +=
      cache.closed_shell_left_ba_block *
      pair_core_adjoint *
      cache.closed_shell_right_ab_block;
  return spatial_sigma;
}

void print_component(
    const char* label,
    double analytic,
    double finite_difference) {
  const double abs_error = std::abs(analytic - finite_difference);
  const double rel_error =
      abs_error / std::max(1.0, std::abs(finite_difference));
  std::cout << "  " << label
            << " analytic=" << analytic
            << " fd=" << finite_difference
            << " abs_error=" << abs_error
            << " rel_error=" << rel_error
            << '\n';
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

    const int n = act.n_active_orbitals;
    const Matrix spatial_overlap =
        dense_mat(act.sso, n, "act.sso");
    const Matrix one_electron_matrix =
        dense_mat(act.hho, n, "act.hho");
    const Matrix left_pairing =
        xmvb::pfaffian_vbscf::decode_antisymmetric_matrix(
            basis.states[opt.row].packed_entries,
            basis.states[opt.row].n_spin_orbitals);
    const Matrix right_pairing =
        xmvb::pfaffian_vbscf::decode_antisymmetric_matrix(
            basis.states[opt.col].packed_entries,
            basis.states[opt.col].n_spin_orbitals);

    const Matrix spin_metric =
        xmvb::pfaffian_vbscf::build_spin_block_diagonal_metric(spatial_overlap);
    const auto cache =
        xmvb::pfaffian_vbscf::PfForwardKernel::build_closed_shell_exact_cache(
            left_pairing.transpose(),
            spin_metric,
            right_pairing,
            basis.n_beta);

    const Matrix overlap_sigma =
        dense_mat(
            xmvb::pfaffian_vbscf::evaluate_spin_resolved_overlap_metric_gradient(
                left_pairing,
                right_pairing,
                spatial_overlap,
                basis.n_alpha,
                basis.n_beta,
                basis.n_beta),
            n,
            "overlap_sigma");
    const Matrix one_electron_sigma =
        xmvb::pfaffian_vbscf::collapse_spin_diagonal_blocks(
            xmvb::pfaffian_vbscf::PfAdjointKernel::build_one_rdm_source_sigma_adjoint(
                cache,
                one_electron_matrix));
    const Matrix one_electron_direct_sigma =
        build_one_electron_direct_spatial_sigma(cache, one_electron_matrix);
    const Matrix one_electron_coefficient_sigma =
        one_electron_sigma - one_electron_direct_sigma;
    const Matrix two_electron_sigma =
        xmvb::pfaffian_vbscf::PfAdjointKernel::evaluate_closed_shell_two_electron(
            cache,
            act.ggo)
            .spatial_density;
    const Matrix total_sigma =
        overlap_sigma + one_electron_sigma + two_electron_sigma;

    std::vector<std::pair<double, int>> ranked;
    ranked.reserve(n * n);
    for (int col = 0; col < n; ++col) {
      for (int row = 0; row < n; ++row) {
        const int linear_index = col * n + row;
        ranked.emplace_back(std::abs(total_sigma(row, col)), linear_index);
      }
    }
    std::sort(
        ranked.begin(),
        ranked.end(),
        [](const auto& left, const auto& right) {
          if (left.first != right.first) {
            return left.first > right.first;
          }
          return left.second < right.second;
        });

    const int n_report = std::min(opt.count, static_cast<int>(ranked.size()));
    std::cout << std::setprecision(12);
    std::cout << "k = " << basis.n_states << '\n';
    std::cout << "seed = " << opt.seed << '\n';
    std::cout << "row = " << opt.row << '\n';
    std::cout << "col = " << opt.col << '\n';
    std::cout << "finite_difference_step = " << opt.step << '\n';
    std::cout << "reported_entries = " << n_report << '\n';

    for (int report_index = 0; report_index < n_report; ++report_index) {
      const int linear_index = ranked[report_index].second;
      const int entry_col = linear_index / n;
      const int entry_row = linear_index % n;

      Matrix plus_overlap = spatial_overlap;
      Matrix minus_overlap = spatial_overlap;
      plus_overlap(entry_row, entry_col) += opt.step;
      minus_overlap(entry_row, entry_col) -= opt.step;

      const PairValues plus =
          eval_pair_values(
              left_pairing,
              right_pairing,
              plus_overlap,
              one_electron_matrix,
              act.ggo,
              basis.n_beta);
      const PairValues minus =
          eval_pair_values(
              left_pairing,
              right_pairing,
              minus_overlap,
              one_electron_matrix,
              act.ggo,
              basis.n_beta);

      const double fd_overlap = (plus.overlap - minus.overlap) / (2.0 * opt.step);
      const double fd_one_electron =
          (plus.one_electron - minus.one_electron) / (2.0 * opt.step);
      const double fd_one_electron_direct =
          (eval_one_electron_with_frozen_coefficients(
               left_pairing,
               right_pairing,
               plus_overlap,
               one_electron_matrix,
               build_density_coefficients(cache)) -
           eval_one_electron_with_frozen_coefficients(
               left_pairing,
               right_pairing,
               minus_overlap,
               one_electron_matrix,
               build_density_coefficients(cache))) /
          (2.0 * opt.step);
      const double fd_one_electron_coefficient =
          fd_one_electron - fd_one_electron_direct;
      const double fd_two_electron =
          (plus.two_electron - minus.two_electron) / (2.0 * opt.step);
      const double fd_total =
          (plus.total_hamiltonian - minus.total_hamiltonian) / (2.0 * opt.step);

      std::cout << "entry[" << report_index << "]"
                << " row=" << entry_row
                << " col=" << entry_col
                << '\n';
      print_component("overlap", overlap_sigma(entry_row, entry_col), fd_overlap);
      print_component(
          "one_electron_sigma",
          one_electron_sigma(entry_row, entry_col),
          fd_one_electron);
      print_component(
          "one_electron_direct_sigma",
          one_electron_direct_sigma(entry_row, entry_col),
          fd_one_electron_direct);
      print_component(
          "one_electron_coefficient_sigma",
          one_electron_coefficient_sigma(entry_row, entry_col),
          fd_one_electron_coefficient);
      print_component(
          "two_electron_sigma",
          two_electron_sigma(entry_row, entry_col),
          fd_two_electron);
      print_component("total_sigma", total_sigma(entry_row, entry_col), fd_total);
    }

    return 0;
  } catch (const std::exception& err) {
    std::cerr << err.what() << '\n';
    return 1;
  }
}
