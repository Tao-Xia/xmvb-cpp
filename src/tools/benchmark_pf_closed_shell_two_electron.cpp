#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pfaffian_vbscf/kernel/pf_forward_kernel.hpp"
#include "pfaffian_vbscf/math/dense_utils.hpp"
#include "pfaffian_vbscf/matrices/pf_act_builder.hpp"
#include "pfaffian_vbscf/scf/pf_basis_factory.hpp"
#include "pfaffian_vbscf/tensor/pf_tensor_contractor.hpp"
#include "runtime/cpp_vb_input_loader.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string input_path;
  int k = 0;
  int row = 0;
  int col = 0;
  int seed = 20260328;
  int repeat = 200000;
};

void print_usage() {
  std::cerr << "usage: benchmark_pf_closed_shell_two_electron <input.xmi> "
               "[--k K] [--row I] [--col J] [--seed S] [--repeat N]\n";
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
    if (name == "--repeat") {
      opt.repeat = std::stoi(value);
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
  if (opt.repeat <= 0) {
    throw std::invalid_argument("--repeat must be positive");
  }
  return opt;
}

xmvb::pfaffian_vbscf::Matrix dense_mat(
    const std::vector<double>& data,
    int dim) {
  xmvb::pfaffian_vbscf::Matrix mat =
      xmvb::pfaffian_vbscf::Matrix::Zero(dim, dim);
  for (int col = 0; col < dim; ++col) {
    for (int row = 0; row < dim; ++row) {
      mat(row, col) = data[col * dim + row];
    }
  }
  return mat;
}

double evaluate_primitive_closed_shell_two_electron(
    const xmvb::pfaffian_vbscf::PfKernelCache& cache,
    const xmvb::pfaffian_vbscf::ScalarBuffer& ggo) {
  using xmvb::pfaffian_vbscf::PfTensorContractor;

  if (cache.trace_order <= 0) {
    return 0.0;
  }

  const int n_active_orbitals = cache.n_active_orbitals;
  const auto& a = cache.closed_shell_left_ba_block;
  const auto& b = cache.closed_shell_right_ab_block;
  const auto& c = cache.closed_shell_pair_core;
  const auto& d = cache.closed_shell_d;

  double value = 0.0;
  value +=
      PfTensorContractor::contract_exchange(
          ggo,
          n_active_orbitals,
          b,
          cache.closed_shell_pair_term_matrix) +
      PfTensorContractor::contract_coulomb(
          ggo,
          n_active_orbitals,
          b,
          cache.closed_shell_pair_term_matrix);

  if (!cache.closed_shell_coeff_n2.empty()) {
    value +=
        2.0 *
            PfTensorContractor::contract_same_spin_separable(
                ggo,
                n_active_orbitals,
                cache.closed_shell_c_poly_h_n2,
                c) +
        PfTensorContractor::contract_direct(
            ggo,
            n_active_orbitals,
            cache.closed_shell_c_poly_h_n2,
            c);
    value +=
        2.0 *
            PfTensorContractor::contract_same_spin_separable(
                ggo,
                n_active_orbitals,
                c,
                cache.closed_shell_frechet_h_c_h_n2) +
        PfTensorContractor::contract_direct(
            ggo,
            n_active_orbitals,
            c,
            cache.closed_shell_frechet_h_c_h_n2);
    value +=
        -2.0 *
        PfTensorContractor::contract_same_spin_bridge(
            ggo,
            n_active_orbitals,
            cache.closed_shell_c_poly_h_n2,
            c);
    value +=
        -2.0 *
        PfTensorContractor::contract_same_spin_bridge(
            ggo,
            n_active_orbitals,
            c,
            cache.closed_shell_frechet_h_c_h_n2);
    value +=
        -0.5 *
            PfTensorContractor::contract_exchange(
                ggo,
                n_active_orbitals,
                cache.closed_shell_d_poly_h_n2,
                a) -
        0.5 *
            PfTensorContractor::contract_coulomb(
                ggo,
                n_active_orbitals,
                cache.closed_shell_d_poly_h_n2,
                a);
    value +=
        -0.5 *
            PfTensorContractor::contract_exchange(
                ggo,
                n_active_orbitals,
                d,
                cache.closed_shell_opposite_bridge_h_split_matrix) -
        0.5 *
            PfTensorContractor::contract_coulomb(
                ggo,
                n_active_orbitals,
                d,
                cache.closed_shell_opposite_bridge_h_split_matrix);
  }

  return value;
}

template <typename Evaluator>
double benchmark_repeated(
    int repeat,
    Evaluator&& evaluator,
    double* accumulated_value,
    double* wall_seconds) {
  const auto started_at = Clock::now();
  double value = 0.0;
  for (int iter = 0; iter < repeat; ++iter) {
    value += evaluator();
  }
  const double seconds = std::chrono::duration<double>(
      Clock::now() - started_at).count();
  if (accumulated_value != nullptr) {
    *accumulated_value = value;
  }
  if (wall_seconds != nullptr) {
    *wall_seconds = seconds;
  }
  return value;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parse_args(argc, argv);
    const auto load =
        xmvb::vb::load_cpp_vb_input_with_timings(opt.input_path);

    xmvb::pfaffian_vbscf::PfBasisFactoryOptions basis_options;
    basis_options.n_states = opt.k;
    basis_options.seed = opt.seed;
    const auto basis =
        xmvb::pfaffian_vbscf::build_structure_pf_basis(
            load.input,
            load.raw_structure_data,
            basis_options);

    if (basis.has_blocked_open_shell() ||
        basis.n_alpha != basis.n_beta ||
        basis.n_beta != basis.n_singlet_pairs) {
      throw std::runtime_error(
          "benchmark_pf_closed_shell_two_electron requires a closed-shell basis");
    }
    if (opt.row >= basis.n_states || opt.col >= basis.n_states) {
      throw std::invalid_argument("--row/--col are out of range for the Pf basis");
    }

    xmvb::pfaffian_vbscf::PfActBuilder active_space_builder;
    const auto active_space = active_space_builder.build(load.input);
    const auto spatial_overlap = dense_mat(
        active_space.sso,
        active_space.n_active_orbitals);
    const auto spin_metric =
        xmvb::pfaffian_vbscf::build_spin_block_diagonal_metric(spatial_overlap);
    const auto left_pairing =
        xmvb::pfaffian_vbscf::decode_pf_state(
            basis.states[opt.row]);
    const auto right_pairing =
        xmvb::pfaffian_vbscf::decode_pf_state(
            basis.states[opt.col]);

    const auto cache =
        xmvb::pfaffian_vbscf::PfForwardKernel::build_closed_shell_exact_cache(
            left_pairing.transpose(),
            spin_metric,
            right_pairing,
            basis.n_singlet_pairs);

    const double primitive_value =
        evaluate_primitive_closed_shell_two_electron(cache, active_space.ggo);
    const double fused_value =
        xmvb::pfaffian_vbscf::PfForwardKernel::evaluate_closed_shell_two_electron(
            cache,
            active_space.ggo);
    const double abs_diff = std::abs(primitive_value - fused_value);

    // Warm up both paths before timing.
    volatile double warmup_sink = 0.0;
    for (int iter = 0; iter < 64; ++iter) {
      warmup_sink += evaluate_primitive_closed_shell_two_electron(cache, active_space.ggo);
      warmup_sink += xmvb::pfaffian_vbscf::PfForwardKernel::evaluate_closed_shell_two_electron(
          cache,
          active_space.ggo);
    }

    double primitive_accumulated = 0.0;
    double primitive_seconds = 0.0;
    benchmark_repeated(
        opt.repeat,
        [&]() {
          return evaluate_primitive_closed_shell_two_electron(cache, active_space.ggo);
        },
        &primitive_accumulated,
        &primitive_seconds);

    double fused_accumulated = 0.0;
    double fused_seconds = 0.0;
    benchmark_repeated(
        opt.repeat,
        [&]() {
          return xmvb::pfaffian_vbscf::PfForwardKernel::evaluate_closed_shell_two_electron(
              cache,
              active_space.ggo);
        },
        &fused_accumulated,
        &fused_seconds);

    std::cout << std::setprecision(16);
    std::cout << "k = " << basis.n_states << '\n';
    std::cout << "row = " << opt.row << '\n';
    std::cout << "col = " << opt.col << '\n';
    std::cout << "n_active_orbitals = " << basis.n_active_orbitals << '\n';
    std::cout << "n_pairs = " << basis.n_singlet_pairs << '\n';
    std::cout << "repeat = " << opt.repeat << '\n';
    std::cout << "primitive_value = " << primitive_value << '\n';
    std::cout << "fused_value = " << fused_value << '\n';
    std::cout << "abs_diff = " << abs_diff << '\n';
    std::cout << "primitive_seconds = " << primitive_seconds << '\n';
    std::cout << "fused_seconds = " << fused_seconds << '\n';
    std::cout << "primitive_ns_per_eval = "
              << primitive_seconds * 1.0e9 / static_cast<double>(opt.repeat) << '\n';
    std::cout << "fused_ns_per_eval = "
              << fused_seconds * 1.0e9 / static_cast<double>(opt.repeat) << '\n';
    std::cout << "speedup = "
              << (fused_seconds > 0.0 ? primitive_seconds / fused_seconds : 0.0) << '\n';
    std::cout << "primitive_accumulated = " << primitive_accumulated << '\n';
    std::cout << "fused_accumulated = " << fused_accumulated << '\n';
    std::cout << "warmup_sink = " << warmup_sink << '\n';
    return 0;
  } catch (const std::exception& err) {
    std::cerr << err.what() << '\n';
    return 1;
  }
}
