#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/cpp_vb_input_ri_cache.hpp"
#include "vb/orbital/ao_effective_one_electron_ri_operator.hpp"

namespace {

using Matrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

struct Options {
  std::string input_path;
  int trials = 4;
  unsigned int seed = 12345;
};

void print_usage() {
  std::cerr << "usage: check_ao_h1e_ri_operator <input.xmi> "
               "[--trials N] [--seed N]\n";
}

Options parse_arguments(int argc, char** argv) {
  if (argc < 2 || ((argc - 2) % 2 != 0)) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }

  Options options;
  options.input_path = argv[1];
  for (int argument_index = 2; argument_index < argc; argument_index += 2) {
    const std::string name = argv[argument_index];
    const std::string value = argv[argument_index + 1];
    if (name == "--trials") {
      options.trials = std::stoi(value);
      continue;
    }
    if (name == "--seed") {
      options.seed = static_cast<unsigned int>(std::stoul(value));
      continue;
    }
    throw std::invalid_argument("unknown argument: " + name);
  }
  if (options.trials <= 0) {
    throw std::invalid_argument("--trials must be positive");
  }
  return options;
}

std::vector<double> flatten_matrix(const Matrix& matrix) {
  return std::vector<double>(matrix.data(), matrix.data() + matrix.size());
}

double dot_product(
    const std::vector<double>& left,
    const std::vector<double>& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("dot-product size mismatch");
  }
  double value = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    value += left[index] * right[index];
  }
  return value;
}

Matrix random_matrix(int n, std::mt19937* generator) {
  if (generator == nullptr) {
    throw std::invalid_argument("generator must not be null");
  }
  std::normal_distribution<double> distribution(0.0, 1.0);
  Matrix matrix(n, n);
  for (int column = 0; column < n; ++column) {
    for (int row = 0; row < n; ++row) {
      matrix(row, column) = distribution(*generator);
    }
  }
  return matrix;
}

Matrix symmetrize(const Matrix& matrix) {
  return 0.5 * (matrix + matrix.transpose()).eval();
}

double max_abs_difference(
    const std::vector<double>& left,
    const std::vector<double>& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("difference size mismatch");
  }
  double value = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    value = std::max(value, std::abs(left[index] - right[index]));
  }
  return value;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
    const auto& input = load_result.input;
    const int n_basis_functions = input.ao_integral_input.n_basis_functions;
    const auto& ri_cache = xmvb::vb::ensure_cpp_vb_input_ri_cache(input);

    std::mt19937 generator(options.seed);
    double max_dense_low_rank_diff_general = 0.0;
    double max_dense_low_rank_diff_symmetric = 0.0;
    double max_general_adjoint_error = 0.0;
    double max_symmetric_adjoint_error = 0.0;

    for (int trial = 0; trial < options.trials; ++trial) {
      const Matrix general_a = random_matrix(n_basis_functions, &generator);
      const Matrix general_b = random_matrix(n_basis_functions, &generator);
      const Matrix symmetric_a = symmetrize(general_a);
      const Matrix symmetric_b = symmetrize(general_b);

      const auto dense_general_a =
          xmvb::vb::apply_ao_effective_one_electron_ri_operator(
              flatten_matrix(general_a),
              ri_cache,
              n_basis_functions,
              {.attempt_spectral_factorization = false});
      const auto low_rank_general_a =
          xmvb::vb::apply_ao_effective_one_electron_ri_operator(
              flatten_matrix(general_a),
              ri_cache,
              n_basis_functions,
              {.attempt_spectral_factorization = true});
      const auto dense_general_b =
          xmvb::vb::apply_ao_effective_one_electron_ri_operator(
              flatten_matrix(general_b),
              ri_cache,
              n_basis_functions,
              {.attempt_spectral_factorization = false});

      const auto dense_symmetric_a =
          xmvb::vb::apply_ao_effective_one_electron_ri_operator(
              flatten_matrix(symmetric_a),
              ri_cache,
              n_basis_functions,
              {.attempt_spectral_factorization = false});
      const auto low_rank_symmetric_a =
          xmvb::vb::apply_ao_effective_one_electron_ri_operator(
              flatten_matrix(symmetric_a),
              ri_cache,
              n_basis_functions,
              {.attempt_spectral_factorization = true});
      const auto dense_symmetric_b =
          xmvb::vb::apply_ao_effective_one_electron_ri_operator(
              flatten_matrix(symmetric_b),
              ri_cache,
              n_basis_functions,
              {.attempt_spectral_factorization = false});

      max_dense_low_rank_diff_general = std::max(
          max_dense_low_rank_diff_general,
          max_abs_difference(dense_general_a, low_rank_general_a));
      max_dense_low_rank_diff_symmetric = std::max(
          max_dense_low_rank_diff_symmetric,
          max_abs_difference(dense_symmetric_a, low_rank_symmetric_a));

      const double general_lhs =
          dot_product(flatten_matrix(general_a), dense_general_b);
      const double general_rhs =
          dot_product(dense_general_a, flatten_matrix(general_b));
      const double symmetric_lhs =
          dot_product(flatten_matrix(symmetric_a), dense_symmetric_b);
      const double symmetric_rhs =
          dot_product(dense_symmetric_a, flatten_matrix(symmetric_b));

      max_general_adjoint_error = std::max(
          max_general_adjoint_error,
          std::abs(general_lhs - general_rhs));
      max_symmetric_adjoint_error = std::max(
          max_symmetric_adjoint_error,
          std::abs(symmetric_lhs - symmetric_rhs));
    }

    std::cout << std::setprecision(12);
    std::cout << "trials = " << options.trials << '\n';
    std::cout << "n_basis_functions = " << n_basis_functions << '\n';
    std::cout << "max_dense_low_rank_diff_general = "
              << max_dense_low_rank_diff_general << '\n';
    std::cout << "max_dense_low_rank_diff_symmetric = "
              << max_dense_low_rank_diff_symmetric << '\n';
    std::cout << "max_general_adjoint_error = "
              << max_general_adjoint_error << '\n';
    std::cout << "max_symmetric_adjoint_error = "
              << max_symmetric_adjoint_error << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
