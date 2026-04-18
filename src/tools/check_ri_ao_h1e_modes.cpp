#include <algorithm>
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
#include "vb/orbital/active_space_orbital_preparer.hpp"
#include "vb/orbital/ao_effective_one_electron_ri_operator.hpp"

namespace {

using Matrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

struct Options {
  std::string input_path;
  unsigned int seed = 12345;
};

Options parse_arguments(int argc, char** argv) {
  if (argc != 2 && argc != 4) {
    throw std::invalid_argument("usage: check_ri_ao_h1e_modes <input.xmi> [--seed N]");
  }
  Options options;
  options.input_path = argv[1];
  if (argc == 4) {
    const std::string name = argv[2];
    if (name != "--seed") {
      throw std::invalid_argument("unknown argument: " + name);
    }
    options.seed = static_cast<unsigned int>(std::stoul(argv[3]));
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

Matrix random_symmetric_matrix(int n, std::mt19937* generator) {
  if (generator == nullptr) {
    throw std::invalid_argument("generator must not be null");
  }
  std::normal_distribution<double> distribution(0.0, 1.0);
  Matrix matrix = Matrix::Zero(n, n);
  for (int column = 0; column < n; ++column) {
    for (int row = 0; row <= column; ++row) {
      const double value = distribution(*generator);
      matrix(row, column) = value;
      matrix(column, row) = value;
    }
  }
  return matrix;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
    const auto& input = load_result.input;
    const int n_basis_functions = input.orbital_preparation_input.n_basis_functions;

    xmvb::vb::ActiveSpaceOrbitalPreparer orbital_preparer;
    const auto orbital_result =
        orbital_preparer.prepare(input.orbital_preparation_input);
    const auto& ri_cache = xmvb::vb::ensure_cpp_vb_input_ri_cache(input);
    const std::vector<double>& inactive_density =
        orbital_result.inactive_density_matrix;

    const auto dense_p =
        xmvb::vb::apply_ao_effective_one_electron_ri_operator(
            inactive_density,
            ri_cache,
            n_basis_functions,
            {.attempt_spectral_factorization = false});
    const auto low_rank_p =
        xmvb::vb::apply_ao_effective_one_electron_ri_operator(
            inactive_density,
            ri_cache,
            n_basis_functions,
            {.attempt_spectral_factorization = true});

    std::mt19937 generator(options.seed);
    const Matrix random_gradient =
        random_symmetric_matrix(n_basis_functions, &generator);
    const auto random_gradient_storage = flatten_matrix(random_gradient);
    const auto dense_g =
        xmvb::vb::apply_ao_effective_one_electron_ri_operator(
            random_gradient_storage,
            ri_cache,
            n_basis_functions,
            {.attempt_spectral_factorization = false});
    const auto low_rank_g =
        xmvb::vb::apply_ao_effective_one_electron_ri_operator(
            random_gradient_storage,
            ri_cache,
            n_basis_functions,
            {.attempt_spectral_factorization = true});

    const double dense_adjoint_error = std::abs(
        dot_product(random_gradient_storage, dense_p) -
        dot_product(dense_g, inactive_density));
    const double low_rank_adjoint_error = std::abs(
        dot_product(random_gradient_storage, low_rank_p) -
        dot_product(low_rank_g, inactive_density));

    std::cout << std::setprecision(12);
    std::cout << "n_basis_functions = " << n_basis_functions << '\n';
    std::cout << "dense_vs_low_rank_on_inactive_density_max_abs_diff = "
              << max_abs_difference(dense_p, low_rank_p) << '\n';
    std::cout << "dense_vs_low_rank_on_random_symmetric_max_abs_diff = "
              << max_abs_difference(dense_g, low_rank_g) << '\n';
    std::cout << "dense_adjoint_error = " << dense_adjoint_error << '\n';
    std::cout << "low_rank_adjoint_error = " << low_rank_adjoint_error << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
