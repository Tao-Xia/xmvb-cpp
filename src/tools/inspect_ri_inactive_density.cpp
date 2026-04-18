#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
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
};

Options parse_arguments(int argc, char** argv) {
  if (argc != 2) {
    throw std::invalid_argument("usage: inspect_ri_inactive_density <input.xmi>");
  }
  return {.input_path = argv[1]};
}

std::vector<double> flatten_matrix(const Matrix& matrix) {
  return std::vector<double>(matrix.data(), matrix.data() + matrix.size());
}

double max_abs_matrix_entry(const Matrix& matrix) {
  double value = 0.0;
  for (int column = 0; column < matrix.cols(); ++column) {
    for (int row = 0; row < matrix.rows(); ++row) {
      value = std::max(value, std::abs(matrix(row, column)));
    }
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

    const Eigen::Map<const Matrix> inactive_density(
        orbital_result.inactive_density_matrix.data(),
        n_basis_functions,
        n_basis_functions);
    const Matrix inactive_density_asymmetry =
        inactive_density - inactive_density.transpose();
    const Matrix symmetrized_inactive_density =
        0.5 * (inactive_density + inactive_density.transpose()).eval();

    const auto ri_from_original =
        xmvb::vb::apply_ao_effective_one_electron_ri_operator(
            orbital_result.inactive_density_matrix,
            ri_cache,
            n_basis_functions,
            {.attempt_spectral_factorization = true});
    const auto ri_from_symmetrized =
        xmvb::vb::apply_ao_effective_one_electron_ri_operator(
            flatten_matrix(symmetrized_inactive_density),
            ri_cache,
            n_basis_functions,
            {.attempt_spectral_factorization = true});

    std::cout << std::setprecision(12);
    std::cout << "n_basis_functions = " << n_basis_functions << '\n';
    std::cout << "inactive_density_max_abs = "
              << max_abs_matrix_entry(inactive_density) << '\n';
    std::cout << "inactive_density_asymmetry_max_abs = "
              << max_abs_matrix_entry(inactive_density_asymmetry) << '\n';
    std::cout << "ri_output_original_vs_symmetrized_max_abs_diff = "
              << max_abs_difference(ri_from_original, ri_from_symmetrized) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
