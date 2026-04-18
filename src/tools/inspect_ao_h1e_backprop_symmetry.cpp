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
#include "vb/orbital/ao_effective_one_electron_backpropagator.hpp"

namespace {

using Matrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

struct Options {
  std::string input_path;
};

Options parse_arguments(int argc, char** argv) {
  if (argc != 2) {
    throw std::invalid_argument("usage: inspect_ao_h1e_backprop_symmetry <input.xmi>");
  }
  return {.input_path = argv[1]};
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

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
    const auto& input = load_result.input;
    const int n_basis_functions = input.orbital_preparation_input.n_basis_functions;

    xmvb::vb::ActiveSpaceOrbitalPreparer orbital_preparer;
    xmvb::vb::AoEffectiveOneElectronBackpropagator backpropagator;
    const auto orbital_result =
        orbital_preparer.prepare(input.orbital_preparation_input);

    const auto backpropagation_result =
        input.ao_integral_input.ao_two_electron_integral_values.empty()
            ? backpropagator.backpropagate(
                  orbital_result.inactive_density_matrix,
                  xmvb::vb::ensure_cpp_vb_input_ri_cache(input),
                  n_basis_functions)
            : backpropagator.backpropagate(
                  orbital_result.inactive_density_matrix,
                  input.ao_integral_input);

    const Eigen::Map<const Matrix> inactive_density_gradient(
        backpropagation_result.inactive_density_gradient.data(),
        n_basis_functions,
        n_basis_functions);
    const Matrix asymmetry =
        inactive_density_gradient - inactive_density_gradient.transpose();

    std::cout << std::setprecision(12);
    std::cout << "n_basis_functions = " << n_basis_functions << '\n';
    std::cout << "gradient_max_abs = "
              << max_abs_matrix_entry(inactive_density_gradient) << '\n';
    std::cout << "gradient_asymmetry_max_abs = "
              << max_abs_matrix_entry(asymmetry) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
