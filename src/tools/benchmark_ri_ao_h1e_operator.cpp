#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/cpp_vb_input_ri_cache.hpp"
#include "vb/orbital/active_space_matrix_backpropagator.hpp"
#include "vb/orbital/active_space_orbital_preparer.hpp"
#include "vb/orbital/ao_effective_one_electron_ri_operator.hpp"
#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"

namespace {

using Matrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

struct Options {
  std::string input_path;
  int repeats = 6;
  unsigned int seed = 12345;
};

void print_usage() {
  std::cerr << "usage: benchmark_ri_ao_h1e_operator <input.xmi> "
               "[--repeats N] [--seed N]\n";
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
    if (name == "--repeats") {
      options.repeats = std::stoi(value);
      continue;
    }
    if (name == "--seed") {
      options.seed = static_cast<unsigned int>(std::stoul(value));
      continue;
    }
    throw std::invalid_argument("unknown argument: " + name);
  }
  if (options.repeats <= 0) {
    throw std::invalid_argument("--repeats must be positive");
  }
  return options;
}

std::vector<double> flatten_matrix(const Matrix& matrix) {
  return std::vector<double>(matrix.data(), matrix.data() + matrix.size());
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

double inf_norm(const std::vector<double>& values) {
  double result = 0.0;
  for (double value : values) {
    result = std::max(result, std::abs(value));
  }
  return result;
}

std::vector<double> build_production_ao_h1e_backprop_input(
    const xmvb::vb::CppVbInput& input) {
  const int n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) / 2;

  xmvb::vb::CppActiveSpaceGradientEvaluator active_space_gradient_evaluator;
  const auto active_space_gradient_result =
      active_space_gradient_evaluator.evaluate(input);

  xmvb::vb::ActiveSpaceMatrixBackpropagator matrix_backpropagator;
  const auto matrix_backpropagation_result = matrix_backpropagator.backpropagate(
      active_space_gradient_result.active_orbital_overlap_gradient,
      active_space_gradient_result.active_one_electron_gradient,
      input.orbital_preparation_input.active_orbital_overlap_matrix,
      active_space_gradient_result.ao_effective_one_electron_result.ao_effective_h1e,
      active_space_gradient_result.orbital_preparation_result.auxiliary_orbital_matrix,
      input.orbital_preparation_input.n_basis_functions,
      n_inactive_doubly_occupied_orbitals,
      input.orbital_preparation_input.n_active_orbitals);

  std::vector<double> total_ao_effective_one_electron_gradient =
      matrix_backpropagation_result.ao_effective_one_electron_gradient;
  if (total_ao_effective_one_electron_gradient.size() !=
      active_space_gradient_result.orbital_preparation_result.inactive_density_matrix.size()) {
    throw std::runtime_error("production AO-H1E backprop input size mismatch");
  }
  for (std::size_t index = 0;
       index < total_ao_effective_one_electron_gradient.size();
       ++index) {
    total_ao_effective_one_electron_gradient[index] +=
        active_space_gradient_result.orbital_preparation_result
            .inactive_density_matrix.data()[index];
  }

  const int n_basis_functions = input.orbital_preparation_input.n_basis_functions;
  const Eigen::Map<const Matrix> ao_effective_gradient(
      total_ao_effective_one_electron_gradient.data(),
      n_basis_functions,
      n_basis_functions);
  const Matrix symmetrized_operator_input =
      ao_effective_gradient + ao_effective_gradient.transpose();
  return std::vector<double>(
      symmetrized_operator_input.data(),
      symmetrized_operator_input.data() + symmetrized_operator_input.size());
}

struct TimedRunResult {
  double seconds = 0.0;
  double checksum = 0.0;
};

TimedRunResult time_operator(
    const std::vector<double>& input_matrix,
    const xmvb::vb::LibcintRiIntegralProviderResult& ri_cache,
    int n_basis_functions,
    bool attempt_spectral_factorization,
    int repeats) {
  TimedRunResult result;
  const auto start_time = std::chrono::steady_clock::now();
  for (int repeat_index = 0; repeat_index < repeats; ++repeat_index) {
    const auto output =
        xmvb::vb::apply_ao_effective_one_electron_ri_operator(
            input_matrix,
            ri_cache,
            n_basis_functions,
            {.attempt_spectral_factorization = attempt_spectral_factorization});
    double partial_checksum = 0.0;
    for (double value : output) {
      partial_checksum += value;
    }
    result.checksum += partial_checksum;
  }
  result.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
  return result;
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

    std::mt19937 generator(options.seed);
    const Matrix random_symmetric_gradient =
        random_symmetric_matrix(n_basis_functions, &generator);
    const auto random_symmetric_gradient_storage =
        flatten_matrix(random_symmetric_gradient);
    const auto production_backprop_gradient =
        build_production_ao_h1e_backprop_input(input);

    const auto inactive_dense = time_operator(
        orbital_result.inactive_density_matrix,
        ri_cache,
        n_basis_functions,
        false,
        options.repeats);
    const auto inactive_low_rank = time_operator(
        orbital_result.inactive_density_matrix,
        ri_cache,
        n_basis_functions,
        true,
        options.repeats);
    const auto gradient_dense = time_operator(
        random_symmetric_gradient_storage,
        ri_cache,
        n_basis_functions,
        false,
        options.repeats);
    const auto gradient_low_rank = time_operator(
        random_symmetric_gradient_storage,
        ri_cache,
        n_basis_functions,
        true,
        options.repeats);
    const auto production_gradient_dense = time_operator(
        production_backprop_gradient,
        ri_cache,
        n_basis_functions,
        false,
        options.repeats);
    const auto production_gradient_low_rank = time_operator(
        production_backprop_gradient,
        ri_cache,
        n_basis_functions,
        true,
        options.repeats);

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "repeats = " << options.repeats << '\n';
    std::cout << "n_basis_functions = " << n_basis_functions << '\n';
    std::cout << "n_auxiliary_functions = " << ri_cache.n_auxiliary_functions << '\n';
    std::cout << "inactive_dense_seconds = " << inactive_dense.seconds << '\n';
    std::cout << "inactive_low_rank_seconds = " << inactive_low_rank.seconds << '\n';
    std::cout << "inactive_dense_per_call_seconds = "
              << inactive_dense.seconds / static_cast<double>(options.repeats) << '\n';
    std::cout << "inactive_low_rank_per_call_seconds = "
              << inactive_low_rank.seconds / static_cast<double>(options.repeats) << '\n';
    std::cout << "gradient_dense_seconds = " << gradient_dense.seconds << '\n';
    std::cout << "gradient_low_rank_seconds = " << gradient_low_rank.seconds << '\n';
    std::cout << "gradient_dense_per_call_seconds = "
              << gradient_dense.seconds / static_cast<double>(options.repeats) << '\n';
    std::cout << "gradient_low_rank_per_call_seconds = "
              << gradient_low_rank.seconds / static_cast<double>(options.repeats) << '\n';
    std::cout << "production_gradient_inf_norm = "
              << inf_norm(production_backprop_gradient) << '\n';
    std::cout << "production_gradient_dense_seconds = "
              << production_gradient_dense.seconds << '\n';
    std::cout << "production_gradient_low_rank_seconds = "
              << production_gradient_low_rank.seconds << '\n';
    std::cout << "production_gradient_dense_per_call_seconds = "
              << production_gradient_dense.seconds /
                     static_cast<double>(options.repeats) << '\n';
    std::cout << "production_gradient_low_rank_per_call_seconds = "
              << production_gradient_low_rank.seconds /
                     static_cast<double>(options.repeats) << '\n';
    std::cout << "inactive_checksum = " << inactive_low_rank.checksum << '\n';
    std::cout << "gradient_checksum = " << gradient_low_rank.checksum << '\n';
    std::cout << "production_gradient_checksum = "
              << production_gradient_low_rank.checksum << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
