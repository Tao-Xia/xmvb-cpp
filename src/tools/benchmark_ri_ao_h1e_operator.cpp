#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "input/loading/loader.hpp"
#include "vbscf/integrals/ao/ri/cache.hpp"
#include "vbscf/integrals/active/matrix/backpropagator.hpp"
#include "vbscf/integrals/active/two_electron/response/ri.hpp"
#include "vbscf/integrals/active/two_electron/transformation/packed_pair_map.hpp"
#include "vbscf/orbitals/preparation/preparer.hpp"
#include "vbscf/integrals/ao/one_electron/ri_operator.hpp"
#include "vbscf/derivatives/gradient/active_space/evaluator.hpp"

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

Matrix random_matrix(int rows, int columns, std::mt19937* generator) {
  if (generator == nullptr) {
    throw std::invalid_argument("generator must not be null");
  }
  std::normal_distribution<double> distribution(0.0, 1.0);
  Matrix matrix(rows, columns);
  for (Eigen::Index index = 0; index < matrix.size(); ++index) {
    matrix.data()[index] = distribution(*generator);
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
    const xmvb::vb::VbScfInput& input) {
  const int n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) / 2;

  xmvb::vb::ActiveSpaceGradientEvaluator active_space_gradient_evaluator;
  const auto active_space_gradient_result =
      active_space_gradient_evaluator.evaluate(input);

  xmvb::vb::ActiveSpaceMatrixBackpropagator matrix_backpropagator;
  const auto matrix_backpropagation_result = matrix_backpropagator.backpropagate(
      active_space_gradient_result.active_orbital_overlap_gradient,
      active_space_gradient_result.active_one_electron_gradient,
      input.orbital_preparation_input.ao_overlap_matrix,
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
    const xmvb::vb::RiAoFactorization& ri_cache,
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

struct FusedBenchmarkResult {
  double dense_seconds = 0.0;
  double spectral_seconds = 0.0;
  double forward_max_error = 0.0;
  double adjoint_max_error = 0.0;
  double checksum = 0.0;
};

struct ActivePairDirectionBenchmarkResult {
  double packed_seconds = 0.0;
  double planned_seconds = 0.0;
  double max_error = 0.0;
  double checksum = 0.0;
};

ActivePairDirectionBenchmarkResult benchmark_active_pair_direction(
    const xmvb::vb::RiAoFactorization& ri_cache,
    const xmvb::vb::ActiveSpaceGradientResult& gradient_result,
    int n_active_orbitals,
    int repeats,
    std::mt19937* generator) {
  const auto cache =
      xmvb::vb::build_ri_active_two_electron_response_cache(
          ri_cache,
          gradient_result.active_space_two_electron_result,
          n_active_orbitals);
  const Matrix direction = random_matrix(
      ri_cache.n_basis_functions, n_active_orbitals, generator);
  xmvb::vb::PackedOrbitalPairMapMatrix pair_direction;
  xmvb::vb::build_packed_orbital_pair_map_directional_derivative(
      gradient_result.active_space_two_electron_result
          .dense_active_coefficients,
      direction,
      &pair_direction);
  Matrix packed_result =
      ri_cache.metric_whitened_ao_pair_factors * pair_direction;
  Matrix planned_result =
      xmvb::vb::compute_ri_active_pair_factor_directional_derivative(
          cache, direction);

  ActivePairDirectionBenchmarkResult result;
  result.max_error =
      (packed_result - planned_result).cwiseAbs().maxCoeff();
  const auto run_packed = [&]() {
    const auto start = std::chrono::steady_clock::now();
    xmvb::vb::build_packed_orbital_pair_map_directional_derivative(
        gradient_result.active_space_two_electron_result
            .dense_active_coefficients,
        direction,
        &pair_direction);
    packed_result.noalias() =
        ri_cache.metric_whitened_ao_pair_factors * pair_direction;
    result.packed_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    result.checksum += packed_result(0, 0);
  };
  const auto run_planned = [&]() {
    const auto start = std::chrono::steady_clock::now();
    planned_result =
        xmvb::vb::compute_ri_active_pair_factor_directional_derivative(
            cache, direction);
    result.planned_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    result.checksum += planned_result(0, 0);
  };
  for (int repeat = 0; repeat < repeats; ++repeat) {
    if (repeat % 2 == 0) {
      run_packed();
      run_planned();
    } else {
      run_planned();
      run_packed();
    }
  }
  return result;
}

FusedBenchmarkResult benchmark_fused_low_rank_operator(
    const xmvb::vb::RiAoFactorization& ri_cache,
    int n_basis_functions,
    int n_inactive_orbitals,
    int n_active_orbitals,
    int repeats,
    std::mt19937* generator) {
  const int source_base_rank = std::max(1, n_inactive_orbitals);
  const int adjoint_base_rank =
      std::max(1, n_inactive_orbitals + n_active_orbitals);
  const Matrix source_u =
      random_matrix(n_basis_functions, source_base_rank, generator);
  const Matrix source_v =
      random_matrix(n_basis_functions, source_base_rank, generator);
  const Matrix adjoint_u =
      random_matrix(n_basis_functions, adjoint_base_rank, generator);
  const Matrix adjoint_v =
      random_matrix(n_basis_functions, adjoint_base_rank, generator);

  const Matrix dense_source =
      source_u * source_v.transpose() + source_v * source_u.transpose();
  const Matrix dense_adjoint =
      adjoint_u * adjoint_v.transpose() + adjoint_v * adjoint_u.transpose();
  xmvb::vb::AoEffectiveOneElectronRiFusedWorkspace dense_workspace;
  Matrix dense_forward;
  Matrix dense_transpose;

  xmvb::vb::apply_ao_effective_one_electron_ri_operator_fused(
      dense_source,
      dense_adjoint,
      ri_cache,
      &dense_workspace,
      &dense_forward,
      &dense_transpose);
  FusedBenchmarkResult result;

  const auto run_dense = [&]() {
    const auto start = std::chrono::steady_clock::now();
    xmvb::vb::apply_ao_effective_one_electron_ri_operator_fused(
        dense_source,
        dense_adjoint,
        ri_cache,
        &dense_workspace,
        &dense_forward,
        &dense_transpose);
    result.checksum += dense_forward(0, 0) + dense_transpose(0, 0);
    result.dense_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
  };

  const std::vector<double> dense_source_storage(
      dense_source.data(), dense_source.data() + dense_source.size());
  const std::vector<double> dense_adjoint_storage(
      dense_adjoint.data(), dense_adjoint.data() + dense_adjoint.size());
  const auto spectral_forward_reference =
      xmvb::vb::apply_ao_effective_one_electron_ri_operator(
          dense_source_storage,
          ri_cache,
          n_basis_functions,
          {.attempt_spectral_factorization = true});
  const auto spectral_transpose_reference =
      xmvb::vb::apply_ao_effective_one_electron_ri_operator(
          dense_adjoint_storage,
          ri_cache,
          n_basis_functions,
          {.attempt_spectral_factorization = true});
  const Eigen::Map<const Matrix> spectral_forward_matrix(
      spectral_forward_reference.data(), n_basis_functions, n_basis_functions);
  const Eigen::Map<const Matrix> spectral_transpose_matrix(
      spectral_transpose_reference.data(), n_basis_functions, n_basis_functions);
  result.forward_max_error =
      (dense_forward - spectral_forward_matrix).cwiseAbs().maxCoeff();
  result.adjoint_max_error =
      (dense_transpose - spectral_transpose_matrix).cwiseAbs().maxCoeff();

  const auto run_spectral = [&]() {
    const auto start = std::chrono::steady_clock::now();
    const auto spectral_forward =
        xmvb::vb::apply_ao_effective_one_electron_ri_operator(
            dense_source_storage,
            ri_cache,
            n_basis_functions,
            {.attempt_spectral_factorization = true});
    const auto spectral_transpose =
        xmvb::vb::apply_ao_effective_one_electron_ri_operator(
            dense_adjoint_storage,
            ri_cache,
            n_basis_functions,
            {.attempt_spectral_factorization = true});
    result.checksum += spectral_forward.front() + spectral_transpose.front();
    result.spectral_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
  };
  for (int repeat = 0; repeat < repeats; ++repeat) {
    if (repeat % 2 == 0) {
      run_dense();
      run_spectral();
    } else {
      run_spectral();
      run_dense();
    }
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto load_result = xmvb::vb::load_vbscf_input_with_timings(options.input_path);
    const auto& input = load_result.input;
    const int n_basis_functions = input.orbital_preparation_input.n_basis_functions;
    const int n_active_orbitals =
        input.orbital_preparation_input.n_active_orbitals;
    const int n_inactive_orbitals =
        (input.orbital_preparation_input.n_total_electrons -
         input.orbital_preparation_input.n_active_electrons) / 2;

    xmvb::vb::ActiveSpaceOrbitalPreparer orbital_preparer;
    const auto orbital_result =
        orbital_preparer.prepare(input.orbital_preparation_input);
    const auto& ri_cache = xmvb::vb::ensure_vbscf_input_ri_cache(input);
    xmvb::vb::ActiveSpaceGradientEvaluator active_space_gradient_evaluator;
    const auto active_space_gradient_result =
        active_space_gradient_evaluator.evaluate(input);

    std::mt19937 generator(options.seed);
    const Matrix random_symmetric_gradient =
        random_symmetric_matrix(n_basis_functions, &generator);
    const auto random_symmetric_gradient_storage =
        flatten_matrix(random_symmetric_gradient);
    const auto production_backprop_gradient =
        build_production_ao_h1e_backprop_input(input);
    const auto inactive_density_storage =
        flatten_matrix(orbital_result.inactive_density_matrix);

    const auto inactive_dense = time_operator(
        inactive_density_storage,
        ri_cache,
        n_basis_functions,
        false,
        options.repeats);
    const auto inactive_low_rank = time_operator(
        inactive_density_storage,
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
    const auto fused_low_rank = benchmark_fused_low_rank_operator(
        ri_cache,
        n_basis_functions,
        n_inactive_orbitals,
        n_active_orbitals,
        options.repeats,
        &generator);
    const auto active_pair_direction = benchmark_active_pair_direction(
        ri_cache,
        active_space_gradient_result,
        n_active_orbitals,
        options.repeats,
        &generator);

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "repeats = " << options.repeats << '\n';
    std::cout << "n_basis_functions = " << n_basis_functions << '\n';
    std::cout << "n_auxiliary_functions = " << ri_cache.n_auxiliary_functions << '\n';
    std::cout << "hvp_source_rank = " << 2 * std::max(1, n_inactive_orbitals) << '\n';
    std::cout << "hvp_adjoint_rank = "
              << 2 * std::max(1, n_inactive_orbitals + n_active_orbitals)
              << '\n';
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
    std::cout << "fused_dense_per_call_seconds = "
              << fused_low_rank.dense_seconds /
                     static_cast<double>(options.repeats) << '\n';
    std::cout << "spectral_pair_per_call_seconds = "
              << fused_low_rank.spectral_seconds /
                     static_cast<double>(options.repeats) << '\n';
    std::cout << "spectral_pair_speedup = "
              << fused_low_rank.dense_seconds /
                     fused_low_rank.spectral_seconds << '\n';
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "spectral_forward_max_error = "
              << fused_low_rank.forward_max_error << '\n';
    std::cout << "spectral_adjoint_max_error = "
              << fused_low_rank.adjoint_max_error << '\n';
    std::cout << std::fixed << std::setprecision(9);
    std::cout << "fused_checksum = " << fused_low_rank.checksum << '\n';
    std::cout << "active_pair_packed_per_call_seconds = "
              << active_pair_direction.packed_seconds /
                     static_cast<double>(options.repeats) << '\n';
    std::cout << "active_pair_planned_per_call_seconds = "
              << active_pair_direction.planned_seconds /
                     static_cast<double>(options.repeats) << '\n';
    std::cout << "active_pair_planned_speedup = "
              << active_pair_direction.packed_seconds /
                     active_pair_direction.planned_seconds << '\n';
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "active_pair_planned_max_error = "
              << active_pair_direction.max_error << '\n';
    std::cout << std::fixed << std::setprecision(9);
    std::cout << "active_pair_checksum = "
              << active_pair_direction.checksum << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
