#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/scf/cpp_orbital_gradient_evaluator.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace {

struct Options {
  std::string input_path;
  xmvb::vb::VBSCFAlgorithm algorithm = xmvb::vb::VBSCFAlgorithm::Original;
  xmvb::vb::StandardTwoElectronMode standard_two_electron_mode =
      xmvb::vb::StandardTwoElectronMode::Auto;
  int repeat = 10;
  int warmup = 1;
};

void print_usage() {
  std::cerr << "usage: benchmark_cpp_orbital_eval <input.xmi> "
               "[--algorithm original] "
               "[--standard-two-electron-mode auto|exact|ri] "
               "[--repeat N] [--warmup N]\n";
}

Options parse_arguments(int argc, char** argv) {
  if (argc < 2 || ((argc - 2) % 2 != 0)) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }

  Options options;
  options.input_path = argv[1];
  for (int argument_index = 2; argument_index < argc; argument_index += 2) {
    const std::string argument_name = argv[argument_index];
    const std::string argument_value = argv[argument_index + 1];
    if (argument_name == "--algorithm") {
      if (argument_value == "original") {
        options.algorithm = xmvb::vb::VBSCFAlgorithm::Original;
      } else {
        throw std::invalid_argument("invalid algorithm: " + argument_value);
      }
      continue;
    }
    if (argument_name == "--standard-two-electron-mode") {
      if (argument_value == "auto") {
        options.standard_two_electron_mode = xmvb::vb::StandardTwoElectronMode::Auto;
      } else if (argument_value == "exact") {
        options.standard_two_electron_mode = xmvb::vb::StandardTwoElectronMode::Exact;
      } else if (argument_value == "ri") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
      } else {
        throw std::invalid_argument(
            "invalid standard two-electron mode: " + argument_value);
      }
      continue;
    }
    if (argument_name == "--repeat") {
      options.repeat = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--warmup") {
      options.warmup = std::stoi(argument_value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument_name);
  }

  if (options.repeat <= 0) {
    throw std::invalid_argument("--repeat must be positive");
  }
  if (options.warmup < 0) {
    throw std::invalid_argument("--warmup must be non-negative");
  }
  return options;
}

void print_summary(
    const Options& options,
    xmvb::vb::StandardTwoElectronMode resolved_mode,
    double reference_energy,
    double mean_total_dt,
    double min_total_dt,
    double max_total_dt,
    double mean_active_space_dt,
    double mean_ao_h1e_dt,
    double mean_active_2e_dt,
    double mean_structure_dt,
    double mean_active_adjoint_dt,
    double mean_active_2e_backprop_dt,
    double mean_ao_h1e_backprop_dt,
    double loop_wall_time_seconds) {
  std::cout << std::setprecision(12);
  std::cout << "algorithm = "
            << xmvb::vb::vb_scf_algorithm_name(options.algorithm)
            << '\n';
  std::cout << "standard_two_electron_mode = "
            << xmvb::vb::standard_two_electron_mode_name(resolved_mode)
            << '\n';
  std::cout << "repeat = " << options.repeat << '\n';
  std::cout << "warmup = " << options.warmup << '\n';
  std::cout << "reference_energy = " << reference_energy << '\n';
  std::cout << "mean_total_dt = " << mean_total_dt << '\n';
  std::cout << "min_total_dt = " << min_total_dt << '\n';
  std::cout << "max_total_dt = " << max_total_dt << '\n';
  std::cout << "mean_active_space_dt = " << mean_active_space_dt << '\n';
  std::cout << "mean_ao_h1e_dt = " << mean_ao_h1e_dt << '\n';
  std::cout << "mean_active_2e_dt = " << mean_active_2e_dt << '\n';
  std::cout << "mean_structure_dt = " << mean_structure_dt << '\n';
  std::cout << "mean_active_adjoint_dt = " << mean_active_adjoint_dt << '\n';
  std::cout << "mean_active_2e_backprop_dt = "
            << mean_active_2e_backprop_dt << '\n';
  std::cout << "mean_ao_h1e_backprop_dt = "
            << mean_ao_h1e_backprop_dt << '\n';
  std::cout << "loop_wall_time_seconds = "
            << loop_wall_time_seconds << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.standard_two_electron_mode = options.standard_two_electron_mode;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(options.input_path, load_options);

    xmvb::vb::CppOrbitalGradientEvaluator evaluator(options.algorithm);

    // Keep all repeats in the same process so RI-cache construction and process
    // startup noise are not repeatedly mixed into the timing comparison.
    for (int warmup_index = 0; warmup_index < options.warmup; ++warmup_index) {
      evaluator.evaluate(load_result.input, load_result.nuclear_repulsion_energy);
    }

    double reference_energy = 0.0;
    double sum_total_dt = 0.0;
    double min_total_dt = std::numeric_limits<double>::infinity();
    double max_total_dt = 0.0;
    double sum_active_space_dt = 0.0;
    double sum_ao_h1e_dt = 0.0;
    double sum_active_2e_dt = 0.0;
    double sum_structure_dt = 0.0;
    double sum_active_adjoint_dt = 0.0;
    double sum_active_2e_backprop_dt = 0.0;
    double sum_ao_h1e_backprop_dt = 0.0;

    const auto loop_start_time = std::chrono::steady_clock::now();
    for (int repeat_index = 0; repeat_index < options.repeat; ++repeat_index) {
      const auto result =
          evaluator.evaluate(load_result.input, load_result.nuclear_repulsion_energy);
      reference_energy = result.scf_result.total_energy;
      sum_total_dt += result.total_wall_time_seconds;
      min_total_dt = std::min(min_total_dt, result.total_wall_time_seconds);
      max_total_dt = std::max(max_total_dt, result.total_wall_time_seconds);
      sum_active_space_dt += result.active_space_gradient_wall_time_seconds;
      sum_ao_h1e_dt += result.ao_effective_one_electron_wall_time_seconds;
      sum_active_2e_dt += result.active_two_electron_wall_time_seconds;
      sum_structure_dt += result.structure_matrix_wall_time_seconds;
      sum_active_adjoint_dt += result.active_space_adjoint_wall_time_seconds;
      sum_active_2e_backprop_dt +=
          result.active_space_two_electron_backpropagation_wall_time_seconds;
      sum_ao_h1e_backprop_dt +=
          result.ao_effective_one_electron_backpropagation_wall_time_seconds;
    }
    const double loop_wall_time_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - loop_start_time)
            .count();

    print_summary(
        options,
        load_result.standard_two_electron_mode,
        reference_energy,
        sum_total_dt / options.repeat,
        min_total_dt,
        max_total_dt,
        sum_active_space_dt / options.repeat,
        sum_ao_h1e_dt / options.repeat,
        sum_active_2e_dt / options.repeat,
        sum_structure_dt / options.repeat,
        sum_active_adjoint_dt / options.repeat,
        sum_active_2e_backprop_dt / options.repeat,
        sum_ao_h1e_backprop_dt / options.repeat,
        loop_wall_time_seconds);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
