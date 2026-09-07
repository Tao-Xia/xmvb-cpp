#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/orbital/nonredundant_optimizer_input_adapter.hpp"
#include "vb/orbital/nonredundant_orbital_space.hpp"
#include "vb/orbital/sparse_orbital_parameter_view.hpp"
#include "vb/scf/cpp_orbital_gradient_evaluator.hpp"
#include "vb/scf/cpp_orbital_gradient_result.hpp"
#include "vb/scf/exact_orbital_second_order_operator.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace {

struct Options {
  std::string input_path;
  int repeats = 1;
  int warmup = 0;
  bool nonredundant_adapt = false;
  xmvb::vb::AoIntegralSource ao_integral_source =
      xmvb::vb::AoIntegralSource::Auto;
};

enum class BenchmarkComponent {
  Full,
  CoreOnly,
  OuterOnly,
};

struct AcceptedPointBenchmarkContext {
  xmvb::vb::CppVbInput input;
  std::shared_ptr<xmvb::vb::CppOrbitalGradientResult> gradient_result;
  std::shared_ptr<const xmvb::vb::CppActiveSpaceSecondOrderContext>
      second_order_context;
  xmvb::vb::SparseOrbitalParameterView parameter_view;
  std::unique_ptr<xmvb::vb::NonredundantOrbitalSpace> nonredundant_space;
  Eigen::VectorXd reduced_direction;
};

struct BenchmarkMeasurement {
  BenchmarkComponent component = BenchmarkComponent::Full;
  double external_wall_time_seconds = 0.0;
  double response_inf_norm = 0.0;
  xmvb::vb::ExactOrbitalSecondOrderOperator::Diagnostics diagnostics;
};

struct BlockBenchmarkMeasurement {
  double external_wall_time_seconds = 0.0;
  double response_inf_norm = 0.0;
  xmvb::vb::ExactOrbitalSecondOrderOperator::Diagnostics diagnostics;
};

void print_usage() {
  std::cerr
      << "usage: benchmark_exact_ctx_hvp <input.xmi>"
      << " [--repeats count]"
      << " [--warmup count]"
      << " [--ao-integral-source auto|legacy|libcint_cpp|runtime_hcore]"
      << " [--nonredundant-adapt true|false]\n";
}

bool parse_bool_argument(const std::string& value) {
  if (value == "true" || value == "1") {
    return true;
  }
  if (value == "false" || value == "0") {
    return false;
  }
  throw std::invalid_argument("invalid boolean value: " + value);
}

int parse_positive_or_zero_int(
    const std::string& value,
    const char* option_name) {
  const int parsed = std::stoi(value);
  if (parsed < 0) {
    throw std::invalid_argument(
        std::string(option_name) + " must be non-negative");
  }
  return parsed;
}

xmvb::vb::AoIntegralSource parse_ao_integral_source(
    const std::string& value) {
  if (value == "auto") {
    return xmvb::vb::AoIntegralSource::Auto;
  }
  if (value == "libcint_cpp") {
    return xmvb::vb::AoIntegralSource::LibcintMaterializedCpp;
  }
  if (value == "runtime_hcore") {
    return xmvb::vb::AoIntegralSource::RuntimeCoreHamiltonianOnly;
  }
  throw std::invalid_argument("invalid AO integral source: " + value);
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
      options.repeats = parse_positive_or_zero_int(value, "--repeats");
      continue;
    }
    if (name == "--warmup") {
      options.warmup = parse_positive_or_zero_int(value, "--warmup");
      continue;
    }
    if (name == "--ao-integral-source") {
      options.ao_integral_source = parse_ao_integral_source(value);
      continue;
    }
    if (name == "--nonredundant-adapt") {
      options.nonredundant_adapt = parse_bool_argument(value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + name);
  }
  if (options.repeats <= 0) {
    throw std::invalid_argument("--repeats must be positive");
  }
  return options;
}

const char* bool_name(bool value) {
  return value ? "true" : "false";
}

const char* benchmark_component_name(BenchmarkComponent component) {
  switch (component) {
    case BenchmarkComponent::Full:
      return "full";
    case BenchmarkComponent::CoreOnly:
      return "core_only";
    case BenchmarkComponent::OuterOnly:
      return "outer_only";
  }
  return "unknown";
}

double vector_infinity_norm(const Eigen::VectorXd& vector) {
  if (vector.size() == 0) {
    return 0.0;
  }
  return vector.cwiseAbs().maxCoeff();
}

double average_wall_time_seconds(
    double total_wall_time_seconds,
    std::size_t count) {
  if (count == 0 || !std::isfinite(total_wall_time_seconds) ||
      total_wall_time_seconds < 0.0) {
    return 0.0;
  }
  return total_wall_time_seconds / static_cast<double>(count);
}

Eigen::VectorXd apply_component(
    const xmvb::vb::ExactOrbitalSecondOrderOperator& exact_operator,
    BenchmarkComponent component,
    const Eigen::VectorXd& reduced_direction) {
  switch (component) {
    case BenchmarkComponent::Full:
      return exact_operator.apply_reduced(reduced_direction);
    case BenchmarkComponent::CoreOnly:
      return exact_operator.apply_reduced(
          reduced_direction,
          {.direct_core_response = true,
           .fixed_upstream_pullback = true,
           .outer_response = false});
    case BenchmarkComponent::OuterOnly:
      return exact_operator.apply_reduced(
          reduced_direction,
          {.direct_core_response = false,
           .fixed_upstream_pullback = false,
           .outer_response = true});
  }
  throw std::invalid_argument("unsupported exact_ctx benchmark component");
}

AcceptedPointBenchmarkContext build_benchmark_context(
    const Options& options,
    xmvb::vb::AoIntegralSource* loaded_ao_integral_source) {
  xmvb::vb::CppVbInputLoadOptions load_options;
  load_options.ao_integral_source = options.ao_integral_source;
  load_options.standard_two_electron_mode =
      xmvb::vb::StandardTwoElectronMode::Exact;
  const auto load_result =
      xmvb::vb::load_cpp_vb_input_with_timings(options.input_path, load_options);
  if (loaded_ao_integral_source != nullptr) {
    *loaded_ao_integral_source = load_result.ao_integral_source;
  }

  AcceptedPointBenchmarkContext context{
      options.nonredundant_adapt
          ? xmvb::vb::build_nonredundant_optimizer_input(load_result.input)
          : load_result.input,
      nullptr,
      nullptr,
      xmvb::vb::SparseOrbitalParameterView(load_result.input.orbital_preparation_input),
      nullptr,
      Eigen::VectorXd()};
  context.parameter_view =
      xmvb::vb::SparseOrbitalParameterView(context.input.orbital_preparation_input);

  xmvb::vb::CppOrbitalGradientEvaluator evaluator(
      xmvb::vb::VBSCFAlgorithm::Original);
  context.gradient_result =
      std::make_shared<xmvb::vb::CppOrbitalGradientResult>(
          evaluator.evaluate_without_reference_energy_gradient(
          context.input,
          {0},
          {1.0},
          load_result.nuclear_repulsion_energy));
  context.second_order_context = context.gradient_result->second_order_context;
  if (context.second_order_context == nullptr) {
    throw std::runtime_error("accepted-point second-order context is unavailable");
  }

  const Eigen::VectorXd packed_gradient =
      context.parameter_view.gather_from_full(
          context.gradient_result->sparse_orbital_energy_gradient);
  const int n_inactive_doubly_occupied_orbitals =
      (context.input.orbital_preparation_input.n_total_electrons -
       context.input.orbital_preparation_input.n_active_electrons) /
      2;
  const int n_occupied_orbitals =
      n_inactive_doubly_occupied_orbitals +
      context.input.orbital_preparation_input.n_active_orbitals;
  const Eigen::MatrixXd& normalized_orbital_matrix =
      context.gradient_result->orbital_preparation_result
          .physical_orbital_frame
          .normalized_orbital_matrix;
  if (normalized_orbital_matrix.size() == 0) {
    throw std::runtime_error(
        "exact_ctx benchmark requires the cached physical orbital frame");
  }

  // The reduced benchmark direction lives in the nonredundant tangent chart
  // used by TNHVP.  We use the projected accepted-point gradient direction so
  // every component is probed on the same physically relevant step vector.
  context.nonredundant_space =
      std::make_unique<xmvb::vb::NonredundantOrbitalSpace>(
          context.input.orbital_preparation_input,
          context.parameter_view,
          context.gradient_result->orbital_preparation_result
              .auxiliary_orbital_matrix
              .leftCols(n_occupied_orbitals),
          normalized_orbital_matrix,
          &context.gradient_result->ao_effective_one_electron_result
               .ao_effective_h1e,
          true);
  context.reduced_direction =
      context.nonredundant_space->project_reduced_gradient(packed_gradient);
  if (context.reduced_direction.size() == 0) {
    throw std::runtime_error("nonredundant space is empty");
  }
  if (!(context.reduced_direction.norm() > 0.0)) {
    context.reduced_direction =
        Eigen::VectorXd::Zero(context.reduced_direction.size());
    context.reduced_direction[0] = 1.0;
  } else {
    context.reduced_direction /= context.reduced_direction.norm();
  }
  return context;
}

BenchmarkMeasurement run_component_benchmark(
    const AcceptedPointBenchmarkContext& context,
    BenchmarkComponent component,
    int warmup_count,
    int repeat_count) {
  if (warmup_count > 0) {
    xmvb::vb::ExactOrbitalSecondOrderOperator warmup_operator(
        context.second_order_context,
        &context.input,
        context.parameter_view,
        context.nonredundant_space.get());
    if (!warmup_operator.supports_analytic_core_model()) {
      throw std::runtime_error("exact_ctx analytic core model is unavailable");
    }
    for (int repeat = 0; repeat < warmup_count; ++repeat) {
      (void) apply_component(
          warmup_operator,
          component,
          context.reduced_direction);
    }
  }

  xmvb::vb::ExactOrbitalSecondOrderOperator exact_operator(
      context.second_order_context,
      &context.input,
      context.parameter_view,
      context.nonredundant_space.get());
  if (!exact_operator.supports_analytic_core_model()) {
    throw std::runtime_error("exact_ctx analytic core model is unavailable");
  }

  Eigen::VectorXd response;
  const auto start_time = std::chrono::steady_clock::now();
  for (int repeat = 0; repeat < repeat_count; ++repeat) {
    response =
        apply_component(
            exact_operator,
            component,
            context.reduced_direction);
  }
  const auto stop_time = std::chrono::steady_clock::now();

  BenchmarkMeasurement measurement;
  measurement.component = component;
  measurement.external_wall_time_seconds =
      std::chrono::duration<double>(stop_time - start_time).count();
  measurement.response_inf_norm = vector_infinity_norm(response);
  measurement.diagnostics = exact_operator.diagnostics();
  return measurement;
}

BlockBenchmarkMeasurement run_full_block_benchmark(
    const AcceptedPointBenchmarkContext& context,
    int warmup_count,
    int repeat_count) {
  Eigen::MatrixXd directions(context.reduced_direction.size(), 2);
  directions.col(0) = context.reduced_direction;
  directions.col(1) = context.reduced_direction.reverse();
  directions.col(1).noalias() -=
      directions.col(0) * directions.col(0).dot(directions.col(1));
  if (!(directions.col(1).norm() > 0.0)) {
    directions.col(1).setZero();
    directions(0, 1) = 1.0;
  }
  directions.col(1).normalize();

  if (warmup_count > 0) {
    xmvb::vb::ExactOrbitalSecondOrderOperator warmup_operator(
        context.second_order_context,
        &context.input,
        context.parameter_view,
        context.nonredundant_space.get());
    for (int repeat = 0; repeat < warmup_count; ++repeat) {
      (void) warmup_operator.apply_reduced_batch(directions);
    }
  }
  xmvb::vb::ExactOrbitalSecondOrderOperator exact_operator(
      context.second_order_context,
      &context.input,
      context.parameter_view,
      context.nonredundant_space.get());
  const auto start_time = std::chrono::steady_clock::now();
  Eigen::MatrixXd response;
  for (int repeat = 0; repeat < repeat_count; ++repeat) {
    response = exact_operator.apply_reduced_batch(directions);
  }
  const auto stop_time = std::chrono::steady_clock::now();
  BlockBenchmarkMeasurement measurement;
  measurement.external_wall_time_seconds =
      std::chrono::duration<double>(stop_time - start_time).count();
  measurement.response_inf_norm = response.cwiseAbs().maxCoeff();
  measurement.diagnostics = exact_operator.diagnostics();
  return measurement;
}

void print_measurement(const BenchmarkMeasurement& measurement) {
  const char* label = benchmark_component_name(measurement.component);
  const auto& diagnostics = measurement.diagnostics;
  std::cout << label << "_apply_count = "
            << diagnostics.apply_count << '\n';
  std::cout << label << "_response_inf_norm = "
            << measurement.response_inf_norm << '\n';
  std::cout << label << "_external_avg_wall_time_seconds = "
            << average_wall_time_seconds(
                   measurement.external_wall_time_seconds,
                   diagnostics.apply_count)
            << '\n';
  std::cout << label << "_diag_avg_apply_wall_time_seconds = "
            << average_wall_time_seconds(
                   diagnostics.total_apply_wall_time_seconds,
                   diagnostics.apply_count)
            << '\n';
  std::cout << label << "_diag_avg_core_setup_wall_time_seconds = "
            << average_wall_time_seconds(
                   diagnostics.core_setup_wall_time_seconds,
                   diagnostics.apply_count)
            << '\n';
  std::cout << label << "_diag_avg_h1e_fused_wall_time_seconds = "
            << average_wall_time_seconds(
                   diagnostics.ao_effective_one_electron_fused_wall_time_seconds,
                   diagnostics.apply_count)
            << '\n';
  std::cout << label << "_diag_avg_active_2e_wall_time_seconds = "
            << average_wall_time_seconds(
                   diagnostics.active_two_electron_wall_time_seconds,
                   diagnostics.apply_count)
            << '\n';
  std::cout << label << "_diag_avg_orbital_backprop_wall_time_seconds = "
            << average_wall_time_seconds(
                   diagnostics.orbital_backprop_wall_time_seconds,
                   diagnostics.apply_count)
            << '\n';
  std::cout << label << "_diag_avg_fixed_upstream_wall_time_seconds = "
            << average_wall_time_seconds(
                   diagnostics.fixed_upstream_pullback_wall_time_seconds,
                   diagnostics.apply_count)
            << '\n';
  std::cout << label << "_diag_avg_outer_response_wall_time_seconds = "
            << average_wall_time_seconds(
                   diagnostics.outer_response_wall_time_seconds,
                   diagnostics.apply_count)
            << '\n';
  std::cout << label
            << "_diag_avg_outer_response_active_space_integrals_wall_time_seconds = "
            << average_wall_time_seconds(
                   diagnostics.outer_response_active_space_integrals_wall_time_seconds,
                   diagnostics.apply_count)
            << '\n';
  std::cout << label
            << "_diag_avg_outer_response_structure_matrices_wall_time_seconds = "
            << average_wall_time_seconds(
                   diagnostics.outer_response_structure_matrices_wall_time_seconds,
                   diagnostics.apply_count)
            << '\n';
  std::cout << label
            << "_diag_avg_outer_response_eigensystem_wall_time_seconds = "
            << average_wall_time_seconds(
                   diagnostics.outer_response_eigensystem_wall_time_seconds,
                   diagnostics.apply_count)
            << '\n';
  std::cout << label
            << "_diag_avg_outer_response_pair_weights_wall_time_seconds = "
            << average_wall_time_seconds(
                   diagnostics.outer_response_pair_weights_wall_time_seconds,
                   diagnostics.apply_count)
            << '\n';
  std::cout << label
            << "_diag_avg_outer_response_active_gradient_wall_time_seconds = "
            << average_wall_time_seconds(
                   diagnostics.outer_response_active_gradient_wall_time_seconds,
                   diagnostics.apply_count)
            << '\n';
  std::cout << label
            << "_diag_avg_outer_response_orbital_pullback_wall_time_seconds = "
            << average_wall_time_seconds(
                   diagnostics.outer_response_orbital_pullback_wall_time_seconds,
                   diagnostics.apply_count)
            << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    xmvb::vb::AoIntegralSource loaded_ao_integral_source =
        xmvb::vb::AoIntegralSource::Auto;
    const AcceptedPointBenchmarkContext context =
        build_benchmark_context(options, &loaded_ao_integral_source);

    std::vector<BenchmarkComponent> components = {
        BenchmarkComponent::Full,
        BenchmarkComponent::CoreOnly,
        BenchmarkComponent::OuterOnly};

    std::vector<BenchmarkMeasurement> measurements;
    measurements.reserve(components.size());
    for (const BenchmarkComponent component : components) {
      measurements.push_back(
          run_component_benchmark(
              context,
              component,
              options.warmup,
              options.repeats));
    }
    const BlockBenchmarkMeasurement block_measurement =
        run_full_block_benchmark(context, options.warmup, options.repeats);

    const auto& first_diagnostics = measurements.front().diagnostics;
    std::cout << std::setprecision(12);
    std::cout << "input = " << options.input_path << '\n';
    std::cout << "ao_integral_source = "
              << xmvb::vb::ao_integral_source_name(loaded_ao_integral_source)
              << '\n';
    std::cout << "reduced_dimension = "
              << context.reduced_direction.size() << '\n';
    const auto space_diagnostics =
        context.nonredundant_space->structural_diagnostics();
    std::cout << "packed_dimension = "
              << space_diagnostics.packed_parameter_size << '\n';
    std::cout << "nros_orbital_count = "
              << space_diagnostics.orbital_count << '\n';
    std::cout << "nros_full_local_rank_orbital_count = "
              << space_diagnostics.full_local_rank_orbital_count << '\n';
    std::cout << "nros_codimension_one_orbital_count = "
              << space_diagnostics.codimension_one_orbital_count << '\n';
    std::cout << "nros_incomplete_local_span_orbital_count = "
              << space_diagnostics.incomplete_local_span_orbital_count << '\n';
    std::cout << "nros_gauge_intersection_orbital_count = "
              << space_diagnostics.gauge_intersection_orbital_count << '\n';
    std::cout << "nros_quotient_dimension_mismatch_orbital_count = "
              << space_diagnostics.quotient_dimension_mismatch_orbital_count << '\n';
    std::cout << "nros_total_gauge_rank = "
              << space_diagnostics.total_gauge_rank << '\n';
    std::cout << "nros_total_expected_quotient_dimension = "
              << space_diagnostics.total_expected_quotient_dimension << '\n';
    std::cout << "nros_min_relative_scaling_residual = "
              << space_diagnostics.minimum_relative_scaling_residual << '\n';
    std::cout << "nros_max_relative_scaling_residual = "
              << space_diagnostics.maximum_relative_scaling_residual << '\n';
    std::cout << "repeats = " << options.repeats << '\n';
    std::cout << "warmup = " << options.warmup << '\n';
    std::cout << "nonredundant_adapt = "
              << bool_name(options.nonredundant_adapt) << '\n';
    std::cout << "supports_analytic_core_model = "
              << bool_name(first_diagnostics.supports_analytic_core_model) << '\n';
    std::cout << "outer_response_runtime_enabled = "
              << bool_name(first_diagnostics.outer_response_enabled) << '\n';

    for (const BenchmarkMeasurement& measurement : measurements) {
      print_measurement(measurement);
    }
    const double block_count = static_cast<double>(options.repeats);
    const double block_average =
        block_measurement.external_wall_time_seconds / block_count;
    constexpr double kBlockWidth = 2.0;
    std::cout << "full_block_width = 2\n";
    std::cout << "full_block_apply_count = "
              << block_measurement.diagnostics.batch_apply_count << '\n';
    std::cout << "full_block_response_inf_norm = "
              << block_measurement.response_inf_norm << '\n';
    std::cout << "full_block_external_avg_wall_time_seconds = "
              << block_average << '\n';
    std::cout << "full_block_external_avg_per_direction_seconds = "
              << block_average / kBlockWidth << '\n';
    std::cout << "full_block_diag_avg_h1e_wall_time_seconds = "
              << average_wall_time_seconds(
                     block_measurement.diagnostics
                         .ao_effective_one_electron_fused_wall_time_seconds,
                     block_measurement.diagnostics.batch_apply_count)
              << '\n';
    std::cout << "full_block_diag_avg_active_space_integrals_wall_time_seconds = "
              << average_wall_time_seconds(
                     block_measurement.diagnostics
                         .outer_response_active_space_integrals_wall_time_seconds,
                     block_measurement.diagnostics.batch_apply_count)
              << '\n';
    const double scalar_full_average =
        measurements.front().external_wall_time_seconds /
        static_cast<double>(options.repeats);
    std::cout << "full_block_speedup_over_two_scalar = "
              << (kBlockWidth * scalar_full_average) / block_average << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
