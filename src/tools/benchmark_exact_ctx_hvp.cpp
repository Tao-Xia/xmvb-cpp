#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <memory>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include "vbscf/diagnostics/orbitals/curvature.hpp"
#include "vbscf/optimization/krylov/orthonormal_basis.hpp"
#include "vbscf/optimization/trust_region/spectral.hpp"
#include "vbscf/diagnostics/hessian/reduced_reference.hpp"

#include "input/loading/loader.hpp"
#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/diagnostics/orbitals/chart_audit.hpp"
#include "vbscf/orbitals/charts/layout.hpp"
#include "vbscf/derivatives/gradient/orbital/evaluator.hpp"
#include "vbscf/derivatives/gradient/orbital/result.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "vbscf/determinants/algebra/cofactor_differential.hpp"

namespace {

enum class BenchmarkComponent {
  Full,
  CoreOnly,
  ActiveRelaxedCore,
  OuterOnly,
  LocalActiveOnly,
  StructureOnly,
};

struct Options {
  std::string input_path;
  int repeats = 1;
  int warmup = 0;
  bool gauge_audit = false;
  int curvature_audit_directions = 0;
  int spectral_audit_dimension = 0;
  double spectral_audit_trust_radius = 0.0;
  int dense_reference_block_width = 0;
  int block_width = 2;
  std::vector<BenchmarkComponent> components = {
      BenchmarkComponent::Full,
      BenchmarkComponent::CoreOnly,
      BenchmarkComponent::OuterOnly};
  bool stream_pair_products = false;
  std::string orbital_value_table_bin_path;
};

struct AcceptedPointBenchmarkContext {
  xmvb::vb::VbScfInput input;
  std::shared_ptr<xmvb::vb::OrbitalGradientResult> gradient_result;
  std::shared_ptr<const xmvb::vb::AcceptedPointContext>
      second_order_context;
  xmvb::vb::SparseParameterLayout parameter_view;
  std::unique_ptr<xmvb::vb::OrbitalChart> nonredundant_space;
  Eigen::VectorXd reduced_direction;
  Eigen::VectorXd reduced_gradient;
  double nuclear_repulsion_energy = 0.0;
  std::vector<int> selected_states;
  std::vector<double> state_weights;
  xmvb::vb::StructureEigensolver structure_eigensolver =
      xmvb::vb::StructureEigensolver::Davidson;
  xmvb::vb::StructureSolveAccuracy structure_solve_accuracy;
};

struct BenchmarkMeasurement {
  BenchmarkComponent component = BenchmarkComponent::Full;
  double external_wall_time_seconds = 0.0;
  double response_inf_norm = 0.0;
  Eigen::VectorXd response;
  xmvb::vb::ExactHvpOperator::Diagnostics diagnostics;
};

struct BlockBenchmarkMeasurement {
  double external_wall_time_seconds = 0.0;
  double response_inf_norm = 0.0;
  double scalar_reference_relative_error = 0.0;
  xmvb::vb::ExactHvpOperator::Diagnostics diagnostics;
};

struct CofactorRepresentationStatistics {
  std::size_t regular_pairs = 0;
  std::size_t interpolated_pairs = 0;
  std::size_t polynomial_pairs = 0;
  std::size_t dynamic_bytes = 0;
  std::array<std::size_t, 9> dangerous_mode_counts{};
  std::array<std::size_t, 6> nullity_counts{};
  std::array<std::size_t, 8> condition_decade_counts{};
};

CofactorRepresentationStatistics collect_cofactor_statistics(
    const xmvb::vb::SameSpinPairCacheContext& cache) {
  CofactorRepresentationStatistics statistics;
  const auto collect = [&statistics](
                           const std::vector<
                               xmvb::vb::SpinDeterminantPairEvaluation>& pairs) {
    for (const auto& pair : pairs) {
      const int nullity = std::clamp(pair.overlap_result.nullity, 0, 5);
      ++statistics.nullity_counts[nullity];
      if (pair.overlap_result.nullity == 0 &&
          pair.overlap_result.inverse_overlap_submatrix.size() != 0) {
        const double matrix_norm = pair.overlap_result.overlap_submatrix
                                       .cwiseAbs()
                                       .rowwise()
                                       .sum()
                                       .maxCoeff();
        const double inverse_norm =
            pair.overlap_result.inverse_overlap_submatrix
                .cwiseAbs()
                .rowwise()
                .sum()
                .maxCoeff();
        const double condition = matrix_norm * inverse_norm;
        int condition_decade = 7;
        if (std::isfinite(condition)) {
          condition_decade = std::clamp(
              static_cast<int>(std::floor(std::log10(
                  std::max(1.0, condition)))) / 2,
              0,
              7);
        }
        ++statistics.condition_decade_counts[condition_decade];
      }
      if (!pair.cofactor_differential) {
        continue;
      }
      if (pair.cofactor_differential->uses_regular_form()) {
        ++statistics.regular_pairs;
      } else if (pair.cofactor_differential->uses_interpolated_form()) {
        ++statistics.interpolated_pairs;
        const int dangerous_modes = std::clamp(
            pair.cofactor_differential->dangerous_mode_count(), 0, 8);
        ++statistics.dangerous_mode_counts[dangerous_modes];
      } else {
        ++statistics.polynomial_pairs;
      }
      statistics.dynamic_bytes +=
          pair.cofactor_differential->dynamic_bytes();
    }
  };
  collect(cache.alpha_pair_cache_ref());
  if (!cache.shares_same_spin_pair_cache_between_spins()) {
    collect(cache.beta_pair_cache_ref());
  }
  return statistics;
}

void print_usage() {
  std::cerr
      << "usage: benchmark_exact_ctx_hvp <input.xmi>"
      << " [--repeats count]"
      << " [--warmup count]"
      << " [--gauge-audit true|false]\n";
  std::cerr << " [--curvature-audit-directions count|0=disabled]\n";
  std::cerr << " [--spectral-audit-dimension count|0=disabled]\n";
  std::cerr << " [--spectral-audit-trust-radius value|0=disabled]\n";
  std::cerr << " [--dense-reference-block-width count|0=disabled]\n";
  std::cerr << " [--block-width count|0=disabled]\n";
  std::cerr
      << " [--components full,core,active-relaxed,outer,local,structure]\n";
  std::cerr << " [--stream-pair-products true|false]\n";
  std::cerr << " [--orbital-value-table-bin path]\n";
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

std::vector<BenchmarkComponent> parse_components(const std::string& value) {
  std::vector<BenchmarkComponent> components;
  std::size_t begin = 0;
  while (begin <= value.size()) {
    const std::size_t end = value.find(',', begin);
    const std::string name = value.substr(begin, end - begin);
    BenchmarkComponent component;
    if (name == "full") {
      component = BenchmarkComponent::Full;
    } else if (name == "core") {
      component = BenchmarkComponent::CoreOnly;
    } else if (name == "active-relaxed") {
      component = BenchmarkComponent::ActiveRelaxedCore;
    } else if (name == "outer") {
      component = BenchmarkComponent::OuterOnly;
    } else if (name == "local") {
      component = BenchmarkComponent::LocalActiveOnly;
    } else if (name == "structure") {
      component = BenchmarkComponent::StructureOnly;
    } else {
      throw std::invalid_argument("unknown benchmark component: " + name);
    }
    if (std::find(components.begin(), components.end(), component) ==
        components.end()) {
      components.push_back(component);
    }
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  if (components.empty()) {
    throw std::invalid_argument("--components must not be empty");
  }
  return components;
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
    if (name == "--gauge-audit") {
      options.gauge_audit = parse_bool_argument(value);
      continue;
    }
    if (name == "--curvature-audit-directions") {
      options.curvature_audit_directions = parse_positive_or_zero_int(value, name.c_str());
      continue;
    }
    if (name == "--spectral-audit-dimension") {
      options.spectral_audit_dimension =
          parse_positive_or_zero_int(value, name.c_str());
      continue;
    }
    if (name == "--spectral-audit-trust-radius") {
      options.spectral_audit_trust_radius = std::stod(value);
      if (options.spectral_audit_trust_radius < 0.0 ||
          !std::isfinite(options.spectral_audit_trust_radius)) {
        throw std::invalid_argument(
            "--spectral-audit-trust-radius must be finite and nonnegative");
      }
      continue;
    }
    if (name == "--dense-reference-block-width") {
      options.dense_reference_block_width =
          parse_positive_or_zero_int(value, name.c_str());
      continue;
    }
    if (name == "--block-width") {
      options.block_width = parse_positive_or_zero_int(value, name.c_str());
      continue;
    }
    if (name == "--components") {
      options.components = parse_components(value);
      continue;
    }
    if (name == "--stream-pair-products") {
      options.stream_pair_products = parse_bool_argument(value);
      continue;
    }
    if (name == "--orbital-value-table-bin") {
      options.orbital_value_table_bin_path = value;
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
    case BenchmarkComponent::ActiveRelaxedCore:
      return "active_relaxed_core";
    case BenchmarkComponent::OuterOnly:
      return "outer_only";
    case BenchmarkComponent::LocalActiveOnly:
      return "local_active_only";
    case BenchmarkComponent::StructureOnly:
      return "structure_only";
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
    const xmvb::vb::ExactHvpOperator& exact_operator,
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
            .local_active_response = false,
            .structure_response = false});
    case BenchmarkComponent::ActiveRelaxedCore:
      return exact_operator.apply_reduced(
          reduced_direction,
          {.direct_core_response = true,
           .fixed_upstream_pullback = true,
           .local_active_response = true,
           .structure_response = false});
    case BenchmarkComponent::OuterOnly:
      return exact_operator.apply_reduced(
          reduced_direction,
           {.direct_core_response = false,
            .fixed_upstream_pullback = false,
            .local_active_response = true,
            .structure_response = true});
    case BenchmarkComponent::LocalActiveOnly:
      return exact_operator.apply_reduced(
          reduced_direction,
          {.direct_core_response = false,
           .fixed_upstream_pullback = false,
           .local_active_response = true,
           .structure_response = false});
    case BenchmarkComponent::StructureOnly:
      return exact_operator.apply_reduced(
          reduced_direction,
          {.direct_core_response = false,
           .fixed_upstream_pullback = false,
           .local_active_response = false,
           .structure_response = true});
  }
  throw std::invalid_argument("unsupported exact_ctx benchmark component");
}

AcceptedPointBenchmarkContext build_benchmark_context(
    const Options& options) {
  const auto load_result =
      xmvb::vb::load_vbscf_input_with_timings(options.input_path);

  AcceptedPointBenchmarkContext context{
      load_result.input,
      nullptr,
      nullptr,
      xmvb::vb::SparseParameterLayout(load_result.input.orbital_preparation_input),
      nullptr,
      Eigen::VectorXd()};
  context.parameter_view =
      xmvb::vb::SparseParameterLayout(context.input.orbital_preparation_input);
  context.nuclear_repulsion_energy = load_result.nuclear_repulsion_energy;
  context.structure_eigensolver = load_result.structure_eigensolver;
  context.selected_states.resize(load_result.state_average_count);
  std::iota(
      context.selected_states.begin(), context.selected_states.end(), 0);
  context.state_weights.assign(context.selected_states.size(), 1.0);
  if (!options.orbital_value_table_bin_path.empty()) {
    auto& values = context.input.orbital_preparation_input.orbital_value_table;
    std::ifstream file(options.orbital_value_table_bin_path, std::ios::binary | std::ios::ate);
    const auto bytes = static_cast<std::streamoff>(values.size() * sizeof(double));
    if (!file || file.tellg() != bytes)
      throw std::runtime_error("orbital-value-table file size does not match input chart");
    file.seekg(0);
    file.read(reinterpret_cast<char*>(values.data()), bytes);
    if (!file || !Eigen::Map<const Eigen::VectorXd>(values.data(), values.size()).allFinite())
      throw std::runtime_error("invalid orbital-value-table file");
  }

  xmvb::vb::OrbitalGradientEvaluator evaluator;
  context.gradient_result =
      std::make_shared<xmvb::vb::OrbitalGradientResult>(
          evaluator.evaluate_without_reference_energy_gradient(
          context.input,
          context.selected_states,
          context.state_weights,
          load_result.nuclear_repulsion_energy,
          context.structure_eigensolver,
          context.structure_solve_accuracy,
          Eigen::MatrixXd()));
  if (options.stream_pair_products &&
      context.gradient_result->second_order_context != nullptr) {
    context.gradient_result->second_order_context->prepared_active_space
        .active_space_two_electron_result.dense_ao_pair_products.resize(0, 0);
  }
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
      std::make_unique<xmvb::vb::OrbitalChart>(
          context.input.orbital_preparation_input,
          context.parameter_view,
          context.gradient_result->orbital_preparation_result
              .auxiliary_orbital_matrix
              .leftCols(n_occupied_orbitals),
          normalized_orbital_matrix,
          &context.gradient_result->ao_effective_one_electron_result
               .ao_effective_h1e,
          true);
  context.reduced_gradient =
      context.nonredundant_space->project_reduced_gradient(packed_gradient);
  context.reduced_direction = context.reduced_gradient;
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

/**
 * @brief Samples the lowest current-point curvature without assembling H.
 *
 * The first direction is the projected gradient. A deterministic independent
 * probe then starts a fully reorthogonalized lowest-Ritz residual iteration.
 * An optional radius evaluates the resulting projected trust-region trial.
 */
void run_spectral_audit(
    const AcceptedPointBenchmarkContext& context,
    int maximum_dimension,
    double trust_radius) {
  if (maximum_dimension <= 0) return;
  using namespace xmvb::vb;
  const auto& space = *context.nonredundant_space;
  ExactHvpOperator exact_operator(
      context.second_order_context,
      &context.input,
      context.parameter_view,
      &space);
  const Eigen::Index dimension = context.reduced_direction.size();
  Eigen::VectorXd independent_probe(dimension);
  for (Eigen::Index row = 0; row < dimension; ++row) {
    const double index = static_cast<double>(row + 1);
    independent_probe[row] =
        std::sin(std::sqrt(2.0) * index) +
        std::cos(std::sqrt(3.0) * index);
  }
  independent_probe.noalias() -=
      context.reduced_direction.dot(independent_probe) *
      context.reduced_direction;
  Eigen::VectorXd candidate = context.reduced_direction;

  std::vector<Eigen::VectorXd> basis;
  std::vector<Eigen::VectorXd> hessian_basis;
  basis.reserve(std::min<Eigen::Index>(maximum_dimension, dimension));
  hessian_basis.reserve(std::min<Eigen::Index>(maximum_dimension, dimension));
  std::cout << std::setprecision(12);
  for (int iteration = 0;
       iteration < std::min<Eigen::Index>(maximum_dimension, dimension);
       ++iteration) {
    Eigen::VectorXd candidate_image;
    if (!append_orthonormal_hvp_direction(
            candidate,
            [&](const Eigen::VectorXd& direction) {
              return exact_operator.apply_reduced(direction);
            },
            &basis,
            &hessian_basis,
            &candidate_image)) {
      break;
    }
    Eigen::MatrixXd q(dimension, basis.size());
    Eigen::MatrixXd hq(dimension, basis.size());
    for (std::size_t column = 0; column < basis.size(); ++column) {
      q.col(static_cast<Eigen::Index>(column)) = basis[column];
      hq.col(static_cast<Eigen::Index>(column)) = hessian_basis[column];
    }
    const Eigen::MatrixXd projected_hessian =
        0.5 * (q.transpose() * hq + hq.transpose() * q);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(
        projected_hessian);
    if (eigensolver.info() != Eigen::Success) {
      throw std::runtime_error("spectral audit eigensolver failed");
    }
    const double lowest_ritz_value = eigensolver.eigenvalues()[0];
    const Eigen::VectorXd lowest_ritz_direction =
        q * eigensolver.eigenvectors().col(0);
    candidate = iteration == 0
        ? independent_probe
        : hq * eigensolver.eigenvectors().col(0) -
              lowest_ritz_value * lowest_ritz_direction;
    std::cout << "spectral_audit_dimension_" << basis.size()
              << "_minimum_ritz = " << lowest_ritz_value << '\n';
    std::cout << "spectral_audit_dimension_" << basis.size()
              << "_ritz_residual = " << candidate.norm() << '\n';
  }
  std::cout << "spectral_audit_hvp_directions = "
            << exact_operator.diagnostics().apply_count << '\n';
  if (!(trust_radius > 0.0) || basis.empty()) return;

  Eigen::MatrixXd q(dimension, basis.size());
  Eigen::MatrixXd hq(dimension, basis.size());
  for (std::size_t column = 0; column < basis.size(); ++column) {
    q.col(static_cast<Eigen::Index>(column)) = basis[column];
    hq.col(static_cast<Eigen::Index>(column)) = hessian_basis[column];
  }
  const Eigen::MatrixXd projected_hessian =
      0.5 * (q.transpose() * hq + hq.transpose() * q);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(
      projected_hessian);
  if (eigensolver.info() != Eigen::Success) {
    throw std::runtime_error("spectral trust-region audit eigensolver failed");
  }
  const Eigen::VectorXd projected_gradient =
      q.transpose() * context.reduced_gradient;
  const Eigen::VectorXd gradient_in_eigenbasis =
      eigensolver.eigenvectors().transpose() * projected_gradient;
  const auto spectral_step = solve_spectral_trust_region(
      eigensolver.eigenvalues(),
      gradient_in_eigenbasis,
      trust_radius);
  const Eigen::VectorXd subspace_coordinates =
      eigensolver.eigenvectors() * spectral_step.step;
  const Eigen::VectorXd reduced_step = q * subspace_coordinates;
  const Eigen::VectorXd hessian_step = hq * subspace_coordinates;
  const double predicted_decrease =
      -context.reduced_gradient.dot(reduced_step) -
      0.5 * reduced_step.dot(hessian_step);
  VbScfInput trial = context.input;
  trial.orbital_preparation_input = space.retract_step(
      context.input.orbital_preparation_input,
      reduced_step);
  OrbitalGradientEvaluator evaluator;
  const auto trial_result =
      evaluator.evaluate_without_reference_energy_gradient(
          trial,
          context.selected_states,
          context.state_weights,
          context.nuclear_repulsion_energy,
          context.structure_eigensolver,
          context.structure_solve_accuracy,
          context.second_order_context->root_eigenvectors);
  const double actual_decrease =
      context.gradient_result->scf_result.total_energy -
      trial_result.scf_result.total_energy;
  std::cout << "spectral_audit_trust_radius = " << trust_radius << '\n';
  std::cout << "spectral_audit_step_norm = " << reduced_step.norm() << '\n';
  std::cout << "spectral_audit_shift = " << spectral_step.shift << '\n';
  std::cout << "spectral_audit_predicted_decrease = "
            << predicted_decrease << '\n';
  std::cout << "spectral_audit_actual_decrease = "
            << actual_decrease << '\n';
  std::cout << "spectral_audit_trust_ratio = "
            << actual_decrease / predicted_decrease << '\n';
}

// A fixed-point linear diagnostic, not an optimizer trajectory: no trust
// boundary, no transported history, and no outer acceptance/stopping changes.
void run_curvature_audit(const AcceptedPointBenchmarkContext& context, int budget) {
  using namespace xmvb::vb;
  const auto& space = *context.nonredundant_space;
  ExactHvpOperator op(context.second_order_context, &context.input,
      context.parameter_view, &space);
  ExactHvpOperator core_op(context.second_order_context, &context.input,
      context.parameter_view, &space);
  ExactHvpOperator outer_op(context.second_order_context, &context.input,
      context.parameter_view, &space);
  ExactHvpOperator local_op(context.second_order_context, &context.input,
      context.parameter_view, &space);
  ExactHvpOperator structure_op(context.second_order_context, &context.input,
      context.parameter_view, &space);
  const Eigen::VectorXd g = space.project_reduced_gradient(
      context.parameter_view.gather_from_full(context.gradient_result->sparse_orbital_energy_gradient));
  auto full = [&](const Eigen::VectorXd& v) { return op.apply_reduced(v); };
  auto core = [&](const Eigen::VectorXd& v) {
    return apply_component(core_op, BenchmarkComponent::CoreOnly, v);
  };
  auto model = [&](const Eigen::VectorXd& v) { return space.apply_reduced_curvature(v); };
  struct Sample {
    Eigen::VectorXd step, residual;
    Eigen::MatrixXd q, hq;
  };
  auto sample = [&](auto&& action) {
    Sample sample;
    sample.step = Eigen::VectorXd::Zero(g.size());
    sample.residual = -g;
    std::vector<Eigen::VectorXd> q, hq;
    for (int j = 0; j < std::min<int>(budget, g.size()); ++j) {
      const Eigen::VectorXd p = space.apply_inverse_reduced_block_preconditioner(sample.residual);
      Eigen::VectorXd hp;
      if (!append_orthonormal_hvp_direction(p, action, &q, &hq, &hp)) break;
      sample.q.resize(g.size(), q.size()); sample.hq.resize(g.size(), q.size());
      for (std::size_t k = 0; k < q.size(); ++k) {
        sample.q.col(k) = q[k]; sample.hq.col(k) = hq[k];
      }
      // Full least squares in evaluated images; valid for nonsymmetric
      // computational-stage components as well as the complete operator.
      const Eigen::VectorXd coefficients = sample.hq.colPivHouseholderQr().solve(-g);
      sample.step = sample.q * coefficients;
      sample.residual = -g - sample.hq * coefficients;
      if (sample.residual.norm() <= std::sqrt(std::numeric_limits<double>::epsilon()) * g.norm()) break;
    }
    sample.q.resize(g.size(), q.size()); sample.hq.resize(g.size(), q.size());
    for (std::size_t j = 0; j < q.size(); ++j) {
      sample.q.col(j) = q[j]; sample.hq.col(j) = hq[j];
    }
    return sample;
  };
  const auto sampled = sample(full);
  const auto core_sampled = sample(core);
  const double gradient_norm = g.norm();
  if (!(gradient_norm > 0.0) || sampled.q.cols() == 0)
    throw std::runtime_error("curvature audit requires a nonzero sampled gradient");
  std::cout << std::setprecision(12);
  std::cout << "audit_gradient_norm = " << gradient_norm << '\n';
  std::cout << "audit_full_directions = " << sampled.q.cols() << '\n';
  std::cout << "audit_core_directions = " << core_sampled.q.cols() << '\n';
  std::cout << "audit_sampler = right_preconditioned_minimum_residual\n";
  std::cout << "audit_full_relative_residual = " << (g+full(sampled.step)).norm()/gradient_norm << '\n';
  std::cout << "audit_core_relative_residual = " << (g+core(core_sampled.step)).norm()/gradient_norm << '\n';
  std::cout << "audit_core_step_full_relative_residual = " << (g+full(core_sampled.step)).norm()/gradient_norm << '\n';

  Eigen::MatrixXd mq(g.size(), sampled.q.cols());
  Eigen::MatrixXd cq(g.size(), sampled.q.cols());
  Eigen::MatrixXd lq(g.size(), sampled.q.cols());
  Eigen::MatrixXd sq(g.size(), sampled.q.cols());
  for (int j = 0; j < sampled.q.cols(); ++j) {
    mq.col(j) = model(sampled.q.col(j));
    cq.col(j) = core(sampled.q.col(j));
    lq.col(j) = apply_component(
        local_op, BenchmarkComponent::LocalActiveOnly, sampled.q.col(j));
    sq.col(j) = apply_component(
        structure_op, BenchmarkComponent::StructureOnly, sampled.q.col(j));
  }
  const Eigen::MatrixXd t = sampled.q.transpose() * sampled.hq;
  const Eigen::MatrixXd c = sampled.q.transpose() * cq;
  const Eigen::MatrixXd l = sampled.q.transpose() * lq;
  const Eigen::MatrixXd s = sampled.q.transpose() * sq;
  const Eigen::MatrixXd m = sampled.q.transpose() * mq;
  const auto symmetric = [](const Eigen::MatrixXd& a) -> Eigen::MatrixXd { return 0.5*(a+a.transpose()); };
  std::cout << "audit_full_relative_skew = " << (t-t.transpose()).norm()/t.norm() << '\n';
  std::cout << "audit_core_relative_skew = " << (c-c.transpose()).norm()/c.norm() << '\n';
  std::cout << "audit_local_active_relative_skew = "
            << (l-l.transpose()).norm()/l.norm() << '\n';
  std::cout << "audit_structure_relative_skew = "
            << (s-s.transpose()).norm()/s.norm() << '\n';
  std::cout << "audit_outer_split_projected_additivity_error = "
            << ((t-c)-(l+s)).norm()/t.norm() << '\n';
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(symmetric(t));
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> relative(symmetric(t), symmetric(m));
  if (eig.info() != Eigen::Success || relative.info() != Eigen::Success)
    throw std::runtime_error("curvature audit eigensolver failed");
  std::cout << "audit_sampled_ritz_min = " << eig.eigenvalues()[0] << '\n';
  std::cout << "audit_sampled_ritz_max = " << eig.eigenvalues().tail(1)[0] << '\n';
  std::cout << "audit_sampled_generalized_min = " << relative.eigenvalues()[0] << '\n';
  std::cout << "audit_sampled_generalized_max = " << relative.eigenvalues().tail(1)[0] << '\n';

  auto finite_difference = [&](const Eigen::VectorXd& v, double step) -> Eigen::VectorXd {
    OrbitalGradientEvaluator evaluator;
    auto plus = context.input, minus = context.input;
    plus.orbital_preparation_input = space.retract_step(context.input.orbital_preparation_input, v, step);
    minus.orbital_preparation_input = space.retract_step(context.input.orbital_preparation_input, v, -step);
    const auto gp = evaluator.evaluate_without_reference_energy_gradient(
        plus,
        context.selected_states,
        context.state_weights,
        context.nuclear_repulsion_energy,
        context.structure_eigensolver,
        context.structure_solve_accuracy,
        context.second_order_context->root_eigenvectors);
    const auto gm = evaluator.evaluate_without_reference_energy_gradient(
        minus,
        context.selected_states,
        context.state_weights,
        context.nuclear_repulsion_energy,
        context.structure_eigensolver,
        context.structure_solve_accuracy,
        context.second_order_context->root_eigenvectors);
    return space.project_reduced_gradient(context.parameter_view.gather_from_full(gp.sparse_orbital_energy_gradient) -
        context.parameter_view.gather_from_full(gm.sparse_orbital_energy_gradient)) / (2.0 * step);
  };
  const Eigen::VectorXd soft = sampled.q * eig.eigenvectors().col(0);
  const Eigen::VectorXd hsoft = full(soft);
  std::cout << "audit_soft_linearity_error = " <<
      (hsoft-sampled.hq*eig.eigenvectors().col(0)).norm()/hsoft.norm() << '\n';
  const Eigen::VectorXd csoft = core(soft);
  const Eigen::VectorXd osoft = apply_component(outer_op, BenchmarkComponent::OuterOnly, soft);
  const Eigen::VectorXd lsoft = apply_component(
      local_op, BenchmarkComponent::LocalActiveOnly, soft);
  const Eigen::VectorXd ssoft = apply_component(
      structure_op, BenchmarkComponent::StructureOnly, soft);
  std::cout << "audit_soft_core_linearity_error_over_full = " <<
      (csoft-cq*eig.eigenvectors().col(0)).norm()/hsoft.norm() << '\n';
  std::cout << "audit_soft_outer_linearity_error_over_full = " <<
      (osoft-(sampled.hq-cq)*eig.eigenvectors().col(0)).norm()/hsoft.norm() << '\n';
  std::cout << "audit_soft_outer_split_additivity_error = " <<
      (osoft-lsoft-ssoft).norm()/hsoft.norm() << '\n';
  std::cout << "audit_soft_local_active_ratio = " <<
      lsoft.norm()/hsoft.norm() << '\n';
  std::cout << "audit_soft_structure_ratio = " <<
      ssoft.norm()/hsoft.norm() << '\n';
  std::cout << "audit_soft_local_active_rayleigh = " << soft.dot(lsoft) << '\n';
  std::cout << "audit_soft_structure_rayleigh = " << soft.dot(ssoft) << '\n';
  for (double scale : {-1.0, 2.0}) {
    std::cout << "audit_soft_homogeneity_error_scale" << scale << " = " <<
        (full(scale*soft)-scale*hsoft).norm()/(std::abs(scale)*hsoft.norm()) << '\n';
  }
  for (double step : {1e-4, 1e-5, 1e-6}) {
    std::cout << "audit_soft_fd_relative_error_h" << step << " = "
              << (finite_difference(soft, step)-hsoft).norm()/hsoft.norm() << '\n';
  }
  Eigen::Index skew_i, skew_j;
  (t-t.transpose()).cwiseAbs().maxCoeff(&skew_i, &skew_j);
  std::cout << "audit_max_bilinear_skew = " << t(skew_i,skew_j)-t(skew_j,skew_i) << '\n';
  const Eigen::VectorXd fresh_skew_image = full(sampled.q.col(skew_j));
  std::cout << "audit_repeatability_error = " <<
      (fresh_skew_image-sampled.hq.col(skew_j)).norm()/fresh_skew_image.norm() << '\n';
  if ((t-t.transpose()).norm() > std::sqrt(std::numeric_limits<double>::epsilon()) * t.norm()) {
    const Eigen::VectorXd a = sampled.q.col(skew_i), b = sampled.q.col(skew_j);
    for (double step : {1e-4, 1e-5, 1e-6}) {
      const Eigen::VectorXd fda = finite_difference(a, step), fdb = finite_difference(b, step);
      std::cout << "audit_fd_bilinear_skew_h" << step << " = " << a.dot(fdb)-b.dot(fda) << '\n';
      std::cout << "audit_skew_direction_fd_relative_error_h" << step << " = "
                << (fdb-sampled.hq.col(skew_j)).norm()/sampled.hq.col(skew_j).norm() << '\n';
    }
  }

  // U is target-block diagonal. Mask packed coefficients, then project back;
  // no full reduced basis or full Hessian is assembled for this decomposition.
  const auto& input = context.input.orbital_preparation_input;
  const auto& slots = context.parameter_view.differentiable_parameter_indices();
  auto project = [&](const Eigen::VectorXd& v, int p) {
    Eigen::VectorXd packed = space.expand_step(v);
    for (int j = 0; j < packed.size(); ++j)
      if (slots[j] / input.n_basis_functions != p) packed[j] = 0.0;
    const Eigen::VectorXd resolved =
        space.project_vector(packed).reduced_gradient;
    if ((space.expand_step(resolved) - packed).norm() >
        1.0e-10 * std::max(1.0, packed.norm())) {
      throw std::runtime_error("orbital projectors do not resolve the audit vector");
    }
    return resolved;
  };
  const std::vector<std::pair<std::string, Eigen::VectorXd>> probes = {
      {"gradient", g}, {"newton_step", sampled.step},
      {"residual", sampled.residual},
      {"soft_ritz", sampled.q * eig.eigenvectors().col(0)}};
  for (const auto& [label, vector] : probes) {
    if (!(vector.norm() > 0.0)) continue;
    const Eigen::VectorXd v = vector.normalized();
    const auto d = xmvb::diagnostics::decompose_curvature(
        v, input.n_orbitals, full, core, model, project);
    const double scale = d.full.norm();
    if (!(scale > 0.0)) throw std::runtime_error("zero audit HVP norm");
    const auto outer_direct = apply_component(outer_op, BenchmarkComponent::OuterOnly, v);
    const double additivity = (d.full-d.core-outer_direct).norm()/scale;
    if (additivity > 1e-8) throw std::runtime_error("HVP component additivity failed");
    const std::string prefix = "audit_" + label + "_";
    std::cout << prefix << "additivity_error = " << additivity << '\n';
    std::cout << prefix << "full_norm = " << scale << '\n';
    std::cout << prefix << "local_error_ratio = " << d.local_error.norm()/scale << '\n';
    std::cout << prefix << "coupling_ratio = " << d.coupling.norm()/scale << '\n';
    std::cout << prefix << "outer_ratio = " << d.outer.norm()/scale << '\n';
    std::cout << prefix << "full_rayleigh = " << v.dot(d.full) << '\n';
    std::cout << prefix << "model_rayleigh = " << v.dot(d.model) << '\n';
    std::cout << prefix << "diagonal_core_rayleigh = " << v.dot(d.diagonal_core) << '\n';
    std::cout << prefix << "coupling_rayleigh = " << v.dot(d.coupling) << '\n';
    std::cout << prefix << "outer_rayleigh = " << v.dot(d.outer) << '\n';
  }
  const auto audit_timing = op.diagnostics();
  const double audit_apply_count = static_cast<double>(
      std::max<std::size_t>(1, audit_timing.apply_count));
  std::cout << "audit_full_apply_count = " << audit_timing.apply_count << '\n';
  std::cout << "audit_full_avg_apply_seconds = "
            << audit_timing.total_apply_wall_time_seconds / audit_apply_count
            << '\n';
  std::cout << "audit_full_avg_outer_seconds = "
            << audit_timing.outer_response_wall_time_seconds /
                   audit_apply_count
            << '\n';
  std::cout << "audit_full_avg_structure_matrix_seconds = "
            << audit_timing.outer_response_structure_matrices_wall_time_seconds /
                   audit_apply_count
            << '\n';
  std::cout << "audit_full_avg_eigensystem_seconds = "
            << audit_timing.outer_response_eigensystem_wall_time_seconds /
                   audit_apply_count
            << '\n';
  std::cout << "audit_full_avg_active_gradient_seconds = "
            << audit_timing.outer_response_active_gradient_wall_time_seconds /
                   audit_apply_count
            << '\n';
  std::cout << "audit_full_avg_orbital_pullback_seconds = "
            << audit_timing.outer_response_orbital_pullback_wall_time_seconds /
                   audit_apply_count
            << '\n';
  std::cout << "audit_full_max_structure_response_iterations = "
            << audit_timing.max_structure_response_iterations << '\n';
  std::cout << "audit_total_hvp_calls = " << op.diagnostics().apply_count +
      core_op.diagnostics().apply_count + outer_op.diagnostics().apply_count << '\n';
}

BenchmarkMeasurement run_component_benchmark(
    const AcceptedPointBenchmarkContext& context,
    BenchmarkComponent component,
    int warmup_count,
    int repeat_count) {
  if (warmup_count > 0) {
    xmvb::vb::ExactHvpOperator warmup_operator(
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

  xmvb::vb::ExactHvpOperator exact_operator(
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
  measurement.response = std::move(response);
  measurement.diagnostics = exact_operator.diagnostics();
  return measurement;
}

BlockBenchmarkMeasurement run_full_block_benchmark(
    const AcceptedPointBenchmarkContext& context,
    int warmup_count,
    int repeat_count,
    int block_width) {
  const Eigen::Index dimension = context.reduced_direction.size();
  const Eigen::Index width = std::min<Eigen::Index>(block_width, dimension);
  Eigen::MatrixXd candidates(dimension, width);
  candidates.col(0) = context.reduced_direction;
  const double pi = std::acos(-1.0);
  for (Eigen::Index column = 1; column < width; ++column) {
    for (Eigen::Index row = 0; row < dimension; ++row) {
      candidates(row, column) =
          std::cos(pi * (row + 0.5) * column / dimension);
    }
  }
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(candidates);
  const Eigen::MatrixXd directions =
      qr.householderQ() * Eigen::MatrixXd::Identity(dimension, width);

  if (warmup_count > 0) {
    xmvb::vb::ExactHvpOperator warmup_operator(
        context.second_order_context,
        &context.input,
        context.parameter_view,
        context.nonredundant_space.get());
    for (int repeat = 0; repeat < warmup_count; ++repeat) {
      (void) warmup_operator.apply_reduced_batch(directions);
    }
  }
  xmvb::vb::ExactHvpOperator exact_operator(
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
  xmvb::vb::ExactHvpOperator scalar_reference_operator(
      context.second_order_context,
      &context.input,
      context.parameter_view,
      context.nonredundant_space.get());
  Eigen::MatrixXd scalar_reference(response.rows(), response.cols());
  for (Eigen::Index column = 0; column < width; ++column) {
    scalar_reference.col(column) =
        scalar_reference_operator.apply_reduced(directions.col(column));
  }
  const auto scalar_reference_diagnostics =
      scalar_reference_operator.diagnostics();
  measurement.scalar_reference_relative_error =
      (response - scalar_reference).norm() /
      std::max(1.0, scalar_reference.norm());
  // Recycled and simultaneous response solves can terminate at different
  // points inside the same certified inexact-response ball. Their HVPs need
  // agree to that declared backward error, not to a tighter hidden threshold.
  const double block_reproducibility_tolerance = std::max({
      8.0 * std::sqrt(std::numeric_limits<double>::epsilon()),
      measurement.diagnostics.max_structure_response_relative_residual,
      scalar_reference_diagnostics.max_structure_response_relative_residual});
  if (measurement.scalar_reference_relative_error >
      block_reproducibility_tolerance) {
    std::ostringstream message;
    message << std::scientific
            << "block HVP differs from independent scalar actions: relative_error="
            << measurement.scalar_reference_relative_error;
    throw std::runtime_error(message.str());
  }
  return measurement;
}

void run_dense_reduced_hessian_reference(
    const AcceptedPointBenchmarkContext& context,
    int block_width) {
  if (block_width <= 0) return;
  xmvb::vb::ExactHvpOperator exact_operator(
      context.second_order_context,
      &context.input,
      context.parameter_view,
      context.nonredundant_space.get());
  const Eigen::Index dimension = context.reduced_direction.size();
  const auto start_time = std::chrono::steady_clock::now();
  const auto reference = xmvb::vb::assemble_reduced_hessian_reference(
      dimension,
      block_width,
      [&](const Eigen::Ref<const Eigen::MatrixXd>& directions) {
        return exact_operator.apply_reduced_batch(directions);
      });
  const auto stop_time = std::chrono::steady_clock::now();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(
      reference.symmetric_hessian);
  if (eigensolver.info() != Eigen::Success) {
    throw std::runtime_error("dense reduced-Hessian eigensolver failed");
  }
  const Eigen::VectorXd direct_action =
      exact_operator.apply_reduced(context.reduced_direction);
  const Eigen::VectorXd assembled_action =
      reference.raw_hessian * context.reduced_direction;
  const double action_scale = std::max(1.0, direct_action.norm());
  const double action_relative_error =
      (assembled_action - direct_action).norm() / action_scale;
  const auto diagnostics = exact_operator.diagnostics();
  const double audit_tolerance = std::max(
      std::sqrt(std::numeric_limits<double>::epsilon()),
      diagnostics.max_structure_response_relative_residual);
  std::cout << "dense_reference_relative_skew = "
            << reference.relative_skew_norm << '\n';
  std::cout << "dense_reference_action_relative_error = "
            << action_relative_error << '\n';
  std::cout << "dense_reference_audit_tolerance = "
            << audit_tolerance << '\n';
  std::cout << "dense_reference_minimum_eigenvalue = "
            << eigensolver.eigenvalues().minCoeff() << '\n';
  std::cout << "dense_reference_maximum_eigenvalue = "
            << eigensolver.eigenvalues().maxCoeff() << '\n';
  if (reference.relative_skew_norm > audit_tolerance ||
      action_relative_error > audit_tolerance) {
    throw std::runtime_error(
        "assembled reduced Hessian failed symmetry or action audit");
  }
  std::cout << "dense_reference_dimension = " << dimension << '\n';
  std::cout << "dense_reference_block_width = " << block_width << '\n';
  std::cout << "dense_reference_batch_calls = "
            << exact_operator.diagnostics().batch_apply_count << '\n';
  std::cout << "dense_reference_storage_bytes = "
            << sizeof(double) * dimension * dimension << '\n';
  std::cout << "dense_reference_wall_time_seconds = "
            << std::chrono::duration<double>(stop_time - start_time).count()
            << '\n';
}

void print_measurement(const BenchmarkMeasurement& measurement) {
  const char* label = benchmark_component_name(measurement.component);
  const auto& diagnostics = measurement.diagnostics;
  std::cout << label << "_apply_count = "
            << diagnostics.apply_count << '\n';
  std::cout << label << "_structure_response_block_actions = "
            << diagnostics.structure_response_block_actions << '\n';
  std::cout << label << "_max_structure_response_iterations = "
            << diagnostics.max_structure_response_iterations << '\n';
  std::cout << label << "_max_structure_response_relative_residual = "
            << diagnostics.max_structure_response_relative_residual << '\n';
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
            << "_diag_avg_outer_response_local_active_gradient_wall_time_seconds = "
            << average_wall_time_seconds(
                   diagnostics
                       .outer_response_local_active_gradient_wall_time_seconds,
                   diagnostics.apply_count)
            << '\n';
  std::cout << label
            << "_diag_avg_outer_response_structure_active_gradient_wall_time_seconds = "
            << average_wall_time_seconds(
                   diagnostics
                       .outer_response_structure_active_gradient_wall_time_seconds,
                   diagnostics.apply_count)
            << '\n';
  std::cout << label
            << "_diag_avg_outer_response_selected_state_rebuild_wall_time_seconds = "
            << average_wall_time_seconds(
                   diagnostics
                       .outer_response_selected_state_rebuild_wall_time_seconds,
                   diagnostics.apply_count)
            << '\n';
  std::cout << label
            << "_diag_avg_outer_response_same_spin_backward_wall_time_seconds = "
            << average_wall_time_seconds(
                   diagnostics
                       .outer_response_same_spin_backward_wall_time_seconds,
                   diagnostics.apply_count)
            << '\n';
  std::cout << label
            << "_diag_avg_outer_response_opposite_spin_backward_wall_time_seconds = "
            << average_wall_time_seconds(
                   diagnostics
                       .outer_response_opposite_spin_backward_wall_time_seconds,
                   diagnostics.apply_count)
            << '\n';
  std::cout << label
            << "_diag_avg_outer_response_opposite_spin_packed_gradient_wall_time_seconds = "
            << average_wall_time_seconds(
                   diagnostics
                       .outer_response_opposite_spin_packed_gradient_wall_time_seconds,
                   diagnostics.apply_count)
            << '\n';
  std::cout << label
            << "_diag_avg_outer_response_opposite_spin_alpha_overlap_wall_time_seconds = "
            << average_wall_time_seconds(
                   diagnostics
                       .outer_response_opposite_spin_alpha_overlap_wall_time_seconds,
                   diagnostics.apply_count)
            << '\n';
  std::cout << label
            << "_diag_avg_outer_response_opposite_spin_beta_overlap_wall_time_seconds = "
            << average_wall_time_seconds(
                   diagnostics
                       .outer_response_opposite_spin_beta_overlap_wall_time_seconds,
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
    const AcceptedPointBenchmarkContext context =
        build_benchmark_context(options);
    std::cerr << "accepted_point_context_ready = true\n";

    std::vector<BenchmarkMeasurement> measurements;
    measurements.reserve(options.components.size());
    for (const BenchmarkComponent component : options.components) {
      std::cerr << "benchmark_component_start = "
                << benchmark_component_name(component) << '\n';
      measurements.push_back(run_component_benchmark(
          context,
          component,
          options.warmup,
          options.repeats));
      std::cerr << "benchmark_component_seconds = "
                << measurements.back().external_wall_time_seconds << '\n';
    }
    const auto find_measurement = [&](BenchmarkComponent component)
        -> const BenchmarkMeasurement* {
      const auto iterator = std::find_if(
          measurements.begin(),
          measurements.end(),
          [component](const BenchmarkMeasurement& measurement) {
            return measurement.component == component;
          });
      return iterator == measurements.end() ? nullptr : &*iterator;
    };
    const BenchmarkMeasurement* outer =
        find_measurement(BenchmarkComponent::OuterOnly);
    const BenchmarkMeasurement* local =
        find_measurement(BenchmarkComponent::LocalActiveOnly);
    const BenchmarkMeasurement* structure =
        find_measurement(BenchmarkComponent::StructureOnly);
    if (outer != nullptr && local != nullptr && structure != nullptr) {
      const double scale = std::max(1.0, outer->response.norm());
      std::cout << "outer_component_additivity_relative_error = "
                << (outer->response - local->response - structure->response)
                           .norm() /
                       scale
                << '\n';
    }
    std::optional<BlockBenchmarkMeasurement> block_measurement;
    if (options.block_width > 0) {
      block_measurement = run_full_block_benchmark(
          context, options.warmup, options.repeats, options.block_width);
    }

    const auto& first_diagnostics = measurements.front().diagnostics;
    const auto& ao = context.input.ao_integral_input;
    const auto& pair_graph = ao.pair_graph;
    const auto& active_two_electron =
        context.second_order_context->prepared_active_space
            .active_space_two_electron_result;
    const CofactorRepresentationStatistics cofactor_statistics =
        collect_cofactor_statistics(
            context.second_order_context->same_spin_pair_cache);
    const std::size_t pair_graph_storage_bytes =
        pair_graph.row_offsets.capacity() * sizeof(int) +
        pair_graph.columns.capacity() * sizeof(int) +
        pair_graph.values.capacity() * sizeof(double) +
        pair_graph.integral_rows.capacity() * sizeof(int) +
        pair_graph.integral_edges.capacity() * sizeof(int) +
        pair_graph.pair_first.capacity() * sizeof(int) +
        pair_graph.pair_second.capacity() * sizeof(int);
    const std::size_t packed_active_storage_bytes =
        active_two_electron.packed_active_two_electron_integrals.capacity() *
        sizeof(double);
    const std::size_t active_coefficient_storage_bytes =
        static_cast<std::size_t>(
            active_two_electron.dense_active_coefficients.size()) *
        sizeof(double);
    std::cout << std::setprecision(12);
    std::cout << "input = " << options.input_path << '\n';
    std::cout << "orbital_value_table_override = " << options.orbital_value_table_bin_path << '\n';
    std::cout << "reduced_dimension = "
              << context.reduced_direction.size() << '\n';
    std::cout << "state_average_count = "
              << context.selected_states.size() << '\n';
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
    if (options.gauge_audit) {
      Eigen::MatrixXd current_packed_reduced_basis(
          context.parameter_view.size(),
          context.nonredundant_space->reduced_size());
      for (int column = 0;
           column < context.nonredundant_space->reduced_size();
           ++column) {
        Eigen::VectorXd unit = Eigen::VectorXd::Zero(
            context.nonredundant_space->reduced_size());
        unit[column] = 1.0;
        current_packed_reduced_basis.col(column) =
            context.nonredundant_space->expand_step(unit);
      }
      const auto gauge_audit = xmvb::vb::audit_orbital_chart(
          context.input.orbital_preparation_input,
          context.parameter_view,
          &current_packed_reduced_basis);
      const Eigen::VectorXd packed_gradient = context.parameter_view.gather_from_full(
          context.gradient_result->sparse_orbital_energy_gradient);
      std::cout << "gauge_audit_gradient_overlap = "
                << (gauge_audit.packed_gauge_basis.transpose() * packed_gradient).norm() /
                       std::max(1.0, packed_gradient.norm()) << '\n';
      std::cout << "gauge_audit_parameter_dimension = "
                << gauge_audit.gauge_parameter_dimension << '\n';
      std::cout << "gauge_audit_support_constraint_rank = "
                << gauge_audit.support_constraint_rank << '\n';
      std::cout << "gauge_audit_admissible_parameter_dimension = "
                << gauge_audit.admissible_gauge_parameter_dimension << '\n';
      std::cout << "gauge_audit_gauge_rank = "
                << gauge_audit.gauge_rank << '\n';
      std::cout << "gauge_audit_quotient_dimension = "
                << gauge_audit.quotient_dimension << '\n';
      std::cout << "gauge_audit_physical_jacobian_rank = "
                << gauge_audit.physical_jacobian_rank << '\n';
      std::cout << "gauge_audit_physical_jacobian_nullity = "
                << gauge_audit.physical_jacobian_nullity << '\n';
      std::cout << "gauge_audit_unmapped_parameter_count = "
                << gauge_audit.unmapped_parameter_count << '\n';
      std::cout << "gauge_audit_relative_annihilation_residual = "
                << gauge_audit.relative_gauge_annihilation_residual << '\n';
      std::cout << "gauge_audit_max_principal_angle_sine = "
                << gauge_audit.maximum_gauge_kernel_principal_angle_sine << '\n';
      std::cout << "gauge_audit_current_reduced_dimension = "
                << gauge_audit.current_reduced_dimension << '\n';
      std::cout << "gauge_audit_current_physical_image_rank = "
                << gauge_audit.current_physical_image_rank << '\n';
      std::cout << "gauge_audit_current_retained_gauge_dimension = "
                << gauge_audit.current_retained_gauge_dimension << '\n';
      std::cout << "gauge_audit_current_missing_physical_dimension = "
                << gauge_audit.current_missing_physical_dimension << '\n';
    }
    std::cout << "repeats = " << options.repeats << '\n';
    std::cout << "warmup = " << options.warmup << '\n';
    const auto& selected_states =
        context.second_order_context->selected_state_matrices;
    std::cout << "n_unique_alpha = " << selected_states.n_unique_alpha << '\n';
    std::cout << "n_unique_beta = " << selected_states.n_unique_beta << '\n';
    std::cout << "cofactor_regular_pair_count = "
              << cofactor_statistics.regular_pairs << '\n';
    std::cout << "cofactor_interpolated_pair_count = "
              << cofactor_statistics.interpolated_pairs << '\n';
    std::cout << "cofactor_polynomial_pair_count = "
              << cofactor_statistics.polynomial_pairs << '\n';
    for (std::size_t dangerous_modes = 1;
         dangerous_modes + 1 <
             cofactor_statistics.dangerous_mode_counts.size();
         ++dangerous_modes) {
      std::cout << "cofactor_interpolated_q_" << dangerous_modes
                << "_pair_count = "
                << cofactor_statistics.dangerous_mode_counts[dangerous_modes]
                << '\n';
    }
    std::cout << "cofactor_interpolated_q_8plus_pair_count = "
              << cofactor_statistics.dangerous_mode_counts.back() << '\n';
    std::cout << "cofactor_dynamic_storage_bytes = "
              << cofactor_statistics.dynamic_bytes << '\n';
    for (std::size_t nullity = 0;
         nullity + 1 < cofactor_statistics.nullity_counts.size();
         ++nullity) {
      std::cout << "cofactor_nullity_" << nullity << "_pair_count = "
                << cofactor_statistics.nullity_counts[nullity] << '\n';
    }
    std::cout << "cofactor_nullity_5plus_pair_count = "
              << cofactor_statistics.nullity_counts.back() << '\n';
    for (std::size_t bin = 0;
         bin + 1 < cofactor_statistics.condition_decade_counts.size();
         ++bin) {
      std::cout << "cofactor_condition_1e" << 2 * bin << "_to_1e"
                << 2 * (bin + 1) << "_pair_count = "
                << cofactor_statistics.condition_decade_counts[bin] << '\n';
    }
    std::cout << "cofactor_condition_ge_1e14_pair_count = "
              << cofactor_statistics.condition_decade_counts.back() << '\n';
    if (!selected_states.states.empty()) {
      std::cout << "selected_alpha_support = "
                << selected_states.states.front().alpha_support.size() << '\n';
      std::cout << "selected_beta_support = "
                << selected_states.states.front().beta_support.size() << '\n';
    }
    std::cout << "supports_analytic_core_model = "
              << bool_name(first_diagnostics.supports_analytic_core_model) << '\n';
    std::cout << "outer_response_runtime_enabled = "
              << bool_name(first_diagnostics.outer_response_enabled) << '\n';
    std::cout << "exact_pair_products_streamed = "
              << bool_name(first_diagnostics.streams_exact_pair_products) << '\n';
    std::cout << "resident_exact_pair_elements = "
              << first_diagnostics.resident_exact_pair_elements << '\n';
    std::cout << "ao_pair_graph_storage_bytes = "
              << pair_graph_storage_bytes << '\n';
    std::cout << "packed_active_2e_storage_bytes = "
              << packed_active_storage_bytes << '\n';
    std::cout << "accepted_active_coefficient_storage_bytes = "
              << active_coefficient_storage_bytes << '\n';
    std::cout << "resident_exact_pair_storage_bytes = "
              << first_diagnostics.resident_exact_pair_elements * sizeof(double)
              << '\n';
    std::cout << "exact_pair_tile_rows = "
              << first_diagnostics.exact_pair_tile_rows << '\n';

    for (const BenchmarkMeasurement& measurement : measurements) {
      print_measurement(measurement);
    }
    if (block_measurement.has_value()) {
      const double block_count = static_cast<double>(options.repeats);
      const double block_average =
          block_measurement->external_wall_time_seconds / block_count;
      const double measured_block_width = std::min<Eigen::Index>(
          options.block_width, context.reduced_direction.size());
      std::cout << "full_block_width = " << measured_block_width << '\n';
      std::cout << "full_block_apply_count = "
                << block_measurement->diagnostics.batch_apply_count << '\n';
      std::cout << "full_block_response_inf_norm = "
                << block_measurement->response_inf_norm << '\n';
      std::cout << "full_block_external_avg_wall_time_seconds = "
                << block_average << '\n';
      std::cout << "full_block_external_avg_per_direction_seconds = "
                << block_average / measured_block_width << '\n';
      std::cout << "full_block_scalar_reference_relative_error = "
                << block_measurement->scalar_reference_relative_error << '\n';
      std::cout << "full_block_diag_avg_h1e_wall_time_seconds = "
                << average_wall_time_seconds(
                       block_measurement->diagnostics
                           .ao_effective_one_electron_fused_wall_time_seconds,
                       block_measurement->diagnostics.batch_apply_count)
                << '\n';
      std::cout << "full_block_diag_avg_active_space_integrals_wall_time_seconds = "
                << average_wall_time_seconds(
                       block_measurement->diagnostics
                           .outer_response_active_space_integrals_wall_time_seconds,
                       block_measurement->diagnostics.batch_apply_count)
                << '\n';
      std::cout << "full_block_diag_avg_structure_matrices_wall_time_seconds = "
                << average_wall_time_seconds(
                       block_measurement->diagnostics
                           .outer_response_structure_matrices_wall_time_seconds,
                       block_measurement->diagnostics.batch_apply_count)
                << '\n';
      const auto full_measurement = std::find_if(
          measurements.begin(),
          measurements.end(),
          [](const BenchmarkMeasurement& measurement) {
            return measurement.component == BenchmarkComponent::Full;
          });
      if (full_measurement != measurements.end()) {
        const double scalar_full_average =
            full_measurement->external_wall_time_seconds /
            static_cast<double>(options.repeats);
        const double block_speedup =
            (measured_block_width * scalar_full_average) / block_average;
        std::cout << "full_block_speedup_over_scalar_actions = "
                  << block_speedup << '\n';
        if (measured_block_width == 2.0) {
          std::cout << "full_block_speedup_over_two_scalar = "
                    << block_speedup << '\n';
        }
      }
    }
    run_dense_reduced_hessian_reference(
        context, options.dense_reference_block_width);
    run_spectral_audit(
        context,
        options.spectral_audit_dimension,
        options.spectral_audit_trust_radius);
    if (options.curvature_audit_directions > 0)
      run_curvature_audit(context, options.curvature_audit_directions);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
