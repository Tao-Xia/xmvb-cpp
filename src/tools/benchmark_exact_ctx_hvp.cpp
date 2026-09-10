#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include "vbscf/diagnostics/curvature_decomposition.hpp"
#include "vbscf/optimization/krylov/orthonormal_hvp_basis.hpp"
#include "vbscf/diagnostics/reduced_hessian_reference.hpp"

#include "runtime/vbscf_input_loader.hpp"
#include "vbscf/orbitals/charts/support_layout_adapter.hpp"
#include "vbscf/orbitals/charts/orbital_chart.hpp"
#include "vbscf/diagnostics/orbital_chart_audit.hpp"
#include "vbscf/orbitals/charts/sparse_parameter_layout.hpp"
#include "vbscf/derivatives/gradient/orbital_gradient_evaluator.hpp"
#include "vbscf/derivatives/gradient/orbital_gradient_result.hpp"
#include "vbscf/derivatives/hessian/accepted_point_context.hpp"
#include "vbscf/derivatives/hessian/exact_hvp_operator.hpp"

namespace {

struct Options {
  std::string input_path;
  int repeats = 1;
  int warmup = 0;
  bool nonredundant_adapt = false;
  bool gauge_audit = false;
  int curvature_audit_directions = 0;
  int dense_reference_block_width = 0;
  int block_width = 2;
  std::string orbital_value_table_bin_path;
  xmvb::vb::AoIntegralSource ao_integral_source =
      xmvb::vb::AoIntegralSource::Auto;
};

enum class BenchmarkComponent {
  Full,
  CoreOnly,
  OuterOnly,
};

struct AcceptedPointBenchmarkContext {
  xmvb::vb::VbScfInput input;
  std::shared_ptr<xmvb::vb::OrbitalGradientResult> gradient_result;
  std::shared_ptr<const xmvb::vb::AcceptedPointContext>
      second_order_context;
  xmvb::vb::SparseParameterLayout parameter_view;
  std::unique_ptr<xmvb::vb::OrbitalChart> nonredundant_space;
  Eigen::VectorXd reduced_direction;
  double nuclear_repulsion_energy = 0.0;
};

struct BenchmarkMeasurement {
  BenchmarkComponent component = BenchmarkComponent::Full;
  double external_wall_time_seconds = 0.0;
  double response_inf_norm = 0.0;
  xmvb::vb::ExactHvpOperator::Diagnostics diagnostics;
};

struct BlockBenchmarkMeasurement {
  double external_wall_time_seconds = 0.0;
  double response_inf_norm = 0.0;
  double scalar_reference_relative_error = 0.0;
  xmvb::vb::ExactHvpOperator::Diagnostics diagnostics;
};

void print_usage() {
  std::cerr
      << "usage: benchmark_exact_ctx_hvp <input.xmi>"
      << " [--repeats count]"
      << " [--warmup count]"
      << " [--ao-integral-source auto|libcint|runtime_hcore]"
      << " [--nonredundant-adapt true|false]"
      << " [--gauge-audit true|false]\n";
  std::cerr << " [--curvature-audit-directions count|0=disabled]\n";
  std::cerr << " [--dense-reference-block-width count|0=disabled]\n";
  std::cerr << " [--block-width count]\n";
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
  if (value == "libcint") {
    return xmvb::vb::AoIntegralSource::LibcintMaterialized;
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
    if (name == "--gauge-audit") {
      options.gauge_audit = parse_bool_argument(value);
      continue;
    }
    if (name == "--curvature-audit-directions") {
      options.curvature_audit_directions = parse_positive_or_zero_int(value, name.c_str());
      continue;
    }
    if (name == "--dense-reference-block-width") {
      options.dense_reference_block_width =
          parse_positive_or_zero_int(value, name.c_str());
      continue;
    }
    if (name == "--block-width") {
      options.block_width = parse_positive_or_zero_int(value, name.c_str());
      if (options.block_width <= 0) {
        throw std::invalid_argument("--block-width must be positive");
      }
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
  xmvb::vb::VbScfInputLoadOptions load_options;
  load_options.ao_integral_source = options.ao_integral_source;
  load_options.standard_two_electron_mode =
      xmvb::vb::StandardTwoElectronMode::Exact;
  const auto load_result =
      xmvb::vb::load_vbscf_input_with_timings(options.input_path, load_options);
  if (loaded_ao_integral_source != nullptr) {
    *loaded_ao_integral_source = load_result.ao_integral_source;
  }

  AcceptedPointBenchmarkContext context{
      options.nonredundant_adapt
          ? xmvb::vb::build_nonredundant_optimizer_input(load_result.input)
          : load_result.input,
      nullptr,
      nullptr,
      xmvb::vb::SparseParameterLayout(load_result.input.orbital_preparation_input),
      nullptr,
      Eigen::VectorXd()};
  context.parameter_view =
      xmvb::vb::SparseParameterLayout(context.input.orbital_preparation_input);
  context.nuclear_repulsion_energy = load_result.nuclear_repulsion_energy;
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

  Eigen::MatrixXd mq(g.size(), sampled.q.cols()), cq(g.size(), sampled.q.cols());
  for (int j = 0; j < sampled.q.cols(); ++j) {
    mq.col(j) = model(sampled.q.col(j)); cq.col(j) = core(sampled.q.col(j));
  }
  const Eigen::MatrixXd t = sampled.q.transpose() * sampled.hq;
  const Eigen::MatrixXd c = sampled.q.transpose() * cq;
  const Eigen::MatrixXd m = sampled.q.transpose() * mq;
  const auto symmetric = [](const Eigen::MatrixXd& a) -> Eigen::MatrixXd { return 0.5*(a+a.transpose()); };
  std::cout << "audit_full_relative_skew = " << (t-t.transpose()).norm()/t.norm() << '\n';
  std::cout << "audit_core_relative_skew = " << (c-c.transpose()).norm()/c.norm() << '\n';
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
        plus, {0}, {1.0}, context.nuclear_repulsion_energy);
    const auto gm = evaluator.evaluate_without_reference_energy_gradient(
        minus, {0}, {1.0}, context.nuclear_repulsion_energy);
    return space.project_reduced_gradient(context.parameter_view.gather_from_full(gp.sparse_orbital_energy_gradient) -
        context.parameter_view.gather_from_full(gm.sparse_orbital_energy_gradient)) / (2.0 * step);
  };
  const Eigen::VectorXd soft = sampled.q * eig.eigenvectors().col(0);
  const Eigen::VectorXd hsoft = full(soft);
  std::cout << "audit_soft_linearity_error = " <<
      (hsoft-sampled.hq*eig.eigenvectors().col(0)).norm()/hsoft.norm() << '\n';
  const Eigen::VectorXd csoft = core(soft);
  const Eigen::VectorXd osoft = apply_component(outer_op, BenchmarkComponent::OuterOnly, soft);
  std::cout << "audit_soft_core_linearity_error_over_full = " <<
      (csoft-cq*eig.eigenvectors().col(0)).norm()/hsoft.norm() << '\n';
  std::cout << "audit_soft_outer_linearity_error_over_full = " <<
      (osoft-(sampled.hq-cq)*eig.eigenvectors().col(0)).norm()/hsoft.norm() << '\n';
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
    return space.project_reduced_gradient(packed);
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
  measurement.scalar_reference_relative_error =
      (response - scalar_reference).norm() /
      std::max(1.0, scalar_reference.norm());
  if (measurement.scalar_reference_relative_error >
      std::sqrt(std::numeric_limits<double>::epsilon())) {
    throw std::runtime_error("block HVP differs from independent scalar actions");
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
  const double audit_tolerance =
      std::sqrt(std::numeric_limits<double>::epsilon());
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
  std::cout << "dense_reference_relative_skew = "
            << reference.relative_skew_norm << '\n';
  std::cout << "dense_reference_action_relative_error = "
            << action_relative_error << '\n';
  std::cout << "dense_reference_minimum_eigenvalue = "
            << eigensolver.eigenvalues().minCoeff() << '\n';
  std::cout << "dense_reference_maximum_eigenvalue = "
            << eigensolver.eigenvalues().maxCoeff() << '\n';
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
        run_full_block_benchmark(
            context, options.warmup, options.repeats, options.block_width);

    const auto& first_diagnostics = measurements.front().diagnostics;
    std::cout << std::setprecision(12);
    std::cout << "input = " << options.input_path << '\n';
    std::cout << "orbital_value_table_override = " << options.orbital_value_table_bin_path << '\n';
    const auto& energies = context.second_order_context->eigen_result.eigenvalues;
    double nearest_gap = std::numeric_limits<double>::infinity();
    for (std::size_t j = 1; j < energies.size(); ++j)
      nearest_gap = std::min(nearest_gap, std::abs(energies[j]-energies[0]));
    std::cout << "selected_state_nearest_gap = " << nearest_gap << '\n';
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
    const double measured_block_width = std::min<Eigen::Index>(
        options.block_width, context.reduced_direction.size());
    std::cout << "full_block_width = " << measured_block_width << '\n';
    std::cout << "full_block_apply_count = "
              << block_measurement.diagnostics.batch_apply_count << '\n';
    std::cout << "full_block_response_inf_norm = "
              << block_measurement.response_inf_norm << '\n';
    std::cout << "full_block_external_avg_wall_time_seconds = "
              << block_average << '\n';
    std::cout << "full_block_external_avg_per_direction_seconds = "
              << block_average / measured_block_width << '\n';
    std::cout << "full_block_scalar_reference_relative_error = "
              << block_measurement.scalar_reference_relative_error << '\n';
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
    std::cout << "full_block_diag_avg_structure_matrices_wall_time_seconds = "
              << average_wall_time_seconds(
                     block_measurement.diagnostics
                         .outer_response_structure_matrices_wall_time_seconds,
                     block_measurement.diagnostics.batch_apply_count)
              << '\n';
    const double scalar_full_average =
        measurements.front().external_wall_time_seconds /
        static_cast<double>(options.repeats);
    const double block_speedup =
        (measured_block_width * scalar_full_average) / block_average;
    std::cout << "full_block_speedup_over_scalar_actions = "
              << block_speedup << '\n';
    if (measured_block_width == 2.0) {
      std::cout << "full_block_speedup_over_two_scalar = "
                << block_speedup << '\n';
    }
    run_dense_reduced_hessian_reference(
        context, options.dense_reference_block_width);
    if (options.curvature_audit_directions > 0)
      run_curvature_audit(context, options.curvature_audit_directions);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
