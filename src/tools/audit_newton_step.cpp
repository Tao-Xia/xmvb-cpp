#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "input/loading/loader.hpp"
#include "vbscf/derivatives/gradient/orbital/evaluator.hpp"
#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "vbscf/optimization/objective/reduced_hvp.hpp"
#include "vbscf/optimization/trust_region/truncated_newton.hpp"
#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/layout.hpp"

namespace {

using namespace xmvb::vb;

struct Options {
  std::string input_path;
  std::string orbitals_path;
  std::string dump_trial_orbitals_path;
  StructureEigensolver eigensolver = StructureEigensolver::Davidson;
  int subspace_dimension = 0;
  double trust_radius = 0.0;
  double target_kkt_relative_residual =
      std::numeric_limits<double>::quiet_NaN();
  bool target_kkt_explicit = false;
};

Options parse_options(int argc, char** argv) {
  if (argc < 2 || (argc - 2) % 2 != 0) {
    throw std::invalid_argument(
        "usage: audit_newton_step input.xmi "
        "--subspace-dimension count --trust-radius value "
        "[--target-kkt-relative value] [--orbital-value-table-bin path] "
        "[--dump-trial-orbitals-bin path] "
        "[--eigensolver davidson|dense]");
  }
  Options options;
  options.input_path = argv[1];
  for (int index = 2; index < argc; index += 2) {
    const std::string name = argv[index];
    const std::string value = argv[index + 1];
    if (name == "--orbital-value-table-bin") {
      options.orbitals_path = value;
    } else if (name == "--dump-trial-orbitals-bin") {
      options.dump_trial_orbitals_path = value;
    } else if (name == "--subspace-dimension") {
      options.subspace_dimension = std::stoi(value);
    } else if (name == "--trust-radius") {
      options.trust_radius = std::stod(value);
    } else if (name == "--target-kkt-relative") {
      options.target_kkt_relative_residual = std::stod(value);
      options.target_kkt_explicit = true;
    } else if (name == "--eigensolver") {
      if (value == "davidson") {
        options.eigensolver = StructureEigensolver::Davidson;
      } else if (value == "dense") {
        options.eigensolver = StructureEigensolver::Dense;
      } else {
        throw std::invalid_argument("unsupported structure eigensolver");
      }
    } else {
      throw std::invalid_argument("unknown option: " + name);
    }
  }
  if (options.subspace_dimension <= 0 ||
      !(options.trust_radius > 0.0) ||
      !std::isfinite(options.trust_radius) ||
      (options.target_kkt_explicit &&
       (!std::isfinite(options.target_kkt_relative_residual) ||
        !(options.target_kkt_relative_residual >= 0.0 &&
          options.target_kkt_relative_residual < 1.0)))) {
    throw std::invalid_argument("invalid accepted-point audit options");
  }
  return options;
}

void load_orbitals(
    const std::string& path,
    OrbitalPreparationInput* input) {
  if (input == nullptr) {
    throw std::invalid_argument("null orbital input");
  }
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  const auto bytes = static_cast<std::streamoff>(
      input->orbital_value_table.size() * sizeof(double));
  if (!file || file.tellg() != bytes) {
    throw std::runtime_error("orbital table does not match the input chart");
  }
  file.seekg(0);
  file.read(
      reinterpret_cast<char*>(input->orbital_value_table.data()), bytes);
  if (!file || !Eigen::Map<const Eigen::VectorXd>(
          input->orbital_value_table.data(),
          input->orbital_value_table.size()).allFinite()) {
    throw std::runtime_error("invalid accepted-point orbital table");
  }
}

std::unique_ptr<OrbitalChart> build_chart(
    const VbScfInput& input,
    const SparseParameterLayout& layout,
    const OrbitalGradientResult& gradient) {
  const auto& orbital_input = input.orbital_preparation_input;
  const int n_inactive =
      (orbital_input.n_total_electrons - orbital_input.n_active_electrons) / 2;
  const int n_occupied = n_inactive + orbital_input.n_active_orbitals;
  return std::make_unique<OrbitalChart>(
      orbital_input,
      layout,
      gradient.orbital_preparation_result.auxiliary_orbital_matrix
          .leftCols(n_occupied),
      gradient.orbital_preparation_result.physical_orbital_frame
          .normalized_orbital_matrix,
      &gradient.ao_effective_one_electron_result.ao_effective_h1e);
}

class AcceptedPointHvp final : public ReducedHvp {
public:
  AcceptedPointHvp(
      const OrbitalGradientResult& gradient,
      const VbScfInput& input,
      const SparseParameterLayout& layout,
      const OrbitalChart& chart)
      : operator_(gradient.second_order_context, &input, layout, &chart) {
    if (gradient.second_order_context == nullptr) {
      throw std::runtime_error("accepted point lacks a second-order context");
    }
  }

  Eigen::VectorXd apply(const Eigen::VectorXd& direction) override {
    return operator_.apply_reduced(direction);
  }

  Eigen::MatrixXd apply_batch(
      const Eigen::Ref<const Eigen::MatrixXd>& directions) override {
    return operator_.apply_reduced_batch(directions);
  }

  ExactHvpOperator::Diagnostics diagnostics() const {
    return operator_.diagnostics();
  }

private:
  ExactHvpOperator operator_;
};

const char* stop_reason_name(TruncatedNewtonStopReason reason) {
  switch (reason) {
    case TruncatedNewtonStopReason::None: return "none";
    case TruncatedNewtonStopReason::ModelKktConverged: return "model_kkt";
    case TruncatedNewtonStopReason::BelowOuterAccuracy: return "below_outer_accuracy";
    case TruncatedNewtonStopReason::SubspaceLimit: return "subspace_limit";
    case TruncatedNewtonStopReason::InteriorPilotLimit: return "interior_pilot_limit";
    case TruncatedNewtonStopReason::DependentDirections: return "dependent_directions";
    case TruncatedNewtonStopReason::InvalidProjectedStep: return "invalid_projected_step";
    case TruncatedNewtonStopReason::RadiusAdjusted: return "radius_adjusted";
    case TruncatedNewtonStopReason::PreconditionedGradient: return "preconditioned_gradient";
  }
  return "unknown";
}

double relative_norm(
    const Eigen::VectorXd& numerator,
    const Eigen::VectorXd& denominator) {
  return numerator.stableNorm() /
      std::max(denominator.stableNorm(),
               std::numeric_limits<double>::min());
}

/** @brief Audits one fixed accepted-point orbital trust-region subproblem. */
void run_audit(const Options& options) {
  VbScfInputLoadOptions load_options;
  load_options.standard_two_electron_mode = StandardTwoElectronMode::Exact;
  auto loaded = load_vbscf_input_with_timings(
      options.input_path, load_options);
  VbScfInput input = std::move(loaded.input);
  if (!options.orbitals_path.empty()) {
    load_orbitals(options.orbitals_path, &input.orbital_preparation_input);
  }

  OrbitalGradientEvaluator evaluator;
  const Eigen::MatrixXd no_initial_eigenvectors;
  const StructureSolveAccuracy accuracy;
  const auto accepted = evaluator.evaluate_without_reference_energy_gradient(
      input, {0}, {1.0}, loaded.nuclear_repulsion_energy,
      options.eigensolver, accuracy, no_initial_eigenvectors);
  const SparseParameterLayout layout(input.orbital_preparation_input);
  auto chart = build_chart(input, layout, accepted);
  const Eigen::VectorXd packed_gradient = layout.gather_from_full(
      accepted.sparse_orbital_energy_gradient);
  const auto projected = chart->project_gradient(packed_gradient);
  const NonredundantRetractionMetric metric(
      *chart, layout, input.orbital_preparation_input);
  AcceptedPointHvp hvp(accepted, input, layout, *chart);

  const auto solve_start = std::chrono::steady_clock::now();
  const double target_kkt_relative_residual =
      options.target_kkt_explicit
      ? options.target_kkt_relative_residual
      : inexact_newton_forcing_term(projected.reduced_gradient.stableNorm());
  auto step = solve_nonredundant_truncated_newton_step(
      metric, *chart, projected, options.trust_radius,
      accuracy.energy_tolerance, accuracy.gradient_tolerance,
      target_kkt_relative_residual,
      options.subspace_dimension, &hvp, nullptr);
  clamp_nonredundant_step_result_to_retract_tangent_radius(
      projected, options.trust_radius, metric, &step);
  const double solve_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - solve_start).count();
  if (!truncated_newton_step_is_usable(
          step, projected.reduced_gradient)) {
    throw std::runtime_error("subproblem produced no usable descent step");
  }

  const Eigen::VectorXd cached_hs = step.reduced_hessian_times_step;
  const Eigen::VectorXd fresh_hs = hvp.apply(step.reduced_step);
  const Eigen::VectorXd fresh_ms = metric.apply(step.reduced_step);
  const Eigen::VectorXd fresh_residual = projected.reduced_gradient +
      fresh_hs + step.trust_region_shift * fresh_ms;
  Eigen::VectorXd cached_residual;
  if (cached_hs.size() == fresh_hs.size()) {
    cached_residual = projected.reduced_gradient + cached_hs +
        step.trust_region_shift * fresh_ms;
  }
  const Eigen::MatrixXd& q = step.subspace.orthonormal_basis;
  double projected_residual_relative =
      std::numeric_limits<double>::quiet_NaN();
  if (q.rows() == fresh_residual.size() && q.cols() > 0) {
    projected_residual_relative =
        (q.transpose() * fresh_residual).stableNorm() /
        projected.reduced_gradient.stableNorm();
  }

  VbScfInput trial = input;
  trial.orbital_preparation_input = chart->retract_step(
      input.orbital_preparation_input, step.reduced_step);
  if (!options.dump_trial_orbitals_path.empty()) {
    const auto& values = trial.orbital_preparation_input.orbital_value_table;
    std::ofstream file(options.dump_trial_orbitals_path, std::ios::binary);
    if (!file) {
      throw std::runtime_error("cannot open trial orbital table output");
    }
    file.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(double)));
    if (!file) {
      throw std::runtime_error("cannot write trial orbital table output");
    }
  }
  const auto trial_gradient = evaluator.evaluate_without_reference_energy_gradient(
      trial, {0}, {1.0}, loaded.nuclear_repulsion_energy,
      options.eigensolver, accuracy, no_initial_eigenvectors);
  const SparseParameterLayout trial_layout(trial.orbital_preparation_input);
  auto trial_chart = build_chart(trial, trial_layout, trial_gradient);
  const Eigen::VectorXd trial_projected_gradient =
      trial_chart->project_reduced_gradient(
          trial_layout.gather_from_full(
              trial_gradient.sparse_orbital_energy_gradient));
  const double actual_decrease =
      accepted.scf_result.total_energy -
      trial_gradient.scf_result.total_energy;
  const double fresh_predicted_decrease =
      -projected.reduced_gradient.dot(step.reduced_step) -
      0.5 * step.reduced_step.dot(fresh_hs);
  const auto diagnostics = hvp.diagnostics();

  std::cout << std::setprecision(15)
            << "eigensolver = " << structure_eigensolver_name(options.eigensolver) << '\n'
            << "reduced_dimension = " << chart->reduced_size() << '\n'
            << "subspace_budget = " << options.subspace_dimension << '\n'
            << "subspace_dimension = " << step.subspace_dimension << '\n'
            << "target_kkt_relative = " << target_kkt_relative_residual << '\n'
            << "stop_reason = " << stop_reason_name(step.stop_reason) << '\n'
            << "source_gradient_l2 = " << projected.reduced_gradient.stableNorm() << '\n'
            << "trial_gradient_l2 = " << trial_projected_gradient.stableNorm() << '\n'
            << "trial_gradient_inf = " << trial_projected_gradient.cwiseAbs().maxCoeff() << '\n'
            << "trust_radius = " << options.trust_radius << '\n'
            << "step_norm = " << metric.norm(step.reduced_step) << '\n'
            << "shift = " << step.trust_region_shift << '\n'
            << "boundary = " << step.reached_boundary << '\n'
            << "negative_curvature = " << step.encountered_negative_curvature << '\n'
            << "cached_kkt_relative = " << step.model_kkt_relative_residual << '\n'
            << "fresh_kkt_relative = "
            << relative_norm(fresh_residual, projected.reduced_gradient) << '\n'
            << "projected_fresh_kkt_relative = "
            << projected_residual_relative << '\n'
            << "cache_image_vs_fresh_relative = "
            << (cached_hs.size() == fresh_hs.size()
                    ? relative_norm(fresh_hs - cached_hs, fresh_hs)
                    : std::numeric_limits<double>::quiet_NaN()) << '\n'
            << "cached_recomputed_kkt_relative = "
            << (cached_residual.size() == fresh_residual.size()
                    ? relative_norm(cached_residual, projected.reduced_gradient)
                    : std::numeric_limits<double>::quiet_NaN()) << '\n'
            << "predicted_decrease = " << step.predicted_decrease << '\n'
            << "fresh_predicted_decrease = " << fresh_predicted_decrease << '\n'
            << "actual_decrease = " << actual_decrease << '\n'
            << "trust_ratio = " << actual_decrease / fresh_predicted_decrease << '\n'
            << "hvp_directions = " << diagnostics.apply_count << '\n'
            << "hvp_seconds = " << diagnostics.total_apply_wall_time_seconds << '\n'
            << "solve_seconds = " << solve_seconds << '\n'
            << "max_structure_response_relative_residual = "
            << diagnostics.max_structure_response_relative_residual << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    run_audit(parse_options(argc, argv));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "audit_newton_step: " << error.what() << '\n';
    return 1;
  }
}
