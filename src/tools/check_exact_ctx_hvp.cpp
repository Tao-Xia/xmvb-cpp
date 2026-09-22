#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "input/loading/loader.hpp"
#include "vbscf/derivatives/gradient/orbital/evaluator.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/derivatives/hessian/coupled/operator.hpp"
#include "vbscf/derivatives/hessian/coupled/structure.hpp"
#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/layout.hpp"
#include "vbscf/optimization/preconditioners/hessian_diagonal.hpp"

namespace {

struct Options {
  std::string input_path;
  std::string orbital_value_table_bin_path;
  xmvb::vb::StructureEigensolver eigensolver =
      xmvb::vb::StructureEigensolver::Dense;
  xmvb::vb::StandardTwoElectronMode two_electron_mode =
      xmvb::vb::StandardTwoElectronMode::Exact;
  bool preconditioned_direction = false;
  double step = 1.0e-4;
  double max_relative_error = std::numeric_limits<double>::infinity();
  double response_tolerance = 1.0e-3;
  bool stream_pair_products = false;
  int diagonal_samples = 0;
  bool coupled_finite_difference = false;
};

void print_usage() {
  std::cerr
      << "usage: check_exact_ctx_hvp <input.xmi>"
      << " [--orbital-value-table-bin path]"
      << " [--step h]"
      << " [--max-rel-error tolerance]"
      << " [--response-tolerance tolerance]"
      << " [--eigensolver dense|davidson]"
      << " [--standard-two-electron-mode exact|ri]"
      << " [--direction gradient|preconditioned_gradient]"
      << " [--stream-pair-products 0|1]"
      << " [--diagonal-samples count]\n";
}

Options parse_arguments(int argc, char** argv) {
  if (argc < 2 || (argc - 2) % 2 != 0) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }
  Options options;
  options.input_path = argv[1];
  for (int argument = 2; argument < argc; argument += 2) {
    const std::string name = argv[argument];
    const std::string value = argv[argument + 1];
    if (name == "--orbital-value-table-bin") {
      options.orbital_value_table_bin_path = value;
    } else if (name == "--step") {
      options.step = std::stod(value);
    } else if (name == "--max-rel-error") {
      options.max_relative_error = std::stod(value);
    } else if (name == "--response-tolerance") {
      options.response_tolerance = std::stod(value);
    } else if (name == "--eigensolver") {
      if (value == "dense") {
        options.eigensolver = xmvb::vb::StructureEigensolver::Dense;
      } else if (value == "davidson") {
        options.eigensolver = xmvb::vb::StructureEigensolver::Davidson;
      } else {
        throw std::invalid_argument("--eigensolver must be dense or davidson");
      }
    } else if (name == "--standard-two-electron-mode") {
      if (value == "exact") {
        options.two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::Exact;
      } else if (value == "ri") {
        options.two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
      } else {
        throw std::invalid_argument(
            "--standard-two-electron-mode must be exact or ri");
      }
    } else if (name == "--direction") {
      if (value == "gradient") {
        options.preconditioned_direction = false;
      } else if (value == "preconditioned_gradient") {
        options.preconditioned_direction = true;
      } else {
        throw std::invalid_argument(
            "--direction must be gradient or preconditioned_gradient");
      }
    } else if (name == "--stream-pair-products") {
      if (value != "0" && value != "1") {
        throw std::invalid_argument("--stream-pair-products must be 0 or 1");
      }
      options.stream_pair_products = value == "1";
    } else if (name == "--diagonal-samples") {
      options.diagonal_samples = std::stoi(value);
    } else if (name == "--coupled-finite-difference") {
      if (value != "0" && value != "1") {
        throw std::invalid_argument(
            "--coupled-finite-difference must be 0 or 1");
      }
      options.coupled_finite_difference = value == "1";
    } else if (name == "--probe") {
      if (value != "full") {
        throw std::invalid_argument("only the complete exact HVP is supported");
      }
    } else {
      throw std::invalid_argument("unknown argument: " + name);
    }
  }
  if (!(options.step > 0.0) || !(options.max_relative_error >= 0.0) ||
      options.diagonal_samples < 0 ||
      !(options.response_tolerance > 0.0)) {
    throw std::invalid_argument("finite-difference tolerances must be non-negative");
  }
  return options;
}

double infinity_norm(const Eigen::Ref<const Eigen::VectorXd>& values) {
  return values.size() == 0 ? 0.0 : values.cwiseAbs().maxCoeff();
}

double relative_matrix_error(
    const Eigen::Ref<const Eigen::MatrixXd>& actual,
    const Eigen::Ref<const Eigen::MatrixXd>& reference) {
  if (actual.rows() != reference.rows() ||
      actual.cols() != reference.cols()) {
    return std::numeric_limits<double>::infinity();
  }
  const double error = actual.size() == 0
      ? 0.0
      : (actual - reference).cwiseAbs().maxCoeff();
  const double scale = reference.size() == 0
      ? 1.0
      : std::max(1.0, reference.cwiseAbs().maxCoeff());
  return error / scale;
}

struct CoupledReferenceAudit {
  double adjoint_relative_error = 0.0;
  double relaxed_relative_error = 0.0;
  int structure_coordinate_dimension = 0;
  int structure_response_rank = 0;
};

Eigen::VectorXd flatten_structure_tangent(
    const xmvb::vb::StructureTangent& tangent) {
  return Eigen::Map<const Eigen::VectorXd>(
      tangent.scaled_coefficients.data(),
      tangent.scaled_coefficients.size());
}

xmvb::vb::StructureTangent unflatten_structure_tangent(
    const Eigen::Ref<const Eigen::VectorXd>& coefficients,
    int n_structures,
    int n_states) {
  if (coefficients.size() != n_structures * n_states) {
    throw std::invalid_argument(
        "coupled reference tangent has inconsistent dimensions");
  }
  return xmvb::vb::StructureTangent{
      Eigen::Map<const Eigen::MatrixXd>(
          coefficients.data(), n_structures, n_states)};
}

/** @brief Independently eliminates the small structure block of coupled NEO. */
CoupledReferenceAudit audit_coupled_structure_elimination(
    const xmvb::vb::ExactHvpOperator& exact_hvp,
    const std::shared_ptr<const xmvb::vb::AcceptedPointContext>& accepted,
    const Eigen::Ref<const Eigen::VectorXd>& orbital_direction,
    const Eigen::Ref<const Eigen::VectorXd>& relaxed_hvp,
    bool enforce) {
  const xmvb::vb::StructureTangentOperator structure(
      accepted, exact_hvp.structure_action());
  const int n_structures = structure.n_structures();
  const int n_states = structure.n_states();
  const int dimension = n_structures * n_states;
  constexpr int kMaximumDenseReferenceDimension = 4096;
  if (dimension <= 0 || dimension > kMaximumDenseReferenceDimension) {
    throw std::runtime_error(
        "coupled dense reference dimension is outside the diagnostic limit");
  }
  const xmvb::vb::OrbitalCouplingAction orbital =
      exact_hvp.apply_orbital_coupling(orbital_direction);
  const xmvb::vb::StructureTangent forcing = structure.project(
      orbital.scaled_structure_forcing);
  const Eigen::VectorXd flat_forcing = flatten_structure_tangent(forcing);

  Eigen::MatrixXd structure_hessian(dimension, dimension);
  for (int coordinate = 0; coordinate < dimension; ++coordinate) {
    Eigen::VectorXd canonical = Eigen::VectorXd::Zero(dimension);
    canonical[coordinate] = 1.0;
    const auto tangent = structure.project(
        unflatten_structure_tangent(
            canonical, n_structures, n_states).scaled_coefficients);
    structure_hessian.col(coordinate) = flatten_structure_tangent(
        structure.apply_hessian(tangent));
  }
  const double structure_scale = std::max(1.0, structure_hessian.norm());
  const double symmetry_error =
      (structure_hessian - structure_hessian.transpose()).norm() /
      structure_scale;
  if (symmetry_error > 2.0e-10) {
    throw std::runtime_error(
        "coupled reference structure Hessian is not symmetric");
  }
  structure_hessian =
      0.5 * (structure_hessian + structure_hessian.transpose()).eval();

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(structure_hessian);
  if (spectrum.info() != Eigen::Success) {
    throw std::runtime_error(
        "coupled reference structure Hessian eigensolve failed");
  }
  const double eigenvalue_scale = std::max(
      1.0, spectrum.eigenvalues().cwiseAbs().maxCoeff());
  const double rank_tolerance = 1.0e3 *
      std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max(1, dimension)) * eigenvalue_scale;
  Eigen::VectorXd inverse_eigenvalues = Eigen::VectorXd::Zero(dimension);
  int response_rank = 0;
  for (int mode = 0; mode < dimension; ++mode) {
    const double eigenvalue = spectrum.eigenvalues()[mode];
    if (std::abs(eigenvalue) <= rank_tolerance) continue;
    inverse_eigenvalues[mode] = 1.0 / eigenvalue;
    ++response_rank;
  }
  const Eigen::VectorXd flat_response =
      -spectrum.eigenvectors() *
      (inverse_eigenvalues.asDiagonal() *
       (spectrum.eigenvectors().transpose() * flat_forcing));
  const xmvb::vb::StructureTangent response = structure.project(
      unflatten_structure_tangent(
          flat_response, n_structures, n_states).scaled_coefficients);
  const xmvb::vb::StructureCouplingAction response_action =
      structure.apply_coupling(response);
  const Eigen::VectorXd response_adjoint =
      exact_hvp.apply_structure_coupling_adjoint(
          response_action.coefficient_response,
          response_action.adjoint_multipliers);
  const Eigen::VectorXd eliminated_hvp =
      orbital.orbital_hessian + response_adjoint;

  Eigen::VectorXd probe_values(dimension);
  for (int coordinate = 0; coordinate < dimension; ++coordinate) {
    probe_values[coordinate] = std::sin(
        0.37 * static_cast<double>(coordinate + 1));
  }
  const xmvb::vb::StructureTangent probe = structure.project(
      unflatten_structure_tangent(
          probe_values, n_structures, n_states).scaled_coefficients);
  const xmvb::vb::StructureCouplingAction probe_action =
      structure.apply_coupling(probe);
  const Eigen::VectorXd probe_adjoint =
      exact_hvp.apply_structure_coupling_adjoint(
          probe_action.coefficient_response,
          probe_action.adjoint_multipliers);
  const double forward_bilinear =
      flatten_structure_tangent(probe).dot(flat_forcing);
  const double adjoint_bilinear = orbital_direction.dot(probe_adjoint);

  CoupledReferenceAudit audit;
  audit.adjoint_relative_error =
      std::abs(forward_bilinear - adjoint_bilinear) /
      std::max({1.0, std::abs(forward_bilinear),
                std::abs(adjoint_bilinear)});
  audit.relaxed_relative_error =
      infinity_norm(eliminated_hvp - relaxed_hvp) /
      std::max(1.0, infinity_norm(relaxed_hvp));
  audit.structure_coordinate_dimension = dimension;
  audit.structure_response_rank = response_rank;
  if (enforce && audit.adjoint_relative_error > 2.0e-9) {
    throw std::runtime_error(
        "coupled orbital-structure actions violate the bilinear adjoint identity");
  }
  if (enforce && audit.relaxed_relative_error > 2.0e-8) {
    throw std::runtime_error(
        "coupled Schur elimination does not reproduce the relaxed HVP");
  }
  return audit;
}

/** @brief Loads an exact accepted-point orbital table for derivative checks. */
void overwrite_orbital_values_from_binary(
    const std::string& path,
    xmvb::vb::OrbitalPreparationInput* orbitals) {
  if (path.empty()) return;
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  const auto expected_bytes = static_cast<std::streamoff>(
      orbitals->orbital_value_table.size() * sizeof(double));
  if (!input || input.tellg() != expected_bytes) {
    throw std::runtime_error(
        "orbital-value-table file size does not match input chart");
  }
  input.seekg(0, std::ios::beg);
  input.read(
      reinterpret_cast<char*>(orbitals->orbital_value_table.data()),
      expected_bytes);
  if (!input ||
      !Eigen::Map<const Eigen::VectorXd>(
           orbitals->orbital_value_table.data(),
           orbitals->orbital_value_table.size()).allFinite()) {
    throw std::runtime_error("invalid orbital-value-table file");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    xmvb::vb::VbScfInputLoadOptions load_options;
    load_options.standard_two_electron_mode = options.two_electron_mode;
    const auto loaded =
        xmvb::vb::load_vbscf_input_with_timings(options.input_path, load_options);
    xmvb::vb::VbScfInput input = loaded.input;
    if (options.two_electron_mode ==
        xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity) {
      // RI derivative checks must remain factor-native.  Removing any exact
      // graph supplied by a future loader change makes an accidental exact
      // fallback fail immediately instead of silently passing this test.
      input.ao_integral_input.pair_graph = {};
    }
    overwrite_orbital_values_from_binary(
        options.orbital_value_table_bin_path,
        &input.orbital_preparation_input);

    xmvb::vb::OrbitalGradientEvaluator evaluator;
    const Eigen::MatrixXd no_initial_eigenvectors;
    std::vector<int> selected_states(loaded.state_average_count);
    std::iota(selected_states.begin(), selected_states.end(), 0);
    const std::vector<double> equal_weights(
        selected_states.size(),
        1.0);
    auto accepted = evaluator.evaluate_without_reference_energy_gradient(
        input,
        selected_states,
        equal_weights,
        loaded.nuclear_repulsion_energy,
        options.eigensolver,
        xmvb::vb::StructureSolveAccuracy{
            1.0e-7,
            options.response_tolerance},
        no_initial_eigenvectors);
    if (accepted.second_order_context == nullptr) {
      throw std::runtime_error("accepted-point second-order context is unavailable");
    }

    xmvb::vb::SparseParameterLayout layout(input.orbital_preparation_input);
    const int n_inactive =
        (input.orbital_preparation_input.n_total_electrons -
         input.orbital_preparation_input.n_active_electrons) /
        2;
    const int n_occupied =
        n_inactive + input.orbital_preparation_input.n_active_orbitals;
    const Eigen::MatrixXd& normalized_orbitals =
        accepted.orbital_preparation_result.physical_orbital_frame
            .normalized_orbital_matrix;
    if (normalized_orbitals.size() == 0) {
      throw std::runtime_error("accepted physical orbital frame is unavailable");
    }
    std::optional<xmvb::vb::AnalyticOrbitalDiagonal> analytic_diagonal;
    if (options.diagonal_samples > 0) {
      analytic_diagonal = xmvb::vb::build_analytic_orbital_diagonal(
          input, *accepted.second_order_context);
    }
    xmvb::vb::OrbitalChart chart(
        input.orbital_preparation_input,
        layout,
        accepted.orbital_preparation_result.auxiliary_orbital_matrix.leftCols(
            n_occupied),
        normalized_orbitals,
        analytic_diagonal
            ? nullptr
            : &accepted.ao_effective_one_electron_result.ao_effective_h1e,
        false,
        input.complete_active_space,
        analytic_diagonal ? &*analytic_diagonal : nullptr);
    const Eigen::VectorXd packed_gradient =
        layout.gather_from_full(accepted.sparse_orbital_energy_gradient);
    const Eigen::VectorXd accepted_reduced_gradient =
        chart.project_reduced_gradient(packed_gradient);
    Eigen::VectorXd direction = accepted_reduced_gradient;
    if (options.preconditioned_direction) {
      direction = chart.apply_inverse_reduced_block_preconditioner(direction);
    }
    if (direction.size() == 0) {
      throw std::runtime_error("nonredundant orbital space is empty");
    }
    if (direction.norm() <= std::sqrt(std::numeric_limits<double>::epsilon())) {
      direction.setZero();
      direction[0] = 1.0;
    } else {
      direction.normalize();
    }

    if (options.stream_pair_products) {
      if (options.two_electron_mode !=
          xmvb::vb::StandardTwoElectronMode::Exact) {
        throw std::invalid_argument(
            "--stream-pair-products is only defined for exact AO integrals");
      }
      accepted.second_order_context->prepared_active_space
          .active_space_two_electron_result.dense_ao_pair_products.resize(0, 0);
    }
    xmvb::vb::ExactHvpOperator exact_hvp(
        accepted.second_order_context,
        &input,
        layout,
        &chart);
    const Eigen::VectorXd analytic = exact_hvp.apply_reduced(direction);

    double coupled_orbital_fd_relative_error = 0.0;
    double coupled_structure_fd_relative_error = 0.0;
    double coupled_directional_curvature_relative_error = 0.0;
    if (options.coupled_finite_difference) {
      xmvb::vb::StructureTangentOperator structure(
          accepted.second_order_context, exact_hvp.structure_action());
      Eigen::VectorXd structure_coordinates = Eigen::VectorXd::LinSpaced(
          structure.tangent_size(), 1.0,
          static_cast<double>(structure.tangent_size()));
      structure_coordinates = structure_coordinates.array().sin().matrix();
      xmvb::vb::StructureTangent structure_direction =
          structure.expand(structure_coordinates);
      const double structure_norm =
          std::sqrt(structure.squared_norm(structure_direction));
      if (!(structure_norm > 0.0) || !std::isfinite(structure_norm)) {
        throw std::runtime_error(
            "coupled finite-difference structure direction is singular");
      }
      structure_coordinates /= structure_norm;
      structure_direction = structure.expand(structure_coordinates);

      xmvb::vb::CoupledHessianOperator coupled_hessian(
          exact_hvp, structure);
      const xmvb::vb::CoupledDirection joint_direction{
          direction, structure_direction};
      const xmvb::vb::CoupledDirection joint_analytic =
          coupled_hessian.apply(joint_direction);
      const Eigen::VectorXd analytic_structure =
          structure.coordinates(joint_analytic.structure);
      const Eigen::MatrixXd coefficient_direction =
          structure.coefficient_response(structure_direction);
      const Eigen::MatrixXd accepted_coefficients =
          accepted.second_order_context->selected_state_eigenvectors;

      const auto fixed_joint_gradient = [&](double step_scale) {
        xmvb::vb::VbScfInput displaced_input = input;
        displaced_input.orbital_preparation_input = chart.retract_step(
            input.orbital_preparation_input, direction, step_scale);
        const Eigen::MatrixXd coefficients =
            accepted_coefficients + step_scale * coefficient_direction;
        const auto fixed = evaluator.evaluate_fixed_structure(
            displaced_input,
            selected_states,
            equal_weights,
            loaded.nuclear_repulsion_energy,
            coefficients);
        Eigen::VectorXd orbital = chart.project_reduced_gradient(
            layout.gather_from_full(
                fixed.gradient.sparse_orbital_energy_gradient));
        Eigen::MatrixXd scaled_residuals = fixed.structure_residuals;
        for (Eigen::Index state = 0;
             state < scaled_residuals.cols(); ++state) {
          scaled_residuals.col(state) *= std::sqrt(
              2.0 * accepted.second_order_context
                  ->normalized_state_weights[static_cast<std::size_t>(state)]);
        }
        return std::pair{
            std::move(orbital),
            structure.project_coordinates(scaled_residuals)};
      };
      const auto plus_joint = fixed_joint_gradient(options.step);
      const auto minus_joint = fixed_joint_gradient(-options.step);
      const Eigen::VectorXd fd_orbital =
          (plus_joint.first - minus_joint.first) / (2.0 * options.step);
      const Eigen::VectorXd fd_structure =
          (plus_joint.second - minus_joint.second) / (2.0 * options.step);
      coupled_orbital_fd_relative_error =
          infinity_norm(joint_analytic.orbital - fd_orbital) /
          std::max(1.0, infinity_norm(fd_orbital));
      coupled_structure_fd_relative_error =
          infinity_norm(analytic_structure - fd_structure) /
          std::max(1.0, infinity_norm(fd_structure));
      const double analytic_curvature =
          direction.dot(joint_analytic.orbital) +
          structure_coordinates.dot(analytic_structure);
      const double finite_difference_curvature =
          direction.dot(fd_orbital) +
          structure_coordinates.dot(fd_structure);
      coupled_directional_curvature_relative_error =
          std::abs(analytic_curvature - finite_difference_curvature) /
          std::max(1.0, std::abs(finite_difference_curvature));
      std::cout << std::setprecision(12)
                << "coupled_orbital_fd_relative_error = "
                << coupled_orbital_fd_relative_error << '\n'
                << "coupled_structure_fd_relative_error = "
                << coupled_structure_fd_relative_error << '\n'
                << "coupled_directional_curvature_relative_error = "
                << coupled_directional_curvature_relative_error << std::endl;
    }

    double diagonal_sample_max_relative_error = 0.0;
    Eigen::Index diagonal_sample_worst_coordinate = -1;
    double diagonal_sample_worst_analytic = 0.0;
    double diagonal_sample_worst_exact = 0.0;
    if (options.diagonal_samples > 0) {
      const Eigen::VectorXd diagonal = chart.augmented_hessian_diagonal();
      const int sample_count = std::min<int>(
          options.diagonal_samples, static_cast<int>(direction.size()));
      for (int sample = 0; sample < sample_count; ++sample) {
        const Eigen::Index coordinate = sample_count == 1
            ? 0
            : static_cast<Eigen::Index>(
                  sample * (direction.size() - 1) / (sample_count - 1));
        Eigen::VectorXd basis = Eigen::VectorXd::Zero(direction.size());
        basis[coordinate] = 1.0;
        const double exact_diagonal =
            exact_hvp.apply_orbital_coupling(basis)
                .orbital_hessian[coordinate];
        const double scale = std::max(1.0, std::abs(exact_diagonal));
        const double error =
            std::abs(diagonal[coordinate] - exact_diagonal) / scale;
        if (error > diagonal_sample_max_relative_error) {
          diagonal_sample_max_relative_error = error;
          diagonal_sample_worst_coordinate = coordinate;
          diagonal_sample_worst_analytic = diagonal[coordinate];
          diagonal_sample_worst_exact = exact_diagonal;
        }
      }
    }

    Eigen::MatrixXd coupling_directions(direction.size(), 2);
    coupling_directions.col(0) = direction;
    Eigen::VectorXd second_direction = Eigen::VectorXd::LinSpaced(
        direction.size(), 1.0, static_cast<double>(direction.size()));
    second_direction.noalias() -=
        direction.dot(second_direction) * direction;
    if (second_direction.norm() <=
        std::sqrt(std::numeric_limits<double>::epsilon())) {
      second_direction = -0.375 * direction;
    } else {
      second_direction.normalize();
    }
    coupling_directions.col(1) = second_direction;
    const xmvb::vb::OrbitalCouplingBlockAction block_coupling =
        exact_hvp.apply_orbital_coupling_batch(coupling_directions);
    Eigen::MatrixXd scalar_orbital_coupling(direction.size(), 2);
    std::vector<Eigen::MatrixXd> scalar_structure_forcing(2);
    for (Eigen::Index column = 0; column < 2; ++column) {
      xmvb::vb::OrbitalCouplingAction scalar_coupling =
          exact_hvp.apply_orbital_coupling(coupling_directions.col(column));
      scalar_orbital_coupling.col(column) = scalar_coupling.orbital_hessian;
      scalar_structure_forcing[static_cast<std::size_t>(column)] =
          std::move(scalar_coupling.scaled_structure_forcing);
    }
    const double coupling_batch_orbital_error = relative_matrix_error(
        block_coupling.orbital_hessian, scalar_orbital_coupling);
    double coupling_batch_structure_error = 0.0;
    for (Eigen::Index column = 0; column < 2; ++column) {
      coupling_batch_structure_error = std::max(
          coupling_batch_structure_error,
          relative_matrix_error(
              block_coupling.scaled_structure_forcing[
                  static_cast<std::size_t>(column)],
              scalar_structure_forcing[static_cast<std::size_t>(column)]));
    }
    if (coupling_batch_orbital_error > 1.0e-10 ||
        coupling_batch_structure_error > 1.0e-10) {
      throw std::runtime_error(
          "orbital coupling block disagrees with scalar actions");
    }
    Eigen::MatrixXd relaxed_images(direction.size(), 2);
    relaxed_images.col(0) = analytic;
    relaxed_images.col(1) = exact_hvp.apply_reduced(second_direction);
    const Eigen::Matrix2d projected_orbital =
        coupling_directions.transpose() * scalar_orbital_coupling;
    const Eigen::Matrix2d projected_relaxed =
        coupling_directions.transpose() * relaxed_images;
    const auto relative_symmetry_error = [](const Eigen::Matrix2d& matrix) {
      return (matrix - matrix.transpose()).norm() /
          std::max(1.0, matrix.norm());
    };
    const double orbital_symmetry_error =
        relative_symmetry_error(projected_orbital);
    const double relaxed_symmetry_error =
        relative_symmetry_error(projected_relaxed);
    const CoupledReferenceAudit coupled_audit =
        audit_coupled_structure_elimination(
            exact_hvp,
            accepted.second_order_context,
            direction,
            analytic,
            !options.coupled_finite_difference);
    const Eigen::VectorXd analytic_core = exact_hvp.apply_reduced(
        direction,
        {.direct_core_response = true,
         .fixed_upstream_pullback = true,
         .local_active_response = false,
         .structure_response = false});
    const Eigen::VectorXd analytic_local_active = exact_hvp.apply_reduced(
        direction,
        {.direct_core_response = false,
         .fixed_upstream_pullback = false,
         .local_active_response = true,
         .structure_response = false});
    const Eigen::VectorXd analytic_structure_response =
        exact_hvp.apply_reduced(
            direction,
            {.direct_core_response = false,
             .fixed_upstream_pullback = false,
             .local_active_response = false,
             .structure_response = true});
    const double component_decomposition_error = infinity_norm(
        analytic - analytic_core - analytic_local_active -
        analytic_structure_response) /
        std::max(1.0, infinity_norm(analytic));
    if (component_decomposition_error > 1.0e-10) {
      throw std::runtime_error(
          "exact HVP component decomposition is inconsistent");
    }
    const auto hvp_diagnostics = exact_hvp.diagnostics();
    if (hvp_diagnostics.orbital_coupling_batch_count != 1 ||
        hvp_diagnostics.orbital_coupling_batch_chunk_count < 1 ||
        hvp_diagnostics.max_orbital_coupling_batch_width < 1 ||
        hvp_diagnostics.max_orbital_coupling_batch_width > 2) {
      throw std::runtime_error(
          "orbital coupling block diagnostics are inconsistent");
    }

    double state_energy_average_error = 0.0;
    double state_gradient_average_error = 0.0;
    double state_hvp_average_error = 0.0;
    if (selected_states.size() > 1) {
      double state_energy_average = 0.0;
      Eigen::VectorXd state_gradient_average = Eigen::VectorXd::Zero(
          accepted_reduced_gradient.size());
      Eigen::VectorXd state_hvp_average = Eigen::VectorXd::Zero(
          analytic.size());
      for (std::size_t offset = 0; offset < selected_states.size(); ++offset) {
        const auto state = evaluator.evaluate_without_reference_energy_gradient(
            input,
            {selected_states[offset]},
            {1.0},
            loaded.nuclear_repulsion_energy,
            options.eigensolver,
            xmvb::vb::StructureSolveAccuracy{
                1.0e-7,
                options.response_tolerance},
            no_initial_eigenvectors);
        const double weight =
            accepted.second_order_context->normalized_state_weights[offset];
        state_energy_average += weight * state.scf_result.total_energy;
        state_gradient_average.noalias() += weight *
            chart.project_reduced_gradient(
                layout.gather_from_full(
                    state.sparse_orbital_energy_gradient));
        xmvb::vb::ExactHvpOperator state_hvp(
            state.second_order_context,
            &input,
            layout,
            &chart);
        state_hvp_average.noalias() +=
            weight * state_hvp.apply_reduced(direction);
      }
      state_energy_average_error = std::abs(
          accepted.scf_result.total_energy - state_energy_average);
      state_gradient_average_error = infinity_norm(
          accepted_reduced_gradient - state_gradient_average);
      state_hvp_average_error = infinity_norm(
          analytic - state_hvp_average);
      if (state_energy_average_error > 1.0e-10 ||
          state_gradient_average_error > 1.0e-9 ||
          state_hvp_average_error > 1.0e-8) {
        throw std::runtime_error(
            "equal-weight state-average linearity check failed");
      }
    }

    xmvb::vb::VbScfInput displaced = input;
    displaced.orbital_preparation_input = chart.retract_step(
        input.orbital_preparation_input,
        direction,
        options.step);
    const auto evaluate_reduced_gradient = [&](const xmvb::vb::VbScfInput& point) {
      const auto point_gradient =
          evaluator.evaluate_without_reference_energy_gradient(
              point,
              accepted.second_order_context->selected_state_indices,
              accepted.second_order_context->normalized_state_weights,
              loaded.nuclear_repulsion_energy);
      return std::pair{
          chart.project_reduced_gradient(
              layout.gather_from_full(
                  point_gradient.sparse_orbital_energy_gradient)),
          point_gradient.scf_result.total_energy};
    };
    const auto [plus_reduced, plus_energy] =
        evaluate_reduced_gradient(displaced);
    const Eigen::VectorXd plus_core_reduced = chart.project_reduced_gradient(
        layout.gather_from_full(
            evaluator.evaluate_sparse_orbital_gradient_with_fixed_active_space_adjoint(
                displaced,
                *accepted.second_order_context,
                loaded.nuclear_repulsion_energy)));

    displaced.orbital_preparation_input = chart.retract_step(
        input.orbital_preparation_input,
        direction,
        -options.step);
    const auto [minus_reduced, minus_energy] =
        evaluate_reduced_gradient(displaced);
    const Eigen::VectorXd minus_core_reduced = chart.project_reduced_gradient(
        layout.gather_from_full(
            evaluator.evaluate_sparse_orbital_gradient_with_fixed_active_space_adjoint(
                displaced,
                *accepted.second_order_context,
                loaded.nuclear_repulsion_energy)));
    const Eigen::VectorXd finite_difference =
        (plus_reduced - minus_reduced) / (2.0 * options.step);
    const Eigen::VectorXd dense_midpoint_reduced_gradient =
        0.5 * (plus_reduced + minus_reduced);
    const double accepted_gradient_vs_dense_midpoint_inf = infinity_norm(
        accepted_reduced_gradient - dense_midpoint_reduced_gradient);
    const Eigen::VectorXd finite_difference_core =
        (plus_core_reduced - minus_core_reduced) / (2.0 * options.step);
    const Eigen::VectorXd analytic_outer = analytic - analytic_core;
    const Eigen::VectorXd finite_difference_outer =
        finite_difference - finite_difference_core;
    const double analytic_directional_curvature = direction.dot(analytic);
    const double gradient_fd_directional_curvature =
        direction.dot(finite_difference);
    const double energy_directional_curvature =
        (plus_energy - 2.0 * accepted.scf_result.total_energy + minus_energy) /
        (options.step * options.step);
    const double energy_curvature_relative_error =
        std::abs(analytic_directional_curvature - energy_directional_curvature) /
        std::max(1.0, std::abs(energy_directional_curvature));

    const double max_abs_difference = infinity_norm(analytic - finite_difference);
    const double relative_error =
        max_abs_difference / std::max(1.0, infinity_norm(finite_difference));
    const double core_relative_error =
        infinity_norm(analytic_core - finite_difference_core) /
        std::max(1.0, infinity_norm(finite_difference_core));
    const double outer_relative_error =
        infinity_norm(analytic_outer - finite_difference_outer) /
        std::max(1.0, infinity_norm(finite_difference_outer));
    std::cout << std::setprecision(12)
              << "input = " << options.input_path << '\n'
              << "orbital_value_table_override = "
              << (options.orbital_value_table_bin_path.empty()
                      ? "none"
                      : options.orbital_value_table_bin_path)
              << '\n'
              << "stream_pair_products = "
              << (options.stream_pair_products ? "true" : "false") << '\n'
              << "eigensolver = "
              << xmvb::vb::structure_eigensolver_name(options.eigensolver)
              << '\n'
              << "direction = "
              << (options.preconditioned_direction
                      ? "preconditioned_gradient"
                      : "gradient") << '\n'
              << "reduced_dimension = " << direction.size() << '\n'
              << "finite_difference_step = " << options.step << '\n'
              << "coupled_finite_difference = "
              << (options.coupled_finite_difference ? "true" : "false") << '\n'
              << "coupled_orbital_fd_relative_error = "
              << coupled_orbital_fd_relative_error << '\n'
              << "coupled_structure_fd_relative_error = "
              << coupled_structure_fd_relative_error << '\n'
              << "coupled_directional_curvature_relative_error = "
              << coupled_directional_curvature_relative_error << '\n'
              << "response_tolerance = " << options.response_tolerance << '\n'
              << "diagonal_sample_max_relative_error = "
              << diagonal_sample_max_relative_error << '\n'
              << "diagonal_sample_worst_coordinate = "
              << diagonal_sample_worst_coordinate << '\n'
              << "diagonal_sample_worst_analytic = "
              << diagonal_sample_worst_analytic << '\n'
              << "diagonal_sample_worst_exact = "
              << diagonal_sample_worst_exact << '\n'
              << "state_average_count = " << selected_states.size() << '\n'
              << "state_energy_average_error = "
              << state_energy_average_error << '\n'
              << "state_gradient_average_inf_error = "
              << state_gradient_average_error << '\n'
              << "state_hvp_average_inf_error = "
              << state_hvp_average_error << '\n'
              << "component_decomposition_error = "
              << component_decomposition_error << '\n'
              << "coupling_batch_orbital_error = "
              << coupling_batch_orbital_error << '\n'
              << "coupling_batch_structure_error = "
              << coupling_batch_structure_error << '\n'
              << "orbital_coupling_symmetry_error = "
              << orbital_symmetry_error << '\n'
              << "relaxed_hvp_symmetry_error = "
              << relaxed_symmetry_error << '\n'
              << "coupling_batch_chunks = "
              << hvp_diagnostics.orbital_coupling_batch_chunk_count << '\n'
              << "coupling_batch_max_width = "
              << hvp_diagnostics.max_orbital_coupling_batch_width << '\n'
              << "coupled_structure_coordinate_dimension = "
              << coupled_audit.structure_coordinate_dimension << '\n'
              << "coupled_structure_response_rank = "
              << coupled_audit.structure_response_rank << '\n'
              << "coupled_adjoint_relative_error = "
              << coupled_audit.adjoint_relative_error << '\n'
              << "coupled_relaxed_hvp_relative_error = "
              << coupled_audit.relaxed_relative_error << '\n'
              << "accepted_gradient_vs_dense_midpoint_inf = "
              << accepted_gradient_vs_dense_midpoint_inf << '\n'
              << "structure_response_iterations = "
              << hvp_diagnostics.max_structure_response_iterations << '\n'
              << "structure_response_relative_residual = "
              << hvp_diagnostics.max_structure_response_relative_residual << '\n'
              << "analytic_inf_norm = " << infinity_norm(analytic) << '\n'
              << "fd_inf_norm = " << infinity_norm(finite_difference) << '\n'
              << "max_abs_diff = " << max_abs_difference << '\n'
              << "max_rel_diff = " << relative_error << '\n'
              << "analytic_directional_curvature = "
              << analytic_directional_curvature << '\n'
              << "gradient_fd_directional_curvature = "
              << gradient_fd_directional_curvature << '\n'
              << "energy_fd_directional_curvature = "
              << energy_directional_curvature << '\n'
              << "energy_curvature_max_rel_diff = "
              << energy_curvature_relative_error << '\n'
              << "core_analytic_inf_norm = " << infinity_norm(analytic_core) << '\n'
              << "core_fd_inf_norm = " << infinity_norm(finite_difference_core) << '\n'
              << "core_max_rel_diff = " << core_relative_error << '\n'
              << "core_analytic_directional_curvature = "
              << direction.dot(analytic_core) << '\n'
              << "core_gradient_fd_directional_curvature = "
              << direction.dot(finite_difference_core) << '\n'
              << "outer_analytic_inf_norm = " << infinity_norm(analytic_outer) << '\n'
              << "outer_fd_inf_norm = " << infinity_norm(finite_difference_outer) << '\n'
              << "outer_max_rel_diff = " << outer_relative_error << '\n'
              << "outer_analytic_directional_curvature = "
              << direction.dot(analytic_outer) << '\n'
              << "outer_gradient_fd_directional_curvature = "
              << direction.dot(finite_difference_outer) << '\n';
    if (relative_error > options.max_relative_error) {
      throw std::runtime_error("exact HVP finite-difference tolerance exceeded");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
