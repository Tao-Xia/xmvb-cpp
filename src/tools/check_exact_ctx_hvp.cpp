#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/Core>

#include "input/loading/loader.hpp"
#include "vbscf/derivatives/gradient/orbital/evaluator.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/layout.hpp"

namespace {

struct Options {
  std::string input_path;
  double step = 1.0e-4;
  double max_relative_error = std::numeric_limits<double>::infinity();
  double response_tolerance = 1.0e-3;
  bool stream_pair_products = false;
};

void print_usage() {
  std::cerr
      << "usage: check_exact_ctx_hvp <input.xmi>"
      << " [--step h]"
      << " [--max-rel-error tolerance]"
      << " [--response-tolerance tolerance]"
      << " [--stream-pair-products 0|1]\n";
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
    if (name == "--step") {
      options.step = std::stod(value);
    } else if (name == "--max-rel-error") {
      options.max_relative_error = std::stod(value);
    } else if (name == "--response-tolerance") {
      options.response_tolerance = std::stod(value);
    } else if (name == "--stream-pair-products") {
      if (value != "0" && value != "1") {
        throw std::invalid_argument("--stream-pair-products must be 0 or 1");
      }
      options.stream_pair_products = value == "1";
    } else if (name == "--probe") {
      if (value != "full") {
        throw std::invalid_argument("only the complete exact HVP is supported");
      }
    } else {
      throw std::invalid_argument("unknown argument: " + name);
    }
  }
  if (!(options.step > 0.0) || !(options.max_relative_error >= 0.0) ||
      !(options.response_tolerance > 0.0)) {
    throw std::invalid_argument("finite-difference tolerances must be non-negative");
  }
  return options;
}

double infinity_norm(const Eigen::Ref<const Eigen::VectorXd>& values) {
  return values.size() == 0 ? 0.0 : values.cwiseAbs().maxCoeff();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    xmvb::vb::VbScfInputLoadOptions load_options;
    load_options.standard_two_electron_mode =
        xmvb::vb::StandardTwoElectronMode::Exact;
    const auto loaded =
        xmvb::vb::load_vbscf_input_with_timings(options.input_path, load_options);
    xmvb::vb::VbScfInput input = loaded.input;

    xmvb::vb::OrbitalGradientEvaluator evaluator;
    const Eigen::MatrixXd no_initial_eigenvectors;
    auto accepted = evaluator.evaluate_without_reference_energy_gradient(
        input,
        {0},
        {1.0},
        loaded.nuclear_repulsion_energy,
        xmvb::vb::StructureEigensolver::Dense,
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
    xmvb::vb::OrbitalChart chart(
        input.orbital_preparation_input,
        layout,
        accepted.orbital_preparation_result.auxiliary_orbital_matrix.leftCols(
            n_occupied),
        normalized_orbitals,
        &accepted.ao_effective_one_electron_result.ao_effective_h1e);

    const Eigen::VectorXd packed_gradient =
        layout.gather_from_full(accepted.sparse_orbital_energy_gradient);
    Eigen::VectorXd direction = chart.project_reduced_gradient(packed_gradient);
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
      accepted.second_order_context->prepared_active_space
          .active_space_two_electron_result.dense_ao_pair_products.resize(0, 0);
    }
    xmvb::vb::ExactHvpOperator exact_hvp(
        accepted.second_order_context,
        &input,
        layout,
        &chart);
    const Eigen::VectorXd analytic = exact_hvp.apply_reduced(direction);
    const Eigen::VectorXd analytic_core = exact_hvp.apply_reduced(
        direction,
        {.direct_core_response = true,
         .fixed_upstream_pullback = true,
         .outer_response = false});
    const auto hvp_diagnostics = exact_hvp.diagnostics();

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
              << "stream_pair_products = "
              << (options.stream_pair_products ? "true" : "false") << '\n'
              << "reduced_dimension = " << direction.size() << '\n'
              << "finite_difference_step = " << options.step << '\n'
              << "response_tolerance = " << options.response_tolerance << '\n'
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
