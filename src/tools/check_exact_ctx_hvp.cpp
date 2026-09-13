#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

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
  bool stream_pair_products = false;
};

void print_usage() {
  std::cerr
      << "usage: check_exact_ctx_hvp <input.xmi>"
      << " [--step h]"
      << " [--max-rel-error tolerance]"
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
  if (!(options.step > 0.0) || !(options.max_relative_error >= 0.0)) {
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
    auto accepted = evaluator.evaluate_without_reference_energy_gradient(
        input,
        {0},
        {1.0},
        loaded.nuclear_repulsion_energy);
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

    xmvb::vb::VbScfInput plus = input;
    xmvb::vb::VbScfInput minus = input;
    plus.orbital_preparation_input = chart.retract_step(
        input.orbital_preparation_input,
        direction,
        options.step);
    minus.orbital_preparation_input = chart.retract_step(
        input.orbital_preparation_input,
        direction,
        -options.step);
    const auto plus_gradient =
        evaluator.evaluate_without_reference_energy_gradient(
            plus,
            accepted.second_order_context->selected_state_indices,
            accepted.second_order_context->normalized_state_weights,
            loaded.nuclear_repulsion_energy);
    const auto minus_gradient =
        evaluator.evaluate_without_reference_energy_gradient(
            minus,
            accepted.second_order_context->selected_state_indices,
            accepted.second_order_context->normalized_state_weights,
            loaded.nuclear_repulsion_energy);
    const Eigen::VectorXd plus_reduced = chart.project_reduced_gradient(
        layout.gather_from_full(plus_gradient.sparse_orbital_energy_gradient));
    const Eigen::VectorXd minus_reduced = chart.project_reduced_gradient(
        layout.gather_from_full(minus_gradient.sparse_orbital_energy_gradient));
    const Eigen::VectorXd finite_difference =
        (plus_reduced - minus_reduced) / (2.0 * options.step);

    const double max_abs_difference = infinity_norm(analytic - finite_difference);
    const double relative_error =
        max_abs_difference / std::max(1.0, infinity_norm(finite_difference));
    std::cout << std::setprecision(12)
              << "input = " << options.input_path << '\n'
              << "stream_pair_products = "
              << (options.stream_pair_products ? "true" : "false") << '\n'
              << "reduced_dimension = " << direction.size() << '\n'
              << "finite_difference_step = " << options.step << '\n'
              << "analytic_inf_norm = " << infinity_norm(analytic) << '\n'
              << "fd_inf_norm = " << infinity_norm(finite_difference) << '\n'
              << "max_abs_diff = " << max_abs_difference << '\n'
              << "max_rel_diff = " << relative_error << '\n';
    if (relative_error > options.max_relative_error) {
      throw std::runtime_error("exact HVP finite-difference tolerance exceeded");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
