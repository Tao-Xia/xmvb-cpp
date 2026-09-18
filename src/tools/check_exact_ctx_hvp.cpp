#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/Core>

#include "input/loading/loader.hpp"
#include "vbscf/derivatives/gradient/orbital/evaluator.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "vbscf/optimization/coupled/accepted_point.hpp"
#include "vbscf/optimization/coupled/forcing.hpp"
#include "vbscf/optimization/coupled/projected_model.hpp"
#include "vbscf/optimization/coupled/projection_cache.hpp"
#include "vbscf/optimization/coupled/workspace.hpp"
#include "vbscf/optimization/krylov/minres.hpp"
#include "vbscf/optimization/trust_region/retraction.hpp"
#include "vbscf/optimization/trust_region/truncated_newton.hpp"
#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/layout.hpp"

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
  bool coupled_projection_audit = false;
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
      << " [--coupled-projection-audit 0|1]\n";
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
    } else if (name == "--coupled-projection-audit") {
      if (value != "0" && value != "1") {
        throw std::invalid_argument(
            "--coupled-projection-audit must be 0 or 1");
      }
      options.coupled_projection_audit = value == "1";
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

struct CoupledProjectionAudit {
  int orbital_subspace_size = 0;
  int response_subspace_size = 0;
  double cached_block_relative_error = 0.0;
  double cached_image_relative_error = 0.0;
  double projected_adjoint_error = 0.0;
  double response_hessian_symmetry_error = 0.0;
  double relaxed_schur_relative_error = 0.0;
  double first_radius_response_residual = 0.0;
  double second_radius_response_residual = 0.0;
  xmvb::vb::CoupledActionCounts action_counts;
  int workspace_orbital_dimension = 0;
  int workspace_response_dimension = 0;
  int workspace_expansions = 0;
  double workspace_forcing_tolerance = 0.0;
  double workspace_orbital_backward_error =
      std::numeric_limits<double>::infinity();
  double workspace_response_backward_error =
      std::numeric_limits<double>::infinity();
  double workspace_predicted_decrease =
      -std::numeric_limits<double>::infinity();
  double workspace_orbital_residual_norm =
      std::numeric_limits<double>::infinity();
  double workspace_response_residual_norm =
      std::numeric_limits<double>::infinity();
  xmvb::vb::CoupledActionCounts workspace_action_counts;
  bool workspace_repeat_zero_action = false;
};

double relative_matrix_error(
    const Eigen::Ref<const Eigen::MatrixXd>& actual,
    const Eigen::Ref<const Eigen::MatrixXd>& reference) {
  if (actual.rows() != reference.rows() ||
      actual.cols() != reference.cols()) {
    return std::numeric_limits<double>::infinity();
  }
  return (actual - reference).stableNorm() /
      std::max(1.0, reference.stableNorm());
}

CoupledProjectionAudit audit_coupled_projection(
    xmvb::vb::AcceptedPointCoupledModel model,
    const xmvb::vb::ExactHvpOperator& exact_hvp,
    const Eigen::Ref<const Eigen::VectorXd>& orbital_gradient,
    const Eigen::Ref<const Eigen::VectorXd>& primary_direction,
    const xmvb::vb::SymmetricOperatorAction&
        apply_inverse_orbital_preconditioner) {
  const xmvb::vb::CoupledNewtonOperator& direct = model.newton_operator;
  xmvb::vb::CoupledProjectionCache cache(direct);

  const int n_orbital = direct.n_orbital_coordinates();
  const int n_orbital_candidates = std::min(3, n_orbital);
  Eigen::MatrixXd orbital_candidates = Eigen::MatrixXd::Zero(
      n_orbital,
      n_orbital_candidates);
  orbital_candidates.col(0) = primary_direction;
  if (n_orbital_candidates >= 2) orbital_candidates(0, 1) = 1.0;
  if (n_orbital_candidates >= 3) {
    orbital_candidates(n_orbital - 1, 2) = 1.0;
  }
  const int accepted_orbitals =
      cache.append_orbital_block(orbital_candidates);
  const xmvb::vb::CoupledActionCounts orbital_counts =
      cache.action_counts();
  if (accepted_orbitals < std::min(2, n_orbital) ||
      orbital_counts.orbital_metric != 1 ||
      orbital_counts.orbital_hessian != 1 ||
      orbital_counts.orbital_to_response != 1 ||
      orbital_counts.response_to_orbital != 0 ||
      orbital_counts.response_hessian != 0) {
    throw std::runtime_error(
        "coupled orbital cache did not use one action per accepted block");
  }

  const int n_response = direct.n_response_coordinates();
  if (cache.append_response_block(
          Eigen::MatrixXd::Identity(n_response, n_response)) != n_response ||
      cache.action_counts().response_to_orbital != 1 ||
      cache.action_counts().response_hessian != 1) {
    throw std::runtime_error(
        "coupled response cache did not retain the complete response space");
  }

  const Eigen::MatrixXd& orbital_basis = cache.orbital_basis();
  const Eigen::MatrixXd& response_basis = cache.response_basis();
  const Eigen::MatrixXd direct_a =
      direct.apply_orbital_hessian(orbital_basis);
  const Eigen::MatrixXd direct_b =
      direct.apply_orbital_to_response(orbital_basis);
  const Eigen::MatrixXd direct_g =
      direct.apply_orbital_metric(orbital_basis);
  const Eigen::MatrixXd direct_bt =
      direct.apply_response_to_orbital(response_basis);
  const Eigen::MatrixXd direct_c =
      direct.apply_response_hessian(response_basis);
  double cached_block_error = 0.0;
  cached_block_error = std::max(
      cached_block_error,
      relative_matrix_error(cache.orbital_hessian_images(), direct_a));
  cached_block_error = std::max(
      cached_block_error,
      relative_matrix_error(cache.orbital_to_response_images(), direct_b));
  cached_block_error = std::max(
      cached_block_error,
      relative_matrix_error(cache.orbital_metric_images(), direct_g));
  cached_block_error = std::max(
      cached_block_error,
      relative_matrix_error(cache.response_to_orbital_images(), direct_bt));
  cached_block_error = std::max(
      cached_block_error,
      relative_matrix_error(cache.response_hessian_images(), direct_c));
  if (cached_block_error > 1.0e-12) {
    throw std::runtime_error(
        "coupled projection cache differs from direct production actions");
  }
  const double projected_adjoint_error =
      cache.projected_coupling_adjoint_error();
  const Eigen::MatrixXd full_projected_c =
      response_basis.transpose() * direct_c;
  const double full_response_symmetry_error = relative_matrix_error(
      full_projected_c,
      full_projected_c.transpose());
  if (projected_adjoint_error > 1.0e-10 ||
      full_response_symmetry_error > 1.0e-10) {
    throw std::runtime_error(
        "coupled production blocks are not symmetric-adjoint consistent");
  }

  const Eigen::VectorXd orbital_coordinates = Eigen::VectorXd::LinSpaced(
      cache.orbital_subspace_size(), -0.31, 0.27);
  const Eigen::VectorXd response_coordinates = Eigen::VectorXd::LinSpaced(
      cache.response_subspace_size(), 0.19, -0.23);
  constexpr double audit_shift = 0.17;
  const xmvb::vb::CoupledCachedBlocks cached_image =
      cache.reconstruct_image(
          orbital_coordinates,
          response_coordinates,
          audit_shift);
  Eigen::VectorXd full_direction(direct.size());
  full_direction.head(n_orbital) = orbital_basis * orbital_coordinates;
  full_direction.tail(n_response) = response_basis * response_coordinates;
  const Eigen::VectorXd direct_image =
      direct.apply(full_direction, audit_shift);
  const double cached_image_error = relative_matrix_error(
      cached_image.packed(),
      direct_image);
  if (cached_image_error > 1.0e-12) {
    throw std::runtime_error(
        "cached coupled image differs from a direct block action");
  }

  const Eigen::MatrixXd projected_a =
      cache.projected_orbital_hessian();
  const Eigen::MatrixXd projected_c =
      cache.projected_response_hessian();
  const Eigen::MatrixXd projected_d = cache.projected_coupling();
  const Eigen::MatrixXd projected_g = cache.projected_orbital_metric();
  const Eigen::VectorXd projected_orbital_gradient =
      orbital_basis.transpose() * orbital_gradient;
  const Eigen::VectorXd projected_response_gradient =
      response_basis.transpose() * model.structure_kkt_residual;

  const xmvb::vb::CoupledActionCounts counts_before_radius_solves =
      cache.action_counts();
  const auto first_radius = xmvb::vb::solve_coupled_projected_model(
      projected_a,
      projected_c,
      projected_d,
      projected_g,
      projected_orbital_gradient,
      projected_response_gradient,
      0.25);
  const auto second_radius = xmvb::vb::solve_coupled_projected_model(
      projected_a,
      projected_c,
      projected_d,
      projected_g,
      projected_orbital_gradient,
      projected_response_gradient,
      0.5);
  if (!first_radius.converged() || !second_radius.converged()) {
    throw std::runtime_error(
        "coupled projected trust model did not solve at both radii");
  }
  if (!(cache.action_counts() == counts_before_radius_solves)) {
    throw std::runtime_error(
        "changing the trust radius triggered a new production action");
  }

  Eigen::MatrixXd relaxed_images(
      orbital_basis.rows(), orbital_basis.cols());
  for (Eigen::Index column = 0; column < orbital_basis.cols(); ++column) {
    relaxed_images.col(column) =
        exact_hvp.apply_reduced(orbital_basis.col(column));
  }
  Eigen::MatrixXd projected_relaxed =
      orbital_basis.transpose() * relaxed_images;
  projected_relaxed =
      0.5 * (projected_relaxed + projected_relaxed.transpose());
  const double relaxed_schur_error = relative_matrix_error(
      first_radius.reduced_hessian,
      projected_relaxed);
  if (relaxed_schur_error > 2.0e-10) {
    std::cerr << std::setprecision(16)
              << "projected_relaxed_hessian =\n"
              << first_radius.reduced_hessian << '\n'
              << "direct_relaxed_hessian =\n"
              << projected_relaxed << '\n'
              << "relative_error = " << relaxed_schur_error << '\n';
    throw std::runtime_error(
        "two-space Schur model disagrees with the relaxed orbital HVP");
  }

  if (!apply_inverse_orbital_preconditioner ||
      !model.response_inverse_preconditioner) {
    throw std::runtime_error(
        "coupled workspace audit requires explicit inverse actions");
  }
  xmvb::vb::AcceptedPointCoupledWorkspace workspace(
      std::move(model),
      orbital_gradient,
      apply_inverse_orbital_preconditioner);
  const double forcing_tolerance =
      xmvb::vb::inexact_newton_forcing_term(orbital_gradient.stableNorm());
  const xmvb::vb::CoupledKktTolerances kkt_tolerances{
      forcing_tolerance,
      forcing_tolerance};
  const xmvb::vb::CoupledWorkspaceResult workspace_first = workspace.solve(
      0.25,
      kkt_tolerances);
  if (!(workspace_first.step.predicted_decrease > 0.0) ||
      workspace_first.step.orbital_backward_error > forcing_tolerance ||
      workspace_first.step.response_backward_error > forcing_tolerance) {
    std::cerr << "workspace_status = "
              << static_cast<int>(workspace_first.status) << '\n'
              << "workspace_step_status = "
              << static_cast<int>(workspace_first.step.status) << '\n'
              << "workspace_orbital_dimension = "
              << workspace_first.orbital_dimension << '\n'
              << "workspace_response_dimension = "
              << workspace_first.response_dimension << '\n'
              << "workspace_orbital_backward_error = "
              << workspace_first.step.orbital_backward_error << '\n'
              << "workspace_response_backward_error = "
              << workspace_first.step.response_backward_error << '\n'
              << "workspace_predicted_decrease = "
              << workspace_first.step.predicted_decrease << '\n';
    throw std::runtime_error(
        "accepted-point coupled workspace lacks a positive decrease or "
        "full-KKT certificate");
  }

  const xmvb::vb::CoupledActionCounts workspace_counts_before_repeat =
      workspace_first.action_counts;
  const xmvb::vb::CoupledWorkspaceResult workspace_repeat = workspace.solve(
      0.25,
      kkt_tolerances);
  if (!(workspace_repeat.step.predicted_decrease > 0.0) ||
      workspace_repeat.step.orbital_backward_error > forcing_tolerance ||
      workspace_repeat.step.response_backward_error > forcing_tolerance ||
      !(workspace_repeat.action_counts == workspace_counts_before_repeat) ||
      workspace_repeat.orbital_dimension !=
          workspace_first.orbital_dimension ||
      workspace_repeat.response_dimension !=
          workspace_first.response_dimension) {
    throw std::runtime_error(
        "repeated coupled workspace solve changed the accepted-point action cache");
  }

  return CoupledProjectionAudit{
      cache.orbital_subspace_size(),
      cache.response_subspace_size(),
      cached_block_error,
      cached_image_error,
      projected_adjoint_error,
      full_response_symmetry_error,
      relaxed_schur_error,
      first_radius.projected_response_kkt_residual_norm,
      second_radius.projected_response_kkt_residual_norm,
      cache.action_counts(),
      workspace_first.orbital_dimension,
      workspace_first.response_dimension,
      workspace_first.expansions,
      forcing_tolerance,
      workspace_first.step.orbital_backward_error,
      workspace_first.step.response_backward_error,
      workspace_first.step.predicted_decrease,
      workspace_first.step.orbital_kkt_residual.stableNorm(),
      workspace_first.step.response_kkt_residual.stableNorm(),
      workspace_first.action_counts,
      true};
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
    xmvb::vb::OrbitalChart chart(
        input.orbital_preparation_input,
        layout,
        accepted.orbital_preparation_result.auxiliary_orbital_matrix.leftCols(
            n_occupied),
        normalized_orbitals,
        &accepted.ao_effective_one_electron_result.ao_effective_h1e);
    const xmvb::vb::NonredundantRetractionMetric retraction_metric(
        chart,
        layout,
        input.orbital_preparation_input);

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
    const auto exact_hvp_owner =
        std::make_shared<xmvb::vb::ExactHvpOperator>(
            accepted.second_order_context,
            &input,
            layout,
            &chart);
    const xmvb::vb::ExactHvpOperator& exact_hvp = *exact_hvp_owner;
    xmvb::vb::AcceptedPointCoupledModel coupled_model =
        xmvb::vb::make_accepted_point_coupled_model(
            *accepted.second_order_context,
            exact_hvp_owner,
            chart.reduced_size(),
            [&retraction_metric](
                const Eigen::Ref<const Eigen::MatrixXd>& directions) {
              Eigen::MatrixXd images(
                  directions.rows(),
                  directions.cols());
              for (Eigen::Index column = 0;
                   column < directions.cols();
                   ++column) {
                images.col(column) =
                    retraction_metric.apply(directions.col(column));
              }
              return images;
            });
    const xmvb::vb::SelectedStructureDirection structure_direction =
        exact_hvp.apply_selected_structure_direction(direction);
    const int n_selected_states = static_cast<int>(
        accepted.second_order_context->selected_state_indices.size());
    const int n_structures = accepted.second_order_context->n_structures;
    if (structure_direction.delta_hamiltonian_selected.rows() != n_structures ||
        structure_direction.delta_hamiltonian_selected.cols() !=
            n_selected_states ||
        structure_direction.delta_overlap_selected.rows() != n_structures ||
        structure_direction.delta_overlap_selected.cols() !=
            n_selected_states ||
        !structure_direction.delta_hamiltonian_selected.allFinite() ||
        !structure_direction.delta_overlap_selected.allFinite()) {
      throw std::runtime_error(
          "orbital-to-structure action returned invalid selected-state images");
    }
    const xmvb::vb::SelectedSubspaceResponseLayout& response_layout =
        coupled_model.newton_operator.response_layout();
    if (coupled_model.structure_kkt_residual.size() !=
            response_layout.response_size() ||
        !coupled_model.structure_kkt_residual.allFinite()) {
      throw std::runtime_error(
          "accepted-point structure KKT residual is invalid");
    }
    const double accepted_structure_kkt_residual_norm =
        coupled_model.structure_kkt_residual.norm();
    const std::vector<xmvb::vb::SelectedStructureResponse>
        accepted_subspace_action =
            exact_hvp.apply_selected_structure_response_batch(
                {xmvb::vb::SelectedStructureResponse{
                    accepted.second_order_context
                        ->selected_state_eigenvectors,
                    Eigen::MatrixXd::Zero(
                        n_selected_states,
                        n_selected_states)}});
    std::vector<xmvb::vb::ClusterResponse> reference_kkt_clusters;
    reference_kkt_clusters.reserve(response_layout.n_clusters());
    int first_selected = 0;
    for (int cluster = 0;
         cluster < response_layout.n_clusters();
         ++cluster) {
      const int width = response_layout.cluster(cluster).n_states;
      reference_kkt_clusters.push_back(xmvb::vb::ClusterResponse{
          accepted_subspace_action.front().coefficients.middleCols(
              first_selected,
              width),
          0.5 *
              (accepted_subspace_action.front().multipliers.block(
                   first_selected,
                   first_selected,
                   width,
                   width) -
               Eigen::MatrixXd::Identity(width, width))});
      first_selected += width;
    }
    const Eigen::VectorXd reference_structure_kkt_residual =
        response_layout.pack(reference_kkt_clusters);
    const double factory_structure_kkt_error =
        (coupled_model.structure_kkt_residual -
         reference_structure_kkt_residual).norm() /
        std::max(1.0, reference_structure_kkt_residual.norm());
    if (factory_structure_kkt_error > 1.0e-12) {
      throw std::runtime_error(
          "accepted-point factory packed the structure KKT residual incorrectly");
    }
    const Eigen::MatrixXd orbital_direction = direction;
    const Eigen::MatrixXd coupling_forcing =
        xmvb::vb::apply_exact_orbital_to_selected_subspace(
            exact_hvp,
            response_layout,
            Eigen::Map<const Eigen::VectorXd>(
                accepted.second_order_context->selected_state_energies.data(),
                n_selected_states),
            accepted.second_order_context->selected_state_eigenvectors,
            orbital_direction);
    if (coupling_forcing.rows() != response_layout.response_size() ||
        coupling_forcing.cols() != 1 || !coupling_forcing.allFinite()) {
      throw std::runtime_error(
          "coupled orbital-to-subspace action returned invalid forcing");
    }
    const Eigen::VectorXd reference_forcing =
        xmvb::vb::pack_selected_subspace_forcing(
            response_layout,
            Eigen::Map<const Eigen::VectorXd>(
                accepted.second_order_context->selected_state_energies.data(),
                n_selected_states),
            accepted.second_order_context->selected_state_eigenvectors,
            structure_direction);
    const double forcing_scale = std::max(1.0, reference_forcing.norm());
    if ((coupling_forcing.col(0) - reference_forcing).norm() >
        1.0e-11 * forcing_scale) {
      throw std::runtime_error(
          "coupled orbital-to-subspace action is not reproducible");
    }
    const Eigen::MatrixXd factory_coupling =
        coupled_model.newton_operator.apply_orbital_to_response(
            orbital_direction);
    const double factory_coupling_error =
        (factory_coupling - coupling_forcing).norm() / forcing_scale;
    if (factory_coupling_error > 1.0e-12) {
      throw std::runtime_error(
          "accepted-point coupled factory changed the orbital coupling");
    }
    const Eigen::VectorXd response_probe = Eigen::VectorXd::LinSpaced(
        response_layout.response_size(), -0.3, 0.2);
    const Eigen::MatrixXd response_probe_block = response_probe;
    const Eigen::MatrixXd adjoint_image =
        xmvb::vb::apply_exact_selected_subspace_to_orbital(
            exact_hvp,
            response_layout,
            response_probe_block);
    const Eigen::MatrixXd factory_adjoint =
        coupled_model.newton_operator.apply_response_to_orbital(
            response_probe_block);
    const double factory_adjoint_error =
        (factory_adjoint - adjoint_image).norm() /
        std::max(1.0, adjoint_image.norm());
    if (factory_adjoint_error > 1.0e-12) {
      throw std::runtime_error(
          "accepted-point coupled factory changed the adjoint coupling");
    }
    Eigen::VectorXd coefficient_probe = response_probe;
    coefficient_probe.segment(
        response_layout.multiplier_offset(0),
        n_selected_states * n_selected_states).setZero();
    const Eigen::VectorXd multiplier_probe =
        response_probe - coefficient_probe;
    const Eigen::MatrixXd coefficient_adjoint =
        xmvb::vb::apply_exact_selected_subspace_to_orbital(
            exact_hvp,
            response_layout,
            Eigen::MatrixXd(coefficient_probe));
    const Eigen::MatrixXd multiplier_adjoint =
        xmvb::vb::apply_exact_selected_subspace_to_orbital(
            exact_hvp,
            response_layout,
            Eigen::MatrixXd(multiplier_probe));
    const auto relative_bilinear_error = [](double forward,
                                            double reverse) {
      return std::abs(forward - reverse) /
             std::max({1.0, std::abs(forward), std::abs(reverse)});
    };
    const double forward_bilinear = response_probe.dot(reference_forcing);
    const double reverse_bilinear = direction.dot(adjoint_image.col(0));
    const double coefficient_forward =
        coefficient_probe.dot(reference_forcing);
    const double coefficient_reverse =
        direction.dot(coefficient_adjoint.col(0));
    const double multiplier_forward = multiplier_probe.dot(reference_forcing);
    const double multiplier_reverse =
        direction.dot(multiplier_adjoint.col(0));
    const double coupling_adjoint_error =
        relative_bilinear_error(forward_bilinear, reverse_bilinear);
    const double coefficient_adjoint_error =
        relative_bilinear_error(coefficient_forward, coefficient_reverse);
    const double multiplier_adjoint_error =
        relative_bilinear_error(multiplier_forward, multiplier_reverse);
    constexpr double coupling_adjoint_tolerance = 1.0e-10;
    if (coupling_adjoint_error > coupling_adjoint_tolerance ||
        coefficient_adjoint_error > coupling_adjoint_tolerance ||
        multiplier_adjoint_error > coupling_adjoint_tolerance) {
      std::cerr << std::setprecision(16)
                << "forward_bilinear=" << forward_bilinear
                << " reverse_bilinear=" << reverse_bilinear
                << " coupling_adjoint_error=" << coupling_adjoint_error
                << " coefficient_forward=" << coefficient_forward
                << " coefficient_reverse=" << coefficient_reverse
                << " coefficient_adjoint_error="
                << coefficient_adjoint_error
                << " multiplier_forward=" << multiplier_forward
                << " multiplier_reverse=" << multiplier_reverse
                << " multiplier_adjoint_error="
                << multiplier_adjoint_error
                << '\n';
      throw std::runtime_error(
          "production orbital-structure coupling is not adjoint consistent");
    }
    Eigen::MatrixXd response_hessian_probes(
        response_layout.response_size(),
        2);
    response_hessian_probes.col(0) = Eigen::VectorXd::LinSpaced(
        response_layout.response_size(), -0.23, 0.31);
    response_hessian_probes.col(1) = Eigen::VectorXd::LinSpaced(
        response_layout.response_size(), 0.19, -0.37);
    const Eigen::MatrixXd response_hessian_images =
        xmvb::vb::apply_exact_selected_subspace_hessian(
            exact_hvp,
            response_layout,
            response_hessian_probes);
    const Eigen::MatrixXd factory_response_hessian =
        coupled_model.newton_operator.apply_response_hessian(
            response_hessian_probes);
    const double factory_response_hessian_error =
        (factory_response_hessian - response_hessian_images).norm() /
        std::max(1.0, response_hessian_images.norm());
    if (factory_response_hessian_error > 1.0e-12) {
      throw std::runtime_error(
          "accepted-point coupled factory changed the response Hessian");
    }
    const double response_hessian_symmetry_error = relative_bilinear_error(
        response_hessian_probes.col(0).dot(
            response_hessian_images.col(1)),
        response_hessian_images.col(0).dot(
            response_hessian_probes.col(1)));

    constexpr double response_hessian_tolerance = 1.0e-10;
    if (response_hessian_symmetry_error > response_hessian_tolerance) {
      throw std::runtime_error(
          "production selected-subspace Hessian failed its symmetry check");
    }
    const auto apply_response_hessian = [&](const Eigen::VectorXd& probe) {
      const Eigen::MatrixXd probe_block = probe;
      return xmvb::vb::apply_exact_selected_subspace_hessian(
                 exact_hvp,
                 response_layout,
                 probe_block)
          .col(0)
          .eval();
    };
    const Eigen::VectorXd response_rhs = -coupling_forcing.col(0);
    const xmvb::vb::MinresResult response_solve =
        xmvb::vb::solve_symmetric_minres(
            apply_response_hessian,
            response_rhs,
            {.relative_residual_tolerance = 1.0e-10,
             .absolute_residual_tolerance = 0.0,
             .maximum_iterations = 0});
    if (!response_solve.converged()) {
      throw std::runtime_error(
          "matrix-free selected-subspace response elimination did not converge");
    }
    const Eigen::MatrixXd eliminated_response = response_solve.solution;
    const double response_solve_relative_residual =
        response_solve.residual_norm /
        std::max(1.0, coupling_forcing.norm());
    if (response_solve_relative_residual > 1.0e-10) {
      throw std::runtime_error(
          "selected-subspace response elimination lacks a residual certificate");
    }
    const Eigen::VectorXd analytic = exact_hvp.apply_reduced(direction);
    const auto diagnostics_before_orbital_a = exact_hvp.diagnostics();
    const Eigen::VectorXd analytic_orbital_a =
        exact_hvp.apply_unrelaxed_orbital_hessian(direction);
    Eigen::MatrixXd orbital_a_probe(direction.size(), 2);
    orbital_a_probe.col(0) = direction;
    orbital_a_probe.col(1) = -0.375 * direction;
    const Eigen::MatrixXd analytic_orbital_a_block =
        exact_hvp.apply_unrelaxed_orbital_hessian_batch(orbital_a_probe);
    const Eigen::MatrixXd factory_orbital_a =
        coupled_model.newton_operator.apply_orbital_hessian(orbital_a_probe);
    const double factory_orbital_a_error =
        (factory_orbital_a - analytic_orbital_a_block).norm() /
        std::max(1.0, analytic_orbital_a_block.norm());
    if (factory_orbital_a_error > 1.0e-12) {
      throw std::runtime_error(
          "accepted-point coupled factory changed the orbital Hessian");
    }
    const Eigen::MatrixXd factory_metric =
        coupled_model.newton_operator.apply_orbital_metric(
            orbital_a_probe);
    Eigen::MatrixXd reference_metric(
        orbital_a_probe.rows(),
        orbital_a_probe.cols());
    for (Eigen::Index column = 0;
         column < orbital_a_probe.cols();
         ++column) {
      reference_metric.col(column) =
          retraction_metric.apply(orbital_a_probe.col(column));
    }
    const double factory_metric_error =
        (factory_metric - reference_metric).norm() /
        std::max(1.0, reference_metric.norm());
    const double factory_metric_identity_difference =
        (factory_metric - orbital_a_probe).norm() /
        std::max(1.0, factory_metric.norm());
    if (factory_metric_error > 1.0e-12) {
      throw std::runtime_error(
          "accepted-point coupled factory changed the retraction metric");
    }
    const Eigen::MatrixXd eliminated_response_adjoint =
        xmvb::vb::apply_exact_selected_subspace_to_orbital(
            exact_hvp,
            response_layout,
            eliminated_response);
    const Eigen::VectorXd schur_image =
        analytic_orbital_a + eliminated_response_adjoint.col(0);
    const double schur_identity_error = infinity_norm(
        schur_image - analytic) /
        std::max(1.0, infinity_norm(analytic));
    if (schur_identity_error > 1.0e-10) {
      throw std::runtime_error(
          "coupled response elimination disagrees with the relaxed orbital HVP");
    }
    const auto diagnostics_after_orbital_a = exact_hvp.diagnostics();
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
    const Eigen::VectorXd analytic_structure_response = exact_hvp.apply_reduced(
      direction,
      {.direct_core_response = false,
       .fixed_upstream_pullback = false,
       .local_active_response = false,
       .structure_response = true});
    const double orbital_a_decomposition_error = infinity_norm(
        analytic_orbital_a - analytic_core - analytic_local_active) /
        std::max(1.0, infinity_norm(analytic_orbital_a));
    const double relaxed_decomposition_error = infinity_norm(
        analytic - analytic_orbital_a - analytic_structure_response) /
        std::max(1.0, infinity_norm(analytic));
    const double orbital_a_block_error = std::max(
        infinity_norm(
            analytic_orbital_a_block.col(0) - analytic_orbital_a),
        infinity_norm(
            analytic_orbital_a_block.col(1) +
            0.375 * analytic_orbital_a)) /
        std::max(1.0, infinity_norm(analytic_orbital_a));
    constexpr double decomposition_tolerance = 1.0e-10;
    if (diagnostics_after_orbital_a.structure_response_block_actions !=
            diagnostics_before_orbital_a.structure_response_block_actions ||
        diagnostics_after_orbital_a
                .outer_response_eigensystem_wall_time_seconds !=
            diagnostics_before_orbital_a
                .outer_response_eigensystem_wall_time_seconds ||
        orbital_a_decomposition_error > decomposition_tolerance ||
        relaxed_decomposition_error > decomposition_tolerance ||
        orbital_a_block_error > decomposition_tolerance) {
      throw std::runtime_error(
          "coupled orbital A action has inconsistent response decomposition");
    }
    const auto hvp_diagnostics = exact_hvp.diagnostics();
    CoupledProjectionAudit coupled_projection_audit;
    if (options.coupled_projection_audit) {
      coupled_projection_audit = audit_coupled_projection(
          std::move(coupled_model),
          exact_hvp,
          accepted_reduced_gradient,
          direction,
          [&chart](const Eigen::VectorXd& residual) {
            return chart.apply_inverse_reduced_block_preconditioner(residual);
          });
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

    if (options.coupled_projection_audit) {
      const xmvb::vb::CoupledActionCounts& counts =
          coupled_projection_audit.action_counts;
      std::cout << std::setprecision(12)
                << "input = " << options.input_path << '\n'
                << "eigensolver = "
                << xmvb::vb::structure_eigensolver_name(options.eigensolver)
                << '\n'
                << "state_average_count = " << selected_states.size() << '\n'
                << "accepted_energy = "
                << accepted.scf_result.total_energy << '\n'
                << "accepted_gradient_inf_norm = "
                << infinity_norm(accepted_reduced_gradient) << '\n'
                << "factory_metric_relative_error = "
                << factory_metric_error << '\n'
                << "factory_metric_identity_difference = "
                << factory_metric_identity_difference << '\n'
                << "state_energy_average_error = "
                << state_energy_average_error << '\n'
                << "state_gradient_average_inf_error = "
                << state_gradient_average_error << '\n'
                << "state_hvp_average_inf_error = "
                << state_hvp_average_error << '\n'
                << "projection_orbital_dimension = "
                << coupled_projection_audit.orbital_subspace_size << '\n'
                << "projection_response_dimension = "
                << coupled_projection_audit.response_subspace_size << '\n'
                << "projection_cached_block_relative_error = "
                << coupled_projection_audit.cached_block_relative_error << '\n'
                << "projection_cached_image_relative_error = "
                << coupled_projection_audit.cached_image_relative_error << '\n'
                << "projection_adjoint_error = "
                << coupled_projection_audit.projected_adjoint_error << '\n'
                << "projection_response_hessian_symmetry_error = "
                << coupled_projection_audit.response_hessian_symmetry_error
                << '\n'
                << "projection_relaxed_schur_relative_error = "
                << coupled_projection_audit.relaxed_schur_relative_error << '\n'
                << "projection_first_radius_response_residual = "
                << coupled_projection_audit.first_radius_response_residual
                << '\n'
                << "projection_second_radius_response_residual = "
                << coupled_projection_audit.second_radius_response_residual
                << '\n'
                << "projection_orbital_hessian_actions = "
                << counts.orbital_hessian << '\n'
                << "projection_orbital_to_response_actions = "
                << counts.orbital_to_response << '\n'
                << "projection_response_to_orbital_actions = "
                << counts.response_to_orbital << '\n'
                << "projection_response_hessian_actions = "
                << counts.response_hessian << '\n'
                << "projection_orbital_metric_actions = "
                << counts.orbital_metric << '\n'
                << "workspace_forcing_tolerance = "
                << coupled_projection_audit.workspace_forcing_tolerance << '\n'
                << "workspace_orbital_dimension = "
                << coupled_projection_audit.workspace_orbital_dimension << '\n'
                << "workspace_response_dimension = "
                << coupled_projection_audit.workspace_response_dimension << '\n'
                << "workspace_expansions = "
                << coupled_projection_audit.workspace_expansions << '\n'
                << "workspace_orbital_backward_error = "
                << coupled_projection_audit.workspace_orbital_backward_error
                << '\n'
                << "workspace_response_backward_error = "
                << coupled_projection_audit.workspace_response_backward_error
                << '\n'
                << "workspace_orbital_residual_norm = "
                << coupled_projection_audit.workspace_orbital_residual_norm
                << '\n'
                << "workspace_response_residual_norm = "
                << coupled_projection_audit.workspace_response_residual_norm
                << '\n'
                << "workspace_predicted_decrease = "
                << coupled_projection_audit.workspace_predicted_decrease << '\n'
                << "workspace_orbital_hessian_actions = "
                << coupled_projection_audit.workspace_action_counts
                       .orbital_hessian
                << '\n'
                << "workspace_orbital_to_response_actions = "
                << coupled_projection_audit.workspace_action_counts
                       .orbital_to_response
                << '\n'
                << "workspace_response_to_orbital_actions = "
                << coupled_projection_audit.workspace_action_counts
                       .response_to_orbital
                << '\n'
                << "workspace_response_hessian_actions = "
                << coupled_projection_audit.workspace_action_counts
                       .response_hessian
                << '\n'
                << "workspace_orbital_metric_actions = "
                << coupled_projection_audit.workspace_action_counts
                       .orbital_metric
                << '\n'
                << "workspace_repeat_zero_action = "
                << (coupled_projection_audit.workspace_repeat_zero_action
                        ? "true"
                        : "false")
                << '\n';
      return 0;
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
              << "response_tolerance = " << options.response_tolerance << '\n'
              << "state_average_count = " << selected_states.size() << '\n'
              << "state_energy_average_error = "
              << state_energy_average_error << '\n'
              << "state_gradient_average_inf_error = "
              << state_gradient_average_error << '\n'
              << "state_hvp_average_inf_error = "
              << state_hvp_average_error << '\n'
              << "coupling_coefficient_adjoint_error = "
              << coefficient_adjoint_error << '\n'
              << "coupling_multiplier_adjoint_error = "
              << multiplier_adjoint_error << '\n'
              << "coupling_total_adjoint_error = "
              << coupling_adjoint_error << '\n'
              << "accepted_structure_kkt_residual_norm = "
              << accepted_structure_kkt_residual_norm << '\n'
              << "factory_structure_kkt_error = "
              << factory_structure_kkt_error << '\n'
              << "factory_orbital_a_error = "
              << factory_orbital_a_error << '\n'
              << "factory_coupling_error = "
              << factory_coupling_error << '\n'
              << "factory_adjoint_error = "
              << factory_adjoint_error << '\n'
              << "factory_response_hessian_error = "
              << factory_response_hessian_error << '\n'
              << "factory_metric_error = "
              << factory_metric_error << '\n'
              << "factory_metric_identity_difference = "
              << factory_metric_identity_difference << '\n'
              << "response_hessian_symmetry_error = "
              << response_hessian_symmetry_error << '\n'
              << "response_solve_iterations = "
              << response_solve.iterations << '\n'
              << "response_solve_relative_residual = "
              << response_solve_relative_residual << '\n'
              << "response_schur_identity_error = "
              << schur_identity_error << '\n'
              << "orbital_a_decomposition_error = "
              << orbital_a_decomposition_error << '\n'
              << "relaxed_a_plus_structure_decomposition_error = "
              << relaxed_decomposition_error << '\n'
              << "orbital_a_block_error = "
              << orbital_a_block_error << '\n'
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
