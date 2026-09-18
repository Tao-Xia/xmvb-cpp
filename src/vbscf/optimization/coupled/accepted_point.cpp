#include "vbscf/optimization/coupled/accepted_point.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "vbscf/optimization/coupled/forcing.hpp"

namespace xmvb::vb {
namespace {

bool same_weight(double left, double right) {
  const double scale = std::max({1.0, std::abs(left), std::abs(right)});
  return std::abs(left - right) <=
      64.0 * std::numeric_limits<double>::epsilon() * scale;
}

void validate_accepted_point(
    const AcceptedPointContext& accepted_point,
    const ExactHvpOperator& exact_operator,
    int n_orbital_coordinates) {
  const int n_selected = static_cast<int>(
      accepted_point.selected_state_indices.size());
  if (n_orbital_coordinates <= 0 || accepted_point.n_structures <= 0 ||
      n_selected <= 0 ||
      accepted_point.normalized_state_weights.size() !=
          static_cast<std::size_t>(n_selected) ||
      accepted_point.selected_state_energies.size() !=
          static_cast<std::size_t>(n_selected) ||
      accepted_point.selected_state_eigenvectors.rows() !=
          accepted_point.n_structures ||
      accepted_point.selected_state_eigenvectors.cols() != n_selected ||
      !accepted_point.selected_state_eigenvectors.allFinite()) {
    throw std::invalid_argument(
        "accepted coupled Newton eigensystem dimensions are inconsistent");
  }
  if (exact_operator.n_structures() != accepted_point.n_structures ||
      exact_operator.selected_state_weights().size() !=
          static_cast<std::size_t>(n_selected) ||
      !exact_operator.represents_accepted_point(accepted_point)) {
    throw std::invalid_argument(
        "exact operator does not represent the supplied accepted point");
  }

  double weight_sum = 0.0;
  for (int state = 0; state < n_selected; ++state) {
    const int root = accepted_point.selected_state_indices[state];
    const double weight = accepted_point.normalized_state_weights[state];
    const double energy = accepted_point.selected_state_energies[state];
    if (root < 0 || !std::isfinite(weight) || !(weight > 0.0) ||
        !std::isfinite(energy) ||
        !same_weight(weight, exact_operator.selected_state_weights()[state])) {
      throw std::invalid_argument(
          "accepted selected-state ordering, weights, or energies are invalid");
    }
    if (std::find(
            accepted_point.selected_state_indices.begin(),
            accepted_point.selected_state_indices.begin() + state,
            root) != accepted_point.selected_state_indices.begin() + state) {
      throw std::invalid_argument(
          "accepted selected-state ordering contains a repeated root");
    }
    weight_sum += weight;
  }
  const double normalization_tolerance =
      256.0 * std::numeric_limits<double>::epsilon() *
      std::max(1, n_selected);
  if (std::abs(weight_sum - 1.0) > normalization_tolerance) {
    throw std::invalid_argument(
        "accepted selected-state weights are not normalized");
  }
}

SelectedSubspaceResponseLayout build_response_layout(
    const AcceptedPointContext& accepted_point) {
  std::vector<SelectedStateCluster> clusters;
  for (const double weight : accepted_point.normalized_state_weights) {
    if (!clusters.empty() &&
        same_weight(clusters.back().state_weight, weight)) {
      ++clusters.back().n_states;
    } else {
      clusters.push_back(SelectedStateCluster{1, weight});
    }
  }
  return SelectedSubspaceResponseLayout(
      accepted_point.n_structures,
      std::move(clusters));
}

Eigen::VectorXd build_structure_kkt_residual(
    const AcceptedPointContext& accepted_point,
    const ExactHvpOperator& exact_operator,
    const SelectedSubspaceResponseLayout& layout) {
  const int n_selected = static_cast<int>(
      accepted_point.selected_state_indices.size());
  const SelectedStructureResponse accepted_subspace{
      accepted_point.selected_state_eigenvectors,
      Eigen::MatrixXd::Zero(n_selected, n_selected)};
  const std::vector<SelectedStructureResponse> action =
      exact_operator.apply_selected_structure_response_batch(
          {accepted_subspace});
  if (action.size() != 1) {
    throw std::runtime_error(
        "accepted selected-subspace KKT action changed its block width");
  }

  std::vector<ClusterResponse> residuals;
  residuals.reserve(static_cast<std::size_t>(layout.n_clusters()));
  int first = 0;
  for (int cluster = 0; cluster < layout.n_clusters(); ++cluster) {
    const int width = layout.cluster(cluster).n_states;
    ClusterResponse residual;
    residual.coefficients = action.front().coefficients.middleCols(
        first,
        width);
    residual.multipliers = 0.5 *
        (action.front().multipliers.block(first, first, width, width) -
         Eigen::MatrixXd::Identity(width, width));
    residuals.push_back(std::move(residual));
    first += width;
  }
  return layout.pack(residuals);
}

}  // namespace

AcceptedPointCoupledModel make_accepted_point_coupled_model(
    const AcceptedPointContext& accepted_point,
    std::shared_ptr<const ExactHvpOperator> exact_operator,
    int n_orbital_coordinates,
    CoupledBlockAction orbital_metric) {
  if (exact_operator == nullptr) {
    throw std::invalid_argument(
        "accepted coupled Newton model requires an exact operator");
  }
  validate_accepted_point(
      accepted_point,
      *exact_operator,
      n_orbital_coordinates);

  SelectedSubspaceResponseLayout layout =
      build_response_layout(accepted_point);
  const Eigen::VectorXd selected_energies = Eigen::Map<const Eigen::VectorXd>(
      accepted_point.selected_state_energies.data(),
      accepted_point.selected_state_energies.size());
  const Eigen::MatrixXd selected_eigenvectors =
      accepted_point.selected_state_eigenvectors;
  const Eigen::VectorXd structure_kkt_residual =
      build_structure_kkt_residual(
          accepted_point,
          *exact_operator,
          layout);

  CoupledNewtonActions actions;
  actions.orbital_hessian =
      [exact_operator](const Eigen::Ref<const Eigen::MatrixXd>& directions) {
        return exact_operator->apply_unrelaxed_orbital_hessian_batch(
            directions);
      };
  actions.orbital_to_response =
      [exact_operator, layout, selected_energies, selected_eigenvectors](
          const Eigen::Ref<const Eigen::MatrixXd>& directions) {
        return apply_exact_orbital_to_selected_subspace(
            *exact_operator,
            layout,
            selected_energies,
            selected_eigenvectors,
            directions);
      };
  actions.response_to_orbital =
      [exact_operator, layout](
          const Eigen::Ref<const Eigen::MatrixXd>& directions) {
        return apply_exact_selected_subspace_to_orbital(
            *exact_operator,
            layout,
            directions);
      };
  actions.response_hessian =
      [exact_operator, layout](
          const Eigen::Ref<const Eigen::MatrixXd>& directions) {
        return apply_exact_selected_subspace_hessian(
            *exact_operator,
            layout,
            directions);
      };
  actions.orbital_metric = std::move(orbital_metric);

  return AcceptedPointCoupledModel{
      exact_operator,
      CoupledNewtonOperator(
          n_orbital_coordinates,
          std::move(layout),
          std::move(actions)),
      structure_kkt_residual};
}

}  // namespace xmvb::vb
