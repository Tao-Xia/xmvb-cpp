#include "vbscf/optimization/coupled/forcing.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace xmvb::vb {
namespace {

int selected_state_count(const SelectedSubspaceResponseLayout& layout) {
  int count = 0;
  for (int cluster = 0; cluster < layout.n_clusters(); ++cluster) {
    count += layout.cluster(cluster).n_states;
  }
  return count;
}

bool same_weight(double left, double right) {
  const double scale = std::max({1.0, std::abs(left), std::abs(right)});
  return std::abs(left - right) <=
      64.0 * std::numeric_limits<double>::epsilon() * scale;
}

void validate_layout_against_accepted_point(
    const ExactHvpOperator& exact_operator,
    const SelectedSubspaceResponseLayout& layout) {
  const std::vector<double>& accepted_weights =
      exact_operator.selected_state_weights();
  if (layout.n_structures() != exact_operator.n_structures() ||
      selected_state_count(layout) !=
          static_cast<int>(accepted_weights.size())) {
    throw std::invalid_argument(
        "selected-subspace layout does not match the accepted point");
  }

  int first = 0;
  for (int cluster = 0; cluster < layout.n_clusters(); ++cluster) {
    const SelectedStateCluster& descriptor = layout.cluster(cluster);
    if (first > 0 && same_weight(
                         accepted_weights[first - 1],
                         accepted_weights[first])) {
      throw std::invalid_argument(
          "equal-weight accepted states must share one response cluster");
    }
    for (int state = 0; state < descriptor.n_states; ++state) {
      const double accepted_weight = accepted_weights[first + state];
      if (!std::isfinite(accepted_weight) ||
          !same_weight(accepted_weight, descriptor.state_weight)) {
        throw std::invalid_argument(
            "selected-subspace cluster order or weight does not match the accepted point");
      }
    }
    if (first + descriptor.n_states <
            static_cast<int>(accepted_weights.size()) &&
        same_weight(
            accepted_weights[first + descriptor.n_states - 1],
            accepted_weights[first + descriptor.n_states])) {
      throw std::invalid_argument(
          "equal-weight accepted states must share one response cluster");
    }
    first += descriptor.n_states;
  }
}

void validate_inputs(
    const SelectedSubspaceResponseLayout& layout,
    const Eigen::Ref<const Eigen::VectorXd>& selected_energies,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const SelectedStructureDirection& direction) {
  const int n_selected = selected_state_count(layout);
  if (selected_energies.size() != n_selected ||
      selected_eigenvectors.rows() != layout.n_structures() ||
      selected_eigenvectors.cols() != n_selected ||
      direction.delta_hamiltonian_selected.rows() != layout.n_structures() ||
      direction.delta_hamiltonian_selected.cols() != n_selected ||
      direction.delta_overlap_selected.rows() != layout.n_structures() ||
      direction.delta_overlap_selected.cols() != n_selected ||
      !selected_energies.allFinite() ||
      !selected_eigenvectors.allFinite() ||
      !direction.delta_hamiltonian_selected.allFinite() ||
      !direction.delta_overlap_selected.allFinite()) {
    throw std::invalid_argument(
        "selected-subspace forcing inputs have inconsistent dimensions or values");
  }
}

}  // namespace

Eigen::VectorXd pack_selected_subspace_forcing(
    const SelectedSubspaceResponseLayout& layout,
    const Eigen::Ref<const Eigen::VectorXd>& selected_energies,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const SelectedStructureDirection& direction) {
  validate_inputs(
      layout,
      selected_energies,
      selected_eigenvectors,
      direction);
  std::vector<ClusterResponse> cluster_forcing;
  cluster_forcing.reserve(static_cast<std::size_t>(layout.n_clusters()));
  int first = 0;
  for (int cluster = 0; cluster < layout.n_clusters(); ++cluster) {
    const int width = layout.cluster(cluster).n_states;
    const auto vectors = selected_eigenvectors.middleCols(first, width);
    const auto delta_overlap =
        direction.delta_overlap_selected.middleCols(first, width);
    ClusterResponse forcing;
    forcing.coefficients =
        direction.delta_hamiltonian_selected.middleCols(first, width) -
        delta_overlap * selected_energies.segment(first, width).asDiagonal();
    const Eigen::MatrixXd metric_direction =
        vectors.transpose() * delta_overlap;
    forcing.multipliers =
        0.25 * (metric_direction + metric_direction.transpose());
    cluster_forcing.push_back(std::move(forcing));
    first += width;
  }
  return layout.pack(cluster_forcing);
}

Eigen::MatrixXd pack_selected_subspace_forcing_block(
    const SelectedSubspaceResponseLayout& layout,
    const Eigen::Ref<const Eigen::VectorXd>& selected_energies,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const std::vector<SelectedStructureDirection>& directions) {
  Eigen::MatrixXd forcing(layout.response_size(), directions.size());
  for (std::size_t column = 0; column < directions.size(); ++column) {
    forcing.col(static_cast<Eigen::Index>(column)) =
        pack_selected_subspace_forcing(
            layout,
            selected_energies,
            selected_eigenvectors,
            directions[column]);
  }
  return forcing;
}

Eigen::MatrixXd apply_exact_orbital_to_selected_subspace(
    const ExactHvpOperator& exact_operator,
    const SelectedSubspaceResponseLayout& layout,
    const Eigen::Ref<const Eigen::VectorXd>& selected_energies,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& orbital_directions) {
  validate_layout_against_accepted_point(exact_operator, layout);
  return pack_selected_subspace_forcing_block(
      layout,
      selected_energies,
      selected_eigenvectors,
      exact_operator.apply_selected_structure_direction_batch(
          orbital_directions));
}

Eigen::MatrixXd apply_exact_selected_subspace_to_orbital(
    const ExactHvpOperator& exact_operator,
    const SelectedSubspaceResponseLayout& layout,
    const Eigen::Ref<const Eigen::MatrixXd>& response_directions) {
  validate_layout_against_accepted_point(exact_operator, layout);
  if (response_directions.rows() != layout.response_size() ||
      response_directions.cols() <= 0 ||
      !response_directions.allFinite()) {
    throw std::invalid_argument(
        "selected-subspace adjoint directions have inconsistent dimensions or values");
  }

  Eigen::MatrixXd orbital_images;
  for (Eigen::Index column = 0;
       column < response_directions.cols();
       ++column) {
    const std::vector<ClusterResponse> clusters =
        layout.unpack(response_directions.col(column));
    Eigen::MatrixXd coefficient_response(
        layout.n_structures(),
        selected_state_count(layout));
    Eigen::MatrixXd state_multipliers = Eigen::MatrixXd::Zero(
        selected_state_count(layout),
        selected_state_count(layout));
    int first = 0;
    for (int cluster = 0; cluster < layout.n_clusters(); ++cluster) {
      const int width = layout.cluster(cluster).n_states;
      coefficient_response.middleCols(first, width) =
          clusters[cluster].coefficients;
      state_multipliers.block(first, first, width, width) =
          clusters[cluster].multipliers;
      first += width;
    }

    const Eigen::VectorXd image =
        exact_operator.apply_structure_response_adjoint(
            coefficient_response,
            state_multipliers);
    if (column == 0) {
      orbital_images.resize(image.size(), response_directions.cols());
    } else if (image.size() != orbital_images.rows()) {
      throw std::runtime_error(
          "selected-subspace adjoint changed its orbital dimension");
    }
    orbital_images.col(column) = image;
  }
  return orbital_images;
}

Eigen::MatrixXd apply_exact_selected_subspace_hessian(
    const ExactHvpOperator& exact_operator,
    const SelectedSubspaceResponseLayout& layout,
    const Eigen::Ref<const Eigen::MatrixXd>& response_directions) {
  validate_layout_against_accepted_point(exact_operator, layout);
  if (response_directions.rows() != layout.response_size() ||
      response_directions.cols() <= 0 ||
      !response_directions.allFinite()) {
    throw std::invalid_argument(
        "selected-subspace Hessian directions have inconsistent dimensions or values");
  }

  const int n_selected = selected_state_count(layout);
  std::vector<SelectedStructureResponse> physical_responses;
  physical_responses.reserve(
      static_cast<std::size_t>(response_directions.cols()));
  for (Eigen::Index column = 0;
       column < response_directions.cols();
       ++column) {
    const std::vector<ClusterResponse> clusters =
        layout.unpack(response_directions.col(column));
    SelectedStructureResponse response{
        Eigen::MatrixXd(layout.n_structures(), n_selected),
        Eigen::MatrixXd::Zero(n_selected, n_selected)};
    int first = 0;
    for (int cluster = 0; cluster < layout.n_clusters(); ++cluster) {
      const int width = layout.cluster(cluster).n_states;
      response.coefficients.middleCols(first, width) =
          clusters[cluster].coefficients;
      response.multipliers.block(first, first, width, width) =
          clusters[cluster].multipliers;
      first += width;
    }
    physical_responses.push_back(std::move(response));
  }

  const std::vector<SelectedStructureResponse> physical_images =
      exact_operator.apply_selected_structure_response_batch(
          physical_responses);
  if (physical_images.size() != physical_responses.size()) {
    throw std::runtime_error(
        "selected-subspace Hessian changed its block width");
  }

  Eigen::MatrixXd images(layout.response_size(), response_directions.cols());
  for (std::size_t column = 0; column < physical_images.size(); ++column) {
    std::vector<ClusterResponse> clusters;
    clusters.reserve(static_cast<std::size_t>(layout.n_clusters()));
    int first = 0;
    for (int cluster = 0; cluster < layout.n_clusters(); ++cluster) {
      const int width = layout.cluster(cluster).n_states;
      clusters.push_back(ClusterResponse{
          physical_images[column].coefficients.middleCols(first, width),
          physical_images[column].multipliers.block(
              first,
              first,
              width,
              width)});
      first += width;
    }
    images.col(static_cast<Eigen::Index>(column)) = layout.pack(clusters);
  }
  return images;
}

}  // namespace xmvb::vb
