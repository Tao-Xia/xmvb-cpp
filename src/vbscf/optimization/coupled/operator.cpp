#include "vbscf/optimization/coupled/operator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace xmvb::vb {
namespace {

int checked_cluster_index(int index, int count) {
  if (index < 0 || index >= count) {
    throw std::out_of_range("selected-state cluster index is out of range");
  }
  return index;
}

void validate_shift(double shift) {
  if (!std::isfinite(shift) || shift < 0.0) {
    throw std::invalid_argument(
        "coupled Newton orbital shift must be finite and nonnegative");
  }
}

}  // namespace

SelectedSubspaceResponseLayout::SelectedSubspaceResponseLayout(
    int n_structures,
    std::vector<SelectedStateCluster> clusters)
    : n_structures_(n_structures), clusters_(std::move(clusters)) {
  if (n_structures_ <= 0 || clusters_.empty()) {
    throw std::invalid_argument(
        "selected-subspace response layout requires positive dimensions");
  }
  coefficient_offsets_.reserve(clusters_.size());
  multiplier_offsets_.reserve(clusters_.size());
  for (const SelectedStateCluster& cluster : clusters_) {
    if (cluster.n_states <= 0 || !std::isfinite(cluster.state_weight) ||
        !(cluster.state_weight > 0.0)) {
      throw std::invalid_argument(
          "selected-state clusters require positive sizes and weights");
    }
    coefficient_offsets_.push_back(response_size_);
    response_size_ += n_structures_ * cluster.n_states;
    multiplier_offsets_.push_back(response_size_);
    response_size_ += cluster.n_states * cluster.n_states;
  }
}

int SelectedSubspaceResponseLayout::n_structures() const noexcept {
  return n_structures_;
}

int SelectedSubspaceResponseLayout::n_clusters() const noexcept {
  return static_cast<int>(clusters_.size());
}

int SelectedSubspaceResponseLayout::response_size() const noexcept {
  return response_size_;
}

const SelectedStateCluster& SelectedSubspaceResponseLayout::cluster(
    int cluster_index) const {
  return clusters_[checked_cluster_index(cluster_index, n_clusters())];
}

int SelectedSubspaceResponseLayout::coefficient_offset(
    int cluster_index) const {
  return coefficient_offsets_[
      checked_cluster_index(cluster_index, n_clusters())];
}

int SelectedSubspaceResponseLayout::multiplier_offset(
    int cluster_index) const {
  return multiplier_offsets_[
      checked_cluster_index(cluster_index, n_clusters())];
}

int SelectedSubspaceResponseLayout::cluster_size(int cluster_index) const {
  const SelectedStateCluster& selected = cluster(cluster_index);
  return n_structures_ * selected.n_states +
      selected.n_states * selected.n_states;
}

double SelectedSubspaceResponseLayout::coordinate_scale(
    int cluster_index) const {
  return std::sqrt(2.0 * cluster(cluster_index).state_weight);
}

Eigen::VectorXd SelectedSubspaceResponseLayout::pack(
    const std::vector<ClusterResponse>& cluster_responses) const {
  if (cluster_responses.size() != clusters_.size()) {
    throw std::invalid_argument(
        "cluster responses do not match the selected-subspace layout");
  }
  Eigen::VectorXd packed(response_size_);
  for (int index = 0; index < n_clusters(); ++index) {
    const int n_states = cluster(index).n_states;
    const ClusterResponse& response = cluster_responses[index];
    if (response.coefficients.rows() != n_structures_ ||
        response.coefficients.cols() != n_states ||
        response.multipliers.rows() != n_states ||
        response.multipliers.cols() != n_states ||
        !response.coefficients.allFinite() ||
        !response.multipliers.allFinite()) {
      throw std::invalid_argument(
          "cluster response matrices do not match the selected subspace");
    }
    const int n_coefficients = n_structures_ * n_states;
    const int n_multipliers = n_states * n_states;
    const double scale = coordinate_scale(index);
    packed.segment(coefficient_offset(index), n_coefficients) =
        scale * Eigen::Map<const Eigen::VectorXd>(
                    response.coefficients.data(), n_coefficients);
    packed.segment(multiplier_offset(index), n_multipliers) =
        scale * Eigen::Map<const Eigen::VectorXd>(
                    response.multipliers.data(), n_multipliers);
  }
  return packed;
}

std::vector<ClusterResponse> SelectedSubspaceResponseLayout::unpack(
    const Eigen::Ref<const Eigen::VectorXd>& response) const {
  if (response.size() != response_size_ || !response.allFinite()) {
    throw std::invalid_argument(
        "packed response does not match the selected-subspace layout");
  }
  std::vector<ClusterResponse> unpacked;
  unpacked.reserve(clusters_.size());
  for (int index = 0; index < n_clusters(); ++index) {
    const int n_states = cluster(index).n_states;
    const double inverse_scale = 1.0 / coordinate_scale(index);
    ClusterResponse cluster_response;
    cluster_response.coefficients = inverse_scale *
        Eigen::Map<const Eigen::MatrixXd>(
            response.data() + coefficient_offset(index),
            n_structures_, n_states);
    cluster_response.multipliers = inverse_scale *
        Eigen::Map<const Eigen::MatrixXd>(
            response.data() + multiplier_offset(index),
            n_states, n_states);
    unpacked.push_back(std::move(cluster_response));
  }
  return unpacked;
}

CoupledNewtonOperator::CoupledNewtonOperator(
    int n_orbital_coordinates,
    SelectedSubspaceResponseLayout response_layout,
    CoupledNewtonActions actions)
    : n_orbital_coordinates_(n_orbital_coordinates),
      response_layout_(std::move(response_layout)),
      actions_(std::move(actions)) {
  if (n_orbital_coordinates_ <= 0) {
    throw std::invalid_argument(
        "coupled Newton operator requires orbital coordinates");
  }
  if (!actions_.orbital_hessian || !actions_.orbital_to_response ||
      !actions_.response_to_orbital || !actions_.response_hessian) {
    throw std::invalid_argument(
        "coupled Newton operator requires all four block actions");
  }
}

int CoupledNewtonOperator::n_orbital_coordinates() const noexcept {
  return n_orbital_coordinates_;
}

int CoupledNewtonOperator::n_response_coordinates() const noexcept {
  return response_layout_.response_size();
}

int CoupledNewtonOperator::size() const noexcept {
  return n_orbital_coordinates() + n_response_coordinates();
}

const SelectedSubspaceResponseLayout&
CoupledNewtonOperator::response_layout() const noexcept {
  return response_layout_;
}

const CoupledActionTimings&
CoupledNewtonOperator::action_timings() const noexcept {
  return action_timings_;
}

Eigen::MatrixXd CoupledNewtonOperator::apply_checked(
    const CoupledBlockAction& action,
    const Eigen::Ref<const Eigen::MatrixXd>& directions,
    int input_rows,
    int output_rows,
    const char* label) const {
  if (directions.rows() != input_rows || !directions.allFinite()) {
    throw std::invalid_argument(
        std::string(label) + " input has inconsistent dimensions or values");
  }
  if (directions.cols() == 0) {
    return Eigen::MatrixXd(output_rows, 0);
  }
  Eigen::MatrixXd images = action(directions);
  if (images.rows() != output_rows ||
      images.cols() != directions.cols() || !images.allFinite()) {
    throw std::runtime_error(
        std::string(label) + " returned inconsistent dimensions or values");
  }
  return images;
}

Eigen::MatrixXd CoupledNewtonOperator::apply_orbital_hessian(
    const Eigen::Ref<const Eigen::MatrixXd>& directions) const {
  const auto start = std::chrono::steady_clock::now();
  Eigen::MatrixXd images = apply_checked(
      actions_.orbital_hessian,
      directions,
      n_orbital_coordinates(),
      n_orbital_coordinates(),
      "orbital Hessian action");
  action_timings_.orbital_hessian_seconds +=
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - start).count();
  return images;
}

Eigen::MatrixXd CoupledNewtonOperator::apply_orbital_to_response(
    const Eigen::Ref<const Eigen::MatrixXd>& directions) const {
  const auto start = std::chrono::steady_clock::now();
  Eigen::MatrixXd images = apply_checked(
      actions_.orbital_to_response,
      directions,
      n_orbital_coordinates(),
      n_response_coordinates(),
      "orbital-to-response action");
  action_timings_.orbital_to_response_seconds +=
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - start).count();
  return images;
}

Eigen::MatrixXd CoupledNewtonOperator::apply_response_to_orbital(
    const Eigen::Ref<const Eigen::MatrixXd>& directions) const {
  const auto start = std::chrono::steady_clock::now();
  Eigen::MatrixXd images = apply_checked(
      actions_.response_to_orbital,
      directions,
      n_response_coordinates(),
      n_orbital_coordinates(),
      "response-to-orbital action");
  action_timings_.response_to_orbital_seconds +=
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - start).count();
  return images;
}

Eigen::MatrixXd CoupledNewtonOperator::apply_response_hessian(
    const Eigen::Ref<const Eigen::MatrixXd>& directions) const {
  const auto start = std::chrono::steady_clock::now();
  Eigen::MatrixXd images = apply_checked(
      actions_.response_hessian,
      directions,
      n_response_coordinates(),
      n_response_coordinates(),
      "response Hessian action");
  action_timings_.response_hessian_seconds +=
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - start).count();
  return images;
}

Eigen::MatrixXd CoupledNewtonOperator::apply_orbital_metric(
    const Eigen::Ref<const Eigen::MatrixXd>& directions) const {
  const auto start = std::chrono::steady_clock::now();
  if (!actions_.orbital_metric) {
    if (directions.rows() != n_orbital_coordinates() ||
        !directions.allFinite()) {
      throw std::invalid_argument(
          "orbital metric input has inconsistent dimensions or values");
    }
    Eigen::MatrixXd images = directions;
    action_timings_.orbital_metric_seconds +=
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
    return images;
  }
  Eigen::MatrixXd images = apply_checked(
      actions_.orbital_metric,
      directions,
      n_orbital_coordinates(),
      n_orbital_coordinates(),
      "orbital metric action");
  action_timings_.orbital_metric_seconds +=
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - start).count();
  return images;
}

Eigen::MatrixXd CoupledNewtonOperator::apply_block(
    const Eigen::Ref<const Eigen::MatrixXd>& directions,
    double orbital_shift) const {
  validate_shift(orbital_shift);
  if (directions.rows() != size() || !directions.allFinite()) {
    throw std::invalid_argument(
        "coupled Newton directions have inconsistent dimensions or values");
  }
  if (directions.cols() == 0) {
    return Eigen::MatrixXd(size(), 0);
  }
  const auto orbital = directions.topRows(n_orbital_coordinates());
  const auto response = directions.bottomRows(n_response_coordinates());
  Eigen::MatrixXd images(size(), directions.cols());
  images.topRows(n_orbital_coordinates()) =
      apply_orbital_hessian(orbital) +
      apply_response_to_orbital(response);
  if (orbital_shift != 0.0) {
    images.topRows(n_orbital_coordinates()).noalias() +=
        orbital_shift * apply_orbital_metric(orbital);
  }
  images.bottomRows(n_response_coordinates()) =
      apply_orbital_to_response(orbital) +
      apply_response_hessian(response);
  return images;
}

Eigen::VectorXd CoupledNewtonOperator::apply(
    const Eigen::Ref<const Eigen::VectorXd>& direction,
    double orbital_shift) const {
  if (direction.size() != size()) {
    throw std::invalid_argument(
        "coupled Newton direction has inconsistent dimensions");
  }
  return apply_block(direction, orbital_shift).col(0);
}

double CoupledNewtonOperator::coupling_adjoint_error(
    const Eigen::Ref<const Eigen::VectorXd>& orbital_direction,
    const Eigen::Ref<const Eigen::VectorXd>& response_direction) const {
  if (orbital_direction.size() != n_orbital_coordinates() ||
      response_direction.size() != n_response_coordinates()) {
    throw std::invalid_argument(
        "coupling-adjoint probes have inconsistent dimensions");
  }
  const double forward = response_direction.dot(
      apply_orbital_to_response(orbital_direction).col(0));
  const double reverse = orbital_direction.dot(
      apply_response_to_orbital(response_direction).col(0));
  return std::abs(forward - reverse) /
      std::max({1.0, std::abs(forward), std::abs(reverse)});
}

}  // namespace xmvb::vb
