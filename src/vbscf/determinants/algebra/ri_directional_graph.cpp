#include "vbscf/determinants/algebra/ri_directional_graph.hpp"

#include <cmath>

#include <Eigen/LU>

#include "vbscf/determinants/algebra/contracted_density.hpp"
#include "vbscf/determinants/pairs/woodbury_overlap.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"

namespace xmvb::vb {
namespace {

int changed_position(
    const std::vector<int>& old_string,
    const std::vector<int>& new_string) {
  if (old_string.size() != new_string.size()) {
    return -1;
  }
  int changed = -1;
  for (int position = 0;
       position < static_cast<int>(old_string.size());
       ++position) {
    if (old_string[position] == new_string[position]) {
      continue;
    }
    if (changed >= 0) {
      return -1;
    }
    changed = position;
  }
  return changed;
}

bool factors_cover_strings(
    const std::vector<int>& occupied_left,
    const std::vector<int>& occupied_right,
    Eigen::Index factor_columns) {
  for (const int right : occupied_right) {
    for (const int left : occupied_left) {
      if (right < 0 || left < 0 ||
          TwoElectronIndexer::packed_pair_index(right, left) >=
              factor_columns) {
        return false;
      }
    }
  }
  return true;
}

Eigen::MatrixXd transition_block(
    const std::vector<int>& occupied_left,
    const std::vector<int>& occupied_right,
    const Eigen::Ref<const Eigen::MatrixXd>& factors,
    Eigen::Index auxiliary) {
  const int n = occupied_left.size();
  Eigen::MatrixXd result(n, n);
  for (int left = 0; left < n; ++left) {
    for (int right = 0; right < n; ++right) {
      result(right, left) = factors(
          auxiliary,
          TwoElectronIndexer::packed_pair_index(
              occupied_right[right], occupied_left[left]));
    }
  }
  return result;
}

}  // namespace

RiDirectionalGraph::RiDirectionalGraph() = default;
RiDirectionalGraph::~RiDirectionalGraph() = default;
RiDirectionalGraph::RiDirectionalGraph(RiDirectionalGraph&&) noexcept =
    default;
RiDirectionalGraph& RiDirectionalGraph::operator=(
    RiDirectionalGraph&&) noexcept = default;

bool RiDirectionalGraph::initialize(
    const std::vector<int>& occupied_left,
    const std::vector<int>& occupied_right,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_direction,
    const Eigen::Ref<const Eigen::MatrixXd>& ri_factors,
    const Eigen::Ref<const Eigen::MatrixXd>& ri_factor_direction) {
  jets_.clear();
  const int n = occupied_left.size();
  if (static_cast<int>(occupied_right.size()) != n ||
      overlap.rows() != n || overlap.cols() != n ||
      overlap_direction.rows() != n || overlap_direction.cols() != n ||
      ri_factor_direction.rows() != ri_factors.rows() ||
      ri_factor_direction.cols() != ri_factors.cols() ||
      !factors_cover_strings(
          occupied_left, occupied_right, ri_factors.cols())) {
    return false;
  }
  const Eigen::FullPivLU<Eigen::MatrixXd> decomposition(overlap);
  if (!decomposition.isInvertible()) {
    return false;
  }
  Eigen::MatrixXd inverse = decomposition.inverse();
  if (!is_certified_regular_overlap(overlap, inverse)) {
    return false;
  }
  const double determinant = overlap.determinant();
  const Eigen::MatrixXd inverse_direction =
      -inverse * overlap_direction * inverse;
  const double determinant_direction =
      determinant * (inverse * overlap_direction).trace();

  jets_.reserve(ri_factors.rows());
  for (Eigen::Index auxiliary = 0;
       auxiliary < ri_factors.rows();
       ++auxiliary) {
    jets_.push_back(std::make_unique<ContractedDensityJet>(
        inverse,
        transition_block(
            occupied_left, occupied_right, ri_factors, auxiliary),
        inverse_direction,
        transition_block(
            occupied_left,
            occupied_right,
            ri_factor_direction,
            auxiliary),
        2));
  }
  occupied_left_ = occupied_left;
  occupied_right_ = occupied_right;
  overlap_ = overlap;
  overlap_direction_ = overlap_direction;
  inverse_ = std::move(inverse);
  inverse_direction_ = std::move(inverse_direction);
  determinant_ = determinant;
  determinant_direction_ = determinant_direction;
  return true;
}

bool RiDirectionalGraph::update_right(
    const std::vector<int>& occupied_right_new,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_new,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_direction_new,
    const Eigen::Ref<const Eigen::MatrixXd>& ri_factors,
    const Eigen::Ref<const Eigen::MatrixXd>& ri_factor_direction) {
  if (!valid() ||
      overlap_new.rows() != inverse_.rows() ||
      overlap_new.cols() != inverse_.cols() ||
      overlap_direction_new.rows() != inverse_.rows() ||
      overlap_direction_new.cols() != inverse_.cols() ||
      ri_factors.rows() != static_cast<Eigen::Index>(jets_.size()) ||
      ri_factor_direction.rows() != ri_factors.rows() ||
      ri_factor_direction.cols() != ri_factors.cols() ||
      !factors_cover_strings(
          occupied_left_, occupied_right_new, ri_factors.cols())) {
    return false;
  }
  const int row = changed_position(occupied_right_, occupied_right_new);
  if (row < 0) {
    return false;
  }

  const Eigen::VectorXd overlap_delta =
      overlap_new.row(row).transpose() - overlap_.row(row).transpose();
  const Eigen::VectorXd overlap_delta_direction =
      overlap_direction_new.row(row).transpose() -
      overlap_direction_.row(row).transpose();
  const Eigen::VectorXd inverse_column = inverse_.col(row);
  const Eigen::VectorXd inverse_column_direction =
      inverse_direction_.col(row);
  const double denominator = 1.0 + overlap_delta.dot(inverse_column);
  const double denominator_direction =
      overlap_delta_direction.dot(inverse_column) +
      overlap_delta.dot(inverse_column_direction);
  if (denominator == 0.0 || !std::isfinite(denominator)) {
    return false;
  }

  const Eigen::VectorXd channel_left = inverse_column / denominator;
  const Eigen::VectorXd channel_left_direction =
      inverse_column_direction / denominator -
      inverse_column *
          (denominator_direction / (denominator * denominator));
  const Eigen::VectorXd inverse_left = -channel_left;
  const Eigen::VectorXd inverse_left_direction =
      -channel_left_direction;
  const Eigen::VectorXd inverse_right =
      inverse_.transpose() * overlap_delta;
  const Eigen::VectorXd inverse_right_direction =
      inverse_direction_.transpose() * overlap_delta +
      inverse_.transpose() * overlap_delta_direction;

  for (Eigen::Index auxiliary = 0;
       auxiliary < static_cast<Eigen::Index>(jets_.size());
       ++auxiliary) {
    Eigen::VectorXd transition_delta(occupied_left_.size());
    Eigen::VectorXd transition_delta_direction(occupied_left_.size());
    for (int left = 0;
         left < static_cast<int>(occupied_left_.size());
         ++left) {
      const int old_pair = TwoElectronIndexer::packed_pair_index(
          occupied_right_[row], occupied_left_[left]);
      const int new_pair = TwoElectronIndexer::packed_pair_index(
          occupied_right_new[row], occupied_left_[left]);
      transition_delta(left) =
          ri_factors(auxiliary, new_pair) -
          ri_factors(auxiliary, old_pair);
      transition_delta_direction(left) =
          ri_factor_direction(auxiliary, new_pair) -
          ri_factor_direction(auxiliary, old_pair);
    }
    const Eigen::MatrixXd& accepted = jets_[auxiliary]->value().channel();
    const Eigen::MatrixXd& directional =
        jets_[auxiliary]->channel_direction();
    const Eigen::VectorXd channel_right =
        transition_delta - accepted.transpose() * overlap_delta;
    const Eigen::VectorXd channel_right_direction =
        transition_delta_direction -
        directional.transpose() * overlap_delta -
        accepted.transpose() * overlap_delta_direction;
    jets_[auxiliary]->update(
        inverse_left,
        inverse_right,
        inverse_left_direction,
        inverse_right_direction,
        channel_left,
        channel_right,
        channel_left_direction,
        channel_right_direction);
  }

  inverse_.noalias() += inverse_left * inverse_right.transpose();
  inverse_direction_.noalias() +=
      inverse_left_direction * inverse_right.transpose() +
      inverse_left * inverse_right_direction.transpose();
  determinant_direction_ =
      determinant_direction_ * denominator +
      determinant_ * denominator_direction;
  determinant_ *= denominator;
  overlap_ = overlap_new;
  overlap_direction_ = overlap_direction_new;
  occupied_right_ = occupied_right_new;
  return inverse_.allFinite() && inverse_direction_.allFinite();
}

RiDirectionalTileValue RiDirectionalGraph::value() const {
  RiDirectionalTileValue result;
  const int n = inverse_.rows();
  result.overlap_gradient_direction = Eigen::MatrixXd::Zero(n, n);
  result.accepted_first_contractions.resize(jets_.size());
  result.directional_first_contractions.resize(jets_.size());
  for (Eigen::Index auxiliary = 0;
       auxiliary < static_cast<Eigen::Index>(jets_.size());
       ++auxiliary) {
    const auto& jet = *jets_[auxiliary];
    result.two_electron += jet.value().contraction(2, determinant_);
    result.two_electron_direction += jet.contraction_direction(
        2, determinant_, determinant_direction_);
    result.overlap_gradient_direction.noalias() +=
        jet.overlap_gradient_direction(
            2, determinant_, determinant_direction_);
    result.accepted_first_contractions(auxiliary) =
        jet.value().contraction(1, determinant_);
    result.directional_first_contractions(auxiliary) =
        jet.contraction_direction(
            1, determinant_, determinant_direction_);
  }
  return result;
}

}  // namespace xmvb::vb
