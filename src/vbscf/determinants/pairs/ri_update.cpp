#include "vbscf/determinants/pairs/ri_update.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "vbscf/determinants/pairs/contractions.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"

namespace xmvb::vb {
namespace {

int changed_position(
    const std::vector<int>& old_string,
    const std::vector<int>& new_string) {
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

bool determinant_update_is_consistent(
    double old_determinant,
    double update_factor,
    double new_determinant) {
  const double predicted = old_determinant * update_factor;
  const double scale = std::max(
      std::numeric_limits<double>::min(),
      std::abs(predicted) + std::abs(new_determinant));
  return std::isfinite(predicted) &&
      std::abs(predicted - new_determinant) <=
          std::sqrt(std::numeric_limits<double>::epsilon()) * scale;
}

double max_abs(const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  return matrix.size() == 0 ? 0.0 : matrix.cwiseAbs().maxCoeff();
}

bool factors_cover_occupied_pairs(
    const std::vector<int>& occupied_left,
    const std::vector<int>& occupied_right,
    Eigen::Index n_factor_columns) {
  for (const int right : occupied_right) {
    for (const int left : occupied_left) {
      if (right < 0 || left < 0 ||
          TwoElectronIndexer::packed_pair_index(right, left) >=
              n_factor_columns) {
        return false;
      }
    }
  }
  return true;
}

double matrix_product_roundoff_factor(int inner_dimension) {
  const double scaled_epsilon =
      inner_dimension * std::numeric_limits<double>::epsilon();
  return scaled_epsilon < 1.0
      ? scaled_epsilon / (1.0 - scaled_epsilon)
      : std::numeric_limits<double>::infinity();
}

double channel_scalar_error_bound(
    const Eigen::Ref<const Eigen::MatrixXd>& channel,
    double element_error,
    double* channel_norm_output = nullptr) {
  const double n = static_cast<double>(channel.rows());
  const double channel_norm = max_abs(channel);
  if (channel_norm_output != nullptr) {
    *channel_norm_output = channel_norm;
  }
  const double trace_norm = channel.diagonal().cwiseAbs().sum();
  return n * trace_norm * element_error +
      n * n * channel_norm * element_error +
      n * n * element_error * element_error;
}

double channel_response_error_bound(
    const Eigen::Ref<const Eigen::MatrixXd>& channel,
    double element_error) {
  const double n = static_cast<double>(channel.rows());
  const double channel_norm = max_abs(channel);
  const double trace_norm = channel.diagonal().cwiseAbs().sum();
  return (trace_norm + 3.0 * n * channel_norm) * element_error +
      2.0 * n * element_error * element_error;
}

}  // namespace

bool RiPairUpdateState::initialize(
    const std::vector<int>& occupied_left,
    const std::vector<int>& occupied_right,
    const DeterminantOverlapResult& overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& ri_factors,
    bool track_response) {
  reset();
  const int n_electrons = static_cast<int>(occupied_left.size());
  if (static_cast<int>(occupied_right.size()) != n_electrons ||
      overlap.nullity != 0 || overlap.overlap_determinant == 0.0 ||
      overlap.inverse_overlap_submatrix.rows() != n_electrons ||
      overlap.inverse_overlap_submatrix.cols() != n_electrons ||
      !factors_cover_occupied_pairs(
          occupied_left, occupied_right, ri_factors.cols())) {
    return false;
  }

  n_electrons_ = n_electrons;
  occupied_left_ = occupied_left;
  occupied_right_ = occupied_right;
  track_response_ = track_response;
  if (track_response_) {
    response_aggregate_ = Eigen::MatrixXd::Zero(n_electrons, n_electrons);
  }
  channels_.resize(
      ri_factors.rows(),
      static_cast<Eigen::Index>(n_electrons) * n_electrons);
  channel_error_bounds_.assign(
      static_cast<std::size_t>(ri_factors.rows()), 0.0);
  Eigen::MatrixXd transition(n_electrons, n_electrons);
  const double epsilon = std::numeric_limits<double>::epsilon();
  const double gamma = matrix_product_roundoff_factor(n_electrons);
  for (Eigen::Index auxiliary = 0;
       auxiliary < ri_factors.rows();
       ++auxiliary) {
    for (int left = 0; left < n_electrons; ++left) {
      for (int right = 0; right < n_electrons; ++right) {
        transition(right, left) = ri_factors(
            auxiliary,
            TwoElectronIndexer::packed_pair_index(
                occupied_right[right], occupied_left[left]));
      }
    }
    Eigen::Map<Eigen::MatrixXd> channel(
        channels_.data() + auxiliary * n_electrons * n_electrons,
        n_electrons,
        n_electrons);
    channel.noalias() = overlap.inverse_overlap_submatrix * transition;
    if (track_response_) {
      const double trace = channel.trace();
      const Eigen::MatrixXd channel_square = channel * channel;
      const Eigen::MatrixXd channel_abs = channel.cwiseAbs();
      const double trace_scale =
          channel.diagonal().cwiseAbs().sum() * max_abs(channel);
      const double square_scale = max_abs(channel_abs * channel_abs);
      const double aggregate_norm = max_abs(response_aggregate_);
      response_aggregate_.noalias() += trace * channel - channel_square;
      response_roundoff_bound_ +=
          (gamma + 4.0 * epsilon) * (trace_scale + square_scale) +
          epsilon * aggregate_norm;
    }
  }
  const double response_scale = track_response_
      ? std::max(
            std::numeric_limits<double>::min(),
            max_abs(response_aggregate_))
      : 1.0;
  valid_ = channels_.allFinite() &&
      (!track_response_ ||
       (response_aggregate_.allFinite() &&
        std::isfinite(response_roundoff_bound_) &&
        response_roundoff_bound_ <=
            std::sqrt(epsilon) * response_scale));
  if (!valid_) {
    reset();
  } else {
    contracted_channels_certified_ = true;
  }
  return valid_;
}

bool RiPairUpdateState::update_right(
    const std::vector<int>& occupied_left,
    const std::vector<int>& occupied_right_old,
    const std::vector<int>& occupied_right_new,
    const DeterminantOverlapResult& overlap_old,
    const DeterminantOverlapResult& overlap_new,
    const Eigen::Ref<const Eigen::MatrixXd>& ri_factors) {
  if (!valid_ || occupied_left != occupied_left_ ||
      occupied_right_old != occupied_right_ ||
      occupied_right_new.size() != occupied_right_old.size() ||
      overlap_old.overlap_submatrix.rows() != n_electrons_ ||
      overlap_old.overlap_submatrix.cols() != n_electrons_ ||
      overlap_old.inverse_overlap_submatrix.rows() != n_electrons_ ||
      overlap_old.inverse_overlap_submatrix.cols() != n_electrons_ ||
      overlap_new.overlap_submatrix.rows() != n_electrons_ ||
      overlap_new.overlap_submatrix.cols() != n_electrons_ ||
      overlap_new.inverse_overlap_submatrix.rows() != n_electrons_ ||
      overlap_new.inverse_overlap_submatrix.cols() != n_electrons_ ||
      ri_factors.rows() != channels_.rows() ||
      channel_error_bounds_.size() !=
          static_cast<std::size_t>(channels_.rows()) ||
      !factors_cover_occupied_pairs(
          occupied_left, occupied_right_old, ri_factors.cols()) ||
      !factors_cover_occupied_pairs(
          occupied_left, occupied_right_new, ri_factors.cols())) {
    return false;
  }
  const int row = changed_position(occupied_right_old, occupied_right_new);
  if (row < 0) {
    return false;
  }

  const Eigen::VectorXd overlap_delta =
      overlap_new.overlap_submatrix.row(row).transpose() -
      overlap_old.overlap_submatrix.row(row).transpose();
  const Eigen::VectorXd old_inverse_column =
      overlap_old.inverse_overlap_submatrix.col(row);
  const double update_factor = 1.0 + overlap_delta.dot(old_inverse_column);
  if (update_factor == 0.0 || !std::isfinite(update_factor) ||
      !determinant_update_is_consistent(
          overlap_old.overlap_determinant,
          update_factor,
          overlap_new.overlap_determinant)) {
    return false;
  }

  // For X' = X + e_r v^T, P'e_r = Pe_r/(1 + v^TPe_r).
  // Taking the column from the already certified P' avoids repeating that
  // division and keeps the channel update consistent with the stored inverse.
  const Eigen::VectorXd update_column =
      overlap_new.inverse_overlap_submatrix.col(row);
  const double epsilon = std::numeric_limits<double>::epsilon();
  const double gamma = matrix_product_roundoff_factor(n_electrons_);
  const double overlap_delta_error = epsilon * (
      overlap_new.overlap_submatrix.row(row).cwiseAbs().sum() +
      overlap_old.overlap_submatrix.row(row).cwiseAbs().sum());
  double phi_error_bound = 0.0;
  double response_channel_error_bound = 0.0;
  double response_update_scale = 0.0;
  bool contracted_channels_certified = true;
  const double response_old_norm = track_response_
      ? max_abs(response_aggregate_)
      : 0.0;

  Eigen::RowVectorXd transition_delta(n_electrons_);
  for (Eigen::Index auxiliary = 0;
       auxiliary < channels_.rows();
       ++auxiliary) {
    for (int left = 0; left < n_electrons_; ++left) {
      const int left_orbital = occupied_left[left];
      transition_delta(left) =
          ri_factors(
              auxiliary,
              TwoElectronIndexer::packed_pair_index(
                  occupied_right_new[row], left_orbital)) -
          ri_factors(
              auxiliary,
              TwoElectronIndexer::packed_pair_index(
                  occupied_right_old[row], left_orbital));
    }
    Eigen::Map<Eigen::MatrixXd> channel(
        channels_.data() + auxiliary * n_electrons_ * n_electrons_,
        n_electrons_,
        n_electrons_);
    const double old_channel_norm = max_abs(channel);
    double transition_delta_error = 0.0;
    for (int left = 0; left < n_electrons_; ++left) {
      transition_delta_error = std::max(
          transition_delta_error,
          epsilon * (
              std::abs(ri_factors(
                  auxiliary,
                  TwoElectronIndexer::packed_pair_index(
                      occupied_right_new[row], occupied_left[left]))) +
              std::abs(ri_factors(
                  auxiliary,
                  TwoElectronIndexer::packed_pair_index(
                      occupied_right_old[row], occupied_left[left])))));
    }
    const Eigen::RowVectorXd residual =
        transition_delta - overlap_delta.transpose() * channel;
    if (track_response_) {
      const double trace = channel.trace();
      const double trace_update = residual.dot(update_column);
      const Eigen::MatrixXd trace_term = trace_update * channel;
      const Eigen::MatrixXd rank_one_term =
          trace * update_column * residual;
      const Eigen::MatrixXd left_product =
          (channel * update_column) * residual;
      const Eigen::MatrixXd right_product =
          update_column * (residual * channel);
      const double update_column_norm =
          update_column.cwiseAbs().maxCoeff();
      const double residual_norm = residual.cwiseAbs().maxCoeff();
      const double trace_update_scale =
          residual.cwiseAbs().dot(update_column.cwiseAbs());
      const double left_product_scale =
          (channel.cwiseAbs() * update_column.cwiseAbs()).maxCoeff() *
          residual_norm;
      const double right_product_scale =
          update_column_norm *
          (residual.cwiseAbs() * channel.cwiseAbs()).maxCoeff();
      response_roundoff_bound_ +=
          epsilon * max_abs(response_aggregate_);
      response_aggregate_.noalias() +=
          trace_term + rank_one_term - left_product - right_product;
      response_update_scale +=
          trace_update_scale * old_channel_norm +
          channel.diagonal().cwiseAbs().sum() *
              update_column_norm * residual_norm +
          left_product_scale + right_product_scale;
    }
    const double old_error =
        channel_error_bounds_[static_cast<std::size_t>(auxiliary)];
    const double overlap_norm = overlap_delta.cwiseAbs().sum();
    const double residual_error =
        transition_delta_error +
        overlap_delta_error * old_channel_norm +
        overlap_norm * old_error +
        gamma * overlap_norm * old_channel_norm +
        epsilon * (
            transition_delta.cwiseAbs().maxCoeff() +
            overlap_norm * old_channel_norm);
    const double correction_norm =
        update_column.cwiseAbs().maxCoeff() *
        residual.cwiseAbs().maxCoeff();
    channel.noalias() += update_column * residual;
    const double new_error =
        (1.0 + update_column.cwiseAbs().maxCoeff() * overlap_norm) *
            old_error +
        update_column.cwiseAbs().maxCoeff() * residual_error +
        3.0 * epsilon * (old_channel_norm + correction_norm);
    channel_error_bounds_[static_cast<std::size_t>(auxiliary)] = new_error;
    double new_channel_norm = 0.0;
    phi_error_bound += channel_scalar_error_bound(
        channel, new_error, &new_channel_norm);
    contracted_channels_certified = contracted_channels_certified &&
        new_error <= std::sqrt(epsilon) * std::max(
            std::numeric_limits<double>::min(), new_channel_norm);
    if (track_response_) {
      response_channel_error_bound +=
          channel_response_error_bound(channel, new_error);
    }
  }
  if (track_response_) {
    response_roundoff_bound_ +=
        (6.0 * gamma + 8.0 * epsilon) * response_update_scale +
        epsilon * (response_old_norm + max_abs(response_aggregate_));
  }
  const double phi = two_electron_phi();
  const double phi_scale = std::max(
      std::numeric_limits<double>::min(), std::abs(phi));
  const double response_scale = track_response_
      ? std::max(
            std::numeric_limits<double>::min(),
            max_abs(response_aggregate_))
      : 1.0;
  const double response_error_bound =
      response_roundoff_bound_ + response_channel_error_bound;
  if (!channels_.allFinite() || !std::isfinite(phi_error_bound) ||
      phi_error_bound > std::sqrt(epsilon) * phi_scale ||
      (track_response_ &&
       (!response_aggregate_.allFinite() ||
        !std::isfinite(response_error_bound) ||
        response_error_bound > std::sqrt(epsilon) * response_scale))) {
    reset();
    return false;
  }
  occupied_right_ = occupied_right_new;
  contracted_channels_certified_ = contracted_channels_certified;
  return true;
}

void RiPairUpdateState::reset() {
  occupied_left_.clear();
  occupied_right_.clear();
  channel_error_bounds_.clear();
  channels_.resize(0, 0);
  response_aggregate_.resize(0, 0);
  response_roundoff_bound_ = 0.0;
  n_electrons_ = 0;
  track_response_ = false;
  contracted_channels_certified_ = false;
  valid_ = false;
}

double RiPairUpdateState::two_electron_phi() const {
  if (!valid_) {
    return 0.0;
  }
  double phi = 0.0;
  for (Eigen::Index auxiliary = 0;
       auxiliary < channels_.rows();
       ++auxiliary) {
    const Eigen::Map<const Eigen::MatrixXd> channel(
        channels_.data() + auxiliary * n_electrons_ * n_electrons_,
        n_electrons_,
        n_electrons_);
    const double trace = channel.trace();
    phi += trace * trace -
        channel.cwiseProduct(channel.transpose()).sum();
  }
  return 0.5 * phi;
}

bool RiPairUpdateState::first_order_cofactor_auxiliary(
    double overlap_determinant,
    Eigen::Ref<Eigen::VectorXd> auxiliary) const {
  if (!valid_ || auxiliary.size() != channels_.rows() ||
      !std::isfinite(overlap_determinant)) {
    return false;
  }
  const double epsilon = std::numeric_limits<double>::epsilon();
  const double trace_roundoff = matrix_product_roundoff_factor(n_electrons_);
  for (Eigen::Index channel_index = 0;
       channel_index < channels_.rows();
       ++channel_index) {
    const Eigen::Map<const Eigen::MatrixXd> channel(
        channels_.data() + channel_index * n_electrons_ * n_electrons_,
        n_electrons_,
        n_electrons_);
    const double trace = channel.trace();
    const double value = overlap_determinant * trace;
    const double trace_error =
        n_electrons_ *
            channel_error_bounds_[static_cast<std::size_t>(channel_index)] +
        trace_roundoff * channel.diagonal().cwiseAbs().sum();
    const double value_error = std::abs(overlap_determinant) * trace_error +
        epsilon * std::abs(value);
    const double scale = std::max(
        std::numeric_limits<double>::min(), std::abs(value));
    if (!std::isfinite(value) || !std::isfinite(value_error) ||
        value_error > std::sqrt(epsilon) * scale) {
      return false;
    }
    auxiliary(channel_index) = value;
  }
  return true;
}

bool RiPairUpdateState::copy_contracted_channel(
    Eigen::Index auxiliary,
    Eigen::Ref<Eigen::MatrixXd> channel) const {
  if (!valid_ || auxiliary < 0 || auxiliary >= channels_.rows() ||
      channel.rows() != n_electrons_ || channel.cols() != n_electrons_) {
    return false;
  }
  if (!contracted_channels_certified_) {
    return false;
  }
  const Eigen::Map<const Eigen::MatrixXd> stored(
      channels_.data() + auxiliary * n_electrons_ * n_electrons_,
      n_electrons_,
      n_electrons_);
  channel = stored;
  return true;
}

Eigen::MatrixXd RiPairUpdateState::two_electron_inverse_overlap_gradient(
    const DeterminantOverlapResult& overlap) const {
  if (!valid_ || !track_response_ ||
      overlap.overlap_submatrix.rows() != n_electrons_ ||
      overlap.overlap_submatrix.cols() != n_electrons_) {
    return {};
  }
  return response_aggregate_.transpose() *
      overlap.overlap_submatrix.transpose();
}

}  // namespace xmvb::vb
