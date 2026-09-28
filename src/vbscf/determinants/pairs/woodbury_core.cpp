#include "vbscf/determinants/pairs/woodbury_core.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/SVD>
#include <Eigen/LU>

#include "vbscf/determinants/algebra/cofactor_differential.hpp"
#include "vbscf/determinants/algebra/overlap.hpp"
#include "vbscf/determinants/pairs/woodbury_overlap.hpp"

namespace xmvb::vb {
namespace {

int pair_index(int first, int second) {
  return second * (second - 1) / 2 + first;
}

double determinant_from_svd(
    const Eigen::Ref<const Eigen::MatrixXd>& left,
    const Eigen::Ref<const Eigen::VectorXd>& singular_values,
    const Eigen::Ref<const Eigen::MatrixXd>& right) {
  const double parity = left.determinant() * right.determinant() < 0.0
      ? -1.0
      : 1.0;
  return parity * singular_values.prod();
}

struct LowRankFactors {
  Eigen::MatrixXd left;
  Eigen::MatrixXd right;
};

LowRankFactors compress_product(
    const Eigen::Ref<const Eigen::MatrixXd>& left,
    const Eigen::Ref<const Eigen::MatrixXd>& right) {
  LowRankFactors result;
  if (left.rows() != right.rows() || left.cols() != right.cols()) {
    throw std::invalid_argument("low-rank factors have inconsistent shapes");
  }
  if (left.cols() == 0) {
    result.left.resize(left.rows(), 0);
    result.right.resize(right.rows(), 0);
    return result;
  }
  const Eigen::JacobiSVD<Eigen::MatrixXd> left_svd(
      left, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::JacobiSVD<Eigen::MatrixXd> right_svd(
      right, Eigen::ComputeThinU | Eigen::ComputeThinV);
  if (left_svd.info() != Eigen::Success ||
      right_svd.info() != Eigen::Success) {
    throw std::runtime_error("low-rank factor compression failed");
  }
  const Eigen::MatrixXd middle =
      left_svd.singularValues().asDiagonal() *
      left_svd.matrixV().transpose() * right_svd.matrixV() *
      right_svd.singularValues().asDiagonal();
  const Eigen::JacobiSVD<Eigen::MatrixXd> middle_svd(
      middle, Eigen::ComputeThinU | Eigen::ComputeThinV);
  if (middle_svd.info() != Eigen::Success) {
    throw std::runtime_error("low-rank middle compression failed");
  }
  const Eigen::VectorXd singular_values = middle_svd.singularValues();
  const double threshold = singular_values.size() == 0
      ? 0.0
      : std::numeric_limits<double>::epsilon() *
            static_cast<double>(std::max(left.rows(), left.cols())) *
            singular_values(0);
  const int rank = static_cast<int>(
      (singular_values.array() > threshold).count());
  result.left.noalias() =
      left_svd.matrixU() * middle_svd.matrixU().leftCols(rank) *
      singular_values.head(rank).asDiagonal();
  result.right.noalias() =
      right_svd.matrixU() * middle_svd.matrixV().leftCols(rank);
  return result;
}

}  // namespace

WoodburyCore::WoodburyCore(
    const Eigen::Ref<const Eigen::MatrixXd>& overlap) {
  if (overlap.rows() != overlap.cols()) {
    throw std::invalid_argument("Woodbury core overlap must be square");
  }
  const int n = static_cast<int>(overlap.rows());
  overlap_ = overlap;
  if (n == 0) {
    base_.resize(0, 0);
    inverse_base_.resize(0, 0);
    core_left_.resize(0, 0);
    core_right_.resize(0, 0);
    inverse_base_core_left_.resize(0, 0);
    return;
  }

  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      overlap, Eigen::ComputeFullU | Eigen::ComputeFullV);
  if (svd.info() != Eigen::Success || !svd.singularValues().allFinite()) {
    throw std::runtime_error("Woodbury core overlap SVD failed");
  }

  const Eigen::VectorXd singular_values = svd.singularValues();
  if (singular_values(n - 1) > 0.0) {
    Eigen::MatrixXd inverse;
    inverse.noalias() =
        svd.matrixV() * singular_values.cwiseInverse().asDiagonal() *
        svd.matrixU().transpose();
    if (is_certified_regular_overlap(overlap, inverse)) {
      base_ = overlap;
      inverse_base_ = std::move(inverse);
      base_determinant_ = determinant_from_svd(
          svd.matrixU(), singular_values, svd.matrixV());
      core_left_.resize(n, 0);
      core_right_.resize(n, 0);
      inverse_base_core_left_.resize(n, 0);
      return;
    }
  }

  constexpr double highest_inverse_power = 4.0;
  const double condition_limit = std::pow(
      std::numeric_limits<double>::epsilon(),
      -1.0 / (2.0 * highest_inverse_power));
  const double spectral_scale =
      singular_values(0) > 0.0 ? singular_values(0) : 1.0;
  const double tau = static_cast<double>(n) * spectral_scale / condition_limit;

  Eigen::VectorXd completed = singular_values;
  int core_rank = 0;
  for (int index = 0; index < n; ++index) {
    if (singular_values(index) < tau) {
      completed(index) = tau;
      ++core_rank;
    }
  }
  base_.noalias() =
      svd.matrixU() * completed.asDiagonal() * svd.matrixV().transpose();
  inverse_base_.noalias() =
      svd.matrixV() * completed.cwiseInverse().asDiagonal() *
      svd.matrixU().transpose();
  if (!is_certified_regular_overlap(base_, inverse_base_)) {
    throw std::runtime_error("Woodbury core completion is not certified");
  }
  base_determinant_ = determinant_from_svd(
      svd.matrixU(), completed, svd.matrixV());

  core_left_.noalias() =
      svd.matrixU().rightCols(core_rank) *
      (singular_values.tail(core_rank) -
       Eigen::VectorXd::Constant(core_rank, tau))
          .asDiagonal();
  core_right_ = svd.matrixV().rightCols(core_rank);
  rebuild_core_contraction();
}

WoodburyCore::~WoodburyCore() = default;

void WoodburyCore::rebuild_core_contraction() {
  inverse_base_core_left_.noalias() = inverse_base_ * core_left_;
  if (rank() == 0) {
    core_determinant_ = 1.0;
    core_nullity_ = 0;
    core_cofactor_.reset();
    return;
  }
  const Eigen::MatrixXd core =
      Eigen::MatrixXd::Identity(rank(), rank()) +
      core_right_.transpose() * inverse_base_core_left_;
  DeterminantOverlapResolver resolver;
  const DeterminantOverlapResult core_overlap = resolver.resolve_matrix(core);
  core_determinant_ = core_overlap.overlap_determinant;
  core_nullity_ = core_overlap.nullity;
  core_cofactor_ =
      std::make_unique<const CofactorDifferential>(core_overlap);
}

WoodburyBaseUpdate WoodburyCore::append(
    const Eigen::Ref<const Eigen::MatrixXd>& update_left,
    const Eigen::Ref<const Eigen::MatrixXd>& update_right) {
  if (update_left.rows() != dimension() ||
      update_right.rows() != dimension() ||
      update_left.cols() != update_right.cols()) {
    throw std::invalid_argument("Woodbury overlap update dimensions differ");
  }
  WoodburyBaseUpdate result;
  result.left.resize(dimension(), 0);
  result.right.resize(dimension(), 0);
  if (update_left.cols() == 0) {
    return result;
  }

  // Preserve the regular fast path.  When no dangerous directions are
  // retained, test the incoming graph edge directly before invoking any thin
  // SVD.  This is the ordinary block-Woodbury update and costs O(n^2 r).
  if (rank() == 0) {
    const Eigen::MatrixXd inverse_left = inverse_base_ * update_left;
    const Eigen::MatrixXd middle =
        Eigen::MatrixXd::Identity(
            update_left.cols(), update_left.cols()) +
        update_right.transpose() * inverse_left;
    Eigen::FullPivLU<Eigen::MatrixXd> middle_lu(middle);
    if (middle_lu.isInvertible()) {
      const Eigen::MatrixXd right_inverse =
          update_right.transpose() * inverse_base_;
      Eigen::MatrixXd inverse_update_left;
      inverse_update_left.noalias() =
          -inverse_left * middle_lu.inverse();
      const Eigen::MatrixXd candidate_base =
          base_ + update_left * update_right.transpose();
      const Eigen::MatrixXd candidate_inverse =
          inverse_base_ + inverse_update_left * right_inverse;
      const double determinant_ratio = middle_lu.determinant();
      if (determinant_ratio != 0.0 &&
          std::isfinite(determinant_ratio) &&
          is_certified_regular_overlap(
              candidate_base, candidate_inverse)) {
        overlap_.noalias() += update_left * update_right.transpose();
        base_ = candidate_base;
        inverse_base_ = candidate_inverse;
        base_determinant_ *= determinant_ratio;
        result.left = std::move(inverse_update_left);
        result.right = update_right;
        rebuild_core_contraction();
        return result;
      }
    }
  }

  const int combined_rank = rank() + update_left.cols();
  Eigen::MatrixXd combined_left(dimension(), combined_rank);
  Eigen::MatrixXd combined_right(dimension(), combined_rank);
  if (rank() > 0) {
    combined_left.leftCols(rank()) = core_left_;
    combined_right.leftCols(rank()) = core_right_;
  }
  combined_left.rightCols(update_left.cols()) = update_left;
  combined_right.rightCols(update_right.cols()) = update_right;
  LowRankFactors compressed = compress_product(combined_left, combined_right);

  const int compressed_rank = compressed.left.cols();
  const Eigen::MatrixXd core_middle =
      Eigen::MatrixXd::Identity(compressed_rank, compressed_rank) +
      compressed.right.transpose() * inverse_base_ * compressed.left;
  const Eigen::JacobiSVD<Eigen::MatrixXd> core_svd(
      core_middle, Eigen::ComputeFullU | Eigen::ComputeFullV);
  if (core_svd.info() != Eigen::Success) {
    throw std::runtime_error("Woodbury overlap core SVD failed");
  }

  bool absorbed = false;
  for (int retained_rank = 0;
       retained_rank <= compressed_rank;
       ++retained_rank) {
    const int safe_rank = compressed_rank - retained_rank;
    Eigen::MatrixXd safe_left(dimension(), safe_rank);
    Eigen::MatrixXd safe_right(dimension(), safe_rank);
    if (safe_rank > 0) {
      const Eigen::MatrixXd safe_basis =
          core_svd.matrixV().leftCols(safe_rank);
      safe_left.noalias() = compressed.left * safe_basis;
      safe_right.noalias() = compressed.right * safe_basis;
    }

    Eigen::MatrixXd candidate_base = base_;
    Eigen::MatrixXd candidate_inverse = inverse_base_;
    Eigen::MatrixXd inverse_update_left(dimension(), safe_rank);
    if (safe_rank > 0) {
      const Eigen::MatrixXd inverse_left = inverse_base_ * safe_left;
      const Eigen::MatrixXd middle =
          Eigen::MatrixXd::Identity(safe_rank, safe_rank) +
          safe_right.transpose() * inverse_left;
      Eigen::FullPivLU<Eigen::MatrixXd> middle_lu(middle);
      if (!middle_lu.isInvertible()) {
        continue;
      }
      const Eigen::MatrixXd right_inverse =
          safe_right.transpose() * inverse_base_;
      inverse_update_left.noalias() =
          -inverse_left * middle_lu.inverse();
      candidate_base.noalias() += safe_left * safe_right.transpose();
      candidate_inverse.noalias() +=
          inverse_update_left * right_inverse;
      if (!is_certified_regular_overlap(
              candidate_base, candidate_inverse)) {
        continue;
      }
      const double determinant_ratio = middle_lu.determinant();
      if (!std::isfinite(determinant_ratio) || determinant_ratio == 0.0) {
        continue;
      }
      base_determinant_ *= determinant_ratio;
      result.left = inverse_update_left;
      result.right = safe_right;
    }

    if (retained_rank > 0) {
      const Eigen::MatrixXd dangerous_basis =
          core_svd.matrixV().rightCols(retained_rank);
      core_left_.noalias() = compressed.left * dangerous_basis;
      core_right_.noalias() = compressed.right * dangerous_basis;
    } else {
      core_left_.resize(dimension(), 0);
      core_right_.resize(dimension(), 0);
    }
    base_ = std::move(candidate_base);
    inverse_base_ = std::move(candidate_inverse);
    absorbed = true;
    break;
  }
  if (!absorbed) {
    throw std::runtime_error("Woodbury overlap update lost its stable base");
  }

  overlap_.noalias() += update_left * update_right.transpose();
  rebuild_core_contraction();
  return result;
}

double WoodburyCore::determinant() const noexcept {
  return base_determinant_ * core_determinant_;
}

double WoodburyCore::core_first_contraction(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix) const {
  if (rank() == 0) {
    return 0.0;
  }
  if (matrix.rows() != rank() || matrix.cols() != rank()) {
    throw std::invalid_argument(
        "Woodbury core first contraction dimensions differ");
  }
  return (core_cofactor_->value().cwiseProduct(matrix)).sum();
}

Eigen::MatrixXd WoodburyCore::first_cofactor() const {
  if (rank() == 0) {
    return base_determinant_ * inverse_base_.transpose();
  }
  const Eigen::MatrixXd correction =
      inverse_base_core_left_ * core_cofactor_->value().transpose() *
      core_right_.transpose() * inverse_base_;
  return base_determinant_ *
      (core_determinant_ * inverse_base_ - correction).transpose();
}

Eigen::MatrixXd WoodburyCore::first_cofactor_direction(
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_direction) const {
  if (overlap_direction.rows() != dimension() ||
      overlap_direction.cols() != dimension()) {
    throw std::invalid_argument(
        "Woodbury first-cofactor direction dimensions differ");
  }
  const Eigen::MatrixXd inverse_direction =
      -inverse_base_ * overlap_direction * inverse_base_;
  const double base_determinant_direction =
      base_determinant_ * (inverse_base_ * overlap_direction).trace();
  if (rank() == 0) {
    return base_determinant_direction * inverse_base_.transpose() +
        base_determinant_ * inverse_direction.transpose();
  }

  const Eigen::MatrixXd base_core_direction =
      inverse_direction * core_left_;
  const Eigen::MatrixXd core_direction =
      core_right_.transpose() * base_core_direction;
  const double core_determinant_direction =
      (core_cofactor_->value().cwiseProduct(core_direction)).sum();
  const Eigen::MatrixXd core_first_direction =
      core_cofactor_->first(core_direction);
  const Eigen::MatrixXd correction =
      inverse_base_core_left_ * core_cofactor_->value().transpose() *
      core_right_.transpose() * inverse_base_;
  const Eigen::MatrixXd correction_direction =
      base_core_direction * core_cofactor_->value().transpose() *
          core_right_.transpose() * inverse_base_ +
      inverse_base_core_left_ * core_first_direction.transpose() *
          core_right_.transpose() * inverse_base_ +
      inverse_base_core_left_ * core_cofactor_->value().transpose() *
          core_right_.transpose() * inverse_direction;
  const Eigen::MatrixXd reduced =
      core_determinant_ * inverse_base_ - correction;
  const Eigen::MatrixXd reduced_direction =
      core_determinant_direction * inverse_base_ +
      core_determinant_ * inverse_direction - correction_direction;
  return (base_determinant_direction * reduced +
          base_determinant_ * reduced_direction)
      .transpose();
}

double WoodburyCore::first_contraction(
    const Eigen::Ref<const Eigen::MatrixXd>& transition) const {
  if (transition.rows() != dimension() ||
      transition.cols() != dimension()) {
    throw std::invalid_argument(
        "Woodbury first contraction dimensions differ");
  }
  const Eigen::MatrixXd channel = inverse_base_ * transition;
  return first_channel_contraction(channel);
}

WoodburyContraction WoodburyCore::first_contraction_gradient(
    const Eigen::Ref<const Eigen::MatrixXd>& transition) const {
  if (transition.rows() != dimension() ||
      transition.cols() != dimension()) {
    throw std::invalid_argument(
        "Woodbury first-gradient dimensions differ");
  }
  WoodburyContraction result;
  const Eigen::MatrixXd channel = inverse_base_ * transition;
  const double trace = channel.trace();
  Eigen::MatrixXd channel_gradient = Eigen::MatrixXd::Zero(
      dimension(), dimension());
  Eigen::MatrixXd inverse_gradient = Eigen::MatrixXd::Zero(
      dimension(), dimension());
  double base_determinant_gradient = 0.0;

  if (rank() == 0) {
    result.value = base_determinant_ * trace;
    base_determinant_gradient = trace;
    channel_gradient.diagonal().array() = base_determinant_;
  } else {
    const Eigen::MatrixXd projected = core_right_.transpose() * channel;
    const Eigen::MatrixXd first_core =
        projected * inverse_base_core_left_;
    const double linear = core_first_contraction(first_core);
    const double reduced = core_determinant_ * trace - linear;
    result.value = base_determinant_ * reduced;
    base_determinant_gradient = reduced;

    Eigen::MatrixXd first_core_gradient =
        -base_determinant_ * core_cofactor_->value();
    Eigen::MatrixXd core_gradient =
        base_determinant_ * trace * core_cofactor_->value();
    core_gradient.noalias() += core_cofactor_->first(
        -base_determinant_ * first_core);

    channel_gradient.diagonal().array() +=
        base_determinant_ * core_determinant_;
    const Eigen::MatrixXd projected_gradient =
        first_core_gradient * inverse_base_core_left_.transpose();
    channel_gradient.noalias() += core_right_ * projected_gradient;

    Eigen::MatrixXd base_core_gradient =
        projected.transpose() * first_core_gradient;
    base_core_gradient.noalias() += core_right_ * core_gradient;
    inverse_gradient.noalias() +=
        base_core_gradient * core_left_.transpose();
  }

  inverse_gradient.noalias() += channel_gradient * transition.transpose();
  result.transition_gradient.noalias() =
      inverse_base_.transpose() * channel_gradient;
  result.overlap_gradient.noalias() =
      base_determinant_gradient * base_determinant_ *
          inverse_base_.transpose() -
      inverse_base_.transpose() * inverse_gradient *
          inverse_base_.transpose();
  return result;
}

WoodburyContractionDirection
WoodburyCore::first_contraction_gradient_direction(
    const Eigen::Ref<const Eigen::MatrixXd>& transition,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_direction,
    const Eigen::Ref<const Eigen::MatrixXd>& transition_direction) const {
  if (transition.rows() != dimension() ||
      transition.cols() != dimension() ||
      overlap_direction.rows() != dimension() ||
      overlap_direction.cols() != dimension() ||
      transition_direction.rows() != dimension() ||
      transition_direction.cols() != dimension()) {
    throw std::invalid_argument(
        "Woodbury first-gradient direction dimensions differ");
  }

  const Eigen::MatrixXd inverse_direction =
      -inverse_base_ * overlap_direction * inverse_base_;
  const double determinant_direction =
      base_determinant_ * (inverse_base_ * overlap_direction).trace();
  const Eigen::MatrixXd channel = inverse_base_ * transition;
  const Eigen::MatrixXd channel_direction =
      inverse_direction * transition +
      inverse_base_ * transition_direction;
  const double trace = channel.trace();
  const double trace_direction = channel_direction.trace();

  Eigen::MatrixXd channel_gradient = Eigen::MatrixXd::Zero(
      dimension(), dimension());
  Eigen::MatrixXd channel_gradient_direction = Eigen::MatrixXd::Zero(
      dimension(), dimension());
  Eigen::MatrixXd inverse_gradient = Eigen::MatrixXd::Zero(
      dimension(), dimension());
  Eigen::MatrixXd inverse_gradient_direction = Eigen::MatrixXd::Zero(
      dimension(), dimension());
  double determinant_gradient = 0.0;
  double determinant_gradient_direction = 0.0;
  WoodburyContractionDirection result;

  if (rank() == 0) {
    result.value = determinant_direction * trace +
        base_determinant_ * trace_direction;
    determinant_gradient = trace;
    determinant_gradient_direction = trace_direction;
    channel_gradient.diagonal().array() = base_determinant_;
    channel_gradient_direction.diagonal().array() = determinant_direction;
  } else {
    const Eigen::MatrixXd base_core_direction =
        inverse_direction * core_left_;
    const Eigen::MatrixXd core_direction =
        core_right_.transpose() * base_core_direction;
    const double core_determinant_direction =
        (core_cofactor_->value().cwiseProduct(core_direction)).sum();
    const Eigen::MatrixXd core_first_direction =
        core_cofactor_->first(core_direction);
    const Eigen::MatrixXd projected =
        core_right_.transpose() * channel;
    const Eigen::MatrixXd projected_direction =
        core_right_.transpose() * channel_direction;
    const Eigen::MatrixXd first_core =
        projected * inverse_base_core_left_;
    const Eigen::MatrixXd first_core_direction =
        projected_direction * inverse_base_core_left_ +
        projected * base_core_direction;
    const double linear = core_first_contraction(first_core);
    const double linear_direction =
        (core_first_direction.cwiseProduct(first_core)).sum() +
        (core_cofactor_->value().cwiseProduct(first_core_direction)).sum();
    const double reduced = core_determinant_ * trace - linear;
    const double reduced_direction =
        core_determinant_direction * trace +
        core_determinant_ * trace_direction - linear_direction;
    result.value = determinant_direction * reduced +
        base_determinant_ * reduced_direction;
    determinant_gradient = reduced;
    determinant_gradient_direction = reduced_direction;

    const Eigen::MatrixXd first_core_gradient =
        -base_determinant_ * core_cofactor_->value();
    const Eigen::MatrixXd first_core_gradient_direction =
        -determinant_direction * core_cofactor_->value() -
        base_determinant_ * core_first_direction;
    const Eigen::MatrixXd core_argument =
        -base_determinant_ * first_core;
    const Eigen::MatrixXd core_argument_direction =
        -determinant_direction * first_core -
        base_determinant_ * first_core_direction;
    Eigen::MatrixXd core_gradient =
        base_determinant_ * trace * core_cofactor_->value();
    core_gradient.noalias() += core_cofactor_->first(core_argument);
    Eigen::MatrixXd core_gradient_direction =
        (determinant_direction * trace +
         base_determinant_ * trace_direction) *
            core_cofactor_->value() +
        base_determinant_ * trace * core_first_direction;
    core_gradient_direction.noalias() +=
        core_cofactor_->mixed(core_direction, core_argument) +
        core_cofactor_->first(core_argument_direction);

    channel_gradient.diagonal().array() +=
        base_determinant_ * core_determinant_;
    channel_gradient_direction.diagonal().array() +=
        determinant_direction * core_determinant_ +
        base_determinant_ * core_determinant_direction;
    const Eigen::MatrixXd projected_gradient =
        first_core_gradient * inverse_base_core_left_.transpose();
    const Eigen::MatrixXd projected_gradient_direction =
        first_core_gradient_direction * inverse_base_core_left_.transpose() +
        first_core_gradient * base_core_direction.transpose();
    channel_gradient.noalias() +=
        core_right_ * projected_gradient;
    channel_gradient_direction.noalias() +=
        core_right_ * projected_gradient_direction;

    Eigen::MatrixXd base_core_gradient =
        projected.transpose() * first_core_gradient;
    base_core_gradient.noalias() += core_right_ * core_gradient;
    Eigen::MatrixXd base_core_gradient_direction =
        projected_direction.transpose() * first_core_gradient +
        projected.transpose() * first_core_gradient_direction;
    base_core_gradient_direction.noalias() +=
        core_right_ * core_gradient_direction;
    inverse_gradient.noalias() +=
        base_core_gradient * core_left_.transpose();
    inverse_gradient_direction.noalias() +=
        base_core_gradient_direction * core_left_.transpose();
  }

  inverse_gradient.noalias() += channel_gradient * transition.transpose();
  inverse_gradient_direction.noalias() +=
      channel_gradient_direction * transition.transpose() +
      channel_gradient * transition_direction.transpose();
  result.transition_gradient.noalias() =
      inverse_direction.transpose() * channel_gradient +
      inverse_base_.transpose() * channel_gradient_direction;

  const double scale = determinant_gradient * base_determinant_;
  const double scale_direction =
      determinant_gradient_direction * base_determinant_ +
      determinant_gradient * determinant_direction;
  result.overlap_gradient.noalias() =
      scale_direction * inverse_base_.transpose() +
      scale * inverse_direction.transpose() -
      inverse_direction.transpose() * inverse_gradient *
          inverse_base_.transpose() -
      inverse_base_.transpose() * inverse_gradient_direction *
          inverse_base_.transpose() -
      inverse_base_.transpose() * inverse_gradient *
          inverse_direction.transpose();
  return result;
}

double WoodburyCore::first_channel_contraction(
    const Eigen::Ref<const Eigen::MatrixXd>& channel) const {
  if (channel.rows() != dimension() || channel.cols() != dimension()) {
    throw std::invalid_argument(
        "Woodbury first channel dimensions differ");
  }
  const double trace = channel.trace();
  if (rank() == 0) {
    return base_determinant_ * trace;
  }
  const Eigen::MatrixXd projected =
      core_right_.transpose() * channel * inverse_base_core_left_;
  return base_determinant_ *
      (core_determinant_ * trace - core_first_contraction(projected));
}

Eigen::MatrixXd WoodburyCore::exterior_square(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("exterior square matrix must be square");
  }
  const int n = static_cast<int>(matrix.rows());
  const int pairs = n * (n - 1) / 2;
  Eigen::MatrixXd result(pairs, pairs);
  for (int row_second = 1; row_second < n; ++row_second) {
    for (int row_first = 0; row_first < row_second; ++row_first) {
      const int row = pair_index(row_first, row_second);
      for (int column_second = 1; column_second < n; ++column_second) {
        for (int column_first = 0;
             column_first < column_second;
             ++column_first) {
          const int column = pair_index(column_first, column_second);
          result(row, column) =
              matrix(row_first, column_first) *
                  matrix(row_second, column_second) -
              matrix(row_first, column_second) *
                  matrix(row_second, column_first);
        }
      }
    }
  }
  return result;
}

Eigen::MatrixXd WoodburyCore::exterior_square_reverse(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    const Eigen::Ref<const Eigen::MatrixXd>& exterior_gradient) {
  const int n = static_cast<int>(matrix.rows());
  const int pairs = n * (n - 1) / 2;
  if (matrix.cols() != n || exterior_gradient.rows() != pairs ||
      exterior_gradient.cols() != pairs) {
    throw std::invalid_argument(
        "exterior-square reverse dimensions differ");
  }
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n, n);
  for (int row_second = 1; row_second < n; ++row_second) {
    for (int row_first = 0; row_first < row_second; ++row_first) {
      const int row = pair_index(row_first, row_second);
      for (int column_second = 1; column_second < n; ++column_second) {
        for (int column_first = 0;
             column_first < column_second;
             ++column_first) {
          const int column = pair_index(column_first, column_second);
          const double weight = exterior_gradient(row, column);
          result(row_first, column_first) +=
              weight * matrix(row_second, column_second);
          result(row_second, column_second) +=
              weight * matrix(row_first, column_first);
          result(row_first, column_second) -=
              weight * matrix(row_second, column_first);
          result(row_second, column_first) -=
              weight * matrix(row_first, column_second);
        }
      }
    }
  }
  return result;
}

double WoodburyCore::second_factor_contraction(
    const Eigen::Ref<const Eigen::MatrixXd>& transition) const {
  if (transition.rows() != dimension() ||
      transition.cols() != dimension()) {
    throw std::invalid_argument(
        "Woodbury second contraction dimensions differ");
  }
  const Eigen::MatrixXd channel = inverse_base_ * transition;
  return second_channel_contraction(channel);
}

WoodburyContraction WoodburyCore::second_factor_contraction_gradient(
    const Eigen::Ref<const Eigen::MatrixXd>& transition) const {
  if (transition.rows() != dimension() ||
      transition.cols() != dimension()) {
    throw std::invalid_argument(
        "Woodbury second-gradient dimensions differ");
  }
  const Eigen::MatrixXd channel = inverse_base_ * transition;
  return second_channel_contraction_gradient(channel, transition);
}

WoodburyContractionDirection
WoodburyCore::second_factor_contraction_gradient_direction(
    const Eigen::Ref<const Eigen::MatrixXd> &transition,
    const Eigen::Ref<const Eigen::MatrixXd> &overlap_direction,
    const Eigen::Ref<const Eigen::MatrixXd> &transition_direction) const {
  if (transition.rows() != dimension() || transition.cols() != dimension() ||
      overlap_direction.rows() != dimension() ||
      overlap_direction.cols() != dimension() ||
      transition_direction.rows() != dimension() ||
      transition_direction.cols() != dimension()) {
    throw std::invalid_argument(
        "Woodbury second-gradient direction dimensions differ");
  }

  const Eigen::MatrixXd channel = inverse_base_ * transition;
  MatrixTable channels(1, dimension() * dimension());
  MatrixTable channel_directions(1, dimension() * dimension());
  MatrixTable transitions(1, dimension() * dimension());
  MatrixTable transition_directions(1, dimension() * dimension());
  Eigen::Map<Eigen::MatrixXd>(channels.data(), dimension(), dimension()) =
      channel;
  const Eigen::MatrixXd inverse_direction =
      -inverse_base_ * overlap_direction * inverse_base_;
  Eigen::Map<Eigen::MatrixXd>(
      channel_directions.data(), dimension(), dimension()).noalias() =
      inverse_direction * transition +
      inverse_base_ * transition_direction;
  Eigen::Map<Eigen::MatrixXd>(transitions.data(), dimension(), dimension()) =
      transition;
  Eigen::Map<Eigen::MatrixXd>(transition_directions.data(), dimension(),
                              dimension()) = transition_direction;
  return second_channel_sum_gradient_direction(
      channels, channel_directions, transitions, transition_directions,
      overlap_direction, true);
}

WoodburyContractionDirection
WoodburyCore::second_channel_sum_gradient_direction(
    const Eigen::Ref<const MatrixTable> &channels,
    const Eigen::Ref<const MatrixTable> &channel_directions,
    const Eigen::Ref<const MatrixTable> &transitions,
    const Eigen::Ref<const MatrixTable> &transition_directions,
    const Eigen::Ref<const Eigen::MatrixXd> &overlap_direction,
    bool transition_gradient) const {
  const int n = dimension();
  const Eigen::Index matrix_size = static_cast<Eigen::Index>(n) * n;
  if (overlap_direction.rows() != n || overlap_direction.cols() != n ||
      channels.cols() != matrix_size ||
      channel_directions.rows() != channels.rows() ||
      channel_directions.cols() != matrix_size ||
      transitions.rows() != channels.rows() ||
      transitions.cols() != matrix_size ||
      transition_directions.rows() != channels.rows() ||
      transition_directions.cols() != matrix_size) {
    throw std::invalid_argument(
        "Woodbury RI directional-batch dimensions differ");
  }

  const Eigen::MatrixXd inverse_direction =
      -inverse_base_ * overlap_direction * inverse_base_;
  const double determinant_direction =
      base_determinant_ * (inverse_base_ * overlap_direction).trace();
  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(n, n);

  Eigen::MatrixXd base_core_direction;
  Eigen::MatrixXd core_direction;
  Eigen::MatrixXd core_first_direction;
  Eigen::MatrixXd core_second_direction;
  double core_determinant_direction = 0.0;
  if (rank() != 0) {
    base_core_direction.noalias() = inverse_direction * core_left_;
    core_direction.noalias() = core_right_.transpose() * base_core_direction;
    core_determinant_direction =
        (core_cofactor_->value().cwiseProduct(core_direction)).sum();
    core_first_direction = core_cofactor_->first(core_direction);
    core_second_direction = core_cofactor_->second_first(core_direction);
  }

  WoodburyContractionDirection result;
  result.transition_gradient = Eigen::MatrixXd::Zero(n, n);
  Eigen::MatrixXd inverse_gradient = Eigen::MatrixXd::Zero(n, n);
  Eigen::MatrixXd inverse_gradient_direction = Eigen::MatrixXd::Zero(n, n);
  double determinant_gradient = 0.0;
  double determinant_gradient_direction = 0.0;

  // Bounds the live directional/adjoint workspace independently of N_aux.
  constexpr Eigen::Index auxiliary_tile_width = 64;
  for (Eigen::Index begin = 0; begin < channels.rows();
       begin += auxiliary_tile_width) {
    const Eigen::Index count =
        std::min(auxiliary_tile_width, channels.rows() - begin);
    const double *channel_data = channels.data() + begin * matrix_size;
    const double *channel_direction_data =
        channel_directions.data() + begin * matrix_size;
    const double *transition_data = transitions.data() + begin * matrix_size;
    const double *transition_direction_data =
        transition_directions.data() + begin * matrix_size;
    const Eigen::Map<const Eigen::MatrixXd> channel_block(channel_data, n,
                                                          n * count);
    const Eigen::Map<const Eigen::MatrixXd> channel_direction_block(
        channel_direction_data, n, n * count);
    const Eigen::Map<const Eigen::MatrixXd> transition_block(transition_data, n,
                                                             n * count);
    const Eigen::Map<const Eigen::MatrixXd> transition_direction_block(
        transition_direction_data, n, n * count);
    Eigen::MatrixXd channel_gradient_block =
        Eigen::MatrixXd::Zero(n, n * count);
    Eigen::MatrixXd channel_gradient_direction_block =
        Eigen::MatrixXd::Zero(n, n * count);

    for (Eigen::Index local = 0; local < count; ++local) {
      const auto channel = channel_block.middleCols(local * n, n);
      const auto channel_direction =
          channel_direction_block.middleCols(local * n, n);
      auto channel_gradient = channel_gradient_block.middleCols(local * n, n);
      auto channel_gradient_direction =
          channel_gradient_direction_block.middleCols(local * n, n);
      const double trace = channel.trace();
      const double trace_direction = channel_direction.trace();
      const double base_phi =
          0.5 *
          (trace * trace - channel.cwiseProduct(channel.transpose()).sum());
      const double base_phi_direction =
          trace * trace_direction -
          channel_direction.cwiseProduct(channel.transpose()).sum();

      if (rank() == 0) {
        result.value += determinant_direction * base_phi +
                        base_determinant_ * base_phi_direction;
        determinant_gradient += base_phi;
        determinant_gradient_direction += base_phi_direction;
        channel_gradient.noalias() =
            base_determinant_ * (trace * identity - channel.transpose());
        channel_gradient_direction.noalias() =
            determinant_direction * (trace * identity - channel.transpose()) +
            base_determinant_ *
                (trace_direction * identity - channel_direction.transpose());
      } else {
        const Eigen::MatrixXd projected = core_right_.transpose() * channel;
        const Eigen::MatrixXd projected_direction =
            core_right_.transpose() * channel_direction;
        const Eigen::MatrixXd first_core = projected * inverse_base_core_left_;
        const Eigen::MatrixXd first_core_direction =
            projected_direction * inverse_base_core_left_ +
            projected * base_core_direction;
        const Eigen::MatrixXd second_core =
            projected * channel * inverse_base_core_left_;
        const Eigen::MatrixXd second_core_direction =
            projected_direction * channel * inverse_base_core_left_ +
            projected * channel_direction * inverse_base_core_left_ +
            projected * channel * base_core_direction;
        const double first_linear = core_first_contraction(first_core);
        const double first_linear_direction =
            (core_first_direction.cwiseProduct(first_core)).sum() +
            (core_cofactor_->value().cwiseProduct(first_core_direction)).sum();
        const double second_linear = core_first_contraction(second_core);
        const double second_linear_direction =
            (core_first_direction.cwiseProduct(second_core)).sum() +
            (core_cofactor_->value().cwiseProduct(second_core_direction)).sum();
        const Eigen::MatrixXd exterior = exterior_square(first_core);
        const Eigen::MatrixXd exterior_direction =
            exterior_square(first_core + first_core_direction) - exterior -
            exterior_square(first_core_direction);
        const double quadratic = core_cofactor_->second_contraction(exterior);
        const double quadratic_direction =
            (core_cofactor_->second_contraction_gradient(exterior).cwiseProduct(
                 core_direction))
                .sum() +
            core_cofactor_->second_contraction(exterior_direction);
        const double reduced = core_determinant_ * base_phi -
                               trace * first_linear + second_linear + quadratic;
        const double reduced_direction =
            core_determinant_direction * base_phi +
            core_determinant_ * base_phi_direction -
            trace_direction * first_linear - trace * first_linear_direction +
            second_linear_direction + quadratic_direction;
        result.value += determinant_direction * reduced +
                        base_determinant_ * reduced_direction;
        determinant_gradient += reduced;
        determinant_gradient_direction += reduced_direction;

        channel_gradient.noalias() = base_determinant_ * core_determinant_ *
                                     (trace * identity - channel.transpose());
        channel_gradient.diagonal().array() +=
            -base_determinant_ * first_linear;
        channel_gradient_direction.noalias() =
            (determinant_direction * core_determinant_ +
             base_determinant_ * core_determinant_direction) *
                (trace * identity - channel.transpose()) +
            base_determinant_ * core_determinant_ *
                (trace_direction * identity - channel_direction.transpose());
        channel_gradient_direction.diagonal().array() +=
            -determinant_direction * first_linear -
            base_determinant_ * first_linear_direction;

        const Eigen::MatrixXd exterior_reverse =
            exterior_square_reverse(first_core, core_cofactor_->second());
        const Eigen::MatrixXd exterior_reverse_direction =
            exterior_square_reverse(first_core_direction,
                                    core_cofactor_->second()) +
            exterior_square_reverse(first_core, core_second_direction);
        Eigen::MatrixXd first_core_gradient =
            -base_determinant_ * trace * core_cofactor_->value();
        first_core_gradient.noalias() += base_determinant_ * exterior_reverse;
        Eigen::MatrixXd first_core_gradient_direction =
            -(determinant_direction * trace +
              base_determinant_ * trace_direction) *
                core_cofactor_->value() -
            base_determinant_ * trace * core_first_direction;
        first_core_gradient_direction.noalias() +=
            determinant_direction * exterior_reverse +
            base_determinant_ * exterior_reverse_direction;
        const Eigen::MatrixXd second_core_gradient =
            base_determinant_ * core_cofactor_->value();
        const Eigen::MatrixXd second_core_gradient_direction =
            determinant_direction * core_cofactor_->value() +
            base_determinant_ * core_first_direction;

        const Eigen::MatrixXd core_argument =
            -base_determinant_ * trace * first_core +
            base_determinant_ * second_core;
        const Eigen::MatrixXd core_argument_direction =
            -(determinant_direction * trace +
              base_determinant_ * trace_direction) *
                first_core -
            base_determinant_ * trace * first_core_direction +
            determinant_direction * second_core +
            base_determinant_ * second_core_direction;
        Eigen::MatrixXd core_gradient =
            base_determinant_ * base_phi * core_cofactor_->value();
        core_gradient.noalias() += core_cofactor_->first(core_argument);
        core_gradient.noalias() +=
            base_determinant_ *
            core_cofactor_->second_contraction_gradient(exterior);
        Eigen::MatrixXd core_gradient_direction =
            (determinant_direction * base_phi +
             base_determinant_ * base_phi_direction) *
                core_cofactor_->value() +
            base_determinant_ * base_phi * core_first_direction;
        core_gradient_direction.noalias() +=
            core_cofactor_->mixed(core_direction, core_argument) +
            core_cofactor_->first(core_argument_direction);
        core_gradient_direction.noalias() +=
            determinant_direction *
                core_cofactor_->second_contraction_gradient(exterior) +
            base_determinant_ *
                core_cofactor_->second_contraction_gradient_direction(
                    core_direction, exterior, exterior_direction);

        Eigen::MatrixXd projected_gradient =
            first_core_gradient * inverse_base_core_left_.transpose();
        projected_gradient.noalias() += second_core_gradient *
                                        inverse_base_core_left_.transpose() *
                                        channel.transpose();
        Eigen::MatrixXd projected_gradient_direction =
            first_core_gradient_direction *
                inverse_base_core_left_.transpose() +
            first_core_gradient * base_core_direction.transpose();
        projected_gradient_direction.noalias() +=
            second_core_gradient_direction *
                inverse_base_core_left_.transpose() * channel.transpose() +
            second_core_gradient * base_core_direction.transpose() *
                channel.transpose() +
            second_core_gradient * inverse_base_core_left_.transpose() *
                channel_direction.transpose();
        channel_gradient.noalias() += projected.transpose() *
                                      second_core_gradient *
                                      inverse_base_core_left_.transpose();
        channel_gradient.noalias() += core_right_ * projected_gradient;
        channel_gradient_direction.noalias() +=
            projected_direction.transpose() * second_core_gradient *
                inverse_base_core_left_.transpose() +
            projected.transpose() * second_core_gradient_direction *
                inverse_base_core_left_.transpose() +
            projected.transpose() * second_core_gradient *
                base_core_direction.transpose();
        channel_gradient_direction.noalias() +=
            core_right_ * projected_gradient_direction;

        Eigen::MatrixXd base_core_gradient =
            projected.transpose() * first_core_gradient;
        base_core_gradient.noalias() +=
            channel.transpose() * projected.transpose() * second_core_gradient;
        base_core_gradient.noalias() += core_right_ * core_gradient;
        Eigen::MatrixXd base_core_gradient_direction =
            projected_direction.transpose() * first_core_gradient +
            projected.transpose() * first_core_gradient_direction;
        base_core_gradient_direction.noalias() +=
            channel_direction.transpose() * projected.transpose() *
                second_core_gradient +
            channel.transpose() * projected_direction.transpose() *
                second_core_gradient +
            channel.transpose() * projected.transpose() *
                second_core_gradient_direction;
        base_core_gradient_direction.noalias() +=
            core_right_ * core_gradient_direction;
        inverse_gradient.noalias() +=
            base_core_gradient * core_left_.transpose();
        inverse_gradient_direction.noalias() +=
            base_core_gradient_direction * core_left_.transpose();
      }
      if (transition_gradient) {
        result.transition_gradient.noalias() +=
            inverse_direction.transpose() * channel_gradient +
            inverse_base_.transpose() * channel_gradient_direction;
      }
    }

    inverse_gradient.noalias() +=
        channel_gradient_block * transition_block.transpose();
    inverse_gradient_direction.noalias() +=
        channel_gradient_direction_block * transition_block.transpose() +
        channel_gradient_block * transition_direction_block.transpose();
  }

  const double scale = determinant_gradient * base_determinant_;
  const double scale_direction =
      determinant_gradient_direction * base_determinant_ +
      determinant_gradient * determinant_direction;
  result.overlap_gradient.noalias() =
      scale_direction * inverse_base_.transpose() +
      scale * inverse_direction.transpose() -
      inverse_direction.transpose() * inverse_gradient *
          inverse_base_.transpose() -
      inverse_base_.transpose() * inverse_gradient_direction *
          inverse_base_.transpose() -
      inverse_base_.transpose() * inverse_gradient *
          inverse_direction.transpose();
  return result;
}

WoodburyContraction WoodburyCore::second_channel_contraction_gradient(
    const Eigen::Ref<const Eigen::MatrixXd>& channel,
    const Eigen::Ref<const Eigen::MatrixXd>& transition) const {
  if (channel.rows() != dimension() || channel.cols() != dimension() ||
      transition.rows() != dimension() ||
      transition.cols() != dimension()) {
    throw std::invalid_argument(
        "Woodbury second channel-gradient dimensions differ");
  }
  WoodburyContraction result;
  const double trace = channel.trace();
  const double base_phi = 0.5 *
      (trace * trace - channel.cwiseProduct(channel.transpose()).sum());
  Eigen::MatrixXd channel_gradient = Eigen::MatrixXd::Zero(
      dimension(), dimension());
  Eigen::MatrixXd inverse_gradient = Eigen::MatrixXd::Zero(
      dimension(), dimension());
  double base_determinant_gradient = 0.0;

  if (rank() == 0) {
    result.value = base_determinant_ * base_phi;
    base_determinant_gradient = base_phi;
    channel_gradient.noalias() = base_determinant_ *
        (trace * Eigen::MatrixXd::Identity(dimension(), dimension()) -
         channel.transpose());
  } else {
    const Eigen::MatrixXd projected = core_right_.transpose() * channel;
    const Eigen::MatrixXd first_core =
        projected * inverse_base_core_left_;
    const Eigen::MatrixXd second_core =
        projected * channel * inverse_base_core_left_;
    const double first_linear = core_first_contraction(first_core);
    const double second_linear = core_first_contraction(second_core);
    const Eigen::MatrixXd exterior = exterior_square(first_core);
    const double quadratic =
        core_cofactor_->second_contraction(exterior);
    const double reduced =
        core_determinant_ * base_phi - trace * first_linear +
        second_linear + quadratic;
    result.value = base_determinant_ * reduced;
    base_determinant_gradient = reduced;

    double trace_gradient = -base_determinant_ * first_linear;
    channel_gradient.noalias() =
        base_determinant_ * core_determinant_ *
        (trace * Eigen::MatrixXd::Identity(dimension(), dimension()) -
         channel.transpose());
    channel_gradient.diagonal().array() += trace_gradient;

    Eigen::MatrixXd first_core_gradient =
        -base_determinant_ * trace * core_cofactor_->value();
    Eigen::MatrixXd second_core_gradient =
        base_determinant_ * core_cofactor_->value();
    first_core_gradient.noalias() += base_determinant_ *
        exterior_square_reverse(
            first_core, core_cofactor_->second());

    Eigen::MatrixXd core_gradient =
        base_determinant_ * base_phi * core_cofactor_->value();
    core_gradient.noalias() += core_cofactor_->first(
        -base_determinant_ * trace * first_core +
        base_determinant_ * second_core);
    core_gradient.noalias() += base_determinant_ *
        core_cofactor_->second_contraction_gradient(exterior);

    Eigen::MatrixXd projected_gradient =
        first_core_gradient * inverse_base_core_left_.transpose();
    projected_gradient.noalias() +=
        second_core_gradient *
        inverse_base_core_left_.transpose() * channel.transpose();
    channel_gradient.noalias() +=
        projected.transpose() * second_core_gradient *
        inverse_base_core_left_.transpose();
    channel_gradient.noalias() += core_right_ * projected_gradient;

    Eigen::MatrixXd base_core_gradient =
        projected.transpose() * first_core_gradient;
    base_core_gradient.noalias() +=
        channel.transpose() * projected.transpose() *
        second_core_gradient;
    base_core_gradient.noalias() += core_right_ * core_gradient;
    inverse_gradient.noalias() +=
        base_core_gradient * core_left_.transpose();
  }

  inverse_gradient.noalias() += channel_gradient * transition.transpose();
  result.transition_gradient.noalias() =
      inverse_base_.transpose() * channel_gradient;
  result.overlap_gradient.noalias() =
      base_determinant_gradient * base_determinant_ *
          inverse_base_.transpose() -
      inverse_base_.transpose() * inverse_gradient *
          inverse_base_.transpose();
  return result;
}

double WoodburyCore::second_channel_contraction(
    const Eigen::Ref<const Eigen::MatrixXd>& channel) const {
  if (channel.rows() != dimension() || channel.cols() != dimension()) {
    throw std::invalid_argument(
        "Woodbury second channel dimensions differ");
  }
  const double trace = channel.trace();
  const double base_phi = 0.5 *
      (trace * trace - channel.cwiseProduct(channel.transpose()).sum());
  if (rank() == 0) {
    return base_determinant_ * base_phi;
  }

  const Eigen::MatrixXd projected = core_right_.transpose() * channel;
  const Eigen::MatrixXd first_core = projected * inverse_base_core_left_;
  const Eigen::MatrixXd second_core =
      projected * channel * inverse_base_core_left_;
  const double linear =
      -trace * core_first_contraction(first_core) +
      core_first_contraction(second_core);
  const double quadratic = core_cofactor_->second_contraction(
      exterior_square(first_core));
  return base_determinant_ *
      (core_determinant_ * base_phi + linear + quadratic);
}

}  // namespace xmvb::vb
