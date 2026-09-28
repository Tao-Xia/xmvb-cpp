#include "vbscf/determinants/algebra/contracted_density.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace xmvb::vb {
namespace {

void validate_factors(
    int dimension,
    const Eigen::Ref<const Eigen::MatrixXd>& left,
    const Eigen::Ref<const Eigen::MatrixXd>& right,
    const char* label) {
  if (left.rows() != dimension || right.rows() != dimension ||
      left.cols() != right.cols()) {
    throw std::invalid_argument(
        std::string(label) + " low-rank factor dimensions differ");
  }
}

}  // namespace

ContractedDensityState::ContractedDensityState(
    const Eigen::Ref<const Eigen::MatrixXd>& inverse_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& transition,
    int maximum_order)
    : maximum_order_(maximum_order),
      inverse_overlap_(inverse_overlap) {
  if (inverse_overlap.rows() != inverse_overlap.cols() ||
      transition.rows() != inverse_overlap.rows() ||
      transition.cols() != inverse_overlap.cols() ||
      maximum_order < 0 || maximum_order > inverse_overlap.rows()) {
    throw std::invalid_argument(
        "contracted-density anchor dimensions or order are invalid");
  }
  channel_.noalias() = inverse_overlap_ * transition;

  const int n = dimension();
  powers_.reserve(maximum_order_ + 1);
  inverse_moments_.reserve(maximum_order_ + 1);
  powers_.push_back(Eigen::MatrixXd::Identity(n, n));
  inverse_moments_.push_back(inverse_overlap_);
  for (int order = 1; order <= maximum_order_; ++order) {
    powers_.push_back(powers_.back() * channel_);
    inverse_moments_.push_back(powers_.back() * inverse_overlap_);
  }
  rebuild_coefficients();
}

void ContractedDensityState::validate_order(int order) const {
  if (order < 0 || order > maximum_order_) {
    throw std::out_of_range(
        "contracted-density order is outside the stored hierarchy");
  }
}

void ContractedDensityState::rebuild_coefficients() {
  coefficients_.assign(maximum_order_ + 1, 0.0);
  coefficients_[0] = 1.0;
  for (int order = 1; order <= maximum_order_; ++order) {
    double value = 0.0;
    for (int power = 1; power <= order; ++power) {
      value += (power % 2 == 0 ? -1.0 : 1.0) *
          coefficients_[order - power] * powers_[power].trace();
    }
    coefficients_[order] = value / static_cast<double>(order);
  }
}

double ContractedDensityState::coefficient(int order) const {
  validate_order(order);
  return coefficients_[order];
}

double ContractedDensityState::contraction(
    int order,
    double overlap_determinant) const {
  return overlap_determinant * coefficient(order);
}

Eigen::MatrixXd ContractedDensityState::transition_gradient(
    int order,
    double overlap_determinant) const {
  validate_order(order);
  if (order == 0) {
    return Eigen::MatrixXd::Zero(dimension(), dimension());
  }
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(dimension(), dimension());
  for (int power = 0; power < order; ++power) {
    const double scale =
        (power % 2 == 0 ? 1.0 : -1.0) *
        coefficients_[order - 1 - power];
    result.noalias() += scale * inverse_moments_[power].transpose();
  }
  result *= overlap_determinant;
  return result;
}

Eigen::MatrixXd ContractedDensityState::overlap_gradient(
    int order,
    double overlap_determinant) const {
  validate_order(order);
  Eigen::MatrixXd result =
      coefficients_[order] * inverse_moments_[0].transpose();
  for (int power = 0; power < order; ++power) {
    const double scale =
        (power % 2 == 0 ? 1.0 : -1.0) *
        coefficients_[order - 1 - power];
    result.noalias() -=
        scale * inverse_moments_[power + 1].transpose();
  }
  result *= overlap_determinant;
  return result;
}

void ContractedDensityState::update(
    const Eigen::Ref<const Eigen::MatrixXd>& inverse_left,
    const Eigen::Ref<const Eigen::MatrixXd>& inverse_right,
    const Eigen::Ref<const Eigen::MatrixXd>& channel_left,
    const Eigen::Ref<const Eigen::MatrixXd>& channel_right) {
  validate_factors(
      dimension(), inverse_left, inverse_right, "inverse");
  validate_factors(
      dimension(), channel_left, channel_right, "channel");

  const Eigen::MatrixXd old_inverse = inverse_overlap_;
  const Eigen::MatrixXd old_channel = channel_;
  const Eigen::MatrixXd new_channel =
      old_channel + channel_left * channel_right.transpose();

  std::vector<Eigen::MatrixXd> left_krylov;
  std::vector<Eigen::MatrixXd> right_krylov;
  left_krylov.reserve(maximum_order_);
  right_krylov.reserve(maximum_order_);
  if (maximum_order_ > 0) {
    left_krylov.push_back(channel_left);
    right_krylov.push_back(channel_right);
    for (int power = 1; power < maximum_order_; ++power) {
      left_krylov.push_back(new_channel * left_krylov.back());
      right_krylov.push_back(old_channel.transpose() * right_krylov.back());
    }
  }

  std::vector<Eigen::MatrixXd> new_powers;
  new_powers.reserve(maximum_order_ + 1);
  new_powers.push_back(powers_[0]);
  for (int order = 1; order <= maximum_order_; ++order) {
    Eigen::MatrixXd value = powers_[order];
    for (int left_power = 0; left_power < order; ++left_power) {
      value.noalias() +=
          left_krylov[left_power] *
          right_krylov[order - 1 - left_power].transpose();
    }
    new_powers.push_back(std::move(value));
  }

  std::vector<Eigen::MatrixXd> new_moments;
  new_moments.reserve(maximum_order_ + 1);
  for (int order = 0; order <= maximum_order_; ++order) {
    Eigen::MatrixXd value = inverse_moments_[order];
    for (int left_power = 0; left_power < order; ++left_power) {
      value.noalias() +=
          left_krylov[left_power] *
          (right_krylov[order - 1 - left_power].transpose() * old_inverse);
    }
    if (inverse_left.cols() != 0) {
      value.noalias() +=
          (new_powers[order] * inverse_left) * inverse_right.transpose();
    }
    new_moments.push_back(std::move(value));
  }

  inverse_overlap_.noalias() +=
      inverse_left * inverse_right.transpose();
  channel_ = new_channel;
  powers_ = std::move(new_powers);
  inverse_moments_ = std::move(new_moments);
  rebuild_coefficients();
}

ContractedDensityJet::ContractedDensityJet(
    const Eigen::Ref<const Eigen::MatrixXd>& inverse_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& transition,
    const Eigen::Ref<const Eigen::MatrixXd>& inverse_direction,
    const Eigen::Ref<const Eigen::MatrixXd>& transition_direction,
    int maximum_order)
    : value_(inverse_overlap, transition, maximum_order),
      inverse_direction_(inverse_direction) {
  if (inverse_direction.rows() != value_.dimension() ||
      inverse_direction.cols() != value_.dimension() ||
      transition_direction.rows() != value_.dimension() ||
      transition_direction.cols() != value_.dimension()) {
    throw std::invalid_argument(
        "contracted-density directional anchor dimensions differ");
  }
  channel_direction_.noalias() =
      inverse_direction_ * transition +
      inverse_overlap * transition_direction;

  const int n = value_.dimension();
  directional_powers_.reserve(maximum_order + 1);
  directional_inverse_moments_.reserve(maximum_order + 1);
  directional_powers_.push_back(Eigen::MatrixXd::Zero(n, n));
  directional_inverse_moments_.push_back(inverse_direction_);
  for (int order = 1; order <= maximum_order; ++order) {
    directional_powers_.push_back(
        directional_powers_.back() * value_.channel_ +
        value_.powers_[order - 1] * channel_direction_);
    directional_inverse_moments_.push_back(
        directional_powers_.back() * value_.inverse_overlap_ +
        value_.powers_[order] * inverse_direction_);
  }
  rebuild_directional_coefficients();
}

void ContractedDensityJet::rebuild_directional_coefficients() {
  directional_coefficients_.assign(value_.maximum_order_ + 1, 0.0);
  for (int order = 1; order <= value_.maximum_order_; ++order) {
    double derivative = 0.0;
    for (int power = 1; power <= order; ++power) {
      const double sign = power % 2 == 0 ? -1.0 : 1.0;
      derivative += sign *
          (directional_coefficients_[order - power] *
               value_.powers_[power].trace() +
           value_.coefficients_[order - power] *
               directional_powers_[power].trace());
    }
    directional_coefficients_[order] =
        derivative / static_cast<double>(order);
  }
}

double ContractedDensityJet::coefficient_direction(int order) const {
  value_.validate_order(order);
  return directional_coefficients_[order];
}

double ContractedDensityJet::contraction_direction(
    int order,
    double overlap_determinant,
    double overlap_determinant_direction) const {
  return overlap_determinant_direction * value_.coefficient(order) +
      overlap_determinant * coefficient_direction(order);
}

Eigen::MatrixXd ContractedDensityJet::transition_gradient_direction(
    int order,
    double overlap_determinant,
    double overlap_determinant_direction) const {
  value_.validate_order(order);
  if (order == 0) {
    return Eigen::MatrixXd::Zero(value_.dimension(), value_.dimension());
  }
  Eigen::MatrixXd value =
      Eigen::MatrixXd::Zero(value_.dimension(), value_.dimension());
  Eigen::MatrixXd direction = value;
  for (int power = 0; power < order; ++power) {
    const double sign = power % 2 == 0 ? 1.0 : -1.0;
    const double coefficient =
        sign * value_.coefficients_[order - 1 - power];
    const double coefficient_direction =
        sign * directional_coefficients_[order - 1 - power];
    value.noalias() +=
        coefficient * value_.inverse_moments_[power].transpose();
    direction.noalias() +=
        coefficient_direction * value_.inverse_moments_[power].transpose() +
        coefficient * directional_inverse_moments_[power].transpose();
  }
  return overlap_determinant_direction * value +
      overlap_determinant * direction;
}

Eigen::MatrixXd ContractedDensityJet::overlap_gradient_direction(
    int order,
    double overlap_determinant,
    double overlap_determinant_direction) const {
  value_.validate_order(order);
  Eigen::MatrixXd accepted =
      value_.coefficients_[order] *
      value_.inverse_moments_[0].transpose();
  Eigen::MatrixXd direction =
      directional_coefficients_[order] *
          value_.inverse_moments_[0].transpose() +
      value_.coefficients_[order] *
          directional_inverse_moments_[0].transpose();
  for (int power = 0; power < order; ++power) {
    const double sign = power % 2 == 0 ? 1.0 : -1.0;
    const double coefficient =
        sign * value_.coefficients_[order - 1 - power];
    const double coefficient_direction =
        sign * directional_coefficients_[order - 1 - power];
    accepted.noalias() -=
        coefficient * value_.inverse_moments_[power + 1].transpose();
    direction.noalias() -=
        coefficient_direction *
            value_.inverse_moments_[power + 1].transpose() +
        coefficient *
            directional_inverse_moments_[power + 1].transpose();
  }
  return overlap_determinant_direction * accepted +
      overlap_determinant * direction;
}

void ContractedDensityJet::update(
    const Eigen::Ref<const Eigen::MatrixXd>& inverse_left,
    const Eigen::Ref<const Eigen::MatrixXd>& inverse_right,
    const Eigen::Ref<const Eigen::MatrixXd>& inverse_left_direction,
    const Eigen::Ref<const Eigen::MatrixXd>& inverse_right_direction,
    const Eigen::Ref<const Eigen::MatrixXd>& channel_left,
    const Eigen::Ref<const Eigen::MatrixXd>& channel_right,
    const Eigen::Ref<const Eigen::MatrixXd>& channel_left_direction,
    const Eigen::Ref<const Eigen::MatrixXd>& channel_right_direction) {
  validate_factors(
      value_.dimension(), inverse_left, inverse_right, "inverse");
  validate_factors(
      value_.dimension(),
      inverse_left_direction,
      inverse_right,
      "inverse-left direction");
  validate_factors(
      value_.dimension(),
      inverse_left,
      inverse_right_direction,
      "inverse-right direction");
  validate_factors(
      value_.dimension(), channel_left, channel_right, "channel");
  validate_factors(
      value_.dimension(),
      channel_left_direction,
      channel_right,
      "channel-left direction");
  validate_factors(
      value_.dimension(),
      channel_left,
      channel_right_direction,
      "channel-right direction");

  const Eigen::MatrixXd old_inverse = value_.inverse_overlap_;
  const Eigen::MatrixXd old_channel = value_.channel_;
  const Eigen::MatrixXd old_inverse_direction = inverse_direction_;
  const Eigen::MatrixXd old_channel_direction = channel_direction_;
  const Eigen::MatrixXd new_channel =
      old_channel + channel_left * channel_right.transpose();
  const Eigen::MatrixXd new_channel_direction =
      old_channel_direction +
      channel_left_direction * channel_right.transpose() +
      channel_left * channel_right_direction.transpose();

  std::vector<Eigen::MatrixXd> left_krylov;
  std::vector<Eigen::MatrixXd> right_krylov;
  std::vector<Eigen::MatrixXd> left_krylov_direction;
  std::vector<Eigen::MatrixXd> right_krylov_direction;
  const int maximum_order = value_.maximum_order_;
  if (maximum_order > 0) {
    left_krylov.push_back(channel_left);
    right_krylov.push_back(channel_right);
    left_krylov_direction.push_back(channel_left_direction);
    right_krylov_direction.push_back(channel_right_direction);
    for (int power = 1; power < maximum_order; ++power) {
      left_krylov_direction.push_back(
          new_channel_direction * left_krylov.back() +
          new_channel * left_krylov_direction.back());
      left_krylov.push_back(new_channel * left_krylov.back());
      right_krylov_direction.push_back(
          old_channel_direction.transpose() * right_krylov.back() +
          old_channel.transpose() * right_krylov_direction.back());
      right_krylov.push_back(
          old_channel.transpose() * right_krylov.back());
    }
  }

  std::vector<Eigen::MatrixXd> new_directional_powers;
  new_directional_powers.reserve(maximum_order + 1);
  new_directional_powers.push_back(directional_powers_[0]);
  for (int order = 1; order <= maximum_order; ++order) {
    Eigen::MatrixXd derivative = directional_powers_[order];
    for (int left_power = 0; left_power < order; ++left_power) {
      const int right_power = order - 1 - left_power;
      derivative.noalias() +=
          left_krylov_direction[left_power] *
              right_krylov[right_power].transpose() +
          left_krylov[left_power] *
              right_krylov_direction[right_power].transpose();
    }
    new_directional_powers.push_back(std::move(derivative));
  }

  value_.update(
      inverse_left, inverse_right, channel_left, channel_right);

  std::vector<Eigen::MatrixXd> new_directional_moments;
  new_directional_moments.reserve(maximum_order + 1);
  for (int order = 0; order <= maximum_order; ++order) {
    Eigen::MatrixXd derivative = directional_inverse_moments_[order];
    for (int left_power = 0; left_power < order; ++left_power) {
      const int right_power = order - 1 - left_power;
      const Eigen::MatrixXd right_times_inverse =
          right_krylov[right_power].transpose() * old_inverse;
      derivative.noalias() +=
          left_krylov_direction[left_power] * right_times_inverse +
          left_krylov[left_power] *
              (right_krylov_direction[right_power].transpose() * old_inverse +
               right_krylov[right_power].transpose() *
                   old_inverse_direction);
    }
    if (inverse_left.cols() != 0) {
      const Eigen::MatrixXd power_times_left =
          value_.powers_[order] * inverse_left;
      derivative.noalias() +=
          (new_directional_powers[order] * inverse_left +
           value_.powers_[order] * inverse_left_direction) *
              inverse_right.transpose() +
          power_times_left * inverse_right_direction.transpose();
    }
    new_directional_moments.push_back(std::move(derivative));
  }

  inverse_direction_.noalias() +=
      inverse_left_direction * inverse_right.transpose() +
      inverse_left * inverse_right_direction.transpose();
  channel_direction_ = new_channel_direction;
  directional_powers_ = std::move(new_directional_powers);
  directional_inverse_moments_ = std::move(new_directional_moments);
  rebuild_directional_coefficients();
}

}  // namespace xmvb::vb
