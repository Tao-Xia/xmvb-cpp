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

}  // namespace xmvb::vb
