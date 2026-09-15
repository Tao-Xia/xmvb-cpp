#include "vbscf/orbitals/charts/physical_metric.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Cholesky>

namespace xmvb::vb {

OrbitalPhysicalMetric::OrbitalPhysicalMetric(const OrbitalPreparationInput& input)
    : n_basis_functions_(input.n_basis_functions),
      n_inactive_((input.n_total_electrons - input.n_active_electrons) / 2),
      n_active_(input.n_active_orbitals) {
  if (n_basis_functions_ <= 0 || n_inactive_ < 0 || n_active_ < 0 ||
      n_inactive_ + n_active_ > input.n_orbitals) {
    throw std::invalid_argument("invalid orbital partition in physical metric");
  }
  overlap_ = Eigen::Map<const Eigen::MatrixXd>(
      input.ao_overlap_matrix.data(), n_basis_functions_, n_basis_functions_);
  if (!overlap_.allFinite()) {
    throw std::invalid_argument("non-finite AO overlap in physical metric");
  }
  Eigen::LLT<Eigen::MatrixXd> ao_factor(overlap_);
  if (ao_factor.info() != Eigen::Success) {
    throw std::runtime_error("AO overlap is not positive definite");
  }
  ao_metric_upper_ = ao_factor.matrixU();

  inactive_ = Eigen::MatrixXd::Zero(n_basis_functions_, n_inactive_);
  active_ = Eigen::MatrixXd::Zero(n_basis_functions_, n_active_);
  orbital_supports_.resize(n_inactive_ + n_active_);
  for (int orbital = 0; orbital < n_inactive_ + n_active_; ++orbital) {
    const int count = stored_sparse_orbital_coefficient_count(input, orbital);
    orbital_supports_[orbital].resize(count);
    for (int slot = 0; slot < count; ++slot) {
      const int flat = orbital * n_basis_functions_ + slot;
      const int basis = input.orbital_basis_index_table[flat] - 1;
      if (basis < 0 || basis >= n_basis_functions_) {
        throw std::invalid_argument("invalid sparse AO index in physical metric");
      }
      orbital_supports_[orbital][slot] = basis;
      if (orbital < n_inactive_) {
        inactive_(basis, orbital) = input.orbital_value_table[flat];
      } else {
        active_(basis, orbital - n_inactive_) = input.orbital_value_table[flat];
      }
    }
  }

  complement_ = Eigen::MatrixXd::Identity(
      n_basis_functions_, n_basis_functions_);
  if (n_inactive_ > 0) {
    const Eigen::MatrixXd inactive_metric =
        inactive_.transpose() * overlap_ * inactive_;
    Eigen::LDLT<Eigen::MatrixXd> factor(inactive_metric);
    if (factor.info() != Eigen::Success || !factor.isPositive()) {
      throw std::runtime_error("singular inactive span in physical metric");
    }
    inactive_inverse_ = factor.solve(
        Eigen::MatrixXd::Identity(n_inactive_, n_inactive_));
    if (!inactive_inverse_.allFinite()) {
      throw std::runtime_error("non-finite inactive metric inverse");
    }
    complement_.noalias() -=
        inactive_ * inactive_inverse_ * inactive_.transpose() * overlap_;
    Eigen::LLT<Eigen::MatrixXd> inactive_factor(inactive_metric);
    if (inactive_factor.info() != Eigen::Success) {
      throw std::runtime_error("inactive overlap is not positive definite");
    }
    const Eigen::MatrixXd upper = inactive_factor.matrixU();
    inactive_inverse_upper_ = upper.triangularView<Eigen::Upper>().solve(
        Eigen::MatrixXd::Identity(n_inactive_, n_inactive_));
  }
  projected_active_ = complement_ * active_;
  projected_norm_squared_ = Eigen::VectorXd::Zero(n_active_);
  for (int active = 0; active < n_active_; ++active) {
    projected_norm_squared_[active] =
        projected_active_.col(active).dot(overlap_ * projected_active_.col(active));
    if (!(projected_norm_squared_[active] > 0.0) ||
        !std::isfinite(projected_norm_squared_[active])) {
      throw std::runtime_error("degenerate projected active ray in physical metric");
    }
  }
}

Eigen::VectorXd OrbitalPhysicalMetric::physical_feature(
    const SparseParameterLayout& layout,
    const Eigen::Ref<const Eigen::VectorXd>& packed_direction) const {
  if (packed_direction.size() != layout.size() ||
      !packed_direction.allFinite()) {
    throw std::invalid_argument("invalid packed direction in physical metric");
  }
  Eigen::MatrixXd delta_inactive =
      Eigen::MatrixXd::Zero(n_basis_functions_, n_inactive_);
  Eigen::MatrixXd delta_active =
      Eigen::MatrixXd::Zero(n_basis_functions_, n_active_);
  for (int orbital = 0; orbital < n_inactive_ + n_active_; ++orbital) {
    const int count = layout.orbital_coefficient_count(orbital);
    if (count > static_cast<int>(orbital_supports_[orbital].size())) {
      throw std::invalid_argument("metric layout exceeds orbital support");
    }
    for (int slot = 0; slot < count; ++slot) {
      const int packed = layout.packed_index(orbital, slot);
      const int basis = orbital_supports_[orbital][slot];
      if (orbital < n_inactive_) {
        delta_inactive(basis, orbital) = packed_direction[packed];
      } else {
        delta_active(basis, orbital - n_inactive_) = packed_direction[packed];
      }
    }
  }

  const Eigen::MatrixXd horizontal_inactive = complement_ * delta_inactive;
  Eigen::VectorXd feature = Eigen::VectorXd::Zero(
      n_basis_functions_ * (n_inactive_ + n_active_));
  if (n_inactive_ > 0) {
    Eigen::Map<Eigen::MatrixXd>(
        feature.data(), n_basis_functions_, n_inactive_) =
        ao_metric_upper_ * horizontal_inactive * inactive_inverse_upper_;
  }

  Eigen::MatrixXd delta_projected_active = complement_ * delta_active;
  if (n_inactive_ > 0 && n_active_ > 0) {
    const Eigen::MatrixXd inactive_active_overlap =
        inactive_.transpose() * overlap_ * active_;
    const Eigen::MatrixXd horizontal_active_overlap =
        horizontal_inactive.transpose() * overlap_ * active_;
    delta_projected_active.noalias() -=
        horizontal_inactive * inactive_inverse_ * inactive_active_overlap;
    delta_projected_active.noalias() -=
        inactive_ * inactive_inverse_ * horizontal_active_overlap;
  }
  for (int active = 0; active < n_active_; ++active) {
    const Eigen::VectorXd projected = projected_active_.col(active);
    Eigen::VectorXd horizontal = delta_projected_active.col(active);
    horizontal.noalias() -= projected *
        (projected.dot(overlap_ * horizontal) /
         projected_norm_squared_[active]);
    feature.segment(
        n_basis_functions_ * (n_inactive_ + active),
        n_basis_functions_) =
        ao_metric_upper_ * horizontal /
        std::sqrt(projected_norm_squared_[active]);
  }
  return feature;
}

double OrbitalPhysicalMetric::squared_norm(
    const SparseParameterLayout& layout,
    const Eigen::Ref<const Eigen::VectorXd>& packed_direction) const {
  const double metric = physical_feature(layout, packed_direction).squaredNorm();
  if (!std::isfinite(metric) || metric < -1.0e-10) {
    throw std::runtime_error("non-finite or negative coupled orbital metric");
  }
  return std::max(0.0, metric);
}

Eigen::VectorXd OrbitalPhysicalMetric::apply(
    const SparseParameterLayout& layout,
    const Eigen::Ref<const Eigen::VectorXd>& packed_direction) const {
  if (packed_direction.size() != layout.size() ||
      !packed_direction.allFinite()) {
    throw std::invalid_argument("invalid packed direction in physical metric");
  }
  Eigen::MatrixXd delta_inactive =
      Eigen::MatrixXd::Zero(n_basis_functions_, n_inactive_);
  Eigen::MatrixXd delta_active =
      Eigen::MatrixXd::Zero(n_basis_functions_, n_active_);
  for (int orbital = 0; orbital < n_inactive_ + n_active_; ++orbital) {
    const int count = layout.orbital_coefficient_count(orbital);
    if (count > static_cast<int>(orbital_supports_[orbital].size())) {
      throw std::invalid_argument("metric layout exceeds orbital support");
    }
    for (int slot = 0; slot < count; ++slot) {
      const int packed = layout.packed_index(orbital, slot);
      const int basis = orbital_supports_[orbital][slot];
      if (orbital < n_inactive_) {
        delta_inactive(basis, orbital) = packed_direction[packed];
      } else {
        delta_active(basis, orbital - n_inactive_) = packed_direction[packed];
      }
    }
  }

  const Eigen::MatrixXd horizontal_inactive = complement_ * delta_inactive;
  Eigen::MatrixXd inactive_cotangent = Eigen::MatrixXd::Zero(
      n_basis_functions_, n_inactive_);
  if (n_inactive_ > 0) {
    inactive_cotangent.noalias() =
        overlap_ * horizontal_inactive * inactive_inverse_;
  }
  Eigen::MatrixXd active_cotangent = Eigen::MatrixXd::Zero(
      n_basis_functions_, n_active_);
  const Eigen::MatrixXd inactive_active_overlap =
      inactive_.transpose() * overlap_ * active_;
  for (int active = 0; active < n_active_; ++active) {
    Eigen::VectorXd delta_ray = complement_ * delta_active.col(active);
    if (n_inactive_ > 0) {
      delta_ray.noalias() -= horizontal_inactive *
          (inactive_inverse_ * inactive_active_overlap.col(active));
      delta_ray.noalias() -= inactive_ * inactive_inverse_ *
          (horizontal_inactive.transpose() * overlap_ * active_.col(active));
    }
    const Eigen::VectorXd ray = projected_active_.col(active);
    const double norm_squared = projected_norm_squared_[active];
    const Eigen::VectorXd overlap_ray = overlap_ * ray;
    const Eigen::VectorXd weighted_ray_cotangent =
        (overlap_ * delta_ray - overlap_ray *
         (overlap_ray.dot(delta_ray) / norm_squared)) / norm_squared;
    active_cotangent.col(active).noalias() +=
        complement_.transpose() * weighted_ray_cotangent;
    if (n_inactive_ > 0) {
      inactive_cotangent.noalias() -= weighted_ray_cotangent *
          (inactive_inverse_ * inactive_active_overlap.col(active)).transpose();
      inactive_cotangent.noalias() -=
          (overlap_ * active_.col(active)) *
          (inactive_inverse_ * inactive_.transpose() *
           weighted_ray_cotangent).transpose();
    }
  }
  if (n_inactive_ > 0) {
    inactive_cotangent =
        (complement_.transpose() * inactive_cotangent).eval();
  }
  Eigen::VectorXd packed_action = Eigen::VectorXd::Zero(layout.size());
  for (int orbital = 0; orbital < n_inactive_ + n_active_; ++orbital) {
    const int count = layout.orbital_coefficient_count(orbital);
    for (int slot = 0; slot < count; ++slot) {
      const int packed = layout.packed_index(orbital, slot);
      const int basis = orbital_supports_[orbital][slot];
      packed_action[packed] = orbital < n_inactive_
          ? inactive_cotangent(basis, orbital)
          : active_cotangent(basis, orbital - n_inactive_);
    }
  }
  if (!packed_action.allFinite()) {
    throw std::runtime_error("non-finite physical metric action");
  }
  return packed_action;
}

}  // namespace xmvb::vb
