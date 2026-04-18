#pragma once

#include <stdexcept>

#include <Eigen/Core>

#include "pfaffian_vbscf/kernel/pf_kernel_cache.hpp"

namespace xmvb::pfaffian_vbscf::detail {

using MatrixBlock = Eigen::Block<const Matrix, Eigen::Dynamic, Eigen::Dynamic>;

inline const Matrix& kernel_power_matrix(
    const PfKernelCache& cache,
    int power) {
  if (power < 0) {
    throw std::invalid_argument("kernel power must be non-negative");
  }
  if (power >= static_cast<int>(cache.kernel_powers.size())) {
    throw std::invalid_argument("kernel power is not available in the cache");
  }
  return cache.kernel_powers[xmvb::to_size(power)];
}

inline const Matrix& kernel_power_times_left_matrix(
    const PfKernelCache& cache,
    int power) {
  if (power < 0) {
    throw std::invalid_argument("kernel power must be non-negative");
  }
  if (power >= static_cast<int>(cache.kernel_power_times_left.size())) {
    throw std::invalid_argument("kernel power-times-left is not available in the cache");
  }
  return cache.kernel_power_times_left[xmvb::to_size(power)];
}

inline const Matrix& kernel_power_times_right_matrix(
    const PfKernelCache& cache,
    int power) {
  if (power < 0) {
    throw std::invalid_argument("kernel power must be non-negative");
  }
  if (power >= static_cast<int>(cache.kernel_power_times_right.size())) {
    throw std::invalid_argument("kernel power-times-right is not available in the cache");
  }
  return cache.kernel_power_times_right[xmvb::to_size(power)];
}

inline const Matrix& kernel_power_times_left_sigma_right_matrix(
    const PfKernelCache& cache,
    int power) {
  if (power < 0) {
    throw std::invalid_argument("kernel power must be non-negative");
  }
  if (power >= static_cast<int>(cache.kernel_power_times_left_sigma_right.size())) {
    throw std::invalid_argument(
        "kernel power-times-left-sigma-right is not available in the cache");
  }
  return cache.kernel_power_times_left_sigma_right[xmvb::to_size(power)];
}

inline const Matrix& right_sigma_times_kernel_power_times_left_sigma_right_matrix(
    const PfKernelCache& cache,
    int power) {
  if (power < 0) {
    throw std::invalid_argument("kernel power must be non-negative");
  }
  if (power >=
      static_cast<int>(cache.right_sigma_times_kernel_power_times_left_sigma_right.size())) {
    throw std::invalid_argument(
        "right-sigma-times-kernel-power-times-left-sigma-right is not available in the cache");
  }
  return cache.right_sigma_times_kernel_power_times_left_sigma_right[
      xmvb::to_size(power)];
}

inline const Matrix& trace_rdm_matrix(
    const PfKernelCache& cache,
    int index) {
  if (index < 0) {
    throw std::invalid_argument("trace RDM index must be non-negative");
  }
  if (index >= static_cast<int>(cache.trace_rdms.size())) {
    throw std::invalid_argument("trace RDM is not available in the cache");
  }
  return cache.trace_rdms[xmvb::to_size(index)];
}

inline const Matrix& materialize_source_matrix(
    const PfKernelCache& cache,
    const PfTensorOperand& operand) {
  switch (operand.source) {
    case PfMatrixSource::Identity:
      return kernel_power_matrix(cache, 0);
    case PfMatrixSource::Left:
      return cache.left;
    case PfMatrixSource::Right:
      return cache.right;
    case PfMatrixSource::Sigma:
      return cache.sigma;
    case PfMatrixSource::KernelPower:
      return kernel_power_matrix(cache, operand.power);
    case PfMatrixSource::KernelPowerTimesLeft:
      return kernel_power_times_left_matrix(cache, operand.power);
    case PfMatrixSource::KernelPowerTimesRight:
      return kernel_power_times_right_matrix(cache, operand.power);
    case PfMatrixSource::KernelPowerTimesLeftSigmaRight:
      return kernel_power_times_left_sigma_right_matrix(cache, operand.power);
    case PfMatrixSource::RightSigmaTimesKernelPowerTimesLeftSigmaRight:
      return right_sigma_times_kernel_power_times_left_sigma_right_matrix(cache, operand.power);
    case PfMatrixSource::OneRDM:
      return cache.one_rdm;
    case PfMatrixSource::TraceRDM:
      return trace_rdm_matrix(cache, operand.power);
  }
  throw std::invalid_argument("unsupported PfMatrixSource value");
}

inline MatrixBlock copy_spin_block(
    const PfKernelCache& cache,
    const Matrix& source,
    PfSpinBlock block) {
  const Eigen::Index n = cache.n_active_orbitals;
  switch (block) {
    case PfSpinBlock::AlphaAlpha:
      return source.block(0, 0, n, n);
    case PfSpinBlock::AlphaBeta:
      return source.block(0, n, n, n);
    case PfSpinBlock::BetaAlpha:
      return source.block(n, 0, n, n);
    case PfSpinBlock::BetaBeta:
      return source.block(n, n, n, n);
  }
  throw std::invalid_argument("unsupported PfSpinBlock value");
}

inline MatrixBlock materialize_operand_block(
    const PfKernelCache& cache,
    const PfTensorOperand& operand) {
  if (operand.source == PfMatrixSource::OneRDM) {
    const Matrix& spatial_matrix = cache.one_rdm;
    if (spatial_matrix.rows() != cache.n_active_orbitals ||
        spatial_matrix.cols() != cache.n_active_orbitals) {
      throw std::invalid_argument("spatial source does not match the active-space dimension");
    }
    return spatial_matrix.block(0, 0, cache.n_active_orbitals, cache.n_active_orbitals);
  }
  return copy_spin_block(cache, materialize_source_matrix(cache, operand), operand.block);
}

}  // namespace xmvb::pfaffian_vbscf::detail
