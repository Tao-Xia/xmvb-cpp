#pragma once

#include <vector>

#include "pfaffian_vbscf/types/eigen_types.hpp"

namespace xmvb::pfaffian_vbscf {

/**
 * @brief Spin block selector for `m x m` spatial slices of a `2m x 2m` matrix.
 */
enum class PfSpinBlock {
  AlphaAlpha,
  AlphaBeta,
  BetaAlpha,
  BetaBeta,
};

/**
 * @brief Source matrix family used by tensor term materialization.
 */
enum class PfMatrixSource {
  Identity,
  Left,
  Right,
  Sigma,
  KernelPower,
  KernelPowerTimesLeft,
  KernelPowerTimesRight,
  KernelPowerTimesLeftSigmaRight,
  RightSigmaTimesKernelPowerTimesLeftSigmaRight,
  OneRDM,
  TraceRDM,
};

/**
 * @brief Tensor contraction flavor for one two-electron term.
 */
enum class PfTensorContraction {
  Direct,
  Exchange,
  Coulomb,
  SameSpinSeparable,
  SameSpinBridge,
};

/**
 * @brief One materialized tensor operand built from the forward cache.
 */
struct PfTensorOperand {
  PfMatrixSource source = PfMatrixSource::Identity;
  int power = 0;
  PfSpinBlock block = PfSpinBlock::AlphaAlpha;
};

/**
 * @brief One packed two-electron contraction term driven by cache-derived blocks.
 */
struct PfTensorTerm {
  PfTensorContraction contraction = PfTensorContraction::Exchange;
  PfTensorOperand left_operand;
  PfTensorOperand right_operand;
  double scale = 1.0;
  bool same_spin = false;
};

/**
 * @brief Forward cache shared by the Pfaffian-VBSCF forward and adjoint kernels.
 */
struct PfKernelCache {
  int n_active_orbitals = 0;
  int n_spin_orbitals = 0;
  int trace_order = 0;
  Matrix left;
  Matrix right;
  Matrix sigma;
  Matrix kernel;
  Matrix right_sigma;
  Matrix left_sigma_right;
  std::vector<Matrix> kernel_powers;
  std::vector<Matrix> kernel_power_times_left;
  std::vector<Matrix> kernel_power_times_right;
  std::vector<Matrix> kernel_power_times_left_sigma_right;
  std::vector<Matrix> right_sigma_times_kernel_power_times_left_sigma_right;
  ScalarBuffer traces;
  ScalarBuffer trace_weights;
  ScalarBuffer projected_overlap_coefficients;
  double overlap_value = 0.0;
  Matrix one_rdm;
  std::vector<Matrix> trace_rdms;

  // Closed-shell exact spatial-kernel auxiliaries reused by the production
  // forward and adjoint two-electron paths.
  Matrix closed_shell_spatial_overlap;
  Matrix closed_shell_left_ba_block;
  Matrix closed_shell_right_ab_block;
  Matrix closed_shell_pair_core;
  Matrix closed_shell_spatial_kernel;
  Matrix closed_shell_sa;
  Matrix closed_shell_h;
  Matrix closed_shell_d;
  ScalarBuffer closed_shell_traces;
  std::vector<Matrix> closed_shell_trace_powers;
  ScalarBuffer closed_shell_pair_coefficients;
  ScalarBuffer closed_shell_pair_tail_coefficients;
  ScalarBuffer closed_shell_coeff_n2;
  Matrix closed_shell_pair_poly;
  Matrix closed_shell_pair_term_matrix;
  Matrix closed_shell_c_poly_h_n2;
  Matrix closed_shell_frechet_h_c_n2;
  Matrix closed_shell_frechet_h_c_h_n2;
  Matrix closed_shell_d_poly_h_n2;
  Matrix closed_shell_opposite_bridge_h_split_matrix;
};

}  // namespace xmvb::pfaffian_vbscf
