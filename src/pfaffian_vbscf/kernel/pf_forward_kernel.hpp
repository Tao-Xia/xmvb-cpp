#pragma once

#include <vector>

#include "pfaffian_vbscf/kernel/pf_kernel_cache.hpp"

namespace xmvb::pfaffian_vbscf {

/**
 * @brief Builds the forward Pfaffian-VBSCF kernel cache and evaluates
 * cache-driven two-electron contractions.
 */
class PfForwardKernel {
public:
  /**
   * @brief Builds `K`, its trace series, and the projected overlap coefficient.
   *
   * @param left Left spin-orbital state matrix.
   * @param sigma Spin-orbital overlap metric.
   * @param right Right spin-orbital state matrix.
   * @param trace_order Requested projection order.
   * @return PfKernelCache Fully populated forward cache.
   */
  static PfKernelCache build_cache(
      const ConstMatrixRef& left,
      const ConstMatrixRef& sigma,
      const ConstMatrixRef& right,
      int trace_order);

  /**
   * @brief Builds only the cache objects required by the current
   * closed-shell exact forward/adjoint path.
   *
   * This avoids materializing the generic tensor-term auxiliary matrix
   * families that are not consumed by `evaluate_closed_shell_two_electron()`
   * or its adjoint.
   */
  static PfKernelCache build_closed_shell_exact_cache(
      const ConstMatrixRef& left,
      const ConstMatrixRef& sigma,
      const ConstMatrixRef& right,
      int trace_order);

  /**
   * @brief Builds the minimal close-shell forward cache directly from spatial
   * pairing blocks.
   *
   * This path is intended for pure forward evaluations where the full
   * spin-orbital `left/sigma/right` matrices are not needed later by the
   * adjoint.
   */
  static PfKernelCache build_closed_shell_exact_spatial_cache(
      const ConstMatrixRef& left_ba,
      const ConstMatrixRef& spatial_overlap,
      const ConstMatrixRef& right_ab,
      int trace_order);

  /**
   * @brief Evaluates a list of packed two-electron contraction terms.
   *
   * @param cache Forward cache returned by `build_cache()`.
   * @param ggo Packed active-space two-electron tensor.
   * @param terms Tensor term descriptors.
   * @return double Summed two-electron contribution.
   */
  static double evaluate_tensor_terms(
      const PfKernelCache& cache,
      const ScalarBuffer& ggo,
      const std::vector<PfTensorTerm>& terms);

  /**
   * @brief Evaluates the exact closed-shell two-electron contribution without
   * materializing the intermediate tensor-term list.
   *
   * This path is specialized to the current singlet `AB/BA` Pf basis used by
   * `pfaffian_vbscf`.
   */
  static double evaluate_closed_shell_two_electron(
      const PfKernelCache& cache,
      const ScalarBuffer& ggo);

  /**
   * @brief Evaluates the closed-shell two-electron contribution directly from
   * RI active-pair factors.
   */
  static double evaluate_closed_shell_two_electron_ri(
      const PfKernelCache& cache,
      int n_auxiliary_functions,
      const ScalarBuffer& ri_active_pair_factors);
};

}  // namespace xmvb::pfaffian_vbscf
