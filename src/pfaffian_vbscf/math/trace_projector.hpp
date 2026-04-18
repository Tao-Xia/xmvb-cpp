#pragma once

#include "pfaffian_vbscf/types/eigen_types.hpp"

namespace xmvb::pfaffian_vbscf {

/**
 * @brief Projection result for the physical `t^n` coefficient of
 * `sqrt(det(I + t K))`.
 *
 * `trace_weights[p - 1]` stores the derivative of the projected coefficient
 * with respect to `trace_p`.
 */
struct TraceProjectionResult {
  double overlap_value = 0.0;
  ScalarBuffer trace_weights;
  ScalarBuffer overlap_coefficients;
};

/**
 * @brief Projects a trace series onto one coefficient of
 * `sqrt(det(I + t K))`.
 *
 * The implementation uses the exponential trace recursion
 * `c_k = (1 / k) * sum_p (1/2) (-1)^(p+1) trace_p c_{k-p}` and a reverse-mode
 * sweep to obtain `d c_order / d trace_p`. This is the pure 1D Newton-style
 * recursion used after eliminating the spin-grid/Vandermonde projection.
 */
class TraceProjector {
public:
  /**
   * @brief Computes the physical `t^order` coefficient and its trace adjoints.
   *
   * @param traces Dense trace series `[trace_1, ..., trace_order]`.
   * @param order Requested coefficient order.
   * @return TraceProjectionResult Overlap coefficient and reverse weights.
   */
  static TraceProjectionResult project(
      const ScalarBuffer& traces,
      int order);

  /**
   * @brief Backpropagates adjoints of `trace_weights` to the input traces.
   *
   * This is the exact reverse sweep through the trace-weight recursion used in
   * `project(...)`. Operationally, it applies the Hessian of the projected
   * coefficient with respect to the trace series to one incoming vector.
   *
   * @param traces Dense trace series `[trace_1, ..., trace_order]`.
   * @param order Requested coefficient order.
   * @param trace_weight_adjoints Incoming adjoints of
   *     `project(traces, order).trace_weights`.
   * @return ScalarBuffer Dense adjoints of the input trace series.
   */
  static ScalarBuffer backpropagate_trace_weight_adjoints(
      const ScalarBuffer& traces,
      int order,
      const ScalarBuffer& trace_weight_adjoints);

  /**
   * @brief Backpropagates adjoints of the full overlap coefficient series
   * `c_0, ..., c_order` to the input traces.
   *
   * This is the direct reverse sweep of the projected overlap recursion used in
   * `project(...)`, without first compressing the incoming adjoint onto only the
   * physical coefficient `c_order`.
   *
   * @param traces Dense trace series `[trace_1, ..., trace_order]`.
   * @param order Requested coefficient order.
   * @param overlap_coefficient_adjoints Incoming adjoints of
   *     `project(traces, order).overlap_coefficients`.
   * @return ScalarBuffer Dense adjoints of the input trace series.
   */
  static ScalarBuffer backpropagate_overlap_coefficient_adjoints(
      const ScalarBuffer& traces,
      int order,
      const ScalarBuffer& overlap_coefficient_adjoints);
};

}  // namespace xmvb::pfaffian_vbscf
