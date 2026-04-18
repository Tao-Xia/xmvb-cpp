#include "pfaffian_vbscf/math/trace_projector.hpp"

#include <stdexcept>
#include <vector>

namespace xmvb::pfaffian_vbscf {

namespace {

double signed_trace_scale(int power) {
  const double sign = ((power % 2) == 1) ? 1.0 : -1.0;
  return 0.5 * sign;
}

}  // namespace

TraceProjectionResult TraceProjector::project(
    const ScalarBuffer& traces,
    int order) 
{
  if (order < 0) {
    throw std::invalid_argument("order must be non-negative");
  }
  if (static_cast<int>(traces.size()) < order) {
    throw std::invalid_argument("traces size is smaller than the requested order");
  }

  TraceProjectionResult result;
  result.trace_weights.assign(xmvb::to_size(order), 0.0);
  ScalarBuffer series_coefficients(xmvb::to_size(order) + 1, 0.0);
  series_coefficients[0] = 1.0;

  for (int k = 1; k <= order; ++k) {
    double coeff = 0.0;
    for (int p = 1; p <= k; ++p) {
      // -0.5 * (-1)^(p + 1) 
      coeff += signed_trace_scale(p) *
          traces[xmvb::to_size(p - 1)] * // t_p
          series_coefficients[xmvb::to_size(k - p)]; // c_{k-p}
    }
    // c_{k}
    series_coefficients[xmvb::to_size(k)] = coeff / static_cast<double>(k);
  }

  // the overlap between two VB structures
  result.overlap_value = series_coefficients[xmvb::to_size(order)];

  result.overlap_coefficients = series_coefficients;
  if (order == 0) {
    return result;
  }

  // Backward Pass with Adjoint method
  ScalarBuffer coeff_adjoints(xmvb::to_size(order) + 1, 0.0);
  coeff_adjoints[xmvb::to_size(order)] = 1.0;

  for (int k = order; k >= 1; --k) {
    const double coeff_bar = coeff_adjoints[xmvb::to_size(k)];
    if (coeff_bar == 0.0) {
      continue;
    }

    const double inv_k = 1.0 / static_cast<double>(k);
    for (int power = 1; power <= k; ++power) {
      const double scale = coeff_bar * inv_k * signed_trace_scale(power);
      result.trace_weights[xmvb::to_size(power - 1)] +=
          scale * series_coefficients[xmvb::to_size(k - power)];
      coeff_adjoints[xmvb::to_size(k - power)] +=
          scale * traces[xmvb::to_size(power - 1)];
    }
  }

  return result;
}

ScalarBuffer TraceProjector::backpropagate_trace_weight_adjoints(
    const ScalarBuffer& traces,
    int order,
    const ScalarBuffer& trace_weight_adjoints) {
  if (order < 0) {
    throw std::invalid_argument("order must be non-negative");
  }
  if (static_cast<int>(traces.size()) < order) {
    throw std::invalid_argument("traces size is smaller than the requested order");
  }
  if (static_cast<int>(trace_weight_adjoints.size()) < order) {
    throw std::invalid_argument(
        "trace_weight_adjoints size is smaller than the requested order");
  }

  ScalarBuffer series_coefficients(xmvb::to_size(order) + 1, 0.0);
  series_coefficients[0] = 1.0;
  std::vector<ScalarBuffer> coefficient_trace_jacobians(
      xmvb::to_size(order) + 1,
      ScalarBuffer(xmvb::to_size(order), 0.0));

  for (int k = 1; k <= order; ++k) {
    double coeff = 0.0;
    for (int power = 1; power <= k; ++power) {
      coeff += signed_trace_scale(power) *
          traces[xmvb::to_size(power - 1)] *
          series_coefficients[xmvb::to_size(k - power)];
    }
    series_coefficients[xmvb::to_size(k)] = coeff / static_cast<double>(k);

    const double inv_k = 1.0 / static_cast<double>(k);
    for (int trace_index = 1; trace_index <= k; ++trace_index) {
      double jacobian = signed_trace_scale(trace_index) *
          series_coefficients[xmvb::to_size(k - trace_index)];
      for (int power = 1; power <= k; ++power) {
        jacobian +=
            signed_trace_scale(power) *
            traces[xmvb::to_size(power - 1)] *
            coefficient_trace_jacobians[xmvb::to_size(k - power)]
                                       [xmvb::to_size(trace_index - 1)];
      }
      coefficient_trace_jacobians[xmvb::to_size(k)]
                                 [xmvb::to_size(trace_index - 1)] =
          inv_k * jacobian;
    }
  }

  ScalarBuffer trace_adjoints(xmvb::to_size(order), 0.0);
  ScalarBuffer coefficient_adjoints(xmvb::to_size(order) + 1, 0.0);
  std::vector<ScalarBuffer> coefficient_trace_jacobian_adjoints(
      xmvb::to_size(order) + 1,
      ScalarBuffer(xmvb::to_size(order), 0.0));
  for (int trace_index = 1; trace_index <= order; ++trace_index) {
    coefficient_trace_jacobian_adjoints[xmvb::to_size(order)]
                                       [xmvb::to_size(trace_index - 1)] =
        trace_weight_adjoints[xmvb::to_size(trace_index - 1)];
  }

  for (int k = order; k >= 1; --k) {
    const double inv_k = 1.0 / static_cast<double>(k);

    for (int trace_index = 1; trace_index <= k; ++trace_index) {
      const double jacobian_bar =
          coefficient_trace_jacobian_adjoints[xmvb::to_size(k)]
                                             [xmvb::to_size(trace_index - 1)];
      if (jacobian_bar == 0.0) {
        continue;
      }

      coefficient_adjoints[xmvb::to_size(k - trace_index)] +=
          jacobian_bar * inv_k * signed_trace_scale(trace_index);
      for (int power = 1; power <= k; ++power) {
        const double scale =
            jacobian_bar * inv_k * signed_trace_scale(power);
        trace_adjoints[xmvb::to_size(power - 1)] +=
            scale *
            coefficient_trace_jacobians[xmvb::to_size(k - power)]
                                       [xmvb::to_size(trace_index - 1)];
        coefficient_trace_jacobian_adjoints[xmvb::to_size(k - power)]
                                           [xmvb::to_size(trace_index - 1)] +=
            scale * traces[xmvb::to_size(power - 1)];
      }
    }

    const double coeff_bar = coefficient_adjoints[xmvb::to_size(k)];
    if (coeff_bar == 0.0) {
      continue;
    }

    for (int power = 1; power <= k; ++power) {
      const double scale = coeff_bar * inv_k * signed_trace_scale(power);
      trace_adjoints[xmvb::to_size(power - 1)] +=
          scale * series_coefficients[xmvb::to_size(k - power)];
      coefficient_adjoints[xmvb::to_size(k - power)] +=
          scale * traces[xmvb::to_size(power - 1)];
    }
  }

  return trace_adjoints;
}

ScalarBuffer TraceProjector::backpropagate_overlap_coefficient_adjoints(
    const ScalarBuffer& traces,
    int order,
    const ScalarBuffer& overlap_coefficient_adjoints) {
  if (order < 0) {
    throw std::invalid_argument("order must be non-negative");
  }
  if (static_cast<int>(traces.size()) < order) {
    throw std::invalid_argument("traces size is smaller than the requested order");
  }
  if (static_cast<int>(overlap_coefficient_adjoints.size()) < order + 1) {
    throw std::invalid_argument(
        "overlap_coefficient_adjoints size is smaller than the requested order + 1");
  }

  ScalarBuffer series_coefficients(xmvb::to_size(order) + 1, 0.0);
  series_coefficients[0] = 1.0;
  for (int k = 1; k <= order; ++k) {
    double coeff = 0.0;
    for (int power = 1; power <= k; ++power) {
      coeff += signed_trace_scale(power) *
          traces[xmvb::to_size(power - 1)] *
          series_coefficients[xmvb::to_size(k - power)];
    }
    series_coefficients[xmvb::to_size(k)] = coeff / static_cast<double>(k);
  }

  ScalarBuffer trace_adjoints(xmvb::to_size(order), 0.0);
  ScalarBuffer coefficient_adjoints(
      overlap_coefficient_adjoints.begin(),
      overlap_coefficient_adjoints.begin() + order + 1);
  for (int k = order; k >= 1; --k) {
    const double coeff_bar = coefficient_adjoints[xmvb::to_size(k)];
    if (coeff_bar == 0.0) {
      continue;
    }

    const double inv_k = 1.0 / static_cast<double>(k);
    for (int power = 1; power <= k; ++power) {
      const double scale = coeff_bar * inv_k * signed_trace_scale(power);
      trace_adjoints[xmvb::to_size(power - 1)] +=
          scale * series_coefficients[xmvb::to_size(k - power)];
      coefficient_adjoints[xmvb::to_size(k - power)] +=
          scale * traces[xmvb::to_size(power - 1)];
    }
  }

  return trace_adjoints;
}

}  // namespace xmvb::pfaffian_vbscf
