#include "vbscf/derivatives/hessian/coupled/coupling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "core/eigen_response.hpp"

namespace xmvb::vb {
namespace {

void validate_weights(const std::vector<double>& weights) {
  if (weights.empty()) {
    throw std::invalid_argument("structure coupling requires state weights");
  }
  double sum = 0.0;
  for (double weight : weights) {
    if (!(weight > 0.0) || !std::isfinite(weight)) {
      throw std::invalid_argument(
          "structure coupling requires positive finite state weights");
    }
    sum += weight;
  }
  if (std::abs(sum - 1.0) > 1.0e-12) {
    throw std::invalid_argument(
        "structure coupling state weights are not normalized");
  }
  if (weights.size() == 1) return;
  const double reference = weights.front();
  const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, std::abs(reference));
  if (!std::all_of(
          weights.begin(), weights.end(), [&](double weight) {
            return std::abs(weight - reference) <= tolerance;
          })) {
    throw std::invalid_argument(
        "structure coupling supports only equal-weight state averages");
  }
}

}  // namespace

ScaledStructureCoupling scale_equal_weight_structure_coupling(
    const xmvb::core::EqualWeightEigenCoupling& coupling,
    const std::vector<double>& normalized_state_weights) {
  validate_weights(normalized_state_weights);
  const Eigen::Index n_states =
      static_cast<Eigen::Index>(normalized_state_weights.size());
  const Eigen::Index n_structures = coupling.horizontal_forcing.rows();
  if (n_structures <= 0 || coupling.horizontal_forcing.cols() != n_states ||
      coupling.gauge_coefficient_response.rows() != n_structures ||
      coupling.gauge_coefficient_response.cols() != n_states ||
      coupling.gauge_selected_matrix_response.rows() != n_states ||
      coupling.gauge_selected_matrix_response.cols() != n_states ||
      !coupling.horizontal_forcing.allFinite() ||
      !coupling.gauge_coefficient_response.allFinite() ||
      !coupling.gauge_selected_matrix_response.allFinite()) {
    throw std::invalid_argument(
        "equal-weight structure coupling dimensions or values are inconsistent");
  }

  ScaledStructureCoupling result;
  result.horizontal_forcing.scaled_coefficients =
      coupling.horizontal_forcing;
  for (Eigen::Index state = 0; state < n_states; ++state) {
    result.horizontal_forcing.scaled_coefficients.col(state) *=
        std::sqrt(2.0 * normalized_state_weights[static_cast<std::size_t>(state)]);
  }
  result.gauge_coefficient_response = coupling.gauge_coefficient_response;
  result.gauge_adjoint_multipliers =
      -coupling.gauge_selected_matrix_response;
  return result;
}

}  // namespace xmvb::vb
