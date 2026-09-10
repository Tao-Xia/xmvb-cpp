#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <Eigen/Core>

namespace xmvb::vb {

struct SpectralTrustRegionSolution {
  Eigen::VectorXd step;
  double shift = 0.0;
  bool boundary = false;
  bool hard_case = false;
};

// Solve min g^T s + s^T diag(d) s / 2, ||s|| <= radius.
// Only the small HVP-subspace eigensystem is supplied, not an orbital Hessian.
inline SpectralTrustRegionSolution solve_spectral_trust_region(
    const Eigen::VectorXd& d, const Eigen::VectorXd& g, double radius) {
  if (d.size() != g.size() || !d.allFinite() || !g.allFinite() ||
      !(radius > 0.0) || !std::isfinite(radius)) {
    throw std::invalid_argument("invalid spectral trust-region problem");
  }
  SpectralTrustRegionSolution result;
  result.step = Eigen::VectorXd::Zero(d.size());
  if (d.size() == 0) return result;

  // Work on the unit ball and normalize the energy scale. Never square the
  // radius. Keep lambda = lower + excess separate so a tiny positive excess
  // is not lost next to a large negative-curvature shift.
  const double scale = std::max(d.cwiseAbs().maxCoeff(), g.stableNorm() / radius);
  if (scale == 0.0) return result;
  if (!std::isfinite(scale)) {
    throw std::runtime_error("unrepresentable trust-region energy scale");
  }
  const Eigen::VectorXd values = d / scale;
  Eigen::VectorXd rhs = g / scale;
  for (Eigen::Index j = 0; j < rhs.size(); ++j) {
    // A subnormal intermediate may lose a gradient that becomes representable
    // again after division by a small radius. Change the order only there.
    if (radius < 1.0 && g[j] != 0.0 &&
        std::abs(rhs[j]) < std::numeric_limits<double>::min()) {
      rhs[j] = (g[j] / radius) / scale;
    } else {
      rhs[j] /= radius;
    }
  }
  Eigen::Index minimum_index = 0;
  const double minimum = values.minCoeff(&minimum_index);
  const double lower = std::max(0.0, -minimum);
  const Eigen::VectorXd shifted_values = values.array() + lower;
  const double epsilon = std::numeric_limits<double>::epsilon();
  const double roundoff = epsilon * std::max<Eigen::Index>(1, d.size());
  const double null_gradient_tolerance = roundoff * rhs.stableNorm();

  auto solve_at = [&](double excess, Eigen::VectorXd* z) {
    z->resize(d.size());
    for (Eigen::Index j = 0; j < d.size(); ++j) {
      const double denominator = shifted_values[j] + excess;
      if (denominator == 0.0) {
        if (std::abs(rhs[j]) > null_gradient_tolerance) return false;
        (*z)[j] = 0.0;  // Pseudoinverse at the spectral endpoint.
      } else {
        (*z)[j] = -rhs[j] / denominator;
      }
    }
    return z->allFinite();
  };

  Eigen::VectorXd z;
  if (solve_at(0.0, &z) && z.stableNorm() <= 1.0) {
    if (minimum < 0.0) {
      const double norm = z.stableNorm();
      // The minimum-eigenvalue component is zero in the pseudoinverse.
      z[minimum_index] = std::copysign(
          std::sqrt(std::max(0.0, (1.0 - norm) * (1.0 + norm))),
          -rhs[minimum_index]);
      result.boundary = true;
      result.hard_case = true;
    }
    result.step = radius * z;
    result.shift = scale * lower;
    result.boundary = result.boundary || z.stableNorm() >= 1.0 - roundoff;
    return result;
  }

  double lo = 0.0;
  // All shifted eigenvalues are nonnegative. This upper bound therefore
  // yields ||z|| <= ||rhs|| / hi <= 1, up to rounding.
  double hi = rhs.stableNorm();
  Eigen::VectorXd feasible;
  while (!solve_at(hi, &feasible) || feasible.stableNorm() > 1.0) {
    hi *= 2.0;
    if (!(hi > 0.0) || !std::isfinite(hi)) {
      throw std::runtime_error("failed to bracket trust-region secular root");
    }
  }
  // The limit covers the floating-point exponent range, not a solver budget.
  // A fixed 64 bisections cannot resolve, e.g., a 1e-60 shift in a unit bracket.
  constexpr int numerical_limit = std::numeric_limits<double>::max_exponent -
      std::numeric_limits<double>::min_exponent +
      std::numeric_limits<double>::digits;
  for (int iteration = 0; iteration < numerical_limit; ++iteration) {
    if (1.0 - feasible.stableNorm() <= roundoff) break;
    const double mid = lo + 0.5 * (hi - lo);
    if (mid == lo || mid == hi) break;
    Eigen::VectorXd trial;
    if (!solve_at(mid, &trial) || trial.stableNorm() > 1.0) {
      lo = mid;
    } else {
      hi = mid;
      feasible = std::move(trial);
    }
  }
  result.step = radius * feasible;
  result.shift = scale * (lower + hi);
  if (!result.step.allFinite() || !std::isfinite(result.shift)) {
    throw std::runtime_error("unrepresentable spectral trust-region solution");
  }
  result.boundary = true;
  return result;
}

}  // namespace xmvb::vb
