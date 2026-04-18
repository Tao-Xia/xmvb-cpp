#include "vb/pdft/radial_grid.hpp"

#include <cmath>
#include <stdexcept>

#include "vb/pdft/bragg_slater_radii.hpp"

namespace xmvb::vb::pdft {

namespace {

double mura_knowles_far_parameter(int atomic_number) {
  switch (atomic_number) {
    case 3:
    case 4:
    case 11:
    case 12:
    case 19:
    case 20:
      return 7.0;
    default:
      return 5.2;
  }
}

/**
 * @brief Standard Mura-Knowles `log3` radial rule.
 *
 * The midpoint samples avoid the singular endpoint at `x = 1` while the
 * returned weight already carries the spherical Jacobian `r^2 dr`.
 */
std::vector<RadialPoint> build_mura_knowles_log3_grid(
    int n_points,
    int atomic_number) {
  std::vector<RadialPoint> points;
  points.reserve(n_points);

  const double far_parameter = mura_knowles_far_parameter(atomic_number);
  for (int point_index = 0; point_index < n_points; ++point_index) {
    const double x = (point_index + 0.5) / n_points;
    const double x_cubed = x * x * x;
    const double denominator = 1.0 - x_cubed;
    const double r = -far_parameter * std::log(denominator);
    const double dr =
        far_parameter * 3.0 * x * x / (denominator * n_points);
    points.push_back({r, r * r * dr});
  }

  return points;
}

/**
 * @brief Legacy atom-centered radial grid used by the original C runtime.
 *
 * The old `xgrids.c` code uses Chebyshev nodes `x_i = cos(i π / (n + 1))`
 * together with the Ahlrichs-style mapping
 *
 *   r(x) = -c (1 + x)^{0.6} log((1 - x) / 2) R_A
 *
 * where `R_A` is the Bragg-Slater radius in bohr and `c = 1.1 / log(2)`.
 * The returned quadrature weight is the full radial measure
 *
 *   w_i = w_x r(x_i)^2 (dr / dx)(x_i)
 *
 * so downstream code should not multiply by an extra `r^2`.
 */
std::vector<RadialPoint> build_legacy_ahlrichs_grid(
    int n_points,
    double bragg_slater_radius) {
  std::vector<RadialPoint> points;
  points.reserve(n_points);

  const double scale = 1.1 / std::log(2.0);
  const double radius_cubed =
      bragg_slater_radius * bragg_slater_radius * bragg_slater_radius;

  for (int i = 1; i <= n_points; ++i) {
    const double x = std::cos(i * M_PI / (n_points + 1.0));
    const double x_weight =
        M_PI / (n_points + 1.0) * std::sqrt(std::max(0.0, 1.0 - x * x));
    const double one_plus_x = 1.0 + x;
    const double one_minus_x = 1.0 - x;
    const double log_argument = 0.5 * one_minus_x;
    const double mapped_radius =
        -scale * std::pow(one_plus_x, 0.6) * std::log(log_argument);
    const double mapped_jacobian =
        scale * std::pow(one_plus_x, 0.6) *
        (-0.6 * std::log(log_argument) / one_plus_x + 1.0 / one_minus_x);
    const double radial_measure_weight =
        x_weight * mapped_radius * mapped_radius * mapped_jacobian *
        radius_cubed;

    points.push_back(
        {mapped_radius * bragg_slater_radius, radial_measure_weight});
  }

  // The Chebyshev nodes are generated from the large-radius side inward.
  // Reverse once so consumers see an increasing radial coordinate.
  for (int left = 0, right = n_points - 1; left < right; ++left, --right) {
    const RadialPoint point = points[left];
    points[left] = points[right];
    points[right] = point;
  }

  return points;
}

}  // anonymous namespace

std::vector<RadialPoint> RadialGridBuilder::build(
    int n_points,
    int atomic_number,
    RadialGridScheme scheme) {
  if (n_points <= 0) {
    throw std::invalid_argument("Number of radial points must be positive");
  }

  const double radius = bragg_slater_radius_bohr(atomic_number);

  switch (scheme) {
    case RadialGridScheme::MuraKnowlesLog3:
      return build_mura_knowles_log3_grid(n_points, atomic_number);
    case RadialGridScheme::LegacyAhlrichsChebyshev:
      return build_legacy_ahlrichs_grid(n_points, radius);
    default:
      throw std::invalid_argument("Unsupported radial grid scheme");
  }
}

}  // namespace xmvb::vb::pdft
