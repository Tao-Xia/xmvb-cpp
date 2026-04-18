#pragma once

#include <vector>

namespace xmvb::vb::pdft {

/**
 * @brief Single radial quadrature point.
 */
struct RadialPoint {
  double r = 0.0;       ///< Radial distance from nucleus (Bohr)
  double weight = 0.0;  ///< Radial-measure weight for \int_0^\infty f(r) r^2 dr
};

/**
 * @brief Radial grid transformation schemes.
 *
 * Different schemes map the finite interval [0, 1] to the semi-infinite
 * range [0, ∞) with varying numerical properties.
 */
enum class RadialGridScheme {
  /**
   * @brief Mura-Knowles `log3` radial grid.
   *
   * This is the standard atom-centered rule
   *
   *   r(x) = -R log(1 - x^3),  x = (i + 1/2) / n
   *
   * with the returned weight storing the full radial measure `r^2 dr`.
   */
  MuraKnowlesLog3,

  /**
   * @brief Legacy Chebyshev radial rule with the Ahlrichs mapping.
   *
   * This is the atom-centered radial grid already used in the historical
   * `mol/xgrids.c` implementation.  The mapped quadrature weight already
   * contains the spherical Jacobian `r^2 dr`, so the downstream molecular
   * grid only needs to multiply by the angular Lebedev weight and Becke
   * partition factor.
   */
  LegacyAhlrichsChebyshev,

  /**
   * @brief Compatibility alias for the standard Mura-Knowles rule.
   */
  MuraKnowles = MuraKnowlesLog3,
};

/**
 * @brief Generates radial quadrature grids for atomic integration.
 *
 * The radial grid builder creates one-dimensional quadrature rules
 * suitable for integrating functions over the range [0, ∞) around
 * an atomic nucleus.
 */
class RadialGridBuilder {
public:
  /**
   * @brief Builds radial grid using the specified scheme.
   *
   * @param n_points Number of radial points.
   * @param atomic_number Atomic number (for radius scaling).
   * @param scheme Radial transformation scheme.
   * @return Vector of radial points with associated weights.
   * @throws std::invalid_argument if atomic_number is invalid.
   */
  static std::vector<RadialPoint> build(
      int n_points,
      int atomic_number,
      RadialGridScheme scheme = RadialGridScheme::MuraKnowlesLog3);
};

}  // namespace xmvb::vb::pdft
