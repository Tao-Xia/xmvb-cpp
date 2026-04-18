#pragma once

#include <vector>

namespace xmvb::vb::pdft {

/**
 * @brief Single Lebedev quadrature point on the unit sphere.
 *
 * Lebedev grids provide spherical quadrature rules that exactly integrate
 * spherical harmonics up to a given order. The grids respect octahedral
 * symmetry and are widely used in molecular DFT calculations.
 */
struct LebedevPoint {
  double x = 0.0;       ///< Cartesian x coordinate on unit sphere
  double y = 0.0;       ///< Cartesian y coordinate on unit sphere
  double z = 0.0;       ///< Cartesian z coordinate on unit sphere
  double weight = 0.0;  ///< Quadrature weight (sum to 4*pi over sphere)
};

/**
 * @brief Lebedev angular quadrature grid generator.
 *
 * Generates Lebedev grids for spherical integration. The implementation
 * follows the tables from Lebedev & Laikov, Doklady Mathematics (1999).
 *
 * Supported grid sizes: 6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194,
 *                       230, 266, 302, 350, 434, 590, 770, 974
 */
class LebedevGrid {
public:
  /**
   * @brief Returns Lebedev grid for the specified number of points.
   *
   * @param n_points Number of angular points (must be a supported size).
   * @return Vector of Lebedev points with weights summing to 4*pi.
   * @throws std::invalid_argument if n_points is not supported.
   */
  static std::vector<LebedevPoint> get_grid(int n_points);

  /**
   * @brief Returns the smallest supported grid size >= requested size.
   *
   * @param requested_size Desired minimum number of points.
   * @return Closest supported size that is >= requested_size.
   */
  static int get_closest_supported_size(int requested_size);

  /**
   * @brief Returns all supported Lebedev grid sizes.
   */
  static std::vector<int> get_supported_sizes();

  /**
   * @brief Checks if a grid size is supported.
   */
  static bool is_supported(int n_points);
};

}  // namespace xmvb::vb::pdft
