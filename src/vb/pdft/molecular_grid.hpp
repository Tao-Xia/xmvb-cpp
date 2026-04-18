#pragma once

#include <string>
#include <vector>
#include <Eigen/Core>

namespace xmvb::vb::pdft {

/**
 * @brief Molecular integration grid with Becke partition weights.
 *
 * The molecular grid combines atom-centered radial and angular grids
 * with Becke partition weights to enable smooth integration over the
 * entire molecular volume.
 */
struct MolecularGrid {
  /**
   * @brief Grid point coordinates in Cartesian space.
   *
   * Column-major matrix: n_points x 3 (x, y, z).
   */
  Eigen::MatrixXd points;

  /**
   * @brief Integration weights including Becke partition.
   *
   * Length: n_points.
   * These weights account for the full atom-centered quadrature measure
   * `r^2 dr d\Omega` together with the Becke partition smoothing.
   */
  Eigen::VectorXd weights;

  /**
   * @brief Atom assignment for each grid point.
   *
   * Length: n_points.
   * Indicates which atom each grid point is centered on.
   * Useful for load balancing and debugging.
   */
  std::vector<int> atom_assignments;

  /**
   * @brief Number of grid points.
   */
  int n_points() const { return static_cast<int>(points.rows()); }
};

/**
 * @brief Configuration for molecular grid generation.
 */
struct MolecularGridConfig {
  /**
   * @brief Number of radial points per atom.
   *
   * Typical values: 50 (coarse), 75 (medium), 100 (fine).
   */
  int radial_points = 75;

  /**
   * @brief Number of angular points (Lebedev grid size).
   *
   * Supported values: 6, 14, 26, 38, 50, 302.
   * Typical values: 302 (medium), 590 (fine), 974 (very fine).
   */
  int angular_points = 302;

  /**
   * @brief Becke partition exponent.
   *
   * Typical value: 3 (Becke's original recommendation).
   * Higher values give sharper partition boundaries.
   */
  int becke_exponent = 3;

  /**
   * @brief Grid pruning scheme.
   *
   * Options: "none", "sg1" (Gill's SG-1 pruning).
   * Pruning reduces the number of angular points far from the nucleus.
   */
  std::string pruning_scheme = "none";
};

/**
 * @brief Builds molecular integration grids with Becke partitioning.
 *
 * The builder generates atom-centered grids by combining radial and
 * angular quadrature, then applies Becke partition weights to ensure
 * smooth integration across atomic boundaries.
 */
class MolecularGridBuilder {
public:
  /**
   * @brief Constructs grid builder with specified configuration.
   */
  explicit MolecularGridBuilder(MolecularGridConfig config);

  /**
   * @brief Builds molecular grid from atomic coordinates.
   *
   * @param atomic_coords Atomic coordinates (n_atoms x 3, column-major).
   * @param atomic_charges Atomic charges (length n_atoms).
   * @return Molecular grid with Becke-partitioned weights.
   */
  MolecularGrid build(
      const Eigen::MatrixXd& atomic_coords,
      const Eigen::VectorXd& atomic_charges) const;

private:
  MolecularGridConfig config_;
};

}  // namespace xmvb::vb::pdft
