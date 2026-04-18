#include "vb/pdft/molecular_grid.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

#include "vb/pdft/lebedev_grid.hpp"
#include "vb/pdft/radial_grid.hpp"
#include "vb/pdft/becke_partition.hpp"

namespace xmvb::vb::pdft {

MolecularGridBuilder::MolecularGridBuilder(MolecularGridConfig config)
    : config_(std::move(config)) {
  // Validate configuration
  if (config_.radial_points <= 0) {
    throw std::invalid_argument("radial_points must be positive");
  }

  if (!LebedevGrid::is_supported(config_.angular_points)) {
    throw std::invalid_argument("Unsupported angular_points value");
  }

  if (config_.becke_exponent < 0) {
    throw std::invalid_argument("becke_exponent must be non-negative");
  }
}

MolecularGrid MolecularGridBuilder::build(
    const Eigen::MatrixXd& atomic_coords,
    const Eigen::VectorXd& atomic_charges) const {
  const int n_atoms = static_cast<int>(atomic_coords.rows());

  if (atomic_charges.size() != n_atoms) {
    throw std::invalid_argument("atomic_charges size must match number of atoms");
  }

  if (atomic_coords.cols() != 3) {
    throw std::invalid_argument("atomic_coords must have 3 columns");
  }

  // Step 1: Generate atomic grids
  // Each atom gets radial_points x angular_points grid points

  const int points_per_atom = config_.radial_points * config_.angular_points;
  const int total_points = n_atoms * points_per_atom;

  Eigen::MatrixXd all_points(total_points, 3);
  Eigen::VectorXd atomic_weights(total_points);
  std::vector<int> atom_assignments(total_points);

  // Get Lebedev angular grid (same for all atoms)
  auto lebedev_grid = LebedevGrid::get_grid(config_.angular_points);

  int point_offset = 0;

  for (int atom = 0; atom < n_atoms; ++atom) {
    const int Z = static_cast<int>(atomic_charges(atom));
    const Eigen::Vector3d center = atomic_coords.row(atom);

    // Build radial grid for this atom
    auto radial_grid = RadialGridBuilder::build(
        config_.radial_points, Z, RadialGridScheme::MuraKnowlesLog3);

    // Combine radial and angular grids
    for (int ir = 0; ir < config_.radial_points; ++ir) {
      const double r = radial_grid[ir].r;
      const double r_weight = radial_grid[ir].weight;

      for (int ia = 0; ia < config_.angular_points; ++ia) {
        const auto& ang_point = lebedev_grid[ia];

        // Cartesian coordinates: center + r * (x, y, z)
        const Eigen::Vector3d point = center + r * Eigen::Vector3d(
            ang_point.x, ang_point.y, ang_point.z);

        // `r_weight` already represents the radial measure contribution
        // `r^2 dr`, so the molecular rule only multiplies the angular weight.
        const double weight = r_weight * ang_point.weight;

        const int idx = point_offset + ir * config_.angular_points + ia;
        all_points.row(idx) = point;
        atomic_weights(idx) = weight;
        atom_assignments[idx] = atom;
      }
    }

    point_offset += points_per_atom;
  }

  // Step 2: Apply Becke partition
  // Compute Becke weights for all grid points

  Eigen::VectorXi atomic_numbers = atomic_charges.cast<int>();
  Eigen::MatrixXd becke_weights = BeckePartition::compute_weights(
      all_points, atomic_coords, atomic_numbers, config_.becke_exponent);

  // Step 3: Multiply atomic weights by Becke weights
  // Final weight: w_final(r) = w_atomic(r) * w_Becke_A(r)
  // where A is the atom this grid point belongs to

  Eigen::VectorXd final_weights(total_points);

  for (int i = 0; i < total_points; ++i) {
    const int atom = atom_assignments[i];
    final_weights(i) = atomic_weights(i) * becke_weights(i, atom);
  }

  // Step 4: Assemble molecular grid

  MolecularGrid grid;
  grid.points = std::move(all_points);
  grid.weights = std::move(final_weights);
  grid.atom_assignments = std::move(atom_assignments);

  return grid;
}

}  // namespace xmvb::vb::pdft
