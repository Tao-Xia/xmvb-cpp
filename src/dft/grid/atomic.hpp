#pragma once

#include <Eigen/Core>

#include "integrals/ao/libcint_input.hpp"

namespace xmvb::dft {

struct GridOptions {
    int radial_points = 80;
    int angular_points = 590;
};

struct Grid {
    Eigen::MatrixXd points;  // N x 3, bohr
    Eigen::VectorXd weights; // N, bohr^3
};

// Atom-centered grid at the origin, before molecular partitioning.
Grid make_atomic_grid(const GridOptions& options = {});

// Becke-partitioned molecular grid. Coordinates in LibcintInput are in bohr.
Grid make_grid(const integrals::LibcintInput& input, const GridOptions& options = {});

} // namespace xmvb::dft
