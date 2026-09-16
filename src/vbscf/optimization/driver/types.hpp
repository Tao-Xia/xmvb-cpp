#pragma once

// Leaf types and constants used by the VBSCF optimizer (vbscf_optimizer),
// its line-search / Krylov / accepted-trace subsystems, and the
// VbScfObjective wrapper. Promoted out of vbscf_optimizer.cpp's
// anonymous namespace so future Phase 2 split TUs can name them without
// redefining them.

#include <Eigen/Core>

#include <vector>

#include "vbscf/core/contracts/orbital_type.hpp"

namespace xmvb::vb {

// One curvature pair in the common packed sparse-coefficient embedding.
// Current-chart tangent and covector projections provide the quotient-space
// vector transport when the pair is reused by the TNHVP preconditioner.
struct PackedSecantPair {
  Eigen::VectorXd packed_step;
  Eigen::VectorXd packed_gradient_change;
};

}  // namespace xmvb::vb
