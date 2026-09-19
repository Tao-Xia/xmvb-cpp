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

// One primal/dual curvature pair lifted from a quotient chart into the common
// packed sparse-coefficient embedding.  The primal step and dual gradient
// change must use expand_step and expand_gradient respectively; current-chart
// vector projection and covector pullback then provide their distinct
// transports after every accepted orbital update.
struct PackedSecantPair {
  Eigen::VectorXd packed_step;
  Eigen::VectorXd packed_gradient_change;
};

}  // namespace xmvb::vb
