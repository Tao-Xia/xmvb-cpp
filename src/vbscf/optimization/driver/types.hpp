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

// One curvature pair in the common packed sparse-coefficient embedding.  The
// step is an accepted retraction tangent and the gradient change is the full
// ambient covector difference.  Projecting both into the target chart retains
// the pullback-curvature contribution from normalization and gauge motion;
// replacing the covector difference by a transported projected gradient would
// discard that contribution.
struct PackedSecantPair {
  Eigen::VectorXd packed_step;
  Eigen::VectorXd packed_gradient_change;
};

}  // namespace xmvb::vb
