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
// step is the accepted finite retraction displacement and the gradient change
// is the full ambient covector difference. Projecting both into a target chart
// gives the first-order secant of the fixed-basis affine pullback used by the
// exact HVP. In particular, do not subtract reduced gradients expressed in two
// independently rebuilt charts: their basis drift is an O(step) false secant.
// See eqs 66d--66h of the orbital-optimization theory document.
struct PackedSecantPair {
  Eigen::VectorXd packed_step;
  Eigen::VectorXd packed_gradient_change;
};

}  // namespace xmvb::vb
