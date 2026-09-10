#pragma once

// Leaf types and constants used by the VBSCF optimizer (vbscf_optimizer),
// its line-search / Krylov / accepted-trace subsystems, and the
// VbScfObjective wrapper. Promoted out of vbscf_optimizer.cpp's
// anonymous namespace so future Phase 2 split TUs can name them without
// redefining them.

#include <Eigen/Core>

#include <vector>

namespace xmvb::vb {

// input orbital-chart enum codes carried by OrbitalPreparationInput::orbital_type.
// Kept as constexpr int because the chart field is a plain int on the input
// struct; new code should treat these as the canonical chart discriminators.
constexpr int kOrbitalTypeHao = 1;
constexpr int kOrbitalTypeBdo = 2;
constexpr int kOrbitalTypeOeo = 3;

// One accepted or trialed secant pair in the optimizer's packed-coordinate
// history. Used by L-BFGS secant transport, the truncated-Newton Krylov
// preconditioner, and line-search fallbacks that lift reduced steps into the
// full sparse-orbital parameter space.
struct PackedSecantPair {
  Eigen::VectorXd packed_step;
  Eigen::VectorXd packed_projected_gradient_change;
};

}  // namespace xmvb::vb
