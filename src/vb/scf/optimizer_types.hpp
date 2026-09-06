#ifndef XMVB_VB_SCF_OPTIMIZER_TYPES_HPP_
#define XMVB_VB_SCF_OPTIMIZER_TYPES_HPP_

// Leaf types and constants used by the VBSCF optimizer (cpp_vb_scf_optimizer),
// its line-search / Krylov / accepted-trace subsystems, and the
// OrbitalObjective wrapper. Promoted out of cpp_vb_scf_optimizer.cpp's
// anonymous namespace so future Phase 2 split TUs can name them without
// redefining them.

#include <Eigen/Core>

#include <vector>

namespace xmvb::vb {

// Legacy orbital-chart enum codes carried by OrbitalPreparationInput::orbital_type.
// Kept as constexpr int because the chart field is a plain int on the input
// struct; new code should treat these as the canonical chart discriminators.
constexpr int kLegacyOrbitalTypeHao = 1;
constexpr int kLegacyOrbitalTypeBdo = 2;
constexpr int kLegacyOrbitalTypeOeo = 3;

// One accepted or trialed secant pair in the optimizer's packed-coordinate
// history. Used by L-BFGS secant transport, the truncated-Newton Krylov
// preconditioner, and line-search fallbacks that lift reduced steps into the
// full sparse-orbital parameter space.
struct PackedSecantPair {
  Eigen::VectorXd packed_step;
  Eigen::VectorXd packed_projected_gradient_change;
};

}  // namespace xmvb::vb

#endif  // XMVB_VB_SCF_OPTIMIZER_TYPES_HPP_
