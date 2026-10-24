#pragma once

#include <optional>
#include "scf/optimization/orbitals.hpp"

namespace xmvb::scf {

struct OrbitalTrial {
    SpinMatrices orbitals, density;
    UnrestrictedEvaluation physical;
};

/** @brief One nonlinear SCF correction using the shared matrix-free CIAH solver.
 * Reanchors curvature after each accepted finite step; no stale microstep replay.
 */
std::optional<OrbitalTrial> ciah_step(const OrbitalModel& model,
                                      const EvaluateUnrestricted& evaluate,
                                      const SpinMatrices& reference_density,
                                      const UnrestrictedEvaluation& reference,
                                      const Eigen::Ref<const Eigen::MatrixXd>& overlap,
                                      const Eigen::VectorXd& history_direction,
                                      int* response_actions);

} // namespace xmvb::scf
