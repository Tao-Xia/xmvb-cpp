#pragma once

#include "mcscf/optimization/neo/solver.hpp"

namespace xmvb::vb {

class ExactHvpOperator;
class StructureTangentOperator;
class OrbitalChart;

/**
 * @brief Gradient-independent probe of the coupled Hessian inertia.
 *
 * Uses diag(P_orbital,I_structure) only to condition the spectral search.
 * A negative Rayleigh direction is a witness, not a converged Newton step.
 * A positive partial-space result is not a global positivity certificate.
 */
mcscf::NeoResult probe_neo_curvature(ExactHvpOperator& orbital_hessian,
                                     const StructureTangentOperator& structure_hessian,
                                     const OrbitalChart& chart);

} // namespace xmvb::vb
