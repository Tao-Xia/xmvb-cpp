#pragma once

#include <Eigen/Core>
#include "integrals/response/types.hpp"
#include "mcscf/optimization/preconditioners/orbital_diagonal.hpp"

namespace xmvb::vb {

struct AcceptedPointContext;
struct VbScfInput;

struct AnalyticOrbitalDiagonal {
    Eigen::MatrixXd mo_coefficients;
    Eigen::MatrixXd rotation_diagonal;
    Eigen::MatrixXd inactive_inverse_transform;
    Eigen::MatrixXd active_inverse_transform;
    // Auxiliary-to-original rotation; preserves the existing physical step cap.
    Eigen::MatrixXd auxiliary_rotation;
};

/**
 * @brief Builds the analytic fixed-structure rotation diagonal at a VBSCF point.
 *
 * HAO and OEO use the same accepted AO-metric orthogonal frame. Sparse
 * supports enter only when this model is pulled into the orbital chart.
 * FULL/full-AO models use a block-semicanonical auxiliary frame when accepted
 * exchange images are available. Only preconditioning changes; the physical
 * orbitals, Hessian action and original component step cap remain unchanged.
 * Disabling auxiliary_frame provides the original-frame diagnostic reference.
 */
AnalyticOrbitalDiagonal
build_analytic_orbital_diagonal(const VbScfInput& input,
                                const AcceptedPointContext& accepted,
                                const integrals::ExactCtxPairMatrix* exchange = nullptr,
                                bool auxiliary_frame = true);

} // namespace xmvb::vb
