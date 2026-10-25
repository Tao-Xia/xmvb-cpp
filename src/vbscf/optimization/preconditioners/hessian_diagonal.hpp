#pragma once

#include <Eigen/Core>
#include "mcscf/optimization/preconditioners/orbital_diagonal.hpp"

namespace xmvb::vb {

struct AcceptedPointContext;
struct VbScfInput;

struct AnalyticOrbitalDiagonal {
    Eigen::MatrixXd mo_coefficients;
    Eigen::MatrixXd rotation_diagonal;
    Eigen::MatrixXd inactive_inverse_transform;
    Eigen::MatrixXd active_inverse_transform;
};

/**
 * @brief Builds the analytic fixed-structure rotation diagonal at a VBSCF point.
 *
 * HAO and OEO use the same accepted AO-metric orthogonal frame. Sparse
 * supports enter only when this model is pulled into the orbital chart.
 */
AnalyticOrbitalDiagonal build_analytic_orbital_diagonal(const VbScfInput& input,
                                                        const AcceptedPointContext& accepted);

} // namespace xmvb::vb
