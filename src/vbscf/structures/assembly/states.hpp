#pragma once

#include <optional>
#include <vector>
#include <Eigen/Core>

#include "vbscf/structures/assembly/action.hpp"

namespace xmvb::vb {

struct SelectedMetric;

/** @brief Fixed selected-state Rayleigh quotient and its derivatives.
 * Coefficients are M-orthonormal Ritz representatives of the prescribed span;
 * coefficient_gradient differentiates the original, unnormalized input frame.
 */
struct FixedStructureStates {
    Eigen::MatrixXd coefficients;
    std::vector<double> energies;
    Eigen::MatrixXd residuals;
    Eigen::MatrixXd coefficient_gradient;
    std::optional<StructureActiveIntegralAdjoint> integral_adjoint;
};

/** @brief Equal-weight fixed-state evaluation, without a CI eigenproblem.
 * Only the small state Gram and projected Hamiltonian are diagonalized.
 * Direct CI shares sigma images with integral adjoints; incomplete one-spin
 * carriers retain their occupied-pair H/M representation.
 * A supplied exact selected Gram must belong to action's orbital chart.
 */
FixedStructureStates evaluate_fixed_states(const StructureAction& action,
                                           const Eigen::Ref<const Eigen::MatrixXd>& coefficients,
                                           const std::vector<double>& weights,
                                           const SelectedMetric* metric = nullptr);

} // namespace xmvb::vb
