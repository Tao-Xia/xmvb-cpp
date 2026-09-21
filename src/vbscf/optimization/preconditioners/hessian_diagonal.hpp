#pragma once

#include <functional>
#include <vector>

#include <Eigen/Core>

#include "vbscf/orbitals/charts/chart.hpp"

namespace xmvb::vb {

class ExactHvpOperator;

using ReducedBlockHvp =
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)>;

/**
 * @brief Extracts @f$\operatorname{diag}(U^T H U)@f$ without forming @f$H@f$.
 *
 * Coordinate directions are grouped by the natural per-orbital blocks of the
 * nonredundant chart. Only the diagonal entries of each returned HVP block are
 * retained.
 */
Eigen::VectorXd extract_reduced_hessian_diagonal(
    int reduced_size,
    const std::vector<OrbitalChart::ReducedBlock>& blocks,
    const ReducedBlockHvp& apply_block);

/**
 * @brief Builds the actual VBSCF orbital-block Hessian diagonal.
 *
 * The action contains direct core, fixed pullback, and local active response.
 * Structure response is excluded because structure amplitudes are explicit
 * variables in the coupled NEO equation.
 */
Eigen::VectorXd build_reduced_hessian_diagonal(
    const ExactHvpOperator& hessian,
    const OrbitalChart& chart);

}  // namespace xmvb::vb
