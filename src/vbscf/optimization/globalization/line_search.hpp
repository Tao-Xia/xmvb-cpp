#pragma once

#include <Eigen/Core>

#include "vbscf/optimization/objective/function.hpp"
#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/layout.hpp"

namespace xmvb::vb {

/** @brief Result of the Moré--Thuente search used by XMVB L-BFGS. */
struct MoreThuenteResult {
  Eigen::VectorXd parameters;
  Eigen::VectorXd gradient;
  double energy = 0.0;
  double step = 0.0;
  int evaluations = 0;
};

/**
 * @brief Finds a raw-coordinate step satisfying the strong Wolfe conditions.
 *
 * The constants and interpolation are the Moré--Thuente implementation used
 * by XMVB 4.0: \f$c_1=10^{-4}\f$, \f$c_2=0.9\f$, at most 20 evaluations.
 */
bool try_xmvb_more_thuente_line_search(
    VbScfObjective* objective,
    const Eigen::VectorXd& parameters,
    double energy,
    const Eigen::VectorXd& gradient,
    const Eigen::VectorXd& direction,
    double initial_step,
    MoreThuenteResult* result);

/** @brief Retracts a reduced step and packs the resulting orbital parameters. */
Eigen::VectorXd build_nonredundant_lifted_trial_parameters(
    const OrbitalPreparationInput& current_orbital_input,
    const OrbitalChart& current_space,
    const SparseParameterLayout& parameter_view,
    const Eigen::VectorXd& reduced_step);

bool try_armijo_backtracking_nonredundant_direction(
    VbScfObjective* objective,
    const OrbitalPreparationInput& current_orbital_input,
    const OrbitalChart& current_space,
    const SparseParameterLayout& parameter_view,
    const Eigen::VectorXd& current_parameters,
    double current_energy,
    const Eigen::VectorXd& current_gradient,
    const Eigen::VectorXd& reduced_search_direction,
    const Eigen::VectorXd& packed_tangent_search_direction,
    double initial_step,
    double minimum_step,
    double armijo_constant,
    Eigen::VectorXd* accepted_parameters,
    Eigen::VectorXd* accepted_gradient,
    double* accepted_energy,
    bool* accepted_chart_changed = nullptr);

}  // namespace xmvb::vb
