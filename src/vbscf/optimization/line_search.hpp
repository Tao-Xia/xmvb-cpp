#pragma once

#include <Eigen/Core>
#include <LBFGS.h>

#include "vbscf/optimization/vbscf_objective.hpp"
#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/layout.hpp"

namespace xmvb::vb {

bool try_build_nonredundant_lifted_trial_parameters(
    const OrbitalPreparationInput& current_orbital_input,
    const OrbitalChart& current_space,
    const SparseParameterLayout& parameter_view,
    const Eigen::VectorXd& reduced_step,
    Eigen::VectorXd* trial_parameters);

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
    double* accepted_energy);

bool try_steepest_descent_armijo_step(
    VbScfObjective* objective,
    const LBFGSpp::LBFGSParam<double>& param,
    const Eigen::VectorXd& start_parameters,
    const Eigen::VectorXd& start_gradient,
    double start_energy,
    double initial_step,
    Eigen::VectorXd* accepted_parameters,
    Eigen::VectorXd* accepted_gradient,
    double* accepted_energy,
    double* accepted_step);

}  // namespace xmvb::vb
