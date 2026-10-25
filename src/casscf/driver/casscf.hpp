#pragma once

#include <string>
#include <functional>
#include <vector>
#include <Eigen/Core>
#include "casscf/driver/casci.hpp"

namespace xmvb::casscf {

struct CasscfIteration;

/** @brief Exact, real, single-state, complete-CAS optimization settings. */
struct CasscfOptions {
    double energy_tolerance = 1e-7;
    double gradient_tolerance = 1e-3;
    int max_iterations = 100;
    CasciOptions ci;
    std::function<void(const CasscfIteration&)> observe_iteration;
};

struct CasscfIteration {
    int iteration = 0;
    int hvp_actions = 0;
    int keyframes = 0;
    int ci_actions = 0;
    int micro_iterations = 0;
    double energy = 0.0;
    double energy_change = 0.0;
    double gradient_norm = 0.0;
    double wall_seconds = 0.0;
};

struct CasscfResult {
    bool converged = false;
    std::string termination_reason;
    double initial_energy = 0.0;
    double initial_gradient_norm = 0.0;
    double energy = 0.0;
    double gradient_norm = 0.0;
    int hvp_actions = 0;
    int ci_actions = 0;
    Eigen::MatrixXd orbitals, coefficients;
    std::vector<CasscfIteration> iterations;
};

/** @brief Optimize a pure-spin CAS in a supplied complete S-orthonormal frame.
 * AO integrals remain shared and borrowed. The active CI has no VB structure
 * representation. Orbital and CI updates use the common matrix-free CIAH.
 */
CasscfResult solve_casscf(const integrals::AoIntegralInput& ao,
                          const Eigen::MatrixXd& orbitals,
                          int ncore,
                          int nactive,
                          int n_electrons,
                          int twice_spin,
                          double nuclear_energy,
                          const CasscfOptions& options = {});

} // namespace xmvb::casscf
