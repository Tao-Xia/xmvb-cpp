#pragma once

#include <Eigen/Core>

#include "integrals/ao/input.hpp"
#include "integrals/ao/ri.hpp"

namespace xmvb::integrals {

/** Coefficients of the physical AO Coulomb and exchange matrices.
 * For a total-electron density P, J(mu,nu) = sum_rs (mu nu|r s) P(r,s)
 * and K(mu,nu) = sum_rs (mu r|nu s) P(r,s).
 */
struct JkWeights {
    double coulomb = 1.0;
    double exchange = 0.0;
};

/** Contract a symmetric total-electron AO density against exact ERIs. */
Eigen::MatrixXd contract_jk(const Eigen::Ref<const Eigen::MatrixXd>& density,
                            const AoIntegralInput& ao,
                            JkWeights weights = {});

/** Contract a symmetric total-electron AO density against RI factors. */
Eigen::MatrixXd contract_jk(const Eigen::Ref<const Eigen::MatrixXd>& density,
                            const RiAoFactorization& ri,
                            JkWeights weights = {});

} // namespace xmvb::integrals
