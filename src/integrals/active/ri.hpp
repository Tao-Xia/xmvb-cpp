#pragma once

#include "integrals/ao/ri.hpp"
#include "integrals/active/two_electron_result.hpp"

namespace xmvb::integrals {

/** @brief Build packed four-index active integrals once from RI factors.
 * Auxiliary contraction uses bounded pair panels, not a second full kernel.
 * Factors remain available only for AO transformation and derivative pullbacks.
 */
std::vector<double> pack_ri_active_integrals(const Eigen::Ref<const Eigen::MatrixXd>& factors);

/** @brief Transform whitened AO factors to active orbital pairs.
 * Coefficients have AO rows and active columns, with no orthogonality
 * requirement. AO factors remain borrowed; accepted-point outputs are owned.
 */
ActiveSpaceTwoElectronResult
transform_ri_active_integrals(const RiAoFactorization& ao,
                              const Eigen::Ref<const Eigen::MatrixXd>& active_coefficients);

} // namespace xmvb::integrals
