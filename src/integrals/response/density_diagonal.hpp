#pragma once

#include "integrals/ao/ri.hpp"
#include "integrals/ao/ri_transform.hpp"
#include "integrals/response/orbital_pair.hpp"

namespace xmvb::integrals {

/** @brief Projected density responses required by the analytic diagonal. */
struct DensityDiagonals {
    Eigen::MatrixXd core, active, core_coulomb;
};

/** @brief Contract all RI density channels without complete AO responses.
 * Core sources are individual orbital projectors. Active source columns are
 * symmetric weights flattened column-major; active_coulomb(p,a*n_active+b)
 * is (p p|a b), already available from the accepted integral transform.
 * Only occupied one-index transforms and their projected second moments are
 * generated. Scratch is auxiliary/pair tiled; workers own small active moments.
 */
DensityDiagonals
contract_ri_density_diagonals(const RiAoFactorization& ri,
                              const Eigen::Ref<const Eigen::MatrixXd>& orbitals,
                              int n_core,
                              int n_active,
                              const Eigen::Ref<const Eigen::MatrixXd>& active_weights,
                              const Eigen::Ref<const OrbitalPairMatrix>& active_coulomb,
                              std::size_t workspace_bytes = workspace::available());

} // namespace xmvb::integrals
