#pragma once

#include <Eigen/Core>

namespace xmvb::mcscf {

/** @brief Real spatial integrals in an orthonormal active orbital basis.
 * Pair entries use core::PairIndex::packed_pair_index. pair_kernel(P,Q)
 * stores the unweighted chemists' Coulomb integral (pq|rs), not an
 * antisymmetrized integral. No overlap or orbital-transform metadata belongs
 * to this physical CI interface.
 */
struct ActiveIntegrals {
    Eigen::MatrixXd one_electron;
    Eigen::MatrixXd pair_kernel;
};

} // namespace xmvb::mcscf
