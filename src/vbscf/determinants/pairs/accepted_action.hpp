#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/determinants/pairs/accepted_tile.hpp"

namespace xmvb::vb {

/** Same-spin overlap and Hamiltonian images of a dense vector block. */
struct AcceptedSpinPairActionResult {
  Eigen::MatrixXd overlap;
  Eigen::MatrixXd hamiltonian;
};

/**
 * @brief Applies accepted same-spin pair matrices without quadratic storage.
 *
 * Pair values are generated one bounded tile at a time. Each tile contributes
 * directly to the corresponding output rows and is released before the next
 * tile is built. The input and output blocks have one row per unique spin
 * string; no persistent pair matrix is materialized.
 */
AcceptedSpinPairActionResult apply_accepted_spin_pair_action(
    const AcceptedPairTileProvider& provider,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron,
    const ActiveSpaceTwoElectronResult& active_two_electron,
    const Eigen::Ref<const Eigen::MatrixXd>& vectors,
    std::size_t workspace_bytes = 256ULL * 1024ULL * 1024ULL);

}  // namespace xmvb::vb
