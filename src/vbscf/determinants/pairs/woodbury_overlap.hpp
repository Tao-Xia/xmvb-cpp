#pragma once

#include <optional>
#include <vector>

#include <Eigen/Core>

#include "vbscf/determinants/contracts/types.hpp"

namespace xmvb::vb {

/**
 * @brief Updates a regular occupied-overlap block after changing the right string.
 *
 * The changed occupied orbitals replace rows of the overlap block.  A
 * Woodbury update is returned only when it is cheaper than a full
 * factorization and the resulting inverse passes backward-error and condition
 * certificates.  Callers must evaluate the pair exactly when no value is
 * returned.
 */
std::optional<DeterminantOverlapResult> try_woodbury_right_overlap_update(
    const std::vector<int>& occupied_left,
    const std::vector<int>& occupied_right_old,
    const std::vector<int>& occupied_right_new,
    const Eigen::Ref<const Eigen::MatrixXd>& active_overlap,
    const DeterminantOverlapResult& old_result);

}  // namespace xmvb::vb
