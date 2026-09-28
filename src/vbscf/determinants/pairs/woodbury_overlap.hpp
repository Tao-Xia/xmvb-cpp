#pragma once

#include <optional>
#include <vector>

#include <Eigen/Core>

#include "vbscf/determinants/contracts/types.hpp"

namespace xmvb::vb {

/**
 * @brief Returns the precision-derived condition limit for fourth-order pair derivatives.
 *
 * The limit bounds amplification by the four inverse factors that can occur in
 * an orbital Hessian-vector product. Diagnostics reuse the same roundoff
 * budget when reporting spectral principal-angle conditioning.
 */
double regular_overlap_condition_limit() noexcept;

/**
 * @brief Tests whether a regular overlap inverse is safe for pair derivatives.
 *
 * The certificate combines a condition estimate with the backward error of
 * the supplied inverse.  It is shared by Woodbury updates and diagnostics so
 * both use exactly the same accuracy contract.
 */
bool is_certified_regular_overlap(
    const Eigen::Ref<const Eigen::MatrixXd>& overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& inverse_overlap);

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

/**
 * @brief Updates a regular occupied-overlap block after changing the left string.
 *
 * The changed occupied orbitals replace columns of the overlap block.  This is
 * the transpose companion of `try_woodbury_right_overlap_update()` and uses
 * the same backward-error and condition certificates.  Callers must evaluate
 * the pair exactly when no value is returned.
 */
std::optional<DeterminantOverlapResult> try_woodbury_left_overlap_update(
    const std::vector<int>& occupied_left_old,
    const std::vector<int>& occupied_left_new,
    const std::vector<int>& occupied_right,
    const Eigen::Ref<const Eigen::MatrixXd>& active_overlap,
    const DeterminantOverlapResult& old_result);

}  // namespace xmvb::vb
