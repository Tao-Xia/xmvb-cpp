#pragma once

#include <vector>

#include <Eigen/Core>

namespace xmvb::vb {

/**
 * @brief Truncated coefficients of an exterior generating determinant.
 *
 * `coefficients[k]` is the coefficient of `t^k` in `det(I + t A)`.
 * No exterior tensor or reduced density matrix is stored.
 */
struct ContractedExteriorJet {
  std::vector<double> coefficients;
};

/**
 * @brief Builds one anchor jet for a dense transition channel.
 *
 * This anchor operation is performed once per traversal component.  Graph
 * edges must use `update_contracted_exterior_jet`, whose scaling is quadratic
 * in the channel dimension for fixed update rank and truncation order.
 */
ContractedExteriorJet build_contracted_exterior_jet(
    const Eigen::Ref<const Eigen::MatrixXd>& channel,
    int maximum_order);

/**
 * @brief Updates a jet after `A <- A + L R^T` without rebuilding powers.
 *
 * The update uses
 *
 * `det(I + t(A + L R^T))`
 * `= det(I + tA) det(I + t R^T (I + tA)^-1 L)`.
 *
 * Only the thin Krylov moments `R^T A^j L` are formed.  The edge cost is
 * `O(k n^2 r + poly(k,r))` for maximum order `k`, dimension `n`, and update
 * rank `r`; the formula does not require `A` to be invertible.
 */
void update_contracted_exterior_jet(
    const Eigen::Ref<const Eigen::MatrixXd>& channel,
    const Eigen::Ref<const Eigen::MatrixXd>& update_left,
    const Eigen::Ref<const Eigen::MatrixXd>& update_right,
    ContractedExteriorJet* jet);

}  // namespace xmvb::vb
