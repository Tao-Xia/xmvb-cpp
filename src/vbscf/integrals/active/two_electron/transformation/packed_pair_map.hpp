#pragma once

#include <Eigen/Core>

namespace xmvb::vb {

/**
 * @brief Row-major matrix used by the representation-neutral packed pair map.
 *
 * Rows use canonical packed AO-pair order and columns use canonical packed
 * active-orbital-pair order.
 */
using PackedOrbitalPairMapMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/**
 * @brief Builds the packed AO-pair to active-pair orbital map `Q(C)`.
 *
 * @param dense_active_coefficients AO-by-active orbital coefficients `C`.
 * @param pair_map Output with packed AO-pair rows and active-pair columns.
 */
void build_packed_orbital_pair_map(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    PackedOrbitalPairMapMatrix* pair_map);

/**
 * @brief Builds the tangent `Q'(C)[D]` for all packed AO-pair rows.
 */
void build_packed_orbital_pair_map_directional_derivative(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    PackedOrbitalPairMapMatrix* directional_pair_map);

/**
 * @brief Builds a contiguous packed AO-row range of `Q'(C)[D]`.
 *
 * The output row zero corresponds to global packed row `row_begin`.
 */
void build_packed_orbital_pair_map_directional_derivative_rows(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    Eigen::Index row_begin,
    Eigen::Index row_count,
    PackedOrbitalPairMapMatrix* directional_pair_map);

/**
 * @brief Accumulates the pair-map adjoint `Q'(C)^*[pair_adjoint]`.
 *
 * Existing entries in `dense_active_gradient` are preserved when its shape is
 * already AO-by-active; otherwise the output is resized and zero-initialized.
 */
void accumulate_packed_orbital_pair_map_adjoint(
    const Eigen::Ref<const PackedOrbitalPairMapMatrix>& pair_adjoint,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    Eigen::MatrixXd* dense_active_gradient);

/**
 * @brief Accumulates the adjoint of a contiguous packed AO-row range.
 *
 * `pair_adjoint` contains local rows while `row_begin` identifies their
 * canonical global packed AO-pair offset.
 */
void accumulate_packed_orbital_pair_map_adjoint_rows(
    const Eigen::Ref<const PackedOrbitalPairMapMatrix>& pair_adjoint,
    Eigen::Index row_begin,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    Eigen::MatrixXd* dense_active_gradient);

}  // namespace xmvb::vb
