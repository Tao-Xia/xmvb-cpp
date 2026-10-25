#pragma once

#include <cstddef>
#include <vector>
#include <Eigen/Core>

#include "integrals/response/ri.hpp"

namespace xmvb::integrals::detail {

// Canonical packed-pair multiplicities shared by the standalone and AO-fused jets.
Eigen::MatrixXd build_ri_active_pair_adjoint(const std::vector<double>& packed_gradient,
                                             const RiActiveTwoElectronResponseCache& cache);

// Accepted rows are laid out as [active orbital][auxiliary][AO].
void fill_ri_factor_direction_tile(const RiActiveTwoElectronResponseCache& cache,
                                   const Eigen::Ref<const Eigen::MatrixXd>& accepted,
                                   const Eigen::Ref<const Eigen::MatrixXd>& direction,
                                   int first_aux,
                                   int count,
                                   Eigen::MatrixXd& result);

/** @brief Shared rectangular factor pullback, with disjoint AO output rows.
 * Packs only a bounded auxiliary-by-active cotangent, never copies the
 * one-index factors or allocates per-thread AO-by-active partial results.
 */
void accumulate_ri_factor_tile(const RiActiveTwoElectronResponseCache& cache,
                               const Eigen::Ref<const Eigen::MatrixXd>& factors,
                               int auxiliary_stride,
                               int auxiliary_offset,
                               const Eigen::Ref<const Eigen::MatrixXd>& adjoint,
                               int adjoint_offset,
                               int count,
                               std::size_t workspace_bytes,
                               Eigen::MatrixXd& result);

// The two fixed-jet terms are (L D)(B G) and (L C)(delta B G).
// Strides/offsets permit a full resident accepted slab and a narrow direction
// slab to share this contraction without repacking either AO-by-active image.
void accumulate_ri_fixed_factor_tile(const RiActiveTwoElectronResponseCache& cache,
                                     const Eigen::Ref<const Eigen::MatrixXd>& accepted,
                                     int accepted_aux_stride,
                                     int accepted_aux_offset,
                                     const Eigen::Ref<const Eigen::MatrixXd>& direction,
                                     int direction_aux_stride,
                                     int direction_aux_offset,
                                     const Eigen::Ref<const Eigen::MatrixXd>& accepted_adjoint,
                                     const Eigen::Ref<const Eigen::MatrixXd>& directional_adjoint,
                                     int adjoint_offset,
                                     int count,
                                     std::size_t worker_budget_bytes,
                                     Eigen::MatrixXd& result);

} // namespace xmvb::integrals::detail
