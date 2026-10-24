#pragma once

#include <vector>

#include "integrals/active/two_electron_result.hpp"

namespace xmvb::integrals {

/** @brief Recover raw active h as R^T hbar R, without AO contractions. */
Eigen::MatrixXd materialize_raw_one_electron(const Eigen::Ref<const Eigen::MatrixXd>& one_electron,
                                             const Eigen::MatrixXd& upper);

/**
 * @brief Transform one packed pair-of-pairs tensor to the raw orbital frame.
 *
 * The input packs a pair-exchange-symmetric kernel. Nonempty upper is the
 * square frame R with Cbar = C R^-1; empty upper returns a copy of packed.
 * No AO integral work is performed.
 */
std::vector<double> materialize_raw_pair_kernel(const std::vector<double>& packed,
                                                const Eigen::MatrixXd& upper);

/**
 * @brief Express a raw packed-kernel cotangent in the fixed accepted frame.
 *
 * This is the adjoint of `materialize_raw_pair_kernel`: raw input entries
 * represent a symmetric packed kernel, and the returned cotangent is packed
 * in the frame described by `upper`. Empty `upper` returns a copy.
 */
std::vector<double>
materialize_framed_pair_kernel_adjoint(const std::vector<double>& raw_packed_gradient,
                                       const Eigen::MatrixXd& upper);

/**
 * @brief Make an owned raw-orbital-frame copy of active two-electron data.
 *
 * When the input stores Cbar = C R^-1 and its pair data in the Cbar frame,
 * this transforms the exact packed kernel and/or RI pair factors by R,
 * restores the dense coefficient cache to C, and discards intermediate caches
 * that cannot be transformed without AO data. Empty R returns a plain copy.
 * This is an explicit adapter for consumers requiring the raw frame; it does
 * not recompute AO integrals or materialize a determinant carrier.
 */
ActiveSpaceTwoElectronResult
materialize_raw_active_integrals(const ActiveSpaceTwoElectronResult& framed);

} // namespace xmvb::integrals
