#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/integrals/active/two_electron/construction/result.hpp"

namespace xmvb::vb {

/**
 * @brief Active-space integrals in a Cholesky-orthonormal orbital basis.
 */
struct OrthogonalActiveIntegrals {
  /** Upper factor in `S_active = R^T R`. */
  Eigen::MatrixXd orbital_transform;
  Eigen::MatrixXd one_electron;
  /** Dense Coulomb kernel over packed symmetric orbital pairs. */
  Eigen::MatrixXd pair_kernel;
  ActiveSpaceTwoElectronResult two_electron;
};

/**
 * @brief Orthogonalizes active integrals without changing their physical span.
 */
OrthogonalActiveIntegrals orthogonalize_active_integrals(
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& one_electron,
    const ActiveSpaceTwoElectronResult& two_electron,
    int n_active_orbitals);

/**
 * @brief Differentiates the Cholesky-orthogonalized active integrals.
 *
 * The returned `orbital_transform` is `delta R`; the other members are the
 * first derivatives of the orthonormal one- and two-electron integrals.
 */
OrthogonalActiveIntegrals orthogonalize_active_integral_direction(
    const OrthogonalActiveIntegrals& accepted,
    const std::vector<double>& overlap_direction,
    const Eigen::Ref<const Eigen::MatrixXd>& one_electron_direction,
    const std::vector<double>& packed_two_electron_direction,
    int n_active_orbitals);

}  // namespace xmvb::vb
