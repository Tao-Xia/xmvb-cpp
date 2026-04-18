#pragma once

#include <vector>

#include "vb/biorthogonal_vbscf/biorthogonal_orbital_frame.hpp"
#include "vb/orbital/active_space_one_electron_result.hpp"
#include "vb/orbital/active_space_two_electron_result.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

/**
 * @brief Left/right active-space integral layer induced by one dual orbital frame.
 *
 * `left_dual_from_right_transform` stores the orbital-space map
 *
 * `X^{-1} = (C^T S C)^{-1}`
 *
 * that turns right/right active integrals into left/right biorthogonal
 * integrals. `left_right_one_electron` stores the dense one-electron matrix
 *
 * `h^{LR} = X^{-1} h^{RR}`
 *
 * with rows indexed by left determinant orbitals and columns indexed by right
 * determinant orbitals.
 *
 * The two-electron layer is not materialized as a dense four-index tensor in
 * this first prototype. Instead, left/right two-electron integrals are
 * evaluated on demand from the existing right/right packed active-space kernel.
 */
struct BiorthogonalOrbitalIntegrals {
  Eigen::MatrixXd left_dual_from_right_transform;
  Eigen::MatrixXd left_right_one_electron;
  int n_orbitals = 0;
};

/**
 * @brief Builds the first biorthogonal active-space integral layer from `HHO`.
 *
 * `right_right_one_electron` stores the existing active-space one-electron
 * matrix `h^{RR}` in column-major `(n_orbitals, n_orbitals)` storage.
 */
BiorthogonalOrbitalIntegrals build_biorthogonal_orbital_integrals(
    const BiorthogonalOrbitalFrame& orbital_frame,
    const std::vector<double>& right_right_one_electron);

/**
 * @brief Builds the integral layer directly from active-space overlap and `HHO`.
 *
 * This overload is the algebraic shortcut used by the first end-to-end
 * biorthogonal forward prototype. The right-orbital overlap matrix
 *
 * `X = C^T S C`
 *
 * is already available as the active-space overlap `ovlp_act`, so the dual
 * transform can be formed directly as `X^{-1}` without rebuilding the AO-side
 * left orbital table first.
 */
BiorthogonalOrbitalIntegrals build_biorthogonal_orbital_integrals(
    int n_orbitals,
    const std::vector<double>& right_orbital_overlap,
    const std::vector<double>& right_right_one_electron);

/**
 * @brief Builds the integral layer directly from one active-space result.
 */
BiorthogonalOrbitalIntegrals build_biorthogonal_orbital_integrals(
    const BiorthogonalOrbitalFrame& orbital_frame,
    const ActiveSpaceOneElectronResult& right_right_one_electron_result);

/**
 * @brief Validates dimensions and finiteness of the integral layer.
 */
void validate_biorthogonal_orbital_integrals(
    const BiorthogonalOrbitalIntegrals& orbital_integrals);

/**
 * @brief Evaluates one mixed active-space two-electron integral on demand.
 *
 * The returned value is the mixed left/right active-space integral
 *
 * `g^{bi}_{r_1 l_1 r_2 l_2}`
 *
 * compatible with the legacy packed `GGO` ordering, where each pair combines
 * one right orbital and one left orbital on the same electron coordinate. The
 * exact value is
 *
 * `g^{bi}_{r_1 l_1 r_2 l_2} = <\widetilde \phi_{l_1} \widetilde \phi_{l_2}| r_{12}^{-1} | \phi_{r_1} \phi_{r_2}>`
 *
 * and is evaluated by transforming only the two left legs with `X^{-1}`.
 */
double evaluate_biorthogonal_two_electron_integral(
    int right_first_orbital,
    int left_first_orbital,
    int right_second_orbital,
    int left_second_orbital,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view);

/**
 * @brief Evaluates one left/right two-electron integral from a forward result.
 */
double evaluate_biorthogonal_two_electron_integral(
    int right_first_orbital,
    int left_first_orbital,
    int right_second_orbital,
    int left_second_orbital,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronResult& right_right_two_electron_result);

}  // namespace xmvb::vb::biorthogonal_vbscf
