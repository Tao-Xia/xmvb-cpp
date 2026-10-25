#pragma once

#include <functional>
#include "mcscf/ci/integrals.hpp"
#include "integrals/ao/input.hpp"
#include "integrals/active/two_electron_result.hpp"

namespace xmvb::casscf {

/** @brief Active Hamiltonian and the inactive-electron plus nuclear constant. */
struct CasIntegrals {
    mcscf::ActiveIntegrals active;
    double core_energy = 0.0;
};

/** Owned accepted exact-CAS forward products, reused by orbital derivatives. */
struct PreparedCasIntegrals {
    CasIntegrals integrals;
    xmvb::integrals::ActiveSpaceTwoElectronResult two_electron;
    Eigen::MatrixXd core_density;
    Eigen::MatrixXd fock;
};

/** @brief Read one chemists' MO Coulomb integral without requiring an MO^4 array. */
using MoCoulomb = std::function<double(int, int, int, int)>;

/** @brief Restrict orthonormal MO integrals to a complete active space.
 * one_electron and coulomb refer to the full MO basis, with doubly
 * occupied inactive orbitals first and active orbitals immediately after.
 * The returned active one-electron operator includes the inactive potential.
 * Only requested core/active entries are read; virtual four-index integrals
 * are neither requested nor stored.
 */
CasIntegrals make_active_integrals(const Eigen::Ref<const Eigen::MatrixXd>& one_electron,
                                   const MoCoulomb& coulomb,
                                   int n_inactive,
                                   int n_active,
                                   double nuclear_energy);

/** @brief Prepare exact CASCI integrals directly from an orthonormal AO frame.
 * orbitals is the full AO-by-MO matrix, with core and active columns first.
 * The caller supplies an AO-metric orthonormal frame. No VB structures,
 * nonorthogonal orbital preparation or virtual four-index ERIs are used.
 */
CasIntegrals make_active_integrals(const integrals::AoIntegralInput& ao,
                                   const Eigen::Ref<const Eigen::MatrixXd>& orbitals,
                                   int n_inactive,
                                   int n_active,
                                   double nuclear_energy);

/** Exact AO→CAS preparation with one core AO response and one active ERI transform. */
PreparedCasIntegrals prepare_active_integrals(const integrals::AoIntegralInput& ao,
                                              const Eigen::Ref<const Eigen::MatrixXd>& orbitals,
                                              int n_inactive,
                                              int n_active,
                                              double nuclear_energy);

} // namespace xmvb::casscf
