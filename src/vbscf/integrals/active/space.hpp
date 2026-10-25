#pragma once

#include <Eigen/Core>

#include "vbscf/core/contracts/input.hpp"
#include "integrals/active/one_electron_result.hpp"
#include "integrals/active/two_electron_result.hpp"
#include "vbscf/integrals/ao/one_electron/result.hpp"
#include "vbscf/orbitals/preparation/result.hpp"

namespace xmvb::vb {

struct PreparedActiveSpaceContext {
    int n_inactive_doubly_occupied_orbitals = 0;
    OrbitalPreparationResult orbital_result;
    AoEffectiveOneElectronResult ao_effective_one_electron_result;
    xmvb::integrals::ActiveSpaceOneElectronResult active_space_one_electron_result;
    xmvb::integrals::ActiveSpaceTwoElectronResult active_space_two_electron_result;
    double one_electron_reference_energy = 0.0;
};

struct PreparedActiveSpaceTimings {
    double orbital_preparation_wall_time_seconds = 0.0;
    double ao_effective_one_electron_wall_time_seconds = 0.0;
    double active_one_electron_wall_time_seconds = 0.0;
    double active_two_electron_wall_time_seconds = 0.0;
};

struct TimedPreparedActiveSpaceContext {
    PreparedActiveSpaceContext prepared_active_space;
    PreparedActiveSpaceTimings timings;
};

double compute_one_electron_reference_energy(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_matrix,
    const Eigen::Ref<const Eigen::MatrixXd>& ao_effective_h1e,
    const Eigen::Ref<const Eigen::MatrixXd>& ao_core_hamiltonian_matrix,
    int n_basis_functions);

/** Build both h and G in `C R^{-1}`. A supplied frame stays fixed at a
 * trial point; an explicitly empty supplied frame requests raw `C` instead.
 */
TimedPreparedActiveSpaceContext
prepare_timed_active_space_context(const VbScfInput& input,
                                   const Eigen::MatrixXd* fixed_integral_frame = nullptr);

/** @brief Map the active ERI adjoint to AO-by-active orbital coefficients.
 * Selects the integral representation; numerical contractions belong to the
 * common integral library. No inactive/virtual zero columns are materialized.
 */
Eigen::MatrixXd pullback_active_integrals(const VbScfInput& input,
                                          const integrals::ActiveSpaceTwoElectronResult& accepted,
                                          const std::vector<double>& packed_gradient);

} // namespace xmvb::vb
