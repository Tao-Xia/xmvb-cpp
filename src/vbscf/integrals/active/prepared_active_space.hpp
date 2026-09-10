#pragma once

#include <Eigen/Core>

#include "vbscf/core/vbscf_input.hpp"
#include "vbscf/integrals/active/active_space_one_electron_builder.hpp"
#include "vbscf/integrals/active/active_space_one_electron_result.hpp"
#include "vbscf/orbitals/preparation/preparer.hpp"
#include "vbscf/integrals/active/active_space_two_electron_builder.hpp"
#include "vbscf/integrals/active/active_space_two_electron_result.hpp"
#include "vbscf/integrals/ao/one_electron/builder.hpp"
#include "vbscf/integrals/ao/one_electron/result.hpp"
#include "vbscf/orbitals/preparation/result.hpp"

namespace xmvb::vb {

struct PreparedActiveSpaceContext {
  int n_inactive_doubly_occupied_orbitals = 0;
  OrbitalPreparationResult orbital_result;
  AoEffectiveOneElectronResult ao_effective_one_electron_result;
  ActiveSpaceOneElectronResult active_space_one_electron_result;
  ActiveSpaceTwoElectronResult active_space_two_electron_result;
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

TimedPreparedActiveSpaceContext prepare_timed_active_space_context(
    const VbScfInput& input,
    const ActiveSpaceOrbitalPreparer& orbital_preparer,
    const AoEffectiveOneElectronBuilder& ao_effective_one_electron_builder,
    const ActiveSpaceOneElectronBuilder& active_space_one_electron_builder,
    const ActiveSpaceTwoElectronBuilder& active_space_two_electron_builder);

}  // namespace xmvb::vb
