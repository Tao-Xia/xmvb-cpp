#include "vbscf/integrals/active/prepared_active_space.hpp"

#include <chrono>
#include <stdexcept>

#include <Eigen/Core>

#include "vbscf/integrals/ao/ri_integral_cache.hpp"
#include "vbscf/integrals/active/ri_active_space_two_electron_builder.hpp"

namespace xmvb::vb {

namespace {

bool use_standard_ri_active_space_path(const VbScfInput& input) {
  return input.standard_two_electron_mode ==
      StandardTwoElectronMode::ResolutionOfIdentity;
}

}  // namespace

double compute_one_electron_reference_energy(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_matrix,
    const Eigen::Ref<const Eigen::MatrixXd>& ao_effective_h1e,
    const Eigen::Ref<const Eigen::MatrixXd>& ao_core_hamiltonian_matrix,
    int n_basis_functions) {
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }

  if (inactive_density_matrix.rows() != n_basis_functions ||
      inactive_density_matrix.cols() != n_basis_functions ||
      ao_effective_h1e.rows() != n_basis_functions ||
      ao_effective_h1e.cols() != n_basis_functions ||
      ao_core_hamiltonian_matrix.rows() != n_basis_functions ||
      ao_core_hamiltonian_matrix.cols() != n_basis_functions) {
    throw std::invalid_argument("one-electron reference energy input size mismatch");
  }
  return (inactive_density_matrix.array() *
          (ao_effective_h1e.array() + ao_core_hamiltonian_matrix.array())).sum();
}

TimedPreparedActiveSpaceContext prepare_timed_active_space_context(
    const VbScfInput& input,
    const ActiveSpaceOrbitalPreparer& orbital_preparer,
    const AoEffectiveOneElectronBuilder& ao_effective_one_electron_builder,
    const ActiveSpaceOneElectronBuilder& active_space_one_electron_builder,
    const ActiveSpaceTwoElectronBuilder& active_space_two_electron_builder) {
  TimedPreparedActiveSpaceContext timed_context;
  auto& context = timed_context.prepared_active_space;
  const bool use_ri_active_space = use_standard_ri_active_space_path(input);
  const RiAoFactorization* ao_ri_result = nullptr;
  if (use_ri_active_space) {
    ao_ri_result = &ensure_vbscf_input_ri_cache(input);
  }

  auto stage_start_time = std::chrono::steady_clock::now();
  context.n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) /
      2;
  context.orbital_result =
      orbital_preparer.prepare(input.orbital_preparation_input);
  timed_context.timings.orbital_preparation_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();

  stage_start_time = std::chrono::steady_clock::now();
  if (use_ri_active_space) {
    context.ao_effective_one_electron_result =
        ao_effective_one_electron_builder.build(
            context.orbital_result,
            input.ao_integral_input.ao_core_hamiltonian_matrix,
            *ao_ri_result,
            input.ao_integral_input.n_basis_functions,
            context.n_inactive_doubly_occupied_orbitals);
  } else {
    if (input.ao_integral_input.ao_two_electron_integral_values.empty()) {
      throw std::invalid_argument(
          "standard exact AO h1e path requires materialized AO two-electron integrals");
    }
    context.ao_effective_one_electron_result =
        ao_effective_one_electron_builder.build(
            context.orbital_result.inactive_density_matrix,
            input.ao_integral_input);
  }
  timed_context.timings.ao_effective_one_electron_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();

  stage_start_time = std::chrono::steady_clock::now();
  context.active_space_one_electron_result =
      active_space_one_electron_builder.build(
          context.ao_effective_one_electron_result.ao_effective_h1e,
          context.orbital_result.auxiliary_orbital_matrix,
          input.ao_integral_input.n_basis_functions,
          context.n_inactive_doubly_occupied_orbitals,
          input.orbital_preparation_input.n_active_orbitals);
  timed_context.timings.active_one_electron_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();

  stage_start_time = std::chrono::steady_clock::now();
  if (use_ri_active_space) {
    RiActiveSpaceTwoElectronBuilder ri_builder;
    context.active_space_two_electron_result =
        ri_builder.build(
            *ao_ri_result,
            context.orbital_result,
            input.ao_integral_input.n_basis_functions,
            input.orbital_preparation_input.n_active_orbitals,
            {
                // Keep the RI factors for reverse-mode, but also reconstruct a
                // packed active-space `GGO` cache so determinant-pair kernels
                // can use direct lookups instead of repeating auxiliary-length
                // dot products inside the hot loops.
                .reconstruct_packed_integrals = true,
            });
  } else {
    context.active_space_two_electron_result =
        active_space_two_electron_builder.build(
            input.ao_integral_input,
            context.orbital_result,
            input.orbital_preparation_input.n_active_orbitals);
  }
  timed_context.timings.active_two_electron_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();
  context.one_electron_reference_energy =
      compute_one_electron_reference_energy(
          context.orbital_result.inactive_density_matrix,
          context.ao_effective_one_electron_result.ao_effective_h1e,
          input.ao_integral_input.ao_core_hamiltonian_matrix,
          input.ao_integral_input.n_basis_functions);
  return timed_context;
}

}  // namespace xmvb::vb
