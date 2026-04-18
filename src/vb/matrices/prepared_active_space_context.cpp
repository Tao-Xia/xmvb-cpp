#include "vb/matrices/prepared_active_space_context.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <Eigen/Core>

#include "vb/matrices/cpp_vb_input_ri_cache.hpp"
#include "vb/orbital/ri_active_space_two_electron_builder.hpp"

namespace xmvb::vb {

namespace {

bool use_standard_ri_active_space_path(const CppVbInput& input) {
  return input.standard_two_electron_mode ==
      StandardTwoElectronMode::ResolutionOfIdentity;
}

bool reconstruct_packed_ri_active_space_integrals_enabled() {
  const char* disable_flag = std::getenv("XMVB_CPP_DISABLE_RI_ACTIVE_PACKED_GGO");
  if (disable_flag == nullptr || disable_flag[0] == '\0') {
    return true;
  }
  return std::strcmp(disable_flag, "0") == 0 ||
      std::strcmp(disable_flag, "false") == 0 ||
      std::strcmp(disable_flag, "FALSE") == 0;
}

}  // namespace

double compute_one_electron_reference_energy(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_matrix,
    const std::vector<double>& ao_effective_h1e,
    const std::vector<double>& ao_core_hamiltonian_matrix,
    int n_basis_functions) {
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }

  const std::size_t matrix_size =
      xmvb::to_size(n_basis_functions) * xmvb::to_size(n_basis_functions);
  if (inactive_density_matrix.rows() != n_basis_functions ||
      inactive_density_matrix.cols() != n_basis_functions ||
      ao_effective_h1e.size() != matrix_size ||
      ao_core_hamiltonian_matrix.size() != matrix_size) {
    throw std::invalid_argument("one-electron reference energy input size mismatch");
  }

  const Eigen::Map<const Eigen::MatrixXd> effective_h1e(
      ao_effective_h1e.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const Eigen::MatrixXd> core_hamiltonian(
      ao_core_hamiltonian_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  return (inactive_density_matrix.array() *
          (effective_h1e.array() + core_hamiltonian.array())).sum();
}

TimedPreparedActiveSpaceContext prepare_timed_active_space_context(
    const CppVbInput& input,
    const ActiveSpaceOrbitalPreparer& orbital_preparer,
    const AoEffectiveOneElectronBuilder& ao_effective_one_electron_builder,
    const ActiveSpaceOneElectronBuilder& active_space_one_electron_builder,
    const ActiveSpaceTwoElectronBuilder& active_space_two_electron_builder) {
  TimedPreparedActiveSpaceContext timed_context;
  auto& context = timed_context.prepared_active_space;
  const bool use_ri_active_space = use_standard_ri_active_space_path(input);
  const LibcintRiIntegralProviderResult* ao_ri_result = nullptr;
  if (use_ri_active_space) {
    ao_ri_result = &ensure_cpp_vb_input_ri_cache(input);
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
            input.ao_integral_input.ao_core_hamiltonian_matrix.vector(),
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
                .reconstruct_packed_integrals =
                    reconstruct_packed_ri_active_space_integrals_enabled(),
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

PreparedActiveSpaceContext prepare_active_space_context(
    const CppVbInput& input,
    const ActiveSpaceOrbitalPreparer& orbital_preparer,
    const AoEffectiveOneElectronBuilder& ao_effective_one_electron_builder,
    const ActiveSpaceOneElectronBuilder& active_space_one_electron_builder,
    const ActiveSpaceTwoElectronBuilder& active_space_two_electron_builder) {
  return prepare_timed_active_space_context(
             input,
             orbital_preparer,
             ao_effective_one_electron_builder,
             active_space_one_electron_builder,
             active_space_two_electron_builder)
      .prepared_active_space;
}

}  // namespace xmvb::vb
