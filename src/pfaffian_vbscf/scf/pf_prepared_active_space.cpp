#include "pfaffian_vbscf/scf/pf_prepared_active_space.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>

#include "runtime/libcint_ri_integral_provider.hpp"
#include "vb/matrices/cpp_vb_input_ri_cache.hpp"
#include "vb/matrices/prepared_active_space_context.hpp"
#include "vb/orbital/active_space_one_electron_builder.hpp"
#include "vb/orbital/active_space_orbital_preparer.hpp"
#include "vb/orbital/active_space_two_electron_builder.hpp"
#include "vb/orbital/ao_effective_one_electron_builder.hpp"
#include "vb/orbital/ri_active_space_two_electron_builder.hpp"

namespace xmvb::pfaffian_vbscf {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(const Clock::time_point& start_time) {
  return std::chrono::duration<double>(Clock::now() - start_time).count();
}

bool pf_ri_two_electron_enabled() {
  const char* mode = std::getenv("XMVB_CPP_PF_TWO_ELECTRON_MODE");
  if (mode != nullptr && mode[0] != '\0') {
    return std::strcmp(mode, "ri") == 0 ||
        std::strcmp(mode, "RI") == 0 ||
        std::strcmp(mode, "df") == 0 ||
        std::strcmp(mode, "DF") == 0;
  }

  const char* flag = std::getenv("XMVB_CPP_PF_ENABLE_RI");
  if (flag == nullptr || flag[0] == '\0') {
    return false;
  }
  return std::strcmp(flag, "0") != 0 &&
      std::strcmp(flag, "false") != 0 &&
      std::strcmp(flag, "FALSE") != 0;
}

bool use_pf_ri_two_electron_mode(
    const xmvb::vb::CppVbInput& input) {
  switch (input.pf_two_electron_mode) {
    case xmvb::vb::PfTwoElectronMode::Exact:
      return false;
    case xmvb::vb::PfTwoElectronMode::ResolutionOfIdentity:
      return true;
    case xmvb::vb::PfTwoElectronMode::Auto:
      return pf_ri_two_electron_enabled();
  }
  return false;
}

}  // namespace

xmvb::vb::TimedPreparedActiveSpaceContext prepare_pf_timed_active_space_context(
    const xmvb::vb::CppVbInput& input) {
  if (!use_pf_ri_two_electron_mode(input)) {
    return xmvb::vb::prepare_timed_active_space_context(
        input,
        xmvb::vb::ActiveSpaceOrbitalPreparer(),
        xmvb::vb::AoEffectiveOneElectronBuilder(),
        xmvb::vb::ActiveSpaceOneElectronBuilder(),
        xmvb::vb::ActiveSpaceTwoElectronBuilder());
  }

  xmvb::vb::TimedPreparedActiveSpaceContext timed_context;
  auto& context = timed_context.prepared_active_space;
  const auto& ri_cache =
      xmvb::vb::ensure_cpp_vb_input_ri_cache(input);

  const Clock::time_point orbital_start_time = Clock::now();
  context.n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) /
      2;
  xmvb::vb::ActiveSpaceOrbitalPreparer orbital_preparer;
  context.orbital_result =
      orbital_preparer.prepare(input.orbital_preparation_input);
  timed_context.timings.orbital_preparation_wall_time_seconds =
      seconds_since(orbital_start_time);

  const Clock::time_point ao_h1e_start_time = Clock::now();
  xmvb::vb::AoEffectiveOneElectronBuilder ao_effective_one_electron_builder;
  context.ao_effective_one_electron_result =
      ao_effective_one_electron_builder.build(
          context.orbital_result.inactive_density_matrix,
          input.ao_integral_input.ao_core_hamiltonian_matrix,
          ri_cache,
          input.ao_integral_input.n_basis_functions);
  timed_context.timings.ao_effective_one_electron_wall_time_seconds =
      seconds_since(ao_h1e_start_time);

  const Clock::time_point active_h1e_start_time = Clock::now();
  xmvb::vb::ActiveSpaceOneElectronBuilder active_space_one_electron_builder;
  context.active_space_one_electron_result =
      active_space_one_electron_builder.build(
          context.ao_effective_one_electron_result.ao_effective_h1e,
          context.orbital_result.auxiliary_orbital_matrix,
          input.ao_integral_input.n_basis_functions,
          context.n_inactive_doubly_occupied_orbitals,
          input.orbital_preparation_input.n_active_orbitals);
  timed_context.timings.active_one_electron_wall_time_seconds =
      seconds_since(active_h1e_start_time);

  const Clock::time_point active_2e_start_time = Clock::now();
  xmvb::vb::RiActiveSpaceTwoElectronBuilder ri_builder;
  xmvb::vb::RiActiveSpaceTwoElectronBuilderOptions ri_options;
  ri_options.reconstruct_packed_integrals = true;
  context.active_space_two_electron_result =
      ri_builder.build(
          ri_cache,
          context.orbital_result,
          input.ao_integral_input.n_basis_functions,
          input.orbital_preparation_input.n_active_orbitals,
          ri_options);
  timed_context.timings.active_two_electron_wall_time_seconds =
      seconds_since(active_2e_start_time);

  context.one_electron_reference_energy =
      xmvb::vb::compute_one_electron_reference_energy(
          context.orbital_result.inactive_density_matrix,
          context.ao_effective_one_electron_result.ao_effective_h1e,
          input.ao_integral_input.ao_core_hamiltonian_matrix,
          input.ao_integral_input.n_basis_functions);
  return timed_context;
}

}  // namespace xmvb::pfaffian_vbscf
