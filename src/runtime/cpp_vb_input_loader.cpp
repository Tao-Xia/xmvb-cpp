#include "runtime/cpp_vb_input_loader.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "runtime/cpp_block_guess_builder.hpp"
#include "runtime/libcint_materialized_integral_provider.hpp"
#include "runtime/materialized_ao_integral_input_builder.hpp"
#include "runtime_c/cpp_runtime_extractor.h"
#include "vb/vb.h"
#include "vb/matrices/full_structure_expander.hpp"
#include "vb/matrices/raw_structure_subspace_selector.hpp"
#include "vb/orbital/sparse_orbital_parameter_view.hpp"
#include "vb/orbital/support_aware_mo_gauge_fix.hpp"

namespace xmvb::vb {

namespace {

struct RuntimeSnapshotOwner {
  RuntimeSnapshotOwner() {
    init_cpp_runtime_snapshot(&snapshot);
  }

  ~RuntimeSnapshotOwner() {
    free_cpp_runtime_snapshot(&snapshot);
  }

  CppRuntimeSnapshot snapshot{};
};

struct AoIntegralExtractionPlan {
  bool skip_legacy_two_electron_integrals = true;
  bool copy_legacy_ao_integrals = false;
  bool legacy_two_electron_integrals_available = false;
};

bool load_progress_logging_enabled() {
  const char* value = std::getenv("XMVB_CPP_LOG_LOAD_PROGRESS");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool ao_effective_one_electron_graph_enabled() {
  const char* disable_flag = std::getenv("XMVB_CPP_DISABLE_AO_H1E_GRAPH");
  return disable_flag == nullptr ||
      disable_flag[0] == '\0' ||
      disable_flag[0] == '0';
}

void log_load_stage(
    const char* stage_name,
    double seconds) {
  if (!load_progress_logging_enabled()) {
    return;
  }
  std::fprintf(stderr, "load_stage %s %.12f\n", stage_name, seconds);
  std::fflush(stderr);
}

constexpr int kLibcintBasSlots = 8;
constexpr int kLibcintPtrExpSlot = 5;
constexpr int kLibcintPtrCoeffSlot = 6;

std::string trim_ascii_whitespace(std::string value) {
  const auto first = std::find_if_not(
      value.begin(),
      value.end(),
      [](unsigned char c) { return std::isspace(c) != 0; });
  const auto last = std::find_if_not(
      value.rbegin(),
      value.rend(),
      [](unsigned char c) { return std::isspace(c) != 0; })
                        .base();
  if (first >= last) {
    return {};
  }
  return std::string(first, last);
}

std::string to_ascii_lower(std::string value) {
  std::transform(
      value.begin(),
      value.end(),
      value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::optional<StandardTwoElectronMode> sniff_standard_two_electron_mode_from_input(
    const std::string& input_file_path) {
  std::ifstream input_stream(input_file_path);
  if (!input_stream) {
    return std::nullopt;
  }

  bool inside_ctrl_block = false;
  std::string line;
  while (std::getline(input_stream, line)) {
    const std::size_t comment_pos = line.find('#');
    if (comment_pos != std::string::npos) {
      line.erase(comment_pos);
    }
    std::string normalized_line = to_ascii_lower(trim_ascii_whitespace(std::move(line)));
    if (normalized_line.empty()) {
      continue;
    }
    if (!inside_ctrl_block) {
      if (normalized_line == "$ctrl") {
        inside_ctrl_block = true;
      }
      continue;
    }
    if (normalized_line == "$end") {
      break;
    }
    if (normalized_line.rfind("int", 0) != 0) {
      continue;
    }
    std::size_t position = 3;
    while (position < normalized_line.size() &&
           std::isspace(static_cast<unsigned char>(normalized_line[position])) != 0) {
      ++position;
    }
    if (position >= normalized_line.size() || normalized_line[position] != '=') {
      continue;
    }
    ++position;
    while (position < normalized_line.size() &&
           std::isspace(static_cast<unsigned char>(normalized_line[position])) != 0) {
      ++position;
    }
    const std::string value =
        trim_ascii_whitespace(normalized_line.substr(position));
    if (value == "ri") {
      return StandardTwoElectronMode::ResolutionOfIdentity;
    }
    if (value == "libcint") {
      return StandardTwoElectronMode::Exact;
    }
  }

  return std::nullopt;
}

std::size_t libcint_env_prefix_size(
    int env_size,
    int n_gaussian_primitives) {
  const int prefix_size = env_size - 2 * n_gaussian_primitives;
  if (env_size < 0 || n_gaussian_primitives < 0 || prefix_size < 0) {
    throw std::runtime_error("invalid libcint env dimensions in runtime snapshot");
  }
  return xmvb::to_size(prefix_size);
}

LibcintInput build_combined_auxiliary_libcint_input(
    const CppRuntimeSnapshot& runtime_snapshot,
    const LibcintInput& primary_input) {
  LibcintInput auxiliary_input;
  if (runtime_snapshot.auxiliary_n_shells <= 0) {
    return auxiliary_input;
  }
  if (runtime_snapshot.auxiliary_libcint_bas == nullptr ||
      runtime_snapshot.auxiliary_libcint_basidx == nullptr ||
      runtime_snapshot.auxiliary_libcint_env == nullptr) {
    throw std::runtime_error("runtime snapshot auxiliary libcint buffers are missing");
  }

  const std::size_t primary_prefix_size = libcint_env_prefix_size(
      runtime_snapshot.libcint_env_size,
      runtime_snapshot.n_gaussian_primitives);
  const std::size_t auxiliary_prefix_size = libcint_env_prefix_size(
      runtime_snapshot.auxiliary_libcint_env_size,
      runtime_snapshot.auxiliary_n_gaussian_primitives);
  if (primary_prefix_size != auxiliary_prefix_size) {
    throw std::runtime_error("primary and auxiliary libcint env prefixes have different sizes");
  }
  if (!std::equal(
          primary_input.env.begin(),
          primary_input.env.begin() +
              static_cast<std::ptrdiff_t>(primary_prefix_size),
          runtime_snapshot.auxiliary_libcint_env)) {
    throw std::runtime_error("auxiliary libcint env atom prefix differs from the primary input");
  }

  auxiliary_input.n_atoms = runtime_snapshot.n_atoms;
  auxiliary_input.n_shells = runtime_snapshot.auxiliary_n_shells;
  auxiliary_input.n_gaussian_primitives =
      runtime_snapshot.auxiliary_n_gaussian_primitives;
  auxiliary_input.atm = primary_input.atm;
  auxiliary_input.bas.assign(
      runtime_snapshot.auxiliary_libcint_bas,
      runtime_snapshot.auxiliary_libcint_bas +
          xmvb::to_size(runtime_snapshot.auxiliary_libcint_bas_size));
  auxiliary_input.basidx.assign(
      runtime_snapshot.auxiliary_libcint_basidx,
      runtime_snapshot.auxiliary_libcint_basidx +
          xmvb::to_size(runtime_snapshot.auxiliary_libcint_basidx_size));

  // Libcint 3c2e expects the auxiliary basis to reuse the primary atom table
  // and the full primary env prefix, with only the auxiliary primitive blocks
  // appended afterwards. The runtime snapshot stores a standalone auxiliary
  // env, so rebuild the combined layout and shift the basis pointers to the
  // appended primitive segment.
  std::vector<double> combined_auxiliary_env;
  combined_auxiliary_env.reserve(
      primary_input.env.size() +
      xmvb::to_size(runtime_snapshot.auxiliary_libcint_env_size) -
      auxiliary_prefix_size);
  combined_auxiliary_env.insert(
      combined_auxiliary_env.end(),
      primary_input.env.begin(),
      primary_input.env.end());
  combined_auxiliary_env.insert(
      combined_auxiliary_env.end(),
      runtime_snapshot.auxiliary_libcint_env +
          static_cast<std::ptrdiff_t>(auxiliary_prefix_size),
      runtime_snapshot.auxiliary_libcint_env +
          static_cast<std::ptrdiff_t>(runtime_snapshot.auxiliary_libcint_env_size));
  auxiliary_input.env = std::move(combined_auxiliary_env);

  const int auxiliary_env_pointer_shift =
      runtime_snapshot.libcint_env_size -
      static_cast<int>(auxiliary_prefix_size);
  for (int shell_index = 0; shell_index < auxiliary_input.n_shells; ++shell_index) {
    const std::size_t shell_offset =
        xmvb::to_size(shell_index) * kLibcintBasSlots;
    auxiliary_input.bas[shell_offset + kLibcintPtrExpSlot] +=
        auxiliary_env_pointer_shift;
    auxiliary_input.bas[shell_offset + kLibcintPtrCoeffSlot] +=
        auxiliary_env_pointer_shift;
  }

  return auxiliary_input;
}

}  // namespace

const char* ao_integral_source_name(AoIntegralSource source) {
  switch (source) {
    case AoIntegralSource::Auto:
      return "auto";
    case AoIntegralSource::LegacyRuntime:
      return "legacy";
    case AoIntegralSource::LibcintMaterializedCpp:
      return "libcint_cpp";
    case AoIntegralSource::RuntimeCoreHamiltonianOnly:
      return "runtime_hcore";
  }
  return "unknown";
}

const char* standard_two_electron_mode_name(StandardTwoElectronMode mode) {
  switch (mode) {
    case StandardTwoElectronMode::Auto:
      return "auto";
    case StandardTwoElectronMode::Exact:
      return "exact";
    case StandardTwoElectronMode::ResolutionOfIdentity:
      return "ri";
  }
  return "unknown";
}

namespace {

bool should_use_standard_ri_two_electron_mode(
    const CppRuntimeSnapshot& runtime_snapshot,
    const CppVbInputLoadOptions& options) {
  switch (options.standard_two_electron_mode) {
    case StandardTwoElectronMode::Exact:
      return false;
    case StandardTwoElectronMode::ResolutionOfIdentity:
      return true;
    case StandardTwoElectronMode::Auto:
      break;
  }

  // In auto mode, follow the input deck semantics rather than switching on a
  // size heuristic. `INT=LIBCINT` requests exact AO integrals, while `INT=RI`
  // requests the standard RI path. The legacy parser already handles
  // case-insensitive keywords before populating this runtime flag.
  return runtime_snapshot.input_requests_ri_two_electron_mode != 0;
}

int configured_openmp_thread_count();

AoIntegralSource resolve_ao_integral_source(
    bool use_standard_ri_two_electron_mode,
    bool legacy_two_electron_integrals_available,
    const CppVbInputLoadOptions& options) {
  (void)legacy_two_electron_integrals_available;
  switch (options.ao_integral_source) {
    case AoIntegralSource::Auto:
      return use_standard_ri_two_electron_mode
          ? AoIntegralSource::RuntimeCoreHamiltonianOnly
          : AoIntegralSource::LibcintMaterializedCpp;
    case AoIntegralSource::LegacyRuntime:
    case AoIntegralSource::LibcintMaterializedCpp:
    case AoIntegralSource::RuntimeCoreHamiltonianOnly:
      return options.ao_integral_source;
  }
  throw std::invalid_argument("invalid AO integral source");
}

AoIntegralExtractionPlan plan_ao_integral_extraction(
    const std::string& input_file_path,
    const CppVbInputLoadOptions& options) {
  switch (options.ao_integral_source) {
    case AoIntegralSource::LegacyRuntime:
      return {false, true, true};
    case AoIntegralSource::LibcintMaterializedCpp:
      return {true, false, false};
    case AoIntegralSource::RuntimeCoreHamiltonianOnly:
      return {true, true, false};
    case AoIntegralSource::Auto:
      break;
  }

  if (options.standard_two_electron_mode == StandardTwoElectronMode::ResolutionOfIdentity) {
    return {true, true, false};
  }
  if (options.standard_two_electron_mode == StandardTwoElectronMode::Exact) {
    return {true, false, false};
  }

  const auto sniffed_mode =
      sniff_standard_two_electron_mode_from_input(input_file_path);
  if (sniffed_mode.has_value()) {
    const bool use_standard_ri_two_electron_mode =
        *sniffed_mode == StandardTwoElectronMode::ResolutionOfIdentity;
    return use_standard_ri_two_electron_mode
        ? AoIntegralExtractionPlan{true, true, false}
        : AoIntegralExtractionPlan{true, false, false};
  }

  // When the lightweight sniff cannot classify the deck, keep only the
  // runtime H-core buffers. The exact path will still materialize AO
  // integrals through the C++ provider after the runtime snapshot reveals the
  // requested integral mode.
  return {true, true, false};
}

std::size_t active_pair_count(int n_active_orbitals) {
  if (n_active_orbitals <= 0) {
    return 0;
  }
  return xmvb::to_size(n_active_orbitals) *
      xmvb::to_size(n_active_orbitals + 1) / 2;
}

int configured_openmp_thread_count() {
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  return n_threads;
}

}  // namespace

CppVbInputLoadResult load_cpp_vb_input_with_timings(
    const std::string& input_file_path,
    const CppVbInputLoadOptions& options) {
  const auto total_start_time = std::chrono::steady_clock::now();
  CppVbInputLoadResult load_result;
  CppVbInput result;
  RuntimeSnapshotOwner runtime_snapshot_owner;
  auto& runtime_snapshot = runtime_snapshot_owner.snapshot;
  CppRuntimeExtractionOptions extraction_options{};
  CppRuntimeExtractionTimings runtime_timings{};
  char error_message[1024] = {0};
  const AoIntegralExtractionPlan extraction_plan =
      plan_ao_integral_extraction(input_file_path, options);

  init_cpp_runtime_extraction_options(&extraction_options);
  if (extraction_plan.skip_legacy_two_electron_integrals) {
    extraction_options.skip_legacy_two_electron_integrals = 1;
  }
  extraction_options.copy_legacy_ao_integrals =
      extraction_plan.copy_legacy_ao_integrals ? 1 : 0;
  if (options.skip_orbital_guess ||
      options.orbital_guess_source == OrbitalGuessSource::Cpp) {
    extraction_options.skip_legacy_vbguess = 1;
    extraction_options.skip_legacy_hf_setup = 1;
  }

  const int status = extract_cpp_runtime_snapshot_with_options(
      input_file_path.c_str(),
      &extraction_options,
      &runtime_snapshot,
      &runtime_timings,
      error_message,
      sizeof(error_message));
  if (status != 0) {
    throw std::runtime_error(
        error_message[0] != '\0' ? error_message : "C++ runtime extraction failed");
  }
  log_load_stage("runtime_extraction", runtime_timings.total_seconds);
  const bool use_standard_ri_two_electron_mode =
      should_use_standard_ri_two_electron_mode(runtime_snapshot, options);
  load_result.standard_two_electron_mode =
      use_standard_ri_two_electron_mode
          ? StandardTwoElectronMode::ResolutionOfIdentity
          : StandardTwoElectronMode::Exact;
  load_result.requested_algorithm =
      runtime_snapshot.input_requests_tbvbscf_mode != 0
          ? VBSCFAlgorithm::BiorthogonalExactSelected
          : VBSCFAlgorithm::Original;
  load_result.requested_scf_max_iterations =
      runtime_snapshot.requested_scf_max_iterations > 0
          ? runtime_snapshot.requested_scf_max_iterations
          : 2000;
  load_result.request_molden_output =
      runtime_snapshot.input_requests_molden_output != 0;
  result.standard_two_electron_mode = load_result.standard_two_electron_mode;
  const AoIntegralSource resolved_ao_integral_source =
      resolve_ao_integral_source(
          use_standard_ri_two_electron_mode,
          extraction_plan.legacy_two_electron_integrals_available,
          options);
  load_result.ao_integral_source = resolved_ao_integral_source;

  result.orbital_preparation_input.n_basis_functions = runtime_snapshot.n_basis_functions;
  result.orbital_preparation_input.n_orbitals = runtime_snapshot.n_orbitals;
  result.orbital_preparation_input.n_active_orbitals = runtime_snapshot.n_active_orbitals;
  result.orbital_preparation_input.n_total_electrons = runtime_snapshot.n_total_electrons;
  result.orbital_preparation_input.n_active_electrons = runtime_snapshot.n_active_electrons;
  result.orbital_preparation_input.spin_multiplicity = runtime_snapshot.spin_multiplicity;
  result.orbital_preparation_input.orbital_type = runtime_snapshot.orbital_type;
  result.orbital_preparation_input.orbital_value_table.assign(
      runtime_snapshot.orbital_value_table,
      runtime_snapshot.orbital_value_table +
          xmvb::to_size(runtime_snapshot.n_basis_functions) *
              runtime_snapshot.n_orbitals);
  result.orbital_preparation_input.orbital_basis_index_table.assign(
      runtime_snapshot.orbital_basis_index_table,
      runtime_snapshot.orbital_basis_index_table +
          xmvb::to_size(runtime_snapshot.n_basis_functions) *
              runtime_snapshot.n_orbitals);
  result.orbital_preparation_input.orbital_basis_counts.assign(
      runtime_snapshot.orbital_basis_counts,
      runtime_snapshot.orbital_basis_counts + runtime_snapshot.n_orbitals);
  result.orbital_preparation_input.original_orbital_basis_counts.assign(
      runtime_snapshot.original_orbital_basis_counts,
      runtime_snapshot.original_orbital_basis_counts + runtime_snapshot.n_orbitals);
  result.orbital_preparation_input.n_blocks = runtime_snapshot.n_blocks;
  result.orbital_preparation_input.block_storage_dimension =
      runtime_snapshot.block_storage_dimension;
  result.orbital_preparation_input.block_partial_overlap =
      runtime_snapshot.block_partial_overlap;
  if (runtime_snapshot.n_blocks > 0 && runtime_snapshot.block_storage_dimension > 0) {
    result.orbital_preparation_input.block_members.assign(
        runtime_snapshot.block_members,
        runtime_snapshot.block_members +
            xmvb::to_size(runtime_snapshot.n_blocks) *
                runtime_snapshot.block_storage_dimension);
    result.orbital_preparation_input.block_orbital_counts.assign(
        runtime_snapshot.block_orbital_counts,
        runtime_snapshot.block_orbital_counts + runtime_snapshot.n_blocks);
    result.orbital_preparation_input.block_basis_counts.assign(
        runtime_snapshot.block_basis_counts,
        runtime_snapshot.block_basis_counts + runtime_snapshot.n_blocks);
  }
  result.orbital_preparation_input.ao_normalization.assign(
      runtime_snapshot.ao_normalization,
      runtime_snapshot.ao_normalization + runtime_snapshot.n_basis_functions);
  result.orbital_preparation_input.active_orbital_overlap_matrix.assign(
      runtime_snapshot.active_orbital_overlap_matrix,
      runtime_snapshot.active_orbital_overlap_matrix +
          xmvb::to_size(runtime_snapshot.n_basis_functions) *
              runtime_snapshot.n_basis_functions);
  // Keep the orbital support chart exactly as produced by the legacy runtime
  // reader. In particular, `orbtyp=oeo` remains a full-AO chart for every
  // orbital; the C++ loader must not reinterpret `$ORB/$ACTORB` into a
  // different sparse active-space manifold.
  enforce_strict_sparse_orbital_support(&result.orbital_preparation_input);
  const OrbitalPreparationInput original_orbital_preparation_input =
      result.orbital_preparation_input;
  if (runtime_snapshot.guess_type == GUS_MO) {
    result.orbital_preparation_input.mo_gauge_reference_orbital_basis_counts =
        original_orbital_preparation_input.orbital_basis_counts;
    result.orbital_preparation_input.mo_gauge_reference_orbital_basis_index_table =
        original_orbital_preparation_input.orbital_basis_index_table;
  }
  result.orbital_preparation_input.hf_overlap_matrix.assign(
      runtime_snapshot.hf_overlap_matrix,
      runtime_snapshot.hf_overlap_matrix +
          xmvb::to_size(runtime_snapshot.n_basis_functions) *
              runtime_snapshot.n_basis_functions);
  result.libcint_input.n_atoms = runtime_snapshot.n_atoms;
  result.libcint_input.n_shells = runtime_snapshot.n_shells;
  result.libcint_input.n_gaussian_primitives = runtime_snapshot.n_gaussian_primitives;
  result.libcint_input.atm.assign(
      runtime_snapshot.libcint_atm,
      runtime_snapshot.libcint_atm + xmvb::to_size(runtime_snapshot.libcint_atm_size));
  result.libcint_input.bas.assign(
      runtime_snapshot.libcint_bas,
      runtime_snapshot.libcint_bas + xmvb::to_size(runtime_snapshot.libcint_bas_size));
  result.libcint_input.basidx.assign(
      runtime_snapshot.libcint_basidx,
      runtime_snapshot.libcint_basidx +
          xmvb::to_size(runtime_snapshot.libcint_basidx_size));
  result.libcint_input.env.assign(
      runtime_snapshot.libcint_env,
      runtime_snapshot.libcint_env + xmvb::to_size(runtime_snapshot.libcint_env_size));
  if (runtime_snapshot.auxiliary_n_shells > 0) {
    result.auxiliary_libcint_input = build_combined_auxiliary_libcint_input(
        runtime_snapshot,
        result.libcint_input);
  }

  const auto ao_integral_provider_start_time = std::chrono::steady_clock::now();
  MaterializedAoIntegralBuffers ao_integral_buffers;
  if (resolved_ao_integral_source == AoIntegralSource::LegacyRuntime) {
    ao_integral_buffers = {
        runtime_snapshot.n_basis_functions,
        std::vector<double>(
            runtime_snapshot.ao_core_hamiltonian_matrix,
            runtime_snapshot.ao_core_hamiltonian_matrix +
                xmvb::to_size(runtime_snapshot.n_basis_functions) *
                    runtime_snapshot.n_basis_functions),
        std::vector<double>(
            runtime_snapshot.ao_two_electron_integral_values,
            runtime_snapshot.ao_two_electron_integral_values +
                runtime_snapshot.n_ao_two_electron_integrals),
        std::vector<int>(
            runtime_snapshot.ao_two_electron_integral_indices,
            runtime_snapshot.ao_two_electron_integral_indices +
                xmvb::to_size(runtime_snapshot.n_ao_two_electron_integrals) * 4),
    };
  } else if (resolved_ao_integral_source == AoIntegralSource::LibcintMaterializedCpp) {
    LibcintMaterializedIntegralProvider provider;
    ao_integral_buffers = provider.build(result.libcint_input);
  } else if (resolved_ao_integral_source != AoIntegralSource::RuntimeCoreHamiltonianOnly) {
    throw std::invalid_argument("invalid AO integral source");
  }
  load_result.ao_integral_provider_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - ao_integral_provider_start_time)
          .count();
  log_load_stage("ao_integral_provider", load_result.ao_integral_provider_seconds);
  const auto ao_integral_input_build_start_time = std::chrono::steady_clock::now();
  if (resolved_ao_integral_source == AoIntegralSource::RuntimeCoreHamiltonianOnly) {
    result.ao_integral_input = build_core_hamiltonian_only_ao_integral_input(
        runtime_snapshot.n_basis_functions,
        std::vector<double>(
            runtime_snapshot.ao_core_hamiltonian_matrix,
            runtime_snapshot.ao_core_hamiltonian_matrix +
                xmvb::to_size(runtime_snapshot.n_basis_functions) *
                    runtime_snapshot.n_basis_functions));
  } else {
    MaterializedAoIntegralInputBuildOptions ao_input_build_options;
    ao_input_build_options.build_pair_graph =
        active_pair_count(result.orbital_preparation_input.n_active_orbitals) <= 64 &&
        configured_openmp_thread_count() > 1;
    ao_input_build_options.build_pair_indices =
        !ao_input_build_options.build_pair_graph;
    ao_input_build_options.build_ao_effective_one_electron_graph =
        ao_effective_one_electron_graph_enabled();
    result.ao_integral_input = build_materialized_ao_integral_input(
        std::move(ao_integral_buffers),
        ao_input_build_options);
  }
  load_result.ao_integral_input_build_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - ao_integral_input_build_start_time)
          .count();
  log_load_stage("ao_integral_input_build", load_result.ao_integral_input_build_seconds);

  if (!options.skip_orbital_guess &&
      options.orbital_guess_source == OrbitalGuessSource::Cpp) {
    const auto orbital_guess_start_time = std::chrono::steady_clock::now();
    build_cpp_initial_guess(
        input_file_path,
        runtime_snapshot.guess_type,
        result.libcint_input,
        result.ao_integral_input,
        &result.orbital_preparation_input);
    if (runtime_snapshot.guess_type == GUS_MO) {
      // Keep `GUESS=MO` on the original legacy sparse support chart. Expanding
      // inactive supports here changes the variational manifold and shifts the
      // converged energy away from the legacy `.xmo` reference. The support-
      // aware gauge fix below is therefore only allowed to act when some other
      // upstream path has already changed the sparse chart relative to the
      // recorded legacy layout.
      apply_support_aware_inactive_mo_gauge_fix(
          &result.orbital_preparation_input);
    }
    load_result.orbital_guess_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - orbital_guess_start_time)
            .count();
    log_load_stage("orbital_guess", load_result.orbital_guess_seconds);
  }

  const auto raw_structure_selection_start_time = std::chrono::steady_clock::now();
  RawStructureData raw_structure_data;
  raw_structure_data.n_structures = runtime_snapshot.n_structures;
  raw_structure_data.n_total_electrons = runtime_snapshot.n_total_electrons;
  raw_structure_data.n_active_electrons = runtime_snapshot.n_active_electrons;
  raw_structure_data.spin_multiplicity = runtime_snapshot.spin_multiplicity;
  raw_structure_data.wavefunction_type = runtime_snapshot.wavefunction_type;
  raw_structure_data.vb_function_type = runtime_snapshot.vb_function_type;
  raw_structure_data.raw_structure_orbitals.assign(
      runtime_snapshot.raw_structure_orbitals,
      runtime_snapshot.raw_structure_orbitals +
          xmvb::to_size(runtime_snapshot.n_structures) *
              runtime_snapshot.n_total_electrons);
  const int source_raw_structure_count = raw_structure_data.n_structures;
  const auto selected_raw_structure_indices =
      select_raw_structure_indices(raw_structure_data, options.raw_structure_selection);
  RawStructureData selected_raw_structure_data;
  if (static_cast<int>(selected_raw_structure_indices.size()) ==
      raw_structure_data.n_structures) {
    selected_raw_structure_data = std::move(raw_structure_data);
  } else {
    selected_raw_structure_data = build_raw_structure_subset(
        raw_structure_data,
        selected_raw_structure_indices);
  }
  load_result.raw_structure_selection_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - raw_structure_selection_start_time)
          .count();
  log_load_stage("raw_structure_selection", load_result.raw_structure_selection_seconds);

  CppVbStaticMoleculeMetadata static_molecule_metadata;
  static_molecule_metadata.n_atoms = runtime_snapshot.n_atoms;
  static_molecule_metadata.n_shells = runtime_snapshot.n_shells;
  static_molecule_metadata.atomic_numbers.assign(
      runtime_snapshot.atomic_numbers,
      runtime_snapshot.atomic_numbers + runtime_snapshot.n_atoms);
  static_molecule_metadata.atomic_coordinates.assign(
      runtime_snapshot.atomic_coordinates,
      runtime_snapshot.atomic_coordinates +
          xmvb::to_size(runtime_snapshot.n_atoms) * 3);
  static_molecule_metadata.shell_to_atom.assign(
      runtime_snapshot.shell_to_atom,
      runtime_snapshot.shell_to_atom + runtime_snapshot.n_shells);
  static_molecule_metadata.shell_angular_momenta.assign(
      runtime_snapshot.shell_angular_momenta,
      runtime_snapshot.shell_angular_momenta + runtime_snapshot.n_shells);
  static_molecule_metadata.shell_n_primitives.assign(
      runtime_snapshot.shell_n_primitives,
      runtime_snapshot.shell_n_primitives + runtime_snapshot.n_shells);
  static_molecule_metadata.shell_ao_starts.assign(
      runtime_snapshot.shell_ao_starts,
      runtime_snapshot.shell_ao_starts + runtime_snapshot.n_shells);
  static_molecule_metadata.shell_ao_counts.assign(
      runtime_snapshot.shell_ao_counts,
      runtime_snapshot.shell_ao_counts + runtime_snapshot.n_shells);
  static_molecule_metadata.ao_to_atom.assign(
      runtime_snapshot.ao_to_atom,
      runtime_snapshot.ao_to_atom + runtime_snapshot.n_basis_functions);
  static_molecule_metadata.ao_to_shell.assign(
      runtime_snapshot.ao_to_shell,
      runtime_snapshot.ao_to_shell + runtime_snapshot.n_basis_functions);
  static_molecule_metadata.ao_angular_momenta.assign(
      runtime_snapshot.ao_angular_momenta,
      runtime_snapshot.ao_angular_momenta + runtime_snapshot.n_basis_functions);
  static_molecule_metadata.ao_shell_local_indices.assign(
      runtime_snapshot.ao_shell_local_indices,
      runtime_snapshot.ao_shell_local_indices + runtime_snapshot.n_basis_functions);
  static_molecule_metadata.ao_cartesian_exponents.assign(
      runtime_snapshot.ao_cartesian_exponents,
      runtime_snapshot.ao_cartesian_exponents +
          xmvb::to_size(runtime_snapshot.n_basis_functions) * 3);

  if (options.expand_selected_raw_structures) {
    const auto structure_expansion_start_time = std::chrono::steady_clock::now();
    FullDeterminantStructureExpander expander;
    result.structure_data = expander.expand(selected_raw_structure_data);
    load_result.structure_expansion_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - structure_expansion_start_time)
            .count();
    log_load_stage("structure_expansion", load_result.structure_expansion_seconds);
  }
  load_result.input = std::move(result);
  load_result.raw_structure_data = std::move(selected_raw_structure_data);
  load_result.static_molecule_metadata = std::move(static_molecule_metadata);
  load_result.runtime_timings = runtime_timings;
  load_result.basis_name = runtime_snapshot.basis_name;
  load_result.nuclear_repulsion_energy = runtime_snapshot.nuclear_repulsion_energy;
  load_result.orbital_guess_source = options.orbital_guess_source;
  load_result.raw_structure_selection = options.raw_structure_selection;
  load_result.source_raw_structure_count = source_raw_structure_count;
  load_result.total_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start_time).count();
  log_load_stage("load_total", load_result.total_seconds);
  return load_result;
}

CppVbInput load_cpp_vb_input(
    const std::string& input_file_path,
    const CppVbInputLoadOptions& options) {
  return load_cpp_vb_input_with_timings(input_file_path, options).input;
}

}  // namespace xmvb::vb
