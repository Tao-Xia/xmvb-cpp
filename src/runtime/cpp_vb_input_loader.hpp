#pragma once

#include <string>
#include <vector>

#include "runtime/cpp_initial_guess_builder.hpp"
#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/matrices/raw_structure_subspace_selector.hpp"
#include "vb/matrices/structure_types.hpp"
#include "vb/vbscf_algorithm.hpp"
#include "runtime_c/cpp_runtime_extractor.h"

namespace xmvb::vb {

enum class AoIntegralSource {
  Auto,
  LegacyRuntime,
  LibcintMaterializedCpp,
  RuntimeCoreHamiltonianOnly,
};

const char* ao_integral_source_name(AoIntegralSource source);

const char* standard_two_electron_mode_name(StandardTwoElectronMode mode);

struct CppVbStaticMoleculeMetadata {
  int n_atoms = 0;
  int n_shells = 0;
  std::vector<int> atomic_numbers;
  std::vector<double> atomic_coordinates;
  std::vector<int> shell_to_atom;
  std::vector<int> shell_angular_momenta;
  std::vector<int> shell_n_primitives;
  std::vector<int> shell_ao_starts;
  std::vector<int> shell_ao_counts;
  std::vector<int> ao_to_atom;
  std::vector<int> ao_to_shell;
  std::vector<int> ao_angular_momenta;
  std::vector<int> ao_shell_local_indices;
  std::vector<int> ao_cartesian_exponents;
};

struct CppVbInputLoadOptions {
  // `Auto` now prefers the C++ materialized AO integral provider for exact
  // integrals and uses runtime-provided H-core data only when the deck
  // requests the RI path.
  AoIntegralSource ao_integral_source = AoIntegralSource::Auto;
  // Default to the pure C++ orbital guess so standalone builds do not depend
  // on the optional legacy HF/VB guess runtime.
  OrbitalGuessSource orbital_guess_source = OrbitalGuessSource::Cpp;
  StandardTwoElectronMode standard_two_electron_mode = StandardTwoElectronMode::Auto;
  RawStructureSelectionMode raw_structure_selection = RawStructureSelectionMode::Full;
  bool skip_orbital_guess = false;
  bool expand_selected_raw_structures = true;
};

struct CppVbInputLoadResult {
  CppVbInput input;
  RawStructureData raw_structure_data;
  CppVbStaticMoleculeMetadata static_molecule_metadata;
  CppRuntimeExtractionTimings runtime_timings{};
  std::string basis_name;
  double nuclear_repulsion_energy = 0.0;
  AoIntegralSource ao_integral_source = AoIntegralSource::Auto;
  OrbitalGuessSource orbital_guess_source = OrbitalGuessSource::Cpp;
  StandardTwoElectronMode standard_two_electron_mode = StandardTwoElectronMode::Auto;
  RawStructureSelectionMode raw_structure_selection = RawStructureSelectionMode::Full;
  VBSCFAlgorithm requested_algorithm = VBSCFAlgorithm::Original;
  int requested_scf_max_iterations = 2000;
  bool request_molden_output = false;
  int source_raw_structure_count = 0;
  double ao_integral_provider_seconds = 0.0;
  double ao_integral_input_build_seconds = 0.0;
  double orbital_guess_seconds = 0.0;
  double raw_structure_selection_seconds = 0.0;
  double structure_expansion_seconds = 0.0;
  double total_seconds = 0.0;
};

/**
 * @brief Loads the full C++ VB input bundle from the mixed C/C++ runtime.
 *
 * The C runtime prepares the raw molecular, VB-structure, orbital, and AO-integral data.
 * The determinant expansion and subsequent VBSCF numerical kernels remain in C++.
 *
 * @param input_file_path Input deck used to initialize the runtime bundle.
 * @return CppVbInput Fully populated matrix-builder input.
 */
CppVbInput load_cpp_vb_input(
    const std::string& input_file_path,
    const CppVbInputLoadOptions& options = {});

CppVbInputLoadResult load_cpp_vb_input_with_timings(
    const std::string& input_file_path,
    const CppVbInputLoadOptions& options = {});

}  // namespace xmvb::vb
