#pragma once

#include <string>
#include <vector>

#include "vbscf/core/vbscf_input.hpp"
#include "vbscf/structures/subspace_selector.hpp"
#include "vbscf/structures/structure_types.hpp"

namespace xmvb::vb {

const char* standard_two_electron_mode_name(StandardTwoElectronMode mode);

enum class RawStructureSource {
  Unknown,
  ExplicitInputBlock,
  GeneratedFromStructureClass,
};

struct VbScfStaticMoleculeMetadata {
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

struct RuntimeExtractionTimings {
  double read_input_seconds = 0.0;
  double vb_input_seconds = 0.0;
  double libcint_buffer_setup_seconds = 0.0;
  double hf_setup_seconds = 0.0;
  double vbprep_seconds = 0.0;
  double vbguess_seconds = 0.0;
  double one_electron_integrals_seconds = 0.0;
  double two_electron_integrals_seconds = 0.0;
  double output_copy_seconds = 0.0;
  double total_seconds = 0.0;
};

struct VbScfInputLoadOptions {
  StandardTwoElectronMode standard_two_electron_mode = StandardTwoElectronMode::Auto;
  RawStructureSelectionMode raw_structure_selection = RawStructureSelectionMode::Full;
  bool skip_orbital_guess = false;
  bool expand_selected_raw_structures = true;
  // The AO-H1E graph is only needed by exact_ctx-style second-order paths.
  // Keep it configurable at load time so plain VBSCF / L-BFGS runs do not pay
  // the upfront graph build cost.
  bool build_ao_effective_one_electron_graph = true;
};

struct VbScfInputLoadResult {
  VbScfInput input;
  RawStructureData raw_structure_data;
  VbScfStaticMoleculeMetadata static_molecule_metadata;
  RuntimeExtractionTimings runtime_timings{};
  std::string basis_name;
  double nuclear_repulsion_energy = 0.0;
  StandardTwoElectronMode standard_two_electron_mode = StandardTwoElectronMode::Auto;
  RawStructureSelectionMode raw_structure_selection = RawStructureSelectionMode::Full;
  RawStructureSource raw_structure_source = RawStructureSource::Unknown;
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
 * @brief Loads the complete VBSCF input bundle.
 *
 * The loader builds basis data, orbital supports, initial guesses, AO
 * integrals, and determinant expansions from the input deck.
 *
 * @param input_file_path Input deck used to initialize the runtime bundle.
 * @return VbScfInput Fully populated matrix-builder input.
 */
VbScfInput load_vbscf_input(
    const std::string& input_file_path,
    const VbScfInputLoadOptions& options = {});

VbScfInputLoadResult load_vbscf_input_with_timings(
    const std::string& input_file_path,
    const VbScfInputLoadOptions& options = {});

}  // namespace xmvb::vb
