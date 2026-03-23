#include "runtime/cpp_vb_input_loader.hpp"

#include <chrono>
#include <stdexcept>

#include "runtime_c/cpp_runtime_extractor.h"
#include "vb/matrices/full_structure_expander.hpp"

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

}  // namespace

CppVbInputLoadResult load_cpp_vb_input_with_timings(
    const std::string& input_file_path) {
  const auto total_start_time = std::chrono::steady_clock::now();
  CppVbInputLoadResult load_result;
  CppVbInput result;
  RuntimeSnapshotOwner runtime_snapshot_owner;
  auto& runtime_snapshot = runtime_snapshot_owner.snapshot;
  CppRuntimeExtractionTimings runtime_timings{};
  char error_message[1024] = {0};

  const int status = extract_cpp_runtime_snapshot(
      input_file_path.c_str(),
      &runtime_snapshot,
      &runtime_timings,
      error_message,
      sizeof(error_message));
  if (status != 0) {
    throw std::runtime_error(
        error_message[0] != '\0' ? error_message : "C++ runtime extraction failed");
  }

  result.orbital_preparation_input.n_basis_functions = runtime_snapshot.n_basis_functions;
  result.orbital_preparation_input.n_orbitals = runtime_snapshot.n_orbitals;
  result.orbital_preparation_input.n_active_orbitals = runtime_snapshot.n_active_orbitals;
  result.orbital_preparation_input.n_total_electrons = runtime_snapshot.n_total_electrons;
  result.orbital_preparation_input.n_active_electrons = runtime_snapshot.n_active_electrons;
  result.orbital_preparation_input.spin_multiplicity = runtime_snapshot.spin_multiplicity;
  result.orbital_preparation_input.orbital_value_table.assign(
      runtime_snapshot.orbital_value_table,
      runtime_snapshot.orbital_value_table +
          static_cast<std::size_t>(runtime_snapshot.n_basis_functions) *
              runtime_snapshot.n_orbitals);
  result.orbital_preparation_input.orbital_basis_index_table.assign(
      runtime_snapshot.orbital_basis_index_table,
      runtime_snapshot.orbital_basis_index_table +
          static_cast<std::size_t>(runtime_snapshot.n_basis_functions) *
              runtime_snapshot.n_orbitals);
  result.orbital_preparation_input.orbital_basis_counts.assign(
      runtime_snapshot.orbital_basis_counts,
      runtime_snapshot.orbital_basis_counts + runtime_snapshot.n_orbitals);
  result.orbital_preparation_input.original_orbital_basis_counts.assign(
      runtime_snapshot.original_orbital_basis_counts,
      runtime_snapshot.original_orbital_basis_counts + runtime_snapshot.n_orbitals);
  result.orbital_preparation_input.active_orbital_overlap_matrix.assign(
      runtime_snapshot.active_orbital_overlap_matrix,
      runtime_snapshot.active_orbital_overlap_matrix +
          static_cast<std::size_t>(runtime_snapshot.n_basis_functions) *
              runtime_snapshot.n_basis_functions);

  result.ao_integral_input.n_basis_functions = runtime_snapshot.n_basis_functions;
  result.ao_integral_input.ao_core_hamiltonian_matrix.assign(
      runtime_snapshot.ao_core_hamiltonian_matrix,
      runtime_snapshot.ao_core_hamiltonian_matrix +
          static_cast<std::size_t>(runtime_snapshot.n_basis_functions) *
              runtime_snapshot.n_basis_functions);
  result.ao_integral_input.ao_two_electron_integral_values.assign(
      runtime_snapshot.ao_two_electron_integral_values,
      runtime_snapshot.ao_two_electron_integral_values +
          runtime_snapshot.n_ao_two_electron_integrals);
  result.ao_integral_input.ao_two_electron_integral_indices.assign(
      runtime_snapshot.ao_two_electron_integral_indices,
      runtime_snapshot.ao_two_electron_integral_indices +
          static_cast<std::size_t>(runtime_snapshot.n_ao_two_electron_integrals) * 4);

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
          static_cast<std::size_t>(runtime_snapshot.n_structures) *
              runtime_snapshot.n_total_electrons);

  CppVbStaticMoleculeMetadata static_molecule_metadata;
  static_molecule_metadata.n_atoms = runtime_snapshot.n_atoms;
  static_molecule_metadata.n_shells = runtime_snapshot.n_shells;
  static_molecule_metadata.atomic_numbers.assign(
      runtime_snapshot.atomic_numbers,
      runtime_snapshot.atomic_numbers + runtime_snapshot.n_atoms);
  static_molecule_metadata.atomic_coordinates.assign(
      runtime_snapshot.atomic_coordinates,
      runtime_snapshot.atomic_coordinates +
          static_cast<std::size_t>(runtime_snapshot.n_atoms) * 3);
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
          static_cast<std::size_t>(runtime_snapshot.n_basis_functions) * 3);

  const auto structure_expansion_start_time = std::chrono::steady_clock::now();
  FullDeterminantStructureExpander expander;
  result.structure_data = expander.expand(raw_structure_data);
  load_result.input = std::move(result);
  load_result.raw_structure_data = std::move(raw_structure_data);
  load_result.static_molecule_metadata = std::move(static_molecule_metadata);
  load_result.runtime_timings = runtime_timings;
  load_result.nuclear_repulsion_energy = runtime_snapshot.nuclear_repulsion_energy;
  load_result.structure_expansion_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - structure_expansion_start_time)
          .count();
  load_result.total_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start_time).count();
  return load_result;
}

CppVbInput load_cpp_vb_input(
    const std::string& input_file_path) {
  return load_cpp_vb_input_with_timings(input_file_path).input;
}

}  // namespace xmvb::vb
