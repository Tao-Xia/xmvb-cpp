#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CppRuntimeExtractionTimings {
  double read_input_seconds;
  double vb_input_seconds;
  double libcint_buffer_setup_seconds;
  double hf_setup_seconds;
  double vbprep_seconds;
  double vbguess_seconds;
  double one_electron_integrals_seconds;
  double two_electron_integrals_seconds;
  double output_copy_seconds;
  double total_seconds;
} CppRuntimeExtractionTimings;

typedef struct CppRuntimeSnapshot {
  int n_structures;
  int n_atoms;
  int n_shells;
  int n_basis_functions;
  int n_orbitals;
  int n_active_orbitals;
  int n_total_electrons;
  int n_active_electrons;
  int spin_multiplicity;
  int wavefunction_type;
  int vb_function_type;
  int* atomic_numbers;
  double* atomic_coordinates;
  int* shell_to_atom;
  int* shell_angular_momenta;
  int* shell_n_primitives;
  int* shell_ao_starts;
  int* shell_ao_counts;
  int* ao_to_atom;
  int* ao_to_shell;
  int* ao_angular_momenta;
  int* ao_shell_local_indices;
  int* ao_cartesian_exponents;
  int* raw_structure_orbitals;
  double* orbital_value_table;
  int* orbital_basis_index_table;
  int* orbital_basis_counts;
  int* original_orbital_basis_counts;
  double* basis_overlap_matrix;
  double nuclear_repulsion_energy;
  long n_ao_two_electron_integrals;
  double* ao_core_hamiltonian_matrix;
  double* ao_two_electron_integral_values;
  int* ao_two_electron_integral_indices;
} CppRuntimeSnapshot;

void init_cpp_runtime_snapshot(CppRuntimeSnapshot* snapshot);

void free_cpp_runtime_snapshot(CppRuntimeSnapshot* snapshot);

int extract_cpp_runtime_snapshot(
    const char* input_file_path,
    CppRuntimeSnapshot* snapshot,
    CppRuntimeExtractionTimings* timings,
    char* error_message,
    size_t error_message_capacity);

int extract_cpp_runtime_input(
    const char* input_file_path,
    int* n_structures,
    int* n_basis_functions,
    int* n_orbitals,
    int* n_active_orbitals,
    int* n_total_electrons,
    int* n_active_electrons,
    int* spin_multiplicity,
    int* wavefunction_type,
    int* vb_function_type,
    int** raw_structure_orbitals,
    double** orbital_value_table,
    int** orbital_basis_index_table,
    int** orbital_basis_counts,
    int** original_orbital_basis_counts,
    double** basis_overlap_matrix,
    double* nuclear_repulsion_energy,
    long* n_ao_two_electron_integrals,
    double** ao_core_hamiltonian_matrix,
    double** ao_two_electron_integral_values,
    int** ao_two_electron_integral_indices,
    CppRuntimeExtractionTimings* timings,
    char* error_message,
    size_t error_message_capacity);

#ifdef __cplusplus
}
#endif
