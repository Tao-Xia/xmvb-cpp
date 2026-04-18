#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define XMVB_CPP_RUNTIME_TEXT_FIELD_CAPACITY 4096

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

typedef struct CppRuntimeExtractionOptions {
  int skip_legacy_two_electron_integrals;
  int copy_legacy_ao_integrals;
  int skip_legacy_vbguess;
  int skip_legacy_hf_setup;
} CppRuntimeExtractionOptions;

typedef struct CppRuntimeSnapshot {
  int n_structures;
  int n_atoms;
  int n_shells;
  int n_gaussian_primitives;
  int auxiliary_n_shells;
  int auxiliary_n_gaussian_primitives;
  int libcint_atm_size;
  int libcint_bas_size;
  int libcint_basidx_size;
  int libcint_env_size;
  int auxiliary_libcint_bas_size;
  int auxiliary_libcint_basidx_size;
  int auxiliary_libcint_env_size;
  int n_basis_functions;
  int n_orbitals;
  int n_active_orbitals;
  int n_total_electrons;
  int n_active_electrons;
  int n_blocks;
  int block_storage_dimension;
  int block_partial_overlap;
  int spin_multiplicity;
  int input_requests_ri_two_electron_mode;
  int input_requests_molden_output;
  int input_requests_tbvbscf_mode;
  int requested_scf_max_iterations;
  int guess_type;
  int orbital_type;
  int fragment_type;
  int wavefunction_type;
  int vb_function_type;
  char basis_name[XMVB_CPP_RUNTIME_TEXT_FIELD_CAPACITY];
  int* atomic_numbers;
  double* atomic_coordinates;
  int* libcint_atm;
  int* libcint_bas;
  int* libcint_basidx;
  double* libcint_env;
  int* auxiliary_libcint_bas;
  int* auxiliary_libcint_basidx;
  double* auxiliary_libcint_env;
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
  int* block_members;
  int* block_orbital_counts;
  int* block_basis_counts;
  double* ao_normalization;
  double* active_orbital_overlap_matrix;
  double* hf_overlap_matrix;
  double* hf_density_matrix;
  double* hf_fock_matrix;
  double nuclear_repulsion_energy;
  long n_ao_two_electron_integrals;
  double* ao_core_hamiltonian_matrix;
  double* ao_two_electron_integral_values;
  int* ao_two_electron_integral_indices;
} CppRuntimeSnapshot;

void init_cpp_runtime_snapshot(CppRuntimeSnapshot* snapshot);

void init_cpp_runtime_extraction_options(CppRuntimeExtractionOptions* options);

void free_cpp_runtime_snapshot(CppRuntimeSnapshot* snapshot);

int extract_cpp_runtime_snapshot(
    const char* input_file_path,
    CppRuntimeSnapshot* snapshot,
    CppRuntimeExtractionTimings* timings,
    char* error_message,
    size_t error_message_capacity);

int extract_cpp_runtime_snapshot_with_options(
    const char* input_file_path,
    const CppRuntimeExtractionOptions* options,
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
    double** active_orbital_overlap_matrix,
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
