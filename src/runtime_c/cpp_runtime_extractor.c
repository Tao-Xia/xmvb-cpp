#include "runtime_c/cpp_runtime_extractor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#ifndef XMVB_CPP_LEGACY_EXECUTABLE_PATH
#define XMVB_CPP_LEGACY_EXECUTABLE_PATH "build-cpp/src/driver/xmvb.exe"
#endif

#include "runtime_c/local_runtime_api.h"

static void set_error_message(char* error_message, size_t capacity, const char* message) {
  if (error_message == NULL || capacity == 0) {
    return;
  }
  snprintf(error_message, capacity, "%s", message);
}

static double monotonic_seconds(void) {
  struct timespec current_time;
  clock_gettime(CLOCK_MONOTONIC, &current_time);
  return (double)current_time.tv_sec + 1.0e-9 * (double)current_time.tv_nsec;
}

static void remove_extension(const char* input_file_path, char* input_stem, size_t capacity) {
  snprintf(input_stem, capacity, "%s", input_file_path);
  char* dot = strrchr(input_stem, '.');
  if (dot != NULL) {
    *dot = '\0';
  }
}

static const char* get_runtime_temp_directory(void) {
  const char* temp_directory = getenv("TMPDIR");
  if (temp_directory == NULL || temp_directory[0] == '\0') {
    temp_directory = "/tmp";
  }
  return temp_directory;
}

static int build_runtime_output_stem(
    const char* input_file_path,
    char* output_stem,
    size_t output_capacity) {
  const char* input_base_name = strrchr(input_file_path, '/');
  input_base_name = (input_base_name == NULL) ? input_file_path : input_base_name + 1;

  char input_base_stem[PATH_MAX];
  snprintf(input_base_stem, sizeof(input_base_stem), "%s", input_base_name);
  remove_extension(input_base_stem, input_base_stem, sizeof(input_base_stem));
  if (input_base_stem[0] == '\0') {
    snprintf(input_base_stem, sizeof(input_base_stem), "input");
  }

  const int written = snprintf(
      output_stem,
      output_capacity,
      "%s/xmvb-cpp-%ld-%s",
      get_runtime_temp_directory(),
      (long)getpid(),
      input_base_stem);
  return written >= 0 && (size_t)written < output_capacity ? 0 : 1;
}

static void free_runtime_output_buffers(
    int** atomic_numbers,
    double** atomic_coordinates,
    int** shell_to_atom,
    int** shell_angular_momenta,
    int** shell_n_primitives,
    int** shell_ao_starts,
    int** shell_ao_counts,
    int** ao_to_atom,
    int** ao_to_shell,
    int** ao_angular_momenta,
    int** ao_shell_local_indices,
    int** ao_cartesian_exponents,
    int** raw_structure_orbitals,
    double** orbital_value_table,
    int** orbital_basis_index_table,
    int** orbital_basis_counts,
    int** original_orbital_basis_counts,
    double** active_orbital_overlap_matrix,
    double** ao_core_hamiltonian_matrix,
    double** ao_two_electron_integral_values,
    int** ao_two_electron_integral_indices) {
  free(*atomic_numbers);
  free(*atomic_coordinates);
  free(*shell_to_atom);
  free(*shell_angular_momenta);
  free(*shell_n_primitives);
  free(*shell_ao_starts);
  free(*shell_ao_counts);
  free(*ao_to_atom);
  free(*ao_to_shell);
  free(*ao_angular_momenta);
  free(*ao_shell_local_indices);
  free(*ao_cartesian_exponents);
  free(*raw_structure_orbitals);
  free(*orbital_value_table);
  free(*orbital_basis_index_table);
  free(*orbital_basis_counts);
  free(*original_orbital_basis_counts);
  free(*active_orbital_overlap_matrix);
  free(*ao_core_hamiltonian_matrix);
  free(*ao_two_electron_integral_values);
  free(*ao_two_electron_integral_indices);
  *atomic_numbers = NULL;
  *atomic_coordinates = NULL;
  *shell_to_atom = NULL;
  *shell_angular_momenta = NULL;
  *shell_n_primitives = NULL;
  *shell_ao_starts = NULL;
  *shell_ao_counts = NULL;
  *ao_to_atom = NULL;
  *ao_to_shell = NULL;
  *ao_angular_momenta = NULL;
  *ao_shell_local_indices = NULL;
  *ao_cartesian_exponents = NULL;
  *raw_structure_orbitals = NULL;
  *orbital_value_table = NULL;
  *orbital_basis_index_table = NULL;
  *orbital_basis_counts = NULL;
  *original_orbital_basis_counts = NULL;
  *active_orbital_overlap_matrix = NULL;
  *ao_core_hamiltonian_matrix = NULL;
  *ao_two_electron_integral_values = NULL;
  *ao_two_electron_integral_indices = NULL;
}

void init_cpp_runtime_snapshot(CppRuntimeSnapshot* snapshot) {
  if (snapshot == NULL) {
    return;
  }
  memset(snapshot, 0, sizeof(*snapshot));
  snapshot->spin_multiplicity = 1;
}

void free_cpp_runtime_snapshot(CppRuntimeSnapshot* snapshot) {
  if (snapshot == NULL) {
    return;
  }
  free_runtime_output_buffers(
      &snapshot->atomic_numbers,
      &snapshot->atomic_coordinates,
      &snapshot->shell_to_atom,
      &snapshot->shell_angular_momenta,
      &snapshot->shell_n_primitives,
      &snapshot->shell_ao_starts,
      &snapshot->shell_ao_counts,
      &snapshot->ao_to_atom,
      &snapshot->ao_to_shell,
      &snapshot->ao_angular_momenta,
      &snapshot->ao_shell_local_indices,
      &snapshot->ao_cartesian_exponents,
      &snapshot->raw_structure_orbitals,
      &snapshot->orbital_value_table,
      &snapshot->orbital_basis_index_table,
      &snapshot->orbital_basis_counts,
      &snapshot->original_orbital_basis_counts,
      &snapshot->active_orbital_overlap_matrix,
      &snapshot->ao_core_hamiltonian_matrix,
      &snapshot->ao_two_electron_integral_values,
      &snapshot->ao_two_electron_integral_indices);
  snapshot->n_structures = 0;
  snapshot->n_atoms = 0;
  snapshot->n_shells = 0;
  snapshot->n_basis_functions = 0;
  snapshot->n_orbitals = 0;
  snapshot->n_active_orbitals = 0;
  snapshot->n_total_electrons = 0;
  snapshot->n_active_electrons = 0;
  snapshot->spin_multiplicity = 1;
  snapshot->wavefunction_type = 0;
  snapshot->vb_function_type = 0;
  snapshot->nuclear_repulsion_energy = 0.0;
  snapshot->n_ao_two_electron_integrals = 0;
}

int extract_cpp_runtime_snapshot(
    const char* input_file_path,
    CppRuntimeSnapshot* snapshot,
    CppRuntimeExtractionTimings* timings,
    char* error_message,
    size_t error_message_capacity) {
  XmvbCppRuntimeHandle* runtime_handle = NULL;
  int status = 1;
  const double total_start_time = monotonic_seconds();
  double stage_start_time = total_start_time;
  char runtime_output_stem[PATH_MAX] = {0};

  if (snapshot == NULL) {
    set_error_message(error_message, error_message_capacity, "runtime snapshot must not be null");
    return 1;
  }
  init_cpp_runtime_snapshot(snapshot);

  if (timings != NULL) {
    memset(timings, 0, sizeof(*timings));
  }

  if (xmvb_cpp_runtime_create(&runtime_handle, error_message, error_message_capacity) != 0) {
    return 1;
  }

  {
    const char executable_name[] = XMVB_CPP_LEGACY_EXECUTABLE_PATH;
    char input_path_buffer[PATH_MAX];
    snprintf(input_path_buffer, sizeof(input_path_buffer), "%s", input_file_path);
    if (build_runtime_output_stem(
            input_file_path,
            runtime_output_stem,
            sizeof(runtime_output_stem)) != 0) {
      set_error_message(error_message, error_message_capacity, "failed to construct runtime output stem");
      goto cleanup;
    }

    if (xmvb_cpp_runtime_load_input(
            runtime_handle,
            input_path_buffer,
            executable_name,
            error_message,
            error_message_capacity) != 0) {
      goto cleanup;
    }
    if (timings != NULL) {
      timings->read_input_seconds = monotonic_seconds() - stage_start_time;
    }
    stage_start_time = monotonic_seconds();
    if (xmvb_cpp_runtime_initialize_wavefunction(
            runtime_handle,
            runtime_output_stem,
            error_message,
            error_message_capacity) != 0) {
      goto cleanup;
    }
    if (timings != NULL) {
      timings->vb_input_seconds = monotonic_seconds() - stage_start_time;
    }
    stage_start_time = monotonic_seconds();

    if (xmvb_cpp_runtime_prepare_libcint_buffers(
            runtime_handle,
            error_message,
            error_message_capacity) != 0) {
      goto cleanup;
    }
    if (timings != NULL) {
      timings->libcint_buffer_setup_seconds = monotonic_seconds() - stage_start_time;
    }
    stage_start_time = monotonic_seconds();

    if (xmvb_cpp_runtime_run_vbprep(
            runtime_handle,
            error_message,
            error_message_capacity) != 0) {
      goto cleanup;
    }
    if (timings != NULL) {
      timings->vbprep_seconds = monotonic_seconds() - stage_start_time;
    }
    stage_start_time = monotonic_seconds();
    if (xmvb_cpp_runtime_run_one_electron_integrals(
            runtime_handle,
            error_message,
            error_message_capacity) != 0) {
      goto cleanup;
    }
    if (timings != NULL) {
      timings->one_electron_integrals_seconds = monotonic_seconds() - stage_start_time;
    }
    stage_start_time = monotonic_seconds();
    if (xmvb_cpp_runtime_run_two_electron_integrals(
            runtime_handle,
            error_message,
            error_message_capacity) != 0) {
      goto cleanup;
    }
    if (timings != NULL) {
      timings->two_electron_integrals_seconds = monotonic_seconds() - stage_start_time;
    }
    stage_start_time = monotonic_seconds();

    if (xmvb_cpp_runtime_finalize_integral_storage(
            runtime_handle,
            error_message,
            error_message_capacity) != 0) {
      goto cleanup;
    }

    if (xmvb_cpp_runtime_setup_hf(
            runtime_handle,
            error_message,
            error_message_capacity) != 0) {
      goto cleanup;
    }
    if (timings != NULL) {
      timings->hf_setup_seconds = monotonic_seconds() - stage_start_time;
    }
    stage_start_time = monotonic_seconds();

    if (xmvb_cpp_runtime_run_vbguess(
            runtime_handle,
            error_message,
            error_message_capacity) != 0) {
      goto cleanup;
    }
    if (timings != NULL) {
      timings->vbguess_seconds = monotonic_seconds() - stage_start_time;
    }
    stage_start_time = monotonic_seconds();

    if (xmvb_cpp_runtime_copy_snapshot(
            runtime_handle,
            snapshot,
            error_message,
            error_message_capacity) != 0) {
      goto cleanup;
    }
    if (timings != NULL) {
      timings->output_copy_seconds = monotonic_seconds() - stage_start_time;
    }
    status = 0;
  }

cleanup:
  if (timings != NULL) {
    timings->total_seconds = monotonic_seconds() - total_start_time;
  }
  if (status != 0) {
    free_cpp_runtime_snapshot(snapshot);
  }
  xmvb_cpp_runtime_destroy(runtime_handle);
  return status;
}

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
    size_t error_message_capacity) {
  CppRuntimeSnapshot snapshot;
  init_cpp_runtime_snapshot(&snapshot);
  const int status = extract_cpp_runtime_snapshot(
      input_file_path,
      &snapshot,
      timings,
      error_message,
      error_message_capacity);
  if (status != 0) {
    return status;
  }

  *n_structures = snapshot.n_structures;
  *n_basis_functions = snapshot.n_basis_functions;
  *n_orbitals = snapshot.n_orbitals;
  *n_active_orbitals = snapshot.n_active_orbitals;
  *n_total_electrons = snapshot.n_total_electrons;
  *n_active_electrons = snapshot.n_active_electrons;
  *spin_multiplicity = snapshot.spin_multiplicity;
  *wavefunction_type = snapshot.wavefunction_type;
  *vb_function_type = snapshot.vb_function_type;
  *nuclear_repulsion_energy = snapshot.nuclear_repulsion_energy;
  *n_ao_two_electron_integrals = snapshot.n_ao_two_electron_integrals;
  *raw_structure_orbitals = snapshot.raw_structure_orbitals;
  *orbital_value_table = snapshot.orbital_value_table;
  *orbital_basis_index_table = snapshot.orbital_basis_index_table;
  *orbital_basis_counts = snapshot.orbital_basis_counts;
  *original_orbital_basis_counts = snapshot.original_orbital_basis_counts;
  *active_orbital_overlap_matrix = snapshot.active_orbital_overlap_matrix;
  *ao_core_hamiltonian_matrix = snapshot.ao_core_hamiltonian_matrix;
  *ao_two_electron_integral_values = snapshot.ao_two_electron_integral_values;
  *ao_two_electron_integral_indices = snapshot.ao_two_electron_integral_indices;

  snapshot.raw_structure_orbitals = NULL;
  snapshot.orbital_value_table = NULL;
  snapshot.orbital_basis_index_table = NULL;
  snapshot.orbital_basis_counts = NULL;
  snapshot.original_orbital_basis_counts = NULL;
  snapshot.active_orbital_overlap_matrix = NULL;
  snapshot.ao_core_hamiltonian_matrix = NULL;
  snapshot.ao_two_electron_integral_values = NULL;
  snapshot.ao_two_electron_integral_indices = NULL;
  free_cpp_runtime_snapshot(&snapshot);
  return 0;
}
