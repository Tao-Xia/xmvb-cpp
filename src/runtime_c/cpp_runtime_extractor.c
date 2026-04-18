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
#include "runtime_c/local_runtime_api_internal.h"

#if defined(__GNUC__) && !defined(XMVB_CPP_HAVE_LEGACY_HF_RUNTIME)
extern int xmvb_cpp_runtime_setup_hf(
    XmvbCppRuntimeHandle* runtime_handle,
    char* error_message,
    size_t error_message_capacity) __attribute__((weak));
#endif

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

static int should_log_runtime_progress(void) {
  const char* progress_flag = getenv("XMVB_CPP_LOG_RUNTIME_PROGRESS");
  return progress_flag != NULL &&
         progress_flag[0] != '\0' &&
         strcmp(progress_flag, "0") != 0;
}

static void log_runtime_progress(
    const char* stage_name,
    double stage_seconds) {
  if (!should_log_runtime_progress()) {
    return;
  }
  fprintf(stderr, "xmvb_cpp_runtime_stage %s %.6f s\n", stage_name, stage_seconds);
  fflush(stderr);
}

// Only `GUESS=AUTO` depends on the legacy HF object in this standalone
// runtime. `READ/RDCI/MO/UNIT` rebuild their orbital coefficients directly
// from the input deck or one-electron data and can therefore be extracted even
// when the optional legacy HF module was not linked.
static int legacy_vbguess_requires_hf_setup(
    const XmvbCppRuntimeHandle* runtime_handle) {
  if (runtime_handle == NULL || runtime_handle->vb_wavefunction == NULL) {
    return 1;
  }

  switch (runtime_handle->vb_wavefunction->iguess) {
    case GUS_UNIT:
    case GUS_READ:
    case GUS_RDCI:
    case GUS_MO:
      return 0;
    case GUS_AUTO:
    default:
      return 1;
  }
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
    int** libcint_atm,
    int** libcint_bas,
    int** libcint_basidx,
    double** libcint_env,
    int** auxiliary_libcint_bas,
    int** auxiliary_libcint_basidx,
    double** auxiliary_libcint_env,
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
    int** block_members,
    int** block_orbital_counts,
    int** block_basis_counts,
    double** ao_normalization,
    double** active_orbital_overlap_matrix,
    double** hf_overlap_matrix,
    double** hf_density_matrix,
    double** hf_fock_matrix,
    double** ao_core_hamiltonian_matrix,
    double** ao_two_electron_integral_values,
    int** ao_two_electron_integral_indices) {
  free(*atomic_numbers);
  free(*atomic_coordinates);
  free(*libcint_atm);
  free(*libcint_bas);
  free(*libcint_basidx);
  free(*libcint_env);
  free(*auxiliary_libcint_bas);
  free(*auxiliary_libcint_basidx);
  free(*auxiliary_libcint_env);
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
  free(*block_members);
  free(*block_orbital_counts);
  free(*block_basis_counts);
  free(*ao_normalization);
  free(*active_orbital_overlap_matrix);
  free(*hf_overlap_matrix);
  free(*hf_density_matrix);
  free(*hf_fock_matrix);
  free(*ao_core_hamiltonian_matrix);
  free(*ao_two_electron_integral_values);
  free(*ao_two_electron_integral_indices);
  *atomic_numbers = NULL;
  *atomic_coordinates = NULL;
  *libcint_atm = NULL;
  *libcint_bas = NULL;
  *libcint_basidx = NULL;
  *libcint_env = NULL;
  *auxiliary_libcint_bas = NULL;
  *auxiliary_libcint_basidx = NULL;
  *auxiliary_libcint_env = NULL;
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
  *block_members = NULL;
  *block_orbital_counts = NULL;
  *block_basis_counts = NULL;
  *ao_normalization = NULL;
  *active_orbital_overlap_matrix = NULL;
  *hf_overlap_matrix = NULL;
  *hf_density_matrix = NULL;
  *hf_fock_matrix = NULL;
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

void init_cpp_runtime_extraction_options(CppRuntimeExtractionOptions* options) {
  if (options == NULL) {
    return;
  }
  options->skip_legacy_two_electron_integrals = 0;
  options->copy_legacy_ao_integrals = 1;
  options->skip_legacy_vbguess = 0;
  options->skip_legacy_hf_setup = 0;
}

void free_cpp_runtime_snapshot(CppRuntimeSnapshot* snapshot) {
  if (snapshot == NULL) {
    return;
  }
  free_runtime_output_buffers(
      &snapshot->atomic_numbers,
      &snapshot->atomic_coordinates,
      &snapshot->libcint_atm,
      &snapshot->libcint_bas,
      &snapshot->libcint_basidx,
      &snapshot->libcint_env,
      &snapshot->auxiliary_libcint_bas,
      &snapshot->auxiliary_libcint_basidx,
      &snapshot->auxiliary_libcint_env,
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
      &snapshot->block_members,
      &snapshot->block_orbital_counts,
      &snapshot->block_basis_counts,
      &snapshot->ao_normalization,
      &snapshot->active_orbital_overlap_matrix,
      &snapshot->hf_overlap_matrix,
      &snapshot->hf_density_matrix,
      &snapshot->hf_fock_matrix,
      &snapshot->ao_core_hamiltonian_matrix,
      &snapshot->ao_two_electron_integral_values,
      &snapshot->ao_two_electron_integral_indices);
  snapshot->n_structures = 0;
  snapshot->n_atoms = 0;
  snapshot->n_shells = 0;
  snapshot->n_gaussian_primitives = 0;
  snapshot->auxiliary_n_shells = 0;
  snapshot->auxiliary_n_gaussian_primitives = 0;
  snapshot->libcint_atm_size = 0;
  snapshot->libcint_bas_size = 0;
  snapshot->libcint_basidx_size = 0;
  snapshot->libcint_env_size = 0;
  snapshot->auxiliary_libcint_bas_size = 0;
  snapshot->auxiliary_libcint_basidx_size = 0;
  snapshot->auxiliary_libcint_env_size = 0;
  snapshot->n_basis_functions = 0;
  snapshot->n_orbitals = 0;
  snapshot->n_active_orbitals = 0;
  snapshot->n_total_electrons = 0;
  snapshot->n_active_electrons = 0;
  snapshot->n_blocks = 0;
  snapshot->block_storage_dimension = 0;
  snapshot->block_partial_overlap = 0;
  snapshot->spin_multiplicity = 1;
  snapshot->input_requests_tbvbscf_mode = 0;
  snapshot->requested_scf_max_iterations = 0;
  snapshot->guess_type = 0;
  snapshot->orbital_type = 0;
  snapshot->fragment_type = 0;
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
  CppRuntimeExtractionOptions options;
  init_cpp_runtime_extraction_options(&options);
  return extract_cpp_runtime_snapshot_with_options(
      input_file_path,
      &options,
      snapshot,
      timings,
      error_message,
      error_message_capacity);
}

int extract_cpp_runtime_snapshot_with_options(
    const char* input_file_path,
    const CppRuntimeExtractionOptions* options,
    CppRuntimeSnapshot* snapshot,
    CppRuntimeExtractionTimings* timings,
    char* error_message,
    size_t error_message_capacity) {
  XmvbCppRuntimeHandle* runtime_handle = NULL;
  int status = 1;
  const double total_start_time = monotonic_seconds();
  double stage_start_time = total_start_time;
  char runtime_output_stem[PATH_MAX] = {0};
  CppRuntimeExtractionOptions resolved_options;

  if (snapshot == NULL) {
    set_error_message(error_message, error_message_capacity, "runtime snapshot must not be null");
    return 1;
  }
  init_cpp_runtime_snapshot(snapshot);
  init_cpp_runtime_extraction_options(&resolved_options);
  if (options != NULL) {
    resolved_options = *options;
  }

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
      log_runtime_progress("read_input", timings->read_input_seconds);
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
      log_runtime_progress("vb_input", timings->vb_input_seconds);
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
      log_runtime_progress("libcint_buffer_setup", timings->libcint_buffer_setup_seconds);
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
      log_runtime_progress("vbprep", timings->vbprep_seconds);
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
      log_runtime_progress(
          "one_electron_integrals",
          timings->one_electron_integrals_seconds);
    }
    stage_start_time = monotonic_seconds();
    if (!resolved_options.skip_legacy_two_electron_integrals) {
      if (xmvb_cpp_runtime_run_two_electron_integrals(
              runtime_handle,
              error_message,
              error_message_capacity) != 0) {
        goto cleanup;
      }
      if (timings != NULL) {
        timings->two_electron_integrals_seconds = monotonic_seconds() - stage_start_time;
        log_runtime_progress(
            "two_electron_integrals",
            timings->two_electron_integrals_seconds);
      }
    }
    stage_start_time = monotonic_seconds();

    if (!resolved_options.skip_legacy_two_electron_integrals) {
      if (xmvb_cpp_runtime_finalize_integral_storage(
              runtime_handle,
              error_message,
              error_message_capacity) != 0) {
        goto cleanup;
      }
    }

    const int need_legacy_hf_setup =
        !resolved_options.skip_legacy_hf_setup &&
        legacy_vbguess_requires_hf_setup(runtime_handle);
    if (need_legacy_hf_setup) {
      if (xmvb_cpp_runtime_setup_hf == NULL) {
        set_error_message(
            error_message,
            error_message_capacity,
            "legacy HF runtime support was not built; rerun with C++ orbital guess or rebuild with legacy HF enabled");
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
        log_runtime_progress("hf_setup", timings->hf_setup_seconds);
      }
    }
    stage_start_time = monotonic_seconds();

    if (!resolved_options.skip_legacy_vbguess) {
      if (xmvb_cpp_runtime_run_vbguess(
              runtime_handle,
              error_message,
              error_message_capacity) != 0) {
        goto cleanup;
      }
      if (timings != NULL) {
        timings->vbguess_seconds = monotonic_seconds() - stage_start_time;
        log_runtime_progress("vbguess", timings->vbguess_seconds);
      }
    } else {
      if (xmvb_cpp_runtime_clear_orbital_guess(
              runtime_handle,
              error_message,
              error_message_capacity) != 0) {
        goto cleanup;
      }
    }
    stage_start_time = monotonic_seconds();

    if (xmvb_cpp_runtime_copy_snapshot_with_options(
            runtime_handle,
            resolved_options.copy_legacy_ao_integrals,
            snapshot,
            error_message,
            error_message_capacity) != 0) {
      goto cleanup;
    }
    if (timings != NULL) {
      timings->output_copy_seconds = monotonic_seconds() - stage_start_time;
      log_runtime_progress("output_copy", timings->output_copy_seconds);
    }
    status = 0;
  }

cleanup:
  if (timings != NULL) {
    timings->total_seconds = monotonic_seconds() - total_start_time;
    log_runtime_progress("total", timings->total_seconds);
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
