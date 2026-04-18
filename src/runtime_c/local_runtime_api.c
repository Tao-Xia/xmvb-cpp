#include "runtime_c/local_runtime_api.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "inpout/input.h"
#include "mol/mol.h"
#include "mol/xint.h"
#include "runtime_c/local_runtime/cint_compat.h"
#include "runtime_c/local_runtime_api_internal.h"
#include "vb/vb.h"

#if defined(__GNUC__) && !defined(XMVB_CPP_HAVE_LEGACY_HF_RUNTIME)
void init_xscf_world(int thread_num) __attribute__((weak));
void del_xscf_world(void) __attribute__((weak));
int xmvb_cpp_vbguess(hf_info hf, inp_info inp_str, vb_info vb_str, int print_level)
    __attribute__((weak));
#else
void init_xscf_world(int thread_num);
void del_xscf_world(void);
int xmvb_cpp_vbguess(hf_info hf, inp_info inp_str, vb_info vb_str, int print_level);
#endif

inp_info xmvb_cpp_readinp(char* inpname, char* exefile);
int xmvb_cpp_checkkeywords(inp_info inp_str);
vb_info xmvb_cpp_readinp_vb(mol_info mol, inp_info inp_str, para_info par_str, char* inpname);
int xmvb_cpp_vbprep(hf_info hf, inp_info inp_str, vb_info vb_str, para_info par_str, int print_level);
int xmvb_cpp_get_ngto(mol_info mol);
void xmvb_cpp_int_data_trans(
    mol_info mol,
    int* atm,
    int* bas,
    int* basidx,
    double* env,
    int ngto);
int xmvb_cpp_cal_1eint(vb_info vb_str);
int xmvb_cpp_cal_2eint(vb_info vb_str, int print_level);

static void set_error_message(
    char* error_message,
    size_t error_message_capacity,
    const char* message) {
  if (error_message == NULL || error_message_capacity == 0) {
    return;
  }
  snprintf(error_message, error_message_capacity, "%s", message);
}

static void restore_runtime_env_var(
    const char* name,
    const char* previous_value) {
  if (previous_value != NULL) {
    setenv(name, previous_value, 1);
  } else {
    unsetenv(name);
  }
}

static void remove_if_present(const char* path) {
  if (path == NULL || path[0] == '\0') {
    return;
  }
  remove(path);
}

static void free_runtime_input_info(inp_info input_info) {
  if (input_info == NULL) {
    return;
  }
  free(input_info->j_grid_file);
  free(input_info->dft_frac);
  free(input_info->dft_id);
  free(input_info->froz_list);
  free(input_info->monomers);
  free(input_info);
}

static int parse_positive_env_int(const char* name) {
  char* end = NULL;
  const char* value = getenv(name);
  long parsed = 0;

  if (value == NULL || value[0] == '\0') {
    return 0;
  }

  errno = 0;
  parsed = strtol(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed <= 0 || parsed > INT_MAX) {
    return 0;
  }

  return (int)parsed;
}

static int resolve_runtime_thread_count(void) {
  int thread_count = parse_positive_env_int("XMVB_CPP_NUM_THREADS");
  if (thread_count > 0) {
    return thread_count;
  }

  thread_count = parse_positive_env_int("OMP_NUM_THREADS");
  if (thread_count > 0) {
    return thread_count;
  }

#ifdef _OPENMP
  thread_count = omp_get_max_threads();
  if (thread_count > 0) {
    return thread_count;
  }
#endif

  return 1;
}

static int allocate_snapshot_buffers(
    CppRuntimeSnapshot* snapshot,
    int copy_legacy_ao_integrals,
    char* error_message,
    size_t error_message_capacity) {
  snapshot->atomic_numbers = (int*)malloc((size_t)snapshot->n_atoms * sizeof(int));
  snapshot->atomic_coordinates =
      (double*)malloc((size_t)snapshot->n_atoms * 3 * sizeof(double));
  snapshot->libcint_atm =
      (int*)malloc((size_t)snapshot->libcint_atm_size * sizeof(int));
  snapshot->libcint_bas =
      (int*)malloc((size_t)snapshot->libcint_bas_size * sizeof(int));
  snapshot->libcint_basidx =
      (int*)malloc((size_t)snapshot->libcint_basidx_size * sizeof(int));
  snapshot->libcint_env =
      (double*)malloc((size_t)snapshot->libcint_env_size * sizeof(double));
  snapshot->auxiliary_libcint_bas = NULL;
  snapshot->auxiliary_libcint_basidx = NULL;
  snapshot->auxiliary_libcint_env = NULL;
  if (snapshot->auxiliary_n_shells > 0) {
    snapshot->auxiliary_libcint_bas =
        (int*)malloc((size_t)snapshot->auxiliary_libcint_bas_size * sizeof(int));
    snapshot->auxiliary_libcint_basidx =
        (int*)malloc((size_t)snapshot->auxiliary_libcint_basidx_size * sizeof(int));
    snapshot->auxiliary_libcint_env =
        (double*)malloc((size_t)snapshot->auxiliary_libcint_env_size * sizeof(double));
  }
  snapshot->shell_to_atom = (int*)malloc((size_t)snapshot->n_shells * sizeof(int));
  snapshot->shell_angular_momenta = (int*)malloc((size_t)snapshot->n_shells * sizeof(int));
  snapshot->shell_n_primitives = (int*)malloc((size_t)snapshot->n_shells * sizeof(int));
  snapshot->shell_ao_starts = (int*)malloc((size_t)snapshot->n_shells * sizeof(int));
  snapshot->shell_ao_counts = (int*)malloc((size_t)snapshot->n_shells * sizeof(int));
  snapshot->ao_to_atom = (int*)malloc((size_t)snapshot->n_basis_functions * sizeof(int));
  snapshot->ao_to_shell = (int*)malloc((size_t)snapshot->n_basis_functions * sizeof(int));
  snapshot->ao_angular_momenta = (int*)malloc((size_t)snapshot->n_basis_functions * sizeof(int));
  snapshot->ao_shell_local_indices =
      (int*)malloc((size_t)snapshot->n_basis_functions * sizeof(int));
  snapshot->ao_cartesian_exponents =
      (int*)malloc((size_t)snapshot->n_basis_functions * 3 * sizeof(int));
  snapshot->raw_structure_orbitals = (int*)malloc(
      (size_t)snapshot->n_structures * (size_t)snapshot->n_total_electrons * sizeof(int));
  snapshot->orbital_value_table = (double*)malloc(
      (size_t)snapshot->n_basis_functions * (size_t)snapshot->n_orbitals * sizeof(double));
  snapshot->orbital_basis_index_table = (int*)malloc(
      (size_t)snapshot->n_basis_functions * (size_t)snapshot->n_orbitals * sizeof(int));
  snapshot->orbital_basis_counts = (int*)malloc((size_t)snapshot->n_orbitals * sizeof(int));
  snapshot->original_orbital_basis_counts = (int*)malloc((size_t)snapshot->n_orbitals * sizeof(int));
  snapshot->block_members = NULL;
  snapshot->block_orbital_counts = NULL;
  snapshot->block_basis_counts = NULL;
  snapshot->ao_normalization = NULL;
  if (snapshot->n_blocks > 0 && snapshot->block_storage_dimension > 0) {
    snapshot->block_members = (int*)malloc(
        (size_t)snapshot->n_blocks * (size_t)snapshot->block_storage_dimension * sizeof(int));
    snapshot->block_orbital_counts = (int*)malloc((size_t)snapshot->n_blocks * sizeof(int));
    snapshot->block_basis_counts = (int*)malloc((size_t)snapshot->n_blocks * sizeof(int));
  }
  if (snapshot->n_basis_functions > 0) {
    snapshot->ao_normalization = (double*)malloc(
        (size_t)snapshot->n_basis_functions * sizeof(double));
  }
  snapshot->active_orbital_overlap_matrix = (double*)malloc(
      (size_t)snapshot->n_basis_functions * (size_t)snapshot->n_basis_functions * sizeof(double));
  snapshot->hf_overlap_matrix = (double*)malloc(
      (size_t)snapshot->n_basis_functions * (size_t)snapshot->n_basis_functions * sizeof(double));
  snapshot->hf_density_matrix = (double*)malloc(
      (size_t)snapshot->n_basis_functions * (size_t)snapshot->n_basis_functions * sizeof(double));
  snapshot->hf_fock_matrix = (double*)malloc(
      (size_t)snapshot->n_basis_functions * (size_t)snapshot->n_basis_functions * sizeof(double));
  snapshot->ao_core_hamiltonian_matrix = NULL;
  snapshot->ao_two_electron_integral_values = NULL;
  snapshot->ao_two_electron_integral_indices = NULL;
  if (copy_legacy_ao_integrals) {
    snapshot->ao_core_hamiltonian_matrix = (double*)malloc(
        (size_t)snapshot->n_basis_functions * (size_t)snapshot->n_basis_functions * sizeof(double));
    if (snapshot->n_ao_two_electron_integrals > 0) {
      snapshot->ao_two_electron_integral_values =
          (double*)malloc((size_t)snapshot->n_ao_two_electron_integrals * sizeof(double));
      snapshot->ao_two_electron_integral_indices =
          (int*)malloc((size_t)snapshot->n_ao_two_electron_integrals * 4 * sizeof(int));
    }
  }
  if (snapshot->atomic_numbers == NULL || snapshot->atomic_coordinates == NULL ||
      snapshot->libcint_atm == NULL || snapshot->libcint_bas == NULL ||
      snapshot->libcint_basidx == NULL || snapshot->libcint_env == NULL ||
      (snapshot->auxiliary_n_shells > 0 &&
       (snapshot->auxiliary_libcint_bas == NULL ||
        snapshot->auxiliary_libcint_basidx == NULL ||
        snapshot->auxiliary_libcint_env == NULL)) ||
      snapshot->shell_to_atom == NULL || snapshot->shell_angular_momenta == NULL ||
      snapshot->shell_n_primitives == NULL || snapshot->shell_ao_starts == NULL ||
      snapshot->shell_ao_counts == NULL || snapshot->ao_to_atom == NULL ||
      snapshot->ao_to_shell == NULL || snapshot->ao_angular_momenta == NULL ||
      snapshot->ao_shell_local_indices == NULL || snapshot->ao_cartesian_exponents == NULL ||
      snapshot->raw_structure_orbitals == NULL || snapshot->orbital_value_table == NULL ||
      snapshot->orbital_basis_index_table == NULL || snapshot->orbital_basis_counts == NULL ||
      snapshot->original_orbital_basis_counts == NULL ||
      (snapshot->n_blocks > 0 && snapshot->block_storage_dimension > 0 &&
       (snapshot->block_members == NULL || snapshot->block_orbital_counts == NULL ||
        snapshot->block_basis_counts == NULL)) ||
      (snapshot->n_basis_functions > 0 && snapshot->ao_normalization == NULL) ||
      snapshot->active_orbital_overlap_matrix == NULL ||
      snapshot->hf_overlap_matrix == NULL ||
      snapshot->hf_density_matrix == NULL ||
      snapshot->hf_fock_matrix == NULL ||
      (copy_legacy_ao_integrals && snapshot->ao_core_hamiltonian_matrix == NULL) ||
      (copy_legacy_ao_integrals && snapshot->n_ao_two_electron_integrals > 0 &&
       (snapshot->ao_two_electron_integral_values == NULL ||
        snapshot->ao_two_electron_integral_indices == NULL))) {
    free_cpp_runtime_snapshot(snapshot);
    set_error_message(
        error_message,
        error_message_capacity,
        "failed to allocate unified runtime buffers");
    return 1;
  }
  return 0;
}

static void fill_cartesian_exponents_for_shell(
    int angular_momentum,
    int* exponents) {
  int component_index = 0;
  for (int lx = angular_momentum; lx >= 0; --lx) {
    for (int ly = angular_momentum - lx; ly >= 0; --ly) {
      const int lz = angular_momentum - lx - ly;
      exponents[component_index * 3 + 0] = lx;
      exponents[component_index * 3 + 1] = ly;
      exponents[component_index * 3 + 2] = lz;
      ++component_index;
    }
  }
}

int xmvb_cpp_runtime_create(
    XmvbCppRuntimeHandle** runtime_handle,
    char* error_message,
    size_t error_message_capacity) {
  XmvbCppRuntimeHandle* handle;

  if (runtime_handle == NULL) {
    set_error_message(error_message, error_message_capacity, "runtime handle output must not be null");
    return 1;
  }

  handle = (XmvbCppRuntimeHandle*)calloc(1, sizeof(*handle));
  if (handle == NULL) {
    set_error_message(error_message, error_message_capacity, "failed to allocate runtime handle");
    return 1;
  }

  handle->parallel_info = (para_info)malloc(sizeof(struct ParaInfo));
  if (handle->parallel_info == NULL) {
    free(handle);
    set_error_message(error_message, error_message_capacity, "failed to allocate ParaInfo");
    return 1;
  }
  handle->parallel_info->myproc = 0;
  handle->parallel_info->nprocs = 1;
  handle->parallel_info->thread_num = resolve_runtime_thread_count();
  handle->parallel_info->ncores = handle->parallel_info->thread_num;
  if (init_xscf_world != NULL) {
    init_xscf_world(handle->parallel_info->thread_num);
    handle->xscf_world_initialized = 1;
  }

  *runtime_handle = handle;
  return 0;
}

void xmvb_cpp_runtime_destroy(XmvbCppRuntimeHandle* runtime_handle) {
  if (runtime_handle == NULL) {
    return;
  }
  if (runtime_handle->hf_wavefunction != NULL &&
      runtime_handle->destroy_hf_fn != NULL) {
    runtime_handle->destroy_hf_fn(runtime_handle->hf_wavefunction);
  }
  if (runtime_handle->vb_wavefunction != NULL) {
    del_vb_str(runtime_handle->vb_wavefunction);
  }
  if (runtime_handle->molecule != NULL) {
    del_mol(runtime_handle->molecule);
  }
  if (runtime_handle->input_info != NULL) {
    remove_if_present(runtime_handle->input_info->inpname);
    remove_if_present(runtime_handle->input_info->mol_name);
    free_runtime_input_info(runtime_handle->input_info);
  }
  remove_if_present(runtime_handle->runtime_xdat_path);
  if (runtime_handle->xscf_world_initialized && del_xscf_world != NULL) {
    del_xscf_world();
  }
  free(runtime_handle->parallel_info);
  free(runtime_handle);
}

int xmvb_cpp_runtime_load_input(
    XmvbCppRuntimeHandle* runtime_handle,
    char* input_file_path,
    const char* executable_path,
    char* error_message,
    size_t error_message_capacity) {
  if (runtime_handle == NULL) {
    set_error_message(error_message, error_message_capacity, "runtime handle must not be null");
    return 1;
  }

  const char* previous_echo_input = getenv("XMVB_CPP_ECHO_INPUT");
  char* previous_echo_input_copy = NULL;
  if (previous_echo_input != NULL) {
    previous_echo_input_copy = strdup(previous_echo_input);
    if (previous_echo_input_copy == NULL) {
      set_error_message(error_message, error_message_capacity, "failed to preserve XMVB_CPP_ECHO_INPUT");
      return 1;
    }
  } else if (setenv("XMVB_CPP_ECHO_INPUT", "0", 1) != 0) {
    set_error_message(error_message, error_message_capacity, "failed to disable input echo");
    return 1;
  }

  runtime_handle->input_info = xmvb_cpp_readinp(input_file_path, (char*)executable_path);
  restore_runtime_env_var("XMVB_CPP_ECHO_INPUT", previous_echo_input_copy);
  free(previous_echo_input_copy);
  xmvb_cpp_checkkeywords(runtime_handle->input_info);
  runtime_handle->input_info->print_level = 0;
  if (runtime_handle->input_info->inttyp != INT_CINT &&
      runtime_handle->input_info->input_requests_ri_two_electron_mode == 0) {
    set_error_message(
        error_message,
        error_message_capacity,
        "standalone xmvb-cpp runtime currently supports only INT=LIBCINT or INT=RI inputs");
    return 1;
  }
  return 0;
}

int xmvb_cpp_runtime_initialize_wavefunction(
    XmvbCppRuntimeHandle* runtime_handle,
    const char* runtime_output_stem,
    char* error_message,
    size_t error_message_capacity) {
  if (runtime_handle == NULL || runtime_handle->input_info == NULL) {
    set_error_message(error_message, error_message_capacity, "runtime input must be loaded first");
    return 1;
  }

  runtime_handle->molecule = init_mol(
      runtime_handle->input_info->mol_name,
      runtime_handle->input_info->basis_name,
      runtime_handle->input_info->unit_bohr);
  runtime_handle->vb_wavefunction = xmvb_cpp_readinp_vb(
      runtime_handle->molecule,
      runtime_handle->input_info,
      runtime_handle->parallel_info,
      (char*)runtime_output_stem);
  runtime_handle->vb_wavefunction->enuc = cal_nuc_rep(runtime_handle->molecule);
  snprintf(
      runtime_handle->runtime_xdat_path,
      sizeof(runtime_handle->runtime_xdat_path),
      "%s.xdat",
      runtime_output_stem);
  return 0;
}

int xmvb_cpp_runtime_prepare_libcint_buffers(
    XmvbCppRuntimeHandle* runtime_handle,
    char* error_message,
    size_t error_message_capacity) {
  if (runtime_handle == NULL || runtime_handle->molecule == NULL ||
      runtime_handle->vb_wavefunction == NULL) {
    set_error_message(
        error_message,
        error_message_capacity,
        "runtime wavefunction must be initialized before libcint setup");
    return 1;
  }

  runtime_handle->vb_wavefunction->ngto = xmvb_cpp_get_ngto(runtime_handle->molecule);
  runtime_handle->vb_wavefunction->nshell = runtime_handle->molecule->bas->nbas;
  runtime_handle->vb_wavefunction->env = (double*)malloc(
      (runtime_handle->vb_wavefunction->ngto * 2 +
       runtime_handle->molecule->atm->natm * 3 + PTR_ENV_START) * sizeof(double));
  runtime_handle->vb_wavefunction->atm =
      (int*)malloc(ATM_SLOTS * runtime_handle->molecule->atm->natm * sizeof(int));
  runtime_handle->vb_wavefunction->bas =
      (int*)malloc(BAS_SLOTS * runtime_handle->molecule->bas->nbas * sizeof(int));
  runtime_handle->vb_wavefunction->basidx =
      (int*)malloc(2 * runtime_handle->molecule->bas->nbas * sizeof(int));
  if (runtime_handle->vb_wavefunction->env == NULL ||
      runtime_handle->vb_wavefunction->atm == NULL ||
      runtime_handle->vb_wavefunction->bas == NULL ||
      runtime_handle->vb_wavefunction->basidx == NULL) {
    set_error_message(error_message, error_message_capacity, "failed to allocate libcint input buffers");
    return 1;
  }

  xmvb_cpp_int_data_trans(
      runtime_handle->molecule,
      runtime_handle->vb_wavefunction->atm,
      runtime_handle->vb_wavefunction->bas,
      runtime_handle->vb_wavefunction->basidx,
      runtime_handle->vb_wavefunction->env,
      runtime_handle->vb_wavefunction->ngto);
  return 0;
}

int xmvb_cpp_runtime_run_vbprep(
    XmvbCppRuntimeHandle* runtime_handle,
    char* error_message,
    size_t error_message_capacity) {
  if (runtime_handle == NULL || runtime_handle->input_info == NULL ||
      runtime_handle->vb_wavefunction == NULL) {
    set_error_message(error_message, error_message_capacity, "runtime state is incomplete for vbprep");
    return 1;
  }

  if (xmvb_cpp_vbprep(
          NULL,
          runtime_handle->input_info,
          runtime_handle->vb_wavefunction,
          runtime_handle->parallel_info,
          runtime_handle->input_info->print_level) != 0) {
    set_error_message(error_message, error_message_capacity, "vbprep failed");
    return 1;
  }
  return 0;
}

int xmvb_cpp_runtime_run_one_electron_integrals(
    XmvbCppRuntimeHandle* runtime_handle,
    char* error_message,
    size_t error_message_capacity) {
  if (runtime_handle == NULL || runtime_handle->vb_wavefunction == NULL) {
    set_error_message(error_message, error_message_capacity, "runtime state is incomplete for cal_1eint");
    return 1;
  }

  if (xmvb_cpp_cal_1eint(runtime_handle->vb_wavefunction) != 0) {
    set_error_message(error_message, error_message_capacity, "cal_1eint failed");
    return 1;
  }
  return 0;
}

int xmvb_cpp_runtime_run_two_electron_integrals(
    XmvbCppRuntimeHandle* runtime_handle,
    char* error_message,
    size_t error_message_capacity) {
  if (runtime_handle == NULL || runtime_handle->input_info == NULL ||
      runtime_handle->vb_wavefunction == NULL) {
    set_error_message(error_message, error_message_capacity, "runtime state is incomplete for cal_2eint");
    return 1;
  }

  if (xmvb_cpp_cal_2eint(
          runtime_handle->vb_wavefunction,
          runtime_handle->input_info->print_level) != 0) {
    set_error_message(error_message, error_message_capacity, "cal_2eint failed");
    return 1;
  }
  return 0;
}

int xmvb_cpp_runtime_finalize_integral_storage(
    XmvbCppRuntimeHandle* runtime_handle,
    char* error_message,
    size_t error_message_capacity) {
  if (runtime_handle == NULL || runtime_handle->vb_wavefunction == NULL) {
    set_error_message(
        error_message,
        error_message_capacity,
        "runtime state is incomplete for integral finalization");
    return 1;
  }

  if ((runtime_handle->vb_wavefunction->inttyp == INT_CINT ||
       runtime_handle->vb_wavefunction->inttyp == INT_READ) &&
      (runtime_handle->vb_wavefunction->iscf == RDM_SCF ||
       runtime_handle->vb_wavefunction->dovbci > 0)) {
    if (runtime_handle->vb_wavefunction->dir2e == INT2E_1D) {
      build_g1d(runtime_handle->vb_wavefunction);
    } else if (runtime_handle->vb_wavefunction->dir2e == INT2E_NB3) {
      build_nb3idx(runtime_handle->vb_wavefunction);
    }
  }

  runtime_handle->vb_wavefunction->ri_pros = 0;
  return 0;
}

int xmvb_cpp_runtime_run_vbguess(
    XmvbCppRuntimeHandle* runtime_handle,
    char* error_message,
    size_t error_message_capacity) {
  if (runtime_handle == NULL || runtime_handle->input_info == NULL ||
      runtime_handle->vb_wavefunction == NULL) {
    set_error_message(error_message, error_message_capacity, "runtime state is incomplete for vbguess");
    return 1;
  }
  if (xmvb_cpp_vbguess == NULL) {
    set_error_message(
        error_message,
        error_message_capacity,
        "legacy VB guess support was not built; rerun with C++ orbital guess or rebuild with legacy HF enabled");
    return 1;
  }

  if (xmvb_cpp_vbguess(
          runtime_handle->hf_wavefunction,
          runtime_handle->input_info,
          runtime_handle->vb_wavefunction,
          runtime_handle->input_info->print_level) != 0) {
    set_error_message(error_message, error_message_capacity, "vbguess failed");
    return 1;
  }
  return 0;
}

int xmvb_cpp_runtime_clear_orbital_guess(
    XmvbCppRuntimeHandle* runtime_handle,
    char* error_message,
    size_t error_message_capacity) {
  if (runtime_handle == NULL || runtime_handle->vb_wavefunction == NULL ||
      runtime_handle->vb_wavefunction->dv == NULL) {
    set_error_message(
        error_message,
        error_message_capacity,
        "runtime state is incomplete for clearing orbital guess");
    return 1;
  }

  memset(
      runtime_handle->vb_wavefunction->dv,
      0,
      (size_t)runtime_handle->vb_wavefunction->nb *
          (size_t)runtime_handle->vb_wavefunction->nor * sizeof(double));
  return 0;
}

int xmvb_cpp_runtime_copy_snapshot(
    const XmvbCppRuntimeHandle* runtime_handle,
    CppRuntimeSnapshot* snapshot,
    char* error_message,
    size_t error_message_capacity) {
  return xmvb_cpp_runtime_copy_snapshot_with_options(
      runtime_handle,
      1,
      snapshot,
      error_message,
      error_message_capacity);
}

int xmvb_cpp_runtime_copy_snapshot_with_options(
    const XmvbCppRuntimeHandle* runtime_handle,
    int copy_legacy_ao_integrals,
    CppRuntimeSnapshot* snapshot,
    char* error_message,
    size_t error_message_capacity) {
  if (runtime_handle == NULL || runtime_handle->vb_wavefunction == NULL || snapshot == NULL) {
    set_error_message(error_message, error_message_capacity, "runtime snapshot copy received invalid input");
    return 1;
  }

  bas_info auxiliary_basis = NULL;
  struct MolInfo auxiliary_molecule;
  memset(&auxiliary_molecule, 0, sizeof(auxiliary_molecule));

  snapshot->n_structures = runtime_handle->vb_wavefunction->nstr;
  snapshot->n_atoms = runtime_handle->molecule->atm->natm;
  snapshot->n_shells = runtime_handle->molecule->bas->nbas;
  snapshot->n_gaussian_primitives = runtime_handle->vb_wavefunction->ngto;
  snapshot->auxiliary_n_shells = 0;
  snapshot->auxiliary_n_gaussian_primitives = 0;
  snapshot->libcint_atm_size = ATM_SLOTS * snapshot->n_atoms;
  snapshot->libcint_bas_size = BAS_SLOTS * snapshot->n_shells;
  snapshot->libcint_basidx_size = snapshot->n_shells * 2;
  snapshot->libcint_env_size =
      snapshot->n_gaussian_primitives * 2 + snapshot->n_atoms * 3 + PTR_ENV_START;
  snapshot->auxiliary_libcint_bas_size = 0;
  snapshot->auxiliary_libcint_basidx_size = 0;
  snapshot->auxiliary_libcint_env_size = 0;
  snapshot->n_basis_functions = runtime_handle->vb_wavefunction->nb;
  snapshot->n_orbitals = runtime_handle->vb_wavefunction->nor;
  snapshot->n_active_orbitals = runtime_handle->vb_wavefunction->nao;
  snapshot->n_total_electrons = runtime_handle->vb_wavefunction->nel;
  snapshot->n_active_electrons = runtime_handle->vb_wavefunction->nae;
  snapshot->n_blocks = runtime_handle->vb_wavefunction->nblock;
  snapshot->block_storage_dimension =
      runtime_handle->vb_wavefunction->dovbci > 0
          ? runtime_handle->vb_wavefunction->nb
          : runtime_handle->vb_wavefunction->nor;
  snapshot->block_partial_overlap = runtime_handle->vb_wavefunction->block_part_ov;
  snapshot->spin_multiplicity = runtime_handle->vb_wavefunction->nmul;
  snapshot->input_requests_ri_two_electron_mode =
      runtime_handle->input_info != NULL
          ? runtime_handle->input_info->input_requests_ri_two_electron_mode
          : 0;
  snapshot->input_requests_molden_output =
      runtime_handle->input_info != NULL
          ? runtime_handle->input_info->molden
          : 0;
  snapshot->input_requests_tbvbscf_mode =
      runtime_handle->input_info != NULL
          ? runtime_handle->input_info->biovb
          : 0;
  snapshot->requested_scf_max_iterations = runtime_handle->vb_wavefunction->itmax;
  snapshot->guess_type = runtime_handle->vb_wavefunction->iguess;
  snapshot->orbital_type = runtime_handle->vb_wavefunction->orbtyp;
  snapshot->fragment_type = runtime_handle->vb_wavefunction->frgtyp;
  snapshot->wavefunction_type = runtime_handle->vb_wavefunction->wfntyp;
  snapshot->vb_function_type = runtime_handle->vb_wavefunction->vbftyp;
  snprintf(
      snapshot->basis_name,
      sizeof(snapshot->basis_name),
      "%s",
      runtime_handle->input_info != NULL ? runtime_handle->input_info->basis_name : "");
  snapshot->nuclear_repulsion_energy = runtime_handle->vb_wavefunction->enuc;
  snapshot->n_ao_two_electron_integrals =
      copy_legacy_ao_integrals ? runtime_handle->vb_wavefunction->n2e : 0;

  if (runtime_handle->input_info != NULL && runtime_handle->input_info->aux_name[0] != '\0') {
    auxiliary_basis = init_bas(
        runtime_handle->input_info->aux_name,
        runtime_handle->molecule->atm);
    if (auxiliary_basis == NULL) {
      set_error_message(error_message, error_message_capacity, "failed to initialize RI auxiliary basis");
      return 1;
    }
    auxiliary_molecule.atm = runtime_handle->molecule->atm;
    auxiliary_molecule.bas = auxiliary_basis;
    snapshot->auxiliary_n_shells = auxiliary_basis->nbas;
    snapshot->auxiliary_n_gaussian_primitives =
        xmvb_cpp_get_ngto(&auxiliary_molecule);
    snapshot->auxiliary_libcint_bas_size =
        BAS_SLOTS * snapshot->auxiliary_n_shells;
    snapshot->auxiliary_libcint_basidx_size =
        snapshot->auxiliary_n_shells * 2;
    snapshot->auxiliary_libcint_env_size =
        snapshot->auxiliary_n_gaussian_primitives * 2 +
        snapshot->n_atoms * 3 + PTR_ENV_START;
  }

  if (allocate_snapshot_buffers(
          snapshot,
          copy_legacy_ao_integrals,
          error_message,
          error_message_capacity) != 0) {
    if (auxiliary_basis != NULL) {
      del_bas(auxiliary_basis);
    }
    return 1;
  }

  for (int atom_index = 0; atom_index < snapshot->n_atoms; ++atom_index) {
    const int coordinate_offset =
        runtime_handle->molecule->atm->atm[CENTER_IND + atom_index * 2];
    snapshot->atomic_numbers[atom_index] =
        runtime_handle->molecule->atm->atm[ELEMENT_VAL + atom_index * 2];
    memcpy(
        snapshot->atomic_coordinates + atom_index * 3,
        runtime_handle->molecule->atm->value + coordinate_offset,
        3 * sizeof(double));
  }

  memcpy(
      snapshot->libcint_atm,
      runtime_handle->vb_wavefunction->atm,
      (size_t)snapshot->libcint_atm_size * sizeof(int));
  memcpy(
      snapshot->libcint_bas,
      runtime_handle->vb_wavefunction->bas,
      (size_t)snapshot->libcint_bas_size * sizeof(int));
  memcpy(
      snapshot->libcint_basidx,
      runtime_handle->vb_wavefunction->basidx,
      (size_t)snapshot->libcint_basidx_size * sizeof(int));
  memcpy(
      snapshot->libcint_env,
      runtime_handle->vb_wavefunction->env,
      (size_t)snapshot->libcint_env_size * sizeof(double));
  if (auxiliary_basis != NULL) {
    xmvb_cpp_int_data_trans(
        &auxiliary_molecule,
        snapshot->libcint_atm,
        snapshot->auxiliary_libcint_bas,
        snapshot->auxiliary_libcint_basidx,
        snapshot->auxiliary_libcint_env,
        snapshot->auxiliary_n_gaussian_primitives);
    del_bas(auxiliary_basis);
    auxiliary_basis = NULL;
  }

  for (int shell_index = 0; shell_index < snapshot->n_shells; ++shell_index) {
    const int shell_atom =
        runtime_handle->molecule->bas->bas[ATOM_IND + shell_index * 5];
    const int angular_momentum =
        runtime_handle->molecule->bas->bas[ANGULAR_VAL + shell_index * 5];
    const int primitive_count =
        runtime_handle->molecule->bas->bas[NPRIM_VAL + shell_index * 5];
    const int shell_ao_start = runtime_handle->molecule->bas->shls_p[shell_index];
    const int shell_ao_count = xint_gtolen(angular_momentum);

    snapshot->shell_to_atom[shell_index] = shell_atom;
    snapshot->shell_angular_momenta[shell_index] = angular_momentum;
    snapshot->shell_n_primitives[shell_index] = primitive_count;
    snapshot->shell_ao_starts[shell_index] = shell_ao_start;
    snapshot->shell_ao_counts[shell_index] = shell_ao_count;

    for (int local_ao_index = 0; local_ao_index < shell_ao_count; ++local_ao_index) {
      const int ao_index = shell_ao_start + local_ao_index;
      snapshot->ao_to_atom[ao_index] = shell_atom;
      snapshot->ao_to_shell[ao_index] = shell_index;
      snapshot->ao_angular_momenta[ao_index] = angular_momentum;
      snapshot->ao_shell_local_indices[ao_index] = local_ao_index;
    }

    fill_cartesian_exponents_for_shell(
        angular_momentum,
        snapshot->ao_cartesian_exponents + shell_ao_start * 3);
  }

  memcpy(
      snapshot->raw_structure_orbitals,
      runtime_handle->vb_wavefunction->ntstr,
      (size_t)snapshot->n_structures * (size_t)snapshot->n_total_electrons * sizeof(int));
  memcpy(
      snapshot->orbital_value_table,
      runtime_handle->vb_wavefunction->dv,
      (size_t)snapshot->n_basis_functions * (size_t)snapshot->n_orbitals * sizeof(double));
  memcpy(
      snapshot->orbital_basis_index_table,
      runtime_handle->vb_wavefunction->nv,
      (size_t)snapshot->n_basis_functions * (size_t)snapshot->n_orbitals * sizeof(int));
  memcpy(
      snapshot->orbital_basis_counts,
      runtime_handle->vb_wavefunction->ma,
      (size_t)snapshot->n_orbitals * sizeof(int));
  memcpy(
      snapshot->original_orbital_basis_counts,
      runtime_handle->vb_wavefunction->ma0,
      (size_t)snapshot->n_orbitals * sizeof(int));
  if (snapshot->n_blocks > 0 && snapshot->block_storage_dimension > 0) {
    memcpy(
        snapshot->block_members,
        runtime_handle->vb_wavefunction->blocks,
        (size_t)snapshot->n_blocks * (size_t)snapshot->block_storage_dimension * sizeof(int));
    memcpy(
        snapshot->block_orbital_counts,
        runtime_handle->vb_wavefunction->noc_block,
        (size_t)snapshot->n_blocks * sizeof(int));
    memcpy(
        snapshot->block_basis_counts,
        runtime_handle->vb_wavefunction->mx_block,
        (size_t)snapshot->n_blocks * sizeof(int));
  }
  memcpy(
      snapshot->ao_normalization,
      runtime_handle->vb_wavefunction->snorm,
      (size_t)snapshot->n_basis_functions * sizeof(double));
  memcpy(
      snapshot->active_orbital_overlap_matrix,
      runtime_handle->vb_wavefunction->ssf,
      (size_t)snapshot->n_basis_functions * (size_t)snapshot->n_basis_functions * sizeof(double));
  memset(
      snapshot->hf_overlap_matrix,
      0,
      (size_t)snapshot->n_basis_functions * (size_t)snapshot->n_basis_functions * sizeof(double));
  memset(
      snapshot->hf_density_matrix,
      0,
      (size_t)snapshot->n_basis_functions * (size_t)snapshot->n_basis_functions * sizeof(double));
  memset(
      snapshot->hf_fock_matrix,
      0,
      (size_t)snapshot->n_basis_functions * (size_t)snapshot->n_basis_functions * sizeof(double));
  if (runtime_handle->hf_wavefunction != NULL) {
    memcpy(
        snapshot->hf_overlap_matrix,
        runtime_handle->hf_wavefunction->s_matrix[0],
        (size_t)snapshot->n_basis_functions * (size_t)snapshot->n_basis_functions * sizeof(double));
    memcpy(
        snapshot->hf_density_matrix,
        runtime_handle->hf_wavefunction->d_matrix[0][0],
        (size_t)snapshot->n_basis_functions * (size_t)snapshot->n_basis_functions * sizeof(double));
    memcpy(
        snapshot->hf_fock_matrix,
        runtime_handle->hf_wavefunction->f_matrix[0][0],
        (size_t)snapshot->n_basis_functions * (size_t)snapshot->n_basis_functions * sizeof(double));
  } else {
    memcpy(
        snapshot->hf_overlap_matrix,
        snapshot->active_orbital_overlap_matrix,
        (size_t)snapshot->n_basis_functions * (size_t)snapshot->n_basis_functions * sizeof(double));
  }
  if (copy_legacy_ao_integrals) {
    memcpy(
        snapshot->ao_core_hamiltonian_matrix,
        runtime_handle->vb_wavefunction->hhf,
        (size_t)snapshot->n_basis_functions * (size_t)snapshot->n_basis_functions * sizeof(double));
    if (snapshot->n_ao_two_electron_integrals > 0) {
      memcpy(
          snapshot->ao_two_electron_integral_values,
          runtime_handle->vb_wavefunction->ggf,
          (size_t)snapshot->n_ao_two_electron_integrals * sizeof(double));
      memcpy(
          snapshot->ao_two_electron_integral_indices,
          runtime_handle->vb_wavefunction->g2eidx,
          (size_t)snapshot->n_ao_two_electron_integrals * 4 * sizeof(int));
    }
  }
  return 0;
}
