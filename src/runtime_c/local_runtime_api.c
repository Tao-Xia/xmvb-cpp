#include "runtime_c/local_runtime_api.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#include "inpout/input.h"
#include "mol/mol.h"
#include "mol/xint.h"
#include "runtime_c/local_runtime/cint_compat.h"
#include "scf/hf.h"
#include "vb/vb.h"

void init_xscf_world(int thread_num);
void del_xscf_world(void);

inp_info xmvb_cpp_readinp(char* inpname, char* exefile);
int xmvb_cpp_checkkeywords(inp_info inp_str);
vb_info xmvb_cpp_readinp_vb(mol_info mol, inp_info inp_str, para_info par_str, char* inpname);
int xmvb_cpp_vbprep(hf_info hf, inp_info inp_str, vb_info vb_str, para_info par_str, int print_level);
int xmvb_cpp_vbguess(hf_info hf, inp_info inp_str, vb_info vb_str, int print_level);
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
hf_info xmvb_cpp_init_hf(const mol_info mol);
void xmvb_cpp_load_hf_jaux(const char* aux_fname, hf_info hf);
void xmvb_cpp_load_hf_kgrids(const char* kgrids_fname, const char* kfgrids_fname, hf_info hf);
void xmvb_cpp_set_hf_coulomb(Jbuilder_type jtype, hf_info hf);
void xmvb_cpp_set_hf_exchange(Kbuilder_type ktype, hf_info hf);
void xmvb_cpp_load_hf_dft(
    double hf_frac,
    GRIDS_TYPE gtype,
    const int dft_id[],
    const double dft_frac[],
    int num,
    int disp_type,
    int dft_name,
    hf_info hf);
void xmvb_cpp_del_hf(hf_info hf);

struct XmvbCppRuntimeHandle {
  para_info parallel_info;
  inp_info input_info;
  mol_info molecule;
  hf_info hf_wavefunction;
  vb_info vb_wavefunction;
  int xscf_world_initialized;
  char runtime_xdat_path[PATH_MAX];
};

static void set_error_message(
    char* error_message,
    size_t error_message_capacity,
    const char* message) {
  if (error_message == NULL || error_message_capacity == 0) {
    return;
  }
  snprintf(error_message, error_message_capacity, "%s", message);
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
    char* error_message,
    size_t error_message_capacity) {
  snapshot->atomic_numbers = (int*)malloc((size_t)snapshot->n_atoms * sizeof(int));
  snapshot->atomic_coordinates =
      (double*)malloc((size_t)snapshot->n_atoms * 3 * sizeof(double));
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
  snapshot->active_orbital_overlap_matrix = (double*)malloc(
      (size_t)snapshot->n_basis_functions * (size_t)snapshot->n_basis_functions * sizeof(double));
  snapshot->ao_core_hamiltonian_matrix = (double*)malloc(
      (size_t)snapshot->n_basis_functions * (size_t)snapshot->n_basis_functions * sizeof(double));
  snapshot->ao_two_electron_integral_values =
      (double*)malloc((size_t)snapshot->n_ao_two_electron_integrals * sizeof(double));
  snapshot->ao_two_electron_integral_indices =
      (int*)malloc((size_t)snapshot->n_ao_two_electron_integrals * 4 * sizeof(int));
  if (snapshot->atomic_numbers == NULL || snapshot->atomic_coordinates == NULL ||
      snapshot->shell_to_atom == NULL || snapshot->shell_angular_momenta == NULL ||
      snapshot->shell_n_primitives == NULL || snapshot->shell_ao_starts == NULL ||
      snapshot->shell_ao_counts == NULL || snapshot->ao_to_atom == NULL ||
      snapshot->ao_to_shell == NULL || snapshot->ao_angular_momenta == NULL ||
      snapshot->ao_shell_local_indices == NULL || snapshot->ao_cartesian_exponents == NULL ||
      snapshot->raw_structure_orbitals == NULL || snapshot->orbital_value_table == NULL ||
      snapshot->orbital_basis_index_table == NULL || snapshot->orbital_basis_counts == NULL ||
      snapshot->original_orbital_basis_counts == NULL || snapshot->active_orbital_overlap_matrix == NULL ||
      snapshot->ao_core_hamiltonian_matrix == NULL ||
      snapshot->ao_two_electron_integral_values == NULL ||
      snapshot->ao_two_electron_integral_indices == NULL) {
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
  init_xscf_world(handle->parallel_info->thread_num);
  handle->xscf_world_initialized = 1;

  *runtime_handle = handle;
  return 0;
}

void xmvb_cpp_runtime_destroy(XmvbCppRuntimeHandle* runtime_handle) {
  if (runtime_handle == NULL) {
    return;
  }
  if (runtime_handle->hf_wavefunction != NULL) {
    xmvb_cpp_del_hf(runtime_handle->hf_wavefunction);
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
  if (runtime_handle->xscf_world_initialized) {
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

  runtime_handle->input_info = xmvb_cpp_readinp(input_file_path, (char*)executable_path);
  xmvb_cpp_checkkeywords(runtime_handle->input_info);
  runtime_handle->input_info->print_level = 0;
  if (runtime_handle->input_info->inttyp != INT_CINT) {
    set_error_message(
        error_message,
        error_message_capacity,
        "standalone xmvb-cpp runtime currently supports only INT=CINT inputs");
    return 1;
  }
  return 0;
}

int xmvb_cpp_runtime_setup_hf(
    XmvbCppRuntimeHandle* runtime_handle,
    char* error_message,
    size_t error_message_capacity) {
  if (runtime_handle == NULL || runtime_handle->input_info == NULL ||
      runtime_handle->molecule == NULL) {
    set_error_message(error_message, error_message_capacity, "runtime state is incomplete for hf setup");
    return 1;
  }

  if (runtime_handle->hf_wavefunction != NULL) {
    return 0;
  }

  runtime_handle->hf_wavefunction = xmvb_cpp_init_hf(runtime_handle->molecule);
  if (runtime_handle->hf_wavefunction == NULL) {
    set_error_message(error_message, error_message_capacity, "failed to initialize hf runtime");
    return 1;
  }

  xmvb_cpp_load_hf_jaux(runtime_handle->input_info->aux_name, runtime_handle->hf_wavefunction);
  xmvb_cpp_load_hf_kgrids(
      runtime_handle->input_info->k_grid_file,
      runtime_handle->input_info->k_grid_file_final,
      runtime_handle->hf_wavefunction);
  xmvb_cpp_set_hf_coulomb(RI_jbuilder, runtime_handle->hf_wavefunction);
  xmvb_cpp_set_hf_exchange(COSX_kbuilder, runtime_handle->hf_wavefunction);

  if (runtime_handle->input_info->ndft > 0) {
    xmvb_cpp_load_hf_dft(
        runtime_handle->input_info->hf_frac,
        runtime_handle->input_info->grid_type,
        runtime_handle->input_info->dft_id,
        runtime_handle->input_info->dft_frac,
        runtime_handle->input_info->ndft,
        runtime_handle->input_info->disp_type,
        runtime_handle->input_info->dft_name,
        runtime_handle->hf_wavefunction);
  }

  runtime_handle->hf_wavefunction->open_type = runtime_handle->input_info->ihf_type;
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

int xmvb_cpp_runtime_copy_snapshot(
    const XmvbCppRuntimeHandle* runtime_handle,
    CppRuntimeSnapshot* snapshot,
    char* error_message,
    size_t error_message_capacity) {
  if (runtime_handle == NULL || runtime_handle->vb_wavefunction == NULL || snapshot == NULL) {
    set_error_message(error_message, error_message_capacity, "runtime snapshot copy received invalid input");
    return 1;
  }

  snapshot->n_structures = runtime_handle->vb_wavefunction->nstr;
  snapshot->n_atoms = runtime_handle->molecule->atm->natm;
  snapshot->n_shells = runtime_handle->molecule->bas->nbas;
  snapshot->n_basis_functions = runtime_handle->vb_wavefunction->nb;
  snapshot->n_orbitals = runtime_handle->vb_wavefunction->nor;
  snapshot->n_active_orbitals = runtime_handle->vb_wavefunction->nao;
  snapshot->n_total_electrons = runtime_handle->vb_wavefunction->nel;
  snapshot->n_active_electrons = runtime_handle->vb_wavefunction->nae;
  snapshot->spin_multiplicity = runtime_handle->vb_wavefunction->nmul;
  snapshot->wavefunction_type = runtime_handle->vb_wavefunction->wfntyp;
  snapshot->vb_function_type = runtime_handle->vb_wavefunction->vbftyp;
  snapshot->nuclear_repulsion_energy = runtime_handle->vb_wavefunction->enuc;
  snapshot->n_ao_two_electron_integrals = runtime_handle->vb_wavefunction->n2e;

  if (allocate_snapshot_buffers(snapshot, error_message, error_message_capacity) != 0) {
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
  memcpy(
      snapshot->active_orbital_overlap_matrix,
      runtime_handle->vb_wavefunction->ssf,
      (size_t)snapshot->n_basis_functions * (size_t)snapshot->n_basis_functions * sizeof(double));
  memcpy(
      snapshot->ao_core_hamiltonian_matrix,
      runtime_handle->vb_wavefunction->hhf,
      (size_t)snapshot->n_basis_functions * (size_t)snapshot->n_basis_functions * sizeof(double));
  memcpy(
      snapshot->ao_two_electron_integral_values,
      runtime_handle->vb_wavefunction->ggf,
      (size_t)snapshot->n_ao_two_electron_integrals * sizeof(double));
  memcpy(
      snapshot->ao_two_electron_integral_indices,
      runtime_handle->vb_wavefunction->g2eidx,
      (size_t)snapshot->n_ao_two_electron_integrals * 4 * sizeof(int));
  return 0;
}
