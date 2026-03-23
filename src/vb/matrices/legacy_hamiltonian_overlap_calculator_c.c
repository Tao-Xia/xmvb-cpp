#include "vb/matrices/legacy_hamiltonian_overlap_calculator_c.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#include "cint.h"
#include "inpout/input.h"
#include "mol/mol.h"
#include "mol/xint.h"
#include "scf/hf.h"
#include "vb/vb.h"

void rdm_vbscf_(vb_info vb_str, int *print_level, char *output);
void orbprep_(vb_info vb_str);
void cal_g11_(vb_info vb_str);
void cal_f11_(vb_info vb_str);
void cal_e11_(double *one_electron_energy, vb_info vb_str);
void xtra_pt0_(vb_info vb_str);
void hamhd_(vb_info vb_str);
void hamov_(vb_info vb_str);
int gradient_rdm_cosx_calc(vb_info vb_str, coulomb_info coulomb, exchange_info exchange);

static void set_error_message(
    char* error_message,
    size_t error_message_capacity,
    const char* message) {
  if (error_message == NULL || error_message_capacity == 0) {
    return;
  }
  snprintf(error_message, error_message_capacity, "%s", message);
}

static void remove_extension(const char* input_file_path, char* input_stem, size_t capacity) {
  snprintf(input_stem, capacity, "%s", input_file_path);
  char* dot = strrchr(input_stem, '.');
  if (dot != NULL) {
    *dot = '\0';
  }
}

int legacy_calculate_hamiltonian_overlap(
    const char* input_file_path,
    int n_threads,
    int* n_structures,
    double* total_energy,
    double* one_electron_energy,
    double** hamiltonian_matrix,
    double** overlap_matrix,
    char* error_message,
    size_t error_message_capacity) {
  if (input_file_path == NULL || input_file_path[0] == '\0') {
    set_error_message(error_message, error_message_capacity, "input_file_path must not be empty");
    return 1;
  }
  if (n_threads <= 0) {
    set_error_message(error_message, error_message_capacity, "n_threads must be positive");
    return 1;
  }

  para_info parallel_info = (para_info)malloc(sizeof(struct ParaInfo));
  if (parallel_info == NULL) {
    set_error_message(error_message, error_message_capacity, "failed to allocate ParaInfo");
    return 1;
  }
  parallel_info->myproc = 0;
  parallel_info->nprocs = 1;
  parallel_info->thread_num = n_threads;
  parallel_info->ncores = n_threads;

  init_xscf_world(n_threads);

  inp_info input_info = NULL;
  mol_info molecule = NULL;
  hf_info hartree_fock = NULL;
  vb_info vb_wavefunction = NULL;
  int status = 1;

  char executable_name[] = "build/install/bin/xmvb.exe";
  char input_path_buffer[4096];
  char input_stem[4096];
  snprintf(input_path_buffer, sizeof(input_path_buffer), "%s", input_file_path);
  remove_extension(input_file_path, input_stem, sizeof(input_stem));

  input_info = readinp(input_path_buffer, executable_name);
  checkkeywords(input_info);

  if (input_info->inttyp != INT_READ) {
    molecule = init_mol(input_info->mol_name, input_info->basis_name, input_info->unit_bohr);
  }

  vb_wavefunction = readinp_vb(molecule, input_info, parallel_info, input_stem);
  vb_wavefunction->hvb = NULL;
  vb_wavefunction->svb = NULL;
  vb_wavefunction->hvb_1e = NULL;
  vb_wavefunction->col = NULL;
  memset(vb_wavefunction->xdat_name, 0, sizeof(vb_wavefunction->xdat_name));
  snprintf(
      vb_wavefunction->xdat_name,
      sizeof(vb_wavefunction->xdat_name),
      "%s.xdat",
      vb_wavefunction->file_name);
  if (input_info->nstr == 0) {
    input_info->nstr = vb_wavefunction->nstr;
  }

  if (input_info->inttyp != INT_READ) {
    vb_wavefunction->ngto = get_ngto(molecule);
    vb_wavefunction->nshell = molecule->bas->nbas;
    vb_wavefunction->env = (double*)malloc(
        (vb_wavefunction->ngto * 2 + molecule->atm->natm * 3 + PTR_ENV_START) * sizeof(double));
    vb_wavefunction->atm = (int*)malloc(ATM_SLOTS * molecule->atm->natm * sizeof(int));
    vb_wavefunction->bas = (int*)malloc(BAS_SLOTS * molecule->bas->nbas * sizeof(int));
    vb_wavefunction->basidx = (int*)malloc(2 * molecule->bas->nbas * sizeof(int));
    int_data_trans(
        molecule,
        vb_wavefunction->atm,
        vb_wavefunction->bas,
        vb_wavefunction->basidx,
        vb_wavefunction->env,
        vb_wavefunction->ngto);
  }

  hartree_fock = init_hf(molecule);
  load_hf_jaux(input_info->aux_name, hartree_fock);
  load_hf_kgrids(input_info->k_grid_file, input_info->k_grid_file_final, hartree_fock);
  set_hf_coulomb(RI_jbuilder, hartree_fock);
  set_hf_exchange(COSX_kbuilder, hartree_fock);
  hartree_fock->open_type = input_info->ihf_type;

  if (vbprep(hartree_fock, input_info, vb_wavefunction, parallel_info, input_info->print_level) ==
      1) {
    set_error_message(error_message, error_message_capacity, "vbprep failed");
    goto cleanup;
  }

  if (vb_wavefunction->inttyp == INT_CINT || vb_wavefunction->inttyp == INT_READ) {
    vb_wavefunction->ri_pros = 0;
  } else if (vb_wavefunction->inttyp >= INT_XINT) {
    vb_wavefunction->ri_pros = 1;
    vb_lowrk_init(vb_wavefunction, hartree_fock, molecule, input_info);
  }

  if (vb_wavefunction->inttyp == INT_READ) {
    readint(vb_wavefunction);
  } else if (vb_wavefunction->inttyp == INT_CINT) {
    cal_1eint(vb_wavefunction);
    if (vb_wavefunction->dir2e != INT2E_DIR) {
      cal_2eint(vb_wavefunction, input_info->print_level);
    }
    if (vb_wavefunction->npoints > 0) {
      cal_pcf_int(vb_wavefunction, input_info->print_level);
    }
  } else if (vb_wavefunction->inttyp >= INT_XINT) {
    vb_wavefunction->ssf = hartree_fock->s_matrix[0];
    vb_wavefunction->hhf = hartree_fock->h_matrix[0][0];
    vb_wavefunction->ekf = hartree_fock->t_matrix[0];
    vb_wavefunction->xxf =
        (double*)malloc(sizeof(double) * vb_wavefunction->nb * vb_wavefunction->nb);
    vb_wavefunction->yyf =
        (double*)malloc(sizeof(double) * vb_wavefunction->nb * vb_wavefunction->nb);
    vb_wavefunction->zzf =
        (double*)malloc(sizeof(double) * vb_wavefunction->nb * vb_wavefunction->nb);
    cal_dip_mat(vb_wavefunction->xxf, vb_wavefunction->yyf, vb_wavefunction->zzf, vb_wavefunction);
  }

  if ((vb_wavefunction->inttyp == INT_CINT || vb_wavefunction->inttyp == INT_READ) &&
      (vb_wavefunction->iscf == RDM_SCF || vb_wavefunction->dovbci > 0)) {
    if (vb_wavefunction->dir2e == INT2E_1D) {
      build_g1d(vb_wavefunction);
    } else if (vb_wavefunction->dir2e == INT2E_NB3) {
      build_nb3idx(vb_wavefunction);
    }
  }
  if (vb_wavefunction->hvb == NULL) {
    vb_wavefunction->hvb =
        (double*)malloc((size_t)vb_wavefunction->nstr * (size_t)vb_wavefunction->nstr * sizeof(double));
  }
  if (vb_wavefunction->svb == NULL) {
    vb_wavefunction->svb =
        (double*)malloc((size_t)vb_wavefunction->nstr * (size_t)vb_wavefunction->nstr * sizeof(double));
  }
  if (vb_wavefunction->hvb_1e == NULL) {
    vb_wavefunction->hvb_1e =
        (double*)malloc((size_t)vb_wavefunction->nstr * (size_t)vb_wavefunction->nstr * sizeof(double));
  }
  if (vb_wavefunction->col == NULL) {
    vb_wavefunction->col =
        (double*)malloc((size_t)vb_wavefunction->nstr * (size_t)vb_wavefunction->nsav * sizeof(double));
  }

  if (vbguess(hartree_fock, input_info, vb_wavefunction, input_info->print_level) != 0) {
    set_error_message(error_message, error_message_capacity, "vbguess failed");
    goto cleanup;
  }

  {
    char output[1] = {'\0'};
    int print_level = input_info->print_level;
    rdm_vbscf_(vb_wavefunction, &print_level, output);
  }

  if (vb_wavefunction->ri_pros == 1 && vb_wavefunction->cosx_pros == 1) {
    gradient_rdm_cosx_calc(vb_wavefunction, hartree_fock->coulomb, hartree_fock->exchange);
  }

  if (vb_wavefunction->cosx_pros != 1) {
    orbprep_(vb_wavefunction);
  }
  cal_g11_(vb_wavefunction);
  cal_f11_(vb_wavefunction);
  cal_e11_(&vb_wavefunction->e11, vb_wavefunction);
  xtra_pt0_(vb_wavefunction);
  hamhd_(vb_wavefunction);
  hamov_(vb_wavefunction);
  {
    double valence_bond_structure_energy = 0.0;
    eigencalc(
        &valence_bond_structure_energy,
        vb_wavefunction->col,
        vb_wavefunction->hvb,
        vb_wavefunction->svb,
        vb_wavefunction->enuc,
        vb_wavefunction->nstr,
        vb_wavefunction);
    vb_wavefunction->energy = vb_wavefunction->e11 + valence_bond_structure_energy;
  }

  if (vb_wavefunction->hvb == NULL || vb_wavefunction->svb == NULL) {
    set_error_message(
        error_message,
        error_message_capacity,
        "VBSCF did not populate hamiltonian_matrix or overlap_matrix");
    goto cleanup;
  }

  {
    const size_t matrix_size = (size_t)vb_wavefunction->nstr * (size_t)vb_wavefunction->nstr;
    *hamiltonian_matrix = (double*)malloc(matrix_size * sizeof(double));
    *overlap_matrix = (double*)malloc(matrix_size * sizeof(double));
    if (*hamiltonian_matrix == NULL || *overlap_matrix == NULL) {
      set_error_message(error_message, error_message_capacity, "failed to allocate result matrices");
      goto cleanup;
    }
    memcpy(*hamiltonian_matrix, vb_wavefunction->hvb, matrix_size * sizeof(double));
    memcpy(*overlap_matrix, vb_wavefunction->svb, matrix_size * sizeof(double));
    *n_structures = vb_wavefunction->nstr;
    *total_energy = vb_wavefunction->energy;
    *one_electron_energy = vb_wavefunction->e11;
  }

  status = 0;

cleanup:
  if (vb_wavefunction != NULL) {
    clean_vb_modules(vb_wavefunction);
    del_vb_str(vb_wavefunction);
  }
  if (hartree_fock != NULL) {
    del_hf(hartree_fock);
  }
  if (molecule != NULL) {
    del_mol(molecule);
  }
  del_xscf_world();
  free(parallel_info);
  return status;
}
