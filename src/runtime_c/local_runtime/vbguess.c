#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inpout/input.h"
#include "inpout/output_func.h"
#include "lapacke.h"
#include "scf/hf.h"
#include "vb/vb.h"

int xmvb_cpp_vb_autoguess(hf_info hf, vb_info vb_str);
int xmvb_cpp_vb_unitguess(vb_info vb_str);
int xmvb_cpp_vb_readguess(inp_info inp_str, vb_info vb_str);

static int get_orbital_basis_count(const vb_info vb_str, int orbital_index) {
  if (vb_str->ma[orbital_index] != 1) {
    return vb_str->ma[orbital_index];
  }

  int count = 0;
  for (int basis_index = 0; basis_index < vb_str->nb; basis_index++) {
    if (vb_str->nv[orbital_index * vb_str->nb + basis_index] == 0) {
      break;
    }
    count++;
  }
  return count;
}

static void scale_guess_back_to_original_basis(vb_info vb_str) {
  for (int orbital_index = 0; orbital_index < vb_str->nor; orbital_index++) {
    for (int coefficient_index = 0; coefficient_index < vb_str->nb; coefficient_index++) {
      const int basis_index =
          vb_str->nv[orbital_index * vb_str->nb + coefficient_index] - 1;
      if (basis_index < 0) {
        break;
      }
      vb_str->dv[orbital_index * vb_str->nb + coefficient_index] *=
          vb_str->snorm[basis_index];
    }
  }
}

static int build_hcore_block_guess(vb_info vb_str) {
  const int nb = vb_str->nb;
  const int nor = vb_str->nor;
  const int nd = (vb_str->dovbci > 0) ? nb : nor;
  double* overlap_block = (double*)malloc(sizeof(double) * nb * nb);
  double* hcore_block = (double*)malloc(sizeof(double) * nb * nb);
  double* eigenvalues = (double*)malloc(sizeof(double) * nb);

  if (overlap_block == NULL || hcore_block == NULL || eigenvalues == NULL) {
    free(overlap_block);
    free(hcore_block);
    free(eigenvalues);
    return 1;
  }

  memset(vb_str->dv, 0, sizeof(double) * nb * nor);

  for (int block_index = 0; block_index < vb_str->nblock; block_index++) {
    const int representative_orbital = vb_str->blocks[block_index * nd];
    const int block_basis_count =
        get_orbital_basis_count(vb_str, representative_orbital);
    const int requested_orbitals = vb_str->noc_block[block_index];

    if (block_basis_count <= 0 || requested_orbitals <= 0) {
      continue;
    }

    memset(overlap_block, 0, sizeof(double) * nb * nb);
    memset(hcore_block, 0, sizeof(double) * nb * nb);

    for (int local_i = 0; local_i < block_basis_count; local_i++) {
      const int global_i =
          vb_str->nv[representative_orbital * nb + local_i] - 1;
      for (int local_j = 0; local_j < block_basis_count; local_j++) {
        const int global_j =
            vb_str->nv[representative_orbital * nb + local_j] - 1;
        overlap_block[local_i * block_basis_count + local_j] =
            vb_str->ssf[global_i * nb + global_j];
        hcore_block[local_i * block_basis_count + local_j] =
            vb_str->hhf[global_i * nb + global_j];
      }
    }

    if (LAPACKE_dsygv(
            LAPACK_COL_MAJOR,
            1,
            'V',
            'U',
            block_basis_count,
            hcore_block,
            block_basis_count,
            overlap_block,
            block_basis_count,
            eigenvalues) != 0) {
      free(overlap_block);
      free(hcore_block);
      free(eigenvalues);
      return 1;
    }

    for (int orbital_offset = 0; orbital_offset < requested_orbitals;
         orbital_offset++) {
      memcpy(
          vb_str->dv + vb_str->blocks[block_index * nd + orbital_offset] * nb,
          hcore_block + orbital_offset * block_basis_count,
          sizeof(double) * block_basis_count);
    }
  }

  scale_guess_back_to_original_basis(vb_str);

  free(overlap_block);
  free(hcore_block);
  free(eigenvalues);
  return 0;
}

static int read_mo_indices(inp_info inp_str, vb_info vb_str, int* mo_indices) {
  FILE* input_file = fopen(inp_str->inpname, "r");
  char line[1024];

  if (input_file == NULL) {
    return 1;
  }
  if (fvbsec(input_file, "$GUS") > 0) {
    fclose(input_file);
    return 1;
  }

  for (int orbital_index = 0; orbital_index < vb_str->nor; orbital_index++) {
    mo_indices[orbital_index] = orbital_index + 1;
  }

  while (fgets(line, sizeof(line), input_file) != NULL) {
    int vb_orbital = 0;
    int mo_orbital = 0;
    if (strstr(line, "$END") != NULL) {
      break;
    }
    if (strchr(line, '#') != NULL) {
      continue;
    }
    if (sscanf(line, "%d %d", &vb_orbital, &mo_orbital) == 2 &&
        vb_orbital >= 1 && vb_orbital <= vb_str->nor) {
      mo_indices[vb_orbital - 1] = mo_orbital;
    }
  }

  fclose(input_file);
  return 0;
}

static int build_hcore_mo_guess(inp_info inp_str, vb_info vb_str) {
  const int nb = vb_str->nb;
  const int nor = vb_str->nor;
  double* overlap_matrix = (double*)malloc(sizeof(double) * nb * nb);
  double* orbital_matrix = (double*)malloc(sizeof(double) * nb * nb);
  double* eigenvalues = (double*)malloc(sizeof(double) * nb);
  int* mo_indices = (int*)malloc(sizeof(int) * nor);

  if (overlap_matrix == NULL || orbital_matrix == NULL ||
      eigenvalues == NULL || mo_indices == NULL) {
    free(overlap_matrix);
    free(orbital_matrix);
    free(eigenvalues);
    free(mo_indices);
    return 1;
  }

  memcpy(overlap_matrix, vb_str->ssf, sizeof(double) * nb * nb);
  memcpy(orbital_matrix, vb_str->hhf, sizeof(double) * nb * nb);
  if (LAPACKE_dsygv(
          LAPACK_COL_MAJOR,
          1,
          'V',
          'U',
          nb,
          orbital_matrix,
          nb,
          overlap_matrix,
          nb,
          eigenvalues) != 0) {
    free(overlap_matrix);
    free(orbital_matrix);
    free(eigenvalues);
    free(mo_indices);
    return 1;
  }

  for (int basis_index = 0; basis_index < nb; basis_index++) {
    for (int mo_index = 0; mo_index < nb; mo_index++) {
      orbital_matrix[mo_index * nb + basis_index] *= vb_str->snorm[basis_index];
    }
  }

  if (read_mo_indices(inp_str, vb_str, mo_indices) != 0) {
    free(overlap_matrix);
    free(orbital_matrix);
    free(eigenvalues);
    free(mo_indices);
    return 1;
  }

  memset(vb_str->dv, 0, sizeof(double) * nb * nor);
  for (int orbital_index = 0; orbital_index < nor; orbital_index++) {
    const double sign = (mo_indices[orbital_index] >= 0) ? 1.0 : -1.0;
    const int selected_mo = abs(mo_indices[orbital_index]) - 1;
    if (selected_mo < 0 || selected_mo >= nb) {
      free(overlap_matrix);
      free(orbital_matrix);
      free(eigenvalues);
      free(mo_indices);
      return 1;
    }

    const int basis_count = get_orbital_basis_count(vb_str, orbital_index);
    for (int coefficient_index = 0; coefficient_index < basis_count; coefficient_index++) {
      const int basis_index =
          vb_str->nv[orbital_index * nb + coefficient_index] - 1;
      vb_str->dv[orbital_index * nb + coefficient_index] =
          sign * orbital_matrix[selected_mo * nb + basis_index];
    }
  }

  normalize(nb, nor, vb_str->dv, vb_str->nv, vb_str->ma, vb_str->ssf);

  free(overlap_matrix);
  free(orbital_matrix);
  free(eigenvalues);
  free(mo_indices);
  return 0;
}

int xmvb_cpp_vbguess(
    hf_info hf,
    inp_info inp_str,
    vb_info vb_str,
    int print_level) {
  (void)hf;

  int status = 0;
  switch (vb_str->iguess) {
    case GUS_AUTO:
      status = (hf != NULL) ? xmvb_cpp_vb_autoguess(hf, vb_str)
                            : build_hcore_block_guess(vb_str);
      break;
    case GUS_UNIT:
      status = xmvb_cpp_vb_unitguess(vb_str);
      break;
    case GUS_READ:
    case GUS_RDCI:
      status = xmvb_cpp_vb_readguess(inp_str, vb_str);
      break;
    case GUS_MO:
      status = build_hcore_mo_guess(inp_str, vb_str);
      break;
    case GUS_NBO:
      printf("Standalone xmvb-cpp runtime does not support GUESS=NBO yet.\n");
      return 1;
    default:
      printf("Unsupported guess type %d in standalone xmvb-cpp runtime.\n", vb_str->iguess);
      return 1;
  }

  if (status != 0) {
    printf("Standalone xmvb-cpp guess construction failed.\n");
    return 1;
  }

  if (print_level > 0) {
    print_init_guess(vb_str);
  }

  return 0;
}
