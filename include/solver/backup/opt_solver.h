#ifndef _OPT_SLOVER_H_
#define _OPT_SLOVER_H_
#include "mol/mol.h"
#define CDIIS_matrix(i, j) CDIIS_matrix[(i) * CDIIS_size + j]
#define ADIIS_matrix(i, j) ADIIS_matrix[(i) * ADIIS_size + j]
#define CDIIS_matrix_b(i, j) CDIIS_matrix_b[(i) * CDIIS_size + j]
#define ADIIS_matrix_b(i, j) ADIIS_matrix_b[(i) * ADIIS_size + j]
typedef struct bfgs *bfgs_info;
bfgs_info init_bfgs(int msize, int size);
void del_bfgs(bfgs_info bfgs);
void do_bfgs(Vector grad, bfgs_info bfgs);

typedef struct DiisInfo *diis_info;
/* DIIS */
diis_info init_diis(int size, int msize);
void set_diis_tr(double tr1, double tr2, diis_info diis);
int del_diis(diis_info diis);
/**
 * @brief this DIIS is used for SCF converging
 *          ADIIS + CDIIS is used
 * @param s_matrix overlap matrix if NULL is given then identity matrix is as given parameter
 * @param f_matrix Fock matrix 
 * @param d_matrix density matrix
 * @param diis 
 * @return double ||FSD-DSF||
 */
double do_diis(Matrix s_matrix, Matrix f_matrix, Matrix d_matrix, diis_info diis);
#endif
