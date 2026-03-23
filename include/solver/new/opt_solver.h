#ifndef _OPT_solver_H_
#define _OPT_solver_H_
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

struct DfDIISInfo
{
    Matrix err_matrix;
    Vector *df_coe;
    Vector *d_vector;
    Vector extra_coe;
    int size, msize, p_size;
};
typedef struct DfDIISInfo *df_diis_info;
df_diis_info init_df_diis(const int msize, const int size);
void del_df_diis(df_diis_info df_diis);
double do_df_diis(Vector d_vector, Vector df_coe, const Matrix kernel, df_diis_info df_diis);
/**
 * @brief Conjugate gradient method for solving linear equation
 *        CG for symmetric matrix and BCG for the normal one
 */
struct CGInfo
{
    int step_num;         // the number of steps for iteration
    int size;             // the length of vector to be solved
    Vector Ax_, Ax, x, b; // the vectors of Ax = b
    Vector p, q, r, s;    // media vector while p,r is used for BCG
    Vector r_, s_;        // vector for the previous step
    double err;           // the error of |Ax - b|
};
typedef struct CGInfo *cg_info;
typedef void (*Ax_builder)(Vector Ax, const Vector x, void *A);
cg_info init_cg(int size, const Vector b); // init structure of CG solver
void del_cg(cg_info cg);
void update_cg(Ax_builder builder, void *info, cg_info cg);
void update_bcg(Ax_builder builder, void *info, cg_info cg);
#endif
