#ifndef _EDA_QMM_H_
#define _EDA_QMM_H_
#include "eda/eda.h"
/**
 * @brief this struct is defined for ligand-protein interaction QMM method
 *
 */
struct QMMInfo
{
    mol_info mol;      // structure of total system
    mol_info sys;      // structure of the system we interest in
    Matrix s_matrix;
    Matrix t_matrix, *v_matrix, *h_matrix;
    Matrix *j_matrix, *k_matrix, *k_matrix_b;
    Matrix *c_matrix, *c_matrix_b;
    Matrix *d_matrix, *d_matrix_b;
    Matrix *f_matrix, *f_matrix_b;
    Matrix *xc_matrix, *xc_matrix_b;
    Vector env_charge; // the screening charges of enviornment atoms
    double tot_e;      // the total number of electrons in QM part
    double T, mu;      // the distribution of electrons in orbitals following Fermi-Dirac as n_i = 1 / 1 + exp(-(e_i - mu)/T)
    coulomb_info coulomb;
    exchange_info exchange;
    dft_info dft;
    eda_info eda;
};
typedef struct QMMInfo* qmm_info;
qmm_info init_qmm(int lig_natm, int lig_charge, int lig_mult, const mol_info mol);
void del_qmm(qmm_info qmm);
#endif
