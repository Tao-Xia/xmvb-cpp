#ifndef _EDA_EDA_H_
#define _EDA_EDA_H_
#include "fock/dft.h"
#include "fock/fock.h"
struct EDAInfo
{
    dft_info dft;
    coulomb_info coulomb;
    exchange_info exchange;
    List alpha_num, beta_num;
    mol_info mol;
    pair_info pair;
    int seg_num, msize;
    List seg_natm;
    Vector tol_energy, nuc_rep;
    Matrix s_matrix, x_matrix, s_half_matrix_;
    Matrix *h_matrix, *v_matrix;
    double TOL, ES, EX, EC, POL, REP;
    double time;
    // used to do real-space EDA to attribute to each atom
    bool if_atomic;
    Vector atm_TOL, atm_ES, atm_EX, atm_EC, atm_POL, atm_REP;
    // if charge transfer term is calculated
    bool if_ct;
    double CT, POL_;
    Vector atm_CT, atm_POL_;
};
typedef struct EDAInfo *eda_info;
eda_info init_eda(const mol_info mol);
void del_eda(eda_info eda);
void load_eda_dft(double hf_frac, GRIDS_TYPE gtype, const int dft_id[], const double dft_frac[], int num, int disp_type, int dft_name, eda_info eda);
void do_eda(const Vector energy, const Matrix *d_matrix_list, const Matrix *d_matrix_b_list, eda_info eda);
void do_eda_atomic(const Matrix *d_matrix_list, const Matrix *d_matrix_b_list, eda_info eda);
void print_eda(const eda_info eda);
// function for calculate charge transfer term with is seemed as short range polarization
/**
 * @brief calculate charge transfer
 * 
 * @param d_matrix_list density matrices of alpha (total) spin
 * @param d_matrix_b_list density matrices of beta spin
 * @param eda struct of doing EDA
 */
void cal_eda_ct(const Matrix *d_matrix_list, const Matrix *d_matrix_b_list, eda_info eda);
#endif
