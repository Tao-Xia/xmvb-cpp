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
    Vector atm_TOL, atm_ES, atm_EX, atm_EC, atm_POL, atm_REP; // energy term on each atom
    /**
     * @brief if charge transfer term is calculated
     * 
     */
    bool if_ct;
    double CT, POL_;
    double energy_ct; // the energy of media state
    Vector atm_CT, atm_POL_;
    Matrix d_matrix_ct; // density matrix of charge-transfer
    /**
     * @brief name of file for J/K builder
     */
    char *jaux_fname, *kaux_fname;       // file name of auxiliary basis function
    char *jgrids_fname, *kgrids_fname;   // file name of grids file for SCF iteration
    char *jfgrids_fname, *kfgrids_fname; // file name of final grids for final integrals to get energy
};
typedef struct EDAInfo *eda_info;
eda_info init_eda(const mol_info mol);
void set_eda_coulomb(Jbuilder_type jtype, eda_info eda);
void set_eda_exchange(Kbuilder_type ktype, eda_info eda);
void load_eda_jaux(const char *aux_fname, eda_info eda);
void load_eda_kaux(const char *aux_fname, eda_info eda);
void load_eda_jgrids(const char *jgrids_fname, const char *jfgrids_fname, eda_info eda);
void load_eda_kgrids(const char *kgrids_fname, const char *kfgrids_fname, eda_info eda);
void del_eda(eda_info eda);
void load_eda_dft(double hf_frac, GRIDS_TYPE gtype, const int dft_id[], const double dft_frac[], int num, int disp_type, int dft_name, eda_info eda);
void do_eda(const Vector energy, const Matrix *d_matrix_list, const Matrix *d_matrix_b_list, eda_info eda);
void do_eda_atomic(const Matrix *d_matrix_list, const Matrix *d_matrix_b_list, eda_info eda);
void print_eda(const eda_info eda);
void cal_eda_ct(const Matrix *d_matrix, eda_info eda);
void cal_eda_bd(const Matrix *d_matrix, const Matrix *d_matrix_b, eda_info eda);
#endif
