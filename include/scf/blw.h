#ifndef _SD_SCF_BLW_H_
#define _SD_SCF_BLW_H_
#include "scf/hf.h"
struct SegmentInfo
{
    int begin_index, end_index;
    int msize, alpha_num, beta_num;
    int charge;
    Matrix s_matrix;
    Matrix f_matrix;
    Matrix c_matrix;
    Matrix x_matrix;
    Vector egn_value;
    int bas_begin_index; // for normal calculation
};
typedef struct SegmentInfo *segment_info;
struct BLWInfo
{
    // hf_info hf;
    mol_info mol;
    int msize;
    coulomb_info coulomb;
    exchange_info exchange;
    dft_info dft;
    Matrix s_matrix;                     // overlap matrix
    Matrix t_matrix, v_matrix, h_matrix; // one-electron Hamiltonian term
    Matrix j_matrix;                     // coulomb matrices point to coulomb_info
    Matrix k_matrix;                     // exchange matrices point to exchange_info
    Matrix f_matrix;                     // Fock matrices
    Matrix c_matrix;                     // MO coefficients matrices
    Matrix d_matrix;                     // density matrices
    Matrix xc_matrix;                    // matrix of V_xc for dft
    double tol_energy;
    double e_t, e_v, e_j, e_k, e_xc;
    double nuc_rep;
    int segment_num;
    segment_info *segments;
    /**
     * @brief subspace BLW is used
     *
     */
    bool if_sub;         // if the BLW calculation is done under the defined subspace
    int sub_size;        // the size of vectors in subspace, this number is smaller than msize
    Matrix subspace;     // matrix with shape of sub_size * msize
    Matrix f_matrix_sub; // Fock matrix in subspace
    Matrix s_matrix_sub; // overlap matrix in subspace
    Matrix c_matrix_sub; // MO matrix of subspace
    Matrix d_matrix_sub; // density matrix of subspace
    /**
     * @brief name of file for J/K builder
     */
    char *jaux_fname, *kaux_fname;       // file name of auxiliary basis function
    char *jgrids_fname, *kgrids_fname;   // file name of grids file for SCF iteration
    char *jfgrids_fname, *kfgrids_fname; // file name of final grids for final integrals to get energy
    /**
     * @brief data used to do BLW-EDA
     *        DOI: https://doi.org/10.1063/1.481185
     */
    double E0;
    double TOL;          // the interaction energy of monomers
    double FRZ, POL, CT; // frozen term, polarization term, charge transfer term
};
typedef struct BLWInfo *blw_info;
/**
 * @brief init blw by define subspace
 *        this inital function allows do BLW calculation with overlap
 * @param space_size the size of each subspace
 * @param subspace coefficient matrix of subspace with shape of space_size * msize
 * @param mol
 * @return blw_info
 */
blw_info init_blw(const List space_size, const Matrix *subspace, const mol_info mol);
blw_info init_blw_eda(const mol_info mol);
void set_blw_coulomb(Jbuilder_type jtype, blw_info blw);
void set_blw_exchange(Kbuilder_type ktype, blw_info blw);
void load_blw_jaux(const char *aux_fname, blw_info blw);
void load_blw_kaux(const char *aux_fname, blw_info blw);
void load_blw_jgrids(const char *jgrids_fname, const char *jfgrids_fname, blw_info blw);
void load_blw_kgrids(const char *kgrids_fname, const char *kfgrids_fname, blw_info blw);
void load_blw_dft(double hf_frac, GRIDS_TYPE gtype, const int dft_id[], const double dft_frac[], int num, int disp_type, int dft_name, blw_info blw);
void del_blw(blw_info blw);
extern void cal_trans_d_matrix(Matrix d_matrix, const Matrix c_matrix1, const Matrix c_matrix2, const Matrix s_matrix, const int orb_num, const int msize);
void do_blw(blw_info blw);
void do_blw_eda(blw_info blw);
#endif
