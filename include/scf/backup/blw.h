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
    /**
     * @brief if normal calculation is done (without overlap between segments)
     *        space_size = 0 and subspace = NULL for default choice
     */
    bool if_set;         // bool data to judge if subspace is pointed
    int bas_begin_index; // for normal calculation
    int space_size;      // when general calculation is done then space_size = msize
    Matrix subspace;     // subspace to be construct BLW orbitals
    Matrix c0_matrix;    // this is coefficient matrix in original basis set
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
void del_blw(blw_info blw);
extern void cal_trans_d_matrix(Matrix d_matrix, const Matrix c_matrix1, const Matrix c_matrix2, const Matrix s_matrix, const int orb_num, const int msize);
void do_blw(blw_info blw);
#endif
