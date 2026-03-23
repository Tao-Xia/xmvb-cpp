#ifndef _SDSCF_HF_H_
#define _SDSCF_HF_H_
#include "fock/fock.h"
#include "fock/dft.h"
#define RHF_WORK -1
#define UHF_WORK 0
#define ROHF_WORK 1
#define CUHF_WORK 2

struct HFInfo
{
    int open_type; // which type open shell calculation is done, 0 for UHF, 1 for ROHF and 2 for CUHF
    int batch_size, msize;
    mol_info mol;
    coulomb_info coulomb;
    exchange_info exchange;
    dft_info dft;                          // as default, Hatree-Fock method is used and dft is setted to NULL
    Matrix s_matrix, x_matrix;             // X = S^{-1/2} and take symmetric form
    Matrix s_half_matrix;                  // S^{1/2} take symmetric form
    Matrix t_matrix, *v_matrix, *h_matrix; // one-electron Hamiltonian term
    Matrix *j_matrix;                      // coulomb matrices point to coulomb_info
    Matrix *k_matrix, *k_matrix_b;         // exchange matrices point to exchange_info
    Matrix *f_matrix, *f_matrix_b;         // Fock matrices
    Matrix *c_matrix, *c_matrix_b;         // MO coefficients matrices
    Matrix *d_matrix, *d_matrix_b;         // density matrices
    Matrix *xc_matrix, *xc_matrix_b;       // matrix of V_xc for dft
    Matrix *ext_p;                         // external potential, default is NULL
    Vector *alpha_egn, *beta_egn;          // orbital's energy
    Vector tol_energy;
    /**
     * @brief Information of segments
     */
    List alpha_seg, beta_seg, mult_seg, charge_seg, atm_seg;
    // kinetic energy, n-e attraction energy, coulomb energy, exchange energy
    Vector e_t, e_v, e_j, e_k, e_xc;
    Vector nuc_rep;
    Vector S2; // spin S^2
    struct
    {
        int max_iter;
        double max_delta_e, max_delta_d;
        double max_fds_err;
    } ctrl; // struct for controlling SCF calculation's converge
    int diis_size, diis_tr1, diis_tr2;
    /**
     * @brief this part is used to constrain optimization orbitals in a subspace
     *        the subspace is spanned by orbitals as
     *          \phi_u = \sum_\nu C_{u\nu}\chi_\nu
     */
    bool if_sub_def, if_sub_use;           // if constrained subspace is defined/used
    int sub_size;                          // the size of subspace
    Matrix subspace;                       // the compends of subspace
    Matrix s_matrix_sub;                   // the overlap matrix of subspace
    Matrix x_matrix_sub;                   // the S^{-1/2} of subspace
    Matrix *f_matrix_sub, *f_matrix_b_sub; // Fock matrix of subspace
    Matrix *c_matrix_sub, *c_matrix_b_sub; // MO matrix of subspace
    Matrix *d_matrix_sub, *d_matrix_b_sub; // density matrix of subspace
    /**
     * @brief this part is used for Anderson impurity model (AIM)
     *        AIM can be seen as the system is coupled with a fermi bath
     *          H = H_s + H_b + H_hyb
     *        H_s: the Hamiltonian of system as normal one
     *        H_b: \sum_i\epsilon_i*c*_ic_i where i is the index of bath orbital
     *        H_byb: \sum_{ip}V_{ip}c*_id_p + h.c. where p is the index of system orbital
     *        as to make the number of system (or called impurity) correct chemical potential \mu is introduced
     *          H' = H + \mu\sum_pn_p where n_p is the occupied number operator of system
     */
    bool if_aim_def;                       // if AIM calculation can be done the default choice is false
    int bath_size;                         // the number of bath site
    double aim_mu;                         // the chemical potential default is 0.0
    double aim_enum;                       // the electron's number of AIM in impurity
    Vector bath_e;                         // the energy of bath site as epsilon_i
    Matrix bath_coup;                      // the coupling with impurity as V_ip which is real as V*_ip = V_ip
    Matrix s_matrix_aim;                   // the overlap matrix of AIM
    Matrix x_matrix_aim;                   // the S^{-1/2} of AIM
    Matrix *f_matrix_aim, *f_matrix_b_aim; // Fock matrix of AIM with shape of (bath_num + sub_size) * (bath_num + sub_size)
    Matrix *c_matrix_aim, *c_matrix_b_aim; // MO matrix of AIM with shape of (bath_num + sub_size) * (bath_num + sub_size)
    Matrix *d_matrix_aim, *d_matrix_b_aim; // density matrix of AIM 
};
typedef struct HFInfo *hf_info;
hf_info init_hf(const mol_info mol);
void set_hf_sub(bool if_sub_use, hf_info hf);
void set_hf_mu(double aim_mu, hf_info hf);
void set_hf_diis(int diis_size, int tr1, int tr2, hf_info hf);
void set_hf_ctrl(int max_iter, double max_delta_e, double max_delta_d, double max_fds_err, hf_info hf);
void load_hf_dft(double hf_frac, GRIDS_TYPE gtype, const int dft_id[], const double dft_frac[], int num, int disp_type, int dft_name, hf_info hf);
void load_hf_sub(int sub_size, const Matrix subspace, hf_info hf);
void load_hf_aim(int bath_size, const Matrix bath_coup, const Vector bath_e, hf_info hf);
void del_hf(hf_info hf);
double cal_hf_energy(const Matrix d_matrix, const Matrix d_matrix_b, const Matrix h_matrix, const Matrix f_matrix, const Matrix f_matrix_b, const int msize);
void cal_c_matrix(Matrix c_matrix, const Matrix f_matrix, const Matrix s_matrix, const Matrix x_matrix, Vector egn_value, int msize);
void do_rhf(int print_level, hf_info hf);
void do_uhf(int print_level, hf_info hf);
void do_hf_aim(int print_level, hf_info hf);
void cal_no_orb(Matrix c_matrix_no, Vector occ_num, const Matrix s_half_matrix, const Matrix x_matrix, const Matrix d_matrix, int msize);
void cal_rhf_grad(Vector* grad, const hf_info hf);
/**
 * @brief Inital guess
 *          HCORE: use only one-electron matrix 'h' as Fock matrix
 *          SAD: single atomic density
 *          SAP: single atomic potential
 */
void do_hcore_InitalGuess(hf_info hf);
void do_sad_InitalGuess(hf_info hf);
// void do_sap_InitalGuess(hf_info hf);
#endif
