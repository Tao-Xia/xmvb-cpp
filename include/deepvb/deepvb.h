#pragma once

#include "vb/vb.h" 
#include <ISO_Fortran_binding.h>
#include "inpout/output_func.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <lapacke.h>
#include <cblas.h>
#include <stdlib.h>
struct deepvbhinfo
{
    vb_info vb;
    double *sso;
    double *hho;
    double *hvb_1e;      // one-electron Hamiltonian matrix     [nstr, nstr]
    double *svb_1e;      // overlap matrix for the determinants [nstr, nstr]
    double *sorb_alpha;  // alpha overlap between determinants  [nae_alpha, nae_alpha]
    double *sorb_beta;   // beta overlap between determinants   [nae_beta, nae_beta]
    double *horb_alpha;  // alpha 1-e integral between determinants(orbital basis) [nae_alpha, nae_alpha]
    double *horb_beta;   // beta 1-e integral between determinants(orbital basis) [nae_beta, nae_beta]
    double *rdm_alpha;   // alpha reduced density matrix        [nae_alpha, nae_alpha]
    double *rdm_beta;    // reduced density matrix              [nae_beta, nae_beta]
    /* Following variables are used for model training and inference */
    int *alpha_det;      // alpha det
    int *beta_det;       // beta det
    // Following variables are used for gradient calculation
    double *G22;
    double *Q22;
    double *grda;
    double *grdv;
    double *grdbas;
    double *grad;
    
};
typedef struct deepvbhinfo* deepvbh_info;


#ifdef __cplusplus
extern "C" {
#endif
void get_detpair_overlap_integral(uint64_t *det1, uint64_t *det2, double *orbital_integral, int norb, double *out_matrix, int n1);
void get_detpair_oneelectron_integral(uint64_t *det1, uint64_t *det2,double *hho, int norb, double *rdm, int n1,double *horb, double *rdm_out);
void recover_generalized_eigenvectors_with_energy(double *energy, double *col, const double *H_ortho_in, double *svb, const double enuc, const int nstr,vb_info vb_str);             
void save_target(double *hvb, double *svb, int nstr, const char *folder_path, const char *file_name);
void save_bovb_data(double *hvb, double *svb, double *hvb_1det, double *svb_1det, int nstr, const char *folder_path, const char *file_name);
void save_Matrix_type_to_bin(Matrix mat, int n, int m, const char *filename);
void save_bit_train_data(int *alpha_bit_i, int *beta_bit_i,int *alpha_bit_j, int *beta_bit_j, int nao, int nstr,char *folder_path, char *file_name);
void orthogonalize_hamiltonian(double *H_ortho, double *hvb, double *svb, int nstr);
void str_expand_to_det(vb_info vb_str, deepvbh_info deepvbh);
void calc_hamiltonian_overlap_rdm(vb_info vb_str, deepvbh_info deepvbh, int *alpha_det_i, int *beta_det_i, int *alpha_det_j, int *beta_det_j, int i, int j);
void calc_detpair_rdm(vb_info vb_str, deepvbh_info deepvbh, uint64_t *det_alpha_i, uint64_t *det_beta_i, uint64_t *det_alpha_j, uint64_t *det_beta_j, int i, int j, double *ov);
void cal_strpair_rdm(vb_info vb_str, deepvbh_info deepvbh);
void deepvbscf_rdm(vb_info vb_str, double *eneygy, double *sums, double *grad);
void deepvbscf_bovb(vb_info vb_str, double *eneygy, double *sums, double *grad);
double orthogonal_matrix_determinant(int n, double* A, int lda);

void cofactor1(double *A, int n, double *C1, double *det);
void transpose_matrix_inplace(double *src, int n);
void eigen_decomposition(double *mat, int n, double *eigval);

void save_matrix_to_bin(double *mat, int n, int m, const char *filename);
void save_vector_to_bin(const void *vec, int n, const char *filename, size_t elem_size);
void save_number_to_bin(void *data, size_t element_size, const char *filename);
void ensure_dir_exists(const char *path);
void save_pairs_for_pytorch(int *ntstr, int nstr, int nel, const char *folder_path, const char *filename);

#ifdef __cplusplus
}
#endif
// void* get_taux_ptr();
// void* get_tori_ptr();
// void* get_hho_ptr();
// void* get_sso_ptr();
                                
                                
