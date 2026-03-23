/*
 * @Author: yyzhang_2062 20420182202049@stu.xmu.edu.cn
 * @Date: 2022-10-17 18:59:48
 * @LastEditors: yyzhang_2062 20420182202049@stu.xmu.edu.cn
 * @LastEditTime: 2022-12-23 09:58:38
 * @FilePath: \chonps\mol.h
 * @Description:
 */
#ifndef _MOL_MOL_H_
#define _MOL_MOL_H_
#include <stdio.h>
#include <stdbool.h>
#include "mol/data_struct.h"
#include <string.h>
#include "lapack.h"
#include "lapacke.h"
#include "cblas.h"
#include <math.h>
#include <complex.h>
#include <omp.h>
#include "xc.h"
#include <assert.h>
#define my_max(a, b) (((a) > (b)) ? (a) : (b))
#define my_min(a, b) (((a) < (b)) ? (a) : (b))
#define ELEMENT_VAL 0
#define CENTER_IND 1
#define ATOM_IND 0
#define ANGULAR_VAL 1
#define EXP_IND 2
#define NPRIM_VAL 3
#define COEFF_IND 4
#define NECP_VAL 0
#define ECPCOE_IND 1
#define ECPPOW_IND 2
#define ECPEXP_IND 3
#define ECPATM_IND 4
#define ECP_PART 5
#define MY_PI 3.141592653589793238462643383279
#define MAX_ANG 6
#define atm(parameter, index) atm->atm[(index) * 2 + parameter]
#define prm(parameter, index) prm->prm[(index) * 3 + parameter]
#define bas(parameter, index) bas->bas[(index) * 5 + parameter]
#define aux(parameter, index) aux->bas[(index) * 5 + parameter]
#define ecp(parameter, index) ecp->ecp[(index) * 6 + parameter]
#define pair(shli, shlj) search_pair(shli, shlj, pair)
#define MAX_MULTIPOLE 5
#define BACKWARD 1
#define FORWARD -1
#define PAIR_TOLER 1E-12
#define CACHE_LEN 2048
#define NORM_S 0.28209479177387814
#define NORM_P 0.4886025119029199
#define A2AU 0.529177249
#define GET_SIGN(x) (x) >= 0 ? 1 : -1
#define SET_ONEBIT(num, n) (num) | (1 << (n))
#define SET_ZEROBIT(num, n) (num) & (~(1 << (n)))
#define GET_BIT(num, n) (((num) >> (n)) & 1)
#define PHI(b) 1 - 2 * ((b) & 1)
#define SQRT_2 1.4142135623730951
#define ALPHA_SPIN 20
#define BETA_SPIN 21
#define copy_tensor3d(A, B, l, m, n) memcpy((A[0][0]), (B[0][0]), sizeof(double) * l * m * n)
#define copy_matrix(A, B, m, n) memcpy((A[0]), (B[0]), sizeof(double) * m * n)
#define copy_vector(A, B, m) memcpy((A), (B), sizeof(double) * m)
typedef double _Complex x_complex;
typedef int *List;
typedef double *Vector;
typedef double **Matrix;
typedef double ***Tensor3D;
typedef double ****Tensor4D;
typedef struct ShellPair *shell_pair;
typedef struct PairInfo *pair_info;
typedef struct AtmInfo *atm_info;
typedef struct BasInfo *bas_info;
typedef struct PrmInfo *prm_info;
typedef struct MolInfo *mol_info;
/**
 * @brief which type builder is used for Fock matrix's calculation or orbitals' integrals
 *
 */
typedef enum builder_type
{
    Exact_builder,         // using exact 4c2e integral
    DensityFit_builder,    // using 3c2e integral to do RI-density fitting, auxiliary basis functions are needed
    SemiNumerical_builder, // using 3c1e integral to do semi-numerical calculation, grids are needed
    Reduce_builder         // reduced form as THC and rCOSX calculation, grids and auxiliary basis functions are both needed
} builder_type;
/**
 * @brief strcut to store information of atoms
 * @author Zhang Yueyang
 * @date 2024.02.27
 */
struct AtmInfo
{
    int natm;                // the number of atoms in system
    int charge, mult;        // charge and multi-spin
    int alpha_num, beta_num; // electron number of alpha and beta spin
    List atm;                // information of atoms
    Vector value;            // double data of atoms
    char *mol_fname;         // the file's name of '.mol' file
};
#ifdef __cplusplus
extern "C"
{
#endif
    atm_info init_atm(const char *mol_fname,int unit_bohr);
    void del_atm(atm_info atm);
    void print_atm(const atm_info atm);
    atm_info get_atm_seg(int charge, int mult, int begin_id, int end_id, const atm_info atm);
    Vector get_atm_coord(int a, const atm_info atm);
#ifdef __cplusplus
}
#endif
/**
 * @brief strcut to store information of basis function (CGTOs)
 * @author Zhang Yueyang
 * @date 2024.02.27
 */
struct BasInfo
{
    atm_info atm;    // which these basis functions belong to
    int nbas, msize; // the number of shells and basis functions
    List bas;        // the information of basis function
    List shls_p;     // the begin index of basis function
    Vector value;    // double data of basis functions
    char *bas_fname; // the file's name of basis function
    int qoff;        // the length of value
};
#ifdef __cplusplus
extern "C"
{
#endif
    bas_info init_bas(const char *bas_fname, const atm_info atm);
    bas_info init_aux(int lev, bool if_star, const bas_info bas);
    void del_bas(bas_info bas);
    double cal_bas_extent(int shli, const bas_info bas);
    void print_bas(const bas_info bas);
    Vector get_bas_exp(int shli, const bas_info bas);
    Vector get_bas_coe(int shli, const bas_info bas);
#ifdef __cplusplus
}
#endif
/**
 * @brief strcut to store information of basis function (PGTOs)
 *        DOI: 10.1021/acs.jctc.8b00358
 * @author Zhang Yueyang
 * @date 2024.02.27
 */
struct PrmInfo
{
    bas_info bas;    // which CGTO function these prime gaussian functions belong to
    atm_info atm;    // which these prime basis functions belong to
    int nprm, msize; // the number of shells and prime basis functions
    List prm;        // the information of prime basis function
    List shls_p;     // the begin index of basis function
    Vector value;    // double data of prime basis functions
    int c2p_len;
    List c2p_index; // (id[0], id[1], coe) for sparse matrix, id[0] is the index of bas and id[1] is the index of prm
    Vector c2p_coe;
};
prm_info init_prm(const bas_info bas);
void del_prm(prm_info prm);
void trans_c2p_mat(Matrix p_matrix, const Matrix c_matrix, const prm_info prm);
void trans_p2c_mat(Matrix c_matrix, const Matrix p_matrix, const prm_info prm);
double get_prm_exp(int shli, const prm_info prm);
double get_prm_coe(int shli, const prm_info prm);
struct ShellPair
{
    int off_vec[3];  // offset vector used for periodic calculation
    int shli, shlj;  // index of pairs' bas
    int di, dj, dij; // number of prim gaussian functions belong to each bas
    int li, lj;
    double AB[3];
    /// @brief the parameters below are all sizeof 'dij' or '3*dij'
    double *xi, *half_xi_; // xi=alpha+beta; half_xi_=0.5/xi
    double *P, *PA, *PB;   // P_i=(alpha*A_i+beta*B_i)/xi
    double *K_ab;          // K_ab=exp(alpha*beta/xi*AB^2) [K_abx, K_aby, K_abz, K_ab,...]
    double extent;         // extent=max{sqrt(log(1E10)/xi)+|P-0.5*(A+B)|^2}
    double EST;            // double used to do precreening for COSX exchange
};
struct PairInfo
{
    int nbas;
    int len;
    shell_pair shp;
    qtree *search_tree; // A binary tree used to search (i,j) in pair list.
};
pair_info init_pair(const bas_info bas);
pair_info init_prm_pair(const prm_info prm);
void del_pair(pair_info pair);
shell_pair search_pair_shp(int shli, int shlj, const pair_info pair);

struct ECPInfo
{
    int necp;     // the number of ECP
    int core_ele;    // the number of electrons in core
    List ecp;        // List used to store ECP's information
    Vector value;    // Vector used to store information of ECP
    char *ecp_fname; // the file's name of ECP
};
typedef struct ECPInfo* ecp_info;
ecp_info init_ecp(const char* ecp_fname, atm_info atm);
void del_ecp(ecp_info ecp);
Vector get_ecp_exp(int n, const ecp_info ecp);
Vector get_ecp_coe(int n, const ecp_info ecp);
Vector get_ecp_pow(int n, const ecp_info ecp);
/**
 * @brief strcut to store information of molecule, this structure is the most basic one
 * @author Zhang Yueyang
 * @date 2024.02.27
 */
struct MolInfo
{
    atm_info atm;   /// atoms
    bas_info bas;   /// basis function
    prm_info prm;   /// prim gaussian function for exact calculation
    pair_info pair; /// the shell pairs
    pair_info prm_pair;
};
int get_mol_charge(const mol_info mol);
int get_mol_mult(const mol_info mol);
int get_mol_natm(const mol_info mol);
int get_mol_nbas(const mol_info mol);
int get_mol_msize(const mol_info mol);
mol_info get_mol_seg(int charge, int mult, int begin_id, int end_id, const mol_info mol);
#ifdef __cplusplus
extern "C"
{
#endif
    /// @brief function of data_type
    List malloc_list(size_t size);
    Vector malloc_vector(size_t size);
    Matrix malloc_matrix(size_t size1, size_t size2);
    Tensor3D malloc_tensor3d(size_t size1, size_t size2, size_t size3);
    Tensor4D malloc_tensor4d(size_t size1, size_t size2, size_t size3, size_t size4);
    void free_list(List data);
    void free_vector(Vector data);
    void free_matrix(Matrix data);
    void free_tensor3d(Tensor3D data);
    void free_tensor4d(Tensor4D data);
    double vv_dot(const Vector a, const Vector b, const int m1);
    double matrix_inner(const Matrix a, const Matrix b, int m_size);
    void trafock(Vector f_matrix, Vector x_matrix, int msize);
    void gtrafock(Matrix f_out, const Matrix c_matrix1, const Matrix c_matrix2, const Matrix f_matrix,
                  const int size1, const int size2, const int mszie);
    void gtrafock_t(Matrix f_out, const Matrix c_matrix1, const Matrix c_matrix2, const Matrix f_matrix,
                    const int size1, const int size2, const int mszie);
    void tensor_contr(Tensor3D T, const Matrix D1, int a, const Matrix D2, int b, const Matrix D3, int c, int d);
    void GramSchmit_orth(Matrix matrix, int msize);
    int GramSchmit_ld_check(Matrix c_matrix, const Matrix s_matrix, int size1, int msize);
    void cal_symatrix_reverse(Matrix matrix, int size);
    void cal_matrix_reverse(Matrix matrix, int size);
    void cal_matrix_halfrev(Matrix matrix, int size, char uplo);
    void cal_matrix_syhalfrev(Matrix matrix, int size);
    double cal_deter(Matrix matrix, int size);
    void print_matrix(Matrix matrix, int x, int y);
    void print_cmatrix(Matrix matrix, int msize, const mol_info mol, const Vector egn_value, int orb_num);
    double Fact(int a);
    double Comb(int n, int m);
    int idnint(double a);
    int xint_gtolen(int ang);
    double xint_norm(int angular, double exp_num);
    mol_info init_mol(const char *mol_fname, const char *bas_fname,int unit_bohr);
    void del_mol(mol_info mol);
    void print_mol(const mol_info mol);
    double cal_nuc_rep(const mol_info mol);
    int ele2charge(const char *ele);
    char *charge2ele(int charge);
    char *get_ang_tag(int i, int ang);
#ifdef __cplusplus
}
#endif
#endif
