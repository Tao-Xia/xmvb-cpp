#ifndef _MOL_XINT_H_
#define _MOL_XINT_H_
#include "mol/mol.h"
typedef struct XIntInfo *xint_info;
/**
 * @brief Functions for Boys function's calculation
 *        Chebyshev interpolation is used
 *        structure of Boys function is global
 */
void init_boys();
void del_boys();
/**
 * @brief calculate Boys function as
 *          F(m,T) = int_0^1exp(-x*t^2)*t^{2m}dt
 * @param m order of Boys function
 * @param x param
 * @param result with the length of 'm' at least
 */
void cal_boys(int m, double x, Vector result);

struct XIntInfo
{
    mol_info mol;
    bas_info aux;                   /// if auxiliary basis is used to do 3c2e integrals
    Vector cache;                   /// cache used for electron integrals
    double lr_frac, sr_frac, omega; /// range-separate calculation the default setting is [1., 0., 0.]
    int max_ang, max_ang_aux;
    // data used to do Rys 3c2e, default number is setted to NULL
    bool if_3c2e; /// the default value is false
    Tensor3D rr3c2e_x, rr3c2e_y, rr3c2e_z;
    Matrix rr3c2e_x0, rr3c2e_y0, rr3c2e_z0;
    /**
     * @brief Integrals for center A, B. The center C's value can be gotten by translation invariability
     *        relationship as d(AB|C)/dx = 0
     */
    Tensor3D rr3c2e_Ax, rr3c2e_Ay, rr3c2e_Az;
    Tensor3D rr3c2e_Bx, rr3c2e_By, rr3c2e_Bz;
};
xint_info init_xint(const mol_info mol);
void load_xint_aux(const bas_info aux, xint_info xint);
void set_xint_rs(double lr_frac, double sr_frac, double omega, xint_info xint);
void del_xint(xint_info xint);
// HRR function
void xint_hrr(Vector result, Vector cache, const int li, const int lj, const double AB[3]);
// overlap type integrals
void xint_overlap(Vector buf, const int shlij, xint_info xint);
void xint_kinetic(Vector buf, const int shlij, xint_info xint);
void xint_polar(Vector buf, const int shlij, xint_info xint);
void xint_overlap_grad(Vector buf, const int shlij, xint_info xint);
void xint_kinetic_grad(Vector buf, const int shlij, xint_info xint);
void cal_overlap_matrix(Matrix s_matrix, const mol_info mol);
void cal_project_matrix(Matrix p_matrix, const mol_info mol1, const mol_info mol2);
void cal_overlap_grad_matrix(Matrix *s_matrix_grad, const mol_info mol);
void cal_kinetic_matrix(Matrix t_matrix, const mol_info mol);
// 3c1e type integrals for nucleus-electron interaction
void xint_3c1e(Vector buf, const int shlij, const double G[3], xint_info xint);
void xint_3c1e_prm(Vector buf, const int shlij, const double G[3], xint_info xint);
void xint_ne(Vector buf, const int shlij, xint_info xint);
void cal_nucleus_matrix(Matrix v_matrix, const mol_info mol);
void xint_3c1e_grad(Vector buf, const int shlij, const double G[3], xint_info xint);
void xint_ne_grad(Vector buf, const int shlij, xint_info xint);
// 2c1e type integrals for auxiliary basis function
void xint_2c1e(Vector buf, const int shla, const double G[3], xint_info xint);
// 2c2e type integrals for auxiliary basis function
void xint_2c2e(Vector buf, const int shla, const int shlb, xint_info xint);
void xint_2c2e_grad(Vector buf, int shla, int shlb, xint_info xint);
void cal_2c2e_matrix(Matrix e2_matrix, const bas_info aux, const mol_info mol);
// 3c2e type integrals for RI
void xint_3c2e(Vector buf, const int shlij, const int shla, xint_info xint);
void xint_3c2e_grad(Vector buf, const int shlij, const int shla, xint_info xint);

struct SliceInfo
{
    int seg_lim;           // the max number of basis^2 can be stored in one segment
    int seg_num;           // the number of segments
    List seg_size;         // the pairs' number for each segments
    List *pair_list;       // the index of pair
    List id_off;           // offset value of each segment's id1 and id2
    List id1_len, id2_len; // the length of each segment's id1 and id2
    xint_info xint;        // structure of integrals
    mol_info mol;          // structure of mol
};
typedef struct SliceInfo *slice_info;
slice_info init_slice(int seg_lim, const mol_info mol);
void load_slice_aux(const bas_info aux, slice_info slice);
void del_slice(slice_info slice);
/**
 * @brief calculate 3c2e integrals and store in buf
 *
 * @param buf the Vector used to store integrals
 * @param slice_id which segment needs to be calculated
 * @param slice
 */
void cal_slice_3c2e(Vector buf, int slice_id, slice_info slice);
/**
 * @brief calculate 3c1e integrals and store in buf
 *
 * @param buf buf the Vector used to store integrals
 * @param g_num the number of grids
 * @param slice_id which segment needs to be calculated
 * @param G_list the list used to store grids
 * @param slice
 */
void cal_slice_3c1e(Vector buf, int g_num, int slice_id, const Vector *G_list, slice_info slice);

void init_xscf_world(int thread_num);
void del_xscf_world();
int get_xscf_threadnum();
void switch_xscf_thread(int blas_xscf);
#endif
