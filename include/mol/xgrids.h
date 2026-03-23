#ifndef _MOL_XGRIDS_H_
#define _MOL_XGRIDS_H_
#include "mol/mol.h"
#define LEB_GRIDS_TYPE 0
#define CUB_GRIDS_TYPE 1
typedef enum
{
    COSXI_GRIDS,
    COSXF_GRIDS,
    XCOARSE_GRIDS,
    COARSE_GRIDS,
    MEDIUM_GRIDS,
    FINE_GRIDS,
    XFINE_GRIDS,
    COSJ_GRIDS,
    COR_GRIDS
} GRIDS_TYPE;
typedef struct grid_t
{
    double x, y, z, w;
    int ia; // the atom this grids belong to
} grid_t;
void eval_ao(double *xyz, int shls, double *psi, double *psix, double *psiy, double *psiz,
             double *psixx, double *psiyy, double *psizz, const atm_info atm, const bas_info bas);
int eval_ao_psi(double *xyz, int shls, double *buf, const atm_info atm, const bas_info bas);

struct GridBlock
{
    int grid_num, bas_num, shell_num;
    List grid_list;
    List shell_list;
    List local2global;
    List global2local;
    List local2global_aux;
    List global2local_aux;
    Matrix psi, psix, psiy, psiz, psixx, psiyy, psizz;
    Matrix psixy, psixz, psiyz;
};
typedef struct GridBlock *grid_block;
struct GridsInfo
{
    double maxR;
    int grid_num;
    grid_t *grids;
    // blocks are get bu oct-tree
    int block_num;
    grid_block blocks;
    Vector bas_extent;
    mol_info mol; // this is a pointer points to MOL
};
typedef struct GridsInfo *grids_info;
void cal_block_bas(int block_index, grids_info grids);
void clear_block_bas(grid_block block);
int del_grid(grids_info grids);
grids_info init_grid(int type, const mol_info mol);
grids_info init_grid_opt(const char *opt_file, const mol_info mol);
grids_info init_cub_grid(const double epsilon, const double delta, const mol_info mol);
void cal_rho(Vector rho, const Matrix d_matrix, const grid_block block);
Matrix cal_overlap_numerical(grids_info grids);
void cal_kinetic(Vector kinetic, const Matrix d_matrix, const grid_block block);
void get_grids();
#endif
