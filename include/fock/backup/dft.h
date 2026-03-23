#ifndef _BUILDER_DFT_H_
#define _BUILDER_DFT_H_
#include "mol/xint.h"
#include "mol/xgrids.h"
#define B3LYP_XC 0.2, MEDIUM_GRIDS, B3LYP_ID, B3LYP_FRAC, 1, false, B3LYP
#define B3LYP_D3_XC 0.2, MEDIUM_GRIDS, B3LYP_ID, B3LYP_FRAC, 1, DFT_D3, B3LYP
#define B3LYP_D3BJ_XC 0.2, MEDIUM_GRIDS, B3LYP_ID, B3LYP_FRAC, 1, DFT_D3_BJ, B3LYP
#define BLYP_XC 0., FINE_GRIDS, BLYP_ID, BLYP_FRAC, 2, false, -1
#define PBE0_XC 0.25, MEDIUM_GRIDS, PBE0_ID, PBE0_FRAC, 1, false, -1
#define PBE_XC 0., FINE_GRIDS, PBE_ID, PBE_FRAC, 2, false, -1
#define max_ele 94
#define max_c 5
#define DFT_D2 2
#define DFT_D3 3
#define DFT_D3M 4
#define DFT_D3_BJ 5
#define DFT_D3M_NJ 6
#define B2LYP 101
#define B3LYP 102
#define B97_D 103
#define BLYP 104
#define B_P 105
#define PBE 106
#define PBE0 107
#define LC_WPBE 108
#define REVPBE 109
#define RPBE 110
#define TPSS 111 
#define TPSS0 112
#define CAM_B3LYP 113
#define M06 114
#define M06X 115
extern int B3LYP_ID[];
extern double B3LYP_FRAC[];
extern int BLYP_ID[];
extern double BLYP_FRAC[];
extern int PBE0_ID[];
extern double PBE0_FRAC[];
extern int PBE_ID[];
extern double PBE_FRAC[];
typedef struct DispInfo *disp_info;
struct DispInfo
{
    double s6, s8, s10;
    double alp6, alp8, alp10;
    double rs6, rs8, rs10;
    Tensor4D cab[3];
    Matrix r0ab;
    List mxc;
    Vector atm_c;
    mol_info mol;
    int func_type;
    int disp_type;
};
disp_info init_disp(const mol_info mol, int func_type, int disp_type);
void del_disp(disp_info disp);
double cal_disp_correct(int start, int end, disp_info disp);
double cal_disp_correct_atomic(int atm_id, int start, int end, disp_info disp);

struct DftInfo
{
    xc_func_type *xc_func;
    xc_func_type *xc_func_pol; // libxc struct used for open shell calculation
    double *xc_frac;
    int func_num;
    int disp_type, dft_type;
    int *type;
    int max_type;
    double hf_frac;
    GRIDS_TYPE grid_type;
    grids_info grids;
    mol_info mol;
    disp_info disp;
};
typedef struct DftInfo *dft_info;
dft_info init_dft(double hf_frac, GRIDS_TYPE gtype, const int dft_id[], const double dft_frac[], int num, int disp_type, int dft_name, const mol_info mol);
void del_dft(dft_info dft);
double cal_dft_xc(const Matrix d_matrix, const Matrix d_matrix_b, Matrix xc_matrix, Matrix xc_matrix_b, dft_info dft);
double cal_dft_disp(int start, int end, dft_info dft);
double cal_dft_exc(const Matrix d_matrix, const Matrix d_matrix_b, dft_info dft);
void cal_exc_grid(Vector rho_xc, const Matrix d_matrix, const Matrix d_matrix_b, dft_info dft);
#endif
