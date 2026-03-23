#include "vb/vb.h"
#include "mol/mol.h"

#define ind_aux(i, j) ind_aux[(i)*3 + j]

dfvb_info parse_lam_dfvb_param(char *FuncName, double lam, const mol_info mol); 
void lam_dfvb(vb_info vb_str, const mol_info mol, int print_level); 
void hc_dfvb(vb_info vb_str, mol_info mol); 
void ms_dfvb(vb_info vb_str, const mol_info mol); 
void cal_det_den(double *aden, double *bden, double *ss, Matrix T, List det, int nalpha, int nbeta, int nb, int nor); 
double cal_det_overlap(double *ss, List det1, List det2, int nalpha, int nbeta, int nor); 
List ind_auxiliary(int iomax1, int ntype, List cnm, vb_info vb_str); 
void gendet_dfvb2(int nalpha, int nbeta, int nae, int nao, int num_hc, int num_cas, int iomin, int iomax, int *cnm, int *ind_cas, int *aorb, int *borb); 
void gentuple(int num, int inact, int nu, int *a, int *na); 
int complementary_set(int *a, int numa, int *b, int numb, int *c, int nao); 
double cal_det_k_ri(double* aden, double* bden, vb_info vb_str);
double cal_det_k_cosx(double *aden, double *bden, vb_info vb_str,exchange_info exchange, int nb);
double cal_det_k(double *aden, double *bden, double *ggf, int *g2eidx, int nb, int n2e); 
int parity(int *a, int n); 
double cal_det(double *a, int n); 
double cal_dfvb_exc(Matrix aden, Matrix bden, dft_info dft, mol_info mol, int nb); 
List dfvb2_str2det(List ntstr, int nbeta, int nalpha, int nstr, vb_info vb_str, int *ndet); 
void getdet(int naea, int naeb, int nel, int nao, int num_hc, int num_cas, List cnm, List ind_cas, List aorb, List borb, List ntdet); 
// int cal_state_den(Matrix aden, Matrix bden, Matrix ss, Matrix T, Vector snor, Vector col1, Vector col2, vb_info vb_str); 
int cal_state_den(Matrix aden, Matrix bden, Vector col, vb_info vb_str); 
double cal_unpaird_density(Matrix aden, Matrix bden, vb_info vb_str, int if_nso); 
void del_dfvb_str(dfvb_info dfvb_str); 
int cal_state_den_tdm(Matrix aden, Matrix bden, Vector col, vb_info vb_str); 
