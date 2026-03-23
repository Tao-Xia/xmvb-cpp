#include "scf/hf.h"
#include "scf/blw.h"
#include "vb/vb.h"
#include "eda/eda.h"
int print_head(int argc, char *argv[],int ncores);
int print_crd(mol_info mol, vb_info vb_str, const char *out_name);
int print_bas_vb(mol_info mol, const char *out_name);
int print_str_info(int *ntstr, int nstr, int nel, int nmul, int wfntyp, const char *filename);
int print_str_coef_weight(double *col, int *sort_list, int *ntstr, int nstr, int nel, int nmul,int ndb, int wfntyp);
int print_orb_info(double *cvic, int *nv, int *ma, int nb, int nor, const char *out_name);
int print_blk_info(int *noc_block, int *blocks, int nblock, int nb, int nor, int dovbci, const char *out_name);
int print_var_info(vb_info vb_str, const char *out_name);
int print_sav_info(int *idxstate, double *wstate, int nsav,const char *out_name);
int print_grp_info(int *idxgrp, int *grplist, double *coef, int ngroup, int nstr, const char *out_name);
int print_final_result(hf_info hf, eda_info eda, blw_info blw, vb_info vb_str, inp_info inp_str, char *inpname);
int print_init_guess(vb_info vb_str);
int print_orb_file(double *dv, int *nv, int *ma, int nb, int nor, char *file_name);
int print_virial(double *aden, double *bden, double *crd, double *zan, double *hhf, double *ekf, double energy, double nuc_rep, double eknuc, int nb, int natom, int inttyp,const char *out_name);
int print_wfn_file(double *no, double *occ, double *eigen, int nocc, double total_energy, double ek, int nb, mol_info mol, char *file_name);
int print_blw_file(blw_info blw, char *inpname);
int print_nos(double *nos, double *occ, int nocc, int nb, mol_info mol, vb_info vb_str);
int print_pop_analysis(double *aden, double *bden,double *ssf,mol_info mol,int nb,vb_info vb_str);
int print_mo_file(double *vec, int nrow, int ncol, char *file);
int print_molden_file(double *dv, int *nv, int *ma, int nb, int nor, mol_info mol, char *file_name); 
int print_xdat_file(double *Tobf, double *T, vb_info vb_str);
void print_moment(double *aden, double *bden, vb_info vb_str);
int print_vbscf(mol_info mol, vb_info vb_str,int molden);
int print_vbci_fund(vb_info vb_str);

int matrec(double *mat, int n, int m);
int fmatrec(double *mat, int n, int m,FILE *fp);
int matrec_row(double *mat, int n, int m);
void str_det_expand(double *col_det, double *col, int *ntdet, int *ndet, vb_info vb_str);
void print_vb_comp_info(vb_info vb_str);
