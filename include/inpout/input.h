#ifndef _INP_H_
#define _INP_H_
#include "mol/xgrids.h"
#include "limits.h"
#include "mol/mol.h" 

struct InpInfo {
  char inpname[1024];
  char mol_name[1024];
  char dftfunc[40];
  char title[1024];
  char basis_name[PATH_MAX];
  char aux_name[PATH_MAX];
  char *j_grid_file;
  char k_grid_file[PATH_MAX];
  char k_grid_file_final[PATH_MAX];
  double hf_frac;
  GRIDS_TYPE grid_type;
  double *dft_frac;
  double ctol;
  int *dft_id;
  int ndft,disp_type,dft_name;
  int unit_bohr;
  int dovb,dodft,ihf_type,iguess;
  int nstr,nor,nmul,nao,nae,iroot,nel,itmax,nb,ncharge,iscf;
  int boysloc,pmloc,dovbscf,dobovb,dovbcis,dovbcisd,dovbcids,dovbpt2,genstr,dogopt,dobfi,fixc,dopop,dowfn;
  int orbtyp,frgtyp,wfntyp,vbftyp;
  int biovb,bio_readcoef,vmax;
  int dir2e;
  int inttyp;
  int ncor;
  int cicut;
  int inci;
  int ngroup;
  int nsav;
  char grpval[1024];
  char strclass[200];
  int idxstate[200];
  double wstate[200];
  char wfn_name[4096];
  int print_level;
  int molden;
  int DoLamDFVB;
  int DohcDFVB; 
  int DoMsDFVB; 
  char DFVBfunc[40]; 
  int *froz_list; 
  int nfroz; 

  int sort;

  int vbcad; 
  double cad_step_size; 

  int int_punch;
  int input_requests_ri_two_electron_mode;

  int read_points;

  // Geometry optimization
  int DoGopt; // 0: Won't perform geometrical optimization, 1: will do
  int Gopt_Type; // Optimize using 0: Cartesian coordinates, 1: internal coordinates (Z-matrix)
  int Grad_Type; // Geometrical gradient 1: analytical, 2: numerical
  int Hess_Type; // Geometrical Hessian 1: guess, 2: analytical, 3: semi-numerical, 4: fully numerical 
  int Max_Gopt_Cycles; // Maximum step for geometrical optimization 
  int nneg; // Number of negative frequencies
  int DoGradient; // calculate gradients 
  int prtGradOnly; // print gradients only

  int doeda;
  int neda;
  int *monomers;
  int doblw;
  int nblw;

};

typedef struct InpInfo* inp_info;

void init_inp_param(inp_info inp_str);
inp_info readinp(char *inpname, char *exefile);
void lower_case(char *str);
void upper_case(char *str);
//void split_string(char *str, char split[][1024], int *nsplit);
void split_string(char *str, char **split, int nlen, int *nsplit);
void getcomputemethod(char *str,char *dftfunc,int *ihf_type,int *dodft,int *dovbscf,int *dobovb,int *dovbcis,int *dovbcisd,int *dovbcids,int *dovbpt2,int *boysloc, int *pmloc);
int getcom(inp_info inp_str, FILE *fp);
int parse_dft_param(inp_info inp_str);
void parse_pople_basis(char *basis);
void getbasisaux(inp_info inp_str);
int checkkeywords(inp_info inp_str);
int parse_sav_info(char *kwd, char *kval, int *idxstate, double *wsate, int *nsav);
List parse_frozen_vb_orbital(char *kval, int *nfroz); 
#endif
