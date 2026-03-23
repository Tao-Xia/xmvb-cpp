#ifndef _VB_H_
#define _VB_H_
#include "fock/fock.h"
#include "fock/dft.h"
#include "mol/xgrids.h"
#include "mol/mol.h"
#include "scf/hf_fwd.h"
#include "inpout/input.h"
#include "parallel/para.h"
#include "vb/int_trans.h"

#define MAX_BATCH 20
#define GEN_TYP 0
#define HAO_TYP 1
#define BDO_TYP 2
#define OEO_TYP 3

#define FRG_ATM 0
#define FRG_SAO 1

#define WFN_STR 0
#define WFN_DET 1

#define VBF_DET 0
#define VBF_PPD 1

#define GUS_AUTO 0
#define GUS_UNIT 1
#define GUS_READ 2
#define GUS_RDCI 3
#define GUS_MO   4
#define GUS_NBO  5

#define TDM_SCF 2
#define RDM_SCF 5
#define HES_RDM_SCF 6

#define INT_CINT 0
#define INT_XINT 1
#define INT_READ 3
#define INT_RI 1
#define INT_COSX 2
#define INT_FCOSX 4
#define INT2E_1D   -1
#define INT2E_NB3   0
#define INT2E_DIR   1

#define INT_TOL  1e-10
#define SVD_TOL  1e-5

#define MAX_STATE 200
#define MAX_PATH  4096

#define AU2DEBYE  2.54175e0

struct DfvbInfo 
{
    int ndft, disp_type, dft_name;
    int *dft_id;
    double vb_frac; 
    double *dft_frac; 
    GRIDS_TYPE grid_type;
    dft_info dft;
};
typedef struct DfvbInfo *dfvb_info; 

struct VbInfo {
  // global ctrl part
  int ir,iw,isc,ier,ida,ide,ifo;
  int boysloc,pmloc,localize,dovb,dovbci,dobfi,dovbscf,dobovb,dovbcis,dovbcisd,dovbcids,dovbpt2,genstr,dogopt,dogradient;
  int orbtyp,frgtyp,wfntyp,vbftyp,orb_with_symm,iguess;
  int iscf,biovb,bio_readcoef,itmax,icovg,vmax;
  int read_points;
  // system part
  int nel,nmul,nao,nae,nb,natom,npb;
  // symmetry part
  int vbsym_d,moor,mnor;
  int vbsym_igrd,vbsym_ind,nop;
  int *nsymc, *nrep;
  int *str_ori;
  // variable part
  int nvar;
  int *indxcx,*indxxc,*ioor,*inor;
  double *cvic;
  // str part
  int nstr,mxbond,ndet,nstr_scf;
  int *ntstr;
  char strclass[200];
  int iomin, iomax; // minimum and maximum ionic order, 0 for cov
  // orbital part
  int nor, nor_scf;
  int *nv;
  int *ma0;
  int *ma;
  int *nvic;
  double *dv;
  // orbital blocks part
  int nblock,block_part_ov;
  int *blocks;
  int *noc_block;
  int *mx_block;
  // libcint part
  int nshell,ngto;
  int *atm;
  int *bas;
  int *basidx;
  double *env;
  double *snorm;
  // int part
  long n2e;
  int dir2e;
  int inttyp;
  double *pcf;
  double *ssf;
  double *hhf;
  double *ggf;
  double *g1d;
  double *xxf;
  double *yyf;
  double *zzf;
  double *ekf;
  int *g2eidx;
  long *nb3num;
  long *nb3idx;
  double *hh,*ss,*gg;
  double *xx,*yy,*zz,*ek;
  // readint part
  char *ele_tag;
  char *bas_tag;
  int nbf,nbr,ncore,nbfido;
  int *limsup;
  int *limlow;
  double *crd;
  double *zan;
  double *cmat;
  double *hfwfn;
  double *ss0;
  // detpair for ISCF=2/5
  int ndetpair;
  int *detpair;
  int *idxpair;
  // vbscf energy part
  int iroot;
  double enuc,eknuc,xnuc,ynuc,znuc;
  double enuc_pn,enuc_pp;
  double energy;
  double vbci_energy;
  double vbci_energy_davidson;
  double vbpt2_energy;
  double energyc;
  double epg,gpg;
  double *col;
  double *hvb,*svb,*hvb_1e;
  // NCOR for BOVB, VBPT2 and VBCI
  int ncor;
  // vbci param cicut

  int cicut;
  int inci;
  int *istr;

  // bovb index part
  int *nbostr;
  int *indxbovb;
  int *inactbo;
  // state average part
  int nsav;
  int idxstate[MAX_STATE];
  double wstate[MAX_STATE];
  double state_energy[MAX_STATE];
  // fixc / group part
  int ngroup;
  int readcoef;
  int *grpidx;
  int *grplist;
  double *fixcol;
  // file name 
  char xdat_name[MAX_PATH];
  char file_name[MAX_PATH];
  char aux_name[MAX_PATH];
  char k_grid_file[MAX_PATH];
  char k_grid_file_final[MAX_PATH];
  // population
  int dopop;
  // wfn
  int dowfn;
  char wfn_name[MAX_PATH];
  // gaapart
  int gaaok;
  int ntgaa[10][10100];
  int ngaa[10];
  int *ntpar;
  // vbntorg
  int norg;
  int ntorg[21];

  // rdm_vbscf parameter
  double *f_p11;
  double *gxx;

  // 3c2e integral
  int ri_pros;
  int cosx_pros;
  double *ij_aux_k;
  int naux;

  // on-fly calculation 
  double *ti;
  double *taux;
  // double *tv;
  double *pa;
  double *pb;
  double *g22;
  double *ga4; // g_tuvw;
  double *go3v; // g_atuv;
  double *ga3b; // g_ptuv;

  // RDM-VBSCF 
  double e11; 

  // normalized factor;
  Vector ssf_norm;

  // RI parameter used for TDM algorithm in derdifb.c 
  Matrix tdm_ij_k;
  Matrix tdm_k_ij;
  int *tdm_nq_i;
  int *tdm_nq_j;
  
  // RI parameter used for batch COSX in derdifb.c
  bool *if_finnish;
  int ipass;
  int iter;
/*   double *old_p11;
  double *old_pa;
  double *old_pb; */
  // Tensor3D pa_batch;
  // Tensor3D pb_batch;

  // DFVB 
  int DoLamDFVB;
  int DohcDFVB; 
  int DoMsDFVB; 
  char DFVBfunc[40]; 
  double dfvb_energy; 
  double *dfvb2_ec; 
  Matrix cad_eff; 
  Matrix ms_dfvb_eff; 
  Matrix ms_dfvb_col_tmp; 
  Tensor3D ms_dfvb_weight; 

  // Simple input 
  int *froz_list; 
  int nfroz; 
  int simple; 

  // Sorting VB Structure
  int sort;

  // CTOL
  double ctol;

  // VBCAD 
  int vbcad; 
  double cad_step_size; 

  // Density Matrix
  double *aden,*bden,*aden0,*bden0;

  // Point Charges
  int npoints;
  double *points;

  intrans_info intrans;
  dfvb_info dfvb_str; 

  // Tested COSX algorithm
  Tensor3D J_K_last;
};

typedef struct VbInfo* vb_info;

int init_vb_param(mol_info mol, inp_info inp_str, vb_info vb_str);
int del_vb_str(vb_info vb_str);

//void expand_str_orb(char split[][1024],int *tmp,int nsplit,int *itmp);
void expand_str_orb(char **split,int *tmp,int nsplit,int *itmp);

//int readinp_vb(mol_info mol,inp_info inp_str, vb_info vb_str,para_info para_str,char *inpname);
vb_info readinp_vb(mol_info mol,inp_info inp_str, para_info para_str,char *inpname);
int vbprep(hf_info hf,inp_info inp_str,vb_info vb_str,para_info para_str, int print_level);

int readstr(char *inpname, int *ntstr, double *col, int *nstr, int nel, int readcoef);
int getstr(char *inpname, int print_level, vb_info vb_str);
int genstr(vb_info vb_str, int print_level);
int getfrg(char *inpname, mol_info mol, vb_info vb_str);
int getorb(char *inpname,vb_info vb_str);
int readorb(char *inpname, vb_info vb_str);
int detect_blocks(vb_info vb_str);
int getvars(vb_info vb_str, int print_level);

int detect_guess(char *inpname); 
int vbguess(hf_info hf, inp_info inp_str, vb_info vb_str, int print_level);
int vb_autoguess(hf_info hf,vb_info vb_str);
int vb_unitguess(vb_info vb_str);
int vb_readguess(inp_info inp_str,vb_info vb_str);
int vb_moguess(hf_info hf, inp_info inp_str,vb_info vb_str);
int cvitra(double *dv, int *nv, int *ma, double *T, int nb, int nor);
int tracvi(double *dv, int *nv, int *ma, double *T, int nb, int nor, int param);

int vbscf(hf_info hf, inp_info inp_str, vb_info vb_str, int gen_guess,int print_level);
int normalize(int nb, int nor, double *dv, int *nv, int *ma, double *ssf);
int gendetpair(vb_info vb_str, int nstr, int nel, int nmul);
int str2det(int *str, int *det, int *ndet,int nel, int nmul, int nd,int wfntyp);
int parput(double *X, double *dv, int *nv, int *ma, int *nsymc, int *indxxc, int vbsym_igrd, int vbsym_ind, int nvar, int nb);
int parget(double *X, double *dv, int *nv, int *ma, int *nsymc, int *indxcx, int *ioor, int *inor, int moor, int mnor, int nvar, int nb);
int invmat(double *A, double *B, int n, int nd,double *det);
int mppinvmat(double *mat, int n, int m);
int orbout(double *dv, int *nv, int *ma, double *col, double energy, double grad, double *ssf, double *svb, int iter, int norb, int nb,int nstr);
int get_ngto(mol_info mol);
//void int_data_trans(mol_info mol, vb_info vb_str);
void update_crd_cint(mol_info mol, int *atm, double *env);
void int_data_trans(mol_info mol, int *atm, int *bas, int *basidx, double *env, int ngto);
int cal_pcf_int(vb_info vb_str,int print_level);
int cal_1eint(vb_info vb_str);
int cal_2eint(vb_info vb_str, int print_level);
int cal_dip_mat(double *xxf, double *yyf, double *zzf, vb_info vb_str);
int cal_overlap_mat(double *ssf, vb_info vb_str);
void readinfo(vb_info vb_str);
void read1e(vb_info vb_str);
void read2e(vb_info vb_str);
void readint(vb_info vb_str);
void readint_head(vb_info vb_str);
//void build_shidx(vb_info vb_str,double *buf,int *shidx, int *nshidx);
void build_shidx(vb_info vb_str,int *shidx, int *nshidx);
int lab(int i, int j);
long lab_long(long i, long j); 
int diag(double *mat, double *u, double *dia,int n, int nd);
int detect_ndb(int *ntstr, int nstr, int nel, int nmul, int wfntyp);
int init_bovb(vb_info vb_str);
void cal_nuc_grad(double *grad, mol_info mol);
void cal_1egrad_cint(Tensor3D ssfg, Tensor3D hhfg, vb_info vb_str,mol_info mol);

int tdm_vbscf(hf_info hf,vb_info vb_str, int print_level);
int vblbfgs(hf_info hf, vb_info vb_str, int print_level, const char *outname);
int derdifb(double *X, double *G, double *energy, double *hsgrad, int *detpair, int *idxpair, int ndetpair, int n, double *sums, hf_info hf,vb_info vb_str);
// int tdm_vbscf(vb_info vb_str, int print_level, const char *out_name);
// int rdm_vbscf(vb_info vb_str, int print_level, const char *out_name);
int rdm_vbscf(hf_info hf, vb_info vb_str, int print_level,const char *out_name);
int tb_vbscf(hf_info hf, vb_info vb_str, int print_level,const char *out_name);
int gradient_rdm(double *energy, double *sums, double *X, double *G, hf_info hf, vb_info vb_str, coulomb_info coulomb, exchange_info exchange);
int gradient_bio(double *energy, double *sums, double *X, double *G, vb_info vb_str);

int hes_rdm_vbscf(hf_info hf, vb_info vb_str, int print_level, const char *out_name);

int compute_vb(hf_info hf, inp_info inp_str, vb_info vb_str, int gen_guess, int print_level, const char *out_name);
void vb_lowrk_init_ri(vb_info vb_str, hf_info hf, mol_info mol);
void vb_lowrk_init_cosx(vb_info vb_str, hf_info hf, mol_info mol);
void vb_lowrk_init(vb_info vb_str, hf_info hf, mol_info mol, inp_info inp_str);

//int boysloc(vb_info vb_str,int print_level, const char *out_name);
int gen_virtual_orb(vb_info vb_str, int print_level, const char *out_name);
int gen_vbci_str(vb_info vb_str, int print_level, const char *out_name);
int vbci(vb_info vb_str, int print_level, const char *out_name);
int vbpt2(vb_info vb_str, int print_level, const char *out_name);
int cal_fock_cint(double *pa,double *pb,double *fa,double *fb,double *ggf,int *g2eidx,long n2e,int nb,int isym);

int clean_vb_modules(vb_info vb_str);
int clean_vbci_modules(vb_info vb_str);
int clean_tb_modules(vb_info vb_str);
int clean_vbpt2_modules(vb_info vb_str);
int clean_rdm_modules(vb_info vb_str);
int eigencalc(double *energy, double *col, double *hvb, double *svb, double enuc, int n, vb_info vb_str);
int calc_den(double *aden, double *bden, double *aden0, double *bden0, vb_info vb_str);
int calc_nos(double *den, double *nos, double *occ, int *nocc, double *ssf, int nb);
int schmidt0(int nb, int nor, int length, int ninao, double *s0, double *trans0, double *trans1, int ndim);
int schmidt1(double *T1, double *T2, double *S, int nb, int nor, int nd);
int xtra1(vb_info vb_str, int nor);
int vbdet(double *hvb, double *svb, int *lvec, int *rvec, int nlstr, int nrstr, vb_info vb_str);
int build_nb3idx(vb_info vb_str);
//void build_shidx(vb_info vb_str,double *buf,int *shidx, int *nshidx);

int dencalc_noci(double *p1,double *p2,double *ov,double *ov0,double *vl,double *vr,double *u,double *vt,double *sv,int nz,int nb,int nel);
int svd_decomp(double *u,double *vt,double *sv,int *nz,double *mat,int *nsl,int *nsr,int nd);

void cal_tdm_grad(double *grad, vb_info vb_str,mol_info mol);

int n_ij(int i, int j, int n);
int localize(vb_info vb_str);

double calc_ecoul_ri(Matrix aden, Matrix bden, vb_info vb_str);
double calc_ecoul_cosx(Matrix aden, Matrix bden, vb_info vb_str, coulomb_info coulomb);
double calc_ecoul(double *aden, double *bden, vb_info vb_str); 
double ddet(double *A, int n, int nd); 
List yanghui(int n, int m); 

void vbcad(vb_info vb_str, int nav, List idxav, Matrix heff, Matrix col, int debug); 
void calc_shalf(Matrix smat, Matrix s_mhalf, Matrix s_phalf, int nstr); 
Tensor3D cal_vbcad_weight(Matrix const svb, Matrix const col, int nstr, int nav); 
int cal_dipole_int(double *dip_nuc, double *xxf, double *yyf, double *zzf, int *atm, int *bas, int *basidx, double *env, int ngto, mol_info mol);
int trans_den(double *aden, double *bden, double *T, double *ss, int *ldet, int *rdet, int nb, int nor, int nel, int nmul);
void build_g1d(vb_info vb_str);
void check_dir2e(int *dir2e, long n2e, int nb, int inttyp);
unsigned long get_total_mem();
char** malloc_string_array(size_t nstring, size_t nlen);
void free_string_array(char **string);
int fvbsec(FILE *fp, char *str);
void getpoints(char *inpname, vb_info vb_str, int print_level);

#endif
