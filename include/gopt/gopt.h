#ifndef _GOPT_H_
#define _GOPT_H_

#include "inpout/input.h"
#include "scf/hf.h"
#include "scf/blw.h"
#include "eda/eda.h"
#include "vb/vb.h"

#define GOPT_NONE 0
#define GOPT_CART 1
#define GOPT_ZMT  2

#define GGRAD_NONE       0
#define GGRAD_ANALYTICAL 1
#define GGRAD_NUMERICAL  2

#define GHESS_NONE            0
#define GHESS_GUESS           1
#define GHESS_ANALYTICAL      2
#define GHESS_SEMI_NUMERICAL  3
#define GHESS_FULLY_NUMERICAL 4

#define ZMT_Idx(Gvar_Index,Atom_Index) ZMT_Idx[(Gvar_Index)*4+Atom_Index]

struct GState
{
  double E;
  double dE;
  double dE_predict;
  double dE_interpolated;
  Vector g;
  Vector dx;
  Vector dx_interpolated;
  Vector X;
  Vector dcrd;
  Matrix crd;
  Matrix H;
  int ngvar;
  int natom;
};
typedef struct GState *state_info;
struct GoptInfo
{
  state_info current;
  state_info next;
  state_info best;
  state_info previous;
  state_info interpolated;
  double gnorm_thresh;
  double gmax_thresh;
  double *weights;
  int Gopt_Type; // Optimize using 1: Cartesian coordinates, 2: internal coordinates (Z-matrix)
  int Grad_Type; // Geometrical gradient 1: analytical, 2: numerical
  int Hess_Type; // Geometrical Hessian 1: guess, 2: analytical, 3: semi-numerical, 4: fully numerical 
  int Max_Gopt_Cycles; // Maximum step for geometrical optimization 
  int nneg; // Number of negative frequencies

  int Ngvar; // Number of geometrical variables to be optimized 

  int ts_order; // Order of Transition State. 0 means global minima, 1 means TS, 2 means second order TS, and so on.

  Vector dCrd; // Change in Cartesian coordinates 
  Vector Dx; // Change in coordinates used in optimization 

  Matrix Hess_for_Gopt; // Hessian used for optimizer 
  Matrix Hess_inv; // Inverse of Hessian 
  Vector Grad_for_Gopt; // Gradient used for optimizer 

  Matrix Hess_in_Cart; // Hessian in Cartesian coordinates 
  Vector Grad_in_Cart; // Gradient in Cartesian coordinates 

  Matrix old_Hess_for_Gopt; // Hessian used for optimizer from last step 
  Vector old_Grad_for_Gopt; // Gradient used for optimizer from last step 

  int Ndist, Nangle, Ndihed, NZMT; // Number of bongs, angles, dihedral angles, Z-matrix variables
  List ZMT_Idx; // The index matrix containing the corresponding atoms of each Z-matrix variables
  Vector ZMT_Value; // Values of Z-matrix variables
  Matrix Distance_Mat; // Distance matrix of all atoms 

  Matrix Cart2ZMT; // Transformation matrix from Cartesian coordinates to internal coordinates 
  Matrix ZMT2Cart; // Transformation matrix from internal coordinates to Cartesian coordinates 
  Matrix Hess_Trans; // Transformation matrix for Hessian 

  Matrix OrbSave; // Orbital coefficients from last step 
  Matrix CSFSave; // CSFs (or VB structures) coefficients from last step 
};
typedef struct GoptInfo *gopt_info;

//gopt_info get_gopt_info(int Natom, double *Crd, int msize, inp_info inp_str); 
gopt_info get_gopt_info(mol_info mol,inp_info inp_str); 
void get_zmt_idx(mol_info mol,gopt_info gopt);
int Get_ZMT_Idx(double *Crd, int Natom, int *Ndist, int *Nangle, int *Ndihed, int *ZMT_Idx); 
void Vector_Cross_Product(double *v1, double *v2, double *p); 
void calc_distance_matrix(Matrix Dist_Mat,double *Crd,int Natom);
void Gen_ZMT(Vector value, int *idx, Matrix C2Z, Matrix Z2C, int Ndist, int Nangle, int Ndihed, int NZMT,mol_info mol); 
void Gen_Dist(Vector value, int *idx, Matrix C2Z, double *Crd, int Ndist); 
void Gen_Angle(Vector value, int *idx, Matrix C2Z, double *Crd, int Ndist, int Nangle); 
void Gen_Dihed(Vector value, int *idx, Matrix C2Z, double *Crd, int Ndist, int Nangle, int Ndihed); 

void ghess_guess(gopt_info gopt,mol_info mol);
void ghess_full_numerical(gopt_info gopt, hf_info hf, eda_info eda, blw_info blw, vb_info vb_str,inp_info inp_str);
void ghess_semi_numerical(gopt_info gopt, hf_info hf, eda_info eda, blw_info blw, vb_info vb_str,inp_info inp_str,int gen_guess, int print_level);
void gopt_grad(gopt_info gopt, hf_info hf, eda_info eda, blw_info blw, vb_info vb_str,inp_info inp_str);
void gopt_grad_analytical(gopt_info gopt, hf_info hf, eda_info eda, blw_info blw, vb_info vb_str,inp_info inp_str);
void gopt_grad_numerical(gopt_info gopt, hf_info hf, eda_info eda, blw_info blw, vb_info vb_str,inp_info inp_str);
void ghess_update(gopt_info gopt);

void gopt_solver(gopt_info gopt, double de, double *de_pre, double *dxmaxt, double *radius,double *ratio,int isearch,int nneg, Matrix gediis_vec, int ngediis);
void unitcom(double *crd, double *comc, double *rm, double *totalmass,int natom);
void hessproj(double *hess_cart,double *comc, double *rm, double totalmass,int natom);
void build_Pmat(Matrix P, double *rm, double *comc, double totalmass, int natom, int ngvar);
void del_gopt_info(gopt_info gopt);

void copy_gstate(state_info dist, state_info source);

void cal_vb_grad(double *grad,vb_info vb_str,gopt_info gopt, mol_info mol);

void print_only_grad(hf_info hf,eda_info eda, blw_info blw, vb_info vb_str,inp_info inp_str, mol_info mol); 

#endif 
