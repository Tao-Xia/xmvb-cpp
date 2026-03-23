#define get_ngto xmvb_cpp_get_ngto
#define int_data_trans xmvb_cpp_int_data_trans

#include <stdio.h>
#include "mol/mol.h"
#include "vb/vb.h"
#include "runtime_c/local_runtime/cint_compat.h"

int get_ngto(mol_info mol) {
  int ngto=0;
//  vb_str->nshell=mol->bas->nbas;
  for (int ibas=0;ibas<mol->bas->nbas;ibas++)
    ngto+=mol->bas->bas[NPRIM_VAL+5*ibas];

  return ngto;
}

void update_crd_cint(mol_info mol, int *atm, double *env) {
  int qoff1,offset;
  for (int i=0;i<mol->atm->natm;i++) {
    qoff1=mol->atm->atm[CENTER_IND+2*i];
    offset=atm[i*ATM_SLOTS+PTR_COORD];
    memcpy(env+offset,mol->atm->value+qoff1,3*sizeof(double));
  }

  return;

}

//void int_data_trans(mol_info mol, vb_info vb_str) {
void int_data_trans(mol_info mol, int *atm, int *bas, int *basidx, double *env, int ngto) {
  int ibas;
//  int ngto=0;
//  vb_str->nshell=mol->bas->nbas;
//  for (ibas=0;ibas<mol->bas->nbas;ibas++)
//    ngto+=mol->bas->bas[NPRIM_VAL+5*ibas];

//  vb_str->env=(double*)malloc((ngto*2+mol->atm->natm*3+PTR_ENV_START)*sizeof(double));
//  vb_str->atm=(int*)malloc(ATM_SLOTS*mol->atm->natm*sizeof(int));
//  vb_str->bas=(int*)malloc(BAS_SLOTS*mol->bas->nbas*sizeof(int));
//  vb_str->basidx=(int*)malloc(2*mol->bas->nbas*sizeof(int));

  memset(env,0,(ngto*2+mol->atm->natm*3+PTR_ENV_START)*sizeof(double));
  memset(atm,0,ATM_SLOTS*mol->atm->natm*sizeof(int));
  memset(bas,0,BAS_SLOTS*mol->bas->nbas*sizeof(int));
  memset(basidx,0,2*mol->bas->nbas*sizeof(int));

  int offset=PTR_ENV_START;

  int i,j,qoff1,qoff2;
  for (i=0;i<mol->atm->natm;i++) {
    qoff1=mol->atm->atm[CENTER_IND+2*i];
    *(atm+i*ATM_SLOTS+PTR_COORD)=offset;
    *(atm+i*ATM_SLOTS+CHARGE_OF)=mol->atm->atm[ELEMENT_VAL+2*i];
    memcpy(env+offset,mol->atm->value+qoff1,3*sizeof(double));
//    for (j=0;j<3;j++)
//      *(vb_str->env+offset+j)=mol->value[qoff+j];
    offset+=3;
  }

  int nprim;
  *(basidx)=0;
  for (ibas=0;ibas<mol->bas->nbas;ibas++) {
    nprim=mol->bas->bas[5*ibas+NPRIM_VAL];
    *(bas+ibas*BAS_SLOTS+ANG_OF)=mol->bas->bas[5*ibas+ANGULAR_VAL];
    *(bas+ibas*BAS_SLOTS+ATOM_OF)=mol->bas->bas[ATOM_IND+5*ibas];
    *(bas+ibas*BAS_SLOTS+NPRIM_OF)=nprim;
    *(bas+ibas*BAS_SLOTS+NCTR_OF)=1.0;
    *(bas+ibas*BAS_SLOTS+PTR_EXP)=offset;
    *(bas+ibas*BAS_SLOTS+PTR_COEFF)=offset+nprim;
    qoff1=mol->bas->bas[EXP_IND+5*ibas];
    qoff2=mol->bas->bas[COEFF_IND+5*ibas];
//    memcpy(vb_str->env+offset,mol->value+qoff1,nprim*sizeof(double));
//    memcpy(vb_str->env+offset+nprim,mol->value+qoff2,nprim*sizeof(double));
    for (i=0;i<nprim;i++) {
      *(env+offset+i)=mol->bas->value[qoff1+i];
//      *(vb_str->env+offset+i+nprim)=mol->value[qoff2+i]*CINTgto_norm(mol->bas[5*ibas+ANGULAR_VAL],mol->value[qoff1+i]);
      *(env+offset+i+nprim)=mol->bas->value[qoff2+i];
    }
    offset+=2*nprim;
    *(basidx+2*ibas+1)=CINTcgto_cart(ibas,bas);
  }

  for (i=1;i<mol->bas->nbas;i++)
    *(basidx+2*i)=*(basidx+2*i-2)+(*(basidx+2*i-1));

  return;
}

void int_data_trans_py(mol_info mol, vb_info vb_str) {
  vb_str->ngto=get_ngto(mol);
  vb_str->nshell=mol->bas->nbas;
  vb_str->env=(double*)malloc((vb_str->ngto*2+mol->atm->natm*3+PTR_ENV_START)*sizeof(double));
  vb_str->atm=(int*)malloc(ATM_SLOTS*mol->atm->natm*sizeof(int));
  vb_str->bas=(int*)malloc(BAS_SLOTS*mol->bas->nbas*sizeof(int));
  vb_str->basidx=(int*)malloc(2*mol->bas->nbas*sizeof(int));
  int_data_trans(mol,vb_str->atm,vb_str->bas,vb_str->basidx,vb_str->env,vb_str->ngto);
  return;
}
