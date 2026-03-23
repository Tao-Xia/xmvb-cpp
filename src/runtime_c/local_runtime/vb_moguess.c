#define vb_moguess xmvb_cpp_vb_moguess
#define do_rhf xmvb_cpp_do_rhf
#define do_uhf xmvb_cpp_do_uhf

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "scf/hf.h"
#include "vb/vb.h"
#include "inpout/input.h"
#include "inpout/output_func.h"
#include "cblas.h"

void rhf_readint_(double *energy, double *CA, vb_info vb_str, int *nd);
void urohf_readint_(double *energy, double *CA, double *CB, vb_info vb_str, int *nd);
void read_int_matrix(FILE *fp, double *array, size_t nlen);

void set_trans_submat(double *trans,int start, int len, int iang, int nd) {

  int bas=start;

  switch (iang) {
    case 2:
      /**
           XX YY ZZ XY XZ YZ
        XX  1  0  0  0  0  0
        XY  0  0  0  1  0  0
        XZ  0  0  0  0  1  0
        YY  0  1  0  0  0  0
        YZ  0  0  0  0  0  1
        ZZ  0  0  1  0  0  0
        **/
      trans[bas*nd+bas]=1e0; // XX
      trans[(bas+1)*nd+bas+3]=1e0; // YY
      trans[(bas+2)*nd+bas+5]=1e0; // ZZ
      trans[(bas+3)*nd+bas+1]=1e0; // XY
      trans[(bas+4)*nd+bas+2]=1e0; // XZ
      trans[(bas+5)*nd+bas+4]=1e0; // YZ
      break;
    case 3:
      /**
            XXX YYY ZZZ XYY XXY XXZ XZZ YZZ YYZ XYZ
        XXX   1   0   0   0   0   0   0   0   0   0
        XXY   0   0   0   0   1   0   0   0   0   0
        XXZ   0   0   0   0   0   1   0   0   0   0
        XYY   0   0   0   1   0   0   0   0   0   0
        XYZ   0   0   0   0   0   0   0   0   0   1
        XZZ   0   0   0   0   0   0   1   0   0   0
        YYY   0   1   0   0   0   0   0   0   0   0
        YYZ   0   0   0   0   0   0   0   0   1   0
        YZZ   0   0   0   0   0   0   0   1   0   0
        ZZZ   0   0   1   0   0   0   0   0   0   0
        **/
      trans[bas*nd+bas]=1e0; // XXX
      trans[(bas+1)*nd+bas+6]=1e0; // YYY
      trans[(bas+2)*nd+bas+9]=1e0; // ZZZ
      trans[(bas+3)*nd+bas+3]=1e0; // XYY
      trans[(bas+4)*nd+bas+1]=1e0; // XXY
      trans[(bas+5)*nd+bas+2]=1e0; // XXZ
      trans[(bas+6)*nd+bas+5]=1e0; // XZZ
      trans[(bas+7)*nd+bas+8]=1e0; // YZZ
      trans[(bas+8)*nd+bas+7]=1e0; // YYZ
      trans[(bas+9)*nd+bas+4]=1e0; // XYZ
      break;
  }
  
  return;
}

void rearrange_bf_order(double *T, mol_info mol) {
  int nb=mol->bas->msize;
  int nbas=mol->bas->nbas;
  double *trans=(double*)malloc(sizeof(double)*nb*nb);
  double *T1=(double*)malloc(sizeof(double)*nb*nb);

  memset(trans,0,sizeof(double)*nb*nb);

  int ib=0;
  for (int ibas=0;ibas<nbas;ibas++) {
    int iang=mol->bas->bas[5*ibas+ANGULAR_VAL];
    int di=xint_gtolen(iang);
    if (iang>=2)
      set_trans_submat(trans,ib,di,iang,nb);
    else
      for (int j=0;j<di;j++)
        trans[(j+ib)*nb+j+ib]=1e0;
    ib+=di;
  }

  cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,nb,nb,nb,1e0,trans,nb,T,nb,0e0,T1,nb);
  memcpy(T,T1,sizeof(double)*nb*nb);

  free(T1);
  free(trans);

  return;
}

int vb_moguess(hf_info hf, inp_info inp_str, vb_info vb_str) {
  int print_level=0;

  FILE *fp=fopen(inp_str->inpname,"r");

  double *T;

  if (fvbsec(fp,"$GUS")>0) {
    printf("Error! No $GUS found.\n");
    exit(1);
  }

  int ncore=vb_str->ncore;
  int nact=vb_str->nbf-ncore;
  int nb=vb_str->nb;
  double energy,ssnor;
  if (vb_str->iguess==GUS_MO) {
    if (vb_str->inttyp==INT_READ) {
      T=(double*)malloc(sizeof(double)*nb*nb);
      if (vb_str->nmul==1)
        rhf_readint_(&energy,T,vb_str,&nb);
      else
        urohf_readint_(&energy,T,T,vb_str,&nb);
//      if (nact!=vb_str->nb) {
//        printf("A previous BPREP with MO frozen is detected.\n");
//        printf("The remained basis function is %d but the remained MO space is %d, not equal\n",vb_str->nb,nact);
//        printf("So GUESS=MO cannot be proceeded\n");
//        exit(1);
//      }
//      T=vb_str->hfwfn;
    }
    else {
      do_sad_InitalGuess(hf);

      if (vb_str->nmul==1) {
        do_rhf(print_level,hf);
      }
      else {
        hf->open_type=ROHF_WORK;
        do_uhf(print_level,hf);
      }
      T=hf->c_matrix[0][0];
      for (int i=0;i<nb;i++) {
        ssnor=sqrt(hf->s_matrix[i][i]);
        for (int j=0;j<nb;j++)
          T[j*nb+i]*=ssnor;
      }
    }
  }
  else if (vb_str->iguess==GUS_NBO) {
    T=(double*)malloc(sizeof(double)*nb*nb);
    memset(T,0,sizeof(double)*nb*nb);
    FILE *fp=fopen("orb.nbo","r");
    read_int_matrix(fp,T,nb*nb);
    fclose(fp);
    if (vb_str->inttyp!=INT_READ) {
      rearrange_bf_order(T,hf->mol);
    }
  }

  int nor=vb_str->nor;
//  int nb=vb_str->nb;
  int *moidx=(int*)malloc(sizeof(int)*nor);

  for (int i=0;i<nor;i++)
    moidx[i]=i+1;

  char cline[1024];
  int imo, ivb;
  while(fgets(cline,1024,fp)!=NULL) {
    if (strstr(cline,"$END")!=NULL)
      break;
    sscanf(cline,"%d %d",&ivb,&imo);
    moidx[ivb-1]=imo;
  }

  fclose(fp);

  double coef;
  int idx;
  for (int i=0;i<nor;i++) {
    if (moidx[i]>0)
      coef=1.0e0;
    else if (moidx[i]<0)
      coef=-1.0e0;
    idx = abs(moidx[i]);
    if (vb_str->ma[i]==1) {
      for (int j=0;j<nb;j++) {
        if (vb_str->nv[i*nb+j]==0)
          break;
//        vb_str->dv[i*nb+j]=coef*hf->c_matrix[0][idx-1][vb_str->nv[i*nb+j]-1];
        vb_str->dv[i*nb+j]=coef*T[(idx-1)*vb_str->nb+vb_str->nv[i*nb+j]-1];
      }
    }
    else {
      for (int j=0;j<vb_str->ma[i];j++) {
//        vb_str->dv[i*nb+j]=coef*hf->c_matrix[0][idx-1][vb_str->nv[i*nb+j]-1];
        vb_str->dv[i*nb+j]=coef*T[(idx-1)*vb_str->nb+vb_str->nv[i*nb+j]-1];
      }
    }
  }

  normalize(nb,nor,vb_str->dv,vb_str->nv,vb_str->ma,vb_str->ssf);

  if (vb_str->inttyp==INT_READ || vb_str->iguess==GUS_NBO)
    free(T);
  else
    T=NULL;

  free(moidx);


  return 0;
}
