#include <stdio.h>
#include <assert.h>
#include <math.h>
//#include "scf/sad.h"
#include "scf/hf.h"
#include "vb/vb.h"
#include "inpout/output_func.h"

void rhf_readint_(double *energy, double *CA, vb_info vb_str, int *nd);
void urohf_readint_(double *energy, double *CA, double *CB, vb_info vb_str, int *nd);

int vb_autoguess(hf_info hf, vb_info vb_str) {

  int print_level=0;

  double *Ftot,*ovmat;

  int ncore=vb_str->ncore;
  int nact=vb_str->nbf-ncore;
  int nb=vb_str->nb;

  double snorm[nb];

  double *CA,energy;

  if (vb_str->inttyp==INT_READ) {
    CA=(double*)malloc(sizeof(double)*nb*nb);
    if (vb_str->nmul==1)
      rhf_readint_(&energy,CA,vb_str,&nb);
    else
      urohf_readint_(&energy,CA,CA,vb_str,&nb);
//    if (nact!=vb_str->nb) {
//      printf("A previous BPREP with MO frozen is detected.\n");
//      printf("The remained basis function is %d but the remained MO space is %d, not equal\n",vb_str->nb,nact);
//      printf("So GUESS=AUTO cannot be proceeded\n");
//      exit(1);
//    }
    double *pa=(double*)malloc(sizeof(double)*nb*nb);
    double *pb=(double*)malloc(sizeof(double)*nb*nb);
    Ftot=(double*)malloc(sizeof(double)*nb*nb);
    double *fb=(double*)malloc(sizeof(double)*nb*nb);
    int nbeta=(vb_str->nel-vb_str->nmul+1)/2;
    int nalpha=vb_str->nel-nbeta;
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,nb,nb,nalpha,1e0,CA,nb,CA,nb,0e0,pa,nb);
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,nb,nb,nbeta,1e0,CA,nb,CA,nb,0e0,pb,nb);
    cal_fock_cint(pa,pb,Ftot,fb,vb_str->ggf,vb_str->g2eidx,vb_str->n2e,nb,0);
    ovmat=vb_str->ssf;
    memcpy(snorm,vb_str->snorm,sizeof(double)*nb);

    free(fb);
    free(pa);
    free(pb);
    free(CA);
  }
  else {
    do_sad_InitalGuess(hf);

    if (vb_str->nmul==1) {
//      hf->hf_type=RHF_;
      do_rhf(print_level,hf);
    }
    else {
      hf->open_type=ROHF_WORK;
      do_uhf(print_level,hf);
    }
    Ftot=hf->f_matrix[0][0];
    ovmat=hf->s_matrix[0];
    for (int i=0;i<nb;i++)
      snorm[i]=sqrt(hf->s_matrix[i][i]);

  }

  // int nb=hf->mol->bas->msize;
  int i,j,k;
  int nor=vb_str->nor;

  int iblk,jidx,kidx,ior;
  double *ssf=(double*)malloc(nb*nb*sizeof(double));;
  double *fock=(double*)malloc(nb*nb*sizeof(double));;
  double egn_value[nb];
  int nd;
  if (vb_str->dovbci>0)
    nd=nb;
  else
    nd=nor;

  for (i=0;i<vb_str->nblock;i++) {
    ior=*(vb_str->blocks+i*nd);
    iblk=*(vb_str->ma+ior);
    if (iblk==1) {
      iblk=0;
      for (int j=0;j<nb;j++) {
        if (vb_str->nv[ior*nb+j]==0)
          break;
        iblk++;
      }
    }
    for (j=0;j<iblk;j++) {
      jidx=*(vb_str->nv+ior*nb+j)-1;
      for (k=0;k<iblk;k++) {
        kidx=*(vb_str->nv+ior*nb+k)-1;
        *(ssf+j*iblk+k)=ovmat[jidx*nb+kidx];
        *(fock+j*iblk+k)=Ftot[jidx*nb+kidx];
      }
    }
    LAPACKE_dsygv(LAPACK_COL_MAJOR,1,'V','U',iblk,fock,iblk,ssf,iblk,egn_value);
    for (j=0;j<*(vb_str->noc_block+i);j++)
      memcpy(vb_str->dv+(*(vb_str->blocks+i*nd+j))*nb,fock+j*iblk,iblk*sizeof(double));
  }

  free(ssf);
  free(fock);

  if (vb_str->inttyp==INT_READ)
    free(Ftot);

  int ibas;
  for (i=0;i<nor;i++) {
//    for (j=0;j<*(vb_str->ma+i);j++) {
    for (j=0;j<nb;j++) {
      jidx=*(vb_str->nv+i*nb+j)-1;
      if (jidx==-1)
        break;
      (*(vb_str->dv+i*nb+j))*=snorm[jidx];
    }
  }

  return 0;
}
