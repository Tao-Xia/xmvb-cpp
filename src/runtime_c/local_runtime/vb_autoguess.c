#define vb_autoguess xmvb_cpp_vb_autoguess
#define do_rhf xmvb_cpp_do_rhf
#define do_uhf xmvb_cpp_do_uhf
#define do_sad_InitalGuess xmvb_cpp_do_sad_InitalGuess

#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "scf/hf.h"
#include "vb/vb.h"
#include "inpout/output_func.h"

int vb_autoguess(hf_info hf, vb_info vb_str) {

  int print_level=0;
  int nb=vb_str->nb;
  double *Ftot,*ovmat;

  double snorm[nb];

  if (vb_str->inttyp==INT_READ) {
    return 1;
  }

  do_sad_InitalGuess(hf);

  if (vb_str->nmul==1) {
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

  int ibas;
  for (i=0;i<nor;i++) {
    for (j=0;j<nb;j++) {
      jidx=*(vb_str->nv+i*nb+j)-1;
      if (jidx==-1)
        break;
      (*(vb_str->dv+i*nb+j))*=snorm[jidx];
    }
  }

  return 0;
}
