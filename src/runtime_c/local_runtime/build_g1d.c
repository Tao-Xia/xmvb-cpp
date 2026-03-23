#include <stdio.h>
#include "vb/vb.h"

void build_g1d(vb_info vb_str) {
  int nb=vb_str->nb;
  long nb2=(nb*nb+nb)/2;
  long nb4=(nb2*nb2+nb2)/2;

  if (vb_str->g1d!=NULL)
    free(vb_str->g1d);
  vb_str->g1d=(double*)malloc(sizeof(double)*nb4);

  memset(vb_str->g1d,0,sizeof(double)*nb4);

  double *g1d=vb_str->g1d;
  double *ggf=vb_str->ggf;
  int *g2eidx=vb_str->g2eidx;
  long n2e=vb_str->n2e;
  
  #pragma omp parallel shared(n2e,ggf,g2eidx,g1d)
  {
    #pragma omp for nowait
    for (long i2e=0;i2e<n2e;i2e++) {
      long i=g2eidx[i2e*4+0];
      long j=g2eidx[i2e*4+1];
      long k=g2eidx[i2e*4+2];
      long l=g2eidx[i2e*4+3];
      long ij=lab_long(i,j);
      long kl=lab_long(k,l);
      long idx=lab_long(ij,kl);
      g1d[idx]=ggf[i2e];
    }
  }


  return;
}
