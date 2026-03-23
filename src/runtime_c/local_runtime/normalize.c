#include <stdio.h>
#include <math.h>
#include "vb/vb.h"

int normalize(int nb, int nor, double *dv, int *nv, int *ma, double *ssf) {

  double snorm;
  int mai;
  int i,j,k;
  int ij,ik;

  for (i=0;i<nor;i++) {
    snorm=0.0;
    if (*(ma+i)==1) {
      mai=0;
      while (mai<nb) {
        if (*(nv+i*nb+mai)==0)
          break;
        mai++;
      }
    }
    else
      mai=*(ma+i);
    for (j=0;j<mai;j++) {
      ij=*(nv+i*nb+j)-1;
      for (k=0;k<mai;k++) {
        ik=*(nv+i*nb+k)-1;
        snorm+=(*(dv+i*nb+j))*(*(dv+i*nb+k))*(*(ssf+ij*nb+ik));
      }
    }

    snorm=sqrt(1.0/snorm);

    for (j=0;j<mai;j++)
      *(dv+i*nb+j)*=snorm;
  }

  double *T=(double*)malloc(sizeof(double)*nb*nor);
  cvitra(dv,nv,ma,T,nb,nor);

  for (i=0;i<nor;i++) {
    snorm=0.0;
    if (*(ma+i)==1) {
      mai=0;
      while (mai<nb) {
        if (*(nv+i*nb+mai)==0)
          break;
        mai++;
      }
    }
    else
      mai=*(ma+i);
    for (j=0;j<mai;j++) {
      ij=*(nv+i*nb+j)-1;
      for (k=0;k<mai;k++) {
        ik=*(nv+i*nb+k)-1;
        snorm+=(*(dv+i*nb+j))*(*(dv+i*nb+k))*(*(ssf+ij*nb+ik));
      }
    }
  }


  free(T);

//  cvitra(dv,nv,ma,T,&nb,&nor);

  return 0;
}

void normalize_(int *nb, int *nor, double *dv, int *nv, int *ma, double *ssf) {
  normalize(*nb,*nor,dv,nv,ma,ssf);
}
