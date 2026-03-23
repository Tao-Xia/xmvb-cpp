#include <string.h>

int cvitra(double *dv, int *nv, int *ma, double *T, int nb, int nor) {
  int i,j;
  memset(T,0,nb*nor*sizeof(double));
  for (i=0;i<nor;i++) {
    if (ma[i]>1) {
      for (j=0;j<*(ma+i);j++)
        *(T+i*nb+*(nv+i*nb+j)-1) = *(dv+i*nb+j);
      }
    else {
      for (j=0;j<nb;j++) {
        if (nv[i*nb+j]==0)
          break;
        *(T+i*nb+*(nv+i*nb+j)-1) = *(dv+i*nb+j);
      }
    }
  }
  return 0;
}

void cvitra_(double *dv, int *nv, int *ma, double *T, int *nb, int *nor) {
  cvitra(dv,nv,ma,T,*nb,*nor);
}
