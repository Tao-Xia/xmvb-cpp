#include <stdio.h>
#include <string.h>
#include "vb/vb.h"

int init_bovb(vb_info vb_str) {

  if (vb_str->nstr==1) {
    vb_str->nor_scf=vb_str->nor;
    return 0;
  }

  int nor=vb_str->nor;
  int nstr=vb_str->nstr;
  int nb=vb_str->nb;
  int nel=vb_str->nel;

  vb_str->nbostr=(int*)malloc(nor*nstr*sizeof(int));
  vb_str->indxbovb=(int*)malloc(nor*nstr*sizeof(int));

  int *iocc=(int*)malloc(nor*nstr*sizeof(int));
  int *ix=(int*)malloc(nor*nstr*sizeof(int));

  double *cvic=(double*)malloc(nb*nor*sizeof(double));
  int *nv=(int*)malloc(nb*nor*sizeof(int));
  int *ma=(int*)malloc(nor*sizeof(int));
  int *ma0=(int*)malloc(nor*sizeof(int));

  memcpy(cvic,vb_str->cvic,nb*nor*sizeof(double));
  memcpy(nv,vb_str->nv,nb*nor*sizeof(int));
  memcpy(ma,vb_str->ma,nor*sizeof(int));
  memcpy(ma0,vb_str->ma0,nor*sizeof(int));

  free(vb_str->cvic);
  free(vb_str->dv);
  free(vb_str->nv);
  free(vb_str->ma);
  free(vb_str->ma0);

  vb_str->cvic=(double*)malloc(nb*nor*nstr*sizeof(double));
  vb_str->dv=(double*)malloc(nb*nor*nstr*sizeof(double));
  vb_str->nv=(int*)malloc(nb*nor*nstr*sizeof(int));
  vb_str->ma=(int*)malloc(nor*nstr*sizeof(int));
  vb_str->ma0=(int*)malloc(nor*nstr*sizeof(int));

  memset(vb_str->cvic,0,nb*nor*nstr*sizeof(double));
  memset(vb_str->dv,0,nb*nor*nstr*sizeof(double));
  memset(vb_str->nv,0,nb*nor*nstr*sizeof(int));
  memset(vb_str->ma,0,nor*nstr*sizeof(int));
  memset(vb_str->ma0,0,nor*nstr*sizeof(int));

  memcpy(vb_str->cvic,cvic,nb*nor*sizeof(double));
  memcpy(vb_str->nv,nv,nb*nor*sizeof(int));
  memcpy(vb_str->ma,ma,nor*sizeof(int));
  memcpy(vb_str->ma0,ma0,nor*sizeof(int));

  free(cvic);
  free(nv);
  free(ma);
  free(ma0);

  memset(vb_str->nbostr,0,nor*nstr*sizeof(int));

  vb_str->nor_scf=vb_str->nor;

  int i,j,k,i1;
  int ibostart=2*vb_str->ncor;
  int inoworb;
  for (i=0;i<nor;i++)
    *(vb_str->indxbovb+i)=i+1;

  for (i=1;i<nstr;i++) {
    memset(iocc,0,nor*nstr*sizeof(int));
    memset(ix,0,nor*nstr*sizeof(int));
    for (j=0;j<=i-1;j++)
      for (k=0;k<nel;k++)
        *(ix+(*(vb_str->ntstr+j*nel+k))-1)=1;
    for (j=ibostart;j<nel;j++) {
      if (*(iocc+j)!=0)
        continue;
      if (*(ix+(*(vb_str->ntstr+i*nel+j))-1)==0)
        continue;
      for (k=nor;k<nor*nstr;k++)
        if (*(ix+k)==0)
          break;
      if (k==nor*nstr) {
        printf("Sorry, too many orbitals\n");
        exit(1);
      }
      inoworb=*(vb_str->ntstr+i*nel+j);
      for (i1=ibostart;i1<nel;i1++)
        if (*(vb_str->ntstr+i*nel+i1)==inoworb) {
          *(iocc+i1)=1;
          *(vb_str->ntstr+i*nel+i1)=k+1;
        }
      *(ix+k)=1;
      *(vb_str->ma+k)=*(vb_str->ma+inoworb-1);
      *(vb_str->ma0+k)=*(vb_str->ma0+inoworb-1);
      *(vb_str->indxbovb+k)=inoworb;
      memcpy(vb_str->nv+k*nb,vb_str->nv+(inoworb-1)*nb,sizeof(int)*nb);
      for (i1=0;i1<vb_str->ma0[k];i1++)
        vb_str->cvic[k*nb+i1]=(double)k+i1*0.001;
//      if (vb_str->ma[k]>1) {
//        for (i1=0;i1<*(vb_str->ma+k);i1++) {
//          *(vb_str->nv+k*nb+i1)=*(vb_str->nv+(inoworb-1)*nb+i1);
//          *(vb_str->cvic+k*nb+i1)=(double)k+i1*0.001;
//        }
//      }
//      else if (vb_str->ma[k]==1) {
//        for (i1=0;i1<nb;i1++) {
//          if (vb_str->nv[(inoworb-1)*nb+i1]==0) {
//            vb_str->ma0[k]=i1+1;
//            break;
//          }
//          *(vb_str->nv+k*nb+i1)=*(vb_str->nv+(inoworb-1)*nb+i1);
//          *(vb_str->cvic+k*nb+i1)=(double)k+i1*0.001;
//        }
//      }
//      memcpy(vb_str->nv+k*nb,vb_str->nv+inoworb*nb,nb*sizeof(int));
    }
  }

  memset(ix,0,nor*nstr*sizeof(int));
  for (i=0;i<nstr;i++)
    for (j=0;j<nel;j++)
      *(ix+(*(vb_str->ntstr+i*nel+j))-1)=1;
  for (i=nor*nstr-1;i>=0;i--) {
    if (*(ix+i)!=0)
      break;
  }
  nor=i+1;
  vb_str->nor=i+1;
  for (i=0;i<nstr;i++)
    for (j=0;j<nel;j++) {
      *(vb_str->nbostr+(*(vb_str->ntstr+i*nel+j))-1)=i+1;
//  70  Ntstr0(J+Nelina,I)=Ntstr(J,I)
    }

  printf("Breathing Orbitals:\n");
  for (i=vb_str->nor_scf;i<nor;i++)
    if (*(vb_str->nbostr+i)!=*(vb_str->nbostr+i-1))
      printf(" Structure %3d: %3d -> %3d\n",*(vb_str->nbostr+i),*(vb_str->indxbovb+i),i+1);
    else
      printf("                %3d -> %3d\n",*(vb_str->indxbovb+i),i+1);

  free(iocc);
  free(ix);

  fflush(stdout);

  return 0;
}
