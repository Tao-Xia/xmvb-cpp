#include <stdio.h>
#include <stdlib.h>
#include "vb/vb.h"

int getvars(vb_info vb_str, int print_level) {

//  #ifdef MPI
//    if (myproc==0) {
//  #endif


  int nb=vb_str->nb;
  int nor=vb_str->nor;
  int i,j;

//  if (vb_str->vbsym_d!=0) {
  if (vb_str->indxcx==NULL)
    vb_str->indxcx=(int*)malloc(nb*nor*sizeof(int));
  if (vb_str->inor==NULL)
    vb_str->inor=(int*)malloc(nor*sizeof(int));
  if (vb_str->ioor==NULL)
    vb_str->ioor=(int*)malloc(nor*sizeof(int));
//  }

  memset(vb_str->inor,0,nor*sizeof(int));
  memset(vb_str->ioor,0,nor*sizeof(int));
  memset(vb_str->indxcx,0,nb*nor*sizeof(int));

  int kk=0,ll=0;
  for (i=0;i<nor;i++) {
//    if (vb_str->ma0[i]==0 || vb_str->ma0[i]==1) {
    if (vb_str->ma0[i]==0) {
      *(vb_str->inor+ll)=i;
      ll++;
    }
    else {
      *(vb_str->ioor+kk)=i;
      kk++;
    }
  }

  vb_str->mnor=ll;
  vb_str->moor=kk;
  int nvar=0;

  for (i=0;i<kk;i++)
    nvar+=*(vb_str->ma0+(*(vb_str->ioor+i)));

  if (vb_str->nsymc!=NULL)
    free(vb_str->nsymc);
  if (vb_str->nrep!=NULL)
    free(vb_str->nrep);
  if (vb_str->indxxc!=NULL)
    free(vb_str->indxxc);

  vb_str->nsymc=(int*)malloc(nvar*sizeof(int));
  vb_str->nrep=(int*)malloc(nvar*sizeof(int));
  vb_str->indxxc=(int*)malloc(2*nvar*sizeof(int));
  double *ecof=(double*)malloc(nvar*sizeof(double));

  memset(vb_str->nsymc,0,nvar*sizeof(int));
  memset(vb_str->nrep,0,nvar*sizeof(int));
  memset(vb_str->indxxc,0,2*nvar*sizeof(int));
  memset(ecof,0,nvar*sizeof(double));

  vb_str->nvar=0;
  for (i=0;i<kk;i++)
    for (j=0;j<*(vb_str->ma0+(*(vb_str->ioor+i)));j++) {
      *(ecof+vb_str->nvar)=*(vb_str->cvic+(*(vb_str->ioor+i))*nb+j);
      *(vb_str->indxxc+2*vb_str->nvar)=j;
      *(vb_str->indxxc+2*(vb_str->nvar)+1)=*(vb_str->ioor+i);
      *(vb_str->indxcx+(*(vb_str->ioor+i))*nb+j)=vb_str->nvar;
      vb_str->nvar++;
    }

  if (vb_str->nvar!=nvar) {
    printf("Error! Number of variables are incoordinate.\n");
    printf("%d %d\n",nvar,vb_str->nvar);
    exit(1);
  }

  int mv=nvar;
  for (i=0;i<vb_str->nvar;i++) {
    if (*(vb_str->nsymc+i)!=0)
      continue;
    for (j=i+1;j<vb_str->nvar;j++) {
      if (*(vb_str->nsymc+j)!=0)
        continue;
      double c1=*(ecof+i)+(*(ecof+j));
      double c2=*(ecof+i)-(*(ecof+j));
      if (c1>-1e-6 && c1<1e-6) {
        *(vb_str->nsymc+j)=-1*i;
        mv=mv-1;
      }
      else if (c2>-1e-6 && c2<1e-6) {
        *(vb_str->nsymc+j)=i;
        mv=mv-1;
      }
    }
  }

  if (print_level>0) {
    printf("\n");
    printf(" Number of variables for VBSCF/BOVB : %d\n",vb_str->nvar);
    printf("\n");
  }

//  if (vb_str->vbsym_d!=0) {
//    if (vb_str->indxcx!=NULL)
//      free(vb_str->indxcx);
//    if (vb_str->inor!=NULL)
//      free(vb_str->inor);
//    if (vb_str->ioor!=NULL)
//      free(vb_str->ioor);
//    free(ecof);
//    return 0;
//  }

//  #ifdef MPI
//    }
//  #endif

  free(ecof);

  return 0;
}
