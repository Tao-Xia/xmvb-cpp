#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "inpout/input.h"
#include "vb/vb.h"

//void readorb_(vb_info vb_str);

int getorb(char *inpname,vb_info vb_str) {

  int i,j;
  int nd;

  if (vb_str->dovbci>0)
    nd=vb_str->nb;
  else
    nd=vb_str->nor;

  vb_str->nv=(int*)malloc(vb_str->nb*nd*sizeof(int));
  vb_str->ma0=(int*)malloc(nd*sizeof(int));
  vb_str->ma=(int*)malloc(nd*sizeof(int));
  vb_str->dv=(double*)malloc(vb_str->nb*nd*sizeof(double));
  vb_str->cvic=(double*)malloc(vb_str->nb*nd*sizeof(double));

  memset(vb_str->nv,0,vb_str->nb*nd*sizeof(int));
  memset(vb_str->ma0,0,nd*sizeof(int));
  memset(vb_str->ma,0,nd*sizeof(int));

  if (vb_str->orbtyp==OEO_TYP) {
    for (i=0;i<vb_str->nor;i++) {
      *(vb_str->ma+i)=vb_str->nb;
      *(vb_str->ma0+i)=vb_str->nb;
      for (j=0;j<vb_str->nb;j++)
        *(vb_str->nv+i*vb_str->nb+j)=j+1;
    }

    if (vb_str->nfroz > 0) {
      for (i = 0; i < vb_str->nfroz; i++) {
        vb_str->ma0[vb_str->froz_list[i]-1] = 0;
      }
    }
  }
  else
    readorb(inpname,vb_str);

  return 0;
}
