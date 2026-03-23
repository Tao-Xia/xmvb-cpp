#include <stdio.h>
#include "vb/vb.h"

int vb_unitguess(vb_info vb_str) {
  int nb=vb_str->nb;
  int nor=vb_str->nor;
  int ior,ibas,jor;

  int *basinfo=(int*)malloc(sizeof(int)*nb);

  memset(vb_str->dv,0,sizeof(double)*nb*nor);
  memset(basinfo,0,sizeof(int)*nb);

  for (int i=0;i<vb_str->nblock;i++) {
    ior=*(vb_str->blocks+i*nor);
//    ibas=*(vb_str->ma+ior);
    for (int j=0;j<*(vb_str->noc_block+i);j++) {
      jor=*(vb_str->blocks+i*nor+j);
      for (int k=0;k<vb_str->ma[jor];k++) {
        ibas=vb_str->nv[jor*nb+k]-1;
        if (basinfo[ibas]==0) {
          *(vb_str->dv+jor*nb+k)=1.0;
          basinfo[ibas]=1;
          break;
        }
      }
    }
  }

  free(basinfo);

  return 0;
}
