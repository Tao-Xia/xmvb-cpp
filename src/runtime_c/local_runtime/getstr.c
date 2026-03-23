#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mol/mol.h"
#include "vb/vb.h"
#include "inpout/input.h"

//void readstr_(vb_info vb_str);
//void genstr_(vb_info vb_str);

int getstr(char *inpname, int print_level, vb_info vb_str) {

  if (vb_str->genstr==1)
    genstr(vb_str,print_level);
//    genstr_(vb_str);
  else {

    if (vb_str->readcoef>0) {
      vb_str->fixcol=(double*)malloc(sizeof(double)*vb_str->nstr);
      memset(vb_str->fixcol,0,sizeof(double)*vb_str->nstr);
    }

    vb_str->ntstr=(int*)malloc(sizeof(int)*vb_str->nel*vb_str->nstr);
//    memset(vb_str->ntstr,0,sizeof(vb_str->ntstr));
    memset(vb_str->ntstr,0,sizeof(int)*vb_str->nel*vb_str->nstr);
//    printf("ready for readstr %d %d %ld %ld\n",vb_str->nel,vb_str->nstr,sizeof(int)*vb_str->nel*vb_str->nstr,sizeof(vb_str->ntstr));
//    readstr_(vb_str);
    readstr(inpname,vb_str->ntstr,vb_str->fixcol,&vb_str->nstr,vb_str->nel,vb_str->readcoef);
    int iorb;
    if (vb_str->nor==0) {
      int i,j;
      for (i=0;i<vb_str->nstr;i++) {
        for (j=0;j<vb_str->nel;j++) {
          iorb=*(vb_str->ntstr+i*vb_str->nel+j);
          if (vb_str->nor<iorb)
            vb_str->nor=iorb;
        }
      }
    }
  }

  // Detect mxbond for further computation.

  vb_str->mxbond=0;
  if (vb_str->wfntyp==WFN_STR) {
    int nbeta=(vb_str->nel-vb_str->nmul+1)/2;
    int mxbond;
    for (int i=0;i<vb_str->nstr;i++) {
      mxbond=0;
      for (int j=0;j<nbeta;j++)
        if (*(vb_str->ntstr+i*vb_str->nel+2*j)!=*(vb_str->ntstr+i*vb_str->nel+2*j+1))
          mxbond++;
      vb_str->mxbond=my_max(vb_str->mxbond,mxbond);
    }
  }

  if (vb_str->ngroup>1) {
    printf("group info:\n");
    for (int i=0;i<vb_str->ngroup;i++) {
      for ( int j=0;j<*(vb_str->grpidx+i);j++) {
        printf("%d %f",*(vb_str->grplist+i*vb_str->nstr+j),*(vb_str->fixcol+*(vb_str->grplist+i*vb_str->nstr+j)));
      }
      printf("\n");
    }
  }

  return 0;
}
