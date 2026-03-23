#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vb/vb.h"

int detect_blocks(vb_info vb_str) {
  int i,j,k,l,nblock,ior,nbas,nbas0;
  int nb = vb_str->nb;
  int nor = vb_str->nor;
  int blkinfo[nor];
  int nd;

  if (vb_str->dovbci>0) {
    vb_str->blocks=(int*)malloc(nb*nb*sizeof(int));
    memset(vb_str->blocks,0,nb*nb*sizeof(int));
    nd=nb;
  }
  else {
    vb_str->blocks=(int*)malloc(nor*nor*sizeof(int));
    memset(vb_str->blocks,0,nor*nor*sizeof(int));
    nd=nor;
  }

  vb_str->noc_block=(int*)malloc(nor*sizeof(int));
  vb_str->mx_block=(int*)malloc(nor*sizeof(int));
  memset(vb_str->noc_block,0,nor*sizeof(int));
  memset(vb_str->mx_block,0,nor*sizeof(int));


  vb_str->block_part_ov=0;
  nblock = 1;
  *vb_str->noc_block=1;
  *vb_str->blocks = 0;
  blkinfo[0]=1;

  if (*vb_str->ma==1) {
    nbas=0;
    for (i=0;i<nb;i++) {
      if (*(vb_str->nv+i)==0)
        break;
      nbas++;
    }
    *vb_str->mx_block=nbas;
  }
  else
    *vb_str->mx_block=*vb_str->ma;

  for (i=1;i<nor;i++) {
    blkinfo[i]=0;
    if (vb_str->ma[i]==1) {
      nbas0=0;
      for (j=0;j<nb;j++) {
        if (*(vb_str->nv+i*nb+j)==0)
          break;
        nbas0++;
      }
    }
    else
      nbas0=*(vb_str->ma+i);
    for (j=0;j<nblock;j++) {
      ior=*(vb_str->blocks+j*nd);
      nbas = 0;
      for (k=0;k<nbas0;k++) {
        for (l=0;l<*(vb_str->mx_block+j);l++) {
          if (*(vb_str->nv+i*nb+k)==*(vb_str->nv+ior*nb+l)) {
            nbas++;
            break;
          }
        }
      }
      if (nbas==nbas0 && nbas0==*(vb_str->mx_block+j)) {
        *(vb_str->blocks+j*nd+*(vb_str->noc_block+j)) = i;
        (*(vb_str->noc_block+j))++;
        blkinfo[i]=j+1;
        break;
      }
      else if (nbas>0 && (nbas<*(vb_str->ma+i) || nbas<*(vb_str->ma+ior))) {
        vb_str->block_part_ov=1;
      }
    }
    if (blkinfo[i]==0) {
      *(vb_str->noc_block+nblock)=1;
      *(vb_str->mx_block+nblock)=nbas0;
      *(vb_str->blocks+nblock*nd)=i;
      nblock++;
      blkinfo[i]=nblock;
    }
  }

  vb_str->nblock=nblock;

  return 0;
}
