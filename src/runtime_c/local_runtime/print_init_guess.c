#include <stdio.h>
#include <assert.h>
#include "vb/vb.h"

int print_init_guess(vb_info vb_str) {

  int nor=vb_str->nor;
  int nb=vb_str->nb;

  int *ma0=(int*)malloc(sizeof(int)*nor);

  int i,j,ibas;

  memcpy(ma0,vb_str->ma,sizeof(int)*nor);

  for (i=0;i<nor;i++) {
    if (ma0[i]==1) {
      ma0[i]=0;
      for (j=0;j<nb;j++) {
        if (vb_str->nv[i*nb+j]==0)
          break;
        ma0[i]++;
      }
    }
  }

  FILE *fp=fopen(vb_str->xdat_name,"w");
  
  printf("\n---------------Initial Guess---------------\n");
  fprintf(fp,"\n---------------Initial Guess---------------\n");

  // print head of orbitals
  for (i=0;i<nor;i++) {
    printf(" %5d",ma0[i]);
    fprintf(fp," %5d",ma0[i]);
    if ((i+1)%25==0) {
      printf("\n");
      fprintf(fp,"\n");
    }
  }
  if (i%25!=0) {
    printf("\n");
    fprintf(fp,"\n");
  }

  for (i=0;i<nor;i++) {
    for (j=0;j<ma0[i];j++) {
      printf("%13.10f %5d  ",*(vb_str->dv+i*nb+j),*(vb_str->nv+i*nb+j));
      fprintf(fp,"%13.10f %5d  ",*(vb_str->dv+i*nb+j),*(vb_str->nv+i*nb+j));
      if ((j+1)%4==0) {
        printf("\n");
        fprintf(fp,"\n");
      }
    }
    if (j%4!=0) {
      printf("\n");
      fprintf(fp,"\n");
    }
  }
  printf("---------------End of Guess--------------\n\n");
  fprintf(fp,"---------------End of Guess--------------\n\n");

  free(ma0);

  fflush(stdout);

  fclose(fp);

  return 0;
}
