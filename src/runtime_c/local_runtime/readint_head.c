#include <stdio.h>
#include "vb/vb.h"

void readint_head(vb_info vb_str) {
  assert("x1e.int" !=NULL);
  assert("INFO" !=NULL);

  FILE *fp1e,*fp2e,*fpinfo;

  fp1e=fopen("x1e.int","r");
  fpinfo=fopen("INFO","r");

  char cline[1024];

  if (fgets(cline,1024,fp1e)!=NULL) {
    sscanf(cline,"%d %d",&vb_str->nb,&vb_str->nel);
  }
  else {
    printf("Error in reading x1e.int\n");
    exit(1);
  }

  if (fgets(cline,1024,fpinfo)!=NULL) {
    sscanf(cline,"%d %d %d",&vb_str->natom,&vb_str->nbf,&vb_str->nbfido);
  }
  else {
    printf("Error in reading INFO\n");
    exit(1);
  }


  fclose(fp1e);
  fclose(fpinfo);
  return;
}
