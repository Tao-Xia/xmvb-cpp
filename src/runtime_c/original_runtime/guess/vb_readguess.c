#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "inpout/input.h"
#include "vb/vb.h"
#include "inpout/output_func.h"

int scfgus2bovb(vb_info vb_str) {
  int nor_scf=vb_str->nor_scf;
  int nor=vb_str->nor;
  int nb=vb_str->nb;

  int i,j,idx;
  for (i=nor_scf;i<nor;i++)
    memcpy(vb_str->dv+i*nb,vb_str->dv+(*(vb_str->indxbovb+i)-1)*nb,nb*sizeof(double));

  return 0;
}

int vb_readguess(inp_info inp_str, vb_info vb_str) {

  FILE *fp=fopen(inp_str->inpname,"r");
  
  char cline[1024];

  char** split=malloc_string_array(2*vb_str->nb*vb_str->nor,1024);

  int nsplit;
  int orb_tmp[2*vb_str->nb*vb_str->nor];
  int *ma, *nv;
  double *dv, *T;

  if (fvbsec(fp,"$GUS")>0) {
    printf("Error! No $GUS found.\n");
    exit(1);
  }

  memset(cline,0,1024);
  memset(orb_tmp,0,2*vb_str->nb*vb_str->nor*sizeof(int));

// get head of $GUS
  int iorb=0;
  int iorb_tmp=0;
  long int cur_pos;
  cur_pos=ftell(fp);
  while(fgets(cline,1024,fp)!=NULL) {
    // printf("the split is %s\n",cline);
    if (strstr(cline,"$END")!=NULL){
      break;
    }
    if (strchr(cline,'.')!=NULL){
      break;
    }
    if (strchr(cline,'#')!=NULL){
      continue;
    }
    nsplit=0;
    split_string(cline,split,1024,&nsplit);
    expand_str_orb(split,orb_tmp,nsplit,&iorb_tmp);
    cur_pos=ftell(fp);

    // if the final site is " ", like 4 4 4 " ";
    if(atoi(split[nsplit-1])==0){
      iorb_tmp=iorb_tmp-1;
    }
  }
  iorb=iorb_tmp;
  if ((vb_str->dobovb>0 && iorb < vb_str->nor_scf) || (vb_str->dobovb==0 && iorb < vb_str->nor)) {
    printf("Error! not enough orbital found in the head of $GUS.\n");
    exit(1);
  }

  int nor;
  if ( iorb >= vb_str->nor)
    nor=vb_str->nor;
  else {
    nor=vb_str->nor_scf;
    printf("\nReading VBSCF guess for BOVB\n");
  }

  int i,j;
  int nb=vb_str->nb;
  nv=(int*)malloc(nor*nb*sizeof(int));
  ma=(int*)malloc(nor*sizeof(int));
  dv=(double*)malloc(nor*nb*sizeof(double));
  T=(double*)malloc(nor*nb*sizeof(double));
  memset(nv,0,nor*nb*sizeof(int));
  memset(ma,0,nor*sizeof(int));
  memset(dv,0,nor*nb*sizeof(double));
  memset(T,0,nor*nb*sizeof(double));

  fseek(fp,cur_pos,SEEK_SET);

  memcpy(ma,orb_tmp,nor*sizeof(int));
  nsplit=0;
  iorb=0;
  
  while(fgets(cline,1024,fp)!=NULL) {
    if (strstr(cline,"$END")!=NULL){
      break;
    }
    if (strchr(cline,'#')!=NULL){
      continue;;
    }
    split_string(cline,split,1024,&nsplit);
    if(atoi(split[nsplit-1])==0){
      nsplit=nsplit-1;
    }
    if (*(ma+iorb) <= nsplit/2) {
      for (i=0;i<*(ma+iorb);i++) {
        *(dv+iorb*nb+i)=atof(split[2*i]);
        *(nv+iorb*nb+i)=atoi(split[2*i+1]);
      }
      iorb++;
      nsplit=0;
    }
    if (iorb>=nor)
      break;
  }
  fclose(fp);
  if (iorb<nor) {
    printf("Error! Not enough guess read.\n");
    exit(1);
  }

  free_string_array(split);

  cvitra(dv,nv,ma,T,nb,nor);

/**

  Possible check for redundency
  guess_check();

  **/

  for (i=0;i<nor;i++) {
    if (vb_str->ma[i]==1) {
      for (j=0;j<nb;j++) {
        if (nv[i*nb+j]==0)
          break;
        vb_str->dv[i*nb+j]=T[i*nb+vb_str->nv[i*nb+j]-1];
      }
    }
    else
      for (j=0;j<vb_str->ma[i];j++)
        vb_str->dv[i*nb+j]=T[i*nb+vb_str->nv[i*nb+j]-1];
  }


/**
  Code to expand VBSCF guess to BOVB
  **/

  if (vb_str->dobovb>0 && (nor >= vb_str->nor_scf && nor < vb_str->nor))
    scfgus2bovb(vb_str);

  free(nv);
  free(dv);
  free(ma);
  free(T);

  return 0;
}
