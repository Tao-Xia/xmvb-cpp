#include <stdio.h>
#include <assert.h>
#include "mol/mol.h"
#include "vb/vb.h"

int print_crd(mol_info mol, vb_info vb_str, const char *out_name) {

  int qoff,i,ele;
//  if (out_name!=NULL) assert(freopen(out_name,"a+",stdout)!=NULL);
  printf(" ATOM      ATOMIC                      COORDINATES (BOHR)\n");
  printf("           CHARGE         X                   Y                   Z\n");

  char **tag;
  double *crd, *zan;
  int natom;
  if (vb_str==NULL) {
    natom=mol->atm->natm;
    tag=(char**)malloc(sizeof(char*)*natom);
    crd=(double*)malloc(sizeof(double)*natom*3);
    zan=(double*)malloc(sizeof(double)*natom);
    for (int i=0;i<natom;i++) {
      qoff=mol->atm->atm[i*2+CENTER_IND];
      zan[i]=(double)mol->atm->atm[i*2+ELEMENT_VAL];
      crd[3*i+0]=mol->atm->value[qoff+0];
      crd[3*i+1]=mol->atm->value[qoff+1];
      crd[3*i+2]=mol->atm->value[qoff+2];
      tag[i]=charge2ele(mol->atm->atm[i*2+ELEMENT_VAL]);
    }
  }
  else {
    natom=vb_str->natom;
    tag=(char**)malloc(sizeof(char*)*natom);
    for (int i=0;i<natom;i++)
      tag[i]=vb_str->ele_tag+i*4;
    zan=vb_str->zan;
    crd=vb_str->crd;
  }

  for (i=0;i<natom;i++)
    printf("  %-2s        %4.1f %15.8f     %15.8f     %15.8f\n",tag[i],zan[i],crd[3*i+0],crd[3*i+1],crd[3*i+2]);

  printf("\n");

  if (vb_str==NULL) {
    for (int i=0;i<natom;i++)
      free(tag[i]);
  }
  else {
    for (int i=0;i<natom;i++)
      tag[i]=NULL;
  }
  free(tag);

  if (vb_str==NULL) {
    free(crd);
    free(zan);
  }
  else {
    zan=NULL;
    crd=NULL;
  }

//  if (out_name!=NULL) {
    fflush(stdout);
//    fclose(stdout);
//  }

  return 0;
}
