#include <stdio.h>
#include <string.h>
#include "mol/xint.h"
#include "mol/mol.h"
#include "vb/vb.h"

int frg2orb(int *frags, int *frgidx, int nfrg, int *nv, int *ma, int *ma0, int nor, int nb) {
//  int bas_temp[nb];
  int *bas_temp=(int*)malloc(sizeof(int)*nb);
  for (int i=0;i<nor;i++) {
    int ibas=0;
    int jfrag=ma0[i];
    if (jfrag>0) {
      for (int j=0;j<jfrag;j++) {
        int ifrg=*(nv+i*nb+j);
        for (int k=0;k<*(frgidx+ifrg-1);k++) {
          bas_temp[ibas+k]=(*(frags+(ifrg-1)*nb+k))+1;
        }
        ibas+=*(frgidx+ifrg-1);
      }
    }
    else {
      for (int j=0;j<nb;j++) {
        int ifrg=nv[i*nb+j];
        if (ifrg==0)
          break;
        for (int k=0;k<frgidx[ifrg-1];k++) {
          bas_temp[ibas+k]=frags[(ifrg-1)*nb+k]+1;
        }
        ibas+=frgidx[ifrg-1];
      }
    }

    ma[i]=ibas;
    if (*(ma0+i)>0)
      ma0[i]=ibas;
    memcpy(nv+i*nb,bas_temp,ibas*sizeof(int));
  }

  free(bas_temp);

  return 0;
}

int sortbas_frg(int *frags,int nbas) {
  int i,j,tmp;

  for (i=0;i<nbas-1;i++)
    for (j=i+1;j<nbas;j++)
      if (*(frags+i) > *(frags+j)) {
        tmp=*(frags+i);
        *(frags+i)=*(frags+j);
        *(frags+j)=tmp;
      }

  return 0;
}

int parse_sao(char *cline,char *baslist, int *bas2atm,int *frags,int *frgidx) {
  char str1[100];
  int atms[1024];
  int iatm;
  int i,nsplit;

  char** str=malloc_string_array(100,1024);

  sscanf(cline,"%s",str1);
  for (i=strlen(str1);i<strlen(cline);i++)
    if (*(cline+i)!=' ') break;
  nsplit=0;
  split_string(cline+i,str,1024,&nsplit);

  iatm=0;
  expand_str_orb(str,atms,nsplit,&iatm);

  if (iatm!=(*frgidx)) {
    printf("Error in parsing SAO fragment %s\n",cline);
    printf("Assigned %d atoms, actually found %d atoms\n",(*frgidx),iatm);
    exit(1);
  }

  int basstart,basend;
  int ichar=0;
  int ibas=0;
  int istep=0;
  char bas[10],currentbas[10];
  memset(currentbas,0,10);
  while(ichar<strlen(str1)) {
    memset(currentbas+1,0,9);
    if (*(str1+ichar)=='S') {
      istep=0;
      strncpy(currentbas,str1+ichar,istep+1);
      ichar+=istep+1;
    }
    else if (*(str1+ichar)=='P') {
      istep=1;
      strncpy(currentbas,str1+ichar,istep+1);
      ichar+=istep+1;
    }
    else if (*(str1+ichar)=='D') {
      istep=2;
      strncpy(currentbas,str1+ichar,istep+1);
      ichar+=istep+1;
    }
    else if (*(str1+ichar)=='F') {
      istep=3;
      strncpy(currentbas,str1+ichar,istep+1);
      ichar+=istep+1;
    }
    else if (*(str1+ichar)=='G') {
      istep=4;
      strncpy(currentbas,str1+ichar,istep+1);
      ichar+=istep+1;
    }
    else if (*(str1+ichar)=='H') {
      istep=5;
      strncpy(currentbas,str1+ichar,istep+1);
      ichar+=istep+1;
    }
    else if (*(str1+ichar)=='I') {
      istep=6;
      strncpy(currentbas,str1+ichar,istep+1);
      ichar+=istep+1;
    }
    else if (*(str1+ichar)==' ') {
      ichar++;
      continue;
    }
    else {
      strncpy(currentbas+1,str1+ichar,istep);
      ichar+=istep;
    }

    for (int i=0;i<*frgidx;i++) {
      iatm=atms[i]-1;
      basstart=*(bas2atm+2*iatm);
      basend=*(bas2atm+2*iatm+1);
      for (int j=basstart;j<=basend;j++) {
        memset(bas,0,10);
        strncpy(bas,baslist+j*10,10);
        if (strstr(bas,currentbas)!=NULL) {
          *(frags+ibas)=j;
          ibas++;
        }
      }
    }
  }

  free_string_array(str);

  *frgidx=ibas;

  return 0;
}

int parse_atom(int *frg_tmp,int *bas2atm,int *frags,int *frgidx) {
  
  int ibas=0;
  int basstart,basend;
  int nfrg=*frgidx;
  for (int i=0;i<nfrg;i++) {
    basstart=*(bas2atm+2*(*(frg_tmp+i)-1));
    basend=*(bas2atm+2*(*(frg_tmp+i))-1);
    for (int j=basstart;j<=basend;j++) {
      *(frags+ibas)=j;
      ibas++;
    }
  }

  *frgidx=ibas;

  return 0;
}


int getbaslist(mol_info mol, char *baslist, int *bas2atm) {
  char *bas;
  int ibas,iatm1,iatm2;
  char bas1[10];

  ibas=0;
  iatm1=0;
  memset(bas2atm,0,mol->atm->natm*2*sizeof(int)); 
  memset(baslist,0,mol->bas->msize*10*sizeof(char));
  for (int i=0;i<mol->bas->nbas;i++) {
    int di=xint_gtolen(mol->bas->bas[i*5+ANGULAR_VAL]);
    for (int j=0;j<di;j++) {
      iatm2=mol->bas->bas[i*5+ATOM_IND];
      if (iatm2>iatm1) {
        *(bas2atm+iatm2*2)=ibas;
        *(bas2atm+iatm1*2+1)=ibas-1;
        iatm1++;
      }
      bas=get_ang_tag(j,mol->bas->bas[i*5+ANGULAR_VAL]);
      strncpy((baslist+ibas*10),bas,strlen(bas));
      ibas++;
    }
  }

  *(bas2atm+2*mol->atm->natm-1)=mol->bas->msize-1;

  return 0;
}



int getfrg(char *inpname, mol_info mol, vb_info vb_str) {

  int *frgidx, *frags, nfrg;
  char cline[1024];
  int *frg_tmp=(int*)malloc(sizeof(int)*2*vb_str->nb);
  int nsplit;

  char** split=malloc_string_array(1024,1024);

  FILE *fp=fopen(inpname,"r");

  char *baslist;
  int *bas2atm;
  if (vb_str->inttyp==INT_READ) {
    baslist=(char*)malloc(vb_str->nbf*10*sizeof(char));
    bas2atm=(int*)malloc(vb_str->natom*2*sizeof(int));
    memset(baslist,'\0',sizeof(char)*vb_str->nbf*10);
    for (int i=0;i<vb_str->nbf;i++) {
      memcpy(baslist+i*10,vb_str->bas_tag+i*9,8*sizeof(char));
    }
    for (int i=0;i<vb_str->natom;i++) {
      bas2atm[2*i]=vb_str->limlow[i];
      bas2atm[2*i+1]=vb_str->limsup[i];
    }
  }
  else {
    baslist=(char*)malloc(vb_str->nb*10*sizeof(char));
    bas2atm=(int*)malloc(vb_str->natom*2*sizeof(int));
    getbaslist(mol,baslist,bas2atm);
  }

  if (fvbsec(fp,"$FRAG")>0) {
    if (vb_str->frgtyp==FRG_ATM) {
      frags=(int*)malloc(vb_str->natom*vb_str->nb*sizeof(int));
      frgidx=(int*)malloc(vb_str->natom*sizeof(int));
      nfrg=vb_str->natom;
      for (int i=0;i<vb_str->natom;i++) {
        frg_tmp[0]=i+1;
        *(frgidx+i)=1;
        parse_atom(frg_tmp,bas2atm,frags+i*vb_str->nb,frgidx+i);
      }
    }
    else {
      printf("Error! No $FRAG found.\n");
      exit(1);
    }
  }
  else {
    // read head of $FRAG
    nfrg=0;
    if (fgets(cline,1024,fp)!=NULL) {
      if (strstr(cline,"$END")==NULL) {
        nsplit=0;
        split_string(cline,split,1024,&nsplit);
        expand_str_orb(split,frg_tmp,nsplit,&nfrg);
        frgidx=(int*)malloc(nfrg*sizeof(int));
        memcpy(frgidx,frg_tmp,nfrg*sizeof(int));
        frags=(int*)malloc(nfrg*vb_str->nb*sizeof(int));
      }
    }

    if (nfrg == 0) {
      printf("Error in readin $FRAG.\n");
      exit(1);
    }

    int ifrg=0;
    if (vb_str->frgtyp==FRG_ATM) {
      // FRGTYP=ATOM
      int iatm=0;
      while(fgets(cline,1024,fp)!=NULL) {
        if (strstr(cline,"$END")!=NULL)
          break;
        nsplit=0;
        split_string(cline,split,1024,&nsplit);
        expand_str_orb(split,frg_tmp,nsplit,&iatm);
        if (iatm>=*(frgidx+ifrg)) {
//          memcpy(frags+ifrg*natom,frg_tmp,(*(frgidx+ifrg))*sizeof(int));
          parse_atom(frg_tmp,bas2atm,frags+ifrg*vb_str->nb,frgidx+ifrg);
          iatm=0;
          ifrg++;
        }
        if (ifrg>=nfrg)
          break;
      }
    }
    else {
      // FRGTYP=SAO
      while(fgets(cline,1024,fp)!=NULL) {
        if (strstr(cline,"$END")!=NULL)
          break;
        parse_sao(cline,baslist,bas2atm,frags+ifrg*vb_str->nb,frgidx+ifrg);
        ifrg++;
        if (ifrg>=nfrg)
          break;
      }
    }

    if (ifrg<nfrg) {
      printf("Error! Not enough number of fragments read.\n");
      exit(1);
    }

  }

  fclose(fp);

  for (int i=0;i<nfrg;i++)
    sortbas_frg(frags+i*vb_str->nb,*(frgidx+i));

  frg2orb(frags,frgidx,nfrg,vb_str->nv,vb_str->ma,vb_str->ma0,vb_str->nor,vb_str->nb);

  free_string_array(split);

  free(frg_tmp);
  free(baslist);
  free(bas2atm);
  free(frgidx);
  free(frags);

  return 0;
}
