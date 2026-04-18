#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vb/vb.h"
#include "inpout/input.h"

int readorb(char *inpname, vb_info vb_str) {

  char cline[1024];
  char** split=malloc_string_array(1024,1024);

  int nsplit;
  int iorb,ibas;
  int nor, nb;
  int orb_tmp[2*vb_str->nor*vb_str->nb];

  nor=vb_str->nor;
  nb=vb_str->nb;

  int ndb;
  if (vb_str->nao>0)
    ndb=nor-vb_str->nao;
  else
    ndb=detect_ndb(vb_str->ntstr,vb_str->nstr,vb_str->nel,vb_str->nmul,vb_str->wfntyp);

  FILE *fp=fopen(inpname,"r");

  int iforb = fvbsec(fp, "$ORB"); 
  int ifactorb = fvbsec(fp, "$ACTORB"); 

  // if (fvbsec(fp,"$ORB")>0) {
  if ((iforb + ifactorb) == 2) {
    printf("Error! No $ORB or $ACTORB found.\n");
    exit(1);
  }
  else if ((iforb + ifactorb) == 0) {
    printf("Error! $ORB and $ACTORB cannot be used together.\n");
    exit(1);
  }

  vb_str->simple = iforb; 

  if (vb_str->simple == 0) {
    fvbsec(fp,"$ORB"); 
    iorb=0;
    while(fgets(cline,1024,fp)!=NULL) {
      if (strstr(cline,"$END")!=NULL)
        break;
      nsplit=0;
      split_string(cline,split,1024,&nsplit);
      expand_str_orb(split,orb_tmp,nsplit,&iorb);
      if (iorb>=nor) {
        memcpy(vb_str->ma0,orb_tmp,nor*sizeof(int));
        break;
      }
    }

    long int cur_pos=ftell(fp);
    int cvic_read=0;
    // cvic_read   0: orb without symm info
    // cvic_read   1: orb with symm info

    if(fgets(cline,1024,fp)==NULL) {
      printf("Error in reading orbital. Not enough content.\n");
      exit(1);
    }
    else {
      if (strchr(cline,'.')!=NULL)
        cvic_read=1;
    }

//    memcpy(vb_str->ma,vb_str->ma0,nor*sizeof(int));

    fseek(fp,cur_pos,SEEK_SET);

    iorb=0;
    ibas=0;

    vb_str->orb_with_symm=cvic_read;

    if (cvic_read==0) {
      while(fgets(cline,1024,fp)!=NULL) {
        if (strstr(cline,"$END")!=NULL)
          break;
        nsplit=0;
        split_string(cline,split,1024,&nsplit);
        expand_str_orb(split,orb_tmp,nsplit,&ibas);
        if (*(vb_str->ma0+iorb)==0) {
//          *(vb_str->ma0+iorb)=ibas;
          memcpy(vb_str->nv+iorb*nb,orb_tmp,ibas*sizeof(int));
          ibas=0;
          iorb++;
        }
        else if (ibas >= *(vb_str->ma0+iorb)) {
          memcpy(vb_str->nv+iorb*nb,orb_tmp,ibas*sizeof(int));
          ibas=0;
          iorb++;
        }
        if (iorb>=nor)
          break;
      }
    }
    else {
      nsplit=0;
      while(fgets(cline,1024,fp)!=NULL) {
        if (strstr(cline,"$END")!=NULL)
          break;
        split_string(cline,split,1024,&nsplit);
        if (*(vb_str->ma0+iorb)==0) {
//          *(vb_str->ma0+iorb)=nsplit/2;
          for (int i=0;i<nsplit/2;i++) {
            *(vb_str->nv+iorb*nb+i)=atoi(split[2*i+1]);
            *(vb_str->cvic+iorb*nb+i)=atof(split[2*i]);
          }
          iorb++;
          nsplit=0;
        }
        else if (*(vb_str->ma0+iorb)<=nsplit/2) {
          for (int i=0;i<*(vb_str->ma0+iorb);i++) {
            *(vb_str->nv+iorb*nb+i)=atoi(split[2*i+1]);
            *(vb_str->cvic+iorb*nb+i)=atof(split[2*i]);
          }
          iorb++;
          nsplit=0;
        }
        if (iorb>=nor)
          break;
      }
    }
  }
  else if (vb_str->simple == 1) {
    fvbsec(fp, "$ACTORB"); 
    // OEO for inactive orbitals 
    for (int i = 0; i < ndb; i++) {
      *(vb_str->ma0 + i) = vb_str->natom; 
      for (int j = 0; j < vb_str->natom; j++) {
        *(vb_str->nv + i*nb + j) = j + 1; 
      }
    }
    // Read for active orbitals 
    iorb = ndb; 
    ibas = 0; 
    while(fgets(cline, 1024, fp) != NULL) {
      if (strstr(cline, "$END") != NULL) {
        break; 
      }

      nsplit = 0; 
      split_string(cline, split, 1024,&nsplit); 
      expand_str_orb(split, orb_tmp, nsplit, &ibas); 
      *(vb_str->ma0 + iorb) = ibas; 
      memcpy(vb_str->nv + iorb*nb, orb_tmp, ibas*sizeof(int)); 
      ibas = 0; 
      iorb++; 

      if (iorb >= nor) {
        break; 
      }
    }
  }

  if (vb_str->nfroz > 0) {
    for (int i = 0; i < vb_str->nfroz; i++) {
      vb_str->ma0[vb_str->froz_list[i]-1] = 0; 
    }
  }

  memcpy(vb_str->ma,vb_str->ma0,nor*sizeof(int));

  if (vb_str->orbtyp!=HAO_TYP &&
      !vb_str->preserve_explicit_sparse_orbital_layout) {
    for (int i=0;i<nor;i++) {
      if (vb_str->ma[i]==1)
        vb_str->ma0[i]=0;
      else if (vb_str->ma[i]==0) {
        ibas=0;
        for (int j=0;j<nb;j++) {
          if (vb_str->nv[i*nb+j]==0)
            break;
          ibas++;
        }
        vb_str->ma[i]=ibas;
      }
    }
  }


  free_string_array(split);

  fclose(fp);

  if (iorb<nor) {
    printf("\n\nError in read VB orbitals. Not enough number of orbitals read.\n");
    printf("Need %d orbitals but read only %d orbitals.\n", nor, iorb); 
    exit(1);
  }

  return 0;
}
