#include <stdio.h>
#include <string.h>
#include "vb/vb.h"

int combi(int m, int n) {
  int ncombi;
  int m0,i;

  m0=(m<n-m?m:n-m);

  if (m0==0)
    return 1;

  ncombi=n;
  if (m0>1) {
    for (i=1;i<=m0-1;i++) {
      ncombi*=(n-i);
      ncombi/=i;
    }
    ncombi/=m0;
  }

  return ncombi;
}

int sn(int n, int spin) {

  if (n==0)
    return 1;

  int val=spin;
  int i;
  int offset=(n-spin+1)/2;
  int off2=(n+spin+1)/2;
  int nloop=(n+spin-1)/2; // n-(n-spin+1)/2

  for (i=1;i<=nloop;i++) {
    val*=(i+offset);
    if (i<=off2)
      val/=i;
  }

  if (off2>nloop) {
    for (i=nloop+1;i<=off2;i++)
      val/=i;
  }

  return val;

}

int strnum(int ionclass, int ndb, int nskip, int nor, int nel, int nmul) {
  return combi(ionclass,nor)*combi(nskip+ionclass-ndb,nor-ionclass)*sn(nor-2*ionclass-nskip+ndb,nmul);
}

int detnum(int ionclass, int ndb, int nskip, int nor, int nel, int nmul) {
  int nalpha=(nel+nmul-1)/2;
  return combi(ionclass,nor)*combi(nskip+ionclass-ndb,nor-ionclass)*combi(nalpha-ionclass,nor-2*ionclass-nskip+ndb);
}

int compute_nstr(int *nstr, int *ndet, int *maxlevel, int *idxclass, int nclass, int nor, int nel, int nmul,int wfntyp, int print_level) {
  int nskip=(nor-nel>0?nor-nel:0);
  int ndb=(nel-nor>0?nel-nor:0);
  int nbeta=(nel-2*ndb-nmul+1)/2;

  *nstr=0;
  *ndet=0;
  *maxlevel=0;
  int i,iclass,istr,idet;
  for (i=0;i<nclass;i++) {
    iclass=*(idxclass+i);
    if (wfntyp!=WFN_DET) {
      istr=strnum(iclass,ndb,nskip,nor,nel,nmul);
      if (print_level>0)
        printf("Number of %3dth ion structures   is : %10d  from %10d to %10d\n",iclass-ndb,istr,*nstr+1,*nstr+istr);
      (*nstr)+=istr;
    }
    idet=detnum(iclass,ndb,nskip,nor,nel,nmul);
    if (print_level>0)
      printf("Number of %3dth ion determinants is : %10d  from %10d to %10d\n",iclass-ndb,idet,*ndet+1,*ndet+idet);
    (*ndet)+=idet;
    if (*maxlevel<idet)
      *maxlevel=idet;
  }

  return 0;
}

int sortclass(int *idxclass, int *nclass) {
  int itmp[100];
  memset(itmp,0,sizeof(itmp));

  int i;
  for (i=0;i<*nclass;i++)
    itmp[*(idxclass+i)]=1;

  memset(idxclass,0,100*sizeof(int));

  *nclass=0;
  for (i=0;i<100;i++)
    if (itmp[i]>0) {
      *(idxclass+(*nclass))=i;
      (*nclass)++;
    }

  return 0;
}


int get_strclass(char *class, int *idxclass, int *nclass,int nor, int nel, int nmul) {
  int nskip=(nor-nel>0?nor-nel:0);
  int ndb=(nel-nor>0?nel-nor:0);
  int nbeta=(nel-2*ndb-nmul+1)/2;

  int i;
  if (strstr(class,"FULL")!=NULL) {
    for (i=0;i<=nbeta;i++)
      *(idxclass+i)=i;
    *nclass=nbeta+1;
  }
  else {
    if (strstr(class,"COV")!=NULL) {
      *(idxclass+(*nclass))=0;
      (*nclass)++;
    }
    if (strstr(class,"ION(")!=NULL) {
      char *is=strchr(class,')');
      if (is==NULL) {
        printf("Error in getting structure classes. Found \"ION(\" but not \")\" found\n");
        exit(1);
      }
      char *ip=strchr(class,'(');
      char ionval[100];
      strncpy(ionval,ip+1,strlen(ip)-1-strlen(is));
      for (i=0;i<strlen(ionval);i++)
        if (ionval[i]==',')
          ionval[i]=' ';
      char** split=malloc_string_array(100,1024);

      int nsplit=0;
      split_string(ionval,split,1024,&nsplit);
      expand_str_orb(split,idxclass,nsplit,nclass);
      free_string_array(split);
    }
    else if (strstr(class,"ION")!=NULL)
      for (i=1;i<=nbeta;i++) {
        *(idxclass+(*nclass))=i;
        (*nclass)++;
    }
  }

  sortclass(idxclass,nclass);
  
  if (*idxclass < 0) {
    printf("Error! Structure class %d cannot be smaller than 0\n",*idxclass);
    exit(1);
  }
  if (*(idxclass+(*nclass)-1) > nbeta) {
    printf("Error! Structure class %d cannot be larger than %d\n",*(idxclass+(*nclass)-1),nbeta);
    exit(1);
  }

  for (i=0;i<*nclass;i++)
    (*(idxclass+i))+=ndb;

  return 0;


}

int genyoung(int *A,int *ncov, int nalpha, int nbeta, int nor, int nalpha0) {

  if (nalpha>0) {

    int i,j;

    for (i=0;i<nalpha;i++)
      *(A+i)=i+1;
    (*ncov)++;

    if (nor==0 || nor < nalpha) {
      printf("There are no covalent structures, When nor=0 OR nor<nalpha\n");
      return 0;
    }
    int nmul=nalpha-nbeta+1;
    int *nmax=(int*)malloc(nalpha*sizeof(int));
    memset(nmax,0,sizeof(int)*nalpha);
    if (nmul==1)
      *(nmax+nbeta-1)=nor-1;
    else {
      *(nmax+nalpha-1)=nor;
      for (i=nalpha-1;i>=nbeta+1;i--)
        *(nmax+i-1)=*(nmax+i)-1;
      if (nbeta>0)
        *(nmax+nbeta-1)=*(nmax+nbeta)-2;
    }
    for (i=nbeta-1;i>=1;i--)
      *(nmax+i-1)=*(nmax+i)-2;

    int skip=0;
    while (skip==0) {
      for (i=nalpha;i>=1;i--) {
        if (*(A+(*ncov-1)*nalpha0+i-1)<*(nmax+i-1)) {
          memcpy(A+(*ncov)*nalpha0,A+(*ncov-1)*nalpha0,(i-1)*sizeof(int));
          *(A+(*ncov)*nalpha0+i-1)=*(A+(*ncov-1)*nalpha0+i-1)+1;
          for (j=i;j<nalpha;j++)
            *(A+(*ncov)*nalpha0+j)=*(A+(*ncov)*nalpha0+j-1)+1;
          (*ncov)++;
          break;
        }
      }
      for (i=0;i<nalpha;i++) {
        if (*(A+(*ncov-1)*nalpha0+i)<*(nmax+i)) {
          skip=0;
          break;
        }
        else
          skip=1;
      }
    }
    free(nmax);
  }

  return 0;
}

int gencov(int *A, int *B, int *ncov, int nalpha, int nbeta, int nor, int nalpha0) {

  if (nbeta>0) {

    int *Atemp=(int*)malloc(sizeof(int)*nalpha*(*ncov));
    int *Btemp=(int*)malloc(sizeof(int)*nbeta*20000);

    int i,j,k,l,need;
    int icov;
    if (nalpha+nbeta==nor) {
      for (icov=0;icov<*ncov;icov++) {
        l=0;
        for (j=0;j<nor;j++) {
          need=1;
          for (k=0;k<nalpha;k++) {
            if (*(A+icov*nalpha0+k)==j+1) {
              need=0;
              break;
            }
          }
          if (need>0) {
            *(B+icov*nalpha0+l)=j+1;
            l++;
          }
        }
      }
    }
    else {
      int ncov0=*ncov;

      for (i=0;i<ncov0;i++)
        for (j=0;j<nalpha;j++)
          *(Atemp+i*nalpha+j)=*(A+i*nalpha0+j);

      *ncov=0;
      int ncov1=0;
      genyoung(Btemp,&ncov1,nbeta,0,nor,nbeta);
      for (icov=0;icov<ncov0;icov++) {
        for (i=0;i<ncov1;i++) {
          need=1;
          for (j=0;j<nbeta;j++) {
            if (*(Btemp+i*nbeta+j) < *(Atemp+icov*nalpha+j)) {
              need=0;
              break;
            }
            for (k=j;k<nalpha;k++) {
              if (*(Btemp+i*nbeta+j) == *(Atemp+icov*nalpha+k)) {
                need=0;
                break;
              }
            }
          }
          if (need>0) {
            memcpy(A+(*ncov)*nalpha0,Atemp+icov*nalpha,nalpha*sizeof(int));
            memcpy(B+(*ncov)*nalpha0,Btemp+i*nbeta,nbeta*sizeof(int));
            (*ncov)++;
          }
        }
      }
    }

    if (nalpha>nbeta)
      for (i=0;i<*ncov;i++)
        for (j=nbeta;j<nalpha;j++)
          *(B+i*nalpha0+j)=nor+nalpha-j;

    free(Atemp);
    free(Btemp);

  }

  return 0;
}


int young2bond(int *vectmp1,int *vectmp,int nvec,int ndim,int nao) {

  int *index1=(int*)malloc(sizeof(int)*3*nao);
  int *index2=(int*)malloc(sizeof(int)*3*nao);
  int *vectmp2=(int*)malloc(sizeof(int)*ndim);

  memset(index1,0,sizeof(int)*3*nao);
  memset(index2,0,sizeof(int)*3*nao);
  memcpy(vectmp2,vectmp,sizeof(int)*ndim);

  int nvec0=nvec;

  int j,j1,j2,maxl1;

  for (j=0;j<nvec;j++)
    *(index2+(*(vectmp2+j))-1)=j+1;

  for (j1=nvec;j1>=1;j1--) {
    maxl1=*(vectmp1+nvec0-1);
    for (j2=0;j2<nvec0;j2++) {
      if (maxl1<=*(vectmp2+j2)) {
        *(index1+maxl1-1)=*(index2+(*(vectmp2+j2))-1);
        break;
      }
    }
    for (j=j2;j<nvec0-1;j++)
      *(vectmp2+j)=*(vectmp2+j+1);
    nvec0--;
  }

  for (j=0;j<nvec;j++)
    *(vectmp2+(*(index1+(*(vectmp1+j))-1))-1)=*(vectmp1+j);

  memcpy(vectmp1,vectmp2,nvec*sizeof(int));

  free(index1);
  free(index2);
  free(vectmp2);

  return 0;
}

int replace_cov(int *A, int *B, int *Atemp, int *Btemp, int *reporblist, int nor_cov, int ncov, int nalpha, int nbeta, int nalpha0) {

  int i,j,k;

  for (i=0;i<ncov;i++) {
    for (j=0;j<nalpha;j++) {
      k=*(A+i*nalpha0+j);
      *(Atemp+i*nalpha+j)=*(reporblist+k-1);
    }
    for (j=0;j<nbeta;j++) {
      k=*(B+i*nalpha0+j);
      *(Btemp+i*nbeta+j)=*(reporblist+k-1);
    }
  }

  return 0;
}

int expandstr(int *ntstr, int *A, int *B, int *ionlist, int *nstr, int totnel, int ncov, int nion, int nalpha, int nbeta, int nalpha0, int nbeta0, int ndb, int nor) {

  int nor_ion = nbeta0 - nbeta;
  int nor_cov = nor - nor_ion;
  int ionclass = nor_ion;

  int *ionpairs=NULL;
  int *reporblist=NULL;

  if (ionclass>0)
    ionpairs=(int*)malloc(sizeof(int)*ionclass*200000);
  if (nor_cov>0)
    reporblist=(int*)malloc(sizeof(int)*nor_cov);

  int i,j,k,l,npairs,need;
  if (ionclass>0) {
    int *Atemp=(int*)malloc(sizeof(int)*nalpha*20000);
    int *Btemp=(int*)malloc(sizeof(int)*nbeta*20000);
    for (l=0;l<nion;l++) {
      if (ncov>0) {
        k=0;
        for (i=0;i<nor;i++) {
          need=1;
          for (j=0;j<nor_ion;j++) {
            if (*(ionlist+l*nbeta0+j)==i+1) {
              need=0;
              break;
            }
          }
          if (need>0) {
            *(reporblist+k)=i+1;
            k++;
          }
        }
        replace_cov(A,B,Atemp,Btemp,reporblist,nor_cov,ncov,nalpha,nbeta,nalpha0);
        for (i=0;i<ncov;i++) {
          for (j=0;j<ionclass;j++) {
            *(ntstr+(*nstr+i)*totnel+2*j)=*(ionlist+l*nbeta0+j);
            *(ntstr+(*nstr+i)*totnel+2*j+1)=*(ionlist+l*nbeta0+j);
          }
          for (j=0;j<nbeta;j++) {
            *(ntstr+(*nstr+i)*totnel+2*ionclass+2*j)=*(Atemp+i*nalpha+j);
            *(ntstr+(*nstr+i)*totnel+2*ionclass+2*j+1)=*(Btemp+i*nbeta+j);
          }
          for (j=nbeta;j<nalpha;j++)
            *(ntstr+(*nstr+i)*totnel+2*ionclass+nbeta+j)=*(Atemp+i*nalpha+j);
        }
        (*nstr)+=ncov;
      }
      else {
        for (j=0;j<ionclass;j++) {
          *(ntstr+(*nstr)*totnel+2*j)=*(ionlist+l*nbeta0+j);
          *(ntstr+(*nstr)*totnel+2*j+1)=*(ionlist+l*nbeta0+j);
        }
        (*nstr)++;
      }
    }

    free(Atemp);
    free(Btemp);
  }
  else {
    for (i=0;i<ncov;i++) {
      for (j=0;j<nbeta;j++) {
        *(ntstr+(*nstr+i)*totnel+2*j)=*(A+i*nalpha0+j);
        *(ntstr+(*nstr+i)*totnel+2*j+1)=*(B+i*nalpha0+j);
      }
      for (j=nbeta;j<nalpha;j++)
        *(ntstr+(*nstr+i)*totnel+j+nbeta)=*(A+i*nalpha0+j);
    }
    (*nstr)+=ncov;
  }



  if (ionpairs!=NULL)
    free(ionpairs);
  if (reporblist!=NULL)
    free(reporblist);

  return 0;
}


int build_structure(int *ntstr, int *idxclass, int nclass, int nor, int nel, int nmul,int maxlevel) {

  int nbeta=(nel-nmul+1)/2;
  int nalpha=nel-nbeta;
  int ndb=(nel-nor>0?nel-nor:0);

  int *ionlist=(int*)malloc(sizeof(int)*nbeta*maxlevel);
  int *A=(int*)malloc(sizeof(int)*nalpha*maxlevel);
  int *B=(int*)malloc(sizeof(int)*nalpha*maxlevel);
  int *Btemp=(int*)malloc(sizeof(int)*nalpha);

  int i,nion,iclass,j;
  int nstr=0;
  int ncov;
  for (i=0;i<nclass;i++) {
    iclass=*(idxclass+i);
//    printf("for iclass = %d\n",iclass);
    nion=0;
    genyoung(ionlist,&nion,iclass,0,nor,nbeta);
//    printf("ionlist\n");
//    for (j=0;j<nion;j++) {
//      for (int k=0;k<nbeta;k++)
//        printf("%d ",*(ionlist+j*nbeta+k));
//      printf("\n");
//    }
    ncov=0;
    genyoung(A,&ncov,nalpha-iclass,nbeta-iclass,nor-iclass,nalpha);
//    printf("ncov = %d\n",ncov);
//    for (j=0;j<ncov;j++) {
//      for (int k=0;k<nalpha;k++)
//        printf("%d ",*(A+j*nalpha+k));
//      printf("\n");
//    }
    gencov(A,B,&ncov,nalpha-iclass,nbeta-iclass,nor-iclass,nalpha);
//    printf("ncov = %d\n",ncov);
//    for (j=0;j<ncov;j++) {
//      for (int k=0;k<nalpha;k++)
//        printf("%d ",*(A+j*nalpha+k));
//      printf("\n");
//      for (int k=0;k<nbeta;k++)
//        printf("%d ",*(B+j*nalpha+k));
//      printf("\n");
//    }

    if (nbeta-iclass>0) {
      for (j=0;j<ncov;j++) {
        young2bond(A+j*nalpha,B+j*nalpha,nalpha-iclass,nalpha,nor);
      }
    }
//    printf("ncov = %d\n",ncov);
//    for (j=0;j<ncov;j++) {
//      for (int k=0;k<nalpha;k++)
//        printf("%d ",*(A+j*nalpha+k));
//      printf("\n");
//      for (int k=0;k<nbeta;k++)
//        printf("%d ",*(B+j*nalpha+k));
//      printf("\n");
//    }
    expandstr(ntstr,A,B,ionlist,&nstr,nel,ncov,nion,nalpha-iclass,nbeta-iclass,nalpha,nbeta,ndb,nor);
//    printf("str for iclass = %d\n",iclass);
//    for (j=0;j<nstr;j++) {
//      for (int k=0;k<nel;k++)
//        printf("%d ",*(ntstr+j*nel+k));
//      printf("\n");
//    }
  }


  free(ionlist);
  free(A);
  free(B);
  free(Btemp);

  return 0;
}

int genstr(vb_info vb_str, int print_level) {

  int nao=vb_str->nao;
  int nae=vb_str->nae;
  int nmul=vb_str->nmul;
  int naeb=(nae-nmul+1)/2;

  if (naeb < 0) {
    printf("Error in generating structures. Nubmer of alpha electrons %d is larger than number of electrons %d\n",nmul-1,nae);
    exit(1);
  }
  if (naeb > nao) {
    printf("Error in generating structures. Nubmer of beta electrons %d is larger than number of orbitals %d\n",naeb,nao);
    exit(1);
  }

  int nel=vb_str->nel;
  int ndb=(nel-nae)/2;
  vb_str->nor=ndb+nao;

  int i,j;
  if ((naeb==0 && nae==nmul-1) || (naeb==nao && nmul==1)) {
    vb_str->nstr=1;
    vb_str->ntstr=(int*)malloc(nel*sizeof(int));
    int nbeta=(nel-nmul+1)/2;
    if (vb_str->wfntyp==WFN_DET) {
      for (i=0;i<ndb;i++) {
        *(vb_str->ntstr+i)=i+1;
        *(vb_str->ntstr+i+nbeta)=i+1;
      }
      for (i=ndb;i<ndb+naeb;i++) {
        *(vb_str->ntstr+i)=i+1;
        *(vb_str->ntstr+i+nbeta)=i+1;
      }
      for (i=0;i<nmul-1;i++)
        *(vb_str->ntstr+2*(ndb+naeb)+i)=i+naeb+ndb+1;
    }
    else {
      for (i=0;i<ndb;i++) {
        *(vb_str->ntstr+2*i)=i+1;
        *(vb_str->ntstr+2*i+1)=i+1;
      }
      for (i=ndb;i<ndb+naeb;i++) {
        *(vb_str->ntstr+2*i)=i+1;
        *(vb_str->ntstr+2*i+1)=i+1;
      }
      for (i=0;i<nmul-1;i++)
        *(vb_str->ntstr+2*(ndb+naeb)+i)=i+naeb+ndb+1;
    }
    vb_str->iomin = ((nae - nao) > 0) ? (nae - nao) : 0;
    vb_str->iomax = (nae - nmul + 1) / 2; 
  }
  else {
    int nclass,idxclass[100];
    nclass=0;
    get_strclass(vb_str->strclass,idxclass,&nclass,nao,nae,nmul);
//    printf("nclass = %d\n",nclass);
//    for (i=0;i<nclass;i++)
//      printf("%d ",*(idxclass+i));
//    printf("\n");
    int nstr,ndet;
    int maxlevel;
    compute_nstr(&nstr,&ndet,&maxlevel,idxclass,nclass,nao,nae,nmul,vb_str->wfntyp,print_level);

    vb_str->iomin = *(idxclass); 
    vb_str->iomax = *(idxclass + nclass - 1); 

    vb_str->ndet=ndet;
    if (vb_str->wfntyp==WFN_DET)
      vb_str->nstr=ndet;
    else
      vb_str->nstr=nstr;
//    printf("nstr = %d ndet = %d\n",nstr,ndet);

    vb_str->ntstr=(int*)malloc(sizeof(int)*nstr*nel);

    int *ntstr=(int*)malloc(sizeof(int)*nae*nstr);

    build_structure(ntstr,idxclass,nclass,nao,nae,nmul,maxlevel);

    for (i=0;i<nstr;i++) {
      for (j=0;j<ndb;j++) {
        *(vb_str->ntstr+i*nel+2*j)=j+1;
        *(vb_str->ntstr+i*nel+2*j+1)=j+1;
      }
      for (j=0;j<nae;j++)
        *(vb_str->ntstr+i*nel+j+2*ndb)=*(ntstr+i*nae+j)+ndb;
    }

//    printf("final structures\n");
//    for (i=0;i<nstr;i++) {
//      for (j=0;j<nel;j++)
//        printf("%d ",*(vb_str->ntstr+i*nel+j));
//      printf("\n");
//    }
    free(ntstr);

  }

  return 0;
}
