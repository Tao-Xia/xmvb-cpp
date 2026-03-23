#include <stdio.h>
#include <string.h>
#include "vb/vb.h"
#include "inpout/output_func.h"


void string_replace(char *string,char *sub,char *repl) {
  int length = strlen(string);
  int sublen = strlen(sub);
  int lenp;
  char str1[1024],str2[1024];
  char *ip;
  ip=strstr(string,sub);
  while (ip!=NULL) {
    lenp=strlen(ip);
    memset(str1,'\0',sizeof(char)*1024);
    memset(str2,'\0',sizeof(char)*1024);
    strncpy(str1,string,length-lenp);
    strncpy(str2,ip+sublen,lenp-sublen);
    memset(string,'\0',sizeof(char)*length);
    strncpy(string,str1,length-lenp);
    strncpy(string+length-lenp,repl,strlen(repl));
    strncpy(string+length-lenp+strlen(repl),str2,lenp-sublen);
    length = strlen(string);
    ip = strstr(string,sub);
  }

  return;
}


void read_int_matrix(FILE *fp, double *array, size_t nlen) {
  int idx=0;
  int nsplit;
  char** split=malloc_string_array(1024,1024);
  char cline[1024];

  while (fgets(cline,1024,fp)!=NULL) {
    string_replace(cline,"D","e");
    nsplit=0;
    split_string(cline,split,1024,&nsplit);
    for (int i=0;i<nsplit;i++)
      array[idx+i]=atof(split[i]);
    idx+=nsplit;
    if (idx>=nlen)
      break;
    memset(cline,'\0',sizeof(char)*1024);
  }

  free_string_array(split);
  return;
}

void readonee(FILE *fp,char *string, double *val,int *param, double *T1, double *T2,double *T3,double *T4,vb_info vb_str) {

  int nd=*param;

  if (fvbsec(fp,string)!=0) {
    printf("Error! Section %s is not found\n",string);
    exit(1);
  }

  char cline[1024];

  memset(cline,'\0',1024*sizeof(char));

  int nmat,idx;
  double *totmat;
  double val1,val2,val3,val4;
  char str1[100],str2[100],str3[100],str4[100];
  if (strstr(string,"XYZE")!=NULL) {
    nmat=4;
    totmat=(double*)malloc(sizeof(double)*nd*nd*nmat);
    memset(totmat,0,sizeof(double)*nd*nd*nmat);
    if (fgets(cline,1024,fp)!=NULL) {
      string_replace(cline,"D","e");
      sscanf(cline,"%s%s%s%s",str1,str2,str3,str4);
      val[0]=atof(str1);
      val[1]=atof(str2);
      val[2]=atof(str3);
      val[3]=atof(str4);
    }
    read_int_matrix(fp,totmat,nmat*nd*nd);

    memcpy(T1,totmat,sizeof(double)*nd*nd);
    memcpy(T2,totmat+nd*nd,sizeof(double)*nd*nd);
    memcpy(T3,totmat+2*nd*nd,sizeof(double)*nd*nd);
    memcpy(T4,totmat+3*nd*nd,sizeof(double)*nd*nd);
    free(totmat);
  }
  else if (strstr(string,"EFSH")!=NULL) {
    if (fgets(cline,1024,fp)!=NULL) {
      string_replace(cline,"D","e");
      val[0]=atof(cline);
    }
    memset(cline,'\0',1024*sizeof(char));
    nmat=2;
    totmat=(double*)malloc(sizeof(double)*nmat*nd*nd);
    memset(totmat,0,sizeof(double)*nd*nd*nmat);
    read_int_matrix(fp,totmat,nmat*nd*nd);

    memcpy(T1,totmat,sizeof(double)*nd*nd);
    memcpy(T2,totmat+nd*nd,sizeof(double)*nd*nd);
    free(totmat);
  }
  else if (strstr(string,"CMAT")!=NULL) {
    if (fgets(cline,1024,fp)!=NULL) {
      string_replace(cline,"D","e");
      sscanf(cline,"%d %d",param,param+1);
    }
    memset(cline,'\0',1024*sizeof(char));
    int ndim=param[0]*param[1];
    vb_str->cmat=(double*)malloc(sizeof(double)*ndim);
    read_int_matrix(fp,vb_str->cmat,ndim);
  }
  else if (strstr(string,"SSF0")!=NULL) {
    read_int_matrix(fp,T1,nd*nd);
  }
  else if (strstr(string,"WNHF")!=NULL) {
    read_int_matrix(fp,T1,nd*nd);
  }

  return;
}

void read2e(vb_info vb_str) {
  assert("x2e.int" !=NULL);

  char cline[1024],str1[1024];

  memset(cline,'\0',1024*sizeof(char));

  // x2e.int part
  int nb=vb_str->nb;
  long nb2=(nb*nb+nb)/2;
  long nb4=(nb2*nb2+nb2)/2;

  double *gg0=(double*)malloc(sizeof(double)*nb4);

  FILE *fp=fopen("x2e.int","r");

  for (int i=0;i<nb4;i++) {
    if (fgets(cline,1024,fp)!=NULL) {
      string_replace(cline,"D","e");
      gg0[i]=atof(cline);
    }
    else {
      printf("Error in reading x2e.int\n");
      exit(1);
    }
    memset(cline,'\0',1024*sizeof(char));
  }

  fclose(fp);

  long n2e=0;
  for (long i=0;i<nb4;i++) {
    if (fabs(gg0[i])>=INT_TOL)
      n2e++;
  }

  vb_str->n2e=n2e;
  vb_str->ggf=(double*)malloc(sizeof(double)*n2e);
  vb_str->g2eidx=(int*)malloc(sizeof(int)*4*n2e);

  int lend;
  long ii=0,i2e=0;
  for (int i=0;i<nb;i++)
    for (int j=0;j<=i;j++)
      for (int k=0;k<=i;k++) {
        if (k==i)
          lend=j;
        else
          lend=k;
        for (int l=0;l<=lend;l++) {
          if (fabs(gg0[ii])>=INT_TOL) {
            vb_str->ggf[i2e]=gg0[ii];
            vb_str->g2eidx[4*i2e+0]=i;
            vb_str->g2eidx[4*i2e+1]=j;
            vb_str->g2eidx[4*i2e+2]=k;
            vb_str->g2eidx[4*i2e+3]=l;
            i2e++;
          }
          ii++;
        }
      }

  free(gg0);

  return;
}

void readinfo(vb_info vb_str) {
  assert("INFO" !=NULL);

  char cline[1024];
  memset(cline,'\0',1024*sizeof(char));

  FILE *fp=fopen("INFO","r");

  // INFO part
  int nat,nbf,nbfido;
  if (fgets(cline,1024,fp)!=NULL) {
    sscanf(cline,"%d %d %d",&nat,&nbf,&nbfido);
  }
  else {
    printf("Error in reading INFO\n");
    exit(1);
  }
  memset(cline,'\0',1024*sizeof(char));

  int *ibf=(int*)malloc(sizeof(int)*nat);
  for (int i=0;i<nat;i++) {
    if (fgets(cline,1024,fp)!=NULL) {
      sscanf(cline,"%d",ibf+i);
    }
    else {
      printf("Error in reading INFO\n");
      exit(1);
    }
    memset(cline,'\0',1024*sizeof(char));
  }
  vb_str->limsup=(int*)malloc(sizeof(int)*nat);
  vb_str->limlow=(int*)malloc(sizeof(int)*nat);
  vb_str->limlow[0]=0;

  for (int i=1;i<nat;i++)
    vb_str->limlow[i]=vb_str->limlow[i-1]+ibf[i];
  for (int i=0;i<nat;i++)
    vb_str->limsup[i]=vb_str->limlow[i]+ibf[i]-1;
  free(ibf);

  vb_str->ele_tag=(char*)malloc(sizeof(char)*4*nat);
  vb_str->zan=(double*)malloc(sizeof(double)*nat);
  vb_str->crd=(double*)malloc(sizeof(double)*3*nat);

  memset(vb_str->ele_tag,'\0',sizeof(char)*4*nat);
  memset(vb_str->zan,0,sizeof(double)*nat);
  memset(vb_str->crd,0,sizeof(double)*3*nat);

  char str1[100],str2[100],str3[100],str4[100];
  for (int i=0;i<nat;i++) {
    if (fgets(cline,1024,fp)!=NULL) {
      string_replace(cline,"D+","e+");
      string_replace(cline,"D-","e-");
      sscanf(cline,"%s%s%s%s%s",vb_str->ele_tag+i*4,str1,str2,str3,str4);
      vb_str->zan[i]=atof(str1);
      vb_str->crd[3*i+0]=atof(str2);
      vb_str->crd[3*i+1]=atof(str3);
      vb_str->crd[3*i+2]=atof(str4);
    }
    else {
      printf("Error in reading INFO\n");
      exit(1);
    }
    memset(cline,'\0',1024*sizeof(char));
  }

  vb_str->bas_tag=(char*)malloc(sizeof(char)*9*nbf);
  memset(vb_str->bas_tag,0,sizeof(char)*9*nbf);
  for (int i=0;i<nbf;i++) {
    memset(cline,0,sizeof(char)*1024);
    if (fgets(cline,1024,fp)==NULL) {
      printf("Error in reading INFO\n");
      exit(1);
    }
    else {
      string_replace(cline,"\n","\0");
      strncpy(vb_str->bas_tag+i*9,cline,8);
    }
  }

  fclose(fp);

  return;
}

void read1e(vb_info vb_str) {
  assert("x1e.int" !=NULL);

  FILE *fp=fopen("x1e.int","r");
  char cline[1024];
  int nb,nel;

  // x1e.int part
  if (fgets(cline,1024,fp)!=NULL) {
    sscanf(cline,"%d %d",&nb,&nel);
  }
  else {
    printf("Error in reading x1e.int\n");
    exit(1);
  }

  memset(cline,'\0',1024*sizeof(char));
  
  double val[4];
  int param[4];

  param[0]=nb;
  if (vb_str->xxf!=NULL)
    free(vb_str->xxf);
  if (vb_str->yyf!=NULL)
    free(vb_str->yyf);
  if (vb_str->zzf!=NULL)
    free(vb_str->zzf);
  if (vb_str->ekf!=NULL)
    free(vb_str->ekf);
  vb_str->xxf=(double*)malloc(sizeof(double)*nb*nb);
  vb_str->yyf=(double*)malloc(sizeof(double)*nb*nb);
  vb_str->zzf=(double*)malloc(sizeof(double)*nb*nb);
  vb_str->ekf=(double*)malloc(sizeof(double)*nb*nb);
  readonee(fp,"XYZE",val,param,vb_str->xxf,vb_str->yyf,vb_str->zzf,vb_str->ekf,vb_str);
  vb_str->xnuc=val[0];
  vb_str->ynuc=val[1];
  vb_str->znuc=val[2];
  vb_str->eknuc=val[3];

  if (vb_str->ssf!=NULL)
    free(vb_str->ssf);
  if (vb_str->hhf!=NULL)
    free(vb_str->hhf);
  vb_str->ssf=(double*)malloc(sizeof(double)*nb*nb);
  vb_str->hhf=(double*)malloc(sizeof(double)*nb*nb);
  readonee(fp,"EFSH",val,param,vb_str->ssf,vb_str->hhf,vb_str->hhf,vb_str->hhf,vb_str);
  vb_str->enuc=val[0];
  if (vb_str->cmat!=NULL)
    free(vb_str->cmat);
  vb_str->cmat=NULL;
  readonee(fp,"CMAT",val,param,vb_str->cmat,vb_str->cmat,vb_str->cmat,vb_str->cmat,vb_str);
  vb_str->nbf=param[0];
  vb_str->nbr=param[1];
  vb_str->ncore=vb_str->nbr-nb;
  if (vb_str->ss0!=NULL)
    free(vb_str->ss0);
  vb_str->ss0=(double*)malloc(sizeof(double)*vb_str->nbf*vb_str->nbf);
  readonee(fp,"SSF0",val,param,vb_str->ss0,vb_str->ss0,vb_str->ss0,vb_str->ss0,vb_str);
  param[0]=vb_str->nbf-vb_str->ncore;
  if (vb_str->hfwfn!=NULL)
    free(vb_str->hfwfn);
  vb_str->hfwfn=(double*)malloc(sizeof(double)*param[0]*param[0]);
  readonee(fp,"WNHF",val,param,vb_str->hfwfn,vb_str->hfwfn,vb_str->hfwfn,vb_str->hfwfn,vb_str);

  fclose(fp);

  vb_str->snorm=(double*)malloc(sizeof(double)*vb_str->nb);
  for (int i=0;i<nb;i++)
    vb_str->snorm[i]=sqrt(vb_str->ssf[i*vb_str->nb+i]);

  return;
}

void readint(vb_info vb_str) {

  read1e(vb_str);
  read2e(vb_str);
  
  return;
}
