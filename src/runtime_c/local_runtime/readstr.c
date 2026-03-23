#include <stdio.h>
#include <string.h>
#include "inpout/input.h"
#include "vb/vb.h"

//void expand_str_orb(char split[][1024],int *tmp,int nsplit,int *itmp) {
void expand_str_orb(char **split,int *tmp,int nsplit,int *itmp) {

  char *comma;
  char *hiph;
  char *star;
  char str1[10];
  int num1,num2;

  for (int i=0;i<nsplit;i++) {
    comma=strchr(split[i],':');
    hiph=strchr(split[i],'-');
    star=strchr(split[i],'*');
    if (comma!=NULL) {
      memset(str1,0,sizeof(str1));
      strncpy(str1,split[i],strlen(split[i])-strlen(comma));
      num1=atoi(str1);
      num2=atoi(comma+1);
      for (int j=0;j<(num2-num1+1);j++) {
        *(tmp+(*itmp)+2*j)=j+num1;
        *(tmp+(*itmp)+2*j+1)=j+num1;
      }
      (*itmp)+=2*(num2-num1+1);
    }
    else if (hiph!=NULL) {
      memset(str1,0,sizeof(str1));
      strncpy(str1,split[i],strlen(split[i])-strlen(hiph));
      num1=atoi(str1);
      num2=atoi(hiph+1);
      for (int j=0;j<(num2-num1+1);j++)
        *(tmp+(*itmp)+j)=j+num1;
      (*itmp)+=num2-num1+1;
    }
    else if (star!=NULL) {
      memset(str1,0,sizeof(str1));
      strncpy(str1,split[i],strlen(split[i])-strlen(star));
      num1=atoi(str1);
      num2=atoi(star+1);
      for (int j=0;j<num2;j++)
        *(tmp+(*itmp)+j)=num1;
      (*itmp)+=num2;
    }
    else {
      *(tmp+(*itmp))=atoi(split[i]);
      (*itmp)++;
    }
  }

  return;
}

int readstr(char *inpname, int *ntstr, double *col, int *nstr, int nel, int readcoef) {
 FILE *fp;
 char cline[1024];
 char** split=malloc_string_array(1024,1024);

 int nsplit;
 int iel,istr;
 int str_tmp[2*nel];

 fp=fopen(inpname,"r");

 if (fvbsec(fp,"$STR")>0) {
   printf("Error! No $STR found.\n");
   exit(1);
 }

 iel=0;
 istr=0;
 while(fgets(cline,1024,fp)!=NULL) {
   if (strstr(cline,"$END")!=NULL)
     break;
   nsplit=0;
   split_string(cline,split,1024,&nsplit);
   if (readcoef==0)
     expand_str_orb(split,str_tmp,nsplit,&iel);
   else
     expand_str_orb(split,str_tmp,nsplit-1,&iel);
   if (iel>=nel) {
     memcpy(ntstr+istr*nel,str_tmp,nel*sizeof(int));
     if (readcoef>0)
       *(col+istr)=atof(split[nsplit-1]);
     istr++;
     iel=0;
   }
   else if (readcoef>0) {
     printf("Error in reading VB structures with coefficients. Not enough number of electrons read in one line.\n");
     exit(1);
   }
  //  if (istr>=nstr)
  //    break;
 }

//  if (istr<nstr) {
//    printf("Error in reading VB structures. Not enough number of structures read.\n");
//    exit(1);
//  }

  *nstr = istr; 

  free_string_array(split);
 fclose(fp);

  return 0;
}
