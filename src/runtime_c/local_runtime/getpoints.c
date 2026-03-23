#include <stdio.h>
#include <stdlib.h>
#include "vb/vb.h"
#include "mol/mol.h"
#include "inpout/input.h"

void print_background_points(vb_info vb_str) {

  printf("\n");
  printf("       Point Charges Read:\n");
  printf("       No.      Coord. (Bohr)             Charge\n");
  printf("              X         Y         Z\n");

  for (int i=0;i<vb_str->npoints;i++)
    printf("     %3d %9.5f %9.5f %9.5f %9.3f\n",i+1, vb_str->points[4*i+0], vb_str->points[4*i+1], vb_str->points[4*i+2], vb_str->points[4*i+3]);
  printf("\n");

  return;
}

void getpoints(char *inpname, vb_info vb_str, int print_level) {

  char cline[1024];
  char** split=malloc_string_array(1024,1024);

  FILE *fp=fopen(inpname,"r");
  if(fvbsec(fp, "$POINT")>0) {
    printf("Error! no $POINT group found.\n");
    exit(1);
  }

  int nsplit, np;
  double coef;
  if (fgets(cline,1024,fp)!=NULL) {
    nsplit=0;
    split_string(cline,split,1024,&nsplit);
    if (nsplit<2) {
      printf("Error! Not enough part in the first line of $POINT.\n");
      exit(1);
    }
    if (strstr(split[0],"ANGS")!=NULL)
      coef=1e0/A2AU;
    else
      coef=1e0;
    vb_str->npoints=atoi(split[1]);
    vb_str->points=(double*)malloc(sizeof(double)*4*vb_str->npoints);
    np=0;
    while(fgets(cline,1024,fp)!=NULL) {
      if (strstr(cline,"$END")!=NULL) {
        printf("Error! Not enough number of point charges.\n");
        exit(1);
      }
      nsplit=0;
      split_string(cline,split,1024,&nsplit);
      vb_str->points[4*np+0]=atof(split[0])*coef;
      vb_str->points[4*np+1]=atof(split[1])*coef;
      vb_str->points[4*np+2]=atof(split[2])*coef;
      if (nsplit==3)
        vb_str->points[4*np+3]=1e0;
      else
        vb_str->points[4*np+3]=atof(split[3]);
      np++;
      if (np==vb_str->npoints)
        break;
    }
    if (np!=vb_str->npoints) {
      printf("Error! Required %d point charges but read %d.\n",vb_str->npoints,np);
      exit(1);
    }
  }

  fclose(fp);

  if (print_level>0)
    print_background_points(vb_str);

  return;
}

void setpoints_py(double *p, int npoint, vb_info vb_str, int print_level) {
  vb_str->npoints=npoint;
  vb_str->points=(double*)malloc(sizeof(double)*4*npoint);
  memcpy(vb_str->points,p,sizeof(double)*4*npoint);

  if (print_level>0)
    print_background_points(vb_str);

  return;
}
