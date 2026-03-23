#include <stdio.h>
#include <string.h>
#include "mol/mol.h"
#include "inpout/input.h"
#include "inpout/output_func.h"
#include "vb/vb.h"

// int readinp_vb(mol_info mol,inp_info inp_str, vb_info vb_str,para_info par_str,char *inpname) {
vb_info readinp_vb(mol_info mol,inp_info inp_str, para_info par_str,char *inpname) {

  vb_info vb_str=(vb_info)malloc(sizeof(struct VbInfo));

  init_vb_param(mol,inp_str,vb_str);

  if (vb_str->inttyp==INT_READ) {
    print_crd(NULL,vb_str,NULL);
  }

  memset(vb_str->file_name,'\0',sizeof(vb_str->file_name));
  strcpy(vb_str->file_name,inpname);
//  memset(vb_str->file_name,0,sizeof(vb_str->file_name));
//  char *inp=strstr(inpname,".inp");
//  char *xmi=strstr(inpname,".xmi");
//  int nlen=strlen(inpname);
//  if (inp!=NULL && xmi!=NULL) {
//    int leninp=strlen(inp);
//    int lenxmi=strlen(xmi);
//    if (leninp>lenxmi) {
//      strncpy(vb_str->file_name,inpname,nlen-leninp);
//    }
//    else {
//      strncpy(vb_str->file_name,inpname,nlen-lenxmi);
//    }
//  }
//  else if (inp!=NULL) {
//    strncpy(vb_str->file_name,inpname,nlen-strlen(inp));
//  }
//  else if (xmi!=NULL) {
//    strncpy(vb_str->file_name,inpname,nlen-strlen(xmi));
//  }
//  else {
//    strncpy(vb_str->file_name,inpname,nlen);
//  }
  strncpy(vb_str->xdat_name,vb_str->file_name,strlen(vb_str->file_name));
  strcat(vb_str->xdat_name,".xdat");

  if (inp_str->dowfn>0)
    if (strlen(inp_str->wfn_name)>0)
      strncpy(vb_str->wfn_name,inp_str->wfn_name,strlen(inp_str->wfn_name));
    else
      strncpy(vb_str->wfn_name,vb_str->file_name,strlen(vb_str->file_name));

  getstr(inp_str->inpname,inp_str->print_level,vb_str);

  getorb(inp_str->inpname,vb_str);
  
  if (vb_str->read_points>0)
    getpoints(inp_str->inpname,vb_str,inp_str->print_level);

  if (vb_str->orbtyp==HAO_TYP)
    getfrg(inp_str->inpname,mol,vb_str);

  if ((vb_str->dovbscf+vb_str->dobovb)>0 && vb_str->orb_with_symm==0) {
    for (int i=0;i<vb_str->nor;i++)
      if (*(vb_str->ma0+i)>0)
        for (int j=0;j<*(vb_str->ma+i);j++)
          *(vb_str->cvic+i*vb_str->nb+j)=i+j/1e3;
  }
  
  // vb_str->iguess = detect_guess(inp_str->inpname); 
  int guess_tmp = 0; 
  guess_tmp = detect_guess(inp_str->inpname); 
  if (guess_tmp != 0) {
    if (vb_str->iguess == 0) {
      vb_str->iguess = guess_tmp; 
    }
    else {
      if (vb_str->iguess != guess_tmp && vb_str->iguess!=GUS_NBO) {
        printf("\n\n$GUS section doesn't match with keyword 'GUESS'\n"); 
        fflush(stdout); 
        char *readguess = (char*)malloc(sizeof(char)*8);
        char *detectguess = (char*)malloc(sizeof(char)*8); 
        memset(readguess, 0, sizeof(readguess)); 
        memset(detectguess, 0, sizeof(detectguess)); 
        switch(vb_str->iguess) {
          case GUS_AUTO: 
            readguess = "AUTO"; 
            break; 
          case GUS_UNIT: 
            readguess = "UNIT"; 
            break; 
          case GUS_READ: 
            readguess = "READ"; 
            break; 
          case GUS_RDCI: 
            readguess = "RDCI"; 
            break; 
          case GUS_MO: 
            readguess = "MO"; 
            break; 
          case GUS_NBO: 
            readguess = "NBO"; 
            break; 
        }
        switch(guess_tmp) {
          case GUS_AUTO: 
            detectguess = "AUTO"; 
            break; 
          case GUS_UNIT: 
            detectguess = "UNIT"; 
            break; 
          case GUS_READ: 
            detectguess = "READ"; 
            break; 
          case GUS_RDCI: 
            detectguess = "RDCI"; 
            break; 
          case GUS_MO: 
            detectguess = "MO"; 
            break; 
          case GUS_NBO: 
            detectguess = "NBO"; 
            break; 
        }
        printf("Reading 'GUESS=%s' but $GUS section detected 'GUESS=%s'\n", readguess, detectguess); 
        printf("Please correct the input file\n");
        fflush(stdout); 
        exit(1); 
      }
    }
  }


  return vb_str;
//  return 0;
}
