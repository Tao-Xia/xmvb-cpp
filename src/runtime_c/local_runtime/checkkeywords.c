#define checkkeywords xmvb_cpp_checkkeywords

#include <stdio.h>
#include <stdlib.h>
#include "scf/hf.h"
#include "inpout/input.h"
#include "vb/vb.h"
#include "gopt/gopt.h"

int check_sav_info(int *idxstate, double *wstate, int *nsav, int nstr) {
    int minav=1;
    int maxav=0;
    for (int i=0;i<*nsav;i++) {
        if (minav > *(idxstate+i))
            minav = *(idxstate+i);
        if (maxav < *(idxstate+i))
            maxav = *(idxstate+i);
    }
    if (minav<1) {
        printf("Error! state number cannot be smaller than 1 (ground state).\n");
        exit(1);
    }
    if (maxav>nstr) {
        printf("Error! state number cannot be larger than number of structures.\n");
        exit(1);
    }
    int idx_redund[nstr];
    double w_redund[nstr];
    memset(idx_redund,0,sizeof(int)*nstr);
    memset(w_redund,0,sizeof(double)*nstr);

    for (int i=0;i<*nsav;i++) {
        idx_redund[*(idxstate+i)-1]++;
        if (idx_redund[*(idxstate+i)-1]>1 && *(wstate+i)!= w_redund[*(idxstate+i)-1]) {
            printf("Error! Redundent state weight for state %d with different weights %f and %f\n",idxstate[i],wstate[i],w_redund[*(idxstate+i)-1]);
            exit(1);
        }
        if (*(wstate+i)<0.0) {
            printf("Error! Weight of state cannot be negative.\n");
            exit(1);
        }
        w_redund[*(idxstate+i)-1]=*(wstate+i);
    }

    *nsav=0;
    memset(idxstate,0,sizeof(int)*200);
    memset(wstate,0,sizeof(double)*200);

    double totalweight=0.0;
    for (int i=0;i<nstr;i++) {
        if (idx_redund[i]>0) {
            *(idxstate+(*nsav))=i;
            *(wstate+(*nsav))=w_redund[i];
            totalweight+=w_redund[i];
            (*nsav)++;
        }
    }

    for (int i=0;i<*nsav;i++)
        *(wstate+i)/=totalweight;

    return 0;
}

int checkkeywords(inp_info inp_str) {

  if (inp_str->ihf_type==-2)
    if (inp_str->nmul==1)
      inp_str->ihf_type=RHF_WORK;
    else
      inp_str->ihf_type=UHF_WORK;
  if (inp_str->itmax<0) {
    if (inp_str->dovb>0)
      inp_str->itmax=200;
    else
      inp_str->itmax=100;
  }

  if (inp_str->dobovb>0) {
    if (inp_str->iscf==0)
      inp_str->iscf=TDM_SCF;
    else if (inp_str->iscf!=TDM_SCF) {
      printf("Error! ISCF=%d cannot be used for BOVB.\n",inp_str->iscf);
      exit(1);
    }
  }
  else if (inp_str->dovbscf>0 && inp_str->iscf==0) {
    inp_str->iscf=RDM_SCF;
  }

  if (inp_str->pmloc>0 && inp_str->inttyp==INT_READ) {
      printf("P-M localization can not be proceeded with INT=READ.\n");
      exit(1);
  }

  int dovbci=inp_str->dovbcis+inp_str->dovbcisd+inp_str->dovbcids;
  if (dovbci>0) {
    if (inp_str->inttyp>=INT_XINT && inp_str->inttyp!=INT_READ) {
      printf("VBCI can be proceeded only with INT=LIBCINT or INT=READ\n");
      exit(1);
    }
    if (inp_str->nsav > 1) {
        printf("VBCI can not be proceeded with keyword WSTATE.\n");
        exit(1);
    }
  }

  int dovb=inp_str->dovbscf+inp_str->dobovb;

  if (dovb>0) {
    if (inp_str->dobovb>0 && (dovbci>0 || inp_str->dovbpt2>0)) {
      printf("Error! BOVB cannot be proceeded with VBCI or VBPT2.\n");
        exit(1);
    }
    if (dovbci>1) {
      printf("Error! Multiple VBCI types (VBCIS, VBCISD) assigned.\n");
      exit(1);
    }
    if (dovbci>0 && inp_str->dovbpt2>0) {
      printf("Error! Both VBPT2 and VBCI are assigned.\n");
      exit(1);
    }
  }

  int dodfvb = inp_str->DoLamDFVB + inp_str->DohcDFVB + inp_str->DoMsDFVB;
  if (inp_str->inttyp==INT_READ && dodfvb>0) {
    printf("Error! DFVB cannot be proceeded with INT=READ.\n");
    exit(1);
  }

//    if (inp_str->nsav > 0) {
//        check_sav_info(inp_str->idxstate,inp_str->wstate,&inp_str->nsav,inp_str->nstr);
//    }

    // lambda-DFVB
    if (inp_str->DoLamDFVB == 1) {
        if (inp_str->dovbscf == 0) 
            inp_str->dovbscf = 1; 
        
        if (!((strlen(inp_str->DFVBfunc) == 4) && (strstr(inp_str->DFVBfunc, "BLYP") != NULL))) {
            printf("Lambda-DFVB can only work with BLYP functional now.\n"); 
            exit(1); 
        }

        if (inp_str->nsav > 1) {
            printf("Lambda-DFVB cannot work with WSTATE. If you want to use Lambda-DFVB with state-average calculation, you can try MS-DFVB!\n"); 
            exit(1); 
        }
    }

    // hc-DFVB 
    if (inp_str->DohcDFVB == 1) {
        if (inp_str->dovbscf == 0) 
            inp_str->dovbscf = 1; 
        
        if ((!((strlen(inp_str->DFVBfunc) == 4) && (strstr(inp_str->DFVBfunc, "BLYP")    != NULL))) && 
            (!((strlen(inp_str->DFVBfunc) == 5) && (strstr(inp_str->DFVBfunc, "B3LYP")   != NULL))) &&
            (!((strlen(inp_str->DFVBfunc) == 6) && (strstr(inp_str->DFVBfunc, "BHHLYP")  != NULL))) &&
            (!((strlen(inp_str->DFVBfunc) == 4) && (strstr(inp_str->DFVBfunc, "PW91")    != NULL))) &&
            (!((strlen(inp_str->DFVBfunc) == 4) && (strstr(inp_str->DFVBfunc, "PBE0")    != NULL))) &&
            (!((strlen(inp_str->DFVBfunc) == 3) && (strstr(inp_str->DFVBfunc, "PBE")     != NULL))) &&
            (!((strlen(inp_str->DFVBfunc) == 7) && (strstr(inp_str->DFVBfunc, "B3LYPV5") != NULL))) )
        {
            printf("hc-DFVB can only work with BLYP, B3LYP, BHHLYP, PW91, PBE0, PBE functional now.\n"); 
            exit(1); 
        }
    }

    // MS-DFVB 
    if (inp_str->DoMsDFVB == 1) {
        if (inp_str->dovbscf == 0) 
            inp_str->dovbscf = 1; 
        
        if (!((strlen(inp_str->DFVBfunc) == 4) && (strstr(inp_str->DFVBfunc, "BLYP") != NULL))) {
            printf("MS-DFVB can only work with BLYP functional now.\n"); 
            exit(1); 
        }

        if (inp_str->nsav<2) {
          printf("Error! MS-DFVB must be proceeded with WSTATE.\n");
          exit(1);
        }
        else if (inp_str->nsav>2) {
          printf("Error! MS-DFVB can only work with 2 states now.\n");
          exit(1);
        }

    }

    // check the number of DFVB 
    if ((inp_str->DoLamDFVB + inp_str->DohcDFVB + inp_str->DoMsDFVB) > 1) {
        printf("Only one of the DFVB method can be used at a time.\n"); 
        exit(1); 
    }

    // geometry optimization 
    if (inp_str->DoGopt == 1){
        if (inp_str->dovbscf == 0) {
            printf("Geometry optimization can only work with VBSCF method currently. \n"); 
            exit(1); 
        }
        
        if (inp_str->Gopt_Type == GOPT_ZMT) {
            printf("Geometry optimization cannot be performed in Z-matrix currently. Please change the keyword into 'GOPT' or 'GOPT=CART'. \n"); 
            exit(1); 
        }

        if (inp_str->nneg == 1) {
            printf("Optimization to a transition state is not available. \n"); 
            exit(1); 
        }

        if (inp_str->DoGradient == 0) {
            inp_str->DoGradient = 1; 
        }

        // Check coordinate
//        if (inp_str->Gopt_Type == GOPT_ZMT) {
//            printf("Geometry optimization currently doesn't support ZMT yet."); 
//            printf("Change to Cartesian coordinates."); 
//            inp_str->Gopt_Type = GOPT_CART; 
//        }

        // Check Hessian type
        if (inp_str->Hess_Type == GHESS_ANALYTICAL) {
            if (!((inp_str->dovbscf == 1) && ((inp_str->iscf == RDM_SCF) || (inp_str->iscf == HES_RDM_SCF)))) {
                printf("Analytical Hessian current only supports VBSCF (iscf = 5 or 6)."); 
                printf("Change to fully numerical Hessian."); 
                inp_str->Hess_Type == GHESS_FULLY_NUMERICAL; 
            }
        }
        else if (inp_str->Hess_Type == GHESS_SEMI_NUMERICAL) {
            if (!((inp_str->ihf_type == RHF_WORK) || ((inp_str->dovbscf == 1) && ((inp_str->iscf == RDM_SCF) || (inp_str->iscf == HES_RDM_SCF))))) {
                printf("Semi-numerical Hessian current only supports RHF and VBSCF (iscf = 5 or 6)."); 
                printf("Change to fully numerical Hessian.");
                inp_str->Hess_Type == GHESS_FULLY_NUMERICAL;  
            }
        }

        // Check gradient 
        if (inp_str->Grad_Type == GGRAD_ANALYTICAL) {
            if (!((inp_str->ihf_type == RHF_WORK) || ((inp_str->dovbscf == 1) && ((inp_str->iscf == RDM_SCF) || (inp_str->iscf == HES_RDM_SCF))))) {
                printf("Analytical gradient current only supports RHF and VBSCF (iscf = 5 or 6)."); 
                printf("Change to numerical gradient."); 
                inp_str->Grad_Type == GGRAD_NUMERICAL; 
            }
        }

        // Default coordinate 
        if (inp_str->Gopt_Type == GOPT_NONE) {
            inp_str->Gopt_Type = GOPT_CART; 
        }

        // Default Hessian type 
        if (inp_str->Hess_Type == GHESS_NONE) {
            if ((inp_str->dovbscf == 1) && ((inp_str->iscf == RDM_SCF) || (inp_str->iscf == HES_RDM_SCF))) {
                inp_str->Hess_Type == GHESS_SEMI_NUMERICAL; 
            }
            else {
                inp_str->Hess_Type == GHESS_FULLY_NUMERICAL; 
            }
        }

        // Default gradient type based on Hessian type 
        if (inp_str->Grad_Type == GGRAD_NONE) {
            if (inp_str->Hess_Type == GHESS_GUESS) {
                if ((inp_str->dovbscf == 1) && ((inp_str->iscf == RDM_SCF) || (inp_str->iscf == HES_RDM_SCF))) {
                    inp_str->Grad_Type = GGRAD_ANALYTICAL; 
                }
                else {
                    inp_str->Grad_Type = GGRAD_NUMERICAL; 
                }
            }
            else if ((inp_str->Hess_Type == GHESS_ANALYTICAL) || (inp_str->Hess_Type == GHESS_SEMI_NUMERICAL)) {
                inp_str->Grad_Type = GGRAD_ANALYTICAL; 
            }
            else if (inp_str->Hess_Type == GHESS_FULLY_NUMERICAL) {
                inp_str->Grad_Type = GGRAD_NUMERICAL; 
            }
        }
    }

    // VBCAD 
    if (inp_str->vbcad != 0) {
        if (inp_str->vbcad>5 || inp_str->vbcad<0) {
            printf("Type of Weight used for VBCAD cannot be recognized\n");
            exit(1);
        }

        if (inp_str->nsav<2) {
            printf("Error! VBCAD must be proceeded with WSTATE.\n");
            exit(1);
        }
        else if (inp_str->nsav>2) {
            printf("Error! VBCAD can only work with 2 states now.\n");
            exit(1);
        }

        printf("\nVBCAD is activated with weight type %d\n", inp_str->vbcad); 
        if (inp_str->cad_step_size <= 1e-6) {
            printf("Using default step size of 0.5 degree for VBCAD\n"); 
            inp_str->cad_step_size = 2.0; 
        }
        else if (inp_str->cad_step_size <= (1e0 / 180e0)) {
            printf("The step size of VBCAD sweep is bigger than 180 degree and it is too large!\n"); 
            printf("Please change the step size to be smaller than 180 degree.\n");
            exit(1);
            inp_str->cad_step_size = 2.0; 
        }
        printf("\n"); 
    }

    if (inp_str->inttyp==INT_READ) {
      if (inp_str->molden>0) {
        printf("Error! Molden file cannot be punched with INT=READ.\n");
        exit(1);
      }
      if (inp_str->dovb==0) {
        printf("Error! INT=READ can be used only with VB computation.\n");
        exit(1);
      }
      if (inp_str->DoLamDFVB>0 || inp_str->DohcDFVB>0) {
        printf("Error! DFVB computation cannot be proceeded with INT=READ.\n");
        exit(1);
      }
      if (inp_str->DoGopt>0 || inp_str->DoGradient>0) {
        printf("Error! Geometry optimization or geometric gradient cannot be proceeded with INT=READ.\n");
        exit(1);
      }
      if (inp_str->dir2e>0) {
        printf("Error! Direct algorithm cannot be proceeded with INT=READ.\n");
        exit(1);
      }

    }

    if (inp_str->froz_list != NULL) {
        if ((inp_str->dovbscf  == 0) || 
            (inp_str->dobovb   == 1) || 
            (inp_str->dovbcis  == 1) || 
            (inp_str->dovbcisd == 1) || 
            (inp_str->dovbcids == 1) || 
            (inp_str->dovbpt2  == 1)) {
            printf("Error! Keyword FRZORB can only be used with VBSCF method.\n"); 
            exit(1); 
        }
        else {
            printf("Frozen orbital list: "); 
            for (int i=0;i<inp_str->nfroz;i++) {
                printf("%d ", inp_str->froz_list[i]); 
            }
            printf("\n\n"); 
        }
    }

    return 0;
}
