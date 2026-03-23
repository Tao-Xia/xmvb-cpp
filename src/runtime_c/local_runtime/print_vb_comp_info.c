#include <stdio.h>
#include <stdlib.h>
#include "vb/vb.h"

void print_vb_comp_info(vb_info vb_str) {

  if (vb_str->dobovb>0) {
    printf(" BOVB algorithm: TDM-VBSCF with L-BFGS.\n");
    if (vb_str->ncor>0) {
      printf(" Number of BOVB core orbitals (no breathing): %d\n",vb_str->ncor);
    }
  }
  else if (vb_str->dovbscf>0) {
    printf(" VBSCF algorithm: ");
    if (vb_str->iscf==TDM_SCF) {
      printf("TDM-VBSCF with L-BFGS.\n");
    }
    else if (vb_str->iscf==RDM_SCF) {
      if (vb_str->biovb>0) {
        printf("Tensor-based VBSCF with L-BFGS.\n");
      }
      else {
        printf("RDM-VBSCF with L-BFGS.\n");
      }
    }
    else if (vb_str->iscf==HES_RDM_SCF) {
      printf("RDM-VBSCF with Hessian.\n");
    }
  }

  printf("\n");
  fflush(stdout);

  printf(" Maximum number of Iterations: %d\n",vb_str->itmax);

  printf("\n");
  if (vb_str->dovbpt2>0) {
    printf(" VBPT2 will be proceeded after VBSCF.\n");
    if (vb_str->ncor>0)
      printf(" Number of frozen cores in VBPT2: %d\n",vb_str->ncor);
  }

  if (vb_str->dovbci>0) {
    if (vb_str->dovbcis>0)
      printf(" VBCIS will be proceeded after VBSCF.\n");
    if (vb_str->dovbcisd>0)
      printf(" VBCISD will be proceeded after VBSCF.\n");
    if (vb_str->dovbcids>0)
      printf(" VBCIDS will be proceeded after VBSCF.\n");
    if (vb_str->ncor>0)
      printf(" Number of frozen cores in VBCI: %d\n",vb_str->ncor);
  }

  int dodfvb = vb_str->DoLamDFVB + vb_str->DohcDFVB + vb_str->DoMsDFVB;
  if (dodfvb>0) {
    printf(" DFVB computation after VBSCF/BOVB: ");
    if (vb_str->DoLamDFVB>0)
      printf("Lambda-DFVB(U)\n");
    if (vb_str->DohcDFVB>0)
      printf("hc-DFVB\n");
    if (vb_str->DoMsDFVB>0)
      printf("MS-DFVB\n");
    printf(" DFT functional in DFVB: %s\n",vb_str->DFVBfunc);
  }
  printf("\n");
  fflush(stdout);

  printf(" Integral evaluation: ");
  if (vb_str->inttyp==INT_CINT) {
    printf("precise integrals by Libcint.\n");
    printf(" 2-e integral strategy: ");
    if (vb_str->dir2e==INT2E_NB3) {
      printf("Descrete with values and indices.\n");
    }
    else if (vb_str->dir2e==INT2E_1D) {
      printf("Continuous storage.\n");
    }
    else if (vb_str->dir2e==INT2E_DIR) {
      printf("Direct algorithm without store 2-e integrals permanently.\n");
    }
  }
  else if (vb_str->inttyp==INT_RI) {
    printf("evaluated by RI.\n");
  }
  else if (vb_str->inttyp==INT_COSX) {
    printf("evaluated by RI and COSX.\n");
  }
  else if (vb_str->inttyp==INT_READ) {
    printf("read from files.\n");
  }
  printf("\n");
  fflush(stdout);

  return;
}

