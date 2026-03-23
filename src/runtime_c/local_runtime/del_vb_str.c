#include <stdio.h>
#include <stdlib.h>
//#include <malloc.h>
#include "vb/vb.h"
#include "dfvb/dfvb.h"

int del_vb_str(vb_info vb_str) {
  if ((vb_str->dovbscf+vb_str->dobovb)>0) {

    if (vb_str->nfroz>0)
      free(vb_str->froz_list);

    if (vb_str->if_finnish!=NULL)
      free(vb_str->if_finnish);

    if (vb_str->npoints>0) {
      free(vb_str->points);
      free(vb_str->pcf);
    }

    free(vb_str->ntstr);
    free(vb_str->blocks);
    free(vb_str->noc_block);
    free(vb_str->mx_block);
    free(vb_str->dv);
    free(vb_str->nv);
    free(vb_str->ma0);
    free(vb_str->ma);
    free(vb_str->cvic);
    free(vb_str->col);
    free(vb_str->hvb);
    free(vb_str->svb);
    if (vb_str->dovbci>0) {
      free(vb_str->istr);
      free(vb_str->ss);
      free(vb_str->hh);
      free(vb_str->gg);
      free(vb_str->ek);
      free(vb_str->xx);
      free(vb_str->yy);
      free(vb_str->zz);
    }
    if (vb_str->indxcx!=NULL)
      free(vb_str->indxcx);
    if (vb_str->inor!=NULL)
      free(vb_str->inor);
    if (vb_str->ioor!=NULL)
      free(vb_str->ioor);
    if (vb_str->iscf==TDM_SCF) {
      free(vb_str->detpair);
      free(vb_str->idxpair);
    }
    if (vb_str->dobovb>0) {
      free(vb_str->nbostr);
      free(vb_str->indxbovb);
    }
    free(vb_str->xxf);
    free(vb_str->yyf);
    free(vb_str->zzf);
    if (vb_str->inttyp==INT_CINT) {
      free(vb_str->env);
      free(vb_str->atm);
      free(vb_str->bas);
      free(vb_str->basidx);
      free(vb_str->ssf);
      free(vb_str->hhf);
      free(vb_str->ekf);
      free(vb_str->snorm);
      if (vb_str->dir2e<=0) {
        free(vb_str->ggf);
        free(vb_str->g2eidx);
        if (vb_str->iscf==RDM_SCF || vb_str->dovbci>0) {
          if (vb_str->dir2e==INT2E_1D)
            free(vb_str->g1d);
          else {
            free(vb_str->nb3idx);
            free(vb_str->nb3num);
          }
        }
      }
    }
    else if (vb_str->inttyp==INT_READ) {
      free(vb_str->ssf);
      free(vb_str->hhf);
      free(vb_str->ekf);
      free(vb_str->snorm);
      free(vb_str->ggf);
      free(vb_str->g2eidx);
      free(vb_str->zan);
      free(vb_str->crd);
      free(vb_str->limsup);
      free(vb_str->limlow);
      free(vb_str->ele_tag);
      free(vb_str->bas_tag);
      free(vb_str->hfwfn);
      free(vb_str->ss0);
      free(vb_str->cmat);
      if (vb_str->iscf==RDM_SCF || vb_str->dovbci>0) {
        if (vb_str->dir2e==INT2E_1D)
          free(vb_str->g1d);
        else if (vb_str->dir2e==INT2E_NB3){
          free(vb_str->nb3idx);
          free(vb_str->nb3num);
        }
      }
    }
    else {
      vb_str->ssf=NULL;
      vb_str->hhf=NULL;
      free(vb_str->env);
      free(vb_str->atm);
      free(vb_str->bas);
      free(vb_str->basidx);
      free(vb_str->ssf_norm);
      if (vb_str->iscf==RDM_SCF) {
        free(vb_str->ij_aux_k);
      }
      if (vb_str->iscf==TDM_SCF) {
        free_matrix(vb_str->tdm_ij_k);
        free_matrix(vb_str->tdm_k_ij);
      }
      if (vb_str->cosx_pros==1) {
        free(vb_str->g22);
        free(vb_str->gxx);
        free(vb_str->taux);
        free(vb_str->ti);
        if(vb_str->inttyp==INT_FCOSX)
          free_tensor3d(vb_str->J_K_last);
      }
    }
  }
  if (vb_str->readcoef>0)
    free(vb_str->fixcol);
  if (vb_str->ngroup>1) {
    free(vb_str->grpidx);
    free(vb_str->grplist);
  }

  if (vb_str->nvar>0) {
    free(vb_str->nsymc);
    free(vb_str->nrep);
    free(vb_str->indxxc);
  }

  if (vb_str->DohcDFVB>0 || vb_str->DoLamDFVB>0 || vb_str->DoMsDFVB>0)
    del_dfvb_str(vb_str->dfvb_str);

  if (vb_str->DohcDFVB>0)
    free(vb_str->dfvb2_ec);

  if (vb_str->DoMsDFVB>0){
    free_matrix(vb_str->cad_eff); 
    free_matrix(vb_str->ms_dfvb_eff); 
    free_matrix(vb_str->ms_dfvb_col_tmp); 
    free_tensor3d(vb_str->ms_dfvb_weight); 
  }

  free(vb_str->aden);
  free(vb_str->bden);
  free(vb_str->aden0);
  free(vb_str->bden0);

  free(vb_str);
  return 0;
}
