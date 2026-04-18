#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mol/mol.h"
#include "inpout/input.h"
#include "vb/vb.h"
#include "runtime_c/local_runtime/cint_compat.h"

int init_vb_param(mol_info mol, inp_info inp_str, vb_info vb_str) {

  memset(vb_str->aux_name,0,sizeof(vb_str->aux_name));
  memset(vb_str->k_grid_file,0,sizeof(vb_str->k_grid_file));
  memset(vb_str->k_grid_file_final,0,sizeof(vb_str->k_grid_file_final));
//  strncpy(vb_str->aux_name,inp_str->aux_name,strlen(inp_str->aux_name));
  strcpy(vb_str->aux_name,inp_str->aux_name);
  strcpy(vb_str->k_grid_file,inp_str->k_grid_file);
  strcpy(vb_str->k_grid_file_final,inp_str->k_grid_file_final);

//  char *comment;
//  comment=getenv("VBDIR");
//  strcpy(vb_str->k_grid_file,comment);
//  strcat(vb_str->k_grid_file,"/data/kgrids.opt");
//  strcpy(vb_str->k_grid_file_final,vb_str->k_grid_file);


//  strncpy(vb_str->k_grid_file,inp_str->k_grid_file,strlen(inp_str->k_grid_file));
//  strncpy(vb_str->k_grid_file_final,inp_str->k_grid_file,strlen(inp_str->k_grid_file));

  vb_str->iw = 6;
  vb_str->ier = 0;

  vb_str->if_finnish=NULL;

  if (inp_str->nb > 0)
    vb_str->nb=inp_str->nb;
  else if (inp_str->inttyp!=INT_READ)
    vb_str->nb=mol->bas->msize;
  else
    vb_str->nb=0;

  vb_str->nor=inp_str->nor;

  if (inp_str->nel > 0)
    vb_str->nel=inp_str->nel;
  else if (inp_str->inttyp!=INT_READ)
    vb_str->nel=mol->atm->alpha_num+mol->atm->beta_num;

  if (inp_str->inttyp==INT_READ) {
    readint_head(vb_str);
    readinfo(vb_str);
  }

  vb_str->vbsym_d=0;
  vb_str->vbsym_igrd=0;
  vb_str->vbsym_ind=0;

  vb_str->dovbscf=inp_str->dovbscf;
  vb_str->dobovb=inp_str->dobovb;
  vb_str->dovbcis=inp_str->dovbcis;
  vb_str->dovbcisd=inp_str->dovbcisd;
  vb_str->dovbcids=inp_str->dovbcids;
  vb_str->dovbpt2=inp_str->dovbpt2;
  vb_str->dogopt = inp_str->DoGopt;
  vb_str->dogradient = inp_str->DoGradient;
  vb_str->iroot=inp_str->iroot;
  vb_str->iscf=inp_str->iscf;
  vb_str->orbtyp=inp_str->orbtyp;
  vb_str->frgtyp=inp_str->frgtyp;
  vb_str->wfntyp=inp_str->wfntyp;
  vb_str->vbftyp=inp_str->vbftyp;
  vb_str->preserve_explicit_sparse_orbital_layout=0;
  vb_str->itmax=inp_str->itmax;
  vb_str->boysloc=inp_str->boysloc;
  vb_str->pmloc=inp_str->pmloc;
  vb_str->localize=vb_str->pmloc+vb_str->boysloc;
  vb_str->nmul=inp_str->nmul;
  vb_str->genstr=inp_str->genstr;
  vb_str->nstr=inp_str->nstr;
  vb_str->ndet=0;
  vb_str->iomin = 0; 
  vb_str->iomax = 0; 
  vb_str->iguess = inp_str->iguess;
  vb_str->dir2e = inp_str->dir2e;
  vb_str->inttyp = inp_str->inttyp;
  vb_str->ncor = inp_str->ncor;
  vb_str->cicut = inp_str->cicut;
  vb_str->inci = inp_str->inci;
  vb_str->nae = inp_str->nae;
  vb_str->nao = inp_str->nao;
  vb_str->dobfi = inp_str->dobfi;
  vb_str->readcoef=inp_str->fixc;
  vb_str->vmax = inp_str->vmax;

  vb_str->read_points=inp_str->read_points;
  vb_str->npoints=0;
  vb_str->pcf=NULL;
  vb_str->enuc_pn=0e0;
  vb_str->enuc_pp=0e0;

  if (inp_str->inttyp!=INT_READ)
    vb_str->natom=mol->atm->natm;

  vb_str->epg=1e-7;
  if (vb_str->iscf==HES_RDM_SCF)
    vb_str->gpg=1e-4;
  else
    vb_str->gpg=2e-3;
  

//  printf("biovb = %d\n",inp_str->biovb);

  vb_str->biovb = inp_str->biovb;
  vb_str->bio_readcoef = inp_str->bio_readcoef;

  vb_str->dopop=inp_str->dopop;
  vb_str->dowfn=inp_str->dowfn;

  vb_str->DoLamDFVB = inp_str->DoLamDFVB; 
  vb_str->DohcDFVB = inp_str->DohcDFVB; 
  vb_str->DoMsDFVB = inp_str->DoMsDFVB; 
  if ((vb_str->DoLamDFVB > 0) || (vb_str->DohcDFVB > 0) || (vb_str->DoMsDFVB > 0)) {
    memset(vb_str->DFVBfunc,0,sizeof(vb_str->DFVBfunc));
    strncpy(vb_str->DFVBfunc,inp_str->DFVBfunc,strlen(inp_str->DFVBfunc));
  }
  vb_str->dfvb2_ec = NULL; 
  vb_str->cad_eff = NULL; 
  vb_str->ms_dfvb_eff = NULL; 
  vb_str->ms_dfvb_col_tmp = NULL; 
  vb_str->ms_dfvb_weight = NULL; 
  vb_str->dfvb_str=NULL;

  // vbcad 
  vb_str->vbcad = inp_str->vbcad; 
  vb_str->cad_step_size = inp_str->cad_step_size; 

  vb_str->norg=1;
  memset(vb_str->ntorg,0,sizeof(int)*21);

  if (vb_str->readcoef>0) {
    if (strlen(inp_str->grpval)>0) {
      for (int i=0;i<strlen(inp_str->grpval);i++) {
        if (inp_str->grpval[i]==',' && inp_str->grpval[i+1]==',') {
          inp_str->grpval[i]=' ';
          inp_str->grpval[i+1]=' ';
          i++;
        }
      }
      char** split=malloc_string_array(1024,1024);

      vb_str->ngroup=0;
      split_string(inp_str->grpval,split,1024,&vb_str->ngroup);
      vb_str->grpidx=(int*)malloc(sizeof(int)*vb_str->ngroup);
      vb_str->grplist=(int*)malloc(sizeof(int)*vb_str->nstr*vb_str->ngroup);
      char** split1=malloc_string_array(1024,1024);
      for (int i=0;i<vb_str->ngroup;i++) {
        for (int j=0;j<strlen(split[i]);j++)
          if (split[i][j]==',')
            split[i][j]=' ';
        *(vb_str->grpidx+i)=0;
        split_string(split[i],split1,1024,vb_str->grpidx+i);
        for (int j=0;j<*(vb_str->grpidx+i);j++)
          *(vb_str->grplist+i*vb_str->nstr+j)=(int)atoi(split1[j])-1;
      }
      free_string_array(split);
      free_string_array(split1);
    }
    else {
      vb_str->ngroup=1;
      vb_str->grpidx=NULL;
      vb_str->grplist=NULL;
    }
  }
  else {
    vb_str->ngroup=0;
    vb_str->grpidx=NULL;
    vb_str->grplist=NULL;
  }

  if (inp_str->nsav>0) {
    vb_str->nsav = inp_str->nsav;
    memcpy(vb_str->idxstate,inp_str->idxstate,sizeof(int)*vb_str->nsav);
    memcpy(vb_str->wstate,inp_str->wstate,sizeof(double)*vb_str->nsav);
    double totw = 0e0;
    for (int i=0;i<vb_str->nsav;i++)
      totw+=vb_str->wstate[i];
    for (int i=0;i<vb_str->nsav;i++)
      vb_str->wstate[i]/=totw;

  }
  else {
    vb_str->nsav=1;
    vb_str->idxstate[0]=vb_str->iroot;
    vb_str->wstate[0]=1.0;
  }

  if (vb_str->dobfi==0)
    vb_str->npb=vb_str->nb;

  vb_str->nsymc=NULL;
  vb_str->nrep=NULL;
  vb_str->str_ori=NULL;

  vb_str->indxcx=NULL;
  vb_str->indxxc=NULL;
  vb_str->ioor=NULL;
  vb_str->inor=NULL;
  vb_str->cvic=NULL;

  vb_str->ntstr=NULL;

  memcpy(vb_str->strclass,inp_str->strclass,sizeof(vb_str->strclass));

  vb_str->nv=NULL;
  vb_str->ma0=NULL;
  vb_str->ma=NULL;
  vb_str->nvic=NULL;
  vb_str->dv=NULL;

  vb_str->blocks=NULL;
  vb_str->noc_block=NULL;
  vb_str->mx_block=NULL;
  vb_str->nblock=0;
  vb_str->block_part_ov=0;

  vb_str->g1d=NULL;
  vb_str->ggf=NULL;
  vb_str->g2eidx=NULL;
  vb_str->n2e=0;
  vb_str->nb3num=NULL;
  vb_str->nb3idx=NULL;
  vb_str->istr=NULL;
  vb_str->hh=NULL;
  vb_str->ss=NULL;
  vb_str->gg=NULL;
  vb_str->xx=NULL;
  vb_str->yy=NULL;
  vb_str->zz=NULL;
  vb_str->ek=NULL;
  vb_str->ncore=0;
  vb_str->nbf=vb_str->nb;

  if (vb_str->inttyp==INT_READ) {
    vb_str->ssf=NULL;
    vb_str->hhf=NULL;
    vb_str->xxf=NULL;
    vb_str->yyf=NULL;
    vb_str->zzf=NULL;
    vb_str->ekf=NULL;
    vb_str->cmat=NULL;
    vb_str->ss0=NULL;
    vb_str->hfwfn=NULL;
  }
  else {
//  if (vb_str->inttyp==INT_CINT) {
//    vb_str->ngto=get_ngto(mol);
//    vb_str->nshell=mol->bas->nbas;
//    vb_str->env=(double*)malloc((vb_str->ngto*2+mol->atm->natm*3+PTR_ENV_START)*sizeof(double));
//    vb_str->atm=(int*)malloc(ATM_SLOTS*mol->atm->natm*sizeof(int));
//    vb_str->bas=(int*)malloc(BAS_SLOTS*mol->bas->nbas*sizeof(int));
//    vb_str->basidx=(int*)malloc(2*mol->bas->nbas*sizeof(int));
//    int_data_trans(mol,vb_str->atm,vb_str->bas,vb_str->basidx,vb_str->env,vb_str->ngto);
//    int_data_trans(mol,vb_str);
    if (vb_str->inttyp!=INT_CINT) {
      vb_str->ssf=NULL;
      vb_str->hhf=NULL;
      vb_str->xxf=NULL;
      vb_str->yyf=NULL;
      vb_str->zzf=NULL;
      vb_str->ekf=NULL;
      vb_str->snorm=NULL;
//      vb_str->atm=NULL;
//      vb_str->bas=NULL;
//      vb_str->env=NULL;
//      vb_str->basidx=NULL;
    }
  }

  vb_str->ndetpair=0;
  vb_str->detpair=NULL;
  vb_str->idxpair=NULL;

  vb_str->col=NULL;
  vb_str->hvb=NULL;
  vb_str->svb=NULL;

  vb_str->nbostr=NULL;
  vb_str->indxbovb=NULL;
  vb_str->inactbo=NULL;

  vb_str->dovb=vb_str->dovbscf+vb_str->dobovb;
  vb_str->dovbci=vb_str->dovbcis+vb_str->dovbcisd+vb_str->dovbcids;

  vb_str->nfroz = inp_str->nfroz; 
  if (vb_str->nfroz != 0) {
    vb_str->froz_list = (int *)malloc(vb_str->nfroz*sizeof(int)); 
    memcpy(vb_str->froz_list,inp_str->froz_list,vb_str->nfroz*sizeof(int)); 
  }
  else {
    vb_str->froz_list = NULL; 
  }

  vb_str->sort = inp_str->sort;
  vb_str->ctol = inp_str->ctol;

  vb_str->aden=NULL;
  vb_str->bden=NULL;
  vb_str->aden0=NULL;
  vb_str->bden0=NULL;


  return 0;
}
