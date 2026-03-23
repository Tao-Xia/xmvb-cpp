#include <stdio.h>
#include <math.h>
#include "vb/vb.h"
#include "cblas.h"
#include "cint_funcs.h"
#include "inpout/output_func.h"

int cint1e_ovlp_cart(double *buf, int *shls,int *atm, int natm, int *bas, int nbas, double *env);
int cint1e_kin_cart(double *buf, int *shls,int *atm, int natm, int *bas, int nbas, double *env);
int cint1e_nuc_cart(double *buf, int *shls,int *atm, int natm, int *bas, int nbas, double *env);
int cint1e_r_cart(double *buf, int *shls,int *atm, int natm, int *bas, int nbas, double *env);
int cint1e_rinv_cart(double *buf, int *shls,int *atm, int natm, int *bas, int nbas, double *env);

int cint1e_ipovlp_cart(double *buf, int *shls,int *atm, int natm, int *bas, int nbas, double *env);
int cint1e_ipkin_cart(double *buf, int *shls,int *atm, int natm, int *bas, int nbas, double *env);
int cint1e_ipnuc_cart(double *buf, int *shls,int *atm, int natm, int *bas, int nbas, double *env);
int cint1e_iprinv_cart(double *buf, int *shls,int *atm, int natm, int *bas, int nbas, double *env);

int cal_overlap_mat(double *ssf, vb_info vb_str) {

  int nb=vb_str->nb;
  int nsh=vb_str->nshell;

  int maxl=0;
  int i,j;
  for (i=0;i<nsh;i++)
    maxl=(maxl>*(vb_str->basidx+2*i+1)?maxl:*(vb_str->basidx+2*i+1));

  int nbi,nbj,ibi,ibj,k,l,kidx,lidx;
  int shls[2];

  double *buf=(double*)malloc(3*maxl*maxl*sizeof(double));

  for (i=0;i<nsh;i++) {
    shls[0]=i;
    nbi=*(vb_str->basidx+2*i);
    ibi=*(vb_str->basidx+2*i+1);
    for (j=0;j<=i;j++) {
      shls[1]=j;
      nbj=*(vb_str->basidx+2*j);
      ibj=*(vb_str->basidx+2*j+1);
      if (cint1e_ovlp_cart(buf,shls,vb_str->atm,vb_str->natom,vb_str->bas,nsh,vb_str->env)!=0) {
        for (k=0;k<ibj;k++) {
          kidx=k+nbj;
          memcpy(ssf+kidx*nb+nbi,buf+k*ibi,ibi*sizeof(double));
        }
      }
    }
  }

  free(buf);

  return 0;

}

//int cal_dip_kin_mat(double *xxf, double *yyf, double *zzf, double *ekf, vb_info vb_str) {
int cal_dip_mat(double *xxf, double *yyf, double *zzf, vb_info vb_str) {

  int nb=vb_str->nb;
  int nsh=vb_str->nshell;

  int maxl=0;
  int i,j;
  for (i=0;i<nsh;i++)
    maxl=(maxl>*(vb_str->basidx+2*i+1)?maxl:*(vb_str->basidx+2*i+1));

  double *ssf=(double*)malloc(sizeof(double)*nb*nb);
  double *snorm=(double*)malloc(sizeof(double)*nb);
  double *buf=(double*)malloc(3*maxl*maxl*sizeof(double));

  int nbi,nbj,ibi,ibj,k,l,kidx,lidx;
  int shls[2];

  for (i=0;i<nsh;i++) {
    shls[0]=i;
    nbi=*(vb_str->basidx+2*i);
    ibi=*(vb_str->basidx+2*i+1);
    for (j=0;j<=i;j++) {
      shls[1]=j;
      nbj=*(vb_str->basidx+2*j);
      ibj=*(vb_str->basidx+2*j+1);
      if (cint1e_ovlp_cart(buf,shls,vb_str->atm,vb_str->natom,vb_str->bas,nsh,vb_str->env)!=0) {
        for (k=0;k<ibj;k++) {
          kidx=k+nbj;
          memcpy(ssf+kidx*nb+nbi,buf+k*ibi,ibi*sizeof(double));
        }
      }
//      if (cint1e_kin_cart(buf,shls,vb_str->atm,vb_str->natom,vb_str->bas,nsh,vb_str->env)!=0) {
//        for (k=0;k<ibj;k++) {
//          kidx=k+nbj;
//          memcpy(ekf+kidx*nb+nbi,buf+k*ibi,ibi*sizeof(double));
//        }
//      }
      if (cint1e_r_cart(buf,shls,vb_str->atm,vb_str->natom,vb_str->bas,nsh,vb_str->env)!=0) {
        for (k=0;k<ibj;k++) {
          kidx=k+nbj;
          memcpy(xxf+kidx*nb+nbi,buf+k*ibi,ibi*sizeof(double));
          memcpy(yyf+kidx*nb+nbi,buf+ibi*ibj+k*ibi,ibi*sizeof(double));
          memcpy(zzf+kidx*nb+nbi,buf+2*ibi*ibj+k*ibi,ibi*sizeof(double));
        }
      }
    }
  }

  for (i=0;i<nb;i++) {
    snorm[i]=1e0/sqrt(ssf[i*nb+i]);
    for (j=i+1;j<nb;j++) {
//      ekf[j*nb+i]=ekf[i*nb+j];
      xxf[j*nb+i]=xxf[i*nb+j];
      yyf[j*nb+i]=yyf[i*nb+j];
      zzf[j*nb+i]=zzf[i*nb+j];
    }
  }

  double sni,snj;
  for (i=0;i<nb;i++) {
    sni=snorm[i];
    for (j=0;j<nb;j++) {
      snj=sni*snorm[j];
//      ekf[i*nb+j]*=snj;
      xxf[i*nb+j]*=snj;
      yyf[i*nb+j]*=snj;
      zzf[i*nb+j]*=snj;
    }
  }

  free(buf);
  free(ssf);
  free(snorm);

  return 0;

}

int cal_1eint(vb_info vb_str) {

  vb_str->ssf=(double*)malloc(vb_str->nb*vb_str->nb*sizeof(double));
  vb_str->hhf=(double*)malloc(vb_str->nb*vb_str->nb*sizeof(double));
  vb_str->xxf=(double*)malloc(vb_str->nb*vb_str->nb*sizeof(double));
  vb_str->yyf=(double*)malloc(vb_str->nb*vb_str->nb*sizeof(double));
  vb_str->zzf=(double*)malloc(vb_str->nb*vb_str->nb*sizeof(double));
  vb_str->ekf=(double*)malloc(vb_str->nb*vb_str->nb*sizeof(double));
  vb_str->snorm=(double*)malloc(vb_str->nb*sizeof(double));

  double *ssf=vb_str->ssf;
  double *hhf=vb_str->hhf;
  double *xxf=vb_str->xxf;
  double *yyf=vb_str->yyf;
  double *zzf=vb_str->zzf;
  double *ekf=vb_str->ekf;
  double *snorm=vb_str->snorm;
  int nb=vb_str->nb;
  int nsh=vb_str->nshell;

  memset(ssf,0,nb*nb*sizeof(double));
  memset(hhf,0,nb*nb*sizeof(double));
  memset(xxf,0,nb*nb*sizeof(double));
  memset(yyf,0,nb*nb*sizeof(double));
  memset(zzf,0,nb*nb*sizeof(double));
  memset(ekf,0,nb*nb*sizeof(double));
  memset(snorm,0,nb*sizeof(double));

  int maxl=0;
  int i,j;
  for (i=0;i<nsh;i++)
    maxl=(maxl>*(vb_str->basidx+2*i+1)?maxl:*(vb_str->basidx+2*i+1));

  double *buf=(double*)malloc(3*maxl*maxl*sizeof(double));

  int nbi,nbj,ibi,ibj,k,l,kidx,lidx;
  int shls[2];
  for (i=0;i<nsh;i++) {
    shls[0]=i;
    nbi=*(vb_str->basidx+2*i);
    ibi=*(vb_str->basidx+2*i+1);
    for (j=0;j<=i;j++) {
      shls[1]=j;
      nbj=*(vb_str->basidx+2*j);
      ibj=*(vb_str->basidx+2*j+1);
      if (cint1e_ovlp_cart(buf,shls,vb_str->atm,vb_str->natom,vb_str->bas,nsh,vb_str->env)!=0) {
        for (k=0;k<ibj;k++) {
          kidx=k+nbj;
          memcpy(ssf+kidx*nb+nbi,buf+k*ibi,ibi*sizeof(double));
        }
      }
      if (cint1e_kin_cart(buf,shls,vb_str->atm,vb_str->natom,vb_str->bas,nsh,vb_str->env)!=0) {
        for (k=0;k<ibj;k++) {
          kidx=k+nbj;
          memcpy(ekf+kidx*nb+nbi,buf+k*ibi,ibi*sizeof(double));
        }
      }
      if (cint1e_nuc_cart(buf,shls,vb_str->atm,vb_str->natom,vb_str->bas,nsh,vb_str->env)!=0) {
        for (k=0;k<ibj;k++) {
          kidx=k+nbj;
          memcpy(hhf+kidx*nb+nbi,buf+k*ibi,ibi*sizeof(double));
        }
      }
      if (cint1e_r_cart(buf,shls,vb_str->atm,vb_str->natom,vb_str->bas,nsh,vb_str->env)!=0) {
        for (k=0;k<ibj;k++) {
          kidx=k+nbj;
          memcpy(xxf+kidx*nb+nbi,buf+k*ibi,ibi*sizeof(double));
          memcpy(yyf+kidx*nb+nbi,buf+ibi*ibj+k*ibi,ibi*sizeof(double));
          memcpy(zzf+kidx*nb+nbi,buf+2*ibi*ibj+k*ibi,ibi*sizeof(double));
        }
      }
    }
  }

  free(buf);

//  for (i=0;i<nb*nb;i++)
//    *(hhf+i)+=*(ekf+i);
  cblas_daxpy(nb*nb,1e0,ekf,1,hhf,1);

  for (i=0;i<nb;i++) {
    snorm[i]=sqrt(1.0/ssf[i*nb+i]);
    for (j=i+1;j<nb;j++) {
      ssf[j*nb+i]=ssf[i*nb+j];
      hhf[j*nb+i]=hhf[i*nb+j];
      ekf[j*nb+i]=ekf[i*nb+j];
      xxf[j*nb+i]=xxf[i*nb+j];
      yyf[j*nb+i]=yyf[i*nb+j];
      zzf[j*nb+i]=zzf[i*nb+j];
    }
  }

  double sn,sn1;
  for (i=0;i<nb;i++) {
    sn=snorm[i];
    for (j=0;j<nb;j++) {
      sn1=sn*snorm[j];
      ssf[i*nb+j]*=sn1;
      hhf[i*nb+j]*=sn1;
      ekf[i*nb+j]*=sn1;
      xxf[i*nb+j]*=sn1;
      yyf[i*nb+j]*=sn1;
      zzf[i*nb+j]*=sn1;
    }
  }

  return 0;
}

void cal_1eint_(vb_info vb_str) {
  cal_1eint(vb_str);
}

int cal_pcf_int(vb_info vb_str,int print_level) {
  int nb=vb_str->nb;
  vb_str->pcf=(double*)malloc(sizeof(double)*nb*nb);
  int nsh=vb_str->nshell;

  double *points=vb_str->points;
  double *hhf=vb_str->hhf;
  double *snorm=vb_str->snorm;
  double *pcf=vb_str->pcf;

  memset(pcf,0,sizeof(double)*nb*nb);

  int maxl=0;
  for (int i=0;i<nsh;i++)
    maxl=(maxl>vb_str->basidx[2*i+1]?maxl:vb_str->basidx[2*i+1]);

  double *buf=(double*)malloc(maxl*maxl*sizeof(double));
  int nbi,nbj,ibi,ibj,kidx,ptr;
  int shls[2];
  double chg,dx,dy,dz;
  for (int ip=0;ip<vb_str->npoints;ip++) {
    memcpy(vb_str->env+PTR_RINV_ORIG,points+4*ip,3*sizeof(double));
    chg=points[4*ip+3];
    for (int ish=0;ish<nsh;ish++) {
      shls[0]=ish;
      nbi=vb_str->basidx[2*ish];
      ibi=vb_str->basidx[2*ish+1];
      for (int jsh=0;jsh<=ish;jsh++) {
        shls[1]=jsh;
        nbj=vb_str->basidx[2*jsh];
        ibj=vb_str->basidx[2*jsh+1];
        if (cint1e_rinv_cart(buf,shls,vb_str->atm,vb_str->natom,vb_str->bas,nsh,vb_str->env)!=0) {
          for (int k=0;k<ibj;k++) {
            kidx=k+nbj;
            for (int l=0;l<ibi;l++) {
              pcf[kidx*nb+nbi+l]-=chg*buf[k*ibi+l];
            }
//            memcpy(pcf+kidx*nb+nbi,buf+k*ibi,ibi*sizeof(double));
          }
        }
      }
    }

    for (int jp=0;jp<vb_str->natom;jp++) {
      chg=points[4*ip+3]*(double)vb_str->atm[jp*ATM_SLOTS+CHARGE_OF];
      ptr=vb_str->atm[jp*ATM_SLOTS+PTR_COORD];
      dx=points[4*ip+0]-vb_str->env[ptr+0];
      dy=points[4*ip+1]-vb_str->env[ptr+1];
      dz=points[4*ip+2]-vb_str->env[ptr+2];
      vb_str->enuc_pn+=chg/sqrt(dx*dx+dy*dy+dz*dz);
    }

    if (vb_str->npoints>1) {
      for (int jp=ip+1;jp<vb_str->npoints;jp++) {
        chg=points[4*ip+3]*points[4*jp+3];
        dx=points[4*ip+0]-points[4*jp+0];
        dy=points[4*ip+1]-points[4*jp+1];
        dz=points[4*ip+2]-points[4*jp+2];
        vb_str->enuc_pp+=chg/sqrt(dx*dx+dy*dy+dz*dz);
      }
    }
  }

  free(buf);

  double si;
  for (int i=0;i<nb;i++) {
    si=snorm[i];
    pcf[i*nb+i]*=si*si;
    for (int j=i+1;j<nb;j++) {
      pcf[i*nb+j]*=si*snorm[j];
      pcf[j*nb+i]=pcf[i*nb+j];
    }
  }

  cblas_daxpy(nb*nb,1e0,pcf,1,hhf,1);

  if (print_level>0) {
    printf("\n");
    printf("Bare Nuclear Repulsion Energy : %12.6f a.u.\n",vb_str->enuc);
    printf("Charge-Nuclear Interaction    : %12.6f a.u.\n",vb_str->enuc_pn);
    printf("Charge-Charge  Interaction    : %12.6f a.u.\n",vb_str->enuc_pp);
  }

  vb_str->enuc+=vb_str->enuc_pn+vb_str->enuc_pp;
  if (print_level>0)
    printf("Total Nuclear Repulsion Energy: %12.6f a.u.\n",vb_str->enuc);

  return 0;

}



// void cal_1egrad_cint(Tensor3D ssfg, Tensor3D hhfg, Tensor3D ekfg, Tensor3D rinvg, vb_info vb_str) {
void cal_1egrad_cint(Tensor3D ssfg, Tensor3D hhfg, vb_info vb_str,mol_info mol) {

  int nb=mol->bas->msize;
  int natom=mol->atm->natm;
  int nsh=vb_str->nshell;

  int shls[2],nbi,ibi,nbj,ibj,iatm,kcrd,kdx;

  double *buf=(double*)malloc(sizeof(double)*200000);
  double atmcharg;

  memset(buf,0,sizeof(double)*200000);

  for (int i=0;i<nsh;i++) {
    shls[0]=i;
    nbi=vb_str->basidx[2*i];
    ibi=vb_str->basidx[2*i+1];
    iatm=vb_str->bas[BAS_SLOTS*i+ATOM_OF];
    for (int j=0;j<nsh;j++) {
      shls[1]=j;
      nbi=vb_str->basidx[2*j];
      ibi=vb_str->basidx[2*j+1];
      if (cint1e_ipovlp_cart(buf,shls,vb_str->atm,vb_str->natom,vb_str->bas,nsh,vb_str->env)!=0) {
        for (int k=0;k<ibi;k++) {
          for (int l=0;l<ibj;l++) {
            ssfg[3*iatm][nbj+l][nbi+k]-=buf[l*ibi+k];
            ssfg[3*iatm+1][nbj+l][nbi+k]-=buf[ibi*ibj+l*ibi+k];
            ssfg[3*iatm+2][nbj+l][nbi+k]-=buf[2*ibi*ibj+l*ibi+k];
          }
        }
      }
      if (cint1e_ipkin_cart(buf,shls,vb_str->atm,vb_str->natom,vb_str->bas,nsh,vb_str->env)!=0) {
        for (int k=0;k<ibi;k++) {
          for (int l=0;l<ibj;l++) {
//            ekfg[iatm][nbj+l][nbi+k]=-buf[l*ibi+k];
//            ekfg[iatm+1][nbj+l][nbi+k]=-buf[ibi*ibj+l*ibi+k];
//            ekfg[iatm+2][nbj+l][nbi+k]=-buf[2*ibi*ibj+l*ibi+k];
            hhfg[3*iatm][nbj+l][nbi+k]-=buf[l*ibi+k];
            hhfg[3*iatm+1][nbj+l][nbi+k]-=buf[ibi*ibj+l*ibi+k];
            hhfg[3*iatm+2][nbj+l][nbi+k]-=buf[2*ibi*ibj+l*ibi+k];
          }
        }
      }
      if (cint1e_ipnuc_cart(buf,shls,vb_str->atm,vb_str->natom,vb_str->bas,nsh,vb_str->env)!=0) {
        for (int k=0;k<ibi;k++) {
          for (int l=0;l<ibj;l++) {
            hhfg[3*iatm][nbj+l][nbi+k]-=buf[l*ibi+k];
            hhfg[3*iatm+1][nbj+l][nbi+k]-=buf[ibi*ibj+l*ibi+k];
            hhfg[3*iatm+2][nbj+l][nbi+k]-=buf[2*ibi*ibj+l*ibi+k];
          }
        }
      }
      for (int k1=0;k1<natom;k1++) {
        kcrd=vb_str->atm[k1*ATM_SLOTS+PTR_COORD];
        kdx=k1*3;
        memcpy(vb_str->env+PTR_RINV_ORIG,vb_str->env+kcrd,3*sizeof(double));
        atmcharg=vb_str->atm[k1*ATM_SLOTS+CHARGE_OF];
        if (cint1e_ipnuc_cart(buf,shls,vb_str->atm,vb_str->natom,vb_str->bas,nsh,vb_str->env)!=0) {
          for (int k=0;k<ibi;k++) {
            for (int l=0;l<ibj;l++) {
//              rinvg[iatm][nbj+l][nbi+k]-=atmcharg*buf[l*ibi+k];
//              rinvg[iatm+1][nbj+l][nbi+k]-=atmcharg*buf[ibi*ibj+l*ibi+k];
//              rinvg[iatm+2][nbj+l][nbi+k]-=atmcharg*buf[2*ibi*ibj+l*ibi+k];
              hhfg[kdx][nbj+l][nbi+k]-=atmcharg*buf[l*ibi+k];
              hhfg[kdx+1][nbj+l][nbi+k]-=atmcharg*buf[ibi*ibj+l*ibi+k];
              hhfg[kdx+2][nbj+l][nbi+k]-=atmcharg*buf[2*ibi*ibj+l*ibi+k];
            }
          }
        }
      }
    }
  }

  double si,sj;
  for (int i=0;i<nb;i++) {
    si=vb_str->snorm[i];
    for (int j=0;j<=i;j++) {
      sj=vb_str->snorm[i];
      for (int k=0;k<natom*3;k++) {
        ssfg[k][i][j]=(ssfg[k][i][j]+ssfg[k][j][i])*si*sj;
        ssfg[k][j][i]=ssfg[k][i][j];
        hhfg[k][i][j]=(hhfg[k][i][j]+hhfg[k][j][i])*si*sj;
        hhfg[k][j][i]=hhfg[k][i][j];
      }
    }
  }


  free(buf);

  return;
}

void cal_nuc_grad(double *grad, mol_info mol) {
  double vdist[3];

  int natom=mol->atm->natm;

  int q1,q2;
  double c1,c2,dist;

  for (int i=0;i<natom;i++) {
    q1=mol->atm->atm[CENTER_IND+2*i];
    c1=(double)mol->atm->atm[2*i+ELEMENT_VAL];
    for (int j=0;j<natom;j++) {
      if (i==j)
        continue;
      q2=mol->atm->atm[CENTER_IND+2*j];
      c2=(double)mol->atm->atm[2*j+ELEMENT_VAL];
      vdist[0]=mol->atm->value[q1]-mol->atm->value[q2];
      vdist[1]=mol->atm->value[q1+1]-mol->atm->value[q2+1];
      vdist[2]=mol->atm->value[q1+2]-mol->atm->value[q2+2];
      dist=sqrt(cblas_ddot(3,vdist,1,vdist,1));
      grad[3*i]-=vdist[0]*c1*c2/dist/dist/dist;
    }
  }

  return;
}

int cal_dipole_int(double *dip_nuc, double *xxf, double *yyf, double *zzf, int *atm, int *bas, int *basidx, double *env, int ngto, mol_info mol) {

  int nb=mol->bas->msize;
  int natom=mol->atm->natm;
  double *crd=(double*)malloc(sizeof(double)*natom*3);
  double *zan=(double*)malloc(sizeof(double)*natom);
  double *snorm=(double*)malloc(sizeof(double)*nb);
  double *buff=(double*)malloc(sizeof(double)*1000);
  int qoff;

  for (int i=0;i<natom;i++) {
    qoff=mol->atm->atm[i*2+CENTER_IND];
    crd[3*i+0]=mol->atm->value[qoff+0];
    crd[3*i+1]=mol->atm->value[qoff+1];
    crd[3*i+2]=mol->atm->value[qoff+2];
    zan[i]=(double)mol->atm->atm[i*2+ELEMENT_VAL];
  }

  double xnuc=0e0, ynuc=0e0, znuc=0e0;
  double xx,yy,zz;

  for (int i=0;i<natom-1;i++)
    for (int j=i+1;j<natom;j++) {
      xx=fabs(crd[3*i+0]-crd[3*j+0]);
      yy=fabs(crd[3*i+1]-crd[3*j+1]);
      zz=fabs(crd[3*i+2]-crd[3*j+2]);
      xnuc+=zan[i]*zan[j]/xx;
      ynuc+=zan[i]*zan[j]/yy;
      znuc+=zan[i]*zan[j]/zz;
    }

  int nbi,ibi,nbj,ibj,kidx,shls[2];
  int nsh=mol->bas->nbas;
  for (int i=0;i<nsh;i++) {
    shls[0]=i;
    nbi=basidx[2*i];
    ibi=basidx[2*i+1];
    for (int j=0;j<=i;j++) {
      shls[1]=j;
      nbj=basidx[2*j];
      ibj=basidx[2*j+1];
      if (j==i) {
        if (cint1e_ovlp_cart(buff,shls,atm,natom,bas,nsh,env)!=0) {
          for (int k=0;k<ibj;k++) {
            snorm[nbj+k]=1e0/sqrt(buff[k*ibj+k]);
          }
        }
      }
      if (cint1e_r_cart(buff,shls,atm,natom,bas,nsh,env)!=0) {
        for (int k=0;k<ibj;k++) {
          kidx=k+nbj;
          memcpy(xxf+kidx*nb+nbi,buff+k*ibi,ibi*sizeof(double));
          memcpy(yyf+kidx*nb+nbi,buff+ibi*ibj+k*ibi,ibi*sizeof(double));
          memcpy(zzf+kidx*nb+nbi,buff+2*ibi*ibj+k*ibi,ibi*sizeof(double));
        }
      }
    }
  }

  for (int i=0;i<nb;i++) {
    for (int j=i;j<nb;j++) {
      xxf[i*nb+j]*=snorm[i]*snorm[j];
      xxf[j*nb+i]=xxf[i*nb+j];
      yyf[i*nb+j]*=snorm[i]*snorm[j];
      yyf[j*nb+i]=yyf[i*nb+j];
      zzf[i*nb+j]*=snorm[i]*snorm[j];
      zzf[j*nb+i]=zzf[i*nb+j];
    }
  }

  dip_nuc[0]=xnuc;
  dip_nuc[1]=ynuc;
  dip_nuc[2]=znuc;
  dip_nuc[3]=0e0;

  free(crd);
  free(zan);
  free(snorm);
  free(buff);
}

