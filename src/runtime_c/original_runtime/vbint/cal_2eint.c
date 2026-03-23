#include <stdio.h>
// #include <stdint.h>
#include <math.h>
#include "mol/mol.h"
#include "vb/vb.h"
#include "cint_funcs.h"
#include "cblas.h"

//void build_shidx(vb_info vb_str,int shidx[][4], int *nshidx,int *n2e,int maxl,CINTOpt *opt) {
//void build_shidx(vb_info vb_str,double *buf,int *shidx, int *nshidx) {
void build_shidx(vb_info vb_str,int *shidx, int *nshidx) {

  int nsh=vb_str->nshell;
//  double gmax[nsh*(nsh+1)/2];
  double *gmax=(double*)malloc(sizeof(double)*(nsh*nsh+nsh)/2);
//  int shls[4];
//  int i,j,gidx,ibi,ibj;
  memset(gmax,0,sizeof(double)*(nsh*nsh+nsh)/2);
//  double *buf=(double*)malloc(sizeof(double)*160000);

  CINTOpt *opt;
  cint2e_cart_optimizer(&opt, vb_str->atm, vb_str->natom, vb_str->bas,nsh,vb_str->env);

  #pragma omp parallel shared(nsh,vb_str,gmax,opt)
  {
    #pragma omp for nowait
    for (int i=0;i<nsh;i++) {
      double *buf=(double*)malloc(sizeof(double)*160000);
      int shls[4];
      shls[0]=i;
      shls[2]=i;
      int ibi=vb_str->basidx[2*i+1];
      for (int j=0;j<=i;j++) {
        shls[1]=j;
        shls[3]=j;
        int ibj=vb_str->basidx[2*j+1];
        int gidx=lab(i,j);
        if (cint2e_cart(buf,shls,vb_str->atm,vb_str->natom,vb_str->bas,nsh,vb_str->env,opt)!=0) {
          for (int idx=0;idx<ibi*ibi*ibj*ibj;idx++) {
            if (fabs(buf[idx])>gmax[gidx])
              gmax[gidx]=fabs(buf[idx]);
          }
        }
      }
      free(buf);
    }
  }


  CINTdel_optimizer(&opt);

  *nshidx=0;
  int k,l,ibk,ibl,lend;
  int ibi,ibj;
  double val;
  for (int i=0;i<nsh;i++) {
//    shls[0]=i;
    ibi=*(vb_str->basidx+2*i+1);
    for (int j=0;j<=i;j++) {
//      shls[1]=j;
      ibj=*(vb_str->basidx+2*j+1);
      for (int k=0;k<=i;k++) {
//        shls[2]=k;
        ibk=*(vb_str->basidx+2*k+1);
//        if (i==k)
//          lend=j;
//        else
//          lend=k;
//        for (l=0;l<=lend;l++) {
        for (int l=0;l<=k;l++) {
//          shls[3]=l;
          ibl=*(vb_str->basidx+2*l+1);
          val=gmax[lab(i,j)]*gmax[lab(k,l)];
//          if (i==k)
//            val=gmax[lab(i,k)]*gmax[lab(j,l)];
//          else
//            val=gmax[lab(i,j)]*gmax[lab(k,l)];
          if (sqrt(val)>=INT_TOL) {
            *(shidx+(*nshidx)*4  )=i;
            *(shidx+(*nshidx)*4+1)=j;
            *(shidx+(*nshidx)*4+2)=k;
            *(shidx+(*nshidx)*4+3)=l;
//            shidx[*nshidx][0]=i;
//            shidx[*nshidx][1]=j;
//            shidx[*nshidx][2]=k;
//            shidx[*nshidx][3]=l;
            (*nshidx)++;
//            (*n2e)+=ibi*ibj*ibk*ibl;
          }
        }
      }
    }
  }

  free(gmax);
  return;
}

int cal_2eint(vb_info vb_str, int print_level) {

  if (vb_str->ggf!=NULL)
    free(vb_str->ggf);
  if (vb_str->g2eidx!=NULL)
    free(vb_str->g2eidx);

  int nsh=vb_str->nshell;
  int maxl=0;

  for (int i=0;i<nsh;i++)
    maxl=my_max(maxl,*(vb_str->basidx+2*i+1));

//  double buf[maxl*maxl*maxl*maxl];

  CINTOpt *opt;
  cint2e_cart_optimizer(&opt, vb_str->atm, vb_str->natom, vb_str->bas,nsh,vb_str->env);
//  int shidx[nsh*nsh*nsh*nsh][4],nshidx;
  int nshidx;
  int *shidx=(int*)malloc(sizeof(int)*nsh*nsh*nsh*nsh*4);

  nshidx=0;
  build_shidx(vb_str,shidx,&nshidx);
//   printf("%d\n",nshidx);
//   fflush(stdout);
  // exit(0);
  // int nbi,ibi,nbj,ibj,nbk,ibk,nbl,ibl;
  //  int p,q,r,s,idx0,gidx;
  //  double sni,snj,snk,val;
  int nb=vb_str->nb;
  long nb2=(nb*nb+nb)/2;
  long nb4=(nb2*nb2+nb2)/2;
  // size_t nn2=(nb2*nb2+nb2)/2;
//  double gg0[nb4];
//  for (int i=0;i<nb4;i++)
//    gg0[i]=0.0;
  // printf("nb4 is %lld, %d\n",nb4,nb2);
  // int64_t big_integer = 9223372036854775807; // 最大的int64_t值
  // printf("Big integer: %lld\n", big_integer);
  // exit(0);
  double *gg0=(double*)malloc(nb4*sizeof(double));
  memset(gg0,0,nb4*sizeof(double));
  #pragma omp parallel shared(gg0,shidx,nshidx,vb_str,opt,maxl)
  {
    #pragma omp for nowait
    for (int gidx=0;gidx<nshidx;gidx++) {
      int shls[4];
      shls[0]=shidx[4*gidx  ];
      shls[1]=shidx[4*gidx+1];
      shls[2]=shidx[4*gidx+2];
      shls[3]=shidx[4*gidx+3];
      int nbi=vb_str->basidx[2*shls[0]];
      int ibi=vb_str->basidx[2*shls[0]+1];
      int nbj=vb_str->basidx[2*shls[1]];
      int ibj=vb_str->basidx[2*shls[1]+1];
      int nbk=vb_str->basidx[2*shls[2]];
      int ibk=vb_str->basidx[2*shls[2]+1];
      int nbl=vb_str->basidx[2*shls[3]];
      int ibl=vb_str->basidx[2*shls[3]+1];
      double *buf=(double*)malloc(sizeof(double)*maxl*maxl*maxl*maxl);
      if (cint2e_cart(buf,shls,vb_str->atm,vb_str->natom,vb_str->bas,nsh,vb_str->env,opt)!=0) {
        for (int p=0;p<ibi;p++) {
          double sni=vb_str->snorm[p+nbi];
          for (int q=0;q<ibj;q++) {
//            if ((q+nbj)>(p+nbi))
//              break;
            double snj=vb_str->snorm[q+nbj];
            for (int r=0;r<ibk;r++) {
//              if ((r+nbk)>(p+nbi))
//                break;
              double snk=vb_str->snorm[r+nbk];
              for (int s=0;s<ibl;s++) {
//                if ((r+nbk)==(p+nbi) && (s+nbl)>(q+nbj))
//                  break;
//                else if ((s+nbl)>(r+nbk))
//                  break;
                int idx0=s*ibk*ibj*ibi+r*ibj*ibi+q*ibi+p;
                double val=buf[idx0]*sni*snj*snk*vb_str->snorm[s+nbl];
                gg0[lab_long(lab_long((long)(p+nbi),(long)(q+nbj)),(lab_long((long)( r+nbk),(long)(s+nbl))))]=val;
              }
            }
          }
        }
      }
      free(buf);
    }
  }

  free(shidx);
  int i,j,k,l,lend;
  // exit(0);
  long n2e0=0;
  for (long i=0;i<nb4;i++)
//    if(__glibc_unlikely(vb_str->iscf==HES_RDM_SCF)){
    if (vb_str->iscf==HES_RDM_SCF) {
      n2e0++;
    }
    else{
      if (fabs(*(gg0+i))>=INT_TOL)
        n2e0++;
    }
  // 
  vb_str->ggf=(double*)malloc(sizeof(double)*n2e0);
  vb_str->g2eidx=(int*)malloc(sizeof(int)*n2e0*4);
  
  long idx=0;
  vb_str->n2e=0;
  for (i=0;i<nb;i++)
    for (j=0;j<=i;j++)
      for (k=0;k<=i;k++) {
        if (k==i)
          lend=j;
        else
          lend=k;
        for (l=0;l<=lend;l++) {
//          if(__glibc_unlikely(vb_str->iscf==HES_RDM_SCF)){
          if (vb_str->iscf==HES_RDM_SCF) {
            *(vb_str->ggf+vb_str->n2e)=gg0[idx];
            *(vb_str->g2eidx+vb_str->n2e*4  )=i;
            *(vb_str->g2eidx+vb_str->n2e*4+1)=j;
            *(vb_str->g2eidx+vb_str->n2e*4+2)=k;
            *(vb_str->g2eidx+vb_str->n2e*4+3)=l;
            vb_str->n2e++;
          }
          else{
            if (fabs(*(gg0+idx))>=INT_TOL) {
              *(vb_str->ggf+vb_str->n2e)=gg0[idx];
              *(vb_str->g2eidx+vb_str->n2e*4  )=i;
              *(vb_str->g2eidx+vb_str->n2e*4+1)=j;
              *(vb_str->g2eidx+vb_str->n2e*4+2)=k;
              *(vb_str->g2eidx+vb_str->n2e*4+3)=l;
              vb_str->n2e++;
            }
          }
          idx++;
        }
      }

  if (print_level>0)
    printf(" Non-zero 2-e integrals: %ld\n",vb_str->n2e);
  CINTdel_optimizer(&opt);
  // exit(0);
  free(gg0);

  return 0;
}

void cal_2eint_(vb_info vb_str, int print_level) {
  cal_2eint(vb_str,print_level);
}


void int_gg3_shell_direct(double *gg3, int ish, vb_info vb_str) {

  int nshell=vb_str->nshell;
  int shls[4];
  double *buf=(double*)malloc(sizeof(double)*160000);

  shls[0]=ish;

  CINTOpt *opt;
  cint2e_cart_optimizer(&opt, vb_str->atm, vb_str->natom, vb_str->bas,nshell,vb_str->env);

  int nbi=*(vb_str->basidx+2*ish);
  int ibi=*(vb_str->basidx+2*ish+1);
  int nbj,ibj,nbk,ibk,nbl,ibl;
  int nb,nb2,nb3,idx1,idx2,offset;
  double *snorm=vb_str->snorm;
  nb=vb_str->nb;
  nb2=(nb+1)*nb/2;
  nb3=nb2*nb;

  for (int j=0;j<nshell;j++) {
    shls[1]=j;
    nbj=*(vb_str->basidx+2*j);
    ibj=*(vb_str->basidx+2*j+1);
    for (int k=0;k<nshell;k++) {
      shls[2]=k;
      nbk=*(vb_str->basidx+2*k);
      ibk=*(vb_str->basidx+2*k+1);
      for (int l=0;l<nshell;l++) {
        shls[3]=l;
        nbl=*(vb_str->basidx+2*l);
        ibl=*(vb_str->basidx+2*l+1);
        cint2e_cart(buf,shls,vb_str->atm,vb_str->natom,vb_str->bas,nshell,vb_str->env,opt);
        for (int i1=0;i1<ibi;i1++) {
          offset=i1*nb3;
          for (int j1=0;j1<ibj;j1++)
            for (int k1=0;k1<ibk;k1++)
              for (int l1=0;l1<ibl;l1++) {
                idx1=lab(k1+nbk,l1+nbl)*nb+j1+nbj;
                idx2=l1*ibi*ibj*ibk+k1*ibj*ibi+j1*ibi+i1;
                gg3[idx1+offset]=buf[idx2]*snorm[i1+nbi]*snorm[j1+nbj]*snorm[k1+nbk]*snorm[l1+nbl];
              }
        }
      }
    }
  }

  free(buf);


  CINTdel_optimizer(&opt);
  return;
}

void int_gg3_shell_direct_(double *gg3, int *ish, vb_info vb_str) {
  int_gg3_shell_direct(gg3,*ish-1,vb_str);
  return;
}

void cal_fock_vb_direct(double *fa, double *fb, double *pa, double *pb, vb_info vb_str) {

  int rfock=0;
  if (fb==NULL || pb == NULL)
    rfock=1;

  int nsh=vb_str->nshell;
  int nb=vb_str->nb;
//  int shls[4];
//  int nbi,ibi,nbj,ibj,nbk,ibk,nbl,ibl;
//  int idx,i,j,k,l;
//  double sni,snj,snk,snl,e2e,a0,a1;
//  double *buf=(double*)malloc(sizeof(double)*160000);
//  double *fa_t=(double*)malloc(sizeof(double)*nb*nb);
//  double *fb_t=(double*)malloc(sizeof(double)*nb*nb);

//  memset(fa_t,0,sizeof(double)*nb*nb);
//  memset(fb_t,0,sizeof(double)*nb*nb);

  CINTOpt *opt;
  cint2e_cart_optimizer(&opt, vb_str->atm, vb_str->natom, vb_str->bas,nsh,vb_str->env);

//  #pragma omp parallel shared(nsh,vb_str,rfock,pa,pb,nb) reduction(+:fa_t[:nb*nb],fb_t[:nb*nb])
  #pragma omp parallel shared(nsh,vb_str,rfock,pa,pb,nb,fa,fb) //  reduction(+:fa_t[:nb*nb],fb_t[:nb*nb])
   {
     double *buf=(double*)malloc(sizeof(double)*160000);
     double *fa_t=(double*)malloc(sizeof(double)*nb*nb);
     double *fb_t=(double*)malloc(sizeof(double)*nb*nb);
     memset(fa_t,0,sizeof(double)*nb*nb);
     memset(fb_t,0,sizeof(double)*nb*nb);
     #pragma omp for nowait
    for (int ish=0;ish<nsh;ish++) {
      int shls[4];
      shls[0]=ish;
      int nbi=vb_str->basidx[2*ish];
      int ibi=vb_str->basidx[2*ish+1];
      for (int jsh=0;jsh<=ish;jsh++) {
        shls[1]=jsh;
        int nbj=vb_str->basidx[2*jsh];
        int ibj=vb_str->basidx[2*jsh+1];
        for (int ksh=0;ksh<=ish;ksh++) {
          shls[2]=ksh;
          int nbk=vb_str->basidx[2*ksh];
          int ibk=vb_str->basidx[2*ksh+1];
          for (int lsh=0;lsh<=ish;lsh++) {
            shls[3]=lsh;
            int nbl=vb_str->basidx[2*lsh];
            int ibl=vb_str->basidx[2*lsh+1];
            if (cint2e_cart(buf,shls,vb_str->atm,vb_str->natom,vb_str->bas,nsh,vb_str->env,opt)!=0) {
              for (int p=0;p<ibi;p++) {
                int i=p+nbi;
                double sni=vb_str->snorm[i];
                for (int q=0;q<ibj;q++) {
                  int j=q+nbj;
                  if (j>i)
                    continue;
                  double snj=vb_str->snorm[j];
                  for (int r=0;r<ibk;r++) {
                    int k=r+nbk;
                    if (k>i)
                      continue;
                    double snk=vb_str->snorm[k];
                    for (int s=0;s<ibl;s++) {
                      int l=s+nbl;
                      if ((i==k && l>j) || (i!=k && l>k))
                        continue;
                      double snl=vb_str->snorm[l];
                      int idx=p+q*ibi+r*ibj*ibi+s*ibk*ibj*ibi;
                      double e2e=buf[idx]*sni*snj*snk*snl;
                      if (i==j) e2e*=0.5e0;
                      if (k==l) e2e*=0.5e0;
                      if (i==k && j==l) e2e*=0.5e0;
                      if (rfock==1) {
                        double a0=pa[j*nb+i]*e2e*4.0e0;
                        double a1=pa[l*nb+k]*e2e*4.0e0;
                        fa_t[j*nb+i]+=a1;
                        fa_t[l*nb+k]+=a0;
                        fa_t[k*nb+i]-=pa[j*nb+l]*e2e;
                        fa_t[j*nb+l]-=pa[i*nb+k]*e2e;
                        fa_t[l*nb+i]-=pa[j*nb+k]*e2e;
                        fa_t[k*nb+j]-=pa[i*nb+l]*e2e;
                      }
                      else {
                        double a0=(pa[j*nb+i]+pa[i*nb+j]+pb[j*nb+i]+pb[i*nb+j])*e2e;
                        double a1=(pa[l*nb+k]+pa[k*nb+l]+pb[l*nb+k]+pb[k*nb+l])*e2e;
                        fa_t[j*nb+i]+=a1;
                        fa_t[i*nb+j]+=a1;
                        fa_t[l*nb+k]+=a0;
                        fa_t[k*nb+l]+=a0;
                        fb_t[j*nb+i]+=a1;
                        fb_t[i*nb+j]+=a1;
                        fb_t[l*nb+k]+=a0;
                        fb_t[k*nb+l]+=a0;
                        fa_t[k*nb+i]-=pa[j*nb+l]*e2e;
                        fa_t[l*nb+j]-=pa[i*nb+k]*e2e;
                        fa_t[l*nb+i]-=pa[j*nb+k]*e2e;
                        fa_t[k*nb+j]-=pa[i*nb+l]*e2e;
                        fa_t[i*nb+k]-=pa[l*nb+j]*e2e;
                        fa_t[j*nb+l]-=pa[k*nb+i]*e2e;
                        fa_t[i*nb+l]-=pa[k*nb+j]*e2e;
                        fa_t[j*nb+k]-=pa[l*nb+i]*e2e;
                        fb_t[k*nb+i]-=pb[j*nb+l]*e2e;
                        fb_t[l*nb+j]-=pb[i*nb+k]*e2e;
                        fb_t[l*nb+i]-=pb[j*nb+k]*e2e;
                        fb_t[k*nb+j]-=pb[i*nb+l]*e2e;
                        fb_t[i*nb+k]-=pb[l*nb+j]*e2e;
                        fb_t[j*nb+l]-=pb[k*nb+i]*e2e;
                        fb_t[i*nb+l]-=pb[k*nb+j]*e2e;
                        fb_t[j*nb+k]-=pb[l*nb+i]*e2e;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    #pragma omp critical
    {
      cblas_daxpy(nb*nb,1e0,fa_t,1,fa,1);
      if (rfock==0)
        cblas_daxpy(nb*nb,1e0,fb_t,1,fb,1);
    }
    free(buf);
    free(fa_t);
    free(fb_t);
  }

  if (rfock==1) {
    #pragma omp parallel shared(nb)
    {
      #pragma omp for nowait
      for (int i=0;i<nb;i++) {
        for (int j=0;j<=i;j++) {
          fa[j*nb+i]+=fa[i*nb+j];
          fa[i*nb+j]=fa[j*nb+i];
        }
      }
    }
  }

//  for (int i=0;i<nb*nb;i++)
//    fa[i]+=fa_t[i];
//  if (rfock==0)
//    for (int i=0;i<nb*nb;i++)
//      fb[i]+=fb_t[i];

//  cblas_daxpy(nb*nb,1e0,fa_t,1,fa,1);
//  if (rfock==0)
//    cblas_daxpy(nb*nb,1e0,fb_t,1,fb,1);

//  free(fa_t);
//  free(fb_t);

//  free(buf);
  CINTdel_optimizer(&opt);
  return;
}

void cal_rfock_vb_direct_(double *fa, double *pa, vb_info vb_str) {
  cal_fock_vb_direct(fa,NULL,pa,NULL,vb_str);
  return;
}

void cal_ufock_vb_direct_(double *fa, double *fb, double *pa, double *pb, vb_info vb_str) {
  cal_fock_vb_direct(fa,fb,pa,pb,vb_str);
  return;
}
