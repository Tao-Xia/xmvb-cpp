#include "solver/opt_solver.h"
struct DiisInfo
{
    int DIIS_size, DIIS_p, msize;
    double **f_matrix_store;
    double **d_matrix_store;
    double **CDIIS_erro;
    double *inter_coe, *extra_coe;
    double *inter_coe_b;
    double *CDIIS_matrix;
    double *EDIIS_matrix;
    double tr1, tr2;
};
/**
 * @brief doing matrix inner product
 * @param [in] a 
 * @param [in] b 
 * @param [in] msize size of matrix 
 * @return double 
 * (a|b)=tr(ab)=sum_{ij}{a_ij*b_ij}
 * @details 
 */
static double minner(double* a, double* b, int msize)
{
    int i;
    double result=0.0;
    double len=msize*msize;
    for (i=0;i<len;i++)
    {
        result+=a[i]*b[i];
    }
    return result;
}
static double optim(double* c, int n, int size, double* matrix, double* vector)
{
    int i,j;
    double result=0.;
    for (i=0;i<n;i++)
    {
        result+=c[i]*vector[i];
        for (j=0;j<n;j++) result+=c[i]*c[j]*matrix[size*i+j];
    }
    return result;
}
diis_info init_diis(int size, int msize){
    diis_info diis=(diis_info)malloc(sizeof(struct DiisInfo));
    diis->DIIS_size=size;
    diis->DIIS_p=1;
    diis->tr1=0.1;
    diis->tr2=1E-4;
    diis->msize=msize;
    diis->f_matrix_store=(double**)malloc(sizeof(double)*size);
    diis->d_matrix_store=(double**)malloc(sizeof(double)*size);
    diis->CDIIS_erro=(double**)malloc(sizeof(double)*size);
    diis->inter_coe=(double*)malloc(sizeof(double)*size);
    diis->extra_coe=(double*)malloc(sizeof(double)*size);
    for (int i=0;i<size;i++)
    {
        diis->f_matrix_store[i]=(double*)malloc(sizeof(double)*msize*msize);
        diis->d_matrix_store[i]=(double*)malloc(sizeof(double)*msize*msize);
        diis->CDIIS_erro[i]=(double*)malloc(sizeof(double)*msize*msize);
        memset(diis->f_matrix_store[i],0,sizeof(double)*msize*msize);
        memset(diis->d_matrix_store[i],0,sizeof(double)*msize*msize);
        memset(diis->CDIIS_erro[i],0,sizeof(double)*msize*msize);
    }
    diis->CDIIS_matrix=(double*)malloc(sizeof(double)*(size)*(size));
    diis->EDIIS_matrix=(double*)malloc(sizeof(double)*(size)*(size));
    memset(diis->CDIIS_matrix,0,sizeof(double)*(size)*(size));
    memset(diis->EDIIS_matrix,0,sizeof(double)*(size)*(size));
    for (int i=0;i<size;i++)
    {
        diis->CDIIS_matrix[i*(size)+i]=1.;
        diis->EDIIS_matrix[i*(size)+i]=1.;
    }
    return diis;
}
int del_diis(diis_info diis)
{
    for (int i=0; i<diis->DIIS_size; i++)
    {
        free(diis->f_matrix_store[i]);
        free(diis->d_matrix_store[i]);
        free(diis->CDIIS_erro[i]);
    }
    free(diis->f_matrix_store);
    free(diis->d_matrix_store);
    free(diis->CDIIS_erro);
    free(diis->inter_coe);
    free(diis->extra_coe);
    free(diis->CDIIS_matrix);
    free(diis->EDIIS_matrix);
    free(diis);
    return 0;
}
double ediis_err(double* f_matrix_1, double* f_matrix_2, double* d_matrix_1, double* d_matrix_2, int msize)
{
    double err=0.0;
    int i;
    for (i=0;i<msize*msize;i++)
    {
        err-=(f_matrix_1[i]-f_matrix_2[i])*(d_matrix_1[i]-d_matrix_2[i]);
    }
    return err;
}
double ffmin(double a,double b,double c)
{
    double result=a;
    if (result>b) result=b;
    if (result>c) result=c;
    return result;
}
double do_diis(Matrix s_matrix, Matrix f_matrix_in, Matrix d_matrix_in, diis_info diis)
{
    double* s_matrix_v=s_matrix[0];
    double* f_matrix=f_matrix_in[0];
    double* d_matrix=d_matrix_in[0];
    int i,j;
    int size=diis->DIIS_size, msize=diis->msize;
    double **f_matrix_store=diis->f_matrix_store;
    double **d_matrix_store=diis->d_matrix_store;
    double *fd_matrix=NULL;
    double *fds_matrix=NULL;
    double *err_vector=NULL;
    double judge=0.0;
    double* DIIS_vector=(double*)malloc(sizeof(double)*size);
    double* DIIS_vector_b=(double*)malloc(sizeof(double)*size);
    fd_matrix=(double*)malloc(sizeof(double)*msize*msize);
    fds_matrix=(double*)malloc(sizeof(double)*msize*msize);
    err_vector=(double*)malloc(sizeof(double)*msize*msize);
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,msize,msize,msize,1.0,f_matrix,msize,d_matrix,msize,0.0,fd_matrix,msize);
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,msize,msize,msize,1.0,fd_matrix,msize,s_matrix_v,msize,0.0,fds_matrix,msize);
    // make a shift of density matrixes and Fock matrixes calculated before.  
    for (i=size-1;i>0;i--)
    {
        memcpy(d_matrix_store[i],d_matrix_store[i-1],sizeof(double)*msize*msize);
        memcpy(f_matrix_store[i],f_matrix_store[i-1],sizeof(double)*msize*msize);
        memcpy(diis->CDIIS_erro[i],diis->CDIIS_erro[i-1],sizeof(double)*msize*msize);
    }
    for (i=size-1;i>0;i--)
    {
        for (j=size-1;j>0;j--)
        {
            diis->CDIIS_matrix[i*size+j]=diis->CDIIS_matrix[(i-1)*size+j-1];
        }
    }
    // store density matrix and Fock matrix in this cycle. 
    memcpy(f_matrix_store[0],f_matrix,sizeof(double)*msize*msize);
    memcpy(d_matrix_store[0],d_matrix,sizeof(double)*msize*msize);
    for (i=0;i<msize;i++)
    {
        for (j=0;j<msize;j++) err_vector[i*msize+j]=fds_matrix[i*msize+j]-fds_matrix[j*msize+i];
    }
    for (i=0;i<msize*msize;i++)
    {
        if (fabs(err_vector[i])>=judge) judge=fabs(err_vector[i]);
    }
    memcpy(diis->CDIIS_erro[0],err_vector,sizeof(double)*msize*msize);
    for (i=0;i<diis->DIIS_p;i++)
    {
        diis->CDIIS_matrix[i*size]=minner(err_vector,diis->CDIIS_erro[i],msize);
        diis->CDIIS_matrix[i*size+diis->DIIS_p]=-1.;
        diis->CDIIS_matrix[i]=diis->CDIIS_matrix[i*size];
        diis->CDIIS_matrix[diis->DIIS_p*size+i]=-1.;
        diis->CDIIS_matrix[i*size+i]*=1+1E-6;
    }
    for (i=0;i<diis->DIIS_p;i++)
    {
        DIIS_vector[i]=0.0;
        for (j=0;j<msize*msize;j++)
        {
            DIIS_vector[i]+=2*(d_matrix_store[i][j]-d_matrix[j])*f_matrix[j];
        }
        for (j=0;j<diis->DIIS_p;j++)
        {
            diis->EDIIS_matrix[i*size+j]=-ediis_err(f_matrix_store[i],f_matrix,d_matrix_store[j],d_matrix,msize);
        }
    }
    if (diis->DIIS_p==1)
    {
        diis->DIIS_p++;
        free(DIIS_vector);
        free(DIIS_vector_b);
        free(fd_matrix);
        free(fds_matrix);
        free(err_vector);
        return -log10(diis->CDIIS_matrix[0]);;
    }
    memset(f_matrix,0,sizeof(double)*msize*msize);
    /* Do CDIIS process */
    int *ipiv=(int*)malloc(sizeof(int)*(size));
    double* cdiis_rev=(double*)malloc(sizeof(double)*(size)*(size));
    double* work=(double*)malloc(sizeof(double)*(size)*(size)*4);
    double* egn_value=(double*)malloc(sizeof(double)*size);
    int nhrs=1;
    int erro=0,lwork=sizeof(double)*(size)*(size)*4;
    memcpy(cdiis_rev,diis->CDIIS_matrix,sizeof(double)*size*size);
    for (int i=0; i<size; i++) cdiis_rev[i*size+i]+=1E-7;
    memset(diis->extra_coe,0,sizeof(double)*size);
    diis->extra_coe[size-1]=-1;
    LAPACK_dsyev("V","U",&size,cdiis_rev,&size,egn_value,work,&lwork,&erro);
    memcpy(cdiis_rev,diis->CDIIS_matrix,sizeof(double)*size*size);
    LAPACK_dsytrf("U",&size,cdiis_rev,&size,ipiv,work,&lwork,&erro);
    LAPACK_dsytri("U",&size,cdiis_rev,&size,ipiv,work,&erro);
    for(int i=0;i<size;++i)  
    {  
        for(int j=i+1;j<size;++j)  
            cdiis_rev[i*size+j]=cdiis_rev[j*size+i];  
    }  
    double sum_extra=0.;
    for (i=0;i<diis->DIIS_p;i++)
    {
        diis->extra_coe[i]=-cdiis_rev[size*i+diis->DIIS_p];
        sum_extra+=diis->extra_coe[i];
    }
    for (i=0;i<diis->DIIS_p;i++) diis->extra_coe[i]/=sum_extra;
    free(ipiv); free(cdiis_rev); free(work); 
    /* Do ADIIS process */
    int if_finish=0, iter=0;
    double step=0.1,e_miuns=1,e_plus=1,e_ref=1;
    double step_p,step_m,sum,fact;
    double* inter_coe=diis->inter_coe;
    double* inter_coe_b=diis->inter_coe_b;
    int n=diis->DIIS_p;
    /// inital guess for adiis interploation
    inter_coe[0]=0.9;
    for (i=1;i<n;i++)
    {
        inter_coe[i]=(1-inter_coe[0])/(n-1);
    }
    e_ref=optim(inter_coe,n,size,diis->EDIIS_matrix,DIIS_vector);   // energy for inital step
    while (if_finish==0 && iter<500)
    {
        iter++;
        if_finish=1;
        for (i=0;i<n-1;i++)
        {
            for (j=i+1;j<n;j++)
            {
                step_p=ffmin(step,1.0-inter_coe[i],inter_coe[j]);
                inter_coe[i]+=step_p;
                inter_coe[j]-=step_p;
                e_plus=optim(inter_coe,n,size,diis->EDIIS_matrix,DIIS_vector);
                inter_coe[i]-=step_p;
                inter_coe[j]+=step_p;
                step_m=ffmin(step,inter_coe[i],1.0-inter_coe[j]);
                inter_coe[i]-=step_m;
                inter_coe[j]+=step_m;
                e_miuns=optim(inter_coe,n,size,diis->EDIIS_matrix,DIIS_vector);
                inter_coe[i]+=step_m;
                inter_coe[j]-=step_m;
                if (e_plus<e_miuns)
                {
                    if (e_plus<e_ref)
                    {
                        inter_coe[i]+=step_p;
                        inter_coe[j]-=step_p;
                        e_ref=e_plus;
                        if_finish=0;
                    }
                }
                else 
                {
                    if (e_miuns<e_ref)
                    {
                        inter_coe[i]-=step_m;
                        inter_coe[j]+=step_m;
                        e_ref=e_miuns;
                        if_finish=0;
                    }
                }
            }
        }
        if (if_finish==1 && step>0.9E-4)
        {
            step*=0.1;
            if_finish=0;
        }
        sum=0.0;
        for (i=0;i<n;i++)
        {
            if (inter_coe[i]>1.0) inter_coe[i]=1.0;
            else if (inter_coe[i]<0.0) inter_coe[i]=0.0;
            sum+=inter_coe[i];
        }
        fact=1.0/sum;
    }
    if_finish=0, iter=0;
    step=0.1,e_miuns=1,e_plus=1,e_ref=1;
    if (judge>=diis->tr1)
    {
        for (i=0;i<diis->DIIS_p;i++)
        {
            for (j=0;j<msize*msize;j++)
            {
                f_matrix[j]+=f_matrix_store[i][j]*inter_coe[i];
            }
        }
    }
    else if (judge>=diis->tr2)
    {
        double inter_p, extra_p;
        inter_p=(judge-diis->tr2)/(diis->tr1-diis->tr2);
        extra_p=1-inter_p;
        for (i=0;i<diis->DIIS_p;i++) 
        {
            inter_coe[i]=inter_coe[i]*inter_p+diis->extra_coe[i]*extra_p;
        }
        for (i=0;i<diis->DIIS_p;i++)
        {
            for (j=0;j<msize*msize;j++)
            {
                f_matrix[j]+=f_matrix_store[i][j]*inter_coe[i];
            }
        }
    }
    else
    {
        for (i=0;i<diis->DIIS_p;i++)
        {
            for (j=0;j<msize*msize;j++)
            {
                f_matrix[j]+=f_matrix_store[i][j]*diis->extra_coe[i];
            }
        }
    }
    if (diis->DIIS_p<size-1) diis->DIIS_p++;
    free(DIIS_vector);
    free(DIIS_vector_b);
    free(fd_matrix);
    free(fds_matrix);
    free(err_vector);
    free(egn_value);
    return -log10(diis->CDIIS_matrix[0]);
}

void set_diis_tr(double tr1, double tr2, diis_info diis)
{
    diis->tr1 = tr1;
    diis->tr2 = tr2;
    return;
}

df_diis_info init_df_diis(const int msize, const int size)
{
    df_diis_info df_diis = (df_diis_info)malloc(sizeof(struct DfDIISInfo));
    df_diis->p_size = 0;
    df_diis->msize = msize;
    df_diis->size = size;
    df_diis->df_coe = malloc_matrix(size, msize);
    df_diis->d_vector = malloc_matrix(size, msize);
    df_diis->err_matrix = malloc_matrix(size + 1, size + 1);
    df_diis->extra_coe = malloc_vector(size);
    return df_diis;
}

void del_df_diis(df_diis_info df_diis)
{
    free_vector(df_diis->extra_coe);
    free_matrix(df_diis->df_coe);
    free_matrix(df_diis->d_vector);
    free_matrix(df_diis->err_matrix);
    free(df_diis);
    return;
}

double do_df_diis(Vector d_vector, Vector df_coe, const Matrix kernel, df_diis_info df_diis)
{
    for (int i = df_diis->size - 1; i > 0; i--)
    {
        memcpy(df_diis->df_coe[i], df_diis->df_coe[i - 1], sizeof(double) * df_diis->msize);
        memcpy(df_diis->d_vector[i], df_diis->d_vector[i - 1], sizeof(double) * df_diis->msize);
    }
    memcpy(df_diis->df_coe[0], df_coe, sizeof(double) * df_diis->msize);
    memcpy(df_diis->d_vector[0], d_vector, sizeof(double) * df_diis->msize);
    if (df_diis->p_size < df_diis->size)
    {
        df_diis->p_size++;
    }
    if (df_diis->p_size == 1)
    {
        df_diis->extra_coe[0] = -1.;
        return 0;
    }
    for (int i = 0; i <= df_diis->size; i++)
        df_diis->err_matrix[i][i] = 1.;
    for (int i = 0; i < df_diis->p_size; i++)
        for (int j = 0; j <= i; j++)
        {
            df_diis->err_matrix[i][j] = 0.;
            if (kernel != NULL)
                for (int a = 0; a < df_diis->msize; a++)
                    for (int b = 0; b < df_diis->msize; b++)
                    {
                        df_diis->err_matrix[i][j] += df_diis->df_coe[i][a] * df_diis->df_coe[j][b] * kernel[a][b];
                    }
            else
                df_diis->err_matrix[i][j] = vv_dot(df_diis->df_coe[i], df_diis->df_coe[j], df_diis->msize);
            df_diis->err_matrix[j][i] = df_diis->err_matrix[i][j];
        }
    df_diis->err_matrix[df_diis->p_size][df_diis->p_size] = 0.;
    for (int i = 0; i < df_diis->p_size; i++)
    {
        df_diis->err_matrix[df_diis->p_size][i] = -1.;
        df_diis->err_matrix[i][df_diis->p_size] = -1.;
    }
    cal_symatrix_reverse(df_diis->err_matrix, df_diis->size + 1);
    memcpy(df_diis->extra_coe, df_diis->err_matrix[df_diis->p_size], sizeof(double) * df_diis->size);
    memset(df_coe, 0, sizeof(double) * df_diis->msize);
    for (int i = 0; i < df_diis->p_size; i++)
        for (int a = 0; a < df_diis->msize; a++)
            df_coe[a] += df_diis->df_coe[i][a] * df_diis->extra_coe[i];
    double err = 0.;
    if (kernel != NULL)
        for (int a = 0; a < df_diis->msize; a++)
            for (int b = 0; b < df_diis->msize; b++)
            {
                err += df_coe[a] * df_coe[b] * kernel[a][b];
            }
    else
        err = vv_dot(df_coe, df_coe, df_diis->msize);
    memset(d_vector, 0, sizeof(double) * df_diis->msize);
    for (int i = 0; i < df_diis->p_size; i++)
        for (int j = 0; j < df_diis->msize; j++)
            d_vector[j] -= df_diis->extra_coe[i] * df_diis->d_vector[i][j];
    // for (int i=0; i<df_diis->size; i++) printf("%d: %f\n",i,df_diis->extra_coe[i]);
    return -log10(err);
}
