#include "mol/mol.h"
List malloc_list(size_t size) { return (List)calloc(size, sizeof(int)); }
Vector malloc_vector(size_t size) { return (Vector)calloc(size, sizeof(double)); }
Matrix malloc_matrix(size_t size1, size_t size2)
{
    Matrix result = (Matrix)malloc(sizeof(Vector) * size1);
    if (size1 > 0)
        result[0] = malloc_vector(size1 * size2);
    for (int i = 1; i < size1; i++)
        result[i] = result[i - 1] + size2;
    return result;
}

Tensor3D malloc_tensor3d(size_t size1, size_t size2, size_t size3)
{
    Tensor3D result = (Tensor3D)malloc(sizeof(Matrix) * size1);
    if (size1 > 0)
    {
        result[0] = (Matrix)malloc(sizeof(Vector) * size1 * size2);
        for (int i = 1; i < size1; i++)
            result[i] = result[i - 1] + size2;
        if (size2 > 0)
            result[0][0] = malloc_vector(size1 * size2 * size3);
        for (int i = 1; i < size1; i++)
            result[i][0] = result[i - 1][0] + size2 * size3;
        for (int j = 0; j < size1; j++)
            for (int i = 1; i < size2; i++)
                result[j][i] = result[j][i - 1] + size3;
    }
    return result;
}

Tensor4D malloc_tensor4d(size_t size1, size_t size2, size_t size3, size_t size4)
{
    Tensor4D result = (Tensor4D)malloc(sizeof(Tensor3D) * size1);
    if (size1 > 0)
    {
        result[0] = (Tensor3D)malloc(sizeof(Matrix) * size1 * size2);
        for (int i = 1; i < size1; i++)
            result[i] = result[i - 1] + size2;
        if (size2 > 0)
            result[0][0] = (Matrix)malloc(sizeof(Vector) * size1 * size2 * size3);
        for (int j = 0; j < size1; j++)
        {
            if (j != 0)
                result[j][0] = result[j - 1][size2 - 1] + size3;
            for (int i = 1; i < size2; i++)
                result[j][i] = result[j][i - 1] + size3;
        }
        if (size3 > 0)
            result[0][0][0] = malloc_vector(size1 * size2 * size3 * size4);
        for (int k = 0; k < size1; k++)
        {
            if (k != 0)
                result[k][0][0] = result[k - 1][size2 - 1][size3 - 1] + size4;
            for (int j = 0; j < size2; j++)
            {
                if (j != 0)
                    result[k][j][0] = result[k][j - 1][size3 - 1] + size4;
                for (int i = 1; i < size3; i++)
                    result[k][j][i] = result[k][j][i - 1] + size4;
            }
        }
    }
    return result;
}
void free_list(List data)
{
    free(data);
    data = NULL;
    return;
}
void free_vector(Vector data)
{
    free(data);
    data = NULL;
    return;
}

void free_matrix(Matrix data)
{
    free(data[0]);
    free(data);
    data = NULL;
    return;
}

void free_tensor3d(Tensor3D data)
{
    free(data[0][0]);
    free(data[0]);
    free(data);
    data = NULL;
    return;
}

void free_tensor4d(Tensor4D data)
{
    free(data[0][0][0]);
    free(data[0][0]);
    free(data[0]);
    free(data);
    data = NULL;
    return;
}

double vv_dot(const Vector a, const Vector b, const int m1)
{
    return cblas_ddot(m1, a, 1, b, 1);
}
double matrix_inner(const Matrix a, const Matrix b, int m_size)
{
    return vv_dot(a[0], b[0], m_size * m_size);
}

// C1(i,k)C2(j,l)F(k,l)--->P(i,j) C_1AC_2^T
extern void gtrafock(Matrix f_out, const Matrix c_matrix1, const Matrix c_matrix2, const Matrix f_matrix,
                     const int size1, const int size2, const int mszie)
{
    if (size1 == 0 || size2 == 0)
        return;
    Matrix tmp_f = malloc_matrix(size1, mszie);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size1, mszie, mszie, 1., c_matrix1[0], mszie, f_matrix[0], mszie, 0., tmp_f[0], mszie);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, size1, size2, mszie, 1., tmp_f[0], mszie, c_matrix2[0], mszie, 0., f_out[0], size2);
    free_matrix(tmp_f);
    return;
}
void trafock(Vector f_matrix, Vector x_matrix, int msize)
{
    int i, j, k;
    double *f_matrix_tmp = (Vector)malloc(sizeof(double) * msize * msize);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, msize, msize, msize, 1.0, f_matrix, msize, x_matrix, msize, 0.0, f_matrix_tmp, msize);
    for (i = 0; i < msize; i++)
    {
        for (j = 0; j <= i; j++)
        {
            f_matrix[i * msize + j] = 0.0;
            for (k = 0; k < msize; k++)
            {
                f_matrix[i * msize + j] += x_matrix[i * msize + k] * f_matrix_tmp[k * msize + j];
            }
            f_matrix[j * msize + i] = f_matrix[i * msize + j];
        }
    }
    free(f_matrix_tmp);
}

// C1(i,k)C2(j,l)F(k,l)--->P(i,j) C_1^TAC_2
extern void gtrafock_t(Matrix f_out, const Matrix c_matrix1, const Matrix c_matrix2, const Matrix f_matrix,
                       const int size1, const int size2, const int mszie)
{
    Matrix tmp_f = malloc_matrix(size1, mszie);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, size1, mszie, mszie, 1., c_matrix1[0], size1, f_matrix[0], mszie, 0., tmp_f[0], mszie);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size1, size2, mszie, 1., tmp_f[0], mszie, c_matrix2[0], size2, 0., f_out[0], size2);
    free_matrix(tmp_f);
    return;
}
// D1(i,l)D2(j,l)D3(k,l)--->T(i,j,k)
extern void tensor_contr(Tensor3D T, const Matrix D1, int a, const Matrix D2, int b, const Matrix D3, int c, int d)
{
    int ab = a * b;
    Matrix DD12 = malloc_matrix(d, ab);
    for (int i = 0; i < d; i++)
        cblas_dger(CblasRowMajor, a, b, 1., D1[i], 1, D2[i], 1, DD12[i], b);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ab, c, d, 1.0, DD12[0], ab, D3[0], c, 1.0, T[0][0], c);
    free_matrix(DD12);
    return;
}

/**
 * @brief do modified Gram-Schmit
 * @note reference NUMERCIAL LINEAR ALGEGRA pg.58
 * @param matrix
 * @param msize
 */
void GramSchmit_orth(Matrix matrix, int msize)
{
    Matrix R = malloc_matrix(msize, msize);
    Vector q;
    for (int i = 0; i < msize; i++)
    {
        R[i][i] = vv_dot(matrix[i], matrix[i], msize);
        q = matrix[i];
        for (int j = 0; j < msize; j++)
            q[j] /= sqrt(R[i][i]);
        for (int j = i + 1; j < msize; j++)
        {
            R[i][j] = vv_dot(q, matrix[j], msize);
            for (int k = 0; k < msize; k++)
                matrix[j][k] -= q[k] * R[i][j];
        }
    }
    free_matrix(R);
    return;
}

int GramSchmit_ld_check(Matrix c_matrix, const Matrix s_matrix, int size1, int msize)
{
    Matrix R = malloc_matrix(size1, msize);
    Vector q, tmp = malloc_vector(msize);
    List ld_list = malloc_list(size1);
    int ld_num = 0;
    for (int i = 0; i < size1; i++)
    {
        q = c_matrix[i];
        if (s_matrix != NULL)
        {
            cblas_dgemv(CblasRowMajor, CblasNoTrans, msize, msize, 1., s_matrix[0], msize, q, 1, 0., tmp, 1.);
            R[i][i] = vv_dot(q, tmp, msize);
        }
        else
            R[i][i] = vv_dot(q, q, msize);
        if (fabs(R[i][i]) < 1E-8)
            continue;
        ld_list[ld_num] = i;
        ld_num++;
        for (int j = 0; j < msize; j++)
            q[j] /= sqrt(R[i][i]);
        for (int j = i + 1; j < size1; j++)
        {
            if (s_matrix != NULL)
            {
                cblas_dgemv(CblasRowMajor, CblasNoTrans, msize, msize, 1., s_matrix[0], msize, q, 1, 0., tmp, 1.);
                R[i][j] = vv_dot(tmp, c_matrix[j], msize);
            }
            else
                R[i][j] = vv_dot(q, c_matrix[j], msize);
            for (int k = 0; k < msize; k++)
                c_matrix[j][k] -= q[k] * R[i][j];
        }
    }
    for (int i = 0; i < ld_num; i++)
        memcpy(R[i], c_matrix[ld_list[i]], sizeof(double) * msize);
    memcpy(c_matrix[0], R[0], sizeof(double) * msize * size1);
    free_matrix(R);
    free(ld_list);
    return ld_num;
}

/**
 * @brief calculation the reversion of hermitian matrix
 *
 * @param[inout] matrix target
 * @param[in] the size of matrix
 */
extern void cal_symatrix_reverse(Matrix matrix, int size)
{
    int *ipiv = (int *)malloc(sizeof(int) * (size));
    memset(ipiv,0,sizeof(int)*size);
    LAPACKE_dsytrf(LAPACK_COL_MAJOR, 'U', size, matrix[0], size, ipiv);
    LAPACKE_dsytri(LAPACK_COL_MAJOR, 'U', size, matrix[0], size, ipiv);
    for (int i = 0; i < size; ++i)
    {
        for (int j = i + 1; j < size; ++j)
            matrix[i][j] = matrix[j][i];
    }
    free(ipiv);
    return;
}

extern void cal_matrix_reverse(Matrix matrix, int size)
{
    int *ipiv = (int *)malloc(sizeof(int) * (size));
    LAPACKE_dgetrf(LAPACK_ROW_MAJOR, size, size, matrix[0], size, ipiv);
    LAPACKE_dgetri(LAPACK_ROW_MAJOR, size, matrix[0], size, ipiv);
    free(ipiv);
    return;
}

void cal_matrix_halfrev(Matrix matrix, int size, char uplo)
{
    // do Cholesky decomposition of (P|Q) ---> [(P|Q)]^{-1/2}
    LAPACKE_dpotrf(LAPACK_ROW_MAJOR, uplo, size, matrix[0], size);
    // solve Lower triangle matrix linear equation
    if (uplo == 'L')
        for (int i = 0; i < size; i++)
        {
            matrix[i][i] = 1. / matrix[i][i];
            for (int j = 1; j < size - i; j++)
            {
                double tmp = 0.;
                for (int k = 0; k < size - i - 1; k++)
                    tmp -= matrix[i][k + i] * matrix[i + k][i];
                matrix[i + j][i] = tmp / matrix[i + j][i];
            }
        }
    else if (uplo == 'U')
        for (int i = 0; i < size; i++)
        {
            matrix[i][i] = 1. / matrix[i][i];
            for (int j = 1; j < size - i; j++)
            {
                double tmp = 0.;
                for (int k = 0; k < size - i - 1; k++)
                    tmp -= matrix[k + i][i] * matrix[i][i + k];
                matrix[i][i + j] = tmp / matrix[i][i + j];
            }
        }
    return;
}

void cal_matrix_syhalfrev(Matrix matrix, int size)
{
    Vector egn_value = malloc_vector(size);
    Vector s_matrix_ = malloc_vector(size * size);
    Vector s_matrix_cp = malloc_vector(size * size);
    memcpy(s_matrix_, matrix[0], sizeof(double) * size * size);
    LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', size, s_matrix_, size, egn_value);
    for (int i = 0; i < size; i++)
    {
        egn_value[i] = 1 / sqrt(egn_value[i]);
        if (egn_value[i] < 1E-3 || !isnormal(egn_value[i]))
            egn_value[i] = 0.;
    }
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            s_matrix_cp[i * size + j] = s_matrix_[i * size + j] * egn_value[i];
        }
    }
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            matrix[i][j] = 0.0;
            for (int k = 0; k < size; k++)
            {
                matrix[i][j] += s_matrix_[k * size + i] * s_matrix_cp[k * size + j];
            }
            matrix[j][i] = matrix[i][j];
        }
    }
    free(egn_value);
    free(s_matrix_);
    free(s_matrix_cp);
    return;
}

double cal_deter(Matrix matrix, int size)
{
    Matrix tmp = malloc_matrix(size, size);
    double result = 0.;
    copy_matrix(tmp, matrix, size, size);
    int *ipiv = (int *)malloc(sizeof(int) * (size));
    LAPACKE_dgetrf(LAPACK_ROW_MAJOR, size, size, tmp[0], size, ipiv);
    for (int i = 0; i < size; i++)
        result *= tmp[i][i];
    free_matrix(tmp);
    return result;
}

void print_matrix(Matrix matrix, int x, int y)
{
    int sec_num = (int)ceil(x / 5.);
    for (int i = 0, n = 0; i < sec_num; i++)
    {
        int m = n;
        printf("================================================="
               "======================\n               ");
        // for (int u=0; u<5; u++,m++) printf("%-12d ",m+1);
        printf("\n================================================="
               "======================\n");
        for (int j = 0; j < y; j++)
        {
            m = n;
            printf("%-5d ", j + 1);
            for (int u = 0; u < 5 && m < x; u++, m++)
            {
                printf("%12.6f", matrix[j][m]);
            }
            printf("\n");
        }
        n = m;
    }
    fflush(stdout);
    return;
}

double Fact(int a)
{
    double r = 1.;
    if (a == 0)
        return 1;
    else
        for (int i = a; i > a; i--)
            r *= i;
    return r;
}

int idnint(double a)
{
    if (a < 0)
        return (int)(a - 0.5);
    else
        return (int)(a + 0.5);
}

double Comb(int n, int m)
{
    return Fact(n) / (Fact(n - m) * Fact(m) * 1.0);
}
