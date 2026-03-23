#include "mol/xint.h"
#define CACHE_MAX_LEN 4096
static Tensor3D slice_tensor3d(int size1, int size2, int size3, const Vector cache)
{
    Tensor3D result = (Tensor3D)malloc(sizeof(Matrix) * size1);
    result[0] = (Matrix)malloc(sizeof(Vector) * size1 * size2);
    for (int i = 1; i < size1; i++)
        result[i] = result[i - 1] + size2;
    result[0][0] = cache;
    for (int i = 1; i < size1; i++)
        result[i][0] = result[i - 1][0] + size2 * size3;
    for (int j = 0; j < size1; j++)
        for (int i = 1; i < size2; i++)
            result[j][i] = result[j][i - 1] + size3;
    return result;
}
static Matrix slice_matrix(int size1, int size2, const Vector cache)
{
    Matrix result = (Matrix)malloc(sizeof(Vector) * size1);
    result[0] = cache;
    for (int i = 1; i < size1; i++)
        result[i] = result[i - 1] + size2;
    return result;
}
xint_info init_xint(const mol_info mol)
{
    xint_info xint = (xint_info)malloc(sizeof(struct XIntInfo));
    xint->mol = mol;
    int max_ang = 0;
    for (int i = 0; i < get_mol_nbas(mol); i++)
    {
        if (mol->bas(ANGULAR_VAL, i) > max_ang)
            max_ang = mol->bas(ANGULAR_VAL, i);
    }
    xint->cache = malloc_vector(CACHE_MAX_LEN * 15);
    xint->max_ang = max_ang;
    xint->max_ang_aux = -1;
    xint->lr_frac = 1.;
    xint->sr_frac = 0.;
    xint->omega = 0.;
    xint->if_3c2e = false;
    return xint;
}
void load_xint_aux(const bas_info aux, xint_info xint)
{
    int max_ang_aux = 0;
    int max_ang = xint->max_ang;
    for (int i = 0; i < aux->nbas; i++)
    {
        if (aux(ANGULAR_VAL, i) > max_ang_aux)
            max_ang_aux = aux(ANGULAR_VAL, i);
    }
    xint->aux = aux;
    xint->max_ang_aux = max_ang_aux;
    xint->if_3c2e = true;
    // slice the cache for Rys 3c2e-integrals
    int cache3c2e_len = 0;
    xint->rr3c2e_x0 = slice_matrix(max_ang_aux + 1, 2 * max_ang + 1, &xint->cache[cache3c2e_len]);
    cache3c2e_len += (max_ang_aux + 1) * (2 * max_ang + 1);
    xint->rr3c2e_y0 = slice_matrix(max_ang_aux + 1, 2 * max_ang + 1, &xint->cache[cache3c2e_len]);
    cache3c2e_len += (max_ang_aux + 1) * (2 * max_ang + 1);
    xint->rr3c2e_z0 = slice_matrix(max_ang_aux + 1, 2 * max_ang + 1, &xint->cache[cache3c2e_len]);
    cache3c2e_len += (max_ang_aux + 1) * (2 * max_ang + 1);
    xint->rr3c2e_x = slice_tensor3d(max_ang_aux + 1, 2 * max_ang + 1, max_ang + 1, &xint->cache[cache3c2e_len]);
    cache3c2e_len += (max_ang_aux + 1) * (2 * max_ang + 1) * (max_ang + 1);
    xint->rr3c2e_y = slice_tensor3d(max_ang_aux + 1, 2 * max_ang + 1, max_ang + 1, &xint->cache[cache3c2e_len]);
    cache3c2e_len += (max_ang_aux + 1) * (2 * max_ang + 1) * (max_ang + 1);
    xint->rr3c2e_z = slice_tensor3d(max_ang_aux + 1, 2 * max_ang + 1, max_ang + 1, &xint->cache[cache3c2e_len]);
    cache3c2e_len += (max_ang_aux + 1) * (2 * max_ang + 1) * (max_ang + 1);
    xint->rr3c2e_Ax = slice_tensor3d(max_ang_aux + 1, max_ang + 1, max_ang + 1, &xint->cache[cache3c2e_len]);
    cache3c2e_len += (max_ang_aux + 1) * (max_ang + 1) * (max_ang + 1);
    xint->rr3c2e_Ay = slice_tensor3d(max_ang_aux + 1, max_ang + 1, max_ang + 1, &xint->cache[cache3c2e_len]);
    cache3c2e_len += (max_ang_aux + 1) * (max_ang + 1) * (max_ang + 1);
    xint->rr3c2e_Az = slice_tensor3d(max_ang_aux + 1, max_ang + 1, max_ang + 1, &xint->cache[cache3c2e_len]);
    cache3c2e_len += (max_ang_aux + 1) * (max_ang + 1) * (max_ang + 1);
    xint->rr3c2e_Bx = slice_tensor3d(max_ang_aux + 1, max_ang + 1, max_ang + 1, &xint->cache[cache3c2e_len]);
    cache3c2e_len += (max_ang_aux + 1) * (max_ang + 1) * (max_ang + 1);
    xint->rr3c2e_By = slice_tensor3d(max_ang_aux + 1, max_ang + 1, max_ang + 1, &xint->cache[cache3c2e_len]);
    cache3c2e_len += (max_ang_aux + 1) * (max_ang + 1) * (max_ang + 1);
    xint->rr3c2e_Bz = slice_tensor3d(max_ang_aux + 1, max_ang + 1, max_ang + 1, &xint->cache[cache3c2e_len]);
    cache3c2e_len += (max_ang_aux + 1) * (max_ang + 1) * (max_ang + 1);
    return;
}
void set_xint_rs(double lr_frac, double sr_frac, double omega, xint_info xint)
{
    xint->lr_frac = lr_frac;
    xint->sr_frac = sr_frac;
    xint->omega = omega;
    return;
}
static void free_tensor3d_o(Tensor3D data)
{
    free(data[0]);
    free(data);
    return;
}
static void free_matrix_o(Matrix data)
{
    free(data);
    return;
}
void del_xint(xint_info xint)
{
    free_vector(xint->cache);
    if (xint->if_3c2e)
    {
        free_matrix_o(xint->rr3c2e_x0);
        free_matrix_o(xint->rr3c2e_y0);
        free_matrix_o(xint->rr3c2e_z0);
        free_tensor3d_o(xint->rr3c2e_x);
        free_tensor3d_o(xint->rr3c2e_y);
        free_tensor3d_o(xint->rr3c2e_z);
        free_tensor3d_o(xint->rr3c2e_Ax);
        free_tensor3d_o(xint->rr3c2e_Ay);
        free_tensor3d_o(xint->rr3c2e_Az);
        free_tensor3d_o(xint->rr3c2e_Bx);
        free_tensor3d_o(xint->rr3c2e_By);
        free_tensor3d_o(xint->rr3c2e_Bz);
    }
    free(xint);
    return;
}
