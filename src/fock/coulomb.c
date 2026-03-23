#include "fock/fock.h"
coulomb_info init_coulomb(int batch_size, const mol_info mol)
{
    int msize = get_mol_msize(mol);
    coulomb_info coulomb = (coulomb_info)malloc(sizeof(struct CoulombInfo));
    coulomb->type = None_jbuilder;
    coulomb->mol = mol;
    coulomb->if_aux = false;
    coulomb->if_grids = false;
    coulomb->if_final = false;
    coulomb->batch_size = batch_size;
    if (batch_size > 0)
        coulomb->J = malloc_tensor3d(batch_size, msize, msize);
    else
        coulomb->J = NULL;
    return coulomb;
}
void set_coulomb_type(Jbuilder_type type, coulomb_info coulomb)
{
    coulomb->type = type;
    return;
}
void load_coulomb_aux(const char *aux_fname, coulomb_info coulomb)
{
    coulomb->if_aux = true;
    coulomb->aux = init_bas(aux_fname, coulomb->mol->atm);
    int msize = coulomb->aux->msize;
    coulomb->aux2e_matrix = malloc_matrix(msize, msize);
    cal_2c2e_matrix(coulomb->aux2e_matrix, coulomb->aux, coulomb->mol);
    if (coulomb->batch_size > 0)
        coulomb->df_coe = malloc_matrix(coulomb->batch_size, msize);
    else
        coulomb->df_coe = NULL;
    return;
}
void load_coulomb_grids(const char *grids_fname, const char *fgrids_fname, coulomb_info coulomb)
{
    int msize = get_mol_msize(coulomb->mol);
    coulomb->if_grids = true;
    coulomb->grids = init_grid_opt(grids_fname, coulomb->mol);
    Matrix s_matrix = malloc_matrix(msize, msize);
    cal_overlap_matrix(s_matrix, coulomb->mol);
    Matrix s_matrix_num = cal_overlap_numerical(coulomb->grids);
    coulomb->o_matrix = malloc_matrix(msize, msize);
    cal_symatrix_reverse(s_matrix_num, msize);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, msize, msize, msize, 1.,
                s_matrix[0], msize, s_matrix_num[0], msize, 0., coulomb->o_matrix[0], msize);
    if (fgrids_fname != NULL)
    {
        coulomb->grids_final = init_grid_opt(fgrids_fname, coulomb->mol);
        free_matrix(s_matrix_num);
        s_matrix_num = cal_overlap_numerical(coulomb->grids_final);
        coulomb->o_matrix_final = malloc_matrix(msize, msize);
        cal_symatrix_reverse(s_matrix_num, msize);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, msize, msize, msize, 1.,
                    s_matrix[0], msize, s_matrix_num[0], msize, 0., coulomb->o_matrix_final[0], msize);
    }
    else
        coulomb->grids_final = NULL;
    free_matrix(s_matrix);
    free_matrix(s_matrix_num);
    return;
}
void del_coulomb(coulomb_info coulomb)
{
    if (coulomb->J != NULL)
        free_tensor3d(coulomb->J);
    if (coulomb->if_aux)
    {
        free_matrix(coulomb->aux2e_matrix);
        del_bas(coulomb->aux);
        if (coulomb->df_coe != NULL)
            free_matrix(coulomb->df_coe);
    }
    if (coulomb->if_grids)
    {
        del_grid(coulomb->grids);
        free_matrix(coulomb->o_matrix);
        if (coulomb->grids_final != NULL)
        {
            del_grid(coulomb->grids_final);
            free_matrix(coulomb->o_matrix_final);
        }
    }
    free(coulomb);
    return;
}
/**
 * @brief calculate coulomb matrix by resolution identity (RI)
 *        DOI: 10.1039/b204199p
 *
 * @param if_finnish
 * @param d_matrix
 * @param coulomb
 */
static void cal_j_matrix_ri(const bool *if_finnish, const Matrix *d_matrix, coulomb_info coulomb)
{
    assert(coulomb->if_aux);
    const mol_info mol = coulomb->mol;
    const bas_info aux = coulomb->aux;
    const bas_info bas = mol->bas;
    int batch_size = coulomb->batch_size;
    int naux = aux->nbas;
    int aux_msize = aux->msize;
    int msize = get_mol_msize(mol);
    Matrix rho_aux = malloc_matrix(batch_size, aux_msize);
#pragma omp parallel
    {
        xint_info xint = init_xint(mol);
        load_xint_aux(aux, xint);
        int max_ang = xint->max_ang;
        int max_ang_aux = xint->max_ang_aux;
        Vector buf = malloc_vector(xint_gtolen(max_ang) * xint_gtolen(max_ang) * xint_gtolen(max_ang_aux));
        Matrix rho_aux_tmp = malloc_matrix(batch_size, aux_msize);
        shell_pair shp;
        double factor = 2.;
#pragma omp for schedule(dynamic, 4) nowait
        for (int ij = 0; ij < mol->pair->len; ij++)
        {
            shp = &mol->pair->shp[ij];
            int shli = shp->shli, shlj = shp->shlj;
            int li = shp->li, lj = shp->lj;
            int di = xint_gtolen(li), dj = xint_gtolen(lj);
            int pi = bas->shls_p[shli], pj = bas->shls_p[shlj];
            if (shli == shlj)
                for (int a = 0; a < naux; a++)
                {
                    xint_3c2e(buf, ij, a, xint);
                    int pa = aux->shls_p[a];
                    int da = xint_gtolen(aux(ANGULAR_VAL, a));
                    for (int a = 0; a < batch_size; a++)
                    {
                        if (if_finnish[a] == 1)
                            continue;
                        for (int l = 0, uu = 0; l < da; l++)
                            for (int m = 0; m < di; m++)
                                for (int n = 0; n < dj; n++, uu++)
                                {
                                    rho_aux_tmp[a][l + pa] += d_matrix[a][pi + m][pj + n] * buf[uu];
                                }
                    }
                }
            else
                for (int a = 0; a < naux; a++)
                {
                    xint_3c2e(buf, ij, a, xint);
                    int pa = aux->shls_p[a];
                    int da = xint_gtolen(aux(ANGULAR_VAL, a));
                    for (int a = 0; a < batch_size; a++)
                    {
                        if (if_finnish[a] == 1)
                            continue;
                        for (int l = 0, uu = 0; l < da; l++)
                            for (int m = 0; m < di; m++)
                                for (int n = 0; n < dj; n++, uu++)
                                {
                                    rho_aux_tmp[a][l + pa] += (d_matrix[a][pi + m][pj + n] + d_matrix[a][pj + n][pi + m]) * buf[uu];
                                }
                    }
                }
        }
#pragma omp critical
        {
            for (int a = 0; a < batch_size; a++)
                for (int i = 0; i < aux_msize; i++)
                    rho_aux[a][i] += rho_aux_tmp[a][i];
        }
        free_matrix(rho_aux_tmp);
        free_vector(buf);
        del_xint(xint);
    }
    List ipiv = malloc_list(aux_msize);
    int lwork = 1;
    Vector aux2e_matrix = malloc_vector(aux_msize * aux_msize);
    for (int a = 0; a < batch_size; a++)
    {
        if (if_finnish[a] == 1)
            continue;
        memcpy(aux2e_matrix, coulomb->aux2e_matrix[0], sizeof(double) * aux_msize * aux_msize);
        LAPACKE_dgesv(LAPACK_ROW_MAJOR, aux_msize, lwork, aux2e_matrix, aux_msize, ipiv, rho_aux[a], 1);
        memcpy(coulomb->df_coe[a], rho_aux[a], sizeof(double) * aux_msize);
    }
    free(ipiv);
    free(aux2e_matrix);
#pragma omp parallel
    {
        xint_info xint = init_xint(mol);
        load_xint_aux(aux, xint);
        int max_ang = xint->max_ang;
        int max_ang_aux = xint->max_ang_aux;
        Vector buf = malloc_vector(xint_gtolen(max_ang) * xint_gtolen(max_ang) * xint_gtolen(max_ang_aux));
        Tensor3D J_tmp = malloc_tensor3d(batch_size, msize, msize);
        shell_pair shp;
        double factor = 1.;
#pragma omp for schedule(dynamic, 4) nowait
        for (int ij = 0; ij < mol->pair->len; ij++)
        {
            shp = &mol->pair->shp[ij];
            int shli = shp->shli, shlj = shp->shlj;
            int li = shp->li, lj = shp->lj;
            int di = xint_gtolen(li), dj = xint_gtolen(lj);
            int pi = bas->shls_p[shli], pj = bas->shls_p[shlj];
            if (shli == shlj)
                factor = 0.5;
            else
                factor = 1.;
            for (int a = 0; a < naux; a++)
            {
                xint_3c2e(buf, ij, a, xint);
                int pa = aux->shls_p[a];
                int da = xint_gtolen(aux(ANGULAR_VAL, a));
                for (int a = 0; a < batch_size; a++)
                {
                    if (if_finnish[a] == 1)
                        continue;
                    for (int l = 0, uu = 0; l < da; l++)
                        for (int m = 0; m < di; m++)
                            for (int n = 0; n < dj; n++, uu++)
                            {
                                J_tmp[a][pi + m][pj + n] += rho_aux[a][pa + l] * buf[uu] * factor;
                                J_tmp[a][pj + n][pi + m] += rho_aux[a][pa + l] * buf[uu] * factor;
                            }
                }
            }
        }
#pragma omp critical
        {
            for (int a = 0; a < batch_size; a++)
                for (int i = 0; i < msize * msize; i++)
                    coulomb->J[a][0][i] += J_tmp[a][0][i];
        }
        free_tensor3d(J_tmp);
        free_vector(buf);
        del_xint(xint);
    }
    free_matrix(rho_aux);
    return;
}

/**
 * @brief calculate coulomb matrix by only 2c1e integrals
 *
 * @param if_finnish
 * @param d_matrix
 * @param coulomb
 */
static void cal_j_matrix_thc(const bool *if_finnish, const Matrix *d_matrix, coulomb_info coulomb)
{
    assert(coulomb->if_aux && coulomb->if_grids);
    const mol_info mol = coulomb->mol;
    const bas_info aux = coulomb->aux;
    const bas_info bas = mol->bas;
    int batch_size = coulomb->batch_size;
    int naux = aux->nbas;
    int aux_msize = aux->msize;
    int msize = get_mol_msize(mol);
    grids_info grids = coulomb->grids;
    Matrix rho_aux = malloc_matrix(batch_size, aux_msize);
/* calculate electronic potential \Gamma_g=\sum_i{C_i(\rho|i)} */
#pragma omp parallel
    {
        double buf[200], G[3];
        xint_info xint = init_xint(mol);
        load_xint_aux(aux, xint);
        Matrix rho_aux_tmp = malloc_matrix(batch_size, aux_msize);
#pragma omp for schedule(dynamic, 1) nowait
        for (int n = 0; n < grids->block_num; n++)
        {
            grid_block block = &grids->blocks[n];
            int grid_num = block->grid_num;
            int shell_num = block->shell_num;
            int bas_num = block->bas_num;
            List global2local = block->global2local;
            List local2global = block->local2global;
            if (bas_num == 0)
                continue;
            Tensor3D d_matrix_local = malloc_tensor3d(batch_size, bas_num, bas_num);
            Matrix rho = malloc_matrix(batch_size, grid_num);
            cal_block_bas(n, grids);
            for (int a = 0; a < batch_size; a++)
            {
                if (if_finnish[a] == 1)
                    continue;
                for (int i = 0; i < bas_num; i++)
                    for (int j = 0; j < bas_num; j++)
                    {
                        d_matrix_local[a][i][j] = (d_matrix[a][local2global[i]][local2global[j]] +
                                                   d_matrix[a][local2global[j]][local2global[i]]) *
                                                  0.5;
                    }
                cal_rho(rho[a], d_matrix_local[a], block);
            }
            for (int g = 0; g < grid_num; g++)
            {
                double weight = grids->grids[block->grid_list[g]].w;
                G[0] = grids->grids[block->grid_list[g]].x;
                G[1] = grids->grids[block->grid_list[g]].y;
                G[2] = grids->grids[block->grid_list[g]].z;
                for (int i = 0; i < naux; i++)
                {
                    // if (fabs(rho[g]*coulomb->aux2e_rev[i])<1E-12) continue;
                    xint_2c1e(buf, i, G, xint);
                    int di = xint_gtolen(aux(ANGULAR_VAL, i));
                    int pi = aux->shls_p[i];
                    for (int u = 0; u < di; u++)
                    {
                        double tmp_buf = buf[u] * weight;
                        for (int a = 0; a < batch_size; a++)
                            rho_aux_tmp[a][pi + u] += rho[a][g] * tmp_buf;
                    }
                }
            }
            free_tensor3d(d_matrix_local);
            free_matrix(rho);
            clear_block_bas(block);
        }
#pragma omp critical
        for (int i = 0; i < batch_size * aux_msize; i++)
        {
            rho_aux[0][i] += rho_aux_tmp[0][i];
        }
        del_xint(xint);
        free_matrix(rho_aux_tmp);
    }
    List ipiv = malloc_list(aux_msize);
    int lwork = 1;
    Vector aux2e_matrix = malloc_vector(aux_msize * aux_msize);
    for (int a = 0; a < batch_size; a++)
    {
        if (if_finnish[a] == 1)
            continue;
        memcpy(aux2e_matrix, coulomb->aux2e_matrix[0], sizeof(double) * aux_msize * aux_msize);
        LAPACKE_dgesv(LAPACK_ROW_MAJOR, aux_msize, lwork, aux2e_matrix, aux_msize, ipiv, rho_aux[a], 1);
    }
    free(ipiv);
    free(aux2e_matrix);
/* Get J matrix by using electronic potential J_{ij}=\sum_g{\Gamma_gX_igX_jg} */
#pragma omp parallel
    {
        double buf[200], G[3];
        xint_info xint = init_xint(mol);
        load_xint_aux(aux, xint);
        Matrix rho_aux_tmp = malloc_matrix(batch_size, aux_msize);
        Tensor3D j_matrix_tmp = malloc_tensor3d(batch_size, msize, msize);
#pragma omp for schedule(dynamic, 1) nowait
        for (int n = 0; n < grids->block_num; n++)
        {
            grid_block block = &grids->blocks[n];
            int grid_num = block->grid_num;
            int shell_num = block->shell_num;
            int bas_num = block->bas_num;
            if (bas_num == 0)
                continue;
            List local2global = block->local2global;
            Matrix hartree_potential = malloc_matrix(batch_size, grid_num);
            Tensor3D j_matrix_local = malloc_tensor3d(batch_size, bas_num, bas_num);
            cal_block_bas(n, grids);
            for (int i = 0; i < naux; i++)
            {
                // if (fabs(rho_aux[i])<1E-12) continue;
                for (int g = 0; g < grid_num; g++)
                {
                    G[0] = grids->grids[block->grid_list[g]].x;
                    G[1] = grids->grids[block->grid_list[g]].y;
                    G[2] = grids->grids[block->grid_list[g]].z;
                    xint_2c1e(buf, i, G, xint);
                    int di = xint_gtolen(aux(ANGULAR_VAL, i));
                    int pi = aux->shls_p[i];
                    for (int a = 0; a < batch_size; a++)
                        for (int u = 0; u < di; u++)
                            hartree_potential[a][g] += rho_aux[a][u + pi] * buf[u];
                }
            }
            Matrix psi_tmp = malloc_matrix(grid_num, bas_num);
            Matrix psi_tmp_ = malloc_matrix(grid_num, bas_num);
            for (int a = 0; a < batch_size; a++)
            {
                if (if_finnish[a] == 1)
                    continue;
                memcpy(psi_tmp[0], block->psi[0], sizeof(double) * grid_num * bas_num);
                memcpy(psi_tmp_[0], block->psi[0], sizeof(double) * grid_num * bas_num);
                for (int g = 0; g < block->grid_num; g++)
                {
                    double weight = sqrt(grids->grids[block->grid_list[g]].w);
                    for (int i = 0; i < block->bas_num; i++)
                    {
                        psi_tmp_[g][i] *= weight;
                        psi_tmp[g][i] *= weight * hartree_potential[a][g];
                    }
                }
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, bas_num, bas_num, grid_num, 1.0, psi_tmp_[0], bas_num, psi_tmp[0],
                            bas_num, 0.0, j_matrix_local[a][0], bas_num);
            }
            free_matrix(psi_tmp);
            free_matrix(psi_tmp_);
            for (int a = 0; a < batch_size; a++)
            {
                if (if_finnish[a] == 1)
                    continue;
                for (int i = 0; i < bas_num; i++)
                {
                    int pi = local2global[i];
                    for (int j = 0; j < bas_num; j++)
                    {
                        int pj = local2global[j];
                        j_matrix_tmp[a][pi][pj] += j_matrix_local[a][i][j];
                    }
                }
            }
            free_tensor3d(j_matrix_local);
            free_matrix(hartree_potential);
            clear_block_bas(block);
        }
#pragma omp critical
        {
            for (int i = 0; i < msize * msize * batch_size; i++)
                coulomb->J[0][0][i] += j_matrix_tmp[0][0][i] / MY_PI;
        }
        del_xint(xint);
        free_matrix(rho_aux_tmp);
        free_tensor3d(j_matrix_tmp);
    }
    free_matrix(rho_aux);
    return;
}

/**
 * @brief calculate coulomb matrix in semi-numerical way and electron integrals is calculated under pc
 *
 * @param if_finnish
 * @param d_matrix
 * @param coulomb
 */
static void cal_j_matrix_sn(const bool *if_finnish, const Matrix *d_matrix, coulomb_info coulomb)
{
    assert(coulomb->if_grids);
    const mol_info mol = coulomb->mol;
    const prm_info prm = mol->prm;
    const pair_info pair = mol->prm_pair;
    int batch_size = coulomb->batch_size;
    int msize = get_mol_msize(mol);
    int prm_msize = prm->msize;
    grids_info grids = coulomb->grids;
    Tensor3D Jprm = malloc_tensor3d(batch_size, prm_msize, prm_msize);
#pragma omp parallel
    {
        double G[3];
        xint_info xint = init_xint(mol);
        int max_ang = xint->max_ang;
        Vector buf = malloc_vector(xint_gtolen(max_ang) * xint_gtolen(max_ang));
        Tensor3D Jprm_tmp = malloc_tensor3d(batch_size, prm_msize, prm_msize);
#pragma omp for schedule(dynamic, 1) nowait
        for (int n = 0; n < grids->block_num; n++)
        {
            grid_block block = &grids->blocks[n];
            int grid_num = block->grid_num;
            int shell_num = block->shell_num;
            int bas_num = block->bas_num;
            List global2local = block->global2local;
            List local2global = block->local2global;
            if (bas_num == 0)
                continue;
            Tensor3D d_matrix_local = malloc_tensor3d(batch_size, bas_num, bas_num);
            Matrix rho = malloc_matrix(batch_size, grid_num);
            cal_block_bas(n, grids);
            for (int a = 0; a < batch_size; a++)
            {
                if (if_finnish[a] == 1)
                    continue;
                for (int i = 0; i < bas_num; i++)
                    for (int j = 0; j < bas_num; j++)
                    {
                        d_matrix_local[a][i][j] = (d_matrix[a][local2global[i]][local2global[j]] +
                                                   d_matrix[a][local2global[j]][local2global[i]]) *
                                                  0.5;
                    }
                cal_rho(rho[a], d_matrix_local[a], block);
            }
            for (int g = 0; g < grid_num; g++)
            {
                double weight = grids->grids[block->grid_list[g]].w;
                double max_rho = 0.;
                G[0] = grids->grids[block->grid_list[g]].x;
                G[1] = grids->grids[block->grid_list[g]].y;
                G[2] = grids->grids[block->grid_list[g]].z;
                for (int n = 0; n < batch_size; n++)
                    if (fabs(rho[n][g]) > max_rho)
                        max_rho = fabs(rho[n][g]);
                if (max_rho < 1E-12)
                    continue;
                for (int ij = 0; ij < pair->len; ij++)
                {
                    int shli = pair->shp[ij].shli;
                    int shlj = pair->shp[ij].shlj;
                    int pi = prm->shls_p[shli];
                    int pj = prm->shls_p[shlj];
                    int li = pair->shp[ij].li;
                    int lj = pair->shp[ij].lj;
                    int di = xint_gtolen(li);
                    int dj = xint_gtolen(lj);
                    double factor = 1.;
                    if (pi == pj)
                        factor *= 0.5;
                    xint_3c1e_prm(buf, ij, G, xint);
                    for (int n = 0; n < batch_size; n++)
                        for (int u = 0, uu = 0; u < di; u++)
                            for (int v = 0; v < dj; v++, uu++)
                            {
                                Jprm_tmp[n][pi + u][pj + v] += factor * buf[uu] * rho[n][g];
                                Jprm_tmp[n][pj + v][pi + u] += factor * buf[uu] * rho[n][g];
                            }
                }
            }
            free_tensor3d(d_matrix_local);
            free_matrix(rho);
            clear_block_bas(block);
        }
        free(buf);
        del_xint(xint);
        free_tensor3d(Jprm_tmp);
    }
    Matrix J_tmp = malloc_matrix(msize, msize);
    for (int a = 0; a < batch_size; a++)
    {
        if (if_finnish[a])
            continue;
        trans_p2c_mat(J_tmp, Jprm[a], prm);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, msize, msize, msize, 1., coulomb->o_matrix[0],
                    msize, J_tmp[0], msize, 1., coulomb->J[a][0], msize);
        for (int i = 0; i < msize; i++)
            for (int j = 0; j < i; j++)
            {
                coulomb->J[a][i][j] += coulomb->J[a][j][i];
                coulomb->J[a][j][i] = coulomb->J[a][i][j] *= 0.5;
            }
    }
    free_matrix(J_tmp);
    free_tensor3d(Jprm);
    return;
}

void cal_j_matrix(const Matrix d_matrix, coulomb_info coulomb)
{
    bool if_finnish = false;
    return cal_j_matrix_batch(&if_finnish, &d_matrix, coulomb);
}

void cal_j_matrix_batch(const bool *if_finnish, const Matrix *d_matrix, coulomb_info coulomb)
{
    double t1 = omp_get_wtime();
    switch (coulomb->type)
    {
    case None_jbuilder:
        break;
    case RI_jbuilder:
        cal_j_matrix_ri(if_finnish, d_matrix, coulomb);
        break;
    case SemiNumerical_builder:
        cal_j_matrix_sn(if_finnish, d_matrix, coulomb);
        break;
    case THC_jbuilder:
        cal_j_matrix_thc(if_finnish, d_matrix, coulomb);
        break;
    default:
        assert(coulomb->type >= 0 && coulomb->type < 4);
        break;
    }
#ifdef DEBUG
    printf("  Time of coulomb matrices' calculation is %12.6f s.\n", omp_get_wtime() - t1);
#endif
    return;
}

/**
 * @brief calculate coulomb interaction by resolution of identity(RI)
 *
 * @param EJ_act interaction energy between monomers which could be seen as off-diagonal elements
 * @param EJ_self coulomb energy of monomers which could be seen as diagonal elements
 * @param d_matrix_act density matrices have interactions
 * @param d_matrix_self density matrices don't have interactions
 * @param act_num the number of d_matrix_act
 * @param self_num the number of d_matrix_self
 * @param coulomb
 */
void cal_coulomb_interact_ri(Matrix EJ_act, Vector EJ_self, const Matrix *d_matrix_act, const Matrix *d_matrix_self,
                             const int act_num, const int self_num, coulomb_info coulomb)
{
    assert(coulomb->if_aux);
    int mon_num = act_num + self_num;
    const mol_info mol = coulomb->mol;
    const bas_info aux = coulomb->aux;
    const bas_info bas = mol->bas;
    int naux = aux->nbas;
    int aux_msize = aux->msize;
    int msize = bas->msize;
    Matrix rho_aux = malloc_matrix(mon_num, aux_msize);
    Matrix rho_aux_ = malloc_matrix(mon_num, aux_msize);
#pragma omp parallel
    {
        xint_info xint = init_xint(mol);
        load_xint_aux(aux, xint);
        int max_ang = xint->max_ang;
        int max_ang_aux = xint->max_ang_aux;
        Vector buf = malloc_vector(xint_gtolen(max_ang) * xint_gtolen(max_ang) * xint_gtolen(max_ang_aux));
        Matrix rho_aux_tmp = malloc_matrix(mon_num, aux_msize);
        shell_pair shp;
        double factor = 2.;
#pragma omp for schedule(dynamic, 4) nowait
        for (int ij = 0; ij < mol->pair->len; ij++)
        {
            shp = &mol->pair->shp[ij];
            int shli = shp->shli, shlj = shp->shlj;
            int li = shp->li, lj = shp->lj;
            int di = xint_gtolen(li), dj = xint_gtolen(lj);
            int pi = bas->shls_p[shli], pj = bas->shls_p[shlj];
            if (shli == shlj)
                factor = 1.;
            else
                factor = 2.;
            for (int a = 0; a < naux; a++)
            {
                xint_3c2e(buf, ij, a, xint);
                int pa = aux->shls_p[a];
                int da = xint_gtolen(aux(ANGULAR_VAL, a));
                for (int a = 0; a < act_num; a++)
                    for (int l = 0, uu = 0; l < da; l++)
                        for (int m = 0; m < di; m++)
                            for (int n = 0; n < dj; n++, uu++)
                            {
                                rho_aux_tmp[a][l + pa] += factor * d_matrix_act[a][pi + m][pj + n] * buf[uu];
                            }
                for (int a = 0, b = act_num; a < self_num; a++, b++)
                    for (int l = 0, uu = 0; l < da; l++)
                        for (int m = 0; m < di; m++)
                            for (int n = 0; n < dj; n++, uu++)
                            {
                                rho_aux_tmp[b][l + pa] += factor * d_matrix_self[a][pi + m][pj + n] * buf[uu];
                            }
            }
        }
#pragma omp critical
        {
            for (int i = 0; i < aux_msize * mon_num; i++)
                rho_aux[0][i] += rho_aux_tmp[0][i];
        }
        free_matrix(rho_aux_tmp);
        free_vector(buf);
        del_xint(xint);
    }
    List ipiv = malloc_list(aux_msize);
    int lwork = 1;
    Vector aux2e_matrix = malloc_vector(aux_msize * aux_msize);
#pragma omp parallel
    {
#pragma omp for schedule(dynamic, 1) nowait
        for (int a = 0; a < mon_num; a++)
        {
            memcpy(rho_aux_[a], rho_aux[a], sizeof(double) * aux_msize);
            memcpy(aux2e_matrix, coulomb->aux2e_matrix[0], sizeof(double) * aux_msize * aux_msize);
            LAPACKE_dgesv(LAPACK_ROW_MAJOR, aux_msize, lwork, aux2e_matrix, aux_msize, ipiv, rho_aux_[a], 1);
        }
    }
    free(ipiv);
    free(aux2e_matrix);
    for (int i = 0; i < act_num; i++)
        for (int j = 0; j <= i; j++)
        {
            EJ_act[i][j] = vv_dot(rho_aux_[i], rho_aux[j], aux_msize);
            EJ_act[j][i] = EJ_act[i][j];
        }
    for (int i = 0; i < self_num; i++)
        EJ_self[i] = vv_dot(rho_aux_[i + act_num], rho_aux[i + act_num], aux_msize);
    free_matrix(rho_aux);
    free_matrix(rho_aux_);
    return;
}

/**
 * @brief calculate coulomb interaction by only 2c1e integrals
 *
 * @param EJ_act interaction energy between monomers which could be seen as off-diagonal elements
 * @param EJ_self coulomb energy of monomers which could be seen as diagonal elements
 * @param d_matrix_act density matrices have interactions
 * @param d_matrix_self density matrices don't have interactions
 * @param act_num the number of d_matrix_act
 * @param self_num the number of d_matrix_self
 * @param coulomb
 */
void cal_coulomb_interact_thc(Matrix EJ_act, Vector EJ_self, const Matrix *d_matrix_act, const Matrix *d_matrix_self,
                              const int act_num, const int self_num, coulomb_info coulomb)
{
    assert(coulomb->if_aux && coulomb->if_grids);
    const mol_info mol = coulomb->mol;
    const bas_info aux = coulomb->aux;
    int mon_num = act_num + self_num;
    int aux_nbas = aux->nbas;
    int aux_msize = aux->msize;
    int msize = get_mol_msize(mol);
    grids_info grids = coulomb->grids;
    Matrix rho_aux = malloc_matrix(mon_num, aux_msize);
    Matrix rho_aux_ = malloc_matrix(mon_num, aux_msize);
/* calculate electronic potential \Gamma_g=\sum_i{C_i(\rho|i)} */
#pragma omp parallel
    {
        double buf[200], G[3];
        xint_info xint = init_xint(mol);
        load_xint_aux(aux, xint);
        Matrix rho_aux_tmp = malloc_matrix(mon_num, aux_msize);
#pragma omp for schedule(dynamic, 1) nowait
        for (int n = 0; n < grids->block_num; n++)
        {
            grid_block block = &grids->blocks[n];
            int grid_num = block->grid_num;
            int shell_num = block->shell_num;
            int bas_num = block->bas_num;
            List global2local = block->global2local;
            List local2global = block->local2global;
            if (bas_num == 0)
                continue;
            Tensor3D d_matrix_local = malloc_tensor3d(mon_num, bas_num, bas_num);
            Matrix rho = malloc_matrix(mon_num, grid_num);
            cal_block_bas(n, grids);
            for (int a = 0; a < act_num; a++)
                for (int i = 0; i < bas_num; i++)
                    for (int j = 0; j < bas_num; j++)
                    {
                        d_matrix_local[a][i][j] = d_matrix_act[a][local2global[i]][local2global[j]];
                    }
            for (int a = act_num, b = 0; a < mon_num; a++, b++)
                for (int i = 0; i < bas_num; i++)
                    for (int j = 0; j < bas_num; j++)
                    {
                        d_matrix_local[a][i][j] = d_matrix_self[b][local2global[i]][local2global[j]];
                    }
            for (int a = 0; a < mon_num; a++)
                cal_rho(rho[a], d_matrix_local[a], block);
            for (int g = 0; g < grid_num; g++)
            {
                double weight = grids->grids[block->grid_list[g]].w;
                G[0] = grids->grids[block->grid_list[g]].x;
                G[1] = grids->grids[block->grid_list[g]].y;
                G[2] = grids->grids[block->grid_list[g]].z;
                for (int i = 0; i < aux_nbas; i++)
                {
                    xint_2c1e(buf, i, G, xint);
                    int di = xint_gtolen(aux(ANGULAR_VAL, i));
                    int pi = aux->shls_p[i];
                    for (int u = 0; u < di; u++)
                    {
                        double tmp_buf = buf[u] * weight;
                        for (int a = 0; a < mon_num; a++)
                            rho_aux_tmp[a][pi + u] += rho[a][g] * tmp_buf;
                    }
                }
            }
            free_tensor3d(d_matrix_local);
            free_matrix(rho);
            clear_block_bas(block);
        }
#pragma omp critical
        for (int i = 0; i < mon_num * aux_msize; i++)
        {
            rho_aux[0][i] += rho_aux_tmp[0][i];
        }
        del_xint(xint);
        free_matrix(rho_aux_tmp);
    }
    List ipiv = malloc_list(aux_msize);
    int lwork = 1;
    Vector aux2e_matrix = malloc_vector(aux_msize * aux_msize);
#pragma omp parallel
    {
#pragma omp for schedule(dynamic, 1) nowait
        for (int a = 0; a < mon_num; a++)
        {
            memcpy(rho_aux_[a], rho_aux[a], sizeof(double) * aux_msize);
            memcpy(aux2e_matrix, coulomb->aux2e_matrix[0], sizeof(double) * aux_msize * aux_msize);
            LAPACKE_dgesv(LAPACK_ROW_MAJOR, aux_msize, lwork, aux2e_matrix, aux_msize, ipiv, rho_aux_[a], 1);
        }
    }
    free(ipiv);
    free(aux2e_matrix);
    for (int i = 0; i < act_num; i++)
        for (int j = 0; j <= i; j++)
        {
            EJ_act[i][j] = vv_dot(rho_aux_[i], rho_aux[j], aux_msize) / MY_PI;
            EJ_act[j][i] = EJ_act[i][j];
        }
    for (int i = 0; i < self_num; i++)
        EJ_self[i] = vv_dot(rho_aux_[i + act_num], rho_aux[i + act_num], aux_msize) / MY_PI;
    free_matrix(rho_aux);
    free_matrix(rho_aux_);
    return;
}
/**
 * @brief calculate coulomb interaction by RI_builder or THC_builder method
 *
 * @param EJ_act interaction energy between monomers which could be seen as off-diagonal elements
 * @param EJ_self coulomb energy of monomers which could be seen as diagonal elements
 * @param d_matrix_act density matrices have interactions
 * @param d_matrix_self density matrices don't have interactions
 * @param act_num the number of d_matrix_act
 * @param self_num the number of d_matrix_self
 * @param coulomb
 */
void cal_coulomb_interact(Matrix EJ_act, Vector EJ_self, const Matrix *d_matrix_act, const Matrix *d_matrix_self,
                          const int act_num, const int self_num, coulomb_info coulomb)
{
    assert(coulomb->if_aux);
    switch (coulomb->type)
    {
    case SemiNumerical_jbuilder:
    case THC_jbuilder:
        return cal_coulomb_interact_thc(EJ_act, EJ_self, d_matrix_act, d_matrix_self, act_num, self_num, coulomb);
        break;
    case None_jbuilder:
    case RI_jbuilder:
        return cal_coulomb_interact_ri(EJ_act, EJ_self, d_matrix_act, d_matrix_self, act_num, self_num, coulomb);
    default:
        assert(coulomb->type >= 0 && coulomb->type < 4);
        break;
    }
    return;
}
/**
 * @brief calculate electrostatic interaction on each grids by RI density fitting and THC
 *
 * @param EJ_act size of act_num * act_num * grid_num
 * @param EJ_self size of self_num * grid_num
 * @param d_matrix_act density matrices have interactions
 * @param d_matrix_self density matrices don't have interactions
 * @param act_num the number of d_matrix_act
 * @param self_num the number of d_matrix_self
 * @param grid_in the grids used to do calculation
 * @param coulomb
 */
void cal_electrostatic_interact_grid(Matrix *EJ_act, Vector *EJ_self, const Matrix *d_matrix_act, const Matrix *d_matrix_self,
                                     const int act_num, const int self_num, const List *atm_id, const grids_info grid_in, coulomb_info coulomb)
{
    grids_info grids;
    if (grid_in == NULL)
        grids = coulomb->grids;
    else
        grids = grid_in;
    int mon_num = act_num + self_num;
    const mol_info mol = coulomb->mol;
    const bas_info bas = mol->bas;
    const bas_info aux = coulomb->aux;
    const atm_info atm = mol->atm;
    int naux = aux->nbas;
    int aux_msize = aux->msize;
    int msize = get_mol_msize(mol);
    int natm = atm->natm;
    Matrix rho_aux = malloc_matrix(mon_num, aux_msize);
// calculate density fitting coefficients
#pragma omp parallel
    {
        xint_info xint = init_xint(mol);
        load_xint_aux(aux, xint);
        int max_ang = xint->max_ang, max_ang_aux = xint->max_ang_aux;
        Vector buf = malloc_vector(xint_gtolen(max_ang) * xint_gtolen(max_ang) * xint_gtolen(max_ang_aux));
        Matrix rho_aux_tmp = malloc_matrix(mon_num, aux_msize);
        shell_pair shp;
        double factor = 2.;
#pragma omp for schedule(dynamic, 4) nowait
        for (int ij = 0; ij < mol->pair->len; ij++)
        {
            shp = &mol->pair->shp[ij];
            int shli = shp->shli, shlj = shp->shlj;
            int li = shp->li, lj = shp->lj;
            int di = xint_gtolen(li), dj = xint_gtolen(lj);
            int pi = bas->shls_p[shli], pj = bas->shls_p[shlj];
            if (shli == shlj)
                factor = 1.;
            else
                factor = 2.;
            for (int a = 0; a < naux; a++)
            {
                xint_3c2e(buf, ij, a, xint);
                int pa = aux->shls_p[a];
                int da = xint_gtolen(aux(ANGULAR_VAL, a));
                for (int a = 0; a < act_num; a++)
                    for (int l = 0, uu = 0; l < da; l++)
                        for (int m = 0; m < di; m++)
                            for (int n = 0; n < dj; n++, uu++)
                            {
                                rho_aux_tmp[a][l + pa] += factor * d_matrix_act[a][pi + m][pj + n] * buf[uu];
                            }
                for (int a = 0, b = act_num; a < self_num; a++, b++)
                    for (int l = 0, uu = 0; l < da; l++)
                        for (int m = 0; m < di; m++)
                            for (int n = 0; n < dj; n++, uu++)
                            {
                                rho_aux_tmp[b][l + pa] += factor * d_matrix_self[a][pi + m][pj + n] * buf[uu];
                            }
            }
        }
#pragma omp critical
        {
            for (int i = 0; i < aux_msize * mon_num; i++)
                rho_aux[0][i] += rho_aux_tmp[0][i];
        }
        free_matrix(rho_aux_tmp);
        free_vector(buf);
        del_xint(xint);
    }
    List ipiv = malloc_list(aux_msize);
    int lwork = 1;
    Vector aux2e_matrix = malloc_vector(aux_msize * aux_msize);
#pragma omp parallel
    {
#pragma omp for schedule(dynamic, 1) nowait
        for (int a = 0; a < mon_num; a++)
        {
            memcpy(aux2e_matrix, coulomb->aux2e_matrix[0], sizeof(double) * aux_msize * aux_msize);
            LAPACKE_dgesv(LAPACK_ROW_MAJOR, aux_msize, lwork, aux2e_matrix, aux_msize, ipiv, rho_aux[a], 1);
        }
    }
    free(ipiv);
    free(aux2e_matrix);
#pragma omp parallel
    {
        double buf[200], G[3], *A;
        xint_info xint = init_xint(mol);
        load_xint_aux(aux, xint);
#pragma omp for schedule(dynamic, 1) nowait
        for (int n = 0; n < grids->block_num; n++)
        {
            grid_block block = &grids->blocks[n];
            int grid_num = block->grid_num;
            int shell_num = block->shell_num;
            int bas_num = block->bas_num;
            List global2local = block->global2local;
            List local2global = block->local2global;
            if (bas_num == 0)
                continue;
            Tensor3D d_matrix_local = malloc_tensor3d(mon_num, bas_num, bas_num);
            Matrix rho = malloc_matrix(mon_num, grid_num);
            Matrix potential = malloc_matrix(mon_num, grid_num);
            cal_block_bas(n, grids);
            for (int a = 0; a < act_num; a++)
                for (int i = 0; i < bas_num; i++)
                    for (int j = 0; j < bas_num; j++)
                    {
                        d_matrix_local[a][i][j] = d_matrix_act[a][local2global[i]][local2global[j]];
                    }
            for (int a = act_num, b = 0; a < mon_num; a++, b++)
                for (int i = 0; i < bas_num; i++)
                    for (int j = 0; j < bas_num; j++)
                    {
                        d_matrix_local[a][i][j] = d_matrix_self[b][local2global[i]][local2global[j]];
                    }
            for (int a = 0; a < mon_num; a++)
                cal_rho(rho[a], d_matrix_local[a], block);
            // calculate potential
            for (int g = 0; g < grid_num; g++)
            {
                double weight = grids->grids[block->grid_list[g]].w;
                G[0] = grids->grids[block->grid_list[g]].x;
                G[1] = grids->grids[block->grid_list[g]].y;
                G[2] = grids->grids[block->grid_list[g]].z;
                for (int i = 0, u = 0; i < naux; i++)
                {
                    xint_2c1e(buf, i, G, xint);
                    int di = xint_gtolen(aux(ANGULAR_VAL, i));
                    for (int j = 0; j < di; j++, u++)
                    {
                        double tmp = weight * buf[j] / sqrt(MY_PI);
                        for (int a = 0; a < mon_num; a++)
                        {
                            potential[a][g] += 2 * tmp * rho_aux[a][u];
                        }
                    }
                }
                for (int i = 0; i < natm; i++)
                {
                    A = get_atm_coord(i, atm);
                    double dist = (A[0] - G[0]) * (A[0] - G[0]) +
                                  (A[1] - G[1]) * (A[1] - G[1]) +
                                  (A[2] - G[2]) * (A[2] - G[2]);
                    dist = 4. / sqrt(dist) * weight;
                    for (int a = 0; a < mon_num; a++)
                    {
                        potential[a][g] -= atm_id[a][i] * dist;
                    }
                }
            }
            // calculate energy
            for (int g = 0; g < grid_num; g++)
            {
                int g_id = block->grid_list[g];
                for (int a = 0; a < act_num; a++)
                    for (int b = 0; b < act_num; b++)
                    {
                        EJ_act[a][b][g_id] = rho[a][g] * potential[b][g];
                    }
                for (int a = act_num; a < mon_num; a++)
                {
                    EJ_self[a - act_num][g_id] = rho[a][g] * potential[a][g];
                }
            }
            free_matrix(rho);
            free_matrix(potential);
            free_tensor3d(d_matrix_local);
            clear_block_bas(block);
        }
    }
    free_matrix(rho_aux);
    return;
}

/**
 * @brief calculate energy gradient of coulomb part as
 *        E^A = 2 * sum_{mu,nu}sum_MP_{mu,nu}(mu^A,nu|M)C_M
 * @param e_grad vector to store gradients
 * @param d_matrix total density matrix
 * @param coulomb
 */
void cal_coulomb_grad_ri(Vector *e_grad, const Matrix *d_matrix, coulomb_info coulomb)
{
    static double scale = 1;
    assert(coulomb->if_aux);
    const mol_info mol = coulomb->mol;
    const bas_info aux = coulomb->aux;
    const bas_info bas = mol->bas;
    int batch_size = coulomb->batch_size;
    int naux = aux->nbas;
    int aux_msize = aux->msize;
    int msize = get_mol_msize(mol);
    Matrix rho_aux = malloc_matrix(batch_size, aux_msize);
    // do calculation to get C_M, this part calculation is just as the same as RI for J
#pragma omp parallel
    {
        xint_info xint = init_xint(mol);
        load_xint_aux(aux, xint);
        int max_ang = xint->max_ang;
        int max_ang_aux = xint->max_ang_aux;
        Vector buf = malloc_vector(xint_gtolen(max_ang) * xint_gtolen(max_ang) * xint_gtolen(max_ang_aux));
        Matrix rho_aux_tmp = malloc_matrix(batch_size, aux_msize);
        shell_pair shp;
        double factor = 2.;
#pragma omp for schedule(dynamic, 4) nowait
        for (int ij = 0; ij < mol->pair->len; ij++)
        {
            shp = &mol->pair->shp[ij];
            int shli = shp->shli, shlj = shp->shlj;
            int li = shp->li, lj = shp->lj;
            int di = xint_gtolen(li), dj = xint_gtolen(lj);
            int pi = bas->shls_p[shli], pj = bas->shls_p[shlj];
            if (shli == shlj)
                for (int a = 0; a < naux; a++)
                {
                    xint_3c2e(buf, ij, a, xint);
                    int pa = aux->shls_p[a];
                    int da = xint_gtolen(aux(ANGULAR_VAL, a));
                    for (int a = 0; a < batch_size; a++)
                    {
                        for (int l = 0, uu = 0; l < da; l++)
                            for (int m = 0; m < di; m++)
                                for (int n = 0; n < dj; n++, uu++)
                                {
                                    rho_aux_tmp[a][l + pa] += d_matrix[a][pi + m][pj + n] * buf[uu];
                                }
                    }
                }
            else
                for (int a = 0; a < naux; a++)
                {
                    xint_3c2e(buf, ij, a, xint);
                    int pa = aux->shls_p[a];
                    int da = xint_gtolen(aux(ANGULAR_VAL, a));
                    for (int a = 0; a < batch_size; a++)
                    {
                        for (int l = 0, uu = 0; l < da; l++)
                            for (int m = 0; m < di; m++)
                                for (int n = 0; n < dj; n++, uu++)
                                {
                                    rho_aux_tmp[a][l + pa] += (d_matrix[a][pi + m][pj + n] + d_matrix[a][pj + n][pi + m]) * buf[uu];
                                }
                    }
                }
        }
#pragma omp critical
        {
            for (int a = 0; a < batch_size; a++)
                for (int i = 0; i < aux_msize; i++)
                    rho_aux[a][i] += rho_aux_tmp[a][i];
        }
        free_matrix(rho_aux_tmp);
        free_vector(buf);
        del_xint(xint);
    }
    List ipiv = malloc_list(aux_msize);
    int lwork = 1;
    Vector aux2e_matrix = malloc_vector(aux_msize * aux_msize);
    for (int a = 0; a < batch_size; a++)
    {
        memcpy(aux2e_matrix, coulomb->aux2e_matrix[0], sizeof(double) * aux_msize * aux_msize);
        LAPACKE_dgesv(LAPACK_ROW_MAJOR, aux_msize, lwork, aux2e_matrix, aux_msize, ipiv, rho_aux[a], 1);
        memcpy(coulomb->df_coe[a], rho_aux[a], sizeof(double) * aux_msize);
    }
    free(ipiv);
    free(aux2e_matrix);
    // run over all integrals to get the gradients
    // memset(e_grad[0], 0, sizeof(double) * batch_size * 3 * get_mol_natm(mol));
    const pair_info pair = mol->pair;
#pragma omp parallel
    {
        xint_info xint = init_xint(mol);
        load_xint_aux(aux, xint);
        int max_ang = xint->max_ang;
        int max_ang_aux = xint->max_ang_aux;
        Vector buf = malloc_vector(6 * xint_gtolen(max_ang) * xint_gtolen(max_ang) * xint_gtolen(max_ang_aux));
        Matrix e_grad_tmp = malloc_matrix(batch_size, 3 * get_mol_natm(mol));
#pragma omp for schedule(dynamic, 1) nowait
        for (int ij = 0; ij < pair->len; ij++)
        {
            int shli = pair->shp[ij].shli;
            int shlj = pair->shp[ij].shlj;
            int pi = bas->shls_p[shli];
            int pj = bas->shls_p[shlj];
            int li = pair->shp[ij].li;
            int lj = pair->shp[ij].lj;
            int di = xint_gtolen(li);
            int dj = xint_gtolen(lj);
            int atm_i = bas(ATOM_IND, shli);
            int atm_j = bas(ATOM_IND, shlj);
            double factor = 8;
            if (shli == shlj)
                factor *= 0.5;
            for (int a = 0; a < naux; a++)
            {
                xint_3c2e_grad(buf, ij, a, xint);
                int atm_a = aux(ATOM_IND, a);
                int pa = aux->shls_p[a];
                int da = xint_gtolen(aux(ANGULAR_VAL, a));
                for (int t = 0; t < batch_size; t++)
                {
                    for (int l = 0, uu = 0; l < da; l++)
                        for (int m = 0; m < di; m++)
                            for (int n = 0; n < dj; n++, uu++)
                            {
                                e_grad_tmp[t][3 * atm_i + 0] += factor * d_matrix[t][pi + m][pj + n] *
                                                                rho_aux[t][l + pa] * buf[6 * uu + 0];
                                e_grad_tmp[t][3 * atm_i + 1] += factor * d_matrix[t][pi + m][pj + n] *
                                                                rho_aux[t][l + pa] * buf[6 * uu + 1];
                                e_grad_tmp[t][3 * atm_i + 2] += factor * d_matrix[t][pi + m][pj + n] *
                                                                rho_aux[t][l + pa] * buf[6 * uu + 2];
                                e_grad_tmp[t][3 * atm_j + 0] += factor * d_matrix[t][pi + m][pj + n] *
                                                                rho_aux[t][l + pa] * buf[6 * uu + 3];
                                e_grad_tmp[t][3 * atm_j + 1] += factor * d_matrix[t][pi + m][pj + n] *
                                                                rho_aux[t][l + pa] * buf[6 * uu + 4];
                                e_grad_tmp[t][3 * atm_j + 2] += factor * d_matrix[t][pi + m][pj + n] *
                                                                rho_aux[t][l + pa] * buf[6 * uu + 5];
                                e_grad_tmp[t][3 * atm_a + 0] -= factor * d_matrix[t][pi + m][pj + n] *
                                                                rho_aux[t][l + pa] * (buf[6 * uu + 0] + buf[6 * uu + 3]);
                                e_grad_tmp[t][3 * atm_a + 1] -= factor * d_matrix[t][pi + m][pj + n] *
                                                                rho_aux[t][l + pa] * (buf[6 * uu + 1] + buf[6 * uu + 4]);
                                e_grad_tmp[t][3 * atm_a + 2] -= factor * d_matrix[t][pi + m][pj + n] *
                                                                rho_aux[t][l + pa] * (buf[6 * uu + 2] + buf[6 * uu + 5]);
                            }
                }
            }
        }
#pragma omp critical
        {
            for (int a = 0; a < batch_size * 3 * get_mol_natm(mol); a++)
                e_grad[0][a] += e_grad_tmp[0][a];
        }
        del_xint(xint);
        free_matrix(e_grad_tmp);
        free(buf);
    }
    double buf[1024];
    xint_info xint = init_xint(mol);
    load_xint_aux(aux, xint);
    for (int i = 0; i < naux; i++)
    {
        int atm_a = aux(ATOM_IND, i);
        int la = aux(ANGULAR_VAL, i);
        int pa = aux->shls_p[i];
        int da = xint_gtolen(la);
        for (int j = 0; j < naux; j++)
        {
            int atm_b = aux(ATOM_IND, j);
            int lb = aux(ANGULAR_VAL, j);
            int pb = aux->shls_p[j];
            int db = xint_gtolen(lb);
            xint_2c2e_grad(buf, i, j, xint);
            for (int a = 0; a < batch_size; a++)
                for (int u = 0, uu = 0; u < da; u++)
                    for (int v = 0; v < db; v++, uu++)
                    {
                        e_grad[a][atm_a * 3 + 0] += 2 * buf[3 * uu + 0] * rho_aux[a][pa + u] * rho_aux[a][pb + v];
                        e_grad[a][atm_a * 3 + 1] += 2 * buf[3 * uu + 1] * rho_aux[a][pa + u] * rho_aux[a][pb + v];
                        e_grad[a][atm_a * 3 + 2] += 2 * buf[3 * uu + 2] * rho_aux[a][pa + u] * rho_aux[a][pb + v];
                        e_grad[a][atm_b * 3 + 0] -= 2 * buf[3 * uu + 0] * rho_aux[a][pa + u] * rho_aux[a][pb + v];
                        e_grad[a][atm_b * 3 + 1] -= 2 * buf[3 * uu + 1] * rho_aux[a][pa + u] * rho_aux[a][pb + v];
                        e_grad[a][atm_b * 3 + 2] -= 2 * buf[3 * uu + 2] * rho_aux[a][pa + u] * rho_aux[a][pb + v];
                    }
        }
    }
    free_matrix(rho_aux);
    del_xint(xint);
    return;
}

/**
 * @brief calculate energy gradient of coulomb part by a THC way
 *        E^A = 2 * sum_{mu,nu}sum_MP_{mu,nu}(mu^A,nu|M)C_M
 *            = 2 * sum_{mu,nu, rg} chi_mu^A(rg) * chi_nu(rg) * V(rg)
 * @param e_grad vector to store gradients
 * @param d_matrix total density matrix
 * @param coulomb
 */
void cal_coulomb_grad_thc(Vector *e_grad, const Matrix *d_matrix, coulomb_info coulomb)
{
    assert(coulomb->if_aux && coulomb->if_grids);
    const mol_info mol = coulomb->mol;
    const bas_info aux = coulomb->aux;
    const bas_info bas = mol->bas;
    int batch_size = coulomb->batch_size;
    int naux = aux->nbas;
    int aux_msize = aux->msize;
    int msize = get_mol_msize(mol);
    grids_info grids = coulomb->grids;
    Matrix rho_aux = malloc_matrix(batch_size, aux_msize);
/* calculate electronic potential \Gamma_g=\sum_i{C_i(\rho|i)} */
#pragma omp parallel
    {
        double buf[200], G[3];
        xint_info xint = init_xint(mol);
        load_xint_aux(aux, xint);
        Matrix rho_aux_tmp = malloc_matrix(batch_size, aux_msize);
#pragma omp for schedule(dynamic, 1) nowait
        for (int n = 0; n < grids->block_num; n++)
        {
            grid_block block = &grids->blocks[n];
            int grid_num = block->grid_num;
            int shell_num = block->shell_num;
            int bas_num = block->bas_num;
            List global2local = block->global2local;
            List local2global = block->local2global;
            if (bas_num == 0)
                continue;
            Tensor3D d_matrix_local = malloc_tensor3d(batch_size, bas_num, bas_num);
            Matrix rho = malloc_matrix(batch_size, grid_num);
            cal_block_bas(n, grids);
            for (int a = 0; a < batch_size; a++)
            {
                for (int i = 0; i < bas_num; i++)
                    for (int j = 0; j < bas_num; j++)
                    {
                        d_matrix_local[a][i][j] = (d_matrix[a][local2global[i]][local2global[j]] +
                                                   d_matrix[a][local2global[j]][local2global[i]]) *
                                                  0.5;
                    }
                cal_rho(rho[a], d_matrix_local[a], block);
            }
            for (int g = 0; g < grid_num; g++)
            {
                double weight = grids->grids[block->grid_list[g]].w;
                G[0] = grids->grids[block->grid_list[g]].x;
                G[1] = grids->grids[block->grid_list[g]].y;
                G[2] = grids->grids[block->grid_list[g]].z;
                for (int i = 0; i < naux; i++)
                {
                    // if (fabs(rho[g]*coulomb->aux2e_rev[i])<1E-12) continue;
                    xint_2c1e(buf, i, G, xint);
                    int di = xint_gtolen(aux(ANGULAR_VAL, i));
                    int pi = aux->shls_p[i];
                    for (int u = 0; u < di; u++)
                    {
                        double tmp_buf = buf[u] * weight;
                        for (int a = 0; a < batch_size; a++)
                            rho_aux_tmp[a][pi + u] += rho[a][g] * tmp_buf;
                    }
                }
            }
            free_tensor3d(d_matrix_local);
            free_matrix(rho);
            clear_block_bas(block);
        }
#pragma omp critical
        for (int i = 0; i < batch_size * aux_msize; i++)
        {
            rho_aux[0][i] += rho_aux_tmp[0][i];
        }
        del_xint(xint);
        free_matrix(rho_aux_tmp);
    }
    List ipiv = malloc_list(aux_msize);
    int lwork = 1;
    Vector aux2e_matrix = malloc_vector(aux_msize * aux_msize);
    for (int a = 0; a < batch_size; a++)
    {
        memcpy(aux2e_matrix, coulomb->aux2e_matrix[0], sizeof(double) * aux_msize * aux_msize);
        LAPACKE_dgesv(LAPACK_ROW_MAJOR, aux_msize, lwork, aux2e_matrix, aux_msize, ipiv, rho_aux[a], 1);
    }
    free(ipiv);
    free(aux2e_matrix);
    /* Get J matrix by using electronic potential J_{ij}=\sum_g{\Gamma_gX_igX_jg} */
    memset(e_grad[0], 0, sizeof(double) * 3 * get_mol_natm(mol) * batch_size);
#pragma omp parallel
    {
        double buf[200], G[3];
        xint_info xint = init_xint(mol);
        load_xint_aux(aux, xint);
        Matrix rho_aux_tmp = malloc_matrix(batch_size, aux_msize);
        Matrix e_grad_tmp = malloc_matrix(batch_size, 3 * get_mol_natm(mol));
#pragma omp for schedule(dynamic, 1) nowait
        for (int n = 0; n < grids->block_num; n++)
        {
            grid_block block = &grids->blocks[n];
            int grid_num = block->grid_num;
            int shell_num = block->shell_num;
            int bas_num = block->bas_num;
            if (bas_num == 0)
                continue;
            List local2global = block->local2global;
            Matrix hartree_potential = malloc_matrix(batch_size, grid_num);
            cal_block_bas(n, grids);
            for (int i = 0; i < naux; i++)
            {
                // if (fabs(rho_aux[i])<1E-12) continue;
                for (int g = 0; g < grid_num; g++)
                {
                    G[0] = grids->grids[block->grid_list[g]].x;
                    G[1] = grids->grids[block->grid_list[g]].y;
                    G[2] = grids->grids[block->grid_list[g]].z;
                    xint_2c1e(buf, i, G, xint);
                    int di = xint_gtolen(aux(ANGULAR_VAL, i));
                    int pi = aux->shls_p[i];
                    for (int a = 0; a < batch_size; a++)
                        for (int u = 0; u < di; u++)
                            hartree_potential[a][g] += rho_aux[a][u + pi] * buf[u];
                }
            }
            Matrix psi_tmp = malloc_matrix(grid_num, bas_num);
            Matrix psixyz_tmp = malloc_matrix(grid_num * 3, bas_num);
            for (int a = 0; a < batch_size; a++)
            {
                memcpy(psi_tmp[0], block->psi[0], sizeof(double) * grid_num * bas_num);
                memcpy(psixyz_tmp[0], block->psix[0], sizeof(double) * grid_num * bas_num);
                memcpy(psixyz_tmp[grid_num], block->psiy[0], sizeof(double) * grid_num * bas_num);
                memcpy(psixyz_tmp[grid_num * 2], block->psiz[0], sizeof(double) * grid_num * bas_num);
                for (int g = 0; g < block->grid_num; g++)
                {
                    double weight = sqrt(grids->grids[block->grid_list[g]].w);
                    for (int i = 0; i < block->bas_num; i++)
                    {
                        psixyz_tmp[g][i] *= weight;
                        psixyz_tmp[grid_num + g][i] *= weight;
                        psixyz_tmp[grid_num * 2 + g][i] *= weight;
                        psi_tmp[g][i] *= weight * hartree_potential[a][g];
                    }
                }
                for (int i = 0, uu = 0; i < block->shell_num; i++)
                {
                    int shl = block->shell_list[i];
                    int di = xint_gtolen(bas(ANGULAR_VAL, shl));
                    int atmi = bas(ATOM_IND, shl);
                    for (int u = 0; u < di; u++, uu++)
                        for (int g = 0; g < block->grid_num; g++)
                            for (int j = 0; j < block->bas_num; j++)
                            {
                                e_grad_tmp[a][atmi * 3 + 0] -= psixyz_tmp[g][uu] * psi_tmp[g][j];
                                e_grad_tmp[a][atmi * 3 + 1] -= psixyz_tmp[grid_num + g][uu] * psi_tmp[g][j];
                                e_grad_tmp[a][atmi * 3 + 2] -= psixyz_tmp[grid_num * 2 + g][uu] * psi_tmp[g][j];
                            }
                }
            }
            free_matrix(psi_tmp);
            free_matrix(psixyz_tmp);
            free_matrix(hartree_potential);
            clear_block_bas(block);
        }
#pragma omp critical
        {
            for (int i = 0; i < 3 * get_mol_natm(mol) * batch_size; i++)
                e_grad[0][i] += e_grad_tmp[0][i] / MY_PI;
        }
        del_xint(xint);
        free_matrix(rho_aux_tmp);
        free_matrix(e_grad_tmp);
    }
    free_matrix(rho_aux);
    return;
}
void cal_coulomb_grad(Vector e_grad, const Matrix d_matrix, coulomb_info coulomb)
{
    return cal_coulomb_grad_batch(&e_grad, &d_matrix, coulomb);
}
void cal_coulomb_grad_batch(Vector *e_grad, const Matrix *d_matrix, coulomb_info coulomb)
{
    switch (coulomb->type)
    {
    case RI_jbuilder:
        cal_coulomb_grad_ri(e_grad, d_matrix, coulomb);
        break;
    case THC_jbuilder:
        cal_coulomb_grad_thc(e_grad, d_matrix, coulomb);
        break;
    default:
        break;
    }
    return;
}
