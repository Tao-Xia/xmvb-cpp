#include "fock/fock.h"
exchange_info init_exchange(int batch_size, const mol_info mol)
{
    int msize = get_mol_msize(mol);
    exchange_info exchange = (exchange_info)malloc(sizeof(struct ExchangeInfo));
    exchange->if_aux = false;
    exchange->if_final = false;
    exchange->if_grids = false;
    exchange->type = None_kbuilder;
    exchange->mol = mol;
    if (batch_size > 0)
    {
        exchange->K = malloc_tensor3d(batch_size, msize, msize);
        exchange->K_b = malloc_tensor3d(batch_size, msize, msize);
    }
    else
    {
        exchange->K = NULL;
        exchange->K_b = NULL;
    }
    exchange->lr_frac = 1.;
    exchange->sr_frac = 0.;
    exchange->omega_ = 0.;
    exchange->batch_size = batch_size;
    return exchange;
}
void set_exchange_type(Kbuilder_type type, exchange_info exchange)
{
    exchange->type = type;
    return;
}
void load_exchange_aux(exchange_info exchange)
{
    exchange->if_aux = true;
    const mol_info mol = exchange->mol;
    const atm_info atm = mol->atm;
    exchange->aux = init_aux(2, false, mol->bas);
    int msize = exchange->aux->msize;
    exchange->aux2e_matrix_rev = malloc_matrix(msize, msize);
    cal_2c2e_matrix(exchange->aux2e_matrix_rev, exchange->aux, mol);
    cal_symatrix_reverse(exchange->aux2e_matrix_rev, msize);
    // aux_final as GEN-A2* is used for energy calculation as final integral
    exchange->aux_final = init_aux(2, true, mol->bas);
    msize = exchange->aux_final->msize;
    exchange->aux2e_matrix_rev_final = malloc_matrix(msize, msize);
    cal_2c2e_matrix(exchange->aux2e_matrix_rev_final, exchange->aux_final, mol);
    cal_symatrix_reverse(exchange->aux2e_matrix_rev_final, msize);
    return;
}
void set_exchange_rs(double lr_frac, double sr_frac, double omega_, exchange_info exchange)
{
    exchange->lr_frac = lr_frac;
    exchange->sr_frac = sr_frac;
    exchange->omega_ = omega_;
    return;
}
void set_exchange_final(bool if_final, exchange_info exchange)
{
    assert(exchange->grids_final != NULL);
    exchange->if_final = if_final;
    return;
}
void load_exchange_grids(const char *grids_fname, const char *fgrids_fname, exchange_info exchange)
{
    exchange->if_grids = true;
    int msize = get_mol_msize(exchange->mol);
    exchange->grids = init_grid_opt(grids_fname, exchange->mol);
    Matrix s_matrix = malloc_matrix(msize, msize);
    cal_overlap_matrix(s_matrix, exchange->mol);
    Matrix s_matrix_num = cal_overlap_numerical(exchange->grids);
    exchange->o_matrix = malloc_matrix(msize, msize);
    cal_symatrix_reverse(s_matrix_num, msize);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, msize, msize, msize, 1.,
                s_matrix[0], msize, s_matrix_num[0], msize, 0., exchange->o_matrix[0], msize);
    if (fgrids_fname != NULL)
    {
        exchange->grids_final = init_grid_opt(fgrids_fname, exchange->mol);
        free_matrix(s_matrix_num);
        s_matrix_num = cal_overlap_numerical(exchange->grids_final);
        exchange->o_matrix_final = malloc_matrix(msize, msize);
        cal_symatrix_reverse(s_matrix_num, msize);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, msize, msize, msize, 1.,
                    s_matrix[0], msize, s_matrix_num[0], msize, 0., exchange->o_matrix_final[0], msize);
    }
    else
        exchange->grids_final = NULL;
    free_matrix(s_matrix);
    free_matrix(s_matrix_num);
    exchange->grids_fname = (char *)malloc(sizeof(char) * (strlen(grids_fname) + 4) / 4 * 4);
    memset(exchange->grids_fname, 0, (strlen(grids_fname) + 4) / 4 * 4);
    memcpy(exchange->grids_fname, grids_fname, strlen(grids_fname));
    return;
}
void del_exchange(exchange_info exchange)
{
    if (exchange->if_aux)
    {
        del_bas(exchange->aux);
        del_bas(exchange->aux_final);
        free_matrix(exchange->aux2e_matrix_rev);
        free_matrix(exchange->aux2e_matrix_rev_final);
    }
    if (exchange->if_grids)
    {
        del_grid(exchange->grids);
        free_matrix(exchange->o_matrix);
        free(exchange->grids_fname);
        if (exchange->grids_final != NULL)
        {
            del_grid(exchange->grids_final);
            free_matrix(exchange->o_matrix_final);
        }
    }
    if (exchange->K != NULL)
        free_tensor3d(exchange->K);
    if (exchange->K_b != NULL)
        free_tensor3d(exchange->K_b);
    free(exchange);
    return;
}
/**
 * @brief calculate exchange matrix by using COSX (or called sn-Link)
 *        DOI: 10.1016/j.chemphys.2008.10.036
 *        DOI: 10.1063/1.3646921
 *        DOI: 10.1063/1.4819264
 *        DOI: 10.1063/1.4819264
 * @param if_finnish if this segment needs to be gapped
 * @param d_matrix density matrix of alpha spin
 * @param d_matrix_b beta spin, if NULL is give then closed shell calculation will be done
 * @param exchange
 */
void cal_k_matrix_cosx(const bool *if_finnish, const Matrix *d_matrix, const Matrix *d_matrix_b, exchange_info exchange)
{
    assert(exchange->if_grids);
    grids_info grids = exchange->grids;
    const mol_info mol = exchange->mol;
    const bas_info bas = mol->bas;
    const pair_info pair = mol->pair;
    int msize = get_mol_msize(mol);
    int batch_size = exchange->batch_size;
    List if_hermit = malloc_list(batch_size);
    List if_hermit_b = malloc_list(batch_size);
    int nbas = bas->nbas;
    double omega_ = exchange->omega_;
    double lr_frac = exchange->lr_frac;
    double sr_frac = exchange->sr_frac;
    Matrix o_matrix = exchange->o_matrix;
    if (exchange->if_final)
    {
        assert(exchange->grids_final != NULL);
        grids = exchange->grids_final;
        o_matrix = exchange->o_matrix_final;
    }
    // judge if density matrix is hermitian one
    for (int i = 0; i < batch_size; i++)
    {
        double tt = 0.;
        for (int n = 0; n < msize; n++)
            for (int m = 0; m < n; m++)
                tt += (d_matrix[i][n][m] - d_matrix[i][m][n]);
        if (fabs(tt) <= 1E-10)
            if_hermit[i] = 1;
        else
        {
            tt = 0.;
            for (int n = 0; n < msize; n++)
                for (int m = 0; m < n; m++)
                    tt += (d_matrix[i][n][m] + d_matrix[i][m][n]);
            if (fabs(tt) <= 1E-10)
                if_hermit[i] = -1;
            else
                if_hermit[i] = 0;
        }
        if (d_matrix_b != NULL)
        {
            tt = 0.;
            for (int n = 0; n < msize; n++)
                for (int m = 0; m < n; m++)
                    tt += (d_matrix_b[i][n][m] - d_matrix_b[i][m][n]);
            if (fabs(tt) <= 1E-10)
                if_hermit_b[i] = 1;
            else
            {
                tt = 0.;
                for (int n = 0; n < msize; n++)
                    for (int m = 0; m < n; m++)
                        tt += (d_matrix_b[i][n][m] + d_matrix_b[i][m][n]);
                if (fabs(tt) <= 1E-10)
                    if_hermit_b[i] = -1;
                else
                    if_hermit_b[i] = 0;
            }
        }
    }
#pragma omp parallel
    {
        xint_info xint;
        xint = init_xint(mol);
        set_xint_rs(lr_frac, sr_frac, omega_, xint);
        Tensor3D K = malloc_tensor3d(batch_size, msize, msize);
        Tensor3D K_b = malloc_tensor3d(batch_size, msize, msize);
#pragma omp for schedule(dynamic, 1) nowait
        for (int n = 0; n < grids->block_num; n++)
        {
            grid_block block = &grids->blocks[n];
            int grid_num = block->grid_num;
            int bas_num = block->bas_num;
            int shell_num = block->shell_num;
            List local2global = block->local2global;
            List global2local = block->global2local;
            Tensor3D d_matrix_local = NULL, d_matrix_b_local = NULL;
            Tensor3D F_ig = NULL, F_ig_b = NULL;
            Matrix F_ig_max = NULL;
            Vector F_ib_max = NULL;
            Matrix X_ig = NULL;
            Vector X_g_max = NULL;
            double X_b_max = 0.;
            Matrix dist_shell_g = NULL;
            Tensor3D G_ig = NULL, G_ig_b = NULL;
            Tensor3D k_matrix_local = NULL, k_matrix_b_local = NULL;
            Matrix o_local = NULL; // matrix used to correct results
            double xyz[3], buf[2048];
            if (bas_num == 0)
                continue;
            dist_shell_g = malloc_matrix(grid_num, nbas);
            o_local = malloc_matrix(bas_num, bas_num);
            // copy O overlap correction matrix to local one
            for (int i = 0, uu = 0; i < shell_num; i++)
            {
                int di = xint_gtolen(mol->bas(ANGULAR_VAL, block->shell_list[i]));
                int pi = bas->shls_p[block->shell_list[i]];
                for (int u = 0; u < di; u++, uu++)
                    for (int j = 0, vv = 0; j < shell_num; j++)
                    {
                        int dj = xint_gtolen(mol->bas(ANGULAR_VAL, block->shell_list[j]));
                        int pj = bas->shls_p[block->shell_list[j]];
                        for (int v = 0; v < dj; v++, vv++)
                            o_local[uu][vv] = o_matrix[pi + u][pj + v];
                    }
            }
            for (int g = 0; g < grid_num; g++)
            {
                xyz[0] = grids->grids[block->grid_list[g]].x;
                xyz[1] = grids->grids[block->grid_list[g]].y;
                xyz[2] = grids->grids[block->grid_list[g]].z;
                for (int i = 0; i < nbas; i++)
                {
                    const Vector A = get_atm_coord(bas(ATOM_IND, i), mol->atm);
                    dist_shell_g[g][i] = (A[0] - xyz[0]) * (A[0] - xyz[0]) + (A[+1] - xyz[1]) * (A[+1] - xyz[1]) +
                                         (A[2] - xyz[2]) * (A[+2] - xyz[2]);
                    dist_shell_g[g][i] = (sqrt(dist_shell_g[g][i]) - grids->bas_extent[i]);
                }
            }
            cal_block_bas(n, grids);
            X_ig = block->psi;
            X_g_max = malloc_vector(grid_num);
            for (int g = 0; g < grid_num; g++)
            {
                X_g_max[g] = fabs(X_ig[g][0]);
                for (int i = 1; i < bas_num; i++)
                {
                    if (fabs(X_ig[g][i]) > X_g_max[g])
                        X_g_max[g] = fabs(X_ig[g][i]);
                }
            }
            for (int g = 0; g < grid_num; g++)
            {
                double tmp_X = 0.;
                for (int i = 0; i < bas_num; i++)
                {
                    tmp_X += X_ig[g][i];
                }
                if (fabs(tmp_X) > X_b_max)
                    X_b_max = fabs(tmp_X);
            }
            for (int g = 0; g < grid_num; g++)
            {
                double weight = sqrt(grids->grids[block->grid_list[g]].w);
                for (int j = 0; j < bas_num; j++)
                {
                    X_ig[g][j] *= weight;
                }
            }
            d_matrix_local = malloc_tensor3d(batch_size, bas_num, msize);
            k_matrix_local = malloc_tensor3d(batch_size, bas_num, msize);
            F_ig = malloc_tensor3d(batch_size, msize, grid_num);
            F_ig_max = malloc_matrix(nbas, grid_num);
            F_ib_max = malloc_vector(nbas);
            G_ig = malloc_tensor3d(batch_size, msize, grid_num);
            for (int a = 0; a < batch_size; a++)
            {
                if (if_finnish[a] == 1)
                    continue;
                for (int i = 0; i < msize; i++)
                {
                    if (global2local[i] < 0)
                        continue;
                    for (int j = 0; j < msize; j++)
                        d_matrix_local[a][global2local[i]][j] = d_matrix[a][i][j];
                }
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, msize, grid_num, bas_num, 1.,
                            d_matrix_local[a][0], msize, X_ig[0], bas_num, 0., F_ig[a][0], grid_num);
                for (int i = 0; i < nbas; i++)
                {
                    int di = xint_gtolen(mol->bas(ANGULAR_VAL, i));
                    for (int g = 0; g < grid_num; g++)
                    {
                        int pi = bas->shls_p[i];
                        for (int d = 0; d < di; d++)
                        {
                            if (fabs(F_ig[a][pi][g]) > F_ig_max[i][g])
                                F_ig_max[i][g] = fabs(F_ig[a][pi][g]);
                            pi++;
                        }
                    }
                }
            }
            if (d_matrix_b != NULL)
            {
                d_matrix_b_local = malloc_tensor3d(batch_size, bas_num, msize);
                k_matrix_b_local = malloc_tensor3d(batch_size, bas_num, msize);
                F_ig_b = malloc_tensor3d(batch_size, msize, grid_num);
                G_ig_b = malloc_tensor3d(batch_size, msize, grid_num);
                for (int a = 0; a < batch_size; a++)
                {
                    if (if_finnish[a] == 1)
                        continue;
                    for (int i = 0; i < msize; i++)
                    {
                        if (global2local[i] < 0)
                            continue;
                        for (int j = 0; j < msize; j++)
                        {
                            d_matrix_b_local[a][global2local[i]][j] = d_matrix_b[a][i][j];
                        }
                    }
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, msize, grid_num, bas_num, 1.,
                                d_matrix_b_local[a][0], msize, X_ig[0], bas_num, 0., F_ig_b[a][0], grid_num);
                    for (int i = 0; i < nbas; i++)
                    {
                        int di = xint_gtolen(mol->bas(ANGULAR_VAL, i));
                        for (int g = 0; g < grid_num; g++)
                        {
                            int pi = bas->shls_p[i];
                            for (int d = 0; d < di; d++)
                            {
                                if (fabs(F_ig_b[a][pi][g]) > F_ig_max[i][g])
                                    F_ig_max[i][g] = fabs(F_ig_b[a][pi][g]);
                                pi++;
                            }
                        }
                    }
                }
            }
            for (int i = 0; i < nbas; i++)
            {
                for (int g = 0; g < grid_num; g++)
                    if (fabs(F_ig_max[i][g]) > F_ib_max[i])
                        F_ib_max[i] = fabs(F_ig_max[i][g]);
            }
            for (int ij = 0; ij < pair->len; ij++)
            {
                double EST = pair->shp[ij].EST;
                int shli = pair->shp[ij].shli;
                int shlj = pair->shp[ij].shlj;
                int di, dj;
                int li, lj;
                int pi, pj;
                double dist_ig, dist_jg;
                double decay, K_est;
                li = mol->bas(ANGULAR_VAL, shli);
                lj = mol->bas(ANGULAR_VAL, shlj);
                di = xint_gtolen(li);
                dj = xint_gtolen(lj);
                pi = bas->shls_p[shli];
                pj = bas->shls_p[shlj];
                K_est = X_b_max * fmax(F_ib_max[shli], F_ib_max[shlj]) * EST;
                if (K_est < 5E-10)
                    continue;
                for (int g = 0; g < grid_num; g++)
                {
                    xyz[0] = grids->grids[block->grid_list[g]].x;
                    xyz[1] = grids->grids[block->grid_list[g]].y;
                    xyz[2] = grids->grids[block->grid_list[g]].z;
                    dist_ig = dist_shell_g[g][shli];
                    dist_jg = dist_shell_g[g][shlj];
                    decay = 1. / fmax(1., fmin(dist_ig, dist_jg));
                    K_est = X_g_max[g] * decay * EST * fmax(F_ig_max[shli][g], F_ig_max[shlj][g]);
                    if (K_est < 5E-11)
                        continue;
                    xint_3c1e(buf, ij, xyz, xint);
                    if (pi == pj)
                        for (int i = 0; i < di * dj; i++)
                            buf[i] *= 0.5;
                    if (li >= lj)
                    {
                        for (int a = 0; a < batch_size; a++)
                            for (int i = 0, uu = 0; i < di; i++)
                                for (int j = 0; j < dj; j++, uu++)
                                {
                                    G_ig[a][pi + i][g] += buf[uu] * F_ig[a][pj + j][g];
                                    G_ig[a][pj + j][g] += buf[uu] * F_ig[a][pi + i][g];
                                }
                        if (G_ig_b != NULL)
                            for (int a = 0; a < batch_size; a++)
                                for (int i = 0, uu = 0; i < di; i++)
                                    for (int j = 0; j < dj; j++, uu++)
                                    {
                                        G_ig_b[a][pi + i][g] += buf[uu] * F_ig_b[a][pj + j][g];
                                        G_ig_b[a][pj + j][g] += buf[uu] * F_ig_b[a][pi + i][g];
                                    }
                    }
                    else
                    {
                        for (int a = 0; a < batch_size; a++)
                            for (int j = 0, uu = 0; j < dj; j++)
                                for (int i = 0; i < di; i++, uu++)
                                {
                                    G_ig[a][pi + i][g] += buf[uu] * F_ig[a][pj + j][g];
                                    G_ig[a][pj + j][g] += buf[uu] * F_ig[a][pi + i][g];
                                }
                        if (G_ig_b != NULL)
                            for (int a = 0; a < batch_size; a++)
                                for (int j = 0, uu = 0; j < dj; j++)
                                    for (int i = 0; i < di; i++, uu++)
                                    {
                                        G_ig_b[a][pi + i][g] += buf[uu] * F_ig_b[a][pj + j][g];
                                        G_ig_b[a][pj + j][g] += buf[uu] * F_ig_b[a][pi + i][g];
                                    }
                    }
                }
            }
            Vector k_matrix_local_tmp = malloc_vector(bas_num * msize);
            for (int a = 0; a < batch_size; a++)
            {
                if (if_finnish[a] == 1)
                    continue;
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, msize, bas_num, grid_num, 1.,
                            G_ig[a][0], grid_num, X_ig[0], bas_num, 0., k_matrix_local_tmp, bas_num);
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, bas_num, msize, bas_num, 1.,
                            o_local[0], bas_num, k_matrix_local_tmp, bas_num, 0., k_matrix_local[a][0], msize);
                for (int i = 0; i < bas_num; i++)
                {
                    for (int j = 0; j < msize; j++)
                        K[a][local2global[i]][j] += k_matrix_local[a][i][j];
                }
            }
            if (k_matrix_b_local != NULL)
            {
                for (int a = 0; a < batch_size; a++)
                {
                    if (if_finnish[a] == 1)
                        continue;
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, msize, bas_num, grid_num, 1.,
                                G_ig_b[a][0], grid_num, X_ig[0], bas_num, 0., k_matrix_local_tmp, bas_num);
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, bas_num, msize, bas_num, 1.,
                                o_local[0], bas_num, k_matrix_local_tmp, bas_num, 0., k_matrix_b_local[a][0], msize);
                    for (int i = 0; i < bas_num; i++)
                    {
                        for (int j = 0; j < msize; j++)
                            K_b[a][local2global[i]][j] += k_matrix_b_local[a][i][j];
                    }
                }
            }
        END:
            clear_block_bas(block);
            free_vector(k_matrix_local_tmp);
            free_tensor3d(F_ig);
            free_tensor3d(G_ig);
            free_matrix(dist_shell_g);
            free_tensor3d(k_matrix_local);
            free_matrix(o_local);
            free_tensor3d(d_matrix_local);
            free_vector(X_g_max);
            free_matrix(F_ig_max);
            free_vector(F_ib_max);
            if (d_matrix_b != NULL)
            {
                free_tensor3d(F_ig_b);
                free_tensor3d(G_ig_b);
                free_tensor3d(k_matrix_b_local);
                free_tensor3d(d_matrix_b_local);
            }
        }
#pragma omp critical
        {
            for (int i = 0; i < msize * msize * batch_size; i++)
                exchange->K[0][0][i] += K[0][0][i];
            if (d_matrix_b != NULL)
                for (int i = 0; i < msize * msize * batch_size; i++)
                    exchange->K_b[0][0][i] += K_b[0][0][i];
            else
                for (int i = 0; i < msize * msize * batch_size; i++)
                    exchange->K_b[0][0][i] += K[0][0][i];
        }
        del_xint(xint);
        free_tensor3d(K);
        free_tensor3d(K_b);
    }
    for (int a = 0; a < batch_size; a++)
    {
        if (if_finnish[a] == 1)
            continue;
        if (if_hermit[a])
            for (int i = 0; i < msize; i++)
                for (int j = 0; j < i; j++)
                    exchange->K[a][i][j] = exchange->K[a][j][i] = 0.5 * (exchange->K[a][i][j] + exchange->K[a][j][i]);
        if (d_matrix_b != NULL)
        {
            if (if_hermit_b[a])
                for (int i = 0; i < msize; i++)
                    for (int j = 0; j < i; j++)
                        exchange->K_b[a][i][j] = exchange->K_b[a][j][i] = 0.5 * (exchange->K_b[a][i][j] + exchange->K_b[a][j][i]);
        }
    }
    free(if_hermit);
    free(if_hermit_b);
    return;
}

void cal_k_matrix(const Matrix d_matrix, const Matrix d_matrix_b, exchange_info exchange)
{
    bool if_finnish = false;
    if (d_matrix_b == NULL)
        return cal_k_matrix_batch(&if_finnish, &d_matrix, NULL, exchange);
    return cal_k_matrix_batch(&if_finnish, &d_matrix, &d_matrix_b, exchange);
}

void cal_k_matrix_batch(const bool *if_finnish, const Matrix *d_matrix, const Matrix *d_matrix_b, exchange_info exchange)
{
    double t1 = omp_get_wtime();
    cal_k_matrix_cosx(if_finnish, d_matrix, d_matrix_b, exchange);
#ifdef DEBUG
    printf("  Time of exchange matrices' calculation is %12.6f s.\n", omp_get_wtime() - t1);
#endif
    return;
}
void cal_exchange_interact(Matrix EK_act, Vector EK_self, const Matrix *d_matrix_act, const Matrix *d_matrix_self, const Matrix *d_matrix_act_b,
                           const Matrix *d_matrix_self_b, const int act_num, const int self_num, exchange_info exchange)
{
    assert(exchange->if_grids);
    const mol_info mol = exchange->mol;
    const pair_info pair = mol->pair;
    const bas_info bas = mol->bas;
    int mon_num = act_num + self_num;
    grids_info grids = exchange->grids;
    int msize = get_mol_msize(mol);
    int nbas = bas->nbas;
    double omega = 0.;
    double lr_frac = 1.;
    double sr_frac = 0.;
    Matrix o_matrix = exchange->o_matrix;
    if (exchange->if_final)
    {
        assert(exchange->grids_final != NULL);
        grids = exchange->grids_final;
        o_matrix = exchange->o_matrix_final;
    }
#pragma omp parallel
    {
        xint_info xint = init_xint(mol);
        ;
        Matrix tmp_EK_act = malloc_matrix(act_num, act_num);
        Vector tmp_EK_self = malloc_vector(self_num);
        Tensor3D K = malloc_tensor3d(mon_num, msize, msize);
        Tensor3D Kb = malloc_tensor3d(mon_num, msize, msize);
#pragma omp for schedule(dynamic, 1) nowait
        for (int n = 0; n < grids->block_num; n++)
        {
            grid_block block = &grids->blocks[n];
            int grid_num = block->grid_num;
            int bas_num = block->bas_num;
            int shell_num = block->shell_num;
            if (bas_num == 0)
                continue;
            const List local2global = block->local2global;
            const List global2local = block->global2local;
            Tensor3D d_matrix_local = NULL, d_matrix_b_local = NULL;
            Tensor3D k_matrix_local = NULL, k_matrix_b_local = NULL;
            Tensor3D Y_ig = NULL, Y_ig_b = NULL;
            Matrix Y_ig_max = NULL;
            Vector Y_ib_max = NULL;
            Matrix X_ig = NULL;
            Vector X_g_max = NULL;
            double X_b_max = 0.;
            Matrix dist_shell_g = NULL;
            Tensor3D Z_ig = NULL, Z_ig_b = NULL;
            Matrix o_local = NULL; // matrix used to correct results
            double xyz[3], buf[2048];
            dist_shell_g = malloc_matrix(grid_num, nbas);
            o_local = malloc_matrix(bas_num, bas_num);
            for (int i = 0, uu = 0; i < shell_num; i++)
            {
                int di = xint_gtolen(mol->bas(ANGULAR_VAL, block->shell_list[i]));
                int pi = bas->shls_p[block->shell_list[i]];
                for (int u = 0; u < di; u++, uu++)
                    for (int j = 0, vv = 0; j < shell_num; j++)
                    {
                        int dj = xint_gtolen(mol->bas(ANGULAR_VAL, block->shell_list[j]));
                        int pj = bas->shls_p[block->shell_list[j]];
                        for (int v = 0; v < dj; v++, vv++)
                            o_local[uu][vv] = o_matrix[pi + u][pj + v];
                    }
            }
            for (int g = 0; g < grid_num; g++)
            {
                xyz[0] = grids->grids[block->grid_list[g]].x;
                xyz[1] = grids->grids[block->grid_list[g]].y;
                xyz[2] = grids->grids[block->grid_list[g]].z;
                for (int i = 0; i < nbas; i++)
                {
                    const Vector A = get_atm_coord(bas(ATOM_IND, i), mol->atm);
                    dist_shell_g[g][i] = (A[0] - xyz[0]) * (A[0] - xyz[0]) + (A[+1] - xyz[1]) * (A[+1] - xyz[1]) +
                                         (A[2] - xyz[2]) * (A[+2] - xyz[2]);
                    dist_shell_g[g][i] = (sqrt(dist_shell_g[g][i]) - grids->bas_extent[i]);
                }
            }
            cal_block_bas(n, grids);
            X_ig = block->psi;
            X_g_max = malloc_vector(grid_num);
            for (int g = 0; g < grid_num; g++)
            {
                for (int i = 0; i < bas_num; i++)
                {
                    if (fabs(X_ig[g][i]) > X_g_max[g])
                        X_g_max[g] = fabs(X_ig[g][i]);
                }
            }
            for (int g = 0; g < grid_num; g++)
            {
                double tmp_X = 0.;
                for (int i = 0; i < bas_num; i++)
                {
                    tmp_X += X_ig[g][i];
                }
                if (fabs(tmp_X) > X_b_max)
                    X_b_max = fabs(tmp_X);
            }
            for (int g = 0; g < grid_num; g++)
            {
                double weight = sqrt(grids->grids[block->grid_list[g]].w);
                for (int j = 0; j < bas_num; j++)
                {
                    X_ig[g][j] *= weight;
                }
            }
            d_matrix_local = malloc_tensor3d(mon_num, bas_num, msize);
            d_matrix_b_local = malloc_tensor3d(mon_num, bas_num, msize);
            k_matrix_local = malloc_tensor3d(mon_num, bas_num, msize);
            k_matrix_b_local = malloc_tensor3d(mon_num, bas_num, msize);
            Y_ig = malloc_tensor3d(mon_num, msize, grid_num);
            Y_ig_b = malloc_tensor3d(mon_num, msize, grid_num);
            Y_ig_max = malloc_matrix(nbas, grid_num);
            Y_ib_max = malloc_vector(nbas);
            Z_ig = malloc_tensor3d(mon_num, msize, grid_num);
            Z_ig_b = malloc_tensor3d(mon_num, msize, grid_num);
            for (int a = 0; a < act_num; a++)
            {
                for (int i = 0; i < msize; i++)
                {
                    if (global2local[i] < 0)
                        continue;
                    for (int j = 0; j < msize; j++)
                    {
                        d_matrix_local[a][global2local[i]][j] = d_matrix_act[a][i][j];
                        if (d_matrix_act_b != NULL)
                            d_matrix_b_local[a][global2local[i]][j] = d_matrix_act_b[a][i][j];
                    }
                }
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, msize, grid_num, bas_num, 1.,
                            d_matrix_local[a][0], msize, X_ig[0], bas_num, 0., Y_ig[a][0], grid_num);
                if (d_matrix_act_b != NULL)
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, msize, grid_num, bas_num, 1.,
                                d_matrix_b_local[a][0], msize, X_ig[0], bas_num, 0., Y_ig_b[a][0], grid_num);
                for (int i = 0; i < nbas; i++)
                {
                    int di = xint_gtolen(mol->bas(ANGULAR_VAL, i));
                    for (int g = 0; g < grid_num; g++)
                    {
                        int pi = bas->shls_p[i];
                        for (int d = 0; d < di; d++)
                        {
                            if (fabs(Y_ig[a][pi][g]) > Y_ig_max[i][g])
                                Y_ig_max[i][g] = fabs(Y_ig[a][pi][g]);
                            if (fabs(Y_ig_b[a][pi][g]) > Y_ig_max[i][g])
                                Y_ig_max[i][g] = fabs(Y_ig_b[a][pi][g]);
                            pi++;
                        }
                    }
                }
            }
            for (int a = 0; a < self_num; a++)
            {
                for (int i = 0; i < msize; i++)
                {
                    if (global2local[i] < 0)
                        continue;
                    for (int j = 0; j < msize; j++)
                    {
                        d_matrix_local[a + act_num][global2local[i]][j] = d_matrix_self[a][i][j];
                        if (d_matrix_self_b != NULL)
                            d_matrix_b_local[a + act_num][global2local[i]][j] = d_matrix_self_b[a][i][j];
                    }
                }
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, msize, grid_num, bas_num, 1.,
                            d_matrix_local[a + act_num][0], msize, X_ig[0], bas_num, 0., Y_ig[a + act_num][0], grid_num);
                if (d_matrix_self_b != NULL)
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, msize, grid_num, bas_num, 1.,
                                d_matrix_b_local[a + act_num][0], msize, X_ig[0], bas_num, 0., Y_ig_b[a + act_num][0], grid_num);
                for (int i = 0; i < nbas; i++)
                {
                    int di = xint_gtolen(mol->bas(ANGULAR_VAL, i));
                    for (int g = 0; g < grid_num; g++)
                    {
                        int pi = bas->shls_p[i];
                        for (int d = 0; d < di; d++)
                        {
                            if (fabs(Y_ig[a + act_num][pi][g]) > Y_ig_max[i][g])
                                Y_ig_max[i][g] = fabs(Y_ig[a + act_num][pi][g]);
                            if (fabs(Y_ig[a + act_num][pi][g]) > Y_ig_max[i][g])
                                Y_ig_max[i][g] = fabs(Y_ig[a + act_num][pi][g]);
                            pi++;
                        }
                    }
                }
            }
            for (int i = 0; i < nbas; i++)
            {
                for (int g = 0; g < grid_num; g++)
                    if (fabs(Y_ig_max[i][g]) > Y_ib_max[i])
                        Y_ib_max[i] = fabs(Y_ig_max[i][g]);
            }
            for (int ij = 0; ij < pair->len; ij++)
            {
                double EST = pair->shp[ij].EST;
                int shli = pair->shp[ij].shli;
                int shlj = pair->shp[ij].shlj;
                int di, dj;
                int li, lj;
                int pi, pj;
                double dist_ig, dist_jg;
                double decay, K_est;
                li = mol->bas(ANGULAR_VAL, shli);
                lj = mol->bas(ANGULAR_VAL, shlj);
                di = xint_gtolen(li);
                dj = xint_gtolen(lj);
                pi = bas->shls_p[shli];
                pj = bas->shls_p[shlj];
                K_est = X_b_max * fmax(Y_ib_max[shli], Y_ib_max[shlj]) * EST;
                if (fabs(K_est) < 1E-10)
                    continue;
                for (int g = 0; g < grid_num; g++)
                {
                    xyz[0] = grids->grids[block->grid_list[g]].x;
                    xyz[1] = grids->grids[block->grid_list[g]].y;
                    xyz[2] = grids->grids[block->grid_list[g]].z;
                    dist_ig = dist_shell_g[g][shli];
                    dist_jg = dist_shell_g[g][shlj];
                    decay = 1. / fmax(1., fmin(dist_ig, dist_jg));
                    K_est = X_g_max[g] * decay * EST * fmax(Y_ig_max[shli][g], Y_ig_max[shlj][g]);
                    if (fabs(K_est) < 5E-11)
                        continue;
                    xint_3c1e(buf, ij, xyz, xint);
                    if (li >= lj)
                        for (int a = 0; a < mon_num; a++)
                        {
                            for (int i = 0, uu = 0; i < di; i++)
                                for (int j = 0; j < dj; j++, uu++)
                                {
                                    Z_ig[a][pi + i][g] += buf[uu] * Y_ig[a][pj + j][g];
                                    Z_ig_b[a][pi + i][g] += buf[uu] * Y_ig_b[a][pj + j][g];
                                    if (pi != pj)
                                    {
                                        Z_ig[a][pj + j][g] += buf[uu] * Y_ig[a][pi + i][g];
                                        Z_ig_b[a][pj + j][g] += buf[uu] * Y_ig_b[a][pi + i][g];
                                    }
                                }
                        }
                    else
                        for (int a = 0; a < mon_num; a++)
                        {
                            for (int j = 0, uu = 0; j < dj; j++)
                                for (int i = 0; i < di; i++, uu++)
                                {
                                    Z_ig[a][pi + i][g] += buf[uu] * Y_ig[a][pj + j][g];
                                    Z_ig_b[a][pi + i][g] += buf[uu] * Y_ig_b[a][pj + j][g];
                                    if (pi != pj)
                                    {
                                        Z_ig[a][pj + j][g] += buf[uu] * Y_ig[a][pi + i][g];
                                        Z_ig_b[a][pj + j][g] += buf[uu] * Y_ig_b[a][pi + i][g];
                                    }
                                }
                        }
                }
            }
            Vector k_matrix_local_tmp = malloc_vector(bas_num * msize);
            for (int a = 0; a < mon_num; a++)
            {
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, msize, bas_num, grid_num, 1.,
                            Z_ig[a][0], grid_num, X_ig[0], bas_num, 0., k_matrix_local_tmp, bas_num);
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, bas_num, msize, bas_num, 1.,
                            o_local[0], bas_num, k_matrix_local_tmp, bas_num, 0., k_matrix_local[a][0], msize);
                for (int i = 0; i < bas_num; i++)
                {
                    for (int j = 0; j < msize; j++)
                    {
                        K[a][local2global[i]][j] += k_matrix_local[a][i][j];
                    }
                }
                if (d_matrix_act_b != NULL)
                {
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, msize, bas_num, grid_num, 1.,
                                Z_ig_b[a][0], grid_num, X_ig[0], bas_num, 0., k_matrix_local_tmp, bas_num);
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, bas_num, msize, bas_num, 1.,
                                o_local[0], bas_num, k_matrix_local_tmp, bas_num, 0., k_matrix_b_local[a][0], msize);
                    for (int i = 0; i < bas_num; i++)
                    {
                        for (int j = 0; j < msize; j++)
                        {
                            Kb[a][local2global[i]][j] += k_matrix_b_local[a][i][j];
                        }
                    }
                }
            }
            free_vector(k_matrix_local_tmp);
            clear_block_bas(block);
            free_tensor3d(Y_ig);
            free_tensor3d(Z_ig);
            free_tensor3d(k_matrix_local);
            free_tensor3d(k_matrix_b_local);
            free_matrix(dist_shell_g);
            free_matrix(o_local);
            free_tensor3d(d_matrix_local);
            free_vector(X_g_max);
            free_matrix(Y_ig_max);
            free_vector(Y_ib_max);
            free_tensor3d(Y_ig_b);
            free_tensor3d(Z_ig_b);
            free_tensor3d(d_matrix_b_local);
        }

#pragma omp critical
        {
            if (d_matrix_act_b != NULL)
            {
                for (int a = 0; a < act_num; a++)
                    for (int b = 0; b < act_num; b++)
                        EK_act[a][b] += matrix_inner(K[a], d_matrix_act[b], msize) + matrix_inner(Kb[a], d_matrix_act_b[b], msize);
                for (int a = act_num; a < mon_num; a++)
                    EK_self[a - act_num] += matrix_inner(K[a], d_matrix_self[a - act_num], msize) + matrix_inner(Kb[a], d_matrix_self_b[a - act_num], msize);
            }
            else
            {
                for (int a = 0; a < act_num; a++)
                    for (int b = 0; b < act_num; b++)
                        EK_act[a][b] += matrix_inner(K[a], d_matrix_act[b], msize) * 2;
                for (int a = act_num; a < mon_num; a++)
                    EK_self[a - act_num] += matrix_inner(K[a], d_matrix_self[a - act_num], msize) * 2;
            }
        }
        del_xint(xint);
        free_matrix(tmp_EK_act);
        free_vector(tmp_EK_self);
        free_tensor3d(K);
        free_tensor3d(Kb);
    }
    for (int i = 0; i < act_num; i++)
        for (int j = 0; j < i; j++)
        {
            EK_act[i][j] += EK_act[j][i];
            EK_act[i][j] *= 0.5;
            EK_act[j][i] = EK_act[i][j];
        }
    return;
}
void cal_exchange_interact_grid(Matrix *EK_act, Vector *EK_self, const Matrix *d_matrix_act, const Matrix *d_matrix_self,
                                const Matrix *d_matrix_act_b, const Matrix *d_matrix_self_b, const int act_num, const int self_num, grids_info grid_in, exchange_info exchange)
{
    assert(exchange->if_grids);
    const mol_info mol = exchange->mol;
    const pair_info pair = mol->pair;
    const bas_info bas = mol->bas;
    int mon_num = act_num + self_num;
    grids_info grids = grid_in;
    int msize = get_mol_msize(mol);
    int nbas = bas->nbas;
    double omega = 0.;
    double lr_frac = 1.;
    double sr_frac = 0.;
    if (grids == NULL)
    {
        grids = exchange->grids;
        if (exchange->if_final)
            grids = exchange->grids_final;
    }
#pragma omp parallel
    {
        xint_info xint = init_xint(mol);
        ;
        Matrix tmp_EK_act = malloc_matrix(act_num, act_num);
        Vector tmp_EK_self = malloc_vector(self_num);
        Tensor3D K = malloc_tensor3d(mon_num, msize, msize);
        Tensor3D Kb = malloc_tensor3d(mon_num, msize, msize);
#pragma omp for schedule(dynamic, 1) nowait
        for (int n = 0; n < grids->block_num; n++)
        {
            grid_block block = &grids->blocks[n];
            int grid_num = block->grid_num;
            int bas_num = block->bas_num;
            int shell_num = block->shell_num;
            if (bas_num == 0)
                continue;
            const List local2global = block->local2global;
            const List global2local = block->global2local;
            Tensor3D d_matrix_local = NULL, d_matrix_b_local = NULL;
            Tensor3D k_matrix_local = NULL, k_matrix_b_local = NULL;
            Tensor3D Y_ig = NULL, Y_ig_b = NULL;
            Matrix Y_ig_max = NULL;
            Vector Y_ib_max = NULL;
            Matrix X_ig = NULL;
            Vector X_g_max = NULL;
            double X_b_max = 0.;
            Matrix dist_shell_g = NULL;
            Tensor3D Z_ig = NULL, Z_ig_b = NULL;
            double xyz[3], buf[2048];
            dist_shell_g = malloc_matrix(grid_num, nbas);
            for (int g = 0; g < grid_num; g++)
            {
                xyz[0] = grids->grids[block->grid_list[g]].x;
                xyz[1] = grids->grids[block->grid_list[g]].y;
                xyz[2] = grids->grids[block->grid_list[g]].z;
                for (int i = 0; i < nbas; i++)
                {
                    const Vector A = get_atm_coord(bas(ATOM_IND, i), mol->atm);
                    dist_shell_g[g][i] = (A[0] - xyz[0]) * (A[0] - xyz[0]) + (A[+1] - xyz[1]) * (A[+1] - xyz[1]) +
                                         (A[2] - xyz[2]) * (A[+2] - xyz[2]);
                    dist_shell_g[g][i] = (sqrt(dist_shell_g[g][i]) - grids->bas_extent[i]);
                }
            }
            cal_block_bas(n, grids);
            X_ig = block->psi;
            X_g_max = malloc_vector(grid_num);
            for (int g = 0; g < grid_num; g++)
            {
                for (int i = 0; i < bas_num; i++)
                {
                    if (fabs(X_ig[g][i]) > X_g_max[g])
                        X_g_max[g] = fabs(X_ig[g][i]);
                }
            }
            for (int g = 0; g < grid_num; g++)
            {
                double tmp_X = 0.;
                for (int i = 0; i < bas_num; i++)
                {
                    tmp_X += X_ig[g][i];
                }
                if (fabs(tmp_X) > X_b_max)
                    X_b_max = fabs(tmp_X);
            }
            for (int g = 0; g < grid_num; g++)
            {
                double weight = sqrt(grids->grids[block->grid_list[g]].w);
                for (int j = 0; j < bas_num; j++)
                {
                    X_ig[g][j] *= weight;
                }
            }
            d_matrix_local = malloc_tensor3d(mon_num, bas_num, msize);
            d_matrix_b_local = malloc_tensor3d(mon_num, bas_num, msize);
            k_matrix_local = malloc_tensor3d(mon_num, bas_num, msize);
            k_matrix_b_local = malloc_tensor3d(mon_num, bas_num, msize);
            Y_ig = malloc_tensor3d(mon_num, msize, grid_num);
            Y_ig_b = malloc_tensor3d(mon_num, msize, grid_num);
            Y_ig_max = malloc_matrix(nbas, grid_num);
            Y_ib_max = malloc_vector(nbas);
            Z_ig = malloc_tensor3d(mon_num, msize, grid_num);
            Z_ig_b = malloc_tensor3d(mon_num, msize, grid_num);
            for (int a = 0; a < act_num; a++)
            {
                for (int i = 0; i < msize; i++)
                {
                    if (global2local[i] < 0)
                        continue;
                    for (int j = 0; j < msize; j++)
                    {
                        d_matrix_local[a][global2local[i]][j] = d_matrix_act[a][i][j];
                        if (d_matrix_act_b != NULL)
                            d_matrix_b_local[a][global2local[i]][j] = d_matrix_act_b[a][i][j];
                    }
                }
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, msize, grid_num, bas_num, 1.,
                            d_matrix_local[a][0], msize, X_ig[0], bas_num, 0., Y_ig[a][0], grid_num);
                if (d_matrix_act_b != NULL)
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, msize, grid_num, bas_num, 1.,
                                d_matrix_b_local[a][0], msize, X_ig[0], bas_num, 0., Y_ig_b[a][0], grid_num);
                for (int i = 0; i < nbas; i++)
                {
                    int di = xint_gtolen(mol->bas(ANGULAR_VAL, i));
                    for (int g = 0; g < grid_num; g++)
                    {
                        int pi = bas->shls_p[i];
                        for (int d = 0; d < di; d++)
                        {
                            if (fabs(Y_ig[a][pi][g]) > Y_ig_max[i][g])
                                Y_ig_max[i][g] = fabs(Y_ig[a][pi][g]);
                            if (fabs(Y_ig_b[a][pi][g]) > Y_ig_max[i][g])
                                Y_ig_max[i][g] = fabs(Y_ig_b[a][pi][g]);
                            pi++;
                        }
                    }
                }
            }
            for (int a = 0; a < self_num; a++)
            {
                for (int i = 0; i < msize; i++)
                {
                    if (global2local[i] < 0)
                        continue;
                    for (int j = 0; j < msize; j++)
                    {
                        d_matrix_local[a + act_num][global2local[i]][j] = d_matrix_self[a][i][j];
                        if (d_matrix_self_b != NULL)
                            d_matrix_b_local[a + act_num][global2local[i]][j] = d_matrix_self_b[a][i][j];
                    }
                }
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, msize, grid_num, bas_num, 1.,
                            d_matrix_local[a + act_num][0], msize, X_ig[0], bas_num, 0., Y_ig[a + act_num][0], grid_num);
                if (d_matrix_self_b != NULL)
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, msize, grid_num, bas_num, 1.,
                                d_matrix_b_local[a + act_num][0], msize, X_ig[0], bas_num, 0., Y_ig_b[a + act_num][0], grid_num);
                for (int i = 0; i < nbas; i++)
                {
                    int di = xint_gtolen(mol->bas(ANGULAR_VAL, i));
                    for (int g = 0; g < grid_num; g++)
                    {
                        int pi = bas->shls_p[i];
                        for (int d = 0; d < di; d++)
                        {
                            if (fabs(Y_ig[a + act_num][pi][g]) > Y_ig_max[i][g])
                                Y_ig_max[i][g] = fabs(Y_ig[a + act_num][pi][g]);
                            if (fabs(Y_ig[a + act_num][pi][g]) > Y_ig_max[i][g])
                                Y_ig_max[i][g] = fabs(Y_ig[a + act_num][pi][g]);
                            pi++;
                        }
                    }
                }
            }
            for (int i = 0; i < nbas; i++)
            {
                for (int g = 0; g < grid_num; g++)
                    if (fabs(Y_ig_max[i][g]) > Y_ib_max[i])
                        Y_ib_max[i] = fabs(Y_ig_max[i][g]);
            }
            for (int ij = 0; ij < pair->len; ij++)
            {
                double EST = pair->shp[ij].EST;
                int shli = pair->shp[ij].shli;
                int shlj = pair->shp[ij].shlj;
                int di, dj;
                int li, lj;
                int pi, pj;
                double dist_ig, dist_jg;
                double decay, K_est;
                li = mol->bas(ANGULAR_VAL, shli);
                lj = mol->bas(ANGULAR_VAL, shlj);
                di = xint_gtolen(li);
                dj = xint_gtolen(lj);
                pi = bas->shls_p[shli];
                pj = bas->shls_p[shlj];
                K_est = X_b_max * fmax(Y_ib_max[shli], Y_ib_max[shlj]) * EST;
                if (fabs(K_est) < 1E-10)
                    continue;
                for (int g = 0; g < grid_num; g++)
                {
                    xyz[0] = grids->grids[block->grid_list[g]].x;
                    xyz[1] = grids->grids[block->grid_list[g]].y;
                    xyz[2] = grids->grids[block->grid_list[g]].z;
                    dist_ig = dist_shell_g[g][shli];
                    dist_jg = dist_shell_g[g][shlj];
                    decay = 1. / fmax(1., fmin(dist_ig, dist_jg));
                    K_est = X_g_max[g] * decay * EST * fmax(Y_ig_max[shli][g], Y_ig_max[shlj][g]);
                    if (fabs(K_est) < 5E-11)
                        continue;
                    xint_3c1e(buf, ij, xyz, xint);
                    if (li >= lj)
                        for (int a = 0; a < mon_num; a++)
                        {
                            for (int i = 0, uu = 0; i < di; i++)
                                for (int j = 0; j < dj; j++, uu++)
                                {
                                    Z_ig[a][pi + i][g] += buf[uu] * Y_ig[a][pj + j][g];
                                    Z_ig_b[a][pi + i][g] += buf[uu] * Y_ig_b[a][pj + j][g];
                                    if (pi != pj)
                                    {
                                        Z_ig[a][pj + j][g] += buf[uu] * Y_ig[a][pi + i][g];
                                        Z_ig_b[a][pj + j][g] += buf[uu] * Y_ig_b[a][pi + i][g];
                                    }
                                }
                        }
                    else
                        for (int a = 0; a < mon_num; a++)
                        {
                            for (int j = 0, uu = 0; j < dj; j++)
                                for (int i = 0; i < di; i++, uu++)
                                {
                                    Z_ig[a][pi + i][g] += buf[uu] * Y_ig[a][pj + j][g];
                                    Z_ig_b[a][pi + i][g] += buf[uu] * Y_ig_b[a][pj + j][g];
                                    if (pi != pj)
                                    {
                                        Z_ig[a][pj + j][g] += buf[uu] * Y_ig[a][pi + i][g];
                                        Z_ig_b[a][pj + j][g] += buf[uu] * Y_ig_b[a][pi + i][g];
                                    }
                                }
                        }
                }
            }
            if (d_matrix_act_b == NULL)
                for (int g = 0; g < grid_num; g++)
                {
                    int g_id = block->grid_list[g];
                    for (int a = 0; a < act_num; a++)
                        for (int b = 0; b <= a; b++)
                        {
                            EK_act[a][b][g_id] = 0.;
                            for (int i = 0; i < msize; i++)
                                EK_act[a][b][g_id] += (Y_ig[a][i][g] * Z_ig[b][i][g] + Y_ig[b][i][g] * Z_ig[a][i][g]);
                            EK_act[b][a][g_id] = EK_act[a][b][g_id];
                        }
                    for (int a = act_num; a < mon_num; a++)
                    {
                        EK_self[a - act_num][g_id] = 0.;
                        for (int i = 0; i < msize; i++)
                            EK_self[a - act_num][g_id] += Y_ig[a][i][g] * Z_ig[a][i][g];
                    }
                }
            else
                for (int g = 0; g < grid_num; g++)
                {
                    int g_id = block->grid_list[g];
                    for (int a = 0; a < act_num; a++)
                        for (int b = 0; b <= a; b++)
                        {
                            EK_act[a][b][g_id] = 0.;
                            for (int i = 0; i < msize; i++)
                                EK_act[a][b][g_id] += 0.25 * (Y_ig[a][i][g] * Z_ig[b][i][g] + Y_ig[b][i][g] * Z_ig[a][i][g] + Y_ig_b[a][i][g] * Z_ig_b[b][i][g] + Y_ig_b[b][i][g] * Z_ig_b[a][i][g]);
                            EK_act[b][a][g_id] = EK_act[b][a][g];
                        }
                    for (int a = act_num; a < mon_num; a++)
                    {
                        EK_self[a - act_num][g_id] = 0.;
                        for (int i = 0; i < msize; i++)
                            EK_self[a - act_num][g_id] += 0.5 * (Y_ig[a][i][g] * Z_ig[a][i][g] + Y_ig_b[a][i][g] * Z_ig_b[a][i][g]);
                    }
                }
            clear_block_bas(block);
            free_tensor3d(Y_ig);
            free_tensor3d(Z_ig);
            free_tensor3d(k_matrix_local);
            free_tensor3d(k_matrix_b_local);
            free_matrix(dist_shell_g);
            free_tensor3d(d_matrix_local);
            free_vector(X_g_max);
            free_matrix(Y_ig_max);
            free_vector(Y_ib_max);
            free_tensor3d(Y_ig_b);
            free_tensor3d(Z_ig_b);
            free_tensor3d(d_matrix_b_local);
        }
        del_xint(xint);
        free_matrix(tmp_EK_act);
        free_vector(tmp_EK_self);
        free_tensor3d(K);
        free_tensor3d(Kb);
    }
    return;
}

void cal_exchange_grad_cosx(Vector *e_grad, const Matrix *d_matrix, const Matrix *d_matrix_b, exchange_info exchange)
{
    assert(exchange->if_grids);
    grids_info grids = exchange->grids;
    const mol_info mol = exchange->mol;
    const bas_info bas = mol->bas;
    const pair_info pair = mol->pair;
    int msize = get_mol_msize(mol);
    int batch_size = exchange->batch_size;
    int nbas = bas->nbas;
    double omega_ = exchange->omega_;
    double lr_frac = exchange->lr_frac;
    double sr_frac = exchange->sr_frac;
    Matrix o_matrix = exchange->o_matrix;
    if (exchange->if_final)
    {
        assert(exchange->grids_final != NULL);
        grids = exchange->grids_final;
        o_matrix = exchange->o_matrix_final;
    }
    double eee = 0.;
    // memset(e_grad[0], 0, sizeof(double) * 3 * get_mol_natm(mol) * batch_size);
#pragma omp parallel
    {
        double ee = 0.;
        xint_info xint;
        Matrix e_grad_tmp = malloc_matrix(batch_size, 3 * get_mol_natm(mol));
        memset(e_grad_tmp[0],0,sizeof(double)*batch_size*3*get_mol_natm(mol));
        xint = init_xint(mol);
        set_xint_rs(lr_frac, sr_frac, omega_, xint);
#pragma omp for schedule(dynamic, 1) nowait
        for (int n = 0; n < grids->block_num; n++)
        {
            grid_block block = &grids->blocks[n];
            int grid_num = block->grid_num;
            int bas_num = block->bas_num;
            int shell_num = block->shell_num;
            List local2global = block->local2global;
            List global2local = block->global2local;
            Tensor3D d_matrix_local = NULL, d_matrix_b_local = NULL;
            Matrix X_tmp = NULL, F_tmp = NULL;
            Tensor3D F_ig = NULL, F_ig_b = NULL;
            Matrix F_ig_max = NULL;
            Vector F_ib_max = NULL;
            Matrix X_ig = NULL;
            Vector X_g_max = NULL;
            double X_b_max = 0.;
            Matrix dist_shell_g = NULL;
            Tensor3D G_ig = NULL, G_ig_b = NULL;
            Matrix Z_ig = NULL, Z_ig_b = NULL;
            Matrix o_local = NULL; // matrix used to correct results
            double xyz[3], buf[2048];
            if (bas_num == 0)
                continue;
            dist_shell_g = malloc_matrix(grid_num, nbas);
            o_local = malloc_matrix(bas_num, bas_num);
            // copy O overlap correction matrix to local one
            for (int i = 0, uu = 0; i < shell_num; i++)
            {
                int di = xint_gtolen(mol->bas(ANGULAR_VAL, block->shell_list[i]));
                int pi = bas->shls_p[block->shell_list[i]];
                for (int u = 0; u < di; u++, uu++)
                    for (int j = 0, vv = 0; j < shell_num; j++)
                    {
                        int dj = xint_gtolen(mol->bas(ANGULAR_VAL, block->shell_list[j]));
                        int pj = bas->shls_p[block->shell_list[j]];
                        for (int v = 0; v < dj; v++, vv++)
                            o_local[uu][vv] = o_matrix[pi + u][pj + v];
                    }
            }
            for (int g = 0; g < grid_num; g++)
            {
                xyz[0] = grids->grids[block->grid_list[g]].x;
                xyz[1] = grids->grids[block->grid_list[g]].y;
                xyz[2] = grids->grids[block->grid_list[g]].z;
                for (int i = 0; i < nbas; i++)
                {
                    const Vector A = get_atm_coord(bas(ATOM_IND, i), mol->atm);
                    dist_shell_g[g][i] = (A[0] - xyz[0]) * (A[0] - xyz[0]) + (A[+1] - xyz[1]) * (A[+1] - xyz[1]) +
                                         (A[2] - xyz[2]) * (A[+2] - xyz[2]);
                    dist_shell_g[g][i] = (sqrt(dist_shell_g[g][i]) - grids->bas_extent[i]);
                }
            }
            cal_block_bas(n, grids);
            X_ig = block->psi;
            X_g_max = malloc_vector(grid_num);
            for (int g = 0; g < grid_num; g++)
            {
                X_g_max[g] = fabs(X_ig[g][0]);
                for (int i = 1; i < bas_num; i++)
                {
                    if (fabs(X_ig[g][i]) > X_g_max[g])
                        X_g_max[g] = fabs(X_ig[g][i]);
                }
            }
            for (int g = 0; g < grid_num; g++)
            {
                double tmp_X = 0.;
                for (int i = 0; i < bas_num; i++)
                {
                    tmp_X += X_ig[g][i];
                }
                if (fabs(tmp_X) > X_b_max)
                    X_b_max = fabs(tmp_X);
            }
            for (int g = 0; g < grid_num; g++)
            {
                double weight = sqrt(grids->grids[block->grid_list[g]].w);
                for (int j = 0; j < bas_num; j++)
                {
                    X_ig[g][j] *= weight;
                    block->psix[g][j] *= -weight;
                    block->psiy[g][j] *= -weight;
                    block->psiz[g][j] *= -weight;
                }
            }
            d_matrix_local = malloc_tensor3d(batch_size, bas_num, msize);
            X_tmp = malloc_matrix(bas_num, grid_num);
            F_tmp = malloc_matrix(msize, grid_num);
            F_ig = malloc_tensor3d(batch_size, msize, grid_num);
            F_ig_max = malloc_matrix(nbas, grid_num);
            F_ib_max = malloc_vector(nbas);
            G_ig = malloc_tensor3d(batch_size, msize, grid_num);
            Z_ig = malloc_matrix(bas_num, grid_num);
            for (int a = 0; a < batch_size; a++)
            {
                for (int i = 0; i < msize; i++)
                {
                    if (global2local[i] < 0)
                        continue;
                    for (int j = 0; j < msize; j++)
                        d_matrix_local[a][global2local[i]][j] = d_matrix[a][i][j];
                }
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, msize, grid_num, bas_num, 1.,
                            d_matrix_local[a][0], msize, X_ig[0], bas_num, 0., F_ig[a][0], grid_num);
                for (int i = 0; i < nbas; i++)
                {
                    int di = xint_gtolen(mol->bas(ANGULAR_VAL, i));
                    for (int g = 0; g < grid_num; g++)
                    {
                        int pi = bas->shls_p[i];
                        for (int d = 0; d < di; d++)
                        {
                            if (fabs(F_ig[a][pi][g]) > F_ig_max[i][g])
                                F_ig_max[i][g] = fabs(F_ig[a][pi][g]);
                            pi++;
                        }
                    }
                }
            }
            if (d_matrix_b != NULL)
            {
                d_matrix_b_local = malloc_tensor3d(batch_size, bas_num, msize);
                F_ig_b = malloc_tensor3d(batch_size, msize, grid_num);
                G_ig_b = malloc_tensor3d(batch_size, msize, grid_num);
                Z_ig_b = malloc_matrix(bas_num, grid_num);
                for (int a = 0; a < batch_size; a++)
                {
                    for (int i = 0; i < msize; i++)
                    {
                        if (global2local[i] < 0)
                            continue;
                        for (int j = 0; j < msize; j++)
                        {
                            d_matrix_b_local[a][global2local[i]][j] = d_matrix_b[a][i][j];
                        }
                    }
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, msize, grid_num, bas_num, 1.,
                                d_matrix_b_local[a][0], msize, X_ig[0], bas_num, 0., F_ig_b[a][0], grid_num);
                    for (int i = 0; i < nbas; i++)
                    {
                        int di = xint_gtolen(mol->bas(ANGULAR_VAL, i));
                        for (int g = 0; g < grid_num; g++)
                        {
                            int pi = bas->shls_p[i];
                            for (int d = 0; d < di; d++)
                            {
                                if (fabs(F_ig_b[a][pi][g]) > F_ig_max[i][g])
                                    F_ig_max[i][g] = fabs(F_ig_b[a][pi][g]);
                                pi++;
                            }
                        }
                    }
                }
            }
            for (int i = 0; i < nbas; i++)
            {
                for (int g = 0; g < grid_num; g++)
                    if (fabs(F_ig_max[i][g]) > F_ib_max[i])
                        F_ib_max[i] = fabs(F_ig_max[i][g]);
            }
            for (int ij = 0; ij < pair->len; ij++)
            {
                double EST = pair->shp[ij].EST;
                int shli = pair->shp[ij].shli;
                int shlj = pair->shp[ij].shlj;
                int di, dj;
                int li, lj;
                int pi, pj;
                double dist_ig, dist_jg;
                double decay, K_est;
                li = mol->bas(ANGULAR_VAL, shli);
                lj = mol->bas(ANGULAR_VAL, shlj);
                di = xint_gtolen(li);
                dj = xint_gtolen(lj);
                pi = bas->shls_p[shli];
                pj = bas->shls_p[shlj];
                K_est = X_b_max * fmax(F_ib_max[shli], F_ib_max[shlj]) * EST;
                if (K_est < 5E-9)
                    continue;
                for (int g = 0; g < grid_num; g++)
                {
                    xyz[0] = grids->grids[block->grid_list[g]].x;
                    xyz[1] = grids->grids[block->grid_list[g]].y;
                    xyz[2] = grids->grids[block->grid_list[g]].z;
                    dist_ig = dist_shell_g[g][shli];
                    dist_jg = dist_shell_g[g][shlj];
                    decay = 1. / fmax(1., fmin(dist_ig, dist_jg));
                    K_est = X_g_max[g] * decay * EST * fmax(F_ig_max[shli][g], F_ig_max[shlj][g]);
                    if (K_est < 5E-11)
                        continue;
                    xint_3c1e(buf, ij, xyz, xint);
                    if (pi == pj)
                        for (int i = 0; i < di * dj; i++)
                            buf[i] *= 0.5;
                    if (li >= lj)
                    {
                        for (int a = 0; a < batch_size; a++)
                            for (int i = 0, uu = 0; i < di; i++)
                                for (int j = 0; j < dj; j++, uu++)
                                {
                                    G_ig[a][pi + i][g] += buf[uu] * F_ig[a][pj + j][g];
                                    G_ig[a][pj + j][g] += buf[uu] * F_ig[a][pi + i][g];
                                }
                        if (G_ig_b != NULL)
                            for (int a = 0; a < batch_size; a++)
                                for (int i = 0, uu = 0; i < di; i++)
                                    for (int j = 0; j < dj; j++, uu++)
                                    {
                                        G_ig_b[a][pi + i][g] += buf[uu] * F_ig_b[a][pj + j][g];
                                        G_ig_b[a][pj + j][g] += buf[uu] * F_ig_b[a][pi + i][g];
                                    }
                    }
                    else
                    {
                        for (int a = 0; a < batch_size; a++)
                            for (int j = 0, uu = 0; j < dj; j++)
                                for (int i = 0; i < di; i++, uu++)
                                {
                                    G_ig[a][pi + i][g] += buf[uu] * F_ig[a][pj + j][g];
                                    G_ig[a][pj + j][g] += buf[uu] * F_ig[a][pi + i][g];
                                }
                        if (G_ig_b != NULL)
                            for (int a = 0; a < batch_size; a++)
                                for (int j = 0, uu = 0; j < dj; j++)
                                    for (int i = 0; i < di; i++, uu++)
                                    {
                                        G_ig_b[a][pi + i][g] += buf[uu] * F_ig_b[a][pj + j][g];
                                        G_ig_b[a][pj + j][g] += buf[uu] * F_ig_b[a][pi + i][g];
                                    }
                    }
                }
            }
            double factor = 2.;
            if (d_matrix_b == NULL)
                factor *= 2;
            Matrix psix_t = malloc_matrix(bas_num, grid_num);
            Matrix psiy_t = malloc_matrix(bas_num, grid_num);
            Matrix psiz_t = malloc_matrix(bas_num, grid_num);
            Matrix tmp_d = malloc_matrix(bas_num, msize);
            for (int i = 0; i < bas_num; i++)
                for (int g = 0; g < grid_num; g++)
                {
                    psix_t[i][g] = block->psix[g][i];
                    psiy_t[i][g] = block->psiy[g][i];
                    psiz_t[i][g] = block->psiz[g][i];
                }
            for (int a = 0; a < batch_size; a++)
            {

                copy_matrix(tmp_d, d_matrix_local[a], bas_num, msize);
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bas_num, msize, bas_num, 1., o_local[0], bas_num,
                            tmp_d[0], msize, 0., d_matrix_local[a][0], msize);
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bas_num, grid_num, msize, 1.,
                            d_matrix_local[a][0], msize, G_ig[a][0], grid_num, 0., Z_ig[0], grid_num);
                if (d_matrix_b != NULL)
                {
                    copy_matrix(tmp_d, d_matrix_b_local[a], bas_num, msize);
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bas_num, msize, bas_num, 1., o_local[0], bas_num,
                                tmp_d[0], msize, 0., d_matrix_b_local[a][0], msize);
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bas_num, grid_num, msize, 1.,
                                d_matrix_b_local[a][0], msize, G_ig_b[a][0], grid_num, 0., Z_ig_b[0], grid_num);
                }
                for (int i = 0, uu = 0, vv = 0; i < shell_num; i++)
                {
                    int atm_id = bas(ATOM_IND, block->shell_list[i]);
                    int di = xint_gtolen(bas(ANGULAR_VAL, block->shell_list[i]));
                    for (int u = 0; u < di; u++, uu++)
                    {
                        e_grad_tmp[a][atm_id * 3 + 0] -= factor * vv_dot(psix_t[uu], Z_ig[uu], grid_num);
                        e_grad_tmp[a][atm_id * 3 + 1] -= factor * vv_dot(psiy_t[uu], Z_ig[uu], grid_num);
                        e_grad_tmp[a][atm_id * 3 + 2] -= factor * vv_dot(psiz_t[uu], Z_ig[uu], grid_num);
                    }
                    if (d_matrix_b != NULL)
                        for (int u = 0; u < di; u++, vv++)
                        {
                            e_grad_tmp[a][atm_id * 3 + 0] -= factor * vv_dot(psix_t[vv], Z_ig_b[vv], grid_num);
                            e_grad_tmp[a][atm_id * 3 + 1] -= factor * vv_dot(psiy_t[vv], Z_ig_b[vv], grid_num);
                            e_grad_tmp[a][atm_id * 3 + 2] -= factor * vv_dot(psiz_t[vv], Z_ig_b[vv], grid_num);
                        }
                }
//                printf("n = %d  a = %d\n",n,a);
//                printf("e_grad_tmp:\n");
//                for (int kk=0;kk<mol->atm->natm;kk++)
//                  printf("%f %f %f\n",e_grad_tmp[a][3*kk],e_grad_tmp[a][3*kk+1],e_grad_tmp[a][3*kk+2]);
            }
            free_matrix(psix_t);
            free_matrix(psiy_t);
            free_matrix(psiz_t);
            free_matrix(tmp_d);
        END:
            clear_block_bas(block);
            free_matrix(X_tmp);
            free_matrix(F_tmp);
            free_tensor3d(F_ig);
            free_tensor3d(G_ig);
            free_matrix(dist_shell_g);
            free_matrix(o_local);
            free_tensor3d(d_matrix_local);
            free_vector(X_g_max);
            free_matrix(F_ig_max);
            free_vector(F_ib_max);
            free_matrix(Z_ig);
            if (d_matrix_b != NULL)
            {
                free_tensor3d(F_ig_b);
                free_tensor3d(G_ig_b);
                free_matrix(Z_ig_b);
                free_tensor3d(d_matrix_b_local);
            }
        }
#pragma omp critical
        {
            for (int a = 0; a < batch_size; a++)
                for (int i = 0; i < 3 * get_mol_natm(mol); i++)
                    e_grad[a][i] += e_grad_tmp[a][i];
            eee += ee;
        }
        del_xint(xint);
        free_matrix(e_grad_tmp);
    }
    printf("EEE: %12.6f, %12.6f\n", eee, matrix_inner(d_matrix[0], exchange->K[0], msize));
    return;
}
