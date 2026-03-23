#include "mol/mol.h"
pair_info init_pair(const bas_info bas)
{
    pair_info pair = (pair_info)malloc(sizeof(struct PairInfo));
    int nbas = bas->nbas;
    double *value_bas = bas->value;
    atm_info atm = bas->atm;
    int if_in;
    double *expi, *expj;
    double *coei, *coej;
    double *A, *B, M[3], PM[3];
    double AB[3], P[3 * 1024], K_ab[4 * 1024];
    double xi[1024], half_xi_[1024];
    double extent;
    double EST;
    shell_pair shp_tmp = (shell_pair)malloc(sizeof(struct ShellPair));
    binary_list *tmp_list = (binary_list *)malloc(sizeof(binary_list) * nbas);
    pair->len = 0;
    pair->shp = (shell_pair)malloc(sizeof(struct ShellPair) * (nbas + 1) * nbas * 0.5);
    shell_pair shp = &pair->shp[0];
    for (int shli = 0; shli < nbas; shli++)
    {
        for (int shlj = 0; shlj <= shli; shlj++)
        {
            EST = 0.;
            if_in = 0;
            extent = 0.;
            if (bas(ANGULAR_VAL, shli) >= bas(ANGULAR_VAL, shlj))
            {
                shp_tmp->shlj = shlj;
                shp_tmp->lj = bas(ANGULAR_VAL, shlj);
                shp_tmp->dj = bas(NPRIM_VAL, shlj);
                shp_tmp->shli = shli;
                shp_tmp->li = bas(ANGULAR_VAL, shli);
                shp_tmp->di = bas(NPRIM_VAL, shli);
                shp_tmp->dij = shp_tmp->di * shp_tmp->dj;
                expi = &value_bas[bas(EXP_IND, shli)];
                coei = &value_bas[bas(COEFF_IND, shli)];
                expj = &value_bas[bas(EXP_IND, shlj)];
                coej = &value_bas[bas(COEFF_IND, shlj)];
                B = get_atm_coord(bas(ATOM_IND, shlj), atm);
                A = get_atm_coord(bas(ATOM_IND, shli), atm);
                M[0] = (A[0] + B[0]) * 0.5;
                M[1] = (A[1] + B[1]) * 0.5;
                M[2] = (A[2] + B[2]) * 0.5;
                AB[0] = A[0] - B[0];
                AB[1] = A[1] - B[1];
                AB[2] = A[2] - B[2];
            }
            else
            {
                shp_tmp->shlj = shli;
                shp_tmp->lj = bas(ANGULAR_VAL, shli);
                shp_tmp->dj = bas(NPRIM_VAL, shli);
                shp_tmp->shli = shlj;
                shp_tmp->li = bas(ANGULAR_VAL, shlj);
                shp_tmp->di = bas(NPRIM_VAL, shlj);
                shp_tmp->dij = shp_tmp->di * shp_tmp->dj;
                expi = &value_bas[bas(EXP_IND, shlj)];
                coei = &value_bas[bas(COEFF_IND, shlj)];
                expj = &value_bas[bas(EXP_IND, shli)];
                coej = &value_bas[bas(COEFF_IND, shli)];
                B = get_atm_coord(bas(ATOM_IND, shli), atm);
                A = get_atm_coord(bas(ATOM_IND, shlj), atm);
                M[0] = (A[0] + B[0]) * 0.5;
                M[1] = (A[1] + B[1]) * 0.5;
                M[2] = (A[2] + B[2]) * 0.5;
                AB[0] = A[0] - B[0];
                AB[1] = A[1] - B[1];
                AB[2] = A[2] - B[2];
            }
            for (int i = 0, u = 0; i < shp_tmp->di; i++)
                for (int j = 0; j < shp_tmp->dj; j++, u++)
                {
                    xi[u] = expi[i] + expj[j];
                    half_xi_[u] = 0.5 / xi[u];
                    K_ab[4 * u + 0] = exp(-2 * AB[0] * AB[0] * half_xi_[u] * expi[i] * expj[j]);
                    K_ab[4 * u + 1] = exp(-2 * AB[1] * AB[1] * half_xi_[u] * expi[i] * expj[j]);
                    K_ab[4 * u + 2] = exp(-2 * AB[2] * AB[2] * half_xi_[u] * expi[i] * expj[j]);
                    K_ab[4 * u + 3] = K_ab[4 * u + 0] * K_ab[4 * u + 1] * K_ab[4 * u + 2];
                    P[3 * u + 0] = 2 * (A[0] * expi[i] + B[0] * expj[j]) * half_xi_[u];
                    P[3 * u + 1] = 2 * (A[1] * expi[i] + B[1] * expj[j]) * half_xi_[u];
                    P[3 * u + 2] = 2 * (A[2] * expi[i] + B[2] * expj[j]) * half_xi_[u];
                    EST += K_ab[4 * u + 3] * 4 * MY_PI * half_xi_[u] * coei[i] * coej[j];
                    if (K_ab[4 * u + 3] >= PAIR_TOLER)
                    {
                        PM[0] = P[0] - M[0];
                        PM[1] = P[1] - M[1];
                        PM[2] = P[2] - M[2];
                        if_in = 1;
                        extent = fmax(extent, PM[0] * PM[0] + PM[1] * PM[1] + PM[2] * PM[2] + sqrt(2 * half_xi_[u] * log(1E10)));
                    }
                }
            if (if_in)
            {
                shp->dij = shp_tmp->dij;
                shp->extent = extent;
                shp->xi = (double *)malloc(sizeof(double) * shp_tmp->dij);
                shp->half_xi_ = (double *)malloc(sizeof(double) * shp_tmp->dij);
                shp->P = (double *)malloc(3 * sizeof(double) * shp_tmp->dij);
                shp->PA = (double *)malloc(3 * sizeof(double) * shp_tmp->dij);
                shp->PB = (double *)malloc(3 * sizeof(double) * shp_tmp->dij);
                shp->K_ab = (double *)malloc(4 * sizeof(double) * shp_tmp->dij);
                shp->EST = fabs(EST);
                shp->AB[0] = AB[0];
                shp->AB[1] = AB[1];
                shp->AB[2] = AB[2];
                shp->shli = shp_tmp->shli;
                shp->shlj = shp_tmp->shlj;
                shp->di = shp_tmp->di;
                shp->dj = shp_tmp->dj;
                shp->li = shp_tmp->li;
                shp->lj = shp_tmp->lj;
                memcpy(shp->xi, xi, sizeof(double) * shp->dij);
                memcpy(shp->half_xi_, half_xi_, sizeof(double) * shp->dij);
                memcpy(shp->P, P, 3 * sizeof(double) * shp->dij);
                memcpy(shp->K_ab, K_ab, 4 * sizeof(double) * shp->dij);
                for (int i = 0; i < shp->dij; i++)
                {
                    shp->PA[i * 3 + 0] = shp->P[i * 3 + 0] - A[0];
                    shp->PA[i * 3 + 1] = shp->P[i * 3 + 1] - A[1];
                    shp->PA[i * 3 + 2] = shp->P[i * 3 + 2] - A[2];
                    shp->PB[i * 3 + 0] = shp->P[i * 3 + 0] - B[0];
                    shp->PB[i * 3 + 1] = shp->P[i * 3 + 1] - B[1];
                    shp->PB[i * 3 + 2] = shp->P[i * 3 + 2] - B[2];
                }
                ++pair->len;
                ++shp;
            }
        }
    }
    for (int i = 0; i < nbas; i++)
    {
        tmp_list[i].len = 0;
        tmp_list[i].data = (binary_num *)malloc(sizeof(binary_num) * nbas);
    }
    for (int i = 0; i < pair->len; i++)
    {
        int u, v;
        u = pair->shp[i].shli;
        v = pair->shp[i].shlj;
        tmp_list[u].data[tmp_list[u].len].i = v;
        tmp_list[u].data[tmp_list[u].len].j = i;
        if (u != v)
        {
            tmp_list[v].data[tmp_list[v].len].i = u;
            tmp_list[v].data[tmp_list[v].len].j = i;
            tmp_list[v].len++;
        }
        tmp_list[u].len++;
    }
    for (int i = 0; i < nbas; i++)
        sort_binary_list(&tmp_list[i]);
    pair->search_tree = (qtree *)malloc(sizeof(qtree) * nbas);
    int *u_list, *v_list;
    u_list = (int *)malloc(sizeof(int) * nbas);
    v_list = (int *)malloc(sizeof(int) * nbas);
    for (int i = 0; i < nbas; i++)
    {
        init_tree(&pair->search_tree[i]);
        for (int j = 0; j < tmp_list[i].len; j++)
        {
            u_list[j] = tmp_list[i].data[j].i;
            v_list[j] = tmp_list[i].data[j].j;
        }
        make_tree(u_list, v_list, tmp_list[i].len, &pair->search_tree[i]);
    }
    free(u_list);
    free(v_list);
    for (int i = 0; i < nbas; i++)
    {
        free(tmp_list[i].data);
    }
    free(tmp_list);
    free(shp_tmp);
    pair->nbas = nbas;
    return pair;
}
pair_info init_prm_pair(const prm_info prm)
{
    pair_info pair = (pair_info)malloc(sizeof(struct PairInfo));
    int nprm = prm->nprm;
    double *value_prm = prm->value;
    atm_info atm = prm->atm;
    double *value_atm = atm->value;
    int if_in;
    double *expi, *expj;
    double A[3], B[3], M[3], PM[3];
    double AB[3], P[3 * 1024], K_ab[4 * 1024];
    double xi[1024], half_xi_[1024];
    double extent;
    double EST;
    shell_pair shp_tmp = (shell_pair)malloc(sizeof(struct ShellPair));
    binary_list *tmp_list = (binary_list *)malloc(sizeof(binary_list) * nprm);
    pair->len = 0;
    pair->shp = (shell_pair)malloc(sizeof(struct ShellPair) * (nprm + 1) * nprm * 0.5);
    shell_pair shp = &pair->shp[0];
    for (int shli = 0; shli < nprm; shli++)
    {
        for (int shlj = 0; shlj <= shli; shlj++)
        {
            EST = 0.;
            if_in = 0;
            extent = 0.;
            if (prm(ANGULAR_VAL, shli) >= prm(ANGULAR_VAL, shlj))
            {
                shp_tmp->shlj = shlj;
                shp_tmp->lj = prm(ANGULAR_VAL, shlj);
                shp_tmp->dj = 1;
                shp_tmp->shli = shli;
                shp_tmp->li = prm(ANGULAR_VAL, shli);
                shp_tmp->di = 1;
                shp_tmp->dij = 1;
                expi = &value_prm[prm(EXP_IND, shli)];
                expj = &value_prm[prm(EXP_IND, shlj)];
                B[0] = value_atm[atm(CENTER_IND, prm(ATOM_IND, shlj)) + 0];
                B[1] = value_atm[atm(CENTER_IND, prm(ATOM_IND, shlj)) + 1];
                B[2] = value_atm[atm(CENTER_IND, prm(ATOM_IND, shlj)) + 2];
                A[0] = value_atm[atm(CENTER_IND, prm(ATOM_IND, shli)) + 0];
                A[1] = value_atm[atm(CENTER_IND, prm(ATOM_IND, shli)) + 1];
                A[2] = value_atm[atm(CENTER_IND, prm(ATOM_IND, shli)) + 2];
                M[0] = (A[0] + B[0]) * 0.5;
                M[1] = (A[1] + B[1]) * 0.5;
                M[2] = (A[2] + B[2]) * 0.5;
                AB[0] = A[0] - B[0];
                AB[1] = A[1] - B[1];
                AB[2] = A[2] - B[2];
            }
            else
            {
                shp_tmp->shlj = shli;
                shp_tmp->lj = prm(ANGULAR_VAL, shli);
                shp_tmp->dj = 1;
                shp_tmp->shli = shlj;
                shp_tmp->li = prm(ANGULAR_VAL, shlj);
                shp_tmp->di = 1;
                shp_tmp->dij = 1;
                expi = &value_prm[prm(EXP_IND, shlj)];
                expj = &value_prm[prm(EXP_IND, shli)];
                B[0] = value_atm[atm(CENTER_IND, prm(ATOM_IND, shli)) + 0];
                B[1] = value_atm[atm(CENTER_IND, prm(ATOM_IND, shli)) + 1];
                B[2] = value_atm[atm(CENTER_IND, prm(ATOM_IND, shli)) + 2];
                A[0] = value_atm[atm(CENTER_IND, prm(ATOM_IND, shlj)) + 0];
                A[1] = value_atm[atm(CENTER_IND, prm(ATOM_IND, shlj)) + 1];
                A[2] = value_atm[atm(CENTER_IND, prm(ATOM_IND, shlj)) + 2];
                M[0] = (A[0] + B[0]) * 0.5;
                M[1] = (A[1] + B[1]) * 0.5;
                M[2] = (A[2] + B[2]) * 0.5;
                AB[0] = A[0] - B[0];
                AB[1] = A[1] - B[1];
                AB[2] = A[2] - B[2];
            }
            for (int i = 0, u = 0; i < shp_tmp->di; i++)
                for (int j = 0; j < shp_tmp->dj; j++, u++)
                {
                    xi[u] = expi[i] + expj[j];
                    half_xi_[u] = 0.5 / xi[u];
                    K_ab[4 * u + 0] = exp(-2 * AB[0] * AB[0] * half_xi_[u] * expi[i] * expj[j]);
                    K_ab[4 * u + 1] = exp(-2 * AB[1] * AB[1] * half_xi_[u] * expi[i] * expj[j]);
                    K_ab[4 * u + 2] = exp(-2 * AB[2] * AB[2] * half_xi_[u] * expi[i] * expj[j]);
                    K_ab[4 * u + 3] = K_ab[4 * u + 0] * K_ab[4 * u + 1] * K_ab[4 * u + 2];
                    P[3 * u + 0] = 2 * (A[0] * expi[i] + B[0] * expj[j]) * half_xi_[u];
                    P[3 * u + 1] = 2 * (A[1] * expi[i] + B[1] * expj[j]) * half_xi_[u];
                    P[3 * u + 2] = 2 * (A[2] * expi[i] + B[2] * expj[j]) * half_xi_[u];
                    EST += K_ab[4 * u + 3] * 4 * MY_PI * half_xi_[u];
                    if (K_ab[4 * u + 3] >= PAIR_TOLER)
                    {
                        PM[0] = P[0] - M[0];
                        PM[1] = P[1] - M[1];
                        PM[2] = P[2] - M[2];
                        if_in = 1;
                        extent = fmax(extent, PM[0] * PM[0] + PM[1] * PM[1] + PM[2] * PM[2] + sqrt(2 * half_xi_[u] * log(1E10)));
                    }
                }
            if (if_in)
            {
                shp->dij = shp_tmp->dij;
                shp->extent = extent;
                shp->xi = (double *)malloc(sizeof(double) * shp_tmp->dij);
                shp->half_xi_ = (double *)malloc(sizeof(double) * shp_tmp->dij);
                shp->P = (double *)malloc(3 * sizeof(double) * shp_tmp->dij);
                shp->PA = (double *)malloc(3 * sizeof(double) * shp_tmp->dij);
                shp->PB = (double *)malloc(3 * sizeof(double) * shp_tmp->dij);
                shp->K_ab = (double *)malloc(4 * sizeof(double) * shp_tmp->dij);
                shp->EST = fabs(EST);
                shp->AB[0] = AB[0];
                shp->AB[1] = AB[1];
                shp->AB[2] = AB[2];
                shp->shli = shp_tmp->shli;
                shp->shlj = shp_tmp->shlj;
                shp->di = shp_tmp->di;
                shp->dj = shp_tmp->dj;
                shp->li = shp_tmp->li;
                shp->lj = shp_tmp->lj;
                memcpy(shp->xi, xi, sizeof(double) * shp->dij);
                memcpy(shp->half_xi_, half_xi_, sizeof(double) * shp->dij);
                memcpy(shp->P, P, 3 * sizeof(double) * shp->dij);
                memcpy(shp->K_ab, K_ab, 4 * sizeof(double) * shp->dij);
                for (int i = 0; i < shp->dij; i++)
                {
                    shp->PA[i * 3 + 0] = shp->P[i * 3 + 0] - A[0];
                    shp->PA[i * 3 + 1] = shp->P[i * 3 + 1] - A[1];
                    shp->PA[i * 3 + 2] = shp->P[i * 3 + 2] - A[2];
                    shp->PB[i * 3 + 0] = shp->P[i * 3 + 0] - B[0];
                    shp->PB[i * 3 + 1] = shp->P[i * 3 + 1] - B[1];
                    shp->PB[i * 3 + 2] = shp->P[i * 3 + 2] - B[2];
                }
                ++pair->len;
                ++shp;
            }
        }
    }
    for (int i = 0; i < nprm; i++)
    {
        tmp_list[i].len = 0;
        tmp_list[i].data = (binary_num *)malloc(sizeof(binary_num) * nprm);
    }
    for (int i = 0; i < pair->len; i++)
    {
        int u, v;
        u = pair->shp[i].shli;
        v = pair->shp[i].shlj;
        tmp_list[u].data[tmp_list[u].len].i = v;
        tmp_list[u].data[tmp_list[u].len].j = i;
        if (u != v)
        {
            tmp_list[v].data[tmp_list[v].len].i = u;
            tmp_list[v].data[tmp_list[v].len].j = i;
            tmp_list[v].len++;
        }
        tmp_list[u].len++;
    }
    for (int i = 0; i < nprm; i++)
        sort_binary_list(&tmp_list[i]);
    pair->search_tree = (qtree *)malloc(sizeof(qtree) * nprm);
    int *u_list, *v_list;
    u_list = (int *)malloc(sizeof(int) * nprm);
    v_list = (int *)malloc(sizeof(int) * nprm);
    for (int i = 0; i < nprm; i++)
    {
        init_tree(&pair->search_tree[i]);
        for (int j = 0; j < tmp_list[i].len; j++)
        {
            u_list[j] = tmp_list[i].data[j].i;
            v_list[j] = tmp_list[i].data[j].j;
        }
        make_tree(u_list, v_list, tmp_list[i].len, &pair->search_tree[i]);
    }
    free(u_list);
    free(v_list);
    for (int i = 0; i < nprm; i++)
    {
        free(tmp_list[i].data);
    }
    free(tmp_list);
    free(shp_tmp);
    pair->nbas = nprm;
    return pair;
}
void del_pair(pair_info pair)
{
    for (int i = 0; i < pair->len; i++)
    {
        free(pair->shp[i].xi);
        free(pair->shp[i].half_xi_);
        free(pair->shp[i].K_ab);
        free(pair->shp[i].P);
        free(pair->shp[i].PA);
        free(pair->shp[i].PB);
    }
    for (int i = 0; i < pair->nbas; i++)
        del_tree(&pair->search_tree[i]);
    free(pair->search_tree);
    free(pair->shp);
    free(pair);
    return;
}
shell_pair search_pair_shp(int shli, int shlj, const pair_info pair)
{
    if (shli < shlj)
    {
        int tmp = shli;
        shli = shlj;
        shlj = tmp;
    }
    if (shli >= pair->nbas)
        return NULL;
    int result = find_tree(shlj, &pair->search_tree[shli]);
    return &pair->shp[result];
}
