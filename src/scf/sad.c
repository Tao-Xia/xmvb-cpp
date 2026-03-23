#include "scf/hf.h"
//#include "output/output_func.h"
void do_hcore_InitalGuess(hf_info hf)
{
    Tensor3D c_matrix = hf->c_matrix;
    Tensor3D c_matrix_b = hf->c_matrix_b;
    Tensor3D d_matrix = hf->d_matrix;
    Tensor3D d_matrix_b = hf->d_matrix_b;
    Tensor3D h_matrix = hf->h_matrix;
    Matrix s_matrix = hf->s_matrix;
    Matrix x_matrix = hf->x_matrix;
    Matrix alpha_egn = hf->alpha_egn;
    int msize = hf->msize;
    for (int a = 0; a < hf->batch_size; a++)
    {
        cal_c_matrix(c_matrix[a], h_matrix[a], s_matrix, x_matrix, alpha_egn[a], msize);
        memcpy(c_matrix_b[a][0], c_matrix[a][0], sizeof(double) * msize * msize);
        memset(d_matrix[a][0], 0, sizeof(double) * msize * msize);
        memset(d_matrix_b[a][0], 0, sizeof(double) * msize * msize);
        for (int i = 0; i < msize; i++)
            for (int j = 0; j < msize; j++)
            {
                for (int k = 0; k < hf->alpha_seg[a]; k++)
                    d_matrix[a][i][j] += c_matrix[a][k][i] * c_matrix[a][k][j];
                for (int k = 0; k < hf->beta_seg[a]; k++)
                    d_matrix_b[a][i][j] += c_matrix_b[a][k][i] * c_matrix_b[a][k][j];
            }
    }
    return;
}

static Matrix cal_atomic_dm(int *msize, int ele_id, const char *bas_fname, const char *aux_fname, const char *kgrids_fname)
{
    static int spin_table[] = {0,
                               1, 0,
                               1, 0, 1, 2, 3, 2, 1, 0,
                               1, 0, 1, 2, 3, 2, 1, 0,
                               1, 0, 1, 2, 3, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0,
                               1, 0, 1, 2, 5, 6, 5, 4, 3, 0, 1, 0, 1, 2, 3, 2, 1, 0,
                               1, 0, 1, 0, 3, 4, 5, 6, 7, 8, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0};
    atm_info atm = (atm_info)malloc(sizeof(struct AtmInfo));
    atm->charge = 0;
    atm->mult = spin_table[ele_id] + 1;
    atm->value = malloc_vector(3);
    atm->atm = malloc_list(2);
    atm(ELEMENT_VAL, 0) = ele_id;
    atm(CENTER_IND, 0) = 0;
    atm->natm = 1;
    atm->mol_fname = NULL;
    ele_id += (atm->mult - 1);
    atm->alpha_num = ele_id / 2;
    ele_id -= (atm->mult - 1);
    atm->beta_num = ele_id - atm->alpha_num;
    bas_info bas = init_bas(bas_fname, atm);
    mol_info mol = (mol_info)malloc(sizeof(struct MolInfo));
    Matrix d_matrix = malloc_matrix(bas->msize, bas->msize);
    *msize = bas->msize;
    mol->prm = NULL;
    mol->prm_pair = NULL;
    mol->atm = atm;
    mol->bas = bas;
    mol->pair = init_pair(bas);
    hf_info hf = init_hf(mol);
    load_coulomb_aux(aux_fname, hf->coulomb);
    load_exchange_grids(kgrids_fname, kgrids_fname, hf->exchange);
    set_coulomb_type(RI_jbuilder, hf->coulomb);
    set_exchange_type(COSX_kbuilder, hf->exchange);
    do_hcore_InitalGuess(hf);
//    printf("Hcore density matrix\n");
//    matrec(hf->d_matrix[0][0],hf->msize,hf->msize);
    if (ele_id==1)
        hf->ctrl.max_iter=0;
    do_uhf(0, hf);
    for (int i = 0; i < hf->msize; i++)
        for (int j = 0; j < hf->msize; j++)
            d_matrix[i][j] = (hf->d_matrix[0][i][j] + hf->d_matrix_b[0][i][j]) * 0.5;
    del_hf(hf);
    del_atm(atm);
    del_bas(bas);
    del_pair(mol->pair);
    free(mol);
    return d_matrix;
}
void do_sad_InitalGuess(hf_info hf)
{
    const mol_info mol = hf->mol;
    int natm = get_mol_natm(mol);
    int msize = hf->msize;
    int p_index = 0;
    hf_info atm_hf;
    Matrix d_matrix_list[118];
    Matrix d_matrix_tmp;
    int ele_num = 0;
    int msize_list[118], atm_list[118];
    for (int i = 0; i < natm; i++)
    {
        int if_in = 0;
        for (int j = 0; j < ele_num; j++)
        {
            if (atm_list[j] == hf->mol->atm(ELEMENT_VAL, i))
            {
                if_in = 1;
                break;
            }
        }
        if (!if_in)
        {
            atm_list[ele_num] = hf->mol->atm(ELEMENT_VAL, i);
            ele_num++;
        }
    }
    for (int i = 0; i < ele_num; i++)
    {
        d_matrix_list[i] = cal_atomic_dm(&msize_list[i], atm_list[i], mol->bas->bas_fname,
                                         hf->coulomb->aux->bas_fname, hf->exchange->grids_fname);
    }
    int seg_index = 1;
    int seg_natm = 0;
    if (hf->batch_size > 1)
        for (int i = 0; i < natm; i++)
        {
            int uu = 0;
            for (int j = 0; j < ele_num; j++, uu++)
                if (atm_list[j] == hf->mol->atm(ELEMENT_VAL, i))
                    break;
            d_matrix_tmp = d_matrix_list[uu];
            for (int a = 0; a < msize_list[uu]; a++)
                for (int b = 0; b <= a; b++)
                    hf->d_matrix[seg_index][p_index + a][p_index + b] = hf->d_matrix[0][p_index + b][p_index + a] =
                        hf->d_matrix_b[seg_index][p_index + a][p_index + b] =
                            hf->d_matrix_b[0][p_index + b][p_index + a] = hf->d_matrix[0][p_index + a][p_index + b] =
                                hf->d_matrix[0][p_index + b][p_index + a] = hf->d_matrix_b[0][p_index + a][p_index + b] =
                                    hf->d_matrix_b[0][p_index + b][p_index + a] = d_matrix_tmp[a][b];
            p_index += msize_list[uu];
            seg_natm++;
            if (seg_natm >= hf->atm_seg[seg_index])
            {
                seg_natm = 0;
                seg_index++;
            }
        }
    else
    {
        for (int i = 0; i < natm; i++)
        {
            int uu = 0;
            for (int j = 0; j < ele_num; j++, uu++)
                if (atm_list[j] == hf->mol->atm(ELEMENT_VAL, i))
                    break;
            d_matrix_tmp = d_matrix_list[uu];
            for (int a = 0; a < msize_list[uu]; a++)
                for (int b = 0; b <= a; b++)
                    hf->d_matrix[0][p_index + a][p_index + b] = hf->d_matrix[0][p_index + b][p_index + a] =
                        hf->d_matrix_b[0][p_index + a][p_index + b] = hf->d_matrix_b[0][p_index + b][p_index + a] = d_matrix_tmp[a][b];
            p_index += msize_list[uu];
        }
    }
    for (int i = 0; i < ele_num; i++)
        free_matrix(d_matrix_list[i]);
    for (int i = 0; i < hf->batch_size; i++)
    {
        memcpy(hf->c_matrix[i][0], hf->d_matrix[i][0], sizeof(double) * msize * msize);
        LAPACKE_dpotrf(101, 'U', msize, hf->c_matrix[i][0], msize);
        memcpy(hf->c_matrix_b[i][0], hf->c_matrix[i][0], sizeof(double) * msize * msize);
    }
    // printf("ELE_NUM: %12.6f\n", matrix_inner(hf->s_matrix, hf->d_matrix[0], msize));
    return;
}
void do_sap_InitalGuess(hf_info hf);
