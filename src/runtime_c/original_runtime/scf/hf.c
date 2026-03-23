#include "scf/hf.h"
#include "solver/opt_solver.h"
#define MAX_BATCH 256
hf_info init_hf(const mol_info mol)
{
    int batch_size = 1;
    int tmp_list[MAX_BATCH];
    char str_num[256] = {'\0'}, str_charge[256] = {'\0'}, str_mult[256] = {'\0'};
    char *tmp, *err;
    const atm_info atm = mol->atm;
    const bas_info bas = mol->bas;
    const char *mol_fname = atm->mol_fname;
    if (mol_fname != NULL)
    {
        FILE *mol_file = fopen(mol_fname, "r");
        err = fgets(str_num, sizeof(str_num), mol_file);
        err = fgets(str_charge, sizeof(str_charge), mol_file);
        err = fgets(str_mult, sizeof(str_mult), mol_file);
        fclose(mol_file);
    }
    else
    {
        goto ONE;
    }
    if (strlen(str_num) <= 0 || strlen(str_num) <= 0 || strlen(str_num) <= 0 || mol_fname == NULL)
        goto ONE;
    str_num[strlen(str_num) - 1] = '\0';
    str_charge[strlen(str_charge) - 1] = '\0';
    str_mult[strlen(str_mult) - 1] = '\0';
    tmp = strtok(str_num, " ");
    if (!(tmp != NULL && atoi(tmp) > 0))
    {
    ONE:;
        str_num[0] = get_mol_natm(mol);
        str_charge[0] = mol->atm->charge;
        str_mult[0] = mol->atm->mult;
        goto INIT;
    }
    tmp_list[1] = atoi(tmp);
    for (int i = 0; i < MAX_BATCH; i++, batch_size++)
    {
        tmp = strtok(NULL, " ");
        if (tmp == NULL)
            break;
        tmp_list[batch_size + 1] = atoi(tmp);
        if (tmp_list[batch_size + 1] <= 0)
            break;
    }
    batch_size++;
INIT:;
    hf_info hf = (hf_info)malloc(sizeof(struct HFInfo));
    int msize;
    hf->mol = mol;
    hf->batch_size = batch_size;
    hf->beta_seg = malloc_list(batch_size);
    hf->alpha_seg = malloc_list(batch_size);
    hf->mult_seg = malloc_list(batch_size);
    hf->charge_seg = malloc_list(batch_size);
    hf->atm_seg = malloc_list(batch_size);
    memcpy(hf->atm_seg, tmp_list, sizeof(int) * batch_size);
    if (batch_size > 1)
    {
        batch_size = 1;
        tmp = strtok(str_charge, " ");
        assert(tmp != NULL);
        hf->charge_seg[1] = atoi(tmp);
        for (int i = 0; i < hf->batch_size - 1; i++, batch_size++)
        {
            tmp = strtok(NULL, " ");
            if (tmp == NULL)
                break;
            hf->charge_seg[batch_size + 1] = atoi(tmp);
        }
        batch_size = 1;
        tmp = strtok(str_mult, " ");
        assert(tmp != NULL);
        hf->mult_seg[1] = fabs(atoi(tmp));
        for (int i = 1; i < hf->batch_size - 1; i++, batch_size++)
        {
            tmp = strtok(NULL, " ");
            if (tmp == NULL)
                break;
            hf->mult_seg[batch_size + 1] = fabs(atoi(tmp));
        }
        batch_size++;
    }
    hf->charge_seg[0] = atm->charge;
    hf->mult_seg[0] = atm->mult;
    msize = get_mol_msize(mol);
    hf->tol_energy = malloc_vector(batch_size);
    hf->e_j = malloc_vector(batch_size);
    hf->e_k = malloc_vector(batch_size);
    hf->e_t = malloc_vector(batch_size);
    hf->e_v = malloc_vector(batch_size);
    hf->e_xc = malloc_vector(batch_size);
    hf->nuc_rep = malloc_vector(batch_size);
    hf->c_matrix = malloc_tensor3d(batch_size, msize, msize);
    hf->c_matrix_b = malloc_tensor3d(batch_size, msize, msize);
    hf->d_matrix = malloc_tensor3d(batch_size, msize, msize);
    hf->d_matrix_b = malloc_tensor3d(batch_size, msize, msize);
    hf->s_matrix = malloc_matrix(msize, msize);
    hf->x_matrix = malloc_matrix(msize, msize);
    hf->s_half_matrix = malloc_matrix(msize, msize);
    hf->t_matrix = malloc_matrix(msize, msize);
    hf->v_matrix = malloc_tensor3d(batch_size, msize, msize);
    hf->h_matrix = malloc_tensor3d(batch_size, msize, msize);
    hf->alpha_egn = malloc_matrix(batch_size, msize);
    hf->beta_egn = malloc_matrix(batch_size, msize);
    cal_overlap_matrix(hf->s_matrix, hf->mol);
    cal_kinetic_matrix(hf->t_matrix, hf->mol);
    copy_matrix(hf->x_matrix, hf->s_matrix, msize, msize);
    cal_matrix_syhalfrev(hf->x_matrix, msize);
    copy_matrix(hf->s_half_matrix, hf->x_matrix, msize, msize);
    cal_symatrix_reverse(hf->s_half_matrix, msize);
    int natm = atm->natm, begin_atm = 0;
    List tmp_atm = malloc_list(2 * natm);
    memcpy(tmp_atm, atm->atm, sizeof(int) * 2 * natm);
    cal_nucleus_matrix(hf->v_matrix[0], hf->mol);
    for (int i = 0; i < msize * msize; i++)
        hf->h_matrix[0][0][i] = hf->t_matrix[0][i] + hf->v_matrix[0][0][i];
    hf->nuc_rep[0] = cal_nuc_rep(hf->mol);
    hf->alpha_seg[0] = atm->alpha_num;
    hf->beta_seg[0] = atm->beta_num;
    for (int n = 1; n < batch_size; n++)
    {
        int ele_num = 0;
        for (int i = 0; i < begin_atm; i++)
            atm(ELEMENT_VAL, i) = 0;
        for (int i = begin_atm; i < begin_atm + tmp_list[n]; i++)
            ele_num += atm(ELEMENT_VAL, i);
        for (int i = begin_atm + hf->atm_seg[n]; i < natm; i++)
            atm(ELEMENT_VAL, i) = 0;
        cal_nucleus_matrix(hf->v_matrix[n], hf->mol);
        begin_atm += tmp_list[n];
        hf->nuc_rep[n] = cal_nuc_rep(hf->mol);
        memcpy(atm->atm, tmp_atm, sizeof(int) * 2 * natm);
        ele_num += (hf->mult_seg[n] - 1);
        hf->alpha_seg[n] = ele_num / 2;
        ele_num -= (hf->mult_seg[n] - 1);
        hf->beta_seg[n] = ele_num - hf->alpha_seg[n];
        for (int i = 0; i < msize * msize; i++)
            hf->h_matrix[n][0][i] = hf->t_matrix[0][i] + hf->v_matrix[n][0][i];
    }
    free(tmp_atm);
    hf->f_matrix = malloc_tensor3d(batch_size, msize, msize);
    hf->f_matrix_b = malloc_tensor3d(batch_size, msize, msize);
    hf->exchange = init_exchange(batch_size, mol);
    hf->coulomb = init_coulomb(batch_size, mol);
    hf->S2 = malloc_vector(batch_size);
    hf->dft = NULL;
    hf->j_matrix = hf->coulomb->J;
    hf->k_matrix = hf->exchange->K;
    hf->k_matrix_b = hf->exchange->K_b;
    hf->xc_matrix = NULL;
    hf->xc_matrix_b = NULL;
    hf->ext_p = NULL;
    hf->ctrl.max_delta_d = 1E-2;
    hf->ctrl.max_delta_e = 1E-6;
    hf->ctrl.max_fds_err = 6;
    hf->ctrl.max_iter = 100;
    hf->diis_size = 6;
    hf->diis_tr1 = 1E-1;
    hf->diis_tr2 = 1E-4;
    hf->open_type = UHF_WORK;
    hf->msize = get_mol_msize(mol);
    hf->if_aim_def = false;
    hf->if_sub_def = false;
    hf->if_sub_use = false;
    hf->jaux_fname = NULL;
    hf->kaux_fname = NULL;
    hf->jgrids_fname = NULL;
    hf->kgrids_fname = NULL;
    hf->jfgrids_fname = NULL;
    hf->kfgrids_fname = NULL;
    hf->background_charge = calloc(batch_size, sizeof(int) + sizeof(Matrix) + 5 * sizeof(Vector));
    return hf;
}

void set_hf_diis(int diis_size, int tr1, int tr2, hf_info hf)
{
    assert(tr1 > tr2);
    hf->diis_size = diis_size;
    hf->diis_tr1 = tr1;
    hf->diis_tr2 = tr2;
    return;
}

void set_hf_ctrl(int max_iter, double max_delta_e, double max_delta_d, double max_fds_err, hf_info hf)
{
    hf->ctrl.max_delta_d = max_delta_d;
    hf->ctrl.max_delta_e = max_delta_e;
    hf->ctrl.max_fds_err = max_fds_err;
    hf->ctrl.max_iter = max_iter;
    return;
}
void load_hf_dft(double hf_frac, GRIDS_TYPE gtype, const int dft_id[], const double dft_frac[], int num, int disp_type, int dft_name, hf_info hf)
{
    int batch_size = hf->batch_size;
    int msize = hf->msize;
    hf->xc_matrix = malloc_tensor3d(batch_size, msize, msize);
    hf->xc_matrix_b = malloc_tensor3d(batch_size, msize, msize);
    hf->dft = init_dft(hf_frac, gtype, dft_id, dft_frac, num, disp_type, dft_name, hf->mol);
    return;
}
void del_hf(hf_info hf)
{
    del_coulomb(hf->coulomb);
    del_exchange(hf->exchange);
    free_tensor3d(hf->c_matrix);
    free_tensor3d(hf->c_matrix_b);
    free_tensor3d(hf->d_matrix);
    free_tensor3d(hf->d_matrix_b);
    free_matrix(hf->s_matrix);
    free_matrix(hf->x_matrix);
    free_matrix(hf->t_matrix);
    free_tensor3d(hf->v_matrix);
    free_tensor3d(hf->h_matrix);
    free_matrix(hf->s_half_matrix);
    free_matrix(hf->alpha_egn);
    free_matrix(hf->beta_egn);
    free_vector(hf->e_j);
    free_vector(hf->e_k);
    free_vector(hf->e_t);
    free_vector(hf->e_v);
    free_vector(hf->e_xc);
    free_vector(hf->tol_energy);
    free_vector(hf->nuc_rep);
    free_vector(hf->S2);
    free(hf->alpha_seg);
    free(hf->beta_seg);
    free(hf->atm_seg);
    free(hf->mult_seg);
    free(hf->charge_seg);
    free_tensor3d(hf->f_matrix);
    free_tensor3d(hf->f_matrix_b);
    if (hf->dft != NULL)
    {
        del_dft(hf->dft);
        free_tensor3d(hf->xc_matrix);
        free_tensor3d(hf->xc_matrix_b);
    }
    del_hf_sub(hf);
    del_hf_aim(hf);
    free(hf->jaux_fname);
    free(hf->kaux_fname);
    free(hf->jgrids_fname);
    free(hf->kgrids_fname);
    free(hf->jfgrids_fname);
    free(hf->kfgrids_fname);
    for (int seg_id = 0; seg_id < hf->batch_size; seg_id++)
    if (hf->background_charge[seg_id].charge_num != 0)
    {
        free_matrix(hf->background_charge[seg_id].ev_matrix);
        free_vector(hf->background_charge[seg_id].charge_value);
        free_vector(hf->background_charge[seg_id].ev_grad);
        free_vector(hf->background_charge[seg_id].e_ev);
        free_vector(hf->background_charge[seg_id].e_nv);
        free_vector(hf->background_charge[seg_id].e_tolv);
    }
    free(hf->background_charge);
    free(hf);
    return;
}
void set_hf_coulomb(Jbuilder_type jtype, hf_info hf)
{
    hf->coulomb->type = jtype;
    return;
}
void set_hf_exchange(Kbuilder_type ktype, hf_info hf)
{
    hf->exchange->type = ktype;
    return;
}
void load_hf_jaux(const char *aux_fname, hf_info hf)
{
    hf->jaux_fname = (char *)malloc(sizeof(char) * 1024);
    memset(hf->jaux_fname, 0, sizeof(char) * 1024);
    memcpy(hf->jaux_fname, aux_fname, sizeof(char) * strlen(aux_fname));
    load_coulomb_aux(aux_fname, hf->coulomb);
    return;
}
void load_hf_kaux(const char *aux_fname, hf_info hf)
{
    return;
}
void load_hf_jgrids(const char *jgrids_fname, const char *jfgrids_fname, hf_info hf)
{
    hf->jgrids_fname = (char *)malloc(sizeof(char) * 1024);
    memset(hf->jgrids_fname, 0, sizeof(char) * 1024);
    memcpy(hf->jgrids_fname, jgrids_fname, sizeof(char) * strlen(jgrids_fname));
    if (jfgrids_fname != NULL)
    {
        hf->jfgrids_fname = (char *)malloc(sizeof(char) * 1024);
        memset(hf->jfgrids_fname, 0, sizeof(char) * 1024);
        memcpy(hf->jfgrids_fname, jfgrids_fname, sizeof(char) * strlen(jfgrids_fname));
    }
    load_coulomb_grids(jgrids_fname, jfgrids_fname, hf->coulomb);
    return;
}
void load_hf_kgrids(const char *kgrids_fname, const char *kfgrids_fname, hf_info hf)
{
    hf->kgrids_fname = (char *)malloc(sizeof(char) * 1024);
    memset(hf->kgrids_fname, 0, sizeof(char) * 1024);
    memcpy(hf->kgrids_fname, kgrids_fname, sizeof(char) * strlen(kgrids_fname));
    if (kfgrids_fname != NULL)
    {
        hf->kfgrids_fname = (char *)malloc(sizeof(char) * 1024);
        memset(hf->kfgrids_fname, 0, sizeof(char) * 1024);
        memcpy(hf->kfgrids_fname, kfgrids_fname, sizeof(char) * strlen(kfgrids_fname));
    }
    load_exchange_grids(kgrids_fname, kfgrids_fname, hf->exchange);
    return;
}

void cal_c_matrix(Matrix c_matrix, const Matrix f_matrix, const Matrix s_matrix, const Matrix x_matrix, Vector egn_value, int msize)
{
    Matrix f_matrix_tmp = malloc_matrix(msize, msize);
    int erro;
    memcpy(f_matrix_tmp[0], f_matrix[0], sizeof(double) * msize * msize);
    trafock(f_matrix_tmp[0], x_matrix[0], msize);
    LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', msize, f_matrix_tmp[0], msize, egn_value);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, msize, msize, msize, 1.0, x_matrix[0], msize, f_matrix_tmp[0], msize, 0.0, c_matrix[0], msize);
    free_matrix(f_matrix_tmp);
    return;
}
static void print_hf_step(const int loop_num, const Vector delta_e, const Vector delta_d, const Vector fds_err, const double time, const hf_info hf, const int print_level)
{
    if (print_level == 0)
        return;
    if (loop_num == 1 && print_level < 4)
    {
        printf("=========================================================================================================\n");
        printf(" Iter           Energy                Delta_Energy             Delta_Density      [F,D]          Time\n");
        printf("=========================================================================================================\n");
    }
    if (print_level >= 1)
        for (int i = 0; i < hf->batch_size; i++)
            if (i == 0)
                printf("  %-3d %20.10f        %20.10f %20.10f    %12.6f %12.6f\n", loop_num, hf->tol_energy[i], delta_e[i], delta_d[i], fds_err[i], time);
            else
                printf("      %20.10f        %20.10f %20.10f    %12.6f \n", hf->tol_energy[i], delta_e[i], delta_d[i], fds_err[i]);
    printf("---------------------------------------------------------------------------------------------------------\n");
    fflush(stdout);
    return;
}
void set_hf_sub(bool if_sub_use, hf_info hf)
{
    hf->if_sub_use = if_sub_use;
}

void set_hf_mu(double aim_mu, hf_info hf)
{
    assert(aim_mu >= 0.);
    hf->aim_mu = aim_mu;
    return;
}
double cal_rhf_energy(const Matrix d_matrix, const Matrix h_matrix, const Matrix f_matrix, const int msize)
{
    double energy = 0.;
    energy += matrix_inner(d_matrix, h_matrix, msize);
    energy += matrix_inner(d_matrix, f_matrix, msize);
    return energy;
}
double cal_uhf_energy(const Matrix d_matrix, const Matrix d_matrix_b, const Matrix h_matrix,
                      const Matrix f_matrix, const Matrix f_matrix_b, const int msize)
{
    double energy = 0.;
    energy += matrix_inner(d_matrix, h_matrix, msize);
    energy += matrix_inner(d_matrix, f_matrix, msize);
    energy += matrix_inner(d_matrix_b, h_matrix, msize);
    energy += matrix_inner(d_matrix_b, f_matrix_b, msize);
    return energy * 0.5;
}

double cal_hf_energy(const Matrix d_matrix, const Matrix d_matrix_b, const Matrix h_matrix, const Matrix f_matrix, const Matrix f_matrix_b, const int msize)
{
    if (d_matrix_b == NULL || f_matrix_b == NULL)
        return cal_rhf_energy(d_matrix, h_matrix, f_matrix, msize);
    return cal_uhf_energy(d_matrix, d_matrix_b, h_matrix, f_matrix, f_matrix_b, msize);
}
// transfer UHF Fock matrix to the ROHF one
static void rofock(Matrix f_matrix, Matrix f_matrix_b, const Matrix c_matrix, const Matrix c_matrix_b, int alpha_orb, int beta_orb, int msize)
{
    /**
     *                           Acc  Bcc  Aoo Boo  Avv  Bvv
     *  Guest and Saunders       1/2  1/2  1/2 1/2  1/2  1/2
     *  Roothaan single matrix  -1/2  3/2  1/2 1/2  3/2 -1/2
     *  Davidson/1988            1/2  1/2   1   0    1    0
     *  Binkley, Pople, Dobosh   1/2  1/2   1   0    0    1
     *  McWeeny and Diercksen    1/3  2/3  1/3 1/3  2/3  1/3
     *  Faegri and Manne         1/2  1/2   1   0   1/2  1/2
     * copy from GAMESS: http://myweb.liu.edu/~nmatsuna/gamess/refs/how.to.scf.html
     */
    int i, j;
    int io, jo;
    Matrix f_matrix_mo = malloc_matrix(msize, msize);
    Matrix f_matrix_b_mo = malloc_matrix(msize, msize);
    memcpy(f_matrix_mo[0], f_matrix[0], sizeof(double) * msize * msize);
    memcpy(f_matrix_b_mo[0], f_matrix_b[0], sizeof(double) * msize * msize);
    trafock(f_matrix_mo[0], c_matrix[0], msize);
    trafock(f_matrix_b_mo[0], c_matrix_b[0], msize);
    for (int i = 0; i < msize; i++)
    {
        if (i < beta_orb)
            io = 1;
        else if (i < alpha_orb)
            io = 2;
        else
            io = 3;
        for (int j = 0; j <= i; j++)
        {
            if (j < beta_orb)
                jo = 1;
            else if (j < alpha_orb)
                jo = 2;
            else
                jo = 3;
            if (io == jo)
            {
                f_matrix[i][j] = 0.5 * (f_matrix_mo[i][j] + f_matrix_b_mo[i][j]);
            }
            else if (io + jo == 3)
                f_matrix[i][j] = f_matrix_b_mo[i][j] * 0.5;
            else if (io + jo == 4)
                f_matrix[i][j] = 0.5 * (f_matrix_mo[i][j] + f_matrix_b_mo[i][j]);
            else if (io + jo == 5)
                f_matrix[i][j] = f_matrix_mo[i][j] * 0.5;
            f_matrix[j][i] = f_matrix[i][j];
        }
    }
    memcpy(f_matrix_mo[0], c_matrix[0], sizeof(double) * msize * msize);
    cal_matrix_reverse(f_matrix_mo, msize);
    trafock(f_matrix[0], f_matrix_mo[0], msize);
    free_matrix(f_matrix_mo);
    free_matrix(f_matrix_b_mo);
    return;
}

void cufock(Matrix f_matrix, Matrix f_matrix_b, const Matrix d_matrix, const Matrix d_matrix_b, const Matrix s_half_matrix, const Matrix x_matrix, int alpha_orb, int beta_orb, int msize)
{
    Matrix c_matrix_no = malloc_matrix(msize, msize);
    Matrix p_matrix = malloc_matrix(msize, msize);
    Matrix delta_matrix = malloc_matrix(msize, msize);
    Matrix lambda_matrix = malloc_matrix(msize, msize);
    Vector occ_num = malloc_vector(msize);
    // averge denstiy matrix
    for (int i = 0; i < msize; i++)
    {
        for (int j = 0; j < msize; j++)
        {
            p_matrix[i][j] = 0.5 * (d_matrix[i][j] + d_matrix_b[i][j]);
            delta_matrix[i][j] = 0.5 * (f_matrix[i][j] - f_matrix_b[i][j]);
        }
    }
    /// get NO coe
    cal_no_orb(c_matrix_no, occ_num, s_half_matrix, x_matrix, p_matrix, msize);
    /// transfer basise set to NO
    trafock(delta_matrix[0], c_matrix_no[0], msize);
    /// lambda matrix (spin polution)
    for (int i = 0; i < alpha_orb - 1; i++)
    {
        for (int j = alpha_orb + 1; j < msize; j++)
        {
            lambda_matrix[i][j] = -delta_matrix[i][j];
            lambda_matrix[j][i] = lambda_matrix[i][j];
        }
    }
    cal_matrix_reverse(c_matrix_no, msize);
    /// transfer basise set to AO
    trafock(lambda_matrix[0], c_matrix_no[0], msize);
    for (int i = 0; i < msize; i++)
    {
        for (int j = 0; j < msize; j++)
        {
            f_matrix[i][j] = f_matrix[i][j] + lambda_matrix[i][j];
            f_matrix_b[i][j] = f_matrix_b[i][j] - lambda_matrix[i][j];
        }
    }
    free_matrix(delta_matrix);
    free_matrix(lambda_matrix);
    free_matrix(p_matrix);
    free_vector(occ_num);
    free_matrix(c_matrix_no);
}

void print_uhf(const int loop_num, const Vector delta_e, const Vector delta_d, const Vector fds_err, const double time, const hf_info hf, const int print_level)
{
    if (print_level == 0)
        return;
    if (loop_num == 1 && print_level < 4)
    {
        printf("=========================================================================================================\n");
        printf(" Iter           Energy                Delta_Energy             Delta_Density      [F,D]          Time\n");
        printf("=========================================================================================================\n");
    }
    if (print_level >= 1)
        for (int i = 0; i < hf->batch_size; i++)
            if (i == 0)
                printf("  %-3d %20.10f        %20.10f %20.10f    %12.6f %12.6f\n", loop_num, hf->tol_energy[i], delta_e[i], delta_d[i], fds_err[i], time);
            else
                printf("      %20.10f        %20.10f %20.10f    %12.6f \n", hf->tol_energy[i], delta_e[i], delta_d[i], fds_err[i]);
    printf("---------------------------------------------------------------------------------------------------------\n");
    fflush(stdout);
    return;
}

void do_rhf(int print_level, hf_info hf)
{
    if (hf->if_sub_use)
        assert(hf->if_sub_def);
    double t_scf0 = omp_get_wtime();
    int batch_size = hf->batch_size;
    double t_J = 0., t_K = 0.;
    double step_t1, step_t2;
    double star_t, end_t;
    int loop_num = 0;
    diis_info *diis = (diis_info *)malloc(sizeof(diis_info) * batch_size);
    Vector fds_err, delta_e, delta_d;
    Vector energy_tmp;
    int msize = get_mol_msize(hf->mol);
    mol_info mol = hf->mol;
    Tensor3D d_matrix_cp = malloc_tensor3d(batch_size, msize, msize);
    Tensor3D d_matrix = hf->d_matrix;
    Tensor3D d_matrix_b = hf->d_matrix_b;
    Tensor3D c_matrix = hf->c_matrix;
    Tensor3D d_matrix_delta = malloc_tensor3d(batch_size, msize, msize);
    Tensor3D d_matrix_tol_delta = malloc_tensor3d(batch_size, msize, msize);
    Tensor3D f_matrix = hf->f_matrix;
    Tensor3D h_matrix = hf->h_matrix;
    Matrix x_matrix = hf->x_matrix;
    Matrix s_matrix = hf->s_matrix;
    Tensor3D j_matrix = hf->j_matrix;
    Tensor3D k_matrix = hf->k_matrix;
    Tensor3D xc_matrix = hf->xc_matrix;
    // Matrix and parameters of subspace calculation
    int sub_size = hf->sub_size;
    Matrix subspace = hf->subspace;
    Matrix s_matrix_sub = hf->s_matrix_sub;
    Matrix x_matrix_sub = hf->x_matrix_sub;
    Matrix *f_matrix_sub = hf->f_matrix_sub;
    Matrix *c_matrix_sub = hf->c_matrix_sub;
    Matrix *d_matrix_sub = hf->d_matrix_sub;
    bool *if_finnish = (bool *)malloc(sizeof(bool) * batch_size);
    double hf_frac = 1.;
    if (hf->dft != NULL)
        hf_frac = hf->dft->hf_frac;
    star_t = omp_get_wtime();
    fds_err = malloc_vector(batch_size);
    delta_d = malloc_vector(batch_size);
    delta_e = malloc_vector(batch_size);
    energy_tmp = malloc_vector(batch_size);
    for (int i = 0; i < batch_size; i++)
    {
        if_finnish[i] = false;
        if (!hf->if_sub_use)
            diis[i] = init_diis(hf->diis_size, msize);
        else
            diis[i] = init_diis(hf->diis_size, sub_size);
        set_diis_tr(hf->diis_tr1, hf->diis_tr2, diis[i]);
    }
    memcpy(d_matrix_delta[0][0], d_matrix[0][0], batch_size * sizeof(double) * msize * msize);
    for (int a = 0; a < batch_size; a++)
        for (int i = 0; i < msize * msize; i++)
            d_matrix_tol_delta[a][0][i] = d_matrix_delta[a][0][i] * 2;
    memset(j_matrix[0][0], 0, batch_size * sizeof(double) * msize * msize);
    memset(k_matrix[0][0], 0, batch_size * sizeof(double) * msize * msize);
    set_exchange_final(false, hf->exchange);
    while (loop_num >= 0)
    {
        step_t1 = omp_get_wtime();
        loop_num++;
        switch_xscf_thread(0);
//       openblas_set_num_threads(1);
        cal_j_matrix_batch(if_finnish, d_matrix_tol_delta, hf->coulomb);
        cal_k_matrix_batch(if_finnish, d_matrix_delta, NULL, hf->exchange);
        for (int i = 0; i < msize * msize * batch_size; i++)
            f_matrix[0][0][i] = h_matrix[0][0][i] - k_matrix[0][0][i] * hf_frac + j_matrix[0][0][i];
        // if subspace calculation is done, the Fock matrix will be projected into the subspace
        for (int a = 0; a < batch_size; a++)
        {
            if (if_finnish[a] == 1)
                continue;
            hf->tol_energy[a] = cal_rhf_energy(d_matrix[a], h_matrix[a], f_matrix[a], msize) + hf->nuc_rep[a];
            if (hf->dft != NULL)
                hf->e_xc[a] = cal_dft_xc(d_matrix[a], NULL, xc_matrix[a], NULL, hf->dft);
            else
                hf->e_xc[a] = 0.;
            if (hf->xc_matrix != NULL)
                for (int i = 0; i < msize * msize; i++)
                    f_matrix[a][0][i] += xc_matrix[a][0][i];
            hf->tol_energy[a] += hf->e_xc[a];
            hf->e_k[a] = -matrix_inner(d_matrix[a], k_matrix[a], msize) * hf_frac;
            hf->e_j[a] = matrix_inner(d_matrix[a], j_matrix[a], msize);
            hf->e_v[a] = 2 * matrix_inner(hf->v_matrix[a], d_matrix[a], msize);
            hf->e_t[a] = 2 * matrix_inner(hf->t_matrix, d_matrix[a], msize);
            delta_e[a] = hf->tol_energy[a] - energy_tmp[a];
            energy_tmp[a] = hf->tol_energy[a];
        }
        // the background charge
//        openblas_set_num_threads(get_xscf_threadnum());
        switch_xscf_thread(get_xscf_threadnum());
        for (int seg_id = 0; seg_id < hf->batch_size; seg_id++)
        {
            if (hf->background_charge[seg_id].charge_num != 0)
            {
                for (int a = 0; a < batch_size; a++)
                {
                    for (int i = 0; i < msize * msize; i++)
                        f_matrix[a][0][i] += hf->background_charge[seg_id].ev_matrix[0][i];
                    hf->background_charge[seg_id].e_ev[a] = matrix_inner(d_matrix[a], hf->background_charge[seg_id].ev_matrix, msize) * 2;
                    hf->background_charge[seg_id].e_tolv[a] = hf->background_charge[seg_id].e_ev[a] + hf->background_charge[seg_id].e_nv[a];
                }
            }
        }
        double t_sf = omp_get_wtime();
        if (loop_num >= hf->ctrl.max_iter)
            break;
        for (int a = 0; a < batch_size; a++)
        {
            if (if_finnish[a] == 1)
                continue;
            if (hf->if_sub_use)
                gtrafock(f_matrix_sub[a], subspace, subspace, f_matrix[a], sub_size, sub_size, msize);
            if (loop_num > 1)
                if (!hf->if_sub_use)
                    fds_err[a] = do_diis(s_matrix, f_matrix[a], d_matrix[a], diis[a]);
                else
                    fds_err[a] = do_diis(s_matrix_sub, f_matrix_sub[a], d_matrix_sub[a], diis[a]);
            if (!hf->if_sub_use)
                cal_c_matrix(c_matrix[a], f_matrix[a], s_matrix, x_matrix, hf->alpha_egn[a], msize);
            // if subspace calculation is done, then a reduced eigenvalue problem will be solved
            else
            {
                cal_c_matrix(c_matrix_sub[a], f_matrix_sub[a], s_matrix_sub, x_matrix_sub, hf->alpha_egn[a], sub_size);
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, sub_size, msize, sub_size, 1., c_matrix_sub[a][0], sub_size,
                            subspace[0], msize, 0., c_matrix[a][0], msize);
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, sub_size, sub_size, hf->alpha_seg[a], 1., c_matrix_sub[a][0],
                            sub_size, c_matrix_sub[a][0], sub_size, 0, d_matrix_sub[a][0], sub_size);
            }
            memcpy(d_matrix_cp[a][0], d_matrix[a][0], sizeof(double) * msize * msize);
            memset(d_matrix[a][0], 0, sizeof(double) * msize * msize);
            for (int i = 0; i < msize; i++)
                for (int j = 0; j < msize; j++)
                    for (int k = 0; k < hf->alpha_seg[a]; k++)
                        d_matrix[a][i][j] += c_matrix[a][k][i] * c_matrix[a][k][j];
            for (int i = 0; i < msize * msize; i++)
            {
                d_matrix_delta[a][0][i] = d_matrix[a][0][i] - d_matrix_cp[a][0][i];
                d_matrix_tol_delta[a][0][i] = d_matrix_delta[a][0][i] * 2;
            }
            delta_d[a] = 0.;
            for (int i = 0; i < msize * msize; i++)
                if (fabs(d_matrix_delta[a][0][i]) > delta_d[a])
                    delta_d[a] = fabs(d_matrix_delta[a][0][i]);
            if (fabs(delta_d[a]) <= hf->ctrl.max_delta_d && fabs(delta_e[a]) <= hf->ctrl.max_delta_e && fds_err[a] >= hf->ctrl.max_fds_err)
                if_finnish[a] = 1;
        }
        step_t2 = omp_get_wtime();
        print_uhf(loop_num, delta_e, delta_d, fds_err, step_t2 - step_t1, hf, print_level);
        int if_break = 1;
        for (int a = 0; a < batch_size; a++)
            if (if_finnish[a] == 0)
            {
                if_break = 0;
                break;
            }
        if (if_break)
            break;
    }
    if (hf->exchange->grids_final != NULL)
    {
//        openblas_set_num_threads(1);
        switch_xscf_thread(0);
        set_exchange_final(true, hf->exchange);
        for (int i = 0; i < batch_size; i++)
            if_finnish[i] = false;
        memset(k_matrix[0][0], 0, sizeof(double) * batch_size * msize * msize);
        memset(j_matrix[0][0], 0, sizeof(double) * batch_size * msize * msize);
        cal_j_matrix_batch(if_finnish, d_matrix, hf->coulomb);
        cal_k_matrix_batch(if_finnish, d_matrix, NULL, hf->exchange);
        for (int i = 0; i < batch_size * msize * msize; i++)
        {
            j_matrix[0][0][i] *= 2;
            f_matrix[0][0][i] = h_matrix[0][0][i] - k_matrix[0][0][i] * hf_frac + j_matrix[0][0][i];
        }
        for (int i = 0; i < batch_size; i++)
        {
            hf->e_k[i] = -matrix_inner(d_matrix[i], k_matrix[i], msize) * hf_frac;
            hf->e_j[i] = matrix_inner(d_matrix[i], j_matrix[i], msize);
            hf->e_v[i] = 2 * matrix_inner(hf->v_matrix[i], d_matrix[i], msize);
            hf->e_t[i] = 2 * matrix_inner(hf->t_matrix, d_matrix[i], msize);
            if (hf->dft != NULL)
                hf->e_xc[i] = cal_dft_exc(d_matrix[i], NULL, hf->dft);
            hf->tol_energy[i] = cal_rhf_energy(d_matrix[i], h_matrix[i], f_matrix[i], msize) + hf->nuc_rep[i] + hf->e_xc[i];
        }
    }
    free_tensor3d(d_matrix_cp);
    free_tensor3d(d_matrix_delta);
    free_tensor3d(d_matrix_tol_delta);
    for (int a = 0; a < batch_size; a++)
        del_diis(diis[a]);
    free(diis);
    end_t = omp_get_wtime();
//    for (int a = 0; a < batch_size; a++)
//        printf("E_tol = %12.6f, E_x = %12.6f, E_ne = %12.6f, T = %12.6f, E_ele = %12.6f, E_xc = %12.6f\n", hf->tol_energy[a], hf->e_k[a], hf->e_v[a], hf->e_t[a], hf->e_j[a], hf->e_xc[a]);
    memcpy(hf->d_matrix_b[0][0], d_matrix[0][0], sizeof(double) * batch_size * msize * msize);
    memcpy(hf->k_matrix_b[0][0], k_matrix[0][0], sizeof(double) * batch_size * msize * msize);
    for (int a = 0; a < batch_size; a++)
        hf->S2[a] = 1.;

    free(if_finnish);
    free_vector(fds_err);
    free_vector(delta_d);
    free_vector(delta_e);
    free_vector(energy_tmp);

//    printf("Time of SCF is %12.6f s.\n", omp_get_wtime() - t_scf0);
    switch_xscf_thread(0);
    return;
}

void do_uhf(int print_level, hf_info hf)
{
    int batch_size = hf->batch_size;
    double step_t1, step_t2;
    double star_t, end_t;
    int loop_num = 0;
    diis_info *diis = (diis_info *)malloc(sizeof(diis_info) * batch_size);
    diis_info *diis_b = (diis_info *)malloc(sizeof(diis_info) * batch_size);
    Vector fds_err, delta_e, delta_d;
    Vector energy_tmp;
    int msize = get_mol_msize(hf->mol);
    mol_info mol = hf->mol;
    Tensor3D d_matrix_cp = malloc_tensor3d(batch_size, msize, msize);
    Tensor3D d_matrix_b_cp = malloc_tensor3d(batch_size, msize, msize);
    Tensor3D d_matrix = hf->d_matrix;
    Tensor3D d_matrix_b = hf->d_matrix_b;
    Tensor3D c_matrix = hf->c_matrix;
    Tensor3D c_matrix_b = hf->c_matrix_b;
    Tensor3D d_matrix_delta = malloc_tensor3d(batch_size, msize, msize);
    Tensor3D d_matrix_b_delta = malloc_tensor3d(batch_size, msize, msize);
    Tensor3D d_matrix_tol_delta = malloc_tensor3d(batch_size, msize, msize);
    Tensor3D f_matrix = hf->f_matrix;
    Tensor3D f_matrix_b = hf->f_matrix_b;
    Tensor3D h_matrix = hf->h_matrix;
    Matrix x_matrix = hf->x_matrix;
    Matrix s_matrix = hf->s_matrix;
    Tensor3D j_matrix = hf->j_matrix;
    Tensor3D k_matrix = hf->k_matrix;
    Tensor3D k_matrix_b = hf->k_matrix_b;
    Tensor3D xc_matrix = hf->xc_matrix;
    Tensor3D xc_matrix_b = hf->xc_matrix_b;
    // Matrix and parameters of subspace calculation
    int sub_size = hf->sub_size;
    Matrix subspace = hf->subspace;
    Matrix s_matrix_sub = hf->s_matrix_sub;
    Matrix x_matrix_sub = hf->x_matrix_sub;
    Matrix *f_matrix_sub = hf->f_matrix_sub;
    Matrix *c_matrix_sub = hf->c_matrix_sub;
    Matrix *d_matrix_sub = hf->d_matrix_sub;
    Matrix *f_matrix_b_sub = hf->f_matrix_b_sub;
    Matrix *c_matrix_b_sub = hf->c_matrix_b_sub;
    Matrix *d_matrix_b_sub = hf->d_matrix_b_sub;
    bool *if_finnish = (bool *)malloc(batch_size * sizeof(bool));
    double hf_frac = 1.;
    if (hf->dft != NULL)
        hf_frac = hf->dft->hf_frac;
    star_t = omp_get_wtime();
    fds_err = malloc_vector(batch_size);
    delta_d = malloc_vector(batch_size);
    delta_e = malloc_vector(batch_size);
    energy_tmp = malloc_vector(batch_size);
    for (int i = 0; i < batch_size; i++)
    {
        if_finnish[i] = false;
        if (!hf->if_sub_use)
        {
            diis[i] = init_diis(hf->diis_size, msize);
            diis_b[i] = init_diis(hf->diis_size, msize);
        }
        else
        {
            diis[i] = init_diis(hf->diis_size, sub_size);
            diis_b[i] = init_diis(hf->diis_size, sub_size);
        }
        set_diis_tr(hf->diis_tr1, hf->diis_tr2, diis[i]);
        set_diis_tr(hf->diis_tr1, hf->diis_tr2, diis_b[i]);
    }
    memcpy(d_matrix_delta[0][0], d_matrix[0][0], batch_size * sizeof(double) * msize * msize);
    memcpy(d_matrix_b_delta[0][0], d_matrix_b[0][0], batch_size * sizeof(double) * msize * msize);
    for (int a = 0; a < batch_size; a++)
        for (int i = 0; i < msize * msize; i++)
            d_matrix_tol_delta[a][0][i] = d_matrix_delta[a][0][i] + d_matrix_b_delta[a][0][i];
    memset(j_matrix[0][0], 0, batch_size * sizeof(double) * msize * msize);
    memset(k_matrix[0][0], 0, batch_size * sizeof(double) * msize * msize);
    memset(k_matrix_b[0][0], 0, batch_size * sizeof(double) * msize * msize);
    set_exchange_final(false, hf->exchange);
    while (loop_num >= 0)
    {
        step_t1 = omp_get_wtime();
        loop_num++;
//        openblas_set_num_threads(1);
        switch_xscf_thread(0);
        cal_j_matrix_batch(if_finnish, d_matrix_tol_delta, hf->coulomb);
        cal_k_matrix_batch(if_finnish, d_matrix_delta, d_matrix_b_delta, hf->exchange);
        for (int i = 0; i < msize * msize * batch_size; i++)
        {
            f_matrix[0][0][i] = h_matrix[0][0][i] - k_matrix[0][0][i] * hf_frac + j_matrix[0][0][i];
            f_matrix_b[0][0][i] = h_matrix[0][0][i] - k_matrix_b[0][0][i] * hf_frac + j_matrix[0][0][i];
        }
        for (int a = 0; a < batch_size; a++)
        {
            if (if_finnish[a] == 1)
                continue;
            hf->tol_energy[a] = cal_uhf_energy(d_matrix[a], d_matrix_b[a], h_matrix[a], f_matrix[a], f_matrix_b[a], msize) + hf->nuc_rep[a];
            if (hf->dft != NULL)
            {
                hf->e_xc[a] = cal_dft_xc(d_matrix[a], d_matrix_b[a], xc_matrix[a], xc_matrix_b[a], hf->dft);
                for (int i = 0; i < msize * msize; i++)
                {
                    f_matrix[a][0][i] += xc_matrix[a][0][i];
                    f_matrix_b[a][0][i] += xc_matrix_b[a][0][i];
                }
            }
            else
                hf->e_xc[a] = 0.;
            hf->tol_energy[a] += hf->e_xc[a];
            hf->e_k[a] = -0.5 * (matrix_inner(d_matrix[a], k_matrix[a], msize) + matrix_inner(d_matrix_b[a], k_matrix_b[a], msize)) * hf_frac;
            hf->e_j[a] = 0.5 * (matrix_inner(d_matrix[a], j_matrix[a], msize) + matrix_inner(d_matrix_b[a], j_matrix[a], msize));
            hf->e_v[a] = matrix_inner(hf->v_matrix[a], d_matrix_b[a], msize) + matrix_inner(hf->v_matrix[a], d_matrix[a], msize);
            hf->e_t[a] = matrix_inner(hf->t_matrix, d_matrix_b[a], msize) + matrix_inner(hf->t_matrix, d_matrix[a], msize);
            delta_e[a] = hf->tol_energy[a] - energy_tmp[a];
            energy_tmp[a] = hf->tol_energy[a];
        }
//        openblas_set_num_threads(get_xscf_threadnum());
        switch_xscf_thread(get_xscf_threadnum());
        for (int seg_id = 0; seg_id < hf->batch_size; seg_id++)
        {
            if (hf->background_charge[seg_id].charge_num != 0)
            {
                for (int a = 0; a < batch_size; a++)
                {
                    for (int i = 0; i < msize * msize; i++)
                        f_matrix[a][0][i] += hf->background_charge[seg_id].ev_matrix[0][i];
                    hf->background_charge[seg_id].e_ev[a] = matrix_inner(d_matrix[a], hf->background_charge[seg_id].ev_matrix, msize) + matrix_inner(d_matrix_b[a], hf->background_charge[seg_id].ev_matrix, msize);
                    hf->background_charge[seg_id].e_tolv[a] = hf->background_charge[seg_id].e_ev[a] + hf->background_charge[seg_id].e_nv[a];
                }
            }
        }
        if (loop_num >= hf->ctrl.max_iter)
            break;
        for (int a = 0; a < batch_size; a++)
        {
            if (if_finnish[a] == 1)
                continue;
            if (hf->if_sub_use)
            {
                gtrafock(f_matrix_sub[a], subspace, subspace, f_matrix[a], sub_size, sub_size, msize);
                gtrafock(f_matrix_b_sub[a], subspace, subspace, f_matrix_b[a], sub_size, sub_size, msize);
            }
            if (loop_num > 1)
            {
                switch (hf->open_type)
                {
                case UHF_WORK:
                    break;
                case ROHF_WORK:
                case CUHF_WORK:
                    cufock(f_matrix[a], f_matrix_b[a], d_matrix[a], d_matrix_b[a], hf->s_half_matrix, x_matrix, hf->alpha_seg[a], hf->beta_seg[a], msize);
                default:
                    break;
                }
                if (!hf->if_sub_use)
                {
                    if (hf->alpha_seg[a] != 0)
                        fds_err[a] = do_diis(s_matrix, f_matrix[a], d_matrix[a], diis[a]);
                    if (hf->beta_seg[a] != 0)
                        fds_err[a] += do_diis(s_matrix, f_matrix_b[a], d_matrix_b[a], diis_b[a]);
                    fds_err[a] *= 0.5;
                }
                else
                {
                    fds_err[a] = do_diis(s_matrix_sub, f_matrix_b[a], d_matrix_sub[a], diis[a]);
                    fds_err[a] += do_diis(s_matrix_sub, f_matrix_b_sub[a], d_matrix_b_sub[a], diis_b[a]);
                    fds_err[a] *= 0.5;
                }
            }
            if (!hf->if_sub_use)
            {
                cal_c_matrix(c_matrix[a], f_matrix[a], s_matrix, x_matrix, hf->alpha_egn[a], msize);
                cal_c_matrix(c_matrix_b[a], f_matrix_b[a], s_matrix, x_matrix, hf->beta_egn[a], msize);
            }
            else
            {
                cal_c_matrix(c_matrix_sub[a], f_matrix_sub[a], s_matrix_sub, x_matrix_sub, hf->alpha_egn[a], sub_size);
                cal_c_matrix(c_matrix_b_sub[a], f_matrix_b_sub[a], s_matrix_sub, x_matrix_sub, hf->beta_egn[a], sub_size);
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, sub_size, msize, sub_size, 1., c_matrix_sub[a][0], sub_size,
                            subspace[0], msize, 0., c_matrix[a][0], msize);
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, sub_size, sub_size, hf->alpha_seg[a], 1., c_matrix_sub[a][0],
                            sub_size, c_matrix_sub[a][0], sub_size, 0, d_matrix_sub[a][0], sub_size);
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, sub_size, msize, sub_size, 1., c_matrix_b_sub[a][0], sub_size,
                            subspace[0], msize, 0., c_matrix_b[a][0], msize);
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, sub_size, sub_size, hf->alpha_seg[a], 1., c_matrix_b_sub[a][0],
                            sub_size, c_matrix_b_sub[a][0], sub_size, 0, d_matrix_b_sub[a][0], sub_size);
            }
            memcpy(d_matrix_cp[a][0], d_matrix[a][0], sizeof(double) * msize * msize);
            memcpy(d_matrix_b_cp[a][0], d_matrix_b[a][0], sizeof(double) * msize * msize);
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
            for (int i = 0; i < msize * msize; i++)
            {
                d_matrix_delta[a][0][i] = d_matrix[a][0][i] - d_matrix_cp[a][0][i];
                d_matrix_b_delta[a][0][i] = d_matrix_b[a][0][i] - d_matrix_b_cp[a][0][i];
                d_matrix_tol_delta[a][0][i] = (d_matrix_delta[a][0][i] + d_matrix_b_delta[a][0][i]);
            }
            delta_d[a] = 0.;
            for (int i = 0; i < msize * msize; i++)
                if (fabs(d_matrix_delta[a][0][i]) > delta_d[a])
                    delta_d[a] = fabs(d_matrix_delta[a][0][i]);
            if (fabs(delta_d[a]) <= hf->ctrl.max_delta_d && fabs(delta_e[a]) <= hf->ctrl.max_delta_e && fds_err[a] >= hf->ctrl.max_fds_err)
                if_finnish[a] = 1;
        }
        step_t2 = omp_get_wtime();
        print_uhf(loop_num, delta_e, delta_d, fds_err, step_t2 - step_t1, hf, print_level);
        int if_break = 1;
        for (int a = 0; a < batch_size; a++)
            if (if_finnish[a] == 0)
            {
                if_break = 0;
                break;
            }
        if (if_break)
            break;
    }
    if (!hf->exchange->grids_final || !hf->exchange->if_aux)
    {
//        openblas_set_num_threads(1);
        switch_xscf_thread(0);
        for (int i = 0; i < batch_size; i++)
            if_finnish[i] = false;
        for (int a = 0; a < batch_size; a++)
            for (int i = 0; i < msize * msize; i++)
                d_matrix_tol_delta[a][0][i] = d_matrix[a][0][i] + d_matrix_b[a][0][i];
        memset(k_matrix[0][0], 0, sizeof(double) * batch_size * msize * msize);
        memset(k_matrix_b[0][0], 0, sizeof(double) * batch_size * msize * msize);
        memset(j_matrix[0][0], 0, sizeof(double) * batch_size * msize * msize);
        cal_j_matrix_batch(if_finnish, d_matrix_tol_delta, hf->coulomb);
        cal_k_matrix_batch(if_finnish, d_matrix, d_matrix_b, hf->exchange);
        for (int i = 0; i < batch_size * msize * msize; i++)
        {
            f_matrix[0][0][i] = h_matrix[0][0][i] - k_matrix[0][0][i] * hf_frac + j_matrix[0][0][i];
            f_matrix_b[0][0][i] = h_matrix[0][0][i] - k_matrix_b[0][0][i] * hf_frac + j_matrix[0][0][i];
        }
        for (int i = 0; i < batch_size; i++)
        {
            hf->e_k[i] = -0.5 * (matrix_inner(d_matrix[i], k_matrix[i], msize) + matrix_inner(d_matrix_b[i], k_matrix_b[i], msize)) * hf_frac;
            hf->e_j[i] = 0.5 * (matrix_inner(d_matrix[i], j_matrix[i], msize) + matrix_inner(d_matrix_b[i], j_matrix[i], msize));
            hf->e_v[i] = matrix_inner(hf->v_matrix[i], d_matrix_b[i], msize) + matrix_inner(hf->v_matrix[i], d_matrix[i], msize);
            hf->e_t[i] = matrix_inner(hf->t_matrix, d_matrix_b[i], msize) + matrix_inner(hf->t_matrix, d_matrix[i], msize);
            if (hf->dft != NULL)
                hf->e_xc[i] = cal_dft_exc(d_matrix[i], d_matrix_b[i], hf->dft);
            hf->tol_energy[i] = cal_uhf_energy(d_matrix[i], d_matrix_b[i], h_matrix[i], f_matrix[i], f_matrix_b[i], msize) + hf->nuc_rep[i] + hf->e_xc[i];
        }
    }
    free_tensor3d(d_matrix_cp);
    free_tensor3d(d_matrix_b_cp);
    free_tensor3d(d_matrix_delta);
    free_tensor3d(d_matrix_b_delta);
    free_tensor3d(d_matrix_tol_delta);
    for (int a = 0; a < batch_size; a++)
    {
        del_diis(diis[a]);
        del_diis(diis_b[a]);
    }
    free(diis);
    free(diis_b);
    end_t = omp_get_wtime();
    if (print_level >= 1)
        for (int a = 0; a < batch_size; a++)
            printf("E_tol = %12.6f, E_x = %12.6f, E_ne = %12.6f, T = %12.6f, E_ele = %12.6f, E_xc = %12.6f\n", hf->tol_energy[a], hf->e_k[a], hf->e_v[a], hf->e_t[a], hf->e_j[a], hf->e_xc[a]);
    for (int a = 0; a < batch_size; a++)
    {
        int alpha_orb = hf->alpha_seg[a], beta_orb = hf->beta_seg[a];
        double *d_matrix = hf->d_matrix[a][0], *d_matrix_b = hf->d_matrix_b[a][0];
        double *s_matrix = hf->s_matrix[0];
        double s_ = 0.5 * (alpha_orb - beta_orb);
        double s2 = s_ * (s_ + 1) + beta_orb;
        int i, j;
        double *d_matrix_mo = (double *)malloc(sizeof(double) * msize * msize);
        double *d_matrix_b_mo = (double *)malloc(sizeof(double) * msize * msize);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, msize, msize, msize, 1.0, d_matrix, msize, s_matrix, msize, 0.0, d_matrix_mo, msize);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, msize, msize, msize, 1.0, d_matrix_b, msize, s_matrix, msize, 0.0, d_matrix_b_mo, msize);
        for (i = 0; i < msize; i++)
        {
            for (j = 0; j < msize; j++)
                s2 -= d_matrix_mo[i * msize + j] * d_matrix_b_mo[j * msize + i];
        }
        hf->S2[a] = s2;
        free(d_matrix_mo);
        free(d_matrix_b_mo);
    }

    free(if_finnish);
    free_vector(fds_err);
    free_vector(delta_d);
    free_vector(delta_e);
    free_vector(energy_tmp);

//    openblas_set_num_threads(1);
    switch_xscf_thread(0);
    return;
}

void cal_no_orb(Matrix c_matrix_no, Vector occ_num, const Matrix s_half_matrix, const Matrix x_matrix, const Matrix d_matrix, int msize)
{
    Vector d_matrix_v = d_matrix[0], c_matrix_no_v = c_matrix_no[0],
           s_half_matrix_v = s_half_matrix[0], x_matrix_v = x_matrix[0];
    int lwork = 4 * msize * msize;
    Vector work = (Vector)malloc(sizeof(double) * lwork);
    int erro;
    int i, j;
    double tmp;
    Vector d_matrix_tmp = (Vector)malloc(sizeof(double) * msize * msize);
    memcpy(d_matrix_tmp, d_matrix_v, sizeof(double) * msize * msize);
    trafock(d_matrix_tmp, s_half_matrix_v, msize);
    LAPACK_dsyev("V", "U", &msize, d_matrix_tmp, &msize, occ_num, work, &lwork, &erro);
    for (i = 0; i < msize * 0.5; i++)
    {
        tmp = occ_num[i];
        occ_num[i] = occ_num[msize - i - 1];
        occ_num[msize - i - 1] = tmp;
        for (j = 0; j < msize; j++)
        {
            tmp = d_matrix_tmp[i * msize + j];
            d_matrix_tmp[i * msize + j] = d_matrix_tmp[(msize - 1 - i) * msize + j];
            d_matrix_tmp[(msize - 1 - i) * msize + j] = tmp;
        }
    }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, msize, msize, msize,
                1.0, d_matrix_tmp, msize, x_matrix_v, msize, 0.0, c_matrix_no_v, msize);
    free(work);
    free(d_matrix_tmp);
    return;
}

/**
 * @brief load subspace for Hartree-Fock calculation
 *
 * @param sub_size the size of subspace
 * @param subspace compends of subspace
 * @param hf data structure of Hartree-Fock
 */
void load_hf_sub(int sub_size, const Matrix subspace, hf_info hf)
{
    int batch_size = hf->batch_size;
    int msize = hf->msize;
    hf->if_sub_def = true;
    hf->sub_size = sub_size;
    hf->subspace = malloc_matrix(sub_size, msize);
    hf->s_matrix_sub = malloc_matrix(sub_size, sub_size);
    hf->x_matrix_sub = malloc_matrix(sub_size, sub_size);
    gtrafock(hf->s_matrix_sub, subspace, subspace, hf->s_matrix, sub_size, sub_size, msize);
    copy_matrix(hf->x_matrix_sub, hf->s_matrix_sub, sub_size, sub_size);
    cal_matrix_syhalfrev(hf->x_matrix_sub, sub_size);
    hf->c_matrix_sub = malloc_tensor3d(batch_size, sub_size, sub_size);
    hf->d_matrix_sub = malloc_tensor3d(batch_size, sub_size, sub_size);
    hf->f_matrix_sub = malloc_tensor3d(batch_size, sub_size, sub_size);
    hf->c_matrix_b_sub = malloc_tensor3d(batch_size, sub_size, sub_size);
    hf->d_matrix_b_sub = malloc_tensor3d(batch_size, sub_size, sub_size);
    hf->f_matrix_b_sub = malloc_tensor3d(batch_size, sub_size, sub_size);
    return;
}
void del_hf_sub(hf_info hf)
{
    if (!hf->if_sub_def)
        return;
    hf->if_sub_def = false;
    hf->if_sub_use = false;
    free_matrix(hf->s_matrix_sub);
    free_matrix(hf->x_matrix_sub);
    free_tensor3d(hf->c_matrix_sub);
    free_tensor3d(hf->d_matrix_sub);
    free_tensor3d(hf->f_matrix_sub);
    free_tensor3d(hf->c_matrix_b_sub);
    free_tensor3d(hf->d_matrix_b_sub);
    free_tensor3d(hf->f_matrix_b_sub);
    return;
}
void load_hf_aim(int bath_size, const Matrix bath_coup, const Vector bath_e, hf_info hf)
{
    assert(hf->if_sub_def);
    int sub_size = hf->sub_size;
    hf->if_aim_def = true;
    hf->aim_mu = 0.;
    hf->bath_coup = malloc_matrix(bath_size, sub_size);
    hf->s_matrix_aim = malloc_matrix(bath_size + sub_size, bath_size + sub_size);
    hf->x_matrix_aim = malloc_matrix(bath_size + sub_size, bath_size + sub_size);
    hf->c_matrix_aim = malloc_tensor3d(hf->batch_size, bath_size + sub_size, bath_size + sub_size);
    hf->d_matrix_aim = malloc_tensor3d(hf->batch_size, bath_size + sub_size, bath_size + sub_size);
    hf->f_matrix_aim = malloc_tensor3d(hf->batch_size, bath_size + sub_size, bath_size + sub_size);
    hf->c_matrix_b_aim = malloc_tensor3d(hf->batch_size, bath_size + sub_size, bath_size + sub_size);
    hf->d_matrix_b_aim = malloc_tensor3d(hf->batch_size, bath_size + sub_size, bath_size + sub_size);
    hf->f_matrix_b_aim = malloc_tensor3d(hf->batch_size, bath_size + sub_size, bath_size + sub_size);
    for (int i = 0; i < sub_size; i++)
        for (int j = 0; j < sub_size; j++)
            hf->s_matrix_aim[i][j] = hf->s_matrix_sub[i][j];
    for (int i = 0; i < bath_size; i++)
        hf->s_matrix_aim[i + sub_size][i + sub_size] = 1.;
    copy_matrix(hf->x_matrix_aim, hf->s_matrix_aim, bath_size + sub_size, bath_size + sub_size);
    cal_matrix_syhalfrev(hf->x_matrix_aim, bath_size + sub_size);
    return;
}
void del_hf_aim(hf_info hf)
{
    if (!hf->if_aim_def)
        return;
    hf->if_aim_def = false;
    free_matrix(hf->s_matrix_aim);
    free_matrix(hf->x_matrix_aim);
    free_tensor3d(hf->c_matrix_aim);
    free_tensor3d(hf->d_matrix_aim);
    free_tensor3d(hf->f_matrix_aim);
    free_tensor3d(hf->c_matrix_b_aim);
    free_tensor3d(hf->d_matrix_b_aim);
    free_tensor3d(hf->f_matrix_b_aim);
    return;
}
/**
 * @brief solve Anderson impurity model by a mean-field approach as Hartree-Fock
 *
 * @param print_level
 * @param hf
 */
void do_rhf_aim(int print_level, hf_info hf)
{
    assert(hf->if_aim_def);
    int batch_size = hf->batch_size;
    int bath_size = hf->bath_size;
    int sub_size = hf->sub_size;
    double step_t1, step_t2;
    double star_t, end_t;
    int loop_num = 0;
    diis_info *diis = (diis_info *)malloc(sizeof(diis_info) * batch_size);
    Vector fds_err, delta_e, delta_d;
    Vector energy_tmp;
    int msize = get_mol_msize(hf->mol);
    mol_info mol = hf->mol;
    Tensor3D d_matrix_cp = malloc_tensor3d(batch_size, msize, msize);
    Tensor3D d_matrix = hf->d_matrix;
    Tensor3D c_matrix = hf->c_matrix;
    Tensor3D d_matrix_delta = malloc_tensor3d(batch_size, msize, msize);
    Tensor3D d_matrix_tol_delta = malloc_tensor3d(batch_size, msize, msize);
    Tensor3D f_matrix = hf->f_matrix;
    Tensor3D h_matrix = hf->h_matrix;
    Matrix x_matrix = hf->x_matrix;
    Matrix s_matrix = hf->s_matrix;
    Tensor3D j_matrix = hf->j_matrix;
    Tensor3D k_matrix = hf->k_matrix;
    // Matrix and parameters of subspace calculation
    Matrix subspace = hf->subspace;
    Matrix s_matrix_sub = hf->s_matrix_sub;
    Matrix x_matrix_sub = hf->x_matrix_sub;
    Matrix *f_matrix_sub = hf->f_matrix_sub;
    Matrix *c_matrix_sub = hf->c_matrix_sub;
    Matrix *d_matrix_sub = hf->d_matrix_sub;
    bool *if_finnish = (bool *)malloc(batch_size * sizeof(bool));
    star_t = omp_get_wtime();
    fds_err = malloc_vector(batch_size);
    delta_d = malloc_vector(batch_size);
    delta_e = malloc_vector(batch_size);
    energy_tmp = malloc_vector(batch_size);
    for (int i = 0; i < batch_size; i++)
    {
        if_finnish[i] = false;
        if (!hf->if_sub_use)
            diis[i] = init_diis(hf->diis_size, msize);
        else
            diis[i] = init_diis(hf->diis_size, sub_size);
        set_diis_tr(hf->diis_tr1, hf->diis_tr2, diis[i]);
    }
    while (loop_num >= 0)
    {
        step_t1 = omp_get_wtime();
        loop_num++;
        cal_j_matrix_batch(if_finnish, d_matrix_tol_delta, hf->coulomb);
        cal_k_matrix_batch(if_finnish, d_matrix_delta, NULL, hf->exchange);
        for (int i = 0; i < msize * msize * batch_size; i++)
            f_matrix[0][0][i] = h_matrix[0][0][i] - k_matrix[0][0][i] + j_matrix[0][0][i];
        // in this part only the energy of impurity is calculated
        for (int a = 0; a < batch_size; a++)
        {
            if (if_finnish[a] == 1)
                continue;
            hf->tol_energy[a] = cal_rhf_energy(d_matrix[a], h_matrix[a], f_matrix[a], msize) + hf->nuc_rep[a];
            hf->e_k[a] = -matrix_inner(d_matrix[a], k_matrix[a], msize);
            hf->e_j[a] = matrix_inner(d_matrix[a], j_matrix[a], msize);
            hf->e_v[a] = 2 * matrix_inner(hf->v_matrix[a], d_matrix[a], msize);
            hf->e_t[a] = 2 * matrix_inner(hf->t_matrix, d_matrix[a], msize);
            delta_e[a] = hf->tol_energy[a] - energy_tmp[a];
            energy_tmp[a] = hf->tol_energy[a];
        }
        double t_sf = omp_get_wtime();
        if (loop_num >= hf->ctrl.max_iter)
            break;
        for (int a = 0; a < batch_size; a++)
        {
            if (if_finnish[a] == 1)
                continue;
            if (hf->if_sub_use)
                gtrafock(f_matrix_sub[a], subspace, subspace, f_matrix[a], sub_size, sub_size, msize);
            if (loop_num > 1)
                if (!hf->if_sub_use)
                    fds_err[a] = do_diis(s_matrix, f_matrix[a], d_matrix[a], diis[a]);
                else
                    fds_err[a] = do_diis(s_matrix_sub, f_matrix_sub[a], d_matrix_sub[a], diis[a]);
            if (!hf->if_sub_use)
                cal_c_matrix(c_matrix[a], f_matrix[a], s_matrix, x_matrix, hf->alpha_egn[a], msize);
            // if subspace calculation is done, then a reduced eigenvalue problem will be solved
            else
            {
                cal_c_matrix(c_matrix_sub[a], f_matrix_sub[a], s_matrix_sub, x_matrix_sub, hf->alpha_egn[a], sub_size);
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, sub_size, msize, sub_size, 1., c_matrix_sub[a][0], sub_size,
                            subspace[0], msize, 0., c_matrix[a][0], msize);
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, sub_size, sub_size, hf->alpha_seg[a], 1., c_matrix_sub[a][0],
                            sub_size, c_matrix_sub[a][0], sub_size, 0, d_matrix_sub[a][0], sub_size);
            }
            memcpy(d_matrix_cp[a][0], d_matrix[a][0], sizeof(double) * msize * msize);
            memset(d_matrix[a][0], 0, sizeof(double) * msize * msize);
            for (int i = 0; i < msize; i++)
                for (int j = 0; j < msize; j++)
                    for (int k = 0; k < hf->alpha_seg[a]; k++)
                        d_matrix[a][i][j] += c_matrix[a][k][i] * c_matrix[a][k][j];
            for (int i = 0; i < msize * msize; i++)
            {
                d_matrix_delta[a][0][i] = d_matrix[a][0][i] - d_matrix_cp[a][0][i];
                d_matrix_tol_delta[a][0][i] = d_matrix_delta[a][0][i] * 2;
            }
            delta_d[a] = 0.;
            for (int i = 0; i < msize * msize; i++)
                if (fabs(d_matrix_delta[a][0][i]) > delta_d[a])
                    delta_d[a] = fabs(d_matrix_delta[a][0][i]);
            if (fabs(delta_d[a]) <= hf->ctrl.max_delta_d && fabs(delta_e[a]) <= hf->ctrl.max_delta_e && fds_err[a] >= hf->ctrl.max_fds_err)
                if_finnish[a] = 1;
        }
        step_t2 = omp_get_wtime();
        print_uhf(loop_num, delta_e, delta_d, fds_err, step_t2 - step_t1, hf, print_level);
        int if_break = 1;
        for (int a = 0; a < batch_size; a++)
            if (if_finnish[a] == 0)
            {
                if_break = 0;
                break;
            }
        if (if_break)
            break;
    }
    return;
}
void cal_rhf_grad(Vector grad, const hf_info hf)
{
    const mol_info mol = hf->mol;
    const atm_info atm = mol->atm;
    const bas_info bas = mol->bas;
    const Matrix *c_matrix = hf->c_matrix;
    const Matrix *d_matrix = hf->d_matrix;
    const Vector *egn = hf->alpha_egn;
    const pair_info pair = mol->pair;
    int batch_size = hf->batch_size;
    int msize = get_mol_msize(mol);
    int natm = atm->natm;
    int nbas = bas->nbas;
    int charge_num = hf->background_charge[0].charge_num;
    Vector ev_grad = hf->background_charge[0].ev_grad;
    Vector charge_value = hf->background_charge[0].charge_value;
    // n-n repulsion part
    Matrix dist_matrix = malloc_matrix(natm, natm);
    if (charge_num != 0)
        memset(ev_grad, 0, sizeof(double) * 3 * get_mol_natm(mol));
    for (int i = 0; i < natm; i++)
    {
        const double *A = get_atm_coord(i, atm);
        for (int j = 0; j < i; j++)
        {
            const double *B = get_atm_coord(j, atm);
            dist_matrix[i][j] = sqrt((A[0] - B[0]) * (A[0] - B[0]) + (A[1] - B[1]) * (A[1] - B[1]) + (A[2] - B[2]) * (A[2] - B[2]));
            dist_matrix[j][i] = dist_matrix[i][j];
        }
    }
    double AB[3];
    for (int i = 0; i < natm; i++)
    {
        int zi = atm(ELEMENT_VAL, i);
        const double *A = get_atm_coord(i, atm);
        for (int j = 0; j < natm; j++)
        {
            int zj = atm(ELEMENT_VAL, j);
            const double *B = get_atm_coord(j, atm);
            if (i == j)
                continue;
            double dist = dist_matrix[i][j];
            AB[0] = A[0] - B[0];
            AB[1] = A[1] - B[1];
            AB[2] = A[2] - B[2];
            grad[3 * i + 0] -= zi * zj * AB[0] / (dist * dist * dist);
            grad[3 * i + 1] -= zi * zj * AB[1] / (dist * dist * dist);
            grad[3 * i + 2] -= zi * zj * AB[2] / (dist * dist * dist);
        }
        if (charge_num != 0)
        {
            for (int g = 0; g < charge_num; g++)
            {
                const double *G = &charge_value[4 * g];
                AB[0] = A[0] - G[0];
                AB[1] = A[1] - G[1];
                AB[2] = A[2] - G[2];
                double dist = sqrt(AB[0] * AB[0] + AB[1] * AB[1] + AB[2] * AB[2]);
                grad[3 * i + 0] -= zi * G[4] * AB[0] / (dist * dist * dist);
                grad[3 * i + 1] -= zi * G[4] * AB[1] / (dist * dist * dist);
                grad[3 * i + 2] -= zi * G[4] * AB[2] / (dist * dist * dist);
                ev_grad[3 * i + 0] += zi * G[4] * AB[0] / (dist * dist * dist);
                ev_grad[3 * i + 1] += zi * G[4] * AB[1] / (dist * dist * dist);
                ev_grad[3 * i + 2] += zi * G[4] * AB[2] / (dist * dist * dist);
            }
        }
    }
    free_matrix(dist_matrix);
    // calculate gradient for one-electron operator matrix
    // overlap part
    xint_info xint = init_xint(mol);
    double buf[1024 * 3];
    for (int ij = 0; ij < pair->len; ij++)
    {
        int i = pair->shp[ij].shli;
        int j = pair->shp[ij].shlj;
        int atmi = bas(ATOM_IND, i);
        int di = xint_gtolen(bas(ANGULAR_VAL, i));
        int pi = bas->shls_p[i];
        int atmj = bas(ATOM_IND, j);
        int dj = xint_gtolen(bas(ANGULAR_VAL, j));
        int pj = bas->shls_p[j];
        if (atmi == atmj)
            continue;
        xint_overlap_grad(buf, ij, xint);
        for (int u = 0, uu = 0; u < di; u++)
            for (int v = 0; v < dj; v++, uu++)
                for (int k = 0; k < hf->alpha_seg[0]; k++)
                {
                    grad[3 * atmi + 0] += 4 * egn[0][k] * c_matrix[0][k][pi + u] *
                                          c_matrix[0][k][pj + v] * buf[3 * uu + 0];
                    grad[3 * atmi + 1] += 4 * egn[0][k] * c_matrix[0][k][pi + u] *
                                          c_matrix[0][k][pj + v] * buf[3 * uu + 1];
                    grad[3 * atmi + 2] += 4 * egn[0][k] * c_matrix[0][k][pi + u] *
                                          c_matrix[0][k][pj + v] * buf[3 * uu + 2];
                    grad[3 * atmj + 0] -= 4 * egn[0][k] * c_matrix[0][k][pi + u] *
                                          c_matrix[0][k][pj + v] * buf[3 * uu + 0];
                    grad[3 * atmj + 1] -= 4 * egn[0][k] * c_matrix[0][k][pi + u] *
                                          c_matrix[0][k][pj + v] * buf[3 * uu + 1];
                    grad[3 * atmj + 2] -= 4 * egn[0][k] * c_matrix[0][k][pi + u] *
                                          c_matrix[0][k][pj + v] * buf[3 * uu + 2];
                }
    }
    // n-e attraction part
    for (int ij = 0; ij < mol->pair->len; ij++)
    {
        int i = mol->pair->shp[ij].shli;
        int j = mol->pair->shp[ij].shlj;
        int atmi = bas(ATOM_IND, i);
        int di = xint_gtolen(bas(ANGULAR_VAL, i));
        int pi = bas->shls_p[i];
        int atmj = bas(ATOM_IND, j);
        int dj = xint_gtolen(bas(ANGULAR_VAL, j));
        int pj = bas->shls_p[j];
        for (int a = 0; a < get_mol_natm(mol); a++)
        {
            const double *G = get_atm_coord(a, atm);
            xint_3c1e_grad(buf, ij, G, xint);
            int z = atm(ELEMENT_VAL, a);
            double factor = 4 * z;
            if (i == j)
                factor *= 0.5;
            for (int u = 0, uu = 0; u < di; u++)
                for (int v = 0; v < dj; v++, uu++)
                {
                    grad[a * 3 + 0] += factor * (buf[6 * uu + 0] + buf[6 * uu + 3]) * d_matrix[0][pi + u][pj + v];
                    grad[a * 3 + 1] += factor * (buf[6 * uu + 1] + buf[6 * uu + 4]) * d_matrix[0][pi + u][pj + v];
                    grad[a * 3 + 2] += factor * (buf[6 * uu + 2] + buf[6 * uu + 5]) * d_matrix[0][pi + u][pj + v];
                }
        }
        // electron-external charge part
        if (charge_num != 0)
        {
            for (int a = 0; a < charge_num; a++)
            {
                const double *G = &charge_value[a * 4];
                xint_3c1e_grad(buf, ij, G, xint);
                double z = G[4];
                double factor = 4 * z;
                if (i == j)
                    factor *= 0.5;
                for (int u = 0, uu = 0; u < di; u++)
                    for (int v = 0; v < dj; v++, uu++)
                    {
                        ev_grad[a * 3 + 0] += factor * (buf[6 * uu + 0] + buf[6 * uu + 3]) * d_matrix[0][pi + u][pj + v];
                        ev_grad[a * 3 + 1] += factor * (buf[6 * uu + 1] + buf[6 * uu + 4]) * d_matrix[0][pi + u][pj + v];
                        ev_grad[a * 3 + 2] += factor * (buf[6 * uu + 2] + buf[6 * uu + 5]) * d_matrix[0][pi + u][pj + v];
                    }
            }
        }
    }
    for (int ij = 0; ij < mol->pair->len; ij++)
    {
        int i = mol->pair->shp[ij].shli;
        int j = mol->pair->shp[ij].shlj;
        int atmi = bas(ATOM_IND, i);
        int di = xint_gtolen(bas(ANGULAR_VAL, i));
        int pi = bas->shls_p[i];
        int atmj = bas(ATOM_IND, j);
        int dj = xint_gtolen(bas(ANGULAR_VAL, j));
        int pj = bas->shls_p[j];
        double factor = 4.;
        if (atmi != atmj)
        {
            xint_kinetic_grad(buf, ij, xint);
            for (int u = 0, uu = 0; u < di; u++)
                for (int v = 0; v < dj; v++, uu++)
                {
                    grad[3 * atmi + 0] -= 4 * buf[3 * uu + 0] * d_matrix[0][pi + u][pj + v];
                    grad[3 * atmj + 0] += 4 * buf[3 * uu + 0] * d_matrix[0][pi + u][pj + v];
                    grad[3 * atmi + 1] -= 4 * buf[3 * uu + 1] * d_matrix[0][pi + u][pj + v];
                    grad[3 * atmj + 1] += 4 * buf[3 * uu + 1] * d_matrix[0][pi + u][pj + v];
                    grad[3 * atmi + 2] -= 4 * buf[3 * uu + 2] * d_matrix[0][pi + u][pj + v];
                    grad[3 * atmj + 2] += 4 * buf[3 * uu + 2] * d_matrix[0][pi + u][pj + v];
                }
        }
        xint_ne_grad(buf, ij, xint);
        if (i == j)
            factor *= 0.5;
        for (int u = 0, uu = 0; u < di; u++)
            for (int v = 0; v < dj; v++, uu++)
            {
                grad[3 * atmi + 0] += factor * buf[6 * uu + 0] * d_matrix[0][pi + u][pj + v];
                grad[3 * atmj + 0] += factor * buf[6 * uu + 3] * d_matrix[0][pi + u][pj + v];
                grad[3 * atmi + 1] += factor * buf[6 * uu + 1] * d_matrix[0][pi + u][pj + v];
                grad[3 * atmj + 1] += factor * buf[6 * uu + 4] * d_matrix[0][pi + u][pj + v];
                grad[3 * atmi + 2] += factor * buf[6 * uu + 2] * d_matrix[0][pi + u][pj + v];
                grad[3 * atmj + 2] += factor * buf[6 * uu + 5] * d_matrix[0][pi + u][pj + v];
            }
    }
//    printf("1-e gradient\n");
//    for (int k=0;k<hf->mol->atm->natm;k++)
//      printf("%f %f %f\n", grad[3*k], grad[3*k+1], grad[3*k+2]);
//    Vector grad0=malloc_vector(3*hf->mol->atm->natm);
//    memcpy(grad0,grad,sizeof(double)*3*hf->mol->atm->natm);
    // two-electron part
    hf->coulomb->batch_size = 1;
    hf->exchange->batch_size = 1;
//    openblas_set_num_threads(1);
    switch_xscf_thread(0);
    cal_coulomb_grad_batch(&grad, hf->d_matrix, hf->coulomb);
//    printf("gradient including 1-e and coulomb\n");
//    for (int k=0;k<hf->mol->atm->natm;k++)
//      printf("%f %f %f\n", grad[3*k], grad[3*k+1], grad[3*k+2]);
//    printf("gradient including only coulomb\n");
//    for (int k=0;k<hf->mol->atm->natm;k++)
//      printf("%f %f %f\n", grad[3*k]-grad0[3*k], grad[3*k+1]-grad0[3*k+1], grad[3*k+2]-grad0[3*k+2]);
//    memcpy(grad0,grad,sizeof(double)*3*hf->mol->atm->natm);
    cal_exchange_grad_cosx(&grad, hf->d_matrix, NULL, hf->exchange);
    hf->coulomb->batch_size = hf->batch_size;
    hf->exchange->batch_size = hf->batch_size;
//    printf("total gradient\n");
//    for (int k=0;k<hf->mol->atm->natm;k++)
//      printf("%f %f %f\n", grad[3*k], grad[3*k+1], grad[3*k+2]);
//    printf("gradient including only exchange\n");
//    for (int k=0;k<hf->mol->atm->natm;k++)
//      printf("%f %f %f\n", grad[3*k]-grad0[3*k], grad[3*k+1]-grad0[3*k+1], grad[3*k+2]-grad0[3*k+2]);
    // calculate the gradient
    return;
}

void load_hf_charge(int seg_id, int charge_num, const Vector charge_value, hf_info hf)
{
    int msize = hf->msize;
    if (hf->background_charge[seg_id].charge_num != 0)
    {
        free_matrix(hf->background_charge[seg_id].ev_matrix);
        free_vector(hf->background_charge[seg_id].charge_value);
        free_vector(hf->background_charge[seg_id].ev_grad);
        free_vector(hf->background_charge[seg_id].e_ev);
        free_vector(hf->background_charge[seg_id].e_nv);
        free_vector(hf->background_charge[seg_id].e_tolv);
    }
    hf->background_charge[seg_id].charge_num = charge_num;
    hf->background_charge[seg_id].charge_value = malloc_vector(4 * charge_num);
    hf->background_charge[seg_id].ev_matrix = malloc_matrix(msize, msize);
    copy_vector(hf->background_charge[seg_id].charge_value, charge_value, 4 * charge_num);
    const mol_info mol = hf->mol;
    const bas_info bas = mol->bas;
    const pair_info pair = mol->pair;
    double buf[1024] = {0};
    int shli, shlj;
    int pi, pj;
    int di, dj;
    int ci, cj;
    xint_info xint;
    Matrix ev_matrix = hf->background_charge[seg_id].ev_matrix;
    xint = init_xint(mol);
    for (int i = 0; i < pair->len; i++)
    {
        for (int a = 0; a < charge_num; a++)
        {
            xint_3c1e(buf, i, &charge_value[4 * a], xint);
            shli = pair->shp[i].shli;
            shlj = pair->shp[i].shlj;
            di = pair->shp[i].di;
            dj = pair->shp[i].dj;
            ci = xint_gtolen(mol->bas(ANGULAR_VAL, shli));
            cj = xint_gtolen(mol->bas(ANGULAR_VAL, shlj));
            pi = bas->shls_p[shli];
            pj = bas->shls_p[shlj];
            for (int u = 0, uu = 0; u < ci; u++)
                for (int v = 0; v < cj; v++, uu++)
                {
                    ev_matrix[u + pi][v + pj] -= charge_value[4 * a + 3] * buf[uu];
                    ev_matrix[v + pj][u + pi] -= charge_value[4 * a + 3] * buf[uu];
                }
        }
    }
    // calculate the interaction of molecule and external charge
    hf->background_charge[seg_id].e_ev = malloc_vector(hf->batch_size);
    hf->background_charge[seg_id].e_nv = malloc_vector(hf->batch_size);
    hf->background_charge[seg_id].e_tolv = malloc_vector(hf->batch_size);
    for (int n = 1, a = 0; n < hf->batch_size; n++)
    {
        for (int i = 0; i < hf->atm_seg[n]; i++, a++)
        {
            const Vector A = get_atm_coord(a, mol->atm);
            double Ac = mol->atm(ELEMENT_VAL, a);
            for (int g = 0; g < charge_num; g++)
            {
                const Vector G = &charge_value[4 * g];
                double dist = sqrt((A[0] - G[0]) * (A[0] - G[0]) + (A[1] - G[1]) * (A[1] - G[1]) + (A[2] - G[2]) * (A[2] - G[2]));
                hf->background_charge[seg_id].e_nv[0] += Ac * G[4] / dist;
                hf->background_charge[seg_id].e_nv[n] += Ac * G[4] / dist;
            }
        }
    }
    hf->background_charge[seg_id].ev_grad = malloc_vector(3 * charge_num);
    del_xint(xint);
    return;
}
