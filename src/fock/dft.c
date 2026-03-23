#include "fock/dft.h"
int B3LYP_ID[] = {402};
double B3LYP_FRAC[] = {1.};
int BLYP_ID[] = {106, 131};
double BLYP_FRAC[] = {
    1.,
    1.,
};
int PBE0_ID[] = {406};
double PBE0_FRAC[] = {1.};
int PBE_ID[] = {101, 130};
double PBE_FRAC[] = {1., 1.};
dft_info init_dft(double hf_frac, GRIDS_TYPE gtype, const int dft_id[], const double dft_frac[], int num, int disp_type, int dft_name, const mol_info mol)
{
    if (num == 0)
        return NULL;
    dft_info dft = (dft_info)malloc(sizeof(struct DftInfo));
    dft->func_num = num;
    dft->xc_frac = (double *)malloc(sizeof(double) * num);
    dft->dft_id = malloc_list(num);
    dft->xc_func = (xc_func_type *)malloc(sizeof(xc_func_type) * num);
    dft->xc_func_pol = (xc_func_type *)malloc(sizeof(xc_func_type) * num);
    dft->type = (int *)malloc(sizeof(int) * num);
    dft->hf_frac = hf_frac;
    dft->grid_type = gtype;
    dft->max_type = 0;
    for (int i = 0; i < num; i++)
    {
        dft->xc_frac[i] = dft_frac[i];
        dft->dft_id[i] = dft_id[i];
        xc_func_init(&dft->xc_func[i], dft_id[i], XC_UNPOLARIZED);
        xc_func_init(&dft->xc_func_pol[i], dft_id[i], XC_POLARIZED);
        switch (dft->xc_func[i].info->family)
        {
        case XC_FAMILY_LDA:
        case XC_FAMILY_HYB_LDA:
            dft->type[i] = 1;
            break;
        case XC_FAMILY_GGA:
        case XC_FAMILY_HYB_GGA:
            dft->type[i] = 2;
            break;
        case XC_FAMILY_MGGA:
        case XC_FAMILY_HYB_MGGA:
            dft->type[i] = 3;
            break;
        default:
            break;
        }
        if (dft->type[i] > dft->max_type)
            dft->max_type = dft->type[i];
    }
    dft->mol = mol;
    // inital grids for DFT numerical integrals
    dft->grids = init_grid(gtype, mol);
    dft->disp_type = disp_type;
    if (disp_type != false)
    {
        dft->disp = init_disp(mol, dft_name, disp_type);
        dft->dft_name = dft_name;
    }
    else
        dft->disp = NULL;
    return dft;
}

void del_dft(dft_info dft)
{
    del_grid(dft->grids);
    free(dft->xc_frac);
    free(dft->dft_id);
    free(dft->type);
    for (int i = 0; i < dft->func_num; i++)
    {
        xc_func_end(&dft->xc_func[i]);
        xc_func_end(&dft->xc_func_pol[i]);
    }
    free(dft->xc_func);
    free(dft->xc_func_pol);
    if (dft->disp != NULL)
        del_disp(dft->disp);
    free(dft);
    return;
}
void cal_lda_block(const Matrix d_matrix, const Matrix d_matrix_b, const grid_block block, int msize, Vector rho, Vector rho_b)
{
    int grid_num = block->grid_num;
    int pol = 0;
    Matrix tmp = malloc_matrix(grid_num, msize);
    // use orbital matrix
    if (d_matrix_b != NULL)
        pol = 1;
    if (pol)
        memset(rho_b, 0, sizeof(double) * grid_num);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, grid_num, msize, msize, 1., block->psi[0], msize,
                d_matrix[0], msize, 0., tmp[0], msize);
    for (int i = 0; i < grid_num; i++)
    {
        rho[i] = vv_dot(tmp[i], block->psi[i], msize);
    }
    if (pol)
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, grid_num, msize, msize, 1., block->psi[0], msize,
                    d_matrix_b[0], msize, 0., tmp[0], msize);
        for (int i = 0; i < grid_num; i++)
        {
            rho_b[i] = vv_dot(tmp[i], block->psi[i], msize);
        }
    }
    free_matrix(tmp);
    return;
}

void cal_gga_block(const Matrix d_matrix, const Matrix d_matrix_b, const grid_block block,
                   int msize, Vector rho, Vector rho_b, Vector rhox, Vector rhoy, Vector rhoz, Vector rhox_b, Vector rhoy_b,
                   Vector rhoz_b, Vector sigma, Vector sigma_b, Vector sigma_ab)
{
    int grid_num = block->grid_num;
    int pol = 0;
    Matrix tmp = malloc_matrix(grid_num, msize);
    if (d_matrix_b != NULL)
        pol = 1;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, grid_num, msize, msize, 1., block->psi[0], msize,
                d_matrix[0], msize, 0., tmp[0], msize);
    for (int i = 0; i < grid_num; i++)
    {
        rho[i] = vv_dot(tmp[i], block->psi[i], msize);
        rhox[i] = vv_dot(tmp[i], block->psix[i], msize) * 2.;
        rhoy[i] = vv_dot(tmp[i], block->psiy[i], msize) * 2.;
        rhoz[i] = vv_dot(tmp[i], block->psiz[i], msize) * 2.;
        sigma[i] = (rhox[i] * rhox[i] + rhoy[i] * rhoy[i] + rhoz[i] * rhoz[i]);
    }
    if (pol)
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, grid_num, msize, msize, 1., block->psi[0], msize,
                    d_matrix_b[0], msize, 0., tmp[0], msize);
        for (int i = 0; i < grid_num; i++)
        {
            rho_b[i] = vv_dot(tmp[i], block->psi[i], msize);
            rhox_b[i] = vv_dot(tmp[i], block->psix[i], msize) * 2.;
            rhoy_b[i] = vv_dot(tmp[i], block->psiy[i], msize) * 2.;
            rhoz_b[i] = vv_dot(tmp[i], block->psiz[i], msize) * 2.;
            sigma_b[i] = (rhox_b[i] * rhox_b[i] + rhoy_b[i] * rhoy_b[i] + rhoz_b[i] * rhoz_b[i]);
            sigma_ab[i] = (rhox[i] * rhox_b[i] + rhoy[i] * rhoy_b[i] + rhoz[i] * rhoz_b[i]);
        }
    }
    else
        for (int i = 0; i < grid_num; i++)
        {
            rho[i] *= 2.;
            sigma[i] *= 4.;
        }
    free_matrix(tmp);
    return;
}

void cal_meta_gga_block(const Matrix d_matrix, const Matrix d_matrix_b, const grid_block block,
                        int msize, Vector rho, Vector rho_b, Vector rhox, Vector rhoy, Vector rhoz, Vector rhox_b, Vector rhoy_b,
                        Vector rhoz_b, Vector sigma, Vector sigma_b, Vector sigma_ab, Vector lapl, Vector lapl_b, Vector tau, Vector tau_b)
{
    int grid_num = block->grid_num;
    int pol = 0;
    Matrix tmp = malloc_matrix(grid_num, msize);
    if (d_matrix_b != NULL)
        pol = 1;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, grid_num, msize, msize, 1., block->psi[0], msize,
                d_matrix[0], msize, 0., tmp[0], msize);
    for (int i = 0; i < grid_num; i++)
    {
        rho[i] = vv_dot(tmp[i], block->psi[i], msize);
        rhox[i] = vv_dot(tmp[i], block->psix[i], msize) * 2.;
        rhoy[i] = vv_dot(tmp[i], block->psiy[i], msize) * 2.;
        rhoz[i] = vv_dot(tmp[i], block->psiz[i], msize) * 2.;
        sigma[i] = (rhox[i] * rhox[i] + rhoy[i] * rhoy[i] + rhoz[i] * rhoz[i]);
        lapl[i] = 2 * (vv_dot(tmp[i], block->psixx[i], msize) +
                       vv_dot(tmp[i], block->psiyy[i], msize) +
                       vv_dot(tmp[i], block->psizz[i], msize));
    }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, grid_num, msize, msize, 1., block->psix[0], msize,
                d_matrix[0], msize, 0., tmp[0], msize);
    for (int i = 0; i < grid_num; i++)
    {
        lapl[i] += 2 * vv_dot(tmp[i], block->psix[i], msize);
        tau[i] = 0.5 * vv_dot(tmp[i], block->psix[i], msize);
    }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, grid_num, msize, msize, 1., block->psiy[0], msize,
                d_matrix[0], msize, 0., tmp[0], msize);
    for (int i = 0; i < grid_num; i++)
    {
        lapl[i] += 2 * vv_dot(tmp[i], block->psiy[i], msize);
        tau[i] += 0.5 * vv_dot(tmp[i], block->psiy[i], msize);
    }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, grid_num, msize, msize, 1., block->psiz[0], msize,
                d_matrix[0], msize, 0., tmp[0], msize);
    for (int i = 0; i < grid_num; i++)
    {
        lapl[i] += 2 * vv_dot(tmp[i], block->psiz[i], msize);
        tau[i] += 0.5 * vv_dot(tmp[i], block->psiz[i], msize);
    }
    if (pol)
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, grid_num, msize, msize, 1., block->psi[0], msize,
                    d_matrix_b[0], msize, 0., tmp[0], msize);
        for (int i = 0; i < grid_num; i++)
        {
            rho_b[i] = vv_dot(tmp[i], block->psi[i], msize);
            rhox_b[i] = vv_dot(tmp[i], block->psix[i], msize) * 2.;
            rhoy_b[i] = vv_dot(tmp[i], block->psiy[i], msize) * 2.;
            rhoz_b[i] = vv_dot(tmp[i], block->psiz[i], msize) * 2.;
            sigma_b[i] = (rhox_b[i] * rhox_b[i] + rhoy_b[i] * rhoy_b[i] + rhoz_b[i] * rhoz_b[i]);
            sigma_ab[i] = (rhox[i] * rhox_b[i] + rhoy[i] * rhoy_b[i] + rhoz[i] * rhoz_b[i]);
            lapl_b[i] = 2 * (vv_dot(tmp[i], block->psixx[i], msize) +
                             vv_dot(tmp[i], block->psiyy[i], msize) +
                             vv_dot(tmp[i], block->psizz[i], msize));
        }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, grid_num, msize, msize, 1., block->psix[0], msize,
                    d_matrix_b[0], msize, 0., tmp[0], msize);
        for (int i = 0; i < grid_num; i++)
        {
            lapl_b[i] += 2 * vv_dot(tmp[i], block->psix[i], msize);
            tau_b[i] = 0.5 * vv_dot(tmp[i], block->psix[i], msize);
        }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, grid_num, msize, msize, 1., block->psiy[0], msize,
                    d_matrix_b[0], msize, 0., tmp[0], msize);
        for (int i = 0; i < grid_num; i++)
        {
            lapl_b[i] += 2 * vv_dot(tmp[i], block->psiy[i], msize);
            tau_b[i] += 0.5 * vv_dot(tmp[i], block->psiy[i], msize);
        }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, grid_num, msize, msize, 1., block->psiz[0], msize,
                    d_matrix_b[0], msize, 0., tmp[0], msize);
        for (int i = 0; i < grid_num; i++)
        {
            lapl_b[i] += 2 * vv_dot(tmp[i], block->psiz[i], msize);
            tau_b[i] += 0.5 * vv_dot(tmp[i], block->psiz[i], msize);
        }
    }
    else
        for (int i = 0; i < grid_num; i++)
        {
            rho[i] *= 2.;
            sigma[i] *= 4.;
            tau[i] *= 2;
            lapl[i] *= 2;
        }
    free_matrix(tmp);
    return;
}

double cal_dft_r(const Matrix d_matrix, Matrix xc_matrix, dft_info dft)
{
    double xc_energy = 0.;
    grids_info grids = dft->grids;
    const mol_info mol = dft->mol;
    int msize = get_mol_msize(dft->mol);
    memset(xc_matrix[0], 0, sizeof(double) * msize * msize);
#pragma omp parallel
    {
        double rho[250], rhox[250], rhoy[250], rhoz[250],
            sigma[250], lapla[250], laplb[250], taua[250];
        Vector vxcx, vxcy, vxcz;
        double exc_tol[250], exc_rho_tol[250], exc_sigma_tol[250], exc_lapl_tol[250], exc_tau_tol[250];
        double exc_tmp[250], exc_rho_tmp[250], exc_sigma_tmp[250], exc_lapl_tmp[250], exc_tau_tmp[250];
        double exc_rho_a[250];
        double exc_sigma_ax[250];
        double exc_sigma_ay[250];
        double exc_sigma_az[250];
        double e_block = 0.;
        Matrix xc_matrix_thread = malloc_matrix(msize, msize);
        int gx, gy;
        double w;
        vxcx = malloc_vector(msize);
        vxcy = malloc_vector(msize);
        vxcz = malloc_vector(msize);
#pragma omp for schedule(dynamic, 1) nowait
        for (int n = 0; n < grids->block_num; n++)
        {
            memset(exc_tol, 0, sizeof(double) * 250);
            memset(exc_rho_tol, 0, sizeof(double) * 250);
            memset(exc_sigma_tol, 0, sizeof(double) * 250);
            memset(exc_lapl_tol, 0, sizeof(double) * 250);
            memset(exc_tau_tol, 0, sizeof(double) * 250);
            grid_block block = &grids->blocks[n];
            Matrix xc_local = malloc_matrix(block->bas_num, block->bas_num);
            Matrix vxc;
            Matrix d_matrix_local;
            if (block->bas_num == 0)
                continue;
            d_matrix_local = malloc_matrix(block->bas_num, block->bas_num);
            vxc = malloc_matrix(block->grid_num, block->bas_num);
            cal_block_bas(n, grids);
            for (int i = 0; i < block->bas_num; i++)
            {
                gx = block->local2global[i];
                for (int j = 0; j <= i; j++)
                {
                    gy = block->local2global[j];
                    d_matrix_local[i][j] = d_matrix_local[j][i] = d_matrix[gx][gy];
                }
            }
            switch (dft->max_type)
            {
            case 1:
                cal_lda_block(d_matrix_local, NULL, block, block->bas_num, rho, NULL);
                break;
            case 2:
                cal_gga_block(d_matrix_local, NULL, block, block->bas_num, rho, NULL, rhox, rhoy, rhoz, NULL, NULL, NULL, sigma, NULL, NULL);
                break;
            case 3:
                cal_meta_gga_block(d_matrix_local, NULL, block, block->bas_num, rho, NULL, rhox, rhoy, rhoz, NULL, NULL, NULL, sigma, NULL, NULL,
                                   lapla, NULL, taua, NULL);
                break;
            default:
                break;
            }
            for (int j = 0; j < dft->func_num; j++)
            {
                switch (dft->type[j])
                {
                case 1:
                    xc_lda_exc_vxc(&dft->xc_func[j], block->grid_num, rho, exc_tmp, exc_rho_tmp);
                    for (int k = 0; k < block->grid_num; k++)
                    {
                        exc_tol[k] += dft->xc_frac[j] * exc_tmp[k];
                        exc_rho_tol[k] += dft->xc_frac[j] * exc_rho_tmp[k];
                    }
                    break;
                case 2:
                    xc_gga_exc_vxc(&dft->xc_func[j], block->grid_num, rho, sigma, exc_tmp, exc_rho_tmp,
                                   exc_sigma_tmp);
                    for (int k = 0; k < block->grid_num; k++)
                    {
                        exc_tol[k] += dft->xc_frac[j] * exc_tmp[k];
                        exc_rho_tol[k] += dft->xc_frac[j] * exc_rho_tmp[k];
                        exc_sigma_tol[k] += dft->xc_frac[j] * exc_sigma_tmp[k];
                    }
                    break;
                case 3:
                    xc_mgga_exc_vxc(&dft->xc_func[j], block->grid_num, rho, sigma, lapla, taua, exc_tmp,
                                    exc_rho_tmp, exc_sigma_tmp, exc_lapl_tmp, exc_tau_tmp);
                    for (int k = 0; k < block->grid_num; k++)
                    {
                        exc_tol[k] += dft->xc_frac[j] * exc_tmp[k];
                        exc_rho_tol[k] += dft->xc_frac[j] * exc_rho_tmp[k];
                        exc_sigma_tol[k] += dft->xc_frac[j] * exc_sigma_tmp[k];
                        exc_lapl_tol[k] += dft->xc_frac[j] * exc_lapl_tmp[k];
                        exc_tau_tol[k] += dft->xc_frac[j] * exc_tau_tmp[k];
                    }
                    break;
                default:
                    break;
                }
            }
            for (int i = 0; i < block->grid_num; i++)
            {
                w = grids->grids[block->grid_list[i]].w;
                e_block += w * exc_tol[i] * (rho[i]);
                exc_rho_a[i] = 0.5 * exc_rho_tol[i] * w;
            }
            if (dft->max_type >= 2)
                for (int i = 0; i < block->grid_num; i++)
                {
                    w = grids->grids[block->grid_list[i]].w;
                    exc_sigma_ax[i] = 4. * exc_sigma_tol[i] * w * rhox[i];
                    exc_sigma_ay[i] = 4. * exc_sigma_tol[i] * w * rhoy[i];
                    exc_sigma_az[i] = 4. * exc_sigma_tol[i] * w * rhoz[i];
                }
            for (int i = 0; i < block->grid_num; i++)
            {
                Vector psi = block->psi[i];
                Vector psix = block->psix[i];
                Vector psiy = block->psiy[i];
                Vector psiz = block->psiz[i];
                for (int j = 0; j < block->bas_num; j++)
                {
                    vxc[i][j] = exc_rho_a[i] * psi[j];
                }
                // GGA type DFT
                if (dft->max_type >= 2)
                    for (int j = 0; j < block->bas_num; j++)
                    {
                        vxc[i][j] += exc_sigma_ax[i] * psix[j] + exc_sigma_ay[i] * psiy[j] + exc_sigma_az[i] * psiz[j];
                    }
            }
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, block->bas_num, block->bas_num, block->grid_num, 1., vxc[0], block->bas_num, block->psi[0],
                        block->bas_num, 0., xc_local[0], block->bas_num);
            // meta-GGA type DFT
            if (dft->max_type >= 3)
            {
                memset(vxcx, 0, sizeof(double) * msize);
                memset(vxcy, 0, sizeof(double) * msize);
                memset(vxcz, 0, sizeof(double) * msize);
                for (int a = 0; a < block->grid_num; a++)
                {
                    Vector psi = block->psi[a];
                    Vector psix = block->psix[a];
                    Vector psiy = block->psiy[a];
                    Vector psiz = block->psiz[a];
                    for (int j = 0; j < block->bas_num; j++)
                    {
                        vxcx[j] += 0.25 * w * exc_tau_tol[a] * psix[j];
                        vxcy[j] += 0.25 * w * exc_tau_tol[a] * psiy[j];
                        vxcz[j] += 0.25 * w * exc_tau_tol[a] * psiz[j];
                    }
                    for (int j = 0; j < block->bas_num; j++)
                    {
                        for (int k = 0; k < block->bas_num; k++)
                        {
                            xc_local[k][j] += (vxcx[j] * psix[k] + vxcy[j] * psiy[k] + vxcz[j] * psiz[k]);
                            xc_local[j][k] += (vxcx[k] * psix[j] + vxcy[k] * psiy[j] + vxcz[k] * psiz[j]);
                        }
                    }
                }
            }
            for (int i = 0; i < block->bas_num; i++)
            {
                gx = block->local2global[i];
                for (int j = 0; j < block->bas_num; j++)
                {
                    gy = block->local2global[j];
                    xc_matrix_thread[gx][gy] += xc_local[i][j];
                }
            }
            clear_block_bas(block);
            free_matrix(xc_local);
            free_matrix(d_matrix_local);
            free_matrix(vxc);
        }
#pragma omp critical
        {
            for (int i = 0; i < msize * msize; i++)
            {
                xc_matrix[0][i] += xc_matrix_thread[0][i] * 2;
            }
            xc_energy += e_block;
        }
        free_matrix(xc_matrix_thread);
        free(vxcx);
        free(vxcy);
        free(vxcz);
    }
    for (int i = 0; i < msize; i++)
        for (int j = 0; j < i; j++)
        {
            xc_matrix[i][j] = xc_matrix[j][i] = 0.5 * (xc_matrix[i][j] + xc_matrix[j][i]);
        }
    return xc_energy;
}

double cal_dft_u(const Matrix d_matrix, const Matrix d_matrix_b, Matrix xc_matrix, Matrix xc_matrix_b, dft_info dft)
{
    double xc_energy = 0.;
    grids_info grids = dft->grids;
    const mol_info mol = dft->mol;
    int msize = get_mol_msize(dft->mol);
    memset(xc_matrix[0], 0, sizeof(double) * msize * msize);
    memset(xc_matrix_b[0], 0, sizeof(double) * msize * msize);
#pragma omp parallel
    {
        double rho[250], rhox[250], rhoy[250], rhoz[250],
            rhob[250], rhoxb[250], rhoyb[250], rhozb[250],
            sigmaaa[250], sigmabb[250], sigmaab[250], lapla[250],
            laplb[250], taua[250], taub[250];
        double rho_tol[500], sigma_tol[750];
        double exc_tol[250], exc_rho_tol[500], exc_sigma_tol[750];
        double exc_tmp[250], exc_rho_tmp[500], exc_sigma_tmp[750];
        double exc_rho_a[250], exc_rho_b[250];
        double exc_sigma_ax[250], exc_sigma_bx[250];
        double exc_sigma_ay[250], exc_sigma_by[250];
        double exc_sigma_az[250], exc_sigma_bz[250];
        double e_block = 0.;
        Matrix xc_matrix_thread = malloc_matrix(msize, msize);
        Matrix xc_matrix_thread_b = malloc_matrix(msize, msize);
        int gx, gy;
        double w;
#pragma omp for schedule(dynamic, 1) nowait
        for (int n = 0; n < grids->block_num; n++)
        {
            memset(exc_tol, 0, sizeof(double) * 250);
            memset(exc_rho_tol, 0, sizeof(double) * 500);
            memset(exc_sigma_tol, 0, sizeof(double) * 750);
            grid_block block = &grids->blocks[n];
            Matrix xc_local = malloc_matrix(block->bas_num, block->bas_num);
            Matrix xc_local_b = malloc_matrix(block->bas_num, block->bas_num);
            Matrix vxc, vxcb;
            Matrix d_matrix_local, d_matrix_b_local;
            if (block->bas_num == 0)
                continue;
            d_matrix_local = malloc_matrix(block->bas_num, block->bas_num);
            d_matrix_b_local = malloc_matrix(block->bas_num, block->bas_num);
            vxc = malloc_matrix(block->grid_num, block->bas_num);
            vxcb = malloc_matrix(block->grid_num, block->bas_num);
            cal_block_bas(n, grids);
            for (int i = 0; i < block->bas_num; i++)
            {
                gx = block->local2global[i];
                for (int j = 0; j <= i; j++)
                {
                    gy = block->local2global[j];
                    d_matrix_local[i][j] = d_matrix_local[j][i] = d_matrix[gx][gy];
                    d_matrix_b_local[i][j] = d_matrix_b_local[j][i] = d_matrix_b[gx][gy];
                }
            }
            switch (dft->max_type)
            {
            case 1:
                cal_lda_block(d_matrix_local, d_matrix_b_local, block, block->bas_num, rho, rhob);
                for (int a = 0; a < block->grid_num; a++)
                {
                    rho_tol[2 * a] = rho[a];
                    rho_tol[2 * a + 1] = rhob[a];
                }
                break;
            case 2:
                cal_gga_block(d_matrix_local, d_matrix_b_local, block, block->bas_num, rho, rhob, rhox, rhoy, rhoz, rhoxb,
                              rhoyb, rhozb, sigmaaa, sigmabb, sigmaab);
                for (int a = 0; a < block->grid_num; a++)
                {
                    rho_tol[2 * a] = rho[a];
                    rho_tol[2 * a + 1] = rhob[a];
                    sigma_tol[3 * a] = sigmaaa[a];
                    sigma_tol[3 * a + 1] = sigmaab[a];
                    sigma_tol[3 * a + 2] = sigmabb[a];
                }
                break;
            default:
                break;
            }
            for (int j = 0; j < dft->func_num; j++)
            {
                switch (dft->type[j])
                {
                case 1:
                    xc_lda_exc_vxc(&dft->xc_func_pol[j], block->grid_num, rho_tol, exc_tmp, exc_rho_tmp);
                    for (int k = 0; k < block->grid_num; k++)
                    {
                        exc_tol[k] += dft->xc_frac[j] * exc_tmp[k];
                        exc_rho_tol[2 * k] += dft->xc_frac[j] * exc_rho_tmp[2 * k];
                        exc_rho_tol[2 * k + 1] += dft->xc_frac[j] * exc_rho_tmp[2 * k + 1];
                    }
                    break;
                case 2:
                    xc_gga_exc_vxc(&dft->xc_func_pol[j], block->grid_num, rho_tol, sigma_tol, exc_tmp, exc_rho_tmp,
                                   exc_sigma_tmp);
                    for (int k = 0; k < block->grid_num; k++)
                    {
                        exc_tol[k] += dft->xc_frac[j] * exc_tmp[k];
                        exc_rho_tol[2 * k] += dft->xc_frac[j] * exc_rho_tmp[2 * k];
                        exc_rho_tol[2 * k + 1] += dft->xc_frac[j] * exc_rho_tmp[2 * k + 1];
                        exc_sigma_tol[3 * k] += dft->xc_frac[j] * exc_sigma_tmp[3 * k];
                        exc_sigma_tol[3 * k + 1] += dft->xc_frac[j] * exc_sigma_tmp[3 * k + 1];
                        exc_sigma_tol[3 * k + 2] += dft->xc_frac[j] * exc_sigma_tmp[3 * k + 2];
                    }
                    break;
                default:
                    break;
                }
            }
            for (int i = 0; i < block->grid_num; i++)
            {
                w = grids->grids[block->grid_list[i]].w;
                e_block += w * exc_tol[i] * (rho[i] + rhob[i]);
                exc_rho_a[i] = 0.5 * exc_rho_tol[2 * i + 0] * w;
                exc_rho_b[i] = 0.5 * exc_rho_tol[2 * i + 1] * w;
            }
            if (dft->max_type >= 2)
                for (int i = 0; i < block->grid_num; i++)
                {
                    w = grids->grids[block->grid_list[i]].w;
                    exc_sigma_ax[i] = 2. * exc_sigma_tol[3 * i + 0] * w * rhox[i] + exc_sigma_tol[3 * i + 1] * w * rhoxb[i];
                    exc_sigma_ay[i] = 2. * exc_sigma_tol[3 * i + 0] * w * rhoy[i] + exc_sigma_tol[3 * i + 1] * w * rhoyb[i];
                    exc_sigma_az[i] = 2. * exc_sigma_tol[3 * i + 0] * w * rhoz[i] + exc_sigma_tol[3 * i + 1] * w * rhozb[i];
                    exc_sigma_bx[i] = 2. * exc_sigma_tol[3 * i + 2] * w * rhoxb[i] + exc_sigma_tol[3 * i + 1] * w * rhox[i];
                    exc_sigma_by[i] = 2. * exc_sigma_tol[3 * i + 2] * w * rhoyb[i] + exc_sigma_tol[3 * i + 1] * w * rhoy[i];
                    exc_sigma_bz[i] = 2. * exc_sigma_tol[3 * i + 2] * w * rhozb[i] + exc_sigma_tol[3 * i + 1] * w * rhoz[i];
                }
            for (int i = 0; i < block->grid_num; i++)
            {
                Vector psi = block->psi[i];
                Vector psix = block->psix[i];
                Vector psiy = block->psiy[i];
                Vector psiz = block->psiz[i];
                for (int j = 0; j < block->bas_num; j++)
                {
                    vxc[i][j] = exc_rho_a[i] * psi[j];
                    vxcb[i][j] = exc_rho_b[i] * psi[j];
                }
                if (dft->max_type >= 2)
                    for (int j = 0; j < block->bas_num; j++)
                    {
                        vxc[i][j] += exc_sigma_ax[i] * psix[j] + exc_sigma_ay[i] * psiy[j] + exc_sigma_az[i] * psiz[j];
                        vxcb[i][j] += exc_sigma_bx[i] * psix[j] + exc_sigma_by[i] * psiy[j] + exc_sigma_bz[i] * psiz[j];
                    }
            }
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, block->bas_num, block->bas_num, block->grid_num, 1., vxc[0], block->bas_num, block->psi[0],
                        block->bas_num, 0., xc_local[0], block->bas_num);
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, block->bas_num, block->bas_num, block->grid_num, 1., vxcb[0], block->bas_num, block->psi[0],
                        block->bas_num, 0., xc_local_b[0], block->bas_num);
            for (int i = 0; i < block->bas_num; i++)
            {
                gx = block->local2global[i];
                for (int j = 0; j < block->bas_num; j++)
                {
                    gy = block->local2global[j];
                    xc_matrix_thread[gx][gy] += xc_local[i][j];
                    xc_matrix_thread_b[gx][gy] += xc_local_b[i][j];
                }
            }
            clear_block_bas(block);
            free_matrix(xc_local);
            free_matrix(xc_local_b);
            free_matrix(d_matrix_local);
            free_matrix(d_matrix_b_local);
            free_matrix(vxc);
            free_matrix(vxcb);
        }
#pragma omp critical
        {
            for (int i = 0; i < msize * msize; i++)
            {
                xc_matrix[0][i] += xc_matrix_thread[0][i] * 2;
                xc_matrix_b[0][i] += xc_matrix_thread_b[0][i] * 2;
            }
            xc_energy += e_block;
        }
        free_matrix(xc_matrix_thread);
        free_matrix(xc_matrix_thread_b);
    }
    for (int i = 0; i < msize; i++)
        for (int j = 0; j < i; j++)
        {
            xc_matrix[i][j] = xc_matrix[j][i] = 0.5 * (xc_matrix[i][j] + xc_matrix[j][i]);
            xc_matrix_b[i][j] = xc_matrix_b[j][i] = 0.5 * (xc_matrix_b[i][j] + xc_matrix_b[j][i]);
        }
    return xc_energy;
}

double cal_dft_xc(const Matrix d_matrix, const Matrix d_matrix_b, Matrix xc_matrix, Matrix xc_matrix_b, dft_info dft)
{
    double t1 = omp_get_wtime();
    double exc = 0.;
    if (d_matrix_b == NULL || xc_matrix_b == NULL)
        exc = cal_dft_r(d_matrix, xc_matrix, dft);
    else
        exc = cal_dft_u(d_matrix, d_matrix_b, xc_matrix, xc_matrix_b, dft);
#ifdef DEBUG
    printf("  Time of V_xc matrix's calculation is %12.6f s.\n", omp_get_wtime() - t1);
#endif
    return exc;
}

double cal_dft_disp(int start, int end, dft_info dft)
{
    assert(dft->disp != NULL);
    return cal_disp_correct(start, end, dft->disp);
}

double cal_exc_u(const Matrix d_matrix, const Matrix d_matrix_b, dft_info dft)
{
    double xc_energy = 0.;
    grids_info grids = dft->grids;
    const mol_info mol = dft->mol;
    int msize = get_mol_msize(mol);
#pragma omp parallel
    {
        double rho[250], rhox[250], rhoy[250], rhoz[250],
            rhob[250], rhoxb[250], rhoyb[250], rhozb[250],
            sigmaaa[250], sigmabb[250], sigmaab[250], lapla[250],
            laplb[250], taua[250], taub[250];
        double rho_tol[500], sigma_tol[750];
        double exc_tol[250];
        double exc_tmp[250];
        double e_block = 0.;
        int gx, gy;
        double w;
#pragma omp for schedule(dynamic, 1) nowait
        for (int n = 0; n < grids->block_num; n++)
        {
            memset(exc_tol, 0, sizeof(double) * 250);
            grid_block block = &grids->blocks[n];
            Matrix d_matrix_local, d_matrix_b_local;
            if (block->bas_num == 0)
                continue;
            d_matrix_local = malloc_matrix(block->bas_num, block->bas_num);
            d_matrix_b_local = malloc_matrix(block->bas_num, block->bas_num);
            cal_block_bas(n, grids);
            for (int i = 0; i < block->bas_num; i++)
            {
                gx = block->local2global[i];
                for (int j = 0; j <= i; j++)
                {
                    gy = block->local2global[j];
                    d_matrix_local[i][j] = d_matrix_local[j][i] = d_matrix[gx][gy];
                    d_matrix_b_local[i][j] = d_matrix_b_local[j][i] = d_matrix_b[gx][gy];
                }
            }
            switch (dft->max_type)
            {
            case 1:
                cal_lda_block(d_matrix_local, d_matrix_b_local, block, block->bas_num, rho, rhob);
                for (int a = 0; a < block->grid_num; a++)
                {
                    rho_tol[2 * a] = rho[a];
                    rho_tol[2 * a + 1] = rhob[a];
                }
                break;
            case 2:
                cal_gga_block(d_matrix_local, d_matrix_b_local, block, block->bas_num, rho, rhob, rhox, rhoy, rhoz, rhoxb,
                              rhoyb, rhozb, sigmaaa, sigmabb, sigmaab);
                for (int a = 0; a < block->grid_num; a++)
                {
                    rho_tol[2 * a] = rho[a];
                    rho_tol[2 * a + 1] = rhob[a];
                    sigma_tol[3 * a] = sigmaaa[a];
                    sigma_tol[3 * a + 1] = sigmaab[a];
                    sigma_tol[3 * a + 2] = sigmabb[a];
                }
                break;
            default:
                break;
            }
            for (int j = 0; j < dft->func_num; j++)
            {
                switch (dft->type[j])
                {
                case 1:
                    xc_lda_exc(&dft->xc_func_pol[j], block->grid_num, rho_tol, exc_tmp);
                    for (int k = 0; k < block->grid_num; k++)
                        exc_tol[k] += dft->xc_frac[j] * exc_tmp[k];
                    break;
                case 2:
                    xc_gga_exc(&dft->xc_func_pol[j], block->grid_num, rho_tol, sigma_tol, exc_tmp);
                    for (int k = 0; k < block->grid_num; k++)
                        exc_tol[k] += dft->xc_frac[j] * exc_tmp[k];
                    break;
                default:
                    break;
                }
            }
            for (int i = 0; i < block->grid_num; i++)
            {
                w = grids->grids[block->grid_list[i]].w;
                e_block += w * exc_tol[i] * (rho[i] + rhob[i]);
            }
            clear_block_bas(block);
            free_matrix(d_matrix_local);
            free_matrix(d_matrix_b_local);
        }
#pragma omp critical
        {
            xc_energy += e_block;
        }
    }
    return xc_energy;
}

double cal_exc_r(const Matrix d_matrix, dft_info dft)
{
    double xc_energy = 0.;
    grids_info grids = dft->grids;
    const mol_info mol = dft->mol;
    int msize = get_mol_msize(mol);
#pragma omp parallel
    {
        double rho[250], rhox[250], rhoy[250], rhoz[250],
            sigma[250], lapla[250], laplb[250], taua[250];
        double exc_tol[250];
        double exc_tmp[250];
        double e_block = 0.;
        Matrix xc_matrix_thread = malloc_matrix(msize, msize);
        int gx, gy;
        double w;
#pragma omp for schedule(dynamic, 1) nowait
        for (int n = 0; n < grids->block_num; n++)
        {
            memset(exc_tol, 0, sizeof(double) * 250);
            grid_block block = &grids->blocks[n];
            Matrix d_matrix_local;
            if (block->bas_num == 0)
                continue;
            d_matrix_local = malloc_matrix(block->bas_num, block->bas_num);
            cal_block_bas(n, grids);
            for (int i = 0; i < block->bas_num; i++)
            {
                gx = block->local2global[i];
                for (int j = 0; j <= i; j++)
                {
                    gy = block->local2global[j];
                    d_matrix_local[i][j] = d_matrix_local[j][i] = d_matrix[gx][gy];
                }
            }
            switch (dft->max_type)
            {
            case 1:
                cal_lda_block(d_matrix_local, NULL, block, block->bas_num, rho, NULL);
                break;
            case 2:
                cal_gga_block(d_matrix_local, NULL, block, block->bas_num, rho, NULL, rhox, rhoy, rhoz, NULL, NULL, NULL, sigma, NULL, NULL);
                break;
            default:
                break;
            }
            for (int j = 0; j < dft->func_num; j++)
            {
                switch (dft->type[j])
                {
                case 1:
                    xc_lda_exc(&dft->xc_func[j], block->grid_num, rho, exc_tmp);
                    for (int k = 0; k < block->grid_num; k++)
                        exc_tol[k] += dft->xc_frac[j] * exc_tmp[k];
                    break;
                case 2:
                    xc_gga_exc(&dft->xc_func[j], block->grid_num, rho, sigma, exc_tmp);
                    for (int k = 0; k < block->grid_num; k++)
                        exc_tol[k] += dft->xc_frac[j] * exc_tmp[k];
                    break;
                default:
                    break;
                }
            }
            for (int i = 0; i < block->grid_num; i++)
            {
                w = grids->grids[block->grid_list[i]].w;
                e_block += w * exc_tol[i] * (rho[i]);
            }
            clear_block_bas(block);
            free_matrix(d_matrix_local);
        }
#pragma omp critical
        {
            xc_energy += e_block;
        }
        free_matrix(xc_matrix_thread);
    }
    return xc_energy;
}

double cal_dft_exc(const Matrix d_matrix, const Matrix d_matrix_b, dft_info dft)
{
    double t1 = omp_get_wtime();
    double exc = 0.;
    if (d_matrix_b == NULL)
        exc = cal_exc_r(d_matrix, dft);
    else
        exc = cal_exc_u(d_matrix, d_matrix_b, dft);
#ifdef DEBUG
    printf("  Time of E_xc calculation is %12.6f s.\n", omp_get_wtime() - t1);
#endif
    return exc;
}

void cal_exc_u_grid(Vector rho_xc, const Matrix d_matrix, const Matrix d_matrix_b, dft_info dft)
{
    grids_info grids = dft->grids;
    const mol_info mol = dft->mol;
    int msize = get_mol_msize(mol);
#pragma omp parallel
    {
        double rho[250], rhox[250], rhoy[250], rhoz[250],
            rhob[250], rhoxb[250], rhoyb[250], rhozb[250],
            sigmaaa[250], sigmabb[250], sigmaab[250], lapla[250],
            laplb[250], taua[250], taub[250];
        double rho_tol[500], sigma_tol[750];
        double exc_tol[250];
        double exc_tmp[250];
        int gx, gy;
        double w;
#pragma omp for schedule(dynamic, 1) nowait
        for (int n = 0; n < grids->block_num; n++)
        {
            memset(exc_tol, 0, sizeof(double) * 250);
            grid_block block = &grids->blocks[n];
            Matrix d_matrix_local, d_matrix_b_local;
            if (block->bas_num == 0)
                continue;
            d_matrix_local = malloc_matrix(block->bas_num, block->bas_num);
            d_matrix_b_local = malloc_matrix(block->bas_num, block->bas_num);
            cal_block_bas(n, grids);
            for (int i = 0; i < block->bas_num; i++)
            {
                gx = block->local2global[i];
                for (int j = 0; j <= i; j++)
                {
                    gy = block->local2global[j];
                    d_matrix_local[i][j] = d_matrix_local[j][i] = d_matrix[gx][gy];
                    d_matrix_b_local[i][j] = d_matrix_b_local[j][i] = d_matrix_b[gx][gy];
                }
            }
            switch (dft->max_type)
            {
            case 1:
                cal_lda_block(d_matrix_local, d_matrix_b_local, block, block->bas_num, rho, rhob);
                for (int a = 0; a < block->grid_num; a++)
                {
                    rho_tol[2 * a] = rho[a];
                    rho_tol[2 * a + 1] = rhob[a];
                }
                break;
            case 2:
                cal_gga_block(d_matrix_local, d_matrix_b_local, block, block->bas_num, rho, rhob, rhox, rhoy, rhoz, rhoxb,
                              rhoyb, rhozb, sigmaaa, sigmabb, sigmaab);
                for (int a = 0; a < block->grid_num; a++)
                {
                    rho_tol[2 * a] = rho[a];
                    rho_tol[2 * a + 1] = rhob[a];
                    sigma_tol[3 * a] = sigmaaa[a];
                    sigma_tol[3 * a + 1] = sigmaab[a];
                    sigma_tol[3 * a + 2] = sigmabb[a];
                }
                break;
            default:
                break;
            }
            for (int j = 0; j < dft->func_num; j++)
            {
                switch (dft->type[j])
                {
                case 1:
                    xc_lda_exc(&dft->xc_func_pol[j], block->grid_num, rho_tol, exc_tmp);
                    for (int k = 0; k < block->grid_num; k++)
                        exc_tol[k] += dft->xc_frac[j] * exc_tmp[k];
                    break;
                case 2:
                    xc_gga_exc(&dft->xc_func_pol[j], block->grid_num, rho_tol, sigma_tol, exc_tmp);
                    for (int k = 0; k < block->grid_num; k++)
                        exc_tol[k] += dft->xc_frac[j] * exc_tmp[k];
                    break;
                default:
                    break;
                }
            }
            for (int i = 0; i < block->grid_num; i++)
            {
                int g_id = block->grid_list[i];
                w = grids->grids[g_id].w;
                rho_xc[g_id] += w * exc_tol[i] * (rho[i] + rhob[i]);
            }
            clear_block_bas(block);
            free_matrix(d_matrix_local);
            free_matrix(d_matrix_b_local);
        }
    }
    return;
}

void cal_exc_r_grid(Vector rho_xc, const Matrix d_matrix, dft_info dft)
{
    grids_info grids = dft->grids;
    const mol_info mol = dft->mol;
    int msize = get_mol_msize(mol);
#pragma omp parallel
    {
        double rho[250], rhox[250], rhoy[250], rhoz[250],
            sigma[250], lapla[250], laplb[250], taua[250];
        double exc_tol[250];
        double exc_tmp[250];
        Matrix xc_matrix_thread = malloc_matrix(msize, msize);
        int gx, gy;
        double w;
#pragma omp for schedule(dynamic, 1) nowait
        for (int n = 0; n < grids->block_num; n++)
        {
            memset(exc_tol, 0, sizeof(double) * 250);
            grid_block block = &grids->blocks[n];
            Matrix d_matrix_local;
            if (block->bas_num == 0)
                continue;
            d_matrix_local = malloc_matrix(block->bas_num, block->bas_num);
            cal_block_bas(n, grids);
            for (int i = 0; i < block->bas_num; i++)
            {
                gx = block->local2global[i];
                for (int j = 0; j <= i; j++)
                {
                    gy = block->local2global[j];
                    d_matrix_local[i][j] = d_matrix_local[j][i] = d_matrix[gx][gy];
                }
            }
            switch (dft->max_type)
            {
            case 1:
                cal_lda_block(d_matrix_local, NULL, block, block->bas_num, rho, NULL);
                break;
            case 2:
                cal_gga_block(d_matrix_local, NULL, block, block->bas_num, rho, NULL, rhox, rhoy, rhoz, NULL, NULL, NULL, sigma, NULL, NULL);
                break;
            default:
                break;
            }
            for (int j = 0; j < dft->func_num; j++)
            {
                switch (dft->type[j])
                {
                case 1:
                    xc_lda_exc(&dft->xc_func[j], block->grid_num, rho, exc_tmp);
                    for (int k = 0; k < block->grid_num; k++)
                        exc_tol[k] += dft->xc_frac[j] * exc_tmp[k];
                    break;
                case 2:
                    xc_gga_exc(&dft->xc_func[j], block->grid_num, rho, sigma, exc_tmp);
                    for (int k = 0; k < block->grid_num; k++)
                        exc_tol[k] += dft->xc_frac[j] * exc_tmp[k];
                    break;
                default:
                    break;
                }
            }
            for (int i = 0; i < block->grid_num; i++)
            {
                int g_id = block->grid_list[i];
                w = grids->grids[g_id].w;
                rho_xc[g_id] = w * exc_tol[i] * (rho[i]);
            }
            clear_block_bas(block);
            free_matrix(d_matrix_local);
        }
        free_matrix(xc_matrix_thread);
    }
    return;
}

void cal_exc_grid(Vector rho_xc, const Matrix d_matrix, const Matrix d_matrix_b, dft_info dft)
{
    if (d_matrix_b == NULL)
        return cal_exc_r_grid(rho_xc, d_matrix, dft);
    return cal_exc_u_grid(rho_xc, d_matrix, d_matrix_b, dft);
}

void cal_dft_grad_r(Vector grad, const Matrix d_matrix, dft_info dft)
{
    grids_info grids = dft->grids;
    const mol_info mol = dft->mol;
    const atm_info atm = mol->atm;
    int natm = atm->natm;
    int msize = get_mol_msize(dft->mol);
    Vector tmp_grad = malloc_vector(3 * natm);
#pragma omp parallel
    {
        double rho[250], rhox[250], rhoy[250], rhoz[250],
            sigma[250], lapla[250], laplb[250], taua[250];
        Vector vxcx, vxcy, vxcz;
        double exc_tol[250], exc_rho_tol[250], exc_sigma_tol[250], exc_lapl_tol[250], exc_tau_tol[250];
        double exc_tmp[250], exc_rho_tmp[250], exc_sigma_tmp[250], exc_lapl_tmp[250], exc_tau_tmp[250];
        double exc_rho_a[250];
        double exc_sigma_ax[250];
        double exc_sigma_ay[250];
        double exc_sigma_az[250];
        double e_block = 0.;
        int gx, gy;
        double w;
        vxcx = malloc_vector(msize);
        vxcy = malloc_vector(msize);
        vxcz = malloc_vector(msize);
#pragma omp for schedule(dynamic, 1) nowait
        for (int n = 0; n < grids->block_num; n++)
        {
            memset(exc_tol, 0, sizeof(double) * 250);
            memset(exc_rho_tol, 0, sizeof(double) * 250);
            memset(exc_sigma_tol, 0, sizeof(double) * 250);
            memset(exc_lapl_tol, 0, sizeof(double) * 250);
            memset(exc_tau_tol, 0, sizeof(double) * 250);
            grid_block block = &grids->blocks[n];
            Matrix vxc;
            Matrix d_matrix_local;
            if (block->bas_num == 0)
                continue;
            d_matrix_local = malloc_matrix(block->bas_num, block->bas_num);
            vxc = malloc_matrix(block->grid_num, block->bas_num);
            cal_block_bas(n, grids);
            for (int i = 0; i < block->bas_num; i++)
            {
                gx = block->local2global[i];
                for (int j = 0; j <= i; j++)
                {
                    gy = block->local2global[j];
                    d_matrix_local[i][j] = d_matrix_local[j][i] = d_matrix[gx][gy];
                }
            }
            switch (dft->max_type)
            {
            case 1:
                cal_lda_block(d_matrix_local, NULL, block, block->bas_num, rho, NULL);
                break;
            case 2:
                cal_gga_block(d_matrix_local, NULL, block, block->bas_num, rho, NULL, rhox, rhoy, rhoz, NULL, NULL, NULL, sigma, NULL, NULL);
                break;
            case 3:
                cal_meta_gga_block(d_matrix_local, NULL, block, block->bas_num, rho, NULL, rhox, rhoy, rhoz, NULL, NULL, NULL, sigma, NULL, NULL,
                                   lapla, NULL, taua, NULL);
                break;
            default:
                break;
            }
            for (int j = 0; j < dft->func_num; j++)
            {
                switch (dft->type[j])
                {
                case 1:
                    xc_lda_exc_vxc(&dft->xc_func[j], block->grid_num, rho, exc_tmp, exc_rho_tmp);
                    for (int k = 0; k < block->grid_num; k++)
                    {
                        exc_tol[k] += dft->xc_frac[j] * exc_tmp[k];
                        exc_rho_tol[k] += dft->xc_frac[j] * exc_rho_tmp[k];
                    }
                    break;
                case 2:
                    xc_gga_exc_vxc(&dft->xc_func[j], block->grid_num, rho, sigma, exc_tmp, exc_rho_tmp,
                                   exc_sigma_tmp);
                    for (int k = 0; k < block->grid_num; k++)
                    {
                        exc_tol[k] += dft->xc_frac[j] * exc_tmp[k];
                        exc_rho_tol[k] += dft->xc_frac[j] * exc_rho_tmp[k];
                        exc_sigma_tol[k] += dft->xc_frac[j] * exc_sigma_tmp[k];
                    }
                    break;
                case 3:
                    xc_mgga_exc_vxc(&dft->xc_func[j], block->grid_num, rho, sigma, lapla, taua, exc_tmp,
                                    exc_rho_tmp, exc_sigma_tmp, exc_lapl_tmp, exc_tau_tmp);
                    for (int k = 0; k < block->grid_num; k++)
                    {
                        exc_tol[k] += dft->xc_frac[j] * exc_tmp[k];
                        exc_rho_tol[k] += dft->xc_frac[j] * exc_rho_tmp[k];
                        exc_sigma_tol[k] += dft->xc_frac[j] * exc_sigma_tmp[k];
                        exc_lapl_tol[k] += dft->xc_frac[j] * exc_lapl_tmp[k];
                        exc_tau_tol[k] += dft->xc_frac[j] * exc_tau_tmp[k];
                    }
                    break;
                default:
                    break;
                }
            }
            for (int i = 0; i < block->grid_num; i++)
            {
                w = grids->grids[block->grid_list[i]].w;
                e_block += w * exc_tol[i] * (rho[i]);
                exc_rho_a[i] = 0.5 * exc_rho_tol[i] * w;
            }
            if (dft->max_type >= 2)
                for (int i = 0; i < block->grid_num; i++)
                {
                    w = grids->grids[block->grid_list[i]].w;
                    exc_sigma_ax[i] = 4. * exc_sigma_tol[i] * w * rhox[i];
                    exc_sigma_ay[i] = 4. * exc_sigma_tol[i] * w * rhoy[i];
                    exc_sigma_az[i] = 4. * exc_sigma_tol[i] * w * rhoz[i];
                }
            for (int i = 0; i < block->grid_num; i++)
            {
                Vector psi = block->psi[i];
                Vector psix = block->psix[i];
                Vector psiy = block->psiy[i];
                Vector psiz = block->psiz[i];
                for (int j = 0; j < block->bas_num; j++)
                {
                    vxc[i][j] = exc_rho_a[i] * psi[j];
                }
                // GGA type DFT
                if (dft->max_type >= 2)
                    for (int j = 0; j < block->bas_num; j++)
                    {
                        vxc[i][j] += exc_sigma_ax[i] * psix[j] + exc_sigma_ay[i] * psiy[j] + exc_sigma_az[i] * psiz[j];
                    }
            }
            for (int i = 0; i < block->bas_num; i++)
            {
                gx = block->local2global[i];
                for (int j = 0; j < block->bas_num; j++)
                {
                    gy = block->local2global[j];
                }
            }
            clear_block_bas(block);
            free_matrix(d_matrix_local);
            free_matrix(vxc);
        }
#pragma omp critical
        {
            for (int i = 0; i < 3 * natm; i++)
                grad[i] += tmp_grad[i] * 2;
        }
        free(vxcx);
        free(vxcy);
        free(vxcz);
    }
    return;
}
