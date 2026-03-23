#include "mol/xint.h"
static void overlap_VRR_0(Vector buf, int angular, double K_ab, double half_xi_,
                          double xi, double PA)
{
    buf[0] = K_ab * sqrt(2 * MY_PI * half_xi_);
    if (angular == 0)
        return;
    buf[1] = buf[0] * PA;
    // P345 (9,3.8) S_{i+1,0}=PA*S_{i,0}+i/(2p)*S_{i-1,j}
    for (int i = 2; i <= angular; i += 1)
    {
        buf[i] = PA * buf[i - 1] + (i - 1) * half_xi_ * buf[i - 2];
    }
    return;
}
static void overlap_RR_0(Vector buf, int angular1, int angular2, double K_ab, double half_xi_,
                         double xi, double PA, double PB)
{
    int size_1 = angular1 + 1, size_2 = angular2 + 2;
    overlap_VRR_0(buf, angular1, K_ab, half_xi_, xi, PA);
    if (angular2 == 0)
        return;
    buf[size_1] = PB * buf[0];
    // P345 (9.3.9) S_{i,j+1}=PB*S_{i,j}+1/2p*(i*S_{i-1,j}+j*S_{i,j-1})
    for (int i = 2; i <= angular2; i++)
    {
        buf[size_1 * i] = PB * buf[size_1 * (i - 1)] + (i - 1) * half_xi_ * buf[size_1 * (i - 2)];
    }
    for (int i = 1; i <= angular1; i++)
    {
        buf[size_1 + i] = PB * buf[i] + i * half_xi_ * (buf[i - 1]);
        for (int j = 2; j <= angular2; j++)
        {
            buf[size_1 * j + i] = PB * buf[size_1 * (j - 1) + i] + half_xi_ * (i * buf[size_1 * (j - 1) + i - 1] + (j - 1) * buf[size_1 * (j - 2) + i]);
        }
    }
    return;
}
static void overlap_RR(Vector buf, double *cache, int angular1, int angular2, Vector K_ab,
                       double half_xi_, double xi, Vector PA, Vector PB, double coe)
{
    int ang_len = (angular1 + 1) * (angular2 + 1);
    overlap_RR_0(&cache[ang_len * 0], angular1, angular2, K_ab[0], half_xi_, xi, PA[0], PB[0]);
    overlap_RR_0(&cache[ang_len * 1], angular1, angular2, K_ab[1], half_xi_, xi, PA[1], PB[1]);
    overlap_RR_0(&cache[ang_len * 2], angular1, angular2, K_ab[2], half_xi_, xi, PA[2], PB[2]);
    for (int lx1 = angular1, uu = 0; lx1 >= 0; lx1--)
        for (int ly1 = angular1; ly1 >= 0; ly1--)
        {
            if (lx1 + ly1 > angular1)
                continue;
            int lz1 = angular1 - lx1 - ly1;
            for (int lx2 = angular2; lx2 >= 0; lx2--)
                for (int ly2 = angular2; ly2 >= 0; ly2--)
                {
                    if (lx2 + ly2 > angular2)
                        continue;
                    int lz2 = angular2 - lx2 - ly2;
                    int index1 = lx1 + lx2 * (1 + angular1);
                    int index2 = ly1 + ly2 * (1 + angular1);
                    int index3 = lz1 + lz2 * (1 + angular1);
                    buf[uu] += coe * cache[ang_len * 0 + index1] * cache[ang_len * 1 + index2] * cache[ang_len * 2 + index3];
                    ++uu;
                }
        }
    return;
}
static void kinetic_RR_0(Vector buf1, Vector buf2, int angular1, int angular2, double K_ab,
                         double half_xi_, double xi, double PA, double PB, double alpha_, double beta_)
{
    int size_1 = angular1 + 1;
    overlap_RR_0(buf1, angular1, angular2, K_ab, half_xi_, xi, PA, PB);
    buf2[0] = (alpha_ - 2 * alpha_ * alpha_ * (PA * PA + half_xi_)) * buf1[0];
    if (angular1 != 0)
    {
        buf2[1] = PA * buf2[0] + 4 * beta_ * half_xi_ * alpha_ * buf1[1];
        for (int i = 2; i <= angular1; i++)
        {
            buf2[i] = PA * buf2[i - 1] + (i - 1) * half_xi_ * buf2[i - 2] + 2 * beta_ * half_xi_ * (2 * alpha_ * buf1[i] - (i - 1) * buf1[i - 2]);
        }
    }
    if (angular2 == 0)
        return;
    buf2[size_1] = PB * buf2[0] + 4 * alpha_ * beta_ * half_xi_ * buf1[size_1];
    for (int i = 2; i <= angular2; i++)
    {
        buf2[size_1 * i] = PB * buf2[size_1 * (i - 1)] + (i - 1) * half_xi_ * buf2[size_1 * (i - 2)] +
                           2 * half_xi_ * alpha_ * (2 * beta_ * buf1[size_1 * i] - (i - 1) * buf1[size_1 * (i - 2)]);
    }
    for (int i = 1; i <= angular1; i++)
    {
        buf2[size_1 + i] = PB * buf2[i] + half_xi_ * (i * buf2[i - 1]) + 4 * alpha_ * beta_ * half_xi_ * buf1[size_1 + i];
        for (int j = 2; j <= angular2; j++)
        {
            buf2[size_1 * j + i] = PB * buf2[size_1 * (j - 1) + i] + half_xi_ * (i * buf2[size_1 * (j - 1) + i - 1] + (j - 1) * buf2[size_1 * (j - 2) + i]) + 2 * alpha_ * half_xi_ * (2 * beta_ * buf1[size_1 * j + i] - (j - 1) * buf1[size_1 * (j - 2) + i]);
        }
    }
    return;
}
static void kinetic_RR(Vector buf, Vector cache, int angular1, int angular2, Vector K_ab,
                       double half_xi_, double xi, Vector PA, Vector PB, double alpha_, double beta_, double coe)
{
    int ang_len = (angular1 + 1) * (angular2 + 1);
    kinetic_RR_0(&cache[ang_len * 0], &cache[ang_len * 1], angular1, angular2, K_ab[0], half_xi_, xi, PA[0],
                 PB[0], alpha_, beta_);
    kinetic_RR_0(&cache[ang_len * 2], &cache[ang_len * 3], angular1, angular2, K_ab[1], half_xi_, xi, PA[1],
                 PB[1], alpha_, beta_);
    kinetic_RR_0(&cache[ang_len * 4], &cache[ang_len * 5], angular1, angular2, K_ab[2], half_xi_, xi, PA[2],
                 PB[2], alpha_, beta_);
    for (int lx1 = angular1, uu = 0; lx1 >= 0; lx1--)
        for (int ly1 = angular1; ly1 >= 0; ly1--)
        {
            if (lx1 + ly1 > angular1)
                continue;
            int lz1 = angular1 - lx1 - ly1;
            for (int lx2 = angular2; lx2 >= 0; lx2--)
                for (int ly2 = angular2; ly2 >= 0; ly2--)
                {
                    if (lx2 + ly2 > angular2)
                        continue;
                    int lz2 = angular2 - lx2 - ly2;
                    int index1 = lx1 + lx2 * (1 + angular1);
                    int index2 = ly1 + ly2 * (1 + angular1);
                    int index3 = lz1 + lz2 * (1 + angular1);
                    buf[uu] += coe * (cache[ang_len * 1 + index1] * cache[ang_len * 2 + index2] * cache[ang_len * 4 + index3] + cache[ang_len * 0 + index1] * cache[ang_len * 3 + index2] * cache[ang_len * 4 + index3] + cache[ang_len * 0 + index1] * cache[ang_len * 2 + index2] * cache[ang_len * 5 + index3]);
                    ++uu;
                }
        }
}
void xint_overlap(Vector buf, const int shlij, xint_info xint)
{
    const mol_info mol = xint->mol;
    const shell_pair shp = &mol->pair->shp[shlij];
    int di = shp->di, dj = shp->dj;
    int angi = shp->li, angj = shp->lj;
    int buf_len = 0.25 * (angi + 1) * (angi + 2) * (angj + 1) * (angj + 2);
    const double *coei, *coej;
    const double *expi, *expj;
    int shli = shp->shli, shlj = shp->shlj;
    const Vector K_ab = shp->K_ab;
    coei = get_bas_coe(shli, mol->bas);
    coej = get_bas_coe(shlj, mol->bas);
    expi = get_bas_exp(shli, mol->bas);
    expj = get_bas_exp(shlj, mol->bas);
    memset(buf, 0, sizeof(double) * buf_len);
    for (int i = 0, uu = 0; i < di; ++i)
        for (int j = 0; j < dj; ++j, uu++)
        {
            overlap_RR(buf, xint->cache, angi, angj, &K_ab[4 * uu], shp->half_xi_[uu], shp->xi[uu],
                       &shp->PA[3 * uu], &shp->PB[3 * uu], coei[i] * coej[j]);
        }
    return;
}
void xint_kinetic(Vector buf, const int shlij, xint_info xint)
{
    const mol_info mol = xint->mol;
    const shell_pair shp = &mol->pair->shp[shlij];
    int di = shp->di, dj = shp->dj;
    int angi = shp->li, angj = shp->lj;
    int buf_len = 0.25 * (angi + 1) * (angi + 2) * (angj + 1) * (angj + 2);
    const double *coei, *coej;
    const double *expi, *expj;
    int shli = shp->shli, shlj = shp->shlj;
    const Vector K_ab = shp->K_ab;
    coei = get_bas_coe(shli, mol->bas);
    coej = get_bas_coe(shlj, mol->bas);
    expi = get_bas_exp(shli, mol->bas);
    expj = get_bas_exp(shlj, mol->bas);
    memset(buf, 0, sizeof(double) * buf_len);
    for (int i = 0, uu = 0; i < di; ++i)
        for (int j = 0; j < dj; ++j, ++uu)
        {
            kinetic_RR(buf, xint->cache, angi, angj, &K_ab[4 * uu], shp->half_xi_[uu], shp->xi[uu],
                       &shp->PA[3 * uu], &shp->PB[3 * uu], expi[i], expj[j], coei[i] * coej[j]);
        }
    return;
}
static void diff1_RR(Vector buf, Vector cache, int angular1, int angular2, Vector K_ab,
                     double half_xi_, double xi, Vector PA, Vector PB, double alpha_, double beta_, const double coe12)
{
    int size1 = angular1 + 1;
    int ang_len = (angular1 + 1) * (angular2 + 2);
    int ang_len0 = (angular1 + 1) * (angular2 + 1);
    Vector tmp_cache, work_cache;
    overlap_RR_0(&cache[ang_len * 0], angular1, angular2 + 1, K_ab[0], half_xi_, xi, PA[0], PB[0]);
    overlap_RR_0(&cache[ang_len * 1], angular1, angular2 + 1, K_ab[1], half_xi_, xi, PA[1], PB[1]);
    overlap_RR_0(&cache[ang_len * 2], angular1, angular2 + 1, K_ab[2], half_xi_, xi, PA[2], PB[2]);
    tmp_cache = cache;
    work_cache = cache + 3 * ang_len;
    /// P348 (9.3.30) D_{ij}^{1}=2aS_{i+1,j}-iS_{i-1,j}
    for (int j = 0; j < angular1 + 1; j++)
        for (int k = 0; k < angular2 + 1; k++)
        {
            int index1 = j + k * (angular1 + 1);       // D_{i,j+1}
            int index2 = j + (k + 1) * (angular1 + 1); // S_{i+1,j}
            int index3 = j + (k - 1) * (angular1 + 1); // S_{i-1,j}
            if (k > 0)
            {
                work_cache[index1 + 0 * ang_len0] = 2 * beta_ * tmp_cache[index2 + ang_len * 0] - k * tmp_cache[index3 + ang_len * 0];
                work_cache[index1 + 1 * ang_len0] = 2 * beta_ * tmp_cache[index2 + ang_len * 1] - k * tmp_cache[index3 + ang_len * 1];
                work_cache[index1 + 2 * ang_len0] = 2 * beta_ * tmp_cache[index2 + ang_len * 2] - k * tmp_cache[index3 + ang_len * 2];
            }
            else
            {
                work_cache[index1 + 0 * ang_len0] = 2 * beta_ * tmp_cache[index2 + ang_len * 0];
                work_cache[index1 + 1 * ang_len0] = 2 * beta_ * tmp_cache[index2 + ang_len * 1];
                work_cache[index1 + 2 * ang_len0] = 2 * beta_ * tmp_cache[index2 + ang_len * 2];
            }
        }
    for (int lx1 = angular1, uu = 0; lx1 >= 0; lx1--)
        for (int ly1 = angular1; ly1 >= 0; ly1--)
        {
            if (lx1 + ly1 > angular1)
                continue;
            int lz1 = angular1 - lx1 - ly1;
            for (int lx2 = angular2; lx2 >= 0; lx2--)
                for (int ly2 = angular2; ly2 >= 0; ly2--)
                {
                    if (lx2 + ly2 > angular2)
                        continue;
                    int lz2 = angular2 - lx2 - ly2;
                    int index1 = lx1 + lx2 * (1 + angular1);
                    int index2 = ly1 + ly2 * (1 + angular1);
                    int index3 = lz1 + lz2 * (1 + angular1);
                    buf[uu * 3 + 0] += coe12 * work_cache[ang_len0 * 0 + index1] * cache[ang_len * 1 + index2] * cache[ang_len * 2 + index3];
                    buf[uu * 3 + 1] += coe12 * cache[ang_len * 0 + index1] * work_cache[ang_len0 * 1 + index2] * cache[ang_len * 2 + index3];
                    buf[uu * 3 + 2] += coe12 * cache[ang_len * 0 + index1] * cache[ang_len * 1 + index2] * work_cache[ang_len0 * 2 + index3];
                    uu++;
                }
        }
    return;
}

void xint_overlap_grad(Vector buf, const int shlij, xint_info xint)
{
    const mol_info mol = xint->mol;
    const shell_pair shp = &mol->pair->shp[shlij];
    int di = shp->di, dj = shp->dj;
    int angi = shp->li, angj = shp->lj;
    int buf_len = 0.75 * (angi + 1) * (angi + 2) * (angj + 1) * (angj + 2);
    const double *coei, *coej;
    const double *expi, *expj;
    int shli = shp->shli, shlj = shp->shlj;
    const Vector K_ab = shp->K_ab;
    coei = get_bas_coe(shli, mol->bas);
    coej = get_bas_coe(shlj, mol->bas);
    expi = get_bas_exp(shli, mol->bas);
    expj = get_bas_exp(shlj, mol->bas);
    memset(buf, 0, sizeof(double) * buf_len);
    for (int i = 0, uu = 0; i < di; ++i)
        for (int j = 0; j < dj; ++j, ++uu)
        {
            diff1_RR(buf, xint->cache, angi, angj, &K_ab[4 * uu], shp->half_xi_[uu], shp->xi[uu],
                     &shp->PA[3 * uu], &shp->PB[3 * uu], expi[i], expj[j], coei[i] * coej[j]);
        }
    return;
}
extern void up_ang_index(int ang, int ind, int *indx, int *indy, int *indz);
extern void low_ang_index(int ang, int ind, int *indx, int *indy, int *indz);
extern void *get_llist(const int ang);
void xint_kinetic_grad(Vector buf, const int shlij, xint_info xint)
{
    const mol_info mol = xint->mol;
    const shell_pair shp = &mol->pair->shp[shlij];
    int di = shp->di, dj = shp->dj;
    int angi = shp->li, angj = shp->lj;
    int si = xint_gtolen(angi);
    int sj = xint_gtolen(angj);
    int(*ang_l)[3];
    int idx, idy, idz;
    int buff0_len = 0.75 * (angi + 2) * (angi + 3) * (angj + 1) * (angj + 2);
    const double *coei, *coej;
    const double *expi, *expj;
    int shli = shp->shli, shlj = shp->shlj;
    const Vector K_ab = shp->K_ab;
    Vector buff0 = malloc_vector(buff0_len);
    Vector buff1 = malloc_vector(buff0_len);
    coei = get_bas_coe(shli, mol->bas);
    coej = get_bas_coe(shlj, mol->bas);
    expi = get_bas_exp(shli, mol->bas);
    expj = get_bas_exp(shlj, mol->bas);
    for (int i = 0, uu = 0; i < di; ++i)
        for (int j = 0; j < dj; ++j, ++uu)
        {
            kinetic_RR(buff0, xint->cache, angi + 1, angj, &K_ab[4 * uu], shp->half_xi_[uu], shp->xi[uu],
                       &shp->PA[3 * uu], &shp->PB[3 * uu], expi[i], expj[j], 2 * expi[i] * coei[i] * coej[j]);
            if (angi >= 1)
                kinetic_RR(buff1, xint->cache, angi - 1, angj, &K_ab[4 * uu], shp->half_xi_[uu], shp->xi[uu],
                           &shp->PA[3 * uu], &shp->PB[3 * uu], expi[i], expj[j], coei[i] * coej[j]);
        }
    for (int i = 0, uu = 0; i < si; i++)
        for (int j = 0; j < sj; j++, uu++)
        {
            up_ang_index(angi, i, &idx, &idy, &idz);
            buf[3 * uu + 0] = -buff0[idx * sj + j];
            buf[3 * uu + 1] = -buff0[idy * sj + j];
            buf[3 * uu + 2] = -buff0[idz * sj + j];
        }
    if (angi >= 1)
    {
        ang_l = get_llist(angi);
        for (int i = 0, uu = 0; i < si; i++)
            for (int j = 0; j < sj; j++, uu++)
            {
                low_ang_index(angi, i, &idx, &idy, &idz);
                buf[3 * uu + 0] += ang_l[i][0] * buff1[idx * sj + j];
                buf[3 * uu + 1] += ang_l[i][1] * buff1[idy * sj + j];
                buf[3 * uu + 2] += ang_l[i][2] * buff1[idz * sj + j];
            }
    }
    free(buff0);
    free(buff1);
    return;
}
static void overlap_RR_polar(Vector buf, double *cache, int angular1, int angular2, Vector K_ab,
                             double half_xi_, double xi, Vector PA, Vector PB, Vector B, double coe)
{
    int ang_len = (angular1 + 1) * (angular2 + 2);
    overlap_RR_0(&cache[ang_len * 0], angular1, angular2 + 1, K_ab[0], half_xi_, xi, PA[0], PB[0]);
    overlap_RR_0(&cache[ang_len * 1], angular1, angular2 + 1, K_ab[1], half_xi_, xi, PA[1], PB[1]);
    overlap_RR_0(&cache[ang_len * 2], angular1, angular2 + 1, K_ab[2], half_xi_, xi, PA[2], PB[2]);
    for (int lx1 = angular1, uu = 0; lx1 >= 0; lx1--)
        for (int ly1 = angular1; ly1 >= 0; ly1--)
        {
            if (lx1 + ly1 > angular1)
                continue;
            int lz1 = angular1 - lx1 - ly1;
            for (int lx2 = angular2; lx2 >= 0; lx2--)
                for (int ly2 = angular2; ly2 >= 0; ly2--)
                {
                    if (lx2 + ly2 > angular2)
                        continue;
                    int lz2 = angular2 - lx2 - ly2;
                    int index1 = lx1 + lx2 * (1 + angular1);
                    int index2 = ly1 + ly2 * (1 + angular1);
                    int index3 = lz1 + lz2 * (1 + angular1);
                    int index1_ = lx1 + (lx2 + 1) * (1 + angular1);
                    int index2_ = ly1 + (ly2 + 1) * (1 + angular1);
                    int index3_ = lz1 + (lz2 + 1) * (1 + angular1);
                    double int_x = B[0] * cache[index1] + cache[index1_];
                    double int_y = B[1] * cache[ang_len + index2] + cache[ang_len + index2_];
                    double int_z = B[2] * cache[ang_len * 2 + index3] + cache[ang_len * 2 + index3_];
                    buf[uu + 0] += coe * int_x * cache[ang_len * 1 + index2] * cache[ang_len * 2 + index3];
                    buf[uu + 1] += coe * int_y * cache[ang_len * 0 + index1] * cache[ang_len * 2 + index3];
                    buf[uu + 2] += coe * int_z * cache[ang_len * 0 + index1] * cache[ang_len * 1 + index2];
                    uu += 3;
                }
        }
    return;
}
void xint_polar(Vector buf, const int shlij, xint_info xint)
{
    const mol_info mol = xint->mol;
    const shell_pair shp = &mol->pair->shp[shlij];
    int di = shp->di, dj = shp->dj;
    int angi = shp->li, angj = shp->lj;
    int buf_len = 0.75 * (angi + 1) * (angi + 2) * (angj + 1) * (angj + 2);
    const double *coei, *coej;
    const double *expi, *expj;
    int shli = shp->shli, shlj = shp->shlj;
    const Vector K_ab = shp->K_ab;
    const Vector B = get_atm_coord(mol->bas(ATOM_IND, shlj), mol->atm);
    coei = get_bas_coe(shli, mol->bas);
    coej = get_bas_coe(shlj, mol->bas);
    expi = get_bas_exp(shli, mol->bas);
    expj = get_bas_exp(shlj, mol->bas);
    memset(buf, 0, sizeof(double) * buf_len);
    for (int i = 0, uu = 0; i < di; ++i)
        for (int j = 0; j < dj; ++j, uu++)
        {
            overlap_RR_polar(buf, xint->cache, angi, angj, &K_ab[4 * uu], shp->half_xi_[uu], shp->xi[uu],
                             &shp->PA[3 * uu], &shp->PB[3 * uu], B, coei[i] * coej[j]);
        }
    return;
}

static void diff2_RR(Vector buf, Vector cache, int angular1, int angular2, Vector K_ab,
                     double half_xi_, double xi, Vector PA, Vector PB, double alpha_, double beta_, double coe)
{
    int size1 = angular1 + 1;
    int ang_len = (angular1 + 1) * (angular2 + 3);
    int ang_len1 = (angular1 + 1) * (angular2 + 2);
    int ang_len0 = (angular1 + 1) * (angular2 + 1);
    Vector tmp_cache, work_cache;
    overlap_RR_0(&cache[ang_len * 0], angular1, angular2 + 2, K_ab[0], half_xi_, xi, PA[0], PB[0]);
    overlap_RR_0(&cache[ang_len * 1], angular1, angular2 + 2, K_ab[1], half_xi_, xi, PA[1], PB[1]);
    overlap_RR_0(&cache[ang_len * 2], angular1, angular2 + 2, K_ab[2], half_xi_, xi, PA[2], PB[2]);
    tmp_cache = cache;
    work_cache = cache + 3 * ang_len;
    /// P348 (9.3.30) D_{ij}^{1}=2aS_{i+1,j}-iS_{i-1,j}
    for (int j = 0; j < angular1; j++)
        for (int k = 0; k < angular2 + 1; k++)
        {
            int index1 = j + k * (angular1 + 2);       // D^2_{i,j}
            int index2 = j + (k + 1) * (angular1 + 3); // D^1_{i+1,j}
            int index3 = j + (k - 1) * (angular1 + 3); // D^1_{i-1,j}
            work_cache[index1 + 0 * ang_len1] = 2 * beta_ * tmp_cache[index2 + ang_len * 0] - j * tmp_cache[index3 + ang_len * 0];
            work_cache[index1 + 1 * ang_len1] = 2 * beta_ * tmp_cache[index2 + ang_len * 1] - j * tmp_cache[index3 + ang_len * 1];
            work_cache[index1 + 2 * ang_len1] = 2 * beta_ * tmp_cache[index2 + ang_len * 2] - j * tmp_cache[index3 + ang_len * 2];
        }
    tmp_cache = work_cache;
    work_cache = work_cache + 3 * ang_len1;
    for (int j = 0; j < angular1; j++)
        for (int k = 0; k < angular2; k++)
        {
            int index1 = j + k * (angular1 + 1);       // D^2_{i,j}
            int index2 = j + (k + 1) * (angular1 + 2); // D^1_{i+1,j}
            int index3 = j + (k - 1) * (angular1 + 2); // D^1_{i-1,j}
            work_cache[index1 + 0 * ang_len0] = 2 * beta_ * tmp_cache[index2 + ang_len * 0] - j * tmp_cache[index3 + ang_len * 0];
            work_cache[index1 + 1 * ang_len0] = 2 * beta_ * tmp_cache[index2 + ang_len * 1] - j * tmp_cache[index3 + ang_len * 1];
            work_cache[index1 + 2 * ang_len0] = 2 * beta_ * tmp_cache[index2 + ang_len * 2] - j * tmp_cache[index3 + ang_len * 2];
        }
    // xx, xy, xz, yy, yz, zz
    work_cache = cache + 3 * ang_len;
    for (int lx1 = angular1, uu = 0; lx1 >= 0; lx1--)
        for (int ly1 = angular1; ly1 >= 0; ly1--)
        {
            if (lx1 + ly1 > angular1)
                continue;
            int lz1 = angular1 - lx1 - ly1;
            for (int lx2 = angular2; lx2 >= 0; lx2--)
                for (int ly2 = angular2; ly2 >= 0; ly2--)
                {
                    if (lx2 + ly2 > angular2)
                        continue;
                    int lz2 = angular2 - lx2 - ly2;
                    int index1 = lx1 + lx2 * (1 + angular1);
                    int index2 = ly1 + ly2 * (1 + angular1);
                    int index3 = lz1 + lz2 * (1 + angular1);
                    buf[uu * 6 + 1] += coe * work_cache[ang_len1 * 0 + index1 + 1 * lx2] * work_cache[ang_len1 * 1 + index2 + 1 * ly2] * cache[ang_len * 2 + index3 + 2 * lz2];
                    buf[uu * 6 + 2] += coe * work_cache[ang_len * 0 + index1 + 1 * lx2] * cache[ang_len1 * 1 + index2 + 2 * ly2] * work_cache[ang_len1 * 2 + index3 + 1 * lz2];
                    buf[uu * 6 + 4] += coe * cache[ang_len * 0 + index1 + 2 * lx2] * work_cache[ang_len1 * 1 + index2 + 1 * ly2] * work_cache[ang_len1 * 2 + index3 + 1 * lz2];
                    uu++;
                }
        }
    work_cache += 3 * ang_len1;
    for (int lx1 = angular1, uu = 0; lx1 >= 0; lx1--)
        for (int ly1 = angular1; ly1 >= 0; ly1--)
        {
            if (lx1 + ly1 > angular1)
                continue;
            int lz1 = angular1 - lx1 - ly1;
            for (int lx2 = angular2; lx2 >= 0; lx2--)
                for (int ly2 = angular2; ly2 >= 0; ly2--)
                {
                    if (lx2 + ly2 > angular2)
                        continue;
                    int lz2 = angular2 - lx2 - ly2;
                    int index1 = lx1 + lx2 * (1 + angular1);
                    int index2 = ly1 + ly2 * (1 + angular1);
                    int index3 = lz1 + lz2 * (1 + angular1);
                    buf[uu * 6 + 0] += coe * work_cache[ang_len0 * 0 + index1 + 0 * lx2] * cache[ang_len * 1 + index2 + 2 * ly2] * cache[ang_len * 2 + index3 + 2 * lz2];
                    buf[uu * 6 + 3] += coe * cache[ang_len * 0 + index1 + 2 * lx2] * work_cache[ang_len0 * 1 + index2 + 0 * ly2] * cache[ang_len * 2 + index3 + 2 * lz2];
                    buf[uu * 6 + 5] += coe * cache[ang_len * 0 + index1 + 2 * lx2] * cache[ang_len * 1 + index2 + 2 * ly2] * work_cache[ang_len0 * 2 + index3 + 0 * lz2];
                    uu++;
                }
        }
    return;
}

static void diff2_KRR(Vector buf, Vector cache, int angular1, int angular2, Vector K_ab,
                      double half_xi_, double xi, Vector PA, Vector PB, double alpha_, double beta_, double coe)
{
    int size1 = angular1 + 1;
    int ang_len = (angular1 + 1) * (angular2 + 3);
    int ang_len1 = (angular1 + 1) * (angular2 + 2);
    int ang_len0 = (angular1 + 1) * (angular2 + 1);
    Vector tmp_cache, work_cache;
    overlap_RR_0(&cache[ang_len * 0], angular1, angular2 + 2, K_ab[0], half_xi_, xi, PA[0], PB[0]);
    overlap_RR_0(&cache[ang_len * 1], angular1, angular2 + 2, K_ab[1], half_xi_, xi, PA[1], PB[1]);
    overlap_RR_0(&cache[ang_len * 2], angular1, angular2 + 2, K_ab[2], half_xi_, xi, PA[2], PB[2]);
    tmp_cache = cache;
    work_cache = cache + 3 * ang_len;
    /// P348 (9.3.30) D_{ij}^{1}=2aS_{i+1,j}-iS_{i-1,j}
    for (int j = 0; j < angular1; j++)
        for (int k = 0; k < angular2 + 1; k++)
        {
            int index1 = j + k * (angular1 + 2);       // D^2_{i,j}
            int index2 = j + (k + 1) * (angular1 + 3); // D^1_{i+1,j}
            int index3 = j + (k - 1) * (angular1 + 3); // D^1_{i-1,j}
            work_cache[index1 + 0 * ang_len1] = 2 * beta_ * tmp_cache[index2 + ang_len * 0] - j * tmp_cache[index3 + ang_len * 0];
            work_cache[index1 + 1 * ang_len1] = 2 * beta_ * tmp_cache[index2 + ang_len * 1] - j * tmp_cache[index3 + ang_len * 1];
            work_cache[index1 + 2 * ang_len1] = 2 * beta_ * tmp_cache[index2 + ang_len * 2] - j * tmp_cache[index3 + ang_len * 2];
        }
    tmp_cache = work_cache;
    work_cache = work_cache + 3 * ang_len1;
    for (int j = 0; j < angular1; j++)
        for (int k = 0; k < angular2; k++)
        {
            int index1 = j + k * (angular1 + 1);       // D^2_{i,j}
            int index2 = j + (k + 1) * (angular1 + 2); // D^1_{i+1,j}
            int index3 = j + (k - 1) * (angular1 + 2); // D^1_{i-1,j}
            work_cache[index1 + 0 * ang_len0] = 2 * beta_ * tmp_cache[index2 + ang_len * 0] - j * tmp_cache[index3 + ang_len * 0];
            work_cache[index1 + 1 * ang_len0] = 2 * beta_ * tmp_cache[index2 + ang_len * 1] - j * tmp_cache[index3 + ang_len * 1];
            work_cache[index1 + 2 * ang_len0] = 2 * beta_ * tmp_cache[index2 + ang_len * 2] - j * tmp_cache[index3 + ang_len * 2];
        }
    // xx, xy, xz, yy, yz, zz
    work_cache = cache + 3 * ang_len;
    for (int lx1 = angular1, uu = 0; lx1 >= 0; lx1--)
        for (int ly1 = angular1; ly1 >= 0; ly1--)
        {
            if (lx1 + ly1 > angular1)
                continue;
            int lz1 = angular1 - lx1 - ly1;
            for (int lx2 = angular2; lx2 >= 0; lx2--)
                for (int ly2 = angular2; ly2 >= 0; ly2--)
                {
                    if (lx2 + ly2 > angular2)
                        continue;
                    int lz2 = angular2 - lx2 - ly2;
                    int index1 = lx1 + lx2 * (1 + angular1);
                    int index2 = ly1 + ly2 * (1 + angular1);
                    int index3 = lz1 + lz2 * (1 + angular1);
                    buf[uu * 6 + 1] += coe * work_cache[ang_len1 * 0 + index1 + 1 * lx2] * work_cache[ang_len1 * 1 + index2 + 1 * ly2] * cache[ang_len * 2 + index3 + 2 * lz2];
                    buf[uu * 6 + 2] += coe * work_cache[ang_len * 0 + index1 + 1 * lx2] * cache[ang_len1 * 1 + index2 + 2 * ly2] * work_cache[ang_len1 * 2 + index3 + 1 * lz2];
                    buf[uu * 6 + 4] += coe * cache[ang_len * 0 + index1 + 2 * lx2] * work_cache[ang_len1 * 1 + index2 + 1 * ly2] * work_cache[ang_len1 * 2 + index3 + 1 * lz2];
                    uu++;
                }
        }
    work_cache += 3 * ang_len1;
    for (int lx1 = angular1, uu = 0; lx1 >= 0; lx1--)
        for (int ly1 = angular1; ly1 >= 0; ly1--)
        {
            if (lx1 + ly1 > angular1)
                continue;
            int lz1 = angular1 - lx1 - ly1;
            for (int lx2 = angular2; lx2 >= 0; lx2--)
                for (int ly2 = angular2; ly2 >= 0; ly2--)
                {
                    if (lx2 + ly2 > angular2)
                        continue;
                    int lz2 = angular2 - lx2 - ly2;
                    int index1 = lx1 + lx2 * (1 + angular1);
                    int index2 = ly1 + ly2 * (1 + angular1);
                    int index3 = lz1 + lz2 * (1 + angular1);
                    buf[uu * 6 + 0] += coe * work_cache[ang_len0 * 0 + index1 + 0 * lx2] * cache[ang_len * 1 + index2 + 2 * ly2] * cache[ang_len * 2 + index3 + 2 * lz2];
                    buf[uu * 6 + 3] += coe * cache[ang_len * 0 + index1 + 2 * lx2] * work_cache[ang_len0 * 1 + index2 + 0 * ly2] * cache[ang_len * 2 + index3 + 2 * lz2];
                    buf[uu * 6 + 5] += coe * cache[ang_len * 0 + index1 + 2 * lx2] * cache[ang_len * 1 + index2 + 2 * ly2] * work_cache[ang_len0 * 2 + index3 + 0 * lz2];
                    uu++;
                }
        }
    return;
}

extern void xint_overlap_hess(Vector buf, const int shlij, xint_info xint)
{
    const mol_info mol = xint->mol;
    const shell_pair shp = &mol->pair->shp[shlij];
    int di = shp->di, dj = shp->dj;
    int angi = shp->li, angj = shp->lj;
    int buf_len = 1.5 * (angi + 1) * (angi + 2) * (angj + 1) * (angj + 2);
    const double *coei, *coej;
    const double *expi, *expj;
    int shli = shp->shli, shlj = shp->shlj;
    const Vector K_ab = shp->K_ab;
    coei = get_bas_coe(shli, mol->bas);
    coej = get_bas_coe(shlj, mol->bas);
    expi = get_bas_exp(shli, mol->bas);
    expj = get_bas_exp(shlj, mol->bas);
    for (int i = 0, uu = 0; i < di; ++i)
        for (int j = 0; j < dj; ++j, ++uu)
        {
            diff2_RR(buf, xint->cache, angi, angj, &K_ab[4 * uu], shp->half_xi_[uu], shp->xi[uu],
                     &shp->PA[3 * uu], &shp->PB[3 * uu], expi[i], expj[j], coei[i] * coej[j]);
        }
    return;
}

extern void xint_kinetic_hess(Vector buf, const int shlij, xint_info xint)
{
    const mol_info mol = xint->mol;
    const shell_pair shp = &mol->pair->shp[shlij];
    int di = shp->di, dj = shp->dj;
    int angi = shp->li, angj = shp->lj;
    int buf_len = 1.5 * (angi + 1) * (angi + 2) * (angj + 1) * (angj + 2);
    const double *coei, *coej;
    const double *expi, *expj;
    int shli = shp->shli, shlj = shp->shlj;
    const Vector K_ab = shp->K_ab;
    coei = get_bas_coe(shli, mol->bas);
    coej = get_bas_coe(shlj, mol->bas);
    expi = get_bas_exp(shli, mol->bas);
    expj = get_bas_exp(shlj, mol->bas);
    for (int i = 0, uu = 0; i < di; ++i)
        for (int j = 0; j < dj; ++j, ++uu)
        {
            diff2_KRR(buf, xint->cache, angi, angj, &K_ab[4 * uu], shp->half_xi_[uu], shp->xi[uu],
                      &shp->PA[3 * uu], &shp->PB[3 * uu], expi[i], expj[j], coei[i] * coej[j]);
        }
    return;
}

void cal_overlap_matrix(Matrix s_matrix, const mol_info mol)
{
    int msize = get_mol_msize(mol);
    const bas_info bas = mol->bas;
    const pair_info pair = mol->pair;
    double buf[1024];
    int shli, shlj;
    int pi, pj;
    int di, dj;
    int ci, cj;
    xint_info xint;
    xint = init_xint(mol);
    for (int i = 0; i < pair->len; i++)
    {
        xint_overlap(buf, i, xint);
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
                s_matrix[u + pi][v + pj] = buf[uu];
                s_matrix[v + pj][u + pi] = buf[uu];
            }
    }
    del_xint(xint);
    return;
}
void cal_project_matrix(Matrix p_matrix, const mol_info mol1, const mol_info mol2)
{
    int msize1 = get_mol_msize(mol1);
    int msize2 = get_mol_msize(mol2);
    int nbas1 = get_mol_nbas(mol1);
    int nbas2 = get_mol_nbas(mol2);
    const double *coei, *coej;
    const double *expi, *expj;
    double buf[1024];
    double xi, half_xi_, K_ab[3];
    double P[3], PA[3], PB[3], AB[3];
    Vector A, B;
    int buf_len;
    xint_info xint = init_xint(mol1);
    for (int i = 0; i < nbas1; i++)
    {
        int li = mol1->bas(ANGULAR_VAL, i);
        int di = mol1->bas(NPRIM_VAL, i);
        int pi = mol1->bas->shls_p[i];
        int si = xint_gtolen(li);
        A = get_atm_coord(mol1->bas(ATOM_IND, i), mol1->atm);
        for (int j = 0; j < nbas1; j++)
        {
            int lj = mol2->bas(ANGULAR_VAL, j);
            int dj = mol2->bas(NPRIM_VAL, j);
            int pj = mol2->bas->shls_p[j];
            int sj = xint_gtolen(lj);
            B = get_atm_coord(mol2->bas(ATOM_IND, j), mol2->atm);
            buf_len = 0.25 * (li + 1) * (li + 2) * (lj + 1) * (lj + 2);
            coei = get_bas_coe(i, mol1->bas);
            coej = get_bas_coe(j, mol2->bas);
            expi = get_bas_exp(i, mol1->bas);
            expj = get_bas_exp(j, mol2->bas);
            memset(buf, 0, sizeof(double) * buf_len);
            AB[0] = A[0] - B[0];
            AB[1] = A[1] - B[1];
            AB[2] = A[2] - B[2];
            for (int a = 0; a < di; a++)
                for (int b = 0; b < dj; b++)
                {
                    xi = expi[a] + expj[b];
                    half_xi_ = 0.5 / xi;
                    K_ab[0] = exp(-2 * AB[0] * AB[0] * half_xi_ * expi[a] * expj[b]);
                    K_ab[1] = exp(-2 * AB[1] * AB[1] * half_xi_ * expi[a] * expj[b]);
                    K_ab[2] = exp(-2 * AB[2] * AB[2] * half_xi_ * expi[a] * expj[b]);
                    P[0] = 2 * (A[0] * expi[a] + B[0] * expj[b]) * half_xi_;
                    P[1] = 2 * (A[1] * expi[a] + B[1] * expj[b]) * half_xi_;
                    P[2] = 2 * (A[2] * expi[a] + B[2] * expj[b]) * half_xi_;
                    PA[0] = P[0] - A[0];
                    PA[1] = P[1] - A[1];
                    PA[2] = P[2] - A[2];
                    PB[0] = P[0] - B[0];
                    PB[1] = P[1] - B[1];
                    PB[2] = P[2] - B[2];
                    overlap_RR(buf, xint->cache, li, lj, K_ab, half_xi_, xi, PA, PB, coei[i] * coej[j]);
                }
            for (int a = 0, uu = 0; a < di; a++)
                for (int b = 0; b < dj; b++, uu++)
                    p_matrix[pi + a][pj + b] = buf[uu];
        }
    }
    return;
}
void cal_overlap_grad_matrix(Matrix *s_matrix_grad, const mol_info mol)
{
    int msize = get_mol_msize(mol);
    const bas_info bas = mol->bas;
    const pair_info pair = mol->pair;
    double buf[1024];
    int shli, shlj;
    int pi, pj;
    int di, dj;
    int ci, cj;
    xint_info xint;
    xint = init_xint(mol);
    for (int i = 0; i < pair->len; i++)
    {
        xint_overlap_grad(buf, i, xint);
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
                s_matrix_grad[0][u + pi][v + pj] = buf[3 * uu];
                s_matrix_grad[0][v + pj][u + pi] = -buf[3 * uu];
                s_matrix_grad[1][u + pi][v + pj] = buf[3 * uu + 1];
                s_matrix_grad[1][v + pj][u + pi] = -buf[3 * uu + 1];
                s_matrix_grad[2][u + pi][v + pj] = buf[3 * uu + 2];
                s_matrix_grad[2][v + pj][u + pi] = -buf[3 * uu + 2];
            }
    }
    del_xint(xint);
    return;
}

void cal_kinetic_matrix(Matrix t_matrix, const mol_info mol)
{
    int msize = get_mol_msize(mol);
    const bas_info bas = mol->bas;
    const pair_info pair = mol->pair;
    double buf[1024];
    int shli, shlj;
    int pi, pj;
    int di, dj;
    int ci, cj;
    xint_info xint;
    xint = init_xint(mol);
    for (int i = 0; i < pair->len; i++)
    {
        xint_kinetic(buf, i, xint);
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
                t_matrix[u + pi][v + pj] = buf[uu];
                t_matrix[v + pj][u + pi] = buf[uu];
            }
    }
    del_xint(xint);
    return;
}
