#include "fock/dft.h"
#include "fock/dftd3.dat"
static void get_func_parm(int disp_info, int func_type, double *s6, double *alp, double *rs6,
                          double *s18, double *rs18);
static void get_coord(int start, int end, mol_info mol, double *coord);
static void D2_param(disp_info disp);
static double get_c6(int *mxc, Tensor4D c6ab[3], int ia, int ja, double in, double jn);
disp_info init_disp(const mol_info mol, int func_type, int disp_type)
{
    int line = 32385;
    disp_info disp = (disp_info)malloc(sizeof(struct DispInfo));
    int ai, aj;
    int ci, cj;
    Vector cab_p = cab;
    disp->disp_type = disp_type;
    disp->mol = mol;
    disp->cab[0] = malloc_tensor4d(max_ele, max_ele, max_c, max_c);
    disp->cab[1] = malloc_tensor4d(max_ele, max_ele, max_c, max_c);
    disp->cab[2] = malloc_tensor4d(max_ele, max_ele, max_c, max_c);
    disp->r0ab = malloc_matrix(max_ele, max_ele);
    disp->mxc = malloc_list(max_ele);
    for (int i = 0; i < line; i++, cab_p += 5)
    {
        ai = cab_p[1];
        aj = cab_p[2];
        ci = 0;
        cj = 0;
        while (ai > 100)
        {
            ai -= 100;
            ci++;
        }
        while (aj > 100)
        {
            aj -= 100;
            cj++;
        }
        ai -= 1;
        aj -= 1;
        disp->cab[0][ai][aj][ci][cj] = cab_p[0];
        disp->cab[1][ai][aj][ci][cj] = cab_p[3];
        disp->cab[2][ai][aj][ci][cj] = cab_p[4];
        disp->cab[0][aj][ai][cj][ci] = cab_p[0];
        disp->cab[1][aj][ai][cj][ci] = cab_p[3];
        disp->cab[2][aj][ai][cj][ci] = cab_p[4];
        if (ci > disp->mxc[ai])
            disp->mxc[ai] = ci;
        if (cj > disp->mxc[aj])
            disp->mxc[aj] = cj;
    }
    for (int i = 0, k = 0; i < max_ele; i++)
    {
        for (int j = 0; j <= i; j++, k++)
        {
            disp->r0ab[i][j] = R0ab[k] / 0.52917726;
            disp->r0ab[j][i] = R0ab[k] / 0.52917726;
        }
    }
    disp->func_type = func_type;
    get_func_parm(disp_type, func_type, &disp->s6, &disp->alp6, &disp->rs6, &disp->s8, &disp->rs8);
    disp->alp8 = disp->alp6 + 2.;
    disp->alp10 = disp->alp6 + 4.;
    disp->rs10 = disp->rs8;
    disp->alp10 = disp->alp8;
    disp->atm_c = (double *)malloc(sizeof(double) * get_mol_natm(mol));
    if (disp_type == DFT_D2)
        D2_param(disp);
    return disp;
}
void D2_param(disp_info disp)
{
    for (int i = 0; i < 86; i++)
        for (int j = 0; j <= i; j++)
        {
            disp->r0ab[i][j] = (D2_r0[i] + D2_r0[j]) / 0.52917726;
            disp->r0ab[j][i] = (D2_r0[i] + D2_r0[j]) / 0.52917726;
            disp->cab[0][j][i][0][0] = sqrt(D2_c6[i] * D2_c6[j]) * 17.3452656005951;
            disp->cab[0][i][j][0][0] = sqrt(D2_c6[i] * D2_c6[j]) * 17.3452656005951;
        }
}
void del_disp(disp_info disp)
{
    free(disp->atm_c);
    free(disp->mxc);
    free_matrix(disp->r0ab);
    free_tensor4d(disp->cab[0]);
    free_tensor4d(disp->cab[1]);
    free_tensor4d(disp->cab[2]);
    free(disp);
    return;
}
void get_func_parm(int disp_info, int func_type, double *s6, double *alp, double *rs6,
                   double *s18, double *rs18)
{
    switch (disp_info)
    {
    case DFT_D2:
        *rs6 = 1.1;
        *s18 = 0.0;
        *alp = 20.0;
        switch (func_type)
        {
        case BLYP:
            *s6 = 1.2;
            break;
        case B_P:
            *s6 = 1.05;
            break;
        case B97_D:
            *s6 = 1.;
            break;
        case TPSS:
            *s6 = 1.0;
            break;
        case B3LYP:
            *s6 = 1.05;
            break;
        case PBE0:
            *s6 = 0.6;
            break;
        case TPSS0:
            *s6 = 0.85;
            break;
        default:
            assert(1 < 0);
            break;
        }
        break;
    case DFT_D3:
        *s6 = 1.;
        *alp = 14;
        *rs18 = 1.;
        switch (func_type)
        {
        case BLYP:
            *rs6 = 0.999;
            *s18 = -1.957;
            *rs18 = 0.697;
            break;
        case B_P:
            *rs6 = 1.094;
            *s18 = 1.682;
            *rs18 = 0.697;
            break;
        case B97_D:
            *rs6 = 1.139;
            *s18 = 1.683;
            break;
        case PBE:
            *rs6 = 1.217;
            *s18 = 0.722;
            break;
        case PBE0:
            *rs6 = 1.287;
            *s18 = 0.928;
            break;
        case B3LYP:
            *rs6 = 1.261;
            *s18 = 1.703;
            break;
        case M06:
            *rs6 = 1.325;
            *s18 = 0.000;
            break;
        case M06X:
            *rs6 = 1.619;
            *s18 = 0.000;
            break;
        default:
            break;
        }
        break;
    case DFT_D3_BJ:
        *s6 = 1.;
        *alp = 14.;
        switch (func_type)
        {
        case B_P:
            *rs6 = 0.3946;
            *s18 = 3.2822;
            *rs18 = 4.8516;
            break;
        case BLYP:
            *rs6 = 0.4298;
            *s18 = 2.6996;
            *rs18 = 4.2359;
            break;
        case B97_D:
            *rs6 = 0.5545;
            *s18 = 2.2609;
            *rs18 = 3.2297;
            break;
        case PBE:
            *rs6 = 0.4289;
            *s18 = 0.7875;
            *rs18 = 4.4407;
            break;
        case B3LYP:
            *rs6 = 0.3981;
            *s18 = 1.9889;
            *rs18 = 4.4211;
            break;
        case PBE0:
            *rs6 = 0.4145;
            *s18 = 1.2177;
            *rs18 = 4.8593;
            break;
        default:
            break;
        }
        break;
    default:
        break;
    }
}

double D2_correct(int start, int end, disp_info disp)
{
    const mol_info mol = disp->mol;
    double e = 0.;
    double dx, dy, dz;
    double r6, r2, r;
    int ai, aj;
    double c6, damp6;
    double alp6 = disp->alp6, rs6 = disp->rs6;
    for (int i = start; i < end - 1; i++)
    {
        const double *A = get_atm_coord(i, mol->atm);
        for (int j = i + 1; j < end; j++)
        {
            const double *B = get_atm_coord(j, mol->atm);
            ai = mol->atm(ELEMENT_VAL, i) - 1;
            aj = mol->atm(ELEMENT_VAL, j) - 1;
            dx = A[0] - B[0];
            dy = A[1] - B[1];
            dz = A[2] - B[2];
            r2 = dx * dx + dy * dy + dz * dz;
            r = sqrt(r2);
            c6 = disp->cab[0][ai][aj][0][0];
            damp6 = 1. / (1. + exp(-alp6 * (r / (rs6 * disp->r0ab[ai][aj]) - 1.)));
            if (disp->func_type == B97_D)
            {
                damp6 = 1. / (1. + alp6 * pow(r / disp->r0ab[ai][aj], -12.));
            }
            r6 = r2 * r2 * r2;
            e += damp6 * c6 / r6;
        }
    }
    return -e * disp->s6;
}

double D3_correct(int start, int end, disp_info disp, int BJ)
{
    const mol_info mol = disp->mol;
    int ij, ik, jk;
    int natm = get_mol_natm(mol);
    double e = 0.;
    double dx, dy, dz;
    double r8, r6, r2, r, rr;
    int ai, aj;
    double c6, damp6, tmp;
    double c8, damp8, c9, tmp2;
    double *r2ab = (double *)malloc(sizeof(double) * natm * natm);
    double *c6ab = (double *)malloc(sizeof(double) * natm * natm);
    double *dmpab = (double *)malloc(sizeof(double) * natm * natm);
    int *if_abc = (int *)calloc(natm * natm, sizeof(int));
    double rav, t1, t2, t3, ang;
    double alp6 = disp->alp6, rs6 = disp->rs6, rs8 = disp->rs8, alp8 = disp->alp8;
    double s6 = disp->s6, s8 = disp->s8;
    get_coord(start, end, mol, disp->atm_c);
    double e6 = 0, e8 = 0;
    for (int i = start; i < end - 1; i++)
    {
        const double *A = get_atm_coord(i, mol->atm);
        for (int j = i + 1; j < end; j++)
        {
            const double *B = get_atm_coord(j, mol->atm);
            if (mol->atm(ELEMENT_VAL, i) == 0 || mol->atm(ELEMENT_VAL, j) == 0)
                continue;
            ai = mol->atm(ELEMENT_VAL, i) - 1;
            aj = mol->atm(ELEMENT_VAL, j) - 1;
            dx = A[0] - B[0];
            dy = A[1] - B[1];
            dz = A[2] - B[2];
            r2 = dx * dx + dy * dy + dz * dz;
            r = sqrt(r2);
            rr = disp->r0ab[ai][aj] / r;
            tmp = rs6 * rr;
            damp6 = 1. / (1 + 6 * pow(tmp, alp6));
            tmp = rs8 * rr;
            damp8 = 1. / (1 + 6 * pow(tmp, alp8));
            r6 = r2 * r2 * r2;
            r8 = r6 * r2;
            c6 = get_c6(disp->mxc, disp->cab, ai, aj, disp->atm_c[i], disp->atm_c[j]);
            c8 = 3. * c6 * r2r4[ai] * r2r4[aj];
            if (!BJ)
            {
                e += s6 * c6 * damp6 / r6 + s8 * c8 * damp8 / r8;
                e6 += c6 * damp6 / r6;
                e8 += c8 * damp8 / r8;
                // printf("%E %E\n",c6*damp6/r6,c8*damp8/r8);
            }
            else
            {
                tmp = sqrt(c8 / c6);
                e += s6 * c6 / (r6 + pow(rs6 * tmp + rs8, 6)) + s8 * c8 / (r8 + pow(rs6 * tmp + rs8, 8));
            }
            if (r2 < 1600)
            {
                if_abc[i * natm + j] = if_abc[j * natm + i] = 1;
                r2ab[i * natm + j] = r2ab[j * natm + i] = r2;
                c6ab[i * natm + j] = c6ab[j * natm + i] = c6;
                dmpab[i * natm + j] = dmpab[j * natm + i] = pow(1. / rr, 1 / 3.);
            }
        }
    }
    if (0)
        for (int i = start; i < end; i++)
            for (int j = start; j < i; j++)
            {
                ij = i * natm + j;
                if (if_abc[ij] == 0)
                    continue;
                for (int k = start; k < j; k++)
                {
                    ik = i * natm + k;
                    jk = j * natm + k;
                    if (if_abc[ik] == 0 || if_abc[jk] == 0)
                        continue;
                    rav = (4. / 3.) / (dmpab[ik] * dmpab[jk] * dmpab[ij]);
                    tmp = 1. / (1. + 6. * pow(rav, alp8));
                    c9 = c6ab[ij] * c6ab[ik] * c6ab[jk];
                    t1 = (r2ab[ij] + r2ab[jk] - r2ab[ik]);
                    t2 = (r2ab[ij] + r2ab[ik] - r2ab[jk]);
                    t3 = (r2ab[ik] + r2ab[jk] - r2ab[ij]);
                    tmp2 = r2ab[ij] * r2ab[jk] * r2ab[ik];
                    ang = (0.375 * t1 * t2 * t3 / tmp2 + 1.0) / pow(tmp2, 1.5);
                    e -= ang * tmp * c9;
                }
            }
    free(r2ab);
    free(c6ab);
    free(if_abc);
    free(dmpab);
    return -e;
}

void get_coord(int start, int end, mol_info mol, double *coord)
{
    double dx, dy, dz;
    double rr, r2, r, rco;
    double tmp;
    for (int i = start; i < end; i++)
    {
        tmp = 0.;
        const double *A = get_atm_coord(i, mol->atm);
        for (int j = start; j < end; j++)
        {
            const double *B = get_atm_coord(j, mol->atm);
            if (i == j)
                continue;
            dx = A[0] - B[0];
            dy = A[1] - B[1];
            dz = A[2] - B[2];
            r2 = dx * dx + dy * dy + dz * dz;
            if (r2 > 1600)
                continue;
            r = sqrt(r2);
            rco = rcov[mol->atm(ELEMENT_VAL, i) - 1] + rcov[mol->atm(ELEMENT_VAL, j) - 1];
            rr = rco / r;
            tmp += 1. / (1. + exp(-16. * (rr - 1.)));
        }
        coord[i] = tmp;
    }
    return;
}

double get_c6(int *mxc, Tensor4D c6ab[3], int ia, int ja, double in, double jn)
{
    double c6, cn1, cn2, c6_mem = -1E20;
    double r, r_sav = 1E20;
    double tmp, rsum = 0., csum = 0.;
    for (int i = 0; i <= mxc[ia]; i++)
    {
        for (int j = 0; j <= mxc[ja]; j++)
        {
            c6 = c6ab[0][ia][ja][i][j];
            if (c6 <= 0.)
                continue;
            cn1 = c6ab[1][ia][ja][i][j];
            cn2 = c6ab[2][ia][ja][i][j];
            // printf("%f %f\n",cn1,cn2);
            r = ((in - cn1) * (in - cn1) + (jn - cn2) * (jn - cn2));
            if (r_sav > r)
            {
                r_sav = r;
                c6_mem = c6;
            }
            tmp = exp(-4. * r);
            rsum = rsum + tmp;
            csum = csum + tmp * c6;
        }
    }
    if (rsum > 1E-10)
        c6 = csum / rsum;
    else
    {
        c6 = c6_mem;
    }
    return c6;
}

double cal_disp_correct(int start, int end, disp_info disp)
{
    switch (disp->disp_type)
    {
    case DFT_D2:
        return D2_correct(start, end, disp);
        break;
    case DFT_D3_BJ:
    case DFT_D3:
        return D3_correct(start, end, disp, 0);
    default:
        break;
    }
    return 0;
}

double D2_correct_atomic(int atm_id, int start, int end, disp_info disp)
{
    const mol_info mol = disp->mol;
    double e = 0.;
    double dx, dy, dz;
    double r6, r2, r;
    int ai, aj;
    double c6, damp6;
    double alp6 = disp->alp6, rs6 = disp->rs6;
    const double *A = get_atm_coord(atm_id, mol->atm);
    for (int j = start; j < end; j++)
    {
        const double *B = get_atm_coord(j, mol->atm);
        ai = mol->atm(ELEMENT_VAL, atm_id) - 1;
        aj = mol->atm(ELEMENT_VAL, j) - 1;
        dx = A[0] - B[0];
        dy = A[1] - B[1];
        dz = A[2] - B[2];
        r2 = dx * dx + dy * dy + dz * dz;
        r = sqrt(r2);
        c6 = disp->cab[0][ai][aj][0][0];
        damp6 = 1. / (1. + exp(-alp6 * (r / (rs6 * disp->r0ab[ai][aj]) - 1.)));
        if (disp->func_type == B97_D)
        {
            damp6 = 1. / (1. + alp6 * pow(r / disp->r0ab[ai][aj], -12.));
        }
        r6 = r2 * r2 * r2;
        e += damp6 * c6 / r6;
    }
    return -e * disp->s6;
}

double D3_correct_atomic(int atm_id, int start, int end, disp_info disp, int BJ)
{
    const mol_info mol = disp->mol;
    int ij, ik, jk;
    int natm = get_mol_natm(mol);
    double e = 0.;
    double dx, dy, dz;
    double r8, r6, r2, r, rr;
    int ai, aj;
    double c6, damp6, tmp;
    double c8, damp8, c9, tmp2;
    double *r2ab = (double *)malloc(sizeof(double) * natm * natm);
    double *c6ab = (double *)malloc(sizeof(double) * natm * natm);
    double *dmpab = (double *)malloc(sizeof(double) * natm * natm);
    int *if_abc = (int *)calloc(natm * natm, sizeof(int));
    double rav, t1, t2, t3, ang;
    double alp6 = disp->alp6, rs6 = disp->rs6, rs8 = disp->rs8, alp8 = disp->alp8;
    double s6 = disp->s6, s8 = disp->s8;
    get_coord(start, end, mol, disp->atm_c);
    double e6 = 0, e8 = 0;
    const double *A = get_atm_coord(atm_id, mol->atm);
    for (int j = start; j < end; j++)
    {
        const double *B = get_atm_coord(j, mol->atm);
        if (mol->atm(ELEMENT_VAL, atm_id) == 0 || mol->atm(ELEMENT_VAL, j) == 0)
            continue;
        ai = mol->atm(ELEMENT_VAL, atm_id) - 1;
        aj = mol->atm(ELEMENT_VAL, j) - 1;
        dx = A[0] - B[0];
        dy = A[1] - B[1];
        dz = A[2] - B[2];
        r2 = dx * dx + dy * dy + dz * dz;
        r = sqrt(r2);
        rr = disp->r0ab[ai][aj] / r;
        tmp = rs6 * rr;
        damp6 = 1. / (1 + 6 * pow(tmp, alp6));
        tmp = rs8 * rr;
        damp8 = 1. / (1 + 6 * pow(tmp, alp8));
        r6 = r2 * r2 * r2;
        r8 = r6 * r2;
        c6 = get_c6(disp->mxc, disp->cab, ai, aj, disp->atm_c[atm_id], disp->atm_c[j]);
        c8 = 3. * c6 * r2r4[ai] * r2r4[aj];
        if (!BJ)
        {
            e += s6 * c6 * damp6 / r6 + s8 * c8 * damp8 / r8;
            e6 += c6 * damp6 / r6;
            e8 += c8 * damp8 / r8;
            // printf("%E %E\n",c6*damp6/r6,c8*damp8/r8);
        }
        else
        {
            tmp = sqrt(c8 / c6);
            e += s6 * c6 / (r6 + pow(rs6 * tmp + rs8, 6)) + s8 * c8 / (r8 + pow(rs6 * tmp + rs8, 8));
        }
        if (r2 < 1600)
        {
            if_abc[atm_id * natm + j] = if_abc[j * natm + atm_id] = 1;
            r2ab[atm_id * natm + j] = r2ab[j * natm + atm_id] = r2;
            c6ab[atm_id * natm + j] = c6ab[j * natm + atm_id] = c6;
            dmpab[atm_id * natm + j] = dmpab[j * natm + atm_id] = pow(1. / rr, 1 / 3.);
        }
    }
    free(r2ab);
    free(c6ab);
    free(if_abc);
    free(dmpab);
    return -e;
}

double cal_disp_correct_atomic(int atm_id, int start, int end, disp_info disp)
{
    switch (disp->disp_type)
    {
    case DFT_D2:
        return D2_correct_atomic(atm_id, start, end, disp);
        break;
    case DFT_D3_BJ:
    case DFT_D3:
        return D3_correct_atomic(atm_id, start, end, disp, 0);
    default:
        break;
    }
    return 0;
}
