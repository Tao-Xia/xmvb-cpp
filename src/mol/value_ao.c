#include "mol/xgrids.h"
static int eval_ao_exp(double r2, int shls, double *buf, const bas_info bas)
{
    int prim_len = bas(NPRIM_VAL, shls);
    int prim_p = bas(EXP_IND, shls);
    for (int i = 0, j = prim_p; i < prim_len; i++, j++)
    {
        buf[i] = exp(-r2 * bas->value[j]);
    }
    return 0;
}
int eval_ao_psi(double *xyz, int shls, double *buf, const atm_info atm, const bas_info bas)
{
    double *bas_center = get_atm_coord(bas(ATOM_IND, shls), atm);
    int l = bas(ANGULAR_VAL, shls);
    double dx = -bas_center[0] + xyz[0], dy = -bas_center[1] + xyz[1], dz = -bas_center[2] + xyz[2];
    double r2 = dx * dx + dy * dy + dz * dz;
    double buf_tmp[28];
    int prim_num = bas(NPRIM_VAL, shls);
    eval_ao_exp(r2, shls, buf_tmp, bas);
    switch (l)
    {
    case 0:
        for (int i = 0; i < 1; i++)
        {
            buf[i] = 0.;
            for (int j = 0; j < prim_num; j++)
            {
                buf[i] += buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
            }
        }
        break;
    case 1:
        for (int i = 0; i < 1; i++)
        {
            buf[3 * i] = 0.;
            buf[3 * i + 1] = 0.;
            buf[3 * i + 2] = 0.;
            for (int j = 0; j < prim_num; j++)
            {
                buf[3 * i] += dx * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[3 * i + 1] += dy * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[3 * i + 2] += dz * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
            }
        }
        break;
    case 2:
        for (int i = 0; i < 1; i++)
        {
            buf[6 * i] = 0.;
            buf[6 * i + 1] = 0.;
            buf[6 * i + 2] = 0.;
            buf[6 * i + 3] = 0.;
            buf[6 * i + 4] = 0.;
            buf[6 * i + 5] = 0.;
            for (int j = 0; j < prim_num; j++)
            {
                buf[6 * i] += dx * dx * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[6 * i + 1] += dx * dy * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[6 * i + 2] += dx * dz * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[6 * i + 3] += dy * dy * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[6 * i + 4] += dy * dz * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[6 * i + 5] += dz * dz * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
            }
        }
        break;
    case 3:
        for (int i = 0; i < 1; i++)
        {
            buf[10 * i] = 0.;
            buf[10 * i + 1] = 0.;
            buf[10 * i + 2] = 0.;
            buf[10 * i + 3] = 0.;
            buf[10 * i + 4] = 0.;
            buf[10 * i + 5] = 0.;
            buf[10 * i + 6] = 0.;
            buf[10 * i + 7] = 0.;
            buf[10 * i + 8] = 0.;
            buf[10 * i + 9] = 0.;
            for (int j = 0; j < prim_num; j++)
            {
                buf[10 * i] += dx * dx * dx * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[10 * i + 1] += dx * dx * dy * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[10 * i + 2] += dx * dx * dz * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[10 * i + 3] += dx * dy * dy * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[10 * i + 4] += dx * dy * dz * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[10 * i + 5] += dx * dz * dz * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[10 * i + 6] += dy * dy * dy * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[10 * i + 7] += dy * dy * dz * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[10 * i + 8] += dy * dz * dz * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[10 * i + 9] += dz * dz * dz * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
            }
        }
        break;
    case 4:
        for (int i = 0; i < 1; i++)
        {
            buf[10 * i] = 0.;
            buf[10 * i + 1] = 0.;
            buf[10 * i + 2] = 0.;
            buf[10 * i + 3] = 0.;
            buf[10 * i + 4] = 0.;
            buf[10 * i + 5] = 0.;
            buf[10 * i + 6] = 0.;
            buf[10 * i + 7] = 0.;
            buf[10 * i + 8] = 0.;
            buf[10 * i + 9] = 0.;
            buf[10 * i + 10] = 0.;
            buf[10 * i + 11] = 0.;
            buf[10 * i + 12] = 0.;
            buf[10 * i + 13] = 0.;
            buf[10 * i + 14] = 0.;
            for (int j = 0; j < prim_num; j++)
            {
                buf[10 * i] += dx * dx * dx * dx * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[10 * i + 1] += dx * dx * dx * dy * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[10 * i + 2] += dx * dx * dx * dz * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[10 * i + 3] += dx * dx * dy * dy * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[10 * i + 4] += dx * dx * dy * dz * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[10 * i + 5] += dx * dx * dz * dz * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[10 * i + 6] += dx * dy * dy * dy * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[10 * i + 7] += dx * dy * dy * dz * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[10 * i + 8] += dx * dy * dz * dz * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[10 * i + 9] += dx * dz * dz * dz * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[10 * i + 10] += dy * dy * dy * dy * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[10 * i + 11] += dy * dy * dy * dz * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[10 * i + 12] += dy * dy * dz * dz * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[10 * i + 13] += dy * dz * dz * dz * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
                buf[10 * i + 14] += dz * dz * dz * dz * buf_tmp[j] * bas->value[bas(COEFF_IND, shls) + j];
            }
        }
        break;
    default:
        break;
    }
    return 0;
}
void eval_ao(double *xyz, int shls, double *psi, double *psix, double *psiy, double *psiz, double *psixx, double *psiyy, double *psizz, const atm_info atm, const bas_info bas)
{
    double *bas_center = get_atm_coord(bas(ATOM_IND, shls), atm);
    int l = bas(ANGULAR_VAL, shls);
    double dx = -bas_center[0] + xyz[0], dy = -bas_center[1] + xyz[1], dz = -bas_center[2] + xyz[2];
    double r2 = dx * dx + dy * dy + dz * dz;
    double buf_tmp[28];
    int prim_num = bas(NPRIM_VAL, shls);
    double phi, coe, gamma;
    eval_ao_exp(r2, shls, buf_tmp, bas);
    switch (l)
    {
    case 0:
        for (int i = 0; i < 1; i++)
        {
            psi[1 * i + 0] = 0.;
            psix[1 * i + 0] = 0.;
            psiy[1 * i + 0] = 0.;
            psiz[1 * i + 0] = 0.;
            psixx[1 * i + 0] = 0.;
            psiyy[1 * i + 0] = 0.;
            psizz[1 * i + 0] = 0.;
            for (int j = 0; j < prim_num; j++)
            {
                phi = buf_tmp[j];
                coe = bas->value[bas(COEFF_IND, shls) + j];
                gamma = bas->value[bas(EXP_IND, shls) + j];
                psi[1 * i + 0] += (phi)*coe;
                psix[1 * i + 0] += (-2 * dx * gamma * phi) * coe;
                psiy[1 * i + 0] += (-2 * dy * gamma * phi) * coe;
                psiz[1 * i + 0] += (-2 * dz * gamma * phi) * coe;
                psixx[1 * i + 0] += (4 * dx * dx * gamma * gamma * phi - 2 * gamma * phi) * coe;
                psiyy[1 * i + 0] += (4 * dy * dy * gamma * gamma * phi - 2 * gamma * phi) * coe;
                psizz[1 * i + 0] += (4 * dz * dz * gamma * gamma * phi - 2 * gamma * phi) * coe;
            }
        }
        break;
    case 1:
        for (int i = 0; i < 1; i++)
        {
            psi[3 * i + 0] = 0.;
            psix[3 * i + 0] = 0.;
            psiy[3 * i + 0] = 0.;
            psiz[3 * i + 0] = 0.;
            psixx[3 * i + 0] = 0.;
            psiyy[3 * i + 0] = 0.;
            psizz[3 * i + 0] = 0.;
            psi[3 * i + 1] = 0.;
            psix[3 * i + 1] = 0.;
            psiy[3 * i + 1] = 0.;
            psiz[3 * i + 1] = 0.;
            psixx[3 * i + 1] = 0.;
            psiyy[3 * i + 1] = 0.;
            psizz[3 * i + 1] = 0.;
            psi[3 * i + 2] = 0.;
            psix[3 * i + 2] = 0.;
            psiy[3 * i + 2] = 0.;
            psiz[3 * i + 2] = 0.;
            psixx[3 * i + 2] = 0.;
            psiyy[3 * i + 2] = 0.;
            psizz[3 * i + 2] = 0.;
            for (int j = 0; j < prim_num; j++)
            {
                phi = buf_tmp[j];
                coe = bas->value[bas(COEFF_IND, shls) + j];
                gamma = bas->value[bas(EXP_IND, shls) + j];
                psi[3 * i + 0] += (dx * phi) * coe;
                psix[3 * i + 0] += (-2 * dx * dx * gamma * phi + phi) * coe;
                psiy[3 * i + 0] += (-2 * dx * dy * gamma * phi) * coe;
                psiz[3 * i + 0] += (-2 * dx * dz * gamma * phi) * coe;
                psixx[3 * i + 0] += (4 * dx * dx * dx * gamma * gamma * phi - 6 * dx * gamma * phi) * coe;
                psiyy[3 * i + 0] += (4 * dx * dy * dy * gamma * gamma * phi - 2 * dx * gamma * phi) * coe;
                psizz[3 * i + 0] += (4 * dx * dz * dz * gamma * gamma * phi - 2 * dx * gamma * phi) * coe;
                psi[3 * i + 1] += (dy * phi) * coe;
                psix[3 * i + 1] += (-2 * dx * dy * gamma * phi) * coe;
                psiy[3 * i + 1] += (-2 * dy * dy * gamma * phi + phi) * coe;
                psiz[3 * i + 1] += (-2 * dy * dz * gamma * phi) * coe;
                psixx[3 * i + 1] += (4 * dx * dx * dy * gamma * gamma * phi - 2 * dy * gamma * phi) * coe;
                psiyy[3 * i + 1] += (4 * dy * dy * dy * gamma * gamma * phi - 6 * dy * gamma * phi) * coe;
                psizz[3 * i + 1] += (4 * dy * dz * dz * gamma * gamma * phi - 2 * dy * gamma * phi) * coe;
                psi[3 * i + 2] += (dz * phi) * coe;
                psix[3 * i + 2] += (-2 * dx * dz * gamma * phi) * coe;
                psiy[3 * i + 2] += (-2 * dy * dz * gamma * phi) * coe;
                psiz[3 * i + 2] += (-2 * dz * dz * gamma * phi + phi) * coe;
                psixx[3 * i + 2] += (4 * dx * dx * dz * gamma * gamma * phi - 2 * dz * gamma * phi) * coe;
                psiyy[3 * i + 2] += (4 * dy * dy * dz * gamma * gamma * phi - 2 * dz * gamma * phi) * coe;
                psizz[3 * i + 2] += (4 * dz * dz * dz * gamma * gamma * phi - 6 * dz * gamma * phi) * coe;
            }
        }
        break;
    case 2:
        for (int i = 0; i < 1; i++)
        {
            psi[6 * i + 0] = 0.;
            psix[6 * i + 0] = 0.;
            psiy[6 * i + 0] = 0.;
            psiz[6 * i + 0] = 0.;
            psixx[6 * i + 0] = 0.;
            psiyy[6 * i + 0] = 0.;
            psizz[6 * i + 0] = 0.;
            psi[6 * i + 1] = 0.;
            psix[6 * i + 1] = 0.;
            psiy[6 * i + 1] = 0.;
            psiz[6 * i + 1] = 0.;
            psixx[6 * i + 1] = 0.;
            psiyy[6 * i + 1] = 0.;
            psizz[6 * i + 1] = 0.;
            psi[6 * i + 2] = 0.;
            psix[6 * i + 2] = 0.;
            psiy[6 * i + 2] = 0.;
            psiz[6 * i + 2] = 0.;
            psixx[6 * i + 2] = 0.;
            psiyy[6 * i + 2] = 0.;
            psizz[6 * i + 2] = 0.;
            psi[6 * i + 3] = 0.;
            psix[6 * i + 3] = 0.;
            psiy[6 * i + 3] = 0.;
            psiz[6 * i + 3] = 0.;
            psixx[6 * i + 3] = 0.;
            psiyy[6 * i + 3] = 0.;
            psizz[6 * i + 3] = 0.;
            psi[6 * i + 4] = 0.;
            psix[6 * i + 4] = 0.;
            psiy[6 * i + 4] = 0.;
            psiz[6 * i + 4] = 0.;
            psixx[6 * i + 4] = 0.;
            psiyy[6 * i + 4] = 0.;
            psizz[6 * i + 4] = 0.;
            psi[6 * i + 5] = 0.;
            psix[6 * i + 5] = 0.;
            psiy[6 * i + 5] = 0.;
            psiz[6 * i + 5] = 0.;
            psixx[6 * i + 5] = 0.;
            psiyy[6 * i + 5] = 0.;
            psizz[6 * i + 5] = 0.;
            for (int j = 0; j < prim_num; j++)
            {
                phi = buf_tmp[j];
                coe = bas->value[bas(COEFF_IND, shls) + j];
                gamma = bas->value[bas(EXP_IND, shls) + j];
                psi[6 * i + 0] += (dx * dx * phi) * coe;
                psix[6 * i + 0] += (-2 * dx * dx * dx * gamma * phi + 2 * dx * phi) * coe;
                psiy[6 * i + 0] += (-2 * dx * dx * dy * gamma * phi) * coe;
                psiz[6 * i + 0] += (-2 * dx * dx * dz * gamma * phi) * coe;
                psixx[6 * i + 0] += (4 * dx * dx * dx * dx * gamma * gamma * phi - 10 * dx * dx * gamma * phi + 2 * phi) * coe;
                psiyy[6 * i + 0] += (4 * dx * dx * dy * dy * gamma * gamma * phi - 2 * dx * dx * gamma * phi) * coe;
                psizz[6 * i + 0] += (4 * dx * dx * dz * dz * gamma * gamma * phi - 2 * dx * dx * gamma * phi) * coe;
                psi[6 * i + 1] += (dx * dy * phi) * coe;
                psix[6 * i + 1] += (-2 * dx * dx * dy * gamma * phi + dy * phi) * coe;
                psiy[6 * i + 1] += (-2 * dx * dy * dy * gamma * phi + dx * phi) * coe;
                psiz[6 * i + 1] += (-2 * dx * dy * dz * gamma * phi) * coe;
                psixx[6 * i + 1] += (4 * dx * dx * dx * dy * gamma * gamma * phi - 6 * dx * dy * gamma * phi) * coe;
                psiyy[6 * i + 1] += (4 * dx * dy * dy * dy * gamma * gamma * phi - 6 * dx * dy * gamma * phi) * coe;
                psizz[6 * i + 1] += (4 * dx * dy * dz * dz * gamma * gamma * phi - 2 * dx * dy * gamma * phi) * coe;
                psi[6 * i + 2] += (dx * dz * phi) * coe;
                psix[6 * i + 2] += (-2 * dx * dx * dz * gamma * phi + dz * phi) * coe;
                psiy[6 * i + 2] += (-2 * dx * dy * dz * gamma * phi) * coe;
                psiz[6 * i + 2] += (-2 * dx * dz * dz * gamma * phi + dx * phi) * coe;
                psixx[6 * i + 2] += (4 * dx * dx * dx * dz * gamma * gamma * phi - 6 * dx * dz * gamma * phi) * coe;
                psiyy[6 * i + 2] += (4 * dx * dy * dy * dz * gamma * gamma * phi - 2 * dx * dz * gamma * phi) * coe;
                psizz[6 * i + 2] += (4 * dx * dz * dz * dz * gamma * gamma * phi - 6 * dx * dz * gamma * phi) * coe;
                psi[6 * i + 3] += (dy * dy * phi) * coe;
                psix[6 * i + 3] += (-2 * dx * dy * dy * gamma * phi) * coe;
                psiy[6 * i + 3] += (-2 * dy * dy * dy * gamma * phi + 2 * dy * phi) * coe;
                psiz[6 * i + 3] += (-2 * dy * dy * dz * gamma * phi) * coe;
                psixx[6 * i + 3] += (4 * dx * dx * dy * dy * gamma * gamma * phi - 2 * dy * dy * gamma * phi) * coe;
                psiyy[6 * i + 3] += (4 * dy * dy * dy * dy * gamma * gamma * phi - 10 * dy * dy * gamma * phi + 2 * phi) * coe;
                psizz[6 * i + 3] += (4 * dy * dy * dz * dz * gamma * gamma * phi - 2 * dy * dy * gamma * phi) * coe;
                psi[6 * i + 4] += (dy * dz * phi) * coe;
                psix[6 * i + 4] += (-2 * dx * dy * dz * gamma * phi) * coe;
                psiy[6 * i + 4] += (-2 * dy * dy * dz * gamma * phi + dz * phi) * coe;
                psiz[6 * i + 4] += (-2 * dy * dz * dz * gamma * phi + dy * phi) * coe;
                psixx[6 * i + 4] += (4 * dx * dx * dy * dz * gamma * gamma * phi - 2 * dy * dz * gamma * phi) * coe;
                psiyy[6 * i + 4] += (4 * dy * dy * dy * dz * gamma * gamma * phi - 6 * dy * dz * gamma * phi) * coe;
                psizz[6 * i + 4] += (4 * dy * dz * dz * dz * gamma * gamma * phi - 6 * dy * dz * gamma * phi) * coe;
                psi[6 * i + 5] += (dz * dz * phi) * coe;
                psix[6 * i + 5] += (-2 * dx * dz * dz * gamma * phi) * coe;
                psiy[6 * i + 5] += (-2 * dy * dz * dz * gamma * phi) * coe;
                psiz[6 * i + 5] += (-2 * dz * dz * dz * gamma * phi + 2 * dz * phi) * coe;
                psixx[6 * i + 5] += (4 * dx * dx * dz * dz * gamma * gamma * phi - 2 * dz * dz * gamma * phi) * coe;
                psiyy[6 * i + 5] += (4 * dy * dy * dz * dz * gamma * gamma * phi - 2 * dz * dz * gamma * phi) * coe;
                psizz[6 * i + 5] += (4 * dz * dz * dz * dz * gamma * gamma * phi - 10 * dz * dz * gamma * phi + 2 * phi) * coe;
            }
        }
        break;
    case 3:
        for (int i = 0; i < 1; i++)
        {
            psi[10 * i + 0] = 0.;
            psix[10 * i + 0] = 0.;
            psiy[10 * i + 0] = 0.;
            psiz[10 * i + 0] = 0.;
            psixx[10 * i + 0] = 0.;
            psiyy[10 * i + 0] = 0.;
            psizz[10 * i + 0] = 0.;
            psi[10 * i + 1] = 0.;
            psix[10 * i + 1] = 0.;
            psiy[10 * i + 1] = 0.;
            psiz[10 * i + 1] = 0.;
            psixx[10 * i + 1] = 0.;
            psiyy[10 * i + 1] = 0.;
            psizz[10 * i + 1] = 0.;
            psi[10 * i + 2] = 0.;
            psix[10 * i + 2] = 0.;
            psiy[10 * i + 2] = 0.;
            psiz[10 * i + 2] = 0.;
            psixx[10 * i + 2] = 0.;
            psiyy[10 * i + 2] = 0.;
            psizz[10 * i + 2] = 0.;
            psi[10 * i + 3] = 0.;
            psix[10 * i + 3] = 0.;
            psiy[10 * i + 3] = 0.;
            psiz[10 * i + 3] = 0.;
            psixx[10 * i + 3] = 0.;
            psiyy[10 * i + 3] = 0.;
            psizz[10 * i + 3] = 0.;
            psi[10 * i + 4] = 0.;
            psix[10 * i + 4] = 0.;
            psiy[10 * i + 4] = 0.;
            psiz[10 * i + 4] = 0.;
            psixx[10 * i + 4] = 0.;
            psiyy[10 * i + 4] = 0.;
            psizz[10 * i + 4] = 0.;
            psi[10 * i + 5] = 0.;
            psix[10 * i + 5] = 0.;
            psiy[10 * i + 5] = 0.;
            psiz[10 * i + 5] = 0.;
            psixx[10 * i + 5] = 0.;
            psiyy[10 * i + 5] = 0.;
            psizz[10 * i + 5] = 0.;
            psi[10 * i + 6] = 0.;
            psix[10 * i + 6] = 0.;
            psiy[10 * i + 6] = 0.;
            psiz[10 * i + 6] = 0.;
            psixx[10 * i + 6] = 0.;
            psiyy[10 * i + 6] = 0.;
            psizz[10 * i + 6] = 0.;
            psi[10 * i + 7] = 0.;
            psix[10 * i + 7] = 0.;
            psiy[10 * i + 7] = 0.;
            psiz[10 * i + 7] = 0.;
            psixx[10 * i + 7] = 0.;
            psiyy[10 * i + 7] = 0.;
            psizz[10 * i + 7] = 0.;
            psi[10 * i + 8] = 0.;
            psix[10 * i + 8] = 0.;
            psiy[10 * i + 8] = 0.;
            psiz[10 * i + 8] = 0.;
            psixx[10 * i + 8] = 0.;
            psiyy[10 * i + 8] = 0.;
            psizz[10 * i + 8] = 0.;
            psi[10 * i + 9] = 0.;
            psix[10 * i + 9] = 0.;
            psiy[10 * i + 9] = 0.;
            psiz[10 * i + 9] = 0.;
            psixx[10 * i + 9] = 0.;
            psiyy[10 * i + 9] = 0.;
            psizz[10 * i + 9] = 0.;
            for (int j = 0; j < prim_num; j++)
            {
                phi = buf_tmp[j];
                coe = bas->value[bas(COEFF_IND, shls) + j];
                gamma = bas->value[bas(EXP_IND, shls) + j];
                psi[10 * i + 0] += (dx * dx * dx * phi) * coe;
                psix[10 * i + 0] += (-2 * dx * dx * dx * dx * gamma * phi + 3 * dx * dx * phi) * coe;
                psiy[10 * i + 0] += (-2 * dx * dx * dx * dy * gamma * phi) * coe;
                psiz[10 * i + 0] += (-2 * dx * dx * dx * dz * gamma * phi) * coe;
                psixx[10 * i + 0] += (4 * dx * dx * dx * dx * dx * gamma * gamma * phi - 14 * dx * dx * dx * gamma * phi + 6 * dx * phi) * coe;
                psiyy[10 * i + 0] += (4 * dx * dx * dx * dy * dy * gamma * gamma * phi - 2 * dx * dx * dx * gamma * phi) * coe;
                psizz[10 * i + 0] += (4 * dx * dx * dx * dz * dz * gamma * gamma * phi - 2 * dx * dx * dx * gamma * phi) * coe;
                psi[10 * i + 1] += (dx * dx * dy * phi) * coe;
                psix[10 * i + 1] += (-2 * dx * dx * dx * dy * gamma * phi + 2 * dx * dy * phi) * coe;
                psiy[10 * i + 1] += (-2 * dx * dx * dy * dy * gamma * phi + dx * dx * phi) * coe;
                psiz[10 * i + 1] += (-2 * dx * dx * dy * dz * gamma * phi) * coe;
                psixx[10 * i + 1] += (4 * dx * dx * dx * dx * dy * gamma * gamma * phi - 10 * dx * dx * dy * gamma * phi + 2 * dy * phi) * coe;
                psiyy[10 * i + 1] += (4 * dx * dx * dy * dy * dy * gamma * gamma * phi - 6 * dx * dx * dy * gamma * phi) * coe;
                psizz[10 * i + 1] += (4 * dx * dx * dy * dz * dz * gamma * gamma * phi - 2 * dx * dx * dy * gamma * phi) * coe;
                psi[10 * i + 2] += (dx * dx * dz * phi) * coe;
                psix[10 * i + 2] += (-2 * dx * dx * dx * dz * gamma * phi + 2 * dx * dz * phi) * coe;
                psiy[10 * i + 2] += (-2 * dx * dx * dy * dz * gamma * phi) * coe;
                psiz[10 * i + 2] += (-2 * dx * dx * dz * dz * gamma * phi + dx * dx * phi) * coe;
                psixx[10 * i + 2] += (4 * dx * dx * dx * dx * dz * gamma * gamma * phi - 10 * dx * dx * dz * gamma * phi + 2 * dz * phi) * coe;
                psiyy[10 * i + 2] += (4 * dx * dx * dy * dy * dz * gamma * gamma * phi - 2 * dx * dx * dz * gamma * phi) * coe;
                psizz[10 * i + 2] += (4 * dx * dx * dz * dz * dz * gamma * gamma * phi - 6 * dx * dx * dz * gamma * phi) * coe;
                psi[10 * i + 3] += (dx * dy * dy * phi) * coe;
                psix[10 * i + 3] += (-2 * dx * dx * dy * dy * gamma * phi + dy * dy * phi) * coe;
                psiy[10 * i + 3] += (-2 * dx * dy * dy * dy * gamma * phi + 2 * dx * dy * phi) * coe;
                psiz[10 * i + 3] += (-2 * dx * dy * dy * dz * gamma * phi) * coe;
                psixx[10 * i + 3] += (4 * dx * dx * dx * dy * dy * gamma * gamma * phi - 6 * dx * dy * dy * gamma * phi) * coe;
                psiyy[10 * i + 3] += (4 * dx * dy * dy * dy * dy * gamma * gamma * phi - 10 * dx * dy * dy * gamma * phi + 2 * dx * phi) * coe;
                psizz[10 * i + 3] += (4 * dx * dy * dy * dz * dz * gamma * gamma * phi - 2 * dx * dy * dy * gamma * phi) * coe;
                psi[10 * i + 4] += (dx * dy * dz * phi) * coe;
                psix[10 * i + 4] += (-2 * dx * dx * dy * dz * gamma * phi + dy * dz * phi) * coe;
                psiy[10 * i + 4] += (-2 * dx * dy * dy * dz * gamma * phi + dx * dz * phi) * coe;
                psiz[10 * i + 4] += (-2 * dx * dy * dz * dz * gamma * phi + dx * dy * phi) * coe;
                psixx[10 * i + 4] += (4 * dx * dx * dx * dy * dz * gamma * gamma * phi - 6 * dx * dy * dz * gamma * phi) * coe;
                psiyy[10 * i + 4] += (4 * dx * dy * dy * dy * dz * gamma * gamma * phi - 6 * dx * dy * dz * gamma * phi) * coe;
                psizz[10 * i + 4] += (4 * dx * dy * dz * dz * dz * gamma * gamma * phi - 6 * dx * dy * dz * gamma * phi) * coe;
                psi[10 * i + 5] += (dx * dz * dz * phi) * coe;
                psix[10 * i + 5] += (-2 * dx * dx * dz * dz * gamma * phi + dz * dz * phi) * coe;
                psiy[10 * i + 5] += (-2 * dx * dy * dz * dz * gamma * phi) * coe;
                psiz[10 * i + 5] += (-2 * dx * dz * dz * dz * gamma * phi + 2 * dx * dz * phi) * coe;
                psixx[10 * i + 5] += (4 * dx * dx * dx * dz * dz * gamma * gamma * phi - 6 * dx * dz * dz * gamma * phi) * coe;
                psiyy[10 * i + 5] += (4 * dx * dy * dy * dz * dz * gamma * gamma * phi - 2 * dx * dz * dz * gamma * phi) * coe;
                psizz[10 * i + 5] += (4 * dx * dz * dz * dz * dz * gamma * gamma * phi - 10 * dx * dz * dz * gamma * phi + 2 * dx * phi) * coe;
                psi[10 * i + 6] += (dy * dy * dy * phi) * coe;
                psix[10 * i + 6] += (-2 * dx * dy * dy * dy * gamma * phi) * coe;
                psiy[10 * i + 6] += (-2 * dy * dy * dy * dy * gamma * phi + 3 * dy * dy * phi) * coe;
                psiz[10 * i + 6] += (-2 * dy * dy * dy * dz * gamma * phi) * coe;
                psixx[10 * i + 6] += (4 * dx * dx * dy * dy * dy * gamma * gamma * phi - 2 * dy * dy * dy * gamma * phi) * coe;
                psiyy[10 * i + 6] += (4 * dy * dy * dy * dy * dy * gamma * gamma * phi - 14 * dy * dy * dy * gamma * phi + 6 * dy * phi) * coe;
                psizz[10 * i + 6] += (4 * dy * dy * dy * dz * dz * gamma * gamma * phi - 2 * dy * dy * dy * gamma * phi) * coe;
                psi[10 * i + 7] += (dy * dy * dz * phi) * coe;
                psix[10 * i + 7] += (-2 * dx * dy * dy * dz * gamma * phi) * coe;
                psiy[10 * i + 7] += (-2 * dy * dy * dy * dz * gamma * phi + 2 * dy * dz * phi) * coe;
                psiz[10 * i + 7] += (-2 * dy * dy * dz * dz * gamma * phi + dy * dy * phi) * coe;
                psixx[10 * i + 7] += (4 * dx * dx * dy * dy * dz * gamma * gamma * phi - 2 * dy * dy * dz * gamma * phi) * coe;
                psiyy[10 * i + 7] += (4 * dy * dy * dy * dy * dz * gamma * gamma * phi - 10 * dy * dy * dz * gamma * phi + 2 * dz * phi) * coe;
                psizz[10 * i + 7] += (4 * dy * dy * dz * dz * dz * gamma * gamma * phi - 6 * dy * dy * dz * gamma * phi) * coe;
                psi[10 * i + 8] += (dy * dz * dz * phi) * coe;
                psix[10 * i + 8] += (-2 * dx * dy * dz * dz * gamma * phi) * coe;
                psiy[10 * i + 8] += (-2 * dy * dy * dz * dz * gamma * phi + dz * dz * phi) * coe;
                psiz[10 * i + 8] += (-2 * dy * dz * dz * dz * gamma * phi + 2 * dy * dz * phi) * coe;
                psixx[10 * i + 8] += (4 * dx * dx * dy * dz * dz * gamma * gamma * phi - 2 * dy * dz * dz * gamma * phi) * coe;
                psiyy[10 * i + 8] += (4 * dy * dy * dy * dz * dz * gamma * gamma * phi - 6 * dy * dz * dz * gamma * phi) * coe;
                psizz[10 * i + 8] += (4 * dy * dz * dz * dz * dz * gamma * gamma * phi - 10 * dy * dz * dz * gamma * phi + 2 * dy * phi) * coe;
                psi[10 * i + 9] += (dz * dz * dz * phi) * coe;
                psix[10 * i + 9] += (-2 * dx * dz * dz * dz * gamma * phi) * coe;
                psiy[10 * i + 9] += (-2 * dy * dz * dz * dz * gamma * phi) * coe;
                psiz[10 * i + 9] += (-2 * dz * dz * dz * dz * gamma * phi + 3 * dz * dz * phi) * coe;
                psixx[10 * i + 9] += (4 * dx * dx * dz * dz * dz * gamma * gamma * phi - 2 * dz * dz * dz * gamma * phi) * coe;
                psiyy[10 * i + 9] += (4 * dy * dy * dz * dz * dz * gamma * gamma * phi - 2 * dz * dz * dz * gamma * phi) * coe;
                psizz[10 * i + 9] += (4 * dz * dz * dz * dz * dz * gamma * gamma * phi - 14 * dz * dz * dz * gamma * phi + 6 * dz * phi) * coe;
            }
        }
        break;
    case 4:
        for (int i = 0; i < 1; i++)
        {
            psi[15 * i + 0] = 0.;
            psix[15 * i + 0] = 0.;
            psiy[15 * i + 0] = 0.;
            psiz[15 * i + 0] = 0.;
            psixx[15 * i + 0] = 0.;
            psiyy[15 * i + 0] = 0.;
            psizz[15 * i + 0] = 0.;
            psi[15 * i + 1] = 0.;
            psix[15 * i + 1] = 0.;
            psiy[15 * i + 1] = 0.;
            psiz[15 * i + 1] = 0.;
            psixx[15 * i + 1] = 0.;
            psiyy[15 * i + 1] = 0.;
            psizz[15 * i + 1] = 0.;
            psi[15 * i + 2] = 0.;
            psix[15 * i + 2] = 0.;
            psiy[15 * i + 2] = 0.;
            psiz[15 * i + 2] = 0.;
            psixx[15 * i + 2] = 0.;
            psiyy[15 * i + 2] = 0.;
            psizz[15 * i + 2] = 0.;
            psi[15 * i + 3] = 0.;
            psix[15 * i + 3] = 0.;
            psiy[15 * i + 3] = 0.;
            psiz[15 * i + 3] = 0.;
            psixx[15 * i + 3] = 0.;
            psiyy[15 * i + 3] = 0.;
            psizz[15 * i + 3] = 0.;
            psi[15 * i + 4] = 0.;
            psix[15 * i + 4] = 0.;
            psiy[15 * i + 4] = 0.;
            psiz[15 * i + 4] = 0.;
            psixx[15 * i + 4] = 0.;
            psiyy[15 * i + 4] = 0.;
            psizz[15 * i + 4] = 0.;
            psi[15 * i + 5] = 0.;
            psix[15 * i + 5] = 0.;
            psiy[15 * i + 5] = 0.;
            psiz[15 * i + 5] = 0.;
            psixx[15 * i + 5] = 0.;
            psiyy[15 * i + 5] = 0.;
            psizz[15 * i + 5] = 0.;
            psi[15 * i + 6] = 0.;
            psix[15 * i + 6] = 0.;
            psiy[15 * i + 6] = 0.;
            psiz[15 * i + 6] = 0.;
            psixx[15 * i + 6] = 0.;
            psiyy[15 * i + 6] = 0.;
            psizz[15 * i + 6] = 0.;
            psi[15 * i + 7] = 0.;
            psix[15 * i + 7] = 0.;
            psiy[15 * i + 7] = 0.;
            psiz[15 * i + 7] = 0.;
            psixx[15 * i + 7] = 0.;
            psiyy[15 * i + 7] = 0.;
            psizz[15 * i + 7] = 0.;
            psi[15 * i + 8] = 0.;
            psix[15 * i + 8] = 0.;
            psiy[15 * i + 8] = 0.;
            psiz[15 * i + 8] = 0.;
            psixx[15 * i + 8] = 0.;
            psiyy[15 * i + 8] = 0.;
            psizz[15 * i + 8] = 0.;
            psi[15 * i + 9] = 0.;
            psix[15 * i + 9] = 0.;
            psiy[15 * i + 9] = 0.;
            psiz[15 * i + 9] = 0.;
            psixx[15 * i + 9] = 0.;
            psiyy[15 * i + 9] = 0.;
            psizz[15 * i + 9] = 0.;
            psi[15 * i + 10] = 0.;
            psix[15 * i + 10] = 0.;
            psiy[15 * i + 10] = 0.;
            psiz[15 * i + 10] = 0.;
            psixx[15 * i + 10] = 0.;
            psiyy[15 * i + 10] = 0.;
            psizz[15 * i + 10] = 0.;
            psi[15 * i + 11] = 0.;
            psix[15 * i + 11] = 0.;
            psiy[15 * i + 11] = 0.;
            psiz[15 * i + 11] = 0.;
            psixx[15 * i + 11] = 0.;
            psiyy[15 * i + 11] = 0.;
            psizz[15 * i + 11] = 0.;
            psi[15 * i + 12] = 0.;
            psix[15 * i + 12] = 0.;
            psiy[15 * i + 12] = 0.;
            psiz[15 * i + 12] = 0.;
            psixx[15 * i + 12] = 0.;
            psiyy[15 * i + 12] = 0.;
            psizz[15 * i + 12] = 0.;
            psi[15 * i + 13] = 0.;
            psix[15 * i + 13] = 0.;
            psiy[15 * i + 13] = 0.;
            psiz[15 * i + 13] = 0.;
            psixx[15 * i + 13] = 0.;
            psiyy[15 * i + 13] = 0.;
            psizz[15 * i + 13] = 0.;
            psi[15 * i + 14] = 0.;
            psix[15 * i + 14] = 0.;
            psiy[15 * i + 14] = 0.;
            psiz[15 * i + 14] = 0.;
            psixx[15 * i + 14] = 0.;
            psiyy[15 * i + 14] = 0.;
            psizz[15 * i + 14] = 0.;
            for (int j = 0; j < prim_num; j++)
            {
                phi = buf_tmp[j];
                coe = bas->value[bas(COEFF_IND, shls) + j];
                gamma = bas->value[bas(EXP_IND, shls) + j];
                psi[15 * i + 0] += (dx * dx * dx * dx * phi) * coe;
                psix[15 * i + 0] += (-2 * dx * dx * dx * dx * dx * gamma * phi + 4 * dx * dx * dx * phi) * coe;
                psiy[15 * i + 0] += (-2 * dx * dx * dx * dx * dy * gamma * phi) * coe;
                psiz[15 * i + 0] += (-2 * dx * dx * dx * dx * dz * gamma * phi) * coe;
                psixx[15 * i + 0] += (4 * dx * dx * dx * dx * dx * dx * gamma * gamma * phi - 18 * dx * dx * dx * dx * gamma * phi + 12 * dx * dx * phi) * coe;
                psiyy[15 * i + 0] += (4 * dx * dx * dx * dx * dy * dy * gamma * gamma * phi - 2 * dx * dx * dx * dx * gamma * phi) * coe;
                psizz[15 * i + 0] += (4 * dx * dx * dx * dx * dz * dz * gamma * gamma * phi - 2 * dx * dx * dx * dx * gamma * phi) * coe;
                psi[15 * i + 1] += (dx * dx * dx * dy * phi) * coe;
                psix[15 * i + 1] += (-2 * dx * dx * dx * dx * dy * gamma * phi + 3 * dx * dx * dy * phi) * coe;
                psiy[15 * i + 1] += (-2 * dx * dx * dx * dy * dy * gamma * phi + dx * dx * dx * phi) * coe;
                psiz[15 * i + 1] += (-2 * dx * dx * dx * dy * dz * gamma * phi) * coe;
                psixx[15 * i + 1] += (4 * dx * dx * dx * dx * dx * dy * gamma * gamma * phi - 14 * dx * dx * dx * dy * gamma * phi + 6 * dx * dy * phi) * coe;
                psiyy[15 * i + 1] += (4 * dx * dx * dx * dy * dy * dy * gamma * gamma * phi - 6 * dx * dx * dx * dy * gamma * phi) * coe;
                psizz[15 * i + 1] += (4 * dx * dx * dx * dy * dz * dz * gamma * gamma * phi - 2 * dx * dx * dx * dy * gamma * phi) * coe;
                psi[15 * i + 2] += (dx * dx * dx * dz * phi) * coe;
                psix[15 * i + 2] += (-2 * dx * dx * dx * dx * dz * gamma * phi + 3 * dx * dx * dz * phi) * coe;
                psiy[15 * i + 2] += (-2 * dx * dx * dx * dy * dz * gamma * phi) * coe;
                psiz[15 * i + 2] += (-2 * dx * dx * dx * dz * dz * gamma * phi + dx * dx * dx * phi) * coe;
                psixx[15 * i + 2] += (4 * dx * dx * dx * dx * dx * dz * gamma * gamma * phi - 14 * dx * dx * dx * dz * gamma * phi + 6 * dx * dz * phi) * coe;
                psiyy[15 * i + 2] += (4 * dx * dx * dx * dy * dy * dz * gamma * gamma * phi - 2 * dx * dx * dx * dz * gamma * phi) * coe;
                psizz[15 * i + 2] += (4 * dx * dx * dx * dz * dz * dz * gamma * gamma * phi - 6 * dx * dx * dx * dz * gamma * phi) * coe;
                psi[15 * i + 3] += (dx * dx * dy * dy * phi) * coe;
                psix[15 * i + 3] += (-2 * dx * dx * dx * dy * dy * gamma * phi + 2 * dx * dy * dy * phi) * coe;
                psiy[15 * i + 3] += (-2 * dx * dx * dy * dy * dy * gamma * phi + 2 * dx * dx * dy * phi) * coe;
                psiz[15 * i + 3] += (-2 * dx * dx * dy * dy * dz * gamma * phi) * coe;
                psixx[15 * i + 3] += (4 * dx * dx * dx * dx * dy * dy * gamma * gamma * phi - 10 * dx * dx * dy * dy * gamma * phi + 2 * dy * dy * phi) * coe;
                psiyy[15 * i + 3] += (4 * dx * dx * dy * dy * dy * dy * gamma * gamma * phi - 10 * dx * dx * dy * dy * gamma * phi + 2 * dx * dx * phi) * coe;
                psizz[15 * i + 3] += (4 * dx * dx * dy * dy * dz * dz * gamma * gamma * phi - 2 * dx * dx * dy * dy * gamma * phi) * coe;
                psi[15 * i + 4] += (dx * dx * dy * dz * phi) * coe;
                psix[15 * i + 4] += (-2 * dx * dx * dx * dy * dz * gamma * phi + 2 * dx * dy * dz * phi) * coe;
                psiy[15 * i + 4] += (-2 * dx * dx * dy * dy * dz * gamma * phi + dx * dx * dz * phi) * coe;
                psiz[15 * i + 4] += (-2 * dx * dx * dy * dz * dz * gamma * phi + dx * dx * dy * phi) * coe;
                psixx[15 * i + 4] += (4 * dx * dx * dx * dx * dy * dz * gamma * gamma * phi - 10 * dx * dx * dy * dz * gamma * phi + 2 * dy * dz * phi) * coe;
                psiyy[15 * i + 4] += (4 * dx * dx * dy * dy * dy * dz * gamma * gamma * phi - 6 * dx * dx * dy * dz * gamma * phi) * coe;
                psizz[15 * i + 4] += (4 * dx * dx * dy * dz * dz * dz * gamma * gamma * phi - 6 * dx * dx * dy * dz * gamma * phi) * coe;
                psi[15 * i + 5] += (dx * dx * dz * dz * phi) * coe;
                psix[15 * i + 5] += (-2 * dx * dx * dx * dz * dz * gamma * phi + 2 * dx * dz * dz * phi) * coe;
                psiy[15 * i + 5] += (-2 * dx * dx * dy * dz * dz * gamma * phi) * coe;
                psiz[15 * i + 5] += (-2 * dx * dx * dz * dz * dz * gamma * phi + 2 * dx * dx * dz * phi) * coe;
                psixx[15 * i + 5] += (4 * dx * dx * dx * dx * dz * dz * gamma * gamma * phi - 10 * dx * dx * dz * dz * gamma * phi + 2 * dz * dz * phi) * coe;
                psiyy[15 * i + 5] += (4 * dx * dx * dy * dy * dz * dz * gamma * gamma * phi - 2 * dx * dx * dz * dz * gamma * phi) * coe;
                psizz[15 * i + 5] += (4 * dx * dx * dz * dz * dz * dz * gamma * gamma * phi - 10 * dx * dx * dz * dz * gamma * phi + 2 * dx * dx * phi) * coe;
                psi[15 * i + 6] += (dx * dy * dy * dy * phi) * coe;
                psix[15 * i + 6] += (-2 * dx * dx * dy * dy * dy * gamma * phi + dy * dy * dy * phi) * coe;
                psiy[15 * i + 6] += (-2 * dx * dy * dy * dy * dy * gamma * phi + 3 * dx * dy * dy * phi) * coe;
                psiz[15 * i + 6] += (-2 * dx * dy * dy * dy * dz * gamma * phi) * coe;
                psixx[15 * i + 6] += (4 * dx * dx * dx * dy * dy * dy * gamma * gamma * phi - 6 * dx * dy * dy * dy * gamma * phi) * coe;
                psiyy[15 * i + 6] += (4 * dx * dy * dy * dy * dy * dy * gamma * gamma * phi - 14 * dx * dy * dy * dy * gamma * phi + 6 * dx * dy * phi) * coe;
                psizz[15 * i + 6] += (4 * dx * dy * dy * dy * dz * dz * gamma * gamma * phi - 2 * dx * dy * dy * dy * gamma * phi) * coe;
                psi[15 * i + 7] += (dx * dy * dy * dz * phi) * coe;
                psix[15 * i + 7] += (-2 * dx * dx * dy * dy * dz * gamma * phi + dy * dy * dz * phi) * coe;
                psiy[15 * i + 7] += (-2 * dx * dy * dy * dy * dz * gamma * phi + 2 * dx * dy * dz * phi) * coe;
                psiz[15 * i + 7] += (-2 * dx * dy * dy * dz * dz * gamma * phi + dx * dy * dy * phi) * coe;
                psixx[15 * i + 7] += (4 * dx * dx * dx * dy * dy * dz * gamma * gamma * phi - 6 * dx * dy * dy * dz * gamma * phi) * coe;
                psiyy[15 * i + 7] += (4 * dx * dy * dy * dy * dy * dz * gamma * gamma * phi - 10 * dx * dy * dy * dz * gamma * phi + 2 * dx * dz * phi) * coe;
                psizz[15 * i + 7] += (4 * dx * dy * dy * dz * dz * dz * gamma * gamma * phi - 6 * dx * dy * dy * dz * gamma * phi) * coe;
                psi[15 * i + 8] += (dx * dy * dz * dz * phi) * coe;
                psix[15 * i + 8] += (-2 * dx * dx * dy * dz * dz * gamma * phi + dy * dz * dz * phi) * coe;
                psiy[15 * i + 8] += (-2 * dx * dy * dy * dz * dz * gamma * phi + dx * dz * dz * phi) * coe;
                psiz[15 * i + 8] += (-2 * dx * dy * dz * dz * dz * gamma * phi + 2 * dx * dy * dz * phi) * coe;
                psixx[15 * i + 8] += (4 * dx * dx * dx * dy * dz * dz * gamma * gamma * phi - 6 * dx * dy * dz * dz * gamma * phi) * coe;
                psiyy[15 * i + 8] += (4 * dx * dy * dy * dy * dz * dz * gamma * gamma * phi - 6 * dx * dy * dz * dz * gamma * phi) * coe;
                psizz[15 * i + 8] += (4 * dx * dy * dz * dz * dz * dz * gamma * gamma * phi - 10 * dx * dy * dz * dz * gamma * phi + 2 * dx * dy * phi) * coe;
                psi[15 * i + 9] += (dx * dz * dz * dz * phi) * coe;
                psix[15 * i + 9] += (-2 * dx * dx * dz * dz * dz * gamma * phi + dz * dz * dz * phi) * coe;
                psiy[15 * i + 9] += (-2 * dx * dy * dz * dz * dz * gamma * phi) * coe;
                psiz[15 * i + 9] += (-2 * dx * dz * dz * dz * dz * gamma * phi + 3 * dx * dz * dz * phi) * coe;
                psixx[15 * i + 9] += (4 * dx * dx * dx * dz * dz * dz * gamma * gamma * phi - 6 * dx * dz * dz * dz * gamma * phi) * coe;
                psiyy[15 * i + 9] += (4 * dx * dy * dy * dz * dz * dz * gamma * gamma * phi - 2 * dx * dz * dz * dz * gamma * phi) * coe;
                psizz[15 * i + 9] += (4 * dx * dz * dz * dz * dz * dz * gamma * gamma * phi - 14 * dx * dz * dz * dz * gamma * phi + 6 * dx * dz * phi) * coe;
                psi[15 * i + 10] += (dy * dy * dy * dy * phi) * coe;
                psix[15 * i + 10] += (-2 * dx * dy * dy * dy * dy * gamma * phi) * coe;
                psiy[15 * i + 10] += (-2 * dy * dy * dy * dy * dy * gamma * phi + 4 * dy * dy * dy * phi) * coe;
                psiz[15 * i + 10] += (-2 * dy * dy * dy * dy * dz * gamma * phi) * coe;
                psixx[15 * i + 10] += (4 * dx * dx * dy * dy * dy * dy * gamma * gamma * phi - 2 * dy * dy * dy * dy * gamma * phi) * coe;
                psiyy[15 * i + 10] += (4 * dy * dy * dy * dy * dy * dy * gamma * gamma * phi - 18 * dy * dy * dy * dy * gamma * phi + 12 * dy * dy * phi) * coe;
                psizz[15 * i + 10] += (4 * dy * dy * dy * dy * dz * dz * gamma * gamma * phi - 2 * dy * dy * dy * dy * gamma * phi) * coe;
                psi[15 * i + 11] += (dy * dy * dy * dz * phi) * coe;
                psix[15 * i + 11] += (-2 * dx * dy * dy * dy * dz * gamma * phi) * coe;
                psiy[15 * i + 11] += (-2 * dy * dy * dy * dy * dz * gamma * phi + 3 * dy * dy * dz * phi) * coe;
                psiz[15 * i + 11] += (-2 * dy * dy * dy * dz * dz * gamma * phi + dy * dy * dy * phi) * coe;
                psixx[15 * i + 11] += (4 * dx * dx * dy * dy * dy * dz * gamma * gamma * phi - 2 * dy * dy * dy * dz * gamma * phi) * coe;
                psiyy[15 * i + 11] += (4 * dy * dy * dy * dy * dy * dz * gamma * gamma * phi - 14 * dy * dy * dy * dz * gamma * phi + 6 * dy * dz * phi) * coe;
                psizz[15 * i + 11] += (4 * dy * dy * dy * dz * dz * dz * gamma * gamma * phi - 6 * dy * dy * dy * dz * gamma * phi) * coe;
                psi[15 * i + 12] += (dy * dy * dz * dz * phi) * coe;
                psix[15 * i + 12] += (-2 * dx * dy * dy * dz * dz * gamma * phi) * coe;
                psiy[15 * i + 12] += (-2 * dy * dy * dy * dz * dz * gamma * phi + 2 * dy * dz * dz * phi) * coe;
                psiz[15 * i + 12] += (-2 * dy * dy * dz * dz * dz * gamma * phi + 2 * dy * dy * dz * phi) * coe;
                psixx[15 * i + 12] += (4 * dx * dx * dy * dy * dz * dz * gamma * gamma * phi - 2 * dy * dy * dz * dz * gamma * phi) * coe;
                psiyy[15 * i + 12] += (4 * dy * dy * dy * dy * dz * dz * gamma * gamma * phi - 10 * dy * dy * dz * dz * gamma * phi + 2 * dz * dz * phi) * coe;
                psizz[15 * i + 12] += (4 * dy * dy * dz * dz * dz * dz * gamma * gamma * phi - 10 * dy * dy * dz * dz * gamma * phi + 2 * dy * dy * phi) * coe;
                psi[15 * i + 13] += (dy * dz * dz * dz * phi) * coe;
                psix[15 * i + 13] += (-2 * dx * dy * dz * dz * dz * gamma * phi) * coe;
                psiy[15 * i + 13] += (-2 * dy * dy * dz * dz * dz * gamma * phi + dz * dz * dz * phi) * coe;
                psiz[15 * i + 13] += (-2 * dy * dz * dz * dz * dz * gamma * phi + 3 * dy * dz * dz * phi) * coe;
                psixx[15 * i + 13] += (4 * dx * dx * dy * dz * dz * dz * gamma * gamma * phi - 2 * dy * dz * dz * dz * gamma * phi) * coe;
                psiyy[15 * i + 13] += (4 * dy * dy * dy * dz * dz * dz * gamma * gamma * phi - 6 * dy * dz * dz * dz * gamma * phi) * coe;
                psizz[15 * i + 13] += (4 * dy * dz * dz * dz * dz * dz * gamma * gamma * phi - 14 * dy * dz * dz * dz * gamma * phi + 6 * dy * dz * phi) * coe;
                psi[15 * i + 14] += (dz * dz * dz * dz * phi) * coe;
                psix[15 * i + 14] += (-2 * dx * dz * dz * dz * dz * gamma * phi) * coe;
                psiy[15 * i + 14] += (-2 * dy * dz * dz * dz * dz * gamma * phi) * coe;
                psiz[15 * i + 14] += (-2 * dz * dz * dz * dz * dz * gamma * phi + 4 * dz * dz * dz * phi) * coe;
                psixx[15 * i + 14] += (4 * dx * dx * dz * dz * dz * dz * gamma * gamma * phi - 2 * dz * dz * dz * dz * gamma * phi) * coe;
                psiyy[15 * i + 14] += (4 * dy * dy * dz * dz * dz * dz * gamma * gamma * phi - 2 * dz * dz * dz * dz * gamma * phi) * coe;
                psizz[15 * i + 14] += (4 * dz * dz * dz * dz * dz * dz * gamma * gamma * phi - 18 * dz * dz * dz * dz * gamma * phi + 12 * dz * dz * phi) * coe;
            }
        }
        break;
    default:
        assert(0);
        break;
    }
    return;
}

