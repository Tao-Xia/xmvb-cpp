#include "mol/xint.h"
static void nuc_VRR_0_0(double *buf, double *cache, double coe, double K_ab, double half_xi_, double xi, const double *PA,
                        const double *PG, double PG_dist2, double sr_frac, double lr_frac, double omega_)
{
    double theta;
    double T_0, T_1;
    double scheme_1[10], scheme_0[10];
    double scheme_0_0_0[1];
    T_0 = xi * PG_dist2;
    if (fabs(omega_) < 1E-8)
    {
        cal_boys(0, T_0, &scheme_0[0]);
        for (int i = 0; i <= 0; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * lr_frac * scheme_0[i];
    }
    else
    {
        theta = sqrt(omega_ * omega_ / (omega_ * omega_ + xi));
        T_1 = xi * PG_dist2 * theta * theta;
        cal_boys(0, T_1, &scheme_1[0]);
        cal_boys(0, T_0, &scheme_0[0]);
        for (int i = 0; i <= 0; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * (lr_frac * scheme_0[i] + sr_frac * theta * pow(theta, 2 * i) * scheme_1[i]);
    }
    buf[0] += cache[0];
    return;
}

static void nuc_VRR_1_0(double *buf, double *cache, double coe, double K_ab, double half_xi_, double xi, const double *PA,
                        const double *PG, double PG_dist2, double sr_frac, double lr_frac, double omega_)
{
    double theta;
    double T_0, T_1;
    double scheme_1[10], scheme_0[10];
    double scheme_0_0_0[2];
    T_0 = xi * PG_dist2;
    if (fabs(omega_) < 1E-8)
    {
        cal_boys(1, T_0, &scheme_0[0]);
        for (int i = 0; i <= 1; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * lr_frac * scheme_0[i];
    }
    else
    {
        theta = sqrt(omega_ * omega_ / (omega_ * omega_ + xi));
        T_1 = xi * PG_dist2 * theta * theta;
        cal_boys(1, T_1, &scheme_1[0]);
        cal_boys(1, T_0, &scheme_0[0]);
        for (int i = 0; i <= 1; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * (lr_frac * scheme_0[i] + sr_frac * theta * pow(theta, 2 * i) * scheme_1[i]);
    }
    for (int i = 0; i < 1; i++)
        cache[2 + i] = PA[0] * cache[0 + i] - PG[0] * cache[0 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[3 + i] = PA[1] * cache[0 + i] - PG[1] * cache[0 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[4 + i] = PA[2] * cache[0 + i] - PG[2] * cache[0 + i + 1];
    buf[0] += cache[2];
    buf[1] += cache[3];
    buf[2] += cache[4];
    return;
}

static void nuc_VRR_1_1(double *buf, double *cache, double coe, double K_ab, double half_xi_, double xi, const double *PA,
                        const double *PG, double PG_dist2, double sr_frac, double lr_frac, double omega_)
{
    double theta;
    double T_0, T_1;
    double scheme_1[10], scheme_0[10];
    double scheme_0_0_0[3];
    T_0 = xi * PG_dist2;
    if (fabs(omega_) < 1E-8)
    {
        cal_boys(2, T_0, &scheme_0[0]);
        for (int i = 0; i <= 2; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * lr_frac * scheme_0[i];
    }
    else
    {
        theta = sqrt(omega_ * omega_ / (omega_ * omega_ + xi));
        T_1 = xi * PG_dist2 * theta * theta;
        cal_boys(2, T_1, &scheme_1[0]);
        cal_boys(2, T_0, &scheme_0[0]);
        for (int i = 0; i <= 2; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * (lr_frac * scheme_0[i] + sr_frac * theta * pow(theta, 2 * i) * scheme_1[i]);
    }
    for (int i = 0; i < 2; i++)
        cache[3 + i] = PA[0] * cache[0 + i] - PG[0] * cache[0 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[5 + i] = PA[1] * cache[0 + i] - PG[1] * cache[0 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[7 + i] = PA[2] * cache[0 + i] - PG[2] * cache[0 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[9 + i] = PA[0] * cache[3 + i] - PG[0] * cache[3 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[10 + i] = PA[0] * cache[5 + i] - PG[0] * cache[5 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[11 + i] = PA[0] * cache[7 + i] - PG[0] * cache[7 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[12 + i] = PA[1] * cache[5 + i] - PG[1] * cache[5 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[13 + i] = PA[1] * cache[7 + i] - PG[1] * cache[7 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[14 + i] = PA[2] * cache[7 + i] - PG[2] * cache[7 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    buf[0] += cache[3];
    buf[1] += cache[5];
    buf[2] += cache[7];
    buf[3] += cache[9];
    buf[4] += cache[10];
    buf[5] += cache[11];
    buf[6] += cache[12];
    buf[7] += cache[13];
    buf[8] += cache[14];
    return;
}

static void nuc_VRR_2_0(double *buf, double *cache, double coe, double K_ab, double half_xi_, double xi, const double *PA,
                        const double *PG, double PG_dist2, double sr_frac, double lr_frac, double omega_)
{
    double theta;
    double T_0, T_1;
    double scheme_1[10], scheme_0[10];
    double scheme_0_0_0[3];
    T_0 = xi * PG_dist2;
    if (fabs(omega_) < 1E-8)
    {
        cal_boys(2, T_0, &scheme_0[0]);
        for (int i = 0; i <= 2; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * lr_frac * scheme_0[i];
    }
    else
    {
        theta = sqrt(omega_ * omega_ / (omega_ * omega_ + xi));
        T_1 = xi * PG_dist2 * theta * theta;
        cal_boys(2, T_1, &scheme_1[0]);
        cal_boys(2, T_0, &scheme_0[0]);
        for (int i = 0; i <= 2; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * (lr_frac * scheme_0[i] + sr_frac * theta * pow(theta, 2 * i) * scheme_1[i]);
    }
    for (int i = 0; i < 2; i++)
        cache[3 + i] = PA[0] * cache[0 + i] - PG[0] * cache[0 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[5 + i] = PA[1] * cache[0 + i] - PG[1] * cache[0 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[7 + i] = PA[2] * cache[0 + i] - PG[2] * cache[0 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[9 + i] = PA[0] * cache[3 + i] - PG[0] * cache[3 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[10 + i] = PA[0] * cache[5 + i] - PG[0] * cache[5 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[11 + i] = PA[0] * cache[7 + i] - PG[0] * cache[7 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[12 + i] = PA[1] * cache[5 + i] - PG[1] * cache[5 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[13 + i] = PA[1] * cache[7 + i] - PG[1] * cache[7 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[14 + i] = PA[2] * cache[7 + i] - PG[2] * cache[7 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    buf[0] += cache[9];
    buf[1] += cache[10];
    buf[2] += cache[11];
    buf[3] += cache[12];
    buf[4] += cache[13];
    buf[5] += cache[14];
    return;
}

static void nuc_VRR_2_1(double *buf, double *cache, double coe, double K_ab, double half_xi_, double xi, const double *PA,
                        const double *PG, double PG_dist2, double sr_frac, double lr_frac, double omega_)
{
    double theta;
    double T_0, T_1;
    double scheme_1[10], scheme_0[10];
    double scheme_0_0_0[4];
    T_0 = xi * PG_dist2;
    if (fabs(omega_) < 1E-8)
    {
        cal_boys(3, T_0, &scheme_0[0]);
        for (int i = 0; i <= 3; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * lr_frac * scheme_0[i];
    }
    else
    {
        theta = sqrt(omega_ * omega_ / (omega_ * omega_ + xi));
        T_1 = xi * PG_dist2 * theta * theta;
        cal_boys(3, T_1, &scheme_1[0]);
        cal_boys(3, T_0, &scheme_0[0]);
        for (int i = 0; i <= 3; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * (lr_frac * scheme_0[i] + sr_frac * theta * pow(theta, 2 * i) * scheme_1[i]);
    }
    for (int i = 0; i < 3; i++)
        cache[4 + i] = PA[0] * cache[0 + i] - PG[0] * cache[0 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[7 + i] = PA[1] * cache[0 + i] - PG[1] * cache[0 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[10 + i] = PA[2] * cache[0 + i] - PG[2] * cache[0 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[13 + i] = PA[0] * cache[4 + i] - PG[0] * cache[4 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[15 + i] = PA[0] * cache[7 + i] - PG[0] * cache[7 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[17 + i] = PA[0] * cache[10 + i] - PG[0] * cache[10 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[19 + i] = PA[1] * cache[7 + i] - PG[1] * cache[7 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[21 + i] = PA[1] * cache[10 + i] - PG[1] * cache[10 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[23 + i] = PA[2] * cache[10 + i] - PG[2] * cache[10 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[25 + i] = PA[0] * cache[13 + i] - PG[0] * cache[13 + i + 1] + 2 * half_xi_ * (cache[4 + i] - cache[4 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[26 + i] = PA[1] * cache[13 + i] - PG[1] * cache[13 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[27 + i] = PA[2] * cache[13 + i] - PG[2] * cache[13 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[28 + i] = PA[0] * cache[19 + i] - PG[0] * cache[19 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[29 + i] = PA[0] * cache[21 + i] - PG[0] * cache[21 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[30 + i] = PA[0] * cache[23 + i] - PG[0] * cache[23 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[31 + i] = PA[1] * cache[19 + i] - PG[1] * cache[19 + i + 1] + 2 * half_xi_ * (cache[7 + i] - cache[7 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[32 + i] = PA[2] * cache[19 + i] - PG[2] * cache[19 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[33 + i] = PA[1] * cache[23 + i] - PG[1] * cache[23 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[34 + i] = PA[2] * cache[23 + i] - PG[2] * cache[23 + i + 1] + 2 * half_xi_ * (cache[10 + i] - cache[10 + i + 1]);
    buf[0] += cache[13];
    buf[1] += cache[15];
    buf[2] += cache[17];
    buf[3] += cache[19];
    buf[4] += cache[21];
    buf[5] += cache[23];
    buf[6] += cache[25];
    buf[7] += cache[26];
    buf[8] += cache[27];
    buf[9] += cache[28];
    buf[10] += cache[29];
    buf[11] += cache[30];
    buf[12] += cache[31];
    buf[13] += cache[32];
    buf[14] += cache[33];
    buf[15] += cache[34];
    return;
}

static void nuc_VRR_2_2(double *buf, double *cache, double coe, double K_ab, double half_xi_, double xi, const double *PA,
                        const double *PG, double PG_dist2, double sr_frac, double lr_frac, double omega_)
{
    double theta;
    double T_0, T_1;
    double scheme_1[10], scheme_0[10];
    double scheme_0_0_0[5];
    T_0 = xi * PG_dist2;
    if (fabs(omega_) < 1E-8)
    {
        cal_boys(4, T_0, &scheme_0[0]);
        for (int i = 0; i <= 4; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * lr_frac * scheme_0[i];
    }
    else
    {
        theta = sqrt(omega_ * omega_ / (omega_ * omega_ + xi));
        T_1 = xi * PG_dist2 * theta * theta;
        cal_boys(4, T_1, &scheme_1[0]);
        cal_boys(4, T_0, &scheme_0[0]);
        for (int i = 0; i <= 4; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * (lr_frac * scheme_0[i] + sr_frac * theta * pow(theta, 2 * i) * scheme_1[i]);
    }
    for (int i = 0; i < 4; i++)
        cache[5 + i] = PA[0] * cache[0 + i] - PG[0] * cache[0 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[9 + i] = PA[1] * cache[0 + i] - PG[1] * cache[0 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[13 + i] = PA[2] * cache[0 + i] - PG[2] * cache[0 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[17 + i] = PA[0] * cache[5 + i] - PG[0] * cache[5 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[20 + i] = PA[0] * cache[9 + i] - PG[0] * cache[9 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[23 + i] = PA[0] * cache[13 + i] - PG[0] * cache[13 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[26 + i] = PA[1] * cache[9 + i] - PG[1] * cache[9 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[29 + i] = PA[1] * cache[13 + i] - PG[1] * cache[13 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[32 + i] = PA[2] * cache[13 + i] - PG[2] * cache[13 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[35 + i] = PA[0] * cache[17 + i] - PG[0] * cache[17 + i + 1] + 2 * half_xi_ * (cache[5 + i] - cache[5 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[37 + i] = PA[1] * cache[17 + i] - PG[1] * cache[17 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[39 + i] = PA[2] * cache[17 + i] - PG[2] * cache[17 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[41 + i] = PA[0] * cache[26 + i] - PG[0] * cache[26 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[43 + i] = PA[0] * cache[29 + i] - PG[0] * cache[29 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[45 + i] = PA[0] * cache[32 + i] - PG[0] * cache[32 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[47 + i] = PA[1] * cache[26 + i] - PG[1] * cache[26 + i + 1] + 2 * half_xi_ * (cache[9 + i] - cache[9 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[49 + i] = PA[2] * cache[26 + i] - PG[2] * cache[26 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[51 + i] = PA[1] * cache[32 + i] - PG[1] * cache[32 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[53 + i] = PA[2] * cache[32 + i] - PG[2] * cache[32 + i + 1] + 2 * half_xi_ * (cache[13 + i] - cache[13 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[55 + i] = PA[0] * cache[35 + i] - PG[0] * cache[35 + i + 1] + 3 * half_xi_ * (cache[17 + i] - cache[17 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[56 + i] = PA[1] * cache[35 + i] - PG[1] * cache[35 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[57 + i] = PA[2] * cache[35 + i] - PG[2] * cache[35 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[58 + i] = PA[0] * cache[41 + i] - PG[0] * cache[41 + i + 1] + 1 * half_xi_ * (cache[26 + i] - cache[26 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[59 + i] = PA[1] * cache[39 + i] - PG[1] * cache[39 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[60 + i] = PA[0] * cache[45 + i] - PG[0] * cache[45 + i + 1] + 1 * half_xi_ * (cache[32 + i] - cache[32 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[61 + i] = PA[0] * cache[47 + i] - PG[0] * cache[47 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[62 + i] = PA[0] * cache[49 + i] - PG[0] * cache[49 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[63 + i] = PA[0] * cache[51 + i] - PG[0] * cache[51 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[64 + i] = PA[0] * cache[53 + i] - PG[0] * cache[53 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[65 + i] = PA[1] * cache[47 + i] - PG[1] * cache[47 + i + 1] + 3 * half_xi_ * (cache[26 + i] - cache[26 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[66 + i] = PA[2] * cache[47 + i] - PG[2] * cache[47 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[67 + i] = PA[1] * cache[51 + i] - PG[1] * cache[51 + i + 1] + 1 * half_xi_ * (cache[32 + i] - cache[32 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[68 + i] = PA[1] * cache[53 + i] - PG[1] * cache[53 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[69 + i] = PA[2] * cache[53 + i] - PG[2] * cache[53 + i + 1] + 3 * half_xi_ * (cache[32 + i] - cache[32 + i + 1]);

	buf[0] += cache[17];
	buf[1] += cache[20];
	buf[2] += cache[23];
	buf[3] += cache[26];
	buf[4] += cache[29];
	buf[5] += cache[32];
	buf[6] += cache[35];
	buf[7] += cache[37];
	buf[8] += cache[39];
	buf[9] += cache[41];
	buf[10] += cache[43];
	buf[11] += cache[45];
	buf[12] += cache[47];
	buf[13] += cache[49];
	buf[14] += cache[51];
	buf[15] += cache[53];
	buf[16] += cache[55];
	buf[17] += cache[56];
	buf[18] += cache[57];
	buf[19] += cache[58];
	buf[20] += cache[59];
	buf[21] += cache[60];
	buf[22] += cache[61];
	buf[23] += cache[62];
	buf[24] += cache[63];
	buf[25] += cache[64];
	buf[26] += cache[65];
	buf[27] += cache[66];
	buf[28] += cache[67];
	buf[29] += cache[68];
	buf[30] += cache[69];
    return;
}

static void nuc_VRR_3_0(double *buf, double *cache, double coe, double K_ab, double half_xi_, double xi, const double *PA,
                        const double *PG, double PG_dist2, double sr_frac, double lr_frac, double omega_)
{
    double theta;
    double T_0, T_1;
    double scheme_1[10], scheme_0[10];
    double scheme_0_0_0[4];
    T_0 = xi * PG_dist2;
    if (fabs(omega_) < 1E-8)
    {
        cal_boys(3, T_0, &scheme_0[0]);
        for (int i = 0; i <= 3; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * lr_frac * scheme_0[i];
    }
    else
    {
        theta = sqrt(omega_ * omega_ / (omega_ * omega_ + xi));
        T_1 = xi * PG_dist2 * theta * theta;
        cal_boys(3, T_1, &scheme_1[0]);
        cal_boys(3, T_0, &scheme_0[0]);
        for (int i = 0; i <= 3; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * (lr_frac * scheme_0[i] + sr_frac * theta * pow(theta, 2 * i) * scheme_1[i]);
    }
    for (int i = 0; i < 3; i++)
        cache[4 + i] = PA[0] * cache[0 + i] - PG[0] * cache[0 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[7 + i] = PA[1] * cache[0 + i] - PG[1] * cache[0 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[10 + i] = PA[2] * cache[0 + i] - PG[2] * cache[0 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[13 + i] = PA[0] * cache[4 + i] - PG[0] * cache[4 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[15 + i] = PA[0] * cache[7 + i] - PG[0] * cache[7 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[17 + i] = PA[0] * cache[10 + i] - PG[0] * cache[10 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[19 + i] = PA[1] * cache[7 + i] - PG[1] * cache[7 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[21 + i] = PA[1] * cache[10 + i] - PG[1] * cache[10 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[23 + i] = PA[2] * cache[10 + i] - PG[2] * cache[10 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[25 + i] = PA[0] * cache[13 + i] - PG[0] * cache[13 + i + 1] + 2 * half_xi_ * (cache[4 + i] - cache[4 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[26 + i] = PA[1] * cache[13 + i] - PG[1] * cache[13 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[27 + i] = PA[2] * cache[13 + i] - PG[2] * cache[13 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[28 + i] = PA[0] * cache[19 + i] - PG[0] * cache[19 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[29 + i] = PA[0] * cache[21 + i] - PG[0] * cache[21 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[30 + i] = PA[0] * cache[23 + i] - PG[0] * cache[23 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[31 + i] = PA[1] * cache[19 + i] - PG[1] * cache[19 + i + 1] + 2 * half_xi_ * (cache[7 + i] - cache[7 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[32 + i] = PA[2] * cache[19 + i] - PG[2] * cache[19 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[33 + i] = PA[1] * cache[23 + i] - PG[1] * cache[23 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[34 + i] = PA[2] * cache[23 + i] - PG[2] * cache[23 + i + 1] + 2 * half_xi_ * (cache[10 + i] - cache[10 + i + 1]);
    buf[0] += cache[25];
    buf[1] += cache[26];
    buf[2] += cache[27];
    buf[3] += cache[28];
    buf[4] += cache[29];
    buf[5] += cache[30];
    buf[6] += cache[31];
    buf[7] += cache[32];
    buf[8] += cache[33];
    buf[9] += cache[34];
    return;
}

static void nuc_VRR_3_1(double *buf, double *cache, double coe, double K_ab, double half_xi_, double xi, const double *PA,
                        const double *PG, double PG_dist2, double sr_frac, double lr_frac, double omega_)
{
    double theta;
    double T_0, T_1;
    double scheme_1[10], scheme_0[10];
    double scheme_0_0_0[5];
    T_0 = xi * PG_dist2;
    if (fabs(omega_) < 1E-8)
    {
        cal_boys(4, T_0, &scheme_0[0]);
        for (int i = 0; i <= 4; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * lr_frac * scheme_0[i];
    }
    else
    {
        theta = sqrt(omega_ * omega_ / (omega_ * omega_ + xi));
        T_1 = xi * PG_dist2 * theta * theta;
        cal_boys(4, T_1, &scheme_1[0]);
        cal_boys(4, T_0, &scheme_0[0]);
        for (int i = 0; i <= 4; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * (lr_frac * scheme_0[i] + sr_frac * theta * pow(theta, 2 * i) * scheme_1[i]);
    }
    for (int i = 0; i < 4; i++)
        cache[5 + i] = PA[0] * cache[0 + i] - PG[0] * cache[0 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[9 + i] = PA[1] * cache[0 + i] - PG[1] * cache[0 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[13 + i] = PA[2] * cache[0 + i] - PG[2] * cache[0 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[17 + i] = PA[0] * cache[5 + i] - PG[0] * cache[5 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[20 + i] = PA[0] * cache[9 + i] - PG[0] * cache[9 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[23 + i] = PA[0] * cache[13 + i] - PG[0] * cache[13 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[26 + i] = PA[1] * cache[9 + i] - PG[1] * cache[9 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[29 + i] = PA[1] * cache[13 + i] - PG[1] * cache[13 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[32 + i] = PA[2] * cache[13 + i] - PG[2] * cache[13 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[35 + i] = PA[0] * cache[17 + i] - PG[0] * cache[17 + i + 1] + 2 * half_xi_ * (cache[5 + i] - cache[5 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[37 + i] = PA[1] * cache[17 + i] - PG[1] * cache[17 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[39 + i] = PA[2] * cache[17 + i] - PG[2] * cache[17 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[41 + i] = PA[0] * cache[26 + i] - PG[0] * cache[26 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[43 + i] = PA[0] * cache[29 + i] - PG[0] * cache[29 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[45 + i] = PA[0] * cache[32 + i] - PG[0] * cache[32 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[47 + i] = PA[1] * cache[26 + i] - PG[1] * cache[26 + i + 1] + 2 * half_xi_ * (cache[9 + i] - cache[9 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[49 + i] = PA[2] * cache[26 + i] - PG[2] * cache[26 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[51 + i] = PA[1] * cache[32 + i] - PG[1] * cache[32 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[53 + i] = PA[2] * cache[32 + i] - PG[2] * cache[32 + i + 1] + 2 * half_xi_ * (cache[13 + i] - cache[13 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[55 + i] = PA[0] * cache[35 + i] - PG[0] * cache[35 + i + 1] + 3 * half_xi_ * (cache[17 + i] - cache[17 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[56 + i] = PA[1] * cache[35 + i] - PG[1] * cache[35 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[57 + i] = PA[2] * cache[35 + i] - PG[2] * cache[35 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[58 + i] = PA[0] * cache[41 + i] - PG[0] * cache[41 + i + 1] + 1 * half_xi_ * (cache[26 + i] - cache[26 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[59 + i] = PA[1] * cache[39 + i] - PG[1] * cache[39 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[60 + i] = PA[0] * cache[45 + i] - PG[0] * cache[45 + i + 1] + 1 * half_xi_ * (cache[32 + i] - cache[32 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[61 + i] = PA[0] * cache[47 + i] - PG[0] * cache[47 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[62 + i] = PA[0] * cache[49 + i] - PG[0] * cache[49 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[63 + i] = PA[0] * cache[51 + i] - PG[0] * cache[51 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[64 + i] = PA[0] * cache[53 + i] - PG[0] * cache[53 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[65 + i] = PA[1] * cache[47 + i] - PG[1] * cache[47 + i + 1] + 3 * half_xi_ * (cache[26 + i] - cache[26 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[66 + i] = PA[2] * cache[47 + i] - PG[2] * cache[47 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[67 + i] = PA[1] * cache[51 + i] - PG[1] * cache[51 + i + 1] + 1 * half_xi_ * (cache[32 + i] - cache[32 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[68 + i] = PA[1] * cache[53 + i] - PG[1] * cache[53 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[69 + i] = PA[2] * cache[53 + i] - PG[2] * cache[53 + i + 1] + 3 * half_xi_ * (cache[32 + i] - cache[32 + i + 1]);
    buf[0] += cache[35];
    buf[1] += cache[37];
    buf[2] += cache[39];
    buf[3] += cache[41];
    buf[4] += cache[43];
    buf[5] += cache[45];
    buf[6] += cache[47];
    buf[7] += cache[49];
    buf[8] += cache[51];
    buf[9] += cache[53];
    buf[10] += cache[55];
    buf[11] += cache[56];
    buf[12] += cache[57];
    buf[13] += cache[58];
    buf[14] += cache[59];
    buf[15] += cache[60];
    buf[16] += cache[61];
    buf[17] += cache[62];
    buf[18] += cache[63];
    buf[19] += cache[64];
    buf[20] += cache[65];
    buf[21] += cache[66];
    buf[22] += cache[67];
    buf[23] += cache[68];
    buf[24] += cache[69];
    return;
}

static void nuc_VRR_3_2(double *buf, double *cache, double coe, double K_ab, double half_xi_, double xi, const double *PA,
                        const double *PG, double PG_dist2, double sr_frac, double lr_frac, double omega_)
{
    double theta;
    double T_0, T_1;
    double scheme_1[10], scheme_0[10];
    double scheme_0_0_0[6];
    T_0 = xi * PG_dist2;
    if (fabs(omega_) < 1E-8)
    {
        cal_boys(5, T_0, &scheme_0[0]);
        for (int i = 0; i <= 5; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * lr_frac * scheme_0[i];
    }
    else
    {
        theta = sqrt(omega_ * omega_ / (omega_ * omega_ + xi));
        T_1 = xi * PG_dist2 * theta * theta;
        cal_boys(5, T_1, &scheme_1[0]);
        cal_boys(5, T_0, &scheme_0[0]);
        for (int i = 0; i <= 5; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * (lr_frac * scheme_0[i] + sr_frac * theta * pow(theta, 2 * i) * scheme_1[i]);
    }
    for (int i = 0; i < 5; i++)
        cache[6 + i] = PA[0] * cache[0 + i] - PG[0] * cache[0 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[11 + i] = PA[1] * cache[0 + i] - PG[1] * cache[0 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[16 + i] = PA[2] * cache[0 + i] - PG[2] * cache[0 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[21 + i] = PA[0] * cache[6 + i] - PG[0] * cache[6 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[25 + i] = PA[0] * cache[11 + i] - PG[0] * cache[11 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[29 + i] = PA[0] * cache[16 + i] - PG[0] * cache[16 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[33 + i] = PA[1] * cache[11 + i] - PG[1] * cache[11 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[37 + i] = PA[1] * cache[16 + i] - PG[1] * cache[16 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[41 + i] = PA[2] * cache[16 + i] - PG[2] * cache[16 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[45 + i] = PA[0] * cache[21 + i] - PG[0] * cache[21 + i + 1] + 2 * half_xi_ * (cache[6 + i] - cache[6 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[48 + i] = PA[1] * cache[21 + i] - PG[1] * cache[21 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[51 + i] = PA[2] * cache[21 + i] - PG[2] * cache[21 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[54 + i] = PA[0] * cache[33 + i] - PG[0] * cache[33 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[57 + i] = PA[0] * cache[37 + i] - PG[0] * cache[37 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[60 + i] = PA[0] * cache[41 + i] - PG[0] * cache[41 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[63 + i] = PA[1] * cache[33 + i] - PG[1] * cache[33 + i + 1] + 2 * half_xi_ * (cache[11 + i] - cache[11 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[66 + i] = PA[2] * cache[33 + i] - PG[2] * cache[33 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[69 + i] = PA[1] * cache[41 + i] - PG[1] * cache[41 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[72 + i] = PA[2] * cache[41 + i] - PG[2] * cache[41 + i + 1] + 2 * half_xi_ * (cache[16 + i] - cache[16 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[75 + i] = PA[0] * cache[45 + i] - PG[0] * cache[45 + i + 1] + 3 * half_xi_ * (cache[21 + i] - cache[21 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[77 + i] = PA[1] * cache[45 + i] - PG[1] * cache[45 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[79 + i] = PA[2] * cache[45 + i] - PG[2] * cache[45 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[81 + i] = PA[0] * cache[54 + i] - PG[0] * cache[54 + i + 1] + 1 * half_xi_ * (cache[33 + i] - cache[33 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[83 + i] = PA[1] * cache[51 + i] - PG[1] * cache[51 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[85 + i] = PA[0] * cache[60 + i] - PG[0] * cache[60 + i + 1] + 1 * half_xi_ * (cache[41 + i] - cache[41 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[87 + i] = PA[0] * cache[63 + i] - PG[0] * cache[63 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[89 + i] = PA[0] * cache[66 + i] - PG[0] * cache[66 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[91 + i] = PA[0] * cache[69 + i] - PG[0] * cache[69 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[93 + i] = PA[0] * cache[72 + i] - PG[0] * cache[72 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[95 + i] = PA[1] * cache[63 + i] - PG[1] * cache[63 + i + 1] + 3 * half_xi_ * (cache[33 + i] - cache[33 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[97 + i] = PA[2] * cache[63 + i] - PG[2] * cache[63 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[99 + i] = PA[1] * cache[69 + i] - PG[1] * cache[69 + i + 1] + 1 * half_xi_ * (cache[41 + i] - cache[41 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[101 + i] = PA[1] * cache[72 + i] - PG[1] * cache[72 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[103 + i] = PA[2] * cache[72 + i] - PG[2] * cache[72 + i + 1] + 3 * half_xi_ * (cache[41 + i] - cache[41 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[105 + i] = PA[0] * cache[75 + i] - PG[0] * cache[75 + i + 1] + 4 * half_xi_ * (cache[45 + i] - cache[45 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[106 + i] = PA[1] * cache[75 + i] - PG[1] * cache[75 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[107 + i] = PA[2] * cache[75 + i] - PG[2] * cache[75 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[108 + i] = PA[0] * cache[81 + i] - PG[0] * cache[81 + i + 1] + 2 * half_xi_ * (cache[54 + i] - cache[54 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[109 + i] = PA[1] * cache[79 + i] - PG[1] * cache[79 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[110 + i] = PA[0] * cache[85 + i] - PG[0] * cache[85 + i + 1] + 2 * half_xi_ * (cache[60 + i] - cache[60 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[111 + i] = PA[0] * cache[87 + i] - PG[0] * cache[87 + i + 1] + 1 * half_xi_ * (cache[63 + i] - cache[63 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[112 + i] = PA[2] * cache[81 + i] - PG[2] * cache[81 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[113 + i] = PA[1] * cache[85 + i] - PG[1] * cache[85 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[114 + i] = PA[0] * cache[93 + i] - PG[0] * cache[93 + i + 1] + 1 * half_xi_ * (cache[72 + i] - cache[72 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[115 + i] = PA[0] * cache[95 + i] - PG[0] * cache[95 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[116 + i] = PA[0] * cache[97 + i] - PG[0] * cache[97 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[117 + i] = PA[0] * cache[99 + i] - PG[0] * cache[99 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[118 + i] = PA[0] * cache[101 + i] - PG[0] * cache[101 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[119 + i] = PA[0] * cache[103 + i] - PG[0] * cache[103 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[120 + i] = PA[1] * cache[95 + i] - PG[1] * cache[95 + i + 1] + 4 * half_xi_ * (cache[63 + i] - cache[63 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[121 + i] = PA[2] * cache[95 + i] - PG[2] * cache[95 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[122 + i] = PA[1] * cache[99 + i] - PG[1] * cache[99 + i + 1] + 2 * half_xi_ * (cache[69 + i] - cache[69 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[123 + i] = PA[1] * cache[101 + i] - PG[1] * cache[101 + i + 1] + 1 * half_xi_ * (cache[72 + i] - cache[72 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[124 + i] = PA[1] * cache[103 + i] - PG[1] * cache[103 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[125 + i] = PA[2] * cache[103 + i] - PG[2] * cache[103 + i + 1] + 4 * half_xi_ * (cache[72 + i] - cache[72 + i + 1]);
    buf[0] += cache[45];
    buf[1] += cache[48];
    buf[2] += cache[51];
    buf[3] += cache[54];
    buf[4] += cache[57];
    buf[5] += cache[60];
    buf[6] += cache[63];
    buf[7] += cache[66];
    buf[8] += cache[69];
    buf[9] += cache[72];
    buf[10] += cache[75];
    buf[11] += cache[77];
    buf[12] += cache[79];
    buf[13] += cache[81];
    buf[14] += cache[83];
    buf[15] += cache[85];
    buf[16] += cache[87];
    buf[17] += cache[89];
    buf[18] += cache[91];
    buf[19] += cache[93];
    buf[20] += cache[95];
    buf[21] += cache[97];
    buf[22] += cache[99];
    buf[23] += cache[101];
    buf[24] += cache[103];
    buf[25] += cache[105];
    buf[26] += cache[106];
    buf[27] += cache[107];
    buf[28] += cache[108];
    buf[29] += cache[109];
    buf[30] += cache[110];
    buf[31] += cache[111];
    buf[32] += cache[112];
    buf[33] += cache[113];
    buf[34] += cache[114];
    buf[35] += cache[115];
    buf[36] += cache[116];
    buf[37] += cache[117];
    buf[38] += cache[118];
    buf[39] += cache[119];
    buf[40] += cache[120];
    buf[41] += cache[121];
    buf[42] += cache[122];
    buf[43] += cache[123];
    buf[44] += cache[124];
    buf[45] += cache[125];
    return;
}

static void nuc_VRR_3_3(double *buf, double *cache, double coe, double K_ab, double half_xi_, double xi, const double *PA,
                        const double *PG, double PG_dist2, double sr_frac, double lr_frac, double omega_)
{
    double theta;
    double T_0, T_1;
    double scheme_1[10], scheme_0[10];
    double scheme_0_0_0[7];
    T_0 = xi * PG_dist2;
    if (fabs(omega_) < 1E-8)
    {
        cal_boys(6, T_0, &scheme_0[0]);
        for (int i = 0; i <= 6; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * lr_frac * scheme_0[i];
    }
    else
    {
        theta = sqrt(omega_ * omega_ / (omega_ * omega_ + xi));
        T_1 = xi * PG_dist2 * theta * theta;
        cal_boys(6, T_1, &scheme_1[0]);
        cal_boys(6, T_0, &scheme_0[0]);
        for (int i = 0; i <= 6; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * (lr_frac * scheme_0[i] + sr_frac * theta * pow(theta, 2 * i) * scheme_1[i]);
    }
    for (int i = 0; i < 6; i++)
        cache[7 + i] = PA[0] * cache[0 + i] - PG[0] * cache[0 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[13 + i] = PA[1] * cache[0 + i] - PG[1] * cache[0 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[19 + i] = PA[2] * cache[0 + i] - PG[2] * cache[0 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[25 + i] = PA[0] * cache[7 + i] - PG[0] * cache[7 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[30 + i] = PA[0] * cache[13 + i] - PG[0] * cache[13 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[35 + i] = PA[0] * cache[19 + i] - PG[0] * cache[19 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[40 + i] = PA[1] * cache[13 + i] - PG[1] * cache[13 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[45 + i] = PA[1] * cache[19 + i] - PG[1] * cache[19 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[50 + i] = PA[2] * cache[19 + i] - PG[2] * cache[19 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[55 + i] = PA[0] * cache[25 + i] - PG[0] * cache[25 + i + 1] + 2 * half_xi_ * (cache[7 + i] - cache[7 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[59 + i] = PA[1] * cache[25 + i] - PG[1] * cache[25 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[63 + i] = PA[2] * cache[25 + i] - PG[2] * cache[25 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[67 + i] = PA[0] * cache[40 + i] - PG[0] * cache[40 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[71 + i] = PA[0] * cache[45 + i] - PG[0] * cache[45 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[75 + i] = PA[0] * cache[50 + i] - PG[0] * cache[50 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[79 + i] = PA[1] * cache[40 + i] - PG[1] * cache[40 + i + 1] + 2 * half_xi_ * (cache[13 + i] - cache[13 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[83 + i] = PA[2] * cache[40 + i] - PG[2] * cache[40 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[87 + i] = PA[1] * cache[50 + i] - PG[1] * cache[50 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[91 + i] = PA[2] * cache[50 + i] - PG[2] * cache[50 + i + 1] + 2 * half_xi_ * (cache[19 + i] - cache[19 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[95 + i] = PA[0] * cache[55 + i] - PG[0] * cache[55 + i + 1] + 3 * half_xi_ * (cache[25 + i] - cache[25 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[98 + i] = PA[1] * cache[55 + i] - PG[1] * cache[55 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[101 + i] = PA[2] * cache[55 + i] - PG[2] * cache[55 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[104 + i] = PA[0] * cache[67 + i] - PG[0] * cache[67 + i + 1] + 1 * half_xi_ * (cache[40 + i] - cache[40 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[107 + i] = PA[1] * cache[63 + i] - PG[1] * cache[63 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[110 + i] = PA[0] * cache[75 + i] - PG[0] * cache[75 + i + 1] + 1 * half_xi_ * (cache[50 + i] - cache[50 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[113 + i] = PA[0] * cache[79 + i] - PG[0] * cache[79 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[116 + i] = PA[0] * cache[83 + i] - PG[0] * cache[83 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[119 + i] = PA[0] * cache[87 + i] - PG[0] * cache[87 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[122 + i] = PA[0] * cache[91 + i] - PG[0] * cache[91 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[125 + i] = PA[1] * cache[79 + i] - PG[1] * cache[79 + i + 1] + 3 * half_xi_ * (cache[40 + i] - cache[40 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[128 + i] = PA[2] * cache[79 + i] - PG[2] * cache[79 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[131 + i] = PA[1] * cache[87 + i] - PG[1] * cache[87 + i + 1] + 1 * half_xi_ * (cache[50 + i] - cache[50 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[134 + i] = PA[1] * cache[91 + i] - PG[1] * cache[91 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[137 + i] = PA[2] * cache[91 + i] - PG[2] * cache[91 + i + 1] + 3 * half_xi_ * (cache[50 + i] - cache[50 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[140 + i] = PA[0] * cache[95 + i] - PG[0] * cache[95 + i + 1] + 4 * half_xi_ * (cache[55 + i] - cache[55 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[142 + i] = PA[1] * cache[95 + i] - PG[1] * cache[95 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[144 + i] = PA[2] * cache[95 + i] - PG[2] * cache[95 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[146 + i] = PA[0] * cache[104 + i] - PG[0] * cache[104 + i + 1] + 2 * half_xi_ * (cache[67 + i] - cache[67 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[148 + i] = PA[1] * cache[101 + i] - PG[1] * cache[101 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[150 + i] = PA[0] * cache[110 + i] - PG[0] * cache[110 + i + 1] + 2 * half_xi_ * (cache[75 + i] - cache[75 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[152 + i] = PA[0] * cache[113 + i] - PG[0] * cache[113 + i + 1] + 1 * half_xi_ * (cache[79 + i] - cache[79 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[154 + i] = PA[2] * cache[104 + i] - PG[2] * cache[104 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[156 + i] = PA[1] * cache[110 + i] - PG[1] * cache[110 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[158 + i] = PA[0] * cache[122 + i] - PG[0] * cache[122 + i + 1] + 1 * half_xi_ * (cache[91 + i] - cache[91 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[160 + i] = PA[0] * cache[125 + i] - PG[0] * cache[125 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[162 + i] = PA[0] * cache[128 + i] - PG[0] * cache[128 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[164 + i] = PA[0] * cache[131 + i] - PG[0] * cache[131 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[166 + i] = PA[0] * cache[134 + i] - PG[0] * cache[134 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[168 + i] = PA[0] * cache[137 + i] - PG[0] * cache[137 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[170 + i] = PA[1] * cache[125 + i] - PG[1] * cache[125 + i + 1] + 4 * half_xi_ * (cache[79 + i] - cache[79 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[172 + i] = PA[2] * cache[125 + i] - PG[2] * cache[125 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[174 + i] = PA[1] * cache[131 + i] - PG[1] * cache[131 + i + 1] + 2 * half_xi_ * (cache[87 + i] - cache[87 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[176 + i] = PA[1] * cache[134 + i] - PG[1] * cache[134 + i + 1] + 1 * half_xi_ * (cache[91 + i] - cache[91 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[178 + i] = PA[1] * cache[137 + i] - PG[1] * cache[137 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[180 + i] = PA[2] * cache[137 + i] - PG[2] * cache[137 + i + 1] + 4 * half_xi_ * (cache[91 + i] - cache[91 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[182 + i] = PA[0] * cache[140 + i] - PG[0] * cache[140 + i + 1] + 5 * half_xi_ * (cache[95 + i] - cache[95 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[183 + i] = PA[1] * cache[140 + i] - PG[1] * cache[140 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[184 + i] = PA[2] * cache[140 + i] - PG[2] * cache[140 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[185 + i] = PA[0] * cache[146 + i] - PG[0] * cache[146 + i + 1] + 3 * half_xi_ * (cache[104 + i] - cache[104 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[186 + i] = PA[1] * cache[144 + i] - PG[1] * cache[144 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[187 + i] = PA[0] * cache[150 + i] - PG[0] * cache[150 + i + 1] + 3 * half_xi_ * (cache[110 + i] - cache[110 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[188 + i] = PA[0] * cache[152 + i] - PG[0] * cache[152 + i + 1] + 2 * half_xi_ * (cache[113 + i] - cache[113 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[189 + i] = PA[2] * cache[146 + i] - PG[2] * cache[146 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[190 + i] = PA[1] * cache[150 + i] - PG[1] * cache[150 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[191 + i] = PA[0] * cache[158 + i] - PG[0] * cache[158 + i + 1] + 2 * half_xi_ * (cache[122 + i] - cache[122 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[192 + i] = PA[0] * cache[160 + i] - PG[0] * cache[160 + i + 1] + 1 * half_xi_ * (cache[125 + i] - cache[125 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[193 + i] = PA[2] * cache[152 + i] - PG[2] * cache[152 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[194 + i] = PA[0] * cache[164 + i] - PG[0] * cache[164 + i + 1] + 1 * half_xi_ * (cache[131 + i] - cache[131 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[195 + i] = PA[1] * cache[158 + i] - PG[1] * cache[158 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[196 + i] = PA[0] * cache[168 + i] - PG[0] * cache[168 + i + 1] + 1 * half_xi_ * (cache[137 + i] - cache[137 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[197 + i] = PA[0] * cache[170 + i] - PG[0] * cache[170 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[198 + i] = PA[0] * cache[172 + i] - PG[0] * cache[172 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[199 + i] = PA[0] * cache[174 + i] - PG[0] * cache[174 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[200 + i] = PA[0] * cache[176 + i] - PG[0] * cache[176 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[201 + i] = PA[0] * cache[178 + i] - PG[0] * cache[178 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[202 + i] = PA[0] * cache[180 + i] - PG[0] * cache[180 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[203 + i] = PA[1] * cache[170 + i] - PG[1] * cache[170 + i + 1] + 5 * half_xi_ * (cache[125 + i] - cache[125 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[204 + i] = PA[2] * cache[170 + i] - PG[2] * cache[170 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[205 + i] = PA[1] * cache[174 + i] - PG[1] * cache[174 + i + 1] + 3 * half_xi_ * (cache[131 + i] - cache[131 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[206 + i] = PA[1] * cache[176 + i] - PG[1] * cache[176 + i + 1] + 2 * half_xi_ * (cache[134 + i] - cache[134 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[207 + i] = PA[1] * cache[178 + i] - PG[1] * cache[178 + i + 1] + 1 * half_xi_ * (cache[137 + i] - cache[137 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[208 + i] = PA[1] * cache[180 + i] - PG[1] * cache[180 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[209 + i] = PA[2] * cache[180 + i] - PG[2] * cache[180 + i + 1] + 5 * half_xi_ * (cache[137 + i] - cache[137 + i + 1]);
    buf[0] += cache[55];
    buf[1] += cache[59];
    buf[2] += cache[63];
    buf[3] += cache[67];
    buf[4] += cache[71];
    buf[5] += cache[75];
    buf[6] += cache[79];
    buf[7] += cache[83];
    buf[8] += cache[87];
    buf[9] += cache[91];
    buf[10] += cache[95];
    buf[11] += cache[98];
    buf[12] += cache[101];
    buf[13] += cache[104];
    buf[14] += cache[107];
    buf[15] += cache[110];
    buf[16] += cache[113];
    buf[17] += cache[116];
    buf[18] += cache[119];
    buf[19] += cache[122];
    buf[20] += cache[125];
    buf[21] += cache[128];
    buf[22] += cache[131];
    buf[23] += cache[134];
    buf[24] += cache[137];
    buf[25] += cache[140];
    buf[26] += cache[142];
    buf[27] += cache[144];
    buf[28] += cache[146];
    buf[29] += cache[148];
    buf[30] += cache[150];
    buf[31] += cache[152];
    buf[32] += cache[154];
    buf[33] += cache[156];
    buf[34] += cache[158];
    buf[35] += cache[160];
    buf[36] += cache[162];
    buf[37] += cache[164];
    buf[38] += cache[166];
    buf[39] += cache[168];
    buf[40] += cache[170];
    buf[41] += cache[172];
    buf[42] += cache[174];
    buf[43] += cache[176];
    buf[44] += cache[178];
    buf[45] += cache[180];
    buf[46] += cache[182];
    buf[47] += cache[183];
    buf[48] += cache[184];
    buf[49] += cache[185];
    buf[50] += cache[186];
    buf[51] += cache[187];
    buf[52] += cache[188];
    buf[53] += cache[189];
    buf[54] += cache[190];
    buf[55] += cache[191];
    buf[56] += cache[192];
    buf[57] += cache[193];
    buf[58] += cache[194];
    buf[59] += cache[195];
    buf[60] += cache[196];
    buf[61] += cache[197];
    buf[62] += cache[198];
    buf[63] += cache[199];
    buf[64] += cache[200];
    buf[65] += cache[201];
    buf[66] += cache[202];
    buf[67] += cache[203];
    buf[68] += cache[204];
    buf[69] += cache[205];
    buf[70] += cache[206];
    buf[71] += cache[207];
    buf[72] += cache[208];
    buf[73] += cache[209];
    return;
}

static void nuc_VRR_4_0(double *buf, double *cache, double coe, double K_ab, double half_xi_, double xi, const double *PA,
                        const double *PG, double PG_dist2, double sr_frac, double lr_frac, double omega_)
{
    double theta;
    double T_0, T_1;
    double scheme_1[10], scheme_0[10];
    double scheme_0_0_0[5];
    T_0 = xi * PG_dist2;
    if (fabs(omega_) < 1E-8)
    {
        cal_boys(4, T_0, &scheme_0[0]);
        for (int i = 0; i <= 4; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * lr_frac * scheme_0[i];
    }
    else
    {
        theta = sqrt(omega_ * omega_ / (omega_ * omega_ + xi));
        T_1 = xi * PG_dist2 * theta * theta;
        cal_boys(4, T_1, &scheme_1[0]);
        cal_boys(4, T_0, &scheme_0[0]);
        for (int i = 0; i <= 4; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * (lr_frac * scheme_0[i] + sr_frac * theta * pow(theta, 2 * i) * scheme_1[i]);
    }
    for (int i = 0; i < 4; i++)
        cache[5 + i] = PA[0] * cache[0 + i] - PG[0] * cache[0 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[9 + i] = PA[1] * cache[0 + i] - PG[1] * cache[0 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[13 + i] = PA[2] * cache[0 + i] - PG[2] * cache[0 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[17 + i] = PA[0] * cache[5 + i] - PG[0] * cache[5 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[20 + i] = PA[0] * cache[9 + i] - PG[0] * cache[9 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[23 + i] = PA[0] * cache[13 + i] - PG[0] * cache[13 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[26 + i] = PA[1] * cache[9 + i] - PG[1] * cache[9 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[29 + i] = PA[1] * cache[13 + i] - PG[1] * cache[13 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[32 + i] = PA[2] * cache[13 + i] - PG[2] * cache[13 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[35 + i] = PA[0] * cache[17 + i] - PG[0] * cache[17 + i + 1] + 2 * half_xi_ * (cache[5 + i] - cache[5 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[37 + i] = PA[1] * cache[17 + i] - PG[1] * cache[17 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[39 + i] = PA[2] * cache[17 + i] - PG[2] * cache[17 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[41 + i] = PA[0] * cache[26 + i] - PG[0] * cache[26 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[43 + i] = PA[0] * cache[29 + i] - PG[0] * cache[29 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[45 + i] = PA[0] * cache[32 + i] - PG[0] * cache[32 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[47 + i] = PA[1] * cache[26 + i] - PG[1] * cache[26 + i + 1] + 2 * half_xi_ * (cache[9 + i] - cache[9 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[49 + i] = PA[2] * cache[26 + i] - PG[2] * cache[26 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[51 + i] = PA[1] * cache[32 + i] - PG[1] * cache[32 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[53 + i] = PA[2] * cache[32 + i] - PG[2] * cache[32 + i + 1] + 2 * half_xi_ * (cache[13 + i] - cache[13 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[55 + i] = PA[0] * cache[35 + i] - PG[0] * cache[35 + i + 1] + 3 * half_xi_ * (cache[17 + i] - cache[17 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[56 + i] = PA[1] * cache[35 + i] - PG[1] * cache[35 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[57 + i] = PA[2] * cache[35 + i] - PG[2] * cache[35 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[58 + i] = PA[0] * cache[41 + i] - PG[0] * cache[41 + i + 1] + 1 * half_xi_ * (cache[26 + i] - cache[26 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[59 + i] = PA[1] * cache[39 + i] - PG[1] * cache[39 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[60 + i] = PA[0] * cache[45 + i] - PG[0] * cache[45 + i + 1] + 1 * half_xi_ * (cache[32 + i] - cache[32 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[61 + i] = PA[0] * cache[47 + i] - PG[0] * cache[47 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[62 + i] = PA[0] * cache[49 + i] - PG[0] * cache[49 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[63 + i] = PA[0] * cache[51 + i] - PG[0] * cache[51 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[64 + i] = PA[0] * cache[53 + i] - PG[0] * cache[53 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[65 + i] = PA[1] * cache[47 + i] - PG[1] * cache[47 + i + 1] + 3 * half_xi_ * (cache[26 + i] - cache[26 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[66 + i] = PA[2] * cache[47 + i] - PG[2] * cache[47 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[67 + i] = PA[1] * cache[51 + i] - PG[1] * cache[51 + i + 1] + 1 * half_xi_ * (cache[32 + i] - cache[32 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[68 + i] = PA[1] * cache[53 + i] - PG[1] * cache[53 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[69 + i] = PA[2] * cache[53 + i] - PG[2] * cache[53 + i + 1] + 3 * half_xi_ * (cache[32 + i] - cache[32 + i + 1]);
    buf[0] += cache[55];
    buf[1] += cache[56];
    buf[2] += cache[57];
    buf[3] += cache[58];
    buf[4] += cache[59];
    buf[5] += cache[60];
    buf[6] += cache[61];
    buf[7] += cache[62];
    buf[8] += cache[63];
    buf[9] += cache[64];
    buf[10] += cache[65];
    buf[11] += cache[66];
    buf[12] += cache[67];
    buf[13] += cache[68];
    buf[14] += cache[69];
    return;
}

static void nuc_VRR_4_1(double *buf, double *cache, double coe, double K_ab, double half_xi_, double xi, const double *PA,
                        const double *PG, double PG_dist2, double sr_frac, double lr_frac, double omega_)
{
    double theta;
    double T_0, T_1;
    double scheme_1[10], scheme_0[10];
    double scheme_0_0_0[6];
    T_0 = xi * PG_dist2;
    if (fabs(omega_) < 1E-8)
    {
        cal_boys(5, T_0, &scheme_0[0]);
        cal_boys(5, T_0, &scheme_0[0]);
        for (int i = 0; i <= 5; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * lr_frac * scheme_0[i];
    }
    else
    {
        theta = sqrt(omega_ * omega_ / (omega_ * omega_ + xi));
        T_1 = xi * PG_dist2 * theta * theta;
        cal_boys(5, T_1, &scheme_1[0]);
        for (int i = 0; i <= 5; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * (lr_frac * scheme_0[i] + sr_frac * theta * pow(theta, 2 * i) * scheme_1[i]);
    }
    for (int i = 0; i < 5; i++)
        cache[6 + i] = PA[0] * cache[0 + i] - PG[0] * cache[0 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[11 + i] = PA[1] * cache[0 + i] - PG[1] * cache[0 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[16 + i] = PA[2] * cache[0 + i] - PG[2] * cache[0 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[21 + i] = PA[0] * cache[6 + i] - PG[0] * cache[6 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[25 + i] = PA[0] * cache[11 + i] - PG[0] * cache[11 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[29 + i] = PA[0] * cache[16 + i] - PG[0] * cache[16 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[33 + i] = PA[1] * cache[11 + i] - PG[1] * cache[11 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[37 + i] = PA[1] * cache[16 + i] - PG[1] * cache[16 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[41 + i] = PA[2] * cache[16 + i] - PG[2] * cache[16 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[45 + i] = PA[0] * cache[21 + i] - PG[0] * cache[21 + i + 1] + 2 * half_xi_ * (cache[6 + i] - cache[6 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[48 + i] = PA[1] * cache[21 + i] - PG[1] * cache[21 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[51 + i] = PA[2] * cache[21 + i] - PG[2] * cache[21 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[54 + i] = PA[0] * cache[33 + i] - PG[0] * cache[33 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[57 + i] = PA[0] * cache[37 + i] - PG[0] * cache[37 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[60 + i] = PA[0] * cache[41 + i] - PG[0] * cache[41 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[63 + i] = PA[1] * cache[33 + i] - PG[1] * cache[33 + i + 1] + 2 * half_xi_ * (cache[11 + i] - cache[11 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[66 + i] = PA[2] * cache[33 + i] - PG[2] * cache[33 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[69 + i] = PA[1] * cache[41 + i] - PG[1] * cache[41 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[72 + i] = PA[2] * cache[41 + i] - PG[2] * cache[41 + i + 1] + 2 * half_xi_ * (cache[16 + i] - cache[16 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[75 + i] = PA[0] * cache[45 + i] - PG[0] * cache[45 + i + 1] + 3 * half_xi_ * (cache[21 + i] - cache[21 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[77 + i] = PA[1] * cache[45 + i] - PG[1] * cache[45 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[79 + i] = PA[2] * cache[45 + i] - PG[2] * cache[45 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[81 + i] = PA[0] * cache[54 + i] - PG[0] * cache[54 + i + 1] + 1 * half_xi_ * (cache[33 + i] - cache[33 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[83 + i] = PA[1] * cache[51 + i] - PG[1] * cache[51 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[85 + i] = PA[0] * cache[60 + i] - PG[0] * cache[60 + i + 1] + 1 * half_xi_ * (cache[41 + i] - cache[41 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[87 + i] = PA[0] * cache[63 + i] - PG[0] * cache[63 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[89 + i] = PA[0] * cache[66 + i] - PG[0] * cache[66 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[91 + i] = PA[0] * cache[69 + i] - PG[0] * cache[69 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[93 + i] = PA[0] * cache[72 + i] - PG[0] * cache[72 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[95 + i] = PA[1] * cache[63 + i] - PG[1] * cache[63 + i + 1] + 3 * half_xi_ * (cache[33 + i] - cache[33 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[97 + i] = PA[2] * cache[63 + i] - PG[2] * cache[63 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[99 + i] = PA[1] * cache[69 + i] - PG[1] * cache[69 + i + 1] + 1 * half_xi_ * (cache[41 + i] - cache[41 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[101 + i] = PA[1] * cache[72 + i] - PG[1] * cache[72 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[103 + i] = PA[2] * cache[72 + i] - PG[2] * cache[72 + i + 1] + 3 * half_xi_ * (cache[41 + i] - cache[41 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[105 + i] = PA[0] * cache[75 + i] - PG[0] * cache[75 + i + 1] + 4 * half_xi_ * (cache[45 + i] - cache[45 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[106 + i] = PA[1] * cache[75 + i] - PG[1] * cache[75 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[107 + i] = PA[2] * cache[75 + i] - PG[2] * cache[75 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[108 + i] = PA[0] * cache[81 + i] - PG[0] * cache[81 + i + 1] + 2 * half_xi_ * (cache[54 + i] - cache[54 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[109 + i] = PA[1] * cache[79 + i] - PG[1] * cache[79 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[110 + i] = PA[0] * cache[85 + i] - PG[0] * cache[85 + i + 1] + 2 * half_xi_ * (cache[60 + i] - cache[60 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[111 + i] = PA[0] * cache[87 + i] - PG[0] * cache[87 + i + 1] + 1 * half_xi_ * (cache[63 + i] - cache[63 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[112 + i] = PA[2] * cache[81 + i] - PG[2] * cache[81 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[113 + i] = PA[1] * cache[85 + i] - PG[1] * cache[85 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[114 + i] = PA[0] * cache[93 + i] - PG[0] * cache[93 + i + 1] + 1 * half_xi_ * (cache[72 + i] - cache[72 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[115 + i] = PA[0] * cache[95 + i] - PG[0] * cache[95 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[116 + i] = PA[0] * cache[97 + i] - PG[0] * cache[97 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[117 + i] = PA[0] * cache[99 + i] - PG[0] * cache[99 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[118 + i] = PA[0] * cache[101 + i] - PG[0] * cache[101 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[119 + i] = PA[0] * cache[103 + i] - PG[0] * cache[103 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[120 + i] = PA[1] * cache[95 + i] - PG[1] * cache[95 + i + 1] + 4 * half_xi_ * (cache[63 + i] - cache[63 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[121 + i] = PA[2] * cache[95 + i] - PG[2] * cache[95 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[122 + i] = PA[1] * cache[99 + i] - PG[1] * cache[99 + i + 1] + 2 * half_xi_ * (cache[69 + i] - cache[69 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[123 + i] = PA[1] * cache[101 + i] - PG[1] * cache[101 + i + 1] + 1 * half_xi_ * (cache[72 + i] - cache[72 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[124 + i] = PA[1] * cache[103 + i] - PG[1] * cache[103 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[125 + i] = PA[2] * cache[103 + i] - PG[2] * cache[103 + i + 1] + 4 * half_xi_ * (cache[72 + i] - cache[72 + i + 1]);
    buf[0] += cache[75];
    buf[1] += cache[77];
    buf[2] += cache[79];
    buf[3] += cache[81];
    buf[4] += cache[83];
    buf[5] += cache[85];
    buf[6] += cache[87];
    buf[7] += cache[89];
    buf[8] += cache[91];
    buf[9] += cache[93];
    buf[10] += cache[95];
    buf[11] += cache[97];
    buf[12] += cache[99];
    buf[13] += cache[101];
    buf[14] += cache[103];
    buf[15] += cache[105];
    buf[16] += cache[106];
    buf[17] += cache[107];
    buf[18] += cache[108];
    buf[19] += cache[109];
    buf[20] += cache[110];
    buf[21] += cache[111];
    buf[22] += cache[112];
    buf[23] += cache[113];
    buf[24] += cache[114];
    buf[25] += cache[115];
    buf[26] += cache[116];
    buf[27] += cache[117];
    buf[28] += cache[118];
    buf[29] += cache[119];
    buf[30] += cache[120];
    buf[31] += cache[121];
    buf[32] += cache[122];
    buf[33] += cache[123];
    buf[34] += cache[124];
    buf[35] += cache[125];
    return;
}

static void nuc_VRR_4_2(double *buf, double *cache, double coe, double K_ab, double half_xi_, double xi, const double *PA,
                        const double *PG, double PG_dist2, double sr_frac, double lr_frac, double omega_)
{
    double theta;
    double T_0, T_1;
    double scheme_1[10], scheme_0[10];
    double scheme_0_0_0[7];
    T_0 = xi * PG_dist2;
    if (fabs(omega_) < 1E-8)
    {
        cal_boys(6, T_0, &scheme_0[0]);
        for (int i = 0; i <= 6; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * lr_frac * scheme_0[i];
    }
    else
    {
        theta = sqrt(omega_ * omega_ / (omega_ * omega_ + xi));
        T_1 = xi * PG_dist2 * theta * theta;
        cal_boys(6, T_1, &scheme_1[0]);
        cal_boys(6, T_0, &scheme_0[0]);
        for (int i = 0; i <= 6; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * (lr_frac * scheme_0[i] + sr_frac * theta * pow(theta, 2 * i) * scheme_1[i]);
    }
    for (int i = 0; i < 6; i++)
        cache[7 + i] = PA[0] * cache[0 + i] - PG[0] * cache[0 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[13 + i] = PA[1] * cache[0 + i] - PG[1] * cache[0 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[19 + i] = PA[2] * cache[0 + i] - PG[2] * cache[0 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[25 + i] = PA[0] * cache[7 + i] - PG[0] * cache[7 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[30 + i] = PA[0] * cache[13 + i] - PG[0] * cache[13 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[35 + i] = PA[0] * cache[19 + i] - PG[0] * cache[19 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[40 + i] = PA[1] * cache[13 + i] - PG[1] * cache[13 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[45 + i] = PA[1] * cache[19 + i] - PG[1] * cache[19 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[50 + i] = PA[2] * cache[19 + i] - PG[2] * cache[19 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[55 + i] = PA[0] * cache[25 + i] - PG[0] * cache[25 + i + 1] + 2 * half_xi_ * (cache[7 + i] - cache[7 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[59 + i] = PA[1] * cache[25 + i] - PG[1] * cache[25 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[63 + i] = PA[2] * cache[25 + i] - PG[2] * cache[25 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[67 + i] = PA[0] * cache[40 + i] - PG[0] * cache[40 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[71 + i] = PA[0] * cache[45 + i] - PG[0] * cache[45 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[75 + i] = PA[0] * cache[50 + i] - PG[0] * cache[50 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[79 + i] = PA[1] * cache[40 + i] - PG[1] * cache[40 + i + 1] + 2 * half_xi_ * (cache[13 + i] - cache[13 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[83 + i] = PA[2] * cache[40 + i] - PG[2] * cache[40 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[87 + i] = PA[1] * cache[50 + i] - PG[1] * cache[50 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[91 + i] = PA[2] * cache[50 + i] - PG[2] * cache[50 + i + 1] + 2 * half_xi_ * (cache[19 + i] - cache[19 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[95 + i] = PA[0] * cache[55 + i] - PG[0] * cache[55 + i + 1] + 3 * half_xi_ * (cache[25 + i] - cache[25 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[98 + i] = PA[1] * cache[55 + i] - PG[1] * cache[55 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[101 + i] = PA[2] * cache[55 + i] - PG[2] * cache[55 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[104 + i] = PA[0] * cache[67 + i] - PG[0] * cache[67 + i + 1] + 1 * half_xi_ * (cache[40 + i] - cache[40 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[107 + i] = PA[1] * cache[63 + i] - PG[1] * cache[63 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[110 + i] = PA[0] * cache[75 + i] - PG[0] * cache[75 + i + 1] + 1 * half_xi_ * (cache[50 + i] - cache[50 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[113 + i] = PA[0] * cache[79 + i] - PG[0] * cache[79 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[116 + i] = PA[0] * cache[83 + i] - PG[0] * cache[83 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[119 + i] = PA[0] * cache[87 + i] - PG[0] * cache[87 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[122 + i] = PA[0] * cache[91 + i] - PG[0] * cache[91 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[125 + i] = PA[1] * cache[79 + i] - PG[1] * cache[79 + i + 1] + 3 * half_xi_ * (cache[40 + i] - cache[40 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[128 + i] = PA[2] * cache[79 + i] - PG[2] * cache[79 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[131 + i] = PA[1] * cache[87 + i] - PG[1] * cache[87 + i + 1] + 1 * half_xi_ * (cache[50 + i] - cache[50 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[134 + i] = PA[1] * cache[91 + i] - PG[1] * cache[91 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[137 + i] = PA[2] * cache[91 + i] - PG[2] * cache[91 + i + 1] + 3 * half_xi_ * (cache[50 + i] - cache[50 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[140 + i] = PA[0] * cache[95 + i] - PG[0] * cache[95 + i + 1] + 4 * half_xi_ * (cache[55 + i] - cache[55 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[142 + i] = PA[1] * cache[95 + i] - PG[1] * cache[95 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[144 + i] = PA[2] * cache[95 + i] - PG[2] * cache[95 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[146 + i] = PA[0] * cache[104 + i] - PG[0] * cache[104 + i + 1] + 2 * half_xi_ * (cache[67 + i] - cache[67 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[148 + i] = PA[1] * cache[101 + i] - PG[1] * cache[101 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[150 + i] = PA[0] * cache[110 + i] - PG[0] * cache[110 + i + 1] + 2 * half_xi_ * (cache[75 + i] - cache[75 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[152 + i] = PA[0] * cache[113 + i] - PG[0] * cache[113 + i + 1] + 1 * half_xi_ * (cache[79 + i] - cache[79 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[154 + i] = PA[2] * cache[104 + i] - PG[2] * cache[104 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[156 + i] = PA[1] * cache[110 + i] - PG[1] * cache[110 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[158 + i] = PA[0] * cache[122 + i] - PG[0] * cache[122 + i + 1] + 1 * half_xi_ * (cache[91 + i] - cache[91 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[160 + i] = PA[0] * cache[125 + i] - PG[0] * cache[125 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[162 + i] = PA[0] * cache[128 + i] - PG[0] * cache[128 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[164 + i] = PA[0] * cache[131 + i] - PG[0] * cache[131 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[166 + i] = PA[0] * cache[134 + i] - PG[0] * cache[134 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[168 + i] = PA[0] * cache[137 + i] - PG[0] * cache[137 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[170 + i] = PA[1] * cache[125 + i] - PG[1] * cache[125 + i + 1] + 4 * half_xi_ * (cache[79 + i] - cache[79 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[172 + i] = PA[2] * cache[125 + i] - PG[2] * cache[125 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[174 + i] = PA[1] * cache[131 + i] - PG[1] * cache[131 + i + 1] + 2 * half_xi_ * (cache[87 + i] - cache[87 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[176 + i] = PA[1] * cache[134 + i] - PG[1] * cache[134 + i + 1] + 1 * half_xi_ * (cache[91 + i] - cache[91 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[178 + i] = PA[1] * cache[137 + i] - PG[1] * cache[137 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[180 + i] = PA[2] * cache[137 + i] - PG[2] * cache[137 + i + 1] + 4 * half_xi_ * (cache[91 + i] - cache[91 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[182 + i] = PA[0] * cache[140 + i] - PG[0] * cache[140 + i + 1] + 5 * half_xi_ * (cache[95 + i] - cache[95 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[183 + i] = PA[1] * cache[140 + i] - PG[1] * cache[140 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[184 + i] = PA[2] * cache[140 + i] - PG[2] * cache[140 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[185 + i] = PA[0] * cache[146 + i] - PG[0] * cache[146 + i + 1] + 3 * half_xi_ * (cache[104 + i] - cache[104 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[186 + i] = PA[1] * cache[144 + i] - PG[1] * cache[144 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[187 + i] = PA[0] * cache[150 + i] - PG[0] * cache[150 + i + 1] + 3 * half_xi_ * (cache[110 + i] - cache[110 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[188 + i] = PA[0] * cache[152 + i] - PG[0] * cache[152 + i + 1] + 2 * half_xi_ * (cache[113 + i] - cache[113 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[189 + i] = PA[2] * cache[146 + i] - PG[2] * cache[146 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[190 + i] = PA[1] * cache[150 + i] - PG[1] * cache[150 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[191 + i] = PA[0] * cache[158 + i] - PG[0] * cache[158 + i + 1] + 2 * half_xi_ * (cache[122 + i] - cache[122 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[192 + i] = PA[0] * cache[160 + i] - PG[0] * cache[160 + i + 1] + 1 * half_xi_ * (cache[125 + i] - cache[125 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[193 + i] = PA[2] * cache[152 + i] - PG[2] * cache[152 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[194 + i] = PA[0] * cache[164 + i] - PG[0] * cache[164 + i + 1] + 1 * half_xi_ * (cache[131 + i] - cache[131 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[195 + i] = PA[1] * cache[158 + i] - PG[1] * cache[158 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[196 + i] = PA[0] * cache[168 + i] - PG[0] * cache[168 + i + 1] + 1 * half_xi_ * (cache[137 + i] - cache[137 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[197 + i] = PA[0] * cache[170 + i] - PG[0] * cache[170 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[198 + i] = PA[0] * cache[172 + i] - PG[0] * cache[172 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[199 + i] = PA[0] * cache[174 + i] - PG[0] * cache[174 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[200 + i] = PA[0] * cache[176 + i] - PG[0] * cache[176 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[201 + i] = PA[0] * cache[178 + i] - PG[0] * cache[178 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[202 + i] = PA[0] * cache[180 + i] - PG[0] * cache[180 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[203 + i] = PA[1] * cache[170 + i] - PG[1] * cache[170 + i + 1] + 5 * half_xi_ * (cache[125 + i] - cache[125 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[204 + i] = PA[2] * cache[170 + i] - PG[2] * cache[170 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[205 + i] = PA[1] * cache[174 + i] - PG[1] * cache[174 + i + 1] + 3 * half_xi_ * (cache[131 + i] - cache[131 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[206 + i] = PA[1] * cache[176 + i] - PG[1] * cache[176 + i + 1] + 2 * half_xi_ * (cache[134 + i] - cache[134 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[207 + i] = PA[1] * cache[178 + i] - PG[1] * cache[178 + i + 1] + 1 * half_xi_ * (cache[137 + i] - cache[137 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[208 + i] = PA[1] * cache[180 + i] - PG[1] * cache[180 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[209 + i] = PA[2] * cache[180 + i] - PG[2] * cache[180 + i + 1] + 5 * half_xi_ * (cache[137 + i] - cache[137 + i + 1]);
    buf[0] += cache[95];
    buf[1] += cache[98];
    buf[2] += cache[101];
    buf[3] += cache[104];
    buf[4] += cache[107];
    buf[5] += cache[110];
    buf[6] += cache[113];
    buf[7] += cache[116];
    buf[8] += cache[119];
    buf[9] += cache[122];
    buf[10] += cache[125];
    buf[11] += cache[128];
    buf[12] += cache[131];
    buf[13] += cache[134];
    buf[14] += cache[137];
    buf[15] += cache[140];
    buf[16] += cache[142];
    buf[17] += cache[144];
    buf[18] += cache[146];
    buf[19] += cache[148];
    buf[20] += cache[150];
    buf[21] += cache[152];
    buf[22] += cache[154];
    buf[23] += cache[156];
    buf[24] += cache[158];
    buf[25] += cache[160];
    buf[26] += cache[162];
    buf[27] += cache[164];
    buf[28] += cache[166];
    buf[29] += cache[168];
    buf[30] += cache[170];
    buf[31] += cache[172];
    buf[32] += cache[174];
    buf[33] += cache[176];
    buf[34] += cache[178];
    buf[35] += cache[180];
    buf[36] += cache[182];
    buf[37] += cache[183];
    buf[38] += cache[184];
    buf[39] += cache[185];
    buf[40] += cache[186];
    buf[41] += cache[187];
    buf[42] += cache[188];
    buf[43] += cache[189];
    buf[44] += cache[190];
    buf[45] += cache[191];
    buf[46] += cache[192];
    buf[47] += cache[193];
    buf[48] += cache[194];
    buf[49] += cache[195];
    buf[50] += cache[196];
    buf[51] += cache[197];
    buf[52] += cache[198];
    buf[53] += cache[199];
    buf[54] += cache[200];
    buf[55] += cache[201];
    buf[56] += cache[202];
    buf[57] += cache[203];
    buf[58] += cache[204];
    buf[59] += cache[205];
    buf[60] += cache[206];
    buf[61] += cache[207];
    buf[62] += cache[208];
    buf[63] += cache[209];
    return;
}

static void nuc_VRR_4_3(double *buf, double *cache, double coe, double K_ab, double half_xi_, double xi, const double *PA,
                        const double *PG, double PG_dist2, double sr_frac, double lr_frac, double omega_)
{
    double theta;
    double T_0, T_1;
    double scheme_1[10], scheme_0[10];
    double scheme_0_0_0[8];
    T_0 = xi * PG_dist2;
    if (fabs(omega_) < 1E-8)
    {
        cal_boys(7, T_0, &scheme_0[0]);
        for (int i = 0; i <= 7; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * lr_frac * scheme_0[i];
    }
    else
    {
        theta = sqrt(omega_ * omega_ / (omega_ * omega_ + xi));
        T_1 = xi * PG_dist2 * theta * theta;
        cal_boys(7, T_1, &scheme_1[0]);
        cal_boys(7, T_0, &scheme_0[0]);
        for (int i = 0; i <= 7; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * (lr_frac * scheme_0[i] + sr_frac * theta * pow(theta, 2 * i) * scheme_1[i]);
    }
    for (int i = 0; i < 7; i++)
        cache[8 + i] = PA[0] * cache[0 + i] - PG[0] * cache[0 + i + 1];
    for (int i = 0; i < 7; i++)
        cache[15 + i] = PA[1] * cache[0 + i] - PG[1] * cache[0 + i + 1];
    for (int i = 0; i < 7; i++)
        cache[22 + i] = PA[2] * cache[0 + i] - PG[2] * cache[0 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[29 + i] = PA[0] * cache[8 + i] - PG[0] * cache[8 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 6; i++)
        cache[35 + i] = PA[0] * cache[15 + i] - PG[0] * cache[15 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[41 + i] = PA[0] * cache[22 + i] - PG[0] * cache[22 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[47 + i] = PA[1] * cache[15 + i] - PG[1] * cache[15 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 6; i++)
        cache[53 + i] = PA[1] * cache[22 + i] - PG[1] * cache[22 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[59 + i] = PA[2] * cache[22 + i] - PG[2] * cache[22 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[65 + i] = PA[0] * cache[29 + i] - PG[0] * cache[29 + i + 1] + 2 * half_xi_ * (cache[8 + i] - cache[8 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[70 + i] = PA[1] * cache[29 + i] - PG[1] * cache[29 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[75 + i] = PA[2] * cache[29 + i] - PG[2] * cache[29 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[80 + i] = PA[0] * cache[47 + i] - PG[0] * cache[47 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[85 + i] = PA[0] * cache[53 + i] - PG[0] * cache[53 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[90 + i] = PA[0] * cache[59 + i] - PG[0] * cache[59 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[95 + i] = PA[1] * cache[47 + i] - PG[1] * cache[47 + i + 1] + 2 * half_xi_ * (cache[15 + i] - cache[15 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[100 + i] = PA[2] * cache[47 + i] - PG[2] * cache[47 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[105 + i] = PA[1] * cache[59 + i] - PG[1] * cache[59 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[110 + i] = PA[2] * cache[59 + i] - PG[2] * cache[59 + i + 1] + 2 * half_xi_ * (cache[22 + i] - cache[22 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[115 + i] = PA[0] * cache[65 + i] - PG[0] * cache[65 + i + 1] + 3 * half_xi_ * (cache[29 + i] - cache[29 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[119 + i] = PA[1] * cache[65 + i] - PG[1] * cache[65 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[123 + i] = PA[2] * cache[65 + i] - PG[2] * cache[65 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[127 + i] = PA[0] * cache[80 + i] - PG[0] * cache[80 + i + 1] + 1 * half_xi_ * (cache[47 + i] - cache[47 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[131 + i] = PA[1] * cache[75 + i] - PG[1] * cache[75 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[135 + i] = PA[0] * cache[90 + i] - PG[0] * cache[90 + i + 1] + 1 * half_xi_ * (cache[59 + i] - cache[59 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[139 + i] = PA[0] * cache[95 + i] - PG[0] * cache[95 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[143 + i] = PA[0] * cache[100 + i] - PG[0] * cache[100 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[147 + i] = PA[0] * cache[105 + i] - PG[0] * cache[105 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[151 + i] = PA[0] * cache[110 + i] - PG[0] * cache[110 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[155 + i] = PA[1] * cache[95 + i] - PG[1] * cache[95 + i + 1] + 3 * half_xi_ * (cache[47 + i] - cache[47 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[159 + i] = PA[2] * cache[95 + i] - PG[2] * cache[95 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[163 + i] = PA[1] * cache[105 + i] - PG[1] * cache[105 + i + 1] + 1 * half_xi_ * (cache[59 + i] - cache[59 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[167 + i] = PA[1] * cache[110 + i] - PG[1] * cache[110 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[171 + i] = PA[2] * cache[110 + i] - PG[2] * cache[110 + i + 1] + 3 * half_xi_ * (cache[59 + i] - cache[59 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[175 + i] = PA[0] * cache[115 + i] - PG[0] * cache[115 + i + 1] + 4 * half_xi_ * (cache[65 + i] - cache[65 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[178 + i] = PA[1] * cache[115 + i] - PG[1] * cache[115 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[181 + i] = PA[2] * cache[115 + i] - PG[2] * cache[115 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[184 + i] = PA[0] * cache[127 + i] - PG[0] * cache[127 + i + 1] + 2 * half_xi_ * (cache[80 + i] - cache[80 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[187 + i] = PA[1] * cache[123 + i] - PG[1] * cache[123 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[190 + i] = PA[0] * cache[135 + i] - PG[0] * cache[135 + i + 1] + 2 * half_xi_ * (cache[90 + i] - cache[90 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[193 + i] = PA[0] * cache[139 + i] - PG[0] * cache[139 + i + 1] + 1 * half_xi_ * (cache[95 + i] - cache[95 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[196 + i] = PA[2] * cache[127 + i] - PG[2] * cache[127 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[199 + i] = PA[1] * cache[135 + i] - PG[1] * cache[135 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[202 + i] = PA[0] * cache[151 + i] - PG[0] * cache[151 + i + 1] + 1 * half_xi_ * (cache[110 + i] - cache[110 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[205 + i] = PA[0] * cache[155 + i] - PG[0] * cache[155 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[208 + i] = PA[0] * cache[159 + i] - PG[0] * cache[159 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[211 + i] = PA[0] * cache[163 + i] - PG[0] * cache[163 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[214 + i] = PA[0] * cache[167 + i] - PG[0] * cache[167 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[217 + i] = PA[0] * cache[171 + i] - PG[0] * cache[171 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[220 + i] = PA[1] * cache[155 + i] - PG[1] * cache[155 + i + 1] + 4 * half_xi_ * (cache[95 + i] - cache[95 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[223 + i] = PA[2] * cache[155 + i] - PG[2] * cache[155 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[226 + i] = PA[1] * cache[163 + i] - PG[1] * cache[163 + i + 1] + 2 * half_xi_ * (cache[105 + i] - cache[105 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[229 + i] = PA[1] * cache[167 + i] - PG[1] * cache[167 + i + 1] + 1 * half_xi_ * (cache[110 + i] - cache[110 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[232 + i] = PA[1] * cache[171 + i] - PG[1] * cache[171 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[235 + i] = PA[2] * cache[171 + i] - PG[2] * cache[171 + i + 1] + 4 * half_xi_ * (cache[110 + i] - cache[110 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[238 + i] = PA[0] * cache[175 + i] - PG[0] * cache[175 + i + 1] + 5 * half_xi_ * (cache[115 + i] - cache[115 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[240 + i] = PA[1] * cache[175 + i] - PG[1] * cache[175 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[242 + i] = PA[2] * cache[175 + i] - PG[2] * cache[175 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[244 + i] = PA[0] * cache[184 + i] - PG[0] * cache[184 + i + 1] + 3 * half_xi_ * (cache[127 + i] - cache[127 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[246 + i] = PA[1] * cache[181 + i] - PG[1] * cache[181 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[248 + i] = PA[0] * cache[190 + i] - PG[0] * cache[190 + i + 1] + 3 * half_xi_ * (cache[135 + i] - cache[135 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[250 + i] = PA[0] * cache[193 + i] - PG[0] * cache[193 + i + 1] + 2 * half_xi_ * (cache[139 + i] - cache[139 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[252 + i] = PA[2] * cache[184 + i] - PG[2] * cache[184 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[254 + i] = PA[1] * cache[190 + i] - PG[1] * cache[190 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[256 + i] = PA[0] * cache[202 + i] - PG[0] * cache[202 + i + 1] + 2 * half_xi_ * (cache[151 + i] - cache[151 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[258 + i] = PA[0] * cache[205 + i] - PG[0] * cache[205 + i + 1] + 1 * half_xi_ * (cache[155 + i] - cache[155 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[260 + i] = PA[2] * cache[193 + i] - PG[2] * cache[193 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[262 + i] = PA[0] * cache[211 + i] - PG[0] * cache[211 + i + 1] + 1 * half_xi_ * (cache[163 + i] - cache[163 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[264 + i] = PA[1] * cache[202 + i] - PG[1] * cache[202 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[266 + i] = PA[0] * cache[217 + i] - PG[0] * cache[217 + i + 1] + 1 * half_xi_ * (cache[171 + i] - cache[171 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[268 + i] = PA[0] * cache[220 + i] - PG[0] * cache[220 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[270 + i] = PA[0] * cache[223 + i] - PG[0] * cache[223 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[272 + i] = PA[0] * cache[226 + i] - PG[0] * cache[226 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[274 + i] = PA[0] * cache[229 + i] - PG[0] * cache[229 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[276 + i] = PA[0] * cache[232 + i] - PG[0] * cache[232 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[278 + i] = PA[0] * cache[235 + i] - PG[0] * cache[235 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[280 + i] = PA[1] * cache[220 + i] - PG[1] * cache[220 + i + 1] + 5 * half_xi_ * (cache[155 + i] - cache[155 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[282 + i] = PA[2] * cache[220 + i] - PG[2] * cache[220 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[284 + i] = PA[1] * cache[226 + i] - PG[1] * cache[226 + i + 1] + 3 * half_xi_ * (cache[163 + i] - cache[163 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[286 + i] = PA[1] * cache[229 + i] - PG[1] * cache[229 + i + 1] + 2 * half_xi_ * (cache[167 + i] - cache[167 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[288 + i] = PA[1] * cache[232 + i] - PG[1] * cache[232 + i + 1] + 1 * half_xi_ * (cache[171 + i] - cache[171 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[290 + i] = PA[1] * cache[235 + i] - PG[1] * cache[235 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[292 + i] = PA[2] * cache[235 + i] - PG[2] * cache[235 + i + 1] + 5 * half_xi_ * (cache[171 + i] - cache[171 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[294 + i] = PA[0] * cache[238 + i] - PG[0] * cache[238 + i + 1] + 6 * half_xi_ * (cache[175 + i] - cache[175 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[295 + i] = PA[1] * cache[238 + i] - PG[1] * cache[238 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[296 + i] = PA[2] * cache[238 + i] - PG[2] * cache[238 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[297 + i] = PA[0] * cache[244 + i] - PG[0] * cache[244 + i + 1] + 4 * half_xi_ * (cache[184 + i] - cache[184 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[298 + i] = PA[1] * cache[242 + i] - PG[1] * cache[242 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[299 + i] = PA[0] * cache[248 + i] - PG[0] * cache[248 + i + 1] + 4 * half_xi_ * (cache[190 + i] - cache[190 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[300 + i] = PA[0] * cache[250 + i] - PG[0] * cache[250 + i + 1] + 3 * half_xi_ * (cache[193 + i] - cache[193 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[301 + i] = PA[2] * cache[244 + i] - PG[2] * cache[244 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[302 + i] = PA[1] * cache[248 + i] - PG[1] * cache[248 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[303 + i] = PA[0] * cache[256 + i] - PG[0] * cache[256 + i + 1] + 3 * half_xi_ * (cache[202 + i] - cache[202 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[304 + i] = PA[0] * cache[258 + i] - PG[0] * cache[258 + i + 1] + 2 * half_xi_ * (cache[205 + i] - cache[205 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[305 + i] = PA[2] * cache[250 + i] - PG[2] * cache[250 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[306 + i] = PA[0] * cache[262 + i] - PG[0] * cache[262 + i + 1] + 2 * half_xi_ * (cache[211 + i] - cache[211 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[307 + i] = PA[1] * cache[256 + i] - PG[1] * cache[256 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[308 + i] = PA[0] * cache[266 + i] - PG[0] * cache[266 + i + 1] + 2 * half_xi_ * (cache[217 + i] - cache[217 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[309 + i] = PA[0] * cache[268 + i] - PG[0] * cache[268 + i + 1] + 1 * half_xi_ * (cache[220 + i] - cache[220 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[310 + i] = PA[2] * cache[258 + i] - PG[2] * cache[258 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[311 + i] = PA[0] * cache[272 + i] - PG[0] * cache[272 + i + 1] + 1 * half_xi_ * (cache[226 + i] - cache[226 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[312 + i] = PA[0] * cache[274 + i] - PG[0] * cache[274 + i + 1] + 1 * half_xi_ * (cache[229 + i] - cache[229 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[313 + i] = PA[1] * cache[266 + i] - PG[1] * cache[266 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[314 + i] = PA[0] * cache[278 + i] - PG[0] * cache[278 + i + 1] + 1 * half_xi_ * (cache[235 + i] - cache[235 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[315 + i] = PA[0] * cache[280 + i] - PG[0] * cache[280 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[316 + i] = PA[0] * cache[282 + i] - PG[0] * cache[282 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[317 + i] = PA[0] * cache[284 + i] - PG[0] * cache[284 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[318 + i] = PA[0] * cache[286 + i] - PG[0] * cache[286 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[319 + i] = PA[0] * cache[288 + i] - PG[0] * cache[288 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[320 + i] = PA[0] * cache[290 + i] - PG[0] * cache[290 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[321 + i] = PA[0] * cache[292 + i] - PG[0] * cache[292 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[322 + i] = PA[1] * cache[280 + i] - PG[1] * cache[280 + i + 1] + 6 * half_xi_ * (cache[220 + i] - cache[220 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[323 + i] = PA[2] * cache[280 + i] - PG[2] * cache[280 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[324 + i] = PA[1] * cache[284 + i] - PG[1] * cache[284 + i + 1] + 4 * half_xi_ * (cache[226 + i] - cache[226 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[325 + i] = PA[1] * cache[286 + i] - PG[1] * cache[286 + i + 1] + 3 * half_xi_ * (cache[229 + i] - cache[229 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[326 + i] = PA[1] * cache[288 + i] - PG[1] * cache[288 + i + 1] + 2 * half_xi_ * (cache[232 + i] - cache[232 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[327 + i] = PA[1] * cache[290 + i] - PG[1] * cache[290 + i + 1] + 1 * half_xi_ * (cache[235 + i] - cache[235 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[328 + i] = PA[1] * cache[292 + i] - PG[1] * cache[292 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[329 + i] = PA[2] * cache[292 + i] - PG[2] * cache[292 + i + 1] + 6 * half_xi_ * (cache[235 + i] - cache[235 + i + 1]);
    buf[0] += cache[115];
    buf[1] += cache[119];
    buf[2] += cache[123];
    buf[3] += cache[127];
    buf[4] += cache[131];
    buf[5] += cache[135];
    buf[6] += cache[139];
    buf[7] += cache[143];
    buf[8] += cache[147];
    buf[9] += cache[151];
    buf[10] += cache[155];
    buf[11] += cache[159];
    buf[12] += cache[163];
    buf[13] += cache[167];
    buf[14] += cache[171];
    buf[15] += cache[175];
    buf[16] += cache[178];
    buf[17] += cache[181];
    buf[18] += cache[184];
    buf[19] += cache[187];
    buf[20] += cache[190];
    buf[21] += cache[193];
    buf[22] += cache[196];
    buf[23] += cache[199];
    buf[24] += cache[202];
    buf[25] += cache[205];
    buf[26] += cache[208];
    buf[27] += cache[211];
    buf[28] += cache[214];
    buf[29] += cache[217];
    buf[30] += cache[220];
    buf[31] += cache[223];
    buf[32] += cache[226];
    buf[33] += cache[229];
    buf[34] += cache[232];
    buf[35] += cache[235];
    buf[36] += cache[238];
    buf[37] += cache[240];
    buf[38] += cache[242];
    buf[39] += cache[244];
    buf[40] += cache[246];
    buf[41] += cache[248];
    buf[42] += cache[250];
    buf[43] += cache[252];
    buf[44] += cache[254];
    buf[45] += cache[256];
    buf[46] += cache[258];
    buf[47] += cache[260];
    buf[48] += cache[262];
    buf[49] += cache[264];
    buf[50] += cache[266];
    buf[51] += cache[268];
    buf[52] += cache[270];
    buf[53] += cache[272];
    buf[54] += cache[274];
    buf[55] += cache[276];
    buf[56] += cache[278];
    buf[57] += cache[280];
    buf[58] += cache[282];
    buf[59] += cache[284];
    buf[60] += cache[286];
    buf[61] += cache[288];
    buf[62] += cache[290];
    buf[63] += cache[292];
    buf[64] += cache[294];
    buf[65] += cache[295];
    buf[66] += cache[296];
    buf[67] += cache[297];
    buf[68] += cache[298];
    buf[69] += cache[299];
    buf[70] += cache[300];
    buf[71] += cache[301];
    buf[72] += cache[302];
    buf[73] += cache[303];
    buf[74] += cache[304];
    buf[75] += cache[305];
    buf[76] += cache[306];
    buf[77] += cache[307];
    buf[78] += cache[308];
    buf[79] += cache[309];
    buf[80] += cache[310];
    buf[81] += cache[311];
    buf[82] += cache[312];
    buf[83] += cache[313];
    buf[84] += cache[314];
    buf[85] += cache[315];
    buf[86] += cache[316];
    buf[87] += cache[317];
    buf[88] += cache[318];
    buf[89] += cache[319];
    buf[90] += cache[320];
    buf[91] += cache[321];
    buf[92] += cache[322];
    buf[93] += cache[323];
    buf[94] += cache[324];
    buf[95] += cache[325];
    buf[96] += cache[326];
    buf[97] += cache[327];
    buf[98] += cache[328];
    buf[99] += cache[329];
    return;
}

static void nuc_VRR_4_4(double *buf, double *cache, double coe, double K_ab, double half_xi_, double xi, const double *PA,
                        const double *PG, double PG_dist2, double sr_frac, double lr_frac, double omega_)
{
    double theta;
    double T_0, T_1;
    double scheme_1[10], scheme_0[10];
    double scheme_0_0_0[9];
    T_0 = xi * PG_dist2;
    if (fabs(omega_) < 1E-8)
    {
        cal_boys(8, T_0, &scheme_0[0]);
        for (int i = 0; i <= 8; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * lr_frac * scheme_0[i];
    }
    else
    {
        theta = sqrt(omega_ * omega_ / (omega_ * omega_ + xi));
        T_1 = xi * PG_dist2 * theta * theta;
        cal_boys(8, T_1, &scheme_1[0]);
        cal_boys(8, T_0, &scheme_0[0]);
        for (int i = 0; i <= 8; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * (lr_frac * scheme_0[i] + sr_frac * theta * pow(theta, 2 * i) * scheme_1[i]);
    }
    for (int i = 0; i < 8; i++)
        cache[9 + i] = PA[0] * cache[0 + i] - PG[0] * cache[0 + i + 1];
    for (int i = 0; i < 8; i++)
        cache[17 + i] = PA[1] * cache[0 + i] - PG[1] * cache[0 + i + 1];
    for (int i = 0; i < 8; i++)
        cache[25 + i] = PA[2] * cache[0 + i] - PG[2] * cache[0 + i + 1];
    for (int i = 0; i < 7; i++)
        cache[33 + i] = PA[0] * cache[9 + i] - PG[0] * cache[9 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 7; i++)
        cache[40 + i] = PA[0] * cache[17 + i] - PG[0] * cache[17 + i + 1];
    for (int i = 0; i < 7; i++)
        cache[47 + i] = PA[0] * cache[25 + i] - PG[0] * cache[25 + i + 1];
    for (int i = 0; i < 7; i++)
        cache[54 + i] = PA[1] * cache[17 + i] - PG[1] * cache[17 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 7; i++)
        cache[61 + i] = PA[1] * cache[25 + i] - PG[1] * cache[25 + i + 1];
    for (int i = 0; i < 7; i++)
        cache[68 + i] = PA[2] * cache[25 + i] - PG[2] * cache[25 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 6; i++)
        cache[75 + i] = PA[0] * cache[33 + i] - PG[0] * cache[33 + i + 1] + 2 * half_xi_ * (cache[9 + i] - cache[9 + i + 1]);
    for (int i = 0; i < 6; i++)
        cache[81 + i] = PA[1] * cache[33 + i] - PG[1] * cache[33 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[87 + i] = PA[2] * cache[33 + i] - PG[2] * cache[33 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[93 + i] = PA[0] * cache[54 + i] - PG[0] * cache[54 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[99 + i] = PA[0] * cache[61 + i] - PG[0] * cache[61 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[105 + i] = PA[0] * cache[68 + i] - PG[0] * cache[68 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[111 + i] = PA[1] * cache[54 + i] - PG[1] * cache[54 + i + 1] + 2 * half_xi_ * (cache[17 + i] - cache[17 + i + 1]);
    for (int i = 0; i < 6; i++)
        cache[117 + i] = PA[2] * cache[54 + i] - PG[2] * cache[54 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[123 + i] = PA[1] * cache[68 + i] - PG[1] * cache[68 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[129 + i] = PA[2] * cache[68 + i] - PG[2] * cache[68 + i + 1] + 2 * half_xi_ * (cache[25 + i] - cache[25 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[135 + i] = PA[0] * cache[75 + i] - PG[0] * cache[75 + i + 1] + 3 * half_xi_ * (cache[33 + i] - cache[33 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[140 + i] = PA[1] * cache[75 + i] - PG[1] * cache[75 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[145 + i] = PA[2] * cache[75 + i] - PG[2] * cache[75 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[150 + i] = PA[0] * cache[93 + i] - PG[0] * cache[93 + i + 1] + 1 * half_xi_ * (cache[54 + i] - cache[54 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[155 + i] = PA[1] * cache[87 + i] - PG[1] * cache[87 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[160 + i] = PA[0] * cache[105 + i] - PG[0] * cache[105 + i + 1] + 1 * half_xi_ * (cache[68 + i] - cache[68 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[165 + i] = PA[0] * cache[111 + i] - PG[0] * cache[111 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[170 + i] = PA[0] * cache[117 + i] - PG[0] * cache[117 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[175 + i] = PA[0] * cache[123 + i] - PG[0] * cache[123 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[180 + i] = PA[0] * cache[129 + i] - PG[0] * cache[129 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[185 + i] = PA[1] * cache[111 + i] - PG[1] * cache[111 + i + 1] + 3 * half_xi_ * (cache[54 + i] - cache[54 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[190 + i] = PA[2] * cache[111 + i] - PG[2] * cache[111 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[195 + i] = PA[1] * cache[123 + i] - PG[1] * cache[123 + i + 1] + 1 * half_xi_ * (cache[68 + i] - cache[68 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[200 + i] = PA[1] * cache[129 + i] - PG[1] * cache[129 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[205 + i] = PA[2] * cache[129 + i] - PG[2] * cache[129 + i + 1] + 3 * half_xi_ * (cache[68 + i] - cache[68 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[210 + i] = PA[0] * cache[135 + i] - PG[0] * cache[135 + i + 1] + 4 * half_xi_ * (cache[75 + i] - cache[75 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[214 + i] = PA[1] * cache[135 + i] - PG[1] * cache[135 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[218 + i] = PA[2] * cache[135 + i] - PG[2] * cache[135 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[222 + i] = PA[0] * cache[150 + i] - PG[0] * cache[150 + i + 1] + 2 * half_xi_ * (cache[93 + i] - cache[93 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[226 + i] = PA[1] * cache[145 + i] - PG[1] * cache[145 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[230 + i] = PA[0] * cache[160 + i] - PG[0] * cache[160 + i + 1] + 2 * half_xi_ * (cache[105 + i] - cache[105 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[234 + i] = PA[0] * cache[165 + i] - PG[0] * cache[165 + i + 1] + 1 * half_xi_ * (cache[111 + i] - cache[111 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[238 + i] = PA[2] * cache[150 + i] - PG[2] * cache[150 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[242 + i] = PA[1] * cache[160 + i] - PG[1] * cache[160 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[246 + i] = PA[0] * cache[180 + i] - PG[0] * cache[180 + i + 1] + 1 * half_xi_ * (cache[129 + i] - cache[129 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[250 + i] = PA[0] * cache[185 + i] - PG[0] * cache[185 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[254 + i] = PA[0] * cache[190 + i] - PG[0] * cache[190 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[258 + i] = PA[0] * cache[195 + i] - PG[0] * cache[195 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[262 + i] = PA[0] * cache[200 + i] - PG[0] * cache[200 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[266 + i] = PA[0] * cache[205 + i] - PG[0] * cache[205 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[270 + i] = PA[1] * cache[185 + i] - PG[1] * cache[185 + i + 1] + 4 * half_xi_ * (cache[111 + i] - cache[111 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[274 + i] = PA[2] * cache[185 + i] - PG[2] * cache[185 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[278 + i] = PA[1] * cache[195 + i] - PG[1] * cache[195 + i + 1] + 2 * half_xi_ * (cache[123 + i] - cache[123 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[282 + i] = PA[1] * cache[200 + i] - PG[1] * cache[200 + i + 1] + 1 * half_xi_ * (cache[129 + i] - cache[129 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[286 + i] = PA[1] * cache[205 + i] - PG[1] * cache[205 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[290 + i] = PA[2] * cache[205 + i] - PG[2] * cache[205 + i + 1] + 4 * half_xi_ * (cache[129 + i] - cache[129 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[294 + i] = PA[0] * cache[210 + i] - PG[0] * cache[210 + i + 1] + 5 * half_xi_ * (cache[135 + i] - cache[135 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[297 + i] = PA[1] * cache[210 + i] - PG[1] * cache[210 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[300 + i] = PA[2] * cache[210 + i] - PG[2] * cache[210 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[303 + i] = PA[0] * cache[222 + i] - PG[0] * cache[222 + i + 1] + 3 * half_xi_ * (cache[150 + i] - cache[150 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[306 + i] = PA[1] * cache[218 + i] - PG[1] * cache[218 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[309 + i] = PA[0] * cache[230 + i] - PG[0] * cache[230 + i + 1] + 3 * half_xi_ * (cache[160 + i] - cache[160 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[312 + i] = PA[0] * cache[234 + i] - PG[0] * cache[234 + i + 1] + 2 * half_xi_ * (cache[165 + i] - cache[165 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[315 + i] = PA[2] * cache[222 + i] - PG[2] * cache[222 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[318 + i] = PA[1] * cache[230 + i] - PG[1] * cache[230 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[321 + i] = PA[0] * cache[246 + i] - PG[0] * cache[246 + i + 1] + 2 * half_xi_ * (cache[180 + i] - cache[180 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[324 + i] = PA[0] * cache[250 + i] - PG[0] * cache[250 + i + 1] + 1 * half_xi_ * (cache[185 + i] - cache[185 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[327 + i] = PA[2] * cache[234 + i] - PG[2] * cache[234 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[330 + i] = PA[0] * cache[258 + i] - PG[0] * cache[258 + i + 1] + 1 * half_xi_ * (cache[195 + i] - cache[195 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[333 + i] = PA[1] * cache[246 + i] - PG[1] * cache[246 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[336 + i] = PA[0] * cache[266 + i] - PG[0] * cache[266 + i + 1] + 1 * half_xi_ * (cache[205 + i] - cache[205 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[339 + i] = PA[0] * cache[270 + i] - PG[0] * cache[270 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[342 + i] = PA[0] * cache[274 + i] - PG[0] * cache[274 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[345 + i] = PA[0] * cache[278 + i] - PG[0] * cache[278 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[348 + i] = PA[0] * cache[282 + i] - PG[0] * cache[282 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[351 + i] = PA[0] * cache[286 + i] - PG[0] * cache[286 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[354 + i] = PA[0] * cache[290 + i] - PG[0] * cache[290 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[357 + i] = PA[1] * cache[270 + i] - PG[1] * cache[270 + i + 1] + 5 * half_xi_ * (cache[185 + i] - cache[185 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[360 + i] = PA[2] * cache[270 + i] - PG[2] * cache[270 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[363 + i] = PA[1] * cache[278 + i] - PG[1] * cache[278 + i + 1] + 3 * half_xi_ * (cache[195 + i] - cache[195 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[366 + i] = PA[1] * cache[282 + i] - PG[1] * cache[282 + i + 1] + 2 * half_xi_ * (cache[200 + i] - cache[200 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[369 + i] = PA[1] * cache[286 + i] - PG[1] * cache[286 + i + 1] + 1 * half_xi_ * (cache[205 + i] - cache[205 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[372 + i] = PA[1] * cache[290 + i] - PG[1] * cache[290 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[375 + i] = PA[2] * cache[290 + i] - PG[2] * cache[290 + i + 1] + 5 * half_xi_ * (cache[205 + i] - cache[205 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[378 + i] = PA[0] * cache[294 + i] - PG[0] * cache[294 + i + 1] + 6 * half_xi_ * (cache[210 + i] - cache[210 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[380 + i] = PA[1] * cache[294 + i] - PG[1] * cache[294 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[382 + i] = PA[2] * cache[294 + i] - PG[2] * cache[294 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[384 + i] = PA[0] * cache[303 + i] - PG[0] * cache[303 + i + 1] + 4 * half_xi_ * (cache[222 + i] - cache[222 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[386 + i] = PA[1] * cache[300 + i] - PG[1] * cache[300 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[388 + i] = PA[0] * cache[309 + i] - PG[0] * cache[309 + i + 1] + 4 * half_xi_ * (cache[230 + i] - cache[230 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[390 + i] = PA[0] * cache[312 + i] - PG[0] * cache[312 + i + 1] + 3 * half_xi_ * (cache[234 + i] - cache[234 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[392 + i] = PA[2] * cache[303 + i] - PG[2] * cache[303 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[394 + i] = PA[1] * cache[309 + i] - PG[1] * cache[309 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[396 + i] = PA[0] * cache[321 + i] - PG[0] * cache[321 + i + 1] + 3 * half_xi_ * (cache[246 + i] - cache[246 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[398 + i] = PA[0] * cache[324 + i] - PG[0] * cache[324 + i + 1] + 2 * half_xi_ * (cache[250 + i] - cache[250 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[400 + i] = PA[2] * cache[312 + i] - PG[2] * cache[312 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[402 + i] = PA[0] * cache[330 + i] - PG[0] * cache[330 + i + 1] + 2 * half_xi_ * (cache[258 + i] - cache[258 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[404 + i] = PA[1] * cache[321 + i] - PG[1] * cache[321 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[406 + i] = PA[0] * cache[336 + i] - PG[0] * cache[336 + i + 1] + 2 * half_xi_ * (cache[266 + i] - cache[266 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[408 + i] = PA[0] * cache[339 + i] - PG[0] * cache[339 + i + 1] + 1 * half_xi_ * (cache[270 + i] - cache[270 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[410 + i] = PA[2] * cache[324 + i] - PG[2] * cache[324 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[412 + i] = PA[0] * cache[345 + i] - PG[0] * cache[345 + i + 1] + 1 * half_xi_ * (cache[278 + i] - cache[278 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[414 + i] = PA[0] * cache[348 + i] - PG[0] * cache[348 + i + 1] + 1 * half_xi_ * (cache[282 + i] - cache[282 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[416 + i] = PA[1] * cache[336 + i] - PG[1] * cache[336 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[418 + i] = PA[0] * cache[354 + i] - PG[0] * cache[354 + i + 1] + 1 * half_xi_ * (cache[290 + i] - cache[290 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[420 + i] = PA[0] * cache[357 + i] - PG[0] * cache[357 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[422 + i] = PA[0] * cache[360 + i] - PG[0] * cache[360 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[424 + i] = PA[0] * cache[363 + i] - PG[0] * cache[363 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[426 + i] = PA[0] * cache[366 + i] - PG[0] * cache[366 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[428 + i] = PA[0] * cache[369 + i] - PG[0] * cache[369 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[430 + i] = PA[0] * cache[372 + i] - PG[0] * cache[372 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[432 + i] = PA[0] * cache[375 + i] - PG[0] * cache[375 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[434 + i] = PA[1] * cache[357 + i] - PG[1] * cache[357 + i + 1] + 6 * half_xi_ * (cache[270 + i] - cache[270 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[436 + i] = PA[2] * cache[357 + i] - PG[2] * cache[357 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[438 + i] = PA[1] * cache[363 + i] - PG[1] * cache[363 + i + 1] + 4 * half_xi_ * (cache[278 + i] - cache[278 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[440 + i] = PA[1] * cache[366 + i] - PG[1] * cache[366 + i + 1] + 3 * half_xi_ * (cache[282 + i] - cache[282 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[442 + i] = PA[1] * cache[369 + i] - PG[1] * cache[369 + i + 1] + 2 * half_xi_ * (cache[286 + i] - cache[286 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[444 + i] = PA[1] * cache[372 + i] - PG[1] * cache[372 + i + 1] + 1 * half_xi_ * (cache[290 + i] - cache[290 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[446 + i] = PA[1] * cache[375 + i] - PG[1] * cache[375 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[448 + i] = PA[2] * cache[375 + i] - PG[2] * cache[375 + i + 1] + 6 * half_xi_ * (cache[290 + i] - cache[290 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[450 + i] = PA[0] * cache[378 + i] - PG[0] * cache[378 + i + 1] + 7 * half_xi_ * (cache[294 + i] - cache[294 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[451 + i] = PA[1] * cache[378 + i] - PG[1] * cache[378 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[452 + i] = PA[2] * cache[378 + i] - PG[2] * cache[378 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[453 + i] = PA[0] * cache[384 + i] - PG[0] * cache[384 + i + 1] + 5 * half_xi_ * (cache[303 + i] - cache[303 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[454 + i] = PA[1] * cache[382 + i] - PG[1] * cache[382 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[455 + i] = PA[0] * cache[388 + i] - PG[0] * cache[388 + i + 1] + 5 * half_xi_ * (cache[309 + i] - cache[309 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[456 + i] = PA[0] * cache[390 + i] - PG[0] * cache[390 + i + 1] + 4 * half_xi_ * (cache[312 + i] - cache[312 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[457 + i] = PA[2] * cache[384 + i] - PG[2] * cache[384 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[458 + i] = PA[1] * cache[388 + i] - PG[1] * cache[388 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[459 + i] = PA[0] * cache[396 + i] - PG[0] * cache[396 + i + 1] + 4 * half_xi_ * (cache[321 + i] - cache[321 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[460 + i] = PA[0] * cache[398 + i] - PG[0] * cache[398 + i + 1] + 3 * half_xi_ * (cache[324 + i] - cache[324 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[461 + i] = PA[2] * cache[390 + i] - PG[2] * cache[390 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[462 + i] = PA[0] * cache[402 + i] - PG[0] * cache[402 + i + 1] + 3 * half_xi_ * (cache[330 + i] - cache[330 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[463 + i] = PA[1] * cache[396 + i] - PG[1] * cache[396 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[464 + i] = PA[0] * cache[406 + i] - PG[0] * cache[406 + i + 1] + 3 * half_xi_ * (cache[336 + i] - cache[336 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[465 + i] = PA[0] * cache[408 + i] - PG[0] * cache[408 + i + 1] + 2 * half_xi_ * (cache[339 + i] - cache[339 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[466 + i] = PA[2] * cache[398 + i] - PG[2] * cache[398 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[467 + i] = PA[0] * cache[412 + i] - PG[0] * cache[412 + i + 1] + 2 * half_xi_ * (cache[345 + i] - cache[345 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[468 + i] = PA[0] * cache[414 + i] - PG[0] * cache[414 + i + 1] + 2 * half_xi_ * (cache[348 + i] - cache[348 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[469 + i] = PA[1] * cache[406 + i] - PG[1] * cache[406 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[470 + i] = PA[0] * cache[418 + i] - PG[0] * cache[418 + i + 1] + 2 * half_xi_ * (cache[354 + i] - cache[354 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[471 + i] = PA[0] * cache[420 + i] - PG[0] * cache[420 + i + 1] + 1 * half_xi_ * (cache[357 + i] - cache[357 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[472 + i] = PA[2] * cache[408 + i] - PG[2] * cache[408 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[473 + i] = PA[0] * cache[424 + i] - PG[0] * cache[424 + i + 1] + 1 * half_xi_ * (cache[363 + i] - cache[363 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[474 + i] = PA[0] * cache[426 + i] - PG[0] * cache[426 + i + 1] + 1 * half_xi_ * (cache[366 + i] - cache[366 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[475 + i] = PA[0] * cache[428 + i] - PG[0] * cache[428 + i + 1] + 1 * half_xi_ * (cache[369 + i] - cache[369 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[476 + i] = PA[1] * cache[418 + i] - PG[1] * cache[418 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[477 + i] = PA[0] * cache[432 + i] - PG[0] * cache[432 + i + 1] + 1 * half_xi_ * (cache[375 + i] - cache[375 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[478 + i] = PA[0] * cache[434 + i] - PG[0] * cache[434 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[479 + i] = PA[0] * cache[436 + i] - PG[0] * cache[436 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[480 + i] = PA[0] * cache[438 + i] - PG[0] * cache[438 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[481 + i] = PA[0] * cache[440 + i] - PG[0] * cache[440 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[482 + i] = PA[0] * cache[442 + i] - PG[0] * cache[442 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[483 + i] = PA[0] * cache[444 + i] - PG[0] * cache[444 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[484 + i] = PA[0] * cache[446 + i] - PG[0] * cache[446 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[485 + i] = PA[0] * cache[448 + i] - PG[0] * cache[448 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[486 + i] = PA[1] * cache[434 + i] - PG[1] * cache[434 + i + 1] + 7 * half_xi_ * (cache[357 + i] - cache[357 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[487 + i] = PA[2] * cache[434 + i] - PG[2] * cache[434 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[488 + i] = PA[1] * cache[438 + i] - PG[1] * cache[438 + i + 1] + 5 * half_xi_ * (cache[363 + i] - cache[363 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[489 + i] = PA[1] * cache[440 + i] - PG[1] * cache[440 + i + 1] + 4 * half_xi_ * (cache[366 + i] - cache[366 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[490 + i] = PA[1] * cache[442 + i] - PG[1] * cache[442 + i + 1] + 3 * half_xi_ * (cache[369 + i] - cache[369 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[491 + i] = PA[1] * cache[444 + i] - PG[1] * cache[444 + i + 1] + 2 * half_xi_ * (cache[372 + i] - cache[372 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[492 + i] = PA[1] * cache[446 + i] - PG[1] * cache[446 + i + 1] + 1 * half_xi_ * (cache[375 + i] - cache[375 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[493 + i] = PA[1] * cache[448 + i] - PG[1] * cache[448 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[494 + i] = PA[2] * cache[448 + i] - PG[2] * cache[448 + i + 1] + 7 * half_xi_ * (cache[375 + i] - cache[375 + i + 1]);
    buf[0] += cache[135];
    buf[1] += cache[140];
    buf[2] += cache[145];
    buf[3] += cache[150];
    buf[4] += cache[155];
    buf[5] += cache[160];
    buf[6] += cache[165];
    buf[7] += cache[170];
    buf[8] += cache[175];
    buf[9] += cache[180];
    buf[10] += cache[185];
    buf[11] += cache[190];
    buf[12] += cache[195];
    buf[13] += cache[200];
    buf[14] += cache[205];
    buf[15] += cache[210];
    buf[16] += cache[214];
    buf[17] += cache[218];
    buf[18] += cache[222];
    buf[19] += cache[226];
    buf[20] += cache[230];
    buf[21] += cache[234];
    buf[22] += cache[238];
    buf[23] += cache[242];
    buf[24] += cache[246];
    buf[25] += cache[250];
    buf[26] += cache[254];
    buf[27] += cache[258];
    buf[28] += cache[262];
    buf[29] += cache[266];
    buf[30] += cache[270];
    buf[31] += cache[274];
    buf[32] += cache[278];
    buf[33] += cache[282];
    buf[34] += cache[286];
    buf[35] += cache[290];
    buf[36] += cache[294];
    buf[37] += cache[297];
    buf[38] += cache[300];
    buf[39] += cache[303];
    buf[40] += cache[306];
    buf[41] += cache[309];
    buf[42] += cache[312];
    buf[43] += cache[315];
    buf[44] += cache[318];
    buf[45] += cache[321];
    buf[46] += cache[324];
    buf[47] += cache[327];
    buf[48] += cache[330];
    buf[49] += cache[333];
    buf[50] += cache[336];
    buf[51] += cache[339];
    buf[52] += cache[342];
    buf[53] += cache[345];
    buf[54] += cache[348];
    buf[55] += cache[351];
    buf[56] += cache[354];
    buf[57] += cache[357];
    buf[58] += cache[360];
    buf[59] += cache[363];
    buf[60] += cache[366];
    buf[61] += cache[369];
    buf[62] += cache[372];
    buf[63] += cache[375];
    buf[64] += cache[378];
    buf[65] += cache[380];
    buf[66] += cache[382];
    buf[67] += cache[384];
    buf[68] += cache[386];
    buf[69] += cache[388];
    buf[70] += cache[390];
    buf[71] += cache[392];
    buf[72] += cache[394];
    buf[73] += cache[396];
    buf[74] += cache[398];
    buf[75] += cache[400];
    buf[76] += cache[402];
    buf[77] += cache[404];
    buf[78] += cache[406];
    buf[79] += cache[408];
    buf[80] += cache[410];
    buf[81] += cache[412];
    buf[82] += cache[414];
    buf[83] += cache[416];
    buf[84] += cache[418];
    buf[85] += cache[420];
    buf[86] += cache[422];
    buf[87] += cache[424];
    buf[88] += cache[426];
    buf[89] += cache[428];
    buf[90] += cache[430];
    buf[91] += cache[432];
    buf[92] += cache[434];
    buf[93] += cache[436];
    buf[94] += cache[438];
    buf[95] += cache[440];
    buf[96] += cache[442];
    buf[97] += cache[444];
    buf[98] += cache[446];
    buf[99] += cache[448];
    buf[100] += cache[450];
    buf[101] += cache[451];
    buf[102] += cache[452];
    buf[103] += cache[453];
    buf[104] += cache[454];
    buf[105] += cache[455];
    buf[106] += cache[456];
    buf[107] += cache[457];
    buf[108] += cache[458];
    buf[109] += cache[459];
    buf[110] += cache[460];
    buf[111] += cache[461];
    buf[112] += cache[462];
    buf[113] += cache[463];
    buf[114] += cache[464];
    buf[115] += cache[465];
    buf[116] += cache[466];
    buf[117] += cache[467];
    buf[118] += cache[468];
    buf[119] += cache[469];
    buf[120] += cache[470];
    buf[121] += cache[471];
    buf[122] += cache[472];
    buf[123] += cache[473];
    buf[124] += cache[474];
    buf[125] += cache[475];
    buf[126] += cache[476];
    buf[127] += cache[477];
    buf[128] += cache[478];
    buf[129] += cache[479];
    buf[130] += cache[480];
    buf[131] += cache[481];
    buf[132] += cache[482];
    buf[133] += cache[483];
    buf[134] += cache[484];
    buf[135] += cache[485];
    buf[136] += cache[486];
    buf[137] += cache[487];
    buf[138] += cache[488];
    buf[139] += cache[489];
    buf[140] += cache[490];
    buf[141] += cache[491];
    buf[142] += cache[492];
    buf[143] += cache[493];
    buf[144] += cache[494];
    return;
}

static void nuc_VRR_5_0(double *buf, double *cache, double coe, double K_ab, double half_xi_, double xi, const double *PA,
                        const double *PG, double PG_dist2, double sr_frac, double lr_frac, double omega_)
{
    double theta;
    double T_0, T_1;
    double scheme_1[10], scheme_0[10];
    double scheme_0_0_0[6];
    T_0 = xi * PG_dist2;
    if (fabs(omega_) < 1E-8)
    {
        cal_boys(5, T_0, &scheme_0[0]);
        for (int i = 0; i <= 5; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * lr_frac * scheme_0[i];
    }
    else
    {
        theta = sqrt(omega_ * omega_ / (omega_ * omega_ + xi));
        T_1 = xi * PG_dist2 * theta * theta;
        cal_boys(5, T_1, &scheme_1[0]);
        cal_boys(5, T_0, &scheme_0[0]);
        for (int i = 0; i <= 5; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * (lr_frac * scheme_0[i] + sr_frac * theta * pow(theta, 2 * i) * scheme_1[i]);
    }
    for (int i = 0; i < 5; i++)
        cache[6 + i] = PA[0] * cache[0 + i] - PG[0] * cache[0 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[11 + i] = PA[1] * cache[0 + i] - PG[1] * cache[0 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[16 + i] = PA[2] * cache[0 + i] - PG[2] * cache[0 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[21 + i] = PA[0] * cache[6 + i] - PG[0] * cache[6 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[25 + i] = PA[0] * cache[11 + i] - PG[0] * cache[11 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[29 + i] = PA[0] * cache[16 + i] - PG[0] * cache[16 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[33 + i] = PA[1] * cache[11 + i] - PG[1] * cache[11 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[37 + i] = PA[1] * cache[16 + i] - PG[1] * cache[16 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[41 + i] = PA[2] * cache[16 + i] - PG[2] * cache[16 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[45 + i] = PA[0] * cache[21 + i] - PG[0] * cache[21 + i + 1] + 2 * half_xi_ * (cache[6 + i] - cache[6 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[48 + i] = PA[1] * cache[21 + i] - PG[1] * cache[21 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[51 + i] = PA[2] * cache[21 + i] - PG[2] * cache[21 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[54 + i] = PA[0] * cache[33 + i] - PG[0] * cache[33 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[57 + i] = PA[0] * cache[37 + i] - PG[0] * cache[37 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[60 + i] = PA[0] * cache[41 + i] - PG[0] * cache[41 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[63 + i] = PA[1] * cache[33 + i] - PG[1] * cache[33 + i + 1] + 2 * half_xi_ * (cache[11 + i] - cache[11 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[66 + i] = PA[2] * cache[33 + i] - PG[2] * cache[33 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[69 + i] = PA[1] * cache[41 + i] - PG[1] * cache[41 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[72 + i] = PA[2] * cache[41 + i] - PG[2] * cache[41 + i + 1] + 2 * half_xi_ * (cache[16 + i] - cache[16 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[75 + i] = PA[0] * cache[45 + i] - PG[0] * cache[45 + i + 1] + 3 * half_xi_ * (cache[21 + i] - cache[21 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[77 + i] = PA[1] * cache[45 + i] - PG[1] * cache[45 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[79 + i] = PA[2] * cache[45 + i] - PG[2] * cache[45 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[81 + i] = PA[0] * cache[54 + i] - PG[0] * cache[54 + i + 1] + 1 * half_xi_ * (cache[33 + i] - cache[33 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[83 + i] = PA[1] * cache[51 + i] - PG[1] * cache[51 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[85 + i] = PA[0] * cache[60 + i] - PG[0] * cache[60 + i + 1] + 1 * half_xi_ * (cache[41 + i] - cache[41 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[87 + i] = PA[0] * cache[63 + i] - PG[0] * cache[63 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[89 + i] = PA[0] * cache[66 + i] - PG[0] * cache[66 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[91 + i] = PA[0] * cache[69 + i] - PG[0] * cache[69 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[93 + i] = PA[0] * cache[72 + i] - PG[0] * cache[72 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[95 + i] = PA[1] * cache[63 + i] - PG[1] * cache[63 + i + 1] + 3 * half_xi_ * (cache[33 + i] - cache[33 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[97 + i] = PA[2] * cache[63 + i] - PG[2] * cache[63 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[99 + i] = PA[1] * cache[69 + i] - PG[1] * cache[69 + i + 1] + 1 * half_xi_ * (cache[41 + i] - cache[41 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[101 + i] = PA[1] * cache[72 + i] - PG[1] * cache[72 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[103 + i] = PA[2] * cache[72 + i] - PG[2] * cache[72 + i + 1] + 3 * half_xi_ * (cache[41 + i] - cache[41 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[105 + i] = PA[0] * cache[75 + i] - PG[0] * cache[75 + i + 1] + 4 * half_xi_ * (cache[45 + i] - cache[45 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[106 + i] = PA[1] * cache[75 + i] - PG[1] * cache[75 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[107 + i] = PA[2] * cache[75 + i] - PG[2] * cache[75 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[108 + i] = PA[0] * cache[81 + i] - PG[0] * cache[81 + i + 1] + 2 * half_xi_ * (cache[54 + i] - cache[54 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[109 + i] = PA[1] * cache[79 + i] - PG[1] * cache[79 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[110 + i] = PA[0] * cache[85 + i] - PG[0] * cache[85 + i + 1] + 2 * half_xi_ * (cache[60 + i] - cache[60 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[111 + i] = PA[0] * cache[87 + i] - PG[0] * cache[87 + i + 1] + 1 * half_xi_ * (cache[63 + i] - cache[63 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[112 + i] = PA[2] * cache[81 + i] - PG[2] * cache[81 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[113 + i] = PA[1] * cache[85 + i] - PG[1] * cache[85 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[114 + i] = PA[0] * cache[93 + i] - PG[0] * cache[93 + i + 1] + 1 * half_xi_ * (cache[72 + i] - cache[72 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[115 + i] = PA[0] * cache[95 + i] - PG[0] * cache[95 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[116 + i] = PA[0] * cache[97 + i] - PG[0] * cache[97 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[117 + i] = PA[0] * cache[99 + i] - PG[0] * cache[99 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[118 + i] = PA[0] * cache[101 + i] - PG[0] * cache[101 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[119 + i] = PA[0] * cache[103 + i] - PG[0] * cache[103 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[120 + i] = PA[1] * cache[95 + i] - PG[1] * cache[95 + i + 1] + 4 * half_xi_ * (cache[63 + i] - cache[63 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[121 + i] = PA[2] * cache[95 + i] - PG[2] * cache[95 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[122 + i] = PA[1] * cache[99 + i] - PG[1] * cache[99 + i + 1] + 2 * half_xi_ * (cache[69 + i] - cache[69 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[123 + i] = PA[1] * cache[101 + i] - PG[1] * cache[101 + i + 1] + 1 * half_xi_ * (cache[72 + i] - cache[72 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[124 + i] = PA[1] * cache[103 + i] - PG[1] * cache[103 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[125 + i] = PA[2] * cache[103 + i] - PG[2] * cache[103 + i + 1] + 4 * half_xi_ * (cache[72 + i] - cache[72 + i + 1]);
    buf[0] += cache[105];
    buf[1] += cache[106];
    buf[2] += cache[107];
    buf[3] += cache[108];
    buf[4] += cache[109];
    buf[5] += cache[110];
    buf[6] += cache[111];
    buf[7] += cache[112];
    buf[8] += cache[113];
    buf[9] += cache[114];
    buf[10] += cache[115];
    buf[11] += cache[116];
    buf[12] += cache[117];
    buf[13] += cache[118];
    buf[14] += cache[119];
    buf[15] += cache[120];
    buf[16] += cache[121];
    buf[17] += cache[122];
    buf[18] += cache[123];
    buf[19] += cache[124];
    buf[20] += cache[125];
    return;
}

static void nuc_VRR_5_1(double *buf, double *cache, double coe, double K_ab, double half_xi_, double xi, const double *PA,
                        const double *PG, double PG_dist2, double sr_frac, double lr_frac, double omega_)
{
    double theta;
    double T_0, T_1;
    double scheme_1[10], scheme_0[10];
    double scheme_0_0_0[7];
    T_0 = xi * PG_dist2;
    if (fabs(omega_) < 1E-8)
    {
        cal_boys(6, T_0, &scheme_0[0]);
        for (int i = 0; i <= 6; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * lr_frac * scheme_0[i];
    }
    else
    {
        theta = sqrt(omega_ * omega_ / (omega_ * omega_ + xi));
        T_1 = xi * PG_dist2 * theta * theta;
        cal_boys(6, T_1, &scheme_1[0]);
        cal_boys(6, T_0, &scheme_0[0]);
        for (int i = 0; i <= 6; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * (lr_frac * scheme_0[i] + sr_frac * theta * pow(theta, 2 * i) * scheme_1[i]);
    }
    for (int i = 0; i < 6; i++)
        cache[7 + i] = PA[0] * cache[0 + i] - PG[0] * cache[0 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[13 + i] = PA[1] * cache[0 + i] - PG[1] * cache[0 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[19 + i] = PA[2] * cache[0 + i] - PG[2] * cache[0 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[25 + i] = PA[0] * cache[7 + i] - PG[0] * cache[7 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[30 + i] = PA[0] * cache[13 + i] - PG[0] * cache[13 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[35 + i] = PA[0] * cache[19 + i] - PG[0] * cache[19 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[40 + i] = PA[1] * cache[13 + i] - PG[1] * cache[13 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[45 + i] = PA[1] * cache[19 + i] - PG[1] * cache[19 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[50 + i] = PA[2] * cache[19 + i] - PG[2] * cache[19 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[55 + i] = PA[0] * cache[25 + i] - PG[0] * cache[25 + i + 1] + 2 * half_xi_ * (cache[7 + i] - cache[7 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[59 + i] = PA[1] * cache[25 + i] - PG[1] * cache[25 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[63 + i] = PA[2] * cache[25 + i] - PG[2] * cache[25 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[67 + i] = PA[0] * cache[40 + i] - PG[0] * cache[40 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[71 + i] = PA[0] * cache[45 + i] - PG[0] * cache[45 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[75 + i] = PA[0] * cache[50 + i] - PG[0] * cache[50 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[79 + i] = PA[1] * cache[40 + i] - PG[1] * cache[40 + i + 1] + 2 * half_xi_ * (cache[13 + i] - cache[13 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[83 + i] = PA[2] * cache[40 + i] - PG[2] * cache[40 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[87 + i] = PA[1] * cache[50 + i] - PG[1] * cache[50 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[91 + i] = PA[2] * cache[50 + i] - PG[2] * cache[50 + i + 1] + 2 * half_xi_ * (cache[19 + i] - cache[19 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[95 + i] = PA[0] * cache[55 + i] - PG[0] * cache[55 + i + 1] + 3 * half_xi_ * (cache[25 + i] - cache[25 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[98 + i] = PA[1] * cache[55 + i] - PG[1] * cache[55 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[101 + i] = PA[2] * cache[55 + i] - PG[2] * cache[55 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[104 + i] = PA[0] * cache[67 + i] - PG[0] * cache[67 + i + 1] + 1 * half_xi_ * (cache[40 + i] - cache[40 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[107 + i] = PA[1] * cache[63 + i] - PG[1] * cache[63 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[110 + i] = PA[0] * cache[75 + i] - PG[0] * cache[75 + i + 1] + 1 * half_xi_ * (cache[50 + i] - cache[50 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[113 + i] = PA[0] * cache[79 + i] - PG[0] * cache[79 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[116 + i] = PA[0] * cache[83 + i] - PG[0] * cache[83 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[119 + i] = PA[0] * cache[87 + i] - PG[0] * cache[87 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[122 + i] = PA[0] * cache[91 + i] - PG[0] * cache[91 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[125 + i] = PA[1] * cache[79 + i] - PG[1] * cache[79 + i + 1] + 3 * half_xi_ * (cache[40 + i] - cache[40 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[128 + i] = PA[2] * cache[79 + i] - PG[2] * cache[79 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[131 + i] = PA[1] * cache[87 + i] - PG[1] * cache[87 + i + 1] + 1 * half_xi_ * (cache[50 + i] - cache[50 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[134 + i] = PA[1] * cache[91 + i] - PG[1] * cache[91 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[137 + i] = PA[2] * cache[91 + i] - PG[2] * cache[91 + i + 1] + 3 * half_xi_ * (cache[50 + i] - cache[50 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[140 + i] = PA[0] * cache[95 + i] - PG[0] * cache[95 + i + 1] + 4 * half_xi_ * (cache[55 + i] - cache[55 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[142 + i] = PA[1] * cache[95 + i] - PG[1] * cache[95 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[144 + i] = PA[2] * cache[95 + i] - PG[2] * cache[95 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[146 + i] = PA[0] * cache[104 + i] - PG[0] * cache[104 + i + 1] + 2 * half_xi_ * (cache[67 + i] - cache[67 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[148 + i] = PA[1] * cache[101 + i] - PG[1] * cache[101 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[150 + i] = PA[0] * cache[110 + i] - PG[0] * cache[110 + i + 1] + 2 * half_xi_ * (cache[75 + i] - cache[75 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[152 + i] = PA[0] * cache[113 + i] - PG[0] * cache[113 + i + 1] + 1 * half_xi_ * (cache[79 + i] - cache[79 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[154 + i] = PA[2] * cache[104 + i] - PG[2] * cache[104 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[156 + i] = PA[1] * cache[110 + i] - PG[1] * cache[110 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[158 + i] = PA[0] * cache[122 + i] - PG[0] * cache[122 + i + 1] + 1 * half_xi_ * (cache[91 + i] - cache[91 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[160 + i] = PA[0] * cache[125 + i] - PG[0] * cache[125 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[162 + i] = PA[0] * cache[128 + i] - PG[0] * cache[128 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[164 + i] = PA[0] * cache[131 + i] - PG[0] * cache[131 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[166 + i] = PA[0] * cache[134 + i] - PG[0] * cache[134 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[168 + i] = PA[0] * cache[137 + i] - PG[0] * cache[137 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[170 + i] = PA[1] * cache[125 + i] - PG[1] * cache[125 + i + 1] + 4 * half_xi_ * (cache[79 + i] - cache[79 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[172 + i] = PA[2] * cache[125 + i] - PG[2] * cache[125 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[174 + i] = PA[1] * cache[131 + i] - PG[1] * cache[131 + i + 1] + 2 * half_xi_ * (cache[87 + i] - cache[87 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[176 + i] = PA[1] * cache[134 + i] - PG[1] * cache[134 + i + 1] + 1 * half_xi_ * (cache[91 + i] - cache[91 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[178 + i] = PA[1] * cache[137 + i] - PG[1] * cache[137 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[180 + i] = PA[2] * cache[137 + i] - PG[2] * cache[137 + i + 1] + 4 * half_xi_ * (cache[91 + i] - cache[91 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[182 + i] = PA[0] * cache[140 + i] - PG[0] * cache[140 + i + 1] + 5 * half_xi_ * (cache[95 + i] - cache[95 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[183 + i] = PA[1] * cache[140 + i] - PG[1] * cache[140 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[184 + i] = PA[2] * cache[140 + i] - PG[2] * cache[140 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[185 + i] = PA[0] * cache[146 + i] - PG[0] * cache[146 + i + 1] + 3 * half_xi_ * (cache[104 + i] - cache[104 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[186 + i] = PA[1] * cache[144 + i] - PG[1] * cache[144 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[187 + i] = PA[0] * cache[150 + i] - PG[0] * cache[150 + i + 1] + 3 * half_xi_ * (cache[110 + i] - cache[110 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[188 + i] = PA[0] * cache[152 + i] - PG[0] * cache[152 + i + 1] + 2 * half_xi_ * (cache[113 + i] - cache[113 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[189 + i] = PA[2] * cache[146 + i] - PG[2] * cache[146 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[190 + i] = PA[1] * cache[150 + i] - PG[1] * cache[150 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[191 + i] = PA[0] * cache[158 + i] - PG[0] * cache[158 + i + 1] + 2 * half_xi_ * (cache[122 + i] - cache[122 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[192 + i] = PA[0] * cache[160 + i] - PG[0] * cache[160 + i + 1] + 1 * half_xi_ * (cache[125 + i] - cache[125 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[193 + i] = PA[2] * cache[152 + i] - PG[2] * cache[152 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[194 + i] = PA[0] * cache[164 + i] - PG[0] * cache[164 + i + 1] + 1 * half_xi_ * (cache[131 + i] - cache[131 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[195 + i] = PA[1] * cache[158 + i] - PG[1] * cache[158 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[196 + i] = PA[0] * cache[168 + i] - PG[0] * cache[168 + i + 1] + 1 * half_xi_ * (cache[137 + i] - cache[137 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[197 + i] = PA[0] * cache[170 + i] - PG[0] * cache[170 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[198 + i] = PA[0] * cache[172 + i] - PG[0] * cache[172 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[199 + i] = PA[0] * cache[174 + i] - PG[0] * cache[174 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[200 + i] = PA[0] * cache[176 + i] - PG[0] * cache[176 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[201 + i] = PA[0] * cache[178 + i] - PG[0] * cache[178 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[202 + i] = PA[0] * cache[180 + i] - PG[0] * cache[180 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[203 + i] = PA[1] * cache[170 + i] - PG[1] * cache[170 + i + 1] + 5 * half_xi_ * (cache[125 + i] - cache[125 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[204 + i] = PA[2] * cache[170 + i] - PG[2] * cache[170 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[205 + i] = PA[1] * cache[174 + i] - PG[1] * cache[174 + i + 1] + 3 * half_xi_ * (cache[131 + i] - cache[131 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[206 + i] = PA[1] * cache[176 + i] - PG[1] * cache[176 + i + 1] + 2 * half_xi_ * (cache[134 + i] - cache[134 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[207 + i] = PA[1] * cache[178 + i] - PG[1] * cache[178 + i + 1] + 1 * half_xi_ * (cache[137 + i] - cache[137 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[208 + i] = PA[1] * cache[180 + i] - PG[1] * cache[180 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[209 + i] = PA[2] * cache[180 + i] - PG[2] * cache[180 + i + 1] + 5 * half_xi_ * (cache[137 + i] - cache[137 + i + 1]);
    buf[0] += cache[140];
    buf[1] += cache[142];
    buf[2] += cache[144];
    buf[3] += cache[146];
    buf[4] += cache[148];
    buf[5] += cache[150];
    buf[6] += cache[152];
    buf[7] += cache[154];
    buf[8] += cache[156];
    buf[9] += cache[158];
    buf[10] += cache[160];
    buf[11] += cache[162];
    buf[12] += cache[164];
    buf[13] += cache[166];
    buf[14] += cache[168];
    buf[15] += cache[170];
    buf[16] += cache[172];
    buf[17] += cache[174];
    buf[18] += cache[176];
    buf[19] += cache[178];
    buf[20] += cache[180];
    buf[21] += cache[182];
    buf[22] += cache[183];
    buf[23] += cache[184];
    buf[24] += cache[185];
    buf[25] += cache[186];
    buf[26] += cache[187];
    buf[27] += cache[188];
    buf[28] += cache[189];
    buf[29] += cache[190];
    buf[30] += cache[191];
    buf[31] += cache[192];
    buf[32] += cache[193];
    buf[33] += cache[194];
    buf[34] += cache[195];
    buf[35] += cache[196];
    buf[36] += cache[197];
    buf[37] += cache[198];
    buf[38] += cache[199];
    buf[39] += cache[200];
    buf[40] += cache[201];
    buf[41] += cache[202];
    buf[42] += cache[203];
    buf[43] += cache[204];
    buf[44] += cache[205];
    buf[45] += cache[206];
    buf[46] += cache[207];
    buf[47] += cache[208];
    buf[48] += cache[209];
    return;
}

static void nuc_VRR_5_2(double *buf, double *cache, double coe, double K_ab, double half_xi_, double xi, const double *PA,
                        const double *PG, double PG_dist2, double sr_frac, double lr_frac, double omega_)
{
    double theta;
    double T_0, T_1;
    double scheme_1[10], scheme_0[10];
    double scheme_0_0_0[8];
    T_0 = xi * PG_dist2;
    if (fabs(omega_) < 1E-8)
    {
        cal_boys(7, T_0, &scheme_0[0]);
        cal_boys(7, T_0, &scheme_0[0]);
        for (int i = 0; i <= 7; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * lr_frac * scheme_0[i];
    }
    else
    {
        theta = sqrt(omega_ * omega_ / (omega_ * omega_ + xi));
        T_1 = xi * PG_dist2 * theta * theta;
        cal_boys(7, T_1, &scheme_1[0]);
        for (int i = 0; i <= 7; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * (lr_frac * scheme_0[i] + sr_frac * theta * pow(theta, 2 * i) * scheme_1[i]);
    }
    for (int i = 0; i < 7; i++)
        cache[8 + i] = PA[0] * cache[0 + i] - PG[0] * cache[0 + i + 1];
    for (int i = 0; i < 7; i++)
        cache[15 + i] = PA[1] * cache[0 + i] - PG[1] * cache[0 + i + 1];
    for (int i = 0; i < 7; i++)
        cache[22 + i] = PA[2] * cache[0 + i] - PG[2] * cache[0 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[29 + i] = PA[0] * cache[8 + i] - PG[0] * cache[8 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 6; i++)
        cache[35 + i] = PA[0] * cache[15 + i] - PG[0] * cache[15 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[41 + i] = PA[0] * cache[22 + i] - PG[0] * cache[22 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[47 + i] = PA[1] * cache[15 + i] - PG[1] * cache[15 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 6; i++)
        cache[53 + i] = PA[1] * cache[22 + i] - PG[1] * cache[22 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[59 + i] = PA[2] * cache[22 + i] - PG[2] * cache[22 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[65 + i] = PA[0] * cache[29 + i] - PG[0] * cache[29 + i + 1] + 2 * half_xi_ * (cache[8 + i] - cache[8 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[70 + i] = PA[1] * cache[29 + i] - PG[1] * cache[29 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[75 + i] = PA[2] * cache[29 + i] - PG[2] * cache[29 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[80 + i] = PA[0] * cache[47 + i] - PG[0] * cache[47 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[85 + i] = PA[0] * cache[53 + i] - PG[0] * cache[53 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[90 + i] = PA[0] * cache[59 + i] - PG[0] * cache[59 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[95 + i] = PA[1] * cache[47 + i] - PG[1] * cache[47 + i + 1] + 2 * half_xi_ * (cache[15 + i] - cache[15 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[100 + i] = PA[2] * cache[47 + i] - PG[2] * cache[47 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[105 + i] = PA[1] * cache[59 + i] - PG[1] * cache[59 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[110 + i] = PA[2] * cache[59 + i] - PG[2] * cache[59 + i + 1] + 2 * half_xi_ * (cache[22 + i] - cache[22 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[115 + i] = PA[0] * cache[65 + i] - PG[0] * cache[65 + i + 1] + 3 * half_xi_ * (cache[29 + i] - cache[29 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[119 + i] = PA[1] * cache[65 + i] - PG[1] * cache[65 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[123 + i] = PA[2] * cache[65 + i] - PG[2] * cache[65 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[127 + i] = PA[0] * cache[80 + i] - PG[0] * cache[80 + i + 1] + 1 * half_xi_ * (cache[47 + i] - cache[47 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[131 + i] = PA[1] * cache[75 + i] - PG[1] * cache[75 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[135 + i] = PA[0] * cache[90 + i] - PG[0] * cache[90 + i + 1] + 1 * half_xi_ * (cache[59 + i] - cache[59 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[139 + i] = PA[0] * cache[95 + i] - PG[0] * cache[95 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[143 + i] = PA[0] * cache[100 + i] - PG[0] * cache[100 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[147 + i] = PA[0] * cache[105 + i] - PG[0] * cache[105 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[151 + i] = PA[0] * cache[110 + i] - PG[0] * cache[110 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[155 + i] = PA[1] * cache[95 + i] - PG[1] * cache[95 + i + 1] + 3 * half_xi_ * (cache[47 + i] - cache[47 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[159 + i] = PA[2] * cache[95 + i] - PG[2] * cache[95 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[163 + i] = PA[1] * cache[105 + i] - PG[1] * cache[105 + i + 1] + 1 * half_xi_ * (cache[59 + i] - cache[59 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[167 + i] = PA[1] * cache[110 + i] - PG[1] * cache[110 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[171 + i] = PA[2] * cache[110 + i] - PG[2] * cache[110 + i + 1] + 3 * half_xi_ * (cache[59 + i] - cache[59 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[175 + i] = PA[0] * cache[115 + i] - PG[0] * cache[115 + i + 1] + 4 * half_xi_ * (cache[65 + i] - cache[65 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[178 + i] = PA[1] * cache[115 + i] - PG[1] * cache[115 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[181 + i] = PA[2] * cache[115 + i] - PG[2] * cache[115 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[184 + i] = PA[0] * cache[127 + i] - PG[0] * cache[127 + i + 1] + 2 * half_xi_ * (cache[80 + i] - cache[80 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[187 + i] = PA[1] * cache[123 + i] - PG[1] * cache[123 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[190 + i] = PA[0] * cache[135 + i] - PG[0] * cache[135 + i + 1] + 2 * half_xi_ * (cache[90 + i] - cache[90 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[193 + i] = PA[0] * cache[139 + i] - PG[0] * cache[139 + i + 1] + 1 * half_xi_ * (cache[95 + i] - cache[95 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[196 + i] = PA[2] * cache[127 + i] - PG[2] * cache[127 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[199 + i] = PA[1] * cache[135 + i] - PG[1] * cache[135 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[202 + i] = PA[0] * cache[151 + i] - PG[0] * cache[151 + i + 1] + 1 * half_xi_ * (cache[110 + i] - cache[110 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[205 + i] = PA[0] * cache[155 + i] - PG[0] * cache[155 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[208 + i] = PA[0] * cache[159 + i] - PG[0] * cache[159 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[211 + i] = PA[0] * cache[163 + i] - PG[0] * cache[163 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[214 + i] = PA[0] * cache[167 + i] - PG[0] * cache[167 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[217 + i] = PA[0] * cache[171 + i] - PG[0] * cache[171 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[220 + i] = PA[1] * cache[155 + i] - PG[1] * cache[155 + i + 1] + 4 * half_xi_ * (cache[95 + i] - cache[95 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[223 + i] = PA[2] * cache[155 + i] - PG[2] * cache[155 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[226 + i] = PA[1] * cache[163 + i] - PG[1] * cache[163 + i + 1] + 2 * half_xi_ * (cache[105 + i] - cache[105 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[229 + i] = PA[1] * cache[167 + i] - PG[1] * cache[167 + i + 1] + 1 * half_xi_ * (cache[110 + i] - cache[110 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[232 + i] = PA[1] * cache[171 + i] - PG[1] * cache[171 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[235 + i] = PA[2] * cache[171 + i] - PG[2] * cache[171 + i + 1] + 4 * half_xi_ * (cache[110 + i] - cache[110 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[238 + i] = PA[0] * cache[175 + i] - PG[0] * cache[175 + i + 1] + 5 * half_xi_ * (cache[115 + i] - cache[115 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[240 + i] = PA[1] * cache[175 + i] - PG[1] * cache[175 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[242 + i] = PA[2] * cache[175 + i] - PG[2] * cache[175 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[244 + i] = PA[0] * cache[184 + i] - PG[0] * cache[184 + i + 1] + 3 * half_xi_ * (cache[127 + i] - cache[127 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[246 + i] = PA[1] * cache[181 + i] - PG[1] * cache[181 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[248 + i] = PA[0] * cache[190 + i] - PG[0] * cache[190 + i + 1] + 3 * half_xi_ * (cache[135 + i] - cache[135 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[250 + i] = PA[0] * cache[193 + i] - PG[0] * cache[193 + i + 1] + 2 * half_xi_ * (cache[139 + i] - cache[139 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[252 + i] = PA[2] * cache[184 + i] - PG[2] * cache[184 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[254 + i] = PA[1] * cache[190 + i] - PG[1] * cache[190 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[256 + i] = PA[0] * cache[202 + i] - PG[0] * cache[202 + i + 1] + 2 * half_xi_ * (cache[151 + i] - cache[151 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[258 + i] = PA[0] * cache[205 + i] - PG[0] * cache[205 + i + 1] + 1 * half_xi_ * (cache[155 + i] - cache[155 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[260 + i] = PA[2] * cache[193 + i] - PG[2] * cache[193 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[262 + i] = PA[0] * cache[211 + i] - PG[0] * cache[211 + i + 1] + 1 * half_xi_ * (cache[163 + i] - cache[163 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[264 + i] = PA[1] * cache[202 + i] - PG[1] * cache[202 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[266 + i] = PA[0] * cache[217 + i] - PG[0] * cache[217 + i + 1] + 1 * half_xi_ * (cache[171 + i] - cache[171 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[268 + i] = PA[0] * cache[220 + i] - PG[0] * cache[220 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[270 + i] = PA[0] * cache[223 + i] - PG[0] * cache[223 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[272 + i] = PA[0] * cache[226 + i] - PG[0] * cache[226 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[274 + i] = PA[0] * cache[229 + i] - PG[0] * cache[229 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[276 + i] = PA[0] * cache[232 + i] - PG[0] * cache[232 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[278 + i] = PA[0] * cache[235 + i] - PG[0] * cache[235 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[280 + i] = PA[1] * cache[220 + i] - PG[1] * cache[220 + i + 1] + 5 * half_xi_ * (cache[155 + i] - cache[155 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[282 + i] = PA[2] * cache[220 + i] - PG[2] * cache[220 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[284 + i] = PA[1] * cache[226 + i] - PG[1] * cache[226 + i + 1] + 3 * half_xi_ * (cache[163 + i] - cache[163 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[286 + i] = PA[1] * cache[229 + i] - PG[1] * cache[229 + i + 1] + 2 * half_xi_ * (cache[167 + i] - cache[167 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[288 + i] = PA[1] * cache[232 + i] - PG[1] * cache[232 + i + 1] + 1 * half_xi_ * (cache[171 + i] - cache[171 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[290 + i] = PA[1] * cache[235 + i] - PG[1] * cache[235 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[292 + i] = PA[2] * cache[235 + i] - PG[2] * cache[235 + i + 1] + 5 * half_xi_ * (cache[171 + i] - cache[171 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[294 + i] = PA[0] * cache[238 + i] - PG[0] * cache[238 + i + 1] + 6 * half_xi_ * (cache[175 + i] - cache[175 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[295 + i] = PA[1] * cache[238 + i] - PG[1] * cache[238 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[296 + i] = PA[2] * cache[238 + i] - PG[2] * cache[238 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[297 + i] = PA[0] * cache[244 + i] - PG[0] * cache[244 + i + 1] + 4 * half_xi_ * (cache[184 + i] - cache[184 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[298 + i] = PA[1] * cache[242 + i] - PG[1] * cache[242 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[299 + i] = PA[0] * cache[248 + i] - PG[0] * cache[248 + i + 1] + 4 * half_xi_ * (cache[190 + i] - cache[190 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[300 + i] = PA[0] * cache[250 + i] - PG[0] * cache[250 + i + 1] + 3 * half_xi_ * (cache[193 + i] - cache[193 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[301 + i] = PA[2] * cache[244 + i] - PG[2] * cache[244 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[302 + i] = PA[1] * cache[248 + i] - PG[1] * cache[248 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[303 + i] = PA[0] * cache[256 + i] - PG[0] * cache[256 + i + 1] + 3 * half_xi_ * (cache[202 + i] - cache[202 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[304 + i] = PA[0] * cache[258 + i] - PG[0] * cache[258 + i + 1] + 2 * half_xi_ * (cache[205 + i] - cache[205 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[305 + i] = PA[2] * cache[250 + i] - PG[2] * cache[250 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[306 + i] = PA[0] * cache[262 + i] - PG[0] * cache[262 + i + 1] + 2 * half_xi_ * (cache[211 + i] - cache[211 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[307 + i] = PA[1] * cache[256 + i] - PG[1] * cache[256 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[308 + i] = PA[0] * cache[266 + i] - PG[0] * cache[266 + i + 1] + 2 * half_xi_ * (cache[217 + i] - cache[217 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[309 + i] = PA[0] * cache[268 + i] - PG[0] * cache[268 + i + 1] + 1 * half_xi_ * (cache[220 + i] - cache[220 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[310 + i] = PA[2] * cache[258 + i] - PG[2] * cache[258 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[311 + i] = PA[0] * cache[272 + i] - PG[0] * cache[272 + i + 1] + 1 * half_xi_ * (cache[226 + i] - cache[226 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[312 + i] = PA[0] * cache[274 + i] - PG[0] * cache[274 + i + 1] + 1 * half_xi_ * (cache[229 + i] - cache[229 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[313 + i] = PA[1] * cache[266 + i] - PG[1] * cache[266 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[314 + i] = PA[0] * cache[278 + i] - PG[0] * cache[278 + i + 1] + 1 * half_xi_ * (cache[235 + i] - cache[235 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[315 + i] = PA[0] * cache[280 + i] - PG[0] * cache[280 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[316 + i] = PA[0] * cache[282 + i] - PG[0] * cache[282 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[317 + i] = PA[0] * cache[284 + i] - PG[0] * cache[284 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[318 + i] = PA[0] * cache[286 + i] - PG[0] * cache[286 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[319 + i] = PA[0] * cache[288 + i] - PG[0] * cache[288 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[320 + i] = PA[0] * cache[290 + i] - PG[0] * cache[290 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[321 + i] = PA[0] * cache[292 + i] - PG[0] * cache[292 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[322 + i] = PA[1] * cache[280 + i] - PG[1] * cache[280 + i + 1] + 6 * half_xi_ * (cache[220 + i] - cache[220 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[323 + i] = PA[2] * cache[280 + i] - PG[2] * cache[280 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[324 + i] = PA[1] * cache[284 + i] - PG[1] * cache[284 + i + 1] + 4 * half_xi_ * (cache[226 + i] - cache[226 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[325 + i] = PA[1] * cache[286 + i] - PG[1] * cache[286 + i + 1] + 3 * half_xi_ * (cache[229 + i] - cache[229 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[326 + i] = PA[1] * cache[288 + i] - PG[1] * cache[288 + i + 1] + 2 * half_xi_ * (cache[232 + i] - cache[232 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[327 + i] = PA[1] * cache[290 + i] - PG[1] * cache[290 + i + 1] + 1 * half_xi_ * (cache[235 + i] - cache[235 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[328 + i] = PA[1] * cache[292 + i] - PG[1] * cache[292 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[329 + i] = PA[2] * cache[292 + i] - PG[2] * cache[292 + i + 1] + 6 * half_xi_ * (cache[235 + i] - cache[235 + i + 1]);
    buf[0] += cache[175];
    buf[1] += cache[178];
    buf[2] += cache[181];
    buf[3] += cache[184];
    buf[4] += cache[187];
    buf[5] += cache[190];
    buf[6] += cache[193];
    buf[7] += cache[196];
    buf[8] += cache[199];
    buf[9] += cache[202];
    buf[10] += cache[205];
    buf[11] += cache[208];
    buf[12] += cache[211];
    buf[13] += cache[214];
    buf[14] += cache[217];
    buf[15] += cache[220];
    buf[16] += cache[223];
    buf[17] += cache[226];
    buf[18] += cache[229];
    buf[19] += cache[232];
    buf[20] += cache[235];
    buf[21] += cache[238];
    buf[22] += cache[240];
    buf[23] += cache[242];
    buf[24] += cache[244];
    buf[25] += cache[246];
    buf[26] += cache[248];
    buf[27] += cache[250];
    buf[28] += cache[252];
    buf[29] += cache[254];
    buf[30] += cache[256];
    buf[31] += cache[258];
    buf[32] += cache[260];
    buf[33] += cache[262];
    buf[34] += cache[264];
    buf[35] += cache[266];
    buf[36] += cache[268];
    buf[37] += cache[270];
    buf[38] += cache[272];
    buf[39] += cache[274];
    buf[40] += cache[276];
    buf[41] += cache[278];
    buf[42] += cache[280];
    buf[43] += cache[282];
    buf[44] += cache[284];
    buf[45] += cache[286];
    buf[46] += cache[288];
    buf[47] += cache[290];
    buf[48] += cache[292];
    buf[49] += cache[294];
    buf[50] += cache[295];
    buf[51] += cache[296];
    buf[52] += cache[297];
    buf[53] += cache[298];
    buf[54] += cache[299];
    buf[55] += cache[300];
    buf[56] += cache[301];
    buf[57] += cache[302];
    buf[58] += cache[303];
    buf[59] += cache[304];
    buf[60] += cache[305];
    buf[61] += cache[306];
    buf[62] += cache[307];
    buf[63] += cache[308];
    buf[64] += cache[309];
    buf[65] += cache[310];
    buf[66] += cache[311];
    buf[67] += cache[312];
    buf[68] += cache[313];
    buf[69] += cache[314];
    buf[70] += cache[315];
    buf[71] += cache[316];
    buf[72] += cache[317];
    buf[73] += cache[318];
    buf[74] += cache[319];
    buf[75] += cache[320];
    buf[76] += cache[321];
    buf[77] += cache[322];
    buf[78] += cache[323];
    buf[79] += cache[324];
    buf[80] += cache[325];
    buf[81] += cache[326];
    buf[82] += cache[327];
    buf[83] += cache[328];
    buf[84] += cache[329];
    return;
}

static void nuc_VRR_5_3(double *buf, double *cache, double coe, double K_ab, double half_xi_, double xi, const double *PA,
                        const double *PG, double PG_dist2, double sr_frac, double lr_frac, double omega_)
{
    double theta;
    double T_0, T_1;
    double scheme_1[10], scheme_0[10];
    double scheme_0_0_0[9];
    T_0 = xi * PG_dist2;
    if (fabs(omega_) < 1E-8)
    {
        cal_boys(8, T_0, &scheme_0[0]);
        cal_boys(8, T_0, &scheme_0[0]);
        for (int i = 0; i <= 8; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * lr_frac * scheme_0[i];
    }
    else
    {
        theta = sqrt(omega_ * omega_ / (omega_ * omega_ + xi));
        T_1 = xi * PG_dist2 * theta * theta;
        cal_boys(8, T_1, &scheme_1[0]);
        for (int i = 0; i <= 8; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * (lr_frac * scheme_0[i] + sr_frac * theta * pow(theta, 2 * i) * scheme_1[i]);
    }
    for (int i = 0; i < 8; i++)
        cache[9 + i] = PA[0] * cache[0 + i] - PG[0] * cache[0 + i + 1];
    for (int i = 0; i < 8; i++)
        cache[17 + i] = PA[1] * cache[0 + i] - PG[1] * cache[0 + i + 1];
    for (int i = 0; i < 8; i++)
        cache[25 + i] = PA[2] * cache[0 + i] - PG[2] * cache[0 + i + 1];
    for (int i = 0; i < 7; i++)
        cache[33 + i] = PA[0] * cache[9 + i] - PG[0] * cache[9 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 7; i++)
        cache[40 + i] = PA[0] * cache[17 + i] - PG[0] * cache[17 + i + 1];
    for (int i = 0; i < 7; i++)
        cache[47 + i] = PA[0] * cache[25 + i] - PG[0] * cache[25 + i + 1];
    for (int i = 0; i < 7; i++)
        cache[54 + i] = PA[1] * cache[17 + i] - PG[1] * cache[17 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 7; i++)
        cache[61 + i] = PA[1] * cache[25 + i] - PG[1] * cache[25 + i + 1];
    for (int i = 0; i < 7; i++)
        cache[68 + i] = PA[2] * cache[25 + i] - PG[2] * cache[25 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 6; i++)
        cache[75 + i] = PA[0] * cache[33 + i] - PG[0] * cache[33 + i + 1] + 2 * half_xi_ * (cache[9 + i] - cache[9 + i + 1]);
    for (int i = 0; i < 6; i++)
        cache[81 + i] = PA[1] * cache[33 + i] - PG[1] * cache[33 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[87 + i] = PA[2] * cache[33 + i] - PG[2] * cache[33 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[93 + i] = PA[0] * cache[54 + i] - PG[0] * cache[54 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[99 + i] = PA[0] * cache[61 + i] - PG[0] * cache[61 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[105 + i] = PA[0] * cache[68 + i] - PG[0] * cache[68 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[111 + i] = PA[1] * cache[54 + i] - PG[1] * cache[54 + i + 1] + 2 * half_xi_ * (cache[17 + i] - cache[17 + i + 1]);
    for (int i = 0; i < 6; i++)
        cache[117 + i] = PA[2] * cache[54 + i] - PG[2] * cache[54 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[123 + i] = PA[1] * cache[68 + i] - PG[1] * cache[68 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[129 + i] = PA[2] * cache[68 + i] - PG[2] * cache[68 + i + 1] + 2 * half_xi_ * (cache[25 + i] - cache[25 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[135 + i] = PA[0] * cache[75 + i] - PG[0] * cache[75 + i + 1] + 3 * half_xi_ * (cache[33 + i] - cache[33 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[140 + i] = PA[1] * cache[75 + i] - PG[1] * cache[75 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[145 + i] = PA[2] * cache[75 + i] - PG[2] * cache[75 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[150 + i] = PA[0] * cache[93 + i] - PG[0] * cache[93 + i + 1] + 1 * half_xi_ * (cache[54 + i] - cache[54 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[155 + i] = PA[1] * cache[87 + i] - PG[1] * cache[87 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[160 + i] = PA[0] * cache[105 + i] - PG[0] * cache[105 + i + 1] + 1 * half_xi_ * (cache[68 + i] - cache[68 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[165 + i] = PA[0] * cache[111 + i] - PG[0] * cache[111 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[170 + i] = PA[0] * cache[117 + i] - PG[0] * cache[117 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[175 + i] = PA[0] * cache[123 + i] - PG[0] * cache[123 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[180 + i] = PA[0] * cache[129 + i] - PG[0] * cache[129 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[185 + i] = PA[1] * cache[111 + i] - PG[1] * cache[111 + i + 1] + 3 * half_xi_ * (cache[54 + i] - cache[54 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[190 + i] = PA[2] * cache[111 + i] - PG[2] * cache[111 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[195 + i] = PA[1] * cache[123 + i] - PG[1] * cache[123 + i + 1] + 1 * half_xi_ * (cache[68 + i] - cache[68 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[200 + i] = PA[1] * cache[129 + i] - PG[1] * cache[129 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[205 + i] = PA[2] * cache[129 + i] - PG[2] * cache[129 + i + 1] + 3 * half_xi_ * (cache[68 + i] - cache[68 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[210 + i] = PA[0] * cache[135 + i] - PG[0] * cache[135 + i + 1] + 4 * half_xi_ * (cache[75 + i] - cache[75 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[214 + i] = PA[1] * cache[135 + i] - PG[1] * cache[135 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[218 + i] = PA[2] * cache[135 + i] - PG[2] * cache[135 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[222 + i] = PA[0] * cache[150 + i] - PG[0] * cache[150 + i + 1] + 2 * half_xi_ * (cache[93 + i] - cache[93 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[226 + i] = PA[1] * cache[145 + i] - PG[1] * cache[145 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[230 + i] = PA[0] * cache[160 + i] - PG[0] * cache[160 + i + 1] + 2 * half_xi_ * (cache[105 + i] - cache[105 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[234 + i] = PA[0] * cache[165 + i] - PG[0] * cache[165 + i + 1] + 1 * half_xi_ * (cache[111 + i] - cache[111 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[238 + i] = PA[2] * cache[150 + i] - PG[2] * cache[150 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[242 + i] = PA[1] * cache[160 + i] - PG[1] * cache[160 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[246 + i] = PA[0] * cache[180 + i] - PG[0] * cache[180 + i + 1] + 1 * half_xi_ * (cache[129 + i] - cache[129 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[250 + i] = PA[0] * cache[185 + i] - PG[0] * cache[185 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[254 + i] = PA[0] * cache[190 + i] - PG[0] * cache[190 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[258 + i] = PA[0] * cache[195 + i] - PG[0] * cache[195 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[262 + i] = PA[0] * cache[200 + i] - PG[0] * cache[200 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[266 + i] = PA[0] * cache[205 + i] - PG[0] * cache[205 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[270 + i] = PA[1] * cache[185 + i] - PG[1] * cache[185 + i + 1] + 4 * half_xi_ * (cache[111 + i] - cache[111 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[274 + i] = PA[2] * cache[185 + i] - PG[2] * cache[185 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[278 + i] = PA[1] * cache[195 + i] - PG[1] * cache[195 + i + 1] + 2 * half_xi_ * (cache[123 + i] - cache[123 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[282 + i] = PA[1] * cache[200 + i] - PG[1] * cache[200 + i + 1] + 1 * half_xi_ * (cache[129 + i] - cache[129 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[286 + i] = PA[1] * cache[205 + i] - PG[1] * cache[205 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[290 + i] = PA[2] * cache[205 + i] - PG[2] * cache[205 + i + 1] + 4 * half_xi_ * (cache[129 + i] - cache[129 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[294 + i] = PA[0] * cache[210 + i] - PG[0] * cache[210 + i + 1] + 5 * half_xi_ * (cache[135 + i] - cache[135 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[297 + i] = PA[1] * cache[210 + i] - PG[1] * cache[210 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[300 + i] = PA[2] * cache[210 + i] - PG[2] * cache[210 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[303 + i] = PA[0] * cache[222 + i] - PG[0] * cache[222 + i + 1] + 3 * half_xi_ * (cache[150 + i] - cache[150 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[306 + i] = PA[1] * cache[218 + i] - PG[1] * cache[218 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[309 + i] = PA[0] * cache[230 + i] - PG[0] * cache[230 + i + 1] + 3 * half_xi_ * (cache[160 + i] - cache[160 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[312 + i] = PA[0] * cache[234 + i] - PG[0] * cache[234 + i + 1] + 2 * half_xi_ * (cache[165 + i] - cache[165 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[315 + i] = PA[2] * cache[222 + i] - PG[2] * cache[222 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[318 + i] = PA[1] * cache[230 + i] - PG[1] * cache[230 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[321 + i] = PA[0] * cache[246 + i] - PG[0] * cache[246 + i + 1] + 2 * half_xi_ * (cache[180 + i] - cache[180 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[324 + i] = PA[0] * cache[250 + i] - PG[0] * cache[250 + i + 1] + 1 * half_xi_ * (cache[185 + i] - cache[185 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[327 + i] = PA[2] * cache[234 + i] - PG[2] * cache[234 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[330 + i] = PA[0] * cache[258 + i] - PG[0] * cache[258 + i + 1] + 1 * half_xi_ * (cache[195 + i] - cache[195 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[333 + i] = PA[1] * cache[246 + i] - PG[1] * cache[246 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[336 + i] = PA[0] * cache[266 + i] - PG[0] * cache[266 + i + 1] + 1 * half_xi_ * (cache[205 + i] - cache[205 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[339 + i] = PA[0] * cache[270 + i] - PG[0] * cache[270 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[342 + i] = PA[0] * cache[274 + i] - PG[0] * cache[274 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[345 + i] = PA[0] * cache[278 + i] - PG[0] * cache[278 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[348 + i] = PA[0] * cache[282 + i] - PG[0] * cache[282 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[351 + i] = PA[0] * cache[286 + i] - PG[0] * cache[286 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[354 + i] = PA[0] * cache[290 + i] - PG[0] * cache[290 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[357 + i] = PA[1] * cache[270 + i] - PG[1] * cache[270 + i + 1] + 5 * half_xi_ * (cache[185 + i] - cache[185 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[360 + i] = PA[2] * cache[270 + i] - PG[2] * cache[270 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[363 + i] = PA[1] * cache[278 + i] - PG[1] * cache[278 + i + 1] + 3 * half_xi_ * (cache[195 + i] - cache[195 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[366 + i] = PA[1] * cache[282 + i] - PG[1] * cache[282 + i + 1] + 2 * half_xi_ * (cache[200 + i] - cache[200 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[369 + i] = PA[1] * cache[286 + i] - PG[1] * cache[286 + i + 1] + 1 * half_xi_ * (cache[205 + i] - cache[205 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[372 + i] = PA[1] * cache[290 + i] - PG[1] * cache[290 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[375 + i] = PA[2] * cache[290 + i] - PG[2] * cache[290 + i + 1] + 5 * half_xi_ * (cache[205 + i] - cache[205 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[378 + i] = PA[0] * cache[294 + i] - PG[0] * cache[294 + i + 1] + 6 * half_xi_ * (cache[210 + i] - cache[210 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[380 + i] = PA[1] * cache[294 + i] - PG[1] * cache[294 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[382 + i] = PA[2] * cache[294 + i] - PG[2] * cache[294 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[384 + i] = PA[0] * cache[303 + i] - PG[0] * cache[303 + i + 1] + 4 * half_xi_ * (cache[222 + i] - cache[222 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[386 + i] = PA[1] * cache[300 + i] - PG[1] * cache[300 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[388 + i] = PA[0] * cache[309 + i] - PG[0] * cache[309 + i + 1] + 4 * half_xi_ * (cache[230 + i] - cache[230 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[390 + i] = PA[0] * cache[312 + i] - PG[0] * cache[312 + i + 1] + 3 * half_xi_ * (cache[234 + i] - cache[234 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[392 + i] = PA[2] * cache[303 + i] - PG[2] * cache[303 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[394 + i] = PA[1] * cache[309 + i] - PG[1] * cache[309 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[396 + i] = PA[0] * cache[321 + i] - PG[0] * cache[321 + i + 1] + 3 * half_xi_ * (cache[246 + i] - cache[246 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[398 + i] = PA[0] * cache[324 + i] - PG[0] * cache[324 + i + 1] + 2 * half_xi_ * (cache[250 + i] - cache[250 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[400 + i] = PA[2] * cache[312 + i] - PG[2] * cache[312 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[402 + i] = PA[0] * cache[330 + i] - PG[0] * cache[330 + i + 1] + 2 * half_xi_ * (cache[258 + i] - cache[258 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[404 + i] = PA[1] * cache[321 + i] - PG[1] * cache[321 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[406 + i] = PA[0] * cache[336 + i] - PG[0] * cache[336 + i + 1] + 2 * half_xi_ * (cache[266 + i] - cache[266 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[408 + i] = PA[0] * cache[339 + i] - PG[0] * cache[339 + i + 1] + 1 * half_xi_ * (cache[270 + i] - cache[270 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[410 + i] = PA[2] * cache[324 + i] - PG[2] * cache[324 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[412 + i] = PA[0] * cache[345 + i] - PG[0] * cache[345 + i + 1] + 1 * half_xi_ * (cache[278 + i] - cache[278 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[414 + i] = PA[0] * cache[348 + i] - PG[0] * cache[348 + i + 1] + 1 * half_xi_ * (cache[282 + i] - cache[282 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[416 + i] = PA[1] * cache[336 + i] - PG[1] * cache[336 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[418 + i] = PA[0] * cache[354 + i] - PG[0] * cache[354 + i + 1] + 1 * half_xi_ * (cache[290 + i] - cache[290 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[420 + i] = PA[0] * cache[357 + i] - PG[0] * cache[357 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[422 + i] = PA[0] * cache[360 + i] - PG[0] * cache[360 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[424 + i] = PA[0] * cache[363 + i] - PG[0] * cache[363 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[426 + i] = PA[0] * cache[366 + i] - PG[0] * cache[366 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[428 + i] = PA[0] * cache[369 + i] - PG[0] * cache[369 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[430 + i] = PA[0] * cache[372 + i] - PG[0] * cache[372 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[432 + i] = PA[0] * cache[375 + i] - PG[0] * cache[375 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[434 + i] = PA[1] * cache[357 + i] - PG[1] * cache[357 + i + 1] + 6 * half_xi_ * (cache[270 + i] - cache[270 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[436 + i] = PA[2] * cache[357 + i] - PG[2] * cache[357 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[438 + i] = PA[1] * cache[363 + i] - PG[1] * cache[363 + i + 1] + 4 * half_xi_ * (cache[278 + i] - cache[278 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[440 + i] = PA[1] * cache[366 + i] - PG[1] * cache[366 + i + 1] + 3 * half_xi_ * (cache[282 + i] - cache[282 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[442 + i] = PA[1] * cache[369 + i] - PG[1] * cache[369 + i + 1] + 2 * half_xi_ * (cache[286 + i] - cache[286 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[444 + i] = PA[1] * cache[372 + i] - PG[1] * cache[372 + i + 1] + 1 * half_xi_ * (cache[290 + i] - cache[290 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[446 + i] = PA[1] * cache[375 + i] - PG[1] * cache[375 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[448 + i] = PA[2] * cache[375 + i] - PG[2] * cache[375 + i + 1] + 6 * half_xi_ * (cache[290 + i] - cache[290 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[450 + i] = PA[0] * cache[378 + i] - PG[0] * cache[378 + i + 1] + 7 * half_xi_ * (cache[294 + i] - cache[294 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[451 + i] = PA[1] * cache[378 + i] - PG[1] * cache[378 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[452 + i] = PA[2] * cache[378 + i] - PG[2] * cache[378 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[453 + i] = PA[0] * cache[384 + i] - PG[0] * cache[384 + i + 1] + 5 * half_xi_ * (cache[303 + i] - cache[303 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[454 + i] = PA[1] * cache[382 + i] - PG[1] * cache[382 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[455 + i] = PA[0] * cache[388 + i] - PG[0] * cache[388 + i + 1] + 5 * half_xi_ * (cache[309 + i] - cache[309 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[456 + i] = PA[0] * cache[390 + i] - PG[0] * cache[390 + i + 1] + 4 * half_xi_ * (cache[312 + i] - cache[312 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[457 + i] = PA[2] * cache[384 + i] - PG[2] * cache[384 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[458 + i] = PA[1] * cache[388 + i] - PG[1] * cache[388 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[459 + i] = PA[0] * cache[396 + i] - PG[0] * cache[396 + i + 1] + 4 * half_xi_ * (cache[321 + i] - cache[321 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[460 + i] = PA[0] * cache[398 + i] - PG[0] * cache[398 + i + 1] + 3 * half_xi_ * (cache[324 + i] - cache[324 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[461 + i] = PA[2] * cache[390 + i] - PG[2] * cache[390 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[462 + i] = PA[0] * cache[402 + i] - PG[0] * cache[402 + i + 1] + 3 * half_xi_ * (cache[330 + i] - cache[330 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[463 + i] = PA[1] * cache[396 + i] - PG[1] * cache[396 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[464 + i] = PA[0] * cache[406 + i] - PG[0] * cache[406 + i + 1] + 3 * half_xi_ * (cache[336 + i] - cache[336 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[465 + i] = PA[0] * cache[408 + i] - PG[0] * cache[408 + i + 1] + 2 * half_xi_ * (cache[339 + i] - cache[339 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[466 + i] = PA[2] * cache[398 + i] - PG[2] * cache[398 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[467 + i] = PA[0] * cache[412 + i] - PG[0] * cache[412 + i + 1] + 2 * half_xi_ * (cache[345 + i] - cache[345 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[468 + i] = PA[0] * cache[414 + i] - PG[0] * cache[414 + i + 1] + 2 * half_xi_ * (cache[348 + i] - cache[348 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[469 + i] = PA[1] * cache[406 + i] - PG[1] * cache[406 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[470 + i] = PA[0] * cache[418 + i] - PG[0] * cache[418 + i + 1] + 2 * half_xi_ * (cache[354 + i] - cache[354 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[471 + i] = PA[0] * cache[420 + i] - PG[0] * cache[420 + i + 1] + 1 * half_xi_ * (cache[357 + i] - cache[357 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[472 + i] = PA[2] * cache[408 + i] - PG[2] * cache[408 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[473 + i] = PA[0] * cache[424 + i] - PG[0] * cache[424 + i + 1] + 1 * half_xi_ * (cache[363 + i] - cache[363 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[474 + i] = PA[0] * cache[426 + i] - PG[0] * cache[426 + i + 1] + 1 * half_xi_ * (cache[366 + i] - cache[366 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[475 + i] = PA[0] * cache[428 + i] - PG[0] * cache[428 + i + 1] + 1 * half_xi_ * (cache[369 + i] - cache[369 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[476 + i] = PA[1] * cache[418 + i] - PG[1] * cache[418 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[477 + i] = PA[0] * cache[432 + i] - PG[0] * cache[432 + i + 1] + 1 * half_xi_ * (cache[375 + i] - cache[375 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[478 + i] = PA[0] * cache[434 + i] - PG[0] * cache[434 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[479 + i] = PA[0] * cache[436 + i] - PG[0] * cache[436 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[480 + i] = PA[0] * cache[438 + i] - PG[0] * cache[438 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[481 + i] = PA[0] * cache[440 + i] - PG[0] * cache[440 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[482 + i] = PA[0] * cache[442 + i] - PG[0] * cache[442 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[483 + i] = PA[0] * cache[444 + i] - PG[0] * cache[444 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[484 + i] = PA[0] * cache[446 + i] - PG[0] * cache[446 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[485 + i] = PA[0] * cache[448 + i] - PG[0] * cache[448 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[486 + i] = PA[1] * cache[434 + i] - PG[1] * cache[434 + i + 1] + 7 * half_xi_ * (cache[357 + i] - cache[357 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[487 + i] = PA[2] * cache[434 + i] - PG[2] * cache[434 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[488 + i] = PA[1] * cache[438 + i] - PG[1] * cache[438 + i + 1] + 5 * half_xi_ * (cache[363 + i] - cache[363 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[489 + i] = PA[1] * cache[440 + i] - PG[1] * cache[440 + i + 1] + 4 * half_xi_ * (cache[366 + i] - cache[366 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[490 + i] = PA[1] * cache[442 + i] - PG[1] * cache[442 + i + 1] + 3 * half_xi_ * (cache[369 + i] - cache[369 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[491 + i] = PA[1] * cache[444 + i] - PG[1] * cache[444 + i + 1] + 2 * half_xi_ * (cache[372 + i] - cache[372 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[492 + i] = PA[1] * cache[446 + i] - PG[1] * cache[446 + i + 1] + 1 * half_xi_ * (cache[375 + i] - cache[375 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[493 + i] = PA[1] * cache[448 + i] - PG[1] * cache[448 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[494 + i] = PA[2] * cache[448 + i] - PG[2] * cache[448 + i + 1] + 7 * half_xi_ * (cache[375 + i] - cache[375 + i + 1]);
    buf[0] += cache[210];
    buf[1] += cache[214];
    buf[2] += cache[218];
    buf[3] += cache[222];
    buf[4] += cache[226];
    buf[5] += cache[230];
    buf[6] += cache[234];
    buf[7] += cache[238];
    buf[8] += cache[242];
    buf[9] += cache[246];
    buf[10] += cache[250];
    buf[11] += cache[254];
    buf[12] += cache[258];
    buf[13] += cache[262];
    buf[14] += cache[266];
    buf[15] += cache[270];
    buf[16] += cache[274];
    buf[17] += cache[278];
    buf[18] += cache[282];
    buf[19] += cache[286];
    buf[20] += cache[290];
    buf[21] += cache[294];
    buf[22] += cache[297];
    buf[23] += cache[300];
    buf[24] += cache[303];
    buf[25] += cache[306];
    buf[26] += cache[309];
    buf[27] += cache[312];
    buf[28] += cache[315];
    buf[29] += cache[318];
    buf[30] += cache[321];
    buf[31] += cache[324];
    buf[32] += cache[327];
    buf[33] += cache[330];
    buf[34] += cache[333];
    buf[35] += cache[336];
    buf[36] += cache[339];
    buf[37] += cache[342];
    buf[38] += cache[345];
    buf[39] += cache[348];
    buf[40] += cache[351];
    buf[41] += cache[354];
    buf[42] += cache[357];
    buf[43] += cache[360];
    buf[44] += cache[363];
    buf[45] += cache[366];
    buf[46] += cache[369];
    buf[47] += cache[372];
    buf[48] += cache[375];
    buf[49] += cache[378];
    buf[50] += cache[380];
    buf[51] += cache[382];
    buf[52] += cache[384];
    buf[53] += cache[386];
    buf[54] += cache[388];
    buf[55] += cache[390];
    buf[56] += cache[392];
    buf[57] += cache[394];
    buf[58] += cache[396];
    buf[59] += cache[398];
    buf[60] += cache[400];
    buf[61] += cache[402];
    buf[62] += cache[404];
    buf[63] += cache[406];
    buf[64] += cache[408];
    buf[65] += cache[410];
    buf[66] += cache[412];
    buf[67] += cache[414];
    buf[68] += cache[416];
    buf[69] += cache[418];
    buf[70] += cache[420];
    buf[71] += cache[422];
    buf[72] += cache[424];
    buf[73] += cache[426];
    buf[74] += cache[428];
    buf[75] += cache[430];
    buf[76] += cache[432];
    buf[77] += cache[434];
    buf[78] += cache[436];
    buf[79] += cache[438];
    buf[80] += cache[440];
    buf[81] += cache[442];
    buf[82] += cache[444];
    buf[83] += cache[446];
    buf[84] += cache[448];
    buf[85] += cache[450];
    buf[86] += cache[451];
    buf[87] += cache[452];
    buf[88] += cache[453];
    buf[89] += cache[454];
    buf[90] += cache[455];
    buf[91] += cache[456];
    buf[92] += cache[457];
    buf[93] += cache[458];
    buf[94] += cache[459];
    buf[95] += cache[460];
    buf[96] += cache[461];
    buf[97] += cache[462];
    buf[98] += cache[463];
    buf[99] += cache[464];
    buf[100] += cache[465];
    buf[101] += cache[466];
    buf[102] += cache[467];
    buf[103] += cache[468];
    buf[104] += cache[469];
    buf[105] += cache[470];
    buf[106] += cache[471];
    buf[107] += cache[472];
    buf[108] += cache[473];
    buf[109] += cache[474];
    buf[110] += cache[475];
    buf[111] += cache[476];
    buf[112] += cache[477];
    buf[113] += cache[478];
    buf[114] += cache[479];
    buf[115] += cache[480];
    buf[116] += cache[481];
    buf[117] += cache[482];
    buf[118] += cache[483];
    buf[119] += cache[484];
    buf[120] += cache[485];
    buf[121] += cache[486];
    buf[122] += cache[487];
    buf[123] += cache[488];
    buf[124] += cache[489];
    buf[125] += cache[490];
    buf[126] += cache[491];
    buf[127] += cache[492];
    buf[128] += cache[493];
    buf[129] += cache[494];
    return;
}

static void nuc_VRR_5_4(double *buf, double *cache, double coe, double K_ab, double half_xi_, double xi, const double *PA,
                        const double *PG, double PG_dist2, double sr_frac, double lr_frac, double omega_)
{
    double theta;
    double T_0, T_1;
    double scheme_1[10], scheme_0[10];
    double scheme_0_0_0[10];
    T_0 = xi * PG_dist2;
    if (fabs(omega_) < 1E-8)
    {
        cal_boys(9, T_0, &scheme_0[0]);
        cal_boys(9, T_0, &scheme_0[0]);
        for (int i = 0; i <= 9; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * lr_frac * scheme_0[i];
    }
    else
    {
        theta = sqrt(omega_ * omega_ / (omega_ * omega_ + xi));
        T_1 = xi * PG_dist2 * theta * theta;
        cal_boys(9, T_1, &scheme_1[0]);
        for (int i = 0; i <= 9; i++)
            cache[i] = 4 * coe * K_ab * half_xi_ * MY_PI * (lr_frac * scheme_0[i] + sr_frac * theta * pow(theta, 2 * i) * scheme_1[i]);
    }
    for (int i = 0; i < 9; i++)
        cache[10 + i] = PA[0] * cache[0 + i] - PG[0] * cache[0 + i + 1];
    for (int i = 0; i < 9; i++)
        cache[19 + i] = PA[1] * cache[0 + i] - PG[1] * cache[0 + i + 1];
    for (int i = 0; i < 9; i++)
        cache[28 + i] = PA[2] * cache[0 + i] - PG[2] * cache[0 + i + 1];
    for (int i = 0; i < 8; i++)
        cache[37 + i] = PA[0] * cache[10 + i] - PG[0] * cache[10 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 8; i++)
        cache[45 + i] = PA[0] * cache[19 + i] - PG[0] * cache[19 + i + 1];
    for (int i = 0; i < 8; i++)
        cache[53 + i] = PA[0] * cache[28 + i] - PG[0] * cache[28 + i + 1];
    for (int i = 0; i < 8; i++)
        cache[61 + i] = PA[1] * cache[19 + i] - PG[1] * cache[19 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 8; i++)
        cache[69 + i] = PA[1] * cache[28 + i] - PG[1] * cache[28 + i + 1];
    for (int i = 0; i < 8; i++)
        cache[77 + i] = PA[2] * cache[28 + i] - PG[2] * cache[28 + i + 1] + 1 * half_xi_ * (cache[0 + i] - cache[0 + i + 1]);
    for (int i = 0; i < 7; i++)
        cache[85 + i] = PA[0] * cache[37 + i] - PG[0] * cache[37 + i + 1] + 2 * half_xi_ * (cache[10 + i] - cache[10 + i + 1]);
    for (int i = 0; i < 7; i++)
        cache[92 + i] = PA[1] * cache[37 + i] - PG[1] * cache[37 + i + 1];
    for (int i = 0; i < 7; i++)
        cache[99 + i] = PA[2] * cache[37 + i] - PG[2] * cache[37 + i + 1];
    for (int i = 0; i < 7; i++)
        cache[106 + i] = PA[0] * cache[61 + i] - PG[0] * cache[61 + i + 1];
    for (int i = 0; i < 7; i++)
        cache[113 + i] = PA[0] * cache[69 + i] - PG[0] * cache[69 + i + 1];
    for (int i = 0; i < 7; i++)
        cache[120 + i] = PA[0] * cache[77 + i] - PG[0] * cache[77 + i + 1];
    for (int i = 0; i < 7; i++)
        cache[127 + i] = PA[1] * cache[61 + i] - PG[1] * cache[61 + i + 1] + 2 * half_xi_ * (cache[19 + i] - cache[19 + i + 1]);
    for (int i = 0; i < 7; i++)
        cache[134 + i] = PA[2] * cache[61 + i] - PG[2] * cache[61 + i + 1];
    for (int i = 0; i < 7; i++)
        cache[141 + i] = PA[1] * cache[77 + i] - PG[1] * cache[77 + i + 1];
    for (int i = 0; i < 7; i++)
        cache[148 + i] = PA[2] * cache[77 + i] - PG[2] * cache[77 + i + 1] + 2 * half_xi_ * (cache[28 + i] - cache[28 + i + 1]);
    for (int i = 0; i < 6; i++)
        cache[155 + i] = PA[0] * cache[85 + i] - PG[0] * cache[85 + i + 1] + 3 * half_xi_ * (cache[37 + i] - cache[37 + i + 1]);
    for (int i = 0; i < 6; i++)
        cache[161 + i] = PA[1] * cache[85 + i] - PG[1] * cache[85 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[167 + i] = PA[2] * cache[85 + i] - PG[2] * cache[85 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[173 + i] = PA[0] * cache[106 + i] - PG[0] * cache[106 + i + 1] + 1 * half_xi_ * (cache[61 + i] - cache[61 + i + 1]);
    for (int i = 0; i < 6; i++)
        cache[179 + i] = PA[1] * cache[99 + i] - PG[1] * cache[99 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[185 + i] = PA[0] * cache[120 + i] - PG[0] * cache[120 + i + 1] + 1 * half_xi_ * (cache[77 + i] - cache[77 + i + 1]);
    for (int i = 0; i < 6; i++)
        cache[191 + i] = PA[0] * cache[127 + i] - PG[0] * cache[127 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[197 + i] = PA[0] * cache[134 + i] - PG[0] * cache[134 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[203 + i] = PA[0] * cache[141 + i] - PG[0] * cache[141 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[209 + i] = PA[0] * cache[148 + i] - PG[0] * cache[148 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[215 + i] = PA[1] * cache[127 + i] - PG[1] * cache[127 + i + 1] + 3 * half_xi_ * (cache[61 + i] - cache[61 + i + 1]);
    for (int i = 0; i < 6; i++)
        cache[221 + i] = PA[2] * cache[127 + i] - PG[2] * cache[127 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[227 + i] = PA[1] * cache[141 + i] - PG[1] * cache[141 + i + 1] + 1 * half_xi_ * (cache[77 + i] - cache[77 + i + 1]);
    for (int i = 0; i < 6; i++)
        cache[233 + i] = PA[1] * cache[148 + i] - PG[1] * cache[148 + i + 1];
    for (int i = 0; i < 6; i++)
        cache[239 + i] = PA[2] * cache[148 + i] - PG[2] * cache[148 + i + 1] + 3 * half_xi_ * (cache[77 + i] - cache[77 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[245 + i] = PA[0] * cache[155 + i] - PG[0] * cache[155 + i + 1] + 4 * half_xi_ * (cache[85 + i] - cache[85 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[250 + i] = PA[1] * cache[155 + i] - PG[1] * cache[155 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[255 + i] = PA[2] * cache[155 + i] - PG[2] * cache[155 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[260 + i] = PA[0] * cache[173 + i] - PG[0] * cache[173 + i + 1] + 2 * half_xi_ * (cache[106 + i] - cache[106 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[265 + i] = PA[1] * cache[167 + i] - PG[1] * cache[167 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[270 + i] = PA[0] * cache[185 + i] - PG[0] * cache[185 + i + 1] + 2 * half_xi_ * (cache[120 + i] - cache[120 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[275 + i] = PA[0] * cache[191 + i] - PG[0] * cache[191 + i + 1] + 1 * half_xi_ * (cache[127 + i] - cache[127 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[280 + i] = PA[2] * cache[173 + i] - PG[2] * cache[173 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[285 + i] = PA[1] * cache[185 + i] - PG[1] * cache[185 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[290 + i] = PA[0] * cache[209 + i] - PG[0] * cache[209 + i + 1] + 1 * half_xi_ * (cache[148 + i] - cache[148 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[295 + i] = PA[0] * cache[215 + i] - PG[0] * cache[215 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[300 + i] = PA[0] * cache[221 + i] - PG[0] * cache[221 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[305 + i] = PA[0] * cache[227 + i] - PG[0] * cache[227 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[310 + i] = PA[0] * cache[233 + i] - PG[0] * cache[233 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[315 + i] = PA[0] * cache[239 + i] - PG[0] * cache[239 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[320 + i] = PA[1] * cache[215 + i] - PG[1] * cache[215 + i + 1] + 4 * half_xi_ * (cache[127 + i] - cache[127 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[325 + i] = PA[2] * cache[215 + i] - PG[2] * cache[215 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[330 + i] = PA[1] * cache[227 + i] - PG[1] * cache[227 + i + 1] + 2 * half_xi_ * (cache[141 + i] - cache[141 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[335 + i] = PA[1] * cache[233 + i] - PG[1] * cache[233 + i + 1] + 1 * half_xi_ * (cache[148 + i] - cache[148 + i + 1]);
    for (int i = 0; i < 5; i++)
        cache[340 + i] = PA[1] * cache[239 + i] - PG[1] * cache[239 + i + 1];
    for (int i = 0; i < 5; i++)
        cache[345 + i] = PA[2] * cache[239 + i] - PG[2] * cache[239 + i + 1] + 4 * half_xi_ * (cache[148 + i] - cache[148 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[350 + i] = PA[0] * cache[245 + i] - PG[0] * cache[245 + i + 1] + 5 * half_xi_ * (cache[155 + i] - cache[155 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[354 + i] = PA[1] * cache[245 + i] - PG[1] * cache[245 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[358 + i] = PA[2] * cache[245 + i] - PG[2] * cache[245 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[362 + i] = PA[0] * cache[260 + i] - PG[0] * cache[260 + i + 1] + 3 * half_xi_ * (cache[173 + i] - cache[173 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[366 + i] = PA[1] * cache[255 + i] - PG[1] * cache[255 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[370 + i] = PA[0] * cache[270 + i] - PG[0] * cache[270 + i + 1] + 3 * half_xi_ * (cache[185 + i] - cache[185 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[374 + i] = PA[0] * cache[275 + i] - PG[0] * cache[275 + i + 1] + 2 * half_xi_ * (cache[191 + i] - cache[191 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[378 + i] = PA[2] * cache[260 + i] - PG[2] * cache[260 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[382 + i] = PA[1] * cache[270 + i] - PG[1] * cache[270 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[386 + i] = PA[0] * cache[290 + i] - PG[0] * cache[290 + i + 1] + 2 * half_xi_ * (cache[209 + i] - cache[209 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[390 + i] = PA[0] * cache[295 + i] - PG[0] * cache[295 + i + 1] + 1 * half_xi_ * (cache[215 + i] - cache[215 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[394 + i] = PA[2] * cache[275 + i] - PG[2] * cache[275 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[398 + i] = PA[0] * cache[305 + i] - PG[0] * cache[305 + i + 1] + 1 * half_xi_ * (cache[227 + i] - cache[227 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[402 + i] = PA[1] * cache[290 + i] - PG[1] * cache[290 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[406 + i] = PA[0] * cache[315 + i] - PG[0] * cache[315 + i + 1] + 1 * half_xi_ * (cache[239 + i] - cache[239 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[410 + i] = PA[0] * cache[320 + i] - PG[0] * cache[320 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[414 + i] = PA[0] * cache[325 + i] - PG[0] * cache[325 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[418 + i] = PA[0] * cache[330 + i] - PG[0] * cache[330 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[422 + i] = PA[0] * cache[335 + i] - PG[0] * cache[335 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[426 + i] = PA[0] * cache[340 + i] - PG[0] * cache[340 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[430 + i] = PA[0] * cache[345 + i] - PG[0] * cache[345 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[434 + i] = PA[1] * cache[320 + i] - PG[1] * cache[320 + i + 1] + 5 * half_xi_ * (cache[215 + i] - cache[215 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[438 + i] = PA[2] * cache[320 + i] - PG[2] * cache[320 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[442 + i] = PA[1] * cache[330 + i] - PG[1] * cache[330 + i + 1] + 3 * half_xi_ * (cache[227 + i] - cache[227 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[446 + i] = PA[1] * cache[335 + i] - PG[1] * cache[335 + i + 1] + 2 * half_xi_ * (cache[233 + i] - cache[233 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[450 + i] = PA[1] * cache[340 + i] - PG[1] * cache[340 + i + 1] + 1 * half_xi_ * (cache[239 + i] - cache[239 + i + 1]);
    for (int i = 0; i < 4; i++)
        cache[454 + i] = PA[1] * cache[345 + i] - PG[1] * cache[345 + i + 1];
    for (int i = 0; i < 4; i++)
        cache[458 + i] = PA[2] * cache[345 + i] - PG[2] * cache[345 + i + 1] + 5 * half_xi_ * (cache[239 + i] - cache[239 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[462 + i] = PA[0] * cache[350 + i] - PG[0] * cache[350 + i + 1] + 6 * half_xi_ * (cache[245 + i] - cache[245 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[465 + i] = PA[1] * cache[350 + i] - PG[1] * cache[350 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[468 + i] = PA[2] * cache[350 + i] - PG[2] * cache[350 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[471 + i] = PA[0] * cache[362 + i] - PG[0] * cache[362 + i + 1] + 4 * half_xi_ * (cache[260 + i] - cache[260 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[474 + i] = PA[1] * cache[358 + i] - PG[1] * cache[358 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[477 + i] = PA[0] * cache[370 + i] - PG[0] * cache[370 + i + 1] + 4 * half_xi_ * (cache[270 + i] - cache[270 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[480 + i] = PA[0] * cache[374 + i] - PG[0] * cache[374 + i + 1] + 3 * half_xi_ * (cache[275 + i] - cache[275 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[483 + i] = PA[2] * cache[362 + i] - PG[2] * cache[362 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[486 + i] = PA[1] * cache[370 + i] - PG[1] * cache[370 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[489 + i] = PA[0] * cache[386 + i] - PG[0] * cache[386 + i + 1] + 3 * half_xi_ * (cache[290 + i] - cache[290 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[492 + i] = PA[0] * cache[390 + i] - PG[0] * cache[390 + i + 1] + 2 * half_xi_ * (cache[295 + i] - cache[295 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[495 + i] = PA[2] * cache[374 + i] - PG[2] * cache[374 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[498 + i] = PA[0] * cache[398 + i] - PG[0] * cache[398 + i + 1] + 2 * half_xi_ * (cache[305 + i] - cache[305 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[501 + i] = PA[1] * cache[386 + i] - PG[1] * cache[386 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[504 + i] = PA[0] * cache[406 + i] - PG[0] * cache[406 + i + 1] + 2 * half_xi_ * (cache[315 + i] - cache[315 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[507 + i] = PA[0] * cache[410 + i] - PG[0] * cache[410 + i + 1] + 1 * half_xi_ * (cache[320 + i] - cache[320 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[510 + i] = PA[2] * cache[390 + i] - PG[2] * cache[390 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[513 + i] = PA[0] * cache[418 + i] - PG[0] * cache[418 + i + 1] + 1 * half_xi_ * (cache[330 + i] - cache[330 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[516 + i] = PA[0] * cache[422 + i] - PG[0] * cache[422 + i + 1] + 1 * half_xi_ * (cache[335 + i] - cache[335 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[519 + i] = PA[1] * cache[406 + i] - PG[1] * cache[406 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[522 + i] = PA[0] * cache[430 + i] - PG[0] * cache[430 + i + 1] + 1 * half_xi_ * (cache[345 + i] - cache[345 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[525 + i] = PA[0] * cache[434 + i] - PG[0] * cache[434 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[528 + i] = PA[0] * cache[438 + i] - PG[0] * cache[438 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[531 + i] = PA[0] * cache[442 + i] - PG[0] * cache[442 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[534 + i] = PA[0] * cache[446 + i] - PG[0] * cache[446 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[537 + i] = PA[0] * cache[450 + i] - PG[0] * cache[450 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[540 + i] = PA[0] * cache[454 + i] - PG[0] * cache[454 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[543 + i] = PA[0] * cache[458 + i] - PG[0] * cache[458 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[546 + i] = PA[1] * cache[434 + i] - PG[1] * cache[434 + i + 1] + 6 * half_xi_ * (cache[320 + i] - cache[320 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[549 + i] = PA[2] * cache[434 + i] - PG[2] * cache[434 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[552 + i] = PA[1] * cache[442 + i] - PG[1] * cache[442 + i + 1] + 4 * half_xi_ * (cache[330 + i] - cache[330 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[555 + i] = PA[1] * cache[446 + i] - PG[1] * cache[446 + i + 1] + 3 * half_xi_ * (cache[335 + i] - cache[335 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[558 + i] = PA[1] * cache[450 + i] - PG[1] * cache[450 + i + 1] + 2 * half_xi_ * (cache[340 + i] - cache[340 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[561 + i] = PA[1] * cache[454 + i] - PG[1] * cache[454 + i + 1] + 1 * half_xi_ * (cache[345 + i] - cache[345 + i + 1]);
    for (int i = 0; i < 3; i++)
        cache[564 + i] = PA[1] * cache[458 + i] - PG[1] * cache[458 + i + 1];
    for (int i = 0; i < 3; i++)
        cache[567 + i] = PA[2] * cache[458 + i] - PG[2] * cache[458 + i + 1] + 6 * half_xi_ * (cache[345 + i] - cache[345 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[570 + i] = PA[0] * cache[462 + i] - PG[0] * cache[462 + i + 1] + 7 * half_xi_ * (cache[350 + i] - cache[350 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[572 + i] = PA[1] * cache[462 + i] - PG[1] * cache[462 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[574 + i] = PA[2] * cache[462 + i] - PG[2] * cache[462 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[576 + i] = PA[0] * cache[471 + i] - PG[0] * cache[471 + i + 1] + 5 * half_xi_ * (cache[362 + i] - cache[362 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[578 + i] = PA[1] * cache[468 + i] - PG[1] * cache[468 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[580 + i] = PA[0] * cache[477 + i] - PG[0] * cache[477 + i + 1] + 5 * half_xi_ * (cache[370 + i] - cache[370 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[582 + i] = PA[0] * cache[480 + i] - PG[0] * cache[480 + i + 1] + 4 * half_xi_ * (cache[374 + i] - cache[374 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[584 + i] = PA[2] * cache[471 + i] - PG[2] * cache[471 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[586 + i] = PA[1] * cache[477 + i] - PG[1] * cache[477 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[588 + i] = PA[0] * cache[489 + i] - PG[0] * cache[489 + i + 1] + 4 * half_xi_ * (cache[386 + i] - cache[386 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[590 + i] = PA[0] * cache[492 + i] - PG[0] * cache[492 + i + 1] + 3 * half_xi_ * (cache[390 + i] - cache[390 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[592 + i] = PA[2] * cache[480 + i] - PG[2] * cache[480 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[594 + i] = PA[0] * cache[498 + i] - PG[0] * cache[498 + i + 1] + 3 * half_xi_ * (cache[398 + i] - cache[398 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[596 + i] = PA[1] * cache[489 + i] - PG[1] * cache[489 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[598 + i] = PA[0] * cache[504 + i] - PG[0] * cache[504 + i + 1] + 3 * half_xi_ * (cache[406 + i] - cache[406 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[600 + i] = PA[0] * cache[507 + i] - PG[0] * cache[507 + i + 1] + 2 * half_xi_ * (cache[410 + i] - cache[410 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[602 + i] = PA[2] * cache[492 + i] - PG[2] * cache[492 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[604 + i] = PA[0] * cache[513 + i] - PG[0] * cache[513 + i + 1] + 2 * half_xi_ * (cache[418 + i] - cache[418 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[606 + i] = PA[0] * cache[516 + i] - PG[0] * cache[516 + i + 1] + 2 * half_xi_ * (cache[422 + i] - cache[422 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[608 + i] = PA[1] * cache[504 + i] - PG[1] * cache[504 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[610 + i] = PA[0] * cache[522 + i] - PG[0] * cache[522 + i + 1] + 2 * half_xi_ * (cache[430 + i] - cache[430 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[612 + i] = PA[0] * cache[525 + i] - PG[0] * cache[525 + i + 1] + 1 * half_xi_ * (cache[434 + i] - cache[434 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[614 + i] = PA[2] * cache[507 + i] - PG[2] * cache[507 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[616 + i] = PA[0] * cache[531 + i] - PG[0] * cache[531 + i + 1] + 1 * half_xi_ * (cache[442 + i] - cache[442 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[618 + i] = PA[0] * cache[534 + i] - PG[0] * cache[534 + i + 1] + 1 * half_xi_ * (cache[446 + i] - cache[446 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[620 + i] = PA[0] * cache[537 + i] - PG[0] * cache[537 + i + 1] + 1 * half_xi_ * (cache[450 + i] - cache[450 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[622 + i] = PA[1] * cache[522 + i] - PG[1] * cache[522 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[624 + i] = PA[0] * cache[543 + i] - PG[0] * cache[543 + i + 1] + 1 * half_xi_ * (cache[458 + i] - cache[458 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[626 + i] = PA[0] * cache[546 + i] - PG[0] * cache[546 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[628 + i] = PA[0] * cache[549 + i] - PG[0] * cache[549 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[630 + i] = PA[0] * cache[552 + i] - PG[0] * cache[552 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[632 + i] = PA[0] * cache[555 + i] - PG[0] * cache[555 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[634 + i] = PA[0] * cache[558 + i] - PG[0] * cache[558 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[636 + i] = PA[0] * cache[561 + i] - PG[0] * cache[561 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[638 + i] = PA[0] * cache[564 + i] - PG[0] * cache[564 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[640 + i] = PA[0] * cache[567 + i] - PG[0] * cache[567 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[642 + i] = PA[1] * cache[546 + i] - PG[1] * cache[546 + i + 1] + 7 * half_xi_ * (cache[434 + i] - cache[434 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[644 + i] = PA[2] * cache[546 + i] - PG[2] * cache[546 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[646 + i] = PA[1] * cache[552 + i] - PG[1] * cache[552 + i + 1] + 5 * half_xi_ * (cache[442 + i] - cache[442 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[648 + i] = PA[1] * cache[555 + i] - PG[1] * cache[555 + i + 1] + 4 * half_xi_ * (cache[446 + i] - cache[446 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[650 + i] = PA[1] * cache[558 + i] - PG[1] * cache[558 + i + 1] + 3 * half_xi_ * (cache[450 + i] - cache[450 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[652 + i] = PA[1] * cache[561 + i] - PG[1] * cache[561 + i + 1] + 2 * half_xi_ * (cache[454 + i] - cache[454 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[654 + i] = PA[1] * cache[564 + i] - PG[1] * cache[564 + i + 1] + 1 * half_xi_ * (cache[458 + i] - cache[458 + i + 1]);
    for (int i = 0; i < 2; i++)
        cache[656 + i] = PA[1] * cache[567 + i] - PG[1] * cache[567 + i + 1];
    for (int i = 0; i < 2; i++)
        cache[658 + i] = PA[2] * cache[567 + i] - PG[2] * cache[567 + i + 1] + 7 * half_xi_ * (cache[458 + i] - cache[458 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[660 + i] = PA[0] * cache[570 + i] - PG[0] * cache[570 + i + 1] + 8 * half_xi_ * (cache[462 + i] - cache[462 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[661 + i] = PA[1] * cache[570 + i] - PG[1] * cache[570 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[662 + i] = PA[2] * cache[570 + i] - PG[2] * cache[570 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[663 + i] = PA[0] * cache[576 + i] - PG[0] * cache[576 + i + 1] + 6 * half_xi_ * (cache[471 + i] - cache[471 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[664 + i] = PA[1] * cache[574 + i] - PG[1] * cache[574 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[665 + i] = PA[0] * cache[580 + i] - PG[0] * cache[580 + i + 1] + 6 * half_xi_ * (cache[477 + i] - cache[477 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[666 + i] = PA[0] * cache[582 + i] - PG[0] * cache[582 + i + 1] + 5 * half_xi_ * (cache[480 + i] - cache[480 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[667 + i] = PA[2] * cache[576 + i] - PG[2] * cache[576 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[668 + i] = PA[1] * cache[580 + i] - PG[1] * cache[580 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[669 + i] = PA[0] * cache[588 + i] - PG[0] * cache[588 + i + 1] + 5 * half_xi_ * (cache[489 + i] - cache[489 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[670 + i] = PA[0] * cache[590 + i] - PG[0] * cache[590 + i + 1] + 4 * half_xi_ * (cache[492 + i] - cache[492 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[671 + i] = PA[2] * cache[582 + i] - PG[2] * cache[582 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[672 + i] = PA[0] * cache[594 + i] - PG[0] * cache[594 + i + 1] + 4 * half_xi_ * (cache[498 + i] - cache[498 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[673 + i] = PA[1] * cache[588 + i] - PG[1] * cache[588 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[674 + i] = PA[0] * cache[598 + i] - PG[0] * cache[598 + i + 1] + 4 * half_xi_ * (cache[504 + i] - cache[504 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[675 + i] = PA[0] * cache[600 + i] - PG[0] * cache[600 + i + 1] + 3 * half_xi_ * (cache[507 + i] - cache[507 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[676 + i] = PA[2] * cache[590 + i] - PG[2] * cache[590 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[677 + i] = PA[0] * cache[604 + i] - PG[0] * cache[604 + i + 1] + 3 * half_xi_ * (cache[513 + i] - cache[513 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[678 + i] = PA[0] * cache[606 + i] - PG[0] * cache[606 + i + 1] + 3 * half_xi_ * (cache[516 + i] - cache[516 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[679 + i] = PA[1] * cache[598 + i] - PG[1] * cache[598 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[680 + i] = PA[0] * cache[610 + i] - PG[0] * cache[610 + i + 1] + 3 * half_xi_ * (cache[522 + i] - cache[522 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[681 + i] = PA[0] * cache[612 + i] - PG[0] * cache[612 + i + 1] + 2 * half_xi_ * (cache[525 + i] - cache[525 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[682 + i] = PA[2] * cache[600 + i] - PG[2] * cache[600 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[683 + i] = PA[0] * cache[616 + i] - PG[0] * cache[616 + i + 1] + 2 * half_xi_ * (cache[531 + i] - cache[531 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[684 + i] = PA[0] * cache[618 + i] - PG[0] * cache[618 + i + 1] + 2 * half_xi_ * (cache[534 + i] - cache[534 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[685 + i] = PA[0] * cache[620 + i] - PG[0] * cache[620 + i + 1] + 2 * half_xi_ * (cache[537 + i] - cache[537 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[686 + i] = PA[1] * cache[610 + i] - PG[1] * cache[610 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[687 + i] = PA[0] * cache[624 + i] - PG[0] * cache[624 + i + 1] + 2 * half_xi_ * (cache[543 + i] - cache[543 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[688 + i] = PA[0] * cache[626 + i] - PG[0] * cache[626 + i + 1] + 1 * half_xi_ * (cache[546 + i] - cache[546 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[689 + i] = PA[2] * cache[612 + i] - PG[2] * cache[612 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[690 + i] = PA[0] * cache[630 + i] - PG[0] * cache[630 + i + 1] + 1 * half_xi_ * (cache[552 + i] - cache[552 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[691 + i] = PA[0] * cache[632 + i] - PG[0] * cache[632 + i + 1] + 1 * half_xi_ * (cache[555 + i] - cache[555 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[692 + i] = PA[0] * cache[634 + i] - PG[0] * cache[634 + i + 1] + 1 * half_xi_ * (cache[558 + i] - cache[558 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[693 + i] = PA[0] * cache[636 + i] - PG[0] * cache[636 + i + 1] + 1 * half_xi_ * (cache[561 + i] - cache[561 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[694 + i] = PA[1] * cache[624 + i] - PG[1] * cache[624 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[695 + i] = PA[0] * cache[640 + i] - PG[0] * cache[640 + i + 1] + 1 * half_xi_ * (cache[567 + i] - cache[567 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[696 + i] = PA[0] * cache[642 + i] - PG[0] * cache[642 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[697 + i] = PA[0] * cache[644 + i] - PG[0] * cache[644 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[698 + i] = PA[0] * cache[646 + i] - PG[0] * cache[646 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[699 + i] = PA[0] * cache[648 + i] - PG[0] * cache[648 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[700 + i] = PA[0] * cache[650 + i] - PG[0] * cache[650 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[701 + i] = PA[0] * cache[652 + i] - PG[0] * cache[652 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[702 + i] = PA[0] * cache[654 + i] - PG[0] * cache[654 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[703 + i] = PA[0] * cache[656 + i] - PG[0] * cache[656 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[704 + i] = PA[0] * cache[658 + i] - PG[0] * cache[658 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[705 + i] = PA[1] * cache[642 + i] - PG[1] * cache[642 + i + 1] + 8 * half_xi_ * (cache[546 + i] - cache[546 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[706 + i] = PA[2] * cache[642 + i] - PG[2] * cache[642 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[707 + i] = PA[1] * cache[646 + i] - PG[1] * cache[646 + i + 1] + 6 * half_xi_ * (cache[552 + i] - cache[552 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[708 + i] = PA[1] * cache[648 + i] - PG[1] * cache[648 + i + 1] + 5 * half_xi_ * (cache[555 + i] - cache[555 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[709 + i] = PA[1] * cache[650 + i] - PG[1] * cache[650 + i + 1] + 4 * half_xi_ * (cache[558 + i] - cache[558 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[710 + i] = PA[1] * cache[652 + i] - PG[1] * cache[652 + i + 1] + 3 * half_xi_ * (cache[561 + i] - cache[561 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[711 + i] = PA[1] * cache[654 + i] - PG[1] * cache[654 + i + 1] + 2 * half_xi_ * (cache[564 + i] - cache[564 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[712 + i] = PA[1] * cache[656 + i] - PG[1] * cache[656 + i + 1] + 1 * half_xi_ * (cache[567 + i] - cache[567 + i + 1]);
    for (int i = 0; i < 1; i++)
        cache[713 + i] = PA[1] * cache[658 + i] - PG[1] * cache[658 + i + 1];
    for (int i = 0; i < 1; i++)
        cache[714 + i] = PA[2] * cache[658 + i] - PG[2] * cache[658 + i + 1] + 8 * half_xi_ * (cache[567 + i] - cache[567 + i + 1]);
    buf[0] += cache[245];
    buf[1] += cache[250];
    buf[2] += cache[255];
    buf[3] += cache[260];
    buf[4] += cache[265];
    buf[5] += cache[270];
    buf[6] += cache[275];
    buf[7] += cache[280];
    buf[8] += cache[285];
    buf[9] += cache[290];
    buf[10] += cache[295];
    buf[11] += cache[300];
    buf[12] += cache[305];
    buf[13] += cache[310];
    buf[14] += cache[315];
    buf[15] += cache[320];
    buf[16] += cache[325];
    buf[17] += cache[330];
    buf[18] += cache[335];
    buf[19] += cache[340];
    buf[20] += cache[345];
    buf[21] += cache[350];
    buf[22] += cache[354];
    buf[23] += cache[358];
    buf[24] += cache[362];
    buf[25] += cache[366];
    buf[26] += cache[370];
    buf[27] += cache[374];
    buf[28] += cache[378];
    buf[29] += cache[382];
    buf[30] += cache[386];
    buf[31] += cache[390];
    buf[32] += cache[394];
    buf[33] += cache[398];
    buf[34] += cache[402];
    buf[35] += cache[406];
    buf[36] += cache[410];
    buf[37] += cache[414];
    buf[38] += cache[418];
    buf[39] += cache[422];
    buf[40] += cache[426];
    buf[41] += cache[430];
    buf[42] += cache[434];
    buf[43] += cache[438];
    buf[44] += cache[442];
    buf[45] += cache[446];
    buf[46] += cache[450];
    buf[47] += cache[454];
    buf[48] += cache[458];
    buf[49] += cache[462];
    buf[50] += cache[465];
    buf[51] += cache[468];
    buf[52] += cache[471];
    buf[53] += cache[474];
    buf[54] += cache[477];
    buf[55] += cache[480];
    buf[56] += cache[483];
    buf[57] += cache[486];
    buf[58] += cache[489];
    buf[59] += cache[492];
    buf[60] += cache[495];
    buf[61] += cache[498];
    buf[62] += cache[501];
    buf[63] += cache[504];
    buf[64] += cache[507];
    buf[65] += cache[510];
    buf[66] += cache[513];
    buf[67] += cache[516];
    buf[68] += cache[519];
    buf[69] += cache[522];
    buf[70] += cache[525];
    buf[71] += cache[528];
    buf[72] += cache[531];
    buf[73] += cache[534];
    buf[74] += cache[537];
    buf[75] += cache[540];
    buf[76] += cache[543];
    buf[77] += cache[546];
    buf[78] += cache[549];
    buf[79] += cache[552];
    buf[80] += cache[555];
    buf[81] += cache[558];
    buf[82] += cache[561];
    buf[83] += cache[564];
    buf[84] += cache[567];
    buf[85] += cache[570];
    buf[86] += cache[572];
    buf[87] += cache[574];
    buf[88] += cache[576];
    buf[89] += cache[578];
    buf[90] += cache[580];
    buf[91] += cache[582];
    buf[92] += cache[584];
    buf[93] += cache[586];
    buf[94] += cache[588];
    buf[95] += cache[590];
    buf[96] += cache[592];
    buf[97] += cache[594];
    buf[98] += cache[596];
    buf[99] += cache[598];
    buf[100] += cache[600];
    buf[101] += cache[602];
    buf[102] += cache[604];
    buf[103] += cache[606];
    buf[104] += cache[608];
    buf[105] += cache[610];
    buf[106] += cache[612];
    buf[107] += cache[614];
    buf[108] += cache[616];
    buf[109] += cache[618];
    buf[110] += cache[620];
    buf[111] += cache[622];
    buf[112] += cache[624];
    buf[113] += cache[626];
    buf[114] += cache[628];
    buf[115] += cache[630];
    buf[116] += cache[632];
    buf[117] += cache[634];
    buf[118] += cache[636];
    buf[119] += cache[638];
    buf[120] += cache[640];
    buf[121] += cache[642];
    buf[122] += cache[644];
    buf[123] += cache[646];
    buf[124] += cache[648];
    buf[125] += cache[650];
    buf[126] += cache[652];
    buf[127] += cache[654];
    buf[128] += cache[656];
    buf[129] += cache[658];
    buf[130] += cache[660];
    buf[131] += cache[661];
    buf[132] += cache[662];
    buf[133] += cache[663];
    buf[134] += cache[664];
    buf[135] += cache[665];
    buf[136] += cache[666];
    buf[137] += cache[667];
    buf[138] += cache[668];
    buf[139] += cache[669];
    buf[140] += cache[670];
    buf[141] += cache[671];
    buf[142] += cache[672];
    buf[143] += cache[673];
    buf[144] += cache[674];
    buf[145] += cache[675];
    buf[146] += cache[676];
    buf[147] += cache[677];
    buf[148] += cache[678];
    buf[149] += cache[679];
    buf[150] += cache[680];
    buf[151] += cache[681];
    buf[152] += cache[682];
    buf[153] += cache[683];
    buf[154] += cache[684];
    buf[155] += cache[685];
    buf[156] += cache[686];
    buf[157] += cache[687];
    buf[158] += cache[688];
    buf[159] += cache[689];
    buf[160] += cache[690];
    buf[161] += cache[691];
    buf[162] += cache[692];
    buf[163] += cache[693];
    buf[164] += cache[694];
    buf[165] += cache[695];
    buf[166] += cache[696];
    buf[167] += cache[697];
    buf[168] += cache[698];
    buf[169] += cache[699];
    buf[170] += cache[700];
    buf[171] += cache[701];
    buf[172] += cache[702];
    buf[173] += cache[703];
    buf[174] += cache[704];
    buf[175] += cache[705];
    buf[176] += cache[706];
    buf[177] += cache[707];
    buf[178] += cache[708];
    buf[179] += cache[709];
    buf[180] += cache[710];
    buf[181] += cache[711];
    buf[182] += cache[712];
    buf[183] += cache[713];
    buf[184] += cache[714];
    return;
}

static int buf_len_list[] =
    {1, 0, 0, 0, 0, 0,
     3, 9, 0, 0, 0, 0,
     6, 16, 31, 0, 0, 0,
     10, 25, 46, 74, 0, 0,
     15, 36, 64, 100, 145, 0,
     21, 49, 85, 130, 185, 251};

typedef void (*NUC_VRR)(double *buf, double *cache, double coe, double K_ab, double half_xi_, double xi, const double *PA,
                        const double *PG, double PG_dist2, double sr_frac, double lr_frac, double omega_);
static NUC_VRR nuc_vrr_list[] = {nuc_VRR_0_0, NULL, NULL, NULL, NULL,
                                 nuc_VRR_1_0, nuc_VRR_1_1, NULL, NULL, NULL,
                                 nuc_VRR_2_0, nuc_VRR_2_1, nuc_VRR_2_2, NULL, NULL,
                                 nuc_VRR_3_0, nuc_VRR_3_1, nuc_VRR_3_2, nuc_VRR_3_3, NULL,
                                 nuc_VRR_4_0, nuc_VRR_4_1, nuc_VRR_4_2, nuc_VRR_4_3, nuc_VRR_4_4,
                                 nuc_VRR_5_0, nuc_VRR_5_1, nuc_VRR_5_2, nuc_VRR_5_3, nuc_VRR_5_4};

void xint_3c1e(Vector buf, const int shlij, const double G[3], xint_info xint)
{
    const mol_info mol = xint->mol;
    const pair_info pair = mol->pair;
    const shell_pair shp = &pair->shp[shlij];
    const Vector coei = get_bas_coe(shp->shli, mol->bas);
    const Vector coej = get_bas_coe(shp->shlj, mol->bas);
    double sr_frac = xint->sr_frac;
    double lr_frac = xint->lr_frac;
    double omega_ = xint->omega;
    int angi = shp->li;
    int angj = shp->lj;
    int di = shp->di;
    int dj = shp->dj;
    double PG[3], PG_dist2;
    NUC_VRR vrr_func = nuc_vrr_list[angj + 5 * angi];
    Vector hrr_vec = &xint->cache[0];
    Vector vrr_vec = &xint->cache[1024];
    int hrr_len = buf_len_list[angi * 6 +  angj];
    memset(hrr_vec, 0, sizeof(double) * hrr_len);
    for (int i = 0, uu = 0; i < di; i++)
    {
        for (int j = 0; j < dj; j++, uu++)
        {
            PG[0] = shp->P[3 * uu + 0] - G[0];
            PG[1] = shp->P[3 * uu + 1] - G[1];
            PG[2] = shp->P[3 * uu + 2] - G[2];
            PG_dist2 = PG[0] * PG[0] + PG[1] * PG[1] + PG[2] * PG[2];
            vrr_func(hrr_vec, vrr_vec, coei[i] * coej[j], shp->K_ab[4 * uu + 3], shp->half_xi_[uu], shp->xi[uu], &shp->PA[3 * uu],
                     PG, PG_dist2, sr_frac, lr_frac, omega_);
        }
    }
    xint_hrr(buf, xint->cache, angi, angj, shp->AB);
    return;
}

void xint_3c1e_prm(Vector buf, const int shlij, const double G[3], xint_info xint)
{
    const mol_info mol = xint->mol;
    const pair_info pair = mol->prm_pair;
    const shell_pair shp = &pair->shp[shlij];
    double sr_frac = xint->sr_frac;
    double lr_frac = xint->lr_frac;
    double omega_ = xint->omega;
    int angi = shp->li;
    int angj = shp->lj;
    int di = shp->di;
    int dj = shp->dj;
    double PG[3], PG_dist2;
    NUC_VRR vrr_func = nuc_vrr_list[angj + 5 * angi];
    Vector hrr_vec = &xint->cache[0];
    Vector vrr_vec = &xint->cache[1024];
    int hrr_len = buf_len_list[angj + 6 * angi];
    memset(hrr_vec, 0, sizeof(double) * hrr_len);
    PG[0] = shp->P[3 + 0] - G[0];
    PG[1] = shp->P[3 + 1] - G[1];
    PG[2] = shp->P[3 + 2] - G[2];
    PG_dist2 = PG[0] * PG[0] + PG[1] * PG[1] + PG[2] * PG[2];
    vrr_func(hrr_vec, vrr_vec, 1, shp->K_ab[4], shp->half_xi_[0], shp->xi[0], shp->PA,
             PG, PG_dist2, sr_frac, lr_frac, omega_);
    xint_hrr(buf, xint->cache, angi, angj, shp->AB);
    return;
}

extern void xint_ne(Vector buf, const int shlij, xint_info xint)
{
    const mol_info mol = xint->mol;
    const shell_pair shp = &mol->pair->shp[shlij];
    int natm = get_mol_natm(mol);
    int li = shp->li;
    int lj = shp->lj;
    int dij = 0.25 * (li + 1) * (li + 2) * (lj + 1) * (lj + 2);
    int charge;
    memset(buf, 0, sizeof(double) * dij);
    double *buf_tmp = (Vector)malloc(sizeof(double) * dij);
    for (int i = 0; i < natm; i++)
    {
        charge = mol->atm(ELEMENT_VAL, i);
        if (charge == 0)
            continue;
        xint_3c1e(buf_tmp, shlij, get_atm_coord(i, mol->atm), xint);
        for (int j = 0; j < dij; j++)
            buf[j] -= buf_tmp[j] * charge;
    }
    free(buf_tmp);
    return;
}

void cal_nucleus_matrix(Matrix v_matrix, const mol_info mol)
{
    int msize = get_mol_msize(mol);
    const bas_info bas = mol->bas;
    const pair_info pair = mol->pair;
    double buf[1024] = {0};
    int shli, shlj;
    int pi, pj;
    int di, dj;
    int ci, cj;
    xint_info xint;
    xint = init_xint(mol);
    for (int i = 0; i < pair->len; i++)
    {
        xint_ne(buf, i, xint);
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
                v_matrix[u + pi][v + pj] = buf[uu];
                v_matrix[v + pj][u + pi] = buf[uu];
            }
    }
    del_xint(xint);
    return;
}
