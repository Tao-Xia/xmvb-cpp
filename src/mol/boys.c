#include "mol/xint.h"
#define BOYS_LEN 4096
typedef struct
{
    double **Chebyshev_coe;
    int mmax, point_num;
    double delta, delta_;
    double *twoi, *halfi;
} boys_info;
static boys_info* GlobalBoys;
static double a0[] = {0.213271302431420E+0, 0.629344460255614E-1, 0.769838037756759E-2, 0.758433197127160E-3, 0.564691197633667E-4};
static double b0[] = {0.879937801660182E+0, 0.338450368470103E+0, 0.738522953299624E-1, 0.101431553402629E-1, 0.955528842975585E-3, 0.720266520392572E-4};
static double a1[] = {0.295195994716045E-1, 0.128790985465415e-1, 0.998165499553218E-3, 0.970927983276419E-4, 0.493839847029699E-5};
static double b1[] = {0.461403194579124E+0, 0.108494164372449E+0, 0.171462934845042E-1, 0.196918657845508E-2, 0.160138863265254E-3, 0.857708713007233E-5};
static double a2[] = {-0.575763488635418E-2, 0.731474973333076E-2, 0.251276149443393E-3, 0.264336244559094E-4};
static double b2[] = {0.274754154712841E+0, 0.425364830353043E-1, 0.493902790955943E-2, 0.437251500927601E-3, 0.288914662393981E-4};
static double a3[] = {-0.290110430424666E-1, 0.561884370781462E-2, 0.301628267382713E-4, 0.110671035361856E-4};
static double b3[] = {0.171637608242892E+0, 0.187571417256877E-1, 0.178536829675118E-2, 0.137360778130936E-3, 0.791915206883054E-5};
static double a4[] = {-0.452693111179624E-1, 0.490070062899003E-2, -0.561789719979307E-4, 0.550814626951998E-5};
static double b4[] = {0.108051989937231E+0, 0.855924943430755E-2, 0.724968571389473E-3, 0.502338223156067E-4, 0.249107837399141E-5};
static double a5[] = {-0.566143259316101E-1, 0.455916894577203E-2, -0.894152721395639E-4, 0.328096732308082E-5};
static double b5[] = {0.662932958471386E-1, 0.383724443872493E-2, 0.327167659811839E-3, 0.210430437682548E-4, 0.883562935089333E-6};
static double a6[] = {-0.503249167534352E-1, 0.273135625430953E-2, -0.310733624819100E-4};
static double b6[] = {0.586609328033371E-1, 0.194044691497128E-2, 0.109442742502192e-3, 0.613406236401726E-5};
static double a7[] = {-0.548201062615785E-1, 0.253099908233175E-2, -0.333589469427863E-4};
static double b7[] = {0.389873128779298E-1, 0.569890860832729E-3, 0.422187129333708E-4, 0.286010059144633E-5};
static double a8[] = {-0.581618006078160E-1, 0.238525529084601E-2, -0.329989020317093E-4};
static double b8[] = {0.240929282666615E-1, -0.202677647499956E-3, 0.119820675974460E-4, 0.145762086904409E-5};
static double a9[] = {-0.334843993901400E-1, 0.846637494147059E-3};
static double b9[] = {0.495875606944471E-1, 0.946642302340943E-3, 0.108367772249790E-4};
static double a10[] = {-0.335292171805959E-1, 0.749168957990503E-3};
static double b10[] = {0.421492932021683E-1, 0.582840985360327E-3, 0.237676790577455E-5};
static double a11[] = {-0.332669773770348E-1, 0.668720489602687E-3};
static double b11[] = {0.363057685289467E-1, 0.345646100984643E-3, -0.190872330373450E-5};
static double a12[] = {-0.326241966410798E-1, 0.598705175467956E-3};
static double b12[] = {0.318680048277695E-1, 0.202419662347765E-3, -0.362095173837973E-5};
static double a13[] = {-0.317754368014894E-1, 0.537678595933584E-2};
static double b13[] = {0.284036027081815E-1, 0.113673420662576E-3, -0.416076810552774E-5};
static double *a_list[] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13};
static double *b_list[] = {b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13};
static int n_list[] = {5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2};
static double max_z[] = {16.3578, 17.4646, 15.2368, 16.0419, 16.8955, 17.7822, 15.8077, 16.5903, 17.3336, 15.6602, 16.5258, 17.5395, 18.5783, 19.6511};
static double a_0[] = {1.0, 0.4807498567691362, 0.5253055608807534, 0.5735131987446477, 0.613685849032916, 0.6466300385008377, 0.6739444475794731, 0.6969278698605402,
                       0.7165414254774158, 0.7334902710845949, 0.7482976009670683, 0.7613579445345817, 0.7729738455277437, 0.7833810369372723};
static double eval(double T, int m, double absolute_precision)
{
    double denom = (m + 0.5);
    double term = 0.5 * exp(-T) / denom;
    double sum = term;
    double epsilon;
    const double relative_zero = 1e-15;
    const double absolute_precision_o_10 = absolute_precision * 0.1;
    do
    {
        denom += 1.0;
        term *= T / denom;
        sum += term;
        epsilon = fmax(absolute_precision_o_10, sum * relative_zero);
    } while (term > epsilon);
    return sum;
}
static int make_ccoe(double x0, double x2, double y0, double y2, int m, double *cc)
{
    double delta = (x2 - x0) * 0.5;
    double delta2_ = 0.5 / (delta * delta);
    double x1 = x0 + delta;
    double y1 = eval(x1, m, 1E-20);
    cc[0] = (x1 * x2 * y0 - x0 * x2 * y1 * 2 + x0 * x1 * y2) * delta2_;
    cc[1] = -((x1 + x2) * y0 - (x0 + x2) * y1 * 2 + (x0 + x1) * y2) * delta2_;
    cc[2] = (y0 - y1 * 2 + y2) * delta2_;
    return 0;
}
extern void init_boys()
{
    double a, b;
    double tmp_fm[BOYS_LEN + 1];
    GlobalBoys = (boys_info*)malloc(sizeof(boys_info));
    GlobalBoys->point_num = BOYS_LEN; /// refer from libint
    GlobalBoys->mmax = 20;
    GlobalBoys->delta = 30. / (GlobalBoys->point_num - 1);
    GlobalBoys->delta_ = 1. / GlobalBoys->delta;
    GlobalBoys->Chebyshev_coe = (Matrix)malloc(sizeof(double *) * (1 + 20));
    GlobalBoys->Chebyshev_coe[0] = (double *)malloc(sizeof(double) * (1 + 20) * 3 * GlobalBoys->point_num);
    GlobalBoys->twoi = (double *)malloc(sizeof(double) * (1 + 20));
    GlobalBoys->halfi = (double *)malloc(sizeof(double) * (1 + 20));
    for (int i = 0; i <= 20; i++)
    {
        GlobalBoys->twoi[i] = 1. / (2. * i + 1);
        GlobalBoys->halfi[i] = (2 * i - 1.) * 0.5;
        for (int j = 0; j <= BOYS_LEN; j++)
            tmp_fm[j] = eval(j * GlobalBoys->delta, i, 1E-20);
        if (i > 0)
            GlobalBoys->Chebyshev_coe[i] = &GlobalBoys->Chebyshev_coe[i - 1][3 * GlobalBoys->point_num];
        for (int j = 0; j < GlobalBoys->point_num; j++)
        {
            a = j * GlobalBoys->delta;
            b = a + GlobalBoys->delta;
            make_ccoe(a, b, tmp_fm[j], tmp_fm[j + 1], i, &GlobalBoys->Chebyshev_coe[i][j * 3]);
        }
    }
    return;
}
extern void del_boys()
{
    free(GlobalBoys->Chebyshev_coe[0]);
    free(GlobalBoys->Chebyshev_coe);
    free(GlobalBoys->twoi);
    free(GlobalBoys->halfi);
    free(GlobalBoys);
    return;
}
extern void cal_boys(int m, double x, Vector result)
{
    if (x < 30.)
    {
        double *d = GlobalBoys->Chebyshev_coe[m];
        double xd = x * GlobalBoys->delta_;
        const int iv = xd;
        int ofs = iv * 3;
        result[m] = d[ofs] + x * (d[ofs + 1] + x * d[ofs + 2]);
        if (m > 0)
        {
            const double x2 = 2.0 * x;
            const double exp_x = exp(-x);
            for (int i = m - 1; i >= 0; i--)
                result[i] = (result[i + 1] * x2 + exp_x) * GlobalBoys->twoi[i];
        }
    }
    else
    {
        double x_ = 1 / x;
        result[0] = 0.88622692545275801365 * sqrt(x_);
        for (int i = 1; i <= m; i++)
            result[i] = result[i - 1] * x_ * GlobalBoys->halfi[i];
    }
    return;
}
