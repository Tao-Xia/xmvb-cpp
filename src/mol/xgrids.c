#include "mol/xgrids.h"
#include "mol/xint.h"
static grid_t gd6[6];
static grid_t gd14[14];
static grid_t gd26[26];
static grid_t gd38[38];
static grid_t gd50[50];
static grid_t gd74[86];
static grid_t gd86[86];
static grid_t gd110[110];
static grid_t gd146[146];
static grid_t gd170[170];
static grid_t gd194[194];
static grid_t gd230[230];
static grid_t gd266[266];
static grid_t gd302[302];
static grid_t gd350[350];
static grid_t gd434[434];
static grid_t gd590[590];
static grid_t gd770[770];
static grid_t gd974[974];
static grid_t gd1202[1202];
static grid_t gd1454[1454];
static grid_t gd1730[1730];
static grid_t gd2030[2030];
static grid_t gd2354[2354];
static grid_t gd2702[2702];
static grid_t gd3074[3074];
static grid_t gd3470[3470];
static grid_t gd3890[3890];
static grid_t gd4334[4334];
static grid_t gd4802[4802];
static grid_t gd5294[5294];
static grid_t gd5810[5810];
static grid_t *gd_list[33];
static int grid_order[] = {3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41,
						   47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131};
static int grid_num_list[] = {6, 14, 26, 38, 50, 86, 86, 110, 146, 170, 194, 230,
							  266, 302, 350, 434, 590, 770, 974, 1202, 1454, 1730, 2030, 2354, 2702, 30074, 3470,
							  3890, 4334, 4802, 5294, 5810};
static int cosx_init[] = {11, 35, 45, 55, 65, 1};
static int cosx_final[] = {19, 45, 55, 65, 65, 1};
static int xcoarse[] = {23, 21, 42, 75, 84, 0};
static int coarse[] = {29, 35, 70, 95, 104, 0};
static int medium[] = {41, 49, 88, 112, 123, 0};
static int fine[] = {47, 70, 123, 130, 141, 0};
static int cosj[] = {35, 40, 55, 85, 94, 0};
static int c_coarse[] = {35, 40, 55, 85, 94, 0};
static double Becke_step(double x);
static double normal_weight_grid(int n, double *p, double *dist, const atm_info atm, int *nn, int nn_num);
static double get_prod(int n, double *p, double *dist, double *dist_g, const atm_info atm, int *nn, int nn_num);
static int get_radii_grid(int order, int Z, double *radii_r, double *radii_w);
static grid_t *get_angular(int order);
static grid_t *get_mol_grid(GRIDS_TYPE type, const atm_info atm, int *len);
static double get_radii_atm(int Z);
static int get_atm_grid(int angular_order, int radii_order,
						int a, double *dist, const atm_info atm, grid_t *grids, int if_prun, int *nn, int nn_num);
static void make_point(double x, double y, double z, double w, grid_t *g)
{
	g->w = w;
	g->x = x;
	g->y = y;
	g->z = z;
	return;
}
static int add_point_1(double v, grid_t *data)
{
	double w = 4. * 3.141592654 * v;
	make_point(1., 0., 0., w, &data[0]);
	make_point(-1., 0., 0., w, &data[1]);
	make_point(0., 1., 0., w, &data[2]);
	make_point(0., -1., 0., w, &data[3]);
	make_point(0., 0., 1., w, &data[4]);
	make_point(0., 0., -1., w, &data[5]);
	return 6;
}
static int add_point_2(double v, grid_t *data)
{
	double w = 4. * 3.141592654 * v;
	double a = sqrt(0.5);
	make_point(0., a, a, w, &data[0]);
	make_point(0., -a, a, w, &data[1]);
	make_point(0., a, -a, w, &data[2]);
	make_point(0., -a, -a, w, &data[3]);
	make_point(a, 0., a, w, &data[4]);
	make_point(-a, 0., a, w, &data[5]);
	make_point(a, 0., -a, w, &data[6]);
	make_point(-a, 0., -a, w, &data[7]);
	make_point(a, a, 0., w, &data[8]);
	make_point(-a, a, 0., w, &data[9]);
	make_point(a, -a, 0., w, &data[10]);
	make_point(-a, -a, 0., w, &data[11]);
	return 12;
}
static int add_point_3(double v, grid_t *data)
{
	double w = 4. * 3.141592654 * v;
	double a = sqrt(1. / 3.);
	make_point(a, a, a, w, &data[0]);
	make_point(-a, a, a, w, &data[1]);
	make_point(a, -a, a, w, &data[2]);
	make_point(a, a, -a, w, &data[3]);
	make_point(-a, -a, a, w, &data[4]);
	make_point(-a, a, -a, w, &data[5]);
	make_point(a, -a, -a, w, &data[6]);
	make_point(-a, -a, -a, w, &data[7]);
	return 8;
}
static int add_point_4(double v, double a, grid_t *data)
{
	double w = 4. * 3.141592654 * v;
	double b = sqrt(1. - 2 * a * a);
	make_point(a, a, b, w, &data[0]);
	make_point(-a, a, b, w, &data[1]);
	make_point(a, -a, b, w, &data[2]);
	make_point(a, a, -b, w, &data[3]);
	make_point(-a, -a, b, w, &data[4]);
	make_point(-a, a, -b, w, &data[5]);
	make_point(a, -a, -b, w, &data[6]);
	make_point(-a, -a, -b, w, &data[7]);
	make_point(a, b, a, w, &data[8]);
	make_point(-a, b, a, w, &data[9]);
	make_point(a, -b, a, w, &data[10]);
	make_point(a, b, -a, w, &data[11]);
	make_point(-a, -b, a, w, &data[12]);
	make_point(-a, b, -a, w, &data[13]);
	make_point(a, -b, -a, w, &data[14]);
	make_point(-a, -b, -a, w, &data[15]);
	make_point(b, a, a, w, &data[16]);
	make_point(-b, a, a, w, &data[17]);
	make_point(b, -a, a, w, &data[18]);
	make_point(b, a, -a, w, &data[19]);
	make_point(-b, -a, a, w, &data[20]);
	make_point(-b, a, -a, w, &data[21]);
	make_point(b, -a, -a, w, &data[22]);
	make_point(-b, -a, -a, w, &data[23]);
	return 24;
}
static int add_point_5(double v, double a, grid_t *data)
{
	double w = 4. * 3.141592654 * v;
	double b = sqrt(1. - a * a);
	make_point(a, b, 0., w, &data[0]);
	make_point(-a, b, 0., w, &data[1]);
	make_point(a, -b, 0., w, &data[2]);
	make_point(-a, -b, 0., w, &data[3]);
	make_point(b, a, 0., w, &data[4]);
	make_point(-b, a, 0., w, &data[5]);
	make_point(b, -a, 0., w, &data[6]);
	make_point(-b, -a, 0., w, &data[7]);
	make_point(a, 0., b, w, &data[8]);
	make_point(-a, 0., b, w, &data[9]);
	make_point(a, 0., -b, w, &data[10]);
	make_point(-a, 0., -b, w, &data[11]);
	make_point(b, 0., a, w, &data[12]);
	make_point(-b, 0., a, w, &data[13]);
	make_point(b, 0., -a, w, &data[14]);
	make_point(-b, 0., -a, w, &data[15]);
	make_point(0., a, b, w, &data[16]);
	make_point(0., -a, b, w, &data[17]);
	make_point(0., a, -b, w, &data[18]);
	make_point(0., -a, -b, w, &data[19]);
	make_point(0., b, a, w, &data[20]);
	make_point(0., -b, a, w, &data[21]);
	make_point(0., b, -a, w, &data[22]);
	make_point(0., -b, -a, w, &data[23]);
	return 24;
}
static int add_point_6(double v, double a, double b, grid_t *data)
{
	double w = 4. * 3.141592654 * v;
	double c = sqrt(1. - a * a - b * b);
	make_point(a, b, c, w, &data[0]);
	make_point(-a, b, c, w, &data[1]);
	make_point(a, -b, c, w, &data[2]);
	make_point(a, b, -c, w, &data[3]);
	make_point(-a, -b, c, w, &data[4]);
	make_point(-a, b, -c, w, &data[5]);
	make_point(a, -b, -c, w, &data[6]);
	make_point(-a, -b, -c, w, &data[7]);
	make_point(a, c, b, w, &data[8]);
	make_point(-a, c, b, w, &data[9]);
	make_point(a, -c, b, w, &data[10]);
	make_point(a, c, -b, w, &data[11]);
	make_point(-a, -c, b, w, &data[12]);
	make_point(-a, c, -b, w, &data[13]);
	make_point(a, -c, -b, w, &data[14]);
	make_point(-a, -c, -b, w, &data[15]);
	make_point(b, a, c, w, &data[16]);
	make_point(-b, a, c, w, &data[17]);
	make_point(b, -a, c, w, &data[18]);
	make_point(b, a, -c, w, &data[19]);
	make_point(-b, -a, c, w, &data[20]);
	make_point(-b, a, -c, w, &data[21]);
	make_point(b, -a, -c, w, &data[22]);
	make_point(-b, -a, -c, w, &data[23]);
	make_point(b, c, a, w, &data[24]);
	make_point(-b, c, a, w, &data[25]);
	make_point(b, -c, a, w, &data[26]);
	make_point(b, c, -a, w, &data[27]);
	make_point(-b, -c, a, w, &data[28]);
	make_point(-b, c, -a, w, &data[29]);
	make_point(b, -c, -a, w, &data[30]);
	make_point(-b, -c, -a, w, &data[31]);
	make_point(c, a, b, w, &data[32]);
	make_point(-c, a, b, w, &data[33]);
	make_point(c, -a, b, w, &data[34]);
	make_point(c, a, -b, w, &data[35]);
	make_point(-c, -a, b, w, &data[36]);
	make_point(-c, a, -b, w, &data[37]);
	make_point(c, -a, -b, w, &data[38]);
	make_point(-c, -a, -b, w, &data[39]);
	make_point(c, b, a, w, &data[40]);
	make_point(-c, b, a, w, &data[41]);
	make_point(c, -b, a, w, &data[42]);
	make_point(c, b, -a, w, &data[43]);
	make_point(-c, -b, a, w, &data[44]);
	make_point(-c, b, -a, w, &data[45]);
	make_point(c, -b, -a, w, &data[46]);
	make_point(-c, -b, -a, w, &data[47]);
	return 48;
}
static void make_6_grid()
{
	grid_t *data = &gd6[0];
	gd_list[0] = &gd6[0];
	data += add_point_1(1 / 6., data);
}

static void make_14_grid()
{
	grid_t *data = &gd14[0];
	gd_list[1] = &gd14[0];
	data += add_point_1(1 / 15., data);
	data += add_point_3(3 / 40., data);
}

static void make_26_grid()
{
	grid_t *data = &gd26[0];
	gd_list[2] = &gd26[0];
	data += add_point_1(1 / 21., data);
	data += add_point_2(4 / 105., data);
	data += add_point_3(9 / 280., data);
}

static void make_38_grid()
{
	grid_t *data = &gd38[0];
	gd_list[3] = &gd38[0];
	data += add_point_1(1 / 105., data);
	data += add_point_3(9 / 280., data);
	data += add_point_5(1 / 35., 0.4597008433809831E+0, data);
}

static void make_50_grid()
{
	grid_t *data = &gd50[0];
	gd_list[4] = &gd50[0];
	data += add_point_1(0.1269841269841270E-1, data);
	data += add_point_2(0.2257495590828924E-1, data);
	data += add_point_3(0.2109375000000000E-1, data);
	data += add_point_4(0.2017333553791887E-1, 0.3015113445777636E+0, data);
}

static void make_74_grid()
{
	grid_t *data = &gd74[0];
	gd_list[5] = &gd74[0];
	data += add_point_1(0.1154401154401154E-1, data);
	data += add_point_3(0.1194390908585628E-1, data);
	data += add_point_4(0.1111055571060340E-1, 0.3696028464541502E+0, data);
	data += add_point_4(0.1187650129453714E-1, 0.6943540066026664E+0, data);
	data += add_point_5(0.1181230374690448E-1, 0.3742430390903412E+0, data);
}

static void make_86_grid()
{
	grid_t *data = &gd86[0];
	gd_list[6] = &gd86[0];
	data += add_point_1(0.1154401154401154E-1, data);
	data += add_point_3(0.1194390908585628E-1, data);
	data += add_point_4(0.1111055571060340E-1, 0.3696028464541502E+0, data);
	data += add_point_4(0.1187650129453714E-1, 0.6943540066026664E+0, data);
	data += add_point_5(0.1181230374690448E-1, 0.3742430390903412E+0, data);
}

static void make_110_grid()
{
	grid_t *data = &gd110[0];
	gd_list[7] = &gd110[0];
	data += add_point_1(0.3828270494937162E-2, data);
	data += add_point_3(0.9793737512487512E-2, data);
	data += add_point_4(0.8211737283191111E-2, 0.1851156353447362E+0, data);
	data += add_point_4(0.9942814891178103E-2, 0.6904210483822922E+0, data);
	data += add_point_4(0.9595471336070963E-2, 0.3956894730559419E+0, data);
	data += add_point_5(0.9694996361663028E-2, 0.4783690288121502E+0, data);
}

static void make_146_grid()
{
	grid_t *data = &gd146[0];
	gd_list[8] = &gd146[0];
	data += add_point_1(0.5996313688621381E-3, data);
	data += add_point_2(0.7372999718620756E-2, data);
	data += add_point_3(0.7210515360144488E-2, data);
	data += add_point_4(0.7116355493117555E-2, 0.6764410400114264E+0, data);
	data += add_point_4(0.6753829486314477E-2, 0.4174961227965453E+0, data);
	data += add_point_4(0.7574394159054034E-2, 0.1574676672039082E+0, data);
	data += add_point_6(0.6991087353303262E-2, 0.1403553811713183E+0, 0.4493328323269557E+0, data);
}

static void make_170_grid()
{
	grid_t *data = &gd170[0];
	gd_list[9] = &gd170[0];
	data += add_point_1(0.5544842902037365E-2, data);
	data += add_point_2(0.6071332770670752E-2, data);
	data += add_point_3(0.6383674773515093E-2, data);
	data += add_point_4(0.5183387587747790E-2, 0.2551252621114134E+0, data);
	data += add_point_4(0.6317929009813725E-2, 0.6743601460362766E+0, data);
	data += add_point_4(0.6201670006589077E-2, 0.4318910696719410E+0, data);
	data += add_point_5(0.5477143385137348E-2, 0.2613931360335988E+0, data);
	data += add_point_6(0.5968383987681156E-2, 0.4990453161796037E+0, 0.1446630744325115E+0, data);
}

static void make_194_grid()
{
	grid_t *data = &gd194[0];
	gd_list[10] = &gd194[0];
	data += add_point_1(0.1782340447244611E-2, data);
	data += add_point_2(0.5716905949977102E-2, data);
	data += add_point_3(0.5573383178848738E-2, data);
	data += add_point_4(0.5608704082587997E-2, 0.6712973442695226E+0, data);
	data += add_point_4(0.5158237711805383E-2, 0.2892465627575439E+0, data);
	data += add_point_4(0.5518771467273614E-2, 0.4446933178717437E+0, data);
	data += add_point_4(0.4106777028169394E-2, 0.1299335447650067E+0, data);
	data += add_point_5(0.5051846064614808E-2, 0.3457702197611283E+0, data);
	data += add_point_6(0.5530248916233094E-2, 0.1590417105383530E+0, 0.8360360154824589E+0, data);
}

static void make_230_grid()
{
	grid_t *data = &gd230[0];
	gd_list[11] = &gd230[0];
	data += add_point_1(-0.5522639919727325E-1, data);
	data += add_point_3(0.4450274607445226E-2, data);
	data += add_point_4(0.4496841067921404E-2, 0.4492044687397611E+0, data);
	data += add_point_4(0.5049153450478750E-2, 0.2520419490210201E+0, data);
	data += add_point_4(0.3976408018051883E-2, 0.6981906658447242E+0, data);
	data += add_point_4(0.4401400650381014E-2, 0.6587405243460960E+0, data);
	data += add_point_4(0.1724544350544401E-1, 0.4038544050097660E-1, data);
	data += add_point_5(0.4231083095357343E-2, 0.5823842309715585E+0, data);
	data += add_point_5(0.5198069864064399E-2, 0.3545877390518688E+0, data);
	data += add_point_6(0.4695720972568883E-2, 0.2272181808998187E+0, 0.4864661535886647E+0, data);
}

static void make_266_grid()
{
	grid_t *data = &gd266[0];
	gd_list[12] = &gd266[0];
	data += add_point_1(-0.1313769127326952E-2, data);
	data += add_point_2(-0.2522728704859336E-2, data);
	data += add_point_3(0.4186853881700583E-2, data);
	data += add_point_4(0.5315167977810885E-2, 0.7039373391585475E+0, data);
	data += add_point_4(0.4047142377086219E-2, 0.1012526248572414E+0, data);
	data += add_point_4(0.4112482394406990E-2, 0.4647448726420539E+0, data);
	data += add_point_4(0.3595584899758782E-2, 0.3277420654971629E+0, data);
	data += add_point_4(0.4256131351428158E-2, 0.6620338663699974E+0, data);
	data += add_point_5(0.4229582700647240E-2, 0.8506508083520399E+0, data);
	data += add_point_6(0.4080914225780505E-2, 0.3233484542692899E+0, 0.1153112011009701E+0, data);
	data += add_point_6(0.4071467593830964E-2, 0.2314790158712601E+0, 0.5244939240922365E+0, data);
}

static void make_302_grid()
{
	grid_t *data = &gd302[0];
	gd_list[13] = &gd302[0];
	data += add_point_1(0.8545911725128148E-3, data);
	data += add_point_3(0.3599119285025571E-2, data);
	data += add_point_4(0.3449788424305883E-2, 0.3515640345570105E+0, data);
	data += add_point_4(0.3604822601419882E-2, 0.6566329410219612E+0, data);
	data += add_point_4(0.3576729661743367E-2, 0.4729054132581005E+0, data);
	data += add_point_4(0.2352101413689164E-2, 0.9618308522614784E-1, data);
	data += add_point_4(0.3108953122413675E-2, 0.2219645236294178E+0, data);
	data += add_point_4(0.3650045807677255E-2, 0.7011766416089545E+0, data);
	data += add_point_5(0.2982344963171804E-2, 0.2644152887060663E+0, data);
	data += add_point_5(0.3600820932216460E-2, 0.5718955891878961E+0, data);
	data += add_point_6(0.3571540554273387E-2, 0.2510034751770465E+0, 0.8000727494073952E+0, data);
	data += add_point_6(0.3392312205006170E-2, 0.1233548532583327E+0, 0.4127724083168531E+0, data);
}

static void make_350_grid()
{
	grid_t *data = &gd350[0];
	gd_list[14] = &gd350[0];
	data += add_point_1(0.3006796749453936E-2, data);
	data += add_point_3(0.3050627745650771E-2, data);
	data += add_point_4(0.1621104600288991E-2, 0.7068965463912316E+0, data);
	data += add_point_4(0.3005701484901752E-2, 0.4794682625712025E+0, data);
	data += add_point_4(0.2990992529653774E-2, 0.1927533154878019E+0, data);
	data += add_point_4(0.2982170644107595E-2, 0.6930357961327123E+0, data);
	data += add_point_4(0.2721564237310992E-2, 0.3608302115520091E+0, data);
	data += add_point_4(0.3033513795811141E-2, 0.6498486161496169E+0, data);
	data += add_point_5(0.3007949555218533E-2, 0.1932945013230339E+0, data);
	data += add_point_5(0.2881964603055307E-2, 0.3800494919899303E+0, data);
	data += add_point_6(0.2958357626535696E-2, 0.2899558825499574E+0, 0.7934537856582316E+0, data);
	data += add_point_6(0.3036020026407088E-2, 0.9684121455103957E-1, 0.8280801506686862E+0, data);
	data += add_point_6(0.2832187403926303E-2, 0.1833434647041659E+0, 0.9074658265305127E+0, data);
}

static void make_434_grid()
{
	grid_t *data = &gd434[0];
	gd_list[15] = &gd434[0];
	data += add_point_1(0.5265897968224436E-3, data);
	data += add_point_2(0.2548219972002607E-2, data);
	data += add_point_3(0.2512317418927307E-2, data);
	data += add_point_4(0.2530403801186355E-2, 0.6909346307509111E+0, data);
	data += add_point_4(0.2014279020918528E-2, 0.1774836054609158E+0, data);
	data += add_point_4(0.2501725168402936E-2, 0.4914342637784746E+0, data);
	data += add_point_4(0.2513267174597564E-2, 0.6456664707424256E+0, data);
	data += add_point_4(0.2302694782227416E-2, 0.2861289010307638E+0, data);
	data += add_point_4(0.1462495621594614E-2, 0.7568084367178018E-1, data);
	data += add_point_4(0.2445373437312980E-2, 0.3927259763368002E+0, data);
	data += add_point_5(0.2417442375638981E-2, 0.8818132877794288E+0, data);
	data += add_point_5(0.1910951282179532E-2, 0.9776428111182649E+0, data);
	data += add_point_6(0.2416930044324775E-2, 0.2054823696403044E+0, 0.8689460322872412E+0, data);
	data += add_point_6(0.2512236854563495E-2, 0.5905157048925271E+0, 0.7999278543857286E+0, data);
	data += add_point_6(0.2496644054553086E-2, 0.5550152361076807E+0, 0.7717462626915901E+0, data);
	data += add_point_6(0.2236607760437849E-2, 0.9371809858553722E+0, 0.3344363145343455E+0, data);
}

static void make_590_grid()
{
	grid_t *data = &gd590[0];
	gd_list[16] = &gd590[0];
	data += add_point_1(0.3095121295306187E-3, data);
	data += add_point_3(0.1852379698597489E-2, data);
	data += add_point_4(0.1871790639277744E-2, 0.7040954938227469E+0, data);
	data += add_point_4(0.1858812585438317E-2, 0.6807744066455243E+0, data);
	data += add_point_4(0.1852028828296213E-2, 0.6372546939258752E+0, data);
	data += add_point_4(0.1846715956151242E-2, 0.5044419707800358E+0, data);
	data += add_point_4(0.1818471778162769E-2, 0.4215761784010967E+0, data);
	data += add_point_4(0.1749564657281154E-2, 0.3317920736472123E+0, data);
	data += add_point_4(0.1617210647254411E-2, 0.2384736701421887E+0, data);
	data += add_point_4(0.1384737234851692E-2, 0.1459036449157763E+0, data);
	data += add_point_4(0.9764331165051050E-3, 0.6095034115507196E-1, data);
	data += add_point_5(0.1857161196774078E-2, 0.6116843442009876E+0, data);
	data += add_point_5(0.1705153996395864E-2, 0.3964755348199858E+0, data);
	data += add_point_5(0.1300321685886048E-2, 0.1724782009907724E+0, data);
	data += add_point_6(0.1842866472905286E-2, 0.5610263808622060E+0, 0.3518280927733519E+0, data);
	data += add_point_6(0.1802658934377451E-2, 0.4742392842551980E+0, 0.2634716655937950E+0, data);
	data += add_point_6(0.1849830560443660E-2, 0.5984126497885380E+0, 0.1816640840360209E+0, data);
	data += add_point_6(0.1713904507106709E-2, 0.3791035407695563E+0, 0.1720795225656878E+0, data);
	data += add_point_6(0.1555213603396808E-2, 0.2778673190586244E+0, 0.8213021581932511E-1, data);
	data += add_point_6(0.1802239128008525E-2, 0.5033564271075117E+0, 0.8999205842074875E-1, data);
}

static void make_770_grid()
{
	grid_t *data = &gd770[0];
	gd_list[17] = &gd770[0];
	data += add_point_1(0.2192942088181184E-3, data);
	data += add_point_2(0.1436433617319080E-2, data);
	data += add_point_3(0.1421940344335877E-2, data);
	data += add_point_4(0.6798123511050502E-3, 0.5087204410502360E-1, data);
	data += add_point_4(0.9913184235294912E-3, 0.1228198790178831E+0, data);
	data += add_point_4(0.1180207833238949E-2, 0.2026890814408786E+0, data);
	data += add_point_4(0.1296599602080921E-2, 0.2847745156464294E+0, data);
	data += add_point_4(0.1365871427428316E-2, 0.3656719078978026E+0, data);
	data += add_point_4(0.1402988604775325E-2, 0.4428264886713469E+0, data);
	data += add_point_4(0.1418645563595609E-2, 0.5140619627249735E+0, data);
	data += add_point_4(0.1421376741851662E-2, 0.6306401219166803E+0, data);
	data += add_point_4(0.1423996475490962E-2, 0.6716883332022612E+0, data);
	data += add_point_4(0.1431554042178567E-2, 0.6979792685336881E+0, data);
	data += add_point_5(0.9254401499865368E-3, 0.1446865674195309E+0, data);
	data += add_point_5(0.1250239995053509E-2, 0.3390263475411216E+0, data);
	data += add_point_5(0.1394365843329230E-2, 0.5335804651263506E+0, data);
	data += add_point_6(0.1127089094671749E-2, 0.6944024393349413E-1, 0.2355187894242326E+0, data);
	data += add_point_6(0.1345753760910670E-2, 0.2269004109529460E+0, 0.4102182474045730E+0, data);
	data += add_point_6(0.1424957283316783E-2, 0.8025574607775339E-1, 0.6214302417481605E+0, data);
	data += add_point_6(0.1261523341237750E-2, 0.1467999527896572E+0, 0.3245284345717394E+0, data);
	data += add_point_6(0.1392547106052696E-2, 0.1571507769824727E+0, 0.5224482189696630E+0, data);
	data += add_point_6(0.1418761677877656E-2, 0.2365702993157246E+0, 0.6017546634089558E+0, data);
	data += add_point_6(0.1338366684479554E-2, 0.7714815866765732E-1, 0.4346575516141163E+0, data);
	data += add_point_6(0.1393700862676131E-2, 0.3062936666210730E+0, 0.4908826589037616E+0, data);
	data += add_point_6(0.1415914757466932E-2, 0.3822477379524787E+0, 0.5648768149099500E+0, data);
}

static void make_974_grid()
{
	grid_t *data = &gd974[0];
	gd_list[18] = &gd974[0];
	data += add_point_1(0.1438294190527431E-3, data);
	data += add_point_3(0.1125772288287004E-2, data);
	data += add_point_4(0.4948029341949241E-3, 0.4292963545341347E-1, data);
	data += add_point_4(0.7357990109125470E-3, 0.1051426854086404E+0, data);
	data += add_point_4(0.8889132771304384E-3, 0.1750024867623087E+0, data);
	data += add_point_4(0.9888347838921435E-3, 0.2477653379650257E+0, data);
	data += add_point_4(0.1053299681709471E-2, 0.3206567123955957E+0, data);
	data += add_point_4(0.1092778807014578E-2, 0.3916520749849983E+0, data);
	data += add_point_4(0.1114389394063227E-2, 0.4590825874187624E+0, data);
	data += add_point_4(0.1123724788051555E-2, 0.5214563888415861E+0, data);
	data += add_point_4(0.1125239325243814E-2, 0.6253170244654199E+0, data);
	data += add_point_4(0.1126153271815905E-2, 0.6637926744523170E+0, data);
	data += add_point_4(0.1130286931123841E-2, 0.6910410398498301E+0, data);
	data += add_point_4(0.1134986534363955E-2, 0.7052907007457760E+0, data);
	data += add_point_5(0.6823367927109931E-3, 0.1236686762657990E+0, data);
	data += add_point_5(0.9454158160447096E-3, 0.2940777114468387E+0, data);
	data += add_point_5(0.1074429975385679E-2, 0.4697753849207649E+0, data);
	data += add_point_5(0.1129300086569132E-2, 0.6334563241139567E+0, data);
	data += add_point_6(0.8436884500901954E-3, 0.5974048614181342E-1, 0.2029128752777523E+0, data);
	data += add_point_6(0.1075255720448885E-2, 0.1375760408473636E+0, 0.4602621942484054E+0, data);
	data += add_point_6(0.1108577236864462E-2, 0.3391016526336286E+0, 0.5030673999662036E+0, data);
	data += add_point_6(0.9566475323783357E-3, 0.1271675191439820E+0, 0.2817606422442134E+0, data);
	data += add_point_6(0.1080663250717391E-2, 0.2693120740413512E+0, 0.4331561291720157E+0, data);
	data += add_point_6(0.1126797131196295E-2, 0.1419786452601918E+0, 0.6256167358580814E+0, data);
	data += add_point_6(0.1022568715358061E-2, 0.6709284600738255E-1, 0.3798395216859157E+0, data);
	data += add_point_6(0.1108960267713108E-2, 0.7057738183256172E-1, 0.5517505421423520E+0, data);
	data += add_point_6(0.1122790653435766E-2, 0.2783888477882155E+0, 0.6029619156159187E+0, data);
	data += add_point_6(0.1032401847117460E-2, 0.1979578938917407E+0, 0.3589606329589096E+0, data);
	data += add_point_6(0.1107249382283854E-2, 0.2087307061103274E+0, 0.5348666438135476E+0, data);
	data += add_point_6(0.1121780048519972E-2, 0.4055122137872836E+0, 0.5674997546074373E+0, data);
}

static void make_1202_grid()
{
	grid_t *data = &gd1202[0];
	gd_list[19] = &gd1202[0];
	data += add_point_1(0.1105189233267572E-3, data);
	data += add_point_2(0.9205232738090741E-3, data);
	data += add_point_3(0.9133159786443561E-3, data);
	data += add_point_4(0.3690421898017899E-3, 0.3712636449657089E-1, data);
	data += add_point_4(0.5603990928680660E-3, 0.9140060412262223E-1, data);
	data += add_point_4(0.6865297629282609E-3, 0.1531077852469906E+0, data);
	data += add_point_4(0.7720338551145630E-3, 0.2180928891660612E+0, data);
	data += add_point_4(0.8301545958894795E-3, 0.2839874532200175E+0, data);
	data += add_point_4(0.8686692550179628E-3, 0.3491177600963764E+0, data);
	data += add_point_4(0.8927076285846890E-3, 0.4121431461444309E+0, data);
	data += add_point_4(0.9060820238568219E-3, 0.4718993627149127E+0, data);
	data += add_point_4(0.9119777254940867E-3, 0.5273145452842337E+0, data);
	data += add_point_4(0.9128720138604181E-3, 0.6209475332444019E+0, data);
	data += add_point_4(0.9130714935691735E-3, 0.6569722711857291E+0, data);
	data += add_point_4(0.9152873784554116E-3, 0.6841788309070143E+0, data);
	data += add_point_4(0.9187436274321654E-3, 0.7012604330123631E+0, data);
	data += add_point_5(0.5176977312965694E-3, 0.1072382215478166E+0, data);
	data += add_point_5(0.7331143682101417E-3, 0.2582068959496968E+0, data);
	data += add_point_5(0.8463232836379928E-3, 0.4172752955306717E+0, data);
	data += add_point_5(0.9031122694253992E-3, 0.5700366911792503E+0, data);
	data += add_point_6(0.6485778453163257E-3, 0.9827986018263947E+0, 0.1771774022615325E+0, data);
	data += add_point_6(0.7435030910982369E-3, 0.9624249230326228E+0, 0.2475716463426288E+0, data);
	data += add_point_6(0.7998527891839054E-3, 0.9402007994128811E+0, 0.3354616289066489E+0, data);
	data += add_point_6(0.8101731497468018E-3, 0.9320822040143202E+0, 0.3173615246611977E+0, data);
	data += add_point_6(0.8483389574594331E-3, 0.9043674199393299E+0, 0.4090268427085357E+0, data);
	data += add_point_6(0.8556299257311812E-3, 0.8912407560074747E+0, 0.3854291150669224E+0, data);
	data += add_point_6(0.8803208679738260E-3, 0.8676435628462708E+0, 0.4932221184851285E+0, data);
	data += add_point_6(0.8811048182425720E-3, 0.8581979986041619E+0, 0.4785320675922435E+0, data);
	data += add_point_6(0.8850282341265444E-3, 0.8396753624049856E+0, 0.4507422593157064E+0, data);
	data += add_point_6(0.9021342299040653E-3, 0.8165288564022188E+0, 0.5632123020762100E+0, data);
	data += add_point_6(0.9010091677105086E-3, 0.8015469370783529E+0, 0.5434303569693900E+0, data);
	data += add_point_6(0.9022692938426915E-3, 0.7773563069070351E+0, 0.5123518486419871E+0, data);
	data += add_point_6(0.9158016174693465E-3, 0.7661621213900394E+0, 0.6394279634749102E+0, data);
	data += add_point_6(0.9131578003189435E-3, 0.7553584143533510E+0, 0.6269805509024392E+0, data);
	data += add_point_6(0.9107813579482705E-3, 0.7344305757559503E+0, 0.6031161693096310E+0, data);
	data += add_point_6(0.9105760258970126E-3, 0.7043837184021765E+0, 0.5693702498468441E+0, data);
}

static void make_1454_grid()
{
	grid_t *data = &gd1454[0];
	gd_list[20] = &gd1454[0];
	data += add_point_1(0.7777160743261247E-4, data);
	data += add_point_3(0.7557646413004701E-3, data);
	data += add_point_4(0.2841633806090617E-3, 0.3229290663413854E-1, data);
	data += add_point_4(0.4374419127053555E-3, 0.8036733271462222E-1, data);
	data += add_point_4(0.5417174740872172E-3, 0.1354289960531653E+0, data);
	data += add_point_4(0.6148000891358593E-3, 0.1938963861114426E+0, data);
	data += add_point_4(0.6664394485800705E-3, 0.2537343715011275E+0, data);
	data += add_point_4(0.7025039356923220E-3, 0.3135251434752570E+0, data);
	data += add_point_4(0.7268511789249627E-3, 0.3721558339375338E+0, data);
	data += add_point_4(0.7422637534208629E-3, 0.4286809575195696E+0, data);
	data += add_point_4(0.7509545035841214E-3, 0.4822510128282994E+0, data);
	data += add_point_4(0.7548535057718401E-3, 0.5320679333566263E+0, data);
	data += add_point_4(0.7554088969774001E-3, 0.6172998195394274E+0, data);
	data += add_point_4(0.7553147174442808E-3, 0.6510679849127481E+0, data);
	data += add_point_4(0.7564767653292297E-3, 0.6777315251687360E+0, data);
	data += add_point_4(0.7587991808518730E-3, 0.6963109410648741E+0, data);
	data += add_point_4(0.7608261832033027E-3, 0.7058935009831749E+0, data);
	data += add_point_5(0.4021680447874916E-3, 0.9955546194091857E+0, data);
	data += add_point_5(0.5804871793945964E-3, 0.9734115901794209E+0, data);
	data += add_point_5(0.6792151955945159E-3, 0.9275693732388626E+0, data);
	data += add_point_5(0.7336741211286294E-3, 0.8568022422795103E+0, data);
	data += add_point_5(0.7581866300989608E-3, 0.7623495553719372E+0, data);
	data += add_point_6(0.7538257859800743E-3, 0.5707522908892223E+0, 0.4387028039889501E+0, data);
	data += add_point_6(0.7483517247053123E-3, 0.5196463388403083E+0, 0.3858908414762617E+0, data);
	data += add_point_6(0.7371763661112059E-3, 0.4646337531215351E+0, 0.3301937372343854E+0, data);
	data += add_point_6(0.7183448895756934E-3, 0.4063901697557691E+0, 0.2725423573563777E+0, data);
	data += add_point_6(0.6895815529822191E-3, 0.3456329466643087E+0, 0.2139510237495250E+0, data);
	data += add_point_6(0.6480105801792886E-3, 0.2831395121050332E+0, 0.1555922309786647E+0, data);
	data += add_point_6(0.5897558896594636E-3, 0.2197682022925330E+0, 0.9892878979686097E-1, data);
	data += add_point_6(0.5095708849247346E-3, 0.1564696098650355E+0, 0.4598642910675510E-1, data);
	data += add_point_6(0.7536906428909755E-3, 0.6027356673721295E+0, 0.3376625140173426E+0, data);
	data += add_point_6(0.7472505965575118E-3, 0.5496032320255096E+0, 0.2822301309727988E+0, data);
	data += add_point_6(0.7343017132279698E-3, 0.4921707755234567E+0, 0.2248632342592540E+0, data);
	data += add_point_6(0.7130871582177445E-3, 0.4309422998598483E+0, 0.1666224723456479E+0, data);
	data += add_point_6(0.6817022032112776E-3, 0.3664108182313672E+0, 0.1086964901822169E+0, data);
	data += add_point_6(0.6380941145604121E-3, 0.2990189057758436E+0, 0.5251989784120085E-1, data);
	data += add_point_6(0.7550381377920310E-3, 0.6268724013144998E+0, 0.2297523657550023E+0, data);
	data += add_point_6(0.7478646640144802E-3, 0.5707324144834607E+0, 0.1723080607093800E+0, data);
	data += add_point_6(0.7335918720601220E-3, 0.5096360901960365E+0, 0.1140238465390513E+0, data);
	data += add_point_6(0.7110120527658118E-3, 0.4438729938312456E+0, 0.5611522095882537E-1, data);
	data += add_point_6(0.7571363978689501E-3, 0.6419978471082389E+0, 0.1164174423140873E+0, data);
	data += add_point_6(0.7489908329079234E-3, 0.5817218061802611E+0, 0.5797589531445219E-1, data);
}

static void make_1730_grid()
{
	grid_t *data = &gd1730[0];
	gd_list[21] = &gd1730[0];
	data += add_point_1(0.6309049437420976E-4, data);
	data += add_point_2(0.6398287705571748E-3, data);
	data += add_point_3(0.6357185073530720E-3, data);
	data += add_point_4(0.2221207162188168E-3, 0.2860923126194662E-1, data);
	data += add_point_4(0.3475784022286848E-3, 0.7142556767711522E-1, data);
	data += add_point_4(0.4350742443589804E-3, 0.1209199540995559E+0, data);
	data += add_point_4(0.4978569136522127E-3, 0.1738673106594379E+0, data);
	data += add_point_4(0.5435036221998053E-3, 0.2284645438467734E+0, data);
	data += add_point_4(0.5765913388219542E-3, 0.2834807671701512E+0, data);
	data += add_point_4(0.6001200359226003E-3, 0.3379680145467339E+0, data);
	data += add_point_4(0.6162178172717512E-3, 0.3911355454819537E+0, data);
	data += add_point_4(0.6265218152438485E-3, 0.4422860353001403E+0, data);
	data += add_point_4(0.6323987160974212E-3, 0.4907781568726057E+0, data);
	data += add_point_4(0.6350767851540569E-3, 0.5360006153211468E+0, data);
	data += add_point_4(0.6354362775297107E-3, 0.6142105973596603E+0, data);
	data += add_point_4(0.6352302462706235E-3, 0.6459300387977504E+0, data);
	data += add_point_4(0.6358117881417972E-3, 0.6718056125089225E+0, data);
	data += add_point_4(0.6373101590310117E-3, 0.6910888533186254E+0, data);
	data += add_point_4(0.6390428961368665E-3, 0.7030467416823252E+0, data);
	data += add_point_5(0.3186913449946576E-3, 0.8354951166354646E-1, data);
	data += add_point_5(0.4678028558591711E-3, 0.2050143009099486E+0, data);
	data += add_point_5(0.5538829697598626E-3, 0.3370208290706637E+0, data);
	data += add_point_5(0.6044475907190476E-3, 0.4689051484233963E+0, data);
	data += add_point_5(0.6313575103509012E-3, 0.5939400424557334E+0, data);
	data += add_point_6(0.4078626431855630E-3, 0.1394983311832261E+0, 0.4097581162050343E-1, data);
	data += add_point_6(0.4759933057812725E-3, 0.1967999180485014E+0, 0.8851987391293348E-1, data);
	data += add_point_6(0.5268151186413440E-3, 0.2546183732548967E+0, 0.1397680182969819E+0, data);
	data += add_point_6(0.5643048560507316E-3, 0.3121281074713875E+0, 0.1929452542226526E+0, data);
	data += add_point_6(0.5914501076613073E-3, 0.3685981078502492E+0, 0.2467898337061562E+0, data);
	data += add_point_6(0.6104561257874195E-3, 0.4233760321547856E+0, 0.3003104124785409E+0, data);
	data += add_point_6(0.6230252860707806E-3, 0.4758671236059246E+0, 0.3526684328175033E+0, data);
	data += add_point_6(0.6305618761760796E-3, 0.5255178579796463E+0, 0.4031134861145713E+0, data);
	data += add_point_6(0.6343092767597889E-3, 0.5718025633734589E+0, 0.4509426448342351E+0, data);
	data += add_point_6(0.5176268945737826E-3, 0.2686927772723415E+0, 0.4711322502423248E-1, data);
	data += add_point_6(0.5564840313313692E-3, 0.3306006819904809E+0, 0.9784487303942695E-1, data);
	data += add_point_6(0.5856426671038980E-3, 0.3904906850594983E+0, 0.1505395810025273E+0, data);
	data += add_point_6(0.6066386925777091E-3, 0.4479957951904390E+0, 0.2039728156296050E+0, data);
	data += add_point_6(0.6208824962234458E-3, 0.5027076848919780E+0, 0.2571529941121107E+0, data);
	data += add_point_6(0.6296314297822907E-3, 0.5542087392260217E+0, 0.3092191375815670E+0, data);
	data += add_point_6(0.6340423756791859E-3, 0.6020850887375187E+0, 0.3593807506130276E+0, data);
	data += add_point_6(0.5829627677107342E-3, 0.4019851409179594E+0, 0.5063389934378671E-1, data);
	data += add_point_6(0.6048693376081110E-3, 0.4635614567449800E+0, 0.1032422269160612E+0, data);
	data += add_point_6(0.6202362317732461E-3, 0.5215860931591575E+0, 0.1566322094006254E+0, data);
	data += add_point_6(0.6299005328403779E-3, 0.5758202499099271E+0, 0.2098082827491099E+0, data);
	data += add_point_6(0.6347722390609353E-3, 0.6259893683876795E+0, 0.2618824114553391E+0, data);
	data += add_point_6(0.6203778981238834E-3, 0.5313795124811891E+0, 0.5263245019338556E-1, data);
	data += add_point_6(0.6308414671239979E-3, 0.5893317955931995E+0, 0.1061059730982005E+0, data);
	data += add_point_6(0.6362706466959498E-3, 0.6426246321215801E+0, 0.1594171564034221E+0, data);
	data += add_point_6(0.6375414170333233E-3, 0.6511904367376113E+0, 0.5354789536565540E-1, data);
}

static void make_2030_grid()
{
	grid_t *data = &gd2030[0];
	gd_list[22] = &gd2030[0];
	data += add_point_1(0.4656031899197431E-4, data);
	data += add_point_3(0.5421549195295507E-3, data);
	data += add_point_4(0.1778522133346553E-3, 0.2540835336814348E-1, data);
	data += add_point_4(0.2811325405682796E-3, 0.6399322800504915E-1, data);
	data += add_point_4(0.3548896312631459E-3, 0.1088269469804125E+0, data);
	data += add_point_4(0.4090310897173364E-3, 0.1570670798818287E+0, data);
	data += add_point_4(0.4493286134169965E-3, 0.2071163932282514E+0, data);
	data += add_point_4(0.4793728447962723E-3, 0.2578914044450844E+0, data);
	data += add_point_4(0.5015415319164265E-3, 0.3085687558169623E+0, data);
	data += add_point_4(0.5175127372677937E-3, 0.3584719706267024E+0, data);
	data += add_point_4(0.5285522262081019E-3, 0.4070135594428709E+0, data);
	data += add_point_4(0.5356832703713962E-3, 0.4536618626222638E+0, data);
	data += add_point_4(0.5397914736175170E-3, 0.4979195686463577E+0, data);
	data += add_point_4(0.5416899441599930E-3, 0.5393075111126999E+0, data);
	data += add_point_4(0.5419308476889938E-3, 0.6115617676843916E+0, data);
	data += add_point_4(0.5416936902030596E-3, 0.6414308435160159E+0, data);
	data += add_point_4(0.5419544338703164E-3, 0.6664099412721607E+0, data);
	data += add_point_4(0.5428983656630975E-3, 0.6859161771214913E+0, data);
	data += add_point_4(0.5442286500098193E-3, 0.6993625593503890E+0, data);
	data += add_point_4(0.5452250345057301E-3, 0.7062393387719380E+0, data);
	data += add_point_5(0.2568002497728530E-3, 0.7479028168349763E-1, data);
	data += add_point_5(0.3827211700292145E-3, 0.1848951153969366E+0, data);
	data += add_point_5(0.4579491561917824E-3, 0.3059529066581305E+0, data);
	data += add_point_5(0.5042003969083574E-3, 0.4285556101021362E+0, data);
	data += add_point_5(0.5312708889976025E-3, 0.5468758653496526E+0, data);
	data += add_point_5(0.5438401790747117E-3, 0.6565821978343439E+0, data);
	data += add_point_6(0.3316041873197344E-3, 0.1253901572367117E+0, 0.3681917226439641E-1, data);
	data += add_point_6(0.3899113567153771E-3, 0.1775721510383941E+0, 0.7982487607213301E-1, data);
	data += add_point_6(0.4343343327201309E-3, 0.2305693358216114E+0, 0.1264640966592335E+0, data);
	data += add_point_6(0.4679415262318919E-3, 0.2836502845992063E+0, 0.1751585683418957E+0, data);
	data += add_point_6(0.4930847981631031E-3, 0.3361794746232590E+0, 0.2247995907632670E+0, data);
	data += add_point_6(0.5115031867540091E-3, 0.3875979172264824E+0, 0.2745299257422246E+0, data);
	data += add_point_6(0.5245217148457367E-3, 0.4374019316999074E+0, 0.3236373482441118E+0, data);
	data += add_point_6(0.5332041499895321E-3, 0.4851275843340022E+0, 0.3714967859436741E+0, data);
	data += add_point_6(0.5384583126021542E-3, 0.5303391803806868E+0, 0.4175353646321745E+0, data);
	data += add_point_6(0.5411067210798852E-3, 0.5726197380596287E+0, 0.4612084406355461E+0, data);
	data += add_point_6(0.4259797391468714E-3, 0.2431520732564863E+0, 0.4258040133043952E-1, data);
	data += add_point_6(0.4604931368460021E-3, 0.3002096800895869E+0, 0.8869424306722721E-1, data);
	data += add_point_6(0.4871814878255202E-3, 0.3558554457457432E+0, 0.1368811706510655E+0, data);
	data += add_point_6(0.5072242910074885E-3, 0.4097782537048887E+0, 0.1860739985015033E+0, data);
	data += add_point_6(0.5217069845235350E-3, 0.4616337666067458E+0, 0.2354235077395853E+0, data);
	data += add_point_6(0.5315785966280310E-3, 0.5110707008417874E+0, 0.2842074921347011E+0, data);
	data += add_point_6(0.5376833708758905E-3, 0.5577415286163795E+0, 0.3317784414984102E+0, data);
	data += add_point_6(0.5408032092069521E-3, 0.6013060431366950E+0, 0.3775299002040700E+0, data);
	data += add_point_6(0.4842744917904866E-3, 0.3661596767261781E+0, 0.4599367887164592E-1, data);
	data += add_point_6(0.5048926076188130E-3, 0.4237633153506581E+0, 0.9404893773654421E-1, data);
	data += add_point_6(0.5202607980478373E-3, 0.4786328454658452E+0, 0.1431377109091971E+0, data);
	data += add_point_6(0.5309932388325743E-3, 0.5305702076789774E+0, 0.1924186388843570E+0, data);
	data += add_point_6(0.5377419770895208E-3, 0.5793436224231788E+0, 0.2411590944775190E+0, data);
	data += add_point_6(0.5411696331677717E-3, 0.6247069017094747E+0, 0.2886871491583605E+0, data);
	data += add_point_6(0.5197996293282420E-3, 0.4874315552535204E+0, 0.4804978774953206E-1, data);
	data += add_point_6(0.5311120836622945E-3, 0.5427337322059053E+0, 0.9716857199366665E-1, data);
	data += add_point_6(0.5384309319956951E-3, 0.5943493747246700E+0, 0.1465205839795055E+0, data);
	data += add_point_6(0.5421859504051886E-3, 0.6421314033564943E+0, 0.1953579449803574E+0, data);
	data += add_point_6(0.5390948355046314E-3, 0.6020628374713980E+0, 0.4916375015738108E-1, data);
	data += add_point_6(0.5433312705027845E-3, 0.6529222529856881E+0, 0.9861621540127005E-1, data);
}

static void make_2354_grid()
{
	grid_t *data = &gd2354[0];
	gd_list[23] = &gd2354[0];
	data += add_point_1(0.3922616270665292E-4, data);
	data += add_point_2(0.4703831750854424E-3, data);
	data += add_point_3(0.4678202801282136E-3, data);
	data += add_point_4(0.1437832228979900E-3, 0.2290024646530589E-1, data);
	data += add_point_4(0.2303572493577644E-3, 0.5779086652271284E-1, data);
	data += add_point_4(0.2933110752447454E-3, 0.9863103576375984E-1, data);
	data += add_point_4(0.3402905998359838E-3, 0.1428155792982185E+0, data);
	data += add_point_4(0.3759138466870372E-3, 0.1888978116601463E+0, data);
	data += add_point_4(0.4030638447899798E-3, 0.2359091682970210E+0, data);
	data += add_point_4(0.4236591432242211E-3, 0.2831228833706171E+0, data);
	data += add_point_4(0.4390522656946746E-3, 0.3299495857966693E+0, data);
	data += add_point_4(0.4502523466626247E-3, 0.3758840802660796E+0, data);
	data += add_point_4(0.4580577727783541E-3, 0.4204751831009480E+0, data);
	data += add_point_4(0.4631391616615899E-3, 0.4633068518751051E+0, data);
	data += add_point_4(0.4660928953698676E-3, 0.5039849474507313E+0, data);
	data += add_point_4(0.4674751807936953E-3, 0.5421265793440747E+0, data);
	data += add_point_4(0.4676414903932920E-3, 0.6092660230557310E+0, data);
	data += add_point_4(0.4674086492347870E-3, 0.6374654204984869E+0, data);
	data += add_point_4(0.4674928539483207E-3, 0.6615136472609892E+0, data);
	data += add_point_4(0.4680748979686447E-3, 0.6809487285958127E+0, data);
	data += add_point_4(0.4690449806389040E-3, 0.6952980021665196E+0, data);
	data += add_point_4(0.4699877075860818E-3, 0.7041245497695400E+0, data);
	data += add_point_5(0.2099942281069176E-3, 0.6744033088306065E-1, data);
	data += add_point_5(0.3172269150712804E-3, 0.1678684485334166E+0, data);
	data += add_point_5(0.3832051358546523E-3, 0.2793559049539613E+0, data);
	data += add_point_5(0.4252193818146985E-3, 0.3935264218057639E+0, data);
	data += add_point_5(0.4513807963755000E-3, 0.5052629268232558E+0, data);
	data += add_point_5(0.4657797469114178E-3, 0.6107905315437531E+0, data);
	data += add_point_6(0.2733362800522836E-3, 0.1135081039843524E+0, 0.3331954884662588E-1, data);
	data += add_point_6(0.3235485368463559E-3, 0.1612866626099378E+0, 0.7247167465436538E-1, data);
	data += add_point_6(0.3624908726013453E-3, 0.2100786550168205E+0, 0.1151539110849745E+0, data);
	data += add_point_6(0.3925540070712828E-3, 0.2592282009459942E+0, 0.1599491097143677E+0, data);
	data += add_point_6(0.4156129781116235E-3, 0.3081740561320203E+0, 0.2058699956028027E+0, data);
	data += add_point_6(0.4330644984623263E-3, 0.3564289781578164E+0, 0.2521624953502911E+0, data);
	data += add_point_6(0.4459677725921312E-3, 0.4035587288240703E+0, 0.2982090785797674E+0, data);
	data += add_point_6(0.4551593004456795E-3, 0.4491671196373903E+0, 0.3434762087235733E+0, data);
	data += add_point_6(0.4613341462749918E-3, 0.4928854782917489E+0, 0.3874831357203437E+0, data);
	data += add_point_6(0.4651019618269806E-3, 0.5343646791958988E+0, 0.4297814821746926E+0, data);
	data += add_point_6(0.4670249536100625E-3, 0.5732683216530990E+0, 0.4699402260943537E+0, data);
	data += add_point_6(0.3549555576441708E-3, 0.2214131583218986E+0, 0.3873602040643895E-1, data);
	data += add_point_6(0.3856108245249010E-3, 0.2741796504750071E+0, 0.8089496256902013E-1, data);
	data += add_point_6(0.4098622845756882E-3, 0.3259797439149485E+0, 0.1251732177620872E+0, data);
	data += add_point_6(0.4286328604268950E-3, 0.3765441148826891E+0, 0.1706260286403185E+0, data);
	data += add_point_6(0.4427802198993945E-3, 0.4255773574530558E+0, 0.2165115147300408E+0, data);
	data += add_point_6(0.4530473511488561E-3, 0.4727795117058430E+0, 0.2622089812225259E+0, data);
	data += add_point_6(0.4600805475703138E-3, 0.5178546895819012E+0, 0.3071721431296201E+0, data);
	data += add_point_6(0.4644599059958017E-3, 0.5605141192097460E+0, 0.3508998998801138E+0, data);
	data += add_point_6(0.4667274455712508E-3, 0.6004763319352512E+0, 0.3929160876166931E+0, data);
	data += add_point_6(0.4069360518020356E-3, 0.3352842634946949E+0, 0.4202563457288019E-1, data);
	data += add_point_6(0.4260442819919195E-3, 0.3891971629814670E+0, 0.8614309758870850E-1, data);
	data += add_point_6(0.4408678508029063E-3, 0.4409875565542281E+0, 0.1314500879380001E+0, data);
	data += add_point_6(0.4518748115548597E-3, 0.4904893058592484E+0, 0.1772189657383859E+0, data);
	data += add_point_6(0.4595564875375116E-3, 0.5375056138769549E+0, 0.2228277110050294E+0, data);
	data += add_point_6(0.4643988774315846E-3, 0.5818255708669969E+0, 0.2677179935014386E+0, data);
	data += add_point_6(0.4668827491646946E-3, 0.6232334858144959E+0, 0.3113675035544165E+0, data);
	data += add_point_6(0.4400541823741973E-3, 0.4489485354492058E+0, 0.4409162378368174E-1, data);
	data += add_point_6(0.4514512890193797E-3, 0.5015136875933150E+0, 0.8939009917748489E-1, data);
	data += add_point_6(0.4596198627347549E-3, 0.5511300550512623E+0, 0.1351806029383365E+0, data);
	data += add_point_6(0.4648659016801781E-3, 0.5976720409858000E+0, 0.1808370355053196E+0, data);
	data += add_point_6(0.4675502017157673E-3, 0.6409956378989354E+0, 0.2257852192301602E+0, data);
	data += add_point_6(0.4598494476455523E-3, 0.5581222330827514E+0, 0.4532173421637160E-1, data);
	data += add_point_6(0.4654916955152048E-3, 0.6074705984161695E+0, 0.9117488031840314E-1, data);
	data += add_point_6(0.4684709779505137E-3, 0.6532272537379033E+0, 0.1369294213140155E+0, data);
	data += add_point_6(0.4691445539106986E-3, 0.6594761494500487E+0, 0.4589901487275583E-1, data);
}

static void make_2702_grid()
{
	grid_t *data = &gd2702[0];
	gd_list[24] = &gd2702[0];
	data += add_point_1(0.2998675149888161E-4, data);
	data += add_point_3(0.4077860529495355E-3, data);
	data += add_point_4(0.1185349192520667E-3, 0.2065562538818703E-1, data);
	data += add_point_4(0.1913408643425751E-3, 0.5250918173022379E-1, data);
	data += add_point_4(0.2452886577209897E-3, 0.8993480082038376E-1, data);
	data += add_point_4(0.2862408183288702E-3, 0.1306023924436019E+0, data);
	data += add_point_4(0.3178032258257357E-3, 0.1732060388531418E+0, data);
	data += add_point_4(0.3422945667633690E-3, 0.2168727084820249E+0, data);
	data += add_point_4(0.3612790520235922E-3, 0.2609528309173586E+0, data);
	data += add_point_4(0.3758638229818521E-3, 0.3049252927938952E+0, data);
	data += add_point_4(0.3868711798859953E-3, 0.3483484138084404E+0, data);
	data += add_point_4(0.3949429933189938E-3, 0.3908321549106406E+0, data);
	data += add_point_4(0.4006068107541156E-3, 0.4320210071894814E+0, data);
	data += add_point_4(0.4043192149672723E-3, 0.4715824795890053E+0, data);
	data += add_point_4(0.4064947495808078E-3, 0.5091984794078453E+0, data);
	data += add_point_4(0.4075245619813152E-3, 0.5445580145650803E+0, data);
	data += add_point_4(0.4076423540893566E-3, 0.6072575796841768E+0, data);
	data += add_point_4(0.4074280862251555E-3, 0.6339484505755803E+0, data);
	data += add_point_4(0.4074163756012244E-3, 0.6570718257486958E+0, data);
	data += add_point_4(0.4077647795071246E-3, 0.6762557330090709E+0, data);
	data += add_point_4(0.4084517552782530E-3, 0.6911161696923790E+0, data);
	data += add_point_4(0.4092468459224052E-3, 0.7012841911659961E+0, data);
	data += add_point_4(0.4097872687240906E-3, 0.7064559272410020E+0, data);
	data += add_point_5(0.1738986811745028E-3, 0.6123554989894765E-1, data);
	data += add_point_5(0.2659616045280191E-3, 0.1533070348312393E+0, data);
	data += add_point_5(0.3240596008171533E-3, 0.2563902605244206E+0, data);
	data += add_point_5(0.3621195964432943E-3, 0.3629346991663361E+0, data);
	data += add_point_5(0.3868838330760539E-3, 0.4683949968987538E+0, data);
	data += add_point_5(0.4018911532693111E-3, 0.5694479240657952E+0, data);
	data += add_point_5(0.4089929432983252E-3, 0.6634465430993955E+0, data);
	data += add_point_6(0.2279907527706409E-3, 0.1033958573552305E+0, 0.3034544009063584E-1, data);
	data += add_point_6(0.2715205490578897E-3, 0.1473521412414395E+0, 0.6618803044247135E-1, data);
	data += add_point_6(0.3057917896703976E-3, 0.1924552158705967E+0, 0.1054431128987715E+0, data);
	data += add_point_6(0.3326913052452555E-3, 0.2381094362890328E+0, 0.1468263551238858E+0, data);
	data += add_point_6(0.3537334711890037E-3, 0.2838121707936760E+0, 0.1894486108187886E+0, data);
	data += add_point_6(0.3700567500783129E-3, 0.3291323133373415E+0, 0.2326374238761579E+0, data);
	data += add_point_6(0.3825245372589122E-3, 0.3736896978741460E+0, 0.2758485808485768E+0, data);
	data += add_point_6(0.3918125171518296E-3, 0.4171406040760013E+0, 0.3186179331996921E+0, data);
	data += add_point_6(0.3984720419937579E-3, 0.4591677985256915E+0, 0.3605329796303794E+0, data);
	data += add_point_6(0.4029746003338211E-3, 0.4994733831718418E+0, 0.4012147253586509E+0, data);
	data += add_point_6(0.4057428632156627E-3, 0.5377731830445096E+0, 0.4403050025570692E+0, data);
	data += add_point_6(0.4071719274114857E-3, 0.5737917830001331E+0, 0.4774565904277483E+0, data);
	data += add_point_6(0.2990236950664119E-3, 0.2027323586271389E+0, 0.3544122504976147E-1, data);
	data += add_point_6(0.3262951734212878E-3, 0.2516942375187273E+0, 0.7418304388646328E-1, data);
	data += add_point_6(0.3482634608242413E-3, 0.3000227995257181E+0, 0.1150502745727186E+0, data);
	data += add_point_6(0.3656596681700892E-3, 0.3474806691046342E+0, 0.1571963371209364E+0, data);
	data += add_point_6(0.3791740467794218E-3, 0.3938103180359209E+0, 0.1999631877247100E+0, data);
	data += add_point_6(0.3894034450156905E-3, 0.4387519590455703E+0, 0.2428073457846535E+0, data);
	data += add_point_6(0.3968600245508371E-3, 0.4820503960077787E+0, 0.2852575132906155E+0, data);
	data += add_point_6(0.4019931351420050E-3, 0.5234573778475101E+0, 0.3268884208674639E+0, data);
	data += add_point_6(0.4052108801278599E-3, 0.5627318647235282E+0, 0.3673033321675939E+0, data);
	data += add_point_6(0.4068978613940934E-3, 0.5996390607156954E+0, 0.4061211551830290E+0, data);
	data += add_point_6(0.3454275351319704E-3, 0.3084780753791947E+0, 0.3860125523100059E-1, data);
	data += add_point_6(0.3629963537007920E-3, 0.3589988275920223E+0, 0.7928938987104867E-1, data);
	data += add_point_6(0.3770187233889873E-3, 0.4078628415881973E+0, 0.1212614643030087E+0, data);
	data += add_point_6(0.3878608613694378E-3, 0.4549287258889735E+0, 0.1638770827382693E+0, data);
	data += add_point_6(0.3959065270221274E-3, 0.5000278512957279E+0, 0.2065965798260176E+0, data);
	data += add_point_6(0.4015286975463570E-3, 0.5429785044928199E+0, 0.2489436378852235E+0, data);
	data += add_point_6(0.4050866785614717E-3, 0.5835939850491711E+0, 0.2904811368946891E+0, data);
	data += add_point_6(0.4069320185051913E-3, 0.6216870353444856E+0, 0.3307941957666609E+0, data);
	data += add_point_6(0.3760120964062763E-3, 0.4151104662709091E+0, 0.4064829146052554E-1, data);
	data += add_point_6(0.3870969564418064E-3, 0.4649804275009218E+0, 0.8258424547294755E-1, data);
	data += add_point_6(0.3955287790534055E-3, 0.5124695757009662E+0, 0.1251841962027289E+0, data);
	data += add_point_6(0.4015361911302668E-3, 0.5574711100606224E+0, 0.1679107505976331E+0, data);
	data += add_point_6(0.4053836986719548E-3, 0.5998597333287227E+0, 0.2102805057358715E+0, data);
	data += add_point_6(0.4073578673299117E-3, 0.6395007148516600E+0, 0.2518418087774107E+0, data);
	data += add_point_6(0.3954628379231406E-3, 0.5188456224746252E+0, 0.4194321676077518E-1, data);
	data += add_point_6(0.4017645508847530E-3, 0.5664190707942778E+0, 0.8457661551921499E-1, data);
	data += add_point_6(0.4059030348651293E-3, 0.6110464353283153E+0, 0.1273652932519396E+0, data);
	data += add_point_6(0.4080565809484880E-3, 0.6526430302051563E+0, 0.1698173239076354E+0, data);
	data += add_point_6(0.4063018753664651E-3, 0.6167551880377548E+0, 0.4266398851548864E-1, data);
	data += add_point_6(0.4087191292799671E-3, 0.6607195418355383E+0, 0.8551925814238349E-1, data);
}

static void make_3074_grid()
{
	grid_t *data = &gd3074[0];
	gd_list[25] = &gd3074[0];
	data += add_point_1(0.2599095953754734E-4, data);
	data += add_point_2(0.3603134089687541E-3, data);
	data += add_point_3(0.3586067974412447E-3, data);
	data += add_point_4(0.9831528474385880E-4, 0.1886108518723392E-1, data);
	data += add_point_4(0.1605023107954450E-3, 0.4800217244625303E-1, data);
	data += add_point_4(0.2072200131464099E-3, 0.8244922058397242E-1, data);
	data += add_point_4(0.2431297618814187E-3, 0.1200408362484023E+0, data);
	data += add_point_4(0.2711819064496707E-3, 0.1595773530809965E+0, data);
	data += add_point_4(0.2932762038321116E-3, 0.2002635973434064E+0, data);
	data += add_point_4(0.3107032514197368E-3, 0.2415127590139982E+0, data);
	data += add_point_4(0.3243808058921213E-3, 0.2828584158458477E+0, data);
	data += add_point_4(0.3349899091374030E-3, 0.3239091015338138E+0, data);
	data += add_point_4(0.3430580688505218E-3, 0.3643225097962194E+0, data);
	data += add_point_4(0.3490124109290343E-3, 0.4037897083691802E+0, data);
	data += add_point_4(0.3532148948561955E-3, 0.4420247515194127E+0, data);
	data += add_point_4(0.3559862669062833E-3, 0.4787572538464938E+0, data);
	data += add_point_4(0.3576224317551411E-3, 0.5137265251275234E+0, data);
	data += add_point_4(0.3584050533086076E-3, 0.5466764056654611E+0, data);
	data += add_point_4(0.3584903581373224E-3, 0.6054859420813535E+0, data);
	data += add_point_4(0.3582991879040586E-3, 0.6308106701764562E+0, data);
	data += add_point_4(0.3582371187963125E-3, 0.6530369230179584E+0, data);
	data += add_point_4(0.3584353631122350E-3, 0.6718609524611158E+0, data);
	data += add_point_4(0.3589120166517785E-3, 0.6869676499894013E+0, data);
	data += add_point_4(0.3595445704531601E-3, 0.6980467077240748E+0, data);
	data += add_point_4(0.3600943557111074E-3, 0.7048241721250522E+0, data);
	data += add_point_5(0.1456447096742039E-3, 0.5591105222058232E-1, data);
	data += add_point_5(0.2252370188283782E-3, 0.1407384078513916E+0, data);
	data += add_point_5(0.2766135443474897E-3, 0.2364035438976309E+0, data);
	data += add_point_5(0.3110729491500851E-3, 0.3360602737818170E+0, data);
	data += add_point_5(0.3342506712303391E-3, 0.4356292630054665E+0, data);
	data += add_point_5(0.3491981834026860E-3, 0.5321569415256174E+0, data);
	data += add_point_5(0.3576003604348932E-3, 0.6232956305040554E+0, data);
	data += add_point_6(0.1921921305788564E-3, 0.9469870086838469E-1, 0.2778748387309470E-1, data);
	data += add_point_6(0.2301458216495632E-3, 0.1353170300568141E+0, 0.6076569878628364E-1, data);
	data += add_point_6(0.2604248549522893E-3, 0.1771679481726077E+0, 0.9703072762711040E-1, data);
	data += add_point_6(0.2845275425870697E-3, 0.2197066664231751E+0, 0.1354112458524762E+0, data);
	data += add_point_6(0.3036870897974840E-3, 0.2624783557374927E+0, 0.1750996479744100E+0, data);
	data += add_point_6(0.3188414832298066E-3, 0.3050969521214442E+0, 0.2154896907449802E+0, data);
	data += add_point_6(0.3307046414722089E-3, 0.3472252637196021E+0, 0.2560954625740152E+0, data);
	data += add_point_6(0.3398330969031360E-3, 0.3885610219026360E+0, 0.2965070050624096E+0, data);
	data += add_point_6(0.3466757899705373E-3, 0.4288273776062765E+0, 0.3363641488734497E+0, data);
	data += add_point_6(0.3516095923230054E-3, 0.4677662471302948E+0, 0.3753400029836788E+0, data);
	data += add_point_6(0.3549645184048486E-3, 0.5051333589553359E+0, 0.4131297522144286E+0, data);
	data += add_point_6(0.3570415969441392E-3, 0.5406942145810492E+0, 0.4494423776081795E+0, data);
	data += add_point_6(0.3581251798496118E-3, 0.5742204122576457E+0, 0.4839938958841502E+0, data);
	data += add_point_6(0.2543491329913348E-3, 0.1865407027225188E+0, 0.3259144851070796E-1, data);
	data += add_point_6(0.2786711051330776E-3, 0.2321186453689432E+0, 0.6835679505297343E-1, data);
	data += add_point_6(0.2985552361083679E-3, 0.2773159142523882E+0, 0.1062284864451989E+0, data);
	data += add_point_6(0.3145867929154039E-3, 0.3219200192237254E+0, 0.1454404409323047E+0, data);
	data += add_point_6(0.3273290662067609E-3, 0.3657032593944029E+0, 0.1854018282582510E+0, data);
	data += add_point_6(0.3372705511943501E-3, 0.4084376778363622E+0, 0.2256297412014750E+0, data);
	data += add_point_6(0.3448274437851510E-3, 0.4499004945751427E+0, 0.2657104425000896E+0, data);
	data += add_point_6(0.3503592783048583E-3, 0.4898758141326335E+0, 0.3052755487631557E+0, data);
	data += add_point_6(0.3541854792663162E-3, 0.5281547442266309E+0, 0.3439863920645423E+0, data);
	data += add_point_6(0.3565995517909428E-3, 0.5645346989813992E+0, 0.3815229456121914E+0, data);
	data += add_point_6(0.3578802078302898E-3, 0.5988181252159848E+0, 0.4175752420966734E+0, data);
	data += add_point_6(0.2958644592860982E-3, 0.2850425424471603E+0, 0.3562149509862536E-1, data);
	data += add_point_6(0.3119548129116835E-3, 0.3324619433027876E+0, 0.7330318886871096E-1, data);
	data += add_point_6(0.3250745225005984E-3, 0.3785848333076282E+0, 0.1123226296008472E+0, data);
	data += add_point_6(0.3355153415935208E-3, 0.4232891028562115E+0, 0.1521084193337708E+0, data);
	data += add_point_6(0.3435847568549328E-3, 0.4664287050829722E+0, 0.1921844459223610E+0, data);
	data += add_point_6(0.3495786831622488E-3, 0.5078458493735726E+0, 0.2321360989678303E+0, data);
	data += add_point_6(0.3537767805534621E-3, 0.5473779816204180E+0, 0.2715886486360520E+0, data);
	data += add_point_6(0.3564459815421428E-3, 0.5848617133811376E+0, 0.3101924707571355E+0, data);
	data += add_point_6(0.3578464061225468E-3, 0.6201348281584888E+0, 0.3476121052890973E+0, data);
	data += add_point_6(0.3239748762836212E-3, 0.3852191185387871E+0, 0.3763224880035108E-1, data);
	data += add_point_6(0.3345491784174287E-3, 0.4325025061073423E+0, 0.7659581935637135E-1, data);
	data += add_point_6(0.3429126177301782E-3, 0.4778486229734490E+0, 0.1163381306083900E+0, data);
	data += add_point_6(0.3492420343097421E-3, 0.5211663693009000E+0, 0.1563890598752899E+0, data);
	data += add_point_6(0.3537399050235257E-3, 0.5623469504853703E+0, 0.1963320810149200E+0, data);
	data += add_point_6(0.3566209152659172E-3, 0.6012718188659246E+0, 0.2357847407258738E+0, data);
	data += add_point_6(0.3581084321919782E-3, 0.6378179206390117E+0, 0.2743846121244060E+0, data);
	data += add_point_6(0.3426522117591512E-3, 0.4836936460214534E+0, 0.3895902610739024E-1, data);
	data += add_point_6(0.3491848770121379E-3, 0.5293792562683797E+0, 0.7871246819312640E-1, data);
	data += add_point_6(0.3539318235231476E-3, 0.5726281253100033E+0, 0.1187963808202981E+0, data);
	data += add_point_6(0.3570231438458694E-3, 0.6133658776169068E+0, 0.1587914708061787E+0, data);
	data += add_point_6(0.3586207335051714E-3, 0.6515085491865307E+0, 0.1983058575227646E+0, data);
	data += add_point_6(0.3541196205164025E-3, 0.5778692716064976E+0, 0.3977209689791542E-1, data);
	data += add_point_6(0.3574296911573953E-3, 0.6207904288086192E+0, 0.7990157592981152E-1, data);
	data += add_point_6(0.3591993279818963E-3, 0.6608688171046802E+0, 0.1199671308754309E+0, data);
	data += add_point_6(0.3595855034661997E-3, 0.6656263089489130E+0, 0.4015955957805969E-1, data);
}

static void make_3470_grid()
{
	grid_t *data = &gd3470[0];
	gd_list[26] = &gd3470[0];
	data += add_point_1(0.2040382730826330E-4, data);
	data += add_point_3(0.3178149703889544E-3, data);
	data += add_point_4(0.8288115128076110E-4, 0.1721420832906233E-1, data);
	data += add_point_4(0.1360883192522954E-3, 0.4408875374981770E-1, data);
	data += add_point_4(0.1766854454542662E-3, 0.7594680813878681E-1, data);
	data += add_point_4(0.2083153161230153E-3, 0.1108335359204799E+0, data);
	data += add_point_4(0.2333279544657158E-3, 0.1476517054388567E+0, data);
	data += add_point_4(0.2532809539930247E-3, 0.1856731870860615E+0, data);
	data += add_point_4(0.2692472184211158E-3, 0.2243634099428821E+0, data);
	data += add_point_4(0.2819949946811885E-3, 0.2633006881662727E+0, data);
	data += add_point_4(0.2920953593973030E-3, 0.3021340904916283E+0, data);
	data += add_point_4(0.2999889782948352E-3, 0.3405594048030089E+0, data);
	data += add_point_4(0.3060292120496902E-3, 0.3783044434007372E+0, data);
	data += add_point_4(0.3105109167522192E-3, 0.4151194767407910E+0, data);
	data += add_point_4(0.3136902387550312E-3, 0.4507705766443257E+0, data);
	data += add_point_4(0.3157984652454632E-3, 0.4850346056573187E+0, data);
	data += add_point_4(0.3170516518425422E-3, 0.5176950817792470E+0, data);
	data += add_point_4(0.3176568425633755E-3, 0.5485384240820989E+0, data);
	data += add_point_4(0.3177198411207062E-3, 0.6039117238943308E+0, data);
	data += add_point_4(0.3175519492394733E-3, 0.6279956655573113E+0, data);
	data += add_point_4(0.3174654952634756E-3, 0.6493636169568952E+0, data);
	data += add_point_4(0.3175676415467654E-3, 0.6677644117704504E+0, data);
	data += add_point_4(0.3178923417835410E-3, 0.6829368572115624E+0, data);
	data += add_point_4(0.3183788287531909E-3, 0.6946195818184121E+0, data);
	data += add_point_4(0.3188755151918807E-3, 0.7025711542057026E+0, data);
	data += add_point_4(0.3191916889313849E-3, 0.7066004767140119E+0, data);
	data += add_point_5(0.1231779611744508E-3, 0.5132537689946062E-1, data);
	data += add_point_5(0.1924661373839880E-3, 0.1297994661331225E+0, data);
	data += add_point_5(0.2380881867403424E-3, 0.2188852049401307E+0, data);
	data += add_point_5(0.2693100663037885E-3, 0.3123174824903457E+0, data);
	data += add_point_5(0.2908673382834366E-3, 0.4064037620738195E+0, data);
	data += add_point_5(0.3053914619381535E-3, 0.4984958396944782E+0, data);
	data += add_point_5(0.3143916684147777E-3, 0.5864975046021365E+0, data);
	data += add_point_5(0.3187042244055363E-3, 0.6686711634580175E+0, data);
	data += add_point_6(0.1635219535869790E-3, 0.8715738780835950E-1, 0.2557175233367578E-1, data);
	data += add_point_6(0.1968109917696070E-3, 0.1248383123134007E+0, 0.5604823383376681E-1, data);
	data += add_point_6(0.2236754342249974E-3, 0.1638062693383378E+0, 0.8968568601900765E-1, data);
	data += add_point_6(0.2453186687017181E-3, 0.2035586203373176E+0, 0.1254086651976279E+0, data);
	data += add_point_6(0.2627551791580541E-3, 0.2436798975293774E+0, 0.1624780150162012E+0, data);
	data += add_point_6(0.2767654860152220E-3, 0.2838207507773806E+0, 0.2003422342683208E+0, data);
	data += add_point_6(0.2879467027765895E-3, 0.3236787502217692E+0, 0.2385628026255263E+0, data);
	data += add_point_6(0.2967639918918702E-3, 0.3629849554840691E+0, 0.2767731148783578E+0, data);
	data += add_point_6(0.3035900684660351E-3, 0.4014948081992087E+0, 0.3146542308245309E+0, data);
	data += add_point_6(0.3087338237298308E-3, 0.4389818379260225E+0, 0.3519196415895088E+0, data);
	data += add_point_6(0.3124608838860167E-3, 0.4752331143674377E+0, 0.3883050984023654E+0, data);
	data += add_point_6(0.3150084294226743E-3, 0.5100457318374018E+0, 0.4235613423908649E+0, data);
	data += add_point_6(0.3165958398598402E-3, 0.5432238388954868E+0, 0.4574484717196220E+0, data);
	data += add_point_6(0.3174320440957372E-3, 0.5745758685072442E+0, 0.4897311639255524E+0, data);
	data += add_point_6(0.2182188909812599E-3, 0.1723981437592809E+0, 0.3010630597881105E-1, data);
	data += add_point_6(0.2399727933921445E-3, 0.2149553257844597E+0, 0.6326031554204694E-1, data);
	data += add_point_6(0.2579796133514652E-3, 0.2573256081247422E+0, 0.9848566980258631E-1, data);
	data += add_point_6(0.2727114052623535E-3, 0.2993163751238106E+0, 0.1350835952384266E+0, data);
	data += add_point_6(0.2846327656281355E-3, 0.3407238005148000E+0, 0.1725184055442181E+0, data);
	data += add_point_6(0.2941491102051334E-3, 0.3813454978483264E+0, 0.2103559279730725E+0, data);
	data += add_point_6(0.3016049492136107E-3, 0.4209848104423343E+0, 0.2482278774554860E+0, data);
	data += add_point_6(0.3072949726175648E-3, 0.4594519699996300E+0, 0.2858099509982883E+0, data);
	data += add_point_6(0.3114768142886460E-3, 0.4965640166185930E+0, 0.3228075659915428E+0, data);
	data += add_point_6(0.3143823673666223E-3, 0.5321441655571562E+0, 0.3589459907204151E+0, data);
	data += add_point_6(0.3162269764661535E-3, 0.5660208438582166E+0, 0.3939630088864310E+0, data);
	data += add_point_6(0.3172164663759821E-3, 0.5980264315964364E+0, 0.4276029922949089E+0, data);
	data += add_point_6(0.2554575398967435E-3, 0.2644215852350733E+0, 0.3300939429072552E-1, data);
	data += add_point_6(0.2701704069135677E-3, 0.3090113743443063E+0, 0.6803887650078501E-1, data);
	data += add_point_6(0.2823693413468940E-3, 0.3525871079197808E+0, 0.1044326136206709E+0, data);
	data += add_point_6(0.2922898463214289E-3, 0.3950418005354029E+0, 0.1416751597517679E+0, data);
	data += add_point_6(0.3001829062162428E-3, 0.4362475663430163E+0, 0.1793408610504821E+0, data);
	data += add_point_6(0.3062890864542953E-3, 0.4760661812145854E+0, 0.2170630750175722E+0, data);
	data += add_point_6(0.3108328279264746E-3, 0.5143551042512103E+0, 0.2545145157815807E+0, data);
	data += add_point_6(0.3140243146201245E-3, 0.5509709026935597E+0, 0.2913940101706601E+0, data);
	data += add_point_6(0.3160638030977130E-3, 0.5857711030329428E+0, 0.3274169910910705E+0, data);
	data += add_point_6(0.3171462882206275E-3, 0.6186149917404392E+0, 0.3623081329317265E+0, data);
	data += add_point_6(0.2812388416031796E-3, 0.3586894569557064E+0, 0.3497354386450040E-1, data);
	data += add_point_6(0.2912137500288045E-3, 0.4035266610019441E+0, 0.7129736739757095E-1, data);
	data += add_point_6(0.2993241256502206E-3, 0.4467775312332510E+0, 0.1084758620193165E+0, data);
	data += add_point_6(0.3057101738983822E-3, 0.4883638346608543E+0, 0.1460915689241772E+0, data);
	data += add_point_6(0.3105319326251432E-3, 0.5281908348434601E+0, 0.1837790832369980E+0, data);
	data += add_point_6(0.3139565514428167E-3, 0.5661542687149311E+0, 0.2212075390874021E+0, data);
	data += add_point_6(0.3161543006806366E-3, 0.6021450102031452E+0, 0.2580682841160985E+0, data);
	data += add_point_6(0.3172985960613294E-3, 0.6360520783610050E+0, 0.2940656362094121E+0, data);
	data += add_point_6(0.2989400336901431E-3, 0.4521611065087196E+0, 0.3631055365867002E-1, data);
	data += add_point_6(0.3054555883947677E-3, 0.4959365651560963E+0, 0.7348318468484350E-1, data);
	data += add_point_6(0.3104764960807702E-3, 0.5376815804038283E+0, 0.1111087643812648E+0, data);
	data += add_point_6(0.3141015825977616E-3, 0.5773314480243768E+0, 0.1488226085145408E+0, data);
	data += add_point_6(0.3164520621159896E-3, 0.6148113245575056E+0, 0.1862892274135151E+0, data);
	data += add_point_6(0.3176652305912204E-3, 0.6500407462842380E+0, 0.2231909701714456E+0, data);
	data += add_point_6(0.3105097161023939E-3, 0.5425151448707213E+0, 0.3718201306118944E-1, data);
	data += add_point_6(0.3143014117890550E-3, 0.5841860556907931E+0, 0.7483616335067346E-1, data);
	data += add_point_6(0.3168172866287200E-3, 0.6234632186851500E+0, 0.1125990834266120E+0, data);
	data += add_point_6(0.3181401865570968E-3, 0.6602934551848843E+0, 0.1501303813157619E+0, data);
	data += add_point_6(0.3170663659156037E-3, 0.6278573968375105E+0, 0.3767559930245720E-1, data);
	data += add_point_6(0.3185447944625510E-3, 0.6665611711264577E+0, 0.7548443301360158E-1, data);
}

static void make_3890_grid()
{
	grid_t *data = &gd3890[0];
	gd_list[27] = &gd3890[0];
	data += add_point_1(0.1807395252196920E-4, data);
	data += add_point_2(0.2848008782238827E-3, data);
	data += add_point_3(0.2836065837530581E-3, data);
	data += add_point_4(0.7013149266673816E-4, 0.1587876419858352E-1, data);
	data += add_point_4(0.1162798021956766E-3, 0.4069193593751206E-1, data);
	data += add_point_4(0.1518728583972105E-3, 0.7025888115257997E-1, data);
	data += add_point_4(0.1798796108216934E-3, 0.1027495450028704E+0, data);
	data += add_point_4(0.2022593385972785E-3, 0.1371457730893426E+0, data);
	data += add_point_4(0.2203093105575464E-3, 0.1727758532671953E+0, data);
	data += add_point_4(0.2349294234299855E-3, 0.2091492038929037E+0, data);
	data += add_point_4(0.2467682058747003E-3, 0.2458813281751915E+0, data);
	data += add_point_4(0.2563092683572224E-3, 0.2826545859450066E+0, data);
	data += add_point_4(0.2639253896763318E-3, 0.3191957291799622E+0, data);
	data += add_point_4(0.2699137479265108E-3, 0.3552621469299578E+0, data);
	data += add_point_4(0.2745196420166739E-3, 0.3906329503406230E+0, data);
	data += add_point_4(0.2779529197397593E-3, 0.4251028614093031E+0, data);
	data += add_point_4(0.2803996086684265E-3, 0.4584777520111870E+0, data);
	data += add_point_4(0.2820302356715842E-3, 0.4905711358710193E+0, data);
	data += add_point_4(0.2830056747491068E-3, 0.5212011669847385E+0, data);
	data += add_point_4(0.2834808950776839E-3, 0.5501878488737995E+0, data);
	data += add_point_4(0.2835282339078929E-3, 0.6025037877479342E+0, data);
	data += add_point_4(0.2833819267065800E-3, 0.6254572689549016E+0, data);
	data += add_point_4(0.2832858336906784E-3, 0.6460107179528248E+0, data);
	data += add_point_4(0.2833268235451244E-3, 0.6639541138154251E+0, data);
	data += add_point_4(0.2835432677029253E-3, 0.6790688515667495E+0, data);
	data += add_point_4(0.2839091722743049E-3, 0.6911338580371512E+0, data);
	data += add_point_4(0.2843308178875841E-3, 0.6999385956126490E+0, data);
	data += add_point_4(0.2846703550533846E-3, 0.7053037748656896E+0, data);
	data += add_point_5(0.1051193406971900E-3, 0.4732224387180115E-1, data);
	data += add_point_5(0.1657871838796974E-3, 0.1202100529326803E+0, data);
	data += add_point_5(0.2064648113714232E-3, 0.2034304820664855E+0, data);
	data += add_point_5(0.2347942745819741E-3, 0.2912285643573002E+0, data);
	data += add_point_5(0.2547775326597726E-3, 0.3802361792726768E+0, data);
	data += add_point_5(0.2686876684847025E-3, 0.4680598511056146E+0, data);
	data += add_point_5(0.2778665755515867E-3, 0.5528151052155599E+0, data);
	data += add_point_5(0.2830996616782929E-3, 0.6329386307803041E+0, data);
	data += add_point_6(0.1403063340168372E-3, 0.8056516651369069E-1, 0.2363454684003124E-1, data);
	data += add_point_6(0.1696504125939477E-3, 0.1156476077139389E+0, 0.5191291632545936E-1, data);
	data += add_point_6(0.1935787242745390E-3, 0.1520473382760421E+0, 0.8322715736994519E-1, data);
	data += add_point_6(0.2130614510521968E-3, 0.1892986699745931E+0, 0.1165855667993712E+0, data);
	data += add_point_6(0.2289381265931048E-3, 0.2270194446777792E+0, 0.1513077167409504E+0, data);
	data += add_point_6(0.2418630292816186E-3, 0.2648908185093273E+0, 0.1868882025807859E+0, data);
	data += add_point_6(0.2523400495631193E-3, 0.3026389259574136E+0, 0.2229277629776224E+0, data);
	data += add_point_6(0.2607623973449605E-3, 0.3400220296151384E+0, 0.2590951840746235E+0, data);
	data += add_point_6(0.2674441032689209E-3, 0.3768217953335510E+0, 0.2951047291750847E+0, data);
	data += add_point_6(0.2726432360343356E-3, 0.4128372900921884E+0, 0.3307019714169930E+0, data);
	data += add_point_6(0.2765787685924545E-3, 0.4478807131815630E+0, 0.3656544101087634E+0, data);
	data += add_point_6(0.2794428690642224E-3, 0.4817742034089257E+0, 0.3997448951939695E+0, data);
	data += add_point_6(0.2814099002062895E-3, 0.5143472814653344E+0, 0.4327667110812024E+0, data);
	data += add_point_6(0.2826429531578994E-3, 0.5454346213905650E+0, 0.4645196123532293E+0, data);
	data += add_point_6(0.2832983542550884E-3, 0.5748739313170252E+0, 0.4948063555703345E+0, data);
	data += add_point_6(0.1886695565284976E-3, 0.1599598738286342E+0, 0.2792357590048985E-1, data);
	data += add_point_6(0.2081867882748234E-3, 0.1998097412500951E+0, 0.5877141038139065E-1, data);
	data += add_point_6(0.2245148680600796E-3, 0.2396228952566202E+0, 0.9164573914691377E-1, data);
	data += add_point_6(0.2380370491511872E-3, 0.2792228341097746E+0, 0.1259049641962687E+0, data);
	data += add_point_6(0.2491398041852455E-3, 0.3184251107546741E+0, 0.1610594823400863E+0, data);
	data += add_point_6(0.2581632405881230E-3, 0.3570481164426244E+0, 0.1967151653460898E+0, data);
	data += add_point_6(0.2653965506227417E-3, 0.3949164710492144E+0, 0.2325404606175168E+0, data);
	data += add_point_6(0.2710857216747087E-3, 0.4318617293970503E+0, 0.2682461141151439E+0, data);
	data += add_point_6(0.2754434093903659E-3, 0.4677221009931678E+0, 0.3035720116011973E+0, data);
	data += add_point_6(0.2786579932519380E-3, 0.5023417939270955E+0, 0.3382781859197439E+0, data);
	data += add_point_6(0.2809011080679474E-3, 0.5355701836636128E+0, 0.3721383065625942E+0, data);
	data += add_point_6(0.2823336184560987E-3, 0.5672608451328771E+0, 0.4049346360466055E+0, data);
	data += add_point_6(0.2831101175806309E-3, 0.5972704202540162E+0, 0.4364538098633802E+0, data);
	data += add_point_6(0.2221679970354546E-3, 0.2461687022333596E+0, 0.3070423166833368E-1, data);
	data += add_point_6(0.2356185734270703E-3, 0.2881774566286831E+0, 0.6338034669281885E-1, data);
	data += add_point_6(0.2469228344805590E-3, 0.3293963604116978E+0, 0.9742862487067941E-1, data);
	data += add_point_6(0.2562726348642046E-3, 0.3697303822241377E+0, 0.1323799532282290E+0, data);
	data += add_point_6(0.2638756726753028E-3, 0.4090663023135127E+0, 0.1678497018129336E+0, data);
	data += add_point_6(0.2699311157390862E-3, 0.4472819355411712E+0, 0.2035095105326114E+0, data);
	data += add_point_6(0.2746233268403837E-3, 0.4842513377231437E+0, 0.2390692566672091E+0, data);
	data += add_point_6(0.2781225674454771E-3, 0.5198477629962928E+0, 0.2742649818076149E+0, data);
	data += add_point_6(0.2805881254045684E-3, 0.5539453011883145E+0, 0.3088503806580094E+0, data);
	data += add_point_6(0.2821719877004913E-3, 0.5864196762401251E+0, 0.3425904245906614E+0, data);
	data += add_point_6(0.2830222502333124E-3, 0.6171484466668390E+0, 0.3752562294789468E+0, data);
	data += add_point_6(0.2457995956744870E-3, 0.3350337830565727E+0, 0.3261589934634747E-1, data);
	data += add_point_6(0.2551474407503706E-3, 0.3775773224758284E+0, 0.6658438928081572E-1, data);
	data += add_point_6(0.2629065335195311E-3, 0.4188155229848973E+0, 0.1014565797157954E+0, data);
	data += add_point_6(0.2691900449925075E-3, 0.4586805892009344E+0, 0.1368573320843822E+0, data);
	data += add_point_6(0.2741275485754276E-3, 0.4970895714224235E+0, 0.1724614851951608E+0, data);
	data += add_point_6(0.2778530970122595E-3, 0.5339505133960747E+0, 0.2079779381416412E+0, data);
	data += add_point_6(0.2805010567646741E-3, 0.5691665792531440E+0, 0.2431385788322288E+0, data);
	data += add_point_6(0.2822055834031040E-3, 0.6026387682680377E+0, 0.2776901883049853E+0, data);
	data += add_point_6(0.2831016901243473E-3, 0.6342676150163307E+0, 0.3113881356386632E+0, data);
	data += add_point_6(0.2624474901131803E-3, 0.4237951119537067E+0, 0.3394877848664351E-1, data);
	data += add_point_6(0.2688034163039377E-3, 0.4656918683234929E+0, 0.6880219556291447E-1, data);
	data += add_point_6(0.2738932751287636E-3, 0.5058857069185980E+0, 0.1041946859721635E+0, data);
	data += add_point_6(0.2777944791242523E-3, 0.5443204666713996E+0, 0.1398039738736393E+0, data);
	data += add_point_6(0.2806011661660987E-3, 0.5809298813759742E+0, 0.1753373381196155E+0, data);
	data += add_point_6(0.2824181456597460E-3, 0.6156416039447128E+0, 0.2105215793514010E+0, data);
	data += add_point_6(0.2833585216577828E-3, 0.6483801351066604E+0, 0.2450953312157051E+0, data);
	data += add_point_6(0.2738165236962878E-3, 0.5103616577251688E+0, 0.3485560643800719E-1, data);
	data += add_point_6(0.2778365208203180E-3, 0.5506738792580681E+0, 0.7026308631512033E-1, data);
	data += add_point_6(0.2807852940418966E-3, 0.5889573040995292E+0, 0.1059035061296403E+0, data);
	data += add_point_6(0.2827245949674705E-3, 0.6251641589516930E+0, 0.1414823925236026E+0, data);
	data += add_point_6(0.2837342344829828E-3, 0.6592414921570178E+0, 0.1767207908214530E+0, data);
	data += add_point_6(0.2809233907610981E-3, 0.5930314017533384E+0, 0.3542189339561672E-1, data);
	data += add_point_6(0.2829930809742694E-3, 0.6309812253390175E+0, 0.7109574040369549E-1, data);
	data += add_point_6(0.2841097874111479E-3, 0.6666296011353230E+0, 0.1067259792282730E+0, data);
	data += add_point_6(0.2843455206008783E-3, 0.6703715271049922E+0, 0.3569455268820809E-1, data);
}

static void make_4334_grid()
{
	grid_t *data = &gd4334[0];
	gd_list[28] = &gd4334[0];
	data += add_point_1(0.1449063022537883E-4, data);
	data += add_point_3(0.2546377329828424E-3, data);
	data += add_point_4(0.6018432961087496E-4, 0.1462896151831013E-1, data);
	data += add_point_4(0.1002286583263673E-3, 0.3769840812493139E-1, data);
	data += add_point_4(0.1315222931028093E-3, 0.6524701904096891E-1, data);
	data += add_point_4(0.1564213746876724E-3, 0.9560543416134648E-1, data);
	data += add_point_4(0.1765118841507736E-3, 0.1278335898929198E+0, data);
	data += add_point_4(0.1928737099311080E-3, 0.1613096104466031E+0, data);
	data += add_point_4(0.2062658534263270E-3, 0.1955806225745371E+0, data);
	data += add_point_4(0.2172395445953787E-3, 0.2302935218498028E+0, data);
	data += add_point_4(0.2262076188876047E-3, 0.2651584344113027E+0, data);
	data += add_point_4(0.2334885699462397E-3, 0.2999276825183209E+0, data);
	data += add_point_4(0.2393355273179203E-3, 0.3343828669718798E+0, data);
	data += add_point_4(0.2439559200468863E-3, 0.3683265013750518E+0, data);
	data += add_point_4(0.2475251866060002E-3, 0.4015763206518108E+0, data);
	data += add_point_4(0.2501965558158773E-3, 0.4339612026399770E+0, data);
	data += add_point_4(0.2521081407925925E-3, 0.4653180651114582E+0, data);
	data += add_point_4(0.2533881002388081E-3, 0.4954893331080803E+0, data);
	data += add_point_4(0.2541582900848261E-3, 0.5243207068924930E+0, data);
	data += add_point_4(0.2545365737525860E-3, 0.5516590479041704E+0, data);
	data += add_point_4(0.2545726993066799E-3, 0.6012371927804176E+0, data);
	data += add_point_4(0.2544456197465555E-3, 0.6231574466449819E+0, data);
	data += add_point_4(0.2543481596881064E-3, 0.6429416514181271E+0, data);
	data += add_point_4(0.2543506451429194E-3, 0.6604124272943595E+0, data);
	data += add_point_4(0.2544905675493763E-3, 0.6753851470408250E+0, data);
	data += add_point_4(0.2547611407344429E-3, 0.6876717970626160E+0, data);
	data += add_point_4(0.2551060375448869E-3, 0.6970895061319234E+0, data);
	data += add_point_4(0.2554291933816039E-3, 0.7034746912553310E+0, data);
	data += add_point_4(0.2556255710686343E-3, 0.7067017217542295E+0, data);
	data += add_point_5(0.9041339695118195E-4, 0.4382223501131123E-1, data);
	data += add_point_5(0.1438426330079022E-3, 0.1117474077400006E+0, data);
	data += add_point_5(0.1802523089820518E-3, 0.1897153252911440E+0, data);
	data += add_point_5(0.2060052290565496E-3, 0.2724023009910331E+0, data);
	data += add_point_5(0.2245002248967466E-3, 0.3567163308709902E+0, data);
	data += add_point_5(0.2377059847731150E-3, 0.4404784483028087E+0, data);
	data += add_point_5(0.2468118955882525E-3, 0.5219833154161411E+0, data);
	data += add_point_5(0.2525410872966528E-3, 0.5998179868977553E+0, data);
	data += add_point_5(0.2553101409933397E-3, 0.6727803154548222E+0, data);
	data += add_point_6(0.1212879733668632E-3, 0.7476563943166086E-1, 0.2193168509461185E-1, data);
	data += add_point_6(0.1472872881270931E-3, 0.1075341482001416E+0, 0.4826419281533887E-1, data);
	data += add_point_6(0.1686846601010828E-3, 0.1416344885203259E+0, 0.7751191883575742E-1, data);
	data += add_point_6(0.1862698414660208E-3, 0.1766325315388586E+0, 0.1087558139247680E+0, data);
	data += add_point_6(0.2007430956991861E-3, 0.2121744174481514E+0, 0.1413661374253096E+0, data);
	data += add_point_6(0.2126568125394796E-3, 0.2479669443408145E+0, 0.1748768214258880E+0, data);
	data += add_point_6(0.2224394603372113E-3, 0.2837600452294113E+0, 0.2089216406612073E+0, data);
	data += add_point_6(0.2304264522673135E-3, 0.3193344933193984E+0, 0.2431987685545972E+0, data);
	data += add_point_6(0.2368854288424087E-3, 0.3544935442438745E+0, 0.2774497054377770E+0, data);
	data += add_point_6(0.2420352089461772E-3, 0.3890571932288154E+0, 0.3114460356156915E+0, data);
	data += add_point_6(0.2460597113081295E-3, 0.4228581214259090E+0, 0.3449806851913012E+0, data);
	data += add_point_6(0.2491181912257687E-3, 0.4557387211304052E+0, 0.3778618641248256E+0, data);
	data += add_point_6(0.2513528194205857E-3, 0.4875487950541643E+0, 0.4099086391698978E+0, data);
	data += add_point_6(0.2528943096693220E-3, 0.5181436529962997E+0, 0.4409474925853973E+0, data);
	data += add_point_6(0.2538660368488136E-3, 0.5473824095600661E+0, 0.4708094517711291E+0, data);
	data += add_point_6(0.2543868648299022E-3, 0.5751263398976174E+0, 0.4993275140354637E+0, data);
	data += add_point_6(0.1642595537825183E-3, 0.1489515746840028E+0, 0.2599381993267017E-1, data);
	data += add_point_6(0.1818246659849308E-3, 0.1863656444351767E+0, 0.5479286532462190E-1, data);
	data += add_point_6(0.1966565649492420E-3, 0.2238602880356348E+0, 0.8556763251425254E-1, data);
	data += add_point_6(0.2090677905657991E-3, 0.2612723375728160E+0, 0.1177257802267011E+0, data);
	data += add_point_6(0.2193820409510504E-3, 0.2984332990206190E+0, 0.1508168456192700E+0, data);
	data += add_point_6(0.2278870827661928E-3, 0.3351786584663333E+0, 0.1844801892177727E+0, data);
	data += add_point_6(0.2348283192282090E-3, 0.3713505522209120E+0, 0.2184145236087598E+0, data);
	data += add_point_6(0.2404139755581477E-3, 0.4067981098954663E+0, 0.2523590641486229E+0, data);
	data += add_point_6(0.2448227407760734E-3, 0.4413769993687534E+0, 0.2860812976901373E+0, data);
	data += add_point_6(0.2482110455592573E-3, 0.4749487182516394E+0, 0.3193686757808996E+0, data);
	data += add_point_6(0.2507192397774103E-3, 0.5073798105075426E+0, 0.3520226949547602E+0, data);
	data += add_point_6(0.2524765968534880E-3, 0.5385410448878654E+0, 0.3838544395667890E+0, data);
	data += add_point_6(0.2536052388539425E-3, 0.5683065353670530E+0, 0.4146810037640963E+0, data);
	data += add_point_6(0.2542230588033068E-3, 0.5965527620663510E+0, 0.4443224094681121E+0, data);
	data += add_point_6(0.1944817013047896E-3, 0.2299227700856157E+0, 0.2865757664057584E-1, data);
	data += add_point_6(0.2067862362746635E-3, 0.2695752998553267E+0, 0.5923421684485993E-1, data);
	data += add_point_6(0.2172440734649114E-3, 0.3086178716611389E+0, 0.9117817776057715E-1, data);
	data += add_point_6(0.2260125991723423E-3, 0.3469649871659077E+0, 0.1240593814082605E+0, data);
	data += add_point_6(0.2332655008689523E-3, 0.3845153566319655E+0, 0.1575272058259175E+0, data);
	data += add_point_6(0.2391699681532458E-3, 0.4211600033403215E+0, 0.1912845163525413E+0, data);
	data += add_point_6(0.2438801528273928E-3, 0.4567867834329882E+0, 0.2250710177858171E+0, data);
	data += add_point_6(0.2475370504260665E-3, 0.4912829319232061E+0, 0.2586521303440910E+0, data);
	data += add_point_6(0.2502707235640574E-3, 0.5245364793303812E+0, 0.2918112242865407E+0, data);
	data += add_point_6(0.2522031701054241E-3, 0.5564369788915756E+0, 0.3243439239067890E+0, data);
	data += add_point_6(0.2534511269978784E-3, 0.5868757697775287E+0, 0.3560536787835351E+0, data);
	data += add_point_6(0.2541284914955151E-3, 0.6157458853519617E+0, 0.3867480821242581E+0, data);
	data += add_point_6(0.2161509250688394E-3, 0.3138461110672113E+0, 0.3051374637507278E-1, data);
	data += add_point_6(0.2248778513437852E-3, 0.3542495872050569E+0, 0.6237111233730755E-1, data);
	data += add_point_6(0.2322388803404617E-3, 0.3935751553120181E+0, 0.9516223952401907E-1, data);
	data += add_point_6(0.2383265471001355E-3, 0.4317634668111147E+0, 0.1285467341508517E+0, data);
	data += add_point_6(0.2432476675019525E-3, 0.4687413842250821E+0, 0.1622318931656033E+0, data);
	data += add_point_6(0.2471122223750674E-3, 0.5044274237060283E+0, 0.1959581153836453E+0, data);
	data += add_point_6(0.2500291752486870E-3, 0.5387354077925727E+0, 0.2294888081183837E+0, data);
	data += add_point_6(0.2521055942764682E-3, 0.5715768898356105E+0, 0.2626031152713945E+0, data);
	data += add_point_6(0.2534472785575503E-3, 0.6028627200136111E+0, 0.2950904075286713E+0, data);
	data += add_point_6(0.2541599713080121E-3, 0.6325039812653463E+0, 0.3267458451113286E+0, data);
	data += add_point_6(0.2317380975862936E-3, 0.3981986708423407E+0, 0.3183291458749821E-1, data);
	data += add_point_6(0.2378550733719775E-3, 0.4382791182133300E+0, 0.6459548193880908E-1, data);
	data += add_point_6(0.2428884456739118E-3, 0.4769233057218166E+0, 0.9795757037087952E-1, data);
	data += add_point_6(0.2469002655757292E-3, 0.5140823911194238E+0, 0.1316307235126655E+0, data);
	data += add_point_6(0.2499657574265851E-3, 0.5496977833862983E+0, 0.1653556486358704E+0, data);
	data += add_point_6(0.2521676168486082E-3, 0.5837047306512727E+0, 0.1988931724126510E+0, data);
	data += add_point_6(0.2535935662645334E-3, 0.6160349566926879E+0, 0.2320174581438950E+0, data);
	data += add_point_6(0.2543356743363214E-3, 0.6466185353209440E+0, 0.2645106562168662E+0, data);
	data += add_point_6(0.2427353285201535E-3, 0.4810835158795404E+0, 0.3275917807743992E-1, data);
	data += add_point_6(0.2468258039744386E-3, 0.5199925041324341E+0, 0.6612546183967181E-1, data);
	data += add_point_6(0.2500060956440310E-3, 0.5571717692207494E+0, 0.9981498331474143E-1, data);
	data += add_point_6(0.2523238365420979E-3, 0.5925789250836378E+0, 0.1335687001410374E+0, data);
	data += add_point_6(0.2538399260252846E-3, 0.6261658523859670E+0, 0.1671444402896463E+0, data);
	data += add_point_6(0.2546255927268069E-3, 0.6578811126669331E+0, 0.2003106382156076E+0, data);
	data += add_point_6(0.2500583360048449E-3, 0.5609624612998100E+0, 0.3337500940231335E-1, data);
	data += add_point_6(0.2524777638260203E-3, 0.5979959659984670E+0, 0.6708750335901803E-1, data);
	data += add_point_6(0.2540951193860656E-3, 0.6330523711054002E+0, 0.1008792126424850E+0, data);
	data += add_point_6(0.2549524085027472E-3, 0.6660960998103972E+0, 0.1345050343171794E+0, data);
	data += add_point_6(0.2542569507009158E-3, 0.6365384364585819E+0, 0.3372799460737052E-1, data);
	data += add_point_6(0.2552114127580376E-3, 0.6710994302899275E+0, 0.6755249309678028E-1, data);
}

static void make_4802_grid()
{
	grid_t *data = &gd4802[0];
	gd_list[29] = &gd4802[0];
	data += add_point_1(0.9687521879420705E-4, data);
	data += add_point_2(0.2307897895367918E-3, data);
	data += add_point_3(0.2297310852498558E-3, data);
	data += add_point_4(0.7386265944001919E-4, 0.2335728608887064E-1, data);
	data += add_point_4(0.8257977698542210E-4, 0.4352987836550653E-1, data);
	data += add_point_4(0.9706044762057630E-4, 0.6439200521088801E-1, data);
	data += add_point_4(0.1302393847117003E-3, 0.9003943631993181E-1, data);
	data += add_point_4(0.1541957004600968E-3, 0.1196706615548473E+0, data);
	data += add_point_4(0.1704459770092199E-3, 0.1511715412838134E+0, data);
	data += add_point_4(0.1827374890942906E-3, 0.1835982828503801E+0, data);
	data += add_point_4(0.1926360817436107E-3, 0.2165081259155405E+0, data);
	data += add_point_4(0.2008010239494833E-3, 0.2496208720417563E+0, data);
	data += add_point_4(0.2075635983209175E-3, 0.2827200673567900E+0, data);
	data += add_point_4(0.2131306638690909E-3, 0.3156190823994346E+0, data);
	data += add_point_4(0.2176562329937335E-3, 0.3481476793749115E+0, data);
	data += add_point_4(0.2212682262991018E-3, 0.3801466086947226E+0, data);
	data += add_point_4(0.2240799515668565E-3, 0.4114652119634011E+0, data);
	data += add_point_4(0.2261959816187525E-3, 0.4419598786519751E+0, data);
	data += add_point_4(0.2277156368808855E-3, 0.4714925949329543E+0, data);
	data += add_point_4(0.2287351772128336E-3, 0.4999293972879466E+0, data);
	data += add_point_4(0.2293490814084085E-3, 0.5271387221431248E+0, data);
	data += add_point_4(0.2296505312376273E-3, 0.5529896780837761E+0, data);
	data += add_point_4(0.2296793832318756E-3, 0.6000856099481712E+0, data);
	data += add_point_4(0.2295785443842974E-3, 0.6210562192785175E+0, data);
	data += add_point_4(0.2295017931529102E-3, 0.6401165879934240E+0, data);
	data += add_point_4(0.2295059638184868E-3, 0.6571144029244334E+0, data);
	data += add_point_4(0.2296232343237362E-3, 0.6718910821718863E+0, data);
	data += add_point_4(0.2298530178740771E-3, 0.6842845591099010E+0, data);
	data += add_point_4(0.2301579790280501E-3, 0.6941353476269816E+0, data);
	data += add_point_4(0.2304690404996513E-3, 0.7012965242212991E+0, data);
	data += add_point_4(0.2307027995907102E-3, 0.7056471428242644E+0, data);
	data += add_point_5(0.9312274696671092E-4, 0.4595557643585895E-1, data);
	data += add_point_5(0.1199919385876926E-3, 0.1049316742435023E+0, data);
	data += add_point_5(0.1598039138877690E-3, 0.1773548879549274E+0, data);
	data += add_point_5(0.1822253763574900E-3, 0.2559071411236127E+0, data);
	data += add_point_5(0.1988579593655040E-3, 0.3358156837985898E+0, data);
	data += add_point_5(0.2112620102533307E-3, 0.4155835743763893E+0, data);
	data += add_point_5(0.2201594887699007E-3, 0.4937894296167472E+0, data);
	data += add_point_5(0.2261622590895036E-3, 0.5691569694793316E+0, data);
	data += add_point_5(0.2296458453435705E-3, 0.6405840854894251E+0, data);
	data += add_point_6(0.1006006990267000E-3, 0.7345133894143348E-1, 0.2177844081486067E-1, data);
	data += add_point_6(0.1227676689635876E-3, 0.1009859834044931E+0, 0.4590362185775188E-1, data);
	data += add_point_6(0.1467864280270117E-3, 0.1324289619748758E+0, 0.7255063095690877E-1, data);
	data += add_point_6(0.1644178912101232E-3, 0.1654272109607127E+0, 0.1017825451960684E+0, data);
	data += add_point_6(0.1777664890718961E-3, 0.1990767186776461E+0, 0.1325652320980364E+0, data);
	data += add_point_6(0.1884825664516690E-3, 0.2330125945523278E+0, 0.1642765374496765E+0, data);
	data += add_point_6(0.1973269246453848E-3, 0.2670080611108287E+0, 0.1965360374337889E+0, data);
	data += add_point_6(0.2046767775855328E-3, 0.3008753376294316E+0, 0.2290726770542238E+0, data);
	data += add_point_6(0.2107600125918040E-3, 0.3344475596167860E+0, 0.2616645495370823E+0, data);
	data += add_point_6(0.2157416362266829E-3, 0.3675709724070786E+0, 0.2941150728843141E+0, data);
	data += add_point_6(0.2197557816920721E-3, 0.4001000887587812E+0, 0.3262440400919066E+0, data);
	data += add_point_6(0.2229192611835437E-3, 0.4318956350436028E+0, 0.3578835350611916E+0, data);
	data += add_point_6(0.2253385110212775E-3, 0.4628239056795531E+0, 0.3888751854043678E+0, data);
	data += add_point_6(0.2271137107548774E-3, 0.4927563229773636E+0, 0.4190678003222840E+0, data);
	data += add_point_6(0.2283414092917525E-3, 0.5215687136707969E+0, 0.4483151836883852E+0, data);
	data += add_point_6(0.2291161673130077E-3, 0.5491402346984905E+0, 0.4764740676087880E+0, data);
	data += add_point_6(0.2295313908576598E-3, 0.5753520160126075E+0, 0.5034021310998277E+0, data);
	data += add_point_6(0.1438204721359031E-3, 0.1388326356417754E+0, 0.2435436510372806E-1, data);
	data += add_point_6(0.1607738025495257E-3, 0.1743686900537244E+0, 0.5118897057342652E-1, data);
	data += add_point_6(0.1741483853528379E-3, 0.2099737037950268E+0, 0.8014695048539634E-1, data);
	data += add_point_6(0.1851918467519151E-3, 0.2454492590908548E+0, 0.1105117874155699E+0, data);
	data += add_point_6(0.1944628638070613E-3, 0.2807219257864278E+0, 0.1417950531570966E+0, data);
	data += add_point_6(0.2022495446275152E-3, 0.3156842271975842E+0, 0.1736604945719597E+0, data);
	data += add_point_6(0.2087462382438514E-3, 0.3502090945177752E+0, 0.2058466324693981E+0, data);
	data += add_point_6(0.2141074754818308E-3, 0.3841684849519686E+0, 0.2381284261195919E+0, data);
	data += add_point_6(0.2184640913748162E-3, 0.4174372367906016E+0, 0.2703031270422569E+0, data);
	data += add_point_6(0.2219309165220329E-3, 0.4498926465011892E+0, 0.3021845683091309E+0, data);
	data += add_point_6(0.2246123118340624E-3, 0.4814146229807701E+0, 0.3335993355165720E+0, data);
	data += add_point_6(0.2266062766915125E-3, 0.5118863625734701E+0, 0.3643833735518232E+0, data);
	data += add_point_6(0.2280072952230796E-3, 0.5411947455119144E+0, 0.3943789541958179E+0, data);
	data += add_point_6(0.2289082025202583E-3, 0.5692301500357246E+0, 0.4234320144403542E+0, data);
	data += add_point_6(0.2294012695120025E-3, 0.5958857204139576E+0, 0.4513897947419260E+0, data);
	data += add_point_6(0.1722434488736947E-3, 0.2156270284785766E+0, 0.2681225755444491E-1, data);
	data += add_point_6(0.1830237421455091E-3, 0.2532385054909710E+0, 0.5557495747805614E-1, data);
	data += add_point_6(0.1923855349997633E-3, 0.2902564617771537E+0, 0.8569368062950249E-1, data);
	data += add_point_6(0.2004067861936271E-3, 0.3266979823143256E+0, 0.1167367450324135E+0, data);
	data += add_point_6(0.2071817297354263E-3, 0.3625039627493614E+0, 0.1483861994003304E+0, data);
	data += add_point_6(0.2128250834102103E-3, 0.3975838937548699E+0, 0.1803821503011405E+0, data);
	data += add_point_6(0.2174513719440102E-3, 0.4318396099009774E+0, 0.2124962965666424E+0, data);
	data += add_point_6(0.2211661839150214E-3, 0.4651706555732742E+0, 0.2445221837805913E+0, data);
	data += add_point_6(0.2240665257813102E-3, 0.4974752649620969E+0, 0.2762701224322987E+0, data);
	data += add_point_6(0.2262439516632620E-3, 0.5286517579627517E+0, 0.3075627775211328E+0, data);
	data += add_point_6(0.2277874557231869E-3, 0.5586001195731895E+0, 0.3382311089826877E+0, data);
	data += add_point_6(0.2287854314454994E-3, 0.5872229902021319E+0, 0.3681108834741399E+0, data);
	data += add_point_6(0.2293268499615575E-3, 0.6144258616235123E+0, 0.3970397446872839E+0, data);
	data += add_point_6(0.1912628201529828E-3, 0.2951676508064861E+0, 0.2867499538750441E-1, data);
	data += add_point_6(0.1992499672238701E-3, 0.3335085485472725E+0, 0.5867879341903510E-1, data);
	data += add_point_6(0.2061275533454027E-3, 0.3709561760636381E+0, 0.8961099205022284E-1, data);
	data += add_point_6(0.2119318215968572E-3, 0.4074722861667498E+0, 0.1211627927626297E+0, data);
	data += add_point_6(0.2167416581882652E-3, 0.4429923648839117E+0, 0.1530748903554898E+0, data);
	data += add_point_6(0.2206430730516600E-3, 0.4774428052721736E+0, 0.1851176436721877E+0, data);
	data += add_point_6(0.2237186938699523E-3, 0.5107446539535904E+0, 0.2170829107658179E+0, data);
	data += add_point_6(0.2260480075032884E-3, 0.5428151370542935E+0, 0.2487786689026271E+0, data);
	data += add_point_6(0.2277098884558542E-3, 0.5735699292556964E+0, 0.2800239952795016E+0, data);
	data += add_point_6(0.2287845715109671E-3, 0.6029253794562866E+0, 0.3106445702878119E+0, data);
	data += add_point_6(0.2293547268236294E-3, 0.6307998987073145E+0, 0.3404689500841194E+0, data);
	data += add_point_6(0.2056073839852528E-3, 0.3752652273692719E+0, 0.2997145098184479E-1, data);
	data += add_point_6(0.2114235865831876E-3, 0.4135383879344028E+0, 0.6086725898678011E-1, data);
	data += add_point_6(0.2163175629770551E-3, 0.4506113885153907E+0, 0.9238849548435643E-1, data);
	data += add_point_6(0.2203392158111650E-3, 0.4864401554606072E+0, 0.1242786603851851E+0, data);
	data += add_point_6(0.2235473176847839E-3, 0.5209708076611709E+0, 0.1563086731483386E+0, data);
	data += add_point_6(0.2260024141501235E-3, 0.5541422135830122E+0, 0.1882696509388506E+0, data);
	data += add_point_6(0.2277675929329182E-3, 0.5858880915113817E+0, 0.2199672979126059E+0, data);
	data += add_point_6(0.2289102112284834E-3, 0.6161399390603444E+0, 0.2512165482924867E+0, data);
	data += add_point_6(0.2295027954625118E-3, 0.6448296482255090E+0, 0.2818368701871888E+0, data);
	data += add_point_6(0.2161281589879992E-3, 0.4544796274917948E+0, 0.3088970405060312E-1, data);
	data += add_point_6(0.2201980477395102E-3, 0.4919389072146628E+0, 0.6240947677636835E-1, data);
	data += add_point_6(0.2234952066593166E-3, 0.5279313026985183E+0, 0.9430706144280313E-1, data);
	data += add_point_6(0.2260540098520838E-3, 0.5624169925571135E+0, 0.1263547818770374E+0, data);
	data += add_point_6(0.2279157981899988E-3, 0.5953484627093287E+0, 0.1583430788822594E+0, data);
	data += add_point_6(0.2291296918565571E-3, 0.6266730715339185E+0, 0.1900748462555988E+0, data);
	data += add_point_6(0.2297533752536649E-3, 0.6563363204278871E+0, 0.2213599519592567E+0, data);
	data += add_point_6(0.2234927356465995E-3, 0.5314574716585696E+0, 0.3152508811515374E-1, data);
	data += add_point_6(0.2261288012985219E-3, 0.5674614932298185E+0, 0.6343865291465561E-1, data);
	data += add_point_6(0.2280818160923688E-3, 0.6017706004970264E+0, 0.9551503504223951E-1, data);
	data += add_point_6(0.2293773295180159E-3, 0.6343471270264178E+0, 0.1275440099801196E+0, data);
	data += add_point_6(0.2300528767338634E-3, 0.6651494599127802E+0, 0.1593252037671960E+0, data);
	data += add_point_6(0.2281893855065666E-3, 0.6050184986005704E+0, 0.3192538338496105E-1, data);
	data += add_point_6(0.2295720444840727E-3, 0.6390163550880400E+0, 0.6402824353962306E-1, data);
	data += add_point_6(0.2303227649026753E-3, 0.6711199107088448E+0, 0.9609805077002909E-1, data);
	data += add_point_6(0.2304831913227114E-3, 0.6741354429572275E+0, 0.3211853196273233E-1, data);
}

static void make_5294_grid()
{
	grid_t *data = &gd5294[0];
	gd_list[30] = &gd5294[0];
	data += add_point_1(0.9080510764308163E-4, data);
	data += add_point_3(0.2084824361987793E-3, data);
	data += add_point_4(0.5011105657239616E-4, 0.2303261686261450E-1, data);
	data += add_point_4(0.5942520409683854E-4, 0.3757208620162394E-1, data);
	data += add_point_4(0.9564394826109721E-4, 0.5821912033821852E-1, data);
	data += add_point_4(0.1185530657126338E-3, 0.8403127529194872E-1, data);
	data += add_point_4(0.1364510114230331E-3, 0.1122927798060578E+0, data);
	data += add_point_4(0.1505828825605415E-3, 0.1420125319192987E+0, data);
	data += add_point_4(0.1619298749867023E-3, 0.1726396437341978E+0, data);
	data += add_point_4(0.1712450504267789E-3, 0.2038170058115696E+0, data);
	data += add_point_4(0.1789891098164999E-3, 0.2352849892876508E+0, data);
	data += add_point_4(0.1854474955629795E-3, 0.2668363354312461E+0, data);
	data += add_point_4(0.1908148636673661E-3, 0.2982941279900452E+0, data);
	data += add_point_4(0.1952377405281833E-3, 0.3295002922087076E+0, data);
	data += add_point_4(0.1988349254282232E-3, 0.3603094918363593E+0, data);
	data += add_point_4(0.2017079807160050E-3, 0.3905857895173920E+0, data);
	data += add_point_4(0.2039473082709094E-3, 0.4202005758160837E+0, data);
	data += add_point_4(0.2056360279288953E-3, 0.4490310061597227E+0, data);
	data += add_point_4(0.2068525823066865E-3, 0.4769586160311491E+0, data);
	data += add_point_4(0.2076724877534488E-3, 0.5038679887049750E+0, data);
	data += add_point_4(0.2081694278237885E-3, 0.5296454286519961E+0, data);
	data += add_point_4(0.2084157631219326E-3, 0.5541776207164850E+0, data);
	data += add_point_4(0.2084381531128593E-3, 0.5990467321921213E+0, data);
	data += add_point_4(0.2083476277129307E-3, 0.6191467096294587E+0, data);
	data += add_point_4(0.2082686194459732E-3, 0.6375251212901849E+0, data);
	data += add_point_4(0.2082475686112415E-3, 0.6540514381131168E+0, data);
	data += add_point_4(0.2083139860289915E-3, 0.6685899064391510E+0, data);
	data += add_point_4(0.2084745561831237E-3, 0.6810013009681648E+0, data);
	data += add_point_4(0.2087091313375890E-3, 0.6911469578730340E+0, data);
	data += add_point_4(0.2089718413297697E-3, 0.6988956915141736E+0, data);
	data += add_point_4(0.2092003303479793E-3, 0.7041335794868720E+0, data);
	data += add_point_4(0.2093336148263241E-3, 0.7067754398018567E+0, data);
	data += add_point_5(0.7591708117365267E-4, 0.3840368707853623E-1, data);
	data += add_point_5(0.1083383968169186E-3, 0.9835485954117399E-1, data);
	data += add_point_5(0.1403019395292510E-3, 0.1665774947612998E+0, data);
	data += add_point_5(0.1615970179286436E-3, 0.2405702335362910E+0, data);
	data += add_point_5(0.1771144187504911E-3, 0.3165270770189046E+0, data);
	data += add_point_5(0.1887760022988168E-3, 0.3927386145645443E+0, data);
	data += add_point_5(0.1973474670768214E-3, 0.4678825918374656E+0, data);
	data += add_point_5(0.2033787661234659E-3, 0.5408022024266935E+0, data);
	data += add_point_5(0.2072343626517331E-3, 0.6104967445752438E+0, data);
	data += add_point_5(0.2091177834226918E-3, 0.6760910702685738E+0, data);
	data += add_point_6(0.9316684484675566E-4, 0.6655644120217392E-1, 0.1936508874588424E-1, data);
	data += add_point_6(0.1116193688682976E-3, 0.9446246161270182E-1, 0.4252442002115869E-1, data);
	data += add_point_6(0.1298623551559414E-3, 0.1242651925452509E+0, 0.6806529315354374E-1, data);
	data += add_point_6(0.1450236832456426E-3, 0.1553438064846751E+0, 0.9560957491205369E-1, data);
	data += add_point_6(0.1572719958149914E-3, 0.1871137110542670E+0, 0.1245931657452888E+0, data);
	data += add_point_6(0.1673234785867195E-3, 0.2192612628836257E+0, 0.1545385828778978E+0, data);
	data += add_point_6(0.1756860118725188E-3, 0.2515682807206955E+0, 0.1851004249723368E+0, data);
	data += add_point_6(0.1826776290439367E-3, 0.2838535866287290E+0, 0.2160182608272384E+0, data);
	data += add_point_6(0.1885116347992865E-3, 0.3159578817528521E+0, 0.2470799012277111E+0, data);
	data += add_point_6(0.1933457860170574E-3, 0.3477370882791392E+0, 0.2781014208986402E+0, data);
	data += add_point_6(0.1973060671902064E-3, 0.3790576960890540E+0, 0.3089172523515731E+0, data);
	data += add_point_6(0.2004987099616311E-3, 0.4097938317810200E+0, 0.3393750055472244E+0, data);
	data += add_point_6(0.2030170909281499E-3, 0.4398256572859637E+0, 0.3693322470987730E+0, data);
	data += add_point_6(0.2049461460119080E-3, 0.4690384114718480E+0, 0.3986541005609877E+0, data);
	data += add_point_6(0.2063653565200186E-3, 0.4973216048301053E+0, 0.4272112491408562E+0, data);
	data += add_point_6(0.2073507927381027E-3, 0.5245681526132446E+0, 0.4548781735309936E+0, data);
	data += add_point_6(0.2079764593256122E-3, 0.5506733911803888E+0, 0.4815315355023251E+0, data);
	data += add_point_6(0.2083150534968778E-3, 0.5755339829522475E+0, 0.5070486445801855E+0, data);
	data += add_point_6(0.1262715121590664E-3, 0.1305472386056362E+0, 0.2284970375722366E-1, data);
	data += add_point_6(0.1414386128545972E-3, 0.1637327908216477E+0, 0.4812254338288384E-1, data);
	data += add_point_6(0.1538740401313898E-3, 0.1972734634149637E+0, 0.7531734457511935E-1, data);
	data += add_point_6(0.1642434942331432E-3, 0.2308694653110130E+0, 0.1039043639882017E+0, data);
	data += add_point_6(0.1729790609237496E-3, 0.2643899218338160E+0, 0.1334526587117626E+0, data);
	data += add_point_6(0.1803505190260828E-3, 0.2977171599622171E+0, 0.1636414868936382E+0, data);
	data += add_point_6(0.1865475350079657E-3, 0.3307293903032310E+0, 0.1942195406166568E+0, data);
	data += add_point_6(0.1917182669679069E-3, 0.3633069198219073E+0, 0.2249752879943753E+0, data);
	data += add_point_6(0.1959851709034382E-3, 0.3953346955922727E+0, 0.2557218821820032E+0, data);
	data += add_point_6(0.1994529548117882E-3, 0.4267018394184914E+0, 0.2862897925213193E+0, data);
	data += add_point_6(0.2022138911146548E-3, 0.4573009622571704E+0, 0.3165224536636518E+0, data);
	data += add_point_6(0.2043518024208592E-3, 0.4870279559856109E+0, 0.3462730221636496E+0, data);
	data += add_point_6(0.2059450313018110E-3, 0.5157819581450322E+0, 0.3754016870282835E+0, data);
	data += add_point_6(0.2070685715318472E-3, 0.5434651666465393E+0, 0.4037733784993613E+0, data);
	data += add_point_6(0.2077955310694373E-3, 0.5699823887764627E+0, 0.4312557784139123E+0, data);
	data += add_point_6(0.2081980387824712E-3, 0.5952403350947741E+0, 0.4577175367122110E+0, data);
	data += add_point_6(0.1521318610377956E-3, 0.2025152599210369E+0, 0.2520253617719557E-1, data);
	data += add_point_6(0.1622772720185755E-3, 0.2381066653274425E+0, 0.5223254506119000E-1, data);
	data += add_point_6(0.1710498139420709E-3, 0.2732823383651612E+0, 0.8060669688588620E-1, data);
	data += add_point_6(0.1785911149448736E-3, 0.3080137692611118E+0, 0.1099335754081255E+0, data);
	data += add_point_6(0.1850125313687736E-3, 0.3422405614587601E+0, 0.1399120955959857E+0, data);
	data += add_point_6(0.1904229703933298E-3, 0.3758808773890420E+0, 0.1702977801651705E+0, data);
	data += add_point_6(0.1949259956121987E-3, 0.4088458383438932E+0, 0.2008799256601680E+0, data);
	data += add_point_6(0.1986161545363960E-3, 0.4410450550841152E+0, 0.2314703052180836E+0, data);
	data += add_point_6(0.2015790585641370E-3, 0.4723879420561312E+0, 0.2618972111375892E+0, data);
	data += add_point_6(0.2038934198707418E-3, 0.5027843561874343E+0, 0.2920013195600270E+0, data);
	data += add_point_6(0.2056334060538251E-3, 0.5321453674452458E+0, 0.3216322555190551E+0, data);
	data += add_point_6(0.2068705959462289E-3, 0.5603839113834030E+0, 0.3506456615934198E+0, data);
	data += add_point_6(0.2076753906106002E-3, 0.5874150706875146E+0, 0.3789007181306267E+0, data);
	data += add_point_6(0.2081179391734803E-3, 0.6131559381660038E+0, 0.4062580170572782E+0, data);
	data += add_point_6(0.1700345216228943E-3, 0.2778497016394506E+0, 0.2696271276876226E-1, data);
	data += add_point_6(0.1774906779990410E-3, 0.3143733562261912E+0, 0.5523469316960465E-1, data);
	data += add_point_6(0.1839659377002642E-3, 0.3501485810261827E+0, 0.8445193201626464E-1, data);
	data += add_point_6(0.1894987462975169E-3, 0.3851430322303653E+0, 0.1143263119336083E+0, data);
	data += add_point_6(0.1941548809452595E-3, 0.4193013979470415E+0, 0.1446177898344475E+0, data);
	data += add_point_6(0.1980078427252384E-3, 0.4525585960458567E+0, 0.1751165438438091E+0, data);
	data += add_point_6(0.2011296284744488E-3, 0.4848447779622947E+0, 0.2056338306745660E+0, data);
	data += add_point_6(0.2035888456966776E-3, 0.5160871208276894E+0, 0.2359965487229226E+0, data);
	data += add_point_6(0.2054516325352142E-3, 0.5462112185696926E+0, 0.2660430223139146E+0, data);
	data += add_point_6(0.2067831033092635E-3, 0.5751425068101757E+0, 0.2956193664498032E+0, data);
	data += add_point_6(0.2076485320284876E-3, 0.6028073872853596E+0, 0.3245763905312779E+0, data);
	data += add_point_6(0.2081141439525255E-3, 0.6291338275278409E+0, 0.3527670026206972E+0, data);
	data += add_point_6(0.1834383015469222E-3, 0.3541797528439391E+0, 0.2823853479435550E-1, data);
	data += add_point_6(0.1889540591777677E-3, 0.3908234972074657E+0, 0.5741296374713106E-1, data);
	data += add_point_6(0.1936677023597375E-3, 0.4264408450107590E+0, 0.8724646633650199E-1, data);
	data += add_point_6(0.1976176495066504E-3, 0.4609949666553286E+0, 0.1175034422915616E+0, data);
	data += add_point_6(0.2008536004560983E-3, 0.4944389496536006E+0, 0.1479755652628428E+0, data);
	data += add_point_6(0.2034280351712291E-3, 0.5267194884346086E+0, 0.1784740659484352E+0, data);
	data += add_point_6(0.2053944466027758E-3, 0.5577787810220990E+0, 0.2088245700431244E+0, data);
	data += add_point_6(0.2068077642882360E-3, 0.5875563763536670E+0, 0.2388628136570763E+0, data);
	data += add_point_6(0.2077250949661599E-3, 0.6159910016391269E+0, 0.2684308928769185E+0, data);
	data += add_point_6(0.2082062440705320E-3, 0.6430219602956268E+0, 0.2973740761960252E+0, data);
	data += add_point_6(0.1934374486546626E-3, 0.4300647036213646E+0, 0.2916399920493977E-1, data);
	data += add_point_6(0.1974107010484300E-3, 0.4661486308935531E+0, 0.5898803024755659E-1, data);
	data += add_point_6(0.2007129290388658E-3, 0.5009658555287261E+0, 0.8924162698525409E-1, data);
	data += add_point_6(0.2033736947471293E-3, 0.5344824270447704E+0, 0.1197185199637321E+0, data);
	data += add_point_6(0.2054287125902493E-3, 0.5666575997416371E+0, 0.1502300756161382E+0, data);
	data += add_point_6(0.2069184936818894E-3, 0.5974457471404752E+0, 0.1806004191913564E+0, data);
	data += add_point_6(0.2078883689808782E-3, 0.6267984444116886E+0, 0.2106621764786252E+0, data);
	data += add_point_6(0.2083886366116359E-3, 0.6546664713575417E+0, 0.2402526932671914E+0, data);
	data += add_point_6(0.2006593275470817E-3, 0.5042711004437253E+0, 0.2982529203607657E-1, data);
	data += add_point_6(0.2033728426135397E-3, 0.5392127456774380E+0, 0.6008728062339922E-1, data);
	data += add_point_6(0.2055008781377608E-3, 0.5726819437668618E+0, 0.9058227674571398E-1, data);
	data += add_point_6(0.2070651783518502E-3, 0.6046469254207278E+0, 0.1211219235803400E+0, data);
	data += add_point_6(0.2080953335094320E-3, 0.6350716157434952E+0, 0.1515286404791580E+0, data);
	data += add_point_6(0.2086284998988521E-3, 0.6639177679185454E+0, 0.1816314681255552E+0, data);
	data += add_point_6(0.2055549387644668E-3, 0.5757276040972253E+0, 0.3026991752575440E-1, data);
	data += add_point_6(0.2071871850267654E-3, 0.6090265823139755E+0, 0.6078402297870770E-1, data);
	data += add_point_6(0.2082856600431965E-3, 0.6406735344387661E+0, 0.9135459984176636E-1, data);
	data += add_point_6(0.2088705858819358E-3, 0.6706397927793709E+0, 0.1218024155966590E+0, data);
	data += add_point_6(0.2083995867536322E-3, 0.6435019674426665E+0, 0.3052608357660639E-1, data);
	data += add_point_6(0.2090509712889637E-3, 0.6747218676375681E+0, 0.6112185773983089E-1, data);
}

static void make_5810_grid()
{
	grid_t *data = &gd5810[0];
	gd_list[31] = &gd5810[0];
	data += add_point_1(0.9735347946175486E-5, data);
	data += add_point_2(0.1907581241803167E-3, data);
	data += add_point_3(0.1901059546737578E-3, data);
	data += add_point_4(0.3926424538919212E-4, 0.1182361662400277E-1, data);
	data += add_point_4(0.6667905467294382E-4, 0.3062145009138958E-1, data);
	data += add_point_4(0.8868891315019135E-4, 0.5329794036834243E-1, data);
	data += add_point_4(0.1066306000958872E-3, 0.7848165532862220E-1, data);
	data += add_point_4(0.1214506743336128E-3, 0.1054038157636201E+0, data);
	data += add_point_4(0.1338054681640871E-3, 0.1335577797766211E+0, data);
	data += add_point_4(0.1441677023628504E-3, 0.1625769955502252E+0, data);
	data += add_point_4(0.1528880200826557E-3, 0.1921787193412792E+0, data);
	data += add_point_4(0.1602330623773609E-3, 0.2221340534690548E+0, data);
	data += add_point_4(0.1664102653445244E-3, 0.2522504912791132E+0, data);
	data += add_point_4(0.1715845854011323E-3, 0.2823610860679697E+0, data);
	data += add_point_4(0.1758901000133069E-3, 0.3123173966267560E+0, data);
	data += add_point_4(0.1794382485256736E-3, 0.3419847036953789E+0, data);
	data += add_point_4(0.1823238106757407E-3, 0.3712386456999758E+0, data);
	data += add_point_4(0.1846293252959976E-3, 0.3999627649876828E+0, data);
	data += add_point_4(0.1864284079323098E-3, 0.4280466458648093E+0, data);
	data += add_point_4(0.1877882694626914E-3, 0.4553844360185711E+0, data);
	data += add_point_4(0.1887716321852025E-3, 0.4818736094437834E+0, data);
	data += add_point_4(0.1894381638175673E-3, 0.5074138709260629E+0, data);
	data += add_point_4(0.1898454899533629E-3, 0.5319061304570707E+0, data);
	data += add_point_4(0.1900497929577815E-3, 0.5552514978677286E+0, data);
	data += add_point_4(0.1900671501924092E-3, 0.5981009025246183E+0, data);
	data += add_point_4(0.1899837555533510E-3, 0.6173990192228116E+0, data);
	data += add_point_4(0.1899014113156229E-3, 0.6351365239411131E+0, data);
	data += add_point_4(0.1898581257705106E-3, 0.6512010228227200E+0, data);
	data += add_point_4(0.1898804756095753E-3, 0.6654758363948120E+0, data);
	data += add_point_4(0.1899793610426402E-3, 0.6778410414853370E+0, data);
	data += add_point_4(0.1901464554844117E-3, 0.6881760887484110E+0, data);
	data += add_point_4(0.1903533246259542E-3, 0.6963645267094598E+0, data);
	data += add_point_4(0.1905556158463228E-3, 0.7023010617153579E+0, data);
	data += add_point_4(0.1907037155663528E-3, 0.7059004636628753E+0, data);
	data += add_point_5(0.5992997844249967E-4, 0.3552470312472575E-1, data);
	data += add_point_5(0.9749059382456978E-4, 0.9151176620841283E-1, data);
	data += add_point_5(0.1241680804599158E-3, 0.1566197930068980E+0, data);
	data += add_point_5(0.1437626154299360E-3, 0.2265467599271907E+0, data);
	data += add_point_5(0.1584200054793902E-3, 0.2988242318581361E+0, data);
	data += add_point_5(0.1694436550982744E-3, 0.3717482419703886E+0, data);
	data += add_point_5(0.1776617014018108E-3, 0.4440094491758889E+0, data);
	data += add_point_5(0.1836132434440077E-3, 0.5145337096756642E+0, data);
	data += add_point_5(0.1876494727075983E-3, 0.5824053672860230E+0, data);
	data += add_point_5(0.1899906535336482E-3, 0.6468283961043370E+0, data);
	data += add_point_6(0.8143252820767350E-4, 0.6095964259104373E-1, 0.1787828275342931E-1, data);
	data += add_point_6(0.9998859890887728E-4, 0.8811962270959388E-1, 0.3953888740792096E-1, data);
	data += add_point_6(0.1156199403068359E-3, 0.1165936722428831E+0, 0.6378121797722990E-1, data);
	data += add_point_6(0.1287632092635513E-3, 0.1460232857031785E+0, 0.8985890813745037E-1, data);
	data += add_point_6(0.1398378643365139E-3, 0.1761197110181755E+0, 0.1172606510576162E+0, data);
	data += add_point_6(0.1491876468417391E-3, 0.2066471190463718E+0, 0.1456102876970995E+0, data);
	data += add_point_6(0.1570855679175456E-3, 0.2374076026328152E+0, 0.1746153823011775E+0, data);
	data += add_point_6(0.1637483948103775E-3, 0.2682305474337051E+0, 0.2040383070295584E+0, data);
	data += add_point_6(0.1693500566632843E-3, 0.2989653312142369E+0, 0.2336788634003698E+0, data);
	data += add_point_6(0.1740322769393633E-3, 0.3294762752772209E+0, 0.2633632752654219E+0, data);
	data += add_point_6(0.1779126637278296E-3, 0.3596390887276086E+0, 0.2929369098051601E+0, data);
	data += add_point_6(0.1810908108835412E-3, 0.3893383046398812E+0, 0.3222592785275512E+0, data);
	data += add_point_6(0.1836529132600190E-3, 0.4184653789358347E+0, 0.3512004791195743E+0, data);
	data += add_point_6(0.1856752841777379E-3, 0.4469172319076166E+0, 0.3796385677684537E+0, data);
	data += add_point_6(0.1872270566606832E-3, 0.4745950813276976E+0, 0.4074575378263879E+0, data);
	data += add_point_6(0.1883722645591307E-3, 0.5014034601410262E+0, 0.4345456906027828E+0, data);
	data += add_point_6(0.1891714324525297E-3, 0.5272493404551239E+0, 0.4607942515205134E+0, data);
	data += add_point_6(0.1896827480450146E-3, 0.5520413051846366E+0, 0.4860961284181720E+0, data);
	data += add_point_6(0.1899628417059528E-3, 0.5756887237503077E+0, 0.5103447395342790E+0, data);
	data += add_point_6(0.1123301829001669E-3, 0.1225039430588352E+0, 0.2136455922655793E-1, data);
	data += add_point_6(0.1253698826711277E-3, 0.1539113217321372E+0, 0.4520926166137188E-1, data);
	data += add_point_6(0.1366266117678531E-3, 0.1856213098637712E+0, 0.7086468177864818E-1, data);
	data += add_point_6(0.1462736856106918E-3, 0.2174998728035131E+0, 0.9785239488772918E-1, data);
	data += add_point_6(0.1545076466685412E-3, 0.2494128336938330E+0, 0.1258106396267210E+0, data);
	data += add_point_6(0.1615096280814007E-3, 0.2812321562143480E+0, 0.1544529125047001E+0, data);
	data += add_point_6(0.1674366639741759E-3, 0.3128372276456111E+0, 0.1835433512202753E+0, data);
	data += add_point_6(0.1724225002437900E-3, 0.3441145160177973E+0, 0.2128813258619585E+0, data);
	data += add_point_6(0.1765810822987288E-3, 0.3749567714853510E+0, 0.2422913734880829E+0, data);
	data += add_point_6(0.1800104126010751E-3, 0.4052621732015610E+0, 0.2716163748391453E+0, data);
	data += add_point_6(0.1827960437331284E-3, 0.4349335453522385E+0, 0.3007127671240280E+0, data);
	data += add_point_6(0.1850140300716308E-3, 0.4638776641524965E+0, 0.3294470677216479E+0, data);
	data += add_point_6(0.1867333507394938E-3, 0.4920046410462687E+0, 0.3576932543699155E+0, data);
	data += add_point_6(0.1880178688638289E-3, 0.5192273554861704E+0, 0.3853307059757764E+0, data);
	data += add_point_6(0.1889278925654758E-3, 0.5454609081136522E+0, 0.4122425044452694E+0, data);
	data += add_point_6(0.1895213832507346E-3, 0.5706220661424140E+0, 0.4383139587781027E+0, data);
	data += add_point_6(0.1898548277397420E-3, 0.5946286755181518E+0, 0.4634312536300553E+0, data);
	data += add_point_6(0.1349105935937341E-3, 0.1905370790924295E+0, 0.2371311537781979E-1, data);
	data += add_point_6(0.1444060068369326E-3, 0.2242518717748009E+0, 0.4917878059254806E-1, data);
	data += add_point_6(0.1526797390930008E-3, 0.2577190808025936E+0, 0.7595498960495142E-1, data);
	data += add_point_6(0.1598208771406474E-3, 0.2908724534927187E+0, 0.1036991083191100E+0, data);
	data += add_point_6(0.1659354368615331E-3, 0.3236354020056219E+0, 0.1321348584450234E+0, data);
	data += add_point_6(0.1711279910946440E-3, 0.3559267359304543E+0, 0.1610316571314789E+0, data);
	data += add_point_6(0.1754952725601440E-3, 0.3876637123676956E+0, 0.1901912080395707E+0, data);
	data += add_point_6(0.1791247850802529E-3, 0.4187636705218842E+0, 0.2194384950137950E+0, data);
	data += add_point_6(0.1820954300877716E-3, 0.4491449019883107E+0, 0.2486155334763858E+0, data);
	data += add_point_6(0.1844788524548449E-3, 0.4787270932425445E+0, 0.2775768931812335E+0, data);
	data += add_point_6(0.1863409481706220E-3, 0.5074315153055574E+0, 0.3061863786591120E+0, data);
	data += add_point_6(0.1877433008795068E-3, 0.5351810507738336E+0, 0.3343144718152556E+0, data);
	data += add_point_6(0.1887444543705232E-3, 0.5619001025975381E+0, 0.3618362729028427E+0, data);
	data += add_point_6(0.1894009829375006E-3, 0.5875144035268046E+0, 0.3886297583620408E+0, data);
	data += add_point_6(0.1897683345035198E-3, 0.6119507308734495E+0, 0.4145742277792031E+0, data);
	data += add_point_6(0.1517327037467653E-3, 0.2619733870119463E+0, 0.2540047186389353E-1, data);
	data += add_point_6(0.1587740557483543E-3, 0.2968149743237949E+0, 0.5208107018543989E-1, data);
	data += add_point_6(0.1649093382274097E-3, 0.3310451504860488E+0, 0.7971828470885599E-1, data);
	data += add_point_6(0.1701915216193265E-3, 0.3646215567376676E+0, 0.1080465999177927E+0, data);
	data += add_point_6(0.1746847753144065E-3, 0.3974916785279360E+0, 0.1368413849366629E+0, data);
	data += add_point_6(0.1784555512007570E-3, 0.4295967403772029E+0, 0.1659073184763559E+0, data);
	data += add_point_6(0.1815687562112174E-3, 0.4608742854473447E+0, 0.1950703730454614E+0, data);
	data += add_point_6(0.1840864370663302E-3, 0.4912598858949903E+0, 0.2241721144376724E+0, data);
	data += add_point_6(0.1860676785390006E-3, 0.5206882758945558E+0, 0.2530655255406489E+0, data);
	data += add_point_6(0.1875690583743703E-3, 0.5490940914019819E+0, 0.2816118409731066E+0, data);
	data += add_point_6(0.1886453236347225E-3, 0.5764123302025542E+0, 0.3096780504593238E+0, data);
	data += add_point_6(0.1893501123329645E-3, 0.6025786004213506E+0, 0.3371348366394987E+0, data);
	data += add_point_6(0.1897366184519868E-3, 0.6275291964794956E+0, 0.3638547827694396E+0, data);
	data += add_point_6(0.1643908815152736E-3, 0.3348189479861771E+0, 0.2664841935537443E-1, data);
	data += add_point_6(0.1696300350907768E-3, 0.3699515545855295E+0, 0.5424000066843495E-1, data);
	data += add_point_6(0.1741553103844483E-3, 0.4042003071474669E+0, 0.8251992715430854E-1, data);
	data += add_point_6(0.1780015282386092E-3, 0.4375320100182624E+0, 0.1112695182483710E+0, data);
	data += add_point_6(0.1812116787077125E-3, 0.4699054490335947E+0, 0.1402964116467816E+0, data);
	data += add_point_6(0.1838323158085421E-3, 0.5012739879431952E+0, 0.1694275117584291E+0, data);
	data += add_point_6(0.1859113119837737E-3, 0.5315874883754966E+0, 0.1985038235312689E+0, data);
	data += add_point_6(0.1874969220221698E-3, 0.5607937109622117E+0, 0.2273765660020893E+0, data);
	data += add_point_6(0.1886375612681076E-3, 0.5888393223495521E+0, 0.2559041492849764E+0, data);
	data += add_point_6(0.1893819575809276E-3, 0.6156705979160163E+0, 0.2839497251976899E+0, data);
	data += add_point_6(0.1897794748256767E-3, 0.6412338809078123E+0, 0.3113791060500690E+0, data);
	data += add_point_6(0.1738963926584846E-3, 0.4076051259257167E+0, 0.2757792290858463E-1, data);
	data += add_point_6(0.1777442359873466E-3, 0.4423788125791520E+0, 0.5584136834984293E-1, data);
	data += add_point_6(0.1810010815068719E-3, 0.4760480917328258E+0, 0.8457772087727143E-1, data);
	data += add_point_6(0.1836920318248129E-3, 0.5085838725946297E+0, 0.1135975846359248E+0, data);
	data += add_point_6(0.1858489473214328E-3, 0.5399513637391218E+0, 0.1427286904765053E+0, data);
	data += add_point_6(0.1875079342496592E-3, 0.5701118433636380E+0, 0.1718112740057635E+0, data);
	data += add_point_6(0.1887080239102310E-3, 0.5990240530606021E+0, 0.2006944855985351E+0, data);
	data += add_point_6(0.1894905752176822E-3, 0.6266452685139695E+0, 0.2292335090598907E+0, data);
	data += add_point_6(0.1898991061200695E-3, 0.6529320971415942E+0, 0.2572871512353714E+0, data);
	data += add_point_6(0.1809065016458791E-3, 0.4791583834610126E+0, 0.2826094197735932E-1, data);
	data += add_point_6(0.1836297121596799E-3, 0.5130373952796940E+0, 0.5699871359683649E-1, data);
	data += add_point_6(0.1858426916241869E-3, 0.5456252429628476E+0, 0.8602712528554394E-1, data);
	data += add_point_6(0.1875654101134641E-3, 0.5768956329682385E+0, 0.1151748137221281E+0, data);
	data += add_point_6(0.1888240751833503E-3, 0.6068186944699046E+0, 0.1442811654136362E+0, data);
	data += add_point_6(0.1896497383866979E-3, 0.6353622248024907E+0, 0.1731930321657680E+0, data);
	data += add_point_6(0.1900775530219121E-3, 0.6624927035731797E+0, 0.2017619958756061E+0, data);
	data += add_point_6(0.1858525041478814E-3, 0.5484933508028488E+0, 0.2874219755907391E-1, data);
	data += add_point_6(0.1876248690077947E-3, 0.5810207682142106E+0, 0.5778312123713695E-1, data);
	data += add_point_6(0.1889404439064607E-3, 0.6120955197181352E+0, 0.8695262371439526E-1, data);
	data += add_point_6(0.1898168539265290E-3, 0.6416944284294319E+0, 0.1160893767057166E+0, data);
	data += add_point_6(0.1902779940661772E-3, 0.6697926391731260E+0, 0.1450378826743251E+0, data);
	data += add_point_6(0.1890125641731815E-3, 0.6147594390585488E+0, 0.2904957622341456E-1, data);
	data += add_point_6(0.1899434637795751E-3, 0.6455390026356783E+0, 0.5823809152617197E-1, data);
	data += add_point_6(0.1904520856831751E-3, 0.6747258588365477E+0, 0.8740384899884715E-1, data);
	data += add_point_6(0.1905534498734563E-3, 0.6772135750395347E+0, 0.2919946135808105E-1, data);
}

static double get_radii_atm(int Z)
{
	// Bragg-Slater radii  J.C. Slater JCP 41, (1964), 3199 [bohr]
	static double BSRadius[] = {
		1.00,																														  // 0 element for nothing
		0.661, 0.661,																												  // He
		2.740, 1.984, 1.606, 1.323, 1.228, 1.134, 0.945, 0.900,																		  //-Ne
		3.402, 2.835, 2.362, 2.079, 1.890, 1.890, 1.890, 1.890,																		  // Ar
		4.157, 3.402, 3.024, 2.656, 2.551, 2.656, 2.656, 2.656, 2.551, 2.551, 2.551, 2.551, 2.457, 2.362, 2.173, 2.173, 2.173, 2.173, // Kr
		4.441, 3.780, 3.402, 2.929, 2.740, 2.740, 2.551, 2.457, 2.551, 2.646, 3.024, 2.929, 2.929, 2.740, 2.740, 2.646, 2.646, 2.646, // Xe
		4.913, 4.063,																												  // Ba
		3.685, 3.496, 3.496, 3.496, 3.496, 3.496, 3.496, 3.402, 3.307, 3.307, 3.307, 3.307, 3.307, 3.307, 3.307,					  //*La-Lu
		2.929, 2.740, 2.551, 2.551, 2.457, 2.551, 2.551, 2.551, 2.835, 3.591, 3.024, 3.024, 3.591, 3.591, 3.591,					  // Rn
		4.063, 4.063,																												  // Fr-Ra
		3.685, 3.401, 3.401, 3.307, 3.307, 3.307, 3.307, 3.307, 3.307, 3.307, 3.307, 3.307, 3.307, 3.307, 3.307,					  // Ac-Lr
	};
	return BSRadius[Z];
}

static grid_t *get_angular(int order)
{
	int i;
	for (i = 0; i < 33; i++)
	{
		if (order == grid_order[i])
		{
			return gd_list[i];
		}
	}
	return NULL;
}

static int get_radii_grid(int order, int Z, double *radii_r, double *radii_w)
{
	double inv_lg2 = 1.1 * 1.4426950408889634074;
	double atm_r = get_radii_atm(Z);
	double atm_r3 = atm_r * atm_r * atm_r;
	// Chebyshev interpolation
	for (int i = 1; i <= order; i++)
	{
		radii_r[i - 1] = cos(i * 3.141592654 / (order + 1));
		radii_w[i - 1] = 3.141592654 / (order + 1) * sqrt(1 - radii_r[i - 1] * radii_r[i - 1]);
	}
	// Ahlrichs transform
	for (int i = 0; i < order; i++)
	{
		double radii_r_ = -inv_lg2 * pow(1 + radii_r[i], 0.6) * log(0.5 * (1 - radii_r[i]));
		double radii_dr_ = inv_lg2 * pow(1 + radii_r[i], 0.6) * (-0.6 * log((1 - radii_r[i]) / 2) / (1 + radii_r[i]) + 1 / (1 - radii_r[i]));
		// double radii_r_=(1.+radii_r[i])/(1.-radii_r[i]);
		radii_w[i] *= radii_r_ * radii_r_ * radii_dr_ * atm_r3;
		// radii_w[i]=2*3.141592654*pow(radii_r[i]+1,2.5)/pow(1-radii_r[i],3.5)*atm_r3/(order+1);
		radii_r[i] = radii_r_ * atm_r;
	}
	for (int i = 0; i < order / 2; i++)
	{
		double tmp = radii_r[i];
		radii_r[i] = radii_r[order - 1 - i];
		radii_r[order - 1 - i] = tmp;
		tmp = radii_w[i];
		radii_w[i] = radii_w[order - 1 - i];
		radii_w[order - 1 - i] = tmp;
	}
	return 0;
}

static double get_prod(int n, double *p, double *dist, double *dist_g, const atm_info atm, int *nn, int nn_num)
{
	int natm = atm->natm;
	double result = 1.;
	int i;
	double dist_ig, dist_jg, dist_ij;
	double mu;
	dist_ig = dist_g[n];
	if (dist_ig < 0.16 * get_radii_atm(atm(0, n)))
		return 1.;
	for (int a = 0; a < nn_num; a++)
	{
		i = nn[a];
		if (i == n)
			continue;
		dist_jg = dist_g[i];
		dist_ij = dist[i * natm + n];
		mu = (dist_ig - dist_jg) / dist_ij;
		if (mu <= -0.64)
			result *= 1.;
		else if (mu >= 0.64)
			return 0.;
		else
		{
			mu /= 0.64;
			result *= 0.5 * (1 - (35 * mu - 35 * mu * mu * mu + 21 * pow(mu, 5) - 5 * pow(mu, 7)) / 16.);
		}
	}
	return result;
}
static double normal_weight_grid(int n, double *p, double *dist, const atm_info atm, int *nn, int nn_num)
{
	int i;
	double w_n, normal;
	double *dist_g = (double *)malloc(sizeof(double) * atm->natm);
	for (int a = 0; a < nn_num; a++)
	{
		i = nn[a];
		if (atm(ELEMENT_VAL, i) <= 0)
			continue;
		dist_g[i] = (p[0] - atm->value[atm(CENTER_IND, i) + 0]) * (p[0] - atm->value[atm(CENTER_IND, i) + 0]) + (p[1] - atm->value[atm(CENTER_IND, i) + 1]) * (p[1] - atm->value[atm(CENTER_IND, i) + 1]) + (p[2] - atm->value[atm(CENTER_IND, i) + 2]) * (p[2] - atm->value[atm(CENTER_IND, i) + 2]);
		dist_g[i] = sqrt(dist_g[i]);
	}
	w_n = get_prod(n, p, dist, dist_g, atm, nn, nn_num);
	if (w_n == 1.)
	{
		free(dist_g);
		return 1.;
	}
	else if (w_n < 1E-8)
	{
		free(dist_g);
		return 0.;
	}
	normal = w_n;
	for (i = 0; i < nn_num; i++)
	{
		if (nn[i] == n)
			continue;
		int a = nn[i];
		normal += get_prod(a, p, dist, dist_g, atm, nn, nn_num);
	}
	free(dist_g);
	return w_n / normal;
}
static double Becke_step(double x)
{
	if (x < -1.2)
		return 1;
	if (x > 1.2)
		return 0.;
	double px = x * (3 - x * x) / 2;
	double ppx = px * (3 - px * px) / 2;
	double pppx = ppx * (3 - ppx * ppx) / 2; // Same as f(x)
	return (1 - pppx) / 2;
}

extern void get_block_shell(grid_t *grids, double *extent, grid_block block, const mol_info mol)
{
	const bas_info bas = mol->bas;
	const atm_info atm = mol->atm;
	int *tmp_shell = (int *)malloc(sizeof(int) * bas->nbas);
	int list_len = 0;
	double xc = 0., yc = 0., zc = 0.;
	double bx, by, bz;
	double gx, gy, gz;
	double rc = 0., rb;
	double dist;
	for (int i = 0; i < block->grid_num; i++)
	{
		xc += grids[block->grid_list[i]].x;
		yc += grids[block->grid_list[i]].y;
		zc += grids[block->grid_list[i]].z;
	}
	xc /= (double)block->grid_num;
	yc /= (double)block->grid_num;
	zc /= (double)block->grid_num;
	for (int i = 0; i < block->grid_num; i++)
	{
		gx = grids[block->grid_list[i]].x;
		gy = grids[block->grid_list[i]].y;
		gz = grids[block->grid_list[i]].z;
		dist = (gx - xc) * (gx - xc) + (gy - yc) * (gy - yc) + (gz - zc) * (gz - zc);
		if (dist > rc)
			rc = dist;
	}
	rc = sqrt(rc);
	for (int i = 0; i < bas->nbas; i++)
	{
		rb = extent[i];
		bx = atm->value[atm(CENTER_IND, bas(ATOM_IND, i)) + 0];
		by = atm->value[atm(CENTER_IND, bas(ATOM_IND, i)) + 1];
		bz = atm->value[atm(CENTER_IND, bas(ATOM_IND, i)) + 2];
		dist = (bx - xc) * (bx - xc) + (by - yc) * (by - yc) + (bz - zc) * (bz - zc);
		if (sqrt(dist) > rc + rb)
			continue;
		for (int j = 0; j < block->grid_num; j++)
		{
			gx = grids[block->grid_list[j]].x;
			gy = grids[block->grid_list[j]].y;
			gz = grids[block->grid_list[j]].z;
			dist = (bx - gx) * (bx - gx) + (by - gy) * (by - gy) + (bz - gz) * (bz - gz);
			if (sqrt(dist) < rb)
			{
				tmp_shell[list_len] = i;
				list_len++;
				break;
			}
		}
	}
	block->bas_num = 0;
	block->shell_list = (int *)malloc(sizeof(int) * (list_len));
	memcpy(block->shell_list, tmp_shell, sizeof(int) * list_len);
	block->shell_num = list_len;
	free(tmp_shell);
	for (int i = 0; i < block->shell_num; i++)
	{
		block->bas_num += xint_gtolen(mol->bas(ANGULAR_VAL, block->shell_list[i]));
	}
	block->local2global = (int *)malloc(sizeof(int) * block->bas_num);
	block->global2local = malloc_list(bas->msize);
	for (int i = 0; i < bas->msize; i++)
		block->global2local[i] = -1;
	for (int i = 0, u = 0; i < block->shell_num; i++)
	{
		int di = xint_gtolen(mol->bas(ANGULAR_VAL, block->shell_list[i]));
		for (int j = 0; j < di; j++, u++)
		{
			block->local2global[u] = bas->shls_p[block->shell_list[i]] + j;
		}
	}
	for (int i = 0; i < block->bas_num; i++)
		block->global2local[block->local2global[i]] = i;
	return;
}
/**
 * @brief function used to divide grids in to blocks by Oct-tree algorithm
 *
 * @param[out] info sturct used to store all information of grids
 * @param[in] maxR max distance used to do cut off
 * @return void
 */
extern void divide_block(grids_info info, double maxR)
{
	grid_t *grids = info->grids;
	int len = info->grid_num;
	double rc, xc, yc, zc;
	double dist, dist_max;
	double **grid_r = (double **)malloc(sizeof(double *) * 3);
	double *grid_rp;
	grid_r[0] = (double *)malloc(sizeof(double) * len);
	grid_r[1] = (double *)malloc(sizeof(double) * len);
	grid_r[2] = (double *)malloc(sizeof(double) * len);
	vector_node *node_p;
	qvector *leaf_p;
	qvector *leaf_r, *leaf_l;
	qvvector *new_leaf = (qvvector *)malloc(sizeof(qvvector));
	qvvector *active_leaf = (qvvector *)malloc(sizeof(qvvector));
	qvvector *complete_tree = (qvvector *)malloc(sizeof(qvvector));
	qvvector *exchange_p;
	init_vvecotr(new_leaf);
	init_vvecotr(active_leaf);
	init_vvecotr(complete_tree);
	push_vvecotr(active_leaf);
	leaf_p = get_vvector_data(active_leaf, 0);
	for (int i = 0; i < len; i++)
	{
		push_vector(leaf_p, i);
		grid_r[0][i] = grids[i].x;
		grid_r[1][i] = grids[i].y;
		grid_r[2][i] = grids[i].z;
	}
	while (1)
	{
		for (int a = 0; a < 3; a++)
		{
			grid_rp = grid_r[a];
			for (int i = 0; i < active_leaf->len; i++)
			{
				rc = 0.;
				xc = 0.;
				yc = 0.;
				zc = 0.;
				leaf_p = get_vvector_data(active_leaf, i);
				node_p = leaf_p->head;
				for (int j = 0; j < leaf_p->len; j++)
				{
					rc += grid_rp[node_p->data];
					xc += grids[node_p->data].x;
					yc += grids[node_p->data].y;
					zc += grids[node_p->data].z;
					node_p = node_p->next;
				}
				rc /= leaf_p->len;
				xc /= leaf_p->len;
				yc /= leaf_p->len;
				zc /= leaf_p->len;
				leaf_r = (qvector *)malloc(sizeof(qvector));
				leaf_l = (qvector *)malloc(sizeof(qvector));
				init_vector(leaf_r);
				init_vector(leaf_l);
				node_p = leaf_p->head;
				for (int j = 0; j < leaf_p->len; j++)
				{
					if (grid_rp[node_p->data] < rc)
						push_vector(leaf_l, node_p->data);
					else
						push_vector(leaf_r, node_p->data);
					node_p = node_p->next;
				}
				if (leaf_l->len >= 250)
				{
				CONTINUE_X_L:
					push_vvecotr(new_leaf);
					new_leaf->head->data.len = leaf_l->len;
					new_leaf->head->data.head = leaf_l->head;
				}
				else if (leaf_l->len <= 100)
				{
				PUSH_IN_X_L:
					push_vvecotr(complete_tree);
					complete_tree->head->data.len = leaf_l->len;
					complete_tree->head->data.head = leaf_l->head;
				}
				else
				{
					node_p = leaf_l->head;
					dist_max = 0.;
					for (int j = 0; j < leaf_l->len; j++)
					{
						dist = (grids[node_p->data].x - xc) * (grids[node_p->data].x - xc) + (grids[node_p->data].y - yc) * (grids[node_p->data].y - yc) + (grids[node_p->data].z - zc) * (grids[node_p->data].z - zc);
						dist = sqrt(dist);
						if (dist_max < dist)
							dist_max = dist;
						node_p = node_p->next;
					}
					if (dist_max <= maxR)
						goto PUSH_IN_X_L;
					else
						goto CONTINUE_X_L;
				}
				if (leaf_r->len >= 250)
				{
				CONTINUE_X_R:
					push_vvecotr(new_leaf);
					new_leaf->head->data.len = leaf_r->len;
					new_leaf->head->data.head = leaf_r->head;
				}
				else if (leaf_r->len <= 100)
				{
				PUSH_IN_X_R:
					push_vvecotr(complete_tree);
					complete_tree->head->data.len = leaf_r->len;
					complete_tree->head->data.head = leaf_r->head;
				}
				else
				{
					node_p = leaf_r->head;
					dist_max = 0.;
					for (int j = 0; j < leaf_r->len; j++)
					{
						dist = (grids[node_p->data].x - xc) * (grids[node_p->data].x - xc) + (grids[node_p->data].y - yc) * (grids[node_p->data].y - yc) + (grids[node_p->data].z - zc) * (grids[node_p->data].z - zc);
						dist = sqrt(dist);
						if (dist_max < dist)
							dist_max = dist;
						node_p = node_p->next;
					}
					if (dist_max <= maxR)
						goto PUSH_IN_X_R;
					else
						goto CONTINUE_X_R;
				}
				free(leaf_l);
				free(leaf_r);
			}
			if (new_leaf->len == 0)
				goto FINISH;
			exchange_p = active_leaf;
			active_leaf = new_leaf;
			del_vvector(exchange_p);
			new_leaf = exchange_p;
		}
	}
FINISH:
	del_vvector(new_leaf);
	del_vvector(active_leaf);
	free(new_leaf);
	free(active_leaf);
	info->blocks = (grid_block)malloc(sizeof(struct GridBlock) * complete_tree->len);
	info->block_num = complete_tree->len;
	for (int i = 0; i < complete_tree->len; i++)
	{
		leaf_p = get_vvector_data(complete_tree, i);
		info->blocks[i].grid_list = malloc_list(leaf_p->len);
		info->blocks[i].grid_num = leaf_p->len;

		node_p = leaf_p->head;
		for (int j = 0; j < leaf_p->len; j++)
		{
			info->blocks[i].grid_list[j] = node_p->data;
			node_p = node_p->next;
		}
	}
	free(grid_r[0]);
	free(grid_r[1]);
	free(grid_r[2]);
	free(grid_r);
	del_vvector(complete_tree);
	free(complete_tree);
	return;
}

extern void cal_block_bas(int block_index, grids_info grids)
{
	const mol_info mol = grids->mol;
	assert(mol != NULL);
	grid_block block;
	double xyz[3];
	block = &grids->blocks[block_index];
	int p = 0;
	/// malloc memory for storing value of basis functions
	block->psi = malloc_matrix(block->grid_num, block->bas_num);
	block->psix = malloc_matrix(block->grid_num, block->bas_num);
	block->psiy = malloc_matrix(block->grid_num, block->bas_num);
	block->psiz = malloc_matrix(block->grid_num, block->bas_num);
	block->psixx = malloc_matrix(block->grid_num, block->bas_num);
	block->psiyy = malloc_matrix(block->grid_num, block->bas_num);
	block->psizz = malloc_matrix(block->grid_num, block->bas_num);
	block->psixz = malloc_matrix(block->grid_num, block->bas_num);
	block->psixy = malloc_matrix(block->grid_num, block->bas_num);
	block->psiyz = malloc_matrix(block->grid_num, block->bas_num);
	for (int a = 0; a < block->shell_num; a++)
	{
		for (int i = 0; i < block->grid_num; i++)
		{
			xyz[0] = grids->grids[block->grid_list[i]].x;
			xyz[1] = grids->grids[block->grid_list[i]].y;
			xyz[2] = grids->grids[block->grid_list[i]].z;
			eval_ao(xyz, block->shell_list[a], &block->psi[i][p], &block->psix[i][p],
					&block->psiy[i][p], &block->psiz[i][p], &block->psixx[i][p], &block->psiyy[i][p],
					&block->psizz[i][p], mol->atm, mol->bas);
		}
		p += xint_gtolen(mol->bas(ANGULAR_VAL, block->shell_list[a]));
	}
	return;
}

void clear_block_bas(grid_block block)
{
	if (block->psi != NULL)
	{
		free_matrix(block->psi);
		free_matrix(block->psix);
		free_matrix(block->psiy);
		free_matrix(block->psiz);
		free_matrix(block->psixx);
		free_matrix(block->psiyy);
		free_matrix(block->psizz);
		free_matrix(block->psixz);
		free_matrix(block->psiyz);
		free_matrix(block->psixy);
	}
	block->psi = NULL;
	block->psix = NULL;
	block->psiy = NULL;
	block->psiz = NULL;
	block->psixx = NULL;
	block->psiyy = NULL;
	block->psizz = NULL;
	block->psixy = NULL;
	block->psiyz = NULL;
	block->psixz = NULL;
	return;
}

int del_grid(grids_info grids)
{
	free(grids->grids);
	for (int i = 0; i < grids->block_num; i++)
	{
		free(grids->blocks[i].grid_list);
		free(grids->blocks[i].shell_list);
		free(grids->blocks[i].local2global);
		free(grids->blocks[i].global2local);
		if (grids->blocks[i].psi != NULL)
			clear_block_bas(&grids->blocks[i]);
	}
	free(grids->blocks);
	free(grids->bas_extent);
	free(grids);
	return 0;
}

static grid_t *get_mol_grid(GRIDS_TYPE type, const atm_info atm, int *len)
{
	int *para_list;
	int angular_order;
	int natm = atm->natm;
	double *dist = (double *)malloc(sizeof(double) * natm * natm);
	int tol_num = 0;
	int angular_max_num;
	grid_t *grids = NULL;
	int **nn_list = (int **)malloc(sizeof(int *) * atm->natm);
	int *nn_num_list = (int *)calloc(atm->natm, sizeof(int));
	nn_list[0] = (int *)malloc(sizeof(int) * atm->natm * atm->natm);
	nn_list[0][0] = 0;
	nn_num_list[0] = 1;
	for (int i = 1; i < atm->natm; i++)
	{
		nn_list[i] = nn_list[i - 1] + atm->natm;
		nn_list[i][0] = i;
		nn_num_list[i] = 1;
	}
	switch (type)
	{
	case COSXI_GRIDS:
		para_list = cosx_init;
		break;
	case COSXF_GRIDS:
		para_list = cosx_final;
		break;
	case XCOARSE_GRIDS:
		para_list = xcoarse;
		break;
	case COARSE_GRIDS:
		para_list = coarse;
		break;
	case MEDIUM_GRIDS:
		para_list = medium;
		break;
	case FINE_GRIDS:
		para_list = fine;
		break;
	case COSJ_GRIDS:
		para_list = cosj;
		break;
	case COR_GRIDS:
		para_list = c_coarse;
		break;
	}
	angular_order = para_list[0];
	for (int i = 0; i < 33; i++)
	{
		if (grid_order[i] == angular_order)
		{
			angular_max_num = grid_num_list[i];
			break;
		}
	}
	grid_t *tmp_grid_mol = (grid_t *)malloc(sizeof(grid_t) * natm * 150 * angular_max_num);
	memset(dist, 0, sizeof(double) * natm * natm);
	for (int i = 0; i < natm; i++)
	{
		for (int j = 0; j < i; j++)
		{
			dist[i * natm + j] = sqrt((atm->value[atm(CENTER_IND, i)] - atm->value[atm(CENTER_IND, j)]) *
										  (atm->value[atm(CENTER_IND, i)] - atm->value[atm(CENTER_IND, j)]) +
									  (atm->value[atm(CENTER_IND, i) + 1] - atm->value[atm(CENTER_IND, j) + 1]) *
										  (atm->value[atm(CENTER_IND, i) + 1] - atm->value[atm(CENTER_IND, j) + 1]) +
									  (atm->value[atm(CENTER_IND, i) + 2] - atm->value[atm(CENTER_IND, j) + 2]) *
										  (atm->value[atm(CENTER_IND, i) + 2] - atm->value[atm(CENTER_IND, j) + 2]));
			dist[j * natm + i] = dist[i * natm + j];
			if (1)
			{
				nn_list[i][nn_num_list[i]] = j;
				nn_num_list[i]++;
			}
			if (1)
			{
				nn_list[j][nn_num_list[j]] = i;
				nn_num_list[j]++;
			}
		}
	}
#pragma omp parallel
	{
		int radii_order;
#pragma omp for schedule(dynamic, 1) nowait
		for (int i = 0; i < natm; i++)
		{
			int ci = atm(ELEMENT_VAL, i);
			if (atm(ELEMENT_VAL, i) <= 10)
				radii_order = para_list[1];
			else if (atm(ELEMENT_VAL, i) <= 18)
				radii_order = para_list[2];
			else if (atm(ELEMENT_VAL, i) <= 36)
				radii_order = para_list[3];
			else
				radii_order = para_list[4];
			;
			grid_t *tmp_grid = (grid_t *)malloc(sizeof(grid_t) * radii_order * angular_max_num);
			int tmp_grid_num;
			tmp_grid_num = get_atm_grid(angular_order, radii_order, i, dist, atm, tmp_grid, para_list[5], nn_list[i], nn_num_list[i]);
#pragma omp critical
			{
				memcpy(&tmp_grid_mol[tol_num], tmp_grid, sizeof(grid_t) * tmp_grid_num);
				tol_num += tmp_grid_num;
			}
			free(tmp_grid);
		}
	}
	grids = (grid_t *)malloc(sizeof(grid_t) * tol_num);
	memcpy(grids, tmp_grid_mol, sizeof(grid_t) * tol_num);
	free(tmp_grid_mol);
	free(dist);
	*len = tol_num;
	free(nn_list[0]);
	free(nn_list);
	free(nn_num_list);
	return grids;
}

static int get_atm_grid(int angular_order, int radii_order,
						int a, double *dist, const atm_info atm, grid_t *grids, int if_prun, int *nn, int nn_num)
{
	double *radii_r, *radii_w;
	grid_t *angular_grids = NULL;
	int layer_1 = floor(radii_order * 0.25);
	int layer_2 = layer_1 * 2;
	int layer_3 = layer_1 * 3;
	int ang_index_1 = -1, ang_index_2 = -1, ang_index_3 = 2;
	int grid_num = 0;
	double x, y, z;
	double p[3];
	radii_r = (double *)malloc(sizeof(double) * radii_order);
	radii_w = (double *)malloc(sizeof(double) * radii_order);
	x = atm->value[atm(CENTER_IND, a) + 0];
	y = atm->value[atm(CENTER_IND, a) + 1];
	z = atm->value[atm(CENTER_IND, a) + 2];
	int tol_size = 0;
	get_radii_grid(radii_order, atm(ELEMENT_VAL, a), radii_r, radii_w);
	for (int i = 0; i < 33; i++)
	{
		if (grid_order[i] == angular_order)
		{
			ang_index_1 = i;
			tol_size += layer_2 * grid_num_list[ang_index_1];
			break;
		}
	}
	if (angular_order - 6 > 7)
		for (int i = 0; i < 33; i++)
		{
			if (grid_order[i] >= angular_order - 6)
			{
				ang_index_2 = i;
				tol_size += layer_1 * grid_num_list[ang_index_2];
				break;
			}
		}
	else
	{
		ang_index_2 = 2;
		tol_size += layer_1 * grid_num_list[ang_index_2];
	}
	tol_size += (radii_order - layer_3) * grid_num_list[ang_index_3];
	if (if_prun == 0)
		layer_2 = radii_order;
	for (int i = 0; i < layer_2; ++i)
	{
		angular_grids = gd_list[ang_index_2];
		int angular_len = grid_num_list[ang_index_2];
		for (int j = 0; j < angular_len; ++j)
		{
			grids[grid_num].x = x + angular_grids[j].x * radii_r[i];
			grids[grid_num].y = y + angular_grids[j].y * radii_r[i];
			grids[grid_num].z = z + angular_grids[j].z * radii_r[i];
			grids[grid_num].w = angular_grids[j].w * radii_w[i];
			grids[grid_num].ia = a;
			p[0] = grids[grid_num].x;
			p[1] = grids[grid_num].y;
			p[2] = grids[grid_num].z;
			grids[grid_num].w *= normal_weight_grid(a, p, dist, atm, nn, nn_num);
			if (grids[grid_num].w >= 1E-15)
			{
				grid_num++;
			}
		}
	}
	if (if_prun == 0)
		goto END_ATM_GRIDS;
	for (int i = layer_2; i < layer_3; i++)
	{
		angular_grids = gd_list[ang_index_2];
		int angular_len = grid_num_list[ang_index_2];
		for (int j = 0; j < angular_len; ++j)
		{
			grids[grid_num].x = x + angular_grids[j].x * radii_r[i];
			grids[grid_num].y = y + angular_grids[j].y * radii_r[i];
			grids[grid_num].z = z + angular_grids[j].z * radii_r[i];
			grids[grid_num].w = angular_grids[j].w * radii_w[i];
			grids[grid_num].ia = a;
			p[0] = grids[grid_num].x;
			p[1] = grids[grid_num].y;
			p[2] = grids[grid_num].z;
			grids[grid_num].w *= normal_weight_grid(a, p, dist, atm, nn, nn_num);
			if (grids[grid_num].w >= 1E-15)
				++grid_num;
		}
	}
	for (int i = layer_3; i < radii_order; i++)
	{
		angular_grids = gd_list[ang_index_3];
		if (radii_r[i] > 10.0)
			goto END_ATM_GRIDS;
		int angular_len = grid_num_list[ang_index_3];
		for (int j = 0; j < angular_len; ++j)
		{
			grids[grid_num].x = x + angular_grids[j].x * radii_r[i];
			grids[grid_num].y = y + angular_grids[j].y * radii_r[i];
			grids[grid_num].z = z + angular_grids[j].z * radii_r[i];
			grids[grid_num].w = angular_grids[j].w * radii_w[i];
			grids[grid_num].ia = a;
			p[0] = grids[grid_num].x;
			p[1] = grids[grid_num].y;
			p[2] = grids[grid_num].z;
			grids[grid_num].w *= normal_weight_grid(a, p, dist, atm, nn, nn_num);
			if (grids[grid_num].w >= 1E-15)
				++grid_num;
		}
	}
END_ATM_GRIDS:
	free(radii_r);
	free(radii_w);
	return grid_num;
}

void get_grids()
{
	make_14_grid();
	make_50_grid();
	make_26_grid();
	make_38_grid();
	make_110_grid();
	make_86_grid();
	make_74_grid();
	make_146_grid();
	make_170_grid();
	make_194_grid();
	make_230_grid();
	make_266_grid();
	make_302_grid();
	make_350_grid();
	make_434_grid();
	make_590_grid();
	make_770_grid();
	make_974_grid();
	make_1202_grid();
	make_1454_grid();
	make_1730_grid();
	return;
}

extern Matrix cal_overlap_numerical(grids_info grids)
{
	const mol_info mol = grids->mol;
	assert(mol != NULL);
	int msize = mol->bas->msize;
	Matrix s_matrix = malloc_matrix(msize, msize);
#pragma omp parallel
	{
		double weight;
		Matrix s_matrix_tmp = malloc_matrix(msize, msize);
#pragma omp for schedule(dynamic, 1) nowait
		for (int a = 0; a < grids->block_num; a++)
		{
			grid_block block = &grids->blocks[a];
			List grid_list = block->grid_list;
			List local2global = block->local2global;
			int bas_num = block->bas_num;
			int grid_num = block->grid_num;
			if (bas_num == 0)
				continue;
			Matrix local_s_matrix = malloc_matrix(block->bas_num, block->bas_num);
			memset(local_s_matrix[0], 0, sizeof(double) * block->bas_num * block->bas_num);
			cal_block_bas(a, grids);
			for (int g = 0; g < grid_num; g++)
			{
				weight = sqrt(grids->grids[grid_list[g]].w);
				for (int i = 0; i < bas_num; i++)
					block->psi[g][i] *= weight;
			}
			cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, bas_num, bas_num, grid_num,
						1.0, block->psi[0], bas_num, block->psi[0], bas_num, 0.0, local_s_matrix[0], bas_num);
			for (int i = 0; i < block->bas_num; i++)
			{
				int ig = local2global[i];
				for (int j = 0; j < block->bas_num; j++)
				{
					int jg = local2global[j];
					s_matrix_tmp[ig][jg] += local_s_matrix[i][j];
				}
			}
			clear_block_bas(block);
			free_matrix(local_s_matrix);
		}
#pragma omp critical
		for (int i = 0; i < msize * msize; i++)
			s_matrix[0][i] += s_matrix_tmp[0][i];
		free_matrix(s_matrix_tmp);
	}
	return s_matrix;
}

void cal_rho(Vector rho, const Matrix d_matrix, const grid_block block)
{
	int grid_num = block->grid_num;
	int bas_num = block->bas_num;
	Vector psi;
	Vector rho_tmp = malloc_vector(bas_num);
	for (int g = 0; g < grid_num; g++)
	{
		psi = block->psi[g];
		cblas_dgemv(CblasRowMajor, CblasNoTrans, bas_num, bas_num, 1., d_matrix[0], bas_num, psi, 1, 0., rho_tmp, 1.);
		rho[g] = vv_dot(psi, rho_tmp, bas_num) * 0.5;
	}
	free_vector(rho_tmp);
	return;
}

void cal_kinetic(Vector kinetic, const Matrix d_matrix, const grid_block block)
{
	int grid_num = block->grid_num;
	int bas_num = block->bas_num;
	Vector psi;
	Vector rho_tmp = malloc_vector(bas_num);
	for (int g = 0; g < grid_num; g++)
	{
		psi = block->psix[g];
		cblas_dgemv(CblasRowMajor, CblasNoTrans, bas_num, bas_num, 1., d_matrix[0], bas_num, psi, 1, 0., rho_tmp, 1.);
		kinetic[g] = vv_dot(psi, rho_tmp, bas_num);
		psi = block->psiy[g];
		cblas_dgemv(CblasRowMajor, CblasNoTrans, bas_num, bas_num, 1., d_matrix[0], bas_num, psi, 1, 0., rho_tmp, 1.);
		kinetic[g] += vv_dot(psi, rho_tmp, bas_num);
		psi = block->psiz[g];
		cblas_dgemv(CblasRowMajor, CblasNoTrans, bas_num, bas_num, 1., d_matrix[0], bas_num, psi, 1, 0., rho_tmp, 1.);
		kinetic[g] += vv_dot(psi, rho_tmp, bas_num);
	}
	free_vector(rho_tmp);
	return;
}

void cal_rho_c(Vector rho, const Matrix c_matrix, const int orb_num, const grid_block block)
{
	int grid_num = block->grid_num;
	int bas_num = block->bas_num;
	Matrix orb_psi = malloc_matrix(grid_num, orb_num);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, grid_num, orb_num, bas_num, 1., block->psi[0], bas_num, c_matrix[0], bas_num, 0., orb_psi[0], orb_num);
	for (int g = 0; g < grid_num; g++)
	{
		Vector psi = orb_psi[g];
		rho[g] = vv_dot(psi, psi, orb_num) * 0.5;
	}
	free_matrix(orb_psi);
	return;
}
/// optimize grids for J and K matrices calculations
/// Frank Neese https://doi.org/10.1063/5.0058766
extern int get_radii_num_opt(int charge, int epsilon, int beta)
{
	int row = 0;
	if (charge <= 2)
		row = 1;
	else if (charge <= 10)
		row = 2;
	else if (charge <= 18)
		row = 3;
	else if (charge <= 36)
		row = 4;
	else if (charge <= 54)
		row = 5;
	else
		row = 6;
	return (15 * epsilon - 50) + beta * row;
}

static int get_radii_grid_opt(int order, int Z, double xi, double *radii_r, double *radii_w)
{
	double inv_lg2 = xi * 1.4426950408889634074;
	double atm_r = get_radii_atm(Z);
	double atm_r3 = atm_r * atm_r * atm_r;
	// Chebyshev interpolation
	for (int i = 1; i <= order; i++)
	{
		radii_r[i - 1] = cos(i * 3.141592654 / (order + 1));
		radii_w[i - 1] = 3.141592654 / (order + 1) * sqrt(1 - radii_r[i - 1] * radii_r[i - 1]);
	}
	// Ahlrichs transform
	for (int i = 0; i < order; i++)
	{
		double radii_r_ = -inv_lg2 * pow(1 + radii_r[i], 0.6) * log(0.5 * (1 - radii_r[i]));
		double radii_dr_ = inv_lg2 * pow(1 + radii_r[i], 0.6) * (-0.6 * log((1 - radii_r[i]) / 2) / (1 + radii_r[i]) + 1 / (1 - radii_r[i]));
		radii_w[i] *= radii_r_ * radii_r_ * radii_dr_ * atm_r3;
		radii_r[i] = radii_r_ * atm_r;
	}
	for (int i = 0; i < order / 2; i++)
	{
		double tmp = radii_r[i];
		radii_r[i] = radii_r[order - 1 - i];
		radii_r[order - 1 - i] = tmp;
		tmp = radii_w[i];
		radii_w[i] = radii_w[order - 1 - i];
		radii_w[order - 1 - i] = tmp;
	}
	return 0;
}

grids_info init_grid(int type, const mol_info mol)
{
	grids_info grids = (grids_info)malloc(sizeof(struct GridsInfo));
	grids->grids = get_mol_grid(type, mol->atm, &grids->grid_num);
	grids->bas_extent = (double *)malloc(sizeof(double) * mol->bas->nbas);
	grids->maxR = 0.;
	for (int i = 0; i < mol->bas->nbas; i++)
	{
		grids->bas_extent[i] = cal_bas_extent(i, mol->bas);
		if (grids->maxR < grids->bas_extent[i])
			grids->maxR = grids->bas_extent[i];
	}
	divide_block(grids, grids->maxR);
#pragma omp parallel
	{
#pragma omp for schedule(dynamic, 1) nowait
		for (int i = 0; i < grids->block_num; i++)
		{
			get_block_shell(grids->grids, grids->bas_extent, &grids->blocks[i], mol);
			grids->blocks[i].psi = NULL;
			grids->blocks[i].psix = NULL;
			grids->blocks[i].psiy = NULL;
			grids->blocks[i].psiz = NULL;
			grids->blocks[i].psixx = NULL;
			grids->blocks[i].psiyy = NULL;
			grids->blocks[i].psizz = NULL;
			grids->blocks[i].psixy = NULL;
			grids->blocks[i].psiyz = NULL;
			grids->blocks[i].psixz = NULL;
		}
	}
	grids->mol = mol;
	return grids;
}

extern int get_atm_grid_opt(double xi, double *layer, int *angular_order, int radii_order, int a, double *dist, const atm_info atm, grid_t *grids, int *nn, int nn_num)
{
	double *radii_r, *radii_w;
	grid_t *angular_grids = NULL;
	int layer_1 = floor(layer[0] * radii_order);
	int layer_2 = floor(layer[1] * radii_order) + layer_1;
	int layer_3 = floor(layer[2] * radii_order) + layer_2;
	int layer_4 = floor(layer[3] * radii_order) + layer_3;
	int ang_index_1 = angular_order[0], ang_index_2 = angular_order[1], ang_index_3 = angular_order[2],
		ang_index_4 = angular_order[3], ang_index_5 = angular_order[4];
	int grid_num = 0;
	double x, y, z;
	double p[3];
	radii_r = (double *)malloc(sizeof(double) * radii_order);
	radii_w = (double *)malloc(sizeof(double) * radii_order);
	x = atm->value[atm(CENTER_IND, a) + 0];
	y = atm->value[atm(CENTER_IND, a) + 1];
	z = atm->value[atm(CENTER_IND, a) + 2];
	get_radii_grid_opt(radii_order, atm(ELEMENT_VAL, a), xi, radii_r, radii_w);
	for (int i = 0; i < layer_1; ++i)
	{
		angular_grids = gd_list[ang_index_1];
		int angular_len = grid_num_list[ang_index_1];
		for (int j = 0; j < angular_len; ++j)
		{
			grids[grid_num].x = x + angular_grids[j].x * radii_r[i];
			grids[grid_num].y = y + angular_grids[j].y * radii_r[i];
			grids[grid_num].z = z + angular_grids[j].z * radii_r[i];
			grids[grid_num].w = angular_grids[j].w * radii_w[i];
			grids[grid_num].ia = a;
			p[0] = grids[grid_num].x;
			p[1] = grids[grid_num].y;
			p[2] = grids[grid_num].z;
			grids[grid_num].w *= normal_weight_grid(a, p, dist, atm, nn, nn_num);
			if (grids[grid_num].w >= 1E-15)
			{
				grid_num++;
			}
		}
	}
	for (int i = layer_1; i < layer_2; i++)
	{
		angular_grids = gd_list[ang_index_2];
		int angular_len = grid_num_list[ang_index_2];
		for (int j = 0; j < angular_len; ++j)
		{
			grids[grid_num].x = x + angular_grids[j].x * radii_r[i];
			grids[grid_num].y = y + angular_grids[j].y * radii_r[i];
			grids[grid_num].z = z + angular_grids[j].z * radii_r[i];
			grids[grid_num].w = angular_grids[j].w * radii_w[i];
			grids[grid_num].ia = a;
			p[0] = grids[grid_num].x;
			p[1] = grids[grid_num].y;
			p[2] = grids[grid_num].z;
			grids[grid_num].w *= normal_weight_grid(a, p, dist, atm, nn, nn_num);
			if (grids[grid_num].w >= 1E-15)
				++grid_num;
		}
	}
	for (int i = layer_2; i < layer_3; i++)
	{
		angular_grids = gd_list[ang_index_3];
		int angular_len = grid_num_list[ang_index_3];
		for (int j = 0; j < angular_len; ++j)
		{
			grids[grid_num].x = x + angular_grids[j].x * radii_r[i];
			grids[grid_num].y = y + angular_grids[j].y * radii_r[i];
			grids[grid_num].z = z + angular_grids[j].z * radii_r[i];
			grids[grid_num].w = angular_grids[j].w * radii_w[i];
			grids[grid_num].ia = a;
			p[0] = grids[grid_num].x;
			p[1] = grids[grid_num].y;
			p[2] = grids[grid_num].z;
			grids[grid_num].w *= normal_weight_grid(a, p, dist, atm, nn, nn_num);
			if (grids[grid_num].w >= 1E-15)
				++grid_num;
		}
	}
	for (int i = layer_3; i < layer_4; i++)
	{
		angular_grids = gd_list[ang_index_4];
		int angular_len = grid_num_list[ang_index_4];
		for (int j = 0; j < angular_len; ++j)
		{
			grids[grid_num].x = x + angular_grids[j].x * radii_r[i];
			grids[grid_num].y = y + angular_grids[j].y * radii_r[i];
			grids[grid_num].z = z + angular_grids[j].z * radii_r[i];
			grids[grid_num].w = angular_grids[j].w * radii_w[i];
			grids[grid_num].ia = a;
			p[0] = grids[grid_num].x;
			p[1] = grids[grid_num].y;
			p[2] = grids[grid_num].z;
			grids[grid_num].w *= normal_weight_grid(a, p, dist, atm, nn, nn_num);
			if (grids[grid_num].w >= 1E-15)
				++grid_num;
		}
	}
	for (int i = layer_4; i < radii_order; i++)
	{
		angular_grids = gd_list[ang_index_5];
		int angular_len = grid_num_list[ang_index_5];
		for (int j = 0; j < angular_len; ++j)
		{
			grids[grid_num].x = x + angular_grids[j].x * radii_r[i];
			grids[grid_num].y = y + angular_grids[j].y * radii_r[i];
			grids[grid_num].z = z + angular_grids[j].z * radii_r[i];
			grids[grid_num].w = angular_grids[j].w * radii_w[i];
			grids[grid_num].ia = a;
			p[0] = grids[grid_num].x;
			p[1] = grids[grid_num].y;
			p[2] = grids[grid_num].z;
			grids[grid_num].w *= normal_weight_grid(a, p, dist, atm, nn, nn_num);
			if (grids[grid_num].w >= 1E-11)
				++grid_num;
		}
	}
	free(radii_r);
	free(radii_w);
	return grid_num;
}

extern grid_t *get_mol_grid_opt(const char *opt_file, const mol_info mol, int *len)
{
	const atm_info atm = mol->atm;
	const bas_info bas = mol->bas;
	int *para_list;
	int angular_order[5];
	int angular_index[5];
	double row_pra[10][5];
	double epsilon, beta;
	int natm = atm->natm;
	double *dist = (double *)malloc(sizeof(double) * natm * natm);
	int tol_num = 0;
	int angular_max_num = 0;
	grid_t *grids = NULL;
	int **nn_list = (int **)malloc(sizeof(int *) * atm->natm);
	int *nn_num_list = (int *)calloc(atm->natm, sizeof(int));
	FILE *opt;
	char string[1024]; /// buf used to store file's words
	char tmp1[64], tmp2[64], tmp3[64], tmp4[64], tmp5[64];
	char *err;
	opt = fopen(opt_file, "r");
	assert(opt != NULL);
	err = fgets(string, sizeof(string), opt);
	int line_num = atoi(string);
	err = fgets(string, sizeof(string), opt);
	sscanf(string, "%s%s", tmp1, tmp2);
	epsilon = atoi(tmp1);
	beta = atoi(tmp2);
	err = fgets(string, sizeof(string), opt);
	sscanf(string, "%s%s%s%s%s", tmp1, tmp2, tmp3, tmp4, tmp5);
	angular_order[0] = atoi(tmp1);
	angular_order[1] = atoi(tmp2);
	angular_order[2] = atoi(tmp3);
	angular_order[3] = atoi(tmp4);
	angular_order[4] = atoi(tmp5);
	for (int i = 0; i < line_num; i++)
	{
		err = fgets(string, sizeof(string), opt);
		sscanf(string, "%s%s%s%s%s", tmp1, tmp2, tmp3, tmp4, tmp5);
		row_pra[i][0] = atof(tmp1);
		row_pra[i][1] = atof(tmp2);
		row_pra[i][2] = atof(tmp3);
		row_pra[i][3] = atof(tmp4);
		row_pra[i][4] = atof(tmp5);
	}
	fclose(opt);
	nn_list[0] = (int *)malloc(sizeof(int) * atm->natm * atm->natm);
	nn_list[0][0] = 0;
	nn_num_list[0] = 1;
	for (int i = 1; i < atm->natm; i++)
	{
		nn_list[i] = nn_list[i - 1] + atm->natm;
		nn_list[i][0] = i;
		nn_num_list[i] = 1;
	}
	for (int i = 0; i < 5; i++)
		if (angular_max_num < angular_order[i])
			angular_max_num = angular_order[i];
	for (int a = 0; a < 5; a++)
		for (int i = 0; i < 30; i++)
		{
			if (grid_num_list[i] == angular_order[a])
			{
				angular_index[a] = i;
				break;
			}
		}
	grid_t *tmp_grid_mol = (grid_t *)malloc(sizeof(grid_t) * natm * 150 * angular_max_num);
	memset(dist, 0, sizeof(double) * natm * natm);
	for (int i = 0; i < natm; i++)
	{
		for (int j = 0; j < i; j++)
		{
			dist[i * natm + j] = sqrt((atm->value[atm(CENTER_IND, i)] - atm->value[atm(CENTER_IND, j)]) *
										  (atm->value[atm(CENTER_IND, i)] - atm->value[atm(CENTER_IND, j)]) +
									  (atm->value[atm(CENTER_IND, i) + 1] - atm->value[atm(CENTER_IND, j) + 1]) *
										  (atm->value[atm(CENTER_IND, i) + 1] - atm->value[atm(CENTER_IND, j) + 1]) +
									  (atm->value[atm(CENTER_IND, i) + 2] - atm->value[atm(CENTER_IND, j) + 2]) *
										  (atm->value[atm(CENTER_IND, i) + 2] - atm->value[atm(CENTER_IND, j) + 2]));
			dist[j * natm + i] = dist[i * natm + j];
			if (1)
			{
				nn_list[i][nn_num_list[i]] = j;
				nn_num_list[i]++;
			}
			if (1)
			{
				nn_list[j][nn_num_list[j]] = i;
				nn_num_list[j]++;
			}
		}
	}
#pragma omp parallel
	{
		int radii_order;
		double xi;
		double *layer;
#pragma omp for schedule(dynamic, 1) nowait
		for (int i = 0; i < natm; i++)
		{
			int ci = atm(ELEMENT_VAL, i);
			int row;
			if (atm(ELEMENT_VAL, i) <= 2)
				row = 1;
			if (atm(ELEMENT_VAL, i) <= 10)
				row = 2;
			else if (atm(ELEMENT_VAL, i) <= 18)
				row = 3;
			else if (atm(ELEMENT_VAL, i) <= 36)
				row = 4;
			else if (atm(ELEMENT_VAL, i) <= 54)
				row = 5;
			else
				radii_order = 6;
			radii_order = get_radii_num_opt(atm(ELEMENT_VAL, i), epsilon, beta);
			layer = row_pra[row - 1];
			xi = row_pra[row - 1][4];
			grid_t *tmp_grid = (grid_t *)malloc(sizeof(grid_t) * radii_order * angular_max_num);
			int tmp_grid_num;
			tmp_grid_num = get_atm_grid_opt(xi, layer, angular_index, radii_order, i, dist, atm, tmp_grid, nn_list[i], nn_num_list[i]);
#pragma omp critical
			{
				memcpy(&tmp_grid_mol[tol_num], tmp_grid, sizeof(grid_t) * tmp_grid_num);
				tol_num += tmp_grid_num;
			}
			free(tmp_grid);
		}
	}
	grids = (grid_t *)malloc(sizeof(grid_t) * tol_num);
	memcpy(grids, tmp_grid_mol, sizeof(grid_t) * tol_num);
	free(tmp_grid_mol);
	free(dist);
	*len = tol_num;
	free(nn_list[0]);
	free(nn_list);
	free(nn_num_list);
	return grids;
}

grids_info init_grid_opt(const char *opt_file, const mol_info mol)
{
	const bas_info bas = mol->bas;
	const atm_info atm = mol->atm;
	grids_info grids = (grids_info)malloc(sizeof(struct GridsInfo));
	grids->grids = get_mol_grid_opt(opt_file, mol, &grids->grid_num);
	grids->bas_extent = (double *)malloc(sizeof(double) * bas->nbas);
	grids->maxR = 0.;
	for (int i = 0; i < bas->nbas; i++)
	{
		grids->bas_extent[i] = cal_bas_extent(i, bas);
		if (grids->maxR < grids->bas_extent[i])
			grids->maxR = grids->bas_extent[i];
	}
	divide_block(grids, grids->maxR);
#pragma omp parallel
	{
#pragma omp for schedule(dynamic, 1) nowait
		for (int i = 0; i < grids->block_num; i++)
		{
			get_block_shell(grids->grids, grids->bas_extent, &grids->blocks[i], mol);
			grids->blocks[i].psi = NULL;
			grids->blocks[i].psix = NULL;
			grids->blocks[i].psiy = NULL;
			grids->blocks[i].psiz = NULL;
			grids->blocks[i].psixx = NULL;
			grids->blocks[i].psiyy = NULL;
			grids->blocks[i].psizz = NULL;
			grids->blocks[i].psixy = NULL;
			grids->blocks[i].psiyz = NULL;
			grids->blocks[i].psixz = NULL;
		}
	}
	grids->mol = mol;
	return grids;
}

grids_info init_cub_grid(const double epsilon, const double delta, const mol_info mol)
{
	const bas_info bas = mol->bas;
	const atm_info atm = mol->atm;
	double x_min, y_min, z_min;
	double x_max, y_max, z_max;
	x_min = atm->value[atm(CENTER_IND, 0) + 0];
	x_max = atm->value[atm(CENTER_IND, 0) + 0];
	y_min = atm->value[atm(CENTER_IND, 0) + 1];
	y_max = atm->value[atm(CENTER_IND, 0) + 1];
	z_min = atm->value[atm(CENTER_IND, 0) + 2];
	z_max = atm->value[atm(CENTER_IND, 0) + 2];
	for (int i = 0; i < atm->natm; i++)
	{
		double x_tmp, y_tmp, z_tmp;
		x_tmp = atm->value[atm(CENTER_IND, i) + 0];
		y_tmp = atm->value[atm(CENTER_IND, i) + 1];
		z_tmp = atm->value[atm(CENTER_IND, i) + 2];
		x_min = my_min(x_min, x_tmp);
		x_max = my_max(x_max, x_tmp);
		y_min = my_min(y_min, y_tmp);
		y_max = my_max(y_max, y_tmp);
		z_min = my_min(z_min, z_tmp);
		z_max = my_max(z_max, z_tmp);
	}
	x_min -= epsilon;
	x_max += epsilon;
	y_min -= epsilon;
	y_max += epsilon;
	z_min = 0;
	z_max = 0;
	int x_num = (x_max - x_min) / delta;
	int y_num = (y_max - y_min) / delta;
	int z_num = 1;
	double weight = delta * delta * delta;
	int tol_num = x_num * y_num * z_num;
	grids_info grids = (grids_info)malloc(sizeof(struct GridsInfo));
	grids->grid_num = tol_num;
	grids->grids = (grid_t *)malloc(sizeof(grid_t) * tol_num);
	for (int i = 0, uu = 0; i < x_num; i++)
		for (int j = 0; j < y_num; j++)
			for (int k = 0; k < z_num; k++, uu++)
			{
				grids->grids[uu].x = x_min + i * delta;
				grids->grids[uu].y = y_min + j * delta;
				grids->grids[uu].z = z_min + 1 * (z_max - z_min) / 2;
				grids->grids[uu].w = weight;
			}
	grids->bas_extent = (double *)malloc(sizeof(double) * bas->nbas);
	grids->maxR = 0.;
	for (int i = 0; i < bas->nbas; i++)
	{
		grids->bas_extent[i] = cal_bas_extent(i, bas);
		if (grids->maxR < grids->bas_extent[i])
			grids->maxR = grids->bas_extent[i];
	}
	divide_block(grids, grids->maxR);
// printf("NUB:%d %d\n",grids->block_num,tol_num);
#pragma omp parallel
	{
#pragma omp for schedule(dynamic, 1) nowait
		for (int i = 0; i < grids->block_num; i++)
		{
			get_block_shell(grids->grids, grids->bas_extent, &grids->blocks[i], mol);
			grids->blocks[i].psi = NULL;
			grids->blocks[i].psix = NULL;
			grids->blocks[i].psiy = NULL;
			grids->blocks[i].psiz = NULL;
			grids->blocks[i].psixx = NULL;
			grids->blocks[i].psiyy = NULL;
			grids->blocks[i].psizz = NULL;
			grids->blocks[i].psixy = NULL;
			grids->blocks[i].psiyz = NULL;
			grids->blocks[i].psixz = NULL;
		}
	}
	grids->mol = mol;
	return grids;
}
