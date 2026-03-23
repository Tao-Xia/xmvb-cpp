#include "mol/xint.h"
static void nuc_VRR_0(double* buf, double omega, double  lr_frac, double sr_frac, double* cache, double half_xi_, double xi, const double* PG, const double PG_dist2, const double coe);
static void nuc_VRR_1(double* buf, double omega, double  lr_frac, double sr_frac, double* cache, double half_xi_, double xi, const double* PG, const double PG_dist2, const double coe);
static void nuc_VRR_2(double* buf, double omega, double  lr_frac, double sr_frac, double* cache, double half_xi_, double xi, const double* PG, const double PG_dist2, const double coe);
static void nuc_VRR_3(double* buf, double omega, double  lr_frac, double sr_frac, double* cache, double half_xi_, double xi, const double* PG, const double PG_dist2, const double coe);
static void nuc_VRR_4(double* buf, double omega, double  lr_frac, double sr_frac, double* cache, double half_xi_, double xi, const double* PG, const double PG_dist2, const double coe);
static void nuc_VRR_5(double* buf, double omega, double  lr_frac, double sr_frac, double* cache, double half_xi_, double xi, const double* PG, const double PG_dist2, const double coe);
typedef void (*NUC_VRR)(double*, double, double, double, double*, double, double, const double*, const double, const double);
static NUC_VRR nuc_vrr_list[]={nuc_VRR_0,nuc_VRR_1,nuc_VRR_2,nuc_VRR_3,nuc_VRR_4,nuc_VRR_5};
static int buf_len_list[]={1,3,6,10,15,21};

static void nuc_VRR_0(double* buf, double omega, double  lr_frac, double sr_frac, double* cache, double half_xi_, double xi, const double* PG, const double PG_dist2, const double coe)
{
	double T_0, T_1;
	double scheme_1[30], scheme_0[30];
	double scheme_0_0_0[1];
	double theta;
	if (fabs(omega)<1E-8) {
		T_0=xi*PG_dist2;
		cal_boys(0, T_0, &scheme_0[0]);
		for (int i=0; i<=0; i++)cache[i]=4*half_xi_*MY_PI*lr_frac*scheme_0[i];
	}
	else{
		theta=sqrt(omega*omega/(omega*omega+xi));
		T_1=xi*PG_dist2*theta*theta;
		T_0=xi*PG_dist2;
		cal_boys(0, T_0, &scheme_0[0]);
		cal_boys(0, T_1, &scheme_1[0]);
		for (int i=0; i<=0; i++)cache[i]=4*half_xi_*MY_PI*(lr_frac*scheme_0[i]+sr_frac*theta*pow(theta,2*i)*scheme_1[i]);
	}
	buf[0]+=coe*cache[0];
	return;
} 

static void nuc_VRR_1(double* buf, double omega, double  lr_frac, double sr_frac, double* cache, double half_xi_, double xi, const double* PG, const double PG_dist2, const double coe)
{

	double T_0, T_1;
	double scheme_1[30], scheme_0[30];
	double scheme_0_0_0[2];
	double theta;
	if (fabs(omega)<1E-8) {
		T_0=xi*PG_dist2;
		cal_boys(1, T_0, &scheme_0[0]);
		for (int i=0; i<=1; i++)cache[i]=4*half_xi_*MY_PI*lr_frac*scheme_0[i];
	}
	else{
		theta=sqrt(omega*omega/(omega*omega+xi));
		T_1=xi*PG_dist2*theta*theta;
		T_0=xi*PG_dist2;
		cal_boys(1, T_0, &scheme_0[0]);
		cal_boys(1, T_1, &scheme_1[0]);
		for (int i=0; i<=1; i++)cache[i]=4*half_xi_*MY_PI*(lr_frac*scheme_0[i]+sr_frac*theta*pow(theta,2*i)*scheme_1[i]);
	}
	for (int i=0; i<1; i++)cache[2+i]=-PG[0]*cache[0+i+1];
	for (int i=0; i<1; i++)cache[3+i]=-PG[1]*cache[0+i+1];
	for (int i=0; i<1; i++)cache[4+i]=-PG[2]*cache[0+i+1];
	buf[0]+=coe*cache[2];
	buf[1]+=coe*cache[3];
	buf[2]+=coe*cache[4];
	return;
} 

static void nuc_VRR_2(double* buf, double omega, double  lr_frac, double sr_frac, double* cache, double half_xi_, double xi, const double* PG, const double PG_dist2, const double coe)
{

	double T_0, T_1;
	double scheme_1[30], scheme_0[30];
	double scheme_0_0_0[3];
	double theta;
	if (fabs(omega)<1E-8) {
		T_0=xi*PG_dist2;
		cal_boys(2, T_0, &scheme_0[0]);
		for (int i=0; i<=2; i++)cache[i]=4*half_xi_*MY_PI*lr_frac*scheme_0[i];
	}
	else{
		theta=sqrt(omega*omega/(omega*omega+xi));
		T_1=xi*PG_dist2*theta*theta;
		T_0=xi*PG_dist2;
		cal_boys(2, T_0, &scheme_0[0]);
		cal_boys(2, T_1, &scheme_1[0]);
		for (int i=0; i<=2; i++)cache[i]=4*half_xi_*MY_PI*(lr_frac*scheme_0[i]+sr_frac*theta*pow(theta,2*i)*scheme_1[i]);
	}
	for (int i=0; i<2; i++)cache[3+i]=-PG[0]*cache[0+i+1];
	for (int i=0; i<2; i++)cache[5+i]=-PG[1]*cache[0+i+1];
	for (int i=0; i<2; i++)cache[7+i]=-PG[2]*cache[0+i+1];
	for (int i=0; i<1; i++)cache[9+i]=-PG[0]*cache[3+i+1]+1*half_xi_*(cache[0+i]-cache[0+i+1]);
	for (int i=0; i<1; i++)cache[10+i]=-PG[0]*cache[5+i+1];
	for (int i=0; i<1; i++)cache[11+i]=-PG[0]*cache[7+i+1];
	for (int i=0; i<1; i++)cache[12+i]=-PG[1]*cache[5+i+1]+1*half_xi_*(cache[0+i]-cache[0+i+1]);
	for (int i=0; i<1; i++)cache[13+i]=-PG[1]*cache[7+i+1];
	for (int i=0; i<1; i++)cache[14+i]=-PG[2]*cache[7+i+1]+1*half_xi_*(cache[0+i]-cache[0+i+1]);
	buf[0]+=coe*cache[9];
	buf[1]+=coe*cache[10];
	buf[2]+=coe*cache[11];
	buf[3]+=coe*cache[12];
	buf[4]+=coe*cache[13];
	buf[5]+=coe*cache[14];
	return;
} 

static void nuc_VRR_3(double* buf, double omega, double  lr_frac, double sr_frac, double* cache, double half_xi_, double xi, const double* PG, const double PG_dist2, const double coe)
{

	double T_0, T_1;
	double scheme_1[30], scheme_0[30];
	double scheme_0_0_0[4];
	double theta;
	if (fabs(omega)<1E-8) {
		T_0=xi*PG_dist2;
		cal_boys(3, T_0, &scheme_0[0]);
		for (int i=0; i<=3; i++)cache[i]=4*half_xi_*MY_PI*lr_frac*scheme_0[i];
	}
	else{
		theta=sqrt(omega*omega/(omega*omega+xi));
		T_1=xi*PG_dist2*theta*theta;
		T_0=xi*PG_dist2;
		cal_boys(3, T_0, &scheme_0[0]);
		cal_boys(3, T_1, &scheme_1[0]);
		for (int i=0; i<=3; i++)cache[i]=4*half_xi_*MY_PI*(lr_frac*scheme_0[i]+sr_frac*theta*pow(theta,2*i)*scheme_1[i]);
	}
	for (int i=0; i<3; i++)cache[4+i]=-PG[0]*cache[0+i+1];
	for (int i=0; i<3; i++)cache[7+i]=-PG[1]*cache[0+i+1];
	for (int i=0; i<3; i++)cache[10+i]=-PG[2]*cache[0+i+1];
	for (int i=0; i<2; i++)cache[13+i]=-PG[0]*cache[4+i+1]+1*half_xi_*(cache[0+i]-cache[0+i+1]);
	for (int i=0; i<2; i++)cache[15+i]=-PG[0]*cache[7+i+1];
	for (int i=0; i<2; i++)cache[17+i]=-PG[0]*cache[10+i+1];
	for (int i=0; i<2; i++)cache[19+i]=-PG[1]*cache[7+i+1]+1*half_xi_*(cache[0+i]-cache[0+i+1]);
	for (int i=0; i<2; i++)cache[21+i]=-PG[1]*cache[10+i+1];
	for (int i=0; i<2; i++)cache[23+i]=-PG[2]*cache[10+i+1]+1*half_xi_*(cache[0+i]-cache[0+i+1]);
	for (int i=0; i<1; i++)cache[25+i]=-PG[0]*cache[13+i+1]+2*half_xi_*(cache[4+i]-cache[4+i+1]);
	for (int i=0; i<1; i++)cache[26+i]=-PG[1]*cache[13+i+1];
	for (int i=0; i<1; i++)cache[27+i]=-PG[2]*cache[13+i+1];
	for (int i=0; i<1; i++)cache[28+i]=-PG[0]*cache[19+i+1];
	for (int i=0; i<1; i++)cache[29+i]=-PG[0]*cache[21+i+1];
	for (int i=0; i<1; i++)cache[30+i]=-PG[0]*cache[23+i+1];
	for (int i=0; i<1; i++)cache[31+i]=-PG[1]*cache[19+i+1]+2*half_xi_*(cache[7+i]-cache[7+i+1]);
	for (int i=0; i<1; i++)cache[32+i]=-PG[2]*cache[19+i+1];
	for (int i=0; i<1; i++)cache[33+i]=-PG[1]*cache[23+i+1];
	for (int i=0; i<1; i++)cache[34+i]=-PG[2]*cache[23+i+1]+2*half_xi_*(cache[10+i]-cache[10+i+1]);
	buf[0]+=coe*cache[25];
	buf[1]+=coe*cache[26];
	buf[2]+=coe*cache[27];
	buf[3]+=coe*cache[28];
	buf[4]+=coe*cache[29];
	buf[5]+=coe*cache[30];
	buf[6]+=coe*cache[31];
	buf[7]+=coe*cache[32];
	buf[8]+=coe*cache[33];
	buf[9]+=coe*cache[34];
	return;
} 

static void nuc_VRR_4(double* buf, double omega, double  lr_frac, double sr_frac, double* cache, double half_xi_, double xi, const double* PG, const double PG_dist2, const double coe)
{

	double T_0, T_1;
	double scheme_1[30], scheme_0[30];
	double scheme_0_0_0[5];
	double theta;
	if (fabs(omega)<1E-8) {
		T_0=xi*PG_dist2;
		cal_boys(4, T_0, &scheme_0[0]);
		for (int i=0; i<=4; i++)cache[i]=4*half_xi_*MY_PI*lr_frac*scheme_0[i];
	}
	else{
		theta=sqrt(omega*omega/(omega*omega+xi));
		T_1=xi*PG_dist2*theta*theta;
		T_0=xi*PG_dist2;
		cal_boys(4, T_0, &scheme_0[0]);
		cal_boys(4, T_1, &scheme_1[0]);
		for (int i=0; i<=4; i++)cache[i]=4*half_xi_*MY_PI*(lr_frac*scheme_0[i]+sr_frac*theta*pow(theta,2*i)*scheme_1[i]);
	}
	for (int i=0; i<4; i++)cache[5+i]=-PG[0]*cache[0+i+1];
	for (int i=0; i<4; i++)cache[9+i]=-PG[1]*cache[0+i+1];
	for (int i=0; i<4; i++)cache[13+i]=-PG[2]*cache[0+i+1];
	for (int i=0; i<3; i++)cache[17+i]=-PG[0]*cache[5+i+1]+1*half_xi_*(cache[0+i]-cache[0+i+1]);
	for (int i=0; i<3; i++)cache[20+i]=-PG[0]*cache[9+i+1];
	for (int i=0; i<3; i++)cache[23+i]=-PG[0]*cache[13+i+1];
	for (int i=0; i<3; i++)cache[26+i]=-PG[1]*cache[9+i+1]+1*half_xi_*(cache[0+i]-cache[0+i+1]);
	for (int i=0; i<3; i++)cache[29+i]=-PG[1]*cache[13+i+1];
	for (int i=0; i<3; i++)cache[32+i]=-PG[2]*cache[13+i+1]+1*half_xi_*(cache[0+i]-cache[0+i+1]);
	for (int i=0; i<2; i++)cache[35+i]=-PG[0]*cache[17+i+1]+2*half_xi_*(cache[5+i]-cache[5+i+1]);
	for (int i=0; i<2; i++)cache[37+i]=-PG[1]*cache[17+i+1];
	for (int i=0; i<2; i++)cache[39+i]=-PG[2]*cache[17+i+1];
	for (int i=0; i<2; i++)cache[41+i]=-PG[0]*cache[26+i+1];
	for (int i=0; i<2; i++)cache[43+i]=-PG[0]*cache[29+i+1];
	for (int i=0; i<2; i++)cache[45+i]=-PG[0]*cache[32+i+1];
	for (int i=0; i<2; i++)cache[47+i]=-PG[1]*cache[26+i+1]+2*half_xi_*(cache[9+i]-cache[9+i+1]);
	for (int i=0; i<2; i++)cache[49+i]=-PG[2]*cache[26+i+1];
	for (int i=0; i<2; i++)cache[51+i]=-PG[1]*cache[32+i+1];
	for (int i=0; i<2; i++)cache[53+i]=-PG[2]*cache[32+i+1]+2*half_xi_*(cache[13+i]-cache[13+i+1]);
	for (int i=0; i<1; i++)cache[55+i]=-PG[0]*cache[35+i+1]+3*half_xi_*(cache[17+i]-cache[17+i+1]);
	for (int i=0; i<1; i++)cache[56+i]=-PG[1]*cache[35+i+1];
	for (int i=0; i<1; i++)cache[57+i]=-PG[2]*cache[35+i+1];
	for (int i=0; i<1; i++)cache[58+i]=-PG[0]*cache[41+i+1]+1*half_xi_*(cache[26+i]-cache[26+i+1]);
	for (int i=0; i<1; i++)cache[59+i]=-PG[1]*cache[39+i+1];
	for (int i=0; i<1; i++)cache[60+i]=-PG[0]*cache[45+i+1]+1*half_xi_*(cache[32+i]-cache[32+i+1]);
	for (int i=0; i<1; i++)cache[61+i]=-PG[0]*cache[47+i+1];
	for (int i=0; i<1; i++)cache[62+i]=-PG[0]*cache[49+i+1];
	for (int i=0; i<1; i++)cache[63+i]=-PG[0]*cache[51+i+1];
	for (int i=0; i<1; i++)cache[64+i]=-PG[0]*cache[53+i+1];
	for (int i=0; i<1; i++)cache[65+i]=-PG[1]*cache[47+i+1]+3*half_xi_*(cache[26+i]-cache[26+i+1]);
	for (int i=0; i<1; i++)cache[66+i]=-PG[2]*cache[47+i+1];
	for (int i=0; i<1; i++)cache[67+i]=-PG[1]*cache[51+i+1]+1*half_xi_*(cache[32+i]-cache[32+i+1]);
	for (int i=0; i<1; i++)cache[68+i]=-PG[1]*cache[53+i+1];
	for (int i=0; i<1; i++)cache[69+i]=-PG[2]*cache[53+i+1]+3*half_xi_*(cache[32+i]-cache[32+i+1]);
	buf[0]+=coe*cache[55];
	buf[1]+=coe*cache[56];
	buf[2]+=coe*cache[57];
	buf[3]+=coe*cache[58];
	buf[4]+=coe*cache[59];
	buf[5]+=coe*cache[60];
	buf[6]+=coe*cache[61];
	buf[7]+=coe*cache[62];
	buf[8]+=coe*cache[63];
	buf[9]+=coe*cache[64];
	buf[10]+=coe*cache[65];
	buf[11]+=coe*cache[66];
	buf[12]+=coe*cache[67];
	buf[13]+=coe*cache[68];
	buf[14]+=coe*cache[69];
	return;
} 

static void nuc_VRR_5(double* buf, double omega, double  lr_frac, double sr_frac, double* cache, double half_xi_, double xi, const double* PG, const double PG_dist2, const double coe)
{

	double T_0, T_1;
	double scheme_1[30], scheme_0[30];
	double scheme_0_0_0[6];
	double theta;
	if (fabs(omega)<1E-8) {
		T_0=xi*PG_dist2;
		cal_boys(5, T_0, &scheme_0[0]);
		for (int i=0; i<=5; i++)cache[i]=4*half_xi_*MY_PI*lr_frac*scheme_0[i];
	}
	else{
		theta=sqrt(omega*omega/(omega*omega+xi));
		T_1=xi*PG_dist2*theta*theta;
		T_0=xi*PG_dist2;
		cal_boys(5, T_0, &scheme_0[0]);
		cal_boys(5, T_1, &scheme_1[0]);
		for (int i=0; i<=5; i++)cache[i]=4*half_xi_*MY_PI*(lr_frac*scheme_0[i]+sr_frac*theta*pow(theta,2*i)*scheme_1[i]);
	}
	for (int i=0; i<5; i++)cache[6+i]=-PG[0]*cache[0+i+1];
	for (int i=0; i<5; i++)cache[11+i]=-PG[1]*cache[0+i+1];
	for (int i=0; i<5; i++)cache[16+i]=-PG[2]*cache[0+i+1];
	for (int i=0; i<4; i++)cache[21+i]=-PG[0]*cache[6+i+1]+1*half_xi_*(cache[0+i]-cache[0+i+1]);
	for (int i=0; i<4; i++)cache[25+i]=-PG[0]*cache[11+i+1];
	for (int i=0; i<4; i++)cache[29+i]=-PG[0]*cache[16+i+1];
	for (int i=0; i<4; i++)cache[33+i]=-PG[1]*cache[11+i+1]+1*half_xi_*(cache[0+i]-cache[0+i+1]);
	for (int i=0; i<4; i++)cache[37+i]=-PG[1]*cache[16+i+1];
	for (int i=0; i<4; i++)cache[41+i]=-PG[2]*cache[16+i+1]+1*half_xi_*(cache[0+i]-cache[0+i+1]);
	for (int i=0; i<3; i++)cache[45+i]=-PG[0]*cache[21+i+1]+2*half_xi_*(cache[6+i]-cache[6+i+1]);
	for (int i=0; i<3; i++)cache[48+i]=-PG[1]*cache[21+i+1];
	for (int i=0; i<3; i++)cache[51+i]=-PG[2]*cache[21+i+1];
	for (int i=0; i<3; i++)cache[54+i]=-PG[0]*cache[33+i+1];
	for (int i=0; i<3; i++)cache[57+i]=-PG[0]*cache[37+i+1];
	for (int i=0; i<3; i++)cache[60+i]=-PG[0]*cache[41+i+1];
	for (int i=0; i<3; i++)cache[63+i]=-PG[1]*cache[33+i+1]+2*half_xi_*(cache[11+i]-cache[11+i+1]);
	for (int i=0; i<3; i++)cache[66+i]=-PG[2]*cache[33+i+1];
	for (int i=0; i<3; i++)cache[69+i]=-PG[1]*cache[41+i+1];
	for (int i=0; i<3; i++)cache[72+i]=-PG[2]*cache[41+i+1]+2*half_xi_*(cache[16+i]-cache[16+i+1]);
	for (int i=0; i<2; i++)cache[75+i]=-PG[0]*cache[45+i+1]+3*half_xi_*(cache[21+i]-cache[21+i+1]);
	for (int i=0; i<2; i++)cache[77+i]=-PG[1]*cache[45+i+1];
	for (int i=0; i<2; i++)cache[79+i]=-PG[2]*cache[45+i+1];
	for (int i=0; i<2; i++)cache[81+i]=-PG[0]*cache[54+i+1]+1*half_xi_*(cache[33+i]-cache[33+i+1]);
	for (int i=0; i<2; i++)cache[83+i]=-PG[1]*cache[51+i+1];
	for (int i=0; i<2; i++)cache[85+i]=-PG[0]*cache[60+i+1]+1*half_xi_*(cache[41+i]-cache[41+i+1]);
	for (int i=0; i<2; i++)cache[87+i]=-PG[0]*cache[63+i+1];
	for (int i=0; i<2; i++)cache[89+i]=-PG[0]*cache[66+i+1];
	for (int i=0; i<2; i++)cache[91+i]=-PG[0]*cache[69+i+1];
	for (int i=0; i<2; i++)cache[93+i]=-PG[0]*cache[72+i+1];
	for (int i=0; i<2; i++)cache[95+i]=-PG[1]*cache[63+i+1]+3*half_xi_*(cache[33+i]-cache[33+i+1]);
	for (int i=0; i<2; i++)cache[97+i]=-PG[2]*cache[63+i+1];
	for (int i=0; i<2; i++)cache[99+i]=-PG[1]*cache[69+i+1]+1*half_xi_*(cache[41+i]-cache[41+i+1]);
	for (int i=0; i<2; i++)cache[101+i]=-PG[1]*cache[72+i+1];
	for (int i=0; i<2; i++)cache[103+i]=-PG[2]*cache[72+i+1]+3*half_xi_*(cache[41+i]-cache[41+i+1]);
	for (int i=0; i<1; i++)cache[105+i]=-PG[0]*cache[75+i+1]+4*half_xi_*(cache[45+i]-cache[45+i+1]);
	for (int i=0; i<1; i++)cache[106+i]=-PG[1]*cache[75+i+1];
	for (int i=0; i<1; i++)cache[107+i]=-PG[2]*cache[75+i+1];
	for (int i=0; i<1; i++)cache[108+i]=-PG[0]*cache[81+i+1]+2*half_xi_*(cache[54+i]-cache[54+i+1]);
	for (int i=0; i<1; i++)cache[109+i]=-PG[1]*cache[79+i+1];
	for (int i=0; i<1; i++)cache[110+i]=-PG[0]*cache[85+i+1]+2*half_xi_*(cache[60+i]-cache[60+i+1]);
	for (int i=0; i<1; i++)cache[111+i]=-PG[0]*cache[87+i+1]+1*half_xi_*(cache[63+i]-cache[63+i+1]);
	for (int i=0; i<1; i++)cache[112+i]=-PG[2]*cache[81+i+1];
	for (int i=0; i<1; i++)cache[113+i]=-PG[1]*cache[85+i+1];
	for (int i=0; i<1; i++)cache[114+i]=-PG[0]*cache[93+i+1]+1*half_xi_*(cache[72+i]-cache[72+i+1]);
	for (int i=0; i<1; i++)cache[115+i]=-PG[0]*cache[95+i+1];
	for (int i=0; i<1; i++)cache[116+i]=-PG[0]*cache[97+i+1];
	for (int i=0; i<1; i++)cache[117+i]=-PG[0]*cache[99+i+1];
	for (int i=0; i<1; i++)cache[118+i]=-PG[0]*cache[101+i+1];
	for (int i=0; i<1; i++)cache[119+i]=-PG[0]*cache[103+i+1];
	for (int i=0; i<1; i++)cache[120+i]=-PG[1]*cache[95+i+1]+4*half_xi_*(cache[63+i]-cache[63+i+1]);
	for (int i=0; i<1; i++)cache[121+i]=-PG[2]*cache[95+i+1];
	for (int i=0; i<1; i++)cache[122+i]=-PG[1]*cache[99+i+1]+2*half_xi_*(cache[69+i]-cache[69+i+1]);
	for (int i=0; i<1; i++)cache[123+i]=-PG[1]*cache[101+i+1]+1*half_xi_*(cache[72+i]-cache[72+i+1]);
	for (int i=0; i<1; i++)cache[124+i]=-PG[1]*cache[103+i+1];
	for (int i=0; i<1; i++)cache[125+i]=-PG[2]*cache[103+i+1]+4*half_xi_*(cache[72+i]-cache[72+i+1]);
	buf[0]+=coe*cache[105];
	buf[1]+=coe*cache[106];
	buf[2]+=coe*cache[107];
	buf[3]+=coe*cache[108];
	buf[4]+=coe*cache[109];
	buf[5]+=coe*cache[110];
	buf[6]+=coe*cache[111];
	buf[7]+=coe*cache[112];
	buf[8]+=coe*cache[113];
	buf[9]+=coe*cache[114];
	buf[10]+=coe*cache[115];
	buf[11]+=coe*cache[116];
	buf[12]+=coe*cache[117];
	buf[13]+=coe*cache[118];
	buf[14]+=coe*cache[119];
	buf[15]+=coe*cache[120];
	buf[16]+=coe*cache[121];
	buf[17]+=coe*cache[122];
	buf[18]+=coe*cache[123];
	buf[19]+=coe*cache[124];
	buf[20]+=coe*cache[125];
	return;
} 

/* |i)_g=\int{\chi_i/r_g}dr */
extern void xint_2c1e(Vector buf, const int shla, const double G[3], xint_info xint)
{
    double sr_frac = xint->sr_frac; 
    double lr_frac = xint->lr_frac; 
    double omega = xint->omega;
    const mol_info mol=xint->mol;
    const atm_info atm = xint->mol->atm;
    const bas_info aux = xint->aux;
    int li=aux(ANGULAR_VAL,shla);
    int di=aux(NPRIM_VAL,shla);
    int buf_len=buf_len_list[li];
    double AG[3], AG_dist2;
    const Vector A = get_atm_coord(aux(ATOM_IND, shla), atm);
    const double* coea=get_bas_coe(shla, aux);
    const double* expa=get_bas_exp(shla, aux);
    NUC_VRR nuc_vrr=nuc_vrr_list[li];
    AG[0]=A[0]-G[0];
    AG[1]=A[1]-G[1];
    AG[2]=A[2]-G[2];
    memset(buf,0,sizeof(double)*buf_len);
    AG_dist2=AG[0]*AG[0]+AG[1]*AG[1]+AG[2]*AG[2];
    for (int i=0; i<di; i++)
	{
		double half_xi_=0.5/expa[i];
    	nuc_vrr(buf,omega,lr_frac,sr_frac,xint->cache,half_xi_,expa[i],AG,AG_dist2,coea[i]);
	}
    return;
}
