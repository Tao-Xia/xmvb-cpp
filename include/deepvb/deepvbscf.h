// deepvbscf.hpp
#pragma once
#include <vector>
#include <iomanip>
#include <string>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <lapacke.h>
#include <cblas.h>
#include <cmath>
#include <complex>
#include <torch/script.h>
#include <torch/torch.h>
extern "C"
{
#include "vb/vb.h"
#include "deepvb/deepvb.h"
}

class deepvbscf_5
{
public:
    // vb structure
    vb_info vb;
    int nao = vb->nao;
    int nb = vb->nb;
    int nstr = vb->nstr;
    int naeb = (vb->nae - vb->nmul + 1) / 2;
    int naea = vb->nae - naeb;
    // matrix data for vbscf
    std::vector<double> sso;
    std::vector<double> hho;
    std::vector<double> hvb_1e;
    std::vector<double> svb_1e;
    std::vector<double> hvb_1e_alpha;
    std::vector<double> svb_1e_alpha;
    std::vector<double> hvb_1e_beta;
    std::vector<double> svb_1e_beta;
    std::vector<double> sorb_alpha;
    std::vector<double> sorb_beta;
    std::vector<double> horb_alpha;
    std::vector<double> horb_beta;
    std::vector<double> rdm_alpha;
    std::vector<double> rdm_beta;
    std::vector<double> train_data;
    // determinant encoding
    std::vector<std::vector<int>> alpha_det;
    std::vector<std::vector<int>> beta_det;
    std::vector<int> det2str_idx;
    std::vector<double> det_sign;
    // gradient related variables
    std::vector<double> G22;
    std::vector<double> Q22;
    std::vector<double> grda;
    std::vector<double> grdv;
    std::vector<double> grdbas;
    std::vector<double> grad;

    // save data for deep learning model
    std::string save_folder;
    // constructor: allocate space according to vb_info
    deepvbscf_5(vb_info vb_in);
    /*expand vb structure to slater determinants*/
    void expand_vbstr_to_det();
    void rdm_scf(
        const std::vector<int> &alpha_det_i,
        const std::vector<int> &alpha_det_j,
        const std::vector<int> &beta_det_i,
        const std::vector<int> &beta_det_j,
        int i, int j,
        int idx_i, int idx_j, 
        double sign
    );
    void compute_strpair();
    void save_deep_data();
    // void cofactor1(double *A, int n, double *C1, double *det);
};

/// @brief Include eigen calculation and matrix diagonalization functions
/// @note EigenSolver is a static class, which contains only static functions.
class EigenSolver
{
public:
    static void ortho_hamiltonian(double *H, double *H_ortho, const double *S, int n);
    static void energy_compute(double *energy, double *col, const double *hvb, const double *svb, const double enuc, const int nstr, vb_info vb_str);
};


class deepvbscf_tdm
{
public:
    vb_info vb;
    std::vector<double> tdm_alpha_, tdm_beta_, sinva_, sinvb_, ssa_, ssb_, hvb_1det_, svb_1det_, sorb_, Cl_, Cr_, pa_, pb_, wrk_, temp_alpha_, temp_beta_;
    std::vector<int> nsl_, nsr_;
    int nalpha, nbeta, nb;
    deepvbscf_tdm(vb_info vb_in);
    void tdm_scf(int *str_i, int *str_j, int i, int j);
    void compute_str_pair();
    // void invmat(std::vector<double> &A, int n, double &det);
};

void normalize_hvb(double *hvb, double *svb, int nstr);