#pragma once

#include <vector>

#include <Eigen/Core>

#include "vb/matrices/determinant_types.hpp"

namespace xmvb::vb {

using Matrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using ConstMatrixMap = Eigen::Map<const Matrix>;

struct SameSpinPhiResult {
  double one_electron_phi = 0.0;
  double total_phi = 0.0;
};

ConstMatrixMap map_column_major_matrix(
    const std::vector<double>& matrix_storage,
    int dimension);

std::vector<double> build_overlap_submatrix(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& ovlp_act,
    int n_orbitals);

Matrix calc_cofactor_1st(
    const DeterminantOverlapResult& det_ovlp_result);

Matrix build_inverse_overlap_submatrix_from_result(
    const DeterminantOverlapResult& det_ovlp_result);

SameSpinPhiResult compute_same_spin_original_phi(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& h1e_act,
    int n_active_orbitals,
    const std::vector<double>& packed_active_two_electron_integrals,
    const DeterminantOverlapResult& det_ovlp_result,
    Matrix* inverse_overlap_gradient);

double compute_opposite_spin_original_phi(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const DeterminantOverlapResult& alpha_overlap_result,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const DeterminantOverlapResult& beta_overlap_result,
    const std::vector<double>& packed_active_two_electron_integrals,
    Matrix* alpha_inverse_overlap_gradient,
    Matrix* beta_inverse_overlap_gradient);

void accumulate_spin_overlap_gradient(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const DeterminantOverlapResult& det_ovlp_result,
    double determinant_overlap_weight,
    const Matrix& inverse_overlap_gradient,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient);

}  // namespace xmvb::vb
