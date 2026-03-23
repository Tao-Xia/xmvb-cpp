#pragma once

#include <vector>

#include <Eigen/Core>

#include "vb/matrices/determinant_overlap_result.hpp"

namespace xmvb::vb {

using ColumnMajorMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using ConstColumnMajorMatrixMap = Eigen::Map<const ColumnMajorMatrixXd>;

struct SameSpinBiorthogonalPhiResult {
  double one_electron_phi = 0.0;
  double total_phi = 0.0;
};

struct SameSpinBiorthogonalHamiltonianResult {
  double one_electron_hamiltonian = 0.0;
  double total_hamiltonian = 0.0;
};

ConstColumnMajorMatrixMap map_column_major_matrix(
    const std::vector<double>& matrix_storage,
    int dimension);

std::vector<double> matrix_to_column_major_storage(const ColumnMajorMatrixXd& matrix);

std::vector<double> build_overlap_submatrix(
    const std::vector<int>& occupied_orbitals_left,
    const std::vector<int>& occupied_orbitals_right,
    const std::vector<double>& basis_overlap_matrix,
    int n_orbitals);

ColumnMajorMatrixXd build_first_order_cofactor_matrix_from_result(
    const DeterminantOverlapResult& overlap_result);

ColumnMajorMatrixXd build_inverse_overlap_submatrix_from_result(
    const DeterminantOverlapResult& overlap_result);

SameSpinBiorthogonalHamiltonianResult compute_same_spin_biorthogonal_hamiltonian(
    const std::vector<int>& occupied_orbitals_left,
    const std::vector<int>& occupied_orbitals_right,
    const std::vector<double>& active_one_electron_matrix,
    int n_active_orbitals,
    const std::vector<double>& packed_active_two_electron_integrals,
    const DeterminantOverlapResult& overlap_result);

SameSpinBiorthogonalPhiResult compute_same_spin_original_phi(
    const std::vector<int>& occupied_orbitals_left,
    const std::vector<int>& occupied_orbitals_right,
    const std::vector<double>& active_one_electron_matrix,
    int n_active_orbitals,
    const std::vector<double>& packed_active_two_electron_integrals,
    const DeterminantOverlapResult& overlap_result,
    ColumnMajorMatrixXd* inverse_overlap_gradient);

SameSpinBiorthogonalPhiResult compute_same_spin_biorthogonal_phi(
    const std::vector<int>& occupied_orbitals_left,
    const std::vector<int>& occupied_orbitals_right,
    const std::vector<double>& active_one_electron_matrix,
    int n_active_orbitals,
    const std::vector<double>& packed_active_two_electron_integrals,
    const DeterminantOverlapResult& overlap_result,
    ColumnMajorMatrixXd* inverse_overlap_gradient);

double compute_opposite_spin_biorthogonal_hamiltonian(
    const std::vector<int>& alpha_occupied_orbitals_left,
    const std::vector<int>& alpha_occupied_orbitals_right,
    const DeterminantOverlapResult& alpha_overlap_result,
    const std::vector<int>& beta_occupied_orbitals_left,
    const std::vector<int>& beta_occupied_orbitals_right,
    const DeterminantOverlapResult& beta_overlap_result,
    const std::vector<double>& packed_active_two_electron_integrals);

double compute_opposite_spin_original_phi(
    const std::vector<int>& alpha_occupied_orbitals_left,
    const std::vector<int>& alpha_occupied_orbitals_right,
    const DeterminantOverlapResult& alpha_overlap_result,
    const std::vector<int>& beta_occupied_orbitals_left,
    const std::vector<int>& beta_occupied_orbitals_right,
    const DeterminantOverlapResult& beta_overlap_result,
    const std::vector<double>& packed_active_two_electron_integrals,
    ColumnMajorMatrixXd* alpha_inverse_overlap_gradient,
    ColumnMajorMatrixXd* beta_inverse_overlap_gradient);

double compute_opposite_spin_biorthogonal_phi(
    const std::vector<int>& alpha_occupied_orbitals_left,
    const std::vector<int>& alpha_occupied_orbitals_right,
    const DeterminantOverlapResult& alpha_overlap_result,
    const std::vector<int>& beta_occupied_orbitals_left,
    const std::vector<int>& beta_occupied_orbitals_right,
    const DeterminantOverlapResult& beta_overlap_result,
    const std::vector<double>& packed_active_two_electron_integrals,
    ColumnMajorMatrixXd* alpha_inverse_overlap_gradient,
    ColumnMajorMatrixXd* beta_inverse_overlap_gradient);

void accumulate_spin_overlap_gradient(
    const std::vector<int>& occupied_orbitals_left,
    const std::vector<int>& occupied_orbitals_right,
    const DeterminantOverlapResult& overlap_result,
    double determinant_overlap_weight,
    const ColumnMajorMatrixXd& inverse_overlap_gradient,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient);

}  // namespace xmvb::vb
