#pragma once

#include <vector>

#include <Eigen/Core>

#include "vb/matrices/determinant_overlap_resolver.hpp"
#include "vb/matrices/full_determinant_pair_evaluator.hpp"
#include "vb/matrices/determinant_types.hpp"
#include "vb/orbital/active_space_two_electron_result.hpp"

namespace xmvb::vb {

struct ActiveSpaceTwoElectronView;

struct SameSpinPhiResult {
  double one_electron_phi = 0.0;
  double total_phi = 0.0;
};

std::vector<double> build_overlap_submatrix(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& ovlp_act,
    int n_orbitals);

Eigen::MatrixXd calc_cofactor_1st(
    const DeterminantOverlapResult& det_ovlp_result);

bool projection_has_any_payload(
    const OppositeSpinPackedPairProjection& projection);

int infer_n_packed_active_pairs(
    const std::vector<SpinDeterminantPairEvaluation>& ordered_spin_pair_cache,
    const char* spin_label);

void validate_sparse_projection_coefficients(
    const OppositeSpinPackedPairProjection& projection,
    int n_packed_active_pairs,
    const char* projection_label);

void gather_dense_submatrix(
    const Eigen::MatrixXd& global_matrix,
    const std::vector<int>& row_indices,
    const std::vector<int>& column_indices,
    Eigen::MatrixXd* local_matrix);

int limited_packed_pair_block_size(
    int n_packed_active_pairs,
    int requested_block_size,
    std::size_t per_matrix_bytes,
    std::size_t bytes_budget);

void cache_first_order_cofactor(
    DeterminantOverlapResult* det_ovlp_result);

bool has_opposite_spin_first_order_projection(
    const OppositeSpinPairCache& pair_cache);

bool has_opposite_spin_inverse_projection(
    const OppositeSpinPairCache& pair_cache);

double contract_opposite_spin_first_order_projections(
    const OppositeSpinPairCache& alpha_pair_cache,
    const OppositeSpinPairCache& beta_pair_cache);

double project_sparse_projection_onto_packed_pair(
    const OppositeSpinPackedPairProjection& sparse_projection,
    int target_packed_pair_index,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals);

Eigen::VectorXd gather_projected_values_for_packed_pair_indices(
    const OppositeSpinPackedPairProjection& sparse_projection,
    const std::vector<int>& target_packed_pair_indices,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals);

Eigen::MatrixXd build_inverse_overlap_submatrix_from_result(
    const DeterminantOverlapResult& det_ovlp_result);

Eigen::MatrixXd build_overlap_submatrix_from_result(
    const DeterminantOverlapResult& det_ovlp_result);

Eigen::MatrixXd build_deleted_minor_matrix(
    const Eigen::MatrixXd& overlap_block,
    const std::vector<int>& deleted_rows,
    const std::vector<int>& deleted_cols);

double calc_deleted_minor_sign(
    int n_rows,
    int n_cols,
    const std::vector<int>& deleted_rows,
    const std::vector<int>& deleted_cols);

double calc_deleted_minor_determinant(
    const Eigen::MatrixXd& overlap_block,
    const std::vector<int>& deleted_rows,
    const std::vector<int>& deleted_cols,
    const DeterminantOverlapResolver& overlap_resolver);

double calc_directional_deleted_minor_determinant(
    const Eigen::MatrixXd& overlap_block,
    const Eigen::MatrixXd& delta_overlap_block,
    const std::vector<int>& deleted_rows,
    const std::vector<int>& deleted_cols,
    const DeterminantOverlapResolver& overlap_resolver);

Eigen::MatrixXd build_directional_first_cofactor_matrix(
    const Eigen::MatrixXd& overlap_block,
    const Eigen::MatrixXd& delta_overlap_block,
    const DeterminantOverlapResult& det_ovlp_result,
    const DeterminantOverlapResolver& overlap_resolver);

SameSpinPhiResult compute_same_spin_original_phi(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& h1e_act,
    int n_active_orbitals,
    const std::vector<double>& packed_active_two_electron_integrals,
    const DeterminantOverlapResult& det_ovlp_result,
    Eigen::MatrixXd* inverse_overlap_gradient);

SameSpinPhiResult compute_same_spin_original_phi(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& h1e_act,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const DeterminantOverlapResult& det_ovlp_result,
    Eigen::MatrixXd* inverse_overlap_gradient);

double compute_opposite_spin_original_phi(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const DeterminantOverlapResult& alpha_overlap_result,
    const OppositeSpinPairCache* alpha_pair_cache,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const DeterminantOverlapResult& beta_overlap_result,
    const OppositeSpinPairCache* beta_pair_cache,
    const std::vector<double>& packed_active_two_electron_integrals,
    Eigen::MatrixXd* alpha_inverse_overlap_gradient,
    Eigen::MatrixXd* beta_inverse_overlap_gradient);

double compute_opposite_spin_original_phi(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const DeterminantOverlapResult& alpha_overlap_result,
    const OppositeSpinPairCache* alpha_pair_cache,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const DeterminantOverlapResult& beta_overlap_result,
    const OppositeSpinPairCache* beta_pair_cache,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    Eigen::MatrixXd* alpha_inverse_overlap_gradient,
    Eigen::MatrixXd* beta_inverse_overlap_gradient);

double compute_opposite_spin_original_phi(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const DeterminantOverlapResult& alpha_overlap_result,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const DeterminantOverlapResult& beta_overlap_result,
    const std::vector<double>& packed_active_two_electron_integrals,
    Eigen::MatrixXd* alpha_inverse_overlap_gradient,
    Eigen::MatrixXd* beta_inverse_overlap_gradient);

double compute_opposite_spin_original_phi(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const DeterminantOverlapResult& alpha_overlap_result,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const DeterminantOverlapResult& beta_overlap_result,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    Eigen::MatrixXd* alpha_inverse_overlap_gradient,
    Eigen::MatrixXd* beta_inverse_overlap_gradient);

void accumulate_spin_overlap_gradient(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const DeterminantOverlapResult& det_ovlp_result,
    double determinant_overlap_weight,
    const Eigen::MatrixXd& inverse_overlap_gradient,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient);

}  // namespace xmvb::vb
