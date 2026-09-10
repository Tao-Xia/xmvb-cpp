#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/same_spin_backward.hpp"

namespace xmvb::vb::detail {

struct SameSpinPairScalarMatrices {
  Eigen::MatrixXd overlap_determinant_matrix;
  Eigen::MatrixXd regular_total_hamiltonian_matrix;
  Eigen::MatrixXd singular_total_hamiltonian_matrix;
};

struct SingleChannelSameSpinWeightMatrices {
  Eigen::MatrixXd alpha_weight_matrix;
  Eigen::MatrixXd beta_weight_matrix;
};

double max_abs_dense_matrix(const Eigen::MatrixXd& matrix);

void multiply_left_symmetric(
    const Eigen::MatrixXd& symmetric_left,
    const Eigen::MatrixXd& right,
    Eigen::MatrixXd* output);

void multiply_right_symmetric(
    const Eigen::MatrixXd& left,
    const Eigen::MatrixXd& symmetric_right,
    Eigen::MatrixXd* output);

bool selected_state_has_close_shell_diagonal(
    const SelectedStateDeterminantCoefficients& state_coefficients,
    int n_unique_alpha,
    int n_unique_beta);

bool selected_state_has_local_support(
    const SelectedStateDeterminantCoefficients& state_coefficients);

void accumulate_diagonal_kernel_image_global(
    const std::vector<double>& diagonal_coefficients,
    const Eigen::MatrixXd& partner_kernel_matrix,
    double scale,
    Eigen::MatrixXd* global_weight_matrix);

void accumulate_directional_diagonal_kernel_image_global(
    const std::vector<double>& diagonal_coefficients,
    const std::vector<double>& directional_diagonal_coefficients,
    const Eigen::MatrixXd& partner_kernel_matrix,
    double scale,
    Eigen::MatrixXd* global_weight_matrix);

void validate_state_coefficient_matrix(
    const SelectedStateDeterminantCoefficients& state_coefficients,
    int n_unique_alpha,
    int n_unique_beta);

void scatter_add_dense_submatrix(
    const Eigen::MatrixXd& local_matrix,
    const std::vector<int>& row_indices,
    const std::vector<int>& column_indices,
    double scale,
    Eigen::MatrixXd* global_matrix);

std::vector<int> build_merged_support_indices(
    const std::vector<int>& first,
    const std::vector<int>& second);

void accumulate_selected_state_alpha_image(
    const SelectedStateDeterminantCoefficients& state_coefficients,
    const Eigen::MatrixXd& beta_kernel_subblock,
    double scale,
    Eigen::MatrixXd* global_alpha_weight_matrix,
    Eigen::MatrixXd* beta_push,
    Eigen::MatrixXd* alpha_image);

void accumulate_selected_state_beta_image(
    const SelectedStateDeterminantCoefficients& state_coefficients,
    const Eigen::MatrixXd& alpha_kernel_subblock,
    double scale,
    Eigen::MatrixXd* global_beta_weight_matrix,
    Eigen::MatrixXd* alpha_push,
    Eigen::MatrixXd* beta_image);

SameSpinPairScalarMatrices build_pair_scalar_matrices(
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants);

SingleChannelSameSpinWeightMatrices
build_dense_same_spin_weight_matrices_from_partner_kernels(
    const SelectedStateDeterminantMatrices& selected_states,
    const Eigen::MatrixXd& alpha_partner_kernel_matrix,
    const Eigen::MatrixXd& beta_partner_kernel_matrix,
    const std::vector<double>& per_state_scales,
    bool close_shell_same_spin);

}  // namespace xmvb::vb::detail
