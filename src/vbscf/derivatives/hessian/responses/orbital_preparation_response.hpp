#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/orbitals/charts/sparse_parameter_layout.hpp"
#include "vbscf/orbitals/orbital_preparation_input.hpp"
#include "vbscf/orbitals/orbital_preparation_result.hpp"

namespace xmvb::vb {

/** Accepted-point data reused by every orbital Hessian action. */
struct AcceptedOrbitalPreparationCache {
  Eigen::MatrixXd normalized_orbitals;
  std::vector<double> inverse_norms;
  Eigen::MatrixXd basis_overlap_times_normalized;
  Eigen::MatrixXd inactive_overlap_inverse;
  Eigen::MatrixXd inactive_auxiliary;
  Eigen::MatrixXd inactive_density;

  std::vector<std::vector<int>> orbital_basis_function_indices;
  std::vector<int> orbital_coefficient_counts;
  std::vector<Eigen::MatrixXd> orbital_overlap_submatrices;

  Eigen::MatrixXd original_orbital_gradient;
  Eigen::MatrixXd inactive_density_gradient_symmetric;

  bool has_inactive_orbitals = false;
  bool has_pullback_cache = false;
};

/** Dense representative of one sparse-coordinate orbital tangent. */
struct DenseOrbitalTangentContext {
  Eigen::MatrixXd normalized_orbitals;
  Eigen::MatrixXd delta_normalized_orbitals;
  std::vector<double> inverse_norms;
  std::vector<double> normalization_direction_projections;
};

/** Directional orbital quantities consumed by the integral response. */
struct OrbitalPreparationDirectionalResult {
  Eigen::MatrixXd delta_inactive_density;
  Eigen::MatrixXd delta_active_auxiliary_orbitals;
  Eigen::MatrixXd basis_overlap_times_delta_active_orbitals;
};

AcceptedOrbitalPreparationCache build_accepted_orbital_preparation_cache(
    const OrbitalPreparationInput& input,
    const Eigen::Ref<const Eigen::MatrixXd>& total_active_auxiliary_gradient,
    const std::vector<double>& total_inactive_density_gradient);

std::vector<double> backpropagate_active_space_orbital_gradient(
    const Eigen::Ref<const Eigen::MatrixXd>& active_auxiliary_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_gradient,
    const OrbitalPreparationInput& input,
    const OrbitalPreparationResult& orbital_preparation_result,
    const AcceptedOrbitalPreparationCache& cache);

DenseOrbitalTangentContext build_dense_orbital_tangent_context(
    const OrbitalPreparationInput& input,
    const SparseParameterLayout& parameter_view,
    const Eigen::VectorXd& packed_direction,
    const AcceptedOrbitalPreparationCache& cache);

OrbitalPreparationDirectionalResult build_orbital_preparation_directional_result(
    const OrbitalPreparationInput& input,
    const DenseOrbitalTangentContext& orbital_tangent_context,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals,
    const AcceptedOrbitalPreparationCache& cache);

std::vector<double> apply_fixed_upstream_orbital_pullback_direction(
    const OrbitalPreparationInput& input,
    const DenseOrbitalTangentContext& orbital_tangent_context,
    const Eigen::Ref<const Eigen::MatrixXd>& total_active_auxiliary_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>&
        basis_overlap_times_delta_active_orbitals,
    const std::vector<double>& total_inactive_density_gradient,
    const Eigen::Ref<const Eigen::VectorXd>& input_retract_tangent,
    const AcceptedOrbitalPreparationCache& cache);

}  // namespace xmvb::vb
