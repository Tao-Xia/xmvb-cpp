#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/optimization/driver/types.hpp"
#include "vbscf/orbitals/preparation/input.hpp"

namespace xmvb::vb {

class LocalizedRepresentativeSelector;
class SparseParameterLayout;
struct OrbitalPreparationResult;
struct SupportPreservingGaugeTransform;

bool orbital_uses_full_ao_support(
    const OrbitalPreparationInput& orbital_preparation_input,
    int orbital_index);

void require_full_ao_orbital_block(
    const OrbitalPreparationInput& orbital_preparation_input,
    int first_orbital,
    int orbital_count,
    const char* label);

Eigen::MatrixXd build_dense_orbital_block_from_sparse_input(
    const OrbitalPreparationInput& orbital_preparation_input,
    int first_orbital,
    int orbital_count);

Eigen::MatrixXd build_dense_orbital_block_from_full_vector(
    const OrbitalPreparationInput& orbital_preparation_input,
    const std::vector<double>& full_vector,
    int first_orbital,
    int orbital_count);

void scatter_dense_orbital_block_to_full_vector(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_block,
    const OrbitalPreparationInput& orbital_preparation_input,
    int first_orbital,
    std::vector<double>* full_vector);

void transform_sparse_oeo_active_representative_gradient(
    const LocalizedRepresentativeSelector& source_selector,
    const LocalizedRepresentativeSelector& target_selector,
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* sparse_orbital_gradient);

void transform_sparse_inactive_orbital_step(
    const SupportPreservingGaugeTransform& transform,
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* sparse_orbital_step);

void transform_sparse_oeo_active_representative_step(
    const LocalizedRepresentativeSelector& source_selector,
    const LocalizedRepresentativeSelector& target_selector,
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* sparse_orbital_step);

std::vector<double> build_full_sparse_vector_from_packed(
    const SparseParameterLayout& parameter_view,
    const OrbitalPreparationInput& orbital_preparation_input,
    const Eigen::VectorXd& packed_vector);

void overwrite_packed_vector_from_full_sparse(
    const SparseParameterLayout& parameter_view,
    const std::vector<double>& full_sparse_vector,
    Eigen::VectorXd* packed_vector);

void transport_packed_secant_history_with_support_aware_inactive_gauge(
    const SupportPreservingGaugeTransform& transform,
    const OrbitalPreparationInput& orbital_preparation_input,
    const SparseParameterLayout& parameter_view,
    std::vector<PackedSecantPair>* packed_secant_history);

void transport_packed_secant_history_with_oeo_active_representative_reset(
    const LocalizedRepresentativeSelector& source_selector,
    const LocalizedRepresentativeSelector& target_selector,
    const OrbitalPreparationInput& orbital_preparation_input,
    const SparseParameterLayout& parameter_view,
    std::vector<PackedSecantPair>* packed_secant_history);

void refresh_cached_localized_representative_selector(
    const OrbitalPreparationInput& orbital_preparation_input,
    OrbitalPreparationResult* orbital_result);

void overwrite_sparse_orbitals_from_dense_physical_frame(
    const Eigen::MatrixXd& dense_orbitals,
    OrbitalPreparationInput* orbital_preparation_input);

Eigen::MatrixXd build_self_adjoint_matrix_power(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    double exponent,
    const char* label);

Eigen::MatrixXd build_inactive_metric_inverse(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_physical_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap_matrix);

Eigen::MatrixXd build_metric_preserving_inactive_repaired_active_physical_orbitals(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_physical_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& current_active_auxiliary,
    const Eigen::Ref<const Eigen::MatrixXd>& current_active_physical_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& reference_active_physical_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap_matrix);

Eigen::MatrixXd build_metric_preserving_oeo_repaired_normalized_orbital_matrix(
    const OrbitalPreparationInput& orbital_preparation_input,
    const OrbitalPreparationResult& orbital_result,
    const Eigen::Ref<const Eigen::MatrixXd>& reference_normalized_orbital_matrix);

}  // namespace xmvb::vb
