#include "vb/biorthogonal_vbscf/biorthogonal_structure_hamiltonian_builder.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "vb/biorthogonal_vbscf/biorthogonal_structure_local_utils.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_spin_pair_tiles.hpp"
#include "vb/matrices/determinant_types.hpp"
#include "vb/matrices/structure_block_kernels.hpp"
#include "vb/matrices/structure_coefficient_blocks.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

namespace {

double contract_local_opposite_spin_block(
    const Eigen::MatrixXd& left_coefficients,
    const Eigen::MatrixXd& right_coefficients,
    const LocalSpinProjectionBlock& alpha_projection_block,
    const LocalSpinProjectionBlock& beta_projection_block,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_orbitals,
    Eigen::MatrixXd* beta_projected_channel_block,
    Eigen::MatrixXd* beta_push,
    Eigen::MatrixXd* image) {
  const int n_packed_active_pairs = packed_active_pair_count(n_orbitals);
  const LocalOppositeSpinChannelFamily alpha_channels =
      build_local_alpha_channel_family(
          alpha_projection_block,
          n_packed_active_pairs);
  double contraction = 0.0;
  for (std::size_t channel_index = 0;
       channel_index < alpha_channels.packed_pair_indices.size();
       ++channel_index) {
    build_local_beta_projected_channel(
        beta_projection_block,
        alpha_channels.packed_pair_indices[channel_index],
        two_electron_view,
        n_orbitals,
        beta_projected_channel_block);
    contraction +=
        contract_dense_structure_pair_kernel(
            left_coefficients,
            right_coefficients,
            xmvb::index_at(alpha_channels.alpha_channel_matrices, channel_index),
            *beta_projected_channel_block,
            beta_push,
            image);
  }
  return contraction;
}

}  // namespace

BiorthogonalStructureHamiltonianBuildResult
build_biorthogonal_structure_hamiltonian(
    const xmvb::vb::FullDeterminantStructureData& structure_data,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view) {
  if (structure_data.n_structures <= 0) {
    throw std::invalid_argument("structure_data.n_structures must be positive");
  }
  if (static_cast<int>(structure_data.alpha_det.size()) !=
          static_cast<int>(structure_data.beta_det.size()) ||
      static_cast<int>(structure_data.alpha_det.size()) !=
          static_cast<int>(structure_data.determinant_to_structure_terms.size())) {
    throw std::invalid_argument("full determinant structure arrays are inconsistent");
  }

  const SpinDeterminantReuseTable alpha_reuse_table =
      build_spin_determinant_reuse_table(structure_data.alpha_det);
  const SpinDeterminantReuseTable beta_reuse_table =
      build_spin_determinant_reuse_table(structure_data.beta_det);
  const auto coefficient_blocks =
      xmvb::vb::build_structure_coefficient_blocks(
          structure_data.determinant_to_structure_terms,
          structure_data.n_structures,
          alpha_reuse_table,
          beta_reuse_table);

  BiorthogonalStructureHamiltonianBuildResult result;
  result.n_unique_alpha =
      static_cast<int>(alpha_reuse_table.unique_determinants.size());
  result.n_unique_beta =
      static_cast<int>(beta_reuse_table.unique_determinants.size());
  result.selected_structure_overlap =
      Eigen::MatrixXd::Zero(structure_data.n_structures, structure_data.n_structures);
  result.selected_structure_one_electron_hamiltonian =
      Eigen::MatrixXd::Zero(structure_data.n_structures, structure_data.n_structures);
  result.selected_structure_hamiltonian =
      Eigen::MatrixXd::Zero(structure_data.n_structures, structure_data.n_structures);

#pragma omp parallel if(structure_data.n_structures > 2)
  {
    BiorthogonalForwardSpinPairTileProvider alpha_pair_provider(
        alpha_reuse_table.unique_determinants,
        true,
        orbital_integrals,
        right_right_two_electron_view);
    BiorthogonalForwardSpinPairTileProvider beta_pair_provider(
        beta_reuse_table.unique_determinants,
        false,
        orbital_integrals,
        right_right_two_electron_view);
    Eigen::MatrixXd alpha_overlap_subblock;
    Eigen::MatrixXd alpha_one_electron_subblock;
    Eigen::MatrixXd alpha_total_subblock;
    Eigen::MatrixXd beta_overlap_subblock;
    Eigen::MatrixXd beta_one_electron_subblock;
    Eigen::MatrixXd beta_total_subblock;
    Eigen::MatrixXd beta_projected_channel_block;
    Eigen::MatrixXd beta_push;
    Eigen::MatrixXd image;
    LocalSpinProjectionBlock alpha_projection_block;
    LocalSpinProjectionBlock beta_projection_block;

#pragma omp for schedule(dynamic)
    for (int right_structure = 0;
         right_structure < structure_data.n_structures;
         ++right_structure) {
      const auto& right_block =
          xmvb::index_at(coefficient_blocks, right_structure);
      for (int left_structure = 0;
           left_structure < structure_data.n_structures;
           ++left_structure) {
        const auto& left_block =
            xmvb::index_at(coefficient_blocks, left_structure);
        if (left_block.local_coefficients.size() == 0 ||
            right_block.local_coefficients.size() == 0) {
          continue;
        }

        gather_biorthogonal_forward_spin_block(
            alpha_pair_provider,
            left_block.alpha_support,
            right_block.alpha_support,
            &alpha_overlap_subblock,
            &alpha_one_electron_subblock,
            &alpha_total_subblock,
            &alpha_projection_block);
        gather_biorthogonal_forward_spin_block(
            beta_pair_provider,
            left_block.beta_support,
            right_block.beta_support,
            &beta_overlap_subblock,
            &beta_one_electron_subblock,
            &beta_total_subblock,
            &beta_projection_block);

        const double overlap_value =
            contract_dense_structure_pair_kernel(
                left_block.local_coefficients,
                right_block.local_coefficients,
                alpha_overlap_subblock,
                beta_overlap_subblock,
                &beta_push,
                &image);
        const double one_electron_value =
            contract_dense_structure_pair_kernel(
                left_block.local_coefficients,
                right_block.local_coefficients,
                alpha_one_electron_subblock,
                beta_overlap_subblock,
                &beta_push,
                &image) +
            contract_dense_structure_pair_kernel(
                left_block.local_coefficients,
                right_block.local_coefficients,
                alpha_overlap_subblock,
                beta_one_electron_subblock,
                &beta_push,
                &image);
        double total_hamiltonian_value =
            contract_dense_structure_pair_kernel(
                left_block.local_coefficients,
                right_block.local_coefficients,
                alpha_total_subblock,
                beta_overlap_subblock,
                &beta_push,
                &image) +
            contract_dense_structure_pair_kernel(
                left_block.local_coefficients,
                right_block.local_coefficients,
                alpha_overlap_subblock,
                beta_total_subblock,
                &beta_push,
                &image);
        total_hamiltonian_value +=
            contract_local_opposite_spin_block(
                left_block.local_coefficients,
                right_block.local_coefficients,
                alpha_projection_block,
                beta_projection_block,
                right_right_two_electron_view,
                structure_data.n_active_orbitals,
                &beta_projected_channel_block,
                &beta_push,
                &image);

        result.selected_structure_overlap(left_structure, right_structure) =
            overlap_value;
        result.selected_structure_one_electron_hamiltonian(left_structure, right_structure) =
            one_electron_value;
        result.selected_structure_hamiltonian(left_structure, right_structure) =
            total_hamiltonian_value;
      }
    }
  }

  validate_biorthogonal_structure_hamiltonian_build_result(
      result,
      structure_data.n_structures);
  return result;
}

BiorthogonalStructureHamiltonianBuildResult
build_biorthogonal_structure_hamiltonian(
    const xmvb::vb::FullDeterminantStructureData& structure_data,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronResult& right_right_two_electron_result) {
  return build_biorthogonal_structure_hamiltonian(
      structure_data,
      orbital_integrals,
      make_active_space_two_electron_view(right_right_two_electron_result));
}

void validate_biorthogonal_structure_hamiltonian_build_result(
    const BiorthogonalStructureHamiltonianBuildResult& build_result,
    int expected_structure_count) {
  if (expected_structure_count <= 0) {
    throw std::invalid_argument("expected_structure_count must be positive");
  }
  if (build_result.n_unique_alpha <= 0 || build_result.n_unique_beta <= 0) {
    throw std::invalid_argument("unique spin-string counts must be positive");
  }
  if (build_result.selected_structure_overlap.rows() != expected_structure_count ||
      build_result.selected_structure_overlap.cols() != expected_structure_count) {
    throw std::invalid_argument(
        "selected_structure_overlap dimensions are inconsistent");
  }
  if (build_result.selected_structure_one_electron_hamiltonian.rows() !=
          expected_structure_count ||
      build_result.selected_structure_one_electron_hamiltonian.cols() !=
          expected_structure_count) {
    throw std::invalid_argument(
        "selected_structure_one_electron_hamiltonian dimensions are inconsistent");
  }
  if (build_result.selected_structure_hamiltonian.rows() != expected_structure_count ||
      build_result.selected_structure_hamiltonian.cols() != expected_structure_count) {
    throw std::invalid_argument(
        "selected_structure_hamiltonian dimensions are inconsistent");
  }
  throw_if_nonfinite(
      build_result.selected_structure_overlap,
      "selected_structure_overlap");
  throw_if_nonfinite(
      build_result.selected_structure_one_electron_hamiltonian,
      "selected_structure_one_electron_hamiltonian");
  throw_if_nonfinite(
      build_result.selected_structure_hamiltonian,
      "selected_structure_hamiltonian");
}

}  // namespace xmvb::vb::biorthogonal_vbscf
