#include "vbscf/derivatives/hessian/responses/opposite_spin_pair_response_internal.hpp"

#include <cmath>
#include <stdexcept>

#include "vbscf/determinants/pair_storage.hpp"
#include "vbscf/determinants/spin_pair_contractions.hpp"
#include "vbscf/integrals/active/active_space_two_electron_kernel.hpp"
#include "vbscf/integrals/active/two_electron_indexer.hpp"

namespace xmvb::vb::detail {
namespace {

constexpr double kContributionTolerance = 1.0e-15;

OppositeSpinPackedPairProjection build_sparse_packed_pair_projection(
    const std::vector<int>& left_occupations,
    const std::vector<int>& right_occupations,
    const Eigen::MatrixXd& coefficient_matrix,
    int n_orbitals) {
  OppositeSpinPackedPairProjection projection;
  if (left_occupations.empty()) {
    return projection;
  }

  const int n_packed_active_pairs = packed_active_pair_count(n_orbitals);
  std::vector<double> dense_pair_values(n_packed_active_pairs, 0.0);
  std::vector<unsigned char> touched_mask(n_packed_active_pairs, 0);
  std::vector<int> touched_indices;
  touched_indices.reserve(left_occupations.size() * right_occupations.size());

  for (int left_column = 0;
       left_column < static_cast<int>(left_occupations.size());
       ++left_column) {
    const int orbital_index_left = left_occupations[left_column];
    for (int right_row = 0;
         right_row < static_cast<int>(right_occupations.size());
         ++right_row) {
      const int packed_pair_index = TwoElectronIndexer::packed_pair_index(
          right_occupations[right_row],
          orbital_index_left);
      if (touched_mask[packed_pair_index] == 0) {
        touched_mask[packed_pair_index] = 1;
        touched_indices.push_back(packed_pair_index);
      }
      dense_pair_values[packed_pair_index] +=
          coefficient_matrix(right_row, left_column);
    }
  }

  projection.packed_pair_indices.reserve(touched_indices.size());
  projection.packed_pair_values.reserve(touched_indices.size());
  for (const int packed_pair_index : touched_indices) {
    const double value = dense_pair_values[packed_pair_index];
    if (std::abs(value) <= kContributionTolerance) {
      continue;
    }
    projection.packed_pair_indices.push_back(packed_pair_index);
    projection.packed_pair_values.push_back(value);
  }
  return projection;
}

std::vector<double> apply_directional_two_electron_kernel(
    int n_active_orbitals,
    const OppositeSpinPackedPairProjection& projection,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  const int n_packed_active_pairs = packed_active_pair_count(n_active_orbitals);
  std::vector<double> image(n_packed_active_pairs, 0.0);
  if (delta_packed_active_two_electron_integrals.empty()) {
    return image;
  }
  for (std::size_t entry_index = 0;
       entry_index < projection.packed_pair_indices.size();
       ++entry_index) {
    const int column_pair = projection.packed_pair_indices[entry_index];
    const double coefficient = projection.packed_pair_values[entry_index];
    for (int row_pair = 0; row_pair < n_packed_active_pairs; ++row_pair) {
      image[row_pair] +=
          delta_packed_active_two_electron_integrals[
              TwoElectronIndexer::packed_pair_of_pairs_index(
                  row_pair,
                  column_pair)] *
          coefficient;
    }
  }
  return image;
}

}  // namespace

std::vector<DirectionalOppositeSpinPairData>
build_directional_opposite_spin_pair_data(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const std::vector<double>& delta_ao_overlap_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    const std::vector<SameSpinPolynomialDirectionalPairData>&
        precomputed_directional_pair_data) {
  const std::size_t expected_size =
      static_cast<std::size_t>(n_unique_determinants) *
      static_cast<std::size_t>(n_unique_determinants);
  if (unique_determinants.size() !=
          static_cast<std::size_t>(n_unique_determinants) ||
      ordered_pair_cache.size() != expected_size ||
      precomputed_directional_pair_data.size() != expected_size) {
    throw std::invalid_argument(
        "opposite-spin pair direction dimensions are inconsistent");
  }

  std::vector<DirectionalOppositeSpinPairData> result(expected_size);
  const ActiveSpaceTwoElectronView two_electron_view =
      make_active_space_two_electron_view(active_space_two_electron_result);
  for (int left = 0; left < n_unique_determinants; ++left) {
    for (int right = 0; right < n_unique_determinants; ++right) {
      const std::size_t pair_index = ordered_spin_pair_storage_index(
          left,
          right,
          n_unique_determinants);
      if (unique_determinants[left].empty()) {
        continue;
      }

      auto& entry = result[pair_index];
      entry.delta_overlap_submatrix = build_overlap_submatrix(
          unique_determinants[left],
          unique_determinants[right],
          delta_ao_overlap_matrix,
          n_active_orbitals);
      entry.delta_first_order_cofactor_projection =
          build_sparse_packed_pair_projection(
              unique_determinants[left],
              unique_determinants[right],
              precomputed_directional_pair_data[pair_index].delta_cofactor_1st,
              n_active_orbitals);

      auto& projected_direction =
          entry.delta_first_order_cofactor_projection;
      projected_direction.projected_pair_values =
          apply_active_space_two_electron_kernel_to_sparse_projection(
              two_electron_view,
              n_active_orbitals,
              projected_direction.packed_pair_indices,
              projected_direction.packed_pair_values);
      const auto& accepted_projection =
          ordered_pair_cache[pair_index]
              .opposite_spin_pair_cache.first_order_cofactor_projection;
      const std::vector<double> kernel_direction =
          apply_directional_two_electron_kernel(
              n_active_orbitals,
              accepted_projection,
              delta_packed_active_two_electron_integrals);
      for (std::size_t packed_pair = 0;
           packed_pair < kernel_direction.size();
           ++packed_pair) {
        projected_direction.projected_pair_values[packed_pair] +=
            kernel_direction[packed_pair];
      }
    }
  }
  return result;
}

}  // namespace xmvb::vb::detail
