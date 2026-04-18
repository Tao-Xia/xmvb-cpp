#include "vb/scf/opposite_spin_matrix_channels.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Sparse>

#include "vb/matrices/same_spin_pair_cache.hpp"
#include "vb/matrices/spin_pair_utils.hpp"

namespace xmvb::vb {

namespace {

using SparseTriplet = Eigen::Triplet<double, int>;

void validate_ordered_pair_cache_shape(
    const std::vector<SpinDeterminantPairEvaluation>& ordered_spin_pair_cache,
    int n_unique_determinants,
    const char* spin_label) {
  if (n_unique_determinants < 0) {
    throw std::invalid_argument("n_unique_determinants must be non-negative");
  }
  const std::size_t expected_size =
      xmvb::to_size(n_unique_determinants) *
      xmvb::to_size(n_unique_determinants);
  if (ordered_spin_pair_cache.size() != expected_size) {
    throw std::invalid_argument(
        std::string("ordered ") + spin_label +
        " pair cache size does not match n_unique_determinants^2");
  }
}

void accumulate_sparse_projection_triplets(
    const OppositeSpinPackedPairProjection& projection,
    int left_unique_index,
    int right_unique_index,
    std::vector<std::vector<SparseTriplet>>* triplets_by_packed_pair) {
  for (std::size_t entry_index = 0;
       entry_index < projection.packed_pair_indices.size();
       ++entry_index) {
    const int packed_pair_index = projection.packed_pair_indices[entry_index];
    const double packed_pair_value = projection.packed_pair_values[entry_index];
    (*triplets_by_packed_pair)[xmvb::to_size(packed_pair_index)]
        .emplace_back(left_unique_index, right_unique_index, packed_pair_value);
  }
}

void finalize_sparse_matrix_family(
    int n_unique_determinants,
    const std::vector<std::vector<SparseTriplet>>& triplets_by_packed_pair,
    std::vector<Eigen::SparseMatrix<double, Eigen::ColMajor, int>>* sparse_matrices_by_packed_pair) {
  sparse_matrices_by_packed_pair->resize(triplets_by_packed_pair.size());
  for (std::size_t packed_pair_index = 0;
       packed_pair_index < triplets_by_packed_pair.size();
       ++packed_pair_index) {
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> sparse_matrix(
        n_unique_determinants,
        n_unique_determinants);
    const auto& triplets =
        triplets_by_packed_pair[packed_pair_index];
    if (!triplets.empty()) {
      sparse_matrix.setFromTriplets(triplets.begin(), triplets.end());
    }
    (*sparse_matrices_by_packed_pair)[packed_pair_index] =
        std::move(sparse_matrix);
  }
}

OppositeSpinPerSpinMatrixChannels build_opposite_spin_per_spin_matrix_channels_impl(
    const std::vector<SpinDeterminantPairEvaluation>& ordered_spin_pair_cache,
    int n_unique_determinants,
    const char* spin_label) {
  validate_ordered_pair_cache_shape(
      ordered_spin_pair_cache,
      n_unique_determinants,
      spin_label);

  OppositeSpinPerSpinMatrixChannels channels;
  channels.n_unique_determinants = n_unique_determinants;
  channels.n_ordered_unique_pairs =
      n_unique_determinants * n_unique_determinants;
  channels.n_packed_active_pairs =
      infer_n_packed_active_pairs(ordered_spin_pair_cache, spin_label);
  channels.overlap_determinant_matrix =
      Eigen::MatrixXd::Zero(
          n_unique_determinants,
          n_unique_determinants);

  channels.has_first_order_projection_by_ordered_pair.assign(
      xmvb::to_size(channels.n_ordered_unique_pairs),
      0u);
  channels.has_inverse_projection_by_ordered_pair.assign(
      xmvb::to_size(channels.n_ordered_unique_pairs),
      0u);

  std::vector<std::vector<SparseTriplet>> first_order_triplets_by_packed_pair(
      xmvb::to_size(channels.n_packed_active_pairs));
  std::vector<std::vector<SparseTriplet>> inverse_overlap_triplets_by_packed_pair(
      xmvb::to_size(channels.n_packed_active_pairs));

  channels.inverse_projected_image_by_packed_pair.resize(
      xmvb::to_size(channels.n_packed_active_pairs));
  for (auto& projected_image_matrix : channels.inverse_projected_image_by_packed_pair) {
    projected_image_matrix =
        Eigen::MatrixXd::Zero(n_unique_determinants, n_unique_determinants);
  }

  // Reindex opposite-spin payloads from "ordered pair -> sparse packed vector"
  // into "packed pair -> matrix over ordered unique pairs". This is the
  // matrix-form object needed by opposite-spin backward contractions.
  for (int left_unique_index = 0;
       left_unique_index < n_unique_determinants;
       ++left_unique_index) {
    for (int right_unique_index = 0;
         right_unique_index < n_unique_determinants;
         ++right_unique_index) {
      const std::size_t ordered_pair_index = ordered_spin_pair_storage_index(
          left_unique_index,
          right_unique_index,
          n_unique_determinants);
      const auto& pair_evaluation =
          ordered_spin_pair_cache[ordered_pair_index];
      const auto& pair_cache = pair_evaluation.opposite_spin_pair_cache;

      channels.overlap_determinant_matrix(left_unique_index, right_unique_index) =
          pair_evaluation.overlap_result.overlap_determinant;

      if (pair_cache.n_packed_active_pairs > 0 &&
          pair_cache.n_packed_active_pairs != channels.n_packed_active_pairs) {
        throw std::invalid_argument(
            std::string("ordered ") + spin_label +
            " pair cache entry has inconsistent n_packed_active_pairs");
      }

      const bool has_first_order_projection =
          has_opposite_spin_first_order_projection(pair_cache);
      if (has_first_order_projection) {
        channels.has_first_order_projection_by_ordered_pair[ordered_pair_index] = 1u;
        validate_sparse_projection_coefficients(
            pair_cache.first_order_cofactor_projection,
            channels.n_packed_active_pairs,
            "first_order_cofactor_projection");
        accumulate_sparse_projection_triplets(
            pair_cache.first_order_cofactor_projection,
            left_unique_index,
            right_unique_index,
            &first_order_triplets_by_packed_pair);
      } else if (projection_has_any_payload(pair_cache.first_order_cofactor_projection)) {
        throw std::invalid_argument(
            std::string("ordered ") + spin_label +
            " pair cache has partial first_order_cofactor_projection payload");
      }

      const bool has_inverse_projection =
          has_opposite_spin_inverse_projection(pair_cache);
      if (has_inverse_projection) {
        channels.has_inverse_projection_by_ordered_pair[ordered_pair_index] = 1u;
        validate_sparse_projection_coefficients(
            pair_cache.inverse_overlap_projection,
            channels.n_packed_active_pairs,
            "inverse_overlap_projection");
        accumulate_sparse_projection_triplets(
            pair_cache.inverse_overlap_projection,
            left_unique_index,
            right_unique_index,
            &inverse_overlap_triplets_by_packed_pair);

        const auto& projected_pair_values =
            pair_cache.inverse_overlap_projection.projected_pair_values;
        if (static_cast<int>(projected_pair_values.size()) !=
            channels.n_packed_active_pairs) {
          throw std::invalid_argument(
              "inverse_overlap_projection.projected_pair_values has wrong size");
        }
        // `inverse_projected_image_by_packed_pair[k]` stores one dense matrix
        // over ordered unique pairs:
        // M_k(left,right) = (G x_{left,right})(k).
        for (int packed_pair_index = 0;
             packed_pair_index < channels.n_packed_active_pairs;
             ++packed_pair_index) {
          channels.inverse_projected_image_by_packed_pair[xmvb::to_size(
              packed_pair_index)](left_unique_index, right_unique_index) =
              projected_pair_values[xmvb::to_size(packed_pair_index)];
        }
      } else if (projection_has_any_payload(pair_cache.inverse_overlap_projection)) {
        throw std::invalid_argument(
            std::string("ordered ") + spin_label +
            " pair cache has partial inverse_overlap_projection payload");
      }
    }
  }

  finalize_sparse_matrix_family(
      n_unique_determinants,
      first_order_triplets_by_packed_pair,
      &channels.first_order_sparse_by_packed_pair);
  finalize_sparse_matrix_family(
      n_unique_determinants,
      inverse_overlap_triplets_by_packed_pair,
      &channels.inverse_overlap_sparse_by_packed_pair);

  return channels;
}

}  // namespace

OppositeSpinPerSpinMatrixChannels build_opposite_spin_per_spin_matrix_channels(
    const std::vector<SpinDeterminantPairEvaluation>& ordered_spin_pair_cache,
    int n_unique_determinants) {
  return build_opposite_spin_per_spin_matrix_channels_impl(
      ordered_spin_pair_cache,
      n_unique_determinants,
      "spin");
}

OppositeSpinMatrixChannels build_opposite_spin_matrix_channels(
    const SameSpinPairCacheContext& same_spin_pair_cache) {
  OppositeSpinMatrixChannels channels;
  channels.alpha = build_opposite_spin_per_spin_matrix_channels_impl(
      same_spin_pair_cache.alpha_pair_cache_ref(),
      static_cast<int>(same_spin_pair_cache.alpha_reuse_table.unique_determinants.size()),
      "alpha");
  channels.beta = build_opposite_spin_per_spin_matrix_channels_impl(
      same_spin_pair_cache.beta_pair_cache_ref(),
      static_cast<int>(same_spin_pair_cache.beta_reuse_table.unique_determinants.size()),
      "beta");
  return channels;
}

}  // namespace xmvb::vb
