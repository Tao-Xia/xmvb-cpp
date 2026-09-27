#include "vbscf/derivatives/hessian/responses/opposite_spin/pair_response_internal.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "core/openmp.hpp"
#include "vbscf/determinants/pairs/storage.hpp"
#include "vbscf/determinants/pairs/contractions.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/tile_policy_internal.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"

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

void scatter_sparse_projection(
    const OppositeSpinPackedPairProjection& projection,
    int column,
    Eigen::MatrixXd* dense) {
  for (std::size_t entry = 0;
       entry < projection.packed_pair_indices.size();
       ++entry) {
    (*dense)(projection.packed_pair_indices[entry], column) +=
        projection.packed_pair_values[entry];
  }
}

void fill_exact_kernel_rows(
    const std::vector<double>& packed_kernel,
    int row_begin,
    int row_end,
    int n_packed_pairs,
    Eigen::MatrixXd* rows) {
  rows->resize(row_end - row_begin, n_packed_pairs);
  for (int row = row_begin; row < row_end; ++row) {
    for (int column = 0; column < n_packed_pairs; ++column) {
      (*rows)(row - row_begin, column) = packed_kernel[
          TwoElectronIndexer::packed_pair_of_pairs_index(row, column)];
    }
  }
}

}  // namespace

const DirectionalOppositeSpinPairData& DirectionalOppositeSpinPairTile::pair(
    int left_local,
    int right_local) const {
  if (left_local < 0 || left_local >= left_size || right_local < 0 ||
      right_local >= right_size) {
    throw std::out_of_range("directional opposite-spin tile index out of range");
  }
  return pairs[static_cast<std::size_t>(left_local) * right_size +
      right_local];
}

DirectionalOppositeSpinPairTileView DirectionalOppositeSpinPairTile::view(
    bool transposed) const {
  return DirectionalOppositeSpinPairTileView(*this, transposed);
}

int DirectionalOppositeSpinPairTileView::left_begin() const noexcept {
  return transposed_ ? storage_->right_begin : storage_->left_begin;
}

int DirectionalOppositeSpinPairTileView::right_begin() const noexcept {
  return transposed_ ? storage_->left_begin : storage_->right_begin;
}

int DirectionalOppositeSpinPairTileView::left_size() const noexcept {
  return transposed_ ? storage_->right_size : storage_->left_size;
}

int DirectionalOppositeSpinPairTileView::right_size() const noexcept {
  return transposed_ ? storage_->left_size : storage_->right_size;
}

DirectionalOppositeSpinPairTileView::ConstChannelMap
DirectionalOppositeSpinPairTileView::channel_view(
    const Eigen::MatrixXd& channels,
    int packed_pair) const {
  if (packed_pair < 0 ||
      packed_pair >= channels.cols()) {
    throw std::out_of_range("opposite-spin tile channel out of range");
  }
  const double* data = channels.col(packed_pair).data();
  if (!transposed_) {
    return ConstChannelMap(
        data,
        storage_->left_size,
        storage_->right_size,
        Stride(storage_->left_size, 1));
  }
  return ConstChannelMap(
      data,
      storage_->right_size,
      storage_->left_size,
      Stride(1, storage_->left_size));
}

DirectionalOppositeSpinPairTileView::ConstChannelMap
DirectionalOppositeSpinPairTileView::raw_channel(
    int packed_pair) const {
  return channel_view(storage_->raw_channel_values, packed_pair);
}

DirectionalOppositeSpinPairTileView::ConstChannelMap
DirectionalOppositeSpinPairTileView::projected_channel(
    int packed_pair) const {
  return channel_view(storage_->projected_channel_values, packed_pair);
}

namespace {

DirectionalOppositeSpinPairTile build_directional_opposite_spin_pair_tile_impl(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>* ordered_pair_cache,
    const AcceptedSpinPairTile* accepted_pair_tile,
    int n_unique_determinants,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const ActiveSpaceIntegralDirectionView& direction,
    const SameSpinDirectionalPairTileView& same_spin_tile,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors,
    const Eigen::MatrixXd* directional_ri_active_pair_factors) {
  const std::size_t expected_size =
      static_cast<std::size_t>(n_unique_determinants) *
      n_unique_determinants;
  const bool full_cache_valid = ordered_pair_cache != nullptr &&
      ordered_pair_cache->size() == expected_size;
  const bool tile_cache_valid = accepted_pair_tile != nullptr &&
      accepted_pair_tile->left_begin == same_spin_tile.left_begin() &&
      accepted_pair_tile->right_begin == same_spin_tile.right_begin() &&
      accepted_pair_tile->left_size == same_spin_tile.left_size() &&
      accepted_pair_tile->right_size == same_spin_tile.right_size();
  if (unique_determinants.size() !=
          static_cast<std::size_t>(n_unique_determinants) ||
      (!full_cache_valid && !tile_cache_valid) ||
      same_spin_tile.left_begin() < 0 || same_spin_tile.right_begin() < 0 ||
      same_spin_tile.left_size() <= 0 || same_spin_tile.right_size() <= 0 ||
      same_spin_tile.left_begin() + same_spin_tile.left_size() >
          n_unique_determinants ||
      same_spin_tile.right_begin() + same_spin_tile.right_size() >
          n_unique_determinants) {
    throw std::invalid_argument(
        "opposite-spin directional tile dimensions are inconsistent");
  }
  if ((accepted_ri_active_pair_factors == nullptr) !=
      (directional_ri_active_pair_factors == nullptr)) {
    throw std::invalid_argument(
        "opposite-spin RI tile requires accepted and directional factors");
  }

  const int n_packed_pairs = packed_active_pair_count(n_active_orbitals);
  if (accepted_ri_active_pair_factors != nullptr &&
      (accepted_ri_active_pair_factors->cols() != n_packed_pairs ||
       accepted_ri_active_pair_factors->rows() !=
           directional_ri_active_pair_factors->rows() ||
       accepted_ri_active_pair_factors->cols() !=
           directional_ri_active_pair_factors->cols())) {
    throw std::invalid_argument(
        "opposite-spin RI tile factor dimensions are inconsistent");
  }

  DirectionalOppositeSpinPairTile result;
  result.left_begin = same_spin_tile.left_begin();
  result.right_begin = same_spin_tile.right_begin();
  result.left_size = same_spin_tile.left_size();
  result.right_size = same_spin_tile.right_size();
  const int work_items = result.left_size * result.right_size;
  result.pairs.resize(static_cast<std::size_t>(work_items));
  result.raw_channel_values =
      Eigen::MatrixXd::Zero(work_items, n_packed_pairs);
  result.projected_channel_values =
      Eigen::MatrixXd::Zero(work_items, n_packed_pairs);

  const int n_threads = xmvb::effective_openmp_thread_count();
#pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
  for (int work = 0; work < work_items; ++work) {
    const int left_local = work % result.left_size;
    const int right_local = work / result.left_size;
    const int left = result.left_begin + left_local;
    const int right = result.right_begin + right_local;
    auto& entry = result.pairs[
        static_cast<std::size_t>(left_local) * result.right_size +
        right_local];
    if (unique_determinants[left].empty()) {
      continue;
    }
    entry.delta_overlap_submatrix = build_overlap_submatrix(
        unique_determinants[left],
        unique_determinants[right],
        direction.overlap,
        n_active_orbitals);
    same_spin_tile.with_pair(
        left_local,
        right_local,
        [&](const SameSpinPolynomialDirectionalPairData& same_pair) {
          entry.delta_first_order_cofactor_projection =
              build_sparse_packed_pair_projection(
                  unique_determinants[left],
                  unique_determinants[right],
                  same_pair.delta_cofactor_1st,
                  n_active_orbitals);
        });
    const auto& raw = entry.delta_first_order_cofactor_projection;
    for (std::size_t projection_entry = 0;
         projection_entry < raw.packed_pair_indices.size();
         ++projection_entry) {
      result.raw_channel_values(
          work, raw.packed_pair_indices[projection_entry]) +=
          raw.packed_pair_values[projection_entry];
    }
  }

  if (accepted_ri_active_pair_factors != nullptr) {
    const Eigen::Index n_auxiliary =
        accepted_ri_active_pair_factors->rows();
    const Eigen::Index workspace_values_per_column =
        2 * n_auxiliary + 3 * n_packed_pairs;
    const Eigen::Index workspace_value_cap =
        static_cast<Eigen::Index>(kPairTileWorkspaceBytes / sizeof(double));
    const Eigen::Index resident_tile_values =
        2 * static_cast<Eigen::Index>(n_packed_pairs) * work_items;
    const Eigen::Index available_workspace_values = std::max<Eigen::Index>(
        1, workspace_value_cap -
               std::min(workspace_value_cap - 1, resident_tile_values));
    const int column_block = std::max<int>(
        1,
        std::min<Eigen::Index>(
            work_items,
            available_workspace_values / workspace_values_per_column));
    for (int begin = 0; begin < work_items; begin += column_block) {
      const int end = std::min(work_items, begin + column_block);
      const int width = end - begin;
      Eigen::MatrixXd accepted_projection =
          Eigen::MatrixXd::Zero(n_packed_pairs, width);
      Eigen::MatrixXd directional_projection =
          result.raw_channel_values.middleRows(begin, width).transpose();
      for (int work = begin; work < end; ++work) {
        const int left_local = work % result.left_size;
        const int right_local = work / result.left_size;
        const int left = result.left_begin + left_local;
        const int right = result.right_begin + right_local;
        const int canonical_left = std::min(left, right);
        const int canonical_right = std::max(left, right);
        const auto& accepted = (accepted_pair_tile != nullptr
            ? accepted_pair_tile->pair(
                  canonical_left - accepted_pair_tile->left_begin,
                  canonical_right - accepted_pair_tile->right_begin)
            : (*ordered_pair_cache)[ordered_spin_pair_storage_index(
                  canonical_left,
                  canonical_right,
                  n_unique_determinants)])
                                   .opposite_spin_pair_cache
                                   .first_order_cofactor_projection;
        const int column = work - begin;
        scatter_sparse_projection(accepted, column, &accepted_projection);
      }

      Eigen::MatrixXd accepted_auxiliary =
          *accepted_ri_active_pair_factors * accepted_projection;
      Eigen::MatrixXd directional_auxiliary =
          *directional_ri_active_pair_factors * accepted_projection;
      directional_auxiliary.noalias() +=
          *accepted_ri_active_pair_factors * directional_projection;
      Eigen::MatrixXd projected =
          accepted_ri_active_pair_factors->transpose() *
          directional_auxiliary;
      projected.noalias() +=
          directional_ri_active_pair_factors->transpose() *
          accepted_auxiliary;
      result.projected_channel_values.middleRows(begin, width) =
          projected.transpose();
    }
    return result;
  }

  const auto& accepted_kernel =
      active_space_two_electron_result.packed_active_two_electron_integrals;
  const auto& directional_kernel = direction.packed_two_electron;
  const std::size_t expected_kernel_size =
      packed_active_two_electron_integral_count(n_active_orbitals);
  if (accepted_kernel.size() != expected_kernel_size ||
      (!directional_kernel.empty() &&
       directional_kernel.size() != expected_kernel_size)) {
    throw std::invalid_argument(
        "exact opposite-spin tile requires packed accepted and directional kernels");
  }

  Eigen::MatrixXd accepted_projection_block =
      Eigen::MatrixXd::Zero(n_packed_pairs, work_items);
  for (int work = 0; work < work_items; ++work) {
    const int left_local = work % result.left_size;
    const int right_local = work / result.left_size;
    const int left = result.left_begin + left_local;
    const int right = result.right_begin + right_local;
    const int canonical_left = std::min(left, right);
    const int canonical_right = std::max(left, right);
    const auto& accepted_pair_projection = (accepted_pair_tile != nullptr
        ? accepted_pair_tile->pair(
              canonical_left - accepted_pair_tile->left_begin,
              canonical_right - accepted_pair_tile->right_begin)
        : (*ordered_pair_cache)[ordered_spin_pair_storage_index(
              canonical_left,
              canonical_right,
              n_unique_determinants)])
                                          .opposite_spin_pair_cache
                                          .first_order_cofactor_projection;
    scatter_sparse_projection(
        accepted_pair_projection, work, &accepted_projection_block);
  }

  const Eigen::Index workspace_value_cap =
      static_cast<Eigen::Index>(kPairTileWorkspaceBytes / sizeof(double));
  const Eigen::Index resident_tile_values =
      3 * static_cast<Eigen::Index>(n_packed_pairs) * work_items;
  const Eigen::Index available_workspace_values = std::max<Eigen::Index>(
      1, workspace_value_cap -
             std::min(workspace_value_cap - 1, resident_tile_values));
  const int row_block = std::max<int>(
      1,
      std::min<Eigen::Index>(
          n_packed_pairs,
          available_workspace_values /
              std::max<Eigen::Index>(
                  1, 2 * n_packed_pairs + work_items)));
  Eigen::MatrixXd accepted_rows;
  Eigen::MatrixXd directional_rows;
  for (int row_begin = 0; row_begin < n_packed_pairs;
       row_begin += row_block) {
    const int row_end = std::min(n_packed_pairs, row_begin + row_block);
    fill_exact_kernel_rows(
        accepted_kernel,
        row_begin,
        row_end,
        n_packed_pairs,
        &accepted_rows);
    Eigen::MatrixXd projected = accepted_rows *
        result.raw_channel_values.transpose();
    if (!directional_kernel.empty()) {
      fill_exact_kernel_rows(
          directional_kernel,
          row_begin,
          row_end,
          n_packed_pairs,
          &directional_rows);
      projected.noalias() += directional_rows * accepted_projection_block;
    }
    result.projected_channel_values.middleCols(
        row_begin, row_end - row_begin) = projected.transpose();
  }
  return result;
}

}  // namespace

DirectionalOppositeSpinPairTile build_directional_opposite_spin_pair_tile(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const ActiveSpaceIntegralDirectionView& direction,
    const SameSpinDirectionalPairTileView& same_spin_tile,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors,
    const Eigen::MatrixXd* directional_ri_active_pair_factors) {
  return build_directional_opposite_spin_pair_tile_impl(
      unique_determinants,
      &ordered_pair_cache,
      nullptr,
      n_unique_determinants,
      n_active_orbitals,
      active_space_two_electron_result,
      direction,
      same_spin_tile,
      accepted_ri_active_pair_factors,
      directional_ri_active_pair_factors);
}

DirectionalOppositeSpinPairTile build_directional_opposite_spin_pair_tile(
    const std::vector<std::vector<int>>& unique_determinants,
    const AcceptedSpinPairTile& accepted_pair_tile,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const ActiveSpaceIntegralDirectionView& direction,
    const SameSpinDirectionalPairTileView& same_spin_tile,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors,
    const Eigen::MatrixXd* directional_ri_active_pair_factors) {
  return build_directional_opposite_spin_pair_tile_impl(
      unique_determinants,
      nullptr,
      &accepted_pair_tile,
      static_cast<int>(unique_determinants.size()),
      n_active_orbitals,
      active_space_two_electron_result,
      direction,
      same_spin_tile,
      accepted_ri_active_pair_factors,
      directional_ri_active_pair_factors);
}

}  // namespace xmvb::vb::detail
