#include "vbscf/derivatives/hessian/responses/opposite_spin/pair_response_internal.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "core/openmp.hpp"
#include "vbscf/determinants/pairs/storage.hpp"
#include "vbscf/determinants/pairs/contractions.hpp"
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

std::vector<double> apply_directional_ri_two_electron_kernel(
    const OppositeSpinPackedPairProjection& projection,
    const Eigen::MatrixXd& accepted_factors,
    const Eigen::MatrixXd& directional_factors) {
  if (accepted_factors.rows() != directional_factors.rows() ||
      accepted_factors.cols() != directional_factors.cols()) {
    throw std::invalid_argument(
        "accepted and directional RI pair factors have inconsistent dimensions");
  }

  const int n_packed_pairs = static_cast<int>(accepted_factors.cols());
  Eigen::VectorXd accepted_auxiliary =
      Eigen::VectorXd::Zero(accepted_factors.rows());
  Eigen::VectorXd directional_auxiliary =
      Eigen::VectorXd::Zero(accepted_factors.rows());
  for (std::size_t entry = 0;
       entry < projection.packed_pair_indices.size();
       ++entry) {
    const int packed_pair = projection.packed_pair_indices[entry];
    if (packed_pair < 0 || packed_pair >= n_packed_pairs) {
      throw std::invalid_argument("opposite-spin packed-pair index out of range");
    }
    const double value = projection.packed_pair_values[entry];
    accepted_auxiliary.noalias() += accepted_factors.col(packed_pair) * value;
    directional_auxiliary.noalias() += directional_factors.col(packed_pair) * value;
  }

  const Eigen::VectorXd image =
      accepted_factors.transpose() * directional_auxiliary +
      directional_factors.transpose() * accepted_auxiliary;
  return std::vector<double>(image.data(), image.data() + image.size());
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

const Eigen::MatrixXd& DirectionalOppositeSpinPairTile::raw_channel(
    int packed_pair) const {
  if (packed_pair < 0 ||
      packed_pair >= static_cast<int>(raw_channels.size())) {
    throw std::out_of_range("raw opposite-spin tile channel out of range");
  }
  return raw_channels[packed_pair];
}

const Eigen::MatrixXd& DirectionalOppositeSpinPairTile::projected_channel(
    int packed_pair) const {
  if (packed_pair < 0 ||
      packed_pair >= static_cast<int>(projected_channels.size())) {
    throw std::out_of_range(
        "projected opposite-spin tile channel out of range");
  }
  return projected_channels[packed_pair];
}

DirectionalOppositeSpinPairTile build_directional_opposite_spin_pair_tile(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const ActiveSpaceIntegralDirectionView& direction,
    const SameSpinDirectionalPairTile& same_spin_tile,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors,
    const Eigen::MatrixXd* directional_ri_active_pair_factors) {
  const std::size_t expected_size =
      static_cast<std::size_t>(n_unique_determinants) *
      n_unique_determinants;
  if (unique_determinants.size() !=
          static_cast<std::size_t>(n_unique_determinants) ||
      ordered_pair_cache.size() != expected_size ||
      same_spin_tile.left_begin < 0 || same_spin_tile.right_begin < 0 ||
      same_spin_tile.left_size() <= 0 || same_spin_tile.right_size() <= 0 ||
      same_spin_tile.left_begin + same_spin_tile.left_size() >
          n_unique_determinants ||
      same_spin_tile.right_begin + same_spin_tile.right_size() >
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
  result.left_begin = same_spin_tile.left_begin;
  result.right_begin = same_spin_tile.right_begin;
  result.left_size = same_spin_tile.left_size();
  result.right_size = same_spin_tile.right_size();
  const int work_items = result.left_size * result.right_size;
  result.pairs.resize(static_cast<std::size_t>(work_items));
  result.raw_channels.assign(
      n_packed_pairs,
      Eigen::MatrixXd::Zero(result.left_size, result.right_size));
  result.projected_channels.assign(
      n_packed_pairs,
      Eigen::MatrixXd::Zero(result.left_size, result.right_size));

  for (int work = 0; work < work_items; ++work) {
    const int left_local = work / result.right_size;
    const int right_local = work % result.right_size;
    const int left = result.left_begin + left_local;
    const int right = result.right_begin + right_local;
    auto& entry = result.pairs[static_cast<std::size_t>(work)];
    if (unique_determinants[left].empty()) {
      continue;
    }
    entry.delta_overlap_submatrix = build_overlap_submatrix(
        unique_determinants[left],
        unique_determinants[right],
        direction.overlap,
        n_active_orbitals);
    entry.delta_first_order_cofactor_projection =
        build_sparse_packed_pair_projection(
            unique_determinants[left],
            unique_determinants[right],
            same_spin_tile.pair(left_local, right_local).delta_cofactor_1st,
            n_active_orbitals);
    const auto& raw = entry.delta_first_order_cofactor_projection;
    for (std::size_t projection_entry = 0;
         projection_entry < raw.packed_pair_indices.size();
         ++projection_entry) {
      result.raw_channels[raw.packed_pair_indices[projection_entry]](
          left_local, right_local) +=
          raw.packed_pair_values[projection_entry];
    }
  }

  if (accepted_ri_active_pair_factors != nullptr) {
    const Eigen::Index n_auxiliary =
        accepted_ri_active_pair_factors->rows();
    const Eigen::Index workspace_values_per_column =
        2 * n_auxiliary + 3 * n_packed_pairs;
    const Eigen::Index factor_values = n_auxiliary * n_packed_pairs;
    const int column_block = std::max<int>(
        1,
        std::min<Eigen::Index>(
            work_items,
            factor_values / workspace_values_per_column));
    for (int begin = 0; begin < work_items; begin += column_block) {
      const int end = std::min(work_items, begin + column_block);
      const int width = end - begin;
      Eigen::MatrixXd accepted_projection =
          Eigen::MatrixXd::Zero(n_packed_pairs, width);
      Eigen::MatrixXd directional_projection =
          Eigen::MatrixXd::Zero(n_packed_pairs, width);
      for (int work = begin; work < end; ++work) {
        const int left_local = work / result.right_size;
        const int right_local = work % result.right_size;
        const int left = result.left_begin + left_local;
        const int right = result.right_begin + right_local;
        const auto& accepted = ordered_pair_cache[
            ordered_spin_pair_storage_index(
                std::min(left, right),
                std::max(left, right),
                n_unique_determinants)]
                                   .opposite_spin_pair_cache
                                   .first_order_cofactor_projection;
        const auto& directional = result.pairs[static_cast<std::size_t>(work)]
                                      .delta_first_order_cofactor_projection;
        const int column = work - begin;
        for (std::size_t entry = 0;
             entry < accepted.packed_pair_indices.size();
             ++entry) {
          accepted_projection(
              accepted.packed_pair_indices[entry], column) +=
              accepted.packed_pair_values[entry];
        }
        for (std::size_t entry = 0;
             entry < directional.packed_pair_indices.size();
             ++entry) {
          directional_projection(
              directional.packed_pair_indices[entry], column) +=
              directional.packed_pair_values[entry];
        }
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
      for (int work = begin; work < end; ++work) {
        const int column = work - begin;
        const int left_local = work / result.right_size;
        const int right_local = work % result.right_size;
        auto& values = result.pairs[static_cast<std::size_t>(work)]
                           .delta_first_order_cofactor_projection
                           .projected_pair_values;
        values.assign(
            projected.col(column).data(),
            projected.col(column).data() + n_packed_pairs);
        for (int packed_pair = 0;
             packed_pair < n_packed_pairs;
             ++packed_pair) {
          result.projected_channels[packed_pair](left_local, right_local) =
              projected(packed_pair, column);
        }
      }
    }
    return result;
  }

  const ActiveSpaceTwoElectronView two_electron_view =
      make_active_space_two_electron_view(active_space_two_electron_result);
  for (int work = 0; work < work_items; ++work) {
    const int left_local = work / result.right_size;
    const int right_local = work % result.right_size;
    const int left = result.left_begin + left_local;
    const int right = result.right_begin + right_local;
    auto& projection = result.pairs[static_cast<std::size_t>(work)]
                           .delta_first_order_cofactor_projection;
    projection.projected_pair_values =
        apply_active_space_two_electron_kernel_to_sparse_projection(
            two_electron_view,
            n_active_orbitals,
            projection.packed_pair_indices,
            projection.packed_pair_values);
    const auto& accepted_projection = ordered_pair_cache[
        ordered_spin_pair_storage_index(
            std::min(left, right),
            std::max(left, right),
            n_unique_determinants)]
                                          .opposite_spin_pair_cache
                                          .first_order_cofactor_projection;
    const std::vector<double> kernel_direction =
        apply_directional_two_electron_kernel(
            n_active_orbitals,
            accepted_projection,
            direction.packed_two_electron);
    for (int packed_pair = 0; packed_pair < n_packed_pairs; ++packed_pair) {
      projection.projected_pair_values[packed_pair] +=
          kernel_direction[packed_pair];
      result.projected_channels[packed_pair](left_local, right_local) =
          projection.projected_pair_values[packed_pair];
    }
  }
  return result;
}

std::vector<DirectionalOppositeSpinPairData>
build_directional_opposite_spin_pair_data(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const ActiveSpaceIntegralDirectionView& direction,
    const std::vector<SameSpinPolynomialDirectionalPairData>&
        precomputed_directional_pair_data,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors,
    const Eigen::MatrixXd* directional_ri_active_pair_factors) {
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
  if ((accepted_ri_active_pair_factors == nullptr) !=
      (directional_ri_active_pair_factors == nullptr)) {
    throw std::invalid_argument(
        "opposite-spin RI pair response requires both accepted and directional factors");
  }
  if (accepted_ri_active_pair_factors != nullptr &&
      (accepted_ri_active_pair_factors->cols() !=
           packed_active_pair_count(n_active_orbitals) ||
       accepted_ri_active_pair_factors->rows() !=
           directional_ri_active_pair_factors->rows() ||
       accepted_ri_active_pair_factors->cols() !=
           directional_ri_active_pair_factors->cols())) {
    throw std::invalid_argument(
        "opposite-spin RI pair response factor dimensions are inconsistent");
  }

  std::vector<DirectionalOppositeSpinPairData> result(expected_size);
  const ActiveSpaceTwoElectronView two_electron_view =
      make_active_space_two_electron_view(active_space_two_electron_result);
  const int n_threads = xmvb::effective_openmp_thread_count();
#pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
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
          direction.overlap,
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
          accepted_ri_active_pair_factors != nullptr
          ? apply_directional_ri_two_electron_kernel(
                accepted_projection,
                *accepted_ri_active_pair_factors,
                *directional_ri_active_pair_factors)
          : apply_directional_two_electron_kernel(
                n_active_orbitals,
                accepted_projection,
                direction.packed_two_electron);
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
