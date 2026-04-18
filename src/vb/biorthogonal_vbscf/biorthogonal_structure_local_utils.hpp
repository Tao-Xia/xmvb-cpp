#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "vb/biorthogonal_vbscf/biorthogonal_spin_pair_tiles.hpp"
#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

/**
 * @brief Throws when one dense biorthogonal block contains NaN or Inf.
 *
 * Several biorthogonal matrix builders assemble intermediate dense blocks from
 * cached unique-spin tiles. Keeping this check in one shared helper avoids
 * copy-pasting the same diagnostics across the selected-structure and direct
 * structure builders.
 */
inline void throw_if_nonfinite(
    const Eigen::MatrixXd& matrix,
    const char* label) {
  if (!matrix.allFinite()) {
    throw std::runtime_error(std::string(label) + " contains non-finite entries");
  }
}

/**
 * @brief Local sparse opposite-spin channel family for one structure block.
 *
 * Each unique packed pair touched by the alpha-side sparse projections owns
 * one dense local alpha channel matrix. The beta-side block is projected onto
 * the same packed-pair index on demand before the final Frobenius-style block
 * contraction.
 */
struct LocalOppositeSpinChannelFamily {
  std::vector<int> packed_pair_indices;
  std::vector<Eigen::MatrixXd> alpha_channel_matrices;
};

/**
 * @brief Groups one local alpha projection block by packed active-orbital pair.
 */
inline LocalOppositeSpinChannelFamily build_local_alpha_channel_family(
    const LocalSpinProjectionBlock& local_projection_block,
    int n_packed_active_pairs) {
  LocalOppositeSpinChannelFamily channel_family;
  if (local_projection_block.n_rows <= 0 ||
      local_projection_block.n_cols <= 0 ||
      n_packed_active_pairs <= 0) {
    return channel_family;
  }

  std::vector<int> local_channel_index_by_packed_pair(
      xmvb::to_size(n_packed_active_pairs),
      -1);
  for (int column_local = 0;
       column_local < local_projection_block.n_cols;
       ++column_local) {
    for (int row_local = 0;
         row_local < local_projection_block.n_rows;
         ++row_local) {
      const auto& projection =
          local_projection_block.first_order_projection(row_local, column_local);
      for (std::size_t entry_index = 0;
           entry_index < projection.packed_pair_indices.size();
           ++entry_index) {
        const int packed_pair_index = projection.packed_pair_indices[entry_index];
        if (packed_pair_index < 0 || packed_pair_index >= n_packed_active_pairs) {
          throw std::out_of_range("packed_pair_index outside local opposite-spin range");
        }
        const double packed_pair_value =
            projection.packed_pair_values[entry_index];
        int& channel_index =
            local_channel_index_by_packed_pair[xmvb::to_size(packed_pair_index)];
        if (channel_index < 0) {
          channel_index = static_cast<int>(channel_family.packed_pair_indices.size());
          channel_family.packed_pair_indices.push_back(packed_pair_index);
          channel_family.alpha_channel_matrices.emplace_back(
              Eigen::MatrixXd::Zero(
                  local_projection_block.n_rows,
                  local_projection_block.n_cols));
        }
        xmvb::index_at(
            channel_family.alpha_channel_matrices,
            channel_index)(row_local, column_local) = packed_pair_value;
      }
    }
  }
  return channel_family;
}

/**
 * @brief Projects one local beta sparse block onto a single packed-pair channel.
 */
inline void build_local_beta_projected_channel(
    const LocalSpinProjectionBlock& local_projection_block,
    int target_packed_pair_index,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_orbitals,
    Eigen::MatrixXd* projected_channel_block) {
  if (projected_channel_block == nullptr) {
    throw std::invalid_argument("projected_channel_block must not be null");
  }
  projected_channel_block->resize(
      local_projection_block.n_rows,
      local_projection_block.n_cols);
  for (int column_local = 0;
       column_local < local_projection_block.n_cols;
       ++column_local) {
    for (int row_local = 0;
         row_local < local_projection_block.n_rows;
         ++row_local) {
      const auto& projection =
          local_projection_block.first_order_projection(row_local, column_local);
      (*projected_channel_block)(row_local, column_local) =
          xmvb::vb::project_sparse_projection_onto_packed_pair(
              projection,
              target_packed_pair_index,
              two_electron_view,
          n_orbitals);
    }
  }
}

/**
 * @brief Applies one matrix-form same-spin image `K_alpha C K_beta^T`.
 *
 * `right_coefficients` has dimensions `(n_right_alpha, n_right_beta)`,
 * `alpha_kernel` has dimensions `(n_left_alpha, n_right_alpha)`, and
 * `beta_kernel` has dimensions `(n_left_beta, n_right_beta)`.
 *
 * The output `image` therefore has dimensions
 * `(n_left_alpha, n_left_beta)`.
 */
inline void build_structure_action_image(
    const Eigen::MatrixXd& right_coefficients,
    const Eigen::MatrixXd& alpha_kernel,
    const Eigen::MatrixXd& beta_kernel,
    Eigen::MatrixXd* beta_push,
    Eigen::MatrixXd* image) {
  if (beta_push == nullptr || image == nullptr) {
    throw std::invalid_argument("beta_push and image must not be null");
  }
  if (right_coefficients.rows() != alpha_kernel.cols() ||
      right_coefficients.cols() != beta_kernel.cols()) {
    throw std::invalid_argument("structure action block shape mismatch");
  }
  if (right_coefficients.size() == 0) {
    image->resize(alpha_kernel.rows(), beta_kernel.rows());
    image->setZero();
    return;
  }

  beta_push->resize(right_coefficients.rows(), beta_kernel.rows());
  beta_push->noalias() = right_coefficients * beta_kernel.transpose();
  image->resize(alpha_kernel.rows(), beta_kernel.rows());
  image->noalias() = alpha_kernel * (*beta_push);
}

/**
 * @brief Adds the opposite-spin local image for one matrix-form action block.
 *
 * The sparse alpha-side first-order projections define the channel family,
 * while the beta-side projections are contracted on demand against the active
 * two-electron tensor. The accumulated `total_image` has the same dimensions
 * as one same-spin block image.
 */
inline void add_local_opposite_spin_action_image(
    const Eigen::MatrixXd& right_coefficients,
    const LocalSpinProjectionBlock& alpha_projection_block,
    const LocalSpinProjectionBlock& beta_projection_block,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_orbitals,
    Eigen::MatrixXd* beta_projected_channel_block,
    Eigen::MatrixXd* beta_push,
    Eigen::MatrixXd* channel_image,
    Eigen::MatrixXd* total_image) {
  if (beta_projected_channel_block == nullptr ||
      beta_push == nullptr ||
      channel_image == nullptr ||
      total_image == nullptr) {
    throw std::invalid_argument("opposite-spin image outputs must not be null");
  }

  const int n_packed_active_pairs = packed_active_pair_count(n_orbitals);
  const LocalOppositeSpinChannelFamily alpha_channels =
      build_local_alpha_channel_family(
          alpha_projection_block,
          n_packed_active_pairs);
  for (std::size_t channel_index = 0;
       channel_index < alpha_channels.packed_pair_indices.size();
       ++channel_index) {
    build_local_beta_projected_channel(
        beta_projection_block,
        alpha_channels.packed_pair_indices[channel_index],
        two_electron_view,
        n_orbitals,
        beta_projected_channel_block);
    build_structure_action_image(
        right_coefficients,
        alpha_channels.alpha_channel_matrices[channel_index],
        *beta_projected_channel_block,
        beta_push,
        channel_image);
    (*total_image) += *channel_image;
  }
}

}  // namespace xmvb::vb::biorthogonal_vbscf
