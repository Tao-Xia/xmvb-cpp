#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vb/matrices/contraction_tile_utils.hpp"

namespace xmvb::vb {

/**
 * @brief Scatters one dense support-local image into the current operator tile.
 *
 * The image is defined on the trimmed support windows
 * `row_support[row_window] x column_support[column_window]`, while the target
 * tile is indexed in the global operator tile coordinates starting at
 * `(row_tile_begin, column_tile_begin)`. This helper keeps the global-to-local
 * remapping shared by all bounded-memory matrix-form contractions.
 */
inline void scatter_add_dense_submatrix_to_tile(
    const Eigen::Ref<const Eigen::MatrixXd>& local_matrix,
    const std::vector<int>& row_support,
    const SupportWindow& row_window,
    int row_tile_begin,
    const std::vector<int>& column_support,
    const SupportWindow& column_window,
    int column_tile_begin,
    double scale,
    Eigen::MatrixXd* tile_matrix) {
  if (tile_matrix == nullptr) {
    throw std::invalid_argument("tile_matrix must not be null");
  }
  if (local_matrix.rows() != row_window.size() ||
      local_matrix.cols() != column_window.size()) {
    throw std::invalid_argument(
        "local_matrix shape does not match support windows");
  }

  for (int column_local = 0; column_local < column_window.size(); ++column_local) {
    const int tile_column =
        column_support[static_cast<std::size_t>(column_window.begin + column_local)] -
        column_tile_begin;
    for (int row_local = 0; row_local < row_window.size(); ++row_local) {
      const int tile_row =
          row_support[static_cast<std::size_t>(row_window.begin + row_local)] -
          row_tile_begin;
      (*tile_matrix)(tile_row, tile_column) +=
          scale * local_matrix(row_local, column_local);
    }
  }
}

/**
 * @brief Gathers one dense kernel block from trimmed global supports.
 *
 * `scalar_accessor(row_global, column_global)` is evaluated only on the
 * support-local windows touched by the current contraction tile.
 */
template <typename ScalarAccessor>
inline void gather_scalar_block_from_support(
    const std::vector<int>& row_support,
    const SupportWindow& row_window,
    const std::vector<int>& column_support,
    const SupportWindow& column_window,
    ScalarAccessor&& scalar_accessor,
    Eigen::MatrixXd* block_matrix) {
  if (block_matrix == nullptr) {
    throw std::invalid_argument("block_matrix must not be null");
  }

  block_matrix->resize(row_window.size(), column_window.size());
  for (int column_local = 0; column_local < column_window.size(); ++column_local) {
    const int column_global =
        column_support[static_cast<std::size_t>(column_window.begin + column_local)];
    for (int row_local = 0; row_local < row_window.size(); ++row_local) {
      const int row_global =
          row_support[static_cast<std::size_t>(row_window.begin + row_local)];
      (*block_matrix)(row_local, column_local) =
          scalar_accessor(row_global, column_global);
    }
  }
}

/**
 * @brief Streams alpha-side local images from one support-local coefficient block.
 *
 * The operator tile lives on the alpha support. The beta support is traversed
 * in bounded-memory partner tiles. For each touched beta tile pair, the helper
 * builds two partner kernel tiles, forms the corresponding alpha images by
 *
 * `L_beta * K_beta * R_beta^T`,
 *
 * and hands both images to `image_consumer(...)` immediately.
 */
template <typename PartnerTileBuilder, typename ImageConsumer>
inline void for_each_alpha_oriented_support_local_image_pair(
    const Eigen::MatrixXd& coefficient_matrix,
    const std::vector<int>& alpha_support,
    const std::vector<int>& beta_support,
    int alpha_left_begin,
    int alpha_left_end,
    int alpha_right_begin,
    int alpha_right_end,
    int beta_tile_size,
    PartnerTileBuilder&& build_partner_tiles,
    ImageConsumer&& image_consumer) {
  if (beta_tile_size <= 0) {
    throw std::invalid_argument("beta_tile_size must be positive");
  }
  if (coefficient_matrix.rows() != static_cast<int>(alpha_support.size()) ||
      coefficient_matrix.cols() != static_cast<int>(beta_support.size())) {
    throw std::invalid_argument(
        "coefficient_matrix shape does not match alpha/beta support");
  }

  const SupportWindow alpha_left_window =
      find_support_window(alpha_support, alpha_left_begin, alpha_left_end);
  const SupportWindow alpha_right_window =
      find_support_window(alpha_support, alpha_right_begin, alpha_right_end);
  if (alpha_left_window.empty() ||
      alpha_right_window.empty() ||
      beta_support.empty()) {
    return;
  }

  Eigen::MatrixXd first_partner_tile;
  Eigen::MatrixXd second_partner_tile;
  Eigen::MatrixXd partner_push;
  Eigen::MatrixXd first_image;
  Eigen::MatrixXd second_image;
  const int n_beta_support = static_cast<int>(beta_support.size());
  for (int beta_left_begin_local = 0;
       beta_left_begin_local < n_beta_support;
       beta_left_begin_local += beta_tile_size) {
    const SupportWindow beta_left_window{
        beta_left_begin_local,
        std::min(n_beta_support, beta_left_begin_local + beta_tile_size),
    };
    const auto left_block =
        coefficient_matrix.block(
            alpha_left_window.begin,
            beta_left_window.begin,
            alpha_left_window.size(),
            beta_left_window.size());

    for (int beta_right_begin_local = 0;
         beta_right_begin_local < n_beta_support;
         beta_right_begin_local += beta_tile_size) {
      const SupportWindow beta_right_window{
          beta_right_begin_local,
          std::min(n_beta_support, beta_right_begin_local + beta_tile_size),
      };
      const auto right_block =
          coefficient_matrix.block(
              alpha_right_window.begin,
              beta_right_window.begin,
              alpha_right_window.size(),
              beta_right_window.size());

      build_partner_tiles(
          beta_support,
          beta_left_window,
          beta_right_window,
          &first_partner_tile,
          &second_partner_tile);
      if (first_partner_tile.rows() != beta_left_window.size() ||
          first_partner_tile.cols() != beta_right_window.size() ||
          second_partner_tile.rows() != beta_left_window.size() ||
          second_partner_tile.cols() != beta_right_window.size()) {
        throw std::invalid_argument(
            "partner beta tile shape does not match support-local windows");
      }

      partner_push.noalias() = left_block * first_partner_tile;
      first_image.noalias() = partner_push * right_block.transpose();
      partner_push.noalias() = left_block * second_partner_tile;
      second_image.noalias() = partner_push * right_block.transpose();
      image_consumer(
          alpha_left_window,
          alpha_right_window,
          first_image,
          second_image);
    }
  }
}

/**
 * @brief Streams alpha-side local images from two support-local coefficient blocks.
 *
 * This is the mixed left/right variant needed by directional response:
 *
 * `L_beta * K_beta * R_beta^T`,
 *
 * where the left and right coefficient blocks may have different trimmed
 * alpha/beta supports. The operator tile still lives on the alpha support and
 * every local image is consumed immediately.
 */
template <typename PartnerTileBuilder, typename ImageConsumer>
inline void for_each_alpha_oriented_support_local_mixed_image_pair(
    const Eigen::MatrixXd& left_coefficient_matrix,
    const std::vector<int>& left_alpha_support,
    const std::vector<int>& left_beta_support,
    const Eigen::MatrixXd& right_coefficient_matrix,
    const std::vector<int>& right_alpha_support,
    const std::vector<int>& right_beta_support,
    int alpha_left_begin,
    int alpha_left_end,
    int alpha_right_begin,
    int alpha_right_end,
    int beta_tile_size,
    PartnerTileBuilder&& build_partner_tiles,
    ImageConsumer&& image_consumer) {
  if (beta_tile_size <= 0) {
    throw std::invalid_argument("beta_tile_size must be positive");
  }
  if (left_coefficient_matrix.rows() != static_cast<int>(left_alpha_support.size()) ||
      left_coefficient_matrix.cols() != static_cast<int>(left_beta_support.size()) ||
      right_coefficient_matrix.rows() != static_cast<int>(right_alpha_support.size()) ||
      right_coefficient_matrix.cols() != static_cast<int>(right_beta_support.size())) {
    throw std::invalid_argument(
        "mixed coefficient matrix shape does not match alpha/beta support");
  }

  const SupportWindow alpha_left_window =
      find_support_window(left_alpha_support, alpha_left_begin, alpha_left_end);
  const SupportWindow alpha_right_window =
      find_support_window(right_alpha_support, alpha_right_begin, alpha_right_end);
  if (alpha_left_window.empty() ||
      alpha_right_window.empty() ||
      left_beta_support.empty() ||
      right_beta_support.empty()) {
    return;
  }

  Eigen::MatrixXd first_partner_tile;
  Eigen::MatrixXd second_partner_tile;
  Eigen::MatrixXd partner_push;
  Eigen::MatrixXd first_image;
  Eigen::MatrixXd second_image;
  const int n_left_beta_support = static_cast<int>(left_beta_support.size());
  const int n_right_beta_support = static_cast<int>(right_beta_support.size());
  for (int beta_left_begin_local = 0;
       beta_left_begin_local < n_left_beta_support;
       beta_left_begin_local += beta_tile_size) {
    const SupportWindow beta_left_window{
        beta_left_begin_local,
        std::min(n_left_beta_support, beta_left_begin_local + beta_tile_size),
    };
    const auto left_block =
        left_coefficient_matrix.block(
            alpha_left_window.begin,
            beta_left_window.begin,
            alpha_left_window.size(),
            beta_left_window.size());

    for (int beta_right_begin_local = 0;
         beta_right_begin_local < n_right_beta_support;
         beta_right_begin_local += beta_tile_size) {
      const SupportWindow beta_right_window{
          beta_right_begin_local,
          std::min(n_right_beta_support, beta_right_begin_local + beta_tile_size),
      };
      const auto right_block =
          right_coefficient_matrix.block(
              alpha_right_window.begin,
              beta_right_window.begin,
              alpha_right_window.size(),
              beta_right_window.size());

      build_partner_tiles(
          left_beta_support,
          beta_left_window,
          right_beta_support,
          beta_right_window,
          &first_partner_tile,
          &second_partner_tile);
      if (first_partner_tile.rows() != beta_left_window.size() ||
          first_partner_tile.cols() != beta_right_window.size() ||
          second_partner_tile.rows() != beta_left_window.size() ||
          second_partner_tile.cols() != beta_right_window.size()) {
        throw std::invalid_argument(
            "mixed partner beta tile shape does not match support-local windows");
      }

      partner_push.noalias() = left_block * first_partner_tile;
      first_image.noalias() = partner_push * right_block.transpose();
      partner_push.noalias() = left_block * second_partner_tile;
      second_image.noalias() = partner_push * right_block.transpose();
      image_consumer(
          alpha_left_window,
          alpha_right_window,
          first_image,
          second_image);
    }
  }
}

/**
 * @brief Streams beta-side local images from one support-local coefficient block.
 *
 * The operator tile lives on the beta support. The alpha support is traversed
 * in bounded-memory partner tiles. For each touched alpha tile pair, the helper
 * builds two partner kernel tiles, forms the corresponding beta images by
 *
 * `L_alpha^T * K_alpha * R_alpha`,
 *
 * and hands both images to `image_consumer(...)` immediately.
 */
template <typename PartnerTileBuilder, typename ImageConsumer>
inline void for_each_beta_oriented_support_local_image_pair(
    const Eigen::MatrixXd& coefficient_matrix,
    const std::vector<int>& alpha_support,
    const std::vector<int>& beta_support,
    int beta_left_begin,
    int beta_left_end,
    int beta_right_begin,
    int beta_right_end,
    int alpha_tile_size,
    PartnerTileBuilder&& build_partner_tiles,
    ImageConsumer&& image_consumer) {
  if (alpha_tile_size <= 0) {
    throw std::invalid_argument("alpha_tile_size must be positive");
  }
  if (coefficient_matrix.rows() != static_cast<int>(alpha_support.size()) ||
      coefficient_matrix.cols() != static_cast<int>(beta_support.size())) {
    throw std::invalid_argument(
        "coefficient_matrix shape does not match alpha/beta support");
  }

  const SupportWindow beta_left_window =
      find_support_window(beta_support, beta_left_begin, beta_left_end);
  const SupportWindow beta_right_window =
      find_support_window(beta_support, beta_right_begin, beta_right_end);
  if (beta_left_window.empty() ||
      beta_right_window.empty() ||
      alpha_support.empty()) {
    return;
  }

  Eigen::MatrixXd first_partner_tile;
  Eigen::MatrixXd second_partner_tile;
  Eigen::MatrixXd partner_push;
  Eigen::MatrixXd first_image;
  Eigen::MatrixXd second_image;
  const int n_alpha_support = static_cast<int>(alpha_support.size());
  for (int alpha_left_begin_local = 0;
       alpha_left_begin_local < n_alpha_support;
       alpha_left_begin_local += alpha_tile_size) {
    const SupportWindow alpha_left_window{
        alpha_left_begin_local,
        std::min(n_alpha_support, alpha_left_begin_local + alpha_tile_size),
    };
    const auto left_block =
        coefficient_matrix.block(
            alpha_left_window.begin,
            beta_left_window.begin,
            alpha_left_window.size(),
            beta_left_window.size());

    for (int alpha_right_begin_local = 0;
         alpha_right_begin_local < n_alpha_support;
         alpha_right_begin_local += alpha_tile_size) {
      const SupportWindow alpha_right_window{
          alpha_right_begin_local,
          std::min(n_alpha_support, alpha_right_begin_local + alpha_tile_size),
      };
      const auto right_block =
          coefficient_matrix.block(
              alpha_right_window.begin,
              beta_right_window.begin,
              alpha_right_window.size(),
              beta_right_window.size());

      build_partner_tiles(
          alpha_support,
          alpha_left_window,
          alpha_right_window,
          &first_partner_tile,
          &second_partner_tile);
      if (first_partner_tile.rows() != alpha_left_window.size() ||
          first_partner_tile.cols() != alpha_right_window.size() ||
          second_partner_tile.rows() != alpha_left_window.size() ||
          second_partner_tile.cols() != alpha_right_window.size()) {
        throw std::invalid_argument(
            "partner alpha tile shape does not match support-local windows");
      }

      partner_push.noalias() = left_block.transpose() * first_partner_tile;
      first_image.noalias() = partner_push * right_block;
      partner_push.noalias() = left_block.transpose() * second_partner_tile;
      second_image.noalias() = partner_push * right_block;
      image_consumer(
          beta_left_window,
          beta_right_window,
          first_image,
          second_image);
    }
  }
}

/**
 * @brief Streams beta-side local images from two support-local coefficient blocks.
 *
 * This mixed variant forms
 *
 * `L_alpha^T * K_alpha * R_alpha`,
 *
 * while allowing distinct left/right trimmed supports. It is the beta-oriented
 * analogue of `for_each_alpha_oriented_support_local_mixed_image_pair(...)`.
 */
template <typename PartnerTileBuilder, typename ImageConsumer>
inline void for_each_beta_oriented_support_local_mixed_image_pair(
    const Eigen::MatrixXd& left_coefficient_matrix,
    const std::vector<int>& left_alpha_support,
    const std::vector<int>& left_beta_support,
    const Eigen::MatrixXd& right_coefficient_matrix,
    const std::vector<int>& right_alpha_support,
    const std::vector<int>& right_beta_support,
    int beta_left_begin,
    int beta_left_end,
    int beta_right_begin,
    int beta_right_end,
    int alpha_tile_size,
    PartnerTileBuilder&& build_partner_tiles,
    ImageConsumer&& image_consumer) {
  if (alpha_tile_size <= 0) {
    throw std::invalid_argument("alpha_tile_size must be positive");
  }
  if (left_coefficient_matrix.rows() != static_cast<int>(left_alpha_support.size()) ||
      left_coefficient_matrix.cols() != static_cast<int>(left_beta_support.size()) ||
      right_coefficient_matrix.rows() != static_cast<int>(right_alpha_support.size()) ||
      right_coefficient_matrix.cols() != static_cast<int>(right_beta_support.size())) {
    throw std::invalid_argument(
        "mixed coefficient matrix shape does not match alpha/beta support");
  }

  const SupportWindow beta_left_window =
      find_support_window(left_beta_support, beta_left_begin, beta_left_end);
  const SupportWindow beta_right_window =
      find_support_window(right_beta_support, beta_right_begin, beta_right_end);
  if (beta_left_window.empty() ||
      beta_right_window.empty() ||
      left_alpha_support.empty() ||
      right_alpha_support.empty()) {
    return;
  }

  Eigen::MatrixXd first_partner_tile;
  Eigen::MatrixXd second_partner_tile;
  Eigen::MatrixXd partner_push;
  Eigen::MatrixXd first_image;
  Eigen::MatrixXd second_image;
  const int n_left_alpha_support = static_cast<int>(left_alpha_support.size());
  const int n_right_alpha_support = static_cast<int>(right_alpha_support.size());
  for (int alpha_left_begin_local = 0;
       alpha_left_begin_local < n_left_alpha_support;
       alpha_left_begin_local += alpha_tile_size) {
    const SupportWindow alpha_left_window{
        alpha_left_begin_local,
        std::min(n_left_alpha_support, alpha_left_begin_local + alpha_tile_size),
    };
    const auto left_block =
        left_coefficient_matrix.block(
            alpha_left_window.begin,
            beta_left_window.begin,
            alpha_left_window.size(),
            beta_left_window.size());

    for (int alpha_right_begin_local = 0;
         alpha_right_begin_local < n_right_alpha_support;
         alpha_right_begin_local += alpha_tile_size) {
      const SupportWindow alpha_right_window{
          alpha_right_begin_local,
          std::min(n_right_alpha_support, alpha_right_begin_local + alpha_tile_size),
      };
      const auto right_block =
          right_coefficient_matrix.block(
              alpha_right_window.begin,
              beta_right_window.begin,
              alpha_right_window.size(),
              beta_right_window.size());

      build_partner_tiles(
          left_alpha_support,
          alpha_left_window,
          right_alpha_support,
          alpha_right_window,
          &first_partner_tile,
          &second_partner_tile);
      if (first_partner_tile.rows() != alpha_left_window.size() ||
          first_partner_tile.cols() != alpha_right_window.size() ||
          second_partner_tile.rows() != alpha_left_window.size() ||
          second_partner_tile.cols() != alpha_right_window.size()) {
        throw std::invalid_argument(
            "mixed partner alpha tile shape does not match support-local windows");
      }

      partner_push.noalias() = left_block.transpose() * first_partner_tile;
      first_image.noalias() = partner_push * right_block;
      partner_push.noalias() = left_block.transpose() * second_partner_tile;
      second_image.noalias() = partner_push * right_block;
      image_consumer(
          beta_left_window,
          beta_right_window,
          first_image,
          second_image);
    }
  }
}

}  // namespace xmvb::vb
