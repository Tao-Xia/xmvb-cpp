#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vb/matrices/same_spin_pair_cache.hpp"
#include "vb/matrices/structure_types.hpp"

namespace xmvb::vb {

/**
 * @brief Local determinant-to-structure coefficient block over unique spin strings.
 *
 * Each selected structure touches only a subset of the unique alpha and beta
 * occupied strings. This block stores that compressed support together with the
 * dense coefficient matrix used later by local structure-matrix contractions.
 */
struct StructureCoefficientBlock {
  std::vector<int> alpha_support;
  std::vector<int> beta_support;
  Eigen::MatrixXd local_coefficients;
  bool close_shell_diagonal = false;
  std::vector<double> local_diagonal_coefficients;
};

namespace detail {

struct StructureCoefficientEntry {
  int alpha_unique_id = 0;
  int beta_unique_id = 0;
  double coefficient = 0.0;
};

inline void sort_and_deduplicate_support(std::vector<int>* support) {
  if (support == nullptr) {
    throw std::invalid_argument("support must not be null");
  }
  std::sort(support->begin(), support->end());
  support->erase(
      std::unique(support->begin(), support->end()),
      support->end());
}

/**
 * @brief Drops support rows or columns whose accumulated coefficients cancel to zero.
 *
 * Different determinants may contribute the same unique-spin pair with opposite
 * signs. After the local coefficient matrix is accumulated, those exact
 * cancellations leave whole rows or columns inactive. Trimming them keeps the
 * later tile gathers aligned with the actual local support.
 */
inline void trim_zero_structure_support(StructureCoefficientBlock* block) {
  if (block == nullptr) {
    throw std::invalid_argument("block must not be null");
  }
  if (block->local_coefficients.size() == 0) {
    block->alpha_support.clear();
    block->beta_support.clear();
    block->local_diagonal_coefficients.clear();
    return;
  }

  std::vector<int> kept_alpha_local_indices;
  std::vector<int> kept_beta_local_indices;
  kept_alpha_local_indices.reserve(block->alpha_support.size());
  kept_beta_local_indices.reserve(block->beta_support.size());

  for (int alpha_local = 0;
       alpha_local < block->local_coefficients.rows();
       ++alpha_local) {
    bool row_has_nonzero = false;
    for (int beta_local = 0;
         beta_local < block->local_coefficients.cols();
         ++beta_local) {
      if (block->local_coefficients(alpha_local, beta_local) != 0.0) {
        row_has_nonzero = true;
        break;
      }
    }
    if (row_has_nonzero) {
      kept_alpha_local_indices.push_back(alpha_local);
    }
  }

  for (int beta_local = 0;
       beta_local < block->local_coefficients.cols();
       ++beta_local) {
    bool column_has_nonzero = false;
    for (int alpha_local = 0;
         alpha_local < block->local_coefficients.rows();
         ++alpha_local) {
      if (block->local_coefficients(alpha_local, beta_local) != 0.0) {
        column_has_nonzero = true;
        break;
      }
    }
    if (column_has_nonzero) {
      kept_beta_local_indices.push_back(beta_local);
    }
  }

  if (kept_alpha_local_indices.size() == block->alpha_support.size() &&
      kept_beta_local_indices.size() == block->beta_support.size()) {
    return;
  }

  Eigen::MatrixXd trimmed_coefficients =
      Eigen::MatrixXd::Zero(
          static_cast<int>(kept_alpha_local_indices.size()),
          static_cast<int>(kept_beta_local_indices.size()));
  std::vector<int> trimmed_alpha_support;
  std::vector<int> trimmed_beta_support;
  trimmed_alpha_support.reserve(kept_alpha_local_indices.size());
  trimmed_beta_support.reserve(kept_beta_local_indices.size());

  for (const int alpha_local : kept_alpha_local_indices) {
    trimmed_alpha_support.push_back(
        block->alpha_support[alpha_local]);
  }
  for (const int beta_local : kept_beta_local_indices) {
    trimmed_beta_support.push_back(
        block->beta_support[beta_local]);
  }

  for (int trimmed_beta_local = 0;
       trimmed_beta_local < static_cast<int>(kept_beta_local_indices.size());
       ++trimmed_beta_local) {
    const int source_beta_local =
        kept_beta_local_indices[trimmed_beta_local];
    for (int trimmed_alpha_local = 0;
         trimmed_alpha_local < static_cast<int>(kept_alpha_local_indices.size());
         ++trimmed_alpha_local) {
      const int source_alpha_local =
          kept_alpha_local_indices[trimmed_alpha_local];
      trimmed_coefficients(trimmed_alpha_local, trimmed_beta_local) =
          block->local_coefficients(source_alpha_local, source_beta_local);
    }
  }

  block->alpha_support = std::move(trimmed_alpha_support);
  block->beta_support = std::move(trimmed_beta_support);
  block->local_coefficients = std::move(trimmed_coefficients);
  if (block->close_shell_diagonal &&
      block->alpha_support == block->beta_support) {
    block->local_diagonal_coefficients.assign(
        block->alpha_support.size(),
        0.0);
    for (int local_index = 0;
         local_index < static_cast<int>(block->alpha_support.size());
         ++local_index) {
      block->local_diagonal_coefficients[local_index] =
          block->local_coefficients(local_index, local_index);
    }
  } else {
    block->close_shell_diagonal = false;
    block->local_diagonal_coefficients.clear();
  }
}

}  // namespace detail

/**
 * @brief Builds one local structure coefficient block per selected structure.
 *
 * The determinant expansion defines a sparse map from full determinants to
 * structures. After same-spin reuse compression, each term becomes one entry
 * in a local `(unique alpha, unique beta)` coefficient matrix for the owning
 * structure. These local dense blocks are the bridge between determinant-space
 * reuse tables and blockwise selected-structure contractions.
 */
inline std::vector<StructureCoefficientBlock> build_structure_coefficient_blocks(
    const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
    const std::vector<int>& structure_to_block_index,
    int n_blocks,
    const SpinDeterminantReuseTable& alpha_reuse_table,
    const SpinDeterminantReuseTable& beta_reuse_table,
    bool prune_zero_support = false) {
  if (n_blocks < 0) {
    throw std::invalid_argument("n_blocks must be non-negative");
  }

  const int n_determinants = static_cast<int>(determinant_to_structure_terms.size());
  const bool close_shell_diagonal =
      alpha_reuse_table.unique_determinants ==
          beta_reuse_table.unique_determinants &&
      alpha_reuse_table.determinant_to_unique_id ==
          beta_reuse_table.determinant_to_unique_id;
  if (static_cast<int>(alpha_reuse_table.determinant_to_unique_id.size()) !=
          n_determinants ||
      static_cast<int>(beta_reuse_table.determinant_to_unique_id.size()) !=
          n_determinants) {
    throw std::invalid_argument(
        "reuse tables determinant_to_unique_id sizes must match n_determinants");
  }

  std::vector<std::vector<detail::StructureCoefficientEntry>> entries_by_structure(
      n_blocks);
  std::vector<std::vector<int>> alpha_support_candidates(
      n_blocks);
  std::vector<std::vector<int>> beta_support_candidates(
      n_blocks);

  for (int determinant_index = 0;
       determinant_index < n_determinants;
       ++determinant_index) {
    const int unique_alpha_id =
        alpha_reuse_table.determinant_to_unique_id[determinant_index];
    const int unique_beta_id =
        beta_reuse_table.determinant_to_unique_id[determinant_index];
    const auto& structure_terms =
        determinant_to_structure_terms[determinant_index];
    for (const auto& term : structure_terms) {
      if (term.structure_index < 0 ||
          term.structure_index >=
              static_cast<int>(structure_to_block_index.size())) {
        throw std::out_of_range("structure index is out of range");
      }
      const int block_index =
          structure_to_block_index[term.structure_index];
      if (block_index < 0) {
        continue;
      }
      if (block_index >= n_blocks) {
        throw std::out_of_range("block index is out of range");
      }
      auto& entries =
          entries_by_structure[block_index];
      entries.push_back(detail::StructureCoefficientEntry{
          unique_alpha_id,
          unique_beta_id,
          term.coefficient});
      alpha_support_candidates[block_index]
          .push_back(unique_alpha_id);
      beta_support_candidates[block_index]
          .push_back(unique_beta_id);
    }
  }

  std::vector<StructureCoefficientBlock> coefficient_blocks(
      n_blocks);
  for (int block_index = 0; block_index < n_blocks; ++block_index) {
    auto& block = coefficient_blocks[block_index];
    block.close_shell_diagonal = close_shell_diagonal;
    block.alpha_support = std::move(
        alpha_support_candidates[block_index]);
    block.beta_support = std::move(
        beta_support_candidates[block_index]);
    detail::sort_and_deduplicate_support(&block.alpha_support);
    detail::sort_and_deduplicate_support(&block.beta_support);

    if (block.alpha_support.empty() || block.beta_support.empty()) {
      block.local_coefficients.resize(0, 0);
      block.local_diagonal_coefficients.clear();
      continue;
    }

    block.local_coefficients =
        Eigen::MatrixXd::Zero(
            static_cast<int>(block.alpha_support.size()),
            static_cast<int>(block.beta_support.size()));
    const auto& entries = entries_by_structure[block_index];
    for (const auto& entry : entries) {
      const auto alpha_iterator =
          std::lower_bound(
              block.alpha_support.begin(),
              block.alpha_support.end(),
              entry.alpha_unique_id);
      const auto beta_iterator =
          std::lower_bound(
              block.beta_support.begin(),
              block.beta_support.end(),
              entry.beta_unique_id);
      if (alpha_iterator == block.alpha_support.end() ||
          *alpha_iterator != entry.alpha_unique_id ||
          beta_iterator == block.beta_support.end() ||
          *beta_iterator != entry.beta_unique_id) {
        throw std::logic_error(
            "failed to locate structure coefficient support entry");
      }
      const int alpha_local =
          static_cast<int>(alpha_iterator - block.alpha_support.begin());
      const int beta_local =
          static_cast<int>(beta_iterator - block.beta_support.begin());
      if (close_shell_diagonal &&
          entry.alpha_unique_id != entry.beta_unique_id) {
        throw std::logic_error(
            "close-shell structure coefficient entry must stay on the diagonal");
      }
      block.local_coefficients(alpha_local, beta_local) += entry.coefficient;
    }

    if (close_shell_diagonal) {
      if (block.alpha_support != block.beta_support) {
        throw std::logic_error(
            "close-shell structure coefficient supports must match");
      }
      block.local_diagonal_coefficients.assign(
          block.alpha_support.size(),
          0.0);
      for (int local_index = 0;
           local_index < static_cast<int>(block.alpha_support.size());
           ++local_index) {
        block.local_diagonal_coefficients[local_index] =
            block.local_coefficients(local_index, local_index);
      }
    }

    if (prune_zero_support) {
      detail::trim_zero_structure_support(&block);
    }
  }

  return coefficient_blocks;
}

inline std::vector<StructureCoefficientBlock> build_structure_coefficient_blocks(
    const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
    int n_structures,
    const SpinDeterminantReuseTable& alpha_reuse_table,
    const SpinDeterminantReuseTable& beta_reuse_table,
    bool prune_zero_support = false) {
  std::vector<int> structure_to_block_index(n_structures, -1);
  for (int structure_index = 0;
       structure_index < n_structures;
       ++structure_index) {
    structure_to_block_index[structure_index] =
        structure_index;
  }
  return build_structure_coefficient_blocks(
      determinant_to_structure_terms,
      structure_to_block_index,
      n_structures,
      alpha_reuse_table,
      beta_reuse_table,
      prune_zero_support);
}

}  // namespace xmvb::vb
