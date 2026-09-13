#include "vbscf/structures/assembly/action.hpp"

#include <algorithm>
#include <stdexcept>

#include "core/openmp.hpp"
#include "vbscf/determinants/pairs/storage.hpp"

namespace xmvb::vb {
namespace {

double contract_opposite_spin(
    const OppositeSpinPackedPairProjection& alpha,
    const OppositeSpinPackedPairProjection& beta,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals,
    int n_packed_pairs) {
  if (alpha.packed_pair_indices.empty() ||
      beta.packed_pair_indices.empty()) {
    return 0.0;
  }
  const bool alpha_projected =
      static_cast<int>(alpha.projected_pair_values.size()) == n_packed_pairs;
  const bool beta_projected =
      static_cast<int>(beta.projected_pair_values.size()) == n_packed_pairs;
  double value = 0.0;
  if (beta_projected &&
      (!alpha_projected ||
       alpha.packed_pair_indices.size() <= beta.packed_pair_indices.size())) {
    for (std::size_t entry = 0;
         entry < alpha.packed_pair_indices.size();
         ++entry) {
      value += alpha.packed_pair_values[entry] *
          beta.projected_pair_values[alpha.packed_pair_indices[entry]];
    }
  } else if (alpha_projected) {
    for (std::size_t entry = 0;
         entry < beta.packed_pair_indices.size();
         ++entry) {
      value += beta.packed_pair_values[entry] *
          alpha.projected_pair_values[beta.packed_pair_indices[entry]];
    }
  } else {
    for (std::size_t alpha_entry = 0;
         alpha_entry < alpha.packed_pair_indices.size();
         ++alpha_entry) {
      for (std::size_t beta_entry = 0;
           beta_entry < beta.packed_pair_indices.size();
           ++beta_entry) {
        value +=
            alpha.packed_pair_values[alpha_entry] *
            lookup_active_space_two_electron_kernel_value(
                two_electron_view,
                alpha.packed_pair_indices[alpha_entry],
                beta.packed_pair_indices[beta_entry],
                n_active_orbitals) *
            beta.packed_pair_values[beta_entry];
      }
    }
  }
  return value;
}

struct DeterminantPairScalars {
  double hamiltonian = 0.0;
  double overlap = 0.0;
};

Eigen::MatrixXd build_pair_matrix(
    const std::vector<SpinDeterminantPairEvaluation>& pair_cache,
    int n_unique,
    bool hamiltonian) {
  Eigen::MatrixXd matrix(n_unique, n_unique);
  for (int left = 0; left < n_unique; ++left) {
    for (int right = 0; right < n_unique; ++right) {
      const auto& pair = pair_cache[
          ordered_spin_pair_storage_index(left, right, n_unique)];
      matrix(left, right) = hamiltonian
          ? pair.total_hamiltonian
          : pair.overlap_result.overlap_determinant;
    }
  }
  return matrix;
}

bool has_projected_pair_values(
    const std::vector<SpinDeterminantPairEvaluation>& pair_cache,
    int n_packed_pairs) {
  for (const auto& pair : pair_cache) {
    const auto& projection =
        pair.opposite_spin_pair_cache.first_order_cofactor_projection;
    if (!projection.packed_pair_indices.empty()) {
      return static_cast<int>(projection.projected_pair_values.size()) ==
          n_packed_pairs;
    }
  }
  return false;
}

bool has_sparse_pair_values(
    const std::vector<SpinDeterminantPairEvaluation>& pair_cache) {
  return std::any_of(
      pair_cache.begin(),
      pair_cache.end(),
      [](const SpinDeterminantPairEvaluation& pair) {
        return !pair.opposite_spin_pair_cache
                    .first_order_cofactor_projection
                    .packed_pair_indices.empty();
      });
}

std::vector<Eigen::MatrixXd> build_projected_pair_matrices(
    const std::vector<SpinDeterminantPairEvaluation>& pair_cache,
    int n_unique,
    int n_packed_pairs) {
  std::vector<Eigen::MatrixXd> matrices;
  matrices.reserve(n_packed_pairs);
  for (int packed_pair = 0;
       packed_pair < n_packed_pairs;
       ++packed_pair) {
    matrices.push_back(Eigen::MatrixXd::Zero(n_unique, n_unique));
  }

  for (int left = 0; left < n_unique; ++left) {
    for (int right = 0; right < n_unique; ++right) {
      const auto& projection = pair_cache[
          ordered_spin_pair_storage_index(left, right, n_unique)]
                                   .opposite_spin_pair_cache
                                   .first_order_cofactor_projection;
      if (projection.packed_pair_indices.empty()) {
        continue;
      }
      if (static_cast<int>(projection.projected_pair_values.size()) !=
          n_packed_pairs) {
        throw std::invalid_argument(
            "factorized structure action requires one projected spin cache");
      }
      for (int packed_pair = 0;
           packed_pair < n_packed_pairs;
           ++packed_pair) {
        matrices[packed_pair](left, right) =
            projection.projected_pair_values[packed_pair];
      }
    }
  }
  return matrices;
}

std::vector<std::vector<Eigen::Triplet<double>>> build_sparse_pair_entries(
    const std::vector<SpinDeterminantPairEvaluation>& pair_cache,
    int n_unique,
    int n_packed_pairs) {
  std::vector<std::vector<Eigen::Triplet<double>>> triplets(n_packed_pairs);
  for (int left = 0; left < n_unique; ++left) {
    for (int right = 0; right < n_unique; ++right) {
      const auto& projection = pair_cache[
          ordered_spin_pair_storage_index(left, right, n_unique)]
                                   .opposite_spin_pair_cache
                                   .first_order_cofactor_projection;
      for (std::size_t entry = 0;
           entry < projection.packed_pair_indices.size();
           ++entry) {
        triplets[projection.packed_pair_indices[entry]].emplace_back(
            left,
            right,
            projection.packed_pair_values[entry]);
      }
    }
  }

  return triplets;
}

struct ChannelSupport {
  std::vector<int> indices;
  bool rows = true;
};

ChannelSupport channel_support(
    const std::vector<Eigen::Triplet<double>>& entries,
    int dimension) {
  std::vector<bool> occupied_rows(dimension, false);
  std::vector<bool> occupied_columns(dimension, false);
  for (const auto& entry : entries) {
    occupied_rows[entry.row()] = true;
    occupied_columns[entry.col()] = true;
  }

  const int n_rows = static_cast<int>(std::count(
      occupied_rows.begin(), occupied_rows.end(), true));
  const int n_columns = static_cast<int>(std::count(
      occupied_columns.begin(), occupied_columns.end(), true));
  ChannelSupport support;
  support.rows = n_rows <= n_columns;
  const auto& occupied = support.rows ? occupied_rows : occupied_columns;
  support.indices.reserve(std::min(n_rows, n_columns));
  for (int index = 0; index < dimension; ++index) {
    if (occupied[index]) {
      support.indices.push_back(index);
    }
  }
  return support;
}

DeterminantPairScalars evaluate_pair(
    const SameSpinPairCacheContext& cache,
    int left_determinant,
    int right_determinant,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals,
    int n_packed_pairs) {
  const int alpha_left =
      cache.alpha_reuse_table.determinant_to_unique_id[left_determinant];
  const int alpha_right =
      cache.alpha_reuse_table.determinant_to_unique_id[right_determinant];
  const int beta_left =
      cache.beta_reuse_table.determinant_to_unique_id[left_determinant];
  const int beta_right =
      cache.beta_reuse_table.determinant_to_unique_id[right_determinant];

  const auto& alpha = cache.alpha_pair_cache_ref()[
      ordered_spin_pair_storage_index(
          alpha_left,
          alpha_right,
          static_cast<int>(
              cache.alpha_reuse_table.unique_determinants.size()))];
  const auto& beta = cache.beta_pair_cache_ref()[
      ordered_spin_pair_storage_index(
          beta_left,
          beta_right,
          static_cast<int>(
              cache.beta_reuse_table.unique_determinants.size()))];

  const double alpha_overlap = alpha.overlap_result.overlap_determinant;
  const double beta_overlap = beta.overlap_result.overlap_determinant;
  DeterminantPairScalars result;
  result.overlap = alpha_overlap * beta_overlap;
  result.hamiltonian =
      alpha.total_hamiltonian * beta_overlap +
      alpha_overlap * beta.total_hamiltonian +
      contract_opposite_spin(
          alpha.opposite_spin_pair_cache.first_order_cofactor_projection,
          beta.opposite_spin_pair_cache.first_order_cofactor_projection,
          two_electron_view,
          n_active_orbitals,
          n_packed_pairs);
  return result;
}

}  // namespace

StructureAction::StructureAction(
    const std::vector<std::vector<StructureExpansionTerm>>&
        determinant_to_structure_terms,
    int n_structures,
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    int n_active_orbitals)
    : n_determinants_(
          static_cast<int>(determinant_to_structure_terms.size())),
      n_structures_(n_structures),
      n_unique_alpha_(static_cast<int>(
          same_spin_pair_cache.alpha_reuse_table.unique_determinants.size())),
      n_unique_beta_(static_cast<int>(
          same_spin_pair_cache.beta_reuse_table.unique_determinants.size())) {
  if (n_determinants_ <= 0 || n_structures_ <= 0) {
    throw std::invalid_argument(
        "matrix-free structure action requires positive dimensions");
  }
  if (!same_spin_pair_cache.enabled()) {
    throw std::invalid_argument(
        "matrix-free structure action requires a same-spin pair cache");
  }
  if (n_unique_alpha_ <= 0 || n_unique_beta_ <= 0) {
    throw std::invalid_argument(
        "matrix-free structure action requires nonempty spin spaces");
  }
  if (static_cast<int>(
          same_spin_pair_cache.alpha_reuse_table
              .determinant_to_unique_id.size()) != n_determinants_ ||
      static_cast<int>(
          same_spin_pair_cache.beta_reuse_table
              .determinant_to_unique_id.size()) != n_determinants_) {
    throw std::invalid_argument(
        "same-spin pair cache does not match determinant expansion");
  }

  const int n_packed_pairs = packed_active_pair_count(n_active_orbitals);
  const auto validate_pair_cache = [n_packed_pairs](
      const std::vector<SpinDeterminantPairEvaluation>& pair_cache,
      int n_unique) {
    if (pair_cache.size() !=
        static_cast<std::size_t>(n_unique) * n_unique) {
      throw std::invalid_argument(
          "ordered same-spin pair cache has incompatible dimensions");
    }
    for (const auto& pair : pair_cache) {
      const auto& opposite_spin = pair.opposite_spin_pair_cache;
      if (opposite_spin.n_packed_active_pairs != n_packed_pairs) {
        throw std::invalid_argument(
            "same-spin pair caches use inconsistent packed-pair dimensions");
      }
      const auto& projection =
          opposite_spin.first_order_cofactor_projection;
      if (projection.packed_pair_indices.size() !=
          projection.packed_pair_values.size()) {
        throw std::invalid_argument(
            "packed-pair projection indices and values differ in size");
      }
      if (!projection.projected_pair_values.empty() &&
          static_cast<int>(projection.projected_pair_values.size()) !=
              n_packed_pairs) {
        throw std::invalid_argument(
            "projected packed-pair vector has incompatible dimensions");
      }
      for (const int packed_pair : projection.packed_pair_indices) {
        if (packed_pair < 0 || packed_pair >= n_packed_pairs) {
          throw std::out_of_range(
              "packed-pair projection index is out of range");
        }
      }
    }
  };
  validate_pair_cache(
      same_spin_pair_cache.alpha_pair_cache_ref(),
      static_cast<int>(
          same_spin_pair_cache.alpha_reuse_table.unique_determinants.size()));
  if (!same_spin_pair_cache.shares_same_spin_pair_cache_between_spins()) {
    validate_pair_cache(
        same_spin_pair_cache.beta_pair_cache_ref(),
        static_cast<int>(
            same_spin_pair_cache.beta_reuse_table.unique_determinants.size()));
  }

  const auto& alpha_pair_cache =
      same_spin_pair_cache.alpha_pair_cache_ref();
  const auto& beta_pair_cache =
      same_spin_pair_cache.beta_pair_cache_ref();
  alpha_overlap_ = build_pair_matrix(
      alpha_pair_cache, n_unique_alpha_, false);
  alpha_hamiltonian_ = build_pair_matrix(
      alpha_pair_cache, n_unique_alpha_, true);
  beta_overlap_ = build_pair_matrix(
      beta_pair_cache, n_unique_beta_, false);
  beta_hamiltonian_ = build_pair_matrix(
      beta_pair_cache, n_unique_beta_, true);

  const bool alpha_projected =
      has_projected_pair_values(alpha_pair_cache, n_packed_pairs);
  const bool beta_projected =
      has_projected_pair_values(beta_pair_cache, n_packed_pairs);
  const bool has_opposite_spin_channels =
      has_sparse_pair_values(alpha_pair_cache) &&
      has_sparse_pair_values(beta_pair_cache);
  if (has_opposite_spin_channels && !alpha_projected && !beta_projected) {
    throw std::invalid_argument(
        "factorized structure action requires one projected spin cache");
  }
  if (has_opposite_spin_channels) {
    alpha_projection_is_dense_ = alpha_projected &&
        (!beta_projected || n_unique_alpha_ <= n_unique_beta_);
    const auto& projected_cache = alpha_projection_is_dense_
        ? alpha_pair_cache
        : beta_pair_cache;
    const auto& sparse_cache = alpha_projection_is_dense_
        ? beta_pair_cache
        : alpha_pair_cache;
    const int n_projected = alpha_projection_is_dense_
        ? n_unique_alpha_
        : n_unique_beta_;
    const int n_sparse = alpha_projection_is_dense_
        ? n_unique_beta_
        : n_unique_alpha_;
    auto projected_matrices = build_projected_pair_matrices(
        projected_cache,
        n_projected,
        n_packed_pairs);
    auto sparse_entries = build_sparse_pair_entries(
        sparse_cache,
        n_sparse,
        n_packed_pairs);
    std::size_t dense_factor_values =
        alpha_overlap_.size() + alpha_hamiltonian_.size() +
        beta_overlap_.size() + beta_hamiltonian_.size();
    for (int packed_pair = 0;
         packed_pair < n_packed_pairs;
         ++packed_pair) {
      if (!projected_matrices[packed_pair].isZero(0.0) &&
          !sparse_entries[packed_pair].empty()) {
        dense_factor_values +=
            projected_matrices[packed_pair].size() +
            static_cast<std::size_t>(n_sparse) * n_sparse;
      }
    }
    const bool dense_factors_fit_memory_bound =
        dense_factor_values <=
        static_cast<std::size_t>(n_structures_) * n_structures_;
    std::vector<int> active_packed_pairs;
    std::vector<ChannelSupport> supports;
    std::size_t row_support_width = 0;
    std::size_t column_support_width = 0;
    for (int packed_pair = 0;
         packed_pair < n_packed_pairs;
         ++packed_pair) {
      if (projected_matrices[packed_pair].isZero(0.0) ||
          sparse_entries[packed_pair].empty()) {
        continue;
      }
      channel_nonzeros_ += sparse_entries[packed_pair].size();
      channel_dense_values_ +=
          static_cast<std::size_t>(n_sparse) * n_sparse;
      active_packed_pairs.push_back(packed_pair);
      supports.push_back(
          channel_support(sparse_entries[packed_pair], n_sparse));
      if (supports.back().indices.size() <
          static_cast<std::size_t>(n_sparse)) {
        auto& family_width = supports.back().rows
            ? row_support_width
            : column_support_width;
        family_width += supports.back().indices.size();
      }
    }

    std::size_t supported_action_bytes =
        static_cast<std::size_t>(
            alpha_overlap_.size() + alpha_hamiltonian_.size() +
            beta_overlap_.size() + beta_hamiltonian_.size()) *
        sizeof(double);
    int row_supported_count = 0;
    int column_supported_count = 0;
    for (std::size_t channel = 0;
         channel < active_packed_pairs.size();
         ++channel) {
      const int packed_pair = active_packed_pairs[channel];
      const auto& support = supports[channel];
      supported_action_bytes +=
          static_cast<std::size_t>(projected_matrices[packed_pair].size()) *
          sizeof(double);
      if (support.indices.size() < static_cast<std::size_t>(n_sparse)) {
        supported_action_bytes +=
            static_cast<std::size_t>(n_sparse) * support.indices.size() *
                sizeof(double) +
            support.indices.size() * sizeof(int);
        if (support.rows) {
          ++row_supported_count;
        } else {
          ++column_supported_count;
        }
        continue;
      }
      const std::size_t dense_bytes =
          static_cast<std::size_t>(n_sparse) * n_sparse * sizeof(double);
      const std::size_t sparse_bytes =
          sparse_entries[packed_pair].size() *
          sizeof(Eigen::Triplet<double>);
      supported_action_bytes += dense_factors_fit_memory_bound
          ? dense_bytes
          : std::min(dense_bytes, sparse_bytes);
    }
    if (row_supported_count > 0) {
      supported_action_bytes +=
          static_cast<std::size_t>(row_supported_count + 1) * sizeof(int);
    }
    if (column_supported_count > 0) {
      supported_action_bytes +=
          static_cast<std::size_t>(column_supported_count + 1) * sizeof(int);
    }
    const bool supported_action_fits_memory_bound =
        supported_action_bytes <=
        static_cast<std::size_t>(n_structures_) * n_structures_ *
            sizeof(double);

    const auto initialize_family = [n_sparse](
        StructureAction::SupportedChannelFamily* family,
        std::size_t width) {
      if (width == 0) {
        return;
      }
      family->offsets.push_back(0);
      family->support.reserve(width);
      family->raw = Eigen::MatrixXd::Zero(n_sparse, width);
    };
    if (supported_action_fits_memory_bound) {
      initialize_family(&row_supported_channels_, row_support_width);
      initialize_family(&column_supported_channels_, column_support_width);
    }

    opposite_spin_channels_.reserve(active_packed_pairs.size());
    for (std::size_t channel = 0;
         channel < active_packed_pairs.size();
         ++channel) {
      const int packed_pair = active_packed_pairs[channel];
      const auto& support = supports[channel];
      const bool use_support_family = supported_action_fits_memory_bound &&
          support.indices.size() < static_cast<std::size_t>(n_sparse);
      if (use_support_family) {
        auto& family = support.rows
            ? row_supported_channels_
            : column_supported_channels_;
        const int first = family.offsets.back();
        std::vector<int> compressed_index(n_sparse, -1);
        for (std::size_t index = 0; index < support.indices.size(); ++index) {
          compressed_index[support.indices[index]] =
              static_cast<int>(index);
        }
        for (const auto& entry : sparse_entries[packed_pair]) {
          const int support_index = support.rows
              ? compressed_index[entry.row()]
              : compressed_index[entry.col()];
          const int raw_index = support.rows ? entry.col() : entry.row();
          family.raw(raw_index, first + support_index) = entry.value();
        }
        family.projected.push_back(
            std::move(projected_matrices[packed_pair]));
        family.support.insert(
            family.support.end(),
            support.indices.begin(),
            support.indices.end());
        family.offsets.push_back(
            first + static_cast<int>(support.indices.size()));
        continue;
      }

      Eigen::MatrixXd dense;
      const std::size_t dense_bytes =
          static_cast<std::size_t>(n_sparse) * n_sparse * sizeof(double);
      const std::size_t sparse_bytes =
          sparse_entries[packed_pair].size() *
          sizeof(Eigen::Triplet<double>);
      if (dense_factors_fit_memory_bound || dense_bytes <= sparse_bytes) {
        dense = Eigen::MatrixXd::Zero(n_sparse, n_sparse);
        for (const auto& entry : sparse_entries[packed_pair]) {
          dense(entry.row(), entry.col()) = entry.value();
        }
        sparse_entries[packed_pair].clear();
        sparse_entries[packed_pair].shrink_to_fit();
      }
      opposite_spin_channels_.push_back(OppositeSpinChannel{
          std::move(projected_matrices[packed_pair]),
          std::move(dense),
          std::move(sparse_entries[packed_pair])});
    }
  }

  determinant_to_spin_product_.resize(n_determinants_);
  for (int determinant = 0;
       determinant < n_determinants_;
       ++determinant) {
    const int alpha = same_spin_pair_cache.alpha_reuse_table
                          .determinant_to_unique_id[determinant];
    const int beta = same_spin_pair_cache.beta_reuse_table
                         .determinant_to_unique_id[determinant];
    determinant_to_spin_product_[determinant] =
        alpha * n_unique_beta_ + beta;
  }

  determinant_term_offsets_.resize(n_determinants_ + 1, 0);
  structure_term_offsets_.resize(n_structures_ + 1, 0);
  std::size_t n_expansion_terms = 0;
  for (int determinant = 0;
       determinant < n_determinants_;
       ++determinant) {
    determinant_term_offsets_[determinant] = n_expansion_terms;
    for (const auto& term : determinant_to_structure_terms[determinant]) {
      if (term.structure_index < 0 ||
          term.structure_index >= n_structures_) {
        throw std::out_of_range(
            "determinant expansion structure index is out of range");
      }
      ++structure_term_offsets_[term.structure_index + 1];
      ++n_expansion_terms;
    }
  }
  determinant_term_offsets_[n_determinants_] = n_expansion_terms;
  for (int structure = 0; structure < n_structures_; ++structure) {
    structure_term_offsets_[structure + 1] +=
        structure_term_offsets_[structure];
  }

  determinant_terms_.reserve(n_expansion_terms);
  structure_terms_.resize(n_expansion_terms);
  std::vector<std::size_t> structure_cursors = structure_term_offsets_;
  for (int determinant = 0;
       determinant < n_determinants_;
       ++determinant) {
    for (const auto& term : determinant_to_structure_terms[determinant]) {
      determinant_terms_.push_back(
          StructureTerm{term.structure_index, term.coefficient});
      structure_terms_[structure_cursors[term.structure_index]++] =
          DeterminantTerm{determinant, term.coefficient};
    }
  }

  diagonal_.hamiltonian = Eigen::VectorXd::Zero(n_structures_);
  diagonal_.overlap = Eigen::VectorXd::Zero(n_structures_);
  const ActiveSpaceTwoElectronView two_electron_view =
      make_active_space_two_electron_view(active_space_two_electron_result);
  const int n_threads = std::max(
      1,
      std::min(xmvb::effective_openmp_thread_count(), n_structures_));

#pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
  for (int structure = 0; structure < n_structures_; ++structure) {
    double hamiltonian = 0.0;
    double overlap = 0.0;
    const std::size_t first = structure_term_offsets_[structure];
    const std::size_t last = structure_term_offsets_[structure + 1];
    for (std::size_t left_index = first; left_index < last; ++left_index) {
      const auto& left = structure_terms_[left_index];
      for (std::size_t right_index = first;
           right_index < last;
           ++right_index) {
        const auto& right = structure_terms_[right_index];
        const DeterminantPairScalars pair = evaluate_pair(
            same_spin_pair_cache,
            left.determinant,
            right.determinant,
            two_electron_view,
            n_active_orbitals,
            n_packed_pairs);
        const double coefficient = left.coefficient * right.coefficient;
        hamiltonian += coefficient * pair.hamiltonian;
        overlap += coefficient * pair.overlap;
      }
    }
    diagonal_.hamiltonian[structure] = hamiltonian;
    diagonal_.overlap[structure] = overlap;
  }
}

void StructureAction::apply_supported_channels(
    const SupportedChannelFamily& family,
    bool supports_rows,
    const Eigen::Ref<const Eigen::MatrixXd>& spin_vector,
    Eigen::MatrixXd* spin_hamiltonian) const {
  if (!family.enabled() || spin_hamiltonian == nullptr) {
    throw std::invalid_argument(
        "supported channel action requires initialized inputs");
  }
  if (alpha_projection_is_dense_) {
    Eigen::MatrixXd raw_images(n_unique_alpha_, family.raw.cols());
    if (supports_rows) {
      raw_images.noalias() = spin_vector * family.raw;
    } else {
      for (int column = 0; column < family.raw.cols(); ++column) {
        raw_images.col(column) = spin_vector.col(family.support[column]);
      }
    }
    Eigen::MatrixXd projected_images(
        n_unique_alpha_, family.raw.cols());
    for (std::size_t channel = 0;
         channel < family.projected.size();
         ++channel) {
      const int first = family.offsets[channel];
      const int width = family.offsets[channel + 1] - first;
      projected_images.middleCols(first, width).noalias() =
          family.projected[channel] *
          raw_images.middleCols(first, width);
    }
    if (supports_rows) {
      for (int column = 0; column < family.raw.cols(); ++column) {
        spin_hamiltonian->col(family.support[column]).noalias() +=
            projected_images.col(column);
      }
    } else {
      spin_hamiltonian->noalias() +=
          projected_images * family.raw.transpose();
    }
    return;
  }

  Eigen::MatrixXd raw_images(family.raw.cols(), n_unique_beta_);
  if (supports_rows) {
    raw_images.noalias() = family.raw.transpose() * spin_vector;
  } else {
    for (int row = 0; row < family.raw.cols(); ++row) {
      raw_images.row(row) = spin_vector.row(family.support[row]);
    }
  }
  Eigen::MatrixXd projected_images(
      family.raw.cols(), n_unique_beta_);
  for (std::size_t channel = 0;
       channel < family.projected.size();
       ++channel) {
    const int first = family.offsets[channel];
    const int width = family.offsets[channel + 1] - first;
    projected_images.middleRows(first, width).noalias() =
        raw_images.middleRows(first, width) *
        family.projected[channel].transpose();
  }
  if (supports_rows) {
    for (int row = 0; row < family.raw.cols(); ++row) {
      spin_hamiltonian->row(family.support[row]).noalias() +=
          projected_images.row(row);
    }
  } else {
    spin_hamiltonian->noalias() += family.raw * projected_images;
  }
}

StructureActionResult StructureAction::apply(
    const Eigen::Ref<const Eigen::MatrixXd>& vectors) const {
  if (vectors.rows() != n_structures_ || vectors.cols() <= 0) {
    throw std::invalid_argument(
        "structure vector block has incompatible dimensions");
  }

  const int block_width = static_cast<int>(vectors.cols());
  Eigen::MatrixXd determinant_vectors =
      Eigen::MatrixXd::Zero(n_determinants_, block_width);
  for (int determinant = 0;
       determinant < n_determinants_;
       ++determinant) {
    for (std::size_t term_index = determinant_term_offsets_[determinant];
         term_index < determinant_term_offsets_[determinant + 1];
         ++term_index) {
      const auto& term = determinant_terms_[term_index];
      determinant_vectors.row(determinant).noalias() +=
          term.coefficient * vectors.row(term.structure);
    }
  }

  Eigen::MatrixXd determinant_hamiltonian =
      Eigen::MatrixXd::Zero(n_determinants_, block_width);
  Eigen::MatrixXd determinant_overlap =
      Eigen::MatrixXd::Zero(n_determinants_, block_width);
  for (int vector = 0; vector < block_width; ++vector) {
    Eigen::MatrixXd spin_vector =
        Eigen::MatrixXd::Zero(n_unique_alpha_, n_unique_beta_);
    for (int determinant = 0;
         determinant < n_determinants_;
         ++determinant) {
      const int spin_product =
          determinant_to_spin_product_[determinant];
      spin_vector(
          spin_product / n_unique_beta_,
          spin_product % n_unique_beta_) =
          determinant_vectors(determinant, vector);
    }

    Eigen::MatrixXd left_product(n_unique_alpha_, n_unique_beta_);
    Eigen::MatrixXd spin_hamiltonian(n_unique_alpha_, n_unique_beta_);
    Eigen::MatrixXd spin_overlap(n_unique_alpha_, n_unique_beta_);
    left_product.noalias() = alpha_hamiltonian_ * spin_vector;
    spin_hamiltonian.noalias() = left_product * beta_overlap_.transpose();
    left_product.noalias() = alpha_overlap_ * spin_vector;
    spin_hamiltonian.noalias() +=
        left_product * beta_hamiltonian_.transpose();
    spin_overlap.noalias() = left_product * beta_overlap_.transpose();
    if (row_supported_channels_.enabled()) {
      apply_supported_channels(
          row_supported_channels_, true, spin_vector, &spin_hamiltonian);
    }
    if (column_supported_channels_.enabled()) {
      apply_supported_channels(
          column_supported_channels_, false, spin_vector, &spin_hamiltonian);
    }
    for (const auto& channel : opposite_spin_channels_) {
      if (alpha_projection_is_dense_) {
        left_product.noalias() = channel.projected * spin_vector;
        if (channel.dense.size() != 0) {
          spin_hamiltonian.noalias() +=
              left_product * channel.dense.transpose();
          continue;
        }
        for (const auto& entry : channel.sparse) {
          spin_hamiltonian.col(entry.row()).noalias() +=
              entry.value() * left_product.col(entry.col());
        }
      } else {
        if (channel.dense.size() != 0) {
          left_product.noalias() = channel.dense * spin_vector;
          spin_hamiltonian.noalias() +=
              left_product * channel.projected.transpose();
          continue;
        }
        left_product.setZero();
        for (const auto& entry : channel.sparse) {
          left_product.row(entry.row()).noalias() +=
              entry.value() * spin_vector.row(entry.col());
        }
        spin_hamiltonian.noalias() +=
            left_product * channel.projected.transpose();
      }
    }

    for (int determinant = 0;
         determinant < n_determinants_;
         ++determinant) {
      const int spin_product =
          determinant_to_spin_product_[determinant];
      const int alpha = spin_product / n_unique_beta_;
      const int beta = spin_product % n_unique_beta_;
      determinant_hamiltonian(determinant, vector) =
          spin_hamiltonian(alpha, beta);
      determinant_overlap(determinant, vector) = spin_overlap(alpha, beta);
    }
  }

  StructureActionResult result;
  result.hamiltonian =
      Eigen::MatrixXd::Zero(n_structures_, block_width);
  result.overlap =
      Eigen::MatrixXd::Zero(n_structures_, block_width);

  for (int structure = 0; structure < n_structures_; ++structure) {
    for (std::size_t term_index = structure_term_offsets_[structure];
         term_index < structure_term_offsets_[structure + 1];
         ++term_index) {
      const auto& term = structure_terms_[term_index];
      result.hamiltonian.row(structure).noalias() +=
          term.coefficient *
          determinant_hamiltonian.row(term.determinant);
      result.overlap.row(structure).noalias() +=
          term.coefficient *
          determinant_overlap.row(term.determinant);
    }
  }
  return result;
}

const StructureDiagonal& StructureAction::diagonal() const noexcept {
  return diagonal_;
}

int StructureAction::n_determinants() const noexcept {
  return n_determinants_;
}

int StructureAction::n_structures() const noexcept {
  return n_structures_;
}

StructureActionStorage StructureAction::storage() const noexcept {
  StructureActionStorage result;
  result.channel_nonzeros = channel_nonzeros_;
  result.channel_dense_values = channel_dense_values_;
  result.expansion_bytes =
      (determinant_term_offsets_.size() + structure_term_offsets_.size()) *
          sizeof(std::size_t) +
      determinant_terms_.size() * sizeof(StructureTerm) +
      structure_terms_.size() * sizeof(DeterminantTerm);
  result.diagonal_bytes =
      static_cast<std::size_t>(
          diagonal_.hamiltonian.size() + diagonal_.overlap.size()) *
      sizeof(double);
  result.factor_bytes =
      static_cast<std::size_t>(
          alpha_overlap_.size() + alpha_hamiltonian_.size() +
          beta_overlap_.size() + beta_hamiltonian_.size()) *
      sizeof(double);
  const auto add_family_storage = [&result](
      const SupportedChannelFamily& family) {
    if (!family.enabled()) {
      return;
    }
    result.factored_channels += static_cast<int>(family.projected.size());
    result.factor_bytes +=
        static_cast<std::size_t>(family.raw.size()) * sizeof(double) +
        (family.support.size() + family.offsets.size()) * sizeof(int);
    for (const auto& projected : family.projected) {
      result.factor_bytes +=
          static_cast<std::size_t>(projected.size()) * sizeof(double);
    }
  };
  add_family_storage(row_supported_channels_);
  add_family_storage(column_supported_channels_);
  for (const auto& channel : opposite_spin_channels_) {
    result.factor_bytes +=
        static_cast<std::size_t>(
            channel.projected.size() + channel.dense.size()) *
            sizeof(double) +
        channel.sparse.size() * sizeof(Eigen::Triplet<double>);
    if (channel.dense.size() != 0) {
      ++result.dense_channels;
    } else {
      ++result.sparse_channels;
    }
  }
  return result;
}

}  // namespace xmvb::vb
