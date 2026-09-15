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

struct ChannelSupport {
  std::vector<int> indices;
  bool rows = true;
};

ChannelSupport channel_support(
    const std::vector<bool>& occupied_rows,
    const std::vector<bool>& occupied_columns) {
  const int n_rows = static_cast<int>(std::count(
      occupied_rows.begin(), occupied_rows.end(), true));
  const int n_columns = static_cast<int>(std::count(
      occupied_columns.begin(), occupied_columns.end(), true));
  ChannelSupport support;
  support.rows = n_rows <= n_columns;
  const auto& occupied = support.rows ? occupied_rows : occupied_columns;
  support.indices.reserve(std::min(n_rows, n_columns));
  for (int index = 0; index < static_cast<int>(occupied.size()); ++index) {
    if (occupied[index]) {
      support.indices.push_back(index);
    }
  }
  return support;
}

enum class ChannelStorage {
  Inactive,
  RowFamily,
  ColumnFamily,
  Dense,
  Sparse,
};

struct ChannelPlan {
  ChannelSupport support;
  std::size_t nonzeros = 0;
  int destination = -1;
  int family_offset = -1;
  bool retained = false;
  ChannelStorage storage = ChannelStorage::Inactive;

  bool active() const noexcept {
    return retained;
  }
};

struct ChannelLayout {
  std::vector<ChannelPlan> plans;
  std::vector<int> active_pairs;
  std::size_t nonzeros = 0;
  std::size_t dense_values = 0;
  std::size_t row_width = 0;
  std::size_t column_width = 0;
  int row_family_count = 0;
  int column_family_count = 0;
  int retained_channel_count = 0;
};

std::vector<ChannelPlan> scan_channel_plans(
    const std::vector<SpinDeterminantPairEvaluation>& projected_cache,
    const std::vector<SpinDeterminantPairEvaluation>& raw_cache,
    int n_projected,
    int n_raw,
    int n_packed_pairs) {
  std::vector<bool> projected_nonzero(n_packed_pairs, false);
  for (int left = 0; left < n_projected; ++left) {
    for (int right = 0; right < n_projected; ++right) {
      const auto& projection = projected_cache[
          ordered_spin_pair_storage_index(left, right, n_projected)]
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
        projected_nonzero[packed_pair] =
            projected_nonzero[packed_pair] ||
            projection.projected_pair_values[packed_pair] != 0.0;
      }
    }
  }

  std::vector<std::size_t> nonzeros(n_packed_pairs, 0);
  std::vector<std::vector<bool>> occupied_rows(
      n_packed_pairs, std::vector<bool>(n_raw, false));
  std::vector<std::vector<bool>> occupied_columns(
      n_packed_pairs, std::vector<bool>(n_raw, false));
  for (int left = 0; left < n_raw; ++left) {
    for (int right = 0; right < n_raw; ++right) {
      const auto& projection = raw_cache[
          ordered_spin_pair_storage_index(left, right, n_raw)]
                                   .opposite_spin_pair_cache
                                   .first_order_cofactor_projection;
      for (const int packed_pair : projection.packed_pair_indices) {
        ++nonzeros[packed_pair];
        occupied_rows[packed_pair][left] = true;
        occupied_columns[packed_pair][right] = true;
      }
    }
  }

  std::vector<ChannelPlan> plans(n_packed_pairs);
  for (int packed_pair = 0;
       packed_pair < n_packed_pairs;
       ++packed_pair) {
    if (!projected_nonzero[packed_pair] || nonzeros[packed_pair] == 0) {
      continue;
    }
    plans[packed_pair].nonzeros = nonzeros[packed_pair];
    plans[packed_pair].support = channel_support(
        occupied_rows[packed_pair], occupied_columns[packed_pair]);
    plans[packed_pair].retained = true;
  }
  return plans;
}

ChannelLayout select_channel_layout(
    std::vector<ChannelPlan> plans,
    int n_projected,
    int n_raw,
    int n_structures,
    std::size_t base_factor_values) {
  ChannelLayout layout;
  layout.plans = std::move(plans);
  layout.active_pairs.reserve(layout.plans.size());
  std::size_t dense_factor_values = base_factor_values;
  for (int packed_pair = 0;
       packed_pair < static_cast<int>(layout.plans.size());
       ++packed_pair) {
    const ChannelPlan& plan = layout.plans[packed_pair];
    if (!plan.active()) {
      continue;
    }
    layout.active_pairs.push_back(packed_pair);
    layout.nonzeros += plan.nonzeros;
    layout.dense_values += static_cast<std::size_t>(n_raw) * n_raw;
    dense_factor_values +=
        static_cast<std::size_t>(n_projected) * n_projected +
        static_cast<std::size_t>(n_raw) * n_raw;
  }

  const std::size_t full_structure_values =
      static_cast<std::size_t>(n_structures) * n_structures;
  const bool dense_factors_fit =
      dense_factor_values <= full_structure_values;
  std::size_t retained_bytes = base_factor_values * sizeof(double);
  int candidate_row_channels = 0;
  int candidate_column_channels = 0;
  for (const int packed_pair : layout.active_pairs) {
    const ChannelPlan& plan = layout.plans[packed_pair];
    retained_bytes +=
        static_cast<std::size_t>(n_projected) * n_projected * sizeof(double);
    if (plan.support.indices.size() < static_cast<std::size_t>(n_raw)) {
      retained_bytes +=
          static_cast<std::size_t>(n_raw) * plan.support.indices.size() *
              sizeof(double) +
          plan.support.indices.size() * sizeof(int);
      if (plan.support.rows) {
        ++candidate_row_channels;
      } else {
        ++candidate_column_channels;
      }
      continue;
    }
    const std::size_t dense_bytes =
        static_cast<std::size_t>(n_raw) * n_raw * sizeof(double);
    const std::size_t sparse_bytes =
        plan.nonzeros * sizeof(Eigen::Triplet<double>);
    retained_bytes += dense_factors_fit
        ? dense_bytes
        : std::min(dense_bytes, sparse_bytes);
  }
  if (candidate_row_channels > 0) {
    retained_bytes +=
        static_cast<std::size_t>(candidate_row_channels + 1) * sizeof(int);
  }
  if (candidate_column_channels > 0) {
    retained_bytes +=
        static_cast<std::size_t>(candidate_column_channels + 1) * sizeof(int);
  }
  const bool support_families_fit =
      retained_bytes <= full_structure_values * sizeof(double);

  for (const int packed_pair : layout.active_pairs) {
    ChannelPlan& plan = layout.plans[packed_pair];
    const bool supported = support_families_fit &&
        plan.support.indices.size() < static_cast<std::size_t>(n_raw);
    if (supported && plan.support.rows) {
      plan.storage = ChannelStorage::RowFamily;
      plan.destination = layout.row_family_count++;
      plan.family_offset = static_cast<int>(layout.row_width);
      layout.row_width += plan.support.indices.size();
      continue;
    }
    if (supported) {
      plan.storage = ChannelStorage::ColumnFamily;
      plan.destination = layout.column_family_count++;
      plan.family_offset = static_cast<int>(layout.column_width);
      layout.column_width += plan.support.indices.size();
      continue;
    }

    const std::size_t dense_bytes =
        static_cast<std::size_t>(n_raw) * n_raw * sizeof(double);
    const std::size_t sparse_bytes =
        plan.nonzeros * sizeof(Eigen::Triplet<double>);
    plan.storage = dense_factors_fit || dense_bytes <= sparse_bytes
        ? ChannelStorage::Dense
        : ChannelStorage::Sparse;
    plan.destination = layout.retained_channel_count++;
  }
  return layout;
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

  build_opposite_spin_channels(
      alpha_pair_cache, beta_pair_cache, n_packed_pairs);

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

void StructureAction::build_opposite_spin_channels(
    const std::vector<SpinDeterminantPairEvaluation>& alpha_pair_cache,
    const std::vector<SpinDeterminantPairEvaluation>& beta_pair_cache,
    int n_packed_pairs) {
  const bool alpha_projected =
      has_projected_pair_values(alpha_pair_cache, n_packed_pairs);
  const bool beta_projected =
      has_projected_pair_values(beta_pair_cache, n_packed_pairs);
  const bool has_channels =
      has_sparse_pair_values(alpha_pair_cache) &&
      has_sparse_pair_values(beta_pair_cache);
  if (!has_channels) {
    return;
  }
  if (!alpha_projected && !beta_projected) {
    throw std::invalid_argument(
        "factorized structure action requires one projected spin cache");
  }

  alpha_projection_is_dense_ = alpha_projected &&
      (!beta_projected || n_unique_alpha_ <= n_unique_beta_);
  const auto& projected_cache = alpha_projection_is_dense_
      ? alpha_pair_cache
      : beta_pair_cache;
  const auto& raw_cache = alpha_projection_is_dense_
      ? beta_pair_cache
      : alpha_pair_cache;
  const int n_projected = alpha_projection_is_dense_
      ? n_unique_alpha_
      : n_unique_beta_;
  const int n_raw = alpha_projection_is_dense_
      ? n_unique_beta_
      : n_unique_alpha_;
  const std::size_t base_factor_values =
      alpha_overlap_.size() + alpha_hamiltonian_.size() +
      beta_overlap_.size() + beta_hamiltonian_.size();
  ChannelLayout layout = select_channel_layout(
      scan_channel_plans(
          projected_cache,
          raw_cache,
          n_projected,
          n_raw,
          n_packed_pairs),
      n_projected,
      n_raw,
      n_structures_,
      base_factor_values);
  channel_nonzeros_ = layout.nonzeros;
  channel_dense_values_ = layout.dense_values;
  auto& plans = layout.plans;
  const auto& active_pairs = layout.active_pairs;

  const auto initialize_family = [n_raw](
      SupportedChannelFamily* family,
      int count,
      std::size_t width) {
    if (count == 0) {
      return;
    }
    family->projected.reserve(count);
    family->offsets.reserve(static_cast<std::size_t>(count) + 1);
    family->offsets.push_back(0);
    family->support.reserve(width);
    family->raw = Eigen::MatrixXd::Zero(n_raw, width);
  };
  initialize_family(
      &row_supported_channels_,
      layout.row_family_count,
      layout.row_width);
  initialize_family(
      &column_supported_channels_,
      layout.column_family_count,
      layout.column_width);
  opposite_spin_channels_.reserve(layout.retained_channel_count);

  for (const int packed_pair : active_pairs) {
    ChannelPlan& plan = plans[packed_pair];
    if (plan.storage == ChannelStorage::RowFamily ||
        plan.storage == ChannelStorage::ColumnFamily) {
      SupportedChannelFamily& family =
          plan.storage == ChannelStorage::RowFamily
          ? row_supported_channels_
          : column_supported_channels_;
      family.projected.push_back(
          Eigen::MatrixXd::Zero(n_projected, n_projected));
      family.support.insert(
          family.support.end(),
          plan.support.indices.begin(),
          plan.support.indices.end());
      family.offsets.push_back(
          plan.family_offset + static_cast<int>(plan.support.indices.size()));
      continue;
    }

    OppositeSpinChannel channel;
    channel.projected = Eigen::MatrixXd::Zero(n_projected, n_projected);
    if (plan.storage == ChannelStorage::Dense) {
      channel.dense = Eigen::MatrixXd::Zero(n_raw, n_raw);
    } else {
      channel.sparse.reserve(plan.nonzeros);
    }
    opposite_spin_channels_.push_back(std::move(channel));
  }

  const auto projected_matrix = [this, &plans](
      int packed_pair) -> Eigen::MatrixXd& {
    const ChannelPlan& plan = plans[packed_pair];
    if (plan.storage == ChannelStorage::RowFamily) {
      return row_supported_channels_.projected[plan.destination];
    }
    if (plan.storage == ChannelStorage::ColumnFamily) {
      return column_supported_channels_.projected[plan.destination];
    }
    return opposite_spin_channels_[plan.destination].projected;
  };
  for (int left = 0; left < n_projected; ++left) {
    for (int right = 0; right < n_projected; ++right) {
      const auto& projection = projected_cache[
          ordered_spin_pair_storage_index(left, right, n_projected)]
                                   .opposite_spin_pair_cache
                                   .first_order_cofactor_projection;
      if (projection.packed_pair_indices.empty()) {
        continue;
      }
      for (const int packed_pair : active_pairs) {
        projected_matrix(packed_pair)(left, right) =
            projection.projected_pair_values[packed_pair];
      }
    }
  }

  const auto support_position = [](const ChannelPlan& plan, int index) {
    const auto position = std::lower_bound(
        plan.support.indices.begin(), plan.support.indices.end(), index);
    if (position == plan.support.indices.end() || *position != index) {
      throw std::logic_error("channel support scan is inconsistent");
    }
    return static_cast<int>(position - plan.support.indices.begin());
  };
  for (int left = 0; left < n_raw; ++left) {
    for (int right = 0; right < n_raw; ++right) {
      const auto& projection = raw_cache[
          ordered_spin_pair_storage_index(left, right, n_raw)]
                                   .opposite_spin_pair_cache
                                   .first_order_cofactor_projection;
      for (std::size_t entry = 0;
           entry < projection.packed_pair_indices.size();
           ++entry) {
        const int packed_pair = projection.packed_pair_indices[entry];
        const double value = projection.packed_pair_values[entry];
        const ChannelPlan& plan = plans[packed_pair];
        if (!plan.active()) {
          continue;
        }
        if (plan.storage == ChannelStorage::Dense) {
          opposite_spin_channels_[plan.destination].dense(left, right) = value;
          continue;
        }
        if (plan.storage == ChannelStorage::Sparse) {
          opposite_spin_channels_[plan.destination].sparse.emplace_back(
              left, right, value);
          continue;
        }

        SupportedChannelFamily& family =
            plan.storage == ChannelStorage::RowFamily
            ? row_supported_channels_
            : column_supported_channels_;
        const int supported_index = plan.support.rows ? left : right;
        const int raw_index = plan.support.rows ? right : left;
        family.raw(
            raw_index,
            plan.family_offset + support_position(plan, supported_index)) =
            value;
      }
    }
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
