#include "vbscf/structures/assembly/action.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>

#include "core/openmp.hpp"
#include "vbscf/determinants/pairs/storage.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"
#include "vbscf/structures/orthogonal_ci/integrals.hpp"
#include "vbscf/structures/orthogonal_ci/planner.hpp"
#include "vbscf/structures/orthogonal_ci/sigma.hpp"

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

/**
 * @brief Transposes each horizontal matrix block without changing block order.
 *
 * The input layout is `[X_0 X_1 ...]`; the returned layout is
 * `[X_0^T X_1^T ...]`. Packing all right-hand sides horizontally lets Eigen
 * execute the common factor contraction as one matrix multiplication.
 */
Eigen::MatrixXd transpose_matrix_blocks(
    const Eigen::Ref<const Eigen::MatrixXd>& blocks,
    int block_rows,
    int block_columns) {
  if (block_rows <= 0 || block_columns <= 0 ||
      blocks.rows() != block_rows ||
      blocks.cols() % block_columns != 0) {
    throw std::invalid_argument("matrix block transpose dimensions differ");
  }
  const int block_count = static_cast<int>(blocks.cols()) / block_columns;
  Eigen::MatrixXd transposed(block_columns, block_count * block_rows);
  for (int block = 0; block < block_count; ++block) {
    transposed.middleCols(block * block_rows, block_rows) =
        blocks.middleCols(block * block_columns, block_columns).transpose();
  }
  return transposed;
}

/**
 * @brief Packs one row slice from every horizontal block as transposed columns.
 */
Eigen::MatrixXd pack_transposed_row_slices(
    const Eigen::Ref<const Eigen::MatrixXd>& blocks,
    int first_row,
    int slice_rows,
    int block_columns) {
  if (first_row < 0 || slice_rows <= 0 ||
      first_row + slice_rows > blocks.rows() ||
      block_columns <= 0 || blocks.cols() % block_columns != 0) {
    throw std::invalid_argument("matrix row-slice dimensions differ");
  }
  const int block_count = static_cast<int>(blocks.cols()) / block_columns;
  Eigen::MatrixXd packed(block_columns, block_count * slice_rows);
  for (int block = 0; block < block_count; ++block) {
    packed.middleCols(block * slice_rows, slice_rows) =
        blocks.block(
            first_row,
            block * block_columns,
            slice_rows,
            block_columns).transpose();
  }
  return packed;
}

DeterminantPairScalars evaluate_spin_product_pair(
    const SameSpinPairCacheContext& cache,
    int left_spin_product,
    int right_spin_product,
    int n_unique_beta,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals,
    int n_packed_pairs) {
  const int alpha_left = left_spin_product / n_unique_beta;
  const int alpha_right = right_spin_product / n_unique_beta;
  const int beta_left = left_spin_product % n_unique_beta;
  const int beta_right = right_spin_product % n_unique_beta;

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

struct StructureAction::OrthogonalDirectCiData {
  OrthogonalActiveIntegrals integrals;
  std::vector<std::vector<int>> alpha_determinants;
  std::optional<std::vector<std::vector<int>>> distinct_beta_determinants;
  ExteriorOrbitalTransform alpha_transform;
  std::unique_ptr<ExteriorOrbitalTransform> distinct_beta_transform;
  DirectCiSigmaAction sigma;

  OrthogonalDirectCiData(
      const std::vector<std::vector<int>>& alpha_determinants,
      const std::vector<std::vector<int>>& beta_determinants,
      const std::vector<double>& active_overlap,
      const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron,
      const ActiveSpaceTwoElectronResult& active_two_electron,
      int n_active_orbitals)
      : integrals(orthogonalize_active_integrals(
            active_overlap,
            active_one_electron,
            active_two_electron,
            n_active_orbitals)),
        alpha_determinants(alpha_determinants),
        distinct_beta_determinants(
            beta_determinants == alpha_determinants
                ? std::nullopt
                : std::optional<std::vector<std::vector<int>>>(
                      beta_determinants)),
        alpha_transform(alpha_determinants, integrals.orbital_transform),
        distinct_beta_transform(
            beta_determinants == alpha_determinants
                ? nullptr
                : std::make_unique<ExteriorOrbitalTransform>(
                      beta_determinants,
                      integrals.orbital_transform)),
        sigma(alpha_determinants, beta_determinants, integrals) {}

  const std::vector<std::vector<int>>& beta_determinants() const noexcept {
    return distinct_beta_determinants
        ? *distinct_beta_determinants
        : alpha_determinants;
  }

  const ExteriorOrbitalTransform& beta_transform() const noexcept {
    return distinct_beta_transform
        ? *distinct_beta_transform
        : alpha_transform;
  }

  std::size_t dynamic_bytes() const noexcept {
    std::size_t bytes = static_cast<std::size_t>(
               integrals.orbital_transform.size() +
               integrals.one_electron.size() +
               integrals.pair_kernel.size() +
               integrals.two_electron.ri_active_pair_factors.size()) *
            sizeof(double) +
        integrals.two_electron.packed_active_two_electron_integrals.capacity() *
            sizeof(double) +
        alpha_transform.dynamic_bytes() +
        (distinct_beta_transform
             ? distinct_beta_transform->dynamic_bytes()
             : 0) +
        sigma.dynamic_bytes();
    const auto determinant_bytes = [](const auto& determinants) {
      std::size_t result =
          determinants.capacity() * sizeof(std::vector<int>);
      for (const auto& determinant : determinants) {
        result += determinant.capacity() * sizeof(int);
      }
      return result;
    };
    bytes += determinant_bytes(alpha_determinants);
    if (distinct_beta_determinants) {
      bytes += determinant_bytes(*distinct_beta_determinants);
    }
    return bytes;
  }
};

StructureAction::~StructureAction() = default;
StructureAction::StructureAction(StructureAction&&) noexcept = default;
StructureAction& StructureAction::operator=(StructureAction&&) noexcept = default;

StructureAction::StructureAction(
    const std::vector<std::vector<StructureExpansionTerm>>&
        determinant_to_structure_terms,
    int n_structures,
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron,
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
  const DirectCiActionPlan direct_ci_plan =
      plan_orthogonal_direct_ci_action(
          same_spin_pair_cache.alpha_reuse_table.unique_determinants,
          same_spin_pair_cache.beta_reuse_table.unique_determinants,
          n_active_orbitals,
          1);
  if (direct_ci_plan.favors_direct_ci()) {
    direct_ci_ = std::make_unique<OrthogonalDirectCiData>(
        same_spin_pair_cache.alpha_reuse_table.unique_determinants,
        same_spin_pair_cache.beta_reuse_table.unique_determinants,
        active_overlap,
        active_one_electron,
        active_space_two_electron_result,
        n_active_orbitals);
  } else {
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
  }

  const int n_spin_products = n_unique_alpha_ * n_unique_beta_;
  std::vector<std::vector<StructureTerm>> terms_by_spin_product(
      n_spin_products);
  for (int determinant = 0;
       determinant < n_determinants_;
       ++determinant) {
    const int alpha = same_spin_pair_cache.alpha_reuse_table
                          .determinant_to_unique_id[determinant];
    const int beta = same_spin_pair_cache.beta_reuse_table
                         .determinant_to_unique_id[determinant];
    const int spin_product = alpha * n_unique_beta_ + beta;
    for (const auto& term : determinant_to_structure_terms[determinant]) {
      if (term.structure_index < 0 ||
          term.structure_index >= n_structures_) {
        throw std::out_of_range(
            "determinant expansion structure index is out of range");
      }
      terms_by_spin_product[spin_product].push_back(
          StructureTerm{term.structure_index, term.coefficient});
    }
  }

  spin_term_offsets_.reserve(static_cast<std::size_t>(n_spin_products) + 1);
  spin_term_offsets_.push_back(0);
  for (auto& terms : terms_by_spin_product) {
    std::sort(
        terms.begin(),
        terms.end(),
        [](const StructureTerm& left, const StructureTerm& right) {
          return left.structure < right.structure;
        });
    const std::size_t first = spin_terms_.size();
    for (const StructureTerm& term : terms) {
      if (spin_terms_.size() > first &&
          spin_terms_.back().structure == term.structure) {
        spin_terms_.back().coefficient += term.coefficient;
      } else {
        spin_terms_.push_back(term);
      }
    }
    spin_terms_.erase(
        std::remove_if(
            spin_terms_.begin() + first,
            spin_terms_.end(),
            [](const StructureTerm& term) {
              return term.coefficient == 0.0;
            }),
        spin_terms_.end());
    spin_term_offsets_.push_back(spin_terms_.size());
  }

  struct StructureSpinTerm {
    int spin_product = 0;
    double coefficient = 0.0;
  };
  std::vector<std::vector<StructureSpinTerm>> terms_by_structure(
      n_structures_);
  for (int spin_product = 0;
       spin_product < n_spin_products;
       ++spin_product) {
    for (std::size_t term_index = spin_term_offsets_[spin_product];
         term_index < spin_term_offsets_[spin_product + 1];
         ++term_index) {
      const StructureTerm& term = spin_terms_[term_index];
      terms_by_structure[term.structure].push_back(
          StructureSpinTerm{spin_product, term.coefficient});
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
    for (const StructureSpinTerm& left : terms_by_structure[structure]) {
      for (const StructureSpinTerm& right : terms_by_structure[structure]) {
        const DeterminantPairScalars pair = evaluate_spin_product_pair(
            same_spin_pair_cache,
            left.spin_product,
            right.spin_product,
            n_unique_beta_,
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

void StructureAction::add_supported_channel_block(
    const SupportedChannelFamily& family,
    bool supports_rows,
    const Eigen::Ref<const Eigen::MatrixXd>& spin_vectors,
    const Eigen::Ref<const Eigen::MatrixXd>& transposed_spin_vectors,
    Eigen::MatrixXd* spin_hamiltonians) const {
  if (!family.enabled() || spin_hamiltonians == nullptr ||
      spin_vectors.rows() != n_unique_alpha_ ||
      spin_vectors.cols() % n_unique_beta_ != 0) {
    throw std::invalid_argument(
        "supported channel block has incompatible dimensions");
  }
  const int block_width =
      static_cast<int>(spin_vectors.cols()) / n_unique_beta_;
  if (transposed_spin_vectors.rows() != n_unique_beta_ ||
      transposed_spin_vectors.cols() != block_width * n_unique_alpha_ ||
      spin_hamiltonians->rows() != n_unique_alpha_ ||
      spin_hamiltonians->cols() != spin_vectors.cols()) {
    throw std::invalid_argument(
        "transposed supported channel block has incompatible dimensions");
  }
  if (alpha_projection_is_dense_) {
    if (supports_rows) {
      add_alpha_projected_row_support(
          family, transposed_spin_vectors, spin_hamiltonians);
      return;
    }
    add_alpha_projected_column_support(
        family, spin_vectors, spin_hamiltonians);
    return;
  }

  if (supports_rows) {
    add_beta_projected_row_support(
        family, spin_vectors, spin_hamiltonians);
    return;
  }

  add_beta_projected_column_support(
      family, spin_vectors, spin_hamiltonians);
}

void StructureAction::add_alpha_projected_row_support(
    const SupportedChannelFamily& family,
    const Eigen::Ref<const Eigen::MatrixXd>& transposed_spin_vectors,
    Eigen::MatrixXd* spin_hamiltonians) const {
  const int block_width = static_cast<int>(
      transposed_spin_vectors.cols()) / n_unique_alpha_;
  const Eigen::MatrixXd raw_images_transposed =
      family.raw.transpose() * transposed_spin_vectors;
  for (std::size_t channel = 0;
       channel < family.projected.size();
       ++channel) {
    const int first = family.offsets[channel];
    const int width = family.offsets[channel + 1] - first;
    const Eigen::MatrixXd packed = pack_transposed_row_slices(
        raw_images_transposed, first, width, n_unique_alpha_);
    const Eigen::MatrixXd projected = family.projected[channel] * packed;
    for (int vector = 0; vector < block_width; ++vector) {
      for (int local = 0; local < width; ++local) {
        spin_hamiltonians->col(
            vector * n_unique_beta_ + family.support[first + local]) +=
            projected.col(vector * width + local);
      }
    }
  }
}

void StructureAction::add_alpha_projected_column_support(
    const SupportedChannelFamily& family,
    const Eigen::Ref<const Eigen::MatrixXd>& spin_vectors,
    Eigen::MatrixXd* spin_hamiltonians) const {
  const int block_width =
      static_cast<int>(spin_vectors.cols()) / n_unique_beta_;
  const int family_width = static_cast<int>(family.raw.cols());
  Eigen::MatrixXd raw_images(n_unique_alpha_, block_width * family_width);
  for (int vector = 0; vector < block_width; ++vector) {
    for (int column = 0; column < family_width; ++column) {
      raw_images.col(vector * family_width + column) =
          spin_vectors.col(
              vector * n_unique_beta_ + family.support[column]);
    }
  }

  Eigen::MatrixXd projected_images(n_unique_alpha_, raw_images.cols());
  for (std::size_t channel = 0;
       channel < family.projected.size();
       ++channel) {
    const int first = family.offsets[channel];
    const int width = family.offsets[channel + 1] - first;
    Eigen::MatrixXd packed(n_unique_alpha_, block_width * width);
    for (int vector = 0; vector < block_width; ++vector) {
      packed.middleCols(vector * width, width) =
          raw_images.middleCols(vector * family_width + first, width);
    }
    const Eigen::MatrixXd projected = family.projected[channel] * packed;
    for (int vector = 0; vector < block_width; ++vector) {
      projected_images.middleCols(
          vector * family_width + first, width) =
          projected.middleCols(vector * width, width);
    }
  }
  const Eigen::MatrixXd projected_images_transposed = transpose_matrix_blocks(
      projected_images, n_unique_alpha_, family_width);
  const Eigen::MatrixXd spin_images_transposed =
      family.raw * projected_images_transposed;
  spin_hamiltonians->noalias() += transpose_matrix_blocks(
      spin_images_transposed, n_unique_beta_, n_unique_alpha_);
}

void StructureAction::add_beta_projected_row_support(
    const SupportedChannelFamily& family,
    const Eigen::Ref<const Eigen::MatrixXd>& spin_vectors,
    Eigen::MatrixXd* spin_hamiltonians) const {
  const int block_width =
      static_cast<int>(spin_vectors.cols()) / n_unique_beta_;
  const Eigen::MatrixXd raw_images = family.raw.transpose() * spin_vectors;
  for (std::size_t channel = 0;
       channel < family.projected.size();
       ++channel) {
    const int first = family.offsets[channel];
    const int width = family.offsets[channel + 1] - first;
    const Eigen::MatrixXd packed = pack_transposed_row_slices(
        raw_images, first, width, n_unique_beta_);
    const Eigen::MatrixXd projected = family.projected[channel] * packed;
    for (int vector = 0; vector < block_width; ++vector) {
      for (int local = 0; local < width; ++local) {
        spin_hamiltonians->row(family.support[first + local])
            .segment(vector * n_unique_beta_, n_unique_beta_) +=
            projected.col(vector * width + local).transpose();
      }
    }
  }
}

void StructureAction::add_beta_projected_column_support(
    const SupportedChannelFamily& family,
    const Eigen::Ref<const Eigen::MatrixXd>& spin_vectors,
    Eigen::MatrixXd* spin_hamiltonians) const {
  const int block_width =
      static_cast<int>(spin_vectors.cols()) / n_unique_beta_;
  const int family_width = static_cast<int>(family.raw.cols());

  Eigen::MatrixXd raw_images(family_width, spin_vectors.cols());
  for (int row = 0; row < family_width; ++row) {
    for (int vector = 0; vector < block_width; ++vector) {
      raw_images.row(row).segment(
          vector * n_unique_beta_, n_unique_beta_) =
          spin_vectors.row(family.support[row]).segment(
              vector * n_unique_beta_, n_unique_beta_);
    }
  }
  Eigen::MatrixXd projected_images(family_width, spin_vectors.cols());
  for (std::size_t channel = 0;
       channel < family.projected.size();
       ++channel) {
    const int first = family.offsets[channel];
    const int width = family.offsets[channel + 1] - first;
    const Eigen::MatrixXd packed = pack_transposed_row_slices(
        raw_images, first, width, n_unique_beta_);
    const Eigen::MatrixXd projected = family.projected[channel] * packed;
    for (int vector = 0; vector < block_width; ++vector) {
      projected_images.block(
          first,
          vector * n_unique_beta_,
          width,
          n_unique_beta_) =
          projected.middleCols(vector * width, width).transpose();
    }
  }
  spin_hamiltonians->noalias() += family.raw * projected_images;
}

void StructureAction::add_individual_channels(
    const Eigen::Ref<const Eigen::MatrixXd>& spin_vector,
    Eigen::MatrixXd* spin_hamiltonian) const {
  if (spin_vector.rows() != n_unique_alpha_ ||
      spin_vector.cols() != n_unique_beta_ ||
      spin_hamiltonian == nullptr ||
      spin_hamiltonian->rows() != n_unique_alpha_ ||
      spin_hamiltonian->cols() != n_unique_beta_) {
    throw std::invalid_argument(
        "opposite-spin channel action has incompatible dimensions");
  }
  Eigen::MatrixXd product(n_unique_alpha_, n_unique_beta_);
  for (const auto& channel : opposite_spin_channels_) {
    if (alpha_projection_is_dense_) {
      product.noalias() = channel.projected * spin_vector;
      if (channel.dense.size() != 0) {
        spin_hamiltonian->noalias() +=
            product * channel.dense.transpose();
        continue;
      }
      for (const auto& entry : channel.sparse) {
        spin_hamiltonian->col(entry.row()).noalias() +=
            entry.value() * product.col(entry.col());
      }
      continue;
    }

    if (channel.dense.size() != 0) {
      product.noalias() = channel.dense * spin_vector;
      spin_hamiltonian->noalias() +=
          product * channel.projected.transpose();
      continue;
    }
    product.setZero();
    for (const auto& entry : channel.sparse) {
      product.row(entry.row()).noalias() +=
          entry.value() * spin_vector.row(entry.col());
    }
    spin_hamiltonian->noalias() +=
        product * channel.projected.transpose();
  }
}

Eigen::MatrixXd StructureAction::contract_spin_product_block(
    const Eigen::Ref<const Eigen::MatrixXd>& spin_images) const {
  if (spin_images.rows() != n_unique_alpha_ ||
      spin_images.cols() <= 0 ||
      spin_images.cols() % n_unique_beta_ != 0) {
    throw std::invalid_argument(
        "unique-string-product image block has incompatible dimensions");
  }
  const int block_width =
      static_cast<int>(spin_images.cols()) / n_unique_beta_;
  Eigen::MatrixXd result =
      Eigen::MatrixXd::Zero(n_structures_, block_width);
  const int n_spin_products = n_unique_alpha_ * n_unique_beta_;
  for (int spin_product = 0;
       spin_product < n_spin_products;
       ++spin_product) {
    const int alpha = spin_product / n_unique_beta_;
    const int beta = spin_product % n_unique_beta_;
    for (std::size_t term_index = spin_term_offsets_[spin_product];
         term_index < spin_term_offsets_[spin_product + 1];
         ++term_index) {
      const StructureTerm& term = spin_terms_[term_index];
      for (int vector = 0; vector < block_width; ++vector) {
        result(term.structure, vector) +=
            term.coefficient *
            spin_images(alpha, vector * n_unique_beta_ + beta);
      }
    }
  }
  return result;
}

Eigen::MatrixXd StructureAction::expand_structure_block(
    const Eigen::Ref<const Eigen::MatrixXd>& vectors) const {
  if (vectors.rows() != n_structures_ || vectors.cols() <= 0) {
    throw std::invalid_argument(
        "structure vector block has incompatible dimensions");
  }

  const int block_width = static_cast<int>(vectors.cols());
  Eigen::MatrixXd spin_vectors = Eigen::MatrixXd::Zero(
      n_unique_alpha_, block_width * n_unique_beta_);
  const int n_spin_products = n_unique_alpha_ * n_unique_beta_;
  for (int spin_product = 0;
       spin_product < n_spin_products;
       ++spin_product) {
    const int alpha = spin_product / n_unique_beta_;
    const int beta = spin_product % n_unique_beta_;
    for (std::size_t term_index = spin_term_offsets_[spin_product];
         term_index < spin_term_offsets_[spin_product + 1];
         ++term_index) {
      const StructureTerm& term = spin_terms_[term_index];
      for (int vector = 0; vector < block_width; ++vector) {
        spin_vectors(alpha, vector * n_unique_beta_ + beta) +=
            term.coefficient * vectors(term.structure, vector);
      }
    }
  }
  return spin_vectors;
}

StructureActionResult StructureAction::apply(
    const Eigen::Ref<const Eigen::MatrixXd>& vectors) const {
  Eigen::MatrixXd spin_vectors = expand_structure_block(vectors);
  const int block_width = static_cast<int>(vectors.cols());
  if (direct_ci_) {
    direct_ci_->alpha_transform.apply_left(&spin_vectors);
    for (int block = 0; block < block_width; ++block) {
      Eigen::MatrixXd coefficient_block = spin_vectors.middleCols(
          block * n_unique_beta_, n_unique_beta_);
      direct_ci_->beta_transform().apply_right(&coefficient_block);
      spin_vectors.middleCols(
          block * n_unique_beta_, n_unique_beta_) = coefficient_block;
    }

    Eigen::MatrixXd spin_hamiltonians = direct_ci_->sigma.apply(spin_vectors);
    Eigen::MatrixXd spin_overlaps = std::move(spin_vectors);
    for (int block = 0; block < block_width; ++block) {
      Eigen::MatrixXd hamiltonian_block = spin_hamiltonians.middleCols(
          block * n_unique_beta_, n_unique_beta_);
      Eigen::MatrixXd overlap_block = spin_overlaps.middleCols(
          block * n_unique_beta_, n_unique_beta_);
      direct_ci_->beta_transform().apply_adjoint_right(&hamiltonian_block);
      direct_ci_->beta_transform().apply_adjoint_right(&overlap_block);
      spin_hamiltonians.middleCols(
          block * n_unique_beta_, n_unique_beta_) = hamiltonian_block;
      spin_overlaps.middleCols(
          block * n_unique_beta_, n_unique_beta_) = overlap_block;
    }
    direct_ci_->alpha_transform.apply_adjoint_left(&spin_hamiltonians);
    direct_ci_->alpha_transform.apply_adjoint_left(&spin_overlaps);
    return StructureActionResult{
        contract_spin_product_block(spin_hamiltonians),
        contract_spin_product_block(spin_overlaps)};
  }

  const Eigen::MatrixXd transposed_spin_vectors = transpose_matrix_blocks(
      spin_vectors, n_unique_alpha_, n_unique_beta_);
  const Eigen::MatrixXd right_overlap_transposed =
      beta_overlap_ * transposed_spin_vectors;
  const Eigen::MatrixXd right_hamiltonian_transposed =
      beta_hamiltonian_ * transposed_spin_vectors;
  const Eigen::MatrixXd right_overlap = transpose_matrix_blocks(
      right_overlap_transposed, n_unique_beta_, n_unique_alpha_);
  const Eigen::MatrixXd right_hamiltonian = transpose_matrix_blocks(
      right_hamiltonian_transposed, n_unique_beta_, n_unique_alpha_);
  Eigen::MatrixXd spin_hamiltonians = alpha_hamiltonian_ * right_overlap;
  spin_hamiltonians.noalias() += alpha_overlap_ * right_hamiltonian;
  Eigen::MatrixXd spin_overlaps = alpha_overlap_ * right_overlap;
  if (row_supported_channels_.enabled()) {
    add_supported_channel_block(
        row_supported_channels_,
        true,
        spin_vectors,
        transposed_spin_vectors,
        &spin_hamiltonians);
  }
  if (column_supported_channels_.enabled()) {
    add_supported_channel_block(
        column_supported_channels_,
        false,
        spin_vectors,
        transposed_spin_vectors,
        &spin_hamiltonians);
  }

  for (int vector = 0; vector < block_width; ++vector) {
    Eigen::MatrixXd channel_hamiltonian =
        Eigen::MatrixXd::Zero(n_unique_alpha_, n_unique_beta_);
    add_individual_channels(
        spin_vectors.middleCols(
            vector * n_unique_beta_, n_unique_beta_),
        &channel_hamiltonian);
    spin_hamiltonians.middleCols(
        vector * n_unique_beta_, n_unique_beta_) += channel_hamiltonian;
  }

  return StructureActionResult{
      contract_spin_product_block(spin_hamiltonians),
      contract_spin_product_block(spin_overlaps)};
}

StructureActionResult StructureAction::apply_integral_direction(
    const Eigen::Ref<const Eigen::MatrixXd>& vectors,
    const std::vector<double>& overlap_direction,
    const std::vector<double>& one_electron_direction,
    const std::vector<double>& packed_two_electron_direction,
    StructureIntegralDirection* direct_ci_direction) const {
  if (!direct_ci_) {
    throw std::logic_error(
        "integral-direction action requires orthogonal direct CI");
  }
  const int n_active_orbitals =
      static_cast<int>(direct_ci_->integrals.one_electron.rows());
  if (one_electron_direction.size() !=
      static_cast<std::size_t>(n_active_orbitals) * n_active_orbitals) {
    throw std::invalid_argument(
        "active one-electron direction has incompatible dimensions");
  }
  const Eigen::Map<const Eigen::MatrixXd> delta_one_electron(
      one_electron_direction.data(),
      n_active_orbitals,
      n_active_orbitals);
  const OrthogonalActiveIntegrals integral_direction =
      orthogonalize_active_integral_direction(
          direct_ci_->integrals,
          overlap_direction,
          delta_one_electron,
          packed_two_electron_direction,
          n_active_orbitals);
  const ExteriorTransformDirection alpha_direction =
      direct_ci_->alpha_transform.direction(
          integral_direction.orbital_transform);
  const ExteriorTransformDirection beta_direction =
      direct_ci_->beta_transform().direction(
          integral_direction.orbital_transform);
  const DirectCiSigmaAction sigma_direction(
      direct_ci_->alpha_determinants,
      direct_ci_->beta_determinants(),
      integral_direction);

  Eigen::MatrixXd spin_vectors = expand_structure_block(vectors);
  Eigen::MatrixXd delta_spin_vectors = Eigen::MatrixXd::Zero(
      spin_vectors.rows(),
      spin_vectors.cols());
  direct_ci_->alpha_transform.apply_directional_left(
      alpha_direction,
      &spin_vectors,
      &delta_spin_vectors);
  const int block_width = static_cast<int>(vectors.cols());
  for (int block = 0; block < block_width; ++block) {
    Eigen::MatrixXd coefficient_block = spin_vectors.middleCols(
        block * n_unique_beta_, n_unique_beta_);
    Eigen::MatrixXd delta_coefficient_block =
        delta_spin_vectors.middleCols(
            block * n_unique_beta_, n_unique_beta_);
    direct_ci_->beta_transform().apply_directional_right(
        beta_direction,
        &coefficient_block,
        &delta_coefficient_block);
    spin_vectors.middleCols(
        block * n_unique_beta_, n_unique_beta_) = coefficient_block;
    delta_spin_vectors.middleCols(
        block * n_unique_beta_, n_unique_beta_) = delta_coefficient_block;
  }

  Eigen::MatrixXd spin_overlaps = spin_vectors;
  Eigen::MatrixXd delta_spin_overlaps = delta_spin_vectors;
  Eigen::MatrixXd spin_hamiltonians = direct_ci_->sigma.apply(spin_vectors);
  Eigen::MatrixXd delta_spin_hamiltonians =
      direct_ci_->sigma.apply(delta_spin_vectors);
  delta_spin_hamiltonians += sigma_direction.apply(spin_vectors);
  if (direct_ci_direction != nullptr) {
    direct_ci_direction->coefficient_direction = delta_spin_vectors;
    direct_ci_direction->sigma_direction = delta_spin_hamiltonians;
  }
  for (int block = 0; block < block_width; ++block) {
    Eigen::MatrixXd hamiltonian_block = spin_hamiltonians.middleCols(
        block * n_unique_beta_, n_unique_beta_);
    Eigen::MatrixXd delta_hamiltonian_block =
        delta_spin_hamiltonians.middleCols(
            block * n_unique_beta_, n_unique_beta_);
    Eigen::MatrixXd overlap_block = spin_overlaps.middleCols(
        block * n_unique_beta_, n_unique_beta_);
    Eigen::MatrixXd delta_overlap_block =
        delta_spin_overlaps.middleCols(
            block * n_unique_beta_, n_unique_beta_);
    direct_ci_->beta_transform().apply_directional_adjoint_right(
        beta_direction,
        &hamiltonian_block,
        &delta_hamiltonian_block);
    direct_ci_->beta_transform().apply_directional_adjoint_right(
        beta_direction,
        &overlap_block,
        &delta_overlap_block);
    spin_hamiltonians.middleCols(
        block * n_unique_beta_, n_unique_beta_) = hamiltonian_block;
    delta_spin_hamiltonians.middleCols(
        block * n_unique_beta_, n_unique_beta_) = delta_hamiltonian_block;
    spin_overlaps.middleCols(
        block * n_unique_beta_, n_unique_beta_) = overlap_block;
    delta_spin_overlaps.middleCols(
        block * n_unique_beta_, n_unique_beta_) = delta_overlap_block;
  }
  direct_ci_->alpha_transform.apply_directional_adjoint_left(
      alpha_direction,
      &spin_hamiltonians,
      &delta_spin_hamiltonians);
  direct_ci_->alpha_transform.apply_directional_adjoint_left(
      alpha_direction,
      &spin_overlaps,
      &delta_spin_overlaps);
  return StructureActionResult{
      contract_spin_product_block(delta_spin_hamiltonians),
      contract_spin_product_block(delta_spin_overlaps)};
}

bool StructureAction::supports_integral_direction() const noexcept {
  return direct_ci_ != nullptr;
}

StructureAdjointState StructureAction::prepare_active_adjoint(
    const SelectedStateDeterminantMatrices& selected_states,
    const std::vector<double>& state_energies) const {
  if (!direct_ci_) {
    throw std::logic_error(
        "active-integral adjoint requires orthogonal direct CI");
  }
  if (selected_states.states.empty() ||
      selected_states.states.size() != state_energies.size() ||
      selected_states.n_unique_alpha != n_unique_alpha_ ||
      selected_states.n_unique_beta != n_unique_beta_) {
    throw std::invalid_argument(
        "selected-state adjoint inputs have incompatible dimensions");
  }

  const int n_orbitals =
      static_cast<int>(direct_ci_->integrals.one_electron.rows());
  const int n_pairs = packed_active_pair_count(n_orbitals);
  StructureAdjointState result;
  result.source_coefficients.reserve(selected_states.states.size());
  result.orthogonal_coefficients.reserve(selected_states.states.size());
  result.residuals.reserve(selected_states.states.size());
  result.weights.reserve(selected_states.states.size());
  result.energies = state_energies;
  result.one_electron_gradient = Eigen::MatrixXd::Zero(
      n_orbitals, n_orbitals);
  result.pair_kernel_gradient = Eigen::MatrixXd::Zero(
      n_pairs, n_pairs);
  result.generator_gradient = Eigen::MatrixXd::Zero(
      n_orbitals, n_orbitals);

  for (std::size_t state = 0; state < selected_states.states.size(); ++state) {
    const auto& selected_state = selected_states.states[state];
    const double state_weight = selected_state.normalized_state_weight;
    if (selected_state.coefficient_matrix.rows() != n_unique_alpha_ ||
        selected_state.coefficient_matrix.cols() != n_unique_beta_ ||
        !selected_state.coefficient_matrix.allFinite() ||
        !std::isfinite(state_weight) || state_weight < 0.0 ||
        !std::isfinite(state_energies[state])) {
      throw std::invalid_argument("invalid selected-state adjoint input");
    }
    result.weights.push_back(state_weight);
    result.source_coefficients.push_back(
        selected_state.coefficient_matrix);
    if (state_weight == 0.0) {
      result.orthogonal_coefficients.emplace_back();
      result.residuals.emplace_back();
      continue;
    }
    Eigen::MatrixXd orthogonal_coefficients =
        result.source_coefficients.back();
    direct_ci_->alpha_transform.apply_left(&orthogonal_coefficients);
    direct_ci_->beta_transform().apply_right(&orthogonal_coefficients);
    const Eigen::MatrixXd sigma =
        direct_ci_->sigma.apply(orthogonal_coefficients);
    result.orthogonal_coefficients.push_back(
        std::move(orthogonal_coefficients));
    result.residuals.push_back(
        sigma - state_energies[state] *
            result.orthogonal_coefficients.back());
    const DirectCiIntegralAdjoint integral_gradient =
        direct_ci_->sigma.integral_adjoint(
            result.orthogonal_coefficients.back(),
            result.orthogonal_coefficients.back());
    result.one_electron_gradient.noalias() +=
        state_weight * integral_gradient.one_electron;
    result.pair_kernel_gradient.noalias() +=
        state_weight * integral_gradient.pair_kernel;
    result.generator_gradient.noalias() +=
        2.0 * state_weight *
        direct_ci_->sigma.one_body_generator_adjoint(
            result.residuals.back(),
            result.orthogonal_coefficients.back());
  }

  return result;
}

StructureActiveIntegralAdjoint StructureAction::active_integral_adjoint(
    const StructureAdjointState& state) const {
  if (!direct_ci_) {
    throw std::logic_error(
        "active-integral adjoint requires orthogonal direct CI");
  }
  NonorthogonalActiveIntegralAdjoint result =
      backpropagate_orthogonal_active_integral_adjoint(
          direct_ci_->integrals,
          state.one_electron_gradient,
          state.pair_kernel_gradient,
          state.generator_gradient);
  return StructureActiveIntegralAdjoint{
      std::move(result.overlap),
      std::move(result.one_electron),
      std::move(result.pair_kernel)};
}

StructureActiveIntegralAdjoint
StructureAction::active_integral_adjoint_direction(
    const StructureAdjointState& state,
    const SelectedStateDeterminantMatrices* directional_selected_states,
    const std::vector<double>* directional_state_energies,
    const std::vector<double>& overlap_direction,
    const std::vector<double>& one_electron_direction,
    const std::vector<double>& packed_two_electron_direction,
    const StructureIntegralDirection* direct_ci_direction,
    bool include_integral_response) const {
  if (!direct_ci_) {
    throw std::logic_error(
        "active-integral adjoint direction requires orthogonal direct CI");
  }
  const bool include_state_response = directional_selected_states != nullptr;
  const std::size_t n_states = state.weights.size();
  if (n_states == 0 || state.energies.size() != n_states ||
      state.source_coefficients.size() != n_states ||
      state.orthogonal_coefficients.size() != n_states ||
      state.residuals.size() != n_states ||
      include_state_response != (directional_state_energies != nullptr) ||
      (include_state_response &&
       (directional_selected_states->states.size() !=
            n_states ||
        directional_state_energies->size() != n_states ||
        directional_selected_states->n_unique_alpha != n_unique_alpha_ ||
        directional_selected_states->n_unique_beta != n_unique_beta_))) {
    throw std::invalid_argument(
        "selected-state adjoint-direction inputs have incompatible dimensions");
  }

  const int n_orbitals =
      static_cast<int>(direct_ci_->integrals.one_electron.rows());
  const int n_pairs = packed_active_pair_count(n_orbitals);
  if (include_integral_response && direct_ci_direction != nullptr &&
      (direct_ci_direction->coefficient_direction.rows() != n_unique_alpha_ ||
       direct_ci_direction->coefficient_direction.cols() !=
           static_cast<Eigen::Index>(n_states) * n_unique_beta_ ||
       direct_ci_direction->sigma_direction.rows() != n_unique_alpha_ ||
       direct_ci_direction->sigma_direction.cols() !=
           static_cast<Eigen::Index>(n_states) * n_unique_beta_)) {
    throw std::invalid_argument(
        "prepared direct-CI direction has incompatible dimensions");
  }
  OrthogonalActiveIntegrals integral_direction;
  if (include_integral_response) {
    if (one_electron_direction.size() !=
        static_cast<std::size_t>(n_orbitals) * n_orbitals) {
      throw std::invalid_argument(
          "active one-electron direction has incompatible dimensions");
    }
    const Eigen::Map<const Eigen::MatrixXd> delta_one_electron(
        one_electron_direction.data(), n_orbitals, n_orbitals);
    integral_direction = orthogonalize_active_integral_direction(
        direct_ci_->integrals,
        overlap_direction,
        delta_one_electron,
        packed_two_electron_direction,
        n_orbitals);
  } else {
    integral_direction.orbital_transform = Eigen::MatrixXd::Zero(
        n_orbitals, n_orbitals);
    integral_direction.one_electron = Eigen::MatrixXd::Zero(
        n_orbitals, n_orbitals);
    integral_direction.pair_kernel = Eigen::MatrixXd::Zero(n_pairs, n_pairs);
  }

  ExteriorTransformDirection alpha_transform_direction;
  ExteriorTransformDirection beta_transform_direction;
  std::unique_ptr<DirectCiSigmaAction> sigma_direction;
  if (include_integral_response && direct_ci_direction == nullptr) {
    alpha_transform_direction = direct_ci_->alpha_transform.direction(
        integral_direction.orbital_transform);
    beta_transform_direction = direct_ci_->beta_transform().direction(
        integral_direction.orbital_transform);
    sigma_direction = std::make_unique<DirectCiSigmaAction>(
        direct_ci_->alpha_determinants,
        direct_ci_->beta_determinants(),
        integral_direction);
  }

  Eigen::MatrixXd orthogonal_one_gradient_direction = Eigen::MatrixXd::Zero(
      n_orbitals, n_orbitals);
  Eigen::MatrixXd orthogonal_pair_gradient_direction = Eigen::MatrixXd::Zero(
      n_pairs, n_pairs);
  Eigen::MatrixXd generator_gradient_direction = Eigen::MatrixXd::Zero(
      n_orbitals, n_orbitals);

  for (std::size_t state_index = 0;
       state_index < n_states;
       ++state_index) {
    const double state_weight = state.weights[state_index];
    if (!std::isfinite(state_weight) || state_weight < 0.0 ||
        !std::isfinite(state.energies[state_index])) {
      throw std::invalid_argument("invalid selected-state adjoint input");
    }
    if (state_weight == 0.0) {
      continue;
    }
    const auto& orthogonal_coefficients =
        state.orthogonal_coefficients[state_index];
    const auto& residual = state.residuals[state_index];
    if (state.source_coefficients[state_index].rows() != n_unique_alpha_ ||
        state.source_coefficients[state_index].cols() != n_unique_beta_ ||
        orthogonal_coefficients.rows() != n_unique_alpha_ ||
        orthogonal_coefficients.cols() != n_unique_beta_ ||
        residual.rows() != n_unique_alpha_ ||
        residual.cols() != n_unique_beta_) {
      throw std::invalid_argument("invalid selected-state adjoint input");
    }

    Eigen::MatrixXd directional_orthogonal_coefficients =
        Eigen::MatrixXd::Zero(n_unique_alpha_, n_unique_beta_);
    if (include_integral_response) {
      if (direct_ci_direction != nullptr) {
        directional_orthogonal_coefficients =
            direct_ci_direction->coefficient_direction.middleCols(
                static_cast<Eigen::Index>(state_index) * n_unique_beta_,
                n_unique_beta_);
      } else {
        Eigen::MatrixXd transformed_source =
            state.source_coefficients[state_index];
        direct_ci_->alpha_transform.apply_directional_left(
            alpha_transform_direction,
            &transformed_source,
            &directional_orthogonal_coefficients);
        direct_ci_->beta_transform().apply_directional_right(
            beta_transform_direction,
            &transformed_source,
            &directional_orthogonal_coefficients);
      }
    }
    double directional_energy = 0.0;
    Eigen::MatrixXd orthogonal_state_response;
    if (include_state_response) {
      const auto& directional_state =
          directional_selected_states->states[state_index];
      if (directional_state.coefficient_matrix.rows() != n_unique_alpha_ ||
          directional_state.coefficient_matrix.cols() != n_unique_beta_ ||
          !directional_state.coefficient_matrix.allFinite() ||
          !std::isfinite((*directional_state_energies)[state_index])) {
        throw std::invalid_argument(
            "invalid directional selected-state adjoint input");
      }
      orthogonal_state_response = directional_state.coefficient_matrix;
      direct_ci_->alpha_transform.apply_left(&orthogonal_state_response);
      direct_ci_->beta_transform().apply_right(&orthogonal_state_response);
      directional_orthogonal_coefficients += orthogonal_state_response;
      directional_energy = (*directional_state_energies)[state_index];
    }

    Eigen::MatrixXd directional_sigma;
    if (include_integral_response && direct_ci_direction != nullptr) {
      directional_sigma = direct_ci_direction->sigma_direction.middleCols(
          static_cast<Eigen::Index>(state_index) * n_unique_beta_,
          n_unique_beta_);
    } else {
      directional_sigma = direct_ci_->sigma.apply(
          directional_orthogonal_coefficients);
    }
    if (include_integral_response && direct_ci_direction == nullptr) {
      directional_sigma.noalias() +=
          sigma_direction->apply(orthogonal_coefficients);
    }
    if (include_state_response && include_integral_response &&
        direct_ci_direction != nullptr) {
      directional_sigma.noalias() +=
          direct_ci_->sigma.apply(orthogonal_state_response);
    }
    const Eigen::MatrixXd directional_residual =
        directional_sigma -
        state.energies[state_index] * directional_orthogonal_coefficients -
        directional_energy * orthogonal_coefficients;

    const DirectCiIntegralAdjoint directional_integral_gradient =
        direct_ci_->sigma.integral_adjoint(
            directional_orthogonal_coefficients,
            orthogonal_coefficients);
    orthogonal_one_gradient_direction.noalias() +=
        2.0 * state_weight * directional_integral_gradient.one_electron;
    orthogonal_pair_gradient_direction.noalias() +=
        2.0 * state_weight * directional_integral_gradient.pair_kernel;
    generator_gradient_direction.noalias() += 2.0 * state_weight *
        (direct_ci_->sigma.one_body_generator_adjoint(
             directional_residual,
             orthogonal_coefficients) +
         direct_ci_->sigma.one_body_generator_adjoint(
             residual,
             directional_orthogonal_coefficients));
  }

  NonorthogonalActiveIntegralAdjoint result =
      backpropagate_orthogonal_active_integral_adjoint_direction(
          direct_ci_->integrals,
          integral_direction,
          state.one_electron_gradient,
          state.pair_kernel_gradient,
          state.generator_gradient,
          orthogonal_one_gradient_direction,
          orthogonal_pair_gradient_direction,
          generator_gradient_direction);
  return StructureActiveIntegralAdjoint{
      std::move(result.overlap),
      std::move(result.one_electron),
      std::move(result.pair_kernel)};
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
  result.orthogonal_direct_ci = direct_ci_ != nullptr;
  if (direct_ci_) {
    result.direct_ci_bytes = direct_ci_->dynamic_bytes();
  }
  result.channel_nonzeros = channel_nonzeros_;
  result.channel_dense_values = channel_dense_values_;
  result.expansion_bytes =
      spin_term_offsets_.size() * sizeof(std::size_t) +
      spin_terms_.size() * sizeof(StructureTerm);
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
