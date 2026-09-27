#include "vbscf/structures/assembly/action.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>

#include "core/openmp.hpp"
#include "vbscf/determinants/pairs/accepted_action.hpp"
#include "vbscf/determinants/pairs/storage.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/tile_policy_internal.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"
#include "vbscf/structures/orthogonal_ci/integrals.hpp"
#include "vbscf/structures/orthogonal_ci/planner.hpp"
#include "vbscf/structures/orthogonal_ci/sigma.hpp"

namespace xmvb::vb {
namespace {

constexpr std::size_t kStructureActionWorkspaceBytes =
    256ULL * 1024ULL * 1024ULL;
constexpr int kMaximumStructureActionBatchColumns = 8;

int bounded_structure_action_width(
    int requested,
    int n_unique_alpha,
    int n_unique_beta,
    bool direct_ci) {
  // A direct-CI column owns determinant-product sigma and exterior-transform
  // workspaces. Keeping its width at one prevents the FCI workspace from being
  // multiplied by the Davidson block width. The factorized action can batch,
  // but bounds its simultaneous spin-product matrices by the same byte budget.
  if (direct_ci) {
    return 1;
  }
  const std::size_t spin_products =
      static_cast<std::size_t>(n_unique_alpha) *
      static_cast<std::size_t>(n_unique_beta);
  constexpr std::size_t kSimultaneousSpinProductMatrices = 8;
  const std::size_t bytes_per_column = std::max<std::size_t>(
      sizeof(double),
      kSimultaneousSpinProductMatrices * spin_products * sizeof(double));
  const int budget_width = static_cast<int>(std::max<std::size_t>(
      1ULL, kStructureActionWorkspaceBytes / bytes_per_column));
  return std::max(
      1,
      std::min({requested,
                kMaximumStructureActionBatchColumns,
                budget_width}));
}

Eigen::MatrixXd build_pair_channel_block(
    const AcceptedSpinPairTile& tile,
    int n_packed_pairs) {
  Eigen::MatrixXd channels = Eigen::MatrixXd::Zero(
      tile.left_size * tile.right_size, n_packed_pairs);
  for (int left = 0; left < tile.left_size; ++left) {
    for (int right = 0; right < tile.right_size; ++right) {
      const int work = left + tile.left_size * right;
      const auto& projection = tile.pair(left, right)
          .opposite_spin_pair_cache.first_order_cofactor_projection;
      for (std::size_t entry = 0;
           entry < projection.packed_pair_indices.size();
           ++entry) {
        channels(work, projection.packed_pair_indices[entry]) +=
            projection.packed_pair_values[entry];
      }
    }
  }
  return channels;
}

struct PairScalarTile {
  Eigen::MatrixXd overlap;
  Eigen::MatrixXd hamiltonian;
};

PairScalarTile build_pair_scalar_tile(const AcceptedSpinPairTile& tile) {
  PairScalarTile result{
      Eigen::MatrixXd(tile.left_size, tile.right_size),
      Eigen::MatrixXd(tile.left_size, tile.right_size)};
  for (int left = 0; left < tile.left_size; ++left) {
    for (int right = 0; right < tile.right_size; ++right) {
      const auto& pair = tile.pair(left, right);
      result.overlap(left, right) =
          pair.overlap_result.overlap_determinant;
      result.hamiltonian(left, right) = pair.total_hamiltonian;
    }
  }
  return result;
}

void add_opposite_spin_tile(
    const AcceptedSpinPairTile& alpha_tile,
    const AcceptedSpinPairTile& beta_tile,
    const Eigen::Ref<const Eigen::MatrixXd>& alpha_channels,
    const Eigen::Ref<const Eigen::MatrixXd>& beta_projected,
    const Eigen::Ref<const Eigen::MatrixXd>& source,
    Eigen::Ref<Eigen::MatrixXd> target) {
  if (source.rows() != alpha_tile.right_size ||
      source.cols() != beta_tile.right_size ||
      target.rows() != alpha_tile.left_size ||
      target.cols() != beta_tile.left_size ||
      alpha_channels.rows() != alpha_tile.left_size * alpha_tile.right_size ||
      beta_projected.rows() != beta_tile.left_size * beta_tile.right_size ||
      alpha_channels.cols() != beta_projected.cols()) {
    throw std::invalid_argument(
        "opposite-spin tile action dimensions are inconsistent");
  }
  using ChannelMap = Eigen::Map<
      const Eigen::MatrixXd,
      Eigen::Unaligned,
      Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
  const Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> alpha_stride(
      alpha_tile.left_size, 1);
  const Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> beta_stride(
      beta_tile.left_size, 1);
  Eigen::MatrixXd push(alpha_tile.left_size, beta_tile.right_size);
  for (int channel = 0; channel < alpha_channels.cols(); ++channel) {
    if (alpha_channels.col(channel).isZero(0.0) ||
        beta_projected.col(channel).isZero(0.0)) {
      continue;
    }
    const ChannelMap alpha_channel(
        alpha_channels.col(channel).data(),
        alpha_tile.left_size,
        alpha_tile.right_size,
        alpha_stride);
    const ChannelMap beta_channel(
        beta_projected.col(channel).data(),
        beta_tile.left_size,
        beta_tile.right_size,
        beta_stride);
    push.noalias() = alpha_channel * source;
    target.noalias() += push * beta_channel.transpose();
  }
}

bool full_factorized_action_fits(
    int n_unique_alpha,
    int n_unique_beta,
    int block_width) {
  constexpr std::size_t kSimultaneousSpinProductMatrices = 8;
  const std::size_t columns = static_cast<std::size_t>(block_width);
  const std::size_t spin_products =
      static_cast<std::size_t>(n_unique_alpha) * n_unique_beta;
  const std::size_t maximum_products = kStructureActionWorkspaceBytes /
      (kSimultaneousSpinProductMatrices * sizeof(double) * columns);
  return spin_products <= maximum_products;
}

void add_tiled_opposite_spin_action(
    const AcceptedPairTileProvider& alpha_provider,
    const AcceptedPairTileProvider& beta_provider,
    const std::vector<std::vector<int>>& alpha_strings,
    const std::vector<std::vector<int>>& beta_strings,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e,
    const ActiveSpaceTwoElectronResult& two_electron,
    int n_active,
    int n_unique_alpha,
    int n_unique_beta,
    const Eigen::Ref<const Eigen::MatrixXd>& spin_vectors,
    Eigen::MatrixXd* spin_hamiltonians) {
  const int block_width =
      static_cast<int>(spin_vectors.cols()) / n_unique_beta;
  const int n_pairs = packed_active_pair_count(n_active);
  const int alpha_electrons = alpha_strings.empty()
      ? 0
      : static_cast<int>(alpha_strings.front().size());
  const int beta_electrons = beta_strings.empty()
      ? 0
      : static_cast<int>(beta_strings.front().size());
  const bool direct_ri = uses_direct_ri_pair_factors(two_electron);
  const int alpha_extent = detail::plan_pair_tile_extent(
      n_unique_alpha,
      alpha_electrons,
      n_active,
      two_electron.n_auxiliary_functions,
      true,
      detail::kPairTileWorkspaceBytes,
      0,
      0,
      0,
      direct_ri);
  const int beta_extent = detail::plan_pair_tile_extent(
      n_unique_beta,
      beta_electrons,
      n_active,
      two_electron.n_auxiliary_functions,
      true,
      detail::kPairTileWorkspaceBytes,
      0,
      0,
      0,
      direct_ri);
  for (int alpha_left = 0;
       alpha_left < n_unique_alpha;
       alpha_left += alpha_extent) {
    const int alpha_left_end =
        std::min(n_unique_alpha, alpha_left + alpha_extent);
    for (int alpha_right = 0;
         alpha_right < n_unique_alpha;
         alpha_right += alpha_extent) {
      const int alpha_right_end =
          std::min(n_unique_alpha, alpha_right + alpha_extent);
      const AcceptedSpinPairTile alpha_tile = alpha_provider.build(
          alpha_left,
          alpha_left_end,
          alpha_right,
          alpha_right_end,
          active_overlap,
          h1e,
          two_electron,
          AcceptedPairTileBuildOptions{
              .materialize_projected_pair_values = false,
              .populate_response_payload = false,
              .populate_opposite_spin_projection = true});
      const Eigen::MatrixXd alpha_channels =
          build_pair_channel_block(alpha_tile, n_pairs);

      for (int beta_left = 0;
           beta_left < n_unique_beta;
           beta_left += beta_extent) {
        const int beta_left_end =
            std::min(n_unique_beta, beta_left + beta_extent);
        for (int beta_right = 0;
             beta_right < n_unique_beta;
             beta_right += beta_extent) {
          const int beta_right_end =
              std::min(n_unique_beta, beta_right + beta_extent);
          const AcceptedSpinPairTile beta_tile = beta_provider.build(
              beta_left,
              beta_left_end,
              beta_right,
              beta_right_end,
              active_overlap,
              h1e,
              two_electron,
              AcceptedPairTileBuildOptions{
                  .materialize_projected_pair_values = false,
                  .populate_response_payload = false,
                  .populate_opposite_spin_projection = true});
          const Eigen::MatrixXd beta_raw =
              build_pair_channel_block(beta_tile, n_pairs);
          const Eigen::MatrixXd beta_projected =
              apply_active_space_two_electron_kernel_block(
                  two_electron, n_active, beta_raw.transpose()).transpose();

          for (int vector = 0; vector < block_width; ++vector) {
            const auto source = spin_vectors.block(
                alpha_right,
                vector * n_unique_beta + beta_right,
                alpha_tile.right_size,
                beta_tile.right_size);
            auto target = spin_hamiltonians->block(
                alpha_left,
                vector * n_unique_beta + beta_left,
                alpha_tile.left_size,
                beta_tile.left_size);
            add_opposite_spin_tile(
                alpha_tile,
                beta_tile,
                alpha_channels,
                beta_projected,
                source,
                target);
          }
        }
      }
    }
  }
}

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

struct StructureAction::TiledPairData {
  std::shared_ptr<const AcceptedPairTileProvider> alpha_provider;
  std::shared_ptr<const AcceptedPairTileProvider> beta_provider;
  std::vector<std::vector<int>> alpha_strings;
  std::vector<std::vector<int>> beta_strings;
  std::vector<double> active_overlap;
  Eigen::MatrixXd one_electron;
  ActiveSpaceTwoElectronResult two_electron;
  int n_active = 0;

  TiledPairData(
      const SameSpinPairCacheContext& cache,
      const std::vector<double>& overlap,
      const Eigen::Ref<const Eigen::MatrixXd>& h1e,
      const ActiveSpaceTwoElectronResult& active_two_electron,
      int n_active_orbitals)
      : alpha_provider(cache.alpha_pair_provider),
        beta_provider(cache.beta_pair_provider),
        alpha_strings(cache.alpha_reuse_table.unique_determinants),
        beta_strings(cache.beta_reuse_table.unique_determinants),
        active_overlap(overlap),
        one_electron(h1e),
        two_electron(active_two_electron),
        n_active(n_active_orbitals) {
    if (!alpha_provider || !beta_provider) {
      throw std::invalid_argument(
          "tiled structure action requires accepted pair providers");
    }
  }

  std::size_t dynamic_bytes() const noexcept {
    std::size_t bytes = active_overlap.capacity() * sizeof(double) +
        static_cast<std::size_t>(one_electron.size() +
                                 two_electron.ri_active_pair_factors.size()) *
            sizeof(double) +
        two_electron.packed_active_two_electron_integrals.capacity() *
            sizeof(double);
    const auto add_strings = [&bytes](const auto& spin_strings) {
      bytes += spin_strings.capacity() * sizeof(std::vector<int>);
      for (const auto& string : spin_strings) {
        bytes += string.capacity() * sizeof(int);
      }
    };
    add_strings(alpha_strings);
    add_strings(beta_strings);
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
    int n_active_orbitals,
    const StructureDiagonal* precomputed_preconditioner)
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

  const DirectCiActionPlan direct_ci_plan =
      plan_orthogonal_direct_ci_action(
          same_spin_pair_cache.alpha_reuse_table.unique_determinants,
          same_spin_pair_cache.beta_reuse_table.unique_determinants,
          n_active_orbitals,
          1);
  const bool use_direct_ci = direct_ci_plan.favors_direct_ci() ||
      (!same_spin_pair_cache.enabled() && direct_ci_plan.complete());
  if (!same_spin_pair_cache.has_pair_providers()) {
    throw std::invalid_argument(
        "structure action requires accepted-pair providers");
  }
  if (!same_spin_pair_cache.enabled() &&
      precomputed_preconditioner == nullptr) {
    throw std::invalid_argument(
        "topology-only structure action requires a diagonal preconditioner");
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
  if (same_spin_pair_cache.enabled()) {
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
  }

  if (use_direct_ci) {
    direct_ci_ = std::make_unique<OrthogonalDirectCiData>(
        same_spin_pair_cache.alpha_reuse_table.unique_determinants,
        same_spin_pair_cache.beta_reuse_table.unique_determinants,
        active_overlap,
        active_one_electron,
        active_space_two_electron_result,
        n_active_orbitals);
  } else {
    tiled_pairs_ = std::make_unique<TiledPairData>(
        same_spin_pair_cache,
        active_overlap,
        active_one_electron,
        active_space_two_electron_result,
        n_active_orbitals);
  }

  struct PendingSpinTerm {
    int spin_product = 0;
    StructureTerm term;
  };
  std::vector<PendingSpinTerm> pending_terms;
  std::size_t n_expansion_terms = 0;
  for (const auto& determinant_terms : determinant_to_structure_terms) {
    n_expansion_terms += determinant_terms.size();
  }
  pending_terms.reserve(n_expansion_terms);
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
      pending_terms.push_back(PendingSpinTerm{
          spin_product,
          StructureTerm{term.structure_index, term.coefficient}});
    }
  }

  std::sort(
      pending_terms.begin(),
      pending_terms.end(),
      [](const PendingSpinTerm& left, const PendingSpinTerm& right) {
        if (left.spin_product != right.spin_product) {
          return left.spin_product < right.spin_product;
        }
        return left.term.structure < right.term.structure;
      });
  spin_products_.reserve(std::min(
      pending_terms.size(),
      static_cast<std::size_t>(n_determinants_)));
  spin_terms_.reserve(pending_terms.size());
  spin_term_offsets_.reserve(spin_products_.capacity() + 1);
  spin_term_offsets_.push_back(0);
  std::size_t pending = 0;
  while (pending < pending_terms.size()) {
    const int spin_product = pending_terms[pending].spin_product;
    spin_products_.push_back(spin_product);
    while (pending < pending_terms.size() &&
           pending_terms[pending].spin_product == spin_product) {
      const int structure = pending_terms[pending].term.structure;
      double coefficient = 0.0;
      while (pending < pending_terms.size() &&
             pending_terms[pending].spin_product == spin_product &&
             pending_terms[pending].term.structure == structure) {
        coefficient += pending_terms[pending].term.coefficient;
        ++pending;
      }
      if (coefficient != 0.0) {
        spin_terms_.push_back(StructureTerm{structure, coefficient});
      }
    }
    spin_term_offsets_.push_back(spin_terms_.size());
  }

  struct StructureSpinTerm {
    int spin_product = 0;
    double coefficient = 0.0;
  };
  std::vector<std::vector<StructureSpinTerm>> terms_by_structure(
      n_structures_);
  for (std::size_t group = 0; group < spin_products_.size(); ++group) {
    const int spin_product = spin_products_[group];
    for (std::size_t term_index = spin_term_offsets_[group];
         term_index < spin_term_offsets_[group + 1];
         ++term_index) {
      const StructureTerm& term = spin_terms_[term_index];
      terms_by_structure[term.structure].push_back(
          StructureSpinTerm{spin_product, term.coefficient});
    }
  }

  if (precomputed_preconditioner != nullptr) {
    if (precomputed_preconditioner->hamiltonian.size() != n_structures_ ||
        precomputed_preconditioner->overlap.size() != n_structures_) {
      throw std::invalid_argument(
          "precomputed diagonal preconditioner has incompatible dimensions");
    }
    preconditioner_diagonal_ = *precomputed_preconditioner;
    return;
  }

  preconditioner_diagonal_.hamiltonian =
      Eigen::VectorXd::Zero(n_structures_);
  preconditioner_diagonal_.overlap = Eigen::VectorXd::Zero(n_structures_);
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
    preconditioner_diagonal_.hamiltonian[structure] = hamiltonian;
    preconditioner_diagonal_.overlap[structure] = overlap;
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
  for (std::size_t group = 0; group < spin_products_.size(); ++group) {
    const int spin_product = spin_products_[group];
    const int alpha = spin_product / n_unique_beta_;
    const int beta = spin_product % n_unique_beta_;
    for (std::size_t term_index = spin_term_offsets_[group];
         term_index < spin_term_offsets_[group + 1];
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

void StructureAction::add_spin_product_tile(
    const Eigen::Ref<const Eigen::MatrixXd>& tile_images,
    int alpha_begin,
    int beta_begin,
    Eigen::MatrixXd* structure_images) const {
  if (structure_images == nullptr || tile_images.rows() <= 0 ||
      tile_images.cols() <= 0 || alpha_begin < 0 || beta_begin < 0 ||
      alpha_begin + tile_images.rows() > n_unique_alpha_ ||
      structure_images->rows() != n_structures_ ||
      structure_images->cols() <= 0 ||
      tile_images.cols() % structure_images->cols() != 0) {
    throw std::invalid_argument(
        "unique-string-product tile has incompatible dimensions");
  }
  const int block_width = static_cast<int>(structure_images->cols());
  const int beta_size = static_cast<int>(tile_images.cols()) / block_width;
  if (beta_begin + beta_size > n_unique_beta_) {
    throw std::invalid_argument(
        "unique-string-product tile exceeds the beta space");
  }

  for (int alpha_local = 0; alpha_local < tile_images.rows(); ++alpha_local) {
    const int alpha = alpha_begin + alpha_local;
    const int first_product = alpha * n_unique_beta_ + beta_begin;
    const int last_product = first_product + beta_size;
    auto group = std::lower_bound(
        spin_products_.begin(), spin_products_.end(), first_product);
    while (group != spin_products_.end() && *group < last_product) {
      const std::size_t group_index =
          static_cast<std::size_t>(group - spin_products_.begin());
      const int beta_local = *group - first_product;
      for (std::size_t term_index = spin_term_offsets_[group_index];
           term_index < spin_term_offsets_[group_index + 1];
           ++term_index) {
        const StructureTerm& term = spin_terms_[term_index];
        for (int vector = 0; vector < block_width; ++vector) {
          (*structure_images)(term.structure, vector) +=
              term.coefficient *
              tile_images(alpha_local, vector * beta_size + beta_local);
        }
      }
      ++group;
    }
  }
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
  for (std::size_t group = 0; group < spin_products_.size(); ++group) {
    const int spin_product = spin_products_[group];
    const int alpha = spin_product / n_unique_beta_;
    const int beta = spin_product % n_unique_beta_;
    for (std::size_t term_index = spin_term_offsets_[group];
         term_index < spin_term_offsets_[group + 1];
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

Eigen::MatrixXd StructureAction::expand_structure_tile(
    const Eigen::Ref<const Eigen::MatrixXd>& vectors,
    int alpha_begin,
    int alpha_size,
    int beta_begin,
    int beta_size) const {
  if (vectors.rows() != n_structures_ || vectors.cols() <= 0 ||
      alpha_begin < 0 || alpha_size <= 0 ||
      alpha_begin + alpha_size > n_unique_alpha_ ||
      beta_begin < 0 || beta_size <= 0 ||
      beta_begin + beta_size > n_unique_beta_) {
    throw std::invalid_argument(
        "structure expansion tile dimensions are inconsistent");
  }
  const int block_width = static_cast<int>(vectors.cols());
  Eigen::MatrixXd tile = Eigen::MatrixXd::Zero(
      alpha_size, block_width * beta_size);
  for (int alpha_local = 0; alpha_local < alpha_size; ++alpha_local) {
    const int alpha = alpha_begin + alpha_local;
    const int first_product = alpha * n_unique_beta_ + beta_begin;
    const int last_product = first_product + beta_size;
    auto group = std::lower_bound(
        spin_products_.begin(), spin_products_.end(), first_product);
    while (group != spin_products_.end() && *group < last_product) {
      const std::size_t group_index =
          static_cast<std::size_t>(group - spin_products_.begin());
      const int beta_local = *group - first_product;
      for (std::size_t term_index = spin_term_offsets_[group_index];
           term_index < spin_term_offsets_[group_index + 1];
           ++term_index) {
        const StructureTerm& term = spin_terms_[term_index];
        for (int vector = 0; vector < block_width; ++vector) {
          tile(alpha_local, vector * beta_size + beta_local) +=
              term.coefficient * vectors(term.structure, vector);
        }
      }
      ++group;
    }
  }
  return tile;
}

StructureActionResult StructureAction::apply_streamed(
    const Eigen::Ref<const Eigen::MatrixXd>& vectors) const {
  if (!tiled_pairs_) {
    throw std::logic_error("tiled structure action has no pair providers");
  }
  const int block_width = static_cast<int>(vectors.cols());
  const int n_pairs = packed_active_pair_count(tiled_pairs_->n_active);
  const int alpha_electrons = tiled_pairs_->alpha_strings.empty()
      ? 0
      : static_cast<int>(tiled_pairs_->alpha_strings.front().size());
  const int beta_electrons = tiled_pairs_->beta_strings.empty()
      ? 0
      : static_cast<int>(tiled_pairs_->beta_strings.front().size());
  const std::size_t spin_workspace = kStructureActionWorkspaceBytes / 2;
  const bool direct_ri =
      uses_direct_ri_pair_factors(tiled_pairs_->two_electron);
  const int alpha_extent = detail::plan_pair_tile_extent(
      n_unique_alpha_,
      alpha_electrons,
      tiled_pairs_->n_active,
      tiled_pairs_->two_electron.n_auxiliary_functions,
      true,
      spin_workspace,
      0,
      0,
      0,
      direct_ri);
  const int beta_extent = detail::plan_pair_tile_extent(
      n_unique_beta_,
      beta_electrons,
      tiled_pairs_->n_active,
      tiled_pairs_->two_electron.n_auxiliary_functions,
      true,
      spin_workspace,
      0,
      0,
      0,
      direct_ri);

  StructureActionResult result{
      Eigen::MatrixXd::Zero(n_structures_, block_width),
      Eigen::MatrixXd::Zero(n_structures_, block_width)};
  for (int alpha_left = 0; alpha_left < n_unique_alpha_;
       alpha_left += alpha_extent) {
    const int alpha_left_end =
        std::min(n_unique_alpha_, alpha_left + alpha_extent);
    const int alpha_left_size = alpha_left_end - alpha_left;
    for (int beta_left = 0; beta_left < n_unique_beta_;
         beta_left += beta_extent) {
      const int beta_left_end =
          std::min(n_unique_beta_, beta_left + beta_extent);
      const int beta_left_size = beta_left_end - beta_left;
      Eigen::MatrixXd hamiltonian_tile = Eigen::MatrixXd::Zero(
          alpha_left_size, block_width * beta_left_size);
      Eigen::MatrixXd overlap_tile = Eigen::MatrixXd::Zero(
          alpha_left_size, block_width * beta_left_size);

      for (int alpha_right = 0; alpha_right < n_unique_alpha_;
           alpha_right += alpha_extent) {
        const int alpha_right_end =
            std::min(n_unique_alpha_, alpha_right + alpha_extent);
        const AcceptedSpinPairTile alpha_pair =
            tiled_pairs_->alpha_provider->build(
                alpha_left,
                alpha_left_end,
                alpha_right,
                alpha_right_end,
                tiled_pairs_->active_overlap,
                tiled_pairs_->one_electron,
                tiled_pairs_->two_electron,
                AcceptedPairTileBuildOptions{
                    .materialize_projected_pair_values = false,
                    .populate_response_payload = false,
                    .populate_opposite_spin_projection = true});
        const PairScalarTile alpha_scalar =
            build_pair_scalar_tile(alpha_pair);
        const Eigen::MatrixXd alpha_channels =
            build_pair_channel_block(alpha_pair, n_pairs);

        for (int beta_right = 0; beta_right < n_unique_beta_;
             beta_right += beta_extent) {
          const int beta_right_end =
              std::min(n_unique_beta_, beta_right + beta_extent);
          const AcceptedSpinPairTile beta_pair =
              tiled_pairs_->beta_provider->build(
                  beta_left,
                  beta_left_end,
                  beta_right,
                  beta_right_end,
                  tiled_pairs_->active_overlap,
                  tiled_pairs_->one_electron,
                  tiled_pairs_->two_electron,
                  AcceptedPairTileBuildOptions{
                      .materialize_projected_pair_values = false,
                      .populate_response_payload = false,
                      .populate_opposite_spin_projection = true});
          const PairScalarTile beta_scalar =
              build_pair_scalar_tile(beta_pair);
          const Eigen::MatrixXd beta_raw =
              build_pair_channel_block(beta_pair, n_pairs);
          const Eigen::MatrixXd beta_projected =
              apply_active_space_two_electron_kernel_block(
                  tiled_pairs_->two_electron,
                  tiled_pairs_->n_active,
                  beta_raw.transpose()).transpose();
          const Eigen::MatrixXd source_tile = expand_structure_tile(
              vectors,
              alpha_right,
              alpha_pair.right_size,
              beta_right,
              beta_pair.right_size);

          for (int vector = 0; vector < block_width; ++vector) {
            const auto source = source_tile.middleCols(
                vector * beta_pair.right_size,
                beta_pair.right_size);
            auto target_hamiltonian = hamiltonian_tile.middleCols(
                vector * beta_left_size, beta_left_size);
            auto target_overlap = overlap_tile.middleCols(
                vector * beta_left_size, beta_left_size);
            Eigen::MatrixXd alpha_push =
                alpha_scalar.overlap * source;
            target_overlap.noalias() +=
                alpha_push * beta_scalar.overlap.transpose();
            target_hamiltonian.noalias() +=
                alpha_push * beta_scalar.hamiltonian.transpose();
            alpha_push.noalias() = alpha_scalar.hamiltonian * source;
            target_hamiltonian.noalias() +=
                alpha_push * beta_scalar.overlap.transpose();
            add_opposite_spin_tile(
                alpha_pair,
                beta_pair,
                alpha_channels,
                beta_projected,
                source,
                target_hamiltonian);
          }
        }
      }
      add_spin_product_tile(
          hamiltonian_tile,
          alpha_left,
          beta_left,
          &result.hamiltonian);
      add_spin_product_tile(
          overlap_tile,
          alpha_left,
          beta_left,
          &result.overlap);
    }
  }
  return result;
}

Eigen::MatrixXd StructureAction::orthogonalize_structure_block(
    const Eigen::Ref<const Eigen::MatrixXd>& vectors) const {
  if (!direct_ci_) {
    throw std::logic_error(
        "orthogonal structure coefficients require direct CI");
  }
  Eigen::MatrixXd orthogonal = expand_structure_block(vectors);
  direct_ci_->alpha_transform.apply_left(&orthogonal);
  const int block_width = static_cast<int>(vectors.cols());
  for (int block = 0; block < block_width; ++block) {
    auto coefficient_block = orthogonal.middleCols(
        block * n_unique_beta_, n_unique_beta_);
    direct_ci_->beta_transform().apply_right_block(coefficient_block);
  }
  return orthogonal;
}

StructureActionResult StructureAction::apply(
    const Eigen::Ref<const Eigen::MatrixXd>& vectors) const {
  if (vectors.rows() != n_structures_ || vectors.cols() <= 0) {
    throw std::invalid_argument(
        "structure vector block has incompatible dimensions");
  }
  const int block_width = static_cast<int>(vectors.cols());
  const int bounded_width = bounded_structure_action_width(
      block_width, n_unique_alpha_, n_unique_beta_, direct_ci_ != nullptr);
  if (block_width > bounded_width) {
    StructureActionResult images{
        Eigen::MatrixXd(n_structures_, block_width),
        Eigen::MatrixXd(n_structures_, block_width)};
    for (int first = 0; first < block_width; first += bounded_width) {
      const int width = std::min(bounded_width, block_width - first);
      StructureActionResult chunk =
          apply(vectors.middleCols(first, width));
      images.hamiltonian.middleCols(first, width) = chunk.hamiltonian;
      images.overlap.middleCols(first, width) = chunk.overlap;
    }
    return images;
  }

  if (!direct_ci_ && !full_factorized_action_fits(
                         n_unique_alpha_, n_unique_beta_, block_width)) {
    return apply_streamed(vectors);
  }

  Eigen::MatrixXd spin_vectors = expand_structure_block(vectors);
  if (direct_ci_) {
    direct_ci_->alpha_transform.apply_left(&spin_vectors);
    for (int block = 0; block < block_width; ++block) {
      auto coefficient_block = spin_vectors.middleCols(
          block * n_unique_beta_, n_unique_beta_);
      direct_ci_->beta_transform().apply_right_block(coefficient_block);
    }

    Eigen::MatrixXd spin_hamiltonians = direct_ci_->sigma.apply(spin_vectors);
    Eigen::MatrixXd spin_overlaps = std::move(spin_vectors);
    for (int block = 0; block < block_width; ++block) {
      auto hamiltonian_block = spin_hamiltonians.middleCols(
          block * n_unique_beta_, n_unique_beta_);
      auto overlap_block = spin_overlaps.middleCols(
          block * n_unique_beta_, n_unique_beta_);
      direct_ci_->beta_transform().apply_adjoint_right_block(
          hamiltonian_block);
      direct_ci_->beta_transform().apply_adjoint_right_block(overlap_block);
    }
    direct_ci_->alpha_transform.apply_adjoint_left(&spin_hamiltonians);
    direct_ci_->alpha_transform.apply_adjoint_left(&spin_overlaps);
    return StructureActionResult{
        contract_spin_product_block(spin_hamiltonians),
        contract_spin_product_block(spin_overlaps)};
  }

  if (!tiled_pairs_) {
    throw std::logic_error("structure action has no retained operator data");
  }
  const Eigen::MatrixXd transposed_spin_vectors = transpose_matrix_blocks(
      spin_vectors, n_unique_alpha_, n_unique_beta_);
  const AcceptedSpinPairActionResult beta_images =
      apply_accepted_spin_pair_action(
          *tiled_pairs_->beta_provider,
          tiled_pairs_->active_overlap,
          tiled_pairs_->one_electron,
          tiled_pairs_->two_electron,
          transposed_spin_vectors,
          kStructureActionWorkspaceBytes);
  const Eigen::MatrixXd right_overlap = transpose_matrix_blocks(
      beta_images.overlap, n_unique_beta_, n_unique_alpha_);
  const Eigen::MatrixXd right_hamiltonian = transpose_matrix_blocks(
      beta_images.hamiltonian, n_unique_beta_, n_unique_alpha_);
  Eigen::MatrixXd alpha_inputs(
      n_unique_alpha_, 2 * right_overlap.cols());
  alpha_inputs.leftCols(right_overlap.cols()) = right_overlap;
  alpha_inputs.rightCols(right_hamiltonian.cols()) = right_hamiltonian;
  const AcceptedSpinPairActionResult alpha_images =
      apply_accepted_spin_pair_action(
          *tiled_pairs_->alpha_provider,
          tiled_pairs_->active_overlap,
          tiled_pairs_->one_electron,
          tiled_pairs_->two_electron,
          alpha_inputs,
          kStructureActionWorkspaceBytes);
  Eigen::MatrixXd spin_hamiltonians =
      alpha_images.hamiltonian.leftCols(right_overlap.cols());
  spin_hamiltonians.noalias() +=
      alpha_images.overlap.rightCols(right_hamiltonian.cols());
  Eigen::MatrixXd spin_overlaps =
      alpha_images.overlap.leftCols(right_overlap.cols());
  add_tiled_opposite_spin_action(
      *tiled_pairs_->alpha_provider,
      *tiled_pairs_->beta_provider,
      tiled_pairs_->alpha_strings,
      tiled_pairs_->beta_strings,
      tiled_pairs_->active_overlap,
      tiled_pairs_->one_electron,
      tiled_pairs_->two_electron,
      tiled_pairs_->n_active,
      n_unique_alpha_,
      n_unique_beta_,
      spin_vectors,
      &spin_hamiltonians);
  return StructureActionResult{
      contract_spin_product_block(spin_hamiltonians),
      contract_spin_product_block(spin_overlaps)};
}

StructureActionResult StructureAction::apply_integral_direction(
    const StructureAdjointState& state,
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
  const DirectCiSigmaAction sigma_direction(
      direct_ci_->alpha_determinants,
      direct_ci_->beta_determinants(),
      integral_direction);
  const Eigen::MatrixXd inverse_orbital_transform =
      direct_ci_->integrals.orbital_transform
          .template triangularView<Eigen::Upper>()
          .solve(Eigen::MatrixXd::Identity(
              n_active_orbitals,
              n_active_orbitals));
  const Eigen::MatrixXd relative_orbital_direction =
      integral_direction.orbital_transform * inverse_orbital_transform;

  const int block_width =
      static_cast<int>(state.source_coefficients.size());
  if (block_width <= 0 ||
      state.orthogonal_coefficients.size() !=
          static_cast<std::size_t>(block_width) ||
      state.residuals.size() != static_cast<std::size_t>(block_width) ||
      state.energies.size() != static_cast<std::size_t>(block_width)) {
    throw std::invalid_argument(
        "prepared direct-CI state does not match the structure-vector block");
  }
  Eigen::MatrixXd spin_vectors(
      n_unique_alpha_, block_width * n_unique_beta_);
  Eigen::MatrixXd spin_hamiltonians(
      n_unique_alpha_, block_width * n_unique_beta_);
  for (int block = 0; block < block_width; ++block) {
    const auto& orthogonal = state.orthogonal_coefficients[block];
    const auto& residual = state.residuals[block];
    if (orthogonal.rows() != n_unique_alpha_ ||
        orthogonal.cols() != n_unique_beta_ ||
        residual.rows() != n_unique_alpha_ ||
        residual.cols() != n_unique_beta_ ||
        !std::isfinite(state.energies[block])) {
      throw std::invalid_argument(
          "prepared direct-CI state has incompatible coefficient blocks");
    }
    spin_vectors.middleCols(
        block * n_unique_beta_, n_unique_beta_) = orthogonal;
    spin_hamiltonians.middleCols(
        block * n_unique_beta_, n_unique_beta_) =
        residual + state.energies[block] * orthogonal;
  }
  Eigen::MatrixXd delta_spin_vectors =
      direct_ci_->sigma.apply_one_body_generator(
          spin_vectors,
          relative_orbital_direction);

  Eigen::MatrixXd delta_spin_overlaps = delta_spin_vectors;
  Eigen::MatrixXd delta_spin_hamiltonians =
      direct_ci_->sigma.apply(delta_spin_vectors);
  delta_spin_hamiltonians += sigma_direction.apply(spin_vectors);
  if (direct_ci_direction != nullptr) {
    direct_ci_direction->coefficient_direction = delta_spin_vectors;
    direct_ci_direction->sigma_direction = delta_spin_hamiltonians;
  }
  const Eigen::MatrixXd transposed_relative_direction =
      relative_orbital_direction.transpose();
  delta_spin_hamiltonians +=
      direct_ci_->sigma.apply_one_body_generator(
          spin_hamiltonians,
          transposed_relative_direction);
  delta_spin_overlaps +=
      direct_ci_->sigma.apply_one_body_generator(
          spin_vectors,
          transposed_relative_direction);
  for (int block = 0; block < block_width; ++block) {
    Eigen::MatrixXd delta_hamiltonian_block =
        delta_spin_hamiltonians.middleCols(
            block * n_unique_beta_, n_unique_beta_);
    Eigen::MatrixXd delta_overlap_block =
        delta_spin_overlaps.middleCols(
            block * n_unique_beta_, n_unique_beta_);
    direct_ci_->beta_transform().apply_adjoint_right(
        &delta_hamiltonian_block);
    direct_ci_->beta_transform().apply_adjoint_right(
        &delta_overlap_block);
    delta_spin_hamiltonians.middleCols(
        block * n_unique_beta_, n_unique_beta_) = delta_hamiltonian_block;
    delta_spin_overlaps.middleCols(
        block * n_unique_beta_, n_unique_beta_) = delta_overlap_block;
  }
  direct_ci_->alpha_transform.apply_adjoint_left(
      &delta_spin_hamiltonians);
  direct_ci_->alpha_transform.apply_adjoint_left(
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
    const Eigen::MatrixXd* directional_state_multipliers,
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
      include_state_response != (directional_state_multipliers != nullptr) ||
      (include_state_response &&
       (directional_selected_states->states.size() !=
            n_states ||
        directional_state_multipliers->rows() !=
            static_cast<Eigen::Index>(n_states) ||
        directional_state_multipliers->cols() !=
            static_cast<Eigen::Index>(n_states) ||
        !directional_state_multipliers->allFinite() ||
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
    Eigen::MatrixXd orthogonal_state_response;
    if (include_state_response) {
      const auto& directional_state =
          directional_selected_states->states[state_index];
      if (directional_state.coefficient_matrix.rows() != n_unique_alpha_ ||
          directional_state.coefficient_matrix.cols() != n_unique_beta_ ||
          !directional_state.coefficient_matrix.allFinite()) {
        throw std::invalid_argument(
            "invalid directional selected-state adjoint input");
      }
      orthogonal_state_response = directional_state.coefficient_matrix;
      direct_ci_->alpha_transform.apply_left(&orthogonal_state_response);
      direct_ci_->beta_transform().apply_right(&orthogonal_state_response);
      directional_orthogonal_coefficients += orthogonal_state_response;
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
    Eigen::MatrixXd directional_residual =
        directional_sigma -
        state.energies[state_index] * directional_orthogonal_coefficients;
    if (include_state_response) {
      for (std::size_t coupled_state = 0;
           coupled_state < n_states;
           ++coupled_state) {
        const double multiplier = (*directional_state_multipliers)(
                static_cast<Eigen::Index>(coupled_state),
                static_cast<Eigen::Index>(state_index)) * 0.5 +
            (*directional_state_multipliers)(
                static_cast<Eigen::Index>(state_index),
                static_cast<Eigen::Index>(coupled_state)) *
                (0.5 * state.weights[coupled_state] / state_weight);
        if (multiplier != 0.0) {
          directional_residual.noalias() +=
              multiplier * state.orthogonal_coefficients[coupled_state];
        }
      }
    }

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

StructureActiveIntegralAdjoint
StructureAction::active_integral_response_adjoint(
    const StructureAdjointState& state,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const Eigen::Ref<const Eigen::MatrixXd>& state_multipliers) const {
  if (!direct_ci_) {
    throw std::logic_error(
        "active-integral response adjoint requires orthogonal direct CI");
  }
  const std::size_t n_states = state.weights.size();
  if (n_states == 0 || state.energies.size() != n_states ||
      state.source_coefficients.size() != n_states ||
      state.orthogonal_coefficients.size() != n_states ||
      state.residuals.size() != n_states ||
      directional_selected_states.states.size() != n_states ||
      directional_selected_states.n_unique_alpha != n_unique_alpha_ ||
      directional_selected_states.n_unique_beta != n_unique_beta_ ||
      state_multipliers.rows() != static_cast<Eigen::Index>(n_states) ||
      state_multipliers.cols() != static_cast<Eigen::Index>(n_states) ||
      !state_multipliers.allFinite()) {
    throw std::invalid_argument(
        "selected-subspace response adjoint inputs have incompatible dimensions");
  }

  const int n_orbitals =
      static_cast<int>(direct_ci_->integrals.one_electron.rows());
  const int n_pairs = packed_active_pair_count(n_orbitals);
  Eigen::MatrixXd one_gradient = Eigen::MatrixXd::Zero(
      n_orbitals, n_orbitals);
  Eigen::MatrixXd pair_gradient = Eigen::MatrixXd::Zero(n_pairs, n_pairs);
  Eigen::MatrixXd generator_gradient = Eigen::MatrixXd::Zero(
      n_orbitals, n_orbitals);

  std::vector<Eigen::MatrixXd> orthogonal_responses;
  orthogonal_responses.reserve(n_states);
  for (std::size_t state_index = 0;
       state_index < n_states;
       ++state_index) {
    const auto& directional_state =
        directional_selected_states.states[state_index];
    if (directional_state.coefficient_matrix.rows() != n_unique_alpha_ ||
        directional_state.coefficient_matrix.cols() != n_unique_beta_ ||
        !directional_state.coefficient_matrix.allFinite()) {
      throw std::invalid_argument(
          "invalid selected-subspace coefficient response");
    }
    Eigen::MatrixXd orthogonal_response =
        directional_state.coefficient_matrix;
    direct_ci_->alpha_transform.apply_left(&orthogonal_response);
    direct_ci_->beta_transform().apply_right(&orthogonal_response);
    orthogonal_responses.push_back(std::move(orthogonal_response));
  }

  for (std::size_t state_index = 0;
       state_index < n_states;
       ++state_index) {
    const double state_weight = state.weights[state_index];
    if (!std::isfinite(state_weight) || state_weight < 0.0 ||
        !std::isfinite(state.energies[state_index])) {
      throw std::invalid_argument("invalid selected-subspace adjoint state");
    }
    if (state_weight == 0.0) {
      continue;
    }
    const auto& accepted = state.orthogonal_coefficients[state_index];
    const auto& response = orthogonal_responses[state_index];
    if (accepted.rows() != n_unique_alpha_ ||
        accepted.cols() != n_unique_beta_ ||
        state.residuals[state_index].rows() != n_unique_alpha_ ||
        state.residuals[state_index].cols() != n_unique_beta_) {
      throw std::invalid_argument("invalid selected-subspace adjoint state");
    }

    const DirectCiIntegralAdjoint integral_gradient =
        direct_ci_->sigma.integral_adjoint(response, accepted);
    one_gradient.noalias() +=
        2.0 * state_weight * integral_gradient.one_electron;
    pair_gradient.noalias() +=
        2.0 * state_weight * integral_gradient.pair_kernel;

    Eigen::MatrixXd response_residual =
        direct_ci_->sigma.apply(response) -
        state.energies[state_index] * response;
    // Orbital overlap directions are symmetric, so only
    // sym(M W), W=diag(state weights), belongs to B^T.  Dividing its current
    // column by w_i lets the existing 2 w_i generator adjoint consume the
    // effective multiplier without retaining an antisymmetric gauge component.
    for (std::size_t coupled_state = 0;
         coupled_state < n_states;
         ++coupled_state) {
      const double multiplier = state_multipliers(
              static_cast<Eigen::Index>(coupled_state),
              static_cast<Eigen::Index>(state_index)) * 0.5 +
          state_multipliers(
              static_cast<Eigen::Index>(state_index),
              static_cast<Eigen::Index>(coupled_state)) *
              (0.5 * state.weights[coupled_state] / state_weight);
      if (multiplier != 0.0) {
        response_residual.noalias() +=
            multiplier * state.orthogonal_coefficients[coupled_state];
      }
    }
    generator_gradient.noalias() += 2.0 * state_weight *
        (direct_ci_->sigma.one_body_generator_adjoint(
             response_residual,
             accepted) +
         direct_ci_->sigma.one_body_generator_adjoint(
             state.residuals[state_index],
             response));
  }

  NonorthogonalActiveIntegralAdjoint result =
      backpropagate_orthogonal_active_integral_adjoint(
          direct_ci_->integrals,
          one_gradient,
          pair_gradient,
          generator_gradient);
  return StructureActiveIntegralAdjoint{
      std::move(result.overlap),
      std::move(result.one_electron),
      std::move(result.pair_kernel)};
}

const StructureDiagonal&
StructureAction::preconditioner_diagonal() const noexcept {
  return preconditioner_diagonal_;
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
  if (tiled_pairs_) {
    result.factor_bytes += tiled_pairs_->dynamic_bytes();
  }
  result.expansion_bytes =
      spin_products_.size() * sizeof(int) +
      spin_term_offsets_.size() * sizeof(std::size_t) +
      spin_terms_.size() * sizeof(StructureTerm);
  result.diagonal_bytes =
      static_cast<std::size_t>(
          preconditioner_diagonal_.hamiltonian.size() +
          preconditioner_diagonal_.overlap.size()) *
      sizeof(double);
  return result;
}

}  // namespace xmvb::vb
