#include "vbscf/structures/assembly/action.hpp"

#include "mcscf/ci/string_table.hpp"
#include "vbscf/structures/assembly/column_tiles.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>

#include "parallel/openmp.hpp"
#include "core/timing.hpp"
#include "vbscf/structures/assembly/sources.hpp"
#include "vbscf/determinants/pairs/accepted_action.hpp"
#include "vbscf/determinants/pairs/storage.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/tile_policy_internal.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"
#include "vbscf/structures/orthogonal_ci/integrals.hpp"
#include "integrals/active/raw.hpp"
#include "mcscf/ci/planner.hpp"
#include "mcscf/ci/product_support.hpp"
#include "mcscf/ci/sigma.hpp"
#include "mcscf/ci/factors.hpp"

namespace xmvb::vb {
namespace {

constexpr std::size_t kStructureActionWorkspaceBytes = detail::kStructureTileWorkspaceBytes;
constexpr int kMaximumStructureActionBatchColumns = 8;

int bounded_structure_action_width(
    int requested,
    int n_unique_alpha,
    int n_unique_beta) {
  const std::size_t spin_products =
      static_cast<std::size_t>(n_unique_alpha) *
      static_cast<std::size_t>(n_unique_beta);
  const std::size_t simultaneous_spin_product_matrices = 8;
  const std::size_t bytes_per_column = std::max<std::size_t>(
      sizeof(double),
      simultaneous_spin_product_matrices * spin_products * sizeof(double));
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
    const xmvb::mcscf::StringTable& alpha_strings,
    const xmvb::mcscf::StringTable& beta_strings,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e,
    const xmvb::integrals::ActiveSpaceTwoElectronResult& two_electron,
    int n_active,
    int n_unique_alpha,
    int n_unique_beta,
    const Eigen::Ref<const Eigen::MatrixXd>& spin_vectors,
    Eigen::MatrixXd* spin_hamiltonians) {
  const int block_width =
      static_cast<int>(spin_vectors.cols()) / n_unique_beta;
  const int n_pairs = xmvb::integrals::packed_active_pair_count(n_active);
  const int alpha_electrons = alpha_strings.empty()
      ? 0
      : alpha_strings.electron_count();
  const int beta_electrons = beta_strings.empty()
      ? 0
      : beta_strings.electron_count();
  const int alpha_extent = detail::plan_pair_tile_extent(
      n_unique_alpha,
      alpha_electrons,
      n_active,
      two_electron.n_auxiliary_functions,
      true);
  const int beta_extent = detail::plan_pair_tile_extent(
      n_unique_beta,
      beta_electrons,
      n_active,
      two_electron.n_auxiliary_functions,
      true);
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
              xmvb::integrals::apply_active_space_two_electron_kernel_block(
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
    const xmvb::integrals::ActiveSpaceTwoElectronView& two_electron_view,
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
            xmvb::integrals::lookup_active_space_two_electron_kernel_value(
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
    mcscf::ProductIndex left_spin_product,
    mcscf::ProductIndex right_spin_product,
    int n_unique_beta,
    const xmvb::integrals::ActiveSpaceTwoElectronView& two_electron_view,
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
  ExteriorOrbitalTransform alpha_transform;
  std::unique_ptr<ExteriorOrbitalTransform> distinct_beta_transform;
  xmvb::mcscf::DirectCiSigmaAction sigma;
  mutable std::optional<Eigen::MatrixXd> carrier_diagonal;

  OrthogonalDirectCiData(
      const mcscf::SpinStrings& alpha_determinants,
      const mcscf::SpinStrings& beta_determinants,
      const std::vector<double>& active_overlap,
      const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron,
      const xmvb::integrals::ActiveSpaceTwoElectronResult& active_two_electron,
      int n_active_orbitals)
      : integrals(orthogonalize_active_integrals(
            active_overlap,
            active_one_electron,
            active_two_electron,
            n_active_orbitals)),
        alpha_transform(alpha_determinants, integrals.orbital_transform),
        distinct_beta_transform(
            beta_determinants.masks() == alpha_determinants.masks()
                ? nullptr
                : std::make_unique<ExteriorOrbitalTransform>(
                      beta_determinants,
                      integrals.orbital_transform)),
        sigma(alpha_transform.strings(),
              (distinct_beta_transform ? distinct_beta_transform->strings():alpha_transform.strings()),integrals) {}

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
               integrals.ri_pair_factors.size()) *
            sizeof(double) +
        alpha_transform.dynamic_bytes() +
        (distinct_beta_transform
             ? distinct_beta_transform->dynamic_bytes()
             : 0) +
        sigma.dynamic_bytes()-alpha_transform.strings().dynamic_bytes()-
        (distinct_beta_transform ? distinct_beta_transform->strings().dynamic_bytes():0);
    if (carrier_diagonal) bytes+=carrier_diagonal->size()*sizeof(double);
    return bytes;
  }
};

struct StructureAction::TiledPairData {
  std::shared_ptr<const AcceptedPairTileProvider> alpha_provider;
  std::shared_ptr<const AcceptedPairTileProvider> beta_provider;
  mcscf::StringTable alpha_strings;
  mcscf::StringTable beta_strings;
  std::vector<double> active_overlap;
  Eigen::MatrixXd one_electron;
  xmvb::integrals::ActiveSpaceTwoElectronResult two_electron;
  int n_active = 0;

  TiledPairData(
      const SameSpinPairCacheContext& cache,
      const std::vector<double>& overlap,
      const Eigen::Ref<const Eigen::MatrixXd>& h1e,
      const xmvb::integrals::ActiveSpaceTwoElectronResult& active_two_electron,
      int n_active_orbitals)
      : alpha_provider(cache.alpha_pair_provider),
        beta_provider(cache.beta_pair_provider),
        alpha_strings(cache.alpha_reuse_table.unique_determinants),
        beta_strings(cache.beta_reuse_table.unique_determinants),
        active_overlap(overlap),
        one_electron(xmvb::integrals::materialize_raw_one_electron(
            h1e, active_two_electron.orbital_transform)),
        two_electron(active_two_electron.orbital_transform.size() != 0
            ? xmvb::integrals::materialize_raw_active_integrals(active_two_electron)
            : active_two_electron),
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
    bytes+=alpha_strings.dynamic_bytes()+beta_strings.dynamic_bytes();
    return bytes;
  }
};

StructureAction::~StructureAction() = default;
StructureAction::StructureAction(StructureAction&&) noexcept = default;
StructureAction& StructureAction::operator=(StructureAction&&) noexcept = default;

StructureAction::StructureAction(
    StructureIncidence determinant_to_structure_terms,
    int n_structures,
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron,
    const xmvb::integrals::ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    int n_active_orbitals,
    const StructureDiagonal* precomputed_preconditioner,
    StructureActionMode mode)
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

  const xmvb::mcscf::DirectCiActionPlan direct_ci_plan =
      xmvb::mcscf::plan_orthogonal_direct_ci_action(
          same_spin_pair_cache.alpha_reuse_table.basis,
          same_spin_pair_cache.beta_reuse_table.basis,
          n_active_orbitals,
          1);
  // A complete fixed-spin carrier admits an exact change of representation
  // to the orthonormal determinant basis.  This is the production operator,
  // not a heuristic fast path: falling back to the nonorthogonal pair graph
  // would restore quadratic unique-string storage and cofactor work.  The
  // pair representation is retained only for mathematically incomplete
  // carriers, where the exterior transform is not closed.
  const bool use_direct_ci = direct_ci_plan.carrier_complete();
  if (!same_spin_pair_cache.has_pair_providers()) {
    throw std::invalid_argument(
        "structure action requires accepted-pair providers");
  }
  if (mode == StructureActionMode::Davidson && !same_spin_pair_cache.enabled() &&
      precomputed_preconditioner == nullptr) {
    throw std::invalid_argument(
        "topology-only structure action requires a diagonal preconditioner");
  }

  const int n_packed_pairs = xmvb::integrals::packed_active_pair_count(n_active_orbitals);
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
        same_spin_pair_cache.alpha_reuse_table.basis,
        same_spin_pair_cache.beta_reuse_table.basis,
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
    mcscf::ProductIndex spin_product = 0;
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
    const auto spin_product = mcscf::product_index(alpha,beta,n_unique_beta_);
    for (const auto& term : determinant_to_structure_terms[determinant]) {
      if (term.structure_index < 0 ||
          term.structure_index >= n_structures_) {
        throw std::out_of_range(
            "determinant expansion structure index is out of range");
      }
      pending_terms.push_back(PendingSpinTerm{
          spin_product,
          StructureTerm{
              0, term.structure_index, term.coefficient}});
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
    const auto spin_product = pending_terms[pending].spin_product;
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
        spin_terms_.push_back(
            StructureTerm{static_cast<int>(spin_products_.size()) - 1, structure, coefficient});
      }
    }
    spin_term_offsets_.push_back(spin_terms_.size());
  }

  structure_term_offsets_.assign(
      static_cast<std::size_t>(n_structures_) + 1, 0);
  for (const StructureTerm& term : spin_terms_) {
    ++structure_term_offsets_[static_cast<std::size_t>(term.structure) + 1];
  }
  for (int structure = 0; structure < n_structures_; ++structure) {
    structure_term_offsets_[static_cast<std::size_t>(structure) + 1] +=
        structure_term_offsets_[static_cast<std::size_t>(structure)];
  }
  structure_term_indices_.resize(spin_terms_.size());
  std::vector<std::size_t> next_structure_term = structure_term_offsets_;
  for (std::size_t term_index = 0;
       term_index < spin_terms_.size();
       ++term_index) {
    const int structure = spin_terms_[term_index].structure;
    structure_term_indices_[next_structure_term[structure]++] = term_index;
  }

  if (direct_ci_) {
    if (direct_ci_->alpha_transform.occupations() ==
        direct_ci_->beta_transform().occupations())
      spin_exchange_parity_ = detect_spin_exchange();
    auto sources = std::make_unique<SpinSources>(spin_products_, n_unique_alpha_, n_unique_beta_, spin_exchange_parity_);
    sources->alpha_projection = direct_ci_->alpha_transform.projection(sources->alpha_rows);
    sources->beta_projection = direct_ci_->beta_transform().projection(sources->columns);
    sources_ = std::move(sources);
    std::vector<mcscf::ProductIndex> touched;
    touched.reserve(spin_products_.size());
    for (std::size_t group=0; group<spin_products_.size(); ++group)
      if (spin_term_offsets_[group] != spin_term_offsets_[group+1])
        touched.push_back(spin_products_[group]);
    auto support = std::make_unique<mcscf::ProductSupport>(n_active_orbitals,
        direct_ci_->alpha_transform.strings(),
        direct_ci_->beta_transform().strings(), touched);
    if (profiling::phase_totals().enabled)
      std::fprintf(stderr, "XMVB_SUPPORT,structures=%d,alpha=%d,beta=%d,seeds=%zu,full=%d\n",
          n_structures_, n_unique_alpha_, n_unique_beta_, touched.size(), support->full());
    if (!support->full()) product_support_=std::move(support);
    source_contraction_ = prefer_source_contraction(direct_ci_->alpha_transform,
        direct_ci_->beta_transform(), direct_ci_->sigma);
  }

  if (mode == StructureActionMode::ActionOnly) return;

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
  const xmvb::integrals::ActiveSpaceTwoElectronView two_electron_view =
      xmvb::integrals::make_active_space_two_electron_view(active_space_two_electron_result);
  const int n_threads = std::max(
      1,
      std::min(xmvb::effective_openmp_thread_count(), n_structures_));
#pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
  for (int structure = 0; structure < n_structures_; ++structure) {
    double hamiltonian = 0.0;
    double overlap = 0.0;
    const std::size_t first = structure_term_offsets_[structure];
    const std::size_t last = structure_term_offsets_[structure + 1];
    for (std::size_t left_position = first;
         left_position < last;
         ++left_position) {
      const StructureTerm& left =
          spin_terms_[structure_term_indices_[left_position]];
      for (std::size_t right_position = first;
           right_position < last;
           ++right_position) {
        const StructureTerm& right =
            spin_terms_[structure_term_indices_[right_position]];
        const DeterminantPairScalars pair = evaluate_spin_product_pair(
            same_spin_pair_cache,
            spin_products_[left.group],
            spin_products_[right.group],
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
  const int n_threads = std::max(
      1,
      std::min(xmvb::effective_openmp_thread_count(), n_structures_));
#pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
  for (int structure = 0; structure < n_structures_; ++structure) {
    for (std::size_t position = structure_term_offsets_[structure];
         position < structure_term_offsets_[structure + 1];
         ++position) {
      const StructureTerm& term =
          spin_terms_[structure_term_indices_[position]];
      const int alpha = spin_products_[term.group] / n_unique_beta_;
      const int beta = spin_products_[term.group] % n_unique_beta_;
      for (int vector = 0; vector < block_width; ++vector) {
        result(structure, vector) +=
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
    const auto first_product = mcscf::product_index(alpha,beta_begin,n_unique_beta_);
    const auto last_product = first_product + beta_size;
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
  const int n_groups = static_cast<int>(spin_products_.size());
  const int n_threads = std::max(
      1,
      std::min(xmvb::effective_openmp_thread_count(), n_groups));
#pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
  for (int group = 0; group < n_groups; ++group) {
    const auto spin_product = spin_products_[group];
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
  const int n_threads = std::max(
      1,
      std::min(xmvb::effective_openmp_thread_count(), alpha_size));
#pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
  for (int alpha_local = 0; alpha_local < alpha_size; ++alpha_local) {
    const int alpha = alpha_begin + alpha_local;
    const auto first_product = mcscf::product_index(alpha,beta_begin,n_unique_beta_);
    const auto last_product = first_product + beta_size;
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

bool StructureAction::supports_carrier_diagonal() const noexcept {
  if (!direct_ci_) return false;
  const double n_active=direct_ci_->integrals.one_electron.rows();
  // Include the occupation/GEMM workspace used to build the shared diagonal,
  // important for strongly unequal alpha/beta string counts.
  const double elements=4.0*n_unique_alpha_*n_unique_beta_+
      3.0*(n_unique_alpha_+static_cast<double>(n_unique_beta_))*n_active+2.0*n_active*n_active;
  return sizeof(double)*elements<=static_cast<double>(kStructureActionWorkspaceBytes);
}

void StructureAction::divide_carrier_diagonal(Eigen::Ref<Eigen::MatrixXd> carrier,
    const Eigen::Ref<const Eigen::VectorXd>& energies,double eigenvalue) const {
  if (!supports_carrier_diagonal() || carrier.rows()!=n_unique_alpha_ ||
      carrier.cols()!=energies.size()*n_unique_beta_ || !energies.allFinite() ||
      !std::isfinite(eigenvalue))
    throw std::invalid_argument("invalid carrier diagonal input");
  if (!direct_ci_->carrier_diagonal)
    direct_ci_->carrier_diagonal=direct_ci_->sigma.diagonal(direct_ci_->integrals.one_electron);
  for (Eigen::Index state=0;state<energies.size();++state)
    for (int j=0;j<n_unique_beta_;++j) for (int i=0;i<n_unique_alpha_;++i) {
      double denominator=(*direct_ci_->carrier_diagonal)(i,j)-energies[state]-eigenvalue;
      if (std::abs(denominator)<1.0e-8) denominator=1.0e-8;
      carrier(i,state*n_unique_beta_+j)/=denominator;
    }
}

StructureActionResult StructureAction::apply_streamed(
    const Eigen::Ref<const Eigen::MatrixXd>& vectors,
    std::size_t workspace_bytes) const {
  if (vectors.rows() != n_structures_ || vectors.cols() <= 0 || workspace_bytes == 0)
    throw std::invalid_argument("invalid streamed structure input block");
  if (direct_ci_)
    return apply_orthogonal_tiled(vectors, direct_ci_->alpha_transform,
        direct_ci_->beta_transform(), direct_ci_->sigma, false, workspace_bytes);
  if (!tiled_pairs_) {
    throw std::logic_error("tiled structure action has no pair providers");
  }
  const int block_width = static_cast<int>(vectors.cols());
  const int n_pairs = xmvb::integrals::packed_active_pair_count(tiled_pairs_->n_active);
  const int alpha_electrons = tiled_pairs_->alpha_strings.empty()
      ? 0
      : tiled_pairs_->alpha_strings.electron_count();
  const int beta_electrons = tiled_pairs_->beta_strings.empty()
      ? 0
      : tiled_pairs_->beta_strings.electron_count();
  const std::size_t spin_workspace = std::max<std::size_t>(1, workspace_bytes / 2);
  const int alpha_extent = detail::plan_pair_tile_extent(
      n_unique_alpha_,
      alpha_electrons,
      tiled_pairs_->n_active,
      tiled_pairs_->two_electron.n_auxiliary_functions,
      true,
      spin_workspace);
  const int beta_extent = detail::plan_pair_tile_extent(
      n_unique_beta_,
      beta_electrons,
      tiled_pairs_->n_active,
      tiled_pairs_->two_electron.n_auxiliary_functions,
      true,
      spin_workspace);

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
              xmvb::integrals::apply_active_space_two_electron_kernel_block(
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

Eigen::MatrixXd StructureAction::carrier_generator_adjoint(
    const Eigen::Ref<const Eigen::MatrixXd>& left,
    const Eigen::Ref<const Eigen::MatrixXd>& right) const {
  if (!direct_ci_) throw std::logic_error("carrier generator requires a complete ambient carrier");
  return direct_ci_->sigma.one_body_generator_adjoint(left, right);
}

double StructureAction::complete_space_residual_norm(
    const Eigen::Ref<const Eigen::VectorXd>& coefficients, double energy) const {
  if (!direct_ci_ || coefficients.size() != n_structures_ ||
      !coefficients.allFinite() || !std::isfinite(energy))
    throw std::invalid_argument("invalid complete-space Ritz residual input");
  const Eigen::MatrixXd carrier = orthogonalize_structure_block(coefficients);
  const double norm = carrier.stableNorm();
  if (!(norm > 0.0))
    throw std::runtime_error("complete-space Ritz state has zero physical norm");
  const Eigen::MatrixXd residual = direct_ci_->sigma.apply(carrier, spin_exchange_parity_) - energy * carrier;
  return residual.stableNorm() / norm;
}

Eigen::MatrixXd StructureAction::apply_carrier_hamiltonian(
    const Eigen::Ref<const Eigen::MatrixXd>& carrier) const {
  return direct_ci_->sigma.apply(carrier, spin_exchange_parity_);
}

void StructureAction::adjoint_orbital_transform(Eigen::MatrixXd* carrier) const {
  for (Eigen::Index first=0; first<carrier->cols(); first+=n_unique_beta_)
    direct_ci_->beta_transform().apply_adjoint_right_block(
        carrier->middleCols(first,n_unique_beta_));
  direct_ci_->alpha_transform.apply_adjoint_left(carrier);
}

void StructureAction::inverse_orbital_transform(Eigen::MatrixXd* spin_products) const {
  if (!direct_ci_ || spin_products == nullptr ||
      spin_products->rows() != n_unique_alpha_ ||
      spin_products->cols() <= 0 || spin_products->cols() % n_unique_beta_ != 0)
    throw std::invalid_argument("inverse orbital transform needs a complete packed spin carrier");
  direct_ci_->alpha_transform.apply_inverse_left(spin_products);
  for (int first = 0; first < spin_products->cols(); first += n_unique_beta_)
    direct_ci_->beta_transform().apply_inverse_right_block(
        spin_products->middleCols(first, n_unique_beta_));
}

void StructureAction::inverse_adjoint_orbital_transform(Eigen::MatrixXd* spin_products) const {
  if (!direct_ci_ || spin_products == nullptr ||
      spin_products->rows() != n_unique_alpha_ ||
      spin_products->cols() <= 0 || spin_products->cols() % n_unique_beta_ != 0)
    throw std::invalid_argument("inverse orbital adjoint needs a complete packed spin carrier");
  direct_ci_->alpha_transform.apply_inverse_adjoint_left(spin_products);
  for (int first = 0; first < spin_products->cols(); first += n_unique_beta_)
    direct_ci_->beta_transform().apply_inverse_adjoint_right_block(
        spin_products->middleCols(first, n_unique_beta_));
}

StructureAction::OrthogonalView StructureAction::orthogonal_view() const {
  if (!direct_ci_)
    throw std::invalid_argument("structure action has no orthogonal representation");
  return {direct_ci_->alpha_transform, direct_ci_->beta_transform(), direct_ci_->sigma,
      spin_products_, spin_term_offsets_, spin_terms_,spin_exchange_parity_,product_support_.get()};
}

Eigen::MatrixXd StructureAction::apply_hamiltonian(
    const Eigen::Ref<const Eigen::MatrixXd>& vectors) const {
  if (vectors.rows()!=n_structures_ || vectors.cols()<=0)
    throw std::invalid_argument("invalid Hamiltonian input block");
  if (!direct_ci_ || source_contraction_) return apply(vectors).hamiltonian;
  return contract_product_tiles(vectors,direct_ci_->alpha_transform,
      direct_ci_->beta_transform(),direct_ci_->sigma,false,
      workspace::available(),false).images.hamiltonian;
}

Eigen::MatrixXd StructureAction::apply_overlap(
    const Eigen::Ref<const Eigen::MatrixXd>& vectors) const {
  if (vectors.rows() != n_structures_ || vectors.cols() <= 0 || !vectors.allFinite())
    throw std::invalid_argument("invalid structure overlap input block");
  if (!direct_ci_) return apply(vectors).overlap;
  return apply_orthogonal_tiled(vectors, direct_ci_->alpha_transform,
      direct_ci_->beta_transform(), direct_ci_->sigma, true).overlap;
}

StructureActionResult StructureAction::apply_sources(
    const Eigen::Ref<const Eigen::MatrixXd>& vectors, std::size_t workspace_bytes) const {
  if (!direct_ci_ || vectors.rows() != n_structures_ || vectors.cols() <= 0)
    throw std::invalid_argument("source contraction needs an orthogonal-CI structure action");
  return apply_source_tiles(vectors, direct_ci_->alpha_transform,
      direct_ci_->beta_transform(), direct_ci_->sigma, false, workspace_bytes);
}

StructureActionResult StructureAction::apply(
    const Eigen::Ref<const Eigen::MatrixXd>& vectors) const {
  if (vectors.rows() != n_structures_ || vectors.cols() <= 0) {
    throw std::invalid_argument(
        "structure vector block has incompatible dimensions");
  }
  if (direct_ci_) {
    return apply_orthogonal_tiled(vectors, direct_ci_->alpha_transform,
        direct_ci_->beta_transform(), direct_ci_->sigma);
  }
  const int block_width = static_cast<int>(vectors.cols());
  const int bounded_width = bounded_structure_action_width(
      block_width, n_unique_alpha_, n_unique_beta_);
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

  if (!full_factorized_action_fits(
                         n_unique_alpha_, n_unique_beta_, block_width)) {
    return apply_streamed(vectors);
  }

  Eigen::MatrixXd spin_vectors = expand_structure_block(vectors);
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
    StructureIntegralDirection* direct_ci_direction,
    StructureResponseRequest* ci_response,
    const Eigen::MatrixXd* ri_factor_direction) const {
  xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::CiIntegralDirection);
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
  const auto prepared_direction = std::make_shared<OrthogonalActiveIntegrals>(
      orthogonalize_active_integral_direction(
          direct_ci_->integrals,
          overlap_direction,
          delta_one_electron,
          packed_two_electron_direction,
          n_active_orbitals,
          ri_factor_direction));
  const auto& integral_direction=*prepared_direction;
  if (direct_ci_direction) direct_ci_direction->integrals=prepared_direction;
  const xmvb::mcscf::DirectCiSigmaAction sigma_direction(
      direct_ci_->sigma,
      integral_direction);
  if (state.source_factors) {
    if (ci_response) ci_response->result=apply_response(state,ci_response->coefficients);
    const Eigen::MatrixXd relative_orbital_direction =
        direct_ci_->integrals.orbital_transform.transpose().triangularView<Eigen::Lower>()
            .solve(integral_direction.orbital_transform.transpose()).transpose();
    return source_integral_action(state,relative_orbital_direction,
        direct_ci_->alpha_transform,direct_ci_->beta_transform(),direct_ci_->sigma,
        sigma_direction,integral_direction,direct_ci_direction);
  }

  // Preserve the cheaper spin-factor contraction when its existing FLOP plan
  // selects it. Only two product-jet consumers share the product tile sweep.
  const bool joint=ci_response && !ci_response->coefficients.isZero(0.0) &&
      !prefer_factor_response(direct_ci_->sigma);
  if (ci_response && !joint) ci_response->result=apply_response(state,ci_response->coefficients);
  auto sweep=product_jet(state,joint ? &ci_response->coefficients:nullptr,
      &integral_direction.orbital_transform,&sigma_direction,
      direct_ci_->alpha_transform,direct_ci_->beta_transform(),direct_ci_->sigma,true,
      direct_ci_direction,joint ? &ci_response->result:nullptr);
  if (direct_ci_direction) {
    direct_ci_direction->generator_direction=std::move(sweep.generator);
    direct_ci_direction->one_electron_direction=std::move(sweep.one_electron);
    direct_ci_direction->pair_kernel_direction=std::move(sweep.pair_kernel);
  }
  return std::move(sweep.images);
}

bool StructureAction::supports_integral_direction() const noexcept {
  return direct_ci_ != nullptr;
}

Eigen::MatrixXd StructureAction::source_generator_adjoint(
    const StructureAdjointState& state, int left, int right) const {
  const Eigen::VectorXd a = expand_support(state.structure_coefficients.col(left));
  Eigen::VectorXd b;
  if (right != left) b = expand_support(state.structure_coefficients.col(right));
  const auto bra = sources_->orthogonal_source(spin_products_, a,
      direct_ci_->alpha_transform, direct_ci_->beta_transform(), true);
  const auto ket = sources_->orthogonal_source(spin_products_, right == left ? a : b,
      direct_ci_->alpha_transform, direct_ci_->beta_transform());
  return mcscf::factor_generator_adjoint(direct_ci_->sigma,bra,ket);
}

StructureAdjointState StructureAction::prepare_active_adjoint(
    const Eigen::Ref<const Eigen::MatrixXd>& structure_coefficients,
    const std::vector<double>& state_weights,
    const std::vector<double>& state_energies,
    StructureActionResult* images,
    const Eigen::MatrixXd* overlap_image) const {
  if (!direct_ci_) {
    throw std::logic_error(
        "active-integral adjoint requires orthogonal direct CI");
  }
  const int n_states = structure_coefficients.cols();
  if (structure_coefficients.rows() != n_structures_ || n_states <= 0 ||
      !structure_coefficients.allFinite() || state_weights.size() != state_energies.size() ||
      state_weights.size() != static_cast<std::size_t>(n_states)) {
    throw std::invalid_argument(
        "selected-state adjoint inputs have incompatible dimensions");
  }

  const int n_orbitals =
      static_cast<int>(direct_ci_->integrals.one_electron.rows());
  const int n_pairs = xmvb::integrals::packed_active_pair_count(n_orbitals);
  StructureAdjointState result;
  result.structure_coefficients = structure_coefficients;
  result.weights = state_weights;
  result.energies = state_energies;
  // Compare the arithmetic of this consumer, independently of the H/M order.
  // Source-pair moments are quadratic in rank and must not be omitted here.
  result.source_factors = prefer_factor_response(direct_ci_->sigma);
  result.one_electron_gradient = Eigen::MatrixXd::Zero(
      n_orbitals, n_orbitals);
  result.pair_kernel_gradient = Eigen::MatrixXd::Zero(
      n_pairs, n_pairs);
  result.generator_gradient = Eigen::MatrixXd::Zero(
      n_orbitals, n_orbitals);
  if (images) {
    images->hamiltonian = Eigen::MatrixXd::Zero(n_structures_, n_states);
    images->overlap = overlap_image ? *overlap_image:
        Eigen::MatrixXd::Zero(n_structures_, n_states);
  }

  if (!result.source_factors) {
    result.product_images=std::make_shared<detail::ProductImages>();
    auto sweep=product_jet(result,nullptr,nullptr,nullptr,direct_ci_->alpha_transform,
        direct_ci_->beta_transform(),direct_ci_->sigma,images != nullptr,
        nullptr,nullptr,overlap_image);
    result.one_electron_gradient=sweep.one_electron;
    result.pair_kernel_gradient=sweep.pair_kernel;
    result.generator_gradient=sweep.generator;
    if (images) *images = std::move(sweep.images);
  }
  for (int state = 0; state < n_states; ++state) {
    const double state_weight = state_weights[state];
    if (!std::isfinite(state_weight) || state_weight < 0.0 ||
        !std::isfinite(state_energies[state])) {
      throw std::invalid_argument("invalid selected-state adjoint input");
    }
    if (!result.source_factors || (state_weight==0 && !images)) continue;
    const Eigen::VectorXd values = expand_support(structure_coefficients.col(state));
    const auto y = sources_->orthogonal_source(spin_products_,values,
        direct_ci_->alpha_transform,direct_ci_->beta_transform());
    const auto physical = sources_->orthogonal_source(spin_products_,values,
        direct_ci_->alpha_transform,direct_ci_->beta_transform(),true);
    const auto integral = mcscf::factor_integral_adjoint(direct_ci_->sigma,physical,y);
    result.one_electron_gradient.noalias() += state_weight*integral.one_electron;
    result.pair_kernel_gradient.noalias() += state_weight*integral.pair_kernel;
    if (images) {
      // The image generator already visits H Y. Consume its structure
      // projection here, rather than replaying sigma in a separate H/M call.
      auto image = project_source_adjoint(y,y,direct_ci_->alpha_transform,
          direct_ci_->beta_transform(),direct_ci_->sigma,nullptr,true,state_energies[state]);
      result.generator_gradient.noalias() += 2*state_weight*image.generator;
      images->hamiltonian.col(state) = image.images.hamiltonian.col(0);
      if (!overlap_image) images->overlap.col(state) = image.images.overlap.col(0);
    } else {
      result.generator_gradient.noalias() += 2*state_weight*mcscf::factor_residual_generator_adjoint(
          direct_ci_->sigma,physical,y,state_energies[state]);
    }
  }

  for (std::size_t i = 0; i < result.weights.size(); ++i)
    for (std::size_t j = 0; j <= i; ++j) {
      if (result.weights[i] == 0.0 || result.weights[j] == 0.0)
        result.overlap_generators.push_back(Eigen::MatrixXd::Zero(n_orbitals, n_orbitals));
      else if (n_states==1 && result.product_images)
        result.overlap_generators.push_back(result.product_images->density);
      else result.overlap_generators.push_back(source_generator_adjoint(result,j,i));
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
    const Eigen::MatrixXd* coefficient_response,
    const Eigen::MatrixXd* directional_state_multipliers,
    const std::vector<double>& overlap_direction,
    const std::vector<double>& one_electron_direction,
    const std::vector<double>& packed_two_electron_direction,
    const StructureIntegralDirection* direct_ci_direction,
    bool include_integral_response,
    const Eigen::MatrixXd* ri_factor_direction) const {
  if (!direct_ci_) {
    throw std::logic_error(
        "active-integral adjoint direction requires orthogonal direct CI");
  }
  const bool include_state_response = coefficient_response != nullptr;
  const std::size_t n_states = state.weights.size();
  if (n_states == 0 || state.energies.size() != n_states ||
      state.structure_coefficients.rows() != n_structures_ ||
      state.structure_coefficients.cols() != static_cast<Eigen::Index>(n_states) ||
      (include_state_response &&
       (coefficient_response->rows() != n_structures_ ||
        coefficient_response->cols() != static_cast<Eigen::Index>(n_states) ||
        !coefficient_response->allFinite())) ||
      (directional_state_multipliers &&
       (directional_state_multipliers->rows() !=
            static_cast<Eigen::Index>(n_states) ||
        directional_state_multipliers->cols() !=
            static_cast<Eigen::Index>(n_states) ||
        !directional_state_multipliers->allFinite()))) {
    throw std::invalid_argument(
        "selected-state adjoint-direction inputs have incompatible dimensions");
  }

  const int n_orbitals =
      static_cast<int>(direct_ci_->integrals.one_electron.rows());
  const int n_pairs = xmvb::integrals::packed_active_pair_count(n_orbitals);
  OrthogonalActiveIntegrals computed_direction;
  const auto* prepared=include_integral_response && direct_ci_direction ?
      direct_ci_direction->integrals.get():nullptr;
  if (include_integral_response && !prepared) {
    if (one_electron_direction.size() !=
        static_cast<std::size_t>(n_orbitals) * n_orbitals) {
      throw std::invalid_argument(
          "active one-electron direction has incompatible dimensions");
    }
    const Eigen::Map<const Eigen::MatrixXd> delta_one_electron(
        one_electron_direction.data(), n_orbitals, n_orbitals);
    computed_direction = orthogonalize_active_integral_direction(
        direct_ci_->integrals,
        overlap_direction,
        delta_one_electron,
        packed_two_electron_direction,
        n_orbitals,
        ri_factor_direction);
  } else if (!include_integral_response) {
    computed_direction.orbital_transform = Eigen::MatrixXd::Zero(
        n_orbitals, n_orbitals);
    computed_direction.one_electron = Eigen::MatrixXd::Zero(
        n_orbitals, n_orbitals);
    computed_direction.pair_kernel = Eigen::MatrixXd::Zero(n_pairs, n_pairs);
  }
  const auto& integral_direction=prepared ? *prepared:computed_direction;

  if (state.source_factors) {
    const Eigen::MatrixXd empty;
    const Eigen::MatrixXd& response = coefficient_response ? *coefficient_response : empty;
    return source_adjoint_direction(state,response,directional_state_multipliers,
        direct_ci_->integrals,integral_direction,direct_ci_->alpha_transform,
        direct_ci_->beta_transform(),direct_ci_->sigma,include_integral_response,direct_ci_direction);
  }

  StructureResponseSweep sweep{{},Eigen::MatrixXd::Zero(n_orbitals,n_orbitals),
      Eigen::MatrixXd::Zero(n_pairs,n_pairs),Eigen::MatrixXd::Zero(n_orbitals,n_orbitals)};
  if (include_integral_response) {
    if (direct_ci_direction) {
      sweep.one_electron=direct_ci_direction->one_electron_direction;
      sweep.pair_kernel=direct_ci_direction->pair_kernel_direction;
      sweep.generator=direct_ci_direction->generator_direction;
    } else {
      const mcscf::DirectCiSigmaAction delta_sigma(direct_ci_->sigma,integral_direction);
      sweep=product_jet(state,nullptr,&integral_direction.orbital_transform,&delta_sigma,
          direct_ci_->alpha_transform,direct_ci_->beta_transform(),direct_ci_->sigma,false);
    }
  }
  if (coefficient_response && !coefficient_response->isZero(0.0)) {
    const auto q=product_jet(state,coefficient_response,nullptr,nullptr,
        direct_ci_->alpha_transform,direct_ci_->beta_transform(),direct_ci_->sigma,false);
    sweep.one_electron+=q.one_electron; sweep.pair_kernel+=q.pair_kernel; sweep.generator+=q.generator;
  }
  if (directional_state_multipliers) add_response_gauge(state,*directional_state_multipliers,sweep.generator);
  auto result=backpropagate_orthogonal_active_integral_adjoint_direction(
      direct_ci_->integrals,integral_direction,state.one_electron_gradient,state.pair_kernel_gradient,
      state.generator_gradient,sweep.one_electron,sweep.pair_kernel,sweep.generator);
  return {std::move(result.overlap),std::move(result.one_electron),std::move(result.pair_kernel)};
}

StructureResponseSweep StructureAction::apply_response(
    const StructureAdjointState& state,
    const Eigen::Ref<const Eigen::MatrixXd>& coefficients) const {
  if (coefficients.rows() != n_structures_ ||
      coefficients.cols() != static_cast<Eigen::Index>(state.weights.size()) ||
      !coefficients.allFinite())
    throw std::invalid_argument("invalid structure response columns");
  if (direct_ci_ && prefer_factor_response(direct_ci_->sigma))
    return source_response(state,coefficients,direct_ci_->alpha_transform,
        direct_ci_->beta_transform(),direct_ci_->sigma);
  return contract_response(state,coefficients,true);
}

StructureActiveIntegralAdjoint StructureAction::active_integral_response_adjoint(
    const StructureAdjointState& state,
    const Eigen::Ref<const Eigen::MatrixXd>& coefficient_response,
    const Eigen::Ref<const Eigen::MatrixXd>& state_multipliers) const {
  if (coefficient_response.rows() != n_structures_ ||
      coefficient_response.cols() != static_cast<Eigen::Index>(state.weights.size()) ||
      !coefficient_response.allFinite())
    throw std::invalid_argument("invalid selected-subspace response dimensions");
  if (state.source_factors) {
    return finish_response(state, source_response(state,coefficient_response,
        direct_ci_->alpha_transform,direct_ci_->beta_transform(),direct_ci_->sigma,false),
        state_multipliers);
  }
  return finish_response(state,contract_response(state,coefficient_response,false),state_multipliers);
}

StructureResponseSweep StructureAction::contract_response(
    const StructureAdjointState& state,
    const Eigen::Ref<const Eigen::MatrixXd>& coefficients,
    bool collect_images) const {
  if (!direct_ci_) {
    throw std::logic_error(
        "active-integral response adjoint requires orthogonal direct CI");
  }
  const Eigen::MatrixXd response=coefficients;
  if (response.isZero(0.0)) {
    const int n=direct_ci_->sigma.n_orbitals(),d=direct_ci_->sigma.channel_kernel().rows();
    return {{Eigen::MatrixXd::Zero(collect_images ? n_structures_:0,response.cols()),
             Eigen::MatrixXd::Zero(collect_images ? n_structures_:0,response.cols())},
            Eigen::MatrixXd::Zero(n,n),Eigen::MatrixXd::Zero(d,d),Eigen::MatrixXd::Zero(n,n)};
  }
  return product_jet(state,&response,nullptr,nullptr,direct_ci_->alpha_transform,
      direct_ci_->beta_transform(),direct_ci_->sigma,collect_images);
}

StructureActiveIntegralAdjoint StructureAction::finish_response(
    const StructureAdjointState& state, StructureResponseSweep sweep,
    const Eigen::Ref<const Eigen::MatrixXd>& state_multipliers) const {
  const Eigen::Index n_states = state.weights.size();
  if (!direct_ci_ || state_multipliers.rows() != n_states ||
      state_multipliers.cols() != n_states || !state_multipliers.allFinite())
    throw std::invalid_argument("invalid response multiplier");
  add_response_gauge(state,state_multipliers,sweep.generator);
  NonorthogonalActiveIntegralAdjoint result =
      backpropagate_orthogonal_active_integral_adjoint(
          direct_ci_->integrals,
          sweep.one_electron,
          sweep.pair_kernel,
          sweep.generator);
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
    if (product_support_) result.direct_ci_bytes += product_support_->dynamic_bytes()-product_support_->basis_bytes();
  }
  if (tiled_pairs_) {
    result.factor_bytes += tiled_pairs_->dynamic_bytes();
  }
  result.expansion_bytes =
      spin_products_.size() * sizeof(mcscf::ProductIndex) +
      spin_term_offsets_.size() * sizeof(std::size_t) +
      spin_terms_.size() * sizeof(StructureTerm) +
      structure_term_offsets_.size() * sizeof(std::size_t) +
      structure_term_indices_.size() * sizeof(std::size_t);
  if (sources_) result.expansion_bytes += sources_->dynamic_bytes();
  result.diagonal_bytes =
      static_cast<std::size_t>(
          preconditioner_diagonal_.hamiltonian.size() +
          preconditioner_diagonal_.overlap.size()) *
      sizeof(double);
  return result;
}

}  // namespace xmvb::vb
