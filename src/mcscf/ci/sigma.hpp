#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <span>
#include <vector>

#include <Eigen/Core>

#include "mcscf/ci/integrals.hpp"
#include "mcscf/ci/links.hpp"
#include "mcscf/ci/string_table.hpp"

namespace xmvb::mcscf {

namespace detail {
class SigmaSchedule;
}

class ProductSupport;
class DirectCiSigmaAction;

/** @brief Add one adjoint column range directly to a caller-owned spin column.
 * Arguments are local beta, RHS, first/last indices into alpha_rows, and a
 * destination, and its optional sorted original alpha indices and count.
 * A null index pointer uses dense original-alpha indexing; otherwise values
 * are accumulated directly into the compact selected row list.
 * Disjoint ranges may run concurrently; the callback must complete before
 * the batch is released.
 */
using DirectCiColumnAccumulator = std::function<void(
    StringIndex, StringIndex, StringIndex, StringIndex, double*, const StringIndex*, StringIndex)>;

/** @brief Dense consumers require explicit zeros; Supported consumers promise
 * not to read outside the producer's exact structural source support.
 * Supported reads may leave those unused slots untouched, without changing
 * the rectangular workspace or introducing compact-index lookups.
 */
enum class DirectCiRead { Dense, Supported };
using DirectCiTileReader = std::function<void(const std::vector<StringIndex>&,
                                              const std::vector<StringIndex>&,
                                              Eigen::Ref<Eigen::MatrixXd>,
                                              DirectCiRead)>;
using DirectCiTileWriter = std::function<void(const std::vector<StringIndex>&,
                                              const std::vector<StringIndex>&,
                                              const DirectCiColumnAccumulator&)>;

/** @brief Release columns after their last one-generator consumer has finished.
 * Indices will not be read or written again in this action. Called by the
 * coordinator after all workers have consumed the batch; views are borrowed.
 */
using DirectCiColumnRelease = std::function<void(std::span<const StringIndex>)>;

/** @brief One temporal response on a common accepted state and generated graph.
 * Image receives H*response, plus direction*state when direction is supplied.
 * Forcing optionally receives only direction*state. All views are borrowed;
 * writers must consume a tile before the next request reuses its storage.
 */
struct DirectCiTileRequest {
    DirectCiTileReader response;
    DirectCiTileWriter image;
    const DirectCiSigmaAction* direction = nullptr;
    DirectCiTileWriter forcing{};
};

/** @brief Borrowed Eigen column payload, valid until the next column read.
 * A null rows pointer denotes all alpha rows; otherwise rows is sorted and
 * values contains only those coordinates. Consumers must finish before
 * requesting another column, which may evict its owning panel.
 */
struct DirectCiColumnView {
    const double* values;
    const StringIndex* rows;
    StringIndex count;
};
using DirectCiColumnReader = std::function<DirectCiColumnView(StringIndex)>;

/** @brief Integral adjoint of one orthonormal direct-CI bilinear form. */
struct DirectCiIntegralAdjoint {
    Eigen::MatrixXd one_electron;
    Eigen::MatrixXd pair_kernel;
};

/** @brief Sigma and integral adjoint sharing one spin-string tile sweep. */
struct DirectCiResponse {
    Eigen::MatrixXd sigma;
    DirectCiIntegralAdjoint integrals;
};

/** @brief One generator action; inputs are views, output is caller-owned. */
struct DirectCiGeneratorRequest {
    Eigen::Ref<const Eigen::MatrixXd> coefficients;
    Eigen::Ref<const Eigen::MatrixXd> generator;
    Eigen::MatrixXd* result;
};

/** @brief Resident right argument of a column-panel generator contraction. */
struct DirectCiGeneratorAdjointRequest {
    Eigen::Ref<const Eigen::MatrixXd> right;
    double scale = 1.0;
    bool transpose = false;
};

/** @brief Generated right argument of a streamed generator contraction.
 * read fills a requested beta/alpha tile. column borrows one beta spectator;
 * its storage may be invalidated by the next read or column call, including
 * a call through another request sharing the same backing cache.
 */
struct DirectCiStreamedGeneratorRequest {
    DirectCiTileReader read;
    DirectCiColumnReader column;
    double scale = 1.0;
    bool transpose = false;
    /** Column storage and indices remain immutable throughout this call. */
    bool retained_columns = false;
};

/**
 * @brief Exact Slater--Condon sigma action in an orthonormal active basis.
 */
class DirectCiSigmaAction {
  public:
    /** @brief E_(spin,P) applied to single-spin source columns.
   * Output column P*width+j contains one channel image of source j.
   * No alpha-by-beta product is constructed.
   */
    void scatter_spin(bool beta,
                      const Eigen::Ref<const Eigen::MatrixXd>& factors,
                      int first,
                      int count,
                      Eigen::Ref<Eigen::MatrixXd> channels) const;

    /** @brief Accumulates sum_P E_(spin,P)^T channels[P] into single-spin images. */
    void gather_spin(bool beta,
                     const Eigen::Ref<const Eigen::MatrixXd>& channels,
                     int first,
                     int count,
                     Eigen::Ref<Eigen::MatrixXd> images) const;

    /** @brief Apply sum_pq kappa(p,q) a_p^+ a_q to single-spin source columns.
   * The generator need not be symmetric. Input and output may be strided views,
   * but must not alias. Each output spin row has one owner.
   */
    void apply_spin_generator(bool beta,
                              const Eigen::Ref<const Eigen::MatrixXd>& factors,
                              const Eigen::Ref<const Eigen::MatrixXd>& generator,
                              Eigen::Ref<Eigen::MatrixXd> image) const;

    /** @brief Unsymmetrized single-spin contraction sum_j <left_j, E_pq right_j>.
   * Input panels may be non-owning views. Only small orbital-matrix reductions
   * are thread-local; no spin/product array is copied per worker.
   */
    Eigen::MatrixXd spin_generator_adjoint(bool beta,
                                           const Eigen::Ref<const Eigen::MatrixXd>& left,
                                           const Eigen::Ref<const Eigen::MatrixXd>& right) const;

    /** @brief Symmetric K in H = (E_alpha+E_beta)^T K (E_alpha+E_beta).
   * Includes the absorbed one-electron Hamiltonian. Borrowed immutable data.
   */
    const Eigen::MatrixXd& channel_kernel() const noexcept {
        return hamiltonian_pair_kernel_;
    }
    /** Borrowed immutable topology used by the packed-channel CUDA action. */
    const SpinConnections& alpha_connections() const noexcept {
        return *alpha_;
    }
    const SpinConnections& beta_connections_view() const noexcept {
        return distinct_beta_ ? *distinct_beta_ : *alpha_;
    }
    bool shared_spin_connections() const noexcept {
        return !distinct_beta_;
    }
    StringIndex n_alpha_strings() const noexcept {
        return n_alpha_;
    }
    StringIndex n_beta_strings() const noexcept {
        return n_beta_;
    }
    int n_orbitals() const noexcept {
        return n_orbitals_;
    }
    /** @brief Occupation plus single-replacement links in both complete spin spaces. */
    std::size_t spin_link_count() const noexcept;
    /** @brief Upper comparison work for the two sorted channel/occupation lookups. */
    long double channel_search_work() const noexcept;

    /** @brief Pulls a K-channel cotangent back to symmetric h/(pq|rs) adjoints.
   * Shared by product and source-factor contractions; includes the absorbed
   * one-electron and normal-ordering terms, with full Frobenius conventions.
   */
    DirectCiIntegralAdjoint
    pullback_channel_kernel(const Eigen::Ref<const Eigen::MatrixXd>& channel_adjoint) const;

    DirectCiSigmaAction(SpinStrings alpha, SpinStrings beta, const ActiveIntegrals& integrals);
    DirectCiSigmaAction(const StringTable& alpha,
                        const StringTable& beta,
                        const ActiveIntegrals& integrals)
        : DirectCiSigmaAction(alpha.basis(integrals.one_electron.rows()),
                              beta.basis(integrals.one_electron.rows()),
                              integrals) {}

    DirectCiSigmaAction(const std::vector<std::vector<int>>& alpha_determinants,
                        const std::vector<std::vector<int>>& beta_determinants,
                        const ActiveIntegrals& integrals);

    /** @brief Reuses immutable spin-string links for another integral kernel.
   *
   * Orbitals must have the same ordering and count as the reference action.
   * Only the numerical integral kernels are rebuilt; links are shared, not copied.
   */
    DirectCiSigmaAction(const DirectCiSigmaAction& reference, const ActiveIntegrals& integrals);

    /**
   * @brief Applies the Hamiltonian to packed determinant-product blocks.
   *
   * The input shape is `(n_alpha, block_width * n_beta)`; beta columns for one
   * block vector are contiguous. A nonzero exchange_parity requires certified
   * common transpose parity in identical spin spaces and returns a full image.
   */
    Eigen::MatrixXd apply(const Eigen::Ref<const Eigen::MatrixXd>& coefficients,
                          int exchange_parity = 0) const;

    /** @brief Stream the exact Hamiltonian action through beta-column callbacks.
   *
   * Each callback receives sorted, distinct beta and alpha indices and a
   * `(alpha_rows.size(), betas.size()*block_width)` slab, with column
   * `beta_local*block_width+rhs`. The
   * reader fills its slab from an immutable input carrier. The writer receives
   * a column accumulator, which gathers directly into caller-owned output;
   * no intermediate image slab or copy is needed. Neither callback may retain
   * batch views or the accumulator across calls. Generated targets
   * span multiple beta strings; their excitation neighborhoods are merged
   * within the batch. Callbacks run in an OpenMP single region and may use
   * taskgroups, but must finish their tasks before returning. The team shares
   * bounded halos and channel slabs, not carrier copies.
   * With product support, the reader must return zero outside S and targets
   * are generated in its one-generator closure I. The adjoint returns the
   * full Hamiltonian image, including rows outside S; a selected-structure
   * caller projects those out in its exterior-transform adjoint.
   * A nonzero exchange_parity (+1 or -1) instead contracts one representative
   * of each alpha/beta exchange orbit with its exact multiplicity. This is
   * valid ONLY when both the input and the final left projection have that
   * transpose parity in identical spin-string spaces. The intermediate
   * output is then a projected-action representative, NOT the full H image.
   */
    /** source_projected_image declares that the final bra is also supported on
   * `support`. Only then may both sides of the channel kernel be restricted.
   * This is a bilinear-form contract, not an amplitude approximation. */
    void apply_tiled(int block_width,
                     const DirectCiTileReader& read_columns,
                     const DirectCiTileWriter& add_columns,
                     const ProductSupport* support = nullptr,
                     int exchange_parity = 0,
                     const DirectCiColumnRelease& release = {},
                     std::size_t workspace_bytes = 64ULL * 1024 * 1024,
                     bool source_projected_image = false) const;

    /** @brief Stream H*response and the integral adjoint of <response,H*state>.
   * Uses the same generated channel sweep and bounded numerical slabs as
   * apply_tiled(). An empty state reader means state=response and reuses its
   * scatter. A distinct reader reuses the input and transformed-channel slabs,
   * not another carrier or per-thread channel array. The optional writer has
   * the same projected-image convention as apply_tiled(); it may be empty for
   * an adjoint-only request. Support must contain both inputs; a nonzero
   * exchange parity certifies that both have that same parity.
   */
    DirectCiIntegralAdjoint apply_tiled_with_adjoint(int block_width,
                                                     const DirectCiTileReader& response,
                                                     const DirectCiTileWriter& add_columns,
                                                     const DirectCiTileReader& state = {},
                                                     const ProductSupport* support = nullptr,
                                                     int exchange_parity = 0,
                                                     const DirectCiColumnRelease& release = {},
                                                     std::size_t workspace_bytes = 64ULL * 1024 *
                                                                                   1024) const;

    /** @brief Fuse H*response + deltaH*state with the integral adjoint.
   * The required state reader supplies unweighted coefficients. Both readers
   * use the same RHS packing, support, and certified exchange parity; the
   * optional writers add the combined image and deltaH*state, respectively,
   * with the projected-image convention of apply_tiled(). The integral
   * adjoint is that of <response,H*state>, excluding the direction kernel.
   * Direction must share this action's immutable spin-string links and
   * orbital ordering (as with the shared-links constructor). Only three
   * shared channel slabs and one input halo are live within the supplied tile
   * workspace; neither reader nor writer may retain a batch view.
   */
    DirectCiIntegralAdjoint apply_tiled_direction(int block_width,
                                                  const DirectCiTileReader& response,
                                                  const DirectCiTileReader& state,
                                                  const DirectCiSigmaAction& direction,
                                                  const DirectCiTileWriter& add_columns,
                                                  const DirectCiTileWriter& add_forcing = {},
                                                  const ProductSupport* support = nullptr,
                                                  int exchange_parity = 0,
                                                  const DirectCiColumnRelease& release = {},
                                                  std::size_t workspace_bytes = 64ULL * 1024 *
                                                                                1024) const;

    /** @brief Independent responses sharing one accepted scatter per tile.
   * Each returned integral adjoint is <request.response,H*state>, without
   * combining directions. All inputs share width, exact support and parity.
   * Requests reuse the same halo and at most three channel slabs under the
   * supplied workspace bound. Carrier storage does not grow with the request
   * count; the small orbital-space integral cotangents remain separate.
   */
    std::vector<DirectCiIntegralAdjoint>
    apply_tiled_responses(int block_width,
                          std::span<const DirectCiTileRequest> requests,
                          const DirectCiTileReader& state,
                          const ProductSupport* support = nullptr,
                          int exchange_parity = 0,
                          const DirectCiColumnRelease& release = {},
                          std::size_t workspace_bytes = 64ULL * 1024 * 1024) const;

    /** @brief Hamiltonian diagonal on the unique alpha/beta product carrier.
   * Uses occupation lists and the retained two-electron kernel, not sigma
   * probes. The one-electron integrals must belong to this immutable action.
   */
    Eigen::MatrixXd diagonal(const Eigen::Ref<const Eigen::MatrixXd>& one_electron) const;

    /**
   * @brief Applies a general one-body orbital generator.
   *
   * The generator follows `sum_pq kappa(p,q) a_p^+ a_q` and need not be
   * symmetric. The coefficient packing is identical to `apply()`.
   */
    Eigen::MatrixXd
    apply_one_body_generator(const Eigen::Ref<const Eigen::MatrixXd>& coefficients,
                             const Eigen::Ref<const Eigen::MatrixXd>& generator) const;

    /** @brief Applies equally shaped generator requests in one parallel team.
   * Inputs stay shared; each request must have a distinct, non-aliasing output.
   */
    void apply_one_body_generators(std::initializer_list<DirectCiGeneratorRequest> requests) const;

    /** @brief One-body generator on a bounded beta panel of an immutable carrier.
   * Column views must remain valid for the entire call. One shared dense input
   * panel and borrowed beta spectators serve all workers. No spin-square
   * generator or thread-local coefficient array is formed. alpha_only omits
   * beta links; a caller may reconstruct them by a complete exchange-parity
   * projection, but an isolated output panel is not a complete spin action.
   */
    void apply_one_body_generator_columns(StringIndex beta_begin,
                                          const DirectCiColumnReader& columns,
                                          const Eigen::Ref<const Eigen::MatrixXd>& generator,
                                          Eigen::Ref<Eigen::MatrixXd> result,
                                          bool alpha_only = false) const;

    /**
   * @brief Differentiates `left.dot(H * right)` with respect to the integrals.
   *
   * Both returned matrices use the symmetric full-matrix Frobenius convention:
   * their inner products with symmetric integral directions equal the exact
   * directional derivative of the bilinear form. The traversal uses the same
   * Slater--Condon connection graph as `apply()`.
   * A nonzero exchange_parity certifies X^T = parity * X for every input
   * block in identical alpha/beta string lists. Zero makes no such assumption.
   */
    DirectCiIntegralAdjoint integral_adjoint(const Eigen::Ref<const Eigen::MatrixXd>& left,
                                             const Eigen::Ref<const Eigen::MatrixXd>& right,
                                             int exchange_parity = 0) const;

    /** @brief Returns H * response and the integral derivative of response.dot(H * state).
   * Reuses the response scatter; two shared tiles serve both contractions.
   * Inputs follow apply() packing and may be non-owning strided blocks.
   * With certified exchange parity, sigma is reconstructed as a full image,
   * not the unprojected representative used by apply_tiled().
   */
    DirectCiResponse apply_with_adjoint(const Eigen::Ref<const Eigen::MatrixXd>& response,
                                        const Eigen::Ref<const Eigen::MatrixXd>& state,
                                        int exchange_parity = 0) const;

    /**
   * @brief Unsymmetrized one-body adjoint for an orbital-generator action.
   *
   * The returned matrix `D` satisfies
   * `left.dot(E(kappa) * right) = D.cwiseProduct(kappa).sum()` for a general,
   * not necessarily symmetric one-body generator `kappa`.
   */
    Eigen::MatrixXd
    one_body_generator_adjoint(const Eigen::Ref<const Eigen::MatrixXd>& left,
                               const Eigen::Ref<const Eigen::MatrixXd>& right) const;

    /** @brief Sum oriented generator adjoints from a bounded left beta panel.
   * Left packs beta_count columns per RHS; right arguments pack the full beta
   * range per RHS. Each request contributes scale*D(left,right), or its
   * transpose. Multiple requests share the signed-link traversal. Only small
   * orbital matrices are thread-local; no coefficient matrix is copied.
   * upper_only restricts the final cotangent, after each requested transpose.
   */
    Eigen::MatrixXd
    generator_adjoint_columns(StringIndex beta_begin,
                              std::int64_t beta_count,
                              const Eigen::Ref<const Eigen::MatrixXd>& left,
                              std::initializer_list<DirectCiGeneratorAdjointRequest> requests,
                              bool upper_only = false) const;

    /** @brief Upper orbital cotangent against generated right columns.
   * Each request adds scale*D(left,right), or its transpose. The requested
   * right arguments share the link traversal and use bounded shared beta
   * tiles and spectator gathers, never a full product carrier. Readers have
   * one RHS and are called serially. Immutable retained columns are borrowed;
   * transient views are copied before another read and parallel contraction.
   * The caller bounds the left panel width. symmetric_sum returns
   * upper(D+D^T), after all requested orientations and scales. This is the
   * Cholesky pullback, not an oriented full one-body density.
   * spin_exchange_reduced emits twice the alpha contribution and omits beta
   * spectators. This is exact only after summing ALL left panels, when each
   * full left/right pair has the same exchange parity in identical spin spaces.
   * An individual panel need not have that parity or equal its unreduced image.
   */
    Eigen::MatrixXd
    generator_adjoint_streamed(StringIndex beta_begin,
                               const Eigen::Ref<const Eigen::MatrixXd>& left,
                               std::initializer_list<DirectCiStreamedGeneratorRequest> requests,
                               bool symmetric_sum = false,
                               bool spin_exchange_reduced = false) const;

    /** @brief Independent left-panel adjoints sharing generated right arguments.
   * Every left has the same shape. Readers, spectator gathers and signed links
   * are shared; each returned orbital cotangent remains independent. Only the
   * small orbital reductions scale with the number of left panels.
   */
    std::vector<Eigen::MatrixXd>
    generator_adjoint_streamed(StringIndex beta_begin,
                               std::span<const Eigen::Ref<const Eigen::MatrixXd>> lefts,
                               std::initializer_list<DirectCiStreamedGeneratorRequest> requests,
                               bool symmetric_sum = false,
                               bool spin_exchange_reduced = false) const;

    /** @brief Numerical matrices plus referenced links, not exclusively owned bytes. */
    std::size_t dynamic_bytes() const noexcept;

  private:
    SpinConnections build_spin_connections(const SpinStrings& basis) const;
    Eigen::MatrixXd build_hamiltonian_pair_kernel(const ActiveIntegrals& integrals,
                                                  int n_electrons) const;
    /** @brief Scatter into (block_width * alpha_count, orbital_pairs), strings contiguous. */
    void scatter_one_body_intermediate(const Eigen::Ref<const Eigen::MatrixXd>& coefficients,
                                       StringIndex alpha_begin,
                                       std::int64_t alpha_count,
                                       std::int64_t beta_target,
                                       Eigen::Ref<Eigen::MatrixXd> intermediate) const;
    const SpinConnections& beta_connections() const noexcept;

    /** @brief Shared channel sweep for sigma, integral adjoints, or both.
   * Non-null sigma receives H * left. A sigma-only request neither constructs
   * a kernel gradient nor scatters/copies the unused right-hand channels.
   */
    DirectCiIntegralAdjoint contract_channels(const Eigen::Ref<const Eigen::MatrixXd>& left,
                                              const Eigen::Ref<const Eigen::MatrixXd>& right,
                                              Eigen::MatrixXd* sigma,
                                              int exchange_parity,
                                              bool collect_integrals) const;

    /** @brief Generated channel graph shared by tiled sigma and its integral VJP. */
    std::vector<DirectCiIntegralAdjoint>
    contract_tiled(int block_width,
                   std::span<const DirectCiTileRequest> requests,
                   const DirectCiTileReader& state,
                   bool collect_integrals,
                   const ProductSupport* support,
                   int exchange_parity,
                   const DirectCiColumnRelease& release = {},
                   std::size_t workspace_bytes = 64ULL * 1024 * 1024,
                   bool source_projected_image = false) const;

    /** @brief Cooperative sigma contraction inside the current OpenMP team.
   * Each output beta column has one owner. All workers must call this method.
   */
    void apply_sigma_tile(StringIndex alpha_begin,
                          std::int64_t alpha_count,
                          int block_begin,
                          int block_count,
                          StringIndex beta_begin,
                          std::int64_t beta_count,
                          int exchange_parity,
                          const Eigen::Ref<const Eigen::MatrixXd>& scattered,
                          Eigen::Ref<Eigen::MatrixXd> workspace,
                          Eigen::Ref<Eigen::MatrixXd> sigma) const;

    int n_orbitals_ = 0;
    StringIndex n_alpha_ = 0;
    StringIndex n_beta_ = 0;
    Eigen::MatrixXd pair_kernel_;
    Eigen::MatrixXd hamiltonian_pair_kernel_;
    std::shared_ptr<const SpinConnections> alpha_;
    std::shared_ptr<const SpinConnections> distinct_beta_;
    std::shared_ptr<detail::SigmaSchedule> schedule_;
};

} // namespace xmvb::mcscf
