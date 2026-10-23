#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include <Eigen/Core>
#include "mcscf/ci/product_index.hpp"
#include "core/workspace.hpp"

#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "integrals/active/two_electron.hpp"
#include "vbscf/structures/assembly/connection.hpp"
#include "vbscf/structures/expansion/types.hpp"
#include "vbscf/structures/overlap/representation.hpp"

namespace xmvb::mcscf {
class DirectCiSigmaAction;
class ProductSupport;
struct DirectCiIntegralAdjoint;
struct SpinFactorSource;
class FactorActionWorkspace;
}

namespace xmvb::vb {
namespace detail { struct ProductImages; }

class ExteriorOrbitalTransform;
struct ExteriorTransformDirection;
struct OrthogonalActiveIntegrals;

class StructureMetricInverse;
class SpinMetric;
struct SpinSources;

/** @brief Davidson needs a diagonal model; fixed-state actions and adjoints do not. */
enum class StructureActionMode { Davidson, ActionOnly };

/** @brief Tile replay counts and retained panel payload (not process peak RSS). */
struct StructureTileWork {
  std::size_t expansions = 0, structure_projections = 0;
  std::size_t input_panels = 0, hamiltonian_projections = 0, overlap_projections = 0;
  std::size_t input_evictions = 0, output_evictions = 0;
  std::size_t input_resident_bytes = 0, output_resident_bytes = 0;
  std::size_t source_tiles = 0, source_columns = 0, channel_blocks = 0;
  std::size_t source_workspace_bound = 0;
  /** Generated sigma sweeps that also collect the integral cotangent. */
  std::size_t integral_adjoint_sweeps = 0;
};

/** @brief Structure H/S images, each of shape `(n_structures, block_width)`.
 * No dense structure-by-structure matrix is formed.
 */
struct StructureActionResult {
  Eigen::MatrixXd hamiltonian;
  Eigen::MatrixXd overlap;
  StructureTileWork tiles;
};

/** @brief Active-integral adjoint of selected structure-space states. */
struct StructureActiveIntegralAdjoint {
  Eigen::MatrixXd overlap;
  Eigen::MatrixXd one_electron;
  Eigen::MatrixXd pair_kernel;
};

/** @brief Shared response sweep; retains only structure images and active tensors.
 * Spin-product workspaces are consumed one state at a time, not retained here.
 */
struct StructureResponseSweep {
  StructureActionResult images;
  Eigen::MatrixXd one_electron;
  Eigen::MatrixXd pair_kernel;
  /** Cholesky-generator cotangent; only its upper triangle is physical. */
  Eigen::MatrixXd generator;
};

/** @brief Borrowed CI direction and its independently accumulated response. */
struct StructureResponseRequest {
  const Eigen::MatrixXd& coefficients;
  StructureResponseSweep result{};
};

/** @brief Direction-independent selected-state data for direct-CI adjoints. */
struct StructureAdjointState {
  /** Accepted coefficients in the originating selected structure space. */
  Eigen::MatrixXd structure_coefficients;
  /** Exact source representation; accepted orthogonal/residual arrays are absent. */
  bool source_factors = false;
  /** Shared, bounded accepted panels; never a full-carrier mandate. */
  std::shared_ptr<detail::ProductImages> product_images;
  std::vector<double> weights;
  std::vector<double> energies;
  Eigen::MatrixXd one_electron_gradient;
  Eigen::MatrixXd pair_kernel_gradient;
  Eigen::MatrixXd generator_gradient;
  /** Upper-triangular accepted-state transition 1-RDMs for gauge multipliers. */
  std::vector<Eigen::MatrixXd> overlap_generators;
};

/** @brief Direct-CI intermediates shared by structure and adjoint directions. */
struct StructureIntegralDirection {
  /** Prepared once by the forward direction; borrowed by its reverse consumer. */
  std::shared_ptr<const OrthogonalActiveIntegrals> integrals;
  /** Weighted generator-gradient direction accumulated before source images are released. */
  Eigen::MatrixXd generator_direction;
  /** Integral cotangents from the same orbital-direction tile sweep. */
  Eigen::MatrixXd one_electron_direction, pair_kernel_direction;
  /** Full-trace diagnostic request; never changes the physical action. */
  bool capture_connection = false;
  std::optional<StructureConnectionDirection> connection;
};

struct StructureActionStorage {
  /** Pair providers, active integrals, and spin strings retained by the action. */
  std::size_t factor_bytes = 0;
  /** Unique-string-pair/structure expansion in contiguous CSR form. */
  std::size_t expansion_bytes = 0;
  /** Hamiltonian and overlap diagonal model retained by Davidson. */
  std::size_t diagonal_bytes = 0;
  /** Whether complete one-spin carriers permit orthogonal direct CI.
   * The selected VB structure space itself need not be complete.
   */
  bool orthogonal_direct_ci = false;
  /** Persistent bytes owned by the orthogonal direct-CI representation. */
  std::size_t direct_ci_bytes = 0;
};

/**
 * @brief Matrix-free structure-space Hamiltonian and overlap action.
 *
 * Let `C` map structure coefficients directly onto unique alpha/beta string
 * products, and reshape `X = C V` over those two spin spaces. Complete one-spin
 * carriers use the exact orthogonal-CI congruence `C^T T^T H_orth T C`, with
 * no closure assumption on the selected structure space. Its contraction uses
 * minimum row/column source factors or generated product tiles, sharing sparse
 * Slater--Condon spin links and bounded numerical slabs. Incomplete one-spin
 * carriers use the mathematically distinct accepted-pair representation.
 *
 * H/M application introduces no determinant coefficient vector, complete
 * transformed product array, or dense structure matrix. The constructed action
 * owns its expansion, numerical data and shared support plans independently
 * of the construction inputs. Derivative preparation has separate storage
 * contracts; bounded H/M application does not imply bounded derivative storage.
 */
class StructureAction {
public:
  /** @brief One entry of the selected-structure to touched-product map. */
  struct StructureTerm {
    int group = 0;
    int structure = 0;
    double coefficient = 0.0;
  };

  /** @brief Borrowed exact orthogonal action data, independent of an executor.
   * All views remain valid for the lifetime of this action. Products use
   * alpha*n_beta+beta addresses; no determinant-indexed numerical array exists.
   */
  struct OrthogonalView {
    const ExteriorOrbitalTransform& alpha;
    const ExteriorOrbitalTransform& beta;
    const mcscf::DirectCiSigmaAction& sigma;
    std::span<const mcscf::ProductIndex> products;
    std::span<const std::size_t> offsets;
    std::span<const StructureTerm> terms;
    int exchange_parity;
    const mcscf::ProductSupport* support;
  };

  /** @brief Export the retained orthogonal representation without rebuilding it. */
  OrthogonalView orthogonal_view() const;

  StructureAction(
      StructureIncidence determinant_to_structure_terms,
      int n_structures,
      const SameSpinPairCacheContext& same_spin_pair_cache,
      const std::vector<double>& active_overlap,
      const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron,
      const xmvb::integrals::ActiveSpaceTwoElectronResult& active_space_two_electron_result,
      int n_active_orbitals,
      const StructureDiagonal* precomputed_preconditioner = nullptr,
      StructureActionMode mode = StructureActionMode::Davidson);

  ~StructureAction();
  StructureAction(StructureAction&&) noexcept;
  StructureAction& operator=(StructureAction&&) noexcept;
  StructureAction(const StructureAction&) = delete;
  StructureAction& operator=(const StructureAction&) = delete;

  /**
   * @brief Applies both structure matrices to one or more vectors.
   *
   * @param vectors Dense `(n_structures, block_width)` input block.
   * @return Hamiltonian and overlap images with the same shape.
   */
  StructureActionResult apply(
      const Eigen::Ref<const Eigen::MatrixXd>& vectors) const;

  /** @brief Hamiltonian image when the caller already owns the selected metric.
   * Product tiles omit the redundant overlap back-projection.
   */
  Eigen::MatrixXd apply_hamiltonian(
      const Eigen::Ref<const Eigen::MatrixXd>& vectors) const;

  /** @brief Exact overlap image; direct-CI needs only forward/adjoint transforms.
   * Shares the accepted-point representation and the existing block budget.
   * Pair-based representations retain their exact combined action.
   */
  Eigen::MatrixXd apply_overlap(
      const Eigen::Ref<const Eigen::MatrixXd>& vectors) const;

  /**
   * @brief Applies H/S with bounded spin-product and pair-tile workspaces.
   *
   * Complete carriers use the orthogonal panel pipeline unconditionally.
   * Incomplete carriers use accepted-pair tiles. Neither path materializes
   * a full `U_alpha x U_beta` intermediate in this entry point.
   */
  StructureActionResult apply_streamed(
      const Eigen::Ref<const Eigen::MatrixXd>& vectors,
      std::size_t workspace_bytes = workspace::available()) const;

  /** @brief Exact H/M through minimum row/column source factors and channel tiles.
   * Arbitrary selected structures need no closure. This explicit contraction
   * order also permits independent comparisons against the product-tile order.
   */
  StructureActionResult apply_sources(
      const Eigen::Ref<const Eigen::MatrixXd>& vectors,
      std::size_t workspace_bytes = 64ULL * 1024ULL * 1024ULL) const;

  /**
   * @brief Applies the exact first derivative of a complete-space direct-CI action.
   *
   * The selected vectors in the prepared state are held fixed while all active
   * overlap and integral tensors are differentiated. This operation is
   * available only when the retained representation is the orthogonal
   * direct-CI form.
   */
  StructureActionResult apply_integral_direction(
      const StructureAdjointState& state,
      const std::vector<double>& overlap_direction,
      const std::vector<double>& one_electron_direction,
      const std::vector<double>& packed_two_electron_direction,
      StructureIntegralDirection* direct_ci_direction,
      StructureResponseRequest* ci_response=nullptr,
      const Eigen::MatrixXd* ri_factor_direction=nullptr) const;

  /** @brief Whether exact orthogonal direct-CI directional actions are available. */
  bool supports_integral_direction() const noexcept;

  /** @brief Precomputes direction-independent selected-state adjoint data.
   * Optional H/M images are consumed from the same sigma sweep, not replayed.
   * A supplied overlap_image is the exact M*C for these input columns and
   * avoids projecting the same overlap again; it is not a CI residual estimate.
   */
  StructureAdjointState prepare_active_adjoint(
      const Eigen::Ref<const Eigen::MatrixXd>& structure_coefficients,
      const std::vector<double>& state_weights,
      const std::vector<double>& state_energies,
      StructureActionResult* images = nullptr,
      const Eigen::MatrixXd* overlap_image = nullptr) const;

  /** @brief Pulls a prepared direct-CI adjoint back to active integrals. */
  StructureActiveIntegralAdjoint active_integral_adjoint(
      const StructureAdjointState& state) const;

  /**
   * @brief Differentiates the selected-state active-integral adjoint.
   *
   * `directional_state_multipliers` is the full multiplier @f$M@f$ in
   * @f$HZ-SZ\Lambda+SCM@f$. Equal-weight selected states therefore retain
   * their off-diagonal horizontal-gauge coupling.
   * `coefficient_response` contains columns in this action's selected
   * structure space. Orbital representation directions are supplied separately
   * and need not lie in the selected VB subspace.
   * Coefficient and multiplier directions are independent; either may be null.
   */
  StructureActiveIntegralAdjoint active_integral_adjoint_direction(
      const StructureAdjointState& state,
      const Eigen::MatrixXd* coefficient_response,
      const Eigen::MatrixXd* directional_state_multipliers,
      const std::vector<double>& overlap_direction,
      const std::vector<double>& one_electron_direction,
      const std::vector<double>& packed_two_electron_direction,
      const StructureIntegralDirection* direct_ci_direction,
      bool include_integral_response,
      const Eigen::MatrixXd* ri_factor_direction=nullptr) const;

  /**
   * @brief Applies the selected-subspace coupling adjoint to active integrals.
   *
   * `coefficient_response` stores the structure coefficient block @f$Z@f$ and
   * `state_multipliers` stores the full selected-state multiplier @f$M@f$ in
   * the accepted-state ordering.  The latter must remain a matrix: within an
   * equal-weight state cluster its off-diagonal entries enforce the horizontal
   * gauge and contribute through @f$C M@f$.  Replacing it by directional
   * eigenvalues would retain only the diagonal part of @f$B^T z@f$.
   */
  StructureActiveIntegralAdjoint active_integral_response_adjoint(
      const StructureAdjointState& state,
      const Eigen::Ref<const Eigen::MatrixXd>& coefficient_response,
      const Eigen::Ref<const Eigen::MatrixXd>& state_multipliers) const;

  /** @brief Computes Hq, Mq and the response adjoint in one direct-CI sweep. */
  StructureResponseSweep apply_response(
      const StructureAdjointState& state,
      const Eigen::Ref<const Eigen::MatrixXd>& coefficients) const;

  /** @brief Adds the horizontal-gauge multiplier and pulls back active tensors. */
  StructureActiveIntegralAdjoint finish_response(
      const StructureAdjointState& state,
      StructureResponseSweep sweep,
      const Eigen::Ref<const Eigen::MatrixXd>& state_multipliers) const;

  /**
   * @brief Expands structure vectors directly onto unique spin products.
   *
   * The returned shape is
   * `(n_unique_alpha, block_width * n_unique_beta)`.
   */
  Eigen::MatrixXd expand_structure_block(
      const Eigen::Ref<const Eigen::MatrixXd>& vectors) const;

  /**
   * @brief Expands structure vectors only on one spin-product tile.
   *
   * Horizontal blocks in the returned matrix have shape
   * `(alpha_size, beta_size)`.  This is the bounded-memory counterpart of
   * `expand_structure_block()` used by the large-space factorized action.
   */
  Eigen::MatrixXd expand_structure_tile(
      const Eigen::Ref<const Eigen::MatrixXd>& vectors,
      int alpha_begin,
      int alpha_size,
      int beta_begin,
      int beta_size) const;

  /**
   * @brief Maps structure columns to the orthonormal determinant-product basis.
   *
   * The returned matrix packs one `(n_unique_alpha, n_unique_beta)` block per
   * input column.  This operation is defined only for the complete-space
   * direct-CI representation.
   */
  Eigen::MatrixXd orthogonalize_structure_block(
      const Eigen::Ref<const Eigen::MatrixXd>& vectors) const;

  /** @brief Representation-only one-body adjoint; never applies the Hamiltonian. */
  Eigen::MatrixXd carrier_generator_adjoint(
      const Eigen::Ref<const Eigen::MatrixXd>& left,
      const Eigen::Ref<const Eigen::MatrixXd>& right) const;

  /** @brief Whether the carrier diagonal and one-column workspace fit the action budget. */
  bool supports_carrier_diagonal() const noexcept;

  /** @brief Physical Ritz residual in the orthogonal spin-string carrier.
   * Valid as a structure-space certificate only for a complete variational
   * spin sector, not an arbitrary selected structure subspace. The caller
   * must establish this completeness. Uses one carrier column, no dense M.
   */
  double complete_space_residual_norm(
      const Eigen::Ref<const Eigen::VectorXd>& coefficients, double energy) const;

  /** @brief Applies inverse exterior transforms to packed spin products in place. */
  void inverse_orbital_transform(Eigen::MatrixXd* spin_products) const;
  /** @brief Applies inverse-adjoint exterior transforms to spin products in place. */
  void inverse_adjoint_orbital_transform(Eigen::MatrixXd* spin_products) const;

  /**
   * @brief Contracts packed unique-string-product images into structures.
   *
   * Each horizontal block has shape `(n_unique_alpha, n_unique_beta)`.
   * The result contains one structure-space column per block and uses the
   * retained direct unique-string-pair/structure expansion.
   */
  Eigen::MatrixXd contract_spin_product_block(
      const Eigen::Ref<const Eigen::MatrixXd>& spin_images) const;

  /** @brief Selected cross Gram action in a bounded exact contraction order.
   * Uses touched beta factors or one shared carrier for broad support.
   * Computes F_left^T F_right V without enlarging the selected structure space.
   * If all four directions are supplied, returns its directional derivative.
   * The workspace budget controls panel/RHS batching, never the physical space.
   */
  Eigen::MatrixXd cross_overlap(
      const Eigen::Ref<const Eigen::MatrixXd>& vectors,
      const ExteriorOrbitalTransform& left_alpha,
      const ExteriorOrbitalTransform& left_beta,
      const ExteriorOrbitalTransform& right_alpha,
      const ExteriorOrbitalTransform& right_beta,
      const ExteriorTransformDirection* delta_left_alpha = nullptr,
      const ExteriorTransformDirection* delta_left_beta = nullptr,
      const ExteriorTransformDirection* delta_right_alpha = nullptr,
      const ExteriorTransformDirection* delta_right_beta = nullptr,
      std::size_t workspace_bytes = 64ULL * 1024ULL * 1024ULL) const;

  /** @brief Bounded pullback of left^T F(left_chart)^T F(right_chart) right.
   * Broad-support common spatial charts use direct one-body generator moments.
   * Optional chart directions request the derivative of the pullback.
   */
  UpperPair cross_overlap_adjoint(const Eigen::VectorXd& left,
      const Eigen::VectorXd& right, const ExteriorOrbitalTransform& la,
      const ExteriorOrbitalTransform& lb, const ExteriorOrbitalTransform& ra,
      const ExteriorOrbitalTransform& rb, const Eigen::MatrixXd* delta_left = nullptr,
      const Eigen::MatrixXd* delta_right = nullptr,
      std::size_t workspace_bytes = 64ULL * 1024ULL * 1024ULL,
      OverlapSide side = OverlapSide::Both) const;

  /** @brief Shared finite-transport VJP and reference coefficient image. */
  OverlapTransportGradient cross_overlap_transport_gradient(
      const Eigen::VectorXd& lambda, const Eigen::VectorXd& base,
      const Eigen::VectorXd& reference, const Eigen::VectorXd& transported,
      const ExteriorOrbitalTransform& current_alpha,
      const ExteriorOrbitalTransform& current_beta,
      const ExteriorOrbitalTransform& origin_alpha,
      const ExteriorOrbitalTransform& origin_beta,
      const Eigen::MatrixXd& origin_direction,
      std::size_t workspace_bytes = 64ULL * 1024ULL * 1024ULL) const;

  /**
   * @brief Contracts one rectangular unique-string-product image tile.
   *
   * Horizontal blocks in `tile_images` have shape
   * `(alpha_size, beta_size)`. Contributions are accumulated directly into
   * `structure_images`; no full alpha-by-beta image is materialized.
   */
  void add_spin_product_tile(
      const Eigen::Ref<const Eigen::MatrixXd>& tile_images,
      int alpha_begin,
      int beta_begin,
      Eigen::MatrixXd* structure_images) const;

  /** @brief H/S preconditioning model; empty for an ActionOnly instance. */
  const StructureDiagonal& preconditioner_diagonal() const noexcept;

  int n_determinants() const noexcept;
  int n_structures() const noexcept;
  int n_alpha_strings() const noexcept { return n_unique_alpha_; }
  int n_beta_strings() const noexcept { return n_unique_beta_; }

  /** @brief Reports the retained factor storage and channel representations. */
  StructureActionStorage storage() const noexcept;

private:
  using FactorImageConsumer = std::function<void(const Eigen::Ref<const Eigen::MatrixXd>&,
      const Eigen::Ref<const Eigen::MatrixXd>&, double)>;
  struct SourceImageAdjoint {
    StructureActionResult images;
    Eigen::MatrixXd generator;
  };
  /** @brief Expand one structure vector only on original touched products. */
  Eigen::VectorXd expand_support(
      const Eigen::Ref<const Eigen::VectorXd>& coefficients) const;
  /** @brief Oriented transition density from structure coefficients and spin factors. */
  Eigen::MatrixXd source_generator_adjoint(
      const StructureAdjointState& state, int left, int right) const;
  /** @brief Bounded resident panels through expansion, sigma and projection. */
  StructureActionResult apply_orthogonal_tiled(
      const Eigen::Ref<const Eigen::MatrixXd>& vectors,
      const ExteriorOrbitalTransform& alpha_transform,
      const ExteriorOrbitalTransform& beta_transform,
      const mcscf::DirectCiSigmaAction& sigma,
      bool overlap_only = false,
      std::size_t workspace_bytes = workspace::available()) const;
  /** @brief Generated accepted/tangent jet with immediate shared consumers. */
  StructureResponseSweep product_jet(const StructureAdjointState& state,
      const Eigen::MatrixXd* response, const Eigen::MatrixXd* delta_upper,
      const mcscf::DirectCiSigmaAction* delta_sigma,
      const ExteriorOrbitalTransform& alpha,const ExteriorOrbitalTransform& beta,
      const mcscf::DirectCiSigmaAction& sigma,bool collect_images,
      StructureIntegralDirection* diagnostic=nullptr,
      StructureResponseSweep* ci_sweep=nullptr,
      const Eigen::MatrixXd* overlap_image=nullptr) const;

  /** @brief Shared generated product graph for structure H/M actions. */
  StructureResponseSweep contract_product_tiles(
      const Eigen::Ref<const Eigen::MatrixXd>& vectors,
      const ExteriorOrbitalTransform& alpha,const ExteriorOrbitalTransform& beta,
      const mcscf::DirectCiSigmaAction& sigma,bool overlap_only,
      std::size_t workspace_bytes,bool collect_overlap=true) const;
  StructureActionResult apply_source_tiles(
      const Eigen::Ref<const Eigen::MatrixXd>& vectors,
      const ExteriorOrbitalTransform& alpha_transform,
      const ExteriorOrbitalTransform& beta_transform,
      const mcscf::DirectCiSigmaAction& sigma, bool overlap_only,
      std::size_t workspace_bytes) const;
  /** @brief Shared source sweep over already expanded original-support columns. */
  StructureActionResult apply_source_columns(int width,
      const std::function<Eigen::VectorXd(int)>& support_column,
      const ExteriorOrbitalTransform& alpha, const ExteriorOrbitalTransform& beta,
      const mcscf::DirectCiSigmaAction& sigma, bool overlap_only,
      std::size_t workspace_bytes) const;
  /** @brief Project one factor image; an optional generator acts on its output. */
  StructureActionResult project_factor_source(const mcscf::SpinFactorSource& source,
      mcscf::FactorActionWorkspace& workspace,
      const ExteriorOrbitalTransform& alpha, const ExteriorOrbitalTransform& beta,
      const mcscf::DirectCiSigmaAction& sigma, bool overlap_only,
      const Eigen::MatrixXd* output_generator = nullptr,
      const FactorImageConsumer& image_consumer = {}, bool project_output = true,
      double residual_energy = 0.0) const;
  /** @brief Shared H projection and (H-E) generator VJP; no separate overlap sweep. */
  SourceImageAdjoint project_source_adjoint(const mcscf::SpinFactorSource& source,
      const mcscf::SpinFactorSource& right,
      const ExteriorOrbitalTransform& alpha, const ExteriorOrbitalTransform& beta,
      const mcscf::DirectCiSigmaAction& sigma, const Eigen::MatrixXd* output_generator,
      bool project_output = true, double residual_energy = 0.0) const;
  /** @brief Direction H/M images without accepted or directional product arrays. */
  StructureActionResult source_integral_action(const StructureAdjointState& state,
      const Eigen::MatrixXd& generator,
      const ExteriorOrbitalTransform& alpha, const ExteriorOrbitalTransform& beta,
      const mcscf::DirectCiSigmaAction& sigma,
      const mcscf::DirectCiSigmaAction& delta_sigma,
      const OrthogonalActiveIntegrals& delta,
      StructureIntegralDirection* direction) const;
  /** @brief Complete source-factor gradient direction, including the moving pullback. */
  StructureActiveIntegralAdjoint source_adjoint_direction(const StructureAdjointState& state,
      const Eigen::Ref<const Eigen::MatrixXd>& response, const Eigen::MatrixXd* multipliers,
      const OrthogonalActiveIntegrals& integrals, const OrthogonalActiveIntegrals& delta,
      const ExteriorOrbitalTransform& alpha, const ExteriorOrbitalTransform& beta,
      const mcscf::DirectCiSigmaAction& sigma, bool include_integrals,
      const StructureIntegralDirection* cached_direction) const;
  /** @brief Weight-symmetrized gauge shared by response and orbital directions. */
  void add_response_gauge(const StructureAdjointState& state,
      const Eigen::Ref<const Eigen::MatrixXd>& multipliers,
      Eigen::Ref<Eigen::MatrixXd> generator) const;
  /** @brief Includes both residual-generator contractions and the H/M images. */
  bool prefer_factor_response(const mcscf::DirectCiSigmaAction& sigma) const;
  /** @brief Exact overlap order from shear/dot work and shared-image storage.
   * Zero extension to the product workspace does not require selected closure.
   */
  bool use_product_overlap(const ExteriorOrbitalTransform& alpha,
      const ExteriorOrbitalTransform& beta, std::size_t workspace_bytes,
      int images) const;
  /** @brief Coupled coefficient response directly from structure-space columns. */
  StructureResponseSweep source_response(const StructureAdjointState& state,
      const Eigen::Ref<const Eigen::MatrixXd>& coefficients,
      const ExteriorOrbitalTransform& alpha, const ExteriorOrbitalTransform& beta,
      const mcscf::DirectCiSigmaAction& sigma, bool collect_images = true) const;
  /** @brief Compares algebraic operation counts; independent of memory budgets. */
  bool prefer_source_contraction(const ExteriorOrbitalTransform& alpha_transform,
      const ExteriorOrbitalTransform& beta_transform,
      const mcscf::DirectCiSigmaAction& sigma) const;
  StructureResponseSweep contract_response(
      const StructureAdjointState& state,
      const Eigen::Ref<const Eigen::MatrixXd>& coefficients,
      bool collect_images) const;
  friend class StructureMetricInverse;
  friend class SpinMetric;
  /** @brief Sigma in the orthogonal carrier; inner contraction remains tiled. */
  Eigen::MatrixXd apply_carrier_hamiltonian(const Eigen::Ref<const Eigen::MatrixXd>& carrier) const;
  void adjoint_orbital_transform(Eigen::MatrixXd* carrier) const;
  /** @brief Shared in-place carrier denominator, with the existing signed floor. */
  void divide_carrier_diagonal(Eigen::Ref<Eigen::MatrixXd> carrier,
      const Eigen::Ref<const Eigen::VectorXd>& energies,double eigenvalue) const;
  struct OrthogonalDirectCiData;
  struct TiledPairData;

  /** @brief Exact common transpose parity of all sparse structure columns. */
  int detect_spin_exchange() const;
  int spin_exchange_parity_ = 0;
  bool source_contraction_ = false;
  std::unique_ptr<const SpinSources> sources_;

  std::vector<mcscf::ProductIndex> spin_products_;
  std::vector<std::size_t> spin_term_offsets_;
  std::vector<StructureTerm> spin_terms_;
  /** Structure-major permutation of `spin_terms_` for race-free C^T actions. */
  std::vector<std::size_t> structure_term_offsets_;
  std::vector<std::size_t> structure_term_indices_;

  StructureDiagonal preconditioner_diagonal_;
  int n_determinants_ = 0;
  int n_structures_ = 0;
  int n_unique_alpha_ = 0;
  int n_unique_beta_ = 0;
  std::unique_ptr<OrthogonalDirectCiData> direct_ci_;
  std::unique_ptr<const mcscf::ProductSupport> product_support_;
  std::unique_ptr<TiledPairData> tiled_pairs_;
};

}  // namespace xmvb::vb
