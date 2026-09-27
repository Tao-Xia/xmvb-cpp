#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"
#include "vbscf/structures/assembly/selected_coefficients.hpp"
#include "vbscf/structures/expansion/types.hpp"

namespace xmvb::vb {

/**
 * @brief Result of applying the structure Hamiltonian and overlap to a vector block.
 *
 * Both matrices have shape `(n_structures, block_width)`. No dense
 * `n_structures x n_structures` matrix is formed.
 */
struct StructureActionResult {
  Eigen::MatrixXd hamiltonian;
  Eigen::MatrixXd overlap;
};

/** @brief Active-integral adjoint of selected structure-space states. */
struct StructureActiveIntegralAdjoint {
  Eigen::MatrixXd overlap;
  Eigen::MatrixXd one_electron;
  Eigen::MatrixXd pair_kernel;
};

/** @brief Direction-independent selected-state data for direct-CI adjoints. */
struct StructureAdjointState {
  std::vector<Eigen::MatrixXd> source_coefficients;
  std::vector<Eigen::MatrixXd> orthogonal_coefficients;
  std::vector<Eigen::MatrixXd> residuals;
  std::vector<double> weights;
  std::vector<double> energies;
  Eigen::MatrixXd one_electron_gradient;
  Eigen::MatrixXd pair_kernel_gradient;
  Eigen::MatrixXd generator_gradient;
};

/** @brief Direct-CI intermediates shared by structure and adjoint directions. */
struct StructureIntegralDirection {
  Eigen::MatrixXd coefficient_direction;
  Eigen::MatrixXd sigma_direction;
};

struct StructureActionStorage {
  /** Payload retained by the spin-factorized action. */
  std::size_t factor_bytes = 0;
  /** Unique-string-pair/structure expansion in contiguous CSR form. */
  std::size_t expansion_bytes = 0;
  /** Hamiltonian and overlap diagonal model retained by Davidson. */
  std::size_t diagonal_bytes = 0;
  /** Opposite-spin channels whose raw side is stored densely. */
  int dense_channels = 0;
  /** Opposite-spin channels whose raw side is stored as exact nonzeros. */
  int sparse_channels = 0;
  /** Opposite-spin channels fused into one support-factor family. */
  int factored_channels = 0;
  /** Exact nonzeros before a sparse channel is optionally densified. */
  std::size_t channel_nonzeros = 0;
  /** Dense values required to store the same active channel matrices. */
  std::size_t channel_dense_values = 0;
  /** Whether the action uses the complete-space orthogonal direct-CI form. */
  bool orthogonal_direct_ci = false;
  /** Persistent bytes owned by the orthogonal direct-CI representation. */
  std::size_t direct_ci_bytes = 0;
};

/**
 * @brief Matrix-free structure-space Hamiltonian and overlap action.
 *
 * Let `C` map structure coefficients directly onto unique alpha/beta string
 * products, and reshape `X = C V` over those two spin spaces. The spin-product
 * Hamiltonian action is factorized as
 *
 * `H_alpha X S_beta^T + S_alpha X H_beta^T`
 *
 * plus one factored alpha/beta product per active-orbital pair channel. This
 * replaces the quadratic full-determinant pair traversal with dense spin-space
 * contractions. The sparse `C` and `C^T` contractions are applied without an
 * intermediate determinant vector, and dense structure-space matrices are
 * never allocated. The constructed action owns every expansion, factor, and
 * diagonal needed by `apply()`; its lifetime is independent of the
 * construction inputs.
 */
class StructureAction {
public:
  StructureAction(
      const std::vector<std::vector<StructureExpansionTerm>>&
          determinant_to_structure_terms,
      int n_structures,
      const SameSpinPairCacheContext& same_spin_pair_cache,
      const std::vector<double>& active_overlap,
      const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron,
      const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
      int n_active_orbitals,
      const StructureDiagonal* precomputed_preconditioner = nullptr);

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
      StructureIntegralDirection* direct_ci_direction) const;

  /** @brief Whether exact orthogonal direct-CI directional actions are available. */
  bool supports_integral_direction() const noexcept;

  /** @brief Precomputes direction-independent selected-state adjoint data. */
  StructureAdjointState prepare_active_adjoint(
      const SelectedStateDeterminantMatrices& selected_states,
      const std::vector<double>& state_energies) const;

  /** @brief Pulls a prepared direct-CI adjoint back to active integrals. */
  StructureActiveIntegralAdjoint active_integral_adjoint(
      const StructureAdjointState& state) const;

  /**
   * @brief Differentiates the selected-state active-integral adjoint.
   *
   * `directional_state_multipliers` is the full multiplier @f$M@f$ in
   * @f$HZ-SZ\Lambda+SCM@f$. Equal-weight selected states therefore retain
   * their off-diagonal horizontal-gauge coupling.
   */
  StructureActiveIntegralAdjoint active_integral_adjoint_direction(
      const StructureAdjointState& state,
      const SelectedStateDeterminantMatrices* directional_selected_states,
      const Eigen::MatrixXd* directional_state_multipliers,
      const std::vector<double>& overlap_direction,
      const std::vector<double>& one_electron_direction,
      const std::vector<double>& packed_two_electron_direction,
      const StructureIntegralDirection* direct_ci_direction,
      bool include_integral_response) const;

  /**
   * @brief Applies the selected-subspace coupling adjoint to active integrals.
   *
   * `directional_selected_states` stores the coefficient block @f$Z@f$ and
   * `state_multipliers` stores the full selected-state multiplier @f$M@f$ in
   * the accepted-state ordering.  The latter must remain a matrix: within an
   * equal-weight state cluster its off-diagonal entries enforce the horizontal
   * gauge and contribute through @f$C M@f$.  Replacing it by directional
   * eigenvalues would retain only the diagonal part of @f$B^T z@f$.
   */
  StructureActiveIntegralAdjoint active_integral_response_adjoint(
      const StructureAdjointState& state,
      const SelectedStateDeterminantMatrices& directional_selected_states,
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
   * @brief Maps structure columns to the orthonormal determinant-product basis.
   *
   * The returned matrix packs one `(n_unique_alpha, n_unique_beta)` block per
   * input column.  This operation is defined only for the complete-space
   * direct-CI representation.
   */
  Eigen::MatrixXd orthogonalize_structure_block(
      const Eigen::Ref<const Eigen::MatrixXd>& vectors) const;

  /**
   * @brief Contracts packed unique-string-product images into structures.
   *
   * Each horizontal block has shape `(n_unique_alpha, n_unique_beta)`.
   * The result contains one structure-space column per block and uses the
   * retained direct unique-string-pair/structure expansion.
   */
  Eigen::MatrixXd contract_spin_product_block(
      const Eigen::Ref<const Eigen::MatrixXd>& spin_images) const;

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

  /** @brief Returns the H/S diagonal model used only for preconditioning. */
  const StructureDiagonal& preconditioner_diagonal() const noexcept;

  int n_determinants() const noexcept;
  int n_structures() const noexcept;

  /** @brief Reports the retained factor storage and channel representations. */
  StructureActionStorage storage() const noexcept;

private:
  struct OrthogonalDirectCiData;
  struct TiledPairData;
  struct StructureTerm {
    int structure = 0;
    double coefficient = 0.0;
  };

  struct OppositeSpinChannel {
    Eigen::MatrixXd projected;
    Eigen::MatrixXd dense;
    std::vector<Eigen::Triplet<double>> sparse;
  };

  struct SupportedChannelFamily {
    std::vector<Eigen::MatrixXd> projected;
    std::vector<int> offsets;
    std::vector<int> support;
    Eigen::MatrixXd raw;
    bool enabled() const noexcept { return !projected.empty(); }
  };

  void build_opposite_spin_channels(
      const std::vector<SpinDeterminantPairEvaluation>& alpha_pair_cache,
      const std::vector<SpinDeterminantPairEvaluation>& beta_pair_cache,
      int n_packed_pairs);
  void add_supported_channel_block(
      const SupportedChannelFamily& family,
      bool supports_rows,
      const Eigen::Ref<const Eigen::MatrixXd>& spin_vectors,
      const Eigen::Ref<const Eigen::MatrixXd>& transposed_spin_vectors,
      Eigen::MatrixXd* spin_hamiltonians) const;
  void add_alpha_projected_row_support(
      const SupportedChannelFamily& family,
      const Eigen::Ref<const Eigen::MatrixXd>& transposed_spin_vectors,
      Eigen::MatrixXd* spin_hamiltonians) const;
  void add_alpha_projected_column_support(
      const SupportedChannelFamily& family,
      const Eigen::Ref<const Eigen::MatrixXd>& spin_vectors,
      Eigen::MatrixXd* spin_hamiltonians) const;
  void add_beta_projected_row_support(
      const SupportedChannelFamily& family,
      const Eigen::Ref<const Eigen::MatrixXd>& spin_vectors,
      Eigen::MatrixXd* spin_hamiltonians) const;
  void add_beta_projected_column_support(
      const SupportedChannelFamily& family,
      const Eigen::Ref<const Eigen::MatrixXd>& spin_vectors,
      Eigen::MatrixXd* spin_hamiltonians) const;
  void add_individual_channels(
      const Eigen::Ref<const Eigen::MatrixXd>& spin_vector,
      Eigen::MatrixXd* spin_hamiltonian) const;

  std::vector<int> spin_products_;
  std::vector<std::size_t> spin_term_offsets_;
  std::vector<StructureTerm> spin_terms_;
  StructureDiagonal preconditioner_diagonal_;
  Eigen::MatrixXd alpha_overlap_;
  Eigen::MatrixXd alpha_hamiltonian_;
  Eigen::MatrixXd beta_overlap_;
  Eigen::MatrixXd beta_hamiltonian_;
  std::vector<OppositeSpinChannel> opposite_spin_channels_;
  SupportedChannelFamily row_supported_channels_;
  SupportedChannelFamily column_supported_channels_;
  bool alpha_projection_is_dense_ = false;
  int n_determinants_ = 0;
  int n_structures_ = 0;
  int n_unique_alpha_ = 0;
  int n_unique_beta_ = 0;
  std::size_t channel_nonzeros_ = 0;
  std::size_t channel_dense_values_ = 0;
  std::unique_ptr<OrthogonalDirectCiData> direct_ci_;
  std::unique_ptr<TiledPairData> tiled_pairs_;
};

}  // namespace xmvb::vb
