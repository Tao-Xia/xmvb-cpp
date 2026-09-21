#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <Eigen/Core>

namespace xmvb::vb {

class OrbitalChart;
class SparseParameterLayout;
class StructureAction;
struct AcceptedPointContext;
struct VbScfInput;

/**
 * @brief Selects which HVP sub-components to include.
 *
 * The orbital Hessian-vector product decomposes into three additive
 * contributions (direct core, fixed-upstream pullback, outer response) that
 * can be independently toggled.
 */
struct HvpComponents {
  bool direct_core_response = true;
  bool fixed_upstream_pullback = true;
  bool local_active_response = true;
  bool structure_response = true;
  /** Requested inexact structure-response tolerance; zero keeps final accuracy. */
  double response_relative_residual_tolerance = 0.0;
  /** Apply the current common response space without enriching it. */
  bool freeze_structure_response = false;
};

/**
 * @brief Accepted-point structure-response Schur model.
 *
 * With orbital coupling columns @f$J=B^T W@f$ and projected response
 * inverse @f$K^\dagger=(W^T C W)^\dagger@f$, the response-dependent orbital
 * Hessian is @f$-J K^\dagger J^T@f$.
 */
struct StructureResponseSchurModel {
  std::uint64_t revision = 0;
  Eigen::MatrixXd orbital_couplings;
  Eigen::MatrixXd projected_inverse;

  Eigen::MatrixXd apply(
      const Eigen::Ref<const Eigen::MatrixXd>& orbital_directions) const;
};

/** @brief Basis-invariant absolute-spectrum summary of a Schur model. */
struct ResponseSpectrumSummary {
  int model_rank = 0;
  int rank_90 = 0;
  int rank_99 = 0;
  double effective_rank = 0.0;
  double top_mode_fraction = 0.0;
  double top_5_fraction = 0.0;
  double top_10_fraction = 0.0;
};

/** @brief Orbital block and orbital-to-structure coupling from one forward pass. */
struct OrbitalCouplingAction {
  Eigen::VectorXd orbital_hessian;
  Eigen::MatrixXd scaled_structure_forcing;
};

/**
 * @brief Diagonalizes the compact nonzero spectrum of @f$-J K^\dagger J^T@f$.
 *
 * The diagnostic never assembles an orbital-space Hessian and does not alter
 * or truncate the supplied response model.
 */
ResponseSpectrumSummary summarize_response_spectrum(
    const StructureResponseSchurModel& model);

/**
 * @brief Accepted-point matrix-free orbital second-order operator.
 *
 * This module is the home for exact direct-action orbital Hessian contributions
 * evaluated at one accepted VBSCF point. The accepted-point context caches the
 * relaxed active-space intermediates, structure matrices, eigensystem, and
 * selected-state adjoints so `H v` applications can stay entirely analytic:
 *
 * - the local orbital / integral-transformation chain is differentiated
 *   directly at fixed accepted active-space adjoints;
 * - the outer structure/eigen response is differentiated analytically through
 *   `\delta SSO`, `\delta HHO`, and `\delta GGO`, then pulled back to the
 *   orbital space as `J^T \delta \lambda`.
 *
 */
class ExactHvpOperator {
public:
  struct Diagnostics {
    bool supports_analytic_core_model = false;
    bool outer_response_enabled = false;
    bool used_reduced_curvature_diagonal = false;
    bool has_same_spin_matrix_form = false;
    bool has_opposite_spin_pair_graph = false;
    bool streams_exact_pair_products = false;
    int n_selected_states = 0;
    int n_active_orbitals = 0;
    int n_blocks = 0;
    std::size_t resident_exact_pair_elements = 0;
    std::size_t exact_pair_tile_rows = 0;
    std::size_t apply_count = 0;
    std::size_t batch_apply_count = 0;
    std::size_t structure_response_block_actions = 0;
    std::size_t structure_response_schur_build_count = 0;
    std::size_t structure_response_schur_new_columns = 0;
    int max_structure_response_iterations = 0;
    double max_structure_response_relative_residual = 0.0;
    double total_apply_wall_time_seconds = 0.0;
    double core_setup_wall_time_seconds = 0.0;
    double ao_effective_one_electron_build_wall_time_seconds = 0.0;
    double ao_effective_one_electron_fused_wall_time_seconds = 0.0;
    double active_two_electron_wall_time_seconds = 0.0;
    double ao_effective_one_electron_backprop_wall_time_seconds = 0.0;
    double orbital_backprop_wall_time_seconds = 0.0;
    double fixed_upstream_pullback_wall_time_seconds = 0.0;
    double outer_response_wall_time_seconds = 0.0;
    // Outer response sub-stage timings
    double outer_response_active_space_integrals_wall_time_seconds = 0.0;
    double outer_response_structure_matrices_wall_time_seconds = 0.0;
    double outer_response_eigensystem_wall_time_seconds = 0.0;
    double outer_response_pair_weights_wall_time_seconds = 0.0;
    double outer_response_active_gradient_wall_time_seconds = 0.0;
    double outer_response_local_active_gradient_wall_time_seconds = 0.0;
    double outer_response_structure_active_gradient_wall_time_seconds = 0.0;
    double outer_response_selected_state_rebuild_wall_time_seconds = 0.0;
    double outer_response_same_spin_backward_wall_time_seconds = 0.0;
    double outer_response_opposite_spin_backward_wall_time_seconds = 0.0;
    double outer_response_opposite_spin_packed_gradient_wall_time_seconds = 0.0;
    double outer_response_opposite_spin_alpha_overlap_wall_time_seconds = 0.0;
    double outer_response_opposite_spin_beta_overlap_wall_time_seconds = 0.0;
    double outer_response_orbital_pullback_wall_time_seconds = 0.0;
    double structure_response_schur_wall_time_seconds = 0.0;
    double structure_response_schur_structure_action_wall_time_seconds = 0.0;
    double structure_response_schur_adjoint_wall_time_seconds = 0.0;
  };

  ExactHvpOperator(
      std::shared_ptr<const AcceptedPointContext> accepted_point_context,
      const VbScfInput* current_input,
      SparseParameterLayout parameter_view,
      const OrbitalChart* nonredundant_space);

  ~ExactHvpOperator();

  /**
   * @brief Applies the accepted-point reduced-space second-order model.
   *
   * The returned vector lives in the same reduced coordinates as the input.
   * HvpComponents selects which sub-components (direct core, fixed-upstream
   * pullback, outer response) are included.
   */
  Eigen::VectorXd apply_reduced(
      const Eigen::VectorXd& reduced_direction,
      HvpComponents components = {}) const;

  /**
   * @brief Applies the matrix-free operator to a block of directions.
   *
   * Columns are independent reduced-space directions.  This interface is the
   * semantic foundation for fused block contractions; it never materializes
   * the reduced Hessian.
   */
  Eigen::MatrixXd apply_reduced_batch(
      const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions,
      HvpComponents components = {}) const;

  /**
   * @brief Applies @f$\bar A p@f$ and @f$Bp@f$ without solving a response.
   *
   * The structure forcing columns use @f$z_s=\sqrt{2w_s}q_s@f$ coordinates.
   * The accepted-point forward/integral direction is evaluated only once.
   */
  OrbitalCouplingAction apply_orbital_coupling(
      const Eigen::VectorXd& reduced_direction) const;

  /** @brief Applies @f$B^Tz@f$ through the existing selected-state adjoint. */
  Eigen::VectorXd apply_structure_coupling_adjoint(
      const Eigen::Ref<const Eigen::MatrixXd>& coefficient_response,
      const Eigen::Ref<const Eigen::MatrixXd>& adjoint_multipliers) const;

  /** @brief Returns the canonical lazy accepted-point structure action. */
  const StructureAction& structure_action() const;

  bool supports_analytic_core_model() const noexcept;

  /** @brief Revision invalidating previously sampled relaxed HVP images. */
  std::uint64_t response_model_revision() const noexcept;

  /** @brief Builds the current exact low-rank response Schur model. */
  StructureResponseSchurModel structure_response_schur_model() const;

  Diagnostics diagnostics() const;

private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace xmvb::vb
