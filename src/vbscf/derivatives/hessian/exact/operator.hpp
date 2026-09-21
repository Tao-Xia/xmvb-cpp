#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <Eigen/Core>

namespace xmvb::vb {

class OrbitalChart;
class SparseParameterLayout;
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
 * @brief Accepted-point low-rank structure-response Schur model.
 *
 * With orbital coupling columns @f$J=B^T W@f$ and projected response
 * inverse @f$K^\dagger=(W^T C W)^\dagger@f$, the response-dependent orbital
 * Hessian is @f$-J K^\dagger J^T@f$.
 */
struct ResponseLowRankModel {
  std::uint64_t revision = 0;
  Eigen::MatrixXd orbital_couplings;
  Eigen::MatrixXd projected_inverse;

  Eigen::MatrixXd apply(
      const Eigen::Ref<const Eigen::MatrixXd>& orbital_directions) const;
};

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

  bool supports_analytic_core_model() const noexcept;

  /** @brief Revision invalidating previously sampled relaxed HVP images. */
  std::uint64_t response_model_revision() const noexcept;

  /** @brief Builds the current exact low-rank response Schur model. */
  ResponseLowRankModel response_low_rank_model() const;

  Diagnostics diagnostics() const;

private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace xmvb::vb
