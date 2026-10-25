#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include <Eigen/Core>

#include "vbscf/core/contracts/input.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "vbscf/derivatives/hessian/responses/active_space/integral_direction.hpp"
#include "vbscf/derivatives/hessian/context/response_internal.hpp"
#include "vbscf/derivatives/hessian/responses/orbital/preparation.hpp"
#include "integrals/response/types.hpp"
#include "integrals/response/ri.hpp"
#include "integrals/response/ri_ao.hpp"
#include "integrals/ao/exact.hpp"
#include "integrals/ao/ri_operator.hpp"
#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/layout.hpp"
#include "vbscf/structures/assembly/coefficient_blocks.hpp"

namespace xmvb::vb {

struct AcceptedOrbitalPreparationCache;
struct ActiveSpaceGradientDirection;
struct ActiveResponseView;

struct ExactHvpOperator::State {
    struct PrecomputedDirection;

    State(std::shared_ptr<const AcceptedPointContext> accepted_point_context,
          const VbScfInput* current_input,
          SparseParameterLayout parameter_view,
          const OrbitalChart* nonredundant_space);

    Eigen::VectorXd apply_reduced(const Eigen::VectorXd& reduced_direction,
                                  HvpComponents components) const;

    Eigen::MatrixXd apply_reduced_batch(const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions,
                                        HvpComponents components) const;

    OrbitalCouplingAction apply_orbital_coupling(const Eigen::VectorXd& reduced_direction,
                                                 StructureResponseRequest* ci_response) const;

    Eigen::VectorXd apply_orbital(const Eigen::VectorXd& reduced_direction) const;

    OrbitalCouplingBlockAction
    apply_orbital_coupling_batch(const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions) const;

    StructureResponseBlock
    solve_structure_response_block(const std::vector<Eigen::MatrixXd>& scaled_structure_forcing,
                                   double relative_residual_tolerance) const;

    Eigen::VectorXd apply_structure_coupling_adjoint(
        const Eigen::Ref<const Eigen::MatrixXd>& coefficient_response,
        const Eigen::Ref<const Eigen::MatrixXd>& adjoint_multipliers) const;

    const StructureAction& structure_action() const;
    Eigen::VectorXd
    apply_structure_coupling_adjoint(const StructureActiveIntegralAdjoint& adjoint) const;

    bool supports_analytic_core_model() const noexcept;
    std::uint64_t response_model_revision() const noexcept;
    StructureResponseSchurModel structure_response_schur_model() const;
    Diagnostics diagnostics() const;

  private:
    Eigen::VectorXd pullback_structure_response(const ActiveSpaceGradientDirection& gradient) const;
    Eigen::MatrixXd
    apply_reduced_batch_impl(const Eigen::Ref<const Eigen::MatrixXd>& reduced_directions,
                             HvpComponents components,
                             OrbitalCouplingBlockAction* coupling_output) const;

    const AcceptedOuterResponseContext& outer_response_context() const;

    PrecomputedDirection prepare_direction(Eigen::VectorXd packed_direction) const;

    std::vector<integrals::RiAoProjectedResponse>
    ri_ao_responses(std::span<const PrecomputedDirection> directions,
                    const Eigen::Ref<const Eigen::MatrixXd>& weighted_active_columns,
                    bool fixed_two_electron) const;

    ActiveSpaceGradientDirection active_response_gradient(const ActiveResponseView& response,
                                                          HvpComponents components) const;

    ActiveSpaceIntegralDirectionView
    build_active_integral_direction(const PrecomputedDirection& direction,
                                    const Eigen::Ref<const Eigen::MatrixXd>& delta_h1e_times_active,
                                    const Eigen::VectorXd* precomputed_delta_packed_two_electron,
                                    ActiveSpaceIntegralDirectionWorkspace* workspace,
                                    bool packed_two_electron_required = true) const;

    Eigen::VectorXd apply_reduced_impl(
        const Eigen::VectorXd& reduced_direction,
        HvpComponents components,
        const Eigen::VectorXd* precomputed_ao_response,
        const Eigen::MatrixXd* precomputed_active_projection,
        const Eigen::VectorXd* precomputed_delta_packed_active_two_electron,
        const Eigen::MatrixXd* precomputed_ri_active_pair_factor_direction,
        const xmvb::integrals::ExactCtxPairMatrix* precomputed_directional_pair_products,
        const PrecomputedDirection* precomputed_direction,
        OrbitalCouplingAction* coupling_output,
        bool horizontal_forcing = true,
        StructureResponseRequest* ci_response = nullptr) const;

    Eigen::VectorXd apply_structure_response_adjoint(
        const Eigen::Ref<const Eigen::MatrixXd>& coefficient_response,
        const Eigen::Ref<const Eigen::MatrixXd>& state_multipliers) const;

    struct ApplyTimingTotals {
        std::size_t apply_count = 0;
        std::size_t normalization_adjoint_reuses = 0;
        std::size_t batch_apply_count = 0;
        std::size_t orbital_coupling_batch_count = 0;
        std::size_t orbital_coupling_batch_chunk_count = 0;
        std::size_t max_orbital_coupling_batch_width = 0;
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

    std::shared_ptr<const AcceptedPointContext> accepted_point_context_;
    const VbScfInput* current_input_ = nullptr;
    SparseParameterLayout parameter_view_;
    const OrbitalChart* nonredundant_space_ = nullptr;
    Eigen::MatrixXd accepted_active_auxiliary_orbitals_;
    Eigen::MatrixXd accepted_basis_overlap_times_active_auxiliary_orbitals_;
    Eigen::MatrixXd accepted_sso_gradient_symmetric_;
    Eigen::MatrixXd accepted_hho_gradient_symmetric_;
    /** R (W + W^T), acting directly on delta F Cbar projections. */
    Eigen::MatrixXd accepted_hho_projection_weight_;
    mutable Eigen::MatrixXd ao_h1e_symmetrized_gradient_workspace_;
    mutable xmvb::integrals::AoH1eBatchWorkspace ao_h1e_batch_workspace_;
    mutable integrals::RiAoResponseWorkspace ri_ao_response_workspace_;
    std::shared_ptr<const Eigen::MatrixXd> accepted_ri_inactive_one_index_factors_;
    Eigen::MatrixXd zero_core_hamiltonian_;
    xmvb::integrals::ExactPackedActiveTwoElectronAdjointCache accepted_exact_two_electron_cache_;
    std::optional<xmvb::integrals::RiActiveTwoElectronResponseCache>
        accepted_ri_two_electron_cache_;
    const xmvb::integrals::RiAoFactorization* accepted_ri_factorization_ = nullptr;
    mutable xmvb::integrals::ExactPackedActiveTwoElectronApplyWorkspace
        accepted_exact_two_electron_apply_workspace_;
    mutable ActiveSpaceIntegralDirectionWorkspace outer_response_integral_direction_workspace_;
    mutable std::vector<double> outer_response_symmetric_active_overlap_gradient_workspace_;
    mutable std::vector<double> outer_response_symmetric_active_one_electron_gradient_workspace_;
    std::unique_ptr<AcceptedOrbitalPreparationCache> accepted_orbital_preparation_cache_;
    mutable std::unique_ptr<AcceptedOuterResponseContext> accepted_outer_response_context_;
    mutable std::vector<Eigen::MatrixXd> response_orbital_couplings_by_state_;
    mutable ApplyTimingTotals apply_timing_totals_;
};

} // namespace xmvb::vb
