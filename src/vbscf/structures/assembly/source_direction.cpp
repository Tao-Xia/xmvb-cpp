#include <cstdint>
#include "core/index.hpp"
#include "vbscf/structures/assembly/action.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

#include "mcscf/ci/factors.hpp"
#include "vbscf/structures/assembly/sources.hpp"
#include "vbscf/structures/orthogonal_ci/integrals.hpp"

namespace xmvb::vb {

StructureActionResult
StructureAction::scalar_integral_action(const StructureAdjointState& state,
                                        const OrthogonalActiveIntegrals& delta) const {
    const auto& r = orthogonal_view().alpha.orbital_transform();
    const Eigen::MatrixXd generator = r.transpose()
                                          .triangularView<Eigen::Lower>()
                                          .solve(delta.orbital_transform.transpose())
                                          .transpose();
    const double dm = 2 * (state.overlap_generators[0].array() * generator.array()).sum();
    const double de = ((state.one_electron_gradient.array() * delta.one_electron.array()).sum() +
                       (state.pair_kernel_gradient.array() * delta.pair_kernel.array()).sum() +
                       (state.generator_gradient.array() * generator.array()).sum()) /
                      state.weights[0];
    const double c = state.structure_coefficients(0, 0);
    return {Eigen::MatrixXd::Constant(1, 1, (de + state.energies[0] * dm) / c),
            Eigen::MatrixXd::Constant(1, 1, dm / c)};
}

StructureActionResult
StructureAction::source_integral_action(const StructureAdjointState& state,
                                        const Eigen::MatrixXd& generator,
                                        const ExteriorOrbitalTransform& alpha,
                                        const ExteriorOrbitalTransform& beta,
                                        const mcscf::DirectCiSigmaAction& sigma,
                                        const mcscf::DirectCiSigmaAction& delta_sigma,
                                        const OrthogonalActiveIntegrals& delta,
                                        StructureIntegralDirection* direction) const {
    const int states = state.structure_coefficients.cols();
    if (states <= 0 || state.structure_coefficients.rows() != n_structures_ ||
        state.weights.size() != static_cast<std::size_t>(states) ||
        state.energies.size() != state.weights.size())
        throw std::invalid_argument("invalid source-factor accepted state");
    StructureActionResult result{Eigen::MatrixXd::Zero(n_structures_, states),
                                 Eigen::MatrixXd::Zero(n_structures_, states)};
    const auto add_work = [&](const StructureTileWork& work) {
        result.tiles.expansions += work.expansions;
        result.tiles.structure_projections += work.structure_projections;
        result.tiles.source_tiles += work.source_tiles;
        result.tiles.source_columns += work.source_columns;
        result.tiles.channel_blocks += work.channel_blocks;
        result.tiles.source_workspace_bound =
            std::max(result.tiles.source_workspace_bound, work.source_workspace_bound);
    };
    const Eigen::MatrixXd adjoint_generator = generator.transpose();
    const bool moving = !generator.isZero(0.0);
    const bool capture = direction && direction->capture_connection;
    // For a one-column VB space the H/M images are scalars. Their derivatives
    // follow from the accepted adjoints; no exterior adjoint or CI projection
    // is needed. Connection diagnostics explicitly request the separate images.
    const bool scalar = n_structures_ == 1 && states == 1 && state.weights[0] != 0 &&
                        state.structure_coefficients(0, 0) != 0 && !capture;
    if (direction) {
        direction->generator_direction =
            Eigen::MatrixXd::Zero(sigma.n_orbitals(), sigma.n_orbitals());
        direction->connection.reset();
        if (capture)
            direction->connection = StructureConnectionDirection{result.hamiltonian,
                                                                 result.hamiltonian,
                                                                 result.hamiltonian,
                                                                 Eigen::VectorXd::Zero(states)};
    }
    if (scalar) {
        result = scalar_integral_action(state, delta);
        if (!direction)
            return result;
    }
    const auto project = [&](const mcscf::SpinFactorSource& source,
                             const mcscf::DirectCiSigmaAction& action,
                             const Eigen::MatrixXd* output_generator,
                             const mcscf::SpinFactorSource* right,
                             double energy = 0.0) -> SourceImageAdjoint {
        if (right)
            return project_source_adjoint(source,
                                          *right,
                                          alpha,
                                          beta,
                                          action,
                                          scalar ? nullptr : output_generator,
                                          !scalar,
                                          energy);
        mcscf::FactorActionWorkspace workspace(source,
                                               action.channel_kernel().rows(),
                                               mcscf::kFactorWorkspaceBytes,
                                               false,
                                               output_generator ? 2 : 0);
        return {
            project_factor_source(source, workspace, alpha, beta, action, false, output_generator),
            Eigen::MatrixXd::Zero(action.n_orbitals(), action.n_orbitals())};
    };
    for (int i = 0; i < states; ++i) {
        const Eigen::VectorXd values = expand_support(state.structure_coefficients.col(i));
        const auto y = sources_->orthogonal_source(spin_products_, values, alpha, beta);
        const bool collect = direction && state.weights[i] != 0;
        auto integrals = project(y, delta_sigma, nullptr, collect ? &y : nullptr);
        Eigen::MatrixXd gradient = std::move(integrals.generator);
        if (!scalar)
            result.hamiltonian.col(i) = integrals.images.hamiltonian.col(0);
        add_work(integrals.images.tiles);
        if (capture)
            direction->connection->integral_forcing.col(i) = integrals.images.hamiltonian.col(0);
        if (!moving) {
            if (direction)
                direction->generator_direction.noalias() += 2 * state.weights[i] * gradient;
            continue;
        }
        const auto dy = mcscf::generator_factors(sigma, y, generator);
        const double energy = collect ? state.energies[i] : 0.0;
        const auto right = project(dy, sigma, nullptr, collect ? &y : nullptr, energy);
        const auto left = project(y, sigma, &adjoint_generator, collect ? &dy : nullptr, energy);
        gradient += right.generator + left.generator;
        if (!scalar) {
            result.hamiltonian.col(i) +=
                right.images.hamiltonian.col(0) + left.images.hamiltonian.col(0);
            result.overlap.col(i) = right.images.overlap.col(0) + left.images.overlap.col(0);
        }
        add_work(right.images.tiles);
        add_work(left.images.tiles);
        if (collect) {
            direction->generator_direction.noalias() += 2 * state.weights[i] * gradient;
        }
        if (capture) {
            auto& diagnostic = *direction->connection;
            diagnostic.representation_forcing.col(i) = right.images.hamiltonian.col(0) +
                                                       left.images.hamiltonian.col(0) -
                                                       state.energies[i] * result.overlap.col(i);
            diagnostic.overlap.col(i) = right.images.overlap.col(0);
            const auto physical =
                sources_->orthogonal_source(spin_products_, values, alpha, beta, true);
            const auto physical_dy = mcscf::generator_factors(sigma, physical, generator);
            const double electrons = alpha.n_electrons() + beta.n_electrons();
            const auto density = mcscf::factor_generator_adjoint(sigma, physical, y);
            const double norm = density.trace() / electrons;
            const double parallel = (density.array() * generator.array()).sum();
            const double square =
                mcscf::factor_generator_adjoint(sigma, physical_dy, dy).trace() / electrons;
            diagnostic.squared_norm[i] = square - (norm > 0 ? parallel * parallel / norm : 0);
        }
    }
    return result;
}

StructureActiveIntegralAdjoint StructureAction::source_adjoint_direction(
    const StructureAdjointState& state,
    const Eigen::Ref<const Eigen::MatrixXd>& response,
    const Eigen::MatrixXd* multipliers,
    const OrthogonalActiveIntegrals& integrals,
    const OrthogonalActiveIntegrals& delta,
    const ExteriorOrbitalTransform& alpha,
    const ExteriorOrbitalTransform& beta,
    const mcscf::DirectCiSigmaAction& sigma,
    bool include_integrals,
    const StructureIntegralDirection* cached_direction) const {
    const int n = sigma.n_orbitals(), d = sigma.channel_kernel().rows();
    const Eigen::MatrixXd generator = integrals.orbital_transform.transpose()
                                          .triangularView<Eigen::Lower>()
                                          .solve(delta.orbital_transform.transpose())
                                          .transpose();
    const bool moving = include_integrals && !generator.isZero(0.0);
    const bool reuse = include_integrals && cached_direction &&
                       cached_direction->generator_direction.rows() == n &&
                       cached_direction->generator_direction.cols() == n;
    std::unique_ptr<mcscf::DirectCiSigmaAction> delta_sigma;
    if (include_integrals && !reuse)
        delta_sigma = std::make_unique<mcscf::DirectCiSigmaAction>(sigma, delta);
    Eigen::MatrixXd one = Eigen::MatrixXd::Zero(n, n), pair = Eigen::MatrixXd::Zero(d, d),
                    gradient = one;
    if (reuse)
        gradient = cached_direction->generator_direction;
    for (std::int64_t i = 0; i < core::checked_index<std::int64_t>(state.weights.size()); ++i) {
        const double weight = state.weights[i];
        if (weight == 0.0)
            continue;
        const Eigen::VectorXd values = expand_support(state.structure_coefficients.col(i));
        const auto y = sources_->orthogonal_source(spin_products_, values, alpha, beta);
        const auto physical =
            sources_->orthogonal_source(spin_products_, values, alpha, beta, true);
        Eigen::VectorXd response_values;
        if (response.size() != 0)
            response_values = expand_support(response.col(i));
        const bool responding = response_values.size() && !response_values.isZero(0.0);
        const auto* q_values = responding ? &response_values : nullptr;
        if (moving || responding) {
            const auto dy = sources_->tangent_source(spin_products_,
                                                     values,
                                                     q_values,
                                                     alpha,
                                                     beta,
                                                     sigma,
                                                     moving ? &generator : nullptr);
            const auto physical_dy = sources_->tangent_source(spin_products_,
                                                              values,
                                                              q_values,
                                                              alpha,
                                                              beta,
                                                              sigma,
                                                              moving ? &generator : nullptr,
                                                              true);
            const auto adjoint = sources_->integral_tangent(sigma, y, physical, dy, physical_dy);
            one.noalias() += 2 * weight * adjoint.one_electron;
            pair.noalias() += 2 * weight * adjoint.pair_kernel;
            if (!reuse) {
                gradient.noalias() += 2 * weight *
                                      mcscf::factor_residual_generator_adjoint(
                                          sigma, physical_dy, y, state.energies[i]);
                gradient.noalias() += 2 * weight *
                                      mcscf::factor_residual_generator_adjoint(
                                          sigma, physical, dy, state.energies[i]);
            }
        }
        if (reuse && responding) {
            const auto q =
                sources_->orthogonal_source(spin_products_, response_values, alpha, beta);
            const auto physical_q =
                sources_->orthogonal_source(spin_products_, response_values, alpha, beta, true);
            gradient.noalias() +=
                2 * weight *
                mcscf::factor_residual_generator_adjoint(sigma, physical_q, y, state.energies[i]);
            gradient.noalias() +=
                2 * weight *
                mcscf::factor_residual_generator_adjoint(sigma, physical, q, state.energies[i]);
        }
        if (include_integrals && !reuse)
            gradient.noalias() +=
                2 * weight * mcscf::factor_residual_generator_adjoint(*delta_sigma, physical, y, 0);
    }
    if (multipliers)
        add_response_gauge(state, *multipliers, gradient);
    auto result =
        backpropagate_orthogonal_active_integral_adjoint_direction(integrals,
                                                                   delta,
                                                                   state.one_electron_gradient,
                                                                   state.pair_kernel_gradient,
                                                                   state.generator_gradient,
                                                                   one,
                                                                   pair,
                                                                   gradient);
    return {
        std::move(result.overlap), std::move(result.one_electron), std::move(result.pair_kernel)};
}

} // namespace xmvb::vb
