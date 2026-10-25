#include "vbscf/structures/assembly/action.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "mcscf/ci/factors.hpp"
#include "vbscf/structures/assembly/sources.hpp"

namespace xmvb::vb {

void StructureAction::add_response_gauge(const StructureAdjointState& state,
                                         const Eigen::Ref<const Eigen::MatrixXd>& multipliers,
                                         Eigen::Ref<Eigen::MatrixXd> generator) const {
    // Only the weight-symmetrized multiplier couples to a symmetric overlap direction.
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(state.weights.size()); ++i) {
        if (state.weights[i] == 0)
            continue;
        for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(state.weights.size()); ++j) {
            const double factor = multipliers(j, i) * 0.5 +
                                  multipliers(i, j) * (0.5 * state.weights[j] / state.weights[i]);
            if (factor == 0)
                continue;
            const auto high = std::max(i, j), low = std::min(i, j);
            const auto& density = state.overlap_generators[high * (high + 1) / 2 + low];
            if (i >= j)
                generator.noalias() += 2 * state.weights[i] * factor * density;
            else
                generator.noalias() += 2 * state.weights[i] * factor * density.transpose();
        }
    }
}

bool StructureAction::prefer_factor_response(const mcscf::DirectCiSigmaAction& sigma) const {
    const long double r = sources_->nodes.size();
    const long double l = r * (spin_exchange_parity_ ? 2 : 1);
    const long double u = static_cast<long double>(n_unique_alpha_) + n_unique_beta_;
    const long double d = sigma.channel_kernel().rows();
    const long double products = spin_exchange_parity_
                                     ? static_cast<long double>(n_unique_alpha_) *
                                           (n_unique_alpha_ + spin_exchange_parity_) / 2
                                     : static_cast<long double>(n_unique_alpha_) * n_unique_beta_;
    // Integral Gram: 2 U r d^2. Shared H q/H Y images: 4 U r d^2.
    // Source-pair moments: 4 U l r (d+1) for the integral adjoint and
    // 8 U l r (d+2) for the two image-generator adjoints.
    // Compare to one fused product sigma/Gram (4 products d^2).
    return r > 0 && 6 * u * r * d * d + u * l * r * (12 * d + 20) < 4 * products * d * d;
}

StructureResponseSweep
StructureAction::source_response(const StructureAdjointState& state,
                                 const Eigen::Ref<const Eigen::MatrixXd>& coefficients,
                                 const ExteriorOrbitalTransform& alpha,
                                 const ExteriorOrbitalTransform& beta,
                                 const mcscf::DirectCiSigmaAction& sigma,
                                 bool collect_images) const {
    const int n = sigma.n_orbitals(), d = sigma.channel_kernel().rows();
    const int n_states = state.weights.size();
    if (n_states <= 0 || state.structure_coefficients.rows() != n_structures_ ||
        state.structure_coefficients.cols() != n_states ||
        state.energies.size() != state.weights.size())
        throw std::invalid_argument("invalid selected-subspace adjoint state");
    StructureResponseSweep result{
        {Eigen::MatrixXd::Zero(collect_images ? n_structures_ : 0, n_states),
         Eigen::MatrixXd::Zero(collect_images ? n_structures_ : 0, n_states)},
        Eigen::MatrixXd::Zero(n, n),
        Eigen::MatrixXd::Zero(d, d),
        Eigen::MatrixXd::Zero(n, n)};
    for (int i = 0; i < n_states; ++i) {
        const double weight = state.weights[i], energy = state.energies[i];
        if (!std::isfinite(weight) || weight < 0 || !std::isfinite(energy))
            throw std::invalid_argument("invalid selected-subspace adjoint state");
        const Eigen::VectorXd response = expand_support(coefficients.col(i));
        if (response.isZero(0.0))
            continue;
        const Eigen::VectorXd accepted = expand_support(state.structure_coefficients.col(i));
        const auto factors = [&](const Eigen::VectorXd& values, bool physical) {
            return sources_->orthogonal_source(spin_products_, values, alpha, beta, physical);
        };
        const auto y = factors(accepted, false);
        const auto q = factors(response, false), physical_q = factors(response, true);
        if (weight == 0.0) {
            if (!collect_images)
                continue;
            const auto image = apply_source_columns(
                1,
                [&](int) { return response; },
                alpha,
                beta,
                sigma,
                false,
                mcscf::kFactorWorkspaceBytes);
            result.images.hamiltonian.col(i) = image.hamiltonian.col(0);
            result.images.overlap.col(i) = image.overlap.col(0);
            continue;
        }
        // H q is generated once: its raw factor images also supply D(H q,Y).
        // The second image needs only its VJP, never another structure projection.
        auto response_image =
            project_source_adjoint(q, y, alpha, beta, sigma, nullptr, collect_images, energy);
        auto accepted_image =
            project_source_adjoint(y, q, alpha, beta, sigma, nullptr, false, energy);
        const auto& images = response_image.images;
        if (collect_images) {
            result.images.hamiltonian.col(i) = images.hamiltonian.col(0);
            result.images.overlap.col(i) = images.overlap.col(0);
        }
        result.images.tiles.expansions +=
            images.tiles.expansions + accepted_image.images.tiles.expansions;
        result.images.tiles.structure_projections += images.tiles.structure_projections;
        result.images.tiles.source_tiles +=
            images.tiles.source_tiles + accepted_image.images.tiles.source_tiles;
        result.images.tiles.source_columns +=
            images.tiles.source_columns + accepted_image.images.tiles.source_columns;
        result.images.tiles.channel_blocks +=
            images.tiles.channel_blocks + accepted_image.images.tiles.channel_blocks;
        result.images.tiles.source_workspace_bound =
            std::max({result.images.tiles.source_workspace_bound,
                      images.tiles.source_workspace_bound,
                      accepted_image.images.tiles.source_workspace_bound});
        const auto integral = mcscf::factor_integral_adjoint(sigma, physical_q, y);
        result.one_electron.noalias() += 2 * weight * integral.one_electron;
        result.pair_kernel.noalias() += 2 * weight * integral.pair_kernel;
        result.generator.noalias() +=
            2 * weight * (response_image.generator + accepted_image.generator);
    }
    return result;
}

} // namespace xmvb::vb
