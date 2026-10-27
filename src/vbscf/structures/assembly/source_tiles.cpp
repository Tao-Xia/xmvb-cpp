#include "core/index.hpp"
#include <cstdint>
#include "vbscf/structures/assembly/action.hpp"
#include "mcscf/ci/workspace.hpp"

#include <algorithm>
#include <chrono>

#include "parallel/openmp.hpp"
#include "core/timing.hpp"
#include "mcscf/ci/factors.hpp"
#include "mcscf/ci/product_support.hpp"
#include "mcscf/ci/sigma.hpp"
#include "vbscf/structures/assembly/sources.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"

namespace xmvb::vb {
Eigen::VectorXd
StructureAction::expand_support(const Eigen::Ref<const Eigen::VectorXd>& vectors) const {
    Eigen::VectorXd coefficients = Eigen::VectorXd::Zero(spin_products_.size());
#pragma omp parallel for schedule(static) if (spin_products_.size() > 32)
    for (std::size_t group = 0; group < spin_products_.size(); ++group)
        for (std::size_t t = spin_term_offsets_[group]; t < spin_term_offsets_[group + 1]; ++t)
            coefficients[group] += spin_terms_[t].coefficient * vectors[spin_terms_[t].structure];
    return coefficients;
}

bool StructureAction::prefer_source_contraction(const ExteriorOrbitalTransform& alpha,
                                                const ExteriorOrbitalTransform& beta,
                                                const mcscf::DirectCiSigmaAction& sigma) const {
    const long double r = sources_->nodes.size();
    const long double na = n_unique_alpha_, nb = n_unique_beta_;
    const long double pairs = sigma.channel_kernel().rows();
    const int n = alpha.n_orbitals(), ea = alpha.n_electrons(), eb = beta.n_electrons();
    // Arithmetic counts for two channel GEMMs, single-spin links, exterior
    // shears and sampling. This compares two exact contraction orders, not
    // two physical spaces; neither a memory budget nor a numerical threshold enters.
    const long double factor_work =
        r * (2 * (na + nb) * pairs * pairs + 4 * (na * ea * (n - ea + 1) + nb * eb * (n - eb + 1)) +
             (pairs + 3) * (2 * (alpha.shear_pair_count() + beta.shear_pair_count()) + na + nb) +
             (2 * pairs + 6) * spin_products_.size());
    const long double full_rows =
        spin_exchange_parity_ == 0 ? na * nb : na * (na + spin_exchange_parity_) / 2;
    const long double channel_work = 2 * pairs * pairs;
    if (factor_work >= full_rows * channel_work)
        return false;
    if (!product_support_)
        return true;

    // Count only generated support until its GEMM work alone exceeds the
    // factor estimate. Stop at that proof; no full-product membership scan.
    auto cursor = product_support_->cursor(1);
    Eigen::Index a, b;
    bool supported;
    long double rows = 0;
    while (cursor.next(a, b, supported)) {
        if (spin_exchange_parity_ && (a > b || (a == b && spin_exchange_parity_ < 0)))
            continue;
        if (++rows * channel_work > factor_work)
            return true;
    }
    return false;
}

StructureActionResult
StructureAction::apply_source_tiles(const Eigen::Ref<const Eigen::MatrixXd>& vectors,
                                    const ExteriorOrbitalTransform& alpha,
                                    const ExteriorOrbitalTransform& beta,
                                    const mcscf::DirectCiSigmaAction& sigma,
                                    bool overlap_only,
                                    std::size_t workspace_bytes) const {
    return apply_source_columns(
        vectors.cols(),
        [&](std::int64_t column) { return expand_support(vectors.col(column)); },
        alpha,
        beta,
        sigma,
        overlap_only,
        workspace_bytes);
}

StructureActionResult
StructureAction::apply_source_columns(int width,
                                      const std::function<Eigen::VectorXd(int)>& support_column,
                                      const ExteriorOrbitalTransform& alpha,
                                      const ExteriorOrbitalTransform& beta,
                                      const mcscf::DirectCiSigmaAction& sigma,
                                      bool overlap_only,
                                      std::size_t workspace_bytes) const {
    const mcscf::CiWorkspace operation(workspace_bytes, false);
    workspace_bytes = operation.bytes();
    const mcscf::SpinFactorSource shape{n_unique_alpha_,
                                        n_unique_beta_,
                                        core::checked_index<std::int64_t>(sources_->nodes.size()),
                                        {}};
    mcscf::FactorActionWorkspace workspace(
        shape, sigma.channel_kernel().rows(), workspace_bytes, overlap_only);
    StructureActionResult result{Eigen::MatrixXd::Zero(n_structures_, width),
                                 Eigen::MatrixXd::Zero(n_structures_, width)};
    for (std::int64_t column = 0; column < width; ++column) {
        const Eigen::VectorXd values = support_column(column);
        workspace::Temporary values_temporary(sizeof(double) *
                                              static_cast<std::size_t>(values.size()));
        const auto source = sources_->orthogonal_source(spin_products_, values, alpha, beta);
        const auto image =
            project_factor_source(source, workspace, alpha, beta, sigma, overlap_only);
        result.hamiltonian.col(column) = image.hamiltonian.col(0);
        result.overlap.col(column) = image.overlap.col(0);
        result.tiles.expansions += image.tiles.expansions;
        result.tiles.structure_projections += image.tiles.structure_projections;
        result.tiles.source_tiles += image.tiles.source_tiles;
        result.tiles.source_columns += image.tiles.source_columns;
        result.tiles.channel_blocks += image.tiles.channel_blocks;
        result.tiles.source_workspace_bound =
            std::max(result.tiles.source_workspace_bound, image.tiles.source_workspace_bound);
    }
    return result;
}

StructureActionResult
StructureAction::project_factor_source(const mcscf::SpinFactorSource& source,
                                       mcscf::FactorActionWorkspace& workspace,
                                       const ExteriorOrbitalTransform& alpha,
                                       const ExteriorOrbitalTransform& beta,
                                       const mcscf::DirectCiSigmaAction& sigma,
                                       bool overlap_only,
                                       const Eigen::MatrixXd* output_generator,
                                       const FactorImageConsumer& image_consumer,
                                       bool project_output,
                                       double residual_energy) const {
    profiling::ScopedPhase phase(profiling::Phase::CiSigma);
    using Clock = std::chrono::steady_clock;
    const bool timing = profiling::phase_totals().enabled;
    const auto tick = [&] { return timing ? Clock::now() : Clock::time_point{}; };
    const auto seconds = [](auto first, auto last) {
        return std::chrono::duration<double>(last - first).count();
    };
    double scatter_time = 0, mix_time = 0, gather_time = 0, pullback_time = 0, sample_time = 0;
    const auto& sources = *sources_;
    const auto& alpha_projection = sources.alpha_projection;
    const auto& beta_projection = sources.beta_projection;
    int source_tile = 0, channel_tile = 0;
    StructureActionResult result{Eigen::MatrixXd::Zero(project_output ? n_structures_ : 0, 1),
                                 Eigen::MatrixXd::Zero(project_output ? n_structures_ : 0, 1)};
    const int extra = project_output && output_generator
                          ? 2 * workspace.source_tile() * std::max(1, workspace.channel_tile())
                          : 0;
    Eigen::MatrixXd ga(n_unique_alpha_, extra), gb(n_unique_beta_, extra);
    workspace::Temporary pullback_temporary(sizeof(double) *
                                            static_cast<std::size_t>(ga.size() + gb.size()));
    const auto pullback =
        [&](bool spin, Eigen::Ref<Eigen::MatrixXd> value, Eigen::Ref<Eigen::MatrixXd> tangent) {
            const auto& transform = spin ? beta : alpha;
            const auto& projection = spin ? beta_projection : alpha_projection;
            if (output_generator) {
                sigma.apply_spin_generator(spin, value, *output_generator, tangent);
                transform.apply_adjoint_left_block(tangent, projection);
            }
            transform.apply_adjoint_left_block(value, projection);
        };
    {
        Eigen::VectorXd h_products =
                            Eigen::VectorXd::Zero(project_output ? spin_products_.size() : 0),
                        m_products = h_products;
        workspace::Temporary products_temporary(
            sizeof(double) * static_cast<std::size_t>(h_products.size() + m_products.size()));
        ++result.tiles.expansions;

        const auto cross = [&](mcscf::FactorCrossBlock block) {
            if (image_consumer)
                image_consumer(block.alpha_channels, block.mixed_beta, 2.0);
            if (!project_output)
                return;
            const auto pulling = tick();
            auto da = ga.leftCols(output_generator ? block.alpha_channels.cols() : 0);
            auto db = gb.leftCols(output_generator ? block.mixed_beta.cols() : 0);
            pullback(false, block.alpha_channels, da);
            pullback(true, block.mixed_beta, db);
            const auto sampled = tick();
            pullback_time += seconds(pulling, sampled);
#pragma omp parallel for schedule(static) if (spin_products_.size() > 32)
            for (std::size_t group = 0; group < spin_products_.size(); ++group) {
                const Eigen::Index i = spin_products_[group] / n_unique_beta_;
                const Eigen::Index j = spin_products_[group] % n_unique_beta_;
                if (output_generator)
                    h_products[group] += 2 * (da.row(i).dot(block.mixed_beta.row(j)) +
                                              block.alpha_channels.row(i).dot(db.row(j)));
                else
                    h_products[group] +=
                        2 * block.alpha_channels.row(i).dot(block.mixed_beta.row(j));
            }
            sample_time += seconds(sampled, tick());
        };
        const auto finish = [&](mcscf::FactorImageBlock block) {
            if (image_consumer && !overlap_only) {
                image_consumer(block.same_alpha, block.beta, 1.0);
                image_consumer(block.alpha, block.same_beta, 1.0);
                if (residual_energy != 0.0)
                    image_consumer(block.alpha, block.beta, -residual_energy);
            }
            if (!project_output)
                return;
            auto da = ga.leftCols(output_generator ? block.width : 0);
            auto db = gb.leftCols(output_generator ? block.width : 0);
            auto dha = ga.middleCols(output_generator ? block.width : 0,
                                     output_generator ? block.width : 0);
            auto dhb = gb.middleCols(output_generator ? block.width : 0,
                                     output_generator ? block.width : 0);
            if (!overlap_only) {
                pullback(false, block.same_alpha, dha);
                pullback(true, block.same_beta, dhb);
            }
            pullback(false, block.alpha, da);
            pullback(true, block.beta, db);
#pragma omp parallel for schedule(static) if (spin_products_.size() > 32)
            for (std::size_t group = 0; group < spin_products_.size(); ++group) {
                const Eigen::Index i = spin_products_[group] / n_unique_beta_;
                const Eigen::Index j = spin_products_[group] % n_unique_beta_;
                if (output_generator) {
                    m_products[group] +=
                        da.row(i).dot(block.beta.row(j)) + block.alpha.row(i).dot(db.row(j));
                    if (!overlap_only)
                        h_products[group] += dha.row(i).dot(block.beta.row(j)) +
                                             block.same_alpha.row(i).dot(db.row(j)) +
                                             da.row(i).dot(block.same_beta.row(j)) +
                                             block.alpha.row(i).dot(dhb.row(j));
                } else {
                    m_products[group] += block.alpha.row(i).dot(block.beta.row(j));
                    if (!overlap_only)
                        h_products[group] += block.same_alpha.row(i).dot(block.beta.row(j)) +
                                             block.alpha.row(i).dot(block.same_beta.row(j));
                }
            }
        };
        const auto work = workspace.apply(sigma, source, cross, finish);
        source_tile = work.source_tile;
        channel_tile = work.channel_tile;
        result.tiles.source_tiles += work.panels;
        result.tiles.source_columns += work.sources;
        result.tiles.channel_blocks += work.channel_blocks;
        result.tiles.source_workspace_bound =
            std::max(result.tiles.source_workspace_bound, work.workspace_bound);
        scatter_time += work.scatter_seconds;
        mix_time += work.mix_seconds;
        gather_time += work.gather_seconds;
        result.tiles.structure_projections += project_output;
#pragma omp parallel for schedule(static) if (n_structures_ > 32)
        for (Eigen::Index structure = 0; structure < (project_output ? n_structures_ : 0);
             ++structure) {
            double h = 0, m = 0;
            for (std::size_t t = structure_term_offsets_[structure];
                 t < structure_term_offsets_[structure + 1];
                 ++t) {
                const auto& term = spin_terms_[structure_term_indices_[t]];
                h += term.coefficient * h_products[term.group];
                m += term.coefficient * m_products[term.group];
            }
            result.hamiltonian(structure, 0) = h;
            result.overlap(structure, 0) = m;
        }
    }
    if (profiling::phase_totals().enabled)
        std::fprintf(
            stderr,
            "XMVB_SOURCES,structures=%d,sources=%zu,products=%zu,source_tile=%d,channel_tile=%d\n",
            n_structures_,
            sources.nodes.size(),
            spin_products_.size(),
            source_tile,
            channel_tile);
    if (timing)
        std::fprintf(
            stderr,
            "XMVB_SOURCE_TIME,scatter=%.9f,mix=%.9f,gather=%.9f,pullback=%.9f,sample=%.9f\n",
            scatter_time,
            mix_time,
            gather_time,
            pullback_time,
            sample_time);
    return result;
}

} // namespace xmvb::vb
