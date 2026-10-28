#include <cstdint>
#include "vbscf/structures/assembly/operator_first_internal.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "core/workspace.hpp"
#include "mcscf/ci/raw_action.hpp"
#include "mcscf/ci/sigma.hpp"
#include "parallel/openmp.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"

namespace xmvb::vb {

OperatorFirstAction::OperatorFirstAction(StructureAction::OrthogonalView view,
                                         Eigen::Index n_structures,
                                         const mcscf::ActiveIntegrals& integrals,
                                         const Eigen::Ref<const Eigen::MatrixXd>& orbital_map)
    : data_(std::make_unique<Data>(view, n_structures, integrals, orbital_map)) {}

OperatorFirstAction::~OperatorFirstAction() = default;

OperatorFirstImages OperatorFirstAction::apply(const Eigen::Ref<const Eigen::MatrixXd>& vectors,
                                               std::size_t workspace_bytes,
                                               bool collect_overlap,
                                               bool overlap_only) const {
    using Clock = std::chrono::steady_clock;
    const auto& data = *data_;
    const auto& view = data.view;
    if (vectors.rows() != data.n_structures || vectors.cols() <= 0)
        throw std::invalid_argument("operator-first structure block dimensions differ");
    const std::int64_t na = view.alpha.dimension(), nb = view.beta.dimension(),
                       rhs = vectors.cols();
    const auto targets = overlap_only ? data.source.source_columns() : data.source.beta_columns();
    const std::size_t families = overlap_only ? 1 : (collect_overlap ? 2 : 1);
    const std::size_t rhs_bytes =
        sizeof(double) * (view.products.size() * (families + 1) + na * families);
    const std::size_t beta_bytes = sizeof(double) * static_cast<std::size_t>(nb);
    const int batch = workspace_bytes > beta_bytes
                          ? std::min<std::size_t>(rhs, (workspace_bytes - beta_bytes) / rhs_bytes)
                          : 0;
    if (batch > 0 && batch < rhs) {
        OperatorFirstImages result;
        result.images.hamiltonian.resize(overlap_only ? 0 : data.n_structures, rhs);
        result.images.overlap.resize(collect_overlap ? data.n_structures : 0, rhs);
        for (std::int64_t first = 0; first < rhs; first += batch) {
            const std::int64_t count = std::min<std::int64_t>(batch, rhs - first);
            auto part = apply(
                vectors.middleCols(first, count), workspace_bytes, collect_overlap, overlap_only);
            if (!overlap_only)
                result.images.hamiltonian.middleCols(first, count) = part.images.hamiltonian;
            if (collect_overlap)
                result.images.overlap.middleCols(first, count) = part.images.overlap;
            result.work.source_products = part.work.source_products;
            result.work.beta_columns = part.work.beta_columns;
            result.work.plan_bytes = part.work.plan_bytes;
            result.work.panel_bytes = std::max(result.work.panel_bytes, part.work.panel_bytes);
            result.work.covalent_reduction = part.work.covalent_reduction;
            result.work.raw_seconds += part.work.raw_seconds;
            result.work.metric_seconds += part.work.metric_seconds;
            result.work.element_evaluations += part.work.element_evaluations;
            auto& t = result.images.tiles;
            const auto& p = part.images.tiles;
            t.expansions += p.expansions;
            t.structure_projections += p.structure_projections;
            t.hamiltonian_projections += p.hamiltonian_projections;
            t.overlap_projections += p.overlap_projections;
            t.source_tiles += p.source_tiles;
            t.source_columns += p.source_columns;
            t.input_panels += p.input_panels;
            t.source_workspace_bound = std::max(t.source_workspace_bound, p.source_workspace_bound);
        }
        return result;
    }
    const std::size_t unit = sizeof(double) * (static_cast<std::size_t>(na) * rhs * families + nb);
    const auto intrinsic = sizeof(double) * view.products.size() * rhs * (families + 1);
    if (intrinsic >= workspace_bytes || workspace_bytes - intrinsic < unit)
        throw std::length_error("operator-first budget cannot hold one source panel");
    // One cache-line of independent vectors per worker, bounded by shared bytes.
    const std::size_t parallel = std::max<std::size_t>(
        1, 8 * cooperative_openmp_thread_count() / static_cast<std::size_t>(rhs));
    const int width = std::min<std::size_t>(
        targets.size(),
        std::min<std::size_t>(std::max<std::int64_t>(1, nb - 1),
                              std::min(parallel, (workspace_bytes - intrinsic) / unit)));
    if (!width)
        throw std::length_error("operator-first source has no target columns");

    OperatorFirstImages result;
    result.work.source_products = view.products.size();
    result.work.beta_columns = targets.size();
    result.work.plan_bytes = data.source.bytes();
    result.work.panel_bytes = width * unit;
    result.work.covalent_reduction = data.source.complete_covalent_support();
    result.images.tiles.expansions = rhs;
    result.images.tiles.structure_projections =
        rhs * ((overlap_only ? 0 : 1) + (collect_overlap ? 1 : 0));
    result.images.tiles.hamiltonian_projections = overlap_only ? 0 : rhs;
    result.images.tiles.overlap_projections = collect_overlap ? rhs : 0;
    result.images.tiles.source_tiles = (targets.size() + width - 1) / width;
    result.images.tiles.source_columns = targets.size() * rhs;
    result.images.tiles.input_panels = result.images.tiles.source_tiles;
    result.images.tiles.source_workspace_bound = intrinsic + width * unit;

    result.images.hamiltonian = Eigen::MatrixXd::Zero(overlap_only ? 0 : data.n_structures, rhs);
    result.images.overlap = Eigen::MatrixXd::Zero(collect_overlap ? data.n_structures : 0, rhs);
    const workspace::Temporary charge(intrinsic + width * unit);
    Eigen::MatrixXd values = Eigen::MatrixXd::Zero(view.products.size(), rhs);
#pragma omp parallel for schedule(static)
    for (std::size_t p = 0; p < view.products.size(); ++p)
        for (std::size_t t = view.offsets[p]; t < view.offsets[p + 1]; ++t) {
            const auto& term = view.terms[t];
            values.row(p).noalias() += term.coefficient * vectors.row(term.structure);
        }
    Eigen::MatrixXd h_products =
        Eigen::MatrixXd::Zero(overlap_only ? 0 : view.products.size(), rhs);
    Eigen::MatrixXd m_products =
        Eigen::MatrixXd::Zero(collect_overlap ? view.products.size() : 0, rhs);
    Eigen::MatrixXd left(na, width * rhs * families), right(nb, width);
    std::vector<Eigen::Index> overlap_betas, overlap_positions;
    for (std::size_t first = 0; first < targets.size(); first += width) {
        const auto prepare = Clock::now();
        const std::int64_t count = std::min<std::size_t>(width, targets.size() - first),
                           columns = overlap_only ? 0 : count * rhs;
        const auto betas = targets.subspan(first, count);
        overlap_betas.clear();
        overlap_positions.clear();
        if (collect_overlap) {
            for (int j = 0; j < count; ++j)
                if (data.source.has_source_column(betas[j])) {
                    overlap_betas.push_back(betas[j]);
                    overlap_positions.push_back(j);
                }
        }
        const auto m_columns = static_cast<Eigen::Index>(overlap_betas.size()) * rhs;
        auto active_left = left.leftCols(columns + m_columns);
        auto active_right = right.leftCols(count);
        if (!overlap_only)
            result.work.element_evaluations +=
                data.source.apply_columns(data.hamiltonian, values, betas, left.leftCols(columns));
        if (m_columns)
            data.source.source_columns(values, overlap_betas, left.middleCols(columns, m_columns));
        active_right.setZero();
        for (int j = 0; j < count; ++j)
            right(betas[j], j) = 1;
        const auto generated = Clock::now();
        result.work.raw_seconds += std::chrono::duration<double>(generated - prepare).count();
        view.alpha.apply_left_block(active_left);
        view.alpha.apply_adjoint_left_block(active_left);
        view.beta.apply_left_block(active_right);
        view.beta.apply_adjoint_left_block(active_right);
#pragma omp parallel for schedule(static)
        for (std::size_t p = 0; p < view.products.size(); ++p) {
            const std::int64_t a = view.products[p] / nb, b = view.products[p] % nb;
            for (int v = 0; v < rhs; ++v) {
                double h = 0, m = 0;
                for (int j = 0; !overlap_only && j < count; ++j) {
                    h += left(a, j * rhs + v) * right(b, j);
                }
                for (std::size_t j = 0; j < overlap_betas.size(); ++j)
                    m += left(a, columns + j * rhs + v) * right(b, overlap_positions[j]);
                if (!overlap_only)
                    h_products(p, v) += h;
                if (collect_overlap)
                    m_products(p, v) += m;
            }
        }
        result.work.metric_seconds +=
            std::chrono::duration<double>(Clock::now() - generated).count();
    }
    // Incidence is traversed once after all source columns have contributed.
#pragma omp parallel for schedule(static)
    for (Eigen::Index structure = 0; structure < data.n_structures; ++structure)
        for (std::size_t t = view.structure_offsets[structure];
             t < view.structure_offsets[structure + 1];
             ++t) {
            const auto& term = view.terms[view.structure_terms[t]];
            if (!overlap_only)
                result.images.hamiltonian.row(structure).noalias() +=
                    term.coefficient * h_products.row(term.group);
            if (collect_overlap)
                result.images.overlap.row(structure).noalias() +=
                    term.coefficient * m_products.row(term.group);
        }
    return result;
}
} // namespace xmvb::vb
