#include "vbscf/structures/assembly/action.hpp"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <vector>

#include "core/index.hpp"
#include "core/workspace.hpp"
#include "linalg/qr.hpp"
#include "mcscf/ci/product_index.hpp"
#include "parallel/openmp.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"

namespace xmvb::vb {

Eigen::MatrixXd StructureAction::selected_overlap_factor(const ExteriorOrbitalTransform& alpha,
                                                         const ExteriorOrbitalTransform& beta,
                                                         std::size_t workspace_bytes) const {
    const auto n = static_cast<std::size_t>(n_structures_);
    const auto na = static_cast<std::size_t>(n_unique_alpha_);
    const auto nb = static_cast<std::size_t>(n_unique_beta_);
    const auto square = core::checked_multiply(n, n);
    const auto total_capacity = workspace_bytes / sizeof(double);
    const auto fixed = core::checked_add(core::checked_multiply(3, square), n);
    const auto minimum = core::checked_add(fixed,
                                           std::max({core::checked_multiply(4, n),
                                                     core::checked_multiply(4, nb),
                                                     core::checked_multiply(2, na)}));
    const int requested = coarse_openmp_thread_count(effective_openmp_thread_count(), square);
    const int parts = bounded_openmp_thread_count(
        requested, std::min(nb, std::max<std::size_t>(1, total_capacity / minimum)));
    const auto capacity = total_capacity / parts;
    if (!n || capacity <= fixed)
        throw std::length_error("selected QR workspace cannot hold its triangular factor");
    // R and the QR top block consume 2 n^2. Reserve another n^2 for
    // reflector/work vectors. Split the rest between physical rows, raw
    // exterior input, and beta rows. Each worker reduces a different physical
    // beta stripe; its work arrays are disjoint, not copies of shared images.
    const auto work = capacity - fixed;
    const auto a_extent = std::min(na, work / (4 * n));
    const auto b_extent =
        std::min({nb, a_extent ? work / (4 * n * a_extent) : 0, work / (4 * nb), work / (2 * na)});
    if (!a_extent || !b_extent)
        throw std::length_error("selected QR workspace cannot hold one physical row panel");
    const auto rhs_extent = std::min(n, work / (2 * na * b_extent));
    const auto reduce = [&](std::size_t begin, std::size_t end) {
        Eigen::MatrixXd upper = Eigen::MatrixXd::Zero(n, n);
        Eigen::MatrixXd stack, raw;
        workspace::Temporary stack_bytes, raw_bytes;
        linalg::QrWorkspace qr(square);
        for (std::size_t b = begin; b < end; b += b_extent) {
            const auto bc = std::min(b_extent, end - b);
            const auto rows = beta.row_block(b, bc);
            for (std::size_t a = 0; a < na; a += a_extent) {
                const auto ac = std::min(a_extent, na - a);
                stack.resize(n + ac * bc, n);
                stack_bytes.resize(sizeof(double) * stack.size());
                stack.topRows(n) = upper;
                for (std::size_t first = 0; first < n; first += rhs_extent) {
                    const auto width = std::min(rhs_extent, n - first);
                    raw.setZero(na, bc * width);
                    raw_bytes.resize(sizeof(double) * raw.size());
                    const int workers = effective_openmp_thread_count();
#pragma omp parallel for schedule(static) if (na > 32) num_threads(workers)
                    for (std::size_t ai = 0; ai < na; ++ai) {
                        const auto start = mcscf::product_index(ai, 0, nb);
                        auto group =
                            std::lower_bound(spin_products_.begin(), spin_products_.end(), start);
                        for (; group != spin_products_.end() && *group < start + nb; ++group) {
                            const auto g = group - spin_products_.begin();
                            const auto bi = *group - start;
                            for (auto t = spin_term_offsets_[g]; t < spin_term_offsets_[g + 1];
                                 ++t) {
                                const auto& term = spin_terms_[t];
                                if (term.structure >= first && term.structure - first < width)
                                    raw.row(ai)
                                        .segment((term.structure - first) * bc, bc)
                                        .noalias() += term.coefficient * rows->col(bi).transpose();
                            }
                        }
                    }
                    alpha.apply_left(&raw);
                    for (std::size_t s = 0; s < width; ++s)
                        for (std::size_t j = 0; j < bc; ++j)
                            stack.col(first + s).segment(n + j * ac, ac) =
                                raw.col(s * bc + j).segment(a, ac);
                }
                // In-place TSQR: discard physical rows after reducing [R; F_tile].
                qr.compute(stack);
                upper = stack.topRows(n).triangularView<Eigen::Upper>();
            }
        }
        return upper;
    };
    std::vector<Eigen::MatrixXd> factors(parts);
    std::vector<std::exception_ptr> errors(parts);
    workspace::Temporary factor_bytes(sizeof(double) * square * parts);
#pragma omp parallel for schedule(static) if (parts > 1) num_threads(parts)
    for (int part = 0; part < parts; ++part) {
        try {
            const auto boundary = [&](std::size_t i) {
                return (nb / parts) * i + std::min<std::size_t>(nb % parts, i);
            };
            factors[part] = reduce(boundary(part), boundary(part + 1));
        } catch (...) {
            errors[part] = std::current_exception();
        }
    }
    for (const auto& error : errors)
        if (error)
            std::rethrow_exception(error);
    // Deterministic TSQR tree. Only independent triangular summaries survive
    // a stripe; no physical carrier is retained for this merge.
    if (parts == 1)
        return std::move(factors.front());
    linalg::QrWorkspace qr(square);
    Eigen::MatrixXd stack(2 * n, n);
    workspace::Temporary stack_bytes(sizeof(double) * stack.size());
    for (int stride = 1; stride < parts; stride *= 2)
        for (int part = 0; part + stride < parts; part += 2 * stride) {
            stack.topRows(n) = factors[part];
            stack.bottomRows(n) = factors[part + stride];
            qr.compute(stack);
            factors[part] = stack.topRows(n).triangularView<Eigen::Upper>();
            factors[part + stride].resize(0, 0);
        }
    return std::move(factors.front());
}

} // namespace xmvb::vb
