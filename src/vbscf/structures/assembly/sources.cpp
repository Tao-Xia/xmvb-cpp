#include <cstdint>
#include "vbscf/structures/assembly/sources.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include "core/timing.hpp"
#include "mcscf/ci/factors_internal.hpp"
#include "parallel/openmp.hpp"

namespace xmvb::vb {
namespace {

/** @brief Maximum bipartite matching, with heap-backed augmenting paths.
 * Hopcroft--Karp layers bound the work by O(edges * sqrt(vertices)). A greedy
 * matching first certifies common diagonal/dense supports without extra phases.
 */
struct SupportMatching {
    const std::vector<std::vector<Eigen::Index>>& edges;
    std::vector<Eigen::Index> left, right, level, cursor, queue, path;
    Eigen::Index shortest = 0;

    SupportMatching(const std::vector<std::vector<Eigen::Index>>& adjacency, Eigen::Index nb)
        : edges(adjacency), left(edges.size(), -1), right(nb, -1), level(edges.size()),
          cursor(edges.size()) {
        Eigen::Index rows = 0, columns = 0, matched = 0;
        std::vector<unsigned char> seen(nb, 0);
        for (Eigen::Index a = 0; a < static_cast<Eigen::Index>(edges.size()); ++a) {
            rows += !edges[a].empty();
            for (const Eigen::Index b : edges[a]) {
                if (!seen[b]) {
                    seen[b] = 1;
                    ++columns;
                }
                if (left[a] < 0 && right[b] < 0) {
                    left[a] = b;
                    right[b] = a;
                    ++matched;
                }
            }
        }
        if (matched == std::min(rows, columns))
            return;
        while (layer()) {
            std::fill(cursor.begin(), cursor.end(), 0);
            for (Eigen::Index a = 0; a < static_cast<Eigen::Index>(left.size()); ++a)
                if (left[a] < 0 && level[a] == 0)
                    augment(a);
        }
    }

    bool layer() {
        std::fill(level.begin(), level.end(), -1);
        queue.clear();
        for (Eigen::Index a = 0; a < static_cast<Eigen::Index>(left.size()); ++a)
            if (left[a] < 0 && !edges[a].empty()) {
                level[a] = 0;
                queue.push_back(a);
            }
        shortest = std::numeric_limits<Eigen::Index>::max();
        for (std::size_t i = 0; i < queue.size(); ++i) {
            const Eigen::Index a = queue[i];
            if (level[a] >= shortest)
                continue;
            for (const Eigen::Index b : edges[a]) {
                const Eigen::Index next = right[b];
                if (next < 0)
                    shortest = level[a] + 1;
                else if (level[next] < 0) {
                    level[next] = level[a] + 1;
                    queue.push_back(next);
                }
            }
        }
        return shortest != std::numeric_limits<Eigen::Index>::max();
    }

    bool augment(Eigen::Index root) {
        path.clear();
        path.push_back(root);
        while (!path.empty()) {
            const Eigen::Index a = path.back();
            bool extended = false;
            while (cursor[a] < static_cast<Eigen::Index>(edges[a].size())) {
                const Eigen::Index b = edges[a][cursor[a]++], next = right[b];
                if (next < 0 && level[a] + 1 == shortest) {
                    Eigen::Index free_beta = b;
                    for (auto row = path.rbegin(); row != path.rend(); ++row) {
                        const Eigen::Index previous = left[*row];
                        left[*row] = free_beta;
                        right[free_beta] = *row;
                        free_beta = previous;
                    }
                    return true;
                }
                if (next >= 0 && level[next] == level[a] + 1 && level[next] < shortest) {
                    path.push_back(next);
                    extended = true;
                    break;
                }
            }
            if (!extended) {
                level[a] = -1;
                path.pop_back();
            }
        }
        return false;
    }
};

} // namespace

SpinSources::SpinSources(const std::vector<mcscf::ProductIndex>& products,
                         Eigen::Index na,
                         Eigen::Index nb,
                         int parity)
    : alpha_offsets(na + 1, 0), beta_groups(nb), parity_(parity) {
    std::vector<std::vector<Eigen::Index>> edges(na);
    const auto representative = [=](Eigen::Index a, Eigen::Index b) {
        return !parity || a < b || (a == b && parity > 0);
    };
    for (std::size_t group = 0; group < products.size(); ++group) {
        const Eigen::Index a = products[group] / nb, b = products[group] % nb;
        ++alpha_offsets[a + 1];
        beta_groups[b].push_back(group);
        if (representative(a, b))
            edges[a].push_back(b);
    }
    for (Eigen::Index a = 0; a < na; ++a) {
        if (alpha_offsets[a + 1])
            alpha_rows.push_back(a);
        alpha_offsets[a + 1] += alpha_offsets[a];
    }
    for (Eigen::Index b = 0; b < nb; ++b)
        if (!beta_groups[b].empty())
            columns.push_back(b);

    const SupportMatching matching(edges, nb);
    // Alternating reachability gives the exact minimum vertex cover:
    // (unreached alpha vertices) union (reached beta vertices).
    std::vector<unsigned char> reached_a(na, 0), reached_b(nb, 0);
    std::vector<Eigen::Index> queue;
    for (Eigen::Index a = 0; a < na; ++a)
        if (matching.left[a] < 0 && !edges[a].empty()) {
            reached_a[a] = 1;
            queue.push_back(a);
        }
    for (std::size_t i = 0; i < queue.size(); ++i)
        for (const Eigen::Index b : edges[queue[i]]) {
            if (b == matching.left[queue[i]] || reached_b[b])
                continue;
            reached_b[b] = 1;
            const Eigen::Index next = matching.right[b];
            if (next >= 0 && !reached_a[next]) {
                reached_a[next] = 1;
                queue.push_back(next);
            }
        }

    std::vector<Eigen::Index> row_source(na, -1), column_source(nb, -1);
    // Prefer the original beta-column order when it is already minimal. This
    // avoids changing summation order on the established covalent fixtures.
    Eigen::Index matched = 0, touched_columns = 0;
    std::vector<unsigned char> input_column(nb, 0);
    for (Eigen::Index a = 0; a < na; ++a) {
        matched += matching.left[a] >= 0;
        for (const Eigen::Index b : edges[a])
            if (!input_column[b]) {
                input_column[b] = 1;
                ++touched_columns;
            }
    }
    if (matched != touched_columns)
        for (Eigen::Index a = 0; a < na; ++a)
            if (!edges[a].empty() && !reached_a[a]) {
                row_source[a] = nodes.size();
                nodes.push_back({a, false});
            }
    for (Eigen::Index b = 0; b < nb; ++b)
        if ((matched == touched_columns && input_column[b]) ||
            (matched != touched_columns && reached_b[b])) {
            column_source[b] = nodes.size();
            nodes.push_back({b, true});
        }
    offsets.assign(nodes.size() + 1, 0);
    for (const mcscf::ProductIndex product : products) {
        const Eigen::Index a = product / nb, b = product % nb;
        if (representative(a, b))
            ++offsets[(row_source[a] >= 0 ? row_source[a] : column_source[b]) + 1];
    }
    for (std::size_t i = 1; i < offsets.size(); ++i)
        offsets[i] += offsets[i - 1];
    groups.resize(offsets.back());
    auto next = offsets;
    for (std::size_t group = 0; group < products.size(); ++group) {
        const Eigen::Index a = products[group] / nb, b = products[group] % nb;
        if (representative(a, b))
            groups[next[row_source[a] >= 0 ? row_source[a] : column_source[b]]++] = group;
    }
}

void SpinSources::load(const std::vector<mcscf::ProductIndex>& products,
                       const Eigen::Ref<const Eigen::VectorXd>& coefficients,
                       Eigen::Index first,
                       Eigen::Index width,
                       Eigen::Ref<Eigen::MatrixXd> alpha,
                       Eigen::Ref<Eigen::MatrixXd> beta) const {
    alpha.setZero();
    beta.setZero();
    const Eigen::Index nb = beta.rows();
    for (int j = 0; j < width; ++j) {
        const Eigen::Index index = first + j;
        const auto& source = nodes[index];
        if (source.beta)
            beta(source.string, j) = 1.0;
        else
            alpha(source.string, j) = 1.0;
        for (std::size_t entry = offsets[index]; entry < offsets[index + 1]; ++entry) {
            const auto group = groups[entry];
            const Eigen::Index row = products[group] / nb, column = products[group] % nb;
            const double value = coefficients[group] * (parity_ && row != column ? 2 : 1);
            if (source.beta)
                alpha(row, j) = value;
            else
                beta(column, j) = value;
        }
    }
}

mcscf::SpinFactorSource
SpinSources::orthogonal_source(const std::vector<mcscf::ProductIndex>& products,
                               const Eigen::Ref<const Eigen::VectorXd>& coefficients,
                               const ExteriorOrbitalTransform& alpha,
                               const ExteriorOrbitalTransform& beta,
                               bool physical) const {
    const bool project = physical && parity_ != 0;
    const std::int64_t rank = nodes.size();
    mcscf::SpinFactorSource result{alpha.dimension(),
                                   beta.dimension(),
                                   rank * (project ? 2 : 1),
                                   [this, &products, coefficients, &alpha, &beta, project, rank](
                                       Eigen::Index first,
                                       Eigen::Index width,
                                       Eigen::Ref<Eigen::MatrixXd> a,
                                       Eigen::Ref<Eigen::MatrixXd> b) {
                                       Eigen::Index offset = 0;
                                       while (offset < width) {
                                           const Eigen::Index local = (first + offset) % rank;
                                           const Eigen::Index count =
                                               std::min(width - offset, rank - local);
                                           auto aa = a.middleCols(offset, count),
                                                bb = b.middleCols(offset, count);
                                           const bool exchanged = project && first + offset >= rank;
                                           if (exchanged)
                                               load(products, coefficients, local, count, bb, aa);
                                           else
                                               load(products, coefficients, local, count, aa, bb);
                                           if (project)
                                               aa *= exchanged ? 0.5 * parity_ : 0.5;
                                           offset += count;
                                       }
                                       alpha.apply_left_block(a);
                                       beta.apply_left_block(b);
                                   },
                                   0,
                                   2.0L * (alpha.dimension() + beta.dimension()) +
                                       2.0L * (alpha.shear_pair_count() + beta.shear_pair_count()) +
                                       2.0L * groups.size() / std::max<std::int64_t>(1, rank) + 1 +
                                       (project ? alpha.dimension() : 0)};
    // Y V = T_alpha [Z (T_beta^T V)]. Z is traversed through the original
    // support indices, never assembled as a product matrix. A physical parity
    // image is P_eta Y, not a second unprojected representative.
    result.contract = [this, &products, coefficients, &alpha, &beta, project](
                          bool transpose,
                          const Eigen::Ref<const Eigen::MatrixXd>& input,
                          Eigen::Ref<Eigen::MatrixXd> output,
                          double scale) {
        const Eigen::Index nb = beta.dimension();
        const auto& incoming = transpose ? alpha : beta;
        const auto& outgoing = transpose ? beta : alpha;
        Eigen::MatrixXd seed = input;
        incoming.apply_adjoint_left(&seed);
        Eigen::MatrixXd image = Eigen::MatrixXd::Zero(outgoing.dimension(), input.cols());
        const int workers =
            coarse_openmp_thread_count(effective_openmp_thread_count(),
                                       static_cast<std::size_t>(products.size()) * input.cols());
#pragma omp parallel for schedule(static) if (workers > 1) num_threads(workers)
        for (Eigen::Index row = 0; row < image.rows(); ++row) {
            const auto consume = [&](bool flip, double weight) {
                const auto add = [&](int group) {
                    const Eigen::Index a = products[group] / nb, b = products[group] % nb;
                    if (parity_ && !(a < b || (a == b && parity_ > 0)))
                        return;
                    const double value = weight * coefficients[group] * (parity_ && a != b ? 2 : 1);
                    image.row(row).noalias() += value * seed.row(flip ? a : b);
                };
                if (flip)
                    for (const auto group : beta_groups[row])
                        add(group);
                else
                    for (std::size_t group = alpha_offsets[row]; group < alpha_offsets[row + 1];
                         ++group)
                        add(group);
            };
            consume(transpose, project ? 0.5 : 1);
            // Certified exchange parity uses identical spin exterior maps. Apply
            // P_eta to raw support inside this same traversal, not by transforming
            // the representative twice. No numerical symmetry is inferred here.
            if (project)
                consume(!transpose, 0.5 * parity_);
        }
        outgoing.apply_left(&image);
        output.noalias() += scale * image;
    };
    result.contract_rows = static_cast<std::size_t>(alpha.dimension()) + beta.dimension();
    result.contract_work = 4.0L * (alpha.shear_pair_count() + beta.shear_pair_count()) +
                           4.0L * (alpha.dimension() + beta.dimension()) +
                           4.0L * (project ? 2 : 1) * products.size();
    return result;
}

mcscf::SpinFactorSource
SpinSources::tangent_source(const std::vector<mcscf::ProductIndex>& products,
                            const Eigen::Ref<const Eigen::VectorXd>& coefficients,
                            const Eigen::VectorXd* coefficient_direction,
                            const ExteriorOrbitalTransform& alpha,
                            const ExteriorOrbitalTransform& beta,
                            const mcscf::DirectCiSigmaAction& sigma,
                            const Eigen::MatrixXd* generator,
                            bool physical) const {
    if (!generator) {
        if (!coefficient_direction)
            throw std::invalid_argument("a source tangent needs a direction");
        return orthogonal_source(products, *coefficient_direction, alpha, beta, physical);
    }
    const auto base = orthogonal_source(products, coefficients, alpha, beta, physical);
    if (!coefficient_direction)
        return mcscf::generator_factors(sigma, base, *generator);
    const bool project = physical && parity_ != 0;
    const std::int64_t rank = nodes.size();
    return {base.n_alpha,
            base.n_beta,
            2 * base.count,
            [this,
             &products,
             coefficient_direction,
             &alpha,
             &beta,
             &sigma,
             generator,
             base,
             project,
             rank](Eigen::Index first,
                   Eigen::Index width,
                   Eigen::Ref<Eigen::MatrixXd> a,
                   Eigen::Ref<Eigen::MatrixXd> b) {
                Eigen::Index offset = 0;
                while (offset < width) {
                    const bool beta_side = first >= base.count;
                    const Eigen::Index local = beta_side ? first - base.count : first;
                    const Eigen::Index count = std::min(width - offset, base.count - local);
                    auto aa = a.middleCols(offset, count), bb = b.middleCols(offset, count);
                    base.load(local, count, aa, bb);
                    Eigen::MatrixXd image(beta_side ? base.n_beta : base.n_alpha, count);
                    sigma.apply_spin_generator(beta_side, beta_side ? bb : aa, *generator, image);
                    if (beta_side)
                        bb = image;
                    else
                        aa = image;

                    // One coordinate anchor is constant. Only its opposite factor can
                    // carry dC. Reuse the generator scratch for that exact derivative.
                    image.setZero();
                    bool varying = false;
                    for (int j = 0; j < count; ++j) {
                        const int node = (local + j) % rank;
                        const bool exchanged = project && local + j >= rank;
                        const bool anchor_beta = nodes[node].beta != exchanged;
                        if (beta_side == anchor_beta)
                            continue;
                        varying = true;
                        for (std::size_t entry = offsets[node]; entry < offsets[node + 1];
                             ++entry) {
                            const auto group = groups[entry];
                            const Eigen::Index row = products[group] / base.n_beta,
                                               column = products[group] % base.n_beta;
                            double value = (*coefficient_direction)[group] *
                                           (parity_ && row != column ? 2 : 1);
                            if (!beta_side && project)
                                value *= exchanged ? 0.5 * parity_ : 0.5;
                            const Eigen::Index index =
                                beta_side ? (exchanged ? row : column) : (exchanged ? column : row);
                            image(index, j) = value;
                        }
                    }
                    if (varying) {
                        if (beta_side) {
                            beta.apply_left_block(image);
                            bb += image;
                        } else {
                            alpha.apply_left_block(image);
                            aa += image;
                        }
                    }
                    first += count;
                    offset += count;
                }
            },
            std::max(base.workspace_rows,
                     static_cast<std::size_t>(std::max(base.n_alpha, base.n_beta))),
            1.5L * base.load_work + sigma.spin_link_count() + base.n_alpha + base.n_beta};
}

mcscf::DirectCiIntegralAdjoint
SpinSources::integral_tangent(const mcscf::DirectCiSigmaAction& sigma,
                              const mcscf::SpinFactorSource& state,
                              const mcscf::SpinFactorSource& physical_state,
                              const mcscf::SpinFactorSource& direction,
                              const mcscf::SpinFactorSource& physical_direction) const {
    if (parity_) {
        const auto directional_bra = mcscf::detail::factor_integral_work(
            sigma, physical_direction, state, mcscf::kFactorWorkspaceBytes);
        const auto accepted_bra = mcscf::detail::factor_integral_work(
            sigma, physical_state, direction, mcscf::kFactorWorkspaceBytes);
        if (profiling::phase_totals().enabled)
            std::fprintf(stderr,
                         "XMVB_TANGENT_ORDER,directional_bra_work=%.0Lf,accepted_bra_work=%.0Lf,"
                         "accepted_bra=%d\n",
                         directional_bra,
                         accepted_bra,
                         accepted_bra < directional_bra);
        if (accepted_bra < directional_bra)
            return mcscf::factor_integral_adjoint(sigma, physical_state, direction);
    }
    return mcscf::factor_integral_adjoint(sigma, physical_direction, state);
}

} // namespace xmvb::vb
