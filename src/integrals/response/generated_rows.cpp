#include "integrals/response/ao_pair_operator.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

namespace xmvb::integrals::detail {
namespace {

std::vector<std::pair<int, int>> active_pairs(int n_active) {
    std::vector<std::pair<int, int>> pairs;
    for (int i = 0; i < n_active; ++i)
        for (int j = 0; j <= i; ++j)
            pairs.emplace_back(i, j);
    return pairs;
}

void validate(const AoPairGraph& graph,
              const Eigen::Ref<const Eigen::MatrixXd>& c,
              Eigen::Index begin,
              Eigen::Index count) {
    const Eigen::Index pairs = c.rows() * (c.rows() + 1) / 2;
    if (c.rows() <= 0 || c.cols() <= 0 || begin < 0 || count < 0 || begin > pairs ||
        count > pairs - begin || graph.pair_first.size() != static_cast<std::size_t>(pairs) ||
        graph.pair_second.size() != static_cast<std::size_t>(pairs))
        throw std::invalid_argument("invalid generated AO-pair row dimensions");
}

double coefficient(const Eigen::Ref<const Eigen::MatrixXd>& c, int i, int j, int a, int b) {
    return c(i, a) * c(j, b) + (i == j ? 0.0 : c(j, a) * c(i, b));
}

} // namespace

void apply_generated_ao_pair_rows(const AoIntegralInput& ao,
                                  const Eigen::Ref<const Eigen::MatrixXd>& c,
                                  const Eigen::MatrixXd* d,
                                  Eigen::Index begin,
                                  Eigen::Index count,
                                  ExactCtxPairMatrix* image,
                                  ExactCtxPairMatrix* directional) {
    const auto& graph = ao.pair_graph;
    validate(graph, c, begin, count);
    if (!image || (d != nullptr) != (directional != nullptr) || c.rows() != ao.n_basis_functions ||
        (d && (d->rows() != c.rows() || d->cols() != c.cols())))
        throw std::invalid_argument("invalid generated AO-pair output or direction");
    const auto pairs = active_pairs(c.cols());
    image->setZero(count, pairs.size());
    if (directional)
        directional->setZero(count, pairs.size());
    const auto accumulate = [&](Eigen::Index target, int source, double integral) {
        const int i = graph.pair_first[source], j = graph.pair_second[source];
        double* output = image->data() + target * pairs.size();
        double* delta = directional ? directional->data() + target * pairs.size() : nullptr;
#pragma omp simd
        for (Eigen::Index p = 0; p < static_cast<Eigen::Index>(pairs.size()); ++p) {
            const int a = pairs[p].first, b = pairs[p].second;
            output[p] += integral * coefficient(c, i, j, a, b);
            if (d) {
                double mixed = (*d)(i, a) * c(j, b) + c(i, a) * (*d)(j, b);
                if (i != j)
                    mixed += (*d)(j, a) * c(i, b) + c(j, a) * (*d)(i, b);
                delta[p] += integral * mixed;
            }
        }
    };
    if (graph.has_row_graph()) {
        if (graph.row_offsets.size() != graph.pair_first.size() + 1 ||
            graph.columns.size() != graph.values.size() ||
            graph.values.size() != static_cast<std::size_t>(graph.row_offsets.back()))
            throw std::invalid_argument("invalid generated-row CSR graph");
        // Disjoint target rows: no private pair images and no scatter races.
#pragma omp parallel for schedule(dynamic, 1)
        for (Eigen::Index row = 0; row < count; ++row)
            for (PairAddress edge = graph.row_offsets[begin + row];
                 edge < graph.row_offsets[begin + row + 1];
                 ++edge)
                accumulate(row, graph.columns[edge], graph.values[edge]);
    } else if (graph.has_compact_integral_stream()) {
        if (!graph.valid_unique_storage())
            throw std::invalid_argument("invalid generated-row integral stream");
        // One stream traversal per tile, never one traversal per target row.
        for (std::size_t k = 0; k < graph.integral_count(); ++k) {
            const auto eri = graph.integral(k);
            if (eri.left_pair >= begin && eri.left_pair < begin + count)
                accumulate(eri.left_pair - begin, eri.right_pair, eri.value);
            if (eri.right_pair != eri.left_pair && eri.right_pair >= begin &&
                eri.right_pair < begin + count)
                accumulate(eri.right_pair - begin, eri.left_pair, eri.value);
        }
    } else
        throw std::invalid_argument("missing generated-row integral graph");
}

} // namespace xmvb::integrals::detail
