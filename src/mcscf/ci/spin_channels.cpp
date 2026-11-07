#include <cstdint>
#include "mcscf/ci/sigma.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

#include "parallel/openmp.hpp"
#include "core/timing.hpp"
#include "core/pair_index.hpp"

namespace xmvb::mcscf {
namespace {

/** @brief Packed boundary decoded once per channel tile, not per string/RHS. */
struct ChannelBoundary {
    int high;
    std::uint64_t low_mask;
};
ChannelBoundary channel_boundary(int pair) {
    int high = 0;
    while ((high + 1) * (high + 2) / 2 <= pair)
        ++high;
    const int low = pair - high * (high + 1) / 2;
    return {high, (std::uint64_t{1} << low) - 1};
}

/** @brief Orbital interval whose packed diagonal channels belong to this tile. */
std::pair<int, int> diagonal_range(int n, int first, int last) {
    std::int64_t begin = 0, end = 0;
    while (begin < n && core::PairIndex::packed_pair_index(begin, begin) < first)
        ++begin;
    end = begin;
    while (end < n && core::PairIndex::packed_pair_index(end, end) < last)
        ++end;
    return {begin, end};
}

} // namespace

Eigen::MatrixXd
DirectCiSigmaAction::spin_generator_adjoint(bool beta,
                                            const Eigen::Ref<const Eigen::MatrixXd>& left,
                                            const Eigen::Ref<const Eigen::MatrixXd>& right) const {
    const SpinGeneratorTerm terms[]{{left, right}};
    return spin_generator_adjoint(beta, terms);
}

Eigen::MatrixXd
DirectCiSigmaAction::spin_generator_adjoint(bool beta,
                                            std::span<const SpinGeneratorTerm> terms) const {
    profiling::ScopedPhase phase(profiling::Phase::CiGeneratorAdjoint);
    const auto& graph = beta ? beta_connections() : *alpha_;
    const auto rows = graph.n_strings();
    if (terms.empty())
        throw std::invalid_argument("missing single-spin generator-adjoint terms");
    const auto cols = terms.front().left.cols();
    for (const auto& term : terms)
        if (term.left.rows() != rows || term.right.rows() != rows || term.left.cols() != cols ||
            term.right.cols() != cols)
            throw std::invalid_argument("invalid single-spin generator-adjoint dimensions");
    const int workers = bounded_openmp_thread_count(
        coarse_openmp_thread_count(cooperative_openmp_thread_count(),
                                   static_cast<std::size_t>(rows) * cols),
        cols);
    std::vector<Eigen::MatrixXd> partials(workers, Eigen::MatrixXd::Zero(n_orbitals_, n_orbitals_));
    // Each worker traverses contiguous coefficient columns. Share the link
    // and its output address between the product-rule terms.
    const auto contract = [&]<std::size_t N>(const SpinGeneratorTerm* pairs) {
        cooperative_openmp_for(workers, workers, [&](std::ptrdiff_t worker) {
            auto& density = partials[worker];
            for (Eigen::Index col = cols * worker / workers; col < cols * (worker + 1) / workers;
                 ++col) {
                std::array<const double*, N> left, right;
                for (std::size_t l = 0; l < N; ++l) {
                    left[l] = pairs[l].left.col(col).data();
                    right[l] = pairs[l].right.col(col).data();
                }
                for (Eigen::Index target = 0; target < rows; ++target) {
                    const auto value = [&](Eigen::Index source) {
                        double sum = 0;
                        for (std::size_t l = 0; l < N; ++l)
                            sum += left[l][target] * right[l][source];
                        return sum;
                    };
                    const double diagonal = value(target);
                    for (const int occupied : graph.occupied(target))
                        density(occupied, occupied) += diagonal;
                    for (const auto& link : graph.singles(target))
                        density(link.created, link.annihilated) += link.sign * value(link.source);
                }
            }
        });
    };
    for (std::size_t l = 0; l + 1 < terms.size(); l += 2)
        contract.template operator()<2>(terms.data() + l);
    if (terms.size() % 2)
        contract.template operator()<1>(&terms.back());
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n_orbitals_, n_orbitals_);
    for (const auto& partial : partials)
        result += partial;
    return result;
}

void DirectCiSigmaAction::apply_spin_generator(bool beta,
                                               const Eigen::Ref<const Eigen::MatrixXd>& factors,
                                               const Eigen::Ref<const Eigen::MatrixXd>& generator,
                                               Eigen::Ref<Eigen::MatrixXd> image) const {
    const auto& graph = beta ? beta_connections() : *alpha_;
    const int rows = beta ? n_beta_ : n_alpha_;
    if (factors.rows() != rows || image.rows() != rows || image.cols() != factors.cols() ||
        generator.rows() != n_orbitals_ || generator.cols() != n_orbitals_)
        throw std::invalid_argument("invalid single-spin generator dimensions");
    const int workers = coarse_openmp_thread_count(effective_openmp_thread_count(),
                                                   static_cast<std::size_t>(rows) * factors.cols());
#pragma omp parallel for schedule(static) if (workers > 1) num_threads(workers)
    for (std::int64_t index = 0; index < rows * factors.cols(); ++index) {
        const std::int64_t column = index / rows, target = index % rows;
        double value = 0;
        for (int occupied : graph.occupied(target))
            value += generator(occupied, occupied) * factors(target, column);
        const auto& links = graph.singles(target);
        for (std::size_t link = 0; link < links.size(); ++link)
            value += links[link].sign * generator(links[link].created, links[link].annihilated) *
                     factors(links[link].source, column);
        image(target, column) = value;
    }
}

void DirectCiSigmaAction::scatter_spin(bool beta,
                                       const Eigen::Ref<const Eigen::MatrixXd>& factors,
                                       int first,
                                       int count,
                                       Eigen::Ref<Eigen::MatrixXd> result) const {
    const auto& graph = beta ? beta_connections() : *alpha_;
    const int rows = beta ? n_beta_ : n_alpha_;
    if (factors.rows() != rows || factors.cols() <= 0 || first < 0 || count <= 0 ||
        first + count > hamiltonian_pair_kernel_.rows() || result.rows() != rows ||
        result.cols() != factors.cols() * count)
        throw std::invalid_argument("invalid single-spin channel scatter");
    const int width = factors.cols();
    const auto diagonal = diagonal_range(n_orbitals_, first, first + count);
    const auto begin = channel_boundary(first), end = channel_boundary(first + count);
    result.setZero();
    const int workers = coarse_openmp_thread_count(effective_openmp_thread_count(), result.size());
#pragma omp parallel for schedule(static) if (workers > 1) num_threads(workers)
    for (std::int64_t index = 0; index < rows * width; ++index) {
        const std::int64_t target = index % rows, column = index / rows;
        const auto add = [&](std::int64_t source, int pair, double sign) {
            result(target, (pair - first) * width + column) += sign * factors(source, column);
        };
        const auto range_mask = ((std::uint64_t{1} << diagonal.second) - 1) ^
                                ((std::uint64_t{1} << diagonal.first) - 1);
        for (const int orbital : OccupiedOrbitals(graph.basis.masks()[target] & range_mask)) {
            add(target, core::PairIndex::packed_pair_index(orbital, orbital), 1.0);
        }
        const auto& links = graph.singles(target);
        const int first_link = graph.link_prefix(target, begin.high, begin.low_mask);
        const int last_link = graph.link_prefix(target, end.high, end.low_mask);
        for (int i = first_link; i < last_link; ++i)
            add(links[i].source, links[i].pair, links[i].sign);
    }
}

void DirectCiSigmaAction::gather_spin(bool beta,
                                      const Eigen::Ref<const Eigen::MatrixXd>& channels,
                                      int first,
                                      int count,
                                      Eigen::Ref<Eigen::MatrixXd> images) const {
    const auto& graph = beta ? beta_connections() : *alpha_;
    const int rows = beta ? n_beta_ : n_alpha_;
    if (images.rows() != rows || images.cols() <= 0 || channels.rows() != rows ||
        channels.cols() != images.cols() * count || first < 0 || count <= 0 ||
        first + count > hamiltonian_pair_kernel_.rows())
        throw std::invalid_argument("invalid single-spin channel gather");
    const int width = images.cols();
    const auto diagonal = diagonal_range(n_orbitals_, first, first + count);
    const auto begin = channel_boundary(first), end = channel_boundary(first + count);
    const int workers =
        coarse_openmp_thread_count(effective_openmp_thread_count(), channels.size());
#pragma omp parallel for schedule(static) if (workers > 1) num_threads(workers)
    for (std::int64_t index = 0; index < rows * width; ++index) {
        const std::int64_t target = index % rows, column = index / rows;
        const auto add = [&](std::int64_t source, int pair, double sign) {
            images(target, column) += sign * channels(source, (pair - first) * width + column);
        };
        const auto range_mask = ((std::uint64_t{1} << diagonal.second) - 1) ^
                                ((std::uint64_t{1} << diagonal.first) - 1);
        for (const int orbital : OccupiedOrbitals(graph.basis.masks()[target] & range_mask)) {
            add(target, core::PairIndex::packed_pair_index(orbital, orbital), 1.0);
        }
        const auto& links = graph.singles(target);
        const int first_link = graph.link_prefix(target, begin.high, begin.low_mask);
        const int last_link = graph.link_prefix(target, end.high, end.low_mask);
        for (int i = first_link; i < last_link; ++i)
            add(links[i].source, links[i].pair, links[i].sign);
    }
}

} // namespace xmvb::mcscf
