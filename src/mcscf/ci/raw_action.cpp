#include <cstdint>
#include "mcscf/ci/raw_action.hpp"
#include "mcscf/ci/canonical.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <numeric>
#include <stdexcept>

#include "parallel/openmp.hpp"

namespace xmvb::mcscf {
namespace {
using Mask = std::uint64_t;

/** @brief Compose at most two existing single-link addresses, without reranking. */
StringIndex replacement_index(const SpinConnections& spin, StringIndex source, Mask target) {
    Mask removed = spin.basis.masks()[source] & ~target;
    Mask created = target & ~spin.basis.masks()[source];
    while (removed) {
        const int q = std::countr_zero(removed), p = std::countr_zero(created);
        const int high = std::max(p, q), low = std::min(p, q);
        const int link = spin.link_prefix(source, high, (Mask{1} << low) - 1);
        source = spin.singles(source)[link].source;
        removed &= removed - 1;
        created &= created - 1;
    }
    return source;
}

/** @brief Stream unique two-replacement masks; no double-link table is retained. */
template <class Function> void doubles(Mask source, Mask full, Function&& consume) {
    const Mask holes = full ^ source;
    for (Mask a = source; a; a &= a - 1)
        for (Mask b = a & (a - 1); b; b &= b - 1)
            for (Mask c = holes; c; c &= c - 1)
                for (Mask d = c & (c - 1); d; d &= d - 1)
                    consume(source ^ (a & -a) ^ (b & -b) ^ (c & -c) ^ (d & -d));
}

std::vector<Mask> subsets(Mask sites, int k) {
    if (!k)
        return {0};
    std::vector<Mask> result;
    for (Mask a = sites; a; a &= a - 1) {
        if (k == 1)
            result.push_back(a & -a);
        else
            for (Mask b = a & (a - 1); b; b &= b - 1)
                result.push_back((a & -a) | (b & -b));
    }
    return result;
}
} // namespace

RawCiWork RawCiPlan::work(const SpinConnections& alpha,
                          const SpinConnections& beta,
                          std::size_t source_products) {
    const long double n = alpha.orbitals, a = alpha.n_occupied(), b = beta.n_occupied();
    const auto choose2 = [](long double k) { return k * (k - 1) / 2; };
    const auto sa = a * (n - a), sb = b * (n - b);
    const auto pure = choose2(a) * choose2(n - a) + choose2(b) * choose2(n - b);
    const auto mixed = sa * sb, edges = 1 + sa + sb + pure + mixed;
    const auto diagonal = a + b + ((a + b) * (a + b) + a * a + b * b) / 2;
    const auto singles = sa * (1 + a + b + a) + sb * (1 + a + b + b);
    const auto terms = diagonal + singles + 2 * pure + mixed;
    // Four difference masks, two population counts, up to four bit positions
    // and two integral addresses per edge, in addition to integral multiply-adds.
    const auto source = source_products * (2 * terms + (4 + 2 + 4 + 2) * edges);
    static const int packet = []() -> int {
#ifdef EIGEN_USE_BLAS
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
        if (__builtin_cpu_supports("avx512f"))
            return 8;
        if (__builtin_cpu_supports("avx2"))
            return 4;
#endif
#endif
        return Eigen::internal::packet_traits<double>::size;
    }();
    const auto pairs = n * (n + 1) / 2;
    // Two spin-channel contractions and their adjoints; each multiply-add
    // counts two FLOPs. This is a schedule model, never an accuracy criterion.
    const auto channel =
        4 * alpha.strings * static_cast<long double>(beta.strings) * pairs * pairs / packet;
    return {source, channel};
}

RawCiPlan::RawCiPlan(const SpinConnections& alpha,
                     const SpinConnections& beta,
                     std::span<const ProductIndex> products)
    : alpha_(alpha), beta_(beta), products_(products), offsets_(beta.strings + 1, 0),
      sources_(products.size()) {
    if (!alpha.basis.complete() || !beta.basis.complete() || alpha.orbitals != beta.orbitals ||
        products.empty() || !std::is_sorted(products.begin(), products.end()) ||
        std::adjacent_find(products.begin(), products.end()) != products.end())
        throw std::invalid_argument("invalid raw CI source topology");
    alpha_offsets_.assign(alpha.strings + 1, 0);
    std::vector<StringIndex> columns;
    covalent_ = alpha.n_occupied() + beta.n_occupied() == alpha.orbitals &&
                products.size() == static_cast<std::size_t>(alpha.strings);
    const Mask full = (Mask{1} << alpha.orbitals) - 1;
    for (const auto product : products) {
        if (product < 0 || product >= static_cast<ProductIndex>(alpha.strings) * beta.strings)
            throw std::out_of_range("raw CI source product is outside its spin spaces");
        const auto a = product / beta.strings, b = product % beta.strings;
        ++alpha_offsets_[a + 1];
        if (!offsets_[b + 1]++)
            columns.push_back(b);
        covalent_ &= (alpha.basis.masks()[a] ^ beta.basis.masks()[b]) == full;
    }
    std::partial_sum(alpha_offsets_.begin(), alpha_offsets_.end(), alpha_offsets_.begin());
    std::partial_sum(offsets_.begin(), offsets_.end(), offsets_.begin());
    auto positions = offsets_;
    for (std::size_t i = 0; i < products.size(); ++i)
        sources_[positions[products[i] % beta.strings]++] = i;
    columns_ = columns;
    std::sort(columns_.begin(), columns_.end());
    if (columns.size() == static_cast<std::size_t>(beta.strings)) {
        targets_.resize(beta.strings);
        std::iota(targets_.begin(), targets_.end(), StringIndex{0});
        return;
    }
    std::vector<std::uint64_t> seen((static_cast<std::size_t>(beta.strings) + 63) / 64);
    const auto append = [&](StringIndex b) {
        const auto bit = Mask{1} << (b % 64);
        if (!(seen[b / 64] & bit)) {
            seen[b / 64] |= bit;
            targets_.push_back(b);
        }
    };
    for (const int b : columns) {
        append(b);
        for (const auto& link : beta.singles(b))
            append(link.source);
        doubles(beta.basis.masks()[b], full, [&](Mask mask) {
            append(replacement_index(beta, b, mask));
        });
    }
    std::sort(targets_.begin(), targets_.end());
}

std::size_t RawCiPlan::bytes() const noexcept {
    return sizeof(std::size_t) *
               (offsets_.capacity() + sources_.capacity() + alpha_offsets_.capacity()) +
           sizeof(int) * (targets_.capacity() + columns_.capacity());
}

void RawCiPlan::validate(const Eigen::Ref<const Eigen::MatrixXd>& values,
                         std::span<const StringIndex> beta,
                         const Eigen::Ref<Eigen::MatrixXd>& output) const {
    if (values.rows() != static_cast<Eigen::Index>(products_.size()) || values.cols() <= 0 ||
        output.rows() != alpha_.strings ||
        output.cols() != static_cast<Eigen::Index>(beta.size()) * values.cols())
        throw std::invalid_argument("raw CI panel dimensions do not match its source");
    for (const int b : beta)
        if (b < 0 || b >= beta_.strings)
            throw std::out_of_range("raw CI panel beta index is outside its spin space");
}

void RawCiPlan::source_columns(const Eigen::Ref<const Eigen::MatrixXd>& values,
                               std::span<const StringIndex> beta,
                               Eigen::Ref<Eigen::MatrixXd> output) const {
    validate(values, beta, output);
    output.setZero();
    const int rhs = values.cols();
#pragma omp parallel for schedule(static)
    for (std::size_t j = 0; j < beta.size(); ++j)
        for (std::size_t s = offsets_[beta[j]]; s < offsets_[beta[j] + 1]; ++s) {
            const auto input = sources_[s];
            const auto a = products_[input] / beta_.strings;
            for (int v = 0; v < rhs; ++v)
                output(a, j * rhs + v) = values(input, v);
        }
}

template <class Function>
std::size_t RawCiPlan::visit_column(StringIndex b, Function&& consume) const {
    if (covalent_) {
        const Mask full = (Mask{1} << alpha_.orbitals) - 1, bm = beta_.basis.masks()[b],
                   holes = full ^ bm;
        const auto original = alpha_.basis.index(holes);
        std::size_t work = 0;
        for (int k = 0; k <= std::min({2, alpha_.n_occupied(), beta_.n_occupied()}); ++k) {
            const auto ds = subsets(bm, k), es = subsets(holes, k);
            for (const Mask d : ds)
                for (const Mask e : es) {
                    const Mask a = holes ^ d ^ e;
                    const auto row = replacement_index(alpha_, original, a);
                    const auto add = [&](Mask source) {
                        const auto input = replacement_index(alpha_, row, source);
                        consume(row, input, a, bm, source, full ^ source);
                        ++work;
                    };
                    canonical::covalent_sources(a, bm, full, add);
                }
        }
        return work;
    }
    const Mask full = (Mask{1} << alpha_.orbitals) - 1, tb = beta_.basis.masks()[b];
    std::size_t work = 0;
    const auto visit_beta = [&](StringIndex source_beta, int distance) {
        const Mask sb = beta_.basis.masks()[source_beta];
        for (auto s = offsets_[source_beta]; s < offsets_[source_beta + 1]; ++s) {
            const auto input = sources_[s];
            const auto source_alpha = products_[input] / beta_.strings;
            const Mask sa = alpha_.basis.masks()[source_alpha];
            const auto add = [&](StringIndex a, Mask ta) {
                consume(a, input, ta, tb, sa, sb);
                ++work;
            };
            add(source_alpha, sa);
            if (distance < 2)
                for (const auto& link : alpha_.singles(source_alpha))
                    add(link.source, alpha_.basis.masks()[link.source]);
            if (!distance)
                doubles(sa, full, [&](Mask ta) {
                    add(replacement_index(alpha_, source_alpha, ta), ta);
                });
        }
    };
    visit_beta(b, 0);
    for (const auto& link : beta_.singles(b))
        visit_beta(link.source, 1);
    doubles(tb, full, [&](Mask sb) { visit_beta(replacement_index(beta_, b, sb), 2); });
    return work;
}

std::size_t RawCiPlan::apply_columns(const SimilarityHamiltonian& h,
                                     const Eigen::Ref<const Eigen::MatrixXd>& values,
                                     std::span<const StringIndex> beta,
                                     Eigen::Ref<Eigen::MatrixXd> output,
                                     const RawCiAdjoint* adjoint,
                                     std::span<const RawCiDirection> directions) const {
    validate(values, beta, output);
    if (h.orbitals() != alpha_.orbitals)
        throw std::invalid_argument("raw CI orbital dimensions differ");
    output.setZero();
    for (const auto& d : directions)
        d.image.setZero(output.rows(), output.cols());
    const int n = alpha_.orbitals, rhs = values.cols();
    const int workers = std::min<int>(cooperative_openmp_thread_count(), beta.size());
    struct Gradient {
        Eigen::MatrixXd one, two;
    };
    const auto make = [&] {
        return Gradient{Eigen::MatrixXd::Zero(n, n), Eigen::MatrixXd::Zero(n * n, n * n)};
    };
    std::vector<Gradient> primal;
    if (adjoint && adjoint->value.one)
        primal.assign(workers, make());
    std::vector<std::vector<Gradient>> gradients(directions.size());
    for (std::size_t l = 0; l < directions.size(); ++l)
        if (directions[l].cotangent.one)
            gradients[l].assign(workers, make());
    std::vector<std::size_t> work(workers);
    cooperative_openmp_for(workers, workers, [&](std::ptrdiff_t w) {
        const CanonicalSeed seed{primal.empty() ? nullptr : &primal[w].one,
                                 primal.empty() ? nullptr : &primal[w].two};
        std::vector<CanonicalDirection> lanes(directions.size());
        for (std::size_t l = 0; l < directions.size(); ++l) {
            lanes[l].op = directions[l].op;
            if (!gradients[l].empty())
                lanes[l].seed = {&gradients[l][w].one, &gradients[l][w].two};
        }
        for (std::size_t j = beta.size() * w / workers; j < beta.size() * (w + 1) / workers; ++j)
            work[w] += visit_column(
                beta[j], [&](int a, std::size_t input, Mask ta, Mask tb, Mask sa, Mask sb) {
                    double weight = 0;
                    if (seed.one)
                        for (int v = 0; v < rhs; ++v)
                            weight += adjoint->seeds(a, j * rhs + v) * values(input, v);
                    for (std::size_t l = 0; l < directions.size(); ++l) {
                        const auto& d = directions[l];
                        auto& lane = lanes[l];
                        lane.weight = 0;
                        if (lane.seed.one)
                            for (int v = 0; v < rhs; ++v) {
                                lane.weight += d.seed(a, j * rhs + v) * values(input, v);
                                if (d.values)
                                    lane.weight += adjoint->seeds(a, j * rhs + v) *
                                                   (*d.values)(input, d.first_column + v);
                            }
                    }
                    const double element = (adjoint || !directions.empty())
                                               ? h.element_jet(ta, tb, sa, sb, weight, seed, lanes)
                                               : h.element(ta, tb, sa, sb);
                    for (int v = 0; v < rhs; ++v) {
                        output(a, j * rhs + v) += element * values(input, v);
                        for (std::size_t l = 0; l < directions.size(); ++l) {
                            const auto& d = directions[l];
                            d.image(a, j * rhs + v) +=
                                lanes[l].value * values(input, v) +
                                (d.values ? element * (*d.values)(input, d.first_column + v) : 0.0);
                        }
                    }
                });
    });
    for (int w = 0; w < workers; ++w) {
        if (!primal.empty()) {
            *adjoint->value.one += primal[w].one;
            *adjoint->value.two += primal[w].two;
        }
        for (std::size_t l = 0; l < directions.size(); ++l)
            if (!gradients[l].empty()) {
                *directions[l].cotangent.one += gradients[l][w].one;
                *directions[l].cotangent.two += gradients[l][w].two;
            }
    }
    return std::accumulate(work.begin(), work.end(), std::size_t{0});
}

void RawCiPlan::multiply_source(const Eigen::Ref<const Eigen::VectorXd>& values,
                                const Eigen::Ref<const Eigen::MatrixXd>& input,
                                Eigen::Ref<Eigen::MatrixXd> output,
                                bool transpose) const {
    const auto rows = transpose ? beta_.strings : alpha_.strings;
    if (values.size() != static_cast<Eigen::Index>(products_.size()) ||
        input.rows() != (transpose ? alpha_.strings : beta_.strings) || output.rows() != rows ||
        output.cols() != input.cols())
        throw std::invalid_argument("invalid sparse metric seed dimensions");
    output.setZero();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const auto first = transpose ? offsets_[i] : alpha_offsets_[i];
        const auto last = transpose ? offsets_[i + 1] : alpha_offsets_[i + 1];
        for (auto s = first; s < last; ++s) {
            const auto p = transpose ? sources_[s] : s;
            const auto j = transpose ? products_[p] / beta_.strings : products_[p] % beta_.strings;
            output.row(i).noalias() += values[p] * input.row(j);
        }
    }
}
Eigen::MatrixXd RawCiPlan::one_body_moment(const Eigen::Ref<const Eigen::MatrixXd>& values,
                                           std::span<const StringIndex> beta,
                                           const Eigen::Ref<const Eigen::MatrixXd>& seeds) const {
    const int n = alpha_.orbitals, rhs = values.cols();
    const int workers = std::min<int>(cooperative_openmp_thread_count(), beta.size());
    std::vector<Eigen::MatrixXd> local(workers, Eigen::MatrixXd::Zero(n, n));
    cooperative_openmp_for(workers, workers, [&](std::ptrdiff_t w) {
        auto& out = local[w];
        for (std::size_t j = beta.size() * w / workers; j < beta.size() * (w + 1) / workers; ++j) {
            const std::int64_t b = beta[j];
            const auto tb = beta_.basis.masks()[b];
            const auto weight = [&](std::size_t s, int a) {
                double v = 0;
                for (int k = 0; k < rhs; ++k)
                    v += values(s, k) * seeds(a, j * rhs + k);
                return v;
            };
            for (auto s = offsets_[b]; s < offsets_[b + 1]; ++s) {
                const auto source = sources_[s];
                const auto a = products_[source] / beta_.strings;
                const auto sa = alpha_.basis.masks()[a];
                const double v = weight(source, a);
                for (auto spin : {sa, tb})
                    for (const int p : OccupiedOrbitals(spin))
                        out(p, p) += v;
                for (const auto& link : alpha_.singles(a)) {
                    const auto ta = alpha_.basis.masks()[link.source];
                    const int p = std::countr_zero(ta & ~sa), q = std::countr_zero(sa & ~ta);
                    out(p, q) += canonical::single_sign(sa, q, p) * weight(source, link.source);
                }
            }
            for (const auto& link : beta_.singles(b)) {
                const std::int64_t sb = link.source;
                const auto mask = beta_.basis.masks()[sb];
                const int p = std::countr_zero(tb & ~mask), q = std::countr_zero(mask & ~tb);
                const double sign = canonical::single_sign(mask, q, p);
                for (auto s = offsets_[sb]; s < offsets_[sb + 1]; ++s) {
                    const auto source = sources_[s];
                    out(p, q) += sign * weight(source, products_[source] / beta_.strings);
                }
            }
        }
    });
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n, n);
    for (const auto& m : local)
        out += m;
    return out;
}
} // namespace xmvb::mcscf
