#include "vbscf/structures/overlap/spin_metric.hpp"

#include "mcscf/ci/string_table.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <cblas.h>
#include <lapacke.h>

#include "core/pair_index.hpp"
#include "parallel/openmp.hpp"
#include "vbscf/structures/assembly/action.hpp"

namespace xmvb::vb {
namespace {

auto occupation_pattern(mcscf::ProductIndex product,
                        std::span<const std::uint64_t> alpha,
                        std::span<const std::uint64_t> beta) {
    const auto a = alpha[product / beta.size()], b = beta[product % beta.size()];
    return std::pair{a | b, a & b};
}

bool is_contiguous(const std::vector<Eigen::Index>& indices) {
    return indices.back() - indices.front() + 1 == static_cast<Eigen::Index>(indices.size());
}

bool gram_blocks_fit(const std::vector<std::vector<Eigen::Index>>& groups,
                     std::size_t byte_limit) {
    std::size_t bytes = 0;
    for (const auto& group : groups) {
        const std::size_t d = group.size();
        if (d > (byte_limit - bytes) / sizeof(double) / d)
            return false;
        bytes += d * d * sizeof(double);
    }
    return true;
}

} // namespace

std::vector<std::vector<Eigen::Index>>
SpinMetric::occupation_groups(const StructureAction& action,
                              const xmvb::mcscf::StringTable& alpha_strings,
                              const xmvb::mcscf::StringTable& beta_strings) {
    if (alpha_strings.size() != static_cast<std::size_t>(action.n_unique_alpha_) ||
        beta_strings.size() != static_cast<std::size_t>(action.n_unique_beta_))
        throw std::invalid_argument("structure metric spin dimensions disagree");
    using Pattern = std::pair<std::uint64_t, std::uint64_t>;
    const auto alpha_masks = alpha_strings.masks(), beta_masks = beta_strings.masks();
    const auto pattern = [&](mcscf::ProductIndex spin_product) {
        return occupation_pattern(spin_product, alpha_masks, beta_masks);
    };
    std::map<Pattern, std::vector<Eigen::Index>> groups;
    for (Eigen::Index structure = 0; structure < action.n_structures_; ++structure) {
        const auto first = action.topology_->structure_offsets[structure];
        const auto last = action.topology_->structure_offsets[structure + 1];
        if (first == last)
            throw std::invalid_argument("structure metric expansion has an empty column");
        const Pattern occupation =
            pattern(action.topology_->products
                        [action.topology_->terms[action.topology_->structure_terms[first]].group]);
        for (auto position = first + 1; position < last; ++position) {
            if (pattern(action.topology_
                            ->products[action.topology_
                                           ->terms[action.topology_->structure_terms[position]]
                                           .group]) != occupation)
                throw std::invalid_argument("structure spans multiple spatial occupation patterns");
        }
        groups[occupation].push_back(structure);
    }
    std::vector<std::vector<Eigen::Index>> result;
    result.reserve(groups.size());
    for (auto& group : groups)
        result.push_back(std::move(group.second));
    return result;
}

bool SpinMetric::fits(const StructureAction& action,
                      const mcscf::StringTable& alpha,
                      const mcscf::StringTable& beta,
                      std::size_t byte_limit) {
    return gram_blocks_fit(occupation_groups(action, alpha, beta), byte_limit);
}

std::shared_ptr<SpinMetric> SpinMetric::try_build(const StructureAction& action,
                                                const mcscf::StringTable& alpha,
                                                const mcscf::StringTable& beta,
                                                std::size_t byte_limit) {
    auto groups = occupation_groups(action, alpha, beta);
    if (!gram_blocks_fit(groups, byte_limit))
        return {};
    return std::shared_ptr<SpinMetric>(
        new SpinMetric(action, std::move(groups), alpha, beta, byte_limit));
}

SpinMetric::SpinMetric(const StructureAction& action,
                       const xmvb::mcscf::StringTable& alpha_strings,
                       const xmvb::mcscf::StringTable& beta_strings,
                       std::size_t factor_byte_limit)
    : SpinMetric(action,
                   occupation_groups(action, alpha_strings, beta_strings),
                   alpha_strings,
                   beta_strings,
                   factor_byte_limit) {}

SpinMetric::SpinMetric(const StructureAction& action,
                       std::vector<std::vector<Eigen::Index>> groups,
                       const mcscf::StringTable& alpha_strings,
                       const mcscf::StringTable& beta_strings,
                       std::size_t factor_byte_limit)
    : dimension_(action.n_structures_), n_alpha_(alpha_strings.electron_count()),
      n_beta_(beta_strings.electron_count()), diagonal_(Eigen::VectorXd::Zero(dimension_)) {
    const std::size_t n = action.n_structures_;
    // Account for the uninterned Gram blocks during assembly.
    for (const auto& group : groups) {
        const std::size_t d = group.size();
        if (d > (factor_byte_limit - factor_bytes_) / sizeof(double) / d)
            throw std::length_error("spin-recoupling factors exceed their byte budget");
        factor_bytes_ += d * d * sizeof(double);
    }
    if (!residency_.try_resize(factor_bytes_))
        throw std::runtime_error("spin-recoupling factors exceed shared workspace headroom");
    blocks_.reserve(groups.size());
    std::vector<Eigen::MatrixXd> grams;
    grams.reserve(groups.size());
    std::vector<Eigen::Index> group_id(n), local_id(n);
    for (auto& group : groups) {
        const Eigen::Index index = core::checked_index<Eigen::Index>(blocks_.size());
        const Eigen::Index d = core::checked_index<Eigen::Index>(group.size());
        for (Eigen::Index local = 0; local < d; ++local) {
            group_id[group[local]] = index;
            local_id[group[local]] = local;
        }
        const auto first = action.topology_->structure_offsets[group.front()];
        const auto term = action.topology_->structure_terms[first];
        const auto [occupied, doubled] =
            occupation_pattern(action.topology_->products[action.topology_->terms[term].group],
                               alpha_strings.masks(),
                               beta_strings.masks());
        blocks_.push_back({std::move(group), 0, occupied, doubled});
        grams.push_back(Eigen::MatrixXd::Zero(d, d));
        max_block_size_ = std::max(max_block_size_, d);
    }
    // C_0^T C_0 from sparse spin-product rows; never form the dense C_0 or Gram.
    for (std::size_t product = 0; product < action.topology_->products.size(); ++product) {
        const auto first = action.topology_->offsets[product];
        const auto last = action.topology_->offsets[product + 1];
        for (auto left = first; left < last; ++left) {
            const auto& a = action.topology_->terms[left];
            for (auto right = first; right < last; ++right) {
                const auto& b = action.topology_->terms[right];
                if (group_id[a.structure] != group_id[b.structure])
                    throw std::logic_error(
                        "one spin product touches inconsistent occupation blocks");
                const Eigen::Index row = local_id[a.structure];
                const Eigen::Index column = local_id[b.structure];
                if (row >= column)
                    grams[group_id[a.structure]](row, column) += a.coefficient * b.coefficient;
            }
        }
    }
    factor_bytes_ = 0;
    {
        std::unordered_multimap<std::size_t, std::size_t> candidates;
        for (std::size_t i = 0; i < blocks_.size(); ++i) {
            auto& block = blocks_[i];
            auto& gram = grams[i];
            for (Eigen::Index row = 0; row < gram.rows(); ++row)
                diagonal_[block.indices[row]] = gram(row, row);
            // Equal occupation size alone does not fix the basis order or phase.
            // Compare the actual Gram entries, with no tolerance or canonical rephasing.
            std::size_t hash = gram.rows();
            for (const double value : gram.reshaped())
                hash = core::IndexPairHash{}({hash, std::hash<double>{}(value)});
            const auto [first, last] = candidates.equal_range(hash);
            const auto match = std::find_if(first, last, [&](const auto& candidate) {
                const auto& other = factors_[candidate.second];
                return gram.rows() == other.rows() && (gram.array() == other.array()).all();
            });
            if (match == last) {
                block.factor = factors_.size();
                candidates.emplace(hash, block.factor);
                factor_bytes_ += static_cast<std::size_t>(gram.size()) * sizeof(double);
                factors_.push_back(std::move(gram));
            } else {
                block.factor = match->second;
                gram.resize(0, 0);
            }
        }
    }
    residency_.try_resize(factor_bytes_);
    for (auto& lower : factors_) {
        const auto d = core::checked_index<lapack_int>(lower.rows());
        const lapack_int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', d, lower.data(), d);
        if (info != 0 || !lower.allFinite())
            throw std::runtime_error(
                "spin-recoupling metric block is not numerically positive definite");
    }

    const auto scattered = std::partition(blocks_.begin(), blocks_.end(), [](const Block& block) {
        return is_contiguous(block.indices);
    });
    contiguous_blocks_ = static_cast<std::size_t>(scattered - blocks_.begin());
    std::sort(blocks_.begin(), scattered, [](const Block& a, const Block& b) {
        return a.indices.front() < b.indices.front();
    });
    std::sort(scattered, blocks_.end(), [](const Block& a, const Block& b) {
        return std::pair{a.factor, a.indices.front()} < std::pair{b.factor, b.indices.front()};
    });
    for (std::size_t first = 0; first < blocks_.size();) {
        const bool contiguous = first < contiguous_blocks_;
        const auto end = contiguous ? contiguous_blocks_ : blocks_.size();
        auto last = first + 1;
        while (
            last < end && blocks_[last].factor == blocks_[first].factor &&
            (!contiguous || blocks_[last - 1].indices.back() + 1 == blocks_[last].indices.front()))
            ++last;
        solve_ranges_.push_back({first, last});
        first = last;
    }
}

Eigen::MatrixXd SpinMetric::solve(const Eigen::Ref<const Eigen::MatrixXd>& rhs, Part factor) const {
    Eigen::MatrixXd result = rhs;
    solve_in_place(result, factor);
    return result;
}

void SpinMetric::solve_in_place(Eigen::Ref<Eigen::MatrixXd> rhs, Part factor) const {
    if (rhs.rows() != dimension_ || rhs.cols() <= 0)
        throw std::invalid_argument("spin metric RHS has incompatible dimensions");

    // Bound the scattered packing scratch independently of the number of blocks.
    constexpr Eigen::Index panel_values = 32768;
    Eigen::VectorXd packed;
    const auto solve_panel = [&](const Eigen::MatrixXd& lower, double* data, Eigen::Index width) {
        const auto d = core::checked_index<lapack_int>(lower.rows());
        if (d == 1) {
            auto values = Eigen::Map<Eigen::VectorXd>(data, width);
            values.array() /= lower(0, 0);
            if (factor == Part::Gram)
                values.array() /= lower(0, 0);
            return;
        }
        const auto columns = core::checked_index<lapack_int>(width);
        const auto triangular = [&](CBLAS_TRANSPOSE trans) {
            cblas_dtrsm(CblasColMajor,
                        CblasLeft,
                        CblasLower,
                        trans,
                        CblasNonUnit,
                        d,
                        columns,
                        1.0,
                        lower.data(),
                        d,
                        data,
                        d);
        };
        if (factor != Part::Upper)
            triangular(CblasNoTrans);
        if (factor != Part::Lower)
            triangular(CblasTrans);
    };
    for (const auto [first, last] : solve_ranges_) {
        const auto& lower = factors_[blocks_[first].factor];
        const Eigen::Index d = lower.rows();
        const bool contiguous = first < contiguous_blocks_;
        const auto width = contiguous ? std::numeric_limits<lapack_int>::max()
                                      : std::max<Eigen::Index>(1, panel_values / d);
        for (std::size_t begin = first; begin < last;) {
            const auto count = std::min(width, core::checked_index<Eigen::Index>(last - begin));
            if (!contiguous && packed.size() < d * count)
                packed.resize(d * count);
            for (Eigen::Index lane = 0; lane < rhs.cols(); ++lane) {
                auto column = rhs.col(lane);
                if (contiguous) {
                    solve_panel(lower, column.data() + blocks_[begin].indices.front(), count);
                    continue;
                }
                Eigen::Map<Eigen::MatrixXd> panel(packed.data(), d, count);
                for (Eigen::Index j = 0; j < count; ++j) {
                    const auto& indices = blocks_[begin + j].indices;
                    for (Eigen::Index i = 0; i < d; ++i)
                        panel(i, j) = column[indices[i]];
                }
                solve_panel(lower, packed.data(), count);
                for (Eigen::Index j = 0; j < count; ++j) {
                    const auto& indices = blocks_[begin + j].indices;
                    for (Eigen::Index i = 0; i < d; ++i)
                        column[indices[i]] = panel(i, j);
                }
            }
            begin += count;
        }
    }
}

Eigen::VectorXd SpinMetric::occupation_diagonal(const mcscf::ActiveIntegrals& integrals) const {
    using core::PairIndex;
    const auto n = integrals.one_electron.rows();
    const auto& eri = integrals.pair_kernel;
    Eigen::MatrixXd coulomb(n, n), exchange(n, n);
    for (int p = 0; p < n; ++p)
        for (int q = 0; q <= p; ++q) {
            const auto pp = PairIndex::packed_pair_index(p, p);
            const auto qq = PairIndex::packed_pair_index(q, q);
            const auto pq = PairIndex::packed_pair_index(p, q);
            coulomb(p, q) = eri(pp, qq);
            exchange(p, q) = eri(pq, pq);
        }

    Eigen::VectorXd diagonal(dimension_);
    const int workers = coarse_openmp_thread_count(effective_openmp_thread_count(), blocks_.size());
#pragma omp parallel for schedule(static) if (workers > 1) num_threads(workers)
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
        const auto& block = blocks_[i];
        const int doubles = std::popcount(block.doubled);
        const int a = n_alpha_ - doubles, b = n_beta_ - doubles;
        const int singles = a + b;
        // Two singly occupied sites have the same spin in this fraction of
        // all assignments at fixed alpha and beta electron counts.
        const double equal_spin = singles < 2 ? 0.0
                                              : static_cast<double>(a * (a - 1) + b * (b - 1)) /
                                                    (singles * (singles - 1));
        double energy = 0.0;
        for (auto orbitals = block.occupied; orbitals; orbitals &= orbitals - 1) {
            const int q = std::countr_zero(orbitals);
            const bool double_q = (block.doubled >> q) & 1;
            const int nq = 1 + double_q;
            energy += nq * integrals.one_electron(q, q) + double_q * coulomb(q, q);
            for (auto later = orbitals & (orbitals - 1); later; later &= later - 1) {
                const int p = std::countr_zero(later);
                const bool double_p = (block.doubled >> p) & 1;
                const double same_spin = double_p || double_q ? double_p + double_q : equal_spin;
                energy += (1 + double_p) * nq * coulomb(p, q) - same_spin * exchange(p, q);
            }
        }
        for (const auto row : block.indices)
            diagonal[row] = energy;
    }
    return diagonal;
}

Eigen::VectorXd SpinMetric::precondition(const Eigen::VectorXd& rhs,
                                         const Eigen::VectorXd& physical_diagonal) const {
    if (rhs.size() != dimension_ || physical_diagonal.size() != dimension_ ||
        !(physical_diagonal.array() > 0.0).all())
        throw std::invalid_argument("invalid spin metric preconditioner diagonal");
    Eigen::VectorXd result = solve(rhs).col(0);
    for (const auto& block : blocks_) {
        double spin_trace = 0.0, physical_trace = 0.0;
        for (const auto i : block.indices) {
            spin_trace += diagonal_[i];
            physical_trace += physical_diagonal[i];
        }
        const double scale = spin_trace / physical_trace;
        for (const auto i : block.indices)
            result[i] *= scale;
    }
    return result;
}

Eigen::MatrixXd
SpinMetric::solve_diagonal_model(const Eigen::Ref<const Eigen::MatrixXd>& rhs,
                                 const Eigen::Ref<const Eigen::VectorXd>& diagonal) const {
    if (diagonal.size() != dimension_ || !diagonal.allFinite() || !(diagonal.array() > 0.0).all())
        throw std::invalid_argument("spin model requires a positive finite diagonal");
    const Eigen::VectorXd scale = (diagonal_.array() / diagonal.array()).sqrt();
    const Eigen::MatrixXd scaled = rhs.array().colwise() * scale.array();
    Eigen::MatrixXd result = solve(scaled);
    result.array().colwise() *= scale.array();
    return result;
}

} // namespace xmvb::vb
