#include "vbscf/structures/overlap/spin_metric.hpp"

#include "mcscf/ci/string_table.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <utility>

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
    std::size_t bytes = 0;
    for (const auto& group : occupation_groups(action, alpha, beta)) {
        const std::size_t d = group.size();
        if (d > (byte_limit - bytes) / sizeof(double) / d)
            return false;
        bytes += d * d * sizeof(double);
    }
    return true;
}

SpinMetric::SpinMetric(const StructureAction& action,
                       const xmvb::mcscf::StringTable& alpha_strings,
                       const xmvb::mcscf::StringTable& beta_strings,
                       std::size_t factor_byte_limit)
    : dimension_(action.n_structures_), n_alpha_(alpha_strings.electron_count()),
      n_beta_(beta_strings.electron_count()), diagonal_(Eigen::VectorXd::Zero(dimension_)) {
    auto groups = occupation_groups(action, alpha_strings, beta_strings);
    const std::size_t n = action.n_structures_;
    // Admit all factor storage before allocating any dense block.
    for (const auto& group : groups) {
        const std::size_t d = group.size();
        if (d > (factor_byte_limit - factor_bytes_) / sizeof(double) / d)
            throw std::length_error("spin-recoupling factors exceed their byte budget");
        factor_bytes_ += d * d * sizeof(double);
    }
    if (!residency_.try_resize(factor_bytes_))
        throw std::runtime_error("spin-recoupling factors exceed shared workspace headroom");
    blocks_.reserve(groups.size());
    std::vector<Eigen::Index> group_id(n), local_id(n);
    for (auto& group : groups) {
        const Eigen::Index index = core::checked_index<Eigen::Index>(blocks_.size());
        const Eigen::Index d = core::checked_index<Eigen::Index>(group.size());
        for (int local = 0; local < d; ++local) {
            group_id[group[local]] = index;
            local_id[group[local]] = local;
        }
        const auto first = action.topology_->structure_offsets[group.front()];
        const auto term = action.topology_->structure_terms[first];
        const auto [occupied, doubled] =
            occupation_pattern(action.topology_->products[action.topology_->terms[term].group],
                               alpha_strings.masks(),
                               beta_strings.masks());
        blocks_.push_back({std::move(group), Eigen::MatrixXd::Zero(d, d), occupied, doubled});
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
                    blocks_[group_id[a.structure]].lower(row, column) +=
                        a.coefficient * b.coefficient;
            }
        }
    }
    for (auto& block : blocks_) {
        for (Eigen::Index i = 0; i < core::checked_index<Eigen::Index>(block.indices.size()); ++i)
            diagonal_[block.indices[i]] = block.lower(i, i);
        const lapack_int d = static_cast<lapack_int>(block.lower.rows());
        const lapack_int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', d, block.lower.data(), d);
        if (info != 0 || !block.lower.allFinite())
            throw std::runtime_error(
                "spin-recoupling metric block is not numerically positive definite");
    }
}

Eigen::MatrixXd SpinMetric::solve(const Eigen::Ref<const Eigen::MatrixXd>& rhs, Part factor) const {
    if (rhs.rows() != dimension_ || rhs.cols() <= 0)
        throw std::invalid_argument("spin metric RHS has incompatible dimensions");
    Eigen::MatrixXd result(rhs.rows(), rhs.cols());
    Eigen::MatrixXd local(max_block_size_, rhs.cols());
    for (const auto& block : blocks_) {
        const Eigen::Index d = core::checked_index<Eigen::Index>(block.indices.size());
        auto values = local.topRows(d);
        for (Eigen::Index row = 0; row < d; ++row)
            values.row(row) = rhs.row(block.indices[row]);
        if (factor != Part::Upper)
            block.lower.triangularView<Eigen::Lower>().solveInPlace(values);
        if (factor != Part::Lower)
            block.lower.transpose().triangularView<Eigen::Upper>().solveInPlace(values);
        for (Eigen::Index row = 0; row < d; ++row)
            result.row(block.indices[row]) = values.row(row);
    }
    return result;
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
        for (const int i : block.indices) {
            spin_trace += diagonal_[i];
            physical_trace += physical_diagonal[i];
        }
        const double scale = spin_trace / physical_trace;
        for (const int i : block.indices)
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
