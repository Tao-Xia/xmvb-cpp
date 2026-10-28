#include "vbscf/structures/overlap/spin_metric.hpp"

#include "mcscf/ci/string_table.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

#include <lapacke.h>

#include "vbscf/structures/assembly/action.hpp"

namespace xmvb::vb {
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
        const auto alpha = alpha_masks[spin_product / action.n_unique_beta_];
        const auto beta = beta_masks[spin_product % action.n_unique_beta_];
        return Pattern{alpha | beta, alpha & beta};
    };
    std::map<Pattern, std::vector<Eigen::Index>> groups;
    for (Eigen::Index structure = 0; structure < action.n_structures_; ++structure) {
        const auto first = action.structure_term_offsets_[structure];
        const auto last = action.structure_term_offsets_[structure + 1];
        if (first == last)
            throw std::invalid_argument("structure metric expansion has an empty column");
        const Pattern occupation = pattern(
            action.spin_products_[action.spin_terms_[action.structure_term_indices_[first]].group]);
        for (auto position = first + 1; position < last; ++position) {
            if (pattern(action.spin_products_
                            [action.spin_terms_[action.structure_term_indices_[position]].group]) !=
                occupation)
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
    : dimension_(action.n_structures_), diagonal_(Eigen::VectorXd::Zero(dimension_)) {
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
        blocks_.push_back({std::move(group), Eigen::MatrixXd::Zero(d, d)});
        max_block_size_ = std::max(max_block_size_, d);
    }
    // C_0^T C_0 from sparse spin-product rows; never form the dense C_0 or Gram.
    for (std::size_t product = 0; product < action.spin_products_.size(); ++product) {
        const auto first = action.spin_term_offsets_[product];
        const auto last = action.spin_term_offsets_[product + 1];
        for (auto left = first; left < last; ++left) {
            const auto& a = action.spin_terms_[left];
            for (auto right = first; right < last; ++right) {
                const auto& b = action.spin_terms_[right];
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
        for (int row = 0; row < d; ++row)
            values.row(row) = rhs.row(block.indices[row]);
        if (factor != Part::Upper)
            block.lower.triangularView<Eigen::Lower>().solveInPlace(values);
        if (factor != Part::Lower)
            block.lower.transpose().triangularView<Eigen::Upper>().solveInPlace(values);
        for (int row = 0; row < d; ++row)
            result.row(block.indices[row]) = values.row(row);
    }
    return result;
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
