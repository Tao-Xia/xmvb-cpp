#include "vbscf/structures/analysis/spin.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xmvb::vb {
namespace {

struct RaisedDeterminant {
    std::vector<std::uint64_t> alpha;
    std::vector<std::uint64_t> beta;

    bool operator==(const RaisedDeterminant& other) const noexcept {
        return alpha == other.alpha && beta == other.beta;
    }
};

struct RaisedDeterminantHash {
    std::size_t operator()(const RaisedDeterminant& determinant) const noexcept {
        std::size_t value = 0;
        for (const auto orbital : determinant.alpha) {
            value = value * 1315423911u + static_cast<std::size_t>(orbital + 257);
        }
        value = value * 2654435761u + 17u;
        for (const auto orbital : determinant.beta) {
            value = value * 1315423911u + static_cast<std::size_t>(orbital + 257);
        }
        return value;
    }
};

struct DeterminantTerm {
    int determinant = 0;
    double coefficient = 0.0;
};

} // namespace

StructureSpinAnalysis analyze_structure_spin(const StructureExpansion& structures,
                                             int spin_multiplicity) {
    const auto alpha_rows = structures.alpha(), beta_rows = structures.beta();
    if (spin_multiplicity <= 0 || structures.n_structures <= 0 || alpha_rows.empty() ||
        alpha_rows.size() != beta_rows.size() ||
        alpha_rows.size() != structures.incidence().size()) {
        throw std::invalid_argument("structure spin dimensions are inconsistent");
    }

    const int n_alpha = alpha_rows.electron_count();
    const int n_beta = beta_rows.electron_count();
    if (n_alpha - n_beta != spin_multiplicity - 1) {
        throw std::invalid_argument(
            "determinant spin projection differs from the requested multiplicity");
    }

    std::vector<std::vector<DeterminantTerm>> structure_terms(structures.n_structures);
    for (int determinant = 0; determinant < static_cast<int>(alpha_rows.size()); ++determinant) {
        if (alpha_rows.electron_count(determinant) != n_alpha ||
            beta_rows.electron_count(determinant) != n_beta)
            throw std::invalid_argument("structure expansion mixes electron-number sectors");
        for (const StructureExpansionTerm& term : structures.incidence()[determinant]) {
            if (term.structure_index < 0 || term.structure_index >= structures.n_structures ||
                !std::isfinite(term.coefficient)) {
                throw std::invalid_argument("structure spin expansion term is invalid");
            }
            structure_terms[term.structure_index].push_back(
                DeterminantTerm{determinant, term.coefficient});
        }
    }

    StructureSpinAnalysis result;
    result.spin = 0.5 * static_cast<double>(spin_multiplicity - 1);
    result.spin_squared = result.spin * (result.spin + 1.0);

    for (int structure = 0; structure < structures.n_structures; ++structure) {
        if (structure_terms[structure].empty()) {
            throw std::invalid_argument("structure has no determinant expansion");
        }
        std::unordered_map<RaisedDeterminant, double, RaisedDeterminantHash> raised_coefficients;
        for (const DeterminantTerm& term : structure_terms[structure]) {
            const auto alpha = alpha_rows.words(term.determinant);
            const auto beta = beta_rows.words(term.determinant);
            const auto width = std::max(alpha.size(), beta.size());
            int alpha_before = 0, beta_before = 0;
            for (std::size_t word = 0; word < width; ++word) {
                const auto a = word < alpha.size() ? alpha[word] : 0;
                const auto b = word < beta.size() ? beta[word] : 0;
                auto candidates = b & ~a;
                while (candidates) {
                    const int orbital = std::countr_zero(candidates);
                    const auto bit = std::uint64_t{1} << orbital, below = bit - 1;
                    const int parity = n_alpha + alpha_before + beta_before +
                                       std::popcount(a & below) + std::popcount(b & below);
                    RaisedDeterminant raised;
                    raised.alpha.assign(width, 0);
                    raised.beta.assign(width, 0);
                    std::copy(alpha.begin(), alpha.end(), raised.alpha.begin());
                    std::copy(beta.begin(), beta.end(), raised.beta.begin());
                    raised.alpha[word] |= bit;
                    raised.beta[word] &= ~bit;
                    raised_coefficients[std::move(raised)] +=
                        (parity % 2 ? -1.0 : 1.0) * term.coefficient;
                    candidates &= candidates - 1;
                }
                alpha_before += std::popcount(a);
                beta_before += std::popcount(b);
            }
        }

        double structure_residual = 0.0;
        for (const auto& [determinant, coefficient] : raised_coefficients) {
            static_cast<void>(determinant);
            structure_residual = std::max(structure_residual, std::abs(coefficient));
        }
        if (structure_residual > result.maximum_raising_residual) {
            result.maximum_raising_residual = structure_residual;
            result.worst_structure_index = structure;
        }
    }
    return result;
}

} // namespace xmvb::vb
