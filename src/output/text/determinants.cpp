#include "output/text/determinants.hpp"

#include <algorithm>
#include <format>
#include <iterator>
#include <limits>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace xmvb::output {
namespace {

std::string orbital_range(const std::vector<int>& orbitals) {
    if (orbitals.empty()) {
        return {};
    }
    std::ostringstream text;
    std::size_t begin = 0;
    while (begin < orbitals.size()) {
        std::size_t end = begin;
        while (end + 1 < orbitals.size() && orbitals[end + 1] == orbitals[end] + 1) {
            ++end;
        }
        if (begin != 0) {
            text << ' ';
        }
        if (end > begin) {
            text << orbitals[begin] << '-' << orbitals[end];
        } else {
            text << orbitals[begin];
        }
        begin = end + 1;
    }
    return text.str();
}

std::string determinant_spin_string(const std::vector<int>& active_orbitals, int n_inactive) {
    std::ostringstream text;
    if (n_inactive == 1) {
        text << '1';
    } else if (n_inactive > 1) {
        text << "1-" << n_inactive;
    }
    std::vector<int> absolute_active_orbitals;
    absolute_active_orbitals.reserve(active_orbitals.size());
    for (const int orbital : active_orbitals) {
        absolute_active_orbitals.push_back(n_inactive + orbital + 1);
    }
    if (!absolute_active_orbitals.empty()) {
        if (n_inactive > 0) {
            text << ' ';
        }
        text << orbital_range(absolute_active_orbitals);
    }
    return text.str();
}

} // namespace

DeterminantReport::DeterminantReport(const vb::StructureExpansion& expansion, int n_inactive)
    : expansion_(expansion), n_inactive_(n_inactive),
      shared_spin_strings_(expansion.alpha_strings.size() < expansion.pair_alpha.size() &&
                           expansion.alpha_strings == expansion.beta_strings) {
    const auto alpha = expansion_.alpha(), beta = expansion_.beta();
    const auto incidence = expansion_.incidence();
    if (alpha.size() != beta.size() || incidence.size() != alpha.size())
        throw std::invalid_argument("final determinant topology has inconsistent dimensions");
    for (const auto& term : expansion_.terms)
        if (term.structure_index < 0 ||
            static_cast<std::size_t>(term.structure_index) >= expansion_.n_structures)
            throw std::out_of_range("determinant expansion refers to an invalid structure index");

    const auto& strings = expansion_.alpha_strings;
    display_order_.resize(alpha.size());
    std::iota(display_order_.begin(), display_order_.end(), std::size_t{0});
    std::stable_sort(display_order_.begin(), display_order_.end(), [&](auto left, auto right) {
        const auto a = incidence[left], b = incidence[right];
        const auto first_a = a.empty() ? std::numeric_limits<std::size_t>::max()
                                       : static_cast<std::size_t>(a.front().structure_index);
        const auto first_b = b.empty() ? std::numeric_limits<std::size_t>::max()
                                       : static_cast<std::size_t>(b.front().structure_index);
        if (first_a != first_b)
            return first_a < first_b;
        return alpha.less(right, left);
    });
    // Retain text only where the pair list reuses the unique-spin table.
    // Unreferenced labels are never decoded; one-off rows use one scratch string.
    if (strings.size() < alpha.size())
        alpha_text_.resize(strings.size());
    if (!shared_spin_strings_ && expansion_.beta_strings.size() < beta.size())
        beta_text_.resize(expansion_.beta_strings.size());
}

std::string_view DeterminantReport::spin_text(const mcscf::StringTable& strings,
                                              std::size_t id,
                                              std::vector<std::optional<std::string>>& cache) {
    if (cache.empty()) {
        scratch_ = determinant_spin_string(strings[id], n_inactive_);
        return scratch_;
    }
    auto& text = cache[id];
    if (!text)
        text = determinant_spin_string(strings[id], n_inactive_);
    return *text;
}

void DeterminantReport::write(std::ostream& output,
                              const Eigen::Ref<const Eigen::VectorXd>& coefficients) {
    if (static_cast<std::size_t>(coefficients.size()) != expansion_.n_structures)
        throw std::invalid_argument("determinant report coefficient count is inconsistent");
    output << "\n\n          ******  UNNORMALIZED DETERMINANT COEFFICIENTS  ******\n\n"
              "                                     INA   A\n"
              "                                     INA   B\n";
    const auto alpha = expansion_.alpha(), beta = expansion_.beta();
    const auto incidence = expansion_.incidence();
    auto& beta_text = shared_spin_strings_ ? alpha_text_ : beta_text_;
    const std::string continuation(37, ' ');
    std::string row;
    for (std::size_t display = 0; display < display_order_.size(); ++display) {
        const auto pair = display_order_[display];
        double coefficient = 0.0;
        for (const auto& term : incidence[pair])
            coefficient += term.coefficient * coefficients[term.structure_index];
        row.clear();
        std::format_to(
            std::back_inserter(row), "{:8}{:17.8f}  ******    ", display + 1, coefficient);
        row += spin_text(expansion_.alpha_strings, alpha.string_id(pair), alpha_text_);
        row += '\n';
        row += continuation;
        row += spin_text(expansion_.beta_strings, beta.string_id(pair), beta_text);
        row += '\n';
        output.write(row.data(), static_cast<std::streamsize>(row.size()));
    }
}

} // namespace xmvb::output
