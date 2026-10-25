#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

#include "input/loading/generated_structures.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

xmvb::vb::RawStructureData
generate(int n_orbitals, int n_electrons, int multiplicity, int n_inactive) {
    xmvb::vb::InputDeckMetadata metadata;
    metadata.structure_class_keyword = "FULL";
    metadata.declared_active_orbitals = n_orbitals;
    metadata.declared_active_electrons = n_electrons;
    metadata.declared_spin_multiplicity = multiplicity;
    return xmvb::vb::build_generated_raw_structures_from_structure_class(
        metadata, n_electrons + 2 * n_inactive);
}

/** @brief Independently enumerate occupied subsets, including all holes. */
void check_all_alpha(int n_orbitals, int n_electrons, int n_inactive) {
    std::set<std::vector<int>> expected;
    for (unsigned mask = 0; mask < (1u << n_orbitals); ++mask) {
        std::vector<int> occupied;
        for (int orbital = 0; orbital < n_orbitals; ++orbital) {
            if (mask & (1u << orbital)) {
                occupied.push_back(n_inactive + orbital + 1);
            }
        }
        if (static_cast<int>(occupied.size()) == n_electrons) {
            expected.insert(occupied);
        }
    }

    const auto raw = generate(n_orbitals, n_electrons, n_electrons + 1, n_inactive);
    require(static_cast<std::size_t>(raw.n_structures) == expected.size(),
            "FULL all-alpha structure count differs from occupied subsets");
    std::set<std::vector<int>> actual;
    for (int structure = 0; structure < raw.n_structures; ++structure) {
        const auto begin = raw.raw_structure_orbitals.begin() + structure * raw.n_total_electrons;
        for (int orbital = 0; orbital < n_inactive; ++orbital) {
            require(begin[2 * orbital] == orbital + 1 && begin[2 * orbital + 1] == orbital + 1,
                    "inactive pairs must remain unchanged");
        }
        std::vector<int> occupied(begin + 2 * n_inactive, begin + raw.n_total_electrons);
        require(std::is_sorted(occupied.begin(), occupied.end()),
                "all-alpha orbital labels must be ordered");
        require(actual.insert(occupied).second,
                "FULL generator emitted a duplicate occupied subset");
    }
    require(actual == expected, "FULL generator omitted or changed an occupied subset");
    require(*std::max_element(raw.raw_structure_orbitals.begin(),
                              raw.raw_structure_orbitals.end()) == n_inactive + n_orbitals,
            "generated labels must retain the declared active orbital extent");
}

void check_closed_shell(int n_orbitals, int n_inactive) {
    const auto raw = generate(n_orbitals, 2 * n_orbitals, 1, n_inactive);
    require(raw.n_structures == 1, "a fully occupied space must remain single-reference");
    for (int orbital = 0; orbital < n_inactive + n_orbitals; ++orbital) {
        require(raw.raw_structure_orbitals[2 * orbital] == orbital + 1 &&
                    raw.raw_structure_orbitals[2 * orbital + 1] == orbital + 1,
                "closed-shell orbital pairs changed");
    }
}

} // namespace

int main() {
    for (int n_inactive : {0, 2}) {
        for (int n_orbitals = 1; n_orbitals <= 7; ++n_orbitals) {
            for (int n_electrons = 1; n_electrons <= n_orbitals; ++n_electrons) {
                check_all_alpha(n_orbitals, n_electrons, n_inactive);
            }
            check_closed_shell(n_orbitals, n_inactive);
        }
    }
    require(generate(3, 2, 1, 0).n_structures == 6,
            "the two-electron singlet FULL space must still contain six structures");
    std::cout << "generated FULL structure checks passed\n";
}
