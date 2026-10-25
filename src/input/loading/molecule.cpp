#include "input/loading/molecule.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

extern "C" {
#include "cint.h"
}

#include "input/deck/model.hpp"
#include "input/deck/primary_basis.hpp"
#include "libcint/direct_shell.hpp"
#include "libcint/materialized_input.hpp"
#include "libcint/materialized_provider.hpp"

namespace xmvb::input {

MoleculeInput load_molecule(const vb::InputDeck& deck) {
    auto basis = vb::build_input_deck_basis(deck, deck.metadata.basis_name);
    MoleculeInput result;
    result.basis = std::move(basis.basis_display_name);
    result.libcint_input = std::move(basis.libcint_input);
    const auto& raw = result.libcint_input;
    result.n_electrons = -deck.metadata.total_charge;
    for (int atom = 0; atom < raw.n_atoms; ++atom) {
        const int z = raw.atm[atom * ATM_SLOTS + CHARGE_OF];
        const int center = raw.atm[atom * ATM_SLOTS + PTR_COORD];
        result.n_electrons += z;
        for (int other = 0; other < atom; ++other) {
            const int offset = raw.atm[other * ATM_SLOTS + PTR_COORD];
            double radius2 = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                const double delta = raw.env[center + axis] - raw.env[offset + axis];
                radius2 += delta * delta;
            }
            if (!(radius2 > 0.0))
                throw std::invalid_argument("molecular geometry contains overlapping nuclei");
            result.nuclear_energy += static_cast<double>(z) *
                                     raw.atm[other * ATM_SLOTS + CHARGE_OF] / std::sqrt(radius2);
        }
    }
    if (result.n_electrons < 1)
        throw std::invalid_argument("molecular electron count must be positive");

    libcint::LibcintDirectShellEvaluator evaluator(raw);
    const int n_bf = evaluator.n_basis_functions();
    result.overlap = Eigen::MatrixXd::Zero(n_bf, n_bf);
    for (int right = 0; right < raw.n_shells; ++right)
        for (int left = 0; left <= right; ++left) {
            const auto block = evaluator.evaluate_overlap_shell_pair(left, right);
            for (int j = 0; j < block.right_ao_count; ++j)
                for (int i = 0; i < block.left_ao_count; ++i) {
                    const int mu = block.left_ao_offset + i, nu = block.right_ao_offset + j;
                    result.overlap(mu, nu) = result.overlap(nu, mu) =
                        block.values[i + j * block.left_ao_count];
                }
        }
    libcint::LibcintMaterializedIntegralProvider provider;
    result.ao = libcint::build_materialized_ao_integral_input(provider.build(raw));
    return result;
}

} // namespace xmvb::input
