#include "input/loading/casscf.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "casscf/orbitals/frame.hpp"
#include "guess/block.hpp"
#include "guess/initial.hpp"
#include "input/deck/model.hpp"
#include "input/loading/molecule.hpp"
#include "vbscf/orbitals/preparation/input.hpp"

namespace xmvb::input {
namespace {

Eigen::MatrixXd
build_seed(const vb::InputDeck& deck, const std::string& input_path, const CasscfInput& input) {
    const int n = input.ao.n_basis_functions;
    const int occupied = input.ncore + input.nactive;
    vb::OrbitalPreparationInput bridge;
    bridge.n_basis_functions = n;
    bridge.n_orbitals = occupied;
    bridge.n_active_orbitals = input.nactive;
    bridge.n_total_electrons = input.n_electrons;
    bridge.n_active_electrons = input.n_electrons - 2 * input.ncore;
    bridge.spin_multiplicity = input.twice_spin + 1;
    bridge.orbital_type = deck.metadata.orbital_type;
    bridge.ao_overlap_matrix = input.overlap;
    bridge.ao_normalization = vb::build_ao_normalization(input.overlap);
    bridge.orbital_basis_counts.assign(occupied, n);
    bridge.original_orbital_basis_counts.assign(occupied, n);
    bridge.orbital_basis_index_table.resize(static_cast<std::size_t>(occupied) * n);
    for (int orbital = 0; orbital < occupied; ++orbital)
        for (int mu = 0; mu < n; ++mu)
            bridge.orbital_basis_index_table[static_cast<std::size_t>(orbital) * n + mu] = mu + 1;
    vb::build_initial_orbital_guess(input_path,
                                    deck.metadata.guess_type,
                                    input.libcint_input,
                                    input.ao,
                                    nullptr,
                                    deck.guess_block.present ? &deck.guess_block.raw_lines
                                                             : nullptr,
                                    &bridge);
    Eigen::MatrixXd seed(n, occupied);
    for (int orbital = 0; orbital < occupied; ++orbital)
        for (int mu = 0; mu < n; ++mu)
            seed(mu, orbital) =
                bridge.orbital_value_table[static_cast<std::size_t>(orbital) * n + mu];
    return seed;
}

} // namespace

CasscfInput load_casscf_input(const vb::InputDeck& deck, const std::string& input_path) {
    const auto& metadata = deck.metadata;
    if (metadata.request_ri_two_electron_mode)
        throw std::invalid_argument("CASSCF currently requires exact AO integrals; RI requested");
    if (metadata.state_average_count != 1)
        throw std::invalid_argument("CASSCF currently supports one state only");
    if (metadata.declared_active_orbitals < 1 || metadata.declared_active_electrons < 1 ||
        metadata.declared_spin_multiplicity < 1)
        throw std::invalid_argument("CASSCF requires NAO, NAE, and NMUL in $CTRL");
    auto molecule = load_molecule(deck);
    CasscfInput result;
    result.libcint_input = std::move(molecule.libcint_input);
    result.basis = std::move(molecule.basis);
    result.overlap = std::move(molecule.overlap);
    result.ao = std::move(molecule.ao);
    const int total_electrons = molecule.n_electrons;
    result.nactive = metadata.declared_active_orbitals;
    result.n_electrons = total_electrons;
    result.twice_spin = metadata.declared_spin_multiplicity - 1;
    result.nuclear_energy = molecule.nuclear_energy;
    result.max_iterations =
        metadata.requested_scf_max_iterations > 0 ? metadata.requested_scf_max_iterations : 2000;
    result.request_molden_output = metadata.request_molden_output;
    const int inactive_electrons = total_electrons - metadata.declared_active_electrons;
    if (inactive_electrons < 0 || inactive_electrons % 2 != 0)
        throw std::invalid_argument("CASSCF NAE leaves an invalid closed-shell inactive count");
    result.ncore = inactive_electrons / 2;
    const int active_electrons = metadata.declared_active_electrons;
    if (result.ncore + result.nactive > result.overlap.rows() ||
        active_electrons > 2 * result.nactive || result.twice_spin > active_electrons ||
        result.twice_spin > 2 * result.nactive - active_electrons ||
        (active_electrons - result.twice_spin) % 2 != 0)
        throw std::invalid_argument(
            "CASSCF active electron, spin, or orbital partition is invalid");
    const Eigen::MatrixXd seed = build_seed(deck, input_path, result);
    result.orbitals = casscf::complete_metric_orthonormal_frame(
        seed, result.overlap, result.ncore, result.nactive);
    return result;
}

} // namespace xmvb::input
