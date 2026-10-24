#include "cli/casscf.hpp"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

#include "casscf/driver/casscf.hpp"
#include "cli/options.hpp"
#include "input/deck/model.hpp"
#include "input/loading/casscf.hpp"
#include "output/molden/writer.hpp"
#include "output/orbitals/writer.hpp"
#include "output/trace/binary.hpp"

namespace xmvb::cli {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

vb::OrbitalPreparationInput make_output_orbitals(
    const Eigen::MatrixXd& frame, int ncore, int nactive, int n_electrons, int twice_spin) {
    const int n_basis = static_cast<int>(frame.rows());
    const int n_output = ncore + nactive;
    if (n_basis <= 0 || n_output <= 0 || frame.cols() < n_output) {
        throw std::invalid_argument("CASSCF result has an invalid orbital frame");
    }
    vb::OrbitalPreparationInput output;
    output.n_basis_functions = static_cast<std::size_t>(n_basis);
    output.n_orbitals = static_cast<std::size_t>(n_output);
    output.n_active_orbitals = static_cast<std::size_t>(nactive);
    output.n_total_electrons = static_cast<std::size_t>(n_electrons);
    output.n_active_electrons = static_cast<std::size_t>(n_electrons - 2 * ncore);
    output.spin_multiplicity = twice_spin + 1;
    output.orbital_basis_counts.assign(output.n_orbitals, n_basis);
    output.orbital_value_table.resize(output.n_orbitals * output.n_basis_functions);
    output.orbital_basis_index_table.resize(output.n_orbitals * output.n_basis_functions);
    for (int orbital = 0; orbital < n_output; ++orbital) {
        for (int mu = 0; mu < n_basis; ++mu) {
            const std::size_t offset = static_cast<std::size_t>(orbital) * n_basis + mu;
            output.orbital_value_table[offset] = frame(mu, orbital);
            output.orbital_basis_index_table[offset] = mu + 1;
        }
    }
    return output;
}

void reject_unsupported_options(const Options& command_line, const vb::InputDeck& deck) {
    if (deck.metadata.request_ri_two_electron_mode ||
        command_line.load.standard_two_electron_mode ==
            vb::StandardTwoElectronMode::ResolutionOfIdentity) {
        throw std::invalid_argument("CASSCF requires exact AO integrals; RI is unsupported");
    }
    if (deck.metadata.state_average_count != 1) {
        throw std::invalid_argument("CASSCF currently supports one state, not state averaging");
    }
    if (command_line.optimizer_backend_explicit &&
        command_line.optimizer.backend != vb::VbScfOptimizerBackend::Ciah) {
        throw std::invalid_argument("CASSCF currently supports only the CIAH optimizer");
    }
    if (deck.metadata.structure_eigensolver != vb::StructureEigensolver::Davidson ||
        command_line.optimizer.structure_eigensolver != vb::StructureEigensolver::Davidson ||
        (command_line.optimizer.orbital_preconditioner != vb::OrbitalPreconditioner::Automatic &&
         command_line.optimizer.orbital_preconditioner !=
             vb::OrbitalPreconditioner::HessianDiagonal) ||
        command_line.load.raw_structure_selection != vb::RawStructureSelectionMode::Full ||
        command_line.load.skip_orbital_guess || !command_line.trace_directory.empty() ||
        !command_line.tnhvp_trace_path.empty() || !command_line.ciah_trace_path.empty()) {
        throw std::invalid_argument(
            "CASSCF supports Davidson CI and the Hessian-diagonal orbital preconditioner; VB "
            "structure, guess-skip, and trace options are unsupported");
    }
}

} // namespace

int run_casscf(const Options& command_line, const vb::InputDeck& deck) {
    reject_unsupported_options(command_line, deck);
    const auto command_start = Clock::now();
    auto input = xmvb::input::load_casscf_input(deck, command_line.input_path);
    input.ao.exact_exchange_memory_budget_bytes = command_line.exact_exchange_memory_budget_bytes;
    const double load_seconds = elapsed_seconds(command_start);

    casscf::CasscfOptions options;
    options.energy_tolerance = command_line.optimizer.energy_tolerance;
    options.gradient_tolerance = command_line.optimizer.gradient_tolerance;
    options.max_iterations = command_line.max_iterations_explicit
                                 ? command_line.optimizer.max_iterations
                                 : input.max_iterations;
    if (command_line.optimizer.verbose) {
        options.observe_iteration = [](const casscf::CasscfIteration& iteration) {
            std::cout << "  CASSCF iteration " << iteration.iteration
                      << ": E=" << std::setprecision(15) << iteration.energy
                      << " dE=" << iteration.energy_change << " |g|=" << iteration.gradient_norm
                      << " CIAH Ritz solves=" << iteration.micro_iterations
                      << " HVP=" << iteration.hvp_actions << " CI actions=" << iteration.ci_actions
                      << " wall=" << iteration.wall_seconds << " s\n";
        };
    }

    std::cout << "CASSCF (exact, single-state, CIAH)\n"
              << "Input: " << command_line.input_path << '\n'
              << "Basis: " << input.basis << '\n'
              << "CAS: " << input.n_electrons - 2 * input.ncore << " electrons in " << input.nactive
              << " orbitals; core=" << input.ncore << ", spin multiplicity=" << input.twice_spin + 1
              << '\n'
              << "Maximum iterations: " << options.max_iterations << '\n'
              << "Exact-exchange memory budget: " << input.ao.exact_exchange_memory_budget_bytes
              << " bytes\n";

    const auto solve_start = Clock::now();
    const auto result = casscf::solve_casscf(input.ao,
                                             input.orbitals,
                                             input.ncore,
                                             input.nactive,
                                             input.n_electrons - 2 * input.ncore,
                                             input.twice_spin,
                                             input.nuclear_energy,
                                             options);
    const double solve_seconds = elapsed_seconds(solve_start);

    const auto output_orbitals = make_output_orbitals(
        result.orbitals, input.ncore, input.nactive, input.n_electrons, input.twice_spin);
    if (!command_line.final_orbitals_path.empty()) {
        output::write_binary_container(std::filesystem::path(command_line.final_orbitals_path),
                                       output_orbitals.orbital_value_table);
    }
    if (result.converged) {
        const auto orbital_path = output::write_orbital_file(
            std::filesystem::path(command_line.input_path), output_orbitals);
        std::cout << "Converged orbitals: " << orbital_path.string() << '\n';
    }
    if (input.request_molden_output) {
        const auto molden_path = vb::write_molden_file(
            std::filesystem::path(command_line.input_path), input.libcint_input, output_orbitals);
        std::cout << "Molden orbitals: " << molden_path.string() << '\n';
    }

    const int keyframes = std::accumulate(result.iterations.begin(),
                                          result.iterations.end(),
                                          0,
                                          [](int total, const casscf::CasscfIteration& iteration) {
                                              return total + iteration.keyframes;
                                          });
    std::cout << std::setprecision(15) << "Initial energy: " << result.initial_energy << " Eh\n"
              << "Initial gradient norm: " << result.initial_gradient_norm << '\n'
              << "Final energy: " << result.energy << " Eh\n"
              << "Final gradient norm: " << result.gradient_norm << '\n'
              << "Accepted iterations: " << result.iterations.size() << '\n'
              << "Accepted-step CIAH keyframes: " << keyframes << '\n'
              << "Hessian-vector actions: " << result.hvp_actions << '\n'
              << "CI actions: " << result.ci_actions << '\n'
              << "Load time: " << load_seconds << " s\n"
              << "Solve time: " << solve_seconds << " s\n"
              << "Total time: " << elapsed_seconds(command_start) << " s\n"
              << "Converged: " << (result.converged ? "yes" : "no") << '\n'
              << "Termination reason: " << result.termination_reason << '\n';
    return result.converged ? 0 : 2;
}

} // namespace xmvb::cli
