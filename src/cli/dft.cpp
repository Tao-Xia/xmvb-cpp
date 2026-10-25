#include "cli/dft.hpp"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include "cli/options.hpp"
#include "dft/driver/ks.hpp"
#include "dft/xc/functional.hpp"
#include "input/deck/model.hpp"
#include "input/loading/molecule.hpp"
#include "output/orbitals/writer.hpp"

namespace xmvb::cli {

int run_dft(const Options& command_line, const vb::InputDeck& deck) {
    const auto& metadata = deck.metadata;
    if (metadata.declared_spin_multiplicity < 1 || metadata.state_average_count != 1)
        throw std::invalid_argument(
            "DFT requires a positive NMUL and does not support state averaging");
    if (metadata.request_ri_two_electron_mode ||
        command_line.load.standard_two_electron_mode ==
            vb::StandardTwoElectronMode::ResolutionOfIdentity)
        throw std::invalid_argument(
            "DFT currently requires exact AO integrals; RI will be added separately");
    if (metadata.guess_type != vb::kGuessTypeAuto || deck.guess_block.present)
        throw std::invalid_argument(
            "DFT currently supports GUESS=AUTO (core Hamiltonian), not supplied guesses");
    if (command_line.optimizer_backend_explicit || command_line.structure_eigensolver_explicit ||
        metadata.scf_optimizer != vb::InputScfOptimizer::Unspecified ||
        !command_line.trace_directory.empty() || !command_line.ciah_trace_path.empty() ||
        !command_line.tnhvp_trace_path.empty() || !command_line.final_orbitals_path.empty() ||
        metadata.request_molden_output)
        throw std::invalid_argument("DFT uses the common SCF/DIIS driver; VB/CASSCF optimizer, "
                                    "trace and Molden options are unsupported");

    const auto start = std::chrono::steady_clock::now();
    auto molecule = input::load_molecule(deck);
    const int twice_ms = metadata.declared_spin_multiplicity - 1;
    if (twice_ms > molecule.n_electrons || (molecule.n_electrons + twice_ms) % 2)
        throw std::invalid_argument("electron count and NMUL are inconsistent");
    const int n_alpha = (molecule.n_electrons + twice_ms) / 2;
    const int n_beta = molecule.n_electrons - n_alpha;
    dft::KsOptions options;
    options.unrestricted = metadata.ks_spin == vb::KohnShamSpin::Unrestricted ||
                           (metadata.ks_spin == vb::KohnShamSpin::Auto && twice_ms != 0);
    if (!options.unrestricted && twice_ms != 0)
        throw std::invalid_argument("RKS requires NMUL=1; use KS=UKS for open shells");
    options.functional = dft::Functional::canonical_name(metadata.xc_functional);
    options.grid.radial_points = metadata.grid_radial_points;
    options.grid.angular_points = metadata.grid_angular_points;
    options.scf.density_tolerance = metadata.scf_tolerance;
    if (metadata.requested_scf_max_iterations > 0)
        options.scf.max_iterations = metadata.requested_scf_max_iterations;
    if (command_line.max_iterations_explicit)
        options.scf.max_iterations = command_line.optimizer.max_iterations;

    std::cout << (options.unrestricted ? "Unrestricted" : "Restricted")
              << " Kohn-Sham DFT (exact J/K, SCF/DIIS)\n"
              << "Input: " << command_line.input_path << '\n'
              << "Basis: " << molecule.basis << '\n'
              << "AO functions: " << molecule.ao.n_basis_functions << '\n'
              << "Electrons: " << molecule.n_electrons << '\n'
              << "Alpha electrons: " << n_alpha << '\n'
              << "Beta electrons: " << n_beta << '\n'
              << "XC: " << options.functional << '\n'
              << "Libxc: " << dft::Functional::library_version() << '\n'
              << "Grid: " << options.grid.radial_points << " radial, "
              << options.grid.angular_points << " angular per atom\n"
              << "SCF density and residual tolerance: " << options.scf.density_tolerance << '\n';
    const auto result = dft::solve_ks(
        n_alpha, n_beta, molecule.libcint_input, molecule.overlap, molecule.ao, options);
    const bool converged = std::visit(
        [&](const auto& state) {
            using State = std::decay_t<decltype(state)>;
            if constexpr (std::is_same_v<State, scf::UnrestrictedResult>) {
                if (state.converged) {
                    for (int spin = 0; spin < 2; ++spin) {
                        std::filesystem::path input(command_line.input_path);
                        input.replace_extension(spin == 0 ? ".alpha.xmi" : ".beta.xmi");
                        const auto path = output::write_orbital_file(input, state.orbitals[spin]);
                        std::cout << "Converged " << (spin == 0 ? "alpha" : "beta")
                                  << " orbitals: " << path.string() << '\n';
                    }
                }
                const double ms = 0.5 * twice_ms;
                const double spin_square =
                    ms * (ms + 1) + n_beta -
                    (state.density[0] * molecule.overlap * state.density[1] * molecule.overlap)
                        .trace();
                std::cout << std::setprecision(15) << "Spin square: " << spin_square << '\n';
            } else {
                if (state.converged) {
                    const auto path =
                        output::write_orbital_file(command_line.input_path, state.orbitals);
                    std::cout << "Converged orbitals: " << path.string() << '\n';
                }
            }
            std::cout << std::setprecision(15) << "Initial energy: "
                      << result.initial_electronic_energy + molecule.nuclear_energy << " Eh\n"
                      << "Final energy: " << state.electronic_energy + molecule.nuclear_energy
                      << " Eh\n"
                      << "SCF iterations: " << state.iterations << '\n'
                      << "Orbital correction steps: " << state.orbital_steps << '\n'
                      << "Fock response actions: " << state.response_actions << '\n'
                      << "Final AO residual: " << state.residual << '\n'
                      << "Converged: " << (state.converged ? "yes" : "no") << '\n';
            return state.converged;
        },
        result.state);
    std::cout << "Exact exchange fraction: " << result.exact_exchange << '\n'
              << "Semilocal XC energy: " << result.xc_energy << " Eh\n"
              << "Grid points: " << result.grid_points << '\n'
              << "Integrated electrons: "
              << result.integrated_electrons[0] + result.integrated_electrons[1] << '\n';
    if (options.unrestricted)
        std::cout << "Integrated alpha electrons: " << result.integrated_electrons[0] << '\n'
                  << "Integrated beta electrons: " << result.integrated_electrons[1] << '\n';
    std::cout << "Total time: "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
              << " s\n";
    return converged ? 0 : 2;
}

} // namespace xmvb::cli
