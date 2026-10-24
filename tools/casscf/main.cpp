#include <exception>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "casscf/driver/casscf.hpp"
#include "fixture.hpp"

int main(int argc, char** argv) {
    const bool initial_only = argc == 3 && std::string(argv[1]) == "--initial-only";
    if (argc != 3) {
        std::cerr << "usage: casscf_fixture FIXTURE.bin trajectory.csv\n"
                  << "       casscf_fixture --initial-only FIXTURE.bin\n";
        return 64;
    }
    try {
        const auto input = xmvb::casscf::fixture::read(argv[initial_only ? 2 : 1]);
        xmvb::casscf::CasscfOptions options;
        options.energy_tolerance = 1e-7;
        options.gradient_tolerance = 1e-3;
        options.max_iterations = 100;
        if (initial_only) {
            const auto cas = xmvb::casscf::make_active_integrals(
                input.ao, input.initial_orbitals, input.ncore, input.nactive, input.nuclear_energy);
            const auto result = xmvb::casscf::solve_casci(
                cas, input.active_electrons, input.twice_spin, options.ci);
            std::cout << std::setprecision(17) << "initial_energy=" << result.energy() << '\n';
            return 0;
        }
        std::ofstream trajectory(argv[2]);
        if (!trajectory)
            throw std::runtime_error("cannot open trajectory output");
        trajectory << "iteration,energy,energy_change,gradient_norm,hvp_actions,"
                      "ci_actions,keyframes,micro_iterations,wall_seconds\n";
        trajectory << std::setprecision(17);
        options.observe_iteration = [&](const xmvb::casscf::CasscfIteration& step) {
            trajectory << step.iteration << ',' << step.energy << ',' << step.energy_change << ','
                       << step.gradient_norm << ',' << step.hvp_actions << ',' << step.ci_actions
                       << ',' << step.keyframes << ',' << step.micro_iterations << ','
                       << step.wall_seconds << '\n'
                       << std::flush;
            if (!trajectory)
                throw std::runtime_error("cannot write trajectory output");
            std::cout << "CASSCF_MACRO " << step.iteration << ' ' << std::setprecision(17)
                      << step.energy << ' ' << step.energy_change << ' ' << step.gradient_norm
                      << ' ' << step.hvp_actions << '\n'
                      << std::flush;
        };
        std::cout << "starting_casscf=1" << std::endl;
        const auto start = std::chrono::steady_clock::now();
        const auto result = xmvb::casscf::solve_casscf(input.ao,
                                                       input.initial_orbitals,
                                                       input.ncore,
                                                       input.nactive,
                                                       input.active_electrons,
                                                       input.twice_spin,
                                                       input.nuclear_energy,
                                                       options);
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << std::setprecision(17) << "converged=" << result.converged << '\n'
                  << "termination_reason=" << result.termination_reason << '\n'
                  << "initial_energy=" << result.initial_energy << '\n'
                  << "energy=" << result.energy << '\n'
                  << "gradient_norm=" << result.gradient_norm << '\n'
                  << "macros=" << result.iterations.size() << '\n'
                  << "hvp_actions=" << result.hvp_actions << '\n'
                  << "ci_actions=" << result.ci_actions << '\n';
        std::cout << "solver_wall_seconds=" << elapsed << '\n';
        return result.converged ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "casscf_fixture: " << error.what() << '\n';
        return 1;
    }
}
