#include <Eigen/Eigenvalues>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "core/memory.hpp"
#include "input/loading/loader.hpp"
#include "vbscf/integrals/active/space.hpp"
#include "vbscf/orbitals/gauge/support_preserving.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/assembly/hamiltonian_overlap.hpp"

/** @brief Small-molecule, same-orbital check against independent nonorthogonal cofactors. */
int main(int argc, char** argv) {
    try {
        if (argc != 2)
            throw std::invalid_argument("usage: test_action_parity INPUT.xmi");
        xmvb::memory::configure({});
        auto loaded = xmvb::vb::load_vbscf_input_with_timings(argv[1]);
        auto& input = loaded.input;
        auto& orbitals = input.orbital_preparation_input;
        if (!orbitals.maintain_inactive_gauge)
            xmvb::vb::apply_support_preserving_inactive_gauge(&orbitals);
        xmvb::vb::balance_active_gauge(&orbitals);
        const auto& structures = input.structure_data;
        const auto n = structures.n_structures;
        if (n == 0 || n > 512)
            throw std::invalid_argument("dense reference diagnostic supports 1--512 structures");
        const int n_active = orbitals.n_active_orbitals;
        const Eigen::MatrixXd raw_frame;
        const auto raw = xmvb::vb::prepare_timed_active_space_context(input, &raw_frame);
        const auto& point = raw.prepared_active_space;
        const auto reference = xmvb::vb::FullDeterminantStructureHamiltonianOverlapBuilder{}.build(
            structures.alpha(),
            structures.beta(),
            structures.incidence(),
            point.orbital_result.active_orbital_overlap_matrix,
            point.active_space_one_electron_result.h1e_act,
            n_active,
            point.active_space_two_electron_result,
            n);
        const Eigen::Map<const Eigen::MatrixXd> h(reference.hamiltonian_matrix.data(), n, n);
        const Eigen::Map<const Eigen::MatrixXd> m(reference.overlap_matrix.data(), n, n);
        const double core = point.one_electron_reference_energy + loaded.nuclear_repulsion_energy;
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> reference_ci(h, m);
        if (reference_ci.info() != Eigen::Success)
            throw std::runtime_error("nonorthogonal reference CI failed");
        std::cout << std::setprecision(17) << "structures=" << n << '\n'
                  << "reference_total_energy=" << core + reference_ci.eigenvalues()[0] << '\n';
        const auto topology = xmvb::vb::build_orthogonal_spin_topology(
            structures.alpha(), structures.beta(), n_active);
        const auto compare = [&](const char* name, const auto& prepared) {
            const xmvb::vb::StructureAction action(
                structures.incidence(),
                n,
                topology,
                prepared.orbital_result.active_orbital_overlap_matrix,
                prepared.active_space_one_electron_result.h1e_act,
                prepared.active_space_two_electron_result,
                n_active,
                nullptr,
                xmvb::vb::StructureActionMode::ActionOnly);
            const auto images = action.apply(Eigen::MatrixXd::Identity(n, n));
            const double dh = (images.hamiltonian - h).cwiseAbs().maxCoeff();
            const double dm = (images.overlap - m).cwiseAbs().maxCoeff();
            Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ci(images.hamiltonian,
                                                                         images.overlap);
            if (ci.info() != Eigen::Success)
                throw std::runtime_error("direct-CI projected solve failed");
            const double de = ci.eigenvalues()[0] - reference_ci.eigenvalues()[0];
            std::cout << name << "_max_H_error=" << dh << '\n'
                      << name << "_max_M_error=" << dm << '\n'
                      << name << "_total_energy=" << core + ci.eigenvalues()[0] << '\n'
                      << name << "_energy_error=" << de << '\n';
            if (!(dh < 2e-9 && dm < 2e-11 && std::abs(de) < 2e-9))
                throw std::runtime_error(std::string(name) +
                                         " disagrees with nonorthogonal reference");
        };
        compare("raw_integrals_direct_ci", point);
        const auto framed = xmvb::vb::prepare_timed_active_space_context(input);
        compare("fused_orthogonal_integrals_direct_ci", framed.prepared_active_space);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
