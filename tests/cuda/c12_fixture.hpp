#pragma once

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>

#include "input/loading/loader.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/integrals/active/space.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/assembly/hamiltonian_overlap.hpp"

namespace xmvb::cuda::test {

using Clock = std::chrono::steady_clock;

inline double elapsed(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

// The input deck's $GUS block supplies the orbitals. No SCF or CPU CI solve
// occurs while constructing this frozen-orbital structure problem.
struct C12Fixture {
    vb::VbScfInputLoadResult loaded;
    vb::TimedPreparedActiveSpaceContext prepared;
    vb::SameSpinPairCacheContext topology;
    std::unique_ptr<vb::StructureAction> action;
    double load_seconds = 0.0;
    double active_space_seconds = 0.0;
    double topology_seconds = 0.0;
    double cpu_action_init_seconds = 0.0;

    explicit C12Fixture(const std::string& path) {
        const auto start = Clock::now();
        loaded = vb::load_vbscf_input_with_timings(path);
        const auto after_load = Clock::now();
        prepared = vb::prepare_timed_active_space_context(loaded.input);
        const auto after_active = Clock::now();
        const auto& input = loaded.input;
        const int n = input.orbital_preparation_input.n_active_orbitals;
        if (n != 12 && n != 14)
            throw std::invalid_argument("carbon fixture requires 12 or 14 active orbitals");
        topology = vb::build_orthogonal_spin_topology(
            input.structure_data.alpha(), input.structure_data.beta(), n);
        const auto after_topology = Clock::now();
        const auto& active = prepared.prepared_active_space;
        const auto diagonal = vb::FullDeterminantStructureHamiltonianOverlapBuilder{}
                                  .build_davidson_jacobi_preconditioner(
                                      input.structure_data.alpha(),
                                      input.structure_data.beta(),
                                      input.structure_data.incidence(),
                                      active.orbital_result.active_orbital_overlap_matrix,
                                      active.active_space_one_electron_result.h1e_act,
                                      n,
                                      active.active_space_two_electron_result,
                                      input.structure_data.n_structures,
                                      topology);
        action = std::make_unique<vb::StructureAction>(
            input.structure_data.incidence(),
            input.structure_data.n_structures,
            topology,
            active.orbital_result.active_orbital_overlap_matrix,
            active.active_space_one_electron_result.h1e_act,
            active.active_space_two_electron_result,
            n,
            &diagonal,
            vb::StructureActionMode::Davidson);
        const auto after_action = Clock::now();
        load_seconds = elapsed(start, after_load);
        active_space_seconds = elapsed(after_load, after_active);
        topology_seconds = elapsed(after_active, after_topology);
        cpu_action_init_seconds = elapsed(after_topology, after_action);
    }

    double energy_offset() const {
        return prepared.prepared_active_space.one_electron_reference_energy +
               loaded.nuclear_repulsion_energy;
    }
};

} // namespace xmvb::cuda::test
