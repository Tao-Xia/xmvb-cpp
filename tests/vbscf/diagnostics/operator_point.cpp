#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "core/memory.hpp"
#include "input/loading/loader.hpp"
#include "vbscf/integrals/active/space.hpp"
#include "vbscf/structures/assembly/operator_first.hpp"
#include "vbscf/structures/orthogonal_ci/integrals.hpp"
#include "vbscf/orbitals/gauge/support_preserving.hpp"

/** @brief Bounded operator-only probe: no initial Davidson, gradient or SCF loop. */
int main(int argc, char** argv) {
    try {
        if (argc != 3)
            throw std::invalid_argument("usage: test_operator_point INPUT.xmi WORKSPACE_MiB");
        const auto start = std::chrono::steady_clock::now();
        const std::size_t budget = std::stoull(argv[2]) * 1024 * 1024;
        xmvb::memory::configure({});
        auto loaded = xmvb::vb::load_vbscf_input_with_timings(argv[1]);
        auto& input = loaded.input;
        if (!input.orbital_preparation_input.maintain_inactive_gauge)
            xmvb::vb::apply_support_preserving_inactive_gauge(&input.orbital_preparation_input);
        xmvb::vb::balance_active_gauge(&input.orbital_preparation_input);
        const auto timed = xmvb::vb::prepare_timed_active_space_context(input);
        const auto& point = timed.prepared_active_space;
        const auto topology = xmvb::vb::build_orthogonal_spin_topology(
            input.structure_data.alpha(),
            input.structure_data.beta(),
            input.orbital_preparation_input.n_active_orbitals);
        const xmvb::vb::StructureAction action(input.structure_data.incidence(),
                                               input.structure_data.n_structures,
                                               topology,
                                               point.orbital_result.active_orbital_overlap_matrix,
                                               point.active_space_one_electron_result.h1e_act,
                                               point.active_space_two_electron_result,
                                               input.orbital_preparation_input.n_active_orbitals,
                                               nullptr,
                                               xmvb::vb::StructureActionMode::ActionOnly);
        const auto integrals = xmvb::vb::orthogonalize_active_integrals(
            point.orbital_result.active_orbital_overlap_matrix,
            point.active_space_one_electron_result.h1e_act,
            point.active_space_two_electron_result,
            input.orbital_preparation_input.n_active_orbitals);
        const xmvb::vb::OperatorFirstAction executor(action.orthogonal_view(),
                                                     action.n_structures(),
                                                     integrals,
                                                     integrals.orbital_transform);
        const auto prepared = std::chrono::steady_clock::now();
        Eigen::MatrixXd vectors(action.n_structures(), 2);
        for (int i = 0; i < vectors.rows(); ++i)
            for (int j = 0; j < vectors.cols(); ++j)
                vectors(i, j) =
                    std::sin(0.13 * (i + 1) * (j + 1)) + std::cos(0.17 * (i + 3) * (j + 2));
        vectors.colwise().normalize();
        Eigen::MatrixXd h(vectors.rows(), 2), m(vectors.rows(), 2);
        const auto action_start = std::chrono::steady_clock::now();
        const auto result = executor.apply(vectors, budget);
        const double action_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - action_start).count();
        h = result.images.hamiltonian;
        m = result.images.overlap;
        std::cout << std::setprecision(17) << "rhs=" << vectors.cols()
                  << ",block_action_seconds=" << action_seconds
                  << ",per_rhs_seconds=" << action_seconds / vectors.cols()
                  << ",raw_seconds=" << result.work.raw_seconds
                  << ",metric_seconds=" << result.work.metric_seconds
                  << ",elements=" << result.work.element_evaluations
                  << ",beta_columns=" << result.work.beta_columns
                  << ",source_products=" << result.work.source_products
                  << ",plan_bytes=" << result.work.plan_bytes
                  << ",panel_bytes=" << result.work.panel_bytes
                  << ",covalent_reduction=" << result.work.covalent_reduction << '\n';
        const Eigen::MatrixXd hp = vectors.transpose() * h, mp = vectors.transpose() * m;
        const double hs = (hp - hp.transpose()).norm() / std::max(1.0, hp.norm());
        const double ms = (mp - mp.transpose()).norm() / std::max(1.0, mp.norm());
        std::cout << "hermiticity_H=" << hs << "\nhermiticity_M=" << ms << "\npreparation_seconds="
                  << std::chrono::duration<double>(prepared - start).count()
                  << "\nscf_steps=0\nhvp_actions=0\n";
        if (!std::isfinite(hs) || !std::isfinite(ms) || hs > 2e-10 || ms > 2e-11 ||
            (mp.diagonal().array() <= 0).any())
            throw std::runtime_error("operator-point Hermiticity/positive-norm check failed");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
