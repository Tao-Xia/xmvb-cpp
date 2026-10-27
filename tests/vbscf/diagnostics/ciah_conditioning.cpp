#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/memory.hpp"
#include "input/loading/loader.hpp"
#include "vbscf/derivatives/gradient/orbital/evaluator.hpp"
#include "vbscf/orbitals/gauge/support_preserving.hpp"
#include "vbscf/orbitals/preparation/preparer.hpp"

namespace {
using namespace xmvb::vb;
Eigen::MatrixXd read_orbitals(const char* path, const OrbitalPreparationInput& o) {
    Eigen::MatrixXd b(o.n_basis_functions, o.n_orbitals);
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(b.data()), b.size() * sizeof(double));
    if (!file)
        throw std::runtime_error("invalid orbital ray snapshot");
    return b;
}
void set_orbitals(OrbitalPreparationInput& o, const Eigen::MatrixXd& b) {
    for (std::size_t i = 0; i < o.n_orbitals; ++i)
        for (int j = 0; j < stored_sparse_orbital_coefficient_count(o, i); ++j) {
            const auto slot = i * o.n_basis_functions + j;
            o.orbital_value_table[slot] = b(o.orbital_basis_index_table[slot] - 1, i);
        }
}
} // namespace

/** @brief Postprocess recorded CIAH points; no optimizer or HVP is executed. */
int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 4 && !(argc == 5 && std::string(argv[4]) == "--endpoints"))
            throw std::runtime_error(
                "usage: check_ciah_conditioning INPUT.xmi [ORIGIN.bin TRIAL.bin]");
        xmvb::memory::configure({});
        auto loaded = load_vbscf_input_with_timings(argv[1]);
        auto& input = loaded.input;
        auto& o = input.orbital_preparation_input;
        if (argc == 2) {
            std::ofstream ao("ao_overlap.bin", std::ios::binary);
            ao.write(reinterpret_cast<const char*>(o.ao_overlap_matrix.data()),
                     o.ao_overlap_matrix.size() * sizeof(double));
            std::ofstream topology("topology.txt");
            const auto& x = input.structure_data;
            topology << std::setprecision(17) << o.n_basis_functions << ' ' << o.n_orbitals << ' '
                     << (o.n_total_electrons - o.n_active_electrons) / 2 << ' '
                     << o.n_active_orbitals << ' ' << x.alpha_strings.size() << ' '
                     << x.beta_strings.size() << ' ' << x.n_structures << ' ' << x.pair_alpha.size()
                     << '\n';
            for (const auto* table : {&x.alpha_strings, &x.beta_strings})
                for (std::size_t i = 0; i < table->size(); ++i) {
                    const auto row = (*table)[i];
                    topology << row.size();
                    for (int a : row)
                        topology << ' ' << a;
                    topology << '\n';
                }
            for (std::size_t p = 0; p < x.pair_alpha.size(); ++p) {
                topology << x.pair_alpha[p] << ' ' << x.pair_beta[p] << ' '
                         << x.offsets[p + 1] - x.offsets[p];
                for (auto j = x.offsets[p]; j < x.offsets[p + 1]; ++j)
                    topology << ' ' << x.terms[j].structure_index << ' ' << x.terms[j].coefficient;
                topology << '\n';
            }
            if (!ao || !topology)
                throw std::runtime_error("cannot write conditioning metadata");
            return 0;
        }
        const Eigen::MatrixXd origin = read_orbitals(argv[2], o), trial = read_orbitals(argv[3], o);
        const OrbitalGradientEvaluator evaluator;
        std::cout << std::setprecision(17);
        const auto fractions =
            argc == 5 ? std::vector<double>{0.0, 1.0} : std::vector<double>{0.25, 0.5, 0.75};
        for (double fraction : fractions) {
            set_orbitals(o, origin + fraction * (trial - origin));
            apply_support_preserving_inactive_gauge(&o);
            balance_active_gauge(&o);
            const auto prepared = ActiveSpaceOrbitalPreparer{}.prepare(o);
            const Eigen::Map<const Eigen::MatrixXd> gram(
                prepared.active_orbital_overlap_matrix.data(),
                o.n_active_orbitals,
                o.n_active_orbitals);
            const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(gram);
            const auto value = evaluator.evaluate_forward(input,
                                                          {0},
                                                          {1.0},
                                                          loaded.nuclear_repulsion_energy,
                                                          StructureEigensolver::Davidson,
                                                          StructureSolveAccuracy{},
                                                          Eigen::MatrixXd{});
            std::cout << "RAY,fraction=" << fraction << ",energy=" << value.total_energy()
                      << ",condition="
                      << spectrum.eigenvalues().tail(1)[0] / spectrum.eigenvalues()[0]
                      << ",lambda_min=" << spectrum.eigenvalues()[0] << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
