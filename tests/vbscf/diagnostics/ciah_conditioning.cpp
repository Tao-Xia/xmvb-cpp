#include "orbital_snapshot.hpp"
#include <algorithm>
#include <limits>
#include <optional>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/memory.hpp"
#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/optimization/preconditioners/hessian_diagonal.hpp"
#include "input/loading/loader.hpp"
#include "vbscf/derivatives/gradient/orbital/evaluator.hpp"
#include "vbscf/orbitals/gauge/support_preserving.hpp"
#include "vbscf/orbitals/preparation/preparer.hpp"

namespace {
using namespace xmvb::vb;

/** @brief Diagnostic only: the ray prototype was not promoted to the optimizer. */
std::optional<double>
ray_fraction(double source, double trial, double slope, double energy_resolution) {
    const double curvature = trial - source - slope;
    if (!(slope < 0.0 && curvature > 0.0))
        return std::nullopt;
    const double fraction = -0.5 * slope / curvature;
    const double gain = curvature * (1.0 - fraction) * (1.0 - fraction);
    const double roundoff = std::numeric_limits<double>::epsilon() *
                            (std::abs(source) + std::abs(trial) + std::abs(slope));
    if (fraction > 0.0 && fraction < 1.0 && gain > std::max(roundoff, energy_resolution))
        return fraction;
    return std::nullopt;
}

using xmvb::test::read_orbitals;
using xmvb::test::set_orbitals;
} // namespace

/** @brief Postprocess recorded CIAH points; no optimizer or HVP is executed. */
int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 4 &&
            !(argc == 5 &&
              (std::string(argv[4]) == "--endpoints" || std::string(argv[4]) == "--refine" ||
               std::string(argv[4]) == "--inverse")))
            throw std::runtime_error("usage: check_ciah_conditioning INPUT.xmi [ORIGIN.bin "
                                     "TRIAL.bin [--endpoints|--refine|--inverse]]");
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
        if (argc == 5 && std::string(argv[4]) == "--inverse") {
            set_orbitals(o, origin);
            const auto accepted = evaluator.evaluate_without_reference_energy_gradient(
                input,
                {0},
                {1.0},
                loaded.nuclear_repulsion_energy,
                StructureEigensolver::Davidson,
                StructureSolveAccuracy{},
                Eigen::MatrixXd{});
            const auto& prep = accepted.orbital_preparation_result;
            const SparseParameterLayout layout(o);
            OrbitalChart chart(o,
                               layout,
                               prep.auxiliary_orbital_matrix.leftCols(o.n_orbitals),
                               prep.physical_orbital_frame.normalized_orbital_matrix,
                               nullptr,
                               false,
                               input.complete_active_space,
                               nullptr,
                               OrbitalChart::Coordinates::Unwhitened);
            const auto model =
                build_analytic_orbital_diagonal(input, *accepted.second_order_context);
            const Eigen::MatrixXd gram =
                model.mo_coefficients.transpose() * o.ao_overlap_matrix * model.mo_coefficients;
            std::cout << "INVERSE,mo_metric_error="
                      << (gram - Eigen::MatrixXd::Identity(gram.rows(), gram.cols())).norm()
                      << std::endl;
            chart.set_curvature(o, nullptr, &model, OrbitalChart::CurvatureUse::AugmentedHessian);
            std::cout << "INVERSE,certificate=PASS" << std::endl;
            return 0;
        }
        const bool refine = argc == 5 && std::string(argv[4]) == "--refine";
        auto fractions =
            argc == 5 ? std::vector<double>{0.0, 1.0} : std::vector<double>{0.25, 0.5, 0.75};
        double source_energy = 0.0, slope = 0.0;
        if (refine) {
            set_orbitals(o, origin);
            const auto source = evaluator.evaluate_without_reference_energy_gradient(
                input,
                {0},
                {1.0},
                loaded.nuclear_repulsion_energy,
                StructureEigensolver::Davidson,
                StructureSolveAccuracy{},
                Eigen::MatrixXd{});
            source_energy = source.scf_result.total_energy;
            for (std::size_t i = 0; i < o.n_orbitals; ++i)
                for (int j = 0; j < stored_sparse_orbital_coefficient_count(o, i); ++j) {
                    const auto slot = i * o.n_basis_functions + j;
                    const auto ao = o.orbital_basis_index_table[slot] - 1;
                    slope += source.sparse_orbital_energy_gradient[slot] *
                             (trial(ao, i) - origin(ao, i));
                }
            fractions = {1.0};
            std::cout << "REFINE,source_energy=" << source_energy << ",slope=" << slope << '\n';
        }
        for (std::size_t sample = 0; sample < fractions.size(); ++sample) {
            const double fraction = fractions[sample];
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
            if (refine && sample == 0) {
                const auto shorter = ray_fraction(source_energy,
                                                  value.total_energy(),
                                                  slope,
                                                  StructureSolveAccuracy{}.energy_tolerance);
                if (shorter)
                    fractions.push_back(*shorter);
            }
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
