#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

#include "core/memory.hpp"
#include "input/loading/loader.hpp"
#include "sampling_stats.hpp"
#include "vbscf/derivatives/gradient/orbital/evaluator.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/orbitals/gauge/support_preserving.hpp"
#include "vbscf/structures/assembly/operator_first.hpp"
#include "vbscf/structures/orthogonal_ci/integrals.hpp"

namespace {
using namespace sampling_audit;
std::size_t rss_bytes() {
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line))
        if (line.starts_with("VmRSS:"))
            return std::stoull(line.substr(6)) * 1024;
    return 0;
}
void self_test() {
    Eigen::MatrixXd y(4, 3), f(2, 3), b(4, 2);
    y << 1, -3, 2, 2, 4, 1, 2, 1, 3, 0, 1, 1;
    f << 1, 1, 1, -1, 0, 2;
    b << 0.5, 1, 2, -0.3, 0.2, 0.7, -0.1, 0.3;
    const Eigen::VectorXd p = (Eigen::Vector3d() << 0.2, 0.3, 0.5).finished();
    Method method{"test", p, b * f.rowwise().sum(), b};
    const Eigen::VectorXd reference = y.rowwise().sum(), c = Eigen::Vector2d(0.7, -0.2);
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(4);
    Eigen::Vector3d direct = Eigen::Vector3d::Zero();
    for (int j = 0; j < 3; ++j) {
        Eigen::VectorXd weights = Eigen::VectorXd::Zero(3);
        weights[j] = 1 / p[j];
        const Eigen::VectorXd value = y * weights + correction(method, f, weights);
        mean.noalias() += p[j] * value;
        const Eigen::VectorXd delta = value - reference;
        const double denom = c.dot(reference.tail(2)), e = c.dot(reference.head(2)) / denom;
        const double de = c.dot(delta.head(2) - e * delta.tail(2)) / denom;
        direct += p[j] * Eigen::Vector3d(
                             delta.head(2).squaredNorm(), delta.tail(2).squaredNorm(), de * de);
    }
    const Eigen::MatrixXd residual = y - b * f;
    if ((mean - reference).norm() > 1e-12 ||
        (direct - variance(residual, p, reference, c)).norm() > 1e-9)
        throw std::runtime_error("independent unbiasedness/variance oracle failed");
    std::cout << "independent importance/control mean and variance checks pass\n";
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            self_test();
            return 0;
        }
        if (argc != 3)
            throw std::invalid_argument("usage: test_sampling INPUT.xmi OUTPUT_PREFIX");
        const std::string prefix = argv[2];
        const auto start = Clock::now();
        xmvb::memory::configure({});
        auto loaded = xmvb::vb::load_vbscf_input_with_timings(argv[1]);
        auto& orbitals = loaded.input.orbital_preparation_input;
        if (!orbitals.maintain_inactive_gauge)
            xmvb::vb::apply_support_preserving_inactive_gauge(&orbitals);
        xmvb::vb::balance_active_gauge(&orbitals);
        const auto evaluated =
            xmvb::vb::OrbitalGradientEvaluator().evaluate_without_reference_energy_gradient(
                loaded.input,
                {0},
                {1.0},
                loaded.nuclear_repulsion_energy,
                xmvb::vb::StructureEigensolver::Davidson,
                {},
                Eigen::MatrixXd{});
        const auto& context = *evaluated.second_order_context;
        const auto& action = *context.structure_action;
        const auto& prepared = context.prepared_active_space;
        const auto integrals = xmvb::vb::orthogonalize_active_integrals(
            prepared.orbital_result.active_orbital_overlap_matrix,
            prepared.active_space_one_electron_result.h1e_act,
            prepared.active_space_two_electron_result,
            orbitals.n_active_orbitals);
        const Eigen::VectorXd c = context.selected_state_eigenvectors.col(0);
        const auto view = action.orthogonal_view();
        Point point(view, integrals, integrals.orbital_transform, c);
        const int k = point.targets.size(), n = c.size();
        std::vector<int> ids(k);
        std::iota(ids.begin(), ids.end(), 0);
        const Eigen::VectorXd ones = Eigen::VectorXd::Ones(k);
        const double preparation = elapsed(start);
        const xmvb::vb::OperatorFirstAction executor(
            view, n, integrals, integrals.orbital_transform);
        const auto exact_start = Clock::now();
        const auto exact = executor.apply(c, xmvb::workspace::available());
        const double exact_time = elapsed(exact_start);
        Eigen::VectorXd reference(2 * n);
        reference << exact.images.hamiltonian.col(0), exact.images.overlap.col(0);
        const auto catalogue_start = Clock::now();
        Eigen::MatrixXd catalogue;
        const auto full = point.apply(ids, ones, &catalogue);
        const double catalogue_time = elapsed(catalogue_start);
        const double full_error = (full - reference).norm() / reference.norm();
        const double catalog_error =
            (catalogue.rowwise().sum() - reference).norm() / reference.norm();
        if (full_error > 2e-10 || catalog_error > 2e-10)
            throw std::runtime_error("column census disagrees with current production action");
        const auto proposal_start = Clock::now();
        auto [features, probability] = point.proposal();
        const double proposal_time = elapsed(proposal_start);
        const auto pilot_start = Clock::now();
        std::vector<int> pilot = ids;
        std::mt19937_64 random(0x727c41ULL);
        std::shuffle(pilot.begin(), pilot.end(), random);
        pilot.resize(std::min<std::size_t>(k, 2 * features.rows()));
        Eigen::MatrixXd pilot_y;
        point.apply(pilot, Eigen::VectorXd::Ones(pilot.size()), &pilot_y);
        Eigen::MatrixXd pilot_features(features.rows(), pilot.size());
        for (int j = 0; j < static_cast<int>(pilot.size()); ++j)
            pilot_features.col(j) = features.col(pilot[j]);
        const Eigen::MatrixXd control = pilot_features.transpose()
                                            .completeOrthogonalDecomposition()
                                            .solve(pilot_y.transpose())
                                            .transpose();
        const double pilot_time = elapsed(pilot_start);
        pilot_y.resize(0, 0);
        std::vector<Method> methods;
        methods.push_back(
            {"uniform", Eigen::VectorXd::Constant(k, 1.0 / k), Eigen::VectorXd::Zero(2 * n), {}});
        methods.push_back(
            {"importance", probability, Eigen::VectorXd::Zero(2 * n), {}, proposal_time});
        methods.push_back({"control",
                           probability,
                           control * features.rowwise().sum(),
                           control,
                           proposal_time + pilot_time});
        std::vector<std::size_t> counts;
        for (int divisor : {64, 16, 4})
            counts.push_back(std::max(1, (k + divisor - 1) / divisor));
        counts.push_back(k);
        counts.push_back(4 * static_cast<std::size_t>(k));
        statistical_survey(prefix, catalogue, features, methods, reference, c, counts);
        // Keep only tiny expected output vectors to verify weighted online execution.
        std::vector<Eigen::VectorXd> expected_samples;
        for (const auto& method : methods)
            for (auto samples : counts) {
                const auto weights = draw(method.probability, samples, 0x3759afULL);
                expected_samples.push_back(catalogue * weights +
                                           correction(method, features, weights));
            }
        // The output-space catalogue is a variance oracle, never an online cache.
        catalogue.resize(0, 0);
        std::size_t expected_index = 0;
        std::ofstream timing(prefix + "-timing.csv");
        timing
            << std::setprecision(17)
            << "method,samples,unique_columns,draw_seconds,contraction_seconds,total_seconds,"
               "one_shot_seconds,exact_seconds,H_relative_error,M_relative_error,energy_error_Eh,"
               "positive_denominator,live_rss_GiB,workspace_GiB\n";
        for (const auto& method : methods)
            for (auto samples : counts) {
                const auto tick = Clock::now();
                const auto weights = draw(method.probability, samples, 0x3759afULL);
                const auto active = touched(weights);
                Eigen::VectorXd active_weights(active.size());
                for (int j = 0; j < static_cast<int>(active.size()); ++j)
                    active_weights[j] = weights[active[j]];
                const double draw_time = elapsed(tick);
                const auto contract_start = Clock::now();
                const Eigen::VectorXd value =
                    point.apply(active, active_weights) + correction(method, features, weights);
                const double contract_time = elapsed(contract_start);
                if ((value - expected_samples[expected_index++]).norm() >
                    2e-10 * std::max(1.0, value.norm()))
                    throw std::runtime_error(
                        "weighted online execution disagrees with column oracle");
                const auto e = error(value, reference, c);
                const auto metadata =
                    sizeof(double) *
                    (method.control.size() + method.probability.size() + method.control_sum.size() +
                     (method.name == "uniform" ? 0 : features.size()));
                timing << method.name << ',' << samples << ',' << active.size() << ',' << draw_time
                       << ',' << contract_time << ',' << draw_time + contract_time << ','
                       << method.setup_seconds + draw_time + contract_time << ',' << exact_time
                       << ',' << e.h << ',' << e.m << ',' << e.energy << ',' << e.positive << ','
                       << rss_bytes() / double(1ULL << 30) << ','
                       << (point.panel_bytes + metadata) / double(1ULL << 30) << '\n';
            }
        std::cout << std::setprecision(17) << "structures=" << n << "\nbeta_columns=" << k
                  << "\npreparation_seconds=" << preparation << "\nexact_seconds=" << exact_time
                  << "\ncatalogue_seconds=" << catalogue_time
                  << "\ncolumn_sum_relative_error=" << catalog_error
                  << "\nproduction_relative_error=" << full_error
                  << "\nproposal_seconds=" << proposal_time << "\npilot_columns=" << pilot.size()
                  << "\npilot_seconds=" << pilot_time << "\nfeature_count=" << features.rows()
                  << "\nactive_Rayleigh_Eh=" << c.dot(reference.head(n)) / c.dot(reference.tail(n))
                  << "\nscf_steps=0\nhvp_actions=0\n";
    } catch (const std::exception& e) {
        std::cerr << "sampling audit: " << e.what() << '\n';
        return 1;
    }
}
