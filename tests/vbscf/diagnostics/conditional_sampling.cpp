#include "conditional_columns.hpp"

#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

#include "core/memory.hpp"
#include "input/loading/loader.hpp"
#include "vbscf/derivatives/gradient/orbital/evaluator.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/orbitals/gauge/support_preserving.hpp"
#include "vbscf/structures/orthogonal_ci/integrals.hpp"

namespace {
using namespace conditional_audit;
/** Only norms are needed to construct a direct Born guide. Hamiltonian population
 * enumeration below is a validation oracle and must not be charged as guide work.
 */
Eigen::VectorXd build_guide(const xmvb::mcscf::RawCiPlan& source,
                            const Eigen::VectorXd& values,
                            const xmvb::vb::ExteriorOrbitalTransform& alpha,
                            const xmvb::vb::ExteriorOrbitalTransform& beta) {
    const int k = beta.dimension();
    const int width = std::min(std::max(1, k - 1), 8 * xmvb::cooperative_openmp_thread_count());
    Eigen::MatrixXd right(k, width), left(alpha.dimension(), width);
    Eigen::VectorXd norms(k);
    for (int first = 0; first < k; first += width) {
        const int count = std::min(width, k - first);
        auto u = right.leftCols(count), y = left.leftCols(count);
        u.setZero();
        for (int j = 0; j < count; ++j)
            u(first + j, j) = 1;
        beta.apply_adjoint_left_block(u);
        source.multiply_source(values, u, y);
        alpha.apply_left_block(y);
        for (int j = 0; j < count; ++j)
            norms[first + j] = y.col(j).squaredNorm();
    }
    return norms;
}
void self_test() {
    for (auto shape : {std::array{4, 2, 2}, std::array{5, 3, 1}, std::array{4, 0, 2}}) {
        const auto [n, na, nb] = shape;
        const xmvb::mcscf::SpinStrings a(n, na), b(n, nb);
        const int pairs = n * (n + 1) / 2;
        xmvb::mcscf::ActiveIntegrals integrals;
        integrals.one_electron.resize(n, n);
        Eigen::MatrixXd v(pairs, pairs);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                integrals.one_electron(i, j) = std::cos(0.21 * (i + j + 1));
        for (int i = 0; i < pairs; ++i)
            for (int j = 0; j < pairs; ++j)
                v(i, j) = 0.03 * std::sin(0.19 * (i + 1) * (j + 2));
        integrals.pair_kernel = v.transpose() * v;
        const xmvb::mcscf::DirectCiSigmaAction sigma(a, b, integrals);
        Eigen::MatrixXd r = Eigen::MatrixXd::Identity(n, n);
        for (int j = 0; j < n; ++j) {
            r(j, j) += 0.02 * j;
            for (int i = 0; i < j; ++i)
                r(i, j) = 0.09 * std::sin(0.2 * (i + 1) * (j + 2));
        }
        const xmvb::vb::ExteriorOrbitalTransform ta(a, r), tb(b, r);
        std::vector<xmvb::mcscf::ProductIndex> products;
        std::vector<double> raw;
        Eigen::MatrixXd x = Eigen::MatrixXd::Zero(a.size(), b.size());
        for (int i = 0; i < a.size(); ++i)
            for (int j = 0; j < b.size(); ++j)
                if ((i * b.size() + j) % 3 == 0) {
                    products.push_back(static_cast<xmvb::mcscf::ProductIndex>(i) * b.size() + j);
                    raw.push_back(x(i, j) = std::sin(0.17 * (i + 2) * (j + 1)));
                }
        Eigen::VectorXd values = Eigen::Map<Eigen::VectorXd>(raw.data(), raw.size());
        const xmvb::mcscf::RawCiPlan plan(
            sigma.alpha_connections(), sigma.beta_connections_view(), products);
        Columns columns(sigma, ta, tb, plan, values);
        Eigen::MatrixXd y = x;
        ta.apply_left(&y);
        tb.apply_right(&y);
        const Eigen::MatrixXd h = sigma.apply(y);
        double error = 0;
        for (int j = 0; j < b.size(); ++j) {
            columns.evaluate(j, 0, false);
            error = std::max(error, (columns.alpha_panel().col(0) - y.col(j)).norm());
            error = std::max(error, (columns.image() - h.col(j)).norm());
        }
        if (error > 3e-11 * std::max(1.0, h.norm()))
            throw std::runtime_error("joint column disagrees with independent full tiny sigma");
        std::cout << "tiny_joint,n=" << n << ",na=" << na << ",nb=" << nb << ",error=" << error
                  << '\n';
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            self_test();
            return 0;
        }
        if (argc != 3)
            throw std::invalid_argument("usage: test_conditional_sampling INPUT.xmi PREFIX");
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
        const auto& ctx = *evaluated.second_order_context;
        const auto& action = *ctx.structure_action;
        const auto view = action.orthogonal_view();
        if (!ctx.selected_state_structure_images)
            throw std::runtime_error("current fixed-point CI reference images unavailable");
        const Eigen::VectorXd c = ctx.selected_state_eigenvectors.col(0);
        const auto& reference = *ctx.selected_state_structure_images;
        const double norm = c.dot(reference.overlap.col(0)),
                     energy = c.dot(reference.hamiltonian.col(0)) / norm;
        Eigen::VectorXd values = Eigen::VectorXd::Zero(view.products.size());
        for (std::size_t p = 0; p < view.products.size(); ++p)
            for (std::size_t t = view.offsets[p]; t < view.offsets[p + 1]; ++t)
                values[p] += view.terms[t].coefficient * c[view.terms[t].structure];
        const xmvb::mcscf::RawCiPlan source(
            view.sigma.alpha_connections(), view.sigma.beta_connections_view(), view.products);
        const double preparation = elapsed(start);
        const auto setup = Clock::now();
        Columns generator(view.sigma, view.alpha, view.beta, source, values);
        const double setup_seconds = elapsed(setup);
        const int k = action.n_beta_strings();
        // Independent streamed Hbar*X reference for three physical output columns.
        // This is a new bounded correctness check, not a historical H/M timing rerun.
        const auto validation = Clock::now();
        const std::array<int, 3> probes{0, k / 2, k - 1};
        Eigen::MatrixXd u = Eigen::MatrixXd::Zero(k, probes.size());
        for (int j = 0; j < static_cast<int>(probes.size()); ++j)
            u(probes[j], j) = 1;
        view.beta.apply_adjoint_left_block(u);
        Eigen::MatrixXd z = Eigen::MatrixXd::Zero(view.alpha.dimension(), 2 * probes.size());
        const auto& prepared = ctx.prepared_active_space;
        const auto integrals = xmvb::vb::orthogonalize_active_integrals(
            prepared.orbital_result.active_orbital_overlap_matrix,
            prepared.active_space_one_electron_result.h1e_act,
            prepared.active_space_two_electron_result,
            orbitals.n_active_orbitals);
        const xmvb::mcscf::SimilarityHamiltonian canonical(integrals, integrals.orbital_transform);
        const auto beta_columns = source.beta_columns();
        Eigen::MatrixXd slab(view.alpha.dimension(),
                             std::min<std::size_t>(256, beta_columns.size()));
        for (std::size_t first = 0; first < beta_columns.size(); first += slab.cols()) {
            const int count = std::min<std::size_t>(slab.cols(), beta_columns.size() - first);
            const auto betas = beta_columns.subspan(first, count);
            Eigen::MatrixXd weights(count, probes.size());
            for (int j = 0; j < count; ++j)
                weights.row(j) = u.row(betas[j]);
            auto active = slab.leftCols(count);
            source.apply_columns(canonical, values, betas, active);
            z.leftCols(probes.size()).noalias() += active * weights;
            source.source_columns(values, betas, active);
            z.rightCols(probes.size()).noalias() += active * weights;
        }
        view.alpha.apply_left_block(z);
        double vector_error = 0;
        for (int j = 0; j < static_cast<int>(probes.size()); ++j) {
            generator.evaluate(probes[j], energy, false);
            vector_error =
                std::max(vector_error,
                         (generator.image() - z.col(j)).norm() / std::max(1.0, z.col(j).norm()));
            vector_error =
                std::max(vector_error,
                         (generator.alpha_panel().col(0) - z.col(probes.size() + j)).norm() /
                             std::max(1.0, z.col(probes.size() + j).norm()));
        }
        if (vector_error > 2e-10)
            throw std::runtime_error(
                "molecular joint columns disagree with canonical streamed reference");
        const double validation_seconds = elapsed(validation);
        z.resize(0, 0);
        slab.resize(0, 0);
        u.resize(0, 0);
        std::vector<Result> population(k);
        std::ofstream census(prefix + "-population.csv");
        census << std::setprecision(17)
               << "beta,norm,numerator,local_energy_Eh,conditional_residual_squared,"
                  "full_residual_squared,row_residual_squared,seconds\n";
        long double total_norm = 0, total_numerator = 0, conditional_variance = 0,
                    full_variance = 0, row_variance = 0;
        const auto survey = Clock::now();
        for (int b = 0; b < k; ++b) {
            const auto tick = Clock::now();
            const auto r = generator.evaluate(b, energy);
            population[b] = r;
            const double seconds = elapsed(tick);
            const double delta = r.numerator - energy * r.norm;
            const double term = r.norm > 0 ? delta * delta / r.norm : 0;
            census << b << ',' << r.norm << ',' << r.numerator << ','
                   << (r.norm > 0 ? r.numerator / r.norm : energy) << ',' << term << ','
                   << r.residual_norm << ',' << r.row_residual_norm << ',' << seconds << '\n';
            total_norm += r.norm;
            total_numerator += r.numerator;
            conditional_variance += term / norm;
            full_variance += r.residual_norm / norm;
            row_variance += r.row_residual_norm / norm;
        }
        const double survey_seconds = elapsed(survey);
        const double norm_error = std::abs(static_cast<double>(total_norm / norm) - 1),
                     energy_error = static_cast<double>(total_numerator / total_norm) - energy;
        if (norm_error > 2e-10 || std::abs(energy_error) > 2e-9 ||
            conditional_variance > row_variance * (1 + 1e-8L) + 1e-20L)
            throw std::runtime_error(
                "physical column norm/energy/conditional variance identity failed");
        // A separate norm-only sweep provides the actual direct Born guide. Its
        // cost is charged; the joint energy population is only a variance oracle.
        const auto guide_start = Clock::now();
        const Eigen::VectorXd guide = build_guide(source, values, view.alpha, view.beta);
        const double guide_seconds = elapsed(guide_start);
        double guide_error = 0;
        std::vector<double> probabilities(k), energies(k);
        for (int b = 0; b < k; ++b) {
            guide_error = std::max(guide_error, std::abs(guide[b] - population[b].norm));
            probabilities[b] = guide[b] / norm;
            energies[b] =
                population[b].norm > 0 ? population[b].numerator / population[b].norm : energy;
        }
        if (guide_error > 2e-10 * norm)
            throw std::runtime_error("norm-only guide disagrees with joint column population");
        std::ofstream trials(prefix + "-trials.csv"), timing(prefix + "-timing.csv");
        trials << std::setprecision(17) << "samples,replicate,unique_columns,energy_error_Eh\n";
        timing << std::setprecision(17)
               << "samples,unique_columns,draw_seconds,joint_seconds,total_seconds,"
                  "including_guide_setup_seconds,energy_error_Eh,workspace_GiB\n";
        std::vector<std::size_t> counts;
        for (int divisor : {64, 16, 4})
            counts.push_back(std::max(1, (k + divisor - 1) / divisor));
        counts.push_back(k);
        counts.push_back(4 * static_cast<std::size_t>(k));
        for (auto samples : counts) {
            std::vector<std::size_t> online_counts;
            for (int replicate = 0; replicate < 64; ++replicate) {
                std::mt19937_64 rng(0x3759afULL + replicate);
                std::discrete_distribution<int> distribution(probabilities.begin(),
                                                             probabilities.end());
                std::vector<std::size_t> multiplicities(k, 0);
                for (std::size_t i = 0; i < samples; ++i)
                    ++multiplicities[distribution(rng)];
                double estimate = 0;
                int unique = 0;
                for (int b = 0; b < k; ++b) {
                    if (multiplicities[b])
                        ++unique;
                    estimate += static_cast<double>(multiplicities[b]) / samples * energies[b];
                }
                trials << samples << ',' << replicate << ',' << unique << ',' << estimate - energy
                       << '\n';
                if (replicate == 0)
                    online_counts = std::move(multiplicities);
            }
            const auto draw_start = Clock::now();
            std::mt19937_64 rng(0x3759afULL);
            std::discrete_distribution<int> distribution(probabilities.begin(),
                                                         probabilities.end());
            std::vector<std::size_t> multiplicities(k, 0);
            for (std::size_t i = 0; i < samples; ++i)
                ++multiplicities[distribution(rng)];
            const double draw_seconds = elapsed(draw_start);
            if (multiplicities != online_counts)
                throw std::runtime_error("replicate seed mismatch");
            const auto execute = Clock::now();
            double estimate = 0;
            int unique = 0;
            for (int b = 0; b < k; ++b)
                if (multiplicities[b]) {
                    const auto r = generator.evaluate(b, energy, false);
                    const double e = r.numerator / r.norm;
                    if (std::abs(e - energies[b]) > 2e-9)
                        throw std::runtime_error(
                            "online joint column disagrees with population reference");
                    estimate += static_cast<double>(multiplicities[b]) / samples * e;
                    ++unique;
                }
            const double joint_seconds = elapsed(execute);
            timing << samples << ',' << unique << ',' << draw_seconds << ',' << joint_seconds << ','
                   << draw_seconds + joint_seconds << ','
                   << setup_seconds + guide_seconds + draw_seconds + joint_seconds << ','
                   << estimate - energy << ',' << generator.bytes() / double(1ULL << 30) << '\n';
        }
        std::cout << std::setprecision(17) << "structures=" << c.size() << "\nbeta_columns=" << k
                  << "\nvalidation_relative_error=" << vector_error
                  << "\nvalidation_seconds=" << validation_seconds
                  << "\npreparation_seconds=" << preparation
                  << "\nspin_setup_seconds=" << setup_seconds << "\nguide_seconds=" << guide_seconds
                  << "\nguide_absolute_error=" << guide_error
                  << "\npopulation_seconds=" << survey_seconds << "\nactive_Rayleigh_Eh=" << energy
                  << "\nenergy_error_Eh=" << energy_error << "\nnorm_relative_error=" << norm_error
                  << "\nconditional_variance_Eh2=" << conditional_variance
                  << "\nrow_variance_Eh2=" << row_variance
                  << "\nfull_residual_norm2=" << full_variance
                  << "\nsamples_rms_1e-7=" << std::ceil(conditional_variance / 1e-14L)
                  << "\nworkspace_GiB=" << generator.bytes() / double(1ULL << 30)
                  << "\nfill_seconds=" << generator.fill_seconds
                  << "\nbeta_transform_seconds=" << generator.beta_seconds
                  << "\nsource_multiply_seconds=" << generator.source_seconds
                  << "\nalpha_transform_seconds=" << generator.alpha_seconds
                  << "\nfinish_seconds=" << generator.finish_seconds
                  << "\nscf_steps=0\nhvp_actions=0\n";
    } catch (const std::exception& e) {
        std::cerr << "conditional sampling audit: " << e.what() << '\n';
        return 1;
    }
}
