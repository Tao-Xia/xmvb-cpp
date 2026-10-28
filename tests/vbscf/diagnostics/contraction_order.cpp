#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/memory.hpp"
#include "input/loading/loader.hpp"
#include "mcscf/ci/product_support.hpp"
#include "mcscf/ci/sigma.hpp"
#include "vbscf/derivatives/gradient/orbital/evaluator.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/orbitals/gauge/support_preserving.hpp"
#include "vbscf/structures/assembly/sources.hpp"
#include "vbscf/structures/assembly/operator_first.hpp"
#include "vbscf/structures/orthogonal_ci/integrals.hpp"

/** @brief Compare two current exact contraction orders at one molecular CI point.
 * Counts arithmetic separately from timings. No full product array, amplitude
 * truncation, optimizer change, or historical executable is used.
 */
int main(int argc, char** argv) {
    try {
        const bool census_only = argc == 4 && std::string(argv[3]) == "--census-only";
        const bool operator_first = argc == 4 && std::string(argv[3]) == "--operator-first";
        if (argc != 3 && !census_only && !operator_first)
            throw std::invalid_argument(
                "usage: test_contraction_order INPUT.xmi WORKSPACE_MiB [--census-only]");
        using Clock = std::chrono::steady_clock;
        const auto seconds = [](auto a, auto b) {
            return std::chrono::duration<double>(b - a).count();
        };
        const auto mib = std::stoull(argv[2]);
        if (!mib || mib > (std::size_t{1} << 20))
            throw std::invalid_argument("invalid diagnostic workspace");
        const std::size_t budget = mib * 1024 * 1024;
        xmvb::memory::configure({});
        auto loaded = xmvb::vb::load_vbscf_input_with_timings(argv[1]);
        auto& orbitals = loaded.input.orbital_preparation_input;
        if (!orbitals.maintain_inactive_gauge)
            xmvb::vb::apply_support_preserving_inactive_gauge(&orbitals);
        xmvb::vb::balance_active_gauge(&orbitals);
        const auto start = Clock::now();
        const auto evaluated =
            xmvb::vb::OrbitalGradientEvaluator().evaluate_without_reference_energy_gradient(
                loaded.input,
                {0},
                {1.0},
                loaded.nuclear_repulsion_energy,
                xmvb::vb::StructureEigensolver::Davidson,
                {},
                Eigen::MatrixXd{});
        const auto initialized = Clock::now();
        const auto& context = evaluated.second_order_context;
        if (!context || !context->structure_action)
            throw std::runtime_error("initial CI did not retain its structure action");
        const auto& action = *context->structure_action;
        const auto view = action.orthogonal_view();
        if (operator_first) {
            if (!context->selected_state_structure_images)
                throw std::runtime_error("initial CI did not retain its reference images");
            const auto setup_start = Clock::now();
            const auto& prepared = context->prepared_active_space;
            const auto integrals = xmvb::vb::orthogonalize_active_integrals(
                prepared.orbital_result.active_orbital_overlap_matrix,
                prepared.active_space_one_electron_result.h1e_act,
                prepared.active_space_two_electron_result,
                orbitals.n_active_orbitals);
            const xmvb::vb::OperatorFirstAction executor(
                view, action.n_structures(), integrals, integrals.orbital_transform);
            const auto action_start = Clock::now();
            const auto result =
                executor.apply(context->selected_state_eigenvectors.leftCols(1), budget);
            const double action_seconds = seconds(action_start, Clock::now());
            const auto& reference = *context->selected_state_structure_images;
            const double h_error = (result.images.hamiltonian - reference.hamiltonian).norm() /
                                   std::max(1.0, reference.hamiltonian.norm());
            const double m_error = (result.images.overlap - reference.overlap).norm() /
                                   std::max(1.0, reference.overlap.norm());
            std::cout << std::setprecision(17)
                      << "operator_first_setup_seconds=" << seconds(setup_start, action_start)
                      << "\noperator_first_raw_seconds=" << result.work.raw_seconds
                      << "\noperator_first_metric_seconds=" << result.work.metric_seconds
                      << "\noperator_first_action_seconds=" << action_seconds
                      << "\noperator_first_phase_sum_seconds="
                      << result.work.raw_seconds + result.work.metric_seconds
                      << "\nelement_evaluations=" << result.work.element_evaluations
                      << "\nsource_products=" << result.work.source_products
                      << "\nbeta_columns=" << result.work.beta_columns
                      << "\ncovalent_reduction=" << result.work.covalent_reduction
                      << "\nplan_bytes=" << result.work.plan_bytes << "\nraw_image_bytes=0"
                      << "\npanel_bytes=" << result.work.panel_bytes
                      << "\nh_relative_error=" << h_error << "\nm_relative_error=" << m_error
                      << '\n';
            if (!std::isfinite(h_error) || !std::isfinite(m_error) || h_error > 2e-10 ||
                m_error > 2e-11)
                throw std::runtime_error("operator-first molecular action disagrees");
            return 0;
        }
        const int n = view.sigma.n_orbitals(), ua = action.n_alpha_strings(),
                  ub = action.n_beta_strings(), pairs = view.sigma.channel_kernel().rows();
        const std::vector<xmvb::mcscf::ProductIndex> products(view.products.begin(),
                                                              view.products.end());
        const xmvb::vb::SpinSources sources(products, ua, ub, view.exchange_parity);
        if (!view.support || !view.support->has_channel_envelopes())
            throw std::runtime_error("this census requires the complete covalent certificate");
        const auto& alpha = view.sigma.alpha_connections();
        const auto& beta = view.sigma.beta_connections_view();
        std::size_t targets = 0, source_targets = 0, links = 0, diagonals = 0;
        long double channel_products = 0, full_channel_products = 0;
        long double occupied_channel_products = 0, diagonal_folded_products = 0;
        std::vector<int> rows;
        for (int b = 0; b < ub; ++b) {
            rows.clear();
            const std::size_t upper =
                view.exchange_parity ? b + (view.exchange_parity > 0 ? 1 : 0) : ua;
            view.support->append_column_rows(b, rows, 1, upper);
            for (const int a : rows) {
                ++targets;
                const auto [first, last] = view.support->channel_envelope(a, b);
                const int channels = first < 0 ? pairs : (first + 1) * (n - 1 - last);
                channel_products += static_cast<long double>(channels) * channels;
                full_channel_products += static_cast<long double>(channels) * pairs;
                std::array<std::uint64_t, 32> channel_mask{};
                if (pairs > static_cast<int>(channel_mask.size() * 64))
                    throw std::runtime_error("channel census mask is too small");
                const auto mark = [&](int pair) {
                    channel_mask[pair / 64] |= std::uint64_t{1} << (pair % 64);
                };
                const auto count_alpha = [&](int link, int) {
                    ++links;
                    mark(alpha.singles(a)[link].pair);
                };
                const auto count_beta = [&](int link, int) {
                    ++links;
                    mark(beta.singles(b)[link].pair);
                };
                int occupied_diagonals = 0;
                if (first < 0) {
                    ++source_targets;
                    diagonals += alpha.n_occupied() + beta.n_occupied();
                    const auto zero = view.support->zero_prefixes(a, b);
                    alpha.visit_supported_sources(a, zero, count_alpha);
                    beta.visit_supported_sources(b, zero, count_beta);
                    const auto mask = alpha.basis.masks()[a] | beta.basis.masks()[b];
                    occupied_diagonals = std::popcount(mask);
                    for (const int p : xmvb::mcscf::OccupiedOrbitals(mask))
                        mark(p * (p + 1) / 2 + p);
                } else {
                    alpha.visit_lower_sources(a, first, last, count_alpha);
                    beta.visit_lower_sources(b, first, last, count_beta);
                }
                int nonzero_channels = 0;
                for (const auto word : channel_mask)
                    nonzero_channels += std::popcount(word);
                occupied_channel_products +=
                    static_cast<long double>(nonzero_channels) * nonzero_channels;
                const int folded = nonzero_channels - occupied_diagonals +
                                   static_cast<int>(occupied_diagonals > 0);
                diagonal_folded_products += static_cast<long double>(folded) * folded;
            }
        }
        const auto counted = Clock::now();
        if (census_only) {
            std::cout << std::setprecision(17) << "orbitals=" << n << "\ntargets=" << targets
                      << "\nproduct_mix_flops=" << 2 * channel_products
                      << "\noccupied_channel_mix_flops_ideal=" << 2 * occupied_channel_products
                      << "\ndiagonal_folded_mix_flops_ideal=" << 2 * diagonal_folded_products
                      << "\ncensus_seconds=" << seconds(initialized, counted) << '\n';
            return 0;
        }
        const Eigen::MatrixXd vector = context->selected_state_eigenvectors.leftCols(1);
        std::cerr << "CONTRACTION_ORDER_BEGIN,kind=product\n";
        const auto product = action.apply_streamed(vector, budget);
        const auto product_done = Clock::now();
        if (product.tiles.source_columns)
            throw std::runtime_error("automatic action chose factors instead of product tiles");
        std::cerr << "CONTRACTION_ORDER_BEGIN,kind=factor\n";
        const auto factor = action.apply_sources(vector, budget);
        const auto factor_done = Clock::now();
        const double h_error = (product.hamiltonian - factor.hamiltonian).norm() /
                               std::max(1.0, product.hamiltonian.norm());
        const double m_error =
            (product.overlap - factor.overlap).norm() / std::max(1.0, product.overlap.norm());
        // One multiply-add counts as two FLOPs. These are the leading dense
        // channel products; source loads, shears and projection are separate.
        const long double r = sources.nodes.size();
        const long double mix_flops = 2 * r * (ua + static_cast<long double>(ub)) * pairs * pairs;
        const long double sample_flops = 2 * r * products.size() * (pairs + 3);
        std::cout << std::setprecision(17) << "orbitals=" << n
                  << "\nstructures=" << action.n_structures() << "\nstrings=" << ua
                  << "\nraw_products=" << products.size()
                  << "\nsource_factors=" << sources.nodes.size()
                  << "\nparity=" << view.exchange_parity << "\nchannels=" << pairs
                  << "\ntargets=" << targets << "\nsource_targets=" << source_targets
                  << "\nproduct_scatter_links=" << links << "\nproduct_diagonal_adds=" << diagonals
                  << "\nproduct_mix_flops=" << 2 * channel_products
                  << "\noccupied_channel_mix_flops_ideal=" << 2 * occupied_channel_products
                  << "\ndiagonal_folded_mix_flops_ideal=" << 2 * diagonal_folded_products
                  << "\nproduct_unrestricted_mix_flops=" << 2 * full_channel_products
                  << "\nfactor_mix_flops=" << mix_flops
                  << "\nfactor_projection_dot_flops=" << sample_flops
                  << "\nfactor_scatter_links_one_channel_pass="
                  << r * (alpha.links.size() + beta.links.size())
                  << "\ninitial_ci_seconds=" << seconds(start, initialized)
                  << "\ncensus_seconds=" << seconds(initialized, counted)
                  << "\nproduct_action_seconds=" << seconds(counted, product_done)
                  << "\nfactor_action_seconds=" << seconds(product_done, factor_done)
                  << "\nworkspace_bytes=" << budget
                  << "\nfactor_workspace_bytes=" << factor.tiles.source_workspace_bound
                  << "\nfactor_source_panels=" << factor.tiles.source_tiles
                  << "\nfactor_channel_blocks=" << factor.tiles.channel_blocks
                  << "\nh_relative_error=" << h_error << "\nm_relative_error=" << m_error
                  << "\ninitial_energy_Eh=" << evaluated.scf_result.total_energy << '\n';
        if (!std::isfinite(h_error) || !std::isfinite(m_error) || h_error > 2e-10 ||
            m_error > 2e-11)
            throw std::runtime_error("exact contraction orders disagree");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
