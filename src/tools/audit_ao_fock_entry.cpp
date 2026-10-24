#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "input/deck/model.hpp"
#include "input/deck/primary_basis.hpp"
#include "libcint/direct_shell.hpp"
#include "integrals/ao/libcint.hpp"

namespace {

constexpr double kIntegralTolerance = 1.0e-10;

struct FockTerms {
    double coulomb = 0.0;
    double exchange = 0.0;
};

struct SkippedTerms {
    std::size_t count = 0;
    double max_abs_integral = 0.0;
    double contribution = 0.0;
};

struct AuditResult {
    FockTerms unscreened;
    FockTerms value_screened;
    FockTerms shell_screened;
    SkippedTerms skipped_coulomb;
    SkippedTerms skipped_exchange;
};

std::size_t pair_index(int first, int second) {
    const auto high = static_cast<std::size_t>(std::max(first, second));
    const auto low = static_cast<std::size_t>(std::min(first, second));
    return high * (high + 1) / 2 + low;
}

std::size_t quartet_index(int i, int j, int k, int l, int ni, int nj, int nk) {
    return static_cast<std::size_t>(i) +
           static_cast<std::size_t>(ni) *
               (static_cast<std::size_t>(j) +
                static_cast<std::size_t>(nj) *
                    (static_cast<std::size_t>(k) +
                     static_cast<std::size_t>(nk) * static_cast<std::size_t>(l)));
}

int parse_ao_index(const char* text, int nbf) {
    const std::string argument(text);
    std::size_t consumed = 0;
    const long parsed = std::stol(argument, &consumed);
    if (consumed != argument.size() || parsed < 0 || parsed >= nbf) {
        throw std::invalid_argument("AO index out of range: " + argument);
    }
    return static_cast<int>(parsed);
}

int containing_shell(const xmvb::integrals::LibcintInput& input, int ao) {
    for (int shell = 0; shell < input.n_shells; ++shell) {
        const int offset = xmvb::integrals::shell_ao_offset(input, shell);
        const int count = xmvb::integrals::shell_ao_count(input, shell);
        if (ao >= offset && ao < offset + count) {
            return shell;
        }
    }
    throw std::runtime_error("AO is not covered by any shell");
}

/** @brief Read a native-double, column-major normalized-AO density exactly. */
std::vector<double> read_density(const std::string& path, int nbf) {
    const auto count = static_cast<std::size_t>(nbf) * static_cast<std::size_t>(nbf);
    if (count >
        static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(double)) {
        throw std::overflow_error("density file is too large to read");
    }
    const auto expected_bytes = static_cast<std::streamsize>(count * sizeof(double));
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != expected_bytes) {
        throw std::runtime_error("density file missing or has wrong byte count: " + path);
    }
    input.seekg(0);
    std::vector<double> density(count);
    input.read(reinterpret_cast<char*>(density.data()), expected_bytes);
    if (!input || !std::all_of(density.begin(), density.end(), [](double value) {
            return std::isfinite(value);
        })) {
        throw std::runtime_error("density file is truncated or contains nonfinite values");
    }
    return density;
}

void add_term(
    double contribution, double integral, bool shell_skipped, bool coulomb, AuditResult& result) {
    double& all = coulomb ? result.unscreened.coulomb : result.unscreened.exchange;
    all += contribution;
    if (std::abs(integral) < kIntegralTolerance) {
        return;
    }
    double& value = coulomb ? result.value_screened.coulomb : result.value_screened.exchange;
    value += contribution;
    if (shell_skipped) {
        SkippedTerms& skipped = coulomb ? result.skipped_coulomb : result.skipped_exchange;
        ++skipped.count;
        skipped.max_abs_integral = std::max(skipped.max_abs_integral, std::abs(integral));
        skipped.contribution += contribution;
    } else {
        double& screened = coulomb ? result.shell_screened.coulomb : result.shell_screened.exchange;
        screened += contribution;
    }
}

void print_layer(const char* label, double hcore, const FockTerms& terms) {
    std::cout << label << ".J=" << terms.coulomb << '\n'
              << label << ".K=" << terms.exchange << '\n'
              << label << ".F=" << hcore + terms.coulomb + terms.exchange << '\n';
}

} // namespace

/**
 * @brief Audit one normalized-AO Fock entry using direct, serial shell quartets.
 *
 * No four-index AO tensor or complete VB-SCF input is constructed. Each
 * workspace quartet reference is consumed before the next evaluator call.
 * The unscreened layer disables external shell/value filtering, not libcint's
 * internal primitive screening.
 */
int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: audit_ao_fock_entry INPUT.xmi "
                     "INACTIVE_DENSITY_F64.bin MU NU\n";
        return 2;
    }

    try {
        const auto deck = xmvb::vb::parse_input_deck_model(argv[1]);
        auto basis = xmvb::vb::build_input_deck_basis(deck, deck.metadata.basis_name);
        const auto& input = basis.libcint_input;
        const int nbf = xmvb::integrals::infer_n_basis_functions(input);
        const int mu = parse_ao_index(argv[3], nbf);
        const int nu = parse_ao_index(argv[4], nbf);
        const auto density = read_density(argv[2], nbf);
        const int shell_mu = containing_shell(input, mu);
        const int shell_nu = containing_shell(input, nu);
        const int local_mu = mu - xmvb::integrals::shell_ao_offset(input, shell_mu);
        const int local_nu = nu - xmvb::integrals::shell_ao_offset(input, shell_nu);

        xmvb::libcint::LibcintDirectShellEvaluator evaluator(input);
        xmvb::libcint::LibcintShellWorkspace workspace;
        const auto& normalization = evaluator.ao_normalization();
        const auto [minimum_norm, maximum_norm] =
            std::minmax_element(normalization.begin(), normalization.end());
        const auto h_block = evaluator.evaluate_core_hamiltonian_shell_pair(shell_mu, shell_nu);
        const double hcore =
            h_block.values[static_cast<std::size_t>(local_mu) +
                           static_cast<std::size_t>(h_block.left_ao_count) * local_nu];

        const std::size_t n_shells = static_cast<std::size_t>(input.n_shells);
        std::vector<double> pair_bounds(n_shells * (n_shells + 1) / 2);
        for (int shell_i = 0; shell_i < input.n_shells; ++shell_i) {
            for (int shell_j = 0; shell_j <= shell_i; ++shell_j) {
                pair_bounds[pair_index(shell_i, shell_j)] =
                    evaluator.evaluate_two_electron_shell_pair_bound(shell_i, shell_j, workspace);
            }
        }

        AuditResult result;
        for (int shell_rho = 0; shell_rho < input.n_shells; ++shell_rho) {
            const int rho_offset = xmvb::integrals::shell_ao_offset(input, shell_rho);
            const int n_rho = xmvb::integrals::shell_ao_count(input, shell_rho);
            for (int shell_sigma = 0; shell_sigma < input.n_shells; ++shell_sigma) {
                const int sigma_offset = xmvb::integrals::shell_ao_offset(input, shell_sigma);
                const int n_sigma = xmvb::integrals::shell_ao_count(input, shell_sigma);

                const bool skip_coulomb =
                    std::sqrt(pair_bounds[pair_index(shell_mu, shell_nu)] *
                              pair_bounds[pair_index(shell_rho, shell_sigma)]) < kIntegralTolerance;
                const auto& j_quartet = evaluator.evaluate_two_electron_shell_quartet(
                    shell_mu, shell_nu, shell_rho, shell_sigma, workspace);
                for (int local_sigma = 0; local_sigma < n_sigma; ++local_sigma) {
                    for (int local_rho = 0; local_rho < n_rho; ++local_rho) {
                        const auto density_index =
                            static_cast<std::size_t>(rho_offset + local_rho) +
                            static_cast<std::size_t>(nbf) * (sigma_offset + local_sigma);
                        const double integral =
                            j_quartet.values[quartet_index(local_mu,
                                                           local_nu,
                                                           local_rho,
                                                           local_sigma,
                                                           j_quartet.ao_count_i,
                                                           j_quartet.ao_count_j,
                                                           j_quartet.ao_count_k)];
                        add_term(2.0 * density[density_index] * integral,
                                 integral,
                                 skip_coulomb,
                                 true,
                                 result);
                    }
                }

                const bool skip_exchange =
                    std::sqrt(pair_bounds[pair_index(shell_mu, shell_rho)] *
                              pair_bounds[pair_index(shell_nu, shell_sigma)]) < kIntegralTolerance;
                const auto& k_quartet = evaluator.evaluate_two_electron_shell_quartet(
                    shell_mu, shell_rho, shell_nu, shell_sigma, workspace);
                for (int local_sigma = 0; local_sigma < n_sigma; ++local_sigma) {
                    for (int local_rho = 0; local_rho < n_rho; ++local_rho) {
                        const auto density_index =
                            static_cast<std::size_t>(rho_offset + local_rho) +
                            static_cast<std::size_t>(nbf) * (sigma_offset + local_sigma);
                        const double integral =
                            k_quartet.values[quartet_index(local_mu,
                                                           local_rho,
                                                           local_nu,
                                                           local_sigma,
                                                           k_quartet.ao_count_i,
                                                           k_quartet.ao_count_j,
                                                           k_quartet.ao_count_k)];
                        add_term(-density[density_index] * integral,
                                 integral,
                                 skip_exchange,
                                 false,
                                 result);
                    }
                }
            }
        }

        std::cout << std::setprecision(17) << "execution=serial\n"
                  << "nbf=" << nbf << '\n'
                  << "nshell=" << input.n_shells << '\n'
                  << "mu=" << mu << '\n'
                  << "nu=" << nu << '\n'
                  << "normalization.min=" << *minimum_norm << '\n'
                  << "normalization.max=" << *maximum_norm << '\n'
                  << "hcore=" << hcore << '\n';
        print_layer("unscreened", hcore, result.unscreened);
        print_layer("normabs_1e-10", hcore, result.value_screened);
        print_layer("schwarz_and_normabs_1e-10", hcore, result.shell_screened);
        std::cout << "shell_skip.J.count_normabs_ge_1e-10=" << result.skipped_coulomb.count << '\n'
                  << "shell_skip.J.max_abs_normalized_integral="
                  << result.skipped_coulomb.max_abs_integral << '\n'
                  << "shell_skip.J.contribution_after_normabs="
                  << result.skipped_coulomb.contribution << '\n'
                  << "shell_skip.K.count_normabs_ge_1e-10=" << result.skipped_exchange.count << '\n'
                  << "shell_skip.K.max_abs_normalized_integral="
                  << result.skipped_exchange.max_abs_integral << '\n'
                  << "shell_skip.K.contribution_after_normabs="
                  << result.skipped_exchange.contribution << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "audit_ao_fock_entry: " << error.what() << '\n';
        return 1;
    }
}
