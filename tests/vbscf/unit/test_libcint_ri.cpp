#include <cmath>
#include <algorithm>
#include <atomic>
#include <iostream>
#include <stdexcept>

extern "C" {
#include "cint.h"
}

#include "libcint/ri_provider.hpp"
#include "libcint/direct_shell.hpp"
#include "libcint/shell_normalization.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

/** @brief One normalized s Gaussian; duplicate shells make the metric singular. */
xmvb::integrals::LibcintInput make_input(int n_shells) {
    xmvb::integrals::LibcintInput input;
    input.n_atoms = 1;
    input.n_shells = n_shells;
    input.n_gaussian_primitives = n_shells;
    input.atm.assign(ATM_SLOTS, 0);
    input.atm[CHARGE_OF] = 1;
    input.atm[PTR_COORD] = PTR_ENV_START;
    input.env.assign(PTR_ENV_START + 3, 0.0);
    input.bas.assign(n_shells * BAS_SLOTS, 0);
    input.basidx.resize(2 * n_shells);
    for (int shell = 0; shell < n_shells; ++shell) {
        const int offset = static_cast<int>(input.env.size());
        input.env.push_back(1.0);
        input.env.push_back(xmvb::libcint::shell_normalization(0, 1.0));
        input.bas[shell * BAS_SLOTS + NPRIM_OF] = 1;
        input.bas[shell * BAS_SLOTS + NCTR_OF] = 1;
        input.bas[shell * BAS_SLOTS + PTR_EXP] = offset;
        input.bas[shell * BAS_SLOTS + PTR_COEFF] = offset + 1;
        input.basidx[2 * shell] = shell;
        input.basidx[2 * shell + 1] = 1;
    }
    return input;
}

/** @brief Independent legacy integral reference, including Cartesian d shells. */
void check_shell_workspace() {
    auto input = make_input(3);
    const int counts[] = {1, 3, 6};
    int offset = 0;
    for (int shell = 0; shell < 3; ++shell) {
        input.bas[shell * BAS_SLOTS + ANG_OF] = shell;
        input.env[input.bas[shell * BAS_SLOTS + PTR_COEFF]] =
            xmvb::libcint::shell_normalization(shell, 1.0);
        input.basidx[2 * shell] = offset;
        input.basidx[2 * shell + 1] = counts[shell];
        offset += counts[shell];
    }
    xmvb::libcint::LibcintDirectShellEvaluator evaluator(input);
    const auto& norm = evaluator.ao_normalization();
    std::vector<std::vector<double>> references(81);
    for (int task = 0; task < 81; ++task) {
        FINT shells[] = {task / 27, (task / 9) % 3, (task / 3) % 3, task % 3};
        auto& values = references[task];
        values.resize(counts[shells[0]] * counts[shells[1]] * counts[shells[2]] *
                      counts[shells[3]]);
        cint2e_cart(values.data(),
                    shells,
                    input.atm.data(),
                    input.n_atoms,
                    input.bas.data(),
                    input.n_shells,
                    input.env.data(),
                    nullptr);
        int index = 0;
        for (int l = 0; l < counts[shells[3]]; ++l)
            for (int k = 0; k < counts[shells[2]]; ++k)
                for (int j = 0; j < counts[shells[1]]; ++j)
                    for (int i = 0; i < counts[shells[0]]; ++i, ++index)
                        values[index] *= norm[input.basidx[2 * shells[0]] + i] *
                                         norm[input.basidx[2 * shells[1]] + j] *
                                         norm[input.basidx[2 * shells[2]] + k] *
                                         norm[input.basidx[2 * shells[3]] + l];
    }
    std::atomic<bool> valid(true);
#pragma omp parallel
    {
        xmvb::libcint::LibcintShellWorkspace workspace;
#pragma omp for schedule(dynamic, 1)
        for (int task = 0; task < 81; ++task) {
            const auto& block = evaluator.evaluate_two_electron_shell_quartet(
                task / 27, (task / 9) % 3, (task / 3) % 3, task % 3, workspace);
            for (std::size_t i = 0; i < block.values.size(); ++i)
                if (std::abs(block.values[i] - references[task][i]) > 1e-12)
                    valid.store(false);
        }
    }
    require(valid.load(), "shared libcint optimizer differs from legacy reference");
    xmvb::libcint::LibcintShellWorkspace workspace;
    evaluator.evaluate_two_electron_shell_quartet(2, 2, 2, 2, workspace);
    const double* values = workspace.quartet.values.data();
    const double* cache = workspace.cache.data();
    for (int repeat = 0; repeat < 20; ++repeat) {
        evaluator.evaluate_two_electron_shell_quartet(2, 2, 2, 2, workspace);
        require(values == workspace.quartet.values.data() && cache == workspace.cache.data(),
                "same-size shell evaluation reallocates workspace");
    }
}

} // namespace

int main() {
    try {
        check_shell_workspace();
        constexpr double pi = 3.14159265358979323846;
        using xmvb::libcint::radial_cartesian_scale;
        require(xmvb::libcint::radial_cartesian_scale(0, 0, 0) == 1.0, "s normalization");
        require(xmvb::libcint::radial_cartesian_scale(1, 0, 0) == 1.0, "p normalization");
        require(std::abs(std::pow(xmvb::libcint::radial_cartesian_scale(2, 0, 0), 2) - 4 * pi / 5) <
                    1e-14,
                "Cartesian dxx radial norm");
        require(std::abs(std::pow(xmvb::libcint::radial_cartesian_scale(1, 1, 0), 2) -
                         4 * pi / 15) < 1e-14,
                "Cartesian dxy radial norm");
        require(std::abs(std::pow(xmvb::libcint::radial_cartesian_scale(3, 0, 0), 2) - 4 * pi / 7) <
                    1e-14,
                "Cartesian fxxx radial norm");

        const auto primary = make_input(1);
        xmvb::libcint::LibcintRiIntegralProvider provider;
        const auto positive = provider.build(primary, make_input(1), {});
        const auto singular = provider.build(primary, make_input(3), {});
        require(positive.n_auxiliary_functions == 1, "positive metric rank");
        require(singular.n_auxiliary_functions == 1, "duplicate auxiliary shells must be removed");
        require(singular.auxiliary_metric_matrix.rows() == 3, "original metric size");
        require(singular.metric_whitened_ao_pair_factors.rows() == 1, "rectangular factor rank");
        const double a = positive.metric_whitened_ao_pair_factors.squaredNorm();
        const double b = singular.metric_whitened_ao_pair_factors.squaredNorm();
        require(std::abs(a - b) < 1e-12, "Cholesky and spectral RI contractions differ");
        std::cout << "RI metric normalization and rank regression passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
