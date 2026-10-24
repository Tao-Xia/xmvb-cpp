#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "cuda/memory/buffer.hpp"
#include "cuda/vb/action.hpp"
#include "integrals/active/two_electron.hpp"
#include "mcscf/ci/strings.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/structures/assembly/action.hpp"

namespace {

using Matrix = Eigen::MatrixXd;
using Strings = std::vector<std::vector<int>>;
using Terms = std::vector<std::vector<xmvb::vb::StructureExpansionTerm>>;

void near(const Matrix& actual, const Matrix& expected, const char* message) {
    if (actual.rows() != expected.rows() || actual.cols() != expected.cols() ||
        !actual.allFinite() || !expected.allFinite() ||
        (actual - expected).norm() > 2.0e-10 * std::max(1.0, expected.norm()))
        throw std::runtime_error(message);
}

Matrix upper_frame(bool changed) {
    Matrix upper = Matrix::Identity(4, 4);
    for (int col = 0; col < 4; ++col) {
        upper(col, col) = (changed ? 0.91 : 1.04) + 0.055 * col;
        for (int row = 0; row < col; ++row)
            upper(row, col) = (changed ? -0.022 : 0.035) * (row + 1) * (col + 1);
    }
    return upper;
}

Terms selected_terms(int alpha, int beta) {
    Terms terms(alpha * beta);
    for (int a = 0; a < alpha; ++a)
        for (int b = 0; b < beta; ++b) {
            const int product = a * beta + b;
            if ((a + 2 * b) % 3 == 0)
                terms[product].push_back({(a + b) % 4, 0.41 + 0.023 * a});
            if ((2 * a + b) % 4 == 1)
                terms[product].push_back({(a + 2 * b + 1) % 4, -0.18 + 0.014 * b});
        }
    // Duplicate structure incidence in one touched product must be summed,
    // not overwritten or counted only once in the right W tile.
    terms[0].push_back({0, 0.37});
    terms[0].push_back({0, -0.09});
    terms[alpha * beta - 1].push_back({3, 0.62});
    return terms;
}

xmvb::vb::StructureAction source_action(int alpha_electrons, int beta_electrons, bool changed) {
    constexpr int orbitals = 4;
    const Strings alpha = xmvb::mcscf::SpinStrings(orbitals, alpha_electrons).decode();
    const Strings beta = xmvb::mcscf::SpinStrings(orbitals, beta_electrons).decode();
    Strings product_alpha, product_beta;
    for (const auto& a : alpha)
        for (const auto& b : beta) {
            product_alpha.push_back(a);
            product_beta.push_back(b);
        }
    const auto topology =
        xmvb::vb::build_same_spin_pair_topology(product_alpha, product_beta, orbitals);
    const Matrix upper = upper_frame(changed);
    const Matrix overlap = upper.transpose() * upper;
    const std::vector<double> overlap_data(overlap.data(), overlap.data() + overlap.size());
    xmvb::integrals::ActiveSpaceTwoElectronResult eri;
    eri.packed_active_two_electron_integrals.assign(
        xmvb::integrals::packed_active_two_electron_integral_count(orbitals), 0.0);
    return xmvb::vb::StructureAction(selected_terms(alpha.size(), beta.size()),
                                     4,
                                     topology,
                                     overlap_data,
                                     Matrix::Zero(orbitals, orbitals),
                                     eri,
                                     orbitals,
                                     nullptr,
                                     xmvb::vb::StructureActionMode::ActionOnly);
}

void check_case(int alpha_electrons, int beta_electrons) {
    constexpr int structures = 4;
    const auto accepted = source_action(alpha_electrons, beta_electrons, false);
    xmvb::cuda::Context context(128ULL * 1024ULL * 1024ULL);
    xmvb::cuda::StructureAction action(context, accepted);
    xmvb::cuda::Matrix<double> metric(context, structures, structures);
    action.build_selected_metric(metric.view());
    Matrix actual(structures, structures);
    xmvb::cuda::copy_to_host(context, actual, metric.view());
    const Matrix identity = Matrix::Identity(structures, structures);
    near(actual,
         accepted.apply_overlap(identity),
         "CUDA selected spin-Gram metric differs from CPU exact overlap");
    near(actual, actual.transpose(), "CUDA selected spin-Gram metric is not symmetric");

    const auto changed = source_action(alpha_electrons, beta_electrons, true);
    const Matrix upper = upper_frame(true);
    xmvb::cuda::Matrix<double> d_upper(context, upper.rows(), upper.cols());
    xmvb::cuda::copy_to_device(context, d_upper.view(), upper);
    action.update_frame(d_upper.view());
    action.build_selected_metric(metric.view());
    xmvb::cuda::copy_to_host(context, actual, metric.view());
    near(actual,
         changed.apply_overlap(identity),
         "CUDA selected spin-Gram metric ignored the current frame");
    {
        const auto groups = accepted.orthogonal_view().products.size();
        const auto per_column =
            sizeof(double) * (static_cast<std::size_t>(action.n_alpha_strings()) +
                              action.n_beta_strings() + groups + 2 * structures);
        const auto remaining =
            context.device_byte_budget() - context.counters().current_device_bytes;
        if (remaining < 2 * per_column)
            throw std::runtime_error("CUDA selected metric fixture lacks a two-column allowance");
        // Force at most two source columns. The existing one-RHS incidence SpMM
        // workspace guarantees that a one-column tile remains feasible.
        xmvb::cuda::Buffer<unsigned char> padding(context, remaining - 2 * per_column);
        action.build_selected_metric(metric.view());
        xmvb::cuda::copy_to_host(context, actual, metric.view());
        near(actual,
             changed.apply_overlap(identity),
             "CUDA budgeted selected spin-Gram tile differs from exact overlap");
    }
    if (context.counters().peak_device_bytes > context.device_byte_budget())
        throw std::runtime_error("CUDA selected spin-Gram metric exceeded its device budget");
}

} // namespace

int main() {
    try {
        check_case(2, 1); // six alpha strings, four beta strings
        check_case(1, 1); // shared alpha/beta exterior topology
        std::cout << "CUDA selected single-spin Gram metric: PASS\n";
    } catch (const std::exception& error) {
        std::cerr << "CUDA selected single-spin Gram metric: " << error.what() << '\n';
        return 1;
    }
}
