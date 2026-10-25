#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "cuda/vb/action.hpp"
#include "cuda/vb/metric.hpp"
#include "integrals/active/two_electron.hpp"
#include "mcscf/ci/strings.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/structures/assembly/action.hpp"

namespace {

void check_full_constructor_residency() {
    constexpr int orbitals = 2;
    const auto strings = xmvb::mcscf::SpinStrings(orbitals, 1).decode();
    const double mixed = 1.0 / std::sqrt(2.0);
    std::vector<std::vector<xmvb::vb::StructureExpansionTerm>> terms(4);
    terms[0].push_back({0, 1.0});
    terms[1].push_back({1, mixed});
    terms[2].push_back({1, mixed});
    terms[3].push_back({2, 1.0});
    const std::vector<std::vector<int>> product_alpha{
        strings[0], strings[0], strings[1], strings[1]};
    const std::vector<std::vector<int>> product_beta{
        strings[0], strings[1], strings[0], strings[1]};
    const auto topology =
        xmvb::vb::build_same_spin_pair_topology(product_alpha, product_beta, orbitals);
    const Eigen::MatrixXd upper = Eigen::MatrixXd::Identity(orbitals, orbitals);
    const Eigen::MatrixXd overlap = upper.transpose() * upper;
    const std::vector<double> overlap_data(overlap.data(), overlap.data() + overlap.size());
    xmvb::integrals::ActiveSpaceTwoElectronResult eri;
    eri.packed_active_two_electron_integrals.assign(
        xmvb::integrals::packed_active_two_electron_integral_count(orbitals), 0.0);
    const xmvb::vb::StructureAction source(terms,
                                           3,
                                           topology,
                                           overlap_data,
                                           Eigen::MatrixXd::Zero(orbitals, orbitals),
                                           eri,
                                           orbitals,
                                           nullptr,
                                           xmvb::vb::StructureActionMode::ActionOnly);
    if (!source.storage().orthogonal_direct_ci)
        throw std::runtime_error("FULL metric fixture lacks direct CI");

    xmvb::cuda::Context context(128ULL * 1024ULL * 1024ULL);
    const xmvb::cuda::SpinMetric spin(context, source, strings, strings, 3 * sizeof(double));
    const xmvb::cuda::StructureAction action(context, source);
    const auto before = context.counters().current_device_bytes;
    {
        const xmvb::cuda::FullMetric full(context, spin, action);
        const auto increase = context.counters().current_device_bytes - before;
        if (increase != 3 * sizeof(double))
            throw std::runtime_error("FULL metric constructor retained a complete product carrier");
    }
    if (context.counters().current_device_bytes != before)
        throw std::runtime_error("FULL metric did not release its structure workspace");

    xmvb::cuda::Matrix<double> vector(context, 3, 1), hamiltonian(context, 3, 1),
        metric(context, 3, 1);
    Eigen::MatrixXd seed = Eigen::MatrixXd::Zero(3, 1);
    seed(0, 0) = 1.0;
    xmvb::cuda::copy_to_device(context, vector.view(), seed);
    action.apply(vector.view(), hamiltonian.view(), metric.view());
    Eigen::MatrixXd first_h(3, 1), first_m(3, 1), second_h(3, 1), second_m(3, 1);
    xmvb::cuda::copy_to_host(context, first_h, hamiltonian.view());
    xmvb::cuda::copy_to_host(context, first_m, metric.view());
    const auto available = context.available_device_bytes();
    action.reserve_workspace(available);
    action.reserve_workspace(available + 1);
    action.apply(vector.view(), hamiltonian.view(), metric.view());
    xmvb::cuda::copy_to_host(context, second_h, hamiltonian.view());
    xmvb::cuda::copy_to_host(context, second_m, metric.view());
    if (!first_h.isApprox(second_h, 0.0) || !first_m.isApprox(second_m, 0.0))
        throw std::runtime_error("FULL pressure replay changed the numerical action");
}

} // namespace

int main() {
    try {
        check_full_constructor_residency();
        std::cout << "CUDA FULL metric constructor residency: PASS\n";
    } catch (const std::exception& error) {
        std::cerr << "CUDA FULL metric constructor residency: " << error.what() << '\n';
        return 1;
    }
}
