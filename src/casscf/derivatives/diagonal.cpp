#include "casscf/derivatives/diagonal.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "casscf/derivatives/point.hpp"
#include "casscf/orbitals/rotation.hpp"
#include "core/pair_index.hpp"
#include "integrals/ao/exact.hpp"
#include "mcscf/optimization/preconditioners/orbital_diagonal.hpp"
#include "parallel/openmp.hpp"

namespace xmvb::casscf {
namespace {

std::vector<double> pack_pair_adjoint(const Eigen::MatrixXd& adjoint) {
    const int pairs = adjoint.rows();
    if (adjoint.cols() != pairs)
        throw std::invalid_argument("diagonal pair adjoint shape mismatch");
    std::vector<double> packed(static_cast<std::size_t>(pairs) * (pairs + 1) / 2);
    for (int p = 0; p < pairs; ++p)
        for (int q = 0; q <= p; ++q)
            packed[core::PairIndex::packed_pair_index(p, q)] =
                p == q ? adjoint(p, q) : adjoint(p, q) + adjoint(q, p);
    return packed;
}

} // namespace

Eigen::VectorXd orbital_hessian_diagonal(const FixedCiPoint& point,
                                         const RotationChart& chart,
                                         const integrals::AoIntegralInput& ao,
                                         int ncore,
                                         int nactive) {
    const Eigen::MatrixXd& u = point.orbitals();
    const int n = ao.n_basis_functions;
    if (n < 1 || u.rows() != n || u.cols() != n || chart.nmo() != n || ncore < 0 || nactive < 1 ||
        ncore + nactive > n || ao.ao_core_hamiltonian_matrix.rows() != n ||
        ao.ao_core_hamiltonian_matrix.cols() != n ||
        point.integrals().active.one_electron.rows() != nactive)
        throw std::invalid_argument("analytic CAS orbital diagonal dimension mismatch");
    mcscf::AnalyticDiagonalIntermediates x;
    x.n_core = ncore;
    x.n_active = nactive;
    x.h_core = u.transpose() * ao.ao_core_hamiltonian_matrix * u;
    x.v_core = u.transpose() * point.core_fock_ao() * u - x.h_core;

    const int active_pairs = nactive * (nactive + 1) / 2;
    const int channels = ncore + active_pairs;
    constexpr std::size_t source_budget = 64ULL * 1024 * 1024;
    const std::size_t source_elements = static_cast<std::size_t>(n) * n;
    const int nthreads = std::min<int>(effective_openmp_thread_count(), n);
    const std::size_t elements_per_channel = source_elements +
                                             6 * static_cast<std::size_t>(n) * nthreads +
                                             2 * static_cast<std::size_t>(n);
    if (elements_per_channel > source_budget / sizeof(double))
        throw std::length_error("analytic diagonal AO source exceeds workspace budget");
    const int batch_width = std::max<std::size_t>(
        1, std::min<std::size_t>(channels, source_budget / sizeof(double) / elements_per_channel));
    x.j_pc.resize(n, ncore);
    x.k_pc.resize(n, ncore);
    x.j_active.resize(static_cast<std::size_t>(n) * nactive * nactive);
    x.k_active.resize(x.j_active.size());
    integrals::AoDensityDiagonalWorkspace workspace;
    for (int begin = 0; begin < channels; begin += batch_width) {
        const int width = std::min(batch_width, channels - begin);
        integrals::AoH1eBatchWorkspace::Channels sources(n * n, width);
        for (int local = 0; local < width; ++local) {
            const int channel = begin + local;
            int a = 0, b = 0;
            if (channel >= ncore) {
                const int pair = channel - ncore;
                for (a = 0; a < nactive; ++a)
                    if (pair < (a + 1) * (a + 2) / 2)
                        break;
                b = pair - a * (a + 1) / 2;
                a += ncore;
                b += ncore;
            } else {
                a = b = channel;
            }
            for (int nu = 0; nu < n; ++nu)
                for (int mu = 0; mu < n; ++mu)
                    sources(mu + n * nu, local) = 0.5 * (u(mu, a) * u(nu, b) + u(mu, b) * u(nu, a));
        }
        Eigen::MatrixXd exchange, coulomb;
        integrals::apply_ao_density_diagonals(
            ao, sources, u, width, nthreads, &workspace, &exchange, &coulomb);
        for (int local = 0; local < width; ++local) {
            const int channel = begin + local;
            if (channel < ncore) {
                x.j_pc.col(channel) = coulomb.col(local);
                x.k_pc.col(channel) = exchange.col(local);
            } else {
                const int pair = channel - ncore;
                int a = 0;
                for (; a < nactive; ++a)
                    if (pair < (a + 1) * (a + 2) / 2)
                        break;
                const int b = pair - a * (a + 1) / 2;
                for (int p = 0; p < n; ++p) {
                    x.j_active[mcscf::orbital_diagonal_slice_index(p, a, b, nactive)] =
                        coulomb(p, local);
                    x.j_active[mcscf::orbital_diagonal_slice_index(p, b, a, nactive)] =
                        coulomb(p, local);
                    x.k_active[mcscf::orbital_diagonal_slice_index(p, a, b, nactive)] =
                        exchange(p, local);
                    x.k_active[mcscf::orbital_diagonal_slice_index(p, b, a, nactive)] =
                        exchange(p, local);
                }
            }
        }
    }
    const auto& pair = point.integrals().active.pair_kernel;
    x.active_eri.resize(static_cast<std::size_t>(nactive) * nactive * nactive * nactive);
    for (int a = 0; a < nactive; ++a)
        for (int b = 0; b < nactive; ++b)
            for (int c = 0; c < nactive; ++c)
                for (int d = 0; d < nactive; ++d)
                    x.active_eri[mcscf::orbital_diagonal_tensor4_index(a, b, c, d, nactive)] =
                        pair(core::PairIndex::packed_pair_index(a, b),
                             core::PairIndex::packed_pair_index(c, d));
    const Eigen::MatrixXd diagonal = mcscf::build_analytic_rotation_hessian_diagonal(
        x,
        point.integral_adjoint().one_electron,
        pack_pair_adjoint(point.integral_adjoint().pair_kernel));
    Eigen::VectorXd selected(chart.size());
    for (int k = 0; k < chart.size(); ++k) {
        const auto [i, j] = chart.pairs()[k];
        selected[k] = diagonal(i, j);
    }
    return selected;
}

} // namespace xmvb::casscf
