#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "core/pair_index.hpp"
#include "vbscf/optimization/preconditioners/hessian_diagonal.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "integrals/ao/graph.hpp"
#include "integrals/ao/ri.hpp"
#include "integrals/ao/exact.hpp"

namespace {

std::size_t eri_index(int p, int q, int r, int s, int n) {
    return (((static_cast<std::size_t>(p) * n + q) * n + r) * n + s);
}

double eri(const std::vector<double>& values, int p, int q, int r, int s, int n) {
    return values[eri_index(p, q, r, s, n)];
}

std::vector<double> make_symmetric_eri(int n) {
    std::vector<double> values(static_cast<std::size_t>(n) * n * n * n, 0.0);
    for (int p = 0; p < n; ++p) {
        for (int q = 0; q <= p; ++q) {
            const int pq = p * (p + 1) / 2 + q;
            for (int r = 0; r < n; ++r) {
                for (int s = 0; s <= r; ++s) {
                    const int rs = r * (r + 1) / 2 + s;
                    if (rs > pq)
                        continue;
                    const double value = 0.013 * (1 + pq) - 0.007 * (1 + rs) + 0.0003 * pq * rs;
                    const int permutations[8][4] = {{p, q, r, s},
                                                    {q, p, r, s},
                                                    {p, q, s, r},
                                                    {q, p, s, r},
                                                    {r, s, p, q},
                                                    {s, r, p, q},
                                                    {r, s, q, p},
                                                    {s, r, q, p}};
                    for (const auto& x : permutations) {
                        values[eri_index(x[0], x[1], x[2], x[3], n)] = value;
                    }
                }
            }
        }
    }
    return values;
}

std::vector<double> rotate_eri(const std::vector<double>& values,
                               const Eigen::Ref<const Eigen::MatrixXd>& u) {
    const int n = static_cast<int>(u.rows());
    std::vector<double> rotated(static_cast<std::size_t>(n) * n * n * n, 0.0);
    for (int p = 0; p < n; ++p)
        for (int q = 0; q < n; ++q)
            for (int r = 0; r < n; ++r)
                for (int s = 0; s < n; ++s)
                    for (int a = 0; a < n; ++a)
                        for (int b = 0; b < n; ++b)
                            for (int c = 0; c < n; ++c)
                                for (int d = 0; d < n; ++d)
                                    rotated[eri_index(p, q, r, s, n)] += u(a, p) * u(b, q) *
                                                                         u(c, r) * u(d, s) *
                                                                         eri(values, a, b, c, d, n);
    return rotated;
}

double energy(const Eigen::Ref<const Eigen::MatrixXd>& h,
              const std::vector<double>& g,
              const Eigen::Ref<const Eigen::MatrixXd>& dm1,
              const std::vector<double>& dm2,
              int n_core) {
    const int n = static_cast<int>(h.rows());
    const int na = static_cast<int>(dm1.rows());
    double value = 0.0;
    for (int c = 0; c < n_core; ++c)
        value += 2.0 * h(c, c);
    for (int a = 0; a < na; ++a)
        for (int b = 0; b < na; ++b)
            value += h(n_core + a, n_core + b) * dm1(a, b);
    for (int c = 0; c < n_core; ++c)
        for (int d = 0; d < n_core; ++d)
            value += 2.0 * eri(g, c, c, d, d, n) - eri(g, c, d, d, c, n);
    for (int c = 0; c < n_core; ++c)
        for (int a = 0; a < na; ++a)
            for (int b = 0; b < na; ++b)
                value += dm1(a, b) * (2.0 * eri(g, c, c, n_core + a, n_core + b, n) -
                                      eri(g, c, n_core + a, n_core + b, c, n));
    for (int a = 0; a < na; ++a)
        for (int b = 0; b < na; ++b)
            for (int c = 0; c < na; ++c)
                for (int d = 0; d < na; ++d)
                    value += 0.5 * dm2[eri_index(a, b, c, d, na)] *
                             eri(g, n_core + a, n_core + b, n_core + c, n_core + d, n);
    return value;
}

xmvb::mcscf::AnalyticDiagonalIntermediates make_intermediates(
    const Eigen::Ref<const Eigen::MatrixXd>& h, const std::vector<double>& g, int n_core, int na) {
    const int n = static_cast<int>(h.rows());
    xmvb::mcscf::AnalyticDiagonalIntermediates x;
    x.n_core = n_core;
    x.n_active = na;
    x.h_core = h;
    x.v_core = Eigen::MatrixXd::Zero(n, n);
    x.j_pc = Eigen::MatrixXd::Zero(n, n_core);
    x.k_pc = Eigen::MatrixXd::Zero(n, n_core);
    x.j_active.resize(static_cast<std::size_t>(n) * na * na);
    x.k_active.resize(x.j_active.size());
    x.active_eri.resize(static_cast<std::size_t>(na) * na * na * na);
    for (int a = 0; a < na; ++a)
        for (int b = 0; b < na; ++b)
            for (int c = 0; c < na; ++c)
                for (int d = 0; d < na; ++d)
                    x.active_eri[eri_index(a, b, c, d, na)] =
                        eri(g, n_core + a, n_core + b, n_core + c, n_core + d, n);
    for (int p = 0; p < n; ++p) {
        for (int q = 0; q < n; ++q) {
            for (int c = 0; c < n_core; ++c) {
                x.v_core(p, q) += 2.0 * eri(g, p, q, c, c, n) - eri(g, p, c, c, q, n);
            }
        }
        for (int a = 0; a < na; ++a)
            for (int b = 0; b < na; ++b) {
                x.j_active[(p * na + a) * na + b] = eri(g, p, p, n_core + a, n_core + b, n);
                x.k_active[(p * na + a) * na + b] = eri(g, p, n_core + a, p, n_core + b, n);
            }
        for (int c = 0; c < n_core; ++c) {
            x.j_pc(p, c) = eri(g, p, p, c, c, n);
            x.k_pc(p, c) = eri(g, p, c, p, c, n);
        }
    }
    return x;
}

/** @brief Exercise fewer density channels, including dependent source columns. */
void check_contracted_density_span() {
    constexpr int n = 7;
    constexpr int na = 5;
    const auto g = make_symmetric_eri(n);
    const Eigen::MatrixXd h = Eigen::MatrixXd::Identity(n, n);
    xmvb::vb::VbScfInput input;
    input.standard_two_electron_mode = xmvb::vb::StandardTwoElectronMode::Exact;
    input.orbital_preparation_input.n_basis_functions = n;
    input.orbital_preparation_input.ao_overlap_matrix = Eigen::MatrixXd::Identity(n, n);
    input.ao_integral_input.n_basis_functions = n;
    input.ao_integral_input.ao_core_hamiltonian_matrix = h;
    std::vector<xmvb::integrals::PairAddress> left, right;
    std::vector<double> values;
    for (int p = 0; p < n; ++p)
        for (int q = 0; q <= p; ++q)
            for (int r = 0; r < n; ++r)
                for (int s = 0; s <= r; ++s) {
                    const int pq = p * (p + 1) / 2 + q, rs = r * (r + 1) / 2 + s;
                    if (pq < rs)
                        continue;
                    left.push_back(pq);
                    right.push_back(rs);
                    values.push_back(eri(g, p, q, r, s, n));
                }
    input.ao_integral_input.pair_graph =
        xmvb::integrals::build_ao_pair_graph(left, right, values, n);
    Eigen::MatrixXd frame = Eigen::MatrixXd::Identity(n, n);
    for (int p = 1; p < n; ++p) {
        Eigen::MatrixXd rotation = Eigen::MatrixXd::Identity(n, n);
        const double angle = 0.11 * p;
        rotation(0, 0) = rotation(p, p) = std::cos(angle);
        rotation(0, p) = -std::sin(angle);
        rotation(p, 0) = std::sin(angle);
        frame = (frame * rotation).eval();
    }
    // Independent four-index reference, including diagonal/repeated AO indices.
    for (int width : {1, 2, 3, 8, 16, 52}) {
        Eigen::MatrixXd sources(n * n, width);
        for (int channel = 0; channel < width; ++channel)
            for (int mu = 0; mu < n; ++mu)
                for (int nu = 0; nu < n; ++nu)
                    sources(mu + nu * n, channel) = std::cos(0.13 * (mu + nu + 1) * (channel + 1));
        for (int threads : {1, 4}) {
            xmvb::integrals::AoH1eBatchWorkspace workspace;
            Eigen::MatrixXd result;
            workspace.sources = sources;
            xmvb::integrals::apply_symmetric_ao_h1e_channels(
                input.ao_integral_input, threads, &workspace, &result);
            xmvb::integrals::AoH1eBatchWorkspace::Channels row_sources(n * n, width);
            for (int row = 0; row < n * n; ++row)
                for (int channel = 0; channel < width; ++channel)
                    row_sources(row, channel) = sources(row, channel);
            xmvb::integrals::AoH1eBatchWorkspace::Channels row_result;
            xmvb::integrals::apply_symmetric_ao_h1e_rows(
                input.ao_integral_input, threads, row_sources, &row_result);
            for (int channel = 0; channel < width; ++channel)
                for (int mu = 0; mu < n; ++mu)
                    for (int nu = 0; nu < n; ++nu) {
                        double expected = 0.0;
                        for (int k = 0; k < n; ++k)
                            for (int l = 0; l < n; ++l)
                                expected +=
                                    (2.0 * eri(g, mu, nu, k, l, n) - eri(g, mu, k, nu, l, n)) *
                                    sources(k + l * n, channel);
                        if (std::abs(result(mu + nu * n, channel) - expected) > 5.0e-12)
                            throw std::runtime_error(
                                "packed symmetric density batch changed the AO operator");
                        if (std::abs(row_result(mu + nu * n, channel) - expected) > 5.0e-12)
                            throw std::runtime_error(
                                "row-owned symmetric density batch changed the AO operator");
                    }
            if ((row_result - result).cwiseAbs().maxCoeff() > 5.0e-12)
                throw std::runtime_error("row-owned and packed AO contractions disagree");
            for (int n_core_sources : {0, width / 2, width}) {
                for (int n_mo : {n, n - 2}) {
                    xmvb::integrals::AoDensityDiagonalWorkspace diagonal_workspace;
                    Eigen::MatrixXd exchange_diagonal, coulomb_diagonal;
                    xmvb::integrals::apply_ao_density_diagonals(input.ao_integral_input,
                                                                row_sources,
                                                                frame.leftCols(n_mo),
                                                                n_core_sources,
                                                                threads,
                                                                &diagonal_workspace,
                                                                &exchange_diagonal,
                                                                &coulomb_diagonal);
                    if (exchange_diagonal.rows() != n_mo || exchange_diagonal.cols() != width ||
                        coulomb_diagonal.rows() != n_mo ||
                        coulomb_diagonal.cols() != n_core_sources)
                        throw std::runtime_error("projected density diagonal dimensions changed");
                    const int columns = width + n_core_sources;
                    for (std::size_t worker = 0; worker < diagonal_workspace.rows.size(); ++worker)
                        if (diagonal_workspace.rows[worker].rows() != n ||
                            diagonal_workspace.rows[worker].cols() != columns ||
                            diagonal_workspace.images[worker].rows() != n_mo ||
                            diagonal_workspace.images[worker].cols() != columns ||
                            diagonal_workspace.partial[worker].rows() != n_mo ||
                            diagonal_workspace.partial[worker].cols() != columns)
                            throw std::runtime_error(
                                "projected density retained a worker AO response matrix");
                    for (int channel = 0; channel < width; ++channel)
                        for (int orbital = 0; orbital < n_mo; ++orbital) {
                            double expected_k = 0.0, expected_j = 0.0;
                            for (int mu = 0; mu < n; ++mu)
                                for (int nu = 0; nu < n; ++nu) {
                                    double k_mu_nu = 0.0, j_mu_nu = 0.0;
                                    for (int k = 0; k < n; ++k)
                                        for (int l = 0; l < n; ++l) {
                                            const double density = sources(k + l * n, channel);
                                            k_mu_nu += eri(g, mu, k, nu, l, n) * density;
                                            if (channel < n_core_sources)
                                                j_mu_nu += eri(g, mu, nu, k, l, n) * density;
                                        }
                                    expected_k += frame(mu, orbital) * k_mu_nu * frame(nu, orbital);
                                    if (channel < n_core_sources)
                                        expected_j +=
                                            frame(mu, orbital) * j_mu_nu * frame(nu, orbital);
                                }
                            if (std::abs(exchange_diagonal(orbital, channel) - expected_k) >
                                5.0e-12)
                                throw std::runtime_error("projected positive exchange diagonal "
                                                         "disagrees with four-index reference");
                            if (channel < n_core_sources &&
                                std::abs(coulomb_diagonal(orbital, channel) - expected_j) > 5.0e-12)
                                throw std::runtime_error("projected Coulomb diagonal disagrees "
                                                         "with four-index reference");
                        }
                }
            }
            for (const auto& partial : workspace.partial)
                if (partial.rows() != n * (n + 1) / 2)
                    throw std::runtime_error(
                        "symmetric density batch retained a full worker matrix");
            Eigen::MatrixXd full;
            xmvb::integrals::apply_symmetric_ao_h1e_batch(
                sources, input.ao_integral_input, threads, &workspace, &full);
            if ((full - result).cwiseAbs().maxCoeff() > 5.0e-12)
                throw std::runtime_error("full and packed AO channel contractions disagree");
        }
    }
    // An empty CSR graph is the sparse zero-integral limit of the same paths.
    auto zero_ao = input.ao_integral_input;
    zero_ao.pair_graph = xmvb::integrals::build_ao_pair_graph({}, {}, {}, n);
    xmvb::integrals::AoH1eBatchWorkspace::Channels zero_sources(n * n, 2);
    zero_sources.setOnes();
    for (int threads : {1, 4}) {
        xmvb::integrals::AoH1eBatchWorkspace::Channels zero_response;
        xmvb::integrals::apply_symmetric_ao_h1e_rows(
            zero_ao, threads, zero_sources, &zero_response);
        if (zero_response.cwiseAbs().maxCoeff() != 0.0)
            throw std::runtime_error("empty CSR graph produced a nonzero AO response");
        xmvb::integrals::AoDensityDiagonalWorkspace zero_workspace;
        Eigen::MatrixXd zero_k, zero_j;
        xmvb::integrals::apply_ao_density_diagonals(
            zero_ao, zero_sources, frame, 1, threads, &zero_workspace, &zero_k, &zero_j);
        if (zero_k.cwiseAbs().maxCoeff() != 0.0 || zero_j.cwiseAbs().maxCoeff() != 0.0)
            throw std::runtime_error("empty CSR graph produced nonzero projected diagonals");
    }
    auto ri = std::make_shared<xmvb::integrals::RiAoFactorization>();
    ri->n_basis_functions = n;
    ri->n_auxiliary_functions = 4;
    ri->n_packed_ao_pairs = n * (n + 1) / 2;
    ri->metric_whitened_ao_pair_factors.resize(4, ri->n_packed_ao_pairs);
    for (int auxiliary = 0; auxiliary < 4; ++auxiliary)
        for (int pair = 0; pair < ri->n_packed_ao_pairs; ++pair)
            ri->metric_whitened_ao_pair_factors(auxiliary, pair) =
                0.13 * std::cos(0.29 * (auxiliary + 1) * (pair + 1));
    std::vector<double> ri_g(g.size());
    for (int p = 0; p < n; ++p)
        for (int q = 0; q < n; ++q)
            for (int r = 0; r < n; ++r)
                for (int s = 0; s < n; ++s)
                    ri_g[eri_index(p, q, r, s, n)] =
                        ri->metric_whitened_ao_pair_factors
                            .col(xmvb::core::PairIndex::packed_pair_index(p, q))
                            .dot(ri->metric_whitened_ao_pair_factors.col(
                                xmvb::core::PairIndex::packed_pair_index(r, s)));
    input.ri_factorization = ri;
    for (const bool use_ri : {false, true}) {
        input.standard_two_electron_mode =
            use_ri ? xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity
                   : xmvb::vb::StandardTwoElectronMode::Exact;
        for (const int nc : {0, 1, 2}) {
            xmvb::vb::AcceptedPointContext accepted;
            accepted.n_active_orbitals = na;
            auto& prepared = accepted.prepared_active_space;
            prepared.n_inactive_doubly_occupied_orbitals = nc;
            auto& orbitals = prepared.orbital_result;
            orbitals.auxiliary_orbital_matrix = Eigen::MatrixXd::Identity(n, n);
            orbitals.active_orbital_overlap_matrix.assign(na * na, 0.0);
            for (int a = 0; a < na; ++a)
                orbitals.active_orbital_overlap_matrix[a * na + a] = 1.0;
            orbitals.physical_orbital_frame.inactive_orthonormal_orbital_matrix =
                orbitals.auxiliary_orbital_matrix.leftCols(nc);
            orbitals.physical_orbital_frame.localized_representative_selector
                .inactive_inverse_transpose_right_transform = Eigen::MatrixXd::Identity(nc, nc);
            const auto reference = make_intermediates(h, use_ri ? ri_g : g, nc, na);
            prepared.ao_effective_one_electron_result.ao_coulomb_exchange_matrix = reference.v_core;
            for (const double scale : {0.0, 1.0}) {
                Eigen::MatrixXd dm1(na, na);
                for (int a = 0; a < na; ++a)
                    for (int b = 0; b < na; ++b)
                        dm1(a, b) = scale * (0.2 * std::cos(a + b) + (a == b ? 0.9 : 0.0));
                accepted.active_one_electron_gradient.assign(dm1.data(), dm1.data() + dm1.size());
                const int pairs = na * (na + 1) / 2;
                accepted.packed_active_two_electron_gradient.resize(pairs * (pairs + 1) / 2);
                for (std::size_t j = 0; j < accepted.packed_active_two_electron_gradient.size();
                     ++j)
                    accepted.packed_active_two_electron_gradient[j] =
                        scale * 0.03 * std::sin(0.19 * (j + 1));
                const auto expected = xmvb::mcscf::build_analytic_rotation_hessian_diagonal(
                    reference, dm1, accepted.packed_active_two_electron_gradient);
                const auto actual = xmvb::vb::build_analytic_orbital_diagonal(input, accepted);
                const double error = (actual.rotation_diagonal - expected).cwiseAbs().maxCoeff();
                if (error > 5.0e-12)
                    throw std::runtime_error("contracted density span changed analytic diagonal");
            }
        }
    }
}

} // namespace

int main() {
    check_contracted_density_span();
    constexpr int n = 5;
    constexpr int nc = 1;
    constexpr int na = 2;
    Eigen::MatrixXd h(n, n);
    for (int p = 0; p < n; ++p)
        for (int q = 0; q <= p; ++q)
            h(p, q) = h(q, p) = 0.03 * (p + 1) - 0.017 * (q + 1) + (p == q ? 0.4 : 0.0);
    const std::vector<double> g = make_symmetric_eri(n);
    Eigen::MatrixXd dm1(na, na);
    dm1 << 1.15, -0.21, -0.21, 0.85;
    std::vector<double> dm2 = make_symmetric_eri(na);
    for (double& value : dm2)
        value *= 2.7;

    const int n_pairs = na * (na + 1) / 2;
    std::vector<double> packed(static_cast<std::size_t>(n_pairs) * (n_pairs + 1) / 2);
    for (int p = 0; p < na; ++p)
        for (int q = 0; q <= p; ++q) {
            const int pq = xmvb::core::PairIndex::packed_pair_index(p, q);
            for (int r = 0; r < na; ++r)
                for (int s = 0; s <= r; ++s) {
                    const int rs = xmvb::core::PairIndex::packed_pair_index(r, s);
                    if (rs > pq)
                        continue;
                    const int multiplicity =
                        (p == q ? 1 : 2) * (r == s ? 1 : 2) * (pq == rs ? 1 : 2);
                    packed[xmvb::core::PairIndex::packed_pair_of_pairs_index(pq, rs)] =
                        0.5 * multiplicity * dm2[eri_index(p, q, r, s, na)];
                }
        }

    const auto x = make_intermediates(h, g, nc, na);
    const Eigen::MatrixXd diagonal =
        xmvb::mcscf::build_analytic_rotation_hessian_diagonal(x, dm1, packed);
    // Exercise the complete AO-to-diagonal path against explicitly transformed
    // four-index integrals. Dense orbitals test off-diagonal density channels,
    // including all coincident-index ERI multiplicities.
    Eigen::MatrixXd mo = Eigen::MatrixXd::Identity(n, n);
    for (int p = 1; p < n; ++p) {
        Eigen::MatrixXd rotation = Eigen::MatrixXd::Identity(n, n);
        const double angle = 0.17 * p;
        rotation(0, 0) = rotation(p, p) = std::cos(angle);
        rotation(0, p) = -std::sin(angle);
        rotation(p, 0) = std::sin(angle);
        mo = (mo * rotation).eval();
    }
    const auto rotated_g = rotate_eri(g, mo);
    const auto rotated_x = make_intermediates(mo.transpose() * h * mo, rotated_g, nc, na);
    const Eigen::MatrixXd reference =
        xmvb::mcscf::build_analytic_rotation_hessian_diagonal(rotated_x, dm1, packed);
    xmvb::vb::VbScfInput input;
    input.standard_two_electron_mode = xmvb::vb::StandardTwoElectronMode::Exact;
    input.orbital_preparation_input.n_basis_functions = n;
    input.orbital_preparation_input.ao_overlap_matrix = Eigen::MatrixXd::Identity(n, n);
    input.ao_integral_input.n_basis_functions = n;
    input.ao_integral_input.ao_core_hamiltonian_matrix = h;
    std::vector<xmvb::integrals::PairAddress> left, right;
    std::vector<double> values;
    for (int p = 0; p < n; ++p)
        for (int q = 0; q <= p; ++q)
            for (int r = 0; r < n; ++r)
                for (int s = 0; s <= r; ++s) {
                    const int pq = p * (p + 1) / 2 + q, rs = r * (r + 1) / 2 + s;
                    if (pq < rs)
                        continue;
                    left.push_back(pq);
                    right.push_back(rs);
                    values.push_back(eri(g, p, q, r, s, n));
                }
    input.ao_integral_input.pair_graph =
        xmvb::integrals::build_ao_pair_graph(left, right, values, n);
    xmvb::vb::AcceptedPointContext accepted;
    accepted.n_active_orbitals = na;
    auto& prepared = accepted.prepared_active_space;
    prepared.n_inactive_doubly_occupied_orbitals = nc;
    auto& orbitals = prepared.orbital_result;
    orbitals.auxiliary_orbital_matrix = mo;
    orbitals.active_orbital_overlap_matrix.assign(na * na, 0.0);
    for (int a = 0; a < na; ++a)
        orbitals.active_orbital_overlap_matrix[a * na + a] = 1.0;
    orbitals.physical_orbital_frame.inactive_orthonormal_orbital_matrix = mo.leftCols(nc);
    orbitals.physical_orbital_frame.localized_representative_selector
        .inactive_inverse_transpose_right_transform = Eigen::MatrixXd::Identity(nc, nc);
    prepared.ao_effective_one_electron_result.ao_coulomb_exchange_matrix =
        mo * rotated_x.v_core * mo.transpose();
    accepted.active_one_electron_gradient.assign(dm1.data(), dm1.data() + dm1.size());
    accepted.packed_active_two_electron_gradient = packed;
    const auto built = xmvb::vb::build_analytic_orbital_diagonal(input, accepted);
    const double contraction_error = (built.rotation_diagonal - reference).cwiseAbs().maxCoeff();
    if (contraction_error > 2.0e-12) {
        std::cerr << "density-contracted diagonal error = " << contraction_error << '\n';
        return 1;
    }
    // Independent factorized-ERI reference for signed rank-two RI sources.
    auto ri = std::make_shared<xmvb::integrals::RiAoFactorization>();
    ri->n_basis_functions = n;
    ri->n_auxiliary_functions = 3;
    ri->n_packed_ao_pairs = n * (n + 1) / 2;
    ri->metric_whitened_ao_pair_factors.resize(3, ri->n_packed_ao_pairs);
    for (int auxiliary = 0; auxiliary < 3; ++auxiliary)
        for (int pair = 0; pair < ri->n_packed_ao_pairs; ++pair)
            ri->metric_whitened_ao_pair_factors(auxiliary, pair) =
                0.2 * std::sin(0.3 * (auxiliary + 1) * (pair + 1));
    std::vector<double> ri_g(g.size());
    for (int p = 0; p < n; ++p)
        for (int q = 0; q < n; ++q)
            for (int r = 0; r < n; ++r)
                for (int s = 0; s < n; ++s) {
                    const int pq = xmvb::core::PairIndex::packed_pair_index(p, q);
                    const int rs = xmvb::core::PairIndex::packed_pair_index(r, s);
                    ri_g[eri_index(p, q, r, s, n)] =
                        ri->metric_whitened_ao_pair_factors.col(pq).dot(
                            ri->metric_whitened_ao_pair_factors.col(rs));
                }
    const auto ri_x = make_intermediates(mo.transpose() * h * mo, rotate_eri(ri_g, mo), nc, na);
    const auto ri_reference =
        xmvb::mcscf::build_analytic_rotation_hessian_diagonal(ri_x, dm1, packed);
    input.standard_two_electron_mode = xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
    input.ri_factorization = ri;
    prepared.ao_effective_one_electron_result.ao_coulomb_exchange_matrix =
        mo * ri_x.v_core * mo.transpose();
    const auto ri_built = xmvb::vb::build_analytic_orbital_diagonal(input, accepted);
    const double ri_error = (ri_built.rotation_diagonal - ri_reference).cwiseAbs().maxCoeff();
    if (ri_error > 2.0e-12) {
        std::cerr << "RI density-contracted diagonal error = " << ri_error << '\n';
        return 1;
    }
    std::cout << "Density-contracted diagonal errors: exact=" << contraction_error
              << ", RI=" << ri_error << '\n';
    const double step = 2.0e-4;
    const double e0 = energy(h, g, dm1, dm2, nc);
    double max_error = 0.0;
    for (int q = 0; q < nc + na; ++q) {
        for (int p = q + 1; p < n; ++p) {
            if ((p < nc && q < nc) || (p < nc + na && q >= nc))
                continue;
            Eigen::MatrixXd up = Eigen::MatrixXd::Identity(n, n);
            Eigen::MatrixXd um = up;
            const double c = std::cos(step);
            const double s = std::sin(step);
            up(p, p) = up(q, q) = c;
            up(p, q) = -s;
            up(q, p) = s;
            um(p, p) = um(q, q) = c;
            um(p, q) = s;
            um(q, p) = -s;
            const double ep = energy(up.transpose() * h * up, rotate_eri(g, up), dm1, dm2, nc);
            const double em = energy(um.transpose() * h * um, rotate_eri(g, um), dm1, dm2, nc);
            const double fd = (ep - 2.0 * e0 + em) / (step * step);
            max_error = std::max(max_error, std::abs(fd - diagonal(p, q)));
        }
    }
    if (max_error > 2.0e-6) {
        std::cerr << "VBSCF diagonal finite-difference error = " << max_error << '\n';
        return 1;
    }
    std::cout << "VBSCF diagonal finite-difference error = " << max_error << '\n';
    return 0;
}
