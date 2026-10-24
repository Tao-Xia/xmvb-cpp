#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <Eigen/Core>
#include "integrals/response/ri_ao.hpp"
#include "integrals/response/density_diagonal.hpp"
#include "parallel/openmp.hpp"

namespace {
using Matrix = Eigen::MatrixXd;

void close(const Matrix& actual, const Matrix& expected) {
    if (actual.rows() != expected.rows() || actual.cols() != expected.cols())
        throw std::runtime_error("RI response output shape differs from reference");
    if (!actual.size())
        return;
    const double error = (actual - expected).cwiseAbs().maxCoeff();
    if (error > 2e-11 * std::max(1.0, expected.cwiseAbs().maxCoeff()))
        throw std::runtime_error("factor-native RI response differs from explicit contraction");
}

/** @brief Independent complete AO contractions, including signed sources. */
void check_diagonals(const Matrix& factors, const Matrix& c, int n_core, std::size_t budget) {
    const int n = c.rows(), na = c.cols(), aux = factors.rows(), n_mo = n_core + na + 2;
    Matrix mo(n, n_mo);
    for (int j = 0; j < n_mo; ++j)
        for (int i = 0; i < n; ++i)
            mo(i, j) = 0.13 * std::sin(0.3 * (i + 1) * (j + 2));
    mo.middleCols(n_core, na) = c;
    Matrix weights(na * na, 3);
    for (int j = 0; j < 3; ++j) {
        Eigen::Map<Matrix> source(weights.col(j).data(), na, na);
        for (int b = 0; b < na; ++b)
            for (int a = 0; a < na; ++a)
                source(a, b) = j == 2 ? 0 : std::cos(0.4 * (a + b + 1)) - (j + 0.3) * (a == b);
    }
    xmvb::integrals::OrbitalPairMatrix coulomb =
        xmvb::integrals::OrbitalPairMatrix::Zero(n_mo, na * na);
    Matrix core_ref = Matrix::Zero(n_mo, n_core), active_ref = Matrix::Zero(n_mo, 3),
           core_j = Matrix::Zero(n_mo, n_core);
    for (int q = 0; q < aux; ++q) {
        Matrix l(n, n);
        for (int mu = 0; mu < n; ++mu)
            for (int nu = 0; nu <= mu; ++nu)
                l(mu, nu) = l(nu, mu) = factors(q, mu * (mu + 1) / 2 + nu);
        const Matrix projected = mo.transpose() * l * mo, active = c.transpose() * l * c;
        for (int p = 0; p < n_mo; ++p)
            for (int a = 0; a < na; ++a)
                for (int b = 0; b < na; ++b)
                    coulomb(p, a * na + b) += projected(p, p) * active(a, b);
        for (int j = 0; j < n_core + 3; ++j) {
            Matrix density(n, n);
            if (j < n_core)
                density.noalias() = mo.col(j) * mo.col(j).transpose();
            else {
                const Eigen::Map<const Matrix> w(weights.col(j - n_core).data(), na, na);
                density.noalias() = c * w * c.transpose();
            }
            const Matrix image = 2 * (l.array() * density.array()).sum() * l - l * density * l;
            const Eigen::VectorXd diagonal = (mo.transpose() * image * mo).diagonal();
            if (j < n_core) {
                core_ref.col(j) += diagonal;
                core_j.col(j) += projected.diagonal() * projected(j, j);
            } else
                active_ref.col(j - n_core) += diagonal;
        }
    }
    xmvb::integrals::RiAoFactorization ri;
    ri.n_basis_functions = n;
    ri.n_auxiliary_functions = aux;
    ri.n_packed_ao_pairs = factors.cols();
    ri.metric_whitened_ao_pair_factors = factors;
    const auto actual = xmvb::integrals::contract_ri_density_diagonals(
        ri, mo, n_core, na, weights, coulomb, budget);
    close(actual.core, core_ref);
    close(actual.active, active_ref);
    close(actual.core_coulomb, core_j);
}

void check(int ni, bool cached, std::size_t budget) {
    constexpr int n = 13, na = 3, aux = 19;
    Matrix factors(aux, n * (n + 1) / 2), c(n, na), a(n, ni), x(n, ni), w(n, na);
    for (int q = 0; q < factors.rows(); ++q)
        for (int p = 0; p < factors.cols(); ++p)
            factors(q, p) = 0.1 * std::sin(0.7 * (p + 1) * (q + 2));
    for (int mu = 0; mu < n; ++mu) {
        for (int j = 0; j < na; ++j) {
            c(mu, j) = 0.2 * std::cos(0.3 * (mu + 1) * (j + 1));
            w(mu, j) = 0.1 * std::sin(0.4 * (mu + 2) * (j + 1));
        }
        for (int j = 0; j < ni; ++j) {
            a(mu, j) = 0.3 * std::sin(0.2 * (mu + 1) * (j + 1));
            x(mu, j) = 0.1 * std::cos(0.7 * (mu + 2) * (j + 1));
        }
    }
    xmvb::integrals::RiActiveTwoElectronResponseCache cache;
    cache.n_basis_functions = n;
    cache.n_active_orbitals = na;
    cache.n_auxiliary_functions = aux;
    cache.metric_whitened_ao_pair_factors = &factors;
    cache.accepted_active_coefficients = &c;
    Matrix pair_factors(aux, na * (na + 1) / 2);
    cache.accepted_active_pair_factors = &pair_factors;
    cache.auxiliary_tile_size = 7;
    for (int j = 0; j < na; ++j)
        for (int i = 0; i <= j; ++i) {
            cache.active_pair_first_indices.push_back(j);
            cache.active_pair_second_indices.push_back(i);
        }
    Matrix weights(na, na);
    for (int j = 0; j < na; ++j)
        for (int i = 0; i < na; ++i)
            weights(i, j) = 0.2 * std::cos(0.3 * (i + j + 1)) - (i == j ? 0.4 : 0.0);
    const Matrix active_density = c * weights * c.transpose();
    Matrix active_reference = Matrix::Zero(n, n);
    std::shared_ptr<Matrix> resident;
    if (cached)
        resident = std::make_shared<Matrix>(aux * na, n);
    const Matrix d = x * a.transpose() + a * x.transpose(),
                 k = w * c.transpose() + c * w.transpose();
    Matrix gd = Matrix::Zero(n, n), gk = gd;
    Matrix inactive_image = Matrix::Zero(aux * (ni + 2), n);
    std::vector<int> selected(ni);
    for (int j = 0; j < ni; ++j)
        selected[j] = ni - j;
    Matrix a0(n, ni + 2);
    for (int j = 0; j < a0.cols(); ++j)
        for (int mu = 0; mu < n; ++mu)
            a0(mu, j) = 0.17 * std::cos(0.19 * (mu + 2) * (j + 1));
    for (int j = 0; j < ni; ++j)
        a0.col(selected[j]) = a.col(j);
    for (int q = 0; q < aux; ++q) {
        Matrix l(n, n);
        for (int mu = 0; mu < n; ++mu)
            for (int nu = 0; nu <= mu; ++nu)
                l(mu, nu) = l(nu, mu) = factors(q, mu * (mu + 1) / 2 + nu);
        gd.noalias() += 2 * (l.array() * d.array()).sum() * l - l * d * l;
        gk.noalias() += 2 * (l.array() * k.array()).sum() * l - l * k * l;
        active_reference.noalias() +=
            2 * (l.array() * active_density.array()).sum() * l - l * active_density * l;
        const Matrix la0 = l * a0;
        for (int j = 0; j < a0.cols(); ++j)
            inactive_image.row(j * aux + q) = la0.col(j).transpose();
        const Matrix lc = l * c;
        const Matrix active_pair = c.transpose() * lc;
        for (int j = 0; j < na; ++j)
            for (int i = 0; i <= j; ++i)
                pair_factors(q, j * (j + 1) / 2 + i) = active_pair(j, i);
        if (cached) {
            for (int j = 0; j < na; ++j)
                resident->row(j * aux + q) = lc.col(j).transpose();
        }
    }
    cache.accepted_one_index_factors = std::move(resident);
    xmvb::integrals::RiAoResponseWorkspace workspace;
    const std::array<xmvb::integrals::RiAoResponseDirection, 1> requests{{{x, w}}};
    auto actual = xmvb::integrals::apply_ri_projected_ao_response_batch(
                      cache, a, requests, a0, workspace, budget, {nullptr, selected})
                      .front();
    close(actual.inactive, 2 * (2 * gd + 0.5 * gk) * a0);
    close(actual.active, gd * c);
    const auto reused = xmvb::integrals::apply_ri_projected_ao_response_batch(
                            cache, a, requests, a0, workspace, budget, {&inactive_image, selected})
                            .front();
    close(reused.inactive, actual.inactive);
    close(reused.active, actual.active);
    for (int tile : {1, 7, aux}) {
        cache.auxiliary_tile_size = tile;
        close(xmvb::integrals::apply_ri_active_density_response(cache, weights), active_reference);
    }
    // Reuse must clear all per-worker projections and leave no stale charge.
    const Matrix zero_x = 0 * x, zero_w = 0 * w;
    const std::array<xmvb::integrals::RiAoResponseDirection, 1> zero_request{{{zero_x, zero_w}}};
    const auto zero =
        xmvb::integrals::apply_ri_projected_ao_response_batch(
            cache, a, zero_request, a0, workspace, budget, {&inactive_image, selected})
            .front();
    close(zero.inactive, Matrix::Zero(n, a0.cols()));
    close(zero.active, Matrix::Zero(n, na));
    check_diagonals(factors, c, ni, budget);
}
} // namespace

int main() {
    for (int threads : {1, 2, 4}) {
#ifdef _OPENMP
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
#endif
        for (int ni : {0, 2, 5})
            for (bool cached : {false, true})
                for (std::size_t budget : {std::size_t{8192}, xmvb::workspace::available()})
                    check(ni, cached, budget);
    }
    std::cout << "ri_ao_response_exact_projection=PASS\n";
}
