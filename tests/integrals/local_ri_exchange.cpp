#include "integrals/lri/exchange.hpp"
#include <mpi.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace xmvb::integrals::lri;

namespace {
void close(const Eigen::MatrixXd& actual, const Eigen::MatrixXd& expected, const char* label) {
    const double error = (actual - expected).cwiseAbs().maxCoeff();
    std::cout << label << " max_abs_error=" << error << '\n';
    if (!actual.allFinite() || error > 2e-11 * std::max(1.0, expected.norm()))
        throw std::runtime_error(label);
}

void test(bool sparse) {
    const std::vector<AtomBasis> atoms{{2, 2}, {1, 3}, {2, 1}};
    const std::vector<std::size_t> ao{0, 2, 3, 5}, aux{0, 2, 5, 6};
    constexpr Eigen::Index n = 5, p = 6;
    Eigen::MatrixXd f = Eigen::MatrixXd::Zero(p, n * n);
    std::vector<PairFit> fits;
    for (std::size_t a = 0; a < atoms.size(); ++a)
        for (std::size_t b = a; b < atoms.size(); ++b) {
            if (sparse && a == 0 && b == 2)
                continue;
            const auto na = atoms[a].orbitals, nb = atoms[b].orbitals;
            PairFit fit{a, b, Eigen::MatrixXd(atoms[a].auxiliaries, na * nb), {}};
            if (a != b)
                fit.right_fit.resize(atoms[b].auxiliaries, na * nb);
            for (std::size_t i = 0; i < na; ++i)
                for (std::size_t j = 0; j < nb; ++j) {
                    const auto gi = ao[a] + i, gj = ao[b] + j;
                    for (std::size_t q = 0; q < atoms[a].auxiliaries; ++q) {
                        const double value = std::sin(0.17 * (q + 1) * (gi + gj + 2));
                        fit.left_fit(q, i * nb + j) = value;
                        f(aux[a] + q, gi * n + gj) = f(aux[a] + q, gj * n + gi) = value;
                    }
                    if (a != b)
                        for (std::size_t q = 0; q < atoms[b].auxiliaries; ++q) {
                            const double value = std::cos(0.23 * (q + 2) * (gi + gj + 3));
                            fit.right_fit(q, i * nb + j) = value;
                            f(aux[b] + q, gi * n + gj) = f(aux[b] + q, gj * n + gi) = value;
                        }
                }
            fits.push_back(std::move(fit));
        }
    Eigen::MatrixXd seed(p, p);
    for (Eigen::Index i = 0; i < p; ++i)
        for (Eigen::Index j = 0; j < p; ++j)
            seed(i, j) = std::sin(0.31 * (i + 1) * (j + 2));
    Eigen::MatrixXd v = seed.transpose() * seed + Eigen::MatrixXd::Identity(p, p);
    std::vector<CoulombBlock> blocks;
    for (std::size_t a = 0; a < atoms.size(); ++a)
        for (std::size_t b = a; b < atoms.size(); ++b)
            blocks.push_back(
                {a, b, v.block(aux[a], aux[b], atoms[a].auxiliaries, atoms[b].auxiliaries)});
    // Independent four-index oracle exists only in this five-AO test.
    const Eigen::MatrixXd eri = f.transpose() * v * f;
    auto reference = [&](const Eigen::MatrixXd& d) {
        Eigen::MatrixXd k = Eigen::MatrixXd::Zero(n, n);
        for (Eigen::Index i = 0; i < n; ++i)
            for (Eigen::Index j = 0; j < n; ++j)
                for (Eigen::Index l = 0; l < n; ++l)
                    for (Eigen::Index m = 0; m < n; ++m)
                        k(i, j) += eri(i * n + l, j * n + m) * d(l, m);
        return k;
    };
    Eigen::MatrixXd d(n, n), delta(n, n);
    for (Eigen::Index i = 0; i < n; ++i)
        for (Eigen::Index j = 0; j < n; ++j) {
            d(i, j) = std::cos(0.37 * (i + j + 1));
            delta(i, j) = std::sin(0.43 * (i + j + 2));
        }
    Exchange exchange(atoms, fits, blocks);
    const Eigen::MatrixXd k = exchange.apply(d), dk = exchange.apply(delta);
    close(k, reference(d), "exchange");
    close(dk, reference(delta), "signed response");
    close(k, k.transpose(), "Hermiticity");
    close(
        exchange.apply(Eigen::MatrixXd::Zero(n, n)), Eigen::MatrixXd::Zero(n, n), "zero response");
    close(exchange.apply(d), k, "repeated density");
    close(exchange.apply(d - 0.7 * delta), k - 0.7 * dk, "linearity");
    const double reciprocal = (d.cwiseProduct(dk)).sum() - (delta.cwiseProduct(k)).sum();
    if (std::abs(reciprocal) > 2e-10)
        throw std::runtime_error("reciprocity");
    constexpr double h = 1e-4;
    auto energy = [&](const Eigen::MatrixXd& x) {
        return 0.5 * x.cwiseProduct(exchange.apply(x)).sum();
    };
    const double derivative = (energy(d + h * delta) - energy(d - h * delta)) / (2 * h);
    const double error = std::abs(derivative - delta.cwiseProduct(k).sum());
    std::cout << "energy derivative abs_error=" << error << '\n';
    if (error > 2e-8)
        throw std::runtime_error("energy derivative");
}
} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int status = 0;
    try {
        test(false);
        test(true);
        try {
            Exchange bad({{static_cast<std::size_t>(2147483647) + 1, 1}}, {}, {});
            throw std::runtime_error("missing LP64 overflow check");
        } catch (const std::invalid_argument&) {
        }
        std::cout << "PASS local RI exchange and directional response\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        status = 1;
    }
    MPI_Finalize();
    return status;
}
