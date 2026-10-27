#include "mcscf/optimization/preconditioners/orbital_diagonal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "core/pair_index.hpp"

namespace xmvb::mcscf {

std::size_t orbital_diagonal_tensor4_index(int p, int q, int r, int s, int dimension) noexcept {
    return (((static_cast<std::size_t>(p) * dimension + q) * dimension + r) * dimension + s);
}

std::size_t orbital_diagonal_slice_index(int p, int a, int b, int n_active) noexcept {
    return (static_cast<std::size_t>(p) * n_active + a) * n_active + b;
}

std::size_t orbital_diagonal_packed_eri_count(int n_active) noexcept {
    const std::size_t pairs = static_cast<std::size_t>(n_active) * (n_active + 1) / 2;
    return pairs * (pairs + 1) / 2;
}

namespace {

std::size_t tensor4_index(int p, int q, int r, int s, int n) noexcept {
    return orbital_diagonal_tensor4_index(p, q, r, s, n);
}
std::size_t slice_index(int p, int a, int b, int n) noexcept {
    return orbital_diagonal_slice_index(p, a, b, n);
}

void validate(const AnalyticDiagonalIntermediates& x,
              const Eigen::Ref<const Eigen::MatrixXd>& dm1,
              const std::vector<double>& packed_gradient) {
    const int nmo = static_cast<int>(x.h_core.rows());
    if (x.n_core < 0 || x.n_active <= 0 || nmo < x.n_core + x.n_active || x.h_core.cols() != nmo ||
        x.v_core.rows() != nmo || x.v_core.cols() != nmo || x.j_pc.rows() != nmo ||
        x.j_pc.cols() != x.n_core || x.k_pc.rows() != nmo || x.k_pc.cols() != x.n_core ||
        dm1.rows() != x.n_active || dm1.cols() != x.n_active ||
        x.j_active.size() != static_cast<std::size_t>(nmo) * x.n_active * x.n_active ||
        x.k_active.size() != x.j_active.size() ||
        x.active_eri.size() !=
            static_cast<std::size_t>(x.n_active) * x.n_active * x.n_active * x.n_active ||
        packed_gradient.size() != orbital_diagonal_packed_eri_count(x.n_active) ||
        !x.h_core.allFinite() || !x.v_core.allFinite() || !x.j_pc.allFinite() ||
        !x.k_pc.allFinite() || !dm1.allFinite())
        throw std::invalid_argument("invalid analytic orbital diagonal input");
    const auto finite = [](double value) { return std::isfinite(value); };
    if (!std::all_of(x.j_active.begin(), x.j_active.end(), finite) ||
        !std::all_of(x.k_active.begin(), x.k_active.end(), finite) ||
        !std::all_of(x.active_eri.begin(), x.active_eri.end(), finite) ||
        !std::all_of(packed_gradient.begin(), packed_gradient.end(), finite))
        throw std::invalid_argument("nonfinite analytic orbital diagonal tensor");
}

} // namespace

Eigen::MatrixXd assemble_rotation_diagonal(const AnalyticDiagonalIntermediates& x,
                                           const Eigen::Ref<const Eigen::MatrixXd>& dm1_active,
                                           const DiagonalContractions& c) {
    const int nmo = x.h_core.rows();
    const int ncore = x.n_core;
    const int nactive = x.n_active;
    const int nocc = ncore + nactive;
    Eigen::MatrixXd dm1 = Eigen::MatrixXd::Zero(nmo, nmo);
    dm1.diagonal().head(ncore).setConstant(2.0);
    dm1.block(ncore, ncore, nactive, nactive) = dm1_active;
    Eigen::MatrixXd diagonal = Eigen::MatrixXd::Zero(nmo, nmo);
    for (int p = 0; p < nmo; ++p) {
        for (int q = 0; q < nmo; ++q)
            diagonal(p, q) = x.h_core(p, p) * dm1(q, q) - x.h_core(p, q) * dm1(p, q) +
                             x.h_core(q, q) * dm1(p, p) - x.h_core(q, p) * dm1(q, p) -
                             c.gradient[p] - c.gradient[q];
        diagonal(p, p) += 2.0 * c.gradient[p];
    }
    for (int p = 0; p < nmo; ++p)
        for (int core = 0; core < ncore; ++core) {
            diagonal(p, core) += 2.0 * c.potential[p];
            diagonal(core, p) += 2.0 * c.potential[p];
        }
    for (int core = 0; core < ncore; ++core)
        diagonal(core, core) -= 4.0 * c.potential[core];
    for (int p = 0; p < nmo; ++p)
        for (int a = 0; a < nactive; ++a) {
            const double value = x.v_core(p, p) * dm1_active(a, a);
            diagonal(p, ncore + a) += value;
            diagonal(ncore + a, p) += value;
        }
    for (int a = 0; a < nactive; ++a)
        for (int b = 0; b < nactive; ++b) {
            const double value = -x.v_core(ncore + a, ncore + b) * dm1_active(a, b);
            diagonal(ncore + a, ncore + b) += value;
            diagonal(ncore + b, ncore + a) += value;
        }
    for (int p = ncore; p < nmo; ++p)
        for (int core = 0; core < ncore; ++core) {
            const double value = 6.0 * x.k_pc(p, core) - 2.0 * x.j_pc(p, core);
            diagonal(p, core) += value;
            diagonal(core, p) += value;
        }
    for (int p = 0; p < nocc; ++p)
        for (int a = 0; a < nactive; ++a) {
            diagonal(p, ncore + a) -= c.one_rdm(p, a);
            diagonal(ncore + a, p) -= c.one_rdm(p, a);
        }
    for (int p = 0; p < nmo; ++p)
        for (int a = 0; a < nactive; ++a) {
            diagonal(ncore + a, p) += c.two_rdm(p, a);
            diagonal(p, ncore + a) += c.two_rdm(p, a);
        }
    diagonal *= 2.0;
    diagonal = (0.5 * (diagonal + diagonal.transpose())).eval();
    if (!diagonal.allFinite())
        throw std::runtime_error("nonfinite orbital Hessian diagonal");
    return diagonal;
}

Eigen::VectorXd diagonal_gradient(const AnalyticDiagonalIntermediates& x,
                                  const Eigen::Ref<const Eigen::MatrixXd>& dm1,
                                  const std::vector<double>& dm2,
                                  const Eigen::VectorXd& potential) {
    const int n = x.n_active;
    const int core = x.n_core;
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(x.h_core.rows());
    gradient.head(core) = 2.0 * (x.h_core.diagonal().head(core) + potential.head(core));
    for (int a = 0; a < n; ++a)
        for (int u = 0; u < n; ++u) {
            gradient[core + a] +=
                (x.h_core(core + a, core + u) + x.v_core(core + a, core + u)) * dm1(u, a);
            for (int w = 0; w < n; ++w)
                for (int z = 0; z < n; ++z)
                    gradient[core + a] += x.active_eri[tensor4_index(a, u, w, z, n)] *
                                          dm2[tensor4_index(w, z, u, a, n)];
        }
    return gradient;
}

std::vector<double> unpack_symmetric_active_two_rdm(const std::vector<double>& packed_gradient,
                                                    int nactive) {
    if (nactive <= 0 || packed_gradient.size() != orbital_diagonal_packed_eri_count(nactive))
        throw std::invalid_argument("invalid packed active ERI gradient");
    std::vector<double> dm2(static_cast<std::size_t>(nactive) * nactive * nactive * nactive);
    for (int p = 0; p < nactive; ++p)
        for (int q = 0; q < nactive; ++q) {
            const auto pq = core::PairIndex::packed_pair_index(p, q);
            for (int r = 0; r < nactive; ++r)
                for (int s = 0; s < nactive; ++s) {
                    const auto rs = core::PairIndex::packed_pair_index(r, s);
                    const auto packed = core::PairIndex::packed_pair_of_pairs_index(pq, rs);
                    const int multiplicity =
                        (p == q ? 1 : 2) * (r == s ? 1 : 2) * (pq == rs ? 1 : 2);
                    dm2[tensor4_index(p, q, r, s, nactive)] =
                        2.0 * packed_gradient[packed] / multiplicity;
                }
        }
    return dm2;
}

Eigen::MatrixXd
build_analytic_rotation_hessian_diagonal(const AnalyticDiagonalIntermediates& x,
                                         const Eigen::Ref<const Eigen::MatrixXd>& dm1_active,
                                         const std::vector<double>& packed_gradient) {
    validate(x, dm1_active, packed_gradient);
    const int nmo = static_cast<int>(x.h_core.rows());
    const int ncore = x.n_core;
    const int nactive = x.n_active;
    const int nocc = ncore + nactive;
    const auto dm2 = unpack_symmetric_active_two_rdm(packed_gradient, nactive);
    Eigen::VectorXd vtotal = x.v_core.diagonal();
    for (int p = 0; p < nmo; ++p)
        for (int a = 0; a < nactive; ++a)
            for (int b = 0; b < nactive; ++b)
                vtotal[p] += x.j_active[slice_index(p, a, b, nactive)] * dm1_active(a, b) -
                             0.5 * x.k_active[slice_index(p, a, b, nactive)] * dm1_active(a, b);
    const Eigen::VectorXd gradient = diagonal_gradient(x, dm1_active, dm2, vtotal);
    Eigen::MatrixXd one = Eigen::MatrixXd::Zero(nocc, nactive);
    Eigen::MatrixXd two = Eigen::MatrixXd::Zero(nmo, nactive);
    for (int p = 0; p < nmo; ++p)
        for (int a = 0; a < nactive; ++a) {
            for (int w = 0; w < nactive; ++w)
                for (int z = 0; z < nactive; ++z) {
                    two(p, a) += x.j_active[slice_index(p, w, z, nactive)] *
                                 dm2[tensor4_index(w, z, a, a, nactive)];
                    two(p, a) += x.k_active[slice_index(p, w, z, nactive)] *
                                 (dm2[tensor4_index(a, w, z, a, nactive)] +
                                  dm2[tensor4_index(w, a, z, a, nactive)]);
                }
            if (p < nocc)
                for (int v = 0; v < nactive; ++v)
                    one(p, a) += (6.0 * x.k_active[slice_index(p, a, v, nactive)] -
                                  2.0 * x.j_active[slice_index(p, a, v, nactive)]) *
                                 dm1_active(a, v);
        }
    return assemble_rotation_diagonal(
        x, dm1_active, {std::move(vtotal), gradient, std::move(one), std::move(two)});
}

} // namespace xmvb::mcscf
