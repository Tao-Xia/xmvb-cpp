#include "integrals/nao/prepare.hpp"
#include "core/index.hpp"
#include "integrals/lri/fit.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace xmvb::integrals::nao {
namespace {
struct Pair {
    std::size_t a, b;
    Eigen::MatrixXd three_center;
};
Eigen::Index checked(std::size_t n) {
    return core::checked_index<Eigen::Index>(n);
}
std::size_t product(std::size_t a, std::size_t b) {
    const auto count = core::checked_multiply(a, b);
    checked(count);
    return count;
}
} // namespace

Prepared prepare(const std::vector<Center>& centers, const PreparationOptions& options) {
    if (centers.empty() || !options.panel_points ||
        !(options.relative_rank_cutoff > 0 && options.relative_rank_cutoff < 1))
        throw std::invalid_argument("invalid NAO preparation options");
    Prepared out;
    std::vector<std::size_t> ao{0}, aux{0};
    Eigen::MatrixXd positions(checked(centers.size()), 3);
    for (std::size_t a = 0; a < centers.size(); ++a) {
        const auto& c = centers[a];
        if (c.orbitals.empty() || c.auxiliaries.empty() || !c.position.allFinite())
            throw std::invalid_argument("NAO center requires orbitals and auxiliaries");
        for (const auto* list : {&c.orbitals, &c.auxiliaries})
            for (const auto& f : *list)
                if (!f.radial || f.m < -f.radial->l() || f.m > f.radial->l())
                    throw std::invalid_argument("invalid NAO function");
        ao.push_back(core::checked_add(ao.back(), c.orbitals.size()));
        aux.push_back(core::checked_add(aux.back(), c.auxiliaries.size()));
        positions.row(checked(a)) = c.position.transpose();
        out.atoms.push_back({c.orbitals.size(), c.auxiliaries.size()});
    }
    const auto n = checked(ao.back()), p = checked(aux.back());
    product(ao.back(), ao.back());
    out.overlap = Eigen::MatrixXd::Zero(n, n);
    out.kinetic = Eigen::MatrixXd::Zero(n, n);
    std::vector<Pair> pairs;
    for (std::size_t a = 0; a < centers.size(); ++a)
        for (std::size_t b = a; b < centers.size(); ++b) {
            const auto pa = out.atoms[a].auxiliaries, pb = out.atoms[b].auxiliaries;
            product(pa, pb);
            out.coulomb.push_back({a, b, Eigen::MatrixXd::Zero(checked(pa), checked(pb))});
            double ra = 0, rb = 0;
            for (const auto& f : centers[a].orbitals)
                ra = std::max(ra, f.radial->cutoff());
            for (const auto& f : centers[b].orbitals)
                rb = std::max(rb, f.radial->cutoff());
            if ((centers[a].position - centers[b].position).norm() >= ra + rb)
                continue;
            const auto cols = product(out.atoms[a].orbitals, out.atoms[b].orbitals);
            const auto rows = core::checked_add(pa, a == b ? 0 : pb);
            product(rows, cols);
            pairs.push_back({a, b, Eigen::MatrixXd::Zero(checked(rows), checked(cols))});
        }
    // One grid is shared; evaluated fields exist only for the current panel.
    const auto grid = dft::make_grid(positions, options.grid);
    const auto width = std::min(checked(options.panel_points), grid.weights.size());
    const auto field_columns = product(2, core::checked_add(ao.back(), aux.back()));
    out.panel_bytes =
        product(product(static_cast<std::size_t>(width), field_columns), sizeof(double));
    Eigen::MatrixXd values(width, n), kinetic(width, n), auxiliary(width, p), potential(width, p);
    for (Eigen::Index start = 0; start < grid.weights.size(); start += width) {
        const auto count = std::min(width, grid.weights.size() - start);
#pragma omp parallel for schedule(static)
        for (Eigen::Index q = 0; q < count; ++q) {
            const Eigen::Vector3d point = grid.points.row(start + q).transpose();
            for (std::size_t a = 0; a < centers.size(); ++a) {
                const auto x = (point - centers[a].position).eval();
                for (std::size_t i = 0; i < centers[a].orbitals.size(); ++i) {
                    const auto& f = centers[a].orbitals[i];
                    values(q, checked(ao[a] + i)) = f.radial->value(x, f.m);
                    kinetic(q, checked(ao[a] + i)) = f.radial->kinetic(x, f.m);
                }
                for (std::size_t i = 0; i < centers[a].auxiliaries.size(); ++i) {
                    const auto& f = centers[a].auxiliaries[i];
                    auxiliary(q, checked(aux[a] + i)) = f.radial->value(x, f.m);
                    potential(q, checked(aux[a] + i)) = f.radial->potential(x, f.m);
                }
            }
        }
        const auto w = grid.weights.segment(start, count);
        out.overlap.noalias() +=
            values.topRows(count).transpose() * w.asDiagonal() * values.topRows(count);
        out.kinetic.noalias() +=
            values.topRows(count).transpose() * w.asDiagonal() * kinetic.topRows(count);
        for (auto& v : out.coulomb) {
            const auto pa = checked(out.atoms[v.left].auxiliaries),
                       pb = checked(out.atoms[v.right].auxiliaries);
            const Eigen::MatrixXd forward =
                auxiliary.block(0, checked(aux[v.left]), count, pa).transpose() * w.asDiagonal() *
                potential.block(0, checked(aux[v.right]), count, pb);
            const Eigen::MatrixXd reverse =
                potential.block(0, checked(aux[v.left]), count, pa).transpose() * w.asDiagonal() *
                auxiliary.block(0, checked(aux[v.right]), count, pb);
            v.values += 0.5 * (forward + reverse);
        }
        for (auto& pair : pairs) {
            const auto a = pair.a, b = pair.b;
            const auto na = out.atoms[a].orbitals, nb = out.atoms[b].orbitals;
            Eigen::MatrixXd products(count, checked(na * nb));
            for (std::size_t i = 0; i < na; ++i)
                for (std::size_t j = 0; j < nb; ++j)
                    products.col(checked(i * nb + j)) =
                        values.col(checked(ao[a] + i)).head(count).array() *
                        values.col(checked(ao[b] + j)).head(count).array() * w.array();
            const auto pa = checked(out.atoms[a].auxiliaries);
            pair.three_center.topRows(pa).noalias() +=
                potential.block(0, checked(aux[a]), count, pa).transpose() * products;
            if (a != b) {
                const auto pb = checked(out.atoms[b].auxiliaries);
                pair.three_center.bottomRows(pb).noalias() +=
                    potential.block(0, checked(aux[b]), count, pb).transpose() * products;
            }
        }
    }
    out.kinetic_asymmetry = (out.kinetic - out.kinetic.transpose()).cwiseAbs().maxCoeff();
    out.kinetic = (0.5 * (out.kinetic + out.kinetic.transpose())).eval();
    auto metric = [&](std::size_t a, std::size_t b) -> const Eigen::MatrixXd& {
        for (const auto& v : out.coulomb)
            if (v.left == a && v.right == b)
                return v.values;
        throw std::logic_error("missing local metric block");
    };
    for (auto& pair : pairs) {
        const auto a = pair.a, b = pair.b;
        Eigen::MatrixXd v;
        const auto pa = checked(out.atoms[a].auxiliaries);
        if (a == b)
            v = metric(a, a);
        else {
            const auto pb = checked(out.atoms[b].auxiliaries);
            product(core::checked_add(out.atoms[a].auxiliaries, out.atoms[b].auxiliaries),
                    core::checked_add(out.atoms[a].auxiliaries, out.atoms[b].auxiliaries));
            v.resize(pa + pb, pa + pb);
            v.topLeftCorner(pa, pa) = metric(a, a);
            v.topRightCorner(pa, pb) = metric(a, b);
            v.bottomLeftCorner(pb, pa) = metric(a, b).transpose();
            v.bottomRightCorner(pb, pb) = metric(b, b);
        }
        auto fit = lri::fit_pair(v, pair.three_center, options.relative_rank_cutoff);
        pair.three_center.resize(0, 0);
        out.fit_ranks.push_back(fit.rank);
        out.max_fit_residual = std::max(out.max_fit_residual, fit.relative_residual);
        lri::PairFit result{a, b, fit.coefficients.topRows(pa), {}};
        if (a != b)
            result.right_fit = fit.coefficients.bottomRows(checked(out.atoms[b].auxiliaries));
        out.fits.push_back(std::move(result));
    }
    return out;
}
} // namespace xmvb::integrals::nao
