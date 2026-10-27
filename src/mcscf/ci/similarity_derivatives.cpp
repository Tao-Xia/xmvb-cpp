#include "mcscf/ci/similarity.hpp"

namespace xmvb::mcscf {
namespace {
/** @brief Small active-orbital matrix and its analytic directional derivative. */
struct Jet {
    Eigen::MatrixXd value, tangent;
    Jet transpose() const {
        return {value.transpose(), tangent.transpose()};
    }
};
Jet operator*(const Jet& a, const Jet& b) {
    Jet result{a.value * b.value, {}};
    if (a.tangent.size())
        result.tangent = a.tangent * b.value + a.value * b.tangent;
    return result;
}
Eigen::MatrixXd unpack(const Eigen::MatrixXd& packed, int n) {
    Eigen::MatrixXd result(n * n, n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l)
                    result(i * n + j, k * n + l) = packed(core::PairIndex::packed_pair_index(i, j),
                                                          core::PairIndex::packed_pair_index(k, l));
    return result;
}
Eigen::MatrixXd change_direction(const Eigen::MatrixXd& inverse,
                                 const Eigen::MatrixXd& r,
                                 const Eigen::MatrixXd& di,
                                 const Eigen::MatrixXd& dr) {
    const int n = r.rows();
    Eigen::MatrixXd result(n * n, n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int p = 0; p < n; ++p)
                for (int q = 0; q < n; ++q)
                    result(i * n + j, p * n + q) = di(i, p) * r(q, j) + inverse(i, p) * dr(q, j);
    return result;
}
} // namespace
SimilarityHamiltonian SimilarityHamiltonian::direction(const ActiveIntegrals& integrals,
                                                       const Eigen::MatrixXd& dr) const {
    const Eigen::MatrixXd di = -inverse_ * dr * inverse_;
    const Eigen::MatrixXd dc = change_direction(inverse_, r_, di, dr);
    Eigen::MatrixXd dh = di * physical_h_ * r_ + inverse_ * integrals.one_electron * r_ +
                         inverse_ * physical_h_ * dr;
    Eigen::MatrixXd dg = dc * physical_g_ * change_.transpose() +
                         change_ * unpack(integrals.pair_kernel, n_) * change_.transpose() +
                         change_ * physical_g_ * dc.transpose();
    return SimilarityHamiltonian(n_, std::move(dh), std::move(dg));
}
ActiveIntegrals SimilarityHamiltonian::pullback(const Eigen::MatrixXd& bh,
                                                const Eigen::MatrixXd& bg,
                                                const Eigen::MatrixXd* dr,
                                                const Eigen::MatrixXd* dbh,
                                                const Eigen::MatrixXd* dbg) const {
    const bool moving = dr != nullptr;
    if (moving != (dbh != nullptr) || moving != (dbg != nullptr))
        throw std::invalid_argument("incomplete canonical adjoint direction");
    const auto zero = [](const Eigen::MatrixXd&) { return Eigen::MatrixXd{}; };
    const Jet r{r_, moving ? *dr : zero(r_)};
    const Jet inv{inverse_,
                  moving ? Eigen::MatrixXd(-inverse_ * (*dr) * inverse_) : zero(inverse_)};
    const Jet c{change_, moving ? change_direction(inverse_, r_, inv.tangent, *dr) : zero(change_)};
    const Jet ah{bh, moving ? *dbh : zero(bh)}, ag{bg, moving ? *dbg : zero(bg)};
    const Jet gh = inv.transpose() * ah * r.transpose();
    const Jet gg = c.transpose() * ag * c;
    ActiveIntegrals result;
    result.one_electron = moving ? gh.tangent : gh.value;
    const auto& full = moving ? gg.tangent : gg.value;
    const int pairs = n_ * (n_ + 1) / 2;
    result.pair_kernel = Eigen::MatrixXd::Zero(pairs, pairs);
    for (int i = 0; i < n_; ++i)
        for (int j = 0; j < n_; ++j)
            for (int k = 0; k < n_; ++k)
                for (int l = 0; l < n_; ++l)
                    result.pair_kernel(core::PairIndex::packed_pair_index(i, j),
                                       core::PairIndex::packed_pair_index(k, l)) +=
                        full(i * n_ + j, k * n_ + l);
    result.one_electron = (0.5 * (result.one_electron + result.one_electron.transpose())).eval();
    result.pair_kernel = (0.5 * (result.pair_kernel + result.pair_kernel.transpose())).eval();
    return result;
}
} // namespace xmvb::mcscf
