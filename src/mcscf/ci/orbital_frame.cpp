#include "mcscf/ci/orbital_frame.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>

#include "integrals/active/transform.hpp"
#include "mcscf/ci/reference.hpp"
#include "parallel/openmp.hpp"

namespace xmvb::mcscf {

CiOrbitalFrame::Neighborhoods CiOrbitalFrame::neighborhoods(const SpinConnections& graph) {
    Neighborhoods result(graph.orbitals - 1);
    for (int p = 0; p + 1 < graph.orbitals; ++p) {
        auto& pairs = result[p];
        for (StringIndex row = 0; row < graph.n_strings(); ++row) {
            const auto mask = graph.basis.masks()[row];
            if (((mask >> p) & 3) != 1)
                continue;
            const auto& link =
                graph.singles(row)[graph.link_prefix(row, p + 1, (std::uint64_t{1} << p) - 1)];
            pairs.push_back({row, link.source});
        }
    }
    return result;
}

CiOrbitalFrame::CiOrbitalFrame(const DirectCiSigmaAction& reference,
                               const ActiveIntegrals& integrals)
    : alpha_(&reference.alpha_connections()), beta_(&reference.beta_connections_view()),
      alpha_pairs_(neighborhoods(*alpha_)) {
    if (alpha_ != beta_)
        beta_pairs_ = neighborhoods(*beta_);
    const auto n = integrals.one_electron.rows();
    Eigen::MatrixXd qr = mean_field_orbitals(integrals, alpha_->n_occupied() + beta_->n_occupied());
    for (int column = 0; column + 1 < n; ++column)
        for (int row = n - 1; row > column; --row) {
            const double a = qr(row - 1, column), b = qr(row, column);
            if (b == 0.0)
                continue;
            const double radius = std::hypot(a, b), c = a / radius, s = b / radius;
            const Eigen::RowVectorXd first = qr.row(row - 1);
            qr.row(row - 1) = c * first + s * qr.row(row);
            qr.row(row) = -s * first + c * qr.row(row);
            rotations_.push_back({row - 1, c, s});
        }
    orbitals_ = Eigen::MatrixXd::Identity(n, n);
    for (int p = 0; p < n; ++p)
        if (qr(p, p) < 0) {
            negative_ |= std::uint64_t{1} << p;
            orbitals_(p, p) = -1;
        }
    for (auto r = rotations_.rbegin(); r != rotations_.rend(); ++r) {
        const Eigen::RowVectorXd first = orbitals_.row(r->first);
        orbitals_.row(r->first) = r->cosine * first - r->sine * orbitals_.row(r->first + 1);
        orbitals_.row(r->first + 1) = r->sine * first + r->cosine * orbitals_.row(r->first + 1);
    }
    integrals_.one_electron = orbitals_.transpose() * integrals.one_electron * orbitals_;
    integrals_.pair_kernel =
        xmvb::integrals::transform_pair_kernel(integrals.pair_kernel, orbitals_);
    action_ = std::make_unique<DirectCiSigmaAction>(reference, integrals_);
    std::size_t bytes = rotations_.capacity() * sizeof(Rotation) +
                        sizeof(double) * (orbitals_.size() + 2 * integrals_.one_electron.size() +
                                          3 * integrals_.pair_kernel.size());
    for (const auto* family : {&alpha_pairs_, &beta_pairs_})
        for (const auto& pairs : *family)
            bytes = core::checked_add(bytes, pairs.capacity() * sizeof(Pair));
    metadata_.resize(bytes);
}

void CiOrbitalFrame::transform(Eigen::Ref<Eigen::MatrixXd> x, bool inverse) const {
    const auto na = alpha_->n_strings(), nb = beta_->n_strings();
    if (x.rows() != na || x.cols() <= 0 || x.cols() % nb)
        throw std::invalid_argument("internal CI orbital frame has incompatible coefficients");
    const int workers = bounded_openmp_thread_count(
        coarse_openmp_thread_count(effective_openmp_thread_count(), x.size()), x.cols());
    const auto diagonal = [&](auto&& element, const SpinConnections& graph) {
        if (negative_)
            for (StringIndex row = 0; row < graph.n_strings(); ++row)
                if (std::popcount(graph.basis.masks()[row] & negative_) % 2)
                    element(row);
    };
#pragma omp parallel for schedule(static) if (workers > 1) num_threads(workers)
    for (Eigen::Index column = 0; column < x.cols(); ++column) {
        auto* values = x.col(column).data();
        const auto signs = [&] {
            diagonal([&](auto row) { values[row] = -values[row]; }, *alpha_);
        };
        const auto rotate = [&](const Rotation& r) {
            const double s = inverse ? -r.sine : r.sine;
            for (const auto& pair : alpha_pairs_[r.first]) {
                const double a = values[pair.first], b = values[pair.second];
                values[pair.first] = r.cosine * a - s * b;
                values[pair.second] = s * a + r.cosine * b;
            }
        };
        if (!inverse)
            signs();
        if (inverse)
            for (const auto& r : rotations_)
                rotate(r);
        else
            for (auto r = rotations_.rbegin(); r != rotations_.rend(); ++r)
                rotate(*r);
        if (inverse)
            signs();
    }
    const auto& beta_pairs = alpha_ == beta_ ? alpha_pairs_ : beta_pairs_;
    // Each borrowed row panel fits the team's L2 when possible. Its rows are
    // disjoint from every other panel; neither rotation needs a scratch copy.
    const auto cache = std::max<std::ptrdiff_t>(1, Eigen::l2CacheSize());
    const auto extent = std::clamp<Eigen::Index>(cache / sizeof(double) / nb, 1, na);
    const auto panels = na / extent + (na % extent != 0);
    const int row_workers = bounded_openmp_thread_count(effective_openmp_thread_count(), panels);
#pragma omp parallel for schedule(static) if (row_workers > 1) num_threads(row_workers)
    for (StringIndex panel = 0; panel < panels; ++panel) {
        const auto first = panel * extent, count = std::min(extent, na - first);
        for (Eigen::Index block = 0; block < x.cols(); block += nb) {
            const auto signs = [&] {
                diagonal([&](auto column) { x.col(block + column).segment(first, count) *= -1; },
                         *beta_);
            };
            const auto rotate = [&](const Rotation& r) {
                const double s = inverse ? -r.sine : r.sine;
                for (const auto& pair : beta_pairs[r.first]) {
                    auto* left = x.col(block + pair.first).data() + first;
                    auto* right = x.col(block + pair.second).data() + first;
#pragma omp simd
                    for (StringIndex row = 0; row < count; ++row) {
                        const double a = left[row], b = right[row];
                        left[row] = r.cosine * a - s * b;
                        right[row] = s * a + r.cosine * b;
                    }
                }
            };
            if (!inverse)
                signs();
            if (inverse)
                for (const auto& r : rotations_)
                    rotate(r);
            else
                for (auto r = rotations_.rbegin(); r != rotations_.rend(); ++r)
                    rotate(*r);
            if (inverse)
                signs();
        }
    }
}

} // namespace xmvb::mcscf
