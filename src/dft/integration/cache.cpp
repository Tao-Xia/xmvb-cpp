#include "dft/integration/cache.hpp"

#include <algorithm>
#include "dft/grid/atomic.hpp"

namespace xmvb::dft {
namespace {
ConstMatrix slice(const Eigen::MatrixXd& a, Eigen::Index begin, Eigen::Index rows) {
    if (a.rows() == 0)
        return a;
    return a.middleRows(begin, rows);
}
} // namespace

BasisTile::BasisTile(const integrals::BasisValues& values, Eigen::Index begin, Eigen::Index rows)
    : value(values.value.middleRows(begin, rows)),
      gradient{slice(values.gradient[0], begin, rows),
               slice(values.gradient[1], begin, rows),
               slice(values.gradient[2], begin, rows)} {}

GridBasis::GridBasis(const integrals::LibcintInput& input, const Grid& grid, bool gradient)
    : grid_(grid), evaluator_(input), gradient_(gradient) {}

void GridBasis::trim(std::size_t minimum_scratch) {
    if (workspace::available() >= minimum_scratch)
        return;
    resident_ = integrals::BasisValues{};
    reservation_.try_resize(0);
}

void GridBasis::prepare(std::size_t output_bytes, std::size_t point_bytes) {
    trim(output_bytes + point_bytes);
    if (resident_rows() != 0)
        return;
    const auto available = workspace::available();
    const std::size_t ao_row = sizeof(double) * n_bf() * (gradient_ ? 4 : 1);
    if (available <= output_bytes || ao_row > available - output_bytes)
        return;
    const auto rows = std::min<std::size_t>(grid_.points.rows(),
                                            (available - output_bytes) / (ao_row + point_bytes));
    if (rows == 0 || !reservation_.try_resize(rows * ao_row))
        return;
    evaluator_.evaluate(grid_.points.topRows(rows), gradient_ ? 1 : 0, &resident_);
}

Eigen::Index GridBasis::rows(Eigen::Index begin, Eigen::Index requested) const {
    if (begin < resident_rows())
        return std::min(requested, resident_rows() - begin);
    return requested;
}

BasisTile
GridBasis::tile(Eigen::Index begin, Eigen::Index rows, integrals::BasisValues* scratch) const {
    if (begin < resident_rows())
        return BasisTile(resident_, begin, rows);
    evaluator_.evaluate(grid_.points.middleRows(begin, rows), gradient_ ? 1 : 0, scratch);
    return BasisTile(*scratch, 0, rows);
}

void ResponseCache::clear() {
    data_ = ReferenceTile{};
    for (auto& d : density_)
        d.resize(0, 0);
    rows_ = 0;
    spins_ = 0;
    reservation_.try_resize(0);
}

void ResponseCache::select(std::span<const ConstMatrix> density) {
    bool same = spins_ == static_cast<int>(density.size());
    for (int s = 0; same && s < spins_; ++s)
        same = density_[s].rows() == density[s].rows() && density_[s].cols() == density[s].cols() &&
               (density_[s].array() == density[s].array()).all();
    if (!same)
        clear();
}

void ResponseCache::prepare(std::span<const ConstMatrix> density,
                            Eigen::Index max_rows,
                            std::size_t output_bytes,
                            std::size_t point_bytes,
                            bool gga) {
    if (rows_ != 0)
        return;
    const int spins = density.size(), ns = spins == 1 ? 1 : 3;
    const std::size_t key = spins * density[0].size() * sizeof(double);
    const std::size_t packed = spins == 1 ? 1 : 3;
    const std::size_t row =
        sizeof(double) * (packed + (gga ? spins * ns + (spins == 1 ? 1 : 6) + 3 * spins + ns : 0));
    const auto available = workspace::available();
    if (available < output_bytes || key > available - output_bytes)
        return;
    rows_ = std::min<std::size_t>(max_rows, (available - output_bytes - key) / (point_bytes + row));
    if (rows_ == 0 || !reservation_.try_resize(key + rows_ * row)) {
        rows_ = 0;
        return;
    }
    spins_ = spins;
    for (int s = 0; s < spins; ++s)
        density_[s] = density[s];
}

Eigen::Index ResponseCache::rows(Eigen::Index begin, Eigen::Index requested) const {
    if (begin < rows_)
        return std::min(requested, rows_ - begin);
    return requested;
}

ReferenceTile* ResponseCache::tile(Eigen::Index begin, Eigen::Index rows) {
    return begin == 0 && rows == rows_ && rows_ > 0 ? &data_ : nullptr;
}
} // namespace xmvb::dft
