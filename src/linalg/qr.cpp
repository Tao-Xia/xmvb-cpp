#include "linalg/qr.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <lapacke.h>

#include "core/index.hpp"

namespace xmvb::linalg {

void QrWorkspace::compute(Eigen::Ref<Eigen::MatrixXd> matrix) {
    const auto m = core::checked_index<lapack_int>(matrix.rows());
    const auto n = core::checked_index<lapack_int>(matrix.cols());
    const auto lda = core::checked_index<lapack_int>(matrix.outerStride());
    if (!m || !n)
        return;
    tau_.resize(std::min(m, n));
    if (columns_ != n) {
        double query = 0.0;
        const auto info = LAPACKE_dgeqrf_work(
            LAPACK_COL_MAJOR, m, n, matrix.data(), lda, tau_.data(), &query, -1);
        if (info != 0 || !std::isfinite(query) || query < n ||
            query > std::numeric_limits<lapack_int>::max())
            throw std::runtime_error("LAPACK QR workspace query failed");
        if (maximum_workspace_ < static_cast<std::size_t>(n))
            throw std::length_error("LAPACK QR workspace cannot hold one column");
        work_.resize(std::min(maximum_workspace_, static_cast<std::size_t>(query)));
        columns_ = n;
    }
    residency_.resize(core::checked_multiply(core::checked_add(tau_.capacity(), work_.capacity()),
                                             sizeof(double)));
    const auto info = LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR,
                                          m,
                                          n,
                                          matrix.data(),
                                          lda,
                                          tau_.data(),
                                          work_.data(),
                                          core::checked_index<lapack_int>(work_.size()));
    if (info != 0)
        throw std::runtime_error("LAPACK QR failed: " + std::to_string(info));
}

} // namespace xmvb::linalg
