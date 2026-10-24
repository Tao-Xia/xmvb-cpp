#include "casscf/orbitals/frame.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/Cholesky>
#include <Eigen/QR>

namespace xmvb::casscf {
namespace {

Eigen::MatrixXd orthonormal_block(const Eigen::MatrixXd& block, double reference_scale) {
    const int n = block.rows(), width = block.cols();
    if (width == 0)
        return Eigen::MatrixXd(n, 0);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(block);
    const double threshold =
        std::numeric_limits<double>::epsilon() * std::max(n, width) * reference_scale;
    for (int i = 0; i < width; ++i)
        if (!(std::abs(qr.matrixQR()(i, i)) > threshold))
            throw std::invalid_argument("core/active orbital seed is rank deficient");
    return qr.householderQ() * Eigen::MatrixXd::Identity(n, width);
}

} // namespace

Eigen::MatrixXd complete_metric_orthonormal_frame(const Eigen::MatrixXd& seed,
                                                  const Eigen::MatrixXd& overlap,
                                                  int ncore,
                                                  int nactive) {
    const int n = overlap.rows();
    const int occupied = ncore + nactive;
    if (n < 1 || ncore < 0 || nactive < 1 || occupied > n || overlap.cols() != n ||
        seed.rows() != n || seed.cols() != occupied || !overlap.allFinite() || !seed.allFinite())
        throw std::invalid_argument("invalid AO overlap or complete-CAS orbital seed");
    const double symmetry_scale = std::max(1.0, overlap.norm());
    if ((overlap - overlap.transpose()).norm() >
        std::numeric_limits<double>::epsilon() * n * symmetry_scale)
        throw std::invalid_argument("AO overlap is not symmetric");
    Eigen::LLT<Eigen::MatrixXd> llt(overlap);
    if (llt.info() != Eigen::Success)
        throw std::invalid_argument("AO overlap is not positive definite");
    const Eigen::MatrixXd lower = llt.matrixL();
    const Eigen::MatrixXd whitened = lower.transpose() * seed;
    const Eigen::MatrixXd core =
        orthonormal_block(whitened.leftCols(ncore), whitened.leftCols(ncore).norm());
    Eigen::MatrixXd active = whitened.rightCols(nactive);
    if (ncore) {
        Eigen::MatrixXd projection = core.transpose() * active;
        active.noalias() -= core * projection;
        projection = core.transpose() * active;
        active.noalias() -= core * projection;
    }
    const Eigen::MatrixXd active_q = orthonormal_block(active, whitened.rightCols(nactive).norm());
    Eigen::MatrixXd occupied_q(n, occupied);
    if (ncore)
        occupied_q.leftCols(ncore) = core;
    occupied_q.rightCols(nactive) = active_q;
    Eigen::MatrixXd white_frame(n, n);
    white_frame.leftCols(occupied) = occupied_q;
    if (occupied < n) {
        Eigen::HouseholderQR<Eigen::MatrixXd> complement_qr(occupied_q);
        const Eigen::MatrixXd whole_q =
            complement_qr.householderQ() * Eigen::MatrixXd::Identity(n, n);
        white_frame.rightCols(n - occupied) = whole_q.rightCols(n - occupied);
    }
    const Eigen::MatrixXd upper = lower.transpose();
    const Eigen::MatrixXd frame = upper.triangularView<Eigen::Upper>().solve(white_frame);
    const Eigen::MatrixXd metric = frame.transpose() * overlap * frame;
    if ((metric - Eigen::MatrixXd::Identity(n, n)).norm() >
        32 * std::numeric_limits<double>::epsilon() * n)
        throw std::runtime_error("completed MO frame lost AO-metric orthogonality");
    return frame;
}

} // namespace xmvb::casscf
