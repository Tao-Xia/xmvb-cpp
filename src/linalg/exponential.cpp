#include "linalg/exponential.hpp"

#include <stdexcept>
#include <unsupported/Eigen/MatrixFunctions>

namespace xmvb::linalg {

Eigen::MatrixXd exponential(const Eigen::MatrixXd& generator) {
    if (generator.rows() != generator.cols())
        throw std::invalid_argument("matrix exponential requires a square generator");
    return generator.exp();
}

Eigen::MatrixXd exponential_pullback(const Eigen::MatrixXd& generator,
                                     const Eigen::MatrixXd& cotangent) {
    const Eigen::Index n = generator.rows();
    if (generator.cols() != n || cotangent.rows() != n || cotangent.cols() != n)
        throw std::invalid_argument("exponential cotangent dimensions do not match");
    if (generator.isZero(0.0))
        return cotangent;
    Eigen::MatrixXd block = Eigen::MatrixXd::Zero(2 * n, 2 * n);
    block.topLeftCorner(n, n) = generator.transpose();
    block.bottomRightCorner(n, n) = generator.transpose();
    block.topRightCorner(n, n) = cotangent;
    return exponential(block).topRightCorner(n, n).eval();
}

} // namespace xmvb::linalg
