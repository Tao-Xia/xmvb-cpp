#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "vbscf/optimization/krylov/orthonormal_hvp_basis.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}  // namespace

int main() {
  try {
    constexpr int dimension = 48;
    const double pi = std::acos(-1.0);
    Eigen::MatrixXd directions(dimension, dimension);
    for (int j = 0; j < dimension; ++j) {
      for (int i = 0; i < dimension; ++i) {
        directions(i, j) = std::cos(pi * (i + 0.5) * j / dimension) *
            std::sqrt((j == 0 ? 1.0 : 2.0) / dimension);
      }
    }
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(dimension, dimension);
    for (int j = 0; j < dimension; ++j) {
      h(j, j) = std::pow(10.0, -3.0 + 6.0 * j / (dimension - 1));
      if (j > 0) h(j, j - 1) = h(j - 1, j) = 0.13;
    }
    h(0, 0) = -2.0;  // Negative curvature must not be altered by the cache.

    std::vector<Eigen::VectorXd> basis, images;
    int calls = 0;
    auto action = [&](const Eigen::VectorXd& q) -> Eigen::VectorXd {
      require(std::abs(q.norm() - 1.0) < 1e-12,
              "HVP was not evaluated on a normalized basis direction");
      ++calls;
      return h * q;
    };
    for (int j = 0; j < dimension; ++j) {
      Eigen::VectorXd candidate = 1e-6 * directions.col(j);
      for (int k = 0; k < j; ++k) {
        candidate += std::sin(0.31 * (j + 1) * (k + 1)) * basis[k];
      }
      // Highly dependent directions with widely different raw amplitudes.
      candidate *= std::pow(10.0, j % 2 == 0 ? 80.0 : -80.0);
      Eigen::VectorXd image;
      require(xmvb::vb::append_orthonormal_hvp_direction(
                  candidate, action, &basis, &images, &image),
              "independent direction was rejected");
      const Eigen::VectorXd reference = h * candidate;
      require((image - reference).stableNorm() / reference.stableNorm() < 1e-12,
              "reconstructed search-direction HVP is inaccurate");
    }
    require(calls == dimension, "more than one HVP per admitted direction");
    Eigen::MatrixXd q(dimension, dimension), hq(dimension, dimension);
    for (int j = 0; j < dimension; ++j) {
      q.col(j) = basis[j];
      hq.col(j) = images[j];
    }
    require((q.transpose() * q - Eigen::MatrixXd::Identity(dimension, dimension))
                    .norm() < 1e-12,
            "basis orthogonality was lost");
    require((hq - h * q).norm() / (h * q).norm() < 1e-12,
            "cached HVP basis does not equal H times basis");
    const Eigen::MatrixXd t = q.transpose() * hq;
    require((t - t.transpose()).norm() / t.norm() < 1e-12,
            "projected Hessian lost symmetry");

    Eigen::VectorXd image;
    require(!xmvb::vb::append_orthonormal_hvp_direction(
                7.0 * basis.front(), action, &basis, &images, &image),
            "dependent direction was admitted");
    require(!xmvb::vb::append_orthonormal_hvp_direction(
                Eigen::VectorXd::Zero(dimension), action,
                &basis, &images, &image),
            "zero direction was admitted");
    require(calls == dimension && basis.size() == dimension &&
                images.size() == dimension,
            "rejected direction mutated the cache or consumed an HVP");

    std::vector<Eigen::VectorXd> block_basis, block_images;
    Eigen::MatrixXd block_candidates(dimension, 4);
    block_candidates.col(0) = directions.col(3);
    block_candidates.col(1) =
        directions.col(7) + 0.2 * directions.col(3);
    block_candidates.col(2) = 9.0 * block_candidates.col(0);  // dependent
    block_candidates.col(3) = directions.col(11);
    int block_calls = 0;
    int block_width = 0;
    const int admitted = xmvb::vb::append_orthonormal_hvp_block(
        block_candidates,
        [&](const Eigen::Ref<const Eigen::MatrixXd>& q_block) {
          ++block_calls;
          block_width = static_cast<int>(q_block.cols());
          return h * q_block;
        },
        &block_basis,
        &block_images);
    require(admitted == 3 && block_calls == 1 && block_width == 3,
            "independent seed directions were not fused into one block HVP");
    Eigen::MatrixXd block_q(dimension, admitted);
    Eigen::MatrixXd block_hq(dimension, admitted);
    for (int j = 0; j < admitted; ++j) {
      block_q.col(j) = block_basis[static_cast<std::size_t>(j)];
      block_hq.col(j) = block_images[static_cast<std::size_t>(j)];
    }
    require((block_q.transpose() * block_q -
             Eigen::MatrixXd::Identity(admitted, admitted)).norm() < 1e-12,
            "block HVP basis is not orthonormal");
    require((block_hq - h * block_q).norm() / (h * block_q).norm() < 1e-12,
            "block HVP images are inaccurate");
    std::cout << "orthonormal HVP basis: passed (48 ill-conditioned directions)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
