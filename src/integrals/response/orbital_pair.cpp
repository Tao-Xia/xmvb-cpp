#include "integrals/response/orbital_pair.hpp"

#include <stdexcept>

#include "core/pair_index.hpp"
#include "core/timing.hpp"

namespace xmvb::integrals {

Eigen::MatrixXd pullback_orbital_pairs(const OrbitalPairMatrix& products,
                                       const OrbitalPairMatrix& pair_adjoint,
                                       std::span<const int> first,
                                       std::span<const int> second,
                                       int n_active) {
    profiling::ScopedPhase phase(profiling::Phase::OrbitalPairPullback);
    const Eigen::Index pairs = first.size();
    Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(n_active * pairs, n_active);
    for (Eigen::Index p = 0; p < pairs; ++p) {
        for (Eigen::Index q = 0; q < pairs; ++q) {
            weights(second[p] * pairs + q, first[p]) += pair_adjoint(q, p);
            weights(first[p] * pairs + q, second[p]) += pair_adjoint(q, p);
        }
    }
    const Eigen::Map<const OrbitalPairMatrix> image(
        products.data(), products.rows() / n_active, n_active * pairs);
    return image * weights;
}

OrbitalPairMatrix
contract_orbital_pair_direction(const OrbitalPairMatrix& products,
                                const Eigen::Ref<const Eigen::MatrixXd>& direction,
                                std::span<const int> first,
                                std::span<const int> second,
                                int n_active) {
    const Eigen::Index pairs = first.size();
    if (n_active <= 0 || pairs != static_cast<Eigen::Index>(n_active) * (n_active + 1) / 2 ||
        second.size() != first.size() || direction.cols() != n_active ||
        products.rows() != direction.rows() * n_active || products.cols() != pairs) {
        throw std::invalid_argument("invalid orbital-pair integral direction");
    }
    for (Eigen::Index p = 0; p < pairs; ++p) {
        if (first[p] < 0 || first[p] >= n_active || second[p] < 0 || second[p] >= n_active) {
            throw std::invalid_argument("invalid orbital-pair component index");
        }
    }
    const Eigen::Map<const OrbitalPairMatrix> image(
        products.data(), direction.rows(), n_active * pairs);
    const Eigen::MatrixXd projected = image.transpose() * direction;
    OrbitalPairMatrix contraction(pairs, pairs);
    for (Eigen::Index p = 0; p < pairs; ++p) {
        const int a = first[p], b = second[p];
        for (Eigen::Index q = 0; q < pairs; ++q) {
            contraction(p, q) = projected(b * pairs + q, a) + projected(a * pairs + q, b);
        }
    }
    return contraction;
}

Eigen::VectorXd packed_orbital_pair_direction(const OrbitalPairMatrix& products,
                                              const Eigen::Ref<const Eigen::MatrixXd>& direction,
                                              std::span<const int> first,
                                              std::span<const int> second,
                                              int n_active) {
    const OrbitalPairMatrix unsym =
        contract_orbital_pair_direction(products, direction, first, second, n_active);
    const Eigen::Index pairs = unsym.rows();
    Eigen::VectorXd packed(pairs * (pairs + 1) / 2);
    for (Eigen::Index p = 0; p < pairs; ++p) {
        for (Eigen::Index q = 0; q <= p; ++q) {
            packed[xmvb::core::PairIndex::packed_pair_of_pairs_index(p, q)] =
                unsym(p, q) + unsym(q, p);
        }
    }
    return packed;
}

} // namespace xmvb::integrals
