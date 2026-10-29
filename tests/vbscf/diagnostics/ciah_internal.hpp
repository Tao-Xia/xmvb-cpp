#pragma once

#include <span>
#include <vector>
#include <Eigen/Core>

#include "mcscf/ci/sigma.hpp"

namespace xmvb::test {

/** @brief One internal orbital/CI jet in the fixed accepted orthogonal frame. */
struct InternalStateDirection {
    Eigen::MatrixXd generator;
    Eigen::MatrixXd orbital;
    Eigen::MatrixXd ci;
    Eigen::MatrixXd normalized_direction;
    Eigen::VectorXd radial;
};

struct InternalStateCurvature {
    Eigen::MatrixXd hessian;
    Eigen::VectorXd gradient;
    Eigen::MatrixXd tangent_curvature;
    Eigen::MatrixXd residual_curvature;
};

/** @brief Exact fixed-span Rayleigh curvature using only accepted active integrals.
 * If D_a is the one-body generator action, the affine orbital second jet is
 * (D_a D_b + D_b D_a - D_(K_a K_b+K_b K_a))/2 + D_Lab.
 * Lab retains raw-orbital normalization, including unequal ionic occupations.
 * All second-generator products are contracted through one-body moments;
 * no third/fourth density tensor or AO integral response is formed.
 * Tiny diagnostic only: the carrier and sampled Hessian are deliberately bounded
 * by audit_joint_geometry, not allocated by a molecular optimization.
 */
inline InternalStateCurvature
internal_state_curvature(const mcscf::DirectCiSigmaAction& sigma,
                         const Eigen::MatrixXd& state,
                         const Eigen::MatrixXd& sigma_state,
                         const Eigen::MatrixXd& r,
                         const Eigen::MatrixXd& inverse,
                         const Eigen::MatrixXd& ao_overlap,
                         std::span<const InternalStateDirection> directions) {
    const Eigen::Index count = directions.size(), n = r.rows();
    const double energy = (state.array() * sigma_state.array()).sum();
    const Eigen::MatrixXd residual = sigma_state - energy * state;
    const Eigen::MatrixXd moment = sigma.one_body_generator_adjoint(residual, state);
    std::vector<Eigen::MatrixXd> orbital_moments, ci_moments, overlap;
    Eigen::MatrixXd first(state.size(), count), images(state.size(), count);
    Eigen::VectorXd parallel(count);
    for (Eigen::Index a = 0; a < count; ++a) {
        const auto& direction = directions[a];
        const Eigen::MatrixXd u = direction.orbital + direction.ci;
        const Eigen::MatrixXd hu = sigma.apply(u);
        first.col(a) = Eigen::Map<const Eigen::VectorXd>(u.data(), u.size());
        images.col(a) = Eigen::Map<const Eigen::VectorXd>(hu.data(), hu.size());
        parallel[a] = (state.array() * u.array()).sum();
        orbital_moments.push_back(direction.orbital.isZero(0) ? Eigen::MatrixXd::Zero(n, n).eval()
                                                              : sigma.one_body_generator_adjoint(
                                                                    residual, direction.orbital));
        ci_moments.push_back(direction.ci.isZero(0)
                                 ? Eigen::MatrixXd::Zero(n, n).eval()
                                 : sigma.one_body_generator_adjoint(residual, direction.ci));
        overlap.push_back(ao_overlap * direction.normalized_direction);
    }
    InternalStateCurvature result;
    result.gradient =
        2 * first.transpose() * Eigen::Map<const Eigen::VectorXd>(residual.data(), residual.size());
    result.tangent_curvature =
        2 * first.transpose() * (images - energy * first) -
        (result.gradient * parallel.transpose() + parallel * result.gradient.transpose());
    result.hessian =
        2 * first.transpose() * (images - energy * first) -
        2 * (result.gradient * parallel.transpose() + parallel * result.gradient.transpose());
    const auto dot = [](const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
        return (a.array() * b.array()).sum();
    };
    for (Eigen::Index a = 0; a < count; ++a)
        for (Eigen::Index b = 0; b <= a; ++b) {
            const auto& da = directions[a];
            const auto& db = directions[b];
            Eigen::VectorXd covariance(n);
            for (Eigen::Index i = 0; i < n; ++i)
                covariance[i] = da.normalized_direction.col(i).dot(overlap[b].col(i));
            const Eigen::MatrixXd second_generator =
                -(da.generator * r * db.radial.asDiagonal() +
                  db.generator * r * da.radial.asDiagonal() + r * covariance.asDiagonal()) *
                inverse;
            const double second =
                .5 *
                    (dot(da.generator, orbital_moments[b]) + dot(db.generator, orbital_moments[a]) -
                     dot((da.generator * db.generator + db.generator * da.generator).eval(),
                         moment)) +
                dot(second_generator, moment) + dot(da.generator, ci_moments[b]) +
                dot(db.generator, ci_moments[a]);
            result.hessian(a, b) += 2 * second;
            if (a != b)
                result.hessian(b, a) += 2 * second;
        }
    result.residual_curvature = result.hessian - result.tangent_curvature;
    return result;
}
} // namespace xmvb::test
