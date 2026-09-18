#pragma once

#include <memory>

#include <Eigen/Core>

#include "vbscf/optimization/coupled/operator.hpp"

namespace xmvb::vb {

class ExactHvpOperator;
struct AcceptedPointContext;

/**
 * @brief Complete matrix-free coupled Newton model at one accepted VBSCF point.
 *
 * `newton_operator` owns its exact accepted-point operator through its block
 * callbacks. `structure_kkt_residual` is the packed selected-subspace residual
 * in the same response coordinates used by the coupled operator.
 */
struct AcceptedPointCoupledModel {
  /** @brief Accepted exact operator retained for diagnostics and preconditioning. */
  std::shared_ptr<const ExactHvpOperator> exact_operator;
  CoupledNewtonOperator newton_operator;
  Eigen::VectorXd structure_kkt_residual;
};

/**
 * @brief Builds the production coupled orbital--structure Newton model.
 *
 * Adjacent selected states with the same normalized weight are collected into
 * maximal equal-weight clusters. The four unshifted Newton blocks are
 *
 * @f[
 * A,\qquad B,\qquad B^T,\qquad C,
 * @f]
 *
 * supplied directly by `ExactHvpOperator`; no dense structure matrix or
 * orbital Hessian is formed. An empty orbital metric callback selects the
 * Euclidean metric implemented by `CoupledNewtonOperator`.
 *
 * The accepted selected-state KKT residual is evaluated with one matrix-free
 * H/S block action and packed cluster by cluster as
 *
 * @f[
 * r_s=\sqrt{2w}
 * \begin{bmatrix}
 * HC-SC\Lambda\\
 * \tfrac12(C^TSC-I)
 * \end{bmatrix}.
 * @f]
 *
 * @param accepted_point Accepted eigensystem and normalized state weights.
 * @param exact_operator Exact operator representing the same accepted point.
 * @param n_orbital_coordinates Dimension of the accepted orbital chart.
 * @param orbital_metric Optional matrix-free orbital trust metric.
 */
AcceptedPointCoupledModel make_accepted_point_coupled_model(
    const AcceptedPointContext& accepted_point,
    std::shared_ptr<const ExactHvpOperator> exact_operator,
    int n_orbital_coordinates,
    CoupledBlockAction orbital_metric = {});

}  // namespace xmvb::vb
