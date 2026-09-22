#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/optimization/neo/problem.hpp"

namespace xmvb::vb {

/** @brief Controls one PySCF-style co-iterative augmented-Hessian update. */
struct AugmentedHessianOptions {
  double convergence_tolerance = 1.0e-12;
  double linear_dependence_tolerance = 1.0e-14;
  double start_tolerance = 5.0e2;
  int start_cycle = 3;
  int maximum_subspace_dimension = 30;
};

/** @brief One incremental direction from a retained AH Davidson subspace. */
struct AugmentedHessianStep {
  Eigen::VectorXd step;
  Eigen::VectorXd hessian_step;
  Eigen::VectorXd residual;
  double eigenvalue = 0.0;
  double residual_norm = 0.0;
  double augmented_component = 0.0;
  double overlap_minimum = 0.0;
  int subspace_dimension = 0;
  int new_hessian_actions = 0;
  bool converged = false;
};

/**
 * @brief Retained matrix-free Davidson space for the augmented Hessian.
 *
 * For a caller-supplied keyframe gradient @f$g@f$, the projected generalized
 * eigenproblem represents
 * @f[
 * \begin{bmatrix}0&g^T\\g&H\end{bmatrix}
 * \begin{bmatrix}1\\x\end{bmatrix}
 * =\omega
 * \begin{bmatrix}1&0\\0&M\end{bmatrix}
 * \begin{bmatrix}1\\x\end{bmatrix}.
 * @f]
 * The physical basis vectors and their @f$H@f$/@f$M@f$ images persist when
 * the keyframe gradient changes, exactly as in co-iterative AH. No Hessian or
 * orbital--structure coupling matrix is formed.
 */
class AugmentedHessianWorkspace {
public:
  explicit AugmentedHessianWorkspace(const NeoProblem& problem);

  AugmentedHessianStep next(
      const Eigen::VectorXd& gradient,
      const AugmentedHessianOptions& options);

  int hessian_actions() const noexcept {
    return static_cast<int>(hessian_images_.size());
  }

private:
  bool append(
      Eigen::VectorXd direction,
      double spectral_shift,
      bool precondition);

  const NeoProblem* problem_ = nullptr;
  std::vector<Eigen::VectorXd> vectors_;
  std::vector<Eigen::VectorXd> hessian_images_;
  std::vector<Eigen::VectorXd> metric_images_;
  Eigen::VectorXd pending_residual_;
  double pending_eigenvalue_ = 0.0;
  bool has_pending_residual_ = false;
  double previous_eigenvalue_ = 0.0;
  bool has_previous_eigenvalue_ = false;
};

}  // namespace xmvb::vb
