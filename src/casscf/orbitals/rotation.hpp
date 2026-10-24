#pragma once

#include <utility>
#include <vector>
#include <Eigen/Core>

namespace xmvb::casscf {

/** Nonredundant real rotations between core, active, and virtual subspaces. */
class RotationChart {
  public:
    RotationChart(int nmo, int ncore, int nactive);
    int size() const noexcept {
        return static_cast<int>(pairs_.size());
    }
    int nmo() const noexcept {
        return nmo_;
    }
    const std::vector<std::pair<int, int>>& pairs() const noexcept {
        return pairs_;
    }

    /** For pair (i<j), K(j,i)=p and K(i,j)=-p. */
    Eigen::MatrixXd generator(const Eigen::VectorXd& parameters) const;
    /** U exp(K(p)); finite, orthogonal update in the supplied frame. */
    Eigen::MatrixXd retract(const Eigen::MatrixXd& orbitals,
                            const Eigen::VectorXd& parameters) const;
    /** Exact D exp(K)^* pullback into the original keyframe coordinates. */
    Eigen::VectorXd pullback(const Eigen::MatrixXd& orbitals,
                             const Eigen::MatrixXd& raw_gradient,
                             const Eigen::VectorXd& parameters) const;
    /** Origin chart Hessian action, including the symmetric exponential jet. */
    Eigen::VectorXd origin_hessian_vector(const Eigen::MatrixXd& orbitals,
                                          const Eigen::MatrixXd& raw_gradient,
                                          const Eigen::MatrixXd& raw_gradient_direction,
                                          const Eigen::VectorXd& direction) const;

  private:
    int nmo_;
    std::vector<std::pair<int, int>> pairs_;
    void check_matrix(const Eigen::MatrixXd& matrix) const;
};

} // namespace xmvb::casscf
